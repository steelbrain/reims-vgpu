//! Persistent ash instance/device + device-loss recreate policy.

#![allow(unsafe_op_in_unsafe_fn)]

use ash::vk;
use std::ffi::{CStr, CString};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use super::counters::EngineCounters;
use super::device_lost::DeviceLostDecline;
use super::init_decline::InitDecline;
use super::types::DrawError;
use super::vk_call::{VkCall, VkOp};
use crate::backend::vulkan::caps::api_floor;
use crate::backend::vulkan::caps::device_select::select_physical_device;
use crate::backend::vulkan::caps::memory_topology::{
    classify_memory, select_memory_type, MappedMemoryKind, MemoryClass, MemoryRequest,
};
use crate::backend::vulkan::caps::{DriverQuirk, HostGpuCaps};

/// Max device recreates per process after DEVICE_LOST (named constant).
pub const MAX_DEVICE_RECREATES: u32 = 3;

/// Bounded fence wait (nanoseconds). Named constant — not env-gated.
pub const FENCE_TIMEOUT_NS: u64 = 5_000_000_000; // 5s

/// `sizeof(VkPipelineCacheHeaderVersionOne)` (Vulkan spec §Pipeline Cache
/// Header): u32 headerSize, u32 headerVersion, u32 vendorID, u32 deviceID,
/// u8[16] pipelineCacheUUID — all integers little-endian.
const PIPELINE_CACHE_HEADER_ONE_LEN: usize = 32;

/// On-disk pipeline-cache blob location for a device, keyed by its
/// pipelineCacheUUID (hex) so blobs from other GPUs/driver versions land in
/// distinct files and never collide.
fn pipeline_cache_disk_path(uuid: &[u8; 16]) -> std::path::PathBuf {
    use std::fmt::Write as _;
    let mut hex = String::with_capacity(32);
    for b in uuid {
        let _ = write!(hex, "{b:02x}");
    }
    std::env::temp_dir().join(format!("reims-vgpu-vk-pipeline-cache-{hex}.bin"))
}

/// Outcome of one atomic pipeline-cache blob save.
#[derive(Debug, PartialEq, Eq)]
enum CacheSaveOutcome {
    /// This save's blob is now the on-disk cache.
    Landed,
    /// A strictly larger snapshot already landed; this one was dropped (the
    /// tmp file is cleaned up) rather than regress the on-disk cache.
    Superseded,
}

/// A pipeline-cache load or persistence failure. The exact filesystem/Vulkan
/// stage survives the cold-start fallback or detached save thread instead of
/// disappearing behind "cache miss".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PipelineCacheDecline {
    Read {
        errno: Option<i32>,
        kind: std::io::ErrorKind,
    },
    Incompatible {
        bytes: usize,
    },
    WarmCreate {
        result: vk::Result,
    },
    Write {
        errno: Option<i32>,
        kind: std::io::ErrorKind,
    },
    Rename {
        errno: Option<i32>,
        kind: std::io::ErrorKind,
    },
}

impl PipelineCacheDecline {
    fn read(error: &std::io::Error) -> Self {
        Self::Read {
            errno: error.raw_os_error(),
            kind: error.kind(),
        }
    }

    fn write(error: &std::io::Error) -> Self {
        Self::Write {
            errno: error.raw_os_error(),
            kind: error.kind(),
        }
    }

    fn rename(error: &std::io::Error) -> Self {
        Self::Rename {
            errno: error.raw_os_error(),
            kind: error.kind(),
        }
    }
}

impl crate::observe::Decline for PipelineCacheDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::Read { .. } => "vk_pipeline_cache_read",
            Self::Incompatible { .. } => "vk_pipeline_cache_incompatible",
            Self::WarmCreate { .. } => "vk_pipeline_cache_warm_create",
            Self::Write { .. } => "vk_pipeline_cache_write",
            Self::Rename { .. } => "vk_pipeline_cache_rename",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::Read { errno, kind }
            | Self::Write { errno, kind }
            | Self::Rename { errno, kind } => vec![
                (
                    "errno",
                    errno.map_or_else(|| "none".to_string(), |value| value.to_string()),
                ),
                ("io_kind", format!("{kind:?}")),
            ],
            Self::Incompatible { bytes } => vec![("bytes", bytes.to_string())],
            Self::WarmCreate { result } => vec![(
                "vk_result",
                result.to_string().replace(char::is_whitespace, "_"),
            )],
        }
    }
}

/// Write `data` to `tmp`, then atomically `rename(tmp → path)` with a
/// best-effort newest-wins guard on `persisted_len` (the largest blob length
/// landed so far). `tmp` MUST be unique per concurrent save (the caller keys it
/// on a per-save sequence) so two in-flight saves never share a tmp file — the
/// bug that made one save's rename move the tmp out from under another's,
/// failing ENOENT. Returns which stage failed (`write`/`rename`) on error so the
/// caller can name the reason. Pure w.r.t. Vulkan (fs + atomic only) → unit-testable.
fn write_cache_atomic(
    path: &std::path::Path,
    tmp: &std::path::Path,
    data: &[u8],
    persisted_len: &AtomicUsize,
) -> Result<CacheSaveOutcome, PipelineCacheDecline> {
    std::fs::write(tmp, data).map_err(|error| PipelineCacheDecline::write(&error))?;
    // Claim newest-wins before the rename: if a larger snapshot already landed,
    // drop this one rather than regress the on-disk cache to a stale subset.
    if persisted_len.fetch_max(data.len(), Ordering::Relaxed) > data.len() {
        let _ = std::fs::remove_file(tmp);
        return Ok(CacheSaveOutcome::Superseded);
    }
    match std::fs::rename(tmp, path) {
        Ok(()) => Ok(CacheSaveOutcome::Landed),
        Err(e) => {
            let _ = std::fs::remove_file(tmp);
            Err(PipelineCacheDecline::rename(&e))
        }
    }
}

/// Validate a candidate initial-data blob against the live device.
/// `vkCreatePipelineCache` valid usage requires initial data to come from a
/// prior `vkGetPipelineCacheData` on a compatible device — feeding it a
/// stale/corrupt file is UB, so the VkPipelineCacheHeaderVersionOne fields
/// are checked here, not left to the driver.
fn pipeline_cache_blob_compatible(blob: &[u8], props: &vk::PhysicalDeviceProperties) -> bool {
    if blob.len() < PIPELINE_CACHE_HEADER_ONE_LEN {
        return false;
    }
    let u32le = |off: usize| u32::from_le_bytes(blob[off..off + 4].try_into().unwrap());
    (u32le(0) as usize) >= PIPELINE_CACHE_HEADER_ONE_LEN
        && u32le(4) == vk::PipelineCacheHeaderVersion::ONE.as_raw() as u32
        && u32le(8) == props.vendor_id
        && u32le(12) == props.device_id
        && blob[16..PIPELINE_CACHE_HEADER_ONE_LEN] == props.pipeline_cache_uuid
}

/// Read and validate the warm-start cache. A missing file is the expected
/// first-boot state; every other read failure or rejected blob is a real cold
/// fallback and therefore reaches the fail-visible boundary.
fn read_pipeline_cache_blob(
    path: &std::path::Path,
    props: &vk::PhysicalDeviceProperties,
) -> Result<Option<Vec<u8>>, PipelineCacheDecline> {
    match std::fs::read(path) {
        Ok(blob) if pipeline_cache_blob_compatible(&blob, props) => Ok(Some(blob)),
        Ok(blob) => Err(PipelineCacheDecline::Incompatible { bytes: blob.len() }),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(PipelineCacheDecline::read(&error)),
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct VertexDivisorCapabilities {
    pub instance_rate_divisor: bool,
    pub zero_divisor: bool,
    pub max_divisor: u32,
}

/// GPU-side timing for the composite readback: how long the GPU spent executing
/// the copy command buffer, as distinct from how long the CPU spent waiting for
/// it.
///
/// `readback_split fence_us` measures wall clock between `vkQueueSubmit` and
/// `vkWaitForFences` returning. That interval contains three different things
/// with three different fixes — the draw batch still executing, the copy
/// executing, and the cost of asking (queue scheduling, the GPU leaving a
/// low-power state, and the CPU's own wake from the fence's signal) — and no
/// number in this device separated them. It matters which, because
/// `ResidentArmCensus` established that holding the host GPU at its top clock
/// moves that same wall clock from 2.55-2.83 ms to 0.40 ms, six sevenths of it
/// with no code change. Six sevenths being the governor is consistent with two
/// opposite readings: the GPU doing the same work slower, or the GPU spending
/// most of the interval not working at all.
///
/// A timestamp written at the top of the copy command buffer and another at its
/// bottom answers it directly and without correlating two clocks: the delta is
/// GPU ticks between two points in the GPU's own timeline, and
/// `timestampPeriod` scales it to nanoseconds. If the copy's execution is a
/// small fraction of `fence_us`, the rest is the batch and the asking, and
/// making the copy cheaper cannot pay.
///
/// Two queries are enough because the readback is serialized: the caller waits
/// on this copy's fence before it can start another, so the pool is never read
/// while a second submission is writing it.
pub(crate) struct TimestampProbe {
    /// Slot 0 is written at `TOP_OF_PIPE` before the barrier, slot 1 at
    /// `TRANSFER` immediately after it, slot 2 at `BOTTOM_OF_PIPE` after the
    /// copy. [`Self::SLOTS`] is the count the pool is created with and the count
    /// the command buffer resets; a mismatch reads as a silent zero rather than
    /// as an error, which is how the two-slot first version shipped and measured
    /// nothing.
    pub pool: vk::QueryPool,
    /// `VkPhysicalDeviceLimits::timestampPeriod` — nanoseconds per tick.
    pub ns_per_tick: f32,
}

impl TimestampProbe {
    /// How many queries the pool holds, the command buffer resets, and the read
    /// asks for. One constant because the three must agree: created with two
    /// while the command buffer wrote three, the reads failed and the census
    /// printed zeros that read exactly like "the GPU did no work".
    pub const SLOTS: u32 = 3;
}

pub(crate) struct DeviceContext {
    pub _entry: ash::Entry,
    pub instance: ash::Instance,
    pub pd: vk::PhysicalDevice,
    pub device: ash::Device,
    /// Where this device lands in the four-cell support matrix, plus the
    /// capability answers every policy decision derives from. Behavior gates
    /// read this, never a driver name or extension string.
    pub caps: HostGpuCaps,
    /// `VkPhysicalDeviceMemoryProperties`, queried once. It is immutable for a
    /// physical device, and the previous code re-queried it through the loader
    /// on every single allocation.
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    /// Queue family used for all engine submits (graphics draws + compute).
    pub gq: u32,
    /// True when `gq` supports both GRAPHICS and COMPUTE (required for engine compute).
    pub compute_capable: bool,
    /// Capability-gated format-less storage-image writes
    /// (`shaderStorageImageWriteWithoutFormat`). Lets a storage image be viewed
    /// as any compatible format (e.g. `B8G8R8A8_UNORM`) while the SPIR-V
    /// `OpTypeImage` declares `Unknown` — the GPU converts the written vec4 to
    /// the view's channel order. Required to composite a guest `BGRA8Unorm`
    /// storage surface without an R/B swap (SPIR-V has no `Bgra8` storage
    /// format). Universally present on desktop NVIDIA / Mesa ANV / RADV.
    pub storage_image_write_without_format: bool,
    /// `R32_SFLOAT` usable as a linearly-filtered sampled image; gates the
    /// native float32 color-LUT sampled rail (see [`DeviceFeatures`]).
    pub sampled_r32f_linear_filter: bool,
    pub pipeline_cache: vk::PipelineCache,
    pub vertex_divisor: VertexDivisorCapabilities,
    /// Which vertex attribute formats this device accepts in a vertex buffer,
    /// probed once. Vulkan makes the three-component 8/16-bit formats optional,
    /// so a pipeline resolves each attribute through this rather than assuming
    /// the format it decoded is bindable.
    pub vertex_formats: crate::backend::vulkan::translate::VertexFormatSupport,
    pub max_sampler_anisotropy: f32,
    pub sampler_anisotropy: bool,
    /// Every device feature and format capability, as resolved by
    /// [`crate::backend::vulkan::caps::device_features`]. Behaviour gates read
    /// this rather than re-querying: a feature asked about in two places is a
    /// feature that will eventually be enabled in one of them.
    pub features: crate::backend::vulkan::caps::device_features::DeviceFeatures,
    /// Combined depth-stencil format supported for DEPTH_STENCIL_ATTACHMENT on
    /// this device (D32_SFLOAT_S8_UINT preferred, D24_UNORM_S8_UINT fallback).
    /// Used only by the stencil-test path; depth-only uses D32_SFLOAT.
    pub depth_stencil_format: vk::Format,
    /// A two-slot timestamp query pool for the composite readback, and the tick
    /// length that turns its delta into wall clock. `None` when this queue
    /// family reports `timestampValidBits == 0` or the device reports a
    /// `timestampPeriod` of zero, both of which Vulkan permits.
    ///
    /// This exists to answer one question the rest of the device cannot:
    /// `readback_split fence_us` is wall clock spent blocked, and a blocked
    /// caller cannot tell GPU work from the latency of asking. See
    /// [`TimestampProbe`].
    pub timestamps: Option<TimestampProbe>,
    /// On-disk VkPipelineCache blob for this device (keyed by
    /// pipelineCacheUUID), or None when persistence is unavailable.
    pub pipeline_cache_path: Option<std::path::PathBuf>,
    /// Byte length of the last persisted cache blob — the growth debounce
    /// for [`Self::persist_pipeline_cache`].
    pub pipeline_cache_saved_len: AtomicUsize,
    /// `VK_KHR_swapchain` was enabled for the engine-owned host window.
    #[cfg(feature = "host-window")]
    pub swapchain: bool,
}

// SAFETY: ash handles; only accessed under engine mutex.
unsafe impl Send for DeviceContext {}

impl DeviceContext {
    pub(crate) unsafe fn create() -> Result<Self, DrawError> {
        let entry = ash::Entry::load().map_err(|e| {
            DrawError::Init(InitDecline::LoadVulkanLoader {
                detail: e.to_string(),
            })
        })?;
        // Ask for what the loader can actually give us, capped at the highest
        // version the engine knows how to use. Hardcoding 1.3 is
        // VK_ERROR_INCOMPATIBLE_DRIVER on a Vulkan 1.0 loader, and on every
        // other loader it is a claim we do not back: nothing here needs a 1.3
        // core feature.
        let loader_version = match entry.try_enumerate_instance_version() {
            Ok(Some(version)) => version,
            Ok(None) => vk::API_VERSION_1_0,
            Err(result) => {
                let decline = InitDecline::EnumerateInstanceVersion { result };
                crate::observe::Emit::decline("vk_loader_version", &decline).fail_once(0);
                vk::API_VERSION_1_0
            }
        };
        let app = vk::ApplicationInfo::default()
            .api_version(api_floor::instance_api_version(loader_version));
        let portability_enumeration = entry
            .enumerate_instance_extension_properties(None)
            .map_err(|result| DrawError::Init(InitDecline::EnumerateInstanceExtensions { result }))?
            .iter()
            .any(|extension| {
                CStr::from_ptr(extension.extension_name.as_ptr())
                    == vk::KHR_PORTABILITY_ENUMERATION_NAME
            });
        let mut instance_extensions = Vec::new();
        if portability_enumeration {
            instance_extensions.push(vk::KHR_PORTABILITY_ENUMERATION_NAME.as_ptr());
        }
        // Surface extensions for the engine-owned host window.
        //
        // The window does not exist yet — the engine context is created on the
        // first draw, long before winit has a handle — so which *platform*
        // surface extension will be needed is not knowable here. Enabling every
        // one the loader advertises is what makes the later
        // `ash_window::create_surface` work for whichever handle arrives, and it
        // costs nothing: an enabled instance extension with no surface created
        // through it is inert.
        //
        // Enabling nothing unless `VK_KHR_surface` *and* at least one platform
        // extension are both present keeps the failure at attach time, where it
        // can be a typed decline and fall back to the CPU staging path, rather
        // than failing instance creation for every headless run.
        #[cfg(feature = "host-window")]
        {
            let advertised = entry
                .enumerate_instance_extension_properties(None)
                .map_err(|result| {
                    DrawError::Init(InitDecline::EnumerateInstanceExtensions { result })
                })?;
            let has_instance_extension = |name: &CStr| {
                advertised
                    .iter()
                    .any(|extension| CStr::from_ptr(extension.extension_name.as_ptr()) == name)
            };
            #[cfg(target_os = "macos")]
            let platform: &[&CStr] = &[ash::ext::metal_surface::NAME];
            // X11 and Wayland are both live on Linux desktops and the session
            // type is a runtime property, so both are offered and each is taken
            // only if advertised.
            #[cfg(target_os = "linux")]
            let platform: &[&CStr] = &[
                ash::khr::xlib_surface::NAME,
                ash::khr::xcb_surface::NAME,
                ash::khr::wayland_surface::NAME,
            ];
            // Win32 is the only WSI platform on Windows — there is no session
            // type to discover at runtime the way there is on Linux.
            #[cfg(target_os = "windows")]
            let platform: &[&CStr] = &[ash::khr::win32_surface::NAME];
            // No host this crate builds for reaches here: the arms are
            // partitioned in `crate::lib` and each names its surface above. The
            // empty slice keeps this expression total rather than asserting a
            // fourth host cannot exist — and because the guard below requires a
            // non-empty list, such a host declines at attach time instead of
            // creating a WSI-less instance that a later `create_surface` would
            // use as though it had one.
            #[cfg(not(any(
                target_os = "macos",
                target_os = "linux",
                target_os = "windows"
            )))]
            let platform: &[&CStr] = &[];
            let available: Vec<&CStr> = platform
                .iter()
                .copied()
                .filter(|name| has_instance_extension(name))
                .collect();
            if has_instance_extension(ash::khr::surface::NAME) && !available.is_empty() {
                instance_extensions.push(ash::khr::surface::NAME.as_ptr());
                for name in available {
                    instance_extensions.push(name.as_ptr());
                }
            }
        }
        let mut ici = vk::InstanceCreateInfo::default()
            .application_info(&app)
            .enabled_extension_names(&instance_extensions);
        if portability_enumeration {
            ici = ici.flags(vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR);
        }
        let instance = entry
            .create_instance(&ici, None)
            .map_err(|result| DrawError::Init(InitDecline::CreateInstance { result }))?;
        let pds = instance
            .enumerate_physical_devices()
            .map_err(|result| DrawError::Init(InitDecline::EnumeratePhysicalDevices { result }))?;
        // Pick the best device that clears the API floor, by rank (discrete >
        // integrated > virtual > other > CPU), keeping the FIRST-enumerated
        // device on a tie. A bare `pds.first()` fallback could pick a software
        // rasterizer (llvmpipe) that enumerated ahead of a real GPU.
        let candidates: Vec<_> = pds
            .iter()
            .copied()
            .map(|p| {
                let props = instance.get_physical_device_properties(p);
                (props.api_version, props.device_type, p)
            })
            .collect();
        let (pd, _chosen_api_version) = select_physical_device(&candidates).map_err(|found| {
            let decline = if found.is_empty() {
                InitDecline::NoPhysicalDevice
            } else {
                InitDecline::BelowApiFloor {
                    minimum: api_floor::MIN_SUPPORTED_API,
                    found,
                }
            };
            crate::observe::Emit::decline("vk_device_select_fail", &decline).fail();
            DrawError::Init(decline)
        })?;
        let qfs = instance.get_physical_device_queue_family_properties(pd);
        // Prefer a combined GRAPHICS|COMPUTE family so draws and dispatches share
        // one queue / submission order. Fall back to graphics-only (compute requests
        // then fail named Unsupported).
        let graphics_compute = qfs.iter().position(|q| {
            q.queue_flags
                .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
        });
        let graphics_only = qfs
            .iter()
            .position(|q| q.queue_flags.contains(vk::QueueFlags::GRAPHICS));
        let (gq, compute_capable) = match (graphics_compute, graphics_only) {
            (Some(i), _) => (i as u32, true),
            (None, Some(i)) => (i as u32, false),
            (None, None) => {
                return Err(DrawError::Init(InitDecline::NoGraphicsQueueFamily));
            }
        };
        let prio = [1.0f32];
        let qci = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(gq)
            .queue_priorities(&prio)];
        let device_extensions = instance
            .enumerate_device_extension_properties(pd)
            .map_err(|result| DrawError::Init(InitDecline::EnumerateDeviceExtensions { result }))?;
        let has_device_extension = |name: &CStr| {
            device_extensions
                .iter()
                .any(|extension| CStr::from_ptr(extension.extension_name.as_ptr()) == name)
        };
        // Every device feature and format capability, resolved in one place.
        // Enumerating extensions first is what lets `mirror_clamp_to_edge`
        // choose between the 1.2 core feature and the KHR extension there
        // rather than here; see `caps::device_features` for why that decision
        // must not be spread across the two.
        let features = crate::backend::vulkan::caps::device_features::query(
            &instance,
            pd,
            &has_device_extension,
        );
        let storage_image_write_without_format_bgra =
            features.storage_image_write_without_format_bgra();
        let sampled_r32f_linear_filter = features.sampled_r32f_linear_filter;
        // Either half of the 16-bit storage struct being wanted is reason to
        // chain it: they are separate bits in one structure.
        let has16 = features.storage16 || features.storage_input_output16;
        let has8 = features.storage8;
        let has_float16 = features.float16;
        let has_int8 = features.int8;
        // Defined bounds-clamped behavior for out-of-range shader buffer access
        // is among these — the ONE feature the Vulkan spec requires every
        // implementation to support, so enabling it is portability-clean and
        // removes a whole UB class (NVIDIA tolerates OOB silently; Apple GPUs
        // page-fault and MoltenVK loses the device). NOTE: the live arm64
        // device-loss draw (kIOGPUCommandBufferCallbackErrorPageFault) still
        // faults WITH it enabled, so that fault is not a robustness-coverable
        // shader buffer access (index fetch, attachment access, and
        // encoder-level suspects remain open).
        let enabled = features.enabled_features();
        let portability_subset = has_device_extension(vk::KHR_PORTABILITY_SUBSET_NAME);
        let vertex_attribute_divisor = has_device_extension(vk::KHR_VERTEX_ATTRIBUTE_DIVISOR_NAME);
        #[cfg(feature = "host-window")]
        let swapchain = has_device_extension(ash::khr::swapchain::NAME);
        // Combined depth-stencil format for the stencil-test path. The Vulkan
        // spec guarantees at least ONE of D32_SFLOAT_S8_UINT / D24_UNORM_S8_UINT
        // supports DEPTH_STENCIL_ATTACHMENT (required-format table) — but NOT
        // which one, so hardcoding D32_S8 is unportable (RADV/ANV may prefer
        // D24_S8). Query and prefer D32_S8 (matches the depth-only D32_SFLOAT
        // path's 32-bit float depth), else fall back to D24_S8. Depth-only stays
        // D32_SFLOAT, which IS spec-mandatory — no query needed there.
        let supports_depth_stencil = |f: vk::Format| {
            instance
                .get_physical_device_format_properties(pd, f)
                .optimal_tiling_features
                .contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
        };
        let depth_stencil_format = if supports_depth_stencil(vk::Format::D32_SFLOAT_S8_UINT) {
            vk::Format::D32_SFLOAT_S8_UINT
        } else if supports_depth_stencil(vk::Format::D24_UNORM_S8_UINT) {
            vk::Format::D24_UNORM_S8_UINT
        } else {
            // The spec forbids this (one of the two is always supported); pick
            // D32_S8 so validation flags the impossible case rather than us
            // silently guessing an unsupported format.
            vk::Format::D32_SFLOAT_S8_UINT
        };
        let mut divisor_features = vk::PhysicalDeviceVertexAttributeDivisorFeaturesKHR::default();
        let mut divisor_properties =
            vk::PhysicalDeviceVertexAttributeDivisorPropertiesKHR::default();
        if vertex_attribute_divisor {
            let mut features =
                vk::PhysicalDeviceFeatures2::default().push_next(&mut divisor_features);
            instance.get_physical_device_features2(pd, &mut features);
            let mut properties =
                vk::PhysicalDeviceProperties2::default().push_next(&mut divisor_properties);
            instance.get_physical_device_properties2(pd, &mut properties);
        }
        let vertex_divisor = VertexDivisorCapabilities {
            instance_rate_divisor: divisor_features.vertex_attribute_instance_rate_divisor
                == vk::TRUE,
            zero_divisor: divisor_features.vertex_attribute_instance_rate_zero_divisor == vk::TRUE,
            max_divisor: divisor_properties.max_vertex_attrib_divisor,
        };
        let vertex_formats =
            crate::backend::vulkan::translate::VertexFormatSupport::probe(&instance, pd);
        let mut enabled_device_extensions = Vec::new();
        if portability_subset {
            enabled_device_extensions.push(vk::KHR_PORTABILITY_SUBSET_NAME.as_ptr());
        }
        if vertex_attribute_divisor {
            enabled_device_extensions.push(vk::KHR_VERTEX_ATTRIBUTE_DIVISOR_NAME.as_ptr());
        }
        #[cfg(feature = "host-window")]
        if swapchain {
            enabled_device_extensions.push(ash::khr::swapchain::NAME.as_ptr());
        }
        let mut enabled_divisor_features =
            vk::PhysicalDeviceVertexAttributeDivisorFeaturesKHR::default()
                .vertex_attribute_instance_rate_divisor(vertex_divisor.instance_rate_divisor)
                .vertex_attribute_instance_rate_zero_divisor(vertex_divisor.zero_divisor);
        let mut enabled_vulkan12 = features.enabled_vulkan12();
        // Any extension the feature set itself requires — today only the
        // pre-1.2 spelling of mirror-clamp-to-edge, on a device that has the
        // extension but not the core feature.
        enabled_device_extensions.extend(features.required_extensions());
        // These three are built in `caps` too. They are bound to locals here
        // only because `push_next` borrows them for the lifetime of `dci`.
        let mut en16 = features.enabled_16bit_storage();
        let mut en8 = features.enabled_8bit_storage();
        let mut enfi = features.enabled_float16_int8();
        let mut endemote = features.enabled_demote_to_helper();
        let mut dci = vk::DeviceCreateInfo::default()
            .queue_create_infos(&qci)
            .enabled_features(&enabled)
            .enabled_extension_names(&enabled_device_extensions)
            .push_next(&mut enabled_vulkan12);
        if vertex_attribute_divisor {
            dci = dci.push_next(&mut enabled_divisor_features);
        }
        if has16 {
            dci = dci.push_next(&mut en16);
        }
        if has8 {
            dci = dci.push_next(&mut en8);
        }
        if has_float16 || has_int8 {
            dci = dci.push_next(&mut enfi);
        }
        // Chained only when the device has the extension, whose name
        // `required_extensions` already added under the same condition.
        if features.shader_demote_to_helper_invocation {
            dci = dci.push_next(&mut endemote);
        }
        let device = instance
            .create_device(pd, &dci, None)
            .map_err(|result| DrawError::Init(InitDecline::CreateDevice { result }))?;
        let props = instance.get_physical_device_properties(pd);
        // Both halves are capability answers, not assumptions: Vulkan permits a
        // queue family to support no timestamps at all, and permits
        // `timestampPeriod` to be any positive float. A device that says either
        // gets no probe and the census reports zero rather than a wrong number.
        let timestamps = (qfs[gq as usize].timestamp_valid_bits > 0
            && props.limits.timestamp_period > 0.0)
            .then(|| {
                let ci = vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(TimestampProbe::SLOTS);
                device
                    .create_query_pool(&ci, None)
                    .map(|pool| TimestampProbe {
                        pool,
                        ns_per_tick: props.limits.timestamp_period,
                    })
                    .map_err(|e| {
                        crate::observe::Emit::decline(
                            "vk_timestamp_pool",
                            &VkCall::new(VkOp::ContextCreateQueryPool, e),
                        )
                        .fail_once(0);
                    })
                    .ok()
            })
            .flatten();
        let memory_properties = instance.get_physical_device_memory_properties(pd);
        let caps = HostGpuCaps {
            memory: classify_memory(&memory_properties),
            quirks: DriverQuirk::for_portability_subset(portability_subset),
            portability_subset,
            device_api_version: props.api_version,
            device_type: props.device_type,
        };
        let device_name = CStr::from_ptr(props.device_name.as_ptr())
            .to_string_lossy()
            .into_owned();
        // One-shot classification line: the memory topology, the signal that
        // decided it, and whether this device can hand a frame to another
        // device without a copy. Load-bearing for portability debugging — "why
        // is this host slow / blank" starts here.
        crate::observe::off(caps.selection_line(&device_name));
        // Fine-grained capabilities that do change what a draw can express.
        crate::observe::off(format!(
            "vk_device_select name={device_name:?} type={:?} depth_stencil_format={:?} bgra_storage_composite={} compute_capable={} quirks_no_deferred_batching={} quirks_guest_pages_authoritative={}",
            props.device_type,
            depth_stencil_format,
            storage_image_write_without_format_bgra,
            compute_capable,
            caps.quirks.no_deferred_draw_batching,
            caps.quirks.guest_pages_stay_authoritative,
        ));
        // Warm-start the pipeline cache from the previous boot's blob. Cold
        // pipeline compiles are the remaining pre-convergence stall class
        // (~256 ms first use per pipeline); the blob is keyed by the device's
        // pipelineCacheUUID so a driver/GPU change can never feed an
        // incompatible cache, and the header is validated before use
        // (passing a blob not produced by vkGetPipelineCacheData for this
        // device is a Vulkan valid-usage violation, not a soft fallback).
        let pipeline_cache_path = pipeline_cache_disk_path(&props.pipeline_cache_uuid);
        let initial_blob = match read_pipeline_cache_blob(&pipeline_cache_path, &props) {
            Ok(blob) => blob,
            Err(decline) => {
                crate::observe::Emit::decline("vk_pipeline_cache_load", &decline).fail_once(0);
                None
            }
        };
        let mut pcci = vk::PipelineCacheCreateInfo::default();
        if let Some(blob) = initial_blob.as_deref() {
            pcci = pcci.initial_data(blob);
        }
        let (pipeline_cache, initial_len) = match device.create_pipeline_cache(&pcci, None) {
            Ok(cache) => (cache, initial_blob.as_ref().map_or(0, Vec::len)),
            Err(result) if initial_blob.is_some() => {
                // The header matched, but the driver rejected the payload.
                // Continue cold, while preserving the warm failure and treating
                // the cold cache as length zero so the next save repairs disk.
                let decline = PipelineCacheDecline::WarmCreate { result };
                crate::observe::Emit::decline("vk_pipeline_cache_load", &decline).fail_once(0);
                let cache = device
                    .create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None)
                    .map_err(|result| {
                        DrawError::Init(InitDecline::CreatePipelineCache { result })
                    })?;
                (cache, 0)
            }
            Err(result) => {
                return Err(DrawError::Init(InitDecline::CreatePipelineCache { result }));
            }
        };
        crate::observe::off(format!(
            "vk_pipeline_cache_load bytes={initial_len} path={}",
            pipeline_cache_path.display()
        ));
        Ok(Self {
            _entry: entry,
            instance,
            pd,
            device,
            caps,
            memory_properties,
            gq,
            compute_capable,
            storage_image_write_without_format: storage_image_write_without_format_bgra,
            sampled_r32f_linear_filter,
            pipeline_cache,
            vertex_divisor,
            vertex_formats,
            max_sampler_anisotropy: features.max_sampler_anisotropy,
            sampler_anisotropy: features.sampler_anisotropy,
            features,
            depth_stencil_format,
            timestamps,
            pipeline_cache_path: Some(pipeline_cache_path),
            pipeline_cache_saved_len: AtomicUsize::new(initial_len),
            #[cfg(feature = "host-window")]
            swapchain,
        })
    }

    /// Persist the pipeline cache to disk when it has grown since the last
    /// save. Called after each actual pipeline creation (cache misses only —
    /// warm hits never reach this). The serialize under the engine lock is a
    /// memcpy; the file write runs on a detached thread so nothing on the
    /// draw path blocks on disk. Saving on creation rather than at context
    /// destroy is deliberate: the testing boot SIGKILLs QEMU, so destroy
    /// never runs there. The tmp-then-rename keeps a concurrent reader (or a
    /// second QEMU process) from ever seeing a torn blob.
    pub(crate) fn persist_pipeline_cache(&self) {
        let Some(path) = self.pipeline_cache_path.clone() else {
            return;
        };
        let data = match unsafe { self.device.get_pipeline_cache_data(self.pipeline_cache) } {
            Ok(d) => d,
            Err(e) => {
                let decline = VkCall::new(VkOp::ContextPipelineCacheGetData, e);
                crate::observe::Emit::decline("vk_pipeline_cache_save", &decline).fail_once(0);
                return;
            }
        };
        // Growth debounce: byte length is the proxy for "a new pipeline
        // landed" (equal-length different-content saves are lost, which only
        // costs a warm-start miss on that one pipeline next boot).
        if data.len()
            == self
                .pipeline_cache_saved_len
                .swap(data.len(), Ordering::Relaxed)
        {
            return;
        }
        // Unique tmp name PER SAVE. Keying only on the (constant) pid meant two
        // concurrent saves — spawned when two calls with different data lengths
        // both clear the growth debounce — wrote the SAME tmp file, so the first
        // thread's rename(tmp→path) moved it out from under the second thread's
        // rename, which then failed ENOENT (the intermittent
        // `vk_pipeline_cache_save reason=vk_pipeline_cache_rename errno=2 ...`).
        // A per-save sequence number makes each thread's tmp file private, so the
        // write→rename is race-free and the newest save always lands.
        static SAVE_SEQ: AtomicU64 = AtomicU64::new(0);
        // Largest cache length already landed on disk. The VkPipelineCache only
        // grows, so `data.len()` orders the snapshots. Best-effort newest-wins:
        // if a strictly larger snapshot already landed by the time this thread is
        // about to rename, drop this smaller one rather than regress the on-disk
        // cache to a stale subset. This narrows (does not fully serialize) the
        // concurrent-save window — a residual reorder only costs one pipeline a
        // warm-start miss next boot and self-heals on the next compile, so a lock
        // is not warranted for a best-effort cache. Keyed per physical device via
        // the UUID-derived path (a DEVICE_LOST recreate reuses the same file), so
        // a process-wide static is correct.
        static PERSISTED_LEN: AtomicUsize = AtomicUsize::new(0);
        let seq = SAVE_SEQ.fetch_add(1, Ordering::Relaxed);
        std::thread::spawn(move || {
            let tmp = path.with_extension(format!("tmp.{}.{}", std::process::id(), seq));
            match write_cache_atomic(&path, &tmp, &data, &PERSISTED_LEN) {
                Ok(CacheSaveOutcome::Landed) => crate::observe::off(format!(
                    "vk_pipeline_cache_save bytes={} path={}",
                    data.len(),
                    path.display()
                )),
                Ok(CacheSaveOutcome::Superseded) => {}
                Err(decline) => {
                    crate::observe::Emit::decline("vk_pipeline_cache_save", &decline).fail_once(0)
                }
            }
        });
    }

    pub(crate) unsafe fn destroy(&mut self) {
        if let Some(probe) = self.timestamps.take() {
            self.device.destroy_query_pool(probe.pool, None);
        }
        self.device
            .destroy_pipeline_cache(self.pipeline_cache, None);
        self.device.destroy_device(None);
        self.instance.destroy_instance(None);
    }

    /// Pick a memory type for `class` on this device.
    ///
    /// This is the ONLY memory-type entry point. Call sites name what the
    /// memory is *for*; the topology-dependent flag choice lives in
    /// [`crate::backend::vulkan::caps::memory_topology`], so a unified host can
    /// skip a staging hop and a discrete host can avoid burning its BAR window
    /// without either decision being duplicated at an allocation site.
    ///
    /// Returns `None` only when no type in `type_bits` carries the class's
    /// *required* flags — the caller must then decline with a named reason.
    pub(crate) fn memory_type_for(&self, type_bits: u32, class: MemoryClass) -> Option<u32> {
        let picked = self.memory_type_with(type_bits, &self.caps.memory_request(class));
        // Once per class per boot. What a class *asks* for is in
        // `MemoryTopology::request` and readable from source; what it *gets* is
        // not, because it depends on this device's memory-type table, and the
        // two answers have very different costs. `vk_alloc_sites` prices
        // `MemoryClass::Upload` at 2.54 ms per MiB allocated against 0.48 for
        // `Readback` and 0.018 for the device-local slab, and a difference that
        // size is a difference in which heap the pick landed in. Naming the
        // index and its flags is what turns that from an inference into a
        // reading.
        if let Some(i) = picked {
            // Keyed on the class and the index together, so a device whose
            // table makes the pick differ between call sites says so instead of
            // latching the first answer for the boot.
            let key = ((class as u64) << 32) | i as u64;
            if crate::observe::first_sight("vk_memory_type_pick", key) {
                let t = self.memory_properties.memory_types[i as usize];
                crate::observe::off(format!(
                    "vk_memory_type_pick class={class:?} index={i} heap={} flags={:?} \
                     heap_bytes={}",
                    t.heap_index,
                    t.property_flags,
                    self.memory_properties.memory_heaps[t.heap_index as usize].size,
                ));
            }
        }
        picked
    }

    /// Escape hatch for a caller that has already built a [`MemoryRequest`]
    /// (the dmabuf import path, which must match a foreign allocation).
    pub(crate) fn memory_type_with(&self, type_bits: u32, req: &MemoryRequest) -> Option<u32> {
        select_memory_type(&self.memory_properties, type_bits, req)
    }

    /// Whether a selected memory type is host-cached and whether it is coherent.
    ///
    /// [`MemoryClass::Readback`] ranks cached above coherent, so a readback
    /// allocation can legitimately be non-coherent and its reader owes an
    /// invalidate. A site that maps memory must ask rather than assume.
    pub(crate) fn mapped_memory_kind(&self, memory_type_index: u32) -> MappedMemoryKind {
        MappedMemoryKind::of(&self.memory_properties, memory_type_index)
    }

    pub(crate) fn queue(&self) -> vk::Queue {
        unsafe { self.device.get_device_queue(self.gq, 0) }
    }
}

/// Process-global engine state ownership (device + recreate policy + init fail cache).
pub(crate) struct ContextOwner {
    pub ctx: Option<DeviceContext>,
    pub init_error: Option<DrawError>,
    pub recreate_count: u32,
    pub poisoned: bool,
    /// Test hook: force next submit/wait to report DEVICE_LOST without real GPU fault.
    pub force_device_lost: bool,
    pub loss_events: AtomicU64,
}

impl ContextOwner {
    pub(crate) fn new() -> Self {
        Self {
            ctx: None,
            init_error: None,
            recreate_count: 0,
            poisoned: false,
            force_device_lost: false,
            loss_events: AtomicU64::new(0),
        }
    }

    pub(crate) fn ensure(
        &mut self,
        counters: &EngineCounters,
    ) -> Result<&DeviceContext, DrawError> {
        if let Some(error) = &self.init_error {
            return Err(error.clone());
        }
        if self.poisoned {
            if self.recreate_count >= MAX_DEVICE_RECREATES {
                return Err(DrawError::DeviceLost(
                    DeviceLostDecline::RecreateCapExhausted {
                        cap: MAX_DEVICE_RECREATES,
                    },
                ));
            }
            self.try_recreate(counters)?;
        }
        if self.ctx.is_none() {
            match unsafe { DeviceContext::create() } {
                Ok(c) => self.ctx = Some(c),
                Err(e) => {
                    self.init_error = Some(e.clone());
                    return Err(e);
                }
            }
        }
        Ok(self.ctx.as_ref().unwrap())
    }

    fn try_recreate(&mut self, counters: &EngineCounters) -> Result<(), DrawError> {
        if self.recreate_count >= MAX_DEVICE_RECREATES {
            return Err(DrawError::DeviceLost(
                DeviceLostDecline::RecreateCapExhausted {
                    cap: MAX_DEVICE_RECREATES,
                },
            ));
        }
        if let Some(mut old) = self.ctx.take() {
            unsafe { old.destroy() };
        }
        self.recreate_count += 1;
        counters.recreates.fetch_add(1, Ordering::Relaxed);
        match unsafe { DeviceContext::create() } {
            Ok(c) => {
                self.ctx = Some(c);
                self.poisoned = false;
                Ok(())
            }
            Err(e) => {
                self.poisoned = true;
                Err(DrawError::DeviceLost(DeviceLostDecline::RecreateFailed {
                    cause: Box::new(e),
                }))
            }
        }
    }

    pub(crate) fn mark_device_lost(&mut self) {
        self.loss_events.fetch_add(1, Ordering::Relaxed);
        self.poisoned = true;
    }
}

/// Main-entry name for shader stages (stable ABI).
pub(crate) fn main_entry() -> CString {
    CString::new("main").expect("static")
}

#[cfg(test)]
mod pipeline_cache_blob_tests {
    use super::*;

    fn props(vendor: u32, device: u32, uuid: [u8; 16]) -> vk::PhysicalDeviceProperties {
        vk::PhysicalDeviceProperties {
            vendor_id: vendor,
            device_id: device,
            pipeline_cache_uuid: uuid,
            ..vk::PhysicalDeviceProperties::default()
        }
    }

    fn blob(header_len: u32, version: u32, vendor: u32, device: u32, uuid: [u8; 16]) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&header_len.to_le_bytes());
        b.extend_from_slice(&version.to_le_bytes());
        b.extend_from_slice(&vendor.to_le_bytes());
        b.extend_from_slice(&device.to_le_bytes());
        b.extend_from_slice(&uuid);
        b
    }

    const UUID: [u8; 16] = [7u8; 16];

    /// A failed initialization is negative-cached so every later draw fails
    /// fast. The cache must retain the typed error itself: the former
    /// `to_string()` + `DrawError::Init(String)` round-trip replaced the
    /// original check with `vk_engine_init_untyped` (and doubled the display
    /// prefix) on every retry.
    #[test]
    fn initialization_negative_cache_preserves_the_exact_decline() {
        let expected = DrawError::Init(InitDecline::CreateDevice {
            result: vk::Result::ERROR_EXTENSION_NOT_PRESENT,
        });
        let mut owner = ContextOwner::new();
        owner.init_error = Some(expected.clone());
        let counters = EngineCounters::default();

        let actual = match owner.ensure(&counters) {
            Ok(_) => panic!("a negative-cached initialization must fail"),
            Err(error) => error,
        };

        assert_eq!(actual, expected);
        assert_eq!(
            crate::observe::Decline::slug(&actual),
            "vk_init_create_device"
        );
        let shown = actual.to_string();
        assert_eq!(
            shown.matches("vk_engine_init:").count(),
            1,
            "the cache must not wrap an already-rendered DrawError: {shown}"
        );
        assert!(
            shown.contains("reason=vk_init_create_device vk_result="),
            "{shown}"
        );
    }

    /// A blob written by vkGetPipelineCacheData for this exact device must be
    /// accepted — that is the whole warm-start path.
    #[test]
    fn matching_header_accepted() {
        let p = props(0x10de, 0x2c02, UUID);
        assert!(pipeline_cache_blob_compatible(
            &blob(32, 1, 0x10de, 0x2c02, UUID),
            &p
        ));
    }

    /// Feeding initial data from another device/driver is a Vulkan
    /// valid-usage violation — every mismatching field must reject.
    #[test]
    fn mismatches_rejected() {
        let p = props(0x10de, 0x2c02, UUID);
        // wrong vendor
        assert!(!pipeline_cache_blob_compatible(
            &blob(32, 1, 0x1002, 0x2c02, UUID),
            &p
        ));
        // wrong device
        assert!(!pipeline_cache_blob_compatible(
            &blob(32, 1, 0x10de, 0x9999, UUID),
            &p
        ));
        // wrong UUID (driver update rotates it)
        assert!(!pipeline_cache_blob_compatible(
            &blob(32, 1, 0x10de, 0x2c02, [8u8; 16]),
            &p
        ));
        // wrong header version
        assert!(!pipeline_cache_blob_compatible(
            &blob(32, 2, 0x10de, 0x2c02, UUID),
            &p
        ));
        // header shorter than VkPipelineCacheHeaderVersionOne
        assert!(!pipeline_cache_blob_compatible(
            &blob(16, 1, 0x10de, 0x2c02, UUID),
            &p
        ));
    }

    /// A truncated file (torn write, disk full) must reject, not panic.
    #[test]
    fn short_blob_rejected() {
        let p = props(0x10de, 0x2c02, UUID);
        assert!(!pipeline_cache_blob_compatible(&[], &p));
        assert!(!pipeline_cache_blob_compatible(&[0u8; 31], &p));
    }

    #[test]
    fn cold_cache_fallbacks_distinguish_absence_corruption_read_and_driver_rejection() {
        use crate::observe::Decline as _;
        let root =
            std::env::temp_dir().join(format!("reims-vgpu-cache-load-test-{}", std::process::id()));
        std::fs::remove_dir_all(&root).ok();
        std::fs::create_dir_all(&root).unwrap();
        let p = props(0x10de, 0x2c02, UUID);

        assert_eq!(
            read_pipeline_cache_blob(&root.join("missing.bin"), &p).unwrap(),
            None,
            "a first boot has no cache and is not a decline"
        );

        let corrupt_path = root.join("corrupt.bin");
        std::fs::write(&corrupt_path, [1, 2, 3]).unwrap();
        let corrupt = read_pipeline_cache_blob(&corrupt_path, &p).unwrap_err();
        assert_eq!(corrupt.slug(), "vk_pipeline_cache_incompatible");
        assert_eq!(corrupt.fields(), vec![("bytes", "3".into())]);

        let read = read_pipeline_cache_blob(&root, &p).unwrap_err();
        assert_eq!(read.slug(), "vk_pipeline_cache_read");
        assert_eq!(read.fields()[1].0, "io_kind");

        let warm = PipelineCacheDecline::WarmCreate {
            result: vk::Result::ERROR_INITIALIZATION_FAILED,
        };
        assert_eq!(warm.slug(), "vk_pipeline_cache_warm_create");
        assert_eq!(warm.fields()[0].0, "vk_result");
        std::fs::remove_dir_all(&root).ok();
    }

    /// The path is UUID-keyed: distinct devices never share a blob file.
    #[test]
    fn disk_path_keyed_by_uuid() {
        let a = pipeline_cache_disk_path(&[1u8; 16]);
        let b = pipeline_cache_disk_path(&[2u8; 16]);
        assert_ne!(a, b);
        assert!(a
            .file_name()
            .unwrap()
            .to_string_lossy()
            .contains(&"01".repeat(16)));
    }

    /// A single save lands the blob at `path` and consumes its tmp file.
    #[test]
    fn write_cache_atomic_lands_and_cleans_tmp() {
        let dir =
            std::env::temp_dir().join(format!("reims-vgpu-cache-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("cache.bin");
        let tmp = dir.join("cache.tmp.0");
        let persisted = AtomicUsize::new(0);
        let out = write_cache_atomic(&path, &tmp, b"pipelines-v1", &persisted).unwrap();
        assert_eq!(out, CacheSaveOutcome::Landed);
        assert_eq!(std::fs::read(&path).unwrap(), b"pipelines-v1");
        assert!(!tmp.exists(), "tmp file consumed by rename");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Newest-wins: a smaller snapshot arriving after a larger one already landed
    /// is dropped (Superseded) and does NOT regress the on-disk cache — and its
    /// tmp file is cleaned up. This is the ordering the concurrent-save guard
    /// protects; each save uses a DISTINCT tmp path (per-seq) so they never
    /// collide (the ENOENT bug).
    #[test]
    fn write_cache_atomic_newest_wins_and_no_tmp_collision() {
        let dir =
            std::env::temp_dir().join(format!("reims-vgpu-cache-test2-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("cache.bin");
        let persisted = AtomicUsize::new(0);
        // Larger snapshot lands first.
        let big = vec![0xABu8; 4096];
        let tmp_big = dir.join("cache.tmp.1");
        assert_eq!(
            write_cache_atomic(&path, &tmp_big, &big, &persisted).unwrap(),
            CacheSaveOutcome::Landed
        );
        // A smaller, later save (distinct tmp path) is superseded, leaves the
        // large blob intact, and cleans its own tmp.
        let small = vec![0xCDu8; 512];
        let tmp_small = dir.join("cache.tmp.2");
        assert_eq!(
            write_cache_atomic(&path, &tmp_small, &small, &persisted).unwrap(),
            CacheSaveOutcome::Superseded
        );
        assert_eq!(
            std::fs::read(&path).unwrap(),
            big,
            "on-disk cache not regressed"
        );
        assert!(!tmp_small.exists(), "superseded tmp cleaned up");
        assert!(!tmp_big.exists(), "landed tmp consumed");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A larger snapshot after a smaller one lands (upgrade path).
    #[test]
    fn write_cache_atomic_larger_upgrades() {
        let dir =
            std::env::temp_dir().join(format!("reims-vgpu-cache-test3-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("cache.bin");
        let persisted = AtomicUsize::new(0);
        assert_eq!(
            write_cache_atomic(&path, &dir.join("t.0"), b"small", &persisted).unwrap(),
            CacheSaveOutcome::Landed
        );
        let big = vec![0x11u8; 2048];
        assert_eq!(
            write_cache_atomic(&path, &dir.join("t.1"), &big, &persisted).unwrap(),
            CacheSaveOutcome::Landed
        );
        assert_eq!(
            std::fs::read(&path).unwrap(),
            big,
            "larger snapshot upgraded the cache"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn cache_persist_failures_name_write_and_rename_separately() {
        use crate::observe::Decline as _;
        let root = std::env::temp_dir().join(format!(
            "reims-vgpu-cache-error-test-{}",
            std::process::id()
        ));
        std::fs::remove_dir_all(&root).ok();
        let persisted = AtomicUsize::new(0);

        let write = write_cache_atomic(
            &root.join("cache.bin"),
            &root.join("missing").join("cache.tmp"),
            b"cache",
            &persisted,
        )
        .expect_err("missing tmp parent must fail the write stage");
        assert_eq!(write.slug(), "vk_pipeline_cache_write");

        std::fs::create_dir_all(&root).unwrap();
        let rename =
            write_cache_atomic(&root, &root.join("cache.tmp"), b"cache-larger", &persisted)
                .expect_err("renaming a file over a directory must fail");
        assert_eq!(rename.slug(), "vk_pipeline_cache_rename");
        for decline in [write, rename] {
            let fields = decline.fields();
            assert_eq!(fields.len(), 2);
            assert_eq!(fields[0].0, "errno");
            assert_eq!(fields[1].0, "io_kind");
            for (_, value) in fields {
                assert!(!value.is_empty());
                assert!(!value.contains(char::is_whitespace));
            }
        }
        std::fs::remove_dir_all(&root).ok();
    }
}
