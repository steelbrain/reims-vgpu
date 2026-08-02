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
use crate::contract::pixel_format::TexelLayout;

/// Max device recreates **that produce no guest work between them**.
///
/// The bound is on a recreate *storm* — lost, rebuilt, lost again before
/// anything ran — which is the case a device cannot get out of and where
/// retrying forever is a spin. It is deliberately not a lifetime count. A GPU
/// that resets is not a GPU that is finished, and a host driver that reset three
/// times over a week of uptime would otherwise leave this device answering
/// `RecreateCapExhausted` to every draw for as long as the guest ran, with hours
/// of correct rendering between each of the three.
///
/// [`ContextOwner::note_work_completed`] is what makes the difference: one draw
/// or dispatch that reaches the GPU on the rebuilt device says that recreate did
/// its job, and the budget starts over. A storm never reaches it.
pub const MAX_DEVICE_RECREATES: u32 = 3;

/// Bounded fence wait (nanoseconds). Named constant — not env-gated.
pub const FENCE_TIMEOUT_NS: u64 = 5_000_000_000; // 5s

/// `sizeof(VkPipelineCacheHeaderVersionOne)` (Vulkan spec §Pipeline Cache
/// Header): u32 headerSize, u32 headerVersion, u32 vendorID, u32 deviceID,
/// u8[16] pipelineCacheUUID — all integers little-endian.
const PIPELINE_CACHE_HEADER_ONE_LEN: usize = 32;

/// Largest warm-start blob worth handing the driver, and largest worth writing
/// back.
///
/// A `VkPipelineCache` has no eviction: `vkGetPipelineCacheData` returns every
/// entry it ever accumulated, and this device saves that snapshot to disk and
/// reloads it next boot. Across many boots of many builds the blob therefore
/// only grows, and the save policy below orders snapshots by length on the
/// explicit assumption that a longer one is a better one.
///
/// That assumption is false past a few megabytes, and the cost is not small.
/// Four driven macos-13 hammer boots, identical snapshot and probe, host
/// quiesced, differing only in the blob on disk:
///
/// ```text
/// loaded      pipeline_misses   pl_compile_us   per compile
///  30.82 MB        229              0.946 s       4.13 ms
///  30.82 MB        236              0.932 s       3.95 ms
///   0.00 MB        220              0.309 s       1.41 ms
///   3.85 MB        220              0.161 s       0.73 ms
/// ```
///
/// The miss count barely moves — the in-memory `ObjectCache` is empty at boot
/// either way, so the same ~220 `vkCreateGraphicsPipelines` calls happen. What
/// changes is what each one costs, and an oversized blob makes it **5.7x** what
/// a right-sized one costs and **2.9x** what no blob at all costs. A warm cache
/// is worth having; an unbounded one is worse than none.
///
/// Why a large blob costs more per compile is the driver's business and is not
/// measured here. Size is the observable, so size is what this bounds.
///
/// 16 MiB is four times the ~3.9 MB a boot of this workload settles at, so a
/// workload several times richer still warms fully, and it is half the 30.8 MB
/// at which the penalty was measured. Growth at steady state is ~0.1 MB a boot
/// (3.85 -> 3.95 across two boots), so this is ~120 boots of headroom before the
/// bound is reached and the blob is rebuilt from one boot.
const PIPELINE_CACHE_MAX_WARM_BYTES: usize = 16 * 1024 * 1024;

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
    /// The blob is well-formed and this device's, but larger than
    /// [`PIPELINE_CACHE_MAX_WARM_BYTES`], so warming from it would cost more per
    /// compile than starting cold. Declined by name rather than silently
    /// ignored: this boot pays real cold-start compiles, and the next boot sees
    /// a rebuilt blob, so a reader has to be able to tell that from a first boot.
    TooLarge {
        bytes: usize,
        cap: usize,
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
            Self::TooLarge { .. } => "vk_pipeline_cache_too_large",
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
            Self::TooLarge { bytes, cap } => {
                vec![("bytes", bytes.to_string()), ("cap", cap.to_string())]
            }
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
        // Size is checked before compatibility only so the larger, cheaper-to-
        // decide refusal wins the report; both are cold starts either way.
        Ok(blob) if blob.len() > PIPELINE_CACHE_MAX_WARM_BYTES => {
            Err(PipelineCacheDecline::TooLarge {
                bytes: blob.len(),
                cap: PIPELINE_CACHE_MAX_WARM_BYTES,
            })
        }
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

/// A timestamp pair per submission ring slot: the top of a draw command buffer
/// and its bottom.
///
/// Why a pair per *slot* and not one pair shared: up to
/// [`super::pools::RING_DEPTH`] command buffers are in flight at once, and a
/// shared pair would be overwritten by the next submission long before the first
/// one's fence made it readable. The slot is the natural key because the slot's
/// fence is exactly the event that makes its pair readable, and the ring already
/// retires a slot before reusing it — see [`super::gpu_span::SlotSpan`].
pub(crate) struct DrawSpanProbe {
    pub pool: vk::QueryPool,
    /// `VkPhysicalDeviceLimits::timestampPeriod` — nanoseconds per tick.
    pub ns_per_tick: f32,
    /// The low `timestampValidBits` of a query result, as a mask.
    ///
    /// Vulkan permits a queue family to write fewer than 64 meaningful bits and
    /// leaves the rest **undefined**, so a raw subtraction of two results is not
    /// the elapsed ticks on such a host — it is a difference of two numbers whose
    /// high halves are whatever the driver left there. Masking both operands
    /// first is the whole fix, and a counter that wraps within the mask is then a
    /// wrapping subtract rather than a garbage one. 32 valid bits at a 1 ns tick
    /// wraps every 4.3 s, which a per-second census crosses regularly.
    pub valid_mask: u64,
}

impl DrawSpanProbe {
    /// Queries per ring slot: the top stamp and the bottom stamp.
    pub const PER_SLOT: u32 = 2;

    /// First query index belonging to ring slot `slot`.
    pub const fn base(slot: usize) -> u32 {
        slot as u32 * Self::PER_SLOT
    }

    /// Elapsed nanoseconds between two raw query results, masked to the bits the
    /// queue family actually writes and treating a wrap within that width as a
    /// wrap rather than as an enormous negative.
    pub fn elapsed_ns(&self, top: u64, bottom: u64) -> u64 {
        let span = (bottom & self.valid_mask).wrapping_sub(top & self.valid_mask) & self.valid_mask;
        (span as f64 * self.ns_per_tick as f64) as u64
    }
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
    /// `VK_EXT_external_memory_host` entry points, loaded only where
    /// [`HostGpuCaps::host_pointer`] resolved to `Supported` — which is also the
    /// only rung on which the extension was enabled, so loading it on any other
    /// would resolve entry points the device was never asked for.
    ///
    /// `None` is the answer on every host without the extension, and the import
    /// site declines by name when it sees one.
    pub external_memory_host: Option<ash::ext::external_memory_host::Device>,
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
    /// The raw `shaderStorageImageWriteWithoutFormat` /
    /// `shaderStorageImageReadWithoutFormat` /
    /// `shaderStorageImageExtendedFormats` features, unmixed with any surface
    /// question.
    ///
    /// [`Self::storage_image_write_without_format`] above is the *BGRA compositing*
    /// answer — it also requires a BGRA8 storage surface — so it is the wrong
    /// gate for "may this module declare the capability". Declaring one whose
    /// feature was not enabled at device creation is invalid usage, and an
    /// invalid module is undefined behaviour inside a driver rather than an
    /// error it returns, so the two questions get two fields.
    pub spirv_storage_write_without_format: bool,
    pub spirv_storage_read_without_format: bool,
    pub spirv_storage_extended_formats: bool,
    /// For each [`crate::contract::pixel_format::TexelLayout`], whether this
    /// host can sample its Vulkan format with linear filtering; gates the
    /// native sampled rails (see [`DeviceFeatures::sampled_linear_filter`]).
    pub sampled_linear_filter: [bool; TexelLayout::ALL.len()],
    pub pipeline_cache: vk::PipelineCache,
    pub vertex_divisor: VertexDivisorCapabilities,
    /// Offset alignment a guest-window import must satisfy before a draw may
    /// bind it directly as a vertex or storage buffer, taken from
    /// `VkPhysicalDeviceLimits`.
    ///
    /// One number for both because the dedup in `stage_buffer_content` shares a
    /// bound buffer between a stream that feeds vertex fetch and one that feeds
    /// a storage binding, so a value legal for only one of them would be a
    /// valid-usage violation the moment the same guest span is used twice.
    /// `min*BufferOffsetAlignment` are the device's own answers; the 16 floor
    /// covers vertex fetch, which the spec gives no queryable limit for and
    /// which implementations may require to be component-aligned — 16 bytes is
    /// the largest component-size any vertex format has.
    pub guest_bind_offset_align: u64,
    /// The widest span this device will bind as one storage buffer, from
    /// `VkPhysicalDeviceLimits::maxStorageBufferRange`.
    ///
    /// Read rather than assumed because a guest RAMBlock is routinely wider than
    /// it — a 16 GiB guest against a limit that is a `uint32_t` — so the
    /// guest-scatter kernel binds a window over the block rather than the block,
    /// and this is the bound that window is checked against. See
    /// [`super::guest_scatter::build_run_table`].
    pub max_storage_buffer_range: u64,
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
    /// Two timestamps per ring slot, for the GPU execution time of a draw
    /// submission. `None` on the same two capability answers as
    /// [`Self::timestamps`], and additionally when [`crate::env::GPU_SPANS`] is
    /// off — which is the whole of how that switch narrows, because a `None` here
    /// means no query is ever reset, written or read.
    ///
    /// Separate from [`Self::timestamps`] rather than a wider pool because the two
    /// are indexed by different things: the readback's three queries are safe to
    /// share for the device's life precisely because that path is serialized,
    /// while a draw's pair belongs to the ring slot whose fence will make it
    /// readable. See [`super::gpu_span`].
    pub draw_spans: Option<DrawSpanProbe>,
    /// The thread that announces a GPU-written completion stamp, and the
    /// timeline semaphore its submissions signal.
    ///
    /// `None` when the device does not advertise `timelineSemaphore` or the
    /// thread would not start — the stamp rail then falls back to blocking the
    /// drain worker, which is what every host did before this existed. See
    /// [`super::stamp_completion`] for why the interrupt cannot be deferred to
    /// anything that runs on a schedule.
    pub stamp_completion: Option<super::stamp_completion::StampCompletion>,
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
            #[cfg(not(any(target_os = "macos", target_os = "linux")))]
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
        // One line per enumerated device, emitted **before** the selection so the
        // list survives a boot where nothing clears the floor and there is no
        // winner to hang it off.
        //
        // A hybrid laptop enumerates two GPUs and this device silently binds one
        // of them; until this line existed, a report from such a host could not
        // say which, nor that the other existed, nor why it lost. The rank is on
        // the line because the rank *is* the policy — a reader who disagrees with
        // the choice can see the number that made it — and the driver identity is
        // there because `DriverQuirk` is the one place driver identity may change
        // behavior and a quirk report needs the driver's own name for itself
        // rather than the marketing name of the silicon.
        //
        // `VkPhysicalDeviceDriverProperties` is Vulkan 1.2 core and 1.2 is the
        // baseline, so it is answerable on every device that could be selected.
        // A device *below* the floor may not answer it; the struct is
        // zero-initialised, so such a device reports empty strings next to
        // `above_floor=false`, which reads correctly.
        for (index, (api, device_type, candidate)) in candidates.iter().enumerate() {
            let props = instance.get_physical_device_properties(*candidate);
            let name = CStr::from_ptr(props.device_name.as_ptr()).to_string_lossy();
            let mut driver = vk::PhysicalDeviceDriverProperties::default();
            let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut driver);
            instance.get_physical_device_properties2(*candidate, &mut props2);
            let driver_name = CStr::from_ptr(driver.driver_name.as_ptr()).to_string_lossy();
            let driver_info = CStr::from_ptr(driver.driver_info.as_ptr()).to_string_lossy();
            let profile =
                classify_memory(&instance.get_physical_device_memory_properties(*candidate));
            crate::observe::off(format!(
                "vk_device_candidate index={index} of={} name={name:?} type={device_type:?} \
                 api={} above_floor={} rank={} driver_id={:?} driver={driver_name:?} \
                 driver_info={driver_info:?} memory={} device_local_mb={} \
                 vendor_id={:#06x} device_id={:#06x}",
                candidates.len(),
                api_floor::version_str(*api),
                api_floor::meets_floor(*api),
                crate::backend::vulkan::caps::device_select::rank_physical_device(*device_type),
                driver.driver_id,
                profile.topology.slug(),
                profile.device_local_bytes >> 20,
                props.vendor_id,
                props.device_id,
            ));
        }
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
        // A queue family that transfers and does nothing else is a dedicated
        // copy engine, and the reason to want one is that this device's largest
        // remaining cost is bytes crossing the bus: a driven Safari drag moves
        // ~2.7 GB of guest buffer runs into device-local memory and writes
        // ~5.1 GB of rendered surface back to guest pages every second, all of
        // it recorded into the draw's own command buffer, where it serialises
        // against the rendering it feeds.
        //
        // Selected here and reported below, and nothing yet asks for a queue
        // from it — the reading comes first, on every pathway, because most
        // integrated parts have no such family and a rail built for one would
        // then be a rail most hosts never take.
        //
        // Selected by what the family can do, never by device or driver name.
        // `TRANSFER` alone is the whole test: `GRAPHICS` or `COMPUTE` implies
        // transfer support, so a family carrying either is the one this device
        // already submits draws to and moving copies there buys nothing.
        // Sparse-binding and the video/optical-flow bits do not disqualify a
        // family — they say what else the hardware block can do, not that it
        // shares the graphics engine.
        //
        // `None` on a host with no such family, where everything stays on `gq`.
        // That is not a fallback: it is the only arrangement most integrated
        // parts offer.
        let transfer_qf = dedicated_transfer_family(&qfs);
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
        let sampled_linear_filter = features.sampled_linear_filter;
        let has16 = features.storage16;
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
        // Whether guest RAM can reach this device as a host-pointer import over
        // whole RAMBlocks, and at what granularity. Same shape as the query
        // above and for the same reason: the answer is the only producer of the
        // extension string it requires.
        let host_pointer =
            crate::backend::vulkan::caps::host_pointer::query(&instance, pd, &has_device_extension);
        // Published for `runtime::guest_ram_map`, which builds the imports and
        // has no device context to read the granularity or the heap sizes from.
        // A negative rung withdraws them rather than publishing zeroes, so the
        // absence of a number is itself the gate and no site can act on a
        // granularity from a device that declined the handle type.
        match host_pointer.rung {
            crate::backend::vulkan::caps::HostPointerImport::Supported => {
                crate::runtime::guest_ram::latch_import_limits(
                    host_pointer.min_alignment,
                    host_pointer.heap_budget,
                );
            }
            _ => crate::runtime::guest_ram::forget_import_limits(),
        }
        // Every import this process holds names a `VkDeviceMemory` that dies
        // with the device below. Dropping them here, before the new one exists,
        // is what makes a recreate rebuild against fresh identities instead of
        // resolving a stale slice against a handle that is gone.
        crate::runtime::guest_ram_map::reset();
        let portability_subset = has_device_extension(vk::KHR_PORTABILITY_SUBSET_NAME);
        let vertex_attribute_divisor = has_device_extension(vk::KHR_VERTEX_ATTRIBUTE_DIVISOR_NAME);
        // A guest draw may name a texture the request never provides, leaving a
        // shader-declared descriptor with nothing written to it. Metal reads
        // such a slot as zeros; Vulkan calls it undefined, and the two ICDs we
        // run on disagree loudly — NVIDIA returns garbage, lavapipe faults on a
        // null dereference and takes the process with it. nullDescriptor makes
        // the read defined as zeros on both, which is the Metal behaviour the
        // guest already expects.
        let null_descriptor = has_device_extension(vk::EXT_ROBUSTNESS2_NAME);
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
        if null_descriptor {
            enabled_device_extensions.push(vk::EXT_ROBUSTNESS2_NAME.as_ptr());
        }
        #[cfg(feature = "host-window")]
        if swapchain {
            enabled_device_extensions.push(ash::khr::swapchain::NAME.as_ptr());
        }
        let mut enabled_robustness2 =
            vk::PhysicalDeviceRobustness2FeaturesEXT::default().null_descriptor(null_descriptor);
        let mut enabled_divisor_features =
            vk::PhysicalDeviceVertexAttributeDivisorFeaturesKHR::default()
                .vertex_attribute_instance_rate_divisor(vertex_divisor.instance_rate_divisor)
                .vertex_attribute_instance_rate_zero_divisor(vertex_divisor.zero_divisor);
        let mut enabled_vulkan12 = features.enabled_vulkan12();
        // Any extension the feature set itself requires — today only the
        // pre-1.2 spelling of mirror-clamp-to-edge, on a device that has the
        // extension but not the core feature.
        enabled_device_extensions.extend(features.required_extensions());
        // Only the `Supported` rung names `VK_EXT_external_memory_host`, so a
        // host without it gets a device rather than a failed `vkCreateDevice`.
        enabled_device_extensions.extend(host_pointer.rung.required_extensions());
        // These three are built in `caps` too. They are bound to locals here
        // only because `push_next` borrows them for the lifetime of `dci`.
        let mut en16 = features.enabled_16bit_storage();
        let mut en8 = features.enabled_8bit_storage();
        let mut enfi = features.enabled_float16_int8();
        let mut dci = vk::DeviceCreateInfo::default()
            .queue_create_infos(&qci)
            .enabled_features(&enabled)
            .enabled_extension_names(&enabled_device_extensions)
            .push_next(&mut enabled_vulkan12);
        if null_descriptor {
            dci = dci.push_next(&mut enabled_robustness2);
        }
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
        // The same two capability answers as above, plus the switch. Both halves
        // matter: a queue family that writes no timestamps must get no probe, and
        // a `valid_bits` of 64 has to become an all-ones mask rather than a shift
        // that overflows.
        let valid_bits = qfs[gq as usize].timestamp_valid_bits;
        let draw_spans = (valid_bits > 0
            && props.limits.timestamp_period > 0.0
            && crate::env::read(crate::env::GPU_SPANS).0 != crate::env::Switch::Off)
            .then(|| {
                let ci = vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(DrawSpanProbe::PER_SLOT * super::pools::RING_DEPTH as u32);
                device
                    .create_query_pool(&ci, None)
                    .map(|pool| DrawSpanProbe {
                        pool,
                        ns_per_tick: props.limits.timestamp_period,
                        valid_mask: if valid_bits >= u64::BITS {
                            u64::MAX
                        } else {
                            (1u64 << valid_bits) - 1
                        },
                    })
                    .map_err(|e| {
                        crate::observe::Emit::decline(
                            "vk_draw_span_pool",
                            &VkCall::new(VkOp::ContextCreateQueryPool, e),
                        )
                        .fail_once(0);
                    })
                    .ok()
            })
            .flatten();
        // Gated on the feature actually being enabled, not on the API version.
        // `timelineSemaphore` is core in 1.2 and this backend's baseline is 1.2,
        // so a device that declines it is out of spec — which is exactly why the
        // answer is read from `features` rather than assumed: an assumption here
        // is a `vkWaitSemaphores` into a driver that never implemented it.
        let stamp_completion = features
            .timeline_semaphore
            .then(|| super::stamp_completion::StampCompletion::start(&device))
            .transpose()
            .map_err(|e| {
                crate::observe::Emit::decline(
                    "vk_stamp_completion",
                    &VkCall::new(VkOp::ContextCreateSemaphore, e),
                )
                .fail_once(0);
            })
            .ok()
            .flatten();
        let memory_properties = instance.get_physical_device_memory_properties(pd);
        let caps = HostGpuCaps {
            memory: classify_memory(&memory_properties),
            quirks: DriverQuirk::for_portability_subset(portability_subset),
            host_pointer,
            portability_subset,
            device_api_version: props.api_version,
            device_type: props.device_type,
        };
        // Loaded from the same answer that enabled the extension, so the two
        // cannot disagree about whether these entry points are legal to call.
        let external_memory_host = caps
            .host_pointer
            .is_available()
            .then(|| ash::ext::external_memory_host::Device::new(&instance, &device));
        let device_name = CStr::from_ptr(props.device_name.as_ptr())
            .to_string_lossy()
            .into_owned();
        // One-shot classification line: the memory topology, the signal that
        // decided it, and whether this device can hand a frame to another
        // device without a copy. Load-bearing for portability debugging — "why
        // is this host slow / blank" starts here.
        crate::observe::off(caps.selection_line(&device_name));
        // Beside `vk_caps` rather than inside it: the queue arrangement is a
        // property of the device this engine created, and `HostGpuCaps` is
        // built before any queue family is chosen. `transfer_family=none` is
        // not a degraded reading — it is the arrangement, and it says this
        // boot's copies share the queue its draws are submitted to.
        crate::observe::off(format!(
            "vk_queues families={} graphics_family={gq} compute_capable={compute_capable} transfer_family={}",
            qfs.len(),
            transfer_qf
                .map(|q| q.to_string())
                .unwrap_or_else(|| "none".to_owned()),
        ));
        // Whether this host could sample guest pages in place, which is what
        // decides if the sampled cache is memoizing a copy that has to exist.
        // Reported and never branched on — see the module doc for why a positive
        // answer here is necessary and not sufficient.
        crate::backend::vulkan::caps::linear_sampled::report(&instance, pd);
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
        // Every optional feature and limit this backend resolved, on one line.
        // `vk_device_select` above names the handful a draw's *expressiveness*
        // turns on; this names the whole resolved set, including the ones that
        // came back false. The two are not redundant: a rail declining by name
        // and a rail never asked for look identical in a log that only reports
        // what was enabled.
        crate::observe::off(features.report_line());
        // What the operator set. A boot whose rails were narrowed from outside
        // the process reads as a slow device unless the narrowing is on the
        // same page as the capabilities.
        crate::observe::off(crate::env::report_line());
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
            external_memory_host,
            gq,
            compute_capable,
            storage_image_write_without_format: storage_image_write_without_format_bgra,
            spirv_storage_write_without_format: features.storage_image_write_without_format,
            spirv_storage_read_without_format: features.storage_image_read_without_format,
            spirv_storage_extended_formats: features.storage_image_extended_formats,
            sampled_linear_filter,
            pipeline_cache,
            vertex_divisor,
            guest_bind_offset_align: props
                .limits
                .min_storage_buffer_offset_alignment
                .max(props.limits.min_uniform_buffer_offset_alignment)
                .max(16),
            max_storage_buffer_range: u64::from(props.limits.max_storage_buffer_range),
            vertex_formats,
            max_sampler_anisotropy: features.max_sampler_anisotropy,
            sampler_anisotropy: features.sampler_anisotropy,
            features,
            depth_stencil_format,
            timestamps,
            draw_spans,
            stamp_completion,
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
        // Never write back a blob the next boot would refuse to load. Without
        // this the bound above still self-heals — one cold boot rebuilds a small
        // blob — but every boot in between pays a 30 MB write to produce a file
        // whose only use is to be declined.
        if data.len() > PIPELINE_CACHE_MAX_WARM_BYTES {
            crate::observe::Emit::decline(
                "vk_pipeline_cache_save",
                &PipelineCacheDecline::TooLarge {
                    bytes: data.len(),
                    cap: PIPELINE_CACHE_MAX_WARM_BYTES,
                },
            )
            .fail_once(0);
            return;
        }
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
        // First, and before anything else this function touches: the completion
        // thread holds a clone of `self.device` and is blocked inside it. Every
        // line below is a use-after-free if it is still running.
        if let Some(mut completion) = self.stamp_completion.take() {
            unsafe { completion.stop(&self.device) };
        }
        if let Some(probe) = self.timestamps.take() {
            self.device.destroy_query_pool(probe.pool, None);
        }
        if let Some(probe) = self.draw_spans.take() {
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
    ///
    /// `bytes` is the allocation this pick is for, and every caller has it in the
    /// `VkMemoryRequirements` it just queried. It is what keeps a large
    /// allocation out of a heap that could not hold it — see
    /// [`select_memory_type`].
    pub(crate) fn memory_type_for(
        &self,
        type_bits: u32,
        bytes: u64,
        class: MemoryClass,
    ) -> Option<u32> {
        let picked = self.memory_type_with(type_bits, bytes, &self.caps.memory_request(class));
        // Once per class per boot. What a class *asks* for is in
        // `MemoryTopology::request` and readable from source; what it *gets* is
        // not, because it depends on this device's memory-type table, and the
        // two answers have very different costs. `vk_alloc_sites` prices
        // `MemoryClass::Upload` at 2.54 ms per MiB allocated against 0.48 for
        // `Readback` and 0.018 for the device-local slab, and a difference that
        // size is a difference in which heap the pick landed in. Naming the
        // index and its flags is what turns that from an inference into a
        // reading.
        if let Some(pick) = picked {
            // Keyed on the class and the index together, so a device whose
            // table makes the pick differ between call sites says so instead of
            // latching the first answer for the boot.
            let key = ((class as u64) << 32) | pick.index as u64;
            if crate::observe::first_sight("vk_memory_type_pick", key) {
                let t = self.memory_properties.memory_types[pick.index as usize];
                crate::observe::off(format!(
                    "vk_memory_type_pick class={class:?} index={} heap={} flags={:?} \
                     heap_bytes={} bytes={bytes} fits={}",
                    pick.index, pick.heap_index, t.property_flags, pick.heap_bytes, pick.fits,
                ));
            }
            self.report_oversized_allocation(class, bytes, pick);
        }
        picked.map(|p| p.index)
    }

    /// Escape hatch for a caller that has already built a [`MemoryRequest`]
    /// (the host-pointer import path, which must intersect what
    /// `vkGetMemoryHostPointerPropertiesEXT` named for the pointer).
    pub(crate) fn memory_type_with(
        &self,
        type_bits: u32,
        bytes: u64,
        req: &MemoryRequest,
    ) -> Option<crate::backend::vulkan::caps::memory_topology::MemoryTypePick> {
        select_memory_type(&self.memory_properties, type_bits, req, bytes)
    }

    /// Report, once per (class, heap), an allocation charged to a heap that could
    /// not hold it even empty.
    ///
    /// Fail-visible rather than off-channel, and it is not a loss of guest work:
    /// the allocation is still attempted and usually succeeds. What it says is
    /// that this device has asked its driver to keep something resident in a pool
    /// with no room for it, which is the condition under which a driver's
    /// residency manager evicts the rest of the working set on every submission.
    /// That reads from the outside as "the whole machine got slow", and until
    /// this line existed there was nothing in a boot's log that named it.
    ///
    /// [`MemoryTypePick::fits`] is a heap-capacity test and not a residency one,
    /// so this is a lower bound: a heap large enough to hold the allocation can
    /// still be too full to. The direction it does catch is unambiguous.
    pub(crate) fn report_oversized_allocation(
        &self,
        class: MemoryClass,
        bytes: u64,
        pick: crate::backend::vulkan::caps::memory_topology::MemoryTypePick,
    ) {
        if pick.fits {
            return;
        }
        let key = ((class as u64) << 32) | pick.heap_index as u64;
        if !crate::observe::first_sight("vk_memory_heap_too_small", key) {
            return;
        }
        crate::observe::fail(format!(
            "vk_memory_heap_too_small reason=vk_memory_heap_too_small class={class:?} \
             index={} heap={} heap_mb={} bytes_mb={} (the allocation is charged to a heap that \
             could not hold it empty; expect driver-side eviction of the working set)",
            pick.index,
            pick.heap_index,
            pick.heap_bytes >> 20,
            bytes >> 20,
        ));
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

/// The index of a queue family that transfers and does nothing else — a copy
/// engine that runs beside the graphics one rather than through it.
///
/// `TRANSFER` alone is the whole test. `GRAPHICS` and `COMPUTE` both imply
/// transfer support, so a family carrying either is one this device already
/// submits draws to, and moving a copy there would buy nothing. Sparse binding,
/// video decode/encode and optical flow do **not** disqualify a family: they say
/// what else that hardware block can do, not that it shares the graphics engine.
///
/// `None` where the host has no such family, which is most integrated parts.
/// That is the arrangement rather than a degraded one, and the caller keeps
/// every copy on the graphics queue.
///
/// # What a boot found, and what it is worth
///
/// This was added without ever being read on a live device. It has been now, on
/// the x86/Vulkan pathway against an RTX 5080 Laptop:
///
/// ```text
/// vk_queues families=6 graphics_family=0 compute_capable=true transfer_family=1
/// ```
///
/// So the copy engine is there, and every byte this device moves is still going
/// to family 0 with the draws. The size of that is measurable rather than
/// arguable, because the guest-page writeback carries its own GPU timestamps: a
/// driven Safari-drag second reports `gpu_us=167437` over `gpu=836` copies —
/// **167 ms of GPU time per second at ~200 us a copy**, which for a 3.33 MB
/// copy is a healthy ~16 GB/s and not a slow rail. Scaling the buffer gather by
/// its share of the bytes (2.74 GB/s against the writeback's 5.19) puts total
/// copy occupancy near 255 ms/s.
///
/// In the same second `draw_phase`'s `slot_us` is 245 ms/s — the drain worker
/// blocked in `begin_entry` on a ring slot whose fence the GPU has not signalled.
/// Those two numbers being within 5 % of each other is the reason to look here:
/// the CPU's wait for the GPU is about the size of the GPU's copy work, and that
/// work is serialised against the rendering only because it shares a queue.
///
/// # An ablation says the whole ceiling is this
///
/// The correspondence above is not a proof, so it was tested directly: a probe
/// boot recorded the writeback's barriers, batch flush, stamp and every CPU-side
/// bookkeeping step exactly as normal, and skipped only the
/// `cmd_copy_image_to_buffer`/scatter commands themselves — the GPU work, and
/// nothing else. The guest loses its frames that way, so it is an ablation and
/// never a shipping arm; it is recorded here because of what it measured.
///
/// | | shipping | writeback GPU work removed |
/// |---|---|---|
/// | `present_hz` median | 72.7-76.4 | **104.0** |
/// | seconds below 100 Hz | 24/24 | **4/25** |
/// | `slot_us` | 245 750 us/s | **3 986 us/s** |
/// | `drain_duty` `duty` | 0.81 | 0.59 |
/// | `draw_us/draw` | 132-139 us | 78 us |
/// | draws | 4 383-4 800/s | 5 916-6 407/s |
///
/// `slot_us` falls by a factor of 62. It was not ring depth, not submission
/// overhead and not jitter: it was this device's own copies sitting in the queue
/// ahead of the draws whose slots it was waiting for. Every earlier attempt on
/// `slot_us` moved a number that was downstream of this one, which is why
/// halving the submissions once bought no frames at all.
///
/// # The prize is not here, and a built rail measured that
///
/// It reads from the table above as if moving the copies off this queue were
/// worth the gap between 76 Hz and 104 Hz. It is not, and the way to find that
/// out was to build it: a second queue, a ring of transfer command buffers, two
/// timeline semaphores, and the writeback's scatter submitted to the copy
/// engine instead of appended to the draw batch. It ran, on the x86/Vulkan
/// pathway against the same host, with `vk_queues transfer_family=1`.
///
/// The split was at the **scratch buffer**, not at the image, and that part of
/// the design was right and stays recorded because it is the cheap answer to the
/// ownership problem below. The detile (`vkCmdCopyImageToBuffer` into the
/// device-local scratch) stayed on `gq`, so the render target never left its
/// family and never gave up its lossless framebuffer compression. Only the
/// scatter — `vkCmdCopyBuffer` out of the scratch into imported guest pages —
/// crossed, and the only resources both queues saw were buffers, which are free
/// to share `CONCURRENT`.
///
/// What four driven Safari-drag boots measured, against a 67.8 Hz baseline taken
/// on the same tree and machine that hour:
///
/// | arrangement | `present_hz` med | `slot_us` | CPU wait on the copy engine |
/// |---|---|---|---|
/// | shipping — every copy on `gq` | 67.8 | 265 000 us/s | — |
/// | scatter on the copy engine, 4-deep ring | 67.4 | **8 000 us/s** | 240 000 us/s |
/// | same, 16-deep | 69.8 | 290 000 us/s | ~0 |
/// | same, 64-deep | 69.0 | 230 000 us/s | ~0 |
///
/// `slot_us` really does collapse — by 33x, close to what the ablation
/// predicted. And it buys nothing, because **the block is conserved**. At depth
/// 4 the drain worker stops waiting for a ring slot and starts waiting for a
/// transfer command buffer instead, for the same 240 ms a second. Deepening the
/// ring removes that wait and the block reappears a third time, as the graphics
/// submission's own write-after-read wait for the scratch buffer it is about to
/// overwrite. Three arrangements, three different counters, one number.
///
/// # Because the wall is the bus, and every queue shares it
///
/// A narrower ablation says so directly. Skipping only the image read, with the
/// scatter still running and still moving its bytes, gives **72.9 Hz** — four
/// Hertz, not thirty. The earlier ablation reached 104 Hz because it removed the
/// scatter too, and with it the bus traffic. A copy engine moves those same
/// bytes over that same link.
///
/// The traffic is the finding: ~1 500 guest-page writebacks a second at ~3.34 MB
/// each is **~5.0 GB/s into guest RAM**, sustained, at ~70 displayed frames a
/// second. That is about **21 full-surface writebacks per frame the user sees**,
/// spread over roughly six surfaces — and it is split across two rails, the
/// render Store at ~613/s (`readback_split`'s `vouch`) and the GVA Store making
/// up the rest of `guest_write_linear`.
///
/// So the route to 120 Hz is fewer bytes crossing, and nothing about which
/// engine carries them. Do not rebuild this rail to chase frames. It is worth
/// rebuilding only *after* the byte volume comes down, when a decoupled
/// `slot_us` would have something left to convert into frames — and the shape it
/// should take is the one above.
///
/// A copy engine is still not free, and anything built here has to answer three
/// costs the shared queue does not pay: a cross-queue dependency needs a
/// semaphore rather than a pipeline barrier, an image written by one family and
/// read by another needs an ownership transfer or `CONCURRENT` sharing, and
/// splitting a copy out of the batch it is currently appended to restores the
/// second submission that appending it removed. Splitting at the scratch buffer
/// answers the middle one for nothing, which is why that is where it belongs.
fn dedicated_transfer_family(families: &[vk::QueueFamilyProperties]) -> Option<u32> {
    families
        .iter()
        .position(|q| {
            q.queue_flags.contains(vk::QueueFlags::TRANSFER)
                && !q
                    .queue_flags
                    .intersects(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
        })
        .map(|i| i as u32)
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
                    self.note_init_failure(&e);
                    return Err(e);
                }
            }
        }
        Ok(self.ctx.as_ref().unwrap())
    }

    /// Decide whether a failed bring-up becomes the permanent answer.
    ///
    /// [`Self::ensure`] consults `init_error` before it consults anything else,
    /// and nothing clears it, so whatever lands here answers every later draw
    /// for the life of the process. That is right for the verdicts bring-up
    /// reaches about the machine — no Vulkan loader, no physical device, none
    /// above the API floor, no graphics queue — because a second attempt walks
    /// the same host and reaches the same one.
    ///
    /// It is wrong for out of memory. `vkCreateInstance` and `vkCreateDevice`
    /// both refuse with `ERROR_OUT_OF_HOST_MEMORY`, which describes what the
    /// host had free at that instant; latching it would answer a draw a minute
    /// later, after whatever was holding the memory had exited, from a machine
    /// state that no longer exists. So it is returned to this caller — the
    /// guest sees the refusal — and forgotten, and the next draw asks the
    /// loader again.
    ///
    /// Split out from `ensure` so the rule is reachable without a Vulkan
    /// device: it is the one part of bring-up that is a decision rather than a
    /// driver call.
    fn note_init_failure(&mut self, error: &DrawError) {
        if error.out_of_memory() {
            return;
        }
        self.init_error = Some(error.clone());
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

    /// Guest work reached the GPU on the current device.
    ///
    /// This is the only evidence that a recreate *worked*, and it is what turns
    /// [`MAX_DEVICE_RECREATES`] from a lifetime allowance into a bound on a
    /// storm. A device that was rebuilt and then executed a draw has recovered;
    /// if it is lost again an hour later that is a new incident, not the fourth
    /// step of the old one, and refusing it because of three resets the guest
    /// never noticed is this device deciding to stop being a GPU.
    ///
    /// One completed submission is the whole threshold, and that is on purpose —
    /// any larger number would be tuned rather than derived. It leaves the
    /// pathological case of a host that loses the device after every single
    /// draw recreating indefinitely, which is the right trade: that guest sees
    /// slow progress instead of a display that never comes back. The churn stays
    /// measurable either way — `EngineCounters::recreates` and `device_lost` are
    /// both in the `cumulative` group, so they count every incident across the
    /// boot and this reset does not touch them.
    ///
    /// Called on the success arm of every draw and dispatch, so it is on the hot
    /// path — hence the branch, which is false on every call of a healthy boot.
    pub(crate) fn note_work_completed(&mut self) {
        if self.recreate_count == 0 {
            return;
        }
        crate::observe::fail(format!(
            "vk_device_recreate_proven recreates={} (guest work ran on the rebuilt device; \
             the storm budget starts over)",
            self.recreate_count
        ));
        self.recreate_count = 0;
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

    /// A blob past [`PIPELINE_CACHE_MAX_WARM_BYTES`] is refused as a cold start,
    /// and refused for being oversized rather than for being malformed — the two
    /// have opposite meanings for whoever reads the boot, because an oversized
    /// blob is this device's own well-formed output and a rebuilt one is the
    /// expected next state.
    ///
    /// Written against a blob that is otherwise perfectly loadable, so it fails
    /// if the size gate is ever moved after the compatibility gate.
    #[test]
    fn an_oversized_pipeline_cache_blob_is_declined_by_size_not_by_shape() {
        use crate::observe::Decline as _;
        let root = std::env::temp_dir().join(format!(
            "reims-vgpu-pcap-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let props = vk::PhysicalDeviceProperties::default();

        // A header this device would otherwise accept, padded past the cap.
        let mut blob = Vec::with_capacity(PIPELINE_CACHE_MAX_WARM_BYTES + 1);
        blob.extend_from_slice(&(PIPELINE_CACHE_HEADER_ONE_LEN as u32).to_le_bytes());
        blob.extend_from_slice(
            &(vk::PipelineCacheHeaderVersion::ONE.as_raw() as u32).to_le_bytes(),
        );
        blob.extend_from_slice(&props.vendor_id.to_le_bytes());
        blob.extend_from_slice(&props.device_id.to_le_bytes());
        blob.extend_from_slice(&props.pipeline_cache_uuid);
        assert!(
            pipeline_cache_blob_compatible(&blob, &props),
            "the fixture must be loadable before it is padded, or this proves nothing"
        );
        blob.resize(PIPELINE_CACHE_MAX_WARM_BYTES + 1, 0);

        let path = root.join("oversized.bin");
        std::fs::write(&path, &blob).unwrap();
        let decline = read_pipeline_cache_blob(&path, &props).unwrap_err();
        assert_eq!(decline.slug(), "vk_pipeline_cache_too_large");
        assert_eq!(
            decline.fields(),
            vec![
                ("bytes", (PIPELINE_CACHE_MAX_WARM_BYTES + 1).to_string()),
                ("cap", PIPELINE_CACHE_MAX_WARM_BYTES.to_string()),
            ]
        );

        // One byte under the cap is still a warm start, so the bound is a bound
        // and not a ban on warming.
        blob.truncate(PIPELINE_CACHE_MAX_WARM_BYTES);
        std::fs::write(&path, &blob).unwrap();
        assert_eq!(
            read_pipeline_cache_blob(&path, &props).unwrap(),
            Some(blob),
            "a blob at exactly the cap warms"
        );
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

#[cfg(test)]
mod transfer_family_tests {
    use super::*;

    fn family(flags: vk::QueueFlags) -> vk::QueueFamilyProperties {
        vk::QueueFamilyProperties::default()
            .queue_flags(flags)
            .queue_count(1)
    }

    /// A family that also draws or dispatches is one this device already
    /// submits to, so picking it would move nothing off the graphics engine
    /// while adding an ownership transfer to every copy. Both bits disqualify,
    /// and `TRANSFER` is often not even spelled on a graphics family — the spec
    /// makes it implicit — so the test is over families that name it and
    /// families that do not.
    #[test]
    fn a_family_that_also_draws_or_dispatches_is_not_a_copy_engine() {
        use vk::QueueFlags as F;
        for flags in [
            F::GRAPHICS,
            F::COMPUTE,
            F::GRAPHICS | F::COMPUTE,
            F::GRAPHICS | F::TRANSFER,
            F::COMPUTE | F::TRANSFER,
            F::GRAPHICS | F::COMPUTE | F::TRANSFER | F::SPARSE_BINDING,
        ] {
            assert_eq!(
                dedicated_transfer_family(&[family(flags)]),
                None,
                "{flags:?} shares the engine this device already submits to"
            );
        }
    }

    /// The bits that say what *else* a copy engine can do must not disqualify
    /// it. A discrete part commonly exposes several transfer-only families that
    /// differ exactly in these, and refusing them would leave a host with a copy
    /// engine reading as a host without one.
    #[test]
    fn the_other_bits_on_a_transfer_only_family_do_not_disqualify_it() {
        use vk::QueueFlags as F;
        for extra in [
            F::empty(),
            F::SPARSE_BINDING,
            F::VIDEO_DECODE_KHR,
            F::VIDEO_ENCODE_KHR,
            F::OPTICAL_FLOW_NV,
        ] {
            assert_eq!(
                dedicated_transfer_family(&[family(F::TRANSFER | extra)]),
                Some(0),
                "TRANSFER | {extra:?} is still a copy engine"
            );
        }
    }

    /// A family with no transfer bit at all is not one, and a host that offers
    /// none answers `None` rather than falling to index zero — which would
    /// submit copies to the graphics family under a name that says otherwise.
    #[test]
    fn a_host_with_no_copy_engine_answers_none_rather_than_the_first_family() {
        use vk::QueueFlags as F;
        assert_eq!(dedicated_transfer_family(&[]), None);
        assert_eq!(
            dedicated_transfer_family(&[family(F::GRAPHICS | F::COMPUTE | F::TRANSFER)]),
            None,
            "the single-family host every integrated part presents"
        );
        assert_eq!(
            dedicated_transfer_family(&[
                family(F::GRAPHICS | F::COMPUTE | F::TRANSFER),
                family(F::TRANSFER | F::SPARSE_BINDING),
            ]),
            Some(1),
            "the second family is the copy engine and the index must be its own"
        );
    }
}

#[cfg(test)]
mod init_latch_tests {
    use super::*;

    /// Bring-up refusing for want of memory must not become the permanent
    /// answer. `ensure` reads `init_error` before it reads anything else and
    /// nothing clears it, so a latched one would answer every draw for the life
    /// of the process from a host state that has since changed.
    #[test]
    fn an_out_of_memory_bring_up_is_not_latched() {
        for decline in [
            InitDecline::CreateInstance {
                result: vk::Result::ERROR_OUT_OF_HOST_MEMORY,
            },
            InitDecline::CreateDevice {
                result: vk::Result::ERROR_OUT_OF_HOST_MEMORY,
            },
            InitDecline::CreateDevice {
                result: vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
            },
        ] {
            let mut owner = ContextOwner::new();
            let error = DrawError::Init(decline.clone());
            assert!(error.out_of_memory(), "{decline:?} is the retryable class");
            owner.note_init_failure(&error);
            assert!(
                owner.init_error.is_none(),
                "{decline:?} must leave the next draw free to ask the loader again"
            );
        }
    }

    /// The converse, so the test above cannot pass by never latching at all. A
    /// verdict about the machine is reached identically by the next attempt, so
    /// it is worth keeping — and re-walking bring-up on every draw to rediscover
    /// that the host has no Vulkan at all is the cost the latch exists to avoid.
    #[test]
    fn a_verdict_about_the_machine_is_still_latched() {
        for decline in [
            InitDecline::NoPhysicalDevice,
            InitDecline::NoGraphicsQueueFamily,
            InitDecline::LoadVulkanLoader {
                detail: "no loader".to_string(),
            },
            InitDecline::BelowApiFloor {
                minimum: vk::API_VERSION_1_2,
                found: vec![vk::API_VERSION_1_0],
            },
            InitDecline::CreateDevice {
                result: vk::Result::ERROR_EXTENSION_NOT_PRESENT,
            },
        ] {
            let mut owner = ContextOwner::new();
            let error = DrawError::Init(decline.clone());
            assert!(!error.out_of_memory(), "{decline:?} is not about memory");
            owner.note_init_failure(&error);
            assert_eq!(
                owner.init_error.as_ref(),
                Some(&error),
                "{decline:?} is reached again by the next attempt, so it is kept"
            );
        }
    }

    /// `out_of_memory` had to learn to look inside `Init` for any of this to
    /// work; before it did, every arm above answered `false` and bring-up
    /// latched unconditionally. Pin that it reads the result the variant
    /// carries rather than the variant's identity.
    #[test]
    fn out_of_memory_reads_the_result_not_the_variant() {
        let oom = DrawError::Init(InitDecline::CreatePipelineCache {
            result: vk::Result::ERROR_OUT_OF_HOST_MEMORY,
        });
        let same_variant_other_result = DrawError::Init(InitDecline::CreatePipelineCache {
            result: vk::Result::ERROR_INITIALIZATION_FAILED,
        });
        assert!(oom.out_of_memory());
        assert!(!same_variant_other_result.out_of_memory());

        // A check this device decides itself carries no result at all.
        assert_eq!(InitDecline::NoPhysicalDevice.vk_result(), None);
        assert_eq!(
            InitDecline::CreateDevice {
                result: vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
            }
            .vk_result(),
            Some(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)
        );
    }
}

#[cfg(test)]
mod recreate_budget_tests {
    use super::*;

    /// A storm — lost, rebuilt, lost again with nothing run in between — still
    /// terminates. This is the case the budget exists for, and `ensure` answers
    /// it without touching Vulkan, which is why the test runs on any host.
    #[test]
    fn a_storm_that_produces_no_work_still_exhausts_the_budget() {
        let mut owner = ContextOwner::new();
        owner.poisoned = true;
        owner.recreate_count = MAX_DEVICE_RECREATES;
        let counters = EngineCounters::default();

        match owner.ensure(&counters) {
            Ok(_) => panic!("an exhausted storm budget must refuse"),
            Err(error) => assert_eq!(
                error,
                DrawError::DeviceLost(DeviceLostDecline::RecreateCapExhausted {
                    cap: MAX_DEVICE_RECREATES,
                })
            ),
        }
    }

    /// A recreate followed by guest work is a recreate that *worked*, and must
    /// not be held against the next incident.
    ///
    /// Before this, `recreate_count` only ever rose. Three device losses over a
    /// week of uptime — each recovered, each with hours of correct rendering
    /// after it — left the fourth refused and every draw after that refused with
    /// it, for as long as the guest ran.
    #[test]
    fn work_on_the_rebuilt_device_clears_the_storm_budget() {
        let mut owner = ContextOwner::new();
        owner.recreate_count = MAX_DEVICE_RECREATES;

        owner.note_work_completed();

        assert_eq!(
            owner.recreate_count, 0,
            "a recreate the guest drew through must not count against the next storm"
        );

        // And the budget is genuinely available again: at the cap this refuses
        // without touching Vulkan, so reaching past that branch is the proof.
        owner.poisoned = true;
        let counters = EngineCounters::default();
        let refusal = owner.ensure(&counters).err();
        assert_ne!(
            refusal,
            Some(DrawError::DeviceLost(
                DeviceLostDecline::RecreateCapExhausted {
                    cap: MAX_DEVICE_RECREATES,
                }
            )),
            "the cap branch must no longer be taken"
        );
    }

    /// The healthy path costs a branch and nothing else. Every draw and dispatch
    /// calls this, so an emission here would be one fail-log line per draw.
    #[test]
    fn a_device_that_was_never_lost_is_left_alone() {
        let mut owner = ContextOwner::new();
        assert_eq!(owner.recreate_count, 0);
        owner.note_work_completed();
        assert_eq!(owner.recreate_count, 0);
        assert!(!owner.poisoned, "noting work must not change device state");
        assert!(owner.init_error.is_none());
    }
}

#[cfg(test)]
mod draw_span_probe_tests {
    use super::*;

    fn probe(valid_bits: u32, ns_per_tick: f32) -> DrawSpanProbe {
        DrawSpanProbe {
            pool: vk::QueryPool::null(),
            ns_per_tick,
            valid_mask: if valid_bits >= u64::BITS {
                u64::MAX
            } else {
                (1u64 << valid_bits) - 1
            },
        }
    }

    /// The ordinary case: a full-width counter, a tick that is not one
    /// nanosecond, and a delta that does not wrap.
    #[test]
    fn a_full_width_counter_scales_its_delta_by_the_tick() {
        let p = probe(64, 2.5);
        assert_eq!(p.elapsed_ns(1_000, 1_400), 1_000);
    }

    /// A queue family that writes 32 meaningful bits leaves the rest
    /// **undefined**, so the high halves of the two results may differ by
    /// anything. Masking both operands is what makes the subtraction the elapsed
    /// ticks rather than a difference of two drivers' scratch bits — and this is
    /// the case a raw `bottom - top` reports as a span of years.
    #[test]
    fn undefined_high_bits_do_not_reach_the_answer() {
        let p = probe(32, 1.0);
        let top = 0xdead_beef_0000_0100u64;
        let bottom = 0x1234_5678_0000_0300u64;
        assert_eq!(p.elapsed_ns(top, bottom), 0x200);
    }

    /// 32 valid bits at a one-nanosecond tick wraps every 4.3 seconds, which a
    /// per-second census crosses several times a boot. A wrap is a wrap and not a
    /// negative: without the mask on the result this reads as ~18 000 000 000 µs
    /// and would own every column it is quoted beside.
    #[test]
    fn a_wrap_within_the_valid_width_is_a_wrap() {
        let p = probe(32, 1.0);
        assert_eq!(p.elapsed_ns(0xffff_ff00, 0x0000_00ff), 0x1ff);
    }

    /// Each ring slot owns a disjoint pair, because the slot's fence is what makes
    /// its pair readable and two slots are in flight at once.
    #[test]
    fn every_ring_slot_gets_its_own_disjoint_pair() {
        let bases: Vec<u32> = (0..super::super::pools::RING_DEPTH)
            .map(DrawSpanProbe::base)
            .collect();
        for w in bases.windows(2) {
            assert_eq!(
                w[1] - w[0],
                DrawSpanProbe::PER_SLOT,
                "slot bases must tile the pool: {bases:?}"
            );
        }
        let last = bases.last().expect("the ring is not empty");
        assert_eq!(
            last + DrawSpanProbe::PER_SLOT,
            DrawSpanProbe::PER_SLOT * super::super::pools::RING_DEPTH as u32,
            "the pool is exactly as large as the ring needs"
        );
    }
}
