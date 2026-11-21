//! The rt platform's device ([`RtDevice`]): one metal2vulkan-runtime device for the process, made
//! of the physical device the library's choice of device picks ([`open`]), with its answers
//! (`super::answers`), its command queues (`super::queue`), its buffers (`super::buffer`), heaps
//! (`super::heap`) and the blocks it knows ([`KnownBlocks`]), its textures (`super::texture`),
//! events and fences, and the reservation of resource indices as rt's index grants. Every other
//! creator is refused while its piece is unbuilt (`super::unbuilt`), as Metal's creator answers
//! where it fails.
//!
//! **One device.** The process makes one rt device, the first time a device is asked for, and
//! every ask answers it (one `VkDevice` per VM, the host process), until the release of the last
//! graphics device lets go of it (`RtPlatform::let_go_of_device`); the next ask makes a new one.
//! A failure to make it is counted once as rt's refusal and again at each later ask, which
//! answers nil.
//!
//! **The choice.** The candidates are the physical devices that meet rt's floor, which rt lists in
//! the loader's order and picks none of (`Instance::physical_devices`, read from code). The pick
//! among them is the library's (`crate::device_choice::chosen_untimed`): the device the host's
//! user's selection variables name, else the one GPU, else the first by kind: discrete GPUs, then
//! integrated, virtual and CPU ones, each kind in the loader's order. It times no GPU, so a weak
//! discrete GPU beside a strong integrated one is picked wrongly: an RX 550 beside a Radeon 780M,
//! which composites a 3840x2400 frame in 24.4 ms to the RX 550's 40.4 ms (**measured**,
//! 2026-11-08, Mesa 26.2.6, by reims-vgpu's `composite_cost`). There the host's user's selection
//! variable is the answer, as the code reads it (`crate::device_choice::SELECTION_VARIABLES`):
//! `MESA_VK_DEVICE_SELECT` (`MESA_VK_DEVICE_SELECT=1002:15bf` names that 780M by its vendor and
//! device IDs), `DRI_PRIME` or `VK_LOADER_DEVICE_SELECT`. The library's rule that a candidate
//! lists the device extensions a display's window takes holds here: the window takes
//! `VK_KHR_swapchain` alone, which rt's floor takes, so it passes over no device that meets the
//! floor.
//!
//! **Its window system.** The instance enables the OS's surface extensions and present fences'
//! where the loader lists them, and the device the extension of present fences where it lists
//! it and its instance extensions are on (`crate::os`'s `rt_instance_extensions` and
//! `rt_present_fences`); `VK_KHR_surface` and `VK_KHR_swapchain` are rt's floor's. A display's
//! window presents through a swapchain on the device (`super::swapchain`), which keeps the
//! instance, and the device's DRM nodes tell its window whether the compositor composites on
//! this GPU ([`RtDevice::composited_here`]).
//!
//! **Its registry ID** folds its `deviceUUID` (`super::answers::registry_id`).
//!
//! **Its over-copies.** Where the device's linear image cannot hold a texture over a buffer in
//! place, rt keeps the texels in an image of its own and copies the bytes between the two,
//! counting each copy and its bytes (rt's `Metrics`). The process's counts line carries them as
//! amounts ([`fold_metrics`], which `crate::answer::write_counts` runs first):
//! [`OVER_COPIES_CODE`] and [`OVER_COPY_BYTES_CODE`]; of those over an imported block,
//! [`OVER_COPIES_IMPORTED_CODE`] and [`OVER_COPY_BYTES_IMPORTED_CODE`]; and by the cause rt gives
//! each copy, `rt-over-copies-<cause>` and `rt-over-copy-bytes-<cause>` ([`over_copy_counts`]).
//! rt counts each copy under one cause, so the causes sum to the over-copies; a copy over an
//! imported block is counted under its import's cause too, so the imported ones are an axis of
//! their own and no cause.

use std::sync::{Arc, OnceLock};

use ash::vk;
use metal2vulkan_runtime as rt;

use crate::core::platform::caps::{DeviceCaps, IdentityCaps, SAMPLE_COUNTS};
use crate::core::platform::device::{GpuDevice, NoCopyDeallocator};
use crate::core::platform::host_range::HostRange;
use crate::core::platform::stage::StageFills;
use crate::core::platform::types::{
    ArgumentLayout, NsErrorText, PixelFormatInfo, SizeAndAlign, TextureLevelInfo, TextureLevelQuery,
};
use crate::core::platform::values::{
    ArgumentDesc, BufferDesc, DepthStencilDesc, FullComputePipelineDesc,
    FullMeshRenderPipelineDesc, FullRenderPipelineDesc, HeapDesc, HeapMade, IcbDesc,
    RenderPipelineDesc, SamplerDesc,
};
use crate::core::platform::{Handle, Platform};
use crate::core::raise::{OnFault, Raise, RaiseResult};
use crate::gpu::caps::METAL_LINEAR_ALIGNMENT;
use crate::gpu::device::{HEAP_DESCRIPTOR_OPTIONS, descriptor_options};
use crate::gpu::identity;

use super::answers::{self, Facts};
use super::argument;
use super::buffer::{self, KnownBlocks};
use super::command::DeviceLoss;
use super::heap;
use super::library::{self, RtLibrary};
use super::pipeline::{self, Numbering};
use super::queue::RtQueue;
use super::render_pipeline;
use super::stage::StageKey;
use super::texture::OwnIndices;
use super::unbuilt::{
    RtRefusal, Unbuilt, refuse, refuse_create, refused_as, refused_error, rt_refused,
    rt_refused_create, rt_refused_texture,
};
use super::{RtObj, RtPlatform};

/// The log target of the device's lines.
const TARGET: &str = "reims_vgpu_vulkan::rt_platform";

/// The code of rt's copies between a texture over a buffer and its image of its own (rt's
/// `Metrics::over_copies`), an amount on the counts line ([`fold_metrics`]).
pub const OVER_COPIES_CODE: &str = "rt-over-copies";

/// The code of those copies' bytes (`Metrics::over_copy_bytes`).
pub const OVER_COPY_BYTES_CODE: &str = "rt-over-copy-bytes";

/// The code of those copies over a buffer in an imported block (`Metrics::over_copies_imported`).
pub const OVER_COPIES_IMPORTED_CODE: &str = "rt-over-copies-imported";

/// The code of their bytes (`Metrics::over_copy_bytes_imported`).
pub const OVER_COPY_BYTES_IMPORTED_CODE: &str = "rt-over-copy-bytes-imported";

/// The over-copy metrics the counts line and the trace's `rtcounts` carry ([`over_copy_counts`]):
/// the over-copies and their bytes, those over an imported block, and the 12 causes' pairs.
pub(super) const OVER_COPY_METRICS: usize = 4 + 2 * 12;

/// The counters of the device the process made last ([`open`]), which [`fold_metrics`] reads,
/// with what it folded so far of each of their metrics, in [`over_copy_counts`]'s order.
type Folding = Option<(rt::Counters, [u64; OVER_COPY_METRICS])>;

/// The process's [`Folding`], once a device is made.
static COUNTERS: crate::core::sync::Mutex<Folding> = crate::core::sync::Mutex::new(None);

/// Counts by code (`crate::answer::count_kind_by`) what rt's over-copy metrics of the process's
/// device rose by since the last fold, so the counts line carries their totals (the module's
/// docs). Nothing before the device is made.
pub fn fold_metrics() {
    if let Some((counters, folded)) = COUNTERS.lock().as_mut() {
        fold(folded, over_copy_counts(&counters.metrics()));
    }
}

/// Makes `counters`, those of a device just made, the ones [`fold_metrics`] reads. What the
/// device before it rose by since the last fold is folded first, so that the counts line keeps
/// its totals after a let-go (`RtPlatform::let_go_of_device`) and adds the new device's.
fn follow(counters: rt::Counters) {
    let mut latest = COUNTERS.lock();
    if let Some((before, folded)) = latest.as_mut() {
        fold(folded, over_copy_counts(&before.metrics()));
    }
    *latest = Some((counters, [0; OVER_COPY_METRICS]));
}

/// The over-copy metrics of `m`, each with its code (the module's docs): the over-copies and
/// their bytes, those over an imported block, then each cause's copies and their bytes, the code
/// named for rt's field.
pub(super) fn over_copy_counts(m: &rt::Metrics) -> [(&'static str, u64); OVER_COPY_METRICS] {
    macro_rules! by_cause {
        ($($cause:literal $copies:ident $bytes:ident),* $(,)?) => {
            [
                (OVER_COPIES_CODE, m.over_copies),
                (OVER_COPY_BYTES_CODE, m.over_copy_bytes),
                (OVER_COPIES_IMPORTED_CODE, m.over_copies_imported),
                (OVER_COPY_BYTES_IMPORTED_CODE, m.over_copy_bytes_imported),
                $(
                    (concat!("rt-over-copies-", $cause), m.$copies),
                    (concat!("rt-over-copy-bytes-", $cause), m.$bytes),
                )*
            ]
        };
    }
    by_cause!(
        "host-memory" over_copies_host_memory over_copy_bytes_host_memory,
        "dma-buf" over_copies_dma_buf over_copy_bytes_dma_buf,
        "dma-bufs" over_copies_dma_bufs over_copy_bytes_dma_bufs,
        "dedicated" over_copies_dedicated over_copy_bytes_dedicated,
        "surface" over_copies_surface over_copy_bytes_surface,
        "spilled" over_copies_spilled over_copy_bytes_spilled,
        "linear" over_copies_linear over_copy_bytes_linear,
        "linear-features" over_copies_linear_features over_copy_bytes_linear_features,
        "pitch" over_copies_pitch over_copy_bytes_pitch,
        "render-reach" over_copies_render_reach over_copy_bytes_render_reach,
        "alignment" over_copies_alignment over_copy_bytes_alignment,
        "image-memory" over_copies_image_memory over_copy_bytes_image_memory,
    )
}

/// rt's metrics of the process's device now (`super::diag`); `None` before it is made.
pub(super) fn metrics() -> Option<rt::Metrics> {
    (COUNTERS.lock().as_ref()).map(|(counters, _)| counters.metrics())
}

/// rt's gauges of the process's device now (`super::diag`); `None` before it is made.
pub(super) fn gauges() -> Option<rt::Gauges> {
    (COUNTERS.lock().as_ref()).map(|(counters, _)| counters.gauges())
}

/// Counts the rise of each over-copy metric of `now` ([`over_copy_counts`]) over `folded` under
/// its code, and keeps `now` as folded.
fn fold(folded: &mut [u64; OVER_COPY_METRICS], now: [(&'static str, u64); OVER_COPY_METRICS]) {
    for ((code, now), was) in now.into_iter().zip(folded.iter_mut()) {
        crate::answer::count_kind_by(code, now.saturating_sub(*was));
        *was = now;
    }
}

/// Metal's `maxCommandBufferCount` of a queue made by `newCommandQueue`.
pub const METAL_MAX_COMMAND_BUFFER_COUNT: u32 = 64;

/// The rt device's cache directory in the user's cache directory, beside the library's own
/// (`crate::os::cache_directory`), which it shares nothing with: rt keeps its lowered functions
/// and its pipeline caches there, within [`SHADER_CACHE_CAPACITY`].
#[cfg(not(any(test, feature = "testkit")))]
const CACHE: &str = "reims-vgpu-rt";

/// A test build's rt cache directory, the tests' own: a test of what a device compiles runs on an
/// empty one as on a full one, and the tests write nothing into the one the library's hosts share.
#[cfg(any(test, feature = "testkit"))]
const CACHE: &str = "reims-vgpu-rt-tests";

/// The most bytes [`CACHE`] holds, 25 GiB. A macOS 26 guest's built-in apps, each launched once,
/// need more than 1 GiB of translations and driver records on an RTX 5080 laptop. An entry dropped
/// past the capacity is compiled again at the next boot that draws with it.
#[cfg(not(any(test, feature = "testkit")))]
pub(crate) const SHADER_CACHE_CAPACITY: u64 = 25 << 30;

/// A test build's capacity, 1 GiB: its directory holds what the tests compile and nothing a
/// guest's boot reuses, so it is kept small on disk.
#[cfg(any(test, feature = "testkit"))]
pub(crate) const SHADER_CACHE_CAPACITY: u64 = 1 << 30;

/// `id<MTLDevice>` on metal2vulkan-runtime: rt's device and what the platform read of it.
pub struct RtDevice {
    /// rt's device.
    rt: rt::Device,
    /// What its answers depend on.
    facts: Facts,
    /// Its capabilities, read as it was made.
    caps: Arc<DeviceCaps>,
    /// The texture indices its textures the guest does not place take (`super::texture`).
    indices: OwnIndices,
    /// The sampler indices its sampler states the guest does not place take (`super::state`), and
    /// its pipeline states' samplers (`super::pipeline`).
    sampler_indices: OwnIndices,
    /// The sampler indices its pipeline states' samplers hold.
    numbering: Numbering,
    /// The platform's shaders as its library, read at the first ask (`super::library`).
    shaders: OnceLock<RtObj<RtLibrary>>,
    /// The blocks it knows by address, which pinned buffers and heaps inside them are ranges of.
    known: KnownBlocks,
    /// The instance it was made on, whose surfaces its windows' swapchains present to.
    instance: rt::Instance,
    /// Its DRM nodes, where it lists `VK_EXT_physical_device_drm`.
    drm: Option<crate::device_choice::DrmNodes>,
    /// Whether it enabled present fences' extension (the module's docs).
    present_fences: bool,
    /// Where its GPU lies on the host.
    host_gpu: HostGpu,
    /// Its loss as its command buffers see it (`super::command`'s docs, "What a completed handler
    /// learns" and "A lost device's line").
    loss: Arc<DeviceLoss>,
    /// The fills of the stages rt refuses for a compile not in its store (`super::stage`), which
    /// its encoders' dispatches wait for.
    stage_fills: Arc<StageFills<StageKey>>,
}

/// Where a device's GPU lies on the host, as Vulkan answers it, which the host's video opens its
/// APIs at: CUDA by the PCI address, VA-API on the render node, and on Windows on the DXGI adapter
/// of the LUID.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HostGpu {
    /// Its PCI domain, bus, device and function (`VK_EXT_pci_bus_info`); `None` where the device
    /// does not list the extension.
    pub pci: Option<(u32, u32, u32, u32)>,
    /// Its render node's major and minor numbers (`VK_EXT_physical_device_drm`); `None` where the
    /// device does not list the extension or has no render node.
    pub render: Option<(u32, u32)>,
    /// Its DXGI adapter's LUID (`VkPhysicalDeviceIDProperties::deviceLUID`); `None` where the
    /// device says it has no valid one, as off Windows.
    pub luid: Option<[u8; vk::LUID_SIZE]>,
}

impl RtDevice {
    /// rt's device.
    pub fn rt(&self) -> &rt::Device {
        &self.rt
    }

    /// What the platform read of it.
    pub fn facts(&self) -> &Facts {
        &self.facts
    }

    /// The texture indices of the platform's own.
    pub fn indices(&self) -> &OwnIndices {
        &self.indices
    }

    /// The sampler indices of the platform's own.
    pub fn sampler_indices(&self) -> &OwnIndices {
        &self.sampler_indices
    }

    /// The sampler indices its pipeline states' samplers hold.
    pub fn numbering(&self) -> &Numbering {
        &self.numbering
    }

    /// Its loss as its command buffers see it (`super::command`'s docs, "What a completed handler
    /// learns" and "A lost device's line"), which each of them keeps.
    pub fn loss(&self) -> &Arc<DeviceLoss> {
        &self.loss
    }

    /// The fills of the stages rt refuses for a compile not in its store (`super::stage`).
    pub fn stage_fills(&self) -> &Arc<StageFills<StageKey>> {
        &self.stage_fills
    }

    /// Its history of released ranges (`super::fault`), which the tables' objects record their
    /// releases in as they go.
    pub fn releases(&self) -> &super::fault::Releases {
        self.loss.watch().releases()
    }

    /// Walks `tables`, a decoder's, at its loss too, placing the loss's fault among their live
    /// objects (`super::fault`).
    pub fn watch_faults(&self, tables: Arc<dyn super::fault::LiveObjects>) {
        self.loss.watch().watch(tables);
    }

    /// Whether it is lost, which nothing makes again: rt's `Device::loss` names a cause. rt sets
    /// it at its own Vulkan call that meets the loss, before any of its command buffers ends with
    /// the device lost, and a window's swapchain reports a loss its own call meets first
    /// (`super::swapchain`'s docs, "A lost device"). A window's swapchain makes no Vulkan call on
    /// it from then on but its end's destroys.
    pub fn is_lost(&self) -> bool {
        self.rt.loss().is_some()
    }

    /// The blocks it knows by address (`super::buffer`'s docs, "A pinned address"): the graphics
    /// lane keeps a clone, the same set, to know and forget its blocks in.
    pub fn known_blocks(&self) -> &KnownBlocks {
        &self.known
    }

    /// The instance it was made on.
    pub fn instance(&self) -> &rt::Instance {
        &self.instance
    }

    /// Whether DRM node `node` (major, minor) is one of this GPU's: `None` where the device lists
    /// no DRM nodes.
    pub fn composited_here(&self, node: (u32, u32)) -> Option<bool> {
        self.drm.map(|drm| drm.names(node))
    }

    /// Whether each present can signal a fence of its own (`swapchainMaintenance1`).
    pub fn has_present_fences(&self) -> bool {
        self.present_fences
    }

    /// Where its GPU lies on the host.
    pub fn host_gpu(&self) -> HostGpu {
        self.host_gpu
    }
}

/// Why the platform has no device.
#[derive(Debug)]
pub enum Unopened {
    /// rt refused a call making it.
    Rt(rt::Error),
    /// No physical device meets rt's floor: why each listed one does not, or that none is listed.
    NoDevice(String),
}

impl Unopened {
    /// Counts the refusal at `sel` ([`super::unbuilt::rt_refused_create`]); its kind.
    pub fn refuse(&self, sel: &'static str) -> RtRefusal {
        match self {
            Unopened::Rt(error) => rt_refused_create(sel, error),
            Unopened::NoDevice(why) => {
                crate::core::platform::device::note_relayed_gap();
                refused_as(sel, RtRefusal::NoDevice, why);
                RtRefusal::NoDevice
            }
        }
    }
}

impl From<rt::Error> for Unopened {
    fn from(error: rt::Error) -> Unopened {
        Unopened::Rt(error)
    }
}

/// Makes the process's rt device (the module's docs), on a host of pages of `page_size` bytes,
/// and logs it.
pub fn open(page_size: u64) -> Result<RtObj<RtDevice>, Unopened> {
    let instance = crate::os::rt_instance(&crate::os::rt_instance_extensions())?;
    let listed = instance.physical_devices()?;
    let (_, raw) = instance.raw();
    let mut below = Vec::new();
    let mut candidates = Vec::new();
    for physical in listed {
        // SAFETY: the instance listed the device and is alive.
        let properties = unsafe { raw.get_physical_device_properties(physical.raw()) };
        let name = (properties.device_name_as_c_str())
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        match physical.check() {
            Ok(floor) => candidates.push((physical, floor)),
            Err(why) => below.push(format!("{name}: {why}")),
        }
    }
    for line in &below {
        log::info!(target: TARGET, "not a candidate for the rt platform's device: {line}");
    }
    let seen: Vec<crate::device_choice::Seen> = (candidates.iter())
        .map(|(physical, _)| {
            // SAFETY: the instance listed the device and is alive.
            let names = unsafe { raw.enumerate_device_extension_properties(physical.raw()) }
                .unwrap_or_default();
            let lists = |name: &std::ffi::CStr| {
                (names.iter()).any(|p| p.extension_name_as_c_str() == Ok(name))
            };
            let drm = lists(ash::ext::physical_device_drm::NAME);
            let pci = lists(ash::ext::pci_bus_info::NAME);
            // The one device extension a display's window takes here, which rt's floor takes
            // too (the module's docs, "Its window system").
            let window_lacks = [ash::khr::swapchain::NAME]
                .into_iter()
                .filter(|&name| !lists(name))
                .collect();
            // SAFETY: as above; rt's instance is at Vulkan 1.3, where `properties2` is core.
            unsafe { crate::os::seen(raw, physical.raw(), true, drm, pci, window_lacks) }
        })
        .collect();
    let Some(at) = crate::device_choice::chosen(&seen) else {
        return Err(Unopened::NoDevice(if below.is_empty() {
            "the loader lists no physical device".to_owned()
        } else {
            below.join("; ")
        }));
    };
    let drm = seen.get(at).and_then(|s| s.drm);
    let luid = candidates.get(at).and_then(|(physical, _)| {
        let mut id = vk::PhysicalDeviceIDProperties::default();
        let mut asked = vk::PhysicalDeviceProperties2::default().push_next(&mut id);
        // SAFETY: the instance listed the device and is alive; rt's instance is at Vulkan 1.3,
        // where the structure is core.
        unsafe { raw.get_physical_device_properties2(physical.raw(), &mut asked) };
        (id.device_luid_valid == vk::TRUE).then_some(id.device_luid)
    });
    let host_gpu = seen.get(at).map_or_else(HostGpu::default, |seen| HostGpu {
        pci: seen.pci,
        render: seen.drm.and_then(|nodes| nodes.render),
        luid,
    });
    let (Some((physical, floor)), Some(uuid)) =
        (candidates.get(at), seen.get(at).and_then(|s| s.uuid))
    else {
        return Err(Unopened::NoDevice(
            "the chosen device has no device UUID".to_owned(),
        ));
    };
    let (mut options, fences) = shared_options(&instance, physical);
    // rt's pool allocations, noted as rt maps them and forgotten as it unmaps them, among which a
    // lost device's fault is placed (`super::fault`).
    let allocations = Arc::new(super::fault::Allocations::default());
    options.block_mappings = Some(Arc::clone(&allocations) as Arc<dyn rt::BlockMappings>);
    options.pipeline_cache_dir = crate::os::user_cache_directory().map(|dir| dir.join(CACHE));
    options.pipeline_cache_cap = Some(SHADER_CACHE_CAPACITY);
    let device = rt::Device::new(physical, options)?;
    // Where it binds no several dma-bufs behind one buffer, a view more than one holds is mapped
    // from its files from the first (`buffer`'s docs, "Pages no pin takes").
    #[cfg(target_os = "linux")]
    crate::os::learn_dma_buf_granule(device.dma_buf_granule());
    let present_fences =
        fences.is_some_and(|name| (device.extensions().iter()).any(|n| n.as_c_str() == name));
    // The process's latest device (the module's docs), the one before it folded first.
    follow(device.counters());
    // Its gauges in the log, where the log asks for them (`super::gauge_log`).
    super::gauge_log::start();
    let facts = facts(&device, floor, &uuid, page_size);
    let driver = {
        let mut driver = vk::PhysicalDeviceDriverProperties::default();
        let mut asked = vk::PhysicalDeviceProperties2::default().push_next(&mut driver);
        // SAFETY: the instance listed the device and is alive; rt's instance is at Vulkan 1.3,
        // where the structure is core.
        unsafe { raw.get_physical_device_properties2(physical.raw(), &mut asked) };
        let text = |name: Result<&std::ffi::CStr, _>| {
            name.map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default()
        };
        format!(
            "{} {}",
            text(driver.driver_name_as_c_str()),
            text(driver.driver_info_as_c_str())
        )
    };
    log::info!(
        target: TARGET,
        "the rt platform's device: {} ({:04x}:{:04x}), driver {driver}, registry ID {:#x}; \
         instance extensions {:?}; device extensions {:?}; present fences: {present_fences}",
        facts.name,
        facts.vendor_id,
        facts.device_id,
        facts.registry_id,
        instance.extensions(),
        device.extensions()
    );
    let caps = answers::caps(&facts);
    let gpu = format!(
        "{} ({:04x}:{:04x})",
        facts.name, facts.vendor_id, facts.device_id
    );
    let known = KnownBlocks::default();
    let watch = super::fault::FaultWatch::new(allocations, known.clone());
    let loss = Arc::new(DeviceLoss::watching(gpu, watch));
    let order = || Arc::clone(loss.flights());
    let stage_fills = Arc::new(super::stage::fills(device.clone(), uuid));
    Ok(RtObj::new(RtDevice {
        rt: device,
        facts,
        caps,
        indices: OwnIndices::new(rt::IndexClass::Texture, order()),
        sampler_indices: OwnIndices::new(rt::IndexClass::Sampler, order()),
        numbering: Numbering::default(),
        shaders: OnceLock::new(),
        known,
        instance,
        drm,
        present_fences,
        host_gpu,
        loss,
        stage_fills,
    }))
}

/// The options a device on `physical`, one of `instance`'s, is opened with both in the VM's
/// process ([`open`]) and in a stage's helper (`super::stage::helper_main`), where rt takes the
/// helper's compile only from a device that enabled what the VM's did (its `StageFill::Enables`
/// otherwise): the extension of present fences, with the feature it takes, where the device lists
/// it and its instance extensions are on (the module's docs, "Its window system"). With that
/// extension's name, where it is asked for.
pub(crate) fn shared_options(
    instance: &rt::Instance,
    physical: &rt::PhysicalDevice,
) -> (rt::DeviceOptions, Option<&'static std::ffi::CStr>) {
    let mut options = rt::DeviceOptions::default();
    let (_, raw) = instance.raw();
    // SAFETY: the instance listed the device and is alive.
    let names =
        unsafe { raw.enumerate_device_extension_properties(physical.raw()) }.unwrap_or_default();
    let lists =
        |name: &std::ffi::CStr| (names.iter()).any(|p| p.extension_name_as_c_str() == Ok(name));
    let fences = crate::os::rt_present_fences(instance.extensions(), lists);
    if let Some(name) = fences {
        let wanted = vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT::default()
            .swapchain_maintenance1(true);
        let needs = |f: &vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT<'static>| {
            f.swapchain_maintenance1 == vk::TRUE
        };
        (options.extensions).push(rt::DeviceExtension::with_features(name, wanted, needs));
    }
    (options, fences)
}

/// What the answers of `device`, made with rt's floor answers `floor`, of `deviceUUID` `uuid`, on
/// a host of `page_size`-byte pages, depend on ([`Facts`]).
fn facts(
    device: &rt::Device,
    floor: &rt::floor::Capabilities,
    uuid: &[u8; vk::UUID_SIZE],
    page_size: u64,
) -> Facts {
    let raw = device.raw();
    let (instance, physical) = (raw.instance, raw.physical_device);
    // SAFETY: the instance and physical device are rt's, read only, while `device` lives.
    let (properties, features) = unsafe {
        (
            instance.get_physical_device_properties(physical),
            instance.get_physical_device_features(physical),
        )
    };
    let float = vk::Format::R32G32B32A32_SFLOAT;
    // SAFETY: as above.
    let filtering = unsafe { instance.get_physical_device_format_properties(physical, float) }
        .optimal_tiling_features
        .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR);
    // SAFETY: as above.
    let msaa = unsafe {
        instance.get_physical_device_image_format_properties(
            physical,
            float,
            vk::ImageType::TYPE_2D,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::COLOR_ATTACHMENT,
            vk::ImageCreateFlags::empty(),
        )
    }
    .is_ok_and(|p| p.sample_counts.contains(vk::SampleCountFlags::TYPE_4));
    let sampled = |count: u64| {
        u32::try_from(count).is_ok_and(|count| device.supports_texture_sample_count(count))
    };
    let barycentrics = (device.extensions().iter())
        .any(|name| name.as_c_str() == ash::khr::fragment_shader_barycentric::NAME);
    Facts {
        name: (properties.device_name_as_c_str())
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default(),
        registry_id: answers::registry_id(uuid),
        vendor_id: properties.vendor_id,
        device_id: properties.device_id,
        pipeline_cache_uuid: properties.pipeline_cache_uuid,
        limits: properties.limits,
        float16: floor.float16,
        unified_memory: device.has_unified_memory(),
        sample_counts: SAMPLE_COUNTS.map(sampled),
        bc: features.texture_compression_bc == vk::TRUE,
        barycentrics,
        float32_filtering: filtering,
        float32_msaa: msaa,
        // SAFETY: as above.
        device_local: unsafe { identity::device_local_bytes_of(instance, physical) },
        page_size,
        heap_buffer_page_private: {
            let sized = device.heap_buffer_size_and_align(page_size, STORAGE_PRIVATE);
            SizeAndAlign {
                size: sized.size,
                align: sized.align,
            }
        },
        linear_alignment_largest: (crate::core::platform::caps::SDK_PIXEL_FORMATS.iter())
            .filter_map(|&format| linear_alignment(device, format))
            .fold(METAL_LINEAR_ALIGNMENT, u64::max),
        linear_alignment_bgra8: device
            .minimum_linear_texture_alignment(rt::PixelFormat::BGRA8Unorm)
            .ok(),
        indirect_tokens: floor.indirect_commands.map(|limits| limits.tokens),
    }
}

/// rt's `minimumLinearTextureAlignmentForPixelFormat:` of the raw pixel format `format`; none
/// where rt has no format of it or refuses it.
fn linear_alignment(device: &rt::Device, format: u64) -> Option<u64> {
    let pf = crate::gpu::runtime_pixel_format(format)?;
    device.minimum_linear_texture_alignment(pf).ok()
}

/// rt's `minimumTextureBufferAlignmentForPixelFormat:` of the raw pixel format `format`, the
/// offset alignment its `Buffer::new_texture` holds a texture buffer of it to; none where rt has
/// no format of it or makes no texture buffer of it.
fn texture_buffer_alignment(device: &rt::Device, format: u64) -> Option<u64> {
    let pf = crate::gpu::runtime_pixel_format(format)?;
    device.minimum_texture_buffer_alignment(pf).ok()
}

/// `MTLResourceStorageModePrivate`, as resource options.
const STORAGE_PRIVATE: u64 = 0x20;

/// Whether a GPU whose driver reports Vulkan's device type `device_type` is an integrated one: of
/// type `INTEGRATED_GPU`, and of no other (discrete, virtual, CPU, other).
fn integrated(device_type: vk::PhysicalDeviceType) -> bool {
    device_type == vk::PhysicalDeviceType::INTEGRATED_GPU
}

/// The raise of a query rt refused at `sel`, counted already under `kind`: Metal's exception at
/// the send, its detail the kind's code and why.
fn refused_raise(sel: &'static str, kind: RtRefusal, why: &dyn std::fmt::Display) -> Raise {
    Raise::platform(sel).with_detail(format!("{}: {why}", kind.code()))
}

/// Grants `indices` of `class` as runs of consecutive indices, each one grant; an index granted
/// already stays granted, as a second reservation of Metal's of an ID reserved changes nothing.
/// One rt refuses otherwise (past the class's range) is counted at `sel`.
fn grant(device: &rt::Device, sel: &'static str, class: rt::IndexClass, indices: &[u64]) {
    let mut sorted: Vec<u64> = indices.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let mut runs: Vec<(u64, u64)> = Vec::new();
    for index in sorted {
        match runs.last_mut() {
            Some((first, count)) if *first + *count == index => *count += 1,
            _ => runs.push((index, 1)),
        }
    }
    for (first, count) in runs {
        let (Ok(first), Ok(count)) = (u32::try_from(first), u32::try_from(count)) else {
            let why = format!("index {first} is past 32 bits");
            refused_as(sel, RtRefusal::Limit, &why);
            continue;
        };
        let Err(error) = device.grant_indices(class, first, count) else {
            continue;
        };
        if !granted_already(&error) {
            rt_refused(sel, &error);
            continue;
        }
        for index in first..first.saturating_add(count) {
            match device.grant_indices(class, index, 1) {
                Err(error) if !granted_already(&error) => {
                    rt_refused(sel, &error);
                }
                _ => {}
            }
        }
    }
}

/// Whether rt refused a grant because an index of it is granted already.
fn granted_already(error: &rt::Error) -> bool {
    matches!(
        error,
        rt::Error::Refused(rt::Refusal::IndicesGranted { .. })
    )
}

/// The selector of the resource indices' reservation.
const RESERVE: &str = "reserveResourceIndicesForResourceType:indices:indexCount:";

impl GpuDevice<RtPlatform> for RtObj<RtDevice> {
    /// Nothing: rt takes a library of any platform's.
    fn allow_foreign_libraries(&self) {}

    /// The translator's layout of the descriptors, Metal's encoder's (`super::argument`); `None`,
    /// counted, where Metal makes no encoder.
    fn argument_layout(&self, arguments: &[ArgumentDesc]) -> RaiseResult<Option<ArgumentLayout>> {
        Ok(argument::descriptors_layout(arguments))
    }

    /// The translator's layout of the function's argument buffer at the index, the one its
    /// translation reads (`super::argument`); `None`, counted, where Metal makes no encoder.
    fn function_argument_layout(
        &self,
        function: &<RtPlatform as Platform>::Function,
        buffer_index: u64,
    ) -> RaiseResult<Option<ArgumentLayout>> {
        Ok(argument::function_layout(function, buffer_index))
    }

    /// rt's answer: the device's image's own memory requirements, rounded to its buffer-image
    /// granularity, which the guest lays its heaps out by, not Metal's. Where rt refuses (a
    /// texture it makes none of, or `Gap::HeapTexture`, an image that takes an allocation of its
    /// own), the refusal is counted under its kind and raised ([`refused_raise`]), which the
    /// decoder answers the guest as refused, and the guest then lays the texture out by a bound
    /// of its own: never 0 bytes at 0, which places the texture at its neighbours' offset. A
    /// value rt's descriptor has no value of ([`super::descriptor::rt_texture_descriptor`]) is
    /// counted as Metal's refusal (`rt-refused-metal`) and raised alike.
    fn heap_texture_size_and_align(
        &self,
        desc: &<RtPlatform as Platform>::TextureDescriptor,
    ) -> RaiseResult<SizeAndAlign> {
        const SEL: &str = "heapTextureSizeAndAlignWithDescriptor:";
        let d = match super::descriptor::rt_texture_descriptor(&desc.desc()) {
            Ok(d) => d,
            Err(why) => {
                refused_as(SEL, RtRefusal::Refused, &why);
                return Err(refused_raise(SEL, RtRefusal::Refused, &why));
            }
        };
        match self.rt.heap_texture_size_and_align(&d) {
            Ok(rt::SizeAndAlign { size, align }) => Ok(SizeAndAlign { size, align }),
            Err(error) => {
                let desc = desc.desc();
                let options = desc.resource_options.unwrap_or(0);
                let what = super::texture::TextureDetail {
                    desc: &desc,
                    options,
                };
                super::unbuilt::refusal_detail(SEL, &error, Some(&what));
                let kind = rt_refused_texture(SEL, &error);
                Err(refused_raise(SEL, kind, &error))
            }
        }
    }

    /// A buffer over rt's import of the pages that hold `bytes` ([`buffer::over_host_memory`]);
    /// nil where Metal or rt refuses it, counted, the deallocator then dropped unrun, as Metal
    /// runs none for a buffer it does not make.
    fn new_buffer_over_host_memory(
        &self,
        on: OnFault,
        bytes: HostRange<'_>,
        options: u64,
        deallocator: Option<NoCopyDeallocator>,
    ) -> RaiseResult<Option<<RtPlatform as Platform>::Buffer>> {
        let mut recorded_call = crate::recorder::rec_call!(
            "newBufferWithBytesNoCopy:length:options:deallocator:",
            self,
            bytes,
            options,
            deallocator
        );
        // The bytes the create copies into the device's own copy of a `Managed` buffer.
        if recorded_call.is_some() {
            let two_copies = !self.facts().unified_memory;
            crate::recorder::no_copy_contents_with(
                &mut recorded_call,
                two_copies,
                options,
                bytes,
                None,
            );
        }
        let recorded_answer = crate::recorder::run(
            || -> RaiseResult<Option<<RtPlatform as Platform>::Buffer>> {
                let _ = on;
                let sel = "newBufferWithBytesNoCopy:length:options:deallocator:";
                let length = bytes.len();
                let made = buffer::over_host_memory(
                    self,
                    sel,
                    bytes,
                    length,
                    options,
                    None,
                    None,
                    deallocator,
                );
                Ok(made)
            },
        );
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// A buffer of zeros in blocks of its own ([`buffer::of_length`]); nil where Metal or rt
    /// refuses it, counted.
    fn new_buffer_of_length(
        &self,
        on: OnFault,
        length: u64,
        options: u64,
    ) -> RaiseResult<Option<<RtPlatform as Platform>::Buffer>> {
        let recorded_call =
            crate::recorder::rec_call!("newBufferWithLength:options:", self, length, options);
        let recorded_answer = crate::recorder::run(
            || -> RaiseResult<Option<<RtPlatform as Platform>::Buffer>> {
                let _ = on;
                let sel = "newBufferWithLength:options:";
                Ok(buffer::of_length(self, sel, length, options))
            },
        );
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// [`new_buffer_of_length`](GpuDevice::new_buffer_of_length)'s, a Shared one not zeroed
    /// ([`buffer::scratch_of_length`]).
    fn new_scratch_buffer_of_length(
        &self,
        on: OnFault,
        length: u64,
        options: u64,
    ) -> RaiseResult<Option<<RtPlatform as Platform>::Buffer>> {
        let recorded_call =
            crate::recorder::rec_call!("newBufferWithLength:options:", self, length, options);
        let recorded_answer = crate::recorder::run(
            || -> RaiseResult<Option<<RtPlatform as Platform>::Buffer>> {
                let _ = on;
                let sel = "newBufferWithLength:options:";
                Ok(buffer::scratch_of_length(self, sel, length, options))
            },
        );
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// [`new_buffer_of_length`](GpuDevice::new_buffer_of_length)'s, a Private one's memory made
    /// at its first use ([`buffer::deferred_of_length`]). A recording holds it as the
    /// `newBufferWithLength:options:` it stands for, a buffer of zeros, as the Vulkan platform's
    /// is, so the copies and binds that name it (an IOSurface device cache's in the per-shape
    /// form) name its stream id.
    fn new_deferred_buffer_of_length(
        &self,
        on: OnFault,
        length: u64,
        options: u64,
    ) -> RaiseResult<Option<<RtPlatform as Platform>::Buffer>> {
        let recorded_call =
            crate::recorder::rec_call!("newBufferWithLength:options:", self, length, options);
        let recorded_answer = crate::recorder::run(
            || -> RaiseResult<Option<<RtPlatform as Platform>::Buffer>> {
                let _ = on;
                let sel = "newBufferWithLength:options:";
                Ok(buffer::deferred_of_length(self, sel, length, options))
            },
        );
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// A buffer of zeros of its own exported block, whose memory CUDA imports as an opaque fd
    /// (`super::export`); `None` where rt makes none, counted, and the caller makes the buffer as
    /// any other.
    #[cfg(unix)]
    fn new_exportable_buffer_of_length(
        &self,
        on: OnFault,
        length: u64,
        options: u64,
        aligned_byte: Option<u64>,
    ) -> RaiseResult<Option<<RtPlatform as Platform>::Buffer>> {
        let recorded_call = crate::recorder::rec_call!(
            "m2v.newExportableBufferOfLength",
            self,
            length,
            options,
            aligned_byte
        );
        let recorded_answer = crate::recorder::run(
            || -> RaiseResult<Option<<RtPlatform as Platform>::Buffer>> {
                let _ = on;
                let sel = "newBufferWithLength:options: (exportable)";
                let handle = rt::ExportHandle::OpaqueFd;
                let made =
                    super::export::exportable(self, sel, handle, length, options, aligned_byte);
                Ok(made)
            },
        );
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// A buffer of zeros of its own exported block, whose memory VA-API imports as a dma-buf
    /// (`super::export`), its byte `aligned_byte`, where given, on a multiple of 256 bytes of the
    /// memory; `None` where rt makes none, counted, and the caller makes the buffer as any other.
    /// The bytes between a texture's rows are no matter: rt makes no texture over a buffer.
    #[cfg(unix)]
    fn new_dma_buf_buffer_of_length_padding_unobserved(
        &self,
        on: OnFault,
        length: u64,
        options: u64,
        aligned_byte: Option<u64>,
    ) -> RaiseResult<Option<<RtPlatform as Platform>::Buffer>> {
        let recorded_call = crate::recorder::rec_call!(
            "m2v.newDmaBufBufferOfLengthPaddingUnobserved",
            self,
            length,
            options,
            aligned_byte
        );
        let recorded_answer = crate::recorder::run(
            || -> RaiseResult<Option<<RtPlatform as Platform>::Buffer>> {
                let _ = on;
                let sel = "newBufferWithLength:options: (dma-buf)";
                let handle = rt::ExportHandle::DmaBuf;
                let made =
                    super::export::exportable(self, sel, handle, length, options, aligned_byte);
                Ok(made)
            },
        );
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// rt's exported surface of `desc`'s planes (`super::surface`); `None` where rt makes none,
    /// counted, and the caller keeps the planes in images of its own.
    #[cfg(unix)]
    fn new_exported_surface(
        &self,
        on: OnFault,
        desc: &crate::core::platform::resource::SurfaceDesc<'_>,
    ) -> RaiseResult<Option<crate::core::platform::resource::ExportedSurface<RtPlatform>>> {
        let _ = on;
        Ok(super::surface::exported_surface(self, desc))
    }

    /// rt's surface over the Direct3D 11 textures `planes` name (`super::d3d11_surface`); `None`
    /// where rt makes none, counted, and the caller keeps the planes in images of its own.
    #[cfg(windows)]
    fn import_d3d11_surface(
        &self,
        on: OnFault,
        planes: &[crate::core::platform::resource::D3d11PlaneDesc<'_>],
    ) -> RaiseResult<Option<crate::core::platform::resource::ImportedSurface<RtPlatform>>> {
        let _ = on;
        Ok(super::d3d11_surface::imported_surface(self, planes))
    }

    /// rt's queue, of Metal's default `maxCommandBufferCount`; nil where rt refuses it.
    fn new_command_queue(&self) -> Option<RtObj<RtQueue>> {
        let recorded_call = crate::recorder::rec_call!("newCommandQueue", self);
        let recorded_answer = crate::recorder::run(|| -> Option<RtObj<RtQueue>> {
            match self.rt.new_command_queue(METAL_MAX_COMMAND_BUFFER_COUNT) {
                Ok(queue) => Some(RtObj::new(RtQueue::new(queue, self.clone_ref()))),
                Err(error) => {
                    rt_refused_create("newCommandQueue", &error);
                    None
                }
            }
        });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// [`new_command_queue`](GpuDevice::new_command_queue)'s: rt submits every command queue's
    /// work to its one GPU queue.
    fn new_command_queue_unshared(&self) -> Option<RtObj<RtQueue>> {
        let recorded_call = crate::recorder::rec_call!("m2v.newCommandQueueUnshared", self);
        let recorded_answer =
            crate::recorder::run(|| -> Option<RtObj<RtQueue>> { self.new_command_queue() });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// No: rt copies no guest RAM by its host addresses.
    fn copies_guest_ram(&self) -> bool {
        false
    }

    /// Suspends its stage fills (`super::stage`): each running ends cancelled and its helper is
    /// killed, and none starts until [`Self::reset_ends`].
    fn reset_begins(&self) {
        self.stage_fills.suspend();
    }

    /// Resumes its stage fills.
    fn reset_ends(&self) {
        self.stage_fills.resume();
    }

    /// rt's `minimumLinearTextureAlignmentForPixelFormat:` of BGRA8, the device's linear row
    /// rounding; 0 where rt refuses it.
    fn linear_row_alignment(&self) -> u64 {
        (self.rt)
            .minimum_linear_texture_alignment(rt::PixelFormat::BGRA8Unorm)
            .unwrap_or(0)
    }

    /// rt's state of the function alone (`super::pipeline`); nil for no function.
    fn new_compute_pipeline_state_for_function(
        &self,
        compute_function: Option<&<RtPlatform as Platform>::Function>,
    ) -> Result<Option<<RtPlatform as Platform>::ComputePipeline>, NsErrorText> {
        let recorded_call = crate::recorder::rec_call!(
            "newComputePipelineStateWithFunction:error:",
            self,
            compute_function
        );
        let recorded_answer = crate::recorder::run(
            || -> Result<Option<<RtPlatform as Platform>::ComputePipeline>, NsErrorText> {
                let Some(function) = compute_function else {
                    return Ok(None);
                };
                pipeline::of_function(self, function).map(Some)
            },
        );
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// The platform's shaders, read once (`super::library`).
    fn new_shader_library(&self) -> Result<Option<<RtPlatform as Platform>::Library>, NsErrorText> {
        let recorded_call = crate::recorder::rec_call!("newLibraryWithSource:options:error:", self);
        let recorded_answer = crate::recorder::run(
            || -> Result<Option<<RtPlatform as Platform>::Library>, NsErrorText> {
                library::shader_library(&self.shaders).map(Some)
            },
        );
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// rt's event; nil where rt refuses it.
    fn new_event(&self, on: OnFault) -> RaiseResult<Option<RtObj<rt::Event>>> {
        let recorded_call = crate::recorder::rec_call!("newEvent", self);
        let recorded_answer = crate::recorder::run(|| -> RaiseResult<Option<RtObj<rt::Event>>> {
            let _ = on;
            Ok(match self.rt.new_event() {
                Ok(event) => Some(RtObj::new(event)),
                Err(error) => {
                    rt_refused_create("newEvent", &error);
                    None
                }
            })
        });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// rt's fence.
    fn new_fence(&self) -> RaiseResult<Option<RtObj<rt::Fence>>> {
        let recorded_call = crate::recorder::rec_call!("newFence", self);
        let recorded_answer = crate::recorder::run(|| -> RaiseResult<Option<RtObj<rt::Fence>>> {
            Ok(Some(RtObj::new(self.rt.new_fence())))
        });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// The translator's library of a metallib (`super::library`).
    fn new_library(
        &self,
        data: &[u8],
    ) -> RaiseResult<Result<Option<<RtPlatform as Platform>::Library>, NsErrorText>> {
        let recorded_call = crate::recorder::rec_call!("newLibraryWithData:error:", self, data);
        let recorded_answer = crate::recorder::run(
            || -> RaiseResult<Result<Option<<RtPlatform as Platform>::Library>, NsErrorText>> {
                Ok(library::new_library(data).map(Some))
            },
        );
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// Unbuilt ([`Unbuilt::DynamicLibrary`]): the `NSError`.
    fn new_dynamic_library(
        &self,
        library: &<RtPlatform as Platform>::Library,
    ) -> RaiseResult<Result<Option<<RtPlatform as Platform>::DynamicLibrary>, NsErrorText>> {
        let recorded_call = crate::recorder::rec_call!("newDynamicLibrary:error:", self, library);
        let recorded_answer =
            crate::recorder::run(|| -> RaiseResult<Result<Option<<RtPlatform as Platform>::DynamicLibrary>, NsErrorText>> {
                let _ = library;
                Ok(Err(refused_error(
                    Unbuilt::DynamicLibrary,
                    "newDynamicLibrary:error:",
                )))
            });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// Unbuilt: nil ([`Unbuilt::RateMap`]).
    fn new_rate_map(
        &self,
        descriptor: &<RtPlatform as Platform>::RateMapDescriptor,
    ) -> RaiseResult<Option<<RtPlatform as Platform>::RateMap>> {
        let recorded_call =
            crate::recorder::rec_call!("newRasterizationRateMapWithDescriptor:", self, descriptor);
        let recorded_answer = crate::recorder::run(
            || -> RaiseResult<Option<<RtPlatform as Platform>::RateMap>> {
                let _ = descriptor;
                refuse_create(Unbuilt::RateMap, "newRasterizationRateMapWithDescriptor:");
                Ok(None)
            },
        );
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// rt's shared event ([`super::event`]); nil where rt refuses it, counted.
    fn new_shared_event(&self) -> Option<<RtPlatform as Platform>::SharedEvent> {
        let recorded_call = crate::recorder::rec_call!("newSharedEvent", self);
        let recorded_answer =
            crate::recorder::run(|| -> Option<<RtPlatform as Platform>::SharedEvent> {
                match self.rt.new_shared_event() {
                    Ok(event) => Some(RtObj::new(event)),
                    Err(error) => {
                        rt_refused_create("newSharedEvent", &error);
                        None
                    }
                }
            });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// Unbuilt: nil ([`Unbuilt::SharedTexture`]).
    fn new_shared_texture(
        &self,
        descriptor: &<RtPlatform as Platform>::TextureDescriptor,
    ) -> RaiseResult<Option<<RtPlatform as Platform>::Texture>> {
        let recorded_call =
            crate::recorder::rec_call!("newSharedTextureWithDescriptor:", self, descriptor);
        let recorded_answer = crate::recorder::run(
            || -> RaiseResult<Option<<RtPlatform as Platform>::Texture>> {
                let _ = descriptor;
                refuse_create(Unbuilt::SharedTexture, "newSharedTextureWithDescriptor:");
                Ok(None)
            },
        );
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// Unbuilt: nil ([`Unbuilt::SharedTexture`]).
    fn open_shared_texture(
        &self,
        handle: Option<&<RtPlatform as Platform>::SharedTextureHandle>,
        resource_index: u64,
    ) -> RaiseResult<Option<<RtPlatform as Platform>::Texture>> {
        let recorded_call = crate::recorder::rec_call!(
            "newSharedTextureWithHandle:withResourceIndex:",
            self,
            handle,
            resource_index
        );
        let recorded_answer = crate::recorder::run(
            || -> RaiseResult<Option<<RtPlatform as Platform>::Texture>> {
                if let Some(handle) = handle {
                    handle.unreachable()
                }
                let _ = resource_index;
                refuse_create(
                    Unbuilt::SharedTexture,
                    "newSharedTextureWithHandle:withResourceIndex:",
                );
                Ok(None)
            },
        );
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// rt's texture of the descriptor ([`super::texture::make`]); a texture buffer rt's over a
    /// buffer of the platform's own that the texture holds
    /// ([`super::texture::make_texture_buffer`]). Nil where rt refuses it, counted, never a
    /// raise.
    fn new_texture(
        &self,
        on: OnFault,
        descriptor: &<RtPlatform as Platform>::TextureDescriptor,
    ) -> RaiseResult<Option<<RtPlatform as Platform>::Texture>> {
        let recorded_call =
            crate::recorder::rec_call!("newTextureWithDescriptor:", self, descriptor);
        let recorded_answer = crate::recorder::run(
            || -> RaiseResult<Option<<RtPlatform as Platform>::Texture>> {
                let _ = on;
                let sel = "newTextureWithDescriptor:";
                let desc = descriptor.desc();
                if desc.texture_type == super::texture::TYPE_TEXTURE_BUFFER {
                    return Ok(super::texture::make_texture_buffer(self, sel, descriptor));
                }
                Ok(super::texture::make(self, sel, &desc))
            },
        );
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// Textures' (type 0) and sampler states' (type 1) indices as rt's grants of their class
    /// ([`grant`]), each also kept from the platform's own ([`OwnIndices::reserve`]);
    /// indirect command buffers' (type 5) are unbuilt ([`Unbuilt::Icb`]: rt gives a buffer no
    /// index), and those of another type ([`Unbuilt::ResourceType`]): each reserves nothing.
    /// Never a raise.
    fn reserve_resource_indices(&self, resource_type: u64, indices: &[u64]) -> RaiseResult<()> {
        let recorded_call = crate::recorder::rec_call!(
            "reserveResourceIndicesForResourceType:indices:indexCount:",
            self,
            resource_type,
            indices
        );
        let recorded_answer = crate::recorder::run(|| -> RaiseResult<()> {
            let class = match resource_type {
                0 => {
                    self.indices.reserve(indices);
                    rt::IndexClass::Texture
                }
                1 => {
                    self.sampler_indices.reserve(indices);
                    rt::IndexClass::Sampler
                }
                5 => {
                    refuse(Unbuilt::Icb, RESERVE);
                    return Ok(());
                }
                _ => {
                    refuse(Unbuilt::ResourceType, RESERVE);
                    return Ok(());
                }
            };
            grant(&self.rt, RESERVE, class, indices);
            Ok(())
        });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// rt's description ([`super::format::description`]): Metal's on an Apple GPU, on every
    /// device.
    fn pixel_format_info(&self, format: u64) -> PixelFormatInfo {
        super::format::description(&self.rt, format)
    }

    /// rt's layout of the level ([`super::format::level_layout`]).
    fn texture_level_info(&self, q: &TextureLevelQuery) -> TextureLevelInfo {
        super::format::level_layout(&self.rt, q)
    }

    /// Read as the device was made ([`answers::caps`]).
    fn caps(&self) -> Arc<DeviceCaps> {
        Arc::clone(&self.caps)
    }

    /// [`answers::identity`], with rt's linear alignment of each format.
    fn identity_caps(&self) -> IdentityCaps {
        identity_of(self)
    }

    /// Yes, as on the Vulkan platform: the graphics lane offers its pipelines' private state, and
    /// rt's pipelines, once built, take it.
    fn takes_private_render_state(&self) -> bool {
        true
    }

    /// The device-local heaps' budgets, as the Vulkan platform answers
    /// (`crate::gpu::identity::device_local_budget`).
    fn own_memory_budget(&self) -> u64 {
        let raw = self.rt.raw();
        // SAFETY: the instance and physical device are rt's, read only, while the device lives.
        unsafe { identity::device_local_budget_of(raw.instance, raw.physical_device) }
    }

    /// Every heap's size, as the Vulkan platform answers.
    fn memory_heaps_total(&self) -> u64 {
        let raw = self.rt.raw();
        // SAFETY: as above.
        let memory =
            unsafe { (raw.instance).get_physical_device_memory_properties(raw.physical_device) };
        identity::memory_heaps_total_of(&memory)
    }

    /// Whether the driver reports rt's physical device as of type `INTEGRATED_GPU`
    /// (`integrated`).
    fn is_integrated(&self) -> bool {
        let raw = self.rt.raw();
        // SAFETY: as above.
        let properties =
            unsafe { (raw.instance).get_physical_device_properties(raw.physical_device) };
        integrated(properties.device_type)
    }

    /// Yes, as on the Vulkan platform: rt's translator links an OpenCL-mode kernel as Metal does.
    fn makes_opencl_mode_pipelines(&self) -> bool {
        true
    }

    /// [`buffer::of_desc`]'s: nil where Metal or rt refuses it, counted, the deallocator then
    /// dropped unrun.
    fn new_buffer(
        &self,
        desc: BufferDesc<'_>,
    ) -> RaiseResult<Option<<RtPlatform as Platform>::Buffer>> {
        let mut recorded_call = crate::recorder::rec_call!("newBufferWithDescriptor:", self, desc);
        // The bytes a copy starts with; a no-copy buffer's arrive as writes, but those the create
        // copies into the device's own copy of a `Managed` one.
        if recorded_call.is_some()
            && let Some(bytes) = desc.contents
        {
            if desc.no_copy {
                let options = descriptor_options(buffer::BUFFER_DESCRIPTOR_OPTIONS, &desc.options);
                let two_copies = !self.facts().unified_memory;
                crate::recorder::no_copy_contents_with(
                    &mut recorded_call,
                    two_copies,
                    options,
                    bytes,
                    desc.zero_from,
                );
            } else {
                crate::recorder::contents(&mut recorded_call, bytes);
            }
        }
        let recorded_answer = crate::recorder::run(
            || -> RaiseResult<Option<<RtPlatform as Platform>::Buffer>> {
                Ok(buffer::of_desc(self, desc))
            },
        );
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// [`heap::of_desc`]'s, with the descriptor's options as Metal's descriptor holds them: nil
    /// where Metal or rt refuses it, or a sparse heap ([`Unbuilt::Heap`]), counted.
    fn new_heap(&self, desc: &HeapDesc) -> RaiseResult<HeapMade<RtPlatform>> {
        let recorded_call = crate::recorder::rec_call!("newHeapWithDescriptor:", self, desc);
        let recorded_answer = crate::recorder::run(|| -> RaiseResult<HeapMade<RtPlatform>> {
            let held = descriptor_options(HEAP_DESCRIPTOR_OPTIONS, &desc.options);
            Ok(HeapMade {
                heap: heap::of_desc(self, desc, held),
                descriptor_resource_options: held,
            })
        });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// [`heap::over_host_memory`]'s, with the descriptor's options: nil where Metal or rt refuses
    /// it, counted. The recording names the bytes and the descriptor, not the deallocator.
    fn new_heap_over_host_memory(
        &self,
        bytes: HostRange<'_>,
        desc: &HeapDesc,
        deallocator: Option<NoCopyDeallocator>,
    ) -> RaiseResult<HeapMade<RtPlatform>> {
        let recorded_call =
            crate::recorder::rec_call!("m2v.newBlockWithLength:storageMode:", self, bytes, desc);
        let recorded_answer = crate::recorder::run(|| -> RaiseResult<HeapMade<RtPlatform>> {
            let held = descriptor_options(HEAP_DESCRIPTOR_OPTIONS, &desc.options);
            Ok(HeapMade {
                heap: heap::over_host_memory(self, bytes, desc, held, deallocator),
                descriptor_resource_options: held,
            })
        });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// rt's sampler state ([`super::state`]); nil, counted, where rt refuses it or a value is one
    /// rt has none of.
    fn new_sampler_state(
        &self,
        desc: &SamplerDesc,
    ) -> RaiseResult<Option<<RtPlatform as Platform>::SamplerState>> {
        let recorded_call =
            crate::recorder::rec_call!("newSamplerStateWithDescriptor:", self, desc);
        let recorded_answer = crate::recorder::run(
            || -> RaiseResult<Option<<RtPlatform as Platform>::SamplerState>> {
                Ok(super::state::new_sampler_state(self, desc))
            },
        );
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// rt's depth-stencil state ([`super::state`]); nil, counted, where a value is one rt has
    /// none of.
    fn new_depth_stencil_state(
        &self,
        desc: &DepthStencilDesc,
    ) -> RaiseResult<Option<<RtPlatform as Platform>::DepthStencilState>> {
        let recorded_call =
            crate::recorder::rec_call!("newDepthStencilStateWithDescriptor:", self, desc);
        let recorded_answer = crate::recorder::run(
            || -> RaiseResult<Option<<RtPlatform as Platform>::DepthStencilState>> {
                Ok(super::state::new_depth_stencil_state(self, desc))
            },
        );
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// rt's buffer (`super::icb`); nil where rt refuses it, counted.
    fn new_indirect_command_buffer(
        &self,
        desc: &IcbDesc,
        max_count: u64,
        options: u64,
    ) -> RaiseResult<Option<<RtPlatform as Platform>::Icb>> {
        let recorded_call = crate::recorder::rec_call!(
            "newIndirectCommandBufferWithDescriptor:maxCommandCount:options:",
            self,
            desc,
            max_count,
            options
        );
        let recorded_answer =
            crate::recorder::run(|| -> RaiseResult<Option<<RtPlatform as Platform>::Icb>> {
                let _ = options;
                Ok(super::icb::make(self, desc, max_count))
            });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// The display's state (`super::render_pipeline`).
    fn new_render_pipeline_state(
        &self,
        desc: &RenderPipelineDesc<'_, RtPlatform>,
    ) -> Result<Option<<RtPlatform as Platform>::RenderPipeline>, NsErrorText> {
        let recorded_call =
            crate::recorder::rec_call!("newRenderPipelineStateWithDescriptor:error:", self, desc);
        let recorded_answer = crate::recorder::run(
            || -> Result<Option<<RtPlatform as Platform>::RenderPipeline>, NsErrorText> {
                render_pipeline::of_value(self, desc).map(Some)
            },
        );
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// The state of the whole descriptor (`super::render_pipeline`).
    fn new_render_pipeline_state_full(
        &self,
        desc: &FullRenderPipelineDesc<'_, RtPlatform>,
    ) -> Result<Option<<RtPlatform as Platform>::RenderPipeline>, NsErrorText> {
        let recorded_call =
            crate::recorder::rec_call!("newRenderPipelineStateWithDescriptor:error:", self, desc);
        let recorded_answer = crate::recorder::run(
            || -> Result<Option<<RtPlatform as Platform>::RenderPipeline>, NsErrorText> {
                render_pipeline::of_descriptor(self, desc).map(Some)
            },
        );
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// Unbuilt ([`Unbuilt::MeshPipeline`]): the `NSError`.
    fn new_mesh_render_pipeline_state_full(
        &self,
        desc: &FullMeshRenderPipelineDesc<'_, RtPlatform>,
    ) -> Result<Option<<RtPlatform as Platform>::RenderPipeline>, NsErrorText> {
        let recorded_call = crate::recorder::rec_call!(
            "newRenderPipelineStateWithMeshDescriptor:options:reflection:error:",
            self,
            desc
        );
        let recorded_answer = crate::recorder::run(
            || -> Result<Option<<RtPlatform as Platform>::RenderPipeline>, NsErrorText> {
                render_pipeline::of_mesh_descriptor(desc).map(Some)
            },
        );
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// rt's state of the whole descriptor (`super::pipeline`).
    fn new_compute_pipeline_state_full(
        &self,
        desc: &FullComputePipelineDesc<'_, RtPlatform>,
    ) -> Result<Option<<RtPlatform as Platform>::ComputePipeline>, NsErrorText> {
        let recorded_call = crate::recorder::rec_call!(
            "newComputePipelineStateWithDescriptor:options:reflection:error:",
            self,
            desc
        );
        let recorded_answer = crate::recorder::run(
            || -> Result<Option<<RtPlatform as Platform>::ComputePipeline>, NsErrorText> {
                pipeline::of_descriptor(self, desc).map(Some)
            },
        );
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }
}

/// The identity of `device` ([`answers::identity`]): rt's linear and texture-buffer alignments
/// of each SDK format rt has a format of, none where it refuses one.
fn identity_of(device: &RtObj<RtDevice>) -> IdentityCaps {
    answers::identity(
        &device.facts,
        &device.caps,
        |format| linear_alignment(&device.rt, format),
        |format| texture_buffer_alignment(&device.rt, format),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::platform::device::relaying_gaps;
    use crate::core::platform::values::{OptionSet, TextureDesc};

    /// Only Vulkan's `INTEGRATED_GPU` is an integrated GPU: a discrete, virtual, CPU or other
    /// device is not
    #[test]
    fn only_the_integrated_device_type_is_integrated() {
        use vk::PhysicalDeviceType as T;
        assert!(integrated(T::INTEGRATED_GPU));
        for other in [T::OTHER, T::DISCRETE_GPU, T::VIRTUAL_GPU, T::CPU] {
            assert!(!integrated(other), "device type {}", other.as_raw());
        }
    }

    /// Each fold counts what each over-copy metric rose by since the last, under its code, so the
    /// counts line carries the metrics' totals: the over-copies and their bytes, those over an
    /// imported block, and each cause's, by the cause's name.
    #[test]
    fn over_copies_fold_into_the_counts_by_their_rise() {
        let codes = [
            OVER_COPIES_CODE,
            OVER_COPY_BYTES_CODE,
            OVER_COPIES_IMPORTED_CODE,
            OVER_COPY_BYTES_IMPORTED_CODE,
            "rt-over-copies-host-memory",
            "rt-over-copy-bytes-host-memory",
            "rt-over-copies-pitch",
            "rt-over-copy-bytes-pitch",
            "rt-over-copies-image-memory",
            "rt-over-copy-bytes-image-memory",
        ];
        let counts = || {
            let all = crate::answer::degradation_counts();
            codes.map(|code| (all.iter()).find(|(c, _)| c == code).map_or(0, |&(_, n)| n))
        };
        let before = counts();
        let mut m = rt::Metrics::default();
        m.over_copies = 2;
        m.over_copy_bytes = 8192;
        m.over_copies_imported = 1;
        m.over_copy_bytes_imported = 4096;
        m.over_copies_host_memory = 1;
        m.over_copy_bytes_host_memory = 4096;
        m.over_copies_pitch = 1;
        m.over_copy_bytes_pitch = 4096;
        let mut folded = [0; OVER_COPY_METRICS];
        fold(&mut folded, over_copy_counts(&m));
        m.over_copies = 3;
        m.over_copy_bytes = 12288;
        m.over_copies_image_memory = 1;
        m.over_copy_bytes_image_memory = 4096;
        fold(&mut folded, over_copy_counts(&m));
        assert_eq!(folded, over_copy_counts(&m).map(|(_, n)| n));
        let after = counts();
        let rose: Vec<u64> = (after.iter().zip(before)).map(|(a, b)| a - b).collect();
        assert_eq!(rose, [3, 12288, 1, 4096, 1, 4096, 1, 4096, 1, 4096]);
        let named = over_copy_counts(&m).map(|(code, _)| code);
        let unique: std::collections::BTreeSet<&str> = named.iter().copied().collect();
        assert_eq!(unique.len(), OVER_COPY_METRICS, "{named:?}");
        assert!(
            named.iter().all(|code| code.starts_with("rt-over-cop")),
            "{named:?}"
        );
    }

    /// The counts read the device the process made last ([`follow`]), on the GPU the test holds:
    /// two devices made in turn, the first let go of before the second is made
    /// (`RtPlatform::let_go_of_device`), each one's submissions read as its own as it is made and
    /// after its work, the first's no more once the second is made; and the over-copies folded
    /// are both devices' ([`fold_metrics`]).
    #[test]
    fn the_counts_read_each_device_made_in_turn_on_vulkan() {
        use crate::core::platform::command::{CommandBuffer as _, CommandQueue as _};
        use crate::core::platform::device::GpuDevice as _;

        let platform = RtPlatform::new();
        let read = || metrics().map(|m| m.submissions);
        let over_copies = || {
            let all = crate::answer::degradation_counts();
            (all.iter())
                .find(|(code, _)| *code == OVER_COPIES_CODE)
                .map_or(0, |&(_, n)| n)
        };
        let counted = over_copies();
        let mut made: Vec<rt::Counters> = Vec::new();
        // Each command buffer fills a buffer, as rt submits nothing for one with no work, and
        // each commit is waited for before the next, so each is a submission of its own: the
        // first device's count differs from the second's at each read.
        for (turn, commits) in [(1, 8), (2, 1)] {
            let device = (platform.system_default_device()).expect("an rt device on the GPU");
            let counters = device.rt().counters();
            let own = || Some(counters.metrics().submissions);
            let others = || (made.iter()).map(|c| Some(c.metrics().submissions));
            assert_eq!(read(), own(), "device {turn} made: its count");
            let before = counters.metrics().submissions;
            let queue = device.new_command_queue().expect("a command queue");
            let filled = device.new_buffer_of_length(OnFault::Return, 4096, 0);
            let filled = filled.expect("never a raise").expect("a Shared buffer");
            for _ in 0..commits {
                let buffer = (queue.command_buffer(OnFault::Return))
                    .expect("never a raise")
                    .expect("a command buffer");
                let mut blit = buffer.rt().blit_command_encoder().expect("a blit encoder");
                blit.fill_buffer(filled.rt(), 0..4096, 0x5a)
                    .expect("a fill");
                blit.end_encoding();
                assert_eq!(buffer.commit(OnFault::Return), Ok(()));
                assert_eq!(buffer.wait_until_completed(OnFault::Return), Ok(()));
            }
            assert_eq!(read(), own(), "device {turn}'s work: its count");
            let rose = counters.metrics().submissions - before;
            assert!(
                rose >= commits,
                "device {turn}: {rose} submissions for {commits} commits"
            );
            assert!(
                others().all(|other| other != own()),
                "device {turn}: the devices' counts tell them apart: {:?} against {:?}",
                own(),
                others().collect::<Vec<_>>()
            );
            drop((filled, queue, device));
            assert_eq!(
                platform.let_go_of_device(std::time::Duration::from_secs(5)),
                Ok(())
            );
            made.push(counters);
        }
        fold_metrics();
        let copied: u64 = (made.iter()).map(|c| c.metrics().over_copies).sum();
        assert_eq!(over_copies() - counted, copied, "both devices' over-copies");
    }

    /// The rt platform's device on the GPU the test holds: made once and answered at every ask,
    /// its registry ID opening it again; its answers bounded by its GPU, its texture-buffer
    /// alignments rt's; its queues, events and fences made; the guest's texture indices granted,
    /// twice without a refusal; a heap texture's size rt's, and a shape rt or Metal refuses a
    /// counted raise, never 0 bytes at 0; every creator of an unbuilt piece nil and counted.
    #[test]
    fn the_device_and_its_answers_on_vulkan() {
        use crate::core::platform::device::GpuDevice as _;
        let platform = RtPlatform::new();
        let device = platform
            .system_default_device()
            .expect("an rt device on the GPU the test holds");
        let again = platform.system_default_device().expect("the same device");
        assert_eq!(again.id(), device.id());
        let caps = device.caps();
        assert_eq!(
            platform.open_device(caps.registry_id).map(|d| d.id()),
            Some(device.id())
        );
        assert!(platform.open_device(caps.registry_id ^ 1).is_none());
        let limits = device.facts().limits;
        assert!(caps.limits.max_texture_width_2d <= u64::from(limits.max_image_dimension2_d));
        assert!(caps.sample_counts[0] == Ok(true));
        let page = SizeAndAlign {
            size: device.facts().page_size,
            align: 256,
        };
        let on = OnFault::Die;
        assert_eq!(caps.heap_buffer_page_private(on), Ok(page));
        let identity = device.identity_caps();
        assert_eq!(
            identity.formats.len(),
            crate::core::platform::caps::SDK_PIXEL_FORMATS.len()
        );
        // The linear texture alignments are the largest format's, and BGRA8's is the device's
        // linear row rounding.
        let largest = (identity.formats.iter())
            .filter_map(|f| f.linear_alignment)
            .fold(16, u64::max);
        assert_eq!(caps.limits.linear_texture_alignment_bytes, largest);
        assert_eq!(caps.limits.device_linear_texture_alignment_bytes, largest);
        assert_eq!(caps.limits.iosurface_texture_alignment_bytes, largest);
        assert_eq!(
            caps.min_linear_alignment_bgra8(),
            Ok(device.linear_row_alignment())
        );
        // Each texture-buffer format answers rt's texture-buffer alignment of it, where rt
        // answers one, and none answers 0; a format Metal answers none of answers none.
        let texel_formats = crate::core::platform::caps::TEXTURE_BUFFER_FORMATS;
        for answers in &identity.formats {
            let rt_answer = texture_buffer_alignment(device.rt(), answers.format);
            let answered = answers.texture_buffer_alignment;
            assert_ne!(answered, Some(0), "format {}", answers.format);
            match texel_formats.contains(&answers.format) {
                true if rt_answer.is_some() => assert_eq!(answered, rt_answer),
                true => {}
                false => assert_eq!(answered, None, "format {}", answers.format),
            }
        }
        let texel_answers: Vec<(u64, u64)> = (identity.formats.iter())
            .filter_map(|f| Some((f.format, f.texture_buffer_alignment?)))
            .collect();
        eprintln!("texture-buffer alignments (format, bytes): {texel_answers:?}");
        assert!(device.own_memory_budget() > 0);
        assert!(device.memory_heaps_total() >= device.own_memory_budget());
        assert!(device.linear_row_alignment() > 0);
        assert!(device.new_command_queue().is_some());
        assert!(device.new_command_queue_unshared().is_some());
        assert!(matches!(device.new_event(OnFault::Return), Ok(Some(_))));
        assert!(matches!(device.new_fence(), Ok(Some(_))));
        let guests: Vec<u64> = (1..=16320).collect();
        let before = crate::answer::degradation_counts();
        assert_eq!(device.reserve_resource_indices(0, &guests), Ok(()));
        assert_eq!(device.reserve_resource_indices(0, &guests[..64]), Ok(()));
        assert_eq!(crate::answer::degradation_counts(), before);
        let sparse = HeapDesc {
            size: 0x4000,
            heap_type: 2,
            options: vec![OptionSet::StorageMode(2)],
            pinned_gpu_address: None,
        };
        let (made, relayed) = relaying_gaps(|| device.new_heap(&sparse));
        assert!(made.is_ok_and(|made| made.heap.is_none()) && relayed);
        assert!(device.new_library(&[]).is_ok_and(|made| made.is_err()));
        let texture = platform.texture_descriptor(&TextureDesc {
            pixel_format: 80,
            width: 64,
            height: 64,
            resource_options: Some(0x20),
            ..TextureDesc::default()
        });
        let sized = device.heap_texture_size_and_align(&texture);
        let Ok(SizeAndAlign { size, align }) = sized else {
            panic!("no size: {sized:?}");
        };
        assert!(
            size >= 64 * 64 * 4 && align.is_power_of_two(),
            "{size} at {align}"
        );
        let counted = |code: &str| {
            (crate::answer::degradation_counts().into_iter())
                .find(|(c, _)| c == code)
                .map_or(0, |(_, n)| n)
        };
        let metal = RtRefusal::Refused.code();
        let before = counted(metal);
        let refused = [
            TextureDesc {
                texture_type: 99,
                ..TextureDesc::default()
            },
            TextureDesc {
                width: 0,
                ..TextureDesc::default()
            },
        ];
        for shape in refused {
            let descriptor = platform.texture_descriptor(&shape);
            let asked = device.heap_texture_size_and_align(&descriptor);
            assert!(asked.is_err(), "{shape:?}: {asked:?}");
        }
        assert_eq!(counted(metal), before + 2);

        let bgra8 = device.pixel_format_info(80);
        assert_eq!(bgra8.name.as_deref(), Some("MTLPixelFormatBGRA8Unorm"));
        let q = TextureLevelQuery {
            pixel_format: 80,
            width: 64,
            height: 32,
            depth: 1,
            array_length: 1,
            ..TextureLevelQuery::default()
        };
        let level = device.texture_level_info(&q);
        assert_eq!((level.bytes_per_row, level.aligned_height), (256, 32));
    }
}
