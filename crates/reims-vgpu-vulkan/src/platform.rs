//! `VulkanPlatform`: `impl Platform` over `metal2vulkan-runtime`, and [`Obj`], the handle every
//! associated type is.
//!
//! Every handle is an [`Obj<T>`]: one counted reference to a value of this crate, which holds the
//! runtime's object (one reference of the runtime's own) and what core's trait asks of the object
//! beside it. Not `Clone` (D6): `.clone_ref()` is the retain and `Drop` the release, so the
//! runtime's object goes when core's last handle to it does. `id()` is the
//! reference's address: the same for every reference to one object, as a Metal object's pointer
//! is.
//!
//! One `VulkanPlatform` is one process's (D17).

use std::fmt;
use std::mem::ManuallyDrop;
use std::path::PathBuf;
use std::sync::{Arc, LazyLock, OnceLock, Weak};

use metal2vulkan_runtime as rt;
use reims_vgpu_core::num::ToU64;
use reims_vgpu_core::platform::os::Threads;
use reims_vgpu_core::platform::os::surface::IoSurfaces;
use reims_vgpu_core::platform::os::vm::HostVm;
use reims_vgpu_core::platform::types::{MtlSize, ResourceAddressRange};
use reims_vgpu_core::platform::values::{RateMapDesc, TextureDesc};
use reims_vgpu_core::platform::{Handle, ObjId, Platform};
use reims_vgpu_core::raise::{OnFault, RaiseResult};

use crate::answer::{get, plain, plain_at, refused};
use crate::gpu::command::{VulkanCommandBuffer, VulkanCommandQueue};
use crate::gpu::desc::{
    RenderPassDescriptorObject, TextureDescriptorObject, TextureFields, VulkanRenderPassDescriptor,
    VulkanTextureDescriptor,
};
use crate::gpu::device::{self, VulkanDevice};
use crate::gpu::encode_blit::VulkanBlitEncoder;
use crate::gpu::encode_compute::VulkanComputeEncoder;
use crate::gpu::encode_render::VulkanRenderEncoder;
use crate::gpu::event::{
    VulkanEvent, VulkanSharedEvent, VulkanSharedEventHandle, VulkanSharedEventListener,
};
use crate::gpu::icb::{VulkanIcb, VulkanIndirectComputeCommand, VulkanIndirectRenderCommand};
use crate::gpu::rate_map::{VulkanRateMap, VulkanRateMapDescriptor, VulkanRateMapLayerDescriptor};
use crate::gpu::resource::{
    VulkanBuffer, VulkanComputePipeline, VulkanDepthStencilState, VulkanFence, VulkanFunction,
    VulkanHeap, VulkanLibrary, VulkanRenderPipeline, VulkanSamplerState, VulkanSharedTextureHandle,
    VulkanTexture,
};
use crate::os::{OsClock, OsSurface, OsSurfaces, OsThreads};
use crate::remap::VulkanVm;

/// One counted reference to an object of this platform (architecture 4.1).
///
/// Not `Clone` (D6): [`Handle::clone_ref`] is the only retain, and `Drop` the release.
pub struct Obj<T: Send + Sync + 'static>(ManuallyDrop<Arc<T>>);

impl<T: Send + Sync + 'static> Obj<T> {
    /// A new object holding `value`: its first reference.
    pub fn new(value: T) -> Obj<T> {
        Obj(ManuallyDrop::new(Arc::new(value)))
    }

    /// A reference that does not hold the object.
    pub fn downgrade(&self) -> Weak<T> {
        Arc::downgrade(&self.0)
    }

    /// Another reference to the object `weak` names, while one lives.
    pub fn upgrade(weak: &Weak<T>) -> Option<Obj<T>> {
        weak.upgrade().map(|arc| Obj(ManuallyDrop::new(arc)))
    }
}

impl<T: Send + Sync + 'static> Drop for Obj<T> {
    /// The release.
    fn drop(&mut self) {
        // SAFETY: the reference is released once, here, and never touched after.
        unsafe { ManuallyDrop::drop(&mut self.0) }
    }
}

impl<T: Send + Sync + 'static> std::ops::Deref for Obj<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T: Send + Sync + fmt::Debug + 'static> fmt::Debug for Obj<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<{:#x} {:?}>", self.id().0, **self.0)
    }
}

impl<T: Send + Sync + 'static> Handle for Obj<T> {
    fn clone_ref(&self) -> Self {
        Obj(ManuallyDrop::new(Arc::clone(&self.0)))
    }

    fn id(&self) -> ObjId {
        ObjId(Arc::as_ptr(&self.0).cast::<u8>().addr().to_u64())
    }
}

/// The runtime listener every listener of the platform is a handle of. The runtime starts a thread
/// for each listener it makes as its first notification is asked for (`SharedEventListener::new`,
/// read from code), which ends once its last handle goes. Core makes one listener for each task:
/// one listener of the process's life holds at most one such thread. Each handle is still an
/// object of its own (`id`).
static LISTENER: OnceLock<rt::SharedEventListener> = OnceLock::new();

/// The process's platform: the runtime's GPU objects, and the OS's IOSurface, VM (reporting the
/// pages it changes, [`crate::remap`]), clocks and threads ([`crate::os`]).
pub struct VulkanPlatform {
    surfaces: OsSurfaces,
    vm: VulkanVm,
    clock: OsClock,
    threads: OsThreads,
}

impl Default for VulkanPlatform {
    fn default() -> VulkanPlatform {
        VulkanPlatform::new()
    }
}

impl fmt::Debug for VulkanPlatform {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VulkanPlatform").finish_non_exhaustive()
    }
}

impl VulkanPlatform {
    /// The process's platform. Make one per process (D17). It installs what the OS's services need
    /// first ([`crate::os::install_process_services`]): on macOS the Objective-C autorelease
    /// pool core's pools push, as IOSurface and Foundation still autorelease into it.
    pub fn new() -> VulkanPlatform {
        crate::os::install_process_services();
        VulkanPlatform {
            surfaces: OsSurfaces::new(),
            vm: VulkanVm::new(),
            clock: OsClock::new(),
            threads: OsThreads::new(),
        }
    }

    /// The runtime's system default device (`MTLCreateSystemDefaultDevice`), for tests; `None`
    /// where the runtime finds no Vulkan device it can use, or no Vulkan loader. Any other
    /// refusal dies here, naming it ([`made_devices`]).
    pub fn system_default_device() -> Option<VulkanDevice> {
        made_devices("MTLCreateSystemDefaultDevice()", || {
            rt::Device::system_default_with(device_options()).map(|d| vec![d])
        })
        .pop()
        .map(device::adopt)
    }
}

/// The devices `make` makes, a creator standing for `sel` (`MTLCreateSystemDefaultDevice()`,
/// `MTLCopyAllDevices()`): none where the runtime finds no Vulkan device it can use, or no Vulkan
/// loader. Any other refusal dies at `sel`, naming it. A device that lists a window extension the
/// options name but lacks its feature (`Error::ExtensionFeature`, `swapchainMaintenance1`) dies at `sel`
/// too, naming the extension and the feature: the runtime refuses it rather than enable the
/// extension without the feature, and the window's probe names the extension only where every
/// device has the feature, asked as the runtime asks ([`crate::os::add_window_extensions`]). So
/// do options that name a device extension without an instance extension it requires
/// (`Error::InstanceExtensionRequired`), which the runtime checks before it makes the instance:
/// the probe names each window extension with those it requires.
fn made_devices(
    sel: &'static str,
    make: impl FnOnce() -> Result<Vec<rt::Device>, rt::Error>,
) -> Vec<rt::Device> {
    get(sel, || match make() {
        Err(rt::Error::NoDevice | rt::Error::LoaderNotFound(_)) => Ok(Vec::new()),
        made => made,
    })
}

/// The creator [`Platform::open_device`] stands for, which its deaths name.
const OPEN_DEVICE: &str = "MTLCopyAllDevices()";

/// The device of registry ID `registry_id` ([`device::registry_id`]) that an open answers
/// ([`Platform::open_device`]): the device of that ID the process adopted, where there is one,
/// with no runtime call; else the runtime's system default device, of those `default` makes
/// (none where the runtime refuses it, whose refusal `every` then decides), where its ID is
/// `registry_id`; else the one of that ID among those `every` makes, the runtime's
/// `Device::all_with`, which makes a device of every physical device there is. Each found is
/// adopted ([`device::adopt`]).
///
/// So `every` runs only for an ID that is neither the default device's nor any adopted one's. A
/// guest's every task opens its device at its handshake (`gpu_task/handshake.rs`), and a device of
/// the runtime's lives only while a handle to it does: each open through `Device::all_with` made,
/// primed and tore down a device on every GPU the process does not use (on a host of two GPUs,
/// once per guest task), and died there when one of those refused, at an open of the GPU it uses.
fn opened(
    registry_id: u64,
    default: impl FnOnce() -> Vec<rt::Device>,
    every: impl FnOnce() -> Vec<rt::Device>,
) -> Option<VulkanDevice> {
    if let Some(kept) = device::kept(registry_id) {
        return Some(kept);
    }
    let of_id = |made: Vec<rt::Device>| {
        made.into_iter()
            .find(|d| device::registry_id(d) == registry_id)
    };
    of_id(default())
        .or_else(|| of_id(every()))
        .map(device::adopt)
}

/// A device of the process, for an answer every device gives alike, at the one macOS release
/// whose Metal the runtime answers as (`rt::MACOS_RELEASE`): a pixel format's description and a
/// texture level's layout ([`crate::gpu::format_info`]). The first device the platform adopted;
/// asked before any is, the runtime's system default device, made for it, which dies as
/// [`made_devices`] does where the runtime makes none, at `sel`, the call that asks. Kept once
/// found, as the platform keeps every device it adopts.
pub(crate) fn answering_device(sel: &'static str) -> &'static rt::Device {
    static DEVICE: OnceLock<rt::Device> = OnceLock::new();
    if let Some(kept) = DEVICE.get() {
        return kept;
    }
    let device = device::first_adopted().unwrap_or_else(|| {
        let made = made_devices(sel, || {
            rt::Device::system_default_with(device_options()).map(|d| vec![d])
        });
        made.into_iter()
            .next()
            .unwrap_or_else(|| refused(sel, &rt::Error::NoDevice))
    });
    DEVICE.get_or_init(|| device)
}

/// The options every device is made with: the runtime's defaults, whose loader search finds
/// Homebrew's (`vulkan-loader`, beside its `molten-vk`), with the window system's extensions where
/// a display's window presents through Vulkan (on Linux, [`crate::os::add_window_extensions`]), the
/// shader cache in
/// `~/.cache/reims-vgpu-vulkan` ([`SHADER_CACHE`]; the runtime's default directory when `HOME` is
/// not set), which every process shares, holding [`SHADER_CACHE_CAPACITY`], each pipeline state's
/// record kept in the file beside it for later processes' warm passes ([`crate::records`]; none
/// in the unit tests), a command buffer's
/// GPU times answered from the host's clock (`gpu_times_from_host`), the runtime's device policy,
/// its fast native forms but its exact completion and heap textures and those `REIMS_VGPU_EXACT`
/// names ([`exact_policy`]), and in a strict process ([`crate::answer::strict_process`]) its
/// exact pipeline wait, under which no draw or dispatch is dropped while its pipeline is still
/// being made (the runtime's `DrawDroppedStillMaking` and `DispatchDroppedStillMaking`, kinds of
/// [`crate::answer::DIES`]), and the degradations the
/// runtime announces sent to the answer layer ([`crate::answer::notice`]), where those that reach
/// what the guest sees are counted, and die in strict mode ([`crate::answer::strict`]), with the
/// kinds announced at every occurrence ([`announced_each`]), its other messages there too but
/// those below the least severity the log takes ([`crate::answer::least_severity`]). One set, so
/// the runtime keeps one device per physical device across `open_device` and every other open.
///
/// **GPU times from the host.** No command buffer writes GPU timestamps: its `GPUStartTime` is the
/// host time its submission ended (`kernelEndTime`, or its queue's latest GPU end before it,
/// where later, so a queue's command buffers do not overlap) and its `GPUEndTime` the host time
/// the runtime saw it complete. On MoltenVK 1.4.1 the runtime answers so whatever the option, as
/// its timestamps wait for every queue's (the runtime's `TimestampsWaitForEveryQueue`). A
/// completed handler learns them (core's `CompletedInfo::gpu_times_ns`), and the graphics lane
/// posts them to its guest, whose bundle answers the run's length from them; this platform's
/// debug lines read them too ([`crate::gpu::timing`]). The apple-gfx lane's host sees none. The
/// timestamps cost latency:
/// measured on the M3 Max (a release build, under a load average of 34), a command buffer of one
/// blit fill completed a median of 283 to 309 us after its commit, against 356 to 399 us with
/// them (`timing`'s `gpu_times_from_host_latency`).
/// The kinds the runtime announces at every occurrence (`DeviceOptions::announce_each`): those a
/// call here hears ([`crate::answer::HEARD`]) and those that die by default
/// ([`crate::answer::DIES_BY_DEFAULT`]), `CopiedHostPages`, which the device's probe of its imports
/// may meet unheard as it is made ([`crate::answer`]'s `unheard`, on Linux), and every other kind
/// of [`crate::answer::DIES`] where `every_dies`, in strict and census mode, which die at or record
/// each occurrence. Elsewhere the
/// runtime announces such a kind once per device, its first occurrence, the only one the default
/// writes: asked for each, a pipeline variant's notices reach the sink again at every draw's bind,
/// a cost on the encoding thread (in a Finder drag on ms01's Iris Xe, 2026-10-18, the sink
/// itself took 0.09 ms per frame of the FIFO thread's 16.3, measured).
fn announced_each(every_dies: bool) -> std::collections::BTreeSet<rt::Degradation> {
    let dies: &[rt::Degradation] = if every_dies {
        &crate::answer::DIES
    } else {
        &crate::answer::DIES_BY_DEFAULT
    };
    (dies.iter().copied())
        .chain(crate::answer::HEARD)
        .chain([rt::Degradation::CopiedHostPages])
        .collect()
}

pub(crate) fn device_options() -> &'static rt::DeviceOptions {
    static OPTIONS: LazyLock<rt::DeviceOptions> = LazyLock::new(|| {
        let mut options = rt::DeviceOptions::default();
        crate::os::add_window_extensions(&mut options);
        options.gpu_times_from_host = true;
        options.policy = exact_policy(std::env::var("REIMS_VGPU_EXACT").ok().as_deref());
        if crate::answer::strict_process() {
            options.policy.pipeline_waits = metal2vulkan::device::PipelineWaitPolicy::Exact;
        }
        let sink = rt::MessageSink::new(crate::answer::notice);
        options.messages = Some(sink.with_least_severity(crate::answer::least_severity()));
        options.announce_each =
            announced_each(crate::answer::strict_process() || crate::answer::census::on());
        if std::env::var_os(reims_vgpu_core::trace::ENV_VAR).is_some() {
            reims_vgpu_core::trace::set_clock(crate::gpu::timing::trace_clock);
            if crate::answer::trace_degradations() && reims_vgpu_core::trace::on() {
                // Every occurrence reaches the sink, which writes each to the trace and relays
                // a kind the runtime announces once only the first time: only where the trace's
                // file opened, the sink's own condition, so a refused file changes nothing.
                options.announce_each = rt::Degradation::ALL.iter().copied().collect();
            }
        }
        if let Some(home) = std::env::var_os("HOME") {
            options.shader_cache =
                rt::ShaderCacheOptions::in_directory(PathBuf::from(&home).join(SHADER_CACHE));
            #[cfg(not(test))]
            {
                options.pipeline_records =
                    crate::records::hook(crate::records::path_in(home.as_ref()));
            }
        }
        options.shader_cache.capacity = SHADER_CACHE_CAPACITY;
        options
    });
    &OPTIONS
}

/// The fields of the runtime's device policy (`metal2vulkan::device::DevicePolicy`) whose exact
/// form `REIMS_VGPU_EXACT` can name, each by the field's own name: each batch of a queue ordered
/// after the earlier ones by the accesses it depends on (`batch_order`,
/// `BatchOrderPolicy::Exact`), a shadow, copied in before each use and merged back after each
/// write, for every texture over the guest's pages where a texture that only samples or reads
/// them would be a linear image in place on a driver whose own first-move check passes
/// (`host_page_textures`, `HostPageTexturePolicy::Shadow`), and, on a device without
/// `shaderFloat16`, each `half` addition, subtraction and product of a fast-math chain rounded
/// where it is computed, where the default rounds the chain where a value leaves it
/// (`half_chains`, `HalfChainPolicy::EachOperation`), a command buffer that waits for each
/// pipeline the driver is still making until it is made, where the default waits 4.5 s from its
/// commit and then drops the draws and dispatches whose pipeline is not made (`pipeline_waits`,
/// `PipelineWaitPolicy::Exact`), and, on a device with `tessellationShader`, a line `setLineWidth:`
/// widens drawn as Metal's parallelogram, two triangles over each segment, where the default draws
/// the device's own line at the width wherever it draws it, which can differ from Metal's
/// (`wide_lines`, `WideLinePolicy::Exact`), and a blit into or out of a `BGRA10_XR` texture Metal
/// keeps compressed rewritten to Metal's encoding (each 16-bit word's low 6 bits clear, alpha's
/// code within 384 to 894) and a fill's texel encoded so, where the default keeps the bytes the
/// device's copy moves (`extended_range_blits`, `ExtendedRangeBlitPolicy::Canonical`), and every
/// command queue on the `VkQueue` it took at its creation, where the default lets command queues
/// share `VkQueue`s on a device whose own check finds its GPU idle at each switch between queues,
/// so a command buffer can wait on the GPU behind another queue's (`queue_sharing`,
/// `QueueSharingPolicy::PerQueue`), and a colour attachment a pass stores `DontCare` keeping the
/// texels it held before the pass, as Apple's GPU leaves them, rendered through a backing its
/// command queue keeps, where the default stores it as the driver stores it, announcing
/// `dont-care-store` where that is not Metal's own (`dont_care_colors`,
/// `DontCareColorPolicy::Kept`). [`EXACT_ALL`] names every one. The
/// policy's `completion` and `heap_compression` are no names of these: every device takes their
/// exact forms ([`exact_policy`]). The rest of Metal's values the runtime answers with the
/// device's own arithmetic, sampling and conversions, each difference a precision note it
/// records, with no exact form to name.
pub const EXACT_FORMS: [&str; 8] = [
    "batch_order",
    "host_page_textures",
    "half_chains",
    "pipeline_waits",
    "wide_lines",
    "extended_range_blits",
    "queue_sharing",
    "dont_care_colors",
];

/// The name `REIMS_VGPU_EXACT` takes for the runtime's `DevicePolicy::exact_answers()`: every
/// field of [`EXACT_FORMS`] in its exact form, as the runtime's validation sets them.
pub const EXACT_ALL: &str = "all";

/// The device policy every device is made with: the runtime's default, whose native forms are
/// fast to make and run and record their bounds off Apple's GPUs (docs/accuracy-ledger.md: the
/// fast method by default, the accurate one behind a switch), but for its completion, which is
/// always the exact form (`CompletionPolicy::Exact`): each command buffer goes to the driver in a
/// submission of its own and is seen complete at its own end, as Metal's, where the runtime's
/// default would take up to four ready together in one kernel submission on Mesa's drivers and
/// see each but the last complete only once the last completed. When the host sees a command
/// buffer complete is the guest's to observe (the stamps and interrupts core writes as its work
/// completes), and the grouped form bought no frame rate in a guest's drag on the Iris Xe (15.7
/// to 17.3 frames a second, against 16.4 to 17.6 exact), so no switch takes it. Its heap
/// compression is always the exact form too, with no switch either: a heap's colour texture Metal
/// keeps losslessly compressed takes an image the driver keeps uncompressed in the heap's bytes
/// (`HeapCompressionPolicy::Uncompressed`), as one Metal keeps uncompressed does, where the
/// runtime's default would take the driver's compressed image on a driver whose compressed images
/// may keep their texels apart from their bytes (ANV). That image changes what
/// `heapTextureSizeAndAlignWithDescriptor:` answers the guest (on the Iris Xe, 0x4100 bytes at
/// 0x10000 for a 64x64 `BGRA8Unorm` texture, where the uncompressed one answers 0x4080 at 0x1000,
/// and Metal 0x4080 at 0x80), and a write through a buffer or texture over its bytes may not show
/// in its texels. Each field of [`EXACT_FORMS`] that `exact`, `REIMS_VGPU_EXACT`'s comma-separated
/// list, names takes the runtime's exact form too, and [`EXACT_ALL`] every such form:
/// `batch_order`'s barriers between batches, `host_page_textures`' copies of each sampled texture
/// over the guest's pages at each command buffer's first use, and `half_chains`' rounding at
/// each `half` operation on a device without `shaderFloat16`, `pipeline_waits`' wait for every
/// pipeline until it is made, `wide_lines`' band variant of each pipeline that draws a line
/// the device's own lines would draw otherwise than Metal, and `extended_range_blits`' rewrite of
/// each extended-range blit to Metal's encoding. A name of neither changes nothing,
/// and a line on stderr says so.
fn exact_policy(exact: Option<&str>) -> metal2vulkan::device::DevicePolicy {
    use metal2vulkan::device::{
        BatchOrderPolicy, CompletionPolicy, DevicePolicy, DontCareColorPolicy,
        ExtendedRangeBlitPolicy, HalfChainPolicy, HeapCompressionPolicy, HostPageTexturePolicy,
        PipelineWaitPolicy, QueueSharingPolicy, WideLinePolicy,
    };
    let mut policy = DevicePolicy::default();
    policy.completion = CompletionPolicy::Exact;
    policy.heap_compression = HeapCompressionPolicy::Uncompressed;
    for name in (exact.unwrap_or_default().split(',').map(str::trim)).filter(|n| !n.is_empty()) {
        match name {
            EXACT_ALL => policy = policy.exact_answers(),
            "batch_order" => policy.batch_order = BatchOrderPolicy::Exact,
            "host_page_textures" => policy.host_page_textures = HostPageTexturePolicy::Shadow,
            "half_chains" => policy.half_chains = HalfChainPolicy::EachOperation,
            "pipeline_waits" => policy.pipeline_waits = PipelineWaitPolicy::Exact,
            "wide_lines" => policy.wide_lines = WideLinePolicy::Exact,
            "extended_range_blits" => {
                policy.extended_range_blits = ExtendedRangeBlitPolicy::Canonical;
            }
            "queue_sharing" => policy.queue_sharing = QueueSharingPolicy::PerQueue,
            "dont_care_colors" => policy.dont_care_colors = DontCareColorPolicy::Kept,
            other => {
                let line = format!(
                    "reims-vgpu: REIMS_VGPU_EXACT names {other:?}, none of {EXACT_ALL}, {}: \
                     ignored\n",
                    EXACT_FORMS.join(", ")
                );
                let _ = std::io::Write::write_all(&mut std::io::stderr(), line.as_bytes());
            }
        }
    }
    policy
}

/// The shader cache's directory under `HOME`, which every process of the library shares. It
/// holds the runtime's translations of the guest's functions and each pipeline's driver record
/// (MoltenVK's included: about 1.7 MB for a macOS 26 guest's `Xgc` pipeline
/// on NVIDIA), within [`SHADER_CACHE_CAPACITY`], the least recently used going first.
#[cfg(not(test))]
const SHADER_CACHE: &str = ".cache/reims-vgpu-vulkan";

/// The most bytes [`SHADER_CACHE`] holds: the runtime's default, 25 GiB (a test keeps it so). A
/// macOS 26 guest's built-in apps, each launched once, need more than 1 GiB of translations and
/// driver records on an RTX 5080 laptop. The runtime seeds each pipeline's making from its own
/// record here; the driver's own disk cache is left as the driver keeps it, and may
/// hold the pipeline too. An entry this cache drops past its capacity is compiled again at the
/// next boot that draws with it, unless the driver's cache still holds it: a freeze of 13 to 26 s
/// there.
#[cfg(not(test))]
const SHADER_CACHE_CAPACITY: u64 = rt::DEFAULT_CAPACITY;

/// The unit tests' capacity, 1 GiB: their directory holds what the tests compile and nothing a
/// guest's boot reuses, so it is kept small on disk.
#[cfg(test)]
const SHADER_CACHE_CAPACITY: u64 = 1 << 30;

/// The unit tests' shader cache directory under `HOME`, their own: a test of what a device
/// compiles runs on an empty one as on a full one, and the tests write nothing into the one the
/// library's hosts share.
#[cfg(test)]
const SHADER_CACHE: &str = ".cache/reims-vgpu-vulkan-tests";

/// The runtime's registry ID of its system default device, read once: `None` where it has none.
pub(crate) fn default_device_registry_id() -> Option<u64> {
    static DEFAULT: LazyLock<Option<u64>> = LazyLock::new(|| {
        rt::Device::system_default_with(device_options())
            .ok()
            .map(|d| d.registry_id())
    });
    *DEFAULT
}

impl Platform for VulkanPlatform {
    type Device = VulkanDevice;
    type CommandQueue = VulkanCommandQueue;
    type CommandBuffer = VulkanCommandBuffer;
    type BlitEncoder = VulkanBlitEncoder;
    type ComputeEncoder = VulkanComputeEncoder;
    type RenderEncoder = VulkanRenderEncoder;
    type Buffer = VulkanBuffer;
    type Texture = VulkanTexture;
    type Heap = VulkanHeap;
    type SamplerState = VulkanSamplerState;
    type DepthStencilState = VulkanDepthStencilState;
    type Fence = VulkanFence;
    type Event = VulkanEvent;
    type SharedEvent = VulkanSharedEvent;
    type SharedEventHandle = VulkanSharedEventHandle;
    type SharedEventListener = VulkanSharedEventListener;
    type SharedTextureHandle = VulkanSharedTextureHandle;
    type Library = VulkanLibrary;
    type Function = VulkanFunction;
    type ComputePipeline = VulkanComputePipeline;
    type RenderPipeline = VulkanRenderPipeline;
    type RateMap = VulkanRateMap;
    type Icb = VulkanIcb;
    type IndirectRenderCommand = VulkanIndirectRenderCommand;
    type IndirectComputeCommand = VulkanIndirectComputeCommand;
    type TextureDescriptor = VulkanTextureDescriptor;
    type RenderPassDescriptor = VulkanRenderPassDescriptor;
    type RateMapDescriptor = VulkanRateMapDescriptor;
    type RateMapLayerDescriptor = VulkanRateMapLayerDescriptor;
    type AddressRangeArray = VulkanAddressRangeArray;
    type Surface = OsSurface;
    type Clock = OsClock;

    /// A descriptor of Metal's defaults.
    fn new_render_pass_descriptor(&self) -> VulkanRenderPassDescriptor {
        Obj::new(plain(RenderPassDescriptorObject::new))
    }

    /// A handle of the process's one runtime listener ([`LISTENER`]), made at the first call.
    fn new_shared_event_listener(&self) -> VulkanSharedEventListener {
        Obj::new(
            LISTENER
                .get_or_init(|| plain_at("new", rt::SharedEventListener::new))
                .clone(),
        )
    }

    /// A descriptor of Metal's defaults, as `+new` makes.
    fn render_pass_descriptor(&self) -> VulkanRenderPassDescriptor {
        Obj::new(plain(RenderPassDescriptorObject::new))
    }

    /// A 2D descriptor with the full chain of levels when `mipmapped`, its pixel format taken as
    /// `setPixelFormat:` takes it: one the runtime lacks is kept raw, and a texture of it dies
    /// where it is made ([`crate::gpu::desc::lacked`]).
    fn texture_2d_descriptor(
        &self,
        _on: OnFault,
        pixel_format: u64,
        width: u64,
        height: u64,
        mipmapped: bool,
    ) -> RaiseResult<VulkanTextureDescriptor> {
        let sel = "texture2DDescriptorWithPixelFormat:width:height:mipmapped:";
        let (format, no_texture_format, lacked_format) =
            crate::gpu::desc::format_set(sel, pixel_format);
        let desc = plain(|| rt::TextureDescriptor::texture_2d(format, width, height, mipmapped));
        Ok(Obj::new(TextureDescriptorObject::new(TextureFields {
            desc,
            no_texture_format,
            lacked_format,
            ..TextureFields::default()
        })))
    }

    /// A layer of zeroed samples, nil for a count of 0 on either axis
    /// ([`crate::gpu::rate_map`]).
    fn new_rate_map_layer(
        &self,
        _on: OnFault,
        sample_count: MtlSize,
    ) -> RaiseResult<Option<VulkanRateMapLayerDescriptor>> {
        Ok(crate::gpu::rate_map::new_layer(sample_count))
    }

    /// A copy of `ranges`, as Metal's array copies them.
    fn new_address_range_array(
        &self,
        ranges: &[ResourceAddressRange],
    ) -> RaiseResult<VulkanAddressRangeArray> {
        Ok(Obj::new(AddressRanges {
            ranges: ranges.to_vec(),
        }))
    }

    /// A new descriptor with the value's properties, set in its order.
    fn texture_descriptor(&self, desc: &TextureDesc) -> VulkanTextureDescriptor {
        Obj::new(TextureDescriptorObject::new(TextureFields::of(desc)))
    }

    fn rate_map_descriptor(&self, desc: &RateMapDesc) -> VulkanRateMapDescriptor {
        crate::gpu::rate_map::new_descriptor(desc)
    }

    /// The device of the runtime's `Device::all_with` (`MTLCopyAllDevices`) whose registry ID, as
    /// this platform answers it ([`device::registry_id`]: Metal's for the default device), is
    /// `registry_id` (+1), or `None`, as where there is no Vulkan loader. In one process it is the
    /// object every other handle to that device is ([`device::adopt`]), asked for no further than
    /// it must be ([`opened`]): a device made before is that object, answered with no runtime call
    /// ([`device::kept`]).
    fn open_device(&self, registry_id: u64) -> Option<VulkanDevice> {
        opened(
            registry_id,
            || {
                made_devices(OPEN_DEVICE, || {
                    let default = rt::Device::system_default_with(device_options());
                    Ok(default.ok().into_iter().collect())
                })
            },
            || made_devices(OPEN_DEVICE, || rt::Device::all_with(device_options())),
        )
    }

    fn surfaces(&self) -> &dyn IoSurfaces<VulkanPlatform> {
        &self.surfaces
    }

    fn vm(&self) -> &dyn HostVm {
        &self.vm
    }

    fn clock(&self) -> &OsClock {
        &self.clock
    }

    fn threads(&self) -> &dyn Threads {
        &self.threads
    }
}

/// The private `MTLResourceAddressRangeArray *`: the ranges, as the array copies them, which a
/// heap is made over and a detached buffer or heap is replaced by ([`crate::gpu::backing`]).
#[derive(Debug)]
pub struct AddressRanges {
    /// The ranges, in order.
    pub ranges: Vec<ResourceAddressRange>,
}

/// The private `MTLResourceAddressRangeArray *`.
pub type VulkanAddressRangeArray = Obj<AddressRanges>;

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::gpu::device::tests::dev;

    /// The kinds a call hears, those that die by default and `CopiedHostPages`, which the
    /// device's probe of its imports meets unheard, are announced at every occurrence in every
    /// mode, and every other kind of `DIES` only in strict and census mode
    #[test]
    fn only_strict_and_census_mode_hear_every_occurrence_of_every_kind_that_dies() {
        use crate::answer::{DIES, DIES_BY_DEFAULT, HEARD};
        let each = announced_each(true);
        assert!(DIES.iter().chain(&HEARD).all(|k| each.contains(k)));
        let by_default = announced_each(false);
        let copied = rt::Degradation::CopiedHostPages;
        assert!(
            (HEARD.iter().chain(&DIES_BY_DEFAULT).chain([&copied])).all(|k| by_default.contains(k))
        );
        let once = DIES
            .iter()
            .filter(|k| !DIES_BY_DEFAULT.contains(k) && !HEARD.contains(k) && **k != copied);
        assert!(once.clone().count() > 100, "most kinds of DIES");
        assert!(once.clone().all(|k| !by_default.contains(k)));
    }

    pub(crate) fn platform() -> VulkanPlatform {
        VulkanPlatform::new()
    }

    /// Each listener the platform makes is an object of its own, and every one runs what is posted
    /// to it on the one thread of the process's runtime listener, where the runtime would start a
    /// thread for each
    #[test]
    fn every_listener_runs_its_notifications_on_the_one_thread_of_the_process() {
        let p = platform();
        let listeners: Vec<_> = (0..4).map(|_| p.new_shared_event_listener()).collect();
        let mut ids: Vec<u64> = listeners.iter().map(|l| l.id().0).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), listeners.len());
        let event = dev().raw.new_shared_event().unwrap();
        let (ran, threads) = std::sync::mpsc::channel();
        for listener in &listeners {
            let ran = ran.clone();
            event.notify_listener(
                listener,
                0,
                Box::new(move |_, _| ran.send(std::thread::current().id()).unwrap()),
            );
        }
        let threads: Vec<_> = threads.iter().take(listeners.len()).collect();
        assert!(threads.iter().all(|&t| t == threads[0]), "{threads:?}");
        assert_ne!(threads[0], std::thread::current().id());
    }

    /// Without `REIMS_VGPU_EXACT` the device policy is the runtime's default, its native forms,
    /// but for its exact completion and uncompressed heap textures, which every device takes and
    /// no name selects; each name of `EXACT_FORMS` takes that field's exact form alone, in any
    /// order, spaces around a name ignored; `all` takes the runtime's `exact_answers()`, every
    /// one; another name, `completion`, `heap_compression` and `sampling` among them, changes
    /// nothing
    #[test]
    fn reims_vgpu_exact_names_the_policys_exact_forms() {
        use metal2vulkan::device::{
            BatchOrderPolicy, CompletionPolicy, DevicePolicy, DontCareColorPolicy,
            ExtendedRangeBlitPolicy, HalfChainPolicy, HeapCompressionPolicy, HostPageTexturePolicy,
            PipelineWaitPolicy, QueueSharingPolicy, WideLinePolicy,
        };
        let runtimes = DevicePolicy::default();
        assert_eq!(runtimes.completion, CompletionPolicy::Grouped);
        assert_eq!(runtimes.heap_compression, HeapCompressionPolicy::Compressed);
        let mut fast = runtimes;
        fast.completion = CompletionPolicy::Exact;
        fast.heap_compression = HeapCompressionPolicy::Uncompressed;
        assert_eq!(fast.batch_order, BatchOrderPolicy::Ends);
        assert_eq!(fast.host_page_textures, HostPageTexturePolicy::InPlace);
        assert_eq!(fast.half_chains, HalfChainPolicy::AtExits);
        assert_eq!(fast.pipeline_waits, PipelineWaitPolicy::Bounded);
        assert_eq!(fast.wide_lines, WideLinePolicy::Device);
        assert_eq!(fast.extended_range_blits, ExtendedRangeBlitPolicy::Kept);
        assert_eq!(fast.queue_sharing, QueueSharingPolicy::Adaptive);
        assert_eq!(fast.dont_care_colors, DontCareColorPolicy::Driver);
        for unchanged in [
            None,
            Some(""),
            Some(","),
            Some("exact, bicubic, All"),
            Some("padded_reads"),
            Some("completion"),
            Some("heap_compression"),
            Some("sampling"),
        ] {
            assert_eq!(exact_policy(unchanged), fast, "{unchanged:?}");
        }
        let one = |name| exact_policy(Some(name));
        type Set = fn(&mut DevicePolicy);
        let alone: [(&str, Set); 8] = [
            ("batch_order", |p| p.batch_order = BatchOrderPolicy::Exact),
            ("host_page_textures", |p| {
                p.host_page_textures = HostPageTexturePolicy::Shadow;
            }),
            ("half_chains", |p| {
                p.half_chains = HalfChainPolicy::EachOperation;
            }),
            ("pipeline_waits", |p| {
                p.pipeline_waits = PipelineWaitPolicy::Exact;
            }),
            ("wide_lines", |p| p.wide_lines = WideLinePolicy::Exact),
            ("extended_range_blits", |p| {
                p.extended_range_blits = ExtendedRangeBlitPolicy::Canonical;
            }),
            ("queue_sharing", |p| {
                p.queue_sharing = QueueSharingPolicy::PerQueue;
            }),
            ("dont_care_colors", |p| {
                p.dont_care_colors = DontCareColorPolicy::Kept;
            }),
        ];
        assert_eq!(alone.map(|(name, _)| name), EXACT_FORMS);
        let mut every = fast;
        for (name, set) in alone {
            let mut want = fast;
            set(&mut want);
            assert_ne!(want, fast, "{name}");
            assert_eq!(one(name), want, "{name}");
            set(&mut every);
        }
        let named = " half_chains,wide_lines,pipeline_waits,host_page_textures , batch_order,\
                     queue_sharing,extended_range_blits,dont_care_colors";
        assert_eq!(exact_policy(Some(named)), every);
        assert_eq!(exact_policy(Some(&EXACT_FORMS.join(","))), every);
        assert_eq!(one(EXACT_ALL), fast.exact_answers());
        assert_eq!(fast.exact_answers(), every);
    }

    /// Obj: clone_ref() is another reference to the same object, and the last drop releases it
    #[test]
    fn obj_references_one_object() {
        let o = Obj::new(7u32);
        let weak = o.downgrade();
        let r = o.clone_ref();
        assert_eq!(r.id(), o.id());
        assert_eq!(*r, 7);
        assert_ne!(Obj::new(7u32).id(), o.id());
        drop(o);
        assert!(Obj::upgrade(&weak).is_some());
        drop(r);
        assert!(Obj::upgrade(&weak).is_none());
        assert!(format!("{:?}", Obj::new(3u8)).ends_with(" 3>"));
    }

    /// A device creator dies at its selector where the runtime refuses its devices otherwise than
    /// for having none, naming the refusal; with no device or no loader it makes none
    #[test]
    fn a_refused_device_dies_at_the_creator() {
        use reims_vgpu_core::raise::death::DeathKind;
        use reims_vgpu_core::testkit::capture::catch_death;
        for sel in ["MTLCreateSystemDefaultDevice()", "MTLCopyAllDevices()"] {
            let refusal = rt::Error::NotImplemented(rt::RuntimeGap::Textures);
            *crate::answer::LAST_LINE.lock() = None;
            let site = catch_death(|| made_devices(sel, || Err(refusal))).unwrap_err();
            let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
            assert_eq!((site.kind, site.file), (DeathKind::Abort, sel), "{line}");
            assert!(made_devices(sel, || Err(rt::Error::NoDevice)).is_empty());
        }
    }

    /// open_device finds the default device by the registry ID the platform answers for it
    /// (Metal's default device's on macOS, the runtime's own where no Metal stands for it), as
    /// the one object every handle to it is; None for an unknown ID, and, where Metal's stands
    /// for it, for the runtime's own ID, which no host sees
    #[test]
    fn open_device_finds_the_device() {
        let p = platform();
        let d = dev();
        let own = d.raw.registry_id();
        let id = crate::os::default_registry_id().unwrap_or(own);
        let found = p.open_device(id).unwrap();
        assert_eq!(found.id(), d.id());
        assert_eq!(d.registry_id, id);
        assert!(p.open_device(id ^ 0xdead_beef).is_none());
        assert_eq!(p.open_device(own).is_some(), id == own);
    }

    /// An open of the default device, or of a device the process adopted, asks the runtime for
    /// no other device: every device, one of each GPU, is asked for only for another ID. The
    /// default device found is adopted, so an open of it again makes no runtime call at all
    #[test]
    fn an_open_of_the_default_device_makes_no_other() {
        let raw = rt::Device::system_default_with(device_options()).unwrap();
        let id = device::registry_id(&raw);
        let every = || -> Vec<rt::Device> { panic!("every device was asked for") };
        let default = || {
            made_devices(OPEN_DEVICE, || {
                Ok(rt::Device::system_default_with(device_options())
                    .ok()
                    .into_iter()
                    .collect())
            })
        };
        let found = opened(id, default, every).unwrap();
        let nothing = || -> Vec<rt::Device> { panic!("the runtime was asked") };
        let again = opened(id, nothing, every).unwrap();
        assert_eq!(again.id(), found.id());
        assert_eq!(found.id(), dev().id());
        let mut asked = false;
        let other = opened(id ^ 0xdead_beef, default, || {
            asked = true;
            Vec::new()
        });
        assert!(other.is_none() && asked);
        drop(raw);
    }

    /// The library's shader cache holds the runtime's default of 25 GiB, sized for a macOS 26
    /// guest's apps on an RTX 5080 laptop: a pin whose default differs fails here,
    /// for review. The unit tests' own directory holds 1 GiB
    #[test]
    fn the_shader_cache_holds_25_gib() {
        assert_eq!(rt::DEFAULT_CAPACITY, 25 << 30);
        assert_eq!(device_options().shader_cache.capacity, 1 << 30);
    }

    /// A strict process, as every test of this crate is, waits for each pipeline until it is
    /// made: no draw or dispatch is dropped for its compile's time, which would die here
    #[test]
    fn a_strict_process_waits_for_every_pipeline() {
        assert_eq!(
            device_options().policy.pipeline_waits,
            metal2vulkan::device::PipelineWaitPolicy::Exact
        );
    }

    /// The services are the ones the platform was made with
    #[test]
    fn services() {
        let p = platform();
        let _ = (p.surfaces(), p.vm(), p.clock(), p.threads());
        assert!(format!("{p:?}").contains("VulkanPlatform"));
        let page = if cfg!(target_os = "macos") {
            16384
        } else {
            4096
        };
        assert_eq!(p.vm().page_size(), page);
    }
}
