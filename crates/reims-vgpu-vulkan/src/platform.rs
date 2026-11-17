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

use crate::core::num::ToU64;
use crate::core::platform::os::Threads;
use crate::core::platform::os::vm::HostVm;
use crate::core::platform::types::MtlSize;
use crate::core::platform::values::{RateMapDesc, TextureDesc};
use crate::core::platform::{Handle, ObjId, Platform};
use crate::core::raise::{OnFault, RaiseResult};
use metal2vulkan_runtime as rt;

use crate::answer::{get, plain, plain_at, refused};
use crate::gpu::command::{VulkanCommandBuffer, VulkanCommandQueue};
use crate::gpu::desc::{
    RenderPassDescriptorObject, TextureDescriptorObject, TextureFields, VulkanRenderPassDescriptor,
    VulkanTextureDescriptor,
};
use crate::gpu::device::{self, VulkanDevice};
use crate::gpu::dynamic_library::VulkanDynamicLibrary;
use crate::gpu::encode_blit::VulkanBlitEncoder;
use crate::gpu::encode_compute::VulkanComputeEncoder;
use crate::gpu::encode_render::VulkanRenderEncoder;
use crate::gpu::event::{VulkanEvent, VulkanSharedEvent, VulkanSharedEventListener};
use crate::gpu::function_table::{VulkanFunctionHandle, VulkanVisibleFunctionTable};
use crate::gpu::icb::{VulkanIcb, VulkanIndirectComputeCommand, VulkanIndirectRenderCommand};
use crate::gpu::rate_map::{VulkanRateMap, VulkanRateMapDescriptor, VulkanRateMapLayerDescriptor};
use crate::gpu::resource::{
    VulkanBuffer, VulkanComputePipeline, VulkanDepthStencilState, VulkanFence, VulkanFunction,
    VulkanHeap, VulkanLibrary, VulkanRenderPipeline, VulkanSamplerState, VulkanSharedTextureHandle,
    VulkanTexture,
};
use crate::os::{OsClock, OsThreads};
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

    /// How many references to the object live, this one among them.
    pub(crate) fn strong_count(&self) -> usize {
        Arc::strong_count(&self.0)
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

/// The process's platform: the runtime's GPU objects, and the OS's VM (reporting the pages it
/// changes, [`crate::remap`]), clocks and threads ([`crate::os`]).
pub struct VulkanPlatform {
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
    /// pool core's pools push, as Foundation still autoreleases into it.
    pub fn new() -> VulkanPlatform {
        crate::os::install_process_services();
        VulkanPlatform {
            vm: VulkanVm::new(),
            clock: OsClock::new(),
            threads: OsThreads::new(),
        }
    }

    /// The registry ID of the process's default device, as the runtime answers it, which
    /// [`crate::device_choice::compositor_gpu`] tells the compositor's GPU from: `None` where it
    /// has none.
    pub fn default_device_registry_id() -> Option<u64> {
        default_device_registry_id()
    }

    /// A device of a window's own on the GPU of registry ID `registry_id` (the compositor's,
    /// [`crate::device_choice::CompositorGpu::Other`]), apart from the process's default device
    /// the guest's work runs on, which it shares no object with ([`device::window_device`]).
    /// `None` where the runtime refuses it, logged: the window then presents from the default
    /// device.
    pub fn window_device(registry_id: u64) -> Option<VulkanDevice> {
        match rt::Device::with_registry_id(registry_id, device_options()) {
            Ok(raw) => Some(device::window_device(raw)),
            Err(e) => {
                log::warn!(
                    target: "reims_vgpu_vulkan::device",
                    "the runtime refused a window's device on registry ID {registry_id:#x} \
                     ({e}): the window presents from the default device"
                );
                None
            }
        }
    }

    /// The process's default device (`MTLCreateSystemDefaultDevice`, [`default_device`]): `None`
    /// where the runtime finds no Vulkan device it can use, or no Vulkan loader. Any other
    /// refusal dies here, naming it ([`made_devices`]).
    pub fn system_default_device() -> Option<VulkanDevice> {
        made_devices("MTLCreateSystemDefaultDevice()", || {
            default_device().map(|d| vec![d])
        })
        .pop()
        .map(device::adopt)
    }
}

/// The devices `make` makes, a creator standing for `sel` (`MTLCreateSystemDefaultDevice()`,
/// `MTLCopyAllDevices()`): none where the runtime finds no Vulkan device it can use, or no Vulkan
/// loader; none too where the host's Vulkan driver refuses the device
/// ([`refused_by_the_driver`]), logged at error naming the refusal, as a host with no device it
/// could use: the caller answers as it does with none (the graphics device's create
/// `Error::HostResources`, `RVG_ERR_HOST_RESOURCES`). On an RTX 2080 whose BAR1 is 256 MiB, of
/// which each device maps 67 MiB, 8 processes making a device at once had 4 to 6 of them refused
/// so (`VK_ERROR_INITIALIZATION_FAILED`, **measured** 2026-11-11, driver 620.70.06).
///
/// Any other refusal dies at `sel`, naming it. A device that lists a window extension the options
/// name but lacks its feature (`Error::ExtensionFeature`, `swapchainMaintenance1`) dies at `sel`
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
        Err(e) if refused_by_the_driver(&e) => {
            log::error!(
                target: "reims_vgpu_vulkan::device",
                "{sel}: the host's Vulkan driver refused the device ({e}): none is made, as on a \
                 host with no Vulkan device"
            );
            Ok(Vec::new())
        }
        made => made,
    })
}

/// Whether `e`, a refusal of a device's make, is the host's Vulkan driver's: a Vulkan call's
/// failure (`VK_ERROR_INITIALIZATION_FAILED` and the rest), its memory run out, or its device
/// lost. Every other is the library's or the runtime's own.
fn refused_by_the_driver(e: &rt::Error) -> bool {
    matches!(
        e,
        rt::Error::Vulkan(_) | rt::Error::OutOfMemory(_) | rt::Error::DeviceLost
    )
}

/// The creator [`Platform::open_device`] stands for, which its deaths name.
const OPEN_DEVICE: &str = "MTLCopyAllDevices()";

/// The device of registry ID `registry_id` ([`device::registry_id`]) that an open answers
/// ([`Platform::open_device`]): the device of that ID the process adopted, where there is one,
/// with no runtime call; else the process's default device ([`default_device`]), of those
/// `default` makes (none where the runtime refuses it, whose refusal `every` then decides), where
/// its ID is `registry_id`; else the one of that ID among those `every` makes, the runtime's
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
/// asked before any is, the process's default device ([`default_device`]), made for it, which
/// dies as [`made_devices`] does where the runtime makes none, at `sel`, the call that asks. Kept
/// once found, as the platform keeps every device it adopts, until the platform lets go of its
/// devices ([`crate::let_go`]).
pub(crate) fn answering_device(sel: &'static str) -> rt::Device {
    if let Some(kept) = ANSWERING.lock().as_ref() {
        return kept.clone();
    }
    let device = device::first_adopted().unwrap_or_else(|| {
        let made = made_devices(sel, || default_device().map(|d| vec![d]));
        made.into_iter()
            .next()
            .unwrap_or_else(|| refused(sel, &rt::Error::NoDevice))
    });
    ANSWERING.lock().get_or_insert(device).clone()
}

/// The device [`answering_device`] answers with, once found.
static ANSWERING: crate::core::sync::Mutex<Option<rt::Device>> =
    crate::core::sync::Mutex::new(None);

/// Takes the device [`answering_device`] kept, for the platform's let-go of its devices
/// ([`crate::let_go`]): the next answer finds one again.
pub(crate) fn take_answering_device() -> Option<rt::Device> {
    ANSWERING.lock().take()
}

/// The kinds the runtime announces at every occurrence (`DeviceOptions::announce_each`): those a
/// call here hears ([`crate::answer::HEARD`]), `CopiedHostPages` and `ShadowedTexture`
/// ([`crate::answer::PROBED`]), which the device's probe of its imports may meet unheard as it is
/// made ([`crate::answer`]'s `unheard`, on Linux: `crate::os`'s `learn_view_imports`), each
/// relayed the first time alone and counted after, and every
/// kind of [`crate::answer::DIES`] where `every_dies`, in strict and census mode, which die at or
/// record each occurrence. Elsewhere the
/// runtime announces such a kind once per device, its first occurrence, the only one the default
/// writes: asked for each, a pipeline variant's notices reach the sink again at every draw's bind,
/// a cost on the encoding thread (in a Finder drag on ms01's Iris Xe, 2026-10-18, the sink
/// itself took 0.09 ms per frame of that thread's 16.3, measured).
fn announced_each(every_dies: bool) -> std::collections::BTreeSet<rt::Degradation> {
    let dies: &[rt::Degradation] = if every_dies {
        &crate::answer::DIES
    } else {
        &[]
    };
    (dies.iter().copied())
        .chain(crate::answer::HEARD)
        .chain(crate::answer::PROBED)
        .collect()
}

/// The class of Mac every device of the process presents to the runtime
/// (`DeviceOptions::mac_class`), fixed once for the process by whichever comes first: the first
/// [`present_mac_class`], which fixes the class it names, or the first read of [`device_options`],
/// which fixes none, so each device presents its own GPU's class (the runtime's
/// `MacClass::of_vendor`: an Apple silicon Mac on an Apple GPU, a Mac with an Intel or AMD GPU on
/// any other).
static MAC_CLASS: OnceLock<Option<rt::MacClass>> = OnceLock::new();

/// Declares the class of Mac every device of the process presents: the runtime then gives that
/// Mac's answers where a Mac's GPU decides them, on whatever GPU the host has. The runtime lists
/// them (its `MacClass`, and docs/API.md, "The Mac a device presents"): what a new `Private`
/// texture reads before anything writes it (on an Apple silicon Mac Metal's fixed pattern, opaque
/// magenta in an `RGBA8Unorm` texture, wherever Metal keeps the texture losslessly compressed and
/// zero elsewhere; on a Mac with an Intel or AMD GPU what its memory held, zero on an Apple GPU),
/// the encoding a GPU blit through a compressed texture writes (`BGRA10_XR`, `RG11B10Float` and
/// `RGB9E5Float` canonicalized on Apple silicon, the bytes kept on Intel or AMD), a `half` NaN
/// clear's sign (kept on Apple silicon in a pass without a draw call, through `half` on Intel or
/// AMD), and how far a `Clear` load reaches past the render area in a pass of one layer without
/// a draw call (on Apple silicon the whole level of a compressed texture and the render area's
/// 32x32 tiles of any other, on Intel or AMD the whole level). The runtime fixes the class of a
/// device as it makes it, and keeps one device per physical device and options, so the class is
/// the process's: the declaration takes only before the process's first read of
/// [`device_options`] (its first device, or the device choice's look at the host's devices).
/// Answers the class the process's devices present: `Some(class)` where this declaration fixed
/// it, or another class or `None` (each device's GPU's) fixed before it.
pub fn present_mac_class(class: rt::MacClass) -> Option<rt::MacClass> {
    *MAC_CLASS.get_or_init(|| Some(class))
}

/// The texture resource IDs a guest places its textures at, which every device the process adopts
/// after their declaration reserves first ([`present_guest_texture_ids`]).
static GUEST_TEXTURE_IDS: OnceLock<std::ops::RangeInclusive<u64>> = OnceLock::new();

/// Declares the texture resource IDs the process's guest places its textures at: every device the
/// process adopts from now on reserves them as its first call (`gpu::device::adopt`), before the
/// library's own work on it makes a command queue or a texture (on Linux, the fold probe's queue
/// as the device learns its imports of views, `crate::os::learn_view_imports`). Metal takes a
/// texture reservation only before the device's first texture, and one made after the device's
/// first command queue keeps the pool's first range beside it, handing those IDs out twice; the
/// runtime imitates both. A later call of the device's for exactly these IDs is then answered at
/// once ([`crate::gpu::device::VulkanDevice`]'s `reserve_resource_indices`). A device adopted
/// before the declaration reserves nothing as it is adopted. The first declaration fixes the IDs
/// for the process; answers the IDs it fixed.
pub fn present_guest_texture_ids(
    ids: std::ops::RangeInclusive<u64>,
) -> std::ops::RangeInclusive<u64> {
    GUEST_TEXTURE_IDS.get_or_init(|| ids).clone()
}

/// The guest's texture resource IDs, where declared ([`present_guest_texture_ids`]).
pub(crate) fn guest_texture_ids() -> Option<std::ops::RangeInclusive<u64>> {
    GUEST_TEXTURE_IDS.get().cloned()
}

/// The options every device is made with: the runtime's defaults, whose loader search finds
/// Homebrew's (`vulkan-loader`, beside its `molten-vk`), with the window system's extensions where
/// a display's window presents through Vulkan (on Linux, [`crate::os::add_window_extensions`]), the
/// shader cache in [`SHADER_CACHE`] in the user's cache directory
/// ([`crate::os::user_cache_directory`]: `~/.cache/reims-vgpu-vulkan`, or the runtime's default
/// directory where `HOME` is not set, on Linux and macOS; `reims-vgpu\reims-vgpu-vulkan` in the
/// user's local application data folder on Windows; a test build's own beside it), which every
/// process shares, holding
/// [`SHADER_CACHE_CAPACITY`], each pipeline state's
/// record kept in the file beside it for later processes' warm passes ([`crate::records`]; none
/// in a test build), a command buffer's
/// GPU times its GPU's own timestamps (the runtime's default), the runtime's device policy,
/// its fast native forms but its exact completion and heap textures and those `REIMS_VGPU_EXACT`
/// names ([`exact_policy`]), and in a strict process ([`crate::answer::strict_process`]) its
/// exact pipeline wait, under which a state's creation returns only once its predicted pipeline
/// is made (under either wait no draw or dispatch is dropped while its pipeline is being made: a
/// command buffer committed before it waits for it and completes later), and the degradations the
/// runtime announces sent to the answer layer ([`crate::answer::notice`]), where those that reach
/// what the guest sees are counted, and die in strict mode ([`crate::answer::strict`]), with the
/// kinds announced at every occurrence ([`announced_each`]), its other messages there too but
/// those below the least severity the log takes ([`crate::answer::least_severity`]), and the class
/// of Mac the process's devices present ([`present_mac_class`]). One set, so the runtime keeps one
/// device per physical device across `open_device` and every other open.
///
/// **No library file opened.** The device opens no file a library's install names name
/// (`load_no_dynamic_library_file`). The install names come from the guest's library bytes, so
/// without it a pipeline's creation would read whatever host path the guest named: `/dev/zero`
/// until the process ran out of memory, a FIFO that blocks a compile for good, a device whose
/// open has effects of its own. A function whose import no library the device was handed
/// supplies fails to make its pipeline as Metal's undefined symbols (code 2 of
/// [`crate::gpu::states`]'s refusals).
///
/// **GPU times are the GPU's.** Every command buffer with work writes two GPU timestamps, at the
/// top of its work and after its end, which the runtime places on the host's clock by a
/// calibration it reads as command buffers complete, at most every 100 ms: its `GPUStartTime`
/// and `GPUEndTime` are when its GPU ran it, as Metal's are (the runtime's `gpu_time.rs`). A
/// completed handler learns them (core's `CompletedInfo::gpu_times_ns`), and the reims-graphics
/// device posts them to its guest, whose bundle answers the run's length from them, which the
/// guest's compositor reads; this platform's debug lines read them too ([`crate::gpu::timing`]).
/// They also let the runtime share `VkQueue`s between command queues where its own check finds
/// the GPU idle at each switch between queues (NVIDIA's), which it does only on a device whose
/// command buffers write timestamps the host reads. On MoltenVK 1.4.1, whose timestamps wait for
/// every queue's (the runtime's `TimestampsWaitForEveryQueue`), the runtime answers GPU times from
/// the host's clock instead, whatever the options say: a command buffer's start is the host time
/// the first driver submission that carried its work was called, or its queue's latest GPU end
/// before it, where later, and its end the host time the runtime saw it complete. Off MoltenVK
/// the timestamps cost a few microseconds: measured on 2026-11-15 (`timing`'s
/// `gpu_times_from_host_latency`, a release build, runs of 1000 command buffers of one blit fill
/// each way, interleaved), a command buffer completed a median of 2 to 7 us later after its commit
/// than with GPU times from the host (`gpu_times_from_host`), within runs whose medians spanned 44
/// to 292 us: RADV on an RX 7900 XTX (+4 us over 13 runs), a Radeon 780M (+4 us over 13 runs, and
/// 0 to 3 us on another) and an RX 550 (+2.5 us over 10 runs), ANV on an Iris Xe (+4.5 us over 10
/// runs), NVIDIA 620.70.06 on an RTX 3090 (+5 us over 10 runs) and an RTX 2080 (+7 us over 10
/// runs). Walks of a macOS 26.7.2 guest's desktop and Launchpad on the RX 7900 XTX, a Radeon 780M,
/// the Iris Xe and the RTX 3090 drew the same presents either way (Launchpad's later openings 54 a
/// second on the RX 7900 XTX and the RTX 3090, 45 on the others), their take-to-show medians
/// within the walks' own spread (the RTX 3090's 8.0 to 8.2 ms over eight walks, with `VkQueue`s
/// shared or not). On the M3 Max, before the runtime answered from the host there, the timestamps cost 356
/// to 399 us against 283 to 309 us.
pub(crate) fn device_options() -> &'static rt::DeviceOptions {
    static OPTIONS: LazyLock<rt::DeviceOptions> = LazyLock::new(|| {
        let mut options = rt::DeviceOptions::default();
        crate::os::add_window_extensions(&mut options);
        options.load_no_dynamic_library_file = true;
        options.mac_class = *MAC_CLASS.get_or_init(|| None);
        options.policy = exact_policy(std::env::var("REIMS_VGPU_EXACT").ok().as_deref());
        if crate::answer::strict_process() {
            options.policy.pipeline_waits = metal2vulkan::device::PipelineWaitPolicy::Exact;
        }
        let sink = rt::MessageSink::new(crate::answer::notice);
        options.messages = Some(sink.with_least_severity(crate::answer::least_severity()));
        options.announce_each =
            announced_each(crate::answer::strict_process() || crate::answer::census::on());
        if std::env::var_os(crate::core::trace::ENV_VAR).is_some() {
            crate::core::trace::set_clock(crate::gpu::timing::trace_clock);
            if crate::answer::trace_degradations() && crate::core::trace::on() {
                // Every occurrence reaches the sink, which writes each to the trace and relays
                // a kind the runtime announces once only the first time: only where the trace's
                // file opened, the sink's own condition, so a refused file changes nothing.
                options.announce_each = rt::Degradation::ALL.iter().copied().collect();
            }
        }
        if let Some(cache) = crate::os::user_cache_directory() {
            options.shader_cache = rt::ShaderCacheOptions::in_directory(cache.join(SHADER_CACHE));
            if crate::records::KEPT {
                options.pipeline_records = crate::records::hook(crate::records::path_in(&cache));
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
/// pipeline the driver is still making until it is made, and a state's creation that returns only
/// once its predicted pipeline is made, where the default's creation returns after 4.5 s with it
/// still being made and a commit records a pass whose executions read the CPU's records at most
/// 4.5 s after it, leaving the pass to its host step past that (`pipeline_waits`,
/// `PipelineWaitPolicy::Exact`; under either, no command is dropped), and, on a device with `tessellationShader`, a line `setLineWidth:`
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
/// `DontCareColorPolicy::Kept`), and an attachment a pass loads `DontCare` and does not store
/// cleared wherever the pass's draws reach it, so they read zero or the clear value, as Metal's
/// draws read, where the default leaves it to the driver's `DONT_CARE` load on a driver not
/// measured to clear it no slower, announcing `dont-care-unstored-load`
/// (`unstored_dont_care_loads`, `UnstoredDontCarePolicy::Cleared`), and a vertex buffer layout
/// a draw's `uint` indices or indirect arguments may fetch past 4 GiB of its bind offset pulled
/// in the vertex function on a driver whose fixed-function fetch wraps there (lavapipe), so it
/// reads Metal's bytes, where the default keeps it in fixed function, announcing
/// `vertex-fetch-may-wrap` (`vertex_reach`, `VertexReachPolicy::Exact`), and a heap texture's
/// size and alignment answered by Metal's own model, the texture refitted to an image that fits
/// that answer or kept as a shadow of Metal's tiles, where the default answers the larger of
/// Metal's model and the device's image and lays the texture in that image (`heap_answers`,
/// `HeapAnswerPolicy::Metal`), and a rasterization-rate-mapped pass's draws drawn region by
/// region, each running its vertex function once per region of the map, where the default draws
/// once each draw a host step records that the runtime's probe finds safe, a geometry stage
/// emitting a copy of each triangle for each region it touches, so a pixel centre within a
/// rounding error of a triangle's edge may be covered by one route and not the other
/// (`rate_map_regions`, `RateMapRegionPolicy::Exact`), and a Managed buffer's modified range
/// copied on the CPU into staging memory at each commit, as the buffer's host copy holds it then,
/// so a CPU write after the commit never reaches a command buffer committed before it, as on the
/// Radeon Pro 5300M, where the default has the GPU copy the range from the host copy as the
/// submission runs, so such a write may be read, as on the UHD 630 (`managed_copy_in`,
/// `ManagedCopyInPolicy::AtCommit`), and an indirect command buffer's records in the runtime's
/// own pages imported by their pointer wherever the device imports host pointers, so a recovery
/// of the device keeps them as the CPU and the GPU left them, as Metal does, where the default
/// lays them in pooled host-visible memory on a device where a held pointer import costs every
/// submission (RADV; ANV on i915), whose recovery loses them and writes them again as a new
/// buffer's (`indirect_records`, `IndirectRecordsPolicy::Carried`). [`EXACT_ALL`] names every
/// one. The
/// policy's `completion` is no name of these: every device takes its exact form
/// ([`exact_policy`]). Nor is its `heap_compression`, whose uncompressed form [`EXACT_ALL`]
/// alone takes. The rest of Metal's values the runtime answers with the
/// device's own arithmetic, sampling and conversions, each difference a precision note it
/// records, with no exact form to name.
pub const EXACT_FORMS: [&str; 14] = [
    "batch_order",
    "host_page_textures",
    "half_chains",
    "pipeline_waits",
    "wide_lines",
    "extended_range_blits",
    "queue_sharing",
    "dont_care_colors",
    "unstored_dont_care_loads",
    "vertex_reach",
    "heap_answers",
    "rate_map_regions",
    "managed_copy_in",
    "indirect_records",
];

/// The name `REIMS_VGPU_EXACT` takes for the runtime's `DevicePolicy::exact_answers()`: every
/// field of [`EXACT_FORMS`] in its exact form, as the runtime's validation sets them, and a heap's
/// colour texture uncompressed in the heap's bytes (`HeapCompressionPolicy::Uncompressed`), which
/// no other name takes.
pub const EXACT_ALL: &str = "all";

/// The device policy every device is made with: the runtime's default, whose native forms are
/// fast to make and run and record their bounds off Apple's GPUs (docs/accuracy-ledger.md: the
/// fast method by default, the accurate one behind a switch), but for its completion, which is
/// always the exact form (`CompletionPolicy::Exact`): each command buffer goes to the driver in a
/// submission of its own and is seen complete at its own end, as Metal's, where the runtime's
/// default would take up to four ready together in one kernel submission on Mesa's drivers and
/// see each but the last complete only once the last completed. When the host sees a command
/// buffer complete is the guest's to observe (the stamps and interrupts core writes as its work
/// completes), and Metal's hosts see each at its own end, whatever is committed after it: of 48
/// command buffers committed back to back on one queue, each of 0.5 to 5 ms of GPU work, each was
/// seen complete (its completed handler, its status) 17 to 41 us after its own GPU end at the
/// median and at most 141 us on a Radeon Pro 5300M and a UHD Graphics 630 (macOS 26.7.1), 86 to 106
/// us and at most 0.8 ms on an M3 Max (macOS 27.0.1), and none only once the next had ended
/// (2026-11-15). The grouped form sees a member complete late by the GPU time of the batches after
/// it in its group, up to the runtime's 16 ms bound, and on RADV it bought nothing for that: in a
/// guest's Launchpad on a Radeon 780M and an RX 7900 XTX it held at most 30 batches in a window
/// once Launchpad was open (21.6 to 24.5 kernel submissions a present either way) and saved 0.15 to
/// 1 a present at its first opening, and in Geekbench 7's Metal pass it took 9% and 7% fewer kernel
/// submissions for no shorter run (174.6 s against 172.6 exact on the 780M, 136.0 against 136.1 on
/// the XTX), seeing command buffers complete up to 60 and 17.5 ms after their own GPU end (2.9 and
/// 3.6 ms exact). Nor did it buy frame rate in a guest's drag on the Iris Xe (15.7 to 17.3 frames a
/// second, against 16.4 to 17.6 exact). So no switch takes it. Under it the runtime groups only a
/// command buffer its caller marks to join the next (`CommandBuffer::set_completion_joins_next`),
/// with the batches found ready behind it, the first unmarked one closing the group and seen
/// complete at its own end: the graphics lane marks a guest command buffer's parts before its
/// last, whose completions the guest never sees ([`crate::answer`], `GroupedCompletion`). Its heap
/// compression is the runtime's default (`HeapCompressionPolicy::Compressed`), which no name of
/// [`EXACT_FORMS`] changes: on a driver whose compressed images may keep their texels apart from
/// their bytes (ANV), a heap's colour texture that Metal keeps losslessly compressed, in a heap
/// of the device's own memory (a guest's `Private` heap), takes the driver's compressed image, its
/// fast clears kept, where that image's 64 KiB alignment fits the heap's answer for the texture
/// (the runtime's `HeapAnswerPolicy::Device`: an image of 64 KiB or more), as Metal keeps the
/// texture compressed. A write through a buffer or texture over its bytes may then leave its
/// texels as they were, where Metal's texture reads opaque magenta in the drawn content the write
/// covered (an M3 Max): neither shows the write as written. That image moves what
/// `heapTextureSizeAndAlignWithDescriptor:` answers the guest on the Iris Xe from a 4 KiB
/// alignment to 64 KiB, its size kept (0x1020000 bytes for a 2048x2048 `RGBA8Unorm` render target,
/// 0x10200 for a 128x128 `RGBA8Unorm` texture only read), and leaves a smaller one's (0x4080 bytes
/// at 0x1000 for a 64x64 `BGRA8Unorm` one). A texture of a heap over the guest's pages takes an
/// image over them under either form. [`EXACT_ALL`] takes the image the driver keeps uncompressed
/// in the heap's bytes (`HeapCompressionPolicy::Uncompressed`), as for a texture Metal keeps
/// uncompressed, which loses the driver's fast clear: a guest's clear of a 2048x2048 `RGBA8Unorm`
/// render target in a `Private` heap took 146.9 us of GPU time on the Iris Xe against 28.6 us
/// compressed, and a copy of it 294.1 us against 148.2. Its in-place YCbCr reads are always the
/// exact form, with no switch: a function that reads a `YCBCR8_420_2P` or `YCBCR10_420_2P`
/// texture's planes in place, on a device whose memory is the host's, converts and filters every
/// tap as Metal does (`YCbCrInPlacePolicy::Exact`), where the runtime's default samples by
/// Vulkan's Y'CbCr model in the device's arithmetic, outside the bound of Metal's answers
/// `gpu::ycbcr::sampling`'s tests hold. Each field of [`EXACT_FORMS`] that `exact`,
/// `REIMS_VGPU_EXACT`'s comma-separated list, names takes the runtime's exact form too, and
/// [`EXACT_ALL`] every such form:
/// `batch_order`'s barriers between batches, `host_page_textures`' copies of each sampled texture
/// over the guest's pages at each command buffer's first use, and `half_chains`' rounding at
/// each `half` operation on a device without `shaderFloat16`, `pipeline_waits`' wait for every
/// pipeline until it is made, `wide_lines`' band variant of each pipeline that draws a line
/// the device's own lines would draw otherwise than Metal, `extended_range_blits`' rewrite of
/// each extended-range blit to Metal's encoding, and `heap_answers`' heap texture sizes and
/// alignments by Metal's model, `rate_map_regions`' region-by-region draws of each
/// rate-mapped pass, and `managed_copy_in`'s CPU copy of a Managed buffer's modified range at
/// each commit. A name of neither changes nothing,
/// and a line on stderr says so.
fn exact_policy(exact: Option<&str>) -> metal2vulkan::device::DevicePolicy {
    use metal2vulkan::device::{
        BatchOrderPolicy, CompletionPolicy, DevicePolicy, DontCareColorPolicy,
        ExtendedRangeBlitPolicy, HalfChainPolicy, HeapAnswerPolicy, HostPageTexturePolicy,
        IndirectRecordsPolicy, ManagedCopyInPolicy, PipelineWaitPolicy, QueueSharingPolicy,
        RateMapRegionPolicy, UnstoredDontCarePolicy, VertexReachPolicy, WideLinePolicy,
        YCbCrInPlacePolicy,
    };
    let mut policy = DevicePolicy::default();
    policy.completion = CompletionPolicy::Exact;
    policy.ycbcr_in_place_reads = YCbCrInPlacePolicy::Exact;
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
            "unstored_dont_care_loads" => {
                policy.unstored_dont_care_loads = UnstoredDontCarePolicy::Cleared;
            }
            "vertex_reach" => policy.vertex_reach = VertexReachPolicy::Exact,
            "heap_answers" => policy.heap_answers = HeapAnswerPolicy::Metal,
            "rate_map_regions" => policy.rate_map_regions = RateMapRegionPolicy::Exact,
            "managed_copy_in" => policy.managed_copy_in = ManagedCopyInPolicy::AtCommit,
            "indirect_records" => policy.indirect_records = IndirectRecordsPolicy::Carried,
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

/// The shader cache's directory in the user's cache directory
/// ([`crate::os::user_cache_directory`]), which every process of the library shares. It
/// holds the runtime's translations of the guest's functions and each pipeline's driver record
/// (MoltenVK's included: about 1.7 MB for a macOS 26 guest's `Xgc` pipeline
/// on NVIDIA), within [`SHADER_CACHE_CAPACITY`], the least recently used going first.
#[cfg(not(any(test, feature = "testkit")))]
const SHADER_CACHE: &str = "reims-vgpu-vulkan";

/// The most bytes [`SHADER_CACHE`] holds: the runtime's default, 25 GiB (a test keeps it so). A
/// macOS 26 guest's built-in apps, each launched once, need more than 1 GiB of translations and
/// driver records on an RTX 5080 laptop. The runtime seeds each pipeline's making from its own
/// record here; the driver's own disk cache is left as the driver keeps it, and may
/// hold the pipeline too. An entry this cache drops past its capacity is compiled again at the
/// next boot that draws with it, unless the driver's cache still holds it: a freeze of 13 to 26 s
/// there.
#[cfg(not(any(test, feature = "testkit")))]
pub(crate) const SHADER_CACHE_CAPACITY: u64 = rt::DEFAULT_CAPACITY;

/// A test build's capacity, 1 GiB: its directory holds what the tests compile and nothing a
/// guest's boot reuses, so it is kept small on disk.
#[cfg(any(test, feature = "testkit"))]
pub(crate) const SHADER_CACHE_CAPACITY: u64 = 1 << 30;

/// A test build's shader cache directory in the user's cache directory, the tests' own: this
/// crate's unit tests', and every crate's tests', which turn on its `testkit` feature (only
/// dev-dependencies do, `cargo xtask testkit`). A test of what a device compiles runs on an empty
/// one as on a full one, and the tests write nothing into the one the library's hosts share, nor
/// read the device checks kept there.
#[cfg(any(test, feature = "testkit"))]
const SHADER_CACHE: &str = "reims-vgpu-vulkan-tests";

/// The library's cache directory, [`SHADER_CACHE`] in the user's cache directory
/// ([`crate::os::user_cache_directory`]): the shader cache's, whose sweep touches only its `v1`
/// (the runtime's), and the device checks' `device-checks` beside it (`crate::device_choice`'s
/// verdict, and the host video's checks of its driver). `None` where the OS names no user cache
/// directory (on Linux and macOS, where `HOME` is not set).
pub fn cache_directory() -> Option<PathBuf> {
    Some(crate::os::user_cache_directory()?.join(SHADER_CACHE))
}

/// The runtime's registry ID of the process's default device ([`default_device`]), read once:
/// `None` where it has none.
pub(crate) fn default_device_registry_id() -> Option<u64> {
    static DEFAULT: LazyLock<Option<u64>> =
        LazyLock::new(|| default_device().ok().map(|d| d.registry_id()));
    *DEFAULT
}

/// The process's default device, which every default open makes or finds
/// ([`VulkanPlatform::system_default_device`], [`answering_device`],
/// [`default_device_registry_id`] and [`Platform::open_device`]): the device of the physical
/// device the library's choice names, by its registry ID ([`crate::device_choice::chosen`]),
/// where that is not the runtime's system default device; else, or where the runtime refuses the
/// one chosen, the runtime's system default device (`Device::system_default_with`), as on a host
/// of one GPU. Both with [`device_options`], so a device of the physical device that lives is the
/// one answered.
pub(crate) fn default_device() -> Result<rt::Device, rt::Error> {
    let options = device_options();
    if let Some(id) = crate::device_choice::chosen() {
        match rt::Device::with_registry_id(id, options) {
            Ok(device) => return Ok(device),
            Err(e) => log::warn!(
                target: "reims_vgpu_vulkan::device_choice",
                "the runtime refused the chosen Vulkan device, registry ID {id:#x} ({e}): the \
                 runtime's system default device instead"
            ),
        }
    }
    rt::Device::system_default_with(options)
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
    type SharedEventListener = VulkanSharedEventListener;
    type SharedTextureHandle = VulkanSharedTextureHandle;
    type Library = VulkanLibrary;
    type Function = VulkanFunction;
    type ComputePipeline = VulkanComputePipeline;
    type FunctionHandle = VulkanFunctionHandle;
    type VisibleFunctionTable = VulkanVisibleFunctionTable;
    type DynamicLibrary = VulkanDynamicLibrary;
    type RenderPipeline = VulkanRenderPipeline;
    type RateMap = VulkanRateMap;
    type Icb = VulkanIcb;
    type IndirectRenderCommand<'a> = VulkanIndirectRenderCommand<'a>;
    type IndirectComputeCommand<'a> = VulkanIndirectComputeCommand<'a>;
    type TextureDescriptor = VulkanTextureDescriptor;
    type RenderPassDescriptor = VulkanRenderPassDescriptor;
    type RateMapDescriptor = VulkanRateMapDescriptor;
    type RateMapLayerDescriptor = VulkanRateMapLayerDescriptor;
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
                    Ok(default_device().ok().into_iter().collect())
                })
            },
            || made_devices(OPEN_DEVICE, || rt::Device::all_with(device_options())),
        )
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

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::gpu::device::tests::dev;

    /// The kinds a call hears, and `CopiedHostPages` and `ShadowedTexture`, which the device's
    /// probe of its imports meets unheard, are announced at every occurrence in every mode, and
    /// every kind of `DIES` only in strict and census mode, none of which dies by default
    #[test]
    fn only_strict_and_census_mode_hear_every_occurrence_of_every_kind_that_dies() {
        use crate::answer::{DIES, HEARD};
        let each = announced_each(true);
        assert!(DIES.iter().chain(&HEARD).all(|k| each.contains(k)));
        let by_default = announced_each(false);
        let probed = crate::answer::PROBED;
        assert!((HEARD.iter().chain(&probed)).all(|k| by_default.contains(k)));
        let once = DIES
            .iter()
            .filter(|k| !HEARD.contains(k) && !probed.contains(k));
        assert!(once.clone().count() > 100, "most kinds of DIES");
        assert!(once.clone().all(|k| !by_default.contains(k)));
        let released = rt::Degradation::ReleasedCommandState;
        assert!(once.clone().any(|k| *k == released) && !by_default.contains(&released));
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
    /// its compressed heap textures among them, but for its exact completion and exact in-place
    /// YCbCr reads, which every device takes and no name selects; each name of `EXACT_FORMS`
    /// takes that field's exact form alone, in any order, spaces around a name ignored; `all`
    /// takes the runtime's `exact_answers()`, every one and uncompressed heap textures, which no
    /// other name takes; another name, `completion`, `heap_compression`, `ycbcr_in_place_reads`
    /// and `sampling` among them, changes nothing
    #[test]
    fn reims_vgpu_exact_names_the_policys_exact_forms() {
        use metal2vulkan::device::{
            BatchOrderPolicy, CompletionPolicy, DevicePolicy, DontCareColorPolicy,
            ExtendedRangeBlitPolicy, HalfChainPolicy, HeapAnswerPolicy, HeapCompressionPolicy,
            HostPageTexturePolicy, IndirectRecordsPolicy, ManagedCopyInPolicy, PipelineWaitPolicy,
            QueueSharingPolicy, RateMapRegionPolicy, UnstoredDontCarePolicy, VertexReachPolicy,
            WideLinePolicy, YCbCrInPlacePolicy,
        };
        let runtimes = DevicePolicy::default();
        assert_eq!(runtimes.completion, CompletionPolicy::Grouped);
        assert_eq!(runtimes.heap_compression, HeapCompressionPolicy::Compressed);
        assert_eq!(runtimes.ycbcr_in_place_reads, YCbCrInPlacePolicy::Model);
        let mut fast = runtimes;
        fast.completion = CompletionPolicy::Exact;
        fast.ycbcr_in_place_reads = YCbCrInPlacePolicy::Exact;
        assert_eq!(fast.heap_compression, HeapCompressionPolicy::Compressed);
        assert_eq!(fast.batch_order, BatchOrderPolicy::Ends);
        assert_eq!(fast.host_page_textures, HostPageTexturePolicy::InPlace);
        assert_eq!(fast.half_chains, HalfChainPolicy::AtExits);
        assert_eq!(fast.pipeline_waits, PipelineWaitPolicy::Bounded);
        assert_eq!(fast.wide_lines, WideLinePolicy::Device);
        assert_eq!(fast.extended_range_blits, ExtendedRangeBlitPolicy::Kept);
        assert_eq!(fast.queue_sharing, QueueSharingPolicy::Adaptive);
        assert_eq!(fast.dont_care_colors, DontCareColorPolicy::Driver);
        assert_eq!(
            fast.unstored_dont_care_loads,
            UnstoredDontCarePolicy::Driver
        );
        assert_eq!(fast.vertex_reach, VertexReachPolicy::Bounded);
        assert_eq!(fast.rate_map_regions, RateMapRegionPolicy::FanOut);
        assert_eq!(fast.managed_copy_in, ManagedCopyInPolicy::AtSubmission);
        assert_eq!(fast.indirect_records, IndirectRecordsPolicy::Untaxed);
        for unchanged in [
            None,
            Some(""),
            Some(","),
            Some("exact, bicubic, All"),
            Some("padded_reads"),
            Some("completion"),
            Some("heap_compression"),
            Some("ycbcr_in_place_reads"),
            Some("sampling"),
        ] {
            assert_eq!(exact_policy(unchanged), fast, "{unchanged:?}");
        }
        let one = |name| exact_policy(Some(name));
        type Set = fn(&mut DevicePolicy);
        let alone: [(&str, Set); 14] = [
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
            ("unstored_dont_care_loads", |p| {
                p.unstored_dont_care_loads = UnstoredDontCarePolicy::Cleared;
            }),
            ("vertex_reach", |p| {
                p.vertex_reach = VertexReachPolicy::Exact;
            }),
            ("heap_answers", |p| p.heap_answers = HeapAnswerPolicy::Metal),
            ("rate_map_regions", |p| {
                p.rate_map_regions = RateMapRegionPolicy::Exact;
            }),
            ("managed_copy_in", |p| {
                p.managed_copy_in = ManagedCopyInPolicy::AtCommit;
            }),
            ("indirect_records", |p| {
                p.indirect_records = IndirectRecordsPolicy::Carried;
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
                     queue_sharing,extended_range_blits,dont_care_colors,unstored_dont_care_loads,\
                     vertex_reach,heap_answers,rate_map_regions,managed_copy_in,indirect_records";
        assert_eq!(exact_policy(Some(named)), every);
        assert_eq!(exact_policy(Some(&EXACT_FORMS.join(","))), every);
        assert_eq!(every.heap_compression, HeapCompressionPolicy::Compressed);
        assert_eq!(one(EXACT_ALL), fast.exact_answers());
        let mut all = every;
        all.heap_compression = HeapCompressionPolicy::Uncompressed;
        assert_eq!(fast.exact_answers(), all);
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

    /// A device the host's Vulkan driver refuses is none at its creator, as with no device or no
    /// loader: a Vulkan call's failure (an RTX 2080's `VK_ERROR_INITIALIZATION_FAILED` with its
    /// BAR1 full), its memory run out, or its device lost; the caller then answers as with no
    /// device, where it died at the creator before
    #[test]
    fn a_device_the_driver_refuses_is_none_at_the_creator() {
        for sel in ["MTLCreateSystemDefaultDevice()", "MTLCopyAllDevices()"] {
            for refusal in [
                rt::Error::Vulkan(rt::VkResultCode(-3)),
                rt::Error::OutOfMemory(rt::MemoryKind::Device),
                rt::Error::OutOfMemory(rt::MemoryKind::Host),
                rt::Error::DeviceLost,
            ] {
                assert!(
                    made_devices(sel, || Err(refusal.clone())).is_empty(),
                    "{refusal}"
                );
            }
        }
    }

    /// A device creator dies at its selector where the runtime refuses its devices otherwise than
    /// for having none or for the driver's refusal, naming the refusal; with no device or no
    /// loader it makes none
    #[test]
    fn a_refused_device_dies_at_the_creator() {
        use crate::core::raise::death::DeathKind;
        use crate::core::testkit::capture::catch_death;
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
        let raw = default_device().unwrap();
        let id = device::registry_id(&raw);
        let every = || -> Vec<rt::Device> { panic!("every device was asked for") };
        let default = || {
            made_devices(OPEN_DEVICE, || {
                Ok(default_device().ok().into_iter().collect())
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

    /// The variable that makes [`a_translation_one_process_keeps_is_the_next_process_s`]'s
    /// process a child: the value of the constant its kernel is specialised by.
    const KEPT_ACROSS_CHILD: &str = "RVG_KEPT_ACROSS_CHILD";

    /// A translation one process keeps in the shader cache, in the user's cache directory, is the
    /// next process's: two children in turn make a device of the platform's options and a compute
    /// state of `fc.metallib`'s kernel specialised by a constant this run drew, which no earlier
    /// run's translation was keyed by. The first translates it, and the second reads the first's
    /// translation from the directory and translates nothing. On Windows the children run with no
    /// `HOME`, as a QEMU started outside MSYS2's shell does
    #[test]
    fn a_translation_one_process_keeps_is_the_next_process_s() {
        use std::hash::{BuildHasher as _, Hasher as _};

        use crate::core::num::Truncate as _;

        const LINE: &str = "KEPT-ACROSS: ";
        if let Some(constant) = std::env::var_os(KEPT_ACROSS_CHILD) {
            let constant = constant.to_str().and_then(|c| c.parse().ok());
            let (hits, misses, dir) = translate_specialised(constant.expect("a u32 constant"));
            println!("{LINE}{hits} {misses} {}", dir.display());
            return;
        }
        let constant = std::hash::RandomState::new()
            .build_hasher()
            .finish()
            .low_u32();
        let run = || {
            let mut child = std::process::Command::new(std::env::current_exe().unwrap());
            child
                .args([
                    "--exact",
                    "platform::tests::a_translation_one_process_keeps_is_the_next_process_s",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env(KEPT_ACROSS_CHILD, constant.to_string())
                .env_remove(crate::os::TESTS_CACHE_DIRECTORY);
            if cfg!(windows) {
                child.env_remove("HOME");
            }
            let out = child.output().unwrap();
            let stdout = String::from_utf8_lossy(&out.stdout);
            let stderr = String::from_utf8_lossy(&out.stderr);
            assert!(out.status.success(), "{stdout}{stderr}");
            let line = stdout
                .lines()
                .find_map(|l| l.find(LINE).map(|at| &l[at + LINE.len()..]));
            let line = line.unwrap_or_else(|| panic!("the child made no state: {stdout}{stderr}"));
            let mut fields = line.splitn(3, ' ');
            let mut count = || fields.next().and_then(|n| n.parse::<u64>().ok()).unwrap();
            let (hits, misses) = (count(), count());
            (hits, misses, PathBuf::from(fields.next().unwrap()))
        };
        let dir = cache_directory().expect("a user cache directory");
        let (_, first_misses, first_dir) = run();
        let (second_hits, second_misses, second_dir) = run();
        assert_eq!(
            (&first_dir, &second_dir),
            (&dir, &dir),
            "the children's directories"
        );
        assert!(first_misses >= 1, "the first child translated nothing");
        assert!(
            second_hits >= 1 && second_misses == 0,
            "the second child: {second_hits} hits, {second_misses} misses"
        );
    }

    /// [`a_translation_one_process_keeps_is_the_next_process_s`]'s child: a device of the
    /// platform's options (`window_device`'s, not adopted, so it goes with its last handle) and a
    /// compute state of `fc_kernel` with its constant `constant`. The shader cache's function hits
    /// and misses of the library's, function's and state's making, and its directory, answered
    /// once the device went: its drop writes what its cache thread still held.
    fn translate_specialised(constant: u32) -> (u64, u64, PathBuf) {
        use crate::core::platform::device::GpuDevice as _;
        use crate::core::platform::resource::{FunctionConstant, Library as _};

        let raw = rt::Device::system_default_with(device_options()).expect("a device");
        let dir = raw
            .shader_cache_directory()
            .expect("the shader cache's directory")
            .to_owned();
        let device = device::window_device(raw);
        let before = device.raw.shader_cache_statistics();
        let library = (device.new_library(include_bytes!("../../../tests/shaders/fc.metallib")))
            .unwrap()
            .unwrap()
            .unwrap();
        let mut value = [0; FunctionConstant::MAX_BYTES];
        value[..4].copy_from_slice(&constant.to_le_bytes());
        let uint = FunctionConstant {
            index: 0,
            data_type: 33,
            value,
        };
        let function = (library.new_function_with_constants(OnFault::Return, "fc_kernel", &[uint]))
            .unwrap()
            .unwrap();
        let state = (device.new_compute_pipeline_state_for_function(Some(&function)))
            .unwrap()
            .unwrap();
        let after = device.raw.shader_cache_statistics();
        drop((state, function, library, device));
        (
            after.function_hits - before.function_hits,
            after.function_misses - before.function_misses,
            dir,
        )
    }

    /// Every device opens no file a library's install names name, which come from the guest's
    /// bytes ([`device_options`], "No library file opened"), where the runtime's default opens
    /// them as Metal does.
    #[test]
    fn no_device_opens_a_file_a_library_names() {
        assert!(!rt::DeviceOptions::default().load_no_dynamic_library_file);
        assert!(device_options().load_no_dynamic_library_file);
    }

    /// The variable naming what [`the_process_s_mac_class_is_fixed_by_its_first_declaration`]'s
    /// child does first: `declare` or `read`.
    const MAC_CLASS_CHILD: &str = "RVG_MAC_CLASS_CHILD";

    /// The class of Mac the process's devices present is fixed once: a declaration before the
    /// options are read fixes its class, which the options carry and a later declaration of
    /// another class leaves; a read of the options before any declaration fixes none (each GPU's
    /// own), which a later declaration leaves too, answering it. Each order in a child process of
    /// its own, as the class is the process's
    #[test]
    fn the_process_s_mac_class_is_fixed_by_its_first_declaration() {
        let (apple, intel) = (rt::MacClass::AppleSilicon, rt::MacClass::IntelOrAmd);
        match std::env::var(MAC_CLASS_CHILD).as_deref() {
            Ok("declare") => {
                assert_eq!(present_mac_class(apple), Some(apple));
                assert_eq!(device_options().mac_class, Some(apple));
                assert_eq!(present_mac_class(intel), Some(apple));
                assert_eq!(device_options().mac_class, Some(apple));
                println!("MAC-CLASS-CHILD declare");
            }
            Ok("read") => {
                assert_eq!(device_options().mac_class, None);
                assert_eq!(present_mac_class(intel), None);
                assert_eq!(device_options().mac_class, None);
                println!("MAC-CLASS-CHILD read");
            }
            Ok(other) => panic!("{MAC_CLASS_CHILD}={other}"),
            Err(_) => {
                for first in ["declare", "read"] {
                    let out = std::process::Command::new(std::env::current_exe().unwrap())
                        .args([
                            "--exact",
                            "platform::tests::the_process_s_mac_class_is_fixed_by_its_first_declaration",
                            "--nocapture",
                            "--test-threads=1",
                        ])
                        .env(MAC_CLASS_CHILD, first)
                        .output()
                        .unwrap();
                    let stdout = String::from_utf8_lossy(&out.stdout);
                    assert!(out.status.success(), "{first}: {out:?}");
                    assert!(
                        stdout.contains(&format!("MAC-CLASS-CHILD {first}")),
                        "{first}: {stdout}"
                    );
                }
            }
        }
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
        let _ = (p.vm(), p.clock(), p.threads());
        assert!(format!("{p:?}").contains("VulkanPlatform"));
        let page = if cfg!(target_os = "macos") {
            16384
        } else {
            4096
        };
        assert_eq!(p.vm().page_size(), page);
    }
}
