//! The library's platform over metal2vulkan-rt, the runtime that replaces metal2vulkan-runtime at
//! the cutover: [`RtPlatform`], core's [`Platform`] over rt's public API. It is built under the
//! `rt` feature, which makes it reims-vgpu's `DefaultPlatform`. The feature is a transition
//! switch, and it goes at the cutover with the old runtime and `VulkanPlatform`. Without it nothing
//! here is compiled.
//!
//! **Built piece by piece**, in the order a guest's boot reaches the pieces: the device and its
//! answers, queues, buffers, textures, libraries and pipelines, command buffers and encoders,
//! events, then the display. Built: the device ([`device`]) with its answers ([`answers`] and
//! [`mod@format`]), its command queues ([`queue`]), events, fences and shared events ([`event`]),
//! the reservation of resource indices, buffers and the textures over them ([`buffer`]), with the
//! copies of host pages rt imports no other way ([`copied`]), on Unix the buffers whose memory
//! another API imports (`export`), heaps but sparse ones ([`heap`]),
//! textures made of a descriptor, with their views ([`texture`]), tiled textures, which take a
//! buffer's bytes at their first use ([`tiled`]),
//! command buffers ([`command`]) with their blit, compute and render encoders ([`blit`],
//! [`compute`], [`mod@render`]), render pass descriptors ([`descriptor`]), sampler and
//! depth-stencil states ([`state`]), libraries and their functions ([`library`]), compute
//! pipeline states ([`pipeline`]), render pipeline states ([`render_pipeline`]), and the swapchain
//! a display's window presents through ([`swapchain`]), its images rt's drawables. Until a piece
//! is built, every call that reaches it is refused ([`unbuilt`]). The call answers as Metal's call answers where it
//! fails: a nil object, an `NSError`, zeros. The refusal is counted under the piece's code,
//! `rt-unbuilt-<piece>`, in the process's degradation counts, and the first of each kind writes a
//! line on stderr. Never a death or a panic.
//!
//! **An object of a kind no call makes yet is [`Unmade`]**: a type with no values, so its methods
//! are never reached. The piece that makes the kind gives it a type of its own.
//!
//! **What it shares with `VulkanPlatform`** is the platform's and not the runtime's: the OS's VM,
//! clocks and threads ([`crate::os`]), the logger, and the degradation counts ([`crate::answer`]).
//! It keeps none of the old runtime's device tables, device checks, vendor, device or driver ID
//! branches, or queue-sharing policies: rt has none.
//!
//! rt's floor refuses MoltenVK (it lacks `nullDescriptor`), so the platform is tested on Linux
//! GPUs. On a Mac it builds.

pub mod answers;
pub mod blit;
pub mod buffer;
pub mod command;
pub mod compute;
pub mod copied;
pub mod descriptor;
pub mod device;
pub mod diag;
pub mod event;
#[cfg(unix)]
pub mod export;
pub mod format;
pub mod heap;
pub mod library;
pub mod pipeline;
pub mod queue;
pub mod render;
pub mod render_pipeline;
pub mod state;
#[cfg(unix)]
pub mod surface;
pub mod swapchain;
pub mod texture;
pub mod tiled;
pub mod unbuilt;
pub mod unmade;

use std::mem::ManuallyDrop;
use std::sync::{Arc, Weak};
use std::time::Duration;

use metal2vulkan_rt as rt;

use crate::core::num::ToU64 as _;
use crate::core::platform::os::Threads;
use crate::core::platform::os::clock::{Clock as _, MonoTime};
use crate::core::platform::os::vm::HostVm;
use crate::core::platform::types::MtlSize;
use crate::core::platform::values::{RateMapDesc, TextureDesc};
use crate::core::platform::{Handle, ObjId, Platform};
use crate::core::raise::{OnFault, RaiseResult};
use crate::core::sync::Mutex;
use crate::os::{OsClock, OsThreads, OsVm};

pub use blit::RtBlitEncoder;
pub use buffer::{KnownBlocks, RtBuffer, import_known};
pub use command::RtCommandBuffer;
pub use compute::RtComputeEncoder;
pub use descriptor::{PassFields, RateMapFields, TextureFields};
pub use device::RtDevice;
pub use heap::RtHeap;
pub use library::{RtFunction, RtLibrary};
pub use pipeline::RtComputePipeline;
pub use queue::RtQueue;
pub use render::RtRenderEncoder;
pub use render_pipeline::RtRenderPipeline;
pub use state::RtSampler;
pub use texture::RtTexture;
pub use unbuilt::Unbuilt;
pub use unmade::{Unmade, kinds};

use unbuilt::{RtRefusal, refuse, refuse_create, refused_again};

/// A reference to an object the rt platform made itself: `clone_ref` takes another, and the last
/// one's drop frees the object. Its identity is its allocation's address.
pub struct RtObj<T: Send + Sync + 'static>(ManuallyDrop<Arc<T>>);

impl<T: Send + Sync + 'static> RtObj<T> {
    /// A new object holding `value`, with its one reference.
    pub fn new(value: T) -> RtObj<T> {
        RtObj(ManuallyDrop::new(Arc::new(value)))
    }

    /// A reference that does not hold the object.
    pub fn downgrade(&self) -> Weak<T> {
        Arc::downgrade(&self.0)
    }
}

impl<T: Send + Sync + 'static> Drop for RtObj<T> {
    /// The release. While the process records its calls ([`crate::recorder`]), the last
    /// reference's is the object's `release` record, written before the object goes.
    fn drop(&mut self) {
        if crate::recorder::on() {
            // SAFETY: the reference is taken once, here, and never touched after.
            let arc = unsafe { ManuallyDrop::take(&mut self.0) };
            let ptr = Arc::as_ptr(&arc).addr().to_u64();
            if let Some(inner) = Arc::into_inner(arc) {
                crate::recorder::released(ptr);
                drop(inner);
            }
            return;
        }
        // SAFETY: the reference is released once, here, and never touched after.
        unsafe { ManuallyDrop::drop(&mut self.0) }
    }
}

impl<T: Send + Sync + 'static> std::ops::Deref for RtObj<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T: Send + Sync + 'static> std::fmt::Debug for RtObj<T> {
    /// Its pointer and its type; while formatted for the recorder's stream, its stream id
    /// ([`crate::recorder::fmt_obj`]).
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let type_name = std::any::type_name::<T>();
        if crate::recorder::on() && crate::recorder::is_formatting() {
            return crate::recorder::fmt_obj(self.id().0, type_name, f);
        }
        write!(f, "<{:#x} {type_name}>", self.id().0)
    }
}

impl<T: Send + Sync + 'static> Handle for RtObj<T> {
    fn clone_ref(&self) -> Self {
        RtObj(ManuallyDrop::new(Arc::clone(&self.0)))
    }
    fn id(&self) -> ObjId {
        ObjId(Arc::as_ptr(&self.0).addr().to_u64())
    }
}

/// `MTLSharedEventListener`: the graphics lane's, which a shared event's notifications run on.
/// rt runs them on its completion thread ([`event`]), so it names no thread of its own.
pub struct Listener;

/// The log target of the platform's lines.
const TARGET: &str = "reims_vgpu_vulkan::rt_platform";

/// The process's platform over metal2vulkan-rt (the module's docs): the OS's VM, clocks and
/// threads, the process's one rt device once asked for, and rt's objects as its pieces are built.
///
/// **Its device goes before the exit.** The release of the last graphics device lets go of the
/// device ([`RtPlatform::let_go_of_device`]), which rt then tears down, as metal2vulkan-rt's
/// contract asks of a process that exits: it drops the device's last handle first, and a process
/// that cannot quiesces and ends with `_exit` (its README, "Quiesce and exit"; on NVIDIA's driver
/// an exit with a device live that ran work can end in `SIGSEGV` inside the driver's exit
/// handler). The quiesce ahead of the exit ([`Platform::quiesce`]) answers false while a device
/// lives, so that the host ends with `_exit`.
pub struct RtPlatform {
    vm: OsVm,
    clock: OsClock,
    threads: OsThreads,
    /// The device, made at the first ask ([`device::open`]), or the kind of rt's refusal to make
    /// it, which every ask answers; empty until the first ask, and again once the device was let
    /// go of ([`RtPlatform::let_go_of_device`]), for the next ask to make a new one.
    device: Mutex<Option<Result<RtObj<RtDevice>, RtRefusal>>>,
    /// The devices a let-go took and did not see torn down within its bound
    /// ([`RtPlatform::let_go_of_device`]), for the exit's quiesce, which answers false while one
    /// lives.
    unended: Mutex<Vec<Unended>>,
}

/// A device a let-go took and did not see torn down within its bound
/// ([`RtPlatform::let_go_of_device`]).
struct Unended {
    /// The device, where rt's quiesce of it answered false: kept to the exit, since its drop
    /// would wait with no deadline for the threads and the GPU work the quiesce did not see end.
    device: Option<RtObj<RtDevice>>,
    /// rt's counters of it, whose gauges all read zero once it is torn down.
    counters: rt::Counters,
    /// The let-go's handle of the instance it was made on, where the let-go's wait for its
    /// teardown ran out: kept to the exit, since rt's README (Teardown) has the consumer wait for
    /// the teardown before it drops its instance.
    _instance: Option<rt::Instance>,
}

impl Unended {
    /// Whether the device is torn down.
    fn ended(&self) -> bool {
        self.device.is_none() && self.counters.gauges() == rt::Gauges::default()
    }
}

impl Default for RtPlatform {
    fn default() -> RtPlatform {
        RtPlatform::new()
    }
}

/// The bound of the let-go of a platform's drop: the release's (reims-vgpu's `LET_GO_BOUND`).
const DROP_LET_GO: Duration = Duration::from_secs(5);

/// A platform that drops lets go of its device first ([`RtPlatform::let_go_of_device`]), so that
/// rt's teardown of it has ended when the drop returns. metal2vulkan-rt's README (Teardown):
/// "Every gauge reads zero only once the `VkDevice` is destroyed, so a consumer whose last handle
/// dropped in a handler waits for that before it drops its instance or exits." Where the
/// device's last handle goes in a shared event's notification or a command buffer's handler, on
/// rt's thread, rt tears the device down on its reaper once that returns, and a process ending
/// meanwhile races the teardown. The let-go holds the platform's handle until rt's threads ended
/// and drops it on the dropping thread, then waits, bounded, for the teardown, and drops its
/// handle of the instance last. The process's platform lives to the exit, so this serves the
/// platforms tests make. A let-go past its bound is logged with what was left (handles beyond
/// the platform's, rt's threads or GPU work, or gauges not at zero), and the devices and
/// instances kept to the exit are left undropped, since their drops would wait with no deadline
/// or come ahead of the teardown.
impl Drop for RtPlatform {
    fn drop(&mut self) {
        if let Err(held) = self.let_go_of_device(DROP_LET_GO) {
            log::warn!(
                target: TARGET,
                "the platform's drop left its device: after {DROP_LET_GO:?}, {held} still held it"
            );
        }
        std::mem::forget(std::mem::take(&mut *self.unended.lock()));
    }
}

impl std::fmt::Debug for RtPlatform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RtPlatform").finish_non_exhaustive()
    }
}

impl RtPlatform {
    /// The process's platform. Make one per process (D17). It installs what the OS's services need
    /// first ([`crate::os::install_process_services`]).
    pub fn new() -> RtPlatform {
        crate::os::install_process_services();
        crate::recorder::start_from_env();
        RtPlatform {
            vm: OsVm::new(),
            clock: OsClock::new(),
            threads: OsThreads::new(),
            device: Mutex::new(None),
            unended: Mutex::new(Vec::new()),
        }
    }

    /// `MTLCreateSystemDefaultDevice()`: the process's one device ([`device`]'s docs), made at the
    /// first ask; `None` where rt refuses to make it, counted at each ask under the kind of its
    /// refusal, and logged at the first.
    pub fn system_default_device(&self) -> Option<RtObj<RtDevice>> {
        self.device_at("MTLCreateSystemDefaultDevice")
    }

    /// The device, made at the first ask after the platform's start or a let-go, from `sel`
    /// (the type's docs).
    fn device_at(&self, sel: &'static str) -> Option<RtObj<RtDevice>> {
        let mut made_now = false;
        let mut slot = self.device.lock();
        let device = slot.get_or_insert_with(|| {
            made_now = true;
            device::open(self.vm.page_size()).map_err(|unopened| unopened.refuse(sel))
        });
        match device {
            Ok(device) => Some(device.clone_ref()),
            Err(kind) => {
                if !made_now {
                    refused_again(*kind);
                }
                None
            }
        }
    }

    /// Lets go of the process's device: what the release of the last graphics device does
    /// (reims-vgpu's `ApiPlatform::let_go_of_devices`), so that no device of rt's is live as the
    /// process exits (the type's docs). In order, each wait ending by `bound`:
    /// - it waits for every handle beyond the platform's own to go (an object, a command buffer
    ///   still running, a thread that holds one), looking every [`crate::let_go::POLL`]. Where
    ///   one outlives the wait, nothing goes, and `Err` says how many still held it;
    /// - rt's `Device::quiesce`, so that the drop's wait for rt's threads and GPU work, which
    ///   has no deadline, finds them ended. Where it answers false, the device is kept to the
    ///   exit, and `Err` says so;
    /// - the drop of the platform's handle, on this thread, which tears the device down where it
    ///   was rt's last; then the wait for the teardown, rt's gauges of the device reading zero, as
    ///   they do once it is torn down; then the drop of the let-go's handle of the instance the
    ///   device was made on, which rt's README (Teardown) puts after that wait. Where rt's objects
    ///   of it outlive the wait, `Err` names them, and the instance's handle is kept to the exit.
    ///
    /// A device taken and not torn down makes the exit's quiesce answer false while it lives.
    /// The next ask makes a new device, whose IDs start from new pools. `Ok`, with nothing let
    /// go of, where no device was made, where rt refused to make it (every ask still answers
    /// the refusal), and once the process's exit began (the exit gate closed): a thread the drop
    /// would join may be parked for good then.
    pub fn let_go_of_device(&self, bound: Duration) -> Result<(), String> {
        if crate::exit_gate::closed() {
            return Ok(());
        }
        let until = self.clock.now_mono().saturating_add(bound);
        let device = loop {
            let mut slot = self.device.lock();
            let beyond = match slot.as_ref() {
                Some(Ok(device)) => Arc::strong_count(&device.0).saturating_sub(1),
                Some(Err(_)) | None => return Ok(()),
            };
            if beyond == 0
                && let Some(Ok(device)) = slot.take()
            {
                break device;
            }
            drop(slot);
            if self.clock.now_mono() >= until {
                return Err(format!(
                    "the rt device by {beyond} handles beyond the platform's"
                ));
            }
            std::thread::sleep(crate::let_go::POLL);
        };
        let facts = device.facts();
        let named = format!("{} (registry ID {:#x})", facts.name, facts.registry_id);
        let counters = device.rt().counters();
        let left = until.saturating_duration_since(self.clock.now_mono());
        if !device.rt().quiesce(left) {
            self.unended.lock().push(Unended {
                device: Some(device),
                counters,
                _instance: None,
            });
            return Err(format!(
                "rt's threads or GPU work on {named}, which its quiesce did not see end,"
            ));
        }
        let instance = device.instance().clone();
        drop(device);
        match self.wait_for_teardown(&counters, until) {
            Ok(()) => {
                drop(instance);
                log::debug!(
                    target: "reims_vgpu_vulkan::let_go",
                    "let go of 1 GPU devices: rt's {named}"
                );
                Ok(())
            }
            Err(gauges) => {
                self.unended.lock().push(Unended {
                    device: None,
                    counters,
                    _instance: Some(instance),
                });
                Err(format!(
                    "rt's objects of {named}, its gauges not at zero ({gauges:?}),"
                ))
            }
        }
    }

    /// Waits, until `until`, for rt's teardown of the device whose counters are `counters`, all
    /// of whose handles are gone: `Err` with the gauges where they did not all read zero by then.
    fn wait_for_teardown(
        &self,
        counters: &rt::Counters,
        until: MonoTime,
    ) -> Result<(), Box<rt::Gauges>> {
        loop {
            let gauges = counters.gauges();
            if gauges == rt::Gauges::default() {
                return Ok(());
            }
            if self.clock.now_mono() >= until {
                return Err(Box::new(gauges));
            }
            std::thread::sleep(crate::let_go::POLL);
        }
    }
}

impl Platform for RtPlatform {
    type Device = RtObj<RtDevice>;
    type CommandQueue = RtObj<RtQueue>;
    type CommandBuffer = RtObj<RtCommandBuffer>;
    type BlitEncoder = RtObj<RtBlitEncoder>;
    type ComputeEncoder = RtObj<RtComputeEncoder>;
    type RenderEncoder = RtObj<RtRenderEncoder>;
    type Buffer = RtObj<RtBuffer>;
    type Texture = RtObj<RtTexture>;
    type Heap = RtObj<RtHeap>;
    type SamplerState = RtObj<RtSampler>;
    type DepthStencilState = RtObj<metal2vulkan_rt::DepthStencilState>;
    type Fence = RtObj<metal2vulkan_rt::Fence>;
    type Event = RtObj<metal2vulkan_rt::Event>;
    type SharedEvent = RtObj<metal2vulkan_rt::SharedEvent>;
    type SharedEventListener = RtObj<Listener>;
    type SharedTextureHandle = Unmade<kinds::SharedTextureHandle>;
    type Library = RtObj<RtLibrary>;
    type Function = RtObj<RtFunction>;
    type ComputePipeline = RtObj<RtComputePipeline>;
    type FunctionHandle = Unmade<kinds::FunctionHandle>;
    type VisibleFunctionTable = Unmade<kinds::VisibleFunctionTable>;
    type DynamicLibrary = Unmade<kinds::DynamicLibrary>;
    type RenderPipeline = RtObj<RtRenderPipeline>;
    type RateMap = Unmade<kinds::RateMap>;
    type Icb = Unmade<kinds::Icb>;
    type IndirectRenderCommand<'a> = Unmade<kinds::IndirectRenderCommand>;
    type IndirectComputeCommand<'a> = Unmade<kinds::IndirectComputeCommand>;
    type TextureDescriptor = RtObj<TextureFields>;
    type RenderPassDescriptor = RtObj<PassFields>;
    type RateMapDescriptor = RtObj<RateMapFields>;
    type RateMapLayerDescriptor = Unmade<kinds::RateMapLayerDescriptor>;
    type Clock = OsClock;

    fn new_render_pass_descriptor(&self) -> RtObj<PassFields> {
        let recorded_call = crate::recorder::rec_call!("MTLRenderPassDescriptor.new", self);
        let recorded_answer =
            crate::recorder::run(|| -> RtObj<PassFields> { RtObj::new(PassFields::default()) });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    fn new_shared_event_listener(&self) -> RtObj<Listener> {
        RtObj::new(Listener)
    }

    fn render_pass_descriptor(&self) -> RtObj<PassFields> {
        let recorded_call =
            crate::recorder::rec_call!("MTLRenderPassDescriptor.renderPassDescriptor", self);
        let recorded_answer =
            crate::recorder::run(|| -> RtObj<PassFields> { RtObj::new(PassFields::default()) });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    fn texture_2d_descriptor(
        &self,
        on: OnFault,
        pixel_format: u64,
        width: u64,
        height: u64,
        mipmapped: bool,
    ) -> RaiseResult<RtObj<TextureFields>> {
        let recorded_call = crate::recorder::rec_call!(
            "MTLTextureDescriptor.texture2DDescriptorWithPixelFormat:width:height:mipmapped:",
            self,
            pixel_format,
            width,
            height,
            mipmapped
        );
        let recorded_answer = crate::recorder::run(|| -> RaiseResult<RtObj<TextureFields>> {
            let _ = on;
            let desc = descriptor::texture_2d_desc(pixel_format, width, height, mipmapped);
            Ok(RtObj::new(TextureFields::new(desc)))
        });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// Unbuilt: nil ([`Unbuilt::RateMap`]), as Metal answers for a count it refuses.
    fn new_rate_map_layer(
        &self,
        on: OnFault,
        sample_count: MtlSize,
    ) -> RaiseResult<Option<Unmade<kinds::RateMapLayerDescriptor>>> {
        let recorded_call = crate::recorder::rec_call!(
            "MTLRasterizationRateLayerDescriptor.initWithSampleCount:",
            self,
            sample_count
        );
        let recorded_answer = crate::recorder::run(
            || -> RaiseResult<Option<Unmade<kinds::RateMapLayerDescriptor>>> {
                let _ = (on, sample_count);
                refuse_create(Unbuilt::RateMap, "initWithSampleCount:");
                Ok(None)
            },
        );
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    fn texture_descriptor(&self, desc: &TextureDesc) -> RtObj<TextureFields> {
        let recorded_call = crate::recorder::rec_call!("m2v.textureDescriptor", self, desc);
        let recorded_answer = crate::recorder::run(|| -> RtObj<TextureFields> {
            RtObj::new(TextureFields::new(*desc))
        });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// Unbuilt ([`Unbuilt::RateMap`]): a descriptor that keeps nothing.
    fn rate_map_descriptor(&self, desc: &RateMapDesc) -> RtObj<RateMapFields> {
        let recorded_call = crate::recorder::rec_call!("m2v.rateMapDescriptor", self, desc);
        let recorded_answer = crate::recorder::run(|| -> RtObj<RateMapFields> {
            let _ = desc;
            refuse(Unbuilt::RateMap, "MTLRasterizationRateMapDescriptor");
            RtObj::new(RateMapFields)
        });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// The process's one device ([`RtPlatform::system_default_device`]) where `registry_id` is
    /// its registry ID; `None` for any other, as Metal answers for an ID no device has.
    fn open_device(&self, registry_id: u64) -> Option<RtObj<RtDevice>> {
        let recorded_call =
            crate::recorder::rec_call!("MTLCreateSystemDefaultDevice", self, registry_id);
        let recorded_answer = crate::recorder::run(|| -> Option<RtObj<RtDevice>> {
            let device = self.device_at("MTLCopyAllDevices")?;
            (device.facts().registry_id == registry_id).then_some(device)
        });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
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

    /// The library's threads first, then rt's: closes the exit gate ahead of the exit, so that a
    /// thread of the library's parks at its next step, and waits for the steps in progress on
    /// other threads ([`crate::exit_gate::close_ahead`]); then, where the process's device lives,
    /// rt's `Device::quiesce` of it with what is left of `bound`: rt's submit thread submits what
    /// is ready and ends the rest unrun, the GPU completes what was submitted, and rt's threads
    /// end. Nothing is freed. True where the steps ended in time and no device of rt's lives:
    /// none was made, or the release of the last graphics device let go of it
    /// ([`RtPlatform::let_go_of_device`]). False where one lives, quiesced or not, since rt
    /// supports no `exit()` with a device live (metal2vulkan-rt's README, "Quiesce and exit"),
    /// and where a step outlasted the wait, which it warns of.
    fn quiesce(&self, bound: Duration) -> bool {
        let until = self.clock.now_mono().saturating_add(bound);
        let parked = crate::exit_gate::close_ahead(until);
        let live = match self.device.lock().as_ref() {
            Some(Ok(device)) => Some(device.clone_ref()),
            Some(Err(_)) | None => None,
        };
        let stopped = live.as_ref().is_none_or(|device| {
            let left = until.saturating_duration_since(self.clock.now_mono());
            device.rt().quiesce(left)
        });
        if !(parked && stopped) {
            log::warn!(
                target: TARGET,
                "the quiesce ahead of the exit ended after {bound:?} with a thread still in the \
                 GPU's work"
            );
        }
        let unended = (self.unended.lock().iter())
            .filter(|unended| !unended.ended())
            .count();
        let live = usize::from(live.is_some()).saturating_add(unended);
        if live > 0 {
            log::debug!(
                target: "reims_vgpu_vulkan::let_go",
                "{live} GPU devices live at the exit: the process ends with _exit"
            );
        }
        parked && live == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::platform::descriptor::TextureDescriptor;

    /// The count of `code` in the process's degradation counts.
    fn counted(code: &str) -> u64 {
        (crate::answer::degradation_counts().into_iter())
            .find(|(c, _)| c == code)
            .map_or(0, |(_, n)| n)
    }

    /// A rate map's layer is refused as nil and counted; its descriptor keeps nothing and reads as
    /// one with no layers.
    #[test]
    fn an_unbuilt_rate_map_is_a_counted_typed_refusal() {
        use crate::core::platform::rate_map::RateMapDescriptor;
        let platform = RtPlatform::new();
        let code = Unbuilt::RateMap.code();
        let before = counted(code);
        let size = MtlSize {
            width: 2,
            height: 2,
            depth: 0,
        };
        assert!(matches!(
            platform.new_rate_map_layer(OnFault::Die, size),
            Ok(None)
        ));
        let descriptor = platform.rate_map_descriptor(&RateMapDesc::default());
        assert_eq!(RateMapDescriptor::<RtPlatform>::layer_count(&descriptor), 0);
        assert!(counted(code) >= before + 3, "{code}");
    }

    /// Commits 8 empty command buffers on a queue of `device`'s, each with a completed handler
    /// that counts its run in the answer.
    fn commit_work(
        device: &RtObj<RtDevice>,
    ) -> (RtObj<RtQueue>, Arc<std::sync::atomic::AtomicU32>) {
        use std::sync::atomic::{AtomicU32, Ordering};

        use crate::core::platform::command::{CommandBuffer as _, CommandQueue as _};
        use crate::core::platform::device::GpuDevice as _;

        let queue = device.new_command_queue().expect("a command queue");
        let ended = Arc::new(AtomicU32::new(0));
        for _ in 0..8 {
            let buffer = (queue
                .command_buffer(OnFault::Return)
                .expect("never a raise"))
            .expect("a command buffer");
            let ended = Arc::clone(&ended);
            let handler = Box::new(move |_: crate::core::platform::command::CompletedInfo| {
                ended.fetch_add(1, Ordering::SeqCst);
            });
            assert_eq!(
                buffer.add_completed_handler(OnFault::Return, handler),
                Ok(())
            );
            assert_eq!(buffer.commit(OnFault::Return), Ok(()));
        }
        (queue, ended)
    }

    /// Runs the test named `test` again in a child process, with `RVG_QUIESCE_CHILD` naming it:
    /// its exit status and its standard error.
    fn run_child(test: &str) -> (Option<i32>, String) {
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", test, "--nocapture", "--test-threads=1"])
            .env("RVG_QUIESCE_CHILD", test)
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
        println!("{stderr}");
        (out.status.code(), stderr)
    }

    /// A quiesce ahead of the exit with the device live, on the GPU the test holds, in a child
    /// process: work committed just before it ends, completed or unrun, with its handler run;
    /// rt's quiesce stopped in time, and the answer is false all the same, since rt supports no
    /// `exit()` with a device live; a second call answers false at once; the exit gate is
    /// closed; a command buffer made after it is refused and counted; and the child then ends
    /// with `_exit`, as the host does on false, with status 0.
    #[test]
    fn on_vulkan_a_quiesce_with_the_device_live_stops_it_and_answers_false() {
        use std::sync::atomic::Ordering;

        use crate::core::platform::command::CommandQueue as _;

        const TEST: &str = "rt_platform::tests::\
            on_vulkan_a_quiesce_with_the_device_live_stops_it_and_answers_false";
        unsafe extern "C" {
            /// The C library's `_exit`, which ends the process with no exit handler run; `safe`,
            /// as the testkit's death tests declare it on Linux and macOS, since a declaration of
            /// another signature clashes with theirs.
            safe fn _exit(status: i32) -> !;
        }
        let say = |line: &str| eprintln!("{line}");
        if std::env::var("RVG_QUIESCE_CHILD").as_deref() == Ok(TEST) {
            let platform = RtPlatform::new();
            let device = (platform.system_default_device()).expect("an rt device on the GPU");
            let (queue, ended) = commit_work(&device);
            let clock = crate::os::OsClock::new();
            let start = clock.now_mono();
            let first = platform.quiesce(Duration::from_secs(5));
            let took = clock.now_mono().saturating_duration_since(start);
            say(&format!(
                "quiesce answered {first} in {} ms",
                took.as_millis()
            ));
            say(&format!(
                "rt's quiesce answered {}",
                device.rt().quiesce(Duration::ZERO)
            ));
            say(&format!("handlers run {}", ended.load(Ordering::SeqCst)));
            let start = clock.now_mono();
            let second = platform.quiesce(Duration::from_secs(5));
            let took = clock.now_mono().saturating_duration_since(start);
            say(&format!(
                "a second quiesce answered {second}, at once {}",
                took < Duration::from_millis(100)
            ));
            say(&format!("gate closed {}", crate::exit_gate::closed()));
            let code = RtRefusal::Quiesced.code();
            let before = counted(code);
            let refused = matches!(queue.command_buffer(OnFault::Return), Ok(None));
            say(&format!(
                "a command buffer after it refused {refused}, counted {}",
                counted(code).saturating_sub(before)
            ));
            _exit(0)
        }
        let (status, stderr) = run_child(TEST);
        assert_eq!(status, Some(0), "{stderr}");
        for line in [
            "quiesce answered false",
            "rt's quiesce answered true",
            "handlers run 8",
            "a second quiesce answered false, at once true",
            "gate closed true",
            "a command buffer after it refused true, counted 1",
        ] {
            assert!(stderr.contains(line), "no {line:?}: {stderr}");
        }
    }

    /// The order the host's quit takes, on the GPU the test holds, in a child process: work
    /// committed, every handle of the device dropped but the platform's, then the let-go, which
    /// waits for the work and tears the device down on this thread (rt's gauges of it all read
    /// zero); then the quiesce ahead of the exit, which finds no device and answers true; then
    /// `exit()`, which runs the exit handlers, the driver's among them, and ends with status 0.
    #[test]
    fn on_vulkan_the_let_go_drops_the_device_so_the_quiesce_answers_true_and_the_exit_runs() {
        use std::sync::atomic::Ordering;

        const TEST: &str = "rt_platform::tests::\
            on_vulkan_the_let_go_drops_the_device_so_the_quiesce_answers_true_and_the_exit_runs";
        unsafe extern "C" {
            /// The C library's `exit`, which runs the exit handlers, as QEMU's exit does.
            fn exit(status: i32) -> !;
        }
        let say = |line: &str| eprintln!("{line}");
        if std::env::var("RVG_QUIESCE_CHILD").as_deref() == Ok(TEST) {
            let platform = RtPlatform::new();
            let device = (platform.system_default_device()).expect("an rt device on the GPU");
            let counters = device.rt().counters();
            let (queue, ended) = commit_work(&device);
            drop((queue, device));
            let let_go = platform.let_go_of_device(Duration::from_secs(5));
            say(&format!("let-go answered {let_go:?}"));
            say(&format!("handlers run {}", ended.load(Ordering::SeqCst)));
            say(&format!(
                "gauges zero {}",
                counters.gauges() == rt::Gauges::default()
            ));
            let clock = crate::os::OsClock::new();
            let start = clock.now_mono();
            let quiesced = platform.quiesce(Duration::from_secs(5));
            let took = clock.now_mono().saturating_duration_since(start);
            say(&format!(
                "quiesce answered {quiesced}, at once {}",
                took < Duration::from_millis(100)
            ));
            // SAFETY: the C library's exit, which never returns.
            unsafe { exit(0) }
        }
        let (status, stderr) = run_child(TEST);
        assert_eq!(status, Some(0), "{stderr}");
        for line in [
            "let-go answered Ok(())",
            "handlers run 8",
            "gauges zero true",
            "quiesce answered true, at once true",
        ] {
            assert!(stderr.contains(line), "no {line:?}: {stderr}");
        }
    }

    /// A platform dropped while a shared event's notification, running on rt's thread, still
    /// holds a handle of its device returns once rt tore the device down: rt's gauges of it read
    /// zero right after the drop. Without the let-go of the platform's drop, the notification's
    /// handle is the device's last, and rt tears the device down on its reaper after the drop
    /// returned.
    #[test]
    fn on_vulkan_a_platform_dropped_while_a_listener_holds_a_handle_waits_for_the_teardown() {
        use std::sync::mpsc;

        use crate::core::platform::device::GpuDevice as _;
        use crate::core::platform::event::{Notification, SharedEvent as _};

        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let counters = device.rt().counters();
        let event = device.new_shared_event().expect("a shared event");
        let listener = platform.new_shared_event_listener();
        let (running, runs) = mpsc::channel::<()>();
        let held = event.clone_ref();
        let notification: Notification = Box::new(move || {
            let _ = running.send(());
            std::thread::sleep(Duration::from_millis(300));
            drop(held);
        });
        assert_eq!(event.notify_listener(&listener, 1, notification), Ok(()));
        assert_eq!(event.set_signaled_value(1), Ok(()));
        (runs.recv_timeout(Duration::from_secs(10))).expect("the notification runs");
        drop((event, listener, device));
        drop(platform);
        assert_eq!(counters.gauges(), rt::Gauges::default());
    }

    /// A texture descriptor reads back what was set, as Metal's does; each made one is an object
    /// of its own.
    #[test]
    fn a_texture_descriptor_reads_back_what_was_set() {
        type D = RtObj<TextureFields>;
        let platform = RtPlatform::new();
        let d: D = platform.texture_descriptor(&TextureDesc::default());
        TextureDescriptor::<RtPlatform>::set_pixel_format(&d, 80);
        TextureDescriptor::<RtPlatform>::set_texture_type(&d, 3);
        TextureDescriptor::<RtPlatform>::set_resource_options(&d, 0x20);
        TextureDescriptor::<RtPlatform>::set_rotation(&d, 2);
        assert_eq!(TextureDescriptor::<RtPlatform>::pixel_format(&d), 80);
        assert_eq!(TextureDescriptor::<RtPlatform>::texture_type(&d), 3);
        assert_eq!(TextureDescriptor::<RtPlatform>::storage_mode(&d), 2);
        assert_eq!(TextureDescriptor::<RtPlatform>::rotation(&d), 2);
        assert_eq!(d.clone_ref().id(), d.id());
        let other: D = platform.texture_descriptor(&TextureDesc::default());
        assert_ne!(other.id(), d.id());
    }
}
