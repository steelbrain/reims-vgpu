//! The library's platform over metal2vulkan-rt, the runtime that replaces metal2vulkan-runtime at
//! the cutover: [`RtPlatform`], core's [`Platform`] over rt's public API. It is built under the
//! `rt` feature, which makes it reims-vgpu's `DefaultPlatform`. The feature is a transition
//! switch, and it goes at the cutover with the old runtime and `VulkanPlatform`. Without it nothing
//! here is compiled.
//!
//! **Built piece by piece**, in the order a guest's boot reaches the pieces: the device and its
//! answers, queues, buffers, textures, libraries and pipelines, command buffers and encoders,
//! events, then the display. Built: the device ([`device`]) with its answers ([`answers`] and
//! [`mod@format`]), its command queues ([`queue`]), events and fences, the reservation of resource
//! indices, and textures made of a descriptor, with their views ([`texture`]). Until a piece is
//! built, every call that reaches it is refused ([`unbuilt`]). The call answers as Metal's call
//! answers where it fails: a nil object, an `NSError`, zeros. The refusal is counted under the
//! piece's code, `rt-unbuilt-<piece>`, in the process's degradation counts, and the first of each
//! kind writes a line on stderr. Never a death or a panic.
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
pub mod descriptor;
pub mod device;
pub mod format;
pub mod queue;
pub mod texture;
pub mod unbuilt;
pub mod unmade;

use std::sync::{Arc, OnceLock};

use crate::core::num::ToU64 as _;
use crate::core::platform::os::Threads;
use crate::core::platform::os::vm::HostVm;
use crate::core::platform::types::MtlSize;
use crate::core::platform::values::{RateMapDesc, TextureDesc};
use crate::core::platform::{Handle, ObjId, Platform};
use crate::core::raise::{OnFault, RaiseResult};
use crate::os::{OsClock, OsThreads, OsVm};

pub use descriptor::{PassFields, RateMapFields, TextureFields};
pub use device::RtDevice;
pub use queue::RtQueue;
pub use texture::RtTexture;
pub use unbuilt::Unbuilt;
pub use unmade::{Unmade, kinds};

use unbuilt::{RtRefusal, refuse, refuse_create, refused_again};

/// A reference to an object the rt platform made itself: `clone_ref` takes another, and the last
/// one's drop frees the object. Its identity is its allocation's address.
pub struct RtObj<T: Send + Sync + 'static>(Arc<T>);

impl<T: Send + Sync + 'static> RtObj<T> {
    /// A new object holding `value`, with its one reference.
    pub fn new(value: T) -> RtObj<T> {
        RtObj(Arc::new(value))
    }
}

impl<T: Send + Sync + 'static> std::ops::Deref for RtObj<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T: Send + Sync + 'static> Handle for RtObj<T> {
    fn clone_ref(&self) -> Self {
        RtObj(Arc::clone(&self.0))
    }
    fn id(&self) -> ObjId {
        ObjId(Arc::as_ptr(&self.0).addr().to_u64())
    }
}

/// `MTLSharedEventListener`: the graphics lane's, which a shared event's notifications run on.
/// The shared events' piece runs them.
pub struct Listener;

/// The process's platform over metal2vulkan-rt (the module's docs): the OS's VM, clocks and
/// threads, the process's one rt device once asked for, and rt's objects as its pieces are built.
pub struct RtPlatform {
    vm: OsVm,
    clock: OsClock,
    threads: OsThreads,
    /// The device, made at the first ask ([`device::open`]), or the kind of rt's refusal to make
    /// it, which every ask answers.
    device: OnceLock<Result<RtObj<RtDevice>, RtRefusal>>,
}

impl Default for RtPlatform {
    fn default() -> RtPlatform {
        RtPlatform::new()
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
        RtPlatform {
            vm: OsVm::new(),
            clock: OsClock::new(),
            threads: OsThreads::new(),
            device: OnceLock::new(),
        }
    }

    /// `MTLCreateSystemDefaultDevice()`: the process's one device ([`device`]'s docs), made at the
    /// first ask; `None` where rt refuses to make it, counted at each ask under the kind of its
    /// refusal, and logged at the first.
    pub fn system_default_device(&self) -> Option<RtObj<RtDevice>> {
        self.device_at("MTLCreateSystemDefaultDevice")
    }

    /// The device, made at the first ask, from `sel` (the type's docs).
    fn device_at(&self, sel: &'static str) -> Option<RtObj<RtDevice>> {
        let mut made_now = false;
        let device = self.device.get_or_init(|| {
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
}

impl Platform for RtPlatform {
    type Device = RtObj<RtDevice>;
    type CommandQueue = RtObj<RtQueue>;
    type CommandBuffer = Unmade<kinds::CommandBuffer>;
    type BlitEncoder = Unmade<kinds::BlitEncoder>;
    type ComputeEncoder = Unmade<kinds::ComputeEncoder>;
    type RenderEncoder = Unmade<kinds::RenderEncoder>;
    type Buffer = Unmade<kinds::Buffer>;
    type Texture = RtObj<RtTexture>;
    type Heap = Unmade<kinds::Heap>;
    type SamplerState = Unmade<kinds::SamplerState>;
    type DepthStencilState = Unmade<kinds::DepthStencilState>;
    type Fence = RtObj<metal2vulkan_rt::Fence>;
    type Event = RtObj<metal2vulkan_rt::Event>;
    type SharedEvent = Unmade<kinds::SharedEvent>;
    type SharedEventListener = RtObj<Listener>;
    type SharedTextureHandle = Unmade<kinds::SharedTextureHandle>;
    type Library = Unmade<kinds::Library>;
    type Function = Unmade<kinds::Function>;
    type ComputePipeline = Unmade<kinds::ComputePipeline>;
    type FunctionHandle = Unmade<kinds::FunctionHandle>;
    type VisibleFunctionTable = Unmade<kinds::VisibleFunctionTable>;
    type DynamicLibrary = Unmade<kinds::DynamicLibrary>;
    type RenderPipeline = Unmade<kinds::RenderPipeline>;
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
        RtObj::new(PassFields)
    }

    fn new_shared_event_listener(&self) -> RtObj<Listener> {
        RtObj::new(Listener)
    }

    fn render_pass_descriptor(&self) -> RtObj<PassFields> {
        RtObj::new(PassFields)
    }

    fn texture_2d_descriptor(
        &self,
        on: OnFault,
        pixel_format: u64,
        width: u64,
        height: u64,
        mipmapped: bool,
    ) -> RaiseResult<RtObj<TextureFields>> {
        let _ = on;
        let desc = descriptor::texture_2d_desc(pixel_format, width, height, mipmapped);
        Ok(RtObj::new(TextureFields::new(desc)))
    }

    /// Unbuilt: nil ([`Unbuilt::RateMap`]), as Metal answers for a count it refuses.
    fn new_rate_map_layer(
        &self,
        on: OnFault,
        sample_count: MtlSize,
    ) -> RaiseResult<Option<Unmade<kinds::RateMapLayerDescriptor>>> {
        let _ = (on, sample_count);
        refuse_create(Unbuilt::RateMap, "initWithSampleCount:");
        Ok(None)
    }

    fn texture_descriptor(&self, desc: &TextureDesc) -> RtObj<TextureFields> {
        RtObj::new(TextureFields::new(*desc))
    }

    /// Unbuilt ([`Unbuilt::RateMap`]): a descriptor that keeps nothing.
    fn rate_map_descriptor(&self, desc: &RateMapDesc) -> RtObj<RateMapFields> {
        let _ = desc;
        refuse(Unbuilt::RateMap, "MTLRasterizationRateMapDescriptor");
        RtObj::new(RateMapFields)
    }

    /// The process's one device ([`RtPlatform::system_default_device`]) where `registry_id` is
    /// its registry ID; `None` for any other, as Metal answers for an ID no device has.
    fn open_device(&self, registry_id: u64) -> Option<RtObj<RtDevice>> {
        let device = self.device_at("MTLCopyAllDevices")?;
        (device.facts().registry_id == registry_id).then_some(device)
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
