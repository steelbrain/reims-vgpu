//! Fake: `FakePlatform`: every platform trait over the fake object models (architecture 13.2).
//!
//! One `FakePlatform` is one fake process: a fresh [`World`] (call log, script, virtual clock,
//! address space, GPU VAs, resource indices) and one default device that `open_device` finds.
//!
//! ```ignore
//! let p = Arc::new(FakePlatform::new());
//! p.script().push("newBufferWithLength:options:", Step::Nil);  // Metal returns nil once
//! // … run the code under test with `p` …
//! assert!(p.call_log().contains_in_order(&["newCommandQueue", "setLabel:"]));
//! p.world().complete_all(CompleteOn::Thread);                   // the GPU finishes
//! ```

use crate::num::ToU64;
use crate::testkit::script::Policy;
use std::sync::Arc;

use super::calllog::{Arg, CallLog};
use super::clock::VirtualClock;
use super::faults::Faults;
use super::impls::descriptor::TextureDescriptorModel;
use super::impls::device::DeviceModel;
use super::objects::{Fk, Model, TexDesc, World, kind};
use super::script::Script;
use super::surfaces::FakeSurfaces;
use super::threads::FakeThreads;
use super::vm::FakeVm;
use crate::platform::os::Threads;
use crate::platform::os::surface::IoSurfaces;
use crate::platform::os::vm::HostVm;
use crate::platform::types::{MtlSize, ResourceAddressRange};
use crate::platform::values::{RateMapDesc, TextureDesc};
use crate::platform::{Handle, ObjId, Platform};
use crate::raise::{OnFault, RaiseResult};

/// The fake's device registry ID (`-[MTLDevice registryID]`).
pub const FAKE_REGISTRY_ID: u64 = 0x1_0000_0f00;

/// The fake's handle types, by `Platform` associated type.
pub mod types {
    use super::super::objects::{Fk, kind};
    /// `id<MTLDevice>`.
    pub type FakeDevice = Fk<kind::Device>;
    /// `id<MTLCommandQueue>`.
    pub type FakeCommandQueue = Fk<kind::CommandQueue>;
    /// `id<MTLCommandBuffer>`.
    pub type FakeCommandBuffer = Fk<kind::CommandBuffer>;
    /// `id<MTLBuffer>`.
    pub type FakeBuffer = Fk<kind::Buffer>;
    /// `id<MTLTexture>`.
    pub type FakeTexture = Fk<kind::Texture>;
    /// `id<MTLHeap>`.
    pub type FakeHeap = Fk<kind::Heap>;
    /// `id<MTLSharedEvent>`.
    pub type FakeSharedEvent = Fk<kind::SharedEvent>;
    /// `MTLTextureDescriptor *`.
    pub type FakeTextureDescriptor = Fk<kind::TextureDescriptor>;
}

/// The fake platform: one fake process.
pub struct FakePlatform {
    world: Arc<World>,
    device: Fk<kind::Device>,
    surfaces: FakeSurfaces,
    threads: FakeThreads,
    registry_id: u64,
}

impl std::fmt::Debug for FakePlatform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FakePlatform")
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

impl Default for FakePlatform {
    fn default() -> Self {
        FakePlatform::new()
    }
}

impl FakePlatform {
    /// A fresh fake process with one device.
    pub fn new() -> FakePlatform {
        let world = World::new();
        let device = world.new_obj(Model::Device(DeviceModel::default()));
        FakePlatform {
            surfaces: FakeSurfaces::new(world.clone()),
            threads: FakeThreads::for_world(world.clone()),
            registry_id: FAKE_REGISTRY_ID,
            device,
            world,
        }
    }

    /// The shared world (models, completion, lookups).
    pub fn world(&self) -> &Arc<World> {
        &self.world
    }

    /// The call log.
    pub fn call_log(&self) -> &CallLog {
        &self.world.log
    }

    /// Starts or stops recording this fake process's events: its objects' creations and last
    /// releases with their release context ([`CallLog::record_events`]).
    pub fn record_events(&self, on: bool) {
        self.world.log.record_events(on);
    }

    /// The events recorded so far ([`CallLog::events`]).
    pub fn events(&self) -> Vec<super::calllog::EventRecord> {
        self.world.log.events()
    }

    /// The script.
    pub fn script(&self) -> &Script {
        &self.world.script
    }

    /// Faults keyed by class and selector, over the script.
    pub fn faults(&self) -> Faults {
        Faults::new(self.world.script.clone())
    }

    /// The virtual clock.
    pub fn virtual_clock(&self) -> &VirtualClock {
        &self.world.clock
    }

    /// The address space.
    pub fn fake_vm(&self) -> &FakeVm {
        &self.world.vm
    }

    /// The IOSurface factory.
    pub fn fake_surfaces(&self) -> &FakeSurfaces {
        &self.surfaces
    }

    /// The thread recorder.
    pub fn fake_threads(&self) -> &FakeThreads {
        &self.threads
    }

    /// A new reference to the fake's device (what a descriptor's `device` holds).
    pub fn device(&self) -> Fk<kind::Device> {
        self.device.clone_ref()
    }

    /// A class method whose position is observable: logged, then its scripted step.
    fn class_call(
        &self,
        class: &'static str,
        sel: &'static str,
        args: Vec<Arg>,
    ) -> super::script::Step {
        self.world.log.push(ObjId(0), class, sel, args);
        self.world.script.next_for(ObjId(0), class, sel)
    }

    /// A fresh descriptor from a class method no one observes the position of (`+new`,
    /// `+renderPassDescriptor`): neither logged nor scripted.
    fn fresh<K: super::objects::Kind>(&self) -> Fk<K> {
        self.world.new_obj(Model::None)
    }
}

impl Platform for FakePlatform {
    type Device = Fk<kind::Device>;
    type CommandQueue = Fk<kind::CommandQueue>;
    type CommandBuffer = Fk<kind::CommandBuffer>;
    type BlitEncoder = Fk<kind::BlitEncoder>;
    type ComputeEncoder = Fk<kind::ComputeEncoder>;
    type RenderEncoder = Fk<kind::RenderEncoder>;
    type Buffer = Fk<kind::Buffer>;
    type Texture = Fk<kind::Texture>;
    type Heap = Fk<kind::Heap>;
    type SamplerState = Fk<kind::SamplerState>;
    type DepthStencilState = Fk<kind::DepthStencilState>;
    type Fence = Fk<kind::Fence>;
    type Event = Fk<kind::Event>;
    type SharedEvent = Fk<kind::SharedEvent>;
    type SharedEventHandle = Fk<kind::SharedEventHandle>;
    type SharedEventListener = Fk<kind::SharedEventListener>;
    type SharedTextureHandle = Fk<kind::SharedTextureHandle>;
    type Library = Fk<kind::Library>;
    type Function = Fk<kind::Function>;
    type ComputePipeline = Fk<kind::ComputePipeline>;
    type RenderPipeline = Fk<kind::RenderPipeline>;
    type RateMap = Fk<kind::RateMap>;
    type Icb = Fk<kind::Icb>;
    type IndirectRenderCommand = Fk<kind::IndirectRenderCommand>;
    type IndirectComputeCommand = Fk<kind::IndirectComputeCommand>;
    type TextureDescriptor = Fk<kind::TextureDescriptor>;
    type RenderPassDescriptor = Fk<kind::RenderPassDescriptor>;
    type RenderPipelineDescriptor = Fk<kind::RenderPipelineDescriptor>;
    type ComputePipelineDescriptor = Fk<kind::ComputePipelineDescriptor>;
    type TileRenderPipelineDescriptor = Fk<kind::TileRenderPipelineDescriptor>;
    type RateMapDescriptor = Fk<kind::RateMapDescriptor>;
    type RateMapLayerDescriptor = Fk<kind::RateMapLayerDescriptor>;
    type AddressRangeArray = Fk<kind::AddressRangeArray>;
    type Surface = Fk<kind::Surface>;
    type Clock = VirtualClock;

    fn new_render_pass_descriptor(&self) -> Self::RenderPassDescriptor {
        self.fresh()
    }

    fn new_shared_event_listener(&self) -> Self::SharedEventListener {
        self.class_call("MTLSharedEventListener", "new", vec![])
            .getter("new");
        self.world.new_obj(Model::None)
    }

    fn render_pass_descriptor(&self) -> Self::RenderPassDescriptor {
        self.fresh()
    }

    fn texture_2d_descriptor(
        &self,
        on: OnFault,
        pixel_format: u64,
        width: u64,
        height: u64,
        mipmapped: bool,
    ) -> RaiseResult<Self::TextureDescriptor> {
        let sel = "texture2DDescriptorWithPixelFormat:width:height:mipmapped:";
        let step = self.class_call(
            "MTLTextureDescriptor",
            sel,
            vec![
                Arg::U64(pixel_format),
                Arg::U64(width),
                Arg::U64(height),
                Arg::Bool(mipmapped),
            ],
        );
        step.throws(Policy::On(on), sel)?;
        let levels = if mipmapped {
            let m = width.max(height).max(1);
            u64::from(64u32.wrapping_sub(m.leading_zeros()))
        } else {
            1
        };
        Ok(self
            .world
            .new_obj(Model::TextureDescriptor(TextureDescriptorModel {
                desc: TexDesc {
                    pixel_format,
                    width,
                    height,
                    mipmap_level_count: levels,
                    ..TexDesc::default()
                },
                reported_rotation: None,
            })))
    }

    fn new_rate_map_layer(
        &self,
        on: OnFault,
        sample_count: MtlSize,
    ) -> RaiseResult<Option<Self::RateMapLayerDescriptor>> {
        let sel = "initWithSampleCount:";
        let step = self.class_call(
            "MTLRasterizationRateLayerDescriptor",
            sel,
            vec![Arg::val(&sample_count)],
        );
        step.create(Policy::On(on), sel, || {
            super::impls::rate_map::new_layer(&self.world, sample_count)
        })
    }

    fn new_address_range_array(
        &self,
        ranges: &[ResourceAddressRange],
    ) -> RaiseResult<Self::AddressRangeArray> {
        let sel = "initWithRanges:count:";
        let step = self.class_call(
            "MTLResourceAddressRangeArray",
            sel,
            vec![Arg::val(ranges), Arg::U64(ranges.len().to_u64())],
        );
        step.throws(Policy::Caught, sel)?;
        Ok(self.world.new_obj(Model::Ranges(ranges.to_vec())))
    }

    /// A fresh descriptor, then the value's setters ([`Fk::apply`]).
    fn texture_descriptor(&self, desc: &TextureDesc) -> Self::TextureDescriptor {
        let d: Self::TextureDescriptor = self.fresh();
        d.apply(desc);
        d
    }

    /// A fresh descriptor holding the value.
    fn rate_map_descriptor(&self, desc: &RateMapDesc) -> Self::RateMapDescriptor {
        let d: Self::RateMapDescriptor = self.fresh();
        d.apply(desc);
        d
    }

    /// The fake's device when `registry_id` is its registry ID.
    fn open_device(&self, registry_id: u64) -> Option<Self::Device> {
        (registry_id == self.registry_id).then(|| self.device.clone_ref())
    }

    fn surfaces(&self) -> &dyn IoSurfaces<Self> {
        &self.surfaces
    }

    fn vm(&self) -> &dyn HostVm {
        &self.world.vm
    }

    fn clock(&self) -> &Self::Clock {
        &self.world.clock
    }

    fn threads(&self) -> &dyn Threads {
        &self.threads
    }
}

#[cfg(test)]
mod tests {
    use super::super::script::Step;
    use super::*;
    use crate::platform::descriptor::TextureDescriptor;
    use crate::platform::device::GpuDevice;
    use crate::platform::rate_map::RateMapLayerDescriptor;
    use crate::platform::values::{RateMapDesc, TextureDesc};

    /// Class methods make fresh objects; only the listener's `new` is ordered, and logged
    #[test]
    fn class_methods() {
        let p = FakePlatform::new();
        let a = p.texture_descriptor(&TextureDesc::default());
        let b = p.texture_descriptor(&TextureDesc::default());
        assert_ne!(a.id(), b.id());
        let _ = p.rate_map_descriptor(&RateMapDesc::default());
        let _ = p.new_render_pass_descriptor();
        let _ = p.render_pass_descriptor();
        let m = p.call_log().mark();
        let _ = p.new_shared_event_listener();
        assert_eq!(p.call_log().sels_since(m), ["new"]);
        assert_eq!(p.call_log().count("renderPassDescriptor"), 0);
    }

    /// The 2D descriptor factory sets its fields and mip count
    #[test]
    fn texture2d_descriptor() {
        let p = FakePlatform::new();
        let d = p
            .texture_2d_descriptor(OnFault::Die, 80, 256, 64, true)
            .unwrap();
        assert_eq!(TextureDescriptor::pixel_format(&d), 80);
        assert_eq!(d.desc().mipmap_level_count, 9);
        let d = p
            .texture_2d_descriptor(OnFault::Die, 80, 1, 1, true)
            .unwrap();
        assert_eq!(d.desc().mipmap_level_count, 1);
        let d = p
            .texture_2d_descriptor(OnFault::Die, 80, 5, 3, false)
            .unwrap();
        assert_eq!((d.desc().width, d.desc().height), (5, 3));
    }

    /// The registry lookup finds only the fake's device
    #[test]
    fn open_device() {
        let p = FakePlatform::new();
        let d = p.open_device(FAKE_REGISTRY_ID).unwrap();
        assert_eq!(d.id(), p.device().id());
        assert_eq!(d.caps().registry_id, FAKE_REGISTRY_ID);
        assert!(p.open_device(1).is_none());
    }

    /// A rate-map layer is nil for a zero or unallocatable count, and when scripted
    #[test]
    fn rate_map_layer_nil() {
        let p = FakePlatform::new();
        assert!(
            p.new_rate_map_layer(OnFault::Die, MtlSize::new(0, 0, 0))
                .unwrap()
                .is_none()
        );
        assert!(
            p.new_rate_map_layer(OnFault::Die, MtlSize::new(3, 0, 0))
                .unwrap()
                .is_none()
        );
        assert!(
            p.new_rate_map_layer(OnFault::Die, MtlSize::new(0, 3, 0))
                .unwrap()
                .is_none()
        );
        assert!(
            p.new_rate_map_layer(OnFault::Die, MtlSize::new(u64::MAX, u64::MAX, 0))
                .unwrap()
                .is_none()
        );
        assert!(
            p.new_rate_map_layer(OnFault::Die, MtlSize::new(2, 2, 0))
                .unwrap()
                .is_some()
        );
        p.script().push("initWithSampleCount:", Step::Nil);
        assert!(
            p.new_rate_map_layer(OnFault::Die, MtlSize::new(2, 2, 0))
                .unwrap()
                .is_none()
        );
        assert!(
            p.new_rate_map_layer(OnFault::Die, MtlSize::new(2, 2, 0))
                .unwrap()
                .is_some()
        );
    }

    /// Layers and range arrays; trace switch; services are reachable
    #[test]
    fn misc() {
        let p = FakePlatform::new();
        let l = p
            .new_rate_map_layer(OnFault::Die, MtlSize::new(3, 2, 1))
            .unwrap()
            .unwrap();
        assert_eq!(l.sample_count(), MtlSize::new(3, 2, 1));
        assert!(!l.horizontal_sample_storage().is_null());
        let r = p
            .new_address_range_array(&[ResourceAddressRange {
                address: 1,
                length: 2,
            }])
            .unwrap();
        assert!(matches!(&r.state().model, Model::Ranges(v) if v.len() == 1));
        assert_eq!(p.vm().page_size(), crate::testkit::guest_ram::PAGE_SIZE);
        let _ = p.clock();
        let _ = p.threads();
    }
}
