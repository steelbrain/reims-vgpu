//! Fake: the platform trait impls for every `Fk<kind::…>`, one file per `platform/` file, and
//! the helpers they share (architecture 13.2).
//!
//! A method whose position the guest or host observes logs one
//! [`Call`](super::calllog::Call) with its selector and arguments, asks the script for a step,
//! and otherwise runs the object's model. A free method (a getter, a query, a setter no other
//! property aliases) reads or writes the model and nothing else; a test changes what it answers
//! through the model's setters. The helpers below give each ordered return shape its outcome
//! channels (4.2), each method passing its send's [`Policy`], the label its Metal implementation
//! gives the send, which decides a scripted raise there:
//! - [`Fk::act`] for `RaiseResult<()>` actions and [`Fk::thr`] for `RaiseResult<T>` queries;
//! - [`Fk::get_logged`] for a getter and [`Fk::set_logged`] for a setter that never raise (a
//!   scripted raise there dies at the send, as `send_get` does);
//! - [`Fk::make`] and [`Fk::make_or_error`] for creators (nil, `NSError`, raise, crash).

pub mod blit;
pub mod command;
pub mod compute;
pub mod descriptor;
pub mod device;
pub mod event;
pub mod icb;
pub mod rate_map;
pub mod render;
pub mod resource;

use std::sync::Arc;

use super::calllog::Arg;
use super::guest_ram::RawRegion;
use super::objects::{
    AnyHandle, BufferModel, FAKE_MAX_BYTES, Fk, Kind, Model, ResourceCommon, Space, TexDesc,
    TextureModel, World, kind,
};
use super::script::Policy;
use crate::host::HostPtr;
use crate::platform::Handle;
use crate::platform::blit::ResourceRef;
use crate::platform::types::NsErrorText;
use crate::raise::RaiseResult;

/// The identity of a resource argument.
pub fn res_id(r: &ResourceRef<'_, super::platform::FakePlatform>) -> crate::platform::ObjId {
    match r {
        ResourceRef::Buffer(b) => b.id(),
        ResourceRef::Texture(t) => t.id(),
        ResourceRef::Icb(i) => i.id(),
    }
}

/// A resource-array argument, by identity.
pub fn res_ids(rs: &[Option<ResourceRef<'_, super::platform::FakePlatform>>]) -> Arg {
    Arg::Objs(rs.iter().map(|r| r.as_ref().map(res_id)).collect())
}

/// The answer of a send labelled [`Policy::Fatal`]: its raise died at the send, so it is never
/// `Err`.
fn fatal<T>(r: RaiseResult<T>) -> T {
    r.unwrap_or_else(|r| r.fatal())
}

impl<K: Kind> Fk<K> {
    /// An action that may raise (`RaiseResult<()>`).
    pub fn act(&self, policy: Policy, sel: &'static str, args: Vec<Arg>) -> RaiseResult<()> {
        self.call(sel, args).throws(policy, sel)
    }

    /// A query that may raise (`RaiseResult<T>`): the scripted value, else `model()`.
    pub fn thr<T: Clone + 'static>(
        &self,
        policy: Policy,
        sel: &'static str,
        args: Vec<Arg>,
        model: impl FnOnce() -> T,
    ) -> RaiseResult<T> {
        self.call(sel, args).throws_value(policy, sel, model)
    }

    /// A getter whose position is observable (a Metal C function core sends with a guest
    /// value): logged, then the scripted value, else `model()`.
    pub fn get_logged<T: Clone + 'static>(
        &self,
        sel: &'static str,
        args: Vec<Arg>,
        model: impl FnOnce() -> T,
    ) -> T {
        self.call(sel, args).getter_value(sel, model)
    }

    /// A setter whose position is observable (it aliases another property, or Metal checks its
    /// guest value): logged, with the script's death and raise outcomes. The caller then
    /// writes the model.
    pub fn set_logged(&self, sel: &'static str, args: Vec<Arg>) {
        self.call(sel, args).getter(sel);
    }

    /// The device this object belongs to: itself for a device.
    pub fn owning_device(&self) -> Option<AnyHandle> {
        if K::CLASS == <kind::Device as Kind>::CLASS {
            Some(self.erase())
        } else {
            self.state().device.as_ref().map(AnyHandle::clone_ref)
        }
    }

    fn adopt<C: Kind>(&self, c: Fk<C>) -> Fk<C> {
        let d = self.owning_device();
        c.state().device = d;
        c
    }

    /// A creator: nil, a raise or a crash when scripted, else `init(world)` (which may itself
    /// return nil). The new object belongs to this object's device.
    pub fn make<C: Kind>(
        &self,
        policy: Policy,
        sel: &'static str,
        args: Vec<Arg>,
        init: impl FnOnce(&Arc<World>) -> Option<Fk<C>>,
    ) -> RaiseResult<Option<Fk<C>>> {
        let step = self.call(sel, args);
        let w = self.world().clone();
        step.create(policy, sel, || init(&w).map(|c| self.adopt(c)))
    }

    /// A creator with an `NSError **`.
    pub fn make_or_error<C: Kind>(
        &self,
        policy: Policy,
        sel: &'static str,
        args: Vec<Arg>,
        init: impl FnOnce(&Arc<World>) -> Option<Fk<C>>,
    ) -> RaiseResult<Result<Option<Fk<C>>, NsErrorText>> {
        let step = self.call(sel, args);
        let w = self.world().clone();
        step.create_or_error(policy, sel, || init(&w).map(|c| self.adopt(c)))
    }

    /// An indexed child getter whose position is observable (an ICB's command at an index): the
    /// same child for each index.
    pub fn index_child<C: Kind>(
        &self,
        policy: Policy,
        sel: &'static str,
        key: &'static str,
        index: u64,
    ) -> RaiseResult<Option<Fk<C>>> {
        let step = self.call(sel, vec![Arg::U64(index)]);
        step.create(policy, sel, || Some(self.child(key, index, || Model::None)))
    }

    /// Runs `f` on the common resource state, if the kind has one.
    pub fn with_common<T>(&self, f: impl FnOnce(&mut ResourceCommon) -> T) -> Option<T> {
        self.state().common().map(f)
    }

    /// `resourceIndex`, 0 when none was assigned.
    pub fn resource_index_value(&self) -> u64 {
        self.state()
            .common_ref()
            .and_then(|c| c.index.as_ref().map(super::objects::IndexGuard::index))
            .unwrap_or(0)
    }

    /// `uniqueIdentifier`.
    pub fn unique_id_value(&self) -> u64 {
        self.state().common_ref().map_or(0, |c| c.unique_id)
    }

    /// `gpuResourceID`.
    pub fn gpu_resource_id_value(&self) -> u64 {
        self.state().common_ref().map_or(0, |c| c.gpu_resource_id)
    }

    /// Makes `gpuResourceID` answer `id` (a kind with no common state keeps answering 0).
    pub fn set_gpu_resource_id(&self, id: u64) {
        let _ = self.with_common(|c| c.gpu_resource_id = id);
    }

    /// `setPurgeableState:`: the previous state; `KeepCurrent` changes nothing.
    pub fn purgeable_swap(&self, new: u64) -> u64 {
        self.with_common(|c| {
            let prev = c.purgeable;
            if new != super::objects::PURGEABLE_KEEP_CURRENT {
                c.purgeable = new;
            }
            prev
        })
        .unwrap_or(0)
    }
}

/// A fresh common state: non-volatile, fresh identifiers, and an index in `space` when given.
pub fn common(w: &Arc<World>, index: Option<(Space, Option<u64>)>) -> ResourceCommon {
    ResourceCommon {
        purgeable: super::objects::PURGEABLE_NON_VOLATILE,
        index: index.map(|(s, forced)| w.alloc_index(s, forced)),
        unique_id: w.next_unique(),
        gpu_resource_id: w.next_unique(),
        label: None,
    }
}

/// Where a new buffer's bytes come from.
pub enum Backing {
    /// Fresh memory the fake owns (none for private storage).
    Fresh,
    /// The caller's memory (no-copy buffers, descriptor contents, IOSurfaces, heap placement).
    At(HostPtr),
}

/// A new buffer of `length` bytes with `options`. `None` when the fake cannot back it
/// ([`FAKE_MAX_BYTES`]).
pub fn new_buffer(
    w: &Arc<World>,
    length: u64,
    options: u64,
    backing: Backing,
    gpu_address: Option<u64>,
) -> Option<Fk<kind::Buffer>> {
    let storage = options.wrapping_shr(4) & 0xf;
    let (mem, contents) = match backing {
        Backing::At(p) => (None, p),
        Backing::Fresh if storage >= 2 => (None, HostPtr::NULL),
        Backing::Fresh => {
            if length > FAKE_MAX_BYTES {
                return None;
            }
            let r = RawRegion::new(length)?;
            let a = r.addr();
            (Some(r), a)
        }
    };
    Some(w.new_obj(Model::Buffer(BufferModel {
        mem,
        contents,
        length,
        options,
        gpu_address: gpu_address.unwrap_or_else(|| w.alloc_gpu_va(length)),
        common: common(w, None),
        ..BufferModel::default()
    })))
}

/// A new texture over `desc` (its forced resource index honoured).
pub fn new_texture(
    w: &Arc<World>,
    desc: TexDesc,
    surface: Option<Fk<kind::Surface>>,
    plane: u64,
    parent: Option<AnyHandle>,
    shareable: bool,
) -> Fk<kind::Texture> {
    let forced = if desc.force_resource_index {
        desc.resource_index
    } else {
        None
    };
    w.new_obj(Model::Texture(TextureModel {
        desc,
        surface,
        plane,
        parent,
        shareable,
        common: common(w, Some((Space::TEXTURE, forced))),
    }))
}

/// A new sampler or depth-stencil state with an index in `space` (forced when given).
pub fn new_state<C: Kind>(w: &Arc<World>, index: Option<(Space, Option<u64>)>) -> Fk<C> {
    w.new_obj(Model::State(common(w, index)))
}

#[cfg(test)]
mod tests {
    use super::super::objects::{CompleteOn, GPU_VA_BASE};
    use super::super::platform::FakePlatform;
    use super::super::script::Step;
    use super::*;
    use crate::platform::Platform;
    use crate::platform::command::{CommandBuffer, CommandQueue};
    use crate::platform::device::GpuDevice;
    use crate::platform::event::SharedEvent;
    use crate::platform::resource::{Buffer, GpuResource, Texture};
    use crate::platform::types::{CommandBufferStatus, MtlSize};
    use crate::platform::values::{IcbDesc, RateMapDesc, TextureDesc};
    use crate::raise::OnFault;

    /// Buffers get a GPU VA, contents unless private, and refuse past the fake's limit
    #[test]
    fn buffers() {
        let p = FakePlatform::new();
        let d = p.device();
        let b = d
            .new_buffer_of_length(OnFault::Die, 64, 0)
            .unwrap()
            .unwrap();
        assert_eq!(b.length(), 64);
        assert!(!b.contents().is_null());
        assert!(b.gpu_address() >= GPU_VA_BASE);
        let private = d
            .new_buffer_of_length(OnFault::Die, 64, 2 << 4)
            .unwrap()
            .unwrap();
        assert!(private.contents().is_null());
        assert_eq!(GpuResource::storage_mode(&private), 2);
        assert!(
            d.new_buffer_of_length(OnFault::Die, FAKE_MAX_BYTES + 1, 0)
                .unwrap()
                .is_none()
        );
        p.script().push("newBufferWithLength:options:", Step::Nil);
        assert!(
            d.new_buffer_of_length(OnFault::Die, 64, 0)
                .unwrap()
                .is_none()
        );
        assert_eq!(p.call_log().count("newBufferWithLength:options:"), 4);
    }

    /// Textures follow their descriptor; a forced index is honoured
    #[test]
    fn textures() {
        let p = FakePlatform::new();
        let d = p.device();
        let td = p.texture_descriptor(&TextureDesc {
            width: 8,
            height: 4,
            pixel_format: 80,
            resource_index: Some((7, true)),
            ..TextureDesc::default()
        });
        let t = d.new_texture(OnFault::Die, &td).unwrap().unwrap();
        assert_eq!((Texture::width(&t), Texture::height(&t)), (8, 4));
        assert_eq!(Texture::pixel_format(&t), 80);
        assert_eq!(t.resource_index(), 7);
        let v = t.new_texture_view(81).unwrap().unwrap();
        assert_eq!(Texture::pixel_format(&v), 81);
        assert_ne!(t.unique_identifier(), v.unique_identifier());
    }

    /// Commit, completed handlers and auto-complete
    #[test]
    fn command_buffers() {
        let p = FakePlatform::new();
        let q = p.device().new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let seen = Arc::new(crate::sync::Mutex::new(Vec::new()));
        let s = seen.clone();
        cb.add_completed_handler(OnFault::Die, Box::new(move |i| s.lock().push(i.status)))
            .unwrap();
        cb.commit(OnFault::Die).unwrap();
        assert_eq!(cb.status(), CommandBufferStatus::SCHEDULED);
        assert_eq!(p.world().complete_all(CompleteOn::Caller), 1);
        assert_eq!(*seen.lock(), vec![CommandBufferStatus::COMPLETED]);
        p.world().set_auto_complete(Some(CompleteOn::Caller));
        let cb2 = q.command_buffer(OnFault::Die).unwrap().unwrap();
        cb2.commit(OnFault::Die).unwrap();
        assert_eq!(cb2.status(), CommandBufferStatus::COMPLETED);
    }

    /// A device answers the fake's default caps until a test sets its own
    #[test]
    fn device_caps() {
        use crate::platform::caps::DeviceCaps;
        let p = FakePlatform::new();
        let d = p.device();
        let read = d.caps();
        assert_eq!(*read, device::fake_caps());
        assert_eq!(read.limits.max_viewport_count, 16);
        assert_eq!(read.registry_id, super::super::platform::FAKE_REGISTRY_ID);
        let mut caps = DeviceCaps::clone(&read);
        caps.limits.max_viewport_count = 3;
        caps.flags.supports_tile_shaders = false;
        caps.families[0] = Err(crate::raise::Raise::foreign("supportsFamily:"));
        caps.gpu_core_count = None;
        d.set_caps(caps.clone());
        assert_eq!(*d.caps(), caps);
    }

    /// The model setters change what the free getters answer, and log nothing
    #[test]
    fn model_setters() {
        use crate::platform::descriptor::TextureDescriptor;
        use crate::platform::icb::{IndirectCommandBuffer, IndirectRenderCommand};
        use crate::platform::rate_map::{RateMap, RateMapDescriptor, RateMapLayerDescriptor};
        use crate::platform::resource::ComputePipeline;
        use crate::platform::types::SizeAndAlign;
        let p = FakePlatform::new();
        let d = p.device();
        let td = p.texture_descriptor(&TextureDesc {
            rotation: Some(1),
            ..TextureDesc::default()
        });
        td.set_reported_rotation(3);
        assert_eq!(td.rotation(), 3);
        let t = d.new_texture(OnFault::Die, &td).unwrap().unwrap();
        let quiet = p.call_log().mark();
        t.set_gpu_resource_id(0x77);
        t.set_protection_options(2);
        t.set_storage_mode(3);
        assert!(p.call_log().since(quiet).is_empty());
        assert_eq!(t.gpu_resource_id(), 0x77);
        assert_eq!(Texture::protection_options(&t), 2);
        assert_eq!(GpuResource::storage_mode(&t), 3);
        let rd = p.rate_map_descriptor(&RateMapDesc {
            screen_size: MtlSize::new(4, 4, 0),
            ..RateMapDesc::default()
        });
        rd.set_layer_count(3);
        assert_eq!(RateMapDescriptor::layer_count(&rd), 3);
        let layer = p
            .new_rate_map_layer(OnFault::Die, MtlSize::new(2, 2, 0))
            .unwrap()
            .unwrap();
        let quiet = p.call_log().mark();
        layer.set_sample_count(MtlSize::new(1, 1, 0));
        layer.set_horizontal_sample_storage(crate::host::HostPtr::NULL);
        layer.set_vertical_sample_storage(crate::host::HostPtr::from_addr(0x40));
        assert!(p.call_log().since(quiet).is_empty());
        assert_eq!(layer.sample_count(), MtlSize::new(1, 1, 0));
        assert!(layer.horizontal_sample_storage().is_null());
        assert_eq!(layer.vertical_sample_storage().addr(), 0x40);
        let map = d.new_rate_map(&rd).unwrap().unwrap();
        let quiet = p.call_log().mark();
        map.set_layer_count(2);
        map.set_mutability(1);
        let buffer = SizeAndAlign { size: 8, align: 4 };
        map.set_parameter_buffer_size_and_align(buffer);
        assert!(p.call_log().since(quiet).is_empty());
        assert_eq!(RateMap::layer_count(&map), 2);
        assert_eq!(map.mutability(), 1);
        assert_eq!(map.parameter_buffer_size_and_align(), buffer);
        let pipeline = d
            .new_compute_pipeline_state_for_function(None)
            .unwrap()
            .unwrap();
        pipeline.set_static_threadgroup_memory_length(0x20);
        assert_eq!(pipeline.static_threadgroup_memory_length(), 0x20);
        let icb = d
            .new_indirect_command_buffer(&IcbDesc::default(), 4, 0)
            .unwrap()
            .unwrap();
        icb.set_size(9);
        assert_eq!(icb.size(), 9);
        let cmd = icb.render_command(0).unwrap().unwrap();
        cmd.set_command_type(0x10);
        assert_eq!(cmd.command_type(), 0x10);
        assert_eq!(cmd.draw_args(), None);
        cmd.encode(crate::platform::icb::IcbRenderCmd::DrawPrimitives {
            primitive_type: 3,
            vertex_start: 1,
            vertex_count: 6,
            instance_count: 2,
            base_instance: 0,
        })
        .unwrap();
        assert_eq!(
            cmd.draw_args(),
            Some(crate::platform::icb::DrawArgs {
                primitive_type: 3,
                vertex_start: 1,
                vertex_count: 6,
                instance_count: 2,
                base_instance: 0,
            })
        );
        // A reset forgets the draw and its type.
        cmd.encode(crate::platform::icb::IcbRenderCmd::Reset)
            .unwrap();
        assert_eq!((cmd.draw_args(), cmd.command_type()), (None, 0));
        let compute = icb.compute_command(1).unwrap().unwrap();
        compute.set_command_type(0x20);
        assert_eq!(
            crate::platform::icb::IndirectComputeCommand::command_type(&compute),
            0x20
        );
    }

    /// Shared events signal and wait
    #[test]
    fn shared_events() {
        let p = FakePlatform::new();
        let e = p.device().new_shared_event().unwrap();
        e.set_signaled_value(5).unwrap();
        assert!(e.wait_until_signaled_value(5, 0));
    }
}
