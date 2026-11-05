//! The deserializer's object lookup, and the manager's reads of task memory.
//!
//! [`ManagerLookup`] holds its manager unretained. Each typed getter is the manager's lookup;
//! on a lookup error it logs the error and raises
//! [`RaiseKind::Lookup`]. The untyped resource lookup answers nil
//! instead, and the writable cursor lookup null.
//!
//! [`ManagerLookup::get_resource_raising_empty`] is the untyped lookup for the one list whose nil
//! entry Metal ends the process at, a concurrent compute encoder's `useResources:count:usage:`
//! (`interplay_decodewalk_26`): a reference whose entry is empty (type 0) raises there, as the
//! typed getters do. A guest empties a task's entries when the process that owns it ends, without
//! waiting for the execs that process queued, so an exec the host runs late names empty entries
//! (`interplay_lifetime_emptied_resource`; the function resolver's exception, `resolver.rs`, is
//! the same). Every other list keeps the nil, which Metal skips.
//!
//! Every getter is nullable: `Ok(None)` is nil (reference 0, or a Metal nil nothing checks),
//! which flows on. The deserializer's `require_…` variants are its own nil checks over these
//! getters. A lookup can create the object (lookup is creation), so the decoders call exactly
//! the getters they need, in their order. Handles are returned retained. A lookup that reaches
//! the kernel task (a backing, a shared texture's handle) holds it to the end of its hold's
//! holder ([`Hold::keep`]), so its release, which the host sees, comes after the work.

use std::sync::{Arc, Weak};

use crate::deserializer::texture_descriptor::{
    SerializedTextureDescriptor, SerializedTextureDescriptor2,
};
use crate::guest::records::{APV_PAGE_SIZE, ObjectEntry};
use crate::memory::cursor::{Cursor, WritableCursor};
use crate::platform::Platform;
use crate::raise::RaiseResult;
use crate::raise::{Raise, RaiseId};
use crate::resources::manager::records::{ObjectBuffer, ObjectCpuBuffer};
use crate::resources::manager::states::{
    ComputePipelineStates, DepthStencilStates, Fences, Functions, RasterizationRateMaps,
    RenderPipelineStates, SamplerStates,
};
use crate::resources::manager::{DeathKind, ManagedKind, ObjectKind, RaiseKind, ResourceManager};
use crate::resources::resource::AnyResource;
use crate::rt::{Held, Hold};

/// The deserializer's lookups over one manager, which it holds unretained.
pub struct ManagerLookup<P: Platform> {
    manager: Weak<ResourceManager<P>>,
}

impl<P: Platform> ManagerLookup<P> {
    /// A lookup over `manager`, unretained.
    pub fn new(manager: Weak<ResourceManager<P>>) -> Arc<Self> {
        Arc::new(ManagerLookup { manager })
    }

    /// The manager, if it is still there.
    pub fn manager(&self) -> Option<Arc<ResourceManager<P>>> {
        self.manager.upgrade()
    }

    /// The manager; its use once gone is a death (the task releases the deserializer, which holds
    /// the lookup, before its manager, so nothing gets here).
    fn live(&self) -> Arc<ResourceManager<P>> {
        match self.manager.upgrade() {
            Some(m) => m,
            None => crate::fatal!(DeathKind::ManagerGone),
        }
    }

    /// A lookup through `K`'s cache; the manager is let go before a failure is answered.
    fn cached<K: ManagedKind<P>>(
        &self,
        hold: Hold<'_>,
        kind: ObjectKind,
        reference: u32,
    ) -> RaiseResult<Option<K::Obj>> {
        let result = self.live().get_object::<K>(hold, reference)?;
        self.answer(kind, result)
    }

    /// A view of a resource; the manager is held until the failure is answered.
    fn view<T>(
        &self,
        kind: ObjectKind,
        get: impl FnOnce(&ResourceManager<P>) -> RaiseResult<Result<Option<T>, String>>,
    ) -> RaiseResult<Option<T>> {
        let m = self.live();
        let result = get(&m)?;
        self.answer(kind, result)
    }

    /// The object, or the failure logged and raised.
    fn answer<T>(
        &self,
        kind: ObjectKind,
        result: Result<Option<T>, String>,
    ) -> RaiseResult<Option<T>> {
        match result {
            Ok(object) => Ok(object),
            Err(error) => {
                // The manager is asked for (a death if it is gone) and held until the raise.
                let _manager = self.live();
                log::warn!("{} lookup failed: {error}", kind.log_name());
                Err(Raise::new(RaiseKind::Lookup(kind)).with_detail(kind.failure()))
            }
        }
    }

    /// The unit tests' view of a lookup: one `ManagerLookup` row in the fake platform's call log
    /// (method, first argument), before the lookup runs.
    #[cfg(test)]
    fn trace(&self, sel: &'static str, arg: crate::testkit::calllog::Arg) {
        let Some(m) = self.manager.upgrade() else {
            return;
        };
        let any: &dyn std::any::Any = m.platform.as_ref();
        if let Some(p) = any.downcast_ref::<crate::testkit::FakePlatform>() {
            p.call_log()
                .push(crate::platform::ObjId(0), "ManagerLookup", sel, vec![arg]);
        }
    }
}

/// The deserializer's typed getters. Each is traced in the unit tests' call log.
impl<P: Platform> ManagerLookup<P> {
    /// The function `reference` names.
    pub fn get_function(&self, hold: Hold<'_>, reference: u32) -> RaiseResult<Option<P::Function>> {
        #[cfg(test)]
        self.trace("get_function", crate::testkit::calllog::Arg::U32(reference));
        self.cached::<Functions>(hold, ObjectKind::Function, reference)
    }

    /// The compute pipeline state `reference` names.
    pub fn get_compute_pipeline_state(
        &self,
        hold: Hold<'_>,
        reference: u32,
    ) -> RaiseResult<Option<P::ComputePipeline>> {
        #[cfg(test)]
        self.trace(
            "get_compute_pipeline_state",
            crate::testkit::calllog::Arg::U32(reference),
        );
        self.cached::<ComputePipelineStates>(hold, ObjectKind::ComputePipeline, reference)
    }

    /// The render pipeline state `reference` names.
    pub fn get_render_pipeline_state(
        &self,
        hold: Hold<'_>,
        reference: u32,
    ) -> RaiseResult<Option<P::RenderPipeline>> {
        #[cfg(test)]
        self.trace(
            "get_render_pipeline_state",
            crate::testkit::calllog::Arg::U32(reference),
        );
        self.cached::<RenderPipelineStates>(hold, ObjectKind::RenderPipeline, reference)
    }

    /// The buffer of the resource `reference` names.
    pub fn get_buffer(&self, hold: Hold<'_>, reference: u32) -> RaiseResult<Option<P::Buffer>> {
        #[cfg(test)]
        self.trace("get_buffer", crate::testkit::calllog::Arg::U32(reference));
        // A resource with no buffer raises its own unrecognized operation, before any log
        self.view(ObjectKind::Buffer, |m| m.get_object_buffer(hold, reference))
    }

    /// The texture of the resource `reference` names.
    pub fn get_texture(&self, hold: Hold<'_>, reference: u32) -> RaiseResult<Option<P::Texture>> {
        #[cfg(test)]
        self.trace("get_texture", crate::testkit::calllog::Arg::U32(reference));
        self.view(ObjectKind::Texture, |m| {
            m.get_object_texture(hold, reference)
        })
    }

    /// The sampler state `reference` names.
    pub fn get_sampler_state(
        &self,
        hold: Hold<'_>,
        reference: u32,
    ) -> RaiseResult<Option<P::SamplerState>> {
        #[cfg(test)]
        self.trace(
            "get_sampler_state",
            crate::testkit::calllog::Arg::U32(reference),
        );
        self.cached::<SamplerStates>(hold, ObjectKind::Sampler, reference)
    }

    /// The depth-stencil state `reference` names.
    pub fn get_depth_stencil_state(
        &self,
        hold: Hold<'_>,
        reference: u32,
    ) -> RaiseResult<Option<P::DepthStencilState>> {
        #[cfg(test)]
        self.trace(
            "get_depth_stencil_state",
            crate::testkit::calllog::Arg::U32(reference),
        );
        self.cached::<DepthStencilStates>(hold, ObjectKind::DepthStencil, reference)
    }

    /// The fence `reference` names.
    pub fn get_fence(&self, hold: Hold<'_>, reference: u32) -> RaiseResult<Option<P::Fence>> {
        #[cfg(test)]
        self.trace("get_fence", crate::testkit::calllog::Arg::U32(reference));
        self.cached::<Fences>(hold, ObjectKind::Fence, reference)
    }

    /// The heap of the resource `reference` names.
    pub fn get_heap(&self, hold: Hold<'_>, reference: u32) -> RaiseResult<Option<P::Heap>> {
        #[cfg(test)]
        self.trace("get_heap", crate::testkit::calllog::Arg::U32(reference));
        self.view(ObjectKind::Heap, |m| m.get_object_heap(hold, reference))
    }

    /// The rasterization rate map `reference` names.
    pub fn get_rasterization_rate_map(
        &self,
        hold: Hold<'_>,
        reference: u32,
    ) -> RaiseResult<Option<P::RateMap>> {
        #[cfg(test)]
        self.trace(
            "get_rasterization_rate_map",
            crate::testkit::calllog::Arg::U32(reference),
        );
        self.cached::<RasterizationRateMaps>(hold, ObjectKind::RateMap, reference)
    }

    /// The indirect command buffer of the resource `reference` names.
    pub fn get_indirect_command_buffer(
        &self,
        hold: Hold<'_>,
        reference: u32,
    ) -> RaiseResult<Option<P::Icb>> {
        #[cfg(test)]
        self.trace(
            "get_indirect_command_buffer",
            crate::testkit::calllog::Arg::U32(reference),
        );
        self.view(ObjectKind::Icb, |m| m.get_object_icb(hold, reference))
    }

    /// Whichever Metal resource the resource `reference` names; an error is nil, with no log
    /// and no raise.
    pub fn get_resource(
        &self,
        hold: Hold<'_>,
        reference: u32,
    ) -> RaiseResult<Option<AnyResource<P>>> {
        #[cfg(test)]
        self.trace("get_resource", crate::testkit::calllog::Arg::U32(reference));
        // An error is nil: no log, no raise
        Ok(self
            .live()
            .get_object_resource(hold, reference)?
            .unwrap_or_default())
    }

    /// [`Self::get_resource`], except that a reference whose entry is empty is logged and raised
    /// as a typed getter's failure is; any other error is still nil.
    pub fn get_resource_raising_empty(
        &self,
        hold: Hold<'_>,
        reference: u32,
    ) -> RaiseResult<Option<AnyResource<P>>> {
        #[cfg(test)]
        self.trace(
            "get_resource_raising_empty",
            crate::testkit::calllog::Arg::U32(reference),
        );
        let m = self.live();
        let result = m.get_object_resource(hold, reference)?;
        if result.is_err()
            && m.get_object_entry(reference)
                .is_ok_and(|entry| entry.object_type() == 0)
        {
            return self.answer(ObjectKind::Resource, result);
        }
        // Any other error is nil: no log, no raise
        Ok(result.unwrap_or_default())
    }

    /// A new texture descriptor for a serialized one.
    pub fn texture_descriptor_for_descriptor(
        &self,
        d: &SerializedTextureDescriptor,
    ) -> RaiseResult<Option<P::TextureDescriptor>> {
        #[cfg(test)]
        self.trace(
            "texture_descriptor_for_descriptor",
            crate::testkit::calllog::Arg::val(d),
        );
        self.live().texture_descriptor_for_descriptor(d)
    }

    /// A new texture descriptor for a serialized one (the second format).
    pub fn texture_descriptor_for_descriptor2(
        &self,
        d: &SerializedTextureDescriptor2,
    ) -> RaiseResult<Option<P::TextureDescriptor>> {
        #[cfg(test)]
        self.trace(
            "texture_descriptor_for_descriptor2",
            crate::testkit::calllog::Arg::val(d),
        );
        self.live().texture_descriptor_for_descriptor2(d)
    }

    /// A writable cursor over a buffer's guest memory (info replies); an error is null,
    /// with no log.
    pub fn new_memory_cursor_for_buffer(
        &self,
        hold: Hold<'_>,
        buffer_reference: u32,
    ) -> RaiseResult<Option<WritableCursor>> {
        #[cfg(test)]
        self.trace(
            "new_memory_cursor_for_buffer",
            crate::testkit::calllog::Arg::U32(buffer_reference),
        );
        // An error is null, with no log: the caller raises
        Ok(self
            .live()
            .new_writable_memory_cursor_for_buffer(hold, buffer_reference)?
            .ok())
    }
}

/// Why a copy of task memory ([`ResourceManager::get_data`]) gave no bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NoData {
    /// The copy's page-table walk met a hole: a page of the range that the task's page table
    /// does not map.
    pub hole: bool,
    /// What refused the copy, for the log: the raise's detail, or that the task is gone or gave
    /// nothing.
    pub why: String,
}

impl NoData {
    /// What `raised`, the copy's raise, says.
    fn raised(raised: &Raise) -> NoData {
        let why = match raised.exception() {
            "" => raised.id().to_string(),
            detail => detail.to_owned(),
        };
        NoData {
            hole: raised.id() == RaiseId::Memory(crate::memory::RaiseKind::WalkHole),
            why,
        }
    }
}

impl std::fmt::Display for NoData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.why)
    }
}

impl<P: Platform> ResourceManager<P> {
    /// The object list's entry at `index`.
    pub fn get_object_entry(&self, index: u32) -> Result<ObjectEntry, String> {
        self.object_lists.get_object_entry(index)
    }

    /// A copy of `length` bytes of task memory at `offset`; a raise, a gone task or no data is
    /// [`NoData`], which says which.
    ///
    /// The read is a landing frame of its own inside the handler ([`Held::raising`]): what its
    /// guest reads hold (the page-table read's device handle) is released when it returns, and
    /// on a raise goes on with the raise to the enclosing holder, where the handler drops it.
    pub fn get_data(&self, hold: Hold<'_>, offset: u64, length: u64) -> Result<Vec<u8>, NoData> {
        let copied = Held::raising(|held| match self.task.upgrade() {
            Some(task) => task
                .data_from_virtual_offset(Hold::of(held), offset, length)
                .map(|data| data.ok_or("the task gave no data")),
            None => Ok(Err("the task is gone")),
        });
        match copied {
            Ok(Ok(data)) => Ok(data),
            Ok(Err(why)) => Err(NoData {
                hole: false,
                why: why.to_owned(),
            }),
            Err(raised) => {
                let missing = NoData::raised(&raised);
                hold.land(raised);
                Err(missing)
            }
        }
    }

    /// A cursor over `length` bytes of task memory at `offset`; `None` when the task gives none
    /// or is gone.
    pub fn new_memory_cursor(
        &self,
        hold: Hold<'_>,
        offset: u64,
        length: u64,
    ) -> RaiseResult<Option<Cursor>> {
        match self.task.upgrade() {
            Some(task) => task.cursor_from_virtual_offset(hold, offset, length),
            None => Ok(None),
        }
    }

    /// A writable cursor over the guest pages of the buffer `buffer_reference` names, under the
    /// creation lock. Every refusal is `Err`, a null cursor from the task (or a gone task)
    /// included.
    pub fn new_writable_memory_cursor_for_buffer(
        &self,
        hold: Hold<'_>,
        buffer_reference: u32,
    ) -> RaiseResult<Result<WritableCursor, String>> {
        let _creation = self.creation.lock();
        // Reference 0 reads entry 0
        let entry = match self.get_object_entry(buffer_reference) {
            Ok(entry) => entry,
            Err(e) => return Ok(Err(e)),
        };
        let offset = entry.offset;
        let Some(mut cursor) = self.new_memory_cursor(hold, offset, u64::from(entry.length()))?
        else {
            return Ok(Err(format!(
                "no cursor over object {buffer_reference}'s record"
            )));
        };
        let (page, length) = match entry.object_type() {
            1 => match cursor.read::<ObjectBuffer>() {
                Ok(d) => (d.page, d.length),
                Err(e) => return Ok(Err(format!("buffer record not readable: {e}"))),
            },
            19 => match cursor.read::<ObjectCpuBuffer>() {
                Ok(d) => (d.page, d.length),
                Err(e) => return Ok(Err(format!("buffer record not readable: {e}"))),
            },
            0 => {
                return Ok(Err(format!("object {buffer_reference} has an empty entry")));
            }
            other => {
                return Ok(Err(format!(
                    "object of type {} is not a buffer",
                    u32::from(other)
                )));
            }
        };
        // The flags are ignored: a private buffer's reply goes to its guest pages
        let virtual_offset = u64::from(page).wrapping_mul(APV_PAGE_SIZE);
        let cursor = match self.task.upgrade() {
            Some(task) => task.writable_cursor_from_virtual_offset(hold, virtual_offset, length)?,
            None => None,
        };
        Ok(cursor.ok_or_else(|| "no writable cursor over the buffer's guest pages".to_owned()))
    }
}

#[cfg(test)]
mod tests {
    use super::super::fixture::*;
    use super::*;
    use crate::platform::Handle;
    use crate::raise::RaiseId;
    use crate::rt::Pool;
    use crate::testkit::calllog::Arg;
    use crate::testkit::capture::catch_death;
    use crate::testkit::{Step, kind};
    use zerocopy::IntoBytes;

    fn fx() -> Fx {
        Fx::new(&[(0, 0, 0), (6, 0x10, REC)], Placement::Mapped)
    }

    fn entries3() -> Vec<(u8, u32, u64)> {
        vec![(0, 0, 0), (6, 0x20, REC), (6, 0x10, REC + 0x100)]
    }

    /// Every getter's error path: under a handler it logs the kind's name and the error, then
    /// raises its kind's lookup failure; reference 0 is nil for each, with no log.
    macro_rules! getter_case {
        ($name:ident, $method:ident, $kind:expr, $literal:literal) => {
            #[test]
            fn $name() {
                let fx = fx();
                let d = fx.m.object_delegate();
                let (nil, lines) = fx.logged(|| d.$method(Hold::NONE, 0).unwrap().is_none());
                assert!(nil);
                assert!(lines.is_empty());
                {
                    let hold = Hold::NONE;
                    // Reference 9 is out of range: the lookup's error, logged then raised.
                    let (r, lines) = fx.logged(|| d.$method(hold, 9));
                    let r = r.err().unwrap();
                    assert_eq!(r.id(), RaiseId::Manager(RaiseKind::Lookup($kind)));
                    assert_eq!(r.detail(), Some($kind.failure()));
                    assert_eq!(
                        lines,
                        vec![format!(
                            "{} lookup failed: reference 9 is past the object list's 2 entries",
                            $literal
                        )]
                    );
                }
            }
        };
    }

    mod getters {
        use super::*;

        getter_case!(function, get_function, ObjectKind::Function, "function");
        getter_case!(
            compute,
            get_compute_pipeline_state,
            ObjectKind::ComputePipeline,
            "compute pipeline"
        );
        getter_case!(
            render,
            get_render_pipeline_state,
            ObjectKind::RenderPipeline,
            "render pipeline"
        );
        getter_case!(buffer, get_buffer, ObjectKind::Buffer, "buffer");
        getter_case!(texture, get_texture, ObjectKind::Texture, "texture");
        getter_case!(
            sampler,
            get_sampler_state,
            ObjectKind::Sampler,
            "sampler state"
        );
        getter_case!(
            depth_stencil,
            get_depth_stencil_state,
            ObjectKind::DepthStencil,
            "depth-stencil state"
        );
        getter_case!(fence, get_fence, ObjectKind::Fence, "fence");
        getter_case!(heap, get_heap, ObjectKind::Heap, "heap");
        getter_case!(
            rate_map,
            get_rasterization_rate_map,
            ObjectKind::RateMap,
            "rasterization rate map"
        );
        getter_case!(
            icb,
            get_indirect_command_buffer,
            ObjectKind::Icb,
            "indirect command buffer"
        );

        #[test]
        fn a_raise_names_the_lookups_site() {
            let fx = fx();
            let d = fx.m.object_delegate();
            assert_eq!(
                d.get_function(Hold::NONE, 9)
                    .map(|_| ())
                    .map_err(|r| r.site()),
                Err(crate::raise::death::DeathSite {
                    kind: crate::raise::death::DeathKind::UnhandledRaise,
                    file: "manager.lookup.function",
                    line: 0
                })
            );
        }

        #[test]
        fn a_cached_object_is_returned() {
            let fx = fx();
            let s = fx
                .platform
                .world()
                .new_obj::<kind::SamplerState>(Default::default());
            assert!(fx.m.register_object::<SamplerStates>(1, s.clone_ref()));
            let got =
                fx.m.object_delegate()
                    .get_sampler_state(Hold::NONE, 1)
                    .unwrap()
                    .unwrap();
            assert_eq!(got.id(), s.id());
        }

        #[test]
        fn a_gone_manager_is_a_death() {
            let fx = fx();
            let d = fx.m.object_delegate();
            let Fx { m, .. } = fx;
            drop(m);
            assert_eq!(
                catch_death(move || {
                    let _ = d.get_fence(Hold::NONE, 1);
                }),
                Err(DeathKind::ManagerGone.site())
            );
        }

        #[test]
        fn holds_the_manager_unretained() {
            let fx = fx();
            let d = ManagerLookup::new(Arc::downgrade(&fx.m));
            assert!(Arc::ptr_eq(&d.manager().unwrap(), &fx.m));
            let Fx { m, .. } = fx;
            drop(m);
            assert!(d.manager().is_none());
        }

        #[test]
        fn a_resource_error_is_nil_without_a_raise() {
            let fx = fx();
            let d = fx.m.object_delegate();
            // No handler: nothing is raised, so nothing dies.
            assert!(d.get_resource(Hold::NONE, 9).unwrap().is_none());
            assert!(d.get_resource(Hold::NONE, 0).unwrap().is_none());
        }

        /// An entry the guest emptied (type 0 and offset all ones, as a task's end leaves each of
        /// its entries) is nil to the untyped lookup, and raises the resource lookup's failure,
        /// with the failure's line, to the one that raises for it; any other error is nil to both.
        #[test]
        fn a_resource_whose_entry_is_empty_raises_only_where_asked() {
            let fx = Fx::new(
                &[(0, 0, 0), (6, 0x10, REC), (0, 0, u64::MAX)],
                Placement::Off,
            );
            let d = fx.m.object_delegate();
            let (nil, lines) = fx.logged(|| d.get_resource(Hold::NONE, 2).unwrap().is_none());
            assert!(nil);
            assert!(lines.is_empty());
            for reference in [0, 9] {
                assert!(
                    d.get_resource_raising_empty(Hold::NONE, reference)
                        .unwrap()
                        .is_none()
                );
            }
            let (r, lines) = fx.logged(|| d.get_resource_raising_empty(Hold::NONE, 2));
            let r = r.err().unwrap();
            assert_eq!(
                r.id(),
                RaiseId::Manager(RaiseKind::Lookup(ObjectKind::Resource))
            );
            assert_eq!(r.detail(), Some("resource lookup failed"));
            assert_eq!(
                lines,
                vec!["resource lookup failed: object 2 has an empty entry"]
            );
        }
    }

    #[test]
    fn a_buffer_cursor_error_is_null_and_a_cursor_passes_through() {
        let fx = Fx::new(&[(7, 0x10, REC), (19, 0xc, REC)], Placement::Off);
        let d = fx.m.object_delegate();
        fx.next_cursor(0, 0x10);
        assert!(
            d.new_memory_cursor_for_buffer(Hold::NONE, 0)
                .unwrap()
                .is_none()
        );
        fx.next_cursor(0, 0xc);
        fx.platform.script().push(
            "writable_cursor_from_virtual_offset",
            Step::value(Some(WritableCursor::default())),
        );
        assert!(
            d.new_memory_cursor_for_buffer(Hold::NONE, 1)
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn texture_descriptors_forward_to_the_manager() {
        let fx = fx();
        let d = fx.m.object_delegate();
        let v1 = SerializedTextureDescriptor::default();
        let v2 = SerializedTextureDescriptor2::default();
        assert!(d.texture_descriptor_for_descriptor(&v1).unwrap().is_some());
        assert!(d.texture_descriptor_for_descriptor2(&v2).unwrap().is_some());
    }

    mod entry_and_cursor {
        use super::*;

        #[test]
        fn get_object_entry_forwards() {
            let fx = Fx::new(&entries3(), Placement::Off);
            let e = fx.m.get_object_entry(2).unwrap();
            assert_eq!(e.object_type(), 6);
            assert_eq!(e.length(), 0x10);
            assert_eq!({ e.offset }, REC + 0x100);
            assert_eq!(
                fx.m.get_object_entry(3).err().unwrap(),
                "reference 3 is past the object list's 3 entries"
            );
        }

        #[test]
        fn new_memory_cursor_asks_the_task_or_is_null() {
            let fx = Fx::new(&entries3(), Placement::Off);
            fx.next_cursor(0, 4);
            assert!(
                fx.m.new_memory_cursor(Hold::NONE, 0x55, 4)
                    .unwrap()
                    .is_some()
            );
            assert_eq!(
                fx.log().calls_to("cursor_from_virtual_offset")[0].args,
                vec![Arg::U64(0x55), Arg::U64(4)]
            );
            let Fx {
                m,
                task,
                task_dyn,
                platform,
                ..
            } = fx;
            drop((task, task_dyn));
            let mark = platform.call_log().calls().len();
            assert!(m.new_memory_cursor(Hold::NONE, 0, 4).unwrap().is_none());
            assert_eq!(platform.call_log().calls().len(), mark);
        }
    }

    mod get_data {
        use super::*;

        fn no_data(why: &str) -> Result<Vec<u8>, NoData> {
            Err(NoData {
                hole: false,
                why: why.to_owned(),
            })
        }

        #[test]
        fn returns_the_task_s_data() {
            let fx = Fx::new(&entries3(), Placement::Off);
            fx.platform.script().push(
                "data_from_virtual_offset",
                Step::value(Some(vec![1u8, 2, 3])),
            );
            assert_eq!(fx.m.get_data(Hold::NONE, 0x10, 3), Ok(vec![1, 2, 3]));
            assert_eq!(
                fx.log().calls_to("data_from_virtual_offset")[0].args,
                vec![Arg::U64(0x10), Arg::U64(3)]
            );
        }

        #[test]
        fn a_raise_is_no_data_with_its_detail() {
            let fx = Fx::new(&entries3(), Placement::Off);
            fx.platform.script().push(
                "data_from_virtual_offset",
                Step::raise("NSRangeException", "x"),
            );
            // No outer handler: the read's own takes it.
            assert_eq!(fx.m.get_data(Hold::NONE, 0, 1), no_data("x"));
        }

        #[test]
        fn a_task_that_gives_nothing_is_no_data() {
            let fx = Fx::new(&entries3(), Placement::Off);
            assert_eq!(
                fx.m.get_data(Hold::NONE, 0, 1),
                no_data("the task gave no data")
            );
        }

        #[test]
        fn a_gone_task_is_no_data() {
            let Fx {
                m, task, task_dyn, ..
            } = Fx::new(&entries3(), Placement::Off);
            drop((task, task_dyn));
            assert_eq!(m.get_data(Hold::NONE, 0, 1), no_data("the task is gone"));
        }

        /// A raise of the page-table walk's hole is a hole; every other raise, and a raise
        /// with no detail (named by its id), is not.
        #[test]
        fn only_the_walk_s_hole_is_a_hole() {
            use crate::memory::RaiseKind as Memory;
            use crate::raise::OnFault;
            let hole = Memory::WalkHole.raise(OnFault::Return, "a hole in the page table");
            assert_eq!(
                NoData::raised(&hole),
                NoData {
                    hole: true,
                    why: "a hole in the page table".to_owned()
                }
            );
            let root = Memory::WalkRootUnread.raise(OnFault::Return, "unread");
            assert!(!NoData::raised(&root).hole);
            let bare = Raise::new(RaiseId::Memory(Memory::WalkHole));
            assert_eq!(NoData::raised(&bare).why, "memory.walk.hole");
        }

        /// The read is a holder, not a pool: what Metal hands back `+0` inside it stays in the
        /// enclosing unit's real pool, which alone pushes and pops.
        #[test]
        fn the_read_pushes_no_pool_of_its_own() {
            use crate::rt::pool::test_native::{events, with_test_native};
            let fx = Fx::new(&entries3(), Placement::Off);
            fx.platform
                .script()
                .push("data_from_virtual_offset", Step::value(Some(vec![7u8])));
            with_test_native(|| {
                let outer = Pool::push(&*fx.platform);
                assert_eq!(fx.m.get_data(Hold::NONE, 0, 1), Ok(vec![7]));
                assert_eq!(events(), vec!["push 0x100"]);
                drop(outer);
                assert_eq!(events(), vec!["pop 0x100"]);
            });
        }

        /// The task's raise is no data and leaves no pool pushed: the enclosing unit's pop
        /// is the only one.
        #[test]
        fn a_raise_leaves_no_pool_of_its_own() {
            use crate::rt::pool::test_native::{events, with_test_native};
            let fx = Fx::new(&entries3(), Placement::Off);
            fx.platform.script().push(
                "data_from_virtual_offset",
                Step::raise("NSRangeException", "Address (0x0) out of range for task"),
            );
            with_test_native(|| {
                let outer = Pool::push(&*fx.platform);
                assert!(fx.m.get_data(Hold::NONE, 0, 1).is_err());
                assert_eq!(events(), vec!["push 0x100"]);
                drop(outer);
                assert_eq!(events(), vec!["pop 0x100"]);
            });
        }
    }

    mod writable_cursor {
        use super::*;

        fn writable(fx: &Fx) {
            fx.platform.script().push(
                "writable_cursor_from_virtual_offset",
                Step::value(Some(WritableCursor::default())),
            );
        }

        #[test]
        fn a_buffer_record_maps_its_pages() {
            let fx = Fx::new(&[(0, 0, 0), (1, 0x10, REC)], Placement::Off);
            let rec = ObjectBuffer {
                length: 0x2345,
                page: 3,
                flags: 1,
                field_0d: [0; 3],
            };
            fx.write(REC, rec.as_bytes());
            fx.next_cursor(0, 0x10);
            writable(&fx);
            assert!(
                fx.m.new_writable_memory_cursor_for_buffer(Hold::NONE, 1)
                    .unwrap()
                    .is_ok()
            );
            assert_eq!(
                fx.log().calls_to("writable_cursor_from_virtual_offset")[0].args,
                vec![Arg::U64(3 * APV_PAGE_SIZE), Arg::U64(0x2345)]
            );
        }

        #[test]
        fn a_cpu_buffer_record_is_twelve_bytes() {
            let fx = Fx::new(&[(0, 0, 0), (19, 0xc, REC)], Placement::Off);
            fx.write(
                REC,
                ObjectCpuBuffer {
                    length: 0x10,
                    page: 0xffff_ffff,
                }
                .as_bytes(),
            );
            fx.next_cursor(0, 0xc);
            writable(&fx);
            assert!(
                fx.m.new_writable_memory_cursor_for_buffer(Hold::NONE, 1)
                    .unwrap()
                    .is_ok()
            );
            assert_eq!(
                fx.log().calls_to("writable_cursor_from_virtual_offset")[0].args,
                vec![Arg::U64(0xffff_ffff * APV_PAGE_SIZE), Arg::U64(0x10)]
            );
        }

        #[test]
        fn a_short_record_fails_the_read() {
            let fx = Fx::new(&[(0, 0, 0), (1, 0xc, REC)], Placement::Off);
            fx.next_cursor(0, 0xc);
            let e =
                fx.m.new_writable_memory_cursor_for_buffer(Hold::NONE, 1)
                    .unwrap()
                    .err()
                    .unwrap();
            assert!(e.starts_with("buffer record not readable: "), "{e}");
        }

        #[test]
        fn refusals() {
            let fx = Fx::new(&[(0, 0, 0), (7, 0x10, REC), (1, 0x10, REC)], Placement::Off);
            // Reference 0 reads entry 0 (type 0) after its cursor.
            fx.next_cursor(0, 0);
            assert_eq!(
                fx.m.new_writable_memory_cursor_for_buffer(Hold::NONE, 0)
                    .unwrap()
                    .err()
                    .unwrap(),
                "object 0 has an empty entry"
            );
            fx.next_cursor(0, 0x10);
            assert_eq!(
                fx.m.new_writable_memory_cursor_for_buffer(Hold::NONE, 1)
                    .unwrap()
                    .err()
                    .unwrap(),
                "object of type 7 is not a buffer"
            );
            assert_eq!(
                fx.m.new_writable_memory_cursor_for_buffer(Hold::NONE, 2)
                    .unwrap()
                    .err()
                    .unwrap(),
                "no cursor over object 2's record"
            );
            assert_eq!(
                fx.m.new_writable_memory_cursor_for_buffer(Hold::NONE, 9)
                    .unwrap()
                    .err()
                    .unwrap(),
                "reference 9 is past the object list's 3 entries"
            );
        }

        #[test]
        fn a_null_task_cursor_is_an_error() {
            let fx = Fx::new(&[(1, 0x10, REC)], Placement::Off);
            fx.next_cursor(0, 0x10);
            assert!(
                fx.m.new_writable_memory_cursor_for_buffer(Hold::NONE, 0)
                    .unwrap()
                    .is_err()
            );
        }
    }
}
