//! Objects whose posted create the host refused (docs/graphics-kext/design.md, "Guest input the
//! host refuses"). The bundle answers most creates at the call without waiting for the host
//! (buffers, heaps, textures, samplers, depth/stencil states, events and fences, video sessions,
//! and pipelines equal to ones the device accepted before), so where the host then refuses one,
//! the guest holds an object the host never made, and goes on naming it. Metal would have made
//! it, or answered nil at the call, and failed none of the command buffers after.
//!
//! So a refused posted create leaves its id standing for a refused object ([`Refused`], an entry
//! of its context's table like any object's, until its destroy, the id's next create, the
//! context's close or the device's reset), and what names it is dropped, never the command
//! buffer that names it:
//! - a resolver that meets one answers [`reference()`], a raise of its own, which the command
//!   buffer's walk takes at the op ([`is_refused`]): an op that uses the object itself (a copy, a
//!   fill, an indirect or indexed draw from it, a fence update or wait) is dropped;
//! - a binding of one binds nil and marks its slot ([`Marks`]), until a binding of an object the
//!   host holds, or of nil, at that slot ends the mark. A buffer, texture or sampler bound so is
//!   the guest's own nil there, and the encoder's draws and dispatches run, as Metal's do with nil
//!   bound: a read of the texture is zero on the AMD 5300M and the Intel 630, and of the buffer
//!   zero on the AMD and undefined on the Intel; a sampler's is the default one's on the AMD and
//!   zero on the Intel (`oracle_nilbind`, docs/graphics-kext/dgpu-oracle.md, Q25). An offset op at
//!   such a buffer slot sets nothing. A pipeline state is not set at all, and a depth/stencil
//!   state or tessellation factor buffer is bound nil: the encoder's draws and dispatches are
//!   dropped while one of these is marked;
//! - an indirect command buffer's CPU encode of one is dropped and its command reset, and that
//!   command's later draws and dispatches are dropped until the guest resets it
//!   ([`super::icb::Dropped`]);
//! - an entry of a use list (`useResources:`, `useHeaps:`, a memory barrier's) is nil;
//! - a render pass with an attachment of one, and a video encoder of one, is skipped whole;
//! - a wait for an event of one is dropped, and a signal of a shared event of one still reaches
//!   the guest's value ([`super::cmdbuf`]);
//! - a create over one (a texture over a refused buffer, a view of a refused texture) is refused
//!   too, and stands refused in turn.
//!
//! The command buffer completes as it would have: what Metal fails it for, it fails for, and
//! nothing else. An id that names nothing the host ever held is a guest protocol error as before,
//! and faults its command buffer.

use std::cell::{Cell, RefCell};
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};

use reims_vgpu_core::raise::{Origin, Raise, RaiseResult};
use reims_vgpu_graphics::wire::{
    RGFX_CMD_BUFFER_AT_OFFSET, RGFX_CMD_BUFFER_CREATE, RGFX_CMD_BUFFER_MANAGED_AT_OFFSET,
    RGFX_CMD_BUFFER_OVER_HOST_MEMORY, RGFX_CMD_BUFFER_OVER_PAGES,
    RGFX_CMD_BUFFER_OVER_SURFACE_PAGES, RGFX_CMD_COMPUTE_PIPELINE_CREATE,
    RGFX_CMD_DEPTH_STENCIL_CREATE, RGFX_CMD_EVENT_CREATE, RGFX_CMD_FENCE_CREATE,
    RGFX_CMD_HEAP_CREATE, RGFX_CMD_HEAP_OVER_HOST_MEMORY, RGFX_CMD_HEAP_OVER_PAGES,
    RGFX_CMD_LINEAR_TEXTURE_FROM_BUFFER, RGFX_CMD_MANAGED_BUFFER_CREATE,
    RGFX_CMD_MANAGED_TEXTURE_CREATE, RGFX_CMD_MESH_RENDER_PIPELINE_CREATE,
    RGFX_CMD_RENDER_PIPELINE_CREATE, RGFX_CMD_SAMPLER_CREATE, RGFX_CMD_SHARED_EVENT_CREATE,
    RGFX_CMD_TEXTURE_CREATE, RGFX_CMD_TEXTURE_FROM_BUFFER, RGFX_CMD_TEXTURE_FROM_IOSURFACE,
    RGFX_CMD_TEXTURE_VIEW, RGFX_CMD_VIDEO_SESSION_CREATE, RgfxObjectCreate,
};

use super::read;

/// The site of [`reference()`]'s raise.
const SITE: &str = "rgfx.decode.refused";

/// What an id whose posted create the host refused stands for: the create's message type, and
/// whether a use of it was logged.
#[derive(Debug)]
pub(crate) struct Refused {
    /// The create's message type (`RGFX_CMD_*`).
    pub(crate) kind: u32,
    /// Whether a use of it was logged: the first is, at WARN, and no other.
    logged: AtomicBool,
}

impl Refused {
    /// The object a create of message type `kind` would have made.
    pub(crate) fn of(kind: u32) -> Self {
        Refused {
            kind,
            logged: AtomicBool::new(false),
        }
    }

    /// Whether the create was a shared event's.
    pub(crate) fn shared_event(&self) -> bool {
        self.kind == RGFX_CMD_SHARED_EVENT_CREATE
    }

    /// Notes a use of `obj_id`, which stands for it: the first is logged.
    pub(crate) fn used(&self, obj_id: u64) {
        if !self.logged.swap(true, Ordering::Relaxed) {
            log::warn!(
                target: "reims_vgpu::graphics",
                "object {obj_id:#x}, whose create (type {:#x}) the host refused, is named: what \
                 names it is dropped, and its command buffers go on",
                self.kind
            );
        }
    }

    /// The raise of a resolver that met it, named by `obj_id` ([`reference()`]).
    #[cold]
    #[inline(never)]
    pub(crate) fn reference(&self, obj_id: u64) -> Raise {
        self.used(obj_id);
        reference(obj_id)
    }
}

/// The raise of a resolver that met `obj_id`, which stands for a refused object: the op that
/// named it is dropped where the command buffer's walk takes it ([`is_refused`]), and elsewhere
/// it is refused as any other raise is. Out of line, as is [`Refused::reference`], so no
/// resolver's own code holds its detail's formatting.
#[cold]
#[inline(never)]
pub(crate) fn reference(obj_id: u64) -> Raise {
    Raise::foreign(SITE).with_detail(format_args!(
        "an op named object {obj_id:#x}, whose create the host refused"
    ))
}

/// Whether `raise` is a resolver's of an object whose create the host refused ([`reference()`]).
pub(crate) fn is_refused(raise: &Raise) -> bool {
    raise.origin() == Origin::Boundary(SITE)
}

/// `got`, a nil-allowed resolve, with an object whose create the host refused resolved to nil:
/// an entry of a use list.
#[inline]
pub(crate) fn or_nil<T>(got: RaiseResult<Option<T>>) -> RaiseResult<Option<T>> {
    match got {
        Ok(object) => Ok(object),
        Err(raise) => nil_if_refused(raise).map(|()| None),
    }
}

/// [`or_nil`]'s raise: a refused object's is nil, any other is the entry's.
#[cold]
#[inline(never)]
fn nil_if_refused(raise: Raise) -> RaiseResult<()> {
    if is_refused(&raise) {
        Ok(())
    } else {
        Err(raise)
    }
}

/// `got`, a resolve that must name an object, with an object whose create the host refused
/// resolved to nil: an entry of a use list.
pub(crate) fn some_or_nil<T>(got: RaiseResult<T>) -> RaiseResult<Option<T>> {
    or_nil(got.map(Some))
}

/// The message type and object id of `message` where it is a create the bundle answers at the
/// call without the host's answer, whose refusal the guest so never learns of. The creates the
/// guest waits for (a queue's, a library's, an indirect command buffer's, a sampler asked for its
/// id) are answered refused and named by nothing after: they stand for nothing. A pipeline's is
/// among these where the bundle posts it unanswered (an equal create's, "Known pipeline
/// creates"); one it waited for stands refused harmlessly, never named.
pub(crate) fn posted_create(message: &[u8]) -> Option<(u32, u64)> {
    let m = read::<RgfxObjectCreate>(message)?;
    let kind = m.header.r#type;
    let posted = matches!(
        kind,
        RGFX_CMD_EVENT_CREATE
            | RGFX_CMD_FENCE_CREATE
            | RGFX_CMD_SHARED_EVENT_CREATE
            | RGFX_CMD_SAMPLER_CREATE
            | RGFX_CMD_DEPTH_STENCIL_CREATE
            | RGFX_CMD_BUFFER_CREATE
            | RGFX_CMD_BUFFER_OVER_HOST_MEMORY
            | RGFX_CMD_BUFFER_OVER_SURFACE_PAGES
            | RGFX_CMD_BUFFER_MANAGED_AT_OFFSET
            | RGFX_CMD_MANAGED_BUFFER_CREATE
            | RGFX_CMD_BUFFER_OVER_PAGES
            | RGFX_CMD_BUFFER_AT_OFFSET
            | RGFX_CMD_HEAP_CREATE
            | RGFX_CMD_HEAP_OVER_HOST_MEMORY
            | RGFX_CMD_HEAP_OVER_PAGES
            | RGFX_CMD_TEXTURE_CREATE
            | RGFX_CMD_MANAGED_TEXTURE_CREATE
            | RGFX_CMD_TEXTURE_VIEW
            | RGFX_CMD_TEXTURE_FROM_BUFFER
            | RGFX_CMD_LINEAR_TEXTURE_FROM_BUFFER
            | RGFX_CMD_TEXTURE_FROM_IOSURFACE
            | RGFX_CMD_VIDEO_SESSION_CREATE
            | RGFX_CMD_RENDER_PIPELINE_CREATE
            | RGFX_CMD_MESH_RENDER_PIPELINE_CREATE
            | RGFX_CMD_COMPUTE_PIPELINE_CREATE
    );
    (posted && m.obj_id != 0).then_some((kind, m.obj_id))
}

/// The stage of an encoder a binding is made for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Stage {
    Vertex,
    Fragment,
    Tile,
    /// A mesh render pipeline's object stage.
    Object,
    /// A mesh render pipeline's mesh stage.
    Mesh,
    Kernel,
    /// The encoder's own state: its pipeline, its depth/stencil state, its tessellation factors.
    Encoder,
}

/// What a binding binds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Bound {
    Buffer,
    Texture,
    Sampler,
    Pipeline,
    DepthStencil,
    TessellationFactors,
}

/// One binding slot of an encoder.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Slot {
    stage: Stage,
    bound: Bound,
    index: u64,
}

impl Slot {
    /// The slot `index` of `stage`'s bindings of `bound`.
    pub(crate) fn new(stage: Stage, bound: Bound, index: u64) -> Self {
        Slot {
            stage,
            bound,
            index,
        }
    }

    /// The encoder's own slot of `bound` (its pipeline, its depth/stencil state).
    pub(crate) fn state(bound: Bound) -> Self {
        Slot::new(Stage::Encoder, bound, 0)
    }

    /// Whether the encoder's draws and dispatches are dropped while the slot holds a refused
    /// object: one of the encoder's own state's, which its draws cannot go without. A buffer,
    /// texture or sampler slot is nil, which Metal draws with.
    fn drops_draws(&self) -> bool {
        matches!(
            self.bound,
            Bound::Pipeline | Bound::DepthStencil | Bound::TessellationFactors
        )
    }
}

/// The open encoder's binding slots that hold an object whose create the host refused, bound as
/// nil: its draws and dispatches are dropped while a slot of its own state is held (a pipeline, a
/// depth/stencil state, tessellation factors), and run while only a buffer, texture or sampler
/// slot is, as with nil bound there. Empty in every run that refuses nothing, where a binding
/// reads one flag ([`Self::any`]), a walk one after each op ([`Self::drops_draws`]), and the set
/// itself is reached only where a slot is marked or its mark ends.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Marks {
    slots: RefCell<HashSet<Slot>>,
    /// Whether `slots` holds any.
    held: Cell<bool>,
    /// Whether the encoder's draws and dispatches are dropped ([`Self::drops_draws`]), kept where
    /// `slots` and `skipped` change.
    drops: Cell<bool>,
    /// The encoder's later ops are all dropped: a predication's start named a refused buffer, so
    /// its ends and what lies between them have nothing to pair with.
    skipped: Cell<bool>,
}

/// A binding, in the op's own function: `$got`, the resolve of an object to bind at `$slot` of
/// `$marks`, as the object, or nil where its create was refused, which marks the slot; any other
/// binding ends the slot's mark, and any other raise returns from the op. The resolve is matched
/// where the resolver wrote it, and only a raise is passed on, out of line
/// ([`Marks::bind_raise`]): through a function, which one of `render::decode_op`'s size calls out
/// of line, each binding passed the whole result (a [`Raise`]'s size) through memory.
macro_rules! bind {
    ($marks:expr, $slot:expr, $got:expr) => {
        match $got {
            Ok(object) => {
                if $marks.any() {
                    $marks.unmark($slot);
                }
                object
            }
            Err(raise) => {
                $marks.bind_raise($slot, raise)?;
                None
            }
        }
    };
}
pub(crate) use bind;

impl Marks {
    /// Whether a slot holds an object whose create was refused.
    #[inline]
    pub(crate) fn any(&self) -> bool {
        self.held.get()
    }

    /// Whether `slot` holds an object whose create was refused.
    #[inline]
    pub(crate) fn holds(&self, slot: Slot) -> bool {
        self.any() && self.marked(slot)
    }

    /// [`bind!`]'s raise: a refused object's marks `slot` and binds nil, any other is the
    /// binding's.
    #[cold]
    #[inline(never)]
    pub(crate) fn bind_raise(&self, slot: Slot, raise: Raise) -> RaiseResult<()> {
        if is_refused(&raise) {
            self.mark(slot);
            Ok(())
        } else {
            Err(raise)
        }
    }

    /// Whether the set holds `slot`.
    #[cold]
    #[inline(never)]
    fn marked(&self, slot: Slot) -> bool {
        self.slots.borrow().contains(&slot)
    }

    /// Marks `slot`.
    #[cold]
    #[inline(never)]
    fn mark(&self, slot: Slot) {
        self.slots.borrow_mut().insert(slot);
        self.held.set(true);
        if slot.drops_draws() {
            self.drops.set(true);
        }
    }

    /// Ends `slot`'s mark, if it has one.
    #[cold]
    #[inline(never)]
    pub(crate) fn unmark(&self, slot: Slot) {
        let mut slots = self.slots.borrow_mut();
        slots.remove(&slot);
        self.held.set(!slots.is_empty());
        let drops = self.skipped.get() || slots.iter().any(Slot::drops_draws);
        self.drops.set(drops);
    }

    /// The encoder's later ops are all dropped.
    pub(crate) fn skip_rest(&self) {
        self.skipped.set(true);
        self.drops.set(true);
    }

    /// Whether the encoder's later ops are all dropped ([`Self::skip_rest`]).
    pub(crate) fn skips(&self) -> bool {
        self.skipped.get()
    }

    /// Whether an op that draws or dispatches is dropped: the encoder's later ops are, or a slot of
    /// its own state holds a refused object ([`Slot::drops_draws`]).
    #[inline]
    pub(crate) fn drops_draws(&self) -> bool {
        self.drops.get()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A binding of `got` at `slot` of `marks`, as an op makes one ([`bind!`]).
    fn binding<T>(
        marks: &Marks,
        slot: Slot,
        got: RaiseResult<Option<T>>,
    ) -> RaiseResult<Option<T>> {
        Ok(bind!(marks, slot, got))
    }

    /// A binding of a refused object binds nil and marks its slot; a binding of another object, or
    /// of nil, at that slot ends the mark; a binding at another slot leaves it. The draws are
    /// dropped while the encoder's own state's slot is marked.
    #[test]
    fn a_slot_is_marked_from_a_refused_binding_to_the_next_binding() {
        let marks = Marks::default();
        let slot = Slot::state(Bound::Pipeline);
        let other = Slot::new(Stage::Fragment, Bound::Texture, 1);
        assert!(!marks.drops_draws());
        assert_eq!(binding::<u32>(&marks, slot, Err(reference(7))), Ok(None));
        assert!(marks.holds(slot) && marks.drops_draws());
        assert_eq!(binding(&marks, other, Ok(Some(&1u32))), Ok(Some(&1)));
        assert!(marks.holds(slot), "a binding elsewhere ends no mark");
        assert_eq!(binding::<u32>(&marks, slot, Ok(None)), Ok(None));
        assert!(!marks.holds(slot) && !marks.drops_draws());
    }

    /// A refused buffer, texture or sampler is nil at its slot, and the draws go on, as Metal's
    /// with nil bound there (oracle_nilbind); its slot stays marked, so an offset op there sets
    /// nothing. A refused state beside it drops them until it is bound again.
    #[test]
    fn a_refused_resource_binds_nil_and_drops_no_draws() {
        let marks = Marks::default();
        let resources = [
            Slot::new(Stage::Vertex, Bound::Buffer, 0),
            Slot::new(Stage::Fragment, Bound::Texture, 2),
            Slot::new(Stage::Kernel, Bound::Sampler, 1),
        ];
        for slot in resources {
            assert_eq!(binding::<u32>(&marks, slot, Err(reference(7))), Ok(None));
            assert!(marks.holds(slot));
        }
        assert!(!marks.drops_draws());
        for bound in [
            Bound::Pipeline,
            Bound::DepthStencil,
            Bound::TessellationFactors,
        ] {
            let state = Slot::state(bound);
            assert_eq!(binding::<u32>(&marks, state, Err(reference(8))), Ok(None));
            assert!(marks.drops_draws());
            assert_eq!(binding(&marks, state, Ok(Some(&1u32))), Ok(Some(&1)));
            assert!(!marks.drops_draws());
        }
        assert!(resources.iter().all(|&slot| marks.holds(slot)));
    }

    /// A raise that is not a refused reference passes through a binding, and marks nothing.
    #[test]
    fn another_raise_passes_through_a_binding() {
        let marks = Marks::default();
        let slot = Slot::state(Bound::Pipeline);
        let other = Raise::foreign("rgfx.decode.object");
        assert!(!is_refused(&other));
        assert_eq!(binding::<u32>(&marks, slot, Err(other.clone())), Err(other));
        assert!(!marks.drops_draws());
    }

    /// A use list's entry of a refused object is nil, and any other answer stands.
    #[test]
    fn a_use_list_entry_of_a_refused_object_is_nil() {
        assert_eq!(some_or_nil::<u32>(Err(reference(9))), Ok(None));
        assert_eq!(some_or_nil(Ok(3u32)), Ok(Some(3)));
        let other = Raise::foreign("rgfx.decode.object");
        assert_eq!(or_nil::<u32>(Err(other.clone())), Err(other));
    }

    /// The creates the bundle answers without the host stand refused; those it waits for, and a
    /// command buffer, do not.
    #[test]
    fn only_posted_creates_stand_refused() {
        use reims_vgpu_graphics::wire::{
            RGFX_CMD_COMMAND_BUFFER, RGFX_CMD_LIBRARY_CREATE, RGFX_CMD_QUEUE_CREATE, RgfxMsgHeader,
        };
        use zerocopy::IntoBytes;
        let create = |r#type, obj_id| {
            RgfxObjectCreate {
                header: RgfxMsgHeader {
                    r#type,
                    size: 0,
                    seqno: 0,
                },
                obj_id,
            }
            .as_bytes()
            .to_vec()
        };
        for kind in [
            RGFX_CMD_TEXTURE_CREATE,
            RGFX_CMD_TEXTURE_FROM_IOSURFACE,
            RGFX_CMD_BUFFER_CREATE,
            RGFX_CMD_MANAGED_BUFFER_CREATE,
            RGFX_CMD_SHARED_EVENT_CREATE,
            RGFX_CMD_RENDER_PIPELINE_CREATE,
            RGFX_CMD_MESH_RENDER_PIPELINE_CREATE,
        ] {
            assert_eq!(posted_create(&create(kind, 5)), Some((kind, 5)));
        }
        for kind in [
            RGFX_CMD_QUEUE_CREATE,
            RGFX_CMD_LIBRARY_CREATE,
            RGFX_CMD_COMMAND_BUFFER,
        ] {
            assert_eq!(posted_create(&create(kind, 5)), None);
        }
        assert_eq!(posted_create(&create(RGFX_CMD_TEXTURE_CREATE, 0)), None);
        assert_eq!(posted_create(&[0u8; 8]), None);
    }
}
