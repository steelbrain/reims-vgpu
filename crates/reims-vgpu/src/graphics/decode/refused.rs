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
//! - a resolver that meets one answers [`reference`], a raise of its own, which the command
//!   buffer's walk takes at the op ([`is_refused`]): an op that uses the object itself (a copy, a
//!   fill, an indirect or indexed draw from it, a fence update or wait) is dropped;
//! - a binding of one binds nil (a pipeline state is not set at all) and marks its slot
//!   ([`Marks`]): the encoder's draws and dispatches are dropped while a slot of it is marked, and
//!   a binding of an object the host holds, or of nil, at that slot ends the mark;
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

use reims_graphics::wire::{
    RGFX_CMD_BUFFER_AT_OFFSET, RGFX_CMD_BUFFER_CREATE, RGFX_CMD_BUFFER_MANAGED_AT_OFFSET,
    RGFX_CMD_BUFFER_OVER_HOST_MEMORY, RGFX_CMD_BUFFER_OVER_PAGES,
    RGFX_CMD_BUFFER_OVER_SURFACE_PAGES, RGFX_CMD_COMPUTE_PIPELINE_CREATE,
    RGFX_CMD_DEPTH_STENCIL_CREATE, RGFX_CMD_EVENT_CREATE, RGFX_CMD_FENCE_CREATE,
    RGFX_CMD_HEAP_CREATE, RGFX_CMD_HEAP_OVER_HOST_MEMORY, RGFX_CMD_HEAP_OVER_PAGES,
    RGFX_CMD_LINEAR_TEXTURE_FROM_BUFFER, RGFX_CMD_MANAGED_BUFFER_CREATE,
    RGFX_CMD_MANAGED_TEXTURE_CREATE, RGFX_CMD_RENDER_PIPELINE_CREATE, RGFX_CMD_SAMPLER_CREATE,
    RGFX_CMD_SHARED_EVENT_CREATE, RGFX_CMD_TEXTURE_CREATE, RGFX_CMD_TEXTURE_FROM_BUFFER,
    RGFX_CMD_TEXTURE_FROM_IOSURFACE, RGFX_CMD_TEXTURE_VIEW, RGFX_CMD_VIDEO_SESSION_CREATE,
    RgfxObjectCreate,
};
use reims_vgpu_core::raise::{Origin, Raise, RaiseResult};

use super::read;

/// The site of [`reference`]'s raise.
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

    /// The raise of a resolver that met it, named by `obj_id` ([`reference`]).
    pub(crate) fn reference(&self, obj_id: u64) -> Raise {
        self.used(obj_id);
        reference(obj_id)
    }
}

/// The raise of a resolver that met `obj_id`, which stands for a refused object: the op that
/// named it is dropped where the command buffer's walk takes it ([`is_refused`]), and elsewhere
/// it is refused as any other raise is.
pub(crate) fn reference(obj_id: u64) -> Raise {
    Raise::foreign(SITE).with_detail(format_args!(
        "an op named object {obj_id:#x}, whose create the host refused"
    ))
}

/// Whether `raise` is a resolver's of an object whose create the host refused ([`reference`]).
pub(crate) fn is_refused(raise: &Raise) -> bool {
    raise.origin() == Origin::Boundary(SITE)
}

/// `got`, a nil-allowed resolve, with an object whose create the host refused resolved to nil:
/// an entry of a use list.
pub(crate) fn or_nil<T>(got: RaiseResult<Option<T>>) -> RaiseResult<Option<T>> {
    match got {
        Err(raise) if is_refused(&raise) => Ok(None),
        got => got,
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
}

/// The open encoder's binding slots that hold an object whose create the host refused, bound as
/// nil: its draws and dispatches are dropped while any is held, as what they would read is
/// not there. Empty in every run that refuses nothing, where each binding costs one look.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Marks {
    slots: RefCell<HashSet<Slot>>,
    /// The encoder's later ops are all dropped: a predication's start named a refused buffer, so
    /// its ends and what lies between them have nothing to pair with.
    skipped: Cell<bool>,
}

impl Marks {
    /// Whether a slot holds an object whose create was refused.
    pub(crate) fn any(&self) -> bool {
        !self.slots.borrow().is_empty()
    }

    /// Whether `slot` holds an object whose create was refused.
    pub(crate) fn holds(&self, slot: Slot) -> bool {
        self.any() && self.slots.borrow().contains(&slot)
    }

    /// `got`, the resolve of an object to bind at `slot`: an object whose create was refused is
    /// bound as nil, and marks the slot; any other binding ends the slot's mark.
    pub(crate) fn bound<T>(
        &self,
        slot: Slot,
        got: RaiseResult<Option<T>>,
    ) -> RaiseResult<Option<T>> {
        match got {
            Err(raise) if is_refused(&raise) => {
                self.slots.borrow_mut().insert(slot);
                Ok(None)
            }
            got => {
                if got.is_ok() && self.any() {
                    self.slots.borrow_mut().remove(&slot);
                }
                got
            }
        }
    }

    /// The encoder's later ops are all dropped.
    pub(crate) fn skip_rest(&self) {
        self.skipped.set(true);
    }

    /// Whether the encoder's later ops are all dropped ([`Self::skip_rest`]).
    pub(crate) fn skips(&self) -> bool {
        self.skipped.get()
    }

    /// Whether an op that draws or dispatches is dropped.
    pub(crate) fn drops_draws(&self) -> bool {
        self.skips() || self.any()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A binding of a refused object binds nil and marks its slot; a binding of another object, or
    /// of nil, at that slot ends the mark; a binding at another slot leaves it.
    #[test]
    fn a_slot_is_marked_from_a_refused_binding_to_the_next_binding() {
        let marks = Marks::default();
        let slot = Slot::new(Stage::Fragment, Bound::Texture, 0);
        let other = Slot::new(Stage::Fragment, Bound::Texture, 1);
        assert!(!marks.drops_draws());
        assert_eq!(marks.bound::<u32>(slot, Err(reference(7))), Ok(None));
        assert!(marks.holds(slot) && marks.drops_draws());
        assert_eq!(marks.bound(other, Ok(Some(&1u32))), Ok(Some(&1)));
        assert!(marks.holds(slot), "a binding elsewhere ends no mark");
        assert_eq!(marks.bound::<u32>(slot, Ok(None)), Ok(None));
        assert!(!marks.holds(slot) && !marks.drops_draws());
    }

    /// A raise that is not a refused reference passes through a binding, and marks nothing.
    #[test]
    fn another_raise_passes_through_a_binding() {
        let marks = Marks::default();
        let slot = Slot::state(Bound::Pipeline);
        let other = Raise::foreign("rgfx.decode.object");
        assert!(!is_refused(&other));
        assert_eq!(marks.bound::<u32>(slot, Err(other.clone())), Err(other));
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
        use reims_graphics::wire::{
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
