//! The Metal state kinds the manager caches beside the resources: sampler and depth-stencil
//! states, fences, functions, compute and render pipeline states and rasterization rate maps.
//!
//! Each kind is a marker naming its cache ([`ManagedKind`]); the manager's lookups, deletes and
//! teardown reach every kind through the one generic body in [`super::cache`], and each kind's
//! creator is its module under [`super::create`].
//!
//! # Contract
//!
//! What the guest observes of a creation, each held by the tests named (under each creator's
//! `tests`):
//! - **Check order**: the entry's object type (7 for a state, 6 for a function), the operation
//!   header, the command type, then the command record; each failure is a refusal with nothing
//!   sent to Metal (`create::sampler::tests::refusals`, `create::pipeline::tests::create::*`).
//! - **Descriptor fill order**: a sampler's fifteen properties and a depth-stencil state's two,
//!   then each flagged face's six, are sent in a fixed order
//!   (`create::sampler::tests::the_descriptor_is_filled_in_order`,
//!   `create::depth_stencil::tests::the_descriptor_and_its_faces_are_filled_in_order`).
//! - **Placement**, per kind: a sampler sets its index when the ID is set and checks the ID when
//!   the index is set, and writes both back only with argument-buffer support; a depth-stencil
//!   state zeroes the index; compute and render states force the index when the ID is set and
//!   always write back index then ID; a tile pipeline, a fence, a function and a rate map touch no
//!   placement.
//! - **Pipeline records**: compute and render hand Metal the cursor's remaining bytes, a tile
//!   pipeline the descriptor's length; Metal resolves the functions a descriptor names through the
//!   manager while it deserializes, and a raise there unwinds out of the creation; the state is
//!   registered by its unique identifier.
//! - **Nil answers**: Metal's nil (a fence, a sampler, a depth-stencil state, a rate map, a
//!   layer) is a refusal; a pipeline or library that is nil with an error is logged (not a render
//!   state's) and refused; nil without an error dies ([`super::DeathKind`]); a nil function
//!   reaches the registration's assert.
//! - **Lifetimes**: a function's library goes when the function is made; a rate map's layers
//!   outlive the map descriptor's release on every return (`create::rate_map::create`).
//! - **Through the manager**: a kind's lookup makes the object once, caches it and writes its
//!   placement into the guest's list (`tests::a_sampler_is_made_cached_and_placed_through_the_manager`);
//!   a pipeline is registered for its reference
//!   (`tests::a_compute_pipeline_is_registered_for_its_reference`).

use crate::guest::GuestPtr;
use crate::guest::records::ObjectEntry;
use crate::memory::cursor::Cursor;
use crate::platform::{Handle, Platform};
use crate::raise::RaiseResult;
use crate::resources::manager::create::{
    depth_stencil, fence, function, pipeline, rate_map, sampler,
};
use crate::resources::manager::{Cache, Caches, ManagedKind, ResourceManager};
use crate::rt::Hold;

/// One state kind: its marker, the handle it caches, its cache, and how its creator is called.
macro_rules! state_kind {
    (
        $(#[$doc:meta])*
        $kind:ident: $obj:ident in $cache:ident,
        |$m:ident, $hold:pat_param, $reference:pat_param, $entry:ident, $cursor:ident, $placement:pat_param| $create:expr
    ) => {
        $(#[$doc])*
        #[derive(Debug)]
        pub struct $kind;

        impl<P: Platform> ManagedKind<P> for $kind {
            type Obj = P::$obj;
            fn sub_manager(c: &Caches<P>) -> &Cache<Self::Obj> {
                &c.$cache
            }
            fn sub_manager_mut(c: &mut Caches<P>) -> &mut Cache<Self::Obj> {
                &mut c.$cache
            }
            fn retain(o: &Self::Obj) -> Self::Obj {
                o.clone_ref()
            }
            fn create_object(
                $hold: Hold<'_>,
                $m: &ResourceManager<P>,
                $reference: u32,
                $entry: &ObjectEntry,
                $cursor: Cursor,
                $placement: Option<GuestPtr>,
            ) -> RaiseResult<Result<Self::Obj, String>> {
                $create
            }
        }
    };
}

state_kind! {
    /// The sampler states.
    SamplerStates: SamplerState in sampler_states,
    |m, _, _, entry, cursor, placement| sampler::create(m, entry, cursor, placement)
}

state_kind! {
    /// The depth-stencil states.
    DepthStencilStates: DepthStencilState in depth_stencil_states,
    |m, _, _, entry, cursor, placement| depth_stencil::create(m, entry, cursor, placement)
}

state_kind! {
    /// The fences.
    Fences: Fence in fences,
    |m, _, _, entry, cursor, _| fence::create(m, entry, cursor)
}

state_kind! {
    /// The functions: released last of the manager's caches, after its lookup.
    Functions: Function in functions,
    |m, hold, reference, entry, cursor, _| function::create(hold, m, reference, entry, cursor)
}

state_kind! {
    /// The compute pipeline states, registered by their unique identifiers.
    ComputePipelineStates: ComputePipeline in compute_pipeline_states,
    |m, hold, reference, entry, cursor, placement| {
        pipeline::create_compute(hold, m, reference, entry, cursor, placement)
    }
}

state_kind! {
    /// The render pipeline states (render and tile), registered by their unique identifiers.
    RenderPipelineStates: RenderPipeline in render_pipeline_states,
    |m, hold, reference, entry, cursor, placement| {
        pipeline::create_render(hold, m, reference, entry, cursor, placement)
    }
}

state_kind! {
    /// The rasterization rate maps.
    RasterizationRateMaps: RateMap in rasterization_rate_maps,
    |m, _, _, entry, cursor, _| rate_map::create(m, entry, cursor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::resource::{ComputePipeline, SamplerState};
    use crate::resources::manager::fixture::{self, Fx, REC};

    /// The header, then `rest`.
    fn record(command_type: u32, rest: &[u8]) -> Vec<u8> {
        let mut v = command_type.to_le_bytes().to_vec();
        v.extend([0; 4]);
        v.extend(rest);
        v
    }

    /// A sampler through the manager: its cache, its creator, and its placement written back
    /// into the guest's placement list; a second lookup makes nothing.
    #[test]
    fn a_sampler_is_made_cached_and_placed_through_the_manager() {
        let fx = Fx::new(&[(0, 0, 0), (7, 0x24, REC)], fixture::Placement::Mapped);
        // supportArgumentBuffers
        let mut command = vec![0; 8];
        command.extend(1_u32.to_le_bytes());
        command.extend([0; 0x10]);
        let r = record(3, &command);
        fx.write(REC, &r);
        fx.next_cursor(0, r.len() as u64);
        let s =
            fx.m.get_object::<SamplerStates>(Hold::NONE, 1)
                .unwrap()
                .unwrap()
                .unwrap();
        assert_eq!(fx.placement(1), (s.resource_index(), s.gpu_resource_id()));
        let mark = fx.clear_log();
        let again =
            fx.m.get_object::<SamplerStates>(Hold::NONE, 1)
                .unwrap()
                .unwrap()
                .unwrap();
        assert_eq!(again.id(), s.id());
        assert!(fx.sels_since(mark).is_empty());
    }

    /// A compute pipeline through the manager is registered by its unique identifier for its
    /// reference.
    #[test]
    fn a_compute_pipeline_is_registered_for_its_reference() {
        let fx = Fx::new(
            &[(0, 0, 0), (0, 0, 0), (7, 0x10, REC)],
            fixture::Placement::Off,
        );
        let r = record(11, &[0; 8]);
        fx.write(REC, &r);
        fx.next_cursor(0, r.len() as u64);
        let s =
            fx.m.get_object::<ComputePipelineStates>(Hold::NONE, 2)
                .unwrap()
                .unwrap()
                .unwrap();
        assert_eq!(fx.m.get_pipeline_ref(s.unique_identifier()), Ok(2));
    }
}
