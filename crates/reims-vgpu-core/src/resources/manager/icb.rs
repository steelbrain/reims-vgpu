//! Indirect command buffers in the manager: their creation, and their CPU paging between the
//! guest's command records and Metal's indirect commands.
//!
//! [`create`] makes an ICB resource from the guest's object record and creation command.
//! [`prepare`] pages the guest's records into the ICB's commands (guest to host), and
//! [`synchronize`] pages the commands back into the records (host to guest); the resource
//! ([`crate::resources::icb`]) runs them on the CPU. A walk resolves the buffers its binds name
//! through [`lookup`], and runs over [`lookup::NoManager`] once the task's teardown has dropped
//! the manager. [`records`] holds the guest's layouts.
//!
//! The walks and the creator are generic over [`IcbWalkOps`] and [`ManagerOps`], what they ask
//! of the manager, so tests drive them with a recording manager (`testing`).
//!
//! # Contract
//!
//! What the guest observes, each held by the tests named (under each module's `tests`):
//! - **Creation** fills the descriptor in order
//!   (`create::tests::the_descriptor_is_filled_in_order`), sets a placement's resource ID as the
//!   index only when it is nonzero, checks the ICB's ID only when the placement's index is nonzero,
//!   and writes the ID back before the index (`create::tests::a_nonzero_index_checks_the_id`);
//!   Metal's creation reference leaks (`create::tests::the_creation_reference_leaks`).
//! - **Prepare** sends, per slot: a reset for an empty slot, the pipeline, the buffers, the
//!   threadgroup memory lengths, the barrier, then the command
//!   (`prepare::tests::render::a_draw_sets_pipeline_binds_strides_barrier_and_draws`,
//!   `prepare::tests::compute::a_dispatch_sets_everything_in_order`); an unresolved bind is
//!   skipped without reading its stride; a failed pipeline lookup is ignored and an unresolved
//!   index, patch or factor buffer skips the draw; an unreadable tessellation factor fails with
//!   text read from the draw record, or dies
//!   (`prepare::tests::render::a_long_string_with_a_size_dies`).
//! - **Synchronize** writes the type, the pipeline's reference, the binds, the lengths, the
//!   barrier and the arguments; only a failed cursor, type or pipeline write ends the walk; a
//!   compute command's strides and lengths take one word each, where prepare reads them
//!   (`synchronize::tests::compute::strides_are_sized_by_the_kernel_count`,
//!   `synchronize::tests::compute::a_prepared_dispatch_round_trips_every_length`); an empty
//!   compute slot writes only its type
//!   (`synchronize::tests::compute::an_empty_slot_writes_only_its_type`).
//! - **Metal's argument objects** are read whole, each getter once, in order, before the binds
//!   their addresses make are looked up
//!   (`synchronize::tests::compute::the_dispatch_arguments_are_read_once_each`).
//! - **With no manager**, a walk answers what it answers before it needs the manager and dies at
//!   its first use of it (`lookup::tests::no_manager::the_first_use_of_the_manager_dies`).

pub mod create;
pub mod lookup;
pub mod prepare;
pub mod records;
pub mod synchronize;
#[cfg(test)]
pub(crate) mod testing;

use std::fmt;
use std::sync::Arc;

use crate::memory::cursor::{Cursor, WritableCursor};
use crate::platform::Platform;
use crate::raise::RaiseResult;
use crate::raise::death::DeathId;
use crate::raise::death::DeathSite;
use crate::resources::icb::{IcbResource, IndirectCommandBufferFlags, IndirectCommandLayout};
use crate::resources::manager::ResourceManager;
use crate::resources::manager::create::Resources;
use crate::resources::manager::lookup::NoData;
use crate::resources::manager::records::ResourceAndOffset;
use crate::resources::manager::states::ComputePipelineStates;
use crate::resources::manager::states::RenderPipelineStates;
use crate::resources::resource::Resource;
use crate::rt::Hold;

#[cfg(test)]
pub(crate) use records::DrawPatches;

/// How the ICB paging dies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeathKind {
    /// A creation needs the manager's task, which is being torn down.
    TaskGone,
    /// A render command's pipeline lookup succeeded with nil (an assert).
    NilRenderPipeline,
    /// A compute command's pipeline lookup succeeded with nil (an assert).
    NilComputePipeline,
    /// A patch draw's unreadable tessellation factor formats a long string with a size read through
    /// a guest-chosen host pointer.
    TessellationText,
    /// The GPU-address lookup's floor lies above the address (an assert that always holds).
    AddressBelowFloor,
    /// A walk uses the manager after the task's teardown dropped it (a call through NULL).
    NoManager,
}

impl DeathKind {
    /// Its dotted id.
    pub const fn id(self) -> &'static str {
        match self {
            DeathKind::TaskGone => "manager.icb.task_gone",
            DeathKind::NilRenderPipeline => "manager.icb.nil_render_pipeline",
            DeathKind::NilComputePipeline => "manager.icb.nil_compute_pipeline",
            DeathKind::TessellationText => "manager.icb.tessellation_text",
            DeathKind::AddressBelowFloor => "manager.icb.address_below_floor",
            DeathKind::NoManager => "manager.icb.no_manager",
        }
    }

    /// The kind of death.
    pub const fn kind(self) -> crate::raise::death::DeathKind {
        match self {
            DeathKind::TaskGone | DeathKind::NoManager => crate::raise::death::DeathKind::NullDeref,
            DeathKind::NilRenderPipeline
            | DeathKind::NilComputePipeline
            | DeathKind::AddressBelowFloor => crate::raise::death::DeathKind::Assert,
            DeathKind::TessellationText => crate::raise::death::DeathKind::OutOfBounds,
        }
    }

    /// The site `testkit::capture` records and the stderr line names.
    pub const fn site(self) -> DeathSite {
        crate::resources::manager::DeathKind::Icb(self).site()
    }
}

impl fmt::Display for DeathKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}

impl From<DeathKind> for DeathId {
    fn from(k: DeathKind) -> Self {
        DeathId::Manager(crate::resources::manager::DeathKind::Icb(k))
    }
}

/// The text of a refused prepare, as bytes: guest bytes, UTF-8 or not. `Display`, which the log
/// line shows, is lossy (U+FFFD for each invalid sequence); [`IcbErrorText::as_bytes`] has the
/// bytes.
#[derive(Clone, PartialEq, Eq)]
pub struct IcbErrorText(Vec<u8>);

impl IcbErrorText {
    /// `prefix` followed by the raw `bytes`.
    fn with_bytes(prefix: &str, bytes: &[u8]) -> Self {
        let mut text = Vec::with_capacity(prefix.len().saturating_add(bytes.len()));
        text.extend_from_slice(prefix.as_bytes());
        text.extend_from_slice(bytes);
        Self(text)
    }

    /// The text's bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

impl From<String> for IcbErrorText {
    fn from(text: String) -> Self {
        Self(text.into_bytes())
    }
}

impl fmt::Display for IcbErrorText {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&String::from_utf8_lossy(&self.0))
    }
}

impl fmt::Debug for IcbErrorText {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.0.escape_ascii().to_string())
    }
}

impl PartialEq<&str> for IcbErrorText {
    fn eq(&self, other: &&str) -> bool {
        self.0 == other.as_bytes()
    }
}

/// What the manager's creators ask of it: its platform and device, Metal's function resolver,
/// task memory, the pipeline registry and the ICB resource. [`ResourceManager`] implements it;
/// the unit tests implement it with a recording manager (`testing`).
pub(crate) trait ManagerOps<P: Platform>: IcbWalkOps<P> {
    /// What [`ManagerOps::new_icb_resource`] makes (`Arc<Resource<P>>` for the manager).
    type IcbResource;
    /// The platform: what the creators make descriptors through.
    fn platform(&self) -> &P;
    /// The device; `None` is nil.
    fn device(&self) -> Option<&P::Device>;
    /// What Metal calls back while it deserializes a pipeline descriptor. `Err` is a raise
    /// inside the lookup, which unwinds through Metal.
    fn function_with_function_ref(
        &self,
        hold: Hold<'_>,
        function_ref: u64,
    ) -> RaiseResult<Option<P::Function>>;
    /// A copy of task memory, or why there is none.
    fn get_data(&self, hold: Hold<'_>, offset: u64, length: u64) -> Result<Vec<u8>, NoData>;
    /// Registers a pipeline's identifier for its reference.
    fn register_pipeline(&self, identifier: u64, reference: u32) -> Result<(), String>;
    /// An ICB resource over `icb` for the manager's task, unretained.
    fn new_icb_resource(
        &self,
        descriptor: &[u8],
        icb: Option<P::Icb>,
        layout: IndirectCommandLayout,
        flags: IndirectCommandBufferFlags,
    ) -> RaiseResult<Option<Self::IcbResource>>;
}

/// What the ICB walks ask of the manager: the pipeline registry, lookups by reference (which
/// make a missing object) and the buffer at a GPU address. Split from [`ManagerOps`] so a walk
/// can run with no manager ([`lookup::NoManager`]).
pub(crate) trait IcbWalkOps<P: Platform> {
    /// The reference registered for a pipeline's identifier.
    fn get_pipeline_ref(&self, identifier: u64) -> Result<u32, String>;
    /// The buffer of the resource `reference` names, made on a miss.
    fn get_buffer_object(
        &self,
        hold: Hold<'_>,
        reference: u32,
    ) -> RaiseResult<Result<Option<P::Buffer>, String>>;
    /// The render pipeline `reference` names, made on a miss.
    fn get_render_pipeline_object(
        &self,
        hold: Hold<'_>,
        reference: u32,
    ) -> RaiseResult<Result<Option<P::RenderPipeline>, String>>;
    /// The compute pipeline `reference` names, made on a miss.
    fn get_compute_pipeline_object(
        &self,
        hold: Hold<'_>,
        reference: u32,
    ) -> RaiseResult<Result<Option<P::ComputePipeline>, String>>;
    /// The buffer at a GPU address ([`lookup::resource_and_offset_for_address`]); never a
    /// raise.
    fn resource_and_offset_at(
        &self,
        hold: Hold<'_>,
        address: u64,
    ) -> Result<ResourceAndOffset<P>, String>;
}

impl<P: Platform> IcbWalkOps<P> for ResourceManager<P> {
    fn get_pipeline_ref(&self, identifier: u64) -> Result<u32, String> {
        ResourceManager::get_pipeline_ref(self, identifier)
    }

    fn get_buffer_object(
        &self,
        hold: Hold<'_>,
        reference: u32,
    ) -> RaiseResult<Result<Option<P::Buffer>, String>> {
        self.get_object_buffer(hold, reference)
    }

    fn get_render_pipeline_object(
        &self,
        hold: Hold<'_>,
        reference: u32,
    ) -> RaiseResult<Result<Option<P::RenderPipeline>, String>> {
        self.get_object::<RenderPipelineStates>(hold, reference)
    }

    fn get_compute_pipeline_object(
        &self,
        hold: Hold<'_>,
        reference: u32,
    ) -> RaiseResult<Result<Option<P::ComputePipeline>, String>> {
        self.get_object::<ComputePipelineStates>(hold, reference)
    }

    fn resource_and_offset_at(
        &self,
        hold: Hold<'_>,
        address: u64,
    ) -> Result<ResourceAndOffset<P>, String> {
        lookup::resource_and_offset_for_address(
            hold,
            address,
            |a| self.gpu_address_at_or_below(a),
            |r| self.get_existing_object::<Resources>(r),
            |resource: &Arc<Resource<P>>| resource.get_buffer(),
        )
    }
}

impl<P: Platform> ManagerOps<P> for ResourceManager<P> {
    type IcbResource = Arc<Resource<P>>;

    fn platform(&self) -> &P {
        ResourceManager::platform(self)
    }

    fn device(&self) -> Option<&P::Device> {
        ResourceManager::device(self)
    }

    fn function_with_function_ref(
        &self,
        hold: Hold<'_>,
        function_ref: u64,
    ) -> RaiseResult<Option<P::Function>> {
        crate::resources::manager::resolver::function_with_function_ref(hold, self, function_ref)
    }

    fn get_data(&self, hold: Hold<'_>, offset: u64, length: u64) -> Result<Vec<u8>, NoData> {
        ResourceManager::get_data(self, hold, offset, length)
    }

    fn register_pipeline(&self, identifier: u64, reference: u32) -> Result<(), String> {
        ResourceManager::register_pipeline(self, identifier, reference)
    }

    fn new_icb_resource(
        &self,
        descriptor: &[u8],
        icb: Option<P::Icb>,
        layout: IndirectCommandLayout,
        flags: IndirectCommandBufferFlags,
    ) -> RaiseResult<Option<Arc<Resource<P>>>> {
        // The resource holds the task unretained. The task is gone only while the task that owns
        // this manager is being torn down.
        let Some(task) = self.task() else {
            crate::fatal!(DeathKind::TaskGone)
        };
        IcbResource::new(Arc::downgrade(&task), descriptor, icb, layout, flags)
    }
}

impl<P: Platform> ResourceManager<P> {
    /// An ICB resource (object type 20) from its record and creation command.
    pub fn create_indirect_command_buffer(
        &self,
        reference: u32,
        entry: &crate::guest::records::ObjectEntry,
        cursor: Cursor,
        placement: Option<crate::guest::GuestPtr>,
    ) -> RaiseResult<Result<Arc<Resource<P>>, String>> {
        let _ = reference;
        create::create_indirect_command_buffer(self, entry, cursor, placement)
    }

    /// Pages the guest's command records at `cursor` into `buffer`'s commands.
    pub fn prepare_indirect_command_buffer(
        &self,
        hold: Hold<'_>,
        buffer: &P::Icb,
        layout: &IndirectCommandLayout,
        flags: &IndirectCommandBufferFlags,
        cursor: Cursor,
    ) -> RaiseResult<Result<(), IcbErrorText>> {
        prepare::prepare(hold, self, buffer, layout, flags, cursor)
    }

    /// Pages `buffer`'s commands back into the guest's command records at `cursor`.
    pub fn synchronize_indirect_command_buffer(
        &self,
        hold: Hold<'_>,
        buffer: &P::Icb,
        layout: &IndirectCommandLayout,
        flags: &IndirectCommandBufferFlags,
        cursor: WritableCursor,
    ) -> RaiseResult<Result<(), String>> {
        synchronize::synchronize(hold, self, buffer, layout, flags, cursor)
    }
}

/// [`ResourceManager::prepare_indirect_command_buffer`] once the task's teardown has dropped
/// the manager: the walk reads the cursor before it needs the manager, so a short cursor is
/// still a refusal, and it dies at its first use of the manager.
pub(crate) fn prepare_indirect_command_buffer_on_null<P: Platform>(
    hold: Hold<'_>,
    buffer: &P::Icb,
    layout: &IndirectCommandLayout,
    flags: &IndirectCommandBufferFlags,
    cursor: Cursor,
) -> RaiseResult<Result<(), IcbErrorText>> {
    prepare::prepare(
        hold,
        &lookup::NoManager::<P>::new(),
        buffer,
        layout,
        flags,
        cursor,
    )
}

/// [`ResourceManager::synchronize_indirect_command_buffer`] with no manager, as the prepare's.
pub(crate) fn synchronize_indirect_command_buffer_on_null<P: Platform>(
    hold: Hold<'_>,
    buffer: &P::Icb,
    layout: &IndirectCommandLayout,
    flags: &IndirectCommandBufferFlags,
    cursor: WritableCursor,
) -> RaiseResult<Result<(), String>> {
    synchronize::synchronize(
        hold,
        &lookup::NoManager::<P>::new(),
        buffer,
        layout,
        flags,
        cursor,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deaths_have_their_kind_and_id() {
        assert_eq!(
            DeathKind::NoManager.site().kind,
            crate::raise::death::DeathKind::NullDeref
        );
        assert_eq!(
            DeathKind::TessellationText.site().kind,
            crate::raise::death::DeathKind::OutOfBounds
        );
        assert_eq!(
            DeathKind::NilRenderPipeline.site().file,
            "manager.icb.nil_render_pipeline"
        );
        assert_eq!(
            DeathId::from(DeathKind::TaskGone).site(),
            DeathKind::TaskGone.site()
        );
    }

    #[test]
    fn a_refusal_keeps_its_bytes() {
        let t = IcbErrorText::with_bytes("a: ", b"\x80\x01");
        assert_eq!(t.as_bytes(), b"a: \x80\x01");
        assert_eq!(t.to_string(), "a: \u{fffd}\u{1}");
        assert_eq!(IcbErrorText::from("x".to_owned()), "x");
    }
}
