//! Platform trait `VisibleFunctionTable`: the `MTLVisibleFunctionTable` selectors core sends.
//! A compute pipeline state gives the handles of the functions it links and makes the tables
//! (`ComputePipeline::function_handle`, `ComputePipeline::new_visible_function_table`), and a
//! compute encoder binds a table at a buffer index (`ComputeCmd::SetVisibleFunctionTable`).
//! `id<MTLFunctionHandle>` has no selector core sends, so `Platform::FunctionHandle` is bounded by
//! `Handle` alone.

use super::types::NsRange;
use super::{Handle, Platform};
use crate::core::raise::RaiseResult;

/// `id<MTLVisibleFunctionTable>`.
pub trait VisibleFunctionTable<P: Platform>: Handle {
    /// sel: gpuResourceID
    /// `MTLResourceID._impl`: what an argument buffer holds of the table.
    fn gpu_resource_id(&self) -> u64;
    /// sel: allocatedSize
    fn allocated_size(&self) -> u64;
    /// sel: resourceOptions
    fn resource_options(&self) -> u64;
    /// sel: setFunctions:withRange:
    /// Entry `range.location + i` calls `functions[i]`, or is null (`None`); `functions`' length is
    /// `range.length`. A range past the table's size, or a handle of another pipeline state where
    /// the platform has no answer of Metal's, raises, and sets none of them.
    fn set_functions(
        &self,
        functions: &[Option<&P::FunctionHandle>],
        range: NsRange,
    ) -> RaiseResult<()>;
}
