//! The Vulkan platform's function handles and visible function tables (`MTLFunctionHandle`,
//! `MTLVisibleFunctionTable`): the runtime's (metal2vulkan's `function_table.rs`). A compute
//! pipeline state gives the handle of a function its kernel links, numbered as the runtime's
//! pipeline numbers its linked functions, and makes the tables that take them; a table is a
//! `Shared` buffer of words the runtime writes through its mapping, so an entry set after a
//! dispatch was encoded and before its commit is the one the dispatch calls, as on Metal, and an
//! argument buffer holds a table as its `gpuResourceID`, read with no `useResource`.

use crate::answer::{Nil, Policy, create, send};
use crate::core::platform::function_table::VisibleFunctionTable;
use crate::core::platform::types::NsRange;
use crate::core::raise::{Raise, RaiseResult};
use crate::gpu::listed;
use crate::platform::{Obj, VulkanPlatform};
use metal2vulkan_runtime as rt;

/// `id<MTLFunctionHandle>`.
pub type VulkanFunctionHandle = Obj<rt::FunctionHandle>;

/// `id<MTLVisibleFunctionTable>`.
pub type VulkanVisibleFunctionTable = Obj<rt::VisibleFunctionTable>;

/// `functionHandleWithFunction:` of `state`: the runtime's handle, `None` where Metal answers nil
/// (a function the kernel does not link, a private one, the kernel itself). A function of another
/// device, which no guest names, dies.
pub(crate) fn compute_handle(
    state: &rt::ComputePipelineState,
    function: &rt::Function,
) -> RaiseResult<Option<VulkanFunctionHandle>> {
    let handle = send("functionHandleWithFunction:", Policy::Dies, || {
        state.function_handle(function)
    })?;
    Ok(handle.map(Obj::new))
}

/// `newVisibleFunctionTableWithDescriptor:` of `state`, of `function_count` null entries: nil
/// where the runtime has no memory for it; a table whose words one buffer of the device cannot
/// hold is the runtime's gap, relayed as nil.
pub(crate) fn compute_table(
    state: &rt::ComputePipelineState,
    function_count: u64,
) -> RaiseResult<Option<VulkanVisibleFunctionTable>> {
    let descriptor = rt::VisibleFunctionTableDescriptor::new(function_count);
    let table = create(
        "newVisibleFunctionTableWithDescriptor:",
        Policy::Dies,
        Nil::Memory,
        || state.new_visible_function_table(&descriptor),
    )?;
    Ok(table.map(Obj::new))
}

/// The runtime's table. A range past it, or a handle of another state the runtime has no answer
/// of Metal's for, which Metal's validation stops, raises, and sets none.
impl VisibleFunctionTable<VulkanPlatform> for VulkanVisibleFunctionTable {
    fn gpu_resource_id(&self) -> u64 {
        crate::answer::plain(|| rt::VisibleFunctionTable::gpu_resource_id(self)).0
    }

    fn allocated_size(&self) -> u64 {
        crate::answer::plain(|| rt::VisibleFunctionTable::allocated_size(self))
    }

    fn resource_options(&self) -> u64 {
        crate::answer::plain(|| rt::VisibleFunctionTable::resource_options(self)).raw()
    }

    fn set_functions(
        &self,
        functions: &[Option<&VulkanFunctionHandle>],
        range: NsRange,
    ) -> RaiseResult<()> {
        let sel = "setFunctions:withRange:";
        // A table holds at most `u32::MAX` entries, so a range from past that is past it.
        let Ok(first) = u32::try_from(range.location) else {
            return Err(Raise::platform(sel)
                .with_detail(format!("the range {range:?} is past every table")));
        };
        listed(
            functions,
            |f| f.map(|f| &**f),
            |handles| {
                send(sel, Policy::Caught, || {
                    rt::VisibleFunctionTable::set_functions(self, handles, first)
                })
            },
        )
    }
}
