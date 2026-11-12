//! Fake: the `platform/function_table.rs` trait over the fake models (architecture 13.2).
//!
//! A table holds its entries, each the identity of the handle set there or none; a compute
//! pipeline state's `functionHandleWithFunction:` and `newVisibleFunctionTableWithDescriptor:`
//! (`impls/resource.rs`) make them. `setFunctions:withRange:` is logged and scripted, then writes
//! the entries in the table's range.

use super::super::calllog::Arg;
use super::super::objects::{Fk, ResourceCommon, kind};
use super::super::platform::FakePlatform;
use crate::core::platform::function_table::*;
use crate::core::platform::types::NsRange;
use crate::core::platform::{Handle, ObjId, Platform};
use crate::core::raise::RaiseResult;
use crate::core::testkit::script::Policy;

/// A visible function table's model.
#[derive(Debug, Default)]
pub struct FunctionTableModel {
    /// Common resource state (its `gpuResourceID`).
    pub common: ResourceCommon,
    /// Its entries, `functionCount` of them: the handle each calls, or none.
    pub entries: Vec<Option<ObjId>>,
}

impl VisibleFunctionTable<FakePlatform> for Fk<kind::VisibleFunctionTable> {
    fn gpu_resource_id(&self) -> u64 {
        self.gpu_resource_id_value()
    }

    /// Metal's on an Apple M3 Max: 256 bytes and 8 per entry.
    fn allocated_size(&self) -> u64 {
        self.with_model(|m: &FunctionTableModel| {
            let entries = u64::try_from(m.entries.len()).unwrap_or(u64::MAX);
            entries.saturating_mul(8).saturating_add(256)
        })
    }

    /// `Shared`, tracked: Metal's 512 on an Apple M3 Max.
    fn resource_options(&self) -> u64 {
        512
    }

    fn set_functions(
        &self,
        functions: &[Option<&<FakePlatform as Platform>::FunctionHandle>],
        range: NsRange,
    ) -> RaiseResult<()> {
        self.act(
            Policy::Caught,
            "setFunctions:withRange:",
            vec![Arg::objs(functions), Arg::val(&range)],
        )?;
        self.with_model_mut(|m: &mut FunctionTableModel| {
            for (function, at) in functions.iter().zip(range.location..) {
                let entry = usize::try_from(at)
                    .ok()
                    .and_then(|at| m.entries.get_mut(at));
                if let Some(entry) = entry {
                    *entry = function.map(Handle::id);
                }
            }
        });
        Ok(())
    }
}

impl Fk<kind::VisibleFunctionTable> {
    /// The table's entries: the identity of the handle each calls, or none.
    pub fn entries(&self) -> Vec<Option<ObjId>> {
        self.with_model(|m: &FunctionTableModel| m.entries.clone())
    }
}
