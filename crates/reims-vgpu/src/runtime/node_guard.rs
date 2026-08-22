//! Runtime controls for the page-table node write instrument.
//!
//! The backend-independent watch and verdict live in `reims-vgpu-core`.
//! Runtime owns only process configuration and the paging walk bound used by
//! the drain adapter.

pub use reims_vgpu_core::{NodeVerdict, NodeWatch};
pub use reims_vgpu_paging::resolve::MAX_TREE_NODES;

/// Whether the guest-page write guards are enabled for this process.
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::env::switch(crate::env::PAGE_GUARDS) != crate::env::Switch::Off)
}
