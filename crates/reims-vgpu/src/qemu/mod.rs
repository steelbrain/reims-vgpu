//! QEMU integration surface: versioned device ABI and HostOps.
//!
//! Every symbol here is one the C shim actually calls. A `cabi` module of
//! standalone decoder/planner entry points used to sit beside these two; it
//! exported 182 `#[no_mangle]` functions that nothing linked, and it is gone.
//! Add an export here only when the shim's link line needs it.

pub mod abi;
pub(crate) mod cstr;
pub mod host_ops;
