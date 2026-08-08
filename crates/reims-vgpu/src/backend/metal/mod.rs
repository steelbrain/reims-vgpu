//! Direct host-Metal backend: pure-Rust Metal encode driven from `runtime/`.
//!
//! macOS only. `backend-metal` on any other target is rejected by the
//! `compile_error!` in `lib.rs`, so there is no non-Apple arm of this module
//! and every `target_os = "macos"` gate below is a statement of that fact
//! rather than a branch.

pub mod abi;
mod constants;
pub mod error;

// ---------------------------------------------------------------------------
// Apple: real Metal encode
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
mod cache;
/// Live entries in each compiled-object cache. See
/// [`cache::cache_levels`] — re-exported because `cache` is private to this
/// module and the census that publishes the reading is in `runtime::drain`.
#[cfg(target_os = "macos")]
pub(crate) use cache::cache_levels;
#[cfg(target_os = "macos")]
pub(crate) mod compute;
#[cfg(target_os = "macos")]
mod device;
#[cfg(target_os = "macos")]
pub(crate) mod format;
#[cfg(target_os = "macos")]
mod function;
#[cfg(target_os = "macos")]
pub(crate) mod mipmap;
#[cfg(target_os = "macos")]
pub(crate) mod mtl_enum;
#[cfg(target_os = "macos")]
pub(crate) mod raw_metal;
#[cfg(target_os = "macos")]
pub(crate) mod render;
#[cfg(target_os = "macos")]
pub(crate) mod runtime;
#[cfg(target_os = "macos")]
pub(crate) mod samplers;
#[cfg(target_os = "macos")]
mod stage_input;
#[cfg(target_os = "macos")]
pub(crate) mod util;

#[cfg(target_os = "macos")]
pub(crate) use device::MetalBackend;
