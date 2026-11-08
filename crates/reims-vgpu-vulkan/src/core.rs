//! What the reims-graphics device and the platform under it share, generic over one
//! `P: Platform` (D3): the platform traits, the runtime (threads, queues, holds), raises and
//! deaths, the guest's RAM as the host maps it, and the testkit's fake platform, which the tests
//! run in the Vulkan platform's place. Links no Apple framework.

// The module denies `unsafe_code` (D43): `forbid` cannot be lowered by the per-module
// `#![allow(unsafe_code)]` that 17.3 grants `testkit/guest_ram.rs` (E0453).
#![deny(unsafe_code)]
// The module's lint set. Tests are exempt: clippy.toml lets them unwrap, index and panic,
// and the conversion and division lints apply outside cfg(test) only, as clippy has no in-tests
// switch for them. `arithmetic_side_effects` is denied only in the modules guest data flows
// through (their own attributes).
#![deny(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::disallowed_types,
    clippy::disallowed_methods,
    clippy::mod_module_files
)]
#![cfg_attr(
    not(test),
    deny(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_possible_wrap,
        clippy::integer_division_remainder_used,
        clippy::undocumented_unsafe_blocks
    )
)]

pub mod collections;
pub mod display;
pub mod host;
pub mod logging;
pub mod memory;
pub mod num;
pub mod platform;
pub mod raise;
pub mod rt;
pub mod sync;
#[cfg(any(test, feature = "testkit"))]
pub mod testkit;
pub mod trace;
