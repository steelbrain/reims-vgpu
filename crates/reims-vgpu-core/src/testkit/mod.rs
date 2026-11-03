//! The fake platform for unit tests (architecture 13.2). Only under `cfg(any(test, feature =
//! "testkit"))`.
//!
//! - [`platform::FakePlatform`]: every platform trait over small object models ([`objects`],
//!   [`impls`]), recording each call whose position is observable in a [`calllog::CallLog`],
//!   scripted by a [`script::Script`], and answering the free calls from the models;
//! - [`surfaces`], [`vm`], [`guest_ram`]: IOSurfaces with guest-visible retain counts, Mach VM
//!   with remap aliasing and maximum protection, guest RAM, all over real memory;
//! - [`clock::VirtualClock`]: virtual time for core's timed waits;
//! - [`threads`]: a thread recorder;
//! - [`logs`]: a capture of the `log` crate's lines;
//! - [`host`]: the host's callbacks as the device reads them, over fake memory, and a display's
//!   queue;
//! - [`resource_host::FakeResourceHost`]: a scripted task, as resources see it;
//! - [`launcher::ThreadLauncher`]: the isolated tasks' launcher, recording what it launched.

// The testkit is outside core's production lint set.
#![allow(clippy::integer_division_remainder_used, clippy::mod_module_files)]

pub mod calllog;
pub mod capture;
pub mod clock;
pub mod death;
pub mod faults;
pub mod guest_ram;
pub mod host;
pub mod impls;
pub mod launcher;
pub mod logs;
pub mod objects;
pub mod platform;
pub mod resource_host;
pub mod script;
pub mod surfaces;
pub mod threads;
pub mod vm;

pub use calllog::{Arg, Call, CallLog};
pub use clock::VirtualClock;
pub use guest_ram::GuestRam;
pub use objects::{CompleteOn, Fk, World, kind};
pub use platform::FakePlatform;
pub use script::{HeldCall, Policy, Script, Step};
