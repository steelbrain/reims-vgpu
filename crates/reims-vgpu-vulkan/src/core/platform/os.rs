//! The OS services core uses beside Metal: time and timed waits, threads and serial queues,
//! autorelease pools and Mach VM.
//! Each service is a submodule; this module re-exports their items, so `platform::os::Clock`
//! names the same trait as `platform::os::clock::Clock`.

pub mod autorelease;
pub mod clock;
pub mod threads;
pub mod vm;

pub use autorelease::Autorelease;
pub use clock::{Clock, Deadline, MonoTime, TimedOut, WallTime};
pub use threads::{Threads, TimerClock};
