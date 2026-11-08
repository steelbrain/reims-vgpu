//! The OS services under core's platform boundary: core's threads ([`threads`]), the
//! Objective-C autorelease pool ([`autorelease`]), the host's clocks ([`clock`]), IOSurface
//! ([`iosurface`]) and Mach VM ([`vm`]).

pub mod autorelease;
pub mod clock;
pub mod iosurface;
pub mod threads;
pub mod vm;
