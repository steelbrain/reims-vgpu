//! The OS services on Windows, as they are ported: the clocks ([`WindowsClock`]).

mod clock;

pub use clock::WindowsClock;
