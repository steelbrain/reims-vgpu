//! The OS services on Windows, as they are ported: the clocks ([`WindowsClock`]), and the
//! scheduling hint a walk thread gives itself (`hints`).

mod clock;
mod hints;

pub use clock::WindowsClock;
