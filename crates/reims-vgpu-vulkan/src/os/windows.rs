//! The OS services on Windows, as they are ported: the clocks ([`WindowsClock`]), threads
//! ([`WindowsThreads`]), and the scheduling hint a walk thread gives itself (`hints`).

mod clock;
mod hints;
mod threads;

pub use clock::WindowsClock;
pub use threads::WindowsThreads;
