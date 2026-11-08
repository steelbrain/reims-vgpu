//! The host's side of the contract: the values it hands over (`types`) and its dispatch queue
//! (`queue`).
//!
//! These are the consumer's (QEMU's) values, not platform APIs: the platform is what core calls
//! on Apple's frameworks, the host is who called core. Every item is at `crate::host::…`.

mod queue;
mod types;

pub use queue::HostQueue;
pub use types::{HostPtr, PhysRange, PhysicalMemoryRange, ZERO_REGION_END};
