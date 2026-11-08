//! Locks: `parking_lot`'s, the same types reims-vgpu-core's `sync` re-exports, which every video
//! crate locks with.
//!
//! None is poisoned: a lock a panicking thread held is the next locker's, as it was, so no caller
//! handles poisoning. In the library a panic on its threads ends the process before anything
//! unwinds (architecture 6.5), so no lock is left mid-change for another thread to find. A timed
//! wait on a [`Condvar`] goes through the runtime's clock ([`crate::core::runtime::Bell`]).

pub use parking_lot::{
    Condvar, MappedMutexGuard, Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard,
};
