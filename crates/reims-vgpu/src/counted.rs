//! The unit tests' allocator: the system's, counting the allocations a thread makes while it asks
//! ([`allocations_in`]), so a test holds a path to allocating nothing (a doorbell's answer, on the
//! guest's vCPU, `graphics`).

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

/// The system's allocator, counted.
pub(crate) struct Counted;

thread_local! {
    /// While set, this thread's allocations are counted.
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    /// This thread's allocations counted.
    static COUNT: Cell<u64> = const { Cell::new(0) };
}

/// Counts an allocation of this thread's, where it counts them.
fn count() {
    // `try_with`: a thread's own counters may be gone as it exits.
    let _ = COUNTING.try_with(|on| {
        if on.get() {
            let _ = COUNT.try_with(|n| n.set(n.get() + 1));
        }
    });
}

// SAFETY: each call is the system allocator's, as asked; the count touches only this thread's
// constant thread-locals, which allocate nothing.
unsafe impl GlobalAlloc for Counted {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count();
        // SAFETY: as the caller's.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count();
        // SAFETY: as the caller's.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count();
        // SAFETY: as the caller's.
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: as the caller's.
        unsafe { System.dealloc(ptr, layout) }
    }
}

/// What `f` makes, and how many allocations it made on this thread.
pub(crate) fn allocations_in<R>(f: impl FnOnce() -> R) -> (R, u64) {
    COUNT.with(|n| n.set(0));
    COUNTING.with(|on| on.set(true));
    let made = f();
    COUNTING.with(|on| on.set(false));
    (made, COUNT.with(Cell::get))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The count sees an allocation made where it is asked, and none of another thread's.
    #[test]
    fn an_allocation_is_counted_on_its_thread() {
        let (bytes, made) = allocations_in(|| vec![1u8; 64]);
        assert_eq!((bytes.len(), made), (64, 1));
        let ((), made) = allocations_in(|| {
            std::thread::scope(|scope| {
                scope.spawn(|| drop(vec![1u8; 64]));
            });
        });
        assert!(made < 8, "another thread's allocation counted here: {made}");
        let ((), made) = allocations_in(|| ());
        assert_eq!(made, 0);
    }
}
