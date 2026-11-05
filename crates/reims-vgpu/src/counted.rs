//! The unit tests' allocator: the system's, counting the allocations and the frees a thread makes
//! while it asks ([`allocator_calls_in`]), so a test holds a path to calling the allocator not at
//! all (a doorbell's answer, on the guest's vCPU, `graphics`): jemalloc, the library's allocator,
//! may purge pages inside either call. It also counts every thread's allocations of one size
//! ([`allocations_of`]), so a test counts the copies the device's own threads make of a message
//! of that length.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// The system's allocator, counted.
pub(crate) struct Counted;

thread_local! {
    /// While set, this thread's allocations are counted.
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    /// This thread's allocations counted.
    static COUNT: Cell<u64> = const { Cell::new(0) };
    /// This thread's frees counted.
    static FREES: Cell<u64> = const { Cell::new(0) };
}

/// The size [`allocations_of`] counts, on every thread; 0 while it counts none.
static SIZED: AtomicUsize = AtomicUsize::new(0);
/// The allocations of that size counted.
static SIZED_COUNT: AtomicU64 = AtomicU64::new(0);

/// Counts an allocation of `size` bytes: this thread's, where it counts them, and any thread's of
/// the size [`allocations_of`] counts.
fn count(size: usize) {
    bump(&COUNT);
    if size != 0 && SIZED.load(Ordering::Relaxed) == size {
        SIZED_COUNT.fetch_add(1, Ordering::Relaxed);
    }
}

/// Counts a free of this thread's, where it counts them.
fn count_free() {
    bump(&FREES);
}

/// Adds one to this thread's `counter`, where it counts.
fn bump(counter: &'static std::thread::LocalKey<Cell<u64>>) {
    // `try_with`: a thread's own counters may be gone as it exits.
    let _ = COUNTING.try_with(|on| {
        if on.get() {
            let _ = counter.try_with(|n| n.set(n.get() + 1));
        }
    });
}

// SAFETY: each call is the system allocator's, as asked; the count touches only this thread's
// constant thread-locals, which allocate nothing.
unsafe impl GlobalAlloc for Counted {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count(layout.size());
        // SAFETY: as the caller's.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count(layout.size());
        // SAFETY: as the caller's.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count(new_size);
        count_free();
        // SAFETY: as the caller's.
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        count_free();
        // SAFETY: as the caller's.
        unsafe { System.dealloc(ptr, layout) }
    }
}

/// What `f` makes, and how many allocations and how many frees it made on this thread (a
/// reallocation is one of each).
pub(crate) fn allocator_calls_in<R>(f: impl FnOnce() -> R) -> (R, (u64, u64)) {
    COUNT.with(|n| n.set(0));
    FREES.with(|n| n.set(0));
    COUNTING.with(|on| on.set(true));
    let made = f();
    COUNTING.with(|on| on.set(false));
    (made, (COUNT.with(Cell::get), FREES.with(Cell::get)))
}

/// What `f` makes, and how many allocations of exactly `size` bytes (not 0) every thread made
/// while it ran: the copies of a message of that length the device's threads make for it, where
/// nothing else allocates that size. One count at a time.
pub(crate) fn allocations_of<R>(size: usize, f: impl FnOnce() -> R) -> (R, u64) {
    assert_ne!(size, 0);
    SIZED_COUNT.store(0, Ordering::SeqCst);
    SIZED.store(size, Ordering::SeqCst);
    let made = f();
    SIZED.store(0, Ordering::SeqCst);
    (made, SIZED_COUNT.load(Ordering::SeqCst))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The count sees an allocation made where it is asked, and none of another thread's.
    #[test]
    fn an_allocation_is_counted_on_its_thread() {
        let (bytes, (made, _)) = allocator_calls_in(|| vec![1u8; 64]);
        assert_eq!((bytes.len(), made), (64, 1));
        let ((), (made, _)) = allocator_calls_in(|| {
            std::thread::scope(|scope| {
                scope.spawn(|| drop(vec![1u8; 64]));
            });
        });
        assert!(made < 8, "another thread's allocation counted here: {made}");
        let ((), (made, _)) = allocator_calls_in(|| ());
        assert_eq!(made, 0);
    }

    /// A free is counted where it is made, apart from the allocations, and a reallocation is one
    /// of each.
    #[test]
    fn a_free_is_counted_on_its_thread() {
        let held = vec![1u8; 64];
        let ((), calls) = allocator_calls_in(|| drop(held));
        assert_eq!(calls, (0, 1));
        let mut grown = Vec::<u8>::with_capacity(8);
        grown.push(1);
        let ((), calls) = allocator_calls_in(|| grown.reserve_exact(4096));
        assert_eq!(calls, (1, 1));
        let ((), calls) = allocator_calls_in(|| ());
        assert_eq!(calls, (0, 0));
    }
}
