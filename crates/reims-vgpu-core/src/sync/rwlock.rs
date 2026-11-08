//! `WriterPrefRwLock<T>`: a shared/unique lock that lets a queued writer in ahead of new readers,
//! but never makes a thread wait for a read it already holds.
//!
//! A thread's first read waits while a writer is queued, so a stream of readers cannot starve a
//! writer, and `try_write` fails while a reader holds the lock or a writer is queued. A read on
//! a thread that already holds a read of the same lock goes through at once, whatever is
//! queued: the writer is waiting for that thread's first read to end, so making the second wait
//! behind it would hang both (`tests::a_nested_read_passes_a_queued_writer`). Which waiter wins
//! after an unlock is a race.
//!
//! The lock is `parking_lot::RwLock`: a first read is its `read()`, which waits while a writer
//! has set the writer bit, and a nested read its `read_recursive()`, which does not. The thread's
//! reads are counted per lock in a thread-local table; the guards are not `Send`, so a read
//! always ends on the thread that took it.

use std::cell::RefCell;
use std::fmt;
use std::ops::Deref;

/// A shared/unique lock around a `T` (module doc). Most locks guard state that lives in the
/// owner's fields: those are `WriterPrefRwLock<()>`.
#[derive(Default)]
pub struct WriterPrefRwLock<T = ()> {
    inner: parking_lot::RwLock<T>,
}

/// A shared guard.
pub struct ReadGuard<'a, T = ()> {
    guard: parking_lot::RwLockReadGuard<'a, T>,
    /// The lock's key in the thread's table of reads.
    key: usize,
}

/// A unique guard.
pub type WriteGuard<'a, T = ()> = parking_lot::RwLockWriteGuard<'a, T>;

thread_local! {
    /// The locks this thread holds reads of, by address, with how many reads of each.
    static READS: RefCell<Vec<(usize, u32)>> = const { RefCell::new(Vec::new()) };
}

impl<T> WriterPrefRwLock<T> {
    /// A lock around `value`.
    pub const fn new(value: T) -> Self {
        WriterPrefRwLock {
            inner: parking_lot::RwLock::new(value),
        }
    }

    /// The lock's key in the thread's table of reads.
    fn key(&self) -> usize {
        std::ptr::from_ref(self).addr()
    }

    /// A shared hold. The thread's first read of this lock waits while a writer is queued; a
    /// nested one does not (module doc).
    pub fn read(&self) -> ReadGuard<'_, T> {
        let key = self.key();
        let nested = READS.with_borrow(|reads| reads.iter().any(|&(k, _)| k == key));
        let guard = if nested {
            self.inner.read_recursive()
        } else {
            self.inner.read()
        };
        READS.with_borrow_mut(|reads| match reads.iter_mut().find(|(k, _)| *k == key) {
            Some((_, n)) => *n = n.saturating_add(1),
            None => reads.push((key, 1)),
        });
        ReadGuard { guard, key }
    }

    /// A unique hold, once every reader has left.
    pub fn write(&self) -> WriteGuard<'_, T> {
        self.inner.write()
    }

    /// A unique hold if the lock is free and no writer is queued.
    pub fn try_write(&self) -> Option<WriteGuard<'_, T>> {
        self.inner.try_write()
    }
}

impl<T: fmt::Debug> fmt::Debug for WriterPrefRwLock<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.fmt(f)
    }
}

impl<T> Deref for ReadGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.guard
    }
}

impl<T> Drop for ReadGuard<'_, T> {
    /// The thread's count for the lock goes down with the read; the lock is released after.
    fn drop(&mut self) {
        let key = self.key;
        READS.with_borrow_mut(|reads| {
            if let Some(i) = reads.iter().position(|&(k, _)| k == key)
                && let Some((_, n)) = reads.get_mut(i)
            {
                *n = n.saturating_sub(1);
                if *n == 0 {
                    reads.swap_remove(i);
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    /// Waits (real time) until `f` holds; fails the test after 5 s.
    fn wait_for(mut f: impl FnMut() -> bool) {
        for _ in 0..5000 {
            if f() {
                return;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        panic!("condition not reached");
    }

    /// Queues a writer behind the calling thread's read of `l`; returns once it waits, with the
    /// flag it sets when it has written.
    fn queue_writer(l: &Arc<WriterPrefRwLock>) -> (std::thread::JoinHandle<()>, Arc<AtomicBool>) {
        let wrote = Arc::new(AtomicBool::new(false));
        let (l2, w2) = (Arc::clone(l), Arc::clone(&wrote));
        #[allow(clippy::disallowed_methods)] // test thread: no Objective-C runs here (4.5)
        let writer = std::thread::spawn(move || {
            drop(l2.write());
            w2.store(true, Ordering::SeqCst);
        });
        // The writer queues: a timed first read from another thread now waits behind it.
        wait_for(|| {
            let l3 = Arc::clone(l);
            #[allow(clippy::disallowed_methods)] // test thread: no Objective-C runs here (4.5)
            let probe = std::thread::spawn(move || {
                l3.inner.try_read_for(Duration::from_millis(5)).is_none()
            });
            probe.join().unwrap_or(false)
        });
        (writer, wrote)
    }

    /// A writer queued behind a reader blocks other threads' first reads and `try_write`, but
    /// not that reader's second read; it gets the lock once the reader leaves
    #[test]
    fn a_nested_read_passes_a_queued_writer() {
        let l = Arc::new(WriterPrefRwLock::new(()));
        let first = l.read();
        let (writer, wrote) = queue_writer(&l);
        let nested = l.read();
        let again = l.read();
        assert!(l.try_write().is_none());
        drop((again, nested));
        assert!(!wrote.load(Ordering::SeqCst));
        drop(first);
        assert!(writer.join().is_ok());
        assert!(wrote.load(Ordering::SeqCst));
        assert!(l.try_write().is_some());
        // The thread's table is empty again: its next read is a first read.
        assert!(READS.with_borrow(Vec::is_empty));
    }

    /// Once the thread's reads have ended, its next read is a first read and waits behind a
    /// queued writer again
    #[test]
    fn a_read_after_the_last_one_ended_is_a_first_read() {
        let l = Arc::new(WriterPrefRwLock::new(()));
        let holder = Arc::new(WriterPrefRwLock::new(()));
        drop(l.read());
        // Another thread holds a read; the writer queues behind it.
        let (release, released) = std::sync::mpsc::channel::<()>();
        let (held, is_held) = std::sync::mpsc::channel::<()>();
        let l2 = Arc::clone(&l);
        #[allow(clippy::disallowed_methods)] // test thread: no Objective-C runs here (4.5)
        let reader = std::thread::spawn(move || {
            let _g = l2.read();
            let _ = held.send(());
            let _ = released.recv();
        });
        is_held.recv().unwrap();
        let (writer, wrote) = queue_writer(&l);
        // This thread holds no read of `l`: its read waits behind the writer.
        assert!(l.inner.try_read_for(Duration::from_millis(50)).is_none());
        // A read of another lock does not count for `l`.
        let _other = holder.read();
        assert!(READS.with_borrow(|r| r.iter().all(|&(k, _)| k != l.key())));
        drop(release);
        assert!(reader.join().is_ok());
        assert!(writer.join().is_ok());
        assert!(wrote.load(Ordering::SeqCst));
    }

    /// Readers share; with no writer queued a nested read goes through
    #[test]
    fn readers_share() {
        let l = WriterPrefRwLock::new(vec![1u32]);
        {
            let (a, b) = (l.read(), l.read());
            assert_eq!((a.len(), b.len()), (1, 1));
            assert!(l.try_write().is_none());
        }
        l.write().push(2);
        assert_eq!(*l.read(), vec![1, 2]);
    }
}
