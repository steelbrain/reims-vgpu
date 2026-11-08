//! Opaque handles.
//!
//! Every `Rvg<Name> *` the API hands out is `Arc::into_raw` of an `Arc<Rvg<Name>>`: one
//! reference per pointer a create returns. A release drops one, and the release that drops the
//! last one ends the object, which may block. Releasing NULL does nothing.

use std::sync::Arc;

/// A new handle holding one reference.
pub(crate) fn new<T>(value: T) -> *mut T {
    Arc::into_raw(Arc::new(value)).cast_mut()
}

/// One reference less; the last runs the value's drop (NULL does nothing).
///
/// # Safety
/// `p` is NULL or a live handle of type `T` from this crate, whose reference the caller owns.
pub(crate) unsafe fn release<T>(p: *mut T) {
    if !p.is_null() {
        // SAFETY: the caller's contract: this gives back one reference `Arc::into_raw` made.
        drop(unsafe { Arc::from_raw(p.cast_const()) });
    }
}

/// The value behind a borrowed handle; `None` for NULL.
///
/// # Safety
/// `p` is NULL or a live handle of type `T` from this crate, live for `'a`.
pub(crate) unsafe fn get<'a, T>(p: *const T) -> Option<&'a T> {
    // SAFETY: the caller's contract.
    unsafe { p.as_ref() }
}

/// Writes `value` through `out` when `out` is not NULL (an optional out-parameter).
///
/// # Safety
/// `out` is NULL or valid for a write of `T`.
pub(crate) unsafe fn write_out<T>(out: *mut T, value: T) {
    if !out.is_null() {
        // SAFETY: the caller's contract.
        unsafe { out.write(value) };
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    static DROPS: AtomicUsize = AtomicUsize::new(0);

    struct Counted(u32);
    impl Drop for Counted {
        fn drop(&mut self) {
            DROPS.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// One reference per pointer, the release drops it, NULL does nothing
    #[test]
    fn refcount() {
        let before = DROPS.load(Ordering::SeqCst);
        let p = new(Counted(7));
        // SAFETY: `p` is live until its release, which gives back its one reference.
        unsafe {
            assert_eq!(get(p).unwrap().0, 7);
            release(p);
        }
        assert_eq!(DROPS.load(Ordering::SeqCst), before + 1);
        let null: *mut Counted = std::ptr::null_mut();
        // SAFETY: NULL is accepted by each.
        unsafe {
            release(null);
            assert!(get(null).is_none());
        }
    }
}
