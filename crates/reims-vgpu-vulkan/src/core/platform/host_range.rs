//! `HostRange`: host memory the platform may read or keep, for the four methods that take a
//! host pointer.
//!
//! `new_buffer_over_host_memory`, `replace_region`, `set_contents` and `fill_texture` are safe
//! methods that take a `HostRange<'_>`. Its only constructor is `unsafe`: the caller proves the
//! range stays mapped for the call, or, handed to a no-copy creator, until its deallocator runs.

use std::marker::PhantomData;

use crate::core::host::HostPtr;
use crate::core::num::ToU64;

/// `len` bytes of host memory at `ptr`, mapped for `'a`.
///
/// The platform sends `ptr` (and, where the selector takes one, `len`) exactly as given: a range
/// is never checked, clamped or rounded here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostRange<'a> {
    ptr: HostPtr,
    len: u64,
    _mapped: PhantomData<&'a [u8]>,
}

impl<'a> HostRange<'a> {
    /// The range `ptr..ptr + len`.
    ///
    /// # Safety
    /// `ptr..ptr + len` stays mapped, and readable by the platform, for as long as the method it
    /// is passed to may touch it: until the call returns, or, for a no-copy buffer's contents,
    /// until Metal runs the buffer's deallocator (with none, for as long as the buffer lives).
    #[allow(unsafe_code)] // architecture 17.3: the one unsafe constructor of host memory
    pub const unsafe fn new(ptr: HostPtr, len: u64) -> HostRange<'a> {
        HostRange {
            ptr,
            len,
            _mapped: PhantomData,
        }
    }

    /// The bytes of `slice`, which its borrow keeps mapped.
    pub fn of_slice(slice: &'a [u8]) -> HostRange<'a> {
        HostRange {
            ptr: HostPtr::from_ptr(slice.as_ptr().cast_mut()),
            len: slice.len().to_u64(),
            _mapped: PhantomData,
        }
    }

    /// The first byte.
    pub const fn ptr(&self) -> HostPtr {
        self.ptr
    }

    /// The length in bytes.
    pub const fn len(&self) -> u64 {
        self.len
    }

    /// Whether the range is empty.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A range hands back exactly the pointer and length it was made with
    #[test]
    #[allow(unsafe_code)]
    fn keeps_pointer_and_length() {
        let bytes = [0u8; 4];
        let p = HostPtr::from_ptr(bytes.as_ptr().cast_mut());
        // SAFETY: `bytes` outlives the range.
        let r = unsafe { HostRange::new(p, 4) };
        assert_eq!((r.ptr(), r.len(), r.is_empty()), (p, 4, false));
        // SAFETY: an empty range at the maximum address is never read.
        let e = unsafe { HostRange::new(HostPtr::from_addr(usize::MAX), 0) };
        assert!(e.is_empty());
    }
}
