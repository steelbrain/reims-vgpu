//! The copy of the guest's bytes a write of the platform's own takes ([`first_bytes`]).

use crate::answer::unrepresentable;
use crate::core::platform::HostRange;

/// A copy of the first `most` bytes of `range` (all of them where it holds fewer), for a write of
/// this platform's own that takes a slice, whose checks of where the range lies take all of it. A
/// non-empty range at a NULL or near-NULL pointer, in the zero region (`ZERO_REGION_END`), or one
/// that wraps past the top of the address space, dies at `sel`, as Metal's read of it faults:
/// core's fill of a texture with bytes reads a buffer's `contents` plus an offset, NULL plus the
/// offset for a buffer made on a surface, and Metal's `fillTexture:level:slice:region:bytes:
/// length:` ends the process with a segmentation fault there (measured on the M3 Max at offsets
/// 0, 0x10 and 0x4000, on surfaces of their own memory and over the guest's pages).
///
/// The bytes are guest memory, which another vCPU or GPU work in flight may write while they are
/// read: `HostRange` promises them mapped and readable for the call, never unchanged, so no
/// shared slice is made over them. They are copied once, through the pointer, into memory this
/// platform owns.
pub(crate) fn first_bytes(sel: &'static str, range: HostRange<'_>, most: usize) -> Vec<u8> {
    let length =
        usize::try_from(range.len()).unwrap_or_else(|_| unrepresentable(sel, "that length"));
    if length == 0 {
        return Vec::new();
    }
    let start = range.ptr().as_ptr::<u8>();
    let first = range.ptr().addr();
    if first < crate::core::host::ZERO_REGION_END || first.checked_add(length).is_none() {
        crate::answer::metal_reads_near_null(sel, &format!("{length} bytes at {first:#x}"));
    }
    let read = length.min(most);
    let mut copy = Vec::with_capacity(read);
    // SAFETY: `HostRange::new`'s contract keeps `start..start + length` mapped and readable for
    // the call, and `read` is at most `length`; `copy` owns `read` bytes of its own, which no
    // other pointer reaches, so the two do not overlap. The copy is made through raw pointers
    // alone: no reference ever covers the guest's bytes, which may change under it. `set_len`
    // follows the copy, which wrote every one of the `read` bytes.
    unsafe {
        std::ptr::copy_nonoverlapping(start, copy.as_mut_ptr(), read);
        copy.set_len(read);
    }
    copy
}
