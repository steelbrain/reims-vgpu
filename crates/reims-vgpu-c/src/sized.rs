//! Reading a struct the host wrote, headed by `uint32_t struct_size` (docs/api.md section 1):
//! - a size below the struct's is `RVG_ERR_INVALID_ARGUMENT`;
//! - fields past `struct_size` read as zero;
//! - a size above the library's is accepted when every byte past the library's size is zero,
//!   and is `RVG_ERR_UNSUPPORTED` otherwise.

use std::mem::MaybeUninit;

use crate::status::{RVG_ERR_INVALID_ARGUMENT, RVG_ERR_UNSUPPORTED, RvgStatus};

/// A struct the host writes, whose first field is `struct_size: u32`.
///
/// # Safety
/// The struct is `#[repr(C)]`, starts with a `u32`, and every field is valid all-zero (integers,
/// floats, `bool`, raw pointers, `Option` of a function pointer).
pub(crate) unsafe trait HostStruct: Copy {
    /// The size of the struct's first version: the library's for every struct of v1.
    const V1_SIZE: usize = size_of::<Self>();
}

/// Reads `*p`: `Ok(None)` for NULL, else the struct as the library knows it.
///
/// # Safety
/// `p` is NULL or points at `struct_size` readable bytes (at least 4) that start the struct.
pub(crate) unsafe fn read<T: HostStruct>(p: *const T) -> Result<Option<T>, RvgStatus> {
    if p.is_null() {
        return Ok(None);
    }
    // SAFETY: the caller's contract: the struct starts with its `u32` size.
    let size = unsafe { p.cast::<u32>().read() };
    let size = usize::try_from(size).map_err(|_| RVG_ERR_UNSUPPORTED)?;
    if size < T::V1_SIZE {
        return Err(RVG_ERR_INVALID_ARGUMENT);
    }
    let ours = size_of::<T>();
    if size > ours {
        // SAFETY: the caller's contract: `size` bytes are readable.
        let past = unsafe { std::slice::from_raw_parts(p.cast::<u8>().add(ours), size - ours) };
        if past.iter().any(|&b| b != 0) {
            return Err(RVG_ERR_UNSUPPORTED);
        }
    }
    let mut value = MaybeUninit::<T>::zeroed();
    // SAFETY: `size.min(ours)` bytes are readable at `p` and writable in `value`; they do not
    // overlap.
    unsafe {
        std::ptr::copy_nonoverlapping(
            p.cast::<u8>(),
            value.as_mut_ptr().cast::<u8>(),
            size.min(ours),
        );
    }
    // SAFETY: `HostStruct`'s contract: all-zero is valid, and the bytes copied over it are the
    // host's fields of the same layout.
    Ok(Some(unsafe { value.assume_init() }))
}

/// Reads `*p`, where NULL is the struct with every field zero (a callback table with every
/// callback absent).
///
/// # Safety
/// As [`read`].
pub(crate) unsafe fn read_or_zero<T: HostStruct>(p: *const T) -> Result<T, RvgStatus> {
    // SAFETY: the caller's contract.
    let read = unsafe { read(p) }?;
    // SAFETY: `HostStruct`'s contract: all-zero is valid.
    Ok(read.unwrap_or_else(|| unsafe { MaybeUninit::<T>::zeroed().assume_init() }))
}

/// A `struct_size` for a struct the library writes: its size, which is below `u32::MAX`.
pub(crate) fn size_u32<T>() -> u32 {
    u32::try_from(size_of::<T>()).unwrap_or(u32::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Two {
        struct_size: u32,
        a: u32,
        b: u64,
    }
    // SAFETY: repr(C), a u32 first, integers only.
    unsafe impl HostStruct for Two {
        const V1_SIZE: usize = 8;
    }

    /// NULL, too small, a shorter known version, the library's own, and a longer one with zero
    /// and non-zero bytes past the library's size
    #[test]
    fn sizes() {
        // SAFETY: NULL is accepted.
        assert_eq!(unsafe { read::<Two>(std::ptr::null()) }, Ok(None));
        let zero = Two {
            struct_size: 0,
            a: 0,
            b: 0,
        };
        // SAFETY: NULL is the zero struct.
        assert_eq!(unsafe { read_or_zero::<Two>(std::ptr::null()) }, Ok(zero));

        let mut words = [0u64; 4];
        let p = words.as_mut_ptr().cast::<u8>();
        let put = |off: usize, v: u32| {
            // SAFETY: inside `words`.
            unsafe { p.add(off).cast::<u32>().write_unaligned(v) }
        };
        put(0, 4);
        // SAFETY: four readable bytes.
        assert_eq!(
            unsafe { read(p.cast::<Two>()) },
            Err(RVG_ERR_INVALID_ARGUMENT)
        );
        put(0, 8);
        put(4, 7);
        put(8, 9);
        // SAFETY: eight readable bytes; `b` past them reads as zero.
        let v = unsafe { read(p.cast::<Two>()) }.unwrap().unwrap();
        assert_eq!((v.a, v.b), (7, 0));
        put(0, 16);
        // SAFETY: the whole struct.
        let v = unsafe { read(p.cast::<Two>()) }.unwrap().unwrap();
        assert_eq!((v.a, v.b), (7, 9));
        put(0, 32);
        // SAFETY: 32 readable bytes, zero past 16.
        assert!(unsafe { read(p.cast::<Two>()) }.is_ok());
        put(20, 1);
        // SAFETY: as above, with a non-zero byte past 16.
        assert_eq!(unsafe { read(p.cast::<Two>()) }, Err(RVG_ERR_UNSUPPORTED));
        assert_eq!(size_u32::<Two>(), 16);
    }
}
