//! Dma-bufs of host memory lent to a device outside the runtime, which writes into it itself: the
//! host video decoders' GPU, writing a decoded frame into the reims-graphics device's surface views
//! and region blocks with no copy by the CPU ([`dma_buf_of_host`]). What a dma-buf holds is the
//! whole pages ranges of the memory a device names lie in, in their order, made by udmabuf of the
//! files behind them as the runtime's imports are made ([`super::dma_buf::of_ranges`]), in one
//! piece, with the offset of each range's first byte in it. One with more runs of file pages than
//! udmabuf's `list_limit` takes, or more bytes than its `size_limit_mb`, is refused with both
//! limits and its own count ([`LendRefusal::TooLarge`]), so that a device may ask for fewer pages
//! at a time: nearly every page of a guest's buffer is a run of its own, so at the stock
//! `list_limit` of 1,024 one dma-buf holds about 4 MiB of one.

use std::os::fd::OwnedFd;
use std::ptr::NonNull;

use crate::core::num::ToU64 as _;

use super::dma_buf::{self, Refusal, Zeros};
use super::vm;

/// A dma-buf of ranges of host memory ([`dma_buf_of_host`]): the descriptor, its size, and where
/// each range's first byte lies in it.
#[derive(Debug)]
pub struct LentDmaBuf {
    /// The dma-buf.
    pub fd: OwnedFd,
    /// Its size in bytes: whole pages.
    pub size: u64,
    /// Where each range's first byte lies in it, in the ranges' order.
    pub offsets: Vec<u64>,
}

/// Why no dma-buf of ranges of host memory was made ([`dma_buf_of_host`]).
#[derive(Debug)]
pub enum LendRefusal {
    /// More runs of file pages, or more bytes, than one udmabuf takes: `pieces` runs and `bytes`
    /// bytes, where one takes `most_pieces` and `most_bytes`. Ranges of fewer pages may fit.
    TooLarge {
        /// The ranges' runs of file pages.
        pieces: u64,
        /// Their bytes, whole pages.
        bytes: u64,
        /// udmabuf's `list_limit`.
        most_pieces: u64,
        /// udmabuf's `size_limit_mb`, in bytes.
        most_bytes: u64,
        /// udmabuf's refusal, its limits named.
        why: String,
    },
    /// Refused otherwise, with why.
    Refused(String),
}

/// A dma-buf of `ranges` of the `length` bytes from `first`, memory the library mapped from files
/// or the host registered (module docs): the whole pages each range lies in, in their order, a
/// page two of them share given once, in one piece. Each range is an offset from `first` and a
/// length, in ascending order and apart; the pages are resolved through the backing table, so
/// `Err` where a byte of them has no file behind it, as well as udmabuf's refusal, its limits
/// named where they are what refused.
///
/// The caller keeps the `length` bytes from `first` mapped for the call.
pub fn dma_buf_of_host(
    first: NonNull<u8>,
    length: u64,
    ranges: &[(u64, u64)],
) -> Result<LentDmaBuf, LendRefusal> {
    let first = first.as_ptr().addr().to_u64();
    let (pages, offsets) = pages_of(first, length, vm::page_size(), ranges)
        .map_err(|why| LendRefusal::Refused(why.to_owned()))?;
    let size = pages.iter().map(|&(_, length)| length).sum();
    match dma_buf::of_ranges(&pages, &Zeros::default()) {
        Ok(fd) => Ok(LentDmaBuf { fd, size, offsets }),
        Err(
            refusal @ Refusal::Limit {
                pieces,
                bytes,
                limits,
                ..
            },
        ) => Err(LendRefusal::TooLarge {
            pieces,
            bytes,
            most_pieces: limits.pieces,
            most_bytes: limits.bytes,
            why: refusal.to_string(),
        }),
        Err(refusal) => Err(LendRefusal::Refused(refusal.to_string())),
    }
}

/// The pages of some ranges ([`pages_of`]): each run of them a first byte and a length, and where
/// each range's first byte lies in the runs laid end to end.
type RangesPages = (Vec<(u64, u64)>, Vec<u64>);

/// The whole pages of `ranges`, each an offset from `first` and a length, inside the `length`
/// bytes from `first`, in ascending order and apart: each run of pages a first byte and a length,
/// those that share or meet a page merged, and where each range's first byte lies in the runs
/// laid end to end. `Err` says why where the ranges are not so.
fn pages_of(
    first: u64,
    length: u64,
    page: u64,
    ranges: &[(u64, u64)],
) -> Result<RangesPages, &'static str> {
    // Each run of pages as its first byte and its end.
    let mut pages: Vec<(u64, u64)> = Vec::new();
    let mut offsets = Vec::with_capacity(ranges.len());
    // The bytes of the runs before the last.
    let mut before = 0;
    let mut after = 0;
    for &(at, bytes) in ranges {
        let end = at.checked_add(bytes).ok_or("a range past the lent bytes")?;
        if bytes == 0 {
            return Err("an empty range");
        }
        if at < after {
            return Err("ranges out of order or overlapping");
        }
        if end > length {
            return Err("a range past the lent bytes");
        }
        after = end;
        // Inside the lent bytes, whose pages are mapped: neither overflows.
        let (from, to) = (first + at, first + end);
        let (start, stop) = (from - from % page, to.next_multiple_of(page));
        match pages.last_mut() {
            Some(last) if start <= last.1 => last.1 = last.1.max(stop),
            _ => {
                before += pages.last().map_or(0, |&(start, stop)| stop - start);
                pages.push((start, stop));
            }
        }
        let run = pages.last().map_or(start, |&(start, _)| start);
        offsets.push(before + (from - run));
    }
    let pages = pages.into_iter().map(|(start, stop)| (start, stop - start));
    Ok((pages.collect(), offsets))
}

#[cfg(test)]
mod tests {
    use std::os::fd::AsRawFd as _;

    use crate::core::host::PhysicalMemoryRange;
    use crate::core::num::ToUsize as _;
    use crate::core::platform::os::surface::{IoSurfaces as _, Surface as _, SurfaceSpec};
    use crate::core::platform::os::vm::HostVm as _;

    use super::super::surface::LinuxSurfaces;
    use super::super::vm::{LinuxVm, allocate_aliasable};
    use super::*;

    const BGRA: u32 = u32::from_be_bytes(*b"BGRA");

    /// `count` fresh pages a view can be made of, each byte its page's index, and the ranges of
    /// them a guest's surface names, last first, so that its view is of pages out of order.
    fn scattered_pages(count: u64) -> (u64, Vec<PhysicalMemoryRange>) {
        let page = vm::page_size();
        let at = allocate_aliasable(count * page).unwrap().addr().to_u64();
        for i in 0..count {
            // SAFETY: the pages were just mapped, readable and writable.
            unsafe {
                std::ptr::write_bytes(
                    std::ptr::with_exposed_provenance_mut::<u8>((at + i * page).to_usize()),
                    u8::try_from(i).unwrap(),
                    page.to_usize(),
                );
            }
        }
        let ranges = (0..count)
            .rev()
            .map(|i| PhysicalMemoryRange {
                physical_address: at + i * page,
                physical_length: page,
            })
            .collect();
        (at, ranges)
    }

    fn over(ranges: &[PhysicalMemoryRange]) -> SurfaceSpec {
        let page = vm::page_size();
        SurfaceSpec {
            pixel_format: BGRA,
            alloc_size: ranges.len().to_u64() * page,
            offset: 0x40,
            width: 16,
            height: 16,
            bytes_per_element: 4,
            bytes_per_row: 64,
            element_width: 1,
            element_height: 1,
            client_address_ranges: Some(ranges.to_vec()),
            ..SurfaceSpec::default()
        }
    }

    fn read(at: u64) -> u8 {
        // SAFETY: the tests' mapped memory.
        unsafe { std::ptr::with_exposed_provenance::<u8>(at.to_usize()).read_volatile() }
    }

    fn free(at: u64, length: u64) {
        let vm = LinuxVm::new();
        let at = crate::core::host::HostPtr::from_addr(at.to_usize());
        assert!(vm.deallocate(at, length).is_success());
    }

    /// A shared mapping of the whole dma-buf, `f` given its bytes, unmapped after.
    fn through<T>(made: &LentDmaBuf, f: impl FnOnce(&mut [u8]) -> T) -> T {
        let length = made.size.to_usize();
        // SAFETY: a fresh shared mapping of the dma-buf, unmapped below.
        unsafe {
            let at = libc::mmap(
                std::ptr::null_mut(),
                length,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                made.fd.as_raw_fd(),
                0,
            );
            assert_ne!(at, libc::MAP_FAILED);
            let r = f(std::slice::from_raw_parts_mut(at.cast::<u8>(), length));
            libc::munmap(at, length);
            r
        }
    }

    /// A dma-buf of ranges of a surface's view holds the pages each lies in, in order, a page two
    /// share once, and says where each range starts in it: a write there reaches the guest's page
    /// the view shows at that range
    #[test]
    fn a_dma_buf_of_ranges_holds_their_pages_in_order() {
        let page = vm::page_size();
        let (at, ranges) = scattered_pages(4);
        let s = LinuxSurfaces::new().create(&over(&ranges)).unwrap();
        let base = s.base_address().addr();
        let lent = NonNull::new(std::ptr::with_exposed_provenance_mut::<u8>(base)).unwrap();
        let first = base.to_u64() - 0x40;
        // From the surface's base address, 0x40 into its view's first page: a range in its first
        // page, one in its third, and one starting in that third page's last bytes and ending in
        // the fourth.
        let wanted = [
            (0x100, 0x200),
            (2 * page, 0x100),
            (3 * page - 0x40 - 0x40, 0x100),
        ];
        match dma_buf_of_host(lent, 4 * page - 0x40, &wanted) {
            Ok(made) => {
                assert_eq!(made.size, 3 * page);
                assert_eq!(made.offsets, [0x140, page + 0x40, 2 * page - 0x40]);
                let seen = through(&made, |bytes| {
                    bytes[(page + 0x40).to_usize()] = 0x5a;
                    [0, page, 2 * page].map(|at| bytes[at.to_usize()])
                });
                // The view's pages 0, 2 and 3, each its file page's index: 3, 1 and 0.
                assert_eq!(seen, [3, 1, 0]);
                assert_eq!(read(first + 2 * page + 0x40), 0x5a, "the view");
                assert_eq!(read(at + page + 0x40), 0x5a, "the guest's page");
            }
            Err(why) => eprintln!("no dma-buf here, the rest checked: {why:?}"),
        }
        drop(s);
        free(at, 4 * page);
    }

    /// A dma-buf of a view the library mapped holds the pages the view shows there; memory no
    /// file is behind is refused
    #[test]
    fn a_dma_buf_of_host_memory_holds_the_pages_its_mapping_shows() {
        let page = vm::page_size();
        let (at, ranges) = scattered_pages(3);
        let s = LinuxSurfaces::new().create(&over(&ranges)).unwrap();
        let view = s.base_address().addr().to_u64() - 0x40;
        let first =
            NonNull::new(std::ptr::with_exposed_provenance_mut::<u8>(view.to_usize())).unwrap();
        match dma_buf_of_host(first, 3 * page, &[(page + 0x80, 0x100)]) {
            Ok(made) => {
                assert_eq!((made.size, &made.offsets[..]), (page, &[0x80][..]));
                // The view's page 1 shows file page 1, its pages being the guest's last first.
                assert_eq!(through(&made, |bytes| bytes[0]), 1);
            }
            Err(why) => eprintln!("no dma-buf here, the rest checked: {why:?}"),
        }
        // SAFETY: a fresh private anonymous mapping, unmapped below.
        let anonymous = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                page.to_usize(),
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert_ne!(anonymous, libc::MAP_FAILED);
        let plain = NonNull::new(anonymous.cast::<u8>()).unwrap();
        let refused = dma_buf_of_host(plain, page, &[(0, page)]);
        assert!(
            matches!(refused, Err(LendRefusal::Refused(_))),
            "{refused:?}"
        );
        // SAFETY: the mapping made above, which nothing holds.
        unsafe { libc::munmap(anonymous, page.to_usize()) };
        drop(s);
        free(at, 3 * page);
    }

    /// The pages of ranges: merged where two share or meet a page, each range's offset in the
    /// runs end to end; ranges out of order, overlapping, empty or past the bytes refused
    #[test]
    fn ranges_take_their_whole_pages_once() {
        let page = 0x1000;
        let first = 0x10_0040;
        // One range: its pages, its offset into the first.
        assert_eq!(
            pages_of(first, 0x8000, page, &[(0, 0x10)]),
            Ok((vec![(0x10_0000, page)], vec![0x40]))
        );
        // Two in one page, one meeting the page after, one apart.
        assert_eq!(
            pages_of(
                first,
                0x8000,
                page,
                &[(0, 0x10), (0x100, 0x10), (0xfc0, 0x10), (0x4000, 0x20)]
            ),
            Ok((
                vec![(0x10_0000, 2 * page), (0x10_4000, page)],
                vec![0x40, 0x140, 0x1000, 2 * page + 0x40]
            ))
        );
        for bad in [
            &[(0x100, 0x10), (0, 0x10)][..],
            &[(0, 0x20), (0x10, 0x10)],
            &[(0, 0)],
            &[(0x7ff0, 0x20)],
            &[(u64::MAX, 2)],
        ] {
            assert!(pages_of(first, 0x8000, page, bad).is_err(), "{bad:?}");
        }
    }
}
