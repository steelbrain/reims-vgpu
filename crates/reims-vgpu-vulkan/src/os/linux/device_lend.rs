//! A surface's memory lent to a device outside the runtime, which writes into it itself: the video
//! device's decoders' GPU, writing a decoded frame into a guest's pixel buffer with no copy by the
//! CPU. What is lent is the bytes [`crate::os::lend_surface_bytes`] lends the CPU, from the
//! surface's base address to its allocation's end, in the two forms a device takes:
//! - **a dma-buf** of ranges of them a device names, the whole pages each lies in, in their
//!   order, made by udmabuf of the files behind them as the runtime's imports are made
//!   ([`super::dma_buf::of_ranges`]), in one piece, with the offset of each range's first byte in
//!   it. One with more runs of file pages than udmabuf's `list_limit` takes, or more bytes than
//!   its `size_limit_mb`, is refused with both limits and its own count
//!   ([`LendRefusal::TooLarge`]), so that a device may ask for fewer pages at a time: nearly every
//!   page of a guest's buffer is a run of its own, so at the stock `list_limit` of 1,024 one
//!   dma-buf holds about 4 MiB of one;
//! - **their host addresses**: the surface's own view of the guest's pages, the one run of them
//!   it lies in, or its own memory, kept mapped by holds on the surface's memory
//!   ([`super::surface::MemoryHolder`]) while the lend, or any hold it made, lives.
//!
//! A lend keeps no reference of the surface: its holder does that (the video device's
//! `MappedSurface`, a hold the guest sees, so the guest's unmap of it is Busy and its pages stay
//! as they are while the device writes them). A hold a lend makes keeps only the memory mapped,
//! as a runtime import's does, so it may outlive the surface: a device's registration of the
//! addresses keeps one until it is dropped. Refused as `lend_surface_bytes` refuses: a protected
//! surface, whose memory the CPU never reaches, and one the guest made read-only for its clients,
//! whose view is read-only and whose pages a device must never write.

use std::os::fd::OwnedFd;
use std::ptr::NonNull;

use reims_vgpu_core::num::{ToU64 as _, ToUsize as _};
use reims_vgpu_core::platform::os::surface::Surface as _;

use super::dma_buf::{self, Refusal, Zeros};
use super::surface::{LinuxSurface, MemoryHolder, recorded_mut};
use super::vm;

/// A surface's bytes lent to a device (module docs).
#[derive(Debug)]
pub struct DeviceLend {
    /// The host address of the first byte, the surface's base address.
    first: NonNull<u8>,
    /// The bytes, to the allocation's end.
    length: u64,
    /// The surface's memory, held mapped while the lend lives.
    holder: MemoryHolder,
}

// SAFETY: `first` is an address in the memory `holder` keeps mapped, which nothing here reads or
// writes through; a device given it writes there only while the surface's holder holds it.
unsafe impl Send for DeviceLend {}
// SAFETY: as above.
unsafe impl Sync for DeviceLend {}

/// A dma-buf of ranges of a lend's bytes ([`DeviceLend::dma_buf_of`]): the descriptor, its size,
/// and where each range's first byte lies in it.
#[derive(Debug)]
pub struct LentDmaBuf {
    /// The dma-buf.
    pub fd: OwnedFd,
    /// Its size in bytes: whole pages.
    pub size: u64,
    /// Where each range's first byte lies in it, in the ranges' order.
    pub offsets: Vec<u64>,
}

/// Why no dma-buf of a lend's ranges was made ([`DeviceLend::dma_buf_of`]).
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

/// `surface`'s bytes lent to a device (module docs); `Err` with why where they are not lent.
pub fn lend_to_devices(surface: &LinuxSurface) -> Result<DeviceLend, &'static str> {
    let base = surface.base_address().addr().to_u64();
    if surface.protection_options() != 0 || base == 0 {
        return Err("a protected surface, whose memory the CPU never reaches");
    }
    if crate::os::gpu_read_only(surface) {
        return Err("a surface the guest made read-only for its clients");
    }
    let offset = recorded_mut(surface, |made| made.offset).unwrap_or(0);
    let length = surface.alloc_size().saturating_sub(offset);
    if length == 0 {
        return Err("a surface with no bytes past its offset");
    }
    let first = NonNull::new(std::ptr::with_exposed_provenance_mut::<u8>(base.to_usize()))
        .ok_or("a protected surface, whose memory the CPU never reaches")?;
    Ok(DeviceLend {
        first,
        length,
        holder: MemoryHolder::of(surface),
    })
}

impl DeviceLend {
    /// The host address of the first byte lent, the surface's base address, mapped while the
    /// lend or a hold it made lives.
    pub fn first(&self) -> NonNull<u8> {
        self.first
    }

    /// The bytes lent, from [`Self::first`] to the allocation's end.
    pub fn length(&self) -> u64 {
        self.length
    }

    /// A new hold on the surface's memory, which keeps [`Self::first`]'s bytes mapped while it
    /// lives, the surface and the lend gone or not, and holds no reference of the surface. The
    /// bytes are mapped from their files short of the headroom of the views' budget, as a device
    /// given their address pins them as they are mapped; past it they stay as they are, the device
    /// refuses to register them, and the session copies the frame on the CPU instead
    /// (`super::import::pages_from_files`).
    pub fn hold(&self) -> Box<dyn Send + Sync> {
        let pinned = super::import::pages_from_files(self.first, self.length);
        Box::new((self.holder.hold(), pinned))
    }

    /// The lent bytes' whole pages: the host address of the first, their bytes, and where
    /// [`Self::first`] lies in them. They are mapped, contiguous, while the lend or a hold it made
    /// lives: the surface's view, the run of the guest's pages it lies in, or its own memory, each
    /// whole pages from below its base address past its allocation's end.
    pub fn pages(&self) -> (NonNull<u8>, u64, u64) {
        let (start, length) = self.page_range();
        let offset = self.first.as_ptr().addr().to_u64() - start;
        // SAFETY: `start` is `first` less its offset into its page, in the same mapping.
        let at = unsafe { NonNull::new_unchecked(self.first.as_ptr().sub(offset.to_usize())) };
        (at, length, offset)
    }

    /// The first byte and the length of the lent bytes' whole pages.
    fn page_range(&self) -> (u64, u64) {
        let page = vm::page_size();
        let first = self.first.as_ptr().addr().to_u64();
        let start = first - first % page;
        // The lent bytes end in the surface's mapping, which ends on a page, so this neither
        // overflows nor passes it.
        let end = (first + self.length).next_multiple_of(page);
        (start, end - start)
    }

    /// A dma-buf of `ranges` of the lent bytes, each an offset from [`Self::first`] and a length,
    /// in ascending order and apart: the whole pages each lies in, in their order, a page two of
    /// them share given once, in one piece (module docs). `Err` with udmabuf's refusal, its limits
    /// named where they are what refused.
    pub fn dma_buf_of(&self, ranges: &[(u64, u64)]) -> Result<LentDmaBuf, LendRefusal> {
        dma_buf_of_host(self.first, self.length, ranges)
    }
}

/// A dma-buf of `ranges` of the `length` bytes from `first`, as [`DeviceLend::dma_buf_of`] makes
/// one of a lend's: for memory the library mapped from files, or the host registered, that is
/// lent to a device other than through a surface (the reims-graphics device's surface views and
/// region blocks). Each range is an offset from `first` and a length, in ascending order and
/// apart; the pages are resolved through the backing table, so `Err` where a byte of them has no
/// file behind it, as well as udmabuf's refusal, its limits named where they are what refused.
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
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    use reims_vgpu_core::host::PhysicalMemoryRange;
    use reims_vgpu_core::platform::Handle as _;
    use reims_vgpu_core::platform::os::surface::{IoSurfaces as _, SurfaceSpec};
    use reims_vgpu_core::platform::os::vm::{HostAddr, HostVm as _};

    use super::super::surface::{LinuxSurfaces, attachment};
    use super::super::vm::tests::mapped;
    use super::super::vm::{LinuxVm, allocate_aliasable};
    use super::*;
    use crate::os::READ_ONLY_DIRECTION;

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

    fn over(ranges: &[PhysicalMemoryRange], direction: Option<u32>) -> SurfaceSpec {
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
            client_direction: direction,
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
        let at = reims_vgpu_core::host::HostPtr::from_addr(at.to_usize());
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

    /// A surface over scattered guest pages lends its bytes from its base address to its
    /// allocation's end, as `lend_surface_bytes` lends them: its whole pages are its view, and
    /// its dma-buf holds the same pages in the same order, the first byte at the same offset, so
    /// a write through the dma-buf reaches the guest's page the view shows there
    #[test]
    fn a_lend_is_the_bytes_the_cpu_is_lent_in_their_order() {
        let page = vm::page_size();
        let (at, ranges) = scattered_pages(3);
        let s = LinuxSurfaces::new().create(&over(&ranges, None)).unwrap();
        let base = s.base_address().addr().to_u64();
        let lend = lend_to_devices(&s).unwrap();
        assert_eq!(lend.first().as_ptr().addr().to_u64(), base);
        assert_eq!(lend.length(), 3 * page - 0x40);
        let (first, size, offset) = lend.pages();
        let first = first.as_ptr().addr().to_u64();
        assert_eq!((first, size, offset), (base - 0x40, 3 * page, 0x40));
        assert_eq!(
            [read(first), read(first + page), read(first + 2 * page)],
            [2, 1, 0]
        );
        match lend.dma_buf_of(&[(0, lend.length())]) {
            Ok(made) => {
                assert_eq!((made.size, &made.offsets[..]), (3 * page, &[0x40][..]));
                let seen = through(&made, |bytes| {
                    bytes[(page + 0x10).to_usize()] = 0x5a;
                    [0, page, 2 * page].map(|at| bytes[at.to_usize()])
                });
                assert_eq!(seen, [2, 1, 0]);
                assert_eq!(read(first + page + 0x10), 0x5a, "the view");
                assert_eq!(read(at + page + 0x10), 0x5a, "the guest's page");
            }
            Err(why) => eprintln!("no dma-buf here, the rest checked: {why:?}"),
        }
        drop((lend, s));
        free(at, 3 * page);
    }

    /// A dma-buf of ranges of a lend holds the pages each lies in, in order, a page two share
    /// once, and says where each range starts in it: a write there reaches the guest's page the
    /// view shows at that range
    #[test]
    fn a_dma_buf_of_ranges_holds_their_pages_in_order() {
        let page = vm::page_size();
        let (at, ranges) = scattered_pages(4);
        let s = LinuxSurfaces::new().create(&over(&ranges, None)).unwrap();
        let lend = lend_to_devices(&s).unwrap();
        let (first, ..) = lend.pages();
        let first = first.as_ptr().addr().to_u64();
        // From the lend's first byte, 0x40 into its first page: a range in its first page, one
        // in its third, and one starting in that third page's last bytes and ending in the fourth.
        let wanted = [
            (0x100, 0x200),
            (2 * page, 0x100),
            (3 * page - 0x40 - 0x40, 0x100),
        ];
        match lend.dma_buf_of(&wanted) {
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
        drop((lend, s));
        free(at, 4 * page);
    }

    /// A dma-buf of a view the library mapped, made by its address rather than through a lend,
    /// holds the pages the view shows there; memory no file is behind is refused
    #[test]
    fn a_dma_buf_of_host_memory_holds_the_pages_its_mapping_shows() {
        let page = vm::page_size();
        let (at, ranges) = scattered_pages(3);
        let s = LinuxSurfaces::new().create(&over(&ranges, None)).unwrap();
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

    /// A protected surface and one the guest made read-only for its clients are lent to no
    /// device, as their bytes are lent to no writer
    #[test]
    fn read_only_and_protected_surfaces_are_not_lent() {
        let page = vm::page_size();
        let (at, ranges) = scattered_pages(2);
        let surfaces = LinuxSurfaces::new();
        let read_only = surfaces
            .create(&over(&ranges, Some(READ_ONLY_DIRECTION)))
            .unwrap();
        assert_eq!(
            lend_to_devices(&read_only).err(),
            Some("a surface the guest made read-only for its clients")
        );
        let protected = surfaces
            .create(&SurfaceSpec {
                client_address_ranges: None,
                offset: 0,
                alloc_size: page,
                protection_options: 1,
                ..over(&ranges, None)
            })
            .unwrap();
        assert_eq!(
            lend_to_devices(&protected).err(),
            Some("a protected surface, whose memory the CPU never reaches")
        );
        drop((read_only, protected));
        free(at, 2 * page);
    }

    /// Dropped with the surface, for the tests.
    struct Flag(Arc<AtomicBool>);

    impl Drop for Flag {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    /// What the host attaches to a surface is one value for every object over it, its lookups
    /// among them, and goes with the last; a hold a lend made keeps the view mapped past it and
    /// keeps nothing attached alive. A value of another type is not answered
    #[test]
    fn an_attachment_lives_as_long_as_its_surface() {
        let page = vm::page_size();
        let (at, ranges) = scattered_pages(2);
        let s = LinuxSurfaces::new().create(&over(&ranges, None)).unwrap();
        let gone = Arc::new(AtomicBool::new(false));
        let made = || Flag(Arc::clone(&gone));
        let key = |flag: &Flag| Arc::as_ptr(&flag.0).addr();
        let first = attachment(&s, made, key).unwrap();
        let lookup = s.lookup_by_port().unwrap();
        let again = attachment(&lookup, || -> Flag { unreachable!() }, key);
        assert_eq!(again, Some(first));
        assert_eq!(attachment(&s.clone_ref(), || 0u8, |_| ()), None);
        let hold = lend_to_devices(&s).unwrap().hold();
        let view = HostAddr::from_addr(s.base_address().addr());
        drop(s);
        assert!(!gone.load(Ordering::SeqCst), "the lookup keeps the surface");
        drop(lookup);
        assert!(gone.load(Ordering::SeqCst), "gone with the last object");
        assert!(mapped(view), "the hold keeps the view");
        drop(hold);
        assert!(!mapped(view));
        free(at, 2 * page);
    }
}
