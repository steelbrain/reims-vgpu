//! IOSurfaces whose GPU view is their own guest pages (`SURFACE_PAGES`,
//! `BUFFER_OVER_SURFACE_PAGES`; docs/graphics-kext/design.md, Decided 14).
//!
//! The accelerator kext wires an IOSurface's allocation while a process holds the surface's
//! shared block, and lists its guest RAM runs in the block (`SURFACE_PAGES`). The host maps the
//! runs as one view through the memory map ([`MemoryMap::new_range_mapping`], the same views the
//! apple-gfx lane's memory maps make), and a `BUFFER_OVER_SURFACE_PAGES` buffer lies over the
//! view, so the GPU reads and writes the surface's own pages in place: no copy of them is made
//! or synced. The view is the block's until its `BLOCK_RELEASE`; each buffer over it holds it
//! until the platform frees the buffer.
//!
//! **The cache.** WindowServer lets a surface's last texture go and makes one again every frame,
//! so the kext gives the surface a fresh block each time (measured on the msi 5080 host,
//! macOS 26 guest, a window dragged: 975 fresh blocks in 30 s, 729 of them one 8 MiB surface's).
//! The [`CACHED_VIEWS`] views made last are kept, by their runs, each with a platform buffer over
//! the whole view that holds the runtime's import of its pages: a block over the same pages takes
//! that view, and buffers over it the live import, with no mapping or import made again. A view
//! maps the guest's pages themselves, so it never goes stale: the memory map's pages stay the
//! guest's RAM while the device lives (`GraphicsConfig::memory_map`).
//!
//! The runs are the guest's: each must be whole pages of the guest's (4 KiB) and of the host's
//! from a page, lie in guest RAM, and the runs must sum to the surface's length; else the block
//! gets no view, `SURFACE_PAGES` is refused, and the kext gives the surface a copy block instead.

use std::any::Any;
use std::collections::{HashMap, VecDeque};
use std::ops::Range;
use std::sync::Arc;

use reims_graphics::wire::{RgfxBufferOverSurfacePages, RgfxPageRun, RgfxSurfacePages};
use reims_vgpu_core::guest::records::APV_PAGE_SIZE;
use reims_vgpu_core::host::{HostPtr, PhysRange};
use reims_vgpu_core::memory::map::MemoryMap;
use reims_vgpu_core::memory::mapping::Mapping;
use reims_vgpu_core::num::{ToU64 as _, ToUsize as _};
use reims_vgpu_core::platform::device::GpuDevice as _;
use reims_vgpu_core::platform::values::{BufferDesc, OptionSet};
use reims_vgpu_core::platform::{Handle, HostRange, Platform};
use reims_vgpu_core::raise::{RaiseResult, refusals};

/// The guest's page: every run is whole pages of it.
pub(crate) const GUEST_PAGE: u64 = 4096;

/// How many views the cache keeps, the last made or taken.
pub(crate) const CACHED_VIEWS: usize = 32;

/// The most bytes of guest pages the cache's views map, past which the oldest go.
pub(crate) const CACHED_BYTES: u64 = 1 << 30;

/// One surface's view of its guest pages, which the buffers of its blocks lie over.
pub(crate) struct SurfaceView<P: Platform> {
    /// The view: the runs mapped in order, `length` bytes.
    mapping: Arc<Mapping<P>>,
    /// The platform's hold on the guest's RAM, kept until the view's teardown.
    _lease: Lease,
}

/// The platform's hold on the guest's RAM the views are made of.
pub(crate) type Lease = Arc<dyn Any + Send + Sync>;

impl<P: Platform> SurfaceView<P> {
    /// The view's bytes: the surface's.
    pub(crate) fn length(&self) -> u64 {
        self.mapping.length()
    }

    /// The view's first byte.
    pub(crate) fn base(&self) -> HostPtr {
        self.mapping.virtual_address()
    }
}

/// Why a `SURFACE_PAGES` left its block with no view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PagesRefusal {
    /// The device has no memory map: the host was given no guest RAM.
    NoMemoryMap,
    /// The host's or the guest's page is not one the runs can be mapped in.
    PageSize,
    /// No runs, or a run that is empty or not whole pages from a page.
    Unaligned,
    /// The runs' lengths do not sum to the surface's.
    Length,
    /// A run lies outside guest RAM, or the host refused the view (its views' budget).
    NotMapped,
    /// The platform made no buffer over the view.
    NoBuffer,
}

/// What the views did, for the log ([`SurfaceViews::counts`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct ViewCounts {
    /// Views made.
    pub(crate) made: u64,
    /// Blocks given a cached view.
    pub(crate) reused: u64,
    /// `SURFACE_PAGES` refused for runs that are not whole pages, or do not sum to the length.
    pub(crate) bad_runs: u64,
    /// `SURFACE_PAGES` refused for runs outside guest RAM, or a view or buffer the host did not
    /// make.
    pub(crate) not_mapped: u64,
}

/// A view the cache keeps, with the buffer that holds the runtime's import of its pages.
struct Cached<P: Platform> {
    /// The runs the view maps, in order.
    runs: Vec<(u64, u64)>,
    view: Arc<SurfaceView<P>>,
    _import: P::Buffer,
}

/// The views of the surfaces whose blocks name their pages, by block offset, and the cache.
pub(crate) struct SurfaceViews<P: Platform> {
    /// The guest's RAM, through which the views are made, and the platform's hold on it; `None`
    /// makes none.
    memory: Option<(Arc<MemoryMap<P>>, Lease)>,
    /// Each block's view, and the size of the block's part its holders map.
    by_block: HashMap<u64, (Arc<SurfaceView<P>>, u64)>,
    /// The views made or taken last, the latest at the back.
    cache: VecDeque<Cached<P>>,
    counts: ViewCounts,
}

impl<P: Platform> SurfaceViews<P> {
    /// No views, made through `memory`.
    pub(crate) fn new(memory: Option<(Arc<MemoryMap<P>>, Lease)>) -> Self {
        SurfaceViews {
            memory,
            by_block: HashMap::new(),
            cache: VecDeque::new(),
            counts: ViewCounts::default(),
        }
    }

    /// Whether views can be made: a memory map, and a host page the guest's runs are whole
    /// pages of ([`GUEST_PAGE`]); the device's [`RGFX_FEATURE_SURFACE_PAGES`] says so.
    ///
    /// [`RGFX_FEATURE_SURFACE_PAGES`]: reims_graphics::wire::RGFX_FEATURE_SURFACE_PAGES
    pub(crate) fn available(&self, platform: &P) -> bool {
        self.memory.is_some()
            && platform.vm().page_size() == GUEST_PAGE
            && APV_PAGE_SIZE == GUEST_PAGE
    }

    /// What the views did so far.
    pub(crate) fn counts(&self) -> ViewCounts {
        self.counts
    }

    /// `SURFACE_PAGES`: the view of `runs` for the block `pages` names, in place of any it had:
    /// the cache's over the same runs, else one made, with a buffer over it on `device`.
    pub(crate) fn surface_pages(
        &mut self,
        platform: &P,
        device: &P::Device,
        pages: &RgfxSurfacePages,
        runs: &[RgfxPageRun],
    ) -> RaiseResult<Result<(), PagesRefusal>> {
        self.by_block.remove(&pages.block_offset);
        match self.view_of(platform, device, pages, runs)? {
            Ok(view) => {
                self.by_block
                    .insert(pages.block_offset, (view, pages.block_size));
                Ok(Ok(()))
            }
            Err(why) => {
                if matches!(why, PagesRefusal::Unaligned | PagesRefusal::Length) {
                    self.counts.bad_runs += 1;
                } else {
                    self.counts.not_mapped += 1;
                }
                Ok(Err(why))
            }
        }
    }

    /// The cache's view of `runs`, taken to the back, or a new one, cached.
    fn view_of(
        &mut self,
        platform: &P,
        device: &P::Device,
        pages: &RgfxSurfacePages,
        runs: &[RgfxPageRun],
    ) -> RaiseResult<Result<Arc<SurfaceView<P>>, PagesRefusal>> {
        let Some((memory, lease)) = self.memory.as_ref() else {
            return Ok(Err(PagesRefusal::NoMemoryMap));
        };
        // Whole pages of the host's, of the guest's and of the length core's mappings take.
        let page = platform.vm().page_size().max(APV_PAGE_SIZE).max(GUEST_PAGE);
        if !page.is_power_of_two() {
            return Ok(Err(PagesRefusal::PageSize));
        }
        if runs.is_empty()
            || runs.iter().any(|r| {
                r.length == 0 || !r.address.is_multiple_of(page) || !r.length.is_multiple_of(page)
            })
        {
            return Ok(Err(PagesRefusal::Unaligned));
        }
        let total = runs
            .iter()
            .try_fold(0u64, |sum, r| sum.checked_add(r.length));
        if total != Some(pages.length) {
            return Ok(Err(PagesRefusal::Length));
        }
        let key: Vec<(u64, u64)> = runs.iter().map(|r| (r.address, r.length)).collect();
        let hit = self.cache.iter().position(|c| c.runs == key);
        if let Some(cached) = hit.and_then(|at| self.cache.remove(at)) {
            let view = Arc::clone(&cached.view);
            self.cache.push_back(cached);
            self.counts.reused += 1;
            return Ok(Ok(view));
        }
        let ranges: Vec<PhysRange> = key
            .iter()
            .map(|&(physical_address, physical_length)| PhysRange {
                physical_address,
                physical_length,
            })
            .collect();
        // Writable: the GPU draws into a surface as it samples one.
        let Some(mapping) = memory.new_range_mapping(&ranges, pages.length, 0) else {
            return Ok(Err(PagesRefusal::NotMapped));
        };
        let view = Arc::new(SurfaceView {
            mapping,
            _lease: Arc::clone(lease),
        });
        let Some(import) = buffer_over(device, &view, view.length(), 0)? else {
            return Ok(Err(PagesRefusal::NoBuffer));
        };
        self.counts.made += 1;
        self.cache.push_back(Cached {
            runs: key,
            view: Arc::clone(&view),
            _import: import,
        });
        let mut bytes: u64 = self.cache.iter().map(|c| c.view.length()).sum();
        while self.cache.len() > CACHED_VIEWS || (bytes > CACHED_BYTES && self.cache.len() > 1) {
            if let Some(old) = self.cache.pop_front() {
                bytes -= old.view.length();
            }
        }
        Ok(Ok(view))
    }

    /// The view of the block at `m.block_offset`, where its holders' part holds the
    /// `m.block_size` bytes `m` names and the view at least `m.length` bytes.
    pub(crate) fn view_for(&self, m: &RgfxBufferOverSurfacePages) -> Option<Arc<SurfaceView<P>>> {
        self.by_block
            .get(&m.block_offset)
            .filter(|(v, size)| m.block_size <= *size && m.length != 0 && m.length <= v.length())
            .map(|(v, _)| Arc::clone(v))
    }

    /// The view of the block at `block`.
    pub(crate) fn view(&self, block: u64) -> Option<Arc<SurfaceView<P>>> {
        self.by_block.get(&block).map(|(v, _)| Arc::clone(v))
    }

    /// The host addresses of the guest pages of `runs` in the mapping of the guest's RAM the
    /// memory map names, as runs ([`crate::device_writes::sorted`]): what the host's devices'
    /// writes of them name. A page the map does not hold is left out.
    pub(crate) fn host_runs(&self, runs: &[RgfxPageRun]) -> Vec<Range<u64>> {
        let Some((memory, _)) = &self.memory else {
            return Vec::new();
        };
        let mut pages = Vec::new();
        for r in runs {
            for at in (r.address..r.address.saturating_add(r.length)).step_by(GUEST_PAGE.to_usize())
            {
                let host = memory.virtual_address_for_physical(at, GUEST_PAGE);
                if !host.is_null() {
                    pages.push(host.addr().to_u64());
                }
            }
        }
        crate::device_writes::page_runs(&pages, GUEST_PAGE)
    }

    /// The block at `offset` went (its `BLOCK_RELEASE`): its view goes once the buffers over it
    /// and the cache have let it go.
    pub(crate) fn released(&mut self, offset: u64) {
        self.by_block.remove(&offset);
    }

    /// The device's reset: every block's view and every cached one go, each buffer over them
    /// let go of in a refusal scope, as no block names its pages any more; what the views did is
    /// still counted.
    pub(crate) fn reset(&mut self) {
        let views = std::mem::take(&mut self.by_block);
        let cache = std::mem::take(&mut self.cache);
        if let Err(refusal) = refusals::refusing(move || drop((views, cache))) {
            log::warn!(
                target: "reims_vgpu::graphics",
                "the IOSurface views were let go of at the device's reset, and a release was \
                 refused: {refusal}"
            );
        }
    }

    /// How many blocks have a view, and how many views the cache keeps (for the tests).
    #[cfg(test)]
    pub(crate) fn len(&self) -> (usize, usize) {
        (self.by_block.len(), self.cache.len())
    }
}

/// A Shared buffer of `options` over the first `length` bytes of `view`, which it holds until the
/// platform frees it.
pub(crate) fn buffer_over<P: Platform>(
    device: &P::Device,
    view: &Arc<SurfaceView<P>>,
    length: u64,
    options: u64,
) -> RaiseResult<Option<P::Buffer>> {
    debug_assert!(length <= view.length());
    // SAFETY: the deallocator holds the view, so its pages stay mapped until the platform frees
    // the buffer; `length` is at most the view's.
    let bytes = unsafe { HostRange::new(view.base(), length) };
    let keep = Arc::clone(view);
    device.new_buffer(BufferDesc {
        length,
        options: vec![OptionSet::ResourceOptions(options)],
        pinned_gpu_address: None,
        contents: Some(bytes),
        no_copy: true,
        deallocator: Some(Box::new(move |_, _| drop(keep))),
    })
}

impl<P: Platform> super::GraphicsDecoder<P> {
    /// `BUFFER_OVER_SURFACE_PAGES`: a Shared buffer over the view of the surface whose shared
    /// block the sender holds, lying over the block's part the holders map (`spans`), as a
    /// buffer over host memory lies over its block. A block the sender does not hold is a guest
    /// protocol error; one with no view, or a length past it, is refused.
    pub(super) fn buffer_over_surface_pages(
        &self,
        device: &P::Device,
        message: &[u8],
    ) -> RaiseResult<Result<(), super::DecodeError>> {
        let Some(m) = super::read::<RgfxBufferOverSurfacePages>(message) else {
            return Ok(Err(super::DecodeError::Malformed));
        };
        if !self.in_own_block(m.block_offset, m.block_size) {
            return Ok(Err(super::DecodeError::BadReference));
        }
        let Some(view) = self.shared.surface_views.lock().view_for(&m) else {
            return Ok(Err(super::DecodeError::Refused));
        };
        // A block with a device cache: the buffer is the cache, over the surface's bytes in the
        // pages' layout (`surface_cache`).
        let cache = (self.surface_caches.lock().buffer(m.block_offset)).map(|c| c.clone_ref());
        if let Some(cache) = cache {
            return self.insert_over(
                m.obj_id,
                super::ObjHandle::Buffer(cache),
                Some((m.block_offset, m.block_size)),
            );
        }
        match buffer_over(device, &view, m.length, m.options.into())? {
            Some(buffer) => self.insert_over(
                m.obj_id,
                super::ObjHandle::Buffer(buffer),
                Some((m.block_offset, m.block_size)),
            ),
            None => Ok(Err(super::DecodeError::Refused)),
        }
    }
}

#[cfg(test)]
pub(super) mod tests {
    use reims_graphics::region::Region;
    use reims_graphics::wire::{
        RGFX_CMD_BUFFER_OVER_SURFACE_PAGES, RGFX_CMD_OBJECT_DESTROY, RGFX_CMD_SURFACE_PAGES,
        RGFX_CONTEXT_ALLOCATIONS_OFFSET, RGFX_CURSOR_SPAN, RGFX_STATUS_OK, RGFX_STATUS_REFUSED,
        RgfxMsgHeader, RgfxObjectDestroy,
    };
    use reims_vgpu_core::memory::descriptor::{GuestPhysicalRange, MemoryMapDescriptor};
    use reims_vgpu_core::testkit::{FakePlatform, GuestRam};
    use zerocopy::IntoBytes;

    use super::super::{DecodeError, GraphicsDecoder, no_raise};
    use super::*;

    /// The processes' blocks start here.
    pub(crate) const BLOCK: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x4000;
    /// The guest RAM's first byte.
    pub(crate) const RAM: u64 = 0x10_0000;

    pub(crate) struct Rig {
        pub(crate) p: Arc<FakePlatform>,
        pub(crate) device: <FakePlatform as Platform>::Device,
        pub(crate) ram: GuestRam,
        pub(crate) dec: GraphicsDecoder<FakePlatform>,
        /// The host's page, which the runs are whole pages of here.
        pub(crate) page: u64,
    }

    /// A decoder over a region with a MiB of blocks, and `pages` host pages of guest RAM from
    /// [`RAM`] behind its memory map.
    pub(crate) fn rig(pages: u64) -> Rig {
        let p = Arc::new(FakePlatform::new());
        let device = p.device();
        let page = p.vm().page_size();
        let size = BLOCK + 0x10_0000 + RGFX_CURSOR_SPAN;
        let region = Arc::new(Region::new(size as usize).unwrap());
        let mut dec = GraphicsDecoder::<FakePlatform>::new(Arc::clone(&p), region);
        let ram = GuestRam::with_range(RAM, pages * page).unwrap();
        p.fake_vm().adopt(&ram);
        let mut desc = MemoryMapDescriptor::new();
        for r in ram.ranges() {
            desc.add_range(GuestPhysicalRange {
                physical_address: r.physical_address,
                physical_length: r.physical().physical_length,
                host_address: r.host(),
            });
        }
        let map = MemoryMap::new(Arc::clone(&p), &desc).unwrap();
        dec.set_guest_memory(map, Arc::new(()));
        Rig {
            p,
            device,
            ram,
            dec,
            page,
        }
    }

    fn header(r#type: u32) -> RgfxMsgHeader {
        RgfxMsgHeader {
            r#type,
            size: 0,
            seqno: 0,
        }
    }

    impl Rig {
        /// A `SURFACE_PAGES` for the block at `block` of `runs`, given as (page, pages) of guest
        /// RAM, over `length` bytes; its answer.
        pub(crate) fn pages(&mut self, block: u64, runs: &[(u64, u64)], length: u64) -> u32 {
            self.pages_cached(block, runs, length, false)
        }

        /// [`Self::pages`], the block marked `cached`.
        pub(crate) fn pages_cached(
            &mut self,
            block: u64,
            runs: &[(u64, u64)],
            length: u64,
            cached: bool,
        ) -> u32 {
            let runs: Vec<RgfxPageRun> = runs
                .iter()
                .map(|&(at, n)| RgfxPageRun {
                    address: RAM + at * self.page,
                    length: n * self.page,
                })
                .collect();
            let m = RgfxSurfacePages {
                header: header(RGFX_CMD_SURFACE_PAGES),
                context: 0,
                run_count: runs.len() as u32,
                block_offset: block,
                block_size: 0x2000,
                runs_offset: block + 0x1000,
                length,
            };
            self.dec.surface_pages(&self.device, &m, &runs, cached)
        }

        /// A `BUFFER_OVER_SURFACE_PAGES` of `length` bytes over the block at `block`.
        pub(crate) fn buffer(
            &mut self,
            obj_id: u64,
            block: u64,
            length: u64,
        ) -> Result<(), DecodeError> {
            let m = RgfxBufferOverSurfacePages {
                header: header(RGFX_CMD_BUFFER_OVER_SURFACE_PAGES),
                obj_id,
                block_offset: block,
                block_size: 0x1000,
                length,
                options: 0,
                reserved_34: 0,
            };
            self.dec.decode(&self.device, m.as_bytes()).unwrap()
        }

        /// The view the block at `block` has.
        fn view(&self, block: u64) -> Arc<SurfaceView<FakePlatform>> {
            Arc::clone(&self.dec.views_of_surfaces().by_block[&block].0)
        }
    }

    /// A surface's runs, scattered and out of order in guest RAM, are one view in their order,
    /// over the guest's own pages: a buffer over the block lies over it, and over the block's
    /// part its holders map, so the block is not released while it lives
    #[test]
    fn a_surface_s_runs_are_one_view_its_buffers_lie_over() {
        let mut r = rig(8);
        let page = r.page;
        // Pages 5 and 6, then page 1.
        assert!(r.ram.write(RAM + 5 * page, b"fifth"));
        assert!(r.ram.write(RAM + page, b"first"));
        assert_eq!(r.pages(BLOCK, &[(5, 2), (1, 1)], 3 * page), RGFX_STATUS_OK);
        let view = r.view(BLOCK);
        assert_eq!(view.length(), 3 * page);
        let fake = r.p.fake_vm();
        assert_eq!(fake.read(view.base(), 5).unwrap(), b"fifth");
        assert_eq!(
            fake.read(view.base().wrapping_add(2 * page), 5).unwrap(),
            b"first"
        );
        // Writes through the view reach the guest's page.
        assert!(fake.write(view.base().wrapping_add(2 * page), b"GPU"));
        fake.sync();
        let mut got = [0u8; 3];
        assert!(r.ram.read(RAM + page, &mut got));
        assert_eq!(&got, b"GPU");

        let mark = r.p.call_log().mark();
        assert_eq!(r.buffer(10, BLOCK, 3 * page), Ok(()));
        assert_eq!(
            r.p.call_log().sels_since(mark),
            ["newBufferWithDescriptor:"]
        );
        assert!(r.dec.span_in_use(BLOCK, 0x1000));
        // Past the view, or a block with none: refused.
        assert_eq!(r.buffer(11, BLOCK, 3 * page + 1), Err(DecodeError::Refused));
        assert_eq!(
            r.buffer(12, BLOCK + 0x10000, page),
            Err(DecodeError::Refused)
        );
        // The buffer holds the view after the block's release.
        r.dec
            .block_released(&r.device, BLOCK, false, Box::new(|| ()), no_raise());
        assert_eq!(r.buffer(13, BLOCK, page), Err(DecodeError::Refused));
        assert!(
            Arc::strong_count(&view) > 2,
            "the buffer and the cache hold it"
        );
        let destroy = RgfxObjectDestroy {
            header: header(RGFX_CMD_OBJECT_DESTROY),
            obj_id: 10,
        };
        assert_eq!(r.dec.decode(&r.device, destroy.as_bytes()).unwrap(), Ok(()));
        assert!(!r.dec.span_in_use(BLOCK, 0x1000));
    }

    /// A block made again over the same runs takes the cached view, with no mapping made; the
    /// cache keeps [`CACHED_VIEWS`] views, the last taken
    #[test]
    fn a_block_over_the_same_runs_takes_the_cached_view() {
        let mut r = rig(64);
        let page = r.page;
        assert_eq!(r.pages(BLOCK, &[(3, 1), (0, 1)], 2 * page), RGFX_STATUS_OK);
        let first = r.view(BLOCK);
        r.dec
            .block_released(&r.device, BLOCK, false, Box::new(|| ()), no_raise());
        let mark = r.p.call_log().mark();
        assert_eq!(
            r.pages(BLOCK + 0x8000, &[(3, 1), (0, 1)], 2 * page),
            RGFX_STATUS_OK
        );
        assert!(
            r.p.call_log().sels_since(mark).is_empty(),
            "no mapping or buffer made"
        );
        assert!(Arc::ptr_eq(&first, &r.view(BLOCK + 0x8000)));
        assert_eq!(r.dec.views_of_surfaces().counts().reused, 1);
        // The same pages in another order are another view.
        assert_eq!(r.pages(BLOCK, &[(0, 1), (3, 1)], 2 * page), RGFX_STATUS_OK);
        assert!(!Arc::ptr_eq(&first, &r.view(BLOCK)));
        // Past CACHED_VIEWS views the oldest goes from the cache.
        for i in 0..CACHED_VIEWS as u64 {
            assert_eq!(r.pages(BLOCK, &[(10 + i, 1)], page), RGFX_STATUS_OK);
        }
        assert_eq!(r.dec.views_of_surfaces().len().1, CACHED_VIEWS);
        r.dec.block_released(
            &r.device,
            BLOCK + 0x8000,
            false,
            Box::new(|| ()),
            no_raise(),
        );
        assert_eq!(
            Arc::strong_count(&first),
            1,
            "only the test holds the first view"
        );
    }

    /// Runs that are not whole pages, that do not sum to the surface's length, or that lie
    /// outside guest RAM make no view, are refused, and are counted
    #[test]
    fn bad_runs_make_no_view() {
        let mut r = rig(8);
        let page = r.page;
        assert_eq!(r.pages(BLOCK, &[], 0), RGFX_STATUS_REFUSED);
        assert_eq!(r.pages(BLOCK, &[(1, 1)], 2 * page), RGFX_STATUS_REFUSED);
        assert_eq!(r.pages(BLOCK, &[(100, 1)], page), RGFX_STATUS_REFUSED);
        let runs = [RgfxPageRun {
            address: RAM + 0x800,
            length: page,
        }];
        let m = RgfxSurfacePages {
            header: header(RGFX_CMD_SURFACE_PAGES),
            run_count: 1,
            block_offset: BLOCK,
            block_size: 0x2000,
            runs_offset: BLOCK + 0x1000,
            length: page,
            ..RgfxSurfacePages::default()
        };
        assert_eq!(
            r.dec.surface_pages(&r.device, &m, &runs, false),
            RGFX_STATUS_REFUSED
        );
        let c = r.dec.views_of_surfaces().counts();
        assert_eq!((c.made, c.bad_runs, c.not_mapped), (0, 3, 1));
        assert_eq!(r.buffer(10, BLOCK, page), Err(DecodeError::Refused));
        // A good one after them has its view.
        assert_eq!(r.pages(BLOCK, &[(1, 1)], page), RGFX_STATUS_OK);
        assert_eq!(r.buffer(10, BLOCK, page), Ok(()));
    }

    /// With no memory map the device has no `SURFACE_PAGES` and makes no view
    #[test]
    fn no_memory_map_makes_no_view() {
        let p = Arc::new(FakePlatform::new());
        let device = p.device();
        let size = BLOCK + 0x10_0000 + RGFX_CURSOR_SPAN;
        let region = Arc::new(Region::new(size as usize).unwrap());
        let mut dec = GraphicsDecoder::<FakePlatform>::new(Arc::clone(&p), region);
        assert!(!dec.surface_pages_available());
        let m = RgfxSurfacePages {
            header: header(RGFX_CMD_SURFACE_PAGES),
            run_count: 1,
            block_offset: BLOCK,
            block_size: 0x2000,
            runs_offset: BLOCK + 0x1000,
            length: 0x4000,
            ..RgfxSurfacePages::default()
        };
        let runs = [RgfxPageRun {
            address: RAM,
            length: 0x4000,
        }];
        assert_eq!(
            dec.surface_pages(&device, &m, &runs, false),
            RGFX_STATUS_REFUSED
        );
    }
}
