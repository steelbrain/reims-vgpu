//! The alias the buffers over a surface lie in, of the surface's pages ([`Alias`]), and the
//! read-only bit each buffer and heap keeps ([`Backing`]).
//!
//! - **An alias per surface.** The surface's pages are remapped, shared, into a fresh range of
//!   this platform's own (`mach_vm_remap`, one call per run of host-contiguous pages), and the
//!   runtime is given that range with a keep-alive that frees it ([`keep_alive`]), and no
//!   deallocator: no one but this platform sees the alias go. A write through the alias shows in
//!   the guest's pages and one through the guest's pages in the alias, so the GPU's writes land
//!   in the guest's memory at once and it reads the guest's latest bytes, as Metal's do. The
//!   runtime keeps an alias, and so the pages under it, until every command buffer committed with
//!   it completed and it freed its import of them, which it does before it drops the keep-alive:
//!   the buffer's release never takes pages a committed command buffer still uses or the device
//!   still imports, and the guest's memory stays mapped under it whatever the guest's own mapping
//!   does meanwhile. The runtime keeps the pages a range had when it imported them
//!   (`tests/no_copy_remap.rs`).
//! - **Whole pages.** The alias takes the pages its ranges lie in, whole: ranges whose bytes
//!   follow on from the previous range's are one run of bytes, and each run of bytes takes the
//!   pages it lies in, from the page of its first byte through the page of its last ([`spans`]).
//!   A first range starting off its page dies where it is given ([`off_the_page`]).
//! - **Read-only.** A read-only alias (of a surface the GPU only reads) has its maximum
//!   protection lowered to read (`mach_vm_protect` with `set_maximum`), as Metal's alias of
//!   read-only memory over several ranges is (measured: current and maximum protection read).
//!   The GPU's writes never reach such pages, and the command buffer completes: later commands
//!   of the same command buffer read what it wrote, and a later command buffer the guest's
//!   bytes, as on Metal (blit fills, measured on Metal; kernel stores are dropped alike on
//!   Metal). On Linux the runtime imports a surface's read-only alias through a writable twin
//!   ([`ImportSite::twins`]), and makes a no-copy buffer or heap made with `MTLResourceOptions`
//!   bit 24 or over pages mapped read-only over a copy of them, whose GPU writes it drops, as the
//!   views' gate refuses read-only pages (`crate::os::import_view`): a shader's, a blit's into a
//!   buffer and a visibility result's are sent, and dropped as Metal's are, while a pass's store
//!   into the object's memory and a write of a texture over it are answered where they are
//!   encoded ([`Backing::read_only_now`], `crate::gpu::writes`).

use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};

use crate::core::host::HostPtr;
use crate::core::num::ToU64;
use crate::core::platform::os::vm::{HostVm, KernReturn, VmProt};
use crate::core::platform::types::ResourceAddressRange;
use crate::core::rt::BlockEnv;
use crate::core::sync::Mutex;
use metal2vulkan_runtime as rt;

use crate::answer::unrepresentable;
use crate::os::OsVm;

/// The runtime rounds a heap's size up to a multiple of this (16 KiB), and the heap's memory
/// covers all of it.
pub(crate) const HEAP_GRANULE: u64 = 0x4000;

/// The read-only bit a buffer or heap was made with, and whether the memory it lies in is
/// read-only.
#[derive(Debug)]
pub struct Backing {
    read_only: bool,
    /// Whether the memory the object lies in is read-only where the runtime does not import
    /// read-only memory as it is mapped ([`crate::os::COPIES_READ_ONLY`]): the memory a no-copy
    /// buffer or heap was made over, mapped read-only or made with bit 24; counted while it holds
    /// ([`crate::gpu::writes::set_kept`]).
    kept: AtomicBool,
}

impl Backing {
    /// The backing of a buffer or heap over the memory it was made with; `read_only` for one
    /// made with `MTLResourceOptions` bit 24.
    pub(crate) fn new(read_only: bool) -> Backing {
        Backing {
            read_only,
            kept: AtomicBool::new(false),
        }
    }

    /// The object, a no-copy buffer or heap, was made over memory of the caller's that is
    /// read-only where `read_only` (mapped so, [`crate::os::ImportView::read_only`], or made with
    /// bit 24): its memory is read-only then, and where it was made with bit 24
    /// ([`Backing::read_only_now`]).
    pub(crate) fn imported(&self, read_only: bool) {
        let kept = crate::os::COPIES_READ_ONLY && (read_only || self.read_only);
        crate::gpu::writes::set_kept(&self.kept, kept);
    }

    /// Whether the memory the object lies in is read-only where the runtime does not import
    /// read-only memory as it is mapped (on Linux, [`crate::os::COPIES_READ_ONLY`]): it makes the
    /// object over a copy of it, dropping the GPU's writes, where Metal's GPU keeps them from the
    /// memory itself and faults a pass that stores into it.
    pub(crate) fn read_only_now(&self) -> bool {
        self.kept.load(Ordering::Acquire)
    }

    /// [`Backing::read_only_now`], asked as work that may write through a texture over the
    /// object is encoded ([`crate::gpu::writes::Kept`]).
    pub(crate) fn kept(&self) -> crate::gpu::writes::Kept {
        match self.read_only_now() {
            true => crate::gpu::writes::Kept::Copied,
            false => crate::gpu::writes::Kept::No,
        }
    }
}

impl Drop for Backing {
    /// The object's release: its memory is read-only no longer ([`crate::gpu::writes::set_kept`]).
    fn drop(&mut self) {
        crate::gpu::writes::set_kept(&self.kept, false);
    }
}

/// A range of this platform's address space whose pages are the guest's, shared, and zeros past
/// them ([`gather_or_nil`]), freed when it drops; or a range another holder maps, which the alias
/// borrows ([`Alias::borrowing`]): a surface's own view of the guest's pages on Linux, which
/// holds the same pages, zeros past them included, and lets the surface's buffers lie over it
/// with no second view of its pages (`crate::os::surface_alias`).
#[derive(Debug)]
pub(crate) struct Alias {
    at: NonNull<u8>,
    length: u64,
    /// What the runtime imports for it while an import of it lives: its pages, or, a surface's,
    /// their writable twin ([`crate::os::import_view`]), one for every import made meanwhile
    /// ([`Alias::into_runtime`]).
    import: Mutex<Weak<crate::os::ImportView>>,
    /// The hold on the memory of a borrowed alias, which its holder maps and frees, dropped with
    /// the alias; `None` for one of this platform's own, which the alias frees.
    borrowed: Option<Box<dyn std::fmt::Debug + Send + Sync>>,
}

// SAFETY: an `Alias` is an address range this platform owns or borrows; nothing reads through it
// here, and it is freed or let go of once, by whichever thread drops it.
unsafe impl Send for Alias {}

// SAFETY: as for `Send`: a shared `Alias` only answers where it is, and imports it.
unsafe impl Sync for Alias {}

impl Alias {
    /// An alias of this platform's own, `length` bytes at `at`, which it frees.
    fn own(at: NonNull<u8>, length: u64) -> Alias {
        Alias::of(at, length, None)
    }

    /// The `length` bytes at `at`, mapped by another holder while `hold` lives, as an alias that
    /// holds `hold` and frees nothing: what it shows is the holder's (`crate::os::surface_alias`).
    #[cfg_attr(
        target_os = "macos",
        expect(dead_code, reason = "only Linux's surfaces lend their own view")
    )]
    pub(crate) fn borrowing(
        at: NonNull<u8>,
        length: u64,
        hold: impl std::fmt::Debug + Send + Sync + 'static,
    ) -> Alias {
        Alias::of(at, length, Some(Box::new(hold)))
    }

    /// The alias of `length` bytes at `at`, borrowed where `borrowed` holds them.
    fn of(
        at: NonNull<u8>,
        length: u64,
        borrowed: Option<Box<dyn std::fmt::Debug + Send + Sync>>,
    ) -> Alias {
        Alias {
            at,
            length,
            import: Mutex::new(Weak::new()),
            borrowed,
        }
    }

    /// The first byte the runtime imports, and the runtime's keep-alive of the alias, which lets
    /// go of it as it drops: once the runtime freed its import, at the buffer's release where
    /// it lies in a live buffer's import ([`keep_alive`]), or at once where the call was
    /// refused. The alias is freed once nothing else holds it.
    ///
    /// The runtime imports the alias's pages, or, a surface's ([`ImportSite::twins`]), where the
    /// host's runtime cannot import them as they are mapped (read-only, on Linux), a writable
    /// twin of them ([`crate::os::import_view`]), whose first byte this answers, and which the
    /// keep-alive holds too: the GPU's writes then reach the pages, where Metal drops them
    /// (`crate::gpu::writes::kept_write`). A read-only alias of any other site is given as it
    /// is, which the runtime copies, dropping the GPU's writes. Every import of the alias made
    /// while another lives is given the same memory, its twin included, so the buffers over one
    /// surface's alias lie at one first byte, in one import of the runtime's (its
    /// `host_aliases.rs`), with one twin's mappings, where each would take a twin of its own.
    /// And the word that the import was made, which the caller gives once the runtime's call
    /// returned ([`crate::os::ImportMade`]).
    pub(crate) fn into_runtime(
        self: Arc<Self>,
        site: ImportSite,
    ) -> (NonNull<u8>, rt::KeepAlive, crate::os::ImportMade) {
        let first = self.at;
        let (view, made) = {
            let mut import = self.import.lock();
            match import.upgrade() {
                Some(view) => {
                    let made = view.again();
                    (view, made)
                }
                None => {
                    let view = crate::os::import_view(first, self.length, site.twins());
                    let made = view.made();
                    let view = Arc::new(view);
                    *import = Arc::downgrade(&view);
                    (view, made)
                }
            }
        };
        let imported = view.at();
        let traced = ImportTrace::new(site, first, self.length, imported);
        (imported, keep_alive((view, self, traced)), made)
    }

    /// The first byte.
    #[cfg(test)]
    pub(crate) fn first(&self) -> NonNull<u8> {
        self.at
    }
}

impl Drop for Alias {
    /// Frees the alias of this platform's own; lets go of a borrowed one's hold.
    fn drop(&mut self) {
        if self.borrowed.take().is_none() {
            let _ = OsVm::new().deallocate(HostPtr::from_ptr(self.at.as_ptr()), self.length);
        }
    }
}

/// Where host memory was handed to the runtime, as the trace names it ([`ImportTrace`]).
#[derive(Debug, Clone, Copy)]
pub(crate) enum ImportSite {
    /// A no-copy buffer over host memory core gives (`newBufferWithBytesNoCopy`).
    NoCopyBytes,
    /// A no-copy buffer a descriptor names (`newBufferWithDescriptor:`).
    NoCopyDescriptor,
    /// A buffer over a surface's alias.
    Surface,
    /// A heap over host memory (`new_heap_over_host_memory`).
    Heap,
    /// A buffer over a surface, at its base address.
    SurfaceBase,
}

impl ImportSite {
    /// Whether read-only memory handed to the runtime here is imported through a writable twin
    /// of its pages where the runtime cannot import them as mapped ([`crate::os::import_view`]),
    /// in place, its writes dying where they are encoded (`crate::gpu::writes::kept_write`): a
    /// buffer over a surface's alias, whose buffers then lie in one import of its pages, so a
    /// writer's GPU writes through another import reach its readers' reads, which a copy made at
    /// each commit would miss while that writer runs.
    ///
    /// Elsewhere the runtime copies such memory, dropping the GPU's writes, as Metal drops them:
    /// the guest's read-only objects (a no-copy buffer or heap made with bit 24), and a no-copy
    /// buffer or heap over pages mapped read-only.
    pub(crate) fn twins(self) -> bool {
        matches!(self, Self::Surface)
    }

    /// The site's number in the trace.
    fn code(self) -> u64 {
        match self {
            Self::NoCopyBytes => 1,
            Self::NoCopyDescriptor => 2,
            Self::Surface => 3,
            Self::Heap => 4,
            Self::SurfaceBase => 6,
        }
    }
}

/// The trace of the host memory this platform hands the runtime for one no-copy buffer or heap,
/// with `REIMS_VGPU_TRACE` set ([`crate::core::trace`]): `imp` as its keep-alive is
/// made (the site, the first byte given, its length, the first byte handed over), and `impfree`
/// as the keep-alive goes ([`keep_alive`]). It marks each buffer, not each import: a buffer
/// made over exactly the bytes a live no-copy buffer was made over lies in that one's import
/// (the runtime's `host_aliases.rs`), and its keep-alive goes at its release; the runtime's
/// `WorkCounts` count those (`host_keep_alives_covered`). A diagnostic: it holds nothing and
/// reads nothing with an effect, and the keep-alive drops it last.
pub(crate) struct ImportTrace(u64);

impl ImportTrace {
    pub(crate) fn new(
        site: ImportSite,
        given: NonNull<u8>,
        length: u64,
        imported: NonNull<u8>,
    ) -> Self {
        let at = imported.as_ptr().addr().to_u64();
        let given = given.as_ptr().addr().to_u64();
        crate::core::trace::event("imp", &[site.code(), given, length, at]);
        Self(at)
    }
}

impl Drop for ImportTrace {
    fn drop(&mut self) {
        crate::core::trace::event("impfree", &[self.0]);
    }
}

/// The runtime's keep-alive of `kept`, what keeps the memory of a no-copy buffer's or heap's
/// import mapped (an [`Alias`], a twin, an [`UnmapHold`](crate::remap::UnmapHold)): the runtime
/// drops it once it freed the device's import of the memory, after the deallocator ran, on
/// the thread that frees the import, which may be one of the runtime's; or, for a buffer whose
/// import keeps another buffer's keep-alive already (one that lies in a live buffer's import),
/// at the buffer's release, after its deallocator (the runtime's `host_memory.rs`, "A
/// keep-alive"). Its drop runs marked ours, so a panic there aborts.
pub(crate) fn keep_alive(kept: impl Send + 'static) -> rt::KeepAlive {
    /// `kept`, dropped marked ours.
    struct Kept<T>(Option<T>);
    impl<T> Drop for Kept<T> {
        fn drop(&mut self) {
            // A callout on the runtime's thread: marked ours, so a panic here aborts.
            let _env = BlockEnv::foreign();
            drop(self.0.take());
        }
    }
    Box::new(Kept(Some(kept)))
}

/// One remap of a gather: `length` bytes from `source`, at `offset` in the alias.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Run {
    /// Where in the alias.
    pub(crate) offset: u64,
    /// The first byte of the guest's pages.
    pub(crate) source: u64,
    /// The bytes, whole pages.
    pub(crate) length: u64,
}

/// The pages of `page` bytes `ranges` lie in, as Metal maps them (the module docs), each a first
/// byte and an end: ranges whose bytes follow on from the previous range's are one run of bytes,
/// and each run of bytes takes the pages it lies in, from the page of its first byte through the
/// page of its last; one of no bytes takes none. `None` past `u64::MAX`.
fn spans(ranges: &[ResourceAddressRange], page: u64) -> Option<Vec<std::ops::Range<u64>>> {
    let mut bytes: Vec<std::ops::Range<u64>> = Vec::new();
    for r in ranges {
        let end = r.address.checked_add(r.length)?;
        match bytes.last_mut() {
            Some(run) if run.end == r.address => run.end = end,
            _ => bytes.push(r.address..end),
        }
    }
    bytes
        .into_iter()
        .filter(|run| !run.is_empty())
        .map(|run| Some(run.start - run.start % page..run.end.checked_next_multiple_of(page)?))
        .collect()
}

/// The remaps of `ranges` into an alias of `size` bytes, whole pages of `page` bytes: the pages
/// of [`spans`] in order, up to `size`, one remap per run of them that follow on in the host's
/// memory. `None` for an empty list or a NULL range, which Metal refuses, and past `u64::MAX`. A
/// first range that starts off its page dies at `sel` ([`off_the_page`]).
pub(crate) fn runs(
    sel: &'static str,
    ranges: &[ResourceAddressRange],
    size: u64,
    page: u64,
) -> Option<Vec<Run>> {
    let first = ranges.first()?;
    if ranges.iter().any(|range| range.address == 0) {
        return None;
    }
    if first.address % page != 0 {
        off_the_page(sel, first.address, first.address % page);
    }
    let mut runs: Vec<Run> = Vec::new();
    let mut offset = 0u64;
    for span in spans(ranges, page)? {
        let length = (span.end - span.start).min(size.saturating_sub(offset));
        if length == 0 {
            break;
        }
        match runs.last_mut() {
            Some(run) if run.source.checked_add(run.length) == Some(span.start) => {
                run.length += length;
            }
            _ => runs.push(Run {
                offset,
                source: span.start,
                length,
            }),
        }
        offset += length;
    }
    Some(runs)
}

/// The death at `sel` of address ranges whose first starts at `address`, `into` bytes into its
/// page ([`runs`]). Metal's `contents` answers the range's bytes from its own address, and its
/// GPU reads and writes them from the start of its page, `into` bytes before it (measured on the
/// M3 Max, macOS 26.7: one range from 0x40 and from 0x1000 into its page, a heap's); a buffer of
/// the runtime's has one address for the CPU and the GPU, the first byte of the memory it
/// imports, so no alias of this platform's gives both.
fn off_the_page(sel: &'static str, address: u64, into: u64) -> ! {
    unrepresentable(
        sel,
        &format!(
            "address ranges whose first starts at {address:#x}, {into:#x} bytes into its page: \
             Metal's contents answers its bytes from that address, and its GPU reads and writes \
             them from the page's start, {into:#x} bytes before it; the runtime's buffer has one \
             address for the CPU and the GPU"
        ),
    )
}

/// The guest's pages of `ranges`, gathered into a fresh alias of `size` bytes rounded up to the
/// host's page ([`runs`]), zero-filled past the ranges' pages, read-only when `read_only` (module
/// docs). `None`, with nothing left mapped, for ranges Metal refuses (an empty list, a NULL or an
/// unmapped range, one with a page the host maps without read access, which the runtime refuses
/// to import, past `size` too). Where the host refuses the alias what it needs past those checks
/// (address space, a remap of the guest's pages, which on Linux takes pages with a file behind
/// them, the block of zeros, a protection), the call it is for, a buffer over a surface, answers
/// nil, as Metal's does when memory runs out, nothing left mapped
/// (`crate::answer::out_of_host_memory`, counted; in strict mode the process ends at `sel`).
///
/// Where the runtime imports a surface's read-only alias through a writable twin of its files
/// ([`crate::os::TWINS_READ_ONLY`], Linux), a read-only alias's zeros past the ranges' pages map
/// the block of zeros ([`map_zeros`]), as a twin can be made of a file alone: the twin's are
/// zeros of its own then ([`crate::os::import_view`]); an alias the runtime copies reads the
/// block's zeros. Elsewhere they are the fresh range's own,
/// which a writable alias with zeros past its ranges' pages takes from
/// [`crate::os::allocate_aliasable`]: on Linux a file of its own, as the runtime imports a range
/// either as host memory, which RADV refuses for any page with a file behind it, or as dma-bufs,
/// which udmabuf makes of files alone (`crate::os::host_memory`), so an alias of the guest's file
/// pages and anonymous zeros would import neither way; on Mach any allocation.
pub(crate) fn gather_or_nil(
    sel: &'static str,
    ranges: &[ResourceAddressRange],
    size: u64,
    read_only: bool,
) -> Option<Alias> {
    gather_short(sel, ranges, size, read_only).unwrap_or_else(|short| {
        crate::answer::out_of_host_memory(sel, || short.0);
        None
    })
}

/// What the host refused [`gather_or_nil`]'s alias past the checks of its ranges, where Metal's call
/// goes on: the line that names it.
#[derive(Debug)]
struct HostShort(String);

impl HostShort {
    /// The host's refusal of `what`, of `length` bytes, with `kr`.
    fn of(what: &str, length: u64, kr: KernReturn) -> HostShort {
        HostShort(format!(
            "the host refused {what} of {length:#x} bytes (kern_return {}), where Metal's call \
             goes on",
            kr.0
        ))
    }
}

/// [`gather_or_nil`], `Err` where the host refuses the alias what it needs, nothing left mapped.
fn gather_short(
    sel: &'static str,
    ranges: &[ResourceAddressRange],
    size: u64,
    read_only: bool,
) -> Result<Option<Alias>, HostShort> {
    let vm = OsVm::new();
    let page = vm.page_size();
    let Some(length) = size.checked_next_multiple_of(page) else {
        return Ok(None);
    };
    // Every page of the ranges, those past `size` too, as Metal refuses one there (measured);
    // checked before the alias takes address space, which a hole among the ranges could hold,
    // and before a refusal of the host's.
    let Some(spans) = spans(ranges, page) else {
        return Ok(None);
    };
    for span in spans {
        let Ok(start) = usize::try_from(span.start) else {
            return Ok(None);
        };
        if !vm.readable(HostPtr::from_addr(start), span.end - span.start) {
            return Ok(None);
        }
    }
    let Some(runs) = runs(sel, ranges, length, page) else {
        return Ok(None);
    };
    // The ranges' pages fill the alias from its start; zeros follow them.
    let pages = runs.last().map_or(0, |run| run.offset + run.length);
    let zero_block = read_only && crate::os::TWINS_READ_ONLY;
    let at = if pages < length && !zero_block {
        crate::os::allocate_aliasable(length)
    } else {
        vm.allocate_anywhere(length)
    }
    .map_err(|kr| HostShort::of("address space for an alias", length, kr))?;
    let Some(first) = NonNull::new(at.as_ptr::<u8>()) else {
        return Ok(None);
    };
    // Unmapped as it drops, so nothing is left mapped where the host refuses the rest.
    let alias = Alias::own(first, length);
    // The runs follow on from the alias's start ([`runs`]), so they are one view, made at once:
    // on Linux through dma-bufs of the pages where the host makes them (`crate::os`'s VM).
    let mut sources = Vec::with_capacity(runs.len());
    for run in &runs {
        let Ok(source) = usize::try_from(run.source) else {
            return Ok(None);
        };
        sources.push((HostPtr::from_addr(source), run.length));
    }
    if let Err((i, kr)) = vm.remap_runs_shared_overwrite(at, &sources) {
        let what = "a remap of the guest's pages into an alias";
        return Err(HostShort::of(what, runs[i].length, kr));
    }
    if read_only {
        if zero_block {
            map_zeros(&vm, at, pages, length)
                .map_err(|ZerosRefused { what, length, kr }| HostShort::of(what, length, kr))?;
        }
        if let Err(kr) = vm.protect_max(at, length, VmProt::READ) {
            return Err(HostShort::of("an alias's read-only protection", length, kr));
        }
    }
    Ok(Some(alias))
}

/// The bytes of [`zero_block`]'s block: 1 MiB, whole pages on every host.
const ZERO_BLOCK: u64 = 0x10_0000;

/// A block of [`ZERO_BLOCK`] bytes of zeros, read-only, made once and never freed, that a
/// read-only alias's zeros past its ranges' pages map where the runtime twins read-only memory
/// ([`map_zeros`]); `Err` with the host's answer where it gave none, which a later call asks
/// again, as the host may have room by then. A view can be made of it
/// ([`crate::os::allocate_aliasable`]).
fn zero_block() -> Result<HostPtr, KernReturn> {
    static BLOCK: Mutex<Option<usize>> = Mutex::new(None);
    let mut block = BLOCK.lock();
    if let Some(at) = *block {
        return Ok(HostPtr::from_addr(at));
    }
    let vm = OsVm::new();
    let at = crate::os::allocate_zeros(ZERO_BLOCK)?;
    if let Err(kr) = vm.protect_max(at, ZERO_BLOCK, VmProt::READ) {
        let _ = vm.deallocate(at, ZERO_BLOCK);
        return Err(kr);
    }
    *block = Some(at.addr());
    Ok(at)
}

/// What the host refused [`map_zeros`]: what, of how many bytes, and its answer.
#[derive(Debug)]
pub(crate) struct ZerosRefused {
    pub(crate) what: &'static str,
    pub(crate) length: u64,
    pub(crate) kr: KernReturn,
}

/// Maps the block of [`zero_block`] over the bytes from `from` to `length` of the range at `at`,
/// whole pages, read-only: one remap maps the block, and each further one doubles what is mapped.
/// Where the runtime imports a surface's read-only alias through a writable twin
/// ([`crate::os::TWINS_READ_ONLY`]), the twin of these is zeros of its own
/// ([`crate::os::import_view`]). `Err` where the host gave no block or refused a remap, what it
/// mapped left as it is.
pub(crate) fn map_zeros(
    vm: &OsVm,
    at: HostPtr,
    from: u64,
    length: u64,
) -> Result<(), ZerosRefused> {
    if from >= length {
        return Ok(());
    }
    let block = zero_block().map_err(|kr| ZerosRefused {
        what: "the block of zeros",
        length: ZERO_BLOCK,
        kr,
    })?;
    let (first, length) = (at.wrapping_add(from), length - from);
    let mut mapped = length.min(ZERO_BLOCK);
    if let Err(kr) = vm.remap_shared_overwrite(first, block, mapped) {
        return Err(ZerosRefused {
            what: "a remap of the block of zeros",
            length: mapped,
            kr,
        });
    }
    while mapped < length {
        let more = mapped.min(length - mapped);
        if let Err(kr) = vm.remap_shared_overwrite(first.wrapping_add(mapped), first, more) {
            return Err(ZerosRefused {
                what: "a remap of zeros",
                length: more,
                kr,
            });
        }
        mapped += more;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::os::PAGE;

    /// The selector the tests' gathers are for.
    const SEL: &str = "newBufferWithIOSurface:";

    fn range(address: u64, length: u64) -> ResourceAddressRange {
        ResourceAddressRange { address, length }
    }

    fn run(offset: u64, source: u64, length: u64) -> Run {
        Run {
            offset,
            source,
            length,
        }
    }

    /// Fresh pages of the host a view can be made of, each byte `byte`.
    fn pages(count: u64, byte: u8) -> HostPtr {
        let at = crate::os::allocate_aliasable(count * PAGE).unwrap();
        // SAFETY: the pages were just allocated, readable and writable.
        unsafe {
            std::ptr::write_bytes(
                at.as_ptr::<u8>(),
                byte,
                usize::try_from(count * PAGE).unwrap(),
            );
        };
        at
    }

    fn read(at: NonNull<u8>, offset: u64) -> u8 {
        // SAFETY: test memory this test mapped readable.
        unsafe {
            at.as_ptr()
                .add(usize::try_from(offset).unwrap())
                .read_volatile()
        }
    }

    fn write(at: HostPtr, offset: u64, byte: u8) {
        // SAFETY: test memory this test mapped writable.
        unsafe { at.wrapping_add(offset).as_ptr::<u8>().write_volatile(byte) }
    }

    fn address(at: HostPtr) -> u64 {
        u64::try_from(at.addr()).unwrap()
    }

    /// Host-contiguous ranges make one run, as core cuts its runs at two pages; the ranges'
    /// bytes are cut at the size, and a short last range is remapped to its page's end
    #[test]
    fn runs_join_contiguous_ranges_up_to_the_size() {
        let at = 0x10_0000;
        let ranges = [
            range(at, PAGE),
            range(at + PAGE, PAGE),
            range(at + 0x10_0000, PAGE),
        ];
        assert_eq!(
            runs(SEL, &ranges, 3 * PAGE, PAGE),
            Some(vec![
                run(0, at, 2 * PAGE),
                run(2 * PAGE, at + 0x10_0000, PAGE)
            ])
        );
        assert_eq!(runs(SEL, &ranges, PAGE, PAGE), Some(vec![run(0, at, PAGE)]));
        assert_eq!(
            runs(SEL, &[range(at, 0x10)], PAGE, PAGE),
            Some(vec![run(0, at, PAGE)])
        );
        assert_eq!(runs(SEL, &[], PAGE, PAGE), None);
        assert_eq!(
            runs(SEL, &[range(at, PAGE), range(0, PAGE)], 2 * PAGE, PAGE),
            None
        );
    }

    /// Each run of ranges whose bytes follow on takes the pages it lies in, whole, as Metal maps
    /// them: a range of part of a page all of it, before another too, and a later range starting
    /// off its page from the page's start, through the page of its last byte; up to the size,
    /// those that follow on in the host's memory one remap
    #[test]
    fn runs_take_the_pages_each_run_of_bytes_lies_in() {
        let g = 0x10_0000;
        let cases = [
            // A range of a quarter page before one of two pages: its page, then the next one's.
            (
                vec![range(g, PAGE / 4), range(g + 2 * PAGE, 2 * PAGE)],
                2 * PAGE,
                vec![run(0, g, PAGE), run(PAGE, g + 2 * PAGE, PAGE)],
            ),
            // A range of a page and 0x40 bytes: two pages.
            (
                vec![range(g, PAGE + 0x40), range(g + 3 * PAGE, PAGE)],
                2 * PAGE,
                vec![run(0, g, 2 * PAGE)],
            ),
            // A later range from 0x40 into its page: from the page's start.
            (
                vec![range(g, PAGE), range(g + 2 * PAGE + 0x40, PAGE)],
                2 * PAGE,
                vec![run(0, g, PAGE), run(PAGE, g + 2 * PAGE, PAGE)],
            ),
            // The same before a third range: through the page of its last byte.
            (
                vec![
                    range(g, PAGE),
                    range(g + 2 * PAGE + 0x40, PAGE),
                    range(g + 5 * PAGE, PAGE),
                ],
                3 * PAGE,
                vec![run(0, g, PAGE), run(PAGE, g + 2 * PAGE, 2 * PAGE)],
            ),
            // Ranges whose bytes follow on across a page: one run of bytes.
            (
                vec![
                    range(g, PAGE + 0x40),
                    range(g + PAGE + 0x40, 2 * PAGE - 0x40),
                ],
                3 * PAGE,
                vec![run(0, g, 3 * PAGE)],
            ),
            // A middle range of a quarter page from its page's start.
            (
                vec![
                    range(g, PAGE),
                    range(g + 2 * PAGE, PAGE / 4),
                    range(g + 4 * PAGE, PAGE),
                ],
                3 * PAGE,
                vec![
                    run(0, g, PAGE),
                    run(PAGE, g + 2 * PAGE, PAGE),
                    run(2 * PAGE, g + 4 * PAGE, PAGE),
                ],
            ),
            // Ranges that do not follow on but share a page: the page twice.
            (
                vec![range(g, PAGE / 4), range(g + PAGE / 2, PAGE)],
                3 * PAGE,
                vec![run(0, g, PAGE), run(PAGE, g, 2 * PAGE)],
            ),
            // A range of part of a page before the next page's range: one remap.
            (
                vec![range(g, PAGE / 4), range(g + PAGE, PAGE)],
                2 * PAGE,
                vec![run(0, g, 2 * PAGE)],
            ),
            // A range of no bytes takes nothing.
            (
                vec![
                    range(g, PAGE),
                    range(g + 0x40 + 4 * PAGE, 0),
                    range(g + 2 * PAGE, PAGE),
                ],
                2 * PAGE,
                vec![run(0, g, PAGE), run(PAGE, g + 2 * PAGE, PAGE)],
            ),
        ];
        for (ranges, size, expected) in cases {
            assert_eq!(
                runs(SEL, &ranges, size, PAGE),
                Some(expected),
                "{ranges:x?}"
            );
        }
    }

    /// A first range starting off its page dies at the creator, naming where it starts and what
    /// Metal's CPU and GPU see; ranges Metal refuses (a NULL one, an unmapped page) are refused
    /// first, as Metal makes nothing of them
    #[test]
    fn a_first_range_off_its_page_dies() {
        use crate::core::raise::death::DeathKind;
        use crate::core::testkit::capture::catch_death;
        let guest = pages(3, 0);
        for (into, sel) in [
            (0x40, "newBufferWithIOSurface:"),
            (PAGE / 4, "newBufferWithIOSurface:"),
        ] {
            let ranges = [range(address(guest) + into, 2 * PAGE)];
            *crate::answer::LAST_LINE.lock() = None;
            let site =
                catch_death(|| gather_or_nil(sel, &ranges, 2 * PAGE, false).is_some()).unwrap_err();
            let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
            assert_eq!((site.kind, site.file), (DeathKind::Abort, sel), "{line}");
            assert!(
                line.contains(&format!("{into:#x} bytes into its page"))
                    && line.contains("Metal's contents answers its bytes from that address"),
                "{line}"
            );
        }
        let null = range(0, PAGE);
        let off = range(address(guest) + 0x40, PAGE);
        assert_eq!(runs(SEL, &[off, null], 2 * PAGE, PAGE), None);
        let vm = OsVm::new();
        assert!(
            vm.deallocate(guest.wrapping_add(2 * PAGE), PAGE)
                .is_success()
        );
        let unmapped = [range(address(guest) + 0x40, 2 * PAGE)];
        assert!(gather_or_nil(SEL, &unmapped, 2 * PAGE, false).is_none());
        assert!(vm.deallocate(guest, 2 * PAGE).is_success());
    }

    /// The alias shows the guest's pages, in order, and the guest's pages its writes; past the
    /// ranges it is zero-filled pages of its own
    #[test]
    fn an_alias_is_the_guests_pages_both_ways() {
        let guest = pages(4, 0xa1);
        write(guest, 3 * PAGE, 0xb2);
        let ranges = [
            range(address(guest), PAGE),
            range(address(guest) + PAGE, PAGE),
            range(address(guest) + 3 * PAGE, PAGE),
        ];
        let alias = gather_or_nil(SEL, &ranges, 4 * PAGE, false).unwrap();
        let first = alias.first();
        assert_eq!(
            [0, PAGE, 2 * PAGE, 3 * PAGE].map(|at| read(first, at)),
            [0xa1, 0xa1, 0xb2, 0]
        );
        write(guest, 5, 0x5c);
        assert_eq!(read(first, 5), 0x5c, "the guest's write shows in the alias");
        write(HostPtr::from_ptr(first.as_ptr()), 2 * PAGE + 7, 0x6d);
        // SAFETY: the guest's pages are mapped.
        let seen = unsafe {
            guest
                .wrapping_add(3 * PAGE + 7)
                .as_ptr::<u8>()
                .read_volatile()
        };
        assert_eq!(seen, 0x6d, "the alias's write shows in the guest's page");
        drop(alias);
        assert!(OsVm::new().deallocate(guest, 4 * PAGE).is_success());
    }

    /// A NULL or unmapped range, one with a page mapped without access, within the size or past
    /// it, or none, gathers nothing; read-only pages gather
    #[test]
    fn refused_ranges_gather_nothing() {
        use crate::core::platform::os::vm::VmProt;
        let vm = OsVm::new();
        // A hole of one page between two mapped ones, where no alias of two pages fits.
        let around = pages(3, 0);
        let hole = around.wrapping_add(PAGE);
        assert!(vm.deallocate(hole, PAGE).is_success());
        let shut = pages(2, 0);
        vm.protect_max(shut.wrapping_add(PAGE), PAGE, VmProt(0))
            .unwrap();
        // A page unmapped before free address space, where the alias would fit.
        let last = pages(2, 0);
        assert!(vm.deallocate(last.wrapping_add(PAGE), PAGE).is_success());
        for ranges in [
            vec![],
            vec![range(0, PAGE)],
            vec![range(address(around), PAGE), range(address(hole), PAGE)],
            vec![range(address(shut), 2 * PAGE)],
            vec![range(address(last), 2 * PAGE)],
        ] {
            assert!(
                gather_or_nil(SEL, &ranges, 2 * PAGE, false).is_none(),
                "{ranges:?}"
            );
        }
        // Past the size, as within it.
        for ranges in [
            vec![range(address(shut), 2 * PAGE)],
            vec![
                range(address(shut), PAGE),
                range(address(shut) + PAGE, PAGE),
            ],
            vec![range(address(last), 2 * PAGE)],
        ] {
            assert!(
                gather_or_nil(SEL, &ranges, PAGE, false).is_none(),
                "{ranges:?}"
            );
        }
        // A later range that ends in its page before an unmapped one: only its page is read.
        let ends = [
            range(address(around), PAGE),
            range(address(last) + 0x40, PAGE - 0x40),
        ];
        assert!(gather_or_nil(SEL, &ends, 2 * PAGE, false).is_some());
        let read_only = pages(1, 0);
        vm.protect_max(read_only, PAGE, VmProt::READ).unwrap();
        assert!(gather_or_nil(SEL, &[range(address(read_only), PAGE)], PAGE, false).is_some());
        for page in [around, around.wrapping_add(2 * PAGE), last, read_only] {
            assert!(vm.deallocate(page, PAGE).is_success());
        }
        assert!(vm.deallocate(shut, 2 * PAGE).is_success());
    }

    /// A read-only alias reads the guest's pages, and zeros past them, and its pages cannot be
    /// made writable
    #[test]
    fn a_read_only_alias_reads_zeros_past_its_pages() {
        let guest = pages(1, 0xa5);
        let alias = gather_or_nil(SEL, &[range(address(guest), PAGE)], 3 * PAGE, true).unwrap();
        assert_eq!(alias.length, 3 * PAGE);
        assert_eq!(read(alias.at, 0), 0xa5);
        for offset in [PAGE, 2 * PAGE + 5, 3 * PAGE - 1] {
            assert_eq!(read(alias.at, offset), 0, "{offset:#x}");
        }
        let at = HostPtr::from_ptr(alias.at.as_ptr());
        assert!(!crate::os::writable(at, alias.length), "writable");
        drop(alias);
        assert!(OsVm::new().deallocate(guest, PAGE).is_success());
    }
}
