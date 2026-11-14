//! The read-only bit each buffer and heap keeps ([`Backing`]), and what this platform hands the
//! runtime with a no-copy buffer's or heap's memory: the trace of where it was handed
//! ([`ImportSite`], [`ImportTrace`]), and the keep-alive that holds what keeps it mapped
//! ([`keep_alive`]).
//!
//! **Read-only.** On Linux the runtime makes a no-copy buffer or heap made with
//! `MTLResourceOptions` bit 24 or over pages mapped read-only over a copy of them, whose GPU
//! writes it drops, as the views' gate refuses read-only pages (`crate::os::import_view`): a
//! shader's, a blit's into a buffer and a visibility result's are sent, and dropped as Metal's
//! are, while a pass's store into the object's memory and a write of a texture over it are
//! answered where they are encoded ([`Backing::read_only_now`], `crate::gpu::writes`). The
//! runtime keeps the pages a range had when it imported them (`tests/no_copy_remap.rs`).

use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::core::num::ToU64;
use crate::core::rt::BlockEnv;
use metal2vulkan_runtime as rt;

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
    /// buffer or heap was made over, mapped read-only or made with bit 24.
    kept: AtomicBool,
    /// Whether, of that memory, the pages are mapped read-only, which a CPU write through them
    /// cannot write ([`Backing::pages_read_only`]).
    mapped_read_only: AtomicBool,
}

impl Backing {
    /// The backing of a buffer or heap over the memory it was made with; `read_only` for one
    /// made with `MTLResourceOptions` bit 24.
    pub(crate) fn new(read_only: bool) -> Backing {
        Backing {
            read_only,
            kept: AtomicBool::new(false),
            mapped_read_only: AtomicBool::new(false),
        }
    }

    /// The object, a no-copy buffer or heap, was made over memory of the caller's whose pages are
    /// mapped read-only where `mapped_read_only` ([`crate::os::ImportView::read_only`]), with
    /// `MTLResourceOptions` bit 24 where `bit_24` or [`Backing::new`] was told so: its memory is
    /// read-only where either holds ([`Backing::read_only_now`]).
    pub(crate) fn imported(&self, mapped_read_only: bool, bit_24: bool) {
        let read_only = mapped_read_only || bit_24 || self.read_only;
        let kept = crate::os::COPIES_READ_ONLY && read_only;
        (self.mapped_read_only).store(kept && mapped_read_only, Ordering::Release);
        self.kept.store(kept, Ordering::Release);
    }

    /// Whether the memory the object lies in is read-only memory the runtime copies
    /// ([`Backing::read_only_now`]) whose pages are mapped read-only: a CPU write of a texture
    /// over it, which the runtime makes through the pages, dies, as Metal's CPU write through
    /// such a mapping ends the process (`crate::gpu::resource`).
    pub(crate) fn pages_read_only(&self) -> bool {
        self.mapped_read_only.load(Ordering::Acquire)
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

/// Where host memory was handed to the runtime, as the trace names it ([`ImportTrace`]).
#[derive(Debug, Clone, Copy)]
pub(crate) enum ImportSite {
    /// A no-copy buffer over host memory core gives (`newBufferWithBytesNoCopy`).
    NoCopyBytes,
    /// A no-copy buffer a descriptor names (`newBufferWithDescriptor:`).
    NoCopyDescriptor,
    /// A heap over host memory (`new_heap_over_host_memory`).
    Heap,
}

impl ImportSite {
    /// The site's number in the trace.
    fn code(self) -> u64 {
        match self {
            Self::NoCopyBytes => 1,
            Self::NoCopyDescriptor => 2,
            Self::Heap => 4,
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
/// import mapped (a twin, an [`UnmapHold`](crate::remap::UnmapHold)): the runtime
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
