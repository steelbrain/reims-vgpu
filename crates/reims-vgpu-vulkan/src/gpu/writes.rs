//! What GPU work may write through read-only memory: whether an object's memory is read-only where
//! the runtime does not answer the GPU's writes through it as Metal's GPU does ([`Kept`]).
//!
//! Read-only memory ([`Kept`]) on Linux is that of **the guest's read-only objects** (a
//! no-copy buffer or heap made with `MTLResourceOptions` bit 24 or over pages mapped read-only;
//! [`Kept::Copied`]; [`crate::os::COPIES_READ_ONLY`]), which the runtime makes over a copy of the
//! pages whose GPU writes it drops: every GPU write there is sent and completes, its write dropped
//! (a shader's store through a slot or a loaded address, a draw's or a kernel's write of a
//! texture, a blit's fill, copy or mipmaps, a pass's visibility results, an attachment the pass
//! does not store, a resolve the pass does not make), but a render pass's store-out, which fails
//! its command buffer with Metal's page fault, as the runtime faults it itself, the fault counting
//! against its queue as Metal's does: the queue's second revokes it (`crate::gpu::encode_render`,
//! `crate::gpu::command`). A store-out is an attachment stored (`Store`,
//! `StoreAndMultisampleResolve`, and a depth attachment's `CustomSampleDepthStore`), or a resolve
//! into such a texture (a colour or stencil attachment's `MultisampleResolve` or
//! `StoreAndMultisampleResolve`, a depth attachment's `MultisampleResolve`, into the descriptor's
//! resolve texture or a late setter's). A depth attachment's `StoreAndMultisampleResolve` into such
//! a texture completes, its resolve dropped and the texture unchanged, as Metal's GPU completes it,
//! but beside a stencil attachment that resolves, a pairing nothing measured, which dies where it
//! is encoded ([`kept_write`]). A CPU write of a texture over such memory (`replaceRegion:`)
//! writes the pages where the process can write them, and dies over pages it maps read-only, as
//! Metal's ends the process there (`crate::gpu::resource`).
//!
//! On macOS it is not taken: MoltenVK hands Metal the pages, or a read-only view of them made
//! with bit 24, and Metal's GPU answers every write there itself.
//!
//! What Metal does there, measured by metal2vulkan's probes (`read_only_view.rs`,
//! `probes/readonly_gpu_writes.m`) on the M3 Max (macOS 27.0.1), 2026-10-25 and since, and by
//! this platform's probe of every write it answered by ending the process (2026-11-14, plain
//! Metal, one shape a process, against a writable control) on the M3 Max and on the dGPU oracle's
//! Radeon Pro 5300M and UHD Graphics 630 (macOS 26.7.1, the x86 guests' line; m3-max-29292 to
//! m3-max-29294 and m3-max-29309 to m3-max-29311; `docs/graphics-kext/dgpu-oracle.md`):
//! - **On the M3 Max** bit 24, and pages the process maps read-only alike, keep the GPU's writes
//!   out. It completes a draw or a dispatch whose shader stores may write through such memory
//!   (through a bound slot or a loaded address, `drawIndexedPrimitives` among them), a kernel's
//!   `write()` of a texture there, a blit's fill, copy, colour fill or mipmaps into it and a pass
//!   that keeps its visibility results in it, and drops those writes, over 8,314 runs and every
//!   run since. A pass's store-out into it fails the command buffer
//!   (`MTLCommandBufferErrorPageFault`) every run, its signal firing, the next command buffer of
//!   its queue and of a new queue running, and every command buffer of the queue after its second
//!   fault failing (`MTLCommandBufferErrorAccessRevoked`; m3-max-29110, m3-max-28796,
//!   m3-max-28797, m3-max-29292, m3-max-29309; the runtime's `command_buffer/revocation.rs`), and
//!   so does a fragment or vertex function's `write()` of a texture there. A depth attachment's
//!   `StoreAndMultisampleResolve` into it completes (4 runs of 6; the other 2 failing as innocent
//!   victims of a recovery, code 1), the resolve texture unchanged. The same attachments under
//!   `DontCare` or `Load` complete, as does a resolve texture attached and not resolved into, a
//!   resolve out of a read-only primary and a resolve of a format pair Metal leaves untouched.
//! - **On the Radeon Pro 5300M and the UHD Graphics 630** bit 24 changes nothing: every write
//!   above lands in the pages and every command buffer completes, as without the bit. Over pages
//!   the process maps read-only, every GPU write measured (a pass's store and resolves, a draw's
//!   and a kernel's texture writes, a blit's copy) completes, the pages unchanged.
//! - **On all three** a CPU `replaceRegion:` of a texture over a bit-24 buffer writes the pages,
//!   and of one over pages the process maps read-only ends the process with a bus error, with the
//!   bit or without; and a colour or stencil attachment that the pass descriptor keeps by
//!   `CustomSampleDepthStore` ends the process in the descriptor's validation, with the bit or
//!   without.
//!
//! Where these GPUs differ, the envelope holds each answer, and this platform takes the cheaper:
//! the runtime's fault at a store-out (the M3 Max's), and completion with the writes dropped
//! elsewhere.
use crate::gpu::resource::{BufferObject, TextureObject, Under};

/// Whether an object's memory is read-only where the runtime does not answer the GPU's writes
/// through it as Metal's GPU does (the module docs), asked as work that may write through the
/// object is encoded ([`Kept::holds`]).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum Kept {
    /// It is not.
    #[default]
    No,
    /// For the object's life: for a texture over a buffer or heap, read-only memory the runtime
    /// makes the buffer or heap over a copy of, dropping the GPU's writes
    /// ([`crate::gpu::backing::Backing::read_only_now`]): a render pass's store or resolve into
    /// it may fault the command buffer as Metal's GPU does (`crate::gpu::encode_render`), and
    /// every other GPU write of it is sent and dropped. A buffer's own memory the runtime copies
    /// is never `Copied`: the runtime drops its writes, as Metal's GPU does.
    Copied,
}

impl Kept {
    /// Whether the object's memory is read-only so now.
    pub fn holds(&self) -> bool {
        matches!(self, Kept::Copied)
    }
}

/// Whether the writes through `t` are not answered as Metal's GPU answers them, its memory being
/// read-only ([`Kept`]): that of the buffer `t` lies over, or of its heap, or of `t`'s heap,
/// which the runtime copies.
pub fn texture_kept(t: &TextureObject) -> bool {
    match &t.under {
        Some(Under::Buffer(b, _)) => under_buffer_kept(b).holds(),
        Some(Under::Heap(h, _)) => h.backing.read_only_now(),
        None => false,
    }
}

/// Whether `t` lies over read-only memory the runtime copies whose pages are mapped read-only
/// ([`crate::gpu::backing::Backing::pages_read_only`]): the memory of the buffer `t` lies over,
/// or of that buffer's heap, or of `t`'s heap.
pub fn texture_pages_read_only(t: &TextureObject) -> bool {
    match &t.under {
        Some(Under::Buffer(b, _)) => match &b.heap {
            Some(h) => h.backing.pages_read_only(),
            None => b.backing.pages_read_only(),
        },
        Some(Under::Heap(h, _)) => h.backing.pages_read_only(),
        None => false,
    }
}

/// What a texture over `b` keeps of `b`'s memory ([`Kept`]): the read-only memory the runtime
/// makes `b`, or the heap it lies in, over a copy of.
fn under_buffer_kept(b: &BufferObject) -> Kept {
    match &b.heap {
        Some(h) => h.backing.kept(),
        None => b.backing.kept(),
    }
}

/// Dies at `sel`, a command that may write `what` an object over read-only memory whose write
/// this platform does not answer as Metal's GPU does ([`Kept`]): a depth attachment's
/// `StoreAndMultisampleResolve` into a texture over read-only guest pages (a read-only heap, a
/// resource made with `MTLResourceOptions` bit 24) the runtime copies, beside a stencil
/// attachment that resolves, which the runtime faults where nothing measured Metal's GPU, and a
/// late resolve texture there of an attachment the pass lacks (`crate::gpu::encode_render`'s
/// `stencil_may_resolve` and `late_resolve`). Ending the process is this platform's divergence
/// from Metal (`docs/accuracy-ledger.md`, `read-only-pages-written`).
pub(crate) fn kept_write(sel: &'static str, what: &str) -> ! {
    crate::answer::unrepresentable(
        sel,
        &format!(
            "{what} read-only memory (a read-only heap or resource the runtime copies), where \
             nothing measured what Metal's GPU does"
        ),
    )
}

#[cfg(test)]
mod tests {
    /// A buffer or heap over read-only memory the runtime copies is read-only while its memory is,
    /// set so again or not, and no longer once it is not. Of such memory, only pages mapped
    /// read-only refuse a CPU write, not memory of bit 24 alone. Linux's and Windows's: on macOS
    /// the runtime copies none
    #[cfg(any(target_os = "linux", windows))]
    #[test]
    fn backings_over_copied_memory_are_read_only_while_it_is() {
        use crate::gpu::backing::Backing;
        let (a, b) = (Backing::new(false), Backing::new(false));
        a.imported(false, false);
        assert!(!a.read_only_now());
        a.imported(true, false);
        a.imported(true, false);
        assert!(a.read_only_now() && a.pages_read_only());
        b.imported(false, true);
        assert!(b.read_only_now() && !b.pages_read_only());
        a.imported(false, false);
        a.imported(false, false);
        assert!(!a.read_only_now() && !a.pages_read_only());
        let bit_24 = Backing::new(true);
        bit_24.imported(false, false);
        assert!(bit_24.read_only_now() && !bit_24.pages_read_only());
    }
}
