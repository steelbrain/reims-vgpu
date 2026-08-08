//! Resolving a byte span to the guest pages under it.

use crate::resolve::{read_task_root, translate_root_run, Geometry, ResolveStatus, Task};
use alloc::vec::Vec;
use reims_vgpu_wire::mem::GuestMemory;

/// How many guest pages `[gva, gva+span)` touches, given `page_size`.
///
/// The `gva % page_size` term is the whole content: a span that starts
/// mid-page reaches one page further than its length alone implies. Callers
/// compare a walk's result against this to decide whether the *whole* span
/// resolved, and getting it wrong reads as "fully covered" for exactly the
/// windows that straddle a page boundary — which is most of them.
pub fn pages_spanned(gva: u64, span: u64, page_size: u64) -> u64 {
    ((gva % page_size) + span).div_ceil(page_size)
}

/// Every guest page of `[gva, gva + span)` under `task`'s page table, in
/// ascending order, as its **page-aligned** GPA.
///
/// The one spelling of "walk this span". Four rails in the device need it and
/// each used to open with the same five steps — pick the geometry, build the
/// task, read the root, refuse a root or depth of zero, count the pages — with
/// the refusals written out by hand at each one. That is the shape where a
/// missing term hides: three of the four carried the zero-root guard and the
/// fourth did not, which is correct for that one and was impossible to see
/// without reading all four together.
///
/// # Two kinds of failure, and why they are returned differently
///
/// A **setup** failure — an unusable geometry, an inactive task, an unreadable
/// directory, a zero root or a zero depth — means no page of the span can
/// resolve, and it comes back as `Err`. The visitor is not called at all, so a
/// caller cannot mistake "nothing resolved" for "the span was empty".
///
/// A **per-page** failure reaches the visitor as its own `Err`, because which
/// page failed is the finding: a caller checking a cached page list against the
/// live table needs the position, and one that is merely reading needs to stop
/// there. Walking on past it is the visitor's choice, exactly as with a
/// resolved page.
///
/// The zero-root and zero-depth refusals are the reason this returns a
/// `Result` at all. [`translate_root_run`] answers both by visiting nothing,
/// which is indistinguishable at the call site from a span that resolved
/// cleanly and had no pages — so every caller that reads bytes has to turn them
/// into a refusal, and now does it here once.
pub fn walk_span(
    mem: &dyn GuestMemory,
    geometry: Geometry,
    task: &Task,
    gva: u64,
    span: u64,
    visit: &mut dyn FnMut(u64, Result<u64, ResolveStatus>) -> bool,
) -> Result<(), ResolveStatus> {
    if geometry.validate().is_err() {
        return Err(ResolveStatus::ErrUnsupportedGeometry);
    }
    let root = read_task_root(mem, task, geometry)?;
    if root.root_pfn == 0 {
        return Err(ResolveStatus::ErrZeroRootPfn);
    }
    if root.depth == 0 {
        return Err(ResolveStatus::ErrZeroDepth);
    }
    if span == 0 {
        return Ok(());
    }
    let page = geometry.page_size();
    let mask = geometry.page_offset_mask();
    // Walk from the span's first page rather than from `gva`, so every page's
    // answer is its own base. The run walker carries the starting offset onto
    // every page it reports, which a caller reading bytes has to undo.
    translate_root_run(
        mem,
        geometry,
        root.root_pfn,
        root.depth,
        gva & !mask,
        pages_spanned(gva, span, page),
        &mut |index, r| visit(index, r.map(|gpa| gpa & !mask)),
    );
    Ok(())
}

/// Why a span did not resolve in full, keeping **where** it refused.
///
/// [`walk_span`] already separates the two — a setup refusal is its `Err` and a
/// per-page refusal reaches its visitor — and every fail-closed caller then
/// flattens them back into one value. They must not flatten to the *same* one:
/// a setup refusal means the walk never reached a page table, so most of its
/// statuses are "the directory did not read" rather than "this address does not
/// translate", and a caller that reports them as an unresolved address names a
/// check that never ran.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpanRefusal {
    /// The walk could not start: an unusable geometry, an inactive task, an
    /// unreadable directory, a zero root PFN or a zero depth. No page of the
    /// span can resolve.
    Setup(ResolveStatus),
    /// The walk ran and a page of the span did not translate.
    Page(ResolveStatus),
}

/// One page's worth of a byte span: the guest-physical bytes it lands on, and
/// which bytes of the caller's buffer they are.
///
/// `gpa` is the page base **plus** the span's offset within that page, so it is
/// the address to read or write directly — unlike [`walk_span`], which reports
/// page bases because its callers index pages rather than bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpanChunk {
    /// First guest-physical byte of this chunk.
    pub gpa: u64,
    /// Offset of this chunk in the caller's buffer.
    pub offset: usize,
    /// Length of the chunk. Never crosses the end of its page.
    pub len: usize,
}

impl SpanChunk {
    /// The chunk's bytes within a caller's buffer, as a range to index with.
    #[inline]
    pub fn range(self) -> core::ops::Range<usize> {
        self.offset..self.offset + self.len
    }
}

/// Cut `[gva, gva+len)` into one [`SpanChunk`] per guest page it touches, in
/// order, stopping at the first page that does not resolve.
///
/// # Why the cutting is here and the transfer is not
///
/// This is the arithmetic every byte-level guest access owes and none of them
/// can see whole: a span starts mid-page, so the first chunk is short, and the
/// bytes left in the current page — not the bytes left in the buffer — bound
/// every chunk after it. It was written twice in the device, once for reads and
/// once for writes, in the same four lines each time, and a mistake in it does
/// not fail: it reads or writes the right number of bytes at an address that
/// belongs to the neighbouring page.
///
/// What stays with the caller is the transfer, because the seam here
/// ([`GuestMemory`]) only reads and only answers `bool`, while the device's own
/// host access reads *and* writes and names **which** transaction failed. A
/// copy loop moved in here would have to flatten that to "a byte did not move".
/// So the caller keeps its typed error and its `&mut` host, and takes the
/// arithmetic from here.
///
/// The visitor answering `false` stops the walk, and a stopped walk is `Ok`:
/// stopping is the caller's decision, not a refusal.
pub fn visit_span_chunks(
    mem: &dyn GuestMemory,
    geometry: Geometry,
    task: &Task,
    gva: u64,
    len: usize,
    visit: &mut dyn FnMut(SpanChunk) -> bool,
) -> Result<(), SpanRefusal> {
    if len == 0 {
        return Ok(());
    }
    let page = geometry.page_size();
    let mask = geometry.page_offset_mask();
    let mut done = 0usize;
    let mut refused = None;
    walk_span(mem, geometry, task, gva, len as u64, &mut |_, r| {
        let page_base = match r {
            Ok(base) => base,
            Err(status) => {
                refused = Some(status);
                return false;
            }
        };
        // `done` advances by whole chunks, so the offset within the page is the
        // span's own start offset on the first page and zero on every page
        // after it. Deriving it from the running position rather than tracking
        // it separately is what keeps the two consistent.
        let cur = gva.saturating_add(done as u64);
        let in_page = cur & mask;
        let n = (len - done).min((page - in_page) as usize);
        let chunk = SpanChunk {
            gpa: page_base + in_page,
            offset: done,
            len: n,
        };
        done += n;
        visit(chunk)
    })
    .map_err(SpanRefusal::Setup)?;
    match refused {
        Some(status) => Err(SpanRefusal::Page(status)),
        None => Ok(()),
    }
}

/// [`visit_span_chunks`] collected, for a caller that cannot transfer while it
/// walks.
///
/// A guest **write** is exactly that caller: the walk reads page tables through
/// a shared borrow of the host and the write needs an exclusive one, so the two
/// cannot interleave and the whole span has to be resolved before the first
/// byte moves. That is not a limitation to work around — it is the same
/// resolve-then-write order a write owes anyway, since a span that refuses on
/// its last page must not have had its first page written.
pub fn span_chunks(
    mem: &dyn GuestMemory,
    geometry: Geometry,
    task: &Task,
    gva: u64,
    len: usize,
) -> Result<Vec<SpanChunk>, SpanRefusal> {
    let mut out = Vec::new();
    visit_span_chunks(mem, geometry, task, gva, len, &mut |chunk| {
        out.push(chunk);
        true
    })?;
    Ok(out)
}

/// The page-aligned GPA of every guest page under `[gva, gva+span)`, in order,
/// refusing the whole span if any page does not resolve.
///
/// The fail-closed counterpart to [`walk_span`]'s visitor, which reports an
/// unresolved page and walks on. Both contracts are needed and they are not
/// interchangeable: a caller checking a cached page list against the live table
/// wants the holes reported in place, and a caller about to hand the list to a
/// host mapping wants no list at all rather than a short one. Handing a short
/// list to a mapper maps the pages that did resolve and silently drops the
/// rest, so the guest reads its own bytes for part of a span and someone else's
/// for the remainder.
pub fn span_page_bases(
    mem: &dyn GuestMemory,
    geometry: Geometry,
    task: &Task,
    gva: u64,
    span: u64,
) -> Result<Vec<u64>, SpanRefusal> {
    let mut out = Vec::with_capacity(pages_spanned(gva, span, geometry.page_size()) as usize);
    let mut refused = None;
    walk_span(mem, geometry, task, gva, span, &mut |_, r| match r {
        Ok(page_base) => {
            out.push(page_base);
            true
        }
        Err(status) => {
            refused = Some(status);
            false
        }
    })
    .map_err(SpanRefusal::Setup)?;
    match refused {
        Some(status) => Err(SpanRefusal::Page(status)),
        None => Ok(out),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use reims_vgpu_wire::mem::SliceMemory;
    use reims_vgpu_wire::page_table::{
        Builder, DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN, PTE_SIZE, X86_64,
    };

    const IMAGE: usize = Builder::image_len(X86_64, 16);

    /// Assemble a task whose directory names `root_pfn` and `depth`, in a page
    /// the builder carves for it.
    ///
    /// The directory's two fields are byte offsets and `poke_entry` indexes by
    /// word, so the two are divided rather than restated — a directory laid out
    /// by hand here could disagree with the one `read_task_root` reads.
    fn directory(b: &mut Builder<'_>, root_pfn: u32, depth: u32) -> Task {
        let dir = b.alloc_page();
        b.poke_entry(dir, (DIRECTORY_ROOT_PFN / PTE_SIZE as u64) as u32, root_pfn);
        b.poke_entry(dir, (DIRECTORY_DEPTH / PTE_SIZE as u64) as u32, depth);
        Task {
            active: true,
            directory_pfn: dir,
        }
    }

    /// The span walk reports one page-aligned GPA per page, in order, and the
    /// offset the span starts at decides how many pages that is.
    #[test]
    fn a_span_walk_reports_each_of_its_pages_once_and_page_aligned() {
        let g = X86_64;
        let page = g.page_size();
        let mut buf = [0u8; IMAGE];
        let mut b = Builder::new(g, &mut buf);
        let root = b.map(1, 0, 0x40);
        b.map_into(root, 1, 1, 0x41);
        b.map_into(root, 1, 2, 0x42);
        let task = directory(&mut b, root, 1);
        let mem = SliceMemory::new(b.bytes());

        // Starting mid-page, a two-page span reaches three pages — the term
        // `pages_spanned` exists for, driven through the walk rather than
        // against the arithmetic alone.
        let mut seen = Vec::new();
        walk_span(&mem, g, &task, page / 2, 2 * page, &mut |i, r| {
            seen.push((i, r));
            true
        })
        .unwrap();
        let got: Vec<u64> = seen.iter().map(|(_, r)| r.unwrap()).collect();
        assert_eq!(
            got,
            [0x40u64 * page, 0x41 * page, 0x42 * page],
            "each page's own base, not the first page's offset"
        );
        assert_eq!(
            seen.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
            [0, 1, 2],
            "indexed from the span's first page, in order"
        );
    }

    /// A setup refusal is returned and visits nothing, so a caller cannot read
    /// it as a span that resolved and had no pages.
    ///
    /// The zero-root and zero-depth arms are the ones this function exists for:
    /// the run walker answers both by visiting nothing, which is exactly what a
    /// clean empty span looks like.
    #[test]
    fn a_setup_refusal_is_returned_rather_than_visited() {
        let g = X86_64;
        let mut buf = [0u8; IMAGE];
        let mut b = Builder::new(g, &mut buf);
        let root = b.map(1, 0, 0x40);
        let live = directory(&mut b, root, 1);
        let zero_root = directory(&mut b, 0, 1);
        let zero_depth = directory(&mut b, root, 0);
        let mem = SliceMemory::new(b.bytes());

        for (task, want) in [
            (zero_root, ResolveStatus::ErrZeroRootPfn),
            (zero_depth, ResolveStatus::ErrZeroDepth),
            (
                Task {
                    active: false,
                    directory_pfn: live.directory_pfn,
                },
                ResolveStatus::ErrInactiveTask,
            ),
            (
                Task {
                    active: true,
                    directory_pfn: 0,
                },
                ResolveStatus::ErrNoDirectory,
            ),
        ] {
            let mut visited = 0;
            let r = walk_span(&mem, g, &task, 0, g.page_size(), &mut |_, _| {
                visited += 1;
                true
            });
            assert_eq!(r, Err(want));
            assert_eq!(visited, 0, "{want:?} must visit nothing");
        }

        // And a geometry off both pathways is refused before the task is read
        // at all, so a bad page shift cannot walk a tree at the wrong stride.
        let bad = Geometry { page_shift: 13 };
        assert_eq!(
            walk_span(&mem, bad, &live, 0, 4096, &mut |_, _| true),
            Err(ResolveStatus::ErrUnsupportedGeometry)
        );
    }

    /// A page that does not resolve reaches the visitor as its own refusal,
    /// carrying its position, and the walk is the visitor's to stop.
    #[test]
    fn an_unresolved_page_reaches_the_visitor_with_its_position() {
        let g = X86_64;
        let page = g.page_size();
        let mut buf = [0u8; IMAGE];
        let mut b = Builder::new(g, &mut buf);
        // Page 1 of the span is left unmapped between two that resolve.
        let root = b.map(1, 0, 0x40);
        b.map_into(root, 1, 2, 0x42);
        let task = directory(&mut b, root, 1);
        let mem = SliceMemory::new(b.bytes());

        let mut seen = Vec::new();
        walk_span(&mem, g, &task, 0, 3 * page, &mut |i, r| {
            seen.push((i, r.is_ok()));
            true
        })
        .unwrap();
        assert_eq!(seen, [(0, true), (1, false), (2, true)]);

        // The visitor stops the walk at the hole when it wants to.
        let mut count = 0;
        walk_span(&mem, g, &task, 0, 3 * page, &mut |_, r| {
            count += 1;
            r.is_ok()
        })
        .unwrap();
        assert_eq!(count, 2, "stopped at the page that refused");
    }

    /// A zero-length span resolves to no pages, rather than to the one its
    /// start address sits in.
    #[test]
    fn a_zero_length_span_covers_no_pages() {
        let g = X86_64;
        let mut buf = [0u8; IMAGE];
        let mut b = Builder::new(g, &mut buf);
        let root = b.map(1, 0, 0x40);
        let task = directory(&mut b, root, 1);
        let mem = SliceMemory::new(b.bytes());

        let mut visited = 0;
        walk_span(&mem, g, &task, 0x800, 0, &mut |_, _| {
            visited += 1;
            true
        })
        .unwrap();
        assert_eq!(visited, 0);
    }

    /// The chunk run covers every byte of the span exactly once, in order, and
    /// no chunk crosses the end of its page.
    ///
    /// The three properties are asserted together because the arithmetic fails
    /// by trading one for another: bounding a chunk by the bytes left in the
    /// buffer instead of the bytes left in the page still covers the span
    /// exactly once, and runs off the end of the first page while doing it.
    #[test]
    fn a_chunk_run_covers_every_byte_once_and_stays_inside_its_page() {
        let g = X86_64;
        let page = g.page_size();
        let mut buf = [0u8; IMAGE];
        let mut b = Builder::new(g, &mut buf);
        let root = b.map(1, 0, 0x40);
        b.map_into(root, 1, 1, 0x41);
        b.map_into(root, 1, 2, 0x42);
        b.map_into(root, 1, 3, 0x43);
        let task = directory(&mut b, root, 1);
        let mem = SliceMemory::new(b.bytes());

        // Starts 100 bytes into the first page and runs two pages past it, so
        // the first chunk is short, the middle is whole and the last is a
        // remainder — the three shapes a chunk can have.
        let start = page + 100;
        let len = (2 * page + 7) as usize;
        let chunks = span_chunks(&mem, g, &task, start, len).unwrap();

        assert_eq!(
            chunks[0],
            SpanChunk {
                gpa: 0x41 * page + 100,
                offset: 0,
                len: (page - 100) as usize,
            },
            "the first chunk starts at the span's offset within its page"
        );
        assert_eq!(chunks.len(), 3);

        let mut want_offset = 0usize;
        for c in &chunks {
            assert_eq!(c.offset, want_offset, "chunks tile the buffer in order");
            let in_page = c.gpa & g.page_offset_mask();
            assert!(
                in_page + c.len as u64 <= page,
                "chunk at {:#x} for {} bytes runs past the end of its page",
                c.gpa,
                c.len
            );
            want_offset += c.len;
        }
        assert_eq!(want_offset, len, "every byte of the span is covered");

        // And the GPAs are the pages the table names, not a run from the first.
        let got: Vec<u64> = chunks.iter().map(|c| c.gpa).collect();
        assert_eq!(got, [0x41 * page + 100, 0x42 * page, 0x43 * page]);
    }

    /// A span with an unresolved page yields no chunks at all, and the refusal
    /// says the walk ran — as distinct from never having started.
    ///
    /// The distinction is the point of [`SpanRefusal`]: a caller that reports a
    /// setup status as an unresolved address names a check that never ran.
    #[test]
    fn a_chunk_run_refuses_the_whole_span_and_says_where_it_refused() {
        let g = X86_64;
        let page = g.page_size();
        let mut buf = [0u8; IMAGE];
        let mut b = Builder::new(g, &mut buf);
        // Page 1 of the span is left unmapped between two that resolve.
        let root = b.map(1, 0, 0x40);
        b.map_into(root, 1, 2, 0x42);
        let live = directory(&mut b, root, 1);
        let zero_root = directory(&mut b, 0, 1);
        let mem = SliceMemory::new(b.bytes());

        assert_eq!(
            span_chunks(&mem, g, &live, 0, (3 * page) as usize),
            Err(SpanRefusal::Page(ResolveStatus::ErrZeroPfn)),
            "a hole mid-span refuses the span rather than returning its head"
        );
        assert_eq!(
            span_chunks(&mem, g, &zero_root, 0, (3 * page) as usize),
            Err(SpanRefusal::Setup(ResolveStatus::ErrZeroRootPfn)),
            "a walk that never started is a setup refusal, not an address that \
             does not translate"
        );

        // A zero-length span is not a refusal — it is nothing to do.
        assert_eq!(span_chunks(&mem, g, &live, 0, 0), Ok(Vec::new()));
    }

    /// The fail-closed page list refuses a span with a hole rather than
    /// returning the pages before it.
    ///
    /// A short list is the dangerous answer: its caller maps what it was given
    /// and the guest then reads its own bytes for the head of the span and
    /// whatever owns those host pages for the rest.
    #[test]
    fn span_page_bases_refuses_a_hole_rather_than_returning_a_short_list() {
        let g = X86_64;
        let page = g.page_size();
        let mut buf = [0u8; IMAGE];
        let mut b = Builder::new(g, &mut buf);
        let root = b.map(1, 0, 0x40);
        b.map_into(root, 1, 2, 0x42);
        let task = directory(&mut b, root, 1);
        let mem = SliceMemory::new(b.bytes());

        assert_eq!(
            span_page_bases(&mem, g, &task, 0, 3 * page),
            Err(SpanRefusal::Page(ResolveStatus::ErrZeroPfn))
        );
        // A span that does resolve reports each page's own base, page-aligned:
        // starting mid-page does not carry the offset onto the answer.
        assert_eq!(
            span_page_bases(&mem, g, &task, page / 2, page / 2).unwrap(),
            [0x40 * page]
        );
        assert_eq!(
            span_page_bases(&mem, g, &task, 2 * page + 8, page - 8).unwrap(),
            [0x42 * page]
        );
    }

    /// A span's page count is decided by where it *starts*, not only by how
    /// long it is.
    ///
    /// The device's rails compare a walk's page count against this to decide
    /// whether the whole span resolved. Drop the offset term and a window that
    /// straddles a page boundary — which is most of them, since a texture row
    /// rarely starts page-aligned — reports fully covered while missing its
    /// last page. The gather then hands the GPU a short buffer, which is a
    /// wrong frame.
    #[test]
    fn pages_spanned_counts_the_page_the_offset_pushes_a_span_into() {
        const PAGE: u64 = 4096;
        // Page-aligned: exactly what the length implies.
        assert_eq!(pages_spanned(0, PAGE, PAGE), 1);
        assert_eq!(pages_spanned(PAGE * 7, PAGE * 3, PAGE), 3);
        // Offset by one byte: the same length now reaches one page further.
        assert_eq!(pages_spanned(1, PAGE, PAGE), 2);
        assert_eq!(pages_spanned(PAGE * 7 + 1, PAGE * 3, PAGE), 4);
        // A span wholly inside one page stays at one, wherever it starts.
        assert_eq!(pages_spanned(PAGE - 1, 1, PAGE), 1);
        // …and one byte longer crosses.
        assert_eq!(pages_spanned(PAGE - 1, 2, PAGE), 2);
        // The arm64 pathway's 16 KiB pages take the same rule.
        assert_eq!(pages_spanned(16384 * 3 + 5, 16384, 16384), 2);
        // A zero span touches nothing.
        assert_eq!(pages_spanned(0, 0, PAGE), 0);
    }
}
