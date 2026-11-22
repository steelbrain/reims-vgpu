//! The guest's slow answers: each ring-0 answer the host gave through the device's deferred
//! answers a frame or more after its make (reims-vgpu-graphics' `Host::answer_slow`,
//! `SLOW_ANSWER_NS`, 16.7 ms; never a `PRESENT_SURFACE`'s, which waits for the display's refresh),
//! set against what rt did meanwhile (`reims_vgpu_vulkan::rt_platform::overlap`). A diagnostic:
//! which lines the host writes is free under the contract, and nothing the guest sees changes.
//!
//! - **Over a make of rt's pool.** A slow answer during which a make of rt's pool ran is counted,
//!   [`OVER_MAKE_CODE`], with the longest, [`OVER_MAKE_MAX_NS_CODE`], and the first of each
//!   command type logged at info, the logger's once per kind, with the thread that gave it, how
//!   many makes ran during it and how long before the answer the last of them ended, each make by
//!   its start and its end after the ask (a start before the ask negative), its length in ms, its
//!   allocation's length and storage mode and its thread (`overlap::PoolMake`), and, for a
//!   release-class answer, its split ([`Split`]). rt tells of a make once it ended, with when it
//!   began and how long it took (its `AllocationInfo`'s `make_start` and `make_nanos`), so a make
//!   that began before the ask is among an answer's makes, and one still running when the answer
//!   is given is not.
//! - **A split over no make.** The first slow release-class answer of each command type during
//!   which no make ran is logged at info too, of a kind of its own, with its split.
//! - **Over compute dispatches' first-use waits.** The first-use waits of compute pipelines (rt's
//!   `Metrics::first_use_variant_waits`, on the counts line as `rt-first-use-variant-waits`) that
//!   met any slow answer, from the earliest wait's start to its make's end, are counted, each once,
//!   [`FIRST_USE_WAITS_CODE`].
//! - **The split.** A release-class answer marks its waits and its frees on rt's host clock as
//!   they end, apart, so a line tells an answer that waited for the guest's command buffers, for
//!   our own settles, or for a call of rt's that releases or frees memory (its name, its ms, the
//!   bytes it gives back and its thread):
//!   - A `BLOCK_RELEASE` (`GraphicsControl::release_block`): 1 the no-copy buffers being made over
//!     the block, 2 its `SURFACE_PAGES`' settle, then the release's own call, then the drain,
//!     which ends with 3 the context's command buffers or 5 the device cache's write-back,
//!     whichever ended last, told by whether the drain's answer ran inside the write-back's end
//!     ([`writing_back`]), then the unshares it waits for, then 4 the displays' last show of a
//!     surface over the block. It frees nothing of rt's on its way: the block is the kext's, and
//!     its surface view and device cache go on the surface worker and the let-go worker, which
//!     the answer does not wait for.
//!   - A `POOL_GIVE` (`GraphicsControl::give_pool_block`): the give's own call, 3 the context's
//!     command buffers, and the ledger's free of the block, which calls nothing of rt's: a piece
//!     left empty goes back to rt later, on the pool's thread.
//!   - A `PRIVATE_GIVE` (`GraphicsControl::give_private_block`): the give's own call, 3 the
//!     context's command buffers, the ledger's free, then rt's release of the block (the source's
//!     `release_block`, `pool::RtBlocks`, over rt's `Device::release_block`), with its bytes.
//!   - A `CONTEXT_CLOSE` (`GraphicsControl::close_context`): the no-copy buffers being made for
//!     the context, the close's own call, then the drain, which ends once 3 the context's command
//!     buffers have finished and the close worker has let go of the context's objects, their
//!     releases in rt (`Closed::let_go`): the let-go's span after the ask, its objects and its
//!     thread; then the Shared pool's free of the context's blocks, which calls nothing of rt's,
//!     and rt's release of its Private blocks, as a `PRIVATE_GIVE`'s, with their bytes.
//!
//!   The frees of a `POOL_GIVE`, a `PRIVATE_GIVE` and a `CONTEXT_CLOSE` run in their drain's
//!   answer, on the thread that then gives the answer, which the line names. Each answer carries
//!   its split to [`slow`] in a thread-local set across its answer's call ([`answering`]).
//! - **Its cost.** Every answer the device times reads its clock at its make and at its post and
//!   compares the two (reims-vgpu-graphics' `Asked`). A `BLOCK_RELEASE` adds an `Arc` of its
//!   marks, two boxed closures around the drain's and the unshares' answers, seven reads of rt's
//!   host clock, four thread-local writes and one read; a `POOL_GIVE` adds four clock reads and two
//!   thread-local writes; a `PRIVATE_GIVE` five clock reads and two thread-local writes; a
//!   `CONTEXT_CLOSE` an `Arc` of its marks, nine clock reads, its let-go's thread's handle and a
//!   copy of its name under an uncontended lock, and two thread-local writes. A make of rt's pool
//!   adds a clone of its thread's handle and an uncontended lock (`overlap::pool_made`): its
//!   start and its length come from rt. Only a slow answer reads more: the last 16 makes under
//!   that lock, its thread's handle, and rt's last 256 compute variant makes under its lock
//!   (`Counters::variant_makes`).

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use metal2vulkan_runtime as rt;
use reims_vgpu_graphics::wire::{
    RGFX_CMD_BLOCK_RELEASE, RGFX_CMD_CONTEXT_CLOSE, RGFX_CMD_POOL_GIVE, RGFX_CMD_POOL_TAKE,
    RGFX_CMD_PRIVATE_GIVE, RGFX_CMD_PRIVATE_TAKE, RGFX_CMD_SURFACE_DOWNLOAD,
    RGFX_CMD_SURFACE_PAGES,
};
use reims_vgpu_vulkan::answer::{count_kind, count_kind_by};
use reims_vgpu_vulkan::core::sync::Mutex;
use reims_vgpu_vulkan::rt_platform::overlap::{self, PoolMake};

/// The code of the slow answers during which a make of rt's pool ran.
pub(crate) const OVER_MAKE_CODE: &str = "slow-answer-over-make";

/// The code of the longest of them, in nanoseconds, which its count rises to.
pub(crate) const OVER_MAKE_MAX_NS_CODE: &str = "slow-answer-over-make-max-ns";

/// The code of the compute dispatches' first-use waits that met a slow answer.
pub(crate) const FIRST_USE_WAITS_CODE: &str = "slow-answer-first-use-waits";

/// The longest slow answer over a make so far, which [`OVER_MAKE_MAX_NS_CODE`] rises to.
static OVER_MAKE_MOST: AtomicU64 = AtomicU64::new(0);

/// The log target of the lines.
const TARGET: &str = "reims_vgpu::graphics";

/// How a line names a thread that has no name.
const UNNAMED: &str = "an unnamed thread";

thread_local! {
    /// The split of the answer this thread is giving ([`answering`]).
    static ANSWERING: Cell<Option<Split>> = const { Cell::new(None) };
    /// Whether this thread is in a `BLOCK_RELEASE`'s write-back's end ([`writing_back`]).
    static WRITING_BACK: Cell<bool> = const { Cell::new(false) };
}

/// The blocks a free gave back, and their bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Freed {
    /// How many.
    pub(crate) blocks: usize,
    /// Their bytes, where the free knows them (0 where it does not).
    pub(crate) bytes: u64,
}

/// A `CONTEXT_CLOSE`'s let-go of its objects on the close worker (`Closed::let_go`), on rt's host
/// clock.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LetGo {
    /// When it began.
    began: u64,
    /// When it ended.
    ended: u64,
    /// The objects it let go of.
    objects: usize,
    /// The name of the thread it ran on.
    thread: String,
}

/// The waits and frees of a slow answer, on rt's host clock (the module's docs, "The split").
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Split {
    /// A `BLOCK_RELEASE`'s: when it was asked, its no-copy buffers had been made, its
    /// `SURFACE_PAGES` had settled (0 where not marked), the release's call returned (0 where the
    /// drain answered inside it), the write-back ended (0 where it had not), the drain ended, and
    /// the unshares ended; and whether the write-back ended the drain.
    Release {
        asked: u64,
        pages: u64,
        settled: u64,
        released: u64,
        written: u64,
        drained: u64,
        unshared: u64,
        write_back_last: bool,
    },
    /// A `POOL_GIVE`'s: when it was asked, the block was given, the drain ended, and the ledger
    /// freed the block of `bytes`.
    Give {
        asked: u64,
        given: u64,
        drained: u64,
        freed: u64,
        bytes: u64,
    },
    /// A `PRIVATE_GIVE`'s: when it was asked, the block was given, the drain ended, the ledger
    /// freed the block, and rt's release of what it gave back, `freed`, returned.
    PrivateGive {
        asked: u64,
        given: u64,
        drained: u64,
        ledger: u64,
        released: u64,
        freed: Freed,
    },
    /// A `CONTEXT_CLOSE`'s: when it was asked, its no-copy buffers had been made, the close's call
    /// handed its objects to the close worker, its let-go (`None` where not marked), the drain
    /// ended, the Shared pool freed the context's blocks, `shared`, the ledger freed its Private
    /// blocks, and rt's release of what it gave back of them, `private`, returned.
    Close {
        asked: u64,
        pages: u64,
        closed: u64,
        let_go: Option<LetGo>,
        drained: u64,
        shared: Freed,
        shared_freed: u64,
        ledger: u64,
        released: u64,
        private: Freed,
    },
}

/// Milliseconds from `from` to `to` of rt's host clock, 0 where `to` is not after it.
fn ms(from: u64, to: u64) -> f64 {
    to.saturating_sub(from) as f64 / 1e6
}

/// Milliseconds from `from` to `to` of rt's host clock, negative where `to` is before it.
fn ms_signed(from: u64, to: u64) -> f64 {
    (i128::from(to) - i128::from(from)) as f64 / 1e6
}

/// `bytes` in MiB.
fn mib(bytes: u64) -> f64 {
    bytes as f64 / f64::from(1u32 << 20)
}

/// The words of rt's release of the Private blocks `freed`, which took `took` ms.
fn private_release(freed: Freed, took: f64) -> String {
    match freed.blocks {
        0 => format!("no Private block of rt's to release, {took:.3} ms"),
        1 => format!(
            "rt's release of its Private block, {:.3} MiB (the source's release_block), \
             {took:.3} ms",
            mib(freed.bytes)
        ),
        n => format!(
            "rt's release of its {n} Private blocks, {:.3} MiB (the source's release_block), \
             {took:.3} ms",
            mib(freed.bytes)
        ),
    }
}

impl Split {
    /// The split's words for an answer given at `answered`.
    fn words(&self, answered: u64) -> String {
        match self {
            &Split::Release {
                asked,
                pages,
                settled,
                released,
                written,
                drained,
                unshared,
                write_back_last,
            } => {
                let released = if released == 0 || released > drained {
                    drained
                } else {
                    released
                };
                let settled = if settled < pages || settled > released {
                    pages
                } else {
                    settled
                };
                let last = if write_back_last {
                    "5 the device cache's write-back"
                } else {
                    "3 the context's command buffers"
                };
                let written = match written {
                    0 => String::new(),
                    at => {
                        let after = ms(released, at);
                        format!(", the write-back ended {after:.3} ms after it")
                    }
                };
                format!(
                    "1 the no-copy buffers {:.3} ms, 2 the SURFACE_PAGES settle {:.3} ms, the \
                     release's call {:.3} ms, then {:.3} ms until {last} ended{written}, the \
                     unshares {:.3} ms, 4 no display showing it {:.3} ms",
                    ms(asked, pages),
                    ms(pages, settled),
                    ms(settled, released),
                    ms(released, drained),
                    ms(drained, unshared),
                    ms(unshared, answered)
                )
            }
            &Split::Give {
                asked,
                given,
                drained,
                freed,
                bytes,
            } => format!(
                "the give's call {:.3} ms, 3 the context's command buffers {:.3} ms, the \
                 ledger's free of its block, {:.3} MiB, {:.3} ms (no call of rt's: a piece left \
                 empty goes back to rt on the pool's thread), then {:.3} ms to the answer",
                ms(asked, given),
                ms(given, drained),
                mib(bytes),
                ms(drained, freed),
                ms(freed, answered)
            ),
            &Split::PrivateGive {
                asked,
                given,
                drained,
                ledger,
                released,
                freed,
            } => format!(
                "the give's call {:.3} ms, 3 the context's command buffers {:.3} ms, the \
                 ledger's free {:.3} ms, {}, then {:.3} ms to the answer",
                ms(asked, given),
                ms(given, drained),
                ms(drained, ledger),
                private_release(freed, ms(ledger, released)),
                ms(released, answered)
            ),
            Split::Close {
                asked,
                pages,
                closed,
                let_go,
                drained,
                shared,
                shared_freed,
                ledger,
                released,
                private,
            } => {
                let (asked, drained) = (*asked, *drained);
                let let_go = match let_go {
                    Some(l) => format!(
                        "the let-go of its {} objects, their releases in rt, on {} from {:.3} to \
                         {:.3} ms after the ask",
                        l.objects,
                        l.thread,
                        ms(asked, l.began),
                        ms(asked, l.ended)
                    ),
                    None => "the let-go of its objects".to_owned(),
                };
                format!(
                    "the no-copy buffers {:.3} ms, the close's call {:.3} ms, then the drain \
                     ended {:.3} ms after the ask (3 the context's command buffers, and {let_go}), \
                     the Shared pool's free of its {} blocks {:.3} ms (no call of rt's), the \
                     ledger's free {:.3} ms, {}, then {:.3} ms to the answer",
                    ms(asked, *pages),
                    ms(*pages, *closed),
                    ms(asked, drained),
                    shared.blocks,
                    ms(drained, *shared_freed),
                    ms(*shared_freed, *ledger),
                    private_release(*private, ms(*ledger, *released)),
                    ms(*released, answered)
                )
            }
        }
    }
}

/// Gives an answer, `answer`, whose split is `split`: a slow one's call of [`slow`], on this
/// thread inside `answer`, reads it.
pub(crate) fn answering(split: Split, answer: impl FnOnce()) {
    ANSWERING.with(|s| s.set(Some(split)));
    answer();
    ANSWERING.with(|s| s.set(None));
}

/// Ends a `BLOCK_RELEASE`'s write-back, `end`: a drain's answer that runs inside it was held last
/// by the write-back ([`ReleaseMarks::drained`]).
pub(crate) fn writing_back(end: impl FnOnce()) {
    WRITING_BACK.with(|w| w.set(true));
    end();
    WRITING_BACK.with(|w| w.set(false));
}

/// The marks of one `BLOCK_RELEASE`'s waits as they end, on rt's host clock ([`Split::Release`]).
#[derive(Debug)]
pub(crate) struct ReleaseMarks {
    asked: u64,
    pages: AtomicU64,
    settled: AtomicU64,
    released: AtomicU64,
    written: AtomicU64,
    drained: AtomicU64,
    unshared: AtomicU64,
    write_back_last: AtomicBool,
}

impl ReleaseMarks {
    /// A release asked now.
    pub(crate) fn asked() -> ReleaseMarks {
        ReleaseMarks {
            asked: rt::host_time(),
            pages: AtomicU64::new(0),
            settled: AtomicU64::new(0),
            released: AtomicU64::new(0),
            written: AtomicU64::new(0),
            drained: AtomicU64::new(0),
            unshared: AtomicU64::new(0),
            write_back_last: AtomicBool::new(false),
        }
    }

    /// Its no-copy buffers were made.
    pub(crate) fn pages(&self) {
        self.pages.store(rt::host_time(), Ordering::Relaxed);
    }

    /// Its `SURFACE_PAGES` settled, inside the release's call.
    pub(crate) fn settled(&self) {
        self.settled.store(rt::host_time(), Ordering::Relaxed);
    }

    /// The release's call returned.
    pub(crate) fn released(&self) {
        self.released.store(rt::host_time(), Ordering::Relaxed);
    }

    /// The write-back ended.
    pub(crate) fn written(&self) {
        self.written.store(rt::host_time(), Ordering::Relaxed);
    }

    /// The drain ended, inside the write-back's end or not ([`writing_back`]).
    pub(crate) fn drained(&self) {
        self.drained.store(rt::host_time(), Ordering::Relaxed);
        let last = WRITING_BACK.with(Cell::get);
        self.write_back_last.store(last, Ordering::Relaxed);
    }

    /// The unshares ended.
    pub(crate) fn unshared(&self) {
        self.unshared.store(rt::host_time(), Ordering::Relaxed);
    }

    /// The split so far.
    pub(crate) fn split(&self) -> Split {
        let at = |mark: &AtomicU64| mark.load(Ordering::Relaxed);
        Split::Release {
            asked: self.asked,
            pages: at(&self.pages),
            settled: at(&self.settled),
            released: at(&self.released),
            written: at(&self.written),
            drained: at(&self.drained),
            unshared: at(&self.unshared),
            write_back_last: self.write_back_last.load(Ordering::Relaxed),
        }
    }
}

/// The marks of one `CONTEXT_CLOSE`'s waits before its drain's answer, on rt's host clock
/// ([`Split::Close`]).
#[derive(Debug)]
pub(crate) struct CloseMarks {
    asked: u64,
    pages: AtomicU64,
    closed: AtomicU64,
    let_go: Mutex<Option<LetGo>>,
}

/// What a `CONTEXT_CLOSE`'s drain's answer did before its answer, on rt's host clock: when the
/// drain ended, the Shared pool freed the context's blocks, `shared`, the ledger freed its Private
/// blocks, and rt's release of what it gave back of them, `private`, returned.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CloseFrees {
    pub(crate) drained: u64,
    pub(crate) shared: Freed,
    pub(crate) shared_freed: u64,
    pub(crate) ledger: u64,
    pub(crate) released: u64,
    pub(crate) private: Freed,
}

impl CloseMarks {
    /// A close asked now.
    pub(crate) fn asked() -> CloseMarks {
        CloseMarks {
            asked: rt::host_time(),
            pages: AtomicU64::new(0),
            closed: AtomicU64::new(0),
            let_go: Mutex::new(None),
        }
    }

    /// Its no-copy buffers were made.
    pub(crate) fn pages(&self) {
        self.pages.store(rt::host_time(), Ordering::Relaxed);
    }

    /// The close's call handed its objects to the close worker.
    pub(crate) fn closed(&self) {
        self.closed.store(rt::host_time(), Ordering::Relaxed);
    }

    /// Lets go of the close's `objects` objects, `let_go`, on this thread, timed.
    pub(crate) fn letting_go(&self, objects: usize, let_go: impl FnOnce()) {
        let began = rt::host_time();
        let_go();
        let ended = rt::host_time();
        let thread = std::thread::current();
        let thread = thread.name().unwrap_or(UNNAMED).to_owned();
        *self.let_go.lock() = Some(LetGo {
            began,
            ended,
            objects,
            thread,
        });
    }

    /// The split, with what the drain's answer did, `frees`.
    pub(crate) fn split(&self, frees: CloseFrees) -> Split {
        let at = |mark: &AtomicU64| mark.load(Ordering::Relaxed);
        Split::Close {
            asked: self.asked,
            pages: at(&self.pages),
            closed: at(&self.closed),
            let_go: self.let_go.lock().clone(),
            drained: frees.drained,
            shared: frees.shared,
            shared_freed: frees.shared_freed,
            ledger: frees.ledger,
            released: frees.released,
            private: frees.private,
        }
    }
}

/// The command type `kind`'s name.
fn name(kind: u32) -> &'static str {
    match kind {
        RGFX_CMD_BLOCK_RELEASE => "BLOCK_RELEASE",
        RGFX_CMD_POOL_GIVE => "POOL_GIVE",
        RGFX_CMD_POOL_TAKE => "POOL_TAKE",
        RGFX_CMD_PRIVATE_TAKE => "PRIVATE_TAKE",
        RGFX_CMD_PRIVATE_GIVE => "PRIVATE_GIVE",
        RGFX_CMD_CONTEXT_CLOSE => "CONTEXT_CLOSE",
        RGFX_CMD_SURFACE_DOWNLOAD => "SURFACE_DOWNLOAD",
        RGFX_CMD_SURFACE_PAGES => "SURFACE_PAGES",
        _ => "ring-0 command",
    }
}

/// A ring-0 answer to a command of type `kind` went on the ring `took_ns` after its make, a
/// frame or more (`Host::answer_slow`), on this thread just now (the module's docs).
pub(crate) fn slow(kind: u32, took_ns: u64) {
    let split = ANSWERING.with(Cell::take);
    let end = rt::host_time();
    let start = end.saturating_sub(took_ns);
    let waits = overlap::first_use_waits_met(start, end);
    let makes = overlap::pool_makes_met(start, end);
    let thread = std::thread::current();
    let thread = thread.name().unwrap_or(UNNAMED);
    // A split over no make is a kind of its own: the first over a make is logged as before it.
    let of = if makes.is_empty() {
        format!("{} over no make", name(kind))
    } else {
        name(kind).to_owned()
    };
    if let Some(line) = counted(kind, took_ns, end, &makes, waits, split, thread) {
        log::info!(target: TARGET, kind:% = of; "{line}");
    }
}

/// The words of `makes`, the makes of rt's pool that ran during an answer asked at `start`.
fn makes_words(makes: &[PoolMake], start: u64) -> String {
    let each: Vec<String> = (makes.iter())
        .map(|make| {
            let class = match make.class {
                rt::MemoryClass::Private => "Private",
                rt::MemoryClass::Shared => "Shared",
            };
            format!(
                "{:.3} to {:.3} ms, {:.3} ms, a {class} {:.3} MiB on {}",
                ms_signed(start, make.began),
                ms_signed(start, make.ended),
                ms(make.began, make.ended),
                mib(make.len),
                make.thread.name().unwrap_or(UNNAMED)
            )
        })
        .collect();
    each.join("; ")
}

/// Counts a slow answer to a command of type `kind`, `took_ns` long and given at `end` on the
/// thread named `thread`, during which `makes` of rt's pool ran, and which met `waits`
/// first-use waits not counted before; its line where it is over a make or has a split, with its
/// split where it has one.
fn counted(
    kind: u32,
    took_ns: u64,
    end: u64,
    makes: &[PoolMake],
    waits: u64,
    split: Option<Split>,
    thread: &str,
) -> Option<String> {
    count_kind_by(FIRST_USE_WAITS_CODE, waits);
    if !makes.is_empty() {
        count_kind(OVER_MAKE_CODE);
        let was = OVER_MAKE_MOST.fetch_max(took_ns, Ordering::Relaxed);
        count_kind_by(OVER_MAKE_MAX_NS_CODE, took_ns.saturating_sub(was));
    } else if split.is_none() {
        return None;
    }
    let name = name(kind);
    let start = end.saturating_sub(took_ns);
    let over = match makes.iter().map(|make| make.ended).max() {
        None => "over no make of rt's pool".to_owned(),
        Some(last) => format!(
            "over {} makes of rt's pool that ran during it, the last ending {:.3} ms before the \
             answer (each from its start to its end after the ask, a start before it negative: \
             {})",
            makes.len(),
            ms(last, end),
            makes_words(makes, start)
        ),
    };
    let waits = match waits {
        0 => String::new(),
        n => format!(", over {n} first-use waits of compute pipelines"),
    };
    let split = match split {
        Some(split) => format!(": {}", split.words(end)),
        None => String::new(),
    };
    Some(format!(
        "a {name} (type {kind:#x}) answered {:.3} ms after it was asked, on {thread}, \
         {over}{waits}{split}",
        took_ns as f64 / 1e6,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A millisecond of rt's host clock.
    const MS: u64 = 1_000_000;

    /// The process's count of `code`.
    fn count(code: &str) -> u64 {
        (reims_vgpu_vulkan::answer::degradation_counts().iter())
            .find(|(c, _)| c == code)
            .map_or(0, |&(_, n)| n)
    }

    /// A make of rt's pool of `len` bytes of `class` that began at `began` and ended at `ended`,
    /// made on a thread named `reims-vgpu.test-maker`.
    fn make(began: u64, ended: u64, len: u64, class: rt::MemoryClass) -> PoolMake {
        let maker = std::thread::Builder::new().name("reims-vgpu.test-maker".into());
        let thread = maker.spawn(std::thread::current).unwrap().join().unwrap();
        PoolMake {
            began,
            ended,
            len,
            class,
            thread,
        }
    }

    /// A slow answer during which no make of rt's pool ran, and with no split, is not counted
    /// over a make and writes no line, but the first-use waits it met are counted
    #[test]
    fn a_slow_answer_over_no_make_counts_only_its_first_use_waits() {
        let (over, waits) = (count(OVER_MAKE_CODE), count(FIRST_USE_WAITS_CODE));
        let line = counted(RGFX_CMD_POOL_TAKE, 20 * MS, 50 * MS, &[], 2, None, "t");
        assert_eq!(line, None);
        assert_eq!(count(OVER_MAKE_CODE), over);
        assert_eq!(count(FIRST_USE_WAITS_CODE), waits + 2);
    }

    /// A slow `BLOCK_RELEASE` over a make is counted, and its line gives its thread, the makes by
    /// their starts and ends after the ask, one that began before it negative, with their ms,
    /// lengths, storage modes and threads, the first-use waits and its split: its settle apart
    /// from the release's call, the drain's end told as the write-back's where the drain answered
    /// inside it, and the release's call ending at the drain where the drain answered inside it
    #[test]
    fn a_slow_release_over_a_make_counts_and_gives_its_makes_and_its_split() {
        let over = count(OVER_MAKE_CODE);
        let split = Split::Release {
            asked: 100 * MS,
            pages: 101 * MS,
            settled: 105 * MS,
            released: 0,
            written: 110 * MS,
            drained: 110 * MS,
            unshared: 111 * MS,
            write_back_last: true,
        };
        let makes = [
            make(96 * MS, 110 * MS, 16 << 20, rt::MemoryClass::Private),
            make(112 * MS, 115 * MS, 32 << 20, rt::MemoryClass::Shared),
        ];
        let line = counted(
            RGFX_CMD_BLOCK_RELEASE,
            20 * MS,
            120 * MS,
            &makes,
            1,
            Some(split),
            "reims-vgpu.graphics-control",
        );
        assert_eq!(count(OVER_MAKE_CODE), over + 1);
        assert_eq!(
            line.as_deref(),
            Some(
                "a BLOCK_RELEASE (type 0x2f2) answered 20.000 ms after it was asked, on \
                 reims-vgpu.graphics-control, over 2 makes of rt's pool that ran during it, the \
                 last ending 5.000 ms before the answer (each from its start to its end after the \
                 ask, a start before it negative: -4.000 to 10.000 ms, 14.000 ms, a Private \
                 16.000 MiB on reims-vgpu.test-maker; 12.000 to 15.000 ms, 3.000 ms, a Shared \
                 32.000 MiB on reims-vgpu.test-maker), over 1 first-use waits of compute \
                 pipelines: 1 the no-copy buffers 1.000 ms, 2 the SURFACE_PAGES \
                 settle 4.000 ms, the release's call 5.000 ms, then 0.000 ms until 5 the device \
                 cache's write-back ended, the write-back ended 0.000 ms after it, the unshares \
                 1.000 ms, 4 no display showing it 9.000 ms"
            )
        );
    }

    /// A `BLOCK_RELEASE` whose settle was not marked tells none, the release's call from its
    /// no-copy buffers on
    #[test]
    fn a_release_with_no_settle_marked_tells_its_call_from_its_buffers() {
        let split = Split::Release {
            asked: 100 * MS,
            pages: 101 * MS,
            settled: 0,
            released: 104 * MS,
            written: 0,
            drained: 130 * MS,
            unshared: 130 * MS,
            write_back_last: false,
        };
        let words = split.words(130 * MS);
        assert!(
            words.starts_with(
                "1 the no-copy buffers 1.000 ms, 2 the SURFACE_PAGES settle 0.000 ms, the \
                 release's call 3.000 ms, then 26.000 ms until 3 the context's command buffers \
                 ended, the unshares"
            ),
            "{words}"
        );
    }

    /// A slow release-class answer during which no make ran is not counted over a make, but its
    /// line gives its split: a `PRIVATE_GIVE`'s ledger's free apart from rt's release of its
    /// block, with its bytes
    #[test]
    fn a_slow_private_give_over_no_make_gives_rts_release_apart() {
        let over = count(OVER_MAKE_CODE);
        let split = Split::PrivateGive {
            asked: 100 * MS,
            given: 101 * MS,
            drained: 121 * MS,
            ledger: 121 * MS + MS / 2,
            released: 124 * MS + MS / 2,
            freed: Freed {
                blocks: 1,
                bytes: 16 << 20,
            },
        };
        let line = counted(
            RGFX_CMD_PRIVATE_GIVE,
            25 * MS,
            125 * MS,
            &[],
            0,
            Some(split),
            "reims-vgpu.graphics-control",
        );
        assert_eq!(count(OVER_MAKE_CODE), over);
        assert_eq!(
            line.as_deref(),
            Some(
                "a PRIVATE_GIVE (type 0x2fe) answered 25.000 ms after it was asked, on \
                 reims-vgpu.graphics-control, over no make of rt's pool: the give's call 1.000 \
                 ms, 3 the context's command buffers 20.000 ms, the ledger's free 0.500 ms, rt's \
                 release of its Private block, 16.000 MiB (the source's release_block), 3.000 ms, \
                 then 0.500 ms to the answer"
            )
        );
    }

    /// A `PRIVATE_GIVE` whose block was the host's pick gives nothing back to rt, and says so
    #[test]
    fn a_private_give_of_a_host_pick_releases_nothing_of_rts() {
        let split = Split::PrivateGive {
            asked: 100 * MS,
            given: 100 * MS,
            drained: 120 * MS,
            ledger: 120 * MS,
            released: 120 * MS,
            freed: Freed::default(),
        };
        let words = split.words(120 * MS);
        assert!(
            words.contains("no Private block of rt's to release, 0.000 ms"),
            "{words}"
        );
    }

    /// A `POOL_GIVE`'s split tells its free as the ledger's, with its bytes, and no call of rt's
    #[test]
    fn a_pool_gives_free_is_told_as_the_ledgers() {
        let split = Split::Give {
            asked: 100 * MS,
            given: 101 * MS,
            drained: 118 * MS,
            freed: 119 * MS,
            bytes: 1 << 20,
        };
        assert_eq!(
            split.words(120 * MS),
            "the give's call 1.000 ms, 3 the context's command buffers 17.000 ms, the ledger's \
             free of its block, 1.000 MiB, 1.000 ms (no call of rt's: a piece left empty goes \
             back to rt on the pool's thread), then 1.000 ms to the answer"
        );
    }

    /// A `CONTEXT_CLOSE`'s split gives the let-go of its objects by its span after the ask, its
    /// objects and its thread, beside the drain's end, then the Shared pool's free apart from rt's
    /// release of its Private blocks, with their bytes
    #[test]
    fn a_close_split_gives_its_let_go_and_its_frees_apart() {
        let split = Split::Close {
            asked: 100 * MS,
            pages: 101 * MS,
            closed: 103 * MS,
            let_go: Some(LetGo {
                began: 104 * MS,
                ended: 150 * MS,
                objects: 345,
                thread: "reims-vgpu.graphics-close".to_owned(),
            }),
            drained: 151 * MS,
            shared: Freed {
                blocks: 2,
                bytes: 0,
            },
            shared_freed: 152 * MS,
            ledger: 153 * MS,
            released: 156 * MS,
            private: Freed {
                blocks: 3,
                bytes: 48 << 20,
            },
        };
        assert_eq!(
            split.words(157 * MS),
            "the no-copy buffers 1.000 ms, the close's call 2.000 ms, then the drain ended \
             51.000 ms after the ask (3 the context's command buffers, and the let-go of its 345 \
             objects, their releases in rt, on reims-vgpu.graphics-close from 4.000 to 50.000 \
             ms after the ask), the Shared pool's free of its 2 blocks 1.000 ms (no call of \
             rt's), the ledger's free 1.000 ms, rt's release of its 3 Private blocks, 48.000 MiB \
             (the source's release_block), 3.000 ms, then 1.000 ms to the answer"
        );
    }

    /// A close's marks keep its let-go's span, objects and thread, and its waits in their order
    #[test]
    fn a_closes_marks_keep_its_let_go_and_its_waits() {
        let marks = CloseMarks::asked();
        marks.pages();
        marks.closed();
        std::thread::scope(|scope| {
            let worker = std::thread::Builder::new().name("reims-vgpu.test-close".into());
            let let_go = worker.spawn_scoped(scope, || marks.letting_go(7, || ()));
            let_go.unwrap().join().unwrap();
        });
        let frees = CloseFrees {
            drained: rt::host_time(),
            shared: Freed::default(),
            shared_freed: 0,
            ledger: 0,
            released: 0,
            private: Freed::default(),
        };
        let Split::Close {
            asked,
            pages,
            closed,
            let_go: Some(let_go),
            drained,
            ..
        } = marks.split(frees)
        else {
            panic!("a close's split with its let-go");
        };
        assert_eq!(let_go.objects, 7);
        assert_eq!(let_go.thread, "reims-vgpu.test-close");
        assert!(asked <= pages && pages <= closed && closed <= let_go.began);
        assert!(let_go.began <= let_go.ended && let_go.ended <= drained);
    }

    /// The longest slow answer over a make rises to its longest, and a shorter one leaves it
    #[test]
    fn the_longest_slow_answer_over_a_make_counts_its_rise() {
        let base = OVER_MAKE_MOST.load(Ordering::Relaxed);
        let before = count(OVER_MAKE_MAX_NS_CODE);
        let end = base + 9_000;
        let made = [make(base, base, 16 << 20, rt::MemoryClass::Shared)];
        let _ = counted(RGFX_CMD_POOL_GIVE, base + 3_000, end, &made, 0, None, "t");
        assert_eq!(count(OVER_MAKE_MAX_NS_CODE), before + 3_000);
        let _ = counted(RGFX_CMD_POOL_GIVE, base + 1_000, end, &made, 0, None, "t");
        assert_eq!(count(OVER_MAKE_MAX_NS_CODE), before + 3_000);
    }

    /// A drain that ends inside the write-back's end is told as the write-back's; one outside as
    /// the command buffers'; a release's settle is marked between its buffers and its call's end;
    /// and an answer's split is read inside its call and gone after it
    #[test]
    fn the_drains_end_is_told_by_the_thread_it_ran_in() {
        let marks = ReleaseMarks::asked();
        let last = |marks: &ReleaseMarks| match marks.split() {
            Split::Release {
                write_back_last, ..
            } => write_back_last,
            _ => false,
        };
        writing_back(|| marks.drained());
        assert!(last(&marks));
        marks.drained();
        assert!(!last(&marks));
        marks.pages();
        marks.settled();
        marks.released();
        let Split::Release {
            pages,
            settled,
            released,
            ..
        } = marks.split()
        else {
            panic!("a release's split");
        };
        assert!(pages != 0 && pages <= settled && settled <= released);
        let split = marks.split();
        let mut seen = None;
        answering(split.clone(), || {
            seen = ANSWERING.with(|s| {
                let split = s.take();
                s.set(split.clone());
                split
            });
        });
        assert_eq!(seen, Some(split));
        assert_eq!(ANSWERING.with(Cell::take), None);
    }
}
