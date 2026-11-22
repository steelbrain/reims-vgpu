//! The guest's slow answers: each ring-0 answer the host gave through the device's deferred
//! answers a frame or more after its make (reims-vgpu-graphics' `Host::answer_slow`,
//! `SLOW_ANSWER_NS`, 16.7 ms; never a `PRESENT_SURFACE`'s, which waits for the display's refresh),
//! set against what rt did meanwhile (`reims_vgpu_vulkan::rt_platform::overlap`). A diagnostic:
//! which lines the host writes is free under the contract, and nothing the guest sees changes.
//!
//! - **Over a make of rt's pool.** A slow answer during which a make of rt's pool ended is
//!   counted, [`OVER_MAKE_CODE`], with the longest, [`OVER_MAKE_MAX_NS_CODE`], and the first of
//!   each command type logged at info, the logger's once per kind, with how many makes ended
//!   during it and how long before the answer the last did, and, for a `BLOCK_RELEASE` or a
//!   `POOL_GIVE`, its split ([`Split`]). rt tells nothing as a make begins, so an answer that ended
//!   inside a make is not counted: every answer that waited for a make to end is.
//! - **Over compute dispatches' first-use waits.** The first-use waits of compute pipelines (rt's
//!   `Metrics::first_use_variant_waits`, on the counts line as `rt-first-use-variant-waits`) that
//!   met any slow answer, from the earliest wait's start to its make's end, are counted, each once,
//!   [`FIRST_USE_WAITS_CODE`].
//! - **The split.** A `BLOCK_RELEASE` (`GraphicsControl::release_block`) marks its waits on rt's
//!   host clock as they end: 1 the no-copy buffers being made over the block, 2 its
//!   `SURFACE_PAGES`' settle and the release's own call, then the drain, which ends with 3 the
//!   context's command buffers or 5 the device cache's write-back, whichever ended last, told by
//!   whether the drain's answer ran inside the write-back's end ([`writing_back`]), then the
//!   unshares it waits for, then 4 the displays' last show of a surface over the block. A
//!   `POOL_GIVE` (`GraphicsControl::give_pool_block`) marks the give's own call, 3 the context's
//!   command buffers, and the free. Each answer carries its split to [`slow`] in a thread-local
//!   set across its answer's call ([`answering`]).
//! - **Its cost.** Every answer the device times reads its clock at its make and at its post and
//!   compares the two (reims-vgpu-graphics' `Asked`). A `BLOCK_RELEASE` adds an `Arc` of its
//!   marks, two boxed closures around the drain's and the unshares' answers, six reads of rt's
//!   host clock, four thread-local writes and one read; a `POOL_GIVE` adds four clock reads and two
//!   thread-local writes. A make of rt's pool adds one clock read and two relaxed atomics
//!   (`overlap::pool_made`). Only a slow answer reads more: the last 16 makes' ends, and rt's last
//!   256 compute variant makes under its lock (`Counters::variant_makes`).

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use metal2vulkan_runtime as rt;
use reims_vgpu_graphics::wire::{
    RGFX_CMD_BLOCK_RELEASE, RGFX_CMD_CONTEXT_CLOSE, RGFX_CMD_POOL_GIVE, RGFX_CMD_POOL_TAKE,
    RGFX_CMD_PRIVATE_GIVE, RGFX_CMD_PRIVATE_TAKE, RGFX_CMD_SURFACE_DOWNLOAD,
    RGFX_CMD_SURFACE_PAGES,
};
use reims_vgpu_vulkan::answer::{count_kind, count_kind_by};
use reims_vgpu_vulkan::rt_platform::overlap;

/// The code of the slow answers during which a make of rt's pool ended.
pub(crate) const OVER_MAKE_CODE: &str = "slow-answer-over-make";

/// The code of the longest of them, in nanoseconds, which its count rises to.
pub(crate) const OVER_MAKE_MAX_NS_CODE: &str = "slow-answer-over-make-max-ns";

/// The code of the compute dispatches' first-use waits that met a slow answer.
pub(crate) const FIRST_USE_WAITS_CODE: &str = "slow-answer-first-use-waits";

/// The longest slow answer over a make so far, which [`OVER_MAKE_MAX_NS_CODE`] rises to.
static OVER_MAKE_MOST: AtomicU64 = AtomicU64::new(0);

/// The log target of the lines.
const TARGET: &str = "reims_vgpu::graphics";

thread_local! {
    /// The split of the answer this thread is giving ([`answering`]).
    static ANSWERING: Cell<Option<Split>> = const { Cell::new(None) };
    /// Whether this thread is in a `BLOCK_RELEASE`'s write-back's end ([`writing_back`]).
    static WRITING_BACK: Cell<bool> = const { Cell::new(false) };
}

/// The waits of a slow answer, on rt's host clock (the module's docs, "The split").
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Split {
    /// A `BLOCK_RELEASE`'s: when it was asked, its no-copy buffers had been made, the release's
    /// call returned (0 where the drain answered inside it), the write-back ended (0 where it had
    /// not), the drain ended, and the unshares ended; and whether the write-back ended the drain.
    Release {
        asked: u64,
        pages: u64,
        released: u64,
        written: u64,
        drained: u64,
        unshared: u64,
        write_back_last: bool,
    },
    /// A `POOL_GIVE`'s: when it was asked, the block was given, the drain ended, and the block was
    /// freed.
    Give {
        asked: u64,
        given: u64,
        drained: u64,
        freed: u64,
    },
}

/// Milliseconds from `from` to `to` of rt's host clock, 0 where `to` is not after it.
fn ms(from: u64, to: u64) -> f64 {
    to.saturating_sub(from) as f64 / 1e6
}

impl Split {
    /// The split's words for an answer given at `answered`.
    fn words(&self, answered: u64) -> String {
        match *self {
            Split::Release {
                asked,
                pages,
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
                    "1 the no-copy buffers {:.3} ms, 2 the SURFACE_PAGES settle and the release's \
                     call {:.3} ms, then {:.3} ms until {last} ended{written}, the unshares \
                     {:.3} ms, 4 no display showing it {:.3} ms",
                    ms(asked, pages),
                    ms(pages, released),
                    ms(released, drained),
                    ms(drained, unshared),
                    ms(unshared, answered)
                )
            }
            Split::Give {
                asked,
                given,
                drained,
                freed,
            } => format!(
                "the give's call {:.3} ms, 3 the context's command buffers {:.3} ms, the free \
                 {:.3} ms, then {:.3} ms to the answer",
                ms(asked, given),
                ms(given, drained),
                ms(drained, freed),
                ms(freed, answered)
            ),
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
            released: at(&self.released),
            written: at(&self.written),
            drained: at(&self.drained),
            unshared: at(&self.unshared),
            write_back_last: self.write_back_last.load(Ordering::Relaxed),
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
    let makes = overlap::pool_makes_ended(start, end);
    if let Some(line) = counted(kind, took_ns, end, makes, waits, split) {
        log::info!(target: TARGET, kind:% = name(kind); "{line}");
    }
}

/// Counts a slow answer to a command of type `kind`, `took_ns` long and given at `end`, during
/// which `makes` of rt's pool ended, the latest at `latest`, and which met `waits` first-use waits
/// not counted before; its line where it is over a make, with its split where it has one.
fn counted(
    kind: u32,
    took_ns: u64,
    end: u64,
    (makes, latest): (u64, u64),
    waits: u64,
    split: Option<Split>,
) -> Option<String> {
    count_kind_by(FIRST_USE_WAITS_CODE, waits);
    if makes == 0 {
        return None;
    }
    count_kind(OVER_MAKE_CODE);
    let was = OVER_MAKE_MOST.fetch_max(took_ns, Ordering::Relaxed);
    count_kind_by(OVER_MAKE_MAX_NS_CODE, took_ns.saturating_sub(was));
    let name = name(kind);
    let waits = match waits {
        0 => String::new(),
        n => format!(", over {n} first-use waits of compute pipelines"),
    };
    let split = match split {
        Some(split) => format!(": {}", split.words(end)),
        None => String::new(),
    };
    Some(format!(
        "a {name} (type {kind:#x}) answered {:.3} ms after it was asked, over {makes} makes of \
         rt's pool that ended during it, the last {:.3} ms before the answer{waits}{split}",
        took_ns as f64 / 1e6,
        ms(latest, end)
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The process's count of `code`.
    fn count(code: &str) -> u64 {
        (reims_vgpu_vulkan::answer::degradation_counts().iter())
            .find(|(c, _)| c == code)
            .map_or(0, |&(_, n)| n)
    }

    /// A slow answer during which no make of rt's pool ended is not counted over a make and writes
    /// no line, but the first-use waits it met are counted
    #[test]
    fn a_slow_answer_over_no_make_counts_only_its_first_use_waits() {
        let (over, waits) = (count(OVER_MAKE_CODE), count(FIRST_USE_WAITS_CODE));
        let line = counted(RGFX_CMD_POOL_TAKE, 20_000_000, 50_000_000, (0, 0), 2, None);
        assert_eq!(line, None);
        assert_eq!(count(OVER_MAKE_CODE), over);
        assert_eq!(count(FIRST_USE_WAITS_CODE), waits + 2);
    }

    /// A slow `BLOCK_RELEASE` over a make is counted, and its line gives the makes, the first-use
    /// waits and its split, the drain's end told as the write-back's where the drain answered
    /// inside it, and the release's call ending at the drain where the drain answered inside it
    #[test]
    fn a_slow_release_over_a_make_counts_and_gives_its_split() {
        let over = count(OVER_MAKE_CODE);
        let ms = 1_000_000;
        let split = Split::Release {
            asked: 100 * ms,
            pages: 101 * ms,
            released: 0,
            written: 110 * ms,
            drained: 110 * ms,
            unshared: 111 * ms,
            write_back_last: true,
        };
        let line = counted(
            RGFX_CMD_BLOCK_RELEASE,
            20 * ms,
            120 * ms,
            (2, 115 * ms),
            1,
            Some(split),
        );
        let line = line.unwrap_or_default();
        assert_eq!(count(OVER_MAKE_CODE), over + 1);
        assert!(line.contains("a BLOCK_RELEASE (type 0x"), "{line}");
        assert!(line.contains("answered 20.000 ms"), "{line}");
        assert!(line.contains("over 2 makes"), "{line}");
        assert!(line.contains("the last 5.000 ms before the answer"), "{line}");
        assert!(line.contains("over 1 first-use waits"), "{line}");
        assert!(line.contains("1 the no-copy buffers 1.000 ms"), "{line}");
        assert!(line.contains("call 9.000 ms, then 0.000 ms until 5 the"), "{line}");
        assert!(
            line.contains("the unshares 1.000 ms, 4 no display showing it 9.000 ms"),
            "{line}"
        );
    }

    /// The longest slow answer over a make rises to its longest, and a shorter one leaves it
    #[test]
    fn the_longest_slow_answer_over_a_make_counts_its_rise() {
        let base = OVER_MAKE_MOST.load(Ordering::Relaxed);
        let before = count(OVER_MAKE_MAX_NS_CODE);
        let end = base + 9_000;
        let _ = counted(RGFX_CMD_POOL_GIVE, base + 3_000, end, (1, base), 0, None);
        assert_eq!(count(OVER_MAKE_MAX_NS_CODE), before + 3_000);
        let _ = counted(RGFX_CMD_POOL_GIVE, base + 1_000, end, (1, base), 0, None);
        assert_eq!(count(OVER_MAKE_MAX_NS_CODE), before + 3_000);
    }

    /// A drain that ends inside the write-back's end is told as the write-back's; one outside as
    /// the command buffers'; and an answer's split is read inside its call and gone after it
    #[test]
    fn the_drains_end_is_told_by_the_thread_it_ran_in() {
        let marks = ReleaseMarks::asked();
        let last = |marks: &ReleaseMarks| match marks.split() {
            Split::Release {
                write_back_last, ..
            } => write_back_last,
            Split::Give { .. } => false,
        };
        writing_back(|| marks.drained());
        assert!(last(&marks));
        marks.drained();
        assert!(!last(&marks));
        let split = marks.split();
        let mut seen = None;
        answering(split, || seen = ANSWERING.with(Cell::get));
        assert_eq!(seen, Some(split));
        assert_eq!(ANSWERING.with(Cell::get), None);
    }
}
