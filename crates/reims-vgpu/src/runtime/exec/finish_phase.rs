//! What [`crate::runtime::drain::ExecPhase::Finish`] is, split by the part of
//! `finish_stream` that spent it.
//!
//! `Finish` is the largest phase in this device and it *contains* the draw
//! encode, so the interesting quantity is the difference. One driven Maps boot,
//! macos-13 rail, x86/Vulkan, banded to its 44 driven census windows and summed
//! over 2 590 808 draws:
//!
//! ```text
//! op0x37_us  11.238   the whole EXEC_INDIRECT2 dispatch
//!   finish     9.606
//!     draw_us  8.362   what `drain_duty` names, i.e. `encode_draw_chain`
//!     ?        1.244   <- this
//!   walk       1.178
//!   header     0.239
//!   preflight  0.163
//!   load       0.046
//! ```
//!
//! That residue is 15 % of the whole draw path and larger than any single bar
//! in `draw_phase`, whose largest — `pipeline_us` — is 0.59. It had no field,
//! so every statement about it was a guess. This is the same division the
//! `Record`, `Stage` and pipeline groups already got inside the engine, in the
//! same shape: lexical spans that tile the function, with the remainder carved
//! out as [`FinishPhase::Prelude`] rather than added beside it, so the parts sum
//! to `finish_us` by construction and the identity is checkable on the line.
//!
//! # What it measured, and what that rules out
//!
//! Driven Maps boot G0, same rail and host, `throttle_ms=0`, 44 driven windows
//! over 2 143 436 draws (`sum` 19.80 µs/draw, 47.6 fps at 1024 draws a frame):
//!
//! ```text
//! fin_encode_us    9.475   `encode_draw_chain`, which `draw_us` also spans
//! fin_prelude_us   0.547   per *stream*: 11.0 µs each, at 20.1 draws a stream
//! fin_binds_us     0.136
//! fin_result_us    0.079
//! fin_retarget_us  0.065
//! fin_tail_us      0.007
//! ```
//!
//! They sum to 10.309 against `finish_us` 10.314, so the tiling holds on a real
//! boot and not only in the test.
//!
//! **The residue is not the per-record request materialization**, which is what
//! the split was built expecting. A `MTLRenderCommandEncoder` holds its
//! attachment set for its life and its argument tables are sticky across the
//! draws issued on it, so a stream states them once and issues N draw records —
//! and this device turns each record back into a whole `DrawRequest` by cloning
//! a template ([`FinishPhase::Retarget`]) and refilling the bind lists
//! ([`FinishPhase::Binds`]). Together that is **0.20 µs a draw, 1 % of the
//! frame**. It is cheap because [`crate::runtime::draw::BindTable`] is an `Arc`,
//! so refilling six sticky tables is twelve atomics and not six copies — the
//! sticky-table model is already paid for at the accumulator, one layer up.
//!
//! Do not go looking here again. What is left is [`FinishPhase::Prelude`], and
//! it is a **per-pass** cost rather than a per-draw one: 11.0 µs per stream,
//! almost all of it the one `mrt_draw_request` that resolves the colour slots
//! for record 0. Halving it would buy 0.27 µs of a 19.80 µs draw.
//!
//! # What the census costs
//!
//! One `Instant::now()` per phase transition, accumulated in a local array and
//! flushed to the shared census once per stream. A packet of ninety draws
//! therefore pays twelve atomics rather than a thousand — the same trade
//! `reims_vgpu_vulkan::engine::draw_phase`'s timer makes per draw.

use crate::runtime::drain::{note_finish_phase, FinishPhase};
use std::time::Instant;

/// Charges `finish_stream`'s wall clock to whichever part is open, and flushes
/// the whole stream's tiling on `Drop`.
///
/// Opens in [`FinishPhase::Prelude`], which is why that variant is the
/// remainder: any span not explicitly entered is charged to it.
pub(crate) struct FinishTimer {
    ns: [u64; FinishPhase::COUNT],
    n: [u64; FinishPhase::COUNT],
    open: usize,
    last: Instant,
}

impl FinishTimer {
    pub(crate) fn open() -> Self {
        let now = Instant::now();
        let mut n = [0u64; FinishPhase::COUNT];
        n[FinishPhase::Prelude.index()] = 1;
        Self {
            ns: [0; FinishPhase::COUNT],
            n,
            open: FinishPhase::Prelude.index(),
            last: now,
        }
    }

    /// Close the open part and open `next`. The count is per entry rather than
    /// per stream, so `fin_encode_n` is the draws this stream encoded and
    /// `fin_retarget_n` the records that were retargeted — the denominators the
    /// microseconds need.
    pub(crate) fn enter(&mut self, next: FinishPhase) {
        let now = Instant::now();
        self.ns[self.open] += crate::observe::phase_clock::charge_ns(now.duration_since(self.last));
        self.last = now;
        self.open = next.index();
        self.n[self.open] += 1;
    }

    /// Close the part that is still open. Separate from [`Drop`] so a test can
    /// read the tiling without going through the process-wide census.
    fn close(&mut self) {
        let now = Instant::now();
        self.ns[self.open] += crate::observe::phase_clock::charge_ns(now.duration_since(self.last));
        self.last = now;
    }
}

impl Drop for FinishTimer {
    fn drop(&mut self) {
        self.close();
        for phase in FinishPhase::ALL {
            let i = phase.index();
            if self.ns[i] == 0 && self.n[i] == 0 {
                continue;
            }
            note_finish_phase(phase, self.ns[i], self.n[i]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole point of the split is that the parts add back up to what
    /// `finish_us` was alone. A part that is entered and never closed, or one
    /// closed into the wrong slot, breaks that identity — and it breaks it
    /// silently, because every individual bar still looks plausible.
    #[test]
    fn the_parts_tile_the_timer_and_nothing_falls_between_them() {
        const SLEEP: std::time::Duration = std::time::Duration::from_millis(4);
        let started = Instant::now();
        let mut fin = FinishTimer::open();
        for _ in 0..4 {
            fin.enter(FinishPhase::Retarget);
            fin.enter(FinishPhase::Binds);
            fin.enter(FinishPhase::Encode);
            // The one span with a floor under it, so "this phase was charged"
            // is a real assertion rather than a claim about a few nanoseconds.
            std::thread::sleep(SLEEP);
            fin.enter(FinishPhase::Result);
        }
        fin.enter(FinishPhase::Tail);
        fin.close();
        let whole = crate::observe::phase_clock::charge_ns(started.elapsed());
        let tiled: u64 = fin.ns.iter().sum();
        assert!(
            tiled <= whole,
            "the parts cannot exceed the span they divide: {tiled} > {whole}"
        );
        // `charge_ns` does not truncate, so the only thing outside the tiling is
        // the clock read `open` makes before the first part starts. A gap wider
        // than one sleep means a whole span was charged to nobody.
        assert!(
            whole - tiled < SLEEP.as_nanos() as u64,
            "a span fell between the parts: tiled {tiled} of {whole}"
        );
        // And it landed where it was spent, not spread across the neighbours.
        let encode = fin.ns[FinishPhase::Encode.index()];
        assert!(
            encode >= 4 * SLEEP.as_nanos() as u64,
            "the sleeps were charged somewhere other than the open part: {encode}"
        );
    }

    /// `_n` is a per-*entry* count, not a per-stream one. It is the denominator
    /// every microsecond on the line is read against — `fin_retarget_n` is the
    /// draw-list trip count and `fin_encode_n` the draws that encoded — so a
    /// timer that reported 1 per stream would make every per-draw figure the
    /// packet's draw count too large.
    #[test]
    fn each_entry_counts_and_the_prelude_starts_open() {
        let mut fin = FinishTimer::open();
        assert_eq!(fin.n[FinishPhase::Prelude.index()], 1);
        for _ in 0..3 {
            fin.enter(FinishPhase::Retarget);
            fin.enter(FinishPhase::Encode);
        }
        fin.enter(FinishPhase::Tail);
        assert_eq!(fin.n[FinishPhase::Retarget.index()], 3);
        assert_eq!(fin.n[FinishPhase::Encode.index()], 3);
        assert_eq!(fin.n[FinishPhase::Tail.index()], 1);
        assert_eq!(fin.n[FinishPhase::Binds.index()], 0);
        // Re-entering never re-opens the prelude: it is the remainder, and a
        // second entry would mean some later span was charged to it.
        assert_eq!(fin.n[FinishPhase::Prelude.index()], 1);
    }
}
