//! What `draw_phase`'s `stage_us` is, split by the mechanism that would fix it.
//!
//! `stage_us` **was** the largest single column in the device, and this split is
//! why it is not any more. On a driven Safari drag, settled x86/PCI, host GPU at
//! P8: 3978 draws in one second spent **200 ms** there, which was 83 % of
//! `draw_phase`'s whole second — everything else in that phase together
//! (pipeline, record, sampled, prep, descriptors, submit) was ~41 ms. The same
//! probe on the same host now reads **10 ms**; see "What that was worth" below
//! for what moved and what did not.
//!
//! One bar cannot choose between the fixes, which is the same argument that
//! split `setup` into four and `binds_us` into three. The phase covers six
//! populations and they want opposite work:
//!
//! | part | what it is | what would fix it |
//! |---|---|---|
//! | `acquire` | [`ResourcePools::acquire_staging`] | pool sizing; a miss creates a buffer and allocates memory |
//! | `bytes` | `write_staging` from a `BufferContent::Bytes` | **the second copy** — those bytes were already assembled out of guest RAM by `load_buffer_content`, which `bind_phase` charges separately |
//! | `runs` | `write_staging_from_runs` | **taking the copy off the CPU** — it is one copy, but it is the CPU's; see below. Zero on a host that can import |
//! | `gather` | `exec`'s `gather_guest_buffer_window` | nothing on the CPU side — this is what `runs` gave way to, and it plans a GPU copy rather than making one |
//! | `swap` | `write_staging_swap_rb` on a seed | nothing — it is the copy that had to happen, with a byte exchange folded in |
//! | `shift` | the `base_instance` prefix a Constant-step vertex stream needs | keeping those binds off the CPU path |
//!
//! The `bytes`/`runs` division is the one to read first, because it prices a
//! lever nobody has costed: `BufferContent::Bytes` arrives as an
//! `Arc<Vec<u8>>` that `load_buffer_content` filled from guest memory, and
//! staging it copies the same bytes a second time. `BufferContent::GuestRuns`
//! does not — `write_staging_from_runs` exists precisely because the deferred
//! snapshot path used to `cpu_bytes()` into a heap `Vec` and then
//! `write_staging` that, and removing the intermediate was worth two copies and
//! an allocation per bind. Whether the *other* arm is still paying that is what
//! `bytes_us` against `runs_us` answers, and the byte counters beside them say
//! at what rate.
//!
//! # That lever has now been costed, and it is the wrong one
//!
//! A driven Safari drag, x86/PCI, quiesced, one `vk_caps`, one census second:
//!
//! ```text
//! bytes_us=448     bytes_n=3498    bytes_b=9 606 736
//! runs_us=104719   runs_n=15758    runs_b=3 627 029 280
//! ```
//!
//! So the second copy the paragraph above suspects is **0.4 % of the phase**,
//! and `runs` is 93 % of it: 105 ms of CPU memcpy per second, moving **3.6 GB/s**
//! out of guest RAM. Against a `draw_phase` whose whole second is ~156 ms, that
//! one memcpy is 67 % of every draw's cost. Chasing `bytes` would have been a
//! rounding error, which is exactly what the split was built to prevent.
//!
//! The `runs` row's fix is therefore not "move fewer bytes" — the bytes are the
//! guest's vertex data and every one of them is needed. It is to stop the *CPU*
//! moving them, which is what `exec`'s `gather_guest_buffer_window` now does:
//! the host-pointer import covers the whole RAMBlock, so a scattered window is
//! one `vkCmdCopyBuffer` per stretch out of that import into a device-local
//! destination, recorded ahead of the draw's own render pass.
//!
//! What had kept every bind on this memcpy was the shape and not the mechanism:
//! `GuestRunSource::pages` was a single `GuestRef`, so a window that is not one
//! GPA-contiguous stretch had nowhere to go — and none of them is one, because
//! the guest backs a surface in 16 KiB granules. `zc_buffer_imported` read **0**
//! against `zc_buffer_gathered` 371 422 on the same boot. It is a list of
//! stretches now, from `guest_ram_map::references_for_runs`.
//!
//! # What that was worth, on the same probe
//!
//! Driven Safari drag, x86/PCI, quiesced, one `vk_caps`, the busiest census
//! second of each boot:
//!
//! ```text
//!                 stage_us   runs_us   runs_n   gather_us  gather_n   draws
//! CPU gather       200 000   104 719   15 758           —         —    3 978
//! GPU gather        10 046         0        0       6 417    14 902    3 004
//! ```
//!
//! The memcpy is gone rather than reduced: `runs_n` is **0** for the whole boot,
//! and `buffer_snapshot_binds` with it. What replaces 105 ms/s of moving 3.6 GB
//! is 6.4 ms/s of *planning* 3.43 GB — the same bytes, crossed by the engine
//! that was going to read them, at a sixteenth of the CPU. `stage_us` falls
//! twentyfold and stops being the largest column in the device.
//!
//! **Present cadence did not move**, and that is the honest headline beside it:
//! 24.35 Hz against 24.50 before, which is the same number. This workload's
//! frame rate was not bound by `stage_us` and is not bound by it now — the win
//! is headroom. `fence_us` rose 480 -> 633 us, which is the trade stated
//! plainly: the GPU now does 14 900 gathers a second it did not do before, and
//! the writeback's fence waits behind them. 15 ms/s of fence against 95 ms/s of
//! CPU.
//!
//! # Reading `runs` now
//!
//! On a host that can import, `runs_us` is no longer this rail's whole traffic:
//! it is the windows the gather turned away, and on the boot above there were
//! none. A `runs_us` that has *not* fallen on a capable host means the gather is
//! declining; the `vk_buffer_gather` declines name the check that refused, and
//! the `zc_buf_runs_*` bands say how wide the windows reaching it are. Width
//! itself is no longer a decline — the region ceiling that used to make it one
//! is retired.
//! On a host without `VK_EXT_external_memory_host` every window is still `runs`
//! and always will be: the same probe under `REIMS_VGPU_GUEST_IMPORT=off` read
//! `runs_us=1 403 990` over 288 196 windows with `guest_gather_block=0:0:0`, so
//! the gate closes before the pool, not after it.
//!
//! Timings here are wall clock on a shared machine and are upper bounds; the
//! counts and byte totals are not.
//!
//! # Why the call sites and not the pool
//!
//! The four pool functions are also called from the sampled-image path, which
//! `draw_phase` charges to `acquire_sampled` and `sampled_upload`. Instrumenting
//! the pool would mix those in and the parts would no longer sum to `stage_us`.
//! `stage_buffer_content` is called from inside the `Stage` span and nowhere
//! else, so wrapping it and the four open-coded sites in that span is exact.
//!
//! # What the census costs
//!
//! Two `Instant::now()` per span, and a span per staging operation rather than
//! per draw — a draw with many distinct vertex streams opens several. Measured
//! against the sum: if `acquire + bytes + runs + gather + swap + shift` starts exceeding
//! `draw_phase`'s `stage_us`, the census is the difference and should be read as
//! such. An audit on this project once moved `land_us` from 328 to 380 µs by
//! reading its own subject, so this is not hypothetical.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// The staging operations inside one draw's `Stage` phase.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Part {
    /// A staging slot was taken from the pool, or created.
    Acquire = 0,
    /// Host bytes already in a `Vec` were copied into mapped staging.
    Bytes = 1,
    /// Guest RAM was gathered straight into mapped staging.
    Runs = 2,
    /// A seed was copied with red and blue exchanged.
    Swap = 3,
    /// A Constant-step vertex stream was rebuilt behind its `base_instance`
    /// prefix. This is the one part that is neither a pool call nor a staging
    /// write — it is a `Vec` allocation and a copy before either.
    Shift = 4,
    /// A scattered guest window was assembled by the GPU rather than by the
    /// CPU: the per-stretch import binds, the region planning and the
    /// device-local destination acquire.
    ///
    /// The one part that charges work the CPU does *not* spend moving bytes —
    /// the copy it plans runs later, in the draw's command buffer — so its
    /// `_b` is what the GPU will move and its rate is not a memcpy rate. It is
    /// here rather than left unmeasured because it is what `runs` gives way to,
    /// and a swap that traded 105 ms of memcpy for 105 ms of planning would
    /// otherwise show up only as `stage_us` not moving.
    Gather = 5,
}

const PARTS: usize = 6;

/// Nanoseconds, per [`crate::observe::phase_clock`]. This census opens a span
/// per staging operation at tens of thousands a second, which is exactly the
/// population a microsecond accumulator reports as free.
static NS: [AtomicU64; PARTS] = [const { AtomicU64::new(0) }; PARTS];
static N: [AtomicU64; PARTS] = [const { AtomicU64::new(0) }; PARTS];
static BYTES: [AtomicU64; PARTS] = [const { AtomicU64::new(0) }; PARTS];

/// One window of the split, as taken by the per-second census.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct StagePhaseWindow {
    pub acquire_us: u64,
    pub acquires: u64,
    pub bytes_us: u64,
    pub bytes_n: u64,
    pub bytes_b: u64,
    pub runs_us: u64,
    pub runs_n: u64,
    pub runs_b: u64,
    pub swap_us: u64,
    pub swap_n: u64,
    pub swap_b: u64,
    pub shift_us: u64,
    pub shift_n: u64,
    pub shift_b: u64,
    pub gather_us: u64,
    pub gather_n: u64,
    pub gather_b: u64,
}

/// Take and clear the window. `None` when nothing staged, so an idle second
/// costs no line.
pub fn take_window() -> Option<StagePhaseWindow> {
    let us =
        |p: Part| crate::observe::phase_clock::to_us(NS[p as usize].swap(0, Ordering::Relaxed));
    let n = |p: Part| N[p as usize].swap(0, Ordering::Relaxed);
    let b = |p: Part| BYTES[p as usize].swap(0, Ordering::Relaxed);
    let w = StagePhaseWindow {
        acquire_us: us(Part::Acquire),
        acquires: n(Part::Acquire),
        bytes_us: us(Part::Bytes),
        bytes_n: n(Part::Bytes),
        bytes_b: b(Part::Bytes),
        runs_us: us(Part::Runs),
        runs_n: n(Part::Runs),
        runs_b: b(Part::Runs),
        swap_us: us(Part::Swap),
        swap_n: n(Part::Swap),
        swap_b: b(Part::Swap),
        shift_us: us(Part::Shift),
        shift_n: n(Part::Shift),
        shift_b: b(Part::Shift),
        gather_us: us(Part::Gather),
        gather_n: n(Part::Gather),
        gather_b: b(Part::Gather),
    };
    // `Acquire` carries no bytes, so it is swapped and dropped rather than
    // left to accumulate into a number nothing reads.
    let _ = b(Part::Acquire);
    let staged = w.acquires + w.bytes_n + w.runs_n + w.swap_n + w.shift_n + w.gather_n;
    (staged > 0).then_some(w)
}

/// Charges one staging operation to one part, from `open` to `Drop`.
pub(crate) struct Span {
    part: Part,
    bytes: u64,
    started: Instant,
}

impl Span {
    /// A part with no byte count of its own — the pool call.
    pub(crate) fn open(part: Part) -> Self {
        Self {
            part,
            bytes: 0,
            started: Instant::now(),
        }
    }

    /// A part that moves `bytes`, so the window can state a rate rather than
    /// only a duration.
    pub(crate) fn moving(part: Part, bytes: u64) -> Self {
        Self {
            part,
            bytes,
            started: Instant::now(),
        }
    }
}

impl Drop for Span {
    fn drop(&mut self) {
        let slot = self.part as usize;
        NS[slot].fetch_add(
            crate::observe::phase_clock::charge_ns(self.started.elapsed()),
            Ordering::Relaxed,
        );
        N[slot].fetch_add(1, Ordering::Relaxed);
        BYTES[slot].fetch_add(self.bytes, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A window reports every part it was given, and clears itself so the next
    /// second starts from zero rather than from the boot's running total.
    #[test]
    fn a_window_takes_what_was_charged_and_leaves_nothing() {
        // Other tests in this binary share these statics, so start from a
        // known-empty window rather than assuming one.
        let _ = take_window();
        drop(Span::open(Part::Acquire));
        drop(Span::moving(Part::Bytes, 4096));
        drop(Span::moving(Part::Bytes, 1024));
        drop(Span::moving(Part::Runs, 8192));
        drop(Span::moving(Part::Gather, 65536));
        let w = take_window().expect("something was staged");
        assert_eq!((w.acquires, w.bytes_n, w.bytes_b), (1, 2, 5120));
        assert_eq!((w.runs_n, w.runs_b), (1, 8192));
        assert_eq!((w.swap_n, w.shift_n), (0, 0));
        // A part that is charged and not reported reads exactly like a part
        // nothing reached — which is how three counters once sat at zero for a
        // whole boot because they were never wired into the census.
        assert_eq!((w.gather_n, w.gather_b), (1, 65536));
        assert_eq!(take_window(), None, "a taken window must not repeat");
    }

    /// A window carrying only a gather is still a window.
    ///
    /// `take_window` answers `None` on an idle second so nothing publishes a
    /// line of zeros, and that test is a sum over the parts — so a part left
    /// out of it makes a second that did nothing *but* gather indistinguishable
    /// from an idle one. On a host that can import, that is now the common
    /// second.
    #[test]
    fn a_second_that_only_gathered_still_reports() {
        let _ = take_window();
        drop(Span::moving(Part::Gather, 4096));
        let w = take_window().expect("a gather is staging work");
        assert_eq!(w.gather_n, 1);
        assert_eq!((w.acquires, w.bytes_n, w.runs_n), (0, 0, 0));
    }

    /// A staging operation is sub-microsecond and there are tens of thousands
    /// of them a second, which is the population
    /// [`crate::observe::phase_clock`] exists for: under a microsecond-
    /// truncating accumulator every span here charges exactly zero and the
    /// column this split was built to divide reads free.
    ///
    /// Threshold measured, not guessed — see `runtime::bind_phase`'s twin,
    /// where 20 000 empty spans read 302-308 µs against 3 truncating.
    #[test]
    fn twenty_thousand_sub_microsecond_spans_are_not_free() {
        let _ = take_window();
        for _ in 0..20_000 {
            drop(Span::moving(Part::Bytes, 1));
        }
        let w = take_window().expect("something was staged");
        assert_eq!(w.bytes_n, 20_000);
        assert!(w.bytes_us > 100, "{w:?}");
    }

    /// The byte counters are per part. A part charged bytes must not leak them
    /// into another, or `bytes_us` and `runs_us` cannot be compared at all —
    /// which is the whole reason this split exists.
    #[test]
    fn bytes_do_not_cross_between_parts() {
        let _ = take_window();
        drop(Span::moving(Part::Swap, 777));
        drop(Span::moving(Part::Shift, 88));
        let w = take_window().expect("something was staged");
        assert_eq!((w.swap_b, w.shift_b), (777, 88));
        assert_eq!((w.bytes_b, w.runs_b), (0, 0));
    }
}
