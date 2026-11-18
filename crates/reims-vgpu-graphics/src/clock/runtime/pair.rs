//! The paired read that places a time of the platform's clock on the device's: the platform's
//! clock, the device's, and the platform's again, back to back, [`TRIES`] times, and the pair
//! whose two platform reads lie closest (its bracket) places the time, by the device's read less
//! the midpoint of the platform's: within half the bracket of the clocks' offset when it was
//! read. On Unix but macOS, where the platform's clock may be `CLOCK_MONOTONIC`
//! ([`super::Domain::Monotonic`]).

/// Pairs read for a placement, of which the one of the least bracket places the times.
pub(super) const TRIES: usize = 3;

/// One paired read of the platform's clock and the device's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Pair {
    /// The platform's clock halfway between its two reads.
    platform: u64,
    /// The device's clock, read between them.
    device: u64,
    /// How far apart the platform's two reads lie: the pair stands within half of it of the
    /// clocks' offset when it was read.
    bracket: u64,
}

impl Pair {
    /// The pair of the least bracket of [`TRIES`], of the platform's clock `platform` and the
    /// device's `device`.
    pub(super) fn read(platform: impl Fn() -> u64, device: impl Fn() -> u64) -> Pair {
        let mut least = Pair::once(&platform, &device);
        for _ in 1..TRIES {
            let next = Pair::once(&platform, &device);
            if next.bracket < least.bracket {
                least = next;
            }
        }
        least
    }

    /// One read of `platform`, `device` and `platform`, back to back.
    fn once(platform: &impl Fn() -> u64, device: &impl Fn() -> u64) -> Pair {
        let before = platform();
        let device = device();
        let after = platform();
        let bracket = after.saturating_sub(before);
        Pair {
            platform: before.saturating_add(bracket / 2),
            device,
            bracket,
        }
    }

    /// `t`, a time of the platform's clock, on the device's: 0 where it falls before the device's
    /// clock began, `u64::MAX` past its end.
    fn place(&self, t: u64) -> u64 {
        let placed = i128::from(t) + i128::from(self.device) - i128::from(self.platform);
        u64::try_from(placed.max(0)).unwrap_or(u64::MAX)
    }

    /// A run's `(start, end)` on the device's clock, both moved by this pair's one offset, so the
    /// run's length stays what the platform counted.
    pub(super) fn place_run(&self, (start, end): (u64, u64)) -> (u64, u64) {
        (self.place(start), self.place(end))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    /// A fake pair of clocks: the device's at `now` nanoseconds, which each read of either moves
    /// on by `step`, and the platform's `offset` from it, slewed by `ppm` parts per million of the
    /// device's time since 0.
    struct Clocks {
        now: Cell<u64>,
        step: u64,
        offset: i128,
        ppm: i128,
    }

    impl Clocks {
        fn platform_at(&self, device: u64) -> u64 {
            let at = i128::from(device);
            u64::try_from(at + self.offset + at * self.ppm / 1_000_000).unwrap()
        }
        fn tick(&self) -> u64 {
            let now = self.now.get();
            self.now.set(now + self.step);
            now
        }
        fn platform(&self) -> u64 {
            let now = self.tick();
            self.platform_at(now)
        }
        fn device(&self) -> u64 {
            self.tick()
        }
    }

    /// Over hours of a platform clock that NTP slews against the device's, at 500 ppm either way
    /// or at the tens of ppm hosts were measured at, ahead of the device's or behind it, a time of
    /// the platform's placed by a fresh pair lands within half the pair's bracket, and the slew
    /// over the time's age, of the device's time it was taken at
    #[test]
    fn a_fresh_pair_places_a_slewing_clock_s_times_within_its_bound() {
        for (offset, ppm) in [
            (3_000_000_000_i128, 500_i128),
            (3_000_000_000, -500),
            (-1_000_000_000, 45),
            (-1_000_000_000, -30),
        ] {
            for start in [10_000_000_000_u64, 3_600_000_000_000, 36_000_000_000_000] {
                let clocks = Clocks {
                    now: Cell::new(start),
                    step: 40,
                    offset,
                    ppm,
                };
                for age in [0_u64, 50_000, 2_000_000, 1_000_000_000] {
                    let taken = clocks.now.get() - age;
                    let stamp = clocks.platform_at(taken);
                    let pair = Pair::read(|| clocks.platform(), || clocks.device());
                    let placed = pair.place(stamp);
                    let off = placed.abs_diff(taken);
                    let slew = u64::try_from(ppm.unsigned_abs()).unwrap() * age / 1_000_000;
                    assert!(
                        off <= pair.bracket / 2 + slew + 2,
                        "offset {offset} ppm {ppm} start {start} age {age}: {placed} for \
                         {taken}, bracket {}",
                        pair.bracket
                    );
                }
            }
        }
    }

    /// Of the tries, the pair whose platform reads lie closest places the times: a try the host
    /// preempted between its reads is passed over
    #[test]
    fn the_pair_of_the_least_bracket_places_the_times() {
        let reads = Cell::new(0_u32);
        // The first try's second platform read comes 100 us after the read before it; every other
        // read 50 ns after the one before. The platform's clock is the device's plus 1 s.
        let now = Cell::new(1_000_000_000_u64);
        let read = || {
            let n = reads.get();
            reads.set(n + 1);
            now.set(now.get() + if n == 2 { 100_000 } else { 50 });
            now.get()
        };
        let pair = Pair::read(|| read() + 1_000_000_000, read);
        assert_eq!(reads.get(), 3 * u32::try_from(TRIES).unwrap());
        assert_eq!(pair.bracket, 100);
        assert_eq!(pair.place(pair.platform), pair.device);
        assert!(pair.place(2_000_000_000).abs_diff(1_000_000_000) <= pair.bracket / 2);
    }

    /// One pair moves a run's start and end by one offset, so its length stays the platform's
    #[test]
    fn a_run_s_length_stays_exact() {
        let pair = Pair {
            platform: 10_000_000_123,
            device: 9_999_000_000,
            bracket: 60,
        };
        let (start, end) = pair.place_run((10_000_000_000, 10_002_345_678));
        assert_eq!(end - start, 2_345_678);
        assert_eq!(start, 9_998_999_877);
    }

    /// A time that falls before the device's clock began is 0, one past its end `u64::MAX`: no
    /// placement wraps
    #[test]
    fn a_placement_saturates() {
        let behind = Pair {
            platform: 5_000,
            device: 1_000,
            bracket: 0,
        };
        assert_eq!(behind.place(3_000), 0);
        let ahead = Pair {
            platform: 1_000,
            device: 5_000,
            bracket: 0,
        };
        assert_eq!(ahead.place(u64::MAX - 1), u64::MAX);
    }

    /// A read of `CLOCK_MONOTONIC` placed by a pair lies within half the pair's bracket of the
    /// `CLOCK_MONOTONIC_RAW` reads on either side of it, give or take the clocks' rate gap over
    /// the microseconds between
    #[test]
    fn a_monotonic_read_lands_between_the_raw_reads_around_it() {
        use crate::clock::{monotonic_ns, now_ns};
        for _ in 0..3 {
            let before = now_ns();
            let monotonic = monotonic_ns();
            let after = now_ns();
            let pair = Pair::read(monotonic_ns, now_ns);
            let read = now_ns();
            let placed = pair.place(monotonic);
            // Half the bracket, and the clocks' rate gap over the time since the read at 0.1%,
            // twice the most NTP corrects a clock's frequency by (Linux's `MAXFREQ`, 500 ppm).
            let slack = pair.bracket / 2 + (read - before) / 1_000 + 1;
            assert!(
                before.saturating_sub(slack) <= placed && placed <= after + slack,
                "{before} {placed} {after}, bracket {}",
                pair.bracket
            );
        }
    }

    /// The brackets of a pair on this host's `CLOCK_MONOTONIC` and `CLOCK_MONOTONIC_RAW`, and what
    /// [`Pair::read`] costs: a measurement, run by hand (`cargo nextest run -p reims-vgpu-graphics
    /// --run-ignored only --no-capture -E 'test(/the_bracket_of_a_pair_on_this_host/)'`), whose
    /// numbers the module documentation above this one gives.
    #[test]
    #[ignore = "a measurement, run by hand"]
    fn the_bracket_of_a_pair_on_this_host() {
        use crate::clock::{monotonic_ns, now_ns};
        const PAIRS: u32 = 100_000;
        let pct = |sorted: &[u64], per_mille: usize| sorted[(sorted.len() - 1) * per_mille / 1000];
        let mut once: Vec<u64> = (0..PAIRS)
            .map(|_| Pair::once(&monotonic_ns, &now_ns).bracket)
            .collect();
        let started = now_ns();
        let mut least: Vec<u64> = (0..PAIRS)
            .map(|_| Pair::read(monotonic_ns, now_ns).bracket)
            .collect();
        let per_read = (now_ns() - started) / u64::from(PAIRS);
        once.sort_unstable();
        least.sort_unstable();
        for (what, sorted) in [("one try", &once), ("least of TRIES", &least)] {
            println!(
                "{what}: bracket p50 {} ns, p99 {} ns, p99.9 {} ns, most {} ns",
                pct(sorted, 500),
                pct(sorted, 990),
                pct(sorted, 999),
                sorted[sorted.len() - 1]
            );
        }
        println!("Pair::read: {per_read} ns a placement");
    }
}
