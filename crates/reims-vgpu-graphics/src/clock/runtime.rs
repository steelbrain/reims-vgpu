//! The platform's GPU times on the device's clock: every GPU time the platform answers that
//! reaches the guest is placed here, on the clock the guest's kext maps the device's times by
//! (`HOST_TIME`, [`super`]). One call does it: a command buffer's GPU start and end, as
//! `reims-vgpu`'s decoder writes them into its `CB_COMPLETE` (`graphics/decode/cmdbuf.rs`).
//!
//! **Which clock the platform's times count** is [`PLATFORM`]:
//! - metal2vulkan's runtime, which the library runs today (`metal2vulkan-runtime`), counts
//!   `CLOCK_MONOTONIC_RAW` on Linux and the performance counter on Windows (**read**: its
//!   `gpu_time.rs`, `BASE`, at the commit the workspace pins), the device's own clocks, so
//!   [`PLATFORM`] is [`Domain::Device`] and the times pass unchanged: what the guest sees is
//!   what it saw before this module. On macOS it counts `CLOCK_UPTIME_RAW`, which stops while
//!   the Mac sleeps, where the device's `CLOCK_MONOTONIC_RAW` does not, so there the two differ by
//!   the time the host slept since it booted. The times pass unchanged there too, as they did
//!   before this module; the guest reads only a run's length (`gpu_end_ns - gpu_start_ns`), which
//!   that gap leaves exact.
//! - metal2vulkan's new runtime (`metal2vulkan-rt`) counts `CLOCK_MONOTONIC` on Linux and the
//!   other Unix systems but macOS, `CLOCK_MONOTONIC_RAW` on macOS and the performance counter on
//!   Windows (**read**: its `times.rs`, `host_time` and `HOST_DOMAIN`). metal2vulkan's ruling of
//!   2026-11-18 keeps Linux on `CLOCK_MONOTONIC`, as lavapipe calibrates its GPU clock against no
//!   `CLOCK_MONOTONIC_RAW`. At the cutover [`PLATFORM`] is `Domain::Monotonic` on Linux, and
//!   stays [`Domain::Device`] on macOS and Windows.
//! - Should the new runtime count `CLOCK_MONOTONIC_RAW` on Linux too (once lavapipe calibrates
//!   it), every OS's times are the device's, and this module goes, with its one call.
//!
//! **A paired read, fresh at each conversion** (`pair.rs`): the platform's clock, the device's,
//! and the platform's again, back to back, `TRIES` times, and the pair whose two platform reads
//! lie closest (its bracket) places the times, within half the bracket of the clocks' offset.
//! **Measured** 2026-11-18 on homelab-um870-1 (Ryzen 7 8745H, CachyOS, Linux 7.3.3, the vDSO's
//! clocks), three runs of 100 000 pairs (`pair.rs`'s `the_bracket_of_a_pair_on_this_host`): the
//! least bracket of three tries was 40 ns at the median, 40 to 100 ns at the 99th percentile and
//! 60 to 140 ns at most, where a single try's reached 5.9 to 12.3 us (a read the host preempted);
//! a conversion took 209 to 232 ns. A pair is read for each command buffer, not kept: NTP moves
//! `CLOCK_MONOTONIC` against the raw clock by the host's frequency correction (tens of ppm,
//! [`super`]'s `unix.rs`) and faster while it slews off a large offset, and a kept pair would
//! place every time off by that rate times the pair's age, an error the kext's bound, 50 ppm of
//! the age of its own reading of `HOST_TIME` (`guest/kext/src/RgfxVblankClock.hpp`), does not
//! count. A fresh pair adds half its bracket, and the clocks' rate gap over the age of the time
//! it places: a GPU end is placed as its completion is handled, microseconds to milliseconds
//! after it, which at the tens of ppm NTP holds comes to tens of nanoseconds at most.
//!
//! **A run's two times move together**: one pair places both, so its length, the GPU's own
//! count between its two timestamps and what the guest reads, stays exact.
//!
//! **A pair of times the guest maps itself keeps one clock.** An answer that pairs a CPU time with
//! a GPU time, as `sampleTimestamps:gpuTimestamp:` does, takes the CPU's from the device's clock
//! and the GPU's through here. Today none reaches the guest: its bundle answers
//! `sampleTimestamps:gpuTimestamp:` with its own `mach_absolute_time` for both, its kernel times
//! from its own clock, and it samples no counters (`guest/bundle/src`).

#[cfg(all(unix, not(target_os = "macos")))]
mod pair;

/// A clock the platform's times may count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Domain {
    /// The device's own clock: the times pass as they are.
    Device,
    /// `CLOCK_MONOTONIC`, which NTP disciplines, the new runtime's on Unix but macOS: each time is
    /// moved onto the device's clock by a paired read (`pair.rs`).
    #[cfg(all(unix, not(target_os = "macos")))]
    Monotonic,
}

/// The clock the platform's GPU times count: the device's, as the runtime the library runs counts
/// (the module documentation).
pub const PLATFORM: Domain = Domain::Device;

/// A command buffer's GPU start and end, `times`, in nanoseconds of [`PLATFORM`]'s clock, on the
/// device's: the same values where the platform counts the device's clock, as today, at the cost
/// of no branch the compiler keeps.
#[inline]
pub fn gpu_times(times: (u64, u64)) -> (u64, u64) {
    in_domain(PLATFORM, times)
}

/// `times`, a run's start and end on `domain`'s clock, on the device's.
#[inline]
fn in_domain(domain: Domain, times: (u64, u64)) -> (u64, u64) {
    match domain {
        Domain::Device => times,
        #[cfg(all(unix, not(target_os = "macos")))]
        Domain::Monotonic => pair::Pair::read(super::monotonic_ns, super::now_ns).place_run(times),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The runtime the library runs counts the device's clock: its GPU times reach the guest as
    /// the platform answers them, bit for bit
    #[test]
    fn the_runtime_s_gpu_times_pass_unchanged() {
        assert_eq!(PLATFORM, Domain::Device);
        for times in [
            (1, 2),
            (5_000_000_000, 5_002_000_000),
            (7_000, 7_000),
            (u64::MAX - 1, u64::MAX),
        ] {
            assert_eq!(gpu_times(times), times);
            assert_eq!(in_domain(Domain::Device, times), times);
        }
    }

    /// `CLOCK_MONOTONIC`'s times are moved onto the device's clock, a run's length unchanged
    #[cfg(all(unix, not(target_os = "macos")))]
    #[test]
    fn a_monotonic_run_keeps_its_length_on_the_device_s_clock() {
        use crate::clock::monotonic_ns;
        let start = monotonic_ns();
        let (placed, end) = in_domain(Domain::Monotonic, (start, start + 1_234));
        assert_eq!(end - placed, 1_234);
    }
}
