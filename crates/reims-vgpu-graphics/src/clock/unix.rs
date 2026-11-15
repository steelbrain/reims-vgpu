//! The clock on Linux and macOS: `CLOCK_MONOTONIC_RAW`, which runs at the counter's own rate, as
//! a guest's clock does (an x86-64 guest's `mach_absolute_time` counts its TSC; an arm64 guest's
//! counter is the host's). `CLOCK_MONOTONIC` runs at the rate NTP disciplines it to, so it drifts
//! from the guest's clock by the host's frequency correction (**measured** 2026-11-15 against
//! `CLOCK_MONOTONIC_RAW`: +45.1 ppm on homelab-um790-xtx-1, -30.1 ppm on homelab-gem10-1 and
//! +5.7 ppm on homelab-um870-1, systemd-timesyncd, over 5 s; +3.9 ppm on the M3 Max over 10 s).
//!
//! The kext maps the device's times by its reading of `HOST_TIME`, which it takes only at a
//! `VBLANK`, and widens a held reading's bound by 50 ppm of its age
//! (`guest/kext/src/RgfxVblankClock.hpp`): on the raw clock that holds whatever NTP does, where a
//! slew correcting a large offset can pass 50 ppm. On `CLOCK_MONOTONIC` the host's map of the
//! guest's times of its command buffers (`device::guest_times`), by the offset the kext last
//! wrote, showed homelab-um790-xtx-1's 45 ppm as a transport that climbed to 230 us between
//! readings 5 s apart, while `VBLANK`s were off.

/// The host's monotonic clock at the counter's rate, in nanoseconds.
pub(crate) fn now_ns() -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `clock_gettime` writes the timespec.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC_RAW, &mut ts) };
    let secs = u64::try_from(ts.tv_sec).unwrap_or(0);
    let nanos = u64::try_from(ts.tv_nsec).unwrap_or(0);
    secs.saturating_mul(1_000_000_000).saturating_add(nanos)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A read of the system's clock `id`, in nanoseconds.
    fn read(id: libc::clockid_t) -> u64 {
        let mut ts = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        // SAFETY: `clock_gettime` writes the timespec.
        unsafe { libc::clock_gettime(id, &mut ts) };
        u64::try_from(ts.tv_sec).unwrap() * 1_000_000_000 + u64::try_from(ts.tv_nsec).unwrap()
    }

    /// The device's clock is the raw monotonic one, at the rate a guest's clock follows: a read of
    /// it lies between two reads of `CLOCK_MONOTONIC_RAW` (`CLOCK_MONOTONIC` lies tens of
    /// microseconds a second of the host's uptime away from it wherever NTP corrects the clock)
    #[test]
    fn the_clock_is_the_raw_monotonic_one() {
        for _ in 0..3 {
            let before = read(libc::CLOCK_MONOTONIC_RAW);
            let now = now_ns();
            let after = read(libc::CLOCK_MONOTONIC_RAW);
            assert!(before <= now && now <= after, "{before} {now} {after}");
        }
    }
}
