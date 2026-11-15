//! The host's monotonic clock, in nanoseconds: what the guest reads through `HOST_TIME_LO` and
//! `_HI`, and what each `VBLANK` and the device's other times are stamped on, each OS's own
//! (`unix.rs`, `windows.rs`). Each runs at the rate the guest's clock does, a counter's that
//! nothing disciplines, so the kext's offset of the two clocks holds between its readings.

#[cfg(unix)]
mod unix;
#[cfg(windows)]
mod windows;

#[cfg(unix)]
pub(crate) use unix::now_ns;
#[cfg(windows)]
pub(crate) use windows::now_ns;

#[cfg(test)]
mod tests {
    use super::*;

    /// The clock counts in nanoseconds and moves on across a sleep, never back
    #[test]
    fn the_clock_moves_on_across_a_sleep() {
        let before = now_ns();
        std::thread::sleep(std::time::Duration::from_millis(5));
        let after = now_ns();
        assert!(after >= before + 4_000_000, "{before} then {after}");
        assert!(after < before + 5_000_000_000, "{before} then {after}");
    }
}
