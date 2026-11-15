//! How the control thread sleeps: on [`Wake`], the device's own, which a doorbell written through
//! the device's MMIO, the guest's `STATUS` and the device's end set, and on the host's doorbell
//! (its `ioeventfd`'s), which the guest's doorbell writes signal without the host's MMIO path. A
//! sleep takes, as it ends, what woke it on each, so the next sleep waits for a new wake.
//!
//! Each OS's own (`unix.rs`, `windows.rs`): on Linux and macOS a pipe and the host's file, polled;
//! on Windows events, waited for, and the host's doorbell an event (QEMU's `EventNotifier` there).
//! Both give the same names: [`Doorbell`] and [`RawDoorbell`], the host's doorbell owned and as
//! the host hands it, [`bell`], [`duplicate`] and [`raw`].

#[cfg(unix)]
mod unix;
#[cfg(windows)]
mod windows;

#[cfg(unix)]
pub use unix::RawDoorbell;
#[cfg(all(test, unix))]
pub(crate) use unix::not_open;
#[cfg(unix)]
pub(crate) use unix::{Doorbell, Wake, bell, duplicate, raw};
#[cfg(windows)]
pub use windows::RawDoorbell;
#[cfg(all(test, windows))]
pub(crate) use windows::not_open;
#[cfg(windows)]
pub(crate) use windows::{Doorbell, Wake, bell, duplicate, raw};

#[cfg(test)]
mod tests {
    use super::*;

    /// A wake ends a sleep, and a doorbell's signal ends one too
    #[test]
    fn a_wake_or_a_doorbell_ends_a_sleep() {
        let wake = Wake::new().unwrap();
        wake.wake();
        wake.sleep(None);
        let bell = Wake::new().unwrap();
        bell.wake();
        wake.sleep(Some(bell.raw()));
        // Both taken: a wake after them ends the next sleep, not what they left.
        wake.wake();
        wake.sleep(Some(bell.raw()));
    }

    /// A sleep for a time with nothing to wake it ends once the time has passed
    #[test]
    fn a_sleep_for_a_time_ends_with_no_wake() {
        let wake = Wake::new().unwrap();
        let bell = Wake::new().unwrap();
        wake.sleep_for(Some(bell.raw()), Some(std::time::Duration::from_millis(1)));
        wake.sleep_for(None, Some(std::time::Duration::ZERO));
    }

    /// The thread is parked only inside a sleep with no wake pending
    #[test]
    fn parked_only_inside_a_sleep_with_no_wake_pending() {
        let wake = Wake::new().unwrap();
        assert!(!wake.parked(None), "not in a sleep");
        std::thread::scope(|scope| {
            scope.spawn(|| wake.sleep(None));
            let parked = (0..2000).any(|_| {
                std::thread::sleep(std::time::Duration::from_millis(1));
                wake.parked(None)
            });
            assert!(parked, "the sleeper parked");
            wake.wake();
        });
        assert!(!wake.parked(None), "out of its sleep");
    }

    /// A copy of a doorbell is a doorbell of its own: it stays open once the original goes, and a
    /// signal of either ends a sleep on the other
    #[test]
    fn a_duplicated_doorbell_outlives_its_original() {
        let wake = Wake::new().unwrap();
        let bell = Wake::new().unwrap();
        let copy = duplicate(bell.raw()).unwrap();
        bell.wake();
        drop(bell);
        wake.sleep(Some(raw(&copy)));
        assert!(
            duplicate(not_open()).is_none(),
            "no copy of a doorbell that is not open"
        );
    }
}
