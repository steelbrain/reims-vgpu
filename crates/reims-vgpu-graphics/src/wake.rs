//! How the control thread sleeps: on [`Wake`], a pipe of the device's own, which a doorbell
//! written through the device's MMIO, the guest's `STATUS` and the device's end write to, and on
//! the host's doorbell file (its `ioeventfd`'s), which the guest's doorbell writes signal without
//! the host's MMIO path. A read of each after it wakes takes what woke it.

use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
#[cfg(test)]
use std::sync::atomic::{AtomicBool, Ordering};

/// The device's own wake: a pipe, both ends non-blocking and closed on exec.
#[derive(Debug)]
pub(crate) struct Wake {
    /// The end the control thread polls.
    read: OwnedFd,
    /// The end a wake writes.
    write: OwnedFd,
    /// Set from just before [`Self::sleep`]'s poll until it returns ([`Self::parked`]).
    #[cfg(test)]
    sleeping: AtomicBool,
}

impl Wake {
    /// A new pipe, or `None` where the host refuses one.
    pub(crate) fn new() -> Option<Wake> {
        let mut fds = [0 as RawFd; 2];
        // SAFETY: `fds` holds the two descriptors `pipe` writes.
        if unsafe { libc::pipe(fds.as_mut_ptr()) } != 0 {
            return None;
        }
        // SAFETY: `pipe` made both, and nothing else owns them.
        let (read, write) = unsafe { (OwnedFd::from_raw_fd(fds[0]), OwnedFd::from_raw_fd(fds[1])) };
        for fd in [&read, &write] {
            set_flags(fd.as_raw_fd());
        }
        Some(Wake {
            read,
            write,
            #[cfg(test)]
            sleeping: AtomicBool::new(false),
        })
    }

    /// Whether the thread that sleeps on it is in its sleep with no wake pending, on the pipe or
    /// on `doorbell`: it took everything it was woken for, so what it writes of the rings stays
    /// as it is until the guest or the host wakes it. A wake that comes while it is awake stays
    /// in the pipe, and has it wake once more with nothing to take, which a test's settled state
    /// must not catch half done.
    #[cfg(test)]
    pub(crate) fn parked(&self, doorbell: Option<RawFd>) -> bool {
        if !self.sleeping.load(Ordering::SeqCst) {
            return false;
        }
        let mut fds = [
            libc::pollfd {
                fd: self.read.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            },
            libc::pollfd {
                fd: doorbell.unwrap_or(-1),
                events: libc::POLLIN,
                revents: 0,
            },
        ];
        // SAFETY: two pollfds, a negative descriptor ignored; a timeout of 0 never blocks.
        let n = unsafe { libc::poll(fds.as_mut_ptr(), 2, 0) };
        n == 0
    }

    /// The end the control thread polls, for a test that stands it in for a host's doorbell.
    #[cfg(test)]
    pub(crate) fn read_fd(&self) -> RawFd {
        self.read.as_raw_fd()
    }

    /// Wakes the control thread. A pipe already full has a wake pending, so a write it refuses
    /// loses nothing.
    pub(crate) fn wake(&self) {
        let byte = 1u8;
        // SAFETY: one byte from a local, to a descriptor the struct owns.
        let _ = unsafe { libc::write(self.write.as_raw_fd(), (&raw const byte).cast(), 1) };
    }

    /// Sleeps until a wake, or until `doorbell` (the host's file, where it gave one) is
    /// readable, then takes what either holds.
    pub(crate) fn sleep(&self, doorbell: Option<RawFd>) {
        let mut fds = [
            libc::pollfd {
                fd: self.read.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            },
            libc::pollfd {
                fd: doorbell.unwrap_or(-1),
                events: libc::POLLIN,
                revents: 0,
            },
        ];
        #[cfg(test)]
        self.sleeping.store(true, Ordering::SeqCst);
        // SAFETY: two pollfds, a negative descriptor ignored.
        let n = unsafe { libc::poll(fds.as_mut_ptr(), 2, -1) };
        // Before the drain: until it, a wake that ended the poll is still pending.
        #[cfg(test)]
        self.sleeping.store(false, Ordering::SeqCst);
        if n <= 0 {
            return;
        }
        for pollfd in fds {
            if pollfd.fd >= 0 && pollfd.revents & libc::POLLIN != 0 {
                drain(pollfd.fd);
            }
        }
    }
}

/// Takes what `fd` holds, at most 64 bytes: an eventfd's count (8 bytes), or a pipe's writes
/// (QEMU's event notifier is a pipe where the host has no eventfd). One read after a readable
/// poll never blocks; what it leaves keeps the descriptor readable for the next sleep.
fn drain(fd: RawFd) {
    let mut buf = [0u8; 64];
    // SAFETY: up to 64 bytes into a local.
    let _ = unsafe { libc::read(fd, buf.as_mut_ptr().cast(), buf.len()) };
}

/// Makes `fd` non-blocking and closed on exec.
fn set_flags(fd: RawFd) {
    // SAFETY: flag changes on a descriptor the caller owns.
    unsafe {
        let flags = libc::fcntl(fd, libc::F_GETFL);
        libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK);
        let fd_flags = libc::fcntl(fd, libc::F_GETFD);
        libc::fcntl(fd, libc::F_SETFD, fd_flags | libc::FD_CLOEXEC);
    }
}

/// A copy of the host's descriptor `fd`, closed on exec, which the device owns; `None` where
/// `fd` is not open.
pub(crate) fn duplicate(fd: RawFd) -> Option<OwnedFd> {
    // SAFETY: `fcntl` duplicates whatever `fd` names, or fails.
    let copy = unsafe { libc::fcntl(fd, libc::F_DUPFD_CLOEXEC, 0) };
    if copy < 0 {
        return None;
    }
    // SAFETY: the duplicate is new, and nothing else owns it.
    Some(unsafe { OwnedFd::from_raw_fd(copy) })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A wake ends a sleep, and a doorbell file's write ends one too
    #[test]
    fn a_wake_or_a_doorbell_ends_a_sleep() {
        let wake = Wake::new().unwrap();
        wake.wake();
        wake.sleep(None);
        let bell = Wake::new().unwrap();
        bell.wake();
        wake.sleep(Some(bell.read.as_raw_fd()));
        // Both drained: a wake after them ends the next sleep, not what they left.
        wake.wake();
        wake.sleep(Some(bell.read.as_raw_fd()));
    }
}
