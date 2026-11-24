//! The control thread's sleep on Linux and macOS: on [`Wake`], a pipe of the device's own, and on
//! the host's doorbell file, polled together. A read of each after it wakes takes what woke it.

use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};

/// The host's doorbell as the device keeps it: a file it owns a copy of.
pub(crate) type Doorbell = OwnedFd;
/// The host's doorbell as the host hands it ([`crate::Device::set_doorbell`]) and as the device
/// hands a queue's bell to the host ([`crate::Host::route_doorbell`]): a file descriptor.
pub type RawDoorbell = RawFd;
#[cfg(test)]
use std::sync::atomic::{AtomicU64, Ordering};

/// The device's own wake: a pipe, both ends non-blocking and closed on exec.
#[derive(Debug)]
pub(crate) struct Wake {
    /// The end the control thread polls.
    read: OwnedFd,
    /// The end a wake writes.
    write: OwnedFd,
    /// Moved on once just before [`Self::sleep`]'s poll and once as it returns, so odd while the
    /// thread is in it, and changed by each sleep ([`Self::parked`]).
    #[cfg(test)]
    sleeps: AtomicU64,
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
            sleeps: AtomicU64::new(0),
        })
    }

    /// Whether the thread that sleeps on it is in its sleep with no wake pending, on the pipe or
    /// on `doorbell`: it took everything it was woken for, so what it writes of the rings stays
    /// as it is until the guest or the host wakes it. A wake that comes while it is awake stays
    /// in the pipe, and has it wake once more with nothing to take, which a test's settled state
    /// must not catch half done. The thread is in one sleep from before the look at the pipe to
    /// after it: a poll that returned just as the look began may have its wake drained before
    /// the look, which then finds nothing pending of a thread that is awake.
    #[cfg(test)]
    pub(crate) fn parked(&self, doorbell: Option<RawFd>) -> bool {
        let sleep = self.sleeps.load(Ordering::SeqCst);
        if sleep % 2 == 0 {
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
        n == 0 && self.sleeps.load(Ordering::SeqCst) == sleep
    }

    /// The end the control thread polls, for a test that stands it in for a host's doorbell.
    #[cfg(test)]
    pub(crate) fn raw(&self) -> RawDoorbell {
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
        self.sleep_for(doorbell, None);
    }

    /// [`Self::sleep`], for `timeout` at most where one is given (rounded up to a millisecond):
    /// what ended it.
    pub(crate) fn sleep_for(
        &self,
        doorbell: Option<RawFd>,
        timeout: Option<std::time::Duration>,
    ) -> super::Woke {
        let timeout_ms = timeout.map_or(-1, |t| {
            libc::c_int::try_from(t.as_nanos().div_ceil(1_000_000)).unwrap_or(libc::c_int::MAX)
        });
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
        self.sleeps.fetch_add(1, Ordering::SeqCst);
        // SAFETY: two pollfds, a negative descriptor ignored.
        let n = unsafe { libc::poll(fds.as_mut_ptr(), 2, timeout_ms) };
        // Before the drain: until it, a wake that ended the poll is still pending.
        #[cfg(test)]
        self.sleeps.fetch_add(1, Ordering::SeqCst);
        let mut woke = super::Woke::default();
        if n <= 0 {
            return woke;
        }
        let [own, bell] = fds;
        if own.revents & libc::POLLIN != 0 {
            drain(own.fd);
            woke.wake = true;
        }
        if bell.fd >= 0 && bell.revents & libc::POLLIN != 0 {
            woke.bells = drain(bell.fd);
        }
        woke
    }
}

/// A command queue's bell: an eventfd, non-blocking and closed on exec, that its service thread
/// sleeps on beside its [`Wake`], and that the host makes readable for the queue's plain doorbell
/// where it routes it ([`crate::device::Host::route_doorbell`]: QEMU's KVM `ioeventfd`). `None`
/// where the host refuses one, and on a host with no eventfd, whose queues' doorbells run the
/// device's write as before.
pub(crate) fn bell() -> Option<OwnedFd> {
    #[cfg(target_os = "linux")]
    {
        // SAFETY: no pointer; the flags make it non-blocking and closed on exec.
        let fd = unsafe { libc::eventfd(0, libc::EFD_NONBLOCK | libc::EFD_CLOEXEC) };
        if fd < 0 {
            return None;
        }
        // SAFETY: `eventfd` made it, and nothing else owns it.
        Some(unsafe { OwnedFd::from_raw_fd(fd) })
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Takes what `fd` holds, at most 64 bytes: an eventfd's count (8 bytes), or a pipe's writes
/// (QEMU's event notifier is a pipe where the host has no eventfd). One read after a readable
/// poll never blocks; what it leaves keeps the descriptor readable for the next sleep. The
/// signals it took: an eventfd's count, else 1 where it read any byte.
fn drain(fd: RawFd) -> u64 {
    let mut buf = [0u8; 64];
    // SAFETY: up to 64 bytes into a local.
    let n = unsafe { libc::read(fd, buf.as_mut_ptr().cast(), buf.len()) };
    match n {
        8 => u64::from_ne_bytes([
            buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
        ]),
        n if n > 0 => 1,
        _ => 0,
    }
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

/// A doorbell that is not open, for the tests: -1.
#[cfg(test)]
pub(crate) fn not_open() -> RawDoorbell {
    -1
}

/// The descriptor `doorbell` holds, to hand to a sleep or to the host.
pub(crate) fn raw(doorbell: &Doorbell) -> RawDoorbell {
    doorbell.as_raw_fd()
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
