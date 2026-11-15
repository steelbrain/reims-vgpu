//! The control thread's sleep on Windows: on [`Wake`], a manual-reset event of the device's own,
//! and on the host's doorbell, an event too (QEMU's `EventNotifier` is a manual-reset event there,
//! which its memory core sets for a doorbell write it matches), waited for together
//! (`WaitForMultipleObjects`). A reset of each that is set, as the sleep ends, takes what woke it,
//! as the pipe's and the eventfd's reads take theirs on Linux and macOS. A signal that comes after
//! the reset sets the event again for the next sleep; one that comes between the look at an event
//! and its reset is taken with it, as two writes an eventfd's one read takes, since the thread
//! looks at what it was woken for only after the sleep has returned.

use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle, RawHandle};
#[cfg(test)]
use std::sync::atomic::{AtomicU64, Ordering};

use windows_sys::Win32::Foundation::{
    DUPLICATE_SAME_ACCESS, DuplicateHandle, HANDLE, WAIT_OBJECT_0,
};
use windows_sys::Win32::System::Threading::{
    CreateEventW, GetCurrentProcess, INFINITE, ResetEvent, SetEvent, WaitForMultipleObjects,
    WaitForSingleObject,
};

/// The host's doorbell as the device keeps it: a handle it owns a copy of.
pub(crate) type Doorbell = OwnedHandle;
/// The host's doorbell as the host hands it ([`crate::Device::set_doorbell`]) and as the device
/// would hand a queue's bell to the host ([`crate::Host::route_doorbell`], which no Windows
/// device calls: [`bell`] gives none): an event's handle.
pub type RawDoorbell = RawHandle;

/// The device's own wake: a manual-reset event, not inherited by child processes.
#[derive(Debug)]
pub(crate) struct Wake {
    /// The event a wake sets and the control thread waits for.
    event: OwnedHandle,
    /// Moved on once just before [`Self::sleep`]'s wait and once as it returns, so odd while the
    /// thread is in it, and changed by each sleep ([`Self::parked`]).
    #[cfg(test)]
    sleeps: AtomicU64,
}

impl Wake {
    /// A new event, or `None` where the host refuses one.
    pub(crate) fn new() -> Option<Wake> {
        // SAFETY: no security attributes (not inherited) and no name; manual reset, not set.
        let event = unsafe { CreateEventW(std::ptr::null(), 1, 0, std::ptr::null()) };
        if event.is_null() {
            return None;
        }
        Some(Wake {
            // SAFETY: `CreateEventW` made it, and nothing else owns it.
            event: unsafe { OwnedHandle::from_raw_handle(event) },
            #[cfg(test)]
            sleeps: AtomicU64::new(0),
        })
    }

    /// Whether the thread that sleeps on it is in its sleep with no wake pending, on its event or
    /// on `doorbell`: as the Unix side's ([`super::unix`]), looked at without taking either,
    /// since a manual-reset event stays set through a wait of no time.
    #[cfg(test)]
    pub(crate) fn parked(&self, doorbell: Option<RawDoorbell>) -> bool {
        let sleep = self.sleeps.load(Ordering::SeqCst);
        if sleep % 2 == 0 {
            return false;
        }
        let (handles, count) = self.handles(doorbell);
        // SAFETY: `count` live handles; a timeout of 0 never blocks.
        let waited = unsafe { WaitForMultipleObjects(count, handles.as_ptr(), 0, 0) };
        waited == windows_sys::Win32::Foundation::WAIT_TIMEOUT
            && self.sleeps.load(Ordering::SeqCst) == sleep
    }

    /// The event, for a test that stands it in for a host's doorbell.
    #[cfg(test)]
    pub(crate) fn raw(&self) -> RawDoorbell {
        self.event.as_raw_handle()
    }

    /// Wakes the control thread. An event already set has a wake pending, so a second set loses
    /// nothing.
    pub(crate) fn wake(&self) {
        // SAFETY: an event the struct owns.
        let _ = unsafe { SetEvent(self.event.as_raw_handle()) };
    }

    /// Sleeps until a wake, or until `doorbell` (the host's event, where it gave one) is set,
    /// then takes what either holds.
    pub(crate) fn sleep(&self, doorbell: Option<RawDoorbell>) {
        self.sleep_for(doorbell, None);
    }

    /// [`Self::sleep`], for `timeout` at most where one is given (rounded up to a millisecond).
    pub(crate) fn sleep_for(
        &self,
        doorbell: Option<RawDoorbell>,
        timeout: Option<std::time::Duration>,
    ) {
        let timeout_ms = timeout.map_or(INFINITE, |t| {
            u32::try_from(t.as_nanos().div_ceil(1_000_000))
                .map_or(INFINITE - 1, |ms| ms.min(INFINITE - 1))
        });
        let (handles, count) = self.handles(doorbell);
        #[cfg(test)]
        self.sleeps.fetch_add(1, Ordering::SeqCst);
        // SAFETY: `count` live handles, waited for until any is set or the time passes.
        let waited = unsafe { WaitForMultipleObjects(count, handles.as_ptr(), 0, timeout_ms) };
        // Before the resets: until them, a wake that ended the wait is still pending.
        #[cfg(test)]
        self.sleeps.fetch_add(1, Ordering::SeqCst);
        if waited >= WAIT_OBJECT_0 + count {
            // Timed out, or failed.
            return;
        }
        for &handle in &handles[..count as usize] {
            take(handle);
        }
    }

    /// The handles a sleep waits for, its own event first, and how many there are.
    fn handles(&self, doorbell: Option<RawDoorbell>) -> ([HANDLE; 2], u32) {
        let own = self.event.as_raw_handle();
        match doorbell {
            Some(doorbell) => ([own, doorbell], 2),
            None => ([own, std::ptr::null_mut()], 1),
        }
    }
}

/// A command queue's bell: none on Windows, where the host has no kernel `ioeventfd` to route a
/// queue's doorbell into one, so the queues' doorbells run the device's write as on macOS.
pub(crate) fn bell() -> Option<Doorbell> {
    None
}

/// Resets `event` where it is set: takes the signal that woke the sleep.
fn take(event: HANDLE) {
    // SAFETY: a live event; a timeout of 0 never blocks.
    if unsafe { WaitForSingleObject(event, 0) } == WAIT_OBJECT_0 {
        // SAFETY: as above.
        let _ = unsafe { ResetEvent(event) };
    }
}

/// A doorbell that is not open, for the tests: a null handle, which no duplicate is made of
/// (unlike -1, the process's own pseudo-handle).
#[cfg(test)]
pub(crate) fn not_open() -> RawDoorbell {
    std::ptr::null_mut()
}

/// The handle `doorbell` holds, to hand to a sleep.
pub(crate) fn raw(doorbell: &Doorbell) -> RawDoorbell {
    doorbell.as_raw_handle()
}

/// A copy of the host's handle `handle`, not inherited by child processes, which the device owns;
/// `None` where `handle` is not open.
pub(crate) fn duplicate(handle: RawDoorbell) -> Option<Doorbell> {
    let mut copy: HANDLE = std::ptr::null_mut();
    // SAFETY: duplicates whatever `handle` names within this process, or fails.
    let copied = unsafe {
        DuplicateHandle(
            GetCurrentProcess(),
            handle,
            GetCurrentProcess(),
            &mut copy,
            0,
            0,
            DUPLICATE_SAME_ACCESS,
        )
    };
    if copied == 0 || copy.is_null() {
        return None;
    }
    // SAFETY: the duplicate is new, and nothing else owns it.
    Some(unsafe { OwnedHandle::from_raw_handle(copy) })
}
