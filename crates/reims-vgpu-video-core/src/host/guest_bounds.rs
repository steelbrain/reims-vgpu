//! Bounds of our own on what a guest can make the host session hold or
//! send: how many sessions it keeps, how much of its messages waits for a
//! session's thread or is gathered from fragments, and how large a message
//! to the guest may be.
//!
//! Apple's host has none of these but the last, where it ends the VM
//! instead. A guest that VideoToolbox drives never comes near them: it
//! keeps a few frames in flight, in messages of 2 MiB at most.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// The largest message the guest takes: its kext posts queue-1 buffers of
/// 0x4000 bytes, and the device ends QEMU on a message longer than the
/// buffer, as Apple's host ends the VM. So none is sent; the sender sees
/// -12902.
pub const GUEST_BUFFER_SIZE: usize = 0x4000;

/// The most mapping IDs one `rels` names: 8 bytes of header and 8 a
/// surface, in [`GUEST_BUFFER_SIZE`]. More are sent in several.
pub const MAX_RELEASES_PER_MESSAGE: usize = (GUEST_BUFFER_SIZE - 8) / 8;

/// The most decoder sessions a host session keeps, decoding or refused,
/// and the most encoder sessions. Each decoding one has a thread, and its
/// backend's surfaces on the GPU. A `+dec` or `+enc` past it is answered
/// with -12913 or -12915, VideoToolbox's "not available now"
/// (`VTErrors.h:42`, `:45` in the macOS SDK), and keeps no session.
pub const MAX_SESSIONS: usize = 16;

/// The bytes of guest messages a host session holds for its sessions'
/// threads: queued, and gathered from a `dfrm`'s fragments. This is the
/// whole host session's backstop, shared by every session, so that the
/// device's memory is bounded however many sessions run.
pub const MAX_HELD_BYTES: usize = 512 << 20;

/// The guest messages it holds so, counted alike.
pub const MAX_HELD_MESSAGES: usize = 4096;

/// The bytes one session alone may hold, below [`MAX_HELD_BYTES`]. A
/// session whose thread is stalled queues its messages faster than it
/// handles them (the guest's wallpaper sent 28,881 `dfrm`s in 50 s);
/// bounding each session on its own keeps one such session from exhausting
/// the shared backstop and starving every other session's messages. It is
/// half the whole, so one stalled session leaves the rest half the
/// backstop, and above [`MAX_FRAME_MESSAGE_BYTES`], so one maximal frame
/// still gathers with room to spare.
pub const MAX_HELD_BYTES_PER_SESSION: usize = MAX_HELD_BYTES / 2;

/// The guest messages one session alone may hold, below
/// [`MAX_HELD_MESSAGES`].
pub const MAX_HELD_MESSAGES_PER_SESSION: usize = MAX_HELD_MESSAGES / 4;

/// The most one `dfrm` gathers from its fragments. The largest picture a
/// driver here decodes, 8192x4352 at 12 bits in 4:2:0, is 80 MB of
/// samples, and its compressed form is smaller.
pub const MAX_FRAME_MESSAGE_BYTES: usize = 128 << 20;

/// The most one reply from the guest gathers from its fragments. A reply
/// holds a status, a UUID or a format description.
pub const MAX_REPLY_BYTES: usize = 16 << 20;

/// What a host session holds of the guest's messages, against a byte and a
/// message count. A session's counter has the whole host session's as its
/// parent, so a hold must fit under both the session's own bound and the
/// shared backstop ([`per_session`](HeldMessages::per_session)).
#[derive(Debug)]
pub(crate) struct HeldMessages {
    bytes: AtomicUsize,
    messages: AtomicUsize,
    max_bytes: usize,
    max_messages: usize,
    /// The shared backstop a per-session counter also draws on, `None` for
    /// the backstop itself.
    parent: Option<Arc<HeldMessages>>,
}

impl HeldMessages {
    /// The whole host session's backstop: [`MAX_HELD_BYTES`] and
    /// [`MAX_HELD_MESSAGES`], shared by every session.
    pub(crate) fn backstop() -> Arc<Self> {
        Arc::new(Self {
            bytes: AtomicUsize::new(0),
            messages: AtomicUsize::new(0),
            max_bytes: MAX_HELD_BYTES,
            max_messages: MAX_HELD_MESSAGES,
            parent: None,
        })
    }

    /// One session's counter: [`MAX_HELD_BYTES_PER_SESSION`] and
    /// [`MAX_HELD_MESSAGES_PER_SESSION`] of its own, over `backstop`.
    pub(crate) fn per_session(backstop: &Arc<Self>) -> Arc<Self> {
        Arc::new(Self {
            bytes: AtomicUsize::new(0),
            messages: AtomicUsize::new(0),
            max_bytes: MAX_HELD_BYTES_PER_SESSION,
            max_messages: MAX_HELD_MESSAGES_PER_SESSION,
            parent: Some(backstop.clone()),
        })
    }

    /// Takes one message of `bytes` from this counter's own totals, against
    /// its own bounds; `true` when it fit. The parent is not touched.
    fn take(&self, bytes: usize) -> bool {
        let charge = |counter: &AtomicUsize, amount: usize, limit: usize| {
            counter
                .try_update(Ordering::AcqRel, Ordering::Acquire, |held| {
                    held.checked_add(amount).filter(|&total| total <= limit)
                })
                .is_ok()
        };
        if !charge(&self.messages, 1, self.max_messages) {
            return false;
        }
        if !charge(&self.bytes, bytes, self.max_bytes) {
            self.messages.fetch_sub(1, Ordering::AcqRel);
            return false;
        }
        true
    }

    /// Gives one message of `bytes` back to this counter's own totals.
    fn give_back(&self, bytes: usize) {
        self.bytes.fetch_sub(bytes, Ordering::AcqRel);
        self.messages.fetch_sub(1, Ordering::AcqRel);
    }

    /// Holds `bytes` more of one message, or `None` when that would pass
    /// this counter's bound or the shared backstop's. What is held is let
    /// go when the returned value is dropped.
    pub(crate) fn hold(self: &Arc<Self>, bytes: usize) -> Option<Held> {
        if !self.take(bytes) {
            return None;
        }
        if let Some(parent) = &self.parent
            && !parent.take(bytes)
        {
            self.give_back(bytes);
            return None;
        }
        Some(Held {
            messages: self.clone(),
            bytes,
        })
    }

    /// The bytes held now.
    #[cfg(test)]
    pub(crate) fn bytes(&self) -> usize {
        self.bytes.load(Ordering::Acquire)
    }
}

/// One message's share of what a host session holds.
#[derive(Debug)]
pub(crate) struct Held {
    messages: Arc<HeldMessages>,
    bytes: usize,
}

impl Drop for Held {
    fn drop(&mut self) {
        self.messages.give_back(self.bytes);
        if let Some(parent) = &self.messages.parent {
            parent.give_back(self.bytes);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn holds_are_let_go_when_dropped() {
        let held = HeldMessages::backstop();
        let first = held.hold(MAX_HELD_BYTES - 1);
        assert!(first.is_some());
        assert!(held.hold(2).is_none());
        let second = held.hold(1);
        assert!(second.is_some());
        drop(first);
        drop(second);
        assert_eq!(held.bytes(), 0);
        let many: Vec<Held> = (0..MAX_HELD_MESSAGES)
            .filter_map(|_| held.hold(0))
            .collect();
        assert_eq!(many.len(), MAX_HELD_MESSAGES);
        assert!(held.hold(0).is_none());
    }

    /// A session that holds up to its own bound leaves the backstop with
    /// room for another session, which one stalled session would
    /// otherwise starve.
    #[test]
    fn one_session_cannot_exhaust_the_backstop() {
        let backstop = HeldMessages::backstop();
        let stalled = HeldMessages::per_session(&backstop);
        let healthy = HeldMessages::per_session(&backstop);
        // The stalled session fills its own message bound and stops there.
        let piled: Vec<Held> = (0..MAX_HELD_MESSAGES_PER_SESSION + 10)
            .filter_map(|_| stalled.hold(0))
            .collect();
        assert_eq!(piled.len(), MAX_HELD_MESSAGES_PER_SESSION);
        assert!(stalled.hold(0).is_none());
        // The healthy session still holds its messages.
        assert!(healthy.hold(1 << 20).is_some());
        // And the backstop carries only the stalled session's share.
        assert_eq!(backstop.messages.load(Ordering::Acquire), piled.len());
        drop(piled);
        assert_eq!(backstop.bytes(), 0);
        assert_eq!(backstop.messages.load(Ordering::Acquire), 0);
    }

    /// A per-session byte bound rolls the backstop back when it is the one
    /// that does not fit, so neither counter is left charged.
    #[test]
    fn a_refused_hold_charges_neither_counter() {
        let backstop = HeldMessages::backstop();
        let session = HeldMessages::per_session(&backstop);
        assert!(session.hold(MAX_HELD_BYTES_PER_SESSION + 1).is_none());
        assert_eq!(session.bytes(), 0);
        assert_eq!(backstop.bytes(), 0);
        assert_eq!(backstop.messages.load(Ordering::Acquire), 0);
    }

    /// One maximal frame still gathers within a session's byte bound.
    #[test]
    fn a_maximal_frame_fits_one_session() {
        const { assert!(MAX_FRAME_MESSAGE_BYTES <= MAX_HELD_BYTES_PER_SESSION) };
        let backstop = HeldMessages::backstop();
        let session = HeldMessages::per_session(&backstop);
        assert!(session.hold(MAX_FRAME_MESSAGE_BYTES).is_some());
    }
}
