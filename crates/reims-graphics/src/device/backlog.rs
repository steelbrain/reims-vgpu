//! Which of ring 0's messages a command buffer waits for: those of its own context, and those of
//! every buffer's, published before its queue's thread took it (the device's module doc,
//! "Per-queue rings").
//!
//! **Whose a message is.** A command buffer names only objects of its own context (their ids
//! carry the context's bits, which the decoder checks) and spans in blocks its context holds, on
//! a queue of its context's, so the ring-0 messages it can depend on are its context's:
//! - an object's (every type from `QUEUE_CREATE` below `CONTEXT_CLOSE`): the context of the
//!   `obj_id` at offset 16. The kext posts a `QUEUE_CREATE` only for the process that owns the
//!   queue, and takes any other message of the range from a process only where that `obj_id` is
//!   in the process's range (`RgfxSubmitGate.hpp`'s `rgfxSubmitAdmitted`), so a type the host
//!   does not know there, which it refuses, is its sender's too;
//! - a context's (`CONTEXT_CLOSE` to `SURFACE_PAGES`): the `context` at offset 16, which the kext
//!   writes itself.
//!
//! An object of another process's reaches a buffer only through a block the kext shares with the
//! buffer's context: an IOSurface's, whose `BLOCK_SHARE` names that context and is posted before
//! the process learns of the block, after the surface's `SURFACE_PAGES`, and after every message
//! the process that made the surface posted before it handed the surface on. The control thread
//! carries ring 0 out in order, so a buffer that waits for its context's share waits for those
//! too. Shared events and textures do not cross processes.
//!
//! No buffer waits for a display command (`GET_DISPLAYS` to `DISPLAY_COLOR`) or a
//! `SURFACE_DOWNLOAD`: a buffer reads nothing they make. A present waits for the writers the
//! guest's record of its surface names (`RgfxSurfaceWriters`), whatever the host has taken. A
//! download waits for the buffers that used the surface's device cache before it was taken, so a
//! buffer posted after it that writes the surface, but taken before it, is among them, as one
//! posted before it is: the guest writes a surface a CPU lock holds only racing the lock
//! (`a_command_buffer_is_handed_over_while_a_download_published_ahead_of_it_is_carried_out`). Every
//! buffer waits for a message of any other type, and for one too short to hold its field: the
//! host refuses either, and the order is as it was.
//!
//! **The index.** The queues' threads read ring 0 ahead of the control thread, under its lock,
//! each message once ([`Backlog::read`]); the control thread indexes a message it takes that
//! no look read before ([`Backlog::took`]), so a message being carried out is in the index. The
//! kext alone writes ring 0, and leaves a published message as it is until the host takes it; a
//! message read otherwise only orders a buffer before a create it names, which the decoder
//! refuses. Where the walk meets a message the protocol refuses, every buffer waits for every
//! message the guest published, as the control thread stops there until a reset.

use std::collections::HashMap;

use crate::region::Region;
use crate::ring::{Consumer, Message, Walked};
use crate::wire::*;

/// How many contexts the index keeps before it forgets those whose messages are all carried out.
const KEEP: usize = 16;

/// How many contexts ring 0's index has room for from its make ([`Backlog::with_room`]): its
/// [`KEEP`] and as many again as a doorbell's look reads messages, so that look adds what it reads
/// allocating nothing ([`Backlog::room_for`]).
const ROOM: usize = 64;

/// Whose a ring-0 message is: which command buffers wait for it (the module doc).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Whose {
    /// The buffers of this context.
    Context(u32),
    /// Every buffer.
    Every,
    /// No buffer.
    NoOne,
}

/// Whose a message of `type` is, its 8 bytes at offset 16 `field`, where it holds them.
pub(super) fn whose(r#type: u32, field: Option<u64>) -> Whose {
    match (r#type, field) {
        (RGFX_CMD_GET_DISPLAYS..=RGFX_CMD_DISPLAY_COLOR | RGFX_CMD_SURFACE_DOWNLOAD, _) => {
            Whose::NoOne
        }
        (RGFX_CMD_QUEUE_CREATE..RGFX_CMD_CONTEXT_CLOSE, Some(obj_id)) => {
            Whose::Context((obj_id >> RGFX_CONTEXT_OBJ_ID_SHIFT) as u32)
        }
        (RGFX_CMD_CONTEXT_CLOSE..=RGFX_CMD_SURFACE_PAGES, Some(field)) => {
            Whose::Context(field as u32)
        }
        _ => Whose::Every,
    }
}

/// Ring 0's messages, by whose each is: the count through the last of each context's, and of
/// every buffer's, of those read so far. Counts are ring 0's, as `control_done` is.
#[derive(Debug, Default)]
pub(super) struct Backlog {
    /// The count through which the guest's messages are read.
    read: u64,
    /// Through the last message of each context's read.
    last: HashMap<u32, u64>,
    /// Through the last message read that every buffer waits for.
    every: u64,
}

impl Backlog {
    /// An index with room for [`ROOM`] contexts: ring 0's.
    pub(super) fn with_room() -> Backlog {
        Backlog {
            last: HashMap::with_capacity(ROOM),
            ..Backlog::default()
        }
    }

    /// Back to an index of nothing read, its room kept (the device's reset).
    pub(super) fn clear(&mut self) {
        self.read = 0;
        self.every = 0;
        self.last.clear();
    }

    /// Whether reading `most` more messages notes their contexts allocating nothing.
    pub(super) fn room_for(&self, most: usize) -> bool {
        self.last.capacity() - self.last.len() >= most
    }

    /// Reads what the guest published on ring 0 since the last look, at most `most` messages of
    /// it, `done` answering whether every message of a context through a count is carried out
    /// (by the control thread, or by the context's lane); false where `most` were read and more
    /// are left, which a look after it reads on.
    pub(super) fn read(
        &mut self,
        consumer: &Consumer,
        region: &Region,
        done: impl Fn(u32, u64) -> bool,
        most: usize,
    ) -> bool {
        let walked = consumer.walk_published(region, self.read, most, |through, r#type, field| {
            note(
                &mut self.last,
                &mut self.every,
                whose(r#type, field),
                through,
            );
        });
        if self.last.len() > KEEP {
            self.last
                .retain(|&context, through| !done(context, *through));
        }
        match walked {
            Walked::All(through) => self.read = through,
            Walked::Most(through) => {
                self.read = through;
                return false;
            }
            Walked::Broken(at) => {
                self.read = at;
                self.every = self.every.max(consumer.published_count(region));
            }
        }
        true
    }

    /// The count a command buffer of `context` waits for the control thread to carry out or hand
    /// to the context's lane: through the last message read of its context's or of every
    /// buffer's.
    pub(super) fn waits_for(&self, context: u32) -> u64 {
        self.own(context).max(self.every)
    }

    /// The count through the last message read of `context`'s, which a command buffer of the
    /// context waits for its lane to carry out; 0 for none.
    pub(super) fn own(&self, context: u32) -> u64 {
        self.last.get(&context).copied().unwrap_or(0)
    }

    /// The control thread took `message`, through count `through`: read here where no look
    /// read it.
    pub(super) fn took(&mut self, message: &Message, through: u64) {
        if through <= self.read {
            return;
        }
        let field = (message.bytes.get(16..24))
            .and_then(|bytes| bytes.try_into().ok())
            .map(u64::from_le_bytes);
        let whose = whose(message.header.r#type, field);
        note(&mut self.last, &mut self.every, whose, through);
        self.read = through;
    }
}

/// Notes a message `whose`, through count `through`.
fn note(last: &mut HashMap<u32, u64>, every: &mut u64, whose: Whose, through: u64) {
    match whose {
        Whose::Context(context) => {
            last.insert(context, through);
        }
        Whose::Every => *every = through,
        Whose::NoOne => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each message is its context's, by its object's id or its context field, but a display
    /// command and a download, which are no buffer's, and anything else, which is every buffer's.
    /// Every type the kext takes from a process by its object's id is its object's context's,
    /// those after `BUFFER_OVER_PAGES` and one the host does not know among them.
    #[test]
    fn a_message_is_its_context_s_by_its_id_or_its_field() {
        let obj = |context: u64| Some((context << RGFX_CONTEXT_OBJ_ID_SHIFT) | 7);
        assert_eq!(whose(RGFX_CMD_QUEUE_CREATE, obj(5)), Whose::Context(5));
        assert_eq!(whose(RGFX_CMD_BUFFER_CREATE, obj(6)), Whose::Context(6));
        assert_eq!(whose(RGFX_CMD_BUFFER_OVER_PAGES, obj(7)), Whose::Context(7));
        for r#type in [
            RGFX_CMD_ICB_CREATE,
            RGFX_CMD_ICB_WRITE,
            RGFX_CMD_BUFFER_MANAGED_AT_OFFSET,
            RGFX_CMD_ICB_SLOTS,
            RGFX_CMD_BUFFER_OVER_PAGES + 1,
            RGFX_CMD_CONTEXT_CLOSE - 1,
        ] {
            assert_eq!(
                whose(r#type, obj(8)),
                Whose::Context(8),
                "type {:#x}",
                r#type
            );
        }
        // The `context` field, a u32, with the next field's bytes above it.
        let field = Some((0xdead_u64 << 32) | 9);
        for r#type in [
            RGFX_CMD_CONTEXT_CLOSE,
            RGFX_CMD_BLOCK_GRANT,
            RGFX_CMD_BLOCK_SHARE,
            RGFX_CMD_SURFACE_PAGES,
        ] {
            assert_eq!(whose(r#type, field), Whose::Context(9));
        }
        for r#type in [
            RGFX_CMD_GET_DISPLAYS,
            RGFX_CMD_PRESENT_SURFACE,
            RGFX_CMD_DISPLAY_COLOR,
            RGFX_CMD_SURFACE_DOWNLOAD,
        ] {
            assert_eq!(whose(r#type, field), Whose::NoOne);
        }
        assert_eq!(whose(RGFX_CMD_BUFFER_CREATE, None), Whose::Every);
        assert_eq!(whose(RGFX_CMD_DISPLAY_COLOR + 1, obj(5)), Whose::Every);
        assert_eq!(whose(RGFX_CMD_SURFACE_DOWNLOAD + 1, obj(5)), Whose::Every);
        assert_eq!(whose(RGFX_CMD_COMMAND_BUFFER, obj(5)), Whose::Every);
    }
}
