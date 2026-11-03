//! Which of ring 0's messages a command buffer waits for: those of its own context, and those of
//! every buffer's, published before its queue's thread took it (the device's module doc,
//! "Per-queue rings").
//!
//! **Whose a message is.** A command buffer names only objects of its own context (their ids
//! carry the context's bits, which the decoder checks) and spans in blocks its context holds, on
//! a queue of its context's, so the ring-0 messages it can depend on are its context's:
//! - an object's (`QUEUE_CREATE` to `BUFFER_OVER_PAGES`): the context of the `obj_id` at offset 16,
//!   which the kext posts only for the process that owns it;
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
//! No buffer waits for a display command (`GET_DISPLAYS` to `PRESENT_SURFACE`) or a
//! `SURFACE_DOWNLOAD`: a buffer reads nothing they make. A buffer posted after a present or a
//! download of a surface it writes, but taken before it, is among the writers that message waits
//! for, as one posted before it is; the guest writes a surface the display pipe shows, or one a
//! CPU lock holds, only racing them. Every buffer waits for a message of any other type, and for
//! one too short to hold its field: the host refuses either, and the order is as it was.
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
        (RGFX_CMD_GET_DISPLAYS..=RGFX_CMD_PRESENT_SURFACE | RGFX_CMD_SURFACE_DOWNLOAD, _) => {
            Whose::NoOne
        }
        (RGFX_CMD_QUEUE_CREATE..=RGFX_CMD_BUFFER_OVER_PAGES, Some(obj_id)) => {
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
    /// Reads what the guest published on ring 0 since the last look, at most `most` messages of
    /// it, the control thread having carried out what lies before count `done`; false where
    /// `most` were read and more are left, which a look after it reads on.
    pub(super) fn read(
        &mut self,
        consumer: &Consumer,
        region: &Region,
        done: u64,
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
            self.last.retain(|_, through| *through > done);
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

    /// The count a command buffer of `context` waits for the control thread to carry out: through
    /// the last message read of its context's or of every buffer's.
    pub(super) fn waits_for(&self, context: u32) -> u64 {
        let own = self.last.get(&context).copied().unwrap_or(0);
        own.max(self.every)
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
    #[test]
    fn a_message_is_its_context_s_by_its_id_or_its_field() {
        let obj = |context: u64| Some((context << RGFX_CONTEXT_OBJ_ID_SHIFT) | 7);
        assert_eq!(whose(RGFX_CMD_QUEUE_CREATE, obj(5)), Whose::Context(5));
        assert_eq!(whose(RGFX_CMD_BUFFER_CREATE, obj(6)), Whose::Context(6));
        assert_eq!(whose(RGFX_CMD_BUFFER_OVER_PAGES, obj(7)), Whose::Context(7));
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
            RGFX_CMD_SURFACE_DOWNLOAD,
        ] {
            assert_eq!(whose(r#type, field), Whose::NoOne);
        }
        assert_eq!(whose(RGFX_CMD_BUFFER_CREATE, None), Whose::Every);
        assert_eq!(whose(RGFX_CMD_BUFFER_OVER_PAGES + 1, obj(5)), Whose::Every);
        assert_eq!(whose(RGFX_CMD_COMMAND_BUFFER, obj(5)), Whose::Every);
    }
}
