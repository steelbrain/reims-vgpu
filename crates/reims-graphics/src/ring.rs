//! The two ends of a ring the host holds (`wire`'s module doc has the protocol): [`Consumer`], the
//! host's end of a submission ring the guest writes, and [`Producer`], its end of a completion
//! ring the guest reads.
//!
//! The host keeps its own count of each ring, and writes it to the header for the guest; it never
//! reads its own count back from guest memory. The guest's count is read with an acquire load,
//! and checked against the host's before anything is read of the ring: a count that runs ahead by
//! more than the ring holds, or behind the host's, is a broken ring ([`Broken`]).
//!
//! **The completion rings' interrupt.** The host publishes a record, fences, then reads and
//! clears `armed`. A guest consumer already awake (from the buffer's `CB_SCHEDULED`, say) can take
//! that record between the publish and the clear and arm again, so the clear takes an arm made
//! after the take: a vector for a record already taken, and the guest asleep with `armed` 0, for
//! which the host raises nothing more. So a post whose clear read 1 reads `consumer` too, and
//! where the guest has taken every record published, sets `armed` back and raises nothing
//! ([`rearm_or_raise`]). The guest stores `consumer` before it arms, so a clear that read its arm
//! sees its take; only the host writes 0, so the store of 1 undoes nothing of the guest's.

use std::sync::atomic::{Ordering, fence};

use zerocopy::{FromBytes, IntoBytes};

use crate::region::Region;
use crate::wire::{RGFX_CB_OK, RGFX_EVT_CB_COMPLETE, RGFX_MSG_PAD, RgfxCbComplete, RgfxMsgHeader};

/// Where a ring is in the region: its header, and its byte ring of `data_size` bytes, a power of
/// two, right after it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RingAt {
    /// The header's offset in the region.
    pub(crate) header: u64,
    /// The byte ring's size.
    pub(crate) data_size: u64,
}

/// The header fields' offsets, from `wire::RgfxRingHeader`.
const PRODUCER: u64 = 0x00;
const COMPLETED_SEQNO: u64 = 0x08;
const CONSUMER: u64 = 0x40;
const CONSUMER_IDLE: u64 = 0x80;
const ARMED: u64 = 0xc0;
/// Where the byte ring starts, after the header.
const DATA: u64 = crate::wire::RGFX_RING_HEADER_SIZE as u64;
/// A message header's bytes: `type` and `size`, then `seqno`.
const HEADER_SIZE: u64 = size_of::<RgfxMsgHeader>() as u64;

/// What makes a ring broken: the guest wrote counts or a message the protocol does not allow.
/// The host takes nothing more from the ring until the device is reset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Broken {
    /// The guest's count is behind the host's, or ahead of it by more than the ring holds.
    Count {
        /// The guest's count.
        guest: u64,
        /// The host's.
        host: u64,
    },
    /// A message's `size` is not a multiple of 8, is below its header (16 bytes, 8 for a pad),
    /// reaches past the ring's end, or past what the producer published.
    Size {
        /// Where the message starts, as a count.
        at: u64,
        /// Its `size`.
        size: u32,
    },
}

impl std::fmt::Display for Broken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Broken::Count { guest, host } => {
                write!(
                    f,
                    "the guest's count {guest} is out of reach of the host's {host}"
                )
            }
            Broken::Size { at, size } => write!(f, "the message at {at} has a size of {size}"),
        }
    }
}

impl RingAt {
    /// Writes the header of a ring nothing has passed through: both counts 0, `armed` 0, and
    /// `consumer_idle` as given.
    pub(crate) fn initialize(&self, region: &Region, consumer_idle: bool) {
        region
            .atomic_u64(self.header + PRODUCER)
            .store(0, Ordering::Release);
        region
            .atomic_u64(self.header + COMPLETED_SEQNO)
            .store(0, Ordering::Release);
        region
            .atomic_u64(self.header + CONSUMER)
            .store(0, Ordering::Release);
        region
            .atomic_u32(self.header + CONSUMER_IDLE)
            .store(u32::from(consumer_idle), Ordering::Release);
        region
            .atomic_u32(self.header + ARMED)
            .store(0, Ordering::Release);
    }

    /// The byte ring's offset in the region for count `count`.
    fn data_at(&self, count: u64) -> u64 {
        self.header + DATA + (count & (self.data_size - 1))
    }

    /// The bytes from count `count` to the byte ring's end.
    fn left_to_end(&self, count: u64) -> u64 {
        self.data_size - (count & (self.data_size - 1))
    }
}

/// A message taken from a submission ring: its header and its whole bytes, header included.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Message {
    /// Its header.
    pub(crate) header: RgfxMsgHeader,
    /// Its `size` bytes, copied out of the ring.
    pub(crate) bytes: Vec<u8>,
}

/// How far [`Consumer::walk_published`] went: the count through the messages it walked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Walked {
    /// Through every message the guest has published.
    All(u64),
    /// Through the most it was to walk, with more published.
    Most(u64),
    /// Up to a count or a message the protocol refuses, where [`Consumer::peek`] would find the
    /// ring broken.
    Broken(u64),
}

/// The host's end of a submission ring: it reads what the guest produces.
#[derive(Debug)]
pub(crate) struct Consumer {
    /// The ring.
    at: RingAt,
    /// The host's count: what it has taken.
    count: u64,
}

impl Consumer {
    /// The consumer of a ring at `at`, nothing taken yet.
    pub(crate) fn new(at: RingAt) -> Consumer {
        Consumer { at, count: 0 }
    }

    /// Back to a ring nothing has passed through, as [`RingAt::initialize`] writes it, with
    /// `consumer_idle` 1: the guest rings the doorbell for its first message.
    pub(crate) fn reset(&mut self, region: &Region) {
        self.count = 0;
        self.at.initialize(region, true);
    }

    /// The guest's count, checked against the host's: how far it runs ahead.
    fn published(&self, region: &Region) -> Result<u64, Broken> {
        let guest = region
            .atomic_u64(self.at.header + PRODUCER)
            .load(Ordering::Acquire);
        let ahead = guest.wrapping_sub(self.count);
        if ahead > self.at.data_size {
            return Err(Broken::Count {
                guest,
                host: self.count,
            });
        }
        Ok(ahead)
    }

    /// Whether the guest has published bytes the host has not taken.
    pub(crate) fn has_pending(&self, region: &Region) -> Result<bool, Broken> {
        Ok(self.published(region)? > 0)
    }

    /// The host's count: how many bytes it has taken. A reader on another thread waits for this to
    /// reach a snapshot of [`Self::published_count`] to order work that depends on messages the
    /// guest published before it; it never consumes the ring itself, so this ring keeps one consumer.
    pub(crate) fn consumed(&self) -> u64 {
        self.count
    }

    /// The guest's published count (the ring's producer), read with an acquire load.
    pub(crate) fn published_count(&self, region: &Region) -> u64 {
        region
            .atomic_u64(self.at.header + PRODUCER)
            .load(Ordering::Acquire)
    }

    /// The next message, with the pads before it skipped, and the host's count moved past it and
    /// written to the header; `None` where the guest has published no more.
    pub(crate) fn next(&mut self, region: &Region) -> Result<Option<Message>, Broken> {
        let message = self.peek(region)?;
        if let Some(message) = &message {
            self.take(region, message);
        }
        Ok(message)
    }

    /// [`Self::next`], with the pads before the message skipped but the message itself left on
    /// the ring, for [`Self::take`].
    pub(crate) fn peek(&mut self, region: &Region) -> Result<Option<Message>, Broken> {
        loop {
            let ahead = self.published(region)?;
            if ahead == 0 {
                return Ok(None);
            }
            let left = self.at.left_to_end(self.count);
            let broken = |size| Broken::Size {
                at: self.count,
                size,
            };
            // A published message has at least its `type` and `size`, which fit before the end:
            // the bytes left there are a multiple of 8.
            let Some(start) = region.read(self.at.data_at(self.count), 8) else {
                return Err(broken(0));
            };
            let r#type = u32::from_le_bytes([start[0], start[1], start[2], start[3]]);
            let size = u32::from_le_bytes([start[4], start[5], start[6], start[7]]);
            let least = if r#type == RGFX_MSG_PAD {
                8
            } else {
                HEADER_SIZE
            };
            let size64 = u64::from(size);
            if size % 8 != 0 || size64 < least || size64 > left || size64 > ahead {
                return Err(broken(size));
            }
            if r#type == RGFX_MSG_PAD {
                self.consume(region, size64);
                continue;
            }
            let Some(bytes) = region.read(self.at.data_at(self.count), size as usize) else {
                return Err(broken(size));
            };
            let Ok((header, _)) = RgfxMsgHeader::read_from_prefix(&bytes) else {
                return Err(broken(size));
            };
            return Ok(Some(Message { header, bytes }));
        }
    }

    /// The highest `seqno` of the messages the guest has published and the host not yet taken,
    /// read where they lie, pads skipped, nothing taken; `None` where there are none. A message
    /// the protocol refuses ends the walk there, as [`Self::peek`] would find the ring broken.
    pub(crate) fn highest_published_seqno(&self, region: &Region) -> Option<u64> {
        let ahead = self.published(region).ok()?;
        let mut highest = None;
        let mut walked = 0u64;
        while walked < ahead {
            let at = self.count.wrapping_add(walked);
            let start = region.read(self.at.data_at(at), 8)?;
            let r#type = u32::from_le_bytes([start[0], start[1], start[2], start[3]]);
            let size = u64::from(u32::from_le_bytes([start[4], start[5], start[6], start[7]]));
            let least = if r#type == RGFX_MSG_PAD {
                8
            } else {
                HEADER_SIZE
            };
            if size % 8 != 0 || size < least || size > self.at.left_to_end(at) {
                break;
            }
            if size > ahead - walked {
                break;
            }
            if r#type != RGFX_MSG_PAD {
                let seqno = region.read(self.at.data_at(at) + 8, 8)?;
                let seqno = u64::from_le_bytes(seqno.try_into().ok()?);
                highest = highest.max(Some(seqno));
            }
            walked += size;
        }
        highest
    }

    /// Walks the messages the guest has published and the host has not taken, in ring order, read
    /// where they lie, pads skipped, nothing taken: from count `from`, where that is past the
    /// host's own count, and at most `most` of them. `visit` is given the count through each, its
    /// `type`, and its 8 bytes at offset 16, its first field past the header, where it holds them.
    pub(crate) fn walk_published(
        &self,
        region: &Region,
        from: u64,
        most: usize,
        mut visit: impl FnMut(u64, u32, Option<u64>),
    ) -> Walked {
        let mut at = from.max(self.count);
        let Ok(ahead) = self.published(region) else {
            return Walked::Broken(at);
        };
        let end = self.count.wrapping_add(ahead);
        let mut walked = 0;
        while at < end {
            if walked == most {
                return Walked::Most(at);
            }
            let Some(start) = region.read(self.at.data_at(at), 8) else {
                return Walked::Broken(at);
            };
            let r#type = u32::from_le_bytes([start[0], start[1], start[2], start[3]]);
            let size = u64::from(u32::from_le_bytes([start[4], start[5], start[6], start[7]]));
            let least = if r#type == RGFX_MSG_PAD {
                8
            } else {
                HEADER_SIZE
            };
            if size % 8 != 0 || size < least || size > self.at.left_to_end(at) || size > end - at {
                return Walked::Broken(at);
            }
            if r#type != RGFX_MSG_PAD {
                let field = (size >= HEADER_SIZE + 8)
                    .then(|| region.read(self.at.data_at(at) + HEADER_SIZE, 8))
                    .flatten()
                    .and_then(|bytes| bytes.try_into().ok())
                    .map(u64::from_le_bytes);
                walked += 1;
                visit(at + size, r#type, field);
            }
            at += size;
        }
        Walked::All(at)
    }

    /// Takes `message`, which [`Self::peek`] answered last: the host's count moves past it and is
    /// written to the header.
    pub(crate) fn take(&mut self, region: &Region, message: &Message) {
        self.consume(region, u64::from(message.header.size));
    }

    /// Moves the host's count on by `size` and publishes it: the guest may write those bytes
    /// again.
    fn consume(&mut self, region: &Region, size: u64) {
        self.count = self.count.wrapping_add(size);
        region
            .atomic_u64(self.at.header + CONSUMER)
            .store(self.count, Ordering::Release);
    }

    /// Before the host sleeps: sets `consumer_idle`, issues a full fence, and answers whether the
    /// guest published more meanwhile, which the host takes before it sleeps (clearing the flag
    /// again). A broken count answers true, so the host finds it.
    pub(crate) fn sleep_or_pending(&self, region: &Region) -> bool {
        let idle = region.atomic_u32(self.at.header + CONSUMER_IDLE);
        idle.store(1, Ordering::SeqCst);
        fence(Ordering::SeqCst);
        let pending = self.has_pending(region).unwrap_or(true);
        if pending {
            idle.store(0, Ordering::SeqCst);
        }
        pending
    }

    /// The host woke: clears `consumer_idle`, so the guest rings no doorbell while it takes.
    pub(crate) fn awake(&self, region: &Region) {
        region
            .atomic_u32(self.at.header + CONSUMER_IDLE)
            .store(0, Ordering::SeqCst);
    }
}

/// What a completion ring's post does about its interrupt ([`rearm_or_raise`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Wake {
    /// The guest had not armed the ring: nothing.
    Unarmed,
    /// Armed, with a record the guest has not taken: the ring stays cleared and the vector is
    /// raised.
    Raise,
    /// Armed, with every record taken: the arm is one the guest made after its take, or one it
    /// makes again before it sleeps, so the ring is armed again and nothing is raised.
    Rearm,
}

/// What a post does about the interrupt, from `armed` as its clear read it and the guest's
/// `consumer` read after the clear, against `count`, the host's count with the message published.
pub(crate) fn rearm_or_raise(prev_armed: u32, consumer: u64, count: u64) -> Wake {
    if prev_armed != 1 {
        Wake::Unarmed
    } else if consumer == count {
        Wake::Rearm
    } else {
        Wake::Raise
    }
}

/// What [`Producer::post`] did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Posted {
    /// The message is in the ring and published. `raise` is whether the guest had armed the
    /// ring and not taken every record ([`rearm_or_raise`]): the host raises the ring's vector.
    Yes {
        /// Raise the ring's interrupt.
        raise: bool,
    },
    /// The ring has no room for it until the guest takes more: nothing was written.
    NoRoom,
    /// The guest's count is out of reach of the host's: nothing was written.
    Broken(Broken),
}

/// The host's end of a completion ring: it writes what the guest reads.
#[derive(Debug)]
pub(crate) struct Producer {
    /// The ring.
    at: RingAt,
    /// The host's count: what it has published.
    count: u64,
    /// Tests: what the guest does between a post's publish and its clear of `armed`.
    #[cfg(test)]
    between: Option<Between>,
}

/// Tests: a guest's steps between a post's publish and its clear, given the region and the
/// host's count.
#[cfg(test)]
struct Between(BetweenSteps);

/// Tests: what [`Between`] runs.
#[cfg(test)]
type BetweenSteps = Box<dyn FnMut(&Region, u64) + Send>;

#[cfg(test)]
impl std::fmt::Debug for Between {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Between")
    }
}

impl Producer {
    /// The producer of a ring at `at`, nothing published yet.
    pub(crate) fn new(at: RingAt) -> Producer {
        Producer {
            at,
            count: 0,
            #[cfg(test)]
            between: None,
        }
    }

    /// Tests: runs `between` in each post from now on, after the publish and before the clear.
    #[cfg(test)]
    pub(crate) fn set_between(&mut self, between: impl FnMut(&Region, u64) + Send + 'static) {
        self.between = Some(Between(Box::new(between)));
    }

    /// Back to a ring nothing has passed through ([`RingAt::initialize`]), `consumer_idle` 0.
    pub(crate) fn reset(&mut self, region: &Region) {
        self.count = 0;
        self.at.initialize(region, false);
    }

    /// Writes `message` (its header first, its `size` its length, a multiple of 8, at least 16
    /// and at most the ring) and publishes it: first a pad to the ring's end where the bytes left
    /// there are fewer than the message. Then a full fence, `armed` read and cleared, and where it
    /// read 1, `consumer` read and the ring armed again where the guest took every record
    /// ([`rearm_or_raise`], [`Posted::Yes`]).
    pub(crate) fn post(&mut self, region: &Region, message: &[u8]) -> Posted {
        let len = message.len() as u64;
        debug_assert!(len.is_multiple_of(8) && len >= HEADER_SIZE && len <= self.at.data_size);
        let guest = region
            .atomic_u64(self.at.header + CONSUMER)
            .load(Ordering::Acquire);
        let used = self.count.wrapping_sub(guest);
        if used > self.at.data_size {
            return Posted::Broken(Broken::Count {
                guest,
                host: self.count,
            });
        }
        let free = self.at.data_size - used;
        let left = self.at.left_to_end(self.count);
        let pad = if left < len { left } else { 0 };
        if pad + len > free {
            return Posted::NoRoom;
        }
        if pad > 0 {
            let mut header = [0u8; 8];
            header[0..4].copy_from_slice(&RGFX_MSG_PAD.to_le_bytes());
            // The bytes left are below the ring's size, a u32.
            header[4..8].copy_from_slice(&(pad as u32).to_le_bytes());
            region.write(self.at.data_at(self.count), &header);
            self.count = self.count.wrapping_add(pad);
        }
        region.write(self.at.data_at(self.count), message);
        self.count = self.count.wrapping_add(len);
        // A command buffer that finished: its seqno is the queue's last completed, before its
        // record is published (`RgfxRingHeader::completed_seqno`).
        if let Ok(done) = RgfxCbComplete::read_from_bytes(message)
            && done.header.r#type == RGFX_EVT_CB_COMPLETE
            && done.status == RGFX_CB_OK
        {
            region
                .atomic_u64(self.at.header + COMPLETED_SEQNO)
                .fetch_max(done.header.seqno, Ordering::Release);
        }
        region
            .atomic_u64(self.at.header + PRODUCER)
            .store(self.count, Ordering::Release);
        #[cfg(test)]
        if let Some(between) = &mut self.between {
            (between.0)(region, self.count);
        }
        fence(Ordering::SeqCst);
        let armed = region.atomic_u32(self.at.header + ARMED);
        let prev = armed.swap(0, Ordering::SeqCst);
        let wake = if prev == 1 {
            let consumer = region
                .atomic_u64(self.at.header + CONSUMER)
                .load(Ordering::SeqCst);
            rearm_or_raise(prev, consumer, self.count)
        } else {
            Wake::Unarmed
        };
        if wake == Wake::Rearm {
            armed.store(1, Ordering::SeqCst);
        }
        Posted::Yes {
            raise: wake == Wake::Raise,
        }
    }
}

/// A message's bytes: `record` (whose first 16 bytes are its header, its `type` and `seqno` set)
/// with its `size` set to its length.
pub(crate) fn sized<T: IntoBytes + zerocopy::Immutable>(record: &T) -> Vec<u8> {
    let mut bytes = record.as_bytes().to_vec();
    set_size(&mut bytes);
    bytes
}

/// Sets the `size` of the message `bytes` holds to its length, which is a multiple of 8.
pub(crate) fn set_size(bytes: &mut [u8]) {
    let size = u32::try_from(bytes.len()).unwrap_or(u32::MAX);
    if let Some(field) = bytes.get_mut(4..8) {
        field.copy_from_slice(&size.to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire::{RgfxComplete, RgfxMsgHeader};

    const AT: RingAt = RingAt {
        header: 0,
        data_size: 256,
    };

    /// A guest writing messages into a submission ring at `AT`.
    struct Guest {
        count: u64,
    }

    impl Guest {
        fn write(&mut self, region: &Region, bytes: &[u8]) {
            let left = AT.left_to_end(self.count);
            if left < bytes.len() as u64 {
                let mut pad = [0u8; 8];
                pad[4..8].copy_from_slice(&(left as u32).to_le_bytes());
                region.write(AT.data_at(self.count), &pad);
                self.count += left;
            }
            region.write(AT.data_at(self.count), bytes);
            self.count += bytes.len() as u64;
            region
                .atomic_u64(PRODUCER)
                .store(self.count, Ordering::Release);
        }
    }

    fn message(r#type: u32, seqno: u64, len: usize) -> Vec<u8> {
        let mut bytes = vec![0xab; len];
        bytes[..16].copy_from_slice(
            RgfxMsgHeader {
                r#type,
                size: len as u32,
                seqno,
            }
            .as_bytes(),
        );
        bytes
    }

    /// Messages come out in order, the pads before the ring's end skipped, and the host's count
    /// reaches the header
    #[test]
    fn a_consumer_takes_messages_in_order_across_the_end() {
        let region = Region::new(4096).unwrap();
        let mut consumer = Consumer::new(AT);
        consumer.reset(&region);
        let mut guest = Guest { count: 0 };
        for seqno in 1..=40u64 {
            let len = 16 + 8 * (seqno as usize % 5);
            guest.write(&region, &message(7, seqno, len));
            let got = consumer.next(&region).unwrap().unwrap();
            assert_eq!(got.header.seqno, seqno);
            assert_eq!(got.bytes.len(), len);
            assert_eq!(consumer.next(&region).unwrap(), None);
            assert_eq!(
                region.atomic_u64(CONSUMER).load(Ordering::Acquire),
                guest.count
            );
        }
    }

    /// A walk reads what the guest published past the host's count, across the end and its pad,
    /// taking nothing: from where it is asked, at most as many as asked, and up to a message the
    /// protocol refuses.
    #[test]
    fn a_walk_reads_the_published_messages_without_taking_them() {
        let region = Region::new(4096).unwrap();
        let mut consumer = Consumer::new(AT);
        consumer.reset(&region);
        let mut guest = Guest { count: 0 };
        let mut field = |seqno: u64, len: usize| {
            let mut bytes = message(0x200 + seqno as u32, seqno, len);
            if len >= 24 {
                bytes[16..24].copy_from_slice(&(seqno * 10).to_le_bytes());
            }
            guest.write(&region, &bytes);
            guest.count
        };
        let first = field(1, 200);
        let short = field(2, 16);
        consumer.next(&region).unwrap().unwrap();
        // Past the end: a 40-byte pad, then the message at the ring's start.
        let wrapped = field(3, 48);
        assert_eq!(wrapped, 256 + 48);
        let mut seen = Vec::new();
        let walked = consumer.walk_published(&region, 0, usize::MAX, |through, r#type, f| {
            seen.push((through, r#type, f));
        });
        assert_eq!(walked, Walked::All(wrapped));
        assert_eq!(
            seen,
            [(short, 0x202, None), (wrapped, 0x203, Some(30))],
            "from the host's count, the pad skipped"
        );
        assert_eq!(consumer.consumed(), first, "nothing taken");
        seen.clear();
        let walked = consumer.walk_published(&region, short, 1, |through, r#type, f| {
            seen.push((through, r#type, f));
        });
        assert_eq!(walked, Walked::All(wrapped));
        assert_eq!(seen, [(wrapped, 0x203, Some(30))]);
        assert_eq!(
            consumer.walk_published(&region, first, 1, |_, _, _| {}),
            Walked::Most(short)
        );
        // A size the protocol refuses ends the walk there.
        let mut bad = message(0x204, 4, 24);
        bad[4..8].copy_from_slice(&20u32.to_le_bytes());
        region.write(AT.data_at(wrapped), &bad);
        region
            .atomic_u64(PRODUCER)
            .store(wrapped + 24, Ordering::Release);
        assert_eq!(
            consumer.walk_published(&region, wrapped, usize::MAX, |_, _, _| {}),
            Walked::Broken(wrapped)
        );
    }

    /// A pad of 8 bytes, the least the end can leave, is skipped
    #[test]
    fn an_eight_byte_pad_is_taken() {
        let region = Region::new(4096).unwrap();
        let mut consumer = Consumer::new(AT);
        consumer.reset(&region);
        let mut guest = Guest { count: 0 };
        guest.write(&region, &message(1, 1, 248));
        assert_eq!(consumer.next(&region).unwrap().unwrap().header.seqno, 1);
        guest.write(&region, &message(1, 2, 16));
        assert_eq!(guest.count, 248 + 8 + 16);
        assert_eq!(consumer.next(&region).unwrap().unwrap().header.seqno, 2);
    }

    /// A count out of reach and a size the protocol refuses break the ring
    #[test]
    fn bad_counts_and_sizes_break_the_ring() {
        let region = Region::new(4096).unwrap();
        let mut consumer = Consumer::new(AT);
        consumer.reset(&region);
        region.atomic_u64(PRODUCER).store(257, Ordering::Release);
        assert!(matches!(consumer.next(&region), Err(Broken::Count { .. })));
        for size in [0u32, 12, 8, 264] {
            let mut consumer = Consumer::new(AT);
            consumer.reset(&region);
            let mut bytes = message(1, 1, 16);
            bytes[4..8].copy_from_slice(&size.to_le_bytes());
            region.write(AT.data_at(0), &bytes);
            region.atomic_u64(PRODUCER).store(16, Ordering::Release);
            assert!(
                matches!(consumer.next(&region), Err(Broken::Size { .. })),
                "size {size}"
            );
        }
    }

    /// The idle flag is set before the host sleeps, and cleared where the guest published
    /// meanwhile
    #[test]
    fn the_consumer_sleeps_only_on_an_empty_ring() {
        let region = Region::new(4096).unwrap();
        let mut consumer = Consumer::new(AT);
        consumer.reset(&region);
        assert_eq!(region.atomic_u32(CONSUMER_IDLE).load(Ordering::Acquire), 1);
        assert!(!consumer.sleep_or_pending(&region));
        assert_eq!(region.atomic_u32(CONSUMER_IDLE).load(Ordering::Acquire), 1);
        consumer.awake(&region);
        assert_eq!(region.atomic_u32(CONSUMER_IDLE).load(Ordering::Acquire), 0);
        Guest { count: 0 }.write(&region, &message(1, 1, 16));
        assert!(consumer.sleep_or_pending(&region));
        assert_eq!(region.atomic_u32(CONSUMER_IDLE).load(Ordering::Acquire), 0);
    }

    /// The producer pads to the end, waits for room, and raises only where the guest armed the
    /// ring, clearing `armed`
    #[test]
    fn a_producer_pads_waits_for_room_and_raises_when_armed() {
        let region = Region::new(4096).unwrap();
        let mut producer = Producer::new(AT);
        producer.reset(&region);
        let complete = |seqno| {
            sized(&RgfxComplete {
                header: RgfxMsgHeader {
                    r#type: crate::wire::RGFX_EVT_COMPLETE,
                    size: 0,
                    seqno,
                },
                status: 0,
                reserved_14: 0,
            })
        };
        // 10 of 24 bytes fill 240 of 256.
        for seqno in 0..10 {
            assert_eq!(
                producer.post(&region, &complete(seqno)),
                Posted::Yes { raise: false }
            );
        }
        assert_eq!(producer.post(&region, &complete(10)), Posted::NoRoom);
        region.atomic_u64(CONSUMER).store(48, Ordering::Release);
        region.atomic_u32(ARMED).store(1, Ordering::Release);
        // 16 bytes to the end: a pad, then the message from the ring's start.
        assert_eq!(
            producer.post(&region, &complete(10)),
            Posted::Yes { raise: true }
        );
        assert_eq!(region.atomic_u32(ARMED).load(Ordering::Acquire), 0);
        assert_eq!(
            region.atomic_u64(PRODUCER).load(Ordering::Acquire),
            256 + 24
        );
        let pad = region.read(AT.data_at(240), 8).unwrap();
        assert_eq!(pad, [0, 0, 0, 0, 16, 0, 0, 0]);
        let first = region.read(AT.data_at(256), 24).unwrap();
        assert_eq!(
            RgfxComplete::read_from_bytes(&first).unwrap().header.seqno,
            10
        );
        region.atomic_u64(CONSUMER).store(1000, Ordering::Release);
        assert!(matches!(
            producer.post(&region, &complete(11)),
            Posted::Broken(_)
        ));
    }

    /// A command buffer that finished with no error moves the completed seqno, before its record
    /// is published; a failed one, any other event and an earlier seqno leave it, and a reset
    /// clears it
    #[test]
    fn a_finished_command_buffer_moves_the_completed_seqno() {
        let region = Region::new(4096).unwrap();
        let mut producer = Producer::new(AT);
        region
            .atomic_u64(COMPLETED_SEQNO)
            .store(77, Ordering::Release);
        producer.reset(&region);
        let completed = || region.atomic_u64(COMPLETED_SEQNO).load(Ordering::Acquire);
        assert_eq!(completed(), 0);
        let done = |seqno, status| {
            sized(&RgfxCbComplete {
                header: RgfxMsgHeader {
                    r#type: RGFX_EVT_CB_COMPLETE,
                    size: 0,
                    seqno,
                },
                cb_obj_id: 0x100 + seqno,
                status,
                ..Default::default()
            })
        };
        let posted = |producer: &mut Producer, bytes: &[u8]| {
            assert!(matches!(producer.post(&region, bytes), Posted::Yes { .. }));
            region.atomic_u64(CONSUMER).store(
                region.atomic_u64(PRODUCER).load(Ordering::Acquire),
                Ordering::Release,
            );
        };
        posted(&mut producer, &done(3, RGFX_CB_OK));
        assert_eq!(completed(), 3);
        posted(&mut producer, &done(5, crate::wire::RGFX_CB_PAGE_FAULT));
        assert_eq!(completed(), 3, "a failed buffer leaves it");
        posted(&mut producer, &done(2, RGFX_CB_OK));
        assert_eq!(completed(), 3, "an earlier seqno leaves it");
        let other = sized(&RgfxComplete {
            header: RgfxMsgHeader {
                r#type: crate::wire::RGFX_EVT_COMPLETE,
                size: 0,
                seqno: 9,
            },
            status: 0,
            reserved_14: 0,
        });
        posted(&mut producer, &other);
        assert_eq!(completed(), 3, "another event leaves it");
        posted(&mut producer, &done(6, RGFX_CB_OK));
        assert_eq!(completed(), 6);
        producer.reset(&region);
        assert_eq!(completed(), 0);
    }

    /// An unarmed ring raises nothing; an armed one raises while a record is not taken, and is
    /// armed again where every record is
    #[test]
    fn a_post_raises_or_arms_again_by_what_the_guest_took() {
        assert_eq!(rearm_or_raise(0, 0, 24), Wake::Unarmed);
        assert_eq!(rearm_or_raise(0, 24, 24), Wake::Unarmed);
        assert_eq!(rearm_or_raise(2, 24, 24), Wake::Unarmed, "only 1 arms");
        assert_eq!(rearm_or_raise(1, 0, 24), Wake::Raise);
        assert_eq!(rearm_or_raise(1, 16, 24), Wake::Raise);
        assert_eq!(rearm_or_raise(1, 24, 24), Wake::Rearm);
        assert_eq!(rearm_or_raise(1, u64::MAX, u64::MAX), Wake::Rearm);
    }

    /// A guest that takes the record and arms between the post's publish and its clear (a
    /// consumer already awake) keeps its arm, and nothing is raised; one that arms with the
    /// record not taken has it raised and cleared; one that only takes has nothing raised
    #[test]
    fn an_arm_after_the_take_survives_the_post() {
        let region = Region::new(4096).unwrap();
        let mut producer = Producer::new(AT);
        producer.reset(&region);
        let complete = sized(&RgfxComplete {
            header: RgfxMsgHeader {
                r#type: crate::wire::RGFX_EVT_COMPLETE,
                size: 0,
                seqno: 1,
            },
            status: 0,
            reserved_14: 0,
        });
        let take = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
        let arm = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
        let (t, a) = (take.clone(), arm.clone());
        producer.set_between(move |region, count| {
            if t.load(Ordering::SeqCst) {
                region.atomic_u64(CONSUMER).store(count, Ordering::Release);
            }
            if a.load(Ordering::SeqCst) {
                region.atomic_u32(ARMED).store(1, Ordering::SeqCst);
            }
        });
        let armed = || region.atomic_u32(ARMED).load(Ordering::SeqCst);
        assert_eq!(
            producer.post(&region, &complete),
            Posted::Yes { raise: false }
        );
        assert_eq!(armed(), 1, "the arm made after the take stands");
        take.store(false, Ordering::SeqCst);
        assert_eq!(
            producer.post(&region, &complete),
            Posted::Yes { raise: true }
        );
        assert_eq!(armed(), 0);
        take.store(true, Ordering::SeqCst);
        arm.store(false, Ordering::SeqCst);
        assert_eq!(
            producer.post(&region, &complete),
            Posted::Yes { raise: false }
        );
        assert_eq!(armed(), 0);
    }
}

#[cfg(test)]
mod interleavings;
