//! VideoToolbox's paravirtual messages ([`RGFX_FEATURE_VT_MESSAGES`]): a `VT_MESSAGE` the guest's
//! VideoToolbox sends its host side, which the host reads from guest RAM and answers with a
//! [`VtAnswer`], and the messages of the host's side to the guest's, which it sends through a
//! [`VtLink`] (docs/graphics-kext/design.md, "Decided", 17). The device reads neither message: it
//! checks a `VT_MESSAGE`'s runs and length, and frames the host's messages.

use std::sync::Weak;
use std::sync::atomic::Ordering;

use zerocopy::{FromBytes, IntoBytes};

use super::Shared;
use crate::ring::Message;
use crate::wire::*;

/// The `VT_STATUS` a `VT_MESSAGE` owes, which the host gives from any thread once its side took
/// the message ([`super::Host::vt_message`]). It holds a weak reference to the device; an answer
/// after the device has ended, or for a message taken before a reset, is dropped, and so is one
/// never given.
pub struct VtAnswer {
    /// The device's shared state, weakly.
    shared: Weak<Shared>,
    /// The message's `seqno`.
    seqno: u64,
    /// The reset count the message was taken at.
    epoch: u64,
}

impl std::fmt::Debug for VtAnswer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VtAnswer")
            .field("seqno", &self.seqno)
            .finish_non_exhaustive()
    }
}

impl VtAnswer {
    /// Posts the `VT_STATUS` with `status` onto the control completion ring, waiting for nothing,
    /// as a [`super::CompleteAnswer`] posts its `COMPLETE`.
    pub fn answer(self, status: i32) {
        if let Some(shared) = self.shared.upgrade() {
            shared.post_unwaiting(&vt_status(self.seqno, status), self.epoch);
        }
    }
}

/// A handle that sends the guest's VideoToolbox the messages of its host side, from any thread
/// ([`Self::send`]). It holds the device weakly, so it keeps it alive no longer, and sends for the
/// reset count it was made at: once the device is reset, what it sends is dropped.
#[derive(Clone)]
pub struct VtLink {
    /// The device's shared state, weakly.
    shared: Weak<Shared>,
    /// The reset count of the `VT_MESSAGE` it came with.
    epoch: u64,
}

impl std::fmt::Debug for VtLink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VtLink")
            .field("epoch", &self.epoch)
            .finish_non_exhaustive()
    }
}

impl VtLink {
    /// Sends `message`, a `vtpv` message or a `rels` box of at most [`RGFX_VT_HOST_MESSAGE_MAX`]
    /// bytes, as a `VT_MESSAGE` event, in the order sent, waiting for nothing: posted here where
    /// the ring has room and nothing is handed over before it, else handed to the control thread.
    /// False, with nothing sent, for a longer message, a guest that did not take
    /// [`RGFX_FEATURE_VT_MESSAGES`], a device that has ended or is not started, or a reset since
    /// the link was made.
    pub fn send(&self, message: &[u8]) -> bool {
        let Some(shared) = self.shared.upgrade() else {
            return false;
        };
        let Some(event) = host_message(message) else {
            return false;
        };
        if !shared.driver_took(RGFX_FEATURE_VT_MESSAGES)
            || !shared.started()
            || self.stale_in(&shared)
        {
            return false;
        }
        shared.post_unwaiting(&event, self.epoch);
        true
    }

    /// Whether the device was reset since the link was made, or has ended: nothing it sends
    /// reaches the guest.
    pub fn stale(&self) -> bool {
        self.shared
            .upgrade()
            .is_none_or(|shared| self.stale_in(&shared))
    }

    fn stale_in(&self, shared: &Shared) -> bool {
        shared.epoch.load(Ordering::Acquire) != self.epoch
    }
}

/// A `VT_MESSAGE`'s answer: `VT_STATUS` with `status`.
pub(crate) fn vt_status(seqno: u64, status: i32) -> Vec<u8> {
    RgfxVtStatus {
        header: RgfxMsgHeader {
            r#type: RGFX_EVT_VT_STATUS,
            size: size_of::<RgfxVtStatus>() as u32,
            seqno,
        },
        status,
        reserved_14: 0,
    }
    .as_bytes()
    .to_vec()
}

/// The `VT_MESSAGE` event of `message`, its size rounded up to a multiple of 8; `None` for a
/// message longer than [`RGFX_VT_HOST_MESSAGE_MAX`].
pub(crate) fn host_message(message: &[u8]) -> Option<Vec<u8>> {
    let length = u32::try_from(message.len())
        .ok()
        .filter(|&l| l <= RGFX_VT_HOST_MESSAGE_MAX)?;
    let bytes = RGFX_VT_HOST_MESSAGE_BYTES_OFFSET + length;
    let size = bytes.next_multiple_of(8);
    let mut event = Vec::with_capacity(size as usize);
    event.extend_from_slice(
        RgfxVtHostMessage {
            header: RgfxMsgHeader {
                r#type: RGFX_EVT_VT_MESSAGE,
                size,
                seqno: 0,
            },
            length,
            reserved_14: 0,
        }
        .as_bytes(),
    );
    event.extend_from_slice(message);
    event.resize(size as usize, 0);
    Some(event)
}

/// A `VT_MESSAGE`, checked: its struct, its runs, and the surfaces it lends.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VtMessage {
    /// The message as the guest posted it.
    pub message: RgfxVtMessage,
    /// Its runs of guest RAM, `(address, length)`, in order, which hold its `length` bytes from
    /// the first run's first byte.
    pub runs: Vec<(u64, u64)>,
    /// The surfaces it lends, in order.
    pub surfaces: Vec<VtSurface>,
}

/// One surface a `VT_MESSAGE` lends: its record, and its runs of guest RAM, `(address, length)`,
/// in order, which hold its `length` bytes from the first run's first byte.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VtSurface {
    /// The record as the guest posted it.
    pub surface: RgfxVtSurface,
    /// Its runs.
    pub runs: Vec<(u64, u64)>,
}

/// `count` runs from the start of `bytes`, and what follows them; `None` where `bytes` holds
/// fewer, or they hold fewer than `length` bytes.
fn read_runs(bytes: &[u8], count: u32, length: u64) -> Option<(Vec<(u64, u64)>, &[u8])> {
    let (runs, rest) =
        bytes.split_at_checked((count as usize).checked_mul(size_of::<RgfxPageRun>())?)?;
    let runs: Vec<(u64, u64)> = runs
        .as_chunks::<{ size_of::<RgfxPageRun>() }>()
        .0
        .iter()
        .filter_map(|r| RgfxPageRun::read_from_bytes(r).ok())
        .map(|r| (r.address, r.length))
        .collect();
    let held = runs
        .iter()
        .try_fold(0u64, |sum, &(_, l)| sum.checked_add(l))?;
    (held >= length).then_some((runs, rest))
}

/// `message` as a `VT_MESSAGE`, checked: `None` where it is shorter than its struct and runs,
/// lends more surfaces than [`RGFX_VT_MESSAGE_SURFACES`] or more than its records hold, a record's
/// lend is not the one `surfaces` names for it, its own runs do not hold its `length`, or the
/// message's `length` is under a `vtpv` header's 48 bytes, over [`RGFX_VT_MESSAGE_MAX`] or more
/// than its runs hold.
pub(crate) fn read_message(message: &Message) -> Option<VtMessage> {
    let (command, rest) = RgfxVtMessage::read_from_prefix(&message.bytes).ok()?;
    let length = command.length;
    if !(48..=RGFX_VT_MESSAGE_MAX).contains(&length)
        || command.surface_count as usize > RGFX_VT_MESSAGE_SURFACES
    {
        return None;
    }
    let (runs, mut rest) = read_runs(rest, command.run_count, length)?;
    let mut surfaces = Vec::with_capacity(command.surface_count as usize);
    for &lend in command.surfaces.iter().take(command.surface_count as usize) {
        let (surface, after) = RgfxVtSurface::read_from_prefix(rest).ok()?;
        if surface.lend != lend || lend == 0 {
            return None;
        }
        let (runs, after) = read_runs(after, surface.run_count, surface.length)?;
        surfaces.push(VtSurface { surface, runs });
        rest = after;
    }
    Some(VtMessage {
        message: command,
        runs,
        surfaces,
    })
}

impl Shared {
    /// The answer `VT_MESSAGE` `seqno`, taken at reset count `epoch`, owes ([`VtAnswer`]).
    pub(crate) fn vt_answer(&self, seqno: u64, epoch: u64) -> VtAnswer {
        VtAnswer {
            shared: self.weak_self.get().cloned().unwrap_or_else(Weak::new),
            seqno,
            epoch,
        }
    }

    /// A link that sends the guest's VideoToolbox host messages at reset count `epoch`
    /// ([`VtLink`]).
    pub(crate) fn vt_link(&self, epoch: u64) -> VtLink {
        VtLink {
            shared: self.weak_self.get().cloned().unwrap_or_else(Weak::new),
            epoch,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ring::Message;

    fn message(command: RgfxVtMessage, runs: &[RgfxPageRun]) -> Message {
        let mut bytes = command.as_bytes().to_vec();
        for run in runs {
            bytes.extend_from_slice(run.as_bytes());
        }
        Message {
            header: command.header,
            bytes,
        }
    }

    fn command(length: u64, run_count: u32) -> RgfxVtMessage {
        RgfxVtMessage {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_VT_MESSAGE,
                size: RGFX_VT_MESSAGE_RUNS_OFFSET + 16 * run_count,
                seqno: 7,
            },
            length,
            run_count,
            ..RgfxVtMessage::default()
        }
    }

    #[test]
    fn a_message_its_runs_hold_is_read_with_them() {
        let runs = [
            RgfxPageRun {
                address: 0x4000,
                length: 0x4000,
            },
            RgfxPageRun {
                address: 0x10000,
                length: 0x4000,
            },
        ];
        let read = read_message(&message(command(0x5000, 2), &runs)).unwrap();
        assert_eq!(read.runs, vec![(0x4000, 0x4000), (0x10000, 0x4000)]);
        assert!(read.surfaces.is_empty());
    }

    /// A message of 64 bytes in one run, lending `lent` surfaces of one run of `length` bytes
    /// each, lend 9 and up, its `surfaces` naming `names`.
    fn lending(lent: u32, length: u64, names: &[u64]) -> Message {
        let mut c = command(64, 1);
        c.surface_count = lent;
        c.surfaces[..names.len()].copy_from_slice(names);
        let mut bytes = c.as_bytes().to_vec();
        bytes.extend_from_slice(
            RgfxPageRun {
                address: 0x4000,
                length: 0x4000,
            }
            .as_bytes(),
        );
        for i in 0..u64::from(lent) {
            let mut surface = RgfxVtSurface {
                lend: 9 + i,
                length,
                surface_id: 0x40 + i as u32,
                pixel_format: u32::from_be_bytes(*b"420v"),
                width: 64,
                height: 48,
                plane_count: 2,
                run_count: 1,
                ..RgfxVtSurface::default()
            };
            surface.planes[1].offset = 0x1000;
            bytes.extend_from_slice(surface.as_bytes());
            bytes.extend_from_slice(
                RgfxPageRun {
                    address: 0x20000 * (i + 1),
                    length: 0x4000,
                }
                .as_bytes(),
            );
        }
        Message {
            header: c.header,
            bytes,
        }
    }

    #[test]
    fn the_surfaces_a_message_lends_are_read_with_their_runs() {
        let read = read_message(&lending(2, 0x4000, &[9, 10])).unwrap();
        assert_eq!(read.surfaces.len(), 2);
        assert_eq!(read.surfaces[0].surface.lend, 9);
        assert_eq!(read.surfaces[0].surface.planes[1].offset, 0x1000);
        assert_eq!(read.surfaces[1].runs, vec![(0x40000, 0x4000)]);
        // A record whose lend is not the one `surfaces` names, or which is missing.
        assert!(read_message(&lending(2, 0x4000, &[9, 11])).is_none());
        assert!(read_message(&lending(1, 0x4000, &[9, 10])).is_some());
        let mut short = lending(2, 0x4000, &[9, 10]);
        short.bytes.truncate(short.bytes.len() - 8);
        assert!(read_message(&short).is_none());
        // A surface whose runs do not hold its length.
        assert!(read_message(&lending(1, 0x4001, &[9])).is_none());
    }

    #[test]
    fn a_message_is_refused_where_its_runs_or_length_do_not_hold() {
        let run = [RgfxPageRun {
            address: 0x4000,
            length: 0x4000,
        }];
        // More bytes than the runs hold.
        assert!(read_message(&message(command(0x4001, 1), &run)).is_none());
        // Under a `vtpv` header.
        assert!(read_message(&message(command(47, 1), &run)).is_none());
        // Over the most a message buffer holds.
        assert!(read_message(&message(command(RGFX_VT_MESSAGE_MAX + 1, 1), &run)).is_none());
        // Runs past the message's end.
        assert!(read_message(&message(command(64, 2), &run)).is_none());
        // More surfaces than a message lends.
        let mut c = command(64, 1);
        c.surface_count = 6;
        assert!(read_message(&message(c, &run)).is_none());
    }

    #[test]
    fn a_host_message_is_framed_and_padded_to_8() {
        let event = host_message(&[1, 2, 3]).unwrap();
        assert_eq!(event.len(), 32);
        let (h, rest) = RgfxVtHostMessage::read_from_prefix(&event).unwrap();
        assert_eq!(h.header.r#type, RGFX_EVT_VT_MESSAGE);
        assert_eq!(h.header.size, 32);
        assert_eq!(h.header.seqno, 0);
        assert_eq!(h.length, 3);
        assert_eq!(rest, &[1, 2, 3, 0, 0, 0, 0, 0]);
        let longest = vec![0u8; RGFX_VT_HOST_MESSAGE_MAX as usize];
        assert!(host_message(&longest).is_some());
        assert!(host_message(&[0u8; RGFX_VT_HOST_MESSAGE_MAX as usize + 1]).is_none());
    }

    #[test]
    fn a_status_answers_its_message() {
        let bytes = vt_status(7, -12909);
        let (s, _) = RgfxVtStatus::read_from_prefix(&bytes).unwrap();
        assert_eq!(s.header.r#type, RGFX_EVT_VT_STATUS);
        assert_eq!(s.header.seqno, 7);
        assert_eq!(s.status, -12909);
    }
}
