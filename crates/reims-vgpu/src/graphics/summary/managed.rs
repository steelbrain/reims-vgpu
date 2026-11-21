//! The summary's **Managed** part ([`line()`]): the ranges of `Managed` buffers the host carried
//! out with the platform's `didModifyRange:` over each period, by who named them, the app's apart
//! from the bundle's upload chunks of `set*Bytes:` data:
//!
//! - **The app's**: the `BUFFER_DID_MODIFY_RANGE` messages carried out, and their bytes
//!   ([`range_carried_out`], which the decoder's `did_modify_range` op calls once the platform's
//!   call returned).
//! - **Upload chunks**: the chunks a command buffer's `RGFX_OP_UPLOADS_WRITTEN` names that were
//!   carried out, their bytes, and the command buffers that named at least one
//!   ([`uploads_carried_out`], which the decoder's `uploads_written` calls once a buffer's records
//!   are carried out).
//!
//! A range or record the host refuses (a malformed range, a bad reference, a record of no
//! `Managed` buffer or past one), skips (a refused create's, a record of no bytes) or cannot carry
//! out (the platform raised) counts in neither.
//!
//! **A chunk's range message counts among the app's.** The bundle names a chunk's bytes in a range
//! message instead of its stream where its queue does not name them there (a guest set that did
//! not take `RGFX_FEATURE_UPLOADS_WRITTEN`, which every host of this tree offers, or a chunk with
//! no range of its own, whose `rgfxRangeObjId` answers 0). Nothing the host reads tells that
//! message from an app's: it is the `BUFFER_DID_MODIFY_RANGE` an app's `didModifyRange:` sends
//! (the bundle's `RgfxBuffer.mm`), of a buffer made as an app's `Managed` buffer is
//! (`RgfxUploads.mm`), from its start, as an app's range can be.
//!
//! The counts are of the platform's calls. Where the GPU shares the CPU's memory the calls copy
//! nothing; on any other GPU rt copies each range into the buffer's device copy by the next
//! commit, and counts the bytes in its own `managed_bytes_captured` (the rt part), after merging
//! the ranges that overlap or touch, with the ranges the platform names itself (a `Managed`
//! buffer made with bytes, texture writes over a `Managed` buffer).
//!
//! Counted only once [`count`] turned counting on ([`super::start`]): off, a range costs one
//! relaxed load, and so does a command buffer whose records carried out a chunk.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Whether the ranges are counted ([`count`]).
static ON: AtomicBool = AtomicBool::new(false);

/// The app's ranges carried out since the take before.
static APP_RANGES: AtomicU64 = AtomicU64::new(0);
/// Their bytes.
static APP_RANGE_BYTES: AtomicU64 = AtomicU64::new(0);
/// The command buffers whose records carried out a chunk since the take before.
static UPLOAD_BUFFERS: AtomicU64 = AtomicU64::new(0);
/// The chunks their records carried out.
static CHUNKS: AtomicU64 = AtomicU64::new(0);
/// Those chunks' bytes.
static CHUNK_BYTES: AtomicU64 = AtomicU64::new(0);

/// What was counted over a period ([`take`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Period {
    /// The app's `BUFFER_DID_MODIFY_RANGE` messages carried out.
    pub(crate) app_ranges: u64,
    /// Their bytes.
    pub(crate) app_range_bytes: u64,
    /// The command buffers whose `UPLOADS_WRITTEN` records carried out at least one chunk.
    pub(crate) upload_buffers: u64,
    /// The chunks those records carried out.
    pub(crate) chunks: u64,
    /// Those chunks' bytes.
    pub(crate) chunk_bytes: u64,
}

/// Turns counting on, for the rest of the process.
pub(crate) fn count() {
    ON.store(true, Ordering::Relaxed);
}

/// Whether the ranges are counted.
fn on() -> bool {
    ON.load(Ordering::Relaxed)
}

/// The platform carried out an app's range of `length` bytes: counted where counting is on.
pub(crate) fn range_carried_out(length: u64) {
    if on() {
        APP_RANGES.fetch_add(1, Ordering::Relaxed);
        APP_RANGE_BYTES.fetch_add(length, Ordering::Relaxed);
    }
}

/// The platform carried out `chunks` of a command buffer's upload chunks, of `bytes` in all: the
/// buffer counted once, where it carried out one or more and counting is on.
pub(crate) fn uploads_carried_out(chunks: u64, bytes: u64) {
    if chunks > 0 && on() {
        UPLOAD_BUFFERS.fetch_add(1, Ordering::Relaxed);
        CHUNKS.fetch_add(chunks, Ordering::Relaxed);
        CHUNK_BYTES.fetch_add(bytes, Ordering::Relaxed);
    }
}

/// What was counted since the take before; each is counted from none again.
pub(crate) fn take() -> Period {
    Period {
        app_ranges: APP_RANGES.swap(0, Ordering::Relaxed),
        app_range_bytes: APP_RANGE_BYTES.swap(0, Ordering::Relaxed),
        upload_buffers: UPLOAD_BUFFERS.swap(0, Ordering::Relaxed),
        chunks: CHUNKS.swap(0, Ordering::Relaxed),
        chunk_bytes: CHUNK_BYTES.swap(0, Ordering::Relaxed),
    }
}

/// The summary's part of `period`: `managed ranges: ...`, the app's with their bytes, then the
/// upload chunks with their bytes and the command buffers that named them.
pub(crate) fn line(period: &Period) -> String {
    format!(
        "managed ranges: the app's {}, their bytes {}; upload chunks {}, their bytes {}, in \
         command buffers {}",
        period.app_ranges,
        period.app_range_bytes,
        period.chunks,
        period.chunk_bytes,
        period.upload_buffers,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The part gives the app's ranges and their bytes, then the chunks, their bytes and their
    /// command buffers, each as counted, none left out at 0.
    #[test]
    fn the_managed_part_gives_the_app_s_ranges_then_the_upload_chunks() {
        let period = Period {
            app_ranges: 12,
            app_range_bytes: 49_152,
            upload_buffers: 2,
            chunks: 3,
            chunk_bytes: 196_608,
        };
        assert_eq!(
            line(&period),
            "managed ranges: the app's 12, their bytes 49152; upload chunks 3, their bytes 196608, \
             in command buffers 2"
        );
        assert_eq!(
            line(&Period::default()),
            "managed ranges: the app's 0, their bytes 0; upload chunks 0, their bytes 0, in \
             command buffers 0"
        );
    }
}
