//! How long the guest waited for its pipeline creates' replies, on the counts line
//! (`reims_vgpu_vulkan::answer`'s `write_counts`). A diagnostic: which counts the host writes is
//! free under the contract, and nothing the guest sees changes.
//!
//! A render, mesh render or compute pipeline create waits from the host's take of it from ring 0,
//! on its context's lane (`GraphicsControl::handle_stage3`), to its reply's post: by the compile's
//! thread (`GraphicsControl::post_reply`), by ring 0's for a create refused as it is prepared, or
//! by the thread that ends the fill a held create waits for. Its process waits for that reply
//! (`compile`, "Order"). Render and compute creates are counted together. Each reply counts one
//! more of [`REPLIES_CODE`], its wait's nanoseconds in [`REPLY_NS_CODE`], summed, with the
//! longest in [`REPLY_MAX_NS_CODE`], which its count rises to, and, where it waited
//! [`SLOW_ANSWER_NS`] (a frame at 60 Hz) or more, one more of [`REPLIES_OVER_16MS_CODE`].
//!
//! [`WAITING_NS_CODE`] is the nanoseconds during which at least one of a device's creates waited,
//! counted as each such span ends: the union of their waits, so waits at once count once, and the
//! most a boot can have been held by its creates.
//!
//! A refused create is counted at its reply as every other, whether its prepare refused it or its
//! make did, and so is one the guest destroyed, or whose context closed, before its reply, which is
//! still posted (`compile`, "Order"). A create whose reply is never posted (one let go of at the
//! cancel of the fill it waited for, its reply finding no room, or a message too short to be
//! answered) ends its wait where it is let go of: in the span, not among the replies. One whose
//! reply the device's reset dropped (`CompletionPoster::post`) is counted as posted. A create
//! still waiting is in none of the amounts until its wait ends.
//!
//! **Its cost.** Two reads of the platform's monotonic clock a create, at its take and at its
//! post, each under the device's lock of its waits, with a push and a removal in the list of the
//! creates waiting at once; and up to five counts a reply.

use std::sync::atomic::{AtomicU64, Ordering};

use reims_vgpu_graphics::SLOW_ANSWER_NS;
use reims_vgpu_graphics::wire::{
    RGFX_CMD_COMPUTE_PIPELINE_CREATE, RGFX_CMD_MESH_RENDER_PIPELINE_CREATE,
    RGFX_CMD_RENDER_PIPELINE_CREATE,
};
use reims_vgpu_vulkan::answer::{count_kind, count_kind_by};
use reims_vgpu_vulkan::core::sync::Mutex;

use crate::graphics::compile::Now;

/// The code of the pipeline creates' replies posted.
pub(crate) const REPLIES_CODE: &str = "pipeline-create-replies";

/// The code of their waits' nanoseconds, from the create's take to its reply's post, summed.
pub(crate) const REPLY_NS_CODE: &str = "pipeline-create-reply-ns";

/// The code of the longest of those waits, in nanoseconds, which its count rises to.
pub(crate) const REPLY_MAX_NS_CODE: &str = "pipeline-create-reply-max-ns";

/// The code of the replies posted [`SLOW_ANSWER_NS`] or more after their create's take.
pub(crate) const REPLIES_OVER_16MS_CODE: &str = "pipeline-create-replies-over-16ms";

/// The code of the nanoseconds during which at least one create waited for its reply.
pub(crate) const WAITING_NS_CODE: &str = "pipeline-create-waiting-ns";

/// The longest wait so far, which [`REPLY_MAX_NS_CODE`] rises to.
static REPLY_MOST: AtomicU64 = AtomicU64::new(0);

/// Whether a ring-0 message of type `r#type` is a pipeline create, whose wait is counted.
pub(crate) fn is_create(r#type: u32) -> bool {
    matches!(
        r#type,
        RGFX_CMD_RENDER_PIPELINE_CREATE
            | RGFX_CMD_MESH_RENDER_PIPELINE_CREATE
            | RGFX_CMD_COMPUTE_PIPELINE_CREATE
    )
}

/// A device's pipeline creates waiting for their replies (the module's docs), on the platform's
/// monotonic clock.
pub(crate) struct CreateWaits {
    /// The platform's monotonic clock.
    now: Now,
    /// The creates waiting, and when their span began.
    waiting: Mutex<Waiting>,
}

/// The creates waiting for their replies.
#[derive(Default)]
struct Waiting {
    /// Each create waiting, by its object's id and its seqno, with the clock at its take.
    creates: Vec<(u64, u64, u64)>,
    /// The clock at the take that began the span: the first create's to wait while none did.
    since: u64,
}

impl CreateWaits {
    /// The waits of a device whose platform's monotonic clock `now` reads.
    pub(crate) fn new(now: Now) -> CreateWaits {
        CreateWaits {
            now,
            waiting: Mutex::default(),
        }
    }

    /// The host took the create of object `obj_id` at seqno `seqno`: its wait begins.
    pub(crate) fn taken(&self, obj_id: u64, seqno: u64) {
        let mut waiting = self.waiting.lock();
        let at = (self.now)().as_nanos();
        if waiting.creates.is_empty() {
            waiting.since = at;
        }
        waiting.creates.push((obj_id, seqno, at));
    }

    /// The reply of the create of `obj_id` at `seqno` was posted: its wait ends, and is counted.
    pub(crate) fn replied(&self, obj_id: u64, seqno: u64) {
        let Some(took) = self.ended(obj_id, seqno) else {
            return;
        };
        count_kind(REPLIES_CODE);
        count_kind_by(REPLY_NS_CODE, took);
        let was = REPLY_MOST.fetch_max(took, Ordering::Relaxed);
        count_kind_by(REPLY_MAX_NS_CODE, took.saturating_sub(was));
        if took >= SLOW_ANSWER_NS {
            count_kind(REPLIES_OVER_16MS_CODE);
        }
    }

    /// The create of `obj_id` at `seqno` was let go of with no reply posted: its wait ends, not
    /// counted as a reply.
    pub(crate) fn let_go(&self, obj_id: u64, seqno: u64) {
        self.ended(obj_id, seqno);
    }

    /// Ends the wait of the create of `obj_id` at `seqno` (the earliest taken, where two wait
    /// under both), and the span where no create waits after it, counted ([`WAITING_NS_CODE`]):
    /// the wait's nanoseconds, or `None` where no such create waits.
    fn ended(&self, obj_id: u64, seqno: u64) -> Option<u64> {
        let mut waiting = self.waiting.lock();
        let at = (self.now)().as_nanos();
        let found = (waiting.creates.iter()).position(|&(o, s, _)| (o, s) == (obj_id, seqno))?;
        let (_, _, taken) = waiting.creates.remove(found);
        if waiting.creates.is_empty() {
            count_kind_by(WAITING_NS_CODE, at.saturating_sub(waiting.since));
        }
        Some(at.saturating_sub(taken))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use reims_vgpu_vulkan::core::platform::os::clock::MonoTime;

    use super::*;
    use crate::platform::counted;

    /// A millisecond of the clock.
    const MS: u64 = 1_000_000;

    /// Waits on a clock the test sets, which reads `clock`.
    fn waits(clock: &Arc<AtomicU64>) -> CreateWaits {
        let clock = Arc::clone(clock);
        CreateWaits::new(Box::new(move || {
            MonoTime::from_nanos(clock.load(Ordering::Relaxed))
        }))
    }

    /// A create let go of with no reply ends its wait in the span, and is not counted as a
    /// reply; a reply of no create waiting counts nothing
    #[test]
    fn a_create_let_go_of_unreplied_ends_its_wait_uncounted() {
        let clock = Arc::new(AtomicU64::new(100 * MS));
        let waits = waits(&clock);
        let codes = [REPLIES_CODE, REPLY_NS_CODE, WAITING_NS_CODE];
        let before = codes.map(counted);
        waits.taken(1, 7);
        clock.store(130 * MS, Ordering::Relaxed);
        waits.let_go(1, 7);
        waits.replied(1, 7);
        assert_eq!(
            codes.map(counted),
            [before[0], before[1], before[2] + 30 * MS]
        );
    }
}
