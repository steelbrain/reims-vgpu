//! An encoded frame: its picture's place in the stream, the bitstream
//! buffer its NAL units went into, the timeline value after which they are
//! there, and what the device says it wrote, from an encode feedback query.

use std::mem::ManuallyDrop;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use ash::vk;
use reims_vgpu_video_core::sync::Mutex;

use crate::device::VideoDevice;
use crate::encode::caps::EncodeProfile;
use crate::encode::io::BitstreamTarget;
use crate::encode::plan::PictureKind;
use crate::error::{Error, Named};
use crate::timeline::Timeline;

/// Feedback queries a pool holds.
const POOL_QUERIES: u32 = 16;

/// Timestamps a picture writes: as its work starts, after its source's
/// copy, and as it ends.
const STAMPS: u32 = 3;

/// What the device says of an encode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EncodeStatus {
    /// Done: the bitstream holds the picture.
    Complete,
    /// The bitstream buffer was too small for the picture, which it holds
    /// cut short (`VK_QUERY_RESULT_STATUS_INSUFFICIENT_BITSTREAM_BUFFER_RANGE_KHR`).
    InsufficientBuffer,
    /// The encode failed.
    Failed,
    /// Another status, raw.
    Other(i64),
}

impl EncodeStatus {
    fn from_raw(status: i64) -> Self {
        match status {
            1 => Self::Complete,
            -1 => Self::Failed,
            -1_000_299_000 => Self::InsufficientBuffer,
            other => Self::Other(other),
        }
    }
}

/// What an encode wrote, by its feedback query.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Feedback {
    /// The status.
    pub status: EncodeStatus,
    /// The in-band parameter sets the GPU wrote before the picture, where
    /// the frame carries them: their offset in the bitstream buffer, and
    /// their bytes.
    pub parameter_sets: Option<(u64, u64)>,
    /// The picture's own NAL units: their offset in the bitstream buffer,
    /// and their bytes.
    pub picture: (u64, u64),
}

impl Feedback {
    /// The frame's bytes as one range of the buffer, offset and length,
    /// its parameter sets first: `None` where the device left a gap
    /// between the parameter sets and the picture.
    pub fn range(&self) -> Option<(u64, u64)> {
        match self.parameter_sets {
            None => Some(self.picture),
            Some((offset, len)) if offset + len == self.picture.0 => {
                Some((offset, len + self.picture.1))
            }
            Some(_) => None,
        }
    }
}

/// The GPU time of a picture's work, from timestamps around it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuTime {
    /// From its start to its end: the source's copy, if any, and the
    /// encode.
    pub total: Duration,
    /// The source's copy alone: zero for a source read in place.
    pub copy: Duration,
}

/// The query pools of an encoder: encode feedback, and timestamps where
/// it times its pictures, each slot one picture's.
pub(crate) struct Queries {
    device: Arc<VideoDevice>,
    profile: EncodeProfile,
    flags: vk::VideoEncodeFeedbackFlagsKHR,
    /// The period of a timestamp tick, in nanoseconds, and the mask of its
    /// valid bits; none where pictures are not timed.
    timing: Option<(f32, u64)>,
    inner: Mutex<QueryPools>,
}

#[derive(Default)]
struct QueryPools {
    /// Feedback pools, each with its timestamp pool (null where untimed).
    pools: Vec<(vk::QueryPool, vk::QueryPool)>,
    free: Vec<(usize, u32)>,
}

impl Queries {
    /// Pools for encodes of `profile`, with the feedback `flags`, timing
    /// each picture where `timing` gives the timestamp period and valid
    /// bits.
    pub(crate) fn new(
        device: &Arc<VideoDevice>,
        profile: EncodeProfile,
        flags: vk::VideoEncodeFeedbackFlagsKHR,
        timing: Option<(f32, u32)>,
    ) -> Arc<Self> {
        Arc::new(Self {
            device: Arc::clone(device),
            profile,
            flags,
            timing: timing.map(|(period, bits)| {
                let mask = if bits >= 64 {
                    u64::MAX
                } else {
                    (1u64 << bits) - 1
                };
                (period, mask)
            }),
            inner: Mutex::new(QueryPools::default()),
        })
    }

    /// A slot no live picture holds.
    pub(crate) fn slot(self: &Arc<Self>) -> Result<QuerySlot, Error> {
        let mut inner = self.inner.lock();
        if inner.free.is_empty() {
            let pools = self.new_pools()?;
            let at = inner.pools.len();
            inner.pools.push(pools);
            inner.free.extend((0..POOL_QUERIES).rev().map(|q| (at, q)));
        }
        let (pool, index) = inner
            .free
            .pop()
            .ok_or_else(|| Error::Unsupported("no query slot".into()))?;
        let (feedback, stamps) = inner
            .pools
            .get(pool)
            .copied()
            .ok_or_else(|| Error::Unsupported("no query pool".into()))?;
        Ok(QuerySlot {
            queries: Arc::clone(self),
            pool,
            index,
            feedback,
            stamps,
        })
    }

    fn new_pools(&self) -> Result<(vk::QueryPool, vk::QueryPool), Error> {
        let vk_device = self.device.device();
        let feedback = self.profile.with_info(|profile| {
            let mut profile = *profile;
            let mut feedback = vk::QueryPoolVideoEncodeFeedbackCreateInfoKHR::default()
                .encode_feedback_flags(self.flags);
            let info = vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::VIDEO_ENCODE_FEEDBACK_KHR)
                .query_count(POOL_QUERIES)
                .push_next(&mut feedback)
                .push_next(&mut profile);
            // SAFETY: `info` and its chain, the profile's own chain
            // included, live through the call.
            unsafe { vk_device.create_query_pool(&info, None) }.named("vkCreateQueryPool")
        })?;
        if self.timing.is_none() {
            return Ok((feedback, vk::QueryPool::null()));
        }
        let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(POOL_QUERIES * STAMPS);
        // SAFETY: `info` lives through the call.
        match unsafe { vk_device.create_query_pool(&info, None) } {
            Ok(stamps) => Ok((feedback, stamps)),
            Err(error) => {
                // SAFETY: the pool is the device's and unused.
                unsafe { vk_device.destroy_query_pool(feedback, None) };
                Err(Error::Vulkan("vkCreateQueryPool", error))
            }
        }
    }
}

impl Drop for Queries {
    fn drop(&mut self) {
        let vk_device = self.device.device();
        for &(feedback, stamps) in &self.inner.lock().pools {
            // SAFETY: every slot is back (each held this), so no work uses
            // the pools; a null pool is nothing.
            unsafe { vk_device.destroy_query_pool(feedback, None) };
            // SAFETY: as above.
            unsafe { vk_device.destroy_query_pool(stamps, None) };
        }
    }
}

/// One picture's feedback query, and its timestamps where it is timed,
/// back to its pools as it goes.
pub(crate) struct QuerySlot {
    queries: Arc<Queries>,
    pool: usize,
    index: u32,
    pub(crate) feedback: vk::QueryPool,
    /// Null where untimed.
    pub(crate) stamps: vk::QueryPool,
}

impl QuerySlot {
    /// Its feedback query's index in [`Self::feedback`].
    pub(crate) fn index(&self) -> u32 {
        self.index
    }

    /// Its first timestamp's index in [`Self::stamps`]: the start, then
    /// after the copy, then the end.
    pub(crate) fn stamp(&self, which: u32) -> u32 {
        self.index * STAMPS + which
    }

    /// Its timestamps' count.
    pub(crate) fn stamps_count() -> u32 {
        STAMPS
    }

    /// The feedback of the encode it was used for: offset, bytes and
    /// status. The work is done.
    fn read_feedback(&self) -> Result<(u64, u64, EncodeStatus), Error> {
        // 32-bit results: a picture's offset and bytes are far under 4 GiB,
        // and NVIDIA's 615.71.09 writes them 32-bit even where 64-bit
        // results are asked for (the status alone at its 64-bit place),
        // which reads as an offset of `bytes << 32` and no bytes (11-03).
        let mut out = [[0u32; 3]; 1];
        // SAFETY: the pool is the device's, and the query was used by work
        // that is done; the output holds one result of three 32-bit values.
        unsafe {
            self.queries.device.device().get_query_pool_results(
                self.feedback,
                self.index,
                &mut out,
                vk::QueryResultFlags::WITH_STATUS_KHR | vk::QueryResultFlags::WAIT,
            )
        }
        .named("vkGetQueryPoolResults")?;
        let [[offset, bytes, status]] = out;
        Ok((
            u64::from(offset),
            u64::from(bytes),
            EncodeStatus::from_raw(i64::from(status.cast_signed())),
        ))
    }

    /// The GPU time of its picture, where timed. The work is done.
    fn read_time(&self) -> Result<Option<GpuTime>, Error> {
        let Some((period, mask)) = self.queries.timing else {
            return Ok(None);
        };
        if self.stamps == vk::QueryPool::null() {
            return Ok(None);
        }
        let mut out = [0u64; 3];
        // SAFETY: as in `read_feedback`, for the three timestamps.
        unsafe {
            self.queries.device.device().get_query_pool_results(
                self.stamps,
                self.stamp(0),
                &mut out,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )
        }
        .named("vkGetQueryPoolResults")?;
        let ticks = |from: u64, to: u64| to.wrapping_sub(from) & mask;
        // A picture's work is far under 2^32 ticks.
        let nanos = |ticks: u64| {
            let ticks = f64::from(u32::try_from(ticks).unwrap_or(u32::MAX));
            Duration::from_secs_f64(ticks * f64::from(period) * 1e-9)
        };
        let [start, copied, end] = out;
        Ok(Some(GpuTime {
            total: nanos(ticks(start, end)),
            copy: nanos(ticks(start, copied)),
        }))
    }
}

impl Drop for QuerySlot {
    fn drop(&mut self) {
        self.queries.inner.lock().free.push((self.pool, self.index));
    }
}

/// An encoded frame, in decode order: what the encoder knows of its
/// picture, and its bitstream target, which holds its NAL units once
/// [`Self::semaphore`] reaches [`Self::value`].
pub struct EncodedFrame<T: BitstreamTarget> {
    /// The caller's tag for the frame.
    pub id: u64,
    /// Its presentation time stamp, as it came.
    pub pts: i64,
    /// Its decode time stamp: the PTS of the frame sent in its place, as
    /// Apple's encoders give it, so increasing, a B picture's after its
    /// PTS ([`reims_vgpu_video_codec::encode::gop::DecodeTimes`]).
    pub dts: i64,
    /// Its type.
    pub kind: PictureKind,
    /// Its order count: H.264's `PicOrderCnt` (twice its place in display
    /// order from its IDR picture), H.265's `PicOrderCntVal` (its place).
    pub order_count: i32,
    /// H.264's `frame_num`; 0 for H.265.
    pub frame_num: u32,
    /// Whether later pictures may predict from it.
    pub reference: bool,
    output: Arc<Output<T>>,
}

/// What a frame holds that its GPU work uses until it is done.
pub(crate) struct Output<T: BitstreamTarget> {
    timeline: Arc<Timeline>,
    held: ManuallyDrop<Held<T>>,
    value: u64,
    /// Where the encode wrote from, in the target's buffer.
    encode_offset: u64,
    /// The in-band parameter sets the GPU wrote, offset and bytes.
    parameter_sets: Option<(u64, u64)>,
    feedback: OnceLock<Feedback>,
}

struct Held<T> {
    target: T,
    query: QuerySlot,
}

impl<T: BitstreamTarget> Output<T> {
    pub(crate) fn new(
        timeline: &Arc<Timeline>,
        target: T,
        query: QuerySlot,
        value: u64,
        encode_offset: u64,
        parameter_sets: Option<(u64, u64)>,
    ) -> Arc<Self> {
        Arc::new(Self {
            timeline: Arc::clone(timeline),
            held: ManuallyDrop::new(Held { target, query }),
            value,
            encode_offset,
            parameter_sets,
            feedback: OnceLock::new(),
        })
    }
}

impl<T: BitstreamTarget> Drop for Output<T> {
    fn drop(&mut self) {
        // SAFETY: the drop runs once, and nothing reads `held` after it.
        let held = unsafe { ManuallyDrop::take(&mut self.held) };
        self.timeline.retire(self.value, Box::new(held));
    }
}

impl<T: BitstreamTarget> EncodedFrame<T> {
    pub(crate) fn new(info: FrameInfo, output: Arc<Output<T>>) -> Self {
        Self {
            id: info.id,
            pts: info.pts,
            dts: info.dts,
            kind: info.kind,
            order_count: info.order_count,
            frame_num: info.frame_num,
            reference: info.reference,
            output,
        }
    }

    /// The bitstream target its NAL units go into.
    pub fn target(&self) -> &T {
        &self.output.held.target
    }

    /// The encoder's timeline semaphore.
    pub fn semaphore(&self) -> vk::Semaphore {
        self.output.timeline.semaphore()
    }

    /// The value [`Self::semaphore`] reaches once the target holds the
    /// frame's NAL units: work that reads them waits for it.
    pub fn value(&self) -> u64 {
        self.output.value
    }

    /// Blocks until the target holds the frame, for `timeout` at most.
    pub fn wait(&self, timeout: Duration) -> Result<(), Error> {
        self.output.timeline.wait(self.output.value, timeout)
    }

    /// Whether its NAL units start with in-band parameter sets.
    pub fn carries_parameter_sets(&self) -> bool {
        self.output.parameter_sets.is_some()
    }

    /// What the encode wrote, once it is done: it waits for that, for
    /// `timeout` at most.
    pub fn feedback(&self, timeout: Duration) -> Result<Feedback, Error> {
        if let Some(feedback) = self.output.feedback.get() {
            return Ok(*feedback);
        }
        self.wait(timeout)?;
        let (offset, bytes, status) = self.output.held.query.read_feedback()?;
        let feedback = Feedback {
            status,
            parameter_sets: self.output.parameter_sets,
            picture: (self.output.encode_offset.saturating_add(offset), bytes),
        };
        Ok(*self.output.feedback.get_or_init(|| feedback))
    }

    /// The GPU time of its picture's work, where the encoder times its
    /// pictures ([`crate::EncodeConfig::timing`]), once it is done.
    pub fn gpu_time(&self, timeout: Duration) -> Result<Option<GpuTime>, Error> {
        self.wait(timeout)?;
        self.output.held.query.read_time()
    }
}

/// What a frame's picture is, as the encoder made it.
pub(crate) struct FrameInfo {
    pub(crate) id: u64,
    pub(crate) pts: i64,
    pub(crate) dts: i64,
    pub(crate) kind: PictureKind,
    pub(crate) order_count: i32,
    pub(crate) frame_num: u32,
    pub(crate) reference: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn statuses_map_from_the_query() {
        assert_eq!(EncodeStatus::from_raw(1), EncodeStatus::Complete);
        assert_eq!(EncodeStatus::from_raw(-1), EncodeStatus::Failed);
        assert_eq!(
            EncodeStatus::from_raw(-1_000_299_000),
            EncodeStatus::InsufficientBuffer
        );
        assert_eq!(EncodeStatus::from_raw(0), EncodeStatus::Other(0));
    }

    #[test]
    fn a_frame_is_one_range_where_its_parts_meet() {
        let mut feedback = Feedback {
            status: EncodeStatus::Complete,
            parameter_sets: Some((740, 28)),
            picture: (768, 1000),
        };
        assert_eq!(feedback.range(), Some((740, 1028)));
        feedback.picture.0 = 1024;
        assert_eq!(feedback.range(), None);
        feedback.parameter_sets = None;
        assert_eq!(feedback.range(), Some((1024, 1000)));
    }
}
