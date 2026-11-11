//! The host's own encode sessions on the reims-graphics device ([`HostEncodeSession`]): hardware
//! encode on a host API whose work no runtime queue carries (VA-API, NVENC), whose access units
//! the host writes into the guest's buffers, and whose session event the host signals. A backend
//! gives the encoder ([`HostEncoder`]) and each frame's encode ([`HostEncodedFrame`]); the rest,
//! here, is the same for every backend:
//!
//! - **An encode** ([`GraphicsVideoSession::encode`]) checks the op on the thread that takes the
//!   command buffer: the settings it carries (the session's first encode carries them, and each
//!   change after), the source's size against them, and the buffers the access unit and its
//!   result go in. It hands the encoder the frame, and sends it to the session's encode thread
//!   (`reims-vgpu.graphics-video-encode`). A refusal faults the command buffer and spends no value
//!   of the event.
//! - **The encode thread** finishes each frame in the order the encodes were taken: it reads the
//!   source's planes as host bytes, in the surface's pages, or, for a plane in the surface's
//!   device cache, in a staging buffer the GPU copies the plane into first. The encode writes the
//!   access unit into the bitstream buffer where it fits; the thread writes the frame's
//!   [`RgfxVideoEncodeResult`] after it, and then signals the encode's value. An access unit
//!   past the capacity is not written ([`RGFX_VIDEO_ENCODE_OVERFLOW`]), and a frame the encode
//!   fails gets [`RGFX_VIDEO_ENCODE_FAILED`]: the command buffer completes as any other.
//! - **The source** is read once its command buffer's waits are done: the guest writes it again
//!   only after the command buffer completes, which waits for the encode's value, so the encode
//!   thread may read it in place.
//! - **An end** ([`Drop`]) closes the encode thread's queue and waits up to [`TEARDOWN_WAIT`] for
//!   the last value: the thread finishes every frame it was given first.
//!
//! The session logs the first frame of each result at `info` ("hardware video: ..."), and its
//! counts at its end.

use std::sync::{Arc, mpsc};

use reims_vgpu_graphics::ops::video::{
    RGFX_VIDEO_ENCODE_FAILED, RGFX_VIDEO_ENCODE_OK, RGFX_VIDEO_ENCODE_OVERFLOW,
    RGFX_VIDEO_ENCODED_KEYFRAME, RGFX_VIDEO_ENCODED_PARAMETER_SETS, RGFX_VIDEO_ENTROPY_CABAC,
    RGFX_VIDEO_ENTROPY_CAVLC, RGFX_VIDEO_RATE_CBR, RGFX_VIDEO_RATE_CQP, RGFX_VIDEO_RATE_VBR,
    RgfxVideoEncodeResult, RgfxVideoEncodeSettings,
};
use reims_vgpu_graphics::wire::{
    RGFX_VIDEO_CODEC_H264, RGFX_VIDEO_CODEC_HEVC, RGFX_VIDEO_FORMAT_NV12, RGFX_VIDEO_FORMAT_P010,
    RGFX_VIDEO_H264_BASELINE, RGFX_VIDEO_H264_HIGH, RGFX_VIDEO_H264_MAIN, RGFX_VIDEO_HEVC_MAIN,
    RGFX_VIDEO_HEVC_MAIN_10,
};
use reims_vgpu_vulkan::core::host::HostPtr;
use reims_vgpu_vulkan::core::platform::Handle as _;
use reims_vgpu_vulkan::core::platform::Platform;
use reims_vgpu_vulkan::core::platform::device::GpuDevice as _;
use reims_vgpu_vulkan::core::platform::event::{EventRef, SharedEvent as _};
use reims_vgpu_vulkan::core::platform::resource::Buffer as _;
use reims_vgpu_vulkan::core::raise::{Raise, RaiseResult};
use reims_vgpu_vulkan::core::sync::Mutex;
use zerocopy::IntoBytes;

use super::super::decode::{Decoded, GraphicsVideoSession, VideoDecode, VideoEncode, op_fault};
use super::{
    BackendError, FAILURES_LOGGED, OutputPlane, SourcePlane, Stage, Staged, TEARDOWN_WAIT,
    output_plane_of, plane_end,
};

/// One frame a backend's encoder takes ([`HostEncoder::encode`]).
#[derive(Clone, Copy, Debug)]
pub(crate) struct EncodeFrame {
    /// The settings from this frame on, where they changed since the last frame; the first
    /// frame's are always here.
    pub(crate) settings: Option<RgfxVideoEncodeSettings>,
    /// The frame's place in display order.
    pub(crate) frame_index: u64,
    /// Whether it must be a keyframe.
    pub(crate) force_keyframe: bool,
}

/// A backend's encoder of one codec, whose session is a [`HostEncodeSession`]. Its calls run on
/// the thread that takes each encode's command buffer, in the order the encodes were taken.
pub(crate) trait HostEncoder: Send + 'static {
    /// Checks settings an encode carries, beyond what the session checks of every backend's
    /// (their size, format, profile, rate control and entropy): what the hardware takes of them.
    /// `Err` faults the encode's command buffer.
    fn check(&self, settings: &RgfxVideoEncodeSettings) -> Result<(), BackendError>;

    /// Takes the next frame, in display order, with the settings it carries applied from it on,
    /// and answers its encode, which the session's encode thread finishes, in the order taken.
    /// `Err` faults the encode's command buffer.
    fn encode(&mut self, frame: EncodeFrame) -> Result<Box<dyn HostEncodedFrame>, BackendError>;
}

/// One frame's encode, which the session's encode thread finishes.
pub(crate) trait HostEncodedFrame: Send {
    /// Encodes the frame from `source`'s planes, and writes its access unit, NAL units in
    /// Annex-B form, its keyframe's parameter sets before its slices, at the front of `out`
    /// where it fits there; where it does not, writes nothing of it. Answers what came of it;
    /// `Err` where the hardware did not encode it, which the session logs.
    fn finish(
        self: Box<Self>,
        source: &EncodeSource,
        out: &mut [u8],
    ) -> Result<Encoded, BackendError>;
}

/// The source of one frame, as host bytes, which [`HostEncodedFrame::finish`] reads.
#[derive(Clone, Copy, Debug)]
pub(crate) struct EncodeSource {
    /// The luma plane: `height` rows of `width` samples.
    pub(crate) luma: SourcePlane,
    /// The chroma plane: half as many rows of `width / 2` pairs of samples.
    pub(crate) chroma: SourcePlane,
    /// The frame's width, the settings'.
    pub(crate) width: u32,
    /// The frame's height, the settings'.
    pub(crate) height: u32,
}

/// What came of one frame's encode ([`HostEncodedFrame::finish`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Encoded {
    /// The access unit's bytes: written at the front of `out`, or, where `overflow`, the bytes
    /// it needed.
    pub(crate) size: u64,
    /// The access unit did not fit, and nothing of it was written; the encoder's next one is a
    /// keyframe.
    pub(crate) overflow: bool,
    /// The access unit is a keyframe.
    pub(crate) keyframe: bool,
    /// It carries the parameter sets.
    pub(crate) parameter_sets: bool,
    /// The frame it holds, by its encode's `frame_index`.
    pub(crate) frame_index: u64,
    /// `RGFX_VIDEO_PICTURE_*`.
    pub(crate) picture_type: u32,
}

impl Encoded {
    /// The result the guest reads.
    fn result(self) -> RgfxVideoEncodeResult {
        let mut flags = 0;
        if self.keyframe {
            flags |= RGFX_VIDEO_ENCODED_KEYFRAME;
        }
        if self.parameter_sets {
            flags |= RGFX_VIDEO_ENCODED_PARAMETER_SETS;
        }
        RgfxVideoEncodeResult {
            status: if self.overflow {
                RGFX_VIDEO_ENCODE_OVERFLOW
            } else {
                RGFX_VIDEO_ENCODE_OK
            },
            flags,
            bitstream_size: self.size,
            frame_index: self.frame_index,
            picture_type: self.picture_type,
            reserved_1c: 0,
        }
    }
}

/// A command-buffer fault for an encode the session could not take.
fn encode_fault(what: impl std::fmt::Display) -> Raise {
    op_fault(format_args!("a video encode: {what}"))
}

/// Checks `settings` of a session of `codec` as every backend takes them: an even size of at
/// least one sample pair, one profile of the codec, the format its bit depth has, a rate control
/// the wire names, and an entropy coder the profile has.
pub(crate) fn check_settings(codec: u32, settings: &RgfxVideoEncodeSettings) -> Result<(), String> {
    let s = settings;
    if s.width == 0 || s.height == 0 || s.width % 2 != 0 || s.height % 2 != 0 {
        return Err(format!(
            "a 4:2:0 frame of {}x{}: not an even size",
            s.width, s.height
        ));
    }
    let format = match (codec, s.profile) {
        (RGFX_VIDEO_CODEC_H264, RGFX_VIDEO_H264_BASELINE | RGFX_VIDEO_H264_MAIN)
        | (RGFX_VIDEO_CODEC_H264, RGFX_VIDEO_H264_HIGH)
        | (RGFX_VIDEO_CODEC_HEVC, RGFX_VIDEO_HEVC_MAIN) => RGFX_VIDEO_FORMAT_NV12,
        (RGFX_VIDEO_CODEC_HEVC, RGFX_VIDEO_HEVC_MAIN_10) => RGFX_VIDEO_FORMAT_P010,
        (codec, profile) => return Err(format!("profile {profile:#x} of codec {codec}")),
    };
    if s.format != format {
        return Err(format!(
            "a source of format {:#x} for profile {:#x}",
            s.format, s.profile
        ));
    }
    if !matches!(
        s.rate_control,
        RGFX_VIDEO_RATE_CQP | RGFX_VIDEO_RATE_CBR | RGFX_VIDEO_RATE_VBR
    ) {
        return Err(format!("rate control {}", s.rate_control));
    }
    let entropy_taken = match (codec, s.entropy) {
        // The profile's own.
        (_, 0) => true,
        (RGFX_VIDEO_CODEC_H264, RGFX_VIDEO_ENTROPY_CAVLC) => true,
        (RGFX_VIDEO_CODEC_H264, RGFX_VIDEO_ENTROPY_CABAC) => s.profile != RGFX_VIDEO_H264_BASELINE,
        _ => false,
    };
    if !entropy_taken {
        return Err(format!(
            "entropy {} for profile {:#x} of codec {codec}",
            s.entropy, s.profile
        ));
    }
    Ok(())
}

/// The bytes a sample of `settings`' format takes.
fn bytes_per_sample(settings: &RgfxVideoEncodeSettings) -> u64 {
    if settings.format == RGFX_VIDEO_FORMAT_P010 {
        2
    } else {
        1
    }
}

/// An encode's source: its planes, held until its value is signalled.
struct Source<P: Platform> {
    luma: OutputPlane<P>,
    chroma: OutputPlane<P>,
    width: u32,
    height: u32,
}

impl<P: Platform> Source<P> {
    /// The source of `encode` at `settings`' size: each plane's rows of the frame, from its
    /// first; a fault where a plane holds fewer rows or narrower ones.
    fn of(encode: &VideoEncode<'_, P>, settings: &RgfxVideoEncodeSettings) -> RaiseResult<Self> {
        let sample = bytes_per_sample(settings);
        let (width, height) = (u64::from(settings.width), u64::from(settings.height));
        let row = width * sample;
        let luma = output_plane_of(&encode.luma, height, row).map_err(encode_fault)?;
        let chroma = output_plane_of(&encode.chroma, height / 2, row).map_err(encode_fault)?;
        Ok(Self {
            luma,
            chroma,
            width: settings.width,
            height: settings.height,
        })
    }
}

/// Where an encode's access unit and result go.
struct Out<P: Platform> {
    _bitstream: P::Buffer,
    at: HostPtr,
    capacity: usize,
    _result: P::Buffer,
    result_at: HostPtr,
}

impl<P: Platform> Out<P> {
    /// The access unit's bytes and the result's of `encode`, which it holds; a fault where either
    /// buffer has no host bytes, or the two overlap.
    fn of(encode: &VideoEncode<'_, P>) -> RaiseResult<Self> {
        let host = |buffer: &P::Buffer, what| {
            let contents = buffer.contents();
            if contents.is_null() {
                return Err(encode_fault(format_args!(
                    "{what} in a buffer with no host bytes"
                )));
            }
            Ok(contents)
        };
        let bitstream = host(encode.bitstream, "an access unit")?;
        let result = host(encode.result, "a result")?;
        let capacity = usize::try_from(encode.bitstream_capacity)
            .map_err(|_| encode_fault("a capacity past the host's address space"))?;
        let at = bitstream.wrapping_add(encode.bitstream_offset);
        let result_at = result.wrapping_add(encode.result_offset);
        let span = |p: HostPtr, len: u64| {
            let first = u64::try_from(p.addr()).unwrap_or(u64::MAX);
            (first, first.saturating_add(len))
        };
        let au = span(at, encode.bitstream_capacity);
        let res = span(result_at, size_of::<RgfxVideoEncodeResult>() as u64);
        if au.0 < res.1 && res.0 < au.1 {
            return Err(encode_fault("a result inside the access unit's capacity"));
        }
        Ok(Self {
            _bitstream: encode.bitstream.clone_ref(),
            at,
            capacity,
            _result: encode.result.clone_ref(),
            result_at,
        })
    }
}

/// One encode's work for the encode thread.
struct Job<P: Platform> {
    value: u64,
    frame: Box<dyn HostEncodedFrame>,
    source: Source<P>,
    out: Out<P>,
}

/// What became of a frame, which the session counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Outcome {
    Written,
    Overflowed,
    Failed,
}

impl Outcome {
    const ALL: [Outcome; 3] = [Outcome::Written, Outcome::Overflowed, Outcome::Failed];

    fn says(self) -> &'static str {
        match self {
            Outcome::Written => "written",
            Outcome::Overflowed => "past its capacity",
            Outcome::Failed => "failed",
        }
    }
}

/// A session's counts of its frames, by [`Outcome`] (its index in [`Outcome::ALL`]), and of
/// those whose source the GPU copied out of a device cache first.
#[derive(Debug, Default)]
struct Counts {
    frames: [u64; 3],
    staged: u64,
}

/// An encode session on a host API ([`HostEncoder`]): its frames finished by its encode
/// thread, which signals its shared event from the host.
pub(crate) struct HostEncodeSession<P: Platform> {
    codec: u32,
    /// The backend's name, which the log lines say.
    name: &'static str,
    state: Mutex<State<P>>,
    event: P::SharedEvent,
    counts: Arc<Mutex<Counts>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

/// A session's encoder, the encode thread's queue, the last value an encode answered, and the
/// settings in force.
struct State<P: Platform> {
    encoder: Box<dyn HostEncoder>,
    jobs: Option<mpsc::SyncSender<Job<P>>>,
    value: u64,
    settings: Option<RgfxVideoEncodeSettings>,
}

/// The frames a session holds unfinished, queued for the encode thread, beside the one it
/// finishes: each holds its source and its buffers until then.
const QUEUED: usize = 2;

impl<P: Platform> HostEncodeSession<P> {
    /// A session of `codec` (`RGFX_VIDEO_CODEC_*`) on `encoder`, the backend `name`'s, its event
    /// and its copies out of device caches on `device`; `None`, logged, where the device makes no
    /// shared event or no thread starts.
    pub(crate) fn new(
        device: &P::Device,
        codec: u32,
        name: &'static str,
        encoder: Box<dyn HostEncoder>,
    ) -> Option<Self> {
        let Some(event) = device.new_shared_event() else {
            log::warn!("hardware video: {name}: no shared event for an encode session");
            return None;
        };
        let (jobs, taken) = mpsc::sync_channel(QUEUED);
        let counts = Arc::new(Mutex::new(Counts::default()));
        let stage = Stage {
            device: device.clone_ref(),
            queue: None,
            buffer: None,
        };
        let thread = {
            let (event, counts) = (event.clone_ref(), Arc::clone(&counts));
            std::thread::Builder::new()
                .name("reims-vgpu.graphics-video-encode".into())
                .spawn(move || finish_frames(&taken, stage, &event, &counts, name))
        };
        let thread = match thread {
            Ok(thread) => thread,
            Err(error) => {
                log::warn!("hardware video: {name}: no encode thread for a session ({error})");
                return None;
            }
        };
        Some(Self {
            codec,
            name,
            state: Mutex::new(State {
                encoder,
                jobs: Some(jobs),
                value: 0,
                settings: None,
            }),
            event,
            counts,
            thread: Some(thread),
        })
    }
}

impl<P: Platform> GraphicsVideoSession<P> for HostEncodeSession<P> {
    fn event(&self) -> EventRef<'_, P> {
        EventRef::Shared(&self.event)
    }

    fn codec(&self) -> u32 {
        self.codec
    }

    fn decode(&self, _decode: &VideoDecode<'_, P>) -> RaiseResult<Decoded> {
        Err(op_fault("a video decode on a session that encodes"))
    }

    fn encode(&self, encode: &VideoEncode<'_, P>) -> RaiseResult<u64> {
        let mut state = self.state.lock();
        if let Some(settings) = &encode.settings {
            check_settings(self.codec, settings)
                .map_err(|why| encode_fault(format_args!("its settings: {why}")))?;
            (state.encoder.check(settings))
                .map_err(|why| encode_fault(format_args!("its settings: {why}")))?;
        }
        let Some(settings) = encode.settings.or(state.settings) else {
            return Err(encode_fault("a first encode with no settings"));
        };
        let source = Source::of(encode, &settings)?;
        let out = Out::of(encode)?;
        let frame = EncodeFrame {
            settings: encode.settings,
            frame_index: encode.frame_index,
            force_keyframe: encode.force_keyframe,
        };
        let frame = state.encoder.encode(frame).map_err(encode_fault)?;
        let value = state.value + 1;
        let job = Job {
            value,
            frame,
            source,
            out,
        };
        let sent = (state.jobs.as_ref()).is_some_and(|jobs| jobs.send(job).is_ok());
        if !sent {
            return Err(encode_fault("the session's encode thread is gone"));
        }
        state.value = value;
        state.settings = Some(settings);
        Ok(value)
    }
}

impl<P: Platform> Drop for HostEncodeSession<P> {
    fn drop(&mut self) {
        let last = {
            let mut state = self.state.lock();
            drop(state.jobs.take());
            state.value
        };
        let millis = u64::try_from(TEARDOWN_WAIT.as_millis()).unwrap_or(u64::MAX);
        let done = last == 0 || self.event.wait_until_signaled_value(last, millis);
        let name = self.name;
        match self.thread.take() {
            Some(thread) if done => {
                if thread.join().is_err() {
                    log::error!("hardware video: {name}: an encode session's thread panicked");
                }
            }
            Some(_) => log::error!(
                "hardware video: {name}: an encode session's last frame was not done within \
                 {TEARDOWN_WAIT:?}"
            ),
            None => {}
        }
        let counts = self.counts.lock();
        let frames = (Outcome::ALL.iter().zip(counts.frames))
            .filter(|(_, n)| *n > 0)
            .map(|(outcome, n)| format!("{n} {}", outcome.says()))
            .collect::<Vec<_>>();
        log::info!(
            "hardware video: {name}: an encode session ended after {last} encodes: {}; {} read \
             from a device cache",
            if frames.is_empty() {
                "none".to_owned()
            } else {
                frames.join(", ")
            },
            counts.staged
        );
    }
}

/// The encode thread: finishes each encode, in the order taken, writes its result, and signals
/// its value.
fn finish_frames<P: Platform>(
    jobs: &mpsc::Receiver<Job<P>>,
    mut stage: Stage<P>,
    event: &P::SharedEvent,
    counts: &Mutex<Counts>,
    name: &'static str,
) {
    for job in jobs {
        // A step of the exit gate's, as a picture's landing is (`land_pictures`).
        let Some(_step) = reims_vgpu_vulkan::exit_gate::step() else {
            reims_vgpu_vulkan::exit_gate::stay()
        };
        let Job {
            value,
            frame,
            source,
            out,
        } = job;
        let (outcome, result, staged) = match finish(frame, &source, &out, &mut stage) {
            Ok((encoded, staged)) => {
                let outcome = if encoded.overflow {
                    Outcome::Overflowed
                } else {
                    Outcome::Written
                };
                (outcome, encoded.result(), staged)
            }
            Err(error) => {
                let failed = counts.lock().frames[2];
                if failed < FAILURES_LOGGED {
                    log::error!(
                        kind:% = error;
                        "hardware video: {name}: a frame was not encoded: {error}"
                    );
                }
                let result = RgfxVideoEncodeResult {
                    status: RGFX_VIDEO_ENCODE_FAILED,
                    ..RgfxVideoEncodeResult::default()
                };
                (Outcome::Failed, result, false)
            }
        };
        // SAFETY: the result's 32 bytes from `result_at` lie in its Shared buffer's host bytes
        // (the op's check), which `out` holds, apart from the access unit's (the session's).
        unsafe {
            std::ptr::copy_nonoverlapping(
                result.as_bytes().as_ptr(),
                out.result_at.as_ptr::<u8>(),
                size_of::<RgfxVideoEncodeResult>(),
            );
        }
        {
            let mut counts = counts.lock();
            if let Some(i) = Outcome::ALL.iter().position(|o| *o == outcome) {
                // A session's first of each outcome; once per process at info (`kind`).
                if counts.frames[i] == 0 && outcome != Outcome::Failed {
                    log::info!(
                        kind = outcome.says();
                        "hardware video: {name}: a frame encoded, {}",
                        outcome.says()
                    );
                }
                counts.frames[i] += 1;
            }
            counts.staged += u64::from(staged);
        }
        // The holds go, and the brackets end, before the guest may read them.
        drop((source, out));
        if let Err(raise) = event.set_signaled_value(value) {
            log::error!(
                kind:% = raise;
                "hardware video: {name}: the session's signal of {value}: {raise}"
            );
        }
    }
}

/// Finishes `frame` from `source` into `out`, copying a source plane out of its device cache
/// first, and answers what came of it and whether a plane was copied so.
fn finish<P: Platform>(
    frame: Box<dyn HostEncodedFrame>,
    source: &Source<P>,
    out: &Out<P>,
    stage: &mut Stage<P>,
) -> Result<(Encoded, bool), BackendError> {
    let staged = stage.copy_out_of_caches(&source.luma, &source.chroma)?;
    let plane = |p: &OutputPlane<P>, at: Option<u64>| -> Result<SourcePlane, BackendError> {
        let first = match (at, p.host) {
            (Some(at), _) => {
                let buffer = stage.buffer.as_ref().ok_or("no staging buffer")?;
                buffer.contents().wrapping_add(at)
            }
            (None, Some(host)) => host,
            (None, None) => return Err("a source plane with no host bytes".into()),
        };
        let (Ok(bytes_per_row), Ok(row_bytes), Ok(rows)) = (
            usize::try_from(p.bytes_per_row),
            usize::try_from(p.row_bytes),
            usize::try_from(p.height),
        ) else {
            return Err("a source plane past the host's address space".into());
        };
        Ok(SourcePlane {
            at: first.as_ptr::<u8>().cast_const(),
            bytes_per_row,
            row_bytes,
            rows,
        })
    };
    let source = EncodeSource {
        luma: plane(&source.luma, staged.luma)?,
        chroma: plane(&source.chroma, staged.chroma)?,
        width: source.width,
        height: source.height,
    };
    // SAFETY: the capacity's bytes from `at` lie in the bitstream's Shared buffer's host bytes
    // (the op's check), which `out` holds until the signal, and nothing else reads or writes them
    // until then: the guest's command buffer waits for the signal, and the result lies apart.
    let bytes = unsafe { std::slice::from_raw_parts_mut(out.at.as_ptr::<u8>(), out.capacity) };
    let encoded = frame.finish(&source, bytes)?;
    let staged = staged.luma.is_some() || staged.chroma.is_some();
    Ok((encoded, staged))
}

/// The bytes a source plane takes staged out of its device cache: its rows at its pitch.
fn staged_len<P: Platform>(p: &OutputPlane<P>) -> u64 {
    plane_end(0, p.bytes_per_row, p.height, p.row_bytes).unwrap_or(0)
}

impl<P: Platform> Stage<P> {
    /// Copies each of `luma` and `chroma` that lies in a device cache into the staging buffer
    /// by the GPU, at its own pitch, and waits for the copy; answers where each such plane is
    /// staged.
    fn copy_out_of_caches(
        &mut self,
        luma: &OutputPlane<P>,
        chroma: &OutputPlane<P>,
    ) -> Result<Staged, BackendError> {
        if !luma.cached && !chroma.cached {
            return Ok(Staged::default());
        }
        let luma_len = luma.cached.then(|| staged_len(luma));
        let chroma_len = chroma.cached.then(|| staged_len(chroma));
        let chroma_at = (luma_len.unwrap_or(0))
            .checked_next_multiple_of(super::STAGING_ALIGNMENT)
            .ok_or("a staged plane past 2^64 bytes")?;
        let total = match chroma_len {
            Some(len) => chroma_at
                .checked_add(len)
                .ok_or("a staged plane past 2^64 bytes")?,
            None => luma_len.unwrap_or(0),
        };
        self.buffer_of(total)?;
        let staged = Staged {
            luma: luma_len.map(|_| 0),
            chroma: chroma_len.map(|_| chroma_at),
        };
        self.copy_with_caches(luma, chroma, staged, false)?;
        Ok(staged)
    }
}

#[cfg(test)]
mod tests;
