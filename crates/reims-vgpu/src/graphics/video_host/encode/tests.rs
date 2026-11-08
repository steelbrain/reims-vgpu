use std::collections::VecDeque;

use reims_vgpu_core::raise::OnFault;
use reims_vgpu_core::testkit::FakePlatform;
use reims_vgpu_core::testkit::calllog::Arg;
use reims_vgpu_core::testkit::objects::CompleteOn;
use reims_vgpu_graphics::ops::video::{RGFX_VIDEO_PICTURE_I, RGFX_VIDEO_PICTURE_P};
use reims_vgpu_graphics::wire::RGFX_VIDEO_CODEC_H264;
use zerocopy::FromBytes;

use super::super::super::decode::VideoPlane;
use super::*;

type P = FakePlatform;
type FakeBuffer = <P as Platform>::Buffer;

/// `MTLResourceStorageModeShared` and `MTLResourceStorageModePrivate`.
const SHARED: u64 = 0;
const PRIVATE: u64 = 2 << 4;
/// What a buffer's bytes hold before the host writes them.
const UNTOUCHED: u8 = 0xee;
const COPY: &str = "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:";

fn buffer(p: &FakePlatform, len: u64, options: u64) -> FakeBuffer {
    let b = (p
        .device()
        .new_buffer_of_length(OnFault::Return, len, options))
    .unwrap()
    .unwrap();
    if options == SHARED {
        // SAFETY: the fake's Shared buffer holds `len` host bytes.
        unsafe { std::ptr::write_bytes(b.contents().as_ptr::<u8>(), UNTOUCHED, len as usize) };
    }
    b
}

/// The bytes of `b`, a Shared buffer of the fake's.
fn bytes(b: &FakeBuffer) -> Vec<u8> {
    // SAFETY: the fake's Shared buffer holds `length` host bytes.
    unsafe { std::slice::from_raw_parts(b.contents().as_ptr::<u8>(), b.length() as usize) }.to_vec()
}

/// Settings of a 64x32 H.264 High stream at a constant QP.
fn settings() -> RgfxVideoEncodeSettings {
    RgfxVideoEncodeSettings {
        width: 64,
        height: 32,
        format: RGFX_VIDEO_FORMAT_NV12,
        profile: RGFX_VIDEO_H264_HIGH,
        rate_control: RGFX_VIDEO_RATE_CQP,
        qp_i: 26,
        qp_p: 28,
        ..RgfxVideoEncodeSettings::default()
    }
}

/// What the encoder and its frames saw, in order.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Seen {
    /// A frame taken: its index, whether it carried settings, and whether it was forced a
    /// keyframe.
    Took(u64, bool, bool),
    /// A frame finished: its index, its source's first luma byte and chroma byte, and the
    /// capacity it had.
    Finished(u64, u8, u8, usize),
}

type Log = Arc<Mutex<Vec<Seen>>>;

/// How a frame of the test's finishes.
#[derive(Clone, Copy, Debug)]
enum Answer {
    /// An access unit of this many bytes, each the frame's index.
    Unit(usize),
    /// A failure.
    Fail,
}

/// An encoder that logs what it is handed and answers `answers` in turn: an access unit of 10
/// bytes once they run out.
struct Encoder {
    log: Log,
    answers: VecDeque<Answer>,
}

impl HostEncoder for Encoder {
    fn check(&self, settings: &RgfxVideoEncodeSettings) -> Result<(), BackendError> {
        if settings.width > 4096 {
            return Err("past the test's largest".into());
        }
        Ok(())
    }

    fn encode(&mut self, frame: EncodeFrame) -> Result<Box<dyn HostEncodedFrame>, BackendError> {
        self.log.lock().push(Seen::Took(
            frame.frame_index,
            frame.settings.is_some(),
            frame.force_keyframe,
        ));
        let answer = self.answers.pop_front().unwrap_or(Answer::Unit(10));
        Ok(Box::new(Frame {
            log: Arc::clone(&self.log),
            index: frame.frame_index,
            answer,
        }))
    }
}

struct Frame {
    log: Log,
    index: u64,
    answer: Answer,
}

impl HostEncodedFrame for Frame {
    fn finish(
        self: Box<Self>,
        source: &EncodeSource,
        out: &mut [u8],
    ) -> Result<Encoded, BackendError> {
        // SAFETY: each plane's first row lies in the source's host bytes.
        let (luma, chroma) = unsafe { (*source.luma.at, *source.chroma.at) };
        (self.log.lock()).push(Seen::Finished(self.index, luma, chroma, out.len()));
        let Answer::Unit(len) = self.answer else {
            return Err("the hardware failed".into());
        };
        let fits = len <= out.len();
        if fits {
            out[..len].fill(self.index as u8);
        }
        Ok(Encoded {
            size: len as u64,
            overflow: !fits,
            keyframe: self.index == 0,
            parameter_sets: self.index == 0,
            frame_index: self.index,
            picture_type: if self.index == 0 {
                RGFX_VIDEO_PICTURE_I
            } else {
                RGFX_VIDEO_PICTURE_P
            },
        })
    }
}

fn session(p: &FakePlatform, answers: Vec<Answer>, log: &Log) -> HostEncodeSession<P> {
    let encoder = Encoder {
        log: Arc::clone(log),
        answers: answers.into(),
    };
    HostEncodeSession::new(
        &p.device(),
        RGFX_VIDEO_CODEC_H264,
        "test",
        Box::new(encoder),
    )
    .unwrap()
}

/// A source of `width` by `height` in `source`: its luma plane at 0, its chroma plane at 8192,
/// rows 128 apart.
fn planes(source: &FakeBuffer, width: u64, height: u64) -> (VideoPlane<'_, P>, VideoPlane<'_, P>) {
    let plane = |offset, height| VideoPlane {
        buffer: source,
        offset,
        bytes_per_row: 128,
        row_bytes: width,
        width,
        height,
        cached: source.contents().is_null(),
        pages: None,
        cache: None,
    };
    (plane(0, height), plane(8192, height / 2))
}

/// An encode of frame `index` from `source` into `out`: the access unit at 256, 512 bytes at
/// most, the result at 1024.
fn encode<'a>(
    source: &'a FakeBuffer,
    out: &'a FakeBuffer,
    index: u64,
    settings: Option<RgfxVideoEncodeSettings>,
) -> VideoEncode<'a, P> {
    let (luma, chroma) = planes(source, 64, 32);
    VideoEncode {
        luma,
        chroma,
        bitstream: out,
        bitstream_offset: 256,
        bitstream_capacity: 512,
        result: out,
        result_offset: 1024,
        frame_index: index,
        force_keyframe: false,
        settings,
    }
}

fn event_of(s: &HostEncodeSession<P>) -> &<P as Platform>::SharedEvent {
    match s.event() {
        EventRef::Shared(event) => event,
        EventRef::Event(_) => panic!("a plain event"),
    }
}

fn result_in(b: &[u8]) -> RgfxVideoEncodeResult {
    RgfxVideoEncodeResult::read_from_bytes(&b[1024..1024 + 32]).unwrap()
}

/// A source in pages, its luma rows 0x11 and its chroma rows 0x22.
fn source(p: &FakePlatform) -> FakeBuffer {
    let b = buffer(p, 16384, SHARED);
    // SAFETY: the fake's Shared buffer holds 16384 host bytes.
    unsafe {
        std::ptr::write_bytes(b.contents().as_ptr::<u8>(), 0x11, 8192);
        std::ptr::write_bytes(b.contents().as_ptr::<u8>().add(8192), 0x22, 8192);
    }
    b
}

/// Each encode answers the next value, signalled once its access unit is in the bitstream's
/// buffer at its offset and its result after it; the frame reads its source in place, and the
/// settings reach the encoder with the frame that carries them.
#[test]
fn an_encode_writes_its_access_unit_then_its_result_then_signals() {
    let p = FakePlatform::new();
    let log = Log::default();
    let s = session(&p, vec![Answer::Unit(20)], &log);
    let src = source(&p);
    let (first, second) = (buffer(&p, 4096, SHARED), buffer(&p, 4096, SHARED));
    assert_eq!(
        s.encode(&encode(&src, &first, 0, Some(settings())))
            .unwrap(),
        1
    );
    assert_eq!(s.encode(&encode(&src, &second, 1, None)).unwrap(), 2);
    assert!(event_of(&s).wait_until_signaled_value(2, 10_000));
    let b = bytes(&first);
    assert!(b[..256].iter().all(|&x| x == UNTOUCHED));
    assert!(b[256..276].iter().all(|&x| x == 0));
    assert!(b[276..1024].iter().all(|&x| x == UNTOUCHED));
    let result = result_in(&b);
    assert_eq!(result.status, RGFX_VIDEO_ENCODE_OK);
    assert_eq!(result.bitstream_size, 20);
    assert_eq!(
        result.flags,
        RGFX_VIDEO_ENCODED_KEYFRAME | RGFX_VIDEO_ENCODED_PARAMETER_SETS
    );
    assert_eq!(
        (result.frame_index, result.picture_type),
        (0, RGFX_VIDEO_PICTURE_I)
    );
    let b = bytes(&second);
    assert!(b[256..266].iter().all(|&x| x == 1));
    let result = result_in(&b);
    assert_eq!((result.bitstream_size, result.flags), (10, 0));
    assert_eq!(
        (result.frame_index, result.picture_type),
        (1, RGFX_VIDEO_PICTURE_P)
    );
    // The frames finish on the encode thread, beside the next encode: each kind is in order.
    let (took, finished): (Vec<Seen>, Vec<Seen>) =
        (log.lock().iter().cloned()).partition(|seen| matches!(seen, Seen::Took(..)));
    assert_eq!(
        took,
        [Seen::Took(0, true, false), Seen::Took(1, false, false)]
    );
    assert_eq!(
        finished,
        [
            Seen::Finished(0, 0x11, 0x22, 512),
            Seen::Finished(1, 0x11, 0x22, 512),
        ]
    );
    drop(s);
}

/// An access unit past the capacity writes none of its bytes, and its result says the bytes it
/// needed; a frame the encoder fails gets a failed result. Both are signalled.
#[test]
fn an_overflow_and_a_failure_are_results_and_are_signalled() {
    let p = FakePlatform::new();
    let log = Log::default();
    let s = session(&p, vec![Answer::Unit(600), Answer::Fail], &log);
    let src = source(&p);
    let (first, second) = (buffer(&p, 4096, SHARED), buffer(&p, 4096, SHARED));
    assert_eq!(
        s.encode(&encode(&src, &first, 0, Some(settings())))
            .unwrap(),
        1
    );
    assert_eq!(s.encode(&encode(&src, &second, 1, None)).unwrap(), 2);
    assert!(event_of(&s).wait_until_signaled_value(2, 10_000));
    let b = bytes(&first);
    assert!(b[..1024].iter().all(|&x| x == UNTOUCHED));
    let result = result_in(&b);
    assert_eq!(
        (result.status, result.bitstream_size),
        (RGFX_VIDEO_ENCODE_OVERFLOW, 600)
    );
    let result = result_in(&bytes(&second));
    assert_eq!(
        result,
        RgfxVideoEncodeResult {
            status: RGFX_VIDEO_ENCODE_FAILED,
            ..RgfxVideoEncodeResult::default()
        }
    );
    drop(s);
}

/// What faults an encode, and spends no value: a first encode with no settings, settings the
/// session or the encoder does not take, a source smaller than the settings' size, and a result
/// inside the access unit's capacity.
#[test]
fn an_encode_the_session_cannot_take_faults_and_spends_no_value() {
    let p = FakePlatform::new();
    let log = Log::default();
    let s = session(&p, vec![], &log);
    let src = source(&p);
    let out = buffer(&p, 4096, SHARED);
    assert!(s.encode(&encode(&src, &out, 0, None)).is_err());
    let odd = RgfxVideoEncodeSettings {
        width: 63,
        ..settings()
    };
    assert!(s.encode(&encode(&src, &out, 0, Some(odd))).is_err());
    let cabac_baseline = RgfxVideoEncodeSettings {
        profile: RGFX_VIDEO_H264_BASELINE,
        entropy: RGFX_VIDEO_ENTROPY_CABAC,
        ..settings()
    };
    assert!(
        s.encode(&encode(&src, &out, 0, Some(cabac_baseline)))
            .is_err()
    );
    let ten_bits = RgfxVideoEncodeSettings {
        format: RGFX_VIDEO_FORMAT_P010,
        ..settings()
    };
    assert!(s.encode(&encode(&src, &out, 0, Some(ten_bits))).is_err());
    let wide = RgfxVideoEncodeSettings {
        width: 8192,
        ..settings()
    };
    assert!(s.encode(&encode(&src, &out, 0, Some(wide))).is_err());
    let taller = RgfxVideoEncodeSettings {
        height: 40,
        ..settings()
    };
    assert!(s.encode(&encode(&src, &out, 0, Some(taller))).is_err());
    let mut inside = encode(&src, &out, 0, Some(settings()));
    inside.result_offset = 512;
    assert!(s.encode(&inside).is_err());
    assert!(log.lock().is_empty());
    assert_eq!(
        s.encode(&encode(&src, &out, 0, Some(settings()))).unwrap(),
        1
    );
    drop(s);
}

/// A source plane in a device cache is copied by the GPU into the staging buffer, its rows at
/// their pitch, before the frame reads it there; a plane in pages beside it is read in place.
#[test]
fn a_source_in_a_device_cache_is_copied_out_of_it_first() {
    let p = FakePlatform::new();
    p.world().set_auto_complete(Some(CompleteOn::Caller));
    let log = Log::default();
    let s = session(&p, vec![], &log);
    let cache = buffer(&p, 16384, PRIVATE);
    let pages = source(&p);
    let out = buffer(&p, 4096, SHARED);
    let (luma, _) = planes(&cache, 64, 32);
    let (_, chroma) = planes(&pages, 64, 32);
    let mut e = encode(&pages, &out, 0, Some(settings()));
    (e.luma, e.chroma) = (luma, chroma);
    assert_eq!(s.encode(&e).unwrap(), 1);
    assert!(event_of(&s).wait_until_signaled_value(1, 10_000));
    let copies = p.call_log().calls_to(COPY);
    // One a row: 64 bytes of each 128.
    assert_eq!(copies.len(), 32);
    let from = Arg::obj(&cache);
    assert_eq!(copies[0].args[0], from);
    assert_eq!(
        (copies[0].args[1].clone(), copies[0].args[4].clone()),
        (Arg::U64(0), Arg::U64(64))
    );
    assert_eq!(copies[31].args[1], Arg::U64(31 * 128));
    assert_eq!(result_in(&bytes(&out)).status, RGFX_VIDEO_ENCODE_OK);
    assert_eq!(s.counts.lock().staged, 1);
    let finished = log.lock().last().cloned();
    assert!(matches!(finished, Some(Seen::Finished(0, _, 0x22, 512))));
    drop(s);
}

/// The settings every backend takes: each H.264 and HEVC profile with its format, and H.264's
/// entropy coders where the profile has them.
#[test]
fn the_settings_every_backend_takes() {
    let h264 = |profile, entropy| RgfxVideoEncodeSettings {
        profile,
        entropy,
        ..settings()
    };
    assert!(check_settings(RGFX_VIDEO_CODEC_H264, &settings()).is_ok());
    assert!(check_settings(RGFX_VIDEO_CODEC_H264, &h264(RGFX_VIDEO_H264_BASELINE, 0)).is_ok());
    let cavlc = h264(RGFX_VIDEO_H264_BASELINE, RGFX_VIDEO_ENTROPY_CAVLC);
    assert!(check_settings(RGFX_VIDEO_CODEC_H264, &cavlc).is_ok());
    let cabac = h264(RGFX_VIDEO_H264_MAIN, RGFX_VIDEO_ENTROPY_CABAC);
    assert!(check_settings(RGFX_VIDEO_CODEC_H264, &cabac).is_ok());
    assert!(check_settings(RGFX_VIDEO_CODEC_H264, &h264(RGFX_VIDEO_H264_HIGH, 3)).is_err());
    let main10 = RgfxVideoEncodeSettings {
        profile: RGFX_VIDEO_HEVC_MAIN_10,
        format: RGFX_VIDEO_FORMAT_P010,
        ..settings()
    };
    assert!(check_settings(RGFX_VIDEO_CODEC_HEVC, &main10).is_ok());
    let main_p010 = RgfxVideoEncodeSettings {
        profile: RGFX_VIDEO_HEVC_MAIN,
        ..main10
    };
    assert!(check_settings(RGFX_VIDEO_CODEC_HEVC, &main_p010).is_err());
    let hevc_cavlc = RgfxVideoEncodeSettings {
        entropy: RGFX_VIDEO_ENTROPY_CAVLC,
        ..main10
    };
    assert!(check_settings(RGFX_VIDEO_CODEC_HEVC, &hevc_cavlc).is_err());
    assert!(check_settings(RGFX_VIDEO_CODEC_HEVC, &settings()).is_err());
    for rate in [0, 4] {
        let s = RgfxVideoEncodeSettings {
            rate_control: rate,
            ..settings()
        };
        assert!(check_settings(RGFX_VIDEO_CODEC_H264, &s).is_err());
    }
}
