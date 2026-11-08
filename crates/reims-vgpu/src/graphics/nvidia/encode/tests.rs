use reims_graphics::ops::video::{
    RGFX_VIDEO_ENCODE_OK, RGFX_VIDEO_ENCODE_OVERFLOW, RGFX_VIDEO_ENCODED_KEYFRAME,
    RGFX_VIDEO_ENCODED_PARAMETER_SETS, RgfxVideoEncodeResult,
};
use reims_graphics::wire::RGFX_VIDEO_DECODE;
use reims_vgpu_core::platform::Platform;
use reims_vgpu_core::platform::device::GpuDevice as _;
use reims_vgpu_core::platform::event::{EventRef, SharedEvent as _};
use reims_vgpu_core::platform::resource::Buffer as _;
use reims_vgpu_core::raise::OnFault;
use reims_vgpu_core::testkit::FakePlatform;
use reims_vgpu_video::core::runtime::{Runtime, StdRuntime};
use reims_vgpu_video::nvidia::{AnnexBCodec, AnnexBDecoder, GpuSelector, encode_profiles};
use zerocopy::FromBytes;

use super::super::super::decode::{GraphicsVideoSession, VideoEncode, VideoPlane};
use super::super::super::video_host::HostEncodeSession;
use super::*;

/// Settings of a `width` by `height` stream of `profile` of `codec` at a constant QP.
fn settings(codec: u32, profile: u32, width: u32, height: u32) -> RgfxVideoEncodeSettings {
    RgfxVideoEncodeSettings {
        width,
        height,
        format: if profile == RGFX_VIDEO_HEVC_MAIN_10 && codec == RGFX_VIDEO_CODEC_HEVC {
            RGFX_VIDEO_FORMAT_P010
        } else {
            RGFX_VIDEO_FORMAT_NV12
        },
        profile,
        rate_control: RGFX_VIDEO_RATE_CQP,
        qp_i: 22,
        qp_p: 24,
        qp_b: 26,
        frame_rate_num: 30,
        frame_rate_den: 1,
        ..RgfxVideoEncodeSettings::default()
    }
}

/// NVENC's answer for `profile`: `min` to `max` pictures, CBR and VBR as given, and several
/// references.
fn support(profile: EncodeProfile, min: u32, max: u32, cbr: bool) -> EncodeProfileSupport {
    EncodeProfileSupport {
        profile,
        min_width: min,
        min_height: min,
        max_width: max,
        max_height: max,
        max_b_frames: 4,
        constant_bit_rate: cbr,
        variable_bit_rate: true,
        max_references: 7,
    }
}

/// An entry for each codec: its profiles, its formats, the least of their sizes and
/// references, and the codec's top level.
#[test]
fn an_entry_for_each_codec_nvenc_encodes() {
    let mut main10 = support(EncodeProfile::HevcMain10, 129, 8192, true);
    main10.max_references = 1;
    let supports = [
        support(EncodeProfile::H264Main, 33, 4096, true),
        support(EncodeProfile::H264High, 145, 4096, true),
        support(EncodeProfile::HevcMain, 65, 8192, true),
        main10,
    ];
    let entries = encode_caps_of(&supports);
    assert_eq!(entries.len(), 2);
    let (h264, hevc) = (&entries[0], &entries[1]);
    assert_eq!(
        (h264.codec, h264.direction),
        (RGFX_VIDEO_CODEC_H264, RGFX_VIDEO_ENCODE)
    );
    assert_eq!(h264.profiles, RGFX_VIDEO_H264_MAIN | RGFX_VIDEO_H264_HIGH);
    assert_eq!(h264.formats, RGFX_VIDEO_FORMAT_NV12);
    assert_eq!((h264.min_size, h264.max_size), ([145; 2], [4096; 2]));
    assert_eq!((h264.max_references, h264.max_dpb_pictures), (7, 8));
    assert_eq!(h264.max_level, H264_TOP_LEVEL);
    assert_eq!(h264.bitstream_alignment, 0);
    assert_eq!(hevc.codec, RGFX_VIDEO_CODEC_HEVC);
    assert_eq!(
        hevc.profiles,
        RGFX_VIDEO_HEVC_MAIN | RGFX_VIDEO_HEVC_MAIN_10
    );
    assert_eq!(
        hevc.formats,
        RGFX_VIDEO_FORMAT_NV12 | RGFX_VIDEO_FORMAT_P010
    );
    assert_eq!((hevc.min_size, hevc.max_size), ([129; 2], [8192; 2]));
    assert_eq!((hevc.max_references, hevc.max_dpb_pictures), (1, 2));
    assert_eq!(hevc.max_level, HEVC_TOP_LEVEL);
    assert!(encode_caps_of(&[]).is_empty());
}

/// An encode entry takes CQP, and CBR and VBR where NVENC lists them for every profile of its
/// codec, and no B-frames; a decode entry takes nothing.
#[test]
fn an_entry_s_rate_controls_are_cqp_and_what_nvenc_lists() {
    let supports = [
        support(EncodeProfile::H264High, 145, 4096, true),
        support(EncodeProfile::HevcMain, 129, 8192, false),
    ];
    let entries = encode_caps_of(&supports);
    let all = (1 << RGFX_VIDEO_RATE_CQP) | (1 << RGFX_VIDEO_RATE_CBR) | (1 << RGFX_VIDEO_RATE_VBR);
    let h264 = encode_caps(&entries[0], &supports);
    assert_eq!((h264.rate_controls, h264.max_b_frames), (all, 0));
    let hevc = encode_caps(&entries[1], &supports);
    assert_eq!(hevc.rate_controls, all & !(1 << RGFX_VIDEO_RATE_CBR));
    let decode = RgfxVideoCodecCaps {
        direction: RGFX_VIDEO_DECODE,
        ..entries[0]
    };
    assert_eq!(
        encode_caps(&decode, &supports),
        RgfxVideoEncodeCaps::default()
    );
}

/// The NVIDIA crate's settings of the wire's: the profile of each bit, 30 frames a second where
/// it says none, a VBR's peak 1.05 times its average where it sets none, a second's buffer
/// where it gives no window, the level, entropy, keyframe interval and colour as given, presets
/// P1 to P7 for quality 0 to 100, and no B-frames.
#[test]
fn the_nvenc_settings_of_the_wire_s() {
    let base = settings(RGFX_VIDEO_CODEC_H264, RGFX_VIDEO_H264_HIGH, 1920, 1080);
    let made = params_of(RGFX_VIDEO_CODEC_H264, &base).unwrap();
    assert_eq!(made.profile, EncodeProfile::H264High);
    assert_eq!((made.width, made.height), (1920, 1080));
    assert_eq!(made.frame_rate, (30, 1));
    assert_eq!(
        made.rate_control,
        EncodeRateControl::ConstantQp {
            intra: 22,
            predicted: 24,
            bidirectional: 26
        }
    );
    assert_eq!(made.buffer_milliseconds, 1000);
    assert_eq!(
        (made.gop_length, made.b_frames, made.level),
        (None, 0, None)
    );
    assert_eq!(
        (made.cabac, made.colour, made.full_range),
        (None, None, false)
    );
    assert_eq!((made.preset, made.tuning), (1, Tuning::HighQuality));
    let made = params_of(
        RGFX_VIDEO_CODEC_H264,
        &RgfxVideoEncodeSettings {
            rate_control: RGFX_VIDEO_RATE_VBR,
            average_bitrate: 8_000_000,
            max_bitrate_window_ms: 500,
            frame_rate_num: 0,
            level: 40,
            entropy: RGFX_VIDEO_ENTROPY_CAVLC,
            key_frame_interval: 60,
            color_primaries: 1,
            transfer: 1,
            matrix: 1,
            full_range: 1,
            quality: 50,
            ..base
        },
    )
    .unwrap();
    assert_eq!(
        made.rate_control,
        EncodeRateControl::VariableBitRate {
            average_bits_per_second: 8_000_000,
            peak_bits_per_second: 8_400_000,
            target_quality: None
        }
    );
    assert_eq!(made.frame_rate, (30, 1));
    assert_eq!(made.buffer_milliseconds, 500);
    assert_eq!((made.gop_length, made.level), (Some(60), Some(40)));
    assert_eq!(made.cabac, Some(false));
    assert_eq!((made.colour, made.full_range), (Some((1, 1, 1)), true));
    assert_eq!(made.preset, 4);
    let made = params_of(
        RGFX_VIDEO_CODEC_H264,
        &RgfxVideoEncodeSettings {
            rate_control: RGFX_VIDEO_RATE_CBR,
            average_bitrate: 1 << 40,
            entropy: RGFX_VIDEO_ENTROPY_CABAC,
            quality: 100,
            ..base
        },
    )
    .unwrap();
    assert_eq!(
        made.rate_control,
        EncodeRateControl::ConstantBitRate {
            bits_per_second: u32::MAX
        }
    );
    assert_eq!((made.cabac, made.preset), (Some(true), 7));
    let hevc = settings(RGFX_VIDEO_CODEC_HEVC, RGFX_VIDEO_HEVC_MAIN_10, 1920, 1080);
    let made = params_of(RGFX_VIDEO_CODEC_HEVC, &hevc).unwrap();
    assert_eq!(made.profile, EncodeProfile::HevcMain10);
    assert!(params_of(RGFX_VIDEO_CODEC_HEVC, &base).is_err());
    assert_eq!(
        [0, 8, 9, 24, 25, 50, 91, 92, 100, 1000].map(preset_of),
        [1, 1, 2, 2, 3, 4, 6, 7, 7, 7]
    );
}

#[test]
fn what_starts_a_new_sequence() {
    let old = settings(RGFX_VIDEO_CODEC_H264, RGFX_VIDEO_H264_HIGH, 640, 360);
    for new in [
        RgfxVideoEncodeSettings { width: 1280, ..old },
        RgfxVideoEncodeSettings {
            profile: RGFX_VIDEO_H264_MAIN,
            ..old
        },
        RgfxVideoEncodeSettings { level: 40, ..old },
        RgfxVideoEncodeSettings {
            key_frame_interval: 30,
            ..old
        },
        RgfxVideoEncodeSettings { matrix: 9, ..old },
        RgfxVideoEncodeSettings {
            entropy: RGFX_VIDEO_ENTROPY_CAVLC,
            ..old
        },
    ] {
        assert!(new_sequence(&old, &new));
    }
    for new in [
        RgfxVideoEncodeSettings { qp_p: 30, ..old },
        RgfxVideoEncodeSettings { quality: 80, ..old },
        RgfxVideoEncodeSettings {
            frame_rate_num: 60,
            ..old
        },
        RgfxVideoEncodeSettings {
            rate_control: RGFX_VIDEO_RATE_CBR,
            average_bitrate: 1_000_000,
            ..old
        },
    ] {
        assert!(!new_sequence(&old, &new));
    }
}

/// `MTLResourceStorageModeShared`.
const SHARED: u64 = 0;

/// The four-byte start code NVENC puts before each NAL unit.
const START_CODE: [u8; 4] = [0, 0, 0, 1];

/// The NAL units of an Annex-B access unit of `codec`, in order: each one's type and its bytes
/// from its header on.
fn nal_units(codec: u32, au: &[u8]) -> Vec<(u8, &[u8])> {
    let mut starts = Vec::new();
    let mut i = 0;
    while i + 3 <= au.len() {
        if au[i..i + 3] == [0, 0, 1] {
            starts.push(i + 3);
            i += 3;
        } else {
            i += 1;
        }
    }
    (starts.iter().enumerate())
        .map(|(n, &at)| {
            let end = starts.get(n + 1).map_or(au.len(), |&next| {
                let mut end = next - 3;
                while end > at && au[end - 1] == 0 {
                    end -= 1;
                }
                end
            });
            let unit = &au[at..end];
            let kind = if codec == RGFX_VIDEO_CODEC_H264 {
                unit[0] & 0x1f
            } else {
                (unit[0] >> 1) & 0x3f
            };
            (kind, unit)
        })
        .collect()
}

/// The parameter set types of `codec`: H.264's SPS and PPS, HEVC's VPS, SPS and PPS.
fn parameter_set_types(codec: u32) -> &'static [u8] {
    if codec == RGFX_VIDEO_CODEC_H264 {
        &[7, 8]
    } else {
        &[32, 33, 34]
    }
}

/// Sample `x`, `y` of plane `plane` (0 luma, 1 chroma, its pairs of samples a pixel each) of
/// frame `n`: smooth gradients that move, with a square that moves faster, in 8 bits.
fn sample(plane: usize, x: usize, y: usize, n: usize) -> u8 {
    let square = (x / 2 + 3 * n) % 160 < 48 && (y + 2 * n) % 120 < 48;
    let value = match plane {
        0 => 32 + (x + 2 * n) % 160 + (y + n) % 32,
        _ if x % 2 == 0 => 96 + (x / 4 + n) % 64,
        _ => 128 + (y / 2 + n) % 48,
    };
    u8::try_from(if square {
        255 - value.min(255)
    } else {
        value.min(255)
    })
    .unwrap_or(255)
}

/// Writes frame `n` of `settings`' size into `source`, its luma at 0 and its chroma at
/// `chroma_at`, rows `pitch` apart, at its format's bytes a sample (P010's high bits).
fn write_frame(
    source: &<FakePlatform as Platform>::Buffer,
    settings: &RgfxVideoEncodeSettings,
    pitch: usize,
    chroma_at: usize,
    n: usize,
) {
    let (w, h) = (settings.width as usize, settings.height as usize);
    let wide = settings.format == RGFX_VIDEO_FORMAT_P010;
    // SAFETY: the fake's Shared buffer holds the luma's and chroma's rows at `pitch` (the
    // caller's sizes), and nothing reads it until the encode that follows.
    let bytes = unsafe {
        std::slice::from_raw_parts_mut(source.contents().as_ptr::<u8>(), source.length() as usize)
    };
    for (plane, at, rows) in [(0, 0, h), (1, chroma_at, h / 2)] {
        for y in 0..rows {
            let row = &mut bytes[at + y * pitch..];
            for x in 0..w {
                let v = sample(plane, x, y, n);
                if wide {
                    row[2 * x..2 * x + 2].copy_from_slice(&(u16::from(v) << 8).to_le_bytes());
                } else {
                    row[x] = v;
                }
            }
        }
    }
}

/// The PSNR, in dB, of `got`'s rows, `bytes_per_row` apart, of plane `plane` of frame `n`, `w`
/// samples by `rows`, against the frame as [`sample`] makes it, at 8 bits (P010's high byte).
fn psnr(
    got: &[u8],
    bytes_per_row: usize,
    plane: usize,
    n: usize,
    w: usize,
    rows: usize,
    wide: bool,
) -> f64 {
    let mut squared = 0f64;
    for y in 0..rows {
        for x in 0..w {
            let at = y * bytes_per_row + if wide { 2 * x + 1 } else { x };
            let diff = f64::from(got[at]) - f64::from(sample(plane, x, y, n));
            squared += diff * diff;
        }
    }
    let mse = squared / (w * rows) as f64;
    if mse == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (255.0f64 * 255.0 / mse).log10()
}

/// The least PSNR each plane of each decoded frame must reach, in dB, at QP 22 and 24.
const LEAST_PSNR: f64 = 35.0;

/// On the NVIDIA host, each codec and profile NVENC encodes, through an encode session of the
/// arm's: 30 frames of a moving picture at 1280x720 at a constant QP, H.264 High at level 4.0.
/// The first access unit, a forced one (frame 10) and the one after an access unit past its
/// capacity (frame 20, 16 bytes) are IDR pictures with their parameter sets first, Annex B; the
/// rest are P pictures with none, a change of QP (frame 15) among them. Each written access
/// unit decodes back on NVDEC to a picture whose planes reach [`LEAST_PSNR`] against the frame.
/// Settings NVENC cannot take are refused.
#[test]
#[ignore = "needs an NVIDIA GPU: run on the NVIDIA host"]
fn each_profile_encodes_through_a_session_and_decodes_back_on_nvdec() {
    let gpu = Arc::new(Gpu::open(&GpuSelector::First).expect("an NVIDIA GPU"));
    let runtime: Arc<dyn Runtime> = StdRuntime::shared();
    let supports = encode_profiles(Arc::clone(&gpu)).expect("NVENC's profiles");
    let entries = encode_caps_of(&supports);
    let h264 = (entries.iter())
        .find(|e| e.codec == RGFX_VIDEO_CODEC_H264)
        .expect("an H.264 encode entry");
    assert_ne!(h264.profiles & RGFX_VIDEO_H264_HIGH, 0, "{h264:?}");
    let p = FakePlatform::new();
    let (width, height) = (1280u32, 720u32);
    for entry in &entries {
        let caps = encode_caps(entry, &supports);
        eprintln!("NVENC's entry {entry:?}, {caps:?}");
        assert_ne!(caps.rate_controls & (1 << RGFX_VIDEO_RATE_CQP), 0);
        for bit in (0..32)
            .map(|b| 1u32 << b)
            .filter(|b| entry.profiles & b != 0)
        {
            let what = format!("codec {} profile {bit:#x}", entry.codec);
            let mut s = settings(entry.codec, bit, width, height);
            if bit == RGFX_VIDEO_H264_HIGH && entry.codec == RGFX_VIDEO_CODEC_H264 {
                s.level = 40;
            }
            let encoder = NvencHostEncoder::new(&gpu, entry.codec, &supports, entry, &caps);
            for refused in [
                RgfxVideoEncodeSettings { b_frames: 1, ..s },
                RgfxVideoEncodeSettings {
                    width: entry.max_size[0] + 2,
                    ..s
                },
            ] {
                assert!(encoder.check(&refused).is_err(), "{what}: {refused:?}");
            }
            encoder.check(&s).expect("settings NVENC takes");
            let session = HostEncodeSession::<FakePlatform>::new(
                &p.device(),
                entry.codec,
                "NVENC",
                Box::new(encoder),
            )
            .expect("a session");
            let event = match session.event() {
                EventRef::Shared(event) => event,
                EventRef::Event(_) => panic!("a plain event"),
            };
            let wide = s.format == RGFX_VIDEO_FORMAT_P010;
            let row = width as usize * if wide { 2 } else { 1 };
            let pitch = row.next_multiple_of(256);
            let chroma_at = pitch * height as usize;
            let len = (chroma_at + pitch * height as usize / 2) as u64;
            let buffer = |len| {
                (p.device()
                    .new_buffer_of_length(OnFault::Return, len, SHARED))
                .expect("no raise")
                .expect("a buffer")
            };
            let source = buffer(len);
            let (bitstream, result) = (buffer(4 << 20), buffer(64));
            let plane = |offset, rows| VideoPlane {
                buffer: &source,
                offset,
                bytes_per_row: pitch as u64,
                row_bytes: row as u64,
                width: u64::from(width),
                height: rows,
                cached: false,
                pages: None,
                cache: None,
            };
            let mut decoder = AnnexBDecoder::new(
                &gpu,
                if entry.codec == RGFX_VIDEO_CODEC_H264 {
                    AnnexBCodec::H264
                } else {
                    AnnexBCodec::Hevc
                },
                &runtime,
            );
            let sets = parameter_set_types(entry.codec);
            let mut least = f64::INFINITY;
            for n in 0..30u64 {
                write_frame(&source, &s, pitch, chroma_at, n as usize);
                let frame_settings = match n {
                    0 => Some(s),
                    15 => Some(RgfxVideoEncodeSettings { qp_p: 26, ..s }),
                    _ => None,
                };
                let encode = VideoEncode {
                    luma: plane(0, u64::from(height)),
                    chroma: plane(chroma_at as u64, u64::from(height / 2)),
                    bitstream: &bitstream,
                    bitstream_offset: 0,
                    bitstream_capacity: if n == 20 { 16 } else { 4 << 20 },
                    result: &result,
                    result_offset: 0,
                    frame_index: n,
                    force_keyframe: n == 10,
                    settings: frame_settings,
                };
                let value = session
                    .encode(&encode)
                    .expect("an encode the session takes");
                assert!(
                    event.wait_until_signaled_value(value, 10_000),
                    "{what} frame {n}"
                );
                // SAFETY: the fake's Shared buffers hold their bytes, which the session wrote
                // before its signal and writes no more until the next encode.
                let (result_bytes, au) = unsafe {
                    (
                        std::slice::from_raw_parts(result.contents().as_ptr::<u8>(), 32),
                        std::slice::from_raw_parts(bitstream.contents().as_ptr::<u8>(), 4 << 20),
                    )
                };
                let r = RgfxVideoEncodeResult::read_from_bytes(result_bytes).expect("a result");
                let what = format!("{what} frame {n}: {r:?}");
                assert_eq!(r.frame_index, n, "{what}");
                if n == 20 {
                    assert_eq!(r.status, RGFX_VIDEO_ENCODE_OVERFLOW, "{what}");
                    assert!(r.bitstream_size > 16, "{what}");
                    continue;
                }
                assert_eq!(r.status, RGFX_VIDEO_ENCODE_OK, "{what}");
                let au = &au[..r.bitstream_size as usize];
                let units = nal_units(entry.codec, au);
                let types: Vec<u8> = units.iter().map(|(t, _)| *t).collect();
                assert_eq!(au.get(..4), Some(&START_CODE[..]), "{what}");
                if matches!(n, 0 | 10 | 21) {
                    assert_eq!(
                        r.flags,
                        RGFX_VIDEO_ENCODED_KEYFRAME | RGFX_VIDEO_ENCODED_PARAMETER_SETS,
                        "{what}"
                    );
                    assert_eq!(r.picture_type, RGFX_VIDEO_PICTURE_I, "{what}");
                    assert_eq!(types.get(..sets.len()), Some(sets), "{what}: {types:?}");
                } else {
                    assert_eq!(r.flags, 0, "{what}");
                    assert_eq!(r.picture_type, RGFX_VIDEO_PICTURE_P, "{what}");
                    assert!(!types.iter().any(|t| sets.contains(t)), "{what}: {types:?}");
                }
                if n == 0 {
                    if s.level == 40 {
                        // The SPS: its header, profile_idc (High's 100), its constraint flags
                        // and level_idc (4.0's 40).
                        let sps = units[0].1;
                        assert_eq!((sps[1], sps[3]), (100, 40), "{what}: {:02x?}", &sps[..4]);
                    }
                    let mut parameter_sets = Vec::new();
                    for (_, unit) in units.iter().filter(|(t, _)| sets.contains(t)) {
                        parameter_sets.extend_from_slice(&START_CODE);
                        parameter_sets.extend_from_slice(unit);
                    }
                    decoder
                        .parameter_sets(&parameter_sets)
                        .expect("the parameter sets");
                }
                let picture = (decoder.decode(au)).expect("a decode").expect("a picture");
                let read = picture.read_back((width, height)).expect("a read back");
                for (plane, read) in read.planes().iter().enumerate() {
                    let rows = read.rows as usize;
                    assert_eq!(rows, height as usize >> plane, "{what}");
                    let db = psnr(
                        read.bytes,
                        read.bytes_per_row,
                        plane,
                        n as usize,
                        width as usize,
                        rows,
                        wide,
                    );
                    least = least.min(db);
                    assert!(db >= LEAST_PSNR, "{what}: plane {plane} at {db:.1} dB");
                }
            }
            eprintln!("{what}: the least PSNR of a plane {least:.1} dB");
            drop(session);
        }
    }
}
