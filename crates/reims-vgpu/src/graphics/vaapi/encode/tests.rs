use reims_vgpu_graphics::wire::RGFX_VIDEO_DECODE;
use std::ptr::NonNull;

use reims_vgpu_video::core::device_memory::{DeviceMemory, HostView, MemoryIdentity, MemoryLife};
use reims_vgpu_video::core::runtime::StdRuntime;
use reims_vgpu_video::core::surfaces::{SurfaceError, SurfaceLayout};
use reims_vgpu_video::paravirt::Codec;
use reims_vgpu_video::vaapi::{H264Decoder, HevcDecoder, encoder_support};

use reims_vgpu_vulkan::RtPlatform;

use super::*;
use crate::graphics::video_choice::tested_gpu::Arm;

/// Settings of a 320x192 stream of `profile` of `codec` at a constant QP.
fn settings(codec: u32, profile: u32) -> RgfxVideoEncodeSettings {
    RgfxVideoEncodeSettings {
        width: 320,
        height: 192,
        format: if profile == RGFX_VIDEO_HEVC_MAIN_10 && codec == RGFX_VIDEO_CODEC_HEVC {
            RGFX_VIDEO_FORMAT_P010
        } else {
            RGFX_VIDEO_FORMAT_NV12
        },
        profile,
        rate_control: RGFX_VIDEO_RATE_CQP,
        qp_i: 26,
        qp_p: 28,
        qp_b: 30,
        frame_rate_num: 30,
        frame_rate_den: 1,
        ..RgfxVideoEncodeSettings::default()
    }
}

/// A sample of NAL units `units`, each behind its four-byte length.
fn sample(units: &[&[u8]]) -> Vec<u8> {
    let mut out = Vec::new();
    for unit in units {
        out.extend_from_slice(&(unit.len() as u32).to_be_bytes());
        out.extend_from_slice(unit);
    }
    out
}

/// An `hvcC` record of `units`, an array each, its fixed part zero but its version and its
/// four-byte lengths.
fn hvcc(units: &[(u8, &[u8])]) -> Vec<u8> {
    let mut record = vec![0u8; 22];
    record[0] = 1;
    record[21] = 3;
    record.push(units.len() as u8);
    for (kind, unit) in units {
        record.push(*kind);
        record.extend_from_slice(&1u16.to_be_bytes());
        record.extend_from_slice(&(unit.len() as u16).to_be_bytes());
        record.extend_from_slice(unit);
    }
    record
}

/// An H.264 sample becomes Annex B: each NAL unit after a start code, its parameter sets first
/// from the `avcC` where it is a keyframe's; an HEVC one takes its VPS, SPS and PPS from the
/// `hvcC`; a length past the sample's end is refused.
#[test]
fn a_sample_becomes_annex_b_its_keyframe_s_parameter_sets_first() {
    let (sps, pps) = ([0x67, 0x64, 0x00, 0x1f], [0x68, 0xee, 0x3c]);
    let record = reims_vgpu_video::codec::encode::h264::avcc(&sps, &pps);
    let slices = sample(&[&[0x65, 0x88, 0x84], &[0x06, 0x05]]);
    let au = annex_b(RGFX_VIDEO_CODEC_H264, Some(&record), &slices).unwrap();
    let want: Vec<u8> = [
        &START_CODE[..],
        &sps,
        &START_CODE,
        &pps,
        &START_CODE,
        &[0x65, 0x88, 0x84],
        &START_CODE,
        &[0x06, 0x05],
    ]
    .concat();
    assert_eq!(au, want);
    let au = annex_b(RGFX_VIDEO_CODEC_H264, None, &slices).unwrap();
    assert_eq!(au, want[4 + 4 + 4 + 3..]);

    let (vps, sps, pps) = ([0x40, 0x01, 0x0c], [0x42, 0x01, 0x01], [0x44, 0x01]);
    let record = hvcc(&[(0x20, &vps), (0x21, &sps), (0x22, &pps)]);
    let slice = sample(&[&[0x26, 0x01, 0xaf]]);
    let au = annex_b(RGFX_VIDEO_CODEC_HEVC, Some(&record), &slice).unwrap();
    let want: Vec<u8> = [
        &START_CODE[..],
        &vps,
        &START_CODE,
        &sps,
        &START_CODE,
        &pps,
        &START_CODE,
        &[0x26, 0x01, 0xaf],
    ]
    .concat();
    assert_eq!(au, want);

    let mut cut = sample(&[&[0x65, 0x88, 0x84]]);
    cut.pop();
    assert!(annex_b(RGFX_VIDEO_CODEC_H264, None, &cut).is_err());
    assert!(annex_b(RGFX_VIDEO_CODEC_H264, None, &[0, 0, 1]).is_err());
}

/// The rate controls the wire's settings ask for: CQP's QPs at most 51, CBR's rate within a u32
/// and at least one, and a VBR's peak 1.05 times its average where it names none, and never
/// below its average.
#[test]
fn the_rate_control_of_each_mode() {
    let s = RgfxVideoEncodeSettings {
        rate_control: RGFX_VIDEO_RATE_CQP,
        qp_i: 20,
        qp_p: 60,
        qp_b: 51,
        ..RgfxVideoEncodeSettings::default()
    };
    assert_eq!(
        rate_of(&s),
        RateControl::ConstantQp {
            intra: 20,
            predicted: 51,
            bidirectional: 51
        }
    );
    let cbr = |average| RgfxVideoEncodeSettings {
        rate_control: RGFX_VIDEO_RATE_CBR,
        average_bitrate: average,
        ..RgfxVideoEncodeSettings::default()
    };
    assert_eq!(
        rate_of(&cbr(1 << 40)),
        RateControl::ConstantBitRate {
            bits_per_second: u32::MAX
        }
    );
    assert_eq!(
        rate_of(&cbr(0)),
        RateControl::ConstantBitRate { bits_per_second: 1 }
    );
    let vbr = |average, max_bitrate| RgfxVideoEncodeSettings {
        rate_control: RGFX_VIDEO_RATE_VBR,
        average_bitrate: average,
        max_bitrate,
        ..RgfxVideoEncodeSettings::default()
    };
    assert_eq!(
        rate_of(&vbr(4_000_000, 0)),
        RateControl::VariableBitRate {
            average_bits_per_second: 4_000_000,
            peak_bits_per_second: 4_200_000
        }
    );
    assert_eq!(
        rate_of(&vbr(4_000_000, 6_000_000)),
        RateControl::VariableBitRate {
            average_bits_per_second: 4_000_000,
            peak_bits_per_second: 6_000_000
        }
    );
    assert_eq!(
        rate_of(&vbr(4_000_000, 1_000_000)),
        RateControl::VariableBitRate {
            average_bits_per_second: 4_000_000,
            peak_bits_per_second: 4_000_000
        }
    );
}

/// `quality` 0 is the driver's fastest level, its range's last, and 100 its best, 1, linearly
/// between; no range, no level.
#[test]
fn quality_maps_onto_the_driver_s_levels() {
    assert_eq!(quality_level(0, Some(7)), Some(7));
    assert_eq!(quality_level(100, Some(7)), Some(1));
    assert_eq!(quality_level(50, Some(7)), Some(4));
    assert_eq!(quality_level(200, Some(7)), Some(1));
    assert_eq!(quality_level(30, Some(1)), Some(1));
    assert_eq!(quality_level(30, None), None);
    assert_eq!(quality_level(30, Some(0)), None);
}

/// A change of the size, profile, keyframe interval or colour starts a new sequence; a change of
/// the bitrate, QPs, quality or frame rate does not.
#[test]
fn what_starts_a_new_sequence() {
    let old = settings(RGFX_VIDEO_CODEC_H264, RGFX_VIDEO_H264_HIGH);
    for new in [
        RgfxVideoEncodeSettings { width: 640, ..old },
        RgfxVideoEncodeSettings {
            profile: RGFX_VIDEO_H264_MAIN,
            ..old
        },
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

/// The VA crate's settings of the wire's: the profile of each bit, the frame rate's numerator
/// the clock (30 a second where it says none), the level, entropy, keyframe interval, colour
/// and hard limit as given, and frames in display order.
#[test]
fn the_va_settings_of_the_wire_s() {
    let made = settings_of(
        RGFX_VIDEO_CODEC_H264,
        &RgfxVideoEncodeSettings {
            level: 41,
            entropy: RGFX_VIDEO_ENTROPY_CAVLC,
            key_frame_interval: 120,
            color_primaries: 1,
            transfer: 1,
            matrix: 1,
            full_range: 1,
            frame_rate_num: 30000,
            frame_rate_den: 1001,
            flags: RGFX_VIDEO_ENCODE_HARD_LIMIT,
            quality: 100,
            ..settings(RGFX_VIDEO_CODEC_H264, RGFX_VIDEO_H264_MAIN)
        },
        Some(4),
    )
    .unwrap();
    assert_eq!(made.profile, EncodeProfile::H264Main);
    assert_eq!((made.width, made.height), (320, 192));
    assert_eq!(made.level, Some(41));
    assert_eq!(made.entropy, Some(Entropy::Cavlc));
    assert_eq!(made.key_frame_interval, Some(120));
    assert_eq!(
        made.colour,
        Some(ColourDescription {
            primaries: 1,
            transfer: 1,
            matrix: 1
        })
    );
    assert!(made.full_range);
    assert_eq!(made.frame_rate, FrameRate::new(30000, 1001));
    assert_eq!(made.timescale, 30000);
    assert_eq!(made.quality_level, Some(1));
    assert!(!made.allow_frame_reordering);
    assert_eq!((made.b_frames, made.max_frame_delay), (Some(0), Some(0)));

    let plain = settings_of(
        RGFX_VIDEO_CODEC_HEVC,
        &RgfxVideoEncodeSettings {
            frame_rate_num: 0,
            ..settings(RGFX_VIDEO_CODEC_HEVC, RGFX_VIDEO_HEVC_MAIN_10)
        },
        None,
    )
    .unwrap();
    assert_eq!(plain.profile, EncodeProfile::HevcMain10);
    assert_eq!(
        (plain.frame_rate, plain.timescale),
        (FrameRate::new(30, 1), 30)
    );
    assert_eq!(
        (plain.level, plain.colour, plain.quality_level),
        (None, None, None)
    );
    assert!(
        settings_of(
            RGFX_VIDEO_CODEC_HEVC,
            &settings(0, RGFX_VIDEO_H264_HIGH),
            None
        )
        .is_err()
    );
}

/// The rate controls an encode entry lists: the wire's bit of each `VA_RC_*` its profiles'
/// configs list; and it codes frames in place, H.264's and HEVC's alike. A decode entry lists
/// none of it.
#[test]
fn an_encode_entry_lists_its_configs_rate_controls() {
    let entry = RgfxVideoCodecCaps {
        codec: RGFX_VIDEO_CODEC_H264,
        direction: RGFX_VIDEO_ENCODE,
        ..RgfxVideoCodecCaps::default()
    };
    let in_place = RgfxVideoEncodeCaps {
        flags: RGFX_VIDEO_ENCODE_CAP_IN_PLACE,
        ..RgfxVideoEncodeCaps::default()
    };
    assert_eq!(encode_caps(&entry, &[]), in_place);
    let hevc = RgfxVideoCodecCaps {
        codec: RGFX_VIDEO_CODEC_HEVC,
        ..entry
    };
    assert_eq!(encode_caps(&hevc, &[]), in_place);
    let decode = RgfxVideoCodecCaps {
        direction: RGFX_VIDEO_DECODE,
        ..entry
    };
    assert_eq!(encode_caps(&decode, &[]), RgfxVideoEncodeCaps::default());
}

/// The VA display `RVG_VAAPI_NODE` names (`/dev/dri/renderD128` where it is unset).
fn display() -> Display {
    let node = std::env::var("RVG_VAAPI_NODE").unwrap_or_else(|_| "/dev/dri/renderD128".into());
    Display::open(&node, Arc::new(StdRuntime)).expect("a VA display")
}

/// Frame `index` of a moving 4:2:0 picture of `settings`' size and format, its luma and chroma
/// planes at their rows' own width.
fn picture(settings: &RgfxVideoEncodeSettings, index: u64) -> (Vec<u8>, Vec<u8>) {
    let (w, h) = (settings.width as usize, settings.height as usize);
    let sample = if settings.format == RGFX_VIDEO_FORMAT_P010 {
        2
    } else {
        1
    };
    let value = |x: usize, y: usize, shift: usize| ((x + y + shift) % 220 + 16) as u16;
    let mut luma = Vec::with_capacity(w * h * sample);
    for y in 0..h {
        for x in 0..w {
            let v = value(x, y, index as usize * 4);
            if sample == 2 {
                luma.extend_from_slice(&(v << 8).to_le_bytes());
            } else {
                luma.push(v as u8);
            }
        }
    }
    let mut chroma = Vec::with_capacity(w * h / 2 * sample);
    for y in 0..h / 2 {
        for x in 0..w {
            let v = 128 + (value(x, y, 0) % 32);
            if sample == 2 {
                chroma.extend_from_slice(&(v << 8).to_le_bytes());
            } else {
                chroma.push(v as u8);
            }
        }
    }
    (luma, chroma)
}

/// The NAL unit types of an Annex-B access unit of `codec`, in order.
fn nal_types(codec: u32, au: &[u8]) -> Vec<u8> {
    let mut types = Vec::new();
    let mut i = 0;
    while i + 4 < au.len() {
        if au[i..i + 4] == START_CODE {
            let header = au[i + 4];
            types.push(if codec == RGFX_VIDEO_CODEC_H264 {
                header & 0x1f
            } else {
                (header >> 1) & 0x3f
            });
            i += 4;
        } else {
            i += 1;
        }
    }
    types
}

/// On the driver, each codec and profile it encodes (8 frames of a moving picture, at a
/// constant QP): the first access unit is a keyframe with its parameter sets, Annex B, the rest
/// P pictures with none; an access unit past a capacity of 16 bytes is not written, and the
/// next one is a keyframe again; and the driver's own decoder decodes every access unit, each
/// to a picture of the size encoded.
#[test]
#[ignore = "needs a VA-API GPU: run on the VA-API host"]
fn each_profile_encodes_on_the_driver_and_decodes_back() {
    let display = display();
    let supports: Arc<[EncoderSupport]> = encoder_support(&display).into();
    let caps = display.capabilities().unwrap();
    let entries = encode_caps_of(&supports, &caps);
    assert!(
        !entries.is_empty(),
        "the driver encodes neither H.264 nor HEVC"
    );
    for entry in &entries {
        let rates = encode_caps(entry, &supports);
        assert_ne!(rates.rate_controls & (1 << RGFX_VIDEO_RATE_CQP), 0);
        for bit in (0..32)
            .map(|b| 1u32 << b)
            .filter(|b| entry.profiles & b != 0)
        {
            let s = settings(entry.codec, bit);
            let encoder = VaEncoder::<RtPlatform>::new(
                &display,
                entry.codec,
                Arc::clone(&supports),
                entry,
                &rates,
                Err("the test copies none by the GPU".to_owned()),
            );
            HostEncoder::<RtPlatform>::check(&encoder, &s).unwrap();
            let sets: &[u8] = if entry.codec == RGFX_VIDEO_CODEC_H264 {
                &[7, 8]
            } else {
                &[32, 33, 34]
            };
            let mut units = Vec::new();
            let capacity = |index: u64| if index == 4 { 16 } else { 1 << 20 };
            for index in 0..8u64 {
                let frame = EncodeFrame {
                    settings: (index == 0).then_some(s),
                    frame_index: index,
                    force_keyframe: false,
                    in_place: false,
                };
                let frame = encoder.frame(frame);
                let (luma, chroma) = picture(&s, index);
                let row = luma.len() / s.height as usize;
                let source = EncodeSource {
                    luma: SourcePlane {
                        at: luma.as_ptr(),
                        bytes_per_row: row,
                        row_bytes: row,
                        rows: s.height as usize,
                    },
                    chroma: SourcePlane {
                        at: chroma.as_ptr(),
                        bytes_per_row: row,
                        row_bytes: row,
                        rows: s.height as usize / 2,
                    },
                    width: s.width,
                    height: s.height,
                };
                let mut out = vec![0u8; capacity(index)];
                let encoded = frame.finish_from_host(&source, &mut out).unwrap();
                assert_eq!(encoded.frame_index, index);
                let au = out[..encoded.size.min(out.len() as u64) as usize].to_vec();
                let types = nal_types(entry.codec, &au);
                let what = format!(
                    "codec {} profile {bit:#x} frame {index}: {types:?}",
                    entry.codec
                );
                match index {
                    4 => assert!(encoded.overflow, "{what}"),
                    0 | 5 => {
                        assert!(encoded.keyframe && encoded.parameter_sets, "{what}");
                        assert_eq!(encoded.picture_type, RGFX_VIDEO_PICTURE_I, "{what}");
                        assert_eq!(types.get(..sets.len()), Some(sets), "{what}");
                        units.push(au);
                    }
                    _ => {
                        assert!(!encoded.overflow && !encoded.keyframe, "{what}");
                        assert_eq!(encoded.picture_type, RGFX_VIDEO_PICTURE_P, "{what}");
                        assert!(!types.iter().any(|t| sets.contains(t)), "{what}");
                        units.push(au);
                    }
                }
            }
            // Frame 4 was not written: the decoder gets the rest, which restart at frame 5's
            // keyframe.
            let mut decoded = 0;
            if entry.codec == RGFX_VIDEO_CODEC_H264 {
                let mut decoder = H264Decoder::new_annex_b(&display);
                for (id, au) in units.iter().enumerate() {
                    for frame in decoder.decode(au, id as u64).unwrap() {
                        assert_eq!((frame.width(), frame.height()), (s.width, s.height));
                        decoded += 1;
                    }
                }
                decoded += decoder.flush().unwrap().len();
            } else {
                let mut decoder = HevcDecoder::new_annex_b(&display);
                for (id, au) in units.iter().enumerate() {
                    for frame in decoder.decode(au, id as u64).unwrap() {
                        assert_eq!((frame.width(), frame.height()), (s.width, s.height));
                        decoded += 1;
                    }
                }
                decoded += decoder.flush().unwrap().len();
            }
            assert_eq!(
                decoded,
                units.len(),
                "codec {} profile {bit:#x}",
                entry.codec
            );
        }
    }
}

/// Each frame of `frames`, `(index, in_place, qp)`, through an encoder of `codec`'s `entry`
/// on the driver, from the host bytes of [`picture`] `index`, at constant QP `qp` (settings
/// sent where it changes): each one's [`Encoded`], and its access unit where it was written.
fn encode_frames(
    display: &Display,
    supports: &Arc<[EncoderSupport]>,
    entry: &RgfxVideoCodecCaps,
    s: &RgfxVideoEncodeSettings,
    frames: &[(u64, bool, u32)],
) -> Vec<Result<(Encoded, Vec<u8>), BackendError>> {
    let rates = encode_caps(entry, supports);
    let encoder = VaEncoder::<RtPlatform>::new(
        display,
        entry.codec,
        Arc::clone(supports),
        entry,
        &rates,
        Err("the test copies none by the GPU".to_owned()),
    );
    let mut qp = None;
    let mut out = Vec::new();
    for &(index, in_place, at) in frames {
        let settings = RgfxVideoEncodeSettings {
            qp_i: at,
            qp_p: at,
            qp_b: at,
            ..*s
        };
        let changed = qp != Some(at);
        qp = Some(at);
        let frame = EncodeFrame {
            settings: changed.then_some(settings),
            frame_index: index,
            force_keyframe: false,
            in_place,
        };
        let frame = encoder.frame(frame);
        let (luma, chroma) = picture(s, index);
        let row = luma.len() / s.height as usize;
        let source = EncodeSource {
            luma: SourcePlane {
                at: luma.as_ptr(),
                bytes_per_row: row,
                row_bytes: row,
                rows: s.height as usize,
            },
            chroma: SourcePlane {
                at: chroma.as_ptr(),
                bytes_per_row: row,
                row_bytes: row,
                rows: s.height as usize / 2,
            },
            width: s.width,
            height: s.height,
        };
        let mut bytes = vec![0u8; 1 << 20];
        let encoded = frame.finish_from_host(&source, &mut bytes);
        out.push(encoded.map(|e| {
            bytes.truncate(e.size as usize);
            (e, bytes)
        }));
    }
    out
}

/// On the driver, each codec and profile it encodes, frames coded again in place of the last
/// (as AppleGVA's limiter codes a frame again at a higher QP): the key frame twice, frame 2
/// three times and frame 4 once more, then frames to 7. Each try is its frame's, of its picture
/// type; the last try of each frame makes the stream, which the driver's own decoder decodes
/// whole, every frame of it. A frame coded in place with none held is refused, and the frame
/// after it is a keyframe.
#[test]
#[ignore = "needs a VA-API GPU: run on the VA-API host"]
fn frames_coded_in_place_decode_back_in_their_places() {
    let display = display();
    let supports: Arc<[EncoderSupport]> = encoder_support(&display).into();
    let caps = display.capabilities().unwrap();
    let entries = encode_caps_of(&supports, &caps);
    assert!(
        !entries.is_empty(),
        "the driver encodes neither H.264 nor HEVC"
    );
    let frames = [
        (0, false, 26),
        (0, true, 29),
        (1, false, 26),
        (2, false, 26),
        (2, true, 29),
        (2, true, 32),
        (3, false, 26),
        (4, false, 26),
        (4, true, 29),
        (5, false, 26),
        (6, false, 26),
        (7, false, 26),
    ];
    for entry in &entries {
        assert_ne!(
            encode_caps(entry, &supports).flags & RGFX_VIDEO_ENCODE_CAP_IN_PLACE,
            0
        );
        for bit in (0..32)
            .map(|b| 1u32 << b)
            .filter(|b| entry.profiles & b != 0)
        {
            let s = settings(entry.codec, bit);
            let what = format!("codec {} profile {bit:#x}", entry.codec);
            let coded = encode_frames(&display, &supports, entry, &s, &frames);
            let mut units: Vec<Vec<u8>> = Vec::new();
            let mut last = None;
            for (n, (try_, &(index, in_place, _))) in coded.iter().zip(&frames).enumerate() {
                let (encoded, au) = try_
                    .as_ref()
                    .unwrap_or_else(|e| panic!("{what} try {n}: {e}"));
                assert_eq!(encoded.frame_index, index, "{what} try {n}");
                assert!(!encoded.overflow, "{what} try {n}");
                let key = index == 0;
                assert_eq!(encoded.keyframe, key, "{what} try {n}");
                assert_eq!(
                    encoded.picture_type,
                    if key {
                        RGFX_VIDEO_PICTURE_I
                    } else {
                        RGFX_VIDEO_PICTURE_P
                    },
                    "{what} try {n}"
                );
                if in_place {
                    assert_eq!(last, Some(index), "{what} try {n}");
                    *units.last_mut().unwrap() = au.clone();
                } else {
                    units.push(au.clone());
                }
                last = Some(index);
            }
            assert_eq!(units.len(), 8, "{what}");
            let mut decoded = 0;
            if entry.codec == RGFX_VIDEO_CODEC_H264 {
                let mut decoder = H264Decoder::new_annex_b(&display);
                for (id, au) in units.iter().enumerate() {
                    decoded += decoder.decode(au, id as u64).unwrap().len();
                }
                decoded += decoder.flush().unwrap().len();
            } else {
                let mut decoder = HevcDecoder::new_annex_b(&display);
                for (id, au) in units.iter().enumerate() {
                    decoded += decoder.decode(au, id as u64).unwrap().len();
                }
                decoded += decoder.flush().unwrap().len();
            }
            assert_eq!(decoded, 8, "{what}");
            // Refused: in place of none, then a new frame is a keyframe.
            let refused = encode_frames(
                &display,
                &supports,
                entry,
                &s,
                &[
                    (0, false, 26),
                    (1, false, 26),
                    (5, true, 26),
                    (2, false, 26),
                ],
            );
            assert!(refused[2].is_err(), "{what}");
            let (encoded, _) = refused[3].as_ref().unwrap();
            assert!(encoded.keyframe, "{what}");
        }
    }
}

/// A frame the runtime device's GPU writes into an input of the encoder's (`fill`'s filler)
/// encodes as the same frame uploaded by the CPU, byte for byte: on the driver of the rt device's
/// GPU, which the arm runs on (`video_choice::tested_gpu::check_rt`), for each codec and profile
/// it encodes, 6 frames of a moving picture at a constant QP, 320x192 (no margin to the coded
/// size) and 330x202 (a margin of columns and rows), each in a buffer of the device's, its own
/// memory or over host pages, by turns, rows wider than the picture's (512 bytes apart for 8-bit
/// samples, 768 or 1024 for 16-bit) from byte 32. Two encoders of the same settings code the
/// frames, one from the host bytes, one from the filled inputs; every access unit of the second
/// is the first's.
#[test]
#[ignore = "needs a VA-API GPU: run on the VA-API host"]
fn a_frame_filled_by_the_gpu_encodes_as_the_uploaded_frame_byte_for_byte() {
    use reims_vgpu_video::codec::encode::{InputFrame, Plane};
    use reims_vgpu_vulkan::core::platform::Handle as _;
    use reims_vgpu_vulkan::core::platform::HostRange;
    use reims_vgpu_vulkan::core::platform::device::GpuDevice as _;
    use reims_vgpu_vulkan::core::platform::resource::Buffer as _;
    use reims_vgpu_vulkan::core::raise::OnFault;

    use super::super::super::video_host::OutputPlane;
    use super::super::{GpuCopies as _, fill};

    let platform = crate::rt_test_end::platform();
    crate::entry::run(|_| {
        let device = platform.system_default_device().expect("an rt device");
        crate::graphics::video_choice::tested_gpu::check_rt(&device, Arm::VaApi);
        let gpu = device.host_gpu();
        let node = reims_vgpu_vulkan::host_gpus::render_node_at(gpu.render, gpu.pci);
        let runtime = crate::video::runtime::PlatformRuntime::of(&platform);
        let display = Display::open(node.expect("a render node"), Arc::clone(&runtime))
            .expect("a VA display on the device's GPU");
        let copies = fill::DeviceFills::<RtPlatform>::new(&device, &runtime);
        let mut filler = copies.filler().expect("a filler");
        let supports: Arc<[EncoderSupport]> = encoder_support(&display).into();
        let entries = encode_caps_of(&supports, &display.capabilities().unwrap());
        assert!(
            !entries.is_empty(),
            "the driver encodes neither H.264 nor HEVC"
        );
        let vm = platform.vm();
        let first = 32u64;
        let mut frames = 0;
        for entry in &entries {
            for bit in (0..32)
                .map(|b| 1u32 << b)
                .filter(|b| entry.profiles & b != 0)
            {
                for (width, height) in [(320u32, 192u32), (330, 202)] {
                    let s = RgfxVideoEncodeSettings {
                        width,
                        height,
                        ..settings(entry.codec, bit)
                    };
                    let what = format!("codec {} profile {bit:#x} {width}x{height}", entry.codec);
                    let made = settings_of(entry.codec, &s, None).expect("the VA settings");
                    let mut by_cpu = VideoEncoder::new(&display, made.clone()).expect("an encoder");
                    let mut by_gpu = VideoEncoder::new(&display, made).expect("an encoder");
                    let sample = made_sample(&s);
                    let row = u64::from(width) * sample;
                    // Rows wider than the picture's, from a byte past the buffer's start: 512
                    // apart for 8-bit samples, 768 or 1024 for 16-bit ones.
                    let pitch = (row + 128).next_multiple_of(256);
                    let chroma_at = first + pitch * u64::from(height);
                    let len = (chroma_at + pitch * u64::from(height / 2))
                        .next_multiple_of(vm.page_size());
                    for n in 0..6u64 {
                        let (luma, chroma) = picture(&s, n);
                        // The frame's bytes in a buffer: the device's own memory, or host pages.
                        let (buffer, at) = if n % 2 == 0 {
                            let buffer = (device.new_buffer_of_length(OnFault::Return, len, 0))
                                .expect("no raise")
                                .expect("a buffer");
                            let at = buffer.contents();
                            (buffer, at)
                        } else {
                            let at = vm.allocate_anywhere(len).expect("host pages");
                            let keep = Arc::clone(&platform);
                            // SAFETY: the pages stay mapped until the platform's release of the
                            // buffer runs the deallocator.
                            let bytes = unsafe { HostRange::new(at, len) };
                            let unmap: Box<dyn FnOnce() + Send> = Box::new(move || {
                                let _ = keep.vm().deallocate(at, len);
                            });
                            let buffer = (device.new_buffer_over_host_memory(
                                OnFault::Return,
                                bytes,
                                0,
                                Some(unmap),
                            ))
                            .expect("no raise")
                            .expect("a buffer over the pages");
                            (buffer, at)
                        };
                        for (plane, from, rows) in [
                            (first, &luma, u64::from(height)),
                            (chroma_at, &chroma, u64::from(height / 2)),
                        ] {
                            for r in 0..rows {
                                let src = &from[(r * row) as usize..((r + 1) * row) as usize];
                                // SAFETY: the row lies in the buffer's `len` host bytes, which no
                                // GPU work reaches yet.
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        src.as_ptr(),
                                        at.wrapping_add(plane + r * pitch).as_ptr::<u8>(),
                                        src.len(),
                                    );
                                }
                            }
                        }
                        let plane = |offset: u64, rows: u64, w: u64| OutputPlane {
                            buffer: buffer.clone_ref(),
                            offset,
                            bytes_per_row: pitch,
                            row_bytes: row,
                            width: w,
                            height: rows,
                            rows_held: rows,
                            cached: false,
                            host: Some(at.wrapping_add(offset)),
                            pages: None,
                            cache: None,
                            surface: None,
                        };
                        let luma_plane = plane(first, u64::from(height), u64::from(width));
                        let chroma_plane =
                            plane(chroma_at, u64::from(height / 2), u64::from(width / 2));
                        let input = InputFrame {
                            planes: [
                                Plane {
                                    data: &luma,
                                    bytes_per_row: row as usize,
                                },
                                Plane {
                                    data: &chroma,
                                    bytes_per_row: row as usize,
                                },
                            ],
                            pts: i64::try_from(n).expect("a time"),
                            duration: 1,
                            force_key_frame: false,
                            id: n,
                        };
                        let uploaded = by_cpu.encode(&input).expect("the upload's encode");
                        let written = (filler.fill(&by_gpu, &luma_plane, &chroma_plane))
                            .expect("the GPU's write");
                        let target = match written {
                            Filled::Written(target) => target,
                            Filled::Declined(why) => panic!("{what}: declined: {why}"),
                        };
                        let filled = by_gpu.encode_target(&target, &input).expect("the encode");
                        let data =
                            |coded: &[Coded<reims_vgpu_video::vaapi::Error>]| -> Vec<Vec<u8>> {
                                (coded.iter())
                                    .map(|c| c.as_ref().expect("a sample").data.to_vec())
                                    .collect()
                            };
                        let (a, b) = (data(&uploaded), data(&filled));
                        assert_eq!(a.len(), b.len(), "{what} frame {n}: samples");
                        assert!(
                            a == b,
                            "{what} frame {n}: the filled frame's access unit differs"
                        );
                        frames += 1;
                        drop((target, luma_plane, chroma_plane, buffer));
                    }
                }
            }
        }
        eprintln!("{frames} frames filled by the GPU encoded as uploaded");
        assert!(frames > 0);
    });
}

/// The bytes of a sample of `settings`' format.
fn made_sample(settings: &RgfxVideoEncodeSettings) -> u64 {
    if settings.format == RGFX_VIDEO_FORMAT_P010 {
        2
    } else {
        1
    }
}

/// A guest surface's memory in host bytes, which a [`HostView`] reaches.
struct HostSurface {
    life: MemoryLife,
    bytes: Box<[u8]>,
}

impl DeviceMemory for HostSurface {
    fn identity(&self) -> MemoryIdentity {
        self.life.identity()
    }

    fn length(&self) -> usize {
        self.bytes.len()
    }

    fn host_view(&self) -> Result<HostView, SurfaceError> {
        Ok(HostView {
            first: NonNull::from(&self.bytes[0]),
            size: self.bytes.len(),
            offset: 0,
            hold: Box::new(()),
        })
    }
}

/// A 66x50 `420v` surface laid out as CoreVideo lays out a pool's buffer, rows 128 bytes apart
/// and chroma from byte 8192, holding frame `index` of a moving picture.
fn guest_surface(index: usize) -> (SurfaceLayout, HostSurface) {
    let (w, h, pitch, chroma_at) = (66, 50, 128, 8192);
    let mut bytes = vec![0u8; chroma_at + pitch * h / 2].into_boxed_slice();
    for y in 0..h {
        for x in 0..w {
            bytes[y * pitch + x] = ((x + y + index * 4) % 220 + 16) as u8;
        }
    }
    for y in 0..h / 2 {
        for x in 0..w {
            bytes[chroma_at + y * pitch + x] = 128 + (x % 32) as u8;
        }
    }
    let plane = |offset, width, height| PlaneLayout {
        offset,
        bytes_per_row: pitch,
        width,
        height,
    };
    let layout = SurfaceLayout {
        pixel_format: reims_vgpu_video::core::fourcc::FourCc::new(*b"420v"),
        width: w as u32,
        height: h as u32,
        planes: vec![
            plane(0, w as u32, h as u32),
            plane(chroma_at, w as u32 / 2, h as u32 / 2),
        ],
        bytes_per_row: 0,
    };
    let life = MemoryLife::new();
    (layout, HostSurface { life, bytes })
}

/// On the driver, each guest encoder it answers (`guest_encoders`), at its first profile: 8
/// frames read from a guest surface wider and taller than the 64x48 stream come out as 8
/// samples, the first a sync sample with the configuration record, and the driver's own decoder
/// decodes each to a 64x48 picture; a reconfigure of the bitrate is taken after a frame, and one
/// of the profile (to the last, Main10 where the driver writes it) and one of the colour (to
/// BT.2020's primaries) are taken and left out, every configuration record the first's.
#[test]
#[ignore = "needs a VA-API GPU: run on the VA-API host"]
fn a_guest_encoder_encodes_the_frames_of_the_surfaces_it_is_lent() {
    let display = display();
    let supports: Arc<[EncoderSupport]> = encoder_support(&display).into();
    let encoders = guest_encoders(&supports);
    assert!(
        !encoders.is_empty(),
        "the driver encodes neither H.264 nor HEVC"
    );
    for encoder in &encoders {
        let profile = encoder.profiles[0];
        let mut settings = EncoderSettings::new(profile, 64, 48);
        settings.timescale = 90_000;
        let config = EncoderConfig {
            settings,
            quality: Some(0.5),
        };
        let mut guest = VaGuestEncoder::<RtPlatform>::new(
            &display,
            &supports,
            &config,
            Err("no GPU copies in this test".to_owned()),
        )
        .unwrap();
        let mut samples = Vec::new();
        for index in 0..8 {
            let (layout, memory) = guest_surface(index);
            let source = GuestSource {
                layout,
                memory: &memory,
                buffer: None,
            };
            let frame = SourceFrame {
                id: index as u64,
                pts: index as i64 * 3000,
                duration: 3000,
                force_key_frame: false,
            };
            samples.extend(guest.encode(&source, &frame).unwrap());
            if index == 0 {
                let mut faster = config.clone();
                faster.settings.rate_control = RateControl::VariableBitRate {
                    average_bits_per_second: 1_000_000,
                    peak_bits_per_second: 1_500_000,
                };
                guest.reconfigure(&faster).unwrap();
                let mut other = faster.clone();
                other.settings.profile = *encoder.profiles.last().unwrap();
                guest.reconfigure(&other).unwrap();
                let mut tinted = faster.clone();
                tinted.settings.colour = Some(ColourDescription {
                    primaries: 9,
                    transfer: 1,
                    matrix: 9,
                });
                guest.reconfigure(&tinted).unwrap();
            }
        }
        samples.extend(guest.flush().unwrap());
        let samples: Vec<_> = samples.into_iter().map(Result::unwrap).collect();
        assert_eq!(samples.len(), 8, "{profile:?}");
        assert!(
            samples[0].is_sync && samples[0].configuration.is_some(),
            "{profile:?}"
        );
        assert!(
            (samples.iter())
                .all(|s| s.configuration.is_none() || s.configuration == samples[0].configuration),
            "{profile:?}: the profile's change made a new sequence"
        );
        let codec = match encoder.codec {
            Codec::H264 => RGFX_VIDEO_CODEC_H264,
            Codec::Hevc => RGFX_VIDEO_CODEC_HEVC,
            Codec::Av1 | Codec::Vp9 => unreachable!("no arm encodes AV1 or VP9"),
        };
        let units: Vec<Vec<u8>> = (samples.iter())
            .map(|s| annex_b(codec, s.configuration.as_deref(), &s.data).unwrap())
            .collect();
        let mut sizes = Vec::new();
        if codec == RGFX_VIDEO_CODEC_H264 {
            let mut decoder = H264Decoder::new_annex_b(&display);
            for (id, au) in units.iter().enumerate() {
                sizes.extend(
                    decoder
                        .decode(au, id as u64)
                        .unwrap()
                        .iter()
                        .map(|f| (f.width(), f.height())),
                );
            }
            sizes.extend(
                decoder
                    .flush()
                    .unwrap()
                    .iter()
                    .map(|f| (f.width(), f.height())),
            );
        } else {
            let mut decoder = HevcDecoder::new_annex_b(&display);
            for (id, au) in units.iter().enumerate() {
                sizes.extend(
                    decoder
                        .decode(au, id as u64)
                        .unwrap()
                        .iter()
                        .map(|f| (f.width(), f.height())),
                );
            }
            sizes.extend(
                decoder
                    .flush()
                    .unwrap()
                    .iter()
                    .map(|f| (f.width(), f.height())),
            );
        }
        assert_eq!(sizes, vec![(64, 48); 8], "{profile:?}");
    }
}

/// On the driver of the rt device's GPU, which the arm runs on
/// (`video_choice::tested_gpu::check_rt`), each guest encoder it answers, at each 8-bit profile
/// it encodes, at a constant QP: 6 frames of a 64x48 stream read from guest surfaces wider and
/// taller than it, written by the GPU (`fill`'s filler) from a buffer of the device's that holds
/// the surface's bytes, code as the same frames read by the CPU from the surface's host bytes,
/// sample for sample and byte for byte; and each of the six went by the GPU.
#[test]
#[ignore = "needs a VA-API GPU: run on the VA-API host"]
fn a_guest_encoders_frames_filled_by_the_gpu_code_as_read_by_the_cpu() {
    use reims_vgpu_vulkan::core::platform::device::GpuDevice as _;
    use reims_vgpu_vulkan::core::platform::resource::Buffer as _;
    use reims_vgpu_vulkan::core::raise::OnFault;

    use super::super::{GpuCopies as _, fill};

    let platform = crate::rt_test_end::platform();
    crate::entry::run(|_| {
        let device = platform.system_default_device().expect("an rt device");
        crate::graphics::video_choice::tested_gpu::check_rt(&device, Arm::VaApi);
        let gpu = device.host_gpu();
        let node = reims_vgpu_vulkan::host_gpus::render_node_at(gpu.render, gpu.pci);
        let runtime = crate::video::runtime::PlatformRuntime::of(&platform);
        let display = Display::open(node.expect("a render node"), Arc::clone(&runtime))
            .expect("a VA display on the device's GPU");
        let copies = fill::DeviceFills::<RtPlatform>::new(&device, &runtime);
        let supports: Arc<[EncoderSupport]> = encoder_support(&display).into();
        let encoders = guest_encoders(&supports);
        assert!(
            !encoders.is_empty(),
            "the driver encodes neither H.264 nor HEVC"
        );
        let mut frames = 0;
        for encoder in &encoders {
            for &profile in &encoder.profiles {
                if profile.bit_depth() != 8 {
                    continue;
                }
                let mut settings = EncoderSettings::new(profile, 64, 48);
                settings.timescale = 90_000;
                settings.rate_control = RateControl::ConstantQp {
                    intra: 26,
                    predicted: 28,
                    bidirectional: 30,
                };
                let config = EncoderConfig {
                    settings,
                    quality: Some(0.5),
                };
                let open = |filler| {
                    VaGuestEncoder::<RtPlatform>::new(&display, &supports, &config, filler)
                        .expect("a guest encoder")
                };
                let mut by_cpu = open(Err("read by the CPU in this test".to_owned()));
                let mut by_gpu = open(copies.filler());
                let (mut read, mut filled) = (Vec::new(), Vec::new());
                for index in 0..6 {
                    let (layout, memory) = guest_surface(index);
                    let length = memory.bytes.len() as u64;
                    let buffer = (device.new_buffer_of_length(OnFault::Return, length, 0))
                        .expect("no raise")
                        .expect("a buffer");
                    // SAFETY: the buffer's `length` host bytes, which no GPU work reaches yet.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            memory.bytes.as_ptr(),
                            buffer.contents().as_ptr::<u8>(),
                            memory.bytes.len(),
                        );
                    }
                    let frame = SourceFrame {
                        id: index as u64,
                        pts: index as i64 * 3000,
                        duration: 3000,
                        force_key_frame: false,
                    };
                    let from_host = GuestSource::<RtPlatform> {
                        layout: layout.clone(),
                        memory: &memory,
                        buffer: None,
                    };
                    let from_buffer = GuestSource {
                        layout,
                        memory: &memory,
                        buffer: Some(&buffer),
                    };
                    read.extend(by_cpu.encode(&from_host, &frame).unwrap());
                    filled.extend(by_gpu.encode(&from_buffer, &frame).unwrap());
                    frames += 1;
                }
                read.extend(by_cpu.flush().unwrap());
                filled.extend(by_gpu.flush().unwrap());
                assert!(
                    by_gpu.fills.said.is_empty(),
                    "{profile:?}: frames uploaded by the CPU: {:?}",
                    by_gpu.fills.said
                );
                assert_eq!(
                    (by_gpu.fills.filled, by_gpu.fills.uploaded),
                    (6, 0),
                    "{profile:?}"
                );
                let data = |coded: Vec<Coded<VideoError>>| -> Vec<Vec<u8>> {
                    let samples = coded.into_iter().map(|c| c.expect("a sample").data);
                    samples.collect()
                };
                let (a, b) = (data(read), data(filled));
                assert_eq!(a.len(), 6, "{profile:?}: samples");
                assert!(a == b, "{profile:?}: a filled frame's sample differs");
            }
        }
        eprintln!("{frames} guest frames filled by the GPU coded as read by the CPU");
        assert!(frames > 0);
    });
}

/// A driver's answer for `profile` up to `max_width` by `max_height`, with no attribute but its
/// quality range.
fn support(profile: EncodeProfile, max_width: u32, max_height: u32) -> EncoderSupport {
    use reims_vgpu_video::vaapi::{EncoderCapabilities, Entrypoint, Profile};
    EncoderSupport {
        profile,
        capabilities: EncoderCapabilities {
            profile: Profile(0),
            entrypoint: Entrypoint(0),
            rt_formats: None,
            rate_control: None,
            packed_headers: None,
            max_ref_frames: None,
            max_slices: None,
            slice_structure: None,
            quality_range: Some(7),
            intra_refresh: None,
            skip_frame: None,
            roi: None,
            rate_control_ext: None,
            max_frame_size: None,
            prediction_direction: None,
            max_width: None,
            max_height: None,
            hevc_features: None,
            hevc_block_sizes: None,
        },
        max_width,
        max_height,
    }
}

/// The guest's encoder list has one encoder for each codec the driver encodes a profile of, with
/// its profiles in the driver's order and the largest picture any of them takes; `Quality` maps
/// onto the profile's levels as `quality` does, and a constant QP the driver does not take is
/// the codec's default rate.
#[test]
fn the_guest_encoders_are_the_drivers_codecs() {
    let supports = [
        support(EncodeProfile::H264Main, 4096, 2304),
        support(EncodeProfile::H264High, 4096, 4096),
        support(EncodeProfile::HevcMain, 8192, 4352),
    ];
    assert_eq!(
        guest_encoders(&supports),
        vec![
            Encoder {
                codec: Codec::H264,
                profiles: vec![EncodeProfile::H264Main, EncodeProfile::H264High],
                max_width: 4096,
                max_height: 4096,
            },
            Encoder {
                codec: Codec::Hevc,
                profiles: vec![EncodeProfile::HevcMain],
                max_width: 8192,
                max_height: 4352,
            },
        ]
    );
    let ranges = vec![GuestProfile {
        profile: EncodeProfile::H264High,
        quality_range: Some(7),
        constant_qp: true,
    }];
    let config = |quality| EncoderConfig {
        settings: EncoderSettings::new(EncodeProfile::H264High, 64, 48),
        quality,
    };
    assert_eq!(guest_settings(&config(None), &ranges).quality_level, None);
    assert_eq!(
        guest_settings(&config(Some(1.0)), &ranges).quality_level,
        Some(1)
    );
    assert_eq!(
        guest_settings(&config(Some(0.0)), &ranges).quality_level,
        Some(7)
    );
    assert_eq!(
        guest_settings(&config(Some(0.5)), &ranges).quality_level,
        quality_level(50, Some(7))
    );
    // A constant QP is coded where the driver takes one, and is the codec's default rate where
    // it does not.
    let mut constant = config(Some(0.75));
    constant.settings.rate_control = RateControl::ConstantQp {
        intra: 21,
        predicted: 21,
        bidirectional: 21,
    };
    assert_eq!(
        guest_settings(&constant, &ranges).rate_control,
        constant.settings.rate_control
    );
    let no_constant_qp = [GuestProfile {
        constant_qp: false,
        ..ranges[0]
    }];
    assert_eq!(
        guest_settings(&constant, &no_constant_qp).rate_control,
        EncoderSettings::new(EncodeProfile::H264High, 64, 48).rate_control
    );
}
