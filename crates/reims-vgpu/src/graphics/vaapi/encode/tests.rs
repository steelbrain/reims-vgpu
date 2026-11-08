use reims_graphics::wire::RGFX_VIDEO_DECODE;
use reims_vgpu_video::core::runtime::StdRuntime;
use reims_vgpu_video::vaapi::{H264Decoder, HevcDecoder, encoder_support};

use super::*;

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
/// configs list; a decode entry lists none.
#[test]
fn an_encode_entry_lists_its_configs_rate_controls() {
    let entry = RgfxVideoCodecCaps {
        codec: RGFX_VIDEO_CODEC_H264,
        direction: RGFX_VIDEO_ENCODE,
        ..RgfxVideoCodecCaps::default()
    };
    assert_eq!(encode_caps(&entry, &[]), RgfxVideoEncodeCaps::default());
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
            let mut encoder =
                VaEncoder::new(&display, entry.codec, Arc::clone(&supports), entry, &rates);
            encoder.check(&s).unwrap();
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
                };
                let frame = encoder.encode(frame).unwrap();
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
                let encoded = frame.finish(&source, &mut out).unwrap();
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
