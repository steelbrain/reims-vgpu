//! VideoToolbox's encoders on NVENC: off the GPU, the encoder list, the settings and the samples
//! of NVENC's access units; on the NVIDIA host, frames read from lent surfaces by the CPU and
//! copied by the GPU code alike and decode back.

use std::ptr::NonNull;

use reims_vgpu_video::codec::encode::nal::{self, START_CODE};
use reims_vgpu_video::codec::encode::{ColourDescription, EncoderSettings, FrameRate};
use reims_vgpu_video::core::device_memory::{DeviceMemory, HostView, MemoryIdentity, MemoryLife};
use reims_vgpu_video::core::fourcc::FourCc;
use reims_vgpu_video::core::surfaces::{PlaneLayout, SurfaceError, SurfaceLayout};
use reims_vgpu_video::nvidia::{Error, Packet};
use reims_vgpu_video::paravirt::Codec;

use super::*;
use crate::graphics::video_choice::tested_gpu::Arm;

/// NVENC's answer for `profile` up to `max_width` by `max_height`.
fn support(profile: EncodeProfile, max_width: u32, max_height: u32) -> EncodeProfileSupport {
    EncodeProfileSupport {
        profile,
        min_width: 145,
        min_height: 49,
        max_width,
        max_height,
        max_b_frames: 4,
        constant_bit_rate: true,
        variable_bit_rate: true,
        max_references: 7,
    }
}

/// The guest's encoder list has one encoder for each codec NVENC encodes a profile of, with
/// those profiles, Baseline as Constrained Baseline, and the largest picture any of them takes.
#[test]
fn the_guest_encoders_are_nvencs_codecs() {
    let supports = [
        support(EncodeProfile::H264Baseline, 4096, 4096),
        support(EncodeProfile::H264High, 4096, 4096),
        support(EncodeProfile::HevcMain, 8192, 8192),
        support(EncodeProfile::HevcMain10, 4096, 8192),
    ];
    assert_eq!(
        guest_encoders(&supports),
        [
            Encoder {
                codec: Codec::H264,
                profiles: vec![Profile::H264ConstrainedBaseline, Profile::H264High],
                max_width: 4096,
                max_height: 4096,
            },
            Encoder {
                codec: Codec::Hevc,
                profiles: vec![Profile::HevcMain, Profile::HevcMain10],
                max_width: 8192,
                max_height: 8192,
            },
        ]
    );
    assert!(guest_encoders(&[]).is_empty());
}

/// A guest's settings for `profile` at 1920x1080, with `quality`.
fn config(profile: Profile, quality: Option<f32>) -> EncoderConfig {
    EncoderConfig {
        settings: EncoderSettings::new(profile, 1920, 1080),
        quality,
    }
}

/// The NVIDIA crate's settings of a guest's: the profile, size, frame rate, level, colour, range
/// and VBV buffer as given; CABAC for H.264 Main and High unless CAVLC is asked, CAVLC for
/// Constrained Baseline, none for HEVC; no GOP and no B-frames; each rate control as NVENC's,
/// a QVBR as a VBR at its target quality, QPs held to 51; and `Quality` onto the preset and
/// tuning the device's quality takes, the middle where the guest sets none.
#[test]
fn the_nvenc_settings_of_a_guests() {
    let mut c = config(Profile::H264High, None);
    c.settings.frame_rate = FrameRate::new(60_000, 1001);
    c.settings.level = Some(42);
    c.settings.full_range = true;
    c.settings.colour = Some(ColourDescription {
        primaries: 9,
        transfer: 16,
        matrix: 9,
    });
    c.settings.buffer_milliseconds = 500;
    c.settings.key_frame_interval = Some(30);
    c.settings.sample_aspect_ratio = Some((4, 3));
    let (preset, tuning) = speed_of(DEFAULT_QUALITY);
    assert_eq!(
        params_of(&c),
        EncodeParams {
            profile: EncodeProfile::H264High,
            width: 1920,
            height: 1080,
            frame_rate: (60_000, 1001),
            rate_control: EncodeRateControl::VariableBitRate {
                average_bits_per_second: 5_000_000,
                peak_bits_per_second: 7_500_000,
                target_quality: None,
            },
            buffer_milliseconds: 500,
            gop_length: None,
            b_frames: 0,
            level: Some(42),
            cabac: Some(true),
            full_range: true,
            colour: Some((9, 16, 9)),
            display_aspect: Some((1920 * 4, 1080 * 3)),
            preset,
            tuning,
        }
    );
    c.settings.entropy = Some(Entropy::Cavlc);
    assert_eq!(params_of(&c).cabac, Some(false));
    let baseline = params_of(&config(Profile::H264ConstrainedBaseline, None));
    assert_eq!(
        (baseline.profile, baseline.cabac),
        (EncodeProfile::H264Baseline, Some(false))
    );
    let hevc = params_of(&config(Profile::HevcMain10, None));
    assert_eq!(
        (hevc.profile, hevc.cabac),
        (EncodeProfile::HevcMain10, None)
    );
    for (quality, percent) in [
        (Some(0.0), 0),
        (Some(0.25), 25),
        (Some(1.0), 100),
        (None, 50),
    ] {
        assert_eq!(
            params_of(&config(Profile::HevcMain, quality)).preset,
            speed_of(percent).0,
            "{quality:?}"
        );
    }
    assert_eq!(params_of(&config(Profile::HevcMain, Some(0.0))).preset, 1);
    assert_eq!(params_of(&config(Profile::HevcMain, Some(1.0))).preset, 7);
    for (rate, nvenc) in [
        (
            RateControl::ConstantQp {
                intra: 20,
                predicted: 52,
                bidirectional: 60,
            },
            EncodeRateControl::ConstantQp {
                intra: 20,
                predicted: 51,
                bidirectional: 51,
            },
        ),
        (
            RateControl::ConstantBitRate { bits_per_second: 0 },
            EncodeRateControl::ConstantBitRate { bits_per_second: 1 },
        ),
        (
            RateControl::VariableBitRate {
                average_bits_per_second: 4_000_000,
                peak_bits_per_second: 3_000_000,
            },
            EncodeRateControl::VariableBitRate {
                average_bits_per_second: 4_000_000,
                peak_bits_per_second: 4_000_000,
                target_quality: None,
            },
        ),
        (
            RateControl::QualityVariableBitRate {
                average_bits_per_second: 4_000_000,
                peak_bits_per_second: 6_000_000,
                quality: 23,
            },
            EncodeRateControl::VariableBitRate {
                average_bits_per_second: 4_000_000,
                peak_bits_per_second: 6_000_000,
                target_quality: Some(23),
            },
        ),
    ] {
        let mut c = config(Profile::H264Main, None);
        c.settings.rate_control = rate;
        assert_eq!(params_of(&c).rate_control, nvenc, "{rate:?}");
    }
}

/// An H.264 SPS's first bytes, of Main at `level_idc`: all the record reads of it.
fn sps(level_idc: u8) -> Vec<u8> {
    vec![0x67, 77, 0x40, level_idc, 0xac, 0x2b]
}

const PPS: &[u8] = &[0x68, 0xce, 0x3c, 0x80];
const IDR: &[u8] = &[0x65, 0x88, 0x84];
const P: &[u8] = &[0x41, 0x9a, 0x02];

/// `units` as an Annex B access unit.
fn annex_b(units: &[&[u8]]) -> Vec<u8> {
    let mut au = Vec::new();
    for unit in units {
        au.extend_from_slice(&START_CODE);
        au.extend_from_slice(unit);
    }
    au
}

/// A packet of frame `index` of `picture_type` holding `units`.
fn packet(index: u64, picture_type: PictureType, units: &[&[u8]]) -> Bitstream {
    Ok(Packet {
        data: annex_b(units),
        index,
        picture_type,
        average_qp: 26,
    })
}

/// Each packet NVENC writes is the oldest pending frame's sample, at its times, its decode time
/// its presentation time: an IDR picture's a sync sample that depends on nothing, with the
/// record on the first sample and where its parameter sets changed, a P picture's not; its
/// parameter sets are taken out. A bitstream that does not read, or holds no slice, fails its
/// frame alone and makes the next frame an IDR picture.
#[test]
fn a_packet_is_its_frames_sample_without_its_parameter_sets() {
    let first = sps(31);
    let mut samples = Samples {
        record: Record::of_annex_b(EncodeCodec::H264, &annex_b(&[&first, PPS])).expect("a record"),
        pending: VecDeque::new(),
        restart: false,
    };
    let record = samples.record.current();
    for id in 0..5 {
        samples.pending.push_back(SourceFrame {
            id,
            pts: 1000 + i64::try_from(id).expect("an id") * 3000,
            duration: 3000,
            force_key_frame: false,
        });
    }
    let coded = samples.of(vec![
        packet(0, PictureType::Idr, &[&first, PPS, IDR]),
        packet(1, PictureType::Predicted, &[P]),
        Err(Error::Unsupported("a lock".to_owned())),
    ]);
    let [Ok(idr), Ok(p), Err(failed)] = &coded[..] else {
        panic!("{coded:?}");
    };
    let mut data = Vec::new();
    nal::push_length_prefixed(&mut data, IDR);
    assert_eq!(
        idr,
        &EncodedSample {
            id: 0,
            data,
            pts: 1000,
            dts: 1000,
            duration: 3000,
            frame_type: FrameType::Idr,
            is_sync: true,
            depends_on_others: false,
            is_depended_on: true,
            configuration: Some(record),
        }
    );
    assert_eq!(
        (p.id, p.pts, p.dts, p.frame_type, p.is_sync),
        (1, 4000, 4000, FrameType::P, false)
    );
    assert!(p.depends_on_others && p.is_depended_on && p.configuration.is_none());
    assert_eq!(failed.id, 2);
    assert!(matches!(failed.error, Some(VideoError::Unsupported(_))));
    assert!(samples.restart);
    samples.restart = false;
    let new = sps(40);
    let coded = samples.of(vec![
        packet(3, PictureType::Predicted, &[PPS]),
        packet(4, PictureType::Idr, &[&new, PPS, IDR]),
    ]);
    let [Err(no_slice), Ok(changed)] = &coded[..] else {
        panic!("{coded:?}");
    };
    assert_eq!(no_slice.id, 3);
    assert!(matches!(no_slice.error, Some(VideoError::Malfunction(_))));
    assert!(samples.restart);
    assert_eq!(changed.id, 4);
    assert!(changed.configuration.as_deref().is_some_and(|r| r[3] == 40));
    assert!(
        samples
            .of(vec![packet(5, PictureType::Predicted, &[P])])
            .is_empty()
    );
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

/// The stream the hardware test encodes: 640x360.
const STREAM: (usize, usize) = (640, 360);

/// Luma sample `x`, `y` of frame `n`: a ramp that moves.
fn luma(x: usize, y: usize, n: usize) -> u8 {
    u8::try_from((x / 2 + y + n * 6) % 200 + 28).expect("a sample")
}

/// A 642x362 `420v` surface laid out as CoreVideo lays out a pool's buffer, rows 704 bytes apart
/// and chroma on the next 4 KiB, holding frame `n` of a moving picture.
fn guest_surface(n: usize) -> (SurfaceLayout, HostSurface) {
    let (w, h, pitch) = (642usize, 362usize, 704usize);
    let chroma_at = (pitch * h).next_multiple_of(0x1000);
    let mut bytes = vec![0u8; chroma_at + pitch * h / 2].into_boxed_slice();
    for y in 0..h {
        for x in 0..w {
            bytes[y * pitch + x] = luma(x, y, n);
        }
    }
    for y in 0..h / 2 {
        for x in 0..w {
            bytes[chroma_at + y * pitch + x] = 128 + u8::try_from((x / 8 + n) % 32).expect("a");
        }
    }
    let plane = |offset, width: usize, height: usize| PlaneLayout {
        offset,
        bytes_per_row: pitch,
        width: u32::try_from(width).expect("a width"),
        height: u32::try_from(height).expect("a height"),
    };
    let layout = SurfaceLayout {
        pixel_format: FourCc::new(*b"420v"),
        width: u32::try_from(w).expect("a width"),
        height: u32::try_from(h).expect("a height"),
        planes: vec![plane(0, w, h), plane(chroma_at, w / 2, h / 2)],
        bytes_per_row: 0,
    };
    let life = MemoryLife::new();
    (layout, HostSurface { life, bytes })
}

/// `sample`'s units as an Annex B access unit, the parameter sets of `record`, where it is given,
/// first.
fn access_unit(codec: Codec, record: Option<&[u8]>, sample: &[u8]) -> Vec<u8> {
    let mut sets: Vec<&[u8]> = Vec::new();
    if let Some(record) = record {
        if codec == Codec::H264 {
            let config = reims_vgpu_video::codec::h264::avcc::DecoderConfiguration::parse(record)
                .expect("an avcC");
            sets.extend(config.sequence_parameter_sets);
            sets.extend(config.picture_parameter_sets);
        } else {
            let config = reims_vgpu_video::codec::hevc::hvcc::DecoderConfiguration::parse(record)
                .expect("an hvcC");
            sets.extend(config.nal_units);
        }
    }
    sets.extend(nal::length_prefixed_units(sample).expect("a sample's units"));
    annex_b(&sets)
}

/// On the NVIDIA host, each guest encoder the arm answers, at each of its 8-bit profiles, at a
/// constant QP: 12 frames of a 640x360 stream read from `420v` surfaces wider and taller than it,
/// with a forced key frame (frame 5) and a key-frame interval of 8, and after the first frame a
/// change of the QPs, one of the profile (to the next the encoder lists, HEVC's Main10 from
/// Main) and one of the colour (to BT.2020's primaries), each taken and left out. Read by the
/// CPU from the surfaces' host bytes, and copied by the GPU from a Shared buffer of the device's
/// that holds them, every frame the GPU's way, they come out as the same samples byte for byte:
/// one a frame, in order, at its times, sync samples at frames 0 and 5 alone, the record on the
/// first alone. Each decodes back on NVDEC to a 640x360 picture whose luma is the frame's top
/// left.
#[test]
#[ignore = "needs an NVIDIA GPU: run on the NVIDIA host"]
fn a_guest_encoders_frames_code_alike_from_the_cpu_and_the_gpu_and_decode_back() {
    use reims_vgpu_video::nvidia::{AnnexBCodec, AnnexBDecoder};
    use reims_vgpu_vulkan::RtPlatform;
    use reims_vgpu_vulkan::core::platform::device::GpuDevice as _;
    use reims_vgpu_vulkan::core::platform::resource::Buffer as _;
    use reims_vgpu_vulkan::core::raise::OnFault;

    use super::super::super::super::decode::GraphicsVideo as _;
    use super::super::super::{NvidiaVideo, Shared};

    let platform = crate::rt_test_end::platform();
    crate::entry::run(|_| {
        let device = platform.system_default_device().expect("an rt device");
        crate::graphics::video_choice::tested_gpu::check_rt(&device, Arm::Nvidia);
        let runtime = crate::video::runtime::PlatformRuntime::of(&platform);
        let gpu = crate::graphics::video_choice::rt_nvidia_gpu(&device);
        let video = NvidiaVideo::<RtPlatform>::new(&device, gpu, &runtime);
        let video = Arc::new(video.expect("NVENC on the GPU"));
        assert!(video.imports.is_some(), "CUDA imports no exportable buffer");
        let encoders = Shared(Arc::clone(&video)).guest_encoders();
        assert!(
            encoders.iter().any(|e| e.codec == Codec::H264),
            "{encoders:?}"
        );
        let (width, height) = STREAM;
        let mut checked = 0;
        for encoder in &encoders {
            for &profile in encoder.profiles.iter().filter(|p| p.bit_depth() == 8) {
                let mut settings = EncoderSettings::new(
                    profile,
                    u32::try_from(width).expect("a width"),
                    u32::try_from(height).expect("a height"),
                );
                settings.timescale = 90_000;
                settings.key_frame_interval = Some(8);
                settings.rate_control = RateControl::ConstantQp {
                    intra: 22,
                    predicted: 24,
                    bidirectional: 26,
                };
                let config = EncoderConfig {
                    settings,
                    quality: Some(0.5),
                };
                let mut sharper = config.clone();
                sharper.settings.rate_control = RateControl::ConstantQp {
                    intra: 20,
                    predicted: 22,
                    bidirectional: 24,
                };
                let mut other = sharper.clone();
                other.settings.profile = *encoder
                    .profiles
                    .iter()
                    .find(|p| **p != profile)
                    .unwrap_or(&profile);
                let mut tinted = sharper.clone();
                tinted.settings.colour = Some(ColourDescription {
                    primaries: 9,
                    transfer: 1,
                    matrix: 9,
                });
                let open = |filler| {
                    NvGuestEncoder::<RtPlatform>::new(&video.gpu, &config, filler)
                        .expect("a guest encoder")
                };
                let mut by_cpu = open(Err("read by the CPU in this test".to_owned()));
                let mut by_gpu = open(video.filler());
                let (mut read, mut filled) = (Vec::new(), Vec::new());
                let mut sources = Vec::new();
                for n in 0..12usize {
                    let (layout, memory) = guest_surface(n);
                    let length = u64::try_from(memory.bytes.len()).expect("a length");
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
                        id: u64::try_from(n).expect("an id"),
                        pts: i64::try_from(n).expect("a time") * 3000,
                        duration: 3000,
                        force_key_frame: n == 5,
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
                    read.extend(by_cpu.encode(&from_host, &frame).expect("a frame taken"));
                    filled.extend(by_gpu.encode(&from_buffer, &frame).expect("a frame taken"));
                    if n == 0 {
                        for guest in [&mut by_cpu, &mut by_gpu] {
                            guest.reconfigure(&sharper).expect("a change of the QPs");
                            guest.reconfigure(&other).expect("a change of the profile");
                            guest
                                .reconfigure(&tinted)
                                .expect("a change of the primaries");
                        }
                    }
                    sources.push(memory);
                }
                read.extend(by_cpu.flush().expect("a flush"));
                filled.extend(by_gpu.flush().expect("a flush"));
                assert!(
                    by_gpu.fills.said.is_empty(),
                    "{profile:?}: frames uploaded by the CPU: {:?}",
                    by_gpu.fills.said
                );
                let read: Vec<EncodedSample> =
                    read.into_iter().map(|c| c.expect("a sample")).collect();
                let filled: Vec<EncodedSample> =
                    filled.into_iter().map(|c| c.expect("a sample")).collect();
                assert_eq!(read.len(), 12, "{profile:?}");
                assert!(
                    read == filled,
                    "{profile:?}: a filled frame's sample differs"
                );
                let sync: Vec<u64> = (read.iter()).filter(|s| s.is_sync).map(|s| s.id).collect();
                assert_eq!(sync, [0, 5], "{profile:?}");
                assert!(read[0].configuration.is_some(), "{profile:?}");
                assert!(
                    read[1..].iter().all(|s| s.configuration.is_none()),
                    "{profile:?}: the profile's change made a new sequence"
                );
                for (n, sample) in read.iter().enumerate() {
                    let id = u64::try_from(n).expect("an id");
                    let at = i64::try_from(n).expect("a time") * 3000;
                    assert_eq!((sample.id, sample.pts, sample.dts), (id, at, at));
                }
                let mut decoder = AnnexBDecoder::new(
                    &video.gpu,
                    match encoder.codec {
                        Codec::H264 => AnnexBCodec::H264,
                        Codec::Hevc => AnnexBCodec::Hevc,
                        Codec::Av1 | Codec::Vp9 => unreachable!("no arm encodes AV1 or VP9"),
                    },
                    &runtime,
                );
                for (n, sample) in read.iter().enumerate() {
                    let sets = sample.configuration.as_deref();
                    let au = access_unit(encoder.codec, sets, &sample.data);
                    let picture = (decoder.decode(&au)).expect("a decode").expect("a picture");
                    let back = (picture.read_back((640, 360))).expect("a read back");
                    let planes = back.planes();
                    let y = &planes[0];
                    assert_eq!(y.rows, 360, "{profile:?}: frame {n}");
                    let mut squared = 0f64;
                    for row in 0..height {
                        for x in 0..width {
                            let got = f64::from(y.bytes[row * y.bytes_per_row + x]);
                            let want = f64::from(luma(x, row, n));
                            squared += (got - want) * (got - want);
                        }
                    }
                    let mse = squared / f64::from(u32::try_from(width * height).expect("samples"));
                    let psnr = 10.0 * (255.0 * 255.0 / mse.max(1e-9)).log10();
                    assert!(psnr >= 35.0, "{profile:?}: frame {n}: luma at {psnr:.1} dB");
                }
                drop(sources);
                checked += 1;
            }
        }
        assert!(checked > 0);
    });
}
