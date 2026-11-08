//! Encodes raw NV12 or P010 frames with
//! [`reims_vgpu_video::vaapi::VideoEncoder`], and writes what came out three
//! ways, to check against ffmpeg:
//!
//! - `PREFIX.mp4`, the samples as VideoToolbox gets them (length-prefixed,
//!   with the configuration record in the sample entry, the time stamps,
//!   and the sync samples);
//! - `PREFIX.264` or `PREFIX.265`, Annex B, with the record's parameter
//!   sets before each sync sample;
//! - `PREFIX.log`, a line a sample: `sample N id ID type T pts P dts D sync
//!   S depends D depended D bytes B config C`;
//! - `PREFIX.record`, the configuration record of the parameter sets the
//!   hardware wrote, and `PREFIX.packed.record`, the one of those we packed,
//!   from before the first frame: they differ where the driver rewrites
//!   our headers.
//!
//! ```sh
//! vaapi-encode /dev/dri/renderD128 input.yuv 1920x1080 PREFIX [options]
//! ```
//!
//! Options: `--profile cb|main|high|hevc|hevc10` (high), `--fps N[/D]` (30),
//! `--frames N` (all), `--cqp I,P,B`, `--cbr BPS`, `--vbr AVG,PEAK`, `--qvbr
//! AVG,PEAK,Q` (`--vbr 5000000,7500000`), `--keyint N`, `--keyint-ticks N`,
//! `--bframes N`, `--no-reorder`, `--max-delay N`, `--force N,N,...`, `--level
//! N`, `--colour P,T,M`, `--full-range`, `--sar H:V`, `--timescale N`
//! (90000), `--quality N`, `--upload derive|put`, `--set-bitrate FRAME,AVG,PEAK`
//! (a mid-stream change), `--cavlc`.
//!
//! Prints the encoder's choices, its speed, and the DRM engine counters of
//! its own render node from `/proc/self/fdinfo` to stderr.

#[cfg(target_os = "linux")]
use reims_vgpu_video_conformance::mp4_write as mp4;

#[cfg(target_os = "linux")]
fn main() -> std::process::ExitCode {
    match linux::run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("vaapi-encode: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("vaapi-encode needs Linux and libva");
}

#[cfg(target_os = "linux")]
mod linux {
    use std::fs::File;
    use std::io::{BufWriter, Read, Write};

    use reims_vgpu_video::codec::encode::{
        Codec, Coded, ColourDescription, EncodedSample, EncoderSettings, Entropy, FrameRate,
        FrameType, InputFrame, Plane, Profile, RateControl, nal,
    };
    use reims_vgpu_video::core::runtime::{Runtime, StdRuntime};
    use reims_vgpu_video::vaapi::{Display, UploadMethod, VideoEncoder};

    use super::mp4;

    type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

    /// The samples, or the first frame's failure: a run checks every frame.
    fn all_coded<E: std::error::Error + 'static>(
        coded: Vec<Coded<E>>,
    ) -> Result<Vec<EncodedSample>> {
        coded
            .into_iter()
            .map(|coded| {
                coded.map_err(|failed| -> Box<dyn std::error::Error> {
                    match failed.error {
                        Some(error) => Box::new(error),
                        None => format!("frame {} was lost with its run", failed.id).into(),
                    }
                })
            })
            .collect()
    }

    fn numbers<T: std::str::FromStr>(text: &str) -> Result<Vec<T>> {
        text.split(',')
            .map(|n| {
                n.parse()
                    .map_err(|_| format!("bad number in {text}").into())
            })
            .collect()
    }

    struct Options {
        node: String,
        input: String,
        width: u32,
        height: u32,
        prefix: String,
        frames: Option<u64>,
        settings: EncoderSettings,
        force: Vec<u64>,
        upload: Option<UploadMethod>,
        set_bitrate: Option<(u64, u32, u32)>,
    }

    fn parse() -> Result<Options> {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let [node, input, size, prefix, rest @ ..] = &args[..] else {
            return Err("usage: vaapi-encode NODE INPUT WxH PREFIX [options]".into());
        };
        let (w, h) = size.split_once('x').ok_or("size is WxH")?;
        let (width, height) = (w.parse()?, h.parse()?);
        let mut profile = Profile::H264High;
        let mut it = rest.iter();
        let mut options: Vec<(String, Option<String>)> = Vec::new();
        while let Some(flag) = it.next() {
            let takes = !matches!(flag.as_str(), "--no-reorder" | "--full-range" | "--cavlc");
            let value = if takes { it.next().cloned() } else { None };
            if takes && value.is_none() {
                return Err(format!("{flag} needs a value").into());
            }
            if flag == "--profile" {
                profile = match value.as_deref() {
                    Some("cb") => Profile::H264ConstrainedBaseline,
                    Some("main") => Profile::H264Main,
                    Some("high") => Profile::H264High,
                    Some("hevc") => Profile::HevcMain,
                    Some("hevc10") => Profile::HevcMain10,
                    _ => return Err("--profile is cb, main, high, hevc or hevc10".into()),
                };
            } else {
                options.push((flag.clone(), value));
            }
        }
        let mut settings = EncoderSettings::new(profile, width, height);
        let mut o = Options {
            node: node.clone(),
            input: input.clone(),
            width,
            height,
            prefix: prefix.clone(),
            frames: None,
            settings: settings.clone(),
            force: Vec::new(),
            upload: None,
            set_bitrate: None,
        };
        for (flag, value) in options {
            let v = value.unwrap_or_default();
            match flag.as_str() {
                "--fps" => {
                    let (n, d) = v.split_once('/').unwrap_or((&v, "1"));
                    settings.frame_rate = FrameRate::new(n.parse()?, d.parse()?);
                }
                "--frames" => o.frames = Some(v.parse()?),
                "--cqp" => {
                    let q: Vec<u8> = numbers(&v)?;
                    let [intra, predicted, bidirectional] = q[..] else {
                        return Err("--cqp I,P,B".into());
                    };
                    settings.rate_control = RateControl::ConstantQp {
                        intra,
                        predicted,
                        bidirectional,
                    };
                }
                "--cbr" => {
                    settings.rate_control = RateControl::ConstantBitRate {
                        bits_per_second: v.parse()?,
                    };
                }
                "--vbr" => {
                    let r: Vec<u32> = numbers(&v)?;
                    let [average_bits_per_second, peak_bits_per_second] = r[..] else {
                        return Err("--vbr AVG,PEAK".into());
                    };
                    settings.rate_control = RateControl::VariableBitRate {
                        average_bits_per_second,
                        peak_bits_per_second,
                    };
                }
                "--qvbr" => {
                    let r: Vec<u32> = numbers(&v)?;
                    let [average_bits_per_second, peak_bits_per_second, quality] = r[..] else {
                        return Err("--qvbr AVG,PEAK,Q".into());
                    };
                    settings.rate_control = RateControl::QualityVariableBitRate {
                        average_bits_per_second,
                        peak_bits_per_second,
                        quality: u8::try_from(quality)?,
                    };
                }
                "--keyint" => settings.key_frame_interval = Some(v.parse()?).filter(|&n| n > 0),
                "--keyint-ticks" => settings.key_frame_interval_duration = Some(v.parse()?),
                "--bframes" => settings.b_frames = Some(v.parse()?),
                "--no-reorder" => settings.allow_frame_reordering = false,
                "--max-delay" => settings.max_frame_delay = Some(v.parse()?),
                "--force" => o.force = numbers(&v)?,
                "--level" => settings.level = Some(v.parse()?),
                "--colour" => {
                    let c: Vec<u8> = numbers(&v)?;
                    let [primaries, transfer, matrix] = c[..] else {
                        return Err("--colour P,T,M".into());
                    };
                    settings.colour = Some(ColourDescription {
                        primaries,
                        transfer,
                        matrix,
                    });
                }
                "--full-range" => settings.full_range = true,
                "--sar" => {
                    let (h, v) = v.split_once(':').ok_or("--sar H:V")?;
                    settings.sample_aspect_ratio = Some((h.parse()?, v.parse()?));
                }
                "--timescale" => settings.timescale = v.parse()?,
                "--quality" => settings.quality_level = Some(v.parse()?),
                "--upload" => {
                    o.upload = Some(match v.as_str() {
                        "derive" => UploadMethod::Derive,
                        "put" => UploadMethod::PutImage,
                        _ => return Err("--upload derive|put".into()),
                    });
                }
                "--set-bitrate" => {
                    let r: Vec<u64> = numbers(&v)?;
                    let [frame, average, peak] = r[..] else {
                        return Err("--set-bitrate FRAME,AVG,PEAK".into());
                    };
                    o.set_bitrate = Some((frame, u32::try_from(average)?, u32::try_from(peak)?));
                }
                "--cavlc" => settings.entropy = Some(Entropy::Cavlc),
                other => return Err(format!("unknown option {other}").into()),
            }
        }
        o.settings = settings;
        Ok(o)
    }

    fn type_name(t: FrameType) -> &'static str {
        match t {
            FrameType::Idr => "I",
            FrameType::P => "P",
            FrameType::B => "B",
        }
    }

    /// The record's parameter sets, as Annex B.
    fn record_units(codec: Codec, record: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut put = |unit: &[u8]| {
            out.extend_from_slice(&nal::START_CODE);
            out.extend_from_slice(unit);
        };
        let length = |at: usize| -> Option<usize> {
            Some(usize::from(u16::from_be_bytes([
                *record.get(at)?,
                *record.get(at + 1)?,
            ])))
        };
        match codec {
            Codec::H264 => {
                // avcC: SPS count at 5, then the SPSes, the PPS count, PPSes.
                let mut at = 6;
                for count_at in [5usize, usize::MAX] {
                    let count = if count_at == 5 {
                        usize::from(record.get(5).copied().unwrap_or(0) & 0x1f)
                    } else {
                        let c = usize::from(record.get(at).copied().unwrap_or(0));
                        at += 1;
                        c
                    };
                    for _ in 0..count {
                        let Some(n) = length(at) else { return out };
                        if let Some(unit) = record.get(at + 2..at + 2 + n) {
                            put(unit);
                        }
                        at += 2 + n;
                    }
                }
            }
            Codec::Hevc => {
                let arrays = usize::from(record.get(22).copied().unwrap_or(0));
                let mut at = 23;
                for _ in 0..arrays {
                    let Some(count) = length(at + 1) else {
                        return out;
                    };
                    at += 3;
                    for _ in 0..count {
                        let Some(n) = length(at) else { return out };
                        if let Some(unit) = record.get(at + 2..at + 2 + n) {
                            put(unit);
                        }
                        at += 2 + n;
                    }
                }
            }
        }
        out
    }

    /// The DRM engine counters of this process's render node, from fdinfo.
    fn engine_counters(node: &str) -> String {
        let Ok(entries) = std::fs::read_dir("/proc/self/fd") else {
            return String::new();
        };
        let mut out = Vec::new();
        for entry in entries.flatten() {
            let Ok(target) = std::fs::read_link(entry.path()) else {
                continue;
            };
            if target.to_string_lossy() != node {
                continue;
            }
            let fd = entry.file_name().to_string_lossy().into_owned();
            let info =
                std::fs::read_to_string(format!("/proc/self/fdinfo/{fd}")).unwrap_or_default();
            let lines: Vec<String> = info
                .lines()
                .filter(|l| l.starts_with("drm-engine-") || l.starts_with("drm-client-id"))
                .map(|l| l.split_whitespace().collect::<Vec<_>>().join(" "))
                .collect();
            out.push(format!("fd {fd}: {}", lines.join(", ")));
        }
        out.join("; ")
    }

    pub(crate) fn run() -> Result<()> {
        let o = parse()?;
        let display = Display::open(&o.node, StdRuntime::shared())?;
        let settings = o.settings.clone();
        let codec = settings.profile.codec();
        let mut encoder = VideoEncoder::new(&display, settings.clone())?;
        if let Some(method) = o.upload {
            encoder.force_upload_method(method);
        }
        std::fs::write(
            format!("{}.packed.record", o.prefix),
            &encoder.configuration()[..],
        )?;
        let (va_profile, entrypoint) = encoder.va_profile();
        eprintln!(
            "encoder: {:?} on VA {} / {}, coded {:?}, level {}, B {}, packed headers {:#x}",
            settings.profile,
            va_profile.name().unwrap_or("?"),
            entrypoint.name().unwrap_or("?"),
            encoder.coded_size(),
            encoder.level_idc(),
            encoder.b_frames(),
            encoder.packed_headers(),
        );
        let sample_bytes = settings.input_format().bytes_per_sample();
        let row = usize::try_from(o.width)? * sample_bytes;
        let luma = row * usize::try_from(o.height)?;
        let frame_bytes = luma + luma / 2;
        let mut input = File::open(&o.input)?;
        let ticks = i64::from(settings.timescale) * i64::from(settings.frame_rate.denominator)
            / i64::from(settings.frame_rate.numerator);
        let mut buffer = vec![0u8; frame_bytes];
        let mut samples: Vec<EncodedSample> = Vec::new();
        let mut frames = 0u64;
        let started = StdRuntime.now();
        loop {
            if o.frames.is_some_and(|limit| frames >= limit) {
                break;
            }
            match input.read_exact(&mut buffer) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e.into()),
            }
            if let Some((at, average, peak)) = o.set_bitrate
                && at == frames
            {
                encoder.set_rate_control(match settings.rate_control {
                    RateControl::ConstantBitRate { .. } => RateControl::ConstantBitRate {
                        bits_per_second: average,
                    },
                    RateControl::QualityVariableBitRate { quality, .. } => {
                        RateControl::QualityVariableBitRate {
                            average_bits_per_second: average,
                            peak_bits_per_second: peak,
                            quality,
                        }
                    }
                    _ => RateControl::VariableBitRate {
                        average_bits_per_second: average,
                        peak_bits_per_second: peak,
                    },
                })?;
            }
            let pts = i64::try_from(frames)? * ticks;
            let frame = InputFrame {
                planes: [
                    Plane {
                        data: &buffer[..luma],
                        bytes_per_row: row,
                    },
                    Plane {
                        data: &buffer[luma..],
                        bytes_per_row: row,
                    },
                ],
                pts,
                duration: ticks,
                force_key_frame: o.force.contains(&frames),
                id: frames,
            };
            samples.extend(all_coded(encoder.encode(&frame)?)?);
            frames += 1;
        }
        samples.extend(all_coded(encoder.flush())?);
        let elapsed = StdRuntime.elapsed_since(started).as_secs_f64();
        eprintln!(
            "encoded {frames} frames into {} samples in {elapsed:.3} s: {:.1} fps, upload {:?}",
            samples.len(),
            frames as f64 / elapsed.max(1e-9),
            encoder.upload_method(),
        );
        let bytes: usize = samples.iter().map(|s| s.data.len()).sum();
        let seconds = frames as f64 * f64::from(settings.frame_rate.denominator)
            / f64::from(settings.frame_rate.numerator);
        eprintln!(
            "bytes {bytes}, {:.0} bit/s over {seconds:.3} s",
            bytes as f64 * 8.0 / seconds.max(1e-9)
        );
        eprintln!("engines: {}", engine_counters(&o.node));

        let mut log = BufWriter::new(File::create(format!("{}.log", o.prefix))?);
        let mut annex_b = BufWriter::new(File::create(format!(
            "{}.{}",
            o.prefix,
            if codec == Codec::H264 { "264" } else { "265" }
        ))?);
        let mut record = encoder.configuration();
        let mut configs = 0;
        for (n, s) in samples.iter().enumerate() {
            if let Some(c) = &s.configuration {
                record = c.clone();
                configs += 1;
            }
            writeln!(
                log,
                "sample {n} id {} type {} pts {} dts {} sync {} depends {} depended {} bytes {} config {}",
                s.id,
                type_name(s.frame_type),
                s.pts,
                s.dts,
                u8::from(s.is_sync),
                u8::from(s.depends_on_others),
                u8::from(s.is_depended_on),
                s.data.len(),
                u8::from(s.configuration.is_some()),
            )?;
            if s.is_sync {
                annex_b.write_all(&record_units(codec, &record))?;
            }
            for unit in nal::length_prefixed_units(&s.data).ok_or("a bad sample")? {
                annex_b.write_all(&nal::START_CODE)?;
                annex_b.write_all(unit)?;
            }
        }
        log.flush()?;
        annex_b.flush()?;
        eprintln!(
            "configuration records: {configs}, last {} bytes",
            record.len()
        );
        let (entry, record_box) = match codec {
            Codec::H264 => (b"avc1", b"avcC"),
            Codec::Hevc => (b"hvc1", b"hvcC"),
        };
        let track: Vec<mp4::Sample> = samples
            .into_iter()
            .map(|s| mp4::Sample {
                data: s.data,
                pts: s.pts,
                dts: s.dts,
                duration: s.duration,
                sync: s.is_sync,
            })
            .collect();
        let file = mp4::write(
            entry,
            record_box,
            &record,
            o.width,
            o.height,
            settings.timescale,
            &track,
        );
        std::fs::write(format!("{}.mp4", o.prefix), file)?;
        std::fs::write(format!("{}.record", o.prefix), &record[..])?;
        Ok(())
    }
}
