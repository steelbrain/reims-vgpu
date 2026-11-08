//! Encodes raw frames on NVENC to an Annex B file, and prints a line for
//! each packet: its frame, picture type, size and average QP. The frames
//! come from a raw NV12 (8-bit) or P010 (Main 10) file, as ffmpeg's
//! `-f rawvideo -pix_fmt nv12` (or `p010le`) writes it; the check decodes the
//! output back with ffmpeg against that file.
//!
//! ```sh
//! nvidia-encode PROFILE WIDTH HEIGHT INPUT OUTPUT [OPTION...]
//!   PROFILE        h264-baseline h264-main h264-high hevc-main hevc-main10
//!   --gpu SEL      a CUDA ordinal, a PCI bus ID, or a DRM node (renderD128)
//!   --fps N/D      frames a second (30/1)
//!   --bframes N    B pictures between references (0)
//!   --gop N        frames from one IDR picture to the next (none: only the
//!                  first, and those --idr asks for)
//!   --rc MODE      cqp:I,P,B | cbr:BPS | vbr:AVG,PEAK[,QUALITY] (cqp:24,26,28)
//!   --idr LIST     frames to force as IDR pictures, by comma
//!   --preset N     P1 to P7 (4)
//!   --tuning T     hq or ll (hq)
//!   --cavlc        H.264's CAVLC rather than CABAC
//!   --frames N     encode at most N frames
//!   --rc-at N:MODE reconfigure the rate control from frame N
//!   --params FILE  write the parameter sets NVENC answers, Annex B
//! ```

#[cfg(target_os = "linux")]
mod linux {
    use std::fs::File;
    use std::io::{BufWriter, Read, Write};
    use std::process::ExitCode;
    use std::sync::Arc;

    use reims_vgpu_video::nvidia::{
        Bitstream, EncodeParams, EncodeProfile, EncodeRateControl, Gpu, GpuSelector, NvencEncoder,
        RawFrame, Tuning,
    };

    type Failure = Box<dyn std::error::Error>;

    struct Options {
        gpu: GpuSelector,
        params: EncodeParams,
        input: String,
        output: String,
        idr: Vec<u64>,
        frames: Option<u64>,
        rc_at: Option<(u64, EncodeRateControl)>,
        sequence_parameters: Option<String>,
    }

    pub(super) fn main() -> ExitCode {
        match run() {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                let message = e.to_string();
                if !message.is_empty() {
                    eprintln!("nvidia-encode: {message}");
                }
                ExitCode::FAILURE
            }
        }
    }

    fn parse_rate_control(text: &str) -> Result<EncodeRateControl, Failure> {
        let (mode, values) = text.split_once(':').ok_or("--rc wants MODE:VALUES")?;
        let values: Vec<u32> = values
            .split(',')
            .map(str::parse)
            .collect::<Result<_, _>>()?;
        Ok(match (mode, values.as_slice()) {
            ("cqp", [intra, predicted, bidirectional]) => EncodeRateControl::ConstantQp {
                intra: *intra,
                predicted: *predicted,
                bidirectional: *bidirectional,
            },
            ("cbr", [bits]) => EncodeRateControl::ConstantBitRate {
                bits_per_second: *bits,
            },
            ("vbr", [average, peak]) => EncodeRateControl::VariableBitRate {
                average_bits_per_second: *average,
                peak_bits_per_second: *peak,
                target_quality: None,
            },
            ("vbr", [average, peak, quality]) => EncodeRateControl::VariableBitRate {
                average_bits_per_second: *average,
                peak_bits_per_second: *peak,
                target_quality: Some(u8::try_from(*quality)?),
            },
            _ => return Err(format!("no rate control {text}").into()),
        })
    }

    fn parse() -> Result<Options, Failure> {
        let mut args = std::env::args().skip(1);
        let mut next = |what: &str| args.next().ok_or_else(|| format!("missing {what}"));
        let profile = match next("PROFILE")?.as_str() {
            "h264-baseline" => EncodeProfile::H264Baseline,
            "h264-main" => EncodeProfile::H264Main,
            "h264-high" => EncodeProfile::H264High,
            "hevc-main" => EncodeProfile::HevcMain,
            "hevc-main10" => EncodeProfile::HevcMain10,
            other => return Err(format!("no profile {other}").into()),
        };
        let width = next("WIDTH")?.parse()?;
        let height = next("HEIGHT")?.parse()?;
        let input = next("INPUT")?;
        let output = next("OUTPUT")?;
        let mut options = Options {
            gpu: GpuSelector::First,
            params: EncodeParams {
                profile,
                width,
                height,
                frame_rate: (30, 1),
                rate_control: EncodeRateControl::ConstantQp {
                    intra: 24,
                    predicted: 26,
                    bidirectional: 28,
                },
                buffer_milliseconds: 1000,
                gop_length: None,
                b_frames: 0,
                level: None,
                cabac: None,
                full_range: false,
                colour: Some((1, 1, 1)),
                display_aspect: None,
                preset: 4,
                tuning: Tuning::HighQuality,
            },
            input,
            output,
            idr: Vec::new(),
            frames: None,
            rc_at: None,
            sequence_parameters: None,
        };
        while let Ok(flag) = next("an option") {
            match flag.as_str() {
                "--gpu" => options.gpu = GpuSelector::parse(&next("--gpu")?),
                "--fps" => {
                    let text = next("--fps")?;
                    let (n, d) = text.split_once('/').ok_or("--fps wants N/D")?;
                    options.params.frame_rate = (n.parse()?, d.parse()?);
                }
                "--bframes" => options.params.b_frames = next("--bframes")?.parse()?,
                "--gop" => options.params.gop_length = Some(next("--gop")?.parse()?),
                "--rc" => options.params.rate_control = parse_rate_control(&next("--rc")?)?,
                "--idr" => {
                    options.idr = next("--idr")?
                        .split(',')
                        .map(str::parse)
                        .collect::<Result<_, _>>()?;
                }
                "--preset" => options.params.preset = next("--preset")?.parse()?,
                "--tuning" => {
                    options.params.tuning = match next("--tuning")?.as_str() {
                        "hq" => Tuning::HighQuality,
                        "ll" => Tuning::LowLatency,
                        other => return Err(format!("no tuning {other}").into()),
                    }
                }
                "--cavlc" => options.params.cabac = Some(false),
                "--frames" => options.frames = Some(next("--frames")?.parse()?),
                "--rc-at" => {
                    let text = next("--rc-at")?;
                    let (frame, mode) = text.split_once(':').ok_or("--rc-at wants N:MODE")?;
                    options.rc_at = Some((frame.parse()?, parse_rate_control(mode)?));
                }
                "--params" => options.sequence_parameters = Some(next("--params")?),
                other => return Err(format!("no option {other}").into()),
            }
        }
        Ok(options)
    }

    /// Writes each packet; the first bitstream that does not read fails the
    /// run.
    fn write_packets(
        out: &mut impl Write,
        bitstreams: Vec<Bitstream>,
        written: &mut u64,
    ) -> Result<(), Failure> {
        for packet in bitstreams {
            let packet = packet?;
            out.write_all(&packet.data)?;
            println!(
                "packet frame={} type={:?} bytes={} qp={}",
                packet.index,
                packet.picture_type,
                packet.data.len(),
                packet.average_qp
            );
            *written += 1;
        }
        Ok(())
    }

    fn run() -> Result<(), Failure> {
        let options = parse()?;
        let gpu = Arc::new(Gpu::open(&options.gpu)?);
        let mut encoder = NvencEncoder::new(gpu, &options.params)?;
        // Said before the encoder is dropped, so that a failure is reported
        // even where the driver's teardown fails after it.
        let result = encode(&options, &mut encoder);
        if let Err(e) = &result {
            eprintln!("nvidia-encode: {e}");
        }
        drop(encoder);
        // Said already: main says nothing more.
        result.map_err(|_| Failure::from(""))
    }

    fn encode(options: &Options, encoder: &mut NvencEncoder) -> Result<(), Failure> {
        let params = &options.params;
        if let Some(path) = &options.sequence_parameters {
            std::fs::write(path, encoder.sequence_parameters())?;
        }
        let sample = if params.profile.bit_depth() > 8 { 2 } else { 1 };
        let width = usize::try_from(params.width)?;
        let height = usize::try_from(params.height)?;
        let stride = width * sample;
        let luma_bytes = stride * height;
        let chroma_bytes = stride * height.div_ceil(2);
        let mut input = File::open(&options.input)?;
        let mut out = BufWriter::new(File::create(&options.output)?);
        let mut frame = vec![0u8; luma_bytes + chroma_bytes];
        let (mut index, mut written) = (0u64, 0u64);
        while options.frames.is_none_or(|limit| index < limit) {
            match input.read_exact(&mut frame) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e.into()),
            }
            if let Some((at, rate_control)) = options.rc_at
                && at == index
            {
                encoder.reconfigure(rate_control, params.frame_rate)?;
                eprintln!("reconfigured at frame {index}: {rate_control:?}");
            }
            let (luma, chroma) = frame.split_at(luma_bytes);
            let raw = RawFrame {
                luma: (luma, stride),
                chroma: (chroma, stride),
                index,
                force_idr: index == 0 || options.idr.contains(&index),
            };
            let packets = encoder.encode(&raw)?;
            write_packets(&mut out, packets, &mut written)?;
            index += 1;
        }
        let packets = encoder.flush()?;
        write_packets(&mut out, packets, &mut written)?;
        out.flush()?;
        eprintln!("{index} frames in, {written} packets out");
        if written != index {
            return Err(format!("{index} frames gave {written} packets").into());
        }
        Ok(())
    }
}

#[cfg(target_os = "linux")]
fn main() -> std::process::ExitCode {
    linux::main()
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("nvidia-encode needs Linux and NVIDIA's driver");
}
