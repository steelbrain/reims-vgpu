//! Decodes an H.264 stream with [`reims_vgpu_video::vaapi::H264Decoder`]
//! and prints each frame's MD5 over its NV12 planes, packed, in the order
//! the decoder emits them. With display order this is what
//! `ffmpeg -f framemd5 -pix_fmt nv12` prints for the same stream, so a
//! software decode is the reference (tools/h264-check.sh).
//!
//! ```sh
//! h264-framemd5 /dev/dri/renderD128 stream.mp4 [options]
//! ```
//!
//! - An `.mp4` is read as VideoToolbox is handed a stream: the `avcC`
//!   record and length-prefixed samples. Anything else is an Annex B
//!   stream, split into access units.
//! - `--decode-order` emits in decode order instead of display order.
//! - `--length-size N` rewrites an MP4's samples and record to `N`-byte
//!   NAL lengths first.
//! - `--stride-pad N` reads back into planes with `N` spare bytes a row.
//! - `--hold N` keeps the last `N` frames alive, as a caller that is slow
//!   to drop them would.
//!
//! Prints `frame <index> id <sample> poc <poc> <md5>` per frame, and the
//! time spent to stderr.

#[cfg(target_os = "linux")]
use reims_vgpu_video_conformance::{md5, mp4};

#[cfg(target_os = "linux")]
#[path = "../common/readback.rs"]
mod readback;

#[cfg(target_os = "linux")]
fn main() -> std::process::ExitCode {
    readback::read_back_as_asked();
    match linux::run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("h264-framemd5: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("h264-framemd5 needs Linux and libva");
}

#[cfg(target_os = "linux")]
mod linux {
    use reims_vgpu_video::core::runtime::{Runtime, StdRuntime};
    use std::collections::VecDeque;
    use std::time::Duration;

    use reims_vgpu_video::codec::h264::{OutputOrder, avcc::DecoderConfiguration, nal};
    use reims_vgpu_video::vaapi::{Display, Frame, H264Decoder};

    use super::md5::{Md5, self_test};
    use super::mp4;

    struct Options {
        node: String,
        input: String,
        order: OutputOrder,
        length_size: Option<u8>,
        stride_pad: usize,
        hold: usize,
    }

    fn options() -> Result<Options, String> {
        let mut args = std::env::args().skip(1);
        let mut positional = Vec::new();
        let mut options = Options {
            node: String::new(),
            input: String::new(),
            order: OutputOrder::Display,
            length_size: None,
            stride_pad: 0,
            hold: 0,
        };
        while let Some(arg) = args.next() {
            let mut number = |name: &str| -> Result<usize, String> {
                args.next()
                    .and_then(|v| v.parse().ok())
                    .ok_or(format!("{name} takes a number"))
            };
            match arg.as_str() {
                "--decode-order" => options.order = OutputOrder::Decode,
                "--length-size" => {
                    let n = number("--length-size")?;
                    if !(1..=4).contains(&n) {
                        return Err("--length-size is 1 to 4".into());
                    }
                    options.length_size = Some(n as u8);
                }
                "--stride-pad" => options.stride_pad = number("--stride-pad")?,
                "--hold" => options.hold = number("--hold")?,
                _ => positional.push(arg),
            }
        }
        let [node, input] = <[String; 2]>::try_from(positional)
            .map_err(|_| "usage: h264-framemd5 <render node> <stream> [options]")?;
        options.node = node;
        options.input = input;
        Ok(options)
    }

    /// Rewrites a sample's NAL lengths from `from` bytes to `to`.
    fn relength(sample: &[u8], from: u8, to: u8) -> Result<Vec<u8>, String> {
        let mut out = Vec::with_capacity(sample.len());
        for unit in nal::LengthPrefixed::new(sample, from) {
            let unit = unit.map_err(|e| e.to_string())?;
            let len = (unit.len() as u64).to_be_bytes();
            let to = usize::from(to);
            if len[..8 - to].iter().any(|&b| b != 0) {
                return Err(format!(
                    "a {}-byte NAL unit needs a longer length",
                    unit.len()
                ));
            }
            out.extend_from_slice(&len[8 - to..]);
            out.extend_from_slice(unit);
        }
        Ok(out)
    }

    struct Reader {
        y: Vec<u8>,
        uv: Vec<u8>,
        pad: usize,
        index: usize,
        readback: Duration,
        /// The readback's stages, as the frames' pool counted them.
        stages: reims_vgpu_video::vaapi::ReadbackTimes,
    }

    impl Reader {
        fn emit(&mut self, frame: &Frame) -> Result<(), String> {
            let (w, h) = (frame.width() as usize, frame.height() as usize);
            let stride = w + self.pad;
            self.y.resize(stride * h, 0);
            self.uv.resize(stride * h / 2, 0);
            let start = StdRuntime.now();
            frame
                .copy_nv12_to(&mut self.y, stride, &mut self.uv, stride)
                .map_err(|e| e.to_string())?;
            self.readback += StdRuntime.elapsed_since(start);
            self.stages = frame.readback_times();
            let mut md5 = Md5::new();
            for row in self.y.chunks(stride).take(h) {
                md5.update(&row[..w]);
            }
            for row in self.uv.chunks(stride).take(h / 2) {
                md5.update(&row[..w]);
            }
            println!(
                "frame {} id {} poc {} {}",
                self.index,
                frame.id(),
                frame.poc(),
                md5.hex()
            );
            self.index += 1;
            Ok(())
        }
    }

    pub(crate) fn run() -> Result<(), String> {
        if !self_test() {
            return Err("MD5 fails its own test vectors".into());
        }
        let options = options()?;
        let display =
            Display::open(&options.node, StdRuntime::shared()).map_err(|e| e.to_string())?;
        let file = std::fs::read(&options.input).map_err(|e| format!("{}: {e}", options.input))?;
        let (mut decoder, samples): (H264Decoder, Vec<Vec<u8>>) = if options.input.ends_with(".mp4")
        {
            let track = mp4::h264_track(&file)?;
            let mut avcc = track.avcc.clone();
            let from = DecoderConfiguration::parse(&avcc)
                .map_err(|e| e.to_string())?
                .length_size;
            let mut samples = Vec::with_capacity(track.samples.len());
            for &(offset, size) in &track.samples {
                let sample = file
                    .get(offset..offset + size)
                    .ok_or("sample past the file's end")?;
                samples.push(match options.length_size {
                    Some(to) => relength(sample, from, to)?,
                    None => sample.to_vec(),
                });
            }
            if let Some(to) = options.length_size {
                avcc[4] = (avcc[4] & !3) | (to - 1);
            }
            let decoder = H264Decoder::new(&display, &avcc).map_err(|e| e.to_string())?;
            (decoder, samples)
        } else {
            let units = nal::access_units(&file);
            (
                H264Decoder::new_annex_b(&display),
                units.into_iter().map(<[u8]>::to_vec).collect(),
            )
        };
        decoder.set_output_order(options.order);
        let mut reader = Reader {
            y: Vec::new(),
            uv: Vec::new(),
            pad: options.stride_pad,
            index: 0,
            readback: Duration::ZERO,
            stages: reims_vgpu_video::vaapi::ReadbackTimes::default(),
        };
        let mut held = VecDeque::new();
        let start = StdRuntime.now();
        for (id, sample) in samples.iter().enumerate() {
            let frames = decoder
                .decode(sample, id as u64)
                .map_err(|e| format!("sample {id}: {e}"))?;
            for frame in frames {
                reader.emit(&frame)?;
                if options.hold > 0 {
                    held.push_back(frame);
                    if held.len() > options.hold {
                        held.pop_front();
                    }
                }
            }
        }
        for frame in decoder.flush().map_err(|e| e.to_string())? {
            reader.emit(&frame)?;
        }
        let total = StdRuntime.elapsed_since(start);
        let stages = reader.stages;
        let per_frame = |d: Duration| d.as_secs_f64() * 1e3 / stages.frames.max(1) as f64;
        eprintln!(
            "readback a frame: wait {:.2} ms, transfer {:.2} ms, copy {:.2} ms ({} frames)",
            per_frame(stages.wait),
            per_frame(stages.transfer),
            per_frame(stages.copy),
            stages.frames
        );
        eprintln!(
            "{} samples, {} frames, profile {}, {:.3} s ({:.1} fps), readback {:.3} s",
            samples.len(),
            reader.index,
            decoder.profile().and_then(|p| p.name()).unwrap_or("none"),
            total.as_secs_f64(),
            reader.index as f64 / total.as_secs_f64(),
            reader.readback.as_secs_f64()
        );
        Ok(())
    }
}
