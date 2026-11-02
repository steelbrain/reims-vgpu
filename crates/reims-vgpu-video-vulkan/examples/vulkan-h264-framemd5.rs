//! Decodes an H.264 stream with [`reims_vgpu_video_vulkan::H264Decoder`]
//! on the device [`VideoDevice::create`] picks, and prints each frame's MD5
//! over its cropped NV12 planes, packed, in the order the decoder emits
//! them. In display order this is what `ffmpeg -f framemd5 -pix_fmt nv12`
//! prints for the same stream, so a software decode is the reference
//! (tools/h264-check.sh), as for the VA-API backend's h264-framemd5.
//!
//! ```sh
//! vulkan-h264-framemd5 stream.mp4 [--decode-order] [--length-size N] [--hold N] [--copy-on-graphics]
//! vulkan-h264-framemd5 --capabilities
//! ```
//!
//! - An `.mp4` is read as VideoToolbox is handed a stream: the `avcC`
//!   record and length-prefixed samples. Anything else is an Annex B
//!   stream, split into access units.
//! - `--decode-order` emits in decode order instead of display order.
//! - `--length-size N` rewrites an MP4's samples and record to `N`-byte
//!   NAL lengths first.
//! - `--hold N` keeps the last `N` frames alive, as a caller that is slow
//!   to drop them would.
//! - `--copy-on-graphics` hands the device to the decoder as another's
//!   ([`VideoDevice::from_raw`]) with its graphics queue as the copy
//!   queue, so that pictures decoded into DPB pictures alone are copied
//!   out on a queue of their own, behind a semaphore wait, as on a device
//!   whose decode family does not copy.
//! - `--capabilities` prints what the device answers for each profile.
//!
//! Prints `frame <index> id <sample> poc <poc> <md5>` per frame, and the
//! time spent to stderr.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use reims_vgpu_video_conformance::md5::{Md5, self_test};
use reims_vgpu_video_conformance::mp4;
use reims_vgpu_video_core::runtime::{Runtime, StdRuntime};
use reims_vgpu_video_vulkan::avcc::DecoderConfiguration;
use reims_vgpu_video_vulkan::{
    DecodeImage, DecodeImages, Frame, H264Decoder, OutputOrder, RawDevice, VideoDevice,
    h264_capabilities, nal,
};

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("vulkan-h264-framemd5: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

struct Options {
    input: String,
    order: OutputOrder,
    length_size: Option<u8>,
    hold: usize,
    copy_on_graphics: bool,
    capabilities: bool,
}

fn options() -> Result<Options, String> {
    let mut args = std::env::args().skip(1);
    let mut positional = Vec::new();
    let mut options = Options {
        input: String::new(),
        order: OutputOrder::Display,
        length_size: None,
        hold: 0,
        copy_on_graphics: false,
        capabilities: false,
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
                options.length_size = Some(
                    u8::try_from(n)
                        .ok()
                        .filter(|n| (1..=4).contains(n))
                        .ok_or("--length-size is 1 to 4")?,
                );
            }
            "--hold" => options.hold = number("--hold")?,
            "--copy-on-graphics" => options.copy_on_graphics = true,
            "--capabilities" => options.capabilities = true,
            _ => positional.push(arg),
        }
    }
    if options.capabilities {
        return Ok(options);
    }
    let [input] = <[String; 1]>::try_from(positional)
        .map_err(|_| "usage: vulkan-h264-framemd5 <stream> [options]")?;
    options.input = input;
    Ok(options)
}

/// Rewrites a sample's NAL lengths from `from` bytes to `to`.
fn relength(sample: &[u8], from: u8, to: u8) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(sample.len());
    for unit in nal::LengthPrefixed::new(sample, from) {
        let unit = unit.map_err(|e| e.to_string())?;
        let len = u64::try_from(unit.len()).unwrap_or(u64::MAX).to_be_bytes();
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
    device: Arc<VideoDevice>,
    index: usize,
    readback: Duration,
}

impl Reader {
    /// Prints the MD5 of the frame's cropped planes.
    fn emit(&mut self, frame: &Frame<DecodeImage>) -> Result<(), String> {
        let start = StdRuntime.now();
        let planes = frame
            .picture
            .read_back(&self.device)
            .map_err(|e| e.to_string())?;
        self.readback += StdRuntime.elapsed_since(start);
        let (x, y, w, h) = frame.crop;
        let [x, y, w, h, stride] =
            [x, y, w, h, planes.width].map(|v| usize::try_from(v).unwrap_or(0));
        let mut md5 = Md5::new();
        for row in planes.y.chunks(stride).skip(y).take(h) {
            md5.update(row.get(x..x + w).ok_or("a crop past the frame")?);
        }
        for row in planes.uv.chunks(stride).skip(y / 2).take(h / 2) {
            md5.update(row.get(x..x + w).ok_or("a crop past the frame")?);
        }
        println!(
            "frame {} id {} poc {} {}",
            self.index,
            frame.id,
            frame.poc,
            md5.hex()
        );
        self.index += 1;
        Ok(())
    }
}

fn print_capabilities(device: &Arc<VideoDevice>) {
    println!(
        "{}: decode family {}, graphics family {}, copies on {}, unified image layouts {}",
        device.name(),
        device.decode_family(),
        device.graphics_family(),
        if device.copies_on_decode_queue() {
            "the decode queue".to_string()
        } else {
            format!("family {}", device.copy_family())
        },
        device.unified_image_layouts()
    );
    for caps in h264_capabilities(device) {
        println!(
            "profile {}: level_idc {}, {}x{} to {}x{}, granularity {}x{}, {} DPB slots, {} active \
             references, coincide {}, distinct {}, decodes into outputs {}, bitstream offset {} \
             size {} alignment",
            caps.profile.profile_idc,
            caps.max_level_idc,
            caps.min_coded_extent.width,
            caps.min_coded_extent.height,
            caps.max_coded_extent.width,
            caps.max_coded_extent.height,
            caps.picture_access_granularity.width,
            caps.picture_access_granularity.height,
            caps.max_dpb_slots,
            caps.max_active_reference_pictures,
            caps.dpb_and_output_coincide,
            caps.dpb_and_output_distinct,
            caps.decodes_into_output,
            caps.min_bitstream_buffer_offset_alignment,
            caps.min_bitstream_buffer_size_alignment,
        );
    }
}

fn run() -> Result<(), String> {
    if !self_test() {
        return Err("MD5 fails its own test vectors".into());
    }
    let options = options()?;
    let owner = VideoDevice::create().map_err(|e| e.to_string())?;
    if options.capabilities {
        print_capabilities(&owner);
        return Ok(());
    }
    let device = if options.copy_on_graphics {
        let graphics = owner
            .graphics_queue()
            .ok_or("a device of our own has a graphics queue")?;
        let raw = RawDevice {
            entry: owner.entry().clone(),
            instance: owner.instance().clone(),
            physical_device: owner.physical_device(),
            device: owner.device().clone(),
            decode_family: owner.decode_family(),
            decode_queue: owner.decode_queue(),
            copy_family: owner.graphics_family(),
            copy_queue: graphics,
            graphics_family: owner.graphics_family(),
            unified_image_layouts_video: owner.unified_image_layouts(),
        };
        // SAFETY: `owner` made every handle as RawDevice says, and outlives
        // `device`, which goes before it at the end of this function along
        // with everything made on it; nothing else submits to the queues.
        unsafe { VideoDevice::from_raw(raw) }
    } else {
        Arc::clone(&owner)
    };
    let file = std::fs::read(&options.input).map_err(|e| format!("{}: {e}", options.input))?;
    let images = DecodeImages::new(&device);
    let (mut decoder, samples): (H264Decoder<DecodeImages>, Vec<Vec<u8>>) =
        if options.input.ends_with(".mp4") {
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
                let byte = avcc.get_mut(4).ok_or("an avcC record of 4 bytes")?;
                *byte = (*byte & !3) | (to - 1);
            }
            let decoder = H264Decoder::new(&device, images, &avcc).map_err(|e| e.to_string())?;
            (decoder, samples)
        } else {
            let units = nal::access_units(&file);
            (
                H264Decoder::new_annex_b(&device, images).map_err(|e| e.to_string())?,
                units.into_iter().map(<[u8]>::to_vec).collect(),
            )
        };
    decoder.set_output_order(options.order);
    let mut reader = Reader {
        device: Arc::clone(&device),
        index: 0,
        readback: Duration::ZERO,
    };
    let mut held = VecDeque::new();
    let start = StdRuntime.now();
    for (id, sample) in (0u64..).zip(&samples) {
        let frames = decoder
            .decode(sample, id)
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
    for frame in decoder.flush() {
        reader.emit(&frame)?;
    }
    let total = StdRuntime.elapsed_since(start);
    eprintln!(
        "{} samples, {} frames, profile {}, {} layouts, copies on {}, {:.3} s, read back {:.3} s, \
         on {}",
        samples.len(),
        reader.index,
        decoder.profile().map_or(0, |p| p.profile_idc),
        if device.unified_image_layouts() {
            "unified"
        } else {
            "DPB"
        },
        if device.copies_on_decode_queue() {
            "the decode queue".to_string()
        } else {
            format!("family {}", device.copy_family())
        },
        total.as_secs_f64(),
        reader.readback.as_secs_f64(),
        device.name(),
    );
    Ok(())
}
