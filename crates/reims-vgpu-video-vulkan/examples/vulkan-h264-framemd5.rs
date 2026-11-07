//! Decodes an H.264 stream with [`reims_vgpu_video_vulkan::H264Decoder`]
//! on the device [`VideoDevice::create`] picks, and prints each frame's MD5
//! over its cropped NV12 planes, packed, in the order the decoder emits
//! them. In display order this is what `ffmpeg -f framemd5 -pix_fmt nv12`
//! prints for the same stream, so a software decode is the reference
//! (tools/h264-check.sh), as for the VA-API backend's h264-framemd5.
//!
//! ```sh
//! vulkan-h264-framemd5 stream.mp4 [--decode-order] [--length-size N] [--hold N] [--copy-on-graphics] [--copy-lent] [--buffer]
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
//! - `--copy-lent` hands it over as another's that lends its graphics
//!   queue for the copies ([`VideoDevice::from_raw_copying_on`]), each
//!   submitted once the host sees its decode done.
//! - `--buffer` lands each picture in a host-visible buffer
//!   ([`reims_vgpu_video_vulkan::HostBuffers`]), cropped, at offsets and
//!   pitches that are not the frame's own, and reads it there, instead of
//!   in an image read back.
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
use reims_vgpu_video_core::sync::Mutex;
use reims_vgpu_video_vulkan::avcc::DecoderConfiguration;
use reims_vgpu_video_vulkan::{
    DecodeImage, DecodeImages, Frame, H264Decoder, HostBuffer, HostBuffers, LentQueue, OutputOrder,
    RawDevice, Surface, SurfaceSource, VideoDevice, h264_capabilities, nal,
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
    copy_lent: bool,
    buffer: bool,
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
        copy_lent: false,
        buffer: false,
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
            "--copy-lent" => options.copy_lent = true,
            "--buffer" => options.buffer = true,
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

/// An output whose frame's cropped planes the tool hashes.
trait Hashed: Surface + Sized {
    /// The MD5 of the frame's cropped planes, packed: the luma rows, then
    /// the chroma rows.
    fn md5(frame: &Frame<Self>, device: &Arc<VideoDevice>) -> Result<String, String>;
}

impl Hashed for DecodeImage {
    fn md5(frame: &Frame<Self>, device: &Arc<VideoDevice>) -> Result<String, String> {
        let planes = frame.picture.read_back(device).map_err(|e| e.to_string())?;
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
        Ok(md5.hex())
    }
}

impl Hashed for HostBuffer {
    fn md5(frame: &Frame<Self>, _: &Arc<VideoDevice>) -> Result<String, String> {
        frame
            .picture
            .wait(Duration::from_secs(10))
            .map_err(|e| e.to_string())?;
        let buffer = frame.picture.surface();
        let (bytes, planes) = (buffer.bytes(), buffer.planes());
        let (_, _, w, h) = frame.crop;
        let [w, h] = [w, h].map(|v| usize::try_from(v).unwrap_or(0));
        let mut md5 = Md5::new();
        for ((offset, pitch), rows) in [(planes.luma, h), (planes.chroma, h / 2)] {
            let [offset, pitch] =
                [offset, u64::from(pitch)].map(|v| usize::try_from(v).unwrap_or(usize::MAX));
            for row in 0..rows {
                let at = offset + row * pitch;
                md5.update(bytes.get(at..at + w).ok_or("a plane past the buffer")?);
            }
        }
        Ok(md5.hex())
    }
}

struct Reader {
    device: Arc<VideoDevice>,
    index: usize,
    readback: Duration,
}

impl Reader {
    /// Prints the MD5 of the frame's cropped planes.
    fn emit<T: Hashed>(&mut self, frame: &Frame<T>) -> Result<(), String> {
        let start = StdRuntime.now();
        let md5 = T::md5(frame, &self.device)?;
        self.readback += StdRuntime.elapsed_since(start);
        println!(
            "frame {} id {} poc {} {md5}",
            self.index, frame.id, frame.poc
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
             references, coincide {}, distinct {}, decodes into outputs {} (outputs are targets \
             {}), bitstream offset {} size {} alignment",
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
            caps.outputs_are_targets,
            caps.min_bitstream_buffer_offset_alignment,
            caps.min_bitstream_buffer_size_alignment,
        );
    }
}

/// The graphics queue of a device of our own, lent for the copies as
/// another submitter's queue would be.
struct LentGraphics {
    queue: Mutex<ash::vk::Queue>,
    family: u32,
}

impl LentQueue for LentGraphics {
    fn family(&self) -> u32 {
        self.family
    }

    fn with_queue(&self, submit: &mut dyn FnMut(ash::vk::Queue)) {
        submit(*self.queue.lock());
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
    let device = if options.copy_on_graphics || options.copy_lent {
        let graphics = owner
            .graphics_queue()
            .ok_or("a device of our own has a graphics queue")?;
        let raw = RawDevice {
            entry: owner.entry().clone(),
            instance: owner.instance().clone(),
            instance_api_version: ash::vk::API_VERSION_1_3,
            physical_device: owner.physical_device(),
            device: owner.device().clone(),
            decode_family: owner.decode_family(),
            decode_queue: owner.decode_queue(),
            copy_family: owner.graphics_family(),
            copy_queue: graphics,
            graphics_family: owner.graphics_family(),
            unified_image_layouts_video: owner.unified_image_layouts(),
        };
        if options.copy_lent {
            let lent = LentGraphics {
                queue: Mutex::new(graphics),
                family: owner.graphics_family(),
            };
            // SAFETY: as below; the lent queue is the graphics queue, of
            // its family.
            unsafe {
                VideoDevice::from_raw_copying_on(
                    raw,
                    ash::vk::VideoCodecOperationFlagsKHR::DECODE_H264,
                    Box::new(lent),
                )
            }
            .map_err(|e| e.to_string())?
        } else {
            // SAFETY: `owner` made every handle as RawDevice says, and
            // outlives `device`, which goes before it at the end of this
            // function along with everything made on it; nothing else
            // submits to the queues.
            unsafe { VideoDevice::from_raw(raw) }
        }
    } else {
        Arc::clone(&owner)
    };
    let input = read_input(&options)?;
    if options.buffer {
        decode_all(&device, HostBuffers::new(&device), &options, &input)
    } else {
        decode_all(&device, DecodeImages::new(&device), &options, &input)
    }
}

/// A stream's samples, and its `avcC` record where it is an MP4.
struct Input {
    avcc: Option<Vec<u8>>,
    samples: Vec<Vec<u8>>,
}

fn read_input(options: &Options) -> Result<Input, String> {
    let file = std::fs::read(&options.input).map_err(|e| format!("{}: {e}", options.input))?;
    if !options.input.ends_with(".mp4") {
        let units = nal::access_units(&file);
        return Ok(Input {
            avcc: None,
            samples: units.into_iter().map(<[u8]>::to_vec).collect(),
        });
    }
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
    Ok(Input {
        avcc: Some(avcc),
        samples,
    })
}

fn decode_all<S: SurfaceSource>(
    device: &Arc<VideoDevice>,
    source: S,
    options: &Options,
    input: &Input,
) -> Result<(), String>
where
    S::Surface: Hashed,
{
    let mut decoder = match &input.avcc {
        Some(avcc) => H264Decoder::new(device, source, avcc),
        None => H264Decoder::new_annex_b(device, source),
    }
    .map_err(|e| e.to_string())?;
    decoder.set_output_order(options.order);
    let mut reader = Reader {
        device: Arc::clone(device),
        index: 0,
        readback: Duration::ZERO,
    };
    let mut held = VecDeque::new();
    let start = StdRuntime.now();
    for (id, sample) in (0u64..).zip(&input.samples) {
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
        "{} samples, {} frames, profile {}, {} layouts, copies on {}, into {}, {:.3} s, read back \
         {:.3} s, on {}",
        input.samples.len(),
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
        if options.buffer { "buffers" } else { "images" },
        total.as_secs_f64(),
        reader.readback.as_secs_f64(),
        device.name(),
    );
    Ok(())
}
