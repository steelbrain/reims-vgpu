//! Decodes an H.265 stream with [`reims_vgpu_video_vulkan::H265Decoder`]
//! on the device [`VideoDevice::create`] picks, and prints each frame's MD5
//! over its cropped planes, packed, in the order the decoder emits them:
//! NV12 for Main, P010 (little-endian 16-bit samples) for Main 10. In
//! display order this is what `ffmpeg -f framemd5 -pix_fmt nv12` (or
//! `p010le`) prints for the same stream, so a software decode is the
//! reference (tools/h265-check.sh), as for the VA-API backend's
//! hevc-framemd5.
//!
//! ```sh
//! vulkan-h265-framemd5 stream.mp4 [--decode-order] [--length-size N] [--hold N] [--skip N] [--copy-on-graphics] [--buffer] [--in-place]
//! vulkan-h265-framemd5 --capabilities
//! ```
//!
//! - An `.mp4` is read as VideoToolbox is handed a stream: the `hvcC`
//!   record and length-prefixed samples. Anything else is an Annex B
//!   stream, split into access units.
//! - `--decode-order` emits in decode order instead of display order.
//! - `--length-size N` rewrites an MP4's samples and record to `N`-byte
//!   NAL lengths first.
//! - `--hold N` keeps the last `N` frames alive, as a caller that is slow
//!   to drop them would.
//! - `--skip N` starts at the `N`th access unit (decode order), as a seek
//!   would. The RASL pictures skipped then give no frame, and are counted
//!   on stderr.
//! - `--copy-on-graphics` hands the device to the decoder as another's
//!   ([`VideoDevice::from_raw_with_codecs`]) with its graphics queue as
//!   the copy queue, so that pictures decoded into DPB pictures alone are
//!   copied out on a queue of their own, behind a semaphore wait, as on a
//!   device whose decode family does not copy.
//! - `--buffer` lands each picture in a host-visible buffer
//!   ([`reims_vgpu_video_vulkan::HostBuffers`]), cropped, at offsets and
//!   pitches that are not the frame's own, and reads it there, instead of
//!   in an image read back.
//! - `--in-place` (Annex B only) puts every access unit in one
//!   host-visible allocation of the device's, each placed as the guest's
//!   video driver places them, its first slice segment's start code at a
//!   multiple of the device's bitstream offset alignment and zeros after
//!   it to its size alignment, and decodes each there
//!   ([`H265Decoder::decode_in_place`]); stderr counts the pictures read in
//!   place and those copied.
//! - `--capabilities` prints what the device answers for each profile.
//!
//! Prints `frame <index> id <sample> poc <poc> <md5>` per frame, and the
//! time spent to stderr.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use ash::vk;
use reims_vgpu_video_conformance::md5::{Md5, self_test};
use reims_vgpu_video_conformance::mp4;
use reims_vgpu_video_core::runtime::{Runtime, StdRuntime};
use reims_vgpu_video_vulkan::hevc::hvcc::DecoderConfiguration;
use reims_vgpu_video_vulkan::{
    DecodeImage, DecodeImages, Error, H265Decoder, H265Frame, HostBuffer, HostBuffers, HostImport,
    OutputOrder, RawDevice, SampleMemory, Surface, SurfaceSource, VideoDevice, h265_capabilities,
    hevc, nal,
};

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("vulkan-h265-framemd5: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

struct Options {
    input: String,
    order: OutputOrder,
    length_size: Option<u8>,
    hold: usize,
    skip: usize,
    copy_on_graphics: bool,
    buffer: bool,
    in_place: bool,
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
        skip: 0,
        copy_on_graphics: false,
        buffer: false,
        in_place: false,
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
            "--skip" => options.skip = number("--skip")?,
            "--copy-on-graphics" => options.copy_on_graphics = true,
            "--buffer" => options.buffer = true,
            "--in-place" => options.in_place = true,
            "--capabilities" => options.capabilities = true,
            _ => positional.push(arg),
        }
    }
    if options.capabilities {
        return Ok(options);
    }
    let [input] = <[String; 1]>::try_from(positional)
        .map_err(|_| "usage: vulkan-h265-framemd5 <stream> [options]")?;
    if options.in_place && input.ends_with(".mp4") {
        return Err("--in-place takes an Annex B stream, whose slices follow start codes".into());
    }
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

/// The cropped rectangle of a frame in bytes of its planes: the offset of
/// its first byte in a row and the bytes a row (both planes alike, 4:2:0
/// halving the chroma rows), and its rows of luma and of chroma.
fn crop_bytes(frame: &H265Frame<impl Surface>) -> [usize; 4] {
    let (x, y, w, h) = frame.crop;
    let bytes = usize::try_from(frame.picture.sample_bytes()).unwrap_or(1);
    let [x, y, w, h] = [x, y, w, h].map(|v| usize::try_from(v).unwrap_or(0));
    // A chroma pair spans two luma samples: the same bytes a row.
    [x * bytes, w * bytes, y, h]
}

/// An output whose frame's cropped planes the tool hashes.
trait Hashed: Surface + Sized {
    /// The MD5 of the frame's cropped planes, packed: the luma rows, then
    /// the chroma rows.
    fn md5(frame: &H265Frame<Self>, device: &Arc<VideoDevice>) -> Result<String, String>;
}

impl Hashed for DecodeImage {
    fn md5(frame: &H265Frame<Self>, device: &Arc<VideoDevice>) -> Result<String, String> {
        let planes = frame.picture.read_back(device).map_err(|e| e.to_string())?;
        let [x, row, y, h] = crop_bytes(frame);
        let stride = usize::try_from(planes.width * planes.sample_bytes).unwrap_or(0);
        let mut md5 = Md5::new();
        for line in planes.y.chunks(stride).skip(y).take(h) {
            md5.update(line.get(x..x + row).ok_or("a crop past the frame")?);
        }
        for line in planes.uv.chunks(stride).skip(y / 2).take(h / 2) {
            md5.update(line.get(x..x + row).ok_or("a crop past the frame")?);
        }
        Ok(md5.hex())
    }
}

impl Hashed for HostBuffer {
    fn md5(frame: &H265Frame<Self>, _: &Arc<VideoDevice>) -> Result<String, String> {
        frame
            .picture
            .wait(Duration::from_secs(10))
            .map_err(|e| e.to_string())?;
        let buffer = frame.picture.surface();
        let (bytes, planes) = (buffer.bytes(), buffer.planes());
        let [_, row, _, h] = crop_bytes(frame);
        let mut md5 = Md5::new();
        for ((offset, pitch), rows) in [(planes.luma, h), (planes.chroma, h / 2)] {
            let [offset, pitch] =
                [offset, u64::from(pitch)].map(|v| usize::try_from(v).unwrap_or(usize::MAX));
            for line in 0..rows {
                let at = offset + line * pitch;
                md5.update(bytes.get(at..at + row).ok_or("a plane past the buffer")?);
            }
        }
        Ok(md5.hex())
    }
}

struct Reader {
    device: Arc<VideoDevice>,
    index: usize,
    bit_depth: u8,
    readback: Duration,
}

impl Reader {
    /// Prints the MD5 of the frame's cropped planes.
    fn emit<T: Hashed>(&mut self, frame: &H265Frame<T>) -> Result<(), String> {
        let start = StdRuntime.now();
        let md5 = T::md5(frame, &self.device)?;
        self.readback += StdRuntime.elapsed_since(start);
        self.bit_depth = frame.bit_depth;
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
        "{}: decode family {}, graphics family {}, copies on {}, unified image layouts {}, \
         codecs {:#x}",
        device.name(),
        device.decode_family(),
        device.graphics_family(),
        if device.copies_on_decode_queue() {
            "the decode queue".to_string()
        } else {
            format!("family {}", device.copy_family())
        },
        device.unified_image_layouts(),
        device.codecs().as_raw(),
    );
    for caps in h265_capabilities(device) {
        println!(
            "profile {} ({} bits): general_level_idc {}, {}x{} to {}x{}, granularity {}x{}, {} \
             DPB slots, {} active references, coincide {}, distinct {}, decodes into outputs {} \
             (outputs are targets {}), bitstream offset {} size {} alignment",
            caps.profile.profile_idc,
            caps.profile.bit_depth(),
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
            instance_api_version: vk::API_VERSION_1_3,
            physical_device: owner.physical_device(),
            device: owner.device().clone(),
            decode_family: owner.decode_family(),
            decode_queue: owner.decode_queue(),
            copy_family: owner.graphics_family(),
            copy_queue: graphics,
            graphics_family: owner.graphics_family(),
            unified_image_layouts_video: owner.unified_image_layouts(),
        };
        // SAFETY: `owner` made every handle as RawDevice says, with the
        // decode extensions of its codecs, and outlives `device`, which goes
        // before it at the end of this function along with everything made
        // on it; nothing else submits to the queues.
        unsafe { VideoDevice::from_raw_with_codecs(raw, owner.codecs()) }
    } else {
        Arc::clone(&owner)
    };
    let input = read_input(&options)?;
    // Held past the decoder, which holds it while its decodes read it.
    let in_place = if options.in_place {
        Some(Arc::new(InPlace::new(&device, &input.samples)?))
    } else {
        None
    };
    if options.buffer {
        decode_all(
            &device,
            HostBuffers::new(&device),
            &options,
            &input,
            in_place.as_ref(),
        )
    } else {
        decode_all(
            &device,
            DecodeImages::new(&device),
            &options,
            &input,
            in_place.as_ref(),
        )
    }
}

/// A stream's samples, and its `hvcC` record where it is an MP4.
struct Input {
    hvcc: Option<Vec<u8>>,
    samples: Vec<Vec<u8>>,
}

fn read_input(options: &Options) -> Result<Input, String> {
    let file = std::fs::read(&options.input).map_err(|e| format!("{}: {e}", options.input))?;
    if !options.input.ends_with(".mp4") {
        let units = hevc::nal::access_units(&file);
        return Ok(Input {
            hvcc: None,
            samples: units.into_iter().map(<[u8]>::to_vec).collect(),
        });
    }
    let track = mp4::hevc_track(&file)?;
    let mut hvcc = track.avcc.clone();
    let from = DecoderConfiguration::parse(&hvcc)
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
        // lengthSizeMinusOne, in the low two bits of byte 21.
        let byte = hvcc.get_mut(21).ok_or("an hvcC record of 21 bytes")?;
        *byte = (*byte & !3) | (to - 1);
    }
    Ok(Input {
        hvcc: Some(hvcc),
        samples,
    })
}

/// Every sample in one host-visible, host-coherent allocation of the
/// device's, mapped while it lives, as the guest's access units lie in its
/// pages: each placed so that its first slice segment's start code is at
/// a multiple of the bitstream offset alignment, and followed by zeros to
/// its size alignment from there.
struct InPlace {
    device: Arc<VideoDevice>,
    memory: vk::DeviceMemory,
    type_index: u32,
    size: u64,
    mapped: *mut u8,
    /// Each sample's first byte in the allocation and its bytes, padding
    /// included.
    places: Vec<(usize, usize)>,
}

// SAFETY: the mapping is the allocation's alone, which this holds.
unsafe impl Send for InPlace {}
// SAFETY: as above; it is read through `&self` alone once written.
unsafe impl Sync for InPlace {}

/// The bytes of [`InPlace`]'s allocation past its last sample.
const TAIL: usize = 64 << 10;

/// Where an Annex B sample's first slice segment's three-byte start code
/// is, if it has one.
fn first_slice_start_code(sample: &[u8]) -> Option<usize> {
    sample.windows(4).position(|w| {
        // NAL unit types 0 to 31 are VCL units (7.4.2.2).
        matches!(w, [0, 0, 1, header] if (header >> 1) & 0x3f < 32)
    })
}

impl InPlace {
    fn new(device: &Arc<VideoDevice>, samples: &[Vec<u8>]) -> Result<Self, String> {
        let caps = h265_capabilities(device);
        let first = caps.first().ok_or("the device decodes no H.265")?;
        let offset_alignment = caps
            .iter()
            .map(|c| c.min_bitstream_buffer_offset_alignment)
            .max()
            .unwrap_or(1)
            .max(1);
        let size_alignment = caps
            .iter()
            .map(|c| c.min_bitstream_buffer_size_alignment)
            .max()
            .unwrap_or(1)
            .max(1);
        let [offset_alignment, size_alignment] =
            [offset_alignment, size_alignment].map(|a| usize::try_from(a).unwrap_or(1));
        let mut places = Vec::with_capacity(samples.len());
        let mut end = 0usize;
        for sample in samples {
            let code = first_slice_start_code(sample).unwrap_or(0);
            let at = (end + code).next_multiple_of(offset_alignment) - code;
            let padded = code + (sample.len() - code).next_multiple_of(size_alignment);
            places.push((at, padded));
            end = at + padded;
        }
        // Memory past the last sample, as guest memory has pages past each
        // sample: a buffer bound over a sample may bind more bytes than it
        // reads, and the last sample's was refused without them on the
        // 5080 (11-03).
        let end = end + TAIL;
        let size = u64::try_from(end).map_err(|e| e.to_string())?;
        let vk_device = device.device();
        // The memory types a decode's bitstream buffer binds to.
        let bits = first.profile.with_list(|list| {
            let info = vk::BufferCreateInfo::default()
                .size(size)
                .usage(vk::BufferUsageFlags::VIDEO_DECODE_SRC_KHR)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .push_next(list);
            // SAFETY: `info` and its chain live through the call.
            let buffer =
                unsafe { vk_device.create_buffer(&info, None) }.map_err(|e| e.to_string())?;
            // SAFETY: the buffer is the device's.
            let bits = unsafe { vk_device.get_buffer_memory_requirements(buffer) }.memory_type_bits;
            // SAFETY: the buffer is bound to nothing and used by nothing.
            unsafe { vk_device.destroy_buffer(buffer, None) };
            Ok::<_, String>(bits)
        })?;
        // SAFETY: the physical device is the instance's.
        let properties = unsafe {
            device
                .instance()
                .get_physical_device_memory_properties(device.physical_device())
        };
        let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let type_index = (0u32..properties.memory_type_count)
            .find(|&i| {
                bits & (1 << i) != 0
                    && properties.memory_types[i as usize]
                        .property_flags
                        .contains(host)
            })
            .ok_or("no host-visible memory type takes a bitstream buffer")?;
        let info = vk::MemoryAllocateInfo::default()
            .allocation_size(size)
            .memory_type_index(type_index);
        // SAFETY: `info` lives through the call.
        let memory =
            unsafe { vk_device.allocate_memory(&info, None) }.map_err(|e| e.to_string())?;
        // SAFETY: the memory is host-visible and unmapped.
        let mapped = match unsafe {
            vk_device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
        } {
            Ok(mapped) => mapped.cast::<u8>(),
            Err(error) => {
                // SAFETY: the memory is unmapped and used by nothing.
                unsafe { vk_device.free_memory(memory, None) };
                return Err(error.to_string());
            }
        };
        let made = Self {
            device: Arc::clone(device),
            memory,
            type_index,
            size,
            mapped,
            places,
        };
        // SAFETY: the mapping is `size` bytes of host-coherent memory no GPU
        // work uses yet.
        let all = unsafe { std::slice::from_raw_parts_mut(made.mapped, end) };
        all.fill(0);
        for (sample, &(at, _)) in samples.iter().zip(&made.places) {
            all[at..at + sample.len()].copy_from_slice(sample);
        }
        Ok(made)
    }

    /// Sample `index`'s bytes in the allocation, its padding included, and
    /// where they lie in device memory.
    fn sample(&self, index: usize, keep: &Arc<Self>) -> Option<(&[u8], SampleMemory)> {
        let &(at, len) = self.places.get(index)?;
        // SAFETY: the bytes lie in the mapping, written once by `new`, and
        // nothing writes them again.
        let bytes = unsafe { std::slice::from_raw_parts(self.mapped.add(at), len) };
        let offset = u64::try_from(at).ok()?;
        let memory = SampleMemory {
            import: HostImport {
                memory: self.memory,
                offset,
                size: self.size - offset,
                type_index: self.type_index,
                handle_type: vk::ExternalMemoryHandleTypeFlags::empty(),
            },
            keep: Arc::clone(keep) as Arc<dyn Send + Sync>,
        };
        Some((bytes, memory))
    }
}

impl Drop for InPlace {
    fn drop(&mut self) {
        // SAFETY: the memory is ours; the decodes that read it hold it
        // through `keep` until their work is done, so nothing uses it now.
        unsafe { self.device.device().free_memory(self.memory, None) };
    }
}

fn decode_all<S: SurfaceSource>(
    device: &Arc<VideoDevice>,
    source: S,
    options: &Options,
    input: &Input,
    in_place: Option<&Arc<InPlace>>,
) -> Result<(), String>
where
    S::Surface: Hashed,
{
    let mut decoder = match &input.hvcc {
        Some(hvcc) => H265Decoder::new(device, source, hvcc),
        None => H265Decoder::new_annex_b(device, source),
    }
    .map_err(|e| e.to_string())?;
    decoder.set_output_order(options.order);
    let mut reader = Reader {
        device: Arc::clone(device),
        index: 0,
        bit_depth: 0,
        readback: Duration::ZERO,
    };
    let mut held = VecDeque::new();
    let mut skipped = 0usize;
    // The samples whose pictures were copied though they lay in place.
    let mut copied_samples = Vec::new();
    let start = StdRuntime.now();
    for (index, sample) in input.samples.iter().enumerate().skip(options.skip) {
        let id = u64::try_from(index).unwrap_or(u64::MAX);
        let copied = decoder.bitstreams().copied;
        let decoded = match in_place {
            Some(place) => {
                let (bytes, memory) = place.sample(index, place).ok_or("no such sample")?;
                // SAFETY: the bytes lie in the allocation `memory` names, at
                // its offset, which `keep` holds; nothing writes them again.
                unsafe { decoder.decode_in_place(bytes, id, memory) }
            }
            None => decoder.decode(sample, id),
        };
        if in_place.is_some() && decoder.bitstreams().copied > copied {
            copied_samples.push(index);
        }
        let frames = match decoded {
            Ok(frames) => frames,
            Err(Error::H265Stream(hevc::Error::SkippedLeadingPicture)) => {
                skipped += 1;
                continue;
            }
            Err(e) => return Err(format!("sample {id}: {e}")),
        };
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
    drop(held);
    let total = StdRuntime.elapsed_since(start);
    let bitstreams = decoder.bitstreams();
    if !copied_samples.is_empty() {
        eprintln!("samples copied though in place: {copied_samples:?}");
    }
    eprintln!(
        "{} samples, {} frames, {skipped} RASL skipped, profile {}, {} bits, {} layouts, copies \
         on {}, into {}, {} pictures read in place and {} copied, {:.3} s, read back {:.3} s, on {}",
        input.samples.len() - options.skip.min(input.samples.len()),
        reader.index,
        decoder.profile().map_or(0, |p| p.profile_idc),
        reader.bit_depth,
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
        bitstreams.in_place,
        bitstreams.copied,
        total.as_secs_f64(),
        reader.readback.as_secs_f64(),
        device.name(),
    );
    Ok(())
}
