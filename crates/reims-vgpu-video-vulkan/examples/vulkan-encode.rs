//! Encodes raw NV12 or P010 frames with [`H264Encoder`] or [`H265Encoder`]
//! on the device [`VideoDevice::create`] picks, and writes an Annex B
//! stream, the parameter sets in front of each IDR picture: the device's
//! in-band ones, or, out of band, the encoder's written there by this tool
//! as a container would carry them. tools/encode-check.sh decodes what it
//! writes with ffmpeg and with this crate's decoders.
//!
//! ```sh
//! vulkan-encode --codec h264|h265 --size WxH --input frames.yuv --output out.264 [options]
//! vulkan-encode --capabilities
//! ```
//!
//! - `--profile baseline|main|high|main10`: H.264's Constrained Baseline,
//!   Main (the default) or High; H.265's Main (the default) or Main 10, from
//!   P010 (16-bit little-endian samples, as ffmpeg's `p010le`).
//! - `--fps N[/D]` (30), `--frames N` (all the input has).
//! - `--rc cqp:I,P,B | cbr:BITS | vbr:AVERAGE,PEAK` (vbr:5000000,7500000),
//!   `--vbv SIZE[,INITIAL]` in bits.
//! - `--idr N|none` (60), `--intra N`, `--b N`, `--refs N`: the GOP.
//! - `--force-idr A,B,...`: frames (from 0, display order) forced IDR.
//! - `--switch-rc FRAME:MODE:VALUES`: `--rc`'s MODE:VALUES from frame
//!   FRAME on ([`H264Encoder::set_rate_control`]).
//! - `--slices N`, `--quality N`, `--level IDC`, `--cavlc`, `--tuning
//!   default|hq|ll|ull`, `--in-band`.
//! - `--source image|buffer`: each frame in an image of the encoder's
//!   [`SourceSpec`] in device-local memory, read in place (the default),
//!   the frame uploaded into it on the graphics queue first and the encode
//!   waiting for that; or in host-visible memory, its planes read where
//!   the input was read into ([`ImportedSource`]), which the encode copies
//!   on the GPU.
//! - `--bitstream host|import`: each picture written into a host-visible
//!   buffer of its own ([`HostBitstreams`], the default), or into a slot of
//!   one host-visible allocation through [`ImportedBitstream`], as the
//!   guest's buffer is written in place.
//! - `--depth N` (4): frames in flight before the oldest is waited for.
//! - `--timing`: each picture's GPU time, its source's copy's apart.
//! - `--record PATH`: writes the `avcC` or `hvcC` record there.
//! - `--capabilities` prints what the device answers for each profile.
//!
//! Prints one line a frame, in decode order: `frame <id> pts <pts> dts
//! <dts> type <IDR|I|P|B> poc <n> frame_num <n> ref <0|1> at <offset> bytes
//! <n> ps <n> status <status> gpu_us <total> copy_us <copy>`; and a summary
//! to stderr, the copies on the CPU among it: none.

use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::Duration;

use ash::vk;
use reims_vgpu_video_codec::encode::FrameRate;
use reims_vgpu_video_core::runtime::{Runtime, StdRuntime};
use reims_vgpu_video_vulkan::{
    BitstreamSource, BitstreamSpec, BitstreamTarget, EncodeCapabilities, EncodeCodec, EncodeConfig,
    EncodeProfile, EncodeSource, EncodeStatus, EncodedFrame, Error, FrameParams, GopConfig,
    H264Encoder, H265Encoder, HostBitstream, HostBitstreams, HostImport, ImportedBitstream,
    ImportedSource, ParameterSetMode, PictureKind, RateControl, SourceCounts, SourcePicture,
    SourceSpec, Tuning, Vbv, VideoDevice, h264_encode_capabilities, h265_encode_capabilities,
};

const TIMEOUT: Duration = Duration::from_secs(10);

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("vulkan-encode: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SourceKind {
    Image,
    Buffer,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TargetKind {
    Host,
    Import,
}

struct Options {
    profile: EncodeProfile,
    config: EncodeConfig,
    input: String,
    output: String,
    frames: Option<u64>,
    force_idr: Vec<u64>,
    switch_rc: Option<(u64, RateControl)>,
    source: SourceKind,
    bitstream: TargetKind,
    depth: usize,
    record: Option<String>,
    capabilities: bool,
}

fn parse<T: std::str::FromStr>(name: &str, value: Option<String>) -> Result<T, String> {
    value
        .and_then(|v| v.parse().ok())
        .ok_or(format!("{name} takes a number"))
}

fn numbers(name: &str, value: Option<String>) -> Result<Vec<u64>, String> {
    let value = value.ok_or(format!("{name} takes numbers"))?;
    value
        .split(',')
        .map(|n| n.parse().map_err(|_| format!("{name}: {n} is no number")))
        .collect()
}

/// `--rc`'s MODE:VALUES.
fn rate_control(value: &str) -> Result<RateControl, String> {
    let (mode, values) = value.split_once(':').ok_or("--rc takes mode:values")?;
    let values = numbers("--rc", Some(values.to_owned()))?;
    let small = |v: u64| u8::try_from(v).map_err(|_| "a QP past 255".to_owned());
    Ok(match (mode, values.as_slice()) {
        ("cqp", &[i, p, b]) => RateControl::ConstantQp {
            intra: small(i)?,
            predicted: small(p)?,
            bidirectional: small(b)?,
        },
        ("cbr", &[bitrate]) => RateControl::ConstantBitrate { bitrate },
        ("vbr", &[average, peak]) => RateControl::VariableBitrate { average, peak },
        _ => return Err("--rc is cqp:I,P,B, cbr:BITS or vbr:AVERAGE,PEAK".into()),
    })
}

fn options() -> Result<Options, String> {
    let mut args = std::env::args().skip(1);
    let mut codec = EncodeCodec::H264;
    let mut profile = None;
    let mut size = None;
    let mut tuning = Tuning::Default;
    let mut fps = FrameRate::new(30, 1);
    let mut rate = None;
    let mut vbv = None;
    let mut gop = GopConfig::default();
    let mut slices = 1;
    let mut quality = None;
    let mut level = None;
    let mut cabac = true;
    let mut in_band = false;
    let mut timing = false;
    let mut options = Options {
        profile: EncodeProfile::H264_MAIN,
        config: EncodeConfig::new(2, 2),
        input: String::new(),
        output: String::new(),
        frames: None,
        force_idr: Vec::new(),
        switch_rc: None,
        source: SourceKind::Image,
        bitstream: TargetKind::Host,
        depth: 4,
        record: None,
        capabilities: false,
    };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--codec" => {
                codec = match args.next().as_deref() {
                    Some("h264") => EncodeCodec::H264,
                    Some("h265") => EncodeCodec::H265,
                    _ => return Err("--codec is h264 or h265".into()),
                }
            }
            "--profile" => profile = args.next(),
            "--size" => {
                let value = args.next().ok_or("--size takes WxH")?;
                let (w, h) = value.split_once('x').ok_or("--size takes WxH")?;
                size = Some((
                    w.parse::<u32>().map_err(|e| e.to_string())?,
                    h.parse::<u32>().map_err(|e| e.to_string())?,
                ));
            }
            "--fps" => {
                let value = args.next().ok_or("--fps takes N[/D]")?;
                let (n, d) = value.split_once('/').unwrap_or((&value, "1"));
                fps = FrameRate::new(
                    n.parse().map_err(|_| "--fps takes N[/D]")?,
                    d.parse().map_err(|_| "--fps takes N[/D]")?,
                );
            }
            "--frames" => options.frames = Some(parse("--frames", args.next())?),
            "--rc" => rate = Some(rate_control(&args.next().ok_or("--rc takes a mode")?)?),
            "--switch-rc" => {
                let value = args.next().ok_or("--switch-rc takes FRAME:MODE:VALUES")?;
                let (frame, rate) = value
                    .split_once(':')
                    .ok_or("--switch-rc takes FRAME:MODE:VALUES")?;
                options.switch_rc = Some((
                    frame.parse().map_err(|_| "--switch-rc takes a frame")?,
                    rate_control(rate)?,
                ));
            }
            "--vbv" => {
                let values = numbers("--vbv", args.next())?;
                vbv = Some(match *values.as_slice() {
                    [size] => Vbv {
                        size_bits: size,
                        initial_bits: size / 4 * 3,
                    },
                    [size, initial] => Vbv {
                        size_bits: size,
                        initial_bits: initial,
                    },
                    _ => return Err("--vbv takes SIZE[,INITIAL]".into()),
                });
            }
            "--idr" => {
                gop.idr_period = match args.next().as_deref() {
                    Some("none") => None,
                    value => Some(parse("--idr", value.map(str::to_owned))?),
                }
            }
            "--intra" => gop.intra_period = Some(parse("--intra", args.next())?),
            "--b" => gop.b_frames = parse("--b", args.next())?,
            "--refs" => gop.references = parse("--refs", args.next())?,
            "--force-idr" => options.force_idr = numbers("--force-idr", args.next())?,
            "--slices" => slices = parse("--slices", args.next())?,
            "--quality" => quality = Some(parse("--quality", args.next())?),
            "--level" => level = Some(parse("--level", args.next())?),
            "--cavlc" => cabac = false,
            "--tuning" => {
                tuning = match args.next().as_deref() {
                    Some("default") => Tuning::Default,
                    Some("hq") => Tuning::HighQuality,
                    Some("ll") => Tuning::LowLatency,
                    Some("ull") => Tuning::UltraLowLatency,
                    _ => return Err("--tuning is default, hq, ll or ull".into()),
                }
            }
            "--in-band" => in_band = true,
            "--source" => {
                options.source = match args.next().as_deref() {
                    Some("image") => SourceKind::Image,
                    Some("buffer") => SourceKind::Buffer,
                    _ => return Err("--source is image or buffer".into()),
                }
            }
            "--bitstream" => {
                options.bitstream = match args.next().as_deref() {
                    Some("host") => TargetKind::Host,
                    Some("import") => TargetKind::Import,
                    _ => return Err("--bitstream is host or import".into()),
                }
            }
            "--depth" => options.depth = parse::<usize>("--depth", args.next())?.max(1),
            "--timing" => timing = true,
            "--record" => options.record = Some(args.next().ok_or("--record takes a path")?),
            "--input" => options.input = args.next().ok_or("--input takes a path")?,
            "--output" => options.output = args.next().ok_or("--output takes a path")?,
            "--capabilities" => options.capabilities = true,
            other => return Err(format!("unknown option {other}")),
        }
    }
    if options.capabilities {
        return Ok(options);
    }
    let (width, height) = size.ok_or("--size is needed")?;
    if options.input.is_empty() || options.output.is_empty() {
        return Err("--input and --output are needed".into());
    }
    options.profile = match (codec, profile.as_deref()) {
        (EncodeCodec::H264, Some("baseline")) => EncodeProfile::H264_BASELINE,
        (EncodeCodec::H264, Some("main") | None) => EncodeProfile::H264_MAIN,
        (EncodeCodec::H264, Some("high")) => EncodeProfile::H264_HIGH,
        (EncodeCodec::H265, Some("main") | None) => EncodeProfile::H265_MAIN,
        (EncodeCodec::H265, Some("main10")) => EncodeProfile::H265_MAIN_10,
        (_, Some(other)) => return Err(format!("no profile {other} for the codec")),
    }
    .with_tuning(tuning);
    let mut config = EncodeConfig::new(width, height);
    config.frame_rate = fps;
    if let Some(rate) = rate {
        config.rate_control = rate;
    }
    config.vbv = vbv;
    config.gop = gop;
    config.slices = slices;
    config.quality_level = quality;
    config.level = level;
    config.cabac = cabac;
    config.timing = timing;
    config.parameter_sets = if in_band {
        ParameterSetMode::InBand
    } else {
        ParameterSetMode::OutOfBand
    };
    options.config = config;
    Ok(options)
}

fn print_capabilities(caps: &EncodeCapabilities) {
    let profile = match (caps.profile.codec, caps.profile.profile_idc) {
        (EncodeCodec::H264, 66) => "H.264 Constrained Baseline",
        (EncodeCodec::H264, 77) => "H.264 Main",
        (EncodeCodec::H264, _) => "H.264 High",
        (EncodeCodec::H265, 2) => "H.265 Main 10",
        (EncodeCodec::H265, _) => "H.265 Main",
    };
    println!(
        "{profile}: level_idc {} coded {}x{}..{}x{} access {}x{} input {}x{} dpb {} refs {} \
         offset_align {} size_align {} flags {:#x} rc {:#x} layers {} max_bitrate {} \
         quality_levels {} feedback {:#x} slices {} p_l0 {} b_l0 {} l1 {} qp {}..{} \
         gop_remaining {} codec_flags {:#x} std_flags {:#x} ctb {:#x} tbs {:#x}",
        caps.max_level_idc,
        caps.min_coded_extent.width,
        caps.min_coded_extent.height,
        caps.max_coded_extent.width,
        caps.max_coded_extent.height,
        caps.picture_access_granularity.width,
        caps.picture_access_granularity.height,
        caps.input_granularity.width,
        caps.input_granularity.height,
        caps.max_dpb_slots,
        caps.max_active_reference_pictures,
        caps.min_bitstream_buffer_offset_alignment,
        caps.min_bitstream_buffer_size_alignment,
        caps.flags.as_raw(),
        caps.rate_control_modes.as_raw(),
        caps.max_rate_control_layers,
        caps.max_bitrate,
        caps.max_quality_levels,
        caps.feedback_flags.as_raw(),
        caps.max_slices,
        caps.max_p_l0,
        caps.max_b_l0,
        caps.max_l1,
        caps.min_qp,
        caps.max_qp,
        caps.requires_gop_remaining_frames,
        caps.codec_flags,
        caps.std_syntax_flags,
        caps.ctb_sizes,
        caps.transform_block_sizes,
    );
    for (level, pref) in caps.quality_levels.iter().enumerate() {
        println!(
            "  quality {level}: rc {:#x} gop {} idr {} b {} qp {:?} l0 {} l1 {}",
            pref.rate_control_mode.as_raw(),
            pref.gop_frame_count,
            pref.idr_period,
            pref.consecutive_b_frames,
            pref.constant_qp,
            pref.max_l0,
            pref.max_l1
        );
    }
}

/// A memory type of `bits` with `flags`.
fn memory_type(device: &VideoDevice, bits: u32, flags: vk::MemoryPropertyFlags) -> Option<u32> {
    // SAFETY: the physical device is the instance's.
    let properties = unsafe {
        device
            .instance()
            .get_physical_device_memory_properties(device.physical_device())
    };
    (0..properties.memory_type_count).find(|&i| {
        bits & (1 << i) != 0
            && properties
                .memory_types
                .get(i as usize)
                .is_some_and(|t| t.property_flags.contains(flags))
    })
}

const HOST: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::from_raw(
    vk::MemoryPropertyFlags::HOST_VISIBLE.as_raw()
        | vk::MemoryPropertyFlags::HOST_COHERENT.as_raw(),
);

/// Host-visible memory, mapped while it lives.
struct Mapped {
    device: Arc<VideoDevice>,
    memory: vk::DeviceMemory,
    type_index: u32,
    size: u64,
    mapped: NonNull<u8>,
}

// SAFETY: the mapping is the memory's alone; GPU work and this tool take
// turns at it, ordered by the encoder's semaphores.
unsafe impl Send for Mapped {}
// SAFETY: as above.
unsafe impl Sync for Mapped {}

impl Mapped {
    /// `size` bytes of a host-visible memory type of `bits`.
    fn new(device: &Arc<VideoDevice>, size: u64, bits: u32) -> Result<Self, String> {
        let type_index = memory_type(device, bits, HOST)
            .ok_or("no host-visible memory type takes the buffer")?;
        let vk_device = device.device();
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
            Ok(mapped) => mapped,
            Err(error) => {
                // SAFETY: the memory is unmapped and used by nothing.
                unsafe { vk_device.free_memory(memory, None) };
                return Err(error.to_string());
            }
        };
        Ok(Self {
            device: Arc::clone(device),
            memory,
            type_index,
            size,
            mapped: NonNull::new(mapped.cast()).ok_or("a null mapping")?,
        })
    }

    /// Its bytes from `at`, `len` of them.
    ///
    /// # Safety
    ///
    /// No GPU work writes them meanwhile.
    unsafe fn bytes(&self, at: u64, len: u64) -> &[u8] {
        let at = usize::try_from(at.min(self.size)).unwrap_or(0);
        let len = usize::try_from(len).unwrap_or(0);
        // SAFETY: within the mapping, which the caller says no GPU work
        // writes meanwhile.
        unsafe { std::slice::from_raw_parts(self.mapped.as_ptr().add(at), len) }
    }

    /// Its bytes from `at`, `len` of them, to write.
    ///
    /// # Safety
    ///
    /// No GPU work reads or writes them meanwhile.
    #[expect(
        clippy::mut_from_ref,
        reason = "the mapping is the GPU's and ours in turn"
    )]
    unsafe fn bytes_mut(&self, at: u64, len: u64) -> &mut [u8] {
        let at = usize::try_from(at.min(self.size)).unwrap_or(0);
        let len = usize::try_from(len).unwrap_or(0);
        // SAFETY: within the mapping, which the caller says no GPU work
        // uses meanwhile.
        unsafe { std::slice::from_raw_parts_mut(self.mapped.as_ptr().add(at), len) }
    }

    fn import(&self, at: u64) -> HostImport {
        HostImport {
            memory: self.memory,
            offset: at,
            size: self.size - at,
            type_index: self.type_index,
            handle_type: vk::ExternalMemoryHandleTypeFlags::empty(),
        }
    }
}

impl Drop for Mapped {
    fn drop(&mut self) {
        // SAFETY: the memory is ours; whatever used it is done: its buffers
        // hold it through `keep` until their work is.
        unsafe { self.device.device().free_memory(self.memory, None) };
    }
}

/// The free slots of a pool, which a slot's holder gives back as it is
/// dropped.
struct Slots {
    give: Sender<usize>,
    free: Receiver<usize>,
}

impl Slots {
    fn new(count: usize) -> Self {
        let (give, free) = channel();
        for slot in 0..count {
            let _ = give.send(slot);
        }
        Self { give, free }
    }

    fn take(&self) -> Option<usize> {
        self.free.try_recv().ok()
    }

    /// What gives a slot back.
    fn giver(&self) -> Sender<usize> {
        self.give.clone()
    }
}

/// Bytes of a frame: NV12 or P010 planes, packed.
fn frame_bytes(width: u32, height: u32, sample: u32) -> (u64, u64) {
    let luma = u64::from(width) * u64::from(height) * u64::from(sample);
    (luma, luma / 2)
}

/// The frames' sources: images read in place, or host-visible planes.
enum Sources {
    Images(Images),
    Buffers(Buffers),
}

/// A frame's source, given back to its pool when the encoder drops it.
struct Source {
    picture: SourcePicture,
    slot: usize,
    slots: Sender<usize>,
    _keep: Option<Box<dyn Send + Sync>>,
}

impl EncodeSource for Source {
    fn picture(&self) -> SourcePicture {
        self.picture
    }
}

impl Drop for Source {
    fn drop(&mut self) {
        let _ = self.slots.send(self.slot);
    }
}

/// A source uploaded into an image, and the semaphore value its upload
/// signals.
type Upload = (Source, (vk::Semaphore, u64));

/// Device-local source images, each filled from a staging buffer of its
/// own on the graphics queue.
struct Images {
    device: Arc<VideoDevice>,
    queue: vk::Queue,
    pool: vk::CommandPool,
    slots: Slots,
    images: Vec<(vk::Image, vk::DeviceMemory, vk::Buffer, vk::CommandBuffer)>,
    staging: Mapped,
    frame: (u64, u64),
    size: (u32, u32),
    semaphore: vk::Semaphore,
    value: u64,
    fresh: Vec<bool>,
}

impl Images {
    fn new(
        device: &Arc<VideoDevice>,
        spec: &SourceSpec,
        size: (u32, u32),
        count: usize,
    ) -> Result<Self, String> {
        let vk_device = device.device();
        let queue = device
            .graphics_queue()
            .ok_or("no graphics queue to upload on")?;
        let sample = u32::from(spec.profile.bit_depth() > 8) + 1;
        let frame = frame_bytes(size.0, size.1, sample);
        let frame_size = frame.0 + frame.1;
        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(device.graphics_family());
        // SAFETY: `pool_info` lives through the call.
        let pool = unsafe { vk_device.create_command_pool(&pool_info, None) }
            .map_err(|e| e.to_string())?;
        let mut type_info =
            vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
        let info = vk::SemaphoreCreateInfo::default().push_next(&mut type_info);
        // SAFETY: `info` and its chain live through the call.
        let semaphore =
            unsafe { vk_device.create_semaphore(&info, None) }.map_err(|e| e.to_string())?;
        let staging_info = vk::BufferCreateInfo::default()
            .size(frame_size * count as u64)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: `staging_info` lives through the call.
        let probe =
            unsafe { vk_device.create_buffer(&staging_info, None) }.map_err(|e| e.to_string())?;
        // SAFETY: the buffer is the device's.
        let requirements = unsafe { vk_device.get_buffer_memory_requirements(probe) };
        // SAFETY: bound to nothing, used by nothing.
        unsafe { vk_device.destroy_buffer(probe, None) };
        let staging = Mapped::new(device, requirements.size, requirements.memory_type_bits)?;
        let mut images = Vec::with_capacity(count);
        for slot in 0..count {
            let image = spec
                .with_create_info(
                    device,
                    vk::ImageUsageFlags::TRANSFER_DST,
                    vk::ImageCreateFlags::empty(),
                    // SAFETY: `info` and its chain live through the call.
                    |info| unsafe { vk_device.create_image(info, None) },
                )
                .map_err(|e| e.to_string())?;
            // SAFETY: the image is the device's.
            let needs = unsafe { vk_device.get_image_memory_requirements(image) };
            let type_index = memory_type(
                device,
                needs.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .ok_or("no device-local memory type takes a source image")?;
            let info = vk::MemoryAllocateInfo::default()
                .allocation_size(needs.size)
                .memory_type_index(type_index);
            // SAFETY: `info` lives through the call.
            let memory =
                unsafe { vk_device.allocate_memory(&info, None) }.map_err(|e| e.to_string())?;
            // SAFETY: the image is unbound, the memory of its requirements.
            unsafe { vk_device.bind_image_memory(image, memory, 0) }.map_err(|e| e.to_string())?;
            let buffer_info = staging_info.size(frame_size);
            // SAFETY: `buffer_info` lives through the call.
            let buffer = unsafe { vk_device.create_buffer(&buffer_info, None) }
                .map_err(|e| e.to_string())?;
            // SAFETY: the buffer is unbound; the staging memory holds a
            // frame at each slot's offset, aligned as the probe's size was.
            unsafe {
                vk_device.bind_buffer_memory(buffer, staging.memory, frame_size * slot as u64)
            }
            .map_err(|e| e.to_string())?;
            let allocate = vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            // SAFETY: `allocate` lives through the call.
            let command = unsafe { vk_device.allocate_command_buffers(&allocate) }
                .map_err(|e| e.to_string())?[0];
            images.push((image, memory, buffer, command));
        }
        Ok(Self {
            device: Arc::clone(device),
            queue,
            pool,
            slots: Slots::new(count),
            images,
            staging,
            frame,
            size,
            semaphore,
            value: 0,
            fresh: vec![true; count],
        })
    }

    /// The next frame from `input`, uploaded into a free image: the source,
    /// and the semaphore value the encode waits for. `None` at the input's
    /// end.
    fn next(&mut self, input: &mut impl Read) -> Result<Option<Upload>, String> {
        let slot = self.slots.take().ok_or("every source image is in use")?;
        let frame_size = self.frame.0 + self.frame.1;
        let (image, _, buffer, command) = self.images[slot];
        // SAFETY: the slot is free: the encode that read the image, and so
        // the upload before it, is done.
        let bytes = unsafe { self.staging.bytes_mut(frame_size * slot as u64, frame_size) };
        if !read_frame(input, bytes)? {
            let _ = self.slots.giver().send(slot);
            return Ok(None);
        }
        let vk_device = self.device.device();
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: the command buffer's last submission is done (as above).
        unsafe { vk_device.begin_command_buffer(command, &begin) }.map_err(|e| e.to_string())?;
        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);
        let barrier = [vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .old_layout(if self.fresh[slot] {
                vk::ImageLayout::UNDEFINED
            } else {
                vk::ImageLayout::GENERAL
            })
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(range)];
        self.fresh[slot] = false;
        // SAFETY: recording; the image is the device's.
        unsafe {
            vk_device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barrier,
            );
        }
        let (width, height) = self.size;
        let region = |aspect, offset, w, h| {
            vk::BufferImageCopy::default()
                .buffer_offset(offset)
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(aspect)
                        .layer_count(1),
                )
                .image_extent(vk::Extent3D {
                    width: w,
                    height: h,
                    depth: 1,
                })
        };
        let regions = [
            region(vk::ImageAspectFlags::PLANE_0, 0, width, height),
            region(
                vk::ImageAspectFlags::PLANE_1,
                self.frame.0,
                width / 2,
                height / 2,
            ),
        ];
        // SAFETY: recording; the buffer holds the frame, the image is in
        // GENERAL.
        unsafe {
            vk_device.cmd_copy_buffer_to_image(
                command,
                buffer,
                image,
                vk::ImageLayout::GENERAL,
                &regions,
            );
        }
        // SAFETY: recording ended once.
        unsafe { vk_device.end_command_buffer(command) }.map_err(|e| e.to_string())?;
        self.value += 1;
        let values = [self.value];
        let mut timeline =
            vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&values);
        let commands = [command];
        let signals = [self.semaphore];
        let submit = [vk::SubmitInfo::default()
            .command_buffers(&commands)
            .signal_semaphores(&signals)
            .push_next(&mut timeline)];
        // SAFETY: the graphics queue is ours alone in this tool; the
        // command buffer is recorded and its resources live.
        unsafe { vk_device.queue_submit(self.queue, &submit, vk::Fence::null()) }
            .map_err(|e| e.to_string())?;
        Ok(Some((
            Source {
                picture: SourcePicture::Image(image),
                slot,
                slots: self.slots.giver(),
                _keep: None,
            },
            (self.semaphore, self.value),
        )))
    }
}

impl Drop for Images {
    fn drop(&mut self) {
        let vk_device = self.device.device();
        // SAFETY: the device is ours to wait on; nothing submits meanwhile.
        let _ = unsafe { vk_device.device_wait_idle() };
        for &(image, memory, buffer, _) in &self.images {
            // SAFETY: idle above; each the device's, used by nothing now.
            unsafe { vk_device.destroy_image(image, None) };
            // SAFETY: as above.
            unsafe { vk_device.free_memory(memory, None) };
            // SAFETY: as above.
            unsafe { vk_device.destroy_buffer(buffer, None) };
        }
        // SAFETY: as above.
        unsafe { vk_device.destroy_command_pool(self.pool, None) };
        // SAFETY: as above.
        unsafe { vk_device.destroy_semaphore(self.semaphore, None) };
    }
}

/// Frames in host-visible memory, read there by the encode's GPU copy.
struct Buffers {
    device: Arc<VideoDevice>,
    memory: Arc<Mapped>,
    slots: Slots,
    frame: (u64, u64),
    size: (u32, u32),
    sample: u32,
}

impl Buffers {
    fn new(
        device: &Arc<VideoDevice>,
        spec: &SourceSpec,
        size: (u32, u32),
        count: usize,
    ) -> Result<Self, String> {
        let sample = u32::from(spec.profile.bit_depth() > 8) + 1;
        let frame = frame_bytes(size.0, size.1, sample);
        // Each frame at a 4 KiB page, as an IOSurface's planes start.
        let stride = (frame.0 + frame.1).next_multiple_of(4096);
        let families = SourceSpec::families(device);
        let info = vk::BufferCreateInfo::default()
            .size(stride * count as u64)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC);
        let info = if families.len() > 1 {
            info.sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(&families)
        } else {
            info.sharing_mode(vk::SharingMode::EXCLUSIVE)
        };
        let vk_device = device.device();
        // SAFETY: `info` lives through the call.
        let probe = unsafe { vk_device.create_buffer(&info, None) }.map_err(|e| e.to_string())?;
        // SAFETY: the buffer is the device's.
        let requirements = unsafe { vk_device.get_buffer_memory_requirements(probe) };
        // SAFETY: bound to nothing, used by nothing.
        unsafe { vk_device.destroy_buffer(probe, None) };
        let memory = Mapped::new(
            device,
            requirements.size.max(stride * count as u64),
            requirements.memory_type_bits,
        )?;
        Ok(Self {
            device: Arc::clone(device),
            memory: Arc::new(memory),
            slots: Slots::new(count),
            frame,
            size,
            sample,
        })
    }

    fn stride(&self) -> u64 {
        (self.frame.0 + self.frame.1).next_multiple_of(4096)
    }

    /// The next frame from `input`, read into a free slot's pages.
    fn next(&mut self, input: &mut impl Read) -> Result<Option<Source>, String> {
        let slot = self.slots.take().ok_or("every source buffer is in use")?;
        let at = self.stride() * slot as u64;
        // SAFETY: the slot is free: the encode that copied it is done.
        let bytes = unsafe { self.memory.bytes_mut(at, self.frame.0 + self.frame.1) };
        if !read_frame(input, bytes)? {
            let _ = self.slots.giver().send(slot);
            return Ok(None);
        }
        let (width, height) = self.size;
        let pitch = width * self.sample;
        let keep: Arc<dyn Send + Sync> = Arc::clone(&self.memory) as Arc<dyn Send + Sync>;
        // SAFETY: the memory is the device's, alive while `keep` lives; the
        // slot is the encoder's until the source is dropped.
        let source = unsafe {
            ImportedSource::new(
                &self.device,
                &self.memory.import(at),
                (0, pitch, height, pitch),
                (self.frame.0, pitch, height / 2, pitch),
                keep,
            )
        }
        .map_err(|e| e.to_string())?
        .ok_or("the device takes no buffer over the source's pages")?;
        Ok(Some(Source {
            picture: source.picture(),
            slot,
            slots: self.slots.giver(),
            _keep: Some(Box::new(source)),
        }))
    }
}

/// Reads one frame into `bytes`: `false` at the input's end.
fn read_frame(input: &mut impl Read, bytes: &mut [u8]) -> Result<bool, String> {
    let mut read = 0;
    while read < bytes.len() {
        match input.read(&mut bytes[read..]) {
            Ok(0) if read == 0 => return Ok(false),
            Ok(0) => return Err("the input ends inside a frame".into()),
            Ok(n) => read += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e.to_string()),
        }
    }
    Ok(true)
}

/// A bitstream target whose bytes this tool reads.
trait Readable: BitstreamTarget {
    /// `len` bytes from `at` in the target's buffer, once its picture is
    /// done.
    fn read(&self, at: u64, len: u64) -> &[u8];
}

impl Readable for HostBitstream {
    fn read(&self, at: u64, len: u64) -> &[u8] {
        let bytes = self.bytes();
        let from = usize::try_from(at).unwrap_or(usize::MAX).min(bytes.len());
        let to = from
            .saturating_add(usize::try_from(len).unwrap_or(0))
            .min(bytes.len());
        &bytes[from..to]
    }
}

/// Bitstream slots in one host-visible allocation, each written in place
/// through an [`ImportedBitstream`].
struct ImportedTargets {
    device: Arc<VideoDevice>,
    count: usize,
    made: Option<(Arc<Mapped>, u64, Slots)>,
}

/// A slot of [`ImportedTargets`], given back as its picture is dropped.
struct ImportedTarget {
    target: ImportedBitstream,
    memory: Arc<Mapped>,
    at: u64,
    slot: usize,
    slots: Sender<usize>,
}

impl BitstreamTarget for ImportedTarget {
    fn bitstream(&self) -> reims_vgpu_video_vulkan::BitstreamBuffer {
        self.target.bitstream()
    }
}

impl Readable for ImportedTarget {
    fn read(&self, at: u64, len: u64) -> &[u8] {
        // The buffer starts at the import's offset less the target's
        // offset in it: the buffer's alignment below the slot.
        // SAFETY: the picture is done: its bytes are ours to read.
        unsafe {
            self.memory
                .bytes(self.at - self.target.bitstream().offset + at, len)
        }
    }
}

impl Drop for ImportedTarget {
    fn drop(&mut self) {
        let _ = self.slots.send(self.slot);
    }
}

impl BitstreamSource for ImportedTargets {
    type Target = ImportedTarget;

    fn target(&mut self, spec: &BitstreamSpec) -> Result<ImportedTarget, Error> {
        let short = |why: &str| Error::Unsupported(why.to_owned());
        if self.made.is_none() {
            // Each slot at a page, as the guest's buffers start.
            let stride = spec.size.next_multiple_of(4096);
            let total = stride * self.count as u64;
            let vk_device = self.device.device();
            let probe = spec.with_create_info(&self.device, total, |info| {
                // SAFETY: `info` and its chain live through the call.
                unsafe { vk_device.create_buffer(info, None) }
            });
            let probe = probe.map_err(|e| Error::Vulkan("vkCreateBuffer", e))?;
            // SAFETY: the buffer is the device's.
            let requirements = unsafe { vk_device.get_buffer_memory_requirements(probe) };
            // SAFETY: bound to nothing, used by nothing.
            unsafe { vk_device.destroy_buffer(probe, None) };
            let memory = Mapped::new(
                &self.device,
                requirements.size.max(total),
                requirements.memory_type_bits,
            )
            .map_err(|e| short(&e))?;
            self.made = Some((Arc::new(memory), stride, Slots::new(self.count)));
        }
        let Some((memory, stride, slots)) = &self.made else {
            return Err(short("no bitstream memory"));
        };
        let slot = slots
            .take()
            .ok_or_else(|| short("every bitstream slot is in use"))?;
        let at = stride * slot as u64;
        let keep: Arc<dyn Send + Sync> = Arc::clone(memory) as Arc<dyn Send + Sync>;
        // SAFETY: the memory is the device's, alive while `keep` lives; the
        // slot is the encoder's until its target is dropped.
        let target = unsafe {
            ImportedBitstream::new(&self.device, spec, &memory.import(at), 0, spec.size, keep)
        }?;
        let Some(target) = target else {
            let _ = slots.giver().send(slot);
            return Err(short("the device takes no bitstream buffer over the slot"));
        };
        Ok(ImportedTarget {
            target,
            memory: Arc::clone(memory),
            at,
            slot,
            slots: slots.giver(),
        })
    }
}

/// An encoder of either codec, as this tool drives it.
enum AnyEncoder<B: BitstreamSource> {
    H264(H264Encoder<Source, B>),
    H265(H265Encoder<Source, B>),
}

impl<B: BitstreamSource> AnyEncoder<B> {
    fn new(
        device: &Arc<VideoDevice>,
        profile: EncodeProfile,
        config: &EncodeConfig,
        targets: B,
    ) -> Result<Self, Error> {
        Ok(match profile.codec {
            EncodeCodec::H264 => Self::H264(H264Encoder::new(device, profile, config, targets)?),
            EncodeCodec::H265 => Self::H265(H265Encoder::new(device, profile, config, targets)?),
        })
    }

    fn source_spec(&self) -> SourceSpec {
        match self {
            Self::H264(e) => e.source_spec(),
            Self::H265(e) => e.source_spec(),
        }
    }

    fn encode(
        &mut self,
        source: Source,
        params: FrameParams,
    ) -> Result<Vec<EncodedFrame<B::Target>>, Error> {
        match self {
            Self::H264(e) => e.encode(source, params),
            Self::H265(e) => e.encode(source, params),
        }
    }

    fn flush(&mut self) -> Result<Vec<EncodedFrame<B::Target>>, Error> {
        match self {
            Self::H264(e) => e.flush(),
            Self::H265(e) => e.flush(),
        }
    }

    fn set_rate_control(&mut self, rate: RateControl, vbv: Option<Vbv>) -> Result<(), Error> {
        match self {
            Self::H264(e) => e.set_rate_control(rate, vbv),
            Self::H265(e) => e.set_rate_control(rate, vbv),
        }
    }

    fn parameter_sets(&self) -> &[u8] {
        match self {
            Self::H264(e) => e.parameter_sets(),
            Self::H265(e) => e.parameter_sets(),
        }
    }

    fn record(&self) -> Vec<u8> {
        match self {
            Self::H264(e) => e.avcc(),
            Self::H265(e) => e.hvcc(),
        }
    }

    fn counts(&self) -> SourceCounts {
        match self {
            Self::H264(e) => e.source_counts(),
            Self::H265(e) => e.source_counts(),
        }
    }

    fn describe(&self) -> String {
        let (level, extent, b, refs) = match self {
            Self::H264(e) => (
                e.level_idc(),
                e.coded_extent(),
                e.b_frames(),
                e.references(),
            ),
            Self::H265(e) => (
                e.level_idc(),
                e.coded_extent(),
                e.b_frames(),
                e.references(),
            ),
        };
        format!(
            "level_idc {level} coded {}x{} b_frames {b} references {refs}",
            extent.width, extent.height
        )
    }
}

/// What the output has seen.
#[derive(Default)]
struct Totals {
    frames: u64,
    bytes: u64,
    gpu: Vec<Duration>,
    copy: Vec<Duration>,
}

fn kind_name(kind: PictureKind) -> &'static str {
    match kind {
        PictureKind::Idr => "IDR",
        PictureKind::I => "I",
        PictureKind::P => "P",
        PictureKind::B => "B",
    }
}

/// Writes `frame` out once it is done, and its line.
fn emit<T: Readable>(
    frame: &EncodedFrame<T>,
    parameter_sets: &[u8],
    out_of_band: bool,
    output: &mut impl Write,
    totals: &mut Totals,
) -> Result<(), String> {
    let feedback = frame.feedback(TIMEOUT).map_err(|e| e.to_string())?;
    if feedback.status != EncodeStatus::Complete {
        return Err(format!(
            "frame {}: the encode says {:?}",
            frame.id, feedback.status
        ));
    }
    let mut ps = 0u64;
    if out_of_band && frame.kind == PictureKind::Idr {
        output
            .write_all(parameter_sets)
            .map_err(|e| e.to_string())?;
        ps = parameter_sets.len() as u64;
    }
    let target = frame.target();
    match (feedback.range(), feedback.parameter_sets) {
        (Some((at, len)), _) => {
            output
                .write_all(target.read(at, len))
                .map_err(|e| e.to_string())?;
        }
        (None, Some((at, len))) => {
            output
                .write_all(target.read(at, len))
                .map_err(|e| e.to_string())?;
            let (at, len) = feedback.picture;
            output
                .write_all(target.read(at, len))
                .map_err(|e| e.to_string())?;
        }
        (None, None) => unreachable!("a frame with no parameter sets is one range"),
    }
    if let Some((_, len)) = feedback.parameter_sets {
        ps += len;
    }
    let time = frame.gpu_time(TIMEOUT).map_err(|e| e.to_string())?;
    let (gpu_us, copy_us) = time.map_or((-1.0, -1.0), |t| {
        totals.gpu.push(t.total);
        totals.copy.push(t.copy);
        (t.total.as_secs_f64() * 1e6, t.copy.as_secs_f64() * 1e6)
    });
    println!(
        "frame {} pts {} dts {} type {} poc {} frame_num {} ref {} at {} bytes {} ps {} status {:?} gpu_us {gpu_us:.1} copy_us {copy_us:.1}",
        frame.id,
        frame.pts,
        frame.dts,
        kind_name(frame.kind),
        frame.order_count,
        frame.frame_num,
        u8::from(frame.reference),
        feedback.picture.0,
        feedback.picture.1,
        ps,
        feedback.status,
    );
    totals.frames += 1;
    totals.bytes += feedback.picture.1 + ps;
    Ok(())
}

fn summary(label: &str, times: &[Duration]) -> String {
    if times.is_empty() {
        return format!("{label} untimed");
    }
    let mut sorted = times.to_vec();
    sorted.sort();
    let mean = sorted.iter().sum::<Duration>() / u32::try_from(sorted.len()).unwrap_or(u32::MAX);
    let at = |q: f64| sorted[((sorted.len() - 1) as f64 * q).round() as usize];
    format!(
        "{label} mean {:.1} us median {:.1} us p95 {:.1} us max {:.1} us",
        mean.as_secs_f64() * 1e6,
        at(0.5).as_secs_f64() * 1e6,
        at(0.95).as_secs_f64() * 1e6,
        at(1.0).as_secs_f64() * 1e6,
    )
}

fn encode_all<B>(device: &Arc<VideoDevice>, options: &Options, targets: B) -> Result<(), String>
where
    B: BitstreamSource,
    B::Target: Readable,
{
    let mut encoder = AnyEncoder::new(device, options.profile, &options.config, targets)
        .map_err(|e| e.to_string())?;
    let spec = encoder.source_spec();
    let size = (options.config.width, options.config.height);
    let count = usize::try_from(options.config.gop.b_frames).unwrap_or(8) + options.depth + 6;
    let mut sources = match options.source {
        SourceKind::Image => Sources::Images(Images::new(device, &spec, size, count)?),
        SourceKind::Buffer => Sources::Buffers(Buffers::new(device, &spec, size, count)?),
    };
    if let Some(path) = &options.record {
        std::fs::write(path, encoder.record()).map_err(|e| format!("{path}: {e}"))?;
    }
    let mut input = std::io::BufReader::with_capacity(
        1 << 22,
        File::open(&options.input).map_err(|e| format!("{}: {e}", options.input))?,
    );
    let mut output = BufWriter::new(
        File::create(&options.output).map_err(|e| format!("{}: {e}", options.output))?,
    );
    let parameter_sets = encoder.parameter_sets().to_vec();
    let out_of_band = options.config.parameter_sets == ParameterSetMode::OutOfBand;
    let mut pending: VecDeque<EncodedFrame<B::Target>> = VecDeque::new();
    let mut totals = Totals::default();
    let started = StdRuntime.now();
    let fps = options.config.frame_rate;
    let mut index = 0u64;
    loop {
        if options.frames.is_some_and(|n| index >= n) {
            break;
        }
        let (source, waits) = match &mut sources {
            Sources::Images(images) => match images.next(&mut input)? {
                Some((source, wait)) => (source, vec![wait]),
                None => break,
            },
            Sources::Buffers(buffers) => match buffers.next(&mut input)? {
                Some(source) => (source, Vec::new()),
                None => break,
            },
        };
        if let Some((at, rate)) = options.switch_rc
            && at == index
        {
            encoder
                .set_rate_control(rate, options.config.vbv)
                .map_err(|e| format!("frame {index}: {e}"))?;
        }
        // Time stamps in frames at the frame rate.
        let pts = i64::try_from(index).map_err(|e| e.to_string())?;
        let params = FrameParams {
            id: index,
            pts,
            force_idr: options.force_idr.contains(&index),
            waits,
        };
        let frames = encoder
            .encode(source, params)
            .map_err(|e| format!("frame {index}: {e}"))?;
        pending.extend(frames);
        while pending.len() > options.depth {
            if let Some(frame) = pending.pop_front() {
                emit(
                    &frame,
                    &parameter_sets,
                    out_of_band,
                    &mut output,
                    &mut totals,
                )?;
            }
        }
        index += 1;
    }
    pending.extend(encoder.flush().map_err(|e| e.to_string())?);
    for frame in pending.drain(..) {
        emit(
            &frame,
            &parameter_sets,
            out_of_band,
            &mut output,
            &mut totals,
        )?;
    }
    output.flush().map_err(|e| e.to_string())?;
    let elapsed = StdRuntime.elapsed_since(started);
    let counts = encoder.counts();
    let seconds =
        totals.frames as f64 * f64::from(fps.denominator) / f64::from(fps.numerator.max(1));
    eprintln!(
        "{}: {} frames {} bytes {:.0} bit/s in {:.3} s ({:.1} fps); {}; sources in_place {} gpu_copies {} cpu_copies {}",
        device.name(),
        totals.frames,
        totals.bytes,
        if seconds > 0.0 {
            totals.bytes as f64 * 8.0 / seconds
        } else {
            0.0
        },
        elapsed.as_secs_f64(),
        totals.frames as f64 / elapsed.as_secs_f64().max(1e-9),
        encoder.describe(),
        counts.in_place,
        counts.gpu_copies,
        counts.cpu_copies,
    );
    eprintln!("{}", summary("gpu", &totals.gpu));
    eprintln!("{}", summary("copy", &totals.copy));
    Ok(())
}

fn run() -> Result<(), String> {
    let options = options()?;
    let device = VideoDevice::create().map_err(|e| e.to_string())?;
    if options.capabilities {
        println!("{}", device.name());
        for caps in h264_encode_capabilities(&device)
            .iter()
            .chain(&h265_encode_capabilities(&device))
        {
            print_capabilities(caps);
        }
        return Ok(());
    }
    match options.bitstream {
        TargetKind::Host => encode_all(&device, &options, HostBitstreams::new(&device)),
        TargetKind::Import => encode_all(
            &device,
            &options,
            ImportedTargets {
                device: Arc::clone(&device),
                count: options.depth
                    + usize::try_from(options.config.gop.b_frames).unwrap_or(8)
                    + 4,
                made: None,
            },
        ),
    }
}
