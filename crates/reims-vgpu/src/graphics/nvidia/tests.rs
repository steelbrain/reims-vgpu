//! The NVIDIA arm on the NVIDIA host: a picture written by a copy within the GPU into a device
//! cache CUDA imported reads, through the runtime, as NVDEC's own read back of it; and
//! VideoToolbox's decoders of the arm land each picture in a lent surface's pages as NVDEC reads
//! it back. Off it: the CPU's copy of a picture read back into a surface's planes, and what a
//! VideoToolbox decoder sees of NVIDIA's errors.

use std::alloc::Layout;
use std::ptr::NonNull;
use std::sync::Arc;

use reims_vgpu_video::core::device_memory::{DeviceMemory, HostView, MemoryLife};
use reims_vgpu_video::core::fourcc::FourCc;
use reims_vgpu_video::core::runtime::{ExitStep, MonoTime, Rings, TimedOut};
use reims_vgpu_video::core::surfaces::{PlaneLayout, SurfaceError, SurfaceLayout};
use reims_vgpu_video::core::sync::{Condvar, MutexGuard};
use reims_vgpu_video::nvidia::{
    EncodeParams, EncodeProfile, EncodeRateControl, NvencEncoder, RawFrame, Tuning,
};
use reims_vgpu_vulkan::RtPlatform;
use reims_vgpu_vulkan::core::platform::Platform;
use reims_vgpu_vulkan::core::platform::blit::{BlitCmd, BlitEncoder as _};
use reims_vgpu_vulkan::core::platform::command::{CommandBuffer as _, CommandQueue as _};
use reims_vgpu_vulkan::core::platform::types::CommandBufferStatus;

use super::*;
use crate::graphics::video_choice::tested_gpu::Arm;

/// What the GPU's copies must leave as it was: the bytes past each row.
const UNTOUCHED: u8 = 0xee;

/// A 1080p frame of NV12 that changes with `n`: the luma a ramp, the chroma a slower one.
fn frame(width: usize, height: usize, n: u64) -> (Vec<u8>, Vec<u8>) {
    let shift = usize::try_from(n).expect("a frame number") * 7;
    let luma = (0..width * height)
        .map(|i| u8::try_from((i % width + i / width + shift) % 256).expect("a byte"))
        .collect();
    let chroma = (0..width * height / 2)
        .map(|i| u8::try_from((i % width / 4 + i / width + shift) % 256).expect("a byte"))
        .collect();
    (luma, chroma)
}

/// NVENC's settings for `profile` at `width` by `height`: constant QP, low latency, no B-frames.
fn params(profile: EncodeProfile, width: u32, height: u32) -> EncodeParams {
    EncodeParams {
        profile,
        width,
        height,
        frame_rate: (30, 1),
        rate_control: EncodeRateControl::ConstantQp {
            intra: 22,
            predicted: 24,
            bidirectional: 26,
        },
        buffer_milliseconds: 1000,
        gop_length: Some(30),
        b_frames: 0,
        level: None,
        cabac: None,
        full_range: false,
        colour: None,
        display_aspect: None,
        preset: 4,
        tuning: Tuning::LowLatency,
    }
}

/// Copies `length` bytes of `from` into `to` on `queue`, and waits for the copy.
fn copy<P: Platform>(queue: &P::CommandQueue, from: &P::Buffer, to: &P::Buffer, length: u64) {
    let cb = (queue.command_buffer(OnFault::Return))
        .expect("no raise")
        .expect("a command buffer");
    let blit = (cb.blit_encoder(OnFault::Return))
        .expect("no raise")
        .expect("a blit encoder");
    let copy = BlitCmd::CopyBufferToBuffer {
        source_buffer: Some(from),
        source_offset: 0,
        destination_buffer: Some(to),
        destination_offset: 0,
        size: length,
    };
    blit.encode(OnFault::Return, copy).expect("the copy");
    blit.encode(OnFault::Return, BlitCmd::EndEncoding)
        .expect("its end");
    cb.commit(OnFault::Return).expect("its commit");
    cb.wait_until_completed(OnFault::Return).expect("its wait");
    assert_eq!(cb.status(), CommandBufferStatus::COMPLETED);
}

/// As [`a_picture_lands_in_an_imported_device_cache_as_it_reads_back`], on the rt platform: its
/// exportable buffers are blocks of their own, exported as opaque fds, which CUDA imports, and
/// the acquire is rt's (`rt_platform::export`).
/// The rt device must be of a GPU the arm runs on (`video_choice::tested_gpu::check_rt`).
#[test]
#[ignore = "needs an NVIDIA GPU: run on the NVIDIA host"]
fn a_picture_lands_in_an_imported_rt_cache_as_it_reads_back() {
    let platform = crate::rt_test_end::platform();
    crate::entry::run(|_| {
        let device = platform.system_default_device().expect("an rt device");
        crate::graphics::video_choice::tested_gpu::check_rt(&device, Arm::Nvidia);
        let runtime = crate::video::runtime::PlatformRuntime::of(&platform);
        let gpu = super::super::video_choice::rt_nvidia_gpu(&device);
        let video = NvidiaVideo::<RtPlatform>::new(&device, gpu, &runtime)
            .expect("NVDEC on the device's GPU");
        lands_in_an_imported_device_cache(&device, &video, &runtime);
    });
}

/// On the rt platform the guest's caps name each codec NVDEC decodes on the GPU, with the
/// profiles it decodes, then each NVENC encodes, as on the Vulkan platform (the input released
/// to CUDA after each frame's copy, `encode::fill`), and no other; VideoToolbox's encoders are
/// NVENC's. They say each picture lands in the device's caches, as every cache on rt is a
/// shape's (`video_caps`), and the arm imports the device's exportable caches, rt's exported
/// blocks, to write them in.
/// The rt device must be of a GPU the arm runs on (`video_choice::tested_gpu::check_rt`).
#[test]
#[ignore = "needs an NVIDIA GPU: run on the NVIDIA host"]
fn the_rt_platforms_caps_name_what_nvdec_decodes_and_nvenc_encodes() {
    use reims_vgpu_graphics::wire::RGFX_VIDEO_CAPS_OUTPUT_CACHED;
    let platform = crate::rt_test_end::platform();
    crate::entry::run(|_| {
        let device = platform.system_default_device().expect("an rt device");
        crate::graphics::video_choice::tested_gpu::check_rt(&device, Arm::Nvidia);
        let runtime = crate::video::runtime::PlatformRuntime::of(&platform);
        let gpu = super::super::video_choice::rt_nvidia_gpu(&device);
        let nvdec = NvidiaVideo::<RtPlatform>::new(&device, gpu, &runtime)
            .expect("NVDEC on the device's GPU");
        let decodes = |codec, bit_depth| supported(&nvdec.gpu, codec, bit_depth).is_some();
        let mut expected = Vec::new();
        if decodes(NvdecCodec::H264, 8) {
            expected.push((RGFX_VIDEO_CODEC_H264, RGFX_VIDEO_H264_HIGH));
        }
        let hevc_profiles = (if decodes(NvdecCodec::Hevc, 8) {
            RGFX_VIDEO_HEVC_MAIN
        } else {
            0
        }) | (if decodes(NvdecCodec::Hevc, 10) {
            RGFX_VIDEO_HEVC_MAIN_10
        } else {
            0
        });
        if hevc_profiles != 0 {
            expected.push((RGFX_VIDEO_CODEC_HEVC, hevc_profiles));
        }
        assert!(!expected.is_empty(), "NVDEC decodes neither codec");
        let encodes: Vec<(u32, u32)> = (nvdec.encode.iter())
            .map(|caps| (caps.codec, caps.profiles))
            .collect();
        drop(nvdec);

        let video = (crate::graphics::rt_graphics_video(&device, Arc::clone(&runtime)))
            .expect("the rt platform's video on the NVIDIA GPU");
        let shapes = <RtPlatform as crate::platform::ApiPlatform>::SURFACE_SHAPES;
        let caps = super::super::video_caps(0, shapes, &*video);
        assert_ne!(caps.flags & RGFX_VIDEO_CAPS_OUTPUT_CACHED, 0);
        assert!(video.imports_device_caches(), "CUDA imports no rt export");
        let named = &caps.codecs[..usize::try_from(caps.codec_count).expect("a count")];
        assert_eq!(named.len(), expected.len() + encodes.len(), "{named:?}");
        let (decodes, encoded) = named.split_at(expected.len());
        for (caps, (codec, profiles)) in decodes.iter().zip(&expected) {
            assert_eq!((caps.codec, caps.direction), (*codec, RGFX_VIDEO_DECODE));
            assert_eq!(caps.profiles & profiles, *profiles, "codec {codec:#x}");
            if *codec == RGFX_VIDEO_CODEC_HEVC {
                assert_eq!(caps.profiles, *profiles, "HEVC's profiles");
            }
        }
        for (caps, (codec, profiles)) in encoded.iter().zip(&encodes) {
            assert_eq!((caps.codec, caps.direction), (*codec, RGFX_VIDEO_ENCODE));
            assert_eq!(caps.profiles, *profiles, "codec {codec:#x}");
        }
        assert_eq!(
            video.guest_encoders().len(),
            encodes.len(),
            "VideoToolbox's encoders"
        );
        assert!(video.guest_decodes());
    });
}

/// The body of the imported cache's tests, on `P`'s `device`, NVDEC's arm on it `video`.
fn lands_in_an_imported_device_cache<P: Platform>(
    device: &P::Device,
    video: &NvidiaVideo<P>,
    runtime: &Arc<dyn Runtime>,
) {
    let imports = (video.imports.as_ref()).expect("CUDA's imports of exportable buffers");
    let (width, height, pitch) = (1920u32, 1080u32, 2048usize);
    let rows = usize::try_from(height).expect("rows");
    let luma_at = 0x20;
    let chroma_at = luma_at + pitch * rows;
    let len = u64::try_from(chroma_at + pitch * rows / 2).expect("a length");
    let aligned = Some(u64::try_from(luma_at).expect("a byte"));
    let made =
        device.new_exportable_buffer_of_length(OnFault::Return, len, STORAGE_PRIVATE, aligned);
    let cache = made.expect("no raise").expect("an exportable buffer");
    let life = MemoryLife::new();
    let imported = (imports.get_or_make(
        &life.identity(),
        0,
        || import::<P>(&cache, &video.gpu),
        || "gone".to_owned(),
    ))
    .expect("the cache's import");
    let read_back = (device.new_buffer_of_length(OnFault::Return, len, 0))
        .expect("no raise")
        .expect("a Shared buffer");
    let fill = (device.new_buffer_of_length(OnFault::Return, len, 0))
        .expect("no raise")
        .expect("a Shared buffer");
    let length = usize::try_from(len).expect("a length");
    // SAFETY: the Shared buffer's `len` bytes, which no GPU work reaches yet.
    unsafe { std::ptr::write_bytes(fill.contents().as_ptr::<u8>(), UNTOUCHED, length) };
    let queue = device.new_command_queue().expect("a command queue");
    for (profile, codec) in [
        (EncodeProfile::H264High, AnnexBCodec::H264),
        (EncodeProfile::HevcMain, AnnexBCodec::Hevc),
    ] {
        let params = params(profile, width, height);
        let mut encoder =
            NvencEncoder::new(Arc::clone(&video.gpu), &params).expect("an NVENC session");
        let mut decoder = AnnexBDecoder::new(&video.gpu, codec, runtime);
        decoder
            .parameter_sets(encoder.sequence_parameters())
            .expect("the parameter sets");
        let mut pictures = 0;
        for n in 0..6 {
            let (luma, chroma) = frame(1920, 1080, n);
            let packets = (encoder.encode(&RawFrame {
                luma: (&luma, 1920),
                chroma: (&chroma, 1920),
                index: n,
                force_idr: n == 0,
            }))
            .expect("a frame NVENC takes");
            for packet in packets {
                let packet = packet.expect("a packet");
                let picture = (decoder.decode(&packet.data))
                    .expect("a decode")
                    .expect("a picture");
                copy::<P>(&queue, &fill, &cache, len);
                let plane = |offset, rows| DevicePlane {
                    buffer: &imported,
                    offset,
                    bytes_per_row: pitch,
                    row_bytes: 1920,
                    rows,
                };
                // SAFETY: the planes lie apart in the cache, which no GPU work of the
                // runtime's reaches until the call returns.
                let written = unsafe {
                    picture.write_into_device(
                        (width, height),
                        plane(luma_at, height),
                        plane(chroma_at, height / 2),
                        Box::new(()),
                    )
                };
                written.expect("the decode").expect("the GPU's copy");
                cache.external_write_done().expect("the acquire");
                copy::<P>(&queue, &cache, &read_back, len);
                let read = picture.read_back((width, height)).expect("a read back");
                // SAFETY: the Shared buffer's `len` bytes, which the copy above finished
                // writing.
                let bytes = unsafe {
                    std::slice::from_raw_parts(read_back.contents().as_ptr::<u8>(), length)
                };
                assert!(
                    bytes[..luma_at].iter().all(|&b| b == UNTOUCHED),
                    "{codec:?} picture {n}: the bytes before the planes"
                );
                for (read, at) in read.planes().iter().zip([luma_at, chroma_at]) {
                    assert_eq!(read.bytes_per_row, 1920);
                    for row in 0..usize::try_from(read.rows).expect("rows") {
                        let landed = &bytes[at + row * pitch..][..pitch];
                        assert_eq!(
                            &landed[..1920],
                            &read.bytes[row * 1920..][..1920],
                            "{codec:?} picture {n}, row {row} of the plane at {at}"
                        );
                        assert!(
                            landed[1920..].iter().all(|&b| b == UNTOUCHED),
                            "{codec:?} picture {n}: bytes past row {row} of the plane at {at}"
                        );
                    }
                }
                pictures += 1;
            }
        }
        assert_eq!(pictures, 6, "{codec:?}");
    }
    drop(imported);
    drop(life);
}

/// A runtime that records the name of each thread it starts, over the platform's.
struct Naming {
    platform: Arc<dyn Runtime>,
    started: Mutex<Vec<&'static str>>,
}

impl Runtime for Naming {
    fn spawn(&self, name: &'static str, body: Box<dyn FnOnce() + Send>) -> std::io::Result<()> {
        self.started.lock().push(name);
        self.platform.spawn(name, body)
    }

    fn now(&self) -> MonoTime {
        self.platform.now()
    }

    fn wait_until(
        &self,
        cv: &Condvar,
        rings: &mut MutexGuard<'_, Rings>,
        deadline: MonoTime,
    ) -> TimedOut {
        self.platform.wait_until(cv, rings, deadline)
    }

    fn exit_step(&self) -> Option<ExitStep> {
        self.platform.exit_step()
    }

    fn arm_exit(&self) {
        self.platform.arm_exit();
    }
}

/// The child's role in [`an_encode_first_processs_waits_run_on_the_arms_runtime`].
const ENCODE_FIRST_CHILD: &str = "RVG_TEST_NVIDIA_ENCODE_FIRST_CHILD";

/// An encode made before any decode in a process, as a guest's first use of NVIDIA's video may
/// be, waits on the runtime the arm opened with: the open hands it to the NVIDIA crate, so the
/// thread NVENC's locks are made on is the runtime's, with its exit gate. Handed by the Annex B
/// decoder alone, an encode-first process's waits ran on std's runtime, with no gate. Run in a
/// child process of its own, in which nothing made the crate's runtime before the open.
#[test]
#[ignore = "needs an NVIDIA GPU: run on the NVIDIA host"]
fn an_encode_first_processs_waits_run_on_the_arms_runtime() {
    if std::env::var_os(ENCODE_FIRST_CHILD).is_some() {
        let platform = crate::rt_test_end::platform();
        crate::entry::run(|_| {
            let device = platform.system_default_device().expect("an rt device");
            crate::graphics::video_choice::tested_gpu::check_rt(&device, Arm::Nvidia);
            let naming = Arc::new(Naming {
                platform: crate::video::runtime::PlatformRuntime::of(&platform),
                started: Mutex::new(Vec::new()),
            });
            let runtime: Arc<dyn Runtime> = Arc::<Naming>::clone(&naming);
            let gpu = super::super::video_choice::rt_nvidia_gpu(&device);
            let video = NvidiaVideo::<RtPlatform>::new(&device, gpu, &runtime)
                .expect("NVDEC on the device's GPU");
            let params = params(EncodeProfile::H264High, 1920, 1080);
            let mut encoder =
                NvencEncoder::new(Arc::clone(&video.gpu), &params).expect("an NVENC session");
            let (luma, chroma) = frame(1920, 1080, 0);
            let packets = (encoder.encode(&RawFrame {
                luma: (&luma, 1920),
                chroma: (&chroma, 1920),
                index: 0,
                force_idr: true,
            }))
            .expect("a frame NVENC takes");
            assert!(!packets.is_empty(), "no access unit for the IDR picture");
            for packet in packets {
                packet.expect("a packet");
            }
            let started = naming.started.lock().clone();
            assert!(
                started.contains(&"reims-vgpu-video.nvenc-locks"),
                "NVENC's lock thread was not the arm's runtime's: it started {started:?}"
            );
        });
        return;
    }
    let name = "graphics::nvidia::tests::an_encode_first_processs_waits_run_on_the_arms_runtime";
    let status = std::process::Command::new(std::env::current_exe().expect("the test"))
        .args([
            "--exact",
            name,
            "--nocapture",
            "--test-threads=1",
            "--ignored",
        ])
        .env(ENCODE_FIRST_CHILD, "1")
        .status()
        .expect("a child");
    assert!(status.success(), "the child's encode: {status:?}");
}

/// The alignment of a test surface's memory: whole pages on every host's page size.
const PAGE: usize = 64 << 10;

/// Whole pages of host memory, as the view of a surface the guest lent.
struct Pages {
    first: NonNull<u8>,
    len: usize,
}

// SAFETY: the memory is the test's own allocation, which any thread may reach; the test reads it
// only once the GPU's writes into it are done.
unsafe impl Send for Pages {}
// SAFETY: as above.
unsafe impl Sync for Pages {}

impl Pages {
    fn layout(len: usize) -> Layout {
        Layout::from_size_align(len, PAGE).expect("a layout")
    }

    fn new(len: usize) -> Arc<Self> {
        // SAFETY: the layout's size is not zero.
        let at = unsafe { std::alloc::alloc(Self::layout(len)) };
        Arc::new(Self {
            first: NonNull::new(at).expect("the test's memory"),
            len,
        })
    }

    /// Sets every byte to `byte`, with no write of the GPU's in flight.
    fn fill(&self, byte: u8) {
        // SAFETY: `len` bytes allocated from `first`, which nothing borrows and the GPU does not
        // write meanwhile.
        unsafe { std::ptr::write_bytes(self.first.as_ptr(), byte, self.len) };
    }

    /// Its bytes, which nothing writes while they are borrowed: a decode's writes are done when
    /// it returns.
    fn bytes(&self) -> &[u8] {
        // SAFETY: `len` bytes allocated from `first`, initialised by `fill` first, with no write
        // in flight while borrowed.
        unsafe { std::slice::from_raw_parts(self.first.as_ptr(), self.len) }
    }
}

impl Drop for Pages {
    fn drop(&mut self) {
        // SAFETY: allocated by `new` with this layout, freed once, after the registration that
        // held the view went.
        unsafe { std::alloc::dealloc(self.first.as_ptr(), Self::layout(self.len)) };
    }
}

/// The pages as the host's video takes a lent surface's: their identity lives as long as this.
struct PagesMemory {
    life: MemoryLife,
    pages: Arc<Pages>,
}

impl DeviceMemory for PagesMemory {
    fn identity(&self) -> MemoryIdentity {
        self.life.identity()
    }

    fn length(&self) -> usize {
        self.pages.len
    }

    fn host_view(&self) -> Result<HostView, SurfaceError> {
        Ok(HostView {
            first: self.pages.first,
            size: self.pages.len,
            offset: 0,
            hold: Box::new(Arc::clone(&self.pages)),
        })
    }
}

/// A `width` by `height` surface of `format` in two planes, luma at `luma_at` and chroma on the
/// next 4 KiB, each row `pitch` bytes apart: as CoreVideo lays out a pool's IOSurface.
fn surface_layout(
    format: [u8; 4],
    (width, height): (u32, u32),
    luma_at: usize,
    pitch: usize,
) -> SurfaceLayout {
    let rows = usize::try_from(height).expect("rows");
    let chroma_at = (luma_at + pitch * rows).next_multiple_of(0x1000);
    SurfaceLayout {
        pixel_format: FourCc::new(format),
        width,
        height,
        planes: vec![
            PlaneLayout {
                offset: luma_at,
                bytes_per_row: pitch,
                width,
                height,
            },
            PlaneLayout {
                offset: chroma_at,
                bytes_per_row: pitch,
                width: width / 2,
                height: height / 2,
            },
        ],
        bytes_per_row: 0,
    }
}

/// Each picture of NVENC's H.264 and HEVC lands, through a VideoToolbox decoder of the arm, in a
/// lent `420v` surface's pages at CoreVideo's pitches, its planes past the surface's first byte,
/// by the GPU into the pages registered with CUDA, with no picture copied by the CPU: at 1080p
/// with 1984-byte rows, and at 640x360 with rows of 640 bytes, the pitch radeonsi's import check
/// refuses (`graphics::vaapi`), each a pitch not a multiple of 256. Each row is as a decoder of
/// the same access units reads its picture back, and the bytes before the planes, past each row
/// and past the chroma are as they were. The arm has a decoder of each codec it decodes and of
/// no other, and a picture of 8-bit samples takes no 10-bit surface.
#[test]
#[ignore = "needs an NVIDIA GPU: run on the NVIDIA host"]
fn a_videotoolbox_picture_lands_in_the_lent_surfaces_pages_as_it_reads_back() {
    let platform = crate::rt_test_end::platform();
    crate::entry::run(|_| {
        let device = platform.system_default_device().expect("an rt device");
        crate::graphics::video_choice::tested_gpu::check_rt(&device, Arm::Nvidia);
        let runtime = crate::video::runtime::PlatformRuntime::of(&platform);
        let gpu = super::super::video_choice::rt_nvidia_gpu(&device);
        let video = NvidiaVideo::<RtPlatform>::new(&device, gpu, &runtime);
        let video = Arc::new(video.expect("NVDEC on the device's GPU"));
        let shared = Shared(Arc::clone(&video));
        assert!(shared.guest_decodes());
        assert!(shared.guest_decoder(RGFX_VIDEO_CODEC_H264).is_some());
        assert!(shared.guest_decoder(RGFX_VIDEO_CODEC_HEVC).is_some());
        assert!(
            shared.guest_decoder(0x7f).is_none(),
            "a codec NVDEC has not"
        );
        for (width, height, pitch) in [(1920u32, 1080u32, 1984usize), (640, 360, 640)] {
            let luma_at = 0x400;
            let (columns, rows) = (
                usize::try_from(width).expect("columns"),
                usize::try_from(height).expect("rows"),
            );
            let layout = surface_layout(*b"420v", (width, height), luma_at, pitch);
            let [luma, chroma] = [layout.planes[0], layout.planes[1]];
            let wide = surface_layout(*b"x420", (width, height), luma_at, 2 * pitch);
            let len = (wide.planes[1].offset + pitch * rows).next_multiple_of(PAGE);
            for (profile, codec) in [
                (EncodeProfile::H264High, AnnexBCodec::H264),
                (EncodeProfile::HevcMain, AnnexBCodec::Hevc),
            ] {
                let at = format!("{codec:?} at {width}x{height}, {pitch}-byte rows");
                let params = params(profile, width, height);
                let mut encoder =
                    NvencEncoder::new(Arc::clone(&video.gpu), &params).expect("an NVENC session");
                let mut guest = NvGuestDecoder {
                    decoder: AnnexBDecoder::new(&video.gpu, codec, &runtime),
                    said: Vec::new(),
                };
                let mut reference = AnnexBDecoder::new(&video.gpu, codec, &runtime);
                reference
                    .parameter_sets(encoder.sequence_parameters())
                    .expect("the parameter sets");
                let memory = PagesMemory {
                    life: MemoryLife::new(),
                    pages: Pages::new(len),
                };
                let output = GuestOutput {
                    layout: layout.clone(),
                    memory: &memory,
                };
                let mut pictures = 0;
                for n in 0..7 {
                    let (y, uv) = frame(columns, rows, n);
                    let packets = (encoder.encode(&RawFrame {
                        luma: (&y, columns),
                        chroma: (&uv, columns),
                        index: n,
                        force_idr: n == 0,
                    }))
                    .expect("a frame NVENC takes");
                    for packet in packets {
                        let packet = packet.expect("a packet");
                        let sets: &[u8] = if pictures == 0 {
                            encoder.sequence_parameters()
                        } else {
                            &[]
                        };
                        if n == 6 {
                            let output = GuestOutput {
                                layout: wide.clone(),
                                memory: &memory,
                            };
                            let refused = guest.decode(sets, &packet.data, &output);
                            let said = match &refused {
                                Err(VideoError::Unsupported(why)) => why.as_str(),
                                _ => "",
                            };
                            assert!(
                                said.contains("1-byte samples for a 'x420' surface"),
                                "{at}: an 8-bit picture in an x420 surface: {refused:?}"
                            );
                            continue;
                        }
                        memory.pages.fill(UNTOUCHED);
                        let landed = guest.decode(sets, &packet.data, &output);
                        assert_eq!(landed, Ok(true), "{at}: picture {n}");
                        let picture = (reference.decode(&packet.data))
                            .expect("a decode")
                            .expect("a picture");
                        let read = picture.read_back((width, height)).expect("a read back");
                        let bytes = memory.pages.bytes();
                        assert!(
                            bytes[..luma_at].iter().all(|&b| b == UNTOUCHED),
                            "{at}: picture {n}: the bytes before the planes"
                        );
                        for (read, plane) in read.planes().iter().zip([luma, chroma]) {
                            assert_eq!(read.bytes_per_row, columns, "{at}");
                            for row in 0..usize::try_from(read.rows).expect("rows") {
                                let first = plane.offset + row * pitch;
                                assert_eq!(
                                    &bytes[first..][..columns],
                                    &read.bytes[row * columns..][..columns],
                                    "{at}: picture {n}, row {row} of the plane at {}",
                                    plane.offset
                                );
                                assert!(
                                    bytes[first + columns..first + pitch]
                                        .iter()
                                        .all(|&b| b == UNTOUCHED),
                                    "{at}: picture {n}: bytes past row {row} of the plane at {}",
                                    plane.offset
                                );
                            }
                        }
                        let chroma_end = chroma.offset + pitch * rows / 2;
                        assert!(
                            bytes[chroma_end..].iter().all(|&b| b == UNTOUCHED),
                            "{at}: picture {n}: the bytes past the chroma"
                        );
                        pictures += 1;
                    }
                }
                assert_eq!(pictures, 6, "{at}");
                assert_eq!(
                    guest.said,
                    Vec::<String>::new(),
                    "{at}: pictures copied by the CPU"
                );
            }
        }
    });
}

/// The CPU's copy of a picture read back lands each plane's rows at the surface's offset and
/// pitch, as many of each row's bytes as the plane's rows hold, and as many rows as both have,
/// leaving the bytes past each row as they were; a plane whose rows lie past the surface is
/// refused, with nothing written.
#[test]
fn a_read_back_picture_is_copied_into_the_surfaces_planes_at_their_pitch() {
    let (luma, chroma): (Vec<u8>, Vec<u8>) = ((0..6 * 4).collect(), (100..100 + 6 * 2).collect());
    let read = [
        ReadPlane {
            bytes: &luma,
            bytes_per_row: 6,
            rows: 4,
        },
        ReadPlane {
            bytes: &chroma,
            bytes_per_row: 6,
            rows: 2,
        },
    ];
    let planes = [
        TargetPlane {
            offset: 2,
            bytes_per_row: 8,
            rows: 4,
            row_bytes: 6,
        },
        TargetPlane {
            offset: 40,
            bytes_per_row: 8,
            rows: 3,
            row_bytes: 4,
        },
    ];
    let mut surface = vec![UNTOUCHED; 64];
    copy_planes(&read, &planes, &mut surface).expect("a copy");
    let mut expected = vec![UNTOUCHED; 64];
    for row in 0..4 {
        expected[2 + row * 8..][..6].copy_from_slice(&luma[row * 6..][..6]);
    }
    for row in 0..2 {
        expected[40 + row * 8..][..4].copy_from_slice(&chroma[row * 6..][..4]);
    }
    assert_eq!(surface, expected);
    let mut short = vec![UNTOUCHED; 49];
    let refused = copy_planes(&read, &planes, &mut short);
    assert_eq!(
        refused,
        Err("a plane at 40 of 2 rows 8 bytes apart past the surface's 49 bytes".to_owned())
    );
    assert!(
        short.iter().all(|&b| b == UNTOUCHED),
        "written when refused"
    );
}

/// A 4:4:4 picture read back, luma, Cb and Cr, lands its luma as it lies and its Cb and Cr
/// interleaved into the surface's chroma plane, Cb first, a pair a luma sample, at the plane's
/// pitch and as far as its rows hold them; Cb and Cr whose rows lie past the surface are
/// refused, with nothing written, and so are two planes.
#[test]
fn a_4_4_4_pictures_cb_and_cr_are_interleaved_into_the_surfaces_chroma_plane() {
    let luma: Vec<u8> = (0..4 * 3).collect();
    let (cb, cr): (Vec<u8>, Vec<u8>) = ((100..100 + 4 * 3).collect(), (200..200 + 4 * 3).collect());
    let plane = |bytes| ReadPlane {
        bytes,
        bytes_per_row: 4,
        rows: 3,
    };
    let read = [plane(&luma), plane(&cb), plane(&cr)];
    let planes = [
        TargetPlane {
            offset: 0,
            bytes_per_row: 8,
            rows: 3,
            row_bytes: 4,
        },
        TargetPlane {
            offset: 32,
            bytes_per_row: 10,
            rows: 3,
            row_bytes: 6,
        },
    ];
    let mut surface = vec![UNTOUCHED; 64];
    interleave_planes(&read, &planes, &mut surface, 1).expect("a copy");
    let mut expected = vec![UNTOUCHED; 64];
    for row in 0..3 {
        expected[row * 8..][..4].copy_from_slice(&luma[row * 4..][..4]);
        for x in 0..3 {
            expected[32 + row * 10 + 2 * x] = cb[row * 4 + x];
            expected[32 + row * 10 + 2 * x + 1] = cr[row * 4 + x];
        }
    }
    assert_eq!(surface, expected);
    let mut short = vec![UNTOUCHED; 57];
    let refused = interleave_planes(&read, &planes, &mut short, 1);
    assert_eq!(
        refused,
        Err("a plane at 32 of 3 rows 10 bytes apart past the surface's 57 bytes".to_owned())
    );
    assert!(
        short.iter().all(|&b| b == UNTOUCHED),
        "written when refused"
    );
    assert_eq!(
        interleave_planes(&read[..2], &planes, &mut surface, 1),
        Err("2 planes read back for 3".to_owned())
    );
}

/// A VideoToolbox decoder sees a stream that does not decode, or a picture NVDEC failed, as bad
/// data; what NVDEC does not decode as unsupported; a skipped leading picture as skipped; and the
/// rest, a picture too small among them, as its malfunction.
#[test]
fn nvidia_errors_reach_videotoolbox_as_their_kind() {
    use reims_vgpu_video::nvidia::Error;
    let said = |what: &str| what.to_owned();
    assert_eq!(
        video_error(Error::Bitstream(said("a short slice"))),
        VideoError::BadData("bad bitstream: a short slice".to_owned())
    );
    assert_eq!(
        video_error(Error::Picture(said("a reference frame is missing"))),
        VideoError::BadData("bad bitstream: a reference frame is missing".to_owned())
    );
    assert_eq!(
        video_error(Error::DecodeFailed(said("slice data"))),
        VideoError::BadData("slice data".to_owned())
    );
    assert_eq!(
        video_error(Error::Unsupported(said("4:4:4"))),
        VideoError::Unsupported("4:4:4".to_owned())
    );
    assert_eq!(
        video_error(Error::SkippedLeadingPicture(said("RASL"))),
        VideoError::SkippedLeadingPicture("RASL".to_owned())
    );
    assert_eq!(
        video_error(Error::TooSmall(said("16x16"))),
        VideoError::Malfunction("too small: 16x16".to_owned())
    );
    assert_eq!(
        video_error(Error::OutOfSurfaces),
        VideoError::Malfunction("every decode surface is held".to_owned())
    );
}
