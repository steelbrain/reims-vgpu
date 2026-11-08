//! The NVIDIA arm on the NVIDIA host: a picture written by a copy within the GPU into a device
//! cache CUDA imported reads, through the runtime, as NVDEC's own read back of it.

use std::sync::Arc;

use reims_vgpu_video::core::device_memory::MemoryLife;
use reims_vgpu_video::nvidia::{
    EncodeParams, EncodeProfile, EncodeRateControl, NvencEncoder, RawFrame, Tuning,
};
use reims_vgpu_vulkan::core::platform::Platform;
use reims_vgpu_vulkan::core::platform::blit::{BlitCmd, BlitEncoder as _};
use reims_vgpu_vulkan::core::platform::command::{CommandBuffer as _, CommandQueue as _};
use reims_vgpu_vulkan::core::platform::types::CommandBufferStatus;

use super::*;

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

/// Copies `length` bytes of `from` into `to` on `queue`, and waits for the copy.
fn copy(
    queue: &<VulkanPlatform as Platform>::CommandQueue,
    from: &VulkanBuffer,
    to: &VulkanBuffer,
    length: u64,
) {
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

/// Each picture of NVENC's H.264 and HEVC at 1080p lands in an exportable buffer of the runtime
/// device's, imported once, by NVDEC's copy within the GPU, its rows alone at a pitch of their
/// own: a copy of the buffer on the runtime's queue reads NVDEC's read back of the picture in
/// each row, and the bytes past each row as the runtime's fill left them.
#[test]
#[ignore = "needs an NVIDIA GPU: run on the NVIDIA host"]
fn a_picture_lands_in_an_imported_device_cache_as_it_reads_back() {
    let platform = Arc::new(VulkanPlatform::new());
    crate::entry::run(|_| {
        let device = VulkanPlatform::system_default_device().expect("a Vulkan device");
        let runtime = crate::video::runtime::PlatformRuntime::of(&platform);
        let video = NvidiaVideo::new(&device, &runtime).expect("NVDEC on the device's GPU");
        let imports = (video.imports.as_ref()).expect("CUDA's imports of exportable buffers");
        let (width, height, pitch) = (1920u32, 1080u32, 2048usize);
        let rows = usize::try_from(height).expect("rows");
        let chroma_at = pitch * rows;
        let len = u64::try_from(chroma_at + pitch * rows / 2).expect("a length");
        let cache = (device.new_exportable_buffer_of_length(OnFault::Return, len, STORAGE_PRIVATE))
            .expect("no raise")
            .expect("an exportable buffer");
        let life = MemoryLife::new();
        let imported = (imports.get_or_make(
            &life.identity(),
            0,
            || import(&cache, &video.gpu),
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
            let params = EncodeParams {
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
            };
            let mut encoder =
                NvencEncoder::new(Arc::clone(&video.gpu), &params).expect("an NVENC session");
            let mut decoder = AnnexBDecoder::new(&video.gpu, codec, &runtime);
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
                    copy(&queue, &fill, &cache, len);
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
                            plane(0, height),
                            plane(chroma_at, height / 2),
                            Box::new(()),
                        )
                    };
                    written.expect("the decode").expect("the GPU's copy");
                    copy(&queue, &cache, &read_back, len);
                    let read = picture.read_back((width, height)).expect("a read back");
                    // SAFETY: the Shared buffer's `len` bytes, which the copy above finished
                    // writing.
                    let bytes = unsafe {
                        std::slice::from_raw_parts(read_back.contents().as_ptr::<u8>(), length)
                    };
                    for (read, at) in read.planes().iter().zip([0, chroma_at]) {
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
    });
}
