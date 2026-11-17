//! The tests' landing of a picture both ways, on every host the arm runs on: by a session's GPU
//! copier and by the CPU's copy, into the same layout, compared byte for byte
//! ([`land_both_ways`]).

use reims_vgpu_video::core::backend::DeviceWrite;
use reims_vgpu_video::vaapi::{Frame, PlaneRows};
use reims_vgpu_vulkan::VulkanPlatform;
use reims_vgpu_vulkan::core::platform::Handle as _;
use reims_vgpu_vulkan::gpu::device::VulkanDevice;

use super::super::video_host::OutputPlane;
use super::GpuCopier;

/// Lands `frame`, the `n`th picture of the run, by `copier` into planes laid out as CoreVideo
/// lays a frame's, in host pages a buffer of `device` lies over, or, every third picture, in a
/// buffer made as a device cache is ([`land_in_a_cache`]), and by the CPU's copy into the same
/// layout in memory of its own; panics where a byte of the two differs.
pub(super) fn land_both_ways(
    copier: &mut dyn GpuCopier<VulkanPlatform>,
    device: &VulkanDevice,
    frame: &Frame,
    n: usize,
    stream: &std::path::Path,
) {
    use reims_vgpu_vulkan::core::platform::HostRange;
    use reims_vgpu_vulkan::core::platform::device::GpuDevice as _;
    use reims_vgpu_vulkan::core::platform::os::vm::HostVm as _;
    use reims_vgpu_vulkan::core::raise::OnFault;
    use reims_vgpu_vulkan::remap::VulkanVm;

    let vm = VulkanVm::new();
    let sample: u64 = if frame.bit_depth() > 8 { 2 } else { 1 };
    let (width, height) = (u64::from(frame.width()), u64::from(frame.height()));
    let luma_bytes = width * sample;
    let chroma_bytes = width.div_ceil(2) * 2 * sample;
    let pitch = if n % 2 == 0 {
        luma_bytes.max(chroma_bytes)
    } else {
        luma_bytes.max(chroma_bytes).next_multiple_of(64) + 64
    };
    let chroma_rows = height.div_ceil(2);
    let (luma_at, chroma_at) = (0x20, 0x20 + pitch * height);
    let len = (chroma_at + pitch * chroma_rows).next_multiple_of(vm.page_size());
    let cached = n % 3 == 2;
    if cached {
        land_in_a_cache(
            copier,
            device,
            frame,
            n,
            stream,
            (luma_at, chroma_at, pitch, len),
        );
        return;
    }
    let at = vm.allocate_anywhere(len).expect("host pages");
    // SAFETY: the pages stay mapped until the runtime lets go of them, when its deallocator
    // unmaps them, and nothing else reads or writes them meanwhile but the test's reads below.
    let bytes = unsafe { HostRange::new(at, len) };
    // The runtime runs the deallocator at the buffer's release, Metal's moment, and frees its
    // import of the pages after it, on a handler thread where a command buffer held the release.
    // The platform's VM leaves pages an import holds mapped until that free
    // (`reims_vgpu_vulkan::remap`, "Unmaps an import waits for"); the OS's unmap would take them
    // from under the import, and RADV, which hands every pointer import to each submission,
    // would reject every submission of the device until the free (`-EFAULT`).
    let unmap: Box<dyn FnOnce() + Send> = Box::new(move || {
        let _ = VulkanVm::new().deallocate(at, len);
    });
    let buffer = (device.new_buffer_over_host_memory(OnFault::Return, bytes, 0, Some(unmap)))
        .expect("no raise")
        .expect("a buffer over the pages");
    // SAFETY: the pages' `len` bytes, which no GPU work reaches yet.
    unsafe { std::ptr::write_bytes(at.as_ptr::<u8>(), UNTOUCHED, usize_of(len)) };
    let plane = |offset: u64, row_bytes: u64, w: u64, h: u64| OutputPlane {
        buffer: buffer.clone_ref(),
        offset,
        bytes_per_row: pitch,
        row_bytes,
        width: w,
        height: h,
        rows_held: h,
        cached: false,
        host: Some(at.wrapping_add(offset)),
        pages: None,
        cache: None,
    };
    let luma = plane(luma_at, luma_bytes, width, height);
    let chroma = plane(chroma_at, chroma_bytes, width.div_ceil(2), chroma_rows);
    match copier.copy(frame, &luma, &chroma).expect("the GPU's copy") {
        DeviceWrite::Written => {}
        DeviceWrite::Declined(why) => panic!("{}: declined: {why}", stream.display()),
    }
    let mut cpu = vec![UNTOUCHED; usize_of(len)];
    let (y, uv) = cpu.split_at_mut(usize_of(chroma_at));
    (frame.read_back().expect("the read back"))
        .copy_into(
            plane_rows(&mut y[usize_of(luma_at)..], pitch, height, luma_bytes),
            plane_rows(uv, pitch, chroma_rows, chroma_bytes),
        )
        .expect("the CPU's copy");
    // SAFETY: the pages' `len` bytes, which the GPU's copy finished writing before it returned,
    // and which nothing writes meanwhile.
    let gpu = unsafe { std::slice::from_raw_parts(at.as_ptr::<u8>(), usize_of(len)) };
    if let Some(first) = (0..gpu.len()).find(|&i| gpu[i] != cpu[i]) {
        panic!(
            "{} picture {n}, {width}x{height} at {sample}-byte samples, rows of {pitch}: byte \
             {first} is {:#x} by the GPU's copy and {:#x} by the CPU's (luma at {luma_at}, chroma \
             at {chroma_at})",
            stream.display(),
            gpu[first],
            cpu[first]
        );
    }
    drop((luma, chroma, buffer));
}

/// [`land_both_ways`]'s landing of `frame` in a buffer of `device`'s own memory, made as a device
/// cache is: Private, the bytes between its rows declared unobserved and its first plane's byte
/// on 256 bytes of its memory (`surface_cache::aligned_byte`), its planes laid out by `(luma_at, chroma_at, pitch)` in its
/// `len` bytes, read back by a blit into a Shared buffer. As a device cache is filled from its
/// surface's pages as it is made, before a picture lands in it, the runtime fills the buffer with
/// [`UNTOUCHED`] first; the CPU's copy starts from those bytes.
fn land_in_a_cache(
    copier: &mut dyn GpuCopier<VulkanPlatform>,
    device: &VulkanDevice,
    frame: &Frame,
    n: usize,
    stream: &std::path::Path,
    (luma_at, chroma_at, pitch, len): (u64, u64, u64, u64),
) {
    use reims_vgpu_vulkan::core::platform::blit::{BlitCmd, BlitEncoder as _};
    use reims_vgpu_vulkan::core::platform::command::{CommandBuffer as _, CommandQueue as _};
    use reims_vgpu_vulkan::core::platform::device::GpuDevice as _;
    use reims_vgpu_vulkan::core::platform::resource::Buffer as _;
    use reims_vgpu_vulkan::core::platform::types::{CommandBufferStatus, NsRange};
    use reims_vgpu_vulkan::core::raise::OnFault;

    let sample: u64 = if frame.bit_depth() > 8 { 2 } else { 1 };
    let (width, height) = (u64::from(frame.width()), u64::from(frame.height()));
    let (luma_bytes, chroma_bytes) = (width * sample, width.div_ceil(2) * 2 * sample);
    let chroma_rows = height.div_ceil(2);
    // `MTLResourceStorageModePrivate`, a device cache's storage.
    let private = 2 << 4;
    let laid_out = device.new_buffer_of_length_padding_unobserved(
        OnFault::Return,
        len,
        private,
        Some(luma_at),
    );
    let cache = laid_out.expect("no raise").expect("a cache");
    let queue = device.new_command_queue().expect("a queue");
    let blit_and_wait = |cmd: BlitCmd<'_, VulkanPlatform>, what: &str| {
        let cb = (queue.command_buffer(OnFault::Return))
            .expect("no raise")
            .expect("a command buffer");
        let blit = (cb.blit_encoder(OnFault::Return))
            .expect("no raise")
            .expect("a blit encoder");
        blit.encode(OnFault::Return, cmd).expect(what);
        blit.encode(OnFault::Return, BlitCmd::EndEncoding)
            .expect("its end");
        cb.commit(OnFault::Return).expect("the commit");
        cb.wait_until_completed(OnFault::Return).expect("the wait");
        assert_eq!(cb.status(), CommandBufferStatus::COMPLETED, "{what}");
    };
    let range = NsRange {
        location: 0,
        length: len,
    };
    blit_and_wait(
        BlitCmd::FillBuffer {
            buffer: &cache,
            range,
            value: UNTOUCHED,
        },
        "the cache's fill",
    );
    let plane = |offset: u64, row_bytes: u64, w: u64, h: u64| OutputPlane {
        buffer: cache.clone_ref(),
        offset,
        bytes_per_row: pitch,
        row_bytes,
        width: w,
        height: h,
        rows_held: h,
        cached: true,
        host: None,
        pages: None,
        cache: None,
    };
    let luma = plane(luma_at, luma_bytes, width, height);
    let chroma = plane(chroma_at, chroma_bytes, width.div_ceil(2), chroma_rows);
    match copier.copy(frame, &luma, &chroma).expect("the GPU's copy") {
        DeviceWrite::Written => {}
        DeviceWrite::Declined(why) => panic!("{}: declined into a cache: {why}", stream.display()),
    }
    let back = (device.new_buffer_of_length(OnFault::Return, len, 0))
        .expect("no raise")
        .expect("a buffer to read the cache back into");
    blit_and_wait(
        BlitCmd::CopyBufferToBuffer {
            source_buffer: Some(&cache),
            source_offset: 0,
            destination_buffer: Some(&back),
            destination_offset: 0,
            size: len,
        },
        "the copy back",
    );
    let mut cpu = vec![UNTOUCHED; usize_of(len)];
    let (y, uv) = cpu.split_at_mut(usize_of(chroma_at));
    (frame.read_back().expect("the read back"))
        .copy_into(
            plane_rows(&mut y[usize_of(luma_at)..], pitch, height, luma_bytes),
            plane_rows(uv, pitch, chroma_rows, chroma_bytes),
        )
        .expect("the CPU's copy");
    // SAFETY: the read-back buffer's `len` bytes, which the blit finished writing before its
    // command buffer completed, and which nothing writes meanwhile.
    let gpu = unsafe { std::slice::from_raw_parts(back.contents().as_ptr::<u8>(), usize_of(len)) };
    if let Some(first) = (0..gpu.len()).find(|&i| gpu[i] != cpu[i]) {
        panic!(
            "{} picture {n} into a cache, {width}x{height} at {sample}-byte samples, rows of \
             {pitch}: byte {first} is {:#x} by the GPU's copy and {:#x} by the CPU's (luma at \
             {luma_at}, chroma at {chroma_at})",
            stream.display(),
            gpu[first],
            cpu[first]
        );
    }
    drop((luma, chroma, cache, back));
}

/// What the GPU's copies leave as it was: every byte past the planes' rows.
pub(super) const UNTOUCHED: u8 = 0xee;

pub(super) fn usize_of(n: u64) -> usize {
    usize::try_from(n).expect("a size")
}

/// `rows` rows of `row_bytes` in `bytes`, `pitch` apart, for the CPU's copy.
pub(super) fn plane_rows(bytes: &mut [u8], pitch: u64, rows: u64, row_bytes: u64) -> PlaneRows<'_> {
    PlaneRows {
        bytes,
        bytes_per_row: usize_of(pitch),
        rows: usize_of(rows),
        row_bytes: usize_of(row_bytes),
    }
}
