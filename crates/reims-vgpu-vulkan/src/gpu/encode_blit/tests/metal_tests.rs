//! The tests of `encode_blit` that compare with Metal on this host: macOS only.

use crate::core::platform::types::{ClearColor, MtlOrigin, MtlSize, NsRange};

use super::*;

/// The two-plane XR10 formats: `MTLPixelFormatRGB10A8_2P_XR10` and its sRGB twin.
const TWO_PLANE_XR10: [u64; 2] = [550, 551];

/// A blit from a buffer into a texture, without and with blit options: where a device that clamps
/// a `Depth32Float` depth into `[0, 1]` dies of `restricted-depth-upload` in strict mode
/// (`gpu::device`'s `uploading_depth`).
const UPLOAD: &str = "copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:\
                      toTexture:destinationSlice:destinationLevel:destinationOrigin:";
const UPLOAD_OPTIONS: &str = "copyFromBuffer:sourceOffset:sourceBytesPerRow:\
                              sourceBytesPerImage:sourceSize:toTexture:destinationSlice:\
                              destinationLevel:destinationOrigin:options:";

/// The 4 by 4 BGRA8Unorm texture a copy of 64 bytes from a no-copy buffer of 63 over a page
/// of `0..=255` fills on platform `P`, read back.
fn copied_past_a_no_copy_buffer_on<P: crate::core::platform::Platform>(
    d: &P::Device,
    desc: &dyn Fn(&TextureDesc) -> P::TextureDescriptor,
) -> Vec<u8> {
    use std::alloc::{Layout, alloc, dealloc};

    use crate::core::host::HostPtr;
    use crate::core::platform::HostRange;
    use crate::core::platform::resource::Buffer as _;
    let layout = Layout::from_size_align(16384, 16384).unwrap();
    // SAFETY: a nonzero size.
    let page = unsafe { alloc(layout) };
    for i in 0..16384 {
        // SAFETY: inside the page just allocated.
        unsafe { *page.add(i) = i as u8 };
    }
    // SAFETY: the page outlives the buffer, which is dropped before it is freed.
    let bytes = unsafe { HostRange::new(HostPtr::from_ptr(page), 63) };
    let short = d
        .new_buffer_over_host_memory(OnFault::Return, bytes, 0, None)
        .unwrap()
        .unwrap();
    let t = d
        .new_texture(
            OnFault::Return,
            &desc(&TextureDesc {
                pixel_format: 80,
                width: 4,
                height: 4,
                usage: 0x13,
                ..TextureDesc::default()
            }),
        )
        .unwrap()
        .unwrap();
    let down = d
        .new_buffer_of_length(OnFault::Return, 64, 0)
        .unwrap()
        .unwrap();
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
    let e = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
    let cmds = [
        BlitCmd::CopyBufferToTexture {
            source_buffer: &short,
            source_offset: 0,
            source_bytes_per_row: 16,
            source_bytes_per_image: 64,
            source_size: MtlSize::new(4, 4, 1),
            destination_texture: &t,
            destination_slice: 0,
            destination_level: 0,
            destination_origin: MtlOrigin::new(0, 0, 0),
        },
        BlitCmd::CopyTextureToBuffer {
            source_texture: &t,
            source_slice: 0,
            source_level: 0,
            source_origin: MtlOrigin::new(0, 0, 0),
            source_size: MtlSize::new(4, 4, 1),
            destination_buffer: &down,
            destination_offset: 0,
            destination_bytes_per_row: 16,
            destination_bytes_per_image: 64,
        },
        BlitCmd::EndEncoding,
    ];
    for cmd in cmds {
        e.encode(OnFault::Return, cmd).unwrap();
    }
    cb.commit(OnFault::Return).unwrap();
    cb.wait_until_completed(OnFault::Return).unwrap();
    // SAFETY: a Shared buffer's 64 bytes, read after the GPU's writes completed.
    let read = unsafe { std::slice::from_raw_parts(down.contents().as_ptr::<u8>(), 64) };
    let read = read.to_vec();
    drop((cb, short));
    // SAFETY: allocated above with `layout`; the buffer over it is gone.
    unsafe { dealloc(page, layout) };
    read
}

/// The 16 by 16 `PVRTC_RGBA_4BPP` texture of `ShaderRead` usage the PVRTC copies are made
/// with: at the descriptor's storage and `allowGPUOptimizedContents` (on), or, `suites`, at
/// `Private` with `allowGPUOptimizedContents` off, the shape of the texture `textures_formats`
/// makes. `newTextureWithDescriptor:` of either is made at Metal's size in Metal's space
/// ([`pvrtc_texture_takes_metals_size`]); `newSharedTextureWithDescriptor:` of either is made,
/// its range sized by Apple's layout, as `textures_gap_pvrtc_shared_backing` makes one.
fn pvrtc_copy_texture(suites: bool) -> TextureDesc {
    let t = TextureDesc {
        pixel_format: 166,
        width: 16,
        height: 16,
        usage: 0x1,
        ..TextureDesc::default()
    };
    match suites {
        true => TextureDesc {
            resource_options: Some(0x20),
            allow_gpu_optimized_contents: Some(false),
            ..t
        },
        false => t,
    }
}

/// The bytes of [`pvrtc_copy_texture`]'s texture of the suite's shape on platform `P`, a
/// shareable one (`newSharedTextureWithDescriptor:`), filled from 128 bytes of `0..128` row by
/// row (`MTLBlitOptionRowLinearPVRTC`), read back row by row with rows 32 and 48 bytes apart.
fn pvrtc_copies_on<P: crate::core::platform::Platform>(
    d: &P::Device,
    desc: &dyn Fn(&TextureDesc) -> P::TextureDescriptor,
) -> [Vec<u8>; 2] {
    let t = d
        .new_shared_texture(&desc(&pvrtc_copy_texture(true)))
        .unwrap()
        .unwrap();
    let up = d
        .new_buffer_of_length(OnFault::Return, 128, 0)
        .unwrap()
        .unwrap();
    for i in 0..128u8 {
        // SAFETY: inside a Shared buffer of 128 bytes the GPU does not use yet.
        unsafe { *up.contents().as_ptr::<u8>().add(usize::from(i)) = i };
    }
    let (rows, padded) = (
        d.new_buffer_of_length(OnFault::Return, 128, 0)
            .unwrap()
            .unwrap(),
        d.new_buffer_of_length(OnFault::Return, 192, 0)
            .unwrap()
            .unwrap(),
    );
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
    let e = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
    let down = |buffer, bytes_per_row, options| BlitCmd::CopyTextureToBufferOptions {
        source_texture: Some(&t),
        source_slice: 0,
        source_level: 0,
        source_origin: MtlOrigin::new(0, 0, 0),
        source_size: MtlSize::new(16, 16, 1),
        destination_buffer: Some(buffer),
        destination_offset: 0,
        destination_bytes_per_row: bytes_per_row,
        destination_bytes_per_image: 0,
        options,
    };
    let cmds = [
        BlitCmd::CopyBufferToTextureOptions {
            source_buffer: Some(&up),
            source_offset: 0,
            source_bytes_per_row: 32,
            source_bytes_per_image: 128,
            source_size: MtlSize::new(16, 16, 1),
            destination_texture: Some(&t),
            destination_slice: 0,
            destination_level: 0,
            destination_origin: MtlOrigin::new(0, 0, 0),
            options: 4,
        },
        down(&rows, 32, 4),
        down(&padded, 48, 4),
        BlitCmd::EndEncoding,
    ];
    for cmd in cmds {
        e.encode(OnFault::Return, cmd).unwrap();
    }
    cb.commit(OnFault::Return).unwrap();
    cb.wait_until_completed(OnFault::Return).unwrap();
    [(&rows, 128), (&padded, 192)].map(|(b, n)| {
        // SAFETY: a Shared buffer's `n` bytes, read after the GPU's writes completed.
        unsafe { std::slice::from_raw_parts(b.contents().as_ptr::<u8>(), n) }.to_vec()
    })
}

/// A PVRTC texture's copies with MTLBlitOptionRowLinearPVRTC move its blocks row by row, as
/// Metal's do on the M3 Max, at the suite's shape ([`pvrtc_copy_texture`]), of a
/// shareable texture
#[test]
fn pvrtc_row_linear_copies_are_metals() {
    let d = dev();
    let vulkan_desc = |t: &TextureDesc| crate::platform::tests::platform().texture_descriptor(t);
    let vulkan = pvrtc_copies_on::<VulkanPlatform>(&d, &vulkan_desc);
    assert_eq!(vulkan[0], (0..128).collect::<Vec<u8>>());
    crate::metal_check_raw!("copies", vulkan);
}

/// A texture of a PVRTC format is made by `newTextureWithDescriptor:`, as Metal makes one on
/// this host, and its size and alignment in Metal's heaps, which place what the guest makes next
/// in Metal's space, are Metal's: the runtime models PVRTC's layout in Metal's heaps at macOS
/// 27.0.1, this host's level (its `unmeasured-heap-texture-size` names it at 26.7 alone).
fn pvrtc_texture_takes_metals_size(t: &TextureDesc) {
    let what = format!("format {}, {}x{}", t.pixel_format, t.width, t.height);
    let desc = crate::platform::tests::platform().texture_descriptor(t);
    let d = dev();
    assert!(
        d.new_texture(OnFault::Return, &desc).unwrap().is_some(),
        "{what}"
    );
    let ours = d.heap_texture_size_and_align(&desc).unwrap();
    crate::metal_check!(format!("{t:?}"), (ours.size, ours.align));
}

/// The PVRTC textures the row-linear copies are measured with ([`pvrtc_copy_texture`]), at the
/// descriptor's storage and `allowGPUOptimizedContents` and at the suite's shape, are made at
/// Metal's size, as [`pvrtc_texture_takes_metals_size`] says
#[test]
fn pvrtc_copy_textures_take_metals_size() {
    pvrtc_texture_takes_metals_size(&pvrtc_copy_texture(false));
    pvrtc_texture_takes_metals_size(&pvrtc_copy_texture(true));
}

/// `n` bytes of a fixed pseudo-random sequence that `seed` picks, every bit of a word set in
/// some of them.
fn noise(seed: u32, n: u64) -> Vec<u8> {
    let mut x = seed;
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            x.to_le_bytes()[3]
        })
        .collect()
}

/// The width and height of the two-plane XR10 textures the copies below make.
const TWO_PLANE_SIZE: (u64, u64) = (48, 20);

/// The bytes copies of a 48 by 20 texture of `format` (550 or 551) give on platform `P`: its
/// texels written by `replaceRegion:` from `noise(1, ..)`, 4 bytes a texel, then copied into a
/// buffer with rows 4 bytes a texel apart and into one with rows 5 bytes a texel apart (its
/// other bytes 0xaa), then, after a copy of a 17 by 9 region to (13, 11) from a buffer at
/// offset 256 with rows 8 bytes a texel apart, the whole texture again.
fn two_plane_copies_on<P: crate::core::platform::Platform>(
    d: &P::Device,
    desc: &dyn Fn(&TextureDesc) -> P::TextureDescriptor,
    format: u64,
) -> [Vec<u8>; 3] {
    use crate::core::num::ToU64 as _;
    use crate::core::platform::HostRange;
    use crate::core::platform::resource::Texture as _;
    let (w, h) = TWO_PLANE_SIZE;
    let t = d
        .new_texture(
            OnFault::Return,
            &desc(&TextureDesc {
                pixel_format: format,
                width: w,
                height: h,
                usage: 0x3,
                ..TextureDesc::default()
            }),
        )
        .unwrap()
        .unwrap();
    let all = MtlRegion::new(MtlOrigin::new(0, 0, 0), MtlSize::new(w, h, 1));
    let texels = noise(1, w * h * 4);
    t.replace_region(all, 0, 0, HostRange::of_slice(&texels), w * 4, 0)
        .unwrap();
    let filled = |bytes: &[u8]| {
        let b = d
            .new_buffer_of_length(OnFault::Return, bytes.len().to_u64(), 0)
            .unwrap()
            .unwrap();
        // SAFETY: inside a Shared buffer of `bytes.len()` bytes the GPU does not use yet.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), b.contents().as_ptr(), bytes.len());
        }
        b
    };
    let (row, region_row) = (w * 4, 17 * 8);
    let tight = filled(&vec![0x55; usize::try_from(row * h).unwrap()]);
    let padded = filled(&vec![0xaa; usize::try_from(w * 5 * h).unwrap()]);
    let region = filled(&noise(2, 256 + region_row * 9));
    let after = filled(&vec![0x55; usize::try_from(row * h).unwrap()]);
    let down = |buffer, bytes_per_row| BlitCmd::CopyTextureToBuffer {
        source_texture: &t,
        source_slice: 0,
        source_level: 0,
        source_origin: MtlOrigin::new(0, 0, 0),
        source_size: MtlSize::new(w, h, 1),
        destination_buffer: buffer,
        destination_offset: 0,
        destination_bytes_per_row: bytes_per_row,
        destination_bytes_per_image: bytes_per_row * h,
    };
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
    let e = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
    let cmds = [
        down(&tight, row),
        down(&padded, w * 5),
        BlitCmd::CopyBufferToTexture {
            source_buffer: &region,
            source_offset: 256,
            source_bytes_per_row: region_row,
            source_bytes_per_image: region_row * 9,
            source_size: MtlSize::new(17, 9, 1),
            destination_texture: &t,
            destination_slice: 0,
            destination_level: 0,
            destination_origin: MtlOrigin::new(13, 11, 0),
        },
        down(&after, row),
        BlitCmd::EndEncoding,
    ];
    for cmd in cmds {
        e.encode(OnFault::Return, cmd).unwrap();
    }
    cb.commit(OnFault::Return).unwrap();
    cb.wait_until_completed(OnFault::Return).unwrap();
    [&tight, &padded, &after].map(|b| {
        let n = usize::try_from(b.length()).unwrap();
        // SAFETY: a Shared buffer's `n` bytes, read after the GPU's writes completed.
        unsafe { std::slice::from_raw_parts(b.contents().as_ptr::<u8>(), n) }.to_vec()
    })
}

/// A texture of a two-plane XR10 format, 550 or 551, copies to and from a buffer as the Metal
/// platform's does on this host: `replaceRegion:` and the blits move each texel's colour word,
/// 4 bytes, at the caller's rows, and leave the bytes between the rows as they were
#[test]
fn two_plane_xr10_copies_are_metals() {
    let d = dev();
    let vulkan_desc = |t: &TextureDesc| crate::platform::tests::platform().texture_descriptor(t);
    let (w, h) = TWO_PLANE_SIZE;
    for format in TWO_PLANE_XR10 {
        let vulkan = two_plane_copies_on::<VulkanPlatform>(&d, &vulkan_desc, format);
        assert_eq!(vulkan[0], noise(1, w * h * 4), "{format}");
        let key = format!("{format}");
        crate::metal_check_raw!(key, vulkan);
    }
}

/// The size of the two-plane XR10 textures [`two_plane_blit_on`] copies between.
const BLIT_SIZE: (u64, u64) = (16, 8);

/// A region a blit copies: its origin, its size and the destination's origin.
type BlitRegion = ((u64, u64), (u64, u64), (u64, u64));

/// A blit between two two-plane XR10 textures of one level and slice: of a region
/// (`CopyTextureRegion`), or of the whole level 0 and slice 0 (`CopyTextureSlices`, one
/// level and one slice).
#[derive(Debug, Clone, Copy)]
enum TwoPlaneCopy {
    /// `copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:...`.
    Region(BlitRegion),
    /// `copyFromTexture:sourceSlice:sourceLevel:toTexture:...sliceCount:levelCount:`.
    Slices,
}

/// `read()` into halves of each texel of a 16 by 8 texture of `to` (550 or 551) after one
/// blit of a 16 by 8 texture of `from` into it, on `P`'s `d`: of the whole texture
/// (`region` `None`), or of a region (origin, size) to a destination origin
/// ([`two_plane_copy_on`]).
fn two_plane_blit_on<P: crate::core::platform::Platform>(
    d: &P::Device,
    desc: &dyn Fn(&TextureDesc) -> P::TextureDescriptor,
    formats: (u64, u64),
    region: Option<BlitRegion>,
) -> Vec<u8> {
    let copy = TwoPlaneCopy::Region(region.unwrap_or(((0, 0), BLIT_SIZE, (0, 0))));
    let shape = ((BLIT_SIZE, BLIT_SIZE), copy, TwoPlaneStorage::Default);
    two_plane_copy_on::<P>(d, desc, formats, shape)
}

/// Where [`two_plane_copy_on`]'s two-plane textures lie.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TwoPlaneStorage {
    /// Each made by the device, of the descriptor's default storage.
    Default,
    /// Each made by the device, `Private`.
    Private,
    /// Both placed in one `Private` placement heap, one after the other.
    Heap,
}

/// The shape of a blit [`two_plane_copy_on`] makes: the textures' sizes (source and
/// destination), the copy, and where the textures lie.
type TwoPlaneShape = (((u64, u64), (u64, u64)), TwoPlaneCopy, TwoPlaneStorage);

/// `read()` into halves of each texel of a texture of `to` (550 or 551) of `sizes.1` after
/// `copy` from a texture of `from` of `sizes.0` into it, the two lying where `storage` says,
/// on `P`'s `d`. Each texture's texels are written first by a kernel from an `RGBA16Float`
/// texture of halves in [-0.5, 1.25), apart in every channel and in each texture, so the
/// blit moves both planes.
fn two_plane_copy_on<P: crate::core::platform::Platform>(
    d: &P::Device,
    desc: &dyn Fn(&TextureDesc) -> P::TextureDescriptor,
    (from, to): (u64, u64),
    (sizes, copy, storage): TwoPlaneShape,
) -> Vec<u8> {
    use crate::core::platform::HostRange;
    use crate::core::platform::compute::{ComputeCmd, ComputeEncoder as _};
    use crate::core::platform::resource::{Heap as _, Library as _, Texture as _};
    use crate::core::platform::types::MtlRegion;
    use crate::core::platform::values::{HeapDesc, OptionSet};
    use crate::gpu::ycbcr::measure::{PROBE, probed_texels};
    let texture_desc = |format, (w, h): (u64, u64), options| {
        desc(&TextureDesc {
            pixel_format: format,
            width: w,
            height: h,
            usage: 0x3,
            resource_options: options,
            ..TextureDesc::default()
        })
    };
    let texture = |format, size| {
        let made = d.new_texture(OnFault::Return, &texture_desc(format, size, None));
        made.unwrap().unwrap()
    };
    let heap = (storage == TwoPlaneStorage::Heap).then(|| {
        let made = d.new_heap(&HeapDesc {
            size: 1 << 20,
            heap_type: 1,
            options: vec![OptionSet::StorageMode(2)],
            address_ranges: None,
            pinned_gpu_address: None,
        });
        made.unwrap().heap.unwrap()
    });
    let two_plane = |format, size, at: u64| match (storage, &heap) {
        (TwoPlaneStorage::Default, _) => texture(format, size),
        (TwoPlaneStorage::Private, _) => {
            let made = d.new_texture(OnFault::Return, &texture_desc(format, size, Some(0x20)));
            made.unwrap().unwrap()
        }
        (TwoPlaneStorage::Heap, heap) => {
            let heap = heap.as_ref().unwrap();
            let made = heap.new_texture(&texture_desc(format, size, Some(0x20)), at);
            made.unwrap().unwrap()
        }
    };
    let library = d.new_library(PROBE).unwrap().unwrap().unwrap();
    let f = library
        .new_function(OnFault::Die, "copy_texels")
        .unwrap()
        .unwrap();
    let state = d
        .new_compute_pipeline_state_for_function(Some(&f))
        .unwrap()
        .unwrap();
    let q = d.new_command_queue().unwrap();
    let written = |format, (w, h): (u64, u64), seed: u32, at: u64| {
        let halves = texture(115, (w, h));
        let bytes: Vec<u8> = noise(seed, w * h * 4)
            .iter()
            .flat_map(|&b| {
                let value = f32::from(b) / 256.0 * 1.75 - 0.5;
                half_bits(value).to_le_bytes()
            })
            .collect();
        let all = MtlRegion::new(MtlOrigin::new(0, 0, 0), MtlSize::new(w, h, 1));
        halves
            .replace_region(all, 0, 0, HostRange::of_slice(&bytes), w * 8, 0)
            .unwrap();
        let t = two_plane(format, (w, h), at);
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = cb.compute_encoder(OnFault::Die).unwrap().unwrap();
        for cmd in [
            ComputeCmd::SetComputePipelineState {
                state: Some(&state),
            },
            ComputeCmd::SetTextures {
                textures: &[Some(&halves), Some(&t)],
                range: NsRange::new(0, 2),
            },
            ComputeCmd::DispatchThreads {
                threads_per_grid: MtlSize::new(w, h, 1),
                threads_per_threadgroup: MtlSize::new(8, 8, 1),
            },
            ComputeCmd::EndEncoding,
        ] {
            e.encode(OnFault::Die, cmd).unwrap();
        }
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        t
    };
    // Each in its own half of the heap, where they lie in one.
    let (source, destination) = (
        written(from, sizes.0, 3, 0),
        written(to, sizes.1, 4, 1 << 19),
    );
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let e = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
    let blit = match copy {
        TwoPlaneCopy::Region(((sx, sy), (cw, ch), (dx, dy))) => BlitCmd::CopyTextureRegion {
            source_texture: &source,
            source_slice: 0,
            source_level: 0,
            source_origin: MtlOrigin::new(sx, sy, 0),
            source_size: MtlSize::new(cw, ch, 1),
            destination_texture: &destination,
            destination_slice: 0,
            destination_level: 0,
            destination_origin: MtlOrigin::new(dx, dy, 0),
        },
        TwoPlaneCopy::Slices => BlitCmd::CopyTextureSlices {
            source_texture: &source,
            source_slice: 0,
            source_level: 0,
            destination_texture: &destination,
            destination_slice: 0,
            destination_level: 0,
            slice_count: 1,
            level_count: 1,
        },
    };
    for cmd in [blit, BlitCmd::EndEncoding] {
        e.encode(OnFault::Die, cmd).unwrap();
    }
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    probed_texels::<P>(d, ("read_texels", 8), &destination, sizes.1, None, None)
}

/// The blits between two-plane XR10 textures
/// [`two_plane_xr10_copies_of_any_region_are_metals`] compares with Metal, beyond
/// [`two_plane_xr10_blits_are_metals`]'s: a copy of the whole of level 0 and
/// slice 0 (`CopyTextureSlices`), and regions whose origins and widths are no multiple of 4,
/// between textures of widths that are a multiple of 16, of 4 and of neither, of one width
/// and of two, the destination's rows wider than the copy's and narrower; of the device's
/// textures of the default storage and `Private`, and, of a placement heap's, the copies that
/// read and write no byte outside the texels (those that do die:
/// [`two_plane_xr10_blits_metal_was_not_measured_at_die`]).
const TWO_PLANE_COPIES: &[TwoPlaneShape] = {
    use TwoPlaneCopy::{Region, Slices};
    use TwoPlaneStorage::{Default as Made, Heap, Private};
    &[
        (((16, 8), (16, 8)), Slices, Made),
        (((16, 8), (16, 8)), Region(((1, 0), (3, 8), (1, 0))), Made),
        (((16, 8), (16, 8)), Region(((1, 2), (3, 4), (5, 1))), Made),
        (((16, 8), (16, 8)), Region(((2, 1), (5, 3), (9, 4))), Made),
        (((16, 8), (16, 8)), Region(((3, 0), (1, 8), (0, 0))), Made),
        (((16, 8), (16, 8)), Region(((4, 0), (4, 8), (1, 0))), Made),
        (((16, 8), (16, 8)), Region(((1, 0), (4, 8), (4, 0))), Made),
        (((16, 8), (16, 8)), Region(((0, 1), (16, 3), (0, 4))), Made),
        (((16, 8), (16, 8)), Region(((5, 3), (7, 5), (2, 1))), Made),
        (((16, 8), (16, 8)), Region(((13, 0), (3, 8), (0, 0))), Made),
        (((12, 8), (16, 8)), Region(((0, 0), (12, 8), (2, 0))), Made),
        (((12, 8), (16, 8)), Region(((1, 1), (11, 6), (3, 2))), Made),
        (((16, 8), (12, 8)), Region(((3, 0), (9, 8), (0, 0))), Made),
        (((12, 6), (12, 6)), Slices, Made),
        (((9, 7), (9, 7)), Region(((0, 0), (8, 7), (1, 0))), Made),
        (((9, 7), (9, 7)), Region(((1, 0), (7, 7), (0, 0))), Made),
        (((9, 7), (9, 7)), Slices, Made),
        (((20, 4), (20, 4)), Slices, Made),
        (((20, 4), (20, 4)), Region(((1, 1), (17, 2), (2, 2))), Made),
        (((5, 5), (5, 5)), Region(((1, 0), (3, 5), (2, 0))), Made),
        (((6, 4), (6, 4)), Region(((1, 0), (4, 4), (2, 0))), Made),
        (((16, 8), (16, 8)), Slices, Private),
        (
            ((16, 8), (16, 8)),
            Region(((1, 0), (3, 8), (1, 0))),
            Private,
        ),
        (
            ((16, 8), (16, 8)),
            Region(((1, 2), (3, 4), (5, 1))),
            Private,
        ),
        (((16, 8), (16, 8)), Region(((0, 0), (16, 1), (0, 0))), Heap),
        (((16, 8), (16, 8)), Region(((1, 1), (3, 1), (5, 0))), Heap),
    ]
};

/// Each blit of [`TWO_PLANE_COPIES`], from either two-plane XR10 format into either, moves
/// both planes' bytes as Metal's does on this host: `read()` of the destination answers
/// Metal's texels
///
/// The runtime binds a two-plane XR10 texture to its kernels as `RGBA16Float`, a format that may
/// hold a NaN, and announces `device-nan-encoding` there: the first dispatch of this platform's
/// side dies, naming the kind ([`crate::answer::DIES`]).
#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "MoltenVK: the runtime records device-pack-conversion, a death in strict mode; Mac-only test, off everywhere; ledger: MoltenVK strict deaths"
)]
fn two_plane_xr10_copies_of_any_region_are_metals() {
    let site = crate::answer::dies_of(metal2vulkan_runtime::Degradation::DeviceNanEncoding, || {
        let d = dev();
        let vulkan_desc =
            |t: &TextureDesc| crate::platform::tests::platform().texture_descriptor(t);
        for &shape in TWO_PLANE_COPIES {
            for formats in [(550, 550), (550, 551), (551, 550), (551, 551)] {
                let key = format!("{shape:?} {formats:?}");
                let ours = two_plane_copy_on::<VulkanPlatform>(&d, &vulkan_desc, formats, shape);
                crate::metal_check_raw!(key, ours);
            }
        }
    });
    assert_eq!(site.file, "dispatchThreads:threadsPerThreadgroup:");
}

/// The bits of the half nearest `value`, which is normal or zero.
fn half_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = u16::try_from(bits >> 31).unwrap() << 15;
    if value == 0.0 {
        return sign;
    }
    let exponent = i32::try_from((bits >> 23) & 0xff).unwrap() - 127 + 15;
    let mantissa = (bits >> 13) & 0x3ff;
    let rounded = u16::try_from((exponent << 10) as u32 + mantissa).unwrap();
    sign | rounded
}

/// A blit between two textures of the two-plane XR10 formats moves both planes' bytes as
/// Metal's does on this host, whole and of a region to another place, from either format
/// into either: `read()` of the destination answers Metal's texels
///
/// The runtime binds a two-plane XR10 texture to its kernels as `RGBA16Float`, a format that may
/// hold a NaN, and announces `device-nan-encoding` there: the first dispatch of this platform's
/// side dies, naming the kind ([`crate::answer::DIES`]).
#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "MoltenVK: the runtime records device-pack-conversion, a death in strict mode; Mac-only test, off everywhere; ledger: MoltenVK strict deaths"
)]
fn two_plane_xr10_blits_are_metals() {
    let site = crate::answer::dies_of(metal2vulkan_runtime::Degradation::DeviceNanEncoding, || {
        let d = dev();
        let vulkan_desc =
            |t: &TextureDesc| crate::platform::tests::platform().texture_descriptor(t);
        for formats in [(550, 550), (550, 551), (551, 550), (551, 551)] {
            let whole = two_plane_blit_on::<VulkanPlatform>(&d, &vulkan_desc, formats, None);
            // Each texel's alpha half, from plane 1: the blit moved more than one value.
            let alphas: std::collections::BTreeSet<&[u8]> =
                whole.chunks(8).map(|texel| &texel[6..]).collect();
            assert!(alphas.len() > 1, "{formats:?}");
            for region in [None, Some(((4, 2), (8, 4), (8, 3)))] {
                let ours = two_plane_blit_on::<VulkanPlatform>(&d, &vulkan_desc, formats, region);
                assert_eq!(region.is_none(), ours == whole, "{formats:?} {region:?}");
                let key = format!("{formats:?} {region:?}");
                crate::metal_check_raw!(key, ours);
            }
        }
    });
    assert_eq!(site.file, "dispatchThreads:threadsPerThreadgroup:");
}

/// The bytes of a placement heap of storage `storage` (0 Shared, 2 Private) on `P`'s `d`
/// after a copy into a 16 by 8 texture of 551 at its offset 0, of one level, from a texture of
/// 550 of the device's whose texels a kernel wrote from an `RGBA16Float` texture of halves of
/// every sign, then `fills` filled, each in a command buffer of its own: of the whole level
/// (`CopyTextureSlices`) where `region` is `None`, or of a region (origin, size) to a
/// destination origin. A buffer placed over the texture's span and 4096 bytes past it, each
/// first its offset's low byte, reads the heap's bytes, through a copy into a Shared buffer.
fn two_plane_copy_into_heap_on<P: crate::core::platform::Platform>(
    d: &P::Device,
    desc: &dyn Fn(&TextureDesc) -> P::TextureDescriptor,
    storage: u64,
    region: Option<BlitRegion>,
    fills: &[Fill],
) -> Vec<u8> {
    use crate::core::platform::HostRange;
    use crate::core::platform::blit::BlitEncoder as _;
    use crate::core::platform::compute::{ComputeCmd, ComputeEncoder as _};
    use crate::core::platform::resource::{Heap as _, Library as _, Texture as _};
    use crate::core::platform::types::MtlRegion;
    use crate::core::platform::values::{HeapDesc, OptionSet};

    use crate::gpu::ycbcr::measure::PROBE;
    let (w, h) = (16, 8);
    let heap = d
        .new_heap(&HeapDesc {
            size: 0x10000,
            heap_type: 1,
            options: vec![
                OptionSet::StorageMode(storage),
                OptionSet::HazardTrackingMode(0),
            ],
            address_ranges: None,
            pinned_gpu_address: None,
        })
        .unwrap()
        .heap
        .unwrap();
    let into = desc(&TextureDesc {
        pixel_format: 551,
        width: w,
        height: h,
        usage: 0x3,
        resource_options: Some(0x100 | storage << 4),
        ..TextureDesc::default()
    });
    let size = d.heap_texture_size_and_align(&into).unwrap().size;
    let t = heap.new_texture(&into, 0).unwrap().unwrap();
    let span = size + 4096;
    let over = heap
        .new_buffer_of_length(span, storage << 4, 0)
        .unwrap()
        .unwrap();
    let shared = |bytes: &[u8]| {
        let length = u64::try_from(bytes.len()).unwrap();
        let b = d
            .new_buffer_of_length(OnFault::Die, length, 0)
            .unwrap()
            .unwrap();
        // SAFETY: a Shared buffer of `bytes.len()` bytes, which no command uses yet.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), b.contents().as_ptr::<u8>(), bytes.len());
        }
        b
    };
    let q = d.new_command_queue().unwrap();
    let blit = |cmd: BlitCmd<'_, P>| {
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
        e.encode(OnFault::Die, cmd).unwrap();
        e.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        assert!(cb.error().is_none(), "{:?}", cb.error());
    };
    let copied = |from: &P::Buffer, to: &P::Buffer| {
        blit(BlitCmd::CopyBufferToBuffer {
            source_buffer: Some(from),
            source_offset: 0,
            destination_buffer: Some(to),
            destination_offset: 0,
            size: span,
        });
    };
    let n = usize::try_from(span).unwrap();
    let offsets: Vec<u8> = (0..n).map(|i| (i & 0xff) as u8).collect();
    copied(&shared(&offsets), &over);
    let halves_texture = d
        .new_texture(
            OnFault::Return,
            &desc(&TextureDesc {
                pixel_format: 115,
                width: w,
                height: h,
                usage: 0x1,
                ..TextureDesc::default()
            }),
        )
        .unwrap()
        .unwrap();
    let mut x = 9u32;
    let halves: Vec<u8> = (0..w * h * 4)
        .flat_map(|_| {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            u16::try_from((x >> 16) & 0xbfff).unwrap().to_le_bytes()
        })
        .collect();
    let all = MtlRegion::new(MtlOrigin::new(0, 0, 0), MtlSize::new(w, h, 1));
    halves_texture
        .replace_region(all, 0, 0, HostRange::of_slice(&halves), w * 8, 0)
        .unwrap();
    let source = d
        .new_texture(
            OnFault::Return,
            &desc(&TextureDesc {
                pixel_format: 550,
                width: w,
                height: h,
                usage: 0x3,
                ..TextureDesc::default()
            }),
        )
        .unwrap()
        .unwrap();
    let library = d.new_library(PROBE).unwrap().unwrap().unwrap();
    let f = library
        .new_function(OnFault::Die, "copy_texels")
        .unwrap()
        .unwrap();
    let state = d
        .new_compute_pipeline_state_for_function(Some(&f))
        .unwrap()
        .unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let e = cb.compute_encoder(OnFault::Die).unwrap().unwrap();
    for cmd in [
        ComputeCmd::SetComputePipelineState {
            state: Some(&state),
        },
        ComputeCmd::SetTextures {
            textures: &[Some(&halves_texture), Some(&source)],
            range: NsRange::new(0, 2),
        },
        ComputeCmd::DispatchThreads {
            threads_per_grid: MtlSize::new(w, h, 1),
            threads_per_threadgroup: MtlSize::new(8, 4, 1),
        },
        ComputeCmd::EndEncoding,
    ] {
        e.encode(OnFault::Die, cmd).unwrap();
    }
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    assert!(cb.error().is_none(), "{:?}", cb.error());
    for f in fills {
        blit(fill_of::<P>(&source, f));
    }
    blit(match region {
        None => BlitCmd::CopyTextureSlices {
            source_texture: &source,
            source_slice: 0,
            source_level: 0,
            destination_texture: &t,
            destination_slice: 0,
            destination_level: 0,
            slice_count: 1,
            level_count: 1,
        },
        Some(((sx, sy), (cw, ch), (dx, dy))) => BlitCmd::CopyTextureRegion {
            source_texture: &source,
            source_slice: 0,
            source_level: 0,
            source_origin: MtlOrigin::new(sx, sy, 0),
            source_size: MtlSize::new(cw, ch, 1),
            destination_texture: &t,
            destination_slice: 0,
            destination_level: 0,
            destination_origin: MtlOrigin::new(dx, dy, 0),
        },
    });
    let back = shared(&vec![0; n]);
    copied(&over, &back);
    // SAFETY: a Shared buffer's `span` bytes, read after the copy completed.
    unsafe { std::slice::from_raw_parts(back.contents().as_ptr::<u8>(), n) }.to_vec()
}

/// A copy into a placement heap's one-level 2D texture of 551, Shared or Private, of the whole
/// level or of rows that end at its last, writes the heap's bytes as Metal's blit does on this
/// host, byte for byte ([`two_plane_copy_into_heap_on`]): at Metal's blit rows, so a copy of
/// more than one row writes alpha bytes past the alpha plane, over what else the heap holds
/// there; the runtime's copy into a heap's texture writes them
///
/// The runtime binds a two-plane XR10 texture to its kernels as `RGBA16Float`, a format that may
/// hold a NaN, and announces `device-nan-encoding` there: the first dispatch of this platform's
/// side dies, naming the kind ([`crate::answer::DIES`]).
#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "MoltenVK: the runtime records device-pack-conversion, a death in strict mode; Mac-only test, off everywhere; ledger: MoltenVK strict deaths"
)]
fn two_plane_copies_into_a_heap_are_metals() {
    let site = crate::answer::dies_of(metal2vulkan_runtime::Degradation::DeviceNanEncoding, || {
        let d = dev();
        let p = crate::platform::tests::platform();
        let ours_desc = |t: &TextureDesc| p.texture_descriptor(t);
        let shapes = [
            None,
            Some(((0, 0), (16, 2), (0, 6))),
            Some(((2, 1), (8, 3), (4, 5))),
        ];
        for storage in [0, 2] {
            for region in shapes {
                let ours = two_plane_copy_into_heap_on::<VulkanPlatform>(
                    &d,
                    &ours_desc,
                    storage,
                    region,
                    &[],
                );
                let key = format!("storage {storage} {region:?}");
                crate::metal_check_raw!(key, ours);
            }
        }
    });
    assert_eq!(site.file, "dispatchThreads:threadsPerThreadgroup:");
}

/// A copy into a texture of bytes past a no-copy buffer's length reads the caller's bytes
/// there, Metal's answer on the M3 Max; past another buffer's end it dies at its
/// selector, where Metal copies bytes no one wrote
#[test]
fn copies_past_a_buffer() {
    let d = dev();
    let page: Vec<u8> = (0..64u8).collect();
    let vulkan_desc = |t: &TextureDesc| crate::platform::tests::platform().texture_descriptor(t);
    assert_eq!(
        copied_past_a_no_copy_buffer_on::<VulkanPlatform>(&d, &vulkan_desc),
        page
    );
    let t = d
        .new_texture(OnFault::Return, &bgra(4, 4, 0))
        .unwrap()
        .unwrap();
    let short = shared(63);
    let cb = buffer();
    let e = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
    let up = BlitCmd::CopyBufferToTexture {
        source_buffer: &short,
        source_offset: 0,
        source_bytes_per_row: 16,
        source_bytes_per_image: 64,
        source_size: MtlSize::new(4, 4, 1),
        destination_texture: &t,
        destination_slice: 0,
        destination_level: 0,
        destination_origin: MtlOrigin::new(0, 0, 0),
    };
    let sel = up.selector();
    let site = catch_death(|| e.encode(OnFault::Return, up)).unwrap_err();
    assert_eq!(site.file, sel);
}

/// `MTLPixelFormatStencil8`.
const STENCIL8: u64 = 253;

/// The bytes on platform `P` of a copy of `units` from a buffer into every sample of a 2D
/// multisample texture of `format`, `samples` samples a pixel, `width` by `height` texels
/// (Private, read by shaders and rendered to), then of the whole texture back into a buffer:
/// rows `units.len() / height` bytes apart, a pixel's samples consecutive, Metal's layout. Where
/// `cleared`, a pass that clears and stores the texture's depth (its stencil, for `Stencil8`)
/// runs first in the same command buffer, as a guest's first pass over a new target does.
fn multisampled_copies_on<P: crate::core::platform::Platform>(
    p: &P,
    d: &P::Device,
    cleared: bool,
    (format, samples, width, height): (u64, u64, u64, u64),
    units: &[u8],
) -> Vec<u8> {
    use crate::core::platform::descriptor::RenderPassDescriptor as _;
    use crate::core::platform::render::{RenderCmd, RenderEncoder as _};
    use crate::core::platform::values::{DepthAttachmentSet, StencilAttachmentSet};
    let t = d
        .new_texture(
            OnFault::Return,
            &p.texture_descriptor(&TextureDesc {
                texture_type: 4,
                pixel_format: format,
                width,
                height,
                sample_count: samples,
                usage: 0x5,
                resource_options: Some(0x20),
                ..TextureDesc::default()
            }),
        )
        .unwrap()
        .unwrap();
    let length = units.len() as u64;
    let [up, down] = [(); 2].map(|()| {
        d.new_buffer_of_length(OnFault::Return, length, 0)
            .unwrap()
            .unwrap()
    });
    // SAFETY: a Shared buffer of `units.len()` bytes the GPU does not use yet.
    unsafe {
        std::ptr::copy_nonoverlapping(units.as_ptr(), up.contents().as_ptr::<u8>(), units.len())
    };
    let row = length / height;
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
    if cleared {
        // MTLLoadActionClear and MTLStoreActionStore.
        let (load_action, store_action) = (Some(2), Some(1));
        let pass = p.render_pass_descriptor();
        match format {
            STENCIL8 => pass.set_stencil_attachment(&StencilAttachmentSet {
                texture: Some(Some(&t)),
                load_action,
                store_action,
                ..StencilAttachmentSet::default()
            }),
            _ => pass.set_depth_attachment(&DepthAttachmentSet {
                texture: Some(Some(&t)),
                load_action,
                store_action,
                ..DepthAttachmentSet::default()
            }),
        }
        let r = cb.render_encoder(OnFault::Return, &pass).unwrap().unwrap();
        r.encode(OnFault::Return, RenderCmd::EndEncoding).unwrap();
    }
    let e = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
    let cmds = [
        BlitCmd::CopyBufferToTexture {
            source_buffer: &up,
            source_offset: 0,
            source_bytes_per_row: row,
            source_bytes_per_image: length,
            source_size: MtlSize::new(width, height, 1),
            destination_texture: &t,
            destination_slice: 0,
            destination_level: 0,
            destination_origin: MtlOrigin::new(0, 0, 0),
        },
        BlitCmd::CopyTextureToBuffer {
            source_texture: &t,
            source_slice: 0,
            source_level: 0,
            source_origin: MtlOrigin::new(0, 0, 0),
            source_size: MtlSize::new(width, height, 1),
            destination_buffer: &down,
            destination_offset: 0,
            destination_bytes_per_row: row,
            destination_bytes_per_image: length,
        },
        BlitCmd::EndEncoding,
    ];
    for cmd in cmds {
        e.encode(OnFault::Return, cmd).unwrap();
    }
    cb.commit(OnFault::Return).unwrap();
    cb.wait_until_completed(OnFault::Return).unwrap();
    // SAFETY: a Shared buffer's `units.len()` bytes, read after the GPU's writes completed.
    unsafe { std::slice::from_raw_parts(down.contents().as_ptr::<u8>(), units.len()) }.to_vec()
}

/// A copy from a buffer into a multisampled `Depth32Float` or `Stencil8` texture and back puts
/// each unit in its own sample and returns the buffer bit for bit, as Metal's does on the M3
/// Max, the texture cleared by a pass first or not, as a macOS 26 guest's screensaver
/// and HiDPI login window copy into a 4-sample `Depth32Float` texture of 54 by 54 and of 80
/// by 80. The depths lie in `[0, 1]`, none with the sign bit set, where the runtime's sample-rate draw writes Metal's
/// values; past them MoltenVK's draw changes them (`drawn-depth-copy`, relayed).
///
/// The two `Depth32Float` shapes Metal keeps losslessly compressed (16 texels or more on the
/// shorter side) read the copied values too where no pass cleared them first: the runtime counts
/// a blit that writes a whole level as writing the pattern Metal's new texture reads.
///
/// On a device without `VK_EXT_depth_range_unrestricted` (MoltenVK), which clamps a depth outside
/// `[0, 1]`, each copy into a `Depth32Float` texture dies of `restricted-depth-upload` in strict
/// mode, and by default returns the buffer bit for bit, its depths inside `[0, 1]` as they are
#[test]
fn multisampled_depth_and_stencil_buffer_copies_are_metals() {
    use crate::gpu::device::tests::uploading_depth;
    let vulkan = crate::platform::tests::platform();
    let depths = |n: u64| -> Vec<u8> {
        (0..n)
            .flat_map(|i| ((i % 4096) as f32 / 4095.0).to_le_bytes())
            .collect()
    };
    // Each case's name, shape and bytes.
    let cases = [
        (
            "the screensaver's Depth32Float",
            (252, 4, 54, 54),
            depths(54 * 54 * 4),
        ),
        (
            "the login window's Depth32Float",
            (252, 4, 80, 80),
            depths(80 * 80 * 4),
        ),
        (
            "Depth32Float of 2 samples",
            (252, 2, 8, 8),
            depths(8 * 8 * 2),
        ),
        (
            "Stencil8 of 4 samples",
            (STENCIL8, 4, 8, 8),
            noise(0x5e, 8 * 8 * 4),
        ),
        (
            "Stencil8 of 2 samples",
            (STENCIL8, 2, 8, 8),
            noise(0x5f, 8 * 8 * 2),
        ),
    ];
    let mut differ = Vec::new();
    for (name, shape, units) in cases {
        for cleared in [true, false] {
            let what = match cleared {
                true => format!("{name}, cleared first"),
                false => format!("{name}, uncleared"),
            };
            *crate::answer::LAST_LINE.lock() = None;
            let d = dev();
            let copies = || multisampled_copies_on(&vulkan, &d, cleared, shape, &units);
            let ours = catch_death(|| match shape.0 {
                252 => uploading_depth(&d, UPLOAD, copies),
                _ => copies(),
            });
            let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
            match ours {
                Ok(ours) if ours == units => {}
                Ok(ours) => {
                    let wrong = (ours.iter().zip(&units)).filter(|(a, b)| a != b).count();
                    differ.push(format!("{what}: {wrong} bytes differ"));
                }
                Err(site) => differ.push(format!("{what}: died at {}: {line}", site.file)),
            }
        }
    }
    assert!(differ.is_empty(), "{differ:#?}");
}

/// A fill of a case: bytes, or a colour in a pixel format; at a level and slice, of a region
/// given as origin and size.
#[derive(Debug, Clone)]
enum Fill {
    Bytes(u64, u64, [u64; 3], [u64; 3], Vec<u8>),
    Color(u64, u64, [u64; 3], [u64; 3], [f64; 4], u64),
}

/// The blit of `fill` into `t`.
fn fill_of<'a, P: crate::core::platform::Platform>(
    t: &'a P::Texture,
    fill: &'a Fill,
) -> BlitCmd<'a, P> {
    let region = |o: [u64; 3], s: [u64; 3]| {
        MtlRegion::new(
            MtlOrigin::new(o[0], o[1], o[2]),
            MtlSize::new(s[0], s[1], s[2]),
        )
    };
    match *fill {
        Fill::Bytes(level, slice, o, s, ref bytes) => BlitCmd::FillTexture {
            texture: t,
            level,
            slice,
            region: region(o, s),
            bytes: crate::core::platform::HostRange::of_slice(bytes),
        },
        Fill::Color(level, slice, o, s, [red, green, blue, alpha], pixel_format) => {
            BlitCmd::FillTextureColor {
                texture: t,
                level,
                slice,
                region: region(o, s),
                color: ClearColor {
                    red,
                    green,
                    blue,
                    alpha,
                },
                pixel_format,
            }
        }
    }
}

/// A texture of `format` (`bpp` bytes a block of `block` texels across and down: one texel
/// but for a compressed format), `texture_type`, `extent`, `levels` and `array` slices,
/// every byte 0xee, then `fills`, `Private`, its `allowGPUOptimizedContents` the
/// descriptor's (on) unless `allow_gpu_optimized_contents` sets it. Its bytes are written and
/// read back as a whole, or, where `aspects` lists them, by each aspect's copies: its blit
/// option and its bytes a texel (`Depth32Float_Stencil8`'s depth, option 1, and stencil,
/// option 2).
struct Case {
    format: u64,
    bpp: u64,
    block: [u64; 2],
    aspects: &'static [(u64, u64)],
    texture_type: u64,
    extent: [u64; 3],
    levels: u64,
    array: u64,
    usage: u64,
    allow_gpu_optimized_contents: Option<bool>,
    /// Whether the texture is shareable (`newSharedTextureWithDescriptor:`).
    shared: bool,
    fills: Vec<Fill>,
}

/// The texture `c` fills.
fn texture_of(c: &Case) -> TextureDesc {
    TextureDesc {
        texture_type: c.texture_type,
        pixel_format: c.format,
        width: c.extent[0],
        height: c.extent[1],
        depth: c.extent[2],
        mipmap_level_count: c.levels,
        array_length: c.array,
        usage: c.usage,
        resource_options: Some(0x20),
        allow_gpu_optimized_contents: c.allow_gpu_optimized_contents,
        ..TextureDesc::default()
    }
}

/// Each slice's levels of `c`'s texture, each level's aspects, in order: its slice, level,
/// extent, blit option and bytes a block.
fn images_of(c: &Case) -> Vec<(u64, u64, [u64; 3], u64, u64)> {
    let slices = match c.texture_type {
        1 | 3 => c.array,
        5 => 6,
        6 => 6 * c.array,
        _ => 1,
    };
    let whole = [(0, c.bpp)];
    let aspects: &[(u64, u64)] = if c.aspects.is_empty() {
        &whole
    } else {
        c.aspects
    };
    (0..slices)
        .flat_map(|slice| {
            (0..c.levels).flat_map(move |level| {
                let e = c.extent.map(|e| (e >> level).max(1));
                aspects
                    .iter()
                    .map(move |&(option, bpp)| (slice, level, e, option, bpp))
            })
        })
        .collect()
}

/// The bytes of a level of `c`'s texture of `extent`, at `bpp` bytes a block.
fn level_bytes(c: &Case, extent: [u64; 3], bpp: u64) -> u64 {
    extent[0].div_ceil(c.block[0]) * bpp * extent[1].div_ceil(c.block[1]) * extent[2]
}

/// Whether `c`'s texture holds a 32-bit float depth: `Depth32Float` (252) or
/// `Depth32Float_Stencil8` (260).
fn holds_float_depth(c: &Case) -> bool {
    matches!(c.format, 252 | 260)
}

/// The ranges of `c`'s bytes read back ([`fills_on`]) that hold its texture's 32-bit float
/// depths, 4 bytes a texel: every image's but a stencil copy's (option 2). A copy of no option of
/// `Depth32Float_Stencil8` moves its depth alone, as a depth copy (option 1) does.
fn depth_ranges(c: &Case) -> Vec<std::ops::Range<usize>> {
    if !holds_float_depth(c) {
        return Vec::new();
    }
    let mut at = 0;
    (images_of(c).into_iter())
        .filter_map(|(_, _, e, option, bpp)| {
            let range = at..at + usize::try_from(level_bytes(c, e, bpp)).unwrap();
            at = range.end;
            (option != 2).then_some(range)
        })
        .collect()
}

/// Each (slice, level) of `c`'s texture after its fills on platform `P`, read back, in order,
/// each aspect's bytes in turn.
fn fills_on<P: crate::core::platform::Platform>(
    d: &P::Device,
    desc: &dyn Fn(&TextureDesc) -> P::TextureDescriptor,
    c: &Case,
) -> Vec<u8> {
    use crate::core::platform::resource::Buffer as _;
    let desc = desc(&texture_of(c));
    let t = match c.shared {
        true => d.new_shared_texture(&desc),
        false => d.new_texture(OnFault::Return, &desc),
    };
    let t = t.unwrap().unwrap();
    let images = images_of(c);
    let buffer_of = |bytes: &[u8]| {
        let b = d
            .new_buffer_of_length(OnFault::Return, bytes.len() as u64, 0)
            .unwrap()
            .unwrap();
        // SAFETY: a fresh Shared buffer of `bytes.len()` bytes, written before any use.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), b.contents().as_ptr(), bytes.len())
        };
        b
    };
    let run = |cmds: Vec<BlitCmd<'_, P>>| {
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
        let e = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
        for cmd in cmds {
            e.encode(OnFault::Return, cmd).unwrap();
        }
        e.encode(OnFault::Return, BlitCmd::EndEncoding).unwrap();
        cb.commit(OnFault::Return).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
    };
    // A level's bytes a row of blocks of `bpp` bytes, and its bytes.
    let row_of = |e: [u64; 3], bpp: u64| e[0].div_ceil(c.block[0]) * bpp;
    let bytes_of = |e: [u64; 3], bpp| row_of(e, bpp) * e[1].div_ceil(c.block[1]) * e[2];
    let ups: Vec<_> = images
        .iter()
        .map(|&(_, _, e, _, bpp)| {
            buffer_of(&vec![0xee; usize::try_from(bytes_of(e, bpp)).unwrap()])
        })
        .collect();
    run(images
        .iter()
        .zip(&ups)
        .map(|(&(slice, level, e, option, bpp), up)| {
            let (source_bytes_per_row, source_bytes_per_image) =
                (row_of(e, bpp), bytes_of([e[0], e[1], 1], bpp));
            let (source_size, origin) = (MtlSize::new(e[0], e[1], e[2]), MtlOrigin::new(0, 0, 0));
            if option == 0 {
                return BlitCmd::CopyBufferToTexture {
                    source_buffer: up,
                    source_offset: 0,
                    source_bytes_per_row,
                    source_bytes_per_image,
                    source_size,
                    destination_texture: &t,
                    destination_slice: slice,
                    destination_level: level,
                    destination_origin: origin,
                };
            }
            BlitCmd::CopyBufferToTextureOptions {
                source_buffer: Some(up),
                source_offset: 0,
                source_bytes_per_row,
                source_bytes_per_image,
                source_size,
                destination_texture: Some(&t),
                destination_slice: slice,
                destination_level: level,
                destination_origin: origin,
                options: option,
            }
        })
        .collect());
    let sources: Vec<_> = c
        .fills
        .iter()
        .map(|f| match f {
            Fill::Bytes(.., bytes) => Some(buffer_of(bytes)),
            Fill::Color(..) => None,
        })
        .collect();
    let region = |o: [u64; 3], s: [u64; 3]| {
        MtlRegion::new(
            MtlOrigin::new(o[0], o[1], o[2]),
            MtlSize::new(s[0], s[1], s[2]),
        )
    };
    run(c
        .fills
        .iter()
        .zip(&sources)
        .map(|(f, source)| match (f, source) {
            (&Fill::Bytes(level, slice, o, sz, ref bytes), Some(source)) => {
                BlitCmd::FillTexture {
                    texture: &t,
                    level,
                    slice,
                    region: region(o, sz),
                    // SAFETY: the source buffer's bytes, live past the send.
                    bytes: unsafe {
                        crate::core::platform::HostRange::new(source.contents(), bytes.len() as u64)
                    },
                }
            }
            (&Fill::Color(level, slice, o, sz, color, pixel_format), _) => {
                BlitCmd::FillTextureColor {
                    texture: &t,
                    level,
                    slice,
                    region: region(o, sz),
                    color: ClearColor {
                        red: color[0],
                        green: color[1],
                        blue: color[2],
                        alpha: color[3],
                    },
                    pixel_format,
                }
            }
            _ => unreachable!(),
        })
        .collect());
    let downs: Vec<_> = images
        .iter()
        .map(|&(_, _, e, _, bpp)| {
            buffer_of(&vec![0x55; usize::try_from(bytes_of(e, bpp)).unwrap()])
        })
        .collect();
    run(images
        .iter()
        .zip(&downs)
        .map(
            |(&(slice, level, e, option, bpp), down)| BlitCmd::CopyTextureToBufferOptions {
                source_texture: Some(&t),
                source_slice: slice,
                source_level: level,
                source_origin: MtlOrigin::new(0, 0, 0),
                source_size: MtlSize::new(e[0], e[1], e[2]),
                destination_buffer: Some(down),
                destination_offset: 0,
                destination_bytes_per_row: row_of(e, bpp),
                destination_bytes_per_image: bytes_of([e[0], e[1], 1], bpp),
                options: option,
            },
        )
        .collect());
    downs
        .iter()
        .flat_map(|b| {
            let length = usize::try_from(b.length()).unwrap();
            // SAFETY: a Shared buffer's bytes, read after the GPU's writes completed.
            unsafe { std::slice::from_raw_parts(b.contents().as_ptr::<u8>(), length) }.to_vec()
        })
        .collect()
}

/// The texture fills write what Metal writes, where it writes it: bytes past a block go
/// unread and a shorter colour repeats; levels, slices, cube faces and 3D depths; regions cut
/// at the level, ignored slices and depths, and fills that write nothing; depth textures;
/// a fill larger than the staging buffer. On a device that clamps a depth outside `[0, 1]`
/// (MoltenVK), the `Depth32Float` texture's first upload dies of `restricted-depth-upload` in
/// strict mode, and its bytes by default are Metal's with each depth clamped
/// ([`fills_are_metals_under`])
#[test]
fn texture_fills_are_metals() {
    let rgba = [1.0, 0.5, 0.25, 1.0];
    let b = |level, slice, o, s, bytes: &[u8]| Fill::Bytes(level, slice, o, s, bytes.to_vec());
    let k = |level, slice, o, s, color, pf| Fill::Color(level, slice, o, s, color, pf);
    let one = [1, 1, 1];
    let case = |format, bpp, texture_type, extent, levels, array, fills| Case {
        format,
        bpp,
        block: [1, 1],
        aspects: &[],
        texture_type,
        extent,
        levels,
        array,
        usage: 3,
        allow_gpu_optimized_contents: None,
        shared: false,
        fills,
    };
    let mut cases = vec![
        case(
            70,
            4,
            2,
            [4, 4, 1],
            1,
            1,
            vec![
                k(0, 0, [1, 1, 0], [2, 2, 1], rgba, 70),
                b(0, 0, [0, 0, 0], [4, 1, 1], &[1, 2, 3, 4, 5, 6, 7, 8]),
                b(0, 0, [3, 3, 0], [4, 4, 1], &[9, 10, 11, 12]),
                k(0, 0, [0, 3, 0], one, [200.0, 100.0, 50.0, 255.0], 73),
                k(0, 0, [1, 3, 0], one, rgba, 10),
                k(0, 0, [2, 3, 0], one, rgba, 115),
                b(0, 7, [0, 2, 5], [1, 1, 0], &[0x21, 0x22, 0x23, 0x24]),
            ],
        ),
        case(
            10,
            1,
            2,
            [4, 2, 1],
            1,
            1,
            vec![
                b(0, 0, [0, 0, 0], [4, 1, 1], &[1, 2, 3, 4]),
                k(0, 0, [0, 1, 0], [2, 1, 1], rgba, 70),
            ],
        ),
        case(
            115,
            8,
            2,
            [2, 2, 1],
            1,
            1,
            vec![
                k(0, 0, [0, 0, 0], one, [1.0, 0.5, -0.25, 70000.0], 115),
                k(0, 0, [1, 0, 0], one, [0.5, 0.0, 0.0, 0.0], 10),
                k(0, 0, [0, 1, 0], [2, 1, 1], [f64::NAN, -2.0, 1e-6, 3.0], 125),
            ],
        ),
        case(
            125,
            16,
            2,
            [2, 1, 1],
            1,
            1,
            vec![
                k(0, 0, [0, 0, 0], one, [1.5, -2.0, 3.25, f64::NAN], 125),
                k(0, 0, [1, 0, 0], one, [70000.0, -1.0, 2.5, 4e9], 123),
            ],
        ),
        case(
            252,
            4,
            2,
            [2, 2, 1],
            1,
            1,
            vec![
                b(0, 0, [0, 0, 0], one, &[0, 0, 0, 0x3f]),
                k(0, 0, [1, 1, 0], one, [0.25, 0.0, 0.0, 0.0], 55),
            ],
        ),
        case(
            250,
            2,
            2,
            [2, 2, 1],
            1,
            1,
            vec![b(0, 0, [1, 0, 0], [1, 2, 1], &[0x34, 0x12])],
        ),
        case(
            70,
            4,
            3,
            [2, 1, 1],
            1,
            2,
            vec![
                b(0, 1, [0, 0, 1], [1, 1, 3], &[1, 2, 3, 4]),
                b(0, 2, [1, 0, 0], one, &[5, 6, 7, 8]),
            ],
        ),
        case(
            70,
            4,
            0,
            [4, 1, 1],
            1,
            1,
            vec![
                b(0, 3, [1, 0, 0], one, &[1, 2, 3, 4]),
                b(0, 0, [2, 1, 0], one, &[5, 6, 7, 8]),
                b(0, 0, [3, 0, 0], [1, 2, 1], &[9, 10, 11, 12]),
            ],
        ),
        case(
            70,
            4,
            1,
            [2, 1, 1],
            1,
            2,
            vec![
                b(0, 1, [0, 0, 0], one, &[1, 2, 3, 4]),
                b(0, 2, [1, 0, 0], one, &[5, 6, 7, 8]),
            ],
        ),
        case(
            70,
            4,
            5,
            [1, 1, 1],
            1,
            1,
            vec![
                b(0, 3, [0, 0, 0], one, &[1, 2, 3, 4]),
                b(0, 6, [0, 0, 0], one, &[5, 6, 7, 8]),
                b(0, 13, [0, 0, 0], one, &[9, 10, 11, 12]),
            ],
        ),
        case(
            70,
            4,
            6,
            [1, 1, 1],
            1,
            2,
            vec![
                b(0, 7, [0, 0, 0], one, &[1, 2, 3, 4]),
                b(0, 12, [0, 0, 0], one, &[5, 6, 7, 8]),
            ],
        ),
        case(
            70,
            4,
            7,
            [2, 2, 2],
            1,
            1,
            vec![
                b(0, 0, [0, 1, 1], [2, 1, 1], &[1, 2, 3, 4]),
                b(0, 1, [0, 0, 0], [1, 1, 3], &[5, 6, 7, 8]),
                b(0, 0, [1, 1, 0], [1, 1, 0], &[9, 10, 11, 12]),
            ],
        ),
        case(
            70,
            4,
            2,
            [4, 4, 1],
            3,
            1,
            vec![
                b(1, 0, [1, 0, 0], [4, 4, 1], &[1, 2, 3, 4]),
                k(2, 0, [0, 0, 0], one, rgba, 70),
                b(3, 0, [0, 0, 0], one, &[5, 6, 7, 8]),
                b(0, 0, [0, 0, 0], [0, 1, 1], &[9, 10, 11, 12]),
            ],
        ),
        case(
            70,
            4,
            2,
            [4096, 2048, 1],
            1,
            1,
            vec![k(
                0,
                0,
                [1, 1, 0],
                [65536, 65536, 1],
                [0.0, 0.25, 0.5, 1.0],
                70,
            )],
        ),
    ];
    let mut plain_usage = case(
        70,
        4,
        2,
        [2, 1, 1],
        1,
        1,
        vec![k(0, 0, [0, 0, 0], one, rgba, 70)],
    );
    plain_usage.usage = 1;
    cases.push(plain_usage);
    for (i, c) in cases.iter().enumerate() {
        fills_are_metals_under(&format!("case {i}"), c);
    }
}

/// `cases` filled on this platform, read back: each case's bytes are Metal's, recorded on the M3
/// Max ([`fills_are_metals_under`]). A case this platform dies at fails, naming the death's site
/// and line.
fn fills_are_metals(cases: &[Case]) {
    for (i, c) in cases.iter().enumerate() {
        fills_are_metals_under(&format!("case {i}, format {}", c.format), c);
    }
}

/// Case `c` filled on this platform, read back: its bytes are Metal's, recorded on the M3 Max
/// under `key`. On a device that clamps a 32-bit float depth into `[0, 1]`
/// (`gpu::device`'s `clamps_depth`: MoltenVK), a texture that holds one dies of
/// `restricted-depth-upload` at its first upload in strict mode, and its bytes by default are
/// Metal's with each depth clamped ([`clamped_fills_are_metals`]). A case this platform dies at
/// fails, naming the death's site and line.
fn fills_are_metals_under(key: &str, c: &Case) {
    use crate::gpu::device::tests::{clamps_depth, uploading_depth};
    let d = dev();
    let vulkan_desc = |t: &TextureDesc| crate::platform::tests::platform().texture_descriptor(t);
    let fills = || fills_on::<VulkanPlatform>(&d, &vulkan_desc, c);
    let upload = match c.aspects.first() {
        Some(&(option, _)) if option != 0 => UPLOAD_OPTIONS,
        _ => UPLOAD,
    };
    let ours = catch_death(|| match holds_float_depth(c) {
        true => uploading_depth(&d, upload, fills),
        false => fills(),
    })
    .unwrap_or_else(|site| {
        let line = crate::answer::LAST_LINE.lock().clone();
        panic!("{key}: died at {} ({:?}): {line:?}", site.file, site.kind)
    });
    if holds_float_depth(c) && clamps_depth(&d) {
        clamped_fills_are_metals(key, c, &ours);
    } else {
        crate::metal_check_bytes!(key, &ours);
    }
}

/// Checks `ours`, case `c`'s bytes read back on a device that clamps a 32-bit float depth into
/// `[0, 1]`, against Metal's under `key`: they are Metal's with each depth clamped
/// (`gpu::device`'s `clamped_depth`). Every byte [`fills_on`] writes first is 0xee, whose depth
/// is negative, which the device stores as 0 where Metal keeps it, and no fill of these cases
/// writes a depth of 0: so with 0xeeeeeeee in place of each depth of 0 in `ours` they are
/// Metal's bytes, recorded, and each such depth clamped answers `ours`.
fn clamped_fills_are_metals(key: &str, c: &Case, ours: &[u8]) {
    use crate::gpu::device::tests::clamped_depth;
    let first = [0xee; 4];
    assert_eq!(clamped_depth(u32::from_le_bytes(first)), 0);
    let mut metals = ours.to_vec();
    let mut clamped = ours.to_vec();
    for range in depth_ranges(c) {
        let (words, _) = metals[range.clone()].as_chunks_mut::<4>();
        let (kept, _) = clamped[range].as_chunks_mut::<4>();
        for (metal, kept) in words.iter_mut().zip(kept) {
            if *metal == [0; 4] {
                *metal = first;
            }
            *kept = clamped_depth(u32::from_le_bytes(*metal)).to_le_bytes();
        }
    }
    assert_eq!(clamped, ours, "{key}: each depth clamped into [0, 1]");
    crate::metal_check_bytes!(key, &metals);
}

/// The bytes of a Private 2D texture of `format`, of `bpp` bytes a texel, 64 texels wide, on
/// platform `P`, every byte 0xee, once each of `colours` filled a texel of it, in order from
/// its first, packed in `format`.
fn packed_on<P: crate::core::platform::Platform>(
    d: &P::Device,
    desc: &dyn Fn(&TextureDesc) -> P::TextureDescriptor,
    (format, bpp): (u64, u64),
    colours: &[[f64; 4]],
) -> Vec<u8> {
    let width = 64;
    let n = u64::try_from(colours.len()).unwrap();
    let fills = (0u64..)
        .zip(colours)
        .map(|(i, &c)| Fill::Color(0, 0, [i % width, i / width, 0], [1, 1, 1], c, format))
        .collect();
    let shape = (2, [width, n.div_ceil(width), 1], 1, 1);
    fills_on::<P>(d, desc, &read_case((format, bpp, [1, 1]), shape, fills))
}

/// Colours of `v`, each value in each component: colour `i` is values `i` to `i + 3`.
fn rotations(v: &[f64]) -> Vec<[f64; 4]> {
    (0..v.len())
        .map(|i| [0, 1, 2, 3].map(|c| v[(i + c) % v.len()]))
        .collect()
}

/// Values for the RGBX colour fills: plain values and each normalized, 8-, 16- and 32-bit
/// integer and half range's ends, halfway points and the values past them, of either sign;
/// sRGB's linear segment's end; denormals, infinities and NaNs, quiet and signalling.
fn rgbx_values() -> Vec<f64> {
    vec![
        0.0,
        -0.0,
        0.25,
        0.5,
        0.3,
        1.0,
        -1.0,
        1.5,
        -1.5,
        0.003_130_8,
        0.5 / 255.0,
        127.5,
        255.0,
        255.5,
        256.0,
        -128.5,
        -129.0,
        65_504.0,
        65_535.5,
        65_536.0,
        -32_768.5,
        70_000.0,
        2_147_483_647.5,
        -2_147_483_648.5,
        4_294_967_295.0,
        4_294_967_296.0,
        1e10,
        -1e10,
        1e-40,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        -f64::NAN,
        f64::from_bits(0x7ff4_0000_0000_0000),
    ]
}

/// The `f64`s at a quarter, a half and three quarters of the way from `a` to `b`, which lie
/// strictly between two neighbouring `f32`s `a` and `b`: exact, as they differ by a power of
/// two.
fn between(a: f64, b: f64) -> [f64; 3] {
    let step = (b - a) / 4.0;
    [a + step, a + 2.0 * step, a + 3.0 * step]
}

/// Values for the bfloat colour fills: plain values, past an `f32`'s range, denormals,
/// signed zeroes and infinities; at each of a range of exponents, values halfway between two
/// bfloats and the `f32`s beside them, of either sign, and the `f64`s strictly between each
/// and its neighbours ([`between`]), where packing through an `f32` first rounds twice; and
/// NaNs, quiet and signalling, of either sign, whose payloads set the bits a bfloat keeps and
/// the bit below them.
fn bfloat_values() -> Vec<f64> {
    let mut v = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.25,
        70_000.0,
        3.4e38,
        -3.4e38,
        f64::MAX,
        1e-40,
        -1e-40,
        1e-45,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ];
    let next = |x: f32, by: i32| f64::from(f32::from_bits(x.to_bits().wrapping_add_signed(by)));
    for exponent in [-133, -127, -126, -100, -20, -1, 0, 1, 20, 100, 126, 127] {
        for step in [0, 1, 2, 63, 126, 127] {
            let tie = (1.0 + f64::from(2 * step + 1) / 256.0) * 2f64.powi(exponent);
            for tie in [tie, -tie] {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "nine bits of mantissa at an f32's exponent, exact"
                )]
                let tie = tie as f32;
                let (below, at, above) = (next(tie, -1), f64::from(tie), next(tie, 1));
                v.extend([at, below, above]);
                v.extend(between(below, at));
                v.extend(between(at, above));
            }
        }
    }
    for bits in [
        0x7ff8_0000_0000_0000u64,
        0x7ff8_1000_0000_0000,
        0x7ff8_2000_0000_0000,
        0x7ffc_0000_0000_0000,
        0x7fff_ffff_ffff_ffff,
        0x7ff0_0000_0000_0001,
        0x7ff0_2000_0000_0000,
        0x7ff4_0000_0000_0000,
    ] {
        v.extend([f64::from_bits(bits), f64::from_bits(bits | 1 << 63)]);
    }
    v
}

/// A colour fill packed in a bfloat format (`R16Bfloat`, `RG16Bfloat`, `RGBA16Bfloat`), an
/// extended-range `_sRGB` twin (`BGRA10_XR_sRGB`, `BGR10_XR_sRGB`) or an RGBX format (each of
/// the 11), which the runtime's colour fill packs ([`fill::OWN_FORMAT_PACKED`]), writes Metal's
/// bytes on this host, each colour into a texel of a texture of its format ([`packed_on`]),
/// each value in each component ([`rotations`]): for bfloat, [`bfloat_values`]; for the
/// twins, the least `f32` of each code and the one below it, as Metal stores them here, the
/// `f64`s between the two ([`between`]), and NaNs, infinities,
/// signed zeroes and values past the ends; for `BGRA10_XR_sRGB`'s linear alpha, each value
/// halfway between two of its codes, `(k + 0.5 - 384) / 510`, the `f32`s beside it and the
/// `f64`s between them; for RGBX, [`rgbx_values`]. A colour of each bfloat format and twin
/// into a texture of another format (`RGBA32Uint`, `R8Unorm`), and an `RGBX8Unorm` colour
/// into an `RGBA8Unorm` texture, which Metal makes, dies at the fill: a twin's bytes there
/// follow no rule measured (a colour filled alone repeats; 2057 colours filled one after
/// another in one encoder each left its packed bytes, then `01` and zeroes, where the
/// runtime repeats), and a bfloat or RGBX colour's are not compared
#[test]
fn colours_the_runtime_packs_are_metals() {
    let vulkan_desc = |t: &TextureDesc| crate::platform::tests::platform().texture_descriptor(t);
    let bfloats = rotations(&bfloat_values());
    // Metal's edges, by their bits: the colours the rest fills.
    let edge_key = "the XR sRGB edges' bits";
    let edge_bits = crate::metal_value!(edge_key);
    let mut edges: Vec<f64> = (edge_bits.trim_matches(['[', ']']).split(", "))
        .map(|b| f64::from_bits(b.parse().unwrap()))
        .collect();
    let pairs: Vec<[f64; 2]> = edges.chunks(2).map(|p| [p[0], p[1]]).collect();
    edges.extend(
        pairs
            .iter()
            .flat_map(|&[below, least]| between(below, least)),
    );
    edges.extend([
        f64::NAN,
        -f64::NAN,
        f64::from_bits(0x7ff4_0000_0000_0000),
        f64::INFINITY,
        f64::NEG_INFINITY,
        0.0,
        -0.0,
        -2.0,
        2.0,
        1e-40,
        -1e-40,
    ]);
    let edges = rotations(&edges);
    let rgbx = rotations(&rgbx_values());
    let next = |x: f32, by: i32| f64::from(f32::from_bits(x.to_bits().wrapping_add_signed(by)));
    let mut alphas = Vec::new();
    for k in 383..=894 {
        let half = (f64::from(k) + 0.5 - 384.0) / 510.0;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the f32 nearest a code's halfway value"
        )]
        let near = half as f32;
        let (below, at, above) = (next(near, -1), f64::from(near), next(near, 1));
        alphas.extend([half, below, at, above]);
        alphas.extend(between(below, at));
        alphas.extend(between(at, above));
    }
    let mut twin_alphas = edges.clone();
    twin_alphas.extend(alphas.iter().map(|&a| [0.25, 0.5, 0.75, a]));
    let cases = [
        ((26, 2), &bfloats),
        ((66, 4), &bfloats),
        ((116, 8), &bfloats),
        ((553, 8), &twin_alphas),
        ((555, 4), &edges),
        ((300, 4), &rgbx),
        ((301, 4), &rgbx),
        ((302, 4), &rgbx),
        ((303, 4), &rgbx),
        ((304, 4), &rgbx),
        ((313, 8), &rgbx),
        ((314, 8), &rgbx),
        ((315, 8), &rgbx),
        ((323, 16), &rgbx),
        ((324, 16), &rgbx),
        ((325, 16), &rgbx),
    ];
    let mut differ = Vec::new();
    for ((format, bpp), colours) in cases {
        let shape = (format, bpp);
        let key = format!("{format}");
        let ours =
            catch_death(|| packed_on::<VulkanPlatform>(&dev(), &vulkan_desc, shape, colours))
                .unwrap_or_else(|site| {
                    let line = crate::answer::LAST_LINE.lock().clone();
                    panic!(
                        "{format}: died at {} ({:?}): {line:?}",
                        site.file, site.kind
                    )
                });
        // The colours' texels alone: past them lie texels no colour filled, whose bytes Metal's
        // blit out of a texture it keeps compressed rewrites (553's), where the runtime keeps them.
        let filled = colours.len() * usize::try_from(bpp).unwrap();
        if !crate::metal_matches_bytes!(key, &ours[..filled]) {
            differ.push(format!("{format}: the bytes differ from Metal's"));
        }
    }
    let colour = [0.25, 0.5, 0.75, 1.0];
    let targets = [(123, 16, "RGBA32Uint"), (10, 1, "R8Unorm")];
    let mut crossed: Vec<(u64, (u64, u64, &str))> = [26, 66, 116, 553, 555]
        .into_iter()
        .flat_map(|format| targets.map(|target| (format, target)))
        .collect();
    crossed.push((300, (70, 4, "RGBA8Unorm")));
    for (format, (target, bpp, name)) in crossed {
        let fill = Fill::Color(0, 0, [0, 0, 0], [1, 1, 1], colour, format);
        let case = read_case((target, bpp, [1, 1]), (2, [1, 1, 1], 1, 1), vec![fill]);
        *crate::answer::LAST_LINE.lock() = None;
        let into = format!("{format} into {name}");
        match catch_death(|| fills_on::<VulkanPlatform>(&dev(), &vulkan_desc, &case)) {
            Err(site) if site.file == "fillTexture:level:slice:region:color:pixelFormat:" => {
                let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
                if !line.contains("into a texture of") {
                    differ.push(format!("{into}: died otherwise: {line}"));
                }
            }
            Err(site) => differ.push(format!("{into}: died at {site:?}")),
            Ok(_) => differ.push(format!("{into}: taken")),
        }
    }
    assert!(
        differ.is_empty(),
        "{} differ, the first: {:#?}",
        differ.len(),
        &differ[..differ.len().min(30)]
    );
}

/// Fills of a compressed cube and cube array (`BC1_RGBA`, 8 by 8 texels a face), which the
/// runtime fills in every texture type, write Metal's bytes on this host: bytes at faces
/// past 6, which take the face modulo 6, at a slice past a cube array's, which writes
/// nothing, at a region of one block and of the whole level, and a colour
#[test]
fn compressed_cube_fills_are_metals() {
    let whole = ([0, 0, 0], [8, 8, 1]);
    let block = ([4, 4, 0], [4, 4, 1]);
    let fills = |slices: [u64; 4]| {
        vec![
            bytes_fill(0, slices[0], whole.0, whole.1, &[1, 2, 3, 4, 5, 6, 7, 8]),
            bytes_fill(0, slices[1], block.0, block.1, &[9, 10, 11]),
            bytes_fill(0, slices[2], whole.0, whole.1, &[12, 13, 14, 15]),
            color_fill((0, slices[3]), block, [0.25, 0.5, 0.75, 1.0], 70),
        ]
    };
    fills_are_metals(&[
        read_case((130, 8, [4, 4]), (5, [8, 8, 1], 1, 1), fills([0, 3, 7, 13])),
        read_case(
            (130, 8, [4, 4]),
            (6, [8, 8, 1], 1, 2),
            fills([7, 11, 12, 2]),
        ),
    ]);
}

/// A case of `format`, a texture of `texture_type` (a 2D texture 2, a 2D array 3, a cube 5,
/// a cube array 6) of `extent`, `levels` and `array` slices, of `bpp` bytes a block of
/// `block` texels, usage `ShaderRead`.
fn read_case(
    (format, bpp, block): (u64, u64, [u64; 2]),
    (texture_type, extent, levels, array): (u64, [u64; 3], u64, u64),
    fills: Vec<Fill>,
) -> Case {
    Case {
        format,
        bpp,
        block,
        aspects: &[],
        texture_type,
        extent,
        levels,
        array,
        usage: 1,
        allow_gpu_optimized_contents: None,
        shared: false,
        fills,
    }
}

/// A fill with bytes, at `level` and `slice`, of the region at `origin` of `size`.
fn bytes_fill(level: u64, slice: u64, origin: [u64; 3], size: [u64; 3], bytes: &[u8]) -> Fill {
    Fill::Bytes(level, slice, origin, size, bytes.to_vec())
}

/// A fill of `color` packed in `pixel_format`, at `level` and `slice`, of the region at
/// `origin` of `size`.
fn color_fill(
    (level, slice): (u64, u64),
    (origin, size): ([u64; 3], [u64; 3]),
    color: [f64; 4],
    pixel_format: u64,
) -> Fill {
    Fill::Color(level, slice, origin, size, color, pixel_format)
}

/// A texture of a two-plane XR10 format (550, 551) of its own memory, a 2D one and a 2D
/// array's slice, of a width a multiple of 4, is filled as Metal fills it: each texel's
/// colour word the pattern cut at 4 bytes, or repeated to them, read back by the blits,
/// which move the colour words; regions inside the level and past its edge, with bytes and
/// with a colour packed in `RGBA8Unorm`
#[test]
fn two_plane_xr10_fills_off_a_surface_are_metals() {
    let rgba = [1.0, 0.5, 0.25, 1.0];
    let two_plane = |format| (format, 4, [1, 1]);
    let (low, high) = ([0x21, 0x22, 0x23, 0x24], [0x31, 0x32, 0x33, 0x34]);
    fills_are_metals(&[
        read_case(
            two_plane(550),
            (2, [16, 3, 1], 1, 1),
            vec![
                bytes_fill(0, 0, [0, 0, 0], [16, 1, 1], &[1, 2, 3, 4, 5, 6, 7, 8]),
                bytes_fill(0, 0, [3, 1, 0], [5, 1, 1], &[9, 10, 11]),
                color_fill((0, 0), ([12, 2, 0], [8, 1, 1]), rgba, 70),
            ],
        ),
        read_case(
            two_plane(551),
            (2, [8, 2, 1], 1, 1),
            vec![bytes_fill(0, 0, [2, 0, 0], [4, 2, 1], &low)],
        ),
        read_case(
            two_plane(550),
            (3, [8, 2, 1], 1, 2),
            vec![bytes_fill(0, 1, [0, 1, 0], [8, 1, 1], &high)],
        ),
    ]);
}

/// A texture of a format whose fill this module does not measure is filled by the runtime's
/// fill as Metal fills it: a compressed format's blocks (BC1, BC7, ASTC 4x4, ETC2, EAC), each
/// the pattern cut at the block's bytes or repeated to them, the region counting blocks and
/// cut at the level's; a 4:2:2 format's texels, 2 bytes each; a stencil format's, 1 byte;
/// `Depth32Float_Stencil8`'s depth, 4 bytes, read back as its depth; at a level past the first
/// and a 2D array's slice, with bytes and with a colour packed in another format. On a device
/// that clamps a depth outside `[0, 1]` (MoltenVK), the `Depth32Float_Stencil8` texture's first
/// upload dies of `restricted-depth-upload` in strict mode, and its bytes by default are Metal's
/// with each depth clamped ([`fills_are_metals_under`])
#[test]
fn fills_the_runtime_writes_are_metals() {
    let rgba = [1.0, 0.5, 0.25, 1.0];
    let eight: Vec<u8> = (1..=8).collect();
    let sixteen: Vec<u8> = (0x41..=0x50).collect();
    fills_are_metals(&[
        read_case(
            (130, 8, [4, 4]),
            (2, [8, 8, 1], 2, 1),
            vec![
                bytes_fill(0, 0, [1, 0, 0], [1, 2, 1], &eight),
                bytes_fill(1, 0, [0, 0, 0], [1, 1, 1], &[0x61, 0x62, 0x63]),
                color_fill((0, 0), ([0, 1, 0], [4, 4, 1]), rgba, 70),
            ],
        ),
        read_case(
            (152, 16, [4, 4]),
            (3, [4, 4, 1], 1, 2),
            vec![
                bytes_fill(0, 1, [0, 0, 0], [1, 1, 1], &sixteen),
                color_fill((0, 0), ([0, 0, 0], [1, 1, 1]), rgba, 115),
            ],
        ),
        read_case(
            (204, 16, [4, 4]),
            (2, [8, 4, 1], 1, 1),
            vec![bytes_fill(0, 0, [1, 0, 0], [1, 1, 1], &[1, 2, 3, 4, 5])],
        ),
        read_case(
            (180, 8, [4, 4]),
            (2, [8, 8, 1], 1, 1),
            vec![bytes_fill(0, 0, [0, 1, 0], [2, 1, 1], &eight)],
        ),
        read_case(
            (170, 8, [4, 4]),
            (2, [4, 4, 1], 1, 1),
            vec![color_fill((0, 0), ([0, 0, 0], [1, 1, 1]), rgba, 10)],
        ),
        read_case(
            (240, 4, [2, 1]),
            (2, [8, 2, 1], 1, 1),
            vec![
                bytes_fill(0, 0, [1, 0, 0], [3, 1, 1], &[1, 2, 3]),
                bytes_fill(0, 0, [6, 1, 0], [4, 1, 1], &[0x71, 0x72]),
            ],
        ),
        read_case(
            (253, 1, [1, 1]),
            (2, [4, 4, 1], 1, 1),
            vec![
                bytes_fill(0, 0, [1, 1, 0], [2, 2, 1], &[0x5a, 0x5b]),
                color_fill((0, 0), ([0, 3, 0], [4, 1, 1]), rgba, 70),
            ],
        ),
        read_case(
            (260, 4, [1, 1]),
            (2, [4, 2, 1], 1, 1),
            vec![
                bytes_fill(0, 0, [1, 0, 0], [2, 2, 1], &[0, 0, 0x80, 0x3f]),
                color_fill((0, 0), ([0, 1, 0], [1, 1, 1]), [0.5; 4], 55),
            ],
        ),
    ]);
}

/// The aspects of `Depth32Float_Stencil8`: its depth (option 1), 4 bytes a texel, and its
/// stencil (option 2), 1 byte.
const DEPTH_AND_STENCIL: &[(u64, u64)] = &[(1, 4), (2, 1)];

/// A `Depth32Float_Stencil8` texture is filled as Metal fills it, its depth and its stencil
/// read back apart: each texel's depth the pattern cut at 4 bytes, its stencil kept, at a
/// level past the first and a 2D array's slice, with bytes and with a colour packed in
/// another format. On a device that clamps a depth outside `[0, 1]` (MoltenVK), each texture's
/// first depth upload dies of `restricted-depth-upload` in strict mode, and its bytes by default
/// are Metal's with each depth clamped, the stencil's as they are
/// ([`fills_are_metals_under`])
#[test]
fn depth32_stencil8_fills_keep_the_stencil_as_metals() {
    let both = |c: Case| Case {
        aspects: DEPTH_AND_STENCIL,
        ..c
    };
    fills_are_metals(&[
        both(read_case(
            (260, 4, [1, 1]),
            (2, [5, 3, 1], 2, 1),
            vec![
                bytes_fill(0, 0, [1, 0, 0], [2, 2, 1], &[0, 0, 0x80, 0x3f]),
                bytes_fill(1, 0, [0, 0, 0], [100, 100, 1], &[0x11, 0x22, 0x33]),
                color_fill((0, 0), ([0, 2, 0], [5, 1, 1]), [0.5; 4], 55),
            ],
        )),
        both(read_case(
            (260, 4, [1, 1]),
            (3, [4, 2, 1], 1, 2),
            vec![bytes_fill(0, 1, [0, 1, 0], [4, 1, 1], &[0, 0, 0, 0x3f])],
        )),
    ]);
}

/// An `X32_Stencil8` texture is filled as Metal fills it, its stencil written and read back
/// by stencil copies (option 2): each texel's byte the pattern's first, the region cut at
/// the level, with bytes and with a colour packed in another format, at a level past the
/// first and a 2D array's slice
#[test]
fn x32_stencil8_fills_are_metals() {
    let stencil = |c: Case| Case {
        aspects: &[(2, 1)],
        ..c
    };
    fills_are_metals(&[
        stencil(read_case(
            (261, 1, [1, 1]),
            (2, [5, 3, 1], 2, 1),
            vec![
                bytes_fill(0, 0, [1, 1, 0], [2, 2, 1], &[0x5a, 0x5b]),
                bytes_fill(0, 0, [3, 0, 0], [100, 1, 1], &[0x31]),
                bytes_fill(1, 0, [0, 0, 0], [2, 1, 1], &[0x41, 0x42, 0x43]),
                color_fill((0, 0), ([0, 2, 0], [5, 1, 1]), [1.0, 0.5, 0.25, 1.0], 70),
            ],
        )),
        stencil(read_case(
            (261, 1, [1, 1]),
            (3, [4, 2, 1], 1, 2),
            vec![bytes_fill(0, 1, [0, 1, 0], [4, 1, 1], &[0x77])],
        )),
    ]);
}

/// A `BGRG422` texture is filled as Metal fills it: each texel's 2 bytes the pattern cut at
/// them or repeated to them, the region counting texels and cut at the level, with bytes and
/// with a colour packed in another format
#[test]
fn bgrg422_fills_are_metals() {
    fills_are_metals(&[read_case(
        (241, 4, [2, 1]),
        (2, [10, 3, 1], 1, 1),
        vec![
            bytes_fill(0, 0, [1, 0, 0], [3, 1, 1], &[1, 2, 3]),
            bytes_fill(0, 0, [6, 1, 0], [40, 1, 1], &[0x71]),
            color_fill((0, 0), ([0, 2, 0], [4, 1, 1]), [1.0, 0.5, 0.25, 1.0], 70),
        ],
    )]);
}

/// The PVRTC fill cases, of each PVRTC format but the sRGB twins at `allowGPUOptimizedContents`
/// `allow`, shareable where `shared`, each `(format, block, side, x)`: a square texture of
/// `side`, its second fill's region from block `x` across, past the level's blocks.
fn pvrtc_fill_cases(
    (allow, shared): (Option<bool>, bool),
    formats: [(u64, [u64; 2], u64, u64); 4],
) -> Vec<Case> {
    let rgba = [1.0, 0.5, 0.25, 1.0];
    let twelve: Vec<u8> = (1..=12).collect();
    (formats.into_iter())
        .map(|(format, block, side, x)| Case {
            aspects: &[(4, 8)],
            allow_gpu_optimized_contents: allow,
            shared,
            ..read_case(
                (format, 8, block),
                (2, [side, side, 1], 1, 1),
                vec![
                    bytes_fill(0, 0, [1, 1, 0], [1, 2, 1], &[0x61, 0x62, 0x63]),
                    bytes_fill(0, 0, [x, 0, 0], [100, 1, 1], &twelve),
                    color_fill((0, 0), ([0, 3, 0], [1, 1, 1]), rgba, 70),
                ],
            )
        })
        .collect()
}

/// A shareable texture (`newSharedTextureWithDescriptor:`) of each PVRTC format but the sRGB
/// twins at the suite's shape (16x16, `Private`, `ShaderRead`, `allowGPUOptimizedContents`
/// off) is filled as Metal fills it: each block's 8 bytes the pattern cut at them or repeated
/// to them, the region counting blocks (8 by 4 texels at 2 bits a texel, 4 by 4 at 4) and cut
/// at the level's, with bytes and with a colour packed in another format. Its bytes are
/// written and read back row by row (`MTLBlitOptionRowLinearPVRTC`), as
/// `pvrtc_row_linear_copies_are_metals` measured those copies
#[test]
fn pvrtc_fills_are_metals() {
    fills_are_metals(&pvrtc_fill_cases(
        (Some(false), true),
        [
            (160, [8, 4], 16, 1),
            (162, [4, 4], 16, 2),
            (164, [8, 4], 16, 1),
            (166, [4, 4], 16, 2),
        ],
    ));
}

/// The PVRTC textures the fills were first measured with (32x32 at 2 bits a texel, 16x16 at
/// 4, `allowGPUOptimizedContents` on), and those of the suite's shape the fills are measured
/// with, are made at Metal's size, as [`pvrtc_texture_takes_metals_size`] says
#[test]
fn pvrtc_fill_textures_take_metals_size() {
    let suites = pvrtc_fill_cases(
        (Some(false), false),
        [
            (160, [8, 4], 16, 1),
            (162, [4, 4], 16, 2),
            (164, [8, 4], 16, 1),
            (166, [4, 4], 16, 2),
        ],
    );
    let cases = pvrtc_fill_cases(
        (None, false),
        [
            (160, [8, 4], 32, 2),
            (162, [4, 4], 16, 2),
            (164, [8, 4], 32, 2),
            (166, [4, 4], 16, 2),
        ],
    );
    for c in cases.iter().chain(&suites) {
        pvrtc_texture_takes_metals_size(&texture_of(c));
    }
}

/// A texture of the ASTC formats of HDR blocks (4 by 4 and 12 by 12) is filled as Metal fills
/// it: each block's 16 bytes the pattern cut at them or repeated to them, the region counting
/// blocks and cut at the level's, at a level past the first, with bytes and with a colour
/// packed in another format
#[test]
fn astc_hdr_fills_are_metals() {
    fills_are_metals(&astc_cases(&[(222, [4, 4]), (236, [12, 12])]));
}

/// A texture of the ASTC formats of blocks other than 4 by 4 (5 by 4 sRGB, 10 by 10 sRGB,
/// 8 by 8 and 12 by 12) is filled as Metal fills it, as
/// [`astc_hdr_fills_are_metals`] says
#[test]
fn astc_fills_of_other_blocks_are_metals() {
    fills_are_metals(&astc_cases(&[
        (187, [5, 4]),
        (198, [10, 10]),
        (212, [8, 8]),
        (218, [12, 12]),
    ]));
}

/// Each ASTC `(format, block)`'s cases: a texture of 4 by 3 blocks and two levels, filled in
/// a region of blocks, past its level's edge at level 1, and with a colour packed in
/// `RGBA32Float` in one block.
fn astc_cases(formats: &[(u64, [u64; 2])]) -> Vec<Case> {
    let sixteen: Vec<u8> = (0x41..=0x50).collect();
    formats
        .iter()
        .map(|&(format, block)| {
            read_case(
                (format, 16, block),
                (2, [4 * block[0], 3 * block[1], 1], 2, 1),
                vec![
                    bytes_fill(0, 0, [1, 1, 0], [2, 1, 1], &sixteen),
                    bytes_fill(1, 0, [0, 0, 0], [100, 100, 1], &[1, 2, 3, 4, 5]),
                    color_fill((0, 0), ([3, 2, 0], [1, 1, 1]), [1.0, 0.5, 0.25, 1.0], 125),
                ],
            )
        })
        .collect()
}

/// The bytes of a placement heap of storage `storage` (0 Shared, 2 Private) on `P`'s `d`
/// after `fills` into a texture of `format` (550 or 551) of `size`, of one level, at its
/// offset 0, in one blit encoder. A buffer placed over the texture's span and 4096 bytes past
/// it, each first its offset's low byte, reads the heap's bytes, the alpha plane's among
/// them, through a copy into a Shared buffer.
fn two_plane_heap_fill_on<P: crate::core::platform::Platform>(
    d: &P::Device,
    desc: &dyn Fn(&TextureDesc) -> P::TextureDescriptor,
    (storage, format, (w, h)): (u64, u64, (u64, u64)),
    fills: &[Fill],
) -> Vec<u8> {
    use crate::core::platform::resource::Heap as _;
    use crate::core::platform::values::{HeapDesc, OptionSet};
    let heap = d
        .new_heap(&HeapDesc {
            size: 0x10000,
            heap_type: 1,
            options: vec![
                OptionSet::StorageMode(storage),
                OptionSet::HazardTrackingMode(0),
            ],
            address_ranges: None,
            pinned_gpu_address: None,
        })
        .unwrap()
        .heap
        .unwrap();
    let into = desc(&TextureDesc {
        pixel_format: format,
        width: w,
        height: h,
        usage: 0x1,
        resource_options: Some(0x100 | storage << 4),
        ..TextureDesc::default()
    });
    let size = d.heap_texture_size_and_align(&into).unwrap().size;
    let t = heap.new_texture(&into, 0).unwrap().unwrap();
    let span = size + 4096;
    let over = heap
        .new_buffer_of_length(span, storage << 4, 0)
        .unwrap()
        .unwrap();
    let n = usize::try_from(span).unwrap();
    let shared = |bytes: &[u8]| {
        let b = d
            .new_buffer_of_length(OnFault::Die, span, 0)
            .unwrap()
            .unwrap();
        // SAFETY: a Shared buffer of `span` bytes, which no command uses yet.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), b.contents().as_ptr::<u8>(), n);
        }
        b
    };
    let q = d.new_command_queue().unwrap();
    let run = |cmds: Vec<BlitCmd<'_, P>>| {
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
        for cmd in cmds {
            e.encode(OnFault::Die, cmd).unwrap();
        }
        e.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        assert!(cb.error().is_none(), "{:?}", cb.error());
    };
    let copy = |from, to| BlitCmd::CopyBufferToBuffer {
        source_buffer: Some(from),
        source_offset: 0,
        destination_buffer: Some(to),
        destination_offset: 0,
        size: span,
    };
    let offsets: Vec<u8> = (0..n).map(|i| (i & 0xff) as u8).collect();
    let first = shared(&offsets);
    run(vec![copy(&first, &over)]);
    run(fills.iter().map(|f| fill_of::<P>(&t, f)).collect());
    let back = shared(&vec![0; n]);
    run(vec![copy(&over, &back)]);
    // SAFETY: a Shared buffer's `span` bytes, read after the copy completed.
    unsafe { std::slice::from_raw_parts(back.contents().as_ptr::<u8>(), n) }.to_vec()
}

/// A fill of a placement heap's one-level 2D texture of a two-plane XR10 format (550, 551),
/// Shared or Private, writes the heap's bytes as Metal's fill does on this host, byte for
/// byte ([`two_plane_heap_fill_on`]): each texel's colour word at Metal's blit rows, which
/// overlap where the width is no multiple of 4, the alpha plane and the heap's other bytes
/// kept; with bytes and with a colour packed in `RGBA8Unorm`, in one encoder
#[test]
fn two_plane_fills_in_a_heap_are_metals() {
    let d = dev();
    let p = crate::platform::tests::platform();
    let ours_desc = |t: &TextureDesc| p.texture_descriptor(t);
    let fills = [
        bytes_fill(0, 0, [0, 0, 0], [100, 100, 1], &[1, 2, 3]),
        bytes_fill(
            0,
            0,
            [1, 1, 0],
            [3, 2, 1],
            &(0x41..=0x48).collect::<Vec<u8>>(),
        ),
        color_fill((0, 0), ([2, 3, 0], [2, 1, 1]), [1.0, 0.5, 0.25, 1.0], 70),
    ];
    for storage in [0, 2] {
        for format in [550, 551] {
            for size in [(16, 8), (6, 5)] {
                let case = (storage, format, size);
                let key = format!("{case:?}");
                let ours = catch_death(|| {
                    two_plane_heap_fill_on::<VulkanPlatform>(&d, &ours_desc, case, &fills)
                })
                .unwrap_or_else(|site| {
                    let line = crate::answer::LAST_LINE.lock().clone();
                    panic!(
                        "{case:?}: died at {} ({:?}): {line:?}",
                        site.file, site.kind
                    )
                });
                crate::metal_check_bytes!(key, &ours);
            }
        }
    }
}

/// A fill of a two-plane XR10 texture of its own memory, 16 texels wide, keeps its alpha
/// plane as Metal's does: a kernel writes its texels, a fill with bytes and one with a colour
/// packed in `RGBA8Unorm` write their colour words, and a copy of the whole level into a
/// heap's texture, which moves both planes as Metal's does
/// ([`two_plane_copies_into_a_heap_are_metals`]), reads the heap's bytes, the
/// alpha plane's among them ([`two_plane_copy_into_heap_on`])
///
/// The runtime binds a two-plane XR10 texture to its kernels as `RGBA16Float`, a format that may
/// hold a NaN, and announces `device-nan-encoding` there: the first dispatch of this platform's
/// side dies, naming the kind ([`crate::answer::DIES`]).
#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "MoltenVK: the runtime records device-pack-conversion, a death in strict mode; Mac-only test, off everywhere; ledger: MoltenVK strict deaths"
)]
fn two_plane_fills_off_a_surface_keep_the_alpha_as_metals() {
    let d = dev();
    let p = crate::platform::tests::platform();
    let ours_desc = |t: &TextureDesc| p.texture_descriptor(t);
    let fills = [
        bytes_fill(0, 0, [1, 1, 0], [5, 3, 1], &(1..=8).collect::<Vec<u8>>()),
        color_fill((0, 0), ([8, 0, 0], [8, 8, 1]), [1.0, 0.5, 0.25, 1.0], 70),
    ];
    let site = crate::answer::dies_of(metal2vulkan_runtime::Degradation::DeviceNanEncoding, || {
        two_plane_copy_into_heap_on::<VulkanPlatform>(&d, &ours_desc, 0, None, &fills)
    });
    assert_eq!(site.file, "dispatchThreads:threadsPerThreadgroup:");
}

/// fillBuffer:range:pattern4: writes what Metal writes: the pattern from the range's start,
/// cut at its end, nothing for an empty range, and across more than the staging buffer
#[test]
fn pattern_fills_are_metals() {
    use crate::core::platform::Platform;
    use crate::core::platform::resource::Buffer as _;
    fn on<P: Platform>(d: &P::Device) -> Vec<u8> {
        let length = 9 << 20;
        let b = d
            .new_buffer_of_length(OnFault::Return, length, 0)
            .unwrap()
            .unwrap();
        // SAFETY: a fresh Shared buffer of `length` bytes, written before any use.
        unsafe { std::ptr::write_bytes(b.contents().as_ptr::<u8>(), 0xee, length as usize) };
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
        let e = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
        let long = length - 120;
        for (at, n) in [
            (1, 6),
            (9, 8),
            (13, 1),
            (20, 0),
            (24, 12),
            (41, 7),
            (63, 33),
            (99, long),
        ] {
            let fill = BlitCmd::FillBufferPattern4 {
                buffer: &b,
                range: NsRange::new(at, n),
                pattern: 0x0403_0201,
            };
            e.encode(OnFault::Return, fill).unwrap();
        }
        e.encode(OnFault::Return, BlitCmd::EndEncoding).unwrap();
        cb.commit(OnFault::Return).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        // SAFETY: the buffer's bytes, read after the GPU's writes completed.
        unsafe { std::slice::from_raw_parts(b.contents().as_ptr::<u8>(), length as usize) }.to_vec()
    }
    let ours = on::<VulkanPlatform>(&dev());
    crate::metal_check_bytes!("the first 64 bytes", &ours[..64]);
    crate::metal_check_bytes!("the fills", &ours);
}

/// The runtime's record of Metal's `generateMipmapsForTexture:` on an Apple M4:
/// textures of `RGB10A2Unorm` and `BGR10A2Unorm` whose first levels were designed, footprint by
/// footprint, so that the 2-bit alpha's weighted sums fall near the halves of its grid, and
/// every texel of their first computed levels where the 2-bit grid's code and the code of the
/// 10-bit code the bits replicate to differ. A `chain` line names a format and a first level's
/// extent; each line below it is a texel of the first computed level: its position, the alpha
/// codes of its four taps (base 4: the first tap's, the next across, the next down, the next
/// across and down) and the code Metal stored. A copy of metal2vulkan's
/// `crates/metal2vulkan/src/bind/mipmap_filter/measured_2bit.txt`, so the test reads it wherever
/// the checkout lies.
const TWO_BIT_TIES: &str = include_str!("measured_2bit.txt");

/// `MTLPixelFormatRGB10A2Unorm` and `MTLPixelFormatBGR10A2Unorm`, whose alpha is a texel's top
/// two bits.
const TEN_BIT_FORMATS: [(&str, u64); 2] = [
    ("MTLPixelFormatRGB10A2Unorm", 90),
    ("MTLPixelFormatBGR10A2Unorm", 94),
];

/// A designed texel of a chain of [`TWO_BIT_TIES`]: its position in the first computed level,
/// its taps' alpha codes, and the code Metal stored.
type Tie = ([u64; 2], [u8; 4], u8);

/// The chains of [`TWO_BIT_TIES`]: each one's format, first level's extent, and texels.
fn two_bit_chains() -> Vec<(u64, [u64; 2], Vec<Tie>)> {
    let mut chains: Vec<(u64, [u64; 2], Vec<Tie>)> = Vec::new();
    for line in TWO_BIT_TIES.lines() {
        let words: Vec<&str> = line.split_whitespace().collect();
        match words.as_slice() {
            [] => {}
            [first, ..] if first.starts_with('#') => {}
            ["chain", name, extent] => {
                let (_, format) = TEN_BIT_FORMATS.iter().find(|(n, _)| n == name).unwrap();
                let (w, h) = extent.split_once('x').unwrap();
                chains.push((
                    *format,
                    [w.parse().unwrap(), h.parse().unwrap()],
                    Vec::new(),
                ));
            }
            [x, y, codes, stored] => {
                let codes = codes.as_bytes();
                assert_eq!(codes.len(), 4, "{line}");
                let tie = (
                    [x.parse().unwrap(), y.parse().unwrap()],
                    std::array::from_fn(|k| codes[k] - b'0'),
                    stored.parse().unwrap(),
                );
                chains.last_mut().unwrap().2.push(tie);
            }
            _ => panic!("a line of no form: {line}"),
        }
    }
    chains
}

/// The first of the two texels along an axis of `big` texels that texel `i` of the next level,
/// of `small` texels (a power of two), reads in Metal's filter, as the runtime places it: its
/// centre `(i + 1/2) big / small - 1/2`, floored, or the texel after where the centre's
/// fraction rounds to a whole 256th. Every operand is exact in `f32` at these sizes, as on the
/// GPU, whose reciprocal of a power of two is exact on the M3 Max and the M4 alike.
fn first_tap(i: u64, big: u64, small: u64) -> u64 {
    assert!(small.is_power_of_two(), "{small}");
    let (centre, den) = ((2 * i + 1) * big - small, 2 * small);
    let (tap, fraction) = (centre / den, centre % den);
    // The fraction in 256ths, rounded half up.
    let q = (2 * 256 * fraction + den) / (2 * den);
    if q == 256 { tap + 1 } else { tap }
}

/// The first level of `extent` of a chain of [`two_bit_chains`]: `noise(seed, ..)`, 4 bytes a
/// texel, but for each designed texel's four taps, whose alpha is the code it lists.
fn designed_first_level(extent: [u64; 2], ties: &[Tie], seed: u32) -> Vec<u8> {
    let [w, h] = extent;
    let mut texels = noise(seed, w * h * 4);
    let small = extent.map(|e| e / 2);
    for &([x, y], codes, _) in ties {
        let (x0, y0) = (first_tap(x, w, small[0]), first_tap(y, h, small[1]));
        for (k, &code) in codes.iter().enumerate() {
            let (dx, dy) = (u64::try_from(k % 2).unwrap(), u64::try_from(k / 2).unwrap());
            let at = ((y0 + dy).min(h - 1) * w + (x0 + dx).min(w - 1)) * 4 + 3;
            let byte = &mut texels[usize::try_from(at).unwrap()];
            *byte = (*byte & 0x3f) | (code << 6);
        }
    }
    texels
}

/// Each level of a Private 2D texture of `format` (4 bytes a texel, 8 for `BGRA10_XR` and
/// `BGRA10_XR_sRGB`), `extent` and `levels` levels, of `usage`, on platform `P`, its first
/// level `first`, read back once `generateMipmapsForTexture:` made the others.
fn mipmaps_on<P: crate::core::platform::Platform>(
    d: &P::Device,
    desc: &dyn Fn(&TextureDesc) -> P::TextureDescriptor,
    (format, extent, levels): (u64, [u64; 2], u64),
    usage: u64,
    first: &[u8],
) -> Vec<Vec<u8>> {
    let t = d
        .new_texture(
            OnFault::Return,
            &desc(&TextureDesc {
                pixel_format: format,
                width: extent[0],
                height: extent[1],
                mipmap_level_count: levels,
                usage,
                resource_options: Some(0x20),
                ..TextureDesc::default()
            }),
        )
        .unwrap()
        .unwrap();
    let buffer_of = |bytes: &[u8]| {
        let b = d
            .new_buffer_of_length(OnFault::Return, bytes.len() as u64, 0)
            .unwrap()
            .unwrap();
        // SAFETY: a fresh Shared buffer of `bytes.len()` bytes, written before any use.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), b.contents().as_ptr(), bytes.len())
        };
        b
    };
    let run = |cmds: Vec<BlitCmd<'_, P>>| {
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
        let e = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
        for cmd in cmds.into_iter().chain([BlitCmd::EndEncoding]) {
            e.encode(OnFault::Return, cmd).unwrap();
        }
        cb.commit(OnFault::Return).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
    };
    let extent_of = |level: u64| extent.map(|e| (e >> level).max(1));
    let texel = match format {
        552 | 553 => 8,
        _ => 4,
    };
    let up = buffer_of(first);
    run(vec![BlitCmd::CopyBufferToTexture {
        source_buffer: &up,
        source_offset: 0,
        source_bytes_per_row: extent[0] * texel,
        source_bytes_per_image: extent[0] * extent[1] * texel,
        source_size: MtlSize::new(extent[0], extent[1], 1),
        destination_texture: &t,
        destination_slice: 0,
        destination_level: 0,
        destination_origin: MtlOrigin::new(0, 0, 0),
    }]);
    run(vec![BlitCmd::GenerateMipmaps { texture: Some(&t) }]);
    let downs: Vec<_> = (0..levels)
        .map(|level| {
            let [w, h] = extent_of(level);
            buffer_of(&vec![0x55; usize::try_from(w * h * texel).unwrap()])
        })
        .collect();
    run((0..levels)
        .zip(&downs)
        .map(|(level, down)| {
            let [w, h] = extent_of(level);
            BlitCmd::CopyTextureToBuffer {
                source_texture: &t,
                source_slice: 0,
                source_level: level,
                source_origin: MtlOrigin::new(0, 0, 0),
                source_size: MtlSize::new(w, h, 1),
                destination_buffer: down,
                destination_offset: 0,
                destination_bytes_per_row: w * texel,
                destination_bytes_per_image: w * h * texel,
            }
        })
        .collect());
    downs
        .iter()
        .map(|b| {
            let length = usize::try_from(b.length()).unwrap();
            // SAFETY: a Shared buffer's bytes, read after the GPU's writes completed.
            unsafe { std::slice::from_raw_parts(b.contents().as_ptr::<u8>(), length) }.to_vec()
        })
        .collect()
}

/// `generateMipmapsForTexture:` of `RGB10A2Unorm` and `BGR10A2Unorm` textures writes Metal's
/// levels, byte for byte, the 2-bit alpha's at sums a hair from a half of its grid included,
/// and announces no `mipmap-tie-rounding`. The runtime filters
/// that alpha as the 10-bit code its bits replicate to, taken to that code's nearest 2-bit
/// code, and floors each filter tap's term of a 10-bit field to 2^-14 of a code before the sum,
/// where the exact sum missed the M3 Max's code by one at a field
/// four times whose sum lay a hair from a half. The textures are [`TWO_BIT_TIES`]' chains of
/// two levels, whose first levels are the runtime's designs over random colours, and at each
/// designed texel Metal and this platform store the code an M4 stored; and chains of every level
/// from 64 by 64 and 45 by 29 texels of random bytes, whose levels the runtime takes from the M3
/// Max's passes
#[test]
fn two_bit_alpha_mipmaps_are_metals() {
    let vulkan_desc = |t: &TextureDesc| crate::platform::tests::platform().texture_descriptor(t);
    let designed = two_bit_chains()
        .into_iter()
        .map(|(format, extent, ties)| ((format, extent, 2), ties));
    let whole = TEN_BIT_FORMATS.into_iter().flat_map(|(_, format)| {
        [([64, 64], 7), ([45, 29], 6)]
            .map(|(extent, levels)| ((format, extent, levels), Vec::new()))
    });
    let mut differ = Vec::new();
    let mut seed = 0;
    for (chain, ties) in designed.chain(whole) {
        seed += 1;
        let (format, extent, levels) = chain;
        let what = format!(
            "format {format} {}x{} of {levels} levels",
            extent[0], extent[1]
        );
        let first = designed_first_level(extent, &ties, seed);
        let ours =
            catch_death(|| mipmaps_on::<VulkanPlatform>(&dev(), &vulkan_desc, chain, 1, &first))
                .unwrap_or_else(|site| {
                    let line = crate::answer::LAST_LINE.lock().clone();
                    panic!("{what}: died at {} ({:?}): {line:?}", site.file, site.kind)
                });
        for (level, o) in ours.iter().enumerate() {
            if !crate::metal_matches_bytes!(format!("chain {seed}: {what} level {level}"), o) {
                differ.push(format!(
                    "{what} level {level}: the texels differ from Metal's"
                ));
            }
        }
        // The alpha this platform stored at the designed texels of the first computed level.
        let width = extent[0] / 2;
        let alpha = |level: &[u8], [x, y]: [u64; 2]| {
            level[usize::try_from((y * width + x) * 4 + 3).unwrap()] >> 6
        };
        let missed: Vec<_> = (ties.iter())
            .filter(|&&(at, _, stored)| alpha(&ours[1], at) != stored)
            .map(|&(at, codes, stored)| (at, codes, stored, alpha(&ours[1], at)))
            .collect();
        if !missed.is_empty() {
            differ.push(format!(
                "{what}: ours stored another alpha than the M4 at {} of {} designed texels, the \
                 first (position, taps, the M4's, stored): {:?}",
                missed.len(),
                ties.len(),
                missed[0]
            ));
        }
    }
    assert!(differ.is_empty(), "{differ:#?}");
}

/// What fails a test of `generateMipmapsForTexture:` of chains of each of `formats`, of 4
/// bytes a texel, whose levels the runtime takes from the Apple M3 Max's passes: a level
/// whose bytes are not Metal's on this host. The chains
/// are Private textures of random bytes, of every level: of `ShaderRead` usage, 256 by 256
/// (Maps' chain's shape), 2048 by 32 and 32 by 2048, which Metal keeps losslessly compressed
/// and whose passes take the M3 Max's step after a level one texel tall or wide, and the strips
/// 1024 by 12, 4096 by 1 and 1000 by 3, which Metal does not compress; and of `ShaderWrite`
/// usage too, 256 by 256, which Metal does not compress, whose passes the runtime knows for
/// every format
fn m3_max_chains(formats: &[u64]) -> Vec<String> {
    let vulkan_desc = |t: &TextureDesc| crate::platform::tests::platform().texture_descriptor(t);
    let chains = [
        ([256, 256], 9),
        ([2048, 32], 12),
        ([32, 2048], 12),
        ([1024, 12], 11),
        ([4096, 1], 13),
        ([1000, 3], 10),
    ];
    crate::answer::RELAYED.lock().clear();
    let mut differ = Vec::new();
    let mut seed = 0x6d33;
    let written = ([256, 256], 9, 3);
    let read = chains.map(|(extent, levels)| (extent, levels, 1));
    for &format in formats {
        for (extent, levels, usage) in read.into_iter().chain([written]) {
            seed += 1;
            let chain = (format, extent, levels);
            let what = format!(
                "format {format} {}x{} of {levels} levels, usage {usage}",
                extent[0], extent[1]
            );
            let first = noise(seed, extent[0] * extent[1] * 4);
            let ours = catch_death(|| {
                mipmaps_on::<VulkanPlatform>(&dev(), &vulkan_desc, chain, usage, &first)
            })
            .unwrap_or_else(|site| {
                let line = crate::answer::LAST_LINE.lock().clone();
                panic!("{what}: died at {} ({:?}): {line:?}", site.file, site.kind)
            });
            for (level, o) in ours.iter().enumerate() {
                if !crate::metal_matches_bytes!(format!("{what} level {level}"), o) {
                    differ.push(format!("{what} level {level}: the texels differ"));
                }
            }
        }
    }
    differ
}

/// `generateMipmapsForTexture:` of `BGRA8Unorm`, `RGBA8Unorm`, `RGB10A2Unorm` and `RG16Unorm`
/// chains writes Metal's levels on this host, byte for byte ([`m3_max_chains`])
#[test]
fn m3_max_mipmap_passes_are_metals() {
    let differ = m3_max_chains(&[80, 70, 90, 60]);
    assert!(differ.is_empty(), "{differ:#?}");
}

/// `generateMipmapsForTexture:` of `RG11B10Float` chains, past 65 texels among them, writes
/// Metal's levels on this host, byte for byte ([`m3_max_chains`]): the runtime takes them from the M3 Max's passes, its tail reading an
/// axis of one texel alone
#[test]
fn rg11b10_float_mipmaps_are_metals() {
    let differ = m3_max_chains(&[92]);
    assert!(differ.is_empty(), "{differ:#?}");
}

/// The 10-bit codes of one texel of `BGRA10_XR` (552: four 16-bit words, the code in each
/// word's top 10 bits) or `BGR10_XR` (554: three codes packed from bit 0 of a 32-bit word)
fn extended_range_codes(format: u64, texel: &[u8]) -> Vec<i32> {
    if format == 552 {
        (texel.chunks(2))
            .map(|w| i32::from(u16::from_le_bytes([w[0], w[1]]) >> 6))
            .collect()
    } else {
        let word = u32::from_le_bytes(texel.try_into().unwrap());
        (0..3)
            .map(|i| ((word >> (10 * i)) & 0x3ff) as i32)
            .collect()
    }
}

/// The texels of `first` as Metal's blits leave a `BGRA10_XR` texture it keeps compressed: each
/// 16-bit word's low 6 bits clear, and alpha's code within 384 to 894
fn canonical_extended_range(first: &mut [u8]) {
    for (i, w) in first.chunks_mut(2).enumerate() {
        let code = u16::from_le_bytes([w[0], w[1]]) >> 6;
        let code = if i % 4 == 3 {
            code.clamp(384, 894)
        } else {
            code
        };
        w.copy_from_slice(&(code << 6).to_le_bytes());
    }
}

/// `generateMipmapsForTexture:` of `BGRA10_XR` and `BGR10_XR` textures keeps Metal's first
/// level byte for byte, and writes each later level within two codes of Metal's in every
/// channel, announcing `DeviceFilteredMipmaps`: the runtime takes a level as the device's
/// linear filter of the one above, where Apple's GPUs round apart among themselves
/// (`answer.rs`). Measured on the M3 Max (macOS 27.0.1): a level filtered from Metal's own
/// level of even size is within one code, and two are reached where a level halves an odd
/// size (45 by 29 to 22 by 14) and where a level is filtered from one already a code apart.
/// The chains are Private textures of `ShaderRead` usage and random codes, of every level: 64
/// by 64, 45 by 29 and 256 by 256. A `BGRA10_XR` chain's codes are in Metal's encoding
/// ([`canonical_extended_range`]), so its first level is Metal's under either
/// `extended_range_blits` form
#[test]
fn extended_range_mipmaps_are_metals() {
    let vulkan_desc = |t: &TextureDesc| crate::platform::tests::platform().texture_descriptor(t);
    crate::answer::RELAYED.lock().clear();
    let mut differ = Vec::new();
    let mut seed = 0x7872;
    for (format, texel) in [(552, 8), (554, 4)] {
        for (extent, levels) in [([64, 64], 7), ([45, 29], 6), ([256, 256], 9)] {
            seed += 1;
            let chain = (format, extent, levels);
            let what = format!(
                "format {format} {}x{} of {levels} levels",
                extent[0], extent[1]
            );
            let mut first = noise(seed, extent[0] * extent[1] * texel);
            if format == 552 {
                canonical_extended_range(&mut first);
            }
            // Level 0 by its digest, compared exactly; the levels computed from it in full,
            // compared within 2 codes.
            let ours = catch_death(|| {
                mipmaps_on::<VulkanPlatform>(&dev(), &vulkan_desc, chain, 1, &first)
            })
            .unwrap_or_else(|site| {
                let line = crate::answer::LAST_LINE.lock().clone();
                panic!("{what}: died at {} ({:?}): {line:?}", site.file, site.kind)
            });
            let step = usize::try_from(texel).unwrap();
            if !crate::metal_matches_bytes!(format!("{what} level 0"), &ours[0]) {
                differ.push(format!("{what} level 0: the texels differ from Metal's"));
            }
            for (level, o) in ours.iter().enumerate().skip(1) {
                let m = crate::metal_full!(format!("{what} level {level}"));
                let m = &m.unwrap_or_else(|| panic!("{what} level {level}: no bytes recorded"));
                assert_eq!(o.len(), m.len(), "{what} level {level}");
                let far = (o.chunks(step).zip(m.chunks(step)))
                    .map(|(a, b)| {
                        let (a, b) = (
                            extended_range_codes(format, a),
                            extended_range_codes(format, b),
                        );
                        a.iter().zip(&b).map(|(x, y)| (x - y).abs()).max().unwrap()
                    })
                    .max()
                    .unwrap();
                if far > 2 {
                    differ.push(format!("{what} level {level}: {far} codes from Metal's"));
                }
            }
        }
    }
    assert!(differ.is_empty(), "{differ:#?}");
    let filtered = rt::Degradation::DeviceFilteredMipmaps;
    assert!(crate::answer::RELAYED.lock().contains(&filtered));
}

/// Each of `cases` read back as Metal's reads back ([`fills_on`]), Metal's bytes the same on a
/// second run. Every case runs, and the ones that differ or die here are named together.
fn each_fill_reads_back_as_metals(cases: &[Case]) {
    let vulkan_desc = |t: &TextureDesc| crate::platform::tests::platform().texture_descriptor(t);
    let mut differ = Vec::new();
    for (i, c) in cases.iter().enumerate() {
        let what = format!("case {i}, {} {:?}: {:?}", c.format, c.extent, c.fills);
        match catch_death(|| fills_on::<VulkanPlatform>(&dev(), &vulkan_desc, c)) {
            Ok(ours) if crate::metal_matches_bytes!(what, &ours) => {}
            Ok(_) => differ.push(format!("{what}: the bytes differ from Metal's")),
            Err(site) => {
                let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
                differ.push(format!("{what}: died at {}: {line}", site.file));
            }
        }
    }
    assert!(
        differ.is_empty(),
        "{} differ:\n{}",
        differ.len(),
        differ.join("\n")
    );
}

/// A fill of a two-plane XR10 texture (550, 551) of its own memory at a width no multiple of 4,
/// narrower than 32 texels, reads back as Metal's does ([`each_fill_reads_back_as_metals`]): 7
/// and 6 texels wide, whole and a region cut at the texture, with bytes and with a colour packed
/// in `RGBA8Unorm`. Measured on the M3 Max (macOS 27.0.1): Metal's fills and its blits between a
/// buffer and the texture both address texel (x, y)'s colour word at y * ⌊4 w / 16⌋ * 16 + 4 x,
/// rows that overlap, so a fill of the 7 by 3 texture's texels 2 to 6 of rows 1 and 2 reads back
/// changed at texel 6 of row 0 and at texels 0 and 1 of row 2 too. The runtime fills at those
/// addresses, and its blits between a buffer and the texture take them, the bytes where rows
/// meet included; at 32 texels wide or wider it announces a copy into such rows
/// ([`a_wide_two_plane_fill_relays_the_copy_into_its_rows`])
#[test]
fn two_plane_fills_at_widths_off_4_are_metals() {
    let rgba = [1.0, 0.5, 0.25, 1.0];
    let eight: Vec<u8> = (1..=8).collect();
    let case = |format, [w, h]: [u64; 2], fill| {
        read_case((format, 4, [1, 1]), (2, [w, h, 1], 1, 1), vec![fill])
    };
    each_fill_reads_back_as_metals(&[
        case(550, [7, 3], bytes_fill(0, 0, [0, 0, 0], [7, 3, 1], &eight)),
        case(
            550,
            [7, 3],
            bytes_fill(0, 0, [2, 1, 0], [9, 5, 1], &[9, 10, 11]),
        ),
        case(
            551,
            [7, 3],
            color_fill((0, 0), ([3, 1, 0], [20, 3, 1]), rgba, 70),
        ),
        case(550, [6, 2], bytes_fill(0, 0, [0, 0, 0], [6, 2, 1], &eight)),
    ]);
}

/// A fill of a two-plane XR10 texture (550, 551) of its own memory 32 texels wide or wider, at a
/// width no multiple of 4, whose copy in from a buffer ([`fills_on`]) meets rows at Metal's
/// addresses: the runtime keeps the earlier row's bytes there, where Metal's GPU writes such rows
/// in an order that varies from run to run, and announces it (`two-plane-buffer-blit`), which
/// dies here at the copy, naming it, and is relayed otherwise. Relayed, the fill reads back
/// Metal's bytes, as the copy writes one byte value everywhere, so no order shows: 33 by 5 of
/// 551, a colour packed in `RGBA8Unorm` over a region cut at the texture, and 37 by 3 of 550, a
/// region with bytes. Measured on the M3 Max (macOS 27.0.1); a texture 31 texels wide announces
/// nothing
#[test]
fn a_wide_two_plane_fill_relays_the_copy_into_its_rows() {
    const COPY_IN: &str = "copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:\
                           sourceSize:toTexture:destinationSlice:destinationLevel:\
                           destinationOrigin:";
    let vulkan_desc = |t: &TextureDesc| crate::platform::tests::platform().texture_descriptor(t);
    let rgba = [1.0, 0.5, 0.25, 1.0];
    let case = |format, [w, h]: [u64; 2], fill| {
        read_case((format, 4, [1, 1]), (2, [w, h, 1], 1, 1), vec![fill])
    };
    for c in [
        case(
            551,
            [33, 5],
            color_fill((0, 0), ([3, 1, 0], [20, 3, 1]), rgba, 70),
        ),
        case(
            550,
            [37, 3],
            bytes_fill(0, 0, [2, 1, 0], [9, 5, 1], &[9, 10, 11]),
        ),
    ] {
        let what = format!("{} {:?}: {:?}", c.format, c.extent, c.fills);
        let site = catch_death(|| fills_on::<VulkanPlatform>(&dev(), &vulkan_desc, &c));
        let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
        assert_eq!(site.map_err(|site| site.file), Err(COPY_IN), "{what}");
        let named = "(metal2vulkan-degrade-two-plane-buffer-blit)";
        assert!(line.contains(named), "{what}: {line}");
        let ours = crate::answer::relaxed(|| fills_on::<VulkanPlatform>(&dev(), &vulkan_desc, &c));
        let metals = crate::metal_matches_bytes!(what, &ours);
        assert!(metals, "{what}: the bytes differ from Metal's");
    }
    let relayed = crate::answer::relayed_counts();
    let kinds: Vec<_> = relayed.iter().map(|&(kind, _)| kind).collect();
    assert_eq!(kinds, [rt::Degradation::TwoPlaneBufferBlit]);
}

/// A fill of a placement heap's texture of a two-plane XR10 format (550) of more than one level
/// or slice writes the heap's bytes as Metal's does, byte for byte
/// ([`two_plane_heap_texture_fill_on`]), Metal's the same on a second run: of a 16 by 8 texture,
/// level 1 of 2 whole, level 2 of 3 whole and a region of level 1 of 3; level 1 of slice 1 of a
/// 2D array of 2 levels; and slices 1 and 0 of a 2D array of one level. Measured on the M3 Max
/// (macOS 27.0.1): Metal writes each texel (x, y) of the region, cut at the level's extent, of
/// slice s at the texture's first byte + s * 4 w0 h0 + y * 4 w0 + 4 x, w0 and h0 level 0's
/// extent, whatever the level: as level 0's colour words lie; it writes no byte of the alpha
/// plane. The arrays of one level are made and filled, announcing nothing. The runtime announces
/// each texture of more than one level as it is made (`heap-two-plane-layout`, which dies here, at
/// `newTextureWithDescriptor:offset:`, naming it): it keeps the first level's texels in the
/// heap's bytes and the later levels' apart from them. Relayed, its fill of a later level writes
/// Metal's bytes into the heap's
#[test]
fn two_plane_heap_fills_of_more_levels_or_slices_are_metals() {
    const PLACED: &str = "newTextureWithDescriptor:offset:";
    let d = dev();
    let p = crate::platform::tests::platform();
    let ours_desc = |t: &TextureDesc| p.texture_descriptor(t);
    let eight: Vec<u8> = (1..=8).collect();
    let mut differ = Vec::new();
    for ((texture_type, levels, array), fill) in [
        ((2, 2, 1), bytes_fill(1, 0, [0, 0, 0], [8, 4, 1], &eight)),
        ((2, 3, 1), bytes_fill(2, 0, [0, 0, 0], [4, 2, 1], &eight)),
        (
            (2, 3, 1),
            bytes_fill(1, 0, [1, 1, 0], [3, 2, 1], &[0x21, 0x22, 0x23, 0x24]),
        ),
        ((3, 2, 2), bytes_fill(1, 1, [0, 0, 0], [8, 4, 1], &eight)),
        ((3, 1, 2), bytes_fill(0, 1, [0, 0, 0], [16, 8, 1], &eight)),
        ((3, 1, 2), bytes_fill(0, 0, [0, 0, 0], [16, 8, 1], &eight)),
    ] {
        let shape = TextureDesc {
            texture_type,
            pixel_format: 550,
            width: 16,
            height: 8,
            mipmap_level_count: levels,
            array_length: array,
            usage: 0x1,
            resource_options: Some(0x100),
            ..TextureDesc::default()
        };
        let fills = [fill];
        let what = format!(
            "{texture_type} {levels} levels {array} slices: {:?}",
            fills[0]
        );
        let ours =
            || two_plane_heap_texture_fill_on::<VulkanPlatform>(&d, &ours_desc, &shape, &fills);
        *crate::answer::LAST_LINE.lock() = None;
        let strict = catch_death(ours);
        let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
        let ours = match (strict, levels > 1) {
            (Ok(ours), false) => ours,
            (Err(site), true)
                if site.file == PLACED
                    && line.contains("(metal2vulkan-degrade-heap-two-plane-layout)") =>
            {
                crate::answer::relaxed(ours)
            }
            (Ok(_), true) => {
                differ.push(format!("{what}: made, announcing nothing"));
                continue;
            }
            (Err(site), _) => {
                differ.push(format!("{what}: died at {}: {line}", site.file));
                continue;
            }
        };
        if !crate::metal_matches_bytes!(what, &ours) {
            differ.push(format!("{what}: the bytes differ from Metal's"));
        }
    }
    assert!(
        differ.is_empty(),
        "{} differ:\n{}",
        differ.len(),
        differ.join("\n")
    );
}

/// The bytes of a Shared placement heap holding a texture of `shape` at offset 0, each byte its
/// offset's low byte first, once `fills` filled the texture in one blit encoder, on platform
/// `P`: the texture's size and a page more.
fn two_plane_heap_texture_fill_on<P: crate::core::platform::Platform>(
    d: &P::Device,
    desc: &dyn Fn(&TextureDesc) -> P::TextureDescriptor,
    shape: &TextureDesc,
    fills: &[Fill],
) -> Vec<u8> {
    use crate::core::platform::resource::{Buffer as _, Heap as _};
    use crate::core::platform::values::{HeapDesc, OptionSet};
    let heap = d
        .new_heap(&HeapDesc {
            size: 0x20000,
            heap_type: 1,
            options: vec![OptionSet::StorageMode(0), OptionSet::HazardTrackingMode(0)],
            address_ranges: None,
            pinned_gpu_address: None,
        })
        .unwrap()
        .heap
        .unwrap();
    let into = desc(shape);
    let size = d.heap_texture_size_and_align(&into).unwrap().size;
    let t = heap.new_texture(&into, 0).unwrap().unwrap();
    let span = size + 4096;
    let over = heap.new_buffer_of_length(span, 0, 0).unwrap().unwrap();
    let n = usize::try_from(span).unwrap();
    // SAFETY: the heap's Shared buffer of `span` bytes, which no command uses yet.
    unsafe {
        let to = over.contents().as_ptr::<u8>();
        for i in 0..n {
            to.add(i).write((i & 0xff) as u8);
        }
    }
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let e = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
    for f in fills {
        e.encode(OnFault::Die, fill_of::<P>(&t, f)).unwrap();
    }
    e.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    assert!(cb.error().is_none(), "{:?}", cb.error());
    // SAFETY: the heap's Shared buffer's `span` bytes, read after the fills completed.
    unsafe { std::slice::from_raw_parts(over.contents().as_ptr::<u8>(), n) }.to_vec()
}

/// Depth24Unorm_Stencil8 (255) and X24_Stencil8 (262), which the runtime's fill refuses: Metal
/// makes no texture of either on this host, its descriptor validation aborting
/// ("MTLTextureDescriptor has invalid pixelFormat (255)", and (262)), measured on the M3 Max
/// (macOS 27.0.1), so no fill reaches one there; this platform's texture of either dies where
/// it is made too
#[test]
fn depth24_stencil8_textures_die_where_made_as_metals() {
    for format in [255, 262] {
        let d = dev();
        let p = crate::platform::tests::platform();
        let desc = p.texture_descriptor(&TextureDesc {
            pixel_format: format,
            width: 8,
            height: 8,
            usage: 5,
            resource_options: Some(0x20),
            ..TextureDesc::default()
        });
        let made = catch_death(|| d.new_texture(OnFault::Return, &desc));
        assert!(
            made.is_err(),
            "{format}: {:?}",
            made.map(|m| m.map(|t| t.is_some()))
        );
    }
}

/// The bytes of a Shared buffer of 512 bytes, or of 64 past the texture where it ends later,
/// byte `i` written `i * 37 % 251` first, holding a texture buffer of `format`, `width` texels
/// wide, at `offset`, once `fills` filled the texture in one blit encoder, on platform `P`.
fn texture_buffer_fill_on<P: crate::core::platform::Platform>(
    d: &P::Device,
    desc: &dyn Fn(&TextureDesc) -> P::TextureDescriptor,
    (format, bpp, width, offset): (u64, u64, u64, u64),
    fills: &[Fill],
) -> Vec<u8> {
    use crate::core::platform::resource::Buffer as _;
    let bytes = usize::try_from((offset + width * bpp + 64).max(512)).unwrap();
    let b = d
        .new_buffer_of_length(OnFault::Return, bytes as u64, 0)
        .unwrap()
        .unwrap();
    // SAFETY: a fresh Shared buffer of `bytes` bytes, which no command uses yet.
    unsafe {
        let to = b.contents().as_ptr::<u8>();
        for i in 0..bytes {
            to.add(i).write(u8::try_from(i * 37 % 251).unwrap());
        }
    }
    let desc = desc(&TextureDesc {
        texture_type: 9,
        pixel_format: format,
        width,
        height: 1,
        usage: 0x1,
        resource_options: Some(0),
        ..TextureDesc::default()
    });
    let t = b
        .new_texture(OnFault::Return, &desc, offset, width * bpp)
        .unwrap()
        .unwrap();
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
    let e = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
    for f in fills {
        e.encode(OnFault::Return, fill_of::<P>(&t, f)).unwrap();
    }
    e.encode(OnFault::Return, BlitCmd::EndEncoding).unwrap();
    cb.commit(OnFault::Return).unwrap();
    cb.wait_until_completed(OnFault::Return).unwrap();
    assert!(cb.error().is_none(), "{:?}", cb.error());
    // SAFETY: the buffer's `bytes` bytes, read after the fills completed.
    unsafe { std::slice::from_raw_parts(b.contents().as_ptr::<u8>(), bytes) }.to_vec()
}

/// A case of [`texture_buffer_fills_are_metals`]: a texture buffer's format, its bytes a texel
/// and its offset in the buffer, and the fills.
type TextureBufferCase = ((u64, u64, u64), Vec<Fill>);

/// [`texture_buffer_fills_of_width_are_metals`] of texture buffers 12 texels wide.
fn texture_buffer_fills_are_metals(cases: &[TextureBufferCase]) {
    texture_buffer_fills_of_width_are_metals(cases, 12);
}

/// Each of `cases` (a texture buffer of a format of `bpp` bytes a texel, `width` texels wide at
/// an offset, and its fills) writes the buffer's bytes as Metal's does, every byte compared
/// ([`texture_buffer_fill_on`]); Metal's the same on a second run. Every case runs, and those
/// that differ or die here are named together.
fn texture_buffer_fills_of_width_are_metals(cases: &[TextureBufferCase], width: u64) {
    let d = dev();
    let p = crate::platform::tests::platform();
    let ours_desc = |t: &TextureDesc| p.texture_descriptor(t);
    let mut differ = Vec::new();
    for (i, ((format, bpp, offset), fills)) in cases.iter().enumerate() {
        let what = format!("{format} at {offset}: {fills:?}");
        let what = what.get(..400).unwrap_or(&what);
        let key = format!("width {width}, case {i}");
        let shape = (*format, *bpp, width, *offset);
        match catch_death(|| texture_buffer_fill_on::<VulkanPlatform>(&d, &ours_desc, shape, fills))
        {
            Ok(ours) if crate::metal_matches_bytes!(key, &ours) => {}
            Ok(_) => differ.push(format!("{what}: the bytes differ from Metal's")),
            Err(site) => {
                let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
                differ.push(format!("{what}: died at {}: {line}", site.file));
            }
        }
    }
    assert!(
        differ.is_empty(),
        "{} differ:\n{}",
        differ.len(),
        differ.join("\n")
    );
}

/// A fill of a texture buffer over a buffer, of a pattern no longer than the texel, inside the
/// width ([`crate::gpu::fill::texture_buffer_writes`]), writes the buffer's bytes as Metal's,
/// every byte compared ([`texture_buffer_fills_are_metals`]): of `R8Unorm`, `RG8Unorm`,
/// `RGBA8Unorm`, `RGBA16Float` and `RGBA32Float` texture buffers 12 texels wide at offsets 0
/// and 64 (and 4 and 68 where a texel is 4 bytes or fewer); with a pattern of each power of two
/// of bytes up to the texel, and a colour packed in a format of no more bytes than the texel;
/// whole, inside, ending at the width, at an odd texel, and at a level, slice, row, height and
/// depth past 0 and 1, which Metal ignores, and of no texels across, which writes nothing.
/// Measured on the M3 Max (macOS 27.0.1): Metal writes the region's texels from its `x` at the
/// texture's offset, one row, the pattern repeated from the region's first byte
#[test]
fn texture_buffer_fills_within_a_texel_are_metals() {
    let colour = [0.25, -0.5, 1.25, 0.75];
    let pattern =
        |n: usize| -> Vec<u8> { (0..n).map(|i| 0xa1 + u8::try_from(i).unwrap()).collect() };
    let mut cases = Vec::new();
    for (format, bpp) in [(10, 1), (30, 2), (70, 4), (115, 8), (125, 16)] {
        let offsets: &[u64] = if bpp <= 4 { &[0, 4, 64, 68] } else { &[0, 64] };
        let units = (0..).map(|k| 1usize << k).take_while(|&u| u as u64 <= bpp);
        let colours = [(10, 1), (30, 2), (70, 4), (115, 8), (125, 16)]
            .into_iter()
            .filter(|&(_, b)| b <= bpp);
        let mut fills: Vec<Vec<Fill>> = Vec::new();
        for unit in units {
            let bytes = pattern(unit);
            fills.push(vec![bytes_fill(0, 0, [0, 0, 0], [12, 1, 1], &bytes)]);
            fills.push(vec![bytes_fill(0, 0, [3, 0, 0], [5, 1, 1], &bytes)]);
            fills.push(vec![bytes_fill(0, 0, [7, 0, 0], [5, 1, 1], &bytes)]);
            fills.push(vec![bytes_fill(3, 7, [2, 5, 4], [4, 3, 2], &bytes)]);
        }
        for (packed, _) in colours {
            fills.push(vec![color_fill(
                (1, 2),
                ([1, 4, 0], [6, 2, 1]),
                colour,
                packed,
            )]);
        }
        fills.push(vec![
            bytes_fill(0, 0, [0, 0, 0], [12, 1, 1], &pattern(1)),
            bytes_fill(0, 0, [5, 0, 0], [0, 1, 1], &pattern(1)),
            bytes_fill(0, 0, [12, 0, 0], [0, 1, 1], &pattern(1)),
            color_fill((0, 0), ([9, 0, 0], [2, 1, 1]), colour, 10),
        ]);
        for &offset in offsets {
            for f in &fills {
                cases.push(((format, bpp, offset), f.clone()));
            }
        }
    }
    texture_buffer_fills_are_metals(&cases);
}

/// A fill of a texture buffer whose pattern Metal runs across the texels' bytes, or past the
/// width, writes the buffer's bytes as Metal's ([`texture_buffer_fills_are_metals`]), as the
/// runtime's fill streams it ([`crate::gpu::fill::texture_buffer_writes`]): a pattern longer
/// than the texel, of 8 bytes into `R8Unorm` texels from an odd byte and into `RGBA8Unorm`
/// texels from a byte a multiple of 16, of 32 bytes into `RGBA8Unorm` and `RGBA32Float` texels
/// and of 256 into `RGBA32Float` texels from such a byte, and a colour packed in `RGBA16Float` into
/// `RGBA8Unorm` texels; a pattern of 3 and of 12 bytes, no power of two, into `R8Unorm` and
/// `RG8Unorm` texels, over no 16 bytes of the buffer whole; a region past the width and one
/// ending past it; and one of no rows, and of no images. Measured on the M3 Max (macOS 27.0.1),
/// each the same on a second run: Metal writes the pattern repeated across the region's bytes
/// from its first, not started again at each texel, past the width too, and one row of a
/// region of no rows or images; a pattern of `n` bytes, no power of two, writes its byte
/// `j & (n - 1)` at the region's byte `j`. In 16 bytes of the buffer a region covers whole, such
/// a pattern's fill reads a block of the pattern's whole copies, and bytes of Metal's stack
/// past them, which the runtime reads as zeros and announces (`texture-buffer-fill-leftover`,
/// one of Metal's answers, relayed): no case here reads those. Into texture buffers 4096 texels
/// wide, long patterns too: one of 4096 bytes across 16384 bytes of `RGBA8Unorm` texels, and one
/// of 32752 bytes, the most Metal fills ([`crate::gpu::fill::MOST_STREAMED_PATTERN_BYTES`]; a
/// longer one ends its process), across 65536 bytes of `RGBA32Float` texels, the pattern run
/// twice, from the texture's first texel and from its fourth at a level, slice and row past
/// 0, past the width to the buffer's end, and across the 4096 bytes of `R8Unorm` texels
#[test]
fn texture_buffer_fills_metal_streams_are_metals() {
    let eight: Vec<u8> = (0xa1..=0xa8).collect();
    let three = [0xb1, 0xb2, 0xb3];
    let twelve: Vec<u8> = (0xd1..=0xdc).collect();
    let thirty_two: Vec<u8> = (0xc0..0xe0).collect();
    let most: Vec<u8> = (0..=255).rev().collect();
    let colour = [0.25, -0.5, 1.25, 0.75];
    texture_buffer_fills_are_metals(&[
        (
            (10, 1, 0),
            vec![bytes_fill(0, 0, [1, 0, 0], [5, 1, 1], &three)],
        ),
        (
            (30, 2, 4),
            vec![bytes_fill(0, 0, [0, 0, 0], [3, 1, 1], &three)],
        ),
        (
            (10, 1, 4),
            vec![bytes_fill(0, 0, [1, 0, 0], [9, 1, 1], &twelve)],
        ),
        (
            (70, 4, 64),
            vec![bytes_fill(0, 0, [0, 0, 0], [12, 1, 1], &thirty_two)],
        ),
        (
            (125, 16, 0),
            vec![bytes_fill(0, 0, [0, 0, 0], [2, 1, 1], &thirty_two)],
        ),
        (
            (125, 16, 64),
            vec![bytes_fill(0, 0, [0, 0, 0], [12, 1, 1], &most)],
        ),
        (
            (10, 1, 0),
            vec![bytes_fill(0, 0, [0, 0, 0], [12, 1, 1], &eight)],
        ),
        (
            (10, 1, 4),
            vec![bytes_fill(0, 0, [3, 0, 0], [5, 1, 1], &eight)],
        ),
        (
            (70, 4, 0),
            vec![bytes_fill(0, 0, [0, 0, 0], [12, 1, 1], &eight)],
        ),
        (
            (70, 4, 64),
            vec![bytes_fill(0, 0, [4, 0, 0], [8, 1, 1], &eight)],
        ),
        (
            (70, 4, 0),
            vec![color_fill((0, 0), ([4, 0, 0], [6, 1, 1]), colour, 115)],
        ),
        (
            (10, 1, 0),
            vec![bytes_fill(0, 0, [12, 0, 0], [3, 1, 1], &eight[..1])],
        ),
        (
            (70, 4, 0),
            vec![bytes_fill(0, 0, [10, 0, 0], [5, 1, 1], &eight[..4])],
        ),
        (
            (10, 1, 0),
            vec![bytes_fill(0, 0, [1, 0, 0], [2, 0, 1], &eight[..1])],
        ),
        (
            (70, 4, 64),
            vec![bytes_fill(0, 0, [1, 0, 0], [2, 1, 0], &eight[..4])],
        ),
    ]);
    let long = |n: usize| -> Vec<u8> { (0..n).map(|i| (i * 7 + i / 251) as u8).collect() };
    let (four_k, most) = (
        long(4096),
        long(crate::gpu::fill::MOST_STREAMED_PATTERN_BYTES),
    );
    texture_buffer_fills_of_width_are_metals(
        &[
            (
                (70, 4, 0),
                vec![bytes_fill(0, 0, [0, 0, 0], [4096, 1, 1], &four_k)],
            ),
            (
                (125, 16, 64),
                vec![bytes_fill(0, 0, [0, 0, 0], [4096, 1, 1], &most)],
            ),
            (
                (125, 16, 64),
                vec![bytes_fill(2, 1, [3, 4, 0], [4096, 2, 1], &most)],
            ),
            (
                (10, 1, 16),
                vec![bytes_fill(0, 0, [0, 0, 0], [4096, 1, 1], &most)],
            ),
        ],
        4096,
    );
}

/// A multisample texture of [`multisample_fills_on`]: its format, samples a texel, bytes a
/// sample, and slices (0 for a 2D multisample texture, else a multisample array of them).
type MultisampleShape = (u64, u64, u64, u64);

/// The bytes of each slice of a 5 by 3 multisample texture of `shape` ([`MultisampleShape`];
/// Private, read by shaders and rendered to) on platform `P`: written from a buffer of `init`,
/// each slice's samples in turn, then `fills` in one blit encoder, then copied back into a
/// buffer, a pixel's samples consecutive, Metal's layout of a multisample copy.
fn multisample_fills_on<P: crate::core::platform::Platform>(
    d: &P::Device,
    desc: &dyn Fn(&TextureDesc) -> P::TextureDescriptor,
    (format, samples, bpp, slices): MultisampleShape,
    init: &[u8],
    fills: &[Fill],
) -> Vec<u8> {
    use crate::core::platform::resource::Buffer as _;
    let (width, height) = (5, 3);
    let t = d
        .new_texture(
            OnFault::Return,
            &desc(&TextureDesc {
                texture_type: if slices == 0 { 4 } else { 8 },
                pixel_format: format,
                width,
                height,
                array_length: slices.max(1),
                sample_count: samples,
                usage: 0x5,
                resource_options: Some(0x20),
                ..TextureDesc::default()
            }),
        )
        .unwrap()
        .unwrap();
    let row = width * samples * bpp;
    let image = row * height;
    let length = image * slices.max(1);
    assert_eq!(init.len() as u64, length);
    let [up, down] = [(); 2].map(|()| {
        d.new_buffer_of_length(OnFault::Return, length, 0)
            .unwrap()
            .unwrap()
    });
    // SAFETY: a Shared buffer of `length` bytes the GPU does not use yet.
    unsafe {
        std::ptr::copy_nonoverlapping(init.as_ptr(), up.contents().as_ptr::<u8>(), init.len())
    };
    let run = |cmds: Vec<BlitCmd<'_, P>>| {
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
        let e = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
        for cmd in cmds {
            e.encode(OnFault::Return, cmd).unwrap();
        }
        e.encode(OnFault::Return, BlitCmd::EndEncoding).unwrap();
        cb.commit(OnFault::Return).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        assert!(cb.error().is_none(), "{:?}", cb.error());
    };
    let (at, size) = (MtlOrigin::new(0, 0, 0), MtlSize::new(width, height, 1));
    run((0..slices.max(1))
        .map(|slice| BlitCmd::CopyBufferToTexture {
            source_buffer: &up,
            source_offset: slice * image,
            source_bytes_per_row: row,
            source_bytes_per_image: image,
            source_size: size,
            destination_texture: &t,
            destination_slice: slice,
            destination_level: 0,
            destination_origin: at,
        })
        .collect());
    run(fills.iter().map(|f| fill_of::<P>(&t, f)).collect());
    run((0..slices.max(1))
        .map(|slice| BlitCmd::CopyTextureToBuffer {
            source_texture: &t,
            source_slice: slice,
            source_level: 0,
            source_origin: at,
            source_size: size,
            destination_buffer: &down,
            destination_offset: slice * image,
            destination_bytes_per_row: row,
            destination_bytes_per_image: image,
        })
        .collect());
    let n = usize::try_from(length).unwrap();
    // SAFETY: a Shared buffer's `length` bytes, read after the GPU's writes completed.
    unsafe { std::slice::from_raw_parts(down.contents().as_ptr::<u8>(), n) }.to_vec()
}

/// A fill of a multisample texture writes its samples as Metal's does, every byte of every
/// sample compared ([`multisample_fills_on`]), Metal's the same on a second run: 2D multisample
/// textures of `R8Unorm`, `RG8Unorm`, `RGBA8Unorm`, `R32Uint`, `RGBA16Float`, `Depth32Float`
/// and `Stencil8` at 4 samples, and of `RGBA8Unorm` and `RGBA32Float` at 2 and 4; a multisample
/// array of 2 slices of `RGBA8Unorm` at 4 samples, at slice 1 and at a slice past them; with
/// patterns of 16 bytes and of 3, whole, at a region and at a slice a 2D texture ignores, and
/// with a colour packed in `RGBA8Unorm` and in `RGBA32Float` (in `R32Float`, of a depth in
/// `[0, 1]`, for `Depth32Float`, whose samples hold such depths before too). Measured on the M3
/// Max (macOS 27.0.1), as the runtime's probe of Metal found: a fill writes every sample of the
/// region's texels and no other byte; where a texel's samples hold 16 bytes or fewer, sample `s`
/// takes the pattern's bytes from `s` times the sample's, the pattern repeated, and past 16
/// each sample takes the pattern cut at the sample's bytes. On a device that clamps a depth
/// outside `[0, 1]` (MoltenVK), each `Depth32Float` texture's upload dies of
/// `restricted-depth-upload` in strict mode, and by default its depths, inside `[0, 1]`, are
/// written as they are, its bytes Metal's
#[test]
fn multisample_fills_are_metals() {
    use crate::gpu::device::tests::uploading_depth;
    let d = dev();
    let p = crate::platform::tests::platform();
    let ours_desc = |t: &TextureDesc| p.texture_descriptor(t);
    let sixteen: Vec<u8> = (0xa1..=0xb0).collect();
    let three = [0xc1, 0xc2, 0xc3];
    let depths: Vec<u8> = [0.25f32, 0.5, 0.75, 1.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let rgba = [0.25, 0.5, 0.75, 1.0];
    let colour_fills = |packed: [u64; 2]| {
        packed.map(|format| color_fill((0, 0), ([1, 1, 0], [3, 2, 1]), rgba, format))
    };
    let mut cases: Vec<(MultisampleShape, Vec<Fill>)> = Vec::new();
    for shape in [
        (10, 4, 1, 0),
        (30, 4, 2, 0),
        (70, 4, 4, 0),
        (70, 2, 4, 0),
        (53, 4, 4, 0),
        (115, 4, 8, 0),
        (125, 2, 16, 0),
        (125, 4, 16, 0),
        (253, 4, 1, 0),
    ] {
        cases.push((
            shape,
            vec![bytes_fill(0, 0, [0, 0, 0], [5, 3, 1], &sixteen)],
        ));
        cases.push((shape, vec![bytes_fill(0, 3, [1, 1, 0], [9, 9, 1], &three)]));
        if shape.0 != 253 {
            cases.push((shape, colour_fills([70, 125]).to_vec()));
        }
    }
    let depth = (252, 4, 4, 0);
    cases.push((depth, vec![bytes_fill(0, 0, [0, 0, 0], [5, 3, 1], &depths)]));
    cases.push((
        depth,
        vec![bytes_fill(0, 0, [2, 0, 0], [2, 3, 1], &depths[4..8])],
    ));
    cases.push((
        depth,
        vec![color_fill((0, 0), ([1, 1, 0], [3, 2, 1]), rgba, 55)],
    ));
    let array = (70, 4, 4, 2);
    cases.push((
        array,
        vec![bytes_fill(0, 1, [0, 0, 0], [5, 3, 1], &sixteen)],
    ));
    cases.push((
        array,
        vec![bytes_fill(0, 2, [0, 0, 0], [5, 3, 1], &sixteen)],
    ));
    let mut differ = Vec::new();
    for (i, (shape, fills)) in cases.iter().enumerate() {
        let (format, samples, bpp, slices) = *shape;
        let what = format!("case {i}, {shape:?}: {fills:?}");
        let n = 5 * 3 * samples * bpp * slices.max(1);
        let init: Vec<u8> = match format {
            252 => (0..n / 4)
                .flat_map(|k| ((k % 64) as f32 / 64.0).to_le_bytes())
                .collect(),
            _ => noise(0x6d5 + u32::try_from(i).unwrap(), n),
        };
        let key = format!("case {i}");
        let filled =
            || multisample_fills_on::<VulkanPlatform>(&d, &ours_desc, *shape, &init, fills);
        match catch_death(|| match format {
            252 => uploading_depth(&d, UPLOAD, filled),
            _ => filled(),
        }) {
            Ok(ours) if crate::metal_matches_bytes!(key, &ours) => {}
            Ok(_) => differ.push(format!("{what}: the bytes differ from Metal's")),
            Err(site) => {
                let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
                differ.push(format!("{what}: died at {}: {line}", site.file));
            }
        }
    }
    assert!(
        differ.is_empty(),
        "{} differ:\n{}",
        differ.len(),
        differ.join("\n")
    );
}
