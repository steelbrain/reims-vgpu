//! Writes through textures of the two-plane YCbCr family over a buffer that stands for a guest's
//! surface, as the reims-graphics device makes them
//! ([`crate::core::platform::resource::Buffer::new_texture_over_surface_plane`] with the surface's
//! facts): a kernel's and a fragment function's `write()`, a kernel's `write()` through an
//! argument buffer, a blit from a buffer and a pass. The runtime stores a write of a 4:4:4
//! two-plane format into its planes as Metal's GPU does, the `_sRGB` ones (523, 527) included,
//! whose stores and passes it encodes by Metal's rule (dgpu-oracle.md Q40), and a write through
//! an argument buffer stores or drops as the same write through a slot does.
use crate::core::platform::Platform as _;
use crate::core::platform::blit::{BlitCmd, BlitEncoder as _, ResourceRef};
use crate::core::platform::command::{CommandBuffer as _, CommandQueue as _};
use crate::core::platform::compute::{ComputeCmd, ComputeEncoder as _};
use crate::core::platform::descriptor::RenderPassDescriptor as _;
use crate::core::platform::device::GpuDevice as _;
use crate::core::platform::render::{RenderCmd, RenderEncoder as _};
use crate::core::platform::resource::{Buffer as _, Library as _, Texture as _};
use crate::core::platform::types::{ClearColor, MtlOrigin, MtlSize, NsRange};
use crate::core::platform::values::{
    ColorAttachmentSet, RenderPipelineDesc, SurfaceFacts, SurfacePlane, TextureDesc,
};
use crate::core::raise::OnFault;
use crate::core::testkit::capture::catch_death;

use super::measure::Shape;
use crate::gpu::command::VulkanCommandBuffer;
use crate::gpu::device::tests::dev;
use crate::gpu::resource::{VulkanBuffer, VulkanTexture};
use crate::platform::tests::platform;

/// The tests' functions of `ycbcr_probe.metal`: `write_texels`, `write_texels_fragment` and
/// `cover_vertex` among them.
const PROBE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ycbcr_probe.metallib"));

/// What a test does with the texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Op {
    /// A kernel's `write()` of 0 into each of its texels (`write_texels`).
    Kernel,
    /// A fragment function's `write()` of 0 into each of its texels (`write_texels_fragment`),
    /// drawn over a `BGRA8Unorm` target of its size.
    Fragment,
    /// A blit of all of it from a buffer of 8 bytes a texel.
    FromBuffer,
    /// A pass that clears it and stores it.
    Clear,
    /// A kernel's `write()` through an argument buffer that holds it, made resident for writing
    /// (`rgbx_write_held_float`, of 16 by 16 texels).
    Held,
}

/// `rgbx.metal`'s kernels: `rgbx_write_held_float` among them.
const RGBX: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rgbx.metallib"));

/// A buffer of a two-plane surface of `ostype` whose luma plane is `size`, laid out as
/// CoreVideo lays one out (rows of a multiple of 64 bytes, plane 1 at a multiple of 4096), filled
/// with a pattern, and where its plane 0 lies in it.
fn surface_buffer(ostype: &[u8; 4], size: (u64, u64)) -> (VulkanBuffer, SurfacePlane) {
    let shape = Shape::of(ostype);
    let (width, height) = size;
    let (chroma_width, chroma_height) = shape.chroma_size(size);
    let luma_row = shape.row_bytes(width).next_multiple_of(64);
    let chroma_row = shape.row_bytes(chroma_width * 2).next_multiple_of(64);
    let chroma_offset = (luma_row * height).next_multiple_of(4096);
    let length = (chroma_offset + chroma_row * chroma_height).next_multiple_of(16384);
    let buffer = (dev().new_buffer_of_length(OnFault::Die, length, 0))
        .unwrap()
        .unwrap();
    let pattern: Vec<u8> = (0..length).map(|i| (i * 37 % 251) as u8).collect();
    // SAFETY: a Shared buffer of `length` bytes, which no command uses yet.
    unsafe {
        let to = buffer.contents().as_ptr::<u8>();
        std::ptr::copy_nonoverlapping(pattern.as_ptr(), to, pattern.len());
    }
    let facts = SurfaceFacts {
        pixel_format: u32::from_be_bytes(*ostype),
        plane_count: 2,
        width,
        height,
        chroma_offset,
        chroma_bytes_per_row: chroma_row,
        ycbcr_matrix: None,
    };
    let plane = SurfacePlane {
        plane: 0,
        offset: 0,
        bytes_per_row: luma_row,
        planar: true,
        facts: Some(facts),
    };
    (buffer, plane)
}

/// The bytes of `buffer`, all of them.
fn bytes_of(buffer: &VulkanBuffer) -> Vec<u8> {
    let length = usize::try_from(buffer.length()).unwrap();
    // SAFETY: the buffer's Shared bytes, read once the work that wrote them completed.
    unsafe { std::slice::from_raw_parts(buffer.contents().as_ptr::<u8>(), length) }.to_vec()
}

/// A texture of `format` of `size` over plane 0 of `buffer` at `plane`, of `usage`.
fn texture(
    buffer: &VulkanBuffer,
    plane: SurfacePlane,
    format: u64,
    size: (u64, u64),
    usage: u64,
) -> VulkanTexture {
    let desc = platform().texture_descriptor(&TextureDesc {
        texture_type: 2,
        pixel_format: format,
        width: size.0,
        height: size.1,
        usage,
        resource_options: Some(0),
        ..TextureDesc::default()
    });
    (buffer.new_texture_over_surface_plane(OnFault::Die, &desc, plane))
        .unwrap()
        .unwrap_or_else(|| panic!("{format}: nil"))
}

/// Runs `op` on a texture of `format` of `size` over plane 0 of a buffer of a surface of
/// `ostype`, and answers the buffer's bytes before and after, once the work completed.
fn written(op: Op, format: u64, ostype: &[u8; 4], size: (u64, u64)) -> (Vec<u8>, Vec<u8>) {
    let (buffer, plane) = surface_buffer(ostype, size);
    let before = bytes_of(&buffer);
    let usage = match op {
        Op::Kernel | Op::Fragment | Op::Held => 0x3,
        Op::FromBuffer => 0x1,
        Op::Clear => 0x5,
    };
    let t = texture(&buffer, plane, format, size, usage);
    encoded(op, &t, size);
    (before, bytes_of(&buffer))
}

/// Runs `op` on `t`, of `size`, in a command buffer of its own, until it completed.
fn encoded(op: Op, t: &VulkanTexture, size: (u64, u64)) {
    let (p, d) = (platform(), dev());
    let q = d.new_command_queue().unwrap();
    let (w, h) = size;
    let all = MtlSize::new(w, h, 1);
    let run = |encode: &dyn Fn(&VulkanCommandBuffer)| {
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        encode(&cb);
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        assert_eq!(cb.error(), None, "{op:?}");
    };
    let library = d.new_library(PROBE).unwrap().unwrap().unwrap();
    let function = |name| library.new_function(OnFault::Die, name).unwrap().unwrap();
    match op {
        Op::Kernel => {
            let f = function("write_texels");
            let state = (d.new_compute_pipeline_state_for_function(Some(&f)))
                .unwrap()
                .unwrap();
            run(&|cb| {
                let e = cb.compute_encoder(OnFault::Die).unwrap().unwrap();
                for cmd in [
                    ComputeCmd::SetComputePipelineState {
                        state: Some(&state),
                    },
                    ComputeCmd::SetTexture {
                        texture: Some(t),
                        index: 0,
                    },
                    ComputeCmd::DispatchThreads {
                        threads_per_grid: all,
                        threads_per_threadgroup: MtlSize::new(8, 8, 1),
                    },
                    ComputeCmd::EndEncoding,
                ] {
                    e.encode(OnFault::Die, cmd).unwrap();
                }
            });
        }
        Op::Fragment => {
            let (vertex, fragment) = (function("cover_vertex"), function("write_texels_fragment"));
            let state = d
                .new_render_pipeline_state(&RenderPipelineDesc {
                    label: None,
                    vertex: &vertex,
                    fragment: &fragment,
                    color0_pixel_format: 80,
                    resource_index: None,
                })
                .unwrap()
                .unwrap();
            let target = p.texture_descriptor(&TextureDesc {
                texture_type: 2,
                pixel_format: 80,
                width: w,
                height: h,
                usage: 4,
                resource_options: Some(0x20),
                ..TextureDesc::default()
            });
            let target = d.new_texture(OnFault::Die, &target).unwrap().unwrap();
            let pass = p.render_pass_descriptor();
            let attachment = ColorAttachmentSet {
                texture: Some(Some(&target)),
                ..ColorAttachmentSet::default()
            };
            pass.set_color_attachment(OnFault::Die, 0, &attachment)
                .unwrap();
            run(&|cb| {
                let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
                for cmd in [
                    RenderCmd::SetRenderPipelineState {
                        pipeline_state: Some(&state),
                    },
                    RenderCmd::SetFragmentTexture {
                        texture: Some(t),
                        index: 0,
                    },
                    RenderCmd::DrawPrimitives {
                        primitive_type: 3,
                        vertex_start: 0,
                        vertex_count: 3,
                    },
                    RenderCmd::EndEncoding,
                ] {
                    e.encode(OnFault::Die, cmd).unwrap();
                }
            });
        }
        Op::FromBuffer => {
            let row = w * 8;
            let source = (d.new_buffer_of_length(OnFault::Die, row * h, 0))
                .unwrap()
                .unwrap();
            run(&|cb| {
                let e = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
                let cmd = BlitCmd::CopyBufferToTexture {
                    source_buffer: &source,
                    source_offset: 0,
                    source_bytes_per_row: row,
                    source_bytes_per_image: row * h,
                    source_size: all,
                    destination_texture: t,
                    destination_slice: 0,
                    destination_level: 0,
                    destination_origin: MtlOrigin::new(0, 0, 0),
                };
                e.encode(OnFault::Die, cmd).unwrap();
                e.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
            });
        }
        Op::Held => {
            let lib = d.new_library(RGBX).unwrap().unwrap().unwrap();
            let f = lib
                .new_function(OnFault::Die, "rgbx_write_held_float")
                .unwrap()
                .unwrap();
            let state = (d.new_compute_pipeline_state_for_function(Some(&f)))
                .unwrap()
                .unwrap();
            let values: Vec<u8> = (0..16 * 16 * 4)
                .flat_map(|i| (f32::from(u16::try_from(i % 97).unwrap()) / 97.0).to_le_bytes())
                .collect();
            let made = |bytes: &[u8]| {
                let b = (d.new_buffer_of_length(OnFault::Die, bytes.len() as u64, 0))
                    .unwrap()
                    .unwrap();
                // SAFETY: a Shared buffer of the bytes' length, which no command uses yet.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        bytes.as_ptr(),
                        b.contents().as_ptr(),
                        bytes.len(),
                    )
                };
                b
            };
            let v = made(&values);
            let held = made(&t.gpu_resource_id().to_le_bytes());
            let bound = [Some(&v), Some(&held)];
            let offsets = [0, 0];
            let resources = [Some(ResourceRef::Texture(t))];
            run(&|cb| {
                let e = cb.compute_encoder(OnFault::Die).unwrap().unwrap();
                for cmd in [
                    ComputeCmd::SetComputePipelineState {
                        state: Some(&state),
                    },
                    ComputeCmd::SetBuffers {
                        buffers: &bound,
                        offsets: &offsets,
                        range: NsRange::new(0, 2),
                    },
                    ComputeCmd::UseResources {
                        resources: &resources,
                        usage: 2,
                    },
                    ComputeCmd::DispatchThreads {
                        threads_per_grid: MtlSize::new(16, 16, 1),
                        threads_per_threadgroup: MtlSize::new(8, 8, 1),
                    },
                    ComputeCmd::EndEncoding,
                ] {
                    e.encode(OnFault::Die, cmd).unwrap();
                }
            });
        }
        Op::Clear => {
            let pass = p.render_pass_descriptor();
            let attachment = ColorAttachmentSet {
                texture: Some(Some(t)),
                load_action: Some(2),
                store_action: Some(1),
                clear_color: Some(ClearColor {
                    red: 0.25,
                    green: 0.5,
                    blue: 0.75,
                    alpha: 1.0,
                }),
                ..ColorAttachmentSet::default()
            };
            pass.set_color_attachment(OnFault::Die, 0, &attachment)
                .unwrap();
            run(&|cb| {
                let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
                e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
            });
        }
    }
}

/// The size of the textures the tests write through a slot or a pass, and blit.
const SIZE: (u64, u64) = (32, 16);

/// The size of the textures the tests write through an argument buffer, as
/// `rgbx_write_held_float` writes 16 by 16 texels.
const HELD: (u64, u64) = (16, 16);

/// What `f` came to: `Ok` of its answer where it completed, else where it died and the line its
/// death named.
fn outcome<R>(f: impl FnOnce() -> R) -> Result<R, (&'static str, String)> {
    *crate::answer::LAST_LINE.lock() = None;
    let got = catch_death(f);
    let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
    got.map_err(|site| (site.file, line))
}

/// Whether `op` on a texture of `format` of `size` over a surface of `ostype` completed and
/// changed the surface's bytes, or where it died and the line its death named.
fn changed(
    op: Op,
    format: u64,
    ostype: &[u8; 4],
    size: (u64, u64),
) -> Result<bool, (&'static str, String)> {
    outcome(|| written(op, format, ostype, size)).map(|(before, after)| before != after)
}

/// Every write into a texture of a 4:4:4 two-plane format over a guest's surface plane completes
/// and changes the planes, as Metal's GPU stores it: a kernel's and a fragment function's
/// `write()`, a blit from a buffer and a pass, for the formats whose writes the runtime stored
/// before (503, 507, 542, 545) and for their `_sRGB` twins (523, 527), which the runtime stores
/// by Metal's rule since its RUNTIME 2305.
#[test]
fn writes_over_surface_planes_are_stored() {
    for (format, ostype) in [
        (503, b"444v"),
        (523, b"444v"),
        (542, b"444v"),
        (507, b"x444"),
        (527, b"x444"),
        (545, b"x444"),
    ] {
        for op in [Op::Kernel, Op::Fragment, Op::FromBuffer, Op::Clear] {
            let got = changed(op, format, ostype, SIZE);
            assert!(matches!(got, Ok(true)), "{format} {op:?}: {got:?}");
        }
    }
}

/// A kernel's write through an argument buffer that holds a two-plane YCbCr texture over a
/// guest's surface plane, made resident for writing, completes and stores or drops as the same
/// write through a slot does (the runtime's SEMANTICS 2114): the 4:4:4 formats' writes (503, 507,
/// 542, 545 and their `_sRGB` twins 523, 527) change the planes, and the 4:2:0 500's, which
/// Metal's GPU drops on this host, leaves them as they were. A `BGRA8Unorm` texture over a buffer
/// written so completes, its bytes written.
///
/// A kernel that only writes 500's texture announces nothing, so strict too it completes, where
/// the device's own sampler Y′CbCr conversion reads 500's planes (Iris Xe) as where the runtime
/// reads them in place (the Mac): the runtime converts the planes of such a texture only for a
/// command that reads it (its RUNTIME 2312).
#[test]
fn writes_through_argument_buffers_of_two_plane_ycbcr_textures_are_the_slots() {
    for (format, ostype, stored) in [
        (500, b"420v", false),
        (503, b"444v", true),
        (523, b"444v", true),
        (542, b"444v", true),
        (507, b"x444", true),
        (527, b"x444", true),
        (545, b"x444", true),
    ] {
        let got = changed(Op::Held, format, ostype, HELD);
        assert_eq!(got, Ok(stored), "{format}");
    }
    let (w, h) = HELD;
    let row = 256;
    let buffer = (dev().new_buffer_of_length(OnFault::Die, row * h, 0))
        .unwrap()
        .unwrap();
    let desc = platform().texture_descriptor(&TextureDesc {
        texture_type: 2,
        pixel_format: 80,
        width: w,
        height: h,
        usage: 0x3,
        resource_options: Some(0),
        ..TextureDesc::default()
    });
    let t = buffer
        .new_texture(OnFault::Die, &desc, 0, row)
        .unwrap()
        .unwrap();
    let before = bytes_of(&buffer);
    encoded(Op::Held, &t, HELD);
    assert_ne!(
        bytes_of(&buffer),
        before,
        "a BGRA8Unorm texture over a buffer"
    );
}
