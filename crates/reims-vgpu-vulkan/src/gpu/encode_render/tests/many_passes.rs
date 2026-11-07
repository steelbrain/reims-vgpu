//! One command buffer of hundreds of passes over one colour target, as a WebGPU frame split at
//! every indexed draw renders: WebKit checks a draw's indices in a 1x1 render encoder of its own
//! before each `drawIndexed`, so the webgpu-samples `stencilMask` frame is about 700 render
//! encoders over its canvas's IOSurface beside a `Depth32Float_Stencil8` attachment. Metal
//! completes it (the AMD 5300M in 21.6 ms, the M3 Max in 52 ms); a runtime that holds a
//! whole-attachment scratch or staging buffer a pass until the command buffer completes runs an
//! RTX 5080 Laptop GPU out of device memory there, its `endEncoding` refused.
//!
//! The passes are the frame's, through this platform as the graphics lane drives it: a first
//! encoder clears the colour, depth and stencil, then each of [`PASSES`] passes loads the colour
//! and the stencil, clears the depth, stores all three and draws one rectangle of two triangles,
//! its vertex data bound from an upload buffer over host pages (the bundle's `setVertexBytes:`),
//! each after a 1x1 encoder of no attachment whose state does not rasterize. The colour target is
//! one the lane makes: an IOSurface plane over guest pages, its rows a multiple of 64 bytes and
//! not of 256, or of 256 (the runtime shadows a no-copy buffer's host pages whatever its rows,
//! and merges the passes back); or a texture over a `Private` buffer, rows a multiple of 256, as
//! over a surface's device cache (a linear image of the buffer's memory, rendered in place on
//! MoltenVK, and apart beside the depth attachment and merged back on NVIDIA's devices). The
//! runtime's device memory grows by no attachment's bytes a pass, and the target holds what the
//! draws left.

use reims_vgpu_core::host::HostPtr;
use reims_vgpu_core::num::{ToU64, ToUsize};
use reims_vgpu_core::platform::os::vm::HostVm as _;
use reims_vgpu_core::platform::resource::Library as _;
use reims_vgpu_core::platform::values::{
    ColorAttachmentDesc, DepthAttachmentSet, DepthStencilDesc, FullRenderPipelineDesc,
    LinkedFunctionsDesc, PassSet, StencilAttachmentSet, StencilDesc, SurfaceFacts, SurfacePlane,
};

use super::*;
use crate::gpu::device::VulkanDevice;
use crate::gpu::resource::{VulkanDepthStencilState, VulkanRenderPipeline};

/// The passes after the clear: the measured frame's 6 + 7 x 100.
const PASSES: usize = 706;

/// The passes of the command buffer run before the measured one, so the runtime's pools hold
/// what one pass takes before the count starts.
const WARM: usize = 8;

/// The targets' height in texels, and the side of a cell of the rectangles' grid.
const H: u64 = 512;
const CELL: u64 = 32;

/// `MTLPixelFormatDepth32Float_Stencil8`, `MTLResourceStorageModePrivate`,
/// `MTLLoadActionLoad`, `MTLPrimitiveTypeTriangle`, `MTLCompareFunctionAlways`.
const DEPTH32F_S8: u64 = 260;
const PRIVATE: u64 = 0x20;
const LOAD: u64 = 1;
const TRIANGLE: u64 = 3;
const ALWAYS: u64 = 7;

/// The bytes the upload buffer gives each bound value, as the bundle places its bytes.
const SLOT: u64 = 256;

/// The tests' state of no rasterization's vertex function (`unrasterized.metal`).
const UNRASTERIZED: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/unrasterized.metallib"));

/// Where a target's bytes lie.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Backed {
    /// An IOSurface's plane over a no-copy buffer of guest pages.
    GuestPages,
    /// A `Private` buffer, as a surface's device cache.
    Private,
}

/// Host pages of at least `len` bytes, freed when dropped: declared before the buffers over
/// them, so dropped after those.
struct Pages {
    at: HostPtr,
    len: u64,
}

impl Pages {
    fn new(len: u64) -> Pages {
        let len = len.next_multiple_of(crate::os::OsVm::new().page_size());
        let at = crate::os::allocate_aliasable(len).unwrap();
        Pages { at, len }
    }

    /// A no-copy buffer over the pages.
    fn buffer(&self, d: &VulkanDevice) -> VulkanBuffer {
        // SAFETY: the pages stay mapped until `self` drops, after every buffer over them.
        let bytes = unsafe { HostRange::new(self.at, self.len) };
        d.new_buffer_over_host_memory(OnFault::Die, bytes, 0, None)
            .unwrap()
            .unwrap()
    }

    /// Write `value` at `at` of the pages, which no command uses now.
    fn write(&mut self, at: u64, value: &[u8]) {
        assert!(at + value.len().to_u64() <= self.len);
        // SAFETY: inside the pages, which no command uses while the test writes them.
        let into = unsafe { self.at.as_ptr::<u8>().add(at.to_usize()) };
        // SAFETY: `value` is the test's own, apart from the pages.
        unsafe { std::ptr::copy_nonoverlapping(value.as_ptr(), into, value.len()) };
    }

    /// The pages' first `len` bytes, which no command writes now.
    fn read(&self, len: u64) -> Vec<u8> {
        assert!(len <= self.len);
        // SAFETY: inside the pages, which no command writes while the test reads them.
        unsafe { std::slice::from_raw_parts(self.at.as_ptr::<u8>(), len.to_usize()) }.to_vec()
    }
}

impl Drop for Pages {
    fn drop(&mut self) {
        assert!(
            crate::os::OsVm::new()
                .deallocate(self.at, self.len)
                .is_success()
        );
    }
}

/// `v` as a float: every value here is below 2^16.
fn float(v: u64) -> f32 {
    f32::from(u16::try_from(v).unwrap())
}

/// Pass `i`'s rectangle of a target `width` texels wide, in texels (`[x0, y0, x1, y1]`,
/// half-open): two by two cells of the grid, its corner at cell `i` (by rows of `H / CELL`),
/// cut to the target; each overlaps its neighbours.
fn rectangle(i: usize, width: u64) -> [u64; 4] {
    let cells = H / CELL;
    let i = u64::try_from(i).unwrap();
    let (x0, y0) = (i % cells * CELL, i / cells % cells * CELL);
    [x0, y0, (x0 + 2 * CELL).min(width), (y0 + 2 * CELL).min(H)]
}

/// Pass `i`'s opaque colour, its channels' bytes red, green, blue and alpha.
fn colour(i: usize) -> [u8; 4] {
    let i = u64::try_from(i).unwrap();
    let byte = |v: u64| u8::try_from(v % 256).unwrap();
    [byte(i * 37), byte(i * 91 + 5), byte(i * 13 + 7), 255]
}

/// Pass `i`'s vertex bytes: its rectangle's two triangles as six clip-space `float4`s (each edge
/// on a texel boundary, so the texels it covers are those its rectangle holds), then its colour
/// as a `float4`, which `render_vbuf` reads at instance 0.
fn vertex_bytes(i: usize, width: u64) -> (Vec<u8>, Vec<u8>) {
    let [x0, y0, x1, y1] = rectangle(i, width);
    let x = |v: u64| 2.0 * float(v) / float(width) - 1.0;
    let y = |v: u64| 1.0 - 2.0 * float(v) / float(H);
    let corners = [(x0, y0), (x1, y0), (x0, y1), (x1, y0), (x1, y1), (x0, y1)];
    let positions: Vec<f32> = (corners.iter())
        .flat_map(|&(a, b)| [x(a), y(b), 0.0, 1.0])
        .collect();
    let rgba: Vec<f32> = (colour(i).iter()).map(|&c| f32::from(c) / 255.0).collect();
    (bytes(&positions), bytes(&rgba))
}

/// The bytes a target of `width` texels a row, `row` bytes apart, holds after the passes `0..n`:
/// the clear's opaque black, then each pass's colour over its rectangle, as `BGRA8Unorm` lays it.
fn expected(width: u64, row: u64, n: usize) -> Vec<u8> {
    let mut want = vec![0u8; (row * H).to_usize()];
    for y in 0..H {
        for x in 0..width {
            let at = (y * row + x * 4).to_usize();
            want[at..at + 4].copy_from_slice(&[0, 0, 0, 255]);
        }
    }
    for i in 0..n {
        let [r, g, b, a] = colour(i);
        let [x0, y0, x1, y1] = rectangle(i, width);
        for y in y0..y1 {
            for x in x0..x1 {
                let at = (y * row + x * 4).to_usize();
                want[at..at + 4].copy_from_slice(&[b, g, r, a]);
            }
        }
    }
    want
}

/// What the passes use beside their target: the states, the depth and stencil texture, the
/// upload buffer over host pages holding every pass's vertex bytes (the caller's pages, which
/// outlive it), and the check encoders' buffer.
struct Frame {
    drawing: VulkanRenderPipeline,
    unrasterized: VulkanRenderPipeline,
    depth_stencil: VulkanDepthStencilState,
    depth: VulkanTexture,
    upload: VulkanBuffer,
    checked: VulkanBuffer,
}

impl Frame {
    fn new(p: &VulkanPlatform, d: &VulkanDevice, width: u64, upload_pages: &mut Pages) -> Frame {
        let (vertex, fragment) = (
            fixture::<VulkanPlatform>(d, "render_vbuf"),
            fixture::<VulkanPlatform>(d, "render_color"),
        );
        let library = d.new_library(UNRASTERIZED).unwrap().unwrap().unwrap();
        let void_vertex = library.new_function(OnFault::Die, "void_vertex").unwrap();
        let void_vertex = void_vertex.unwrap();
        let state = |vertex, fragment, colors, depth, rasterization_enabled| {
            d.new_render_pipeline_state_full(&FullRenderPipelineDesc {
                label: None,
                vertex,
                fragment,
                vertex_descriptor: None,
                colors,
                depth_pixel_format: depth,
                stencil_pixel_format: depth,
                raster_sample_count: 1,
                alpha_to_coverage: false,
                alpha_to_one: false,
                rasterization_enabled,
                input_primitive_topology: 0,
                max_vertex_amplification_count: 1,
                support_indirect_command_buffers: false,
                vertex_linked: LinkedFunctionsDesc::default(),
                fragment_linked: LinkedFunctionsDesc::default(),
            })
            .unwrap()
            .unwrap()
        };
        let color = ColorAttachmentDesc {
            pixel_format: BGRA8,
            ..ColorAttachmentDesc::default()
        };
        let drawing = state(
            &vertex,
            Some(&fragment),
            vec![(0, color)],
            DEPTH32F_S8,
            true,
        );
        let unrasterized = state(&void_vertex, None, Vec::new(), 0, false);
        let face = StencilDesc {
            stencil_compare_function: ALWAYS,
            stencil_failure_operation: 0,
            depth_failure_operation: 0,
            depth_stencil_pass_operation: 0,
            read_mask: 0xff,
            write_mask: 0xff,
        };
        let depth_stencil = d
            .new_depth_stencil_state(&DepthStencilDesc {
                depth_compare_function: Some(ALWAYS),
                depth_write_enabled: Some(true),
                front: Some(face),
                back: Some(face),
            })
            .unwrap()
            .unwrap();
        let desc = p.texture_descriptor(&TextureDesc {
            pixel_format: DEPTH32F_S8,
            width,
            height: H,
            usage: TARGET,
            resource_options: Some(PRIVATE),
            ..TextureDesc::default()
        });
        let depth = d.new_texture(OnFault::Die, &desc).unwrap().unwrap();
        for i in 0..PASSES {
            let (positions, rgba) = vertex_bytes(i, width);
            upload_pages.write(slot(2 * i), &positions);
            upload_pages.write(slot(2 * i + 1), &rgba);
        }
        let upload = upload_pages.buffer(d);
        let checked = buffer::<VulkanPlatform>(d, 16, 0, &[0; 16]);
        Frame {
            drawing,
            unrasterized,
            depth_stencil,
            depth,
            upload,
            checked,
        }
    }
}

/// Where the upload buffer holds bound value `n`.
fn slot(n: usize) -> u64 {
    u64::try_from(n).unwrap() * SLOT
}

/// One command buffer of the frame over `target`: the clear, then the passes `0..n`, each after
/// a check encoder; committed and completed without an error.
fn run(p: &VulkanPlatform, d: &VulkanDevice, frame: &Frame, target: &VulkanTexture, n: usize) {
    let pass = |first: bool| {
        let pass = p.render_pass_descriptor();
        let (load, depth_load) = match first {
            true => (CLEAR, CLEAR),
            false => (LOAD, CLEAR),
        };
        pass.set_color_attachment(
            OnFault::Die,
            0,
            &ColorAttachmentSet {
                texture: Some(Some(target)),
                load_action: Some(load),
                store_action: Some(STORE),
                clear_color: Some(ClearColor {
                    red: 0.0,
                    green: 0.0,
                    blue: 0.0,
                    alpha: 1.0,
                }),
                ..ColorAttachmentSet::default()
            },
        )
        .unwrap();
        pass.set_depth_attachment(&DepthAttachmentSet {
            texture: Some(Some(&frame.depth)),
            load_action: Some(depth_load),
            store_action: Some(STORE),
            clear_depth: Some(1.0),
            ..DepthAttachmentSet::default()
        });
        pass.set_stencil_attachment(&StencilAttachmentSet {
            texture: Some(Some(&frame.depth)),
            load_action: Some(load),
            store_action: Some(STORE),
            clear_stencil: Some(0),
            ..StencilAttachmentSet::default()
        });
        pass
    };
    let (clear, load) = (pass(true), pass(false));
    let check = p.render_pass_descriptor();
    check.set_pass(&PassSet {
        render_target_width: Some(1),
        render_target_height: Some(1),
        default_raster_sample_count: Some(1),
        ..PassSet::default()
    });
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let e = cb.render_encoder(OnFault::Die, &clear).unwrap().unwrap();
    e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
    for i in 0..n {
        let e = cb.render_encoder(OnFault::Die, &check).unwrap().unwrap();
        for cmd in [
            RenderCmd::SetRenderPipelineState {
                pipeline_state: Some(&frame.unrasterized),
            },
            RenderCmd::SetVertexBuffer {
                buffer: Some(&frame.checked),
                offset: 0,
                index: 0,
            },
            RenderCmd::DrawPrimitives {
                primitive_type: TRIANGLE,
                vertex_start: 0,
                vertex_count: 3,
            },
            RenderCmd::EndEncoding,
        ] {
            e.encode(OnFault::Die, cmd).unwrap();
        }
        let e = cb.render_encoder(OnFault::Die, &load).unwrap().unwrap();
        for cmd in [
            RenderCmd::SetRenderPipelineState {
                pipeline_state: Some(&frame.drawing),
            },
            RenderCmd::SetDepthStencilState {
                depth_stencil_state: Some(&frame.depth_stencil),
            },
            RenderCmd::SetVertexBuffer {
                buffer: Some(&frame.upload),
                offset: slot(2 * i),
                index: 0,
            },
            RenderCmd::SetVertexBuffer {
                buffer: Some(&frame.upload),
                offset: slot(2 * i + 1),
                index: 1,
            },
            RenderCmd::DrawPrimitives {
                primitive_type: TRIANGLE,
                vertex_start: 0,
                vertex_count: 6,
            },
            RenderCmd::EndEncoding,
        ] {
            e.encode(OnFault::Die, cmd).unwrap();
        }
    }
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    assert!(cb.error().is_none(), "{:?}", cb.error());
}

/// The first `len` bytes of the `Private` buffer `b`, through a blit.
fn private_bytes(d: &VulkanDevice, b: &VulkanBuffer, len: u64) -> Vec<u8> {
    let out = buffer::<VulkanPlatform>(d, len, 0, &[]);
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
    for cmd in [
        BlitCmd::CopyBufferToBuffer {
            source_buffer: Some(b),
            source_offset: 0,
            destination_buffer: Some(&out),
            destination_offset: 0,
            size: len,
        },
        BlitCmd::EndEncoding,
    ] {
        blit.encode(OnFault::Die, cmd).unwrap();
    }
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    assert!(cb.error().is_none(), "{:?}", cb.error());
    // SAFETY: a Shared buffer's bytes, read after the blit completed.
    unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), len.to_usize()) }.to_vec()
}

/// The frame over a target over `over`'s bytes, `width` texels a row, rows `4 * width` bytes
/// apart: a few passes first, then [`PASSES`] of them, whose command buffer grows the runtime's
/// device memory by less than a quarter of an attachment's bytes a pass (one a pass would hold
/// [`PASSES`] times the attachment), and leaves the target holding the draws.
fn many_passes(over: Backed, width: u64) {
    let (p, d) = (platform(), dev());
    let row = 4 * width;
    let len = row * H;
    let mut upload_pages = Pages::new(slot(2 * PASSES));
    let frame = Frame::new(&p, &d, width, &mut upload_pages);
    let desc = p.texture_descriptor(&TextureDesc {
        pixel_format: BGRA8,
        width,
        height: H,
        usage: TARGET | READ,
        resource_options: (over == Backed::Private).then_some(PRIVATE),
        ..TextureDesc::default()
    });
    let mut pages = Pages::new(len);
    pages.write(0, &vec![0xa5; len.to_usize()]);
    let buffer = match over {
        Backed::GuestPages => pages.buffer(&d),
        Backed::Private => (d.new_buffer_of_length(OnFault::Die, len, PRIVATE))
            .unwrap()
            .unwrap(),
    };
    let target = match over {
        Backed::GuestPages => {
            let facts = SurfaceFacts {
                pixel_format: u32::from_be_bytes(*b"BGRA"),
                plane_count: 0,
                width,
                height: H,
                chroma_offset: 0,
                chroma_bytes_per_row: 0,
                ycbcr_matrix: None,
            };
            let plane = SurfacePlane {
                plane: 0,
                offset: 0,
                bytes_per_row: row,
                planar: false,
                facts: Some(facts),
            };
            buffer.new_texture_over_surface_plane(OnFault::Die, &desc, plane)
        }
        Backed::Private => buffer.new_texture(OnFault::Die, &desc, 0, row),
    };
    let target = target.unwrap().unwrap();
    run(&p, &d, &frame, &target, WARM);
    let before = d.raw.work_counts();
    run(&p, &d, &frame, &target, PASSES);
    let after = d.raw.work_counts();
    let delta = |of: fn(&rt::WorkCounts) -> u64| of(&after) - of(&before);
    let grown = delta(|c| c.device_memory_allocated);
    println!(
        "MANY-PASSES {over:?} {width}x{H}, rows of {row}: device memory grew {grown} bytes ({} \
         attachments); linear: {} passes apart, merges {} recorded and {} taken over, scratches \
         {} made and {} reused; shadow: {} copies, {} texels kept, {} scratches reused",
        grown / len,
        delta(|c| c.linear_passes_through_copies),
        delta(|c| c.linear_merges_recorded),
        delta(|c| c.linear_merges_taken_over),
        delta(|c| c.linear_merge_scratches_made),
        delta(|c| c.linear_merge_scratches_reused),
        delta(|c| c.shadow_copies),
        delta(|c| c.shadow_texels_kept),
        delta(|c| c.shadow_merge_scratches_reused),
    );
    let passes = u64::try_from(PASSES).unwrap();
    assert!(
        grown < len * passes / 4,
        "{PASSES} passes over one {over:?} target grew the device's memory by {grown} bytes, {} \
         of its attachments",
        grown / len
    );
    let got = match over {
        Backed::GuestPages => pages.read(len),
        Backed::Private => private_bytes(&d, &buffer, len),
    };
    let want = expected(width, row, PASSES);
    let differ = (0..got.len()).filter(|&i| got[i] != want[i]).count();
    assert_eq!(
        differ,
        0,
        "the target's bytes differ from the draws', the first at {:?}",
        (0..got.len()).find(|&i| got[i] != want[i])
    );
    drop((target, buffer, frame));
}

/// Over an IOSurface plane over guest pages, rows of 1984 bytes, a multiple of the surface's 64
/// and not of 256, as the measured frame's canvas (rows of 4800).
#[test]
fn many_passes_over_a_surface_s_pages_hold_no_attachment_a_pass() {
    many_passes(Backed::GuestPages, 496);
}

/// Over an IOSurface plane over guest pages, rows of 2048 bytes, a multiple of 256, as the
/// measured frame at 1024 and 1216 texels a row.
#[test]
fn many_passes_over_a_surface_s_aligned_pages_hold_no_attachment_a_pass() {
    many_passes(Backed::GuestPages, 512);
}

/// Over a `Private` buffer, rows of 2048 bytes, as over a surface's device cache (stencilprobe's
/// `-T`): a linear image of the buffer's memory, rendered apart beside the depth attachment on
/// NVIDIA's devices.
#[test]
fn many_passes_over_a_private_buffer_hold_no_attachment_a_pass() {
    many_passes(Backed::Private, 512);
}

/// A few of the frame's passes over an IOSurface plane over guest pages at a width that is not a
/// multiple of 4, its rows padded to 64 bytes as a guest's IOSurface rows are: the draws land in
/// the rows, `row` bytes apart, and the padding past each row's texels keeps the bytes it held
#[test]
fn a_few_passes_over_padded_rows_at_an_odd_width_land_in_the_rows() {
    let (p, d) = (platform(), dev());
    let mut wrong = Vec::new();
    for width in [1831u64, 1001] {
        let row = (4 * width).next_multiple_of(64);
        let len = row * H;
        let mut upload_pages = Pages::new(slot(2 * PASSES));
        let frame = Frame::new(&p, &d, width, &mut upload_pages);
        let desc = p.texture_descriptor(&TextureDesc {
            pixel_format: BGRA8,
            width,
            height: H,
            usage: TARGET | READ,
            ..TextureDesc::default()
        });
        let mut pages = Pages::new(len);
        pages.write(0, &vec![0xa5; len.to_usize()]);
        let buffer = pages.buffer(&d);
        let facts = SurfaceFacts {
            pixel_format: u32::from_be_bytes(*b"BGRA"),
            plane_count: 0,
            width,
            height: H,
            chroma_offset: 0,
            chroma_bytes_per_row: 0,
            ycbcr_matrix: None,
        };
        let plane = SurfacePlane {
            plane: 0,
            offset: 0,
            bytes_per_row: row,
            planar: false,
            facts: Some(facts),
        };
        let target = buffer.new_texture_over_surface_plane(OnFault::Die, &desc, plane);
        let target = target.unwrap().unwrap();
        run(&p, &d, &frame, &target, WARM);
        let got = pages.read(len);
        let want = expected(width, row, WARM);
        let texels = (4 * width).to_usize();
        let differ: Vec<(usize, usize)> = (0..H.to_usize())
            .flat_map(|y| (0..texels).map(move |b| (b, y)))
            .filter(|&(b, y)| got[y * row.to_usize() + b] != want[y * row.to_usize() + b])
            .collect();
        let padding = (0..H.to_usize())
            .flat_map(|y| (texels..row.to_usize()).map(move |b| y * row.to_usize() + b))
            .filter(|&i| got[i] != 0xa5)
            .count();
        if !differ.is_empty() || padding != 0 {
            wrong.push(format!(
                "{width} wide, rows of {row}: {} texel bytes differ (the first at byte {:?}), \
                 {padding} padding bytes written",
                differ.len(),
                differ.first()
            ));
        }
        drop((target, buffer, frame));
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}
