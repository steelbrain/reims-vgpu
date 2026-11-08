//! Writes through a surface made over the guest's pages with `kIOSurfaceClientDirection` 2, as
//! core makes one the guest described read-only, on this crate's platform.
//!
//! Measured on Metal, one case per process ([`sweep`]): the GPU drops a fill of a buffer on the
//! surface, a copy into one from another buffer, a kernel's stores into one, plain or by an atomic
//! exchange, and a vertex or fragment function's, at every extent tried, 8 bytes to 4 MiB: the
//! command buffer completes, its own later commands see the write (what of it the GPU's caches
//! still hold: [`dropped`]), and neither later command buffers nor the guest's pages do. Its other
//! writes into the surface's memory (a blit into the texture or from a texture into the buffer, a
//! kernel's or a draw's through the texture, a pass's) are dropped so at an extent of 2 texels and
//! fault the command buffer (`kIOGPUCommandBufferCallbackErrorPageFault`) at 64 and more, and a
//! pass that stores what it cleared or drew faults it too. A blit's copy or colour fill of 1 to 8
//! texels each way is dropped at every origin tried, on rows of 256 bytes or more; of a row of 16
//! texels or more it faults at some origins. This crate's platform sends the stores into a buffer
//! on the surface as Metal does, and a blit's copy from a buffer or colour fill of 1 to 8 texels
//! each way into a `BGRA8Unorm` texture made on the surface whose rows hold 256 bytes or more,
//! and dies at the encode of every other write (`os::dropped_write`), never sending it.
//! `replaceRegion:` writes through the surface's base address: on a surface over one page that is
//! the guest's page, which the write reaches; on one over several, a read-only mapping of the
//! surface's own, and Metal's write ends the process with a bus error, where this crate's platform
//! dies at the call.

#![cfg(target_os = "macos")]
// Checks against Metal's answers, measured on macOS.
// The proof that a command buffer's result is well-formed walks the runtime's types for `Send` and
// `Sync`, deeper than rustc's default limit of 128, as in the library (`src/lib.rs`).
#![recursion_limit = "256"]
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    missing_docs,
    unsafe_code
)]

use reims_vgpu_core::host::{HostPtr, PhysicalMemoryRange};
use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder, ResourceRef};
use reims_vgpu_core::platform::command::{CommandBuffer, CommandQueue};
use reims_vgpu_core::platform::compute::{ComputeCmd, ComputeEncoder};
use reims_vgpu_core::platform::descriptor::RenderPassDescriptor;
use reims_vgpu_core::platform::device::GpuDevice;
use reims_vgpu_core::platform::os::surface::SurfaceSpec;
use reims_vgpu_core::platform::os::vm::HostVm;
use reims_vgpu_core::platform::render::{RenderCmd, RenderEncoder};
use reims_vgpu_core::platform::resource::{Buffer, Library, Texture};
use reims_vgpu_core::platform::types::{
    ClearColor, MtlOrigin, MtlRegion, MtlSize, NsRange, Swizzle,
};
use reims_vgpu_core::platform::values::{ColorAttachmentSet, RenderPipelineDesc, TextureDesc};
use reims_vgpu_core::platform::{HostRange, Platform};
use reims_vgpu_core::raise::OnFault;
use reims_vgpu_core::testkit::capture::catch_death;
use reims_vgpu_vulkan::VulkanPlatform;
use reims_vgpu_vulkan::gpu::resource::VulkanTexture;
use reims_vgpu_vulkan::os::OsVm;

/// `MTLPixelFormatBGRA8Unorm`; `MTLTextureUsageShaderRead | ShaderWrite`.
const BGRA8: u64 = 80;
const READ_WRITE: u64 = 3;

/// The sweep library (`src/gpu/rounding.metal`) and the kernel that reaches memory through an
/// address it loads (`src/gpu/reach.metal`).
const ROUNDING: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rounding.metallib"));
const REACH: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/reach.metallib"));

/// What writes the surface, over the extent of its case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Writer {
    /// Nothing.
    None,
    /// A blit's copy of `0xab` bytes into the texture.
    Copy,
    /// A blit's fill of the texture with red (`fillTexture:...color:pixelFormat:`).
    FillColor,
    /// `sweep_kernel` of sweep 1 (1.0 in every channel) into the texture.
    Kernel,
    /// A pass that clears to blue and stores, drawing nothing.
    Clear,
    /// A pass that clears to blue and stores nothing.
    ClearDontCare,
    /// A pass that loads, stores and draws the sweep over the whole target.
    Draw,
    /// A pass that loads, draws the sweep and stores nothing.
    DrawDontCare,
    /// A pass that loads and stores, drawing nothing.
    Load,
    /// A draw into a target of the extent's own whose fragment function, `write_fragment` of
    /// sweep 1, writes the texture.
    FragmentWrite,
    /// `replaceRegion:` of `0xcd` bytes into the texture.
    Replace,
    /// A blit fill of a buffer on the surface with `0xee`.
    BufferFill,
    /// [`Writer::BufferFill`], then a copy of 64 MiB between two buffers of the device's own.
    BufferFillThrash,
    /// A blit's copy of `0xab` bytes into a buffer on the surface.
    BufferCopy,
    /// A blit's copy of a texture of `0xab` bytes into a buffer on the surface.
    BufferFromTexture,
    /// The reach kernel's store of `0x11223344` into a buffer on the surface's first word.
    BufferKernel,
    /// `store_words` over the extent's texels: a word a thread into a buffer on the surface.
    BufferStores,
    /// [`Writer::BufferStores`] by an atomic exchange, `exchange_words`.
    BufferAtomics,
    /// A draw of the extent's texels as vertices, a line list whose vertex function,
    /// `store_vertex`, stores a word a vertex into a buffer on the surface.
    VertexStores,
    /// A draw into a target of the extent's own whose fragment function, `store_fragment`,
    /// stores a word a fragment into a buffer on the surface.
    FragmentStores,
}

/// Every writer, by its name.
const WRITERS: [Writer; 20] = [
    Writer::None,
    Writer::Copy,
    Writer::FillColor,
    Writer::Kernel,
    Writer::Clear,
    Writer::ClearDontCare,
    Writer::Draw,
    Writer::DrawDontCare,
    Writer::Load,
    Writer::FragmentWrite,
    Writer::Replace,
    Writer::BufferFill,
    Writer::BufferFillThrash,
    Writer::BufferCopy,
    Writer::BufferFromTexture,
    Writer::BufferKernel,
    Writer::BufferStores,
    Writer::BufferAtomics,
    Writer::VertexStores,
    Writer::FragmentStores,
];

/// How much of the written texels a read showed the write in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Shown {
    /// Every one, every other texel the guest's.
    All,
    /// None, every texel the guest's.
    None,
    /// Some: how many texels showed the write, and how many other bytes.
    Other(usize, usize),
}

/// What a case showed: the command buffer's status and error code, then how much of the write
/// the texture's texels showed copied out in the same command buffer after the write, in a
/// later one, and the guest's pages.
#[derive(Debug, PartialEq, Eq)]
struct Seen {
    status: u64,
    error: Option<i64>,
    same: Shown,
    later: Shown,
    page: Shown,
}

/// The guest's byte at `i`.
fn guest_byte(i: u64) -> u8 {
    (i * 3 + 3) as u8
}

/// The guest's pages, `length` bytes: byte `i` is [`guest_byte`].
fn guest_pages(length: u64) -> HostPtr {
    let at = OsVm::new().allocate_anywhere(length).unwrap();
    for i in 0..length {
        // SAFETY: the pages just allocated, readable and writable.
        unsafe { at.as_ptr::<u8>().add(i as usize).write(guest_byte(i)) };
    }
    at
}

fn shared<P: Platform>(d: &P::Device, length: u64, byte: u8) -> P::Buffer {
    let b = d
        .new_buffer_of_length(OnFault::Die, length, 0)
        .unwrap()
        .unwrap();
    // SAFETY: a Shared buffer of `length` bytes no command uses yet.
    unsafe { std::ptr::write_bytes(b.contents().as_ptr::<u8>(), byte, length as usize) };
    b
}

/// A copy of the whole texture out to a fresh buffer in `cb`.
fn copy_out<P: Platform>(d: &P::Device, cb: &P::CommandBuffer, t: &P::Texture) -> P::Buffer {
    let (w, h) = (t.width(), t.height());
    let out = shared::<P>(d, w * 4 * h, 0);
    let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
    blit.encode(
        OnFault::Die,
        BlitCmd::CopyTextureToBuffer {
            source_texture: t,
            source_slice: 0,
            source_level: 0,
            source_origin: MtlOrigin::new(0, 0, 0),
            source_size: MtlSize::new(w, h, 1),
            destination_buffer: &out,
            destination_offset: 0,
            destination_bytes_per_row: w * 4,
            destination_bytes_per_image: w * 4 * h,
        },
    )
    .unwrap();
    blit.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
    out
}

/// A case: the surface's size in texels, the texture's usage, the writer and the extent it
/// writes, texels from the first (a buffer writer's: that many texels' bytes from the buffer's
/// first), or from `origin` (a blit's copy or colour fill of the texture's).
#[derive(Debug, Clone, Copy)]
struct Case {
    size: (u64, u64),
    usage: u64,
    writer: Writer,
    extent: (u64, u64),
    origin: (u64, u64),
}

/// The texel `case`'s writer writes.
fn texel(case: Case) -> [u8; 4] {
    match case.writer {
        Writer::Copy | Writer::BufferCopy | Writer::BufferFromTexture => [0xab; 4],
        Writer::FillColor => [0x00, 0x00, 0xff, 0xff],
        Writer::Kernel | Writer::Draw | Writer::DrawDontCare | Writer::FragmentWrite => [0xff; 4],
        Writer::Clear | Writer::ClearDontCare => [0xff, 0x00, 0x00, 0xff],
        Writer::Replace => [0xcd; 4],
        Writer::BufferFill | Writer::BufferFillThrash => [0xee; 4],
        Writer::BufferKernel
        | Writer::BufferStores
        | Writer::BufferAtomics
        | Writer::VertexStores
        | Writer::FragmentStores => [0x44, 0x33, 0x22, 0x11],
        Writer::None | Writer::Load => [0; 4],
    }
}

/// Whether `case`'s writer writes the texel at `(x, y)`.
fn inside(case: Case, x: u64, y: u64) -> bool {
    let (width, _) = case.size;
    let (w, h) = case.extent;
    match case.writer {
        Writer::None | Writer::Load => false,
        Writer::Draw | Writer::DrawDontCare | Writer::Clear | Writer::ClearDontCare => true,
        Writer::BufferFill
        | Writer::BufferFillThrash
        | Writer::BufferCopy
        | Writer::BufferStores
        | Writer::BufferAtomics
        | Writer::VertexStores
        | Writer::FragmentStores => y * width + x < w * h,
        Writer::BufferKernel => y == 0 && x == 0,
        Writer::Copy | Writer::FillColor => {
            let (ox, oy) = case.origin;
            (ox..ox + w).contains(&x) && (oy..oy + h).contains(&y)
        }
        _ => x < w && y < h,
    }
}

/// How much of the write `bytes`, rows of the case's width in texels, show: [`Shown`].
fn shown(case: Case, bytes: &[u8]) -> Shown {
    let written = texel(case);
    let (width, height) = case.size;
    let (mut hit, mut total, mut other) = (0, 0, 0);
    for y in 0..height {
        for x in 0..width {
            let at = ((y * width + x) * 4) as usize;
            let guest: Vec<u8> = (0..4).map(|k| guest_byte(at as u64 + k)).collect();
            let got = &bytes[at..at + 4];
            if inside(case, x, y) {
                total += 1;
                if got == written {
                    hit += 1;
                } else if got != guest.as_slice() {
                    other += 1;
                }
            } else if got != guest.as_slice() {
                other += 1;
            }
        }
    }
    match (hit, other) {
        (0, 0) => Shown::None,
        (h, 0) if h == total => Shown::All,
        (h, o) => Shown::Other(h, o),
    }
}

/// A surface over the guest's pages with `kIOSurfaceClientDirection` 2, a texture on it, the
/// case's write, and what each read showed.
fn case<P: Platform>(p: &P, d: &P::Device, c: Case) -> Seen {
    let (width, height) = c.size;
    let row = width * 4;
    let page = OsVm::new().page_size();
    let alloc = (row * height).next_multiple_of(page);
    let pages = guest_pages(alloc);
    let spec = SurfaceSpec {
        pixel_format: u32::from_be_bytes(*b"BGRA"),
        alloc_size: alloc,
        width,
        height,
        bytes_per_element: 4,
        bytes_per_row: row,
        element_width: 1,
        element_height: 1,
        client_direction: Some(2),
        client_address_ranges: Some(
            (0..alloc / page)
                .map(|i| PhysicalMemoryRange {
                    physical_address: pages.addr() as u64 + i * page,
                    physical_length: page,
                })
                .collect(),
        ),
        ..SurfaceSpec::default()
    };
    let s = p.surfaces().create(&spec).unwrap();
    let desc = p.texture_descriptor(&TextureDesc {
        pixel_format: BGRA8,
        width,
        height,
        usage: c.usage,
        resource_options: Some(0),
        ..TextureDesc::default()
    });
    let t = d
        .new_texture_on_surface(&desc, Some(&s), 0)
        .unwrap()
        .unwrap();
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let (w, h) = c.extent;
    let src = shared::<P>(d, (row * height).max(w * h * 4), 0xab);
    let rounding = d.new_library(ROUNDING).unwrap().unwrap().unwrap();
    let function = |l: &P::Library, name| l.new_function(OnFault::Die, name).unwrap().unwrap();
    let kind = shared::<P>(d, 16, 0);
    // SAFETY: the buffer's first word, which no command uses yet.
    unsafe { kind.contents().as_ptr::<u32>().write(1) };
    let mut held: Vec<P::Buffer> = Vec::new();
    let blit = |cmd: BlitCmd<'_, P>| {
        let e = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
        e.encode(OnFault::Die, cmd).unwrap();
        e.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
    };
    let on_surface = || d.new_buffer_on_surface(Some(&s)).unwrap().unwrap();
    match c.writer {
        Writer::None => {}
        Writer::Copy => blit(BlitCmd::CopyBufferToTexture {
            source_buffer: &src,
            source_offset: 0,
            source_bytes_per_row: w * 4,
            source_bytes_per_image: w * 4 * h,
            source_size: MtlSize::new(w, h, 1),
            destination_texture: &t,
            destination_slice: 0,
            destination_level: 0,
            destination_origin: MtlOrigin::new(c.origin.0, c.origin.1, 0),
        }),
        Writer::FillColor => blit(BlitCmd::FillTextureColor {
            texture: &t,
            level: 0,
            slice: 0,
            region: MtlRegion::new(
                MtlOrigin::new(c.origin.0, c.origin.1, 0),
                MtlSize::new(w, h, 1),
            ),
            color: ClearColor {
                red: 1.0,
                green: 0.0,
                blue: 0.0,
                alpha: 1.0,
            },
            pixel_format: BGRA8,
        }),
        Writer::Kernel => {
            let f = function(&rounding, "sweep_kernel");
            let state = d
                .new_compute_pipeline_state_for_function(Some(&f))
                .unwrap()
                .unwrap();
            let e = cb.compute_encoder(OnFault::Die).unwrap().unwrap();
            for cmd in [
                ComputeCmd::SetComputePipelineState {
                    state: Some(&state),
                },
                ComputeCmd::SetTexture {
                    texture: Some(&t),
                    index: 0,
                },
                ComputeCmd::SetBuffers {
                    buffers: &[Some(&kind)],
                    offsets: &[0],
                    range: NsRange::new(0, 1),
                },
                ComputeCmd::DispatchThreads {
                    threads_per_grid: MtlSize::new(w, h, 1),
                    threads_per_threadgroup: MtlSize::new(w.min(16), h.min(16), 1),
                },
                ComputeCmd::EndEncoding,
            ] {
                e.encode(OnFault::Die, cmd).unwrap();
            }
        }
        Writer::Clear
        | Writer::ClearDontCare
        | Writer::Draw
        | Writer::DrawDontCare
        | Writer::Load => {
            let draw = matches!(c.writer, Writer::Draw | Writer::DrawDontCare);
            let load = if matches!(c.writer, Writer::Clear | Writer::ClearDontCare) {
                2
            } else {
                1
            };
            let store = if matches!(c.writer, Writer::ClearDontCare | Writer::DrawDontCare) {
                0
            } else {
                1
            };
            let pass = p.render_pass_descriptor();
            let attachment = ColorAttachmentSet {
                texture: Some(Some(&t)),
                load_action: Some(load),
                store_action: Some(store),
                clear_color: Some(ClearColor {
                    red: 0.0,
                    green: 0.0,
                    blue: 1.0,
                    alpha: 1.0,
                }),
                ..ColorAttachmentSet::default()
            };
            pass.set_color_attachment(OnFault::Die, 0, &attachment)
                .unwrap();
            let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
            if draw {
                let (v, f) = (
                    function(&rounding, "sweep_vertex"),
                    function(&rounding, "sweep_fragment"),
                );
                let state = d
                    .new_render_pipeline_state(&RenderPipelineDesc {
                        label: None,
                        vertex: &v,
                        fragment: &f,
                        color0_pixel_format: BGRA8,
                        resource_index: None,
                    })
                    .unwrap()
                    .unwrap();
                for cmd in [
                    RenderCmd::SetRenderPipelineState {
                        pipeline_state: Some(&state),
                    },
                    RenderCmd::SetFragmentBuffer {
                        buffer: Some(&kind),
                        offset: 0,
                        index: 0,
                    },
                    RenderCmd::DrawPrimitives {
                        primitive_type: 3,
                        vertex_start: 0,
                        vertex_count: 3,
                    },
                ] {
                    e.encode(OnFault::Die, cmd).unwrap();
                }
            }
            e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
        }
        Writer::FragmentWrite => {
            let desc = p.texture_descriptor(&TextureDesc {
                pixel_format: BGRA8,
                width: w,
                height: h,
                usage: 4,
                resource_options: Some(0),
                ..TextureDesc::default()
            });
            let target = d.new_texture(OnFault::Die, &desc).unwrap().unwrap();
            let pass = p.render_pass_descriptor();
            let attachment = ColorAttachmentSet {
                texture: Some(Some(&target)),
                load_action: Some(0),
                store_action: Some(0),
                ..ColorAttachmentSet::default()
            };
            pass.set_color_attachment(OnFault::Die, 0, &attachment)
                .unwrap();
            let (v, f) = (
                function(&rounding, "sweep_vertex"),
                function(&rounding, "write_fragment"),
            );
            let state = d
                .new_render_pipeline_state(&RenderPipelineDesc {
                    label: None,
                    vertex: &v,
                    fragment: &f,
                    color0_pixel_format: BGRA8,
                    resource_index: None,
                })
                .unwrap()
                .unwrap();
            let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
            for cmd in [
                RenderCmd::SetRenderPipelineState {
                    pipeline_state: Some(&state),
                },
                RenderCmd::SetFragmentBuffer {
                    buffer: Some(&kind),
                    offset: 0,
                    index: 0,
                },
                RenderCmd::SetFragmentTexture {
                    texture: Some(&t),
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
        }
        Writer::Replace => {
            let region = MtlRegion::new(MtlOrigin::new(0, 0, 0), MtlSize::new(w, h, 1));
            let bytes = vec![0xcd; (w * 4 * h) as usize];
            t.replace_region(region, 0, 0, HostRange::of_slice(&bytes), w * 4, 0)
                .unwrap();
        }
        Writer::BufferFill => {
            let b = on_surface();
            blit(BlitCmd::FillBuffer {
                buffer: &b,
                range: NsRange::new(0, w * h * 4),
                value: 0xee,
            });
            held.push(b);
        }
        Writer::BufferFillThrash => {
            let b = on_surface();
            let (from, to) = (shared::<P>(d, 64 << 20, 1), shared::<P>(d, 64 << 20, 2));
            let e = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
            for cmd in [
                BlitCmd::FillBuffer {
                    buffer: &b,
                    range: NsRange::new(0, w * h * 4),
                    value: 0xee,
                },
                BlitCmd::CopyBufferToBuffer {
                    source_buffer: Some(&from),
                    source_offset: 0,
                    destination_buffer: Some(&to),
                    destination_offset: 0,
                    size: 64 << 20,
                },
                BlitCmd::EndEncoding,
            ] {
                e.encode(OnFault::Die, cmd).unwrap();
            }
            held.extend([b, from, to]);
        }
        Writer::BufferCopy => {
            let b = on_surface();
            blit(BlitCmd::CopyBufferToBuffer {
                source_buffer: Some(&src),
                source_offset: 0,
                destination_buffer: Some(&b),
                destination_offset: 0,
                size: w * h * 4,
            });
            held.push(b);
        }
        Writer::BufferFromTexture => {
            let b = on_surface();
            let desc = p.texture_descriptor(&TextureDesc {
                pixel_format: BGRA8,
                width: w,
                height: h,
                usage: 1,
                ..TextureDesc::default()
            });
            let from = d.new_texture(OnFault::Die, &desc).unwrap().unwrap();
            let region = MtlRegion::new(MtlOrigin::new(0, 0, 0), MtlSize::new(w, h, 1));
            let bytes = vec![0xab; (w * 4 * h) as usize];
            from.replace_region(region, 0, 0, HostRange::of_slice(&bytes), w * 4, 0)
                .unwrap();
            blit(BlitCmd::CopyTextureToBuffer {
                source_texture: &from,
                source_slice: 0,
                source_level: 0,
                source_origin: MtlOrigin::new(0, 0, 0),
                source_size: MtlSize::new(w, h, 1),
                destination_buffer: &b,
                destination_offset: 0,
                destination_bytes_per_row: row,
                destination_bytes_per_image: row * h,
            });
            held.push(b);
        }
        Writer::BufferKernel => {
            let b = on_surface();
            let reach = d.new_library(REACH).unwrap().unwrap().unwrap();
            let f = function(&reach, "reach");
            let state = d
                .new_compute_pipeline_state_for_function(Some(&f))
                .unwrap()
                .unwrap();
            let word = shared::<P>(d, 16, 0);
            let addresses = shared::<P>(d, 16, 0);
            // SAFETY: the buffers' first words, which no command uses yet.
            unsafe {
                word.contents().as_ptr::<u32>().write(0x1122_3344);
                addresses
                    .contents()
                    .as_ptr::<u64>()
                    .write(word.gpu_address());
            }
            let e = cb.compute_encoder(OnFault::Die).unwrap().unwrap();
            for cmd in [
                ComputeCmd::SetComputePipelineState {
                    state: Some(&state),
                },
                ComputeCmd::SetBuffers {
                    buffers: &[Some(&addresses), Some(&b)],
                    offsets: &[0, 0],
                    range: NsRange::new(0, 2),
                },
                ComputeCmd::UseResources {
                    resources: &[Some(ResourceRef::Buffer(&word))],
                    usage: 3,
                },
                ComputeCmd::DispatchThreads {
                    threads_per_grid: MtlSize::new(1, 1, 1),
                    threads_per_threadgroup: MtlSize::new(1, 1, 1),
                },
                ComputeCmd::EndEncoding,
            ] {
                e.encode(OnFault::Die, cmd).unwrap();
            }
            held.extend([b, word, addresses]);
        }
        Writer::BufferStores | Writer::BufferAtomics => {
            let b = on_surface();
            let name = match c.writer {
                Writer::BufferStores => "store_words",
                _ => "exchange_words",
            };
            let f = function(&rounding, name);
            let state = d
                .new_compute_pipeline_state_for_function(Some(&f))
                .unwrap()
                .unwrap();
            let e = cb.compute_encoder(OnFault::Die).unwrap().unwrap();
            for cmd in [
                ComputeCmd::SetComputePipelineState {
                    state: Some(&state),
                },
                ComputeCmd::SetBuffers {
                    buffers: &[Some(&b)],
                    offsets: &[0],
                    range: NsRange::new(1, 1),
                },
                ComputeCmd::DispatchThreads {
                    threads_per_grid: MtlSize::new(w * h, 1, 1),
                    threads_per_threadgroup: MtlSize::new((w * h).min(64), 1, 1),
                },
                ComputeCmd::EndEncoding,
            ] {
                e.encode(OnFault::Die, cmd).unwrap();
            }
            held.push(b);
        }
        Writer::VertexStores | Writer::FragmentStores => {
            let b = on_surface();
            let vertices = c.writer == Writer::VertexStores;
            let (target_size, names) = match vertices {
                true => ((64, 16), ("store_vertex", "sweep_fragment")),
                false => ((w, h), ("sweep_vertex", "store_fragment")),
            };
            let desc = p.texture_descriptor(&TextureDesc {
                pixel_format: BGRA8,
                width: target_size.0,
                height: target_size.1,
                usage: 4,
                resource_options: Some(0x20),
                ..TextureDesc::default()
            });
            let target = d.new_texture(OnFault::Die, &desc).unwrap().unwrap();
            let pass = p.render_pass_descriptor();
            let attachment = ColorAttachmentSet {
                texture: Some(Some(&target)),
                load_action: Some(0),
                store_action: Some(0),
                ..ColorAttachmentSet::default()
            };
            pass.set_color_attachment(OnFault::Die, 0, &attachment)
                .unwrap();
            let (v, f) = (function(&rounding, names.0), function(&rounding, names.1));
            let state = d
                .new_render_pipeline_state(&RenderPipelineDesc {
                    label: None,
                    vertex: &v,
                    fragment: &f,
                    color0_pixel_format: BGRA8,
                    resource_index: None,
                })
                .unwrap()
                .unwrap();
            let width = shared::<P>(d, 16, 0);
            // SAFETY: the buffer's first word, which no command uses yet.
            unsafe { width.contents().as_ptr::<u32>().write(w as u32) };
            let (stage_buffer, primitive_type, vertex_count) = match vertices {
                true => (
                    RenderCmd::SetVertexBuffer {
                        buffer: Some(&b),
                        offset: 0,
                        index: 1,
                    },
                    1,
                    w * h,
                ),
                false => (
                    RenderCmd::SetFragmentBuffer {
                        buffer: Some(&b),
                        offset: 0,
                        index: 1,
                    },
                    3,
                    3,
                ),
            };
            let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
            for cmd in [
                RenderCmd::SetRenderPipelineState {
                    pipeline_state: Some(&state),
                },
                RenderCmd::SetFragmentBuffer {
                    buffer: Some(&kind),
                    offset: 0,
                    index: 0,
                },
                RenderCmd::SetFragmentBuffer {
                    buffer: Some(&width),
                    offset: 0,
                    index: 2,
                },
                stage_buffer,
                RenderCmd::DrawPrimitives {
                    primitive_type,
                    vertex_start: 0,
                    vertex_count,
                },
                RenderCmd::EndEncoding,
            ] {
                e.encode(OnFault::Die, cmd).unwrap();
            }
            held.extend([b, width]);
        }
    }
    let same = copy_out::<P>(d, &cb, &t);
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    let (status, error) = (cb.status().0, cb.error().map(|e| e.code));
    let cb2 = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let later = copy_out::<P>(d, &cb2, &t);
    cb2.commit(OnFault::Die).unwrap();
    cb2.wait_until_completed(OnFault::Die).unwrap();
    let length = (row * height) as usize;
    // SAFETY: Shared buffers of `length` bytes and the guest's pages, read after the GPU work.
    let bytes = |at: *const u8| unsafe { std::slice::from_raw_parts(at, length) }.to_vec();
    let seen = Seen {
        status,
        error,
        same: shown(c, &bytes(same.contents().as_ptr::<u8>())),
        later: shown(c, &bytes(later.contents().as_ptr::<u8>())),
        page: shown(c, &bytes(pages.as_ptr::<u8>())),
    };
    drop((held, t, s));
    seen
}

/// `n` by `m`, read from `name`'s value.
fn dims(name: &str) -> (u64, u64) {
    let value = std::env::var(name).unwrap();
    let (a, b) = value.split_once('x').unwrap();
    (a.parse().unwrap(), b.parse().unwrap())
}

/// One case, from the environment: `WRITER`, `SIZE`, `EXTENT` and `ORIGIN` (`n`x`m`, the origin
/// 0x0 where unset) and `USAGE`. A measurement, run one case per process.
#[test]
#[ignore = "a measurement: one case per process, from the environment"]
fn sweep() {
    let name = std::env::var("WRITER").unwrap();
    let writer = WRITERS
        .into_iter()
        .find(|w| format!("{w:?}") == name)
        .unwrap();
    let c = Case {
        size: dims("SIZE"),
        usage: std::env::var("USAGE").map_or(READ_WRITE, |u| u.parse().unwrap()),
        writer,
        extent: dims("EXTENT"),
        origin: std::env::var("ORIGIN").map_or((0, 0), |_| dims("ORIGIN")),
    };
    let p = VulkanPlatform::new();
    let d = VulkanPlatform::system_default_device().unwrap();
    println!("vulkan {:?}", case::<VulkanPlatform>(&p, &d, c));
}

/// What every case shows that writes nothing the guest sees: the command buffer completes, its
/// own later commands see the write, and neither later command buffers nor the guest's pages do.
const DROPPED: Seen = Seen {
    status: 4,
    error: None,
    same: Shown::All,
    later: Shown::None,
    page: Shown::None,
};

/// Whether `seen` is [`DROPPED`], but for what the command buffer's later commands read, which is
/// Metal's own variance: its GPU's caches hold the dropped write for them until they evict it, so
/// each texel they read is the write or, where it was evicted, the guest's bytes, never anything
/// else. Of a blit's or a kernel's, with nothing else on the GPU Metal showed all of the write, of
/// 8 bytes to 4 MiB; beside another test's GPU work, 1,039,376 of 1,048,576 texels of 4 MiB, and,
/// beside another test's 64 MiB copies, less than all of 4 KiB or less in one run of four; after 64
/// MiB of other work in its own encoder, none at every extent ([`evicted`]). Of a draw's stores, in
/// one process and with nothing else on the GPU, its later commands read all, part (170,312 and
/// 373,400 of 1,048,576 fragments' words) or none, case to case and run to run, a vertex function's
/// of 1,024 words all in one run and none in the next. This platform's writes reach Metal's GPU
/// through a read-only mapping too, and must fall inside what Metal showed: every texel the write
/// or the guest's.
fn dropped(seen: &Seen) -> bool {
    let same = matches!(seen.same, Shown::All | Shown::None | Shown::Other(_, 0));
    same && *seen
        == Seen {
            same: seen.same,
            ..DROPPED
        }
}

/// The writes the GPU drops at every extent, and no write: a fill of a buffer on the surface, a
/// copy into one from another buffer, a kernel's stores into one, plain or by an atomic exchange, a
/// vertex function's of a word a vertex and a fragment function's of a word a fragment, of 8 bytes,
/// 4 KiB and 4 MiB (and one word through a loaded address), and none.
fn dropped_everywhere<P: Platform>(p: &P, d: &P::Device) {
    let writes = [
        Writer::BufferFill,
        Writer::BufferCopy,
        Writer::BufferKernel,
        Writer::BufferStores,
        Writer::BufferAtomics,
        Writer::VertexStores,
        Writer::FragmentStores,
    ];
    for size in [(64, 16), (1024, 1024)] {
        for extent in [(2, 1), size] {
            for writer in writes {
                let c = Case {
                    size,
                    usage: READ_WRITE,
                    writer,
                    extent,
                    origin: (0, 0),
                };
                let seen = case(p, d, c);
                assert!(dropped(&seen), "{c:?}: {seen:?}");
            }
        }
    }
    let none = Case {
        size: (64, 16),
        usage: READ_WRITE,
        writer: Writer::None,
        extent: (0, 0),
        origin: (0, 0),
    };
    let nothing = Seen {
        same: Shown::None,
        ..DROPPED
    };
    assert_eq!(case(p, d, none), nothing);
}

#[test]
fn this_platform_drops_a_buffers_writes_at_every_extent() {
    let p = VulkanPlatform::new();
    dropped_everywhere(&p, &VulkanPlatform::system_default_device().unwrap());
}

/// A buffer fill on the surface of 8 bytes and 4 KiB, then 64 MiB of other GPU work in the same
/// encoder: the command buffer's later commands read the guest's bytes, the dropped write gone
/// from the GPU's caches, as later command buffers and the guest's pages do.
fn evicted<P: Platform>(p: &P, d: &P::Device) {
    for extent in [(2, 1), (64, 16)] {
        let c = Case {
            size: (64, 16),
            usage: READ_WRITE,
            writer: Writer::BufferFillThrash,
            extent,
            origin: (0, 0),
        };
        let nothing = Seen {
            same: Shown::None,
            ..DROPPED
        };
        assert_eq!(case(p, d, c), nothing, "{c:?}");
    }
}

#[test]
fn this_platform_reads_the_pages_once_other_gpu_work_evicts_a_dropped_write() {
    let p = VulkanPlatform::new();
    evicted(&p, &VulkanPlatform::system_default_device().unwrap());
}

/// `replaceRegion:` on a surface over one page, of 2 texels and of the whole surface: the
/// guest's page, and every read after, shows it.
fn replaced_on_one_page<P: Platform>(p: &P, d: &P::Device) {
    for extent in [(2, 1), (64, 16)] {
        let c = Case {
            size: (64, 16),
            usage: READ_WRITE,
            writer: Writer::Replace,
            extent,
            origin: (0, 0),
        };
        let written = Seen {
            status: 4,
            error: None,
            same: Shown::All,
            later: Shown::All,
            page: Shown::All,
        };
        assert_eq!(case(p, d, c), written, "{c:?}");
    }
}

#[test]
fn this_platform_replaces_into_the_guests_page_of_a_surface_over_one() {
    let p = VulkanPlatform::new();
    replaced_on_one_page(&p, &VulkanPlatform::system_default_device().unwrap());
}

/// `replaceRegion:` of 2 texels on a surface over one page at `kIOSurfaceOffset` 0x40: the
/// write reaches the page at the offset, through the surface's base address, which lies there,
/// and nothing else of the page changes.
fn replaced_at_an_offset<P: Platform>(p: &P, d: &P::Device) {
    let page = OsVm::new().page_size();
    let pages = guest_pages(page);
    let spec = SurfaceSpec {
        pixel_format: u32::from_be_bytes(*b"BGRA"),
        offset: 0x40,
        alloc_size: page,
        width: 16,
        height: 16,
        bytes_per_element: 4,
        bytes_per_row: 64,
        element_width: 1,
        element_height: 1,
        client_direction: Some(2),
        client_address_ranges: Some(vec![PhysicalMemoryRange {
            physical_address: pages.addr() as u64,
            physical_length: page,
        }]),
        ..SurfaceSpec::default()
    };
    let s = p.surfaces().create(&spec).unwrap();
    let desc = p.texture_descriptor(&TextureDesc {
        pixel_format: BGRA8,
        width: 16,
        height: 16,
        usage: READ_WRITE,
        resource_options: Some(0),
        ..TextureDesc::default()
    });
    let t = d
        .new_texture_on_surface(&desc, Some(&s), 0)
        .unwrap()
        .unwrap();
    let region = MtlRegion::new(MtlOrigin::new(0, 0, 0), MtlSize::new(2, 1, 1));
    let bytes = [0xcd; 8];
    t.replace_region(region, 0, 0, HostRange::of_slice(&bytes), 64, 0)
        .unwrap();
    // SAFETY: the guest's page, mapped until the process ends.
    let now = unsafe { std::slice::from_raw_parts(pages.as_ptr::<u8>(), page as usize) };
    let expected: Vec<u8> = (0..page)
        .map(|i| match i {
            0x40..0x48 => 0xcd,
            _ => guest_byte(i),
        })
        .collect();
    assert!(now == expected.as_slice());
    drop((t, s));
}

#[test]
fn this_platform_replaces_at_the_offset_into_the_guests_page() {
    let p = VulkanPlatform::new();
    replaced_at_an_offset(&p, &VulkanPlatform::system_default_device().unwrap());
}

/// Every other write into the surface's memory dies at its encode, at the selector that names
/// it, at the extent Metal drops (2 texels) as at the one it faults on (the whole surface); a
/// blit's copy or colour fill into the texture, which this platform sends at 1 to 8 texels each
/// way inside the level ([`this_platform_drops_small_blits_into_the_texture`]), dies past that
/// (9 and 16 texels of a row, the whole surface), past the level's last row (8 rows from row 10
/// of 16), which Metal drops ([`metal_drops_small_blits_into_the_texture`]), and on a surface of
/// rows of fewer than 256 bytes; a pass dies at its creation, whatever it loads, stores or draws;
/// `replaceRegion:` on a surface over several pages dies at the call.
#[test]
fn this_platform_dies_at_every_other_write() {
    let p = VulkanPlatform::new();
    let d = VulkanPlatform::system_default_device().unwrap();
    let pass = "renderCommandEncoderWithDescriptor:";
    let dispatch = "dispatchThreads:threadsPerThreadgroup:";
    let draw = "drawPrimitives:vertexStart:vertexCount:";
    let dies = [
        (
            Writer::Copy,
            "copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:\
             toTexture:destinationSlice:destinationLevel:destinationOrigin:",
        ),
        (
            Writer::FillColor,
            "fillTexture:level:slice:region:color:pixelFormat:",
        ),
        (Writer::Kernel, dispatch),
        (
            Writer::BufferFromTexture,
            "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:\
             destinationOffset:destinationBytesPerRow:destinationBytesPerImage:",
        ),
        (Writer::FragmentWrite, draw),
        (Writer::Clear, pass),
        (Writer::ClearDontCare, pass),
        (Writer::Draw, pass),
        (Writer::DrawDontCare, pass),
        (Writer::Load, pass),
    ];
    for (writer, selector) in dies {
        let blit = matches!(writer, Writer::Copy | Writer::FillColor);
        let extents: &[(u64, u64)] = match blit {
            true => &[(9, 1), (16, 1), (64, 16)],
            false => &[(2, 1), (64, 16)],
        };
        // Each case's size, extent and origin.
        let mut cases: Vec<_> = extents
            .iter()
            .map(|&extent| ((64, 16), extent, (0, 0)))
            .collect();
        if blit {
            cases.push(((32, 32), (2, 1), (0, 0)));
            cases.extend([(1, 8), (8, 8)].map(|extent| ((64, 16), extent, (40, 10))));
        }
        for (size, extent, origin) in cases {
            let c = Case {
                size,
                usage: READ_WRITE,
                writer,
                extent,
                origin,
            };
            let site = catch_death(|| case(&p, &d, c)).unwrap_err();
            assert_eq!(site.file, selector, "{c:?}");
        }
    }
    let several = Case {
        size: (128, 128),
        usage: READ_WRITE,
        writer: Writer::Replace,
        extent: (2, 1),
        origin: (0, 0),
    };
    let site = catch_death(|| case(&p, &d, several)).unwrap_err();
    assert_eq!(
        site.file,
        "replaceRegion:mipmapLevel:slice:withBytes:bytesPerRow:bytesPerImage:"
    );
}

/// A blit's copy from a buffer or colour fill into the texture of 1 to 8 texels each way, on
/// surfaces of 64 by 16 and 256 by 256 texels (rows of 256 and 1,024 bytes), at origins 0,0, 1,1
/// and one far inside, is dropped, as Metal dropped each in the [`sweep`] (two runs, one case a
/// process; rows of 16 texels or more faulted it at some origins). Where `past_the_level`, so is
/// one past the level's last row (8 rows from row 10 of 64 by 16), which Metal drops too and
/// this platform dies at ([`this_platform_dies_at_every_other_write`]).
fn small_blits_dropped<P: Platform>(p: &P, d: &P::Device, past_the_level: bool) {
    for (size, far) in [((64, 16), (40, 10)), ((256, 256), (150, 200))] {
        for writer in [Writer::Copy, Writer::FillColor] {
            for origin in [(0, 0), (1, 1), far] {
                for extent in [(1, 1), (2, 1), (3, 1), (8, 1), (1, 8), (4, 4), (8, 8)] {
                    let inside = origin.0 + extent.0 <= size.0 && origin.1 + extent.1 <= size.1;
                    if !inside && !past_the_level {
                        continue;
                    }
                    let c = Case {
                        size,
                        usage: READ_WRITE,
                        writer,
                        extent,
                        origin,
                    };
                    let seen = case(p, d, c);
                    assert!(dropped(&seen), "{c:?}: {seen:?}");
                }
            }
        }
    }
}

#[test]
fn this_platform_drops_small_blits_into_the_texture() {
    let p = VulkanPlatform::new();
    small_blits_dropped(&p, &VulkanPlatform::system_default_device().unwrap(), false);
}

/// A surface over the guest's pages with `kIOSurfaceClientDirection` 2, 64 by 16 `BGRA8Unorm`
/// texels, and a texture on it of `desc`.
fn read_only_texture<P: Platform>(
    p: &P,
    d: &P::Device,
    desc: &P::TextureDescriptor,
) -> (P::Surface, P::Texture) {
    let page = OsVm::new().page_size();
    let pages = guest_pages(page);
    let spec = SurfaceSpec {
        pixel_format: u32::from_be_bytes(*b"BGRA"),
        alloc_size: page,
        width: 64,
        height: 16,
        bytes_per_element: 4,
        bytes_per_row: 256,
        element_width: 1,
        element_height: 1,
        client_direction: Some(2),
        client_address_ranges: Some(vec![PhysicalMemoryRange {
            physical_address: pages.addr() as u64,
            physical_length: page,
        }]),
        ..SurfaceSpec::default()
    };
    let s = p.surfaces().create(&spec).unwrap();
    let t = d
        .new_texture_on_surface(desc, Some(&s), 0)
        .unwrap()
        .unwrap();
    (s, t)
}

/// A `BGRA8Unorm` texture descriptor of `size`, `usage` and `samples` (a 2D multisample texture
/// of `Private` storage above 1, a 2D `Shared` one otherwise).
fn bgra_desc((width, height): (u64, u64), usage: u64, samples: u64) -> TextureDesc {
    TextureDesc {
        texture_type: if samples > 1 { 4 } else { 2 },
        pixel_format: BGRA8,
        width,
        height,
        sample_count: samples,
        usage,
        resource_options: Some(if samples > 1 { 0x20 } else { 0 }),
        ..TextureDesc::default()
    }
}

/// [`bgra_desc`]'s descriptor of the platform's.
fn bgra<P: Platform>(p: &P, size: (u64, u64), usage: u64, samples: u64) -> P::TextureDescriptor {
    p.texture_descriptor(&bgra_desc(size, usage, samples))
}

/// A render pass descriptor of colour attachment 0 `attachment`, and the visibility result
/// buffer `visibility`.
fn pass<P: Platform>(
    p: &P,
    attachment: &ColorAttachmentSet<'_, P>,
    visibility: Option<&P::Buffer>,
) -> P::RenderPassDescriptor {
    use reims_vgpu_core::platform::values::PassSet;
    let pass = p.render_pass_descriptor();
    pass.set_color_attachment(OnFault::Die, 0, attachment)
        .unwrap();
    if let Some(b) = visibility {
        pass.set_pass(&PassSet {
            visibility_result_buffer: Some(Some(b)),
            ..PassSet::default()
        });
    }
    pass
}

/// The sweep kernel's state, which writes texture 0 (`sweep_kernel`), and its kind buffer.
fn sweep_kernel<P: Platform>(d: &P::Device) -> (P::ComputePipeline, P::Buffer) {
    let rounding = d.new_library(ROUNDING).unwrap().unwrap().unwrap();
    let f = rounding
        .new_function(OnFault::Die, "sweep_kernel")
        .unwrap()
        .unwrap();
    let state = d
        .new_compute_pipeline_state_for_function(Some(&f))
        .unwrap()
        .unwrap();
    let kind = shared::<P>(d, 16, 0);
    (state, kind)
}

/// A dispatch of `state` over 64 by 16 threads with `texture` at slot 0, `kind` at buffer 0,
/// and `used` made resident with `usage`.
fn dispatch<P: Platform>(
    e: &P::ComputeEncoder,
    (state, kind): (&P::ComputePipeline, &P::Buffer),
    texture: &P::Texture,
    used: Option<(&P::Texture, u64)>,
) {
    let mut cmds = vec![
        ComputeCmd::SetComputePipelineState { state: Some(state) },
        ComputeCmd::SetTexture {
            texture: Some(texture),
            index: 0,
        },
    ];
    let resources;
    if let Some((t, usage)) = used {
        resources = [Some(ResourceRef::Texture(t))];
        cmds.push(ComputeCmd::UseResources {
            resources: &resources,
            usage,
        });
    }
    let buffers = [Some(kind)];
    cmds.push(ComputeCmd::SetBuffers {
        buffers: &buffers,
        offsets: &[0],
        range: NsRange::new(0, 1),
    });
    cmds.push(ComputeCmd::DispatchThreads {
        threads_per_grid: MtlSize::new(64, 16, 1),
        threads_per_threadgroup: MtlSize::new(16, 16, 1),
    });
    for cmd in cmds {
        e.encode(OnFault::Die, cmd).unwrap();
    }
}

/// Colour attachment 0 of `texture`, cleared, that resolves into `resolve` at store action
/// `store`.
fn resolving<'a>(
    texture: &'a VulkanTexture,
    resolve: &'a VulkanTexture,
    store: u64,
) -> ColorAttachmentSet<'a, VulkanPlatform> {
    ColorAttachmentSet {
        texture: Some(Some(texture)),
        resolve_texture: Some(Some(resolve)),
        load_action: Some(2),
        store_action: Some(store),
        ..ColorAttachmentSet::default()
    }
}

/// Every way a write reaches the surface's memory dies at its encode, never sent: a pass that
/// resolves into the texture, named by the descriptor or set later, at the pass's creation or
/// at the setter; one undecided over it at the late store action that resolves; one whose
/// visibility result buffer lies on the surface at its creation; and a dispatch that makes the
/// texture resident for writing, or writes a view of it or a texture over a buffer on the
/// surface. An undecided pass that comes to store without resolving, a draw of no vertices
/// through the texture, and a dispatch that makes it resident for reading, write nothing and go
/// on.
#[test]
fn this_platform_dies_at_every_way_a_write_reaches_the_surface() {
    let p = VulkanPlatform::new();
    let d = VulkanPlatform::system_default_device().unwrap();
    let q = d.new_command_queue().unwrap();
    let cb = || q.command_buffer(OnFault::Die).unwrap().unwrap();
    let pass_sel = "renderCommandEncoderWithDescriptor:";
    let dispatch_sel = "dispatchThreads:threadsPerThreadgroup:";
    let (s, t) = read_only_texture(&p, &d, &bgra(&p, (64, 16), READ_WRITE | 4, 1));
    let ms = d
        .new_texture(OnFault::Die, &bgra::<VulkanPlatform>(&p, (64, 16), 4, 4))
        .unwrap()
        .unwrap();
    let own = d
        .new_texture(OnFault::Die, &bgra::<VulkanPlatform>(&p, (64, 16), 4, 1))
        .unwrap()
        .unwrap();
    let resolving = |resolve, store| resolving(&ms, resolve, store);
    // Resolved into at the pass's creation.
    let site = catch_death(|| {
        let b = cb();
        b.render_encoder(OnFault::Die, &pass(&p, &resolving(&t, 2), None))
            .map(|e| e.is_some())
    })
    .unwrap_err();
    assert_eq!(site.file, pass_sel, "a resolve the descriptor names");
    // Undecided, then resolved into by a late store action.
    let site = catch_death(|| {
        let b = cb();
        let e = (b.render_encoder(OnFault::Die, &pass(&p, &resolving(&t, 4), None)))
            .unwrap()
            .unwrap();
        e.encode(
            OnFault::Die,
            RenderCmd::SetColorStoreAction {
                store_action: 2,
                color_attachment_index: 0,
            },
        )
    })
    .unwrap_err();
    assert_eq!(site.file, "setColorStoreAction:atIndex:");
    // Undecided, then stored without a resolve: nothing written, nothing dies.
    let b = cb();
    let e = (b.render_encoder(OnFault::Die, &pass(&p, &resolving(&t, 4), None)))
        .unwrap()
        .unwrap();
    for cmd in [
        RenderCmd::SetColorStoreAction {
            store_action: 1,
            color_attachment_index: 0,
        },
        RenderCmd::EndEncoding,
    ] {
        e.encode(OnFault::Die, cmd).unwrap();
    }
    // A resolve texture set after the pass's creation.
    let site = catch_death(|| {
        let b = cb();
        let e = (b.render_encoder(OnFault::Die, &pass(&p, &resolving(&own, 2), None)))
            .unwrap()
            .unwrap();
        e.encode(
            OnFault::Die,
            RenderCmd::SetColorResolveTexture {
                texture: Some(&t),
                slice: 0,
                depth_plane: 0,
                level: 0,
                y_invert: false,
                index: 0,
            },
        )
    })
    .unwrap_err();
    assert_eq!(
        site.file,
        "setColorResolveTexture:slice:depthPlane:level:yInvert:atIndex:"
    );
    // A visibility result buffer on the surface.
    let on_surface = d.new_buffer_on_surface(Some(&s)).unwrap().unwrap();
    let target = ColorAttachmentSet {
        texture: Some(Some(&own)),
        load_action: Some(0),
        store_action: Some(0),
        ..ColorAttachmentSet::default()
    };
    let site = catch_death(|| {
        let b = cb();
        b.render_encoder(OnFault::Die, &pass(&p, &target, Some(&on_surface)))
            .map(|e| e.is_some())
    })
    .unwrap_err();
    assert_eq!(site.file, pass_sel, "a visibility result buffer");
    // Dispatches: made resident for writing, a view, a texture over a buffer on the surface.
    let (state, kind) = sweep_kernel::<VulkanPlatform>(&d);
    let plain = d
        .new_texture(
            OnFault::Die,
            &bgra::<VulkanPlatform>(&p, (64, 16), READ_WRITE, 1),
        )
        .unwrap()
        .unwrap();
    let view = t.new_texture_view(BGRA8).unwrap().unwrap();
    let over = on_surface
        .new_texture(
            OnFault::Die,
            &bgra::<VulkanPlatform>(&p, (64, 16), READ_WRITE, 1),
            0,
            256,
        )
        .unwrap()
        .unwrap();
    let writes = [
        (&plain, Some((&t, 3)), "made resident for writing"),
        (&view, None, "a view"),
        (&over, None, "a texture over a buffer on the surface"),
    ];
    for (texture, used, what) in writes {
        let site = catch_death(|| {
            let b = cb();
            let e = b.compute_encoder(OnFault::Die).unwrap().unwrap();
            dispatch::<VulkanPlatform>(&e, (&state, &kind), texture, used);
        })
        .unwrap_err();
        assert_eq!(site.file, dispatch_sel, "{what}");
    }
    // Made resident for reading: nothing written, nothing dies.
    let b = cb();
    let e = b.compute_encoder(OnFault::Die).unwrap().unwrap();
    dispatch::<VulkanPlatform>(&e, (&state, &kind), &plain, Some((&t, 1)));
    e.encode(OnFault::Die, ComputeCmd::EndEncoding).unwrap();
    // A draw of no vertices through the texture goes on; one of three dies.
    let rounding = d.new_library(ROUNDING).unwrap().unwrap().unwrap();
    let function = |name| rounding.new_function(OnFault::Die, name).unwrap().unwrap();
    let (v, f) = (function("sweep_vertex"), function("write_fragment"));
    let drawn = d
        .new_render_pipeline_state(&RenderPipelineDesc {
            label: None,
            vertex: &v,
            fragment: &f,
            color0_pixel_format: BGRA8,
            resource_index: None,
        })
        .unwrap()
        .unwrap();
    let site = catch_death(|| {
        let b = cb();
        let e = (b.render_encoder(OnFault::Die, &pass(&p, &target, None)))
            .unwrap()
            .unwrap();
        for vertex_count in [0, 3] {
            for cmd in [
                RenderCmd::SetRenderPipelineState {
                    pipeline_state: Some(&drawn),
                },
                RenderCmd::SetFragmentBuffer {
                    buffer: Some(&kind),
                    offset: 0,
                    index: 0,
                },
                RenderCmd::SetFragmentTexture {
                    texture: Some(&t),
                    index: 0,
                },
                RenderCmd::DrawPrimitives {
                    primitive_type: 3,
                    vertex_start: 0,
                    vertex_count,
                },
            ] {
                e.encode(OnFault::Die, cmd).unwrap();
            }
        }
    })
    .unwrap_err();
    assert_eq!(site.file, "drawPrimitives:vertexStart:vertexCount:");
}

/// A texture on a surface made read-only for the GPU answers the usage Metal's does, for each
/// descriptor usage, `Unknown` and `RenderTarget` alone among them, though this platform makes its
/// runtime texture without `RenderTarget`; so do a view of it, a swizzled view, which keeps only
/// `ShaderRead` and `PixelFormatView` of it, a texture over a buffer made on the surface, and a
/// texture of a swizzled descriptor, of the usages Metal makes one of (none with `ShaderWrite`
/// or `RenderTarget`, at which its descriptor validation asserts).
#[test]
fn read_only_textures_answer_metals_usage() {
    const SWAPPED: Swizzle = Swizzle {
        red: 4,
        green: 3,
        blue: 2,
        alpha: 5,
    };
    fn usages<P: Platform>(p: &P, d: &P::Device) -> (Vec<[u64; 4]>, Vec<u64>) {
        let one = NsRange::new(0, 1);
        let usage = |usage| {
            let (s, t) = read_only_texture(p, d, &bgra(p, (64, 16), usage, 1));
            let view = t.new_texture_view(BGRA8).unwrap().unwrap();
            let subview = t.new_swizzled_texture_subview(BGRA8, 2, one, one, SWAPPED);
            let on_surface = d.new_buffer_on_surface(Some(&s)).unwrap().unwrap();
            let desc = bgra(p, (64, 16), usage, 1);
            let over = on_surface.new_texture(OnFault::Die, &desc, 0, 256);
            [
                t.usage(),
                view.usage(),
                subview.unwrap().unwrap().usage(),
                over.unwrap().unwrap().usage(),
            ]
        };
        let of_swizzled = |usage| {
            let desc = p.texture_descriptor(&TextureDesc {
                swizzle: Some(SWAPPED),
                ..bgra_desc((64, 16), usage, 1)
            });
            read_only_texture(p, d, &desc).1.usage()
        };
        (
            [0, 1, 3, 4, 5, 7, 0x10, 0x14].map(usage).to_vec(),
            [0, 1, 0x10, 0x11].map(of_swizzled).to_vec(),
        )
    }
    let vulkan = usages(
        &VulkanPlatform::new(),
        &VulkanPlatform::system_default_device().unwrap(),
    );
    reims_vgpu_core::metal_check!("usages", vulkan);
}

/// The tests' vertex function that returns nothing (`src/gpu/unrasterized.metal`).
const UNRASTERIZED: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/unrasterized.metallib"));

/// A draw of a render state with rasterization off whose fragment function writes texture 0
/// (`write_fragment`), with a texture on a surface made read-only for the GPU bound there: the
/// command buffer's status. Its vertex function returns nothing (`void_vertex`), and colour 0 is
/// `BGRA8Unorm`.
fn unrasterized_write<P: Platform>(p: &P, d: &P::Device) -> u64 {
    use reims_vgpu_core::platform::values::{ColorAttachmentDesc, FullRenderPipelineDesc};
    let q = d.new_command_queue().unwrap();
    let (_s, t) = read_only_texture(p, d, &bgra(p, (64, 16), 3, 1));
    let own = d
        .new_texture(OnFault::Die, &bgra::<P>(p, (64, 16), 4, 1))
        .unwrap()
        .unwrap();
    let target = ColorAttachmentSet {
        texture: Some(Some(&own)),
        load_action: Some(0),
        store_action: Some(1),
        ..ColorAttachmentSet::default()
    };
    let kind = shared::<P>(d, 16, 0);
    let first = |bytes: &[u8]| {
        let library = d.new_library(bytes).unwrap().unwrap().unwrap();
        let name = library.function_names().remove(0);
        library.new_function(OnFault::Die, &name).unwrap().unwrap()
    };
    let rounding = d.new_library(ROUNDING).unwrap().unwrap().unwrap();
    let (vertex, fragment) = (
        first(UNRASTERIZED),
        rounding
            .new_function(OnFault::Die, "write_fragment")
            .unwrap()
            .unwrap(),
    );
    // Rasterization off, and colour 0 `BGRA8Unorm` (80).
    let state = d
        .new_render_pipeline_state_full(&FullRenderPipelineDesc {
            colors: vec![(
                0,
                ColorAttachmentDesc {
                    pixel_format: 80,
                    ..ColorAttachmentDesc::default()
                },
            )],
            rasterization_enabled: false,
            ..FullRenderPipelineDesc::new(&vertex, Some(&fragment))
        })
        .unwrap()
        .unwrap();
    let b = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let e = (b.render_encoder(OnFault::Die, &pass(p, &target, None)))
        .unwrap()
        .unwrap();
    for cmd in [
        RenderCmd::SetRenderPipelineState {
            pipeline_state: Some(&state),
        },
        RenderCmd::SetFragmentBuffer {
            buffer: Some(&kind),
            offset: 0,
            index: 0,
        },
        RenderCmd::SetFragmentTexture {
            texture: Some(&t),
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
    b.commit(OnFault::Die).unwrap();
    b.wait_until_completed(OnFault::Die).unwrap();
    b.status().0
}

/// A draw of a render state with rasterization off, whose fragment function writes a texture
/// on a surface made read-only for the GPU, completes its command buffer, as Metal's does: no
/// fragment runs, so no write reaches the surface
#[test]
fn an_unrasterized_draw_writes_through_no_fragment_texture() {
    let p = VulkanPlatform::new();
    let d = VulkanPlatform::system_default_device().unwrap();
    let ours = unrasterized_write(&p, &d);
    assert_eq!(ours, 4);
}

/// A draw whose function only reads a buffer on a surface made read-only for the GPU, through a
/// `device` pointer that is not `const` (`read_vertex`, `read_fragment`), completes its command
/// buffer, as Metal's does: a draw's stores into such a buffer are sent, whether or not its
/// function stores through it.
#[test]
fn this_platform_draws_through_a_device_pointer_into_a_read_only_buffer() {
    let p = VulkanPlatform::new();
    let d = VulkanPlatform::system_default_device().unwrap();
    let q = d.new_command_queue().unwrap();
    let (s, _t) = read_only_texture(&p, &d, &bgra(&p, (64, 16), 1, 1));
    let on_surface = d.new_buffer_on_surface(Some(&s)).unwrap().unwrap();
    let own = d
        .new_texture(OnFault::Die, &bgra::<VulkanPlatform>(&p, (64, 16), 4, 1))
        .unwrap()
        .unwrap();
    let target = ColorAttachmentSet {
        texture: Some(Some(&own)),
        load_action: Some(0),
        store_action: Some(1),
        ..ColorAttachmentSet::default()
    };
    let kind = shared::<VulkanPlatform>(&d, 16, 0);
    let rounding = d.new_library(ROUNDING).unwrap().unwrap().unwrap();
    let function = |name| rounding.new_function(OnFault::Die, name).unwrap().unwrap();
    let state = |vertex, fragment| {
        let (vertex, fragment) = (function(vertex), function(fragment));
        d.new_render_pipeline_state(&RenderPipelineDesc {
            label: None,
            vertex: &vertex,
            fragment: &fragment,
            color0_pixel_format: BGRA8,
            resource_index: None,
        })
        .unwrap()
        .unwrap()
    };
    let draw = |(vertex, fragment), at_vertex: bool| {
        let state = state(vertex, fragment);
        let b = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = (b.render_encoder(OnFault::Die, &pass(&p, &target, None)))
            .unwrap()
            .unwrap();
        let (buffer, offset, index) = (Some(&on_surface), 0, 1);
        let bind = match at_vertex {
            true => RenderCmd::SetVertexBuffer {
                buffer,
                offset,
                index,
            },
            false => RenderCmd::SetFragmentBuffer {
                buffer,
                offset,
                index,
            },
        };
        for cmd in [
            RenderCmd::SetRenderPipelineState {
                pipeline_state: Some(&state),
            },
            RenderCmd::SetFragmentBuffer {
                buffer: Some(&kind),
                offset: 0,
                index: 0,
            },
            bind,
            RenderCmd::DrawPrimitives {
                primitive_type: 3,
                vertex_start: 0,
                vertex_count: 3,
            },
            RenderCmd::EndEncoding,
        ] {
            e.encode(OnFault::Die, cmd).unwrap();
        }
        b.commit(OnFault::Die).unwrap();
        b.wait_until_completed(OnFault::Die).unwrap();
        b.status().0
    };
    assert_eq!(draw(("read_vertex", "sweep_fragment"), true), 4);
    assert_eq!(draw(("sweep_vertex", "read_fragment"), false), 4);
}

/// What a creator answers for a texture of a swizzle other than the default (B, G, R, A) and
/// `USAGE` (`writeSwizzleEnabled` where `WSE` is 1), from the environment: `CREATOR` `surface`
/// (on a surface made read-only for the GPU), `buffer` (over a buffer made on such a surface) or
/// `device`. Prints the texture's usage, or nil, or where this platform dies. A measurement, one
/// case a process.
#[test]
#[ignore = "a measurement: one case per process, from the environment"]
fn swizzled_texture_on_read_only_memory() {
    fn made<P: Platform>(p: &P, d: &P::Device) -> String {
        let creator = std::env::var("CREATOR").unwrap();
        let usage: u64 = std::env::var("USAGE").unwrap().parse().unwrap();
        let wse = std::env::var("WSE").as_deref() == Ok("1");
        let desc = p.texture_descriptor(&TextureDesc {
            swizzle: Some(Swizzle {
                red: 4,
                green: 3,
                blue: 2,
                alpha: 5,
            }),
            write_swizzle_enabled: wse.then_some(true),
            ..bgra_desc((64, 16), usage, 1)
        });
        let plain = bgra(p, (64, 16), 1, 1);
        let t = match creator.as_str() {
            "surface" => {
                let (s, _) = read_only_texture(p, d, &plain);
                let t = d.new_texture_on_surface(&desc, Some(&s), 0).unwrap();
                drop(s);
                t
            }
            "buffer" => {
                let (s, _) = read_only_texture(p, d, &plain);
                let on_surface = d.new_buffer_on_surface(Some(&s)).unwrap().unwrap();
                on_surface.new_texture(OnFault::Die, &desc, 0, 256).unwrap()
            }
            _ => d.new_texture(OnFault::Die, &desc).unwrap(),
        };
        t.map_or("nil".into(), |t| format!("made usage {:#x}", t.usage()))
    }
    let p = VulkanPlatform::new();
    let d = VulkanPlatform::system_default_device().unwrap();
    match catch_death(|| made(&p, &d)) {
        Ok(seen) => println!("vulkan {seen}"),
        Err(site) => println!("vulkan died at {} ({:?})", site.file, site.kind),
    }
}
