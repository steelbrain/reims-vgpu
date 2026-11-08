//! What Metal leaves where its documentation leaves a value undefined, measured on this host's
//! Metal beside this platform's answer: the bytes a render pass leaves in the tiles its draws do
//! not reach ([`dont_care_load`], [`unreached_tiles`]), the bits an extended-range format's codes
//! leave unused ([`extended_range`]), and the bytes of a texture placed in a heap
//! ([`heap_layout`]). Each module's docs state Metal's rule over every shape it probes, whether
//! Metal's answer is one value (each probe runs [`RUNS`] times, each run on resources of its
//! own), what this platform answers today, and what it would take to answer as Metal.
//!
//! **Measured on one GPU.** Every answer here is an Apple M3 Max's (40-core GPU) under macOS 26.7
//! (25G229). Other Apple GPU generations were not measured: an answer here may not hold on them.

use reims_vgpu_core::platform::HostRange;
use reims_vgpu_core::platform::Platform;
use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder as _};
use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
use reims_vgpu_core::platform::descriptor::RenderPassDescriptor as _;
use reims_vgpu_core::platform::device::GpuDevice as _;
use reims_vgpu_core::platform::render::{RenderCmd, RenderEncoder as _};
use reims_vgpu_core::platform::resource::{Buffer as _, Library as _, Texture as _};
use reims_vgpu_core::platform::types::{ClearColor, MtlOrigin, MtlRegion, MtlSize};
use reims_vgpu_core::platform::values::{
    ColorAttachmentSet, DepthAttachmentSet, DepthStencilDesc, FullRenderPipelineDesc,
    StencilAttachmentSet, StencilDesc, TextureDesc,
};
use reims_vgpu_core::raise::OnFault;
use reims_vgpu_metal::MetalPlatform;

use crate::gpu::device::tests::colours_of;

/// How many times each probe runs on Metal, each run on resources of its own: every answer
/// recorded here was the same in each.
pub(super) const RUNS: usize = 5;

/// `rounding.metal`'s library: `reach_vertex` places buffer 0's float4s, one a vertex, and
/// `xr_constant_fragment` writes (0.5, 0.25, 0.75, 1) as a `half4` to colour 0.
const ROUNDING: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rounding.metallib"));

/// `MTLLoadAction` and `MTLStoreAction` values.
pub(super) const DONT_CARE: u64 = 0;
pub(super) const LOAD: u64 = 1;
pub(super) const CLEAR: u64 = 2;
pub(super) const STORE: u64 = 1;
pub(super) const MULTISAMPLE_RESOLVE: u64 = 2;
pub(super) const STORE_AND_RESOLVE: u64 = 3;

/// `MTLResourceOptions`' storage modes.
pub(super) const SHARED: u64 = 0;
pub(super) const PRIVATE: u64 = 0x20;

/// `MTLTextureUsage` bits.
pub(super) const READ: u64 = 1;
pub(super) const WRITE: u64 = 2;
pub(super) const TARGET: u64 = 4;

/// Pixel formats.
pub(super) const RGBA8: u64 = 70;
pub(super) const RGBA16F: u64 = 115;
pub(super) const RGBA32F: u64 = 125;
pub(super) const DEPTH32F: u64 = 252;
pub(super) const STENCIL8: u64 = 253;

/// Metal's platform and device, where the host has Metal.
pub(super) fn metal() -> Option<(MetalPlatform, <MetalPlatform as Platform>::Device)> {
    MetalPlatform::system_default_device().map(|d| (MetalPlatform::new(), d))
}

/// The bytes a texel of `format` takes.
pub(super) fn texel_bytes(format: u64) -> usize {
    match format {
        10..=14 | STENCIL8 => 1,
        20..=25 | 250 => 2,
        112 | 115 | 552 | 553 => 8,
        RGBA32F => 16,
        _ => 4,
    }
}

/// Whether `format` is a depth format, a stencil format, or neither.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Aspect {
    Colour,
    Depth,
    Stencil,
}

impl Aspect {
    pub(super) fn of(format: u64) -> Aspect {
        match format {
            250 | DEPTH32F => Aspect::Depth,
            STENCIL8 => Aspect::Stencil,
            _ => Aspect::Colour,
        }
    }
}

/// A Shared buffer of `length` bytes, `contents` at its start.
pub(super) fn buffer<P: Platform>(d: &P::Device, length: usize, contents: &[u8]) -> P::Buffer {
    let b = d
        .new_buffer_of_length(OnFault::Die, u64::try_from(length.max(16)).unwrap(), SHARED)
        .unwrap()
        .unwrap();
    // SAFETY: a Shared buffer of at least `contents.len()` bytes no command uses yet.
    unsafe {
        std::ptr::copy_nonoverlapping(
            contents.as_ptr(),
            b.contents().as_ptr::<u8>(),
            contents.len(),
        );
    }
    b
}

/// The first `length` bytes of `b`, a Shared buffer no command writes now.
pub(super) fn bytes_of<P: Platform>(b: &P::Buffer, length: usize) -> Vec<u8> {
    // SAFETY: a Shared buffer of at least `length` bytes, read after its writers completed.
    unsafe { std::slice::from_raw_parts(b.contents().as_ptr::<u8>(), length) }.to_vec()
}

/// A texture's shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Shape {
    pub(super) format: u64,
    pub(super) size: (u64, u64),
    pub(super) samples: u64,
    pub(super) options: u64,
    pub(super) usage: u64,
    pub(super) optimized: Option<bool>,
}

impl Shape {
    /// A single-sample render target of `format` and `size`, Private, of Metal's default
    /// `allowGPUOptimizedContents`.
    pub(super) fn of(format: u64, size: (u64, u64)) -> Shape {
        Shape {
            format,
            size,
            samples: 1,
            options: PRIVATE,
            usage: TARGET | READ,
            optimized: None,
        }
    }

    pub(super) fn desc(self) -> TextureDesc {
        TextureDesc {
            texture_type: if self.samples > 1 { 4 } else { 2 },
            pixel_format: self.format,
            width: self.size.0,
            height: self.size.1,
            sample_count: self.samples,
            usage: self.usage,
            resource_options: Some(self.options),
            allow_gpu_optimized_contents: self.optimized,
            ..TextureDesc::default()
        }
    }

    /// The bytes of one sample of every texel, rows tight.
    pub(super) fn length(self) -> usize {
        let (w, h) = self.size;
        usize::try_from(w * h).unwrap() * texel_bytes(self.format)
    }

    pub(super) fn texture<P: Platform>(self, p: &P, d: &P::Device) -> P::Texture {
        let desc = p.texture_descriptor(&self.desc());
        d.new_texture(OnFault::Die, &desc).unwrap().unwrap()
    }
}

/// `n` bytes from a fixed seed: every bit pattern a texel may hold, NaNs and codes past a
/// format's range among them.
pub(super) fn random_bytes(n: usize, seed: u32) -> Vec<u8> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12345);
            u8::try_from(state >> 23 & 0xff).unwrap()
        })
        .collect()
}

/// The commands of one command buffer, in order.
pub(super) enum Step<'a, P: Platform> {
    /// A blit of `bytes` into the whole of a single-sample texture.
    Upload(&'a P::Texture, Shape, &'a [u8]),
    /// A pass.
    Pass(Pass<'a, P>),
    /// A blit of the whole of a single-sample texture into a buffer; its bytes come back in order.
    Read(&'a P::Texture, Shape),
    /// A blit of the whole of one single-sample texture of `Shape` into another of its shape.
    Copy(&'a P::Texture, &'a P::Texture, Shape),
}

/// A colour attachment: its texture, load and store actions, resolve texture, and clear colour
/// ([`CLEAR_COLOUR`] where unset).
pub(super) struct Colour<'a, P: Platform> {
    pub(super) texture: &'a P::Texture,
    pub(super) load: u64,
    pub(super) store: u64,
    pub(super) resolve: Option<&'a P::Texture>,
    pub(super) clear: Option<ClearColor>,
}

impl<'a, P: Platform> Colour<'a, P> {
    pub(super) fn new(texture: &'a P::Texture, (load, store): (u64, u64)) -> Colour<'a, P> {
        Colour {
            texture,
            load,
            store,
            resolve: None,
            clear: None,
        }
    }
}

/// A render pass: its attachments (a depth or stencil one with its load and store actions) and
/// its draws.
pub(super) struct Pass<'a, P: Platform> {
    pub(super) colours: Vec<Colour<'a, P>>,
    pub(super) depth: Option<(&'a P::Texture, u64, u64)>,
    pub(super) stencil: Option<(&'a P::Texture, u64, u64)>,
    pub(super) draws: Vec<Draw<'a, P>>,
}

impl<'a, P: Platform> Pass<'a, P> {
    /// A pass of `texture` of `aspect`, loaded and stored by `actions`, drawing `draws`.
    pub(super) fn of(
        texture: &'a P::Texture,
        aspect: Aspect,
        actions: (u64, u64),
        draws: Vec<Draw<'a, P>>,
    ) -> Pass<'a, P> {
        let (load, store) = actions;
        Pass {
            colours: if aspect == Aspect::Colour {
                vec![Colour::new(texture, actions)]
            } else {
                Vec::new()
            },
            depth: (aspect == Aspect::Depth).then_some((texture, load, store)),
            stencil: (aspect == Aspect::Stencil).then_some((texture, load, store)),
            draws,
        }
    }
}

/// A triangle list's draw of `vertices` (`reach_vertex`'s positions) under `state` and, when
/// set, `depth_stencil`, with the stencil reference value 0x33.
pub(super) struct Draw<'a, P: Platform> {
    pub(super) state: &'a P::RenderPipeline,
    pub(super) depth_stencil: Option<&'a P::DepthStencilState>,
    pub(super) positions: &'a P::Buffer,
    pub(super) vertices: u64,
}

/// The clear values every pass here sets: a colour of (0.125, 0.375, 0.625, 0.875), depth
/// 0.25, stencil 0x5a.
pub(super) const CLEAR_COLOUR: ClearColor = ClearColor {
    red: 0.125,
    green: 0.375,
    blue: 0.625,
    alpha: 0.875,
};
pub(super) const CLEAR_DEPTH: f64 = 0.25;
pub(super) const CLEAR_STENCIL: u32 = 0x5a;

fn encode_pass<P: Platform>(p: &P, cb: &P::CommandBuffer, pass: &Pass<'_, P>) {
    let desc = p.render_pass_descriptor();
    for (index, c) in pass.colours.iter().enumerate() {
        let set = ColorAttachmentSet {
            texture: Some(Some(c.texture)),
            resolve_texture: c.resolve.map(Some),
            load_action: Some(c.load),
            store_action: Some(c.store),
            clear_color: Some(c.clear.unwrap_or(CLEAR_COLOUR)),
            ..ColorAttachmentSet::default()
        };
        desc.set_color_attachment(OnFault::Die, u64::try_from(index).unwrap(), &set)
            .unwrap();
    }
    if let Some((texture, load, store)) = pass.depth {
        desc.set_depth_attachment(&DepthAttachmentSet {
            texture: Some(Some(texture)),
            load_action: Some(load),
            store_action: Some(store),
            clear_depth: Some(CLEAR_DEPTH),
            ..DepthAttachmentSet::default()
        });
    }
    if let Some((texture, load, store)) = pass.stencil {
        desc.set_stencil_attachment(&StencilAttachmentSet {
            texture: Some(Some(texture)),
            load_action: Some(load),
            store_action: Some(store),
            clear_stencil: Some(CLEAR_STENCIL),
            ..StencilAttachmentSet::default()
        });
    }
    let e = cb.render_encoder(OnFault::Die, &desc).unwrap().unwrap();
    for draw in &pass.draws {
        let cmds = [
            RenderCmd::SetRenderPipelineState {
                pipeline_state: Some(draw.state),
            },
            RenderCmd::SetDepthStencilState {
                depth_stencil_state: draw.depth_stencil,
            },
            RenderCmd::SetStencilFrontReferenceValue {
                front_reference_value: 0x33,
                back_reference_value: 0x33,
            },
            RenderCmd::SetVertexBuffer {
                buffer: Some(draw.positions),
                offset: 0,
                index: 0,
            },
            RenderCmd::DrawPrimitives {
                primitive_type: 3,
                vertex_start: 0,
                vertex_count: draw.vertices,
            },
        ];
        for cmd in cmds {
            e.encode(OnFault::Die, cmd).unwrap();
        }
    }
    e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
}

/// Runs `steps` in one command buffer on `P` and answers the bytes of each [`Step::Read`], in
/// order.
pub(super) fn run<P: Platform>(p: &P, d: &P::Device, steps: &[Step<'_, P>]) -> Vec<Vec<u8>> {
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let mut reads = Vec::new();
    let mut keep = Vec::new();
    for step in steps {
        let blit = |cmd: BlitCmd<'_, P>| {
            let b = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
            b.encode(OnFault::Die, cmd).unwrap();
            b.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
        };
        match step {
            Step::Upload(t, shape, bytes) => {
                let up = buffer::<P>(d, bytes.len(), bytes);
                let (w, h) = shape.size;
                let row = w * u64::try_from(texel_bytes(shape.format)).unwrap();
                blit(BlitCmd::CopyBufferToTexture {
                    source_buffer: &up,
                    source_offset: 0,
                    source_bytes_per_row: row,
                    source_bytes_per_image: row * h,
                    source_size: MtlSize::new(w, h, 1),
                    destination_texture: t,
                    destination_slice: 0,
                    destination_level: 0,
                    destination_origin: MtlOrigin::new(0, 0, 0),
                });
                keep.push(up);
            }
            Step::Pass(pass) => encode_pass(p, &cb, pass),
            Step::Read(t, shape) => {
                let down = buffer::<P>(d, shape.length(), &[]);
                let (w, h) = shape.size;
                let row = w * u64::try_from(texel_bytes(shape.format)).unwrap();
                blit(BlitCmd::CopyTextureToBuffer {
                    source_texture: t,
                    source_slice: 0,
                    source_level: 0,
                    source_origin: MtlOrigin::new(0, 0, 0),
                    source_size: MtlSize::new(w, h, 1),
                    destination_buffer: &down,
                    destination_offset: 0,
                    destination_bytes_per_row: row,
                    destination_bytes_per_image: row * h,
                });
                reads.push((down, shape.length()));
            }
            Step::Copy(from, to, shape) => blit(BlitCmd::CopyTextureRegion {
                source_texture: from,
                source_slice: 0,
                source_level: 0,
                source_origin: MtlOrigin::new(0, 0, 0),
                source_size: MtlSize::new(shape.size.0, shape.size.1, 1),
                destination_texture: to,
                destination_slice: 0,
                destination_level: 0,
                destination_origin: MtlOrigin::new(0, 0, 0),
            }),
        }
    }
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    let platform = std::any::type_name::<P>();
    assert!(cb.error().is_none(), "{platform}: {:?}", cb.error());
    drop(keep);
    reads.iter().map(|(b, n)| bytes_of::<P>(b, *n)).collect()
}

/// `bytes` written into the whole of a Shared single-sample texture by `replaceRegion`.
pub(super) fn replace<P: Platform>(t: &P::Texture, shape: Shape, bytes: &[u8]) {
    let (w, h) = shape.size;
    let row = w * u64::try_from(texel_bytes(shape.format)).unwrap();
    let region = MtlRegion::new(MtlOrigin::new(0, 0, 0), MtlSize::new(w, h, 1));
    t.replace_region(region, 0, 0, HostRange::of_slice(bytes), row, row * h)
        .unwrap();
}

/// A pipeline state of `reach_vertex`, with `xr_constant_fragment` where `colours` names any
/// format (colour 0 of the first; the others take no output), of `samples` samples, with the
/// depth and stencil attachments' formats where set.
pub(super) fn state<P: Platform>(
    d: &P::Device,
    colours: &[u64],
    depth_stencil: (Option<u64>, Option<u64>),
    samples: u64,
) -> P::RenderPipeline {
    masked_state::<P>(d, colours, depth_stencil, (samples, None))
}

/// [`state`], every colour attachment's write mask `mask` where set.
pub(super) fn masked_state<P: Platform>(
    d: &P::Device,
    colours: &[u64],
    (depth, stencil): (Option<u64>, Option<u64>),
    (samples, mask): (u64, Option<u32>),
) -> P::RenderPipeline {
    let library = d.new_library(ROUNDING).unwrap().unwrap().unwrap();
    let function = |name| library.new_function(OnFault::Die, name).unwrap().unwrap();
    let functions = [function("reach_vertex"), function("xr_constant_fragment")];
    let elements: Vec<Vec<(u8, u32)>> = colours
        .iter()
        .enumerate()
        .map(|(i, &f)| {
            let mut e = vec![
                (0, u32::try_from(i).unwrap()),
                (1, u32::try_from(f).unwrap()),
            ];
            e.extend(mask.map(|m| (9, m)));
            e
        })
        .collect();
    let elements: Vec<&[(u8, u32)]> = elements.iter().map(Vec::as_slice).collect();
    let fragment = (!colours.is_empty()).then_some(&functions[1]);
    d.new_render_pipeline_state_full(&FullRenderPipelineDesc {
        colors: colours_of(&elements),
        depth_pixel_format: depth.unwrap_or(0),
        stencil_pixel_format: stencil.unwrap_or(0),
        raster_sample_count: samples,
        ..FullRenderPipelineDesc::new(&functions[0], fragment)
    })
    .unwrap()
    .unwrap()
}

/// A depth-stencil state that writes depth (`Always`, written) or stencil (`Always`, the
/// reference value on pass, write mask 0xff), as `aspect` says; none for a colour.
pub(super) fn writing<P: Platform>(d: &P::Device, aspect: Aspect) -> Option<P::DepthStencilState> {
    /// `MTLCompareFunctionAlways` and `MTLStencilOperationReplace`.
    const ALWAYS: u64 = 7;
    const REPLACE: u64 = 2;
    let stencil = StencilDesc {
        stencil_compare_function: ALWAYS,
        stencil_failure_operation: 0,
        depth_failure_operation: 0,
        depth_stencil_pass_operation: REPLACE,
        read_mask: 0xff,
        write_mask: 0xff,
    };
    let desc = match aspect {
        Aspect::Colour => return None,
        Aspect::Depth => DepthStencilDesc {
            depth_compare_function: Some(ALWAYS),
            depth_write_enabled: Some(true),
            ..DepthStencilDesc::default()
        },
        Aspect::Stencil => DepthStencilDesc {
            front: Some(stencil),
            back: Some(stencil),
            ..DepthStencilDesc::default()
        },
    };
    Some(d.new_depth_stencil_state(&desc).unwrap().unwrap())
}

/// The float4 clip-space positions of `pixels` over a target of `size`, depth 0.5, as
/// `reach_vertex` reads them.
pub(super) fn positions(pixels: &[(f32, f32)], (w, h): (u64, u64)) -> Vec<u8> {
    #[expect(
        clippy::as_conversions,
        reason = "a target's side, at most 16384, as an f32"
    )]
    let (w, h) = (w as f32, h as f32);
    pixels
        .iter()
        .flat_map(|&(x, y)| [x / w * 2.0 - 1.0, 1.0 - y / h * 2.0, 0.5, 1.0])
        .flat_map(f32::to_le_bytes)
        .collect()
}

/// The triangle whose pixels lie inside tile (1, 1) of a 32 by 32 grid: corners (36, 36),
/// (56, 36) and (36, 56), 190 pixel centres.
pub(super) const IN_TILE_1_1: [(f32, f32); 3] = [(36.0, 36.0), (56.0, 36.0), (36.0, 56.0)];

/// Triangles, each its three corners in pixels.
pub(super) type Triangles<'a> = &'a [&'a [(f32, f32)]];

/// Whether the pixel at `(x, y)` lies within 1.5 pixels of the triangle `t` (pixel corners):
/// a pixel a draw of it may have written.
pub(super) fn near(t: &[(f32, f32)], (x, y): (usize, usize)) -> bool {
    #[expect(
        clippy::as_conversions,
        reason = "a pixel's coordinates, at most 16384, as f32"
    )]
    let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
    let area = (t[1].0 - t[0].0) * (t[2].1 - t[0].1) - (t[1].1 - t[0].1) * (t[2].0 - t[0].0);
    (0..3).all(|i| {
        let (a, b) = (t[i], t[(i + 1) % 3]);
        let (ex, ey) = (b.0 - a.0, b.1 - a.1);
        let cross = ex * (py - a.1) - ey * (px - a.0);
        cross * area.signum() / ex.hypot(ey) >= -1.5
    })
}

/// Each tile of a `tile`-sized grid over a texture of `size` and `texel` bytes, as a letter, rows
/// separated by `/`: `K` where every texel of `after` is `before`'s, `F` where none is but those
/// `fill` also is (each other texel `fill` or what a draw wrote), and `?` elsewhere. A texel near
/// one of the `drawn` triangles ([`near`]) whose draw may have written `before`'s value again
/// counts as written.
pub(super) fn tiles(
    (before, after): (&[u8], &[u8]),
    (fill, drawn): (&[u8], Triangles<'_>),
    (size, tile): ((u64, u64), (u64, u64)),
) -> String {
    let texel = fill.len();
    let (w, h) = (
        usize::try_from(size.0).unwrap(),
        usize::try_from(size.1).unwrap(),
    );
    let (tw, th) = (
        usize::try_from(tile.0).unwrap(),
        usize::try_from(tile.1).unwrap(),
    );
    let mut grid = Vec::new();
    for ty in 0..h.div_ceil(th) {
        let mut row = String::new();
        for tx in 0..w.div_ceil(tw) {
            let (mut kept, mut kept_alone, mut total) = (0, 0, 0);
            for y in ty * th..((ty + 1) * th).min(h) {
                for x in tx * tw..((tx + 1) * tw).min(w) {
                    let at = (y * w + x) * texel;
                    let (b, a) = (&before[at..at + texel], &after[at..at + texel]);
                    total += 1;
                    if a == b {
                        kept += 1;
                        if a != fill && !drawn.iter().any(|t| near(t, (x, y))) {
                            kept_alone += 1;
                        }
                    }
                }
            }
            row.push(match () {
                () if kept == total => 'K',
                () if kept_alone == 0 => 'F',
                () => '?',
            });
        }
        grid.push(row);
    }
    grid.join("/")
}

/// A grid of `size` in `tile`s, `F` at `filled`, `K` elsewhere, as [`tiles`] writes one.
pub(super) fn grid(size: (u64, u64), tile: (u64, u64), filled: &[(u64, u64)]) -> String {
    (0..size.1.div_ceil(tile.1))
        .map(|y| {
            (0..size.0.div_ceil(tile.0))
                .map(|x| if filled.contains(&(x, y)) { 'F' } else { 'K' })
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("/")
}

/// Answers `probe` on Metal [`RUNS`] times and requires every run's answer to be the first's,
/// which it returns: Metal's answer is one value.
pub(super) fn once<T: PartialEq + std::fmt::Debug>(what: &str, mut probe: impl FnMut() -> T) -> T {
    let first = probe();
    for run in 1..RUNS {
        let again = probe();
        assert!(
            again == first,
            "{what}: Metal's run {run} differs from its first"
        );
    }
    first
}

/// Whether the runtime announced `kind` in this process. It announces a relayed kind once per
/// device (`crate::platform::device_options`), so a test checks what the process heard.
pub(super) fn heard(kind: metal2vulkan_runtime::Degradation) -> bool {
    crate::answer::RELAYED.lock().contains(&kind)
}

mod dont_care_load;
mod extended_range;
mod heap_layout;
mod unreached_tiles;
