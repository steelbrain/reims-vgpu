//! The comparisons of the runtime's textures of Apple's eleven RGBX formats with Metal's on this
//! host. The runtime holds each in its RGBA twin (reads take alpha 1, GPU writes
//! store the green field's bits in the fourth field, Metal's X), as it measured on an Apple M4
//! under macOS 27.0.1; these tests hold its answers to the Metal platform's here, byte for byte:
//! - what kernels read, sample and gather of random bytes, in bounds and out, through each
//!   address mode and border colour, and through an argument buffer;
//! - what kernels write, in each of their types, and read and write;
//! - what `replaceRegion:`, blits and the private fills store, a colour fill in an RGBX format
//!   by the runtime's packer (`encode_blit`'s `colours_the_runtime_packs_are_the_metal_platforms`);
//! - `RGBX8Unorm_sRGB`'s storage reads and writes, and its RGBA twin's, at every code and around
//!   every step of Metal's twelve-bit encode table, which the runtime converts through those
//!   tables.
//!
//! A half read's or sample's NaN bits follow the device, which the runtime announces
//! (`device-nan-encoding`): the dispatch dies, naming the kind ([`crate::answer::DIES`]), and
//! each test here pins the reads that die. A half write keeps its NaN's bits, as Metal's does,
//! through the texel's unsigned integer view, which
//! [`half_nan_writes_are_metals`] holds.
//!
//! What the runtime refuses dies here, named: a pass into such a texture, with `RenderTarget`
//! usage or without; the mipmaps of one of two levels; a texture buffer a kernel reads; and a
//! write through an argument buffer. Metal's side of each runs first.
//! The kernels are `rgbx.metal`'s, each at its thread's texel of a 16 by 16 texture.
//!
//! The same readers and writers hold the runtime's textures of the three bfloat formats to
//! Metal's ([`bfloat_textures_read_and_write_as_metals`]), and the readers its textures of the
//! single-plane extended-range formats, swizzled and not
//! ([`extended_range_textures_read_as_metals`]).

use metal2vulkan_runtime as rt;
use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder as _, ResourceRef};
use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
use reims_vgpu_core::platform::compute::{ComputeCmd, ComputeEncoder as _};
use reims_vgpu_core::platform::descriptor::RenderPassDescriptor as _;
use reims_vgpu_core::platform::device::GpuDevice as _;
use reims_vgpu_core::platform::render::{RenderCmd, RenderEncoder as _};
use reims_vgpu_core::platform::resource::{Buffer as _, Library as _, Texture as _};
use reims_vgpu_core::platform::types::{
    ClearColor, MtlOrigin, MtlRegion, MtlSize, NsRange, Swizzle,
};
use reims_vgpu_core::platform::values::{ColorAttachmentSet, SamplerDesc, TextureDesc};
use reims_vgpu_core::platform::{HostRange, Platform};
use reims_vgpu_core::raise::OnFault;
use reims_vgpu_core::raise::death::{DeathKind, DeathSite};
use reims_vgpu_core::testkit::capture::catch_death;
use reims_vgpu_metal::MetalPlatform;

use crate::gpu::device::tests::dev;
use crate::gpu::encode_render::tests::metal_tests::memoryless_fetched;
use crate::gpu::ycbcr::measure::sampler;
use crate::platform::tests::platform;

/// The library of `rgbx.metal`.
const LIB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rgbx.metallib"));

/// A texture's width and height.
const SIDE: u64 = 16;

/// The texels of a texture, and the threads of a dispatch.
const TEXELS: usize = 256;

/// `MTLResourceStorageModeShared`.
const SHARED: u64 = 0;

/// `MTLResourceStorageModePrivate`.
const PRIVATE: u64 = 0x20;

/// `MTLTextureUsageShaderRead`.
const READ: u64 = 0x1;

/// `MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite`.
const READ_WRITE: u64 = 0x3;

/// `MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget`.
const RENDER: u64 = 0x5;

/// How a format's texels read in a kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Class {
    /// As floats, or halves.
    Float,
    /// As unsigned integers.
    Uint,
    /// As signed integers.
    Int,
}

/// Each RGBX format: its raw value, its name, its texel's bytes, how kernels read it, and its
/// RGBA twin's raw value.
const FORMATS: [(u64, &str, usize, Class, u64); 11] = [
    (300, "RGBX8Unorm", 4, Class::Float, 70),
    (301, "RGBX8Unorm_sRGB", 4, Class::Float, 71),
    (302, "RGBX8Snorm", 4, Class::Float, 72),
    (303, "RGBX8Uint", 4, Class::Uint, 73),
    (304, "RGBX8Sint", 4, Class::Int, 74),
    (313, "RGBX16Uint", 8, Class::Uint, 113),
    (314, "RGBX16Sint", 8, Class::Int, 114),
    (315, "RGBX16Float", 8, Class::Float, 115),
    (323, "RGBX32Uint", 16, Class::Uint, 123),
    (324, "RGBX32Sint", 16, Class::Int, 124),
    (325, "RGBX32Float", 16, Class::Float, 125),
];

/// `RGBX8Unorm_sRGB`, whose storage slots convert sRGB in the shader
/// ([`srgb_storage_reads_and_writes_are_metals`]).
const SRGB_X: u64 = 301;

/// `RGBA8Unorm_sRGB`, [`SRGB_X`]'s RGBA twin.
const SRGB_RGBA: u64 = 71;

/// The usages of the textures the stores write: `ShaderRead` and `ShaderWrite`, and for
/// [`SRGB_X`]'s also `ShaderRead` alone, so its textures a function may write and its others are
/// both held to Metal's.
fn stored_usages(format: u64) -> &'static [u64] {
    if format == SRGB_X {
        &[READ_WRITE, READ]
    } else {
        &[READ_WRITE]
    }
}

/// The formats Metal renders into: passes into the others' textures run no
/// pipeline on Metal here.
const RENDERED: [u64; 3] = [300, 315, 325];

/// The samplers the sampling kernels read through: the filter (0 nearest, 1 linear), the
/// address mode on both axes (0 clamp to edge, 2 repeat, 4 clamp to zero, 5 clamp to the border
/// colour) and the border colour (0 transparent black, 1 opaque black, 2 opaque white).
const SAMPLERS: [(u64, u64, u64); 8] = [
    (0, 0, 0),
    (1, 0, 0),
    (1, 2, 0),
    (1, 4, 0),
    (1, 5, 0),
    (1, 5, 1),
    (1, 5, 2),
    (0, 5, 2),
];

/// The samplers the gathering kernel reads through, as [`SAMPLERS`].
const GATHERS: [(u64, u64, u64); 3] = [(0, 0, 0), (0, 5, 0), (0, 5, 2)];

/// The region the stores and copies write: 9 by 7 texels from (3, 5).
fn region() -> MtlRegion {
    MtlRegion::new(MtlOrigin::new(3, 5, 0), MtlSize::new(9, 7, 1))
}

/// A deterministic stream of random words (xorshift64*).
struct Rng(u64);

impl Rng {
    /// The next word.
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// `n` random bytes.
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next().to_le_bytes()[7]).collect()
    }

    /// A random word of 32 bits.
    fn word(&mut self) -> u32 {
        u32::try_from(self.next() >> 32).unwrap()
    }

    /// A random value in [0, 1), in steps of 2^-16.
    fn unit(&mut self) -> f32 {
        f32::from(u16::try_from(self.next() >> 48).unwrap()) / 65536.0
    }
}

/// `values`' bytes, each little-endian.
fn le(values: impl IntoIterator<Item = u32>) -> Vec<u8> {
    values.into_iter().flat_map(u32::to_le_bytes).collect()
}

/// The normalized coordinates the sampling kernels read at, two floats a thread: every fourth a
/// random texel's centre, the others random in [-0.25, 1.25), past the edges.
fn coordinates() -> Vec<u8> {
    let mut rng = Rng(0x6a09_e667_f3bc_c908);
    let side = f32::from(u8::try_from(SIDE).unwrap());
    le((0..TEXELS).flat_map(|i| {
        let pair = if i % 4 == 0 {
            [0, 1].map(|_| (f32::from(u8::try_from(rng.next() % SIDE).unwrap()) + 0.5) / side)
        } else {
            [0, 1].map(|_| rng.unit() * 1.5 - 0.25)
        };
        pair.map(f32::to_bits)
    }))
}

/// Float values that reach the formats' edges: zeroes, ones, NaNs, infinities, a denormal, the
/// half float's largest and past it, the sRGB curve's linear end and halfway points of 8 bits.
const SPECIALS: [f32; 24] = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.5,
    f32::NAN,
    -f32::NAN,
    f32::INFINITY,
    f32::NEG_INFINITY,
    1e-40,
    65504.0,
    65520.0,
    70000.0,
    1e30,
    -1e30,
    0.003_130_8,
    0.5 / 255.0,
    1.5 / 255.0,
    127.5 / 255.0,
    254.5 / 255.0,
    -0.5 / 127.0,
    -127.5 / 127.0,
    1.000_000_1,
    -2.0,
];

/// Whether `half`'s bits are a NaN.
fn half_nan(half: u16) -> bool {
    half & 0x7c00 == 0x7c00 && half & 0x03ff != 0
}

/// The values a writer of `kernel` writes into a format whose channels have `bits` bits, as its
/// buffer's bytes, four a texel: for floats, [`SPECIALS`] first, then random bits and random
/// values in [-2, 2) in turn; for halves, random bits, where `nans` is false each NaN made the
/// infinity of its sign ([`half_nan_writes_are_metals`]); for integers, random words and
/// random values of the channel's range in turn.
fn values(kernel: &str, bits: u32, seed: u64, nans: bool) -> Vec<u8> {
    let mut rng = Rng(seed);
    let lanes = TEXELS * 4;
    if kernel.ends_with("half") {
        return (0..lanes)
            .map(|_| u16::try_from(rng.next() >> 48).unwrap())
            .map(|h| if nans || !half_nan(h) { h } else { h & 0xfc00 })
            .flat_map(u16::to_le_bytes)
            .collect();
    }
    le((0..lanes).map(|i| {
        let raw = rng.word();
        let ranged = i % 8 >= 4;
        if kernel.ends_with("float") {
            match SPECIALS.get(i) {
                Some(v) => v.to_bits(),
                None if ranged => (rng.unit() * 4.0 - 2.0).to_bits(),
                None => raw,
            }
        } else if !ranged || bits == 32 {
            raw
        } else if kernel.ends_with("uint") {
            raw & ((1 << bits) - 1)
        } else {
            let shift = 32 - bits;
            (((raw << shift).cast_signed()) >> shift).cast_unsigned()
        }
    }))
}

/// The floats `rgbx_store_float` writes into an 8-bit sRGB format, four a texel. Into the colour
/// fields: the five floats nearest each half step `(i + 0.5) / 4095` between steps of Metal's
/// twelve-bit encode table, then [`SPECIALS`], then random values in [-0.25, 1.25). Into the
/// fourth field, in turn: the three floats nearest each half step `(k + 0.5) / 255` between an
/// 8-bit unorm's codes, alpha's.
fn srgb_floats() -> Vec<u8> {
    let near = |centre: f64, n: usize| {
        let c = centre as f32;
        let (mut below, mut above) = (vec![c], Vec::new());
        for _ in 0..n {
            below.push(below.last().copied().unwrap_or(c).next_down());
            above.push(above.last().copied().unwrap_or(c).next_up());
        }
        below.into_iter().rev().chain(above).collect::<Vec<f32>>()
    };
    let mut colour: Vec<f32> = (0..4096)
        .flat_map(|i| near((f64::from(i) + 0.5) / 4095.0, 2))
        .collect();
    colour.extend(SPECIALS);
    let mut rng = Rng(0x510e_527f_ade6_82d1);
    colour.extend((0..3072).map(|_| rng.unit() * 1.5 - 0.25));
    let alpha: Vec<f32> = (0..256)
        .flat_map(|k| near((f64::from(k) + 0.5) / 255.0, 1))
        .collect();
    let texels = colour.len().div_ceil(3);
    le((0..texels).flat_map(|t| {
        let lane = |l: usize| colour.get(3 * t + l).copied().unwrap_or(0.0);
        [lane(0), lane(1), lane(2), alpha[t % alpha.len()]].map(f32::to_bits)
    }))
}

/// The halves `rgbx_store_half` writes into an 8-bit sRGB format, four a texel: every half in
/// turn into the colour fields, NaNs among them, and the texel's index into the fourth.
fn every_half() -> Vec<u8> {
    let texels = (1usize << 16).div_ceil(3);
    (0..texels)
        .flat_map(|t| {
            let lane = |l: usize| u16::try_from((3 * t + l) % (1 << 16)).unwrap();
            [lane(0), lane(1), lane(2), u16::try_from(t).unwrap()]
        })
        .flat_map(u16::to_le_bytes)
        .collect()
}

/// One platform's device, a queue of it, and the kernels' library.
struct Gpu<'a, P: Platform> {
    /// The platform, which makes the descriptors.
    p: &'a P,
    /// The device.
    d: &'a P::Device,
    /// A queue of the device.
    q: P::CommandQueue,
    /// `rgbx.metal`'s library.
    lib: P::Library,
}

impl<'a, P: Platform> Gpu<'a, P> {
    /// `d`'s queue and library.
    fn new(p: &'a P, d: &'a P::Device) -> Gpu<'a, P> {
        let q = d.new_command_queue().unwrap();
        let lib = d.new_library(LIB).unwrap().unwrap().unwrap();
        Gpu { p, d, q, lib }
    }

    /// A Shared buffer holding `bytes`.
    fn buffer(&self, bytes: &[u8]) -> P::Buffer {
        let length = u64::try_from(bytes.len()).unwrap().max(16);
        let b = self
            .d
            .new_buffer_of_length(OnFault::Die, length, SHARED)
            .unwrap()
            .unwrap();
        // SAFETY: a Shared buffer of at least `bytes.len()` bytes, which no command uses yet.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), b.contents().as_ptr::<u8>(), bytes.len());
        }
        b
    }

    /// The first `n` bytes of `b`, a Shared buffer the work that wrote it completed.
    fn bytes_of(b: &P::Buffer, n: usize) -> Vec<u8> {
        // SAFETY: a Shared buffer of at least `n` bytes, read after its work completed.
        unsafe { std::slice::from_raw_parts(b.contents().as_ptr::<u8>(), n) }.to_vec()
    }

    /// Runs what `encode` encodes in a command buffer of its own, and waits for it.
    fn run(&self, encode: impl FnOnce(&P::CommandBuffer)) {
        let cb = self.q.command_buffer(OnFault::Die).unwrap().unwrap();
        encode(&cb);
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        assert_eq!(cb.error(), None);
    }

    /// Runs `cmds` in a blit encoder.
    fn blit(&self, cmds: Vec<BlitCmd<'_, P>>) {
        self.run(|cb| {
            let e = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
            for cmd in cmds {
                e.encode(OnFault::Die, cmd).unwrap();
            }
            e.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
        });
    }

    /// A 16 by 16 texture of `format`, of `usage` and `levels` levels, Shared or Private
    /// (`options`).
    fn texture_of(&self, format: u64, usage: u64, options: u64, levels: u64) -> P::Texture {
        let desc = self.p.texture_descriptor(&TextureDesc {
            pixel_format: format,
            width: SIDE,
            height: SIDE,
            mipmap_level_count: levels,
            usage,
            resource_options: Some(options),
            ..TextureDesc::default()
        });
        self.d.new_texture(OnFault::Die, &desc).unwrap().unwrap()
    }

    /// [`Gpu::texture_of`] of one level, holding `bytes`, a row of 16 texels after another:
    /// written by `replaceRegion:` where it is Shared, by a blit from a buffer where it is
    /// Private.
    fn texture(&self, format: u64, usage: u64, options: u64, bytes: &[u8]) -> P::Texture {
        let t = self.texture_of(format, usage, options, 1);
        let row = u64::try_from(bytes.len()).unwrap() / SIDE;
        let all = MtlSize::new(SIDE, SIDE, 1);
        if options == SHARED {
            let region = MtlRegion::new(MtlOrigin::new(0, 0, 0), all);
            t.replace_region(region, 0, 0, HostRange::of_slice(bytes), row, 0)
                .unwrap();
        } else {
            let b = self.buffer(bytes);
            self.blit(vec![BlitCmd::CopyBufferToTexture {
                source_buffer: &b,
                source_offset: 0,
                source_bytes_per_row: row,
                source_bytes_per_image: row * SIDE,
                source_size: all,
                destination_texture: &t,
                destination_slice: 0,
                destination_level: 0,
                destination_origin: MtlOrigin::new(0, 0, 0),
            }]);
        }
        t
    }

    /// A 16 by 16 texture of `format` and `ShaderRead` usage over a Shared buffer, holding
    /// `bytes`, a row of 16 texels after another, its rows 256 bytes apart in the buffer.
    fn over_buffer(&self, format: u64, bytes: &[u8]) -> P::Texture {
        const PITCH: usize = 256;
        let row = bytes.len() / TEXELS * 16;
        let mut laid = vec![0; PITCH * 16];
        for (y, texels) in bytes.chunks(row).enumerate() {
            laid[y * PITCH..y * PITCH + row].copy_from_slice(texels);
        }
        let desc = self.p.texture_descriptor(&TextureDesc {
            pixel_format: format,
            width: SIDE,
            height: SIDE,
            usage: READ,
            resource_options: Some(SHARED),
            ..TextureDesc::default()
        });
        let b = self.buffer(&laid);
        let pitch = u64::try_from(PITCH).unwrap();
        b.new_texture(OnFault::Die, &desc, 0, pitch)
            .unwrap()
            .unwrap()
    }

    /// A Private 16 by 16 texture of `format` and `ShaderRead` usage under `swizzle`, holding
    /// `bytes` ([`Gpu::texture`]'s), written by a blit from a buffer.
    fn swizzled(&self, format: u64, swizzle: Swizzle, bytes: &[u8]) -> P::Texture {
        let desc = self.p.texture_descriptor(&TextureDesc {
            pixel_format: format,
            width: SIDE,
            height: SIDE,
            usage: READ,
            resource_options: Some(PRIVATE),
            swizzle: Some(swizzle),
            ..TextureDesc::default()
        });
        let t = self.d.new_texture(OnFault::Die, &desc).unwrap().unwrap();
        let row = u64::try_from(bytes.len()).unwrap() / SIDE;
        let b = self.buffer(bytes);
        self.blit(vec![BlitCmd::CopyBufferToTexture {
            source_buffer: &b,
            source_offset: 0,
            source_bytes_per_row: row,
            source_bytes_per_image: row * SIDE,
            source_size: MtlSize::new(SIDE, SIDE, 1),
            destination_texture: &t,
            destination_slice: 0,
            destination_level: 0,
            destination_origin: MtlOrigin::new(0, 0, 0),
        }]);
        t
    }

    /// A view of `format` under `swizzle` of an unswizzled Private texture of `format` holding
    /// `bytes` ([`Gpu::texture`]).
    fn view(&self, format: u64, swizzle: Swizzle, bytes: &[u8]) -> P::Texture {
        let plain = self.texture(format, READ, PRIVATE, bytes);
        let whole = NsRange::new(0, 1);
        let view = plain.new_swizzled_texture_subview(format, 2, whole, whole, swizzle);
        view.unwrap().unwrap()
    }

    /// The bytes of `t`'s 16 by 16 texels of `texel` bytes, by a blit into a buffer.
    fn read(&self, t: &P::Texture, texel: usize) -> Vec<u8> {
        let n = TEXELS * texel;
        let b = self.buffer(&vec![0; n]);
        let row = SIDE * u64::try_from(texel).unwrap();
        self.blit(vec![BlitCmd::CopyTextureToBuffer {
            source_texture: t,
            source_slice: 0,
            source_level: 0,
            source_origin: MtlOrigin::new(0, 0, 0),
            source_size: MtlSize::new(SIDE, SIDE, 1),
            destination_buffer: &b,
            destination_offset: 0,
            destination_bytes_per_row: row,
            destination_bytes_per_image: row * SIDE,
        }]);
        Self::bytes_of(&b, n)
    }

    /// A dispatch of `kernel` over the 16 by 16 grid: `texture` at texture 0, `buffers` from
    /// buffer 0, `sampler` at sampler 0, and `held` made resident for its usage.
    fn dispatch(
        &self,
        kernel: &str,
        texture: Option<&P::Texture>,
        buffers: &[&P::Buffer],
        sampler: Option<&P::SamplerState>,
        held: Option<(&P::Texture, u64)>,
    ) {
        let f = self
            .lib
            .new_function(OnFault::Die, kernel)
            .unwrap()
            .unwrap();
        let state = self
            .d
            .new_compute_pipeline_state_for_function(Some(&f))
            .unwrap()
            .unwrap();
        let bound: Vec<_> = buffers.iter().map(|&b| Some(b)).collect();
        let offsets = vec![0; bound.len()];
        let samplers = [sampler];
        let resident = held.map(|(t, usage)| ([Some(ResourceRef::Texture(t))], usage));
        self.run(|cb| {
            let e = cb.compute_encoder(OnFault::Die).unwrap().unwrap();
            let mut cmds = vec![
                ComputeCmd::SetComputePipelineState {
                    state: Some(&state),
                },
                ComputeCmd::SetBuffers {
                    buffers: &bound,
                    offsets: &offsets,
                    range: NsRange::new(0, u64::try_from(bound.len()).unwrap()),
                },
            ];
            if let Some(t) = texture {
                cmds.push(ComputeCmd::SetTexture {
                    texture: Some(t),
                    index: 0,
                });
            }
            if sampler.is_some() {
                cmds.push(ComputeCmd::SetSamplerStates {
                    samplers: &samplers,
                    range: NsRange::new(0, 1),
                });
            }
            if let Some((resources, usage)) = &resident {
                cmds.push(ComputeCmd::UseResources {
                    resources,
                    usage: *usage,
                });
            }
            cmds.push(ComputeCmd::DispatchThreads {
                threads_per_grid: MtlSize::new(SIDE, SIDE, 1),
                threads_per_threadgroup: MtlSize::new(8, 8, 1),
            });
            cmds.push(ComputeCmd::EndEncoding);
            for cmd in cmds {
                e.encode(OnFault::Die, cmd).unwrap();
            }
        });
    }

    /// What `kernel` writes into a buffer of `bytes` bytes a thread, `inputs` after it, over
    /// `texture`, through `sampler`.
    fn output(
        &self,
        kernel: &str,
        texture: &P::Texture,
        bytes: usize,
        inputs: &[&P::Buffer],
        sampler: Option<&P::SamplerState>,
    ) -> Vec<u8> {
        let out = self.buffer(&vec![0; TEXELS * bytes]);
        let buffers: Vec<_> = std::iter::once(&out)
            .chain(inputs.iter().copied())
            .collect();
        self.dispatch(kernel, Some(texture), &buffers, sampler, None);
        Self::bytes_of(&out, TEXELS * bytes)
    }

    /// Each reader's case and what it reads of `t`, of `class`: [`Gpu::direct_reads`], then
    /// [`Gpu::held_read`].
    fn reads(&self, class: Class, t: &P::Texture) -> Vec<(String, Vec<u8>)> {
        let mut out = self.direct_reads(class, t);
        out.push(self.held_read(class, t));
        out
    }

    /// Each reader's case and what it reads of `t`, of `class`, bound to the kernel's texture
    /// slot: `read()` in each type at each texel and 8 texels down and across, past the edges;
    /// for floats also `sample()` as floats and halves through each of [`SAMPLERS`], and
    /// `gather()` of each component through each of [`GATHERS`].
    fn direct_reads(&self, class: Class, t: &P::Texture) -> Vec<(String, Vec<u8>)> {
        let mut out = Vec::new();
        let readers: &[(&str, usize)] = match class {
            Class::Float => &[("rgbx_read_float", 16), ("rgbx_read_half", 8)],
            Class::Uint => &[("rgbx_read_uint", 16)],
            Class::Int => &[("rgbx_read_int", 16)],
        };
        for &(kernel, bytes) in readers {
            for offset in [[0, 0], [8, 8]] {
                let offset_buffer = self.buffer(&le(offset));
                let read = self.output(kernel, t, bytes, &[&offset_buffer], None);
                out.push((format!("{kernel} at +{offset:?}"), read));
            }
        }
        if class != Class::Float {
            return out;
        }
        let at = self.buffer(&coordinates());
        for (filter, address, border) in SAMPLERS {
            let s = sampler::<P>(self.d, (filter, address, true), border);
            for (kernel, bytes) in [("rgbx_sample_float", 16), ("rgbx_sample_half", 8)] {
                let sampled = self.output(kernel, t, bytes, &[&at], Some(&s));
                let case = format!("{kernel} filter {filter} address {address} border {border}");
                out.push((case, sampled));
            }
        }
        for (filter, address, border) in GATHERS {
            let s = sampler::<P>(self.d, (filter, address, true), border);
            let gathered = self.output("rgbx_gather", t, 64, &[&at], Some(&s));
            let case = format!("rgbx_gather filter {filter} address {address} border {border}");
            out.push((case, gathered));
        }
        out
    }

    /// What `kernel` samples of `t`, `bytes` bytes a thread, at [`coordinates`] through a sampler
    /// of `filter` (0 nearest, 1 linear) that clamps to the edge.
    fn sampled(&self, kernel: &str, t: &P::Texture, bytes: usize, filter: u64) -> Vec<u8> {
        let at = self.buffer(&coordinates());
        let s = sampler::<P>(self.d, (filter, 0, true), 0);
        self.output(kernel, t, bytes, &[&at], Some(&s))
    }

    /// The case and what `read()` reads of `t`, of `class`, through an argument buffer that holds
    /// its ID, in the type of `class` (a `texture2d` of `float`, `uint` or `int`, as the format's
    /// type; Metal defines no read of a texture as another type), the texture made resident for
    /// reading.
    fn held_read(&self, class: Class, t: &P::Texture) -> (String, Vec<u8>) {
        let kernel = match class {
            Class::Float => "rgbx_read_held_float",
            Class::Uint => "rgbx_read_held_uint",
            Class::Int => "rgbx_read_held_int",
        };
        let held = self.buffer(&t.gpu_resource_id().to_le_bytes());
        let read = self.buffer(&[0; TEXELS * 16]);
        self.dispatch(kernel, None, &[&read, &held], None, Some((t, 1)));
        (kernel.to_owned(), Self::bytes_of(&read, TEXELS * 16))
    }

    /// Each case and what it reads of `t`, of a format read as floats, through an argument
    /// buffer that holds its ID ([`Gpu::held_read`]): `read()` as floats and as halves, and its
    /// width, height and level count.
    fn held_float_reads(&self, t: &P::Texture) -> Vec<(String, Vec<u8>)> {
        let held = self.buffer(&t.gpu_resource_id().to_le_bytes());
        let mut out = vec![self.held_read(Class::Float, t)];
        for (kernel, bytes) in [("rgbx_read_held_half", 8), ("rgbx_query_held", 16)] {
            let read = self.buffer(&vec![0; TEXELS * bytes]);
            self.dispatch(kernel, None, &[&read, &held], None, Some((t, 1)));
            out.push((kernel.to_owned(), Self::bytes_of(&read, TEXELS * bytes)));
        }
        out
    }

    /// Each case and what it reads of `t`, of a format read as floats, through an argument
    /// buffer that holds its ID, at [`coordinates`], through a sampler at sampler slot 0 that
    /// clamps to the edge: `sample()`, and `gather()` of red, through a nearest and a linear
    /// sampler of no mip filter; or, where `mipmapped`, through a linear sampler of a linear mip
    /// filter alone.
    fn held_samples(&self, t: &P::Texture, mipmapped: bool) -> Vec<(String, Vec<u8>)> {
        let held = self.buffer(&t.gpu_resource_id().to_le_bytes());
        let at = self.buffer(&coordinates());
        // Each sampler's filter and mip filter (`MTLSamplerMipFilterLinear` 2).
        let samplers: &[(u64, u64)] = if mipmapped {
            &[(1, 2)]
        } else {
            &[(0, 0), (1, 0)]
        };
        let mut out = Vec::new();
        for &(filter, mip) in samplers {
            let desc = SamplerDesc {
                min_filter: Some(filter),
                mag_filter: Some(filter),
                mip_filter: Some(mip),
                s_address_mode: Some(0),
                t_address_mode: Some(0),
                normalized_coordinates: Some(true),
                ..SamplerDesc::default()
            };
            let s = self.d.new_sampler_state(&desc).unwrap().unwrap();
            for kernel in ["rgbx_sample_held", "rgbx_gather_held"] {
                let read = self.buffer(&[0; TEXELS * 16]);
                self.dispatch(kernel, None, &[&read, &held, &at], Some(&s), Some((t, 1)));
                let case = format!("{kernel} filter {filter} mip filter {mip}");
                out.push((case, Self::bytes_of(&read, TEXELS * 16)));
            }
        }
        out
    }

    /// Each writer's case and the bytes of a texture of `format` (of `texel` bytes, `class`, of
    /// `options`) holding `bytes` once it wrote [`values`] of `seed` into two of every three
    /// texels; and what `read_write`'s kernel read and left.
    fn writes(
        &self,
        (format, texel, class): (u64, usize, Class),
        options: u64,
        bytes: &[u8],
        seed: u64,
    ) -> Vec<(String, Vec<u8>)> {
        let bits = u32::try_from(texel * 2).unwrap();
        let writers: &[&str] = match class {
            Class::Float => &["rgbx_write_float", "rgbx_write_half"],
            Class::Uint => &["rgbx_write_uint"],
            Class::Int => &["rgbx_write_int"],
        };
        let mut out = Vec::new();
        for &kernel in writers {
            let t = self.texture(format, READ_WRITE, options, bytes);
            let v = self.buffer(&values(kernel, bits, seed, false));
            self.dispatch(kernel, Some(&t), &[&v], None, None);
            out.push((kernel.to_owned(), self.read(&t, texel)));
        }
        let kernel = match class {
            Class::Float => "rgbx_read_write_float",
            Class::Uint => "rgbx_read_write_uint",
            Class::Int => "rgbx_read_write_int",
        };
        let t = self.texture(format, READ_WRITE, options, bytes);
        let read = self.output(kernel, &t, 16, &[], None);
        out.push((format!("{kernel} reads"), read));
        out.push((format!("{kernel} leaves"), self.read(&t, texel)));
        out
    }

    /// The bytes of a texture of `format` (of `texel` bytes, of `options`) holding `bytes` once a
    /// kernel's `write()` of halves wrote [`values`] of `seed`, NaNs among them, into two of every
    /// three texels.
    fn half_writes(
        &self,
        (format, texel): (u64, usize),
        options: u64,
        bytes: &[u8],
        seed: u64,
    ) -> Vec<u8> {
        let t = self.texture(format, READ_WRITE, options, bytes);
        let v = self.buffer(&values("rgbx_write_half", 16, seed, true));
        self.dispatch("rgbx_write_half", Some(&t), &[&v], None, None);
        self.read(&t, texel)
    }

    /// Each case of the storage sweep of a texture of `format`, an 8-bit sRGB format, of
    /// `ShaderRead` and `ShaderWrite` usage and `options`, and what it read or left. Of a texture
    /// holding every code in each colour field (texel `t`'s fields `t`, `255 - t`, `7t + 3` and
    /// `13t`, of 256): each of [`Gpu::reads`]' readers, and `read()` through `read_write` as
    /// floats and as halves, with the texture its write-back of each texel's components
    /// reversed left. And the bytes of textures `write()` stored [`srgb_floats`] and
    /// [`every_half`] into, 256 texels a texture, every texel written.
    fn srgb_storage(&self, format: u64, options: u64) -> Vec<(String, Vec<u8>)> {
        let codes: Vec<u8> = (0..=255u8)
            .flat_map(|t| {
                [
                    t,
                    255 - t,
                    t.wrapping_mul(7).wrapping_add(3),
                    t.wrapping_mul(13),
                ]
            })
            .collect();
        let t = self.texture(format, READ_WRITE, options, &codes);
        let mut out = self.reads(Class::Float, &t);
        for (kernel, bytes) in [("rgbx_read_write_float", 16), ("rgbx_read_write_half", 8)] {
            let t = self.texture(format, READ_WRITE, options, &codes);
            let read = self.output(kernel, &t, bytes, &[], None);
            out.push((format!("{kernel} reads"), read));
            out.push((format!("{kernel} leaves"), self.read(&t, 4)));
        }
        for (kernel, lane, values) in [
            ("rgbx_store_float", 4, srgb_floats()),
            ("rgbx_store_half", 2, every_half()),
        ] {
            for (chunk, v) in values.chunks(TEXELS * 4 * lane).enumerate() {
                let mut v = v.to_vec();
                v.resize(TEXELS * 4 * lane, 0);
                let t = self.texture_of(format, READ_WRITE, options, 1);
                let b = self.buffer(&v);
                self.dispatch(kernel, Some(&t), &[&b], None, None);
                out.push((
                    format!("{kernel} texels from {}", chunk * TEXELS),
                    self.read(&t, 4),
                ));
            }
        }
        out
    }

    /// Each store's case and the bytes of a texture of `format` (of `texel` bytes, of `usage` and
    /// `options`) holding `bytes` once it stored [`region`]: by `replaceRegion:` (a Shared
    /// texture's), a blit from a buffer, a blit from another texture holding `other`, a byte
    /// fill, and colour fills in the pixel format `twin`; and the bytes of a blit of the region
    /// into a buffer of rows 16 bytes longer.
    fn stores(
        &self,
        (format, texel, twin): (u64, usize, u64),
        (usage, options): (u64, u64),
        bytes: &[u8],
        other: &[u8],
    ) -> Vec<(String, Vec<u8>)> {
        let r = region();
        let texel_bytes = u64::try_from(texel).unwrap();
        let row = r.size.width * texel_bytes;
        let source = Rng(0x3c6e_f372_fe94_f82b).bytes(usize::try_from(row * 7).unwrap());
        let mut out = Vec::new();
        let mut stored = |case: &str, store: &dyn Fn(&P::Texture)| {
            let t = self.texture(format, usage, options, bytes);
            store(&t);
            out.push((case.to_owned(), self.read(&t, texel)));
        };
        if options == SHARED {
            stored("replaceRegion", &|t| {
                t.replace_region(r, 0, 0, HostRange::of_slice(&source), row, 0)
                    .unwrap();
            });
        }
        let from = self.buffer(&source);
        stored("a blit from a buffer", &|t| {
            self.blit(vec![BlitCmd::CopyBufferToTexture {
                source_buffer: &from,
                source_offset: 0,
                source_bytes_per_row: row,
                source_bytes_per_image: row * 7,
                source_size: r.size,
                destination_texture: t,
                destination_slice: 0,
                destination_level: 0,
                destination_origin: r.origin,
            }]);
        });
        let other = self.texture(format, usage, options, other);
        stored("a blit from a texture", &|t| {
            self.blit(vec![BlitCmd::CopyTextureRegion {
                source_texture: &other,
                source_slice: 0,
                source_level: 0,
                source_origin: MtlOrigin::new(2, 1, 0),
                source_size: r.size,
                destination_texture: t,
                destination_slice: 0,
                destination_level: 0,
                destination_origin: r.origin,
            }]);
        });
        let pattern: Vec<u8> = (0..16).map(|k| 0x80 + 7 * k).collect();
        stored("a byte fill", &|t| {
            self.blit(vec![BlitCmd::FillTexture {
                texture: t,
                level: 0,
                slice: 0,
                region: r,
                bytes: HostRange::of_slice(&pattern),
            }]);
        });
        let colors = [
            (0.25, 0.5, 0.75, 1.0),
            (0.1, 0.9, 0.3, 0.5),
            (1.5, -0.5, 7.0, 0.0),
        ];
        for (red, green, blue, alpha) in colors {
            let color = ClearColor {
                red,
                green,
                blue,
                alpha,
            };
            stored(&format!("a fill of {color:?} in {twin}"), &|t| {
                self.blit(vec![BlitCmd::FillTextureColor {
                    texture: t,
                    level: 0,
                    slice: 0,
                    region: r,
                    color,
                    pixel_format: twin,
                }]);
            });
        }
        let t = self.texture(format, usage, options, bytes);
        let padded = row + 16;
        let down = self.buffer(&vec![0xa5; usize::try_from(padded * 7).unwrap()]);
        self.blit(vec![BlitCmd::CopyTextureToBuffer {
            source_texture: &t,
            source_slice: 0,
            source_level: 0,
            source_origin: r.origin,
            source_size: r.size,
            destination_buffer: &down,
            destination_offset: 0,
            destination_bytes_per_row: padded,
            destination_bytes_per_image: padded * 7,
        }]);
        let n = usize::try_from(padded * 7).unwrap();
        out.push(("a blit to a buffer".to_owned(), Self::bytes_of(&down, n)));
        out
    }

    /// A pass into a texture of `format` of `usage` that clears it and stores it.
    fn pass(&self, format: u64, usage: u64) {
        drop(self.cleared(format, usage, [0.25, 0.5, 0.75, 1.0]));
    }

    /// A Private texture of `format` and `usage` once a pass cleared it to `colour` (red, green,
    /// blue, alpha) and stored it.
    fn cleared(&self, format: u64, usage: u64, [red, green, blue, alpha]: [f64; 4]) -> P::Texture {
        let t = self.texture_of(format, usage, PRIVATE, 1);
        let pass = self.p.render_pass_descriptor();
        let attachment = ColorAttachmentSet {
            texture: Some(Some(&t)),
            load_action: Some(2),
            store_action: Some(1),
            clear_color: Some(ClearColor {
                red,
                green,
                blue,
                alpha,
            }),
            ..ColorAttachmentSet::default()
        };
        pass.set_color_attachment(OnFault::Die, 0, &attachment)
            .unwrap();
        self.run(|cb| {
            let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
            e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
        });
        t
    }

    /// What a pass that clears a Private 16 by 16 4-sample texture of `format` (of `texel` bytes)
    /// and `RENDER` usage to `colour` (red, green, blue, alpha), stores it and resolves it into a
    /// Private one of the format leaves: the resolve, by a blit ([`Gpu::read`]), and each sample
    /// as a `float4`, by a kernel (`rgbx_read_samples`).
    fn multisample_cleared(
        &self,
        format: u64,
        texel: usize,
        [red, green, blue, alpha]: [f64; 4],
    ) -> (Vec<u8>, Vec<u8>) {
        /// `MTLStoreActionStoreAndMultisampleResolve`.
        const STORE_AND_RESOLVE: u64 = 3;
        let desc = self.p.texture_descriptor(&TextureDesc {
            texture_type: 4,
            sample_count: 4,
            pixel_format: format,
            width: SIDE,
            height: SIDE,
            usage: RENDER,
            resource_options: Some(PRIVATE),
            ..TextureDesc::default()
        });
        let samples = self.d.new_texture(OnFault::Die, &desc).unwrap().unwrap();
        let target = self.texture_of(format, RENDER, PRIVATE, 1);
        let pass = self.p.render_pass_descriptor();
        let attachment = ColorAttachmentSet {
            texture: Some(Some(&samples)),
            resolve_texture: Some(Some(&target)),
            load_action: Some(2),
            store_action: Some(STORE_AND_RESOLVE),
            clear_color: Some(ClearColor {
                red,
                green,
                blue,
                alpha,
            }),
            ..ColorAttachmentSet::default()
        };
        pass.set_color_attachment(OnFault::Die, 0, &attachment)
            .unwrap();
        self.run(|cb| {
            let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
            e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
        });
        let resolved = self.read(&target, texel);
        (
            resolved,
            self.output("rgbx_read_samples", &samples, 64, &[], None),
        )
    }

    /// The mipmaps of a Private texture of `format` of two levels.
    fn mipmaps(&self, format: u64) {
        let t = self.texture_of(format, READ, PRIVATE, 2);
        self.blit(vec![BlitCmd::GenerateMipmaps { texture: Some(&t) }]);
    }

    /// A read of each texel of a texture buffer of `format` (`texel` bytes, `class`) over a
    /// buffer of random bytes.
    fn texture_buffer(&self, format: u64, texel: usize, class: Class) {
        let bytes = Rng(format).bytes(TEXELS * texel);
        let b = self.buffer(&bytes);
        let desc = self.p.texture_descriptor(&TextureDesc {
            texture_type: 9,
            pixel_format: format,
            width: u64::try_from(TEXELS).unwrap(),
            usage: READ,
            resource_options: Some(SHARED),
            ..TextureDesc::default()
        });
        let row = u64::try_from(bytes.len()).unwrap();
        let t = b.new_texture(OnFault::Die, &desc, 0, row).unwrap().unwrap();
        let kernel = match class {
            Class::Float => "rgbx_read_buffer_float",
            Class::Uint => "rgbx_read_buffer_uint",
            Class::Int => "rgbx_read_buffer_int",
        };
        self.output(kernel, &t, 16, &[], None);
    }

    /// A write of random values into a texture of `format` (`texel` bytes, `class`) an
    /// argument buffer holds.
    fn held_write(&self, format: u64, texel: usize, class: Class) {
        let t = self.texture(format, READ_WRITE, PRIVATE, &vec![0; TEXELS * texel]);
        let kernel = match class {
            Class::Float => "rgbx_write_held_float",
            Class::Uint => "rgbx_write_held_uint",
            Class::Int => "rgbx_write_held_int",
        };
        let bits = u32::try_from(texel * 2).unwrap();
        let v = self.buffer(&values(kernel, bits, format, true));
        let held = self.buffer(&t.gpu_resource_id().to_le_bytes());
        self.dispatch(kernel, None, &[&v, &held], None, Some((&t, 2)));
    }

    /// A Private texture of `format` once a colour fill of all of it wrote `colour` (red, green,
    /// blue, alpha) in the pixel format `format`.
    fn filled(&self, format: u64, [red, green, blue, alpha]: [f64; 4]) -> P::Texture {
        let t = self.texture_of(format, READ, PRIVATE, 1);
        self.blit(vec![BlitCmd::FillTextureColor {
            texture: &t,
            level: 0,
            slice: 0,
            region: MtlRegion::new(MtlOrigin::new(0, 0, 0), MtlSize::new(SIDE, SIDE, 1)),
            color: ClearColor {
                red,
                green,
                blue,
                alpha,
            },
            pixel_format: format,
        }]);
        t
    }
}

/// The Metal platform and its device, where the host has one.
fn metal() -> Option<(MetalPlatform, <MetalPlatform as Platform>::Device)> {
    MetalPlatform::system_default_device().map(|d| (MetalPlatform::new(), d))
}

/// What `f` answers, or where it died and the line its death named.
fn survived<R>(f: impl FnOnce() -> R) -> Result<R, (DeathSite, String)> {
    *crate::answer::LAST_LINE.lock() = None;
    let died = catch_death(f);
    let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
    died.map_err(|site| (site, line))
}

/// Whether the bytes `metal` and `ours` of one thread's halves differ only where both are NaNs:
/// a half read's or sample's NaN, whose bits follow the device (`device-nan-encoding`, which the
/// runtime announces at a read, sample, gather or fetch of a half texel, and which dies). Each
/// test asserts that no such thread is left where nothing died.
fn half_nans_only(metal: &[u8], ours: &[u8]) -> bool {
    let halves = |b: &[u8]| {
        b.as_chunks::<2>()
            .0
            .iter()
            .map(|&h| u16::from_le_bytes(h))
            .collect::<Vec<_>>()
    };
    halves(metal)
        .into_iter()
        .zip(halves(ours))
        .all(|(m, o)| m == o || (half_nan(m) && half_nan(o)))
}

/// Adds to `differ` each of `ours`' cases whose bytes are not `theirs`', named after `of`: how
/// many of its 256 threads' (or texels') bytes differ, and the first such, both sides'. A half
/// read's or sample's thread whose halves differ only in NaNs' bits ([`half_nans_only`]) is
/// counted in `nans` instead, for the test to check the note was relayed.
fn compare(
    differ: &mut Vec<String>,
    nans: &mut usize,
    of: &str,
    theirs: &[(String, Vec<u8>)],
    ours: &[(String, Vec<u8>)],
) {
    assert_eq!(theirs.len(), ours.len(), "{of}");
    for ((case, metal), (_, mine)) in theirs.iter().zip(ours) {
        assert_eq!(metal.len(), mine.len(), "{of} {case}");
        let element = (metal.len() / TEXELS).max(1);
        let halves = ["rgbx_read_half", "rgbx_sample_half", "rgbx_read_held_half"]
            .iter()
            .any(|half| case.starts_with(half));
        let differs = |(m, o): &(&[u8], &[u8])| m != o && !(halves && half_nans_only(m, o));
        *nans += (metal.chunks(element).zip(mine.chunks(element)))
            .filter(|(m, o)| m != o && halves && half_nans_only(m, o))
            .count();
        let mut chunks = metal.chunks(element).zip(mine.chunks(element)).enumerate();
        let wrong = chunks.clone().filter(|(_, pair)| differs(pair)).count();
        if let Some((i, (m, o))) = chunks.find(|(_, pair)| differs(pair)) {
            differ.push(format!(
                "{of} {case}: {wrong} of {} differ, the first at {i}: Metal {m:02x?}, ours \
                 {o:02x?}",
                metal.len() / element
            ));
        }
    }
}

/// Reads of random bytes of a texture of each RGBX format, Shared and Private, by `read()` in
/// each type the format reads as, in bounds and out, and by `read()` in bounds of the texture an
/// argument buffer holds, as floats or as unsigned or signed integers by the format's type; and
/// for the formats read as floats, `sample()` as floats and halves at random coordinates past the
/// edges through each filter, each address mode and each border colour, and `gather()` of each
/// component: each reads as Metal's does, bit for bit, but where the read's kernel binds a
/// texture of a float format, whose half reads, samples and gathers the runtime announces
/// `device-nan-encoding` for: Metal hands back a 16-bit texel's own NaN and the canonical one for
/// a 32-bit texel's, the runtime the canonical one for a 16-bit texel's and a 32-bit texel's own.
/// That dispatch dies, naming the kind ([`crate::answer::DIES`]), for the float formats alone
#[test]
fn reads_of_rgbx_textures_are_metals() {
    let Some((mp, md)) = metal() else {
        return;
    };
    let (vp, vd) = (platform(), dev());
    let (theirs, ours) = (Gpu::new(&mp, &md), Gpu::new(&vp, &vd));
    let (mut differ, mut nans, mut died) = (Vec::new(), 0, Vec::new());
    let sel = "dispatchThreads:threadsPerThreadgroup:";
    let named = "(metal2vulkan-degrade-device-nan-encoding)";
    for (seed, (format, name, texel, class, _)) in (1u64..).zip(FORMATS) {
        for options in [SHARED, PRIVATE] {
            let bytes = Rng(0x9e37_79b9_7f4a_7c15 ^ seed).bytes(TEXELS * texel);
            let metals = theirs.reads(class, &theirs.texture(format, READ, options, &bytes));
            let of = format!("{name} options {options:#x}");
            let mine = survived(|| ours.reads(class, &ours.texture(format, READ, options, &bytes)));
            match mine {
                Ok(mine) => compare(&mut differ, &mut nans, &of, &metals, &mine),
                Err((site, line))
                    if (site.kind, site.file) == (DeathKind::Abort, sel)
                        && line.contains(named)
                        && matches!(class, Class::Float) =>
                {
                    died.push(of);
                }
                Err((site, line)) => differ.push(format!("{of}: died at {site:?}: {line}")),
            }
        }
    }
    assert!(differ.is_empty(), "{} differ: {differ:#?}", differ.len());
    assert_eq!(nans, 0, "half NaNs differ where nothing died");
    let floats = ["RGBX16Float", "RGBX32Float"]
        .map(|f| [SHARED, PRIVATE].map(|o| format!("{f} options {o:#x}")));
    assert_eq!(
        died,
        floats.concat(),
        "the reads that die of device-nan-encoding"
    );
}

/// Kernel writes into two of every three texels of a texture of each RGBX format holding random
/// bytes, Shared and Private, in each type the format reads as (floats with NaNs, infinities and
/// values past each format's range, halves of every kind but NaNs, which
/// [`half_nan_writes_are_metals`] holds, integers past each channel's range), and a kernel's
/// `read()` and `write()` of each texel's components reversed: the texture's bytes and what the
/// kernel read are Metal's, bit for bit, and nothing dies: `RGBX16Float`'s halves are written
/// through the unsigned integer view of its RGBA twin, keeping their bits
#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "MoltenVK: the runtime records device-nan-encoding, a death in strict mode; Mac-only test, off everywhere; ledger: MoltenVK strict deaths"
)]
fn kernel_writes_of_rgbx_textures_are_metals() {
    let Some((mp, md)) = metal() else {
        return;
    };
    let (vp, vd) = (platform(), dev());
    let (theirs, ours) = (Gpu::new(&mp, &md), Gpu::new(&vp, &vd));
    let (mut differ, mut nans) = (Vec::new(), 0);
    for (seed, (format, name, texel, class, _)) in (1u64..).zip(FORMATS) {
        for options in [SHARED, PRIVATE] {
            let bytes = Rng(0xbb67_ae85_84ca_a73b ^ seed).bytes(TEXELS * texel);
            let shape = (format, texel, class);
            let metals = theirs.writes(shape, options, &bytes, seed);
            let of = format!("{name} options {options:#x}");
            match survived(|| ours.writes(shape, options, &bytes, seed)) {
                Ok(mine) => compare(&mut differ, &mut nans, &of, &metals, &mine),
                Err((site, line)) => differ.push(format!("{of}: died at {site:?}: {line}")),
            }
        }
    }
    assert!(differ.is_empty(), "{} differ: {differ:#?}", differ.len());
    assert_eq!(nans, 0);
}

/// A texture read in a type other than its format's, which Metal's shading language defines no
/// answer for, and which a guest's shader makes only where it mistypes a texture (no boot or
/// suite case is known to make one). A texture of `RGBA8Unorm`, `RGBA8Uint`, `RGBA32Sint` and
/// `RGBX8Uint`, Shared and Private, of random bytes, read by `read()` in each other type of
/// `float`, `uint` and `int`, bound to the kernel's slot and held by an argument buffer
/// ([`Gpu::held_read`]), in bounds: measured on this host, Metal hands back the 32-bit lanes the
/// read in the format's own type does, bit for bit. The runtime reads them so at the slot and
/// through the argument buffer: each is Metal's, bit for bit
#[test]
fn mistyped_reads_are_metals() {
    let Some((mp, md)) = metal() else {
        return;
    };
    let (vp, vd) = (platform(), dev());
    let (theirs, ours) = (Gpu::new(&mp, &md), Gpu::new(&vp, &vd));
    let reader = |class| match class {
        Class::Float => "rgbx_read_float",
        Class::Uint => "rgbx_read_uint",
        Class::Int => "rgbx_read_int",
    };
    let formats = [
        (70, "RGBA8Unorm", 4, Class::Float),
        (73, "RGBA8Uint", 4, Class::Uint),
        (124, "RGBA32Sint", 16, Class::Int),
        (303, "RGBX8Uint", 4, Class::Uint),
    ];
    let mut differ = Vec::new();
    for (seed, (format, name, texel, class)) in (1u64..).zip(formats) {
        for options in [SHARED, PRIVATE] {
            let bytes = Rng(0x2545_f491_4f6c_dd1d ^ seed).bytes(TEXELS * texel);
            let metals = theirs.texture(format, READ, options, &bytes);
            let mine = ours.texture(format, READ, options, &bytes);
            let at = (theirs.buffer(&le([0, 0])), ours.buffer(&le([0, 0])));
            let lanes = theirs.output(reader(class), &metals, 16, &[&at.0], None);
            let others = [Class::Float, Class::Uint, Class::Int].into_iter();
            for read in others.filter(|&read| read != class) {
                for held in [false, true] {
                    let of = format!("{name} options {options:#x} read as {read:?} held {held}");
                    let metal = if held {
                        theirs.held_read(read, &metals).1
                    } else {
                        theirs.output(reader(read), &metals, 16, &[&at.0], None)
                    };
                    if metal != lanes {
                        differ.push(format!("{of}: Metal's are not the texels' own lanes"));
                    }
                    let read = survived(|| {
                        if held {
                            ours.held_read(read, &mine).1
                        } else {
                            ours.output(reader(read), &mine, 16, &[&at.1], None)
                        }
                    });
                    match read {
                        Ok(read) => {
                            let pairs = read.chunks(16).zip(metal.chunks(16));
                            let off = pairs.filter(|(o, m)| o != m).count();
                            if off != 0 {
                                differ.push(format!("{of}: {off} of {TEXELS} not Metal's"));
                            }
                        }
                        Err((site, line)) => differ.push(format!("{of}: died at {site:?}: {line}")),
                    }
                }
            }
        }
    }
    assert!(differ.is_empty(), "{} differ: {differ:#?}", differ.len());
}

/// A texture sampled at a kernel's slot in a numeric class other than its format's: an `R32Uint`
/// texture of random words sampled through a `texture2d` of `float`, `half` and `int`, and an
/// `RGBA8Unorm` one of random bytes through one of `uint` and `int`, Shared, at [`coordinates`]
/// through a nearest and a linear sampler that clamp to the edge ([`Gpu::sampled`]). Metal
/// samples the texture in its format's own class and hands back the words as the declared type:
/// a `half` sample of an integer format each word's low 16 bits, and an integer format's linear
/// sample its nearest one. The runtime samples it so, through the slot's `cross_class` variant,
/// announcing nothing: each sample is Metal's, bit for bit
#[test]
fn cross_class_samples_are_metals() {
    let Some((mp, md)) = metal() else {
        return;
    };
    let (vp, vd) = (platform(), dev());
    let (theirs, ours) = (Gpu::new(&mp, &md), Gpu::new(&vp, &vd));
    // The kernels that sample each format in the other classes, with their bytes a thread.
    let r32 = [
        ("rgbx_sample_float", 16),
        ("rgbx_sample_half", 8),
        ("rgbx_sample_int", 16),
    ];
    let rgba8 = [("rgbx_sample_uint", 16), ("rgbx_sample_int", 16)];
    // Each format: its raw value, its name, the kernel that samples it in its own class, whether
    // that class is an integer one, and the kernels of the other classes.
    let formats = [
        (53, "R32Uint", "rgbx_sample_uint", true, &r32[..]),
        (70, "RGBA8Unorm", "rgbx_sample_float", false, &rgba8[..]),
    ];
    let mut differ = Vec::new();
    for (seed, (format, name, own, integer, others)) in (1u64..).zip(formats) {
        let bytes = Rng(0x3c6e_f372_fe94_f82b ^ seed).bytes(TEXELS * 4);
        let metals = theirs.texture(format, READ, SHARED, &bytes);
        let mine = ours.texture(format, READ, SHARED, &bytes);
        for filter in [0, 1] {
            // Metal samples an integer format through a linear filter as through a nearest one.
            let words = theirs.sampled(own, &metals, 16, if integer { 0 } else { filter });
            for &(kernel, bytes) in others {
                let of = format!("{name} by {kernel} filter {filter}");
                let handed: Vec<u8> = if bytes == 16 {
                    words.clone()
                } else {
                    words.chunks(4).flat_map(|w| [w[0], w[1]]).collect()
                };
                if theirs.sampled(kernel, &metals, bytes, filter) != handed {
                    differ.push(format!("{of}: Metal's are not the own class's words"));
                }
                match survived(|| ours.sampled(kernel, &mine, bytes, filter)) {
                    Ok(read) => {
                        let pairs = read.chunks(bytes).zip(handed.chunks(bytes));
                        let off = pairs.filter(|(o, m)| o != m).count();
                        if off != 0 {
                            differ.push(format!("{of}: {off} of {TEXELS} not Metal's"));
                        }
                    }
                    Err((site, line)) => differ.push(format!("{of}: died at {site:?}: {line}")),
                }
            }
        }
    }
    assert!(differ.is_empty(), "{} differ: {differ:#?}", differ.len());
}

/// Reads of random bytes of a texture of each bfloat format (`R16Bfloat`, `RG16Bfloat`,
/// `RGBA16Bfloat`), Shared and Private, bound to the kernel's texture slot
/// ([`Gpu::direct_reads`]), then through an argument buffer that holds the texture's ID
/// ([`Gpu::held_float_reads`], [`Gpu::held_samples`]), and kernel writes into one
/// ([`Gpu::writes`]), as [`reads_of_rgbx_textures_are_metals`] and
/// [`kernel_writes_of_rgbx_textures_are_metals`] make them. The writes are Metal's, bit for bit.
/// The reads die at their first dispatch, for every format and option, naming
/// `half-texel-conversion` ([`crate::answer::DIES`]), which the runtime announces for their
/// kernel (a bfloat texel converted to `half` by a rounding Metal's may not be), before
/// `device-nan-encoding`, which it announces there too. The runtime holds such a texture's
/// codes and decodes and encodes them in the function (measured on an Apple M4 under macOS
/// 27.0.1). A sample or a gather through an argument
/// buffer and a sampler of a mip filter, which Metal makes, dies at the dispatch naming the
/// runtime's refusal, `ArgumentBufferTextures` ([`held_reads_die`]).
#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "MoltenVK: the runtime records device-pack-conversion, a death in strict mode; Mac-only test, off everywhere; ledger: MoltenVK strict deaths"
)]
fn bfloat_textures_read_and_write_as_metals() {
    let Some((mp, md)) = metal() else {
        return;
    };
    let (vp, vd) = (platform(), dev());
    let (theirs, ours) = (Gpu::new(&mp, &md), Gpu::new(&vp, &vd));
    let (mut differ, mut nans, mut died) = (Vec::new(), 0, Vec::new());
    let sel = "dispatchThreads:threadsPerThreadgroup:";
    let named = "(metal2vulkan-degrade-half-texel-conversion)";
    let formats = [
        (26, "R16Bfloat", 2),
        (66, "RG16Bfloat", 4),
        (116, "RGBA16Bfloat", 8),
    ];
    for (seed, (format, name, texel)) in (1u64..).zip(formats) {
        for options in [SHARED, PRIVATE] {
            let of = format!("{name} options {options:#x}");
            let bytes = Rng(0x3c6e_f372_fe94_f82b ^ seed).bytes(TEXELS * texel);
            let t = theirs.texture(format, READ, options, &bytes);
            let mut metals = theirs.direct_reads(Class::Float, &t);
            metals.extend(theirs.held_float_reads(&t));
            metals.extend(theirs.held_samples(&t, false));
            theirs.held_samples(&t, true);
            let t = ours.texture(format, READ, options, &bytes);
            let read = || {
                let mut mine = ours.direct_reads(Class::Float, &t);
                mine.extend(ours.held_float_reads(&t));
                mine.extend(ours.held_samples(&t, false));
                mine
            };
            match survived(read) {
                Ok(mine) => compare(&mut differ, &mut nans, &of, &metals, &mine),
                Err((site, line))
                    if (site.kind, site.file) == (DeathKind::Abort, sel)
                        && line.contains(named) =>
                {
                    died.push(of.clone());
                }
                Err((site, line)) => differ.push(format!("{of} reads: died at {site:?}: {line}")),
            }
            held_reads_die(
                &mut differ,
                &format!("{of} samples through a mip filter"),
                "ArgumentBufferTextures",
                survived(|| ours.held_samples(&t, true)),
            );
            let shape = (format, texel, Class::Float);
            let metals = theirs.writes(shape, options, &bytes, seed);
            match survived(|| ours.writes(shape, options, &bytes, seed)) {
                Ok(mine) => compare(&mut differ, &mut nans, &of, &metals, &mine),
                Err((site, line)) => differ.push(format!("{of} writes: died at {site:?}: {line}")),
            }
        }
    }
    assert!(differ.is_empty(), "{} differ: {differ:#?}", differ.len());
    assert_eq!(nans, 0, "half NaNs differ where nothing died");
    let all: Vec<String> = formats
        .iter()
        .flat_map(|(_, name, _)| [SHARED, PRIVATE].map(|o| format!("{name} options {o:#x}")))
        .collect();
    assert_eq!(died, all, "the reads that die of half-texel-conversion");
}

/// The new Private textures [`new_textures_read_as_metals`] reads level by level
/// ([`fresh_levels`]), most of which Metal keeps losslessly compressed: each case, its
/// descriptor, its texel's bytes and its slices. A 16 by 16 texture of one level, of
/// `ShaderRead` usage, of each of 20 formats: `R8Unorm`, `RG8Unorm`, `R16Float`, `R32Float`,
/// `RG16Float`, `RGBA8Unorm` and its sRGB, snorm and uint forms, `BGRA8Unorm` and its sRGB form,
/// `RGB10A2Unorm`, `BGR10A2Unorm`, `RG11B10Float`, `RGB9E5Float`, `RGBA16Unorm`, `RGBA16Float`,
/// `RGBA32Float`, `BGRA10_XR` and `BGR10_XR`; one of `RenderTarget` usage too; chains of
/// `RGBA8Unorm` (64 by 64, 100 by 40 and 17 by 17, to 1 by 1) and of `RGBA16Float` (64 by 64);
/// a cube and a 2D array of two slices, 16 by 16; and a 1D texture 64 wide, which Metal keeps
/// uncompressed.
fn compressed_shapes() -> Vec<(String, TextureDesc, usize, u64)> {
    let shape = |format, (width, height), levels| TextureDesc {
        pixel_format: format,
        width,
        height,
        mipmap_level_count: levels,
        usage: READ,
        resource_options: Some(PRIVATE),
        ..TextureDesc::default()
    };
    let formats = [
        (10, "R8Unorm", 1),
        (30, "RG8Unorm", 2),
        (25, "R16Float", 2),
        (55, "R32Float", 4),
        (65, "RG16Float", 4),
        (70, "RGBA8Unorm", 4),
        (71, "RGBA8Unorm_sRGB", 4),
        (72, "RGBA8Snorm", 4),
        (73, "RGBA8Uint", 4),
        (80, "BGRA8Unorm", 4),
        (81, "BGRA8Unorm_sRGB", 4),
        (90, "RGB10A2Unorm", 4),
        (94, "BGR10A2Unorm", 4),
        (92, "RG11B10Float", 4),
        (93, "RGB9E5Float", 4),
        (110, "RGBA16Unorm", 8),
        (115, "RGBA16Float", 8),
        (125, "RGBA32Float", 16),
        (552, "BGRA10_XR", 8),
        (554, "BGR10_XR", 4),
    ];
    let mut out: Vec<_> = (formats.into_iter())
        .map(|(format, name, texel)| (name.to_owned(), shape(format, (SIDE, SIDE), 1), texel, 1))
        .collect();
    let rendered = TextureDesc {
        usage: RENDER,
        ..shape(70, (SIDE, SIDE), 1)
    };
    out.push(("RGBA8Unorm RenderTarget".to_owned(), rendered, 4, 1));
    for (name, format, size, levels, texel) in [
        ("RGBA8Unorm 64 by 64 chain", 70, (64, 64), 7, 4),
        ("RGBA8Unorm 100 by 40 chain", 70, (100, 40), 7, 4),
        ("RGBA8Unorm 17 by 17 chain", 70, (17, 17), 5, 4),
        ("RGBA16Float 64 by 64 chain", 115, (64, 64), 7, 8),
    ] {
        out.push((name.to_owned(), shape(format, size, levels), texel, 1));
    }
    let cube = TextureDesc {
        texture_type: 5,
        ..shape(70, (SIDE, SIDE), 1)
    };
    let array = TextureDesc {
        texture_type: 3,
        array_length: 2,
        ..shape(70, (SIDE, SIDE), 1)
    };
    let line = TextureDesc {
        texture_type: 0,
        ..shape(70, (64, 1), 1)
    };
    out.push(("RGBA8Unorm cube".to_owned(), cube, 4, 6));
    out.push(("RGBA8Unorm 2D array".to_owned(), array, 4, 2));
    out.push(("RGBA8Unorm 1D".to_owned(), line, 4, 1));
    out
}

/// What `g` reads of each level of each of the `slices` slices of a new texture of `desc`, of
/// texels of `texel` bytes, each by a blit into a buffer of its own command buffer: the case,
/// and the bytes.
fn fresh_levels<P: Platform>(
    g: &Gpu<'_, P>,
    desc: &TextureDesc,
    texel: usize,
    slices: u64,
) -> Vec<(String, Vec<u8>)> {
    let t = g.d.new_texture(OnFault::Die, &g.p.texture_descriptor(desc));
    let t = t.unwrap().unwrap();
    let mut out = Vec::new();
    for slice in 0..slices {
        for level in 0..desc.mipmap_level_count.max(1) {
            let (w, h) = ((desc.width >> level).max(1), (desc.height >> level).max(1));
            let row = w * u64::try_from(texel).unwrap();
            let n = usize::try_from(row * h).unwrap();
            let b = g.buffer(&vec![0; n]);
            g.blit(vec![BlitCmd::CopyTextureToBuffer {
                source_texture: &t,
                source_slice: slice,
                source_level: level,
                source_origin: MtlOrigin::new(0, 0, 0),
                source_size: MtlSize::new(w, h, 1),
                destination_buffer: &b,
                destination_offset: 0,
                destination_bytes_per_row: row,
                destination_bytes_per_image: row * h,
            }]);
            let case = format!("slice {slice} level {level} ({w} by {h})");
            out.push((case, Gpu::<P>::bytes_of(&b, n)));
        }
    }
    out
}

/// What a new texture of its own memory reads, made right after one of its shape that held
/// random bytes was let go, on this platform and Metal's, Private textures most of all: a 16 by
/// 16 `RGBA8Unorm` texture,
/// Shared, and Private of `ShaderWrite` usage, which Metal keeps uncompressed, and a Private
/// `RGBA16Float` one of that usage, read by a blit; a Private 4-sample `RGBA8Unorm` render target
/// without `allowGPUOptimizedContents`, each sample read by a kernel; and a Private
/// `RGB10A8_2P_XR10` texture, both planes, read by a kernel. Each reads zero codes, as Metal's,
/// bit for bit. So does a memoryless `RGBA8Unorm` colour attachment loaded and stored `DontCare`,
/// which a fragment reads by framebuffer fetch (`encode_render`'s `memoryless_fetched`), alone in
/// its command buffer and after a pass that wrote other bytes through the tiles. A new Private
/// texture Metal keeps losslessly compressed reads a fixed texel of its format in each level whose
/// shorter side is 16 or more, and zero below, every slice and sample, which the runtime writes as
/// Metal's: a Private 16 by 16 `RGBA8Unorm` texture of `ShaderRead` usage alone, made right after
/// one of its shape was let go, `ff 00 ff ff` each on this host's M3 Max, bit for bit; each of
/// [`compressed_shapes`], level by level and slice by slice ([`fresh_levels`]), bit for bit; and a
/// 4-sample `RGBA8Unorm` render target, each sample read by a kernel. So does a 4-sample one whose
/// texel is a NaN, of `R16Float`, `RG16Float`, `RGBA16Float`, `R32Float` or `RG11B10Float`, which
/// the runtime copies from an integer image through a view of the image's format, which MoltenVK's
/// copy writes, where a copy between formats leaves the destination zero (the runtime's
/// `fresh-pattern-zeroed`, which dies here)
#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "MoltenVK: the runtime records device-pack-conversion, a death in strict mode; Mac-only test, off everywhere; ledger: MoltenVK strict deaths"
)]
fn new_textures_read_as_metals() {
    /// A new texture of `desc`'s, the shape of `desc` but `options` and `usage`.
    fn made<P: Platform>(g: &Gpu<'_, P>, desc: &TextureDesc) -> P::Texture {
        let desc = g.p.texture_descriptor(desc);
        g.d.new_texture(OnFault::Die, &desc).unwrap().unwrap()
    }
    /// What `g` reads of a new texture of `desc` (of `texel` bytes, its level read by a blit)
    /// made right after one of its shape held `bytes` and was let go.
    fn fresh<P: Platform>(
        g: &Gpu<'_, P>,
        desc: &TextureDesc,
        texel: usize,
        bytes: &[u8],
    ) -> Vec<u8> {
        drop(g.texture(
            desc.pixel_format,
            desc.usage,
            desc.resource_options.unwrap_or(0),
            bytes,
        ));
        let t = made(g, desc);
        g.read(&t, texel)
    }
    let Some((mp, md)) = metal() else {
        return;
    };
    let (vp, vd) = (platform(), dev());
    let (theirs, ours) = (Gpu::new(&mp, &md), Gpu::new(&vp, &vd));
    let shape = |format, usage, options| TextureDesc {
        pixel_format: format,
        width: SIDE,
        height: SIDE,
        usage,
        resource_options: Some(options),
        ..TextureDesc::default()
    };
    let blitted = [
        ("RGBA8Unorm Shared", shape(70, READ, SHARED), 4),
        (
            "RGBA8Unorm Private ShaderWrite",
            shape(70, READ_WRITE, PRIVATE),
            4,
        ),
        (
            "RGBA16Float Private ShaderWrite",
            shape(115, READ_WRITE, PRIVATE),
            8,
        ),
    ];
    let mut differ = Vec::new();
    let zero = |b: &[u8]| b.iter().all(|&x| x == 0);
    for (seed, (name, desc, texel)) in (1u64..).zip(blitted) {
        let bytes = Rng(0x510e_527f_ade6_82d1 ^ seed).bytes(TEXELS * texel);
        let metals = fresh(&theirs, &desc, texel, &bytes);
        let mine = survived(|| fresh(&ours, &desc, texel, &bytes));
        match mine {
            Ok(mine) if mine == metals && zero(&mine) => {}
            Ok(mine) => differ.push(format!(
                "{name}: Metal {:02x?}, ours {:02x?}",
                &metals[..16],
                &mine[..16]
            )),
            Err((site, line)) => differ.push(format!("{name}: died at {site:?}: {line}")),
        }
    }
    let multisample = TextureDesc {
        texture_type: 4,
        sample_count: 4,
        allow_gpu_optimized_contents: Some(false),
        ..shape(70, RENDER, PRIVATE)
    };
    let two_plane = shape(550, READ, PRIVATE);
    let at = (theirs.buffer(&le([0, 0])), ours.buffer(&le([0, 0])));
    let kernels = [
        (
            "RGBA8Unorm 4 samples",
            &multisample,
            "rgbx_read_samples",
            64,
        ),
        ("RGB10A8_2P_XR10 Private", &two_plane, "rgbx_read_float", 16),
    ];
    for (name, desc, kernel, bytes) in kernels {
        let metals = theirs.output(kernel, &made(&theirs, desc), bytes, &[&at.0], None);
        let mine = survived(|| ours.output(kernel, &made(&ours, desc), bytes, &[&at.1], None));
        match mine {
            Ok(mine) if mine == metals => {}
            Ok(mine) => differ.push(format!(
                "{name}: Metal {:02x?}, ours {:02x?}",
                &metals[..16],
                &mine[..16]
            )),
            Err((site, line)) => differ.push(format!("{name}: died at {site:?}: {line}")),
        }
    }
    let compressed = shape(70, READ, PRIVATE);
    let bytes = Rng(0x510e_527f_ade6_82d1).bytes(TEXELS * 4);
    let metals = fresh(&theirs, &compressed, 4, &bytes);
    let pattern = [0xff, 0x00, 0xff, 0xff];
    if metals != pattern.repeat(TEXELS) {
        let off = metals.chunks(4).filter(|t| *t != pattern).count();
        let first = &metals[..16];
        differ.push(format!(
            "compressed: Metal's {off} of {TEXELS} texels are not {pattern:02x?}: {first:02x?}"
        ));
    }
    match survived(|| fresh(&ours, &compressed, 4, &bytes)) {
        Ok(mine) if mine == metals => {}
        Ok(mine) => differ.push(format!("compressed: ours {:02x?}", &mine[..16])),
        Err((site, line)) => differ.push(format!("compressed: died at {site:?}: {line}")),
    }
    for (name, desc, texel, slices) in compressed_shapes() {
        let metals = fresh_levels(&theirs, &desc, texel, slices);
        match survived(|| fresh_levels(&ours, &desc, texel, slices)) {
            Ok(mine) => {
                for ((level, m), (_, o)) in metals.iter().zip(&mine) {
                    if m != o {
                        differ.push(format!(
                            "{name} {level}: Metal {:02x?}, ours {:02x?}",
                            &m[..m.len().min(16)],
                            &o[..o.len().min(16)]
                        ));
                    }
                }
            }
            Err((site, line)) => differ.push(format!("{name}: died at {site:?}: {line}")),
        }
    }
    let samples = |format| TextureDesc {
        texture_type: 4,
        sample_count: 4,
        ..shape(format, RENDER, PRIVATE)
    };
    let at = (theirs.buffer(&le([0, 0])), ours.buffer(&le([0, 0])));
    let read = "rgbx_read_samples";
    let name = "RGBA8Unorm 4 samples, compressed";
    let metals = theirs.output(read, &made(&theirs, &samples(70)), 64, &[&at.0], None);
    match survived(|| ours.output(read, &made(&ours, &samples(70)), 64, &[&at.1], None)) {
        Ok(mine) if mine == metals => {}
        Ok(mine) => differ.push(format!(
            "{name}: Metal {:02x?}, ours {:02x?}",
            &metals[..16],
            &mine[..16]
        )),
        Err((site, line)) => differ.push(format!("{name}: died at {site:?}: {line}")),
    }
    for (format, name) in [
        (25, "R16Float"),
        (65, "RG16Float"),
        (115, "RGBA16Float"),
        (55, "R32Float"),
        (92, "RG11B10Float"),
    ] {
        let name = format!("{name} 4 samples, compressed");
        let metals = theirs.output(read, &made(&theirs, &samples(format)), 64, &[&at.0], None);
        match survived(|| ours.output(read, &made(&ours, &samples(format)), 64, &[&at.1], None)) {
            Ok(mine) if mine == metals => {}
            Ok(mine) => differ.push(format!(
                "{name}: Metal {:02x?}, ours {:02x?}",
                &metals[..16],
                &mine[..16]
            )),
            Err((site, line)) => differ.push(format!("{name}: died at {site:?}: {line}")),
        }
    }
    for tiles_written in [false, true] {
        let name = format!("memoryless DontCare fetched, tiles written first {tiles_written}");
        let metals = memoryless_fetched(&mp, &md, tiles_written);
        match survived(|| memoryless_fetched(&vp, &vd, tiles_written)) {
            Ok(mine) if mine == metals && zero(&mine) => {}
            Ok(mine) => differ.push(format!(
                "{name}: Metal {:02x?}, ours {:02x?}",
                &metals[..16],
                &mine[..16]
            )),
            Err((site, line)) => differ.push(format!("{name}: died at {site:?}: {line}")),
        }
    }
    assert!(differ.is_empty(), "{} differ: {differ:#?}", differ.len());
}

/// The colours [`clears_and_bfloat_fills_are_metals`] fills with, red, green, blue and alpha:
/// plain values, values halfway between two bfloats and a hair off halfway, past a half's range,
/// a denormal, negative zero, infinities and NaN.
const BFLOAT_COLOURS: [[f64; 4]; 4] = [
    [0.25, 0.5, 0.75, 1.0],
    [1.003_906_25, 1.011_718_75, 1.003_906_369_209_289_6, -0.0],
    [-3.0e38, 1.0e-40, 70_000.0, -2.5],
    [f64::INFINITY, f64::NEG_INFINITY, f64::NAN, 1.0e-3],
];

/// The colours [`clears_and_bfloat_fills_are_metals`] clears to, red, green, blue and alpha:
/// colours of values `half` and bfloat both hold exactly (zeros, ±1, 0.5, -2.5, the infinities,
/// half's smallest step and smallest normal, 32768, 61440), which either of Metal's routes
/// stores as its own code; and colours of one value each that its routes store apart, straight
/// to bfloat and through `half` (on the M3 Max): -3e38 (`ff62`, `ff80`), 1e-40 (`0001`, `0000`),
/// 70000 (`4789`, `7f80`) and 1.0039063692092896 (`3f81`, `3f80`, rounded twice), in red; 70000
/// in alpha; and NaNs in green, quiet, negative (`ffc0` straight), of a payload
/// (`0x7ff8_4000_0000_0000`, `7fc2` straight) and all ones (`7fff` straight), each `7fc0`
/// through `half`.
const BFLOAT_CLEARS: [[f64; 4]; 13] = [
    [0.25, 0.5, 0.75, 1.0],
    [0.0, -0.0, -1.0, -2.5],
    [
        f64::INFINITY,
        f64::NEG_INFINITY,
        5.960_464_477_539_063e-8,
        61_440.0,
    ],
    [3.0, -0.375, 32_768.0, 6.103_515_625e-5],
    [-3.0e38, 0.5, 0.5, 1.0],
    [1.0e-40, 0.5, 0.5, 1.0],
    [70_000.0, 0.5, 0.5, 1.0],
    [1.003_906_369_209_289_6, 0.5, 0.5, 1.0],
    [0.5, 0.25, 0.75, 70_000.0],
    [0.5, f64::NAN, 0.75, 1.0],
    [0.5, -f64::NAN, 0.75, 1.0],
    [0.5, f64::from_bits(0x7ff8_4000_0000_0000), 0.75, 1.0],
    [0.5, f64::from_bits(0x7fff_ffff_ffff_ffff), 0.75, 1.0],
];

/// Of a Private 16 by 16 texture of each bfloat format (`R16Bfloat`, `RG16Bfloat`,
/// `RGBA16Bfloat`): a pass's clear and store of it to each of [`BFLOAT_CLEARS`], by both of
/// Metal's routes, which the runtime takes: of `RENDER` usage, which
/// Metal keeps losslessly compressed, in a pass without a draw call, straight to bfloat; and of
/// `ShaderWrite` usage too, which it keeps uncompressed, through `half`; a pass that loads it,
/// holding random codes, draws nothing and stores it, which keeps every code; and a colour fill
/// in the format of each of [`BFLOAT_COLOURS`] (`fillTexture:level:slice:region:color:
/// pixelFormat:`), which the runtime's colour fill packs ([`crate::gpu::fill::OWN_FORMAT_PACKED`]).
/// Each stores Metal's bytes, bit for bit. Of a Private 16 by 16 `RGBA16Float` texture of 4
/// samples, a pass's clear to each of [`FLOAT_CLEARS`], and its store, stores Metal's bytes, bit
/// for bit, the resolve's and each sample's ([`Gpu::multisample_cleared`])
#[test]
fn clears_and_bfloat_fills_are_metals() {
    /// The bytes of a texture of `texel` bytes `g` reads after a pass that loads it, holding
    /// `bytes`, and stores it, drawing nothing.
    fn loaded<P: Platform>(g: &Gpu<'_, P>, format: u64, texel: usize, bytes: &[u8]) -> Vec<u8> {
        let t = g.texture(format, RENDER, PRIVATE, bytes);
        let pass = g.p.render_pass_descriptor();
        let attachment = ColorAttachmentSet {
            texture: Some(Some(&t)),
            load_action: Some(1),
            store_action: Some(1),
            ..ColorAttachmentSet::default()
        };
        pass.set_color_attachment(OnFault::Die, 0, &attachment)
            .unwrap();
        g.run(|cb| {
            let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
            e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
        });
        g.read(&t, texel)
    }
    let Some((mp, md)) = metal() else {
        return;
    };
    let (vp, vd) = (platform(), dev());
    let (theirs, ours) = (Gpu::new(&mp, &md), Gpu::new(&vp, &vd));
    let formats = [
        (26, "R16Bfloat", 2),
        (66, "RG16Bfloat", 4),
        (116, "RGBA16Bfloat", 8),
    ];
    let mut differ = Vec::new();
    let mut compare =
        |of: String, metals: Vec<u8>, mine: Result<Vec<u8>, (DeathSite, String)>| match mine {
            Ok(mine) if mine == metals => {}
            Ok(mine) => {
                let off = (metals.iter().zip(&mine)).position(|(m, o)| m != o);
                let at = off.unwrap_or(0) / 16 * 16;
                let end = metals.len().min(at + 16);
                differ.push(format!(
                    "{of}: from byte {at}, Metal {:02x?}, ours {:02x?}",
                    &metals[at..end],
                    &mine[at..end]
                ));
            }
            Err((site, line)) => differ.push(format!("{of}: died at {site:?}: {line}")),
        };
    for (seed, (format, name, texel)) in (1u64..).zip(formats) {
        for colour in BFLOAT_CLEARS {
            for (usage, route) in [(RENDER, "straight"), (RENDER | READ_WRITE, "through half")] {
                let of = format!("{name} clear to {colour:?} ({route})");
                let metals = theirs.read(&theirs.cleared(format, usage, colour), texel);
                let mine = survived(|| ours.read(&ours.cleared(format, usage, colour), texel));
                compare(of, metals, mine);
            }
        }
        for colour in BFLOAT_COLOURS {
            let metals = theirs.read(&theirs.filled(format, colour), texel);
            let mine = survived(|| ours.read(&ours.filled(format, colour), texel));
            compare(format!("{name} fill of {colour:?}"), metals, mine);
        }
        let bytes = Rng(0x1f83_d9ab_fb41_bd6b ^ seed).bytes(TEXELS * texel);
        let metals = loaded(&theirs, format, texel, &bytes);
        let mine = survived(|| loaded(&ours, format, texel, &bytes));
        compare(format!("{name} loaded, nothing drawn"), metals, mine);
    }
    for colour in FLOAT_CLEARS {
        let of = format!("RGBA16Float 4 samples clear to {colour:?}");
        let (stored, held) = theirs.multisample_cleared(115, 8, colour);
        match survived(|| ours.multisample_cleared(115, 8, colour)) {
            Ok((mine, _)) if mine != stored => compare(of, stored, Ok(mine)),
            Ok((_, mine)) => compare(format!("{of}, samples"), held, Ok(mine)),
            Err(death) => compare(of, stored, Err(death)),
        }
    }
    assert!(differ.is_empty(), "{} differ: {differ:#?}", differ.len());
}

/// The state of `rounding.metal`'s `sweep_vertex` and `sweep_fragment` into colour attachment 0
/// of `format` at `samples` samples, made from the whole render value on `d`.
fn sweep_state<P: Platform>(d: &P::Device, format: u64, samples: u64) -> P::RenderPipeline {
    use crate::gpu::device::tests::render_value;
    use crate::gpu::encode_render::tests::ROUNDING;

    let library = d.new_library(ROUNDING).unwrap().unwrap().unwrap();
    let function = |name| library.new_function(OnFault::Die, name).unwrap().unwrap();
    let functions = [function("sweep_vertex"), function("sweep_fragment")];
    let colour: &[(u8, u32)] = &[(0, 0), (1, u32::try_from(format).unwrap())];
    let desc = render_value::<P>((&functions[0], &functions[1]), samples, &[colour]);
    d.new_render_pipeline_state_full(&desc).unwrap().unwrap()
}

/// What a pass that clears a Private `side` by `side` texture of `format` (of `texel` bytes),
/// `usage` and `samples` samples to `colour` (red, green, blue, alpha) and stores it leaves on
/// `g`, drawing nothing, or, where `drawn` holds a vertex count, drawing that many vertices of a
/// triangle list ([`sweep_state`]), which make no triangle: at one sample, its bytes; at more,
/// those of a Private texture of the format and `RENDER` usage the pass resolves it into, by its
/// store action (`StoreAndMultisampleResolve`), and, at a side of 16, each sample as a `float4`
/// (`rgbx_read_samples`). Each texture's bytes are read by a blit into a buffer. `None` where the
/// device makes no such texture.
fn nan_cleared<P: Platform>(
    g: &Gpu<'_, P>,
    (format, texel): (u64, usize),
    (usage, side, samples): (u64, u64, u64),
    [red, green, blue, alpha]: [f64; 4],
    drawn: Option<u64>,
) -> Option<(Vec<u8>, Vec<u8>)> {
    let made = |samples, usage| {
        let desc = g.p.texture_descriptor(&TextureDesc {
            texture_type: if samples > 1 { 4 } else { 2 },
            sample_count: samples,
            pixel_format: format,
            width: side,
            height: side,
            usage,
            resource_options: Some(PRIVATE),
            ..TextureDesc::default()
        });
        g.d.new_texture(OnFault::Die, &desc).unwrap()
    };
    let cleared = made(samples, usage)?;
    let resolve = match samples > 1 {
        true => Some(made(1, RENDER)?),
        false => None,
    };
    let state = drawn.map(|_| sweep_state::<P>(g.d, format, samples));
    let pass = g.p.render_pass_descriptor();
    let attachment = ColorAttachmentSet {
        texture: Some(Some(&cleared)),
        resolve_texture: resolve.as_ref().map(Some),
        load_action: Some(2),
        store_action: Some(if resolve.is_some() { 3 } else { 1 }),
        clear_color: Some(ClearColor {
            red,
            green,
            blue,
            alpha,
        }),
        ..ColorAttachmentSet::default()
    };
    pass.set_color_attachment(OnFault::Die, 0, &attachment)
        .unwrap();
    let kind = g.buffer(&[0; 4]);
    g.run(|cb| {
        let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
        if let (Some(state), Some(vertex_count)) = (&state, drawn) {
            for cmd in [
                RenderCmd::SetRenderPipelineState {
                    pipeline_state: Some(state),
                },
                RenderCmd::SetFragmentBuffer {
                    buffer: Some(&kind),
                    offset: 0,
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
        e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
    });
    let stored = resolve.as_ref().unwrap_or(&cleared);
    let n = usize::try_from(side * side).unwrap() * texel;
    let b = g.buffer(&vec![0; n]);
    let row = side * u64::try_from(texel).unwrap();
    g.blit(vec![BlitCmd::CopyTextureToBuffer {
        source_texture: stored,
        source_slice: 0,
        source_level: 0,
        source_origin: MtlOrigin::new(0, 0, 0),
        source_size: MtlSize::new(side, side, 1),
        destination_buffer: &b,
        destination_offset: 0,
        destination_bytes_per_row: row,
        destination_bytes_per_image: row * side,
    }]);
    // A read of a NaN's samples takes the device's NaN encoding, which the runtime announces
    // (`device-nan-encoding`), here this host's own: their bytes are compared.
    let held = match samples > 1 && side == SIDE {
        true => crate::answer::relaxed(|| g.output("rgbx_read_samples", &cleared, 64, &[], None)),
        false => Vec::new(),
    };
    Some((Gpu::<P>::bytes_of(&b, n), held))
}

/// The NaNs [`nan_clears_are_metals`] clears to, as doubles: the plain quiet NaN, a negative
/// one, one of a payload, all ones, and a signalling one.
const CLEAR_NANS: [u64; 5] = [
    0x7ff8_0000_0000_0000,
    0xfff8_0000_0000_0000,
    0x7ff8_4000_0000_0000,
    0x7fff_ffff_ffff_ffff,
    0x7ff0_0000_0000_0001,
];

/// A pass's clear of a colour attachment to a NaN, and its store, stores Metal's bytes, bit for
/// bit, as the runtime writes it: each of [`CLEAR_NANS`] in red and blue, and in green and
/// alpha, beside other values; of half, float and packed float formats (`R16Float`,
/// `RG16Float`, `RGBA16Float`, `R32Float`, `RG32Float`, `RGBA32Float`, `RG11B10Float`,
/// `RGB9E5Float`) and of the extended-range ones (`BGRA10_XR`, `BGR10_XR`, `RGB10A8_2P_XR10`
/// and their `_sRGB` twins; Metal makes no `RGBA16Float_XR` texture); 16 and 4 texels a side;
/// of `RENDER` usage, which Metal keeps losslessly compressed, and of shader writes too, which it
/// keeps uncompressed; at one sample, and at two and four of the half, float and `RG11B10Float`
/// formats, the resolve's bytes and, at 16 a side, each sample's; in a pass with no draw call,
/// and, of the plain and the negative NaN, in one with a draw call of no vertices or of one, the
/// resolve of a multisampled float attachment cleared to a NaN in a pass whose draw draws nothing
/// among them (Metal counts the draw and resolves to `7fc00000`, which the runtime writes after
/// the pass, its `nan_clear.rs`). The runtime refuses a pass of an RGBX format (the module docs),
/// whatever it clears to; a multisampled extended-range one's resolve is
/// `encode_render`'s `drawn_extended_range_samples_resolve_as_metals`'. Metal makes no
/// multisampled texture of a bfloat format, nor a bfloat view of one
#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "MoltenVK: the runtime records device-nan-encoding, a death in strict mode; Mac-only test, off everywhere; ledger: MoltenVK strict deaths"
)]
fn nan_clears_are_metals() {
    let Some((mp, md)) = metal() else {
        return;
    };
    let (vp, vd) = (platform(), dev());
    let (theirs, ours) = (Gpu::new(&mp, &md), Gpu::new(&vp, &vd));
    let nans: Vec<[f64; 4]> = CLEAR_NANS
        .iter()
        .map(|&bits| f64::from_bits(bits))
        .flat_map(|v| [[v, 0.5, v, 1.0], [0.25, v, 0.75, v]])
        .collect();
    let formats: [(u64, usize, &str, &[u64]); 14] = [
        (25, 2, "R16Float", &[1, 2, 4]),
        (65, 4, "RG16Float", &[1, 2, 4]),
        (115, 8, "RGBA16Float", &[1, 2, 4]),
        (55, 4, "R32Float", &[1, 2, 4]),
        (105, 8, "RG32Float", &[1, 2, 4]),
        (125, 16, "RGBA32Float", &[1, 2, 4]),
        (92, 4, "RG11B10Float", &[1, 2, 4]),
        (93, 4, "RGB9E5Float", &[1]),
        (552, 8, "BGRA10_XR", &[1]),
        (553, 8, "BGRA10_XR_sRGB", &[1]),
        (554, 4, "BGR10_XR", &[1]),
        (555, 4, "BGR10_XR_sRGB", &[1]),
        (550, 4, "RGB10A8_2P_XR10", &[1]),
        (551, 4, "RGB10A8_2P_XR10_sRGB", &[1]),
    ];
    let mut differ = Vec::new();
    let mut cases = 0;
    for (format, texel, name, counts) in formats {
        for (at, &colour) in nans.iter().enumerate() {
            let draws: &[Option<u64>] = match at < 4 {
                true => &[None, Some(0), Some(1)],
                false => &[None],
            };
            let shapes = counts
                .iter()
                .flat_map(|&s| draws.iter().map(move |&d| (s, d)));
            for usage in [RENDER, RENDER | READ_WRITE] {
                for side in [SIDE, 4] {
                    for (samples, drawn) in shapes.clone() {
                        let shape = (usage, side, samples);
                        let of = format!(
                            "{name} {samples} samples {side} a side, usage {usage:#x}, drawn \
                             {drawn:?}, clear to {:#x?}",
                            colour.map(f64::to_bits)
                        );
                        let metals = nan_cleared(&theirs, (format, texel), shape, colour, drawn)
                            .unwrap_or_else(|| panic!("{of}: Metal makes no such texture"));
                        let mine =
                            survived(|| nan_cleared(&ours, (format, texel), shape, colour, drawn));
                        cases += 1;
                        match mine {
                            Ok(Some(mine)) if mine == metals => {}
                            Ok(None) => differ.push(format!("{of}: no texture made")),
                            Ok(Some(mine)) => {
                                let at = (metals.0.iter().zip(&mine.0)).position(|(m, o)| m != o);
                                let at = at.map_or(0, |at| at / 16 * 16);
                                let end = metals.0.len().min(at + 16);
                                differ.push(format!(
                                    "{of}: from byte {at}, Metal {:02x?}, ours {:02x?}{}",
                                    &metals.0[at..end],
                                    &mine.0[at..end],
                                    if mine.1 == metals.1 {
                                        ""
                                    } else {
                                        ", samples differ"
                                    }
                                ));
                            }
                            Err((site, line)) => {
                                differ.push(format!("{of}: died at {site:?}: {line}"));
                            }
                        }
                    }
                }
            }
        }
    }
    assert_eq!(cases, 2016);
    assert!(differ.is_empty(), "{} differ: {differ:#?}", differ.len());
}

/// The clears of a 16 by 16 `RGBA16Float` texture of 4 samples
/// [`clears_and_bfloat_fills_are_metals`] makes, red, green, blue and alpha: plain values; and
/// half's largest, an infinity and negative zero. Its clears to a NaN are
/// [`nan_clears_are_metals`]'.
const FLOAT_CLEARS: [[f64; 4]; 2] = [
    [0.25, 1.0, 0.5, -2.0],
    [65_504.0, f64::NEG_INFINITY, -0.0, 1.0e-3],
];

/// Adds to `differ` what of `read`, of the case `of`, a read, sample or gather through an
/// argument buffer ([`Gpu::held_float_reads`], [`Gpu::held_samples`]), is other than the
/// dispatch's death naming `named`. The runtime reads an unswizzled `BGRA10_XR`, `BGR10_XR`,
/// `R10Unorm_X6`, `RG10Unorm_X12` or bfloat texture through an argument buffer as at a slot, its
/// device heap holding it by its raw texels, and samples and gathers it so as at a slot: a 2D
/// sample or gather of a `float` or `half` result, through a constexpr
/// sampler or a sampler slot of one filter and no mip filter or anisotropy. Any other sample or
/// gather of one a command reaches (names, or makes resident) it refuses as it binds the
/// command's sets (`Unsupported` of `ArgumentBufferTextures`), announcing nothing; one it does
/// not reach changes nothing for the command, whose samples of other textures are Metal's.
/// Their `_sRGB` twins and swizzled ones it refuses there (`ArgumentBufferTextures`).
fn held_reads_die<R>(
    differ: &mut Vec<String>,
    of: &str,
    named: &str,
    read: Result<R, (DeathSite, String)>,
) {
    let dispatch = "dispatchThreads:threadsPerThreadgroup:";
    match read {
        Err((site, line))
            if (site.kind, site.file) == (DeathKind::Abort, dispatch) && line.contains(named) => {}
        Err((site, line)) => differ.push(format!("{of}: died at {site:?}: {line}")),
        Ok(_) => differ.push(format!("{of}: taken")),
    }
}

/// A read of random bytes through an argument buffer that holds the texture's ID
/// ([`Gpu::held_float_reads`]: `read()` as floats and as halves, and its width, height and level
/// count) of a texture of `BGRA10_XR` and `BGR10_XR`, Shared and Private, and of `R10Unorm_X6`
/// and `RG10Unorm_X12` over a Shared buffer (the runtime makes those over a buffer or a surface
/// alone), with a sample or a gather of it so through a nearest or linear sampler of no mip
/// filter ([`Gpu::held_samples`]), dies at the dispatch naming `half-texel-conversion`
/// ([`crate::answer::DIES`]): the runtime announces the kind for a `half` read of an argument
/// buffer's texture, whose format the function does not know and whose texel it converts by a
/// rounding Metal's may not be (and `device-nan-encoding` after it, though none of these
/// formats holds a NaN). A sample or gather so of an `RGBA8Unorm` texture while it lives, which the
/// command does not reach, is Metal's. Which Metal makes and dies at the dispatch
/// ([`held_reads_die`]): a sample or a gather of it so through a sampler of a mip filter, and a
/// read of one of their `_sRGB` twins, and of a `BGRA10_XR` texture under a swizzle, each naming
/// the runtime's refusal, `ArgumentBufferTextures`
#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "MoltenVK: the runtime records native-half-conversion, a death in strict mode; Mac-only test, off everywhere; ledger: MoltenVK strict deaths"
)]
fn extended_range_textures_an_argument_buffer_holds_read_as_metals() {
    /// A 16 by 16 texture of `format` holding `bytes`: over a Shared buffer where `over`, else of
    /// `options` ([`Gpu::texture`]).
    fn made<P: Platform>(
        g: &Gpu<'_, P>,
        format: u64,
        (over, options): (bool, u64),
        bytes: &[u8],
    ) -> P::Texture {
        match over {
            true => g.over_buffer(format, bytes),
            false => g.texture(format, READ, options, bytes),
        }
    }
    let Some((mp, md)) = metal() else {
        return;
    };
    let (vp, vd) = (platform(), dev());
    let (theirs, ours) = (Gpu::new(&mp, &md), Gpu::new(&vp, &vd));
    // Each format, its name and texel's bytes, whether a buffer holds it, and its `_sRGB` twin.
    let formats = [
        (552, "BGRA10_XR", 8, false, 553),
        (554, "BGR10_XR", 4, false, 555),
        (576, "R10Unorm_X6", 2, true, 577),
        (578, "RG10Unorm_X12", 4, true, 579),
    ];
    let refused = "ArgumentBufferTextures";
    let (mut differ, mut nans, mut died) = (Vec::new(), 0, Vec::new());
    let sel = "dispatchThreads:threadsPerThreadgroup:";
    let named = "(metal2vulkan-degrade-half-texel-conversion)";
    for (seed, (format, name, texel, over, twin)) in (1u64..).zip(formats) {
        let options: &[u64] = if over { &[SHARED] } else { &[SHARED, PRIVATE] };
        for &options in options {
            let shape = (over, options);
            let bytes = Rng(0x6a09_e667_f3bc_c908 ^ seed).bytes(TEXELS * texel);
            let of = format!("{name} options {options:#x}");
            let t = made(&theirs, format, shape, &bytes);
            let mut metals = theirs.held_float_reads(&t);
            metals.extend(theirs.held_samples(&t, false));
            theirs.held_samples(&t, true);
            let t = made(&ours, format, shape, &bytes);
            let read = || {
                let mut mine = ours.held_float_reads(&t);
                mine.extend(ours.held_samples(&t, false));
                mine
            };
            match survived(read) {
                Ok(mine) => compare(&mut differ, &mut nans, &of, &metals, &mine),
                Err((site, line))
                    if (site.kind, site.file) == (DeathKind::Abort, sel)
                        && line.contains(named) =>
                {
                    died.push(of.clone());
                }
                Err((site, line)) => differ.push(format!("{of} reads: died at {site:?}: {line}")),
            }
            let samples = format!("{of} samples through a mip filter");
            held_reads_die(
                &mut differ,
                &samples,
                refused,
                survived(|| ours.held_samples(&t, true)),
            );
            let other = Rng(0xbb67_ae85_84ca_a73b ^ seed).bytes(TEXELS * 4);
            let metals = theirs.held_samples(&theirs.texture(70, READ, SHARED, &other), false);
            let plain = ours.texture(70, READ, SHARED, &other);
            let beside = format!("{of}: an RGBA8Unorm texture's samples beside it");
            match survived(|| ours.held_samples(&plain, false)) {
                Ok(mine) => compare(&mut differ, &mut nans, &beside, &metals, &mine),
                Err((site, line)) => differ.push(format!("{beside}: died at {site:?}: {line}")),
            }
            let of = format!("{name}_sRGB options {options:#x}");
            theirs.held_float_reads(&made(&theirs, twin, shape, &bytes));
            let t = made(&ours, twin, shape, &bytes);
            held_reads_die(
                &mut differ,
                &of,
                refused,
                survived(|| ours.held_float_reads(&t)),
            );
        }
    }
    let bytes = Rng(0x6a09_e667_f3bc_c908).bytes(TEXELS * 8);
    let [red, green, blue, alpha] = XR_SWIZZLES[1];
    let swizzle = Swizzle {
        red,
        green,
        blue,
        alpha,
    };
    theirs.held_float_reads(&theirs.swizzled(552, swizzle, &bytes));
    let t = ours.swizzled(552, swizzle, &bytes);
    let of = "BGRA10_XR under a swizzle";
    held_reads_die(
        &mut differ,
        of,
        refused,
        survived(|| ours.held_float_reads(&t)),
    );
    assert!(differ.is_empty(), "{} differ: {differ:#?}", differ.len());
    assert_eq!(nans, 0, "{nans} threads' half NaNs differ");
    let all = [
        "BGRA10_XR options 0x0",
        "BGRA10_XR options 0x20",
        "BGR10_XR options 0x0",
        "BGR10_XR options 0x20",
        "R10Unorm_X6 options 0x0",
        "RG10Unorm_X12 options 0x0",
    ];
    assert_eq!(died, all, "the reads that die naming half-texel-conversion");
}

/// The swizzles [`extended_range_textures_read_as_metals`] reads through, as
/// `MTLTextureSwizzle`s (zero 0, one 1, red 2, green 3, blue 4, alpha 5): the identity, and four
/// that move, repeat, drop and replace channels.
const XR_SWIZZLES: [[u8; 4]; 5] = [
    [2, 3, 4, 5],
    [5, 2, 1, 0],
    [4, 3, 2, 5],
    [3, 5, 4, 2],
    [0, 1, 5, 3],
];

/// Reads of random bytes of a Private texture of each single-plane extended-range format
/// (`BGRA10_XR`, `BGRA10_XR_sRGB`, `BGR10_XR`, `BGR10_XR_sRGB`), under each of [`XR_SWIZZLES`],
/// the texture's own swizzle and a view's of an unswizzled texture, bound to the kernel's
/// texture slot ([`Gpu::direct_reads`]): `read()` as floats and halves, in bounds and out,
/// `sample()` as floats and halves through each filter, address mode and border colour, and
/// `gather()` of each component. Each is Metal's, bit for bit. The runtime decodes such a
/// texture's codes before it swizzles them, a border texel included, and gathers these formats
/// as floats (measured on an Apple M4 under macOS 27.0.1)
#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "MoltenVK: the runtime records device-pack-conversion, a death in strict mode; Mac-only test, off everywhere; ledger: MoltenVK strict deaths"
)]
fn extended_range_textures_read_as_metals() {
    let Some((mp, md)) = metal() else {
        return;
    };
    let (vp, vd) = (platform(), dev());
    let (theirs, ours) = (Gpu::new(&mp, &md), Gpu::new(&vp, &vd));
    let (mut differ, mut nans) = (Vec::new(), 0);
    let formats = [
        (552, "BGRA10_XR", 8),
        (553, "BGRA10_XR_sRGB", 8),
        (554, "BGR10_XR", 4),
        (555, "BGR10_XR_sRGB", 4),
    ];
    for (seed, (format, name, texel)) in (1u64..).zip(formats) {
        let bytes = Rng(0xa54f_f53a_5f1d_36f1 ^ seed).bytes(TEXELS * texel);
        for [red, green, blue, alpha] in XR_SWIZZLES {
            let swizzle = Swizzle {
                red,
                green,
                blue,
                alpha,
            };
            let of = format!("{name} swizzle {:?}", [red, green, blue, alpha]);
            let metals =
                theirs.direct_reads(Class::Float, &theirs.swizzled(format, swizzle, &bytes));
            match survived(|| {
                ours.direct_reads(Class::Float, &ours.swizzled(format, swizzle, &bytes))
            }) {
                Ok(mine) => compare(&mut differ, &mut nans, &of, &metals, &mine),
                Err((site, line)) => differ.push(format!("{of}: died at {site:?}: {line}")),
            }
            let of = format!("{of}, a view's");
            let metals = theirs.direct_reads(Class::Float, &theirs.view(format, swizzle, &bytes));
            match survived(|| ours.direct_reads(Class::Float, &ours.view(format, swizzle, &bytes)))
            {
                Ok(mine) => compare(&mut differ, &mut nans, &of, &metals, &mine),
                Err((site, line)) => differ.push(format!("{of}: died at {site:?}: {line}")),
            }
        }
    }
    assert!(differ.is_empty(), "{} differ: {differ:#?}", differ.len());
    assert_eq!(nans, 0);
}

/// The storage slots of `RGBX8Unorm_sRGB` ([`SRGB_X`]) and of its RGBA twin `RGBA8Unorm_sRGB`
/// ([`SRGB_RGBA`]) are Metal's, bit for bit, in textures of `ShaderRead` and `ShaderWrite` usage,
/// Shared and Private ([`Gpu::srgb_storage`]): every code of each colour field read by `read()`,
/// `sample()` and `gather()` as floats and as halves, and through `read_write`; `write()` of the
/// floats either side of each step of Metal's twelve-bit encode table and of every half; and a
/// `read_write` write-back of every code. The runtime decodes and encodes those slots through
/// Metal's twelve-bit tables, and notes no device transcendentals for them, so a
/// texture of `RGBX8Unorm_sRGB` a function may write is made, as Metal makes it
#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "MoltenVK: the runtime records device-pack-conversion, a death in strict mode; Mac-only test, off everywhere; ledger: MoltenVK strict deaths"
)]
fn srgb_storage_reads_and_writes_are_metals() {
    let Some((mp, md)) = metal() else {
        return;
    };
    let (vp, vd) = (platform(), dev());
    let (theirs, ours) = (Gpu::new(&mp, &md), Gpu::new(&vp, &vd));
    crate::answer::RELAYED.lock().clear();
    let (mut differ, mut nans) = (Vec::new(), 0);
    for (format, name) in [(SRGB_X, "RGBX8Unorm_sRGB"), (SRGB_RGBA, "RGBA8Unorm_sRGB")] {
        for options in [SHARED, PRIVATE] {
            let metals = theirs.srgb_storage(format, options);
            let of = format!("{name} options {options:#x}");
            match survived(|| ours.srgb_storage(format, options)) {
                Ok(mine) => compare(&mut differ, &mut nans, &of, &metals, &mine),
                Err((site, line)) => differ.push(format!("{of}: died at {site:?}: {line}")),
            }
        }
    }
    assert!(differ.is_empty(), "{} differ: {differ:#?}", differ.len());
    assert_eq!(nans, 0);
    let relayed = crate::answer::RELAYED.lock().clone();
    let note = rt::Degradation::DeviceTranscendentals;
    assert!(!relayed.contains(&note), "{} was relayed", note.code());
}

/// Stores into a region of a texture of each RGBX format holding random bytes, Shared and
/// Private, of each of [`stored_usages`]: `replaceRegion:` (Shared), blits from a buffer and from
/// another such texture, the private byte fill, colour fills in the RGBA twin's pixel format, and
/// a blit of the region into a buffer: the bytes stored are Metal's
#[test]
fn stores_into_rgbx_textures_are_metals() {
    let Some((mp, md)) = metal() else {
        return;
    };
    let (vp, vd) = (platform(), dev());
    let (theirs, ours) = (Gpu::new(&mp, &md), Gpu::new(&vp, &vd));
    let (mut differ, mut nans) = (Vec::new(), 0);
    for (seed, (format, name, texel, _, twin)) in (1u64..).zip(FORMATS) {
        for (&usage, options) in stored_usages(format)
            .iter()
            .flat_map(|u| [(u, SHARED), (u, PRIVATE)])
        {
            let bytes = Rng(0x3c6e_f372_fe94_f82b ^ seed).bytes(TEXELS * texel);
            let other = Rng(0xa54f_f53a_5f1d_36f1 ^ seed).bytes(TEXELS * texel);
            let shape = (format, texel, twin);
            let metals = theirs.stores(shape, (usage, options), &bytes, &other);
            let of = format!("{name} usage {usage:#x} options {options:#x}");
            let mine = survived(|| ours.stores(shape, (usage, options), &bytes, &other));
            match mine {
                Ok(mine) => compare(&mut differ, &mut nans, &of, &metals, &mine),
                Err((site, line)) => differ.push(format!("{of}: died at {site:?}: {line}")),
            }
        }
    }
    assert!(differ.is_empty(), "{} differ: {differ:#?}", differ.len());
    assert_eq!(nans, 0);
}

/// What the runtime refuses of a texture of each RGBX format dies here where it is encoded,
/// naming the refusal, where Metal runs it (Metal's side first; a pass and mipmaps only of the
/// formats Metal renders into): a pass that clears it and stores it, with `RenderTarget` usage
/// or without (`Format`); the mipmaps of one of two levels (`MipmapGeneration`); a kernel's read
/// of a texture buffer of the format (`Format`); a kernel's write into such a texture an
/// argument buffer holds (`ArgumentBufferTextures`). A colour fill in the format, which the
/// runtime's colour fill packs ([`crate::gpu::fill::OWN_FORMAT_PACKED`]), is compared with Metal's
/// by `encode_blit`'s `colours_the_runtime_packs_are_the_metal_platforms`
#[test]
fn what_the_runtime_refuses_of_rgbx_textures_dies() {
    const PASS: &str = "renderCommandEncoderWithDescriptor:";
    const MIPMAPS: &str = "generateMipmapsForTexture:";
    const DISPATCH: &str = "dispatchThreads:threadsPerThreadgroup:";
    let theirs = metal();
    let theirs = theirs.as_ref().map(|(mp, md)| Gpu::new(mp, md));
    let (vp, vd) = (platform(), dev());
    let ours = Gpu::new(&vp, &vd);
    let mut differ = Vec::new();
    for (format, name, texel, class, _) in FORMATS {
        if let Some(g) = &theirs {
            if RENDERED.contains(&format) {
                g.pass(format, RENDER);
                g.mipmaps(format);
            }
            g.texture_buffer(format, texel, class);
            g.held_write(format, texel, class);
        }
        let cases: [(&str, &str, &str, &dyn Fn()); 5] = [
            ("a pass", PASS, "Format", &|| ours.pass(format, RENDER)),
            ("a pass without RenderTarget usage", PASS, "Format", &|| {
                ours.pass(format, READ)
            }),
            ("mipmaps", MIPMAPS, "MipmapGeneration", &|| {
                ours.mipmaps(format);
            }),
            ("a texture buffer", DISPATCH, "Format", &|| {
                ours.texture_buffer(format, texel, class);
            }),
            (
                "a write through an argument buffer",
                DISPATCH,
                "ArgumentBufferTextures",
                &|| ours.held_write(format, texel, class),
            ),
        ];
        for (case, sel, named, run) in cases {
            match survived(run) {
                Ok(()) => differ.push(format!("{name} {case}: taken")),
                Err((site, line)) if (site.kind, site.file) != (DeathKind::Abort, sel) => {
                    differ.push(format!("{name} {case}: died at {site:?}: {line}"));
                }
                Err((_, line)) if !line.contains(named) => {
                    differ.push(format!("{name} {case}: names no {named}: {line}"));
                }
                Err(_) => {}
            }
        }
    }
    assert!(differ.is_empty(), "{} differ: {differ:#?}", differ.len());
}

/// A kernel's `write()` of halves, NaNs among them, into two of every three texels of a texture
/// of `RGBX16Float`, `RGBX32Float` and their RGBA twins, `RGBA16Float` and `RGBA32Float`, Shared
/// and Private, is Metal's, and nothing dies: into a 32-bit texel, whose NaN Metal and the runtime
/// store as the canonical `0x7fc00000`, and into a 16-bit texel, whose half
/// the runtime writes through an unsigned integer view, keeping its bits, a signalling NaN's too,
/// as Metal stores them (`RGBX16Float` through its RGBA twin's view, green's bits in the
/// fourth field, Metal's X). With [`kernel_writes_of_rgbx_textures_are_metals`]'s values and
/// texels
#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "MoltenVK: the runtime records device-nan-encoding, a death in strict mode; Mac-only test, off everywhere; ledger: MoltenVK strict deaths"
)]
fn half_nan_writes_are_metals() {
    let Some((mp, md)) = metal() else {
        return;
    };
    let (vp, vd) = (platform(), dev());
    let (theirs, ours) = (Gpu::new(&mp, &md), Gpu::new(&vp, &vd));
    // Each format, its texel's bytes, and the seed of the writes test's values.
    let cases = [(315, 8, 8), (325, 16, 11), (115, 8, 8), (125, 16, 11)];
    let mut differ = Vec::new();
    for (format, texel, seed) in cases {
        for options in [SHARED, PRIVATE] {
            let of = format!("{format} options {options:#x}");
            let bytes = Rng(0xbb67_ae85_84ca_a73b ^ seed).bytes(TEXELS * texel);
            let metals = theirs.half_writes((format, texel), options, &bytes, seed);
            match survived(|| ours.half_writes((format, texel), options, &bytes, seed)) {
                Ok(mine) if mine == metals => {}
                Ok(_) => differ.push(format!("{of}: the texels differ")),
                Err((site, line)) => differ.push(format!("{of}: died at {site:?}: {line}")),
            }
        }
    }
    assert!(differ.is_empty(), "{differ:#?}");
}

/// A texture of usage 0x8 alone, which Metal's texture answers no usage for and keeps as one of
/// `ShaderRead` alone (the module docs of [`crate::gpu::desc`]): a `BGRA8Unorm` one, Shared and
/// Private, holding random bytes, answers usage 0 as Metal's does, and so does a view of it in
/// its sRGB format; its reads, samples through each sampler and gathers are Metal's, bit for bit,
/// and so are a pass's clear of it and the heap's size query of a 64 by 64 one (16512 bytes,
/// where a texture of `Unknown` usage answers 16384); a swizzled view of it dies, where Metal
/// asserts (`_usage != MTLTextureUsageUnknown`, measured)
#[test]
fn a_texture_of_usage_0x8_alone_is_metals() {
    /// The usage bit Metal's texture drops.
    const DROPPED: u64 = 0x8;
    /// `MTLPixelFormatBGRA8Unorm`, and its sRGB twin.
    const BGRA: (u64, u64) = (80, 81);
    let Some((mp, md)) = metal() else {
        return;
    };
    let (vp, vd) = (platform(), dev());
    let (theirs, ours) = (Gpu::new(&mp, &md), Gpu::new(&vp, &vd));
    let (mut differ, mut nans) = (Vec::new(), 0);
    for options in [SHARED, PRIVATE] {
        let bytes = Rng(0x0008_d409 ^ options).bytes(TEXELS * 4);
        let metals = theirs.texture(BGRA.0, DROPPED, options, &bytes);
        let mine = ours.texture(BGRA.0, DROPPED, options, &bytes);
        let of = format!("options {options:#x}");
        let view = |t: &dyn Fn() -> u64| t();
        assert_eq!(
            (
                mine.usage(),
                view(&|| mine.new_texture_view(BGRA.1).unwrap().unwrap().usage())
            ),
            (0, 0),
            "{of}"
        );
        assert_eq!(
            (
                metals.usage(),
                view(&|| metals.new_texture_view(BGRA.1).unwrap().unwrap().usage())
            ),
            (0, 0),
            "{of}: Metal's"
        );
        let (m, o) = (
            theirs.direct_reads(Class::Float, &metals),
            ours.direct_reads(Class::Float, &mine),
        );
        compare(&mut differ, &mut nans, &of, &m, &o);
    }
    assert!(differ.is_empty(), "{} differ: {differ:#?}", differ.len());
    assert_eq!(nans, 0);
    let colour = [0.25, 0.5, 0.75, 1.0];
    assert_eq!(
        ours.read(&ours.cleared(BGRA.0, DROPPED, colour), 4),
        theirs.read(&theirs.cleared(BGRA.0, DROPPED, colour), 4),
        "a pass's clear"
    );
    let sized = TextureDesc {
        pixel_format: BGRA.0,
        width: 64,
        height: 64,
        usage: DROPPED,
        resource_options: Some(PRIVATE),
        ..TextureDesc::default()
    };
    let size = vd
        .heap_texture_size_and_align(&vp.texture_descriptor(&sized))
        .unwrap();
    let metals = md
        .heap_texture_size_and_align(&mp.texture_descriptor(&sized))
        .unwrap();
    assert_eq!((size, size.size), (metals, 16512));
    let t = ours.texture_of(BGRA.0, DROPPED, PRIVATE, 1);
    let one = NsRange::new(0, 1);
    let swapped = Swizzle {
        red: 4,
        green: 3,
        blue: 2,
        alpha: 5,
    };
    let died = catch_death(|| t.new_swizzled_texture_subview(BGRA.0, 2, one, one, swapped));
    let site = died.map(drop).unwrap_err();
    assert_eq!(
        (site.kind, site.file),
        (
            DeathKind::Abort,
            "newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:"
        )
    );
}
