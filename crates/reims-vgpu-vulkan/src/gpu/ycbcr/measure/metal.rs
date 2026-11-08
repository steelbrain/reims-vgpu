//! The measurement harness's Metal side: Metal's textures over a surface, and a surface's values
//! (`IOSurfaceSetValue`), which only IOSurface itself takes.

use std::ffi::c_void;

use reims_vgpu_core::platform::Handle as _;
use reims_vgpu_core::platform::Platform;
use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
use reims_vgpu_core::platform::compute::{ComputeCmd, ComputeEncoder as _};
use reims_vgpu_core::platform::device::GpuDevice as _;
use reims_vgpu_core::platform::resource::{Buffer as _, Library as _};
use reims_vgpu_core::platform::types::{MtlSize, NsRange};
use reims_vgpu_core::platform::values::{SamplerDesc, TextureDesc};
use reims_vgpu_core::raise::OnFault;
use reims_vgpu_metal::MetalPlatform;

use super::*;
use crate::os::OsSurface;

#[link(name = "IOSurface", kind = "framework")]
unsafe extern "C" {
    fn IOSurfaceSetValue(buffer: *const c_void, key: *const c_void, value: *const c_void);
}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    fn CFStringCreateWithBytes(
        alloc: *const c_void,
        bytes: *const u8,
        num_bytes: isize,
        encoding: u32,
        is_external_representation: u8,
    ) -> *const c_void;
    fn CFRelease(cf: *const c_void);
}

/// A CFString of `s`, +1.
fn cf_string(s: &str) -> *const c_void {
    // SAFETY: `s`'s bytes, live for the call, which copies them.
    unsafe {
        CFStringCreateWithBytes(
            std::ptr::null(),
            s.as_ptr(),
            isize::try_from(s.len()).unwrap(),
            0x0800_0100,
            0,
        )
    }
}

/// Sets `surface`'s `key` to the string `value` (`IOSurfaceSetValue`).
pub(crate) fn set_value(surface: &OsSurface, key: &str, value: &str) {
    let (key, value) = (cf_string(key), cf_string(value));
    let surface =
        std::ptr::with_exposed_provenance::<c_void>(usize::try_from(surface.id().0).unwrap());
    // SAFETY: a live IOSurfaceRef and two CFStrings, released after the call, which retains
    // what it keeps.
    unsafe {
        IOSurfaceSetValue(surface, key, value);
        CFRelease(key);
        CFRelease(value);
    }
}

/// Metal's texture of `format` of `size` over plane `plane` of `surface`, and its device; `None`
/// with no Metal device.
pub(crate) fn metals(
    format: u64,
    surface: &OsSurface,
    plane: u64,
    size: (u64, u64),
) -> Option<(
    <MetalPlatform as Platform>::Device,
    <MetalPlatform as Platform>::Texture,
)> {
    let md = MetalPlatform::system_default_device()?;
    let desc = reims_vgpu_metal::gpu::desc::texture_descriptor(&desc(format, size));
    let t = md
        .new_texture_on_surface(&desc, Some(surface), plane)
        .unwrap()
        .unwrap();
    Some((md, t))
}

/// The tests' kernels that read a texture (`ycbcr_probe.metal`).
pub(crate) const PROBE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ycbcr_probe.metallib"));

/// The layout of [`surface_420`]: 10-bit codes in 16-bit samples, the chroma 4:2:0, whatever
/// `ostype` names.
fn shape_420(ostype: &[u8; 4]) -> Shape {
    Shape {
        ostype: *ostype,
        subsampling: (2, 2),
        bits: 10,
        container: 2,
        packed: false,
    }
}

/// A host-only surface of `ostype` with the two planes of a `width` by `height` texture of
/// 505, rows of a multiple of 64 bytes and plane 1 at a multiple of 4096, as the measured
/// surfaces were laid out ([`Shape::spec`]).
pub(crate) fn surface_420(ostype: &[u8; 4], size: (u64, u64)) -> OsSurface {
    shape_420(ostype).surface(size)
}

/// [`surface_420`], made over pages of its own the host allocates, as a guest's surface is
/// made over the guest's pages (`kIOSurfaceClientAddressRanges`), and those pages: their first
/// byte and length, which the caller frees once the surface is gone.
pub(crate) fn surface_420_over(
    ostype: &[u8; 4],
    size: (u64, u64),
) -> (OsSurface, reims_vgpu_core::host::HostPtr, u64) {
    surface_420_over_at(ostype, size, 0)
}

/// [`surface_420_over`], its bytes `offset` bytes into its pages (`kIOSurfaceOffset`), its
/// planes' offsets too, which count from the pages' first byte.
pub(crate) fn surface_420_over_at(
    ostype: &[u8; 4],
    size: (u64, u64),
    offset: u64,
) -> (OsSurface, reims_vgpu_core::host::HostPtr, u64) {
    use reims_vgpu_core::host::PhysicalMemoryRange;
    use reims_vgpu_core::num::ToU64;
    use reims_vgpu_core::platform::os::vm::HostVm as _;
    let mut spec = spec_420(ostype, size);
    let vm = crate::os::OsVm::new();
    let page = vm.page_size();
    for plane in spec.planes.iter_mut().flatten() {
        plane.offset += offset;
        plane.base += offset;
    }
    spec.offset = offset;
    spec.alloc_size = (spec.alloc_size + offset).next_multiple_of(page);
    let pages = vm.allocate_anywhere(spec.alloc_size).unwrap();
    let first = pages.addr().to_u64();
    spec.client_address_ranges = Some(
        (0..spec.alloc_size / page)
            .map(|i| PhysicalMemoryRange {
                physical_address: first + i * page,
                physical_length: page,
            })
            .collect(),
    );
    let s = OsSurfaces::new().create(&spec).unwrap();
    (s, pages, spec.alloc_size)
}

/// The spec of [`surface_420`]'s surface.
fn spec_420(ostype: &[u8; 4], size: (u64, u64)) -> SurfaceSpec {
    shape_420(ostype).spec(size)
}

/// A xorshift sequence from `seed`: the same 16-bit values on every run.
pub(crate) fn noise(seed: u64) -> impl FnMut() -> u16 {
    let mut state = seed.max(1);
    move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        u16::try_from(state >> 48).unwrap()
    }
}

/// Fills `surface`, of [`surface_420`]'s layout, with `noise(seed)`'s samples, the low 6 bits
/// too.
pub(crate) fn fill_noise(surface: &OsSurface, size: (u64, u64), seed: u64) {
    shape_420(b"x420").fill_noise(surface, size, seed);
}

/// What `kernel` of [`PROBE`] writes on `p`'s device `d`, over `t`: a texel of 8 bytes for
/// each of a grid of `grid`, with the sampler `sampler` and the probe `probe` (its bytes)
/// where given.
pub(crate) fn probed<P: Platform>(
    d: &P::Device,
    kernel: &str,
    t: &P::Texture,
    grid: (u64, u64),
    sampler: Option<&P::SamplerState>,
    probe: Option<&[u8]>,
) -> Vec<u8> {
    probed_texels::<P>(d, (kernel, 8), t, grid, sampler, probe)
}

/// [`probed`] of a kernel that writes texels of `texel` bytes.
pub(crate) fn probed_texels<P: Platform>(
    d: &P::Device,
    (kernel, texel): (&str, u64),
    t: &P::Texture,
    grid: (u64, u64),
    sampler: Option<&P::SamplerState>,
    probe: Option<&[u8]>,
) -> Vec<u8> {
    // What Metal autoreleases for the probe (its command buffer and encoder, which a dev build
    // cannot elide) goes at its end.
    let _pool = reims_vgpu_core::rt::Pool::here();
    let library = d.new_library(PROBE).unwrap().unwrap().unwrap();
    let f = library.new_function(OnFault::Die, kernel).unwrap().unwrap();
    let state = d
        .new_compute_pipeline_state_for_function(Some(&f))
        .unwrap()
        .unwrap();
    let length = grid.0 * grid.1 * texel;
    let out = d
        .new_buffer_of_length(OnFault::Die, length, 0)
        .unwrap()
        .unwrap();
    let probe = probe.map(|bytes| {
        let b = d
            .new_buffer_of_length(OnFault::Die, u64::try_from(bytes.len()).unwrap(), 0)
            .unwrap()
            .unwrap();
        // SAFETY: a Shared buffer of `bytes.len()` bytes, which no command uses yet.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), b.contents().as_ptr::<u8>(), bytes.len());
        }
        b
    });
    let q = crate::gpu::device::tests::metal_tests::test_queue::<P>(d);
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let e = cb.compute_encoder(OnFault::Die).unwrap().unwrap();
    let buffers = [Some(&out), probe.as_ref()];
    let bound = if probe.is_some() { 2 } else { 1 };
    let samplers = [sampler];
    let mut cmds = vec![
        ComputeCmd::SetComputePipelineState {
            state: Some(&state),
        },
        ComputeCmd::SetTexture {
            texture: Some(t),
            index: 0,
        },
        ComputeCmd::SetBuffers {
            buffers: &buffers[..bound],
            offsets: &[0, 0][..bound],
            range: NsRange::new(0, u64::try_from(bound).unwrap()),
        },
    ];
    if sampler.is_some() {
        cmds.push(ComputeCmd::SetSamplerStates {
            samplers: &samplers,
            range: NsRange::new(0, 1),
        });
    }
    cmds.push(ComputeCmd::DispatchThreads {
        threads_per_grid: MtlSize::new(grid.0, grid.1, 1),
        threads_per_threadgroup: MtlSize::new(8, 8, 1),
    });
    cmds.push(ComputeCmd::EndEncoding);
    for cmd in cmds {
        e.encode(OnFault::Die, cmd).unwrap();
    }
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    assert!(cb.error().is_none(), "{:?}", cb.error());
    let n = usize::try_from(length).unwrap();
    // SAFETY: a Shared buffer's bytes, read after the command buffer completed.
    unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), n) }.to_vec()
}

/// A 2D descriptor of 505 of `size`, one level, usage `ShaderRead`, storage `Shared`.
pub(crate) fn desc_505((width, height): (u64, u64)) -> TextureDesc {
    TextureDesc {
        texture_type: 2,
        pixel_format: 505,
        width,
        height,
        usage: 1,
        resource_options: Some(0),
        ..TextureDesc::default()
    }
}

/// A 2D descriptor of `format` of `size`, one level, usage `ShaderRead`, storage `Shared`.
pub(crate) fn desc(format: u64, (width, height): (u64, u64)) -> TextureDesc {
    TextureDesc {
        texture_type: 2,
        pixel_format: format,
        width,
        height,
        usage: 1,
        resource_options: Some(0),
        ..TextureDesc::default()
    }
}

/// The halves of each texel of `bytes` (four a texel), as their bits.
pub(crate) fn texels(bytes: &[u8]) -> Vec<[u16; 4]> {
    bytes
        .chunks(8)
        .map(|t| {
            let h = |i: usize| u16::from_le_bytes([t[i * 2], t[i * 2 + 1]]);
            [h(0), h(1), h(2), h(3)]
        })
        .collect()
}

/// The bits of each float of `texels`: what a comparison bit for bit compares, where `==` of
/// the floats takes -0 for +0 and no NaN for itself.
pub(crate) fn float_bits(texels: &[[f32; 4]]) -> Vec<[u32; 4]> {
    texels.iter().map(|t| t.map(f32::to_bits)).collect()
}

/// The floats of each texel of `bytes` (four a texel).
pub(crate) fn float_texels(bytes: &[u8]) -> Vec<[f32; 4]> {
    bytes
        .chunks(16)
        .map(|t| {
            let f = |i: usize| f32::from_le_bytes(t[i * 4..i * 4 + 4].try_into().unwrap());
            [f(0), f(1), f(2), f(3)]
        })
        .collect()
}

/// `read()` of every texel of `t` of `size` on `p`'s device `d`, into floats.
pub(crate) fn read_floats<P: Platform>(
    d: &P::Device,
    t: &P::Texture,
    size: (u64, u64),
) -> Vec<[f32; 4]> {
    let bytes = probed_texels::<P>(d, ("read_texels_float", 16), t, size, None, None);
    float_texels(&bytes)
}

/// The bytes of a probe (`Probe` in `ycbcr_probe.metal`): `grid` positions from `origin`, `step`
/// apart, in texels or `normalized`, and what it reads (`read`: 0 sample, 1 gather of red, 2
/// gather of blue).
pub(crate) fn probe(
    origin: (f32, f32),
    step: f32,
    grid: (u64, u64),
    normalized: bool,
    read: u32,
) -> Vec<u8> {
    let mut b = Vec::new();
    for f in [origin.0, origin.1, step, step] {
        b.extend(f.to_le_bytes());
    }
    let words = [
        u32::try_from(grid.0).unwrap(),
        u32::try_from(grid.1).unwrap(),
        u32::from(normalized),
        read,
    ];
    for w in words {
        b.extend(w.to_le_bytes());
    }
    b
}

/// What a probe of a sampler (`filter`: 0 nearest, 1 linear; `address` on both axes; normalized
/// coordinates or not; the border colour) reads of `t` on `p`'s device `d`: `sample_texels` with
/// the probe `bytes` over `grid`.
pub(crate) fn sampled<P: Platform>(
    d: &P::Device,
    t: &P::Texture,
    (filter, address, normalized, border): (u64, u64, bool, u64),
    grid: (u64, u64),
    bytes: &[u8],
) -> Vec<u8> {
    let s = sampler::<P>(d, (filter, address, normalized), border);
    probed::<P>(d, "sample_texels", t, grid, Some(&s), Some(bytes))
}

/// [`sampled`] into floats: `sample_texels_float`.
pub(crate) fn sampled_floats<P: Platform>(
    d: &P::Device,
    t: &P::Texture,
    (filter, address, normalized, border): (u64, u64, bool, u64),
    grid: (u64, u64),
    bytes: &[u8],
) -> Vec<[f32; 4]> {
    sampled_by::<P>(
        d,
        "sample_texels_float",
        t,
        (filter, address, normalized, border),
        grid,
        bytes,
    )
}

/// [`sampled_floats`] through `sample_only_texels_float`, a reader holding no gather: for a
/// texture of a format the runtime gathers none of (the two-plane XR10 formats), whose function
/// holding a gather it refuses.
pub(crate) fn sampled_only_floats<P: Platform>(
    d: &P::Device,
    t: &P::Texture,
    state: (u64, u64, bool, u64),
    grid: (u64, u64),
    bytes: &[u8],
) -> Vec<[f32; 4]> {
    sampled_by::<P>(d, "sample_only_texels_float", t, state, grid, bytes)
}

/// What float reader `kernel` reads of `t` through [`sampler`] of `state`, with the probe `bytes`
/// over `grid`.
fn sampled_by<P: Platform>(
    d: &P::Device,
    kernel: &str,
    t: &P::Texture,
    (filter, address, normalized, border): (u64, u64, bool, u64),
    grid: (u64, u64),
    bytes: &[u8],
) -> Vec<[f32; 4]> {
    let s = sampler::<P>(d, (filter, address, normalized), border);
    let read = probed_texels::<P>(d, (kernel, 16), t, grid, Some(&s), Some(bytes));
    float_texels(&read)
}

/// What `fragment` (`sample_fragment`, `probe_fragment` or either's `_float` reader, writing
/// texels of `texel` bytes) reads of `t` through `sampler` on `p`'s device `d`, with the probe
/// `bytes` over `grid`: one triangle (`cover_vertex`) over a BGRA8 target of `grid`, one fragment
/// a probe position.
pub(crate) fn drawn<P: Platform>(
    p: &P,
    d: &P::Device,
    (fragment, texel): (&str, u64),
    t: &P::Texture,
    grid: (u64, u64),
    sampler: &P::SamplerState,
    bytes: &[u8],
) -> Vec<u8> {
    use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
    use reims_vgpu_core::platform::descriptor::RenderPassDescriptor as _;
    use reims_vgpu_core::platform::render::{RenderCmd, RenderEncoder as _};
    use reims_vgpu_core::platform::resource::{Buffer as _, Library as _};
    use reims_vgpu_core::platform::types::NsRange;
    use reims_vgpu_core::platform::values::{ColorAttachmentSet, RenderPipelineDesc};
    use reims_vgpu_core::raise::OnFault;
    // What Metal autoreleases for the draw goes at its end, and its queue is the tests' one
    // (`test_queue`): a draw of its own queue each would pass the process's limit of queues.
    let _pool = reims_vgpu_core::rt::Pool::here();
    let library = d.new_library(PROBE).unwrap().unwrap().unwrap();
    let function = |name| library.new_function(OnFault::Die, name).unwrap().unwrap();
    let (vertex, fragment) = (function("cover_vertex"), function(fragment));
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
    let buffer = |length: u64| {
        d.new_buffer_of_length(OnFault::Die, length, 0)
            .unwrap()
            .unwrap()
    };
    let out = buffer(grid.0 * grid.1 * texel);
    let probe = buffer(u64::try_from(bytes.len()).unwrap());
    // SAFETY: a Shared buffer of `bytes.len()` bytes, which no command uses yet.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), probe.contents().as_ptr::<u8>(), bytes.len());
    }
    let target = d
        .new_texture(
            OnFault::Die,
            &p.texture_descriptor(&TextureDesc {
                texture_type: 2,
                pixel_format: 80,
                width: grid.0,
                height: grid.1,
                usage: 4,
                resource_options: Some(0x20),
                ..TextureDesc::default()
            }),
        )
        .unwrap()
        .unwrap();
    let pass = p.render_pass_descriptor();
    let attachment = ColorAttachmentSet {
        texture: Some(Some(&target)),
        ..ColorAttachmentSet::default()
    };
    pass.set_color_attachment(OnFault::Die, 0, &attachment)
        .unwrap();
    let q = crate::gpu::device::tests::metal_tests::test_queue::<P>(d);
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
    let buffers = [Some(&out), Some(&probe)];
    let samplers = [Some(sampler)];
    for cmd in [
        RenderCmd::SetRenderPipelineState {
            pipeline_state: Some(&state),
        },
        RenderCmd::SetFragmentTexture {
            texture: Some(t),
            index: 0,
        },
        RenderCmd::SetFragmentSamplerStates {
            samplers: &samplers,
            range: NsRange::new(0, 1),
        },
        RenderCmd::SetFragmentBuffers {
            buffers: &buffers,
            offsets: &[0, 0],
            range: NsRange::new(0, 2),
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
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    assert!(cb.error().is_none(), "{:?}", cb.error());
    let n = usize::try_from(grid.0 * grid.1 * texel).unwrap();
    // SAFETY: a Shared buffer's bytes, read after the command buffer completed.
    unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), n) }.to_vec()
}

/// A sampler state of `d`'s: `filter` (0 nearest, 1 linear) and `address` on both axes,
/// `normalized` coordinates or not, and the border colour `border`.
pub(crate) fn sampler<P: Platform>(
    d: &P::Device,
    (filter, address, normalized): (u64, u64, bool),
    border: u64,
) -> P::SamplerState {
    let desc = SamplerDesc {
        min_filter: Some(filter),
        mag_filter: Some(filter),
        s_address_mode: Some(address),
        t_address_mode: Some(address),
        normalized_coordinates: Some(normalized),
        border_color: Some(border),
        ..SamplerDesc::default()
    };
    d.new_sampler_state(&desc).unwrap().unwrap()
}

impl Shape {
    /// The sample that stores `code` in its top bits; of a packed shape, the code.
    pub(crate) fn sample(&self, code: u16) -> u16 {
        if self.packed {
            return code;
        }
        let shift = u32::try_from(self.container * 8).unwrap() - self.bits;
        code << shift
    }

    /// Writes `luma(x, y)` into each sample of `surface`'s plane 0 and `chroma(x, y)` (Cb, Cr)
    /// into each of plane 1's, whole samples, low bits included; of a packed shape, each the
    /// low 10 bits into its code's place in its word.
    pub(crate) fn fill(
        &self,
        surface: &OsSurface,
        size: (u64, u64),
        luma: impl Fn(u64, u64) -> u16,
        chroma: impl Fn(u64, u64) -> (u16, u16),
    ) {
        use reims_vgpu_core::platform::os::surface::Surface as _;
        let (y_plane, c_plane) = (
            crate::os::plane(surface, 0).unwrap(),
            crate::os::plane(surface, 1).unwrap(),
        );
        let base = surface.base_address().as_ptr::<u8>();
        let element = self.container;
        if self.packed {
            return self.fill_packed(surface, size, luma, chroma);
        }
        let write = |at: u64, v: u16| {
            let at = usize::try_from(at).unwrap();
            // SAFETY: `at` lies inside the surface's allocation (a plane's offset, a row inside
            // its height and an element inside its width), which no GPU work uses yet.
            unsafe {
                if element == 1 {
                    base.add(at).write(u8::try_from(v & 0xff).unwrap());
                } else {
                    base.add(at).cast::<u16>().write_unaligned(v);
                }
            }
        };
        let (width, height) = size;
        for y in 0..height {
            for x in 0..width {
                write(
                    y_plane.offset + y * y_plane.bytes_per_row + x * element,
                    luma(x, y),
                );
            }
        }
        let (chroma_width, chroma_height) = self.chroma_size(size);
        for y in 0..chroma_height {
            for x in 0..chroma_width {
                let (cb, cr) = chroma(x, y);
                let at = c_plane.offset + y * c_plane.bytes_per_row + x * element * 2;
                write(at, cb);
                write(at + element, cr);
            }
        }
    }

    /// [`Shape::fill`] of a packed shape: code `i` of a row in its word `i / 3`, at bits
    /// `10 (i mod 3)`.
    fn fill_packed(
        &self,
        surface: &OsSurface,
        (width, height): (u64, u64),
        luma: impl Fn(u64, u64) -> u16,
        chroma: impl Fn(u64, u64) -> (u16, u16),
    ) {
        use reims_vgpu_core::platform::os::surface::Surface as _;
        let base = surface.base_address().as_ptr::<u8>();
        let write = |plane: &metal2vulkan_runtime::IoSurfacePlane, y: u64, i: u64, code: u16| {
            let at = plane.offset + y * plane.bytes_per_row + i / 3 * 4;
            let shift = 10 * (i % 3);
            let at = usize::try_from(at).unwrap();
            // SAFETY: `at` lies inside the surface's allocation (a plane's offset, a row inside
            // its height and a word inside its codes), which no GPU work uses yet.
            unsafe {
                let word = base.add(at).cast::<u32>();
                let kept = word.read_unaligned().to_le() & !(0x3ff << shift);
                let code = u32::from(code & 0x3ff) << shift;
                word.write_unaligned((kept | code).to_le());
            }
        };
        let (y_plane, c_plane) = (
            crate::os::plane(surface, 0).unwrap(),
            crate::os::plane(surface, 1).unwrap(),
        );
        for y in 0..height {
            for x in 0..width {
                write(&y_plane, y, x, luma(x, y));
            }
        }
        let (chroma_width, chroma_height) = self.chroma_size((width, height));
        for y in 0..chroma_height {
            for x in 0..chroma_width {
                let (cb, cr) = chroma(x, y);
                write(&c_plane, y, 2 * x, cb);
                write(&c_plane, y, 2 * x + 1, cr);
            }
        }
    }

    /// Fills `surface` with codes: `luma(x, y)` and `chroma(x, y)` (Cb, Cr), each stored in its
    /// sample's top bits with the low bits 0.
    pub(crate) fn fill_codes(
        &self,
        surface: &OsSurface,
        size: (u64, u64),
        luma: impl Fn(u64, u64) -> u16,
        chroma: impl Fn(u64, u64) -> (u16, u16),
    ) {
        self.fill(
            surface,
            size,
            |x, y| self.sample(luma(x, y)),
            |x, y| {
                let (cb, cr) = chroma(x, y);
                (self.sample(cb), self.sample(cr))
            },
        );
    }

    /// Fills `surface` with `noise(seed)`'s samples, the low bits too.
    pub(crate) fn fill_noise(&self, surface: &OsSurface, size: (u64, u64), seed: u64) {
        let mut next = noise(seed);
        let (width, height) = size;
        let (chroma_width, chroma_height) = self.chroma_size(size);
        let luma: Vec<u16> = (0..width * height).map(|_| next()).collect();
        let chroma: Vec<(u16, u16)> = (0..chroma_width * chroma_height)
            .map(|_| (next(), next()))
            .collect();
        let at = |x: u64, y: u64, w: u64| usize::try_from(y * w + x).unwrap();
        self.fill(
            surface,
            size,
            |x, y| luma[at(x, y, width)],
            |x, y| chroma[at(x, y, chroma_width)],
        );
    }
}
