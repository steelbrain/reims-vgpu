//! A render blob's SPI keys drawn on both platforms: the values the runtime holds draw as Metal's
//! own deserializer has Metal draw them, and so do the values that decode to nothing, which draw
//! as Metal draws without them; the values that die draw otherwise on Metal (the module docs,
//! [`super::render_spi`]). A compute blob's key the runtime holds runs as Metal's.

use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder as _};
use reims_vgpu_core::platform::command::{CommandBuffer, CommandQueue as _};
use reims_vgpu_core::platform::compute::{ComputeCmd, ComputeEncoder as _};
use reims_vgpu_core::platform::descriptor::RenderPassDescriptor as _;
use reims_vgpu_core::platform::device::GpuDevice;
use reims_vgpu_core::platform::render::{RenderCmd, RenderEncoder as _};
use reims_vgpu_core::platform::resource::{Buffer as _, Library as _};
use reims_vgpu_core::platform::types::{ClearColor, MtlOrigin, MtlSize, NsRange};
use reims_vgpu_core::platform::values::{
    ColorAttachmentSet, DepthAttachmentSet, DepthStencilDesc, PassSet, SamplerDesc,
    StencilAttachmentSet, StencilDesc, TextureDesc,
};
use reims_vgpu_core::platform::{Handle as _, Platform};
use reims_vgpu_core::raise::OnFault;
use reims_vgpu_core::raise::death::DeathKind;
use reims_vgpu_core::rt::Hold;
use reims_vgpu_core::tasks::launcher::{HostSide, ServiceLink, TaskLauncher};
use reims_vgpu_core::testkit::capture::catch_death;
use reims_vgpu_metal::MetalPlatform;
use std::sync::Arc;

use super::RENDER_SEL;
use super::tests::rounded_render_of;
use crate::gpu::device::tests::dev;
use crate::platform::VulkanPlatform;
use crate::platform::tests::platform;

/// The library of `spi_keys.metal`.
const SPI_KEYS: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/spi_keys.metallib"));

/// The target's side, in pixels.
const SIDE: u64 = 8;

/// `MTLPixelFormatRGBA8Unorm`, `RGBA8Unorm_sRGB`, `RGBA16Float`, `Depth32Float_Stencil8`.
const RGBA8: u32 = 70;
const RGBA8_SRGB: u32 = 71;
const RGBA16F: u32 = 115;
const DEPTH_STENCIL: u32 = 260;

/// `MTLTextureUsageShaderRead | RenderTarget`, `MTLTextureType2DMultisample`, the private
/// storage mode's resource options, `MTLLoadActionClear`, `MTLStoreActionStore` and
/// `MultisampleResolve`, `MTLVisibilityResultModeCounting`.
const USAGE: u64 = 5;
const MULTISAMPLE: u64 = 4;
const PRIVATE: u64 = 0x20;
const CLEAR: u64 = 2;
const STORE: u64 = 1;
const RESOLVE: u64 = 2;
const COUNTING: u64 = 2;

/// `MTLCompareFunctionLess`, `Equal`, `Always`, `MTLStencilOperationKeep`, `Replace`.
const LESS: u64 = 1;
const EQUAL: u64 = 2;
const ALWAYS: u64 = 7;
const KEEP: u64 = 0;
const REPLACE: u64 = 2;

/// A launcher no test launches through.
struct NoLauncher;

impl TaskLauncher<MetalPlatform> for NoLauncher {
    fn launch(
        &self,
        _platform: &Arc<MetalPlatform>,
        _host: Arc<dyn HostSide<MetalPlatform>>,
    ) -> Arc<dyn ServiceLink<MetalPlatform>> {
        panic!("no task is launched in these tests")
    }
}

/// The state of a render blob of the vertex function `vertex` (ref 1) and the fragment function
/// `fragment` (ref 2) of `spi_keys.metal`, its 4-byte `entries` and colour attachment elements
/// `colours`, as core decodes a guest's: the death of the decode, where it dies.
fn state<P: Platform>(
    d: &P::Device,
    (vertex, fragment): (&str, &str),
    entries: &[&str],
    colours: &[&[(u8, u32)]],
) -> Result<P::RenderPipeline, reims_vgpu_core::raise::death::DeathSite> {
    let library = d.new_library(SPI_KEYS).unwrap().unwrap().unwrap();
    let functions =
        [vertex, fragment].map(|name| library.new_function(OnFault::Die, name).unwrap().unwrap());
    let mut all = vec!["02 04 02000000"];
    all.extend_from_slice(entries);
    let blob = rounded_render_of(&all, colours, 0);
    let mut resolve = |_: Hold<'_>, r: u64| {
        let at = usize::try_from(r - 1).unwrap();
        Ok(Some(functions[at].clone_ref()))
    };
    let desc = catch_death(|| {
        d.deserialize_render_pipeline_descriptor(Hold::NONE, &blob, &mut resolve)
            .unwrap()
            .unwrap()
    })?;
    Ok(d.new_render_pipeline_state_from_descriptor(Some(&desc), 0)
        .unwrap()
        .unwrap()
        .unwrap())
}

/// A buffer holding `contents`.
fn buffer<P: Platform>(d: &P::Device, length: u64, contents: &[u8]) -> P::Buffer {
    let b = d
        .new_buffer_of_length(OnFault::Die, length, 0)
        .unwrap()
        .unwrap();
    // SAFETY: a Shared buffer of at least `contents.len()` bytes, which no command uses yet.
    unsafe {
        std::ptr::copy_nonoverlapping(
            contents.as_ptr(),
            b.contents().as_ptr::<u8>(),
            contents.len(),
        )
    };
    b
}

/// A `SIDE` by `SIDE` texture of `format`, of `samples` samples (private, where more than one).
fn target<P: Platform>(p: &P, d: &P::Device, format: u32, samples: u64) -> P::Texture {
    let desc = p.texture_descriptor(&TextureDesc {
        texture_type: if samples > 1 { MULTISAMPLE } else { 2 },
        pixel_format: u64::from(format),
        width: SIDE,
        height: SIDE,
        sample_count: samples,
        usage: USAGE,
        resource_options: (samples > 1 || format == DEPTH_STENCIL).then_some(PRIVATE),
        ..TextureDesc::default()
    });
    d.new_texture(OnFault::Die, &desc).unwrap().unwrap()
}

/// The bytes of `t`, `bytes` a texel, read back by a blit into a Shared buffer, after `encode`
/// (the render pass) on the same command buffer.
fn read_back<P: Platform>(
    d: &P::Device,
    cb: &P::CommandBuffer,
    t: &P::Texture,
    bytes: u64,
) -> P::Buffer {
    let length = SIDE * SIDE * bytes;
    let out = buffer::<P>(d, length, &[]);
    let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
    for cmd in [
        BlitCmd::CopyTextureToBuffer {
            source_texture: t,
            source_slice: 0,
            source_level: 0,
            source_origin: MtlOrigin::new(0, 0, 0),
            source_size: MtlSize::new(SIDE, SIDE, 1),
            destination_buffer: &out,
            destination_offset: 0,
            destination_bytes_per_row: SIDE * bytes,
            destination_bytes_per_image: length,
        },
        BlitCmd::EndEncoding,
    ] {
        blit.encode(OnFault::Die, cmd).unwrap();
    }
    out
}

/// The bytes of `b`'s first `n`, read after its command buffer completed.
fn bytes_of<P: Platform>(b: &P::Buffer, n: u64) -> Vec<u8> {
    let n = usize::try_from(n).unwrap();
    // SAFETY: a Shared buffer's bytes, read after the command buffer that wrote them completed.
    unsafe { std::slice::from_raw_parts(b.contents().as_ptr::<u8>(), n) }.to_vec()
}

/// What a drawing draws beside its state and buffers: `count` vertices of primitive type
/// `primitive`, and, at index 0, a texture and a sampler of the vertex stage and of the
/// fragment stage, where given.
struct Binds<'a, P: Platform> {
    primitive: u64,
    count: u64,
    vertex: Option<(&'a P::Texture, &'a P::SamplerState)>,
    fragment: Option<(&'a P::Texture, &'a P::SamplerState)>,
}

impl<P: Platform> Binds<'_, P> {
    /// One triangle, and nothing more.
    fn triangle() -> Self {
        Binds {
            primitive: 3,
            count: 3,
            vertex: None,
            fragment: None,
        }
    }
}

/// A drawing of one case: the target's colour attachment 0 of `format`, of `samples` samples,
/// cleared to opaque blue, under `state`, one triangle over it of colour `(1, 0.5, 0.25, 0.5)`,
/// read back (resolved, where more than one sample).
fn drawn<P: Platform>(
    p: &P,
    d: &P::Device,
    state: &P::RenderPipeline,
    (format, samples): (u32, u64),
) -> Vec<u8> {
    drawn_with(p, d, state, (format, samples), &Binds::triangle())
}

/// [`drawn`], drawing what `binds` says.
fn drawn_with<P: Platform>(
    p: &P,
    d: &P::Device,
    state: &P::RenderPipeline,
    (format, samples): (u32, u64),
    binds: &Binds<'_, P>,
) -> Vec<u8> {
    let bytes = if format == RGBA16F { 8 } else { 4 };
    let resolved = target(p, d, format, 1);
    let sampled = (samples > 1).then(|| target(p, d, format, samples));
    let pass = p.render_pass_descriptor();
    pass.set_color_attachment(
        OnFault::Die,
        0,
        &ColorAttachmentSet {
            texture: Some(Some(sampled.as_ref().unwrap_or(&resolved))),
            resolve_texture: sampled.as_ref().map(|_| Some(&resolved)),
            load_action: Some(CLEAR),
            store_action: Some(if sampled.is_some() { RESOLVE } else { STORE }),
            clear_color: Some(ClearColor {
                red: 0.0,
                green: 0.0,
                blue: 1.0,
                alpha: 1.0,
            }),
            ..ColorAttachmentSet::default()
        },
    )
    .unwrap();
    let z = buffer::<P>(d, 4, &0.0f32.to_ne_bytes());
    let colour: Vec<u8> = [1.0f32, 0.5, 0.25, 0.5]
        .iter()
        .flat_map(|f| f.to_ne_bytes())
        .collect();
    let colour = buffer::<P>(d, 16, &colour);
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
    let one = NsRange::new(0, 1);
    let mut cmds = vec![
        RenderCmd::SetRenderPipelineState {
            pipeline_state: Some(state),
        },
        RenderCmd::SetVertexBuffer {
            buffer: Some(&z),
            offset: 0,
            index: 0,
        },
        RenderCmd::SetFragmentBuffer {
            buffer: Some(&colour),
            offset: 0,
            index: 0,
        },
    ];
    let (vertex, fragment) = (
        binds.vertex.map(|(t, s)| ([Some(t)], [Some(s)])),
        binds.fragment.map(|(t, s)| ([Some(t)], [Some(s)])),
    );
    if let Some((textures, samplers)) = &vertex {
        cmds.push(RenderCmd::SetVertexTextures {
            textures,
            range: one,
        });
        cmds.push(RenderCmd::SetVertexSamplerStates {
            samplers,
            range: one,
        });
    }
    if let Some((textures, samplers)) = &fragment {
        cmds.push(RenderCmd::SetFragmentTextures {
            textures,
            range: one,
        });
        cmds.push(RenderCmd::SetFragmentSamplerStates {
            samplers,
            range: one,
        });
    }
    cmds.push(RenderCmd::DrawPrimitives {
        primitive_type: binds.primitive,
        vertex_start: 0,
        vertex_count: binds.count,
    });
    cmds.push(RenderCmd::EndEncoding);
    for cmd in cmds {
        e.encode(OnFault::Die, cmd).unwrap();
    }
    let out = read_back::<P>(d, &cb, &resolved, bytes);
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    assert!(cb.error().is_none(), "{:?}", cb.error());
    bytes_of::<P>(&out, SIDE * SIDE * bytes)
}

/// One case: a blob's vertex function, its SPI entries beside the fragment ref, its colour
/// attachment's format, its samples (key 4 too, where more than one), and its colour element's
/// further entries.
type Case<'a> = (&'a str, &'a [&'a str], u32, u64, &'a [(u8, u32)]);

/// The drawing of `case` on `p`, or the death of its decode.
fn case_on<P: Platform>(
    p: &P,
    d: &P::Device,
    (vertex, entries, format, samples, colour): Case<'_>,
) -> Result<Vec<u8>, reims_vgpu_core::raise::death::DeathSite> {
    let count = format!("04 04 {:02x}000000", samples);
    let mut all: Vec<&str> = entries.to_vec();
    if samples > 1 {
        all.push(&count);
    }
    let mut element = vec![(0, 0), (1, format)];
    element.extend_from_slice(colour);
    let state = state::<P>(d, (vertex, "paint"), &all, &[&element])?;
    Ok(drawn(p, d, &state, (format, samples)))
}

/// `colour` element entries that blend by the source's alpha: `blendingEnabled`,
/// `sourceRGBBlendFactor` `SourceAlpha`, `destinationRGBBlendFactor` `OneMinusSourceAlpha`.
const BLENDS: &[(u8, u32)] = &[(2, 1), (3, 4), (4, 5)];

/// A write mask of red and alpha.
const RED_ALPHA: &[(u8, u32)] = &[(9, 9)];

/// Each SPI value that decodes to nothing, or that the runtime holds, draws as Metal draws it,
/// byte for byte, and as Metal draws without it: `sampleMask` with a bit for each sample
/// (OpenGL's `0xffffffff` among them), at 1 and 4 samples; `logicOperation` off (OpenGL's 2 among them), and 2 on where nothing
/// blends, at a partial write mask, into unorm, sRGB and half-float attachments; the alpha
/// test's function off, and Always on; `clipDistanceEnableMask` at every mask under no, one and
/// two clip distances; `depthStencilWriteDisabled` without a depth attachment; and each
/// default written. Each value that dies draws otherwise on Metal than without it
#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "MoltenVK: the runtime records device-denormals, a death in strict mode; Mac-only test, off everywhere; ledger: MoltenVK strict deaths"
)]
fn spi_values_that_decode_draw_as_metals() {
    let Some(md) = MetalPlatform::system_default_device() else {
        return;
    };
    let mp = MetalPlatform::new(Box::new(NoLauncher));
    let (p, d) = (platform(), dev());
    let none: &[(u8, u32)] = &[];
    let mut decoded: Vec<Case<'_>> = vec![
        ("cover", &["1c 04 ffffffff"], RGBA8, 1, none),
        ("cover", &["1c 04 01000000"], RGBA8, 1, none),
        ("cover", &["1c 04 0f000000"], RGBA8, 4, none),
        ("cover", &["1c 04 ffffffff"], RGBA8, 4, none),
        ("cover", &["1b 04 0000803f"], RGBA8, 4, none),
        ("cover", &["3a 04 ffffffff"], RGBA8, 1, none),
        ("cover", &["38 04 01000000"], RGBA8, 1, none),
        ("cover", &["2e 04 04000000"], RGBA8, 1, none),
        (
            "cover",
            &["2e 04 07000000", "2d 04 01000000"],
            RGBA8,
            1,
            none,
        ),
        ("cover", &["37 04 0a000000"], RGBA8, 1, BLENDS),
    ];
    for format in [RGBA8, RGBA8_SRGB, RGBA16F] {
        decoded.push(("cover", &["37 04 02000000"], format, 1, none));
        decoded.push((
            "cover",
            &["2f 04 01000000", "37 04 02000000"],
            format,
            1,
            none,
        ));
        decoded.push((
            "cover",
            &["37 04 02000000", "2f 04 01000000"],
            format,
            1,
            RED_ALPHA,
        ));
    }
    const MASKS: [&[&str]; 4] = [
        &["30 04 00000000"],
        &["30 04 01000000"],
        &["30 04 02000000"],
        &["30 04 ff000000"],
    ];
    for vertex in ["cover", "cover_clip1", "cover_clip2"] {
        for mask in MASKS {
            decoded.push((vertex, mask, RGBA8, 1, none));
        }
    }
    for case @ (vertex, _, format, samples, colour) in decoded {
        let ours = case_on(&p, &d, case).unwrap();
        let theirs = case_on(&mp, &md, case).unwrap();
        assert!(ours == theirs, "{case:?}");
        let without = case_on(&mp, &md, (vertex, &[], format, samples, colour)).unwrap();
        assert!(theirs == without, "{case:?}");
    }
    let dying: [Case<'_>; 3] = [
        ("cover", &["2d 04 01000000"], RGBA8, 1, none),
        (
            "cover",
            &["2f 04 01000000", "37 04 0a000000"],
            RGBA8,
            1,
            none,
        ),
        (
            "cover",
            &["2f 04 01000000", "37 04 02000000"],
            RGBA8,
            1,
            BLENDS,
        ),
    ];
    let smooth: Case<'_> = ("cover", &["31 04 01000000"], RGBA8, 1, none);
    for case @ (vertex, _, format, samples, colour) in dying.into_iter().chain([smooth]) {
        let site = case_on(&p, &d, case).unwrap_err();
        assert_eq!(
            (site.kind, site.file),
            (DeathKind::Abort, RENDER_SEL),
            "{case:?}"
        );
        let theirs = case_on(&mp, &md, case).unwrap();
        let without = case_on(&mp, &md, (vertex, &[], format, samples, colour)).unwrap();
        assert!(theirs != without, "{case:?}");
    }
}

/// A Private `SIDE` by `SIDE` texture of `format` holding `texels` (`bytes` a texel, row by row),
/// written by a blit from a buffer.
fn holding<P: Platform>(
    p: &P,
    d: &P::Device,
    (format, bytes): (u32, u64),
    texels: &[u8],
) -> P::Texture {
    let t = d
        .new_texture(
            OnFault::Die,
            &p.texture_descriptor(&TextureDesc {
                texture_type: 2,
                pixel_format: u64::from(format),
                width: SIDE,
                height: SIDE,
                sample_count: 1,
                usage: USAGE,
                resource_options: Some(PRIVATE),
                ..TextureDesc::default()
            }),
        )
        .unwrap()
        .unwrap();
    let source = buffer::<P>(d, SIDE * SIDE * bytes, texels);
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
    for cmd in [
        BlitCmd::CopyBufferToTexture {
            source_buffer: &source,
            source_offset: 0,
            source_bytes_per_row: SIDE * bytes,
            source_bytes_per_image: SIDE * SIDE * bytes,
            source_size: MtlSize::new(SIDE, SIDE, 1),
            destination_texture: &t,
            destination_slice: 0,
            destination_level: 0,
            destination_origin: MtlOrigin::new(0, 0, 0),
        },
        BlitCmd::EndEncoding,
    ] {
        blit.encode(OnFault::Die, cmd).unwrap();
    }
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    assert!(cb.error().is_none(), "{:?}", cb.error());
    t
}

/// A sampler of nearest filtering, normalized coordinates and `desc`'s address modes, border
/// colour and compare function.
fn sampler<P: Platform>(d: &P::Device, desc: SamplerDesc) -> P::SamplerState {
    d.new_sampler_state(&SamplerDesc {
        min_filter: Some(0),
        mag_filter: Some(0),
        mip_filter: Some(0),
        normalized_coordinates: Some(true),
        ..desc
    })
    .unwrap()
    .unwrap()
}

/// `MTLSamplerAddressModeClampToBorderColor`, and the raw border colours the tests sample past
/// the edge under: transparent black, opaque black, opaque white, and 3 and 7, which no
/// enumerator is and Metal takes.
const CLAMP_TO_BORDER: u64 = 5;
const BORDERS: [u64; 5] = [0, 1, 2, 3, 7];

/// A texture of 8-bit texels each a different colour, for the samples past its edge.
fn patterned<P: Platform>(p: &P, d: &P::Device) -> P::Texture {
    let texels: Vec<u8> = (0..SIDE * SIDE)
        .flat_map(|i| {
            let i = u8::try_from(i).unwrap();
            [i.wrapping_mul(37), 255 - i, i.wrapping_mul(11), 200]
        })
        .collect();
    holding(p, d, (RGBA8, 4), &texels)
}

/// What `border_kernel` writes under a compute blob of its function and the 4-byte `entries`,
/// sampling `t` through a sampler of border colour `border`, on `p`'s device `d`: one `float4` a
/// thread of an 8 by 8 grid, as bytes.
fn border_dispatched<P: Platform>(
    d: &P::Device,
    t: &P::Texture,
    border: u64,
    entries: &[&str],
) -> Vec<u8> {
    let library = d.new_library(SPI_KEYS).unwrap().unwrap().unwrap();
    let function = library
        .new_function(OnFault::Die, "border_kernel")
        .unwrap()
        .unwrap();
    let mut blob = format!("{:02x} 00 04 01000000", entries.len() + 1);
    for entry in entries {
        blob.push(' ');
        blob.push_str(entry);
    }
    let blob = super::tests::hex(&blob);
    let mut resolve = |_: Hold<'_>, r: u64| {
        assert_eq!(r, 1);
        Ok(Some(function.clone_ref()))
    };
    let desc = d
        .deserialize_compute_pipeline_descriptor(Hold::NONE, &blob, &mut resolve)
        .unwrap()
        .unwrap();
    let state = d
        .new_compute_pipeline_state(Some(&desc), 0)
        .unwrap()
        .unwrap()
        .unwrap();
    let s = sampler::<P>(
        d,
        SamplerDesc {
            s_address_mode: Some(CLAMP_TO_BORDER),
            t_address_mode: Some(CLAMP_TO_BORDER),
            border_color: Some(border),
            ..SamplerDesc::default()
        },
    );
    let length = SIDE * SIDE * 16;
    let out = buffer::<P>(d, length, &[]);
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let e = cb.compute_encoder(OnFault::Die).unwrap().unwrap();
    for cmd in [
        ComputeCmd::SetComputePipelineState {
            state: Some(&state),
        },
        ComputeCmd::SetTexture {
            texture: Some(t),
            index: 0,
        },
        ComputeCmd::SetSamplerStates {
            samplers: &[Some(&s)],
            range: NsRange::new(0, 1),
        },
        ComputeCmd::SetBuffers {
            buffers: &[Some(&out)],
            offsets: &[0],
            range: NsRange::new(0, 1),
        },
        ComputeCmd::DispatchThreads {
            threads_per_grid: MtlSize::new(SIDE, SIDE, 1),
            threads_per_threadgroup: MtlSize::new(SIDE, SIDE, 1),
        },
        ComputeCmd::EndEncoding,
    ] {
        e.encode(OnFault::Die, cmd).unwrap();
    }
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    assert!(cb.error().is_none(), "{:?}", cb.error());
    bytes_of::<P>(&out, length)
}

/// Each SPI value the runtime holds draws as Metal draws it, byte for byte, on this host (an
/// Apple M3 Max), where the runtime's measurement of it is an M4's (the module docs):
/// - `sampleMask` without a bit for each sample: 0 and 2 at 1 sample, and 0, 5 and `0xa` at 4,
///   each of which Metal draws otherwise than without it;
/// - `sampleCoverage` below 1: 0.5 and 0.25 at 1 sample, 0.5 at 2, and 0.25, 0.5 and 0.75 at 4,
///   0.5 beside the sample mask 5 and `0xa`, each of which Metal draws otherwise than without
///   it;
/// - `pointCoordLowerLeft`, `pointSizeOutputVS`, `twoSideEnabled`, `vertexDepthCompareClampMask`
///   and `needsCustomBorderColorSamplers` (render 57), each at 1 and at `0xffffffff`: over a
///   triangle; over points of a size the vertex function writes, coloured by their
///   `[[point_coord]]`; over triangles of both windings, coloured by `[[front_facing]]`; over a
///   vertex function's comparisons of references outside 0 to 1 against a depth of 0 and of 1,
///   under `Less` and `Greater`; and over samples past a texture's edge in a fragment function,
///   under each of [`BORDERS`]. Metal draws each as without it;
/// - a compute blob's `needsCustomBorderColorSamplers` (12) at 1, over a kernel's samples past the
///   edge under each of [`BORDERS`], which Metal runs as without it
#[test]
fn spi_values_the_runtime_holds_draw_as_metals() {
    // What `p` draws of `functions` under the 4-byte `entries`, with `binds`.
    fn held<P: Platform>(
        p: &P,
        d: &P::Device,
        functions: (&str, &str),
        entries: &[&str],
        binds: &Binds<'_, P>,
    ) -> Vec<u8> {
        let element: &[(u8, u32)] = &[(0, 0), (1, RGBA8)];
        let state = state::<P>(d, functions, entries, &[element]).unwrap();
        drawn_with(p, d, &state, (RGBA8, 1), binds)
    }
    let Some(md) = MetalPlatform::system_default_device() else {
        return;
    };
    let mp = MetalPlatform::new(Box::new(NoLauncher));
    let (p, d) = (platform(), dev());
    let mut differ = Vec::new();
    // Adds to `differ` where `ours`, of the case `of`, is not `theirs`, or, where `unchanged`,
    // `theirs` is not `without`.
    let mut compare = |of: String, ours: Result<Vec<u8>, String>, theirs, without, unchanged| {
        match ours {
            Err(line) => differ.push(format!("{of}: died: {line}")),
            Ok(ours) if ours != theirs => differ.push(format!("{of}: ours differ from Metal's")),
            Ok(_) => {}
        }
        if unchanged && theirs != without {
            differ.push(format!("{of}: Metal's differ from its own without the key"));
        }
        if !unchanged && theirs == without {
            differ.push(format!("{of}: Metal's are its own without the mask"));
        }
    };
    let survived = |f: &dyn Fn() -> Vec<u8>| {
        *crate::answer::LAST_LINE.lock() = None;
        catch_death(f).map_err(|site| {
            let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
            format!("{site:?} {line}")
        })
    };
    let none: &[(u8, u32)] = &[];
    let masks: [Case<'_>; 13] = [
        ("cover", &["1c 04 00000000"], RGBA8, 1, none),
        ("cover", &["1c 04 02000000"], RGBA8, 1, none),
        ("cover", &["1c 04 00000000"], RGBA8, 4, none),
        ("cover", &["1c 04 05000000"], RGBA8, 4, none),
        ("cover", &["1c 04 0a000000"], RGBA8, 4, none),
        ("cover", &["1b 04 0000003f"], RGBA8, 1, none),
        ("cover", &["1b 04 0000803e"], RGBA8, 1, none),
        ("cover", &["1b 04 0000003f"], RGBA8, 2, none),
        ("cover", &["1b 04 0000803e"], RGBA8, 4, none),
        ("cover", &["1b 04 0000003f"], RGBA8, 4, none),
        ("cover", &["1b 04 0000403f"], RGBA8, 4, none),
        (
            "cover",
            &["1b 04 0000003f", "1c 04 05000000"],
            RGBA8,
            4,
            none,
        ),
        (
            "cover",
            &["1b 04 0000003f", "1c 04 0a000000"],
            RGBA8,
            4,
            none,
        ),
    ];
    for case @ (vertex, _, format, samples, colour) in masks {
        let ours = survived(&|| case_on(&p, &d, case).unwrap());
        let theirs = case_on(&mp, &md, case).unwrap();
        let without = case_on(&mp, &md, (vertex, &[], format, samples, colour)).unwrap();
        compare(format!("{case:?}"), ours, theirs, without, false);
    }
    let (ot, ts) = (patterned(&p, &d), patterned(&mp, &md));
    let mut cases = 0;
    for key in [0x32u8, 0x33, 0x34, 0x35, 0x39] {
        for value in ["01000000", "ffffffff"] {
            let entry = format!("{key:02x} 04 {value}");
            let entries = [entry.as_str()];
            // The runtime notes `RestrictedDepthRange` at a vertex function's depth comparison
            // on this host's MoltenVK, under the key or not, which strict mode dies of: that
            // draw is compared as the process runs by default.
            let mut both =
                |what: &str,
                 functions: (&str, &str),
                 binds: (&Binds<'_, VulkanPlatform>, &Binds<'_, MetalPlatform>)| {
                    let (ob, tb) = binds;
                    let ours = survived(&|| {
                        let draw = || held(&p, &d, functions, &entries, ob);
                        match functions.0 {
                            "compare_vertex" => crate::answer::relaxed(draw),
                            _ => draw(),
                        }
                    });
                    let theirs = held(&mp, &md, functions, &entries, tb);
                    let without = held(&mp, &md, functions, &[], tb);
                    compare(format!("{entry} {what}"), ours, theirs, without, true);
                    cases += 1;
                };
            // A triangle; points coloured by their coordinate; and triangles of both windings,
            // coloured by their facing.
            for (functions, primitive, count) in [
                (("cover", "paint"), 3, 3),
                (("points", "paint_coord"), 0, 4),
                (("facing", "paint_facing"), 3, 6),
            ] {
                let binds = (
                    &Binds {
                        primitive,
                        count,
                        ..Binds::triangle()
                    },
                    &Binds {
                        primitive,
                        count,
                        ..Binds::triangle()
                    },
                );
                both(functions.1, functions, binds);
            }
            // A vertex function's comparisons against a depth of 0 and of 1.
            for depth in [0.0f32, 1.0] {
                let texels: Vec<u8> = (0..SIDE * SIDE).flat_map(|_| depth.to_ne_bytes()).collect();
                let (od, td) = (
                    holding(&p, &d, (DEPTH32, 4), &texels),
                    holding(&mp, &md, (DEPTH32, 4), &texels),
                );
                for compare_function in [1, 4] {
                    let desc = SamplerDesc {
                        compare_function: Some(compare_function),
                        ..SamplerDesc::default()
                    };
                    let (os, tss) = (
                        sampler::<VulkanPlatform>(&d, desc),
                        sampler::<MetalPlatform>(&md, desc),
                    );
                    let binds = (
                        &Binds {
                            vertex: Some((&od, &os)),
                            ..Binds::triangle()
                        },
                        &Binds {
                            vertex: Some((&td, &tss)),
                            ..Binds::triangle()
                        },
                    );
                    both(
                        &format!("comparisons against {depth} by {compare_function}"),
                        ("compare_vertex", "paint_shaded"),
                        binds,
                    );
                }
            }
            // Samples past a texture's edge.
            for border in BORDERS {
                let desc = SamplerDesc {
                    s_address_mode: Some(CLAMP_TO_BORDER),
                    t_address_mode: Some(CLAMP_TO_BORDER),
                    border_color: Some(border),
                    ..SamplerDesc::default()
                };
                let (os, tss) = (
                    sampler::<VulkanPlatform>(&d, desc),
                    sampler::<MetalPlatform>(&md, desc),
                );
                let binds = (
                    &Binds {
                        fragment: Some((&ot, &os)),
                        ..Binds::triangle()
                    },
                    &Binds {
                        fragment: Some((&ts, &tss)),
                        ..Binds::triangle()
                    },
                );
                both(
                    &format!("samples past the edge, border colour {border}"),
                    ("cover", "paint_border"),
                    binds,
                );
            }
        }
    }
    assert_eq!(cases, 5 * 2 * (3 + 4 + BORDERS.len()));
    for border in BORDERS {
        let set = ["0c 04 01000000"];
        let ours = survived(&|| border_dispatched::<VulkanPlatform>(&d, &ot, border, &set));
        let theirs = border_dispatched::<MetalPlatform>(&md, &ts, border, &set);
        let without = border_dispatched::<MetalPlatform>(&md, &ts, border, &[]);
        compare(
            format!("kernel, border colour {border}"),
            ours,
            theirs,
            without,
            true,
        );
    }
    assert!(differ.is_empty(), "{} differ: {differ:#?}", differ.len());
}

/// What a pass of a depth-stencil attachment draws under a state of `entries`: its colour, then
/// the visibility count of its first draw. The attachment is cleared to depth 0.5 and stencil
/// 0. The first draw writes the right half (its left half discarded) at depth 0.25, depth
/// `Less` and written, stencil replaced by 7; the second, everywhere at 0.375, depth `Less`
/// unwritten, so it draws where the first wrote no depth; the third, everywhere at 0, depth
/// `Always`, stencil `Equal` 7, so it draws where the first wrote the stencil.
fn depth_drawn<P: Platform>(p: &P, d: &P::Device, entries: &[&str]) -> (Vec<u8>, u64) {
    let mut all = vec!["09 04 04010000", "0a 04 04010000"];
    all.extend_from_slice(entries);
    let element: &[(u8, u32)] = &[(0, 0), (1, RGBA8)];
    let states = ["paint_right", "paint"]
        .map(|fragment| state::<P>(d, ("cover", fragment), &all, &[element]).unwrap());
    let colour = target(p, d, RGBA8, 1);
    let depth = target(p, d, DEPTH_STENCIL, 1);
    let pass = p.render_pass_descriptor();
    pass.set_color_attachment(
        OnFault::Die,
        0,
        &ColorAttachmentSet {
            texture: Some(Some(&colour)),
            load_action: Some(CLEAR),
            store_action: Some(STORE),
            ..ColorAttachmentSet::default()
        },
    )
    .unwrap();
    pass.set_depth_attachment(&DepthAttachmentSet {
        texture: Some(Some(&depth)),
        load_action: Some(CLEAR),
        store_action: Some(STORE),
        clear_depth: Some(0.5),
        ..DepthAttachmentSet::default()
    });
    pass.set_stencil_attachment(&StencilAttachmentSet {
        texture: Some(Some(&depth)),
        load_action: Some(CLEAR),
        store_action: Some(STORE),
        clear_stencil: Some(0),
        ..StencilAttachmentSet::default()
    });
    let visible = buffer::<P>(d, 8, &[0; 8]);
    pass.set_pass(&PassSet {
        visibility_result_buffer: Some(Some(&visible)),
        ..PassSet::default()
    });
    let stencil = |compare, pass| StencilDesc {
        stencil_compare_function: compare,
        stencil_failure_operation: KEEP,
        depth_failure_operation: KEEP,
        depth_stencil_pass_operation: pass,
        read_mask: 0xff,
        write_mask: 0xff,
    };
    let ds = |compare, write, s: StencilDesc| {
        d.new_depth_stencil_state(&DepthStencilDesc {
            depth_compare_function: Some(compare),
            depth_write_enabled: Some(write),
            front: Some(s),
            back: Some(s),
        })
        .unwrap()
        .unwrap()
    };
    let writes = ds(LESS, true, stencil(ALWAYS, REPLACE));
    let tests_depth = ds(LESS, false, stencil(ALWAYS, KEEP));
    let tests_stencil = ds(ALWAYS, false, stencil(EQUAL, KEEP));
    let float = |f: f32| buffer::<P>(d, 4, &f.to_ne_bytes());
    let zs = [float(0.25), float(0.375), float(0.0)];
    let colours = [
        [1.0f32, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
    ]
    .map(|c| {
        buffer::<P>(
            d,
            16,
            &c.iter().flat_map(|f| f.to_ne_bytes()).collect::<Vec<_>>(),
        )
    });
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
    e.encode(
        OnFault::Die,
        RenderCmd::SetStencilFrontReferenceValue {
            front_reference_value: 7,
            back_reference_value: 7,
        },
    )
    .unwrap();
    let draws = [
        (&states[0], &writes, Some(COUNTING)),
        (&states[1], &tests_depth, Some(0)),
        (&states[1], &tests_stencil, None),
    ];
    for (i, (state, depth_stencil, mode)) in draws.into_iter().enumerate() {
        let mut cmds = vec![
            RenderCmd::SetRenderPipelineState {
                pipeline_state: Some(state),
            },
            RenderCmd::SetDepthStencilState {
                depth_stencil_state: Some(depth_stencil),
            },
            RenderCmd::SetVertexBuffer {
                buffer: Some(&zs[i]),
                offset: 0,
                index: 0,
            },
            RenderCmd::SetFragmentBuffer {
                buffer: Some(&colours[i]),
                offset: 0,
                index: 0,
            },
        ];
        cmds.extend(mode.map(|mode| RenderCmd::SetVisibilityResultMode { mode, offset: 0 }));
        cmds.push(RenderCmd::DrawPrimitives {
            primitive_type: 3,
            vertex_start: 0,
            vertex_count: 3,
        });
        for cmd in cmds {
            e.encode(OnFault::Die, cmd).unwrap();
        }
    }
    e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
    let out = read_back::<P>(d, &cb, &colour, 4);
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    assert!(cb.error().is_none(), "{:?}", cb.error());
    let count = u64::from_ne_bytes(bytes_of::<P>(&visible, 8).try_into().unwrap());
    (bytes_of::<P>(&out, SIDE * SIDE * 4), count)
}

/// `MTLPixelFormatDepth32Float`, and `MTLLoadActionLoad`.
const DEPTH32: u32 = 252;
const LOAD: u64 = 1;

/// The positive and negative denormal depths, smallest and largest, which a depth texture
/// stores and a depth test compares (`depth-denormals`).
const DENORMALS: [u32; 4] = [0x0000_0001, 0x007f_ffff, 0x8000_0001, 0x807f_ffff];

/// What a pass draws over a `Depth32Float` texture whose every texel holds `stored`, written by
/// a blit from a buffer, or by the pass's clear where `cleared`: the colour of one white triangle
/// at depth 0 under depth `compare`, unwritten, over a black clear, then the depth's bytes, each
/// read back by a blit.
fn denormal_depth_drawn<P: Platform>(
    p: &P,
    d: &P::Device,
    stored: f32,
    cleared: bool,
    compare: u64,
) -> (Vec<u8>, Vec<u8>) {
    let element: &[(u8, u32)] = &[(0, 0), (1, RGBA8)];
    let state = state::<P>(d, ("cover", "paint"), &["09 04 fc000000"], &[element]).unwrap();
    let colour = target(p, d, RGBA8, 1);
    let depth = d
        .new_texture(
            OnFault::Die,
            &p.texture_descriptor(&TextureDesc {
                texture_type: 2,
                pixel_format: u64::from(DEPTH32),
                width: SIDE,
                height: SIDE,
                sample_count: 1,
                usage: USAGE,
                resource_options: Some(PRIVATE),
                ..TextureDesc::default()
            }),
        )
        .unwrap()
        .unwrap();
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let texels: Vec<u8> = (0..SIDE * SIDE)
        .flat_map(|_| stored.to_ne_bytes())
        .collect();
    let source = buffer::<P>(d, SIDE * SIDE * 4, &texels);
    if !cleared {
        let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
        for cmd in [
            BlitCmd::CopyBufferToTexture {
                source_buffer: &source,
                source_offset: 0,
                source_bytes_per_row: SIDE * 4,
                source_bytes_per_image: SIDE * SIDE * 4,
                source_size: MtlSize::new(SIDE, SIDE, 1),
                destination_texture: &depth,
                destination_slice: 0,
                destination_level: 0,
                destination_origin: MtlOrigin::new(0, 0, 0),
            },
            BlitCmd::EndEncoding,
        ] {
            blit.encode(OnFault::Die, cmd).unwrap();
        }
    }
    let pass = p.render_pass_descriptor();
    pass.set_color_attachment(
        OnFault::Die,
        0,
        &ColorAttachmentSet {
            texture: Some(Some(&colour)),
            load_action: Some(CLEAR),
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
        texture: Some(Some(&depth)),
        load_action: Some(if cleared { CLEAR } else { LOAD }),
        store_action: Some(STORE),
        clear_depth: Some(f64::from(stored)),
        ..DepthAttachmentSet::default()
    });
    let tests_depth = d
        .new_depth_stencil_state(&DepthStencilDesc {
            depth_compare_function: Some(compare),
            depth_write_enabled: Some(false),
            front: None,
            back: None,
        })
        .unwrap()
        .unwrap();
    let z = buffer::<P>(d, 4, &0.0f32.to_ne_bytes());
    let white = buffer::<P>(
        d,
        16,
        &[1.0f32; 4]
            .iter()
            .flat_map(|f| f.to_ne_bytes())
            .collect::<Vec<_>>(),
    );
    let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
    for cmd in [
        RenderCmd::SetRenderPipelineState {
            pipeline_state: Some(&state),
        },
        RenderCmd::SetDepthStencilState {
            depth_stencil_state: Some(&tests_depth),
        },
        RenderCmd::SetVertexBuffer {
            buffer: Some(&z),
            offset: 0,
            index: 0,
        },
        RenderCmd::SetFragmentBuffer {
            buffer: Some(&white),
            offset: 0,
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
    let drawn = read_back::<P>(d, &cb, &colour, 4);
    let kept = read_back::<P>(d, &cb, &depth, 4);
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    assert!(cb.error().is_none(), "{:?}", cb.error());
    (
        bytes_of::<P>(&drawn, SIDE * SIDE * 4),
        bytes_of::<P>(&kept, SIDE * SIDE * 4),
    )
}

/// A `Depth32Float` texture keeps a denormal depth that a blit from a buffer writes, positive or
/// negative, or that a pass clears it to, and a draw at depth 0 compares against it, under every
/// compare function but `Never` and `Always`, as Metal on this host does (an Apple M3 Max): the
/// drawn colour and the depth's bytes are Metal's. Metal compares a positive denormal above
/// zero, so a draw at 0 under `Less` covers the target, and keeps its bytes. The runtime
/// announces `depth-denormals` at each such blit and clear on this host's MoltenVK, which it has
/// measured on an Apple M4 alone; this is the measurement on this host's GPU
#[test]
fn denormal_depths_store_and_compare_as_metals() {
    let Some(md) = MetalPlatform::system_default_device() else {
        return;
    };
    let mp = MetalPlatform::new(Box::new(NoLauncher));
    let (p, d) = (platform(), dev());
    let white = [255u8; 4].repeat(64);
    let mut cases = 0;
    for bits in DENORMALS {
        let stored = f32::from_bits(bits);
        // A pass clears to a depth within [0, 1] alone: the positive denormals.
        let ways: &[bool] = if bits & 0x8000_0000 == 0 {
            &[false, true]
        } else {
            &[false]
        };
        for &cleared in ways {
            for compare in 1..=6 {
                let theirs = denormal_depth_drawn(&mp, &md, stored, cleared, compare);
                let ours = denormal_depth_drawn(&p, &d, stored, cleared, compare);
                let case = format!("{bits:#010x}, cleared {cleared}, compare {compare}");
                assert_eq!(ours, theirs, "{case}");
                if bits & 0x8000_0000 == 0 && compare == LESS {
                    assert_eq!(theirs.0, white, "{case}: Metal compares it above zero");
                    let kept: Vec<u8> = (0..64).flat_map(|_| bits.to_ne_bytes()).collect();
                    assert_eq!(theirs.1, kept, "{case}: Metal keeps its bytes");
                }
                cases += 1;
            }
        }
    }
    assert_eq!(cases, 36);
}

/// `depthStencilWriteDisabled` decodes to nothing at every value: under 1, as without it, on
/// Metal and here alike, depth and stencil writes land, a discarded fragment writes neither, and
/// the visibility count is the same
#[test]
fn depth_stencil_write_disabled_draws_as_metals() {
    let Some(md) = MetalPlatform::system_default_device() else {
        return;
    };
    let mp = MetalPlatform::new(Box::new(NoLauncher));
    let (p, d) = (platform(), dev());
    let without = depth_drawn(&mp, &md, &[]);
    // The left half drew the second draw's green, the right half the third's white.
    let row: Vec<u8> = (0..SIDE)
        .flat_map(|x| if x < 4 { [0, 255, 0, 255] } else { [255; 4] })
        .collect();
    assert_eq!(without.0, row.repeat(8));
    assert_eq!(without.1, 32);
    for value in ["01000000", "ffffffff"] {
        let entry = format!("38 04 {value}");
        let theirs = depth_drawn(&mp, &md, &[&entry]);
        assert_eq!(theirs, without, "{value}");
        assert_eq!(depth_drawn(&p, &d, &[&entry]), theirs, "{value}");
    }
}
