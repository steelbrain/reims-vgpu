//! A `Depth32Float` texture's denormal depths, stored and compared as Metal does: what the
//! runtime announces as `depth-denormals` on this host's MoltenVK is measured against Metal here.

use crate::core::platform::Platform;
use crate::core::platform::blit::{BlitCmd, BlitEncoder as _};
use crate::core::platform::command::{CommandBuffer as _, CommandQueue as _};
use crate::core::platform::descriptor::RenderPassDescriptor as _;
use crate::core::platform::device::GpuDevice as _;
use crate::core::platform::render::{RenderCmd, RenderEncoder as _};
use crate::core::platform::resource::{Buffer as _, Library as _};
use crate::core::platform::types::{ClearColor, MtlOrigin, MtlSize};
use crate::core::platform::values::{
    ColorAttachmentSet, DepthAttachmentSet, DepthStencilDesc, FullRenderPipelineDesc, TextureDesc,
};
use crate::core::raise::OnFault;

use crate::gpu::device::tests::dev;
use crate::platform::tests::platform;

/// The library of `depth_denormals.metal`.
const LIBRARY: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/depth_denormals.metallib"));

/// The target's side, in pixels.
const SIDE: u64 = 8;

/// `MTLPixelFormatRGBA8Unorm`.
const RGBA8: u32 = 70;

/// `MTLTextureUsageShaderRead | RenderTarget`, the private storage mode's resource options,
/// `MTLLoadActionClear`, `MTLStoreActionStore`.
const USAGE: u64 = 5;
const PRIVATE: u64 = 0x20;
const CLEAR: u64 = 2;
const STORE: u64 = 1;

/// The state of the vertex function `vertex` and the fragment function `fragment` of
/// `depth_denormals.metal`, into `colours` ([`colours_of`]) over a `Depth32Float` depth
/// attachment.
///
/// [`colours_of`]: crate::gpu::device::tests::colours_of
fn state<P: Platform>(
    d: &P::Device,
    (vertex, fragment): (&str, &str),
    colours: &[&[(u8, u32)]],
) -> P::RenderPipeline {
    let library = d.new_library(LIBRARY).unwrap().unwrap().unwrap();
    let [vertex, fragment] =
        [vertex, fragment].map(|name| library.new_function(OnFault::Die, name).unwrap().unwrap());
    d.new_render_pipeline_state_full(&FullRenderPipelineDesc {
        colors: crate::gpu::device::tests::colours_of(colours),
        depth_pixel_format: u64::from(DEPTH32),
        ..FullRenderPipelineDesc::new(&vertex, Some(&fragment))
    })
    .unwrap()
    .unwrap()
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

/// A `SIDE` by `SIDE` texture of `format`.
fn target<P: Platform>(p: &P, d: &P::Device, format: u32) -> P::Texture {
    let desc = p.texture_descriptor(&TextureDesc {
        texture_type: 2,
        pixel_format: u64::from(format),
        width: SIDE,
        height: SIDE,
        sample_count: 1,
        usage: USAGE,
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
    let state = state::<P>(d, ("cover", "paint"), &[element]);
    let colour = target(p, d, RGBA8);
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
    let (p, d) = (platform(), dev());
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
                let ours = denormal_depth_drawn(&p, &d, stored, cleared, compare);
                let case = format!("{bits:#010x}, cleared {cleared}, compare {compare}");
                crate::metal_check_raw!(&case, ours);
                cases += 1;
            }
        }
    }
    assert_eq!(cases, 36);
}
