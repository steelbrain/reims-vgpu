//! The tests of `encode_render` that compare with Metal on this host: macOS only.

use std::sync::Arc;

use crate::core::platform::values::FullRenderPipelineDesc;

use super::*;

/// The display's full-screen triangle strip's positions (float4 × 4).
const POS_DATA: [f32; 16] = [
    -1.0, -1.0, 0.0, 1.0, -1.0, 1.0, 0.0, 1.0, 1.0, -1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0,
];
/// Its texture coordinates (float2 × 4).
const TEX_DATA: [f32; 8] = [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0];

/// `MTLPixelFormatRGBA8Unorm`, `BGRA8Unorm`, `RGBA16Float` and `R32Float`.
const RGBA8: u64 = 70;

const R32F: u64 = 55;

/// `MTLResourceStorageModeManaged`: the property buffer's options.
const MANAGED: u64 = 0x10;

/// The fragment properties as core writes them: the matrix's four columns of three (each
/// padded to four), then the slopes at 0 and at 1.
fn properties(matrix: [[f32; 4]; 3], at_zero: [f32; 3], at_one: [f32; 3]) -> [f32; 24] {
    let mut p = [0.0f32; 24];
    for (col, column) in p.chunks_mut(4).take(4).enumerate() {
        for (row, v) in column.iter_mut().take(3).enumerate() {
            *v = matrix[row][col];
        }
    }
    p[16..19].copy_from_slice(&at_zero);
    p[19..22].copy_from_slice(&at_one);
    p
}

/// A gamma pass's table (the same ramp in each of the three slices) and properties.
struct Gamma {
    ramp: Vec<f32>,
    properties: [f32; 24],
}

/// The gamma table texture core makes: a 1D array of three `R32Float` slices of `ramp`.
fn gamma_table<P: Platform>(p: &P, d: &P::Device, ramp: &[f32]) -> P::Texture {
    let n = u64::try_from(ramp.len()).unwrap();
    let desc = p.texture_descriptor(&TextureDesc {
        texture_type: 1,
        pixel_format: R32F,
        width: n,
        array_length: 3,
        usage: READ,
        ..TextureDesc::default()
    });
    let t = d.new_texture(OnFault::Die, &desc).unwrap().unwrap();
    let row = bytes(ramp);
    let region = MtlRegion::new(MtlOrigin::new(0, 0, 0), MtlSize::new(n, 1, 1));
    for slice in 0..3 {
        t.replace_region(region, 0, slice, HostRange::of_slice(&row), n * 4, n * 4)
            .unwrap();
    }
    t
}

/// The present pass as core's `display/render.rs` encodes it, from `frame` into a BGRA8
/// target of its size, with the display's pipeline of core's cache (the plain copy with
/// `gamma` `None`, else the gamma shader): the target cleared to opaque black, the pipeline,
/// the viewport of the target with `znear = zfar = 0`, the strip's buffers at vertex 0 and 1,
/// the gamma table (nil without one) at fragment texture 1, the frame at 0, the Managed
/// property buffer at fragment buffer 0, a four-vertex strip. The target's pixels, read back
/// by a blit, as RGBA.
fn present<P: Platform>(
    p: &P,
    d: &P::Device,
    frame: &P::Texture,
    gamma: Option<&Gamma>,
) -> Vec<[u8; 4]> {
    present_as(p, d, frame, gamma, Variant::Strip)
}

/// How a test pass draws after the present pass's sets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Variant {
    /// The present pass's strip.
    Strip,
    /// The strip of 16-bit indices 0 to 3.
    Indexed,
    /// The strip of indices 1 to 4 at a base vertex of -1, one instance.
    IndexedBase,
    /// The strip as one instance of an instanced draw at base instance 0.
    Instances,
    /// The strip under a scissor rectangle of the target's left half.
    Scissor,
    /// The indexed strip with index type 2, which Metal draws nothing for.
    IndexType2,
    /// The strip's four vertices as primitive type 7, one quad.
    Primitive7,
    /// `count` of [`scattered`]'s vertices from `start` as private primitive type
    /// `primitive_type`, `instances` instances of them.
    Private {
        primitive_type: u64,
        start: u64,
        count: u64,
        instances: u64,
    },
    /// The strip as an instanced draw of no instances, direct or indexed.
    NoInstances { indexed: bool },
    /// The strip after a nil depth-stencil state, Metal's default.
    NilDepthStencil,
    /// The strip after a late colour resolve where the pass named none.
    UnnamedColorResolve,
}

/// The commands of `variant` that end a test pass before `endEncoding`.
fn variant_cmds<'a, P: Platform>(
    variant: Variant,
    indices: &'a P::Buffer,
    frame: &'a P::Texture,
    (width, height): (u64, u64),
) -> Vec<RenderCmd<'a, P>> {
    let strip = RenderCmd::DrawPrimitives {
        primitive_type: STRIP,
        vertex_start: 0,
        vertex_count: 4,
    };
    let indexed = |index_type| RenderCmd::DrawIndexedPrimitives {
        primitive_type: STRIP,
        index_count: 4,
        index_type,
        index_buffer: indices,
        index_buffer_offset: 0,
    };
    match variant {
        Variant::Strip => vec![strip],
        Variant::Indexed => vec![indexed(U16)],
        Variant::IndexedBase => vec![RenderCmd::DrawIndexedPrimitivesBase {
            primitive_type: STRIP,
            index_count: 4,
            index_type: U16,
            index_buffer: indices,
            index_buffer_offset: 8,
            instance_count: 1,
            base_vertex: -1,
            base_instance: 0,
        }],
        Variant::Instances => vec![RenderCmd::DrawPrimitivesBase {
            primitive_type: STRIP,
            vertex_start: 0,
            vertex_count: 4,
            instance_count: 1,
            base_instance: 0,
        }],
        Variant::Scissor => vec![
            RenderCmd::SetScissorRect {
                rect: MtlScissorRect {
                    x: 0,
                    y: 0,
                    width: width / 2,
                    height,
                },
            },
            strip,
        ],
        Variant::IndexType2 => vec![indexed(2)],
        Variant::Primitive7 => vec![RenderCmd::DrawPrimitives {
            primitive_type: 7,
            vertex_start: 0,
            vertex_count: 4,
        }],
        Variant::Private {
            primitive_type,
            start,
            count,
            instances,
        } => vec![RenderCmd::DrawPrimitivesInstances {
            primitive_type,
            vertex_start: start,
            vertex_count: count,
            instance_count: instances,
        }],
        Variant::NoInstances { indexed: false } => vec![RenderCmd::DrawPrimitivesInstances {
            primitive_type: STRIP,
            vertex_start: 0,
            vertex_count: 4,
            instance_count: 0,
        }],
        Variant::NoInstances { indexed: true } => {
            vec![RenderCmd::DrawIndexedPrimitivesInstances {
                primitive_type: STRIP,
                index_count: 4,
                index_type: U16,
                index_buffer: indices,
                index_buffer_offset: 0,
                instance_count: 0,
            }]
        }
        Variant::NilDepthStencil => vec![
            RenderCmd::SetDepthStencilState {
                depth_stencil_state: None,
            },
            strip,
        ],
        Variant::UnnamedColorResolve => vec![
            RenderCmd::SetColorResolveTexture {
                texture: Some(frame),
                slice: 0,
                depth_plane: 0,
                level: 0,
                y_invert: true,
                index: 0,
            },
            strip,
        ],
    }
}

/// [`present`]'s pass, drawn as `variant` says.
fn present_as<P: Platform>(
    p: &P,
    d: &P::Device,
    frame: &P::Texture,
    gamma: Option<&Gamma>,
    variant: Variant,
) -> Vec<[u8; 4]> {
    let (w, h) = (frame.width(), frame.height());
    let index_bytes: Vec<u8> = [0u16, 1, 2, 3, 1, 2, 3, 4]
        .iter()
        .flat_map(|i| i.to_ne_bytes())
        .collect();
    let indices = buffer::<P>(d, 16, 0, &index_bytes);
    let cache = DisplayPipelineCache::<P>::new(d.clone_ref()).unwrap();
    let pipeline = cache
        .pipeline_for(PipelineKey {
            format: BGRA8,
            gamma: gamma.is_some(),
        })
        .unwrap();
    let dst = texture(p, d, BGRA8, TARGET | READ, (w, h), None);
    let (pos_data, tex_data) = match variant {
        Variant::Private { .. } => scattered(),
        _ => (POS_DATA.to_vec(), TEX_DATA.to_vec()),
    };
    let length = |data: &[f32]| u64::try_from(data.len() * 4).unwrap();
    let pos = buffer::<P>(d, length(&pos_data), 0, &bytes(&pos_data));
    let tex = buffer::<P>(d, length(&tex_data), 0, &bytes(&tex_data));
    let identity = properties(IDENTITY, [0.0; 3], [0.0; 3]);
    let props = gamma.map_or(identity, |g| g.properties);
    let props = buffer::<P>(d, 96, MANAGED, &bytes(&props));
    props
        .did_modify_range(OnFault::Die, NsRange::new(0, 96))
        .unwrap();
    let table = gamma.map(|g| gamma_table(p, d, &g.ramp));
    let pass = p.render_pass_descriptor();
    pass.set_color_attachment(
        OnFault::Die,
        0,
        &ColorAttachmentSet {
            texture: Some(Some(&dst)),
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
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
    #[expect(clippy::as_conversions, reason = "small sizes, exact in f64")]
    let viewport = MtlViewport {
        origin_x: 0.0,
        origin_y: 0.0,
        width: w as f64,
        height: h as f64,
        znear: 0.0,
        zfar: 0.0,
    };
    for cmd in [
        RenderCmd::SetRenderPipelineState {
            pipeline_state: Some(&pipeline),
        },
        RenderCmd::SetViewport { viewport },
        RenderCmd::SetVertexBuffer {
            buffer: Some(&pos),
            offset: 0,
            index: 0,
        },
        RenderCmd::SetVertexBuffer {
            buffer: Some(&tex),
            offset: 0,
            index: 1,
        },
        RenderCmd::SetFragmentTexture {
            texture: table.as_ref(),
            index: 1,
        },
        RenderCmd::SetFragmentTexture {
            texture: Some(frame),
            index: 0,
        },
        RenderCmd::SetFragmentBuffer {
            buffer: Some(&props),
            offset: 0,
            index: 0,
        },
    ]
    .into_iter()
    .chain(variant_cmds::<P>(variant, &indices, frame, (w, h)))
    .chain([RenderCmd::EndEncoding])
    {
        e.encode(OnFault::Die, cmd).unwrap();
    }
    let out = buffer::<P>(d, w * h * 4, 0, &[]);
    let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
    blit.encode(
        OnFault::Die,
        BlitCmd::CopyTextureToBuffer {
            source_texture: &dst,
            source_slice: 0,
            source_level: 0,
            source_origin: MtlOrigin::new(0, 0, 0),
            source_size: MtlSize::new(w, h, 1),
            destination_buffer: &out,
            destination_offset: 0,
            destination_bytes_per_row: w * 4,
            destination_bytes_per_image: w * h * 4,
        },
    )
    .unwrap();
    blit.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    assert!(cb.error().is_none(), "{:?}", cb.error());
    let n = usize::try_from(w * h * 4).unwrap();
    // SAFETY: a Shared buffer's bytes, read after the blit completed.
    let texels = unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), n) };
    texels
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| [b[2], b[1], b[0], b[3]])
        .collect()
}

/// Runs `check` on this platform, whose expected values are Metal's.
fn on_both(check: impl Fn(&dyn Fn(Option<&Gamma>, Frame) -> Vec<[u8; 4]>)) {
    let p = platform();
    let d = dev();
    check(&|gamma, frame| {
        let f = frame.make(&p, &d);
        present(&p, &d, &f, gamma)
    });
}

/// A frame to present: its format, size and texels.
#[derive(Clone, Copy)]
struct Frame<'a> {
    format: u64,
    size: (u64, u64),
    texels: &'a [u8],
    bytes_per_texel: u64,
}

impl Frame<'_> {
    fn make<P: Platform>(&self, p: &P, d: &P::Device) -> P::Texture {
        let row = self.size.0 * self.bytes_per_texel;
        texture(p, d, self.format, READ, self.size, Some((self.texels, row)))
    }
}

/// An RGBA8 frame of `px`, `w` × `h`.
fn rgba8(px: &[[u8; 4]], size: (u64, u64)) -> (Vec<u8>, (u64, u64)) {
    (px.as_flattened().to_vec(), size)
}

/// The bytes of half floats.
fn halves(h: &[[u16; 4]]) -> Vec<u8> {
    h.iter().flatten().flat_map(|v| v.to_ne_bytes()).collect()
}

/// The copy draws the frame upright, pixel for pixel, alpha included: neither premultiplied
/// nor blended (`shaders.rs`'s Metal expectation)
#[test]
fn the_copy_passes_pixels_and_alpha_through() {
    let px: Vec<[u8; 4]> = (0..32u8)
        .map(|i| {
            [
                i.wrapping_mul(37),
                200 - i,
                i.wrapping_mul(5) + 3,
                [0, 0x7f, 0xff, 0x80][usize::from(i % 4)],
            ]
        })
        .collect();
    let (texels, size) = rgba8(&px, (8, 4));
    on_both(|present| {
        let frame = Frame {
            format: RGBA8,
            size,
            texels: &texels,
            bytes_per_texel: 4,
        };
        assert_eq!(present(None, frame), px);
    });
}

/// An RGBA16Float frame is converted to unorm8 by the target: clamped, rounded to nearest
#[test]
fn the_copy_clamps_extended_range_in_the_target() {
    // -0.5, 0.5, 1.75, 0.25 and 1.0, 0.0, 0.5, 0.75, as half floats.
    let texels = halves(&[
        [0xb800, 0x3800, 0x3f00, 0x3400],
        [0x3c00, 0, 0x3800, 0x3a00],
    ]);
    on_both(|present| {
        let frame = Frame {
            format: RGBA16F,
            size: (2, 1),
            texels: &texels,
            bytes_per_texel: 8,
        };
        assert_eq!(
            present(None, frame),
            vec![[0, 0x80, 0xff, 0x40], [0xff, 0, 0x80, 0xbf]]
        );
    });
}

/// With identity ramps (256 entries) and the identity matrix, the gamma shader is the copy
/// for colour, and every alpha is 0xff
#[test]
fn identity_gamma_is_the_copy_with_opaque_alpha() {
    let px: Vec<[u8; 4]> = (0..=255u8)
        .map(|i| [i, 255 - i, i.wrapping_mul(7), i.wrapping_mul(3)])
        .collect();
    let (texels, size) = rgba8(&px, (16, 16));
    let gamma = Gamma {
        ramp: (0..256u16).map(|i| f32::from(i) / 255.0).collect(),
        properties: properties(IDENTITY, [1.0; 3], [1.0; 3]),
    };
    on_both(|present| {
        let frame = Frame {
            format: RGBA8,
            size,
            texels: &texels,
            bytes_per_texel: 4,
        };
        let out = present(Some(&gamma), frame);
        let opaque: Vec<_> = px.iter().map(|p| [p[0], p[1], p[2], 0xff]).collect();
        assert_eq!(out, opaque);
    });
}

/// A two-entry ramp is a threshold at 0.5, not an interpolation
#[test]
fn a_two_entry_table_is_a_threshold() {
    let (texels, size) = rgba8(
        &[[0x7f, 0x80, 0x00, 0x10], [0xff, 0x40, 0xc0, 0x20]],
        (2, 1),
    );
    let gamma = Gamma {
        ramp: vec![0.25, 0.75],
        properties: properties(IDENTITY, [0.5; 3], [0.5; 3]),
    };
    on_both(|present| {
        let frame = Frame {
            format: RGBA8,
            size,
            texels: &texels,
            bytes_per_texel: 4,
        };
        // 0.25 is 63.75, 0.75 191.25, each rounded to nearest.
        assert_eq!(
            present(Some(&gamma), frame),
            vec![[0x40, 0xbf, 0x40, 0xff], [0xbf, 0x40, 0xbf, 0xff]]
        );
    });
}

/// Row c of the matrix gives channel c from the looked-up colour and the fourth column
#[test]
fn the_matrix_follows_the_table() {
    let (texels, size) = rgba8(
        &[[0xff, 0x00, 0x80, 0xff], [0x00, 0xff, 0x00, 0x00]],
        (2, 1),
    );
    // Red and blue swapped; green is half of green plus 0.25.
    let matrix = [
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.5, 0.0, 0.25],
        [1.0, 0.0, 0.0, 0.0],
    ];
    let gamma = Gamma {
        ramp: (0..256u16).map(|i| f32::from(i) / 255.0).collect(),
        properties: properties(matrix, [1.0; 3], [1.0; 3]),
    };
    on_both(|present| {
        let frame = Frame {
            format: RGBA8,
            size,
            texels: &texels,
            bytes_per_texel: 4,
        };
        assert_eq!(
            present(Some(&gamma), frame),
            vec![[0x80, 0x40, 0xff, 0xff], [0x00, 0xbf, 0x00, 0xff]]
        );
    });
}

/// Outside [0, 1] the table's end entry is extended by its slope at that end
#[test]
fn out_of_range_components_follow_the_end_slopes() {
    // -0.25, 1.5, 0.5 as half floats.
    let texels = halves(&[[0xb400, 0x3e00, 0x3800, 0x3c00]]);
    let gamma = Gamma {
        ramp: vec![0.2, 0.4, 0.6],
        // Red's slope at 0 is 0.4 (-0.25 gives 0.1); green's at 1 is 0.2 (1.5 gives 0.7).
        properties: properties(IDENTITY, [0.4, 9.0, 9.0], [9.0, 0.2, 9.0]),
    };
    on_both(|present| {
        let frame = Frame {
            format: RGBA16F,
            size: (1, 1),
            texels: &texels,
            bytes_per_texel: 8,
        };
        // 0.1 is 25.5, 0.7 178.5, 0.4 102.
        assert_eq!(present(Some(&gamma), frame), vec![[0x1a, 0xb3, 0x66, 0xff]]);
    });
}

/// `MTLPixelFormatR32Uint`.
const R32UINT: u64 = 53;

/// An RGBA8 frame of 8 × 4 distinct opaque pixels.
fn opaque_frame() -> (Vec<[u8; 4]>, Vec<u8>) {
    let px: Vec<[u8; 4]> = (0..32u8)
        .map(|i| [i.wrapping_mul(29), 250 - i, i.wrapping_mul(3) + 1, 0xff])
        .collect();
    let texels = px.as_flattened().to_vec();
    (px, texels)
}

/// Each variant on this platform, checked against Metal's answer when `on_metal`.
fn variant_on_both(variant: Variant, on_metal: bool) -> Vec<[u8; 4]> {
    let (_, texels) = opaque_frame();
    let frame = Frame {
        format: RGBA8,
        size: (8, 4),
        texels: &texels,
        bytes_per_texel: 4,
    };
    let (p, d) = (platform(), dev());
    let vulkan = present_as(&p, &d, &frame.make(&p, &d), None, variant);
    if on_metal {
        crate::metal_check!(format!("{variant:?}"), vulkan);
    }
    vulkan
}

/// Indexed, based and instanced draws of the strip draw what the strip draws, as on Metal
#[test]
fn draw_forms_draw_the_strip() {
    let (px, _) = opaque_frame();
    for variant in [
        Variant::Indexed,
        Variant::IndexedBase,
        Variant::Instances,
        Variant::NilDepthStencil,
    ] {
        assert_eq!(variant_on_both(variant, true), px, "{variant:?}");
    }
    assert_eq!(variant_on_both(Variant::UnnamedColorResolve, false), px);
}

/// A scissor rectangle keeps the draw to it; the rest keeps the clear colour, as on Metal
#[test]
fn a_scissor_rectangle_bounds_the_draw() {
    let (px, _) = opaque_frame();
    let out = variant_on_both(Variant::Scissor, true);
    for (i, (got, want)) in out.iter().zip(&px).enumerate() {
        let expect = if i % 8 < 4 { *want } else { [0, 0, 0, 0xff] };
        assert_eq!(*got, expect, "pixel {i}");
    }
}

/// An index type no `MTLIndexType` is, and a draw of no instances, draw nothing, as on Metal
#[test]
fn an_index_type_metal_lacks_and_no_instances_draw_nothing() {
    for variant in [
        Variant::IndexType2,
        Variant::NoInstances { indexed: false },
        Variant::NoInstances { indexed: true },
    ] {
        assert_eq!(
            variant_on_both(variant, true),
            vec![[0, 0, 0, 0xff]; 32],
            "{variant:?}"
        );
    }
}

/// Metal's private primitive types 5 to 8 (a line loop, a triangle fan, quads and a quad
/// strip) draw what Metal draws, pixel for pixel, at every count from 0 to 10, from vertex 0
/// and from vertex 2 as two instances; one of too few vertices draws nothing
#[test]
fn private_primitive_types_draw_as_on_metal() {
    let clear = vec![[0, 0, 0, 0xff]; 32];
    assert_ne!(variant_on_both(Variant::Primitive7, true), clear);
    let mut drawn = 0;
    for primitive_type in 5..=8 {
        for count in 0..=10 {
            for (start, instances) in [(0, 1), (2, 2)] {
                let variant = Variant::Private {
                    primitive_type,
                    start,
                    count,
                    instances,
                };
                let out = variant_on_both(variant, true);
                if count < 2
                    || (primitive_type >= 6 && count < 3)
                    || (primitive_type >= 7 && count < 4)
                {
                    assert_eq!(out, clear, "{variant:?}");
                }
                drawn += usize::from(out != clear);
            }
        }
    }
    assert!(drawn > 40, "{drawn} draws drew anything");
}

/// The functions of a state whose colour is each primitive's `[[primitive_id]]`.
const PRIMITIVES: (&str, &str) = ("vertex_ids:ids_vertex", "vertex_ids:primitive_fragment");

/// Quads and quad strips, of which Metal counts a quad one primitive, give a fragment function's
/// `[[primitive_id]]` as Metal's, directly at every count from 4 to 10, from vertex 0 and from
/// vertex 2 as two instances, at `setProvokingVertexMode:` 0 and 1; over indices and from
/// indirect arguments in `private_types_draw_over_indices_and_arguments_as_metals`
#[test]
fn quads_give_metals_primitive_ids() {
    let (p, d) = (platform(), dev());
    let clear = vec![[0, 0, 0, 0xff]; 256];
    for primitive_type in [7, 8] {
        for count in 4..=10 {
            for (start, instances) in [(0, 1), (2, 2)] {
                for mode in [0, 1] {
                    let draw = (primitive_type, start, count, instances);
                    let key = format!("{draw:?} at mode {mode}");
                    let ours = flat_pass(&p, &d, PRIMITIVES, mode, draw);
                    assert!(ours != clear, "{key}");
                    assert!(crate::metal_matches_raw!(key, ours), "{key}");
                }
            }
        }
    }
}

/// Metal's private types, and its public ones but points (which this state, of no point size,
/// does not size), drawn directly from a vertex start whose last vertex passes `i32::MAX`
/// (`2^31 - 3`, `2^31 + 3`, and `2^32 - 4`, whose vertex IDs wrap past `u32::MAX`), of one instance
/// and two, draw what Metal draws, pixel for pixel, each vertex's `[[vertex_id]]` and the draw's
/// `[[base_vertex]]` Metal's (`wrapped_vertex`), at `setProvokingVertexMode:` 0 and 1, but lines,
/// line strips and the line loop's lines at 1 (`lines_take_metals_flat_inputs_at_the_last_vertex`).
/// A public type's at 1 the runtime draws as the list of its primitives too, where it refused one
#[test]
fn private_types_from_far_vertex_starts_draw_as_metals() {
    let (p, d) = (platform(), dev());
    let wrapped = ("vertex_ids:wrapped_vertex", "vertex_ids:ids_fragment");
    let clear = vec![[0, 0, 0, 0xff]; 256];
    let mut differ = Vec::new();
    for t in 1..=8 {
        let modes: &[u64] = if matches!(t, 1 | 2 | 5) {
            &[0]
        } else {
            &[0, 1]
        };
        for start in [(1 << 31) - 3, (1 << 31) + 3, (1 << 32) - 4] {
            for (count, instances) in [(8, 1), (10, 2)] {
                for &mode in modes {
                    let draw = (t, start, count, instances);
                    *crate::answer::LAST_LINE.lock() = None;
                    let ours = catch_death(|| flat_pass(&p, &d, wrapped, mode, draw));
                    let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
                    let ours = match ours {
                        Ok(ours) => ours,
                        Err(site) => {
                            differ
                                .push(format!("{draw:?} at mode {mode}: died at {site:?}: {line}"));
                            continue;
                        }
                    };
                    if ours == clear {
                        differ.push(format!("{draw:?} at mode {mode}: drew nothing"));
                    }
                    let key = format!("{draw:?} at mode {mode}");
                    if !crate::metal_matches_raw!(key, ours) {
                        differ.push(format!("{key}: pixels differ"));
                    }
                }
            }
        }
    }
    assert!(differ.is_empty(), "{} differ: {differ:#?}", differ.len());
}

/// Indices of twelve vertices out of order; of twelve with 3 again among them, which a quad strip
/// restarts at after `setPrimitiveRestartEnabled:index:` of 3 and the other private types take as
/// a vertex; and of twelve with the 16-bit all-ones index (the 32-bit one at the same place)
/// after six, which a quad strip restarts at.
const SCRAMBLED: &[u32] = &[5, 0, 9, 1, 7, 3, 2, 8, 4, 10, 11, 6];
const SCRAMBLED_AT_3: &[u32] = &[0, 1, 2, 3, 4, 5, 6, 3, 7, 8, 9, 10, 11];
const QUADS_RESTARTED: &[u32] = &[0, 1, 2, 3, 4, 5, 0xffff, 6, 7, 8, 9, 10, 11];
const QUADS_RESTARTED_WIDE: &[u32] = &[0, 1, 2, 3, 4, 5, 0xffff_ffff, 6, 7, 8, 9, 10, 11];

/// Metal's private types 5 to 8 draw what Metal draws over 16- and 32-bit indices, from a base
/// vertex, of instances, after `setPrimitiveRestartEnabled:index:`, a quad strip restarting at
/// its index type's all-ones index, and from indirect arguments, over indices too, at
/// `setProvokingVertexMode:` 0 and 1, but the line loop's lines at 1
/// (`lines_take_metals_flat_inputs_at_the_last_vertex`), and as an indirect command buffer's
/// command, over its vertices and over indices, inheriting the encoder's state and buffers or
/// with its own: under a state whose colour is each vertex's, flat, one whose colour is its
/// provoking vertex's `[[vertex_id]]`, and one whose colour is its primitive's
/// `[[primitive_id]]`. All but a fan or quads after the restart index set draw something
#[test]
fn private_types_draw_over_indices_and_arguments_as_metals() {
    let (p, d) = (platform(), dev());
    let ids = ("vertex_ids:ids_vertex", "vertex_ids:ids_fragment");
    let indirect = |primitive_type, index_type, indices, arguments| Provoked::IndexedIndirect {
        primitive_type,
        index_type,
        indices,
        arguments,
    };
    let executed = |inherits, primitive_type, count, indices| Provoked::Executed {
        inherits,
        primitive_type,
        count,
        indices,
    };
    let clear = vec![[0, 0, 0, 0xff]; 256];
    let mut blank = 0;
    for t in 5..=8 {
        let mut cases = vec![
            (over((t, U16), SCRAMBLED, (1, 2)), None),
            (over((t, U32), SCRAMBLED, (0, 1)), None),
            (over((t, U16), SCRAMBLED_AT_3, (0, 1)), Some((true, 3))),
            (Provoked::Indirect(t, [10, 1, 2, 0]), None),
            (Provoked::Indirect(t, [9, 2, 0, 0]), None),
            (indirect(t, U16, SCRAMBLED, [12, 1, 0, 2, 0]), None),
            (indirect(t, U32, SCRAMBLED, [8, 2, 3, 0, 0]), None),
            (executed(false, t, 10, None), None),
            (executed(true, t, 9, None), None),
            (executed(false, t, 0, Some((U16, SCRAMBLED))), None),
            (executed(true, t, 0, Some((U32, SCRAMBLED))), None),
        ];
        if t == 8 {
            cases.push((over((t, U16), QUADS_RESTARTED, (0, 1)), None));
            cases.push((over((t, U32), QUADS_RESTARTED_WIDE, (0, 1)), None));
        }
        let modes: &[u64] = if t == 5 { &[0] } else { &[0, 1] };
        for (draw, restart) in cases {
            for functions in [FLAT, ids, PRIMITIVES] {
                for &mode in modes {
                    let ours = provoked(&p, &d, functions, (mode, restart), draw);
                    blank += usize::from(ours == clear);
                    let key = format!("{draw:?} of {functions:?} at mode {mode}");
                    assert!(crate::metal_matches_raw!(key, ours), "{key}");
                }
            }
        }
    }
    assert!(blank <= 12, "{blank} draws drew nothing");
}

/// The pixels of [`pass_pixels`] after `instances` instances of `count` of [`scattered`]'s
/// vertices from `start`, as `primitive_type`, under `setProvokingVertexMode:` of `mode` and
/// a state of the fixtures `functions`, over [`flat_vertices`] in buffers 0 and 1.
fn flat_pass<P: Platform>(
    p: &P,
    d: &P::Device,
    functions: (&str, &str),
    mode: u64,
    (primitive_type, start, count, instances): (u64, u64, u64, u64),
) -> Vec<[u8; 4]> {
    let draw = RenderCmd::DrawPrimitivesInstances {
        primitive_type,
        vertex_start: start,
        vertex_count: count,
        instance_count: instances,
    };
    flat_draw(p, d, functions, mode, draw)
}

/// The pixels of [`flat_pass`] after `draw` in place of its own.
fn flat_draw<P: Platform>(
    p: &P,
    d: &P::Device,
    functions: (&str, &str),
    mode: u64,
    draw: RenderCmd<'_, P>,
) -> Vec<[u8; 4]> {
    set_draw(
        p,
        d,
        functions,
        RenderCmd::SetProvokingVertexMode { mode },
        draw,
    )
}

/// The pixels of [`flat_draw`] after `setter` in place of its `setProvokingVertexMode:`.
fn set_draw<P: Platform>(
    p: &P,
    d: &P::Device,
    (vertex, fragment): (&str, &str),
    setter: RenderCmd<'_, P>,
    draw: RenderCmd<'_, P>,
) -> Vec<[u8; 4]> {
    let state = fixture_state::<P>(d, vertex, fragment);
    let (pos, colours) = flat_vertices::<P>(d, 16);
    pass_pixels(
        p,
        d,
        vec![
            RenderCmd::SetRenderPipelineState {
                pipeline_state: Some(&state),
            },
            RenderCmd::SetVertexBuffer {
                buffer: Some(&pos),
                offset: 0,
                index: 0,
            },
            RenderCmd::SetVertexBuffer {
                buffer: Some(&colours),
                offset: 0,
                index: 1,
            },
            setter,
            draw,
        ],
    )
}

/// A guest's state of the fixtures `vertex` and `fragment` for a BGRA8 target, the whole render
/// value `edit` leaves (the input primitive topology class, indirect command buffer support, the
/// sample count, the depth format), Metal's defaults elsewhere.
fn value_state<P: Platform>(
    d: &P::Device,
    (vertex, fragment): (&str, &str),
    edit: impl FnOnce(&mut FullRenderPipelineDesc<'_, P>),
) -> P::RenderPipeline {
    let (vertex, fragment) = (fixture::<P>(d, vertex), fixture::<P>(d, fragment));
    let mut desc =
        crate::gpu::device::tests::render_value::<P>((&vertex, &fragment), 1, &[&[(1, 80)]]);
    edit(&mut desc);
    d.new_render_pipeline_state_full(&desc).unwrap().unwrap()
}

/// A draw of [`provoked`]'s: `(type, start, count, instances)` over the vertices; over `indices`
/// of an index type, from a base vertex, of instances; from indirect arguments, over `indices`
/// too; or an execution of an indirect command buffer's one command of a type, of `count` vertices
/// from vertex 0, or of `indices` of an index type where it has them, which inherits the
/// encoder's state and buffers or sets its own.
#[derive(Clone, Copy, Debug)]
enum Provoked<'a> {
    Vertices(u64, u64, u64, u64),
    Indices {
        primitive_type: u64,
        index_type: u64,
        indices: &'a [u32],
        base_vertex: i64,
        instances: u64,
    },
    Indirect(u64, [u32; 4]),
    IndexedIndirect {
        primitive_type: u64,
        index_type: u64,
        indices: &'a [u32],
        arguments: [u32; 5],
    },
    Executed {
        inherits: bool,
        primitive_type: u64,
        count: u64,
        indices: Option<(u64, &'a [u32])>,
    },
}

/// The bytes of `indices` as indices of `index_type`, a 16-bit one each index's low half.
fn index_bytes(index_type: u64, indices: &[u32]) -> Vec<u8> {
    if index_type == U16 {
        let low = |i: u32| u16::try_from(i & 0xffff).unwrap();
        indices.iter().flat_map(|&i| low(i).to_le_bytes()).collect()
    } else {
        u32s(indices)
    }
}

/// The pixels of [`pass_pixels`] after `draw` under a state of the fixtures `functions` (of
/// indirect command buffer support for an execution) over [`flat_vertices`] in buffers 0 and 1,
/// at `setProvokingVertexMode:` of `mode`, after `restart`'s `setPrimitiveRestartEnabled:index:`
/// where it has one.
fn provoked<P: Platform>(
    p: &P,
    d: &P::Device,
    functions: (&str, &str),
    (mode, restart): (u64, Option<(bool, u64)>),
    draw: Provoked<'_>,
) -> Vec<[u8; 4]> {
    use crate::core::platform::icb::{
        IcbRenderCmd, IndirectCommandBuffer as _, IndirectRenderCommand as _,
    };
    use crate::core::platform::values::IcbDesc;

    let executed = matches!(draw, Provoked::Executed { .. });
    let state = value_state::<P>(d, functions, |v| {
        v.support_indirect_command_buffers = executed
    });
    let (pos, colours) = flat_vertices::<P>(d, 64);
    let (index_buffer, indirect_buffer) = provoked_buffers::<P>(d, draw);
    let icb = match draw {
        Provoked::Executed {
            inherits,
            primitive_type,
            count,
            indices,
        } => {
            let icb = d
                .new_indirect_command_buffer(
                    &IcbDesc {
                        // `MTLIndirectCommandTypeDrawIndexed`, or `Draw`.
                        command_types: if indices.is_some() { 2 } else { 1 },
                        inherit_pipeline_state: inherits,
                        inherit_buffers: inherits,
                        max_vertex_buffer_bind_count: 2,
                        ..IcbDesc::default()
                    },
                    1,
                    0,
                )
                .unwrap()
                .unwrap();
            let c = icb.render_command(0).unwrap().unwrap();
            let mut cmds = Vec::new();
            if !inherits {
                cmds.extend([
                    IcbRenderCmd::SetRenderPipelineState {
                        pipeline_state: Some(&state),
                    },
                    IcbRenderCmd::SetVertexBuffer {
                        buffer: Some(&pos),
                        offset: 0,
                        index: 0,
                    },
                    IcbRenderCmd::SetVertexBuffer {
                        buffer: Some(&colours),
                        offset: 0,
                        index: 1,
                    },
                ]);
            }
            cmds.push(match indices {
                Some((index_type, indices)) => IcbRenderCmd::DrawIndexedPrimitives {
                    primitive_type,
                    index_count: u64::try_from(indices.len()).unwrap(),
                    index_type,
                    index_buffer: Some(&index_buffer),
                    index_buffer_offset: 0,
                    instance_count: 1,
                    base_vertex: 0,
                    base_instance: 0,
                },
                None => IcbRenderCmd::DrawPrimitives {
                    primitive_type,
                    vertex_start: 0,
                    vertex_count: count,
                    instance_count: 1,
                    base_instance: 0,
                },
            });
            for cmd in cmds {
                c.encode(cmd).unwrap();
            }
            Some(icb)
        }
        _ => None,
    };
    let used = icb.as_ref().map(|icb| {
        [
            Some(ResourceRef::Icb(icb)),
            Some(ResourceRef::Buffer(&pos)),
            Some(ResourceRef::Buffer(&colours)),
            Some(ResourceRef::Buffer(&index_buffer)),
        ]
    });
    let mut cmds = vec![
        RenderCmd::SetRenderPipelineState {
            pipeline_state: Some(&state),
        },
        RenderCmd::SetVertexBuffer {
            buffer: Some(&pos),
            offset: 0,
            index: 0,
        },
        RenderCmd::SetVertexBuffer {
            buffer: Some(&colours),
            offset: 0,
            index: 1,
        },
    ];
    if let Some((enabled, index)) = restart {
        cmds.push(RenderCmd::SetPrimitiveRestartEnabled { enabled, index });
    }
    cmds.push(RenderCmd::SetProvokingVertexMode { mode });
    match provoked_draw(draw, (&index_buffer, &indirect_buffer)) {
        Some(drawn) => cmds.push(drawn),
        None => {
            let (icb, used) = (icb.as_ref().unwrap(), used.as_ref().unwrap());
            cmds.extend([
                RenderCmd::UseResources {
                    resources: used,
                    usage: 1,
                },
                RenderCmd::ExecuteCommandsInBuffer {
                    indirect_command_buffer: icb,
                    execution_range: NsRange::new(0, 1),
                },
            ]);
        }
    }
    pass_pixels(p, d, cmds)
}

/// The index and indirect buffers of `draw`, a draw of [`provoked`]'s, on `d`: its indices and its
/// arguments, each empty where it has none.
fn provoked_buffers<P: Platform>(d: &P::Device, draw: Provoked<'_>) -> (P::Buffer, P::Buffer) {
    let (indices, arguments) = match draw {
        Provoked::Indices {
            index_type,
            indices,
            ..
        } => (index_bytes(index_type, indices), Vec::new()),
        Provoked::Indirect(_, arguments) => (Vec::new(), u32s(&arguments)),
        Provoked::IndexedIndirect {
            index_type,
            indices,
            arguments,
            ..
        } => (index_bytes(index_type, indices), u32s(&arguments)),
        Provoked::Executed {
            indices: Some((index_type, indices)),
            ..
        } => (index_bytes(index_type, indices), Vec::new()),
        _ => (Vec::new(), Vec::new()),
    };
    (
        buffer::<P>(d, 256, 0, &indices),
        buffer::<P>(d, 64, 0, &arguments),
    )
}

/// The command of `draw`, a draw of [`provoked`]'s, over `index_buffer` and `indirect_buffer`
/// ([`provoked_buffers`]); `None` for an execution, which [`provoked`] encodes itself.
fn provoked_draw<'a, P: Platform>(
    draw: Provoked<'_>,
    (index_buffer, indirect_buffer): (&'a P::Buffer, &'a P::Buffer),
) -> Option<RenderCmd<'a, P>> {
    Some(match draw {
        Provoked::Vertices(primitive_type, start, count, instances) => {
            RenderCmd::DrawPrimitivesInstances {
                primitive_type,
                vertex_start: start,
                vertex_count: count,
                instance_count: instances,
            }
        }
        Provoked::Indices {
            primitive_type,
            index_type,
            indices,
            base_vertex,
            instances,
        } => RenderCmd::DrawIndexedPrimitivesBase {
            primitive_type,
            index_count: u64::try_from(indices.len()).unwrap(),
            index_type,
            index_buffer,
            index_buffer_offset: 0,
            instance_count: instances,
            base_vertex,
            base_instance: 0,
        },
        Provoked::Indirect(primitive_type, _) => RenderCmd::DrawPrimitivesIndirect {
            primitive_type,
            indirect_buffer,
            indirect_buffer_offset: 0,
        },
        Provoked::IndexedIndirect {
            primitive_type,
            index_type,
            ..
        } => RenderCmd::DrawIndexedPrimitivesIndirect {
            primitive_type,
            index_type,
            index_buffer,
            index_buffer_offset: 0,
            indirect_buffer,
            indirect_buffer_offset: 0,
        },
        Provoked::Executed { .. } => return None,
    })
}

/// A draw of `indices` of `index_type` as `primitive_type`, from `base_vertex`, of `instances`.
fn over<'a>(
    (primitive_type, index_type): (u64, u64),
    indices: &'a [u32],
    (base_vertex, instances): (i64, u64),
) -> Provoked<'a> {
    Provoked::Indices {
        primitive_type,
        index_type,
        indices,
        base_vertex,
        instances,
    }
}

/// Indices of a list of three triangles, of a strip of six, of a strip restarting at the 16-bit
/// all-ones index after one triangle, then of five more (the 32-bit all-ones index at the same
/// place), and of a strip of eleven indices restarting at 3 after one triangle.
const LIST: &[u32] = &[5, 0, 9, 1, 7, 3, 2, 8, 4];
const STRIP_OF_SIX: &[u32] = &[0, 3, 1, 4, 2, 5, 7, 6];
const RESTARTED: &[u32] = &[0, 1, 2, 3, 0xffff, 4, 5, 6, 7, 8];
const RESTARTED_WIDE: &[u32] = &[0, 1, 2, 3, 0xffff_ffff, 4, 5, 6, 7, 8];
const RESTARTED_AT_3: &[u32] = &[0, 1, 2, 3, 4, 3, 5, 6, 7, 8, 9];

/// A state with a flat input draws at `setProvokingVertexMode:` 0 and 1 what Metal draws at
/// every kind of draw of triangles: lists and strips over their vertices, from vertex 0 and
/// later, of one instance and two; over 16- and 32-bit indices from a base vertex; strips
/// restarting at the all-ones index and at the index `setPrimitiveRestartEnabled:index:` set,
/// each piece its own parity; from indirect arguments, over indices too; Metal's private types
/// 6 to 8; and an execution of an indirect command buffer's strip or list, whose commands take
/// the first vertex at both modes, inheriting the encoder's state or with their own. The last
/// vertex's colours fill the triangles at mode 1, but an execution's. A draw of too few
/// vertices for a triangle draws nothing at either mode
#[test]
fn flat_inputs_take_metals_provoking_vertex_at_every_draw() {
    let (p, d) = (platform(), dev());
    let indirect = |primitive_type, index_type, indices, arguments| Provoked::IndexedIndirect {
        primitive_type,
        index_type,
        indices,
        arguments,
    };
    let executed = |inherits, primitive_type, count| Provoked::Executed {
        inherits,
        primitive_type,
        count,
        indices: None,
    };
    let cases = [
        (Provoked::Vertices(3, 0, 6, 1), None),
        (Provoked::Vertices(3, 1, 9, 2), None),
        (Provoked::Vertices(STRIP, 0, 5, 1), None),
        (Provoked::Vertices(STRIP, 3, 8, 2), None),
        (Provoked::Vertices(6, 2, 6, 2), None),
        (Provoked::Vertices(7, 1, 8, 1), None),
        (Provoked::Vertices(8, 0, 8, 1), None),
        (over((3, U16), LIST, (2, 1)), None),
        (over((3, U32), LIST, (0, 2)), None),
        (over((STRIP, U16), STRIP_OF_SIX, (0, 1)), None),
        (over((STRIP, U32), STRIP_OF_SIX, (1, 1)), None),
        (over((STRIP, U16), RESTARTED, (0, 1)), None),
        (over((STRIP, U32), RESTARTED_WIDE, (0, 2)), None),
        (over((STRIP, U16), RESTARTED_AT_3, (0, 1)), Some((true, 3))),
        (over((STRIP, U32), RESTARTED_AT_3, (0, 1)), Some((false, 3))),
        (Provoked::Indirect(3, [9, 2, 1, 0]), None),
        (Provoked::Indirect(STRIP, [8, 1, 3, 0]), None),
        (indirect(STRIP, U16, RESTARTED, [10, 1, 0, 0, 0]), None),
        (indirect(3, U32, LIST, [9, 1, 0, 2, 0]), None),
        (executed(true, STRIP, 7), None),
        (executed(false, STRIP, 7), None),
        (executed(false, 3, 9), None),
    ];
    let clear = vec![[0, 0, 0, 0xff]; 256];
    for (draw, restart) in cases {
        let first = provoked(&p, &d, FLAT, (0, restart), draw);
        let last = provoked(&p, &d, FLAT, (1, restart), draw);
        assert_ne!(first, clear, "{draw:?}");
        let execution = matches!(draw, Provoked::Executed { .. });
        assert_eq!(first == last, execution, "{draw:?}");
        for (mode, ours) in [(0, &first), (1, &last)] {
            let key = format!("{draw:?} at mode {mode}");
            assert!(crate::metal_matches_raw!(key, ours), "{key}");
        }
    }
    for draw in [
        Provoked::Vertices(3, 0, 2, 1),
        Provoked::Vertices(STRIP, 0, 2, 1),
    ] {
        for mode in [0, 1] {
            assert_eq!(
                provoked(&p, &d, FLAT, (mode, None), draw),
                clear,
                "{draw:?}"
            );
        }
    }
}

/// The set of colours `pixels` hold, black aside.
fn colours_of(pixels: &[[u8; 4]]) -> std::collections::BTreeSet<[u8; 4]> {
    (pixels.iter().copied())
        .filter(|&px| px != [0, 0, 0, 0xff])
        .collect()
}

/// A state with a flat input draws lines, line strips and Metal's line loop at
/// `setProvokingVertexMode:` 0 as Metal does, over their vertices, over indices (restarting too)
/// and from indirect arguments. At 1 it draws them as Metal does where the device has the last
/// vertex; elsewhere the runtime draws each line from its last vertex to its first, which dies in
/// strict mode (`provoking-vertex-lines`), and by default draws, relayed, the colours Metal's
/// lines take, over pixels a few of which differ from Metal's
#[test]
fn lines_take_metals_flat_inputs_at_the_last_vertex() {
    let (p, d) = (platform(), dev());
    let kind = rt::Degradation::ProvokingVertexLines;
    for draw in [
        Provoked::Vertices(1, 0, 8, 1),
        Provoked::Vertices(2, 3, 7, 2),
        Provoked::Vertices(5, 0, 6, 1),
        over((1, U16), STRIP_OF_SIX, (0, 1)),
        over((2, U16), RESTARTED, (0, 1)),
        Provoked::Indirect(2, [7, 1, 3, 0]),
    ] {
        let first = provoked(&p, &d, FLAT, (0, None), draw);
        let native = super::last_vertex_native();
        let last = if native {
            provoked(&p, &d, FLAT, (1, None), draw)
        } else {
            crate::answer::dies_of(kind, || provoked(&p, &d, FLAT, (1, None), draw));
            let before = super::relayed_kind(kind);
            let last = crate::answer::relaxed(|| provoked(&p, &d, FLAT, (1, None), draw));
            assert!(super::relayed_kind(kind) > before, "{draw:?}");
            last
        };
        assert!(first != last, "{draw:?}");
        let keys = [0, 1].map(|mode| format!("{draw:?} at mode {mode}"));
        let colours = format!("{draw:?} at mode 1: the colours");
        assert!(crate::metal_matches_raw!(keys[0], first), "{draw:?}");
        if native {
            assert!(crate::metal_matches_raw!(keys[1], last), "{draw:?}");
        } else {
            crate::metal_check!(colours, colours_of(&last));
        }
    }
}

/// `setLineWidth:` reaches the runtime's encoder, which draws Metal's widths: lines and line
/// strips over the flat vertices at widths 1, 3, 2.5 and 0 draw as Metal draws them,
/// where the runtime draws the width exactly (its band of triangles, or device lines it takes as
/// exact), at 0 the few pixels Metal lights on the segments (3 of the list's, 5 of the strip's
/// here, the M3 Max). A device's own lines that differ from Metal's say so (`device-wide-lines`),
/// relayed by default, and are not compared. A width of 3 lights more pixels than a width of 1.
/// The first vertex is white: [`vertex_colours`]' first is the clear's black, which would hide
/// the line it provokes
#[test]
fn lines_draw_at_metals_widths() {
    let (p, d) = (platform(), dev());
    let kind = rt::Degradation::DeviceWideLines;
    /// The flat state's draw of `primitive_type` over the first 8 flat vertices at `line_width`,
    /// the first white.
    fn lines<P: Platform>(
        p: &P,
        d: &P::Device,
        line_width: f32,
        primitive_type: u64,
    ) -> Vec<[u8; 4]> {
        let state = fixture_state::<P>(d, FLAT.0, FLAT.1);
        let (pos, _) = flat_vertices::<P>(d, 16);
        let mut colours = vertex_colours();
        colours[..4].copy_from_slice(&[1.0; 4]);
        let length = u64::try_from(colours.len() * 4).unwrap();
        let colours = buffer::<P>(d, length, 0, &bytes(&colours));
        pass_pixels(
            p,
            d,
            vec![
                RenderCmd::SetRenderPipelineState {
                    pipeline_state: Some(&state),
                },
                RenderCmd::SetVertexBuffer {
                    buffer: Some(&pos),
                    offset: 0,
                    index: 0,
                },
                RenderCmd::SetVertexBuffer {
                    buffer: Some(&colours),
                    offset: 0,
                    index: 1,
                },
                RenderCmd::SetLineWidth { line_width },
                RenderCmd::DrawPrimitivesInstances {
                    primitive_type,
                    vertex_start: 0,
                    vertex_count: 8,
                    instance_count: 1,
                },
            ],
        )
    }
    let lit = |px: &[[u8; 4]]| px.iter().filter(|&&t| t != [0, 0, 0, 0xff]).count();
    for primitive_type in [1, 2] {
        let mut lit_at = Vec::new();
        for w in [1.0, 3.0, 2.5, 0.0] {
            let before = super::relayed_kind(kind);
            let ours = crate::answer::relaxed(|| lines(&p, &d, w, primitive_type));
            let exact = super::relayed_kind(kind) == before;
            lit_at.push(lit(&ours));
            let key = format!("{primitive_type} at {w}");
            if exact {
                assert!(crate::metal_matches_raw!(key, ours), "{key}");
            }
        }
        assert!(lit_at[1] > lit_at[0], "{primitive_type}: {lit_at:?}");
    }
}

/// At `setProvokingVertexMode:` 1, a state whose fragment function reads `[[barycentric_coord]]`
/// and no flat input draws what Metal draws: a list's weights in mode 0's order and a strip's odd
/// triangles' moved, over their vertices, over indices, restarting, from indirect arguments, and
/// Metal's triangle fan. One that also reads a flat input draws as Metal does where the device
/// has the last vertex; elsewhere the runtime draws its triangles from their last vertices, their
/// first weights Metal's and the other two Metal's but for rounding, which dies in strict mode
/// (`provoking-vertex-unmatched`). At mode 0 all of them draw as Metal's
#[test]
fn barycentric_weights_take_metals_order_at_the_last_vertex() {
    let (p, d) = (platform(), dev());
    let smooth = ("barycentric:smooth_vertex", "barycentric:smooth_weights");
    let flat = ("barycentric:flat_vertex", "barycentric:flat_weights");
    let draws = [
        Provoked::Vertices(3, 1, 9, 2),
        Provoked::Vertices(STRIP, 3, 8, 1),
        Provoked::Vertices(6, 0, 6, 1),
        over((STRIP, U16), STRIP_OF_SIX, (0, 1)),
        over((STRIP, U16), RESTARTED, (0, 1)),
        Provoked::Indirect(STRIP, [8, 1, 3, 0]),
    ];
    for draw in draws {
        for mode in [0, 1] {
            let ours = provoked(&p, &d, smooth, (mode, None), draw);
            assert_ne!(ours, vec![[0, 0, 0, 0xff]; 256], "{draw:?}");
            let key = format!("smooth: {draw:?} at mode {mode}");
            assert!(crate::metal_matches_raw!(key, ours), "{key}");
        }
    }
    for draw in [
        Provoked::Vertices(3, 1, 9, 1),
        Provoked::Vertices(STRIP, 3, 8, 1),
        over((3, U16), LIST, (0, 1)),
    ] {
        let first = provoked(&p, &d, flat, (0, None), draw);
        let kind = rt::Degradation::ProvokingVertexUnmatched;
        let last = if super::last_vertex_native() {
            Some(provoked(&p, &d, flat, (1, None), draw))
        } else {
            crate::answer::dies_of(kind, || provoked(&p, &d, flat, (1, None), draw));
            None
        };
        let keys = [0, 1].map(|mode| format!("flat: {draw:?} at mode {mode}"));
        assert!(crate::metal_matches_raw!(keys[0], first), "{draw:?}");
        if let Some(last) = last {
            assert!(crate::metal_matches_raw!(keys[1], last), "{draw:?}");
        }
    }
}

/// Metal draws every `setProvokingVertexMode:` mode from 2 up alike, its top bit set or not: a
/// list's triangles as at mode 1, and a strip, a line strip and points as at neither 0 nor 1.
/// This platform draws each as mode 1: a list's triangles as Metal's, relaying nothing, and the
/// strip, the line strip and the sized points relayed at the draw (`provoking-vertex-unmatched`,
/// which dies there in strict mode)
#[test]
fn modes_past_1_draw_metals_third_way() {
    let (p, d) = (platform(), dev());
    let modes = [
        2,
        7,
        0xff,
        0x8000_0000,
        0xffff_ffff,
        1 << 32,
        1 << 62,
        (1 << 63) - 1,
        1 << 63,
        u64::MAX - 1,
        u64::MAX,
    ];
    let kind = rt::Degradation::ProvokingVertexUnmatched;
    let list = (FLAT, Provoked::Vertices(3, 1, 9, 1));
    let others = [
        (FLAT, Provoked::Vertices(STRIP, 0, 7, 1)),
        (FLAT, Provoked::Vertices(2, 3, 7, 1)),
        (SIZED, Provoked::Vertices(0, 0, 3, 1)),
    ];
    for (at, (functions, draw)) in [list].into_iter().chain(others).enumerate() {
        let ours =
            |mode| crate::answer::relaxed(|| provoked(&p, &d, functions, (mode, None), draw));
        let last = ours(1);
        for mode in modes {
            let before = super::relayed_kind(kind);
            assert!(ours(mode) == last, "{draw:?} at {mode:#x}");
            let relayed = super::relayed_kind(kind) > before;
            assert_eq!(relayed, at != 0, "{draw:?} at {mode:#x}");
        }
        // Metal's third way, which a list's triangles draw as at mode 1.
        let key = format!("{draw:?}: the third way");
        if at == 0 {
            assert!(crate::metal_matches_raw!(key, last), "{draw:?}");
        }
    }
}

/// A draw the runtime draws over a list of its own, of a private type, or of a list or strip of
/// triangles at the last vertex under a state that reads a flat input without the device's last
/// vertex, gives each vertex's `[[vertex_id]]` as Metal's non-indexed draw gives it
#[test]
fn re_indexed_draws_keep_metals_vertex_ids() {
    let (p, d) = (platform(), dev());
    let ids = ("vertex_ids:ids_vertex", "vertex_ids:ids_fragment");
    for (mode, draw) in [
        (0, (5, 3, 5, 1)),
        (0, (6, 2, 6, 2)),
        (0, (7, 1, 8, 1)),
        (0, (8, 4, 8, 1)),
        (1, (3, 3, 6, 1)),
        (1, (STRIP, 2, 7, 1)),
    ] {
        let ours = flat_pass(&p, &d, ids, mode, draw);
        let lit: Vec<_> = ours.iter().filter(|px| **px != [0, 0, 0, 0xff]).collect();
        let (start, count) = (u8::try_from(draw.1).unwrap(), u8::try_from(draw.2).unwrap());
        let drawn = |px: &[u8; 4]| (start..start + count).contains(&px[2]);
        assert!(
            !lit.is_empty() && lit.iter().all(|px| drawn(px)),
            "{draw:?}"
        );
        let key = format!("{mode} {draw:?}");
        crate::metal_check_raw!(key, ours);
    }
}

/// The draws of [`re_indexed_draws_keep_metals_vertex_ids`] and a public list from a later
/// vertex, of instances from base instance 3, give `[[base_vertex]]` as their first vertex
/// and `[[base_instance]]` as 3, as Metal's non-indexed draws give them
#[test]
fn re_indexed_draws_keep_metals_draw_bases() {
    /// `draw`, `(type, start, count, instances)`, from base instance 3.
    fn based<'a, P: Platform>(
        (primitive_type, start, count, instances): (u64, u64, u64, u64),
    ) -> RenderCmd<'a, P> {
        RenderCmd::DrawPrimitivesBase {
            primitive_type,
            vertex_start: start,
            vertex_count: count,
            instance_count: instances,
            base_instance: 3,
        }
    }
    let (p, d) = (platform(), dev());
    let bases = ("vertex_ids:bases_vertex", "vertex_ids:ids_fragment");
    for (mode, draw) in [
        (0, (3, 2, 6, 2)),
        (0, (5, 3, 5, 1)),
        (0, (6, 2, 6, 2)),
        (0, (7, 1, 8, 1)),
        (0, (8, 4, 8, 2)),
        (1, (3, 3, 6, 1)),
        (1, (STRIP, 2, 7, 2)),
    ] {
        let ours = flat_draw(&p, &d, bases, mode, based(draw));
        let lit: Vec<_> = ours.iter().filter(|px| **px != [0, 0, 0, 0xff]).collect();
        let start = u8::try_from(draw.1).unwrap();
        assert!(
            !lit.is_empty() && lit.iter().all(|px| px[1] == start && px[0] == 3),
            "{mode} {draw:?}"
        );
        let key = format!("{mode} {draw:?}");
        crate::metal_check_raw!(key, ours);
    }
}

/// A state of `vertex` and `ids_fragment` for a BGRA8 target, of the input primitive
/// topology `class`, with indirect command buffer support where `indirect` says.
fn classed_state(
    d: &crate::gpu::device::VulkanDevice,
    (vertex, indirect): (&str, bool),
    class: PrimitiveTopologyClass,
) -> crate::gpu::resource::VulkanRenderPipeline {
    use crate::core::platform::resource::Library as _;

    let library = d.new_library(VERTEX_IDS).unwrap().unwrap().unwrap();
    let function = |name| library.new_function(OnFault::Die, name).unwrap().unwrap();
    let (vertex, fragment) = (function(vertex), function("ids_fragment"));
    let mut desc = crate::gpu::device::tests::render_value::<VulkanPlatform>(
        (&vertex, &fragment),
        1,
        &[&[(1, 80)]],
    );
    desc.input_primitive_topology = match class {
        PrimitiveTopologyClass::Unspecified => 0,
        PrimitiveTopologyClass::Point => 1,
        PrimitiveTopologyClass::Line => 2,
        PrimitiveTopologyClass::Triangle => 3,
        other => panic!("no test makes a state of {other:?}"),
    };
    desc.support_indirect_command_buffers = indirect;
    d.new_render_pipeline_state_full(&desc).unwrap().unwrap()
}

/// Points die under a state whose vertex function writes no `[[point_size]]` outside the
/// point input primitive topology class, where Metal sizes none, drawn directly, over
/// indices or from indirect arguments, but a draw of none; under one that writes it, or in
/// the point class, they draw as Metal's do, a pixel at each vertex
#[test]
fn points_die_where_metal_sizes_none() {
    let (p, d) = (platform(), dev());
    let draw = "drawPrimitives:vertexStart:vertexCount:";
    let indexed = "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:";
    let unclassed = [
        PrimitiveTopologyClass::Unspecified,
        PrimitiveTopologyClass::Line,
        PrimitiveTopologyClass::Triangle,
    ];
    for class in unclassed {
        let (e, indices) = open_drawing(&classed_state(&d, ("point_vertex", false), class));
        let indirect = RenderCmd::DrawPrimitivesIndirect {
            primitive_type: 0,
            indirect_buffer: &indices,
            indirect_buffer_offset: 0,
        };
        assert_eq!(dies_at(&e, three(0, None)), draw, "{class:?}");
        assert_eq!(dies_at(&e, three(0, Some(&indices))), indexed, "{class:?}");
        assert_eq!(
            dies_at(&e, indirect),
            "drawPrimitives:indirectBuffer:indirectBufferOffset:"
        );
        let none = RenderCmd::DrawPrimitives {
            primitive_type: 0,
            vertex_start: 0,
            vertex_count: 0,
        };
        send_all(&e, vec![none, three(3, None), RenderCmd::EndEncoding]);
    }
    let sized = fixture_state::<VulkanPlatform>(&d, SIZED.0, SIZED.1);
    let ours = pass_pixels(&p, &d, points::<VulkanPlatform>(&sized));
    let lit: Vec<usize> = (0..256).filter(|&i| ours[i] != [0, 0, 0, 0xff]).collect();
    assert_eq!(lit, [4 * 16 + 4, 8 * 16 + 8, 12 * 16 + 12]);
    let classed = classed_state(&d, ("point_vertex", false), PrimitiveTopologyClass::Point);
    assert_eq!(
        pass_pixels(&p, &d, points::<VulkanPlatform>(&classed)),
        ours
    );
    crate::metal_check_raw!("the sized points", ours);
}

/// An indirect command buffer of two commands whose first draws `count` points from vertex 1
/// under `state`, set on the command, or under the encoder's where `state` is `None`.
fn points_icb(
    d: &crate::gpu::device::VulkanDevice,
    state: Option<&crate::gpu::resource::VulkanRenderPipeline>,
    count: u64,
) -> crate::gpu::icb::VulkanIcb {
    use crate::core::platform::icb::{
        IcbRenderCmd, IndirectCommandBuffer as _, IndirectRenderCommand as _,
    };
    use crate::core::platform::values::IcbDesc;

    let icb = d
        .new_indirect_command_buffer(
            &IcbDesc {
                command_types: 1,
                inherit_pipeline_state: state.is_none(),
                ..IcbDesc::default()
            },
            2,
            0,
        )
        .unwrap()
        .unwrap();
    let c = icb.render_command(0).unwrap().unwrap();
    if let Some(state) = state {
        let set = IcbRenderCmd::SetRenderPipelineState {
            pipeline_state: Some(state),
        };
        c.encode(set).unwrap();
    }
    c.encode(IcbRenderCmd::DrawPrimitives {
        primitive_type: 0,
        vertex_start: 1,
        vertex_count: count,
        instance_count: 1,
        base_instance: 0,
    })
    .unwrap();
    icb
}

/// The commands that execute the first command of `icb`, after `state` where one is given.
fn execute_first<'a>(
    icb: &'a crate::gpu::icb::VulkanIcb,
    state: Option<&'a crate::gpu::resource::VulkanRenderPipeline>,
) -> Vec<RenderCmd<'a, VulkanPlatform>> {
    let set = state.map(|state| RenderCmd::SetRenderPipelineState {
        pipeline_state: Some(state),
    });
    set.into_iter()
        .chain([RenderCmd::ExecuteCommandsInBuffer {
            indirect_command_buffer: icb,
            execution_range: NsRange::new(0, 1),
        }])
        .collect()
}

/// The line of the death at the commit of a pass of `cmds`, which must die there.
fn commit_dies(cmds: Vec<RenderCmd<'_, VulkanPlatform>>) -> String {
    let (p, d) = (platform(), dev());
    let (_, pass) = pass_of(&p, &d, (16, 16));
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
    for cmd in cmds.into_iter().chain([RenderCmd::EndEncoding]) {
        e.encode(OnFault::Die, cmd).unwrap();
    }
    *crate::answer::LAST_LINE.lock() = None;
    let site = catch_death(|| cb.commit(OnFault::Return)).unwrap_err();
    assert_eq!(site.file, "commit");
    crate::answer::LAST_LINE.lock().take().unwrap_or_default()
}

/// An execution's draws of points die at the commit where a direct draw's die, its commands'
/// records read as they stand then: under a state of its own that writes no `[[point_size]]`
/// outside the point input primitive topology class, or under such a state of the encoder's
/// that its commands inherit. An execution of a buffer whose records the GPU may have written (a
/// blit's reset of another of its commands) is asked at the runtime's host step instead, as its
/// commands run (`gpu_written_points_die_at_the_host_step`). A draw of no points, and one under a
/// state of the point class or one that writes its point size, own or inherited, draw as the
/// direct draws do, which are Metal's ([`points_die_where_metal_sizes_none`]), the GPU-written
/// buffer's included. A command buffer committed while a command buffer
/// executes the buffer as its commit read it, and before it ends, dies at a blit's reset of
/// the buffer's commands
#[test]
fn executed_points_die_where_metal_sizes_none() {
    use crate::core::platform::event::EventRef;

    let (p, d) = (platform(), dev());
    let unsizing = classed_state(
        &d,
        ("point_vertex", true),
        PrimitiveTopologyClass::Unspecified,
    );
    let classed = classed_state(&d, ("point_vertex", true), PrimitiveTopologyClass::Point);
    let sized = classed_state(
        &d,
        ("sized_vertex", true),
        PrimitiveTopologyClass::Unspecified,
    );
    let undrawn = "command 0 of executeCommandsInBuffer:withRange:'s indirect command buffer, a \
                   draw of points under a vertex function that writes no [[point_size]]";
    let own = points_icb(&d, Some(&unsizing), 3);
    assert!(commit_dies(execute_first(&own, None)).contains(undrawn));
    let inherited = points_icb(&d, None, 3);
    let line = commit_dies(execute_first(&inherited, Some(&unsizing)));
    assert!(line.contains(undrawn), "{line}");
    let none = points_icb(&d, Some(&unsizing), 0);
    let (lit, unlit) = (
        pass_pixels(&p, &d, points::<VulkanPlatform>(&classed)),
        pass_pixels(&p, &d, Vec::new()),
    );
    assert_eq!(pass_pixels(&p, &d, execute_first(&none, None)), unlit);
    for state in [&classed, &sized] {
        let direct = pass_pixels(&p, &d, points::<VulkanPlatform>(state));
        assert_eq!(direct, lit);
        let own = points_icb(&d, Some(state), 3);
        assert_eq!(pass_pixels(&p, &d, execute_first(&own, None)), direct);
        let inherits = execute_first(&inherited, Some(state));
        assert_eq!(pass_pixels(&p, &d, inherits), direct);
    }
    // Another command reset: the GPU may have written the buffer's records.
    let written = points_icb(&d, Some(&classed), 3);
    let reset = BlitCmd::ResetCommandsInBuffer {
        buffer: &written,
        range: NsRange::new(1, 1),
    };
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
    for cmd in [reset, BlitCmd::EndEncoding] {
        blit.encode(OnFault::Die, cmd).unwrap();
    }
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    assert_eq!(pass_pixels(&p, &d, execute_first(&written, None)), lit);
    // A reset while an execution that read the records waits.
    let read = points_icb(&d, Some(&classed), 3);
    let (_, pass) = pass_of(&p, &d, (16, 16));
    let event = d.new_shared_event().unwrap();
    let held = q.command_buffer(OnFault::Die).unwrap().unwrap();
    held.encode_wait(Some(EventRef::Shared(&event)), 1).unwrap();
    let e = held.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
    for cmd in execute_first(&read, None)
        .into_iter()
        .chain([RenderCmd::EndEncoding])
    {
        e.encode(OnFault::Die, cmd).unwrap();
    }
    held.commit(OnFault::Die).unwrap();
    let cb = d
        .new_command_queue()
        .unwrap()
        .command_buffer(OnFault::Die)
        .unwrap()
        .unwrap();
    let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
    let reset = BlitCmd::ResetCommandsInBuffer {
        buffer: &read,
        range: NsRange::new(1, 1),
    };
    let sel = reset.selector();
    let site = catch_death(|| blit.encode(OnFault::Return, reset)).unwrap_err();
    assert_eq!(site.file, sel);
    event.set_signaled_value(1);
    held.wait_until_completed(OnFault::Die).unwrap();
    assert_eq!(held.error(), None);
    // Once it ended, the reset goes on.
    let reset = BlitCmd::ResetCommandsInBuffer {
        buffer: &read,
        range: NsRange::new(1, 1),
    };
    blit.encode(OnFault::Die, reset).unwrap();
}

crate::death_test!(
    /// An execution of a buffer whose records the GPU may have written (a blit's reset of another
    /// of its commands) is asked at the runtime's host step, once the work before it completed:
    /// its draw of points under a state of its own that writes no `[[point_size]]` outside the
    /// point input primitive topology class dies there, at `commit`, as a commit that read the
    /// records dies at it ([`executed_points_die_where_metal_sizes_none`])
    gpu_written_points_die_at_the_host_step,
    |_child| {
        let (p, d) = (platform(), dev());
        let unsizing = classed_state(
            &d,
            ("point_vertex", true),
            PrimitiveTopologyClass::Unspecified,
        );
        let written = points_icb(&d, Some(&unsizing), 3);
        let cb = (d.new_command_queue().unwrap())
            .command_buffer(OnFault::Die)
            .unwrap()
            .unwrap();
        let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
        let reset = BlitCmd::ResetCommandsInBuffer {
            buffer: &written,
            range: NsRange::new(1, 1),
        };
        for cmd in [reset, BlitCmd::EndEncoding] {
            blit.encode(OnFault::Die, cmd).unwrap();
        }
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        pass_pixels(&p, &d, execute_first(&written, None));
    },
    |report| {
        use crate::core::raise::death::{DeathId, DeathKind};
        report.assert_died(
            DeathId::Platform {
                kind: DeathKind::Abort,
                call: "commit",
            }
            .site(),
        );
    }
);

/// A draw of points, and an execution of commands that inherit the encoder's state, under a
/// state of the point class whose vertex function can encode indirect commands run in an encoder
/// that made resident for writing an indirect command buffer numbered past a reservation of 2
/// and 2^23 + 1, which the runtime takes as Metal does, so the buffer's ID, 3, is the runtime's
/// and Metal's alike, as a compute dispatch does (`crate::gpu::encode_compute`); so does an
/// execution whose command's own state can, which its commit reads, a draw under a state that
/// cannot, and one where the buffer made resident for writing is numbered 1
#[test]
fn work_that_can_encode_after_an_icb_made_writable_runs() {
    use crate::core::platform::icb::{
        IcbRenderCmd, IndirectCommandBuffer as _, IndirectRenderCommand as _,
    };
    use crate::core::platform::values::IcbDesc;

    let (p, d) = (platform(), dev());
    d.reserve_resource_indices(5, &[2, (1 << 23) + 1]).unwrap();
    let make = |inherit_pipeline_state| {
        d.new_indirect_command_buffer(
            &IcbDesc {
                command_types: 1,
                inherit_pipeline_state,
                ..IcbDesc::default()
            },
            1,
            0,
        )
        .unwrap()
        .unwrap()
    };
    let (reaching, written) = (make(false), make(false));
    assert_eq!(
        (reaching.gpu_resource_id(), written.gpu_resource_id()),
        (1, 3)
    );
    let encoding = classed_state(&d, ("encoding_vertex", true), PrimitiveTopologyClass::Point);
    let other = classed_state(&d, ("point_vertex", true), PrimitiveTopologyClass::Point);
    let drawn = || RenderCmd::DrawPrimitives {
        primitive_type: 0,
        vertex_start: 1,
        vertex_count: 3,
    };
    let points = |state: Option<&crate::gpu::resource::VulkanRenderPipeline>| {
        let icb = make(state.is_none());
        let c = icb.render_command(0).unwrap().unwrap();
        let set = state.map(|state| IcbRenderCmd::SetRenderPipelineState {
            pipeline_state: Some(state),
        });
        let draw = IcbRenderCmd::DrawPrimitives {
            primitive_type: 0,
            vertex_start: 1,
            vertex_count: 3,
            instance_count: 1,
            base_instance: 0,
        };
        for cmd in set.into_iter().chain([draw]) {
            c.encode(cmd).unwrap();
        }
        icb
    };
    let (own, inheriting) = (points(Some(&encoding)), points(None));
    let held = buffer::<VulkanPlatform>(&d, 16, 0, &[0; 16]);
    let (the_written, the_reaching) = (
        [Some(ResourceRef::Icb(&written))],
        [Some(ResourceRef::Icb(&reaching))],
    );
    let opening = |state, resident: &'static str| {
        let list = if resident == "written" {
            &the_written
        } else {
            &the_reaching
        };
        vec![
            RenderCmd::UseResources {
                resources: list,
                usage: 3,
            },
            RenderCmd::SetVertexBuffer {
                buffer: Some(&held),
                offset: 0,
                index: 1,
            },
            RenderCmd::SetRenderPipelineState {
                pipeline_state: Some(state),
            },
        ]
    };
    let execution = |icb| RenderCmd::ExecuteCommandsInBuffer {
        indirect_command_buffer: icb,
        execution_range: NsRange::new(0, 1),
    };
    let mut inherited = opening(&encoding, "written");
    inherited.push(execution(&inheriting));
    let mut executed = opening(&other, "written");
    executed.push(execution(&own));
    for cmds in [inherited, executed] {
        pass_pixels(&p, &d, cmds);
    }
    let unlit = pass_pixels(&p, &d, Vec::new());
    for (state, resident) in [
        (&other, "written"),
        (&encoding, "written"),
        (&encoding, "reaching"),
    ] {
        let mut cmds = opening(state, resident);
        cmds.push(drawn());
        assert_ne!(pass_pixels(&p, &d, cmds), unlit);
    }
}

/// The vertices [`vertex_runs`] counts.
const RUNS: usize = 1024;

/// The runs of `counting_vertex` at each of the first [`RUNS`] vertex IDs, after `draw` as
/// [`flat_pass`] draws it, at `setProvokingVertexMode:` of `mode`.
fn vertex_runs<P: Platform>(
    p: &P,
    d: &P::Device,
    mode: u64,
    (primitive_type, start, count, instances): (u64, u64, u64, u64),
) -> Vec<u32> {
    let draw = Provoked::Vertices(primitive_type, start, count, instances);
    provoked_runs(p, d, (mode, None), draw)
}

/// The runs of `counting_vertex` at each of the first [`RUNS`] vertex IDs after `draw`, a draw
/// of [`provoked`]'s but an execution, under a state of `counting_vertex` and `ids_fragment`
/// over [`flat_vertices`] in buffer 0 and the runs in buffer 1, at `setProvokingVertexMode:` of
/// `mode`, after `restart`'s `setPrimitiveRestartEnabled:index:` where it has one.
fn provoked_runs<P: Platform>(
    p: &P,
    d: &P::Device,
    (mode, restart): (u64, Option<(bool, u64)>),
    draw: Provoked<'_>,
) -> Vec<u32> {
    let state = fixture_state::<P>(d, "vertex_ids:counting_vertex", SIZED.1);
    let (pos, _) = flat_vertices::<P>(d, RUNS);
    let runs = buffer::<P>(d, 4 * RUNS as u64, 0, &[0; 4 * RUNS]);
    let (index_buffer, indirect_buffer) = provoked_buffers::<P>(d, draw);
    let mut cmds = vec![
        RenderCmd::SetRenderPipelineState {
            pipeline_state: Some(&state),
        },
        RenderCmd::SetVertexBuffer {
            buffer: Some(&pos),
            offset: 0,
            index: 0,
        },
        RenderCmd::SetVertexBuffer {
            buffer: Some(&runs),
            offset: 0,
            index: 1,
        },
    ];
    if let Some((enabled, index)) = restart {
        cmds.push(RenderCmd::SetPrimitiveRestartEnabled { enabled, index });
    }
    cmds.push(RenderCmd::SetProvokingVertexMode { mode });
    cmds.extend(provoked_draw(draw, (&index_buffer, &indirect_buffer)));
    pass_pixels(p, d, cmds);
    // SAFETY: a Shared buffer of `4 * RUNS` bytes, read after the pass completed.
    let bytes = unsafe { std::slice::from_raw_parts(runs.contents().as_ptr::<u8>(), 4 * RUNS) };
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes(*w))
        .collect()
}

/// A draw the runtime draws over a list of its own (a private type's, or a strip's at the last
/// vertex without the device's) runs its vertex function as many times at each vertex as Metal's
/// draw, as a draw as it stands does: once for each primitive the vertex is in, on both
#[test]
fn re_indexed_draws_run_the_vertex_function_as_often_as_metals() {
    let (p, d) = (platform(), dev());
    let at = |start: usize, runs: &[u32]| -> Vec<u32> {
        let mut out = vec![0; RUNS];
        out[start..start + runs.len()].copy_from_slice(runs);
        out
    };
    for (mode, draw, expected) in [
        (0, (3, 1, 6, 1), at(1, &[1; 6])),
        (0, (STRIP, 2, 7, 1), at(2, &[1, 2, 3, 3, 3, 2, 1])),
        (0, (2, 3, 4, 1), at(3, &[1, 2, 2, 1])),
        (0, (5, 3, 5, 1), at(3, &[2; 5])),
        (0, (6, 2, 6, 1), at(2, &[4, 1, 2, 2, 2, 1])),
        (0, (7, 1, 8, 1), at(1, &[1, 2, 1, 2, 1, 2, 1, 2])),
        (0, (8, 4, 8, 1), at(4, &[2, 1, 3, 3, 3, 3, 1, 2])),
        (1, (3, 3, 6, 1), at(3, &[1; 6])),
        (1, (STRIP, 2, 7, 1), at(2, &[1, 2, 3, 3, 3, 2, 1])),
    ] {
        assert_eq!(vertex_runs(&p, &d, mode, draw), expected, "{mode} {draw:?}");
    }
    for (mode, draw) in [
        (0, (6, 0, 1000, 3)),
        (0, (7, 0, 1000, 2)),
        (0, (8, 0, 1000, 2)),
        (0, (5, 0, 1000, 2)),
        (1, (STRIP, 0, 1000, 2)),
        (1, (3, 0, 999, 2)),
    ] {
        let ours = vertex_runs(&p, &d, mode, draw);
        assert!(ours.iter().any(|&n| n > 1), "{mode} {draw:?}");
        let key = format!("{mode} {draw:?}");
        crate::metal_check_raw!(key, ours);
    }
}

/// Metal's private types run a vertex function that writes memory as many times at each vertex
/// as Metal's draws, once for each vertex of the list Metal draws them as, over 16- and 32-bit
/// indices, from a base vertex, of instances, after `setPrimitiveRestartEnabled:index:`, a quad
/// strip restarting at its index type's all-ones index, and from indirect arguments, over indices
/// too, at `setProvokingVertexMode:` 0 and 1 (the line loop's lines at 1 relayed,
/// `provoking-vertex-lines`, where the device lacks the last vertex). Each draw runs some vertex
/// more than once
#[test]
fn private_types_run_the_vertex_function_as_often_as_metals_at_every_draw() {
    let (p, d) = (platform(), dev());
    let indirect = |primitive_type, index_type, indices, arguments| Provoked::IndexedIndirect {
        primitive_type,
        index_type,
        indices,
        arguments,
    };
    for t in 5..=8 {
        let mut cases = vec![
            (over((t, U16), SCRAMBLED, (1, 2)), None),
            (over((t, U32), SCRAMBLED, (0, 1)), None),
            (over((t, U16), SCRAMBLED_AT_3, (0, 1)), Some((true, 3))),
            (Provoked::Indirect(t, [10, 1, 2, 0]), None),
            (Provoked::Indirect(t, [9, 2, 0, 0]), None),
            (indirect(t, U16, SCRAMBLED, [12, 1, 0, 2, 0]), None),
            (indirect(t, U32, SCRAMBLED, [8, 2, 3, 0, 0]), None),
        ];
        if t == 8 {
            cases.push((over((t, U16), QUADS_RESTARTED, (0, 1)), None));
            cases.push((over((t, U32), QUADS_RESTARTED_WIDE, (0, 1)), None));
        }
        for (draw, restart) in cases {
            for mode in [0, 1] {
                let ours = crate::answer::relaxed(|| provoked_runs(&p, &d, (mode, restart), draw));
                assert!(ours.iter().any(|&n| n > 1), "{draw:?} at mode {mode}");
                let key = format!("{draw:?} at mode {mode}");
                crate::metal_check_raw!(key, ours);
            }
        }
    }
}

/// [`ROUNDING`]'s functions compiled with fast math off (`-fno-fast-math`).
const ROUNDING_SAFE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rounding_safe.metallib"));

/// How [`swept`] draws: the fragment function, `sweep_fragment` (a `float4`),
/// `sweep_half_fragment` (a `half4`) or `sweep_four_fragment` (a `float4` at colours 0 to 3),
/// the colour attachment's entries past its index and pixel format (key 2 blending, 3 to 8
/// its factors and operations, 9 its write mask: [`colours_of`]), the swept attachment's
/// index (past 0, colour 0 is an `RGBA8Unorm` attachment of Metal's defaults, cleared), and its
/// load
/// action (`MTLLoadAction`: 2 clears it, 1 loads what the new texture holds).
#[derive(Clone, Copy, Debug)]
struct Drawn<'a> {
    fragment: &'a str,
    colour: &'a [(u8, u32)],
    index: u32,
    load: u64,
}

/// `sweep_fragment` into an attachment of Metal's defaults: no blend, every channel.
const PLAIN: Drawn<'static> = Drawn {
    fragment: "sweep_fragment",
    colour: &[],
    index: 0,
    load: CLEAR,
};

/// The bytes of a 256 by 256 texture of `format`, `bytes` a texel, after the sweep `kind`
/// (`rounding.metal`) is written into it: by the kernel where `kernel`, else drawn into it as the
/// colour attachment `drawn` says, as it says.
fn swept<P: Platform>(
    p: &P,
    d: &P::Device,
    kernel: bool,
    (format, bytes): (u64, u64),
    kind: u32,
    drawn: Drawn<'_>,
) -> Vec<u8> {
    use crate::core::platform::compute::{ComputeCmd, ComputeEncoder as _};
    use crate::core::platform::resource::Library as _;
    use crate::core::platform::types::NsRange;

    use crate::core::platform::values::FullComputePipelineDesc;

    use crate::gpu::device::tests::render_value;

    let library = d.new_library(ROUNDING).unwrap().unwrap().unwrap();
    let function = |name| library.new_function(OnFault::Die, name).unwrap().unwrap();
    let functions = [
        function("sweep_vertex"),
        function(drawn.fragment),
        function("sweep_kernel"),
    ];
    let t = texture(p, d, format, READ | 2 | TARGET, (256, 256), None);
    let kinds = buffer::<P>(d, 4, 0, &kind.to_le_bytes());
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    if kernel {
        let state = d
            .new_compute_pipeline_state_full(&FullComputePipelineDesc::new(&functions[2]))
            .unwrap()
            .unwrap();
        let e = cb.compute_encoder(OnFault::Die).unwrap().unwrap();
        let buffers = [Some(&kinds)];
        for cmd in [
            ComputeCmd::SetComputePipelineState {
                state: Some(&state),
            },
            ComputeCmd::SetTexture {
                texture: Some(&t),
                index: 0,
            },
            ComputeCmd::SetBuffers {
                buffers: &buffers,
                offsets: &[0],
                range: NsRange::new(0, 1),
            },
            ComputeCmd::DispatchThreads {
                threads_per_grid: MtlSize::new(256, 256, 1),
                threads_per_threadgroup: MtlSize::new(16, 16, 1),
            },
            ComputeCmd::EndEncoding,
        ] {
            e.encode(OnFault::Die, cmd).unwrap();
        }
    } else {
        let format = u32::try_from(format).unwrap();
        let mut colour = vec![(0, drawn.index), (1, format)];
        colour.extend(drawn.colour);
        let first: &[(u8, u32)] = &[(0, 0), (1, 70)];
        let colours: Vec<&[(u8, u32)]> = match drawn.index {
            0 => vec![&colour],
            _ => vec![first, &colour],
        };
        let desc = render_value::<P>((&functions[0], &functions[1]), 1, &colours);
        let state = d.new_render_pipeline_state_full(&desc).unwrap().unwrap();
        let pass = p.render_pass_descriptor();
        let attachment = ColorAttachmentSet {
            texture: Some(Some(&t)),
            load_action: Some(drawn.load),
            store_action: Some(STORE),
            ..ColorAttachmentSet::default()
        };
        pass.set_color_attachment(OnFault::Die, u64::from(drawn.index), &attachment)
            .unwrap();
        let target = texture(p, d, 70, READ | TARGET, (256, 256), None);
        if drawn.index != 0 {
            let first = ColorAttachmentSet {
                texture: Some(Some(&target)),
                load_action: Some(CLEAR),
                store_action: Some(STORE),
                ..ColorAttachmentSet::default()
            };
            pass.set_color_attachment(OnFault::Die, 0, &first).unwrap();
        }
        let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
        for cmd in [
            RenderCmd::SetRenderPipelineState {
                pipeline_state: Some(&state),
            },
            RenderCmd::SetFragmentBuffer {
                buffer: Some(&kinds),
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
    }
    let length = 256 * 256 * bytes;
    let out = buffer::<P>(d, length, 0, &[]);
    let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
    blit.encode(
        OnFault::Die,
        BlitCmd::CopyTextureToBuffer {
            source_texture: &t,
            source_slice: 0,
            source_level: 0,
            source_origin: MtlOrigin::new(0, 0, 0),
            source_size: MtlSize::new(256, 256, 1),
            destination_buffer: &out,
            destination_offset: 0,
            destination_bytes_per_row: 256 * bytes,
            destination_bytes_per_image: length,
        },
    )
    .unwrap();
    blit.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    assert!(cb.error().is_none(), "{:?}", cb.error());
    let n = usize::try_from(length).unwrap();
    // SAFETY: a Shared buffer's bytes, read after the blit completed.
    unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), n) }.to_vec()
}

/// Texture writes are Metal's, byte for byte, over sweeps through unorm values, past 1 by
/// float steps and past the largest half: a kernel's, and a fragment's into an unblended
/// colour attachment
#[test]
fn texture_writes_round_as_metals() {
    let (p, d) = (platform(), dev());
    let half = [(25, 2), (65, 4), (115, 8)];
    let rest = [(70, 4), (12, 1), (20, 2), (90, 4), (92, 4)];
    let srgb = (81, 4);
    for kind in [0, 1, 3] {
        for format in half.into_iter().chain(rest) {
            let key = format!("kernel {format:?} sweep {kind}");
            let ours = swept(&p, &d, true, format, kind, PLAIN);
            assert!(crate::metal_matches_bytes!(key, &ours), "{key}");
        }
        for format in half.into_iter().chain(rest).chain([srgb]) {
            let key = format!("render {format:?} sweep {kind}");
            let ours = swept(&p, &d, false, format, kind, PLAIN);
            assert!(crate::metal_matches_bytes!(key, &ours), "{key}");
        }
    }
}

/// A render state whose fragment function reads its extended-range colour 0 by framebuffer
/// fetch, blended there, is made where the runtime blends it in the shader, as Metal makes it:
/// into `RGB10A8_2P_XR10`, a `half4` output under source factor zero and destination factor
/// one minus the source's alpha (a guest's WindowServer's), and under source factor the
/// source's alpha. The function computes each output from the fetched colour under fast
/// math, which the runtime notes `extended-range-blend-source` and
/// announces at each draw that binds the state's variant: a draw of either over an attachment
/// a pass clears dies there, naming the kind ([`crate::answer::DIES`]), where Metal draws it.
/// A `float4` output so into `RGB10A8_2P_XR10` under that
/// first blend and under source and destination factor one is made, and the runtime notes it
/// `extended-range-float-blend-source`, which it announces at each draw that binds
/// the state's variant: a draw of it over an attachment a pass clears dies there, naming the
/// kind ([`crate::answer::DIES`]), where Metal draws it. So does one into
/// `RGB10A8_2P_XR10_sRGB` under source and destination factor one, whose blend the runtime
/// computes in the `_sRGB` form as in the linear one, which Metal's lanes measured equal
#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "MoltenVK: the runtime records device-pack-conversion, a death in strict mode; Mac-only test, off everywhere; ledger: MoltenVK strict deaths"
)]
fn extended_range_fetches_are_made_where_the_runtime_blends_them() {
    use crate::core::platform::resource::Library as _;
    use crate::core::raise::death::DeathKind;

    use crate::gpu::device::tests::render_value;

    fn state<P: Platform>(
        d: &P::Device,
        (fragment, format): (&str, u32),
        (source, destination): (u32, u32),
    ) -> Option<P::RenderPipeline> {
        let library = d.new_library(ROUNDING).unwrap().unwrap().unwrap();
        let function = |name| library.new_function(OnFault::Die, name).unwrap().unwrap();
        let functions = [function("sweep_vertex"), function(fragment)];
        let colour: &[(u8, u32)] = &[
            (0, 0),
            (1, format),
            (2, 1),
            (3, source),
            (4, destination),
            (5, 0),
            (6, source),
            (7, destination),
            (8, 0),
        ];
        let desc = render_value::<P>((&functions[0], &functions[1]), 1, &[colour]);
        d.new_render_pipeline_state_full(&desc).unwrap()
    }
    /// A draw of `state` over a Private 64 by 64 attachment of `format` that its pass clears.
    fn drawn<P: Platform>(p: &P, d: &P::Device, state: &P::RenderPipeline, format: u32) {
        let t = texture(p, d, u64::from(format), READ | TARGET, (64, 64), None);
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let pass = p.render_pass_descriptor();
        let attachment = ColorAttachmentSet {
            texture: Some(Some(&t)),
            load_action: Some(CLEAR),
            store_action: Some(STORE),
            ..ColorAttachmentSet::default()
        };
        pass.set_color_attachment(OnFault::Die, 0, &attachment)
            .unwrap();
        let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
        for cmd in [
            RenderCmd::SetRenderPipelineState {
                pipeline_state: Some(state),
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
    }
    const DRAW: &str = "drawPrimitives:vertexStart:vertexCount:";
    let (p, d) = (platform(), dev());
    // Each death, the selector it dies at, and what its line names.
    let half_source = "(metal2vulkan-degrade-extended-range-blend-source)";
    let float_source = "extended-range-float-blend-source";
    let dies = [
        ((("fetch_half_fragment", 550), (0, 5)), DRAW, half_source),
        ((("fetch_half_fragment", 550), (4, 5)), DRAW, half_source),
        ((("fetch_float_fragment", 550), (0, 5)), DRAW, float_source),
        ((("fetch_float_fragment", 550), (1, 1)), DRAW, float_source),
        ((("fetch_float_fragment", 551), (1, 1)), DRAW, float_source),
    ];
    for ((fragment, blend), sel, named) in dies {
        *crate::answer::LAST_LINE.lock() = None;
        let site = catch_death(|| {
            let state = state::<VulkanPlatform>(&d, fragment, blend).unwrap();
            drawn(&p, &d, &state, fragment.1);
        })
        .unwrap_err();
        let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
        assert_eq!(
            (site.kind, site.file),
            (DeathKind::Abort, sel),
            "{fragment:?} {blend:?}: {line}"
        );
        assert!(line.contains(named), "{line}");
    }
}

/// The blends the extended-range tests draw under, of every factor kind and operation: each the
/// colour's source factor, destination factor and operation, then alpha's (`MTLBlendFactor`,
/// `MTLBlendOperation`).
const XR_BLENDS: [[u32; 6]; 22] = [
    [1, 1, 0, 1, 1, 0],
    [0, 5, 0, 0, 5, 0],
    [1, 5, 0, 1, 5, 0],
    [4, 5, 0, 1, 5, 0],
    [4, 5, 0, 4, 5, 0],
    [6, 0, 0, 6, 0, 0],
    [6, 5, 0, 1, 5, 0],
    [3, 8, 0, 1, 1, 0],
    [3, 6, 0, 1, 1, 0],
    [3, 5, 0, 1, 5, 0],
    [6, 2, 0, 1, 1, 0],
    [1, 1, 1, 1, 1, 1],
    [1, 1, 2, 1, 1, 2],
    [1, 1, 3, 1, 1, 3],
    [1, 1, 4, 1, 1, 4],
    [7, 1, 0, 1, 1, 0],
    [9, 1, 0, 9, 1, 0],
    [2, 0, 0, 2, 0, 0],
    [10, 1, 0, 1, 1, 0],
    [8, 9, 0, 8, 9, 0],
    [11, 12, 0, 13, 14, 0],
    [5, 4, 0, 5, 4, 0],
];

/// The linear extended-range formats: `RGB10A8_2P_XR10`, `BGRA10_XR` and `BGR10_XR`.
const XR_LINEAR: [u32; 3] = [550, 552, 554];

/// Their `_sRGB` twins: `RGB10A8_2P_XR10_sRGB`, `BGRA10_XR_sRGB` and `BGR10_XR_sRGB`.
const XR_SRGB: [u32; 3] = [551, 553, 555];

/// The blends the runtime models into an `_sRGB` twin, as [`XR_BLENDS`]: added
/// colour and alpha, each of source factor zero or one and destination factor zero, one or one
/// minus the source's alpha.
const XR_SRGB_BLENDS: [[u32; 6]; 8] = [
    [0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 1, 0],
    [0, 5, 0, 0, 5, 0],
    [1, 0, 0, 1, 0, 0],
    [1, 1, 0, 1, 1, 0],
    [1, 5, 0, 1, 5, 0],
    [1, 5, 0, 1, 1, 0],
    [0, 5, 0, 1, 5, 0],
];

/// How many of the 4096 texels `XrSide::blended` read of `ours` differ from `theirs`, and the
/// first, both sides; `None` where none does.
fn xr_wrong(ours: &[u8], theirs: &[u8]) -> Option<String> {
    let wrong = (ours.chunks(16).zip(theirs.chunks(16)).enumerate()).filter(|(_, (o, t))| o != t);
    let count = wrong.clone().count();
    let (i, (o, t)) = wrong.into_iter().next()?;
    Some(format!(
        "{count} of 4096 texels differ, the first ({}, {}): ours {:?}, Metal {:?}",
        i % 64,
        i / 64,
        xr_lanes(o),
        xr_lanes(t)
    ))
}

/// The values `xr_loaded_fragment` loads: 4096 `float4`s a draw, of seeds 1 and 2, as
/// `xr_float_fragment` draws them (colour lanes in [-0.9375, 1.5) and alpha in
/// [-0.125, 1.125), each a multiple of 2^-11), from a fixed sequence.
fn xr_values() -> Vec<u8> {
    let mut x = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = |span: u64, low: i64| {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        let code = i64::try_from(x % span).unwrap() - low;
        code as f32 / 2048.0
    };
    (0..2 * 4096)
        .flat_map(|_| {
            [
                next(4992, 1920),
                next(4992, 1920),
                next(4992, 1920),
                next(2560, 256),
            ]
        })
        .flat_map(f32::to_le_bytes)
        .collect()
}

/// One platform's side of the extended-range comparisons: the platform, its device, and the
/// one command queue a test makes of it for all its configurations ([`xr_sides`]), for the
/// reason [`test_queue`](crate::gpu::device::tests::metal_tests::test_queue) gives: a Metal
/// queue made per configuration lives to the test's end, and past 512 of them Metal's passes
/// stored nothing.
struct XrSide<P: Platform> {
    p: P,
    d: P::Device,
    q: P::CommandQueue,
}

impl<P: Platform> XrSide<P> {
    /// The side of `p`'s device `d`, with a queue of its own.
    fn new(p: P, d: P::Device) -> XrSide<P> {
        let q = d.new_command_queue().unwrap();
        XrSide { p, d, q }
    }

    /// The floats `xr_read` reads of a Private 64 by 64 attachment of `format` after a draw
    /// of `fragment` (`xr_float_fragment`, `xr_half_fragment` or `xr_loaded_fragment`, buffer
    /// 1 holding [`xr_values`]) of `library` ([`ROUNDING`] or [`ROUNDING_SAFE`]) of seed 1
    /// into it unblended, in a pass that clears it, then, under `blend` (the colour
    /// attachment's keys 3 to 8: the colour's source and destination factors and operation,
    /// then alpha's), one of seed 2 blended: in a pass of its own that loads what the first
    /// stored where `across`, else in the first's pass. With no `blend`, the first draw
    /// alone.
    fn blended(
        &self,
        (library, fragment): (&[u8], &str),
        format: u32,
        blend: Option<[u32; 6]>,
        across: bool,
    ) -> Vec<u8> {
        self.masked((library, fragment), format, (blend, None), across)
    }

    /// [`XrSide::blended`], the blended draw's state of the colour write mask `mask` (key 9,
    /// `MTLColorWriteMask`: red 8, green 4, blue 2, alpha 1) where one is given, the
    /// unblended one's of every channel ([`xr_blended_masked`]).
    fn masked(
        &self,
        drawn: (&[u8], &str),
        format: u32,
        shape: (Option<[u32; 6]>, Option<u32>),
        across: bool,
    ) -> Vec<u8> {
        xr_blended_masked(&self.p, &self.d, &self.q, drawn, format, shape, across)
    }
}

/// The answer of `newRenderPipelineStateWithDescriptor:options:reflection:error:` on `d` for a
/// state of `sweep_vertex` and `fragment` of `library` into colour 0 of `format`, under `blend`
/// ([`XR_DUAL_BLENDS`]'s layout) or unblended: `Ok` where it is made, else the `NSError`'s code.
fn blended_state<P: Platform>(
    d: &P::Device,
    (library, fragment): (&[u8], &str),
    format: u32,
    blend: Option<[u32; 6]>,
) -> Result<(), i64> {
    use crate::core::platform::resource::Library as _;

    use crate::gpu::device::tests::render_value;

    let library = d.new_library(library).unwrap().unwrap().unwrap();
    let function = |name| library.new_function(OnFault::Die, name).unwrap().unwrap();
    let functions = [function("sweep_vertex"), function(fragment)];
    let mut colour = vec![(0, 0), (1, format)];
    if let Some(blend) = blend {
        colour.push((2, 1));
        colour.extend((3..).zip(blend));
    }
    let desc = render_value::<P>((&functions[0], &functions[1]), 1, &[&colour]);
    match d.new_render_pipeline_state_full(&desc) {
        Ok(Some(_)) => Ok(()),
        Ok(None) => panic!("{fragment} {format}: nil with no error"),
        Err(e) => Err(e.code),
    }
}

/// A render state blended by a factor of the second source whose fragment function writes no
/// second output (`xr_half_fragment`, `xr_float_fragment`) is answered as Metal answers it, nil
/// and an `NSError` of Metal's code, 3, where Metal refuses it, or made where Metal makes it:
/// into the extended-range forms and twins, `RGBA8Unorm`, `BGRA8Unorm` and `RGBA16Float`,
/// compiled with fast math and without, under each blend of [`XR_DUAL_BLENDS`], each of which
/// reads the second source. The runtime refuses such a state with Metal's error
/// (`InvalidDescriptor(ColorAttachment)`), which the creator answers
/// ([`crate::gpu::states::pipeline_error`]), where this platform died at the runtime's refusal
/// of an extended-range one. A function that writes a second output (`xr_dual_half_fragment`) is
/// made as on Metal, compiled with fast math or without, into every one of those formats. And a
/// state of a `half4` output into `RGBA8Uint`, unblended, whose output the
/// runtime refuses for its type class (`InvalidDescriptor(ColorAttachment)` too), is answered as
/// Metal answers it
#[test]
fn second_source_blends_without_a_second_output_are_refused_as_metals() {
    let d = dev();
    let mut differ = Vec::new();
    let formats = [552, 554, 553, 555, 70, 80, 115];
    let fragments = [
        "xr_half_fragment",
        "xr_float_fragment",
        "xr_dual_half_fragment",
    ];
    // Each state: its library, fragment function, format and blend.
    type Case = (&'static [u8], &'static str, u32, Option<[u32; 6]>);
    let mut cases: Vec<Case> = Vec::new();
    for format in formats {
        for library in [ROUNDING, ROUNDING_SAFE] {
            for fragment in fragments {
                cases.extend(XR_DUAL_BLENDS.map(|blend| (library, fragment, format, Some(blend))));
            }
        }
    }
    cases.push((ROUNDING_SAFE, "xr_half_fragment", 73, None));
    for (library, fragment, format, blend) in cases {
        let fast = if library == ROUNDING { "fast" } else { "safe" };
        let what = format!("{fragment} ({fast}) format {format} blend {blend:?}");
        *crate::answer::LAST_LINE.lock() = None;
        let ours =
            catch_death(|| blended_state::<VulkanPlatform>(&d, (library, fragment), format, blend));
        let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
        match ours {
            Ok(ours) if crate::metal_matches!(what, ours) => {}
            Ok(ours) => differ.push(format!("{what}: ours {ours:?}, not Metal's")),
            Err(site) => differ.push(format!("{what}: died at {}: {line}", site.file)),
        }
    }
    assert!(differ.is_empty(), "{} differ: {differ:#?}", differ.len());
}

/// This platform's side: one command queue for a test ([`XrSide`]).
fn xr_sides() -> XrSides {
    XrSide::new(platform(), dev())
}

/// The four floats of each texel of `bytes`, as `xr_read` writes them.
fn xr_lanes(bytes: &[u8]) -> Vec<f32> {
    (bytes.as_chunks::<4>().0.iter())
        .map(|&l| f32::from_le_bytes(l))
        .collect()
}

/// [`XrSide::masked`] on `p`'s device `d`, its command buffer of `q`, in an autorelease
/// pool of its own: what Metal autoreleased (each command buffer and encoder, which a dev
/// build cannot elide) goes at the configuration's end.
fn xr_blended_masked<P: Platform>(
    p: &P,
    d: &P::Device,
    q: &P::CommandQueue,
    (library, fragment): (&[u8], &str),
    format: u32,
    (blend, mask): (Option<[u32; 6]>, Option<u32>),
    across: bool,
) -> Vec<u8> {
    use crate::core::platform::render::{RenderCmd, RenderEncoder as _};
    use crate::core::platform::resource::Library as _;
    use crate::core::rt::Pool;

    use crate::gpu::device::tests::render_value;

    let _pool = Pool::here();
    let library = d.new_library(library).unwrap().unwrap().unwrap();
    let function = |name| library.new_function(OnFault::Die, name).unwrap().unwrap();
    let functions = [function("sweep_vertex"), function(fragment)];
    let state = |blend: Option<[u32; 6]>| {
        let mut colour = vec![(0, 0), (1, format)];
        if let Some(blend) = blend {
            colour.push((2, 1));
            colour.extend((3..).zip(blend));
            colour.extend(mask.map(|mask| (9, mask)));
        }
        let desc = render_value::<P>((&functions[0], &functions[1]), 1, &[&colour]);
        d.new_render_pipeline_state_full(&desc).unwrap().unwrap()
    };
    let states: Vec<_> = std::iter::once(None)
        .chain(blend.map(Some))
        .map(state)
        .collect();
    let t = d
        .new_texture(OnFault::Die, &xr_attachment(p, format))
        .unwrap()
        .unwrap();
    let seeds = [1u32, 2].map(|seed| buffer::<P>(d, 4, 0, &seed.to_le_bytes()));
    let values = xr_values();
    let values = buffer::<P>(d, u64::try_from(values.len()).unwrap(), 0, &values);
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    // `MTLLoadActionLoad`.
    const LOAD: u64 = 1;
    let draws: Vec<_> = states.iter().zip(&seeds).collect();
    let passes = match across {
        true => draws.chunks(1).collect::<Vec<_>>(),
        false => vec![&draws[..]],
    };
    for (n, draws) in passes.into_iter().enumerate() {
        let pass = p.render_pass_descriptor();
        let attachment = ColorAttachmentSet {
            texture: Some(Some(&t)),
            load_action: Some(if n == 0 { CLEAR } else { LOAD }),
            store_action: Some(STORE),
            ..ColorAttachmentSet::default()
        };
        pass.set_color_attachment(OnFault::Die, 0, &attachment)
            .unwrap();
        let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
        for &(state, seed) in draws {
            for cmd in [
                RenderCmd::SetRenderPipelineState {
                    pipeline_state: Some(state),
                },
                RenderCmd::SetFragmentBuffer {
                    buffer: Some(seed),
                    offset: 0,
                    index: 0,
                },
                RenderCmd::SetFragmentBuffer {
                    buffer: Some(&values),
                    offset: 0,
                    index: 1,
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
    xr_read_back::<P>(d, cb, &library, &t)
}

/// The descriptor of the Private 64 by 64 attachment of `format` the extended-range
/// comparisons draw into and read.
fn xr_attachment<P: Platform>(p: &P, format: u32) -> P::TextureDescriptor {
    p.texture_descriptor(&TextureDesc {
        pixel_format: u64::from(format),
        width: 64,
        height: 64,
        usage: READ | TARGET,
        resource_options: Some(0x20),
        ..TextureDesc::default()
    })
}

/// Ends `cb` with `xr_read` of `library` reading `t` (64 by 64) into a Shared buffer of `d`,
/// a float4 a texel, commits it, waits for it, and answers the buffer's bytes: no error.
fn xr_read_back<P: Platform>(
    d: &P::Device,
    cb: P::CommandBuffer,
    library: &P::Library,
    t: &P::Texture,
) -> Vec<u8> {
    use crate::core::platform::compute::{ComputeCmd, ComputeEncoder as _};
    use crate::core::platform::resource::Library as _;

    let read = library
        .new_function(OnFault::Die, "xr_read")
        .unwrap()
        .unwrap();
    let read = d
        .new_compute_pipeline_state_for_function(Some(&read))
        .unwrap()
        .unwrap();
    let length = 64 * 64 * 16;
    let out = buffer::<P>(d, length, 0, &[]);
    let c = cb.compute_encoder(OnFault::Die).unwrap().unwrap();
    let outs = [Some(&out)];
    for cmd in [
        ComputeCmd::SetComputePipelineState { state: Some(&read) },
        ComputeCmd::SetTexture {
            texture: Some(t),
            index: 0,
        },
        ComputeCmd::SetBuffers {
            buffers: &outs,
            offsets: &[0],
            range: NsRange::new(0, 1),
        },
        ComputeCmd::DispatchThreads {
            threads_per_grid: MtlSize::new(64, 64, 1),
            threads_per_threadgroup: MtlSize::new(8, 8, 1),
        },
        ComputeCmd::EndEncoding,
    ] {
        c.encode(OnFault::Die, cmd).unwrap();
    }
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    assert!(cb.error().is_none(), "{:?}", cb.error());
    let n = usize::try_from(length).unwrap();
    // SAFETY: a Shared buffer's bytes, read after the kernel completed.
    unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), n) }.to_vec()
}

/// A blend of a pinned gap ([`xr_across`]): an attachment's format, the blend (as
/// [`XR_BLENDS`]), and how many of its 4096 texels differ from Metal's.
type XrGap = (u32, [u32; 6], usize);

/// What [`xr_across`] holds the runtime's blends across passes to.
#[derive(Clone, Copy)]
enum XrExpect<'a> {
    /// Metal's texels, but at a pinned gap's blends, each of which differs at exactly its
    /// count of texels.
    Exact(&'a [XrGap]),
    /// The blends whose texels differ from Metal's are returned, and where the list is
    /// given, only its blends may differ, at any count: the runtime announces
    /// `extended-range-blend-source` for the sweep's states, whose draws die there,
    /// which the caller's test requires.
    Announced(Option<&'a [XrGap]>),
}

/// The relayed kinds a test of [`xr_across`] and [`xr_within`] checks. The runtime announces
/// each once per device (`crate::platform::device_options`), so a test checks what the
/// process heard: a kind unheard after the sweeps that must not announce it, heard after
/// those that must.
fn heard(kind: rt::Degradation) -> bool {
    crate::answer::RELAYED.lock().contains(&kind)
}

/// This platform's side of an extended-range comparison with Metal's answers ([`xr_sides`]).
type XrSides = XrSide<VulkanPlatform>;

/// Each of `blends` and the unblended draw of `fragment` of `library` into an attachment of
/// each of `formats` ([`XrSide::blended`]) across passes, on this platform and Metal's, the
/// test's `sides`: what fails the caller's test, and the blends whose texels differ from
/// Metal's where `expect` allows it. Each state is made, the unblended draw's texels are
/// Metal's, and each blend is held to `expect`.
fn xr_across(
    sides: &XrSides,
    drawn: (&str, &[u8], &str),
    shape: (&[u32], &[[u32; 6]]),
    expect: XrExpect<'_>,
) -> (Vec<String>, Vec<String>) {
    xr_compared(sides, drawn, shape, expect, true)
}

/// [`xr_across`]'s comparison, the blended draw in a pass of its own where `across`, else in
/// the unblended draw's pass ([`XrSide::blended`]).
fn xr_compared(
    ours: &XrSides,
    (mode, library, fragment): (&str, &[u8], &str),
    (formats, blends): (&[u32], &[[u32; 6]]),
    expect: XrExpect<'_>,
    across: bool,
) -> (Vec<String>, Vec<String>) {
    let (mut differ, mut missed) = (Vec::new(), Vec::new());
    // The texels `list` gives the blend, where it lists it.
    let listed = |list: &[XrGap], format, blend| {
        (list.iter())
            .find(|&&(f, b, _)| (f, Some(b)) == (format, blend))
            .map(|&(_, _, texels)| texels)
    };
    for &format in formats {
        for blend in std::iter::once(None).chain(blends.iter().copied().map(Some)) {
            let within = if across { "" } else { " in one pass" };
            let what = format!("{mode} {fragment} format {format} blend {blend:?}{within}");
            let drawn = (library, fragment);
            *crate::answer::LAST_LINE.lock() = None;
            let ours = catch_death(|| ours.blended(drawn, format, blend, across));
            let ours = match ours {
                Ok(ours) => ours,
                Err(site) => {
                    let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
                    differ.push(format!("{what}: died at {site:?}: {line}"));
                    continue;
                }
            };
            let (count, wrong) = if crate::metal_matches_bytes!(what, &ours) {
                (0, None)
            } else if let Some(theirs) = crate::metal_full!(what) {
                let count = (ours.chunks(16).zip(theirs.chunks(16)))
                    .filter(|(o, t)| o != t)
                    .count();
                (count, xr_wrong(&ours, &theirs))
            } else {
                let digest = "differs from Metal's, recorded by its digest".to_owned();
                (usize::MAX, Some(digest))
            };
            let wrong = wrong.map(|w| format!("{what}: {w}"));
            let allowed = match (expect, blend) {
                (_, None) => Some(0),
                (XrExpect::Exact(pinned), _) => Some(listed(pinned, format, blend).unwrap_or(0)),
                (XrExpect::Announced(None), _) => None,
                (XrExpect::Announced(Some(list)), _) => {
                    listed(list, format, blend).map_or(Some(0), |_| None)
                }
            };
            match allowed {
                Some(pinned) if count != pinned => differ.push(format!(
                    "{what}: {count} texels differ, pinned {pinned}: {}",
                    wrong.unwrap_or_default()
                )),
                Some(_) => {}
                None => missed.extend(wrong),
            }
        }
    }
    (differ, missed)
}

/// Each of [`XR_BLENDS`] of `fragment` of `library` into an attachment of each linear
/// extended-range format in the first draw's pass ([`XrSide::blended`]), on this platform and
/// Metal's, the test's `sides`, as [`xr_across`] compares them across passes: what fails the
/// caller's test, and the blends whose texels differ from Metal's where `expect` allows it. The
/// runtime holds such an attachment's `half` values through a pass of one sample and one layer,
/// as Metal's tile memory does, from the first draw that blends onto what an earlier draw of
/// the pass wrote, whose pass it records again, and announces no
/// `extended-range-blends-stored-codes` for a draw that read them, which the
/// caller's test requires unheard.
fn xr_within(
    sides: &XrSides,
    drawn: (&str, &[u8], &str),
    expect: XrExpect<'_>,
) -> (Vec<String>, Vec<String>) {
    xr_compared(
        sides,
        drawn,
        (&XR_LINEAR[..], &XR_BLENDS[..]),
        expect,
        false,
    )
}

/// A `float4` output blended into an attachment of each extended-range format where the
/// runtime's model is Metal's blend: compiled with fast math ([`ROUNDING`]), one each of whose
/// lanes is a load (`xr_loaded_fragment`), into each linear format (`RGB10A8_2P_XR10`,
/// `BGRA10_XR`, `BGR10_XR`) under each of [`XR_BLENDS`], of every factor kind and operation
/// (those of the runtime's model among them: one minus the source colour beside an
/// alpha, the destination colour or one minus the source's alpha; the source times the
/// destination twice over; `min` and `max`), over a first draw's values (colour lanes from
/// past the formats' low end to past their high end, alpha past 0 and 1); and compiled
/// without fast math ([`ROUNDING_SAFE`]), a computed one (`xr_float_fragment`) and a loaded
/// one, into each linear format under each of [`XR_BLENDS`] and into each `_sRGB` twin under
/// each of [`XR_SRGB_BLENDS`], whose products the runtime rounds to `half` as Metal's
/// compiler does there. Across passes ([`xr_across`]) each state is made and
/// stores Metal's values on this host, read back by a kernel, bit for bit, as does the first
/// draw alone, and none announces `extended-range-blend-source` or
/// `extended-range-blends-stored-codes`; in the first draw's pass ([`xr_within`]) each linear
/// one is made and stores Metal's values, bit for bit, and none announces the stored codes,
/// as the runtime holds the attachment's `half` values through the pass
#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "MoltenVK: the runtime records device-pack-conversion, a death in strict mode; Mac-only test, off everywhere; ledger: MoltenVK strict deaths"
)]
fn float4_extended_range_blends_are_metals_here() {
    let linear = (&XR_LINEAR[..], &XR_BLENDS[..]);
    let srgb = (&XR_SRGB[..], &XR_SRGB_BLENDS[..]);
    let sweeps = [
        (("fast math", ROUNDING, "xr_loaded_fragment"), linear),
        (
            ("fast math off", ROUNDING_SAFE, "xr_float_fragment"),
            linear,
        ),
        (
            ("fast math off", ROUNDING_SAFE, "xr_loaded_fragment"),
            linear,
        ),
        (("fast math off", ROUNDING_SAFE, "xr_float_fragment"), srgb),
        (("fast math off", ROUNDING_SAFE, "xr_loaded_fragment"), srgb),
    ];
    let sides = xr_sides();
    let mut differ = Vec::new();
    for (drawn, shape) in sweeps {
        differ.extend(xr_across(&sides, drawn, shape, XrExpect::Exact(&[])).0);
    }
    let source = rt::Degradation::ExtendedRangeBlendSource;
    let stored_codes = rt::Degradation::ExtendedRangeBlendsStoredCodes;
    for kind in [source, stored_codes] {
        if heard(kind) {
            differ.push(format!("{} across passes", kind.code()));
        }
    }
    for &(drawn, _) in &sweeps[..3] {
        differ.extend(xr_within(&sides, drawn, XrExpect::Exact(&[])).0);
    }
    if heard(stored_codes) {
        differ.push(format!("{} in one pass", stored_codes.code()));
    }
    assert!(differ.is_empty(), "{} differ: {differ:#?}", differ.len());
}

/// A `float4` output blended into an attachment of each linear extended-range format where
/// Metal may round the blend's source otherwise than the runtime's model: one whose
/// lanes the function computes (`xr_float_fragment`), compiled with fast math ([`ROUNDING`]).
/// The runtime notes such a state `extended-range-float-blend-source`, apart from
/// a `half4` output's `extended-range-blend-source`, and announces it at each draw that binds
/// the state's variant: the state is made, and its draw dies there, naming the kind
/// ([`crate::answer::DIES`]), where Metal makes it and draws: 17 of these 66
/// blends missed Metal's by one code across passes at 35 to 409 of 4096 texels each (at the
/// runtime's 7647e5c22 and 173b29636): in each format (colour, then alpha) (4, 5, add; 1, 5,
/// add), (4, 5, add; 4, 5, add), (3, 5, add; 1, 5, add), (6, 2, add; 1, 1, add) and (5, 4,
/// add; 5, 4, add), and in `RGB10A8_2P_XR10` and `BGRA10_XR` (8, 9, add; 8, 9, add). The
/// function's unblended draw is Metal's, bit for bit
#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "MoltenVK: the runtime records device-pack-conversion, a death in strict mode; Mac-only test, off everywhere; ledger: MoltenVK strict deaths"
)]
fn float4_extended_range_blends_die_naming_their_source() {
    use crate::core::raise::death::DeathKind;

    const DRAW: &str = "drawPrimitives:vertexStart:vertexCount:";
    let drawn = ("fast math", ROUNDING, "xr_float_fragment");
    let sides = xr_sides();
    let (mut differ, _) = xr_across(&sides, drawn, (&XR_LINEAR[..], &[]), XrExpect::Exact(&[]));
    let ours = &sides;
    let named = rt::Degradation::ExtendedRangeFloatBlendSource.code();
    for format in XR_LINEAR {
        for blend in XR_BLENDS {
            let what = format!("format {format} blend {blend:?}");
            let drawn = (ROUNDING, "xr_float_fragment");
            *crate::answer::LAST_LINE.lock() = None;
            let ours = catch_death(|| ours.blended(drawn, format, Some(blend), true));
            let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
            match ours {
                Ok(_) => differ.push(format!("{what}: drawn")),
                Err(site)
                    if (site.kind, site.file) != (DeathKind::Abort, DRAW)
                        || !line.contains(named) =>
                {
                    differ.push(format!("{what}: died at {site:?}: {line}"));
                }
                Err(_) => {}
            }
        }
    }
    assert!(differ.is_empty(), "{} differ: {differ:#?}", differ.len());
}

/// Blends of the colour channels' blend colour with its one minus under a difference
/// (`MTLBlendOperationSubtract` 1, `MTLBlendOperationReverseSubtract` 2), as [`XR_BLENDS`],
/// which the runtime draws without fast math alone.
const XR_CONSTANT_DIFFERENCES: [[u32; 6]; 2] = [[11, 12, 1, 1, 1, 0], [12, 11, 2, 1, 1, 0]];

/// The blends the runtime draws into an extended-range attachment without fast math alone, of a
/// `half4` output (`xr_half_fragment`), a
/// computed `float4` one (`xr_float_fragment`) and a loaded one (`xr_loaded_fragment`),
/// compiled without fast math ([`ROUNDING_SAFE`]): [`XR_CONSTANT_DIFFERENCES`] into each
/// linear form, and each of [`XR_BLENDS`] and [`XR_CONSTANT_DIFFERENCES`] into each `_sRGB`
/// twin, those into `BGRA10_XR_sRGB` whose colour channels read the blend colour itself
/// (`MTLBlendFactorBlendColor`) among them, which the runtime draws. Across passes ([`xr_across`])
/// each state is made and stores Metal's values, bit for bit, and none announces
/// `extended-range-blend-source`, `extended-range-float-blend-source` or
/// `extended-range-blends-stored-codes`. Which Metal makes and dies as the state is made,
/// naming the runtime's refusal, compiled with fast math ([`ROUNDING`]):
/// [`XR_CONSTANT_DIFFERENCES`] into `BGRA10_XR` (`ExtendedRangeBlend(ConstantColorDifference)`)
/// and one of source factor the source's alpha into `BGRA10_XR_sRGB`, beyond the runtime's
/// measured blends (`ExtendedRangeBlend(SrgbForm)`), of the `half4` output. A blend under a write
/// mask of one colour channel, or by a factor of the second source, which the runtime draws
/// without fast math too, is
/// [`extended_range_masked_blends_and_two_output_states_are_metals`]'s
#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "MoltenVK: the runtime records device-pack-conversion, a death in strict mode; Mac-only test, off everywhere; ledger: MoltenVK strict deaths"
)]
fn extended_range_blends_drawn_without_fast_math_alone_are_metals() {
    use crate::core::raise::death::DeathKind;

    const STATE: &str = "newRenderPipelineStateWithDescriptor:options:reflection:error:";
    let fragments = [
        "xr_half_fragment",
        "xr_float_fragment",
        "xr_loaded_fragment",
    ];
    let srgb: Vec<[u32; 6]> = XR_BLENDS
        .into_iter()
        .chain(XR_CONSTANT_DIFFERENCES)
        .collect();
    let sides = xr_sides();
    let mut differ = Vec::new();
    for fragment in fragments {
        let drawn = ("fast math off", ROUNDING_SAFE, fragment);
        let shape = (&XR_LINEAR[..], &XR_CONSTANT_DIFFERENCES[..]);
        differ.extend(xr_across(&sides, drawn, shape, XrExpect::Exact(&[])).0);
        for format in XR_SRGB {
            differ.extend(xr_across(&sides, drawn, (&[format], &srgb), XrExpect::Exact(&[])).0);
        }
    }
    for kind in [
        rt::Degradation::ExtendedRangeBlendSource,
        rt::Degradation::ExtendedRangeFloatBlendSource,
        rt::Degradation::ExtendedRangeBlendsStoredCodes,
    ] {
        if heard(kind) {
            differ.push(format!("{} across passes", kind.code()));
        }
    }
    let fast = ROUNDING;
    let named = "ExtendedRangeBlend(ConstantColorDifference)";
    let refused = (XR_CONSTANT_DIFFERENCES.map(|b| ((fast, "xr_half_fragment"), 552, b, named)))
        .into_iter()
        .chain(std::iter::once((
            (fast, "xr_half_fragment"),
            553,
            [4, 5, 0, 1, 5, 0],
            "ExtendedRangeBlend(SrgbForm)",
        )));
    let ours = &sides;
    for ((library, fragment), format, blend, named) in refused {
        let what = format!("{fragment} format {format} blend {blend:?}");
        let drawn = (library, fragment);
        *crate::answer::LAST_LINE.lock() = None;
        let ours = catch_death(|| ours.blended(drawn, format, Some(blend), true));
        let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
        match ours {
            Ok(_) => differ.push(format!("{what}: drawn")),
            Err(site)
                if (site.kind, site.file) != (DeathKind::Abort, STATE) || !line.contains(named) =>
            {
                differ.push(format!("{what}: died at {site:?}: {line}"));
            }
            Err(_) => {}
        }
    }
    assert!(differ.is_empty(), "{} differ: {differ:#?}", differ.len());
}

/// The blends [`extended_range_masked_blends_and_two_output_states_are_metals`] draws
/// under a write mask, as [`XR_BLENDS`]: added terms of factors one, zero, the source's
/// alpha and colour and their one minus, the destination's alpha and its one minus.
const XR_MASKED_BLENDS: [[u32; 6]; 6] = [
    [1, 1, 0, 1, 1, 0],
    [0, 5, 0, 0, 5, 0],
    [4, 5, 0, 1, 5, 0],
    [3, 8, 0, 1, 1, 0],
    [5, 4, 0, 5, 4, 0],
    [8, 9, 0, 8, 9, 0],
];

/// The write masks (`MTLColorWriteMask`: red 8, green 4, blue 2, alpha 1) of one colour
/// channel, red, green and alpha, and blue, and of two, red and green, and green, blue and
/// alpha.
const XR_MASKS: [u32; 5] = [0x8, 0x5, 0x2, 0xc, 0x7];

/// Blends of the second source (`MTLBlendFactorSource1Color` 15, `OneMinusSource1Color` 16,
/// `Source1Alpha` 17, `OneMinusSource1Alpha` 18), as [`XR_BLENDS`]: in each factor, beside
/// one and the first source's alpha, under add, subtract (1) and reverse subtract (2).
const XR_DUAL_BLENDS: [[u32; 6]; 6] = [
    [15, 16, 0, 17, 18, 0],
    [1, 16, 0, 1, 18, 0],
    [17, 18, 0, 17, 18, 0],
    [15, 1, 0, 1, 1, 0],
    [4, 16, 0, 17, 5, 0],
    [16, 15, 2, 18, 17, 1],
];

/// The two blends the runtime says it draws into an extended-range attachment without
/// fast math alone that [`extended_range_blends_drawn_without_fast_math_alone_are_metals`]
/// leaves out, compiled without fast math ([`ROUNDING_SAFE`]), into each linear form and each
/// `_sRGB` twin, across passes ([`xr_blended_masked`]), the first draw unblended under every
/// channel. Each of [`XR_MASKED_BLENDS`] under each of [`XR_MASKS`], of one colour channel
/// and of two, of a `half4` output (`xr_half_fragment`), a computed `float4` one
/// (`xr_float_fragment`) and a loaded one (`xr_loaded_fragment`): nothing of this platform's
/// stands between such a state and the runtime, which makes it and draws it, announcing
/// nothing; each state is made and stores Metal's values, bit for bit, and none announces
/// `extended-range-blend-source`, `extended-range-float-blend-source` or
/// `extended-range-blends-stored-codes`. A state of a function that returns two outputs into
/// colour 0, two `half4` ones (`xr_dual_half_fragment`) or two `float4` ones
/// (`xr_dual_float_fragment`), into each form and twin, is made and stores Metal's values, bit
/// for bit: unblended, compiled without fast math and with it ([`ROUNDING`]), its first
/// output's, the runtime storing the second nowhere, as Metal's write reads none;
/// and compiled without fast math under each of [`XR_DUAL_BLENDS`], which read its second
/// output, which the runtime blends in the shader. Which Metal makes and dies as the
/// state is made, naming the runtime's refusal: compiled with fast math, such a state under a
/// blend of the second source into `BGRA10_XR` (`ExtendedRangeBlend(DualSource)`), and a
/// blend under a mask of one colour channel into `BGRA10_XR`
/// (`ExtendedRangeBlend(OneColorChannel)`)
#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "MoltenVK: the runtime records device-pack-conversion, a death in strict mode; Mac-only test, off everywhere; ledger: MoltenVK strict deaths"
)]
fn extended_range_masked_blends_and_two_output_states_are_metals() {
    use crate::core::raise::death::DeathKind;

    const STATE: &str = "newRenderPipelineStateWithDescriptor:options:reflection:error:";
    let ours = xr_sides();
    let mut differ = Vec::new();
    for format in XR_LINEAR.into_iter().chain(XR_SRGB) {
        for fragment in [
            "xr_half_fragment",
            "xr_float_fragment",
            "xr_loaded_fragment",
        ] {
            for (blend, mask) in XR_MASKS
                .into_iter()
                .flat_map(|mask| XR_MASKED_BLENDS.map(|blend| (blend, mask)))
            {
                let shape = (Some(blend), Some(mask));
                let what = format!("{fragment} format {format} blend {blend:?} mask {mask:#x}");
                let library = (ROUNDING_SAFE, fragment);
                *crate::answer::LAST_LINE.lock() = None;
                let ours = catch_death(|| ours.masked(library, format, shape, true));
                match ours {
                    Ok(ours) if crate::metal_matches_bytes!(what, &ours) => {}
                    Ok(_) => differ.push(format!("{what}: the texels differ from Metal's")),
                    Err(site) => {
                        let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
                        differ.push(format!("{what}: died at {site:?}: {line}"));
                    }
                }
            }
        }
    }
    // Each two-output state compared: unblended with and without fast math, and blended by
    // each of `XR_DUAL_BLENDS` without.
    let dual = (std::iter::once((ROUNDING, None))).chain(
        std::iter::once(None)
            .chain(XR_DUAL_BLENDS.map(Some))
            .map(|b| (ROUNDING_SAFE, b)),
    );
    let dual: Vec<(&[u8], Option<[u32; 6]>)> = dual.collect();
    for format in XR_LINEAR.into_iter().chain(XR_SRGB) {
        for fragment in ["xr_dual_half_fragment", "xr_dual_float_fragment"] {
            for &(library, blend) in &dual {
                let fast = if library == ROUNDING {
                    "fast math"
                } else {
                    "fast math off"
                };
                let what = format!("{fragment} {fast} format {format} blend {blend:?}");
                let library = (library, fragment);
                let shape = (blend, None);
                *crate::answer::LAST_LINE.lock() = None;
                let ours = catch_death(|| ours.masked(library, format, shape, true));
                match ours {
                    Ok(ours) if crate::metal_matches_bytes!(what, &ours) => {}
                    Ok(_) => differ.push(format!("{what}: the texels differ from Metal's")),
                    Err(site) => {
                        let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
                        differ.push(format!("{what}: died at {site:?}: {line}"));
                    }
                }
            }
        }
    }
    for kind in [
        rt::Degradation::ExtendedRangeBlendSource,
        rt::Degradation::ExtendedRangeFloatBlendSource,
        rt::Degradation::ExtendedRangeBlendsStoredCodes,
    ] {
        if heard(kind) {
            differ.push(format!("{} across passes", kind.code()));
        }
    }
    // Each death: its library and function, format, blend and mask, and what its line names.
    type Refused = (
        (&'static [u8], &'static str),
        u32,
        (Option<[u32; 6]>, Option<u32>),
        &'static str,
    );
    let refused: [Refused; 2] = [
        (
            (ROUNDING, "xr_dual_half_fragment"),
            552,
            (Some([15, 16, 0, 17, 18, 0]), None),
            "ExtendedRangeBlend(DualSource)",
        ),
        (
            (ROUNDING, "xr_half_fragment"),
            552,
            (Some([4, 5, 0, 1, 5, 0]), Some(0x8)),
            "ExtendedRangeBlend(OneColorChannel)",
        ),
    ];
    for (library, format, shape, named) in refused {
        let what = format!("{} format {format} {shape:?}", library.1);
        *crate::answer::LAST_LINE.lock() = None;
        let ours = catch_death(|| ours.masked(library, format, shape, true));
        let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
        match ours {
            Ok(_) => differ.push(format!("{what}: drawn")),
            Err(site)
                if (site.kind, site.file) != (DeathKind::Abort, STATE) || !line.contains(named) =>
            {
                differ.push(format!("{what}: died at {site:?}: {line}"));
            }
            Err(_) => {}
        }
    }
    assert!(differ.is_empty(), "{} differ: {differ:#?}", differ.len());
}

/// The blends of `xr_half_fragment` compiled with fast math across passes ([`xr_across`])
/// whose texels missed Metal's on this host's M3 Max at the runtime's 7647e5c22, each as
/// [`XrGap`]: its format, blend (as [`XR_BLENDS`]) and how many of its 4096 texels differed,
/// each by one code of a channel, at blends whose factors are other than zero and one. The
/// runtime announces a computed `half4` under fast math as `extended-range-blend-source`, which
/// dies at the draw
/// ([`crate::answer::DIES`]); the half4 test's sweeps let these alone differ, at any count, of
/// a draw that does not die. At the runtime's 173b29636 the same 13 differ.
const HALF4_FAST_GAP: [XrGap; 13] = [
    (550, [3, 8, 0, 1, 1, 0], 333),
    (550, [3, 6, 0, 1, 1, 0], 483),
    (550, [3, 5, 0, 1, 5, 0], 130),
    (550, [8, 9, 0, 8, 9, 0], 413),
    (550, [5, 4, 0, 5, 4, 0], 425),
    (552, [3, 8, 0, 1, 1, 0], 315),
    (552, [3, 6, 0, 1, 1, 0], 483),
    (552, [3, 5, 0, 1, 5, 0], 130),
    (552, [8, 9, 0, 8, 9, 0], 412),
    (552, [5, 4, 0, 5, 4, 0], 425),
    (554, [3, 6, 0, 1, 1, 0], 483),
    (554, [3, 5, 0, 1, 5, 0], 130),
    (554, [5, 4, 0, 5, 4, 0], 425),
];

/// A `half4` output (`xr_half_fragment`, each lane a value a `half` holds, computed from the
/// pixel), compiled with fast math ([`ROUNDING`]), drawn into an attachment of each linear
/// extended-range format unblended and blended under each of [`XR_BLENDS`], across passes
/// ([`xr_across`]) and in the first draw's pass ([`xr_within`]), which Metal draws: every draw
/// of this platform's, the blended and the unblended alike, dies at its selector
/// naming `device-pack-conversion` ([`crate::answer::DIES`]). The runtime holds such an
/// attachment in its codes, which the fragment function converts by the device's arithmetic
/// (a read multiplies a code by the `f32` nearest `1 / 510`, a write rounds its value's `f32`
/// product by 510, where Metal answers the correctly rounded quotient and encodes by the exact
/// product), and the translation of every function that converts such a texel records
/// `DevicePackConversion` on every device, MoltenVK among them, which a device on Metal does not
/// relay (the runtime's `lower/arith/extended_range.rs`; measured on the M3 Max under macOS
/// 27.0.1, MoltenVK 1.4.2, at the runtime's 57b3b26ba0: all 138 draws). That is the line a
/// blended draw dies with, where at the runtime's earlier commits it named the render state's
/// `extended-range-blend-source`, which the runtime announces for such an output; a draw that
/// did not die is held to [`HALF4_FAST_GAP`], where those blends missed Metal's
#[test]
fn half4_extended_range_blends_announce_their_source() {
    let drawn = ("fast math", ROUNDING, "xr_half_fragment");
    let shape = (&XR_LINEAR[..], &XR_BLENDS[..]);
    let named = "(metal2vulkan-degrade-device-pack-conversion)";
    let at = "\"drawPrimitives:vertexStart:vertexCount:\"";
    let expect = XrExpect::Announced(Some(&HALF4_FAST_GAP));
    let sides = xr_sides();
    let (across, _) = xr_across(&sides, drawn, shape, expect);
    let (within, _) = xr_within(&sides, drawn, expect);
    let (died, differ): (Vec<_>, Vec<_>) = (across.into_iter().chain(within))
        .partition(|what| what.contains(named) && what.contains(at));
    assert!(differ.is_empty(), "{} differ: {differ:#?}", differ.len());
    let draws = 2 * XR_LINEAR.len() * (1 + XR_BLENDS.len());
    assert_eq!(died.len(), draws, "the draws that die: {died:#?}");
}

/// A `half4` output (`xr_half_fragment`) compiled without fast math ([`ROUNDING_SAFE`]), as
/// RenderBox's `primitive_image` functions are, blended into an attachment of each linear
/// extended-range format under each of [`XR_BLENDS`] and of each `_sRGB` twin under each of
/// [`XR_SRGB_BLENDS`]: the runtime rounds each product to `half` as Metal's compiler does
/// there. Across passes ([`xr_across`]) each state is made and stores Metal's
/// values, bit for bit, and none announces `extended-range-blend-source` or
/// `extended-range-blends-stored-codes`; in the first draw's pass ([`xr_within`]) each linear
/// one is made and stores Metal's values, bit for bit, and none announces the stored codes,
/// as the runtime holds the attachment's `half` values through the pass
#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "MoltenVK: the runtime records device-pack-conversion, a death in strict mode; Mac-only test, off everywhere; ledger: MoltenVK strict deaths"
)]
fn half4_extended_range_blends_are_metals_without_fast_math() {
    let drawn = ("fast math off", ROUNDING_SAFE, "xr_half_fragment");
    let source = rt::Degradation::ExtendedRangeBlendSource;
    let stored_codes = rt::Degradation::ExtendedRangeBlendsStoredCodes;
    let sides = xr_sides();
    let mut differ = Vec::new();
    for shape in [
        (&XR_LINEAR[..], &XR_BLENDS[..]),
        (&XR_SRGB[..], &XR_SRGB_BLENDS[..]),
    ] {
        differ.extend(xr_across(&sides, drawn, shape, XrExpect::Exact(&[])).0);
    }
    for kind in [source, stored_codes] {
        if heard(kind) {
            differ.push(format!("{} across passes", kind.code()));
        }
    }
    differ.extend(xr_within(&sides, drawn, XrExpect::Exact(&[])).0);
    if heard(stored_codes) {
        differ.push(format!("{} in one pass", stored_codes.code()));
    }
    assert!(differ.is_empty(), "{} differ: {differ:#?}", differ.len());
}

/// The floats `xr_read` reads of a Private 64 by 64 `BGRA10_XR` attachment after one pass on
/// platform `P` that clears it to `clear` and draws `xr_half_fragment` of seed 1 into it under
/// source and destination factor one, added, colour and alpha.
fn xr_blended_onto_clear<P: Platform>(p: &P, d: &P::Device, clear: [f64; 4]) -> Vec<u8> {
    use crate::core::platform::compute::{ComputeCmd, ComputeEncoder as _};
    use crate::core::platform::render::{RenderCmd, RenderEncoder as _};
    use crate::core::platform::resource::Library as _;
    use crate::core::platform::types::ClearColor;

    use crate::gpu::device::tests::render_value;

    let library = d.new_library(ROUNDING).unwrap().unwrap().unwrap();
    let function = |name| library.new_function(OnFault::Die, name).unwrap().unwrap();
    let functions = [function("sweep_vertex"), function("xr_half_fragment")];
    let colour = [
        (0, 0),
        (1, 552),
        (2, 1),
        (3, 1),
        (4, 1),
        (5, 0),
        (6, 1),
        (7, 1),
        (8, 0),
    ];
    let desc = render_value::<P>((&functions[0], &functions[1]), 1, &[&colour]);
    let state = d.new_render_pipeline_state_full(&desc).unwrap().unwrap();
    let read = d
        .new_compute_pipeline_state_for_function(Some(&function("xr_read")))
        .unwrap()
        .unwrap();
    let private = p.texture_descriptor(&TextureDesc {
        pixel_format: 552,
        width: 64,
        height: 64,
        usage: READ | TARGET,
        resource_options: Some(0x20),
        ..TextureDesc::default()
    });
    let t = d.new_texture(OnFault::Die, &private).unwrap().unwrap();
    let seed = buffer::<P>(d, 4, 0, &1u32.to_le_bytes());
    let length = 64 * 64 * 16;
    let out = buffer::<P>(d, length, 0, &[]);
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let pass = p.render_pass_descriptor();
    let [red, green, blue, alpha] = clear;
    let attachment = ColorAttachmentSet {
        texture: Some(Some(&t)),
        load_action: Some(CLEAR),
        store_action: Some(STORE),
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
    let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
    for cmd in [
        RenderCmd::SetRenderPipelineState {
            pipeline_state: Some(&state),
        },
        RenderCmd::SetFragmentBuffer {
            buffer: Some(&seed),
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
    let c = cb.compute_encoder(OnFault::Die).unwrap().unwrap();
    let outs = [Some(&out)];
    for cmd in [
        ComputeCmd::SetComputePipelineState { state: Some(&read) },
        ComputeCmd::SetTexture {
            texture: Some(&t),
            index: 0,
        },
        ComputeCmd::SetBuffers {
            buffers: &outs,
            offsets: &[0],
            range: NsRange::new(0, 1),
        },
        ComputeCmd::DispatchThreads {
            threads_per_grid: MtlSize::new(64, 64, 1),
            threads_per_threadgroup: MtlSize::new(8, 8, 1),
        },
        ComputeCmd::EndEncoding,
    ] {
        c.encode(OnFault::Die, cmd).unwrap();
    }
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    assert!(cb.error().is_none(), "{:?}", cb.error());
    let n = usize::try_from(length).unwrap();
    // SAFETY: a Shared buffer's bytes, read after the kernel completed.
    unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), n) }.to_vec()
}

/// A `half4` blend of one and one (`xr_blended_onto_clear`, its output computed under fast
/// math) onto a `BGRA10_XR` attachment its pass cleared: onto a clear colour of codes (every
/// channel 0 but alpha 0.5, which `BGRA10_XR` holds exactly), and onto a clear colour whose red
/// lies between two codes (0.0005, between 0 and 1/510), where Metal's tile memory holds the
/// clear's halves. This platform's draw dies at its selector naming
/// `extended-range-blend-source` ([`crate::answer::DIES`]), which the runtime announces for a
/// computed `half4` under fast math blended into an extended-range attachment, whatever the
/// factors, though a blend of one and one is among those it measured Metal's
/// ([`HALF4_FAST_GAP`] lists none such)
#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "MoltenVK: the runtime records device-pack-conversion, a death in strict mode; Mac-only test, off everywhere; ledger: MoltenVK strict deaths"
)]
fn a_blend_onto_a_clear_between_codes_is_metals() {
    let (p, d) = (platform(), dev());
    let kind = rt::Degradation::ExtendedRangeBlendSource;
    for clear in [[0.0, 0.0, 0.0, 0.5], [0.0005, 0.0, 0.0, 0.5]] {
        let site = crate::answer::dies_of(kind, || xr_blended_onto_clear(&p, &d, clear));
        assert_eq!(
            site.file, "drawPrimitives:vertexStart:vertexCount:",
            "{clear:?}"
        );
    }
}

/// How a texel's bytes read as channels, for [`half_outputs_store_within_the_bound_or_die_named`]:
/// their codes, a float format's bits as ordered integers (one apart a unit in the last place), and
/// a shared-exponent format's values in units of the finer of the two codes compared.
#[derive(Clone, Copy, Debug)]
enum Channels {
    /// One byte each.
    U8,
    /// `RGB10A2Unorm`'s 10, 10, 10 and 2 bits.
    Packed1010102,
    /// Two bytes each, unsigned.
    U16,
    /// `RG11B10Float`'s 11, 11 and 10 bits, each ordered as its code.
    Packed111110,
    /// `RGB9E5Float`'s three 9-bit mantissas over a shared 5-bit exponent.
    Shared9e5,
    /// A `half` each.
    Half,
    /// A `float` each.
    Float,
}

impl Channels {
    /// The bytes per channel group the decoder steps over.
    fn decoded(self, bytes: &[u8]) -> Vec<f64> {
        let word = |c: &[u8]| u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        let ordered = |bits: i64, sign: i64| {
            if bits & sign != 0 {
                -(bits & (sign - 1))
            } else {
                bits
            }
        };
        match self {
            Channels::U8 => bytes.iter().map(|&b| f64::from(b)).collect(),
            Channels::U16 => bytes
                .chunks(2)
                .map(|c| f64::from(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            Channels::Packed1010102 => (bytes.chunks(4).map(word))
                .flat_map(|w| [w & 0x3ff, (w >> 10) & 0x3ff, (w >> 20) & 0x3ff, w >> 30])
                .map(f64::from)
                .collect(),
            Channels::Packed111110 => (bytes.chunks(4).map(word))
                .flat_map(|w| [w & 0x7ff, (w >> 11) & 0x7ff, w >> 22])
                .map(f64::from)
                .collect(),
            Channels::Shared9e5 => (bytes.chunks(4).map(word))
                .flat_map(|w| {
                    let scale = (2f64).powi(i32::try_from(w >> 27).unwrap() - 24);
                    [w & 0x1ff, (w >> 9) & 0x1ff, (w >> 18) & 0x1ff].map(|m| f64::from(m) * scale)
                })
                .collect(),
            Channels::Half => bytes
                .chunks(2)
                .map(|c| ordered(i64::from(u16::from_le_bytes([c[0], c[1]])), 0x8000) as f64)
                .collect(),
            Channels::Float => (bytes.chunks(4).map(word))
                .map(|w| ordered(i64::from(w), 0x8000_0000) as f64)
                .collect(),
        }
    }

    /// How far apart `ours` and `theirs` lie at most, in codes, units in the last place, or, for
    /// a shared exponent, the finer code of the two compared; and at how many channels they
    /// differ.
    fn apart(self, ours: &[u8], theirs: &[u8]) -> (f64, usize) {
        let (o, m) = (self.decoded(ours), self.decoded(theirs));
        let step = |b: &[u8]| -> Vec<f64> {
            (b.chunks(4))
                .flat_map(|c| {
                    let e = u32::from_le_bytes([c[0], c[1], c[2], c[3]]) >> 27;
                    [(2f64).powi(i32::try_from(e).unwrap() - 24); 3]
                })
                .collect()
        };
        let steps = match self {
            Channels::Shared9e5 => Some((step(ours), step(theirs))),
            _ => None,
        };
        let mut most = 0f64;
        let mut differ = 0;
        for (i, (&a, &b)) in o.iter().zip(&m).enumerate() {
            if a != b {
                differ += 1;
            }
            let unit = steps.as_ref().map_or(1.0, |(s, t)| s[i].min(t[i]));
            most = most.max((a - b).abs() / unit);
        }
        (most, differ)
    }
}

/// A `half` colour output (`rounding.metal`'s `sweep_half_fragment`, a `half4` converted from
/// `float`, every channel the sweep's value), which the runtime rounds to `half` before the blend
/// and the format's conversion, drawn over the sweeps from -0.25 to 1.25 (0) and from -2 to 2 (4)
/// onto a 256 by 256 attachment loaded with texels from 0 to 1 (`LOAD`; zeros for `RG11B10Float`
/// and `RGB9E5Float`), from a congruential sequence (seed 7919, `x * 1103515245 + 12345`, bits 8 to
/// 23 of each step: a unorm byte its low byte, a `half` the step modulo 0x3c01, a `float` the step
/// over 65535), blended as each case says, the blend colour (0.3, 0.6, 0.9, 0.45). Compiled with
/// fast math ([`ROUNDING`]), the function notes `half-output-fast-math-blend`, relayed in the
/// accepted class of fast-math rewrites: every draw draws, and one the rule below relayed
/// stores within the bound, against Metal's bytes: one code, one unit in the
/// last place of a `half`, or a `float` channel's own bits. Compiled without it
/// ([`ROUNDING_SAFE`]), the function notes `half-output-precision`, and a draw into an attachment
/// that stores it past the bound under fast math (`encode_render`'s `half_apart`) dies at its
/// selector, naming the kind; the cases measured past it are the ones the record of
/// 2026-10-11 lists (`One` and `One` into `R16Float` 384 `half` units apart, into `RG16Float`
/// 3072, `DestinationColor` and `OneMinusDestinationColor` into `RGBA16Float` 2213, `SourceAlpha`
/// and `OneMinusSourceAlpha` into `RGBA32Float` 1 float unit, into an 8-bit `_sRGB` format 2
/// codes at `DestinationColor` and `OneMinusDestinationColor`), and those that die with the same
/// rule within it (`One` and `Zero` into `RGBA16Float`, whose zero factor's product Metal keeps
/// at texels this sweep does not load, premultiplied `One` and `OneMinusSourceAlpha` into
/// `RGBA16Float` and `RG16Float`, `BlendColor` and `OneMinusBlendColor` into `RGBA16Float`,
/// `OneMinusBlendColor`
/// and `Zero` into `R16Float`, `SourceColor` and `DestinationColor` into `RGBA32Float`,
/// `SourceAlpha` and `OneMinusSourceAlpha` into `R32Float` and `RG32Float`). That death is strict
/// mode's: by default (`answer::relaxed`) each such draw draws, relaying the difference, counted
/// (`answer::Inexact::HalfOutputBlend`). A
/// `float4` output (`sweep_fragment`) under the same state, which the runtime notes nothing of,
/// draws and stores Metal's bytes in every case
#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "MoltenVK: the runtime records device-nan-encoding, a death in strict mode; Mac-only test, off everywhere; ledger: MoltenVK strict deaths"
)]
fn half_outputs_store_within_the_bound_or_die_named() {
    use crate::core::platform::resource::Library as _;

    use crate::gpu::device::tests::render_value;

    /// `MTLLoadActionLoad`.
    const LOAD: u64 = 1;
    /// `MTLBlendOperation`: add, minimum and maximum.
    const ADD: u32 = 0;
    const MIN: u32 = 3;
    const MAX: u32 = 4;

    /// The bytes of a 256 by 256 attachment of `format`, `bytes` a texel, loaded with `under`,
    /// after `fragment` draws the sweep `kind` into it, blended as `blend` says (each channel's
    /// source and destination factors and operation), unblended where `None`.
    fn drawn<P: Platform>(
        (p, d): (&P, &P::Device),
        (format, bytes): (u64, u64),
        (library, fragment, kind): (&[u8], &str, u32),
        blend: Option<(u32, u32, u32)>,
        under: &[u8],
    ) -> Vec<u8> {
        let library = d.new_library(library).unwrap().unwrap().unwrap();
        let function = |name| library.new_function(OnFault::Die, name).unwrap().unwrap();
        let functions = [function("sweep_vertex"), function(fragment)];
        let t = texture(
            p,
            d,
            format,
            READ | TARGET,
            (256, 256),
            Some((under, 256 * bytes)),
        );
        let kinds = buffer::<P>(d, 4, 0, &kind.to_le_bytes());
        let mut colour = vec![(0, 0), (1, u32::try_from(format).unwrap())];
        if let Some((source, destination, operation)) = blend {
            colour.extend([(2, 1), (3, source), (4, destination), (5, operation)]);
            colour.extend([(6, source), (7, destination), (8, operation)]);
        }
        let desc = render_value::<P>((&functions[0], &functions[1]), 1, &[&colour]);
        let state = d.new_render_pipeline_state_full(&desc).unwrap().unwrap();
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let pass = p.render_pass_descriptor();
        let attachment = ColorAttachmentSet {
            texture: Some(Some(&t)),
            load_action: Some(LOAD),
            store_action: Some(STORE),
            ..ColorAttachmentSet::default()
        };
        pass.set_color_attachment(OnFault::Die, 0, &attachment)
            .unwrap();
        let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
        for cmd in [
            RenderCmd::SetRenderPipelineState {
                pipeline_state: Some(&state),
            },
            RenderCmd::SetFragmentBuffer {
                buffer: Some(&kinds),
                offset: 0,
                index: 0,
            },
            RenderCmd::SetBlendColorRed {
                red: 0.3,
                green: 0.6,
                blue: 0.9,
                alpha: 0.45,
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
        let length = 256 * 256 * bytes;
        let out = buffer::<P>(d, length, 0, &[]);
        let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
        blit.encode(
            OnFault::Die,
            BlitCmd::CopyTextureToBuffer {
                source_texture: &t,
                source_slice: 0,
                source_level: 0,
                source_origin: MtlOrigin::new(0, 0, 0),
                source_size: MtlSize::new(256, 256, 1),
                destination_buffer: &out,
                destination_offset: 0,
                destination_bytes_per_row: 256 * bytes,
                destination_bytes_per_image: length,
            },
        )
        .unwrap();
        blit.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        assert!(cb.error().is_none(), "{:?}", cb.error());
        let n = usize::try_from(length).unwrap();
        // SAFETY: a Shared buffer's bytes, read after the blit completed.
        unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), n) }.to_vec()
    }

    let (p, d) = (platform(), dev());
    let blends_relayed = || {
        (crate::answer::inexact_counts().into_iter())
            .find(|&(kind, _)| kind == crate::answer::Inexact::HalfOutputBlend)
            .map_or(0, |(_, n)| n)
    };
    let mut state = 7919u32;
    let (mut bytes, mut halves, mut floats) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..256 * 256 * 8 {
        state = state.wrapping_mul(1_103_515_245).wrapping_add(12345);
        let step = (state >> 8) & 0xffff;
        bytes.push(u8::try_from(step & 0xff).unwrap());
        halves.extend(u16::try_from(step % 0x3c01).unwrap().to_le_bytes());
        floats.extend((f32::from(u16::try_from(step).unwrap()) / 65535.0).to_le_bytes());
    }
    let zeros = vec![0u8; 256 * 256 * 16];
    // Each format: its raw `MTLPixelFormat`, bytes a texel, channels and loaded texels; then the
    // blends relayed within the bound (`true`) or dying at the draw (`false`), as (source factor,
    // destination factor, operation) for every channel (`MTLBlendFactor`: 0 zero, 1 one, 2 source
    // colour, 3 one minus it, 4 source alpha, 5 one minus it, 6 destination colour, 7 one minus
    // it, 11 blend colour, 12 one minus it, 13 blend alpha, 14 one minus it).
    let unblended: Option<(u32, u32, u32)> = None;
    type Case<'a> = (
        (u64, u64),
        Channels,
        &'a [u8],
        Vec<(Option<(u32, u32, u32)>, bool)>,
    );
    let cases: Vec<Case<'_>> = vec![
        (
            (115, 8),
            Channels::Half,
            &halves,
            vec![
                (unblended, true),
                (Some((1, 0, ADD)), false),
                (Some((1, 5, ADD)), false),
                (Some((11, 12, ADD)), false),
                (Some((4, 5, MIN)), true),
                (Some((4, 5, MAX)), true),
                (Some((4, 5, ADD)), false),
                (Some((0, 5, ADD)), false),
                (Some((6, 7, ADD)), false),
                (Some((13, 14, ADD)), false),
            ],
        ),
        (
            (25, 2),
            Channels::Half,
            &halves,
            vec![
                (unblended, true),
                (Some((12, 0, ADD)), false),
                (Some((1, 1, ADD)), false),
                (Some((1, 5, ADD)), false),
            ],
        ),
        (
            (65, 4),
            Channels::Half,
            &halves,
            vec![(Some((1, 5, ADD)), false), (Some((1, 1, ADD)), false)],
        ),
        (
            (125, 16),
            Channels::Float,
            &floats,
            vec![
                (unblended, true),
                (Some((1, 5, ADD)), true),
                (Some((2, 6, ADD)), false),
                (Some((4, 5, ADD)), false),
                (Some((3, 3, ADD)), false),
                (Some((11, 5, ADD)), false),
            ],
        ),
        (
            (55, 4),
            Channels::Float,
            &floats,
            vec![(Some((4, 5, ADD)), false)],
        ),
        (
            (105, 8),
            Channels::Float,
            &floats,
            vec![(Some((4, 5, ADD)), false)],
        ),
        (
            (71, 4),
            Channels::U8,
            &bytes,
            vec![(Some((4, 5, ADD)), true), (Some((6, 7, ADD)), false)],
        ),
        (
            (81, 4),
            Channels::U8,
            &bytes,
            vec![(Some((6, 7, ADD)), false)],
        ),
        (
            (70, 4),
            Channels::U8,
            &bytes,
            vec![
                (unblended, true),
                (Some((4, 5, ADD)), true),
                (Some((6, 7, ADD)), true),
                (Some((0, 3, ADD)), true),
            ],
        ),
        (
            (90, 4),
            Channels::Packed1010102,
            &bytes,
            vec![(Some((4, 5, ADD)), true), (Some((6, 7, ADD)), true)],
        ),
        (
            (110, 8),
            Channels::U16,
            &bytes,
            vec![(Some((4, 5, ADD)), true)],
        ),
        (
            (92, 4),
            Channels::Packed111110,
            &zeros,
            vec![(unblended, true), (Some((4, 5, ADD)), true)],
        ),
        (
            (93, 4),
            Channels::Shared9e5,
            &zeros,
            vec![(unblended, true)],
        ),
    ];
    let mut wrong = Vec::new();
    for (format, channels, under, blends) in &cases {
        for &(blend, relayed) in blends {
            for kind in [0, 4] {
                let float4 = (ROUNDING, "sweep_fragment", kind);
                let key = format!("{format:?} {blend:?} sweep {kind}: a float4 output");
                let ours = drawn((&p, &d), *format, float4, blend, under);
                if !crate::metal_matches_bytes!(key, &ours) {
                    wrong.push(format!(
                        "{format:?} {blend:?} sweep {kind}: a float4 output"
                    ));
                }
            }
            if !relayed {
                let safe = (ROUNDING_SAFE, "sweep_half_fragment", 0);
                let site = crate::answer::dies_of(rt::Degradation::HalfOutputPrecision, || {
                    drawn((&p, &d), *format, safe, blend, under)
                });
                if site.file != "drawPrimitives:vertexStart:vertexCount:" {
                    wrong.push(format!("{format:?} {blend:?}: died at {}", site.file));
                }
                let before = blends_relayed();
                crate::answer::relaxed(|| drawn((&p, &d), *format, safe, blend, under));
                if blends_relayed() != before + 1 {
                    wrong.push(format!("{format:?} {blend:?}: not relayed by default"));
                }
                let fast = (ROUNDING, "sweep_half_fragment", 0);
                drawn((&p, &d), *format, fast, blend, under);
                continue;
            }
            let (mut most, mut differ) = (0f64, 0);
            for kind in [0, 4] {
                let half4 = (ROUNDING, "sweep_half_fragment", kind);
                let key = format!("{format:?} {blend:?} sweep {kind}: a half4 output");
                let ours = drawn((&p, &d), *format, half4, blend, under);
                // Metal's bytes in full where ours differ from them, else by their digest.
                let (far, count) = if crate::metal_matches_bytes!(key, &ours) {
                    (0.0, 0)
                } else if let Some(theirs) = crate::metal_full!(key) {
                    channels.apart(&ours, &theirs)
                } else {
                    (f64::INFINITY, ours.len())
                };
                (most, differ) = (most.max(far), differ + count);
            }
            eprintln!(
                "half output into {format:?}, blend {blend:?}: {most} apart at most, {differ} \
                 channels differ"
            );
            let bound = match channels {
                Channels::Float => 0.0,
                _ => 1.0,
            };
            if most > bound {
                wrong.push(format!(
                    "{format:?} {blend:?}: {most} apart, {differ} channels"
                ));
            }
        }
    }
    let relayed = crate::answer::RELAYED.lock().clone();
    if !relayed.contains(&rt::Degradation::HalfOutputFastMathBlend) {
        wrong.push(format!(
            "half-output-fast-math-blend not relayed: {relayed:?}"
        ));
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// A fragment function's `float4` output (`sweep_fragment`) and `half4` one
/// (`sweep_half_fragment`) drawn unblended into an attachment of each bfloat format
/// (`R16Bfloat`, `RG16Bfloat`, `RGBA16Bfloat`), over each of `rounding.metal`'s five sweeps
/// (through unorm values, past 1 and past -1 by float steps, past the largest half, and across
/// -2 to 2), stores Metal's codes, byte for byte: the runtime writes such an output as Metal's
/// tile holds and stores it (a `float` rounded to `half` toward zero, saturating at 65504, then
/// the nearest bfloat code, ties to even). A state
/// that blends into one, by one and zero, which Metal's validation refuses, dies as it is made,
/// naming the runtime's refusal (`Unsupported` of `Format`); Metal's is not made here. A draw
/// in a pass that loads one and stores it, which Metal makes, dies at the draw, naming the
/// attachment (`bfloat_loaded`: the tiles the runtime's pass leaves are not Metal's)
#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "MoltenVK: the runtime records device-pack-conversion, a death in strict mode; Mac-only test, off everywhere; ledger: MoltenVK strict deaths"
)]
fn bfloat_draws_store_metals_codes() {
    use crate::core::raise::death::DeathKind;

    const STATE: &str = "newRenderPipelineStateWithDescriptor:error:";
    const DRAW: &str = "drawPrimitives:vertexStart:vertexCount:";

    let (p, d) = (platform(), dev());
    let half = Drawn {
        fragment: "sweep_half_fragment",
        ..PLAIN
    };
    let mut differ = Vec::new();
    for format in [(26, 2), (66, 4), (116, 8)] {
        for (output, how) in [("float4", PLAIN), ("half4", half)] {
            for kind in [0, 1, 2, 3, 4] {
                let case = format!("{format:?} {output} sweep {kind}");
                *crate::answer::LAST_LINE.lock() = None;
                match catch_death(|| swept(&p, &d, false, format, kind, how)) {
                    Ok(ours) if crate::metal_matches_bytes!(case, &ours) => {}
                    Ok(_) => differ.push(format!("{case}: the bytes differ from Metal's")),
                    Err(site) => {
                        let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
                        differ.push(format!("{case}: died at {site:?}: {line}"));
                    }
                }
            }
        }
        let replace: &[(u8, u32)] = &[(2, 1), (3, 1), (4, 0), (5, 0), (6, 1), (7, 0), (8, 0)];
        let blended = Drawn {
            colour: replace,
            ..PLAIN
        };
        *crate::answer::LAST_LINE.lock() = None;
        let site = catch_death(|| swept(&p, &d, false, format, 0, blended));
        let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
        match site {
            Err(site)
                if (site.kind, site.file) == (DeathKind::Abort, STATE)
                    && line.contains("feature: Format") => {}
            Err(site) => differ.push(format!("{format:?} blended: died at {site:?}: {line}")),
            Ok(_) => differ.push(format!("{format:?} blended: drawn")),
        }
        let loaded = Drawn { load: 1, ..PLAIN };
        *crate::answer::LAST_LINE.lock() = None;
        let site = catch_death(|| swept(&p, &d, false, format, 0, loaded));
        let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
        match site {
            Err(site)
                if (site.kind, site.file) == (DeathKind::Abort, DRAW)
                    && line.contains("loads and stores the bfloat colour attachment 0") => {}
            Err(site) => differ.push(format!("{format:?} loaded: died at {site:?}: {line}")),
            Ok(_) => differ.push(format!("{format:?} loaded: drawn")),
        }
    }
    assert!(differ.is_empty(), "{} differ: {differ:#?}", differ.len());
}

/// The resolved bytes of a 256 by 256 `format` (`bytes` a texel) colour attachment of `samples`
/// samples, cleared to 0.3, after `sweep_fragment` draws the sweep `kind` into it over the viewport
/// from `offset` (of a pixel, both ways: past 0 the first row and column are partly covered, so
/// their resolve averages the draw's value and the clear's), resolved by the drawing pass where
/// `apart` is false, else by a later pass that loads the samples and draws nothing.
fn resolved<P: Platform>(
    p: &P,
    d: &P::Device,
    samples: u64,
    (format, bytes): (u64, u64),
    kind: u32,
    (offset, apart): (f64, bool),
) -> Vec<u8> {
    use crate::core::platform::resource::Library as _;
    use crate::core::platform::types::ClearColor;

    use crate::gpu::device::tests::render_value;

    /// `MTLTextureType2DMultisample`, `MTLLoadActionLoad`, `MTLStoreActionMultisampleResolve`
    /// and `MTLStoreActionStoreAndMultisampleResolve`.
    const MULTISAMPLE: u64 = 4;
    const LOAD: u64 = 1;
    const RESOLVE: u64 = 2;
    const STORE_AND_RESOLVE: u64 = 3;
    let library = d.new_library(ROUNDING).unwrap().unwrap().unwrap();
    let function = |name| library.new_function(OnFault::Die, name).unwrap().unwrap();
    let functions = [function("sweep_vertex"), function("sweep_fragment")];
    let format32 = u32::try_from(format).unwrap();
    let colour: &[(u8, u32)] = &[(0, 0), (1, format32)];
    let desc = render_value::<P>((&functions[0], &functions[1]), samples, &[colour]);
    let state = d.new_render_pipeline_state_full(&desc).unwrap().unwrap();
    let samples = d
        .new_texture(
            OnFault::Die,
            &p.texture_descriptor(&TextureDesc {
                texture_type: MULTISAMPLE,
                pixel_format: format,
                width: 256,
                height: 256,
                sample_count: samples,
                usage: TARGET,
                resource_options: Some(0x20),
                ..TextureDesc::default()
            }),
        )
        .unwrap()
        .unwrap();
    let target = texture(p, d, format, READ | TARGET, (256, 256), None);
    let kinds = buffer::<P>(d, 4, 0, &kind.to_le_bytes());
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let pass = |load, store, resolves: bool| {
        let pass = p.render_pass_descriptor();
        let attachment = ColorAttachmentSet {
            texture: Some(Some(&samples)),
            resolve_texture: resolves.then_some(Some(&target)),
            load_action: Some(load),
            store_action: Some(store),
            clear_color: Some(ClearColor {
                red: 0.3,
                green: 0.3,
                blue: 0.3,
                alpha: 0.3,
            }),
            ..ColorAttachmentSet::default()
        };
        pass.set_color_attachment(OnFault::Die, 0, &attachment)
            .unwrap();
        pass
    };
    let first = match apart {
        false => pass(CLEAR, STORE_AND_RESOLVE, true),
        true => pass(CLEAR, STORE, false),
    };
    let e = cb.render_encoder(OnFault::Die, &first).unwrap().unwrap();
    for cmd in [
        RenderCmd::SetRenderPipelineState {
            pipeline_state: Some(&state),
        },
        RenderCmd::SetViewport {
            viewport: MtlViewport {
                origin_x: offset,
                origin_y: offset,
                width: 256.0 - offset,
                height: 256.0 - offset,
                znear: 0.0,
                zfar: 1.0,
            },
        },
        RenderCmd::SetFragmentBuffer {
            buffer: Some(&kinds),
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
    if apart {
        let e = cb
            .render_encoder(OnFault::Die, &pass(LOAD, RESOLVE, true))
            .unwrap()
            .unwrap();
        e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
    }
    let length = 256 * 256 * bytes;
    let out = buffer::<P>(d, length, 0, &[]);
    let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
    blit.encode(
        OnFault::Die,
        BlitCmd::CopyTextureToBuffer {
            source_texture: &target,
            source_slice: 0,
            source_level: 0,
            source_origin: MtlOrigin::new(0, 0, 0),
            source_size: MtlSize::new(256, 256, 1),
            destination_buffer: &out,
            destination_offset: 0,
            destination_bytes_per_row: 256 * bytes,
            destination_bytes_per_image: length,
        },
    )
    .unwrap();
    blit.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    assert!(cb.error().is_none(), "{:?}", cb.error());
    let n = usize::try_from(length).unwrap();
    // SAFETY: a Shared buffer's bytes, read after the blit completed.
    unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), n) }.to_vec()
}

/// The resolved bytes of a 64 by 64 `format` (`bytes` a texel) colour attachment of
/// `samples` samples after `xr_loaded_fragment` draws two tables of `float4`s, a value a
/// pixel, into it (`tables`, [`drawn_specials`]'s or [`drawn_extended`]'s): the first over every
/// sample, the second over the
/// viewport from half a pixel in, which covers part of the first row's and column's samples,
/// so their resolve averages a value of each table. Resolved by the drawing pass's store
/// action `store` (`MultisampleResolve` or `StoreAndMultisampleResolve`), or, where `store`
/// is `Store`, by a later pass that loads the samples and draws nothing.
fn drawn_resolved<P: Platform>(
    p: &P,
    d: &P::Device,
    (format, bytes, samples): (u64, u64, u64),
    store: u64,
    tables: &[Vec<[f32; 4]>; 2],
) -> Vec<u8> {
    use crate::core::platform::resource::Library as _;

    use crate::gpu::device::tests::render_value;

    /// `MTLTextureType2DMultisample`, `MTLLoadActionLoad` and
    /// `MTLStoreActionMultisampleResolve`.
    const MULTISAMPLE: u64 = 4;
    const LOAD: u64 = 1;
    const RESOLVE: u64 = 2;
    let library = d.new_library(ROUNDING).unwrap().unwrap().unwrap();
    let function = |name| library.new_function(OnFault::Die, name).unwrap().unwrap();
    let functions = [function("sweep_vertex"), function("xr_loaded_fragment")];
    let format32 = u32::try_from(format).unwrap();
    let colour: &[(u8, u32)] = &[(0, 0), (1, format32)];
    let desc = render_value::<P>((&functions[0], &functions[1]), samples, &[colour]);
    let state = d.new_render_pipeline_state_full(&desc).unwrap().unwrap();
    let multisampled = d
        .new_texture(
            OnFault::Die,
            &p.texture_descriptor(&TextureDesc {
                texture_type: MULTISAMPLE,
                pixel_format: format,
                width: 64,
                height: 64,
                sample_count: samples,
                usage: TARGET,
                resource_options: Some(0x20),
                ..TextureDesc::default()
            }),
        )
        .unwrap()
        .unwrap();
    let target = texture(p, d, format, READ | TARGET, (64, 64), None);
    let words: Vec<u8> = (tables.iter().flatten().flatten())
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let values = buffer::<P>(d, 2 * 4096 * 16, 0, &words);
    let seeds = [1u32, 2].map(|seed| buffer::<P>(d, 4, 0, &seed.to_le_bytes()));
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let pass = |load, store, resolves: bool| {
        let pass = p.render_pass_descriptor();
        let attachment = ColorAttachmentSet {
            texture: Some(Some(&multisampled)),
            resolve_texture: resolves.then_some(Some(&target)),
            load_action: Some(load),
            store_action: Some(store),
            ..ColorAttachmentSet::default()
        };
        pass.set_color_attachment(OnFault::Die, 0, &attachment)
            .unwrap();
        pass
    };
    let first = pass(CLEAR, store, store != STORE);
    let e = cb.render_encoder(OnFault::Die, &first).unwrap().unwrap();
    e.encode(
        OnFault::Die,
        RenderCmd::SetRenderPipelineState {
            pipeline_state: Some(&state),
        },
    )
    .unwrap();
    e.encode(
        OnFault::Die,
        RenderCmd::SetFragmentBuffer {
            buffer: Some(&values),
            offset: 0,
            index: 1,
        },
    )
    .unwrap();
    for (seed, offset) in seeds.iter().zip([0.0, 0.5]) {
        for cmd in [
            RenderCmd::SetViewport {
                viewport: MtlViewport {
                    origin_x: offset,
                    origin_y: offset,
                    width: 64.0 - offset,
                    height: 64.0 - offset,
                    znear: 0.0,
                    zfar: 1.0,
                },
            },
            RenderCmd::SetFragmentBuffer {
                buffer: Some(seed),
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
    if store == STORE {
        let e = cb
            .render_encoder(OnFault::Die, &pass(LOAD, RESOLVE, true))
            .unwrap()
            .unwrap();
        e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
    }
    let length = 64 * 64 * bytes;
    let out = buffer::<P>(d, length, 0, &[]);
    let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
    blit.encode(
        OnFault::Die,
        BlitCmd::CopyTextureToBuffer {
            source_texture: &target,
            source_slice: 0,
            source_level: 0,
            source_origin: MtlOrigin::new(0, 0, 0),
            source_size: MtlSize::new(64, 64, 1),
            destination_buffer: &out,
            destination_offset: 0,
            destination_bytes_per_row: 64 * bytes,
            destination_bytes_per_image: length,
        },
    )
    .unwrap();
    blit.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    assert!(cb.error().is_none(), "{:?}", cb.error());
    let n = usize::try_from(length).unwrap();
    // SAFETY: a Shared buffer's bytes, read after the blit completed.
    unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), n) }.to_vec()
}

/// The two tables [`drawn_resolved`] draws, 4096 `float4`s each, a pixel's at `y * 64 + x`:
/// each lane one of a quiet NaN, a negative one, one of a payload, either infinity, a value
/// past the largest half, and two plain values, in another order in each lane and table, so
/// that over the first row, whose samples the resolve averages two values of, each lane
/// pairs each value with each.
fn drawn_specials() -> [Vec<[f32; 4]>; 2] {
    let specials = [
        f32::NAN,
        -f32::NAN,
        f32::from_bits(0x7fc0_1234),
        f32::INFINITY,
        f32::NEG_INFINITY,
        131_008.0,
        0.5,
        -0.25,
    ];
    // The first table's lane `c` of pixel `i`, and the second's.
    let at = |i: usize, c: usize| [(i + c) % 8, (i + i / 8 + 2 * c) % 8];
    [0, 1].map(|table| {
        (0..4096)
            .map(|i| [0, 1, 2, 3].map(|c| specials[at(i, c)[table]]))
            .collect()
    })
}

/// A multisampled float colour attachment whose samples draws wrote as NaN (quiet, negative
/// and of a payload), as either infinity, past the largest half or as plain values, two
/// values in some pixels' samples, resolves as Metal's, byte for byte ([`drawn_resolved`]):
/// `R16Float`, `RG16Float` and `RGBA16Float`, which MoltenVK averages by Metal's own resolve,
/// announcing nothing (another device resolves them by the runtime's rules of Metal's resolve),
/// and `R32Float`, which the runtime resolves by its own rule, at 2 and 4 samples, resolved by
/// the drawing pass (`MultisampleResolve`, `StoreAndMultisampleResolve`) and by a later pass
/// that loads the samples. Each case whose bytes are not Metal's is named, with the texels off.
/// `RG11B10Float`'s cases die at the blit that copies the resolve texture into a buffer,
/// naming `device-nan-encoding`: Metal keeps that 64 by 64 render target, of the descriptor's
/// default options, losslessly compressed, and its blit through the compression makes a NaN
/// field the quiet NaN, where the
/// device's copy keeps the texels' bytes, which the runtime announces at the blit and which
/// dies on every device ([`crate::answer::DIES`]; the runtime's `compressed_blit.rs`; measured
/// on the M3 Max under macOS 27.0.1, MoltenVK 1.4.2, at the runtime's 57b3b26ba0). Their
/// resolved bytes are not compared
#[test]
fn drawn_nan_and_infinite_samples_resolve_as_metals() {
    /// `MTLStoreActionMultisampleResolve` and `MTLStoreActionStoreAndMultisampleResolve`.
    const RESOLVE: u64 = 2;
    const STORE_AND_RESOLVE: u64 = 3;
    /// The blit [`drawn_resolved`] reads the resolve texture back with.
    const TO_BUFFER: &str = "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:\
                             toBuffer:destinationOffset:destinationBytesPerRow:\
                             destinationBytesPerImage:";
    let (p, d) = (platform(), dev());
    let mut differ = Vec::new();
    for samples in [2, 4] {
        for store in [RESOLVE, STORE_AND_RESOLVE, STORE] {
            for (format, bytes) in [(25, 2), (65, 4), (115, 8), (55, 4), (92, 4)] {
                let case = (format, bytes, samples);
                if format == 92 {
                    let ours = || drawn_resolved(&p, &d, case, store, &drawn_specials());
                    let line = died_at(TO_BUFFER, ours);
                    if !line.contains("(metal2vulkan-degrade-device-nan-encoding)") {
                        differ.push(format!("{format} {samples}x store {store}: {line}"));
                    }
                    continue;
                }
                let key = format!("{format} {samples}x store {store}");
                let ours = drawn_resolved(&p, &d, case, store, &drawn_specials());
                if !crate::metal_matches_bytes!(key, &ours) {
                    differ.push(format!("{key}: the texels differ from Metal's"));
                }
            }
        }
    }
    assert!(differ.is_empty(), "{differ:#?}");
}

/// The two tables [`drawn_resolved`] draws into an extended-range attachment, as
/// [`drawn_specials`]' are laid out: each lane one of eight values within the formats' range
/// (-0.75 to 1.25), in another order in each lane and table.
fn drawn_extended() -> [Vec<[f32; 4]>; 2] {
    let values = [-0.75, -0.3, 0.0, 0.1, 0.33, 0.5, 0.9, 1.25];
    let at = |i: usize, c: usize| [(i + c) % 8, (i + i / 8 + 2 * c) % 8];
    [0, 1].map(|table| {
        (0..4096)
            .map(|i| [0, 1, 2, 3].map(|c| values[at(i, c)[table]]))
            .collect()
    })
}

/// The 10-bit codes of `bytes`, texels of extended-range `format`, and whether each is alpha: a
/// `BGRA10_XR` form's four 16-bit words a texel, each code in the high 10 bits and its 6 low bits
/// padding, here kept beside the code (the code then the padding); a `BGR10_XR` form's three
/// fields of a 32-bit word, from bit 0, its top two bits padding (as a fourth value).
fn extended_codes(format: u64, bytes: &[u8]) -> Vec<(u32, u32, bool)> {
    match format {
        552 | 553 => (bytes.as_chunks::<2>().0.iter().enumerate())
            .map(|(i, w)| {
                let w = u32::from(u16::from_le_bytes(*w));
                (w >> 6, w & 0x3f, i % 4 == 3)
            })
            .collect(),
        _ => (bytes.as_chunks::<4>().0.iter())
            .flat_map(|w| {
                let w = u32::from_le_bytes(*w);
                [w & 0x3ff, (w >> 10) & 0x3ff, (w >> 20) & 0x3ff].map(|c| (c, w >> 30, false))
            })
            .collect(),
    }
}

/// A multisampled extended-range colour attachment whose samples draws wrote, two values in some
/// pixels' samples ([`drawn_resolved`] of [`drawn_extended`]'s tables), resolves as Metal's, at
/// 2 and 4 samples, by the drawing pass (`MultisampleResolve`, `StoreAndMultisampleResolve`)
/// and by a later pass that loads the samples: `BGR10_XR`'s, byte for byte, announcing nothing.
/// The runtime resolves the other forms, whose resolve Metal averages from values finer than
/// the codes they hold, by a near-equivalent, and announces `extended-range-resolve` as the
/// pass ends, which dies there, at `endEncoding`, naming it; relayed, `BGRA10_XR`'s colour is
/// Metal's and its alpha within a code, and an `_sRGB` twin's colour within 7 codes (its alpha
/// within one), the bounds the runtime measured on the M3 Max (macOS 27.0.1), every padding bit
/// Metal's
#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "MoltenVK: the runtime records device-pack-conversion, a death in strict mode; Mac-only test, off everywhere; ledger: MoltenVK strict deaths"
)]
fn drawn_extended_range_samples_resolve_as_metals() {
    /// `MTLStoreActionMultisampleResolve` and `MTLStoreActionStoreAndMultisampleResolve`.
    const RESOLVE: u64 = 2;
    const STORE_AND_RESOLVE: u64 = 3;
    let (p, d) = (platform(), dev());
    let tables = drawn_extended();
    let mut differ = Vec::new();
    for samples in [2, 4] {
        for store in [RESOLVE, STORE_AND_RESOLVE, STORE] {
            for (format, bytes, [colour, alpha]) in [
                (554, 4, [0, 0]),
                (552, 8, [0, 1]),
                (555, 4, [7, 0]),
                (553, 8, [7, 1]),
            ] {
                let case = (format, bytes, samples);
                let what = format!("{format} {samples}x store {store}");
                let ours = || drawn_resolved(&p, &d, case, store, &tables);
                *crate::answer::LAST_LINE.lock() = None;
                let strict = catch_death(ours);
                let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
                let ours = match (strict, format) {
                    (Ok(ours), 554) => ours,
                    (Err(site), 552 | 553 | 555)
                        if site.file == "endEncoding"
                            && line.contains("(metal2vulkan-degrade-extended-range-resolve)") =>
                    {
                        crate::answer::relaxed(ours)
                    }
                    (Ok(_), _) => {
                        differ.push(format!("{what}: resolved, announcing nothing"));
                        continue;
                    }
                    (Err(site), _) => {
                        differ.push(format!("{what}: died at {site:?}: {line}"));
                        continue;
                    }
                };
                // Metal's texels in full where ours differ from them, else by their digest.
                if crate::metal_matches_bytes!(what, &ours) {
                    continue;
                }
                let Some(theirs) = crate::metal_full!(what) else {
                    differ.push(format!(
                        "{what}: differs from Metal's, recorded by its digest"
                    ));
                    continue;
                };
                let codes = extended_codes(format, &ours).into_iter();
                let pairs = codes.zip(extended_codes(format, &theirs));
                let (mut worst, mut padding) = ([0, 0], 0);
                for ((o, o_pad, is_alpha), (m, m_pad, _)) in pairs {
                    let at = usize::from(is_alpha);
                    worst[at] = worst[at].max(o.abs_diff(m));
                    padding += usize::from(o_pad != m_pad);
                }
                if worst[0] > colour || worst[1] > alpha || padding != 0 {
                    differ.push(format!(
                        "{what}: colour {} and alpha {} codes off at most, {padding} paddings off",
                        worst[0], worst[1]
                    ));
                }
            }
        }
    }
    assert!(differ.is_empty(), "{differ:#?}");
}

/// How the resolve texture of [`resolved_into`] is made: a texture of its own of a storage
/// mode (`MTLResourceOptions`), which Metal may compress losslessly or not
/// (`allowGPUOptimizedContents`); one over a host IOSurface of a storage mode, presented as a
/// drawable (`setIsDrawable:` and `setFramebufferOnly:`, render target usage alone) or not; or a
/// Private shareable texture (`newSharedTextureWithDescriptor:`), or one opened from its handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Made {
    Own { options: u64, optimized: bool },
    Surface { options: u64, drawable: bool },
    Shareable { opened: bool },
}

/// A Shared texture of its own.
const SHARED: Made = Made::Own {
    options: 0,
    optimized: true,
};

/// A Private texture of its own, which Metal may compress.
const PRIVATE: Made = Made::Own {
    options: 0x20,
    optimized: true,
};

/// The resolve textures [`resolved_into`] measures plain Metal at: of its own, Shared, Managed
/// and Private (compressed where Metal may, and not), over an IOSurface, Shared and Managed,
/// each as a drawable too, and a shareable one and one opened from its handle.
const SHAPES: [Made; 10] = [
    SHARED,
    Made::Own {
        options: MANAGED,
        optimized: true,
    },
    PRIVATE,
    Made::Own {
        options: 0x20,
        optimized: false,
    },
    Made::Surface {
        options: 0,
        drawable: false,
    },
    Made::Surface {
        options: 0,
        drawable: true,
    },
    Made::Surface {
        options: MANAGED,
        drawable: false,
    },
    Made::Surface {
        options: MANAGED,
        drawable: true,
    },
    Made::Shareable { opened: false },
    Made::Shareable { opened: true },
];

/// The sizes [`resolved_into`] measures plain Metal at: 16 by 16, and 96 by 37, whose tiles
/// at its right and bottom edges it covers in part.
const SIZES: [(u64, u64); 2] = [(16, 16), (96, 37)];

/// `MTLLoadActionDontCare` and `MTLStoreActionDontCare`; `MTLLoadActionLoad`;
/// `MTLStoreActionMultisampleResolve` and `StoreAndMultisampleResolve`.
const DONT_CARE: u64 = 0;
const LOAD: u64 = 1;
const RESOLVE: u64 = 2;
const STORE_AND_RESOLVE: u64 = 3;

/// `MTLTextureUsageShaderWrite` and `PixelFormatView`.
const WRITE: u64 = 2;
const VIEW: u64 = 0x10;

/// `MTLPixelFormatDepth32Float` and `Depth32Float_Stencil8`.
const DEPTH32F: u64 = 252;
const DEPTH32F_S8: u64 = 260;

/// `MTLResourceStorageModePrivate` and `Memoryless`.
const PRIVATE_OPTIONS: u64 = 0x20;
const MEMORYLESS: u64 = 0x30;

/// A depth attachment of a pass [`resolved_into`] runs: a 4-sample texture of a format (of
/// `Depth32Float_Stencil8`, the stencil attachment too), of `MTLResourceOptions`, cleared to 1
/// or not loaded by its load action and stored by its store action.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Depth {
    format: u64,
    options: u64,
    load: u64,
    store: u64,
}

/// Another colour attachment beside the resolving one in the pass [`resolved_into`] runs: a
/// 4-sample texture of a format at an index (the resolving attachment is at 1 where this is at
/// 0, else at 0), cleared, and resolved by the resolving store action into a Shared texture of
/// its format where `resolves`, else stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Beside {
    index: u32,
    format: u64,
    resolves: bool,
}

/// What the resolving pass of [`resolved_into`] draws.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Calls {
    /// A triangle over the area [`Resolve::load`] names.
    Triangle,
    /// No draw call, nor any other command.
    Nothing,
    /// No draw call, after the pipeline state, viewport and fragment buffer a draw takes.
    StateOnly,
    /// A draw call of no vertices.
    NoVertices,
    /// A draw call of two vertices of a triangle list, no primitive.
    TwoVertices,
    /// A draw call of no instances.
    NoInstances,
    /// An indirect draw call whose arguments name no vertices.
    IndirectNoVertices,
}

/// A resolve [`resolved_into`] runs.
#[derive(Clone, Copy, Debug)]
struct Resolve {
    /// The 4-sample attachment's format.
    from: u64,
    /// The resolve texture's format, and its bytes a texel.
    into: (u64, u64),
    /// How the resolve texture is made.
    made: Made,
    /// The width and height of the attachment and the resolve texture.
    size: (u64, u64),
    /// The resolving pass's load action, the pass after one that clears the samples to another
    /// colour and stores them: `Clear`, and its triangle ([`Calls::Triangle`]) draws over the
    /// top left 8 by 16 texels, `Load`, and it draws there too, or `DontCare`, and it draws over
    /// every texel.
    load: u64,
    /// The resolving store action.
    store: u64,
    /// Whether the resolving pass sets its store action again before it ends
    /// (`setColorStoreAction:atIndex:`).
    again: bool,
    /// The usage of the attachment's texture and of the resolve texture's (a drawable's is
    /// `RenderTarget` alone, whatever this says).
    usage: (u64, u64),
    /// Whether the attachment's texture is made with `allowGPUOptimizedContents`.
    optimized: bool,
    /// The pass's depth attachment, if any.
    depth: Option<Depth>,
    /// Another colour attachment beside it, if any.
    beside: Option<Beside>,
    /// Whether the resolving pass has a rasterization rate map, of the attachment's size and
    /// one sample of 1 on each axis.
    rate_map: bool,
    /// What the resolving pass draws.
    calls: Calls,
}

impl Resolve {
    /// The resolve of a `from` attachment into `into` made as `made`, 16 by 16, by `store` in a
    /// pass of `load`: the attachment of usage `RenderTarget`, compressed where Metal may, the
    /// resolve texture of `ShaderRead` and `RenderTarget`, in a pass of no other attachment and
    /// no rate map.
    fn of(from: u64, into: (u64, u64), made: Made, (load, store): (u64, u64)) -> Resolve {
        Resolve {
            from,
            into,
            made,
            size: (16, 16),
            load,
            store,
            again: false,
            usage: (TARGET, READ | TARGET),
            optimized: true,
            depth: None,
            beside: None,
            rate_map: false,
            calls: Calls::Triangle,
        }
    }

    /// The index of the resolving attachment.
    fn index(self) -> u32 {
        match self.beside {
            Some(Beside { index: 0, .. }) => 1,
            _ => 0,
        }
    }
}

/// What [`resolved_into`] reads: the resolve texture's bytes; the samples after the resolving
/// pass, resolved into a texture of their own format by a later pass that loads them; and the
/// resolve texture's bytes of the attachment beside it that resolves, if any.
#[derive(Debug, PartialEq, Eq)]
struct Resolved {
    resolved: Vec<u8>,
    samples: Vec<u8>,
    beside: Option<Vec<u8>>,
}

/// The states [`resolved_into`] draws with on a device, each made once ([`sweep_state`]), by
/// the colour attachments' indices and formats and the depth attachment's format.
struct States<'a, P: Platform> {
    d: &'a P::Device,
    made: std::cell::RefCell<StateMap<P>>,
}

/// The states of [`States`], by what they were made for.
type StateMap<P> =
    std::collections::HashMap<(Vec<(u32, u64)>, Option<u64>), <P as Platform>::RenderPipeline>;

impl<'a, P: Platform> States<'a, P> {
    fn new(d: &'a P::Device) -> States<'a, P> {
        States {
            d,
            made: std::cell::RefCell::new(StateMap::<P>::new()),
        }
    }

    /// The state that draws into the colour attachments of `c`'s resolving pass.
    fn of(&self, c: Resolve) -> P::RenderPipeline {
        let mut colours = vec![(c.index(), c.from)];
        colours.extend(c.beside.map(|b| (b.index, b.format)));
        colours.sort_unstable();
        let key = (colours, c.depth.map(|z| z.format));
        let mut made = self.made.borrow_mut();
        let state = made
            .entry(key.clone())
            .or_insert_with(|| sweep_state::<P>(self.d, &key.0, key.1));
        state.clone_ref()
    }
}

/// The state [`resolved_into`] draws with: `sweep_vertex`, and `sweep_fragment` into a 4-sample
/// colour attachment 0 of the one format of `colours`, else `sweep_four_fragment` into each of
/// `colours` (index and format); `depth` the depth attachment's format, a stencil one's the
/// stencil attachment's too.
fn sweep_state<P: Platform>(
    d: &P::Device,
    colours: &[(u32, u64)],
    depth: Option<u64>,
) -> P::RenderPipeline {
    use crate::core::platform::resource::Library as _;

    use crate::gpu::device::tests::render_value;

    let library = d.new_library(ROUNDING).unwrap().unwrap().unwrap();
    let function = |name| library.new_function(OnFault::Die, name).unwrap().unwrap();
    let fragment = match colours {
        [(0, _)] => "sweep_fragment",
        _ => "sweep_four_fragment",
    };
    let functions = [function("sweep_vertex"), function(fragment)];
    let elements: Vec<[(u8, u32); 2]> = colours
        .iter()
        .map(|&(index, format)| [(0, index), (1, u32::try_from(format).unwrap())])
        .collect();
    let elements: Vec<&[(u8, u32)]> = elements.iter().map(|e| &e[..]).collect();
    let mut desc = render_value::<P>((&functions[0], &functions[1]), 4, &elements);
    if let Some(format) = depth {
        desc.depth_pixel_format = format;
        if depth == Some(DEPTH32F_S8) {
            desc.stencil_pixel_format = format;
        }
    }
    d.new_render_pipeline_state_full(&desc).unwrap().unwrap()
}

/// `cmd` in a blit encoder of its own on `cb`.
fn blit_one<P: Platform>(cb: &P::CommandBuffer, cmd: BlitCmd<'_, P>) {
    let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
    blit.encode(OnFault::Die, cmd).unwrap();
    blit.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
}

/// The copy of a `width` × `height` texture of `bytes` a texel into `out`, rows tight.
fn copy_out<'a, P: Platform>(
    texture: &'a P::Texture,
    (width, height, bytes): (u64, u64, u64),
    out: &'a P::Buffer,
) -> BlitCmd<'a, P> {
    BlitCmd::CopyTextureToBuffer {
        source_texture: texture,
        source_slice: 0,
        source_level: 0,
        source_origin: MtlOrigin::new(0, 0, 0),
        source_size: MtlSize::new(width, height, 1),
        destination_buffer: out,
        destination_offset: 0,
        destination_bytes_per_row: width * bytes,
        destination_bytes_per_image: width * height * bytes,
    }
}

/// A rate map of a `width` by `height` screen, with one layer of one sample of 1 on each axis.
fn even_rate_map<P: Platform>(p: &P, d: &P::Device, (width, height): (u64, u64)) -> P::RateMap {
    use crate::core::platform::rate_map::{RateMapDescriptor as _, RateMapLayerDescriptor as _};
    use crate::core::platform::values::RateMapDesc;

    let m = p.rate_map_descriptor(&RateMapDesc {
        screen_size: MtlSize::new(width, height, 1),
        ..RateMapDesc::default()
    });
    let layer = p
        .new_rate_map_layer(OnFault::Die, MtlSize::new(1, 1, 0))
        .unwrap()
        .unwrap();
    for at in [
        layer.horizontal_sample_storage(),
        layer.vertical_sample_storage(),
    ] {
        // SAFETY: the layer's storage holds one sample on each axis.
        unsafe { *at.as_ptr::<f32>() = 1.0 };
    }
    m.set_layer(OnFault::Die, &layer, 0).unwrap();
    d.new_rate_map(&m).unwrap().unwrap()
}

/// What a resolve texture made as `c.made` says holds, prefilled with 0xa5 (a surface's
/// through its base address, another by a blit), after a 4-sample `c.from` colour attachment
/// of its size, cleared to (0.25, 0.5, 0.75, 1) and stored by an earlier pass, is drawn by
/// `states`' state ([`sweep_state`]) as `c.calls` says and resolved into it by `c.store` in a
/// pass of `c.load`, as [`Resolve`] says, and what its samples and the resolve texture of an
/// attachment beside it
/// hold after that pass ([`Resolved`]). The resolving pass clears to (0.75, 0.25, 0.5, 1), an
/// attachment beside to (0.125, 0.25, 0.5, 1). A surface's bytes are read through its base
/// address, as the guest reads them; another's by a blit. It runs in an autorelease pool of its
/// own: what Metal autoreleased (each command buffer and encoder, which a dev build cannot elide,
/// and through them the case's queue, textures and buffers) goes at its end, where the test
/// thread, which pops no pool, would keep every case's until the test ends.
fn resolved_into<P: Platform<Surface = crate::os::OsSurface>>(
    p: &P,
    d: &P::Device,
    states: &States<'_, P>,
    c: Resolve,
) -> Resolved {
    use crate::core::platform::os::surface::{Surface as _, SurfaceSpec};
    use crate::core::platform::values::{DepthAttachmentSet, PassSet, StencilAttachmentSet};

    /// `kIOSurfaceLockReadOnly`.
    const READ_ONLY: u32 = 1;
    let _pool = crate::core::rt::Pool::here();
    let (into, bytes) = c.into;
    let (width, height) = c.size;
    let length = width * height * bytes;
    let n = usize::try_from(length).unwrap();
    let prefill = vec![0xa5u8; n];
    let multisample = |format, usage, options, optimized: bool| {
        let desc = p.texture_descriptor(&TextureDesc {
            texture_type: 4,
            pixel_format: format,
            width,
            height,
            sample_count: 4,
            usage,
            resource_options: Some(options),
            allow_gpu_optimized_contents: (!optimized).then_some(false),
            ..TextureDesc::default()
        });
        d.new_texture(OnFault::Die, &desc).unwrap().unwrap()
    };
    let samples = multisample(c.from, c.usage.0, PRIVATE_OPTIONS, c.optimized);
    let (options, usage, optimized, drawable) = match c.made {
        Made::Own { options, optimized } => (options, c.usage.1, optimized, false),
        Made::Surface {
            options,
            drawable: false,
        } => (options, c.usage.1, true, false),
        Made::Surface {
            options,
            drawable: true,
        } => (options, TARGET, true, true),
        Made::Shareable { .. } => (PRIVATE_OPTIONS, c.usage.1, true, false),
    };
    let desc = p.texture_descriptor(&TextureDesc {
        pixel_format: into,
        width,
        height,
        usage,
        resource_options: Some(options),
        allow_gpu_optimized_contents: (!optimized).then_some(false),
        framebuffer_only: drawable.then_some(true),
        is_drawable: drawable.then_some(true),
        ..TextureDesc::default()
    });
    let surface = matches!(c.made, Made::Surface { .. }).then(|| {
        let ostype = if bytes == 8 { *b"RGhA" } else { *b"w30r" };
        let s = p
            .surfaces()
            .create(&SurfaceSpec {
                pixel_format: u32::from_be_bytes(ostype),
                alloc_size: length,
                width,
                height,
                bytes_per_element: bytes,
                bytes_per_row: width * bytes,
                element_width: 1,
                element_height: 1,
                ..SurfaceSpec::default()
            })
            .unwrap();
        assert!(s.lock(0).is_success());
        let base = s.base_address().as_ptr::<u8>();
        // SAFETY: the surface's first `n` bytes, locked, which no texture uses yet.
        unsafe { std::ptr::copy_nonoverlapping(prefill.as_ptr(), base, n) };
        assert!(s.unlock(0).is_success());
        s
    });
    let shareable = matches!(c.made, Made::Shareable { .. })
        .then(|| d.new_shared_texture(&desc).unwrap().unwrap());
    let target = match (&surface, &shareable) {
        (Some(s), _) => d
            .new_texture_on_surface(&desc, Some(s), 0)
            .unwrap()
            .unwrap(),
        (None, Some(t)) if c.made == (Made::Shareable { opened: true }) => {
            let handle = t.new_shared_texture_handle().unwrap().unwrap();
            d.open_shared_texture(Some(&handle), 0).unwrap().unwrap()
        }
        (None, Some(t)) => t.clone_ref(),
        (None, None) => d.new_texture(OnFault::Die, &desc).unwrap().unwrap(),
    };
    let beside = c.beside.map(|b| {
        let other = multisample(b.format, TARGET, PRIVATE_OPTIONS, true);
        let resolve = b.resolves.then(|| {
            let texels = Some((
                &prefill[..usize::try_from(width * height * 4).unwrap()],
                width * 4,
            ));
            texture(p, d, b.format, READ | TARGET, c.size, texels)
        });
        (b, other, resolve)
    });
    let depth = c
        .depth
        .map(|z| (z, multisample(z.format, TARGET, z.options, true)));
    let map = c.rate_map.then(|| even_rate_map(p, d, c.size));
    let kinds = buffer::<P>(d, 4, 0, &0u32.to_le_bytes());
    let source = buffer::<P>(d, length, 0, &prefill);
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    if surface.is_none() {
        blit_one::<P>(
            &cb,
            BlitCmd::CopyBufferToTexture {
                source_buffer: &source,
                source_offset: 0,
                source_bytes_per_row: width * bytes,
                source_bytes_per_image: length,
                source_size: MtlSize::new(width, height, 1),
                destination_texture: &target,
                destination_slice: 0,
                destination_level: 0,
                destination_origin: MtlOrigin::new(0, 0, 0),
            },
        );
    }
    let colour = |red, green, blue| ClearColor {
        red,
        green,
        blue,
        alpha: 1.0,
    };
    // The samples, cleared and stored, by a pass of their own.
    let earlier = p.render_pass_descriptor();
    let attachment = ColorAttachmentSet {
        texture: Some(Some(&samples)),
        load_action: Some(CLEAR),
        store_action: Some(STORE),
        clear_color: Some(colour(0.25, 0.5, 0.75)),
        ..ColorAttachmentSet::default()
    };
    earlier
        .set_color_attachment(OnFault::Die, 0, &attachment)
        .unwrap();
    let e = cb.render_encoder(OnFault::Die, &earlier).unwrap().unwrap();
    e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
    // The resolving pass.
    let index = c.index();
    let resolving = p.render_pass_descriptor();
    let attachment = ColorAttachmentSet {
        texture: Some(Some(&samples)),
        resolve_texture: Some(Some(&target)),
        load_action: Some(c.load),
        store_action: Some(c.store),
        clear_color: Some(colour(0.75, 0.25, 0.5)),
        ..ColorAttachmentSet::default()
    };
    resolving
        .set_color_attachment(OnFault::Die, u64::from(index), &attachment)
        .unwrap();
    if let Some((b, other, resolve)) = &beside {
        let attachment = ColorAttachmentSet {
            texture: Some(Some(other)),
            resolve_texture: resolve.as_ref().map(Some),
            load_action: Some(CLEAR),
            store_action: Some(if b.resolves { c.store } else { STORE }),
            clear_color: Some(colour(0.125, 0.25, 0.5)),
            ..ColorAttachmentSet::default()
        };
        resolving
            .set_color_attachment(OnFault::Die, u64::from(b.index), &attachment)
            .unwrap();
    }
    if let Some((z, texture)) = &depth {
        resolving.set_depth_attachment(&DepthAttachmentSet {
            texture: Some(Some(texture)),
            load_action: Some(z.load),
            store_action: Some(z.store),
            clear_depth: Some(1.0),
            ..DepthAttachmentSet::default()
        });
        if z.format == DEPTH32F_S8 {
            resolving.set_stencil_attachment(&StencilAttachmentSet {
                texture: Some(Some(texture)),
                load_action: Some(z.load),
                store_action: Some(z.store),
                clear_stencil: Some(0),
                ..StencilAttachmentSet::default()
            });
        }
    }
    if let Some(map) = &map {
        resolving.set_pass(&PassSet {
            rasterization_rate_map: Some(Some(map)),
            ..Default::default()
        });
    }
    let e = cb
        .render_encoder(OnFault::Die, &resolving)
        .unwrap()
        .unwrap();
    let state = states.of(c);
    let drawn = match c.load {
        DONT_CARE => (width, height),
        _ => (8, 16),
    };
    let again = c.again.then_some(RenderCmd::SetColorStoreAction {
        store_action: c.store,
        color_attachment_index: u64::from(index),
    });
    // `MTLDrawPrimitivesIndirectArguments` of no vertices and one instance.
    let arguments = buffer::<P>(d, 16, 0, &[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    let call = match c.calls {
        Calls::Nothing | Calls::StateOnly => None,
        Calls::Triangle | Calls::NoVertices | Calls::TwoVertices => {
            Some(RenderCmd::DrawPrimitives {
                primitive_type: 3,
                vertex_start: 0,
                vertex_count: match c.calls {
                    Calls::NoVertices => 0,
                    Calls::TwoVertices => 2,
                    _ => 3,
                },
            })
        }
        Calls::NoInstances => Some(RenderCmd::DrawPrimitivesInstances {
            primitive_type: 3,
            vertex_start: 0,
            vertex_count: 3,
            instance_count: 0,
        }),
        Calls::IndirectNoVertices => Some(RenderCmd::DrawPrimitivesIndirect {
            primitive_type: 3,
            indirect_buffer: &arguments,
            indirect_buffer_offset: 0,
        }),
    };
    let draw = [
        RenderCmd::SetRenderPipelineState {
            pipeline_state: Some(&state),
        },
        RenderCmd::SetViewport {
            viewport: MtlViewport {
                origin_x: 0.0,
                origin_y: 0.0,
                width: f64::from(u32::try_from(drawn.0).unwrap()),
                height: f64::from(u32::try_from(drawn.1).unwrap()),
                znear: 0.0,
                zfar: 1.0,
            },
        },
        RenderCmd::SetFragmentBuffer {
            buffer: Some(&kinds),
            offset: 0,
            index: 0,
        },
    ];
    let set = call.is_some() || c.calls == Calls::StateOnly;
    let draw = (set.then_some(draw).into_iter().flatten()).chain(call);
    for cmd in draw.chain(again).chain([RenderCmd::EndEncoding]) {
        e.encode(OnFault::Die, cmd).unwrap();
    }
    // The samples, resolved into a texture of their format by a later pass that loads them.
    let check = texture(p, d, c.from, READ | TARGET, c.size, None);
    let later = p.render_pass_descriptor();
    let attachment = ColorAttachmentSet {
        texture: Some(Some(&samples)),
        resolve_texture: Some(Some(&check)),
        load_action: Some(LOAD),
        store_action: Some(RESOLVE),
        ..ColorAttachmentSet::default()
    };
    later
        .set_color_attachment(OnFault::Die, 0, &attachment)
        .unwrap();
    let e = cb.render_encoder(OnFault::Die, &later).unwrap().unwrap();
    e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
    let texels = width * height * 4;
    let checked = buffer::<P>(d, texels, 0, &[]);
    blit_one::<P>(&cb, copy_out::<P>(&check, (width, height, 4), &checked));
    let besides = beside.as_ref().and_then(|(_, _, resolve)| {
        let resolve = resolve.as_ref()?;
        let out = buffer::<P>(d, texels, 0, &[]);
        blit_one::<P>(&cb, copy_out::<P>(resolve, (width, height, 4), &out));
        Some(out)
    });
    let out = buffer::<P>(d, length, 0, &[]);
    if surface.is_none() {
        blit_one::<P>(&cb, copy_out::<P>(&target, (width, height, bytes), &out));
    }
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    assert!(cb.error().is_none(), "{c:?}: {:?}", cb.error());
    let read = |b: &P::Buffer, n: usize| {
        // SAFETY: a Shared buffer's first `n` bytes, read after the command buffer completed.
        unsafe { std::slice::from_raw_parts(b.contents().as_ptr::<u8>(), n) }.to_vec()
    };
    let resolved = match &surface {
        Some(s) => {
            assert!(s.lock(READ_ONLY).is_success());
            // SAFETY: the surface's first `n` bytes, locked, after the command buffer
            // completed.
            let bytes = unsafe { std::slice::from_raw_parts(s.base_address().as_ptr::<u8>(), n) };
            let bytes = bytes.to_vec();
            assert!(s.unlock(READ_ONLY).is_success());
            bytes
        }
        None => read(&out, n),
    };
    let texels = usize::try_from(texels).unwrap();
    Resolved {
        resolved,
        samples: read(&checked, texels),
        beside: besides.map(|b| read(&b, texels)),
    }
}

/// The death of `f`, which must abort at `sel`, and its line.
fn died_at<T>(sel: &str, f: impl FnOnce() -> T) -> String {
    use crate::core::raise::death::DeathKind;
    *crate::answer::LAST_LINE.lock() = None;
    let Err(site) = catch_death(f) else {
        panic!("no death at {sel}");
    };
    let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
    assert_eq!((site.kind, site.file), (DeathKind::Abort, sel), "{line}");
    line
}

/// A 16 by 16 4-sample 2D multisample texture of `format`, of `MTLResourceOptions` `options`.
fn multisampled<P: Platform>(p: &P, d: &P::Device, format: u64, options: u64) -> P::Texture {
    let desc = p.texture_descriptor(&TextureDesc {
        texture_type: 4,
        pixel_format: format,
        width: 16,
        height: 16,
        sample_count: 4,
        usage: TARGET,
        resource_options: Some(options),
        ..TextureDesc::default()
    });
    d.new_texture(OnFault::Die, &desc).unwrap().unwrap()
}

/// The pass of `samples` as colour attachment 0, cleared and stored by `store` into `resolve`
/// at its level and slice `at`.
fn resolve_pass<P: Platform>(
    p: &P,
    (samples, store): (&P::Texture, u64),
    resolve: &P::Texture,
    (level, slice): (u64, u64),
) -> P::RenderPassDescriptor {
    let pass = p.render_pass_descriptor();
    let attachment = ColorAttachmentSet {
        texture: Some(Some(samples)),
        resolve_texture: Some(Some(resolve)),
        resolve_level: Some(level),
        resolve_slice: Some(slice),
        load_action: Some(CLEAR),
        store_action: Some(store),
        ..ColorAttachmentSet::default()
    };
    pass.set_color_attachment(OnFault::Die, 0, &attachment)
        .unwrap();
    pass
}

/// The encoder of `pass`, open on the command buffer it returns.
fn encoder_on<P: Platform>(
    d: &P::Device,
    pass: &P::RenderPassDescriptor,
) -> (P::CommandBuffer, P::RenderEncoder) {
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let e = cb.render_encoder(OnFault::Die, pass).unwrap().unwrap();
    (cb, e)
}

/// A 4-sample `from` colour attachment, Private and of its own, cleared and stored by `store`
/// into `resolve` at its level and slice `at`, and the encoder of its pass, open on the command
/// buffer it returns.
fn resolving_pass<P: Platform>(
    p: &P,
    d: &P::Device,
    (from, store): (u64, u64),
    resolve: &P::Texture,
    at: (u64, u64),
) -> (P::CommandBuffer, P::RenderEncoder) {
    let samples = multisampled(p, d, from, PRIVATE_OPTIONS);
    encoder_on::<P>(d, &resolve_pass(p, (&samples, store), resolve, at))
}

/// A 4-sample `BGRA8Unorm` or `BGRA8Unorm_sRGB` colour attachment, cleared, drawn and resolved
/// into a texture of another format ([`resolved_into`]) by `MultisampleResolve` and
/// `StoreAndMultisampleResolve`, by its pass and by a later one that loads its samples: into
/// `RGBA16Float`, `BGR10_XR` and `BGR10_XR_sRGB`, which plain Metal leaves untouched, of every
/// shape of [`SHAPES`] at each of [`SIZES`], the pass is made without the resolve and the
/// texture keeps its bytes, as Metal's, and the samples hold what Metal's hold:
/// `StoreAndMultisampleResolve` stores them as `Store` does, and `MultisampleResolve` leaves
/// them as they were before the pass, as `DontCare` does ([`untouched_resolves`]: macOS
/// 26.6.2's Maps); the resolving action set again before the pass ends encodes
/// nothing, as Metal's. So do they, at 16 by 16 into a Shared texture of its own (and a Private
/// one, for the resolve texture's usages), after `Clear`, `Load` and `DontCare`: of the resolve
/// texture's usage `RenderTarget`, or with `ShaderWrite`, `PixelFormatView` or both beside
/// `ShaderRead`; of the attachment's usage `ShaderRead` or `PixelFormatView` beside
/// `RenderTarget`, or without `allowGPUOptimizedContents`; beside a depth attachment
/// (`Depth32Float` memoryless and not stored, or Private and stored, and
/// `Depth32Float_Stencil8` as both depth and stencil attachment, memoryless, or Private, cleared
/// or not loaded, and not stored); beside another colour attachment resolving into its format
/// (`RGBA8Unorm` at 0 or 1, `BGRA8Unorm` at 2) or storing (`R32Float` at 2), whose resolve
/// texture holds what Metal's holds; and in a pass of a rate map of one sample of 1 on each axis.
/// So do they in a pass that draws nothing, after each load action, by both actions, the
/// attachment compressed where Metal may and not: with no draw call, into each format, 16 by 16
/// and 96 by 37 into a Shared texture of its own and 16 by 16 into a Private one; with only the
/// state a draw takes, or a draw call of no vertices, of no instances, of two vertices of a
/// triangle list or indirect of no vertices, into `RGBA16Float` at 16 by 16, but a `DontCare`
/// load stored by `StoreAndMultisampleResolve` under the last two, where the runtime's
/// `DontCareLoadTiles` keeps any pass that loads by `DontCare` and stores from Metal's samples.
/// And so do they at 1920 by 1080 into each format, drawn, after `Clear`, `Load` and
/// `DontCare`, and at 1023 by 771 into `RGBA16Float`, cleared with no draw call, and drawn after
/// `DontCare`.
/// Into `RGB10A2Unorm`, `BGR10A2Unorm`, `BGRA10_XR` and `BGRA10_XR_sRGB`, where Metal writes into
/// it, the pass dies at its creation naming both formats and the shape Metal was measured at
/// ([`refused_resolve_format`]); into the other attachment format of the pair, the twin, the
/// resolve writes, its bytes Metal's, in every shape of [`SHAPES`], which shows each read sees a
/// resolve's writes (the samples `MultisampleResolve` leaves there:
/// `samples_resolved_into_the_twin_are_metals`)
#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "MoltenVK: the runtime records device-rounded-resolve, a death in strict mode; Mac-only test, off everywhere; ledger: MoltenVK strict deaths"
)]
fn resolves_into_other_formats_are_metals_or_die_named() {
    let (p, d) = (platform(), dev());
    let untouched = [(115, 8), (554, 4), (555, 4)];
    let written = [(90, 4), (94, 4), (552, 8), (553, 8)];
    let kept = |bytes: &[u8]| bytes.iter().all(|&b| b == 0xa5);
    // Whether every texel of the samples is the same and not 0: the earlier pass's clear.
    let untouched_samples = |bytes: &[u8]| {
        let mut texels = bytes.chunks(4);
        let first = texels.next().unwrap_or_default();
        first != [0; 4] && texels.all(|t| t == first)
    };
    // The first bytes of each, and whether the resolve texture kept its 0xa5s.
    let shown = |r: &Resolved| {
        let head = |b: &[u8]| b.iter().take(8).copied().collect::<Vec<u8>>();
        format!(
            "kept {}, {:02x?}, samples {:02x?}, beside {:02x?}",
            kept(&r.resolved),
            head(&r.resolved),
            head(&r.samples),
            r.beside.as_deref().map(head)
        )
    };
    let ours = States::<VulkanPlatform>::new(&d);
    let mut differ = Vec::new();
    let raw = |r: &Resolved| (r.resolved.clone(), r.samples.clone(), r.beside.clone());
    let mut both = |c: Resolve, keeps: bool| {
        let resolved = resolved_into(&p, &d, &ours, c);
        let key = format!("{c:?}");
        if !crate::metal_matches_raw!(key, raw(&resolved)) {
            differ.push(format!("{c:?}: ours {}, not Metal's", shown(&resolved)));
        }
        if kept(&resolved.resolved) != keeps {
            differ.push(format!("{c:?}: ours {}", shown(&resolved)));
        }
        // Where the texture keeps its bytes, `MultisampleResolve` leaves the samples as the
        // earlier pass stored them, cleared alike, and `StoreAndMultisampleResolve` stores what
        // the pass drew: some texels cleared and some drawn, or, drawn over every texel, 0. A
        // pass that draws nothing leaves them alike either way, the earlier clear or its own.
        let drawn = c.calls == Calls::Triangle;
        if keeps && untouched_samples(&resolved.samples) != (c.store == RESOLVE || !drawn) {
            differ.push(format!("{c:?}: ours {}", shown(&resolved)));
        }
    };
    let actions = [CLEAR, LOAD]
        .into_iter()
        .flat_map(|load| [RESOLVE, STORE_AND_RESOLVE].map(|store| (load, store)));
    for from in [80, 81] {
        let twin = if from == 80 { 81 } else { 80 };
        for (load, store) in actions.clone() {
            let shapes = SHAPES.into_iter().flat_map(|m| untouched.map(|i| (m, i)));
            for ((made, into), size) in shapes.flat_map(|c| SIZES.map(|s| (c, s))) {
                let agains: &[bool] = if made == SHARED {
                    &[false, true]
                } else {
                    &[false]
                };
                for &again in agains {
                    let c = Resolve {
                        size,
                        again,
                        ..Resolve::of(from, into, made, (load, store))
                    };
                    both(c, true);
                }
            }
            if store == RESOLVE {
                for made in SHAPES {
                    both(Resolve::of(from, (twin, 4), made, (load, store)), false);
                }
            }
        }
    }
    // Each shape beside the base one, at 16 by 16, after each load action.
    let depth = |format, options, load, store| {
        Some(Depth {
            format,
            options,
            load,
            store,
        })
    };
    let beside = |index, format, resolves| {
        Some(Beside {
            index,
            format,
            resolves,
        })
    };
    let resolve_usages = [
        TARGET,
        READ | WRITE | TARGET,
        READ | TARGET | VIEW,
        READ | WRITE | TARGET | VIEW,
    ];
    let attachment_usages = [READ | TARGET, TARGET | VIEW];
    let depths = [
        depth(DEPTH32F, MEMORYLESS, CLEAR, DONT_CARE),
        depth(DEPTH32F, PRIVATE_OPTIONS, CLEAR, STORE),
        depth(DEPTH32F_S8, MEMORYLESS, CLEAR, DONT_CARE),
        depth(DEPTH32F_S8, PRIVATE_OPTIONS, CLEAR, DONT_CARE),
        depth(DEPTH32F_S8, PRIVATE_OPTIONS, DONT_CARE, DONT_CARE),
    ];
    let besides = [
        beside(1, RGBA8, true),
        beside(0, RGBA8, true),
        beside(2, BGRA8, true),
        beside(2, R32F, false),
    ];
    for from in [80, 81] {
        for (load, store) in [CLEAR, LOAD, DONT_CARE]
            .into_iter()
            .flat_map(|load| [RESOLVE, STORE_AND_RESOLVE].map(|store| (load, store)))
        {
            for into in untouched {
                let base = Resolve::of(from, into, SHARED, (load, store));
                let mut cases = vec![base];
                for made in [SHARED, PRIVATE] {
                    for usage in resolve_usages {
                        cases.push(Resolve {
                            made,
                            usage: (TARGET, usage),
                            ..base
                        });
                    }
                }
                for usage in attachment_usages {
                    cases.push(Resolve {
                        usage: (usage, READ | TARGET),
                        ..base
                    });
                }
                cases.push(Resolve {
                    optimized: false,
                    ..base
                });
                cases.extend(depths.map(|depth| Resolve { depth, ..base }));
                cases.extend(besides.map(|beside| Resolve { beside, ..base }));
                cases.push(Resolve {
                    rate_map: true,
                    ..base
                });
                for c in cases {
                    both(c, true);
                }
            }
        }
    }
    // Passes that draw nothing, after each load action, by both actions, the attachment
    // compressed where Metal may and not: with no draw call into each format, 16 by 16 and 96 by
    // 37 into a Shared texture of its own and 16 by 16 into a Private one; and into
    // `RGBA16Float`, 16 by 16, with only the state a draw takes or a draw call that draws
    // nothing.
    let nothing = [
        Calls::StateOnly,
        Calls::NoVertices,
        Calls::NoInstances,
        Calls::TwoVertices,
        Calls::IndirectNoVertices,
    ];
    for from in [80, 81] {
        for (load, store) in [CLEAR, LOAD, DONT_CARE]
            .into_iter()
            .flat_map(|load| [RESOLVE, STORE_AND_RESOLVE].map(|store| (load, store)))
        {
            for optimized in [true, false] {
                for into in untouched {
                    for (size, made) in
                        [((16, 16), SHARED), ((96, 37), SHARED), ((16, 16), PRIVATE)]
                    {
                        both(
                            Resolve {
                                size,
                                optimized,
                                calls: Calls::Nothing,
                                ..Resolve::of(from, into, made, (load, store))
                            },
                            true,
                        );
                    }
                }
                for calls in nothing {
                    // A `DontCare` load stored by `StoreAndMultisampleResolve` under a draw call
                    // that draws nothing on the GPU keeps the samples on Metal, where the
                    // runtime's `Store` clears their tiles to zero: its `DontCareLoadTiles`,
                    // announced and relayed for any pass that loads by `DontCare` and stores,
                    // the drop or not (`unmeasured_shape`).
                    let on_the_gpu =
                        matches!(calls, Calls::TwoVertices | Calls::IndirectNoVertices);
                    if (load, store) == (DONT_CARE, STORE_AND_RESOLVE) && on_the_gpu {
                        continue;
                    }
                    both(
                        Resolve {
                            optimized,
                            calls,
                            ..Resolve::of(from, (115, 8), SHARED, (load, store))
                        },
                        true,
                    );
                }
            }
        }
    }
    // Larger sizes, into a Shared texture of its own: 1920 by 1080 into each format, drawn,
    // after `Clear` by `MultisampleResolve`, `Load` by `StoreAndMultisampleResolve` and
    // `DontCare` by `MultisampleResolve`; 1023 by 771 into `RGBA16Float`, after `Clear` by
    // `MultisampleResolve` with no draw call, and after `DontCare` by
    // `StoreAndMultisampleResolve`, drawn.
    for from in [80, 81] {
        for into in untouched {
            for actions in [
                (CLEAR, RESOLVE),
                (LOAD, STORE_AND_RESOLVE),
                (DONT_CARE, RESOLVE),
            ] {
                let c = Resolve::of(from, into, SHARED, actions);
                both(
                    Resolve {
                        size: (1920, 1080),
                        ..c
                    },
                    true,
                );
            }
        }
        for (actions, calls) in [
            ((CLEAR, RESOLVE), Calls::Nothing),
            ((DONT_CARE, STORE_AND_RESOLVE), Calls::Triangle),
        ] {
            let c = Resolve::of(from, (115, 8), SHARED, actions);
            both(
                Resolve {
                    size: (1023, 771),
                    calls,
                    ..c
                },
                true,
            );
        }
    }
    for (from, (load, store), into) in [80, 81]
        .into_iter()
        .flat_map(|from| actions.clone().map(move |a| (from, a)))
        .flat_map(|(from, a)| written.map(|into| (from, a, into)))
    {
        let c = Resolve::of(from, into, SHARED, (load, store));
        let line = died_at("renderCommandEncoderWithDescriptor:", || {
            resolved_into(&p, &d, &ours, c)
        });
        let format = PixelFormat::try_from(u32::try_from(into.0).unwrap()).unwrap();
        let named = format!("into a texture of {}", format.name());
        assert!(line.contains(&named), "{c:?}: {line}");
        let measured = "plain Metal writes into such a texture, measured from a Private 2D \
                        multisample texture of its own into a Shared single-level 2D texture of \
                        its own, both 16 by 16";
        assert!(line.contains(measured), "{c:?}: {line}");
    }
    assert!(differ.is_empty(), "{differ:#?}");
}

/// A 4-sample `BGRA8Unorm` or `BGRA8Unorm_sRGB` colour attachment, cleared and stored by an
/// earlier pass, then cleared or loaded, drawn and resolved into a texture of the other format of
/// the pair by `MultisampleResolve`, at 16 by 16 into each shape of [`SHAPES`]
/// ([`resolved_into`]), keeps the samples the earlier pass stored, as Metal's does (measured
/// beside them, M3 Max, macOS 27.0.1, 2026-10-25). The runtime resolves into the twin by a
/// resolve of its own, whose texels are Metal's
/// (`resolves_into_other_formats_are_metals_or_die_named`)
#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "MoltenVK: the runtime records device-rounded-resolve, a death in strict mode; Mac-only test, off everywhere; ledger: MoltenVK strict deaths"
)]
fn samples_resolved_into_the_twin_are_metals() {
    let (p, d) = (platform(), dev());
    let ours = States::<VulkanPlatform>::new(&d);
    let mut differ = Vec::new();
    for (from, twin) in [(80, 81), (81, 80)] {
        for load in [CLEAR, LOAD] {
            for made in SHAPES {
                let c = Resolve::of(from, (twin, 4), made, (load, RESOLVE));
                let key = format!("{c:?}");
                let o = resolved_into(&p, &d, &ours, c);
                if !crate::metal_matches_bytes!(key, &o.samples) {
                    let head = |b: &[u8]| b.iter().take(8).copied().collect::<Vec<u8>>();
                    let o = head(&o.samples);
                    differ.push(format!("{c:?}: ours {o:02x?}, not Metal's"));
                }
            }
        }
    }
    assert!(differ.is_empty(), "{differ:#?}");
}

/// A resolve into a texture of another format the runtime does not resolve into dies at the
/// encoder's creation, named with what plain Metal was measured to do there: a pair Metal is
/// not measured at (a 4-sample `RGBA8Unorm` or 2-sample `BGRA8Unorm` attachment into
/// `RGBA16Float`, a 4-sample `RGBA16Float` one into `RGBA32Float`); and a pair Metal leaves
/// untouched ([`measured_pair`]), a 4-sample `BGRA8Unorm_sRGB` attachment into `RGBA16Float`,
/// in a shape not measured ([`unmeasured_shape`]), whatever the runtime refuses of it, named
/// with what lies outside: a resolve texture of another size, level 1 of a texture whose level
/// 1 is the attachment's size, a 2D array's slice 1, a memoryless texture, a Private texture
/// over an IOSurface, a texture over a plane of a surface with planes, a view, a texture over a
/// buffer, and one in a heap; an attachment memoryless, in a heap, or with store action
/// options; a depth attachment that resolves; another colour attachment that resolves into a
/// texture of another format the runtime does not resolve into; and a rate map the runtime does
/// not take as the identity, a mutable one not yet reset and one whose rates vary. So does one
/// by a store action left `Unknown`, which the runtime refuses as the pass is made
#[test]
fn untouched_resolves_in_shapes_not_measured_die_named() {
    use crate::core::platform::os::surface::{SurfacePlaneSpec, SurfaceSpec};
    use crate::core::platform::rate_map::{RateMapDescriptor as _, RateMapLayerDescriptor as _};
    use crate::core::platform::resource::Heap as _;
    use crate::core::platform::values::{
        DepthAttachmentSet, HeapDesc, OptionSet, PassSet, RateMapDesc,
    };

    /// `MTLStoreActionUnknown`; `MTLPixelFormatBGRA8Unorm_sRGB` and `RGBA32Float`.
    const UNKNOWN: u64 = 4;
    const BGRA8_SRGB: u64 = 81;
    const RGBA32F: u64 = 125;
    const SEL: &str = "renderCommandEncoderWithDescriptor:";
    let (p, d) = (platform(), dev());
    let desc = |texture_type, (width, height), levels, slices, options| {
        p.texture_descriptor(&TextureDesc {
            texture_type,
            pixel_format: RGBA16F,
            width,
            height,
            mipmap_level_count: levels,
            array_length: slices,
            usage: READ | TARGET,
            resource_options: Some(options),
            ..TextureDesc::default()
        })
    };
    let made = |texture_type, size, levels, slices, options| {
        let desc = desc(texture_type, size, levels, slices, options);
        d.new_texture(OnFault::Die, &desc).unwrap().unwrap()
    };
    let flat = desc(2, (16, 16), 1, 1, 0);
    let surface = |planes: Option<Vec<SurfacePlaneSpec>>| {
        let size = 16 * 16 * 8;
        p.surfaces()
            .create(&SurfaceSpec {
                planes,
                pixel_format: u32::from_be_bytes(*b"RGhA"),
                alloc_size: 2 * size,
                width: 16,
                height: 16,
                bytes_per_element: 8,
                bytes_per_row: 16 * 8,
                element_width: 1,
                element_height: 1,
                ..SurfaceSpec::default()
            })
            .unwrap()
    };
    let plane = |base: u64| SurfacePlaneSpec {
        base,
        offset: base,
        width: 16,
        height: 16,
        size: 16 * 16 * 8,
        bytes_per_row: 16 * 8,
        bytes_per_element: 8,
        element_width: 1,
        element_height: 1,
        ..SurfacePlaneSpec::default()
    };
    let flat_surface = surface(None);
    let planar = surface(Some(vec![plane(0), plane(16 * 16 * 8)]));
    let heap = d
        .new_heap(&HeapDesc {
            size: 1 << 20,
            heap_type: 1,
            options: vec![OptionSet::StorageMode(2)],
            address_ranges: None,
            pinned_gpu_address: None,
        })
        .unwrap()
        .heap
        .unwrap();
    let bytes = d.new_buffer_of_length(OnFault::Die, 16 * 16 * 8, 0);
    let bytes = bytes.unwrap().unwrap();
    let own = made(2, (16, 16), 1, 1, 0);
    let resolves: [(&str, VulkanTexture, (u64, u64)); 10] = [
        (
            "the resolve texture is 32 by 32, the attachment 16 by 16",
            made(2, (32, 32), 1, 1, 0),
            (0, 0),
        ),
        (
            "the resolve texture is of Type2D, 1 samples and 2 levels",
            made(2, (32, 32), 2, 1, 0),
            (1, 0),
        ),
        (
            "the resolve texture is of Type2DArray, 1 samples and 1 levels",
            made(3, (16, 16), 1, 2, 0),
            (0, 1),
        ),
        (
            "the resolve texture is a texture of its own of storage mode Memoryless",
            made(2, (16, 16), 1, 1, MEMORYLESS),
            (0, 0),
        ),
        (
            "the resolve texture is a texture over an IOSurface of storage mode Private",
            d.new_texture_on_surface(
                &desc(2, (16, 16), 1, 1, PRIVATE_OPTIONS),
                Some(&flat_surface),
                0,
            )
            .unwrap()
            .unwrap(),
            (0, 0),
        ),
        (
            "the resolve texture is a texture over a plane of an IOSurface with planes",
            d.new_texture_on_surface(&flat, Some(&planar), 0)
                .unwrap()
                .unwrap(),
            (0, 0),
        ),
        (
            "the resolve texture is a view of another texture",
            own.new_texture_view(RGBA16F).unwrap().unwrap(),
            (0, 0),
        ),
        (
            "the resolve texture is a texture over a buffer",
            bytes
                .new_texture(OnFault::Die, &flat, 0, 16 * 8)
                .unwrap()
                .unwrap(),
            (0, 0),
        ),
        (
            "the resolve texture is a texture in a heap",
            heap.new_texture(&desc(2, (16, 16), 1, 1, PRIVATE_OPTIONS), 0)
                .unwrap()
                .unwrap(),
            (0, 0),
        ),
        (
            "its store action left Unknown, whose late resolve plain Metal is not measured at",
            own.clone_ref(),
            (0, 0),
        ),
    ];
    for (i, (what, resolve, at)) in resolves.iter().enumerate() {
        let store = if i == resolves.len() - 1 {
            UNKNOWN
        } else {
            RESOLVE
        };
        let line = died_at(SEL, || {
            resolving_pass(&p, &d, (BGRA8_SRGB, store), resolve, *at)
        });
        assert!(
            line.contains("into a texture of MTLPixelFormatRGBA16Float"),
            "{what}: {line}"
        );
        assert!(line.contains(what), "{what}: {line}");
    }
    let heaped = heap
        .new_texture(
            &p.texture_descriptor(&TextureDesc {
                texture_type: 4,
                pixel_format: BGRA8_SRGB,
                width: 16,
                height: 16,
                sample_count: 4,
                usage: TARGET,
                resource_options: Some(PRIVATE_OPTIONS),
                ..TextureDesc::default()
            }),
            1 << 16,
        )
        .unwrap()
        .unwrap();
    let attachments = [
        (
            "the attachment is Memoryless",
            multisampled(&p, &d, BGRA8_SRGB, MEMORYLESS),
        ),
        ("the attachment is a texture in a heap", heaped),
    ];
    for (what, samples) in &attachments {
        let pass = resolve_pass(&p, (samples, RESOLVE), &own, (0, 0));
        let line = died_at(SEL, || encoder_on::<VulkanPlatform>(&d, &pass));
        assert!(line.contains(what), "{what}: {line}");
    }
    let samples = multisampled(&p, &d, BGRA8_SRGB, PRIVATE_OPTIONS);
    // Store action options, a depth attachment that resolves, and another colour attachment
    // that resolves into a texture of another format.
    let options = resolve_pass(&p, (&samples, RESOLVE), &own, (0, 0));
    let attachment = ColorAttachmentSet {
        store_action_options: Some(1),
        ..ColorAttachmentSet::default()
    };
    options
        .set_color_attachment(OnFault::Die, 0, &attachment)
        .unwrap();
    let depth = resolve_pass(&p, (&samples, RESOLVE), &own, (0, 0));
    let depth_samples = multisampled(&p, &d, DEPTH32F, PRIVATE_OPTIONS);
    let depth_resolve = texture(&p, &d, DEPTH32F, TARGET, (16, 16), None);
    depth.set_depth_attachment(&DepthAttachmentSet {
        texture: Some(Some(&depth_samples)),
        resolve_texture: Some(Some(&depth_resolve)),
        load_action: Some(CLEAR),
        store_action: Some(RESOLVE),
        ..DepthAttachmentSet::default()
    });
    let two = resolve_pass(&p, (&samples, RESOLVE), &own, (0, 0));
    let other = multisampled(&p, &d, BGRA8_SRGB, PRIVATE_OPTIONS);
    let into = made(2, (16, 16), 1, 1, 0);
    let attachment = ColorAttachmentSet {
        texture: Some(Some(&other)),
        resolve_texture: Some(Some(&into)),
        load_action: Some(CLEAR),
        store_action: Some(RESOLVE),
        ..ColorAttachmentSet::default()
    };
    two.set_color_attachment(OnFault::Die, 1, &attachment)
        .unwrap();
    // Rate maps the runtime does not take as the identity: a mutable one not yet reset, and
    // one whose rates vary, over a screen of 256 by 256 (whose rates vary between its tiles).
    let mapped = |mutability, horizontal: &[f32]| {
        let m = p.rate_map_descriptor(&RateMapDesc {
            screen_size: MtlSize::new(256, 256, 1),
            mutability,
            ..RateMapDesc::default()
        });
        let count = u64::try_from(horizontal.len()).unwrap();
        let layer = p.new_rate_map_layer(OnFault::Die, MtlSize::new(count, 1, 0));
        let layer = layer.unwrap().unwrap();
        let rates = layer.horizontal_sample_storage().as_ptr::<f32>();
        for (i, &v) in horizontal.iter().enumerate() {
            // SAFETY: the layer's horizontal storage holds `count` samples.
            unsafe { *rates.add(i) = v };
        }
        // SAFETY: its vertical storage holds one.
        unsafe { *layer.vertical_sample_storage().as_ptr::<f32>() = 1.0 };
        m.set_layer(OnFault::Die, &layer, 0).unwrap();
        let map = d.new_rate_map(&m).unwrap().unwrap();
        let pass = resolve_pass(&p, (&samples, RESOLVE), &own, (0, 0));
        pass.set_pass(&PassSet {
            rasterization_rate_map: Some(Some(&map)),
            ..Default::default()
        });
        pass
    };
    let passes = [
        (
            "the attachment's store action options are StoreActionOptions(1)",
            options,
        ),
        ("the pass's depth or stencil attachment resolves", depth),
        (
            "colour attachment 1 resolves into a texture of another format too",
            two,
        ),
        (
            "the pass has a mutable rasterization rate map not yet reset",
            mapped(Some(1), &[1.0]),
        ),
        (
            "the pass has a rasterization rate map whose rates vary",
            mapped(None, &[1.0, 0.25]),
        ),
    ];
    for (what, pass) in &passes {
        let line = died_at(SEL, || encoder_on::<VulkanPlatform>(&d, pass));
        assert!(line.contains(what), "{what}: {line}");
    }
    let pairs = [(RGBA8, 4), (BGRA8, 2), (RGBA16F, 4)];
    for (from, count) in pairs {
        let desc = p.texture_descriptor(&TextureDesc {
            texture_type: 4,
            pixel_format: from,
            width: 16,
            height: 16,
            sample_count: count,
            usage: TARGET,
            resource_options: Some(PRIVATE_OPTIONS),
            ..TextureDesc::default()
        });
        let samples = d.new_texture(OnFault::Die, &desc).unwrap().unwrap();
        let into = match from {
            RGBA16F => texture(&p, &d, RGBA32F, READ | TARGET, (16, 16), None),
            _ => own.clone_ref(),
        };
        let pass = resolve_pass(&p, (&samples, RESOLVE), &into, (0, 0));
        let line = died_at(SEL, || encoder_on::<VulkanPlatform>(&d, &pass));
        assert!(
            line.contains("a pair plain Metal is not measured at"),
            "{from} at {count}: {line}"
        );
    }
}

/// A pass made without a resolve plain Metal leaves untouched ([`untouched_resolves`]): a
/// late resolve texture of that attachment dies, as nothing measured it; a late store action
/// is compared with the descriptor's own, so another dies and the resolving one again encodes
/// nothing, the pass completing. A late resolve texture of its depth or stencil attachment
/// whose store action is left `Unknown` dies too, as the pass's creation dies where the
/// descriptor names one ([`unmeasured_shape`]): `Depth32Float`'s, and `Depth32Float_Stencil8`'s
/// stencil's; a late nil one, or one of an attachment stored by `Store`, goes on. (One stored by
/// a resolving action with no resolve texture dies at the pass's creation, which the runtime
/// refuses.)
#[test]
fn late_setters_of_a_dropped_resolve_are_the_descriptors() {
    use crate::core::platform::values::{DepthAttachmentSet, StencilAttachmentSet};

    /// `MTLPixelFormatBGRA8Unorm_sRGB`; `MTLStoreActionUnknown`.
    const BGRA8_SRGB: u64 = 81;
    const UNKNOWN: u64 = 4;
    /// A late depth resolve texture `t`.
    fn depth_late(t: Option<&VulkanTexture>) -> RenderCmd<'_, VulkanPlatform> {
        RenderCmd::SetDepthResolveTexture {
            texture: t,
            slice: 0,
            depth_plane: 0,
            level: 0,
            y_invert: false,
        }
    }
    /// A late stencil resolve texture `t`.
    fn stencil_late(t: Option<&VulkanTexture>) -> RenderCmd<'_, VulkanPlatform> {
        RenderCmd::SetStencilResolveTexture {
            texture: t,
            slice: 0,
            depth_plane: 0,
            level: 0,
            y_invert: false,
        }
    }
    let (p, d) = (platform(), dev());
    let target = texture(&p, &d, RGBA16F, READ | TARGET, (16, 16), None);
    let open = || resolving_pass(&p, &d, (BGRA8_SRGB, RESOLVE), &target, (0, 0));
    let (_cb, e) = open();
    let late = RenderCmd::SetColorResolveTexture {
        texture: Some(&target),
        slice: 0,
        depth_plane: 0,
        level: 0,
        y_invert: false,
        index: 0,
    };
    let sel = "setColorResolveTexture:slice:depthPlane:level:yInvert:atIndex:";
    let line = died_at(sel, || e.encode(OnFault::Die, late));
    assert!(
        line.contains("resolve by MultisampleResolve into MTLPixelFormatRGBA16Float"),
        "{line}"
    );
    let (_cb, e) = open();
    let store = RenderCmd::SetColorStoreAction {
        store_action: STORE,
        color_attachment_index: 0,
    };
    let line = died_at("setColorStoreAction:atIndex:", || {
        e.encode(OnFault::Die, store)
    });
    assert!(
        line.contains("a late store action Store over the known MultisampleResolve"),
        "{line}"
    );
    let (cb, e) = open();
    let again = RenderCmd::SetColorStoreAction {
        store_action: RESOLVE,
        color_attachment_index: 0,
    };
    for cmd in [again, RenderCmd::EndEncoding] {
        e.encode(OnFault::Die, cmd).unwrap();
    }
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    assert!(cb.error().is_none(), "{:?}", cb.error());
    // A depth or stencil attachment beside, of no resolve texture, given one late.
    let samples = multisampled(&p, &d, BGRA8_SRGB, PRIVATE_OPTIONS);
    let beside = |format, (depth, stencil): (u64, Option<u64>)| {
        let pass = resolve_pass(&p, (&samples, RESOLVE), &target, (0, 0));
        let z = multisampled(&p, &d, format, PRIVATE_OPTIONS);
        pass.set_depth_attachment(&DepthAttachmentSet {
            texture: Some(Some(&z)),
            load_action: Some(CLEAR),
            store_action: Some(depth),
            ..DepthAttachmentSet::default()
        });
        if let Some(store) = stencil {
            pass.set_stencil_attachment(&StencilAttachmentSet {
                texture: Some(Some(&z)),
                load_action: Some(CLEAR),
                store_action: Some(store),
                ..StencilAttachmentSet::default()
            });
        }
        let resolve = texture(&p, &d, format, TARGET, (16, 16), None);
        (encoder_on::<VulkanPlatform>(&d, &pass), resolve)
    };
    let ((_cb, e), resolve) = beside(DEPTH32F, (UNKNOWN, None));
    let late = depth_late(Some(&resolve));
    let sel = late.selector();
    let line = died_at(sel, || e.encode(OnFault::Die, late));
    let named = "a late depth resolve texture in a pass made without the resolve of colour \
                 attachment 0 by MultisampleResolve into MTLPixelFormatRGBA16Float";
    assert!(line.contains(named), "{line}");
    assert!(
        line.contains("depth or stencil attachment resolves"),
        "{line}"
    );
    let ((_cb, e), resolve) = beside(DEPTH32F_S8, (DONT_CARE, Some(UNKNOWN)));
    let late = stencil_late(Some(&resolve));
    let sel = late.selector();
    let line = died_at(sel, || e.encode(OnFault::Die, late));
    assert!(line.contains("a late stencil resolve texture"), "{line}");
    // A nil one, and one of an attachment stored, which resolves nowhere.
    let ((cb, e), resolve) = beside(DEPTH32F_S8, (STORE, Some(UNKNOWN)));
    for cmd in [
        depth_late(Some(&resolve)),
        stencil_late(None),
        RenderCmd::EndEncoding,
    ] {
        e.encode(OnFault::Die, cmd).unwrap();
    }
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    assert!(cb.error().is_none(), "{:?}", cb.error());
}

/// A pass made without a colour resolve plain Metal leaves untouched ([`untouched_resolves`]),
/// with a depth or stencil attachment whose store action the descriptor left `Unknown` and no
/// resolve texture: once a late store action that does not resolve (`Store`) decides it, a late
/// resolve texture of it no longer dies as resolving beside the drop, as nothing resolves there.
/// It goes to the runtime, as in a pass whose colour resolve is the runtime's, which refuses a
/// resolve texture under an action that does not resolve (`ResolveStoreAction`): the setter
/// dies there, naming that refusal, in both passes alike. The attachment left undecided still
/// dies as resolving beside the drop: `Depth32Float_Stencil8`'s stencil, after its depth's
/// late `Store`
#[test]
fn late_resolves_beside_a_dropped_one_follow_late_store_actions() {
    use crate::core::platform::values::{DepthAttachmentSet, StencilAttachmentSet};

    /// `MTLPixelFormatBGRA8Unorm_sRGB`; `MTLStoreActionUnknown`.
    const BGRA8_SRGB: u64 = 81;
    const UNKNOWN: u64 = 4;
    /// A late resolve texture `t` of the depth attachment, or else the stencil one.
    fn late(depth: bool, t: Option<&VulkanTexture>) -> RenderCmd<'_, VulkanPlatform> {
        match depth {
            true => RenderCmd::SetDepthResolveTexture {
                texture: t,
                slice: 0,
                depth_plane: 0,
                level: 0,
                y_invert: false,
            },
            false => RenderCmd::SetStencilResolveTexture {
                texture: t,
                slice: 0,
                depth_plane: 0,
                level: 0,
                y_invert: false,
            },
        }
    }
    let (p, d) = (platform(), dev());
    let samples = multisampled(&p, &d, BGRA8_SRGB, PRIVATE_OPTIONS);
    // Plain Metal leaves the `RGBA16Float` one untouched, a drop; the runtime resolves into the
    // attachment's own format.
    let dropped = texture(&p, &d, RGBA16F, READ | TARGET, (16, 16), None);
    let runtimes = texture(&p, &d, BGRA8_SRGB, READ | TARGET, (16, 16), None);
    let beside = |target: &VulkanTexture, format, stencil: bool| {
        let pass = resolve_pass(&p, (&samples, RESOLVE), target, (0, 0));
        let z = multisampled(&p, &d, format, PRIVATE_OPTIONS);
        pass.set_depth_attachment(&DepthAttachmentSet {
            texture: Some(Some(&z)),
            load_action: Some(CLEAR),
            store_action: Some(UNKNOWN),
            ..DepthAttachmentSet::default()
        });
        if stencil {
            pass.set_stencil_attachment(&StencilAttachmentSet {
                texture: Some(Some(&z)),
                load_action: Some(CLEAR),
                store_action: Some(UNKNOWN),
                ..StencilAttachmentSet::default()
            });
        }
        let resolve = texture(&p, &d, format, TARGET, (16, 16), None);
        (encoder_on::<VulkanPlatform>(&d, &pass), resolve)
    };
    let stored = || RenderCmd::SetDepthStoreAction {
        store_action: STORE,
    };
    let beside_drop = "depth or stencil attachment resolves";
    for (name, target) in [("dropped", &dropped), ("the runtime's", &runtimes)] {
        let ((_cb, e), resolve) = beside(target, DEPTH32F, false);
        e.encode(OnFault::Die, stored()).unwrap();
        let cmd = late(true, Some(&resolve));
        let sel = cmd.selector();
        let line = died_at(sel, || e.encode(OnFault::Die, cmd));
        assert!(line.contains("ResolveStoreAction"), "{name}: {line}");
        assert!(!line.contains(beside_drop), "{name}: {line}");
    }
    let ((_cb, e), resolve) = beside(&dropped, DEPTH32F_S8, true);
    e.encode(OnFault::Die, stored()).unwrap();
    let cmd = late(false, Some(&resolve));
    let sel = cmd.selector();
    let line = died_at(sel, || e.encode(OnFault::Die, cmd));
    assert!(line.contains("a late stencil resolve texture"), "{line}");
    assert!(line.contains(beside_drop), "{line}");
}

/// A resolve into the attachment's sRGB twin, or across the 8-bit four-channel family's
/// channel orders, is the runtime's to resolve or refuse ([`runtime_resolves`]). The pass of a
/// 4-sample `R8Unorm_sRGB` or `RG8Unorm_sRGB` attachment resolving into its UNORM twin is made
/// and completes: the runtime's resolve after the pass averages the samples' codes by the unorm
/// rule. That of an `R8Unorm` or `RG8Unorm` attachment resolving into its sRGB twin, on a
/// device that renders the twin, dies as it ends, at `endEncoding`, naming
/// `device-rounded-resolve`: the runtime's resolve decodes the samples and encodes their average
/// by the device's own sRGB conversions, where Metal's resolve takes its 12-bit tables, and the
/// runtime announces that on every device, MoltenVK among them, which a device on Metal does not
/// relay (the runtime's `color_resolve.rs`; measured on the M3 Max under macOS 27.0.1, MoltenVK
/// 1.4.2, at the runtime's 57b3b26ba0). On a device that lacks the twin the runtime takes those
/// tables and announces nothing, so the pass is made and completes (read from the runtime's
/// code). Other pairs are not
#[test]
fn resolves_into_twins_are_the_runtimes() {
    use PixelFormat as P;
    let runtimes = [
        (P::R8Unorm, P::R8UnormSRGB),
        (P::RG8Unorm, P::RG8UnormSRGB),
        (P::RGBA8Unorm, P::RGBA8UnormSRGB),
        (P::BGRA8Unorm, P::BGRA8UnormSRGB),
        (P::BGR10XR, P::BGR10XRSRGB),
        (P::BGRA10XR, P::BGRA10XRSRGB),
        (P::RGBA8Unorm, P::BGRA8Unorm),
        (P::RGBA8UnormSRGB, P::BGRA8Unorm),
        (P::RGBA16Float, P::RGBA16Float),
    ];
    for (a, b) in runtimes {
        assert!(
            runtime_resolves(a, b) && runtime_resolves(b, a),
            "{a:?} {b:?}"
        );
    }
    let others = [
        (P::BGRA8UnormSRGB, P::RGBA16Float),
        (P::BGRA8Unorm, P::BGR10XR),
        (P::BGRA8Unorm, P::RGB10A2Unorm),
        (P::R8Unorm, P::RG8Unorm),
        (P::R8Unorm, P::RGBA8UnormSRGB),
        (P::RGBA16Float, P::RG16Float),
    ];
    for (a, b) in others {
        assert!(!runtime_resolves(a, b), "{a:?} {b:?}");
    }
    use metal2vulkan::device::FormatUsage;
    use metal2vulkan::vk::Format;

    let (p, d) = (platform(), dev());
    let profile = d.raw.compile_device().profile();
    // Each pair, with the sRGB twin a UNORM attachment resolves into.
    let pairs = [
        (10, 11, Some(Format::R8_SRGB)),
        (11, 10, None),
        (30, 31, Some(Format::R8G8_SRGB)),
        (31, 30, None),
    ];
    for (from, into, srgb) in pairs {
        let target = texture(&p, &d, into, READ | TARGET, (16, 16), None);
        let (cb, e) = resolving_pass(&p, &d, (from, RESOLVE), &target, (0, 0));
        let end = || e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
        if srgb.is_some_and(|twin| profile.supports_format(twin, FormatUsage::ColorAttachment)) {
            let line = died_at("endEncoding", end);
            let named = "(metal2vulkan-degrade-device-rounded-resolve)";
            assert!(line.contains(named), "{from} into {into}: {line}");
            continue;
        }
        end();
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        assert!(cb.error().is_none(), "{from} into {into}: {:?}", cb.error());
    }
}

/// A render state whose half-float colour attachment of two samples, or of four, stores what
/// the fragment function writes as written resolves as Metal's, byte for byte
/// ([`resolves_as_metals`])
#[test]
fn multisampled_half_float_attachments_resolve_as_metals() {
    resolves_as_metals(2);
    resolves_as_metals(4);
}

/// Checks that a render state whose half-float colour attachment of `samples` samples stores
/// what the fragment function writes as written resolves as Metal's, byte for byte, over the
/// sweeps through unorm values, past 1 by float steps and past the largest half: `R16Float`,
/// `RG16Float` and `RGBA16Float`, fully covered and a quarter pixel in, resolved by the drawing
/// pass, and a quarter pixel in by a later pass that loads the samples and draws nothing
/// ([`resolved`]), Metal's measured first.
fn resolves_as_metals(samples: u64) {
    // Fully covered, a quarter pixel in, and a quarter pixel in resolved apart.
    let hows = [(0.0, false), (0.25, false), (0.25, true)];
    let mut cases = Vec::new();
    for kind in [0, 1, 3] {
        for format in [(25, 2), (65, 4), (115, 8)] {
            cases.push((kind, format));
        }
    }
    let key = |format, kind, how| format!("{samples} samples {format:?} sweep {kind} {how:?}");
    // Metal's first, before this platform makes anything in the process.
    let (p, d) = (platform(), dev());
    for (kind, format) in cases {
        for how in hows {
            let ours = resolved(&p, &d, samples, format, kind, how);
            let key = key(format, kind, how);
            assert!(crate::metal_matches_bytes!(key, &ours), "{key}");
        }
    }
}

/// A guest's state of the fixture `vertex` and `vertex_ids`' `plain_fragment` for a BGRA8
/// target, of the input primitive topology `class` (2 lines, 3 triangles), which Metal asks of a
/// vertex function that writes a layer or viewport index.
fn routed_state<P: Platform>(d: &P::Device, vertex: &str, class: u8) -> P::RenderPipeline {
    value_state::<P>(d, (vertex, "vertex_ids:plain_fragment"), |v| {
        v.input_primitive_topology = u64::from(class);
    })
}

/// The pixels of both slices of a 16 by 16 BGRA8 2D array of 2, cleared to black, after
/// `draw` in a pass of both, its render target array length 2, under a [`routed_state`] of
/// `vertex` and `class` over [`flat_vertices`] in buffer 0 and `indices`, each vertex's layer or
/// viewport, in buffer 1, at `setProvokingVertexMode:` of `mode`, with two viewports, the
/// target's left and right halves.
fn routed_pass<P: Platform>(
    p: &P,
    d: &P::Device,
    (vertex, class): (&str, u8),
    (indices, mode): (&[u32], u64),
    draw: RenderCmd<'_, P>,
) -> Vec<[u8; 4]> {
    use crate::core::platform::types::MtlViewport;
    use crate::core::platform::values::PassSet;

    let state = routed_state::<P>(d, vertex, class);
    let (pos, _) = flat_vertices::<P>(d, 16);
    let mut words: Vec<u8> = indices.iter().flat_map(|i| i.to_le_bytes()).collect();
    words.resize(64, 0);
    let routes = buffer::<P>(d, 64, 0, &words);
    let desc = p.texture_descriptor(&TextureDesc {
        texture_type: 3,
        pixel_format: BGRA8,
        width: 16,
        height: 16,
        array_length: 2,
        usage: TARGET | READ,
        ..TextureDesc::default()
    });
    let t = d.new_texture(OnFault::Die, &desc).unwrap().unwrap();
    let pass = p.render_pass_descriptor();
    pass.set_color_attachment(
        OnFault::Die,
        0,
        &ColorAttachmentSet {
            texture: Some(Some(&t)),
            load_action: Some(CLEAR),
            store_action: Some(STORE),
            ..ColorAttachmentSet::default()
        },
    )
    .unwrap();
    pass.set_pass(&PassSet {
        render_target_array_length: Some(2),
        ..PassSet::default()
    });
    let half = |x| MtlViewport {
        origin_x: x,
        origin_y: 0.0,
        width: 8.0,
        height: 16.0,
        znear: 0.0,
        zfar: 1.0,
    };
    let viewports = [half(0.0), half(8.0)];
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
    for cmd in [
        RenderCmd::SetRenderPipelineState {
            pipeline_state: Some(&state),
        },
        RenderCmd::SetViewports {
            viewports: &viewports,
        },
        RenderCmd::SetVertexBuffer {
            buffer: Some(&pos),
            offset: 0,
            index: 0,
        },
        RenderCmd::SetVertexBuffer {
            buffer: Some(&routes),
            offset: 0,
            index: 1,
        },
        RenderCmd::SetProvokingVertexMode { mode },
        draw,
        RenderCmd::EndEncoding,
    ] {
        e.encode(OnFault::Die, cmd).unwrap();
    }
    let out = buffer::<P>(d, 2 * 1024, 0, &[]);
    let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
    for slice in 0..2 {
        blit.encode(
            OnFault::Die,
            BlitCmd::CopyTextureToBuffer {
                source_texture: &t,
                source_slice: slice,
                source_level: 0,
                source_origin: MtlOrigin::new(0, 0, 0),
                source_size: MtlSize::new(16, 16, 1),
                destination_buffer: &out,
                destination_offset: slice * 1024,
                destination_bytes_per_row: 64,
                destination_bytes_per_image: 1024,
            },
        )
        .unwrap();
    }
    blit.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    assert!(cb.error().is_none(), "{:?}", cb.error());
    // SAFETY: a Shared buffer's bytes, read after the blit completed.
    let texels = unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), 2048) };
    texels.as_chunks::<4>().0.to_vec()
}

/// Under a vertex function that writes a layer or a viewport index, triangles land where Metal's
/// do at either provoking vertex, each primitive in its first vertex's layer or viewport at mode 0
/// and in its last's at mode 1: lists and strips over their vertices, over indices and from
/// indirect arguments. Line strips land where Metal's do where the device has the last vertex;
/// elsewhere the runtime draws each line from its last vertex to its first, which dies in strict
/// mode at mode 1 (`provoking-vertex-lines`), and by default lands, relayed
#[test]
fn the_last_vertex_routes_layers_and_viewports_as_metals() {
    let (p, d) = (platform(), dev());
    /// The draw of `kind`: 0 a list of 6 vertices, 1 a strip of 6, 2 a list over [`LIST`]'s
    /// first 6 indices, 3 a strip over [`STRIP_OF_SIX`], 4 a strip of 6 from indirect
    /// arguments, 5 a line strip of 6, over `bytes`, the indices or arguments.
    fn draw<'a, P: Platform>(kind: usize, bytes: &'a P::Buffer) -> RenderCmd<'a, P> {
        let indexed = |primitive_type, index_count| RenderCmd::DrawIndexedPrimitives {
            primitive_type,
            index_count,
            index_type: U16,
            index_buffer: bytes,
            index_buffer_offset: 0,
        };
        let vertices = |primitive_type| RenderCmd::DrawPrimitives {
            primitive_type,
            vertex_start: 0,
            vertex_count: 6,
        };
        match kind {
            0 => vertices(3),
            1 => vertices(STRIP),
            2 => indexed(3, 6),
            3 => indexed(STRIP, 8),
            4 => RenderCmd::DrawPrimitivesIndirect {
                primitive_type: STRIP,
                indirect_buffer: bytes,
                indirect_buffer_offset: 0,
            },
            _ => vertices(2),
        }
    }
    /// The buffer of [`draw`]'s `kind`.
    fn bytes<P: Platform>(d: &P::Device, kind: usize) -> P::Buffer {
        let contents = match kind {
            2 => index_bytes(U16, &LIST[..6]),
            3 => index_bytes(U16, STRIP_OF_SIX),
            _ => u32s(&[6, 1, 0, 0]),
        };
        buffer::<P>(d, 64, 0, &contents)
    }
    let routes: [&[u32]; 3] = [
        &[0, 0, 1, 1, 1, 0, 0, 1],
        &[0, 1, 0, 1, 1, 0, 1, 1],
        &[1, 0, 0, 0, 1, 1, 1, 0],
    ];
    let native = super::last_vertex_native();
    for vertex in ["vertex_ids:layer_vertex", "vertex_ids:viewport_vertex"] {
        for indices in routes {
            for mode in [0, 1] {
                for kind in 0..6 {
                    let class = if kind == 5 { 2 } else { 3 };
                    let shape = (indices, mode);
                    let ours_bytes = bytes::<VulkanPlatform>(&d, kind);
                    let ours =
                        || routed_pass(&p, &d, (vertex, class), shape, draw(kind, &ours_bytes));
                    let ours = if kind == 5 && mode == 1 && !native {
                        let lines = rt::Degradation::ProvokingVertexLines;
                        crate::answer::dies_of(lines, ours);
                        crate::answer::relaxed(ours)
                    } else {
                        ours()
                    };
                    let lit = ours.iter().filter(|px| **px != [0, 0, 0, 0xff]).count();
                    assert!(lit > 0, "{vertex} {indices:?} {mode} {kind}");
                    let key = format!("{vertex} {indices:?} at mode {mode}, draw {kind}");
                    let compared = kind < 5 || mode == 0 || native;
                    if compared {
                        assert!(crate::metal_matches_raw!(key, ours), "{key}");
                    }
                }
            }
        }
    }
}

/// After `setPrimitiveRestartEnabled:index:`, enabled or not, an indexed triangle strip of
/// 16- or 32-bit indices draws what Metal's does, directly and from indirect arguments
/// alike, each compared with Metal's own: it restarts at the setter's index alone, and draws
/// the all-ones index as a vertex; an indexed list draws as without the setter. 16-bit
/// indices of an odd and an even count are drawn from offsets 0 and 2, where the last of
/// them lies alone in its 4-byte word or past one (an odd count from 0, an even one from 2)
#[test]
fn indexed_strips_restart_at_the_set_index_as_metals_do() {
    let (p, d) = (platform(), dev());
    let odd = u16s(&[0, 1, 2, 3, 0xffff, 4, 5, 6, 7]);
    let even = u16s(&[0, 1, 2, 3, 0xffff, 4, 5, 6, 7, 8]);
    let wide = u32s(&[0, 1, 2, 3, 4, 5, 6, 7, 8]);
    for (index_type, indices, offset) in [
        (U16, &odd, 0),
        (U16, &even, 0),
        (U16, &odd, 2),
        (U16, &even, 2),
        (U32, &wide, 0),
    ] {
        for primitive_type in [STRIP, 3] {
            let drawn = (primitive_type, index_type, &indices[..]);
            let unset = restart_draw(&p, &d, drawn, None, (false, offset));
            for (enabled, k) in [(false, 3), (true, 3), (true, 0), (false, 15)] {
                let count = indices.len() / if index_type == U16 { 2 } else { 4 };
                let case = format!(
                    "type {primitive_type} of {count} of {index_type} from {offset} at \
                     {enabled} {k}"
                );
                let ours = restart_draw(&p, &d, drawn, Some((enabled, k)), (false, offset));
                let indirect = restart_draw(&p, &d, drawn, Some((enabled, k)), (true, offset));
                assert_eq!(indirect, ours, "{case}");
                if primitive_type != STRIP {
                    assert_eq!(ours, unset, "{case}");
                } else if k == 3 {
                    assert_ne!(ours, unset, "{case}");
                }
                for indirectly in [false, true] {
                    let key = format!("{case}, Metal's indirect {indirectly}");
                    crate::metal_check_raw!(key, ours);
                }
            }
        }
    }
}

/// An indirect strip of 200,000 16- or 32-bit indices that restarts at the setter's index
/// ([`long_restarted_strips`]) draws what Metal's does, within the command buffer's index pool
/// (`a_long_restarted_strip_draws_within_the_index_pool`)
#[test]
fn a_long_restarted_strip_draws_within_the_index_pool_as_metals() {
    let (p, d) = (platform(), dev());
    for (index_type, indices) in long_restarted_strips() {
        let drawn = (STRIP, index_type, &indices[..]);
        let restart = Some((true, 3));
        let ours = restart_draw(&p, &d, drawn, restart, (true, 0));
        let key = format!("{index_type}");
        crate::metal_check_raw!(key, ours);
    }
}

/// Metal's private types 5 to 8 of ten vertices draw after `setPrimitiveRestartEnabled:index:`
/// what Metal's do, and a line loop, a fan and quads what they draw without it: the setter's
/// index among their vertices is a vertex still
#[test]
fn private_types_draw_after_a_restart_set_as_metals_do() {
    let (p, d) = (platform(), dev());
    /// The pixels of ten vertices of `primitive_type`, after a setter of `k` where one is.
    fn drawn<P: Platform>(
        p: &P,
        d: &P::Device,
        primitive_type: u64,
        k: Option<u64>,
    ) -> Vec<[u8; 4]> {
        let draw = RenderCmd::DrawPrimitives {
            primitive_type,
            vertex_start: 0,
            vertex_count: 10,
        };
        match k {
            None => flat_draw(p, d, FLAT, 0, draw),
            Some(index) => {
                let set = RenderCmd::SetPrimitiveRestartEnabled {
                    enabled: true,
                    index,
                };
                set_draw(p, d, FLAT, set, draw)
            }
        }
    }
    for primitive_type in 5..=8 {
        let unset = drawn(&p, &d, primitive_type, None);
        for k in [0, 3] {
            let ours = drawn(&p, &d, primitive_type, Some(k));
            if primitive_type < 8 {
                assert_eq!(ours, unset, "type {primitive_type} at {k}");
            }
            let key = format!("type {primitive_type} at {k}");
            crate::metal_check_raw!(key, ours);
        }
    }
}

/// The pixels of [`restart_draw`]'s triangle strip, drawn directly.
fn restart_strip<P: Platform>(
    p: &P,
    d: &P::Device,
    (index_type, indices): (u64, &[u8]),
    restart: Option<(bool, u64)>,
) -> Vec<[u8; 4]> {
    restart_draw(p, d, (STRIP, index_type, indices), restart, (false, 0))
}

/// The pixels of [`restart_strip`]'s strip as an execution of an indirect command buffer
/// whose one command draws it, with its state and vertices, after `restart`'s
/// `setPrimitiveRestartEnabled:index:`.
fn executed_strip(
    p: &VulkanPlatform,
    d: &crate::gpu::device::VulkanDevice,
    (index_type, indices): (u64, &[u8]),
    (enabled, index): (bool, u64),
) -> Vec<[u8; 4]> {
    use crate::core::platform::icb::{
        IcbRenderCmd, IndirectCommandBuffer as _, IndirectRenderCommand as _,
    };
    use crate::core::platform::values::IcbDesc;
    use metal2vulkan::metal::PixelFormat;
    use metal2vulkan::pipeline::{
        ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource,
    };

    let function = |name: &str| {
        let path = crate::gpu::device::tests::fixture_shaders().join(format!("{name}.metallib"));
        d.raw
            .new_library_with_data(&std::fs::read(path).unwrap())
            .unwrap()
            .new_function_with_name(name)
            .unwrap()
            .function_descriptor()
    };
    let mut desc = RenderPipelineDescriptor::new(StageSource::Function(function(FLAT.0)));
    desc.fragment = Some(StageSource::Function(function(FLAT.1)));
    desc.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::BGRA8Unorm));
    desc.support_indirect_command_buffers = true;
    let state = crate::gpu::resource::render_pipeline(
        d.raw
            .new_render_pipeline_state(&desc, &rt::PipelineOptions::default())
            .unwrap(),
    );
    let (pos, colours) = flat_vertices::<VulkanPlatform>(d, 0x10000);
    let length = u64::try_from(indices.len()).unwrap();
    let index_buffer = buffer::<VulkanPlatform>(d, length, 0, indices);
    let icb = d
        .new_indirect_command_buffer(
            &IcbDesc {
                command_types: 2,
                max_vertex_buffer_bind_count: 2,
                ..IcbDesc::default()
            },
            1,
            0,
        )
        .unwrap()
        .unwrap();
    let c = icb.render_command(0).unwrap().unwrap();
    for cmd in [
        IcbRenderCmd::SetRenderPipelineState {
            pipeline_state: Some(&state),
        },
        IcbRenderCmd::SetVertexBuffer {
            buffer: Some(&pos),
            offset: 0,
            index: 0,
        },
        IcbRenderCmd::SetVertexBuffer {
            buffer: Some(&colours),
            offset: 0,
            index: 1,
        },
        IcbRenderCmd::DrawIndexedPrimitives {
            primitive_type: STRIP,
            index_count: length / if index_type == U16 { 2 } else { 4 },
            index_type,
            index_buffer: Some(&index_buffer),
            index_buffer_offset: 0,
            instance_count: 1,
            base_vertex: 0,
            base_instance: 0,
        },
    ] {
        c.encode(cmd).unwrap();
    }
    pass_pixels(
        p,
        d,
        vec![
            RenderCmd::UseResources {
                resources: &[
                    Some(ResourceRef::Icb(&icb)),
                    Some(ResourceRef::Buffer(&pos)),
                    Some(ResourceRef::Buffer(&colours)),
                    Some(ResourceRef::Buffer(&index_buffer)),
                ],
                usage: 1,
            },
            RenderCmd::SetPrimitiveRestartEnabled { enabled, index },
            RenderCmd::ExecuteCommandsInBuffer {
                indirect_command_buffer: &icb,
                execution_range: NsRange::new(0, 1),
            },
        ],
    )
}

/// An execution of an indirect command buffer's indexed triangle strip, 16- and 32-bit,
/// draws after `setPrimitiveRestartEnabled:index:`, enabled or not, what Metal's does: what
/// the strip draws directly without the setter, restarting at the all-ones index alone and
/// drawing the setter's index as a vertex. A direct draw after the setter draws otherwise
/// ([`indexed_strips_restart_at_the_set_index_as_metals_do`])
#[test]
fn an_execution_draws_its_strips_after_a_restart_set_as_metals_does() {
    let (p, d) = (platform(), dev());
    let k = 3;
    for (index_type, indices) in [
        (U16, u16s(&[0, 1, 2, 3, 0xffff, 4, 5, 6, 7])),
        (U32, u32s(&[0, 1, 2, 3, 4, 5, 6, 7, 8])),
    ] {
        let strip = (index_type, &indices[..]);
        let direct = restart_strip(&p, &d, strip, None);
        assert_ne!(direct, vec![[0, 0, 0, 0xff]; 256], "{index_type}");
        for enabled in [false, true] {
            let executed = executed_strip(&p, &d, strip, (enabled, k));
            assert_eq!(executed, direct, "{index_type} {enabled}");
        }
        let restarted = restart_strip(&p, &d, strip, Some((true, k)));
        assert_ne!(restarted, direct, "{index_type}");
        let keys = [
            format!("{index_type} direct"),
            format!("{index_type} restarted"),
        ];
        crate::metal_check_raw!(keys[0], direct);
        crate::metal_check_raw!(keys[1], restarted);
    }
}

/// An execution of an empty range, at the start or the end of an indirect command buffer
/// that inherits its pipeline state or not, runs nothing and the pass completes, as Metal's
/// does, under a pipeline state made without `supportIndirectCommandBuffers` too
/// (`render_draw_icb`)
#[test]
fn empty_executions_run_nothing_under_any_pipeline_state() {
    use crate::core::platform::values::IcbDesc;

    fn pixels<P: Platform>(p: &P, d: &P::Device, inherit: bool) -> Vec<[u8; 4]> {
        let state = fixture_state::<P>(d, FLAT.0, FLAT.1);
        let icb = d
            .new_indirect_command_buffer(
                &IcbDesc {
                    command_types: 1,
                    inherit_pipeline_state: inherit,
                    inherit_buffers: inherit,
                    ..IcbDesc::default()
                },
                1,
                0,
            )
            .unwrap()
            .unwrap();
        let execute = |location| RenderCmd::ExecuteCommandsInBuffer {
            indirect_command_buffer: &icb,
            execution_range: NsRange::new(location, 0),
        };
        pass_pixels(
            p,
            d,
            vec![
                RenderCmd::SetRenderPipelineState {
                    pipeline_state: Some(&state),
                },
                execute(0),
                execute(1),
            ],
        )
    }
    let (p, d) = (platform(), dev());
    for inherit in [true, false] {
        assert_eq!(
            pixels(&p, &d, inherit),
            vec![[0, 0, 0, 0xff]; 256],
            "{inherit}"
        );
    }
}

/// A pass whose `renderTargetArrayLength` runs past its attachment's slices, slice 1 of a 2D
/// array of 2 at a length of 2, dies at the encoder's creation, where Metal clears the
/// missing slice 2 into the memory after the texture: for a 2D array over a buffer, made as
/// core makes one, the buffer's bytes at `0x100`, slice 1's being at `0x80`
#[test]
fn a_pass_past_its_attachments_slices_dies_where_metal_writes_past_the_texture() {
    use crate::core::platform::values::PassSet;
    use crate::core::raise::death::DeathKind;

    /// The pass on `d`, and the ranges of the 16 KiB buffer under the texture that its
    /// commit changed from `0x55`.
    fn pass<P: Platform>(p: &P, d: &P::Device) -> (P::RenderPassDescriptor, P::Buffer, P::Texture) {
        let desc = p.texture_descriptor(&TextureDesc {
            texture_type: 3,
            pixel_format: BGRA8,
            width: 4,
            height: 4,
            array_length: 2,
            usage: TARGET | READ,
            ..TextureDesc::default()
        });
        let backing = buffer::<P>(d, 0x4000, 0, &[0x55; 0x4000]);
        let t = backing
            .new_linear_texture(&desc, 0, 16, 0)
            .unwrap()
            .unwrap();
        let pass = p.render_pass_descriptor();
        pass.set_color_attachment(
            OnFault::Die,
            0,
            &ColorAttachmentSet {
                texture: Some(Some(&t)),
                slice: Some(1),
                load_action: Some(CLEAR),
                store_action: Some(STORE),
                ..ColorAttachmentSet::default()
            },
        )
        .unwrap();
        pass.set_pass(&PassSet {
            render_target_array_length: Some(2),
            ..PassSet::default()
        });
        (pass, backing, t)
    }
    let (p, d) = (platform(), dev());
    let (descriptor, _backing, _t) = pass(&p, &d);
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let site = catch_death(|| cb.render_encoder(OnFault::Return, &descriptor)).unwrap_err();
    assert_eq!(
        (site.kind, site.file),
        (DeathKind::Abort, "renderCommandEncoderWithDescriptor:")
    );
}

/// The resolve texture's texels after a pass of a 4-sample colour and depth attachment, each
/// over a resolve texture, whose store actions stay `Unknown` through `endEncoding`.
fn unknown_stores_over_resolves<P: Platform>(p: &P, d: &P::Device) -> Vec<u8> {
    /// `MTLTextureType2DMultisample`, `MTLPixelFormatDepth32Float`, `MTLStoreActionUnknown`.
    const MULTISAMPLE: u64 = 4;
    const DEPTH32F: u64 = 252;
    const UNKNOWN: u64 = 4;
    let samples = |format| {
        let desc = p.texture_descriptor(&TextureDesc {
            texture_type: MULTISAMPLE,
            pixel_format: format,
            width: 4,
            height: 4,
            sample_count: 4,
            usage: TARGET,
            resource_options: Some(0x20),
            ..TextureDesc::default()
        });
        d.new_texture(OnFault::Die, &desc).unwrap().unwrap()
    };
    let before: Vec<u8> = (0..64u8).map(|i| i.wrapping_mul(37) | 1).collect();
    let resolve = texture(p, d, BGRA8, TARGET | READ, (4, 4), Some((&before, 16)));
    let depth_resolve = {
        let desc = p.texture_descriptor(&TextureDesc {
            pixel_format: DEPTH32F,
            width: 4,
            height: 4,
            usage: TARGET,
            resource_options: Some(0x20),
            ..TextureDesc::default()
        });
        d.new_texture(OnFault::Die, &desc).unwrap().unwrap()
    };
    let (color, depth) = (samples(BGRA8), samples(DEPTH32F));
    let pass = p.render_pass_descriptor();
    pass.set_color_attachment(
        OnFault::Die,
        0,
        &ColorAttachmentSet {
            texture: Some(Some(&color)),
            resolve_texture: Some(Some(&resolve)),
            load_action: Some(CLEAR),
            store_action: Some(UNKNOWN),
            clear_color: Some(ClearColor {
                red: 1.0,
                green: 0.0,
                blue: 0.0,
                alpha: 1.0,
            }),
            ..ColorAttachmentSet::default()
        },
    )
    .unwrap();
    pass.set_depth_attachment(&crate::core::platform::values::DepthAttachmentSet {
        texture: Some(Some(&depth)),
        resolve_texture: Some(Some(&depth_resolve)),
        load_action: Some(CLEAR),
        store_action: Some(UNKNOWN),
        ..Default::default()
    });
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
    e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
    let out = buffer::<P>(d, 64, 0, &[]);
    let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
    blit.encode(
        OnFault::Die,
        BlitCmd::CopyTextureToBuffer {
            source_texture: &resolve,
            source_slice: 0,
            source_level: 0,
            source_origin: MtlOrigin::new(0, 0, 0),
            source_size: MtlSize::new(4, 4, 1),
            destination_buffer: &out,
            destination_offset: 0,
            destination_bytes_per_row: 16,
            destination_bytes_per_image: 64,
        },
    )
    .unwrap();
    blit.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    assert!(cb.error().is_none(), "{:?}", cb.error());
    // SAFETY: a Shared buffer's bytes, read after the blit completed.
    unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), 64) }.to_vec()
}

/// A colour and a depth attachment whose store actions stay `Unknown` over a resolve texture
/// end their pass, and resolve nothing, as on Metal
#[test]
fn unknown_stores_over_resolves_resolve_nothing() {
    let before: Vec<u8> = (0..64u8).map(|i| i.wrapping_mul(37) | 1).collect();
    let (p, d) = (platform(), dev());
    assert_eq!(unknown_stores_over_resolves(&p, &d), before);
}

/// The bytes of a 16 by 16 resolve texture of `format` (`bytes` a texel), a depth or the stencil
/// one, that a pass cleared (depth to 0.75, stencil to `0x5a`) and stored, after a pass of a
/// 4-sample attachment of `format` (Memoryless where `memoryless`, else Private) beside no other,
/// cleared, stored by `store`, a store action that does not resolve, with that resolve texture,
/// drawing nothing.
fn unused_resolve<P: Platform>(
    p: &P,
    d: &P::Device,
    (format, bytes): (u64, u64),
    store: u64,
    memoryless: bool,
) -> Vec<u8> {
    use crate::core::platform::values::{DepthAttachmentSet, StencilAttachmentSet};
    let resolve = d
        .new_texture(
            OnFault::Die,
            &p.texture_descriptor(&TextureDesc {
                pixel_format: format,
                width: 16,
                height: 16,
                usage: TARGET,
                resource_options: Some(PRIVATE_OPTIONS),
                ..TextureDesc::default()
            }),
        )
        .unwrap()
        .unwrap();
    let options = if memoryless {
        MEMORYLESS
    } else {
        PRIVATE_OPTIONS
    };
    let samples = multisampled(p, d, format, options);
    // A pass of `texture` alone, cleared, stored by `store`, into `resolve_texture`.
    let pass = |texture, resolve_texture, store, (depth, stencil)| {
        let pass = p.render_pass_descriptor();
        if format == STENCIL8 {
            pass.set_stencil_attachment(&StencilAttachmentSet {
                texture: Some(Some(texture)),
                resolve_texture: Some(resolve_texture),
                load_action: Some(CLEAR),
                store_action: Some(store),
                clear_stencil: Some(stencil),
                ..StencilAttachmentSet::default()
            });
        } else {
            pass.set_depth_attachment(&DepthAttachmentSet {
                texture: Some(Some(texture)),
                resolve_texture: Some(resolve_texture),
                load_action: Some(CLEAR),
                store_action: Some(store),
                clear_depth: Some(depth),
                ..DepthAttachmentSet::default()
            });
        }
        pass
    };
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    for pass in [
        pass(&resolve, None, STORE, (0.75, 0x5a)),
        pass(&samples, Some(&resolve), store, (0.25, 0x11)),
    ] {
        let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
        e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
    }
    let length = 16 * 16 * bytes;
    let out = buffer::<P>(d, length, 0, &[]);
    let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
    for cmd in [
        BlitCmd::CopyTextureToBuffer {
            source_texture: &resolve,
            source_slice: 0,
            source_level: 0,
            source_origin: MtlOrigin::new(0, 0, 0),
            source_size: MtlSize::new(16, 16, 1),
            destination_buffer: &out,
            destination_offset: 0,
            destination_bytes_per_row: 16 * bytes,
            destination_bytes_per_image: length,
        },
        BlitCmd::EndEncoding,
    ] {
        blit.encode(OnFault::Die, cmd).unwrap();
    }
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    assert!(cb.error().is_none(), "{:?}", cb.error());
    let n = usize::try_from(length).unwrap();
    // SAFETY: a Shared buffer's bytes, read after the blit completed.
    unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), n) }.to_vec()
}

/// `MTLPixelFormatStencil8`.
const STENCIL8: u64 = 253;

/// A depth or stencil attachment beside a resolve texture its store action does not use
/// (`Store` or `DontCare`, of a Private attachment, and `DontCare` of a Memoryless one) ends its
/// pass, and the resolve texture keeps its texels, as on Metal
#[test]
fn unused_depth_and_stencil_resolves_leave_their_textures() {
    let (p, d) = (platform(), dev());
    let depth: Vec<u8> = (0..256).flat_map(|_| 0.75f32.to_ne_bytes()).collect();
    let stencil = vec![0x5a; 256];
    for (format, kept) in [((DEPTH32F, 4), &depth), ((STENCIL8, 1), &stencil)] {
        for (store, memoryless) in [(STORE, false), (DONT_CARE, false), (DONT_CARE, true)] {
            let case = format!("{format:?} store {store} memoryless {memoryless}");
            let ours = unused_resolve(&p, &d, format, store, memoryless);
            assert_eq!(&ours, kept, "{case}");
        }
    }
}

/// A load action, store action or store action options the runtime has no value of is
/// taken, as Metal's setters take it, and read back as set; the encoder's creation dies at
/// it, where Metal asserts ("-[MTLRenderPassDescriptorInternal validate:width:height:]:2401:
/// failed assertion `Render Pass Descriptor Validation\nloadAction is not a valid
/// MTLLoadAction.'", and "storeAction is not a valid MTLStoreAction.", "storeActionOptions
/// contains unrecognized bits (0x8)" for 9 each, measured one case a process with
/// `MTL_DEBUG_LAYER` unset; under it, the store action options' assertion is
/// `_MTLDebugValidateRenderPassDescriptorAndTrackAttachments:370`'s, at the same send),
/// whether or not the attachment has a texture; so does a resolve filter the runtime has no
/// value of. A value the runtime has, set after it, makes the encoder
#[test]
fn render_pass_values_metal_refuses_die_at_the_encoder() {
    let p = platform();
    let d = dev();
    // Each case's load action, store action and store action options.
    fn set_of<P: Platform>(
        (load_action, store_action, store_action_options): (Option<u64>, Option<u64>, Option<u64>),
    ) -> ColorAttachmentSet<'static, P> {
        ColorAttachmentSet {
            load_action,
            store_action,
            store_action_options,
            ..ColorAttachmentSet::default()
        }
    }
    let cases = [
        (Some(9), None, None),
        (None, Some(9), None),
        (None, None, Some(9)),
    ];
    for case in cases {
        let set = &set_of::<VulkanPlatform>(case);
        let (_dst, pass) = pass_of(&p, &d, (4, 4));
        pass.set_color_attachment(OnFault::Die, 0, set).unwrap();
        let state = pass.color_attachment_state(0).unwrap();
        let read = (state.load_action, state.store_action);
        let expected = (
            set.load_action.unwrap_or(CLEAR),
            set.store_action.unwrap_or(STORE),
        );
        assert_eq!(read, expected);
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        *crate::answer::LAST_LINE.lock() = None;
        let site = catch_death(|| cb.render_encoder(OnFault::Die, &pass).is_ok()).unwrap_err();
        let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
        assert_eq!(
            (site.kind, site.file),
            (
                crate::core::raise::death::DeathKind::Abort,
                "renderCommandEncoderWithDescriptor:"
            ),
            "{line}"
        );
        assert!(line.contains("Metal aborts at"), "{line}");
        // A value the runtime has, set after it.
        let (_dst, pass) = pass_of(&p, &d, (4, 4));
        pass.set_color_attachment(OnFault::Die, 0, set).unwrap();
        let again = ColorAttachmentSet {
            load_action: set.load_action.map(|_| CLEAR),
            store_action: set.store_action.map(|_| STORE),
            store_action_options: set.store_action_options.map(|_| 0),
            ..ColorAttachmentSet::default()
        };
        pass.set_color_attachment(OnFault::Die, 0, &again).unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
        e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
    }
    // The depth and stencil attachments', with no texture: a load action of 9, a store
    // action of 9, a depth resolve filter of 9 and a stencil resolve filter of 2 or 9
    // ("resolveFilter is not a valid MTLMultisampleDepthResolveFilter.", and
    // "MTLMultisampleStencilResolveFilter."). A depth resolve filter of 2 makes the encoder.
    use crate::core::platform::values::{DepthAttachmentSet, StencilAttachmentSet};
    let q = d.new_command_queue().unwrap();
    let encoder = |pass: &VulkanRenderPassDescriptor| {
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        *crate::answer::LAST_LINE.lock() = None;
        let made = catch_death(|| {
            let e = cb.render_encoder(OnFault::Die, pass).unwrap();
            e.map(|e| e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap())
                .is_some()
        });
        let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
        (made, line)
    };
    let depth = |load_action, depth_resolve_filter| DepthAttachmentSet {
        load_action,
        depth_resolve_filter,
        ..DepthAttachmentSet::default()
    };
    let stencil = |store_action, stencil_resolve_filter| StencilAttachmentSet {
        store_action,
        stencil_resolve_filter,
        ..StencilAttachmentSet::default()
    };
    for (depth, stencil) in [
        (depth(Some(9), None), stencil(None, None)),
        (depth(None, Some(9)), stencil(None, None)),
        (depth(None, None), stencil(Some(9), None)),
        (depth(None, None), stencil(None, Some(2))),
        (depth(None, None), stencil(None, Some(9))),
    ] {
        let (_dst, pass) = pass_of(&p, &d, (4, 4));
        pass.set_depth_attachment(&depth);
        pass.set_stencil_attachment(&stencil);
        let (made, line) = encoder(&pass);
        let site = made.unwrap_err();
        assert_eq!(
            (site.kind, site.file),
            (
                crate::core::raise::death::DeathKind::Abort,
                "renderCommandEncoderWithDescriptor:"
            ),
            "{line}"
        );
        assert!(line.contains("Metal aborts at"), "{line}");
    }
    let (_dst, pass) = pass_of(&p, &d, (4, 4));
    pass.set_depth_attachment(&depth(None, Some(2)));
    let (made, line) = encoder(&pass);
    assert!(matches!(made, Ok(true)), "{line}");
}

/// The display's functions cannot write an integer target: no pipeline, as on Metal
#[test]
fn an_integer_target_has_no_display_pipeline() {
    let d = dev();
    let cache = DisplayPipelineCache::<VulkanPlatform>::new(d.clone_ref()).unwrap();
    let key = PipelineKey {
        format: R32UINT,
        gamma: false,
    };
    assert!(cache.pipeline_for(key).is_none());
}

/// The display's pipeline answers Metal's values for a plain state
#[test]
fn a_display_pipeline_answers_metals_values() {
    fn values<P: Platform>(d: &P::Device) -> (bool, u64, u64, bool, u64) {
        use crate::core::platform::resource::RenderPipeline as _;
        let cache = DisplayPipelineCache::<P>::new(d.clone_ref()).unwrap();
        let state = cache
            .pipeline_for(PipelineKey {
                format: BGRA8,
                gamma: true,
            })
            .unwrap();
        (
            state.support_indirect_command_buffers(),
            state.resource_index(),
            state.imageblock_sample_length(),
            state.threadgroup_size_matches_tile_size(),
            state.max_total_threads_per_threadgroup(),
        )
    }
    let vulkan = values::<VulkanPlatform>(&dev());
    crate::metal_check!("values", vulkan);
}

/// A pass over a rasterization rate map of no layers made from an immutable descriptor that
/// only loads or clears its target and stores it stores what Metal's does, in a process that has
/// released no map of layers and after one was made, drawn with and released, which makes
/// Metal's draws in such a pass draw. Over a mutable map reset to no
/// layers both store the same, each pass ending with a draw of no vertices
/// (`crate::gpu::rate_map::no_draw`). A draw in such a pass,
/// of a public or a private type, and one of primitive type 9 that fails the command buffer,
/// dies at its selector, naming the map, and a tile dispatch there dies by its own line
#[test]
fn a_drawless_pass_over_a_map_of_no_layers_is_metals() {
    use crate::core::platform::rate_map::{
        RateMap as _, RateMapDescriptor as _, RateMapLayerDescriptor as _,
    };
    use crate::core::platform::values::{PassSet, RateMapDesc};
    use crate::core::raise::death::DeathKind;

    /// `MTLLoadActionLoad`.
    const LOAD: u64 = 1;
    const SIDE: u64 = 16;

    /// A descriptor of a `SIDE` by `SIDE` screen of `mutability` (Immutable when unset), with
    /// no layers.
    fn none<P: Platform>(p: &P, mutability: Option<u64>) -> P::RateMapDescriptor {
        p.rate_map_descriptor(&RateMapDesc {
            screen_size: MtlSize::new(SIDE, SIDE, 1),
            mutability,
            ..RateMapDesc::default()
        })
    }

    /// A descriptor of a `SIDE` by `SIDE` screen of `mutability`, with one layer of one
    /// sample of 1 on each axis.
    fn one_layer<P: Platform>(p: &P, mutability: Option<u64>) -> P::RateMapDescriptor {
        let desc = none(p, mutability);
        let layer = p
            .new_rate_map_layer(OnFault::Die, MtlSize::new(1, 1, 0))
            .unwrap()
            .unwrap();
        for at in [
            layer.horizontal_sample_storage(),
            layer.vertical_sample_storage(),
        ] {
            // SAFETY: the layer's storage holds one sample on each axis.
            unsafe { *at.as_ptr::<f32>() = 1.0 };
        }
        desc.set_layer(OnFault::Die, &layer, 0).unwrap();
        desc
    }

    /// A map of no layers of a `SIDE` by `SIDE` screen: made so, or a mutable map of one
    /// layer reset to none.
    fn no_layers<P: Platform>(p: &P, d: &P::Device, reset: bool) -> P::RateMap {
        if !reset {
            return d.new_rate_map(&none(p, None)).unwrap().unwrap();
        }
        let map = d.new_rate_map(&one_layer(p, Some(1))).unwrap().unwrap();
        map.reset(&none(p, None));
        map
    }

    /// Makes an immutable map of one layer, draws a triangle in a pass over it, and releases
    /// it once its command buffer completed: on Metal, a map of no layers made after it draws,
    /// which a pass that only loads or clears and stores must not change.
    fn draw_and_release<P: Platform>(p: &P, d: &P::Device) {
        let _pool = crate::core::rt::Pool::here();
        let map = d.new_rate_map(&one_layer(p, None)).unwrap().unwrap();
        let (_dst, pass) = pass(p, d, &map, false);
        let state = fixture_state::<P>(d, "vert:tri_vert", "frag:tri_frag");
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
        for cmd in [
            RenderCmd::SetRenderPipelineState {
                pipeline_state: Some(&state),
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
    }

    /// A `SIDE` by `SIDE` BGRA8 target of scattered texels and a pass of it over `map`,
    /// loading it or clearing it to (0.25, 0.5, 0.75, 1), and storing it.
    fn pass<P: Platform>(
        p: &P,
        d: &P::Device,
        map: &P::RateMap,
        load: bool,
    ) -> (P::Texture, P::RenderPassDescriptor) {
        let texels: Vec<u8> = (0..SIDE * SIDE * 4)
            .map(|i| u8::try_from(i * 7 % 251).unwrap())
            .collect();
        let dst = texture(
            p,
            d,
            BGRA8,
            TARGET | READ,
            (SIDE, SIDE),
            Some((&texels, SIDE * 4)),
        );
        let pass = p.render_pass_descriptor();
        pass.set_color_attachment(
            OnFault::Die,
            0,
            &ColorAttachmentSet {
                texture: Some(Some(&dst)),
                load_action: Some(if load { LOAD } else { CLEAR }),
                store_action: Some(STORE),
                clear_color: Some(ClearColor {
                    red: 0.25,
                    green: 0.5,
                    blue: 0.75,
                    alpha: 1.0,
                }),
                ..ColorAttachmentSet::default()
            },
        )
        .unwrap();
        pass.set_pass(&PassSet {
            rasterization_rate_map: Some(Some(map)),
            ..Default::default()
        });
        (dst, pass)
    }

    /// The target's bytes after a drawless pass over a map of no layers, made so or by a
    /// reset, loading or clearing; none where the pass makes no encoder.
    fn stored<P: Platform>(p: &P, d: &P::Device, reset: bool, load: bool) -> Option<Vec<u8>> {
        let map = no_layers(p, d, reset);
        let (dst, pass) = pass(p, d, &map, load);
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = cb.render_encoder(OnFault::Die, &pass).unwrap()?;
        e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
        let out = buffer::<P>(d, SIDE * SIDE * 4, 0, &[]);
        let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
        for cmd in [
            BlitCmd::CopyTextureToBuffer {
                source_texture: &dst,
                source_slice: 0,
                source_level: 0,
                source_origin: MtlOrigin::new(0, 0, 0),
                source_size: MtlSize::new(SIDE, SIDE, 1),
                destination_buffer: &out,
                destination_offset: 0,
                destination_bytes_per_row: SIDE * 4,
                destination_bytes_per_image: SIDE * SIDE * 4,
            },
            BlitCmd::EndEncoding,
        ] {
            blit.encode(OnFault::Die, cmd).unwrap();
        }
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        assert!(cb.error().is_none(), "{:?}", cb.error());
        let n = usize::try_from(SIDE * SIDE * 4).unwrap();
        // SAFETY: a Shared buffer's bytes, read after the blit completed.
        Some(unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), n) }.to_vec())
    }

    let p = platform();
    let d = dev();
    let shapes = [(false, false), (false, true), (true, false), (true, true)];
    let ours = shapes.map(|(reset, load)| stored(&p, &d, reset, load));
    for (shape, bytes) in shapes.iter().zip(&ours) {
        assert!(bytes.is_some(), "reset, load {shape:?}: no encoder");
    }
    let state = fixture_state::<VulkanPlatform>(&d, "vert:tri_vert", "frag:tri_frag");
    let q = d.new_command_queue().unwrap();
    // A draw, of a public type, of a private one (a triangle fan and quads) and of primitive
    // type 9 that hangs the GPU, named by the map; and a tile dispatch, by its own line.
    let deaths = [
        (3, 3, "rate map of no layers"),
        (6, 4, "rate map of no layers"),
        (7, 4, "rate map of no layers"),
        (9, 3, "rate map of no layers"),
        (
            0,
            0,
            "a tile dispatch in a pass of a rasterization rate map of no layers",
        ),
    ];
    for reset in [false, true] {
        let map = no_layers(&p, &d, reset);
        let (_dst, pass) = pass(&p, &d, &map, false);
        for (primitive_type, vertex_count, named) in deaths {
            let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
            let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
            e.encode(
                OnFault::Die,
                RenderCmd::SetRenderPipelineState {
                    pipeline_state: Some(&state),
                },
            )
            .unwrap();
            *crate::answer::LAST_LINE.lock() = None;
            let cmd = match primitive_type {
                0 => RenderCmd::DispatchThreadsPerTile {
                    threads_per_tile: MtlSize::new(1, 1, 1),
                },
                _ => RenderCmd::DrawPrimitives {
                    primitive_type,
                    vertex_start: 0,
                    vertex_count,
                },
            };
            let sel = cmd.selector();
            let site = catch_death(|| e.encode(OnFault::Die, cmd)).unwrap_err();
            let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
            assert_eq!(
                (site.kind, site.file),
                (DeathKind::Abort, sel),
                "reset {reset}: {line}"
            );
            assert!(line.contains(named), "{line}");
        }
    }
    for (shape, o) in shapes.iter().zip(&ours) {
        let key = format!("reset, load {shape:?}");
        assert!(crate::metal_matches_raw!(key, o), "{key}");
    }
    // After a map of one layer was drawn with and released, each platform's own.
    draw_and_release(&p, &d);
    let ours = shapes.map(|(reset, load)| stored(&p, &d, reset, load));
    for (shape, o) in shapes.iter().zip(&ours) {
        let key = format!("after a released map, reset, load {shape:?}");
        assert!(crate::metal_matches_raw!(key, o), "{key}");
    }
}

/// A pass of no draw call over a rasterization rate map made of a `Mutable` descriptor, whose
/// `mutability` answers 1, runs and stores its clear, as Metal's does, each ending with a draw of
/// no vertices (`crate::gpu::rate_map::no_draw`), without which the runtime fails its command buffer as a GPU hang, as
/// Metal's GPU does: its state set or not, over a map of one layer not yet reset, reset to the
/// identity, reset to half its rate on each axis, and over one made of no layers, in a pass of
/// one BGRA8 target and, with no state set (the fixtures' state is of one sample, which neither
/// binds in a pass of four), in one of a 4-sample BGRA8 target resolved into another, each in a
/// command buffer of its own queue, before any command buffer drew over a mutable map
#[test]
fn drawless_passes_over_mutable_maps_store_as_metals() {
    use crate::core::platform::rate_map::{
        RateMap as _, RateMapDescriptor as _, RateMapLayerDescriptor as _,
    };
    use crate::core::platform::values::{PassSet, RateMapDesc};

    const SIDE: u64 = 64;
    /// `MTLMutabilityMutable`.
    const MUTABLE: u64 = 1;
    /// `MTLTextureType2DMultisample`, `MTLStoreActionMultisampleResolve` and
    /// `MTLResourceStorageModePrivate`.
    const MULTISAMPLE: u64 = 4;
    const RESOLVE: u64 = 2;
    const PRIVATE: u64 = 0x20;

    /// A descriptor of a `SIDE` by `SIDE` screen of `mutability`, with one layer of one sample
    /// of `rate` on each axis where `rate` is some, else no layers.
    fn map_desc<P: Platform>(
        p: &P,
        mutability: Option<u64>,
        rate: Option<f32>,
    ) -> P::RateMapDescriptor {
        let desc = p.rate_map_descriptor(&RateMapDesc {
            screen_size: MtlSize::new(SIDE, SIDE, 1),
            mutability,
            ..RateMapDesc::default()
        });
        if let Some(rate) = rate {
            let layer = p
                .new_rate_map_layer(OnFault::Die, MtlSize::new(1, 1, 0))
                .unwrap()
                .unwrap();
            for at in [
                layer.horizontal_sample_storage(),
                layer.vertical_sample_storage(),
            ] {
                // SAFETY: the layer's storage holds one sample on each axis.
                unsafe { *at.as_ptr::<f32>() = rate };
            }
            desc.set_layer(OnFault::Die, &layer, 0).unwrap();
        }
        desc
    }

    /// A `SIDE` by `SIDE` BGRA8 target, a pass that clears it and stores it, or with
    /// `multisample` clears a 4-sample one and resolves it into it, and the 4-sample one.
    fn target<P: Platform>(
        p: &P,
        d: &P::Device,
        multisample: bool,
    ) -> (P::Texture, P::RenderPassDescriptor, Option<P::Texture>) {
        if !multisample {
            let (dst, pass) = pass_of(p, d, (SIDE, SIDE));
            return (dst, pass, None);
        }
        let samples = d
            .new_texture(
                OnFault::Die,
                &p.texture_descriptor(&TextureDesc {
                    texture_type: MULTISAMPLE,
                    pixel_format: BGRA8,
                    width: SIDE,
                    height: SIDE,
                    sample_count: 4,
                    usage: TARGET,
                    resource_options: Some(PRIVATE),
                    ..TextureDesc::default()
                }),
            )
            .unwrap()
            .unwrap();
        let dst = texture(p, d, BGRA8, TARGET | READ, (SIDE, SIDE), None);
        let pass = p.render_pass_descriptor();
        let attachment = ColorAttachmentSet {
            texture: Some(Some(&samples)),
            resolve_texture: Some(Some(&dst)),
            load_action: Some(CLEAR),
            store_action: Some(RESOLVE),
            ..ColorAttachmentSet::default()
        };
        pass.set_color_attachment(OnFault::Die, 0, &attachment)
            .unwrap();
        (dst, pass, Some(samples))
    }

    /// A drawless pass over a mutable map made of `made` and reset to `reset` where some, the
    /// state set where `state`, into a target of `multisample` ([`target`]), in a command buffer
    /// of a queue of its own that then copies the target out, committed and completed: the
    /// map's `mutability`, the command buffer's status and error code, and the target's bytes.
    fn drawless<P: Platform>(
        p: &P,
        d: &P::Device,
        (made, reset, state): (Option<f32>, Option<Option<f32>>, bool),
        multisample: bool,
    ) -> (u64, u64, Option<i64>, Vec<u8>) {
        let map = d
            .new_rate_map(&map_desc(p, Some(MUTABLE), made))
            .unwrap()
            .unwrap();
        let mutability = map.mutability();
        if let Some(rate) = reset {
            map.reset(&map_desc(p, None, rate));
        }
        let (dst, pass, _samples) = target(p, d, multisample);
        pass.set_pass(&PassSet {
            rasterization_rate_map: Some(Some(&map)),
            ..Default::default()
        });
        let pipeline = fixture_state::<P>(d, "vert:tri_vert", "frag:tri_frag");
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
        if state {
            let set = RenderCmd::SetRenderPipelineState {
                pipeline_state: Some(&pipeline),
            };
            e.encode(OnFault::Die, set).unwrap();
        }
        e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
        let out = buffer::<P>(d, SIDE * SIDE * 4, 0, &[]);
        let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
        for cmd in [
            BlitCmd::CopyTextureToBuffer {
                source_texture: &dst,
                source_slice: 0,
                source_level: 0,
                source_origin: MtlOrigin::new(0, 0, 0),
                source_size: MtlSize::new(SIDE, SIDE, 1),
                destination_buffer: &out,
                destination_offset: 0,
                destination_bytes_per_row: SIDE * 4,
                destination_bytes_per_image: SIDE * SIDE * 4,
            },
            BlitCmd::EndEncoding,
        ] {
            blit.encode(OnFault::Die, cmd).unwrap();
        }
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        let n = usize::try_from(SIDE * SIDE * 4).unwrap();
        // SAFETY: a Shared buffer's bytes, read after the command buffer completed.
        let bytes = unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), n) };
        let code = cb.error().map(|e| e.code);
        (mutability, cb.status().0, code, bytes.to_vec())
    }

    let cases = [
        (Some(1.0), None, false),
        (Some(1.0), None, true),
        (Some(1.0), Some(Some(1.0)), false),
        (Some(0.5), Some(Some(0.5)), false),
        (Some(0.5), Some(Some(0.5)), true),
        (None, None, false),
    ];
    let clear = [0, 0, 0, 0xff].repeat(usize::try_from(SIDE * SIDE).unwrap());
    let runs: Vec<_> = [false, true]
        .into_iter()
        .flat_map(|multisample| cases.map(|case| (case, multisample)))
        .filter(|&((_, _, state), multisample)| !(state && multisample))
        .collect();
    let (p, d) = (platform(), dev());
    let ours: Vec<_> = (runs.iter())
        .map(|&(case, multisample)| drawless(&p, &d, case, multisample))
        .collect();
    for (run, answer) in runs.iter().zip(&ours) {
        assert_eq!(answer, &(MUTABLE, 4, None, clear.clone()), "{run:?}");
    }
}

/// A pass that draws over a mutable rasterization rate map draws as Metal's GPU reads the map
/// as it runs the pass: not yet reset by then, it rasterizes nothing and stores its clear; reset
/// after the pass ended and before its command buffer's commit, or after the commit while the
/// command buffer waits for an event signalled after the reset, to the identity over the map's
/// screen, it draws by the map as reset, as Metal's does, and a map of the identity made over a
/// screen smaller than the target draws its corner, as Metal's. Reset so to the identity of
/// another screen size, it draws over the screen the map had at the encoder's creation, to the
/// 32-pixel tiles of the new screen where that spans one tile fewer, as Metal's GPU does: all of
/// a 16-pixel target from a 16-pixel screen reset to 8, the corner from 8 reset to 16, the 32 by
/// 32 corner of a 64-pixel target from a 64-pixel screen reset to 32 or to 24, and all of it
/// reset to 48
#[test]
fn a_pass_over_a_mutable_map_draws_by_the_map_as_reset_before_it_runs() {
    use crate::core::platform::event::{EventRef, SharedEvent as _};
    use crate::core::platform::rate_map::{
        RateMap as _, RateMapDescriptor as _, RateMapLayerDescriptor as _,
    };
    use crate::core::platform::values::{PassSet, RateMapDesc};
    use crate::core::raise::death::DeathSite;

    const SIDE: u64 = 16;
    const HALF: u64 = SIDE / 2;
    /// The side of the target and screen whose resets cross a 32-pixel tile.
    const WIDE: u64 = 64;

    /// How the pass's map is made, and when it is reset, if at all.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Reset {
        /// A mutable map, never reset.
        Never,
        /// A mutable map, reset after the pass ended and before the commit.
        BeforeCommit,
        /// A mutable map, reset after the commit, before the event its command buffer waits for.
        AfterCommit,
        /// An immutable map, never reset.
        Made,
    }

    /// A descriptor of a `side` by `side` screen of `mutability` (Immutable when unset), with
    /// one layer of one sample of 1 on each axis.
    fn identity<P: Platform>(p: &P, side: u64, mutability: Option<u64>) -> P::RateMapDescriptor {
        let desc = p.rate_map_descriptor(&RateMapDesc {
            screen_size: MtlSize::new(side, side, 1),
            mutability,
            ..RateMapDesc::default()
        });
        let layer = p
            .new_rate_map_layer(OnFault::Die, MtlSize::new(1, 1, 0))
            .unwrap()
            .unwrap();
        for at in [
            layer.horizontal_sample_storage(),
            layer.vertical_sample_storage(),
        ] {
            // SAFETY: the layer's storage holds one sample on each axis.
            unsafe { *at.as_ptr::<f32>() = 1.0 };
        }
        desc.set_layer(OnFault::Die, &layer, 0).unwrap();
        desc
    }

    /// The `target` by `target` texture's bytes after a full-screen triangle in a pass, which a
    /// command buffer waiting for an event runs, over a map of the identity over a `from` by
    /// `from` screen: mutable and not yet reset, reset `when` to an immutable descriptor of the
    /// identity over a `to` by `to` screen; or, `Made`, an immutable map of `to`. The event is
    /// signalled after the commit. A death at the reset is answered once the command buffer, if
    /// committed, completed.
    fn drawn<P: Platform>(
        p: &P,
        d: &P::Device,
        (when, target, from, to): (Reset, u64, u64, u64),
    ) -> Result<Vec<u8>, DeathSite> {
        let map = match when {
            Reset::Made => d.new_rate_map(&identity(p, to, None)),
            _ => d.new_rate_map(&identity(p, from, Some(1))),
        };
        let map = map.unwrap().unwrap();
        let (dst, pass) = pass_of(p, d, (target, target));
        pass.set_pass(&PassSet {
            rasterization_rate_map: Some(Some(&map)),
            ..Default::default()
        });
        let state = fixture_state::<P>(d, "vert:tri_vert", "frag:tri_frag");
        let q = d.new_command_queue().unwrap();
        let event = d.new_shared_event().unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        cb.encode_wait(Some(EventRef::Shared(&event)), 1).unwrap();
        let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
        for cmd in [
            RenderCmd::SetRenderPipelineState {
                pipeline_state: Some(&state),
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
        let out = buffer::<P>(d, target * target * 4, 0, &[]);
        let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
        for cmd in [
            BlitCmd::CopyTextureToBuffer {
                source_texture: &dst,
                source_slice: 0,
                source_level: 0,
                source_origin: MtlOrigin::new(0, 0, 0),
                source_size: MtlSize::new(target, target, 1),
                destination_buffer: &out,
                destination_offset: 0,
                destination_bytes_per_row: target * 4,
                destination_bytes_per_image: target * target * 4,
            },
            BlitCmd::EndEncoding,
        ] {
            blit.encode(OnFault::Die, cmd).unwrap();
        }
        let reset = |at| match when == at {
            true => catch_death(|| map.reset(&identity(p, to, None))),
            false => Ok(()),
        };
        let before = reset(Reset::BeforeCommit);
        if before.is_ok() {
            cb.commit(OnFault::Die).unwrap();
        }
        let after = reset(Reset::AfterCommit);
        event.set_signaled_value(1).unwrap();
        before?;
        cb.wait_until_completed(OnFault::Die).unwrap();
        assert!(cb.error().is_none(), "{:?}", cb.error());
        after?;
        let n = usize::try_from(target * target * 4).unwrap();
        // SAFETY: a Shared buffer's bytes, read after the blit completed.
        Ok(unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), n) }.to_vec())
    }

    /// A `target` by `target` texture as cleared, with the top left `side` by `side` texels of
    /// the fragment function's colour, (1, 0.5, 0.25, 1) as BGRA8.
    fn shaded(target: u64, side: u64) -> Vec<u8> {
        (0..target * target)
            .flat_map(|i| match (i % target < side, i / target < side) {
                (true, true) => [64, 128, 255, 255],
                _ => [0, 0, 0, 255],
            })
            .collect()
    }

    let cases = [
        ((Reset::Never, SIDE, SIDE, SIDE), 0),
        ((Reset::BeforeCommit, SIDE, SIDE, SIDE), SIDE),
        ((Reset::AfterCommit, SIDE, SIDE, SIDE), SIDE),
        ((Reset::Never, SIDE, HALF, HALF), 0),
        ((Reset::BeforeCommit, SIDE, HALF, HALF), HALF),
        ((Reset::AfterCommit, SIDE, HALF, HALF), HALF),
        ((Reset::Made, SIDE, 0, HALF), HALF),
        ((Reset::BeforeCommit, SIDE, SIDE, HALF), SIDE),
        ((Reset::AfterCommit, SIDE, SIDE, HALF), SIDE),
        ((Reset::BeforeCommit, SIDE, HALF, SIDE), HALF),
        ((Reset::AfterCommit, SIDE, HALF, SIDE), HALF),
        ((Reset::BeforeCommit, WIDE, WIDE, 32), 32),
        ((Reset::AfterCommit, WIDE, WIDE, 24), 32),
        ((Reset::BeforeCommit, WIDE, WIDE, 48), WIDE),
    ];
    let p = platform();
    let d = dev();
    for (case, side) in cases {
        assert!(drawn(&p, &d, case) == Ok(shaded(case.1, side)), "{case:?}");
    }
}

/// A reset of a mutable rasterization rate map whose rates vary to one of another screen size,
/// after a pass took it and before the pass ran, dies at the reset: no measurement says what
/// Metal's GPU draws then ([`crate::gpu::rate_map`])
#[test]
fn a_reset_of_a_varying_map_to_another_screen_dies_while_its_pass_may_run() {
    use crate::core::platform::rate_map::{
        RateMap as _, RateMapDescriptor as _, RateMapLayerDescriptor as _,
    };
    use crate::core::platform::values::{PassSet, RateMapDesc};

    /// An immutable descriptor of a `side` by `side` screen of one layer of one sample of 0.5
    /// on each axis.
    fn half(p: &VulkanPlatform, side: u64) -> <VulkanPlatform as Platform>::RateMapDescriptor {
        let desc = p.rate_map_descriptor(&RateMapDesc {
            screen_size: MtlSize::new(side, side, 1),
            ..RateMapDesc::default()
        });
        let layer = p
            .new_rate_map_layer(OnFault::Die, MtlSize::new(1, 1, 0))
            .unwrap()
            .unwrap();
        for at in [
            layer.horizontal_sample_storage(),
            layer.vertical_sample_storage(),
        ] {
            // SAFETY: the layer's storage holds one sample on each axis.
            unsafe { *at.as_ptr::<f32>() = 0.5 };
        }
        desc.set_layer(OnFault::Die, &layer, 0).unwrap();
        desc
    }

    let (p, d) = (platform(), dev());
    let made = p.rate_map_descriptor(&RateMapDesc {
        screen_size: MtlSize::new(128, 128, 1),
        mutability: Some(1),
        ..RateMapDesc::default()
    });
    let map = d.new_rate_map(&made).unwrap().unwrap();
    map.reset(&half(&p, 128));
    let (_dst, pass) = pass_of(&p, &d, (128, 128));
    pass.set_pass(&PassSet {
        rasterization_rate_map: Some(Some(&map)),
        ..Default::default()
    });
    let state = fixture_state::<VulkanPlatform>(&d, "vert:tri_vert", "frag:tri_frag");
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
    crate::answer::relaxed(|| {
        for cmd in [
            RenderCmd::SetRenderPipelineState {
                pipeline_state: Some(&state),
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
    *crate::answer::LAST_LINE.lock() = None;
    let site = catch_death(|| map.reset(&half(&p, 64))).unwrap_err();
    let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
    assert_eq!(site.file, "resetUsingDescriptor:", "{line}");
    assert!(line.contains("of a 64 by 64 screen"), "{line}");
}

/// The tile size of a one-sample pass is Metal's
#[test]
fn the_tile_size_is_metals() {
    fn tile<P: Platform>(p: &P, d: &P::Device) -> (u64, u64) {
        let (_dst, pass) = pass_of(p, d, (64, 64));
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
        let size = (e.tile_width(), e.tile_height());
        e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
        size
    }
    let vulkan = tile(&platform(), &dev());
    assert_eq!(vulkan, (32, 32));
}

/// A guest's render state on `d` of `vertex` and `value_fragment` of `simd_stages.metal` into
/// an `RGBA8Unorm` colour 0, made from the whole render value.
fn simd_state<P: Platform>(
    d: &P::Device,
    vertex: &str,
) -> Result<Option<P::RenderPipeline>, crate::core::platform::types::NsErrorText> {
    let v = fixture::<P>(d, &format!("simd_stages:{vertex}"));
    let f = fixture::<P>(d, "simd_stages:value_fragment");
    d.new_render_pipeline_state_full(&crate::gpu::device::tests::render_value::<P>(
        (&v, &f),
        1,
        &[&[(1, 70)]],
    ))
}

/// The pixels `vertex` and `value_fragment` of `simd_stages.metal` draw over [`pass_pixels`]'
/// target: three triangles of 9 vertices, each covering it, so the last, of vertices 6 to 8,
/// colours every pixel with the value its provoking vertex, 6, passes.
fn simd_vertex_pixels<P: Platform>(p: &P, d: &P::Device, vertex: &str) -> Vec<[u8; 4]> {
    let state = fixture_state::<P>(
        d,
        &format!("simd_stages:{vertex}"),
        "simd_stages:value_fragment",
    );
    pass_pixels(
        p,
        d,
        vec![
            RenderCmd::SetRenderPipelineState {
                pipeline_state: Some(&state),
            },
            RenderCmd::DrawPrimitives {
                primitive_type: 3,
                vertex_start: 0,
                vertex_count: 9,
            },
        ],
    )
}

/// A vertex function using `simd_sum` or an explicit-width `simd_shuffle_and_fill_down`,
/// which the runtime runs in the vertex stage on this device as Metal does
/// (`DriverDeviation::MetalVertexSimdgroups`), makes its state by the guest's creator and
/// draws what Metal's draws: vertex 6's ID, the sum of the 9 vertices' IDs, and the filling
/// value of vertex 1 (1 ^ 0xff), as each vertex runs in the lane of its ID
#[test]
fn simd_operations_in_a_vertex_function_draw_as_metals() {
    let (p, d) = (platform(), dev());
    for (vertex, red) in [
        ("plain_vertex", 6),
        ("sum_vertex", 36),
        ("fill_vertex", 0xfe),
    ] {
        let made = simd_state::<VulkanPlatform>(&d, vertex);
        assert!(matches!(made, Ok(Some(_))), "{vertex}: {made:?}");
        let ours = simd_vertex_pixels(&p, &d, vertex);
        assert!(
            ours.iter().all(|&px| px == [0, 0, red, 255]),
            "{vertex}: {ours:?}"
        );
    }
}

/// The words `simd_kernel` of `simd_stages.metal` writes on `d`, two a thread, over 64 threads
/// in one threadgroup: each thread's `simd_sum` and explicit-width fill.
fn simd_kernel_words<P: Platform>(d: &P::Device) -> Vec<[u32; 2]> {
    use crate::core::platform::compute::{ComputeCmd, ComputeEncoder as _};
    let f = fixture::<P>(d, "simd_stages:simd_kernel");
    let state = d
        .new_compute_pipeline_state_for_function(Some(&f))
        .unwrap()
        .unwrap();
    let out = buffer::<P>(d, 64 * 8, 0, &[]);
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let e = cb.compute_encoder(OnFault::Die).unwrap().unwrap();
    let buffers = [Some(&out)];
    for cmd in [
        ComputeCmd::SetComputePipelineState {
            state: Some(&state),
        },
        ComputeCmd::SetBuffers {
            buffers: &buffers,
            offsets: &[0],
            range: NsRange::new(0, 1),
        },
        ComputeCmd::DispatchThreads {
            threads_per_grid: MtlSize::new(64, 1, 1),
            threads_per_threadgroup: MtlSize::new(64, 1, 1),
        },
        ComputeCmd::EndEncoding,
    ] {
        e.encode(OnFault::Die, cmd).unwrap();
    }
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    assert!(cb.error().is_none(), "{:?}", cb.error());
    // SAFETY: a Shared buffer's 512 bytes, read after the dispatch completed.
    let words = unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u32>(), 128) };
    words.as_chunks::<2>().0.to_vec()
}

/// The pixels `plain_vertex` and `fragment` of `simd_stages.metal` draw over [`pass_pixels`]'
/// target, one triangle covering it.
fn simd_fragment_pixels<P: Platform>(p: &P, d: &P::Device, fragment: &str) -> Vec<[u8; 4]> {
    let state = fixture_state::<P>(
        d,
        "simd_stages:plain_vertex",
        &format!("simd_stages:{fragment}"),
    );
    pass_pixels(
        p,
        d,
        vec![
            RenderCmd::SetRenderPipelineState {
                pipeline_state: Some(&state),
            },
            RenderCmd::DrawPrimitives {
                primitive_type: 3,
                vertex_start: 0,
                vertex_count: 3,
            },
        ],
    )
}

/// A kernel and a fragment function using `simd_sum` and an explicit-width
/// `simd_shuffle_and_fill_down`, the classes of device info key 33's SIMD bits, write what
/// Metal's write: these stages take them on this device, though key 33 answers no for them
/// (`caps`)
#[test]
fn simd_operations_in_kernels_and_fragments_run_as_metals() {
    let (p, d) = (platform(), dev());
    let words = simd_kernel_words::<VulkanPlatform>(&d);
    // Each simdgroup's sum of 7i + 1 over its 32 threads, and the fill of thread 5, the
    // filling value of thread 0 in its segment of 8 (1 ^ 0xff), and of thread 4, thread 7's.
    assert_eq!(
        (words[0][0], words[32][0], words[5][1], words[4][1]),
        (3504, 10672, 0xfe, 50),
        "{words:?}"
    );
    let pixels = ["sum_fragment", "fill_fragment"].map(|f| simd_fragment_pixels(&p, &d, f));
    assert!(pixels[0].iter().all(|px| px[3] == 255), "{:?}", pixels[0]);
    let fragments = ["sum_fragment", "fill_fragment"];
    crate::metal_check_raw!("the kernel's words", words);
    for (fragment, ours) in fragments.into_iter().zip(&pixels) {
        crate::metal_check_raw!(fragment, ours);
    }
}

/// The identity colour matrix, as rows.
const IDENTITY: [[f32; 4]; 3] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
];

/// Three points from vertex 1 under `state`, of [`SIZED`]'s positions.
fn points<P: Platform>(state: &P::RenderPipeline) -> Vec<RenderCmd<'_, P>> {
    vec![
        RenderCmd::SetRenderPipelineState {
            pipeline_state: Some(state),
        },
        RenderCmd::DrawPrimitives {
            primitive_type: 0,
            vertex_start: 1,
            vertex_count: 3,
        },
    ]
}

/// The bytes of `seen`, a Private 64 by 64 `RGBA8Unorm` attachment, after a pass on platform
/// `P` whose colour 0 is a memoryless 64 by 64 `RGBA8Unorm` attachment loaded and stored
/// `DontCare`, and colour 1 `seen`, cleared to grey: it draws `fetch_seen_fragment` over both,
/// so `seen` holds what each fragment read of colour 0 by framebuffer fetch. With
/// `tiles_written`, a pass before it in its command buffer clears two other Private
/// attachments of that shape to (1, 0.5, 0.25, 1) and draws the same function over them, so
/// the tiles held other bytes before the memoryless attachment's pass.
pub(crate) fn memoryless_fetched<P: Platform>(
    p: &P,
    d: &P::Device,
    tiles_written: bool,
) -> Vec<u8> {
    use crate::core::platform::render::{RenderCmd, RenderEncoder as _};
    use crate::core::platform::resource::Library as _;

    use crate::gpu::device::tests::render_value;

    /// `MTLLoadActionDontCare` and `MTLStoreActionDontCare`.
    const DONT_CARE: u64 = 0;
    /// `MTLResourceStorageModePrivate` and `MTLResourceStorageModeMemoryless`.
    const PRIVATE: u64 = 0x20;
    const MEMORYLESS: u64 = 0x30;
    let library = d.new_library(ROUNDING).unwrap().unwrap().unwrap();
    let function = |name| library.new_function(OnFault::Die, name).unwrap().unwrap();
    let functions = [function("sweep_vertex"), function("fetch_seen_fragment")];
    let colours: [&[(u8, u32)]; 2] = [&[(0, 0), (1, 70)], &[(0, 1), (1, 70)]];
    let desc = render_value::<P>((&functions[0], &functions[1]), 1, &colours);
    let state = d.new_render_pipeline_state_full(&desc).unwrap().unwrap();
    let texture = |usage, options| {
        let desc = p.texture_descriptor(&TextureDesc {
            pixel_format: 70,
            width: 64,
            height: 64,
            usage,
            resource_options: Some(options),
            ..TextureDesc::default()
        });
        d.new_texture(OnFault::Die, &desc).unwrap().unwrap()
    };
    let memoryless = texture(TARGET, MEMORYLESS);
    let seen = texture(TARGET, PRIVATE);
    let others = [texture(TARGET, PRIVATE), texture(TARGET, PRIVATE)];
    let length = 64 * 64 * 4;
    let down = buffer::<P>(d, length, 0, &[]);
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let clear = |red, green, blue, alpha| ClearColor {
        red,
        green,
        blue,
        alpha,
    };
    let written = [
        (&others[0], CLEAR, STORE, clear(1.0, 0.5, 0.25, 1.0)),
        (&others[1], CLEAR, STORE, clear(1.0, 0.5, 0.25, 1.0)),
    ];
    let fetched = [
        (&memoryless, DONT_CARE, DONT_CARE, clear(0.0, 0.0, 0.0, 0.0)),
        (&seen, CLEAR, STORE, clear(0.5, 0.5, 0.5, 0.5)),
    ];
    let passes = match tiles_written {
        true => vec![written, fetched],
        false => vec![fetched],
    };
    for attachments in passes {
        let pass = p.render_pass_descriptor();
        for (index, (texture, load, store, clear)) in (0..).zip(attachments) {
            let attachment = ColorAttachmentSet {
                texture: Some(Some(texture)),
                load_action: Some(load),
                store_action: Some(store),
                clear_color: Some(clear),
                ..ColorAttachmentSet::default()
            };
            pass.set_color_attachment(OnFault::Die, index, &attachment)
                .unwrap();
        }
        let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
        for cmd in [
            RenderCmd::SetRenderPipelineState {
                pipeline_state: Some(&state),
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
    let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
    for cmd in [
        BlitCmd::CopyTextureToBuffer {
            source_texture: &seen,
            source_slice: 0,
            source_level: 0,
            source_origin: MtlOrigin::new(0, 0, 0),
            source_size: MtlSize::new(64, 64, 1),
            destination_buffer: &down,
            destination_offset: 0,
            destination_bytes_per_row: 64 * 4,
            destination_bytes_per_image: length,
        },
        BlitCmd::EndEncoding,
    ] {
        blit.encode(OnFault::Die, cmd).unwrap();
    }
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    assert!(cb.error().is_none(), "{:?}", cb.error());
    let n = usize::try_from(length).unwrap();
    // SAFETY: a Shared buffer's bytes, read after the command buffer completed.
    unsafe { std::slice::from_raw_parts(down.contents().as_ptr::<u8>(), n) }.to_vec()
}

/// What [`filling`]'s completed handlers saw: each one's fill value, status and error code.
type Seen = Arc<crate::core::sync::Mutex<Vec<(u8, u64, Option<i64>)>>>;

/// A command buffer of `q` that waits for `wait` where given, fills a new 16-byte buffer of `d`'s
/// with `value` in a blit, then signals a new event of its own to 1, with a completed handler that
/// keeps what it saw in `seen`; the buffer, and the event.
fn filling<P: Platform>(
    d: &P::Device,
    q: &P::CommandQueue,
    (value, wait): (u8, Option<&P::SharedEvent>),
    seen: &Seen,
) -> (P::CommandBuffer, P::Buffer, P::SharedEvent) {
    use crate::core::platform::event::EventRef;
    let b = buffer::<P>(d, 16, 0, &[0; 16]);
    let event = d.new_shared_event().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    if let Some(wait) = wait {
        cb.encode_wait(Some(EventRef::Shared(wait)), 1).unwrap();
    }
    let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
    for cmd in [
        BlitCmd::FillBuffer {
            buffer: &b,
            range: NsRange::new(0, 16),
            value,
        },
        BlitCmd::EndEncoding,
    ] {
        blit.encode(OnFault::Die, cmd).unwrap();
    }
    cb.encode_signal(Some(EventRef::Shared(&event)), 1).unwrap();
    let seen = Arc::clone(seen);
    cb.add_completed_handler(
        OnFault::Die,
        Box::new(move |info| {
            let code = info.error.map(|e| e.code);
            seen.lock().push((value, info.status.0, code));
        }),
    )
    .unwrap();
    (cb, b, event)
}

/// A draw of primitive type 9 of [`type_9_pass`]'s: directly, `(count, instances)`; over
/// `(count, instances)` of the 16-bit indices 0, 1 and 2; from indirect arguments of 3 vertices
/// and an instance; or from indirect arguments of those indices and an instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Nine {
    Direct(u64, u64),
    Indexed(u64, u64),
    Indirect,
    IndexedIndirect,
}

/// What a command buffer of a new queue shows on `p`'s device `d` whose render pass, over a 4 by
/// 4 target filled with 0x11, draws primitive type 9 as `draw` says ([`type_9_pass_on`]).
fn type_9_pass<P: Platform>(p: &P, d: &P::Device, draw: Nine) -> String {
    type_9_pass_on(p, d, &d.new_command_queue().unwrap(), draw)
}

/// What a command buffer of `q` shows on `p`'s device `d` whose render pass, over a 4 by 4 target
/// filled with 0x11, draws primitive type 9 as `draw` says, a blit before it filling a buffer
/// with 0x5a and one after it another with 0x6b, then an event signal: its status and error, the
/// first byte of each buffer, whether the signal fired, and the target's first texel, black
/// where the pass's clear ran, read through a new queue.
fn type_9_pass_on<P: Platform>(p: &P, d: &P::Device, q: &P::CommandQueue, draw: Nine) -> String {
    use crate::core::platform::event::{EventRef, SharedEvent as _};
    let (dst, pass) = pass_of(p, d, (4, 4));
    let region = MtlRegion::new(MtlOrigin::new(0, 0, 0), MtlSize::new(4, 4, 1));
    dst.replace_region(region, 0, 0, HostRange::of_slice(&[0x11; 64]), 16, 0)
        .unwrap();
    let state = fixture_state::<P>(d, "vert:tri_vert", "frag:tri_frag");
    let (before, after) = (
        buffer::<P>(d, 16, 0, &[0; 16]),
        buffer::<P>(d, 16, 0, &[0; 16]),
    );
    let indices = buffer::<P>(d, 16, 0, &index_bytes(U16, &[0, 1, 2]));
    let arguments = match draw {
        Nine::IndexedIndirect => u32s(&[3, 1, 0, 0, 0]),
        _ => u32s(&[3, 1, 0, 0]),
    };
    let arguments = buffer::<P>(d, 32, 0, &arguments);
    let event = d.new_shared_event().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let fill = |b: &P::Buffer, value| {
        let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
        let range = NsRange::new(0, 16);
        let fill = BlitCmd::FillBuffer {
            buffer: b,
            range,
            value,
        };
        blit.encode(OnFault::Die, fill).unwrap();
        blit.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
    };
    fill(&before, 0x5a);
    let drawn = match draw {
        Nine::Direct(count, instances) => RenderCmd::DrawPrimitivesInstances {
            primitive_type: 9,
            vertex_start: 0,
            vertex_count: count,
            instance_count: instances,
        },
        Nine::Indexed(count, instances) => RenderCmd::DrawIndexedPrimitivesInstances {
            primitive_type: 9,
            index_count: count,
            index_type: U16,
            index_buffer: &indices,
            index_buffer_offset: 0,
            instance_count: instances,
        },
        Nine::Indirect => RenderCmd::DrawPrimitivesIndirect {
            primitive_type: 9,
            indirect_buffer: &arguments,
            indirect_buffer_offset: 0,
        },
        Nine::IndexedIndirect => RenderCmd::DrawIndexedPrimitivesIndirect {
            primitive_type: 9,
            index_type: U16,
            index_buffer: &indices,
            index_buffer_offset: 0,
            indirect_buffer: &arguments,
            indirect_buffer_offset: 0,
        },
    };
    let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
    for cmd in [
        RenderCmd::SetRenderPipelineState {
            pipeline_state: Some(&state),
        },
        drawn,
        RenderCmd::EndEncoding,
    ] {
        e.encode(OnFault::Die, cmd).unwrap();
    }
    fill(&after, 0x6b);
    cb.encode_signal(Some(EventRef::Shared(&event)), 5).unwrap();
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    let out = buffer::<P>(d, 4, 0, &[]);
    let read = d.new_command_queue().unwrap();
    let read = read.command_buffer(OnFault::Die).unwrap().unwrap();
    let blit = read.blit_encoder(OnFault::Die).unwrap().unwrap();
    for cmd in [
        BlitCmd::CopyTextureToBuffer {
            source_texture: &dst,
            source_slice: 0,
            source_level: 0,
            source_origin: MtlOrigin::new(0, 0, 0),
            source_size: MtlSize::new(1, 1, 1),
            destination_buffer: &out,
            destination_offset: 0,
            destination_bytes_per_row: 4,
            destination_bytes_per_image: 4,
        },
        BlitCmd::EndEncoding,
    ] {
        blit.encode(OnFault::Die, cmd).unwrap();
    }
    read.commit(OnFault::Die).unwrap();
    read.wait_until_completed(OnFault::Die).unwrap();
    // SAFETY: Shared buffers' bytes, read after their command buffers completed.
    let first = |b: &P::Buffer| unsafe { *b.contents().as_ptr::<u8>() };
    // SAFETY: as above, the 4 bytes of `out`.
    let texel = unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), 4) };
    format!(
        "status {} error {:?} before {:#x} after {:#x} signal {} texel {texel:x?}",
        cb.status().0,
        cb.error().map(|e| e.code),
        first(&before),
        first(&after),
        event.wait_until_signaled_value(5, 1),
    )
}

/// [`type_9_pass`]'s lines on Metal, each of its draws: directly, of 3 vertices or more and an
/// instance it hangs the GPU, failing the command buffer with `MTLCommandBufferErrorTimeout`
/// (status 5, code 2), the blit before the pass run and neither the pass's clear nor the blit
/// after it, the signal firing; of fewer, or of no instance, it fails nothing. Over indices, of
/// any count or instances, it faults the GPU (`MTLCommandBufferErrorPageFault`, code 3), and from
/// indirect arguments, over indices or not, fails the command buffer as a victim of the GPU's
/// recovery (`MTLCommandBufferErrorInternal`, code 1), each as a hang fails it: Metal's answer in
/// a steady stream of work, which the runtime takes. There Metal's GPU races: in some runs the
/// blit after the pass runs too, as where the GPU was idle for some seconds before (measured on
/// the M3 Max, macOS 27.0.1, drawing these after the hangs).
const METALS_TYPE_9: [(Nine, &str); 11] = [
    (
        Nine::Direct(0, 1),
        "status 4 error None before 0x5a after 0x6b signal true texel [0, 0, 0, ff]",
    ),
    (
        Nine::Direct(1, 1),
        "status 4 error None before 0x5a after 0x6b signal true texel [0, 0, 0, ff]",
    ),
    (
        Nine::Direct(2, 1),
        "status 4 error None before 0x5a after 0x6b signal true texel [0, 0, 0, ff]",
    ),
    (
        Nine::Direct(3, 0),
        "status 4 error None before 0x5a after 0x6b signal true texel [0, 0, 0, ff]",
    ),
    (
        Nine::Direct(3, 1),
        "status 5 error Some(2) before 0x5a after 0x0 signal true texel [11, 11, 11, 11]",
    ),
    (
        Nine::Direct(6, 2),
        "status 5 error Some(2) before 0x5a after 0x0 signal true texel [11, 11, 11, 11]",
    ),
    (
        Nine::Indexed(3, 1),
        "status 5 error Some(3) before 0x5a after 0x0 signal true texel [11, 11, 11, 11]",
    ),
    (
        Nine::Indexed(0, 1),
        "status 5 error Some(3) before 0x5a after 0x0 signal true texel [11, 11, 11, 11]",
    ),
    (
        Nine::Indexed(3, 0),
        "status 5 error Some(3) before 0x5a after 0x0 signal true texel [11, 11, 11, 11]",
    ),
    (
        Nine::Indirect,
        "status 5 error Some(1) before 0x5a after 0x0 signal true texel [11, 11, 11, 11]",
    ),
    (
        Nine::IndexedIndirect,
        "status 5 error Some(1) before 0x5a after 0x0 signal true texel [11, 11, 11, 11]",
    ),
];

/// [`memoryless_slot_pass`]'s lines are Metal's, [`METALS_MEMORYLESS_SLOT`]
#[test]
fn memoryless_slots_fail_as_metals() {
    let (p, d) = (platform(), dev());
    for (fragment, metals) in METALS_MEMORYLESS_SLOT {
        assert_eq!(memoryless_slot_pass(&p, &d, fragment), metals, "{fragment}");
    }
}

/// A draw of primitive type 9 fails its command buffer as Metal's GPU does ([`METALS_TYPE_9`]):
/// directly it hangs at 3 vertices or more and an instance, and fails nothing at fewer or no
/// instance; over indices it faults, and from indirect arguments it fails a victim of the GPU's
/// recovery, each of a new queue
#[test]
fn type_9_draws_fail_their_command_buffers_as_metals() {
    let (p, d) = (platform(), dev());
    for (draw, metals) in METALS_TYPE_9 {
        assert_eq!(type_9_pass(&p, &d, draw), metals, "{draw:?}");
    }
}

/// What a new queue of `p`'s device `d` shows once `draws`, command buffers of
/// [`type_9_pass_on`]'s, were committed on it in turn, each once the one before completed: each
/// one's status and error, then a command buffer's committed on it once the last completed,
/// which fills a buffer and signals an event ([`filling`]), and one's committed so on another
/// new queue: each one's status and error, the buffer's first byte and whether the signal fired.
/// What else of a type-9 command buffer ran is left out: on Metal it races, the encoders
/// after a fault running where the GPU was idle before (the process's first fault here), and the
/// queue's second fault running nothing of its command buffer, the blit before it included, in
/// some runs (the runtime's measurement, `command_buffer/revocation.rs`), where
/// [`type_9_draws_fail_as_metals`] compares it on a GPU that ran just before.
fn after_type_9s<P: Platform>(p: &P, d: &P::Device, draws: &[Nine]) -> Vec<String> {
    use crate::core::platform::event::SharedEvent as _;
    let q = d.new_command_queue().unwrap();
    let mut lines: Vec<String> = (draws.iter())
        .map(|&draw| {
            let line = type_9_pass_on(p, d, &q, draw);
            let failed = line.split(" before ").next().unwrap_or_default();
            format!("{draw:?}: {failed}")
        })
        .collect();
    let seen = Seen::default();
    for (what, q) in [
        ("its queue", &q),
        ("a new queue", &d.new_command_queue().unwrap()),
    ] {
        let (cb, b, event) = filling::<P>(d, q, (0x11, None), &seen);
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        // SAFETY: a Shared buffer's bytes, read after its command buffer completed.
        let first = unsafe { *b.contents().as_ptr::<u8>() };
        lines.push(format!(
            "{what}: status {} error {:?} filled {first:#x} signal {}",
            cb.status().0,
            cb.error().map(|e| e.code),
            event.wait_until_signaled_value(1, 1),
        ));
    }
    lines
}

/// [`after_type_9s`]'s draws: two faults, a hang then a fault, and a victim of the GPU's recovery
/// then a fault.
const REVOKING: [&[Nine]; 3] = [
    &[Nine::Indexed(3, 1), Nine::Indexed(3, 1)],
    &[Nine::Direct(3, 1), Nine::Indexed(3, 1)],
    &[Nine::Indirect, Nine::Indexed(3, 1)],
];

/// The last two of [`after_type_9s`]'s lines on Metal after each of [`REVOKING`]'s: a queue whose
/// second command buffer that a draw of primitive type 9 faulted or hung completed fails every
/// command buffer committed on it after that with `MTLCommandBufferErrorAccessRevoked` (status 5,
/// code 4), running none of its work and firing none of its signals, where a new queue runs; a
/// victim of the GPU's recovery does not count (measured on the M3 Max, macOS 27.0.1,
/// `type_9_faults_revoke_queues_as_metals`).
const METALS_REVOKING: [[&str; 2]; 3] = [
    [
        "its queue: status 5 error Some(4) filled 0x0 signal false",
        "a new queue: status 4 error None filled 0x11 signal true",
    ],
    [
        "its queue: status 5 error Some(4) filled 0x0 signal false",
        "a new queue: status 4 error None filled 0x11 signal true",
    ],
    [
        "its queue: status 4 error None filled 0x11 signal true",
        "a new queue: status 4 error None filled 0x11 signal true",
    ],
];

/// A queue's second command buffer that a draw of primitive type 9 faulted or hung revokes it,
/// as Metal's ([`METALS_REVOKING`]), and a victim of the GPU's recovery does not: the type-9
/// command buffers fail with the status and error [`METALS_TYPE_9`] says
#[test]
fn type_9_faults_revoke_queues_as_on_metal() {
    let (p, d) = (platform(), dev());
    for (draws, metals) in REVOKING.into_iter().zip(METALS_REVOKING) {
        let lines = after_type_9s(&p, &d, draws);
        let (failed, later) = lines.split_at(draws.len());
        for (line, draw) in failed.iter().zip(draws) {
            let (_, metals) = METALS_TYPE_9.iter().find(|(d, _)| d == draw).unwrap();
            let status = metals.split(" before ").next().unwrap_or_default();
            assert_eq!(*line, format!("{draw:?}: {status}"));
        }
        assert_eq!(later, metals, "{draws:?}");
    }
}

/// A step of a visibility case's pass: `setVisibilityResultMode:offset:` of a mode at an offset,
/// a draw of `render_depth`'s triangle over the whole target at a depth (0.25 passes the depth
/// test of a pass with depth, 0.75 fails it), or a scissor rectangle over the target's left
/// columns.
#[derive(Clone, Copy, Debug)]
enum Vis {
    Mode(u64, u64),
    Draw(f32),
    Scissor(u64),
}

/// A pass of a visibility case: its samples, whether it has a `Depth32Float` attachment
/// (cleared to 0.5, tested `Less`, unwritten), its `visibilityResultType`, and its steps.
#[derive(Clone, Copy, Debug)]
struct VisPass<'a> {
    samples: u64,
    depth: bool,
    result_type: u64,
    steps: &'a [Vis],
}

/// The side of a visibility case's target.
const VIS_SIDE: u64 = 16;

/// The 64-bit words of a visibility result buffer holding `init` after `passes` drew over it in
/// one command buffer, each pass over its own `VIS_SIDE` square BGRA8 target, cleared.
fn visibility_results<P: Platform>(
    p: &P,
    d: &P::Device,
    init: &[u64],
    passes: &[VisPass<'_>],
) -> Vec<u64> {
    use crate::core::platform::values::{DepthStencilDesc, PassSet};

    let words: Vec<u8> = init.iter().flat_map(|w| w.to_ne_bytes()).collect();
    let length = u64::try_from(words.len()).unwrap();
    let results = buffer::<P>(d, length, 0, &words);
    let tested = d
        .new_depth_stencil_state(&DepthStencilDesc {
            depth_compare_function: Some(1), // Less
            depth_write_enabled: Some(false),
            front: None,
            back: None,
        })
        .unwrap()
        .unwrap();
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let (mut kept_buffers, mut kept_textures) = (Vec::new(), Vec::new());
    for pass in passes {
        let state = value_state::<P>(d, ("render_depth", "render_color"), |v| {
            v.raster_sample_count = pass.samples;
            if pass.depth {
                v.depth_pixel_format = 252; // Depth32Float
            }
        });
        let attachment = |format: u64| {
            let desc = p.texture_descriptor(&TextureDesc {
                texture_type: if pass.samples > 1 { 4 } else { 2 },
                pixel_format: format,
                width: VIS_SIDE,
                height: VIS_SIDE,
                sample_count: pass.samples,
                usage: TARGET,
                resource_options: Some(0x20), // Private
                ..TextureDesc::default()
            });
            d.new_texture(OnFault::Die, &desc).unwrap().unwrap()
        };
        let colour = attachment(BGRA8);
        let rp = p.render_pass_descriptor();
        rp.set_color_attachment(
            OnFault::Die,
            0,
            &ColorAttachmentSet {
                texture: Some(Some(&colour)),
                load_action: Some(CLEAR),
                store_action: Some(0), // DontCare
                ..ColorAttachmentSet::default()
            },
        )
        .unwrap();
        let depth = pass.depth.then(|| attachment(252));
        if let Some(depth) = &depth {
            rp.set_depth_attachment(&crate::core::platform::values::DepthAttachmentSet {
                texture: Some(Some(depth)),
                load_action: Some(CLEAR),
                store_action: Some(0),
                clear_depth: Some(0.5),
                ..Default::default()
            });
        }
        rp.set_pass(&PassSet {
            visibility_result_buffer: Some(Some(&results)),
            visibility_result_type: Some(pass.result_type),
            ..PassSet::default()
        });
        let e = cb.render_encoder(OnFault::Die, &rp).unwrap().unwrap();
        e.encode(
            OnFault::Die,
            RenderCmd::SetRenderPipelineState {
                pipeline_state: Some(&state),
            },
        )
        .unwrap();
        if pass.depth {
            e.encode(
                OnFault::Die,
                RenderCmd::SetDepthStencilState {
                    depth_stencil_state: Some(&tested),
                },
            )
            .unwrap();
        }
        for step in pass.steps {
            match *step {
                Vis::Mode(mode, offset) => e
                    .encode(
                        OnFault::Die,
                        RenderCmd::SetVisibilityResultMode { mode, offset },
                    )
                    .unwrap(),
                Vis::Scissor(width) => e
                    .encode(
                        OnFault::Die,
                        RenderCmd::SetScissorRect {
                            rect: MtlScissorRect {
                                x: 0,
                                y: 0,
                                width,
                                height: VIS_SIDE,
                            },
                        },
                    )
                    .unwrap(),
                Vis::Draw(z) => {
                    let params = [z, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
                    let params = buffer::<P>(d, 32, 0, &bytes(&params));
                    e.encode(
                        OnFault::Die,
                        RenderCmd::SetVertexBuffer {
                            buffer: Some(&params),
                            offset: 0,
                            index: 0,
                        },
                    )
                    .unwrap();
                    e.encode(
                        OnFault::Die,
                        RenderCmd::DrawPrimitives {
                            primitive_type: 3,
                            vertex_start: 0,
                            vertex_count: 3,
                        },
                    )
                    .unwrap();
                    kept_buffers.push(params);
                }
            }
        }
        e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
        kept_textures.push(colour);
        kept_textures.extend(depth);
    }
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    assert!(cb.error().is_none(), "{passes:?}: {:?}", cb.error());
    let n = init.len();
    // SAFETY: a Shared buffer of `n` words, which no command uses any more.
    let words = unsafe { std::slice::from_raw_parts(results.contents().as_ptr::<u64>(), n) };
    words.to_vec()
}

/// A visibility result buffer's words, each Metal's own fill until a pass writes it.
const VIS_FILL: u64 = 0xeeee_eeee_eeee_eeee;

/// A case of [`visibility_results_are_metals`]: its name, its passes, the buffer's words before
/// them and the words Metal leaves.
type VisCase<'a> = (&'static str, Vec<VisPass<'a>>, [u64; 4], [u64; 4]);

/// A pass's visibility results are Metal's: a `Counting` result is the samples that passed the
/// depth test (every sample of the target, at 1 and 4 samples, the scissor's alone, none where
/// the depth test fails, two draws' together at one offset), a `Boolean` one 1 where any did and
/// 0 where none did; an offset set with no draw after it gets 0, one left set at the encoder's
/// end gets its draws', one never set keeps its bytes, and a later pass in the command buffer
/// rewrites only its own offsets; `Disabled` writes nothing. A pass of type `Accumulate` resets
/// its results as the AMD Radeon Pro 5300M and the Intel UHD 630 do under macOS 26.7.1, where
/// Metal's docs promise accumulation and the M3 Max accumulates: the runtime does not build it.
/// Left out, as the Macs differ: a draw after `Disabled` at an offset once a mode was set, which
/// the M3 Max writes 0 at and the x86-64 Macs, and the runtime, leave.
#[test]
fn visibility_results_are_metals() {
    use Vis::{Draw, Mode, Scissor};
    let (p, d) = (platform(), dev());
    let (all, quad) = (VIS_SIDE * VIS_SIDE, VIS_SIDE * VIS_SIDE * 4);
    let one = |steps| VisPass {
        samples: 1,
        depth: false,
        result_type: 0,
        steps,
    };
    let fill = [VIS_FILL; 4];
    let modes: &[Vis] = &[Mode(2, 0), Draw(0.25), Mode(1, 8), Draw(0.25), Mode(0, 0)];
    let unclosed: &[Vis] = &[Mode(2, 0), Draw(0.25), Mode(1, 8), Draw(0.25)];
    let undrawn: &[Vis] = &[Mode(2, 0), Mode(1, 8)];
    let scissored: &[Vis] = &[Scissor(4), Mode(2, 0), Draw(0.25), Mode(1, 8), Draw(0.25)];
    let twice: &[Vis] = &[Mode(2, 0), Draw(0.25), Draw(0.25), Mode(1, 8), Draw(0.25)];
    let depth_tested: &[Vis] = &[
        Mode(2, 0),
        Draw(0.75),
        Mode(1, 8),
        Draw(0.75),
        Mode(2, 16),
        Draw(0.25),
        Mode(1, 24),
        Draw(0.25),
    ];
    let disabled: &[Vis] = &[Mode(0, 8), Draw(0.25)];
    let later: &[Vis] = &[Mode(2, 16), Draw(0.25)];
    let accumulating: &[Vis] = &[
        Mode(2, 0),
        Draw(0.25),
        Mode(1, 8),
        Draw(0.25),
        Mode(1, 16),
        Mode(2, 24),
    ];
    let cases: Vec<VisCase<'_>> = vec![
        (
            "modes",
            vec![one(modes)],
            fill,
            [all, 1, VIS_FILL, VIS_FILL],
        ),
        (
            "unclosed",
            vec![one(unclosed)],
            fill,
            [all, 1, VIS_FILL, VIS_FILL],
        ),
        (
            "undrawn",
            vec![one(undrawn)],
            fill,
            [0, 0, VIS_FILL, VIS_FILL],
        ),
        (
            "scissored",
            vec![one(scissored)],
            fill,
            [all / 4, 1, VIS_FILL, VIS_FILL],
        ),
        (
            "twice",
            vec![one(twice)],
            fill,
            [2 * all, 1, VIS_FILL, VIS_FILL],
        ),
        (
            "multisampled",
            vec![VisPass {
                samples: 4,
                ..one(unclosed)
            }],
            fill,
            [quad, 1, VIS_FILL, VIS_FILL],
        ),
        (
            "depth tested",
            vec![VisPass {
                depth: true,
                ..one(depth_tested)
            }],
            fill,
            [0, 0, all, 1],
        ),
        (
            "multisampled, depth tested",
            vec![VisPass {
                samples: 4,
                depth: true,
                ..one(depth_tested)
            }],
            fill,
            [0, 0, quad, 1],
        ),
        ("disabled", vec![one(disabled)], fill, fill),
        (
            "two passes",
            vec![one(modes), one(later)],
            fill,
            [all, 1, all, VIS_FILL],
        ),
    ];
    for (name, passes, init, want) in &cases {
        let ours = visibility_results(&p, &d, init, passes);
        assert_eq!(ours, want, "{name}");
    }
    let start = [100, 5, 7, 9];
    let accumulated = [VisPass {
        result_type: 1,
        ..one(accumulating)
    }];
    let ours = visibility_results(&p, &d, &start, &accumulated);
    assert_eq!(ours, [all, 1, 0, 0]);
}
