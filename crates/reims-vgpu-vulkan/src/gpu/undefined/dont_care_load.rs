//! `dont-care-load-tiles`: what a render pass that loads an attachment by `DontCare` leaves in
//! it, and what a `DontCare` or `MultisampleResolve` store leaves.
//!
//! **Metal's rule** (this host, every probe below the same in each of 5 runs):
//! - A pass's attachments are stored a tile at a time: 32 by 32 pixels at 1 sample, 32 by 16 at
//!   4. A tile is *reached* where a draw's primitive covers a pixel or sample centre of it (a
//!   draw of two vertices of a triangle list, and a triangle too small to hold a centre, reach
//!   none).
//! - A `DontCare`-loaded colour attachment the pass stores (`Store`, or
//!   `StoreAndMultisampleResolve` for its samples) holds zero in every texel of a reached tile
//!   that no fragment wrote, in **every** colour attachment of the pass: one the fragment
//!   function writes no output for, and one whose write mask is none, included (and one whose
//!   fragments all discard: an `ObjC` probe of this host). Every tile no draw reaches keeps its
//!   texels, and so does every tile of a pass with no draw call or whose draws reach nothing.
//! - A `DontCare`-loaded depth (stencil) attachment holds the pass's clear depth (stencil) in
//!   each reached tile only where some draw of the pass is under a depth-stencil state that
//!   writes depth (`depthWriteEnabled`) or stencil (`Replace` on pass, write mask 0xff). The
//!   rule is the pass's: a draw under a nil state reaches its tiles for the clear once another
//!   draw of the pass writes the aspect. Where no draw writes it, every tile keeps its texels,
//!   the reached ones too (a `Less` compare without writes, and a stencil `Replace` of write mask
//!   0, wrote nothing, and a `Never` compare with writes cleared: an `ObjC` probe of this host).
//! - A `DontCare` store, after any load action, drawn or not, keeps every texel the texture held
//!   before the pass (single-sample colour, depth and stencil); `MultisampleResolve` keeps every
//!   sample as `DontCare` does, and `StoreAndMultisampleResolve` stores the samples as `Store`.
//! - Alike Private and Shared, tile-aligned (64 by 64) or not (100 by 70), on a texture never
//!   written (a Private one's fixed pattern, a Shared one's contents, are kept), uploaded by a
//!   blit, or cleared by an earlier pass, for `RGBA8Unorm`, `RGBA16Float` and `RGBA32Float`
//!   colours, `Depth32Float` and `Stencil8` (separate textures), and 4-sample `RGBA8Unorm`.
//!
//! **Not probed**: other formats and sample counts; depth and stencil in one
//! `Depth32Float_Stencil8` texture (an `ObjC` probe of this host answered as for separate ones,
//! not recorded here); layered passes, a level or slice past the first, a rasterization-rate
//! map, tile dispatches and executions of indirect commands; points and lines; a stencil state
//! whose failure operations alone write; a depth-writing state bound only to a draw that reaches
//! nothing; other Apple GPU generations.
//!
//! **This platform today** answers a pass with no draw call as Metal. In a pass with a draw call
//! that submits vertices (two of a triangle list, or a triangle too small to hold a pixel centre,
//! included), the runtime clears a `DontCare`-loaded attachment to zero (a colour) or its clear
//! value (depth, stencil) over its draws' viewports and scissors, where Metal clears only the
//! reached tiles, and announces `dont-care-load-tiles`, relayed ([`crate::answer`]); it clears a
//! depth or stencil attachment whatever the depth-stencil states write, where Metal keeps it when
//! no draw writes it. Its `DontCare` and `MultisampleResolve` stores are Metal's. The tests hold
//! Metal's rule and require each difference of this platform's to come with the relayed kind.
//!
//! **To answer as Metal**: before the pass, write zero (colour) or the clear value (depth or
//! stencil, only where a draw of the pass binds a state that writes that aspect) into the tiles
//! the draws' primitives reach (the runtime's tile-reach rule and capture replay, which its load
//! pass of extended-range attachments uses), then load every attachment by `Load`.

use crate::core::platform::Platform;
use metal2vulkan_runtime as rt;

use super::{
    Aspect, CLEAR, Colour, DEPTH32F, DONT_CARE, Draw, IN_TILE_1_1, LOAD, MULTISAMPLE_RESOLVE,
    PRIVATE, Pass, RGBA8, RGBA16F, RGBA32F, SHARED, STENCIL8, STORE, STORE_AND_RESOLVE, Shape,
    Step, buffer, heard, masked_state, positions, random_bytes, run, state, writing,
};
use crate::gpu::device::tests::dev;
use crate::platform::tests::platform;

const KIND: rt::Degradation = rt::Degradation::DontCareLoadTiles;

/// What a probed texture held before its pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Earlier {
    /// Nothing: a new texture.
    Fresh,
    /// [`random_bytes`] by a blit (a depth texture: floats in [0, 1)).
    Uploaded,
    /// A pass's `Clear`, stored.
    Cleared,
}

/// The draws of a probed pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Drawn {
    /// No draw call.
    None,
    /// [`IN_TILE_1_1`].
    InTile,
    /// Two vertices of a triangle list: a draw call that draws nothing.
    TwoVertices,
    /// A triangle inside tile (1, 0) too small to hold a pixel centre.
    Sliver,
}

impl Drawn {
    fn pixels(self) -> Option<&'static [(f32, f32)]> {
        match self {
            Drawn::None => None,
            Drawn::InTile | Drawn::TwoVertices => Some(&IN_TILE_1_1),
            Drawn::Sliver => Some(&[(40.1, 4.1), (40.3, 4.1), (40.1, 4.3)]),
        }
    }

    fn vertices(self) -> u64 {
        if self == Drawn::TwoVertices { 2 } else { 3 }
    }
}

/// The bytes `shape`'s texture holds after `earlier`, uploaded.
fn earlier_bytes(shape: Shape) -> Vec<u8> {
    if Aspect::of(shape.format) == Aspect::Depth {
        (0..shape.length() / 4)
            .flat_map(|i| (f32::from(u16::try_from(i % 997).unwrap()) / 1000.0).to_le_bytes())
            .collect()
    } else {
        random_bytes(shape.length(), 17)
    }
}

/// The steps that leave `earlier` in `t`.
fn earlier_steps<'a, P: Platform>(
    t: &'a P::Texture,
    shape: Shape,
    earlier: Earlier,
    bytes: &'a [u8],
) -> Vec<Step<'a, P>> {
    let aspect = Aspect::of(shape.format);
    match earlier {
        Earlier::Fresh => Vec::new(),
        Earlier::Uploaded => vec![Step::Upload(t, shape, bytes)],
        Earlier::Cleared => vec![Step::Pass(Pass::of(t, aspect, (CLEAR, STORE), Vec::new()))],
    }
}

/// The bytes of a single-sample `shape` before and after a pass that loads it by `load`, stores
/// it by `store` and draws `drawn`, the draw under a depth-stencil state that writes its aspect
/// where `writes`.
fn single<P: Platform>(
    (p, d): (&P, &P::Device),
    (shape, earlier): (Shape, Earlier),
    (load, store): (u64, u64),
    (drawn, writes): (Drawn, bool),
) -> (Vec<u8>, Vec<u8>) {
    let aspect = Aspect::of(shape.format);
    let t = shape.texture(p, d);
    let bytes = earlier_bytes(shape);
    let colours: &[u64] = if aspect == Aspect::Colour {
        &[shape.format]
    } else {
        &[]
    };
    let formats = match aspect {
        Aspect::Colour => (None, None),
        Aspect::Depth => (Some(shape.format), None),
        Aspect::Stencil => (None, Some(shape.format)),
    };
    let s = state::<P>(d, colours, formats, 1);
    let ds = if writes {
        writing::<P>(d, aspect)
    } else {
        None
    };
    let at = drawn
        .pixels()
        .map(|px| buffer::<P>(d, 48, &positions(px, shape.size)));
    let draws = at
        .iter()
        .map(|positions| Draw {
            state: &s,
            depth_stencil: ds.as_ref(),
            positions,
            vertices: drawn.vertices(),
        })
        .collect();
    let mut steps = earlier_steps(&t, shape, earlier, &bytes);
    steps.push(Step::Read(&t, shape));
    steps.push(Step::Pass(Pass::of(&t, aspect, (load, store), draws)));
    steps.push(Step::Read(&t, shape));
    let mut reads = run(p, d, &steps).into_iter();
    (reads.next().unwrap(), reads.next().unwrap())
}

/// Compares this platform's answer to Metal's for each case `probe` names: equal, or with
/// `dont-care-load-tiles` announced. Answers the cases that differ. Metal's answers are the ones
/// recorded.
fn against_metal<C: std::fmt::Debug + Copy>(
    cases: &[C],
    mut ours: impl FnMut(C) -> (Vec<u8>, Vec<u8>),
) -> Vec<String> {
    let mut differ = Vec::new();
    for &c in cases {
        let key = format!("{c:?}");
        let o = ours(c);
        if !crate::metal_matches_raw!(&key, o) {
            assert!(
                heard(KIND),
                "{c:?}: off Metal's, with no {} heard",
                KIND.code()
            );
            differ.push(format!("{c:?}"));
        }
    }
    differ
}

/// A `DontCare` load of a colour attachment, stored, zeroes the tiles a draw's primitive reaches
/// and keeps every other tile, as Metal does on this host in every run: `RGBA8Unorm`,
/// `RGBA16Float` and `RGBA32Float`, 64 by 64 and 100 by 70, Private and Shared, fresh, uploaded
/// and cleared before the pass; a pass with no draw call, a draw of two vertices of a triangle
/// list, or a triangle too small to hold a pixel centre keeps every tile. This platform's answer
/// is Metal's, or the runtime announced `dont-care-load-tiles`
#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "MoltenVK: the runtime records device-nan-encoding, a death in strict mode; Mac-only test, off everywhere; ledger: MoltenVK strict deaths"
)]
fn colour_dont_care_loads_zero_the_reached_tiles_alone() {
    let (p, d) = (platform(), dev());
    let mut cases = Vec::new();
    for format in [RGBA8, RGBA16F, RGBA32F] {
        for size in [(64, 64), (100, 70)] {
            for options in [PRIVATE, SHARED] {
                for earlier in [Earlier::Fresh, Earlier::Uploaded, Earlier::Cleared] {
                    for drawn in [
                        Drawn::None,
                        Drawn::InTile,
                        Drawn::TwoVertices,
                        Drawn::Sliver,
                    ] {
                        let shape = Shape {
                            options,
                            ..Shape::of(format, size)
                        };
                        cases.push((shape, earlier, drawn));
                    }
                }
            }
        }
    }
    let actions = (DONT_CARE, STORE);
    let differ = against_metal(&cases, |(shape, earlier, drawn)| {
        single((&p, &d), (shape, earlier), actions, (drawn, false))
    });
    eprintln!("off Metal's, dont-care-load-tiles heard: {differ:#?}");
}

/// A `DontCare` load of a depth or stencil attachment, stored, holds the clear value in the
/// tiles a draw's primitive reaches where the pass's depth-stencil state writes the aspect, and
/// keeps every tile, the reached ones too, where it writes nothing (a nil state): as Metal does
/// on this host in every run, `Depth32Float` and `Stencil8`, 64 by 64 and 100 by 70, uploaded or
/// cleared before the pass. This platform's answer is Metal's, or the runtime announced
/// `dont-care-load-tiles`
#[test]
fn depth_and_stencil_dont_care_loads_take_the_clear_value_where_the_pass_writes() {
    let (p, d) = (platform(), dev());
    let mut cases = Vec::new();
    for format in [DEPTH32F, STENCIL8] {
        for size in [(64, 64), (100, 70)] {
            for earlier in [Earlier::Uploaded, Earlier::Cleared] {
                for drawn in [Drawn::None, Drawn::InTile, Drawn::TwoVertices] {
                    for writes in [false, true] {
                        cases.push((Shape::of(format, size), earlier, (drawn, writes)));
                    }
                }
            }
        }
    }
    let actions = (DONT_CARE, STORE);
    let differ = against_metal(&cases, |(shape, earlier, drawn)| {
        single((&p, &d), (shape, earlier), actions, drawn)
    });
    eprintln!("off Metal's, dont-care-load-tiles heard: {differ:#?}");
}

/// A triangle inside tile (0, 0).
const TILE_0_0: [(f32, f32); 3] = [(4.0, 4.0), (24.0, 4.0), (4.0, 24.0)];

/// The bytes of a 64 by 64 colour (`RGBA8Unorm`) attachment, and of a depth (`Depth32Float`)
/// one, each uploaded, then loaded by `DontCare` and stored by a pass that draws [`IN_TILE_1_1`]
/// under a depth-stencil state that writes depth where `first` holds, then, where `second`
/// holds, a triangle in tile (0, 0) under a nil state.
fn colour_and_depth<P: Platform>(
    (p, d): (&P, &P::Device),
    (first, second): (bool, bool),
) -> [Vec<u8>; 4] {
    let size = (64, 64);
    let (colour, depth) = (Shape::of(RGBA8, size), Shape::of(DEPTH32F, size));
    let (c, z) = (colour.texture(p, d), depth.texture(p, d));
    let (cb, zb) = (earlier_bytes(colour), earlier_bytes(depth));
    let s = state::<P>(d, &[RGBA8], (Some(DEPTH32F), None), 1);
    let ds = writing::<P>(d, Aspect::Depth);
    let a = buffer::<P>(d, 48, &positions(&IN_TILE_1_1, size));
    let b = buffer::<P>(d, 48, &positions(&TILE_0_0, size));
    let mut draws = vec![Draw {
        state: &s,
        depth_stencil: if first { ds.as_ref() } else { None },
        positions: &a,
        vertices: 3,
    }];
    if second {
        draws.push(Draw {
            state: &s,
            depth_stencil: None,
            positions: &b,
            vertices: 3,
        });
    }
    let pass = Pass {
        colours: vec![Colour::new(&c, (DONT_CARE, STORE))],
        depth: Some((&z, DONT_CARE, STORE)),
        stencil: None,
        draws,
    };
    let steps = [
        Step::Upload(&c, colour, &cb),
        Step::Upload(&z, depth, &zb),
        Step::Read(&c, colour),
        Step::Read(&z, depth),
        Step::Pass(pass),
        Step::Read(&c, colour),
        Step::Read(&z, depth),
    ];
    let reads: Vec<Vec<u8>> = run(p, d, &steps);
    [
        reads[0].clone(),
        reads[1].clone(),
        reads[2].clone(),
        reads[3].clone(),
    ]
}

/// In a pass that loads a colour and a depth attachment by `DontCare`, the colour's reached
/// tiles hold zero whatever the depth-stencil state, and the depth's hold the clear value only
/// where a draw of the pass writes depth, the tile of a later draw under a nil state too: a draw
/// under a nil state alone leaves the depth as it was, as Metal does on this host in every run.
/// This platform's answer is Metal's, or the runtime announced `dont-care-load-tiles`
#[test]
fn a_depth_dont_care_load_clears_the_reached_tiles_once_the_pass_writes_depth() {
    let (p, d) = (platform(), dev());
    let mut differ = Vec::new();
    for draws in [(false, false), (true, false), (true, true)] {
        let key = format!("{draws:?}");
        if !crate::metal_matches_raw!(&key, colour_and_depth((&p, &d), draws)) {
            assert!(
                heard(KIND),
                "{draws:?}: off Metal's, with no {} heard",
                KIND.code()
            );
            differ.push(draws);
        }
    }
    eprintln!("off Metal's, dont-care-load-tiles heard: {differ:?}");
}

/// The bytes of two 64 by 64 `RGBA8Unorm` attachments, uploaded, before and after a pass that
/// loads both by `DontCare`, stores them, and draws [`IN_TILE_1_1`] under a state of colour
/// write mask `mask` (the fragment function writes colour 0 alone).
fn two_colours<P: Platform>((p, d): (&P, &P::Device), mask: Option<u32>) -> [Vec<u8>; 4] {
    let size = (64, 64);
    let shape = Shape::of(RGBA8, size);
    let (a, b) = (shape.texture(p, d), shape.texture(p, d));
    let (ab, bb) = (
        random_bytes(shape.length(), 3),
        random_bytes(shape.length(), 4),
    );
    let s = masked_state::<P>(d, &[RGBA8, RGBA8], (None, None), (1, mask));
    let at = buffer::<P>(d, 48, &positions(&IN_TILE_1_1, size));
    let pass = Pass {
        colours: vec![
            Colour::new(&a, (DONT_CARE, STORE)),
            Colour::new(&b, (DONT_CARE, STORE)),
        ],
        depth: None,
        stencil: None,
        draws: vec![Draw {
            state: &s,
            depth_stencil: None,
            positions: &at,
            vertices: 3,
        }],
    };
    let steps = [
        Step::Upload(&a, shape, &ab),
        Step::Upload(&b, shape, &bb),
        Step::Read(&a, shape),
        Step::Read(&b, shape),
        Step::Pass(pass),
        Step::Read(&a, shape),
        Step::Read(&b, shape),
    ];
    let r = run(p, d, &steps);
    [r[0].clone(), r[1].clone(), r[2].clone(), r[3].clone()]
}

/// A draw's reached tile is zeroed in every colour attachment a pass loads by `DontCare`: the
/// one the fragment function writes no output for, and both where the write mask is none, as
/// Metal does on this host in every run. This platform's answer is Metal's, or the runtime
/// announced `dont-care-load-tiles`
#[test]
fn every_colour_attachment_of_a_reached_tile_is_zeroed() {
    let (p, d) = (platform(), dev());
    let mut differ = Vec::new();
    for mask in [None, Some(0)] {
        let key = format!("{mask:?}");
        if !crate::metal_matches_raw!(&key, two_colours((&p, &d), mask)) {
            assert!(
                heard(KIND),
                "{mask:?}: off Metal's, with no {} heard",
                KIND.code()
            );
            differ.push(mask);
        }
    }
    eprintln!("off Metal's, dont-care-load-tiles heard: {differ:?}");
}

/// A `DontCare` store keeps every texel the texture held before the pass, after a `DontCare`,
/// `Load` or `Clear` load, drawn or not, as Metal does on this host in every run:
/// `RGBA8Unorm`, `RGBA16Float`, `Depth32Float` and `Stencil8` (their draws writing the aspect),
/// 64 by 64 and 100 by 70, Private, uploaded before the pass; and `RGBA8Unorm` Shared. This
/// platform's answer is Metal's
#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "MoltenVK: the runtime records device-nan-encoding, a death in strict mode; Mac-only test, off everywhere; ledger: MoltenVK strict deaths"
)]
fn dont_care_stores_keep_every_texel() {
    let (p, d) = (platform(), dev());
    let mut cases = Vec::new();
    for (format, options) in [
        (RGBA8, PRIVATE),
        (RGBA8, SHARED),
        (RGBA16F, PRIVATE),
        (DEPTH32F, PRIVATE),
        (STENCIL8, PRIVATE),
    ] {
        for size in [(64, 64), (100, 70)] {
            for load in [DONT_CARE, LOAD, CLEAR] {
                for drawn in [Drawn::None, Drawn::InTile] {
                    let shape = Shape {
                        options,
                        ..Shape::of(format, size)
                    };
                    cases.push((shape, load, drawn));
                }
            }
        }
    }
    for (shape, load, drawn) in cases {
        let (before, after) = single(
            (&p, &d),
            (shape, Earlier::Uploaded),
            (load, DONT_CARE),
            (drawn, true),
        );
        assert!(
            before == after,
            "{shape:?} {load} {drawn:?}: the store changed texels"
        );
    }
}

/// The samples of a 4-sample `RGBA8Unorm` attachment, cleared and stored by a first pass, then
/// loaded by `DontCare`, drawn by `drawn` and stored by `store`, each pixel's resolved by a
/// third pass (its samples alike but at a drawn primitive's edges), before the second pass and
/// after it: the third pass loads them, draws over the whole target under a write mask of none
/// (a `Load` keeps `RGBA8Unorm` bytes in the tiles it reaches) and resolves them into a
/// single-sample texture.
fn multisampled<P: Platform>(
    (p, d): (&P, &P::Device),
    size: (u64, u64),
    (store, drawn): (u64, Drawn),
) -> (Vec<u8>, Vec<u8>) {
    let shape = Shape {
        samples: 4,
        usage: super::TARGET,
        ..Shape::of(RGBA8, size)
    };
    let single = Shape::of(RGBA8, size);
    let t = shape.texture(p, d);
    let (before, after, other) = (
        single.texture(p, d),
        single.texture(p, d),
        single.texture(p, d),
    );
    let s = state::<P>(d, &[RGBA8], (None, None), 4);
    let masked = masked_state::<P>(d, &[RGBA8], (None, None), (4, Some(0)));
    let (w, h) = size;
    #[expect(
        clippy::as_conversions,
        reason = "a target's side, at most 16384, as an f32"
    )]
    let whole = [(0.0, 0.0), (2.0 * w as f32, 0.0), (0.0, 2.0 * h as f32)];
    let all = buffer::<P>(d, 48, &positions(&whole, size));
    let at = drawn
        .pixels()
        .map(|px| buffer::<P>(d, 48, &positions(px, size)));
    let resolve = |into| Pass {
        colours: vec![Colour {
            resolve: Some(into),
            ..Colour::new(&t, (LOAD, MULTISAMPLE_RESOLVE))
        }],
        depth: None,
        stencil: None,
        draws: vec![Draw {
            state: &masked,
            depth_stencil: None,
            positions: &all,
            vertices: 3,
        }],
    };
    let probed = Pass {
        colours: vec![Colour {
            resolve: (store == MULTISAMPLE_RESOLVE || store == STORE_AND_RESOLVE).then_some(&other),
            ..Colour::new(&t, (DONT_CARE, store))
        }],
        depth: None,
        stencil: None,
        draws: at
            .iter()
            .map(|positions| Draw {
                state: &s,
                depth_stencil: None,
                positions,
                vertices: 3,
            })
            .collect(),
    };
    let steps = [
        Step::Pass(Pass::of(&t, Aspect::Colour, (CLEAR, STORE), Vec::new())),
        Step::Pass(resolve(&before)),
        Step::Pass(probed),
        Step::Pass(resolve(&after)),
        Step::Read(&before, single),
        Step::Read(&after, single),
    ];
    let mut r = run(p, d, &steps).into_iter();
    (r.next().unwrap(), r.next().unwrap())
}

/// A `DontCare` load of a 4-sample `RGBA8Unorm` attachment zeroes the 32 by 16 tiles a draw
/// reaches where it stores the samples (`Store`, `StoreAndMultisampleResolve`), and keeps every
/// sample where it does not (`DontCare`, `MultisampleResolve`) or nothing is drawn, 64 by 64 and
/// 100 by 70, as Metal does on this host in every run. This platform's answer is Metal's, or the
/// runtime announced `dont-care-load-tiles`
#[test]
fn multisample_dont_care_loads_zero_reached_tiles_where_the_samples_are_stored() {
    let (p, d) = (platform(), dev());
    let mut cases = Vec::new();
    for size in [(64, 64), (100, 70)] {
        for store in [DONT_CARE, STORE, MULTISAMPLE_RESOLVE, STORE_AND_RESOLVE] {
            for drawn in [Drawn::None, Drawn::InTile] {
                cases.push((size, store, drawn));
            }
        }
    }
    let differ = against_metal(&cases, |(size, store, drawn)| {
        multisampled((&p, &d), size, (store, drawn))
    });
    eprintln!("off Metal's, dont-care-load-tiles heard: {differ:#?}");
}
