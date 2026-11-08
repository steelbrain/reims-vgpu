//! `extended-range-bits-kept`: what becomes of the bits an extended-range format's codes leave
//! unused, and of codes past its range, through blits, CPU uploads, loads, clears and draws.
//!
//! **Metal's rule** (this host, every probe below the same in each of 5 runs; random bytes,
//! 64 by 64, Private and Shared, with and without `allowGPUOptimizedContents`, and of
//! `ShaderWrite` usage):
//! - *Canonical* codes: `BGRA10_XR` (and `_sRGB`) with the low six bits of each 16-bit field zero
//!   and the alpha code clamped to [384, 894]; `BGR10_XR` (and `_sRGB`) with bits 30 and 31 ones.
//! - A GPU blit into a `BGRA10_XR` texture Metal keeps losslessly compressed
//!   (`allowGPUOptimizedContents`, neither `ShaderWrite` nor `PixelFormatView` usage, the shorter
//!   side 16 or more; Private or Shared alike) writes canonical codes; into one it keeps
//!   uncompressed it writes the bytes. A blit between two such textures, and a pass that loads
//!   and stores without drawing, keep the texture's bytes. `BGR10_XR` keeps every bit through
//!   every blit, compressed or not. A CPU `replaceRegion` keeps every bit, and a blit out of the
//!   texture then reads ([`replaced_then_read`]'s answer).
//! - A pass that loads and stores the texture and draws rewrites every undrawn texel of each
//!   tile its draws reach to its canonical code, and keeps every byte of the tiles they do not
//!   reach. A draw writes canonical codes. A pass with no draw call that clears writes
//!   `BGRA10_XR`'s canonical codes but `BGR10_XR`'s bits 30 and 31 as zeros (an `ObjC` probe of
//!   this host: a draw writes them ones), values past the range as the nearest code, and a NaN
//!   channel as code 0, or 384 (zero) in an `_sRGB` twin ([`cleared`]; the test compares the
//!   bytes).
//!
//! **Not probed**: `RGB10A8_2P_XR10` and `RGBA16Float_XR`; the padded `R10Unorm_X6` and
//! `RG10Unorm_X12` (an `ObjC` probe of this host found random bytes through a blit into a
//! compressed one move half its texels, and a CPU `replaceRegion` then `getBytes` read zeros,
//! not recorded here); multisample textures; levels past the first; other Apple GPU generations.
//!
//! **This platform today**: the runtime models the compressed blit's rule. A pass that loads and
//! stores the texture keeps the bytes its draws do not write, the device's own load and store,
//! and each draw in it announces `extended-range-bits-kept`, relayed. A clear to a NaN dies at
//! the encoder, named. The tests require each other difference of this platform's bytes from
//! Metal's to come with that kind.
//!
//! **To answer as Metal**: a rewrite of the undrawn texels' codes in the tiles the draws'
//! primitives reach, in every attachment (each sample's code of a multisampled one, a shadowed or
//! sparse texture's image); a NaN clear's codes (0, or 384 in an `_sRGB` twin) taken into the
//! runtime's clear.

use crate::core::platform::Platform;
use crate::core::platform::types::ClearColor;
use metal2vulkan_runtime as rt;

use super::{
    Aspect, CLEAR, Colour, Draw, IN_TILE_1_1, LOAD, Pass, READ, SHARED, STORE, Shape, Step, TARGET,
    WRITE, buffer, heard, positions, random_bytes, replace, run, state,
};

use crate::gpu::device::tests::dev;
use crate::platform::tests::platform;

/// `BGRA10_XR`, `BGRA10_XR_sRGB`, `BGR10_XR` and `BGR10_XR_sRGB`.
const FORMATS: [u64; 4] = [552, 553, 554, 555];

/// The shapes probed: Private, Private without `allowGPUOptimizedContents`, Shared, and Private of
/// `ShaderWrite` usage, each 64 by 64.
fn shapes(format: u64) -> [Shape; 4] {
    let base = Shape::of(format, (64, 64));
    [
        base,
        Shape {
            optimized: Some(false),
            ..base
        },
        Shape {
            options: SHARED,
            ..base
        },
        Shape {
            usage: READ | WRITE | TARGET,
            ..base
        },
    ]
}

/// The bytes of a `shape` texture after: a blit of `raw` into it; a blit of it into another of
/// its shape; a pass that loads and stores it with no draw; one that draws [`IN_TILE_1_1`].
fn paths<P: Platform>((p, d): (&P, &P::Device), shape: Shape, raw: &[u8]) -> [Vec<u8>; 4] {
    let (t, copy) = (shape.texture(p, d), shape.texture(p, d));
    let s = state::<P>(d, &[shape.format], (None, None), 1);
    let at = buffer::<P>(d, 48, &positions(&IN_TILE_1_1, shape.size));
    let draw = Draw {
        state: &s,
        depth_stencil: None,
        positions: &at,
        vertices: 3,
    };
    let steps = [
        Step::Upload(&t, shape, raw),
        Step::Read(&t, shape),
        Step::Copy(&t, &copy, shape),
        Step::Read(&copy, shape),
        Step::Pass(Pass::of(&t, Aspect::Colour, (LOAD, STORE), Vec::new())),
        Step::Read(&t, shape),
        Step::Pass(Pass::of(&t, Aspect::Colour, (LOAD, STORE), vec![draw])),
        Step::Read(&t, shape),
    ];
    let r = run(p, d, &steps);
    [r[0].clone(), r[1].clone(), r[2].clone(), r[3].clone()]
}

/// Requires `ours` to be Metal's bytes, recorded, or the runtime to have announced
/// `extended-range-bits-kept`; answers whether they differ.
fn against(case: &str, ours: &[u8]) -> bool {
    let differ = !crate::metal_matches_bytes!(case, ours);
    if differ {
        assert!(
            heard(rt::Degradation::ExtendedRangeBitsKept),
            "{case}: off Metal's, with extended-range-bits-kept unheard"
        );
    }
    differ
}

/// Each extended-range format's random bytes through a blit into each of [`shapes`], a blit
/// between two textures, a pass with no draw and a pass that draws: Metal's rule on this host in
/// every run. This platform's bytes are Metal's, or the runtime announced
/// `extended-range-bits-kept`
#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "MoltenVK: the runtime records device-pack-conversion, a death in strict mode; Mac-only test, off everywhere; ledger: MoltenVK strict deaths"
)]
fn extended_range_codes_through_blits_and_loads_are_metals() {
    let (p, d) = (platform(), dev());
    let mut differ = Vec::new();
    for format in FORMATS {
        for (name, shape) in ["Private", "Private unoptimized", "Shared", "ShaderWrite"]
            .into_iter()
            .zip(shapes(format))
        {
            let raw = random_bytes(shape.length(), 41);
            let o = paths((&p, &d), shape, &raw);
            for (i, path) in ["blit", "copy", "no draw", "drawn"].iter().enumerate() {
                if against(&format!("{shape:?} {path}"), &o[i]) {
                    differ.push(format!("{format} {name} {path}"));
                }
            }
        }
    }
    eprintln!("off Metal's, a kind heard: {differ:#?}");
}

/// The bytes a blit reads out of a Shared `format` texture whose texels a CPU `replaceRegion`
/// wrote from `raw`.
fn replaced_then_read<P: Platform>((p, d): (&P, &P::Device), shape: Shape, raw: &[u8]) -> Vec<u8> {
    let t = shape.texture(p, d);
    replace::<P>(&t, shape, raw);
    run(p, d, &[Step::Read(&t, shape)]).remove(0)
}

/// The bytes of a `format` texture, cleared to `colour` by a pass that stores it.
fn cleared<P: Platform>((p, d): (&P, &P::Device), format: u64, colour: ClearColor) -> Vec<u8> {
    let shape = Shape::of(format, (16, 16));
    let t = shape.texture(p, d);
    let pass = Pass {
        colours: vec![Colour {
            clear: Some(colour),
            ..Colour::new(&t, (CLEAR, STORE))
        }],
        depth: None,
        stencil: None,
        draws: Vec::new(),
    };
    run(p, d, &[Step::Pass(pass), Step::Read(&t, shape)]).remove(0)
}

/// A CPU `replaceRegion` of random bytes into a Shared texture of each extended-range format,
/// read back by a blit, and clears to values inside and past the formats' ranges: Metal's bytes
/// are one value on this host in every run (a blit out of a compressed `BGRA10_XR` texture reads
/// canonical codes; `BGR10_XR` keeps every bit). This platform's are Metal's, or the runtime
/// announced `extended-range-bits-kept`; a clear to a NaN among them, where Metal stores a code
/// (0, or 384 in an `_sRGB` twin), as the runtime does
#[test]
fn extended_range_cpu_uploads_and_clears_are_metals() {
    let (p, d) = (platform(), dev());
    let colours = [
        ClearColor {
            red: 0.25,
            green: 0.5,
            blue: 0.75,
            alpha: 1.0,
        },
        ClearColor {
            red: -0.75,
            green: 1.5,
            blue: 2.25,
            alpha: -1.0,
        },
        ClearColor {
            red: 1e6,
            green: -1e6,
            blue: f64::NAN,
            alpha: 3.0,
        },
    ];
    let mut differ = Vec::new();
    for format in FORMATS {
        {
            let shape = Shape {
                options: SHARED,
                ..Shape::of(format, (64, 64))
            };
            let raw = random_bytes(shape.length(), 43);
            let o = replaced_then_read((&p, &d), shape, &raw);
            if against(&format!("{format} replaced"), &o) {
                differ.push(format!("{format} replaced"));
            }
        }
        for colour in colours {
            let o = cleared((&p, &d), format, colour);
            if against(&format!("{format} {colour:?}"), &o) {
                differ.push(format!("{format} {colour:?}"));
            }
        }
    }
    eprintln!("off Metal's, a kind heard: {differ:#?}");
}
