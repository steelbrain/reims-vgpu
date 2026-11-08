//! The bytes a render pass leaves in the tiles its draws never rasterise into, by load action,
//! and what its store makes of the tiles they do reach.
//!
//! **Metal's rule** (this host, every probe below the same in each of 5 runs; random
//! bytes uploaded by a blit, so NaN payloads, denormals, out-of-range and non-canonical codes
//! among them, Private 64 by 64 and 100 by 70 textures):
//! - A tile no draw's primitive reaches keeps every byte under `Load` and `DontCare`, for every
//!   format probed, in a pass that draws elsewhere and in one with no draw call. `Clear` writes
//!   the clear value into every tile, drawn or not.
//! - A tile a draw reaches is stored from tile memory. Under `Load`, an undrawn texel keeps its
//!   bytes but where tile memory holds the value otherwise: a signed normalized channel's most
//!   negative code (`0x80`, `0x8000`) comes back one above (`0x81`, `0x8001`, both -1.0), and a
//!   `Depth32Float` texel loses its sign bit (a negative value or NaN comes back positive) where
//!   a draw of the pass writes depth (where none writes it, the depth is not stored and every
//!   tile keeps its bytes: an `ObjC` probe of this host, and [`super::dont_care_load`]'s rule).
//!   A denormal and a value past 1 keep their bits (that probe too). `RGBA8Unorm`,
//!   `RGBA8Unorm_sRGB`, `RGB10A2Unorm`, `RG11B10Float`, `RGB9E5Float`, `RGBA16Float`,
//!   `RGBA32Float`, `Depth16Unorm` and `Stencil8` keep every bit. The extended-range formats are
//!   [`super::extended_range`]'s.
//!
//! **Not probed**: other formats (the 16-bit and 32-bit integer ones, the packed 16-bit ones,
//! `Depth32Float_Stencil8`), multisample textures, layered passes, rate maps, tile dispatches,
//! executions of indirect commands; other Apple GPU generations.
//!
//! **This platform today**: on MoltenVK the device's own render pass stores through Metal's
//! tile memory, so a reached tile's bytes follow Metal's rule where the runtime renders the
//! texture's image in the pass and reaches the tiles Metal's draws reach; the tests hold this
//! platform's bytes to Metal's under `Load` and `Clear`. A `DontCare` load differs where
//! [`super::dont_care_load`] says, relayed as `dont-care-load-tiles`. A native Vulkan device
//! stores every texel as loaded: it keeps the two rewrites above in reached tiles, which no kind
//! names (not run here).
//!
//! **To answer as Metal** on a device whose tile store is not Metal's: a rewrite of the codes of
//! the tiles a pass's draws reach, which the extended-range formats would take too
//! ([`super::extended_range`]), would take the signed normalized formats (the most negative code
//! one up) and `Depth32Float` (the sign bit cleared, where a draw of the pass writes depth).

use crate::core::platform::Platform;

use super::{
    Aspect, CLEAR, DEPTH32F, DONT_CARE, Draw, IN_TILE_1_1, LOAD, Pass, STENCIL8, STORE, Shape,
    Step, buffer, heard, positions, random_bytes, run, state, writing,
};
use crate::gpu::device::tests::dev;
use crate::platform::tests::platform;

/// The formats probed: `RGBA8Unorm`, `RGBA8Unorm_sRGB`, `RGBA8Snorm`, `R8Snorm`,
/// `RGB10A2Unorm`, `RG11B10Float`, `RGB9E5Float`, `R16Snorm`, `RGBA16Snorm`, `RGBA16Float`,
/// `RGBA32Float`, `Depth16Unorm`, `Depth32Float` and `Stencil8`.
const FORMATS: [u64; 14] = [
    70, 71, 72, 12, 90, 92, 93, 22, 112, 115, 125, 250, DEPTH32F, STENCIL8,
];

/// The bytes of a single-sample `format` texture of `size`, uploaded with random bytes, before
/// and after a pass that loads it by `load`, stores it, and draws `drawn` (a draw writing its
/// depth or stencil aspect).
fn probe<P: Platform>(
    (p, d): (&P, &P::Device),
    (format, size): (u64, (u64, u64)),
    (load, drawn): (u64, bool),
) -> (Vec<u8>, Vec<u8>) {
    let shape = Shape::of(format, size);
    let aspect = Aspect::of(format);
    let t = shape.texture(p, d);
    let bytes = random_bytes(shape.length(), 29);
    let colours: &[u64] = if aspect == Aspect::Colour {
        &[format]
    } else {
        &[]
    };
    let formats = match aspect {
        Aspect::Colour => (None, None),
        Aspect::Depth => (Some(format), None),
        Aspect::Stencil => (None, Some(format)),
    };
    let s = state::<P>(d, colours, formats, 1);
    let ds = writing::<P>(d, aspect);
    let at = buffer::<P>(d, 48, &positions(&IN_TILE_1_1, size));
    let draws = if drawn {
        vec![Draw {
            state: &s,
            depth_stencil: ds.as_ref(),
            positions: &at,
            vertices: 3,
        }]
    } else {
        Vec::new()
    };
    let steps = [
        Step::Upload(&t, shape, &bytes),
        Step::Read(&t, shape),
        Step::Pass(Pass::of(&t, aspect, (load, STORE), draws)),
        Step::Read(&t, shape),
    ];
    let mut r = run(p, d, &steps).into_iter();
    (r.next().unwrap(), r.next().unwrap())
}

/// Every tile no draw reaches keeps its bytes under `Load` and `DontCare`, `Clear` writes every
/// tile, and a reached tile's `Load` round trip moves a signed normalized format's most negative
/// code and a written depth's sign as Metal's tile memory does, for each of [`FORMATS`] of random
/// bytes, 64 by 64 and 100 by 70, with a draw in tile (1, 1) and with none: Metal's rule on this
/// host in every run. This platform's bytes are Metal's under `Load` and `Clear`; under
/// `DontCare` they are, or the runtime announced `dont-care-load-tiles`
#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "MoltenVK: the runtime records device-nan-encoding, a death in strict mode; Mac-only test, off everywhere; ledger: MoltenVK strict deaths"
)]
fn unreached_tiles_keep_their_bytes_under_load_and_dont_care() {
    let (p, d) = (platform(), dev());
    let mut differ = Vec::new();
    for format in FORMATS {
        for size in [(64, 64), (100, 70)] {
            for load in [DONT_CARE, LOAD, CLEAR] {
                for drawn in [false, true] {
                    let case = format!("{format} {size:?} load {load} drawn {drawn}");
                    let o = probe((&p, &d), (format, size), (load, drawn));
                    if !crate::metal_matches_raw!(&case, o) {
                        let kind = metal2vulkan_runtime::Degradation::DontCareLoadTiles;
                        assert!(load == DONT_CARE && heard(kind), "{case}: off Metal's");
                        differ.push(case);
                    }
                }
            }
        }
    }
    eprintln!("off Metal's, dont-care-load-tiles heard: {differ:#?}");
}
