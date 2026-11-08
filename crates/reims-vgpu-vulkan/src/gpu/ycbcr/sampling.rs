//! The comparisons of the runtime's textures of the two-plane YCbCr family over surfaces with
//! Metal's ([`super::runtime_makes`]): what they read and sample, their IDs and usage, their
//! views, where the runtime cannot map their samplers, and what they show of writes of their
//! surfaces. The harness is [`super::measure`]'s.
use metal2vulkan::metal::PixelFormat;
use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder as _};
use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
use reims_vgpu_core::platform::compute::{ComputeCmd, ComputeEncoder as _};
use reims_vgpu_core::platform::device::GpuDevice as _;
use reims_vgpu_core::platform::os::surface::Surface as _;
use reims_vgpu_core::platform::resource::{
    Buffer as _, GpuResource as _, Library as _, Texture as _,
};
use reims_vgpu_core::platform::types::{MtlOrigin, MtlSize, NsRange};
use reims_vgpu_core::platform::values::TextureDesc;
use reims_vgpu_core::platform::{Handle as _, Platform};
use reims_vgpu_core::raise::OnFault;
use reims_vgpu_core::raise::death::DeathKind;
use reims_vgpu_core::testkit::capture::catch_death;
use reims_vgpu_metal::MetalPlatform;

use super::measure::metal::set_value;
use super::measure::{
    self, PROBE, Shape, desc_505, fill_noise, float_bits, probe, probed, probed_texels, sampled,
    sampled_floats, surface_420, surface_420_over,
};
use crate::gpu::device::VulkanDevice;
use crate::gpu::device::tests::dev;
use crate::gpu::resource::{VulkanComputePipeline, VulkanTexture};
use crate::os::{MATRIX_KEY, OsSurface};
use crate::platform::VulkanPlatform;
use crate::platform::tests::platform;

/// `MTLPixelFormatRGB8_420_2P`, the runtime's own over a surface, compared here with the
/// family's members.
const RGB8_420_2P: u64 = 540;

/// Metal's system default device and a texture of `format` of `size` over `surface` made by
/// it.
fn metals(
    format: u64,
    surface: &OsSurface,
    size: (u64, u64),
) -> Option<(
    <MetalPlatform as Platform>::Device,
    <MetalPlatform as Platform>::Texture,
)> {
    measure::metal::metals(format, surface, 0, size)
}

/// This platform's texture of `format` of `size` over `surface`.
fn ours(format: u64, surface: &OsSurface, size: (u64, u64)) -> VulkanTexture {
    ours_on(&dev(), format, surface, size)
}

/// [`ours`] on the device `d`.
fn ours_on(d: &VulkanDevice, format: u64, surface: &OsSurface, size: (u64, u64)) -> VulkanTexture {
    let desc = platform().texture_descriptor(&measure::desc(format, size));
    d.new_texture_on_surface(&desc, Some(surface), 0)
        .unwrap()
        .unwrap()
}

/// The device whose samples of the family the comparisons with Metal's read: the platform's
/// own ([`dev`]). Its linear samples are Metal's filter exactly where the runtime reads the
/// planes in place, and elsewhere Metal's own sampler's filter of the converted texels, within the
/// bound [`the_default_filter_samples_within_its_bound_of_metals`] checks.
fn exact() -> VulkanDevice {
    dev()
}

/// The two-plane OSTypes a texture of the family is made over: 8-bit, 10-bit and packed
/// 10-bit, 4:2:0, 4:2:2 and 4:4:4, video and full range.
const OSTYPES: [&[u8; 4]; 18] = [
    b"420v", b"420f", b"422v", b"422f", b"444v", b"444f", b"x420", b"xf20", b"x422", b"xf22",
    b"x444", b"xf44", b"p420", b"pf20", b"p422", b"pf22", b"p444", b"pf44",
];

/// The formats the runtime makes over a surface ([`super::runtime_makes`]), raw.
fn family() -> Vec<u64> {
    (0..2048u32)
        .filter_map(|raw| PixelFormat::try_from(raw).ok())
        .filter(|&pf| super::runtime_makes(pf))
        .map(|pf| u64::from(pf.as_raw()))
        .collect()
}

/// The format of `format`, raw, and the OSTypes of [`OSTYPES`] whose surfaces hold its codes:
/// of its bits, packed as its are, whose chroma plane holds its own, at its subsampling or a
/// finer one (`fits`), or holds less than its own (`!fits`).
fn surfaces_of(format: u64, fits: bool) -> (PixelFormat, Vec<&'static [u8; 4]>) {
    let pf = PixelFormat::try_from(u32::try_from(format).unwrap()).unwrap();
    let f = metal2vulkan::metal::format::ycbcr_format(pf).unwrap();
    let [across, down] = f.subsampling.halved().map(|h| if h { 2 } else { 1 });
    let ostypes = OSTYPES.into_iter().filter(|ostype| {
        let shape = Shape::of(ostype);
        let finer = shape.subsampling.0 <= across && shape.subsampling.1 <= down;
        shape.bits == f.bits && shape.packed == f.packed && finer == fits
    });
    (pf, ostypes.collect())
}

/// The formats the runtime makes over a surface are the 24 the comparisons below cover, each
/// over the OSTypes of its bits whose chroma plane holds its own: a format the runtime gains
/// fails here until it joins them
#[test]
fn the_family_is_the_runtimes() {
    assert_eq!(
        family(),
        [
            500,
            502,
            503,
            505,
            506,
            507,
            508,
            509,
            510,
            520,
            522,
            523,
            525,
            526,
            527,
            528,
            529,
            530,
            RGB8_420_2P,
            541,
            542,
            543,
            544,
            545
        ]
    );
    let named = |format| {
        let (_, ostypes) = surfaces_of(format, true);
        ostypes
            .iter()
            .map(|o| String::from_utf8_lossy(&o[..]).into_owned())
            .collect::<Vec<_>>()
    };
    assert_eq!(named(500), ["420v", "420f", "422v", "422f", "444v", "444f"]);
    assert_eq!(named(506), ["x422", "xf22", "x444", "xf44"]);
    assert_eq!(named(527), ["x444", "xf44"]);
    assert_eq!(named(541), ["422v", "422f", "444v", "444f"]);
    assert_eq!(named(545), ["x444", "xf44"]);
    assert_eq!(named(508), ["p420", "pf20", "p422", "pf22", "p444", "pf44"]);
    assert_eq!(named(529), ["p422", "pf22", "p444", "pf44"]);
    assert_eq!(named(510), ["p444", "pf44"]);
}

/// A texture of `format`, one the runtime makes over a surface, is Metal's bit for bit where
/// it reads, samples nearest and linearly and gathers, into halves and into floats, under the
/// runtime's exact filter of the format ([`exact`], [`sample_as_metals_do`]): over each OSType
/// it is made over ([`surfaces_of`]), with no matrix, BT.709, BT.2020 and SMPTE 240M, at an even
/// and an odd size.
fn family_samples_as_metals(format: u64) {
    let (_, ostypes) = surfaces_of(format, true);
    for ostype in ostypes {
        let shape = Shape::of(ostype);
        let matrices = [
            None,
            Some("ITU_R_709_2"),
            Some("ITU_R_2020"),
            Some("SMPTE_240M_1995"),
        ];
        for matrix in matrices {
            for (size, seed) in [((40, 24), 5), ((33, 17), 6)] {
                let s = shape.surface(size);
                if let Some(m) = matrix {
                    set_value(&s, MATRIX_KEY, m);
                }
                shape.fill_noise(&s, size, seed);
                let what = format!(
                    "{format} over {} {matrix:?} {size:?}",
                    String::from_utf8_lossy(&shape.ostype)
                );
                sample_as_metals_do(format, &s, size, &what);
            }
        }
    }
}

/// One test of [`family_samples_as_metals`] for each format of the family.
macro_rules! family_samples {
    ($($test:ident: $format:expr,)*) => {$(
        #[doc = concat!(
            "A texture of ", stringify!($format), " over each surface the runtime makes it \
             over reads and samples as Metal's does ([`family_samples_as_metals`]), and its \
             properties are the guest descriptor's"
        )]
        #[test]
        #[cfg_attr(
            target_os = "macos",
            ignore = "MoltenVK: the runtime records pixel-sampler-scaling, a death in strict mode; Mac-only test, off everywhere; ledger: MoltenVK strict deaths"
        )]
        fn $test() {
            family_samples_as_metals($format);
        }
    )*};
}

family_samples! {
    ycbcr8_420_samples_as_metals: 500,
    ycbcr8_422_samples_as_metals: 502,
    ycbcr8_444_samples_as_metals: 503,
    ycbcr10_420_samples_as_metals: 505,
    ycbcr10_422_samples_as_metals: 506,
    ycbcr10_444_samples_as_metals: 507,
    ycbcr8_420_srgb_samples_as_metals: 520,
    ycbcr8_422_srgb_samples_as_metals: 522,
    ycbcr8_444_srgb_samples_as_metals: 523,
    ycbcr10_420_srgb_samples_as_metals: 525,
    ycbcr10_422_srgb_samples_as_metals: 526,
    ycbcr10_444_srgb_samples_as_metals: 527,
    ycbcr10_420_packed_samples_as_metals: 508,
    ycbcr10_422_packed_samples_as_metals: 509,
    ycbcr10_444_packed_samples_as_metals: 510,
    ycbcr10_420_packed_srgb_samples_as_metals: 528,
    ycbcr10_422_packed_srgb_samples_as_metals: 529,
    ycbcr10_444_packed_srgb_samples_as_metals: 530,
    rgb8_420_samples_as_metals: RGB8_420_2P,
    rgb8_422_samples_as_metals: 541,
    rgb8_444_samples_as_metals: 542,
    rgb10_420_samples_as_metals: 543,
    rgb10_422_samples_as_metals: 544,
    rgb10_444_samples_as_metals: 545,
}

/// Whether `format` is an `_sRGB` twin of a converted format of the family (520 to 530).
fn twin(format: u64) -> bool {
    (520..=530).contains(&format)
}

/// Whether `format` is a converted format of 10-bit codes, packed or not (505 to 510): over a
/// `DCI_P3` or `P3_D65` surface, which reads each code alone, the runtime binds its texture as an
/// image of codes whose read into halves it does not convert as Metal does
/// (`HalfTexelConversion`).
fn converted_10_bit(format: u64) -> bool {
    use metal2vulkan::metal::format::{YCbCrReads, ycbcr_format};
    let pf = PixelFormat::try_from(u32::try_from(format).unwrap()).unwrap();
    let f = ycbcr_format(pf).unwrap();
    f.bits == 10 && f.reads == YCbCrReads::Converted
}

/// Whether the rows of a texture of `format` of `size` over plane `plane` of a surface of
/// `shape` ([`Shape::spec`]) overlap there: more than one row, of more than 16 bytes of the
/// format's luma codes, at a `bytesPerRow` whose multiple of 16 below it is shorter than the row
/// (a 10-bit format over rows of 8-bit codes). Metal makes such a texture, and the runtime
/// refuses it, a gap of its own (`OverlappingSurfaceRows`).
fn rows_overlap(format: u64, shape: &Shape, plane: u64, size: (u64, u64)) -> bool {
    let pf = PixelFormat::try_from(u32::try_from(format).unwrap()).unwrap();
    let f = metal2vulkan::metal::format::ycbcr_format(pf).unwrap();
    let own = Shape {
        bits: f.bits,
        packed: f.packed,
        container: match (f.bits, f.packed) {
            (8, _) => 1,
            (_, true) => 4,
            _ => 2,
        },
        ..*shape
    };
    let row = own.row_bytes(size.0);
    let planes = shape.spec(size).planes.unwrap();
    let bytes_per_row = planes[usize::try_from(plane).unwrap()].bytes_per_row;
    size.1 > 1 && row > 16 && bytes_per_row / 16 * 16 < row
}

/// A texture of `format` of `size` over plane `plane` of `s`, a surface of `shape`, which Metal
/// makes, reads as Metal's does, bit for bit, into halves and into floats, with the guest
/// descriptor's properties; one whose rows overlap on the surface ([`rows_overlap`]) dies where
/// it is made, naming the runtime's refusal. Where `half_relayed`, the read into halves relays
/// the note `HalfTexelConversion`, a death in strict mode, and reads as Metal's past it.
fn reads_as_metals_do(
    format: u64,
    (shape, s): (&Shape, &OsSurface),
    plane: u64,
    size: (u64, u64),
    half_relayed: bool,
    what: &str,
) {
    let d = dev();
    let desc = platform().texture_descriptor(&measure::desc(format, size));
    *crate::answer::LAST_LINE.lock() = None;
    let made = catch_death(|| d.new_texture_on_surface(&desc, Some(s), plane));
    let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
    let metal = measure::metal::metals(format, s, plane, size);
    if let Some((_, mt)) = &metal {
        assert_eq!(mt.pixel_format(), format, "Metal: {what}");
    }
    if rows_overlap(format, shape, plane, size) {
        let site = made.map(drop).unwrap_err();
        let sel = "newTextureWithDescriptor:iosurface:plane:";
        assert_eq!(
            (site.kind, site.file),
            (DeathKind::Abort, sel),
            "{what}: {line}"
        );
        assert!(line.contains("OverlappingSurfaceRows"), "{what}: {line}");
        return;
    }
    let t = made.unwrap_or_else(|site| panic!("{what}: {site:?}: {line}"));
    let t = t.unwrap().unwrap_or_else(|| panic!("nil: {what}"));
    assert_eq!(
        (t.pixel_format(), t.width(), t.height()),
        (format, size.0, size.1),
        "{what}"
    );
    type V = crate::platform::VulkanPlatform;
    let read = || probed::<V>(&d, "read_texels", &t, size, None, None);
    if half_relayed {
        crate::answer::dies_of(metal2vulkan_runtime::Degradation::HalfTexelConversion, read);
    }
    let ours = crate::answer::relaxed(read);
    let ours_floats = measure::read_floats::<V>(&d, &t, size);
    let Some((md, mt)) = metal else {
        return;
    };
    let theirs = probed::<MetalPlatform>(&md, "read_texels", &mt, size, None, None);
    assert!(ours == theirs, "read: {what}");
    let theirs = measure::read_floats::<MetalPlatform>(&md, &mt, size);
    assert!(
        float_bits(&ours_floats) == float_bits(&theirs),
        "read into floats: {what}"
    );
}

/// A texture of each format of the family over plane 0 of a surface of each two-plane OSType
/// whose codes are not its own (of other bits or packing, or a chroma plane coarser than its
/// own), which Metal makes, reads as Metal's does ([`reads_as_metals_do`]): the luma codes from
/// plane 0 and the chroma pairs from plane 1, each in the format's own layout and subsampling,
/// whatever the surface's, converted by the OSType's range. So does one of a converted or a
/// code-reading format over its own OSType whose `kIOSurfaceYCbCrMatrix` is `DCI_P3` or
/// `P3_D65`, each code read alone, a 10-bit converted format's read into halves relayed as one the
/// runtime does not convert as Metal does ([`converted_10_bit`]), though it reads as Metal's
/// here; an `_sRGB` twin over such a surface, whose reads follow no rule
/// the runtime found, dies where it is made, naming the runtime's refusal, where Metal makes it,
/// and so does a texture whose rows overlap on the surface, a 10-bit format's over rows of 8-bit
/// codes ([`rows_overlap`])
#[test]
fn textures_over_other_ostypes_read_as_metals() {
    let size = (33, 17);
    for format in family() {
        let (_, own) = surfaces_of(format, true);
        for ostype in OSTYPES.iter().filter(|o| !own.contains(o)) {
            let shape = Shape::of(ostype);
            let s = shape.surface(size);
            shape.fill_noise(&s, size, 9);
            let what = format!("{format} over {}", String::from_utf8_lossy(&ostype[..]));
            reads_as_metals_do(format, (&shape, &s), 0, size, false, &what);
        }
        for matrix in ["DCI_P3", "P3_D65"] {
            let shape = Shape::of(own[0]);
            let s = shape.surface(size);
            set_value(&s, MATRIX_KEY, matrix);
            shape.fill_noise(&s, size, 10);
            let what = format!(
                "{format} over {} {matrix}",
                String::from_utf8_lossy(&own[0][..])
            );
            if !twin(format) {
                let half_relayed = converted_10_bit(format);
                reads_as_metals_do(format, (&shape, &s), 0, size, half_relayed, &what);
                continue;
            }
            let desc = platform().texture_descriptor(&measure::desc(format, size));
            let made = catch_death(|| dev().new_texture_on_surface(&desc, Some(&s), 0));
            let site = made.map(drop).unwrap_err();
            let sel = "newTextureWithDescriptor:iosurface:plane:";
            assert_eq!((site.kind, site.file), (DeathKind::Abort, sel), "{what}");
            if let Some((_, mt)) = measure::metal::metals(format, &s, 0, size) {
                assert_eq!(mt.pixel_format(), format, "Metal: {what}");
            }
        }
    }
}

/// A texture of each format of the family over plane 1 of a surface of each two-plane OSType,
/// which Metal makes, reads as Metal's does ([`reads_as_metals_do`]): plane 1's codes as the
/// luma, at plane 1's rows and past the plane where it is shorter than the texture, with chroma
/// codes 0. One whose rows overlap on the plane ([`rows_overlap`]) dies where it is made, naming
/// the runtime's refusal, where Metal makes it
#[test]
fn textures_over_plane_1_read_as_metals() {
    let size = (33, 17);
    for format in family() {
        for ostype in OSTYPES {
            let shape = Shape::of(ostype);
            let s = shape.surface(size);
            shape.fill_noise(&s, size, 11);
            let what = format!(
                "{format} over plane 1 of {}",
                String::from_utf8_lossy(&ostype[..])
            );
            reads_as_metals_do(format, (&shape, &s), 1, size, false, &what);
        }
    }
}

/// The probes [`sample_as_metals_do`] reads a texture of `size` with, and their grid: sample,
/// gather of red and of blue, clamped to the edge, to zero, and to each border colour
/// (transparent black, opaque black, opaque white); a sample under mirrored clamp to the edge,
/// repeat and mirrored repeat; and a linear sample clamped to the edge; each with its sampler's
/// state (filter, address mode, normalized coordinates, border colour).
fn cases(size: (u64, u64)) -> ((u64, u64), Vec<Probe>) {
    let grid = (size.0 * 2 + 12, size.1 * 2 + 12);
    let (texels, normalized) = (
        |read| probe((-2.75, -2.75), 0.5, grid, false, read),
        |read| probe((-2.8, -2.8), 0.5, grid, true, read),
    );
    let mut cases = vec![((0, 0, false, 0), texels(0)), ((0, 4, false, 0), texels(0))];
    for read in [0, 1, 2] {
        for state in [(0, 0, true, 0), (0, 4, true, 0)] {
            cases.push((state, normalized(read)));
        }
        for border in [0, 1, 2] {
            cases.push(((0, 5, true, border), normalized(read)));
        }
    }
    for state in [
        (0, 1, true, 0),
        (0, 2, true, 0),
        (0, 3, true, 0),
        (1, 0, true, 0),
    ] {
        cases.push((state, normalized(0)));
    }
    (grid, cases)
}

/// A sampler's state in a probe: filter (0 nearest, 1 linear), address mode on both axes,
/// normalized coordinates or not, and border colour.
type State = (u64, u64, bool, u64);

/// A probe of [`cases`]: its sampler's state, and its bytes ([`probe`]).
type Probe = (State, Vec<u8>);

/// The comparison of a texture of `format` over `s` with Metal's: its properties; `read()`
/// into halves and into floats; and, with coordinates in texels and normalized, nearest
/// samples and gathers of red and of blue clamped to the edge, to zero and to each border
/// colour (alpha 1 at the border too), nearest samples under mirrored clamp to the edge,
/// repeat and mirrored repeat, and linear samples clamped to the edge, each into halves and
/// into floats.
fn sample_as_metals_do(format: u64, s: &OsSurface, size: (u64, u64), what: &str) {
    let d = exact();
    let t = ours_on(&d, format, s, size);
    assert_eq!(
        (
            t.pixel_format(),
            t.usage(),
            t.storage_mode(),
            t.texture_type()
        ),
        (format, 1, 0, 2),
        "{what}"
    );
    assert_eq!(
        (t.width(), t.height(), t.iosurface().map(|s| s.id())),
        (size.0, size.1, Some(s.id()))
    );
    type V = crate::platform::VulkanPlatform;
    let ours_read = probed::<V>(&d, "read_texels", &t, size, None, None);
    let ours_floats = measure::read_floats::<V>(&d, &t, size);
    let (grid, cases) = cases(size);
    let ours_sampled: Vec<(Vec<u8>, Vec<[f32; 4]>)> = cases
        .iter()
        .map(|(state, bytes)| {
            (
                sampled::<V>(&d, &t, *state, grid, bytes),
                sampled_floats::<V>(&d, &t, *state, grid, bytes),
            )
        })
        .collect();
    let Some((md, mt)) = metals(format, s, size) else {
        return;
    };
    assert_eq!(
        (
            mt.pixel_format(),
            mt.usage(),
            mt.storage_mode(),
            mt.texture_type()
        ),
        (format, 1, 0, 2),
        "Metal: {what}"
    );
    let metal_read = probed::<MetalPlatform>(&md, "read_texels", &mt, size, None, None);
    assert!(ours_read == metal_read, "read: {what}");
    let metal_floats = measure::read_floats::<MetalPlatform>(&md, &mt, size);
    assert!(
        float_bits(&ours_floats) == float_bits(&metal_floats),
        "read into floats: {what}"
    );
    for ((state, bytes), (ours, ours_floats)) in cases.iter().zip(&ours_sampled) {
        let metal = sampled::<MetalPlatform>(&md, &mt, *state, grid, bytes);
        assert!(*ours == metal, "{state:?} {}: {what}", bytes[28]);
        let metal = sampled_floats::<MetalPlatform>(&md, &mt, *state, grid, bytes);
        assert!(
            float_bits(ours_floats) == float_bits(&metal),
            "{state:?} {} into floats: {what}",
            bytes[28]
        );
    }
}

/// A texture of `format`, one the runtime makes over a surface, samples and gathers in a fragment
/// function as Metal's does, bit for bit, into halves and into floats, under the runtime's exact
/// filter of the format ([`exact`], [`samples_in_fragments_as_metals_do`]): over each OSType it
/// is made over ([`surfaces_of`]),
/// with no matrix, BT.709, BT.2020 and SMPTE 240M, at an odd size, whose probes reach past every
/// edge as an even one's do.
fn family_samples_in_fragments_as_metals(format: u64) {
    let (_, ostypes) = surfaces_of(format, true);
    for ostype in ostypes {
        let shape = Shape::of(ostype);
        let matrices = [
            None,
            Some("ITU_R_709_2"),
            Some("ITU_R_2020"),
            Some("SMPTE_240M_1995"),
        ];
        for matrix in matrices {
            let size = (33, 17);
            let s = shape.surface(size);
            if let Some(m) = matrix {
                set_value(&s, MATRIX_KEY, m);
            }
            shape.fill_noise(&s, size, 7);
            let what = format!(
                "{format} over {} {matrix:?} in a fragment function",
                String::from_utf8_lossy(&shape.ostype)
            );
            samples_in_fragments_as_metals_do(format, &s, size, &what);
        }
    }
}

/// What a fragment function reads of `t` on `p`'s device `d` at each probe of `cases` over
/// `grid`, through a sampler of the probe's state: `probe_fragment`'s halves and
/// `probe_fragment_float`'s floats, as bytes, one fragment a probe position
/// ([`measure::drawn`]).
fn fragment_reads<P: Platform>(
    p: &P,
    d: &P::Device,
    t: &P::Texture,
    grid: (u64, u64),
    cases: &[Probe],
) -> Vec<(Vec<u8>, Vec<u8>)> {
    cases
        .iter()
        .map(|&((filter, address, normalized, border), ref bytes)| {
            let s = measure::sampler::<P>(d, (filter, address, normalized), border);
            (
                measure::drawn::<P>(p, d, ("probe_fragment", 8), t, grid, &s, bytes),
                measure::drawn::<P>(p, d, ("probe_fragment_float", 16), t, grid, &s, bytes),
            )
        })
        .collect()
}

/// [`sample_as_metals_do`]'s samples and gathers ([`cases`]) of a texture of `format` over `s`
/// in a fragment function ([`fragment_reads`]), compared with Metal's bit for bit.
fn samples_in_fragments_as_metals_do(format: u64, s: &OsSurface, size: (u64, u64), what: &str) {
    let (p, d) = (platform(), exact());
    let t = ours_on(&d, format, s, size);
    let (grid, cases) = cases(size);
    let ours = fragment_reads::<VulkanPlatform>(&p, &d, &t, grid, &cases);
    let Some((md, mt)) = metals(format, s, size) else {
        return;
    };
    let mp = MetalPlatform::new();
    let theirs = fragment_reads::<MetalPlatform>(&mp, &md, &mt, grid, &cases);
    for (((state, bytes), ours), theirs) in cases.iter().zip(&ours).zip(&theirs) {
        assert!(ours.0 == theirs.0, "{state:?} {}: {what}", bytes[28]);
        assert!(
            ours.1 == theirs.1,
            "{state:?} {} into floats: {what}",
            bytes[28]
        );
    }
}

/// One test of [`family_samples_in_fragments_as_metals`] for each format of the family.
macro_rules! family_fragment_samples {
    ($($test:ident: $format:expr,)*) => {$(
        #[doc = concat!(
            "A texture of ", stringify!($format), " over each surface the runtime makes it \
             over samples and gathers in a fragment function as Metal's does \
             ([`family_samples_in_fragments_as_metals`])"
        )]
        #[test]
        #[cfg_attr(
            target_os = "macos",
            ignore = "MoltenVK: the runtime records pixel-sampler-scaling, a death in strict mode; Mac-only test, off everywhere; ledger: MoltenVK strict deaths"
        )]
        fn $test() {
            family_samples_in_fragments_as_metals($format);
        }
    )*};
}

family_fragment_samples! {
    ycbcr8_420_samples_in_fragments_as_metals: 500,
    ycbcr8_422_samples_in_fragments_as_metals: 502,
    ycbcr8_444_samples_in_fragments_as_metals: 503,
    ycbcr10_420_samples_in_fragments_as_metals: 505,
    ycbcr10_422_samples_in_fragments_as_metals: 506,
    ycbcr10_444_samples_in_fragments_as_metals: 507,
    ycbcr8_420_srgb_samples_in_fragments_as_metals: 520,
    ycbcr8_422_srgb_samples_in_fragments_as_metals: 522,
    ycbcr8_444_srgb_samples_in_fragments_as_metals: 523,
    ycbcr10_420_srgb_samples_in_fragments_as_metals: 525,
    ycbcr10_422_srgb_samples_in_fragments_as_metals: 526,
    ycbcr10_444_srgb_samples_in_fragments_as_metals: 527,
    ycbcr10_420_packed_samples_in_fragments_as_metals: 508,
    ycbcr10_422_packed_samples_in_fragments_as_metals: 509,
    ycbcr10_444_packed_samples_in_fragments_as_metals: 510,
    ycbcr10_420_packed_srgb_samples_in_fragments_as_metals: 528,
    ycbcr10_422_packed_srgb_samples_in_fragments_as_metals: 529,
    ycbcr10_444_packed_srgb_samples_in_fragments_as_metals: 530,
    rgb8_420_samples_in_fragments_as_metals: RGB8_420_2P,
    rgb8_422_samples_in_fragments_as_metals: 541,
    rgb8_444_samples_in_fragments_as_metals: 542,
    rgb10_420_samples_in_fragments_as_metals: 543,
    rgb10_422_samples_in_fragments_as_metals: 544,
    rgb10_444_samples_in_fragments_as_metals: 545,
}

/// A view of each converted format of the family in its `_sRGB` twin (500 in 520, 502 in 522,
/// 503 in 523, 505 in 525, 506 in 526, 507 in 527, and the packed 508 in 528, 509 in 529, 510
/// in 530) answers `ShaderRead` alone, as Metal's view
/// of the texture's `ShaderRead` does, and reads into floats as Metal's view does, bit for bit,
/// over the two OSTypes of the format's own bits and chroma, at no matrix, BT.709, BT.2020 and
/// SMPTE 240M; the reverse view, of a twin in its converted format, which the runtime refuses
/// and nothing measured, dies where it is made. The twins' own textures are compared with
/// Metal's with the family ([`family_samples_as_metals`])
#[test]
fn views_in_srgb_twins_read_as_metals() {
    type V = crate::platform::VulkanPlatform;
    let family = family();
    let twins: Vec<(u64, u64)> = (family.iter())
        .flat_map(|&base| family.iter().map(move |&twin| (base, twin)))
        .filter(|&(base, twin)| base < twin && super::twin_view(base, twin))
        .collect();
    assert_eq!(
        twins,
        [
            (500, 520),
            (502, 522),
            (503, 523),
            (505, 525),
            (506, 526),
            (507, 527),
            (508, 528),
            (509, 529),
            (510, 530)
        ]
    );
    let size = (33, 17);
    for (base, format) in twins {
        let (pf, ostypes) = surfaces_of(base, true);
        let own = metal2vulkan::metal::format::ycbcr_format(pf)
            .unwrap()
            .subsampling;
        let [across, down] = own.halved().map(|h| if h { 2 } else { 1 });
        let native = (ostypes.into_iter()).filter(|o| Shape::of(o).subsampling == (across, down));
        for ostype in native {
            let shape = Shape::of(ostype);
            let matrices = [
                None,
                Some("ITU_R_709_2"),
                Some("ITU_R_2020"),
                Some("SMPTE_240M_1995"),
            ];
            for matrix in matrices {
                let s = shape.surface(size);
                if let Some(m) = matrix {
                    set_value(&s, MATRIX_KEY, m);
                }
                shape.fill_noise(&s, size, 8);
                let what = format!(
                    "{base} in {format} over {} {matrix:?}",
                    String::from_utf8_lossy(ostype)
                );
                let d = dev();
                let view = ours(base, &s, size)
                    .new_texture_view(format)
                    .unwrap()
                    .unwrap();
                assert_eq!(view.pixel_format(), format, "{what}");
                let viewed = measure::read_floats::<V>(&d, &view, size);
                let twin = ours(format, &s, size);
                let site = catch_death(|| twin.new_texture_view(base).map(drop)).unwrap_err();
                let sel = "newTextureViewWithPixelFormat:";
                assert_eq!((site.kind, site.file), (DeathKind::Abort, sel), "{what}");
                let Some((md, mt)) = metals(base, &s, size) else {
                    continue;
                };
                let theirs = mt.new_texture_view(format).unwrap().unwrap();
                assert_eq!(
                    (view.usage(), theirs.usage()),
                    (1, 1),
                    "the view's usage, ours and Metal's: {what}"
                );
                let metal = measure::read_floats::<MetalPlatform>(&md, &theirs, size);
                assert!(
                    float_bits(&viewed) == float_bits(&metal),
                    "the view: {what}"
                );
            }
        }
    }
}

/// A view of a texture of each format of the family in its own format and type is made, as
/// Metal makes it, with the default swizzle and with (B, G, R, A), and reads into floats as
/// Metal's view does, bit for bit, answering Metal's type and usage. So is one as a 2D array,
/// which Metal reads otherwise than the texture (the luma codes alone): its `read()` of every
/// texel, and its nearest and linear samples and its gathers of red and of blue under the
/// runtime's exact filter of the format ([`exact`]), at positions a third of a texel apart from
/// 0.9 of a texel before the texture to as far past it, are Metal's bit for bit. Of an `_sRGB`
/// twin, whose such view Metal reads no defined value of (measured here: two reads of one view of
/// one texture differ), they are those of Metal's view of its converted format's texture over the
/// same surface, which the runtime reads. The 2D array view is the runtime's luma view, whose
/// bindings present the format that reads its codes, so its samples announce nothing that dies,
/// at 8 bits and at 10, in strict mode. Over the format's first OSType of its own bits and
/// chroma
#[test]
fn own_format_views_are_metals() {
    use metal2vulkan::metal::format::{YCbCrReads, ycbcr_format};
    use reims_vgpu_core::platform::types::Swizzle;
    const BGRA: Swizzle = Swizzle {
        red: 4,
        green: 3,
        blue: 2,
        alpha: 5,
    };
    /// The view of `t` in `format` of `texture_type` and `swizzle`.
    fn view<P: Platform>(
        t: &P::Texture,
        format: u64,
        (texture_type, swizzle): (u64, Option<Swizzle>),
    ) -> Option<P::Texture> {
        let (levels, slices) = (NsRange::new(0, 1), NsRange::new(0, 1));
        match swizzle {
            None => t.new_texture_subview(format, texture_type, levels, slices),
            Some(s) => t.new_swizzled_texture_subview(format, texture_type, levels, slices, s),
        }
        .unwrap()
    }
    /// The type and usage of `view`, and what it reads into floats, as their bits.
    fn viewed<P: Platform>(
        d: &P::Device,
        view: &P::Texture,
        size: (u64, u64),
    ) -> (u64, u64, Vec<[u32; 4]>) {
        let texels = measure::read_floats::<P>(d, view, size);
        (view.texture_type(), view.usage(), float_bits(&texels))
    }
    /// What `view`, a 2D array view of a texture of `size`, reads of its slice into floats, as
    /// their bits: `read()` of every texel (`sample` none), or the sampler's (filter, read) at
    /// every position of the probe's grid.
    fn arrayed<P: Platform>(
        d: &P::Device,
        view: &P::Texture,
        size: (u64, u64),
        sample: Option<(u64, u32)>,
    ) -> Vec<[u32; 4]> {
        let bytes = match sample {
            None => {
                let kernel = ("read_slice_texels_float", 16);
                probed_texels::<P>(d, kernel, view, size, None, None)
            }
            Some((filter, read)) => {
                let grid = ((size.0 + 2) * 3, (size.1 + 2) * 3);
                let probe = probe((-0.9, -0.9), 1.0 / 3.0, grid, true, read);
                let s = measure::sampler::<P>(d, (filter, 0, true), 0);
                let kernel = ("sample_slice_texels_float", 16);
                probed_texels::<P>(d, kernel, view, grid, Some(&s), Some(&probe))
            }
        };
        float_bits(&measure::float_texels(&bytes))
    }
    /// The bits, chroma, packing and reads of `format`, raw.
    fn of(format: u64) -> metal2vulkan::metal::format::YCbCrFormat {
        let pf = PixelFormat::try_from(u32::try_from(format).unwrap()).unwrap();
        ycbcr_format(pf).unwrap()
    }
    /// The format whose 2D array view Metal reads as the runtime reads `format`'s: an `_sRGB`
    /// twin's converted format, or `format`.
    fn read_as(format: u64) -> u64 {
        let f = of(format);
        if f.reads != YCbCrReads::Srgb {
            return format;
        }
        let converted = |c: &u64| {
            let c = of(*c);
            (c.bits, c.subsampling, c.packed) == (f.bits, f.subsampling, f.packed)
                && c.reads == YCbCrReads::Converted
        };
        family().into_iter().find(converted).unwrap()
    }
    let size = (33, 17);
    let (d, e) = (dev(), exact());
    let mut differ = Vec::new();
    for format in family() {
        let (_, ostypes) = surfaces_of(format, true);
        let shape = Shape::of(ostypes[0]);
        let s = shape.surface(size);
        shape.fill_noise(&s, size, 12);
        let t = ours(format, &s, size);
        let Some((md, mt)) = metals(format, &s, size) else {
            return;
        };
        for swizzle in [None, Some(BGRA)] {
            let ours = viewed::<VulkanPlatform>(
                &d,
                &view::<VulkanPlatform>(&t, format, (2, swizzle)).unwrap(),
                size,
            );
            let metal = view::<MetalPlatform>(&mt, format, (2, swizzle)).unwrap();
            let theirs = viewed::<MetalPlatform>(&md, &metal, size);
            assert!(
                ours == theirs,
                "{format} {swizzle:?}: {:?} {:?}",
                (ours.0, ours.1),
                (theirs.0, theirs.1)
            );
            let array = view::<VulkanPlatform>(&t, format, (3, swizzle)).unwrap();
            let metal = view::<MetalPlatform>(&mt, format, (3, swizzle)).unwrap();
            assert_eq!(
                (array.texture_type(), array.usage()),
                (metal.texture_type(), metal.usage()),
                "the 2D array view's type and usage: {format} {swizzle:?}"
            );
            let filtered = ours_on(&e, format, &s, size);
            let filtered = view::<VulkanPlatform>(&filtered, format, (3, swizzle)).unwrap();
            let base = read_as(format);
            let (rd, read) = metals(base, &s, size).unwrap();
            let read = view::<MetalPlatform>(&read, base, (3, swizzle)).unwrap();
            for sample in [None, Some((0, 0)), Some((1, 0)), Some((0, 1)), Some((0, 2))] {
                let (on, subject) = match sample {
                    None => (&d, &array),
                    Some(_) => (&e, &filtered),
                };
                let ours = arrayed::<VulkanPlatform>(on, subject, size, sample);
                let theirs = arrayed::<MetalPlatform>(&rd, &read, size, sample);
                let what = format!("{format} {swizzle:?} {sample:?}");
                if ours != theirs {
                    let n = ours.iter().zip(&theirs).filter(|(a, b)| a != b).count();
                    let at = ours.iter().zip(&theirs).position(|(a, b)| a != b);
                    let first = at.map(|i| (ours[i], theirs[i]));
                    differ.push(format!("{what}: {n} differ, first at {at:?}: {first:?}"));
                }
            }
        }
    }
    assert!(differ.is_empty(), "{}", differ.join("\n"));
}

/// A texture of `RGB10_420_2P` (543), which reads its codes, samples as Metal's bit for bit,
/// into floats, linear filtering included under the runtime's exact filter of the format
/// ([`exact`]), inside the texture and outside it: nearest and
/// linear `sample()` and `gather()` of red and of blue, over 'x420' and 'xf22', at every
/// address mode and border colour ([`samples_under_every_mode_as_metals`]); and so does a
/// texture of `RGB8_420_2P` (540), which reads its codes as 543 does, over '420v' and '420f'
#[test]
fn code_reading_members_sample_under_every_mode_as_metals() {
    samples_under_every_mode_as_metals(543, [b"x420", b"xf22"]);
    samples_under_every_mode_as_metals(RGB8_420_2P, [b"420v", b"420f"]);
}

/// A texture of `YCBCR10_420_2P` (505), read converted, samples as Metal's bit for bit, into
/// floats, under the runtime's exact filter of the format ([`exact`]), at the strictness of
/// 543's and 540's
/// ([`code_reading_members_sample_under_every_mode_as_metals`]): nearest and linear `sample()`
/// and `gather()` of red and of blue, inside the texture and outside it, over 'x420' and 'xf22',
/// at every address mode and border colour. Metal's 505
/// samples repeat, mirrored repeat and mirror clamp to the edge as clamp to the edge, and so
/// does the runtime's, whose address-mode map takes those modes there
#[test]
fn ycbcr10_420_samples_under_every_mode_as_metals() {
    samples_under_every_mode_as_metals(505, [b"x420", b"xf22"]);
}

/// The comparison of [`code_reading_members_sample_under_every_mode_as_metals`] of a texture of
/// `format`, 40 by 24, over each of `ostypes`, filled with noise: `sample()` nearest and
/// linear, and `gather()` of red and of blue, at positions a third of a texel apart from 0.9 of
/// a texel before the texture to as far past it, normalized, under every address mode, and
/// clamped to the border under each border colour, into floats, bit for bit, on the device of
/// the exact filter ([`exact`]).
fn samples_under_every_mode_as_metals(format: u64, ostypes: [&[u8; 4]; 2]) {
    let size = (40, 24);
    for ostype in ostypes {
        let shape = Shape::of(ostype);
        let s = shape.surface(size);
        shape.fill_noise(&s, size, 7);
        let d = exact();
        let t = ours_on(&d, format, &s, size);
        let Some((md, mt)) = metals(format, &s, size) else {
            return;
        };
        let grid = ((size.0 + 2) * 3, (size.1 + 2) * 3);
        type V = crate::platform::VulkanPlatform;
        for (read, filter) in [(0, 0), (0, 1), (1, 0), (2, 0)] {
            let bytes = probe((-0.9, -0.9), 1.0 / 3.0, grid, true, read);
            for (address, border) in [
                (0, 0),
                (1, 0),
                (2, 0),
                (3, 0),
                (4, 0),
                (5, 0),
                (5, 1),
                (5, 2),
            ] {
                let sample = |t, d| {
                    let s = measure::sampler::<V>(d, (filter, address, true), border);
                    let read = probed_texels::<V>(
                        d,
                        ("sample_texels_float", 16),
                        t,
                        grid,
                        Some(&s),
                        Some(&bytes),
                    );
                    measure::float_texels(&read)
                };
                let ours = sample(&t, &d);
                let s = measure::sampler::<MetalPlatform>(&md, (filter, address, true), border);
                let metal = measure::float_texels(&probed_texels::<MetalPlatform>(
                    &md,
                    ("sample_texels_float", 16),
                    &mt,
                    grid,
                    Some(&s),
                    Some(&bytes),
                ));
                let (ours, metal) = (float_bits(&ours), float_bits(&metal));
                let differ = ours.iter().zip(&metal).filter(|(a, b)| a != b).count();
                assert_eq!(
                    differ,
                    0,
                    "{format} over {}: read {read} filter {filter} address {address} \
                     border {border}",
                    String::from_utf8_lossy(ostype)
                );
            }
        }
    }
}

/// A linear sample of a texture of `YCBCR8_420_2P` (500) and of its `_sRGB` twin (520) over a
/// '420v' surface, through a function that also gathers it, at positions a third of a texel apart
/// from 0.9 of a texel before the texture to as far past it, on this host (MoltenVK, the
/// platform's own device, [`dev`]), which reads host memory with no bus between:
/// - 500's planes are read in place (the runtime's `YCbCrInPlace`, which gathers in place too,
///   so a function that also gathers keeps it): its samples are Metal's filter exactly, its floats
///   and halves Metal's bit for bit, and nothing that dies is announced;
/// - 520, which is not read in place, takes the runtime's default filter: Metal's own sampler
///   filters the texture's converted texels, which hold Metal's reads exactly, at Metal's weights
///   but without the format's truncated sum on its grid, within one step of the grid of 1/8192 of
///   Metal's into floats, and within 1/2048 into halves, one `half` unit in the last place in
///   [0.5, 1): the bound metal2vulkan states for MoltenVK (its `docs/FAST-PATHS.md`, the two-plane
///   YCbCr row, at the pinned commit; at most 0.1245 codes of 1/255, measured on the M3 Max).
///   Each dispatch that samples it announces `ycbcr-filter-not-exact` ([`crate::answer::DIES`]):
///   in strict mode it dies there, at its selector, and by default it is relayed and counted, as
///   is `half-texel-conversion` at a `half` sample through a runtime sampler, whose nearest texel
///   on the grid of 1/8192 rounds to nearest even where Metal's rounds ties away.
///
/// Every observation is gathered first, so a failure names them all
#[test]
fn the_default_filter_samples_within_its_bound_of_metals() {
    use crate::answer::{LAST_LINE, relaxed, relayed_counts};
    use metal2vulkan_runtime::Degradation as D;
    type V = crate::platform::VulkanPlatform;
    /// What one format's samples met and how they lie against Metal's.
    #[derive(Debug)]
    struct Seen {
        #[expect(dead_code, reason = "read through Debug alone, in a failure's message")]
        format: u64,
        /// The selector a strict float sample died at, and the kind its line names.
        strict: Option<(&'static str, Option<D>)>,
        /// The kinds of DIES the relaxed samples relayed.
        relayed: Vec<D>,
        /// How many floats differ from Metal's, and the largest difference in the grid's steps.
        floats: (usize, f32),
        /// How many halves differ from Metal's, and the largest difference in codes of 1/255.
        halves: (usize, f32),
    }
    // The value of a `half`'s bits.
    let value = |h: u16| {
        let (exponent, fraction) = (i32::from((h >> 10) & 0x1f), f32::from(h & 0x3ff));
        let magnitude = match exponent {
            0 => fraction * 2f32.powi(-24),
            31 if fraction == 0.0 => f32::INFINITY,
            31 => f32::NAN,
            e => (1.0 + fraction / 1024.0) * 2f32.powi(e - 15),
        };
        if h & 0x8000 == 0 {
            magnitude
        } else {
            -magnitude
        }
    };
    let d = dev();
    let size = (40, 24);
    let grid = ((size.0 + 2) * 3, (size.1 + 2) * 3);
    let bytes = probe((-0.9, -0.9), 1.0 / 3.0, grid, true, 0);
    let linear = (1, 0, true, 0);
    let mut seen = Vec::new();
    for (format, steps) in [(500, 2048.0_f32), (520, 8192.0)] {
        let s = surface_420(b"420v", size);
        fill_noise(&s, size, 9);
        let t = ours_on(&d, format, &s, size);
        *LAST_LINE.lock() = None;
        let strict = catch_death(|| sampled_floats::<V>(&d, &t, linear, grid, &bytes))
            .err()
            .map(|site| {
                let line = LAST_LINE.lock().take().unwrap_or_default();
                let kind =
                    (D::ALL.iter().copied()).find(|k| line.contains(&format!("({})", k.code())));
                (site.file, kind)
            });
        let before = relayed_counts();
        let (halves, floats) = relaxed(|| {
            (
                sampled::<V>(&d, &t, linear, grid, &bytes),
                sampled_floats::<V>(&d, &t, linear, grid, &bytes),
            )
        });
        let relayed = (relayed_counts().into_iter())
            .filter(|&(k, n)| !before.contains(&(k, n)))
            .map(|(k, _)| k)
            .collect();
        let Some((md, mt)) = metals(format, &s, size) else {
            return;
        };
        let metal_halves = sampled::<MetalPlatform>(&md, &mt, linear, grid, &bytes);
        let metal_floats = sampled_floats::<MetalPlatform>(&md, &mt, linear, grid, &bytes);
        let (ours, metal) = (floats.as_flattened(), metal_floats.as_flattened());
        assert_eq!(ours.len(), metal.len(), "{format}");
        let worst = (ours.iter().zip(metal)).fold(0.0_f32, |w, (o, m)| w.max((o - m).abs()));
        let differ = (ours.iter().zip(metal))
            .filter(|(o, m)| o.to_bits() != m.to_bits())
            .count();
        let floats = (differ, worst * steps);
        let (ours, metal) = (measure::texels(&halves), measure::texels(&metal_halves));
        let (ours, metal) = (ours.as_flattened(), metal.as_flattened());
        assert_eq!(ours.len(), metal.len(), "{format}");
        let worst = (ours.iter().zip(metal))
            .fold(0.0_f32, |w, (&o, &m)| w.max((value(o) - value(m)).abs()));
        let differ = (ours.iter().zip(metal)).filter(|(o, m)| o != m).count();
        let halves = (differ, worst * 255.0);
        seen.push(Seen {
            format,
            strict,
            relayed,
            floats,
            halves,
        });
    }
    let all = format!("{seen:#?}");
    let [in_place, twin] = seen.as_slice() else {
        panic!("{all}");
    };
    assert!(
        in_place.strict.is_none() && in_place.relayed.is_empty(),
        "500 read in place announces nothing that dies: {all}"
    );
    assert_eq!(
        (in_place.floats.0, in_place.halves.0),
        (0, 0),
        "500 read in place samples Metal's bits: {all}"
    );
    assert_eq!(
        twin.strict,
        Some((
            "dispatchThreads:threadsPerThreadgroup:",
            Some(D::YCbCrFilterNotExact)
        )),
        "{all}"
    );
    assert!(
        twin.relayed.contains(&D::YCbCrFilterNotExact)
            && twin.relayed.contains(&D::HalfTexelConversion),
        "{all}"
    );
    assert!(twin.floats.1 <= 1.0, "520's floats within one step: {all}");
    assert!(
        twin.halves.1 <= 255.0 / 2048.0,
        "520's halves within 1/2048: {all}"
    );
}

/// The texture of 505 takes the ID Metal's pool gives it, after a plain texture and before
/// another, which takes the ID Metal's takes; once it goes, the next plain texture takes the
/// ID it left, also while a command buffer that names it nowhere is in flight. A kernel's read
/// of it through an argument buffer holding its ID, which Metal makes, dies at the dispatch
/// naming `half-texel-conversion` ([`crate::answer::DIES`]): the runtime announces the kind for
/// a `half` read of an argument buffer's texture, whose format the function does not know and
/// whose texel it converts by a rounding Metal's may not be. This platform's IDs are taken
/// without that read, Metal's with it
#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "MoltenVK: the runtime records native-half-conversion, a death in strict mode; Mac-only test, off everywhere; ledger: MoltenVK strict deaths"
)]
fn the_guests_id_is_metals() {
    use reims_vgpu_core::platform::event::SharedEvent as _;
    fn ids<P: Platform<Surface = OsSurface>>(
        d: &P::Device,
        descriptor: impl Fn(&TextureDesc) -> P::TextureDescriptor,
        s: &OsSurface,
        size: (u64, u64),
        read_held: bool,
    ) -> (Vec<u64>, Vec<u8>) {
        let plain = || {
            let desc = descriptor(&TextureDesc {
                pixel_format: 80,
                width: 8,
                height: 8,
                usage: 1,
                ..TextureDesc::default()
            });
            d.new_texture(OnFault::Die, &desc).unwrap().unwrap()
        };
        let first = plain();
        let desc = descriptor(&desc_505(size));
        let t = d
            .new_texture_on_surface(&desc, Some(s), 0)
            .unwrap()
            .unwrap();
        let second = plain();
        let held = match read_held {
            true => held::<P>(d, &t, size),
            false => Vec::new(),
        };
        let mut ids = vec![
            first.gpu_resource_id(),
            t.gpu_resource_id(),
            t.resource_index(),
            second.gpu_resource_id(),
        ];
        let (cb, event) = in_flight::<P>(d, &second);
        drop(t);
        ids.push(plain().gpu_resource_id());
        event.set_signaled_value(1).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        ids.push(plain().gpu_resource_id());
        (ids, held)
    }
    let size = (24, 16);
    let s = surface_420(b"xf20", size);
    fill_noise(&s, size, 8);
    // Metal's first, before this platform's runtime makes any Metal texture of its own in
    // the process, which would take IDs of Metal's pool.
    let metal = MetalPlatform::system_default_device().map(|md| {
        let descriptor = reims_vgpu_metal::gpu::desc::texture_descriptor;
        ids::<MetalPlatform>(&md, descriptor, &s, size, true)
    });
    let p = platform();
    let (ours, _) = ids::<crate::platform::VulkanPlatform>(
        &dev(),
        |d| p.texture_descriptor(d),
        &s,
        size,
        false,
    );
    assert_eq!(ours[1], ours[2]);
    assert_eq!(
        (ours[1], ours[4], ours[5]),
        (ours[0] + 1, ours[1], ours[1]),
        "{ours:?}"
    );
    if let Some((metal, _)) = metal {
        assert_eq!(ours, metal);
    }
    let d = dev();
    let t = self::ours(505, &s, size);
    let kind = metal2vulkan_runtime::Degradation::HalfTexelConversion;
    let site = crate::answer::dies_of(kind, || {
        held::<crate::platform::VulkanPlatform>(&d, &t, size)
    });
    assert_eq!(site.file, "dispatchThreads:threadsPerThreadgroup:");
}

/// A texture of 505 answers the usage bits Metal takes and names none of but 0x8, as Metal's
/// does: 0x10005, a macOS 26.6.2 guest's usage, whole, and 0x1c00d as 0x1c005
#[test]
fn usage_bits_metal_takes_are_answered_as_metals() {
    let size = (16, 8);
    let s = surface_420(b"x420", size);
    let md = MetalPlatform::system_default_device();
    for (usage, answered) in [(0x1_0005, 0x1_0005), (0x1_c00d, 0x1_c005)] {
        let desc = TextureDesc {
            usage,
            ..desc_505(size)
        };
        let t = dev()
            .new_texture_on_surface(&platform().texture_descriptor(&desc), Some(&s), 0)
            .unwrap()
            .unwrap();
        assert_eq!(t.usage(), answered, "{usage:#x}");
        if let Some(md) = &md {
            let metal = reims_vgpu_metal::gpu::desc::texture_descriptor(&desc);
            let mt = md
                .new_texture_on_surface(&metal, Some(&s), 0)
                .unwrap()
                .unwrap();
            assert_eq!(mt.usage(), answered, "Metal: {usage:#x}");
        }
    }
}

/// A command buffer of `d`'s committed in flight, waiting for its event's value 1, with a
/// compute encoder before it that reads `t` and names no other texture, and the event.
fn in_flight<P: Platform>(d: &P::Device, t: &P::Texture) -> (P::CommandBuffer, P::SharedEvent) {
    use reims_vgpu_core::platform::event::EventRef;
    let library = d.new_library(PROBE).unwrap().unwrap().unwrap();
    let f = library
        .new_function(OnFault::Die, "read_texels")
        .unwrap()
        .unwrap();
    let state = d
        .new_compute_pipeline_state_for_function(Some(&f))
        .unwrap()
        .unwrap();
    let out = d
        .new_buffer_of_length(OnFault::Die, 8 * 8 * 8, 0)
        .unwrap()
        .unwrap();
    let event = d.new_shared_event().unwrap();
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    cb.encode_wait(Some(EventRef::Shared(&event)), 1).unwrap();
    let e = cb.compute_encoder(OnFault::Die).unwrap().unwrap();
    let buffers = [Some(&out)];
    for cmd in [
        ComputeCmd::SetComputePipelineState {
            state: Some(&state),
        },
        ComputeCmd::SetTexture {
            texture: Some(t),
            index: 0,
        },
        ComputeCmd::SetBuffers {
            buffers: &buffers,
            offsets: &[0],
            range: NsRange::new(0, 1),
        },
        ComputeCmd::DispatchThreads {
            threads_per_grid: MtlSize::new(8, 8, 1),
            threads_per_threadgroup: MtlSize::new(8, 8, 1),
        },
        ComputeCmd::EndEncoding,
    ] {
        e.encode(OnFault::Die, cmd).unwrap();
    }
    cb.commit(OnFault::Die).unwrap();
    (cb, event)
}

/// `read_held` of an argument buffer holding `t`'s ID, made resident for reading.
fn held<P: Platform>(d: &P::Device, t: &P::Texture, (width, height): (u64, u64)) -> Vec<u8> {
    use reims_vgpu_core::platform::blit::ResourceRef;
    let library = d.new_library(PROBE).unwrap().unwrap().unwrap();
    let f = library
        .new_function(OnFault::Die, "read_held")
        .unwrap()
        .unwrap();
    let state = d
        .new_compute_pipeline_state_for_function(Some(&f))
        .unwrap()
        .unwrap();
    let length = width * height * 8;
    let out = d
        .new_buffer_of_length(OnFault::Die, length, 0)
        .unwrap()
        .unwrap();
    let args = d
        .new_buffer_of_length(OnFault::Die, 64, 0)
        .unwrap()
        .unwrap();
    // SAFETY: a Shared buffer of 64 bytes, which no command uses yet.
    unsafe { args.contents().as_ptr::<u64>().write(t.gpu_resource_id()) };
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let e = cb.compute_encoder(OnFault::Die).unwrap().unwrap();
    let buffers = [Some(&out), Some(&args)];
    for cmd in [
        ComputeCmd::SetComputePipelineState {
            state: Some(&state),
        },
        ComputeCmd::SetBuffers {
            buffers: &buffers,
            offsets: &[0, 0],
            range: NsRange::new(0, 2),
        },
        ComputeCmd::UseResources {
            resources: &[Some(ResourceRef::Texture(t))],
            usage: 1,
        },
        ComputeCmd::DispatchThreads {
            threads_per_grid: MtlSize::new(width, height, 1),
            threads_per_threadgroup: MtlSize::new(8, 8, 1),
        },
        ComputeCmd::EndEncoding,
    ] {
        e.encode(OnFault::Die, cmd).unwrap();
    }
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    let n = usize::try_from(length).unwrap();
    // SAFETY: a Shared buffer's bytes, read after the command buffer completed.
    unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), n) }.to_vec()
}

/// A command that samples a texture of 543, whose address modes the runtime maps, dies at its
/// dispatch where the runtime cannot map its sampler: through a sampler slot that samples, in
/// the same stage, an `RGBA8Unorm` texture the map addresses otherwise (under repeat). The same
/// slot sampling both under `clamp_to_edge`, which the map keeps, runs. A constexpr sampler
/// under repeat runs over 505 and 543, as the runtime maps it
/// ([`constexpr_samplers_sample_as_metals`]), and one under `clamp_to_edge`, which the map
/// keeps, samples 543 as Metal's does, inside and outside 0 to 1
#[test]
fn sampling_a_mapped_texture_where_the_runtime_cannot_map_it_dies() {
    let size = (16, 8);
    let s = surface_420(b"x420", size);
    fill_noise(&s, size, 9);
    let d = dev();
    let p = platform();
    let (mapped, unmapped) = (ours(543, &s, size), ours(505, &s, size));
    let other = {
        let desc = p.texture_descriptor(&TextureDesc {
            pixel_format: 70,
            width: size.0,
            height: size.1,
            usage: 1,
            ..TextureDesc::default()
        });
        d.new_texture(OnFault::Die, &desc).unwrap().unwrap()
    };
    let library = d.new_library(PROBE).unwrap().unwrap().unwrap();
    let kernel = |name| {
        let f = library.new_function(OnFault::Die, name).unwrap().unwrap();
        d.new_compute_pipeline_state_for_function(Some(&f))
            .unwrap()
            .unwrap()
    };
    let (constexpr, two) = (kernel("sample_constexpr"), kernel("sample_two"));
    let out = d
        .new_buffer_of_length(OnFault::Die, size.0 * size.1 * 8, 0)
        .unwrap()
        .unwrap();
    let q = d.new_command_queue().unwrap();
    let dispatch = |state: &VulkanComputePipeline,
                    textures: &[Option<&VulkanTexture>],
                    address: Option<u64>| {
        let sampler = address.map(|a| measure::sampler::<VulkanPlatform>(&d, (0, a, true), 0));
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = cb.compute_encoder(OnFault::Die).unwrap().unwrap();
        let buffers = [Some(&out)];
        let samplers = [sampler.as_ref()];
        let mut cmds = vec![
            ComputeCmd::SetComputePipelineState { state: Some(state) },
            ComputeCmd::SetTextures {
                textures,
                range: NsRange::new(0, u64::try_from(textures.len()).unwrap()),
            },
            ComputeCmd::SetBuffers {
                buffers: &buffers,
                offsets: &[0],
                range: NsRange::new(0, 1),
            },
        ];
        if sampler.is_some() {
            cmds.push(ComputeCmd::SetSamplerStates {
                samplers: &samplers,
                range: NsRange::new(0, 1),
            });
        }
        cmds.push(ComputeCmd::DispatchThreads {
            threads_per_grid: MtlSize::new(size.0, size.1, 1),
            threads_per_threadgroup: MtlSize::new(8, 8, 1),
        });
        cmds.push(ComputeCmd::EndEncoding);
        for cmd in cmds {
            e.encode(OnFault::Die, cmd).unwrap();
        }
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
    };
    let dies = |f: &dyn Fn()| {
        let site = catch_death(f).unwrap_err();
        assert_eq!(
            (site.kind, site.file),
            (DeathKind::Abort, "dispatchThreads:threadsPerThreadgroup:")
        );
    };
    dispatch(&constexpr, &[Some(&unmapped)], None);
    dispatch(&two, &[Some(&mapped), Some(&other)], Some(0));
    dispatch(&constexpr, &[Some(&mapped)], None);
    dies(&|| dispatch(&two, &[Some(&mapped), Some(&other)], Some(2)));
    let grid = (size.0 + 9, size.1 + 9);
    let bytes = probe((-4.25, -4.25), 1.0, grid, false, 0);
    let clamped = probed::<VulkanPlatform>(
        &d,
        "sample_constexpr_clamped",
        &mapped,
        grid,
        None,
        Some(&bytes),
    );
    if let Some((md, mt)) = metals(543, &s, size) {
        let theirs = probed::<MetalPlatform>(
            &md,
            "sample_constexpr_clamped",
            &mt,
            grid,
            None,
            Some(&bytes),
        );
        assert!(clamped == theirs);
    }
}

/// A texture of the family samples through a constexpr sampler as Metal's does, bit for bit,
/// into floats, under the runtime's exact filter of the format ([`exact`]), where the runtime
/// maps the sampler's modes as Metal addresses the format:
/// nearest under repeat, mirrored repeat and clamp to zero, linear under repeat and mirrored
/// repeat (a constexpr sampler has no mirror clamp to the edge), and under clamp to the border
/// colour, nearest to opaque white and linear to opaque black and to transparent black, at
/// positions a third of a texel apart from four texels before the texture to four past it. Over
/// the formats whose modes the map changes (505 and 500 under repeat and mirroring, 543 and 540
/// under every mode, the border colour's among them) and a 4:4:4 one, whose map keeps repeat
/// and mirroring (503). The runtime reads 4:2:0 planes in place, and 503's otherwise: its linear
/// samples take the runtime's default filter, and each dispatch of one announces
/// `ycbcr-filter-not-exact` ([`crate::answer::DIES`]), dying in strict mode at its selector, and
/// by default reads within one step of the grid of Metal's step of the format, 1/2048 for 8-bit
/// codes read converted (the bound metal2vulkan states for Metal's own sampler over the converted
/// texels; measured on the M3 Max: half a step); its nearest samples are Metal's bit for bit
#[test]
fn constexpr_samplers_sample_as_metals() {
    let size = (16, 8);
    // Each format, its OSType, and whether the runtime reads its planes in place.
    let cases: [(u64, &[u8; 4], bool); 5] = [
        (505, b"x420", true),
        (543, b"x420", true),
        (500, b"420v", true),
        (RGB8_420_2P, b"420f", true),
        (503, b"444v", false),
    ];
    let kernels = [
        "sample_constexpr_repeat",
        "sample_constexpr_mirrored_repeat",
        "sample_constexpr_zero",
        "sample_constexpr_repeat_linear",
        "sample_constexpr_mirrored_repeat_linear",
        "sample_constexpr_border_white",
        "sample_constexpr_border_black_linear",
        "sample_constexpr_border_clear_linear",
    ];
    let grid = ((size.0 + 8) * 3, (size.1 + 8) * 3);
    let bytes = probe((-4.0, -4.0), 1.0 / 3.0, grid, true, 0);
    let d = exact();
    let mut differ = Vec::new();
    for (format, ostype, in_place) in cases {
        let shape = Shape::of(ostype);
        let s = shape.surface(size);
        shape.fill_noise(&s, size, 11);
        let t = ours_on(&d, format, &s, size);
        let Some((md, mt)) = metals(format, &s, size) else {
            return;
        };
        for kernel in kernels {
            let read = |bytes_read: Vec<u8>| -> Vec<[u32; 4]> {
                (measure::float_texels(&bytes_read).into_iter())
                    .map(|t| t.map(f32::to_bits))
                    .collect()
            };
            let read_ours = || {
                read(probed_texels::<VulkanPlatform>(
                    &d,
                    (kernel, 16),
                    &t,
                    grid,
                    None,
                    Some(&bytes),
                ))
            };
            let theirs = read(probed_texels::<MetalPlatform>(
                &md,
                (kernel, 16),
                &mt,
                grid,
                None,
                Some(&bytes),
            ));
            let what = format!(
                "{format} over {}: {kernel}",
                String::from_utf8_lossy(ostype)
            );
            if in_place || !kernel.ends_with("_linear") {
                let ours = read_ours();
                if ours != theirs {
                    differ.push(format!("{what}: not Metal's bits"));
                }
                continue;
            }
            *crate::answer::LAST_LINE.lock() = None;
            let site = catch_death(read_ours).map(drop);
            let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
            let named = "(metal2vulkan-degrade-ycbcr-filter-not-exact)";
            if site.map_err(|s| s.file) != Err("dispatchThreads:threadsPerThreadgroup:")
                || !line.contains(named)
            {
                differ.push(format!("{what}: did not die naming {named}: {line}"));
            }
            let ours = crate::answer::relaxed(read_ours);
            let (ours, theirs) = (ours.as_flattened(), theirs.as_flattened());
            // How far apart two values are: a NaN against a number is infinitely far.
            let apart = |(&o, &m): (&u32, &u32)| {
                let (o, m) = (f32::from_bits(o), f32::from_bits(m));
                match (o.is_nan(), m.is_nan()) {
                    (false, false) => (o - m).abs(),
                    (true, true) => 0.0,
                    _ => f32::INFINITY,
                }
            };
            let worst = (ours.iter().zip(theirs)).map(apart).fold(0.0_f32, f32::max);
            if ours.len() != theirs.len() || worst * 2048.0 > 1.0 {
                differ.push(format!("{what}: {} steps of 1/2048 off", worst * 2048.0));
            }
        }
    }
    assert!(differ.is_empty(), "{}", differ.join("\n"));
}

/// How a test's work writes the surface's luma before a read of its texture of 505.
#[derive(Debug, Clone, Copy)]
enum Write {
    /// A kernel writes it through a texture over plane 0, in an encoder before the read's.
    EncoderBefore,
    /// The same, in the read's encoder, a dispatch before it.
    DispatchBefore,
    /// A blit copies zeros into that texture, in an encoder before the read's.
    Blit,
    /// The CPU writes it, after a command buffer that read it and before one that reads it.
    Cpu,
    /// A kernel writes it through an untracked texture over plane 0, in an encoder that then
    /// updates a fence, before the read's encoder, which waits for it.
    Fenced,
    /// The same, in a command buffer committed before the read's.
    FencedCommandBuffer,
}

/// `read_texels` of a texture of `format` over `s` after `write` wrote half the surface's luma,
/// on `p`'s device `d` (`descriptor` makes `p`'s texture descriptors).
fn after<P: Platform<Surface = OsSurface>>(
    d: &P::Device,
    (format, descriptor): (u64, impl Fn(&TextureDesc) -> P::TextureDescriptor),
    s: &OsSurface,
    (width, height): (u64, u64),
    write: Write,
) -> Vec<u8> {
    let t = d
        .new_texture_on_surface(
            &descriptor(&measure::desc(format, (width, height))),
            Some(s),
            0,
        )
        .unwrap()
        .unwrap();
    let fenced = matches!(write, Write::Fenced | Write::FencedCommandBuffer);
    let luma = TextureDesc {
        texture_type: 2,
        pixel_format: 20,
        width,
        height,
        usage: 3,
        // `MTLResourceHazardTrackingModeUntracked` where a fence orders the write.
        resource_options: Some(if fenced { 0x100 } else { 0 }),
        ..TextureDesc::default()
    };
    let luma = d
        .new_texture_on_surface(&descriptor(&luma), Some(s), 0)
        .unwrap()
        .unwrap();
    let library = d.new_library(PROBE).unwrap().unwrap().unwrap();
    let kernel = |name| {
        let f = library.new_function(OnFault::Die, name).unwrap().unwrap();
        d.new_compute_pipeline_state_for_function(Some(&f))
            .unwrap()
            .unwrap()
    };
    let (writes, reads) = (kernel("write_texels"), kernel("read_texels"));
    let length = width * height * 8;
    let out = d
        .new_buffer_of_length(OnFault::Die, length, 0)
        .unwrap()
        .unwrap();
    let zeros = d
        .new_buffer_of_length(OnFault::Die, length, 0)
        .unwrap()
        .unwrap();
    let half = MtlSize::new(width / 2, height, 1);
    let group = MtlSize::new(8, 8, 1);
    let q = d.new_command_queue().unwrap();
    let fence = d.new_fence().unwrap().unwrap();
    let read_cmds = |e: &P::ComputeEncoder| {
        let buffers = [Some(&out)];
        for cmd in [
            ComputeCmd::SetComputePipelineState {
                state: Some(&reads),
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
                threads_per_grid: MtlSize::new(width, height, 1),
                threads_per_threadgroup: group,
            },
        ] {
            e.encode(OnFault::Die, cmd).unwrap();
        }
    };
    let write_cmds = |e: &P::ComputeEncoder| {
        for cmd in [
            ComputeCmd::SetComputePipelineState {
                state: Some(&writes),
            },
            ComputeCmd::SetTexture {
                texture: Some(&luma),
                index: 0,
            },
            ComputeCmd::DispatchThreads {
                threads_per_grid: half,
                threads_per_threadgroup: group,
            },
        ] {
            e.encode(OnFault::Die, cmd).unwrap();
        }
    };
    let run = |encode: &dyn Fn(&P::CommandBuffer)| {
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        encode(&cb);
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
    };
    let read = |cb: &P::CommandBuffer| {
        let e = cb.compute_encoder(OnFault::Die).unwrap().unwrap();
        read_cmds(&e);
        e.encode(OnFault::Die, ComputeCmd::EndEncoding).unwrap();
    };
    let write_fenced = |cb: &P::CommandBuffer| {
        let e = cb.compute_encoder(OnFault::Die).unwrap().unwrap();
        write_cmds(&e);
        e.encode(OnFault::Die, ComputeCmd::UpdateFence { fence: &fence })
            .unwrap();
        e.encode(OnFault::Die, ComputeCmd::EndEncoding).unwrap();
    };
    let read_fenced = |cb: &P::CommandBuffer| {
        let e = cb.compute_encoder(OnFault::Die).unwrap().unwrap();
        e.encode(OnFault::Die, ComputeCmd::WaitForFence { fence: &fence })
            .unwrap();
        read_cmds(&e);
        e.encode(OnFault::Die, ComputeCmd::EndEncoding).unwrap();
    };
    match write {
        Write::EncoderBefore => run(&|cb| {
            let e = cb.compute_encoder(OnFault::Die).unwrap().unwrap();
            write_cmds(&e);
            e.encode(OnFault::Die, ComputeCmd::EndEncoding).unwrap();
            read(cb);
        }),
        Write::DispatchBefore => run(&|cb| {
            let e = cb.compute_encoder(OnFault::Die).unwrap().unwrap();
            write_cmds(&e);
            read_cmds(&e);
            e.encode(OnFault::Die, ComputeCmd::EndEncoding).unwrap();
        }),
        Write::Blit => run(&|cb| {
            let e = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
            let cmd = BlitCmd::CopyBufferToTexture {
                source_buffer: &zeros,
                source_offset: 0,
                source_bytes_per_row: width * 2,
                source_bytes_per_image: width * height * 2,
                source_size: half,
                destination_texture: &luma,
                destination_slice: 0,
                destination_level: 0,
                destination_origin: MtlOrigin::new(0, 0, 0),
            };
            e.encode(OnFault::Die, cmd).unwrap();
            e.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
            read(cb);
        }),
        Write::Cpu => {
            run(&read);
            let base = s.base_address().as_ptr::<u8>();
            let n = usize::try_from(width * 2).unwrap();
            // SAFETY: plane 0's first row, inside the surface's allocation, which no GPU
            // work uses now.
            unsafe { std::ptr::write_bytes(base, 0, n) };
            run(&read);
        }
        Write::Fenced => run(&|cb| {
            write_fenced(cb);
            read_fenced(cb);
        }),
        Write::FencedCommandBuffer => {
            let first = q.command_buffer(OnFault::Die).unwrap().unwrap();
            write_fenced(&first);
            first.commit(OnFault::Die).unwrap();
            run(&read_fenced);
            first.wait_until_completed(OnFault::Die).unwrap();
        }
    }
    let n = usize::try_from(length).unwrap();
    // SAFETY: a Shared buffer's bytes, read after the command buffers completed.
    unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), n) }.to_vec()
}

/// A texture of 505 and of 543 samples what was written to its surface before it as Metal's
/// does: by a kernel in an earlier encoder of its command buffer or an earlier dispatch of its
/// encoder, by a blit, by the CPU between command buffers, and through an untracked texture
/// before a fence its encoder waits for, in its command buffer or an earlier one
#[test]
fn samples_see_the_writes_before_them() {
    let size = (32, 16);
    for format in [505, 543] {
        for write in [
            Write::EncoderBefore,
            Write::DispatchBefore,
            Write::Blit,
            Write::Cpu,
            Write::Fenced,
            Write::FencedCommandBuffer,
        ] {
            let surface = || {
                let s = surface_420(b"x420", size);
                fill_noise(&s, size, 11);
                s
            };
            // Metal's first, before this platform's runtime makes any texture of its own.
            let metal = MetalPlatform::system_default_device().map(|md| {
                let descriptor = reims_vgpu_metal::gpu::desc::texture_descriptor;
                after::<MetalPlatform>(&md, (format, descriptor), &surface(), size, write)
            });
            let p = platform();
            let unwritten = surface();
            let before = {
                let d = dev();
                let desc = p.texture_descriptor(&measure::desc(format, size));
                let t = d
                    .new_texture_on_surface(&desc, Some(&unwritten), 0)
                    .unwrap()
                    .unwrap();
                probed::<VulkanPlatform>(&d, "read_texels", &t, size, None, None)
            };
            let ours = after::<VulkanPlatform>(
                &dev(),
                (format, |d: &TextureDesc| p.texture_descriptor(d)),
                &surface(),
                size,
                write,
            );
            assert!(ours != before, "{format} {write:?}: the write shows");
            if let Some(metal) = metal {
                assert!(ours == metal, "{format} {write:?}");
            }
        }
    }
}

/// Where [`through`] writes, before the read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Before {
    /// In a dispatch before the read's in its encoder, a memory barrier between them.
    Dispatch,
    /// In a compute encoder before the read's in its command buffer.
    Encoder,
    /// In a command buffer before the read's, which completes first.
    CommandBuffer,
}

/// `read_texels` of a texture of 505 over `s`, after work `before` it zeroed the first luma
/// row through a buffer `newBufferWithIOSurface:` makes over `s`, on `p`'s device `d`.
fn through<P: Platform<Surface = OsSurface>>(
    d: &P::Device,
    descriptor: impl Fn(&TextureDesc) -> P::TextureDescriptor,
    s: &OsSurface,
    (width, height): (u64, u64),
    before: Before,
) -> Vec<u8> {
    let t = d
        .new_texture_on_surface(&descriptor(&desc_505((width, height))), Some(s), 0)
        .unwrap()
        .unwrap();
    let over = d.new_buffer_on_surface(Some(s)).unwrap().unwrap();
    let library = d.new_library(PROBE).unwrap().unwrap().unwrap();
    let kernel = |name| {
        let f = library.new_function(OnFault::Die, name).unwrap().unwrap();
        d.new_compute_pipeline_state_for_function(Some(&f))
            .unwrap()
            .unwrap()
    };
    let (zeros, reads) = (kernel("zero_words"), kernel("read_texels"));
    let length = width * height * 8;
    let out = d
        .new_buffer_of_length(OnFault::Die, length, 0)
        .unwrap()
        .unwrap();
    let q = d.new_command_queue().unwrap();
    let (written, read) = ([Some(&over)], [Some(&out)]);
    let mut cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let mut e = cb.compute_encoder(OnFault::Die).unwrap().unwrap();
    for cmd in [
        ComputeCmd::SetComputePipelineState {
            state: Some(&zeros),
        },
        ComputeCmd::SetBuffers {
            buffers: &written,
            offsets: &[0],
            range: NsRange::new(0, 1),
        },
        ComputeCmd::DispatchThreads {
            threads_per_grid: MtlSize::new(width / 2, 1, 1),
            threads_per_threadgroup: MtlSize::new(width / 2, 1, 1),
        },
        // Buffers and textures.
        ComputeCmd::MemoryBarrierWithScope { scope: 3 },
    ] {
        e.encode(OnFault::Die, cmd).unwrap();
    }
    if before != Before::Dispatch {
        e.encode(OnFault::Die, ComputeCmd::EndEncoding).unwrap();
        if before == Before::CommandBuffer {
            cb.commit(OnFault::Die).unwrap();
            cb.wait_until_completed(OnFault::Die).unwrap();
            cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        }
        e = cb.compute_encoder(OnFault::Die).unwrap().unwrap();
    }
    for cmd in [
        ComputeCmd::SetComputePipelineState {
            state: Some(&reads),
        },
        ComputeCmd::SetTexture {
            texture: Some(&t),
            index: 0,
        },
        ComputeCmd::SetBuffers {
            buffers: &read,
            offsets: &[0],
            range: NsRange::new(0, 1),
        },
        ComputeCmd::DispatchThreads {
            threads_per_grid: MtlSize::new(width, height, 1),
            threads_per_threadgroup: MtlSize::new(8, 8, 1),
        },
        ComputeCmd::EndEncoding,
    ] {
        e.encode(OnFault::Die, cmd).unwrap();
    }
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    let n = usize::try_from(length).unwrap();
    // SAFETY: a Shared buffer's bytes, read after the command buffer completed.
    unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), n) }.to_vec()
}

/// A kernel's write of a surface's bytes through a buffer `newBufferWithIOSurface:` made over
/// it, of a surface over the guest's pages, reaches the surface through its own mapping:
/// Metal's read of a texture of 505 over it shows the write in the same command buffer, after
/// a barrier or in a later encoder, as in a later one, and so does this platform's
#[test]
fn a_write_through_a_buffer_on_the_surface_shows_in_the_same_command_buffer() {
    use reims_vgpu_core::platform::os::vm::HostVm as _;
    let size = (32, 16);
    let read = |p: &dyn Fn(&OsSurface) -> Vec<u8>| {
        let (s, pages, length) = surface_420_over(b"x420", size);
        fill_noise(&s, size, 13);
        let read = p(&s);
        drop(s);
        crate::os::OsVm::new().deallocate(pages, length);
        read
    };
    let all = [Before::Dispatch, Before::Encoder, Before::CommandBuffer];
    // Metal's first, before this platform's runtime makes any texture of its own.
    let metal = MetalPlatform::system_default_device().map(|md| {
        let descriptor = reims_vgpu_metal::gpu::desc::texture_descriptor;
        all.map(|before| read(&|s| through::<MetalPlatform>(&md, descriptor, s, size, before)))
    });
    let p = platform();
    let descriptor = |d: &TextureDesc| p.texture_descriptor(d);
    let ours =
        all.map(|before| read(&|s| through::<VulkanPlatform>(&dev(), descriptor, s, size, before)));
    let unwritten = read(&|s| {
        let d = dev();
        let t = d
            .new_texture_on_surface(&p.texture_descriptor(&desc_505(size)), Some(s), 0)
            .unwrap()
            .unwrap();
        probed::<VulkanPlatform>(&d, "read_texels", &t, size, None, None)
    });
    assert!(ours.iter().all(|o| *o != unwritten), "the write shows");
    assert!(ours.iter().all(|o| *o == ours[2]));
    if let Some(metal) = metal {
        assert!(metal == ours, "Metal's reads show the write: measure again");
    }
}

/// A texture of 505 made while a compute encoder is open is read by the encoder's next
/// dispatch as Metal's is
#[test]
fn a_texture_made_in_an_encoder_reads_as_metals() {
    fn read<P: Platform<Surface = OsSurface>>(
        d: &P::Device,
        descriptor: impl Fn(&TextureDesc) -> P::TextureDescriptor,
        s: &OsSurface,
        (width, height): (u64, u64),
    ) -> Vec<u8> {
        let library = d.new_library(PROBE).unwrap().unwrap().unwrap();
        let f = library
            .new_function(OnFault::Die, "read_texels")
            .unwrap()
            .unwrap();
        let reads = d
            .new_compute_pipeline_state_for_function(Some(&f))
            .unwrap()
            .unwrap();
        let length = width * height * 8;
        let out = d
            .new_buffer_of_length(OnFault::Die, length, 0)
            .unwrap()
            .unwrap();
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = cb.compute_encoder(OnFault::Die).unwrap().unwrap();
        let t = d
            .new_texture_on_surface(&descriptor(&desc_505((width, height))), Some(s), 0)
            .unwrap()
            .unwrap();
        let buffers = [Some(&out)];
        for cmd in [
            ComputeCmd::SetComputePipelineState {
                state: Some(&reads),
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
                threads_per_grid: MtlSize::new(width, height, 1),
                threads_per_threadgroup: MtlSize::new(8, 8, 1),
            },
            ComputeCmd::EndEncoding,
        ] {
            e.encode(OnFault::Die, cmd).unwrap();
        }
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        let n = usize::try_from(length).unwrap();
        // SAFETY: a Shared buffer's bytes, read after the command buffer completed.
        unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), n) }.to_vec()
    }
    let size = (24, 16);
    let s = surface_420(b"x420", size);
    fill_noise(&s, size, 12);
    // Metal's first, before this platform's runtime makes any texture of its own.
    let metal = MetalPlatform::system_default_device().map(|md| {
        let descriptor = reims_vgpu_metal::gpu::desc::texture_descriptor;
        read::<MetalPlatform>(&md, descriptor, &s, size)
    });
    let p = platform();
    let ours = read::<VulkanPlatform>(&dev(), |d| p.texture_descriptor(d), &s, size);
    assert!(
        ours.iter().any(|&b| b != 0),
        "the texture reads its surface"
    );
    if let Some(metal) = metal {
        assert!(ours == metal);
    }
}

/// What a test does with a texture of the family ([`written`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Op {
    /// `replaceRegion:` of all of it from bytes of 8 a texel.
    Replace,
    /// A blit of all of it to a buffer, 8 bytes a texel.
    ToBuffer,
    /// A blit of all of it from that buffer.
    FromBuffer,
    /// A blit of 8 by 4 of its texels into a texture of its format over `other`.
    Copy,
    /// A pass that clears it to (0.25, 0.5, 0.75, 1) and stores it.
    Clear,
    /// A pass that clears it as [`Op::Clear`]'s, draws `corner_vertex`'s triangle over its top
    /// left half in `gradient_fragment`'s colours, unblended, and stores it.
    Draw,
    /// A pass that loads it, draws as [`Op::Draw`] does, and stores it.
    Load,
    /// A pass that loads it, draws nothing, and stores it.
    LoadUndrawn,
    /// A pass that clears it and draws as [`Op::Draw`]'s, and stores it by `DontCare`.
    DontCareStore,
    /// A pass that clears it as [`Op::Clear`]'s, draws as [`Op::Draw`] does under a write mask of
    /// red and blue, and stores it.
    Masked,
    /// A pass that clears it as [`Op::Clear`]'s, draws as [`Op::Draw`] does blended by
    /// `SourceColor` and `OneMinusSourceColor`, added, colour and alpha, and stores it.
    Blended,
    /// A kernel's `write()` of 0 into each of its texels (`write_texels`).
    Kernel,
    /// A fragment function's `write()` of 0 into each of its texels (`write_texels_fragment`),
    /// drawn over a `BGRA8Unorm` target of its size.
    Fragment,
}

impl Op {
    /// The selector of the op's call.
    pub(crate) fn sel(self) -> &'static str {
        match self {
            Op::Replace => "replaceRegion:mipmapLevel:slice:withBytes:bytesPerRow:bytesPerImage:",
            Op::ToBuffer => {
                "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:\
                 destinationOffset:destinationBytesPerRow:destinationBytesPerImage:"
            }
            Op::FromBuffer => {
                "copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:\
                 toTexture:destinationSlice:destinationLevel:destinationOrigin:"
            }
            Op::Copy => {
                "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:\
                 destinationSlice:destinationLevel:destinationOrigin:"
            }
            Op::Clear
            | Op::Draw
            | Op::Load
            | Op::LoadUndrawn
            | Op::DontCareStore
            | Op::Masked
            | Op::Blended => "renderCommandEncoderWithDescriptor:",
            Op::Kernel => "dispatchThreads:threadsPerThreadgroup:",
            Op::Fragment => "drawPrimitives:vertexStart:vertexCount:",
        }
    }
}

/// The bytes of `surface`, all of its allocation.
fn bytes_of(surface: &OsSurface) -> Vec<u8> {
    let length = usize::try_from(surface.alloc_size()).unwrap();
    // SAFETY: the surface's allocation, read once the work that wrote it completed.
    unsafe { std::slice::from_raw_parts(surface.base_address().as_ptr::<u8>(), length) }.to_vec()
}

/// What `op` writes of a texture of `format` of `size` over plane 0 of `s` on `p`'s device `d`
/// (`other` a second surface of its shape): the bytes of `s`, of `other`, and of a `Shared`
/// buffer of 8 bytes a texel that held a pattern first, once the work completed.
pub(crate) fn written<P: Platform<Surface = OsSurface>>(
    p: &P,
    d: &P::Device,
    op: Op,
    format: u64,
    surfaces: (&OsSurface, &OsSurface),
    size: (u64, u64),
) -> [Vec<u8>; 3] {
    written_at(p, d, (op, 0), format, surfaces, size)
}

/// What `op` writes as [`written`] says, of a texture over plane `plane` of `s`.
fn written_at<P: Platform<Surface = OsSurface>>(
    p: &P,
    d: &P::Device,
    (op, plane): (Op, u64),
    format: u64,
    (s, other): (&OsSurface, &OsSurface),
    size: (u64, u64),
) -> [Vec<u8>; 3] {
    use reims_vgpu_core::platform::HostRange;
    use reims_vgpu_core::platform::descriptor::RenderPassDescriptor as _;
    use reims_vgpu_core::platform::render::{RenderCmd, RenderEncoder as _};
    use reims_vgpu_core::platform::types::{ClearColor, MtlRegion};
    use reims_vgpu_core::platform::values::ColorAttachmentSet;
    let mut desc = measure::desc(format, size);
    desc.usage = match op {
        Op::Clear
        | Op::Draw
        | Op::Load
        | Op::LoadUndrawn
        | Op::DontCareStore
        | Op::Masked
        | Op::Blended => 0x5,
        Op::Kernel | Op::Fragment => 0x3,
        _ => 0x1,
    };
    let desc = p.texture_descriptor(&desc);
    let t = d
        .new_texture_on_surface(&desc, Some(s), plane)
        .unwrap()
        .unwrap();
    let q = d.new_command_queue().unwrap();
    let (w, h) = size;
    let row = w * 8;
    let pattern: Vec<u8> = (0..row * h).map(|i| (i * 37 % 251) as u8).collect();
    let buffer = d
        .new_buffer_of_length(OnFault::Die, row * h, 0)
        .unwrap()
        .unwrap();
    // SAFETY: a Shared buffer of `row * h` bytes, which no command uses yet.
    unsafe {
        let to = buffer.contents().as_ptr::<u8>();
        std::ptr::copy_nonoverlapping(pattern.as_ptr(), to, pattern.len());
    }
    let run = |encode: &dyn Fn(&P::CommandBuffer)| {
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        encode(&cb);
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        assert_eq!(cb.error(), None);
    };
    let blit = |cmd: BlitCmd<'_, P>| {
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
        e.encode(OnFault::Die, cmd).unwrap();
        e.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        assert_eq!(cb.error(), None);
    };
    let (origin, all) = (MtlOrigin::new(0, 0, 0), MtlSize::new(w, h, 1));
    match op {
        Op::Replace => {
            let region = MtlRegion::new(origin, all);
            let bytes = HostRange::of_slice(&pattern);
            t.replace_region(region, 0, 0, bytes, row, 0).unwrap();
        }
        Op::ToBuffer => blit(BlitCmd::CopyTextureToBuffer {
            source_texture: &t,
            source_slice: 0,
            source_level: 0,
            source_origin: origin,
            source_size: all,
            destination_buffer: &buffer,
            destination_offset: 0,
            destination_bytes_per_row: row,
            destination_bytes_per_image: row * h,
        }),
        Op::FromBuffer => blit(BlitCmd::CopyBufferToTexture {
            source_buffer: &buffer,
            source_offset: 0,
            source_bytes_per_row: row,
            source_bytes_per_image: row * h,
            source_size: all,
            destination_texture: &t,
            destination_slice: 0,
            destination_level: 0,
            destination_origin: origin,
        }),
        Op::Copy => {
            let to = d
                .new_texture_on_surface(&desc, Some(other), 0)
                .unwrap()
                .unwrap();
            blit(BlitCmd::CopyTextureRegion {
                source_texture: &t,
                source_slice: 0,
                source_level: 0,
                source_origin: origin,
                source_size: MtlSize::new(8, 4, 1),
                destination_texture: &to,
                destination_slice: 0,
                destination_level: 0,
                destination_origin: MtlOrigin::new(4, 4, 0),
            });
        }
        Op::Clear
        | Op::Draw
        | Op::Load
        | Op::LoadUndrawn
        | Op::DontCareStore
        | Op::Masked
        | Op::Blended => {
            use reims_vgpu_core::platform::values::RenderPipelineDesc;
            use reims_vgpu_core::rt::Hold;

            use crate::gpu::serialized::tests::rounded_render_of;
            let pass = p.render_pass_descriptor();
            let color = ClearColor {
                red: 0.25,
                green: 0.5,
                blue: 0.75,
                alpha: 1.0,
            };
            // `MTLLoadActionLoad` or `Clear`; `MTLStoreActionDontCare` or `Store`.
            let load = if matches!(op, Op::Load | Op::LoadUndrawn) {
                1
            } else {
                2
            };
            let store = u64::from(op != Op::DontCareStore);
            let attachment = ColorAttachmentSet {
                texture: Some(Some(&t)),
                load_action: Some(load),
                store_action: Some(store),
                clear_color: Some(color),
                ..ColorAttachmentSet::default()
            };
            pass.set_color_attachment(OnFault::Die, 0, &attachment)
                .unwrap();
            let library = d.new_library(PROBE).unwrap().unwrap().unwrap();
            let function = |name| library.new_function(OnFault::Die, name).unwrap().unwrap();
            let (vertex, fragment) = (function("corner_vertex"), function("gradient_fragment"));
            // The colour attachment's keys of a described state: its write mask of red and blue
            // (`MTLColorWriteMask` 0xa), or its blend.
            let described = match op {
                Op::Masked => Some(vec![(9, 0xa)]),
                Op::Blended => Some(vec![(2, 1), (3, 2), (4, 3), (5, 0), (6, 2), (7, 3), (8, 0)]),
                _ => None,
            };
            let state = (!matches!(op, Op::Clear | Op::LoadUndrawn)).then(|| match described {
                Some(keys) => {
                    let functions = [&vertex, &fragment];
                    let mut resolve = |_: Hold<'_>, r: u64| {
                        let at = usize::try_from(r - 1).unwrap();
                        Ok(Some(functions[at].clone_ref()))
                    };
                    let mut colour = vec![(0, 0), (1, u32::try_from(format).unwrap())];
                    colour.extend(keys);
                    let blob = rounded_render_of(&["02 04 02000000"], &[&colour], 0);
                    let desc = d
                        .deserialize_render_pipeline_descriptor(Hold::NONE, &blob, &mut resolve)
                        .unwrap()
                        .unwrap();
                    d.new_render_pipeline_state_from_descriptor(Some(&desc), 0)
                        .unwrap()
                        .unwrap()
                        .unwrap()
                }
                None => d
                    .new_render_pipeline_state(&RenderPipelineDesc {
                        label: None,
                        vertex: &vertex,
                        fragment: &fragment,
                        color0_pixel_format: format,
                        blend_over: false,
                        resource_index: None,
                    })
                    .unwrap()
                    .unwrap(),
            });
            run(&|cb| {
                let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
                if let Some(state) = &state {
                    for cmd in [
                        RenderCmd::SetRenderPipelineState {
                            pipeline_state: Some(state),
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
            });
        }
        Op::Kernel => {
            let library = d.new_library(PROBE).unwrap().unwrap().unwrap();
            let f = library
                .new_function(OnFault::Die, "write_texels")
                .unwrap()
                .unwrap();
            let state = d
                .new_compute_pipeline_state_for_function(Some(&f))
                .unwrap()
                .unwrap();
            run(&|cb| {
                let e = cb.compute_encoder(OnFault::Die).unwrap().unwrap();
                for cmd in [
                    ComputeCmd::SetComputePipelineState {
                        state: Some(&state),
                    },
                    ComputeCmd::SetTexture {
                        texture: Some(&t),
                        index: 0,
                    },
                    ComputeCmd::DispatchThreads {
                        threads_per_grid: all,
                        threads_per_threadgroup: MtlSize::new(8, 8, 1),
                    },
                    ComputeCmd::EndEncoding,
                ] {
                    e.encode(OnFault::Die, cmd).unwrap();
                }
            });
        }
        Op::Fragment => {
            use reims_vgpu_core::platform::values::RenderPipelineDesc;
            let library = d.new_library(PROBE).unwrap().unwrap().unwrap();
            let function = |name| library.new_function(OnFault::Die, name).unwrap().unwrap();
            let (vertex, fragment) = (function("cover_vertex"), function("write_texels_fragment"));
            let state = d
                .new_render_pipeline_state(&RenderPipelineDesc {
                    label: None,
                    vertex: &vertex,
                    fragment: &fragment,
                    color0_pixel_format: 80,
                    blend_over: false,
                    resource_index: None,
                })
                .unwrap()
                .unwrap();
            let target = p.texture_descriptor(&TextureDesc {
                texture_type: 2,
                pixel_format: 80,
                width: w,
                height: h,
                usage: 4,
                resource_options: Some(0x20),
                ..TextureDesc::default()
            });
            let target = d.new_texture(OnFault::Die, &target).unwrap().unwrap();
            let pass = p.render_pass_descriptor();
            let attachment = ColorAttachmentSet {
                texture: Some(Some(&target)),
                ..ColorAttachmentSet::default()
            };
            pass.set_color_attachment(OnFault::Die, 0, &attachment)
                .unwrap();
            run(&|cb| {
                let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
                for cmd in [
                    RenderCmd::SetRenderPipelineState {
                        pipeline_state: Some(&state),
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
            });
        }
    }
    let n = usize::try_from(row * h).unwrap();
    // SAFETY: the Shared buffer's bytes, once the work completed.
    let held = unsafe { std::slice::from_raw_parts(buffer.contents().as_ptr::<u8>(), n) };
    [bytes_of(s), bytes_of(other), held.to_vec()]
}

/// Two surfaces of `format`'s first OSType of its own bits and chroma, of `size`, filled with
/// noise of `seed` and of the next.
fn noisy_pair(format: u64, size: (u64, u64), seed: u64) -> (OsSurface, OsSurface) {
    let (_, ostypes) = surfaces_of(format, true);
    noisy_pair_of(ostypes[0], size, seed)
}

/// Two surfaces of `ostype`, of `size`, filled with noise of `seed` and of the next.
fn noisy_pair_of(ostype: &[u8; 4], size: (u64, u64), seed: u64) -> (OsSurface, OsSurface) {
    let shape = Shape::of(ostype);
    let (s, other) = (shape.surface(size), shape.surface(size));
    shape.fill_noise(&s, size, seed);
    shape.fill_noise(&other, size, seed + 1);
    (s, other)
}

/// `replaceRegion:` of a texture of 505, 543 or 503 over a surface, and its blits to and from a
/// buffer, write the surface's bytes and the buffer's as Metal's do, byte for byte
#[test]
fn cpu_writes_and_buffer_blits_of_family_textures_are_metals() {
    let Some(md) = MetalPlatform::system_default_device() else {
        return;
    };
    let mp = MetalPlatform::new();
    let size = (32, 16);
    for format in [505, 543, 503] {
        for op in [Op::Replace, Op::ToBuffer, Op::FromBuffer] {
            let ours = noisy_pair(format, size, 3);
            let ours = written(&platform(), &dev(), op, format, (&ours.0, &ours.1), size);
            let theirs = noisy_pair(format, size, 3);
            let theirs = written(&mp, &md, op, format, (&theirs.0, &theirs.1), size);
            assert!(ours == theirs, "{format} {op:?}");
        }
    }
}

/// A blit of a texture of 505, 543 or 503 over a surface into another such texture, and a pass
/// that clears a texture of an `_sRGB` twin (525) and stores it, die where they are encoded,
/// naming the runtime's refusal (`Format`), where Metal copies and clears; the pass's death names
/// the format, its raw value and Metal's name ([`crate::gpu::ycbcr::pass_refused`]). The passes
/// into the formats read converted or as codes are [`passes_into_8_bit_formats_are_metals`]',
/// [`passes_into_10_bit_formats_are_metals`]' and [`passes_into_code_formats_are_metals`]'
#[test]
fn copies_and_attachments_of_family_textures_die() {
    let md = MetalPlatform::system_default_device();
    let mp = MetalPlatform::new();
    let size = (32, 16);
    for (format, op) in [
        (505, Op::Copy),
        (543, Op::Copy),
        (503, Op::Copy),
        (525, Op::Clear),
    ] {
        let (s, other) = noisy_pair(format, size, 5);
        *crate::answer::LAST_LINE.lock() = None;
        let site = catch_death(|| written(&platform(), &dev(), op, format, (&s, &other), size))
            .unwrap_err();
        let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
        let case = format!("{format} {op:?}: {line}");
        assert_eq!(
            (site.kind, site.file),
            (DeathKind::Abort, op.sel()),
            "{case}"
        );
        assert!(line.contains("Format"), "{case}");
        if op == Op::Clear {
            let named = format!(
                "pixel format {format} ({})",
                crate::gpu::format_info::name(op.sel(), format)
            );
            assert!(line.contains(&named), "{case}");
            assert!(
                line.contains("by a rule the runtime does not hold"),
                "{case}"
            );
        }
        if let Some(md) = &md {
            let (s, other) = noisy_pair(format, size, 5);
            written(&mp, md, op, format, (&s, &other), size);
        }
    }
}

/// A pass into a texture of an 8-bit format of the family (500, 502, 503) over plane 0 of a
/// surface of each OSType of its bits whose chroma plane holds its own, that clears it and
/// stores it, or clears it, draws over its top left half and stores it, leaves both of the
/// surface's planes as Metal's does, byte for byte, and no other byte (every texel stored, drawn
/// or not)
#[test]
fn passes_into_8_bit_formats_are_metals() {
    passes_are_metals(&[500, 502, 503]);
}

/// A pass into a texture of a 10-bit format of the family read converted (505, 506, 507), as
/// [`passes_into_8_bit_formats_are_metals`]'s: over each OSType of its bits whose chroma plane
/// holds its own ('x420', 'xf20', 'x422', 'xf22', 'x444', 'xf44'), each code in the high 10 bits
/// of its 16-bit word (measured on an M4)
#[test]
fn passes_into_10_bit_formats_are_metals() {
    passes_are_metals(&[505, 506, 507]);
}

/// A pass into a texture of a format of the family read as codes (540 to 545, red Cr, green Y
/// and blue Cb), as [`passes_into_8_bit_formats_are_metals`]'s: over each OSType of its bits
/// whose chroma plane holds its own, where this platform died at the runtime's refusal
#[test]
fn passes_into_code_formats_are_metals() {
    passes_are_metals(&[540, 541, 542, 543, 544, 545]);
}

/// The comparison of [`passes_into_8_bit_formats_are_metals`], of `formats`.
fn passes_are_metals(formats: &[u64]) {
    let md = MetalPlatform::system_default_device();
    let mp = MetalPlatform::new();
    let size = (32, 16);
    for &format in formats {
        let (_, ostypes) = surfaces_of(format, true);
        assert!(!ostypes.is_empty(), "{format}");
        for ostype in ostypes {
            for op in [Op::Clear, Op::Draw] {
                let case = format!("{format} over {} {op:?}", String::from_utf8_lossy(ostype));
                let (s, other) = noisy_pair_of(ostype, size, 7);
                let before = bytes_of(&s);
                let ours = written(&platform(), &dev(), op, format, (&s, &other), size);
                assert!(ours[0] != before, "{case}: the pass stored nothing");
                let Some(md) = &md else { continue };
                let (s, other) = noisy_pair_of(ostype, size, 7);
                let theirs = written(&mp, md, op, format, (&s, &other), size);
                let differ = (ours[0].iter().zip(&theirs[0]))
                    .filter(|(a, b)| a != b)
                    .count();
                assert_eq!(differ, 0, "{case}: bytes of the surface");
                assert!(
                    ours[1..] == theirs[1..],
                    "{case}: the other surface and the buffer"
                );
            }
        }
    }
}

/// A pass into a texture of an 8- or 10-bit format of the family read converted (500, 502, 503,
/// 505, 506, 507) or as codes (540 to 545) over plane 0 of a surface of each OSType of its bits whose chroma plane holds
/// its own leaves both of the surface's planes as Metal's does, byte for byte, and no other byte,
/// where this platform died at the runtime's refusal: one that loads it and stores it, drawing
/// nothing, which stores nothing ([`Op::LoadUndrawn`]); one that draws and stores it by
/// `DontCare`, which stores nothing ([`Op::DontCareStore`]); and one that draws under a write
/// mask of red and blue ([`Op::Masked`]). One that loads it, draws over its top left half and
/// stores it ([`Op::Load`]) dies at `endEncoding`, naming `ycbcr-load-tiles`: the runtime
/// re-encodes every tile of the rectangle bounding the draws' scissors, where Metal stores only
/// the 32 by 32 tiles a primitive reaches, and here the one tile of a 32 by 16 texture, which the
/// draw reaches; relayed, the pass leaves Metal's bytes. One that draws blended
/// ([`Op::Blended`]) dies at the draw, naming `ycbcr-blend-rounding`; relayed, each of the
/// planes' codes is within one of Metal's, every other bit Metal's
#[test]
fn loaded_masked_and_blended_passes_are_metals() {
    const DRAW: &str = "drawPrimitives:vertexStart:vertexCount:";
    let Some(md) = MetalPlatform::system_default_device() else {
        return;
    };
    let mp = MetalPlatform::new();
    let size = (32, 16);
    let mut differ = Vec::new();
    for format in [500, 502, 503, 505, 506, 507, 540, 541, 542, 543, 544, 545] {
        // An 8-bit format's codes are its bytes; a 10-bit one's are each 16-bit word's high 10
        // bits, its low 6 bits kept beside.
        let codes = |bytes: &[u8]| -> Vec<(u32, u32)> {
            match format {
                500..=503 | 540..=542 => bytes.iter().map(|&b| (u32::from(b), 0)).collect(),
                _ => (bytes.as_chunks::<2>().0.iter())
                    .map(|w| {
                        let w = u32::from(u16::from_le_bytes(*w));
                        (w >> 6, w & 0x3f)
                    })
                    .collect(),
            }
        };
        let (_, ostypes) = surfaces_of(format, true);
        assert!(!ostypes.is_empty(), "{format}");
        for ostype in ostypes {
            for op in [
                Op::LoadUndrawn,
                Op::DontCareStore,
                Op::Masked,
                Op::Load,
                Op::Blended,
            ] {
                let case = format!("{format} over {} {op:?}", String::from_utf8_lossy(ostype));
                let (s, other) = noisy_pair_of(ostype, size, 7);
                let theirs = written(&mp, &md, op, format, (&s, &other), size);
                let ours = |relaxed: bool| {
                    let (s, other) = noisy_pair_of(ostype, size, 7);
                    let run = || written(&platform(), &dev(), op, format, (&s, &other), size);
                    match relaxed {
                        true => Ok(crate::answer::relaxed(run)),
                        false => catch_death(run),
                    }
                };
                *crate::answer::LAST_LINE.lock() = None;
                let strict = ours(false);
                let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
                let dies = match op {
                    Op::Load => Some(("endEncoding", "(metal2vulkan-degrade-ycbcr-load-tiles)")),
                    Op::Blended => Some((DRAW, "(metal2vulkan-degrade-ycbcr-blend-rounding)")),
                    _ => None,
                };
                let ours = match (strict, dies) {
                    (Ok(ours), None) => ours,
                    (Err(site), Some((sel, named))) if site.file == sel && line.contains(named) => {
                        ours(true).unwrap()
                    }
                    (Ok(_), Some(_)) => {
                        differ.push(format!("{case}: ran, announcing nothing"));
                        continue;
                    }
                    (Err(site), _) => {
                        differ.push(format!("{case}: died at {site:?}: {line}"));
                        continue;
                    }
                };
                if ours[1..] != theirs[1..] {
                    differ.push(format!("{case}: the other surface or the buffer"));
                }
                let within = u32::from(op == Op::Blended);
                let pairs = codes(&ours[0]).into_iter().zip(codes(&theirs[0]));
                let off = pairs
                    .filter(|((o, o_low), (m, m_low))| o.abs_diff(*m) > within || o_low != m_low)
                    .count();
                if off != 0 {
                    differ.push(format!("{case}: {off} codes of the surface off"));
                }
            }
        }
    }
    assert!(differ.is_empty(), "{} differ: {differ:#?}", differ.len());
}

/// `replaceRegion:` of a texture of a packed format of the family (508 to 510, 528 to 530) over
/// a surface leaves the surface's bytes as they were, as Metal's does; a blit of it to or from a
/// buffer, which Metal runs at rows that follow no layout of the texture's, dies where it is
/// sent, naming the runtime's refusal (`Format`). A pass that clears it and stores it dies where
/// it is made,
/// naming the format and that no exact form exists ([`crate::gpu::ycbcr::pass_refused`]): Metal
/// fails that pass's command buffer, which no comparison here runs, as a second such failure has
/// Metal ignore the queue's later command buffers
#[test]
fn packed_textures_copy_nothing_and_their_blits_die() {
    let md = MetalPlatform::system_default_device();
    let mp = MetalPlatform::new();
    let size = (32, 16);
    let named = |line: Option<String>| line.unwrap_or_default();
    for format in [508, 509, 510, 528, 529, 530] {
        let (s, other) = noisy_pair(format, size, 11);
        let before = bytes_of(&s);
        let ours = written(&platform(), &dev(), Op::Replace, format, (&s, &other), size);
        assert!(ours[0] == before, "{format}");
        if let Some(md) = &md {
            let (s, other) = noisy_pair(format, size, 11);
            let theirs = written(&mp, md, Op::Replace, format, (&s, &other), size);
            assert!(ours == theirs, "Metal: {format}");
        }
        for op in [Op::ToBuffer, Op::FromBuffer] {
            *crate::answer::LAST_LINE.lock() = None;
            let site = catch_death(|| written(&platform(), &dev(), op, format, (&s, &other), size))
                .unwrap_err();
            let line = named(crate::answer::LAST_LINE.lock().take());
            assert_eq!(
                (site.kind, site.file),
                (DeathKind::Abort, op.sel()),
                "{line}"
            );
            assert!(line.contains("Format"), "{format} {op:?}: {line}");
        }
        *crate::answer::LAST_LINE.lock() = None;
        let site =
            catch_death(|| written(&platform(), &dev(), Op::Clear, format, (&s, &other), size))
                .unwrap_err();
        let line = named(crate::answer::LAST_LINE.lock().take());
        assert_eq!(
            (site.kind, site.file),
            (DeathKind::Abort, Op::Clear.sel()),
            "{line}"
        );
        let pass = format!("a render pass into a texture of pixel format {format} (");
        assert!(line.contains(&pass), "{format} Clear: {line}");
        assert!(
            line.contains("no exact form exists"),
            "{format} Clear: {line}"
        );
        assert!(line.contains("Format"), "{format} Clear: {line}");
    }
}

/// A render target of a 4:2:0 or 4:2:2 format of the family over a surface of an odd width or
/// height is nil, as Metal's is, and one of a 4:4:4 format is made, as Metal's is
#[test]
fn odd_render_targets_of_subsampled_formats_are_nil_as_metals() {
    let md = MetalPlatform::system_default_device();
    for format in family() {
        let (pf, ostypes) = surfaces_of(format, true);
        let f = metal2vulkan::metal::format::ycbcr_format(pf).unwrap();
        let subsampled = f.subsampling.halved().contains(&true);
        for size in [(33, 17), (33, 16), (32, 17)] {
            let s = Shape::of(ostypes[0]).surface(size);
            let mut desc = measure::desc(format, size);
            desc.usage = 0x5;
            let ours =
                dev().new_texture_on_surface(&platform().texture_descriptor(&desc), Some(&s), 0);
            let what = format!("{format} of {size:?}");
            assert_eq!(ours.unwrap().is_some(), !subsampled, "{what}");
            if let Some(md) = &md {
                let desc = reims_vgpu_metal::gpu::desc::texture_descriptor(&desc);
                let theirs = md.new_texture_on_surface(&desc, Some(&s), 0).unwrap();
                assert_eq!(theirs.is_some(), !subsampled, "Metal: {what}");
            }
        }
    }
}

/// A kernel's `write()` into a texture of a 4:2:0 or 4:2:2 format of the family over a surface,
/// packed or not, or of a packed 4:4:4 one (510, 530), leaves the surface's bytes as they were,
/// as Metal's does on this host
#[test]
fn kernel_writes_metal_drops_are_dropped_as_metals() {
    let Some(md) = MetalPlatform::system_default_device() else {
        return;
    };
    let mp = MetalPlatform::new();
    let size = (32, 16);
    let dropped = family().into_iter().filter(|&f| {
        let pf = PixelFormat::try_from(u32::try_from(f).unwrap()).unwrap();
        metal2vulkan::metal::format::writes_dropped(pf)
    });
    for format in dropped {
        let (s, other) = noisy_pair(format, size, 7);
        let before = bytes_of(&s);
        let ours = written(&platform(), &dev(), Op::Kernel, format, (&s, &other), size);
        assert!(ours[0] == before, "{format}");
        let (s, other) = noisy_pair(format, size, 7);
        let theirs = written(&mp, &md, Op::Kernel, format, (&s, &other), size);
        assert!(ours == theirs, "Metal: {format}");
    }
}

/// A fragment function's `write()` into a texture of 503, 507, 542 or 545 over a surface, which
/// a draw makes over every texel, stores its codes into the surface's planes as Metal's does,
/// byte for byte, as a kernel's does (`writes_of_444_textures_are_metals`)
#[test]
fn fragment_writes_of_444_textures_are_metals() {
    let md = MetalPlatform::system_default_device();
    let mp = MetalPlatform::new();
    let size = (32, 16);
    for format in [503, 507, 542, 545] {
        let (s, other) = noisy_pair(format, size, 11);
        let before = bytes_of(&s);
        let ours = written(
            &platform(),
            &dev(),
            Op::Fragment,
            format,
            (&s, &other),
            size,
        );
        assert!(ours[0] != before, "{format}'s planes");
        if let Some(md) = &md {
            let (s, other) = noisy_pair(format, size, 11);
            let theirs = written(&mp, md, Op::Fragment, format, (&s, &other), size);
            assert!(ours == theirs, "Metal: {format}");
        }
    }
}

/// A kernel's `write()` into a texture of 503, 507, 542 or 545 over a surface stores its codes
/// into the surface's planes as Metal's does, byte for byte
#[test]
fn writes_of_444_textures_are_metals() {
    let md = MetalPlatform::system_default_device();
    let mp = MetalPlatform::new();
    let size = (32, 16);
    let stored = family().into_iter().filter(|&f| {
        let pf = PixelFormat::try_from(u32::try_from(f).unwrap()).unwrap();
        metal2vulkan::metal::format::ycbcr_writes_stored(pf)
    });
    let stored: Vec<u64> = stored.collect();
    assert_eq!(stored, [503, 507, 542, 545]);
    for format in stored {
        let (s, other) = noisy_pair(format, size, 9);
        let before = bytes_of(&s);
        let ours = written(&platform(), &dev(), Op::Kernel, format, (&s, &other), size);
        assert!(ours[0] != before, "{format}'s planes");
        if let Some(md) = &md {
            let (s, other) = noisy_pair(format, size, 9);
            let theirs = written(&mp, md, Op::Kernel, format, (&s, &other), size);
            assert!(ours == theirs, "Metal: {format}");
        }
    }
}

/// A kernel's `write()` into a texture over a surface of a 4:4:4 format of the family whose
/// writes the runtime refuses dies at the dispatch, naming the write and the format: the `_sRGB`
/// twins 523 and 527, whose planes Metal's changes
#[test]
fn writes_the_runtime_refuses_die() {
    let md = MetalPlatform::system_default_device();
    let mp = MetalPlatform::new();
    let size = (32, 16);
    let refused = family().into_iter().filter(|&f| {
        super::writes_refused(PixelFormat::try_from(u32::try_from(f).unwrap()).unwrap())
    });
    let refused: Vec<u64> = refused.collect();
    assert_eq!(refused, [523, 527]);
    for format in refused {
        let pf = PixelFormat::try_from(u32::try_from(format).unwrap()).unwrap();
        let (s, other) = noisy_pair(format, size, 9);
        *crate::answer::LAST_LINE.lock() = None;
        let site =
            catch_death(|| written(&platform(), &dev(), Op::Kernel, format, (&s, &other), size))
                .unwrap_err();
        let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
        assert_eq!(
            (site.kind, site.file),
            (DeathKind::Abort, Op::Kernel.sel()),
            "{format}: {line}"
        );
        assert!(line.contains(&format!("a texture of {pf:?}")), "{line}");
        assert!(line.contains("the runtime refuses"), "{line}");
        if let Some(md) = &md {
            let (s, other) = noisy_pair(format, size, 9);
            let before = bytes_of(&s);
            let theirs = written(&mp, md, Op::Kernel, format, (&s, &other), size);
            assert!(theirs[0] != before, "Metal: {format}");
        }
    }
}

/// A kernel's `write()` into a texture of 503, 507, 542 or 545 over plane 1 of a surface, which
/// the runtime makes, dies at the dispatch, naming the runtime's refusal (`Format`): it stores
/// such a texture's writes over plane 0 alone, and Metal's over plane 1 are not measured
#[test]
fn writes_of_444_textures_over_plane_1_die() {
    let size = (32, 16);
    for format in [503, 507, 542, 545] {
        let (s, other) = noisy_pair(format, size, 13);
        *crate::answer::LAST_LINE.lock() = None;
        let site = catch_death(|| {
            written_at(
                &platform(),
                &dev(),
                (Op::Kernel, 1),
                format,
                (&s, &other),
                size,
            )
        })
        .unwrap_err();
        let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
        assert_eq!(
            (site.kind, site.file),
            (DeathKind::Abort, Op::Kernel.sel()),
            "{format}: {line}"
        );
        assert!(line.contains("Format"), "{format}: {line}");
    }
}

/// A texture of 505 over a surface whose luma samples hold every 10-bit code, 1024 texels of
/// 32 by 32, under chroma codes that vary by block, reads into floats as Metal's does, bit for
/// bit, at the video and the full range ('x420', 'xf20')
#[test]
fn every_luma_code_of_505_reads_as_metals() {
    let size = (32, 32);
    let d = dev();
    for ostype in [b"x420", b"xf20"] {
        let shape = Shape::of(ostype);
        let s = shape.surface(size);
        shape.fill_codes(
            &s,
            size,
            |x, y| u16::try_from(y * 32 + x).unwrap(),
            |x, y| {
                let cb = (x * 67 + y * 131) % 1024;
                let cr = (x * 97 + y * 29 + 512) % 1024;
                (u16::try_from(cb).unwrap(), u16::try_from(cr).unwrap())
            },
        );
        let t = ours(505, &s, size);
        let ours = float_bits(&measure::read_floats::<VulkanPlatform>(&d, &t, size));
        let Some((md, mt)) = metals(505, &s, size) else {
            return;
        };
        let theirs = float_bits(&measure::read_floats::<MetalPlatform>(&md, &mt, size));
        let differ = ours.iter().zip(&theirs).filter(|(a, b)| a != b).count();
        assert_eq!(differ, 0, "{}", String::from_utf8_lossy(ostype));
    }
}
