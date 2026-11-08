//! The two-plane YCbCr pixel formats, which Metal makes textures of over an IOSurface's planes
//! alone. A texture descriptor holds each as the runtime's format, and every creator but the one
//! over a surface answers it as Metal does; 505 and 543 were measured first. One harness measures them (`measure.rs`: the tests' surfaces of each OSType's
//! layout, and the probes that read a texture of any of the formats on Metal and here), and the
//! tests compare the runtime's textures of each format it reads with Metal's (`sampling.rs`).
//!
//! **505, `MTLPixelFormatYCBCR10_420_2P`**, Metal's name for it (`MTLPixelFormatGetInfoForDevice()`
//! answers it), which a guest's mapper textures over IOSurfaces take. Measured on Metal on this
//! host (M3 Max, macOS 26.7, without API validation, one case per process):
//! - its description: flags `0x783001d`, castClass 48, block words 2, 515, 6 and 6
//!   (the runtime's, [`crate::gpu::format_info`]);
//! - `newTextureWithDescriptor:`, a buffer's `newTextureWithDescriptor:offset:bytesPerRow:` and
//!   `newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:`, and a placement heap's
//!   `newTextureWithDescriptor:offset:` answer nil for every descriptor their validation passes:
//!   1D, 1D array, 2D, 2D array, 3D, cube, cube array and texture buffer types, at
//!   sizes from 1 by 1 to the largest, with every level count the size takes, in every storage
//!   mode, of every usage but `ShaderAtomic`, swizzled without `ShaderWrite` or `RenderTarget`
//!   usage or with `writeSwizzleEnabled`, `framebufferOnly`, and rotated 2 or 4 with
//!   `framebufferOnly`. The validation aborts first where it does for every format (a size of
//!   0, a rotation past 4 or without `framebufferOnly`, a usage bit Metal does not define), and
//!   at a multisample texture ("pixelFormat (MTLPixelFormatYCBCR10_420_2P) does not support
//!   multisample textures") and `ShaderAtomic` usage ("is not a valid format to be used with
//!   texture atomics"). The nil comes before each creator's own checks, which a texture of
//!   another format aborts or is nil at: `framebufferOnly`, an offset or rows off their alignment
//!   or short of a row, bytes past the buffer's or the heap's end, and a storage mode other than
//!   the heap's;
//! - `newSharedTextureWithDescriptor:` answers nil for every descriptor, first: one the validation
//!   aborts at too;
//! - `heapTextureSizeAndAlignWithDescriptor:` answers 0 bytes, after the same validation, at an
//!   alignment of the type alone: 128 for a 2D array, 16 for every other type (4720 descriptors
//!   of every type, at sizes from 1 by 1 to 16384 by 16384, array lengths from 1 to 2048, one
//!   level and the whole chain, three usages and every storage mode);
//! - `newTextureWithDescriptor:iosurface:plane:` makes a texture over plane 0 of every surface
//!   tried ('x420', 'xf20', '420v', '420f', 'x422', 'xf22', 'x444', 'xf44', '&xv0', 'BGRA',
//!   'L016', '2C16') and over plane 1 of a two-plane one, and aborts at plane 1 of a one-plane
//!   surface. It takes `RenderTarget` and `ShaderWrite` usage; a view of it in `R16Unorm` aborts;
//!   a blit of it to a buffer completes;
//! - sampling that texture answers RGB, alpha 1, in half precision. Each texel is converted from
//!   the plane-0 luma and the plane-1 chroma, which is filtered bilinearly at the luma texel's
//!   position, with the chroma sited at the middle of each 2 by 2 block of luma; by BT.601, or by
//!   BT.709 where the surface's `IOSurfaceYCbCrMatrix` is `ITU_R_709_2`; over the full range for
//!   'xf20' and the narrow range for 'x420' and an OSType Metal does not know; and clamped to 0
//!   to 1. The sampler then filters those texels at the luma's size, and `read` answers the
//!   nearest sample at a texel's centre. A surface made as core makes one, with no attachment
//!   but its OSType, is sampled alike.
//!
//! **543, `MTLPixelFormatRGB10_420_2P`**, Metal's name for it, measured alike:
//! - its description: 505's but castClass 0 (the runtime's);
//! - Metal's validation and answers are 505's in every case above: nil for the same
//!   descriptors from the same creators, the heap's size query alike, the same aborts, a texture
//!   over plane 0 or 1 of every surface tried and an abort
//!   at plane 1 of a one-plane surface, `RenderTarget` and `ShaderWrite` usage taken, and an
//!   `R16Unorm` view aborting;
//! - sampling it answers red Cr, green Y and blue Cb, each code over 1023 at full precision,
//!   alpha 1, with 505's chroma filter, and no range, offset or matrix: as an `RGB10A2Unorm`
//!   texture of those codes samples, filters included;
//! - under every address mode, Metal's 543 clamps to the edge outside 0 to 1, as
//!   `clamp_to_edge` does.
//!
//! **The other two-plane formats**, the 67 more of the runtime's `ycbcr_format` with 505 and 543
//! ([`two_plane`]; `MTLPixelFormatYCBCR8_420_2P`, 500, and `MTLPixelFormatRGB8_420_2P`, 540,
//! among them), a descriptor holds as the runtime's format too. Every creator
//! but the one over an IOSurface answers a descriptor of one as Metal answers one of 505, and so
//! does the runtime, which answers each of the 69 ([`crate::gpu::no_texture::descriptor`], where
//! it does not). Measured on Metal on this host for each of the 69 where Metal answers (the
//! tests' cases: nil from every creator, and 0 bytes from the heap's size query, at 505's
//! alignment); where Metal's validation aborts, which ends its process, 505's and 543's
//! measurements stand for them all (and the runtime's of every one at a multisample texture),
//! and the runtime's validation dies there for each of the 69. Over an IOSurface the runtime makes
//! textures of the 24 of them it reads ([`runtime_makes`], `crate::gpu::device`): 500, 502, 503,
//! 505, 506 and 507, the packed 508, 509 and 510, their `_sRGB` twins 520, 522, 523, 525, 526,
//! 527, 528, 529 and 530, and 540 to 545. It refuses the others, which die there, naming the refusal, where Metal makes them. It makes a
//! view of each converted format in its twin, where Metal makes them, and refuses the reverse ones
//! ([`twin_view`]). A view in a texture's own format as a 2D array reads the luma codes alone, as
//! Metal's does, and a twin's as its converted format's, where Metal's reads no defined value
//! (`own_format_views_are_metals`). A packed format's texture writes nothing on `replaceRegion:`, as Metal's
//! leaves both planes; its blits to and from a buffer, which Metal runs at rows that follow no
//! layout of the texture's, die, naming the runtime's refusal
//! (`packed_textures_copy_nothing_and_their_blits_die`). A render target of a 4:2:0 or 4:2:2
//! format of an odd width or height is nil, as Metal's is
//! (`odd_render_targets_of_subsampled_formats_are_nil_as_metals`).
//!
//! The runtime makes no texture of 505 or 543 from a descriptor alone. So the descriptor holds
//! the format and answers it ([`crate::gpu::desc::TextureFields`]), and every creator but the one
//! over an IOSurface passes it to the runtime, which answers it as Metal does: nil, or 0 bytes from the heap's size query, where Metal answers so, and a death where
//! Metal's validation aborts ([`crate::gpu::no_texture::descriptor`]). Over an IOSurface the
//! runtime makes both, as it makes 500's ([`runtime_makes`]): over plane 0 of an 'x420', 'xf20',
//! 'x422', 'xf22', 'x444' or 'xf44' surface whose `kIOSurfaceYCbCrMatrix` is BT.601, BT.709,
//! BT.2020, SMPTE 240M or none, read as Metal reads them (the tests compare them with Metal over
//! each format and OSType). So does each format of the family over plane 0 of a two-plane
//! surface of another OSType (of other bits or packing, or a coarser chroma plane), each code in
//! the format's own layout and subsampling, over plane 1 of one, its codes the luma and the chroma
//! codes 0, and, but an `_sRGB` twin, over a surface whose matrix is `DCI_P3` or `P3_D65`, each
//! code read alone, each read as Metal's (`textures_over_other_ostypes_read_as_metals`,
//! `textures_over_plane_1_read_as_metals`). What the runtime refuses of them (an `_sRGB` twin
//! over a P3 surface, a texture whose rows overlap on the surface, such as a 10-bit format's over
//! rows of 8-bit codes, a lossless compressed OSType, planes off its alignment, a load, write or
//! pass it has no form for) dies at its selector, naming the refusal, where Metal goes on.
//!
//! **Address modes.** Metal addresses a texture of the family by its format: a 4:2:0 or 4:2:2 one
//! samples repeat, mirrored repeat and mirror-clamp-to-edge as clamp-to-edge, a 4:4:4 one as
//! those modes, and one read as codes (540 to 545) samples clamp-to-zero and
//! clamp-to-border-colour as clamp-to-edge too (the comparisons hold the runtime's textures to
//! it). The runtime maps the samplers it binds so, and a constexpr sampler too, through a variant
//! of the pipeline whose modes are mapped (the comparisons hold it to Metal under repeat,
//! mirrored repeat and clamp to zero). It refuses a command that samples such a texture where it
//! cannot map its sampler: through a sampler slot that samples, in the same stage, a texture
//! whose map gives it other modes, by a function that samples through handles (an argument
//! buffer's or a heap's) able to reach it, and through a constexpr sampler whose modes are not
//! stated or that the map sends to a mode a constexpr sampler cannot address. That command dies, its line naming which it is
//! ([`crate::answer::refusal`]), where Metal samples the texture.
//!
//! **In a fragment function** a texture of the family samples and gathers as Metal's does: each
//! of the 24 formats over each OSType it is made over, at each matrix, through every sampler and
//! probe the kernels' comparisons take, into halves and into floats, is Metal's bit for bit there
//! (`family_samples_in_fragments_as_metals`), as 500's linear samples magnified and minified are
//! (`ycbcr8_420_textures_over_surfaces_read_as_metals`).
//!
//! Not measured: anisotropic filtering (a texture over an IOSurface has one level).

use metal2vulkan::metal::PixelFormat;
use metal2vulkan_runtime as rt;

use crate::answer::unrepresentable;

/// Whether the runtime refuses a GPU write of a texture of `pf` over a surface that Metal's GPU
/// writes otherwise: a 4:4:4 format of the family whose writes the runtime neither stores as
/// Metal does (`metal::format::ycbcr_writes_stored`) nor drops as Metal does
/// (`metal::format::writes_dropped`), the `_sRGB` twins 523 and 527, whose planes Metal's kernel
/// stores change by a rule the runtime does not hold (measured:
/// `writes_the_runtime_refuses_die`). The runtime stores a write of 503, 507, 542 and 545 into
/// the surface's planes as Metal's does, a kernel's and a fragment function's
/// (`writes_of_444_textures_are_metals`, `fragment_writes_of_444_textures_are_metals`), and
/// drops a 4:2:0 or 4:2:2 one's and the packed 510's and 530's, as Metal's GPU drops them on
/// this host (`kernel_writes_metal_drops_are_dropped_as_metals`).
pub(crate) fn writes_refused(pf: PixelFormat) -> bool {
    use metal2vulkan::metal::format::{writes_dropped, ycbcr_format, ycbcr_writes_stored};
    runtime_makes(pf)
        && !ycbcr_writes_stored(pf)
        && !writes_dropped(pf)
        && ycbcr_format(pf).is_some_and(|f| f.subsampling.halved() == [false, false])
}

/// Dies at `sel`, a command that may write `what` a texture of `pf` over a surface
/// ([`writes_refused`]), whose write the runtime refuses where Metal's GPU stores it into the
/// surface's planes.
pub(crate) fn refused_write(sel: &'static str, what: &str, pf: PixelFormat) -> ! {
    crate::answer::unrepresentable(
        sel,
        &format!(
            "{what} a texture of {pf:?} over a surface, a write the runtime refuses and Metal's \
             GPU stores into the surface's planes"
        ),
    )
}

/// Dies at `sel`, the creator of the render pass `d`, which the runtime refused with `e`, a
/// refusal of a pixel format ([`crate::gpu::desc::is_format_refusal`]), where a colour attachment
/// of `d` is a texture of a YCbCr format: one of the two-plane family ([`two_plane`]), or a 4:2:2
/// one (`GBGR422`, `BGRG422`). It returns where none is, and where the refusal names the format
/// of none of them: the refusal is another's. Its line names the format of the attachment the
/// refusal names (the first such attachment where it names none), its raw value and Metal's name,
/// and what Metal does where the runtime refuses, then the refusal ([`crate::answer::refusal`]):
/// - a packed 10-bit format's pass (508 to 510, 528 to 530, the runtime's `ycbcr_pass_refused`),
///   whose command buffer Metal fails, the error (`Hang` or `InnocentVictim`) and the luma bytes
///   it leaves varying from run to run, and after a second such failure ignores the queue's later
///   command buffers: no exact form exists, so the pass dies where it is made;
/// - a pass into 500, 502, 503, 505, 506 or 507, or into one read as codes (540 to 545), the
///   runtime does not store as Metal
///   (`ycbcr_renders`: it stores a pass over plane 0 of a size its chroma blocks tile, by any load
///   action, a `Store` or `DontCare` store, under any write mask and blend, its blend's codes
///   within one of Metal's and a loaded pass's tiles no draw reaches re-encoded, which the runtime
///   announces, `ycbcr-blend-rounding` and `ycbcr-load-tiles` of [`crate::answer::DIES`]): a
///   resolve, a texture over plane 1, or a size the blocks do not tile, none of them measured;
/// - every other format's pass, which Metal stores into the texture by a rule the runtime does not
///   hold.
pub(crate) fn pass_refused(sel: &'static str, d: &rt::RenderPassDescriptor, e: &rt::Error) {
    use metal2vulkan::metal::format::{ycbcr_pass_refused, ycbcr_renders};
    let ycbcr = |pf: PixelFormat| {
        two_plane(pf) || matches!(pf, PixelFormat::GBGR422 | PixelFormat::BGRG422)
    };
    let mut formats = (d.color_attachments.iter())
        .filter_map(|c| c.texture.as_ref())
        .map(rt::Texture::pixel_format)
        .filter(|&pf| ycbcr(pf));
    // The format the refusal names, by Metal's enumerator name (`naming_format`).
    let named = match e {
        rt::Error::Unsupported(u) => u.symbol.as_ref().map(ToString::to_string),
        _ => None,
    };
    let found = match named {
        Some(named) => formats.find(|pf| pf.name() == named),
        None => formats.next(),
    };
    let Some(pf) = found else {
        return;
    };
    let raw = u64::from(pf.as_raw());
    let name = crate::gpu::format_info::name(sel, raw);
    let metal = if ycbcr_pass_refused(pf) {
        "whose command buffer Metal fails, its error (Hang or InnocentVictim) and the luma bytes \
         it leaves varying from run to run, and after a second such failure ignores the queue's \
         later command buffers: no exact form exists"
    } else if ycbcr_renders(pf) {
        "which Metal stores into both of the surface's planes, where the runtime stores a pass \
         over plane 0 of a size its chroma blocks tile, without a resolve: no other shape was \
         measured"
    } else {
        "which Metal stores into the texture by a rule the runtime does not hold"
    };
    unrepresentable(
        sel,
        &format!(
            "a render pass into a texture of pixel format {raw} ({name}), {metal}; {}",
            crate::answer::refusal(e)
        ),
    );
}

#[cfg(test)]
pub(crate) mod measure;
#[cfg(test)]
#[cfg(target_os = "macos")]
pub(crate) mod sampling;

/// Whether the runtime makes a texture of `pf` over an IOSurface's plane, given the surface's
/// YCbCr facts (`crate::os::ycbcr_facts`), by its own answer
/// (`metal2vulkan::metal::format::ycbcr`): the two-plane formats of unpacked 8- or 10-bit codes
/// read converted (`YCBCR8_420_2P`, 500, `YCBCR8_422_2P`, 502, `YCBCR8_444_2P`, 503, and the
/// 10-bit 505, 506 and 507) or through the sRGB curve (their `_sRGB` twins, 520, 522, 523, 525,
/// 526 and 527), the packed 10-bit ones, three codes to a 32-bit word (508, 509 and 510, and
/// their twins 528, 529 and 530, over the packed 'p420', 'pf20', 'p422', 'pf22', 'p444' and
/// 'pf44' alone), and the ones read as codes (`RGB8_420_2P`, 540, `RGB8_422_2P`, 541,
/// `RGB8_444_2P`, 542, and the 10-bit 543, 544 and 545, red Cr, green Y and blue Cb, each its
/// code over the largest, alpha 1). It makes each over plane 0 of a surface of the format's
/// bits and packing whose chroma plane holds the format's own (a 4:2:0 format over a 4:2:0,
/// 4:2:2 or 4:4:4 surface, its chroma read as the format's at the surface's rows; a 4:2:2 one
/// over a 4:2:2 or 4:4:4 one; a 4:4:4 one over a 4:4:4 one), whose `kIOSurfaceYCbCrMatrix` is
/// none, BT.601, BT.709, BT.2020 or SMPTE 240M (read as BT.709), as Metal reads them. It refuses
/// the others, typed: each a death at the creator, where Metal makes the texture.
pub fn runtime_makes(pf: PixelFormat) -> bool {
    metal2vulkan::metal::format::ycbcr(pf)
}

/// Whether a view of a texture of `source` in `view`, Metal's raw formats, is the runtime's to
/// answer ahead of the formats Metal makes no view in from a texture of a format the runtime has
/// (`crate::gpu::no_texture::no_view`): one between two formats the runtime makes over a surface
/// ([`runtime_makes`]) of the same codes and chroma, one read converted and the other its `_sRGB`
/// twin. The runtime makes a view of a converted format in its twin, as Metal does (`ShaderRead`
/// alone), reading what a texture of the twin over the same surface reads, and refuses the
/// reverse ones, which nothing measured, typed: they die at the view.
pub fn twin_view(source: u64, view: u64) -> bool {
    use metal2vulkan::metal::format::{YCbCrReads, ycbcr_format};
    let made = |raw: u64| {
        let pf = PixelFormat::try_from(u32::try_from(raw).ok()?).ok()?;
        runtime_makes(pf).then(|| ycbcr_format(pf)).flatten()
    };
    let (Some(a), Some(b)) = (made(source), made(view)) else {
        return false;
    };
    let twins = matches!(
        (a.reads, b.reads),
        (YCbCrReads::Converted, YCbCrReads::Srgb) | (YCbCrReads::Srgb, YCbCrReads::Converted)
    );
    twins && (a.bits, a.subsampling, a.packed) == (b.bits, b.subsampling, b.packed)
}

/// Whether `pf`, a format the runtime has, is one of the two-plane YCbCr and RGB formats Metal
/// makes textures of over an IOSurface's planes alone (the runtime's `ycbcr_format`, 69 of them,
/// the formats here among them): Metal answers a descriptor of one as it answers one of 505's
/// from every other creator (the module's docs).
pub fn two_plane(pf: PixelFormat) -> bool {
    metal2vulkan::metal::format::ycbcr_format(pf).is_some()
}

#[cfg(test)]
mod tests {
    use crate::core::raise::death::DeathKind;
    use crate::core::testkit::capture::catch_death;

    use crate::gpu::device::tests::dev;

    /// A pass the runtime refuses, of two 4:2:2 colour attachments, dies naming the attachment
    /// whose format the refusal names, `GBGR422` (240) at index 1, not the first, `BGRG422` (241);
    /// one whose refusal names no format names the first; one whose refusal names a format of
    /// none of them returns, the refusal another's
    #[test]
    fn a_refused_pass_names_the_attachment_the_refusal_names() {
        use metal2vulkan::error::{Feature, Unsupported};
        use metal2vulkan::metal::PixelFormat;
        use metal2vulkan_runtime as rt;
        let d = dev();
        let texture = |pf| {
            let desc = rt::TextureDescriptor::texture_2d(pf, 16, 16, false);
            d.raw.new_texture(&desc).unwrap()
        };
        let mut pass = rt::RenderPassDescriptor::new();
        pass.color_attachments[0].texture = Some(texture(PixelFormat::BGRG422));
        pass.color_attachments[1].texture = Some(texture(PixelFormat::GBGR422));
        let refusal = |named: Option<PixelFormat>| {
            let refused = Unsupported::not_implemented(Feature::Format);
            rt::Error::Unsupported(match named {
                Some(pf) => refused.naming_format(pf),
                None => refused,
            })
        };
        let sel = "renderCommandEncoderWithDescriptor:";
        for (named, raw) in [(Some(PixelFormat::GBGR422), 240), (None, 241)] {
            *crate::answer::LAST_LINE.lock() = None;
            let site =
                catch_death(|| super::pass_refused(sel, &pass, &refusal(named))).unwrap_err();
            let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
            assert_eq!((site.kind, site.file), (DeathKind::Abort, sel), "{line}");
            let format = format!("a render pass into a texture of pixel format {raw} (");
            assert!(line.contains(&format), "{named:?}: {line}");
        }
        super::pass_refused(sel, &pass, &refusal(Some(PixelFormat::BGRA8Unorm)));
    }

    #[cfg(target_os = "macos")]
    mod metal_tests;
}
