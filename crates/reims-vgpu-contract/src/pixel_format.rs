//! Metal pixel-format helpers (port of `host/utils/reims-vgpu-pixel-format`).

use crate::endian::{ld16, st16};

pub const COMPONENT_COUNT: usize = 4;
pub const COMPONENT_R: usize = 0;
pub const COMPONENT_G: usize = 1;
pub const COMPONENT_B: usize = 2;
pub const COMPONENT_A: usize = 3;

pub const R8_BPP: u32 = 1;
pub const RG8_BPP: u32 = 2;
pub const RGBA8_BPP: u32 = 4;
pub const BGRA8_BPP: u32 = RGBA8_BPP;
pub const R16F_BPP: u32 = 2;
/// One sixteen-bit normalized channel — a ten-bit video luma plane.
pub const R16_BPP: u32 = 2;
/// Two of them: the matching chroma plane.
pub const RG16_BPP: u32 = 4;
pub const R32F_BPP: u32 = 4;
pub const RG16F_BPP: u32 = 4;
pub const RGBA16_BPP: u32 = 8;
pub const RGBA16F_BPP: u32 = RGBA16_BPP;
pub const RGBA32_BPP: u32 = 16;
pub const RGBA32F_BPP: u32 = RGBA32_BPP;
pub const R32_BPP: u32 = 4;

// MTLPixelFormat values (Metal.framework Headers/MTLPixelFormat.h).
pub const MTL_FORMAT_A8_UNORM: u16 = 0x01;
pub const MTL_FORMAT_R8_UNORM: u16 = 0x0a;
/// `MTLPixelFormatR8Uint`. Metal numbers the R8 family consecutively from
/// `R8Unorm` at 10 — `R8Unorm_sRGB`, `R8Snorm`, `R8Uint`, `R8Sint` — so the
/// unsigned-integer member is 13.
///
/// Declared for the same reason as [`MTL_FORMAT_RGBA16_UNORM`] below: its
/// absence was a *decode* gap, not a rail gap. `bytes_per_pixel` answered `None`
/// for it, and since `crate::runtime::draw::effective_view_sample_format` asks
/// that question about both the base and the view before anything else looks at
/// the bind, every path refused it as `format_incompatible` — a slug that reads
/// as "the guest asked for an illegal reinterpretation" when what happened is
/// that this crate had never heard of the format. A macOS 26 guest stages one
/// into compute dispatches, which is what surfaced it.
///
/// Being *declared* is not being *sampled*: an integer texel must not be run
/// through the unorm converters, so it has no `crate::backend::vulkan` texel
/// layout and no storage selector, and both of those decline it by name.
pub const MTL_FORMAT_R8_UINT: u16 = 0x0d;
/// `MTLPixelFormatR16Unorm`. The luma plane of a ten-bit biplanar video
/// surface (`'x420'`), where the eight-bit shape uses
/// [`MTL_FORMAT_R8_UNORM`].
pub const MTL_FORMAT_R16_UNORM: u16 = 0x14;
pub const MTL_FORMAT_R16_FLOAT: u16 = 0x19;
pub const MTL_FORMAT_RG8_UNORM: u16 = 0x1e;
/// `MTLPixelFormatRG8Uint`. The RG8 family runs from `RG8Unorm` at 30 the same
/// way the R8 family runs from `R8Unorm` at 10, so the unsigned-integer member
/// is 33.
///
/// Declared for the reason [`MTL_FORMAT_R8_UINT`] was, and *found* by declaring
/// it: with `R8Uint` admitted to the width gate, the same macOS 26 compute
/// dispatches came back refusing `0x21` instead, at an identical count. One
/// dispatch binds both, so neither alone recovers it — which is why the count
/// did not move and reading it as "the fix did nothing" would have been wrong.
///
/// Whether that pairing is the integer twin of the [`MTL_FORMAT_R8_UNORM`] /
/// [`MTL_FORMAT_RG8_UNORM`] biplanar shape this table already carries is a
/// guess, and is deliberately not written into the code.
///
/// Integer texels, so the same restriction as [`MTL_FORMAT_R8_UINT`]: declared,
/// and refused by name by every rail that would have to give them a meaning.
pub const MTL_FORMAT_RG8_UINT: u16 = 0x21;
/// `MTLPixelFormatRG16Unorm`. The chroma half of [`MTL_FORMAT_R16_UNORM`],
/// as [`MTL_FORMAT_RG8_UNORM`] is of [`MTL_FORMAT_R8_UNORM`].
pub const MTL_FORMAT_RG16_UNORM: u16 = 0x3c;
/// `MTLPixelFormatRG16Uint`. The integer member of the RG16 family, the same
/// four bytes as its `Unorm` and `Float` siblings — a family's members share a
/// width and differ only in how the word is read.
pub const MTL_FORMAT_RG16_UINT: u16 = 0x3f;
/// `MTLPixelFormatRG16Sint`. Four bytes, for the reason above.
pub const MTL_FORMAT_RG16_SINT: u16 = 0x40;
pub const MTL_FORMAT_R32_UINT: u16 = 0x35;
pub const MTL_FORMAT_R32_SINT: u16 = 0x36;
pub const MTL_FORMAT_R32_FLOAT: u16 = 0x37;
pub const MTL_FORMAT_RG16_FLOAT: u16 = 0x41;
pub const MTL_FORMAT_RGBA8_UNORM: u16 = 0x46;
pub const MTL_FORMAT_RGBA8_UNORM_SRGB: u16 = 0x47;
pub const MTL_FORMAT_RGBA8_UINT: u16 = 0x49;
pub const MTL_FORMAT_RGBA8_SINT: u16 = 0x4a;
pub const MTL_FORMAT_BGRA8_UNORM: u16 = 0x50;
pub const MTL_FORMAT_BGRA8_UNORM_SRGB: u16 = 0x51;
/// `MTLPixelFormatRGB10A2Unorm`. Ten bits per colour channel and two of alpha,
/// packed into one 32-bit word with red in the low bits.
///
/// # Why the packed family is declared together, and why it is native
///
/// Metal numbers `RGB10A2Unorm`, `RGB10A2Uint`, `RG11B10Float`, `RGB9E5Float`
/// and `BGR10A2Unorm` consecutively from 90; this table carried the fourth of
/// the five and none of the others, so a guest naming any of the rest was
/// refused at the *width* gate — `bytes_per_pixel` answered `None`, and
/// `crate::runtime::draw::effective_view_sample_format` asks that before
/// anything else looks at the bind. The refusal reads as "the guest asked for an
/// illegal reinterpretation" when what happened is that this crate had never
/// heard of the format. That is the same gap [`MTL_FORMAT_R8_UINT`] records, one
/// storage shape over.
///
/// Unlike that one, three of these have an *exact* Vulkan spelling, so they do
/// not stop at being declared: `A2B10G10R10_UNORM_PACK32`,
/// `A2R10G10B10_UNORM_PACK32` and `B10G11R11_UFLOAT_PACK32` are the same bits in
/// the same order, so the guest's own word is sampled unchanged. Converting one
/// to unorm8 instead would throw away the two bits of colour resolution the
/// guest chose the format for, which is [`TexelLayout::R16Unorm`]'s argument at
/// a different bit depth — and [`texel_to_rgba8`] has no arm that could do it
/// anyway.
///
/// This device advertises `isRGB10A2GammaSupported` to the guest
/// (`model::regs::DEVICE_INFO_KEY_RGB10A2_GAMMA`), so a guest taking it at its
/// word and binding one of these was being refused by the device that invited it.
pub const MTL_FORMAT_RGB10A2_UNORM: u16 = 0x5a;
/// `MTLPixelFormatRGB10A2Uint`. The integer twin of
/// [`MTL_FORMAT_RGB10A2_UNORM`], declared for its width and nothing more:
/// integer texels must not run through the unorm converters, so it has no
/// [`TexelLayout`] and no storage selector, and both decline it by name. Same
/// rule as [`MTL_FORMAT_R8_UINT`].
pub const MTL_FORMAT_RGB10A2_UINT: u16 = 0x5b;
/// `MTLPixelFormatRG11B10Float`. Eleven bits of red and green, ten of blue, no
/// alpha, in one 32-bit word — `VK_FORMAT_B10G11R11_UFLOAT_PACK32` exactly. An
/// HDR-intermediate colour format; see [`MTL_FORMAT_RGB10A2_UNORM`].
pub const MTL_FORMAT_RG11B10_FLOAT: u16 = 0x5c;
/// Packed RGB9E5 shared-exponent float. 32-bit texels.
pub const MTL_FORMAT_RGB9E5_FLOAT: u16 = 0x5d;
/// `MTLPixelFormatBGR10A2Unorm`. [`MTL_FORMAT_RGB10A2_UNORM`] with the colour
/// channels the other way round in the word — `VK_FORMAT_A2R10G10B10_UNORM_PACK32`
/// exactly, as `BGRA8Unorm` is to `RGBA8Unorm` one storage shape up.
pub const MTL_FORMAT_BGR10A2_UNORM: u16 = 0x5e;
/// `MTLPixelFormatRGBA16Unorm`. Its ordinal sits between two this table
/// already carries — `RGBA16Uint` at `0x71` and `RGBA16Float` at `0x73` — and
/// its absence was a decode gap rather than a rail gap: `bytes_per_pixel`
/// answered `None`, so every path that asks about a texel width refused it, not
/// just the sampled one.
pub const MTL_FORMAT_RGBA16_UNORM: u16 = 0x6e;
/// The BC (a.k.a. DXT / S3TC) block-compressed families, as Apple numbers them.
///
/// Every one of these stores a **4x4 block of texels** in a fixed 8 or 16 bytes,
/// which is what separates them from everything else in this file: there is no
/// bytes-per-texel for a BC1 texel — it is half a byte — so
/// [`bytes_per_pixel`] deliberately answers `None` for all of them and
/// [`block_geometry`] is the accessor that can describe them. That `None` is
/// load-bearing: it is what keeps a BC format out of every rail that sizes work
/// per texel, which is every rail except the sampled bind.
///
/// # Why the whole family arrives at once
///
/// Elsewhere this file adds members on measurement — `RG8_UNORM` is still absent
/// from [`render_target_bpp`] for exactly that reason. This family is added
/// whole, and the difference is real rather than convenience:
///
/// * **One capability covers all of them.** Vulkan gates BC1 through BC7 behind
///   the single `textureCompressionBC` feature bit, and Metal behind the single
///   `supportsBCTextureCompression`. There is no per-member capability to
///   measure, so a member left out is refused by *this table* rather than by the
///   host.
/// * **There is no per-member conversion to write.** The guest's bytes are the
///   `VK_FORMAT_BC*_BLOCK` payload already, so a member costs one row in each
///   mapping and nothing else. The three conversion arms `render_target_bpp`'s
///   doc obliges a *renderable* format to satisfy do not arise: a BC format is
///   sampled-only, and every other rail refuses it by name.
/// * **A partial family is the shape that bites later.** A game ships BC1 for
///   opaque albedo, BC3 for alpha, BC4/BC5 for single- and two-channel maps and
///   BC7 for anything modern, in one asset pipeline. Admitting the one that was
///   measured and refusing its four siblings buys a second black-texture report.
///
/// # What was measured
///
/// `BC3_RGBA` (`0x86`), on a driven macos-13 x86/Vulkan boot running Asphalt 8:
/// six distinct textures, and the 12 419 draws that sampled them were the only
/// draws still failing once the `'l10r'` attachment and the pipeline-tag
/// refusals were fixed. Its geometry is the confirmation and it is exact — the
/// guest's own descriptors read `L0=1024x1024 bpr=4096`, which is
/// `(1024/4) * 16`, with `alloc=1400832` for the eleven-level pyramid of a
/// 1 MiB base; and `L0=64x64 bpr=256`, which is `(64/4) * 16`. The guest's SDK
/// names `MTLPixelFormatBC3_RGBA = 134` and its paravirt device answers
/// `supportsBCTextureCompression = 1`.
pub const MTL_FORMAT_BC1_RGBA: u16 = 130;
/// sRGB spelling of [`MTL_FORMAT_BC1_RGBA`]. Identical stored bytes; the
/// qualifier is the conversion the sampler applies, which is why it folds onto
/// the same [`TexelLayout`] and is carried as a transfer function instead.
pub const MTL_FORMAT_BC1_RGBA_SRGB: u16 = 131;
/// `MTLPixelFormatBC2_RGBA` — 16 bytes a block, four-bit explicit alpha.
pub const MTL_FORMAT_BC2_RGBA: u16 = 132;
/// sRGB spelling of [`MTL_FORMAT_BC2_RGBA`].
pub const MTL_FORMAT_BC2_RGBA_SRGB: u16 = 133;
/// `MTLPixelFormatBC3_RGBA` — 16 bytes a block, interpolated alpha. DXT5, and
/// the member this family was measured through.
pub const MTL_FORMAT_BC3_RGBA: u16 = 134;
/// sRGB spelling of [`MTL_FORMAT_BC3_RGBA`].
pub const MTL_FORMAT_BC3_RGBA_SRGB: u16 = 135;
/// `MTLPixelFormatBC4_RUnorm` — one channel, 8 bytes a block.
pub const MTL_FORMAT_BC4_R_UNORM: u16 = 140;
/// `MTLPixelFormatBC4_RSnorm` — the signed twin of [`MTL_FORMAT_BC4_R_UNORM`].
pub const MTL_FORMAT_BC4_R_SNORM: u16 = 141;
/// `MTLPixelFormatBC5_RGUnorm` — two channels, 16 bytes a block. The usual
/// tangent-space normal-map encoding.
pub const MTL_FORMAT_BC5_RG_UNORM: u16 = 142;
/// `MTLPixelFormatBC5_RGSnorm` — the signed twin of [`MTL_FORMAT_BC5_RG_UNORM`].
pub const MTL_FORMAT_BC5_RG_SNORM: u16 = 143;
/// `MTLPixelFormatBC6H_RGBFloat` — three signed half-float channels, 16 bytes a
/// block. An HDR source format.
pub const MTL_FORMAT_BC6H_RGB_FLOAT: u16 = 150;
/// `MTLPixelFormatBC6H_RGBUfloat` — the unsigned twin of
/// [`MTL_FORMAT_BC6H_RGB_FLOAT`].
pub const MTL_FORMAT_BC6H_RGB_UFLOAT: u16 = 151;
/// `MTLPixelFormatBC7_RGBAUnorm` — 16 bytes a block, the modern four-channel
/// mode-switching encoding.
pub const MTL_FORMAT_BC7_RGBA_UNORM: u16 = 152;
/// sRGB spelling of [`MTL_FORMAT_BC7_RGBA_UNORM`].
pub const MTL_FORMAT_BC7_RGBA_UNORM_SRGB: u16 = 153;

/// Side of a BC block, in texels. Every BC family uses the same 4x4 grid.
pub const BC_BLOCK_SIDE: u32 = 4;
/// Bytes one block occupies in the two BC weight classes.
pub const BC_BLOCK_BYTES_8: u32 = 8;
/// The 16-byte class — everything that carries explicit alpha or two channels.
pub const BC_BLOCK_BYTES_16: u32 = 16;
/// A 4x4 block of BC1 is 64 texels' worth of colour in 8 bytes, and of BC3 in
/// 16. Stated as a relation so a wrong constant cannot look right.
const _: () = assert!(
    BC_BLOCK_BYTES_16 == BC_BLOCK_BYTES_8 * 2
        && BC_BLOCK_SIDE * BC_BLOCK_SIDE == 16
        && BC_BLOCK_BYTES_8 * 2 == BC_BLOCK_SIDE * BC_BLOCK_SIDE
);

/// The texel grid one addressable unit of storage covers, and its size.
///
/// Uncompressed formats have a 1x1 block whose `bytes` is [`bytes_per_pixel`],
/// so this is not a second vocabulary for them — it is the same number with the
/// grid stated. That is what lets [`tight_row_bytes`] be one expression for
/// both families instead of a branch, and what
/// `a_block_geometry_agrees_with_the_texel_table` holds honest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockGeometry {
    /// Texels a block spans horizontally.
    pub width: u32,
    /// Texels a block spans vertically.
    pub height: u32,
    /// Bytes one block occupies.
    pub bytes: u32,
}

impl BlockGeometry {
    /// Blocks needed to cover `texels` along one axis, rounding up.
    ///
    /// The rounding is the contract and not a convenience: a 2x2 BC3 mip level
    /// still occupies one whole 16-byte block, so a caller that divided down
    /// would read four bytes of a level and none of the tail of a pyramid.
    pub const fn blocks_for(divisor: u32, texels: u32) -> u32 {
        if divisor == 0 {
            return 0;
        }
        texels.div_ceil(divisor)
    }

    /// Blocks in one row of a `texels`-wide image.
    pub const fn blocks_across(self, texels: u32) -> u32 {
        Self::blocks_for(self.width, texels)
    }

    /// Rows of blocks in a `texels`-tall image — what a row loop over this
    /// format must count, rather than the texel height.
    pub const fn block_rows(self, texels: u32) -> u32 {
        Self::blocks_for(self.height, texels)
    }

    /// Whether this describes a compressed format, i.e. a block wider or taller
    /// than one texel.
    pub const fn is_compressed(self) -> bool {
        self.width > 1 || self.height > 1
    }
}

/// Bytes one BC block of `format` occupies, or `None` if it is not a BC format.
///
/// The one place the family membership is spelled. Everything else asks
/// [`block_geometry`] or [`is_block_compressed`].
const fn bc_block_bytes(format: u16) -> Option<u32> {
    Some(match format {
        MTL_FORMAT_BC1_RGBA
        | MTL_FORMAT_BC1_RGBA_SRGB
        | MTL_FORMAT_BC4_R_UNORM
        | MTL_FORMAT_BC4_R_SNORM => BC_BLOCK_BYTES_8,
        MTL_FORMAT_BC2_RGBA
        | MTL_FORMAT_BC2_RGBA_SRGB
        | MTL_FORMAT_BC3_RGBA
        | MTL_FORMAT_BC3_RGBA_SRGB
        | MTL_FORMAT_BC5_RG_UNORM
        | MTL_FORMAT_BC5_RG_SNORM
        | MTL_FORMAT_BC6H_RGB_FLOAT
        | MTL_FORMAT_BC6H_RGB_UFLOAT
        | MTL_FORMAT_BC7_RGBA_UNORM
        | MTL_FORMAT_BC7_RGBA_UNORM_SRGB => BC_BLOCK_BYTES_16,
        _ => return None,
    })
}

/// Whether `format` stores texels in compressed blocks rather than one at a time.
pub const fn is_block_compressed(format: u16) -> bool {
    bc_block_bytes(format).is_some()
}

/// The [`TexelLayout`] a block-compressed guest format's bytes are already in,
/// or `None` for an uncompressed one.
///
/// **The single mapping**, consulted by both the backend-independent sampled
/// loaders in `runtime::draw::texture_view` and by
/// `backend::vulkan::translate::pixel::sampled_pixels`. The uncompressed
/// families reach their layout through [`sampled_class`], whose vocabulary is
/// deliberately narrow and has no room for a block; rather than widen it or keep
/// a second table in the backend, the compressed families are answered here and
/// both sides ask.
///
/// The four sRGB spellings fold onto their linear sibling's layout, as
/// `RGBA8Unorm_sRGB` folds onto `Rgba8`: identical stored bytes, and the
/// qualifier travels as [`SampledByteFormat`]'s source format so the bind picks
/// the `_SRGB_BLOCK` view.
///
/// Saying `Some` is **not** a claim that this host can sample it. That is a
/// capability — `engine::supports_block_compressed_sampled` — and the rail
/// carries it in `NativeUploads::block_compressed`. Naming the layout
/// unconditionally is what makes the refusal a typed one rather than an
/// unrecognised format.
pub fn block_compressed_layout(format: u16) -> Option<TexelLayout> {
    Some(match format {
        MTL_FORMAT_BC1_RGBA | MTL_FORMAT_BC1_RGBA_SRGB => TexelLayout::Bc1Rgba,
        MTL_FORMAT_BC2_RGBA | MTL_FORMAT_BC2_RGBA_SRGB => TexelLayout::Bc2Rgba,
        MTL_FORMAT_BC3_RGBA | MTL_FORMAT_BC3_RGBA_SRGB => TexelLayout::Bc3Rgba,
        MTL_FORMAT_BC4_R_UNORM => TexelLayout::Bc4RUnorm,
        MTL_FORMAT_BC4_R_SNORM => TexelLayout::Bc4RSnorm,
        MTL_FORMAT_BC5_RG_UNORM => TexelLayout::Bc5RgUnorm,
        MTL_FORMAT_BC5_RG_SNORM => TexelLayout::Bc5RgSnorm,
        MTL_FORMAT_BC6H_RGB_FLOAT => TexelLayout::Bc6hRgbFloat,
        MTL_FORMAT_BC6H_RGB_UFLOAT => TexelLayout::Bc6hRgbUfloat,
        MTL_FORMAT_BC7_RGBA_UNORM | MTL_FORMAT_BC7_RGBA_UNORM_SRGB => TexelLayout::Bc7Rgba,
        _ => return None,
    })
}

/// The storage grid `format` addresses, or `None` for a format this crate has
/// no width for at all.
///
/// Derived from [`bytes_per_pixel`] for the uncompressed families rather than
/// re-listed, so the two cannot disagree — the second spelling is the bug this
/// avoids having.
pub fn block_geometry(format: u16) -> Option<BlockGeometry> {
    if let Some(bytes) = bc_block_bytes(format) {
        return Some(BlockGeometry {
            width: BC_BLOCK_SIDE,
            height: BC_BLOCK_SIDE,
            bytes,
        });
    }
    Some(BlockGeometry {
        width: 1,
        height: 1,
        bytes: bytes_per_pixel(format)?,
    })
}

pub const MTL_FORMAT_RGBA16_UINT: u16 = 0x71;
pub const MTL_FORMAT_RGBA16_FLOAT: u16 = 0x73;
pub const MTL_FORMAT_RGBA32_UINT: u16 = 0x7b;
pub const MTL_FORMAT_RGBA32_FLOAT: u16 = 0x7d;
// Depth / stencil (Metal.framework Headers/MTLPixelFormat.h).
pub const MTL_FORMAT_DEPTH16_UNORM: u16 = 250;
pub const MTL_FORMAT_DEPTH32_FLOAT: u16 = 252;
pub const MTL_FORMAT_STENCIL8: u16 = 253;
pub const MTL_FORMAT_DEPTH24_UNORM_STENCIL8: u16 = 255;
pub const MTL_FORMAT_DEPTH32_FLOAT_STENCIL8: u16 = 260;
pub const MTL_FORMAT_X32_STENCIL8: u16 = 261;
pub const MTL_FORMAT_X24_STENCIL8: u16 = 262;

/// The compute rail's own narrowing of `MTLPixelFormat` to the formats a
/// storage image may be, produced once by [`storage_selector`].
///
/// # Every backend owes an answer for every member
///
/// This travels as the enum, never as its ordinal, and that is load-bearing
/// rather than tidiness. It used to be narrowed to `u32` the moment
/// [`storage_selector`] produced it, and both backends then matched raw
/// integers — the Vulkan one with thirteen `s if s == S::X as u32` guard arms,
/// the Metal one against a hand-copied list of constants. Neither shape has
/// coverage a compiler can check, and the Metal one had silently gone a member
/// short: `R32Uint` was declared here and had no arm there, so every `R32Uint`
/// storage bind on the whole arm64 pathway refused while the x86 pathway ran it.
///
/// Both maps are now exhaustive `match`es over this type and both are total, so
/// adding a variant here fails the build until each backend names it. The Metal
/// arm's half of that is reached from a Linux host only by the cross-compiled
/// clippy run, which is the point of stating the rule here rather than there.
///
/// The discriminants are explicit because they are logged as `simg=` and read
/// back from boots; nothing else depends on their values.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StorageImageSelector {
    Rgba8Uint = 0,
    Rgba8Sint = 1,
    Rgba16Uint = 2,
    Rgba16Float = 3,
    Rgba32Float = 4,
    Rgba8Unorm = 5,
    Bgra8Unorm = 6,
    R16Float = 7,
    Rg16Float = 8,
    R8Unorm = 9,
    Rg8Unorm = 10,
    Rgba32Uint = 11,
    R32Uint = 12,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampledClass {
    A8Unorm,
    R8Unorm,
    Rg8Unorm,
    Rgba8Unorm,
    Bgra8Unorm,
    Rgba16Float,
    Rg16Float,
    /// The packed 32-bit word `MTLPixelFormatBGR10A2Unorm` stores a texel in.
    ///
    /// Declared for the cross-check and not for a CPU upload rail. This class is
    /// how [`store_texel_order`]'s admission of that format is held against an
    /// independent statement of the same byte layout, which is what
    /// `a_byte_copy_destination_is_the_texel_every_other_table_agrees_it_is`
    /// asks. `runtime::draw::texture_view`'s `linear_native_upload_format`
    /// answers `None` for it, so a **sampled** bind still takes the native
    /// packed rail `translate::pixel::sampled_pixels` has carried all along;
    /// nothing here moves such a bind to the CPU, which could not serve it
    /// anyway — its channels do not sit on byte boundaries.
    Bgr10a2Unorm,
    /// The two `uint16` channels `MTLPixelFormatRG16Uint` stores a texel in.
    ///
    /// Declared for the cross-check, on [`Self::Bgr10a2Unorm`]'s terms and for
    /// its reason: [`store_texel_order`] admits that format because the byte
    /// copy is the only rail that can land it, and this is the independent
    /// statement of the same byte layout that
    /// `a_byte_copy_destination_is_the_texel_every_other_table_agrees_it_is`
    /// holds it against.
    ///
    /// It is **not** a CPU upload class and there is no rail that would make it
    /// one: an integer texel has no eight-bit expansion, which is the whole of
    /// [`TexelLayout::Rg16Uint`]'s argument. A sampled bind of this format takes
    /// the native rail.
    Rg16Uint,
    /// Four `float32` channels, the widest sampled texel this device binds.
    ///
    /// See [`TexelLayout::Rgba32Float`] for why it is native-or-nothing.
    Rgba32Float,
}

/// The byte layout of one guest texel on the sampled rails, independent of any
/// host graphics API.
///
/// This is the vocabulary `runtime/` speaks about sampled texels. It is
/// deliberately *narrow*: these are exactly the layouts a CPU-origin upload or
/// an in-place guest gather can hand a sampled image without a conversion pass.
/// A rail that carries the full format set names the host format instead — the
/// engine stores `VkFormat` and can therefore express an sRGB sampled view,
/// which a layout enum by construction cannot.
///
/// It lives in the contract rather than in either the runtime or the backend
/// because both used to hold their own copy of it, with a hand-written mapping
/// between the two that nothing checked.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TexelLayout {
    /// 4 bytes/texel, guest channel order R,G,B,A — the default CPU-origin
    /// layout, and what every convert-to-RGBA8 loader produces.
    Rgba8,
    /// 4 bytes/texel, guest channel order B,G,R,A. Uploaded as-is so the
    /// sampler swaps channels in hardware and the CPU never runs a per-pixel
    /// swizzle.
    Bgra8,
    /// 1 byte/texel — a biplanar video luma plane, sampled at its native
    /// footprint rather than expanded to RGBA8.
    R8,
    /// 2 bytes/texel — a biplanar video chroma plane, likewise native.
    Rg8,
    /// 2 bytes/texel — a single-channel `float16` texture, sampled natively as
    /// `R16_SFLOAT` (the shader reads `.x`, the other lanes expand to `0,0,1`).
    /// Color-management 1D LUTs (macOS WindowServer's `UberCompositeFragment`
    /// display-profile pass) are stored this way; converting them to unorm8
    /// would quantize the transfer curve, and the CPU `texel_to_rgba8` loader
    /// has no float arm, so this native rail is the only correct path. Not a
    /// four-byte color layout, so it never rides the RGBA8-shaped loaders.
    R16Float,
    /// 4 bytes/texel — a single-channel `float32` texture, sampled natively as
    /// `R32_SFLOAT`. Same color-LUT role as [`Self::R16Float`], but its
    /// linear-filter feature is optional (absent on Apple/MoltenVK), so the
    /// rail that emits this layout must first confirm the host supports it.
    /// Four bytes wide but **not** a colour order, so it stays out of the
    /// RGBA8-shaped loaders and `is_four_byte_color`.
    R32Float,
    /// 2 bytes/texel — a **ten-bit** biplanar video luma plane, sampled natively
    /// as `R16_UNORM`.
    ///
    /// The same role as [`Self::R8`] one bit depth up. A `'x420'` surface is
    /// `kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange`, and its planes are
    /// `MTLPixelFormatR16Unorm` and `MTLPixelFormatRG16Unorm` — not the `R8`/`RG8`
    /// of the eight-bit `'420v'` shape.
    ///
    /// Native rather than converted, for [`Self::R16Float`]'s reason: narrowing
    /// to unorm8 would drop two bits of luma from content graded for them, and a
    /// banded frame is a wrong frame this device chose rather than one the guest
    /// asked for. Two bytes wide and not a colour order, so like the float
    /// layouts it stays out of the RGBA8-shaped loaders.
    R16Unorm,
    /// 4 bytes/texel — the chroma half of [`Self::R16Unorm`], sampled natively as
    /// `R16G16_UNORM`. Four bytes wide but **not** a colour order.
    Rg16Unorm,
    /// 16 bytes/texel, four `float32` channels, sampled natively as
    /// `R32G32B32A32_SFLOAT`.
    ///
    /// A macos-15 guest binds tiny linear textures of this format to a **vertex**
    /// sampler — 1x1 and 4x1, `bytesPerRow` exactly `width * 16` — and the whole
    /// draw was refused because this crate had no layout for it.
    ///
    /// It has **no CPU loader arm and must not gain one**, and that is a
    /// stronger statement than the half-float layouts' lossiness. Narrowing an
    /// `f32` texel to unorm8 clamps to `[0,1]` and quantises to 256 levels, and
    /// [`narrows_to_unorm8`]'s own doc says what that costs a texture whose
    /// texels are not colours: "a lookup table, a coordinate pair, a chain of
    /// offsets — it is data loss with no upper bound on the consequence, and it
    /// is silent, because the conversion succeeds". A 1x1 four-channel float
    /// sampled in a vertex shader is that texture. So the native bind is the
    /// only rail, and a host that cannot filter the format declines by name
    /// rather than quantising the guest's numbers.
    ///
    /// The native bind is capability-gated for a real reason: Vulkan mandates
    /// `SAMPLED_IMAGE` for `R32G32B32A32_SFLOAT` but **not**
    /// `SAMPLED_IMAGE_FILTER_LINEAR`, so the filter is measured from the device
    /// through the sampled-filter mask rather than assumed.
    Rgba32Float,
    /// 4 bytes/texel, two `uint16` channels, sampled natively as `R16G16_UINT`.
    ///
    /// The same bytes as [`Self::Rg16Unorm`] and a different reading of them,
    /// which is why it is a layout of its own rather than a spelling of that
    /// one: the two disagree about what a texel *means* everywhere a value
    /// crosses between integer and normalized, and the map to a backend format
    /// is total, so a shared variant would have to pick one and be wrong for
    /// the other.
    ///
    /// It has no [`rgba8_to_texel`] arm and must not gain one. An unorm8
    /// intermediate cannot express an integer texel — a clear colour of `1.0`
    /// is the integer `1` here, not `255` and not `65535` — so every rail that
    /// funnels through eight-bit colour has to decline for this layout rather
    /// than convert. The byte copy is exact and is the rail that serves it.
    Rg16Uint,
    /// 8 bytes/texel, four `float16` channels in R,G,B,A order, sampled
    /// natively as `R16G16B16A16_SFLOAT`.
    ///
    /// This is what a recent macOS window server composites in. It is the only
    /// layout here **wider** than four bytes, and the only one whose CPU
    /// alternative is lossy in a way the guest can see rather than merely slow:
    /// [`texel_to_rgba8`] does carry an arm for it, through
    /// `f16_to_unorm8_lut`, and that arm quantizes the channel to 256 levels
    /// and clamps everything above 1.0. A compositor working in extended range
    /// puts values above 1.0 there on purpose, so the conversion is not a
    /// rounding difference — it is the highlight range removed.
    ///
    /// Sampling the guest's own bytes is exact: `MTLPixelFormatRGBA16Float` and
    /// `VK_FORMAT_R16G16B16A16_SFLOAT` are both four little-endian IEEE binary16
    /// channels in that order, so the texel is byte-identical and no conversion
    /// exists to be lossy.
    Rgba16Float,
    /// 4 bytes/texel, two `float16` channels, sampled natively as
    /// `R16G16_SFLOAT`. The two-channel companion to [`Self::Rgba16Float`],
    /// with the same exactness argument and the same lossy CPU arm. Four bytes
    /// wide and **not** a colour order.
    Rg16Float,
    /// 8 bytes/texel, four sixteen-bit normalized channels in R,G,B,A order,
    /// sampled natively as `R16G16B16A16_UNORM`.
    ///
    /// Native for [`Self::R16Unorm`]'s reason one channel count up: narrowing
    /// sixteen bits of colour to eight would band content graded for them, and
    /// [`texel_to_rgba8`] carries no arm that could do it anyway. Eight bytes
    /// wide, so like [`Self::Rgba16Float`] it stays out of the four-byte colour
    /// rails despite being a colour order.
    Rgba16Unorm,
    /// 4 bytes/texel — ten bits per colour channel and two of alpha in one
    /// packed word, red in the low bits, sampled natively as
    /// `A2B10G10R10_UNORM_PACK32`.
    ///
    /// `MTLPixelFormatRGB10A2Unorm` and that Vulkan format are the same bits in
    /// the same order, so the guest's word is sampled unchanged. Native for
    /// [`Self::R16Unorm`]'s reason two bits down: an unorm8 conversion would
    /// discard the resolution the guest picked the format for, and
    /// [`texel_to_rgba8`] has no arm that could cut a non-byte channel boundary
    /// anyway. Four bytes wide and **not** a byte-order colour layout, so it
    /// stays out of the RGBA8-shaped loaders and `is_four_byte_color`.
    Rgb10a2Unorm,
    /// 4 bytes/texel — [`Self::Rgb10a2Unorm`] with the colour channels the
    /// other way round in the word, sampled natively as
    /// `A2R10G10B10_UNORM_PACK32`. The same relation `Bgra8` has to `Rgba8`,
    /// one storage shape up.
    Bgr10a2Unorm,
    /// 4 bytes/texel — eleven bits of red and green, ten of blue, no alpha,
    /// sampled natively as `B10G11R11_UFLOAT_PACK32`. An HDR-intermediate
    /// colour format, native for the same reason as its two neighbours.
    Rg11b10Float,
    // The BC block-compressed layouts. Every one is 4x4 texels in 8 or 16 bytes,
    // so `bytes_per_texel` answers about a **block** for these and
    // `block_geometry` is what a caller sizing an image must ask — read that
    // method's doc before using either on one of these.
    //
    // They are the only layouts here that no CPU rail can serve: decoding a
    // block needs a decompressor this crate does not have and does not want, so
    // `has_cpu_loader_arm` is false, the two `rgba8` conversions refuse, and the
    // bind is native or nothing. Nothing is a typed refusal, which is why the
    // capability gate lives at `translate::pixel::sampled_pixels`.
    //
    // The sRGB spellings of BC1/BC2/BC3/BC7 fold onto these same layouts, as
    // `Rgba8`'s sRGB spelling folds onto `Rgba8`: the stored bytes are
    // identical and the qualifier is a sampler conversion carried by
    // `SampledByteFormat`'s source format.
    /// 8 bytes per 4x4 block — BC1/DXT1, three colour channels plus one bit of
    /// alpha, sampled as `VK_FORMAT_BC1_RGBA_UNORM_BLOCK`.
    Bc1Rgba,
    /// 16 bytes per 4x4 block — BC2/DXT3, four-bit explicit alpha.
    Bc2Rgba,
    /// 16 bytes per 4x4 block — BC3/DXT5, interpolated alpha. The member a guest
    /// was measured binding; see [`MTL_FORMAT_BC3_RGBA`].
    Bc3Rgba,
    /// 8 bytes per 4x4 block — BC4, one unsigned channel.
    Bc4RUnorm,
    /// 8 bytes per 4x4 block — BC4, one signed channel.
    Bc4RSnorm,
    /// 16 bytes per 4x4 block — BC5, two unsigned channels.
    Bc5RgUnorm,
    /// 16 bytes per 4x4 block — BC5, two signed channels.
    Bc5RgSnorm,
    /// 16 bytes per 4x4 block — BC6H, three signed half-float channels (HDR).
    Bc6hRgbFloat,
    /// 16 bytes per 4x4 block — BC6H, three unsigned half-float channels.
    Bc6hRgbUfloat,
    /// 16 bytes per 4x4 block — BC7, four channels, mode-switching.
    Bc7Rgba,
}

/// Which of the two per-layout capability masks a question belongs to.
///
/// Carried as a type rather than as the `bool` this started out as, for the
/// reason `AGENTS.md` gives for every selector this crate owns: a `bool`
/// parameter named for one of its two values reads correctly at the definition
/// and ambiguously at every call, and adding a third mask would silently widen
/// one arm of an `if` instead of failing the build in the `match` below.
///
/// The two spans are deliberately different — see
/// [`TexelLayout::is_render_target_layout`] and
/// [`TexelLayout::needs_sampled_filter_query`] for what each asks and why
/// sharing one index space cost the bit budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CapabilityMask {
    /// Layouts this device creates a colour attachment at.
    RenderTarget,
    /// Layouts whose sampled bind must ask the host about linear filtering.
    SampledFilter,
}

impl TexelLayout {
    /// Every layout, so a sweep over the vocabulary is derived rather than
    /// hand-listed.
    ///
    /// Held honest by `every_texel_layout_is_in_the_all_list_exactly_once`. The
    /// tests below used to iterate hand-written arrays whose inner `match` was
    /// exhaustive, which catches a variant added without an answer but not one
    /// added without being *swept* — the same half-check `translate::reason`
    /// found in its own `ALL` list.
    pub const ALL: &'static [Self] = &[
        Self::Rgba8,
        Self::Bgra8,
        Self::R8,
        Self::Rg8,
        Self::R16Float,
        Self::R32Float,
        Self::R16Unorm,
        Self::Rg16Unorm,
        Self::Rg16Uint,
        Self::Rgba32Float,
        Self::Rgba16Float,
        Self::Rg16Float,
        Self::Rgba16Unorm,
        Self::Rgb10a2Unorm,
        Self::Bgr10a2Unorm,
        Self::Rg11b10Float,
        Self::Bc1Rgba,
        Self::Bc2Rgba,
        Self::Bc3Rgba,
        Self::Bc4RUnorm,
        Self::Bc4RSnorm,
        Self::Bc5RgUnorm,
        Self::Bc5RgSnorm,
        Self::Bc6hRgbFloat,
        Self::Bc6hRgbUfloat,
        Self::Bc7Rgba,
    ];

    /// This layout's position in [`Self::ALL`], so a host-side table can be an
    /// array sized by `ALL.len()` rather than a map or a hand-widened bitmask.
    ///
    /// Exhaustive on purpose: a new variant cannot reach such a table without
    /// being given a position here, and the array it indexes cannot be too
    /// narrow for it because its length is `ALL.len()`. That is the shape
    /// `AGENTS.md` asks for over `mask |= 1 << index`, which bounds a set to 32
    /// with nothing declared and wraps rather than failing.
    pub fn index(self) -> usize {
        match self {
            Self::Rgba8 => 0,
            Self::Bgra8 => 1,
            Self::R8 => 2,
            Self::Rg8 => 3,
            Self::R16Float => 4,
            Self::R32Float => 5,
            Self::R16Unorm => 6,
            Self::Rg16Unorm => 7,
            Self::Rg16Uint => 8,
            Self::Rgba32Float => 9,
            Self::Rgba16Float => 10,
            Self::Rg16Float => 11,
            Self::Rgba16Unorm => 12,
            Self::Rgb10a2Unorm => 13,
            Self::Bgr10a2Unorm => 14,
            Self::Rg11b10Float => 15,
            Self::Bc1Rgba => 16,
            Self::Bc2Rgba => 17,
            Self::Bc3Rgba => 18,
            Self::Bc4RUnorm => 19,
            Self::Bc4RSnorm => 20,
            Self::Bc5RgUnorm => 21,
            Self::Bc5RgSnorm => 22,
            Self::Bc6hRgbFloat => 23,
            Self::Bc6hRgbUfloat => 24,
            Self::Bc7Rgba => 25,
        }
    }

    /// The texel grid one unit of this layout's storage covers.
    ///
    /// 1x1 for every uncompressed layout, 4x4 for the BC families. A caller
    /// sizing an image or striding a row must ask this rather than
    /// [`Self::bytes_per_texel`], which for a BC layout answers about a block
    /// and would under-count a row by four and an image by sixteen.
    pub fn block(self) -> BlockGeometry {
        let side = if self.is_block_compressed() {
            BC_BLOCK_SIDE
        } else {
            1
        };
        BlockGeometry {
            width: side,
            height: side,
            bytes: self.bytes_per_texel(),
        }
    }

    /// Whether this layout's channels are read as integers rather than as a
    /// value in a continuous range.
    ///
    /// The distinction the eight-bit conversion rails and the capability masks
    /// both turn on: an integer texel is a count, so there is no unorm8 byte
    /// that stands for it and no meaning to interpolating between two of them.
    /// It is also what separates the two questions a colour attachment is
    /// asked — Vulkan mandates no `COLOR_ATTACHMENT_BLEND` on an integer
    /// format, so requiring blend of one would refuse every host, while
    /// `COLOR_ATTACHMENT` itself is a real and answerable question. See
    /// `caps::device_features`'s per-layout colour-attachment probe.
    pub const fn is_integer(self) -> bool {
        matches!(self, Self::Rg16Uint)
    }

    /// Whether this layout stores a 4x4 block per addressable unit.
    ///
    /// Exhaustive rather than a range test on [`Self::index`]: the positions are
    /// an implementation detail of the table above and a new uncompressed layout
    /// appended after the BC block would silently join the compressed set.
    pub const fn is_block_compressed(self) -> bool {
        match self {
            Self::Rgba8
            | Self::Bgra8
            | Self::R8
            | Self::Rg8
            | Self::R16Float
            | Self::R32Float
            | Self::R16Unorm
            | Self::Rg16Unorm
            | Self::Rg16Uint
            | Self::Rgba32Float
            | Self::Rgba16Float
            | Self::Rg16Float
            | Self::Rgba16Unorm
            | Self::Rgb10a2Unorm
            | Self::Bgr10a2Unorm
            | Self::Rg11b10Float => false,
            Self::Bc1Rgba
            | Self::Bc2Rgba
            | Self::Bc3Rgba
            | Self::Bc4RUnorm
            | Self::Bc4RSnorm
            | Self::Bc5RgUnorm
            | Self::Bc5RgSnorm
            | Self::Bc6hRgbFloat
            | Self::Bc6hRgbUfloat
            | Self::Bc7Rgba => true,
        }
    }

    /// Whether this layout is one this device will render into.
    ///
    /// The layout-side spelling of [`render_target_bpp`]'s admission set. It is
    /// a `const fn` over the layouts rather than a second list of formats, and
    /// `the_two_render_target_vocabularies_name_the_same_layouts` holds it
    /// against that function so the two cannot drift.
    ///
    /// It is also one of the two per-layout capability vocabularies, and
    /// deliberately not the same one as [`Self::needs_sampled_filter_query`].
    /// The two masks answer different questions about different sets and used
    /// to share one index space, which sized both by the union and wasted every
    /// bit where they disagree — a `u64` published lock-free has none to spare.
    ///
    /// **The set is every layout this device renders into, integer ones
    /// included, and that is load-bearing.** This vocabulary was briefly
    /// narrowed to the *blendable* layouts, which excluded the integer one —
    /// so `engine::render_target_layout_supported` had no bit to read for
    /// `Rg16Uint`, answered `false`, and every `RG16Uint` render target was
    /// built at the neutral eight-bit format instead of its own. Both Store
    /// arms then refused it: the GPU-direct one on `ResidentFormatMismatch`
    /// and the copying one in the row converter, which has no integer arm by
    /// design. A mask whose vocabulary is narrower than the question its reader
    /// asks does not decline, it answers wrongly.
    pub const fn is_render_target_layout(self) -> bool {
        matches!(
            self,
            Self::Rgba8
                | Self::Bgra8
                | Self::R8
                | Self::R16Float
                | Self::Rg16Float
                | Self::Rgba16Float
                | Self::Bgr10a2Unorm
                | Self::Rg16Uint
        )
    }

    /// Whether a sampled bind of this layout must ask the host about linear
    /// filtering.
    ///
    /// The other capability vocabulary. Two layouts are excluded and neither is
    /// an omission:
    ///
    /// - **Block-compressed.** Vulkan's mandatory-format table *requires*
    ///   `SAMPLED_IMAGE_FILTER_LINEAR` of every BC format on a device that
    ///   enables `textureCompressionBC`, and that feature is what admits the
    ///   format in the first place. There is nothing to query.
    /// - **Integer.** Vulkan permits no `VK_FILTER_LINEAR` on an integer
    ///   format, so the answer is statically "no".
    pub const fn needs_sampled_filter_query(self) -> bool {
        !self.is_block_compressed() && !self.is_integer()
    }

    /// How many layouts each mask spans, counted from [`Self::ALL`] so neither
    /// can fall behind the vocabulary.
    pub const RENDER_TARGET_COUNT: usize = Self::count_where(CapabilityMask::RenderTarget);
    /// See [`Self::RENDER_TARGET_COUNT`].
    pub const FILTER_COUNT: usize = Self::count_where(CapabilityMask::SampledFilter);

    /// Whether `self` occupies a bit of `mask`.
    const fn in_mask(self, mask: CapabilityMask) -> bool {
        match mask {
            CapabilityMask::RenderTarget => self.is_render_target_layout(),
            CapabilityMask::SampledFilter => self.needs_sampled_filter_query(),
        }
    }

    const fn count_where(mask: CapabilityMask) -> usize {
        let mut count = 0;
        let mut i = 0;
        while i < Self::ALL.len() {
            if Self::ALL[i].in_mask(mask) {
                count += 1;
            }
            i += 1;
        }
        count
    }

    /// This layout's bit in the render-target mask, or `None` when the mask has
    /// no question about it.
    pub fn render_target_index(self) -> Option<usize> {
        self.index_where(CapabilityMask::RenderTarget)
    }

    /// This layout's bit in the sampled-filter mask, or `None` when the mask
    /// has no question about it.
    pub fn filter_index(self) -> Option<usize> {
        self.index_where(CapabilityMask::SampledFilter)
    }

    /// Walks [`Self::ALL`] rather than keeping a second ordering: one list, one
    /// order, nothing to drift.
    fn index_where(self, mask: CapabilityMask) -> Option<usize> {
        if !self.in_mask(mask) {
            return None;
        }
        let mut index = 0;
        for layout in Self::ALL {
            if *layout == self {
                return Some(index);
            }
            if layout.in_mask(mask) {
                index += 1;
            }
        }
        None
    }

    /// Bytes one tightly-packed row of `width` texels of this layout occupies.
    ///
    /// One row of **blocks** for a compressed layout, so a caller comparing a
    /// guest row stride against "one tight row of the upload layout" gets the
    /// right answer for both families from one expression. `None` on overflow.
    pub fn tight_row_bytes(self, width: u32) -> Option<u32> {
        let block = self.block();
        block.blocks_across(width).checked_mul(block.bytes)
    }

    /// Rows a copy of a `height`-tall image of this layout must walk — a quarter
    /// of the texel height, rounded up, for a compressed layout.
    pub fn tight_row_count(self, height: u32) -> u32 {
        self.block().block_rows(height)
    }

    /// Bytes a whole `width` x `height` image of this layout occupies, tightly
    /// packed.
    ///
    /// The one sizing expression for both families. `None` on overflow, so a
    /// caller declines rather than allocating a wrapped length.
    pub fn image_bytes(self, width: u32, height: u32) -> Option<u64> {
        let block = self.block();
        let across = u64::from(block.blocks_across(width));
        let down = u64::from(block.block_rows(height));
        across
            .checked_mul(down)?
            .checked_mul(u64::from(block.bytes))
    }

    /// Bytes occupied by one texel in guest linear storage — or by one **4x4
    /// block**, for the BC layouts.
    ///
    /// The name is kept because ninety-odd call sites use it and every one of
    /// them is on a rail a BC layout cannot reach: a BC format has no
    /// [`render_target_bpp`], no [`storage_selector`], no [`sampled_class`] and
    /// no [`bytes_per_pixel`], so it is refused before any of them.
    /// `a_bc_format_is_refused_by_every_rail_but_the_sampled_bind` is that
    /// argument as a test rather than as this paragraph.
    ///
    /// For anything that sizes storage, ask [`Self::image_bytes`] or
    /// [`Self::block`]. Multiplying this by width and height is correct for the
    /// uncompressed layouts and wrong by sixteen for the BC ones.
    pub fn bytes_per_texel(self) -> u32 {
        match self {
            Self::Rgba8 | Self::Bgra8 => RGBA8_BPP,
            Self::R8 => R8_BPP,
            Self::Rg8 => RG8_BPP,
            Self::R16Float => R16F_BPP,
            Self::R32Float => R32F_BPP,
            Self::R16Unorm => R16_BPP,
            Self::Rg16Unorm | Self::Rg16Uint => RG16_BPP,
            Self::Rgba32Float => RGBA32_BPP,
            Self::Rgba16Float => RGBA16F_BPP,
            Self::Rg16Float => RG16F_BPP,
            Self::Bc1Rgba => BC_BLOCK_BYTES_8,
            Self::Bc2Rgba => BC_BLOCK_BYTES_16,
            Self::Bc3Rgba => BC_BLOCK_BYTES_16,
            Self::Bc4RUnorm => BC_BLOCK_BYTES_8,
            Self::Bc4RSnorm => BC_BLOCK_BYTES_8,
            Self::Bc5RgUnorm => BC_BLOCK_BYTES_16,
            Self::Bc5RgSnorm => BC_BLOCK_BYTES_16,
            Self::Bc6hRgbFloat => BC_BLOCK_BYTES_16,
            Self::Bc6hRgbUfloat => BC_BLOCK_BYTES_16,
            Self::Bc7Rgba => BC_BLOCK_BYTES_16,

            Self::Rgba16Unorm => RGBA16_BPP,
            Self::Rgb10a2Unorm | Self::Bgr10a2Unorm | Self::Rg11b10Float => RGBA8_BPP,
        }
    }

    /// Whether this layout is one of the two four-byte colour orders.
    ///
    /// Several rails admit only these: the RGBA8-shaped diagnostics, the
    /// tight-row loaders and the zero-copy gathers all assume a four-byte
    /// texel. Named once so a rail states which set it takes instead of
    /// re-listing the variants.
    pub fn is_four_byte_color(self) -> bool {
        matches!(self, Self::Rgba8 | Self::Bgra8)
    }

    /// Whether [`texel_to_rgba8`] carries an arm for this layout, so a rail
    /// that declines has a CPU path to decline *to*.
    ///
    /// This is the question a performance threshold is really asking. A floor
    /// that turns a small window away from a GPU gather is only a cost
    /// decision when the CPU loader can serve it instead; for a layout with no
    /// arm the same floor is a correctness gate wearing a threshold's clothes,
    /// and the sample goes black or fail-visible rather than slow. The two
    /// float layouts are colour-management LUTs, whose transfer curve unorm8
    /// would quantize — which is why `texel_to_rgba8` deliberately has no arm
    /// for them and why they must bypass any such floor.
    ///
    /// Spelled here rather than at the floors, because it is a property of the
    /// layout and the loader, and a rail that re-lists the variants drifts the
    /// first time one is added. It is deliberately **not** `is_four_byte_color`
    /// even though the two agreed for as long as only four-byte colour reached
    /// a floor: they answer different questions and diverge on `R8`/`Rg8`,
    /// which have arms and are not four bytes.
    pub fn has_cpu_loader_arm(self) -> bool {
        match self {
            Self::Rgba8 | Self::Bgra8 | Self::R8 | Self::Rg8 => true,
            // The two half-float colour layouts answer `true` because the arm
            // genuinely exists — `texel_to_rgba8` converts both through
            // `f16_to_unorm8_lut`. This is a statement about the loader, not an
            // endorsement, and [`Self::cpu_loader_arm_is_lossy`] is where the
            // endorsement is withheld.
            Self::Rgba16Float | Self::Rg16Float => true,
            // The two sixteen-bit normalized layouts join the floats here for
            // the same reason and a different quantity: `texel_to_rgba8` has no
            // arm for them because an arm would have to narrow ten bits of video
            // luma to eight.
            //
            // The three packed 32-bit colour layouts answer `false` for a third
            // reason: their channels do not sit on byte boundaries at all, so
            // there is nothing for a byte-shaped loader to pick up.
            //
            // `Rg16Uint` answers `false` for a fourth reason, and it is the
            // only one of the four that is about meaning rather than width: an
            // eight-bit loader would have to decide what integer a unorm8 byte
            // stands for, and there is no answer — the guest's texel is a count,
            // not a fraction of full scale.
            Self::R16Float
            | Self::R32Float
            | Self::R16Unorm
            | Self::Rg16Unorm
            | Self::Rg16Uint
            | Self::Rgba32Float
            | Self::Rgba16Unorm
            | Self::Rgb10a2Unorm
            | Self::Bgr10a2Unorm
            | Self::Rg11b10Float => false,
            // No CPU rail can serve a BC layout, and none should be written:
            // decoding a block needs a decompressor, and a decompressed block
            // is not the guest's bytes. `false` here is what keeps a cost
            // threshold from routing one to a loader that does not exist — the
            // native bind or a typed refusal are the only two answers.
            Self::Bc1Rgba
            | Self::Bc2Rgba
            | Self::Bc3Rgba
            | Self::Bc4RUnorm
            | Self::Bc4RSnorm
            | Self::Bc5RgUnorm
            | Self::Bc5RgSnorm
            | Self::Bc6hRgbFloat
            | Self::Bc6hRgbUfloat
            | Self::Bc7Rgba => false,
        }
    }

    /// Whether [`texel_to_rgba8`]'s arm for this layout loses information the
    /// guest can see.
    ///
    /// An arm existing and an arm being *equivalent* are two different facts,
    /// and conflating them is how a cost threshold becomes a data-loss gate.
    /// The half-float colour arms go through `f16_to_unorm8_lut`: every channel
    /// is clamped to `[0, 1]` and quantized to 256 levels, so a compositor
    /// working in extended range has its highlights removed and a colour ramp
    /// banded. That is not a slower way to the same pixels; it is different
    /// pixels.
    ///
    /// Every other arm is exact — `Rgba8` is an identity copy, `Bgra8` a
    /// channel swap, `R8`/`Rg8` a zero-extend that the matching one- and
    /// two-channel Vulkan formats sample to identically.
    pub fn cpu_loader_arm_is_lossy(self) -> bool {
        match self {
            Self::Rgba16Float | Self::Rg16Float => true,
            Self::Rgba8
            | Self::Bgra8
            | Self::R8
            | Self::Rg8
            | Self::R16Float
            | Self::R32Float
            | Self::R16Unorm
            | Self::Rg16Unorm
            | Self::Rg16Uint
            | Self::Rgba32Float
            | Self::Rgba16Unorm
            | Self::Rgb10a2Unorm
            | Self::Bgr10a2Unorm
            | Self::Rg11b10Float => false,
            // Vacuously false: there is no arm, so no arm loses anything.
            // `has_cpu_loader_arm` is the question a caller should be asking
            // about these, and it answers `false`.
            Self::Bc1Rgba
            | Self::Bc2Rgba
            | Self::Bc3Rgba
            | Self::Bc4RUnorm
            | Self::Bc4RSnorm
            | Self::Bc5RgUnorm
            | Self::Bc5RgSnorm
            | Self::Bc6hRgbFloat
            | Self::Bc6hRgbUfloat
            | Self::Bc7Rgba => false,
        }
    }

    /// Whether a **cost** threshold may turn this layout away from a GPU rail
    /// onto the CPU byte loader.
    ///
    /// This is the question the zero-copy floors are really asking, and it is
    /// derived from the two above rather than re-listed, so a layout added with
    /// a lossy arm cannot be waved past by a floor that only checked whether an
    /// arm existed. A floor is a performance decision exactly when the path it
    /// declines to produces the same pixels; where the only CPU arm is lossy —
    /// or absent — the same floor is a correctness gate wearing a threshold's
    /// clothes.
    ///
    /// The half-float colour layouts are why this exists. They answered
    /// `has_cpu_loader_arm()` truthfully and were therefore turned away by the
    /// 64 KiB sampled floor, which is above a 64x64 `RGBA16Float` texture's
    /// 32 KiB — so a colour-management LUT the guest stored in extended range
    /// reached the shader clamped and quantized, every boot, reported by
    /// `sampled_texture_narrowed` and by nothing else.
    pub fn a_cost_floor_may_decline(self) -> bool {
        self.has_cpu_loader_arm() && !self.cpu_loader_arm_is_lossy()
    }

    /// Whether an sRGB-encoded image can be *stored* in this layout, so a host
    /// sampled view of it can carry the transfer function.
    ///
    /// sRGB is defined on eight-bit unsigned normalized colour channels and
    /// nowhere else, which is also why [`is_srgb`] names exactly two Metal
    /// formats. A layout that answers `false` here cannot hold an sRGB image at
    /// all, so a [`SampledByteFormat`] pairing one with an sRGB source is a
    /// loader that converted the values *out* of the encoding's domain — the
    /// one case the fold has to report rather than honour.
    ///
    /// Deliberately equal to [`Self::is_four_byte_color`] rather than written as
    /// it: the two agree today because the eight-bit colour orders are both the
    /// four-byte ones, and they answer different questions. A three-byte
    /// `RGB8_SRGB` layout would separate them.
    pub fn has_srgb_encoding(self) -> bool {
        matches!(
            self,
            Self::Rgba8
                | Self::Bgra8
                | Self::Bc1Rgba
                | Self::Bc2Rgba
                | Self::Bc3Rgba
                | Self::Bc7Rgba
        )
    }
}

/// What a CPU loader produced for a sampled bind: the channel layout the bytes
/// are in, and the guest format whose transfer function they still carry.
///
/// # Why this is two facts and not one
///
/// [`TexelLayout`] is linear by construction — it names a channel order and a
/// width, and nothing in it can say that the stored values are sRGB-encoded.
/// For as long as the sampled byte rails carried a bare layout, every CPU
/// upload of an `MTLPixelFormatBGRA8Unorm_sRGB` texture was bound through a
/// `_UNORM` view: the hardware never decoded, and the next sRGB attachment
/// write encoded values that had never been decoded. Meanwhile the *zero-copy*
/// rails, which carry a resolved host format, bound the `_SRGB` spelling and
/// decoded correctly — so one guest texture got two different colours depending
/// on which rail won, and which rail wins is a cost decision.
///
/// The two axes are genuinely independent: a loader may reorder channels
/// (BGRA to RGBA) without touching the transfer function, so the layout it
/// wrote and the encoding it preserved are separate answers and pairing them is
/// not a flag threaded past a resolver. The source format is kept rather than
/// boiled down to a `bool` so the fail log can name what was traded away when
/// the fold cannot be honoured.
///
/// # Construct it where the source format is known
///
/// [`Self::from_source`] is the only way to say "these bytes came from a guest
/// texture", and it takes that texture's `MTLPixelFormat`. A loader that has to
/// reach for [`Self::synthesised`] is saying there is no guest format behind
/// the values at all — a solid clear colour, for instance, which the guest
/// specified in the attachment's decoded space. Choosing `synthesised` for
/// bytes that *do* have a source format is the same silent loss this type
/// exists to end.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SampledByteFormat {
    layout: TexelLayout,
    /// The guest format the values were loaded from, when there is one. Held
    /// privately because the layout above is what the bytes are *in*: deriving
    /// one of these from the other is exactly the confusion this type ends.
    source: Option<u16>,
}

impl SampledByteFormat {
    /// Bytes a loader read out of a guest texture declared as `source`.
    ///
    /// `layout` is what the loader wrote, which may differ from `source`'s own
    /// order; the transfer function is `source`'s and survives any reordering.
    pub fn from_source(layout: TexelLayout, source: u16) -> Self {
        Self {
            layout,
            source: Some(source),
        }
    }

    /// Bytes this device built itself, with no guest texture behind them.
    ///
    /// Linear by construction and by contract, not by omission: the values are
    /// whatever this device computed, in the space the guest named them in.
    pub const fn synthesised(layout: TexelLayout) -> Self {
        Self {
            layout,
            source: None,
        }
    }

    /// The channel order and width the bytes are in.
    pub const fn layout(self) -> TexelLayout {
        self.layout
    }

    /// The guest format these values are sRGB-encoded by, or `None` when they
    /// are linear.
    ///
    /// The format, not a `bool`, because the only caller that acts on a `Some`
    /// either honours it — where the answer is the layout's sRGB spelling — or
    /// reports it, where naming the guest format is the whole value of the line.
    pub fn srgb_source(self) -> Option<u16> {
        self.source.filter(|&source| is_srgb(source))
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SwizzleSource {
    Zero = 0,
    One = 1,
    R = 2,
    G = 3,
    B = 4,
    A = 5,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SwizzlePlan {
    pub source: [SwizzleSource; COMPONENT_COUNT],
}

impl Default for SwizzlePlan {
    fn default() -> Self {
        swizzle_identity()
    }
}

impl SwizzlePlan {
    /// Whether this plan moves nothing — every output channel takes its own
    /// input.
    ///
    /// Derived from [`swizzle_identity`] rather than spelling the four sources a
    /// second time, so there is nothing for a second spelling to disagree with.
    /// The question is asked wherever a rail can carry channels one way but not
    /// the other: a hardware component mapping expresses any plan, and a view
    /// this device did not create expresses only this one.
    pub fn is_identity(&self) -> bool {
        *self == swizzle_identity()
    }

    /// This plan applied **after** `inner`, as one plan.
    ///
    /// Two remaps stack whenever a texture's own channel layout does not sit
    /// identically on the host format carrying it *and* the guest asked for a
    /// view swizzle on top. `A8Unorm` is the standing case: its byte rides in
    /// `R8_UNORM`, so the format contributes "alpha is in red" and the guest's
    /// type-8 view contributes whatever it asked for. Binding either alone
    /// gives the shader the wrong channels.
    ///
    /// A hardware component mapping takes one plan, not two, so they have to be
    /// folded here. The fold is a lookup, because a plan is a function from
    /// output channel to input channel: this plan names a channel of `inner`'s
    /// *output*, and `inner` says where that came from. A constant selector has
    /// no channel to chase and passes through.
    ///
    /// Composition is not commutative and the argument order is the one that
    /// reads: `view.after(&format)`.
    pub fn after(self, inner: &SwizzlePlan) -> SwizzlePlan {
        let mut source = self.source;
        for slot in &mut source {
            *slot = match *slot {
                SwizzleSource::Zero => SwizzleSource::Zero,
                SwizzleSource::One => SwizzleSource::One,
                SwizzleSource::R => inner.source[COMPONENT_R],
                SwizzleSource::G => inner.source[COMPONENT_G],
                SwizzleSource::B => inner.source[COMPONENT_B],
                SwizzleSource::A => inner.source[COMPONENT_A],
            };
        }
        SwizzlePlan { source }
    }
}

const UNORM8_MIN: u8 = 0x00;
const UNORM8_MAX: u8 = 0xff;

const F16_SIGN_MASK: u16 = 0x8000;
const F16_EXP_SHIFT: u32 = 10;
const F16_EXP_MASK: u32 = 0x1f;
const F16_MANT_MASK: u32 = 0x03ff;
const F16_HIDDEN_BIT: u32 = 0x0400;
const F16_EXP_BIAS: i32 = 15;
const F16_EXP_INF_NAN: u32 = F16_EXP_MASK;
const F16_INF_BITS: u16 = 0x7c00;
const F16_SUBNORMAL_EXP_MIN: i32 = -10;
const F16_SUBNORMAL_SHIFT_BASE: i32 = 14;
const F16_F32_SIGN_SHIFT: u32 = 16;
const F32_EXP_SHIFT: u32 = 23;
const F32_EXP_MASK: u32 = 0xff;
const F32_MANT_MASK: u32 = 0x007f_ffff;
const F32_HIDDEN_BIT: u32 = 0x0080_0000;
const F32_EXP_BIAS: i32 = 127;
const F32_INF_BITS: u32 = 0x7f80_0000;
const F16_F32_MANT_SHIFT: u32 = 13;
const F32_TO_F16_ROUND_BIT: u32 = 0x1000;

/// Bytes one texel of `format` occupies in guest linear storage, or `None` for
/// a format this contract does not define.
///
/// # The `X*_Stencil8` arms are the parent cell, deliberately
///
/// `X32_Stencil8` and `X24_Stencil8` are **stencil-aspect views of a combined
/// depth-stencil texture**, not formats with storage of their own. This answers
/// the size of the cell they view — 8 and 4 — because that is what every caller
/// of this function needs: `crate::backend::vulkan::translate::pixel` binds
/// them to `D32_SFLOAT_S8_UINT` and `D24_UNORM_S8_UINT`, whose texels are
/// exactly those widths, and a resource sized at anything else is a short
/// allocation.
///
/// An external per-format table will say **1** for both, and it is not wrong —
/// it is describing the stencil plane alone, which is a different question. This
/// crate answers that one too, as [`depth_stencil_packing`]'s
/// `stencil_plane_bpp`, and it is 1 there. **Do not reconcile the two.** They are
/// two numbers for two purposes and we hold both on purpose; moving this arm to
/// 1 to agree with a vendor table would under-allocate every stencil-aspect
/// parent.
pub fn bytes_per_pixel(format: u16) -> Option<u32> {
    Some(match format {
        MTL_FORMAT_A8_UNORM | MTL_FORMAT_R8_UNORM | MTL_FORMAT_R8_UINT | MTL_FORMAT_STENCIL8 => {
            R8_BPP
        }
        MTL_FORMAT_R16_FLOAT
        | MTL_FORMAT_RG8_UNORM
        | MTL_FORMAT_RG8_UINT
        | MTL_FORMAT_DEPTH16_UNORM => RG8_BPP,
        MTL_FORMAT_R16_UNORM => R16_BPP,
        MTL_FORMAT_RG16_UNORM | MTL_FORMAT_RG16_UINT | MTL_FORMAT_RG16_SINT => RG16_BPP,
        MTL_FORMAT_RG16_FLOAT => RG16F_BPP,
        MTL_FORMAT_RGBA8_UNORM
        | MTL_FORMAT_RGBA8_UNORM_SRGB
        | MTL_FORMAT_RGBA8_UINT
        | MTL_FORMAT_RGBA8_SINT
        | MTL_FORMAT_BGRA8_UNORM
        | MTL_FORMAT_BGRA8_UNORM_SRGB
        | MTL_FORMAT_R32_UINT
        | MTL_FORMAT_R32_SINT
        | MTL_FORMAT_R32_FLOAT
        // The packed families are four bytes for the same reason the byte
        // orders are: one 32-bit word per texel. What differs is where the
        // channel boundaries fall inside it, which is the format code's answer
        // and not the width's.
        | MTL_FORMAT_RGB9E5_FLOAT
        | MTL_FORMAT_RGB10A2_UNORM
        | MTL_FORMAT_RGB10A2_UINT
        | MTL_FORMAT_RG11B10_FLOAT
        | MTL_FORMAT_BGR10A2_UNORM
        | MTL_FORMAT_DEPTH32_FLOAT
        | MTL_FORMAT_DEPTH24_UNORM_STENCIL8
        | MTL_FORMAT_X24_STENCIL8 => RGBA8_BPP,
        // Depth32Float_Stencil8 / X32_Stencil8: 64-bit cells on Apple Silicon
        // (40-bit logical DS + pad; Metal allocates 8 B/texel for this family).
        MTL_FORMAT_DEPTH32_FLOAT_STENCIL8 | MTL_FORMAT_X32_STENCIL8 => 8,
        MTL_FORMAT_RGBA16_UNORM | MTL_FORMAT_RGBA16_UINT | MTL_FORMAT_RGBA16_FLOAT => RGBA16_BPP,
        MTL_FORMAT_RGBA32_UINT | MTL_FORMAT_RGBA32_FLOAT => RGBA32_BPP,
        _ => return None,
    })
}

/// Whether `format` has a depth plane (for `MTLBlitOptionDepthFromDepthStencil`).
pub fn format_has_depth_aspect(format: u16) -> bool {
    matches!(
        format,
        MTL_FORMAT_DEPTH16_UNORM
            | MTL_FORMAT_DEPTH32_FLOAT
            | MTL_FORMAT_DEPTH24_UNORM_STENCIL8
            | MTL_FORMAT_DEPTH32_FLOAT_STENCIL8
    )
}

/// Whether `format` has a stencil plane (for `MTLBlitOptionStencilFromDepthStencil`).
pub fn format_has_stencil_aspect(format: u16) -> bool {
    matches!(
        format,
        MTL_FORMAT_STENCIL8
            | MTL_FORMAT_DEPTH24_UNORM_STENCIL8
            | MTL_FORMAT_DEPTH32_FLOAT_STENCIL8
            | MTL_FORMAT_X32_STENCIL8
            | MTL_FORMAT_X24_STENCIL8
    )
}

/// Linear packing of a combined depth-stencil texel (full cell in guest storage).
///
/// Plane sizes when extracted to a buffer match Metal blit options:
/// depth plane → 4 B (Depth32Float / Depth24 expanded unorm32), stencil → 1 B.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DepthStencilPacking {
    /// Full texel size in guest linear storage.
    pub full_bpp: u32,
    /// Byte offset of the depth field within the texel (if present).
    pub depth_offset: u32,
    /// Buffer-side depth plane size after Metal extraction.
    pub depth_plane_bpp: u32,
    /// Byte offset of the stencil field within the texel (if present).
    pub stencil_offset: u32,
    pub stencil_plane_bpp: u32,
    /// How depth is stored in the packed texel.
    pub depth_layout: DepthFieldLayout,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DepthFieldLayout {
    /// Not a depth-bearing format (stencil-only combined / X\*\_Stencil8).
    None,
    /// IEEE-754 binary32 LE at `depth_offset`.
    Float32,
    /// 24-bit unorm in bits \[8:31\] of a LE u32 (stencil in low 8 bits).
    Unorm24High,
}

/// Combined depth-stencil packing for formats that interleave both planes.
///
/// Pure Depth32Float / Stencil8 / Depth16 return `None` (no repack; aspect is identity).
pub fn depth_stencil_packing(format: u16) -> Option<DepthStencilPacking> {
    match format {
        // Apple docs: 40-bit logical (32f + 8); Apple Silicon cells are 8 B.
        // Layout: depth f32 @0, stencil u8 @4, pad @5..7.
        MTL_FORMAT_DEPTH32_FLOAT_STENCIL8 => Some(DepthStencilPacking {
            full_bpp: 8,
            depth_offset: 0,
            depth_plane_bpp: 4,
            stencil_offset: 4,
            stencil_plane_bpp: 1,
            depth_layout: DepthFieldLayout::Float32,
        }),
        // 32-bit cell: stencil in low 8, depth unorm24 in high 24 (Metal/macOS common packing).
        MTL_FORMAT_DEPTH24_UNORM_STENCIL8 => Some(DepthStencilPacking {
            full_bpp: 4,
            depth_offset: 0,
            depth_plane_bpp: 4,
            stencil_offset: 0,
            stencil_plane_bpp: 1,
            depth_layout: DepthFieldLayout::Unorm24High,
        }),
        // X32_Stencil8: same 8 B cell as Depth32Float_Stencil8 without meaningful depth.
        MTL_FORMAT_X32_STENCIL8 => Some(DepthStencilPacking {
            full_bpp: 8,
            depth_offset: 0,
            depth_plane_bpp: 0,
            stencil_offset: 4,
            stencil_plane_bpp: 1,
            depth_layout: DepthFieldLayout::None,
        }),
        // X24_Stencil8: 4 B cell, stencil in low 8.
        MTL_FORMAT_X24_STENCIL8 => Some(DepthStencilPacking {
            full_bpp: 4,
            depth_offset: 0,
            depth_plane_bpp: 0,
            stencil_offset: 0,
            stencil_plane_bpp: 1,
            depth_layout: DepthFieldLayout::None,
        }),
        _ => None,
    }
}

/// Selected texture aspect for a buffer↔texture / options-bearing copy.
///
/// Three states, and exactly three: `MTLBlitOption`'s depth and stencil bits
/// are mutually exclusive, and
/// `crate::runtime::decode::blit::parse_blit_options` refuses the pair with
/// `ConflictingAspects` rather than producing one.
///
/// Lives here, below the decoder, because every consumer of the choice is a
/// pure format question — which plane of a packed texel, and how wide. The
/// decoder re-exports it under its own name.
///
/// This whole family used to travel as `(depth_aspect: bool, stencil_aspect:
/// bool)`, which spells a fourth state the decoder cannot emit, and the five
/// functions below did not agree on what it meant: two rejected `(true,
/// true)`, one treated it as a repack, and two read it as depth. One enum is
/// what makes that disagreement unwritable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlitAspect {
    /// Full texel (options None / zero).
    Full,
    /// Depth plane of a depth or depth-stencil texture.
    Depth,
    /// Stencil plane of a stencil or depth-stencil texture.
    Stencil,
}

/// Bytes per texel for a blit aspect selection.
///
/// Pure depth/stencil formats: aspect matches full bpp (option is identity).
/// Combined formats: Full uses packed `full_bpp`; depth plane is 4 B; stencil is 1 B.
pub fn blit_aspect_bytes_per_pixel(format: u16, aspect: BlitAspect) -> Option<u32> {
    match aspect {
        BlitAspect::Depth => {
            if !format_has_depth_aspect(format) {
                return None;
            }
            if let Some(p) = depth_stencil_packing(format) {
                return (p.depth_plane_bpp != 0).then_some(p.depth_plane_bpp);
            }
            match format {
                MTL_FORMAT_DEPTH16_UNORM => Some(2),
                MTL_FORMAT_DEPTH32_FLOAT => Some(4),
                _ => None,
            }
        }
        BlitAspect::Stencil => {
            if !format_has_stencil_aspect(format) {
                return None;
            }
            if let Some(p) = depth_stencil_packing(format) {
                return Some(p.stencil_plane_bpp);
            }
            Some(1)
        }
        // The whole texel — or the whole **block**, for a compressed format,
        // whose addressable unit is what a copy strides by. Derived from
        // `block_geometry` rather than `bytes_per_pixel` so the two families are
        // one expression: an uncompressed block is 1x1 and its `bytes` *is* the
        // bytes-per-texel, so this is the same number it always was for them.
        //
        // A caller taking this for a compressed format is being told the size of
        // one block and must stride in blocks. `runtime::blit_exec`'s
        // texture-to-texture copy is the one rail that does; every other blit
        // rail refuses a compressed format by name.
        BlitAspect::Full => block_geometry(format).map(|block| block.bytes),
    }
}

/// Whether a plane extract/insert pass is required (combined DS + aspect option).
pub fn blit_aspect_needs_repack(format: u16, aspect: BlitAspect) -> bool {
    if aspect == BlitAspect::Full {
        return false;
    }
    depth_stencil_packing(format).is_some()
}

/// Extract one plane from a packed depth-stencil texel into `dst` (plane-native size).
pub fn extract_depth_stencil_plane(
    format: u16,
    aspect: BlitAspect,
    texel: &[u8],
    dst: &mut [u8],
) -> bool {
    if aspect == BlitAspect::Full {
        return false;
    }
    let Some(p) = depth_stencil_packing(format) else {
        return false;
    };
    if texel.len() < p.full_bpp as usize {
        return false;
    }
    if aspect == BlitAspect::Depth {
        if p.depth_plane_bpp == 0 || dst.len() < p.depth_plane_bpp as usize {
            return false;
        }
        match p.depth_layout {
            DepthFieldLayout::Float32 => {
                let o = p.depth_offset as usize;
                dst[..4].copy_from_slice(&texel[o..o + 4]);
            }
            DepthFieldLayout::Unorm24High => {
                // Packed LE u32: stencil @bits0-7, depth unorm24 @bits8-31.
                // Metal depth buffer plane is 32-bit unorm (depth in low 24).
                let packed = u32::from_le_bytes([texel[0], texel[1], texel[2], texel[3]]);
                let depth24 = packed >> 8;
                dst[..4].copy_from_slice(&depth24.to_le_bytes());
            }
            DepthFieldLayout::None => return false,
        }
        return true;
    }
    // Stencil plane.
    if dst.len() < p.stencil_plane_bpp as usize {
        return false;
    }
    dst[0] = texel[p.stencil_offset as usize];
    true
}

/// Insert one plane into a packed depth-stencil texel (read-modify-write).
///
/// `texel` holds the current full cell (updated in place). `src` is plane-native.
pub fn insert_depth_stencil_plane(
    format: u16,
    aspect: BlitAspect,
    src: &[u8],
    texel: &mut [u8],
) -> bool {
    if aspect == BlitAspect::Full {
        return false;
    }
    let Some(p) = depth_stencil_packing(format) else {
        return false;
    };
    if texel.len() < p.full_bpp as usize {
        return false;
    }
    if aspect == BlitAspect::Depth {
        if p.depth_plane_bpp == 0 || src.len() < p.depth_plane_bpp as usize {
            return false;
        }
        match p.depth_layout {
            DepthFieldLayout::Float32 => {
                let o = p.depth_offset as usize;
                texel[o..o + 4].copy_from_slice(&src[..4]);
            }
            DepthFieldLayout::Unorm24High => {
                let depth24 = u32::from_le_bytes([src[0], src[1], src[2], src[3]]) & 0x00ff_ffff;
                let packed = u32::from_le_bytes([texel[0], texel[1], texel[2], texel[3]]);
                let stencil = packed & 0xff;
                let out = stencil | (depth24 << 8);
                texel[..4].copy_from_slice(&out.to_le_bytes());
            }
            DepthFieldLayout::None => return false,
        }
        return true;
    }
    if src.is_empty() {
        return false;
    }
    texel[p.stencil_offset as usize] = src[0];
    true
}

/// Extract a tight plane row from a strided packed texture row.
pub fn extract_plane_row(
    format: u16,
    aspect: BlitAspect,
    src_row: &[u8],
    width: u32,
    dst_plane: &mut [u8],
) -> bool {
    let Some(p) = depth_stencil_packing(format) else {
        return false;
    };
    let plane_bpp = match aspect {
        BlitAspect::Depth => p.depth_plane_bpp,
        BlitAspect::Stencil => p.stencil_plane_bpp,
        BlitAspect::Full => return false,
    } as usize;
    let full = p.full_bpp as usize;
    let w = width as usize;
    let Some(need_src) = full.checked_mul(w) else {
        return false;
    };
    let Some(need_dst) = plane_bpp.checked_mul(w) else {
        return false;
    };
    if src_row.len() < need_src || dst_plane.len() < need_dst {
        return false;
    }
    for x in 0..w {
        let t = &src_row[x * full..x * full + full];
        let d = &mut dst_plane[x * plane_bpp..x * plane_bpp + plane_bpp];
        if !extract_depth_stencil_plane(format, aspect, t, d) {
            return false;
        }
    }
    true
}

/// Insert a tight plane row into a strided packed texture row (RMW per texel).
pub fn insert_plane_row(
    format: u16,
    aspect: BlitAspect,
    src_plane: &[u8],
    width: u32,
    dst_row: &mut [u8],
) -> bool {
    let Some(p) = depth_stencil_packing(format) else {
        return false;
    };
    let plane_bpp = match aspect {
        BlitAspect::Depth => p.depth_plane_bpp,
        BlitAspect::Stencil => p.stencil_plane_bpp,
        BlitAspect::Full => return false,
    } as usize;
    let full = p.full_bpp as usize;
    let w = width as usize;
    let Some(need_dst) = full.checked_mul(w) else {
        return false;
    };
    let Some(need_src) = plane_bpp.checked_mul(w) else {
        return false;
    };
    if dst_row.len() < need_dst || src_plane.len() < need_src {
        return false;
    }
    for x in 0..w {
        let s = &src_plane[x * plane_bpp..x * plane_bpp + plane_bpp];
        let t = &mut dst_row[x * full..x * full + full];
        if !insert_depth_stencil_plane(format, aspect, s, t) {
            return false;
        }
    }
    true
}

/// Whether `format` stores sRGB-encoded values, so Metal decodes on sample and
/// encodes on write.
///
/// The class lookups below deliberately fold each `_SRGB` format onto its
/// linear sibling — the classes name a *byte layout*, and the two share one.
/// That fold is only safe because this predicate exists beside it: a caller that
/// takes the class has lost the qualifier and can ask here whether it just did.
/// Without it the loss is indistinguishable from the format never having been
/// sRGB at all.
pub fn is_srgb(format: u16) -> bool {
    matches!(
        format,
        MTL_FORMAT_RGBA8_UNORM_SRGB
            | MTL_FORMAT_BGRA8_UNORM_SRGB
            // The four BC families Apple gives an sRGB spelling. The fold this
            // predicate exists beside is the same one: `block_compressed_layout`
            // maps both spellings of each onto one layout, so the qualifier is
            // recoverable only from here.
            | MTL_FORMAT_BC1_RGBA_SRGB
            | MTL_FORMAT_BC2_RGBA_SRGB
            | MTL_FORMAT_BC3_RGBA_SRGB
            | MTL_FORMAT_BC7_RGBA_UNORM_SRGB
    )
}

/// Which **CPU-upload fast path** a sampled format's guest bytes qualify for, or
/// `None` for one that has to go through a per-texel convert.
///
/// # This is not the sampling admission rule
///
/// The name reads like one and it is not, which has cost a session: a `None`
/// here was once written up as "this device cannot sample that format", and a
/// plan was built on adding a variant to unblock a render target.
///
/// Nothing in this crate can *refuse* a bind because of this function. It has
/// two non-test callers, both in `runtime/draw/texture_view.rs`
/// (`linear_native_upload_format` and the tight-linear-load fast path), and both
/// treat a `None` — or any variant that is not `Rgba8Unorm`/`Bgra8Unorm` — as
/// "take the ordinary convert path". The convert path is not a decline.
///
/// What actually admits a sampled format is `translate::pixel::sampled_pixels`,
/// which answers a [`TexelLayout`] and declines by name with
/// `TranslateReason::NoSampledLayout`. That table is strictly wider than this
/// one: it carries the half-float and sixteen-bit-normalized layouts natively.
/// **Ask it, not this, when the question is whether a format can be sampled.**
///
/// So the members here are the layouts a loader can hand the uploader with no
/// conversion — which is why they are almost all eight-bit channel orders. A
/// format belongs here when its guest bytes are already in a final upload order,
/// and nowhere else does membership mean anything.
///
/// `Rgba16Float` is the one member no upload fast path reads. It is here as the
/// independent statement of a byte layout that
/// `a_byte_copy_destination_is_the_texel_every_other_table_agrees_it_is` checks
/// [`store_texel_order`] against — a consistency cross-check between two tables,
/// again not a capability claim.
pub fn sampled_class(format: u16) -> Option<SampledClass> {
    Some(match format {
        MTL_FORMAT_A8_UNORM => SampledClass::A8Unorm,
        MTL_FORMAT_R8_UNORM => SampledClass::R8Unorm,
        MTL_FORMAT_RG8_UNORM => SampledClass::Rg8Unorm,
        MTL_FORMAT_RGBA8_UNORM | MTL_FORMAT_RGBA8_UNORM_SRGB => SampledClass::Rgba8Unorm,
        MTL_FORMAT_BGRA8_UNORM | MTL_FORMAT_BGRA8_UNORM_SRGB => SampledClass::Bgra8Unorm,
        MTL_FORMAT_RGBA16_FLOAT => SampledClass::Rgba16Float,
        MTL_FORMAT_RG16_FLOAT => SampledClass::Rg16Float,
        MTL_FORMAT_BGR10A2_UNORM => SampledClass::Bgr10a2Unorm,
        MTL_FORMAT_RG16_UINT => SampledClass::Rg16Uint,
        MTL_FORMAT_RGBA32_FLOAT => SampledClass::Rgba32Float,
        _ => return None,
    })
}

/// Which storage-image selector a Metal format maps to, or `None` for a format
/// this host will not expose as a storage image.
///
/// The texel width is **not** returned. It used to be, as a second column
/// beside each selector, and the only thing that column was ever used for was a
/// `debug_assert_eq!` against [`bytes_per_pixel`] at three of the four call
/// sites — the fourth discarded it. A number stated twice that nothing reads is
/// a number that can disagree with itself in a release build, where a
/// `debug_assert` is not compiled at all. `storage_texel_width_matches_the_pixel_table`
/// now holds the same invariant for every `u16`, at test time, exhaustively.
pub fn storage_selector(format: u16) -> Option<StorageImageSelector> {
    Some(match format {
        MTL_FORMAT_R8_UNORM => StorageImageSelector::R8Unorm,
        MTL_FORMAT_R32_UINT => StorageImageSelector::R32Uint,
        MTL_FORMAT_RG8_UNORM => StorageImageSelector::Rg8Unorm,
        MTL_FORMAT_R16_FLOAT => StorageImageSelector::R16Float,
        MTL_FORMAT_RG16_FLOAT => StorageImageSelector::Rg16Float,
        MTL_FORMAT_RGBA8_UNORM => StorageImageSelector::Rgba8Unorm,
        MTL_FORMAT_BGRA8_UNORM => StorageImageSelector::Bgra8Unorm,
        MTL_FORMAT_RGBA8_UINT => StorageImageSelector::Rgba8Uint,
        MTL_FORMAT_RGBA8_SINT => StorageImageSelector::Rgba8Sint,
        MTL_FORMAT_RGBA16_UINT => StorageImageSelector::Rgba16Uint,
        MTL_FORMAT_RGBA16_FLOAT => StorageImageSelector::Rgba16Float,
        MTL_FORMAT_RGBA32_UINT => StorageImageSelector::Rgba32Uint,
        MTL_FORMAT_RGBA32_FLOAT => StorageImageSelector::Rgba32Float,
        _ => return None,
    })
}

/// Storage bytes per texel of a format this host will render into, or `None`
/// for one it will not.
///
/// The match arms *are* the renderable set — the answer to "may a colour
/// attachment be this format" is `.is_some()`, which is how
/// `runtime/draw` asks it, and since the two arms were made one answer it is
/// also how `translate::pixel::color_attachment` asks it. There used to be a
/// `RenderTargetClass` enum returned alongside the width, one variant per arm
/// below; every caller discarded it, so it named the same six formats a second
/// time and could disagree with this list without anything noticing.
///
/// Adding an arm here is therefore the whole of adding a renderable format, and
/// it obliges two things of the layout tables: the layout must narrow to RGBA8
/// (`narrow_texel_to_rgba8`, for the readback rails) and expand from it
/// (`expand_rgba8_to_texel`, for a CPU `Load` seed). A format admitted here that
/// those refuse is a render target the guest can create and then lose the frame
/// of on any host that cannot land it GPU-direct.
///
/// `R16_FLOAT` is here because macOS 26 asks for it — a linear GVA target at
/// `fmt=0x19`, previously refused three times a driven boot as `rt_resolve
/// reason=rt_linear_format`. Metal renders to it on every Apple GPU and Vulkan
/// mandates `R16_SFLOAT` for both `COLOR_ATTACHMENT_BIT` and
/// `COLOR_ATTACHMENT_BLEND_BIT` in optimal tiling, so no capability gate is owed
/// and no host can decline it.
///
/// The gap that had to close first was the third rail, not the sampler: the CPU
/// Store converter reaches [`rgba8_to_texel`], which carried no `R16_FLOAT` arm,
/// so the format would have rendered fine and then lost every frame on any host
/// without a guest-RAM import. [`sampled_class`] answering `None` for it is
/// **not** a blocker and was once recorded as one — read that function's doc
/// before believing otherwise; it selects a CPU-upload fast path and cannot
/// refuse a bind. What admits a sampled `R16_FLOAT` is
/// `translate::pixel::sampled_pixels`, which has carried it as a native
/// [`TexelLayout::R16Float`] rail throughout.
///
/// `R8_UNORM` is here for the same kind of reading one rail over: macOS 26 also
/// renders into a single-channel *eight-bit* linear GVA target — a coverage,
/// mask or shadow layer — refused once a driven boot as `rt_resolve
/// reason=rt_linear_format fmt=0xa`. Vulkan mandates `R8_UNORM` as a colour
/// attachment, so again no capability is owed. It needed three arms rather than
/// one, because a one-byte texel had never been a render target: the readback
/// narrow, the CPU `Load` seed expansion and the CPU Store converter.
///
/// `BGR10A2_UNORM` is here because a **game** asks for it, and it is the first
/// packed 32-bit colour format admitted. A driven macos-13 x86/Vulkan boot
/// running Asphalt 8 creates its 1280x720 render surface as an `'l10r'`
/// IOSurface — `kCVPixelFormatType_ARGB2101010LEPacked`, which
/// `runtime::objects::iosurface_pixel_format_to_mtl` now names — and before
/// either half of that change every draw of the frame was refused at
/// `draw::render_target`'s `rt_type4_base_format` and the window was black:
/// 20 822 `draw_fail_clear_fallback` records and zero successful draws in one
/// 100 s capture. Vulkan does not *mandate* `A2R10G10B10_UNORM_PACK32` as a
/// colour attachment the way it does `R16_SFLOAT`, so unlike the two members
/// above this one is a format a host could in principle decline —
/// `translate::pixel::color_attachment` is where such a decline would surface,
/// and the NVIDIA host this was measured on advertises it.
///
/// It needed all three conversion rails plus a fourth thing the two members
/// above did not: an arm in [`store_texel_order`], because its channels do not
/// sit on byte boundaries and so the CPU converter's eight-bit round trip is the
/// only lossy rail in the set. The byte copy is what normally runs and it is
/// exact.
///
/// `RG8_UNORM` is deliberately *not* here. No guest has been observed declaring
/// a two-channel eight-bit render target, and admitting a format costs three
/// conversions plus a census line, so members are added on measurement rather
/// than by family resemblance.
///
/// It is absent from [`store_texel_order`] for the reason `RG16_FLOAT` is, which
/// that function's doc states: a missing arm there is a performance bug, not a
/// loss, and the byte-copy rail declines by name to the CPU converter above.
///
/// # `RG16_UINT` is here, and it is the member that changed what admission means
///
/// A macos-15 x86/Vulkan boot declares linear `RG16Uint` colour attachments and
/// used to lose every pass that named one, refused at `draw::render_target`'s
/// `rt_linear_format` rung.
///
/// It could not be admitted by adding a table entry.
/// `the_renderable_set_is_one_answer_and_every_member_survives_both_rails` is
/// where the obligation is written down: every member above survives the
/// readback narrow, the CPU `Load` seed expansion and the CPU `Store` row
/// converter, all three of which pass a texel through eight-bit RGBA. An
/// integer texel has no eight-bit form — a clear colour of `1.0` is the integer
/// `1`, not `255` and not `65535` — so all three would have had to invent
/// bytes, and a format admitted on those terms renders correctly on a host with
/// the guest-RAM import and loses every frame, silently, on one without.
///
/// So the rail was built instead of the exemption. A render Store now carries
/// the destination's own texel end to end — [`store_texel_order`] on the
/// GPU-direct arm, and `draw::FrameRows::Native` over
/// `engine::ReadbackTexel::Native` on the copying arm — so the byte copy serves
/// this format exactly on both, and neither invents a byte. That test asserts
/// the alternative rather than skipping: a renderable layout that narrows to
/// RGBA8 owes the three eight-bit rails, and one that does not owes the native
/// one.
///
/// Two consequences worth carrying, because both have already cost a boot:
///
/// * The layout must be in the render-target capability vocabulary
///   ([`TexelLayout::is_render_target_layout`]), not in a *blend* one. Vulkan
///   mandates no `COLOR_ATTACHMENT_BLEND` for an integer format, so a mask that
///   asks about blending has no bit to hold this layout's answer and reads
///   `false` — which builds the resident at eight bits and loses the frame at a
///   later rung under a different name.
/// * The CPU `Load` seed still has no integer arm, so a LOAD-action pass on
///   such a target refuses by name (`SeedFormatUnwritable`) and loses its prior
///   contents. The Store does not.
///
/// sRGB variants share storage bpp with their unorm counterparts (Metal texture
/// view rules).
///
/// # The width is derived, never re-listed
///
/// This used to spell out a bytes-per-texel for each member, which made it the
/// third independent transcription of a width [`bytes_per_pixel`] already owns
/// — the exact second spelling [`block_geometry`]'s doc says it exists to avoid
/// having. The two agreed when this was written and nothing held them there.
///
/// So the width now comes from [`bytes_per_pixel`] and this function answers
/// only the question that is its own: **will this device serve `format` as a
/// colour render target**. That question is genuinely narrower than "is the
/// width known" — the contract defines a width for every format in its table,
/// including depth and block-compressed ones no colour attachment may name —
/// and conflating the two is what made a missing width read as a missing
/// capability.
pub fn render_target_bpp(format: u16) -> Option<u32> {
    render_target_numeric_type(format)?;
    // Every admitted member has a width, because admission is a strictly
    // smaller set than the width table. A member without one is a bug in this
    // list rather than a format to refuse quietly.
    bytes_per_pixel(format)
}

/// The numeric interpretation of a colour render target's channels.
///
/// This is also the admission set [`render_target_bpp`]'s doc argues for. A
/// caller cannot learn that a format is renderable without also learning which
/// numeric class its clear value and fragment output use, so an integer member
/// cannot accidentally inherit the continuous-colour representation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorNumericType {
    Float,
    Uint,
    Sint,
}

/// A decoded clear after its attachment's numeric class chose the backend
/// clear-value carrier.
///
/// The integer carriers are deliberately 32-bit even when the destination's
/// channels are narrower. That is the render API contract: the backend narrows
/// a clear to the attachment format from this carrier, so CPU publication must
/// start from the same value rather than cast the decoded double straight to a
/// destination channel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ClearComponents {
    Float([f32; COMPONENT_COUNT]),
    Uint([u32; COMPONENT_COUNT]),
    Sint([i32; COMPONENT_COUNT]),
}

impl ColorNumericType {
    /// Convert Metal's double-precision component carrier into the typed clear
    /// value consumed by the backend and CPU publication rails.
    pub fn clear_components(self, components: [f64; COMPONENT_COUNT]) -> ClearComponents {
        match self {
            Self::Float => ClearComponents::Float(components.map(|value| value as f32)),
            Self::Uint => ClearComponents::Uint(components.map(|value| value as u32)),
            Self::Sint => ClearComponents::Sint(components.map(|value| value as i32)),
        }
    }
}

/// Return the numeric type for every colour render target this device serves.
///
/// Adding a member here is the three-conversion commitment
/// [`render_target_bpp`] describes, not just a table entry. The return value is
/// deliberately richer than a boolean: Vulkan clear values are a union, and
/// choosing its float member for a `Uint` attachment reinterprets `1.0` as the
/// integer bit pattern `1065353216`.
pub fn render_target_numeric_type(format: u16) -> Option<ColorNumericType> {
    Some(match format {
        MTL_FORMAT_RGBA8_UNORM
        | MTL_FORMAT_RGBA8_UNORM_SRGB
        | MTL_FORMAT_BGRA8_UNORM
        | MTL_FORMAT_BGRA8_UNORM_SRGB
        | MTL_FORMAT_RGBA16_FLOAT
        | MTL_FORMAT_RG16_FLOAT
        | MTL_FORMAT_R16_FLOAT
        | MTL_FORMAT_R8_UNORM
        | MTL_FORMAT_BGR10A2_UNORM => ColorNumericType::Float,
        MTL_FORMAT_RG16_UINT => ColorNumericType::Uint,
        _ => return None,
    })
}

/// The texel layout a render Store's destination stores its texels in, or
/// `None` for a destination whose texel this device cannot name a layout for.
///
/// This is the whole admission rule for landing a resident render target in
/// guest memory with an image→buffer copy: that copy moves bytes and converts
/// nothing, so the destination's texel and the resident's must be **the same
/// layout**. Say `Some(layout)` and the caller compares it against its
/// resident's format; say `None` and the only route left is
/// [`convert_rgba8_to_row`], which is a CPU pass over the frame.
///
/// Named once because both writeback rails ask it — the type-11 mapping rail
/// wants `Bgra8` specifically and the GVA rail takes whichever layout its
/// resident was built in — and a rail that re-lists the formats drifts the
/// first time one is added.
///
/// # Why `RGBA16_FLOAT` is here now
///
/// It used to be excluded, and this doc used to say so: *"`RGBA16_FLOAT` is
/// renderable and is not a byte-copy destination."* That was true, but of the
/// **resident** rather than of the contract. Every render target's resident was
/// eight bits per channel — the identity could hold a channel order and nothing
/// wider — so a half-float destination could never be the same bytes as the
/// image, at any order, and the copy was correctly refused.
///
/// A resident now carries the format the guest declared, so for a target that
/// got one the copy is byte-for-byte valid and this says so. The caller is what
/// makes that safe: it compares this layout against the resident's actual
/// format, so a half-float destination whose resident *did* fall back to eight
/// bits still takes the CPU rail. Returning a layout is an admission that the
/// two **may** match, never a claim that they do.
///
/// The rule for membership is the doc's own and not a second list's: a layout
/// belongs here when a guest render target can declare it and a copy of the
/// resident's bytes is what the guest should read. That is not mechanically
/// [`render_target_bpp`]`.is_some()` — a renderable format could in principle
/// want a conversion on the way out — but every renderable format so far
/// satisfies it, because a resident is created at the format the guest declared
/// and the guest reads its own destination back at that same format.
///
/// **A renderable format missing from here is a performance bug, not a loss.**
/// The byte-copy rail declines by name and the caller falls to the CPU
/// converter, which is [`convert_rgba8_to_row`] — so the obligation a missing
/// arm creates lands there instead, and that one *is* a loss if unmet.
/// `RG16_FLOAT` is renderable and absent from here for that reason and no
/// other: `a_byte_copy_destination_is_the_texel_every_other_table_agrees_it_is`
/// requires an admitted format to have a [`sampled_class`] naming the same
/// texel, and `RG16_FLOAT` has none.
///
/// sRGB folds onto its linear sibling for the same reason [`sampled_class`]
/// folds it: the qualifier describes how a sampler interprets the bytes, not
/// how they are stored, and only the storage matters to a copy.
pub fn store_texel_order(format: u16) -> Option<TexelLayout> {
    Some(match format {
        MTL_FORMAT_RGBA8_UNORM | MTL_FORMAT_RGBA8_UNORM_SRGB => TexelLayout::Rgba8,
        MTL_FORMAT_BGRA8_UNORM | MTL_FORMAT_BGRA8_UNORM_SRGB => TexelLayout::Bgra8,
        MTL_FORMAT_RGBA16_FLOAT => TexelLayout::Rgba16Float,
        // The packed ten-bit colour word. Admitted because the byte copy is the
        // only rail that can land it without loss: the CPU converter reaches
        // [`rgba8_to_texel`], whose arm for this format requantizes each channel
        // from eight bits back up to ten, and ten bits is what the guest picked
        // the format for. A resident created in the declared format holds the
        // identical `VK_FORMAT_A2R10G10B10_UNORM_PACK32` word the guest's
        // destination does, so the copy converts nothing.
        MTL_FORMAT_BGR10A2_UNORM => TexelLayout::Bgr10a2Unorm,
        // The integer colour target, and the member whose absence here would be
        // a **loss** rather than a slow path. Every other member declines to the
        // CPU converter; this one has no CPU converter to decline to, because
        // [`rgba8_to_texel`] has no arm for an integer texel and must not gain
        // one. The byte copy is its only rail, on the GPU-direct arm and on the
        // copying arm alike — which is why the copying arm learned to carry a
        // native frame rather than always an eight-bit one.
        MTL_FORMAT_RG16_UINT => TexelLayout::Rg16Uint,
        _ => return None,
    })
}

/// Bytes one row of `width` texels occupies with no padding.
///
/// For a compressed format this is one row of **blocks**, which is the row a
/// copy strides by and the row `VkBufferImageCopy::bufferRowLength` describes
/// when it is left at zero. One expression covers both families because an
/// uncompressed format's block is 1x1 — see [`block_geometry`].
pub fn tight_row_bytes(width: u32, format: u16) -> Option<u32> {
    if width == 0 {
        return None;
    }
    let block = block_geometry(format)?;
    block.blocks_across(width).checked_mul(block.bytes)
}

/// Rows a copy of a `height`-tall image of `format` must walk.
///
/// The texel height for an uncompressed format and a quarter of it, rounded up,
/// for BC. Named rather than open-coded because the row loops that need it are
/// in three files and a `height` left un-divided reads as correct.
pub fn tight_row_count(height: u32, format: u16) -> Option<u32> {
    Some(block_geometry(format)?.block_rows(height))
}

pub fn swizzle_identity() -> SwizzlePlan {
    SwizzlePlan {
        source: [
            SwizzleSource::R,
            SwizzleSource::G,
            SwizzleSource::B,
            SwizzleSource::A,
        ],
    }
}

fn swizzle_selector_source(selector: u8) -> Option<SwizzleSource> {
    Some(match selector {
        0 => SwizzleSource::Zero,
        1 => SwizzleSource::One,
        2 => SwizzleSource::R,
        3 => SwizzleSource::G,
        4 => SwizzleSource::B,
        5 => SwizzleSource::A,
        _ => return None,
    })
}

pub fn swizzle_plan(raw: &[u8; COMPONENT_COUNT]) -> Option<SwizzlePlan> {
    let mut source = [SwizzleSource::Zero; COMPONENT_COUNT];
    for i in 0..COMPONENT_COUNT {
        source[i] = swizzle_selector_source(raw[i])?;
    }
    Some(SwizzlePlan { source })
}

pub fn swizzle_is_identity(plan: &SwizzlePlan) -> bool {
    plan.source
        == [
            SwizzleSource::R,
            SwizzleSource::G,
            SwizzleSource::B,
            SwizzleSource::A,
        ]
}

pub fn apply_swizzle_rgba8(plan: &SwizzlePlan, in_rgba: [u8; 4]) -> [u8; 4] {
    let mut out = [0u8; 4];
    for (component, source) in out.iter_mut().zip(plan.source) {
        *component = match source {
            SwizzleSource::Zero => UNORM8_MIN,
            SwizzleSource::One => UNORM8_MAX,
            SwizzleSource::R => in_rgba[COMPONENT_R],
            SwizzleSource::G => in_rgba[COMPONENT_G],
            SwizzleSource::B => in_rgba[COMPONENT_B],
            SwizzleSource::A => in_rgba[COMPONENT_A],
        };
    }
    out
}

pub fn f64_to_unorm8(value: f64) -> u8 {
    if !matches!(value.partial_cmp(&0.0), Some(std::cmp::Ordering::Greater)) {
        UNORM8_MIN
    } else if value >= 1.0 {
        UNORM8_MAX
    } else {
        (value * f64::from(UNORM8_MAX) + 0.5) as u8
    }
}

/// A tightly-packed `w`×`h` RGBA8 image of one colour.
///
/// What this device hands back when it services a colour attachment's CLEAR
/// itself, rather than encoding one. `clear` is the guest's `MTLClearColor`, so
/// each channel goes through [`f64_to_unorm8`], which is where the out-of-range
/// and NaN rules live.
///
/// # One definition, because the two it had disagreed
///
/// This was `runtime::exec::solid_rgba` and `runtime::draw::solid_rgba_local`,
/// byte-identical bodies six call sites apart, and both carried the same defect:
/// the buffer's length widened each axis before multiplying, so it cannot
/// overflow on any host this runs on, while the fill counted texels in `u32` as
/// `0..(w * h) as usize`, which overflows at 65536×65536. Only a zero
/// axis is refused upstream of either; `MAX_SCANOUT_DIM` bounds the scanout
/// registers and says nothing about a render target's geometry. A debug build
/// panics there, taking the guest down; a release build wraps to a small count
/// and returns a full-size buffer filled for a fraction of it — a clear that
/// silently did not clear.
///
/// The fill therefore walks the buffer instead of counting texels. A
/// `chunks_exact_mut` cannot describe a different image from the one that was
/// allocated, where a second expression always can — the rule
/// [`crate::extent::tight_image_layout`] states for a length and its
/// stride, one level down.
///
/// Here rather than in either caller because it is arithmetic both rails need
/// and neither owns, and because `contract` is the tree that gets tested on
/// every arm.
pub fn solid_rgba8(w: u32, h: u32, clear: &[f64; 4]) -> Vec<u8> {
    solid_image8(w, h, unorm8_rgba(clear))
}

/// How a CPU-published clear image carries its pixels.
///
/// Continuous-colour attachments keep the existing semantic RGBA8 carrier and
/// are converted by the destination writer. Integer attachments have no such
/// carrier: their component values are counts rather than fractions, so their
/// image is already encoded as the destination's native texels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClearImageEncoding {
    Rgba8,
    Native,
}

/// A clear-only render result ready to be published into guest memory.
///
/// The encoding and bytes are produced together from the admitted render-target
/// format. Callers can therefore choose the converted or native writer without
/// reconstructing the numeric-format rule that chose the bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClearImage {
    encoding: ClearImageEncoding,
    pixels: Vec<u8>,
    row_bytes: u32,
}

impl ClearImage {
    pub fn encoding(&self) -> ClearImageEncoding {
        self.encoding
    }

    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }

    pub fn row_bytes(&self) -> u32 {
        self.row_bytes
    }
}

/// Lower a render-target clear into the representation guest-memory writers
/// can publish without changing its numeric meaning.
///
/// This is the CPU counterpart to the backend's format-aware clear value. A
/// format admitted by [`render_target_numeric_type`] must have an arm here: a
/// clear-only pass has no GPU draw from which to obtain a frame, so this image
/// is the pass result.
pub fn solid_clear_image(
    format: u16,
    width: u32,
    height: u32,
    clear: &[f64; COMPONENT_COUNT],
) -> Option<ClearImage> {
    let numeric = render_target_numeric_type(format)?;
    match numeric.clear_components(*clear) {
        ClearComponents::Float(_) => {
            let row_bytes = width.checked_mul(RGBA8_BPP)?;
            let pixels = solid_rgba8(width, height, clear);
            let need = (row_bytes as usize).checked_mul(height as usize)?;
            (pixels.len() == need).then_some(ClearImage {
                encoding: ClearImageEncoding::Rgba8,
                pixels,
                row_bytes,
            })
        }
        ClearComponents::Uint(components) => match format {
            MTL_FORMAT_RG16_UINT => {
                let mut bytes = [0u8; RG16_BPP as usize];
                st16(&mut bytes[0..2], components[COMPONENT_R] as u16);
                st16(&mut bytes[2..4], components[COMPONENT_G] as u16);
                solid_native_clear_image(format, width, height, &bytes)
            }
            _ => None,
        },
        ClearComponents::Sint(_) => None,
    }
}

fn solid_native_clear_image(
    format: u16,
    width: u32,
    height: u32,
    texel: &[u8],
) -> Option<ClearImage> {
    let row_bytes = tight_row_bytes(width, format)?;
    let bpp = bytes_per_pixel(format)? as usize;
    if texel.len() != bpp {
        return None;
    }
    let texels = (width as usize).checked_mul(height as usize)?;
    let need = texels.checked_mul(bpp)?;
    let pixels = solid_bytes(need, texel)?;
    Some(ClearImage {
        encoding: ClearImageEncoding::Native,
        pixels,
        row_bytes,
    })
}

/// Fill exactly `len` bytes with whole copies of `unit`.
fn solid_bytes(len: usize, unit: &[u8]) -> Option<Vec<u8>> {
    if len == 0 {
        return Some(Vec::new());
    }
    if unit.is_empty() || !len.is_multiple_of(unit.len()) {
        return None;
    }
    let mut bytes = Vec::with_capacity(len);
    bytes.extend_from_slice(unit);
    while bytes.len() < len {
        let take = (len - bytes.len()).min(bytes.len());
        bytes.extend_from_within(..take);
    }
    Some(bytes)
}

/// The clear colour as one unorm8 RGBA texel.
fn unorm8_rgba(clear: &[f64; 4]) -> [u8; 4] {
    [
        f64_to_unorm8(clear[COMPONENT_R]),
        f64_to_unorm8(clear[COMPONENT_G]),
        f64_to_unorm8(clear[COMPONENT_B]),
        f64_to_unorm8(clear[COMPONENT_A]),
    ]
}

/// `w * h` copies of `px`, tightly packed, each byte written exactly once.
///
/// The fill doubles what it has already written rather than walking texels, so
/// the buffer is filled by `memcpy` at growing sizes instead of by a four-byte
/// store per texel, and it is never zeroed first — `vec![0u8; n]` followed by a
/// fill writes every byte twice.
///
/// The length is still driven by the buffer rather than by a texel count, which
/// is the property [`solid_rgba8`]'s doc above requires: `extend_from_within`
/// can only copy bytes this buffer already holds, and the final truncation is to
/// the same `n` the capacity was reserved for, so no arithmetic here can
/// describe a different image from the one that was allocated.
fn solid_image8(w: u32, h: u32, px: [u8; 4]) -> Vec<u8> {
    let n = (w as usize)
        .saturating_mul(h as usize)
        .saturating_mul(px.len());
    if n < px.len() {
        // A zero-texel image, or one whose geometry saturated. Either way there
        // is no first texel to double from.
        return Vec::new();
    }
    let mut img = Vec::with_capacity(n);
    img.extend_from_slice(&px);
    while img.len() < n {
        let take = (n - img.len()).min(img.len());
        img.extend_from_within(..take);
    }
    img
}

pub fn f16_to_f32(half_bits: u16) -> f32 {
    let sign = (u32::from(half_bits & F16_SIGN_MASK)) << F16_F32_SIGN_SHIFT;
    let exp = (u32::from(half_bits) >> F16_EXP_SHIFT) & F16_EXP_MASK;
    let mut mant = u32::from(half_bits) & F16_MANT_MASK;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut normal_exp: i32 = 1;
            while (mant & F16_HIDDEN_BIT) == 0 {
                mant <<= 1;
                normal_exp -= 1;
            }
            mant &= F16_MANT_MASK;
            sign | (((normal_exp - F16_EXP_BIAS + F32_EXP_BIAS) as u32) << F32_EXP_SHIFT)
                | (mant << F16_F32_MANT_SHIFT)
        }
    } else if exp == F16_EXP_INF_NAN {
        sign | F32_INF_BITS | (mant << F16_F32_MANT_SHIFT)
    } else {
        let f32_exp = (exp as i32 - F16_EXP_BIAS + F32_EXP_BIAS) as u32;
        sign | (f32_exp << F32_EXP_SHIFT) | (mant << F16_F32_MANT_SHIFT)
    };
    f32::from_bits(bits)
}

fn build_f16_to_unorm8_lut() -> Box<[u8; 65536]> {
    let mut lut = Box::new([0u8; 65536]);
    for i in 0..=u16::MAX {
        let f = f16_to_f32(i);
        lut[i as usize] = if !matches!(f.partial_cmp(&0.0), Some(std::cmp::Ordering::Greater)) {
            UNORM8_MIN
        } else if f >= 1.0 {
            UNORM8_MAX
        } else {
            (f * f32::from(UNORM8_MAX) + 0.5) as u8
        };
    }
    lut
}

fn f16_to_unorm8_lut() -> &'static [u8; 65536] {
    use std::sync::OnceLock;
    static LUT: OnceLock<Box<[u8; 65536]>> = OnceLock::new();
    LUT.get_or_init(build_f16_to_unorm8_lut)
}

fn unorm8_to_f16_slow(value: u8) -> u16 {
    let f = f32::from(value) / f32::from(UNORM8_MAX);
    let x = f.to_bits();
    let sign = ((x >> F16_F32_SIGN_SHIFT) as u16) & F16_SIGN_MASK;
    let e = ((x >> F32_EXP_SHIFT) & F32_EXP_MASK) as i32 - F32_EXP_BIAS + F16_EXP_BIAS;
    let mut m = x & F32_MANT_MASK;

    if f <= 0.0 {
        return sign;
    }
    if e >= F16_EXP_INF_NAN as i32 {
        return sign | F16_INF_BITS;
    }
    if e <= 0 {
        if e < F16_SUBNORMAL_EXP_MIN {
            return sign;
        }
        m |= F32_HIDDEN_BIT;
        let shift = (F16_SUBNORMAL_SHIFT_BASE - e) as u32;
        let mut hm = m >> shift;
        if ((m >> (shift - 1)) & 1) != 0 {
            hm += 1;
        }
        return sign | (hm as u16);
    }

    let mut h = sign | (((e as u32) << F16_EXP_SHIFT) as u16) | ((m >> F16_F32_MANT_SHIFT) as u16);
    if (m & F32_TO_F16_ROUND_BIT) != 0 {
        h = h.wrapping_add(1);
    }
    h
}

fn unorm8_to_f16_lut() -> &'static [u16; 256] {
    use std::sync::OnceLock;
    static LUT: OnceLock<[u16; 256]> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut lut = [0u16; 256];
        for i in 0..=UNORM8_MAX {
            lut[i as usize] = unorm8_to_f16_slow(i);
        }
        lut
    })
}

/// Bit position of each channel in `MTLPixelFormatBGR10A2Unorm`'s texel word.
///
/// One little-endian 32-bit word: blue in bits 0..9, green in 10..19, red in
/// 20..29 and alpha in 30..31. Those are the same bits in the same order as
/// `VK_FORMAT_A2R10G10B10_UNORM_PACK32`, which is why the byte copy is exact and
/// [`store_texel_order`] admits the format — the two functions below serve only
/// the rails that cannot copy.
///
/// Stated as shifts rather than as a struct because the channels do not sit on
/// byte boundaries, which is also why no byte-shaped loader can serve this
/// format and why [`TexelLayout::has_cpu_loader_arm`] answers `false` for it.
const BGR10A2_BLUE_SHIFT: u32 = 0;
const BGR10A2_GREEN_SHIFT: u32 = 10;
const BGR10A2_RED_SHIFT: u32 = 20;
const BGR10A2_ALPHA_SHIFT: u32 = 30;
/// Width of one colour channel in [`BGR10A2_RED_SHIFT`]'s word.
const BGR10A2_COLOR_MASK: u32 = 0x3ff;
/// Width of the alpha channel in the same word.
const BGR10A2_ALPHA_MASK: u32 = 0x3;
/// The three colour channels tile the word below alpha, and alpha closes it.
const _: () = assert!(
    BGR10A2_ALPHA_SHIFT == BGR10A2_RED_SHIFT + BGR10A2_COLOR_MASK.count_ones()
        && BGR10A2_RED_SHIFT == BGR10A2_GREEN_SHIFT + BGR10A2_COLOR_MASK.count_ones()
        && BGR10A2_GREEN_SHIFT == BGR10A2_BLUE_SHIFT + BGR10A2_COLOR_MASK.count_ones()
        && BGR10A2_ALPHA_SHIFT + BGR10A2_ALPHA_MASK.count_ones() == u32::BITS
);

/// One `BGR10A2Unorm` word read as the four channels a semantic RGBA8 frame
/// holds — the narrowing half of the pair, for the host readback rails.
///
/// A truncation and nothing else: ten bits of unorm become the eight most
/// significant of them, and two bits of alpha become the four-fold replication
/// of themselves, so the two-bit values `0..3` read back as `0, 85, 170, 255`.
/// That replication is what makes the pair below an identity on every value a
/// widened channel can hold, which is the property
/// `a_packed_ten_bit_texel_survives_the_seed_and_readback_round_trip` checks.
fn bgr10a2_word_to_rgba8(word: u32) -> [u8; 4] {
    let channel = |shift: u32| (((word >> shift) & BGR10A2_COLOR_MASK) >> 2) as u8;
    let alpha = ((word >> BGR10A2_ALPHA_SHIFT) & BGR10A2_ALPHA_MASK) as u8;
    let mut out = [0u8; COMPONENT_COUNT];
    out[COMPONENT_R] = channel(BGR10A2_RED_SHIFT);
    out[COMPONENT_G] = channel(BGR10A2_GREEN_SHIFT);
    out[COMPONENT_B] = channel(BGR10A2_BLUE_SHIFT);
    out[COMPONENT_A] = alpha * BGR10A2_ALPHA_REPLICATE;
    out
}

/// `0b01` in two bits is this value in eight, so multiplying replicates the pair
/// across the byte and maps `0b11` to full scale rather than to `0xc0`.
const BGR10A2_ALPHA_REPLICATE: u8 = 0x55;

/// Four semantic-RGBA8 channels written into one `BGR10A2Unorm` word — the
/// widening half, for a CPU `Load` seed and for the CPU Store converter.
///
/// Each colour channel gains two bits and they are filled with the value's own
/// top two, which is the unorm widening that keeps both endpoints: `0` stays `0`
/// and `255` becomes `1023`. Alpha loses six bits and keeps its top two, which
/// is [`bgr10a2_word_to_rgba8`]'s replication inverted.
fn rgba8_to_bgr10a2_word(rgba: [u8; COMPONENT_COUNT]) -> u32 {
    let channel = |v: u8| {
        let v = u32::from(v);
        (v << 2) | (v >> 6)
    };
    (u32::from(rgba[COMPONENT_A]) >> 6) << BGR10A2_ALPHA_SHIFT
        | channel(rgba[COMPONENT_R]) << BGR10A2_RED_SHIFT
        | channel(rgba[COMPONENT_G]) << BGR10A2_GREEN_SHIFT
        | channel(rgba[COMPONENT_B]) << BGR10A2_BLUE_SHIFT
}

/// Restate a semantic-RGBA8 frame as `layout`'s own texels.
///
/// The render-target seed's counterpart to [`convert_rgba8_to_row`], keyed on a
/// [`TexelLayout`] rather than on a guest format because the caller is the
/// engine and what it holds is the *attachment's* layout.
///
/// # Why a colour attachment needs this at all
///
/// A CPU `MTLLoadActionLoad` seed is staged into a buffer and copied into the
/// attachment, and a Vulkan buffer→image copy converts nothing: it reads the
/// image format's texel width per pixel, straight out of the buffer. While
/// every render target was eight bits per channel the seed was already those
/// bytes and the only question was the channel order. A wider attachment reads
/// twice as many bytes per texel as an RGBA8 seed provides, walks off the end
/// of the staging slot, and seeds the frame with whatever followed it.
///
/// `false` for a layout no colour attachment can be created at, so the caller
/// declines by name rather than staging a frame of the wrong size. Exhaustive
/// over [`TexelLayout`] on purpose — a new layout that becomes renderable has
/// to say here how its seed is written.
///
/// The expansion is lossy in the direction that was already lost: the seed
/// arrives with eight bits per channel whatever this writes it as. What it buys
/// is that the seed lands as the attachment's texels instead of as a quarter of
/// them, and the *rendering* on top of it keeps the full range.
pub fn expand_rgba8_to_texel(
    layout: TexelLayout,
    src_rgba: &[u8],
    pixels: u32,
    dst: &mut [u8],
) -> bool {
    let px = pixels as usize;
    let Some(dst_len) = px.checked_mul(layout.bytes_per_texel() as usize) else {
        return false;
    };
    let Some(src_len) = px.checked_mul(RGBA8_BPP as usize) else {
        return false;
    };
    if src_rgba.len() < src_len || dst.len() < dst_len {
        return false;
    }
    match layout {
        TexelLayout::Rgba8 => dst[..src_len].copy_from_slice(&src_rgba[..src_len]),
        TexelLayout::Bgra8 => {
            for i in 0..px {
                let (s, d) = (i * 4, i * 4);
                dst[d] = src_rgba[s + COMPONENT_B];
                dst[d + 1] = src_rgba[s + COMPONENT_G];
                dst[d + 2] = src_rgba[s + COMPONENT_R];
                dst[d + 3] = src_rgba[s + COMPONENT_A];
            }
        }
        TexelLayout::Rgba16Float => {
            let lut = unorm8_to_f16_lut();
            for i in 0..px {
                let (s, d) = (i * RGBA8_BPP as usize, i * RGBA16F_BPP as usize);
                for c in 0..4 {
                    st16(
                        &mut dst[d + c * 2..d + c * 2 + 2],
                        lut[src_rgba[s + c] as usize],
                    );
                }
            }
        }
        // The one- and two-channel float attachments. A seed is semantic RGBA8,
        // so the channels the destination does not have are dropped — which is
        // what a Metal shader writing `float4` into an `R16Float` attachment
        // does too, and the inverse of `narrow_texel_to_rgba8` filling the
        // missing channels with the same zeros and opaque alpha a shader reads
        // back from them.
        TexelLayout::R16Float | TexelLayout::Rg16Float => {
            let lut = unorm8_to_f16_lut();
            let chans = (layout.bytes_per_texel() / 2) as usize;
            for i in 0..px {
                let (s, d) = (
                    i * RGBA8_BPP as usize,
                    i * layout.bytes_per_texel() as usize,
                );
                for c in 0..chans {
                    st16(
                        &mut dst[d + c * 2..d + c * 2 + 2],
                        lut[src_rgba[s + c] as usize],
                    );
                }
            }
        }
        // The single-channel eight-bit attachment — a coverage, mask or shadow
        // layer, which is what a one-channel unorm target almost always is.
        // Drops G, B and A for the same reason the float arms above do: a seed
        // is semantic RGBA8 and the destination has one channel to put it in.
        TexelLayout::R8 => {
            for (i, d) in dst[..px].iter_mut().enumerate() {
                *d = src_rgba[i * RGBA8_BPP as usize + COMPONENT_R];
            }
        }
        // The packed ten-bit colour word, which a guest was measured rendering
        // into: an `'l10r'` IOSurface, named `BGR10A2Unorm` by
        // `runtime::objects::iosurface_pixel_format_to_mtl`. A seed is semantic
        // RGBA8, so each channel is widened into the bits it gains rather than
        // shifted into place — see [`rgba8_to_bgr10a2_word`].
        TexelLayout::Bgr10a2Unorm => {
            for i in 0..px {
                let (s, d) = (i * RGBA8_BPP as usize, i * RGBA8_BPP as usize);
                let mut rgba = [0u8; COMPONENT_COUNT];
                rgba.copy_from_slice(&src_rgba[s..s + COMPONENT_COUNT]);
                dst[d..d + 4].copy_from_slice(&rgba8_to_bgr10a2_word(rgba).to_le_bytes());
            }
        }
        // Not colour-attachment layouts this device creates a render target at,
        // so a seed for one is a wiring error rather than a conversion. What
        // decides that is `render_target_bpp`, whose doc states the obligation
        // in the other direction: a format admitted there and refused here is a
        // render target that loses its seed.
        //
        // `Rg8` sits here rather than beside `R8` above deliberately: no guest
        // has been observed declaring a two-channel eight-bit render target, and
        // the arm is trivial to add when one is. Admitting a layout costs three
        // conversions and a census line, so they are added on measurement.
        //
        // The two remaining packed 32-bit colour layouts are here because
        // `render_target_bpp` does not admit their formats: no guest has been
        // observed declaring a render target in one, so there is no seed to
        // convert and an arm would be a conversion written against nothing. Add
        // both halves together if one is ever measured — the obligation
        // `render_target_bpp` states runs in that direction, and `Bgr10a2Unorm`
        // is the member that has now been measured and moved out.
        TexelLayout::Rg8
        | TexelLayout::R32Float
        | TexelLayout::R16Unorm
        | TexelLayout::Rg16Unorm
        | TexelLayout::Rg16Uint
        | TexelLayout::Rgba32Float
        | TexelLayout::Rgba16Unorm
        | TexelLayout::Rgb10a2Unorm
        | TexelLayout::Rg11b10Float => return false,
        // A BC layout is never a render target, so it never has a `Load` seed to
        // widen. `render_target_bpp` has no arm for any BC format, which is what
        // makes that a fact rather than an expectation.
        TexelLayout::Bc1Rgba
        | TexelLayout::Bc2Rgba
        | TexelLayout::Bc3Rgba
        | TexelLayout::Bc4RUnorm
        | TexelLayout::Bc4RSnorm
        | TexelLayout::Bc5RgUnorm
        | TexelLayout::Bc5RgSnorm
        | TexelLayout::Bc6hRgbFloat
        | TexelLayout::Bc6hRgbUfloat
        | TexelLayout::Bc7Rgba => return false,
    }
    true
}

/// Read `layout`'s texels back as a semantic-RGBA8 frame — the inverse of
/// [`expand_rgba8_to_texel`], for the host readback rails.
///
/// Those rails hand their bytes to consumers that only speak RGBA8, so a
/// resident wider than four bytes a texel has to be narrowed on the way out or
/// refused. Narrowing is the right answer and refusing is not: this is the
/// *fallback* a Store takes when the GPU could not write the frame into guest
/// pages directly, and a refusal there loses the frame outright, where before
/// render targets could be wide the same frame was merely quantized. Quantized
/// is what this returns, and it is strictly what the eight-bit resident used to
/// produce.
///
/// `false` for a layout with no defined reading, so the caller declines by name.
/// Exhaustive over [`TexelLayout`] for the same reason as its inverse.
pub fn narrow_texel_to_rgba8(
    layout: TexelLayout,
    src: &[u8],
    pixels: u32,
    dst_rgba: &mut [u8],
) -> bool {
    let px = pixels as usize;
    let Some(src_len) = px.checked_mul(layout.bytes_per_texel() as usize) else {
        return false;
    };
    let Some(dst_len) = px.checked_mul(RGBA8_BPP as usize) else {
        return false;
    };
    if src.len() < src_len || dst_rgba.len() < dst_len {
        return false;
    }
    match layout {
        TexelLayout::Rgba8 => dst_rgba[..dst_len].copy_from_slice(&src[..dst_len]),
        TexelLayout::Bgra8 => {
            for i in 0..px {
                let (s, d) = (i * 4, i * 4);
                dst_rgba[d] = src[s + COMPONENT_B];
                dst_rgba[d + 1] = src[s + COMPONENT_G];
                dst_rgba[d + 2] = src[s + COMPONENT_R];
                dst_rgba[d + 3] = src[s + COMPONENT_A];
            }
        }
        TexelLayout::Rgba16Float => {
            let lut = f16_to_unorm8_lut();
            for i in 0..px {
                let (s, d) = (i * RGBA16F_BPP as usize, i * RGBA8_BPP as usize);
                for c in 0..4 {
                    let h = u16::from_le_bytes([src[s + c * 2], src[s + c * 2 + 1]]);
                    dst_rgba[d + c] = lut[h as usize];
                }
            }
        }
        // The one- and two-channel float attachments, filled out to RGBA8 the
        // way a shader sampling them reads them: the channels the source does
        // not carry are zero, and alpha is opaque. `expand_rgba8_to_texel` is
        // the inverse and drops exactly those channels again.
        TexelLayout::R16Float | TexelLayout::Rg16Float => {
            let lut = f16_to_unorm8_lut();
            let chans = (layout.bytes_per_texel() / 2) as usize;
            for i in 0..px {
                let (s, d) = (
                    i * layout.bytes_per_texel() as usize,
                    i * RGBA8_BPP as usize,
                );
                for c in 0..chans {
                    let h = u16::from_le_bytes([src[s + c * 2], src[s + c * 2 + 1]]);
                    dst_rgba[d + c] = lut[h as usize];
                }
                for c in chans..3 {
                    dst_rgba[d + c] = 0;
                }
                dst_rgba[d + COMPONENT_A] = UNORM8_MAX;
            }
        }
        // The single-channel eight-bit attachment, filled out the way a shader
        // sampling it reads it and the way `texel_to_rgba8`'s `R8_UNORM` arm
        // already does: the channels it does not carry are zero and alpha is
        // opaque. `expand_rgba8_to_texel` is the inverse and drops exactly those.
        TexelLayout::R8 => {
            for (i, &s) in src[..px].iter().enumerate() {
                let d = i * RGBA8_BPP as usize;
                dst_rgba[d + COMPONENT_R] = s;
                dst_rgba[d + COMPONENT_G] = 0;
                dst_rgba[d + COMPONENT_B] = 0;
                dst_rgba[d + COMPONENT_A] = UNORM8_MAX;
            }
        }
        // The packed ten-bit colour word, read back the way a shader sampling
        // it reads it: each channel truncated to its top eight bits and the
        // two-bit alpha replicated out. `expand_rgba8_to_texel` is the inverse.
        // This is the *fallback* rail — a host with no guest-RAM import, where
        // refusing would lose the frame outright rather than quantize it.
        TexelLayout::Bgr10a2Unorm => {
            for i in 0..px {
                let (s, d) = (i * RGBA8_BPP as usize, i * RGBA8_BPP as usize);
                let word = u32::from_le_bytes([src[s], src[s + 1], src[s + 2], src[s + 3]]);
                dst_rgba[d..d + COMPONENT_COUNT].copy_from_slice(&bgr10a2_word_to_rgba8(word));
            }
        }
        TexelLayout::Rg8
        | TexelLayout::R32Float
        | TexelLayout::R16Unorm
        | TexelLayout::Rg16Unorm
        | TexelLayout::Rg16Uint
        | TexelLayout::Rgba32Float
        | TexelLayout::Rgba16Unorm
        | TexelLayout::Rgb10a2Unorm
        | TexelLayout::Rg11b10Float => return false,
        // Nothing reads a BC resident back: there is no BC render target to read
        // back from, and a sampled BC image is never the source of a readback.
        TexelLayout::Bc1Rgba
        | TexelLayout::Bc2Rgba
        | TexelLayout::Bc3Rgba
        | TexelLayout::Bc4RUnorm
        | TexelLayout::Bc4RSnorm
        | TexelLayout::Bc5RgUnorm
        | TexelLayout::Bc5RgSnorm
        | TexelLayout::Bc6hRgbFloat
        | TexelLayout::Bc6hRgbUfloat
        | TexelLayout::Bc7Rgba => return false,
    }
    true
}

pub fn texel_to_rgba8(format: u16, src: &[u8]) -> Option<[u8; 4]> {
    let bpp = bytes_per_pixel(format)? as usize;
    if src.len() < bpp {
        return None;
    }
    let mut rgba = [0u8; 4];
    match format {
        MTL_FORMAT_A8_UNORM => {
            rgba[COMPONENT_A] = src[0];
        }
        MTL_FORMAT_R8_UNORM => {
            rgba[COMPONENT_R] = src[0];
            rgba[COMPONENT_A] = UNORM8_MAX;
        }
        MTL_FORMAT_RG8_UNORM => {
            rgba[COMPONENT_R] = src[0];
            rgba[COMPONENT_G] = src[1];
            rgba[COMPONENT_A] = UNORM8_MAX;
        }
        MTL_FORMAT_RGBA8_UNORM | MTL_FORMAT_RGBA8_UNORM_SRGB => {
            rgba.copy_from_slice(&src[..4]);
        }
        MTL_FORMAT_BGRA8_UNORM | MTL_FORMAT_BGRA8_UNORM_SRGB => {
            rgba[COMPONENT_R] = src[2];
            rgba[COMPONENT_G] = src[1];
            rgba[COMPONENT_B] = src[0];
            rgba[COMPONENT_A] = src[3];
        }
        MTL_FORMAT_RGBA16_FLOAT => {
            let lut = f16_to_unorm8_lut();
            rgba[COMPONENT_R] = lut[ld16(&src[0..2]) as usize];
            rgba[COMPONENT_G] = lut[ld16(&src[2..4]) as usize];
            rgba[COMPONENT_B] = lut[ld16(&src[4..6]) as usize];
            rgba[COMPONENT_A] = lut[ld16(&src[6..8]) as usize];
        }
        MTL_FORMAT_RG16_FLOAT => {
            // Two float16 channels → R,G; B has no source (0), A opaque. Mirrors
            // the RGBA16Float LUT path (values clamp to [0,1] through the u8 LUT).
            let lut = f16_to_unorm8_lut();
            rgba[COMPONENT_R] = lut[ld16(&src[0..2]) as usize];
            rgba[COMPONENT_G] = lut[ld16(&src[2..4]) as usize];
            rgba[COMPONENT_A] = UNORM8_MAX;
        }
        _ => return None,
    }
    Some(rgba)
}

/// Whether a semantic RGBA8 colour can be written into one texel of `format`.
///
/// It **probes [`rgba8_to_texel`] itself** rather than listing the formats that
/// have an arm, so the two cannot drift. This is specifically the eight-bit
/// conversion rail's answer, not clear admission: [`solid_clear_image`] carries
/// integer clear values as native texels and deliberately returns an image for
/// [`TexelLayout::Rg16Uint`] while this predicate remains false.
pub fn solid_color_reaches_texel(format: u16) -> bool {
    let Some(bpp) = bytes_per_pixel(format) else {
        return false;
    };
    // Widest texel the contract defines, so the probe never short-slices.
    let mut probe = [0u8; RGBA32_BPP as usize];
    let Some(cell) = probe.get_mut(..bpp as usize) else {
        return false;
    };
    rgba8_to_texel(format, [0, 0, 0, 0], cell)
}

pub fn rgba8_to_texel(format: u16, rgba: [u8; 4], dst: &mut [u8]) -> bool {
    let Some(bpp) = bytes_per_pixel(format) else {
        return false;
    };
    if dst.len() < bpp as usize {
        return false;
    }
    match format {
        MTL_FORMAT_RGBA8_UNORM | MTL_FORMAT_RGBA8_UNORM_SRGB => {
            dst[..4].copy_from_slice(&rgba);
        }
        MTL_FORMAT_BGRA8_UNORM | MTL_FORMAT_BGRA8_UNORM_SRGB => {
            dst[0] = rgba[COMPONENT_B];
            dst[1] = rgba[COMPONENT_G];
            dst[2] = rgba[COMPONENT_R];
            dst[3] = rgba[COMPONENT_A];
        }
        MTL_FORMAT_RGBA16_FLOAT => {
            let lut = unorm8_to_f16_lut();
            st16(&mut dst[0..2], lut[rgba[COMPONENT_R] as usize]);
            st16(&mut dst[2..4], lut[rgba[COMPONENT_G] as usize]);
            st16(&mut dst[4..6], lut[rgba[COMPONENT_B] as usize]);
            st16(&mut dst[6..8], lut[rgba[COMPONENT_A] as usize]);
        }
        MTL_FORMAT_RG16_FLOAT => {
            // R,G → two float16 channels; B,A have no destination (RG16 is
            // 4 bytes). Inverse of the texel_to_rgba8 RG16Float path.
            let lut = unorm8_to_f16_lut();
            st16(&mut dst[0..2], lut[rgba[COMPONENT_R] as usize]);
            st16(&mut dst[2..4], lut[rgba[COMPONENT_G] as usize]);
        }
        MTL_FORMAT_R8_UNORM => {
            // R → the single byte; G,B,A have no destination. The inverse of
            // `texel_to_rgba8`'s `R8_UNORM` arm, which is where the round trip
            // for this format is checked.
            dst[0] = rgba[COMPONENT_R];
        }
        MTL_FORMAT_R16_FLOAT => {
            // R → one float16 channel; G,B,A have no destination (R16 is
            // 2 bytes). One channel count below the RG16Float arm above, and
            // the same LUT.
            //
            // There is deliberately no `texel_to_rgba8` arm in the other
            // direction: this one exists so a renderable format's Store can be
            // written, while [`TexelLayout::has_cpu_loader_arm`] answers `false`
            // for `R16Float` so a sampled bind of one keeps its native rail
            // instead of being quantized to 256 levels on the way in. The two
            // directions are governed separately and nothing here couples them.
            let lut = unorm8_to_f16_lut();
            st16(&mut dst[0..2], lut[rgba[COMPONENT_R] as usize]);
        }
        MTL_FORMAT_BGR10A2_UNORM => {
            // The third of the three rails `render_target_bpp`'s doc obliges a
            // renderable format to satisfy. Lossy in the direction that matters
            // — a seed only ever carries eight bits — which is why
            // `store_texel_order` admits this format so the byte copy is what
            // normally runs.
            dst[..4].copy_from_slice(&rgba8_to_bgr10a2_word(rgba).to_le_bytes());
        }
        _ => return false,
    }
    true
}

fn row_walk_backward(
    src_len: usize,
    src_stride: usize,
    dst_len: usize,
    dst_stride: usize,
    same_base: bool,
) -> bool {
    // Non-overlapping or zero lengths: forward.
    if src_len == 0 || dst_len == 0 {
        return false;
    }
    // We cannot detect true pointer overlap without raw pointers; for Rust
    // slice APIs we only allow in-place when same_base is true (caller asserts
    // src and dst alias the same allocation).
    if !same_base {
        return false;
    }
    dst_stride > src_stride
}

/// Whether [`texel_to_rgba8`] answers for this format by **narrowing** it
/// rather than by rearranging bytes it can carry exactly.
///
/// The float arms go through `f16_to_unorm8_lut`, which does two things the
/// unorm8 arms never do: it **clamps to `[0,1]`** and it **quantises to 256
/// levels**. For a colour that is a small visible error. For a texture whose
/// texels are *not* colours — a lookup table, a coordinate pair, a chain of
/// offsets — it is data loss with no upper bound on the consequence, and it is
/// silent, because the conversion succeeds.
///
/// [`TexelLayout::R16Float`] exists precisely so a single-channel float sampled
/// bind escapes this. Its two- and four-channel siblings have no such layout, so
/// they do not escape it, and the callers that convert a sampled texture are
/// expected to say so — see `runtime::draw::note_sampled_narrowing`.
///
/// Asked of the **guest's** `MTLPixelFormat`, which is the only place the
/// original width is still known: past the conversion every texel is four bytes
/// and nothing downstream can tell a narrowed one from a native one.
pub fn narrows_to_unorm8(format: u16) -> bool {
    matches!(format, MTL_FORMAT_RGBA16_FLOAT | MTL_FORMAT_RG16_FLOAT)
}

pub fn convert_row_to_rgba8(format: u16, src: &[u8], pixels: u32, dst_rgba: &mut [u8]) -> bool {
    convert_row_to_rgba8_ex(format, src, pixels, dst_rgba, false)
}

fn convert_row_to_rgba8_ex(
    format: u16,
    src: &[u8],
    pixels: u32,
    dst_rgba: &mut [u8],
    same_base: bool,
) -> bool {
    if pixels == 0 {
        return true;
    }
    let Some(bpp) = bytes_per_pixel(format) else {
        return false;
    };
    let src_len = match (pixels as u64).checked_mul(bpp as u64) {
        Some(v) => v as usize,
        None => return false,
    };
    let dst_len = match (pixels as u64).checked_mul(RGBA8_BPP as u64) {
        Some(v) => v as usize,
        None => return false,
    };
    if src.len() < src_len || dst_rgba.len() < dst_len {
        return false;
    }
    let backward = row_walk_backward(
        src_len,
        bpp as usize,
        dst_len,
        RGBA8_BPP as usize,
        same_base,
    );

    if format == MTL_FORMAT_RGBA16_FLOAT {
        let lut = f16_to_unorm8_lut();
        let iter: Box<dyn Iterator<Item = u32>> = if backward {
            Box::new((0..pixels).rev())
        } else {
            Box::new(0..pixels)
        };
        for i in iter {
            let sp = (i as usize) * RGBA16F_BPP as usize;
            let dp = (i as usize) * RGBA8_BPP as usize;
            dst_rgba[dp + COMPONENT_R] = lut[ld16(&src[sp..sp + 2]) as usize];
            dst_rgba[dp + COMPONENT_G] = lut[ld16(&src[sp + 2..sp + 4]) as usize];
            dst_rgba[dp + COMPONENT_B] = lut[ld16(&src[sp + 4..sp + 6]) as usize];
            dst_rgba[dp + COMPONENT_A] = lut[ld16(&src[sp + 6..sp + 8]) as usize];
        }
        return true;
    }

    let iter: Box<dyn Iterator<Item = u32>> = if backward {
        Box::new((0..pixels).rev())
    } else {
        Box::new(0..pixels)
    };
    for i in iter {
        let sp = (i as usize) * bpp as usize;
        let dp = (i as usize) * RGBA8_BPP as usize;
        let Some(rgba) = texel_to_rgba8(format, &src[sp..sp + bpp as usize]) else {
            return false;
        };
        dst_rgba[dp..dp + 4].copy_from_slice(&rgba);
    }
    true
}

pub fn convert_rgba8_to_row(format: u16, src_rgba: &[u8], pixels: u32, dst: &mut [u8]) -> bool {
    if pixels == 0 {
        return true;
    }
    let Some(bpp) = bytes_per_pixel(format) else {
        return false;
    };
    let src_len = match (pixels as u64).checked_mul(RGBA8_BPP as u64) {
        Some(v) => v as usize,
        None => return false,
    };
    let dst_len = match (pixels as u64).checked_mul(bpp as u64) {
        Some(v) => v as usize,
        None => return false,
    };
    if src_rgba.len() < src_len || dst.len() < dst_len {
        return false;
    }

    if format == MTL_FORMAT_RGBA16_FLOAT {
        let lut = unorm8_to_f16_lut();
        for i in 0..pixels {
            let sp = (i as usize) * RGBA8_BPP as usize;
            let dp = (i as usize) * RGBA16F_BPP as usize;
            st16(
                &mut dst[dp..dp + 2],
                lut[src_rgba[sp + COMPONENT_R] as usize],
            );
            st16(
                &mut dst[dp + 2..dp + 4],
                lut[src_rgba[sp + COMPONENT_G] as usize],
            );
            st16(
                &mut dst[dp + 4..dp + 6],
                lut[src_rgba[sp + COMPONENT_B] as usize],
            );
            st16(
                &mut dst[dp + 6..dp + 8],
                lut[src_rgba[sp + COMPONENT_A] as usize],
            );
        }
        return true;
    }

    for i in 0..pixels {
        let sp = (i as usize) * RGBA8_BPP as usize;
        let dp = (i as usize) * bpp as usize;
        let mut rgba = [0u8; 4];
        rgba.copy_from_slice(&src_rgba[sp..sp + 4]);
        if !rgba8_to_texel(format, rgba, &mut dst[dp..dp + bpp as usize]) {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// [`TexelLayout::has_cpu_loader_arm`] answers for the loader it names.
    ///
    /// The two are separate spellings of one fact — a `match` over the layouts
    /// and a `match` over the Metal formats — and nothing in the type system
    /// holds them together, so this asks the loader directly rather than
    /// re-listing the answer. A layout that gains a `texel_to_rgba8` arm, or
    /// loses one, fails here instead of silently moving a zero-copy floor.
    ///
    /// The zero-copy sampled floor is the caller that cares: it may only turn a
    /// window away when there is a CPU path to turn it away *to*, so a `true`
    /// here that the loader does not honour is a black sample rather than a
    /// slow one.
    #[test]
    fn the_cpu_loader_arm_predicate_agrees_with_the_loader() {
        // One representative Metal format per layout, and a source buffer wide
        // enough for the widest of them.
        let cases = [
            (TexelLayout::Rgba8, MTL_FORMAT_RGBA8_UNORM),
            (TexelLayout::Bgra8, MTL_FORMAT_BGRA8_UNORM),
            (TexelLayout::R8, MTL_FORMAT_R8_UNORM),
            (TexelLayout::Rg8, MTL_FORMAT_RG8_UNORM),
            (TexelLayout::R16Float, MTL_FORMAT_R16_FLOAT),
            (TexelLayout::R32Float, MTL_FORMAT_R32_FLOAT),
        ];
        let src = [0u8; 8];
        for (layout, mtl) in cases {
            assert_eq!(
                layout.has_cpu_loader_arm(),
                texel_to_rgba8(mtl, &src).is_some(),
                "{layout:?} ({mtl:#x}): the predicate and texel_to_rgba8 disagree"
            );
        }
        // The case the predicate exists to separate: `Rg8` has an arm and is
        // not four-byte colour, so the two questions genuinely differ. Without
        // this the predicate could be `is_four_byte_color` and every assertion
        // above would still pass on the layouts that reached a floor before.
        assert!(TexelLayout::Rg8.has_cpu_loader_arm());
        assert!(!TexelLayout::Rg8.is_four_byte_color());
    }

    /// `Rg8` guest bytes sample the same texel through the CPU loader and
    /// through a native `R8G8_UNORM` image.
    ///
    /// This is the whole correctness argument for admitting `Rg8` to the
    /// zero-copy linear gather, so it is asserted rather than described. The
    /// CPU rail expands two guest bytes to `(r, g, 0, 255)`; Vulkan samples
    /// `R8G8_UNORM` as `(r, g, 0, 1)`, which is that texel in unorm. Same for
    /// `R8`. If the loader ever stopped writing an opaque alpha or started
    /// filling blue, the gather and the fallback would paint differently
    /// depending only on the window's size relative to the floor — the worst
    /// shape a divergence can take, because it reproduces intermittently.
    #[test]
    fn the_two_byte_and_one_byte_layouts_sample_as_the_native_image_does() {
        let rg = texel_to_rgba8(MTL_FORMAT_RG8_UNORM, &[0x11, 0x22]).expect("Rg8 has an arm");
        assert_eq!(
            rg,
            [0x11, 0x22, UNORM8_MIN, UNORM8_MAX],
            "R8G8_UNORM samples (r, g, 0, 1)"
        );
        let r = texel_to_rgba8(MTL_FORMAT_R8_UNORM, &[0x33]).expect("R8 has an arm");
        assert_eq!(
            r,
            [0x33, UNORM8_MIN, UNORM8_MIN, UNORM8_MAX],
            "R8_UNORM samples (r, 0, 0, 1)"
        );
    }

    /// Every byte of the buffer is the colour, at a geometry with no square
    /// root and no power of two, so an off-by-one in either axis shows.
    ///
    /// The property the two deleted copies could not state: length and fill are
    /// one derivation, so "as long as `w`×`h`×4" and "filled end to end" cannot
    /// come apart. A fill that counted texts separately would satisfy the first
    /// assertion and fail the second the moment the two expressions disagreed.
    #[test]
    fn a_solid_clear_fills_every_texel_it_allocates() {
        let img = solid_rgba8(37, 11, &[1.0, 0.0, 0.5, 1.0]);
        assert_eq!(img.len(), 37 * 11 * RGBA8_BPP as usize);
        let expect = [UNORM8_MAX, UNORM8_MIN, f64_to_unorm8(0.5), UNORM8_MAX];
        assert!(
            img.chunks_exact(RGBA8_BPP as usize).all(|t| t == expect),
            "a texel was left unwritten, so the fill and the length disagree"
        );
        assert_eq!(
            img.chunks_exact(RGBA8_BPP as usize).count(),
            37 * 11,
            "the buffer holds a whole number of texels and exactly w*h of them"
        );
    }

    /// A zero axis is an empty image, not a one-texel one.
    ///
    /// Both callers reach this with a colour attachment's decoded geometry, and
    /// `chunks_exact_mut` over an empty buffer yields nothing — so the zero case
    /// needs no special arm and must not grow one.
    #[test]
    fn a_clear_with_a_zero_axis_is_empty() {
        assert!(solid_rgba8(0, 64, &[1.0; 4]).is_empty());
        assert!(solid_rgba8(64, 0, &[1.0; 4]).is_empty());
        assert!(solid_rgba8(0, 0, &[1.0; 4]).is_empty());
    }

    /// The clear colour travels channel by channel, in RGBA order.
    ///
    /// Pins the ordering against a transposition: four `f64_to_unorm8` calls in
    /// a row are exactly the shape where a swap compiles and looks right.
    #[test]
    fn a_clear_colour_keeps_its_channel_order() {
        let img = solid_rgba8(1, 1, &[0.0, 1.0, 0.0, 0.0]);
        assert_eq!(
            img,
            vec![UNORM8_MIN, UNORM8_MAX, UNORM8_MIN, UNORM8_MIN],
            "only green was asked for"
        );
    }

    #[test]
    fn bytes_per_pixel_matrix() {
        let cases = [
            (MTL_FORMAT_A8_UNORM, 1),
            (MTL_FORMAT_R8_UNORM, 1),
            (MTL_FORMAT_R8_UINT, 1),
            (MTL_FORMAT_RG8_UINT, 2),
            (MTL_FORMAT_R16_FLOAT, 2),
            (MTL_FORMAT_RG8_UNORM, 2),
            (MTL_FORMAT_R32_UINT, 4),
            (MTL_FORMAT_R32_SINT, 4),
            (MTL_FORMAT_R32_FLOAT, 4),
            (MTL_FORMAT_RG16_FLOAT, 4),
            (MTL_FORMAT_RGBA8_UNORM, 4),
            (MTL_FORMAT_RGBA8_UNORM_SRGB, 4),
            (MTL_FORMAT_RGBA8_UINT, 4),
            (MTL_FORMAT_RGBA8_SINT, 4),
            (MTL_FORMAT_BGRA8_UNORM, 4),
            (MTL_FORMAT_BGRA8_UNORM_SRGB, 4),
            (MTL_FORMAT_RGB9E5_FLOAT, 4),
            (MTL_FORMAT_RGBA16_UINT, 8),
            (MTL_FORMAT_RGBA16_FLOAT, 8),
            (MTL_FORMAT_RGBA32_UINT, 16),
            (MTL_FORMAT_RGBA32_FLOAT, 16),
            (MTL_FORMAT_DEPTH16_UNORM, 2),
            (MTL_FORMAT_DEPTH32_FLOAT, 4),
            (MTL_FORMAT_STENCIL8, 1),
            (MTL_FORMAT_DEPTH24_UNORM_STENCIL8, 4),
            (MTL_FORMAT_DEPTH32_FLOAT_STENCIL8, 8),
            (MTL_FORMAT_X32_STENCIL8, 8),
            (MTL_FORMAT_X24_STENCIL8, 4),
        ];
        for (fmt, bpp) in cases {
            assert_eq!(bytes_per_pixel(fmt), Some(bpp));
        }
        assert_eq!(bytes_per_pixel(0xffff), None);
    }

    /// A packed depth-stencil format's texel width is one number, held by two
    /// tables, so they are required to agree.
    ///
    /// [`depth_stencil_packing`] declares `full_bpp` for the cell it describes
    /// the interior of, and [`bytes_per_pixel`] declares the same width for the
    /// same format; a copy sized by one and laid out by the other is a short
    /// read the moment they differ. Swept over every `u16` rather than listed,
    /// so a format added to either table is covered without anyone adding a line.
    ///
    /// This is also what pins the `X*_Stencil8` arms against being "corrected"
    /// to the stencil plane's own width — that number is `stencil_plane_bpp`,
    /// which this checks is a *field within* the cell rather than the cell.
    #[test]
    fn the_two_tables_agree_on_a_packed_depth_stencil_texel() {
        let mut seen = 0;
        for fmt in 0..=u16::MAX {
            let Some(packing) = depth_stencil_packing(fmt) else {
                continue;
            };
            seen += 1;
            assert_eq!(
                bytes_per_pixel(fmt),
                Some(packing.full_bpp),
                "{fmt:#x}: the pixel table and the packing disagree on the texel width"
            );
            // Every plane named must fit inside the cell it is a plane of.
            assert!(
                packing.stencil_offset + packing.stencil_plane_bpp <= packing.full_bpp,
                "{fmt:#x}: the stencil field runs past the texel"
            );
            assert!(
                packing.depth_offset + packing.depth_plane_bpp <= packing.full_bpp,
                "{fmt:#x}: the depth field runs past the texel"
            );
        }
        assert_eq!(seen, 4, "the packed depth-stencil family is four formats");
    }

    #[test]
    fn blit_aspect_bpp_depth_stencil() {
        // Pure depth + depth option.
        assert_eq!(
            blit_aspect_bytes_per_pixel(MTL_FORMAT_DEPTH32_FLOAT, BlitAspect::Depth),
            Some(4)
        );
        assert_eq!(
            blit_aspect_bytes_per_pixel(MTL_FORMAT_DEPTH32_FLOAT, BlitAspect::Full),
            Some(4)
        );
        // Pure depth cannot take stencil option.
        assert_eq!(
            blit_aspect_bytes_per_pixel(MTL_FORMAT_DEPTH32_FLOAT, BlitAspect::Stencil),
            None
        );
        // Pure stencil.
        assert_eq!(
            blit_aspect_bytes_per_pixel(MTL_FORMAT_STENCIL8, BlitAspect::Stencil),
            Some(1)
        );
        // Combined: depth plane 4 B, stencil 1 B, full = packing full_bpp.
        assert_eq!(
            blit_aspect_bytes_per_pixel(MTL_FORMAT_DEPTH32_FLOAT_STENCIL8, BlitAspect::Depth),
            Some(4)
        );
        assert_eq!(
            blit_aspect_bytes_per_pixel(MTL_FORMAT_DEPTH32_FLOAT_STENCIL8, BlitAspect::Stencil),
            Some(1)
        );
        assert_eq!(
            blit_aspect_bytes_per_pixel(MTL_FORMAT_DEPTH32_FLOAT_STENCIL8, BlitAspect::Full),
            Some(8)
        );
        assert!(blit_aspect_needs_repack(
            MTL_FORMAT_DEPTH32_FLOAT_STENCIL8,
            BlitAspect::Depth
        ));
        assert!(!blit_aspect_needs_repack(
            MTL_FORMAT_DEPTH32_FLOAT,
            BlitAspect::Depth
        ));
        // Color cannot take DS options.
        assert_eq!(
            blit_aspect_bytes_per_pixel(MTL_FORMAT_BGRA8_UNORM, BlitAspect::Depth),
            None
        );
    }

    /// Exactly the two `_SRGB` wire values carry the transfer function, and the
    /// class lookups fold each onto the linear sibling's byte layout. Both
    /// halves matter: the fold is what makes a class usable, `is_srgb` is what
    /// keeps the fold from being a silent loss.
    #[test]
    fn srgb_is_named_beside_the_class_that_folds_it() {
        assert!(is_srgb(MTL_FORMAT_RGBA8_UNORM_SRGB));
        assert!(is_srgb(MTL_FORMAT_BGRA8_UNORM_SRGB));
        for fmt in [
            MTL_FORMAT_RGBA8_UNORM,
            MTL_FORMAT_BGRA8_UNORM,
            MTL_FORMAT_A8_UNORM,
            MTL_FORMAT_RGBA16_FLOAT,
            MTL_FORMAT_DEPTH32_FLOAT,
            0xffff,
        ] {
            assert!(!is_srgb(fmt), "{fmt:#x}");
        }
        assert_eq!(
            sampled_class(MTL_FORMAT_RGBA8_UNORM_SRGB),
            sampled_class(MTL_FORMAT_RGBA8_UNORM)
        );
        assert_eq!(
            sampled_class(MTL_FORMAT_BGRA8_UNORM_SRGB),
            sampled_class(MTL_FORMAT_BGRA8_UNORM)
        );
        // The render-target rail does NOT fold the qualifier, and `is_srgb` is
        // where it is kept. `render_target_bpp` deliberately cannot say: a
        // storage width is the same eight bits either way, and the enum that
        // used to carry both the width and an sRGB-qualified variant name was
        // read by nobody, so the qualifier only ever came from here.
        assert!(is_srgb(MTL_FORMAT_RGBA8_UNORM_SRGB) != is_srgb(MTL_FORMAT_RGBA8_UNORM));
        assert_eq!(
            render_target_bpp(MTL_FORMAT_RGBA8_UNORM_SRGB),
            render_target_bpp(MTL_FORMAT_RGBA8_UNORM),
            "the sRGB qualifier does not change how wide a texel is"
        );
    }

    #[test]
    fn sampled_and_storage() {
        assert_eq!(
            sampled_class(MTL_FORMAT_A8_UNORM),
            Some(SampledClass::A8Unorm)
        );
        assert_eq!(sampled_class(MTL_FORMAT_R16_FLOAT), None);
        assert_eq!(
            storage_selector(MTL_FORMAT_R8_UNORM),
            Some(StorageImageSelector::R8Unorm)
        );
        assert_eq!(storage_selector(MTL_FORMAT_A8_UNORM), None);
        // R32Uint is storage-capable (specialized to the R32ui storage path);
        // its single-channel sint/float siblings are not.
        assert_eq!(
            storage_selector(MTL_FORMAT_R32_UINT),
            Some(StorageImageSelector::R32Uint)
        );
        assert_eq!(storage_selector(MTL_FORMAT_R32_SINT), None);
        assert_eq!(storage_selector(MTL_FORMAT_R32_FLOAT), None);
        assert_eq!(render_target_bpp(MTL_FORMAT_BGRA8_UNORM), Some(4));
        assert_eq!(render_target_bpp(MTL_FORMAT_RGBA8_UNORM), Some(4));
        assert_eq!(render_target_bpp(MTL_FORMAT_RGBA8_UNORM_SRGB), Some(4));
        assert_eq!(render_target_bpp(MTL_FORMAT_BGRA8_UNORM_SRGB), Some(4));
        assert_eq!(render_target_bpp(MTL_FORMAT_RGBA16_FLOAT), Some(8));
        assert_eq!(render_target_bpp(MTL_FORMAT_RG16_FLOAT), Some(4));
        // The two single-channel members, both added on a macos-26 reading of
        // `rt_resolve reason=rt_linear_format`: a half-float blur/backdrop
        // intermediate and an eight-bit coverage/mask layer.
        assert_eq!(render_target_bpp(MTL_FORMAT_R16_FLOAT), Some(R16F_BPP));
        assert_eq!(render_target_bpp(MTL_FORMAT_R8_UNORM), Some(R8_BPP));
        // `RG16_UINT` is admitted on a macos-15 measurement, so "integer" is no
        // longer the reason a format is out — a family resemblance to an
        // admitted member still is not a reason to be in. `RGBA8_UINT` has
        // never been observed as a guest colour attachment and stays out.
        assert_eq!(render_target_bpp(MTL_FORMAT_RGBA8_UINT), None);
        assert_eq!(render_target_bpp(MTL_FORMAT_A8_UNORM), None);
        // Sampled but not renderable, so "has a layout" is not "is a colour
        // attachment" — the distinction `sampled_pixels` and this table each own
        // one half of.
        assert_eq!(render_target_bpp(MTL_FORMAT_R32_FLOAT), None);
        assert_eq!(render_target_bpp(MTL_FORMAT_RG8_UNORM), None);
    }

    /// A linear `RG16Uint` colour attachment resolves, and its width is the one
    /// the widths table already held.
    ///
    /// The measurement: on a macos-15 x86/Vulkan boot the guest declares linear
    /// `RG16Uint` colour attachments, and every pass naming one was refused at
    /// `draw::render_target`'s `rt_linear_format` rung — seven dropped clears
    /// and thirty-two unresolved MRT slots in one boot. The format was absent
    /// from this crate entirely: not a constant, not a width, not a layout.
    #[test]
    fn an_integer_colour_target_is_admitted_at_the_width_the_table_already_knew() {
        assert_eq!(MTL_FORMAT_RG16_UINT, 63, "MTLPixelFormatRG16Uint");
        assert_eq!(bytes_per_pixel(MTL_FORMAT_RG16_UINT), Some(RG16_BPP));
        assert_eq!(render_target_bpp(MTL_FORMAT_RG16_UINT), Some(RG16_BPP));
        // The family shares a width, which is why the widths table groups it.
        assert_eq!(
            bytes_per_pixel(MTL_FORMAT_RG16_UINT),
            bytes_per_pixel(MTL_FORMAT_RG16_UNORM)
        );
        assert_eq!(
            bytes_per_pixel(MTL_FORMAT_RG16_SINT),
            bytes_per_pixel(MTL_FORMAT_RG16_FLOAT)
        );
    }

    /// Admission never invents a width: it takes the one [`bytes_per_pixel`]
    /// holds.
    ///
    /// This is the relation that replaced a hand-written per-member width list.
    /// It is the only thing standing between the two spellings now, so it walks
    /// the whole `u16` space rather than a list a new member could be left off.
    #[test]
    fn every_admitted_render_target_takes_its_width_from_the_one_widths_table() {
        let mut admitted = 0;
        for format in 0..=u16::MAX {
            let Some(bpp) = render_target_bpp(format) else {
                continue;
            };
            admitted += 1;
            assert_eq!(
                Some(bpp),
                bytes_per_pixel(format),
                "{format:#x} is admitted as a render target at a width \
                 `bytes_per_pixel` does not agree with"
            );
        }
        // A guard on the walk itself: an admission set that silently emptied
        // would satisfy every assertion above.
        assert_eq!(admitted, 10, "the admitted colour render target formats");
    }

    /// An integer texel has no semantic eight-bit solid colour.
    ///
    /// The predicate probes the converter rather than listing formats, so this
    /// asserts the behaviour the clear path actually depends on: a `false` for
    /// the integer target and a `true` for every colour order that has an arm.
    #[test]
    fn an_integer_clear_cannot_take_the_semantic_rgba8_conversion_rail() {
        assert!(!solid_color_reaches_texel(MTL_FORMAT_RG16_UINT));
        // Its own family's normalized and float members are unaffected: they go
        // through the same rail and keep their arms.
        assert!(solid_color_reaches_texel(MTL_FORMAT_RG16_FLOAT));
        assert!(solid_color_reaches_texel(MTL_FORMAT_BGRA8_UNORM));
        assert!(solid_color_reaches_texel(MTL_FORMAT_RGBA8_UNORM));
        assert!(solid_color_reaches_texel(MTL_FORMAT_RGBA16_FLOAT));
        // And it stays in step with the converter it probes, which is the
        // whole point of probing rather than listing.
        let mut cell = [0u8; RGBA32_BPP as usize];
        for format in [
            MTL_FORMAT_RG16_UINT,
            MTL_FORMAT_RG16_FLOAT,
            MTL_FORMAT_BGRA8_UNORM,
            MTL_FORMAT_R8_UNORM,
        ] {
            let bpp = bytes_per_pixel(format).expect("a width") as usize;
            assert_eq!(
                solid_color_reaches_texel(format),
                rgba8_to_texel(format, [1, 2, 3, 4], &mut cell[..bpp]),
                "{format:#x}: the probe and the converter disagree"
            );
        }
    }

    /// Every admitted target can publish a clear-only result, and the integer
    /// member keeps its native component values all the way to the bytes.
    #[test]
    fn every_admitted_render_target_has_a_format_aware_clear_image() {
        let clear = [1.0, 258.0, 65_535.0, 0.0];
        let mut admitted = 0;
        for format in 0..=u16::MAX {
            if render_target_bpp(format).is_none() {
                continue;
            }
            admitted += 1;
            let image = solid_clear_image(format, 2, 2, &clear)
                .unwrap_or_else(|| panic!("render target {format:#x} cannot publish its clear"));
            assert_eq!(image.pixels().len(), image.row_bytes() as usize * 2);
        }
        assert_eq!(admitted, 10, "the admitted colour render target formats");

        let integer = solid_clear_image(MTL_FORMAT_RG16_UINT, 2, 2, &clear)
            .expect("RG16Uint is an admitted render target");
        assert_eq!(integer.encoding(), ClearImageEncoding::Native);
        assert_eq!(integer.row_bytes(), 2 * RG16_BPP);
        assert_eq!(
            integer.pixels(),
            [1u16.to_le_bytes(), 258u16.to_le_bytes()]
                .concat()
                .repeat(4),
            "the two uint16 channels are values, not float or unorm bit patterns"
        );

        let narrowed =
            solid_clear_image(MTL_FORMAT_RG16_UINT, 1, 1, &[65_536.0, 65_537.0, 0.0, 0.0])
                .expect("RG16Uint clear with values wider than a channel");
        assert_eq!(
            narrowed.pixels(),
            [0u16.to_le_bytes(), 1u16.to_le_bytes()].concat(),
            "the CPU rail must narrow the same uint32 carrier as the GPU clear"
        );
    }

    /// The byte copy must carry the integer target, because nothing else can.
    ///
    /// For every other member of [`store_texel_order`] a missing arm is a
    /// performance bug — the CPU converter serves it instead. For this one the
    /// CPU converter has no arm and must not gain one, so an absent arm here
    /// would be a silent loss of the guest's draw.
    #[test]
    fn the_integer_colour_target_reaches_the_exact_rail_and_no_other() {
        assert_eq!(
            store_texel_order(MTL_FORMAT_RG16_UINT),
            Some(TexelLayout::Rg16Uint)
        );
        assert!(!TexelLayout::Rg16Uint.has_cpu_loader_arm());
        assert!(TexelLayout::Rg16Uint.is_integer());
        assert_eq!(TexelLayout::Rg16Uint.bytes_per_texel(), RG16_BPP);
    }

    /// The two capability masks span two vocabularies, and each is dense in its
    /// own.
    ///
    /// They ask different questions of different sets — creating a colour
    /// attachment, and filtering a sampled texel — and sharing one index space
    /// sized both by the union. This holds each index dense and distinct inside
    /// its own count, which is what `DeviceCapabilitySnapshot` sizes its fields
    /// by, and pins the exclusions that are contract facts rather than
    /// omissions.
    ///
    /// The integer assertions below are the regression pin. This vocabulary was
    /// briefly the *blendable* layouts, which put `Rg16Uint` in neither mask —
    /// so `engine::render_target_layout_supported` had no bit to read for it,
    /// answered `false`, and every `RG16Uint` render target was built at the
    /// neutral eight-bit format and then lost by both Store arms. An integer
    /// layout must be in the render-target mask and out of the filter one.
    #[test]
    fn each_capability_mask_is_dense_in_its_own_vocabulary() {
        for (mask, count) in [
            (
                CapabilityMask::RenderTarget,
                TexelLayout::RENDER_TARGET_COUNT,
            ),
            (CapabilityMask::SampledFilter, TexelLayout::FILTER_COUNT),
        ] {
            let mut seen = Vec::new();
            for layout in TexelLayout::ALL {
                let index = match mask {
                    CapabilityMask::RenderTarget => layout.render_target_index(),
                    CapabilityMask::SampledFilter => layout.filter_index(),
                };
                let Some(i) = index else { continue };
                assert!(i < count, "{layout:?} indexes past its own count");
                assert!(!seen.contains(&i), "{layout:?} reuses index {i}");
                seen.push(i);
            }
            assert_eq!(seen.len(), count);
        }

        // An integer layout is rendered into and never filtered. Both halves
        // matter: the first is the bit that was missing, and the second is why
        // it cannot simply ride the other mask.
        assert!(
            TexelLayout::Rg16Uint.is_render_target_layout(),
            "an integer colour attachment must carry a render-target bit, or              the resident falls back to eight bits and both Stores refuse it"
        );
        assert!(TexelLayout::Rg16Uint.render_target_index().is_some());
        assert!(!TexelLayout::Rg16Uint.needs_sampled_filter_query());
        assert_eq!(TexelLayout::Rg16Uint.filter_index(), None);

        // Block-compressed layouts are in neither, for two different reasons —
        // no host renders into one, and their linear filtering is mandated.
        assert!(!TexelLayout::Bc7Rgba.is_render_target_layout());
        assert_eq!(TexelLayout::Bc7Rgba.render_target_index(), None);
        assert!(!TexelLayout::Bc7Rgba.needs_sampled_filter_query());

        // The vocabularies genuinely differ, which is the whole point: a
        // four-channel float is sampled and filtered but never rendered into,
        // and a normalized colour order is both.
        assert!(TexelLayout::Rgba32Float.needs_sampled_filter_query());
        assert!(!TexelLayout::Rgba32Float.is_render_target_layout());
        assert!(TexelLayout::Bgra8.needs_sampled_filter_query());
        assert!(TexelLayout::Bgra8.is_render_target_layout());
        const { assert!(TexelLayout::RENDER_TARGET_COUNT < TexelLayout::FILTER_COUNT) };
    }

    /// The layout-side render-target set and [`render_target_bpp`] name the
    /// same thing.
    ///
    /// Two spellings of one admission — one over `TexelLayout`, one over
    /// `MTLPixelFormat` — and nothing else compares them. A format admitted by
    /// one and not the other is a capability bit read for a layout this device
    /// never renders into, or withheld from one it does.
    #[test]
    fn the_two_render_target_vocabularies_name_the_same_layouts() {
        let mut from_formats: Vec<TexelLayout> = Vec::new();
        for format in 0..=u16::MAX {
            if render_target_bpp(format).is_none() {
                continue;
            }
            // Exhaustive on the sampled classes on purpose: a new one must say
            // which layout it is before a renderable format can reach this
            // vocabulary.
            let from_class = sampled_class(format).map(|class| match class {
                SampledClass::Rgba8Unorm => TexelLayout::Rgba8,
                SampledClass::Bgra8Unorm => TexelLayout::Bgra8,
                SampledClass::A8Unorm | SampledClass::R8Unorm => TexelLayout::R8,
                SampledClass::Rg8Unorm => TexelLayout::Rg8,
                SampledClass::Rgba16Float => TexelLayout::Rgba16Float,
                SampledClass::Rg16Float => TexelLayout::Rg16Float,
                SampledClass::Bgr10a2Unorm => TexelLayout::Bgr10a2Unorm,
                SampledClass::Rg16Uint => TexelLayout::Rg16Uint,
                SampledClass::Rgba32Float => TexelLayout::Rgba32Float,
            });
            // Renderable, single-channel, and named by neither table above —
            // admitted for macOS 26's blur intermediate.
            let single_channel_float =
                (format == MTL_FORMAT_R16_FLOAT).then_some(TexelLayout::R16Float);
            let layout = store_texel_order(format)
                .or(from_class)
                .or(single_channel_float)
                .unwrap_or_else(|| panic!("{format:#x} is renderable with no layout"));
            if !from_formats.contains(&layout) {
                from_formats.push(layout);
            }
        }
        for layout in &from_formats {
            assert!(
                layout.is_render_target_layout(),
                "{layout:?} is renderable by format but not by layout"
            );
        }
        for layout in TexelLayout::ALL {
            if layout.is_render_target_layout() {
                assert!(
                    from_formats.contains(layout),
                    "{layout:?} claims to be a render target but no admitted format maps to it"
                );
            }
        }
    }

    /// RG16Float MRT slots (vibrancy UI tile masks) must admit as color RTs so
    /// `mrt_draw_request` no longer drops the whole pass. Two channels survive
    /// the RGBA8-intermediate round trip; B has no source and A is opaque.
    #[test]
    fn rg16float_render_target_roundtrips_two_channels() {
        assert_eq!(render_target_bpp(MTL_FORMAT_RG16_FLOAT), Some(4));
        let w = 16u32;
        let mut rgba = vec![0u8; (w as usize) * 4];
        for i in 0..(w as usize) {
            rgba[i * 4] = 40; // R
            rgba[i * 4 + 1] = 90; // G
            rgba[i * 4 + 2] = 200; // B (dropped by RG16)
            rgba[i * 4 + 3] = 128; // A (dropped by RG16)
        }
        let tight = tight_row_bytes(w, MTL_FORMAT_RG16_FLOAT).unwrap();
        assert_eq!(tight, w * 4);
        let mut native = vec![0u8; tight as usize];
        assert!(convert_rgba8_to_row(
            MTL_FORMAT_RG16_FLOAT,
            &rgba,
            w,
            &mut native
        ));
        let mut back = vec![0u8; (w as usize) * 4];
        assert!(convert_row_to_rgba8(
            MTL_FORMAT_RG16_FLOAT,
            &native,
            w,
            &mut back
        ));
        // R,G round-trip through the u8→f16→u8 LUT; B has no source (0); A opaque.
        assert_eq!(back[0], 40);
        assert_eq!(back[1], 90);
        assert_eq!(back[2], 0);
        assert_eq!(back[3], 255);
    }

    /// A single-channel half-float render target survives the two rails a
    /// render target actually uses, in both directions.
    ///
    /// macOS 26 renders into one — a linear GVA blur/backdrop intermediate at
    /// `fmt=0x19` — and this device refused it three times a driven boot as
    /// `rt_resolve reason=rt_linear_format`. The refusal was recorded for a
    /// while as a missing [`sampled_class`], which it was not: `sampled_pixels`
    /// has carried `R16_FLOAT` as a native sampled layout throughout. The
    /// missing piece was the CPU Store converter's [`rgba8_to_texel`] arm, so
    /// the format would have rendered and then lost every frame on a host with
    /// no guest-RAM import.
    ///
    /// The round trip is deliberately *not* `convert_row_to_rgba8`: there is no
    /// [`texel_to_rgba8`] arm for `R16_FLOAT` and there should not be, because
    /// [`TexelLayout::has_cpu_loader_arm`] answers `false` for it so a sampled
    /// bind keeps the native rail. The readback direction a render target uses
    /// is the layout-level [`narrow_texel_to_rgba8`], which is what this asks.
    #[test]
    fn an_r16float_render_target_survives_the_store_and_readback_rails() {
        assert_eq!(render_target_bpp(MTL_FORMAT_R16_FLOAT), Some(R16F_BPP));
        // The claim the fail-log reading rests on: the sampler was never the
        // blocker, so admitting the target does not create a write-only one.
        assert!(
            TexelLayout::R16Float.bytes_per_texel() == R16F_BPP,
            "the render-target width and the sampled layout must be one texel"
        );

        let w = 16u32;
        let mut rgba = vec![0u8; (w as usize) * RGBA8_BPP as usize];
        for i in 0..(w as usize) {
            rgba[i * 4] = 40; // R — the only channel R16Float carries
            rgba[i * 4 + 1] = 90; // G, dropped
            rgba[i * 4 + 2] = 200; // B, dropped
            rgba[i * 4 + 3] = 128; // A, dropped
        }

        // Rail one: the synchronous Store's row converter, which is the arm
        // that was missing. Without it this returns false and the guest loses
        // the frame.
        let tight = tight_row_bytes(w, MTL_FORMAT_R16_FLOAT).unwrap();
        assert_eq!(tight, w * R16F_BPP);
        let mut native = vec![0u8; tight as usize];
        assert!(
            convert_rgba8_to_row(MTL_FORMAT_R16_FLOAT, &rgba, w, &mut native),
            "the CPU Store converter cannot write an admitted render target"
        );

        // Rail two: the readback rails' narrow, over the same bytes.
        let mut back = vec![0u8; (w as usize) * RGBA8_BPP as usize];
        assert!(narrow_texel_to_rgba8(
            TexelLayout::R16Float,
            &native,
            w,
            &mut back
        ));
        // R round-trips through the u8→f16→u8 LUT; G and B have no source, and
        // alpha is opaque — the way a shader sampling one channel reads it.
        assert_eq!(back[0], 40);
        assert_eq!(back[1], 0);
        assert_eq!(back[2], 0);
        assert_eq!(back[3], UNORM8_MAX);

        // The CPU `Load` seed, the third obligation a renderable format owes.
        let mut seed = vec![0u8; tight as usize];
        assert!(expand_rgba8_to_texel(
            TexelLayout::R16Float,
            &rgba,
            w,
            &mut seed
        ));
        assert_eq!(seed, native, "the seed and the Store must write one texel");
    }

    /// A single-channel eight-bit render target survives all three CPU rails.
    ///
    /// macOS 26 renders into one — a linear GVA coverage/mask layer at
    /// `fmt=0xa` — and this device refused it once a driven boot as `rt_resolve
    /// reason=rt_linear_format`, alongside the `R16_FLOAT` refusals the test
    /// above covers.
    ///
    /// This one needed three arms rather than one, and that is the point of
    /// testing it separately: a one-byte texel had never been a render target,
    /// so `narrow_texel_to_rgba8`, `expand_rgba8_to_texel` and the CPU Store's
    /// `rgba8_to_texel` all refused it, each in a different function. Unlike
    /// `R16_FLOAT` this format does have a [`texel_to_rgba8`] arm already
    /// (`has_cpu_loader_arm` is true for `R8`), so the round trip can be closed
    /// through the Metal-format converters as well as the layout ones.
    #[test]
    fn an_r8unorm_render_target_survives_all_three_cpu_rails() {
        assert_eq!(render_target_bpp(MTL_FORMAT_R8_UNORM), Some(R8_BPP));

        let w = 16u32;
        let mut rgba = vec![0u8; (w as usize) * RGBA8_BPP as usize];
        for i in 0..(w as usize) {
            rgba[i * 4] = 77; // R — the only channel R8Unorm carries
            rgba[i * 4 + 1] = 90; // G, dropped
            rgba[i * 4 + 2] = 200; // B, dropped
            rgba[i * 4 + 3] = 128; // A, dropped
        }

        // Rail one: the synchronous Store's row converter.
        let tight = tight_row_bytes(w, MTL_FORMAT_R8_UNORM).unwrap();
        assert_eq!(tight, w * R8_BPP);
        let mut native = vec![0u8; tight as usize];
        assert!(
            convert_rgba8_to_row(MTL_FORMAT_R8_UNORM, &rgba, w, &mut native),
            "the CPU Store converter cannot write an admitted render target"
        );
        assert!(native.iter().all(|&b| b == 77));

        // Rail two: the readback narrow. One channel out, the rest filled the
        // way a shader sampling it reads them.
        let mut back = vec![0u8; (w as usize) * RGBA8_BPP as usize];
        assert!(narrow_texel_to_rgba8(
            TexelLayout::R8,
            &native,
            w,
            &mut back
        ));
        assert_eq!(back[0], 77);
        assert_eq!(back[1], 0);
        assert_eq!(back[2], 0);
        assert_eq!(back[3], UNORM8_MAX);

        // Rail three: the CPU `Load` seed, which must write the same texel the
        // Store does or a seeded pass and a stored one disagree.
        let mut seed = vec![0u8; tight as usize];
        assert!(expand_rgba8_to_texel(TexelLayout::R8, &rgba, w, &mut seed));
        assert_eq!(seed, native, "the seed and the Store must write one texel");

        // And the Metal-format loader agrees with the layout narrow, which is
        // available for this format and was not for `R16_FLOAT`.
        assert_eq!(
            texel_to_rgba8(MTL_FORMAT_R8_UNORM, &native[..1]),
            Some([77, 0, 0, UNORM8_MAX])
        );
    }

    /// Metal color-renderable 8-bit + f16 set used as Reims VGPU pass attachments.
    /// Bring-up only admitted BGRA8/RGBA16F (compositor FBs); apps use RGBA8.
    #[test]
    fn color_renderable_formats_admit_app_rts() {
        for (fmt, bpp) in [
            (MTL_FORMAT_RGBA8_UNORM, 4u32),
            (MTL_FORMAT_RGBA8_UNORM_SRGB, 4),
            (MTL_FORMAT_BGRA8_UNORM, 4),
            (MTL_FORMAT_BGRA8_UNORM_SRGB, 4),
            (MTL_FORMAT_RGBA16_FLOAT, 8),
        ] {
            assert_eq!(render_target_bpp(fmt), Some(bpp), "fmt={fmt:#x}");
            // Round-trip tight row for write_gva / mapping store.
            let w = 16u32;
            let mut rgba = vec![0u8; (w as usize) * 4];
            for i in 0..(w as usize) {
                rgba[i * 4] = 10;
                rgba[i * 4 + 1] = 20;
                rgba[i * 4 + 2] = 30;
                rgba[i * 4 + 3] = 255;
            }
            let tight = tight_row_bytes(w, fmt).unwrap();
            let mut native = vec![0u8; tight as usize];
            assert!(
                convert_rgba8_to_row(fmt, &rgba, w, &mut native),
                "convert host RGBA8 → guest fmt={fmt:#x}"
            );
            let mut back = vec![0u8; (w as usize) * 4];
            assert!(
                convert_row_to_rgba8(fmt, &native, w, &mut back),
                "convert guest fmt={fmt:#x} → host RGBA8"
            );
            // 8-bit unorm/sRGB and float16 (via unorm8 LUT) keep the solid color.
            assert_eq!(back[0], 10);
            assert_eq!(back[1], 20);
            assert_eq!(back[2], 30);
            assert_eq!(back[3], 255);
        }
    }

    /// The 128-byte-aligned IOSurface row lives in
    /// [`crate::iosurface_pages::packed_span_estimate`], which is the
    /// one the mapper rail reads. A second `iosurface_row_bytes` here computed
    /// the same rule from its own copy of the alignment and served nothing but
    /// this test. At height 1 the estimate is exactly one aligned row.
    #[test]
    fn rows_and_image_size() {
        use crate::iosurface_pages::packed_span_estimate;
        let bpr = |w, fmt| packed_span_estimate(fmt, w, 1);
        assert_eq!(bpr(200, MTL_FORMAT_BGRA8_UNORM), Some(896));
        assert_eq!(bpr(64, MTL_FORMAT_BGRA8_UNORM), Some(256));
        assert_eq!(bpr(250, MTL_FORMAT_BGRA8_UNORM), Some(1024));
        assert_eq!(bpr(200, MTL_FORMAT_RGBA16_FLOAT), Some(1664));
        assert_eq!(bpr(0, MTL_FORMAT_BGRA8_UNORM), None);
        // Same 4 Bpp packing as BGRA8 → same 128 B aligned row for w=200.
        assert_eq!(bpr(200, MTL_FORMAT_RGBA8_UNORM), Some(896));
        assert_eq!(tight_row_bytes(200, MTL_FORMAT_BGRA8_UNORM), Some(800));
    }

    #[test]
    fn swizzle_and_texels() {
        let plan = swizzle_plan(&[2, 3, 4, 5]).unwrap();
        assert!(swizzle_is_identity(&plan));
        let bgra = [10u8, 20, 30, 40];
        let rgba = texel_to_rgba8(MTL_FORMAT_BGRA8_UNORM, &bgra).unwrap();
        assert_eq!(rgba, [30, 20, 10, 40]);
        let mut out = [0u8; 4];
        assert!(rgba8_to_texel(MTL_FORMAT_BGRA8_UNORM, rgba, &mut out));
        assert_eq!(out, bgra);

        let row = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let mut rgba_row = [0u8; 8];
        assert!(convert_row_to_rgba8(
            MTL_FORMAT_RGBA8_UNORM,
            &row,
            2,
            &mut rgba_row
        ));
        assert_eq!(rgba_row, row);
        let mut back = [0u8; 8];
        assert!(convert_rgba8_to_row(
            MTL_FORMAT_RGBA8_UNORM,
            &rgba_row,
            2,
            &mut back
        ));
        assert_eq!(back, row);

        // Property: the two lookup tables the conversion paths index are
        // exact inverses over every byte. Asserted against the tables
        // themselves, which is what those paths read.
        let to_f16 = unorm8_to_f16_lut();
        let to_u8 = f16_to_unorm8_lut();
        for v in 0u8..=255 {
            assert_eq!(to_u8[to_f16[v as usize] as usize], v);
        }
        let _ = f64_to_unorm8(0.5);
        let _ = f16_to_f32(0x3c00); // 1.0
    }

    /// Every storage-capable format has a texel width, and it is the width the
    /// selector table used to carry beside it.
    ///
    /// This replaces three `debug_assert_eq!(selector_bpp, bpp)` at the
    /// `compute_exec` staging call sites. Those checked the same thing, but only
    /// for formats a running guest happened to bind, and only in a debug build —
    /// a release build compiled them out entirely, so the disagreement they
    /// guarded against would have shipped silently. Sweeping the whole `u16`
    /// space costs microseconds and cannot miss an arm.
    ///
    /// The widths are spelled out rather than derived, because that is the point:
    /// a change to `bytes_per_pixel` that silently redefined a storage format's
    /// stride is exactly what the deleted column existed to catch.
    #[test]
    fn storage_texel_width_matches_the_pixel_table() {
        let expected: &[(u16, u32)] = &[
            (MTL_FORMAT_R8_UNORM, R8_BPP),
            (MTL_FORMAT_R32_UINT, R32_BPP),
            (MTL_FORMAT_RG8_UNORM, RG8_BPP),
            (MTL_FORMAT_R16_FLOAT, R16F_BPP),
            (MTL_FORMAT_RG16_FLOAT, RG16F_BPP),
            (MTL_FORMAT_RGBA8_UNORM, RGBA8_BPP),
            (MTL_FORMAT_BGRA8_UNORM, BGRA8_BPP),
            (MTL_FORMAT_RGBA8_UINT, RGBA8_BPP),
            (MTL_FORMAT_RGBA8_SINT, RGBA8_BPP),
            (MTL_FORMAT_RGBA16_UINT, RGBA16_BPP),
            (MTL_FORMAT_RGBA16_FLOAT, RGBA16F_BPP),
            (MTL_FORMAT_RGBA32_UINT, RGBA32_BPP),
            (MTL_FORMAT_RGBA32_FLOAT, RGBA32F_BPP),
        ];
        for (fmt, bpp) in expected {
            assert!(
                storage_selector(*fmt).is_some(),
                "format {fmt:#x} lost its storage selector"
            );
            assert_eq!(
                bytes_per_pixel(*fmt),
                Some(*bpp),
                "storage format {fmt:#x} changed texel width"
            );
        }

        // And no arm was added to one table without the other. Exhaustive, so
        // the two lists cannot drift apart in either direction.
        for fmt in 0u16..=u16::MAX {
            if storage_selector(fmt).is_some() {
                assert!(
                    bytes_per_pixel(fmt).is_some(),
                    "storage-capable {fmt:#x} has no texel width"
                );
                assert!(
                    expected.iter().any(|(f, _)| *f == fmt),
                    "storage-capable {fmt:#x} is not pinned above"
                );
            }
        }
    }

    /// A [`TexelLayout`]'s width is the width of the guest format it stands for.
    ///
    /// The layout enum and the `u16` format table are two vocabularies for one
    /// fact, and every sampled rail crosses between them: the format decides
    /// what the guest wrote, the layout decides the stride the host reads it
    /// at. Nothing compared them, and a disagreement is not a decode error —
    /// it is rows read at the wrong stride, which produces a sheared image
    /// rather than a refusal.
    ///
    /// The `match` is exhaustive on purpose. A new layout cannot be added
    /// without naming the guest format it represents, which is the question a
    /// new variant most needs to answer.
    #[test]
    fn a_texel_layout_is_as_wide_as_the_guest_format_it_stands_for() {
        for &layout in TexelLayout::ALL {
            let mtl = match layout {
                TexelLayout::Rgba8 => MTL_FORMAT_RGBA8_UNORM,
                TexelLayout::Bgra8 => MTL_FORMAT_BGRA8_UNORM,
                TexelLayout::R8 => MTL_FORMAT_R8_UNORM,
                TexelLayout::Rg8 => MTL_FORMAT_RG8_UNORM,
                TexelLayout::R16Float => MTL_FORMAT_R16_FLOAT,
                TexelLayout::R32Float => MTL_FORMAT_R32_FLOAT,
                TexelLayout::R16Unorm => MTL_FORMAT_R16_UNORM,
                TexelLayout::Rg16Unorm => MTL_FORMAT_RG16_UNORM,
                TexelLayout::Rg16Uint => MTL_FORMAT_RG16_UINT,
                TexelLayout::Rgba32Float => MTL_FORMAT_RGBA32_FLOAT,
                TexelLayout::Rgba16Float => MTL_FORMAT_RGBA16_FLOAT,
                TexelLayout::Rg16Float => MTL_FORMAT_RG16_FLOAT,
                TexelLayout::Rgba16Unorm => MTL_FORMAT_RGBA16_UNORM,
                TexelLayout::Rgb10a2Unorm => MTL_FORMAT_RGB10A2_UNORM,
                TexelLayout::Bgr10a2Unorm => MTL_FORMAT_BGR10A2_UNORM,
                TexelLayout::Rg11b10Float => MTL_FORMAT_RG11B10_FLOAT,
                TexelLayout::Bc1Rgba => MTL_FORMAT_BC1_RGBA,
                TexelLayout::Bc2Rgba => MTL_FORMAT_BC2_RGBA,
                TexelLayout::Bc3Rgba => MTL_FORMAT_BC3_RGBA,
                TexelLayout::Bc4RUnorm => MTL_FORMAT_BC4_R_UNORM,
                TexelLayout::Bc4RSnorm => MTL_FORMAT_BC4_R_SNORM,
                TexelLayout::Bc5RgUnorm => MTL_FORMAT_BC5_RG_UNORM,
                TexelLayout::Bc5RgSnorm => MTL_FORMAT_BC5_RG_SNORM,
                TexelLayout::Bc6hRgbFloat => MTL_FORMAT_BC6H_RGB_FLOAT,
                TexelLayout::Bc6hRgbUfloat => MTL_FORMAT_BC6H_RGB_UFLOAT,
                TexelLayout::Bc7Rgba => MTL_FORMAT_BC7_RGBA_UNORM,
            };
            // Compared as whole **block geometries** rather than as texel
            // widths. That subsumes the reading this used to take — an
            // uncompressed block is 1x1 and its `bytes` *is* `bytes_per_pixel`
            // — and it is the only form that can say anything about a
            // compressed layout, whose `bytes_per_pixel` is deliberately
            // `None`. A layout and its guest format disagreeing on the grid is
            // rows read at the wrong stride, which shears an image rather than
            // refusing it.
            assert_eq!(
                Some(layout.block()),
                block_geometry(mtl),
                "{layout:?} and its guest format {mtl:#x} disagree on the storage grid"
            );
            assert_eq!(
                layout.is_block_compressed(),
                is_block_compressed(mtl),
                "{layout:?} and its guest format {mtl:#x} disagree on being compressed"
            );
            // [`TexelLayout::has_cpu_loader_arm`] is a claim about
            // [`texel_to_rgba8`], so it is checked against `texel_to_rgba8`.
            // It used to be checked against `sampled_class(mtl).is_some()`,
            // which is a different table and already disagreed: `RG16Float` has
            // a conversion arm and no sampled class. Asking the function the
            // doc names removes the proxy rather than widening the other table
            // to match it.
            //
            // Both directions matter. A layout answering `false` here is one
            // that must *not* be handed to a performance floor that would send
            // it to the CPU, because there is nothing there to serve it — an
            // absence a later "just add the missing arm" edit removes without
            // noticing what it was for.
            // Widest texel any format in the table occupies, so one buffer
            // satisfies `texel_to_rgba8`'s length check for every layout.
            let src = [0u8; RGBA32_BPP as usize];
            assert_eq!(
                texel_to_rgba8(mtl, &src).is_some(),
                layout.has_cpu_loader_arm(),
                "{layout:?} ({mtl:#x}) disagrees with texel_to_rgba8 about having an arm"
            );
        }
    }

    /// Composing two plans is the same as applying them one after the other.
    ///
    /// Exhaustive over **every** pair of plans — all six selectors in all four
    /// slots, both sides, 6^4 x 6^4 = 1 679 616 pairs — checked against
    /// [`apply_swizzle_rgba8`] on an input whose four channels are distinct, so
    /// any two channels being confused is visible. This is the derivation
    /// rather than a restatement: [`SwizzlePlan::after`] is only correct if it
    /// agrees with actually applying the two, and that is what is asserted.
    ///
    /// A hardware component mapping takes one plan, so a bind that needs both a
    /// format remap and a view swizzle has to fold them. Getting the fold
    /// backwards produces a plausible image with two channels exchanged, which
    /// is exactly the class no screenshot catches.
    #[test]
    fn composing_two_swizzles_is_applying_them_in_order() {
        const SELECTORS: [SwizzleSource; 6] = [
            SwizzleSource::Zero,
            SwizzleSource::One,
            SwizzleSource::R,
            SwizzleSource::G,
            SwizzleSource::B,
            SwizzleSource::A,
        ];
        // Distinct, and none equal to what `Zero`/`One` produce, so a slot
        // taking a constant cannot accidentally match a channel.
        let input = [0x11, 0x22, 0x33, 0x44];
        let plans = || {
            SELECTORS.iter().flat_map(move |&r| {
                SELECTORS.iter().flat_map(move |&g| {
                    SELECTORS.iter().flat_map(move |&b| {
                        SELECTORS.iter().map(move |&a| SwizzlePlan {
                            source: [r, g, b, a],
                        })
                    })
                })
            })
        };
        for inner in plans() {
            let once = apply_swizzle_rgba8(&inner, input);
            for outer in plans() {
                assert_eq!(
                    apply_swizzle_rgba8(&outer.after(&inner), input),
                    apply_swizzle_rgba8(&outer, once),
                    "outer {:?} after inner {:?}",
                    outer.source,
                    inner.source
                );
            }
        }
    }

    /// Identity is the unit on both sides, and composition is not commutative.
    ///
    /// The unit law is what lets a caller compose unconditionally instead of
    /// branching on "did this format need a remap" — a branch that is the
    /// obvious place to forget one of the two cases. The non-commutativity is
    /// pinned because the argument order is otherwise easy to swap and the
    /// result still type-checks.
    #[test]
    fn identity_composes_away_and_order_matters() {
        let identity = swizzle_identity();
        let alpha_in_red = SwizzlePlan {
            source: [
                SwizzleSource::Zero,
                SwizzleSource::Zero,
                SwizzleSource::Zero,
                SwizzleSource::R,
            ],
        };
        assert_eq!(alpha_in_red.after(&identity), alpha_in_red);
        assert_eq!(identity.after(&alpha_in_red), alpha_in_red);

        // `A8Unorm` sampled through a view asking for alpha in every channel.
        // The byte is in the host format's red, so the answer names red four
        // times — and the reverse order does not, which is the point.
        let alpha_everywhere = SwizzlePlan {
            source: [
                SwizzleSource::A,
                SwizzleSource::A,
                SwizzleSource::A,
                SwizzleSource::A,
            ],
        };
        assert_eq!(
            alpha_everywhere.after(&alpha_in_red).source,
            [
                SwizzleSource::R,
                SwizzleSource::R,
                SwizzleSource::R,
                SwizzleSource::R
            ]
        );
        assert_ne!(
            alpha_everywhere.after(&alpha_in_red),
            alpha_in_red.after(&alpha_everywhere)
        );
    }

    /// [`TexelLayout::ALL`] really does hold every variant, exactly once, and
    /// [`TexelLayout::index`] agrees with the position it holds.
    ///
    /// Both halves are load-bearing. The host's linear-filter capability table
    /// is `[bool; ALL.len()]` indexed by `index()`, so a variant missing from
    /// `ALL` makes that array one short and a variant whose `index()` does not
    /// match its slot reads another layout's answer — neither of which is a
    /// compile error, and both of which decide whether a guest texture samples
    /// or is declined.
    #[test]
    fn every_texel_layout_is_in_the_all_list_exactly_once() {
        for (slot, &layout) in TexelLayout::ALL.iter().enumerate() {
            assert_eq!(
                layout.index(),
                slot,
                "{layout:?} sits at ALL[{slot}] but indexes {}",
                layout.index()
            );
        }
        let mut seen: Vec<usize> = TexelLayout::ALL.iter().map(|l| l.index()).collect();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(
            seen.len(),
            TexelLayout::ALL.len(),
            "two layouts share an index, or ALL holds one twice"
        );
    }

    /// **A cost floor may never be the reason a texture loses precision.**
    ///
    /// [`TexelLayout::a_cost_floor_may_decline`] is the zero-copy floors' whole
    /// admission rule, and it must stay the conjunction it is derived from: a
    /// layout may be turned away onto the CPU byte loader only where that
    /// loader has an arm *and* the arm is exact. The half-float colour pair is
    /// the case that made this necessary — both answer
    /// `has_cpu_loader_arm() == true` truthfully, so a floor asking only that
    /// question turned a 32 KiB `RGBA16Float` away from the exact rail and onto
    /// one that clamps to `[0, 1]` and quantizes to 256 levels.
    #[test]
    fn a_cost_floor_may_only_decline_a_layout_whose_cpu_arm_is_exact() {
        for &layout in TexelLayout::ALL {
            assert_eq!(
                layout.a_cost_floor_may_decline(),
                layout.has_cpu_loader_arm() && !layout.cpu_loader_arm_is_lossy(),
                "{layout:?} — the floor rule must stay derived, not re-listed"
            );
            if layout.cpu_loader_arm_is_lossy() {
                assert!(
                    layout.has_cpu_loader_arm(),
                    "{layout:?} — an arm that does not exist cannot be lossy"
                );
                assert!(
                    !layout.a_cost_floor_may_decline(),
                    "{layout:?} has only a lossy CPU arm, so a byte threshold \
                     declining it is data loss and not a cost decision"
                );
            }
        }
        // Named rather than derived, because the point is that these two are
        // *not* in the set a floor may turn away, and a test that only checked
        // the identity above would pass with the set empty.
        for layout in [TexelLayout::Rgba16Float, TexelLayout::Rg16Float] {
            assert!(layout.cpu_loader_arm_is_lossy(), "{layout:?}");
            assert!(!layout.a_cost_floor_may_decline(), "{layout:?}");
        }
        for layout in [TexelLayout::Rgba8, TexelLayout::Bgra8, TexelLayout::R8] {
            assert!(!layout.cpu_loader_arm_is_lossy(), "{layout:?}");
            assert!(layout.a_cost_floor_may_decline(), "{layout:?}");
        }
    }

    /// Four bytes wide is not the same as a four-byte colour order.
    ///
    /// [`TexelLayout::is_four_byte_color`] gates the RGBA8-shaped loaders, the
    /// tight-row gathers and the zero-copy rails, all of which reinterpret the
    /// four bytes as R,G,B,A. `R32Float` is exactly as wide and is a single
    /// channel, so admitting it on width would hand a float LUT to a loader
    /// that swizzles it as colour. The enum's own doc states this; nothing
    /// checked it.
    #[test]
    fn four_bytes_wide_does_not_admit_a_layout_to_the_colour_rails() {
        for layout in [TexelLayout::Rgba8, TexelLayout::Bgra8] {
            assert!(layout.is_four_byte_color(), "{layout:?} is a colour order");
            assert_eq!(layout.bytes_per_texel(), RGBA8_BPP);
        }
        // Three layouts are four bytes wide and are not colour orders: a
        // single-channel float LUT, the two-channel sixteen-bit chroma plane of
        // a ten-bit video, and the two-channel half-float. All would be
        // swizzled as R,G,B,A by a rail that admitted them on width.
        for layout in [
            TexelLayout::R32Float,
            TexelLayout::Rg16Unorm,
            TexelLayout::Rg16Float,
        ] {
            assert_eq!(layout.bytes_per_texel(), RGBA8_BPP);
            assert!(
                !layout.is_four_byte_color(),
                "{layout:?} is four bytes wide and is not a colour order"
            );
        }
        // `Rgba16Float` is the one layout *wider* than four bytes, and it is a
        // colour order — which is exactly why it must not answer
        // `is_four_byte_color`. The rails that ask reinterpret four bytes as
        // R,G,B,A; handing them an eight-byte texel would read every second
        // pixel and shear the image.
        for layout in [TexelLayout::Rgba16Float, TexelLayout::Rgba16Unorm] {
            assert_eq!(layout.bytes_per_texel(), RGBA16_BPP);
            assert!(!layout.is_four_byte_color(), "{layout:?}");
        }
        for layout in [
            TexelLayout::R8,
            TexelLayout::Rg8,
            TexelLayout::R16Float,
            TexelLayout::R16Unorm,
        ] {
            assert!(!layout.is_four_byte_color());
            assert_ne!(layout.bytes_per_texel(), RGBA8_BPP);
        }
    }

    /// A colour attachment's seed is written as that attachment's texel, and
    /// the wide arm agrees with the row converter that already knew how.
    ///
    /// Derived rather than restated: [`convert_rgba8_to_row`] has expanded
    /// RGBA8 into `RGBA16_FLOAT` since the sampled half-float work, so this
    /// asserts the two produce the same bytes instead of hand-writing an f16
    /// encoding a second time. A divergence between them is the bug — they are
    /// the same conversion reached from the guest's format and from the
    /// attachment's layout.
    #[test]
    fn a_wide_seed_expands_to_the_same_bytes_the_row_converter_writes() {
        const PIXELS: u32 = 4;
        let src: Vec<u8> = (0..PIXELS * RGBA8_BPP)
            .map(|i| (i * 7 % 256) as u8)
            .collect();

        let mut viaraw = vec![0u8; (PIXELS * RGBA16F_BPP) as usize];
        assert!(convert_rgba8_to_row(
            MTL_FORMAT_RGBA16_FLOAT,
            &src,
            PIXELS,
            &mut viaraw
        ));
        let mut via_layout = vec![0u8; (PIXELS * RGBA16F_BPP) as usize];
        assert!(expand_rgba8_to_texel(
            TexelLayout::Rgba16Float,
            &src,
            PIXELS,
            &mut via_layout
        ));
        assert_eq!(via_layout, viaraw);

        // The whole point of the arm: eight bytes a texel, not four. A seed
        // staged at the RGBA8 length under this attachment reads off the end of
        // its slot.
        assert_eq!(via_layout.len(), (PIXELS * RGBA8_BPP * 2) as usize);
    }

    /// Widening a seed and narrowing a readback are inverses, for every layout
    /// a colour attachment is created at.
    ///
    /// The two run on opposite ends of one target's life — the seed goes in as
    /// the attachment's texel, the fallback readback comes back out as RGBA8 —
    /// so a disagreement between them shows up as a frame that drifts every
    /// time it round-trips, which is far harder to read than a wrong colour
    /// once. Exact and not approximate: `unorm8 -> f16` lands on values the
    /// reverse LUT maps back to the same byte, so a Store that falls to the CPU
    /// rail loses the *range* the guest asked for and nothing else.
    /// A layout that carries fewer than four channels is included by giving it a
    /// seed whose missing channels already hold what the narrowing puts back —
    /// zero, and opaque alpha, which is what a shader sampling an `R16Float`
    /// attachment reads out of the channels it does not have. Stating it that
    /// way keeps one assertion (`back == src`) for every layout, rather than an
    /// exact case and an approximate one that could both pass while disagreeing
    /// about which channels are real.
    #[test]
    fn a_seed_widened_and_read_back_is_the_seed_it_started_as() {
        const PIXELS: u32 = 256;
        for (layout, channels) in [
            (TexelLayout::Rgba8, 4usize),
            (TexelLayout::Bgra8, 4),
            (TexelLayout::Rgba16Float, 4),
            (TexelLayout::Rg16Float, 2),
            (TexelLayout::R16Float, 1),
        ] {
            let src: Vec<u8> = (0..PIXELS * RGBA8_BPP)
                .map(|i| {
                    let c = (i % u32::from(RGBA8_BPP as u16)) as usize;
                    match () {
                        _ if c < channels => (i % 256) as u8,
                        _ if c == COMPONENT_A && channels < 4 => UNORM8_MAX,
                        _ => 0,
                    }
                })
                .collect();
            let mut wide = vec![0u8; (PIXELS * layout.bytes_per_texel()) as usize];
            assert!(
                expand_rgba8_to_texel(layout, &src, PIXELS, &mut wide),
                "{layout:?} must be writable as a seed"
            );
            let mut back = vec![0u8; (PIXELS * RGBA8_BPP) as usize];
            assert!(
                narrow_texel_to_rgba8(layout, &wide, PIXELS, &mut back),
                "{layout:?} must be readable back"
            );
            assert_eq!(back, src, "{layout:?} did not round-trip");
        }
    }

    /// The eight-bit arms stay a copy and an exchange, and a layout no colour
    /// attachment is created at is refused rather than written short.
    #[test]
    fn a_seed_is_refused_for_a_layout_no_colour_attachment_takes() {
        const PIXELS: u32 = 2;
        let src = [1u8, 2, 3, 4, 5, 6, 7, 8];

        let mut rgba = [0u8; 8];
        assert!(expand_rgba8_to_texel(
            TexelLayout::Rgba8,
            &src,
            PIXELS,
            &mut rgba
        ));
        assert_eq!(rgba, src);

        let mut bgra = [0u8; 8];
        assert!(expand_rgba8_to_texel(
            TexelLayout::Bgra8,
            &src,
            PIXELS,
            &mut bgra
        ));
        assert_eq!(bgra, [3, 2, 1, 4, 7, 6, 5, 8]);

        // What this list may hold is not this test's to decide: a layout is
        // seedable exactly when `render_target_bpp` admits the guest format it
        // stands for, and
        // `translate::pixel::…::the_renderable_set_is_one_answer_and_every_member_survives_both_rails`
        // is what holds the two together. Removing a layout from here without
        // that test agreeing means the two have drifted again.
        for layout in [
            TexelLayout::Rg8,
            TexelLayout::R32Float,
            TexelLayout::R16Unorm,
            TexelLayout::Rg16Unorm,
            TexelLayout::Rgba16Unorm,
        ] {
            let mut dst = [0u8; 64];
            assert!(
                !expand_rgba8_to_texel(layout, &src, PIXELS, &mut dst),
                "{layout:?} is not a colour attachment this device seeds"
            );
        }

        // A destination too short for the texels asked for is refused, not
        // partially written.
        let mut short = [0u8; 4];
        assert!(!expand_rgba8_to_texel(
            TexelLayout::Rgba8,
            &src,
            PIXELS,
            &mut short
        ));
    }

    /// Every format [`store_texel_order`] admits must survive a raw byte copy.
    ///
    /// Exhaustive over `u16` rather than over a list, because the failure this
    /// guards is a format being *added* to the admitted set whose texel this
    /// crate's other tables describe differently. That would land a frame in
    /// guest memory under the wrong layout, and it is invisible at the copy —
    /// which converts nothing and cannot notice.
    ///
    /// The rule used to be "four bytes of the order it claims", and it is now
    /// the agreement itself. Four was never the contract: it was the width of
    /// the only residents a render target could have, and with that gone what
    /// has to hold is that the copy's stride, the render-target table and the
    /// sampled table all name **one** texel for a given format. A width
    /// assertion would only re-state whichever of them was consulted first.
    #[test]
    fn a_byte_copy_destination_is_the_texel_every_other_table_agrees_it_is() {
        for fmt in 0u16..=u16::MAX {
            let Some(order) = store_texel_order(fmt) else {
                continue;
            };
            assert_eq!(
                bytes_per_pixel(fmt),
                Some(order.bytes_per_texel()),
                "{fmt:#x} stores a texel the copy would mis-stride"
            );
            assert_eq!(
                render_target_bpp(fmt),
                Some(order.bytes_per_texel()),
                "{fmt:#x} is a Store destination this device will not render into"
            );
            // The sampled table is the independent statement of the same byte
            // layout, so a disagreement here is one of the two being wrong.
            assert_eq!(
                sampled_class(fmt),
                Some(match order {
                    TexelLayout::Rgba8 => SampledClass::Rgba8Unorm,
                    TexelLayout::Bgra8 => SampledClass::Bgra8Unorm,
                    TexelLayout::Rgba16Float => SampledClass::Rgba16Float,
                    TexelLayout::Bgr10a2Unorm => SampledClass::Bgr10a2Unorm,
                    TexelLayout::Rg16Uint => SampledClass::Rg16Uint,
                    // Named rather than defaulted. This arm used to be
                    // `_ => SampledClass::Bgra8Unorm`, which was true only while
                    // the admitted set was {Rgba8, Bgra8, Rgba16Float}: the next
                    // member widened into it would have been asserted against a
                    // class describing a different word, and the assertion would
                    // have *passed* if that class happened to be what
                    // `sampled_class` answered. A panic here is the honest
                    // failure — name the new layout's class above.
                    other => panic!(
                        "{fmt:#x} is admitted to the byte copy as {other:?}, which this \
                         cross-check has no sampled class for"
                    ),
                }),
                "{fmt:#x} is read as one layout by the sampler and copied as another"
            );
        }
        // A renderable format that is still not a byte-copy destination, so a
        // further widening of the set above has to change this line to pass.
        assert!(render_target_bpp(MTL_FORMAT_RG16_FLOAT).is_some());
        assert!(
            store_texel_order(MTL_FORMAT_RG16_FLOAT).is_none(),
            "RG16_FLOAT renders but is not admitted to a copy"
        );
        // The widened one, named so that removing it from the rule is a test
        // failure rather than a silent narrowing back to eight bits.
        assert_eq!(
            store_texel_order(MTL_FORMAT_RGBA16_FLOAT),
            Some(TexelLayout::Rgba16Float),
            "a half-float render target must reach the byte-copy rail, or its \
             frame is quantized to eight bits on the way to the guest"
        );
        assert_eq!(
            store_texel_order(MTL_FORMAT_BGR10A2_UNORM),
            Some(TexelLayout::Bgr10a2Unorm),
            "a packed ten-bit render target must reach the byte-copy rail: the CPU \
             converter requantizes its channels through eight bits, which is the \
             resolution the guest picked the format for"
        );
    }

    /// A packed ten-bit texel survives the seed and readback pair unchanged.
    ///
    /// The pair is the *fallback* rail — a host with no guest-RAM import seeds a
    /// `Load` from semantic RGBA8 and narrows the readback back to it — so the
    /// property that matters is not that ten bits survive (they cannot; a seed
    /// carries eight) but that the widening and the narrowing invert each other
    /// exactly. If they did not, a frame that took the CPU rail twice would
    /// drift, and a drift of one level per pass is invisible until it is not.
    ///
    /// Both endpoints are checked explicitly because the bit-replication
    /// widening exists for them: a truncating `v << 2` would map 255 to 1020 and
    /// read back as 254, so full white would darken on every round trip.
    #[test]
    fn a_packed_ten_bit_texel_survives_the_seed_and_readback_round_trip() {
        for r in 0u16..=255 {
            let rgba = [r as u8, (255 - r) as u8, r.wrapping_mul(7) as u8, 0];
            for a in [0u8, 0x55, 0xaa, 0xff] {
                let mut src = rgba;
                src[COMPONENT_A] = a;
                let word = rgba8_to_bgr10a2_word(src);
                assert_eq!(
                    bgr10a2_word_to_rgba8(word),
                    src,
                    "{src:?} did not survive the pair (word {word:#010x})"
                );
            }
        }
        // The endpoints of the widening, stated as words so a channel landing in
        // the wrong bits fails here rather than in a frame.
        assert_eq!(rgba8_to_bgr10a2_word([0, 0, 0, 0xff]), 0xc000_0000);
        assert_eq!(rgba8_to_bgr10a2_word([0xff, 0, 0, 0]), 0x3ff << 20);
        assert_eq!(rgba8_to_bgr10a2_word([0, 0xff, 0, 0]), 0x3ff << 10);
        assert_eq!(rgba8_to_bgr10a2_word([0, 0, 0xff, 0]), 0x3ff);
        // And the whole-frame wrappers agree with the per-texel pair, so a
        // caller cannot be served a different conversion by going through the
        // row functions the rails actually call.
        let rgba: Vec<u8> = (0u8..=63)
            .flat_map(|v| [v, 255 - v, v << 2, 0xff])
            .collect();
        let pixels = (rgba.len() / 4) as u32;
        let mut packed = vec![0u8; rgba.len()];
        assert!(expand_rgba8_to_texel(
            TexelLayout::Bgr10a2Unorm,
            &rgba,
            pixels,
            &mut packed
        ));
        let mut back = vec![0u8; rgba.len()];
        assert!(narrow_texel_to_rgba8(
            TexelLayout::Bgr10a2Unorm,
            &packed,
            pixels,
            &mut back
        ));
        assert_eq!(back, rgba, "the row rails and the texel pair disagree");
        // The CPU Store converter is the same widening, one texel at a time.
        let mut one = [0u8; 4];
        assert!(rgba8_to_texel(
            MTL_FORMAT_BGR10A2_UNORM,
            [12, 34, 56, 0xff],
            &mut one
        ));
        assert_eq!(
            u32::from_le_bytes(one),
            rgba8_to_bgr10a2_word([12, 34, 56, 0xff])
        );
    }

    /// Every BC format the contract names, so a sweep is derived rather than
    /// hand-listed the way `TexelLayout::ALL` is for the layouts.
    const BC_FORMATS: &[u16] = &[
        MTL_FORMAT_BC1_RGBA,
        MTL_FORMAT_BC1_RGBA_SRGB,
        MTL_FORMAT_BC2_RGBA,
        MTL_FORMAT_BC2_RGBA_SRGB,
        MTL_FORMAT_BC3_RGBA,
        MTL_FORMAT_BC3_RGBA_SRGB,
        MTL_FORMAT_BC4_R_UNORM,
        MTL_FORMAT_BC4_R_SNORM,
        MTL_FORMAT_BC5_RG_UNORM,
        MTL_FORMAT_BC5_RG_SNORM,
        MTL_FORMAT_BC6H_RGB_FLOAT,
        MTL_FORMAT_BC6H_RGB_UFLOAT,
        MTL_FORMAT_BC7_RGBA_UNORM,
        MTL_FORMAT_BC7_RGBA_UNORM_SRGB,
    ];

    /// The list above is exactly the formats `is_block_compressed` claims.
    ///
    /// Swept over the whole `u16` space, so a family added to `bc_block_bytes`
    /// and not to the list — or the reverse — fails here instead of being missed
    /// by every test that iterates the list.
    #[test]
    fn the_bc_sweep_list_is_every_block_compressed_format() {
        let claimed: Vec<u16> = (0..=u16::MAX).filter(|&f| is_block_compressed(f)).collect();
        assert_eq!(claimed, BC_FORMATS.to_vec());
    }

    /// A block geometry is the texel table with its grid stated, for every
    /// uncompressed format — not a second opinion about the same number.
    ///
    /// This is the invariant that lets [`tight_row_bytes`] be one expression
    /// over both families. Swept over the whole space rather than a sample: the
    /// two functions would agree on any list chosen after the fact.
    #[test]
    fn a_block_geometry_agrees_with_the_texel_table() {
        for format in 0..=u16::MAX {
            match (bytes_per_pixel(format), block_geometry(format)) {
                (Some(bpp), Some(block)) => {
                    assert!(
                        !is_block_compressed(format),
                        "{format:#x} has a bytes-per-texel and claims to be compressed"
                    );
                    assert_eq!(
                        (block.width, block.height, block.bytes),
                        (1, 1, bpp),
                        "{format:#x}: an uncompressed block must be 1x1 of its own texel"
                    );
                }
                (None, Some(block)) => {
                    assert!(
                        is_block_compressed(format),
                        "{format:#x} has a block but no texel width and is not compressed"
                    );
                    assert_eq!((block.width, block.height), (BC_BLOCK_SIDE, BC_BLOCK_SIDE));
                    assert!(block.bytes == BC_BLOCK_BYTES_8 || block.bytes == BC_BLOCK_BYTES_16);
                }
                (None, None) => {}
                (Some(_), None) => panic!("{format:#x} has a texel width and no block"),
            }
        }
    }

    /// A BC format is refused by every rail except the sampled bind.
    ///
    /// **This is the whole safety argument for `TexelLayout::bytes_per_texel`
    /// answering about a block**, and it is a test rather than a paragraph
    /// because ninety-odd call sites call that method and none of them was
    /// audited by hand. Each rail below is a total gate: a format the gate says
    /// `None` for cannot reach the sizing code behind it, so the only rail a BC
    /// layout travels is the sampled bind — where the staging buffer is sized
    /// from the loader's own byte count and `VkBufferImageCopy` does the block
    /// arithmetic itself.
    ///
    /// If a rail is ever widened to admit one, this test is what says the
    /// argument has to be re-made.
    #[test]
    fn a_bc_format_is_refused_by_every_rail_but_the_sampled_bind() {
        for &format in BC_FORMATS {
            assert!(
                bytes_per_pixel(format).is_none(),
                "{format:#x}: a BC1 texel is half a byte, so there is no honest answer here"
            );
            assert!(
                render_target_bpp(format).is_none(),
                "{format:#x} must not be a colour attachment"
            );
            assert!(
                storage_selector(format).is_none(),
                "{format:#x} must not be a storage image — a shader cannot write a block"
            );
            assert!(
                sampled_class(format).is_none(),
                "{format:#x} must not claim a CPU-upload fast path"
            );
            assert!(
                store_texel_order(format).is_none(),
                "{format:#x} must not be a byte-copy Store destination"
            );
            assert!(
                texel_to_rgba8(format, &[0u8; 16]).is_none(),
                "{format:#x} must have no CPU loader arm"
            );
            assert!(
                !rgba8_to_texel(format, [1, 2, 3, 4], &mut [0u8; 16]),
                "{format:#x} must have no CPU Store converter"
            );
            // And the one rail it does travel names it.
            let layout = block_compressed_layout(format)
                .unwrap_or_else(|| panic!("{format:#x} must name a sampled layout"));
            assert!(layout.is_block_compressed());
            assert!(!layout.has_cpu_loader_arm());
            assert_eq!(Some(layout.block()), block_geometry(format));
        }
    }

    /// A BC3 level is sized and strided in blocks, on the geometry a guest was
    /// measured sending.
    ///
    /// The bug class: every one of these numbers was computed per **texel**
    /// before, so a 64x64 BC3 level read a 64-byte row instead of a 256-byte one
    /// and claimed sixty-four rows instead of sixteen. Nothing about that is a
    /// refusal — it is a texture bound from the wrong bytes — which is why the
    /// figures here are the descriptors' own rather than round numbers.
    ///
    /// The mip tail is the half nobody writes down: a 2x2 and even a 1x1 level
    /// still occupy one whole block, so the rounding is the contract and a
    /// division would read four bytes of a sixteen-byte level.
    #[test]
    fn a_bc3_level_is_sized_and_strided_in_blocks() {
        const BC3: u16 = MTL_FORMAT_BC3_RGBA;
        // Measured on the boot that found the family: `L0=1024x1024 bpr=4096`
        // and `L0=64x64 bpr=256`, both from the guest's own descriptors.
        assert_eq!(tight_row_bytes(1024, BC3), Some(4096));
        assert_eq!(tight_row_bytes(64, BC3), Some(256));
        assert_eq!(tight_row_count(1024, BC3), Some(256));
        assert_eq!(tight_row_count(64, BC3), Some(16));
        // A 1 MiB base, which is what `alloc=1400832` is the eleven-level
        // pyramid of.
        assert_eq!(
            TexelLayout::Bc3Rgba.image_bytes(1024, 1024),
            Some(1024 * 1024)
        );
        // The tail. Every one of these is one block.
        for side in [1u32, 2, 3, 4] {
            assert_eq!(
                tight_row_bytes(side, BC3),
                Some(BC_BLOCK_BYTES_16),
                "a {side}-wide BC3 row is one block"
            );
            assert_eq!(tight_row_count(side, BC3), Some(1));
            assert_eq!(
                TexelLayout::Bc3Rgba.image_bytes(side, side),
                Some(u64::from(BC_BLOCK_BYTES_16))
            );
        }
        // Five texels need two blocks, which is the rounding stated as a case
        // rather than as a formula.
        assert_eq!(tight_row_bytes(5, BC3), Some(2 * BC_BLOCK_BYTES_16));
        assert_eq!(tight_row_count(5, BC3), Some(2));
        // BC1 is the same grid at half the weight, so a wrong block-bytes
        // constant cannot hide behind BC3's.
        assert_eq!(tight_row_bytes(64, MTL_FORMAT_BC1_RGBA), Some(128));
        assert_eq!(
            TexelLayout::Bc1Rgba.image_bytes(1024, 1024),
            Some(512 * 1024)
        );
        // And an uncompressed format is unchanged by all of it.
        assert_eq!(tight_row_bytes(64, MTL_FORMAT_BGRA8_UNORM), Some(256));
        assert_eq!(tight_row_count(64, MTL_FORMAT_BGRA8_UNORM), Some(64));
    }

    #[test]
    fn unsupported_fail_closed() {
        // Unknown formats fail closed. Depth/stencil families have bpp for blit
        // packing but remain unsampled / non-storage / non-RT.
        for fmt in [0xffffu16, 130, 204] {
            assert!(bytes_per_pixel(fmt).is_none());
            assert!(sampled_class(fmt).is_none());
            assert!(storage_selector(fmt).is_none());
            assert!(render_target_bpp(fmt).is_none());
            assert!(texel_to_rgba8(fmt, &[0; 16]).is_none());
        }
        for fmt in [
            MTL_FORMAT_DEPTH32_FLOAT,
            MTL_FORMAT_STENCIL8,
            MTL_FORMAT_DEPTH32_FLOAT_STENCIL8,
            MTL_FORMAT_DEPTH24_UNORM_STENCIL8,
        ] {
            assert!(bytes_per_pixel(fmt).is_some());
            assert!(sampled_class(fmt).is_none());
            assert!(storage_selector(fmt).is_none());
            assert!(render_target_bpp(fmt).is_none());
            assert!(texel_to_rgba8(fmt, &[0; 16]).is_none());
        }
    }

    #[test]
    fn depth32_stencil8_plane_roundtrip() {
        let fmt = MTL_FORMAT_DEPTH32_FLOAT_STENCIL8;
        let p = depth_stencil_packing(fmt).unwrap();
        assert_eq!(p.full_bpp, 8);
        // Depth = 1.0f32, stencil = 0xAB
        let mut texel = [0u8; 8];
        texel[0..4].copy_from_slice(&1.0f32.to_bits().to_le_bytes());
        texel[4] = 0xab;
        let mut depth = [0u8; 4];
        assert!(extract_depth_stencil_plane(
            fmt,
            BlitAspect::Depth,
            &texel,
            &mut depth
        ));
        assert_eq!(depth, 1.0f32.to_bits().to_le_bytes());
        let mut st = [0u8; 1];
        assert!(extract_depth_stencil_plane(
            fmt,
            BlitAspect::Stencil,
            &texel,
            &mut st
        ));
        assert_eq!(st[0], 0xab);
        // Insert new depth, keep stencil.
        let mut t2 = texel;
        let new_d = 0.5f32.to_bits().to_le_bytes();
        assert!(insert_depth_stencil_plane(
            fmt,
            BlitAspect::Depth,
            &new_d,
            &mut t2
        ));
        assert_eq!(t2[4], 0xab);
        let mut d2 = [0u8; 4];
        assert!(extract_depth_stencil_plane(
            fmt,
            BlitAspect::Depth,
            &t2,
            &mut d2
        ));
        assert_eq!(d2, new_d);
        // Row extract 2 pixels.
        let mut row = [0u8; 16];
        row[..8].copy_from_slice(&texel);
        row[8..16].copy_from_slice(&t2);
        let mut planes = [0u8; 8];
        assert!(extract_plane_row(
            fmt,
            BlitAspect::Depth,
            &row,
            2,
            &mut planes
        ));
        assert_eq!(&planes[0..4], &1.0f32.to_bits().to_le_bytes());
        assert_eq!(&planes[4..8], &new_d);
    }

    /// `Full` is not a plane, and every plane entry point refuses it.
    ///
    /// The aspect used to travel as `(depth: bool, stencil: bool)`, where
    /// "neither" and "both" were two distinct values that had to be refused
    /// separately — and the five functions did not agree on how. Two rejected
    /// `(true, true)`, `blit_aspect_needs_repack` read it as a plane pass, and
    /// the two row helpers read it as depth. Collapsing to three states makes
    /// "both" unwritable; this pins the one refusal that is left.
    #[test]
    fn the_full_aspect_is_not_a_plane_at_any_entry_point() {
        let fmt = MTL_FORMAT_DEPTH32_FLOAT_STENCIL8;
        let texel = [0u8; 8];
        let mut out = [0u8; 8];
        assert!(!extract_depth_stencil_plane(
            fmt,
            BlitAspect::Full,
            &texel,
            &mut out
        ));
        let mut t = texel;
        assert!(!insert_depth_stencil_plane(
            fmt,
            BlitAspect::Full,
            &out,
            &mut t
        ));
        let row = [0u8; 16];
        let mut planes = [0u8; 8];
        assert!(!extract_plane_row(
            fmt,
            BlitAspect::Full,
            &row,
            2,
            &mut planes
        ));
        let mut dst_row = [0u8; 16];
        assert!(!insert_plane_row(
            fmt,
            BlitAspect::Full,
            &planes,
            2,
            &mut dst_row
        ));
        assert!(!blit_aspect_needs_repack(fmt, BlitAspect::Full));
        // And `Full` is still the aspect that asks for the whole packed texel.
        assert_eq!(blit_aspect_bytes_per_pixel(fmt, BlitAspect::Full), Some(8));
    }

    #[test]
    fn depth24_stencil8_plane_pack() {
        let fmt = MTL_FORMAT_DEPTH24_UNORM_STENCIL8;
        // stencil=0x11, depth24=0xAABBCC → packed LE
        let depth24 = 0x00aabbccu32;
        let packed = 0x11u32 | (depth24 << 8);
        let texel = packed.to_le_bytes();
        let mut depth = [0u8; 4];
        assert!(extract_depth_stencil_plane(
            fmt,
            BlitAspect::Depth,
            &texel,
            &mut depth
        ));
        assert_eq!(u32::from_le_bytes(depth), depth24);
        let mut st = [0u8; 1];
        assert!(extract_depth_stencil_plane(
            fmt,
            BlitAspect::Stencil,
            &texel,
            &mut st
        ));
        assert_eq!(st[0], 0x11);
        let mut t2 = [0u8; 4];
        assert!(insert_depth_stencil_plane(
            fmt,
            BlitAspect::Stencil,
            &[0x22],
            &mut t2
        ));
        assert!(insert_depth_stencil_plane(
            fmt,
            BlitAspect::Depth,
            &depth24.to_le_bytes(),
            &mut t2
        ));
        assert_eq!(u32::from_le_bytes(t2), 0x22 | (depth24 << 8));
    }

    #[test]
    fn property_fuzz_row_roundtrip_rgba8() {
        // Corpus-driven property: random-ish patterns through BGRA/RGBA convert.
        let patterns: &[&[u8]] = &[
            &[0, 0, 0, 0],
            &[255, 255, 255, 255],
            &[1, 2, 3, 4],
            &[10, 20, 30, 40, 50, 60, 70, 80],
            &[0xde, 0xad, 0xbe, 0xef, 0xca, 0xfe, 0xba, 0xbe],
        ];
        for p in patterns {
            let pixels = (p.len() / 4) as u32;
            if pixels == 0 {
                continue;
            }
            let mut rgba = vec![0u8; p.len()];
            assert!(convert_row_to_rgba8(
                MTL_FORMAT_RGBA8_UNORM,
                p,
                pixels,
                &mut rgba
            ));
            let mut back = vec![0u8; p.len()];
            assert!(convert_rgba8_to_row(
                MTL_FORMAT_RGBA8_UNORM,
                &rgba,
                pixels,
                &mut back
            ));
            assert_eq!(&back[..], *p);
        }
    }
}

#[cfg(test)]
mod solid_fill_tests {
    use super::solid_rgba8;

    /// The one-pass fill produces exactly what walking texels produced.
    ///
    /// The doubling fill writes a growing `memcpy` rather than a texel at a
    /// time, so the arithmetic it can get wrong is a length: a short image, a
    /// tail that is not a whole doubling, and the zero-texel case are the three
    /// places it could differ from the idiom it replaced. Every geometry below
    /// is checked against that idiom rather than against a hand-written
    /// expectation, so the test states equivalence and not a second spelling.
    #[test]
    fn the_doubling_fill_matches_a_texel_walk_at_every_geometry() {
        fn walk(w: u32, h: u32, px: [u8; 4]) -> Vec<u8> {
            let n = (w as usize) * (h as usize) * px.len();
            let mut img = vec![0u8; n];
            for texel in img.chunks_exact_mut(px.len()) {
                texel.copy_from_slice(&px);
            }
            img
        }
        let clear = [0.25_f64, 0.5, 0.75, 1.0];
        let px = [
            super::f64_to_unorm8(clear[0]),
            super::f64_to_unorm8(clear[1]),
            super::f64_to_unorm8(clear[2]),
            super::f64_to_unorm8(clear[3]),
        ];
        for (w, h) in [
            (0, 0),
            (0, 8),
            (8, 0),
            (1, 1),
            (1, 3),
            (3, 1),
            (5, 7),
            (17, 13),
            (64, 64),
            (129, 3),
        ] {
            assert_eq!(
                solid_rgba8(w, h, &clear),
                walk(w, h, px),
                "rgba {w}x{h} must match the texel walk"
            );
        }
    }
}
