//! The descriptors core keeps as objects (D5): the texture descriptor, the render pass
//! descriptor, and the pipeline descriptors Metal makes from serialized data. The runtime's
//! descriptors are values, so each object holds its value behind a lock, which core's setters
//! change in place.
//!
//! **A texture descriptor** takes each raw value as the runtime's type at its setter, so a value
//! the runtime cannot hold dies there, but where Metal's setter takes it and acts on it where a
//! texture is made (measured: `texture_setters_act_where_metal_does`). A texture type or a swizzle
//! channel the runtime has no value of is kept, and every creator aborts at it, as Metal's does;
//! protection options are kept and answered, and every creator makes the texture as without them,
//! a texture answering protection options 0, as Metal's buffer, heap and IOSurface creators do
//! (measured: `texture_setters_act_where_metal_does`), where Metal's `newTextureWithDescriptor:`
//! and `newSharedTextureWithDescriptor:` answer nil for options other than none, relayed
//! ([`TextureFields::made_unprotected`]); a resource index places the texture, forced or not.
//! `setPixelFormat:` takes what Metal's takes
//! ([`crate::gpu::format_info::setter_takes`]), and keeps raw and answers the formats Metal makes
//! no texture of ([`crate::gpu::no_texture`]), and any other value the runtime has no format of,
//! a texture of which dies where it is made, or is nil ([`lacked`]); any other value dies at the
//! setter, where Metal's asserts. A descriptor of a format Metal makes no texture of from a
//! descriptor alone (those, and every two-plane YCbCr format, which it holds as the runtime's,
//! [`crate::gpu::ycbcr::two_plane`]) is nil from every creator, or 0 bytes from the
//! heap's size query, where Metal's validation passes it, and dies where that aborts, as the
//! runtime answers it ([`TextureFields::descriptor`]). `framebufferOnly` and `isDrawable` are kept
//! as set, and sent to the runtime only in a descriptor of such a format, for its validation:
//! neither changes what a texture holds. `writeSwizzleEnabled` (SPI) is sent to the runtime, which
//! makes a texture of a swizzle other than the identity read alone, as Metal does, and refuses one
//! written or rendered, which Metal makes and writes through the swizzle: that dies where it is
//! made, and the shared creator's, which Metal's validation stops, too
//! (`write_swizzles_are_made_where_read_alone`). An invalid pixel format and an empty size die, as
//! Metal aborts for them, and so does `framebufferOnly` where a texture other than one over an
//! IOSurface or a shared one is made with it: a device's, a buffer's (linear or not) and a heap's
//! creators assert it is not set (measured on this host, without API validation:
//! "framebufferOnly == NO"), `newSharedTextureWithDescriptor:` drops it, and
//! `heapTextureSizeAndAlignWithDescriptor:` answers as without it ([`Maker`]).
//!
//! **A rotation** (`setRotation:`, SPI) is kept as set and answered, and the runtime's descriptor
//! of a texture over an IOSurface takes it, as does one of a format Metal makes no texture of from
//! a descriptor alone, for its validation. Measured on Metal on this host (M3 Max, without API
//! validation), with a 64 by 16 `BGRA8Unorm` render target:
//! - the descriptor keeps every value, `u64::MAX` included, and answers it;
//! - making a texture of a rotation other than 0 aborts unless the descriptor is
//!   `framebufferOnly` ("MTLTextureDescriptor has a non-zero rotation set but is not marked as
//!   framebufferOnly"), and one past 4 aborts whatever it is ("rotation is not a valid
//!   MTLTextureRotation"); `heapTextureSizeAndAlignWithDescriptor:` aborts alike, and answers
//!   rotations 1 to 4 as it answers 0;
//! - only a texture over an IOSurface takes one: a device's, a buffer's (linear or not) and a
//!   heap's creators abort at any rotation but 0, and `newSharedTextureWithDescriptor:` drops
//!   `framebufferOnly` first, so it aborts at any too ([`Maker`]);
//! - a texture over an IOSurface of rotations 1 to 4 is made at the descriptor's size, whatever
//!   the surface's, and so are its views, each keeping its rotation.
//!
//! The runtime makes a texture over an IOSurface of rotations 1 to 4 with the rotation and
//! `framebufferOnly`, and builds what Metal does with it: a pass stores its texels turned in the
//! surface's bytes (rotation 2 by 180 degrees, 1 and 4 a quarter turn, its store the texture's
//! height wide, 3 the other quarter turn) and loads them turned back, but a pass that draws loads
//! one of rotation 4 transposed, as Metal does, so it stores the texels it loaded mirrored left
//! to right; functions see the stored frame unturned, their writes dropped; CPU access and blits
//! address the plane as a texture of the texture's own extent at the plane's rows. A view of the
//! whole texture, unswizzled, in `BGRA8Unorm`, `BGRA8Unorm_sRGB`, `RGBA8Unorm` or
//! `RGBA8Unorm_sRGB`, the texture being of one of the four, is made as Metal makes it (its API
//! validation stopping it): of the texture's frame and rotation, its passes storing, loading and
//! resolving as the texture's would, in the view's format. What it does not build it refuses,
//! each refusal dying at its send: a view in another format, of another type or range, or
//! swizzled (which Metal makes), CPU access or a blit whose rows run past its buffer, and a
//! binding beyond its usage. One difference it announces, once per device
//! (`rotated-load-tiles`): a pass that loads a texture of rotation 4 and draws lands every 32 by
//! 32 tile of the frame mirrored, where Metal loads and stores only the tiles a draw's primitives
//! reach and keeps the others' texels in place; the two agree where the draws reach every tile.
//! Such a pass dies at `endEncoding`, naming the kind ([`crate::answer::DIES`]), whether or not
//! its draws reach every tile.
//!
//! **A usage** (`setUsage:`) is kept whole, as Metal's descriptor keeps every bit set: the bits the
//! runtime takes (`ShaderRead`, `ShaderWrite`, `RenderTarget`, `PixelFormatView`, `ShaderAtomic`,
//! Apple's private `NoConcurrentAccess`, `Video` and `BlockWritesOnly`, 0x4000, 0x8000 and
//! 0x10000, and 0x8, which Metal's texture drops) in its descriptor, and the rest raw
//! ([`TextureFields::private_usage`]: bits Metal does not define), sent nowhere. Measured on Metal
//! on this host
//! (M3 Max, macOS 26.7, without API validation and, for a few, with it, one case per process):
//! every bit from 0 to 63, alone and with `ShaderRead | RenderTarget`, of a 2D BGRA8Unorm texture;
//! 0x10000 of 2D textures of 15 formats (R8Unorm to RGBA32Float, the depth and stencil ones, BC1
//! and the XR formats) in Shared and Private storage, and of BGRA8Unorm in every type and storage
//! mode; 0x8, 0x4000 and 0x8000 of four formats and three types; and each of BGRA8Unorm at every
//! creator:
//! - the descriptor's `usage` answers every bit as set;
//! - every creator, and the heap's size query, takes four bits Metal's public headers name none
//!   of, 0x8, 0x4000, 0x8000 and 0x10000 (a macOS 26.6.2 guest's textures of usage 0x10005 have
//!   it): it makes the texture as without them, of the same size, alignment and compression
//!   (`isCompressed`), which clears, renders, reads, samples and is written alike (BGRA8Unorm,
//!   RGBA8Unorm, RGBA16Float and Depth32Float compared). The texture's `usage` answers 0x4000,
//!   0x8000 and 0x10000 among its bits and drops 0x8, and so do its views and a texture opened from
//!   its shared handle: the runtime's texture keeps the four, answering its usage without 0x8, as
//!   its views and shared opens do. A texture of 0x8 alone is no texture of `Unknown` usage: it
//!   answers a usage of 0 and the heap's size query a compressed one's size (16512 bytes for a
//!   64x64 BGRA8Unorm one in Private storage, where `Unknown` answers 16384), as a texture of
//!   `ShaderRead` alone does, which reads, samples and renders alike (measured on the M3 Max under
//!   macOS 27.0.1, Shared and Private storage: a kernel's read and sample, a pass's clear and
//!   draw); a format view of it answers 0, and a swizzled view of it Metal asserts at
//!   (`_usage != MTLTextureUsageUnknown`), as the runtime refuses it (`SwizzledViewOfNoUsage`). A
//!   view of a swizzle other than the default keeps
//!   `ShaderRead` and `PixelFormatView` of the public bits, and 0x4000 and 0x10000 but not 0x8000
//!   (a texture of 0x1c01d answers a view of 0x14011), as the runtime's view does;
//! - every creator, and the heap's size query, aborts at any other bit Metal does not define
//!   ("MTLTextureUsage has unknown bits 0x40."), but `newSharedTextureWithDescriptor:` of a format
//!   it makes no texture of, which answers nil first ([`crate::gpu::no_texture`]; 0x40, 0x20000
//!   and bit 63 measured there for each).
//!
//! **A render pass descriptor** is the runtime's value of Metal's defaults, which core's typed
//! sets change in their order; its attachments hold the runtime's textures they are given, as
//! Metal's attachments retain theirs, until the descriptor goes. The encoder's creation checks it
//! as Metal's does. Its rate map
//! is held by reference, as Metal holds it, and an encoder takes the runtime's map, which draws
//! the pass by the map as the GPU runs it, and decides what it decides by the map's class at the
//! encoder's creation ([`crate::gpu::rate_map`]).

use metal2vulkan::metal::PixelFormat;
use metal2vulkan::pipeline::{
    ComputePipelineDescriptor, RenderPipelineDescriptor, TileRenderPipelineDescriptor,
};
use metal2vulkan_runtime as rt;
use reims_vgpu_core::platform::Handle;
use reims_vgpu_core::platform::descriptor::{
    self, RenderPassDescriptor, TextureDescriptor, TileRenderPipelineDescriptor as TileTrait,
};
use reims_vgpu_core::platform::types::{ClearColor, MtlSamplePosition};
use reims_vgpu_core::platform::values::TextureDesc;
use reims_vgpu_core::platform::values::{
    AttachmentState, ColorAttachmentSet, DepthAttachmentSet, PassSet, StencilAttachmentSet,
};
use reims_vgpu_core::raise::{OnFault, RaiseResult};
use reims_vgpu_core::sync::Mutex;

use crate::answer::{Policy, metal_aborts, plain, send, unrepresentable};
use crate::gpu::rate_map::VulkanRateMap;
use crate::gpu::resource::{Under, VulkanBuffer, VulkanTexture};
use crate::gpu::serialized::{Decoded, Ignored, RawFormat};
use crate::gpu::{resource_options, texture_usage};
use crate::platform::{Obj, VulkanPlatform};

/// What a texture descriptor holds: the runtime's descriptor, and the private properties it has
/// no field for, as core set them.
#[derive(Debug, Clone, Copy, Default)]
pub struct TextureFields {
    /// The runtime's descriptor.
    pub desc: rt::TextureDescriptor,
    /// `framebufferOnly` (SPI), when set.
    pub framebuffer_only: Option<bool>,
    /// `isDrawable` (SPI), when set.
    pub is_drawable: Option<bool>,
    /// `writeSwizzleEnabled` (SPI), when set.
    pub write_swizzle_enabled: Option<bool>,
    /// `rotation` (SPI): Metal's `MTLTextureRotation`, 0 until set, kept as set.
    pub rotation: u64,
    /// The usage bits `setUsage:` took that the runtime does not take, raw: bits Metal does not
    /// define (measured: the module's docs).
    pub private_usage: u64,
    /// A format Metal makes no texture of, raw, as `setPixelFormat:` took it
    /// ([`crate::gpu::no_texture`]); `desc` then holds `MTLPixelFormatInvalid`.
    pub no_texture_format: Option<u64>,
    /// A value Metal's setter takes that the runtime has no format of, or that Metal describes no
    /// format for, raw, as `setPixelFormat:` took it ([`lacked`]); `desc` then holds
    /// `MTLPixelFormatInvalid`.
    pub lacked_format: Option<u64>,
    /// A texture type the runtime has no type of, raw, as `setTextureType:` took it, which every
    /// creator aborts at ([`TextureFields::descriptor`]); `desc` then holds the type before it.
    pub raw_texture_type: Option<u64>,
    /// A swizzle of a channel the runtime has no swizzle of, raw (red, green, blue, alpha), as
    /// `setSwizzle:` took it, which every creator asserts at ([`TextureFields::descriptor`]);
    /// `desc` then holds the swizzle before it.
    pub raw_swizzle: Option<[u8; 4]>,
    /// `protectionOptions` (SPI), as `setProtectionOptions:` took it: the runtime has no
    /// protected content, and every creator makes the texture as without them
    /// ([`TextureFields::made_unprotected`]).
    pub protection_options: u64,
}

/// `MTLTextureRotation0`, the rotation of a texture made without one.
pub(crate) const DEFAULT_ROTATION: u64 = 0;

/// What `setUsage:` of `raw` sets: the runtime's usage of the bits it takes, its flags and 0x8,
/// which Metal's descriptor takes and its texture drops (`TextureUsage::DROPPED`), and the other
/// bits raw ([`TextureFields::private_usage`]).
fn usage_set(raw: u64) -> (metal2vulkan::metal::TextureUsage, u64) {
    use metal2vulkan::metal::TextureUsage;
    let runtime = u64::from(TextureUsage::ALL_KNOWN.as_raw() | TextureUsage::DROPPED);
    (texture_usage("setUsage:", raw & runtime), raw & !runtime)
}

/// What makes a texture of a descriptor, which decides what Metal takes of its rotation
/// (measured: the module's docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Maker {
    /// `heapTextureSizeAndAlignWithDescriptor:`, which makes none: the descriptor's validation
    /// alone.
    SizeQuery,
    /// `newTextureWithDescriptor:iosurface:plane:`, which takes a rotation.
    Surface,
    /// `newSharedTextureWithDescriptor:`, which drops `framebufferOnly` before the validation.
    Shared,
    /// A device's, a buffer's and a heap's other creators, which take no rotation and no
    /// `framebufferOnly`.
    Other,
}

/// What `setPixelFormat:` (`sel`) of `raw` sets: the runtime's format, or `MTLPixelFormatInvalid`
/// and the raw value of a format Metal makes no texture of ([`crate::gpu::no_texture::taken`]),
/// as the second, or of any other value Metal's setter takes
/// ([`crate::gpu::format_info::setter_takes`]) that the runtime has no format of, or that Metal
/// describes no format for but `MTLPixelFormatInvalid` (`Depth24Unorm_Stencil8` and
/// `X24_Stencil8`, which the runtime has), as the third, a texture of which dies where it is
/// made, or is nil ([`lacked`]). Any other value dies here, where Metal's setter asserts.
pub(crate) fn format_set(sel: &'static str, raw: u64) -> (PixelFormat, Option<u64>, Option<u64>) {
    if crate::gpu::no_texture::taken(raw) {
        (PixelFormat::Invalid, Some(raw), None)
    } else if !crate::gpu::format_info::setter_takes(raw) {
        metal_aborts(
            sel,
            &format!("pixel format {raw}, not a valid MTLPixelFormat"),
        )
    } else {
        match crate::gpu::held_format(sel, raw) {
            Ok(pf) => (pf, None, None),
            Err(raw) => (PixelFormat::Invalid, None, Some(raw)),
        }
    }
}

/// Dies at `sel`, which makes a texture of `raw`, a value Metal's setter takes that this platform
/// keeps raw, which Metal describes no format for (the runtime has every format it describes,
/// [`lacked_as`]), where Metal asserts ([`crate::gpu::format_info::setter_takes`]; measured:
/// `formats_the_runtime_lacks_die_where_textures_are_made`).
pub(crate) fn lacked(sel: &'static str, raw: u64) -> ! {
    lacked_as(sel, "a texture of", raw, None)
}

/// Dies at `sel`, which makes `what` (`a texture of`, `a view of`, a state whose attachment is)
/// pixel format `raw`, which the runtime cannot make: named, with Metal's description's name for
/// it, where Metal describes the format, and as Metal's abort where it describes none. `refusal`
/// is the runtime's, where it refused the format ([`format_refused`]); without one, the value is
/// one this platform keeps raw, which Metal describes no format for, as the runtime has every
/// format Metal describes ([`crate::gpu::held_format`]; tested:
/// `the_runtime_has_every_format_metal_describes`).
pub(crate) fn lacked_as(sel: &'static str, what: &str, raw: u64, refusal: Option<&rt::Error>) -> ! {
    if !crate::gpu::format_info::described(sel, raw) {
        metal_aborts(
            sel,
            &format!("{what} pixel format {raw}, which Metal describes no format for"),
        );
    }
    let name = crate::gpu::format_info::name(sel, raw);
    let why = match refusal {
        None => unreachable!("pixel format {raw}, which Metal describes, kept raw"),
        Some(e) => format!("which the runtime refuses: {}", crate::answer::refusal(e)),
    };
    unrepresentable(
        sel,
        &format!("{what} pixel format {raw} ({name}, Metal's description's name), {why}"),
    )
}

/// The runtime's refusal `e` at `sel`, which makes `what` pixel format `raw`: a refusal of the
/// format (`Feature::Format`) of one Metal describes dies as [`lacked_as`] does, named by Metal's
/// description, whichever runtime refuses it; any other refusal is `e`, the caller's to decide.
pub(crate) fn format_refused(sel: &'static str, what: &str, raw: u64, e: rt::Error) -> rt::Error {
    if is_format_refusal(&e) && crate::gpu::format_info::described(sel, raw) {
        lacked_as(sel, what, raw, Some(&e));
    }
    e
}

/// Whether `e` refuses a pixel format (`Unsupported`, `Feature::Format`).
pub(crate) fn is_format_refusal(e: &rt::Error) -> bool {
    matches!(e, rt::Error::Unsupported(u) if u.feature == metal2vulkan::error::Feature::Format)
}

impl TextureFields {
    /// The fields of `d`, each set as its setter takes it, in the value's order.
    pub fn of(d: &TextureDesc) -> TextureFields {
        let mut desc = rt::TextureDescriptor::default();
        let raw_texture_type = match crate::gpu::runtime_texture_type(d.texture_type) {
            Some(t) => {
                desc.texture_type = t;
                None
            }
            None => Some(d.texture_type),
        };
        let (format, no_texture_format, lacked_format) =
            format_set("setPixelFormat:", d.pixel_format);
        desc.pixel_format = format;
        desc.width = d.width;
        desc.height = d.height;
        desc.depth = d.depth;
        desc.mipmap_level_count = d.mipmap_level_count;
        desc.sample_count = d.sample_count;
        desc.array_length = d.array_length;
        let (usage, private_usage) = usage_set(d.usage);
        desc.usage = usage;
        if let Some(options) = d.resource_options {
            desc.resource_options = resource_options("setResourceOptions:", options);
        }
        let protection_options = d
            .protection_options
            .unwrap_or(rt::DEFAULT_PROTECTION_OPTIONS);
        let mut raw_swizzle = None;
        if let Some(s) = d.swizzle {
            match crate::gpu::runtime_swizzle(s) {
                Some(swizzle) => desc.swizzle = swizzle,
                None => raw_swizzle = Some([s.red, s.green, s.blue, s.alpha]),
            }
        }
        // Metal places the texture at the index whether or not it is forced (measured:
        // `texture_setters_act_where_metal_does`).
        if let Some((index, _)) = d.resource_index {
            desc.placement = rt::IdPlacement::At(rt::ResourceId(index));
        }
        if let Some(allow) = d.allow_gpu_optimized_contents {
            desc.allow_gpu_optimized_contents = allow;
        }
        TextureFields {
            desc,
            framebuffer_only: d.framebuffer_only,
            is_drawable: d.is_drawable,
            write_swizzle_enabled: d.write_swizzle_enabled,
            rotation: d.rotation.unwrap_or(DEFAULT_ROTATION),
            private_usage,
            no_texture_format,
            lacked_format,
            raw_texture_type,
            raw_swizzle,
            protection_options,
        }
    }

    /// Whether the descriptor holds protection options other than none, of which Metal's
    /// `newTextureWithDescriptor:` and `newSharedTextureWithDescriptor:` answer nil once
    /// the descriptor's checks pass; every other creator makes the texture, and the heap's size
    /// query answers, as without them (measured: `texture_setters_act_where_metal_does`).
    pub fn protected(&self) -> bool {
        self.protection_options != rt::DEFAULT_PROTECTION_OPTIONS
    }

    /// Relays at `sel`, `newTextureWithDescriptor:` or `newSharedTextureWithDescriptor:`, which
    /// made a texture of the descriptor, or dies there in strict mode
    /// ([`crate::answer::Inexact::ProtectedContent`]), when the descriptor holds protection
    /// options other than none ([`TextureFields::protected`]). The texture is made as without
    /// them, as Metal's buffer, heap and IOSurface creators make one: it answers protection
    /// options 0, as theirs does (measured: `texture_setters_act_where_metal_does`, options 1, 2
    /// and 0x10), so core pages it as any other. Metal's two answer nil for them to a process
    /// without the protected-content entitlement, which only Apple's signature carries (QEMU's does
    /// not, docs/qemu.md), and make protected content only for one with it: a macOS 26 guest's
    /// WindowServer on the x86 device asks for such textures once protected video is on screen (the
    /// TV app: options 0x1, `BGRA8Unorm` and `RGBA16Float`, `Private`, usage 0x5), and each nil
    /// faulted its exec. Nothing here holds protected memory, so the texture's contents are not
    /// kept from the host's process.
    pub fn made_unprotected(&self, sel: &'static str) {
        if self.protected() {
            crate::answer::inexact(sel, crate::answer::Inexact::ProtectedContent, || {
                format!(
                    "protected content, a texture of protection options {:#x}: made as without \
                     them, as Metal's other creators make one, where Metal answers nil without \
                     the protected-content entitlement",
                    self.protection_options
                )
            });
        }
    }

    /// The pixel format as set, raw, as the descriptor answers it: one held raw, or the
    /// runtime's.
    pub fn raw_pixel_format(&self) -> u64 {
        (self.no_texture_format.or(self.lacked_format))
            .unwrap_or(u64::from(self.desc.pixel_format.as_raw()))
    }

    /// The usage bits set that Metal does not define, at which its creators abort (measured: the
    /// module's docs).
    pub fn undefined_usage(&self) -> u64 {
        self.private_usage
    }

    /// The runtime's descriptor, for a texture `maker` makes for `sel`, holding
    /// `writeSwizzleEnabled`, which the runtime decides (the module's docs):
    /// `MTLPixelFormatInvalid` and a size of 0 on any axis die, where Metal aborts making the
    /// texture (`textures_invalid_pixel_format`, `textures_backing_bad_element_size`;
    /// "MTLTextureDescriptor has width of zero", `covgap_mem_backing_zero_width`) and the runtime
    /// would refuse it, but a size of 0 of a shareable texture of storage other than `Private`,
    /// which Metal's shared creator answers nil for where it aborts at `Private`, as the runtime
    /// does (its measurement on an M3 Max under macOS 26.7), a
    /// rotation `maker` does not take ([`TextureFields::rotation_taken`]),
    /// `framebufferOnly` where `maker` is a device's, a buffer's or a heap's ([`Maker::Other`]),
    /// and a usage bit Metal does not define (the module's docs).
    ///
    /// A descriptor of a format Metal makes no texture of from a descriptor alone
    /// ([`TextureFields::formatless`]) meets none of these checks from any `maker` but the
    /// IOSurface creator: Metal answers nil before them, once its validation passes, and the
    /// runtime answers it as Metal does, from the descriptor
    /// [`crate::gpu::no_texture::descriptor`] makes, which dies first where the runtime's
    /// validation passes what Metal's aborts at. `newSharedTextureWithDescriptor:` answers nil
    /// for one before its validation, so it asks none (`VulkanDevice::new_shared_texture`). The
    /// IOSurface creator answers a format of [`crate::gpu::no_texture`] itself, and passes a
    /// two-plane one to the runtime, from this.
    pub fn descriptor(&self, sel: &'static str, maker: Maker) -> rt::TextureDescriptor {
        if maker != Maker::Surface
            && let Some(pf) = self.formatless()
        {
            return crate::gpu::no_texture::descriptor(sel, self, pf);
        }
        self.checked(sel, maker)
    }

    /// The format `self` holds of which Metal makes no texture from a descriptor alone, as the
    /// runtime's: one of [`crate::gpu::no_texture`]'s, or a two-plane YCbCr or RGB format
    /// ([`crate::gpu::ycbcr::two_plane`], held as the runtime's). `None` for any other format.
    pub fn formatless(&self) -> Option<PixelFormat> {
        match self.no_texture_format {
            Some(raw) => Some(
                u32::try_from(raw)
                    .ok()
                    .and_then(|raw| PixelFormat::try_from(raw).ok())
                    .unwrap_or_else(|| {
                        unrepresentable("setPixelFormat:", &format!("pixel format {raw}"))
                    }),
            ),
            None => {
                let pf = self.desc.pixel_format;
                crate::gpu::ycbcr::two_plane(pf).then_some(pf)
            }
        }
    }

    /// The runtime's descriptor after the checks of every format ([`TextureFields::descriptor`]).
    /// Over an IOSurface it holds a rotation other than 0, which only a `framebufferOnly` descriptor
    /// passes the checks with, and `framebufferOnly` with it, as the runtime takes a rotation
    /// only with that (the module's docs).
    fn checked(&self, sel: &'static str, maker: Maker) -> rt::TextureDescriptor {
        if let Some(raw) = self.raw_texture_type {
            metal_aborts(sel, &format!("a texture of texture type {raw}"));
        }
        if let Some([r, g, b, a]) = self.raw_swizzle {
            metal_aborts(
                sel,
                &format!(
                    "a texture of swizzle ({r}, {g}, {b}, {a}), not a valid MTLTextureSwizzle"
                ),
            );
        }
        self.rotation_taken(sel, maker);
        if maker == Maker::Other && self.framebuffer_only == Some(true) {
            metal_aborts(sel, "a framebufferOnly texture not over an IOSurface");
        }
        let unknown = self.undefined_usage();
        if unknown != 0 {
            metal_aborts(
                sel,
                &format!("usage bits {unknown:#x} Metal does not define"),
            );
        }
        // `MTLStorageModePrivate`, in a resource options' storage bits.
        const PRIVATE: u64 = 2;
        let d = &self.desc;
        let nil_first = maker == Maker::Shared && (d.resource_options.raw() >> 4) & 0xf != PRIVATE;
        if let Some(raw) = self.lacked_format {
            lacked(sel, raw);
        }
        if d.pixel_format == PixelFormat::Invalid {
            metal_aborts(sel, "a texture of MTLPixelFormatInvalid");
        }
        if (d.width == 0 || d.height == 0 || d.depth == 0) && !nil_first {
            metal_aborts(
                sel,
                &format!("a texture of {} x {} x {}", d.width, d.height, d.depth),
            );
        }
        let mut desc = self.desc;
        desc.write_swizzle_enabled = self.write_swizzle_enabled == Some(true);
        if maker == Maker::Surface && self.rotation != DEFAULT_ROTATION {
            desc.rotation = self.rotation;
            desc.framebuffer_only = true;
        }
        desc
    }

    /// Dies where Metal aborts at the rotation, making a texture by `maker` for `sel`: a rotation
    /// other than 0 of a descriptor `maker` does not see as `framebufferOnly`, one past 4, and
    /// one `maker` does not take (measured: the module's docs).
    fn rotation_taken(&self, sel: &'static str, maker: Maker) {
        let rotation = self.rotation;
        if rotation == DEFAULT_ROTATION {
            return;
        }
        if rotation > 4 {
            metal_aborts(
                sel,
                &format!("rotation {rotation}, not an MTLTextureRotation"),
            );
        }
        let framebuffer_only = self.framebuffer_only == Some(true) && maker != Maker::Shared;
        if !framebuffer_only {
            metal_aborts(
                sel,
                &format!("rotation {rotation} of a descriptor not framebufferOnly"),
            );
        }
        if maker == Maker::Other {
            metal_aborts(
                sel,
                &format!("rotation {rotation} of a texture not over an IOSurface"),
            );
        }
    }
}

impl TextureDescriptorObject {
    /// A descriptor holding `fields`.
    pub fn new(fields: TextureFields) -> TextureDescriptorObject {
        TextureDescriptorObject {
            fields: Mutex::new(fields),
        }
    }

    /// The fields as they stand.
    pub fn fields(&self) -> TextureFields {
        *self.fields.lock()
    }
}

/// `MTLTextureDescriptor *`: its fields, which core's setters change.
#[derive(Debug)]
pub struct TextureDescriptorObject {
    /// The fields.
    pub fields: Mutex<TextureFields>,
}

/// `MTLTextureDescriptor *`.
pub type VulkanTextureDescriptor = Obj<TextureDescriptorObject>;

/// `MTLRenderPassDescriptor *`: the runtime's descriptor, which core's sets change.
#[derive(Debug)]
pub struct RenderPassDescriptorObject {
    /// The runtime's descriptor.
    pub desc: Mutex<rt::RenderPassDescriptor>,
    /// The rate map set, held by reference as Metal holds it: an encoder takes its class as it is
    /// at the encoder's creation ([`crate::gpu::rate_map`]).
    rate_map: Mutex<Option<VulkanRateMap>>,
    /// What the attachments' textures are beyond the runtime's textures.
    sources: Mutex<Sources>,
    /// The attachments' properties the runtime has no value of, raw, as their setters took them
    /// ([`RawPass`]).
    raw: Mutex<RawPass>,
}

/// The values of a pass's attachment properties the runtime has no value of, raw, as Metal's
/// setters take them, by attachment (the colour attachments by index, then [`DEPTH`] and
/// [`STENCIL`]) and the property's setter: a load or store action, store action options, a
/// resolve filter. The runtime's attachment keeps the value before; core reads the load and
/// store actions back raw, and the encoder's creation asserts at one, as Metal's
/// ([`RenderPassDescriptorObject::raw_value`]).
#[derive(Debug, Default, Clone)]
pub struct RawPass(std::collections::BTreeMap<(usize, &'static str), u64>);

/// The depth attachment's place in a [`RawPass`].
pub const DEPTH: usize = rt::MAX_COLOR_ATTACHMENTS;

/// The stencil attachment's place in a [`RawPass`].
pub const STENCIL: usize = rt::MAX_COLOR_ATTACHMENTS + 1;

impl RawPass {
    /// `raw` as `from_raw` takes it, for the property `sel` of `attachment`: the runtime's value,
    /// or `None` with `raw` kept.
    fn held<T>(
        &mut self,
        (attachment, sel): (usize, &'static str),
        raw: u64,
        from_raw: impl FnOnce(u64) -> Result<T, rt::Error>,
    ) -> Option<T> {
        match from_raw(raw) {
            Ok(value) => {
                self.0.remove(&(attachment, sel));
                Some(value)
            }
            Err(_) => {
                self.0.insert((attachment, sel), raw);
                None
            }
        }
    }

    /// The raw value of `attachment`'s property `sel`, if kept.
    fn get(&self, attachment: usize, sel: &'static str) -> Option<u64> {
        self.0.get(&(attachment, sel)).copied()
    }
}

/// An attachment's texture or resolve texture as core set it, beyond the runtime's texture the
/// attachment holds: whether a pass's writes through it reach its memory
/// ([`Source::gpu_read_only`]). The default is nil's.
#[derive(Debug, Default)]
pub struct Source(Option<VulkanTexture>);

impl Clone for Source {
    fn clone(&self) -> Source {
        Source(self.0.as_ref().map(Handle::clone_ref))
    }
}

impl Source {
    /// Whether the GPU's writes to its texels never reach their memory
    /// ([`crate::gpu::resource::TextureObject::gpu_read_only()`]).
    pub fn gpu_read_only(&self) -> bool {
        self.0.as_ref().is_some_and(|t| t.gpu_read_only())
    }

    /// Whether the GPU's writes to its texels reach read-only memory, where Metal's drop them
    /// ([`crate::gpu::writes::texture_kept`]).
    pub fn writes_kept(&self) -> bool {
        (self.0.as_ref()).is_some_and(|t| crate::gpu::writes::texture_kept(t))
    }

    /// The texture, if any.
    pub fn texture(&self) -> Option<&VulkanTexture> {
        self.0.as_ref()
    }

    /// What its texture's texels lie in, as far as a pass tells them apart; `None` for nil.
    pub fn over(&self) -> Option<Over> {
        let t = self.0.as_ref()?;
        Some(match (&t.under, &t.plane) {
            _ if t.raw.parent_texture().is_some() => Over::View,
            (Some(Under::Buffer(..)), _) => Over::Buffer,
            (Some(Under::Heap(..)), _) => Over::Heap,
            (None, Some(plane)) if plane.planar => Over::Plane,
            (None, Some(_)) => Over::Surface,
            (None, None) if t.raw.is_shareable() => Over::Shareable,
            (None, None) => Over::Own,
        })
    }
}

/// What an attachment's texture or resolve texture's texels lie in ([`Source::over`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Over {
    /// Memory of its own, not shareable: `newTextureWithDescriptor:`.
    Own,
    /// Memory of its own, shareable: a texture of `newSharedTextureWithDescriptor:`, or one
    /// opened from its handle.
    Shareable,
    /// An IOSurface without planes (`newTextureWithDescriptor:iosurface:plane:` at its plane 0).
    Surface,
    /// A plane of an IOSurface with planes.
    Plane,
    /// A buffer's bytes (`newTextureWithDescriptor:offset:bytesPerRow:`).
    Buffer,
    /// A heap's memory.
    Heap,
    /// Another texture's texels: a view.
    View,
}

impl Over {
    /// What a texture whose texels lie in `over` is, as a death names it; nil for `None`.
    pub fn name(over: Option<Over>) -> &'static str {
        match over {
            Some(Over::Own) => "a texture of its own",
            Some(Over::Shareable) => "a shareable texture, or one opened from a shared handle",
            Some(Over::Surface) => "a texture over an IOSurface",
            Some(Over::Plane) => "a texture over a plane of an IOSurface with planes",
            Some(Over::Buffer) => "a texture over a buffer",
            Some(Over::Heap) => "a texture in a heap",
            Some(Over::View) => "a view of another texture",
            None => "nil",
        }
    }
}

/// The [`Source`] of each attachment's texture and resolve texture, in that order, and the
/// visibility result buffer, which a pass writes.
#[derive(Debug, Default)]
pub struct Sources {
    /// The colour attachments'.
    pub color: [[Source; 2]; rt::MAX_COLOR_ATTACHMENTS],
    /// The depth attachment's.
    pub depth: [Source; 2],
    /// The stencil attachment's.
    pub stencil: [Source; 2],
    /// The visibility result buffer.
    pub visibility: Option<VulkanBuffer>,
}

impl Clone for Sources {
    fn clone(&self) -> Sources {
        Sources {
            color: self.color.clone(),
            depth: self.depth.clone(),
            stencil: self.stencil.clone(),
            visibility: self.visibility.as_ref().map(Handle::clone_ref),
        }
    }
}

/// `MTLRenderPassDescriptor *`.
pub type VulkanRenderPassDescriptor = Obj<RenderPassDescriptorObject>;

/// A pipeline descriptor Metal makes from serialized data ([`crate::gpu::serialized`]): the
/// translator's descriptor, the resource index core sets on it (`setResourceIndex:` then
/// `setForceResourceIndex:`), and the blob's entries Metal ignores.
#[derive(Debug)]
pub struct PipelineDescriptorObject<D> {
    /// The descriptor.
    pub desc: Mutex<D>,
    /// The resource index and its force flag, when set.
    pub resource_index: Mutex<Option<(u64, bool)>>,
    /// The entries of keys the format does not define, as the blob had them, the first
    /// [`crate::gpu::serialized::KEPT_IGNORED`] alone: Metal ignores them.
    pub ignored: Vec<Ignored>,
    /// Whether the blob's `textureWriteRoundingMode` is `NEAREST_EVEN` (render blobs alone),
    /// which a state made from it is decided for ([`crate::gpu::serialized`]).
    pub nearest_even: bool,
    /// The blob's label, if it has one, which a state made from the descriptor takes.
    pub label: Option<String>,
    /// A pixel format of the blob's the runtime holds raw, which the state's creation dies at.
    pub raw_format: Option<RawFormat>,
}

impl<D> PipelineDescriptorObject<D> {
    /// A descriptor decoded from a blob, with no resource index set.
    pub fn new(decoded: Decoded<D>) -> PipelineDescriptorObject<D> {
        PipelineDescriptorObject {
            desc: Mutex::new(decoded.desc),
            resource_index: Mutex::new(None),
            ignored: decoded.ignored,
            nearest_even: decoded.nearest_even,
            label: decoded.label,
            raw_format: decoded.raw_format,
        }
    }

    /// Where a state made from the descriptor goes among the pipeline states' IDs: at the
    /// resource index core set, forced or not (Metal places the state either way, the runtime's
    /// `PipelineOptions::placement` says), else where the runtime picks.
    pub fn placement(&self) -> rt::IdPlacement {
        match *self.resource_index.lock() {
            Some((index, _)) => rt::IdPlacement::At(rt::ResourceId(index)),
            None => rt::IdPlacement::Any,
        }
    }
}

/// `MTLRenderPipelineDescriptor *`.
pub type VulkanRenderPipelineDescriptor = Obj<PipelineDescriptorObject<RenderPipelineDescriptor>>;

/// `MTLComputePipelineDescriptor *`.
pub type VulkanComputePipelineDescriptor = Obj<PipelineDescriptorObject<ComputePipelineDescriptor>>;

/// `MTLTileRenderPipelineDescriptor *`.
pub type VulkanTileRenderPipelineDescriptor =
    Obj<PipelineDescriptorObject<TileRenderPipelineDescriptor>>;

impl TextureDescriptor<VulkanPlatform> for VulkanTextureDescriptor {
    fn set_resource_options(&self, options: u64) {
        let options = resource_options("setResourceOptions:", options);
        plain(|| self.fields.lock().desc.resource_options = options);
    }

    /// Kept as set, any value (measured: the module's docs).
    fn set_rotation(&self, rotation: u64) {
        plain(|| self.fields.lock().rotation = rotation);
    }

    /// Every bit, as Metal's descriptor keeps it (measured: the module's docs).
    fn set_usage(&self, usage: u64) {
        let (usage, private_usage) = usage_set(usage);
        plain(|| {
            let mut fields = self.fields.lock();
            fields.desc.usage = usage;
            fields.private_usage = private_usage;
        });
    }

    fn set_width(&self, width: u64) {
        plain(|| self.fields.lock().desc.width = width);
    }

    fn set_height(&self, height: u64) {
        plain(|| self.fields.lock().desc.height = height);
    }

    fn set_depth(&self, depth: u64) {
        plain(|| self.fields.lock().desc.depth = depth);
    }

    /// Any value, as Metal's setter takes it: one the runtime has no type of is kept raw, and
    /// every creator aborts at it.
    fn set_texture_type(&self, raw: u64) {
        let t = crate::gpu::runtime_texture_type(raw);
        plain(|| {
            let mut fields = self.fields.lock();
            match t {
                Some(t) => {
                    fields.desc.texture_type = t;
                    fields.raw_texture_type = None;
                }
                None => fields.raw_texture_type = Some(raw),
            }
        });
    }

    /// A format the runtime has, and the other values Metal's setter takes kept raw: the formats
    /// Metal makes no texture of ([`crate::gpu::no_texture`]), and any other, a texture of which
    /// dies where it is made, or is nil ([`lacked`]). Any other value dies, as Metal's setter
    /// asserts.
    fn set_pixel_format(&self, raw: u64) {
        let (format, no_texture_format, lacked_format) = format_set("setPixelFormat:", raw);
        plain(|| {
            let mut fields = self.fields.lock();
            fields.desc.pixel_format = format;
            fields.no_texture_format = no_texture_format;
            fields.lacked_format = lacked_format;
        });
    }

    fn rotation(&self) -> u64 {
        plain(|| self.fields.lock().rotation)
    }

    /// Bits 4 to 7 of the resource options, as Metal's descriptor reads them.
    fn storage_mode(&self) -> u64 {
        plain(|| (self.fields.lock().desc.resource_options.raw() >> 4) & 0xf)
    }

    /// As set: a format Metal makes no texture of, and one the runtime lacks, too.
    fn pixel_format(&self) -> u64 {
        plain(|| self.fields.lock().raw_pixel_format())
    }

    /// As set: one the runtime has no type of too.
    fn texture_type(&self) -> u64 {
        plain(|| {
            let fields = self.fields.lock();
            fields
                .raw_texture_type
                .unwrap_or(u64::from(fields.desc.texture_type.as_raw()))
        })
    }
}

impl RenderPassDescriptorObject {
    /// A descriptor of Metal's defaults (`+new` and `+renderPassDescriptor` alike).
    pub fn new() -> RenderPassDescriptorObject {
        RenderPassDescriptorObject {
            desc: Mutex::new(rt::RenderPassDescriptor::new()),
            rate_map: Mutex::new(None),
            sources: Mutex::new(Sources::default()),
            raw: Mutex::new(RawPass::default()),
        }
    }

    /// The first property the runtime holds raw ([`RawPass`]), by attachment then setter: the
    /// attachment's name, the setter and the value.
    pub fn raw_value(&self) -> Option<(String, &'static str, u64)> {
        let first =
            plain(|| (self.raw.lock().0.first_key_value()).map(|(&at, &value)| (at, value)));
        first.map(|((attachment, sel), value)| {
            let name = match attachment {
                DEPTH => "the depth attachment".to_owned(),
                STENCIL => "the stencil attachment".to_owned(),
                i => format!("colour attachment {i}"),
            };
            (name, sel, value)
        })
    }

    /// What the attachments' textures are as they stand ([`Source`]).
    pub fn sources(&self) -> Sources {
        plain(|| self.sources.lock().clone())
    }

    /// The runtime's descriptor as it stands, without its rate map.
    pub fn descriptor(&self) -> rt::RenderPassDescriptor {
        plain(|| self.desc.lock().clone())
    }

    /// The runtime's descriptor as it stands, without its rate map, and the rate map set, whose
    /// class an encoder of it takes at the encoder's creation ([`crate::gpu::rate_map`]).
    pub fn pass(&self) -> (rt::RenderPassDescriptor, Option<VulkanRateMap>) {
        let map = plain(|| self.rate_map.lock().as_ref().map(Handle::clone_ref));
        (self.descriptor(), map)
    }
}

impl Default for RenderPassDescriptorObject {
    fn default() -> RenderPassDescriptorObject {
        RenderPassDescriptorObject::new()
    }
}

/// Core's `MTLClearColor`.
fn clear_color(c: ClearColor) -> rt::ClearColor {
    rt::ClearColor::new(c.red, c.green, c.blue, c.alpha)
}

/// The runtime's texture of a texture core sets on an attachment (`Some(None)` sets nil): the
/// attachment holds it, as Metal's attachment retains its texture, and with it the IOSurface a
/// texture over one keeps.
fn attachment_texture(t: Option<&VulkanTexture>) -> Option<rt::Texture> {
    t.map(|t| t.raw.clone())
}

/// The [`Source`] of a texture core sets on an attachment.
fn attachment_source(t: Option<&VulkanTexture>) -> Source {
    Source(t.map(Handle::clone_ref))
}

/// Sets the properties every attachment shares from `$set` on `$a`, each when set, in the order
/// `platform::values` states: texture, resolve texture, level, slice, depth plane, resolve level,
/// slice and depth plane, load and store actions, store action options. `$s` takes the textures'
/// sources, and `$raw` the values the runtime has none of, as attachment `$i`'s ([`RawPass`]).
macro_rules! set_attachment_common {
    ($set:ident, $a:ident, $s:expr, $raw:ident, $i:expr) => {
        if let Some(t) = $set.texture {
            $a.texture = attachment_texture(t);
            $s[0] = attachment_source(t);
        }
        if let Some(t) = $set.resolve_texture {
            $a.resolve_texture = attachment_texture(t);
            $s[1] = attachment_source(t);
        }
        if let Some(v) = $set.level {
            $a.level = v;
        }
        if let Some(v) = $set.slice {
            $a.slice = v;
        }
        if let Some(v) = $set.depth_plane {
            $a.depth_plane = v;
        }
        if let Some(v) = $set.resolve_level {
            $a.resolve_level = v;
        }
        if let Some(v) = $set.resolve_slice {
            $a.resolve_slice = v;
        }
        if let Some(v) = $set.resolve_depth_plane {
            $a.resolve_depth_plane = v;
        }
        if let Some(v) = $set.load_action {
            let key = ($i, "setLoadAction:");
            if let Some(v) = $raw.held(key, v, rt::LoadAction::from_raw) {
                $a.load_action = v;
            }
        }
        if let Some(v) = $set.store_action {
            let key = ($i, "setStoreAction:");
            if let Some(v) = $raw.held(key, v, rt::StoreAction::from_raw) {
                $a.store_action = v;
            }
        }
        if let Some(v) = $set.store_action_options {
            let key = ($i, "setStoreActionOptions:");
            if let Some(v) = $raw.held(key, v, rt::StoreActionOptions::from_raw) {
                $a.store_action_options = v;
            }
        }
    };
}

/// The runtime's colour attachments: Metal's `colorAttachments` array, whose
/// `objectAtIndexedSubscript:` raises past its end, as `policy` (the caller's) says.
fn color_index(policy: Policy, index: u64) -> RaiseResult<usize> {
    let sel = "objectAtIndexedSubscript:";
    let limit = rt::MAX_COLOR_ATTACHMENTS;
    match usize::try_from(index).ok().filter(|&i| i < limit) {
        Some(i) => Ok(i),
        None => send(sel, policy, || {
            Err(rt::Error::Invalid(rt::Invalid::new(
                "color_attachments",
                rt::Problem::IndexOutOfRange {
                    index: u32::try_from(index).unwrap_or(u32::MAX),
                    limit: u32::try_from(limit).unwrap_or(u32::MAX),
                },
            )))
        }),
    }
}

/// What core reads back of attachment `i`: its load and store actions, as Metal's raw values,
/// those the runtime holds raw as set ([`RawPass`]).
fn attachment_state(
    raw: &RawPass,
    i: usize,
    load: rt::LoadAction,
    store: rt::StoreAction,
) -> AttachmentState {
    AttachmentState {
        load_action: raw.get(i, "setLoadAction:").unwrap_or(load.raw()),
        store_action: raw.get(i, "setStoreAction:").unwrap_or(store.raw()),
    }
}

/// The runtime's descriptor is a value of Metal's defaults, which each typed set changes in core's
/// order. Each property is Metal's raw value as the runtime holds it: one it does not define (a
/// load or store action, store action options, a resolve filter) is kept raw, as Metal's setter
/// takes it, and the encoder's creation dies at it, where Metal asserts ([`RawPass`]). Two of the
/// pass's properties are kept nowhere: `defaultColorSampleCount` (SPI), which on this host
/// changes nothing a pass shows, and `visibilityResultType`, whose `Accumulate` the runtime does
/// not build ([`RenderPassDescriptor::set_pass`]).
impl RenderPassDescriptor<VulkanPlatform> for VulkanRenderPassDescriptor {
    /// Kept as set: the encoder's creation checks them against the pass, where Metal's
    /// assertions stop a count or a coordinate it does not take.
    fn set_sample_positions(&self, positions: &[MtlSamplePosition]) -> RaiseResult<()> {
        let positions: Vec<_> = positions
            .iter()
            .map(|p| rt::SamplePosition::new(p.x, p.y))
            .collect();
        plain(|| self.desc.lock().sample_positions = positions);
        Ok(())
    }

    fn set_pass(&self, set: &PassSet<'_, VulkanPlatform>) {
        let mut d = plain(|| self.desc.lock());
        if let Some(b) = set.visibility_result_buffer {
            d.visibility_result_buffer = b.map(|b| b.raw.clone());
            plain(|| self.sources.lock().visibility = b.map(Handle::clone_ref));
        }
        // `setVisibilityResultType:` is kept nowhere: the runtime has no accumulation, so a pass
        // of `Accumulate` resets its results as one of `Reset` does, as the AMD Radeon Pro
        // 5300M and the Intel UHD 630 do under macOS 26.7.1, where the M3 Max under macOS 27
        // adds a `Counting` result to the word's and ORs a `Boolean` one into it (measured), as
        // Metal's docs promise. Accumulating is the runtime's to build.
        if set.visibility_result_type.is_some_and(|t| t != 0) {
            static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
                log::warn!(
                    target: "reims_vgpu_vulkan::gpu",
                    "a render pass of visibilityResultType Accumulate: the runtime resets its \
                     visibility results"
                );
            }
        }
        if let Some(v) = set.render_target_array_length {
            d.render_target_array_length = v;
        }
        if let Some(v) = set.render_target_width {
            d.render_target_width = v;
        }
        if let Some(v) = set.render_target_height {
            d.render_target_height = v;
        }
        if let Some(v) = set.default_raster_sample_count {
            d.default_raster_sample_count = v;
        }
        // `setDefaultColorSampleCount:` (SPI) is kept nowhere: on this host it changes nothing a
        // pass shows, with attachments or without (`mutgap_dec_render_default_color_samples`).
        let _ = set.default_color_sample_count;
        if let Some(map) = set.rasterization_rate_map {
            *plain(|| self.rate_map.lock()) = map.map(Handle::clone_ref);
        }
        if let Some(v) = set.imageblock_sample_length {
            d.imageblock_sample_length = v;
        }
        if let Some(v) = set.threadgroup_memory_length {
            d.threadgroup_memory_length = v;
        }
        if let Some(v) = set.tile_width {
            d.tile_width = v;
        }
        if let Some(v) = set.tile_height {
            d.tile_height = v;
        }
    }

    /// An index past the eight attachments raises before any set, as Metal's subscript does.
    fn set_color_attachment(
        &self,
        on: OnFault,
        index: u64,
        set: &ColorAttachmentSet<'_, VulkanPlatform>,
    ) -> RaiseResult<()> {
        let i = color_index(Policy::On(on), index)?;
        let mut d = plain(|| self.desc.lock());
        let mut sources = plain(|| self.sources.lock());
        let mut raw = plain(|| self.raw.lock());
        let a = &mut d.color_attachments[i];
        set_attachment_common!(set, a, sources.color[i], raw, i);
        if let Some(c) = set.clear_color {
            a.clear_color = clear_color(c);
        }
        Ok(())
    }

    fn set_depth_attachment(&self, set: &DepthAttachmentSet<'_, VulkanPlatform>) {
        let mut d = plain(|| self.desc.lock());
        let mut sources = plain(|| self.sources.lock());
        let mut raw = plain(|| self.raw.lock());
        let a = &mut d.depth_attachment;
        set_attachment_common!(set, a, sources.depth, raw, DEPTH);
        if let Some(v) = set.clear_depth {
            a.clear_depth = v;
        }
        if let Some(v) = set.depth_resolve_filter {
            let key = (DEPTH, "setDepthResolveFilter:");
            if let Some(v) = raw.held(key, v, rt::MultisampleDepthResolveFilter::from_raw) {
                a.depth_resolve_filter = v;
            }
        }
    }

    fn set_stencil_attachment(&self, set: &StencilAttachmentSet<'_, VulkanPlatform>) {
        let mut d = plain(|| self.desc.lock());
        let mut sources = plain(|| self.sources.lock());
        let mut raw = plain(|| self.raw.lock());
        let a = &mut d.stencil_attachment;
        set_attachment_common!(set, a, sources.stencil, raw, STENCIL);
        if let Some(v) = set.clear_stencil {
            a.clear_stencil = v;
        }
        if let Some(v) = set.stencil_resolve_filter {
            let key = (STENCIL, "setStencilResolveFilter:");
            if let Some(v) = raw.held(key, v, rt::MultisampleStencilResolveFilter::from_raw) {
                a.stencil_resolve_filter = v;
            }
        }
    }

    /// An index past the eight attachments raises, the caller's to catch.
    fn color_attachment_state(&self, index: u64) -> RaiseResult<AttachmentState> {
        let i = color_index(Policy::Caught, index)?;
        let d = plain(|| self.desc.lock());
        let raw = plain(|| self.raw.lock());
        let a = &d.color_attachments[i];
        Ok(attachment_state(&raw, i, a.load_action, a.store_action))
    }

    fn depth_attachment_state(&self) -> AttachmentState {
        let d = plain(|| self.desc.lock());
        let raw = plain(|| self.raw.lock());
        let a = &d.depth_attachment;
        attachment_state(&raw, DEPTH, a.load_action, a.store_action)
    }

    fn stencil_attachment_state(&self) -> AttachmentState {
        let d = plain(|| self.desc.lock());
        let raw = plain(|| self.raw.lock());
        let a = &d.stencil_attachment;
        attachment_state(&raw, STENCIL, a.load_action, a.store_action)
    }
}

/// Kept for the state's creation, which places the state there
/// ([`PipelineDescriptorObject::placement`]).
impl descriptor::RenderPipelineDescriptor<VulkanPlatform> for VulkanRenderPipelineDescriptor {
    fn set_resource_index(&self, resource_index: u64, force: bool) {
        plain(|| *self.resource_index.lock() = Some((resource_index, force)));
    }
}

/// Kept for the state's creation, which places the state there
/// ([`PipelineDescriptorObject::placement`]).
impl descriptor::ComputePipelineDescriptor<VulkanPlatform> for VulkanComputePipelineDescriptor {
    fn set_resource_index(&self, resource_index: u64, force: bool) {
        plain(|| *self.resource_index.lock() = Some((resource_index, force)));
    }
}

impl TileTrait<VulkanPlatform> for VulkanTileRenderPipelineDescriptor {}
