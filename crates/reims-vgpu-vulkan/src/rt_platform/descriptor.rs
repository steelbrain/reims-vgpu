//! The descriptors the rt platform keeps as objects: the texture descriptor, the render pass
//! descriptor and the rasterization rate map's.
//!
//! - **The texture descriptor** keeps Metal's raw values as they are set ([`TextureDesc`]) and
//!   reads them back as Metal's descriptor reads them. A texture's creation is what turns them into
//!   rt's descriptor. Metal also ends the process at a `setPixelFormat:` of a format it makes no
//!   texture of. The rt platform keeps any value, so it is the texture's creation that refuses it.
//! - **The render pass descriptor** keeps what rt's pass has a field of, as it is set
//!   ([`PassFields`]), from Metal's defaults: each attachment's texture, held as Metal's
//!   descriptor retains it, its level and slice, its load and store actions as Metal's raw values,
//!   which its reads answer, and its clear value; its resolve texture, held so too, its resolve
//!   level, slice and depth plane, and a depth or stencil attachment's resolve filter as Metal's
//!   raw value, by which rt's pass resolves where a store action resolves (rt's docs); the render
//!   target's width and height; the render target array length, by which rt's pass renders, at 2
//!   or more, that many slices of each attachment and resolve texture from its own slice, loading,
//!   storing and resolving each (rt's docs); the default raster sample count, at which rt's pass
//!   with no attachment renders, refused at 0, Metal's default, as Metal refuses it (rt's docs);
//!   the visibility result buffer, held as Metal's descriptor retains it; and the
//!   `visibilityResultType`. rt's command buffer keeps that buffer until it retires, and rt's pass
//!   writes the result of each `Boolean` or `Counting` mode its encoder sets at the mode's offset,
//!   every other byte staying as it was; in a pass of `Accumulate` such a mode is rt's gap at the
//!   encoder (`super::render`). A render encoder's creation turns them into rt's
//!   descriptor ([`PassFields::rt_pass`]): an action, resolve filter or visibility result type
//!   Metal has none of, or a level, slice or plane past 32 bits, is Metal's refusal there, as
//!   Metal asserts at the encoder's creation; a width or height past 32 bits renders the attachments' area, as Metal's
//!   does; and an array length past 32 bits is past every device's layers, which rt refuses as it
//!   refuses any other past them, as it refuses an attachment with fewer slices from its own than
//!   the length. A colour attachment past the eighth raises at its subscript, as Metal's does.
//!   - **Kept nowhere, changing nothing rt renders**: a depth plane (rt renders into no 3D
//!     texture); store action options (rt builds none); and `defaultColorSampleCount` (SPI).
//!   - **Unbuilt** ([`Unbuilt::RenderPass`]), counted at a set of other than Metal's default, and
//!     kept nowhere, so the pass renders without them: an imageblock sample length, a threadgroup
//!     memory length, a tile width or height, and sample positions. rt's pass renders its samples
//!     at Vulkan's standard locations, which are Metal's default positions of 1, 2 and 4 samples
//!     ([`METAL_DEFAULT_SAMPLE_POSITIONS`]), so a set of positions equal to Metal's defaults of
//!     their count renders as Metal's does and counts nothing; a set that differs, and a set of
//!     any other count, whose defaults are not measured, is counted, the process's first line of
//!     the piece, where such a set writes it, naming the set's count and each position
//!     ([`SamplePositions`]). No rasterization rate map is ever made to set.
//! - **The rate map descriptor** is unbuilt ([`Unbuilt::RateMap`]): rt has no rate maps.

use crate::core::platform::descriptor::{RenderPassDescriptor, TextureDescriptor};
use crate::core::platform::rate_map::RateMapDescriptor;
use crate::core::platform::types::{ClearColor, MtlSamplePosition};
use crate::core::platform::values::{
    AttachmentState, ColorAttachmentSet, DepthAttachmentSet, PassSet, StencilAttachmentSet,
    TextureDesc,
};
use crate::core::platform::{Handle, Platform};
use crate::core::raise::{OnFault, Raise, RaiseResult};
use crate::core::sync::{Mutex, MutexGuard};
use metal2vulkan_runtime as rt;

use super::diag::{InBuffer, Point, Target};
use super::tiled::Tiled;
use super::unbuilt::{Unbuilt, refuse, refuse_with};
use super::{RtBuffer, RtObj, RtPlatform, RtTexture};

/// `MTLTextureDescriptor`: the values set, as Metal's descriptor keeps them.
pub struct TextureFields(Mutex<TextureDesc>);

impl TextureFields {
    /// A descriptor of `desc`'s values.
    pub fn new(desc: TextureDesc) -> TextureFields {
        TextureFields(Mutex::new(desc))
    }

    /// The values set so far.
    pub fn desc(&self) -> TextureDesc {
        *self.0.lock()
    }

    /// The values, to set or read.
    fn lock(&self) -> MutexGuard<'_, TextureDesc> {
        self.0.lock()
    }
}

/// Apple's private texture usage 0x8, which Metal's texture drops.
const USAGE_0X8: u32 = 0x8;

/// rt's descriptor of a texture descriptor's values `d`: each raw value as rt's type of it, the
/// storage mode of bits 4 to 7 of the resource options (Shared before any set), and each channel
/// from itself before a swizzle is set. A usage of 0x8 alone is `ShaderRead`: Metal makes a texture
/// of 0x8 alone as one of `ShaderRead` alone in every respect it shows, its heap size included,
/// and not as one of `Unknown` usage, which rt takes as every usage (measured, `crate::gpu::desc`'s
/// docs); the texture answers the guest a usage of 0 all the same. The private values (`resourceIndex`, `protectionOptions`,
/// `framebufferOnly`, `isDrawable`, `writeSwizzleEnabled`, `allowGPUOptimizedContents` and
/// `rotation`) have no field in rt's descriptor and are not sent.
///
/// Err: the first value rt's descriptor has no value of, named.
pub fn rt_texture_descriptor(d: &TextureDesc) -> Result<rt::TextureDescriptor, String> {
    let word = |what: &str, raw: u64| u32::try_from(raw).map_err(|_| format!("{what} {raw}"));
    let texture_type = word("texture type", d.texture_type)?;
    let texture_type = rt::TextureType::try_from(texture_type)
        .map_err(|_| format!("texture type {texture_type}"))?;
    let pixel_format = word("pixel format", d.pixel_format)?;
    let pixel_format = rt::PixelFormat::try_from(pixel_format)
        .map_err(|_| format!("pixel format {pixel_format}"))?;
    let (width, height) = (word("width", d.width)?, word("height", d.height)?);
    let mut out = rt::TextureDescriptor::new(texture_type, pixel_format, width, height);
    out.depth = word("depth", d.depth)?;
    out.mipmap_level_count = word("mipmap level count", d.mipmap_level_count)?;
    out.sample_count = word("sample count", d.sample_count)?;
    out.array_length = word("array length", d.array_length)?;
    let usage = match word("texture usage", d.usage)? {
        USAGE_0X8 => rt::TextureUsage::SHADER_READ.as_raw(),
        usage => usage,
    };
    out.usage =
        rt::TextureUsage::try_from(usage).map_err(|_| format!("texture usage {usage:#x}"))?;
    out.storage_mode = match (d.resource_options.unwrap_or(0) >> 4) & 0xf {
        0 => rt::StorageMode::Shared,
        1 => rt::StorageMode::Managed,
        2 => rt::StorageMode::Private,
        3 => rt::StorageMode::Memoryless,
        mode => return Err(format!("storage mode {mode}")),
    };
    if let Some(s) = d.swizzle {
        out.swizzle = crate::gpu::runtime_swizzle(s).ok_or_else(|| {
            let channels = [s.red, s.green, s.blue, s.alpha];
            format!("texture swizzle {channels:?}")
        })?;
    }
    Ok(out)
}

/// rt's storage mode `mode` as Metal numbers it (`MTLStorageMode`), the resource options' bits 4
/// to 7 that [`rt_texture_descriptor`] reads it from.
pub fn storage_number(mode: rt::StorageMode) -> u32 {
    match mode {
        rt::StorageMode::Shared => 0,
        rt::StorageMode::Managed => 1,
        rt::StorageMode::Private => 2,
        rt::StorageMode::Memoryless => 3,
    }
}

/// The values of `texture2DDescriptorWithPixelFormat:width:height:mipmapped:`: a 2D texture of
/// the format and size, of every level down to 1×1 where it is mipmapped (the levels of the larger
/// side), and Metal's defaults for the rest.
pub fn texture_2d_desc(pixel_format: u64, width: u64, height: u64, mipmapped: bool) -> TextureDesc {
    let levels = if mipmapped {
        u64::from(u64::BITS - width.max(height).max(1).leading_zeros())
    } else {
        1
    };
    TextureDesc {
        pixel_format,
        width,
        height,
        mipmap_level_count: levels,
        ..TextureDesc::default()
    }
}

impl<P: Platform> TextureDescriptor<P> for RtObj<TextureFields> {
    fn set_resource_options(&self, resource_options: u64) {
        let recorded_call =
            crate::recorder::rec_call!("setResourceOptions:", self, resource_options);
        crate::recorder::run(|| {
            self.lock().resource_options = Some(resource_options);
        });
        crate::recorder::rec_done!(recorded_call, ());
    }
    fn set_rotation(&self, rotation: u64) {
        let recorded_call = crate::recorder::rec_call!("setRotation:", self, rotation);
        crate::recorder::run(|| {
            self.lock().rotation = Some(rotation);
        });
        crate::recorder::rec_done!(recorded_call, ());
    }
    fn set_usage(&self, usage: u64) {
        let recorded_call = crate::recorder::rec_call!("setUsage:", self, usage);
        crate::recorder::run(|| {
            self.lock().usage = usage;
        });
        crate::recorder::rec_done!(recorded_call, ());
    }
    fn set_width(&self, width: u64) {
        let recorded_call = crate::recorder::rec_call!("setWidth:", self, width);
        crate::recorder::run(|| {
            self.lock().width = width;
        });
        crate::recorder::rec_done!(recorded_call, ());
    }
    fn set_height(&self, height: u64) {
        let recorded_call = crate::recorder::rec_call!("setHeight:", self, height);
        crate::recorder::run(|| {
            self.lock().height = height;
        });
        crate::recorder::rec_done!(recorded_call, ());
    }
    fn set_depth(&self, depth: u64) {
        let recorded_call = crate::recorder::rec_call!("setDepth:", self, depth);
        crate::recorder::run(|| {
            self.lock().depth = depth;
        });
        crate::recorder::rec_done!(recorded_call, ());
    }
    fn set_texture_type(&self, texture_type: u64) {
        let recorded_call = crate::recorder::rec_call!("setTextureType:", self, texture_type);
        crate::recorder::run(|| {
            self.lock().texture_type = texture_type;
        });
        crate::recorder::rec_done!(recorded_call, ());
    }
    fn set_pixel_format(&self, pixel_format: u64) {
        let recorded_call = crate::recorder::rec_call!("setPixelFormat:", self, pixel_format);
        crate::recorder::run(|| {
            self.lock().pixel_format = pixel_format;
        });
        crate::recorder::rec_done!(recorded_call, ());
    }
    /// As set, 0 before any set.
    fn rotation(&self) -> u64 {
        self.lock().rotation.unwrap_or(0)
    }
    /// Bits 4 to 7 of the resource options, as Metal's descriptor reads them: 0 before any set.
    fn storage_mode(&self) -> u64 {
        (self.lock().resource_options.unwrap_or(0) >> 4) & 0xf
    }
    fn pixel_format(&self) -> u64 {
        self.lock().pixel_format
    }
    fn texture_type(&self) -> u64 {
        self.lock().texture_type
    }
}

/// `MTLLoadActionDontCare`.
const LOAD_DONT_CARE: u64 = 0;

/// `MTLLoadActionClear`.
const LOAD_CLEAR: u64 = 2;

/// `MTLStoreActionDontCare`.
const STORE_DONT_CARE: u64 = 0;

/// `MTLStoreActionStore`.
const STORE_STORE: u64 = 1;

/// The colour attachments' subscript, which raises past the eighth.
const SUBSCRIPT: &str = "objectAtIndexedSubscript:";

/// `MTLRenderPassDescriptor`: what is set, as Metal's descriptor keeps it (the module's docs).
#[derive(Default)]
pub struct PassFields(Mutex<Pass>);

/// One attachment as set: its texture, held as Metal's descriptor retains it, its level and slice,
/// its load and store actions as Metal's raw values, its resolve texture, held so too, its resolve
/// level, slice and depth plane, and its resolve filter, Metal's raw value (a depth or stencil
/// attachment's; a colour attachment has none).
struct Attachment {
    texture: Option<RtObj<RtTexture>>,
    level: u64,
    slice: u64,
    load_action: u64,
    store_action: u64,
    resolve_texture: Option<RtObj<RtTexture>>,
    resolve_level: u64,
    resolve_slice: u64,
    resolve_depth_plane: u64,
    resolve_filter: u64,
}

/// What a set sends of an attachment's resolve: its texture (nil for `Some(None)`), level, slice
/// and depth plane, and its filter, each `None` where it is not sent.
#[derive(Default)]
struct ResolveSet<'a> {
    texture: Option<Option<&'a RtObj<RtTexture>>>,
    level: Option<u64>,
    slice: Option<u64>,
    depth_plane: Option<u64>,
    filter: Option<u64>,
}

impl Attachment {
    /// One of no texture, level and slice 0, the actions `load_action` and `store_action`, and no
    /// resolve texture, its level, slice, plane and filter 0.
    fn of(load_action: u64, store_action: u64) -> Attachment {
        Attachment {
            texture: None,
            level: 0,
            slice: 0,
            load_action,
            store_action,
            resolve_texture: None,
            resolve_level: 0,
            resolve_slice: 0,
            resolve_depth_plane: 0,
            resolve_filter: 0,
        }
    }

    /// Takes the texture, level, slice, actions and resolve a set sends; nil for `Some(None)`.
    fn set(
        &mut self,
        texture: Option<Option<&RtObj<RtTexture>>>,
        (level, slice): (Option<u64>, Option<u64>),
        (load_action, store_action): (Option<u64>, Option<u64>),
        resolve: ResolveSet<'_>,
    ) {
        if let Some(texture) = texture {
            self.texture = texture.map(Handle::clone_ref);
        }
        self.level = level.unwrap_or(self.level);
        self.slice = slice.unwrap_or(self.slice);
        self.load_action = load_action.unwrap_or(self.load_action);
        self.store_action = store_action.unwrap_or(self.store_action);
        if let Some(texture) = resolve.texture {
            self.resolve_texture = texture.map(Handle::clone_ref);
        }
        self.resolve_level = resolve.level.unwrap_or(self.resolve_level);
        self.resolve_slice = resolve.slice.unwrap_or(self.resolve_slice);
        self.resolve_depth_plane = resolve.depth_plane.unwrap_or(self.resolve_depth_plane);
        self.resolve_filter = resolve.filter.unwrap_or(self.resolve_filter);
    }

    /// Its actions, as Metal reads them.
    fn state(&self) -> AttachmentState {
        AttachmentState {
            load_action: self.load_action,
            store_action: self.store_action,
        }
    }

    /// rt's texture, level, slice, actions and resolve of it, which `what` names; `Err` names a
    /// level, slice or plane past 32 bits, or an action Metal has none of.
    fn rt(&self, what: &str) -> Result<RtAttachment, String> {
        let word =
            |name: &str, raw: u64| u32::try_from(raw).map_err(|_| format!("{what}'s {name} {raw}"));
        let load = match self.load_action {
            0 => rt::LoadAction::DontCare,
            1 => rt::LoadAction::Load,
            2 => rt::LoadAction::Clear,
            raw => return Err(format!("{what}'s load action {raw}")),
        };
        let store = match self.store_action {
            0 => rt::StoreAction::DontCare,
            1 => rt::StoreAction::Store,
            2 => rt::StoreAction::MultisampleResolve,
            3 => rt::StoreAction::StoreAndMultisampleResolve,
            4 => rt::StoreAction::Unknown,
            5 => rt::StoreAction::CustomSampleDepthStore,
            raw => return Err(format!("{what}'s store action {raw}")),
        };
        Ok(RtAttachment {
            texture: self.texture.as_ref().map(|t| t.rt().clone()),
            level: word("level", self.level)?,
            slice: word("slice", self.slice)?,
            load,
            store,
            resolve_texture: self.resolve_texture.as_ref().map(|t| t.rt().clone()),
            resolve_level: word("resolve level", self.resolve_level)?,
            resolve_slice: word("resolve slice", self.resolve_slice)?,
            resolve_depth_plane: word("resolve depth plane", self.resolve_depth_plane)?,
        })
    }
}

/// What rt's attachment descriptors share, of one attachment.
struct RtAttachment {
    texture: Option<rt::Texture>,
    level: u32,
    slice: u32,
    load: rt::LoadAction,
    store: rt::StoreAction,
    resolve_texture: Option<rt::Texture>,
    resolve_level: u32,
    resolve_slice: u32,
    resolve_depth_plane: u32,
}

/// `MTLVisibilityResultTypeReset`, Metal's default.
const VISIBILITY_RESET: u64 = 0;
/// `MTLVisibilityResultTypeAccumulate`.
const VISIBILITY_ACCUMULATE: u64 = 1;

/// What a render pass descriptor keeps.
struct Pass {
    colors: [Attachment; rt::MAX_COLOR_ATTACHMENTS],
    clear_colors: [ClearColor; rt::MAX_COLOR_ATTACHMENTS],
    depth: Attachment,
    clear_depth: f64,
    stencil: Attachment,
    clear_stencil: u32,
    render_target_width: u64,
    render_target_height: u64,
    render_target_array_length: u64,
    default_raster_sample_count: u64,
    visibility_result_buffer: Option<RtObj<RtBuffer>>,
    /// `MTLVisibilityResultType`, Metal's raw value.
    visibility_result_type: u64,
}

impl Default for Pass {
    /// Metal's: colour attachments of `DontCare` loads and `Store` stores, cleared to opaque
    /// black; depth and stencil of `Clear` loads and `DontCare` stores, cleared to 1 and 0; no
    /// texture, level and slice 0; no render target size, a render target array length and a
    /// default raster sample count of 0, no visibility result buffer, and a visibility result type
    /// of `Reset`.
    fn default() -> Pass {
        let black = ClearColor {
            alpha: 1.0,
            ..ClearColor::default()
        };
        Pass {
            colors: std::array::from_fn(|_| Attachment::of(LOAD_DONT_CARE, STORE_STORE)),
            clear_colors: [black; rt::MAX_COLOR_ATTACHMENTS],
            depth: Attachment::of(LOAD_CLEAR, STORE_DONT_CARE),
            clear_depth: 1.0,
            stencil: Attachment::of(LOAD_CLEAR, STORE_DONT_CARE),
            clear_stencil: 0,
            render_target_width: 0,
            render_target_height: 0,
            render_target_array_length: 0,
            default_raster_sample_count: 0,
            visibility_result_buffer: None,
            visibility_result_type: VISIBILITY_RESET,
        }
    }
}

impl PassFields {
    /// The pass's values, to set or read.
    fn lock(&self) -> MutexGuard<'_, Pass> {
        self.0.lock()
    }

    /// rt's descriptor of what is set (the module's docs). `Err` names the first value Metal
    /// takes no pass of: an action, resolve filter or visibility result type Metal has none of, or
    /// a level, slice or plane past 32 bits.
    pub fn rt_pass(&self) -> Result<rt::RenderPassDescriptor, String> {
        let pass = self.lock();
        let mut out = rt::RenderPassDescriptor::default();
        let colors = (out.color_attachments.iter_mut())
            .zip(&pass.colors)
            .zip(&pass.clear_colors);
        for (index, ((out, a), c)) in colors.enumerate() {
            let a = a.rt(&format!("colour attachment {index}"))?;
            *out = rt::RenderPassColorAttachmentDescriptor {
                texture: a.texture,
                level: a.level,
                slice: a.slice,
                load_action: a.load,
                store_action: a.store,
                clear_color: rt::ClearColor::new(c.red, c.green, c.blue, c.alpha),
                resolve_texture: a.resolve_texture,
                resolve_level: a.resolve_level,
                resolve_slice: a.resolve_slice,
                resolve_depth_plane: a.resolve_depth_plane,
            };
        }
        let what = "the depth attachment";
        let a = pass.depth.rt(what)?;
        out.depth_attachment = rt::RenderPassDepthAttachmentDescriptor {
            texture: a.texture,
            level: a.level,
            slice: a.slice,
            load_action: a.load,
            store_action: a.store,
            clear_depth: pass.clear_depth,
            resolve_texture: a.resolve_texture,
            resolve_level: a.resolve_level,
            resolve_slice: a.resolve_slice,
            resolve_depth_plane: a.resolve_depth_plane,
            depth_resolve_filter: match pass.depth.resolve_filter {
                0 => rt::MultisampleDepthResolveFilter::Sample0,
                1 => rt::MultisampleDepthResolveFilter::Min,
                2 => rt::MultisampleDepthResolveFilter::Max,
                raw => return Err(format!("{what}'s depth resolve filter {raw}")),
            },
        };
        let what = "the stencil attachment";
        let a = pass.stencil.rt(what)?;
        out.stencil_attachment = rt::RenderPassStencilAttachmentDescriptor {
            texture: a.texture,
            level: a.level,
            slice: a.slice,
            load_action: a.load,
            store_action: a.store,
            clear_stencil: pass.clear_stencil,
            resolve_texture: a.resolve_texture,
            resolve_level: a.resolve_level,
            resolve_slice: a.resolve_slice,
            resolve_depth_plane: a.resolve_depth_plane,
            stencil_resolve_filter: match pass.stencil.resolve_filter {
                0 => rt::MultisampleStencilResolveFilter::Sample0,
                1 => rt::MultisampleStencilResolveFilter::DepthResolvedSample,
                raw => return Err(format!("{what}'s stencil resolve filter {raw}")),
            },
        };
        // A size past 32 bits renders the area a size of the most 32 bits do: the attachments'.
        out.render_target_width = u32::try_from(pass.render_target_width).unwrap_or(u32::MAX);
        out.render_target_height = u32::try_from(pass.render_target_height).unwrap_or(u32::MAX);
        // A length past 32 bits is past every device's layers, which rt refuses as any other.
        let layers = pass.render_target_array_length;
        out.render_target_array_length = u32::try_from(layers).unwrap_or(u32::MAX);
        // A count past 32 bits is no count Metal takes; rt refuses it as it refuses any other.
        let samples = pass.default_raster_sample_count;
        out.default_raster_sample_count = u32::try_from(samples).unwrap_or(u32::MAX);
        let kept = pass.visibility_result_buffer.as_ref();
        out.visibility_result_buffer = kept.map(|b| b.rt().clone());
        out.visibility_result_type = match pass.visibility_result_type {
            VISIBILITY_RESET => rt::VisibilityResultType::Reset,
            VISIBILITY_ACCUMULATE => rt::VisibilityResultType::Accumulate,
            raw => return Err(format!("the visibility result type {raw}")),
        };
        Ok(out)
    }

    /// The pass's targets, for the boot diagnostics (`super::diag`): each attachment of a texture,
    /// with its level and slice, but one whose level or slice is past 32 bits, which rt's pass
    /// takes none of.
    pub fn targets(&self) -> Vec<Target> {
        let pass = self.lock();
        let one = |point: Point, a: &Attachment| {
            let texture = a.texture.as_ref()?;
            let d = texture.rt().descriptor();
            Some(Target {
                point,
                level: u32::try_from(a.level).ok()?,
                slice: u32::try_from(a.slice).ok()?,
                texture_type: d.texture_type.as_raw(),
                samples: d.sample_count,
                index: texture.rt().gpu_resource_id().index,
                id: texture.named().id(),
                made: std::sync::Arc::clone(texture.made()),
            })
        };
        let colors = (pass.colors.iter().enumerate()).filter_map(|(i, a)| one(Point::Color(i), a));
        (colors.chain(one(Point::Depth, &pass.depth)))
            .chain(one(Point::Stencil, &pass.stencil))
            .collect()
    }

    /// The tiled textures the pass's attachments render, each attachment's (`super::tiled`).
    pub fn tiled(&self) -> Vec<std::sync::Arc<Tiled>> {
        let pass = self.lock();
        let attachments = (pass.colors.iter()).chain([&pass.depth, &pass.stencil]);
        (attachments.filter_map(|a| a.texture.as_ref()?.tiled().cloned())).collect()
    }

    /// Notes in `used` the copies the pass's attachments, their resolves and its visibility
    /// result buffer lie over ([`super::copied::Used`]), each in a use that can write it.
    pub fn note_copies(&self, used: &super::copied::Used) {
        if !super::copied::any() {
            return;
        }
        let pass = self.lock();
        let attachments = (pass.colors.iter()).chain([&pass.depth, &pass.stencil]);
        for a in attachments {
            for texture in [&a.texture, &a.resolve_texture].into_iter().flatten() {
                used.add(texture.copied(), true);
            }
        }
        if let Some(buffer) = &pass.visibility_result_buffer {
            used.buffer(buffer);
        }
    }
}

/// Colour attachment `index`, or Metal's raise at its subscript past the eighth.
fn color_index(index: u64) -> Result<usize, Raise> {
    match usize::try_from(index) {
        Ok(i) if i < rt::MAX_COLOR_ATTACHMENTS => Ok(i),
        _ => Err(Raise::platform(SUBSCRIPT).with_detail(format!(
            "colour attachment {index} of {}",
            rt::MAX_COLOR_ATTACHMENTS
        ))),
    }
}

/// Metal's default sample positions of 1, 2 and 4 samples, as `getDefaultSamplePositions:count:`
/// answers them (measured on the M3 Max, macOS 27: `guest/bundle/tools/samplepos_probe.m`, which
/// compares the device's answers with these): Vulkan's standard locations of those counts, at
/// which rt's pass renders. No other count's defaults are measured.
const METAL_DEFAULT_SAMPLE_POSITIONS: [&[MtlSamplePosition]; 3] = [
    &[MtlSamplePosition { x: 0.5, y: 0.5 }],
    &[
        MtlSamplePosition { x: 0.75, y: 0.75 },
        MtlSamplePosition { x: 0.25, y: 0.25 },
    ],
    &[
        MtlSamplePosition { x: 0.375, y: 0.125 },
        MtlSamplePosition { x: 0.875, y: 0.375 },
        MtlSamplePosition { x: 0.125, y: 0.625 },
        MtlSamplePosition { x: 0.625, y: 0.875 },
    ],
];

/// Whether `positions` are Metal's default positions of their count, each `f32` equal
/// ([`METAL_DEFAULT_SAMPLE_POSITIONS`]): never at a count whose defaults are not measured.
fn metal_default_sample_positions(positions: &[MtlSamplePosition]) -> bool {
    METAL_DEFAULT_SAMPLE_POSITIONS.contains(&positions)
}

/// A set of sample positions as the line of a refusal names them: their count, then each
/// `(x, y)`, each coordinate in the shortest form that reads back as its `f32`, so two sets that
/// differ print differently.
struct SamplePositions<'a>(&'a [MtlSamplePosition]);

impl std::fmt::Display for SamplePositions<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "sample positions, count {}:", self.0.len())?;
        for p in self.0 {
            write!(f, " ({:?}, {:?})", p.x, p.y)?;
        }
        Ok(())
    }
}

impl RenderPassDescriptor<RtPlatform> for RtObj<PassFields> {
    /// Counted ([`Unbuilt::RenderPass`]) where the positions differ from Metal's default positions
    /// of their count ([`metal_default_sample_positions`]), and at a count whose defaults are not
    /// measured: rt's pass keeps none, and renders its samples at Vulkan's standard locations,
    /// which rt requires of the device: Metal's default positions of 1, 2 and 4 samples. A set
    /// equal to them, or of none, renders as Metal's does, counting and writing nothing. The
    /// process's first line of the piece, where a set writes it, names the set's count and each
    /// position ([`SamplePositions`]).
    fn set_sample_positions(&self, positions: &[MtlSamplePosition]) -> RaiseResult<()> {
        let recorded_call =
            crate::recorder::rec_call!("setSamplePositions:count:", self, positions);
        let recorded_answer = crate::recorder::run(|| -> RaiseResult<()> {
            if !positions.is_empty() && !metal_default_sample_positions(positions) {
                let set = SamplePositions(positions);
                refuse_with(Unbuilt::RenderPass, "setSamplePositions:count:", &set);
            }
            Ok(())
        });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// The render target's width, height and array length, the default raster sample count, the
    /// visibility result buffer and its type; the rest as the module's docs say.
    fn set_pass(&self, set: &PassSet<'_, RtPlatform>) {
        let recorded_call = crate::recorder::rec_call!("m2v.setPass", self, set);
        crate::recorder::run(|| {
            let unbuilt = [
                (
                    set.imageblock_sample_length.is_some_and(|n| n != 0),
                    "setImageblockSampleLength:",
                ),
                (
                    set.threadgroup_memory_length.is_some_and(|n| n != 0),
                    "setThreadgroupMemoryLength:",
                ),
                (set.tile_width.is_some_and(|n| n != 0), "setTileWidth:"),
                (set.tile_height.is_some_and(|n| n != 0), "setTileHeight:"),
            ];
            for (_, sel) in unbuilt.into_iter().filter(|&(sent, _)| sent) {
                refuse(Unbuilt::RenderPass, sel);
            }
            let mut pass = self.lock();
            if let Some(width) = set.render_target_width {
                pass.render_target_width = width;
            }
            if let Some(height) = set.render_target_height {
                pass.render_target_height = height;
            }
            if let Some(layers) = set.render_target_array_length {
                pass.render_target_array_length = layers;
            }
            if let Some(samples) = set.default_raster_sample_count {
                pass.default_raster_sample_count = samples;
            }
            if let Some(buffer) = set.visibility_result_buffer {
                pass.visibility_result_buffer = buffer.map(Handle::clone_ref);
            }
            if let Some(kind) = set.visibility_result_type {
                pass.visibility_result_type = kind;
            }
        });
        crate::recorder::rec_done!(recorded_call, ());
    }

    /// An index past the eight attachments raises before any set, as Metal's subscript does.
    fn set_color_attachment(
        &self,
        on: OnFault,
        index: u64,
        set: &ColorAttachmentSet<'_, RtPlatform>,
    ) -> RaiseResult<()> {
        let recorded_call = crate::recorder::rec_call!("m2v.setColorAttachment", self, index, set);
        let recorded_answer = crate::recorder::run(|| -> RaiseResult<()> {
            let i = match color_index(index) {
                Ok(i) => i,
                Err(raise) => return on.throw(raise),
            };
            let mut pass = self.lock();
            let actions = (set.load_action, set.store_action);
            let resolve = ResolveSet {
                texture: set.resolve_texture,
                level: set.resolve_level,
                slice: set.resolve_slice,
                depth_plane: set.resolve_depth_plane,
                filter: None,
            };
            pass.colors[i].set(set.texture, (set.level, set.slice), actions, resolve);
            if let Some(c) = set.clear_color {
                pass.clear_colors[i] = c;
            }
            Ok(())
        });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    fn set_depth_attachment(&self, set: &DepthAttachmentSet<'_, RtPlatform>) {
        let recorded_call = crate::recorder::rec_call!("m2v.setDepthAttachment", self, set);
        crate::recorder::run(|| {
            let mut pass = self.lock();
            let actions = (set.load_action, set.store_action);
            let resolve = ResolveSet {
                texture: set.resolve_texture,
                level: set.resolve_level,
                slice: set.resolve_slice,
                depth_plane: set.resolve_depth_plane,
                filter: set.depth_resolve_filter,
            };
            pass.depth
                .set(set.texture, (set.level, set.slice), actions, resolve);
            pass.clear_depth = set.clear_depth.unwrap_or(pass.clear_depth);
        });
        crate::recorder::rec_done!(recorded_call, ());
    }

    fn set_stencil_attachment(&self, set: &StencilAttachmentSet<'_, RtPlatform>) {
        let recorded_call = crate::recorder::rec_call!("m2v.setStencilAttachment", self, set);
        crate::recorder::run(|| {
            let mut pass = self.lock();
            let actions = (set.load_action, set.store_action);
            let resolve = ResolveSet {
                texture: set.resolve_texture,
                level: set.resolve_level,
                slice: set.resolve_slice,
                depth_plane: set.resolve_depth_plane,
                filter: set.stencil_resolve_filter,
            };
            pass.stencil
                .set(set.texture, (set.level, set.slice), actions, resolve);
            pass.clear_stencil = set.clear_stencil.unwrap_or(pass.clear_stencil);
        });
        crate::recorder::rec_done!(recorded_call, ());
    }

    /// An index past the eight attachments raises, the caller's to catch.
    fn color_attachment_state(&self, index: u64) -> RaiseResult<AttachmentState> {
        let i = color_index(index)?;
        Ok(self.lock().colors[i].state())
    }

    fn depth_attachment_state(&self) -> AttachmentState {
        self.lock().depth.state()
    }

    fn stencil_attachment_state(&self) -> AttachmentState {
        self.lock().stencil.state()
    }
}

/// `MTLRasterizationRateMapDescriptor`, unbuilt ([`Unbuilt::RateMap`]): no layer exists to set
/// on it, and it reads as one with none.
pub struct RateMapFields;

impl<P: Platform> RateMapDescriptor<P> for RtObj<RateMapFields> {
    fn set_layer(
        &self,
        on: OnFault,
        layer: &P::RateMapLayerDescriptor,
        layer_index: u64,
    ) -> RaiseResult<()> {
        let _ = (on, layer, layer_index);
        refuse(Unbuilt::RateMap, "setLayer:atIndex:");
        Ok(())
    }
    fn layer_count(&self) -> u64 {
        refuse(Unbuilt::RateMap, "layerCount");
        0
    }
    fn layer(
        &self,
        on: OnFault,
        layer_index: u64,
    ) -> RaiseResult<Option<P::RateMapLayerDescriptor>> {
        let _ = (on, layer_index);
        refuse(Unbuilt::RateMap, "layers");
        Ok(None)
    }
}

/// What a render pass asked of rt, for the process's first line of each kind of rt's refusals
/// of it (`super::unbuilt::refusal_detail`, at `renderCommandEncoderWithDescriptor:`), as
/// structure alone: each attachment of a texture, the depth and stencil attachments' store
/// actions, and the render target's width and height ([`PassFields::detail`]).
pub struct PassDetail {
    /// The attachments, colours in index order, then depth, then stencil.
    attachments: Vec<PassAttachment>,
    /// The depth and stencil attachments' store actions, whether or not each names a texture.
    stores: (rt::StoreAction, rt::StoreAction),
    /// `renderTargetWidth` and `renderTargetHeight`.
    size: [u32; 2],
}

/// An attachment of a [`PassDetail`].
struct PassAttachment {
    /// Which attachment it is.
    point: Point,
    /// Its texture.
    texture: AttachedTexture,
    /// The level and slice rendered.
    at: [u32; 2],
    /// Its load and store actions.
    actions: (rt::LoadAction, rt::StoreAction),
    /// Its resolve texture, with the level and slice resolved into, where it names one.
    resolve: Option<(AttachedTexture, [u32; 2])>,
    /// A depth attachment's clear depth.
    clear_depth: Option<f64>,
}

/// A texture an attachment names: its pixel format, type, sample count, size, usage and storage
/// mode, and how it was made (`super::diag::Made::made_as`: `drawable` for a window
/// swapchain's), with a texture over a buffer's bytes per row, offset in the buffer and the
/// buffer's length, as its make gave them (`super::diag::Made::in_buffer`).
struct AttachedTexture {
    /// Its pixel format.
    format: rt::PixelFormat,
    /// Its texture type.
    texture_type: rt::TextureType,
    /// Its sample count.
    samples: u32,
    /// Its width, height and depth.
    size: [u32; 3],
    /// Its usage, Metal's bits.
    usage: u32,
    /// Its storage mode.
    storage: rt::StorageMode,
    /// How it was made.
    made_as: &'static str,
    /// Where it lies in the buffer it was made over, where it was made over one.
    in_buffer: Option<InBuffer>,
}

impl AttachedTexture {
    /// The texture `t`, of rt's descriptor of it.
    fn of(t: &RtTexture) -> AttachedTexture {
        let d = t.rt().descriptor();
        AttachedTexture {
            format: d.pixel_format,
            texture_type: d.texture_type,
            samples: d.sample_count,
            size: [d.width, d.height, d.depth],
            usage: d.usage.as_raw(),
            storage: d.storage_mode,
            made_as: t.made().made_as(),
            in_buffer: t.made().in_buffer(),
        }
    }
}

impl PassFields {
    /// The detail of rt's pass `pass`, which [`PassFields::rt_pass`] made of what is set, for the
    /// process's first line of each kind of rt's refusals of it ([`PassDetail`]).
    pub fn detail(&self, pass: &rt::RenderPassDescriptor) -> PassDetail {
        let set = self.lock();
        let colors = (pass.color_attachments.iter().zip(&set.colors).enumerate()).filter_map(
            |(i, (c, a))| {
                let actions = (c.load_action, c.store_action);
                let resolved_at = [c.resolve_level, c.resolve_slice];
                attached(Point::Color(i), a, [c.level, c.slice], actions, resolved_at)
            },
        );
        let d = &pass.depth_attachment;
        let depth = attached(
            Point::Depth,
            &set.depth,
            [d.level, d.slice],
            (d.load_action, d.store_action),
            [d.resolve_level, d.resolve_slice],
        )
        .map(|a| PassAttachment {
            clear_depth: Some(d.clear_depth),
            ..a
        });
        let s = &pass.stencil_attachment;
        let stencil = attached(
            Point::Stencil,
            &set.stencil,
            [s.level, s.slice],
            (s.load_action, s.store_action),
            [s.resolve_level, s.resolve_slice],
        );
        PassDetail {
            attachments: colors.chain(depth).chain(stencil).collect(),
            stores: (d.store_action, s.store_action),
            size: [pass.render_target_width, pass.render_target_height],
        }
    }
}

/// The attachment `point` of a pass, `set`'s texture and resolve texture at the level and slice
/// `at` and `resolved_at`, of the actions `actions`, with no clear depth, or `None` where it names
/// no texture.
fn attached(
    point: Point,
    set: &Attachment,
    at: [u32; 2],
    actions: (rt::LoadAction, rt::StoreAction),
    resolved_at: [u32; 2],
) -> Option<PassAttachment> {
    Some(PassAttachment {
        point,
        texture: AttachedTexture::of(set.texture.as_ref()?),
        at,
        actions,
        resolve: (set.resolve_texture.as_ref()).map(|t| (AttachedTexture::of(t), resolved_at)),
        clear_depth: None,
    })
}

impl std::fmt::Display for AttachedTexture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let [width, height, depth] = self.size;
        write!(
            f,
            "{:?} {:?} {width}x{height}x{depth}, {} samples, usage {:#x}, storage {:?}, {}",
            self.format, self.texture_type, self.samples, self.usage, self.storage, self.made_as
        )?;
        if let Some(in_buffer) = self.in_buffer {
            write!(f, " ({in_buffer})")?;
        }
        Ok(())
    }
}

impl std::fmt::Display for PassAttachment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.point {
            Point::Color(index) => write!(f, "colour {index}")?,
            Point::Depth => f.write_str("depth")?,
            Point::Stencil => f.write_str("stencil")?,
        }
        let ([level, slice], (load, store)) = (self.at, self.actions);
        write!(
            f,
            " {} at level {level} slice {slice}, load {load:?}, store {store:?}",
            self.texture
        )?;
        if let Some((texture, [level, slice])) = &self.resolve {
            write!(f, ", resolve into {texture} at level {level} slice {slice}")?;
        }
        if let Some(depth) = self.clear_depth {
            write!(f, ", clear depth {depth}")?;
        }
        Ok(())
    }
}

impl std::fmt::Display for PassDetail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ([width, height], (depth, stencil)) = (self.size, self.stores);
        write!(
            f,
            "a pass of {} attachments, render target {width}x{height}, depth store {depth:?}, \
             stencil store {stencil:?}",
            self.attachments.len()
        )?;
        for a in &self.attachments {
            write!(f, "; {a}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::raise::Origin;

    /// A refused pass's detail names each attachment's format, type, size, sample count, usage,
    /// storage mode, how its texture was made, level, slice, load and store actions, its resolve
    /// texture so too and a depth attachment's clear depth; the depth and stencil store actions;
    /// and the render target's size.
    #[test]
    fn a_refused_pass_s_detail_names_each_attachment() {
        use rt::{LoadAction, PixelFormat, StoreAction, TextureType};
        let texture = |format, texture_type, samples| AttachedTexture {
            format,
            texture_type,
            samples,
            size: [64, 32, 1],
            usage: 0x4,
            storage: rt::StorageMode::Private,
            made_as: "ordinary",
            in_buffer: None,
        };
        let colour = PassAttachment {
            point: Point::Color(0),
            texture: texture(PixelFormat::BGRA8Unorm, TextureType::Type2DMultisample, 4),
            at: [0, 0],
            actions: (LoadAction::Clear, StoreAction::MultisampleResolve),
            resolve: Some((
                AttachedTexture {
                    usage: 0x5,
                    made_as: "drawable",
                    ..texture(PixelFormat::BGRA8Unorm, TextureType::Type2D, 1)
                },
                [1, 0],
            )),
            clear_depth: None,
        };
        let depth = PassAttachment {
            point: Point::Depth,
            texture: texture(PixelFormat::Depth32Float, TextureType::Type2D, 1),
            at: [0, 2],
            actions: (LoadAction::Clear, StoreAction::DontCare),
            resolve: None,
            clear_depth: Some(1.5),
        };
        let what = PassDetail {
            attachments: vec![colour, depth],
            stores: (StoreAction::DontCare, StoreAction::DontCare),
            size: [0, 0],
        };
        assert_eq!(
            what.to_string(),
            "a pass of 2 attachments, render target 0x0, depth store DontCare, stencil store \
             DontCare; colour 0 BGRA8Unorm Type2DMultisample 64x32x1, 4 samples, usage 0x4, \
             storage Private, ordinary at level 0 slice 0, load Clear, store MultisampleResolve, \
             resolve into BGRA8Unorm Type2D 64x32x1, 1 samples, usage 0x5, storage Private, \
             drawable at level 1 slice 0; depth Depth32Float Type2D 64x32x1, 1 samples, usage \
             0x4, storage Private, ordinary at level 0 slice 2, load Clear, store DontCare, clear \
             depth 1.5"
        );
        let none = PassDetail {
            attachments: Vec::new(),
            stores: (StoreAction::Store, StoreAction::MultisampleResolve),
            size: [16, 16],
        };
        assert_eq!(
            none.to_string(),
            "a pass of 0 attachments, render target 16x16, depth store Store, stencil store \
             MultisampleResolve"
        );
    }

    /// A refused pass's detail names where a texture over a buffer lies in it, as its make gave
    /// it: its bytes per row, its offset in the buffer and the buffer's length, after how it was
    /// made, a resolve texture's so too.
    #[test]
    fn a_refused_pass_s_detail_names_where_an_over_buffer_texture_lies() {
        use rt::{LoadAction, PixelFormat, StoreAction, TextureType};
        let colour = PassAttachment {
            point: Point::Color(0),
            texture: AttachedTexture {
                format: PixelFormat::BGRA8Unorm,
                texture_type: TextureType::Type2DMultisample,
                samples: 4,
                size: [640, 480, 1],
                usage: 0x4,
                storage: rt::StorageMode::Private,
                made_as: "ordinary",
                in_buffer: None,
            },
            at: [0, 0],
            actions: (LoadAction::Clear, StoreAction::MultisampleResolve),
            resolve: Some((
                AttachedTexture {
                    format: PixelFormat::BGRA8Unorm,
                    texture_type: TextureType::Type2D,
                    samples: 1,
                    size: [640, 480, 1],
                    usage: 0x5,
                    storage: rt::StorageMode::Managed,
                    made_as: "over-buffer",
                    in_buffer: Some(InBuffer {
                        bytes_per_row: 2560,
                        offset: 256,
                        buffer_length: 1_229_056,
                    }),
                },
                [0, 0],
            )),
            clear_depth: None,
        };
        let what = PassDetail {
            attachments: vec![colour],
            stores: (StoreAction::DontCare, StoreAction::DontCare),
            size: [640, 480],
        };
        assert_eq!(
            what.to_string(),
            "a pass of 1 attachments, render target 640x480, depth store DontCare, stencil store \
             DontCare; colour 0 BGRA8Unorm Type2DMultisample 640x480x1, 4 samples, usage 0x4, \
             storage Private, ordinary at level 0 slice 0, load Clear, store \
             MultisampleResolve, resolve into BGRA8Unorm Type2D 640x480x1, 1 samples, usage 0x5, \
             storage Managed, over-buffer (bytes per row 2560, offset 256, buffer length 1229056) \
             at level 0 slice 0"
        );
    }

    /// On the GPU the test holds: a texture made over a buffer
    /// (`newTextureWithDescriptor:offset:bytesPerRow:`) names, as an attachment of a refused pass,
    /// its bytes per row, its offset in the buffer and the buffer's length as its make gave them;
    /// a texture of its own names none.
    #[test]
    fn an_over_buffer_texture_names_where_it_lies_on_vulkan() {
        use crate::core::platform::device::GpuDevice as _;
        use crate::core::platform::resource::Buffer as _;
        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let on = OnFault::Return;
        let buffer = device.new_buffer_of_length(on, 12288, 0);
        let buffer = buffer.expect("never a raise").expect("a buffer");
        let descriptor = platform.texture_descriptor(&TextureDesc {
            pixel_format: 80,
            width: 64,
            height: 4,
            usage: 0x1,
            ..TextureDesc::default()
        });
        let texture = buffer.new_texture(on, &descriptor, 4096, 256);
        let texture = texture
            .expect("never a raise")
            .expect("a texture over the buffer");
        let named = AttachedTexture::of(&texture).to_string();
        assert!(
            named.ends_with(", over-buffer (bytes per row 256, offset 4096, buffer length 12288)"),
            "{named}"
        );
        let own = device.new_texture(on, &descriptor);
        let own = own.expect("never a raise").expect("a texture");
        let named = AttachedTexture::of(&own).to_string();
        assert!(named.ends_with(", ordinary"), "{named}");
    }

    /// A mipmapped 2D descriptor has the levels of its larger side down to 1×1; one not mipmapped
    /// has one, as Metal's class method makes them.
    #[test]
    fn a_2d_descriptor_has_the_levels_of_its_larger_side() {
        assert_eq!(texture_2d_desc(70, 1024, 512, true).mipmap_level_count, 11);
        assert_eq!(texture_2d_desc(70, 1000, 3, true).mipmap_level_count, 10);
        assert_eq!(texture_2d_desc(70, 1, 1, true).mipmap_level_count, 1);
        assert_eq!(texture_2d_desc(70, 1024, 512, false).mipmap_level_count, 1);
        let d = texture_2d_desc(80, 16, 8, false);
        assert_eq!((d.pixel_format, d.width, d.height), (80, 16, 8));
        assert_eq!(d.texture_type, TextureDesc::default().texture_type);
    }

    /// rt's descriptor takes each value as rt's type of it, the storage mode from the resource
    /// options, and names the first value it has no type of.
    #[test]
    fn rt_s_descriptor_takes_each_raw_value() {
        let d = rt_texture_descriptor(&TextureDesc::default()).unwrap();
        assert_eq!(d.texture_type, rt::TextureType::Type2D);
        assert_eq!(d.pixel_format, rt::PixelFormat::RGBA8Unorm);
        assert_eq!((d.width, d.height, d.depth), (1, 1, 1));
        assert_eq!(d.usage, rt::TextureUsage::SHADER_READ);
        assert_eq!(d.storage_mode, rt::StorageMode::Shared);
        let private = TextureDesc {
            pixel_format: 80,
            width: 64,
            height: 32,
            resource_options: Some(0x20),
            ..TextureDesc::default()
        };
        let d = rt_texture_descriptor(&private).unwrap();
        assert_eq!(d.pixel_format, rt::PixelFormat::BGRA8Unorm);
        assert_eq!(d.storage_mode, rt::StorageMode::Private);
        let wide = TextureDesc {
            width: 1 << 32,
            ..TextureDesc::default()
        };
        assert_eq!(
            rt_texture_descriptor(&wide),
            Err(format!("width {}", 1u64 << 32))
        );
        let typed = TextureDesc {
            texture_type: 99,
            ..TextureDesc::default()
        };
        assert_eq!(
            rt_texture_descriptor(&typed),
            Err("texture type 99".to_owned())
        );
        let stored = TextureDesc {
            resource_options: Some(0x70),
            ..TextureDesc::default()
        };
        assert_eq!(
            rt_texture_descriptor(&stored),
            Err("storage mode 7".to_owned())
        );
    }

    fn counted(code: &str) -> u64 {
        (crate::answer::degradation_counts().into_iter())
            .find(|(c, _)| c == code)
            .map_or(0, |(_, n)| n)
    }

    /// A pass descriptor answers Metal's defaults and the actions set, raw; rt's descriptor takes
    /// each value kept, the resolves' among them, and names an action or resolve filter Metal has
    /// none of or a level past 32 bits; a colour attachment past the eighth raises at the
    /// subscript; a property rt has no field of is counted where it is set to other than Metal's
    /// default.
    /// rt makes a pass with no attachment at the descriptor's `defaultRasterSampleCount`, and
    /// refuses one of 0, Metal's default: the guest's count reaches it as set.
    #[test]
    fn a_pass_hands_rt_its_default_raster_sample_count() {
        let pass = RtObj::new(PassFields::default());
        let d = pass.rt_pass().expect("rt's descriptor");
        assert_eq!(d.default_raster_sample_count, 0);
        for samples in [1, 4] {
            pass.set_pass(&PassSet {
                default_raster_sample_count: Some(samples),
                ..PassSet::default()
            });
            let d = pass.rt_pass().expect("rt's descriptor");
            assert_eq!(u64::from(d.default_raster_sample_count), samples);
        }
        // A set that sends no count keeps the last one.
        pass.set_pass(&PassSet {
            render_target_width: Some(16),
            ..PassSet::default()
        });
        let d = pass.rt_pass().expect("rt's descriptor");
        assert_eq!(d.default_raster_sample_count, 4);
    }

    #[test]
    fn a_pass_keeps_what_rt_renders() {
        let pass = RtObj::new(PassFields::default());
        let state = |load_action, store_action| AttachmentState {
            load_action,
            store_action,
        };
        assert_eq!(pass.color_attachment_state(7), Ok(state(0, 1)));
        assert_eq!(pass.depth_attachment_state(), state(2, 0));
        assert_eq!(pass.stencil_attachment_state(), state(2, 0));
        let d = pass.rt_pass().expect("rt's descriptor");
        let c = &d.color_attachments[0];
        assert_eq!(c.load_action, rt::LoadAction::DontCare);
        assert_eq!(c.store_action, rt::StoreAction::Store);
        assert_eq!(c.clear_color, rt::ClearColor::new(0.0, 0.0, 0.0, 1.0));
        assert_eq!(d.depth_attachment.load_action, rt::LoadAction::Clear);
        assert_eq!(d.depth_attachment.clear_depth, 1.0);
        assert_eq!(d.stencil_attachment.store_action, rt::StoreAction::DontCare);
        assert_eq!((d.render_target_width, d.render_target_height), (0, 0));
        assert!(c.resolve_texture.is_none());
        let depth_filter = d.depth_attachment.depth_resolve_filter;
        assert_eq!(depth_filter, rt::MultisampleDepthResolveFilter::Sample0);
        let stencil_filter = d.stencil_attachment.stencil_resolve_filter;
        assert_eq!(stencil_filter, rt::MultisampleStencilResolveFilter::Sample0);

        let on = OnFault::Return;
        let red = ColorAttachmentSet {
            slice: Some(2),
            resolve_level: Some(1),
            resolve_depth_plane: Some(5),
            load_action: Some(2),
            store_action: Some(0),
            clear_color: Some(ClearColor {
                red: 1.0,
                alpha: 1.0,
                ..ClearColor::default()
            }),
            ..ColorAttachmentSet::default()
        };
        assert_eq!(pass.set_color_attachment(on, 3, &red), Ok(()));
        pass.set_depth_attachment(&DepthAttachmentSet {
            load_action: Some(1),
            store_action: Some(4),
            clear_depth: Some(0.5),
            resolve_slice: Some(3),
            depth_resolve_filter: Some(2),
            ..DepthAttachmentSet::default()
        });
        pass.set_stencil_attachment(&StencilAttachmentSet {
            level: Some(1),
            clear_stencil: Some(0x1ff),
            resolve_level: Some(4),
            stencil_resolve_filter: Some(1),
            ..StencilAttachmentSet::default()
        });
        pass.set_pass(&PassSet {
            render_target_width: Some(64),
            render_target_height: Some(1 << 40),
            ..PassSet::default()
        });
        assert_eq!(pass.color_attachment_state(3), Ok(state(2, 0)));
        assert_eq!(pass.depth_attachment_state(), state(1, 4));
        let d = pass.rt_pass().expect("rt's descriptor");
        let c = &d.color_attachments[3];
        assert_eq!((c.slice, c.level), (2, 0));
        assert_eq!(c.load_action, rt::LoadAction::Clear);
        assert_eq!(c.store_action, rt::StoreAction::DontCare);
        assert_eq!(c.clear_color, rt::ClearColor::new(1.0, 0.0, 0.0, 1.0));
        let resolve = (c.resolve_level, c.resolve_slice, c.resolve_depth_plane);
        assert_eq!(resolve, (1, 0, 5));
        assert_eq!(d.depth_attachment.load_action, rt::LoadAction::Load);
        assert_eq!(d.depth_attachment.store_action, rt::StoreAction::Unknown);
        assert_eq!(d.depth_attachment.clear_depth, 0.5);
        let z = &d.depth_attachment;
        assert_eq!((z.resolve_level, z.resolve_slice), (0, 3));
        assert_eq!(
            z.depth_resolve_filter,
            rt::MultisampleDepthResolveFilter::Max
        );
        let s = &d.stencil_attachment;
        assert_eq!((s.level, s.clear_stencil), (1, 0x1ff));
        assert_eq!((s.resolve_level, s.resolve_slice), (4, 0));
        let stencil_filter = rt::MultisampleStencilResolveFilter::DepthResolvedSample;
        assert_eq!(s.stencil_resolve_filter, stencil_filter);
        assert_eq!(
            (d.render_target_width, d.render_target_height),
            (64, u32::MAX)
        );

        // Metal's refusals, named at the encoder's creation.
        let unknown = ColorAttachmentSet {
            load_action: Some(7),
            ..ColorAttachmentSet::default()
        };
        assert_eq!(pass.set_color_attachment(on, 3, &unknown), Ok(()));
        assert_eq!(pass.color_attachment_state(3), Ok(state(7, 0)));
        let named = pass.rt_pass().err();
        assert_eq!(
            named.as_deref(),
            Some("colour attachment 3's load action 7")
        );
        let known = ColorAttachmentSet {
            load_action: Some(0),
            ..ColorAttachmentSet::default()
        };
        assert_eq!(pass.set_color_attachment(on, 3, &known), Ok(()));
        for (filter, named) in [
            (3, Some("the depth attachment's depth resolve filter 3")),
            (0, None),
        ] {
            pass.set_depth_attachment(&DepthAttachmentSet {
                depth_resolve_filter: Some(filter),
                ..DepthAttachmentSet::default()
            });
            assert_eq!(pass.rt_pass().err().as_deref(), named);
        }
        for (filter, named) in [
            (2, Some("the stencil attachment's stencil resolve filter 2")),
            (0, None),
        ] {
            pass.set_stencil_attachment(&StencilAttachmentSet {
                stencil_resolve_filter: Some(filter),
                ..StencilAttachmentSet::default()
            });
            assert_eq!(pass.rt_pass().err().as_deref(), named);
        }
        pass.set_depth_attachment(&DepthAttachmentSet {
            level: Some(1 << 32),
            ..DepthAttachmentSet::default()
        });
        let named = pass.rt_pass().err();
        assert_eq!(
            named.as_deref(),
            Some("the depth attachment's level 4294967296")
        );

        // The ninth colour attachment raises at Metal's subscript, setting nothing.
        let raised = pass.set_color_attachment(on, 8, &red).err();
        let origin = raised.map(|r| r.origin());
        assert_eq!(origin, Some(Origin::Platform(SUBSCRIPT)));
        let read = pass.color_attachment_state(u64::MAX).err();
        assert_eq!(read.map(|r| r.origin()), Some(Origin::Platform(SUBSCRIPT)));

        // What rt has no field of: counted where it is other than Metal's default. The render
        // target array length is rt's, kept as set and counted nowhere.
        let code = Unbuilt::RenderPass.code();
        let before = counted(code);
        pass.set_pass(&PassSet {
            render_target_array_length: Some(1),
            tile_width: Some(0),
            visibility_result_buffer: Some(None),
            ..PassSet::default()
        });
        assert_eq!(pass.set_sample_positions(&[]), Ok(()));
        assert_eq!(counted(code), before);
        let layers = |p: &RtObj<PassFields>| p.rt_pass().map(|d| d.render_target_array_length);
        let fresh = RtObj::new(PassFields::default());
        assert_eq!(layers(&fresh), Ok(0));
        fresh.set_pass(&PassSet {
            render_target_array_length: Some(1 << 32),
            ..PassSet::default()
        });
        assert_eq!(layers(&fresh), Ok(u32::MAX));
        pass.set_pass(&PassSet {
            render_target_array_length: Some(2),
            imageblock_sample_length: Some(32),
            threadgroup_memory_length: Some(1024),
            tile_width: Some(16),
            tile_height: Some(16),
            ..PassSet::default()
        });
        let positions = [
            MtlSamplePosition { x: 0.25, y: 0.75 },
            MtlSamplePosition { x: 0.75, y: 0.25 },
        ];
        assert_eq!(pass.set_sample_positions(&positions), Ok(()));
        assert_eq!(counted(code), before + 5);
    }

    /// `(x, y)` as Metal's `MTLSamplePositionMake` makes it.
    fn at(x: f32, y: f32) -> MtlSamplePosition {
        MtlSamplePosition { x, y }
    }

    /// Metal's default positions of 1, 2 and 4 samples (measured on the M3 Max, macOS 27,
    /// `guest/bundle/tools/samplepos_probe.m`) are what rt's pass renders at, so a set of them
    /// counts nothing; a set that differs from its count's defaults by one position counts, as
    /// does a set of their positions in another order, and a set of 8, whose defaults are not
    /// measured.
    #[test]
    fn sample_positions_count_only_where_they_differ_from_metal_s_defaults() {
        let pass = RtObj::new(PassFields::default());
        let code = Unbuilt::RenderPass.code();
        let before = counted(code);
        let one = [at(0.5, 0.5)];
        let two = [at(0.75, 0.75), at(0.25, 0.25)];
        let four = [
            at(0.375, 0.125),
            at(0.875, 0.375),
            at(0.125, 0.625),
            at(0.625, 0.875),
        ];
        for defaults in [&one[..], &two, &four] {
            assert_eq!(pass.set_sample_positions(defaults), Ok(()));
        }
        assert_eq!(pass.set_sample_positions(&[]), Ok(()));
        assert_eq!(counted(code), before);

        let mut moved = four;
        moved[3] = at(0.625, 0.75);
        assert_eq!(pass.set_sample_positions(&moved), Ok(()));
        assert_eq!(counted(code), before + 1);
        let swapped = [two[1], two[0]];
        assert_eq!(pass.set_sample_positions(&swapped), Ok(()));
        assert_eq!(counted(code), before + 2);
        let eight = [at(0.5, 0.5); 8];
        assert_eq!(pass.set_sample_positions(&eight), Ok(()));
        assert_eq!(counted(code), before + 3);
    }

    /// A set's line names its count and each position as the `f32` it is, so sets one ulp apart
    /// print apart.
    #[test]
    fn sample_positions_print_their_count_and_each_position() {
        let set = [at(0.125, 0.625), at(0.875, 0.375)];
        assert_eq!(
            SamplePositions(&set).to_string(),
            "sample positions, count 2: (0.125, 0.625) (0.875, 0.375)"
        );
        let half = SamplePositions(&[at(0.5, 0.5)]).to_string();
        assert_eq!(half, "sample positions, count 1: (0.5, 0.5)");
        let past_half = [at(f32::from_bits(0.5f32.to_bits() + 1), 0.5)];
        let past_half = SamplePositions(&past_half).to_string();
        assert_eq!(past_half, "sample positions, count 1: (0.50000006, 0.5)");
    }

    /// The process's first count of a render pass's unbuilt piece, from a set of sample positions
    /// that differs from Metal's defaults, writes the piece's line naming the set's count and each
    /// position; a set of Metal's defaults before it writes none, and a set after it none (in a
    /// child, whose stderr is its own).
    #[test]
    fn the_first_differing_sample_positions_name_their_count_and_positions() {
        use crate::core::testkit::death::{Expect, Options, Outcome, run};
        run(
            Expect::Any,
            Options::default(),
            module_path!(),
            "the_first_differing_sample_positions_name_their_count_and_positions",
            |_| {
                let pass = RtObj::new(PassFields::default());
                let defaults = [at(0.75, 0.75), at(0.25, 0.25)];
                let left = [
                    at(0.125, 0.125),
                    at(0.375, 0.375),
                    at(0.125, 0.625),
                    at(0.375, 0.875),
                ];
                for set in [&defaults[..], &left, &[at(0.5, 0.25); 8]] {
                    assert_eq!(pass.set_sample_positions(set), Ok(()));
                }
            },
            |r| {
                assert_eq!(r.outcome(), &Outcome::Returned, "{}", r.stderr());
                let out = r.stderr();
                let lines: Vec<&str> = (out.lines())
                    .filter(|l| l.contains("rt-unbuilt-render-pass"))
                    .collect();
                // A line ends with what was written: the child's first line starts with
                // libtest's own "test NAME ... ".
                let said = "reims-vgpu: setSamplePositions:count:: a render pass's memory \
                            lengths, tiles or samples is not built on metal2vulkan-runtime; \
                            refused (rt-unbuilt-render-pass), the first of its kind: sample \
                            positions, count 4: (0.125, 0.125) (0.375, 0.375) (0.125, 0.625) \
                            (0.375, 0.875)";
                assert_eq!(lines.len(), 1, "{out}");
                assert!(lines.iter().all(|l| l.ends_with(said)), "{out}");
            },
        );
    }
}
