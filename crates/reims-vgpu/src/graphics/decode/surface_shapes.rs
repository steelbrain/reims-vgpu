//! `TEXTURE_FROM_IOSURFACE` over a device cache in the per-shape form (`surface_cache::shapes`):
//! the guest's texture is a view, at the guest's resource index and in the guest's format, of
//! its shape's image, which the cache keeps for every texture of that shape over the surface.
//!
//! The usage the guest declares picks the shape's kind (`shapes`, "The shape"): a texture of
//! Metal's Unknown usage (0), `ShaderWrite` or `ShaderAtomic` takes storage, and views the storage
//! shape of its extent, made of the Unknown usage where the cache holds none; any other views the
//! storage shape of its extent where the cache holds one, else its plain shape, made of
//! `ShaderRead | RenderTarget` alone. A texture over a plain shape in a format neither the
//! shape's nor its sRGB sibling, a view of the plain image that rt refuses as Metal refuses one
//! of a texture with no `PixelFormatView` usage, takes the storage shape of its extent instead,
//! counted ([`shapes::FORMAT_STORAGE`], logged once); one over a plain shape that declares
//! `PixelFormatView` is counted too.
//!
//! A texture takes a shape where it is a 2D texture of one level, one slice and one sample in a
//! colour format of texels ([`shapes::texel_bytes`]) lying wholly inside the cache's buffer; any
//! other, and one whose image or view the platform does not make, lies over the buffer as in the
//! default form, counted (`ShapeCounts::unshaped`).
//!
//! In the default form a texture over a cache lies over its buffer, and holds a reference to the
//! shape the per-shape form would give it in the cache's shadow, which has no image: the counts
//! of the bytes the per-shape form's bound would hold read it (`CacheCounts::shape_bound_most`),
//! and nothing the guest sees changes.
//!
//! Every surface takes a device cache in the per-shape form: no texture lies in place over a
//! surface's pages. A `SURFACE_PAGES` that asks whether its block needs one is answered that it
//! does, and one that names its block uncached, or cached where the platform made no cache, is
//! refused ([`UNCACHED_COPIED`]), so the kext gives the surface a copy block. A block named
//! cached whose pages the host makes no view of, or no buffer over (no mapping of them, or none
//! the platform imports), takes its cache all the same, its pages copied by the host's CPU
//! through its mapping of the guest's RAM ([`surface_cache::Caches::make_copied`]), counted
//! ([`PAGES_COPIED`], logged once): a copy block's textures would be no shape's, and the platform
//! makes no texture over a surface's buffer that is not one. Runs that are not the surface's
//! bytes in whole guest pages, or that the host maps no part of, are refused still.

use std::sync::atomic::{AtomicBool, Ordering};

use reims_vgpu_graphics::wire::{
    RGFX_STATUS_OK, RGFX_STATUS_REFUSED, RgfxPageRun, RgfxSurfacePages, RgfxTextureFromIOSurface,
};
use reims_vgpu_vulkan::core::platform::Platform;
use reims_vgpu_vulkan::core::platform::resource::{Buffer as _, Texture as _};
use reims_vgpu_vulkan::core::platform::types::{NsRange, PixelFormatInfo};
use reims_vgpu_vulkan::core::platform::values::TextureDesc;
use reims_vgpu_vulkan::core::raise::RaiseResult;
use reims_vgpu_vulkan::gpu::no_texture::views_without_usage;

use super::surface_cache::shapes::{self, FORMAT_STORAGE, ShapeCounts, ShapeKey, ShapeUse};
use super::surface_cache::{self, CacheLayout, STORAGE_PRIVATE, ShapeMaker};
use super::{GraphicsDecoder, ObjHandle, TEXTURE_USAGE_WRITES, texture_desc};

/// `MTLTextureType2D`.
const TEXTURE_TYPE_2D: u64 = 2;

/// `MTLTextureUsageShaderAtomic`.
const TEXTURE_USAGE_ATOMIC: u64 = 0x20;

/// `MTLTextureUsagePixelFormatView`.
const TEXTURE_USAGE_PIXEL_FORMAT_VIEW: u64 = 0x10;

/// A plain shape's image's usage: `MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget`
/// (`shapes`, "The shape").
const PLAIN_USAGE: u64 = 0x1 | 0x4;

/// Whether a texture over a plain shape took the storage shape for its format was logged
/// ([`FORMAT_STORAGE`]): its line is logged the first time alone, the rest counted
/// (docs/conventions.md).
static FORMAT_STORAGE_LOGGED: AtomicBool = AtomicBool::new(false);

/// The degradation code of a block refused as uncached in the per-shape form, which the kext
/// gives a copy block instead (module doc).
pub(super) const UNCACHED_COPIED: &str = "rt-surface-uncached-copied";

/// Whether a block was refused as uncached ([`UNCACHED_COPIED`]): its line is logged the first
/// time alone, the rest counted (docs/conventions.md).
static UNCACHED_LOGGED: AtomicBool = AtomicBool::new(false);

/// The degradation code of a block named cached in the per-shape form whose pages the host made
/// no view of, or no buffer over, which takes its cache with its pages copied by the host's CPU
/// (module doc).
pub(super) const PAGES_COPIED: &str = "rt-surface-pages-copied";

/// Whether a block took its cache copied ([`PAGES_COPIED`]): its line is logged the first time
/// alone, the rest counted (docs/conventions.md).
static COPIED_LOGGED: AtomicBool = AtomicBool::new(false);

impl<P: Platform> GraphicsDecoder<P> {
    /// Takes the per-shape form from now on (module doc), each shape's image made by `maker`,
    /// placed where `places` and its first fill is made with its make
    /// (`surface_cache::shapes`, "Placed images"), the shapes' clock the platform's, for their
    /// age rule (`surface_cache::shapes`, "The bound"): set once, at the device's make, before
    /// any surface is named.
    pub(crate) fn set_surface_shapes(&self, maker: ShapeMaker<P>, places: bool) {
        use reims_vgpu_vulkan::core::platform::os::clock::Clock as _;
        let platform = std::sync::Arc::clone(&self.platform);
        let now = std::sync::Arc::new(move || platform.clock().now_mono());
        self.surface_caches.set_shapes(maker, places, now);
    }

    /// Refuses the block at `block` as uncached in the per-shape form (module doc): any device
    /// cache it had goes, as at a release, and so does any view of its pages, the refusal counted
    /// ([`UNCACHED_COPIED`]). [`RGFX_STATUS_REFUSED`].
    pub(super) fn refuse_uncached(&self, device: &P::Device, block: u64) -> u32 {
        self.uncache_surface(device, block);
        let idle = self.platform.vm().idle_view_mappings();
        let old = self.shared.surface_views.lock().released(block, idle);
        drop(old);
        self.surface_caches
            .count_shapes(|c| c.uncached_refused += 1);
        reims_vgpu_vulkan::answer::count_kind(UNCACHED_COPIED);
        if !UNCACHED_LOGGED.swap(true, Ordering::Relaxed) {
            log::warn!(
                target: "reims_vgpu::graphics", kind:% = UNCACHED_COPIED;
                "the IOSurface block at {block:#x}: named with no device cache, which every \
                 surface takes in the per-shape form; refused, the surface takes a copy block \
                 ({UNCACHED_COPIED}; the rest are counted)"
            );
        }
        RGFX_STATUS_REFUSED
    }

    /// The block `pages` names cached in the per-shape form, whose view of its `runs` the host
    /// refused (`why`), given its device cache with its pages copied by the host's CPU (module
    /// doc), laid out as the kext's ask gave its surface, else as `carried` (`decode`'s
    /// `cache_surface`): [`RGFX_STATUS_OK`], counted ([`PAGES_COPIED`]) and logged the first time.
    /// `None` where the host's mapping of the guest's RAM does not hold the pages, or no cache was
    /// made, for the caller's refusal.
    pub(super) fn cache_copied(
        &self,
        device: &P::Device,
        pages: &RgfxSurfacePages,
        runs: &[RgfxPageRun],
        carried: CacheLayout,
        why: &str,
    ) -> Option<u32> {
        let block = pages.block_offset;
        let (copied, host_pages, laid_out) = {
            let mut views = self.shared.surface_views.lock();
            let copied = views.host_pages_in_order(runs);
            (copied, views.host_runs(runs), views.take_layout(block))
        };
        let copied = copied?;
        let layout = laid_out.map_or(carried, CacheLayout::Asked);
        // The CPU's generation the kext wrote into the block's header as it cached the block.
        let generation = surface_cache::cpu_generation(&self.shared.region, block).unwrap_or(0);
        let made = self.surface_caches.make_copied(
            device,
            block,
            pages.length,
            pages.block_size,
            &copied,
            host_pages,
            generation,
            layout,
        );
        match made {
            Ok(true) => {}
            Ok(false) => return None,
            Err(raise) => {
                log::warn!(
                    target: "reims_vgpu::graphics", kind:% = raise;
                    "the IOSurface block at {block:#x}: the platform raised at its device cache \
                     with its pages copied ({raise:?})"
                );
                return None;
            }
        }
        self.surface_caches.count_shapes(|c| c.pages_copied += 1);
        reims_vgpu_vulkan::answer::count_kind(PAGES_COPIED);
        if !COPIED_LOGGED.swap(true, Ordering::Relaxed) {
            log::warn!(
                target: "reims_vgpu::graphics", kind:% = PAGES_COPIED;
                "the IOSurface block of {:#x} bytes at {block:#x}: no view of its {:#x} bytes in \
                 {} runs ({why}); its device cache takes them copied, the host's CPU moving \
                 their bytes ({PAGES_COPIED}; the rest are counted)",
                pages.block_size, pages.length, runs.len()
            );
        }
        Some(RGFX_STATUS_OK)
    }

    /// The texture `m` asks, of the format `info` describes, as a view of its shape's image over
    /// the cache of the block at `block` (module doc), with the shape it holds a reference to.
    /// `None` where the caches take the default form, or the texture takes no shape here, which
    /// is counted where the block has a cache in the per-shape form.
    pub(super) fn shaped_surface_texture(
        &self,
        device: &P::Device,
        m: &RgfxTextureFromIOSurface,
        block: Option<u64>,
        info: &PixelFormatInfo,
    ) -> RaiseResult<Option<(P::Texture, ShapeUse)>> {
        let caches = &self.surface_caches;
        let Some(block) = block.filter(|&block| caches.shaped() && caches.has(block)) else {
            return Ok(None);
        };
        let unshaped = |count: fn(&mut ShapeCounts)| {
            caches.count_shapes(|c| {
                count(c);
                c.unshaped += 1;
            });
            Ok(None)
        };
        let Some(key) = self.surface_shape_key(m, info) else {
            return unshaped(|_| {});
        };
        let d = &m.desc;
        let pixel_format = u64::from(d.pixel_format);
        // The image: the guest's format and size, Private, at no index of the guest's; a storage
        // shape's of Metal's Unknown usage, which takes every use and a view in any format of the
        // texel's size, a plain shape's `ShaderRead | RenderTarget` alone (`shapes`, "The
        // shape").
        let image = TextureDesc {
            texture_type: TEXTURE_TYPE_2D,
            pixel_format,
            width: key.width,
            height: key.height,
            usage: if key.storage { 0 } else { PLAIN_USAGE },
            resource_options: Some(STORAGE_PRIVATE),
            ..TextureDesc::default()
        };
        let Some(mut held) = caches.shape(&self.platform, device, block, key, &image) else {
            return unshaped(|_| {});
        };
        if held.1.usage() == PLAIN_USAGE && !self.views_plain(m, held.1.pixel_format()) {
            // A plain image is viewed in its own format and its sRGB sibling alone: the texture
            // takes the storage shape of its extent, made of the Unknown usage where the cache
            // holds none, which rule 1 keeps current with the plain one (`shapes`, "The shape").
            drop(held);
            let key = key.as_storage();
            let image = TextureDesc { usage: 0, ..image };
            let Some(storage) = caches.shape(&self.platform, device, block, key, &image) else {
                return unshaped(|_| {});
            };
            held = storage;
        }
        let (reference, image) = held;
        let swizzle = texture_desc(d, m.resource_id).swizzle;
        let one_of = NsRange::new(0, 1);
        let view = match swizzle {
            Some(swizzle) => image.new_swizzled_texture_subview_at_index(
                pixel_format,
                TEXTURE_TYPE_2D,
                one_of,
                one_of,
                swizzle,
                m.resource_id,
            )?,
            None => image.new_texture_view_at_index(pixel_format, m.resource_id)?,
        };
        let Some(view) = view else {
            return unshaped(|c| c.views_refused += 1);
        };
        // The rt platform's boot diagnostics name the usage the guest declared of the view.
        reims_vgpu_vulkan::rt_platform::diag::declared(&view, u64::from(d.usage));
        let shape = ShapeUse {
            block,
            writes: takes_storage(m),
            video_plane: caches.video_plane(block, &reference),
            reference,
            diag: reims_vgpu_vulkan::rt_platform::diag::token(&view),
        };
        Ok(Some((view, shape)))
    }

    /// Whether the plain shape of `shape_format` the texture `m` asks lies over takes it as a
    /// view: in `shape_format` or its sRGB sibling, as Metal views a texture with no
    /// `PixelFormatView` usage. One that declares `PixelFormatView` is counted
    /// ([`ShapeCounts::plain_format_views`]), and one in any other format, which takes the storage
    /// shape of its extent instead, is counted too ([`FORMAT_STORAGE`], logged once).
    fn views_plain(&self, m: &RgfxTextureFromIOSurface, shape_format: u64) -> bool {
        let caches = &self.surface_caches;
        if u64::from(m.desc.usage) & TEXTURE_USAGE_PIXEL_FORMAT_VIEW != 0 {
            caches.count_shapes(|c| c.plain_format_views += 1);
        }
        let pixel_format = u64::from(m.desc.pixel_format);
        if views_without_usage(shape_format, pixel_format) {
            return true;
        }
        caches.count_shapes(|c| c.format_storage += 1);
        reims_vgpu_vulkan::answer::count_kind(FORMAT_STORAGE);
        reims_vgpu_vulkan::rt_platform::diag::plain_view_refused(shape_format, pixel_format);
        if !FORMAT_STORAGE_LOGGED.swap(true, Ordering::Relaxed) {
            log::info!(
                target: "reims_vgpu::graphics", kind:% = FORMAT_STORAGE;
                "the texture {:#x} over plane {} of the IOSurface {}: in format {pixel_format}, \
                 neither its plain shape's format {shape_format} nor its sRGB sibling, it takes \
                 the storage shape of its extent ({FORMAT_STORAGE}; the rest are counted)",
                m.obj_id, m.plane, m.iosurface_id
            );
        }
        false
    }

    /// The shape the texture `m` asks would take over its surface, of the format `info`
    /// describes (module doc): a 2D texture of one level, one slice and one sample in a colour
    /// format of texels, lying wholly inside its buffer at its place and pitch. `None` for any
    /// other.
    fn surface_shape_key(
        &self,
        m: &RgfxTextureFromIOSurface,
        info: &PixelFormatInfo,
    ) -> Option<ShapeKey> {
        let d = &m.desc;
        let one = |n: u32| n <= 1;
        let plain = u64::from(d.texture_type) == TEXTURE_TYPE_2D
            && one(d.mipmap_level_count)
            && one(d.array_length)
            && one(d.sample_count)
            && one(d.depth);
        let texel_bytes = shapes::texel_bytes(u64::from(d.pixel_format), info).filter(|_| plain)?;
        let key = ShapeKey {
            plane: u64::from(m.plane),
            offset: m.offset,
            bytes_per_row: m.bytes_per_row,
            width: u64::from(d.width),
            height: u64::from(d.height),
            texel_bytes,
            storage: takes_storage(m),
        };
        let length = match self.named(m.buffer_obj_id) {
            Some(ObjHandle::Buffer(buffer)) => buffer.length(),
            _ => 0,
        };
        let row = key.width.saturating_mul(texel_bytes);
        (key.span().end <= length && key.bytes_per_row >= row).then_some(key)
    }

    /// In the default form, the shape the per-shape form would give the texture `m`, made over the
    /// buffer of the block at `block`'s cache, of the format `info` describes: kept in the
    /// cache's shadow (`SurfaceCaches::shadow_shape`) for the counts of the bytes the per-shape
    /// form's bound would hold, with no image and nothing the guest sees. `None` in the per-shape
    /// form, where the block has no cache or one for the host's video, and where the texture
    /// would take no shape.
    pub(super) fn shadow_surface_shape(
        &self,
        m: &RgfxTextureFromIOSurface,
        block: Option<u64>,
        info: &PixelFormatInfo,
    ) -> Option<ShapeUse> {
        let block = block?;
        let key = self.surface_shape_key(m, info)?;
        let reference = self.surface_caches.shadow_shape(block, &key)?;
        Some(ShapeUse {
            block,
            writes: takes_storage(m),
            reference,
            diag: None,
            video_plane: None,
        })
    }
}

/// Whether the texture `m` asks takes storage (`shapes`, "The shape"), and so may be written
/// where it is bound: its usage is Unknown or holds `ShaderWrite` or `ShaderAtomic`.
fn takes_storage(m: &RgfxTextureFromIOSurface) -> bool {
    let usage = u64::from(m.desc.usage);
    usage == 0 || usage & (TEXTURE_USAGE_WRITES | TEXTURE_USAGE_ATOMIC) != 0
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use reims_vgpu_graphics::wire::{RGFX_STATUS_CACHE, RGFX_STATUS_OK};
    use reims_vgpu_vulkan::core::testkit::FakePlatform;
    use reims_vgpu_vulkan::core::testkit::calllog::{Arg, Call};

    use reims_vgpu_graphics::ops::{RGFX_SURFACE_USE_WRITES, RgfxOpSurfaceUse};
    use reims_vgpu_vulkan::core::platform::command::{CommandBuffer as _, CommandQueue as _};
    use reims_vgpu_vulkan::core::platform::device::GpuDevice as _;
    use reims_vgpu_vulkan::core::raise::OnFault;
    use reims_vgpu_vulkan::core::testkit::objects::CompleteOn;

    #[cfg(unix)]
    use reims_vgpu_vulkan::core::platform::types::ClearColor;
    use reims_vgpu_vulkan::core::platform::types::MtlSize;
    #[cfg(unix)]
    use reims_vgpu_vulkan::core::platform::values::PagesPlane;
    use reims_vgpu_vulkan::core::sync::Mutex;

    use super::super::surface_cache::tests::{
        NO_REACH, blit_with, blits_with, buffer_bytes, cache_id, cached, complete_round,
        complete_staged, copies, counted, download, encoder_with, pages_id, queues, ram_bytes,
        reading, release, surface_texture, surface_texture_as, surface_textures, use_surfaces,
    };
    use super::super::surface_pages::tests::{BLOCK, Rig, one_plane, rig};
    use super::*;
    use crate::platform::ApiPlatform as _;
    use reims_vgpu_vulkan::core::testkit::Step;
    use reims_vgpu_vulkan::core::testkit::impls::resource::TEXTURE_SLOT_WRITES;

    const ON: OnFault = OnFault::Return;

    /// A shape's image (`ApiPlatform::own_texture`).
    const IMAGE: &str = "newTextureWithDescriptor:";
    /// A guest texture as a view of its shape's image, at its resource index.
    const VIEW: &str = "newTextureViewWithPixelFormat:resourceIndex:";
    /// A shape's fill from its cache's buffer ([`shapes::fill`]).
    const FILL: &str = "copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:\
                        sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:";

    /// A shape written back into its cache's buffer ([`shapes::write_back`]).
    const TO_BUFFER: &str = "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:\
                             toBuffer:destinationOffset:destinationBytesPerRow:\
                             destinationBytesPerImage:";

    /// The (pixel format, usage) of each shape's image the rig's decoder made, in order.
    type Made = Arc<Mutex<Vec<(u64, u64)>>>;

    /// The rig's decoder takes the per-shape form, each image the platform's own texture, kept
    /// in the list returned.
    fn take_shapes(r: &Rig) -> Made {
        let p = Arc::clone(&r.p);
        let made = Made::default();
        let kept = Arc::clone(&made);
        r.dec.set_surface_shapes(
            Arc::new(
                move |device: &<FakePlatform as Platform>::Device,
                      desc: &TextureDesc,
                      _placed: bool| {
                    kept.lock().push((desc.pixel_format, desc.usage));
                    p.own_texture(device, desc)
                },
            ),
            false,
        );
        made
    }

    /// Whether each image the rig's decoder made was asked placed, in order.
    type Placed = Arc<Mutex<Vec<bool>>>;

    /// [`shaped`]'s rig with a maker that places images ("Placed images"), the fake's own
    /// textures standing in for placed ones, and whether each make asked one placed.
    fn placing() -> (Rig, Placed) {
        let mut r = rig(8);
        let page = r.page;
        let placed = places(&r);
        r.dec.hold_resource_ids(0, 1);
        assert_eq!(
            r.pages_cached(BLOCK, &[(1, 2)], 2 * page, true),
            RGFX_STATUS_OK
        );
        queues(&mut r);
        assert_eq!(r.buffer(10, BLOCK, 2 * page), Ok(()));
        (r, placed)
    }

    /// [`placing`]'s rig over a deferred block of two pages (`surface_cache`'s module doc,
    /// "Deferred blocks"), as a surface the CPU has not locked for writing: no use fills its
    /// cache from its pages, which are not named, so its buffer holds no memory until a copy
    /// goes through it. Buffer 10 is the cache.
    fn placing_deferred() -> (Rig, Placed) {
        use reims_vgpu_graphics::wire::{RGFX_SURFACE_PAGES_CACHED, RGFX_SURFACE_PAGES_DEFERRED};
        let mut r = rig(8);
        let page = r.page;
        r.dec.set_deferred_budget(8 * page);
        let placed = places(&r);
        r.dec.hold_resource_ids(0, 1);
        let flags = RGFX_SURFACE_PAGES_CACHED | RGFX_SURFACE_PAGES_DEFERRED;
        let answer = r.pages_flagged(BLOCK, 0, &[], 2 * page, flags);
        assert_eq!(answer, RGFX_STATUS_OK);
        queues(&mut r);
        assert_eq!(r.buffer(10, BLOCK, 2 * page), Ok(()));
        (r, placed)
    }

    /// The rig's decoder takes the per-shape form with a maker that places images, the fake's own
    /// textures standing in for placed ones: whether each make asked one placed, in order.
    fn places(r: &Rig) -> Placed {
        let p = Arc::clone(&r.p);
        let placed = Placed::default();
        let kept = Arc::clone(&placed);
        r.dec.set_surface_shapes(
            Arc::new(
                move |device: &<FakePlatform as Platform>::Device,
                      desc: &TextureDesc,
                      asked: bool| {
                    kept.lock().push(asked);
                    p.own_texture(device, desc)
                },
            ),
            true,
        );
        placed
    }

    /// [`cached`]'s rig in the per-shape form, its texture resource ids held, with buffer 10 over
    /// the block's two pages.
    fn shaped() -> Rig {
        shaped_made().0
    }

    /// [`shaped`], and the images its decoder makes ([`take_shapes`]).
    fn shaped_made() -> (Rig, Made) {
        let mut r = rig(8);
        let page = r.page;
        let made = take_shapes(&r);
        r.dec.hold_resource_ids(0, 1);
        assert_eq!(
            r.pages_cached(BLOCK, &[(1, 2)], 2 * page, true),
            RGFX_STATUS_OK
        );
        queues(&mut r);
        assert_eq!(r.buffer(10, BLOCK, 2 * page), Ok(()));
        (r, made)
    }

    /// The cache's buffer's memory made, as a copy through it makes it, so its fills go through
    /// it: a fill straight into the shapes goes past a buffer that holds none (`surface_cache`'s
    /// module doc, "Straight fills").
    fn buffer_made(r: &Rig) {
        let caches = r.dec.caches();
        let buffer = caches.buffer(BLOCK).unwrap();
        assert!(r.dec.surface_caches.memory_of(None, buffer));
    }

    /// Texture `obj_id` over buffer 10, of `format` and `size`.
    fn texture(r: &mut Rig, obj_id: u64, format: u32, size: (u32, u32)) {
        let message = surface_texture(obj_id, 10, format, size);
        assert_eq!(r.dec.decode(&r.device, &message).unwrap(), Ok(()));
    }

    /// [`texture`], and the image of its shape, which its view is of.
    fn texture_image(r: &mut Rig, obj_id: u64, format: u32, size: (u32, u32)) -> u64 {
        texture_as_image(r, obj_id, format, size, 1)
    }

    /// [`texture_image`] of a texture that declares `usage`.
    fn texture_as_image(
        r: &mut Rig,
        obj_id: u64,
        format: u32,
        size: (u32, u32),
        usage: u32,
    ) -> u64 {
        let mark = r.p.call_log().mark();
        let message = surface_texture_as(obj_id, 10, format, size, (usage, 0));
        assert_eq!(r.dec.decode(&r.device, &message).unwrap(), Ok(()));
        views(&r.p.call_log().since(mark))[0].0
    }

    /// The buffer over no surface that [`write`]'s blits copy from, as a guest's do: the bundle
    /// names no cache's buffer in a command.
    const SOURCE: u64 = 40;

    /// Command buffer `cb_obj_id` on queue 1, naming [`BLOCK`] written, that writes texture
    /// `obj_id` of `size` by a blit from buffer [`SOURCE`], made first where it is not.
    fn write(r: &mut Rig, cb_obj_id: u64, obj_id: u64, size: (u32, u32)) {
        source(r);
        blit_with(r, cb_obj_id, true, write_op(obj_id, size));
    }

    /// [`write`] by a blit from buffer `from`, which a test names the cache's buffer by.
    fn write_from(r: &mut Rig, cb_obj_id: u64, obj_id: u64, size: (u32, u32), from: u64) {
        blit_with(r, cb_obj_id, true, write_op_from(obj_id, size, from));
    }

    /// Buffer [`SOURCE`], a Shared one of 64 KiB, made where it is not.
    fn source(r: &mut Rig) {
        use reims_vgpu_graphics::wire::{RGFX_CMD_BUFFER_CREATE, RgfxBufferCreate, RgfxMsgHeader};
        use zerocopy::IntoBytes as _;
        if r.dec.object(SOURCE).is_some() {
            return;
        }
        let m = RgfxBufferCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_BUFFER_CREATE,
                size: size_of::<RgfxBufferCreate>() as u32,
                seqno: 0,
            },
            obj_id: SOURCE,
            length: 0x1_0000,
            ..RgfxBufferCreate::default()
        };
        assert_eq!(r.dec.decode(&r.device, m.as_bytes()).unwrap(), Ok(()));
    }

    /// A blit op that writes texture `obj_id` of `size` from buffer [`SOURCE`], which the test
    /// makes ([`source`]).
    fn write_op(obj_id: u64, size: (u32, u32)) -> (u16, Vec<u8>) {
        write_op_from(obj_id, size, SOURCE)
    }

    /// [`write_op`] from buffer `from`.
    fn write_op_from(obj_id: u64, size: (u32, u32), from: u64) -> (u16, Vec<u8>) {
        use reims_vgpu_graphics::ops::blit::{
            RGFX_OP_BLIT_COPY_BUFFER_TO_TEXTURE, RgfxBlitOpCopyBufferToTexture,
        };
        use zerocopy::IntoBytes as _;
        let (width, height) = (u64::from(size.0), u64::from(size.1));
        let m = RgfxBlitOpCopyBufferToTexture {
            source_buffer_obj_id: from,
            destination_texture_obj_id: obj_id,
            source_bytes_per_row: width * 4,
            source_bytes_per_image: width * height * 4,
            source_size_width: width,
            source_size_height: height,
            source_size_depth: 1,
            ..RgfxBlitOpCopyBufferToTexture::default()
        };
        (RGFX_OP_BLIT_COPY_BUFFER_TO_TEXTURE, m.as_bytes().to_vec())
    }

    /// The (source image, size) of each copy of a shape's write into its cache's buffer at an
    /// encoder's end in `calls`, and the (image, size) of each shape that took it from there, the
    /// guest's own copies into views left out ([`shapes::copy_overlaps`]).
    #[allow(clippy::type_complexity)] // the two lists the test reads
    fn overlaps(calls: &[Call], images: &[u64]) -> (Vec<(u64, Arg)>, Vec<(u64, Arg)>) {
        let id = |a: &Arg| a.as_obj().map_or(0, |o| o.0);
        let out = (calls.iter())
            .filter(|c| c.sel == TO_BUFFER)
            .map(|c| (id(&c.args[0]), c.args[4].clone()))
            .collect();
        let into = (calls.iter())
            .filter(|c| c.sel == FILL && images.contains(&id(&c.args[5])))
            .map(|c| (id(&c.args[5]), c.args[4].clone()))
            .collect();
        (out, into)
    }

    /// `width` by `height` texels, as the call log holds a copy's size.
    fn texels(width: u64, height: u64) -> Arg {
        Arg::val(&MtlSize::new(width, height, 1))
    }

    /// The (image, buffer, offset) of each shape written back in `calls`.
    fn written_back(calls: &[Call]) -> Vec<(u64, u64, u64)> {
        let id = |a: &Arg| a.as_obj().map_or(0, |o| o.0);
        (calls.iter())
            .filter(|c| c.sel == TO_BUFFER)
            .map(|c| {
                (
                    id(&c.args[0]),
                    id(&c.args[5]),
                    c.args[6].as_u64().unwrap_or(0),
                )
            })
            .collect()
    }

    /// Where the first call to `sel` is in `calls`.
    fn first(calls: &[Call], sel: &str) -> usize {
        calls.iter().position(|c| c.sel == sel).unwrap()
    }

    /// The guest destroys object `obj_id`.
    fn destroy(r: &mut Rig, obj_id: u64) {
        use reims_vgpu_graphics::wire::{
            RGFX_CMD_OBJECT_DESTROY, RgfxMsgHeader, RgfxObjectDestroy,
        };
        use zerocopy::IntoBytes as _;
        let destroy = RgfxObjectDestroy {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_OBJECT_DESTROY,
                size: size_of::<RgfxObjectDestroy>() as u32,
                seqno: 0,
            },
            obj_id,
        };
        assert_eq!(r.dec.decode(&r.device, destroy.as_bytes()).unwrap(), Ok(()));
    }

    /// The (image, pixel format, resource index) of each view in `calls`.
    fn views(calls: &[Call]) -> Vec<(u64, u64, u64)> {
        (calls.iter())
            .filter(|c| c.sel == VIEW)
            .map(|c| {
                let at = |i: usize| c.args[i].as_u64().unwrap_or(u64::MAX);
                (c.obj.0, at(0), at(1))
            })
            .collect()
    }

    /// The (buffer, offset, image) of each fill of a shape in `calls`.
    fn fills(calls: &[Call]) -> Vec<(u64, u64, u64)> {
        let id = |a: &Arg| a.as_obj().map_or(0, |o| o.0);
        (calls.iter())
            .filter(|c| c.sel == FILL)
            .map(|c| {
                (
                    id(&c.args[0]),
                    c.args[1].as_u64().unwrap_or(0),
                    id(&c.args[5]),
                )
            })
            .collect()
    }

    /// In the per-shape form a texture over a cache is a view, at the guest's resource index and
    /// in the guest's format, of its shape's image, and no texture lies over the buffer: a second
    /// texture of the same shape, in its format or the format's sRGB sibling, is another view of
    /// the same image; one of another size has an image of its own, and so has one that takes
    /// storage ([`a_texture_that_takes_storage_over_a_plain_shape_makes_a_second_shape`]). Each
    /// holds its shape, and writes it where its usage lets a shader write. In the default form
    /// the texture lies over the buffer, and holds only its shape in the cache's shadow, which
    /// makes no image
    #[test]
    fn a_texture_over_a_cache_is_a_view_of_its_shapes_image() {
        let mut r = shaped();
        let mark = r.p.call_log().mark();
        texture(&mut r, 21, 80, (32, 8));
        texture(&mut r, 22, 80, (32, 8));
        texture(&mut r, 23, 81, (32, 8));
        texture(&mut r, 24, 80, (16, 8));
        let message = surface_texture_as(25, 10, 80, (32, 8), (0x2, 0));
        assert_eq!(r.dec.decode(&r.device, &message).unwrap(), Ok(()));
        let calls = r.p.call_log().since(mark);
        assert!(surface_textures(&calls).is_empty(), "none over the buffer");
        assert_eq!(calls.iter().filter(|c| c.sel == IMAGE).count(), 3);
        let seen = views(&calls);
        let at = |n: u64| (seen[0].0, if n == 23 { 81 } else { 80 }, 0x40 + n);
        assert_eq!(seen.len(), 5);
        assert_eq!(seen[..3], [at(21), at(22), at(23)]);
        assert_ne!(seen[3].0, seen[0].0, "another shape, another image");
        assert_eq!((seen[3].1, seen[3].2), (80, 0x40 + 24));
        let others = [seen[0].0, seen[3].0];
        assert!(!others.contains(&seen[4].0), "storage's own image");
        assert_eq!((seen[4].1, seen[4].2), (80, 0x40 + 25));
        let counts = r.dec.caches().counts().shapes;
        assert_eq!((counts.made, counts.found, counts.unshaped), (3, 2, 0));
        assert_eq!(counts.made_bytes, 2 * 32 * 8 * 4 + 16 * 8 * 4);
        assert_eq!(counts.format_storage, 0);
        let shape = |n: u64| {
            let entry = r.dec.entry(n).unwrap();
            entry.shape.as_ref().map(|s| (s.block, s.writes))
        };
        assert_eq!(shape(21), Some((BLOCK, false)));
        assert_eq!(shape(25), Some((BLOCK, true)), "ShaderWrite writes");

        let mut r = cached();
        r.dec.hold_resource_ids(0, 1);
        assert_eq!(r.buffer(10, BLOCK, 2 * r.page), Ok(()));
        let mark = r.p.call_log().mark();
        texture(&mut r, 21, 80, (32, 8));
        let calls = r.p.call_log().since(mark);
        assert_eq!(surface_textures(&calls), [(cache_id(&r), 0, 128)]);
        assert!(views(&calls).is_empty());
        assert!(calls.iter().all(|c| c.sel != IMAGE));
        let entry = r.dec.entry(21).unwrap();
        let shadow = entry.shape.as_ref().map(|s| (s.block, s.writes));
        assert_eq!(shadow, Some((BLOCK, false)), "the shadow's");
        assert_eq!(r.dec.caches().counts().shapes, ShapeCounts::default());
    }

    /// Where the platform places shapes' images, a new shape of a cache no command buffer uses,
    /// filled from its pages once, takes a placed image, filled at its make before its view is
    /// answered, straight from the pages, as the buffer holds no memory, and keeps it; one made
    /// while a command buffer begun and not finished uses the cache takes the platform's own
    /// image, its fill asked of that buffer as before, counted (`rt-shape-unplaced-busy`). A
    /// platform that places none is asked for none
    #[test]
    fn a_placed_shape_is_filled_at_its_make_and_one_of_a_busy_cache_is_the_platforms_own() {
        let (mut r, placed) = placing();
        let pages = pages_id(&r);
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 2, false)]);
        r.p.world().complete_all(CompleteOn::Caller);
        let mark = r.p.call_log().mark();
        let image = texture_image(&mut r, 21, 80, (32, 8));
        let calls = r.p.call_log().since(mark);
        assert_eq!(*placed.lock(), [true]);
        assert_eq!(fills(&calls), [(pages, 0, image)], "filled at its make");
        assert!(first(&calls, FILL) < first(&calls, VIEW), "before its view");
        let counts = r.dec.caches().counts().shapes;
        assert_eq!((counts.placed, counts.placed_bytes), (1, 32 * 8 * 4));
        assert_eq!((counts.made, counts.fills_made), (1, 1));
        assert_eq!((counts.unplaced_busy, counts.unplaced_unfilled), (0, 0));
        assert_eq!(texture_image(&mut r, 22, 80, (32, 8)), image, "its shape's");
        assert_eq!(*placed.lock(), [true], "found, not made");

        let queue = r.device.new_command_queue().unwrap();
        let cb = queue.command_buffer(ON).unwrap().unwrap();
        let (_, uses) = (r.dec.surface_caches)
            .before(&r.device, &cb, &reading(BLOCK, 0), &NO_REACH)
            .unwrap();
        let mark = r.p.call_log().mark();
        let own = texture_image(&mut r, 23, 80, (16, 4));
        let calls = r.p.call_log().since(mark);
        assert_ne!(own, image);
        assert_eq!(*placed.lock(), [true, false], "the platform's own");
        assert!(fills(&calls).is_empty(), "owed, as before");
        let counts = r.dec.caches().counts().shapes;
        assert_eq!((counts.placed, counts.unplaced_busy), (1, 1));
        assert_eq!(counts.made, 2);
        (r.dec.surface_caches)
            .finish(&r.device, 1, None, &cb, uses.unwrap(), || cb.commit(ON))
            .unwrap();
        r.p.world().complete_all(CompleteOn::Caller);

        let (mut r, made) = shaped_made();
        texture(&mut r, 21, 80, (32, 8));
        assert_eq!(made.lock().len(), 1);
        let counts = r.dec.caches().counts().shapes;
        assert_eq!(
            (counts.placed, counts.unplaced_busy),
            (0, 0),
            "asked for none"
        );
    }

    /// Zeros: a new shape of a cache whose buffer holds no memory yet, and no shape a write the
    /// buffer lacks, takes the platform's own image though the platform places images, counted
    /// (`rt-shape-unplaced-zeros`); the first use of the surface takes its fill as made, with no
    /// copy and the buffer's memory left unmade, counted (`rt-shape-zero-taken`), as the platform
    /// cleared its image and the buffer reads zero. A CPU lock, its pages named then, moves none
    /// of the shape's bytes: it holds the buffer's zeros. Once a shape holds a write, a new shape
    /// takes the buffer's bytes, the write copied into them first: placed and filled at its make,
    /// the buffer's memory made for that fill
    #[test]
    fn a_shape_over_a_buffer_with_no_memory_takes_its_zeros_with_no_copy() {
        use reims_vgpu_graphics::wire::{RGFX_SURFACE_PAGES_ATTACH, RGFX_SURFACE_PAGES_CACHED};
        use reims_vgpu_vulkan::core::testkit::impls::resource::MAKE_MEMORY;
        let (mut r, placed) = placing_deferred();
        let page = r.page;
        let cache = cache_id(&r);
        let image = texture_image(&mut r, 21, 80, (32, 8));
        assert_eq!(*placed.lock(), [false], "the platform's own");
        let counts = r.dec.caches().counts().shapes;
        assert_eq!((counts.placed, counts.unplaced_zeros), (0, 1));
        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 0, false)]);
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(fills(&r.p.call_log().since(mark)).is_empty(), "taken");
        assert_eq!(r.p.call_log().count(MAKE_MEMORY), 0, "left unmade");
        let counts = r.dec.caches().counts().shapes;
        assert_eq!(
            (counts.zero_taken, counts.zero_taken_bytes, counts.fills),
            (1, 32 * 8 * 4, 0)
        );

        let flags = RGFX_SURFACE_PAGES_CACHED | RGFX_SURFACE_PAGES_ATTACH;
        let attached = r.pages_flagged(BLOCK, 7, &[(1, 2)], 2 * page, flags);
        assert_eq!(attached, RGFX_STATUS_OK);
        let mark = r.p.call_log().mark();
        let downloaded = download(&r, BLOCK);
        r.p.world().complete_all(CompleteOn::Caller);
        let calls = r.p.call_log().since(mark);
        assert!(downloaded.load(Ordering::SeqCst));
        assert!(written_back(&calls).is_empty(), "nothing written");
        assert!(fills(&calls).is_empty());

        write(&mut r, 0x101, 21, (32, 8));
        r.p.world().complete_all(CompleteOn::Caller);
        let mark = r.p.call_log().mark();
        let new = texture_image(&mut r, 22, 80, (16, 4));
        assert_eq!(*placed.lock(), [false, true]);
        assert_eq!(fills(&r.p.call_log().since(mark)), [(cache, 0, new)]);
        assert_eq!(r.p.call_log().count(MAKE_MEMORY), 1, "its fill's");
        let counts = r.dec.caches().counts().shapes;
        assert_eq!(
            (counts.zero_taken, counts.unplaced_zeros, counts.placed),
            (1, 1, 1)
        );
        assert_ne!(new, image);
    }

    /// Zeros are looked at as a shape's fill is made, not at its make: where the cache's buffer
    /// takes its memory between the two (a copy through it made with the caches' lock let go of),
    /// the shape is filled from the buffer, with what that copy wrote, and nothing is taken
    #[test]
    fn a_shapes_zeros_are_looked_at_as_its_fill_is_made() {
        let (mut r, placed) = placing_deferred();
        let cache = cache_id(&r);
        let image = texture_image(&mut r, 21, 80, (32, 8));
        assert_eq!(*placed.lock(), [false], "zeros at its make");
        buffer_made(&r);
        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 0, false)]);
        r.p.world().complete_all(CompleteOn::Caller);
        assert_eq!(fills(&r.p.call_log().since(mark)), [(cache, 0, image)]);
        let counts = r.dec.caches().counts().shapes;
        assert_eq!((counts.zero_taken, counts.fills), (0, 1));
    }

    /// A shape holding a write the buffer lacks keeps a new shape from zeros: a shape whose
    /// zeros were taken and that a shader then wrote is dirty, so a new shape of the cache, whose
    /// buffer holds no memory still, is placed and filled at its make, the dirty shape written
    /// back into the buffer first, the buffer's memory made before that copy
    #[test]
    fn a_new_shape_over_a_dirty_one_is_filled_though_the_buffer_held_no_memory() {
        use reims_vgpu_vulkan::core::testkit::impls::resource::MAKE_MEMORY;
        let (mut r, placed) = placing_deferred();
        let cache = cache_id(&r);
        compute_pipeline(&mut r, 31);
        let written = writable_image(&mut r, 25);
        computes(&mut r, 0x100, &[bind_pipeline(31), bind_texture(25, 0)]);
        r.p.world().complete_all(CompleteOn::Caller);
        assert_eq!(r.p.call_log().count(MAKE_MEMORY), 0, "zeros taken");
        assert_eq!(r.dec.caches().counts().shapes.zero_taken, 1);

        let mark = r.p.call_log().mark();
        let new = texture_image(&mut r, 22, 80, (16, 4));
        let calls = r.p.call_log().since(mark);
        assert_eq!(*placed.lock(), [false, true]);
        assert_eq!(written_back(&calls), [(written, cache, 0)]);
        assert_eq!(fills(&calls), [(cache, 0, new)]);
        let back = first(&calls, TO_BUFFER);
        assert!(first(&calls, MAKE_MEMORY) < back, "made before");
        assert!(back < first(&calls, FILL), "written back first");
        let counts = r.dec.caches().counts().shapes;
        assert_eq!(
            (counts.zero_taken, counts.unplaced_zeros, counts.placed),
            (1, 1, 1)
        );
    }

    /// The usage a texture declares picks its shape's kind: one that declares Metal's Unknown
    /// usage (0), `ShaderWrite` or `ShaderAtomic` takes storage, its image made of the Unknown
    /// usage, and writes its shape where it is bound; any other takes a plain shape, its image
    /// made of `ShaderRead | RenderTarget` alone, and leaves it clean; each storage shape made
    /// over the plane while it holds the plain one is counted, whatever its extent. Over a plain
    /// shape a texture in the shape's format's sRGB sibling is a view of it, and one that declares
    /// `PixelFormatView` is counted; one in another format, a view of the plain image rt refuses,
    /// takes the storage shape of its extent, made of the Unknown usage beside the plain one,
    /// counted, and a later texture of the extent views that storage shape
    #[test]
    fn a_textures_declared_usage_picks_its_shapes_kind() {
        let (mut r, made) = shaped_made();
        let plain = texture_as_image(&mut r, 21, 80, (32, 8), 0x5);
        let mut storage = Vec::new();
        for (obj_id, usage, width) in [(22, 0x2, 16), (23, 0x20, 8), (24, 0, 24)] {
            storage.push(texture_as_image(&mut r, obj_id, 80, (width, 8), usage));
        }
        assert!(!storage.contains(&plain));
        assert_eq!(*made.lock(), [(80, 0x5), (80, 0), (80, 0), (80, 0)]);
        let writes = |r: &Rig, n: u64| r.dec.entry(n).unwrap().shape.as_ref().map(|s| s.writes);
        let seen = [21, 22, 23, 24].map(|n| writes(&r, n));
        assert_eq!(seen, [Some(false), Some(true), Some(true), Some(true)]);

        let sibling = texture_as_image(&mut r, 25, 81, (32, 8), 0x1);
        let format_view = texture_as_image(&mut r, 26, 80, (32, 8), 0x11);
        assert_eq!([sibling, format_view], [plain; 2]);
        assert_eq!(made.lock().len(), 4);
        let other_format = texture_as_image(&mut r, 27, 70, (32, 8), 0x5);
        assert!(![plain].iter().chain(&storage).any(|&i| i == other_format));
        assert_eq!(made.lock()[4..], [(70, 0)], "a storage shape of its extent");
        assert_eq!(writes(&r, 27), Some(false), "writes as its usage says");
        assert_eq!(texture_as_image(&mut r, 28, 80, (32, 8), 0x5), other_format);
        let counts = r.dec.caches().counts().shapes;
        assert_eq!((counts.made, counts.found), (5, 4));
        assert_eq!(counts.order_case, 4, "beside the plain one");
        let plain_counts = (counts.format_storage, counts.plain_format_views);
        assert_eq!(plain_counts, (1, 1));
        assert_eq!((counts.unshaped, counts.views_refused), (0, 0));
    }

    /// A texture over a plain shape in a format the plain image is not viewed in without
    /// `PixelFormatView` (`RGBA8Unorm` over a `BGRA8Unorm` plain shape), a view rt refuses, is a
    /// view of the storage shape of its extent, made beside the plain one, and nothing lies over
    /// the buffer: rule 1 keeps the two current, a write through the plain shape reaching it at
    /// the end of its encoder, as a Mac's next encoder reads a write made through another texture
    /// over the same memory. Over the buffer, as before, it would have read the write only once
    /// a copy out of the cache wrote the plain shape back
    #[test]
    fn a_texture_in_a_format_its_plain_shape_is_not_viewed_in_takes_the_storage_shape() {
        let (mut r, made) = shaped_made();
        let mark = r.p.call_log().mark();
        let plain = texture_as_image(&mut r, 21, 80, (32, 8), 0x5);
        let other = texture_as_image(&mut r, 27, 70, (32, 8), 0x1);
        assert_ne!(other, plain);
        assert_eq!(*made.lock(), [(80, 0x5), (70, 0)]);
        let calls = r.p.call_log().since(mark);
        assert!(surface_textures(&calls).is_empty(), "none over the buffer");
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 0, false)]);
        r.p.world().complete_all(CompleteOn::Caller);

        let mark = r.p.call_log().mark();
        write(&mut r, 0x101, 21, (32, 8));
        let (out, into) = overlaps(&r.p.call_log().since(mark), &[plain, other]);
        assert_eq!(out, [(plain, texels(32, 8))]);
        assert_eq!(into, [(other, texels(32, 8))]);
        r.p.world().complete_all(CompleteOn::Caller);
        let counts = r.dec.caches().counts().shapes;
        assert_eq!((counts.format_storage, counts.order_case), (1, 1));
        assert_eq!((counts.unshaped, counts.views_refused), (0, 0));
    }

    /// A texture that takes storage over a plane with only a plain shape of its extent makes a
    /// storage shape beside it, counted, and rule 1 keeps the two current: a write through the
    /// plain shape reaches the storage shape at the end of its encoder, counted as a copy between
    /// the kinds. A later texture of the extent views the storage shape, whatever it declares
    #[test]
    fn a_texture_that_takes_storage_over_a_plain_shape_makes_a_second_shape() {
        let (mut r, made) = shaped_made();
        let plain = texture_as_image(&mut r, 21, 80, (32, 8), 0x5);
        let storage = texture_as_image(&mut r, 25, 80, (32, 8), 0x2);
        assert_ne!(storage, plain);
        assert_eq!(*made.lock(), [(80, 0x5), (80, 0)]);
        assert_eq!(r.dec.caches().counts().shapes.order_case, 1);
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 0, false)]);
        r.p.world().complete_all(CompleteOn::Caller);

        let mark = r.p.call_log().mark();
        write(&mut r, 0x101, 21, (32, 8));
        let (out, into) = overlaps(&r.p.call_log().since(mark), &[plain, storage]);
        assert_eq!(out, [(plain, texels(32, 8))]);
        assert_eq!(into, [(storage, texels(32, 8))]);
        let counts = r.dec.caches().counts().shapes;
        assert_eq!((counts.order_copies, counts.order_bytes), (1, 32 * 8 * 4));
        r.p.world().complete_all(CompleteOn::Caller);

        assert_eq!(texture_as_image(&mut r, 26, 80, (32, 8), 0x5), storage);
        assert_eq!(texture_as_image(&mut r, 27, 70, (32, 8), 0x1), storage);
        let counts = r.dec.caches().counts().shapes;
        assert_eq!((counts.made, counts.found, counts.order_case), (2, 2, 1));
        assert_eq!(counts.format_storage, 0, "any view of storage");
    }

    /// In the default form the counts read what the per-shape form would hold, and nothing else
    /// changes: at each new most of the caches' bytes, those of surfaces a texture's shape
    /// references apart from those of surfaces none does; and the most the bound would hold,
    /// each referenced shape and the 4 unreferenced of each cache used last, by their texels
    #[test]
    fn the_default_form_counts_what_the_per_shape_form_would_hold() {
        let mut r = cached();
        let page = r.page;
        r.dec.hold_resource_ids(0, 1);
        assert_eq!(r.buffer(10, BLOCK, 2 * page), Ok(()));
        texture(&mut r, 21, 80, (32, 8));
        let other = BLOCK + 0x10000;
        assert_eq!(r.pages_cached(other, &[(4, 1)], page, true), RGFX_STATUS_OK);
        let counts = r.dec.caches().counts();
        assert_eq!(counts.held_most, 3 * page);
        let split = (counts.held_referenced, counts.held_kept, counts.held_video);
        assert_eq!(split, (2 * page, page, 0), "referenced, kept, video");
        assert_eq!(counts.shape_bound_most, 32 * 8 * 4);
        // Six shapes more, each unreferenced once made: the bound keeps four of them.
        for (n, width) in (1..=6).enumerate() {
            let obj_id = 22 + n as u64;
            texture(&mut r, obj_id, 80, (width, 8));
            destroy(&mut r, obj_id);
        }
        // Widths 1 to 5 at width 5's make; then 2 to 6, width 1 gone, at width 6's.
        let counts = r.dec.caches().counts();
        assert_eq!(counts.shape_bound_most, (32 + 2 + 3 + 4 + 5 + 6) * 8 * 4);
        assert_eq!(counts.held_most, 3 * page, "no bytes held for the shadow");
        assert_eq!(counts.shapes, ShapeCounts::default());
    }

    /// A new shape owes its fill, which the next command buffer that names the block makes: the
    /// cache's first fill, its buffer holding no memory, copies every shape's bytes straight from
    /// the pages; a shape made later is filled alone in the next that names the block, straight
    /// too, with no copy into the buffer. Once the buffer holds memory, a newer generation fills
    /// the buffer from the pages and every shape from the buffer
    #[test]
    fn a_new_shape_takes_the_surfaces_bytes_before_its_first_use() {
        let mut r = shaped();
        let (cache, pages) = (cache_id(&r), pages_id(&r));
        texture(&mut r, 21, 80, (32, 8));
        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 2, false)]);
        let calls = r.p.call_log().since(mark);
        assert!(copies(&calls).is_empty(), "straight into the shape");
        let first = fills(&calls);
        assert_eq!(first.len(), 1);
        assert_eq!((first[0].0, first[0].1), (pages, 0));
        // A second shape, filled alone.
        texture(&mut r, 22, 80, (16, 4));
        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 1, 0x101, &[(BLOCK, 2, false)]);
        let calls = r.p.call_log().since(mark);
        assert!(copies(&calls).is_empty(), "no copy into the buffer");
        let second = fills(&calls);
        assert_eq!(second.len(), 1);
        assert_eq!(second[0].0, pages);
        assert_ne!(second[0].2, first[0].2, "the new shape's image");
        // Nothing owed: nothing filled.
        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 1, 0x102, &[(BLOCK, 2, false)]);
        assert!(fills(&r.p.call_log().since(mark)).is_empty());
        // The buffer's memory made, a newer generation: the buffer, then every shape from it.
        r.p.world().complete_all(CompleteOn::Caller);
        buffer_made(&r);
        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 1, 0x103, &[(BLOCK, 3, false)]);
        let calls = r.p.call_log().since(mark);
        assert_eq!(copies(&calls), [(pages, cache)]);
        let third = fills(&calls);
        assert_eq!(third.len(), 2);
        assert!(
            third.iter().all(|&(from, ..)| from == cache),
            "from the buffer"
        );
        let counts = r.dec.caches().counts().shapes;
        assert_eq!(counts.fills, 4);
        assert_eq!(counts.filled_bytes, 2 * (32 * 8 * 4 + 16 * 4 * 4));
        assert_eq!(counts.straight_fills, 2);
    }

    /// The bound: once a texture's shape is unreferenced it stays, up to four of the block's
    /// unreferenced shapes, the least recently used going first, counted with its bytes; a
    /// texture of a shape that went makes it again, counted, and a referenced shape never goes
    #[test]
    fn unreferenced_shapes_past_four_go_least_recently_used_first() {
        let mut r = shaped();
        texture(&mut r, 20, 80, (32, 8));
        for (n, width) in (1..=6).enumerate() {
            let obj_id = 21 + n as u64;
            texture(&mut r, obj_id, 80, (width, 8));
            destroy(&mut r, obj_id);
        }
        // Width 1 went at width 6's make, five unreferenced then.
        let counts = r.dec.caches().counts().shapes;
        assert_eq!((counts.made, counts.dropped), (7, 1));
        assert_eq!(counts.dropped_bytes, 8 * 4, "width 1's");
        // Width 1 again: made again, counted, and width 2 goes; width 6 is still held.
        texture(&mut r, 30, 80, (1, 8));
        texture(&mut r, 31, 80, (6, 8));
        let counts = r.dec.caches().counts().shapes;
        assert_eq!((counts.made, counts.again, counts.found), (8, 1, 1));
        assert_eq!(counts.dropped, 2, "the referenced one kept");
        assert_eq!(counts.dropped_bytes, (1 + 2) * 8 * 4);
        assert!(r.dec.entry(20).unwrap().shape.is_some());
    }

    /// The age rule: a shape no texture references, unused for [`shapes::UNREFERENCED_FOR`], goes
    /// at the next new shape of any block's cache, though the bound would keep it, counted with
    /// its bytes; one used within the period and a referenced one stay, and a texture of the
    /// shape that went makes it again, counted
    #[test]
    fn unreferenced_shapes_unused_for_the_period_go_at_any_new_shape() {
        let mut r = shaped();
        let page = r.page;
        texture(&mut r, 20, 80, (32, 8));
        texture(&mut r, 21, 80, (4, 8));
        destroy(&mut r, 21);
        let other = BLOCK + 0x10000;
        assert_eq!(r.pages_cached(other, &[(4, 1)], page, true), RGFX_STATUS_OK);
        assert_eq!(r.buffer(11, other, page), Ok(()));
        let half = shapes::UNREFERENCED_FOR / 2;
        r.p.virtual_clock().advance(half);
        texture(&mut r, 22, 80, (5, 8));
        destroy(&mut r, 22);
        r.p.virtual_clock().advance(half + std::time::Duration::from_millis(1));
        assert_eq!(r.dec.caches().counts().shapes.dropped, 0);
        let message = surface_texture(30, 11, 80, (8, 8));
        assert_eq!(r.dec.decode(&r.device, &message).unwrap(), Ok(()));
        let counts = r.dec.caches().counts().shapes;
        assert_eq!((counts.dropped, counts.aged), (1, 1), "width 4's, at the other's make");
        assert_eq!((counts.dropped_bytes, counts.aged_bytes), (4 * 8 * 4, 4 * 8 * 4));
        texture(&mut r, 31, 80, (5, 8));
        texture(&mut r, 32, 80, (4, 8));
        let counts = r.dec.caches().counts().shapes;
        assert_eq!((counts.found, counts.again, counts.aged_again), (1, 1, 1));
        assert!(r.dec.entry(20).unwrap().shape.is_some());
    }

    /// The age rule at a use: with no new shape made, the first use of any shape a period after
    /// the rule's last pass takes out the shapes unused for the period, their bytes given back; a
    /// use within the period takes none out
    #[test]
    fn a_use_a_period_on_takes_out_the_aged_shapes() {
        let mut r = shaped();
        texture(&mut r, 20, 80, (32, 8));
        texture(&mut r, 21, 80, (4, 8));
        destroy(&mut r, 21);
        let row = 8 * 4;
        assert_eq!(r.dec.caches().live().cache_bytes, (32 + 4) * row);
        r.p.virtual_clock().advance(shapes::UNREFERENCED_FOR / 2);
        texture(&mut r, 22, 80, (32, 8));
        assert_eq!(r.dec.caches().counts().shapes.aged, 0, "within the period");
        r.p.virtual_clock().advance(shapes::UNREFERENCED_FOR);
        texture(&mut r, 23, 80, (32, 8));
        let counts = r.dec.caches().counts().shapes;
        assert_eq!((counts.made, counts.found), (2, 2), "no new shape");
        assert_eq!((counts.aged, counts.dropped), (1, 1));
        assert_eq!(counts.aged_bytes, 4 * row);
        assert_eq!(r.dec.caches().live().cache_bytes, 32 * row);
    }

    /// Past the caches' budget the shapes no texture references are trimmed least recently used
    /// first, each giving its bytes back as it goes, so the trims stop once the caches fit: of
    /// three unreferenced shapes, the two oldest go, and the newest stays, found by its next
    /// texture
    #[test]
    fn the_budget_trims_shapes_only_until_the_caches_fit() {
        let mut r = shaped();
        for (n, width) in [4, 5, 6].into_iter().enumerate() {
            let obj_id = 21 + n as u64;
            texture(&mut r, obj_id, 80, (width, 8));
            destroy(&mut r, obj_id);
        }
        // 8 rows of 4-byte texels each; the buffer holds no memory.
        let row = 8 * 4;
        assert_eq!(r.dec.caches().live().cache_bytes, (4 + 5 + 6) * row);
        r.dec.set_cache_budget(500);
        texture(&mut r, 30, 80, (7, 8));
        let counts = r.dec.caches().counts().shapes;
        assert_eq!(counts.dropped, 2, "the two oldest");
        assert_eq!(counts.dropped_bytes, (4 + 5) * row);
        assert_eq!(r.dec.caches().live().cache_bytes, (6 + 7) * row);
        texture(&mut r, 31, 80, (6, 8));
        let counts = r.dec.caches().counts().shapes;
        assert_eq!((counts.found, counts.again), (1, 0), "the newest kept");
    }

    /// A shape a command buffer writes is dirty from its finish: a download copies it into the
    /// cache's buffer, at its place there, before it copies the buffer into the pages (rule 3),
    /// the buffer holding the pages here, its memory made before the first fill ([`buffer_made`]),
    /// and once: the next download, nothing written since, writes nothing back. The surface reads
    /// back then, and the next writer's early copy into the pages writes the shape back first
    /// too. A release after a write writes it back before its copy
    #[test]
    fn a_download_or_release_writes_a_dirty_shape_back_before_its_copy() {
        let mut r = shaped();
        buffer_made(&r);
        let cache = cache_id(&r);
        let image = texture_image(&mut r, 21, 80, (32, 8));
        write(&mut r, 0x100, 21, (32, 8));
        r.p.world().complete_all(CompleteOn::Caller);
        let mark = r.p.call_log().mark();
        let downloaded = download(&r, BLOCK);
        let calls = r.p.call_log().since(mark);
        assert_eq!(written_back(&calls), [(image, cache, 0)]);
        let back = first(&calls, TO_BUFFER);
        assert!(copies(&calls[..back]).is_empty(), "written back first");
        assert_eq!(
            copies(&calls[back..]).len(),
            1,
            "then the buffer into the pages"
        );
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(downloaded.load(Ordering::SeqCst));
        let counts = r.dec.caches().counts().shapes;
        assert_eq!(
            (counts.written_back, counts.written_back_bytes),
            (1, 32 * 8 * 4)
        );

        let mark = r.p.call_log().mark();
        let again = download(&r, BLOCK);
        let calls = r.p.call_log().since(mark);
        assert!(written_back(&calls).is_empty(), "clean");
        assert!(copies(&calls).is_empty(), "nothing written");
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(again.load(Ordering::SeqCst));

        let mark = r.p.call_log().mark();
        write(&mut r, 0x101, 21, (32, 8));
        let calls = r.p.call_log().since(mark);
        assert_eq!(
            written_back(&calls),
            [(image, cache, 0)],
            "the early copy's"
        );
        let back = first(&calls, TO_BUFFER);
        assert!(copies(&calls[..back]).is_empty(), "written back first");
        assert_eq!(copies(&calls[back..]).len(), 1);
        r.p.world().complete_all(CompleteOn::Caller);
        let counts = r.dec.caches().counts();
        assert_eq!((counts.early, counts.shapes.written_back), (1, 2));

        let mut r = shaped();
        buffer_made(&r);
        let cache = cache_id(&r);
        let image = texture_image(&mut r, 21, 80, (32, 8));
        write(&mut r, 0x100, 21, (32, 8));
        r.p.world().complete_all(CompleteOn::Caller);
        let mark = r.p.call_log().mark();
        let released = release(&r, BLOCK);
        let calls = r.p.call_log().since(mark);
        assert_eq!(written_back(&calls), [(image, cache, 0)]);
        let back = first(&calls, TO_BUFFER);
        assert!(copies(&calls[..back]).is_empty(), "written back first");
        assert_eq!(copies(&calls[back..]).len(), 1);
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(released.load(Ordering::SeqCst));
        assert_eq!(r.dec.caches().counts().shapes.written_back, 1);
    }

    /// A buffer-to-buffer copy (`copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:`).
    const BUFFER_COPY: &str = "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:";

    /// In the per-shape form a cache's buffer is made deferred, its memory made at the first use
    /// that copies through it, once, before that copy: a texture over the cache, a view of its
    /// shape's image, makes none, nor does the cache's fill, which goes straight into the shape;
    /// the first copy through the buffer, a download's after a write, makes it, and nothing after
    /// makes it again
    #[test]
    fn a_shaped_cache_s_buffer_holds_its_memory_from_its_first_copy() {
        use reims_vgpu_vulkan::core::testkit::impls::command::FAIL_OUT_OF_MEMORY;
        use reims_vgpu_vulkan::core::testkit::impls::device::DEFERRED_BUFFER;
        use reims_vgpu_vulkan::core::testkit::impls::resource::MAKE_MEMORY;
        let mut r = shaped();
        let deferred = r.p.call_log().count(DEFERRED_BUFFER);
        assert_eq!(deferred, 1, "the cache's buffer, deferred");
        texture_image(&mut r, 21, 80, (32, 8));
        assert_eq!(
            r.p.call_log().count(MAKE_MEMORY),
            0,
            "no copy through it yet"
        );
        write(&mut r, 0x100, 21, (32, 8));
        r.p.world().complete_all(CompleteOn::Caller);
        assert_eq!(r.p.call_log().count(MAKE_MEMORY), 0, "filled straight");

        let mark = r.p.call_log().mark();
        let downloaded = download(&r, BLOCK);
        r.p.world().complete_all(CompleteOn::Caller);
        let calls = r.p.call_log().since(mark);
        let made = first(&calls, MAKE_MEMORY);
        assert!(
            !(calls[..made].iter()).any(|c| [BUFFER_COPY, FILL, TO_BUFFER].contains(&c.sel)),
            "made before the first copy"
        );
        assert!(!calls.iter().any(|c| c.sel == FAIL_OUT_OF_MEMORY));
        assert!(downloaded.load(Ordering::SeqCst));
        write(&mut r, 0x101, 21, (32, 8));
        r.p.world().complete_all(CompleteOn::Caller);
        let downloaded = download(&r, BLOCK);
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(downloaded.load(Ordering::SeqCst));
        assert_eq!(r.p.call_log().count(MAKE_MEMORY), 1, "made once");
    }

    /// The caches' budget counts a shaped cache's buffer from its memory's make, not from the
    /// cache's: a cache whose buffer holds no memory counts its shapes alone, a fill straight into
    /// them adds nothing, and the first copy through the buffer, a new shape's fill after a
    /// write, which makes it, adds its bytes, the most the caches held among them. The counts of
    /// what the caches hold tell the buffers' made bytes beside all of theirs, the cache's own
    /// among the caches holding the most, and, of the made bytes, those of buffers copied through
    /// once since, by what made them: the fill's, until a newer generation's fill goes through it
    /// again
    #[test]
    fn the_caches_budget_counts_a_buffer_from_its_memorys_make() {
        let mut r = shaped();
        let page = r.page;
        let shape = 32 * 8 * 4;
        texture_image(&mut r, 21, 80, (32, 8));
        let live = r.dec.caches().live();
        assert_eq!((live.buffer_bytes, live.buffer_made_bytes), (2 * page, 0));
        assert_eq!(live.cache_bytes, shape, "the shape alone");
        assert_eq!(live.largest[0].held, shape);
        write(&mut r, 0x100, 21, (32, 8));
        r.p.world().complete_all(CompleteOn::Caller);
        let live = r.dec.caches().live();
        let straight = (live.buffer_made_bytes, live.cache_bytes);
        assert_eq!(straight, (0, shape), "filled straight");

        let shapes = shape + 16 * 4 * 4;
        texture_image(&mut r, 22, 80, (16, 4));
        use_surfaces(&mut r, 1, 0x101, &[(BLOCK, 0, false)]);
        r.p.world().complete_all(CompleteOn::Caller);
        let live = r.dec.caches().live();
        assert_eq!(
            live.buffer_made_bytes,
            2 * page,
            "the new shape's fill made it"
        );
        assert_eq!(live.cache_bytes, 2 * page + shapes);
        let largest = live.largest[0];
        assert_eq!(
            (largest.held, largest.buffer_made_bytes),
            (2 * page + shapes, 2 * page)
        );
        let once = (
            live.once_fill_bytes,
            live.once_download_bytes,
            live.once_other_bytes,
        );
        assert_eq!(
            once,
            (2 * page, 0, 0),
            "copied through once, made by its fill"
        );
        assert_eq!(r.dec.caches().counts().held_most, 2 * page + shapes);

        use_surfaces(&mut r, 1, 0x102, &[(BLOCK, 3, false)]);
        r.p.world().complete_all(CompleteOn::Caller);
        let live = r.dec.caches().live();
        assert_eq!(live.buffer_made_bytes, 2 * page);
        assert_eq!(live.once_fill_bytes, 0, "copied through twice");
    }

    /// A buffer whose memory a download made counts, once copied through, as the download's; one
    /// a command of the guest's made, which names the buffer itself, counts made with no copy of
    /// the caches' own through it, and is not among those copied through once
    #[test]
    fn the_buffers_copied_through_once_are_told_by_what_made_them() {
        use reims_vgpu_graphics::wire::{RGFX_SURFACE_PAGES_ATTACH, RGFX_SURFACE_PAGES_CACHED};
        let (mut r, _) = placing_deferred();
        let page = r.page;
        compute_pipeline(&mut r, 31);
        writable_image(&mut r, 25);
        computes(&mut r, 0x100, &[bind_pipeline(31), bind_texture(25, 0)]);
        r.p.world().complete_all(CompleteOn::Caller);
        assert_eq!(r.dec.caches().live().buffer_made_bytes, 0, "zeros taken");
        let flags = RGFX_SURFACE_PAGES_CACHED | RGFX_SURFACE_PAGES_ATTACH;
        let attached = r.pages_flagged(BLOCK, 7, &[(1, 2)], 2 * page, flags);
        assert_eq!(attached, RGFX_STATUS_OK);
        let mark = r.p.call_log().mark();
        let downloaded = download(&r, BLOCK);
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(downloaded.load(Ordering::SeqCst));
        assert_eq!(
            written_back(&r.p.call_log().since(mark)).len(),
            1,
            "its write"
        );
        let live = r.dec.caches().live();
        assert_eq!(live.buffer_made_bytes, 2 * page);
        let once = (
            live.once_fill_bytes,
            live.once_download_bytes,
            live.once_other_bytes,
        );
        assert_eq!(once, (0, 2 * page, 0), "the download's");

        let mut r = shaped();
        texture_image(&mut r, 21, 80, (32, 8));
        write_from(&mut r, 0x100, 21, (32, 8), 10);
        r.p.world().complete_all(CompleteOn::Caller);
        let live = r.dec.caches().live();
        assert_eq!(live.buffer_made_bytes, 2 * r.page);
        let once = (
            live.once_fill_bytes,
            live.once_download_bytes,
            live.once_other_bytes,
        );
        assert_eq!(once, (0, 0, 0), "named by the guest's blit");
    }

    /// Where the platform refuses a shaped cache's buffer its memory, nothing is copied through
    /// the buffer, and the command buffer whose begin's fill found it refused fails with Metal's
    /// `MTLCommandBufferErrorOutOfMemory`, once: here a new shape's fill over a dirty shape, as
    /// the cache's first fill went straight into the shapes and asked for no memory. A command
    /// that names the buffer is dropped so too
    /// ([`a_refused_cache_buffer_drops_the_command_naming_it_and_copies_nothing`])
    #[test]
    fn a_shaped_cache_s_refused_memory_fails_its_command_buffer_out_of_memory() {
        use reims_vgpu_vulkan::core::testkit::impls::command::FAIL_OUT_OF_MEMORY;
        use reims_vgpu_vulkan::core::testkit::impls::resource::MAKE_MEMORY;
        let mut r = shaped();
        let cache = cache_id(&r);
        texture_image(&mut r, 21, 80, (32, 8));
        r.p.script().always(MAKE_MEMORY, Step::value(false));
        write(&mut r, 0x100, 21, (32, 8));
        r.p.world().complete_all(CompleteOn::Caller);
        assert_eq!(r.p.call_log().count(MAKE_MEMORY), 0, "filled straight");
        assert_eq!(r.p.call_log().count(FAIL_OUT_OF_MEMORY), 0);
        texture_image(&mut r, 22, 80, (16, 4));
        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 1, 0x101, &[(BLOCK, 0, false)]);
        r.p.world().complete_all(CompleteOn::Caller);
        let calls = r.p.call_log().since(mark);
        assert!(calls.iter().any(|c| c.sel == MAKE_MEMORY), "asked");
        assert_eq!(
            calls.iter().filter(|c| c.sel == FAIL_OUT_OF_MEMORY).count(),
            1
        );
        let id = |a: &Arg| a.as_obj().map_or(0, |o| o.0);
        let through = |c: &&Call| {
            [BUFFER_COPY, FILL, TO_BUFFER].contains(&c.sel) && c.args.iter().any(|a| id(a) == cache)
        };
        assert_eq!(
            calls.iter().find(through),
            None,
            "nothing through the buffer"
        );
    }

    /// Where the platform refuses a shaped cache's buffer its memory, a guest's command that names
    /// the buffer (a blit's fill of buffer 10, over the cache) is dropped, and its command buffer
    /// fails with Metal's `MTLCommandBufferErrorOutOfMemory`, once; a download of the cache then
    /// copies nothing through the buffer and is done, as where the platform makes no command
    /// buffer for it, and nothing ends the process
    #[test]
    fn a_refused_cache_buffer_drops_the_command_naming_it_and_copies_nothing() {
        use reims_vgpu_graphics::ops::blit::{RGFX_OP_BLIT_FILL_BUFFER, RgfxBlitOpFillBuffer};
        use reims_vgpu_vulkan::core::testkit::impls::command::FAIL_OUT_OF_MEMORY;
        use reims_vgpu_vulkan::core::testkit::impls::resource::MAKE_MEMORY;
        use zerocopy::IntoBytes as _;
        let mut r = shaped();
        let cache = cache_id(&r);
        r.p.script().always(MAKE_MEMORY, Step::value(false));
        let fill = RgfxBlitOpFillBuffer {
            buffer_obj_id: 10,
            range_location: 0,
            range_length: 64,
            value: 7,
            ..RgfxBlitOpFillBuffer::default()
        };
        let mark = r.p.call_log().mark();
        blit_with(
            &mut r,
            0x100,
            true,
            (RGFX_OP_BLIT_FILL_BUFFER, fill.as_bytes().to_vec()),
        );
        r.p.world().complete_all(CompleteOn::Caller);
        let calls = r.p.call_log().since(mark);
        assert!(calls.iter().any(|c| c.sel == MAKE_MEMORY), "asked");
        let failed = calls.iter().filter(|c| c.sel == FAIL_OUT_OF_MEMORY).count();
        assert_eq!(failed, 1);
        assert!(
            !calls.iter().any(|c| c.sel.starts_with("fillBuffer:")),
            "dropped"
        );

        let id = |a: &Arg| a.as_obj().map_or(0, |o| o.0);
        let through = |c: &&Call| {
            [BUFFER_COPY, FILL, TO_BUFFER].contains(&c.sel) && c.args.iter().any(|a| id(a) == cache)
        };
        let mark = r.p.call_log().mark();
        let downloaded = download(&r, BLOCK);
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(downloaded.load(Ordering::SeqCst), "done");
        let calls = r.p.call_log().since(mark);
        assert_eq!(
            calls.iter().find(through),
            None,
            "nothing through the buffer"
        );
        assert!(!calls.iter().any(|c| c.sel == FAIL_OUT_OF_MEMORY));
    }

    /// Where the platform refuses a shaped cache's buffer its memory, a write through one shape
    /// whose copy into the other shapes of its plane goes through the buffer (rule 1) fails its
    /// command buffer with `MTLCommandBufferErrorOutOfMemory`, once, and the other shape takes
    /// nothing: the cache's first use took its shapes' zeros, its buffer's memory left unmade
    /// ([`placing_deferred`]), so the write's copy is its first use
    #[test]
    fn a_refused_cache_buffer_fails_a_writes_copy_into_its_planes_other_shapes() {
        use reims_vgpu_vulkan::core::testkit::impls::command::FAIL_OUT_OF_MEMORY;
        use reims_vgpu_vulkan::core::testkit::impls::resource::MAKE_MEMORY;
        let (mut r, _) = placing_deferred();
        let a = texture_image(&mut r, 21, 80, (32, 8));
        let b = texture_image(&mut r, 22, 80, (16, 4));
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 0, false)]);
        r.p.world().complete_all(CompleteOn::Caller);
        r.p.script().always(MAKE_MEMORY, Step::value(false));
        let mark = r.p.call_log().mark();
        write(&mut r, 0x101, 21, (32, 8));
        r.p.world().complete_all(CompleteOn::Caller);
        let calls = r.p.call_log().since(mark);
        assert!(calls.iter().any(|c| c.sel == MAKE_MEMORY), "asked");
        let (_, into) = overlaps(&calls, &[a, b]);
        assert!(into.is_empty(), "B takes nothing");
        let failed = calls.iter().filter(|c| c.sel == FAIL_OUT_OF_MEMORY).count();
        assert_eq!(failed, 1);
    }

    /// A new shape is filled from the buffer after the dirty shapes are written back into it, in
    /// one command buffer of the transfer queue, so it takes their writes (rule 2); the buffer,
    /// which the cache's first fill went past straight into the shapes, takes the pages before
    /// the writes go over them, its memory made then, counted. The shape written back is clean,
    /// so a download then copies the buffer, the write in it, into the pages alone
    #[test]
    fn a_new_shape_takes_a_dirty_shapes_writes() {
        use reims_vgpu_vulkan::core::testkit::impls::resource::MAKE_MEMORY;
        let mut r = shaped();
        let (cache, pages) = (cache_id(&r), pages_id(&r));
        let written = texture_image(&mut r, 21, 80, (32, 8));
        write(&mut r, 0x100, 21, (32, 8));
        r.p.world().complete_all(CompleteOn::Caller);
        assert_eq!(r.p.call_log().count(MAKE_MEMORY), 0, "filled straight");
        let new = texture_image(&mut r, 22, 80, (16, 4));
        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 1, 0x101, &[(BLOCK, 0, false)]);
        let calls = r.p.call_log().since(mark);
        assert_eq!(
            copies(&calls),
            [(pages, cache)],
            "the pages into the buffer"
        );
        assert_eq!(written_back(&calls), [(written, cache, 0)]);
        assert_eq!(fills(&calls), [(cache, 0, new)]);
        let refill = first(&calls, BUFFER_COPY);
        assert!(first(&calls, MAKE_MEMORY) < refill, "made first");
        assert!(
            refill < first(&calls, TO_BUFFER),
            "the write over the pages"
        );
        assert!(
            first(&calls, TO_BUFFER) < first(&calls, FILL),
            "written back first"
        );
        let counts = r.dec.caches().counts();
        assert_eq!(
            (counts.shapes.fills_made, counts.shapes.written_back),
            (1, 1)
        );
        assert_eq!((counts.buffer_refills, counts.buffer_refills_named), (1, 0));
        r.p.world().complete_all(CompleteOn::Caller);

        let mark = r.p.call_log().mark();
        let downloaded = download(&r, BLOCK);
        let calls = r.p.call_log().since(mark);
        assert!(written_back(&calls).is_empty(), "clean");
        assert_eq!(
            copies(&calls),
            [(cache, pages)],
            "the buffer into the pages"
        );
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(downloaded.load(Ordering::SeqCst));
    }

    /// Straight fills: a fill of a cache whose buffer holds no memory copies each shape's bytes
    /// straight from the view of the pages, one copy a shape, none into the buffer, and makes no
    /// memory, counted; a surface filled so and never copied through has made no buffer: a
    /// download with nothing written copies nothing, and a newer generation's fill goes straight
    /// again
    #[test]
    fn a_surface_filled_once_makes_no_buffer_and_its_shapes_take_the_pages_bytes() {
        use reims_vgpu_vulkan::core::testkit::impls::resource::MAKE_MEMORY;
        let mut r = shaped();
        let pages = pages_id(&r);
        let a = texture_image(&mut r, 21, 80, (32, 8));
        let b = texture_image(&mut r, 22, 80, (16, 4));
        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 2, false)]);
        r.p.world().complete_all(CompleteOn::Caller);
        let calls = r.p.call_log().since(mark);
        assert!(copies(&calls).is_empty(), "none into the buffer");
        assert_eq!(fills(&calls), [(pages, 0, a), (pages, 0, b)]);
        assert_eq!(r.p.call_log().count(MAKE_MEMORY), 0, "no memory made");
        assert_eq!(r.dec.caches().live().buffer_made_bytes, 0);
        let counts = r.dec.caches().counts();
        assert_eq!((counts.straight, counts.buffer_refills), (1, 0));
        let sh = counts.shapes;
        let straight = (
            sh.straight_fills,
            sh.straight_copies,
            sh.straight_split_rows,
        );
        assert_eq!(straight, (2, 2, 0));
        assert_eq!(sh.straight_bytes, 32 * 8 * 4 + 16 * 4 * 4);
        assert_eq!(sh.fills, 2);

        let mark = r.p.call_log().mark();
        let downloaded = download(&r, BLOCK);
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(downloaded.load(Ordering::SeqCst));
        assert!(
            copies(&r.p.call_log().since(mark)).is_empty(),
            "nothing written"
        );

        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 1, 0x101, &[(BLOCK, 3, false)]);
        r.p.world().complete_all(CompleteOn::Caller);
        let calls = r.p.call_log().since(mark);
        assert_eq!(
            fills(&calls),
            [(pages, 0, a), (pages, 0, b)],
            "straight again"
        );
        assert!(copies(&calls).is_empty());
        assert_eq!(r.p.call_log().count(MAKE_MEMORY), 0);
        assert_eq!(r.dec.caches().counts().straight, 2);
    }

    /// A download or a release of a cache whose fill went straight into its shapes, a shape
    /// written since, makes the buffer's memory and copies the pages into it first, then the
    /// shape's write over them, then the buffer into the pages, counted with the pages' bytes:
    /// the bytes beside the shape are the pages' own, not the zeros of the memory's make. Once
    /// the buffer holds the pages, the next copy through it, the writer's early copy behind the
    /// download, takes none
    #[test]
    fn a_copy_through_a_buffer_its_fill_went_past_takes_the_pages_first() {
        use reims_vgpu_vulkan::core::testkit::impls::resource::MAKE_MEMORY;
        let written = || {
            let mut r = shaped();
            let image = texture_image(&mut r, 21, 80, (32, 8));
            write(&mut r, 0x100, 21, (32, 8));
            r.p.world().complete_all(CompleteOn::Caller);
            assert_eq!(r.p.call_log().count(MAKE_MEMORY), 0, "filled straight");
            let ids = (cache_id(&r), pages_id(&r), image);
            (r, ids)
        };
        let takes_first = |calls: &[Call], (cache, pages, image): (u64, u64, u64)| {
            assert_eq!(copies(calls), [(pages, cache), (cache, pages)]);
            assert_eq!(written_back(calls), [(image, cache, 0)]);
            let (made, refill) = (first(calls, MAKE_MEMORY), first(calls, BUFFER_COPY));
            assert!(made < refill, "made first");
            assert!(
                refill < first(calls, TO_BUFFER),
                "the pages, then the write"
            );
        };

        let (mut r, ids) = written();
        let mark = r.p.call_log().mark();
        let downloaded = download(&r, BLOCK);
        takes_first(&r.p.call_log().since(mark), ids);
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(downloaded.load(Ordering::SeqCst));
        let counts = r.dec.caches().counts();
        let refilled = (counts.buffer_refills, counts.buffer_refilled_bytes);
        assert_eq!(refilled, (1, 2 * r.page));
        let mark = r.p.call_log().mark();
        write(&mut r, 0x101, 21, (32, 8));
        let calls = r.p.call_log().since(mark);
        let (cache, pages, image) = ids;
        assert_eq!(copies(&calls), [(cache, pages)], "the early copy, alone");
        assert_eq!(written_back(&calls), [(image, cache, 0)]);
        r.p.world().complete_all(CompleteOn::Caller);
        let counts = r.dec.caches().counts();
        assert_eq!((counts.early, counts.buffer_refills), (1, 1));
        assert_eq!(r.p.call_log().count(MAKE_MEMORY), 1, "made once");

        let (r, ids) = written();
        let mark = r.p.call_log().mark();
        let released = release(&r, BLOCK);
        takes_first(&r.p.call_log().since(mark), ids);
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(released.load(Ordering::SeqCst));
        assert_eq!(r.dec.caches().counts().buffer_refills, 1);
    }

    /// A command that names a cache's buffer while it lacks the pages its shapes took, its
    /// memory not made, reads the zeros of the memory's make, which the command's use makes:
    /// nothing copies the pages into it first. It is counted (`rt-shape-lacking-named`), as the
    /// guest's bundle names no cache's buffer in a command; the buffer lacks the pages still, so
    /// the next copy through it, a download's, takes them first, over what the command wrote
    #[test]
    fn a_command_naming_a_buffer_that_lacks_the_pages_reads_its_makes_zeros() {
        use reims_vgpu_vulkan::core::testkit::impls::resource::MAKE_MEMORY;
        let mut r = shaped();
        let (cache, pages) = (cache_id(&r), pages_id(&r));
        let image = texture_image(&mut r, 21, 80, (32, 8));
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 0, false)]);
        r.p.world().complete_all(CompleteOn::Caller);
        assert_eq!(r.p.call_log().count(MAKE_MEMORY), 0, "filled straight");
        let was = counted(shapes::LACKING_NAMED);
        let mark = r.p.call_log().mark();
        write_from(&mut r, 0x101, 21, (32, 8), 10);
        r.p.world().complete_all(CompleteOn::Caller);
        let calls = r.p.call_log().since(mark);
        assert_eq!(r.p.call_log().count(MAKE_MEMORY), 1, "the command's");
        assert!(
            copies(&calls).is_empty(),
            "nothing into it: the make's zeros"
        );
        let id = |a: &Arg| a.as_obj().map_or(0, |o| o.0);
        let read = (calls.iter()).position(|c| c.sel == FILL && id(&c.args[0]) == cache);
        assert!(
            read.is_some_and(|at| first(&calls, MAKE_MEMORY) < at),
            "read after its make"
        );
        assert_eq!(r.dec.caches().counts().lacking_named, 1);
        assert_eq!(counted(shapes::LACKING_NAMED) - was, 1);

        let mark = r.p.call_log().mark();
        let downloaded = download(&r, BLOCK);
        let calls = r.p.call_log().since(mark);
        assert_eq!(
            copies(&calls),
            [(pages, cache), (cache, pages)],
            "the pages first"
        );
        assert_eq!(written_back(&calls), [(image, cache, 0)]);
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(downloaded.load(Ordering::SeqCst));
        assert_eq!(r.dec.caches().counts().buffer_refills, 1);
    }

    /// A texture over a cache's buffer itself (`TEXTURE_FROM_BUFFER` of buffer 10) made while
    /// the buffer lacks the pages its shapes took makes the buffer's memory and copies the pages
    /// into it, so the texture reads the surface, counted among the buffers that took them for a
    /// texture or a video op; no command then reads the make's zeros
    #[test]
    fn a_texture_over_a_buffer_that_lacks_the_pages_takes_them_first() {
        use reims_vgpu_graphics::wire::{
            RGFX_CMD_TEXTURE_FROM_BUFFER, RgfxMsgHeader, RgfxTextureDescriptor,
            RgfxTextureFromBuffer,
        };
        use reims_vgpu_vulkan::core::testkit::impls::resource::MAKE_MEMORY;
        use zerocopy::IntoBytes as _;
        let mut r = shaped();
        let (cache, pages) = (cache_id(&r), pages_id(&r));
        texture_image(&mut r, 21, 80, (32, 8));
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 0, false)]);
        r.p.world().complete_all(CompleteOn::Caller);
        assert_eq!(r.p.call_log().count(MAKE_MEMORY), 0, "filled straight");
        let m = RgfxTextureFromBuffer {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_TEXTURE_FROM_BUFFER,
                size: size_of::<RgfxTextureFromBuffer>() as u32,
                seqno: 0,
            },
            obj_id: 30,
            resource_id: 0x41,
            buffer_obj_id: 10,
            offset: 0,
            bytes_per_row: 64 * 4,
            desc: RgfxTextureDescriptor {
                pixel_format: 80,
                texture_type: 2,
                width: 64,
                height: 16,
                depth: 1,
                mipmap_level_count: 1,
                array_length: 1,
                sample_count: 1,
                usage: 1,
                ..RgfxTextureDescriptor::default()
            },
        };
        let mark = r.p.call_log().mark();
        assert_eq!(r.dec.decode(&r.device, m.as_bytes()).unwrap(), Ok(()));
        let calls = r.p.call_log().since(mark);
        assert_eq!(
            copies(&calls),
            [(pages, cache)],
            "the pages into the buffer"
        );
        assert!(
            first(&calls, MAKE_MEMORY) < first(&calls, BUFFER_COPY),
            "made first"
        );
        r.p.world().complete_all(CompleteOn::Caller);
        let counts = r.dec.caches().counts();
        assert_eq!((counts.buffer_refills, counts.buffer_refills_named), (1, 1));
        assert_eq!(counts.lacking_named, 0);
    }

    /// Where the caches' copies are staged, a fill of a shape asked while a download of its
    /// cache circulates its pieces (`surface_cache`'s `Rounds`) waits on the GPU for the copy's
    /// own event at its last round's drain, which the host sets only once the pages hold every
    /// round, so the shapes' writes the fill takes back into the cache first reach none of the
    /// download's rounds; the download is answered as its last round lands
    #[test]
    fn a_shapes_fill_asked_while_a_download_circulates_waits_for_its_last_round() {
        const WAIT: &str = "encodeWaitForEvent:value:";
        const SIGNAL: &str = "encodeSignalEvent:value:";
        // The value of the copy's own event once its last round's bytes are in the pages.
        const DRAINED: u64 = 2;
        // As `shaped_staged`, in pieces of half a page; a blit over the cache's buffer makes the
        // download whole, four pieces' length, its memory made first so the blit reads the pages
        // ([`buffer_made`]).
        let mut r = rig(8);
        r.dec.set_staged_surface_copies();
        let page = r.page;
        r.dec.surface_caches.set_staging_piece(page / 2);
        take_shapes(&r);
        r.dec.hold_resource_ids(0, 1);
        assert_eq!(
            r.pages_cached(BLOCK, &[(1, 2)], 2 * page, true),
            RGFX_STATUS_OK
        );
        queues(&mut r);
        assert_eq!(r.buffer(10, BLOCK, 2 * page), Ok(()));
        let cache = cache_id(&r);
        writable_at(&mut r, 25, 0);
        buffer_made(&r);
        write_from(&mut r, 0x100, 25, (32, 8), 10);
        complete_staged(&r);

        let mark = r.p.call_log().mark();
        let done = download(&r, BLOCK);
        let rounds = r.p.world().committed();
        assert_eq!(rounds.len(), 2, "the download's first two rounds");
        // A shape of its own, not texture 25's, which owes its fill.
        let image = texture_image(&mut r, 21, 80, (16, 4));
        let own = (r.p.call_log().since(mark).iter())
            .find(|c| c.obj == rounds[1] && c.sel == SIGNAL)
            .and_then(|c| c.args[0].as_obj())
            .expect("the copy's own event");
        let held = r.p.world().find(own).expect("the copy's event");
        let own_now = || reims_vgpu_vulkan::core::testkit::objects::event_value(held.obj());

        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 2, 0x101, &[(BLOCK, 0, false)]);
        let calls = r.p.call_log().since(mark);
        assert_eq!(fills(&calls), [(cache, 0, image)], "the shape's fill");
        assert!(
            (calls.iter()).any(|c| c.sel == WAIT
                && c.args[0].as_obj() == Some(own)
                && c.args[1].as_u64() == Some(DRAINED)),
            "the fill waits for the download's last round"
        );
        let mut ended = 0;
        while let Some(&cb) = r.p.world().committed().first() {
            assert!(own_now() < DRAINED, "the fill still waits");
            complete_round(&r, cb);
            ended += 1;
            if done.load(Ordering::SeqCst) {
                break;
            }
        }
        assert!(ended > 2, "the download's rounds");
        assert_eq!(own_now(), DRAINED, "set once the last round landed");
        complete_staged(&r);
        assert!(done.load(Ordering::SeqCst));
    }

    /// A new shape's fill asked while a command buffer that uses the cache is begun and not
    /// finished is made at that buffer's finish, behind it, and writes back what it wrote first
    /// (rule 2), a reader asked meanwhile waiting for it on the GPU. A writer counted after the
    /// fill was asked, whose write follows the fill on the GPU, leaves its shape dirty past the
    /// fill, though it finished first: a download writes it back
    #[test]
    fn a_shapes_fill_behind_a_writer_takes_its_write_alone() {
        let mut r = shaped();
        let cache = cache_id(&r);
        let written = texture_image(&mut r, 21, 80, (32, 8));
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 0, false)]);
        r.p.world().complete_all(CompleteOn::Caller);
        let writes = [RgfxOpSurfaceUse {
            flags: RGFX_SURFACE_USE_WRITES,
            ..reading(BLOCK, 0)[0]
        }];
        let shape = |r: &Rig| {
            let entry = r.dec.entry(21).unwrap();
            (BLOCK, entry.shape.as_ref().unwrap().reference.clone_ref())
        };
        let (queue, other) = (
            r.device.new_command_queue().unwrap(),
            r.device.new_command_queue().unwrap(),
        );
        let writer = queue.command_buffer(ON).unwrap().unwrap();
        let (_, writer_uses) = (r.dec.surface_caches)
            .before(&r.device, &writer, &writes, &NO_REACH)
            .unwrap();
        let mut writer_uses = writer_uses.unwrap();
        writer_uses.saw_shapes_written([shape(&r)]);

        let new = texture_image(&mut r, 22, 80, (16, 4));
        let mark = r.p.call_log().mark();
        let reader = queue.command_buffer(ON).unwrap().unwrap();
        let (waits, reader_uses) = (r.dec.surface_caches)
            .before(&r.device, &reader, &reading(BLOCK, 0), &NO_REACH)
            .unwrap();
        assert!(waits, "the reader waits for the fill on the GPU");
        let calls = r.p.call_log().since(mark);
        assert!(
            written_back(&calls).is_empty() && fills(&calls).is_empty(),
            "deferred"
        );
        assert_eq!(r.dec.caches().counts().shapes.fills_deferred, 1);

        let later = other.command_buffer(ON).unwrap().unwrap();
        let (_, later_uses) = (r.dec.surface_caches)
            .before(&r.device, &later, &writes, &NO_REACH)
            .unwrap();
        let mut later_uses = later_uses.unwrap();
        later_uses.saw_shapes_written([shape(&r)]);
        (r.dec.surface_caches)
            .finish(&r.device, 2, None, &later, later_uses, || later.commit(ON))
            .unwrap();
        let mark = r.p.call_log().mark();
        (r.dec.surface_caches)
            .finish(&r.device, 1, None, &writer, writer_uses, || {
                writer.commit(ON)
            })
            .unwrap();
        let calls = r.p.call_log().since(mark);
        assert_eq!(
            written_back(&calls),
            [(written, cache, 0)],
            "the first's write"
        );
        assert_eq!(fills(&calls), [(cache, 0, new)]);
        assert!(
            first(&calls, TO_BUFFER) < first(&calls, FILL),
            "written back first"
        );
        drop(reader_uses);
        r.p.world().complete_all(CompleteOn::Caller);

        let mark = r.p.call_log().mark();
        let downloaded = download(&r, BLOCK);
        let calls = r.p.call_log().since(mark);
        assert_eq!(
            written_back(&calls),
            [(written, cache, 0)],
            "the later's write"
        );
        assert_eq!(copies(&calls).len(), 1);
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(downloaded.load(Ordering::SeqCst));
        let counts = r.dec.caches().counts().shapes;
        let made = (
            counts.fills_deferred,
            counts.fills_made,
            counts.written_back,
        );
        assert_eq!(made, (1, 1, 2));
    }

    /// A write through one shape reaches the other shapes of its plane at the end of the encoder
    /// that wrote it, in the same command buffer (rule 1): the writer's encoder updates the
    /// writers' fence, and an encoder that waits for it copies the texels the other shapes share
    /// with the written one into the cache's buffer, then each other shape takes its share from
    /// there; the written shape alone is dirty, so a download writes it alone back. An unreferenced
    /// shape that holds no write of its own is dropped instead, and a plane with one shape copies
    /// nothing
    #[test]
    fn a_write_through_one_shape_reaches_its_planes_other_shapes() {
        let mut r = shaped();
        let cache = cache_id(&r);
        let a = texture_image(&mut r, 21, 80, (32, 8));
        let b = texture_image(&mut r, 22, 80, (16, 4));
        let gone = texture_image(&mut r, 23, 80, (8, 2));
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 0, false)]);
        destroy(&mut r, 23);
        r.p.world().complete_all(CompleteOn::Caller);

        let mark = r.p.call_log().mark();
        write(&mut r, 0x101, 21, (32, 8));
        let calls = r.p.call_log().since(mark);
        let (out, into) = overlaps(&calls, &[a, b, gone]);
        assert_eq!(out, [(a, texels(16, 4))], "what B shares with A");
        assert_eq!(into, [(b, texels(16, 4))], "the dropped one takes nothing");
        let update = first(&calls, "updateFence:");
        let wait = first(&calls, "waitForFence:");
        assert!(update < wait && wait < first(&calls, TO_BUFFER));
        assert_eq!(written_back(&calls)[0].1, cache);
        let counts = r.dec.caches().counts().shapes;
        let rule_1 = (
            counts.overlap_copies,
            counts.overlap_bytes,
            counts.overlap_dropped,
        );
        assert_eq!(rule_1, (1, 16 * 4 * 4, 1));
        assert_eq!((counts.dropped, counts.unordered), (1, 0));
        r.p.world().complete_all(CompleteOn::Caller);

        let mark = r.p.call_log().mark();
        let downloaded = download(&r, BLOCK);
        let calls = r.p.call_log().since(mark);
        assert_eq!(written_back(&calls), [(a, cache, 0)], "A alone is dirty");
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(downloaded.load(Ordering::SeqCst));

        // B's write reaches A, its texels at its own size; B alone is dirty then, so the early
        // copy that follows the download's write writes B alone back.
        let mark = r.p.call_log().mark();
        write(&mut r, 0x102, 22, (16, 4));
        let calls = r.p.call_log().since(mark);
        let (out, into) = overlaps(&calls, &[a, b]);
        assert_eq!(into, [(a, texels(16, 4))]);
        assert_eq!(
            out,
            [(b, texels(16, 4)), (b, texels(16, 4))],
            "rule 1's, then the early copy's"
        );
        assert_eq!(written_back(&calls), [(b, cache, 0), (b, cache, 0)]);
        r.p.world().complete_all(CompleteOn::Caller);
        assert_eq!(r.dec.caches().counts().early, 1);

        // B unreferenced and clean: A's next write drops it, with no fence and no copy into a
        // shape; the early copy writes A back.
        destroy(&mut r, 22);
        let mark = r.p.call_log().mark();
        write(&mut r, 0x103, 21, (32, 8));
        let calls = r.p.call_log().since(mark);
        assert!(calls.iter().all(|c| c.sel != "updateFence:"), "no fence");
        let (out, into) = overlaps(&calls, &[a, b]);
        assert!(into.is_empty());
        assert!(
            out.iter()
                .all(|(image, size)| (*image, size) == (a, &texels(32, 8)))
        );
        assert_eq!(r.dec.caches().counts().shapes.overlap_dropped, 2, "B too");
    }

    /// An unreferenced shape that holds a write of its own is not dropped at another shape's
    /// write: it takes the write, so the two agree where they overlap whichever is written back
    /// first
    #[test]
    fn an_unreferenced_dirty_shape_takes_another_shapes_write() {
        let mut r = shaped();
        let a = texture_image(&mut r, 21, 80, (32, 8));
        let b = texture_image(&mut r, 22, 80, (16, 4));
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 0, false)]);
        r.p.world().complete_all(CompleteOn::Caller);
        write(&mut r, 0x101, 22, (16, 4));
        r.p.world().complete_all(CompleteOn::Caller);
        destroy(&mut r, 22);
        let mark = r.p.call_log().mark();
        write(&mut r, 0x102, 21, (32, 8));
        let (out, into) = overlaps(&r.p.call_log().since(mark), &[a, b]);
        assert_eq!(
            (out, into),
            (vec![(a, texels(16, 4))], vec![(b, texels(16, 4))])
        );
        let counts = r.dec.caches().counts().shapes;
        assert_eq!((counts.overlap_dropped, counts.dropped), (0, 0));
    }

    /// An encoder that may write two shapes of one plane is refused, counted: the shape the walk
    /// noted last is taken as the plane's write, and the other takes its texels
    #[test]
    fn two_shapes_of_one_plane_written_in_one_encoder_are_refused() {
        let mut r = shaped();
        let a = texture_image(&mut r, 21, 80, (32, 8));
        let b = texture_image(&mut r, 22, 80, (16, 4));
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 0, false)]);
        r.p.world().complete_all(CompleteOn::Caller);
        source(&mut r);
        let mark = r.p.call_log().mark();
        let ops = [write_op(22, (16, 4)), write_op(21, (32, 8))];
        blits_with(&mut r, 0x101, true, &ops);
        let (out, into) = overlaps(&r.p.call_log().since(mark), &[a, b]);
        assert_eq!(
            (out, into),
            (vec![(a, texels(16, 4))], vec![(b, texels(16, 4))])
        );
        assert_eq!(r.dec.caches().counts().shapes.unordered, 1);
        r.p.world().complete_all(CompleteOn::Caller);
        let mark = r.p.call_log().mark();
        let downloaded = download(&r, BLOCK);
        let back = written_back(&r.p.call_log().since(mark));
        assert_eq!(back.len(), 2, "both written");
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(downloaded.load(Ordering::SeqCst));
    }

    /// A command buffer whose work is video alone reads and writes the buffer, not the shapes'
    /// images: a dirty shape is copied into the buffer before it runs, and after it, where it may
    /// write the cache, every shape owes a fill, which the next command buffer that names the
    /// block makes, counted. A command buffer of other work that writes the cache unseen leaves
    /// the shapes as they were, counted.
    #[test]
    fn a_video_buffer_has_shapes_written_back_before_it_and_filled_after() {
        use super::super::surface_cache::Reach;
        let mut r = shaped();
        let cache = cache_id(&r);
        let image = texture_image(&mut r, 21, 80, (32, 8));
        write(&mut r, 0x100, 21, (32, 8));
        r.p.world().complete_all(CompleteOn::Caller);
        let writes = [RgfxOpSurfaceUse {
            flags: RGFX_SURFACE_USE_WRITES,
            ..reading(BLOCK, 0)[0]
        }];
        let queue = r.device.new_command_queue().unwrap();
        let run = |r: &mut Rig, reach: &Reach| {
            let cb = queue.command_buffer(ON).unwrap().unwrap();
            let mark = r.p.call_log().mark();
            let (_, uses) = (r.dec.surface_caches)
                .before(&r.device, &cb, &writes, reach)
                .unwrap();
            let calls = r.p.call_log().since(mark);
            (r.dec.surface_caches)
                .finish(&r.device, 1, None, &cb, uses.unwrap(), || cb.commit(ON))
                .unwrap();
            r.p.world().complete_all(CompleteOn::Caller);
            calls
        };
        let video = Reach {
            video: true,
            ..Reach::default()
        };
        let calls = run(&mut r, &video);
        assert_eq!(written_back(&calls), [(image, cache, 0)], "before it");
        assert!(fills(&calls).is_empty());
        let counts = r.dec.caches().counts().shapes;
        assert_eq!((counts.video_owed, counts.unseen), (1, 0));
        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 1, 0x101, &[(BLOCK, 0, false)]);
        let calls = r.p.call_log().since(mark);
        assert_eq!(
            fills(&calls),
            [(cache, 0, image)],
            "the picture into the shape"
        );
        assert!(written_back(&calls).is_empty());
        r.p.world().complete_all(CompleteOn::Caller);

        let work = Reach {
            work: true,
            ..video
        };
        let calls = run(&mut r, &work);
        assert!(written_back(&calls).is_empty() && fills(&calls).is_empty());
        let counts = r.dec.caches().counts().shapes;
        assert_eq!((counts.video_owed, counts.unseen), (1, 1));
        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 1, 0x102, &[(BLOCK, 0, false)]);
        assert!(
            fills(&r.p.call_log().since(mark)).is_empty(),
            "nothing owed"
        );
    }

    /// A command buffer whose video op names a cache's buffer (`Reach::named`, a decode's or an
    /// encode's) while the buffer lacks the pages its shapes took makes the buffer's memory at
    /// its begin, the caches' lock let go of, and copies the pages into it before the op, counted
    /// among the buffers that took them for a texture or a video op: the op reads the surface,
    /// and no command reads the make's zeros
    #[test]
    fn a_video_op_naming_a_buffer_that_lacks_the_pages_takes_them_at_its_begin() {
        use super::super::surface_cache::Reach;
        use reims_vgpu_vulkan::core::testkit::impls::resource::MAKE_MEMORY;
        for encoded in [false, true] {
            let mut r = shaped();
            let (cache, pages) = (cache_id(&r), pages_id(&r));
            texture_image(&mut r, 21, 80, (32, 8));
            use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 0, false)]);
            r.p.world().complete_all(CompleteOn::Caller);
            assert_eq!(r.p.call_log().count(MAKE_MEMORY), 0, "filled straight");
            let reach = Reach {
                video: true,
                named: vec![(BLOCK, encoded)],
                ..Reach::default()
            };
            let queue = r.device.new_command_queue().unwrap();
            let cb = queue.command_buffer(ON).unwrap().unwrap();
            let mark = r.p.call_log().mark();
            let (_, uses) = (r.dec.surface_caches)
                .before(&r.device, &cb, &reading(BLOCK, 0), &reach)
                .unwrap();
            let calls = r.p.call_log().since(mark);
            assert_eq!(copies(&calls), [(pages, cache)], "encoded {encoded}");
            assert!(
                first(&calls, MAKE_MEMORY) < first(&calls, BUFFER_COPY),
                "made first"
            );
            let counts = r.dec.caches().counts();
            assert_eq!((counts.buffer_refills, counts.buffer_refills_named), (1, 1));
            (r.dec.surface_caches)
                .finish(&r.device, 1, None, &cb, uses.unwrap(), || cb.commit(ON))
                .unwrap();
            r.p.world().complete_all(CompleteOn::Caller);
            assert_eq!(r.dec.caches().counts().lacking_named, 0);
            assert_eq!(r.p.call_log().count(MAKE_MEMORY), 1);
        }
    }

    /// NV12's planes as CoreVideo lays them in four pages, luma from 0x20 and chroma after its
    /// rows: as the kext lays them out, and as a decode writes them whole.
    #[cfg(unix)]
    fn nv12_planes() -> [PagesPlane; 2] {
        let luma = PagesPlane {
            offset: 0x20,
            bytes_per_row: 512,
            row_bytes: 480,
            rows: 16,
        };
        let chroma = PagesPlane {
            offset: 0x20 + 16 * 512,
            rows: 8,
            ..luma
        };
        [luma, chroma]
    }

    /// [`shaped`]'s rig over a cache made for the host's video, a dma-buf of [`nv12_planes`] in
    /// four pages, whose platform makes the planes' surface (`surface_cache::exported`): buffer
    /// 10 is the cache.
    #[cfg(unix)]
    fn video_shaped() -> Rig {
        video_shaped_made().0
    }

    /// [`nv12_planes`] as a `SURFACE_PAGES` lays them out.
    #[cfg(unix)]
    fn nv12_layout() -> reims_vgpu_graphics::wire::RgfxSurfacePagesLayout {
        use reims_vgpu_graphics::wire::{RgfxSurfacePagesLayout, RgfxSurfacePlaneLayout};
        let mut layout = RgfxSurfacePagesLayout {
            plane_count: 2,
            ..Default::default()
        };
        for (laid, plane) in layout.planes.iter_mut().zip(nv12_planes()) {
            *laid = RgfxSurfacePlaneLayout {
                offset: plane.offset,
                bytes_per_row: plane.bytes_per_row,
                row_bytes: plane.row_bytes,
                rows: plane.rows,
            };
        }
        layout
    }

    /// [`video_shaped`], and the images its decoder makes ([`take_shapes`]).
    #[cfg(unix)]
    fn video_shaped_made() -> (Rig, Made) {
        use reims_vgpu_vulkan::core::platform::values::PagesInPlace;
        let layout = nv12_layout();
        let mut r = rig(16);
        let page = r.page;
        let made = take_shapes(&r);
        r.dec.set_unified_memory(true);
        r.dec.set_dma_buf_caches();
        let world = Arc::clone(r.p.world());
        world.answer_surface_pages(PagesInPlace::No("off the device's alignment".into()));
        world.give_exported_surfaces(true);
        r.dec.hold_resource_ids(0, 1);
        assert_eq!(
            r.pages_asking(BLOCK, &[(1, 4)], 4 * page, Some(layout)),
            RGFX_STATUS_CACHE
        );
        assert_eq!(
            r.pages_cached(BLOCK, &[(1, 4)], 4 * page, true),
            RGFX_STATUS_OK
        );
        queues(&mut r);
        assert_eq!(r.buffer(10, BLOCK, 4 * page), Ok(()));
        (r, made)
    }

    /// Another cache made for the host's video on [`video_shaped`]'s rig, of the same layout in
    /// pages 5 to 8, at `block`: buffer `buffer` is the cache.
    #[cfg(unix)]
    fn other_video_cache(r: &mut Rig, block: u64, buffer: u64) {
        let page = r.page;
        assert_eq!(
            r.pages_asking(block, &[(5, 4)], 4 * page, Some(nv12_layout())),
            RGFX_STATUS_CACHE
        );
        assert_eq!(
            r.pages_cached(block, &[(5, 4)], 4 * page, true),
            RGFX_STATUS_OK
        );
        assert_eq!(r.buffer(buffer, block, 4 * page), Ok(()));
    }

    /// Texture `obj_id` over buffer 10 that shows plane `plane` of [`nv12_planes`] whole
    /// ([`plane_texture_over`]).
    #[cfg(unix)]
    fn plane_texture(r: &mut Rig, obj_id: u64, plane: u32) -> u64 {
        plane_texture_over(r, obj_id, 10, plane)
    }

    /// Texture `obj_id` over buffer `buffer` that shows plane `plane` of [`nv12_planes`] whole,
    /// as a decode's luma (`R8Unorm`) or chroma (`RG8Unorm`) texture: the image of its shape.
    /// The rig's device describes every pixel format with the plane's texel bytes from then on,
    /// where the fake's own answer is four.
    #[cfg(unix)]
    fn plane_texture_over(r: &mut Rig, obj_id: u64, buffer: u64, plane: u32) -> u64 {
        plane_texture_with(r, obj_id, buffer, plane, false)
    }

    /// [`plane_texture_over`], its message carrying the surface's facts where `facts`, as a
    /// decoder's output's texture's does: a decode through it shows the plane whole
    /// (`video::whole_plane`).
    #[cfg(unix)]
    fn plane_texture_with(r: &mut Rig, obj_id: u64, buffer: u64, plane: u32, facts: bool) -> u64 {
        use super::super::surface_cache::tests::with_facts;
        use reims_vgpu_graphics::wire::RgfxTextureFromIOSurface;
        use reims_vgpu_vulkan::core::platform::types::PixelFormatInfo;
        use zerocopy::{FromBytes as _, IntoBytes as _};
        let texel = u64::from(plane + 1);
        let info = PixelFormatInfo {
            name: Some(["MTLPixelFormatR8Unorm", "MTLPixelFormatRG8Unorm"][plane as usize].into()),
            block_bytes: texel,
            block_width: 1,
            block_height: 1,
            block_depth: 1,
            ..PixelFormatInfo::default()
        };
        r.p.script().clear();
        (r.p.script()).always("MTLPixelFormatGetInfoForDevice()", Step::value(info));
        let laid = nv12_planes()[plane as usize];
        let width = laid.row_bytes / texel;
        let format = [10, 30][plane as usize];
        let size = (width as u32, laid.rows as u32);
        let bytes = surface_texture_as(obj_id, buffer, format, size, (1, 0));
        let mut m = RgfxTextureFromIOSurface::read_from_bytes(&bytes).unwrap();
        m.offset = laid.offset;
        m.bytes_per_row = laid.bytes_per_row;
        m.plane = plane;
        let mut message = m.as_bytes().to_vec();
        if facts {
            let [luma, chroma] = nv12_planes();
            let size = (luma.row_bytes as u32, luma.rows as u32);
            message = with_facts(message, size, (chroma.offset, chroma.bytes_per_row));
        }
        let mark = r.p.call_log().mark();
        assert_eq!(r.dec.decode(&r.device, &message).unwrap(), Ok(()));
        views(&r.p.call_log().since(mark))[0].0
    }

    /// The (buffer, image) of each fill of a shape in `calls`.
    #[cfg(unix)]
    fn filled_images(calls: &[Call]) -> Vec<(u64, u64)> {
        fills(calls).into_iter().map(|(b, _, i)| (b, i)).collect()
    }

    /// The render passes' descriptors the fake platform makes.
    #[cfg(unix)]
    type Passes = Arc<Mutex<Vec<<FakePlatform as Platform>::RenderPassDescriptor>>>;

    /// A render pass begun ([`shapes::fill_clears`]).
    #[cfg(unix)]
    const PASS: &str = "renderCommandEncoderWithDescriptor:";

    /// A render pass's colour attachment's (load action, store action, clear colour).
    #[cfg(unix)]
    type Attachment = (Option<u64>, Option<u64>, Option<ClearColor>);

    /// The attachment of a pass that clears its image to zeros and stores it:
    /// `MTLLoadActionClear`, `MTLStoreActionStore`, 0.0 in each channel.
    #[cfg(unix)]
    const CLEARS: Attachment = (
        Some(2),
        Some(1),
        Some(ClearColor {
            red: 0.0,
            green: 0.0,
            blue: 0.0,
            alpha: 0.0,
        }),
    );

    /// The render passes' descriptors the fake platform makes from here on, kept alive (up to
    /// `n`), so a test reads what each pass's attachments were set to.
    #[cfg(unix)]
    fn keep_passes(r: &Rig, n: usize) -> Passes {
        use reims_vgpu_vulkan::core::platform::Handle as _;
        use reims_vgpu_vulkan::core::testkit::objects::kind;
        let kept = Passes::default();
        for _ in 0..n {
            let kept = Arc::clone(&kept);
            (r.p.world()).on_new::<kind::RenderPassDescriptor>(move |p| {
                kept.lock().push(p.clone_ref());
            });
        }
        kept
    }

    /// The image of each render pass begun in `calls`, each of one colour attachment, and that
    /// attachment's (load action, store action, clear colour); each pass's descriptor one that
    /// `kept` holds.
    #[cfg(unix)]
    fn passes(calls: &[Call], kept: &Passes) -> Vec<(u64, Attachment)> {
        use reims_vgpu_vulkan::core::platform::Handle as _;
        use reims_vgpu_vulkan::core::testkit::impls::descriptor::PassModel;
        let kept = kept.lock();
        (calls.iter())
            .filter(|c| c.sel == PASS)
            .map(|c| {
                let id = c.args[0].as_obj();
                let pass = (kept.iter().find(|p| Some(p.id()) == id)).expect("a kept descriptor");
                pass.with_model(|m: &PassModel| {
                    assert_eq!(
                        m.colors.keys().collect::<Vec<_>>(),
                        [&0u64],
                        "one attachment"
                    );
                    let a = &m.colors[&0];
                    let image = a.texture_id().map_or(0, |t| t.0);
                    (image, (a.load_action, a.store_action, a.clear_color))
                })
            })
            .collect()
    }

    /// A video cache's planes (`surface_cache::exported`) take their fill as the first texture
    /// over either is made, which makes them: on the transfer queue, committed, with no wait on
    /// the CPU, counted as filled at their make. The cache's buffer holds no memory, so the zeros
    /// it would give them come from a render pass for each that clears its image to zeros and
    /// stores it, with no copy and no buffer made, counted, the cache's buffer's memory left
    /// unmade (`shapes`, "Zeros"). A texture over the other plane fills nothing, and the decode's
    /// begin owes the planes no fill
    #[cfg(unix)]
    #[test]
    fn a_video_caches_planes_take_their_fill_at_their_make() {
        use super::super::surface_cache::Reach;
        use reims_vgpu_vulkan::core::testkit::impls::device::EXPORTED_SURFACE_PLANE;
        let mut r = video_shaped();
        let cache = cache_id(&r);
        let kept = keep_passes(&r, 4);
        let mark = r.p.call_log().mark();
        let luma = plane_texture(&mut r, 21, 0);
        let calls = r.p.call_log().since(mark);
        let sels: Vec<&str> = calls.iter().map(|c| c.sel).collect();
        let surfaces = sels
            .iter()
            .filter(|&&s| s == EXPORTED_SURFACE_PLANE)
            .count();
        assert_eq!(surfaces, 2, "the surface's planes: {sels:?}");
        let cleared = passes(&calls, &kept);
        assert_eq!(cleared.len(), 2, "both planes: {cleared:?}");
        assert_eq!(cleared[0], (luma, CLEARS), "luma's");
        let chroma = cleared[1].0;
        assert!(
            cleared[1].1 == CLEARS && chroma != luma,
            "chroma's: {cleared:?}"
        );
        assert!(fills(&calls).is_empty(), "no copy into them");
        assert_eq!(
            zeros_into(&calls, cache),
            0,
            "no zeros of the cache's buffer"
        );
        assert!(
            !sels.contains(&"newBufferWithLength:options:"),
            "no buffer made: {sels:?}"
        );
        let memory = (r.dec.caches().buffer(BLOCK)).is_some_and(|b| b.memory_made());
        assert!(!memory, "the cache's buffer holds no memory");
        assert!(
            first(&calls, PASS) < first(&calls, "commit"),
            "committed: {sels:?}"
        );
        assert!(!sels.contains(&"waitUntilCompleted"), "no wait: {sels:?}");
        let counts = r.dec.caches().counts().shapes;
        assert_eq!((counts.planes_filled_at_make, counts.plane_fills), (2, 2));
        assert_eq!(counts.zero_cleared, 2, "both cleared");
        assert_eq!(counts.planes_owed_busy, 0);
        assert_eq!(
            (counts.surfaces, counts.surfaces_declined),
            (1, 0),
            "a surface in place is kept"
        );
        assert!(r.dec.surface_caches.surface_targets(BLOCK).is_some());

        let mark = r.p.call_log().mark();
        assert_eq!(
            plane_texture(&mut r, 22, 1),
            chroma,
            "the chroma plane's image"
        );
        assert!(fills(&r.p.call_log().since(mark)).is_empty());

        let queue = r.device.new_command_queue().unwrap();
        let cb = queue.command_buffer(ON).unwrap().unwrap();
        let decode = Reach {
            video: true,
            decoded: vec![(BLOCK, nv12_planes())],
            ..Reach::default()
        };
        let writes = [RgfxOpSurfaceUse {
            flags: RGFX_SURFACE_USE_WRITES,
            ..reading(BLOCK, 0)[0]
        }];
        let mark = r.p.call_log().mark();
        let (_, uses) = (r.dec.surface_caches)
            .before(&r.device, &cb, &writes, &decode)
            .unwrap();
        let calls = r.p.call_log().since(mark);
        assert!(fills(&calls).is_empty(), "none at the decode's begin");
        assert!(copies(&calls).is_empty(), "the decode takes the first fill");
        (r.dec.surface_caches)
            .finish(&r.device, 1, None, &cb, uses.unwrap(), || cb.commit(ON))
            .unwrap();
        r.p.world().complete_all(CompleteOn::Caller);
        let counts = r.dec.caches().counts().shapes;
        assert_eq!(
            (counts.planes_filled_at_make, counts.plane_fills),
            (2, 2),
            "none at a later use"
        );
    }

    /// A video cache's surface the platform made in its copy form (`surface_cache::exported`),
    /// each plane its copy that it moves the bytes into and back, is declined as the first
    /// texture over a plane is made: let go of, counted, and that texture's shape made as any
    /// other cache's, of the platform's own image. The cache asks for no surface again, and a
    /// decode into its planes is given none, so the host's video writes the cache's buffer
    #[cfg(unix)]
    #[test]
    fn a_video_caches_surface_in_the_copy_form_is_declined() {
        use reims_vgpu_vulkan::core::testkit::impls::device::EXPORTED_SURFACE_PLANE;
        let (mut r, made) = video_shaped_made();
        let world = Arc::clone(r.p.world());
        world.copy_exported_surfaces(true);
        let surface_planes = |calls: &[Call]| {
            (calls.iter())
                .filter(|c| c.sel == EXPORTED_SURFACE_PLANE)
                .count()
        };
        let live = || {
            let live = world.live("MTLTexture").into_iter();
            live.map(|t| t.0).collect::<Vec<u64>>()
        };
        let before = live();
        let mark = r.p.call_log().mark();
        let luma = plane_texture(&mut r, 21, 0);
        let calls = r.p.call_log().since(mark);
        assert_eq!(surface_planes(&calls), 2, "the surface was made");
        assert_eq!(made.lock().len(), 1, "luma's own image");
        let new: Vec<u64> = (live().into_iter())
            .filter(|t| !before.contains(t))
            .collect();
        assert_eq!(
            new.len(),
            2,
            "the shape's image and the texture's view, the planes let go of: {new:?}"
        );
        assert!(new.contains(&luma), "{new:?}");
        let counts = r.dec.caches().counts().shapes;
        assert_eq!(
            (
                counts.surfaces,
                counts.surfaces_declined,
                counts.surfaces_refused
            ),
            (0, 1, 0)
        );
        assert_eq!((counts.planes_made, counts.planes_filled_at_make), (0, 0));
        assert!(
            r.dec.surface_caches.surface_targets(BLOCK).is_none(),
            "a decode lands in the cache's buffer"
        );

        let mark = r.p.call_log().mark();
        let chroma = plane_texture(&mut r, 22, 1);
        assert_eq!(
            surface_planes(&r.p.call_log().since(mark)),
            0,
            "asked once a cache"
        );
        assert_ne!(chroma, luma);
        assert_eq!(made.lock().len(), 2, "chroma's own image");
        assert_eq!(r.dec.caches().counts().shapes.surfaces_declined, 1);
    }

    /// A video cache whose surface would be made of a description the platform made in its copy
    /// form before (`surface_cache::exported`) asks for none: declined unmade, counted, its
    /// textures' shapes made as a declined cache's are, of the platform's own images, and a
    /// decode into its planes given none
    #[cfg(unix)]
    #[test]
    fn a_video_cache_asks_no_surface_of_a_description_made_in_the_copy_form() {
        use reims_vgpu_vulkan::core::testkit::impls::device::EXPORTED_SURFACE_PLANE;
        const OTHER: u64 = BLOCK + 0x4000;
        let (mut r, made) = video_shaped_made();
        r.p.world().copy_exported_surfaces(true);
        let surface_planes = |calls: &[Call]| {
            (calls.iter())
                .filter(|c| c.sel == EXPORTED_SURFACE_PLANE)
                .count()
        };
        let mark = r.p.call_log().mark();
        plane_texture(&mut r, 21, 0);
        let calls = r.p.call_log().since(mark);
        assert_eq!(
            surface_planes(&calls),
            2,
            "the first cache's surface was made"
        );
        other_video_cache(&mut r, OTHER, 11);

        let mark = r.p.call_log().mark();
        let luma = plane_texture_over(&mut r, 23, 11, 0);
        let chroma = plane_texture_over(&mut r, 24, 11, 1);
        assert_eq!(surface_planes(&r.p.call_log().since(mark)), 0, "none asked");
        assert_ne!(chroma, luma);
        assert_eq!(made.lock().len(), 3, "the shapes' own images");
        let counts = r.dec.caches().counts().shapes;
        assert_eq!(
            (
                counts.surfaces,
                counts.surfaces_declined,
                counts.surfaces_unasked,
                counts.surfaces_refused
            ),
            (0, 1, 1, 0),
            "asked once a cache"
        );
        assert!(
            r.dec.surface_caches.surface_targets(OTHER).is_none(),
            "a decode lands in the cache's buffer"
        );
    }

    /// A picture smaller than its planes leaves the rest of them holding zeros, as a new
    /// IOSurface reads: the planes took zeros at their make, each from a render pass that clears
    /// it, the cache's buffer holding no memory (`shapes`, "Zeros"), and nothing of the pages
    /// since, as the decode takes the surface's first fill; the decode op's wait on its
    /// thread for the cache's last fill lasts until that fill is done, so the hardware's picture
    /// lands over it; and its command buffer leaves the planes owing nothing, so no use after it
    /// fills them again. The cache's buffer's memory is never made. The fake holds no texels: the
    /// order pins the bytes
    #[cfg(unix)]
    #[test]
    fn a_picture_smaller_than_its_planes_leaves_the_rest_reading_zeros() {
        use super::super::surface_cache::Reach;
        let mut r = video_shaped();
        let cache = cache_id(&r);
        let kept = keep_passes(&r, 8);
        let luma = plane_texture(&mut r, 21, 0);
        let chroma = plane_texture(&mut r, 22, 1);
        let queue = r.device.new_command_queue().unwrap();
        let cb = queue.command_buffer(ON).unwrap().unwrap();
        let decode = Reach {
            video: true,
            decoded: vec![(BLOCK, nv12_planes())],
            ..Reach::default()
        };
        let writes = [RgfxOpSurfaceUse {
            flags: RGFX_SURFACE_USE_WRITES,
            ..reading(BLOCK, 0)[0]
        }];
        let (_, uses) = (r.dec.surface_caches)
            .before(&r.device, &cb, &writes, &decode)
            .unwrap();
        // The decode op, on its thread.
        let mut short = || true;
        let now = || reims_vgpu_vulkan::core::platform::os::clock::MonoTime::from_nanos(0);
        assert!(
            !r.dec.surface_caches.wait_filled(BLOCK, &mut short, &now),
            "the planes' fill not done: the wait goes on"
        );
        assert_eq!(r.dec.caches().counts().shapes.video_fill_waits, 1);
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(r.dec.surface_caches.wait_filled(BLOCK, &mut short, &now));
        assert!(
            r.dec.surface_caches.surface_targets(BLOCK).is_some(),
            "the picture lands in the planes"
        );
        (r.dec.surface_caches)
            .finish(&r.device, 1, None, &cb, uses.unwrap(), || cb.commit(ON))
            .unwrap();
        r.p.world().complete_all(CompleteOn::Caller);
        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 1, 0x101, &[(BLOCK, 0, false)]);
        assert!(
            fills(&r.p.call_log().since(mark)).is_empty(),
            "the planes hold the picture, and the zeros beside it"
        );

        let calls = r.p.call_log().since(0);
        assert_eq!(
            zeros_into(&calls, cache),
            0,
            "no zeros of the cache's buffer"
        );
        assert_eq!(
            passes(&calls, &kept),
            [(luma, CLEARS), (chroma, CLEARS)],
            "each cleared"
        );
        let filled = filled_images(&calls);
        assert!(filled.is_empty(), "no copy into them: {filled:?}");
        let memory = (r.dec.caches().buffer(BLOCK)).is_some_and(|b| b.memory_made());
        assert!(!memory, "the cache's buffer holds no memory");
        let into_cache: Vec<_> = (copies(&calls).into_iter())
            .filter(|&(_, to)| to == cache)
            .collect();
        assert!(
            into_cache.is_empty(),
            "nothing of the pages: {into_cache:?}"
        );
        let counts = r.dec.caches().counts();
        assert_eq!(counts.decode_firsts, 1, "the decode took the first fill");
        assert_eq!(
            (counts.shapes.planes_landed, counts.shapes.plane_fills),
            (1, 2)
        );
    }

    /// A video cache's planes made while a command buffer begun and not finished uses the cache
    /// owe their fill to its next use, counted, as a placed image's make finds the cache busy
    /// (`shapes`, "Placed images"): that use fills them, counted as filled at a later use, from
    /// the pages straight, as the cache's buffer holds no memory (the module doc of
    /// `surface_cache`, "Straight fills")
    #[cfg(unix)]
    #[test]
    fn a_busy_video_caches_planes_owe_their_fill_to_its_next_use() {
        use reims_vgpu_vulkan::core::platform::resource::Buffer as _;
        let mut r = video_shaped();
        let cache = cache_id(&r);
        let queue = r.device.new_command_queue().unwrap();
        let reader = queue.command_buffer(ON).unwrap().unwrap();
        let (_, uses) = (r.dec.surface_caches)
            .before(&r.device, &reader, &reading(BLOCK, 0), &NO_REACH)
            .unwrap();
        let mark = r.p.call_log().mark();
        let luma = plane_texture(&mut r, 21, 0);
        assert!(
            fills(&r.p.call_log().since(mark)).is_empty(),
            "none while a command buffer uses the cache"
        );
        let counts = r.dec.caches().counts().shapes;
        assert_eq!(
            (counts.planes_filled_at_make, counts.planes_owed_busy),
            (0, 1)
        );
        (r.dec.surface_caches)
            .finish(&r.device, 1, None, &reader, uses.unwrap(), || {
                reader.commit(ON)
            })
            .unwrap();
        r.p.world().complete_all(CompleteOn::Caller);
        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 1, 0x101, &[(BLOCK, 0, false)]);
        let filled = filled_images(&r.p.call_log().since(mark));
        assert_eq!(filled.len(), 2, "both planes at the next use: {filled:?}");
        assert!(
            filled
                .iter()
                .any(|&(from, image)| image == luma && from != cache),
            "into the plane from the pages: {filled:?}"
        );
        let memory = (r.dec.caches().buffer(BLOCK)).is_some_and(|b| b.memory_made());
        assert!(!memory, "the buffer's memory not made");
        let counts = r.dec.caches().counts().shapes;
        assert_eq!((counts.planes_filled_at_make, counts.plane_fills), (0, 2));
    }

    /// A decode whose textures show a video cache's planes whole, where the cache has its
    /// surface, is given the surface's planes, which its picture lands in, and makes no memory of
    /// the cache's buffer, which the picture does not touch ("Deferred memory"). A decode whose
    /// picture lands in the buffer (its textures not showing the planes whole, or the cache having
    /// no surface) makes it, once, and zeros it for the host's video, which writes it with no
    /// work of the platform's, before its wait for the cache's fills (`SurfaceCaches`'s
    /// `zero_for_video`)
    #[cfg(unix)]
    #[test]
    fn a_decode_into_a_caches_surface_makes_no_memory_of_its_buffer() {
        use reims_vgpu_vulkan::core::platform::resource::Buffer as _;
        use reims_vgpu_vulkan::core::testkit::impls::resource::MAKE_MEMORY;
        // (the textures show the planes whole, the platform makes the surface): whether the
        // picture lands in the surface.
        for (whole, surfaces) in [(true, true), (false, true), (true, false)] {
            let case = format!("whole {whole}, surfaces {surfaces}");
            let mut r = video_shaped();
            let cache = cache_id(&r);
            r.p.world().give_exported_surfaces(surfaces);
            plane_texture_with(&mut r, 21, 10, 0, whole);
            plane_texture_with(&mut r, 22, 10, 1, whole);
            r.p.world().complete_all(CompleteOn::Caller);
            // The clears' command buffer completes as it is committed, so the planes' wait for it
            // ends.
            r.p.world().set_auto_complete(Some(CompleteOn::Thread));
            let mark = r.p.call_log().mark();
            let (luma, chroma) = r.dec.video_planes(0, 21, 22, true).unwrap();
            let apart = whole && surfaces;
            assert_eq!(
                (luma.surface.is_some(), chroma.surface.is_some()),
                (apart, apart),
                "{case}: the surface's planes"
            );
            drop((luma, chroma));
            let calls = r.p.call_log().since(mark);
            let made = calls.iter().filter(|c| c.sel == MAKE_MEMORY).count();
            assert_eq!(made, usize::from(!apart), "{case}: the buffer's memory");
            let zeroed = (
                zeros_into(&calls, cache),
                r.dec.caches().counts().shapes.video_zeroed,
            );
            let want = (usize::from(!apart), u64::from(!apart));
            assert_eq!(zeroed, want, "{case}: the video's zeros");
            let memory = r
                .dec
                .caches()
                .buffer(BLOCK)
                .is_some_and(|b| b.memory_made());
            assert_eq!(memory, !apart, "{case}: made");
        }
    }

    /// The fills of the block's video cache's buffer with zeros in `calls`: its cache is `cache`.
    #[cfg(unix)]
    fn zeros_into(calls: &[Call], cache: u64) -> usize {
        let id = |a: &Arg| a.as_obj().map_or(0, |o| o.0);
        (calls.iter())
            .filter(|c| c.sel == "fillBuffer:range:value:" && id(&c.args[0]) == cache)
            .count()
    }

    /// A cache made for the host's video in the per-shape form is a deferred buffer whose memory
    /// is made at its first use, exported then (a dma-buf for VA-API, an exportable buffer for
    /// NVDEC), with no zeros at its make (the module doc of `surface_cache`, "Deferred memory")
    #[cfg(unix)]
    #[test]
    fn a_video_cache_in_the_per_shape_form_is_made_deferred() {
        use reims_vgpu_graphics::wire::RGFX_SURFACE_PAGES_CACHED;
        use reims_vgpu_vulkan::core::platform::resource::Buffer as _;
        use reims_vgpu_vulkan::core::testkit::impls::device::{
            DEFERRED_DMA_BUF_BUFFER, DEFERRED_EXPORTABLE_BUFFER, DMA_BUF_BUFFER, EXPORTABLE_BUFFER,
        };
        for exportable in [false, true] {
            let mut r = rig(16);
            let page = r.page;
            let _made = take_shapes(&r);
            if exportable {
                r.dec.set_exportable_caches();
            } else {
                r.dec.set_dma_buf_caches();
            }
            let mark = r.p.call_log().mark();
            let flags = RGFX_SURFACE_PAGES_CACHED;
            let layout = Some(nv12_layout());
            assert_eq!(
                r.pages_laid_out(BLOCK, 0, &[(1, 4)], 4 * page, flags, layout),
                RGFX_STATUS_OK
            );
            let sels = r.p.call_log().sels_since(mark);
            let count = |sel: &str| sels.iter().filter(|&&s| s == sel).count();
            let (deferred, whole) = match exportable {
                true => (DEFERRED_EXPORTABLE_BUFFER, EXPORTABLE_BUFFER),
                false => (DEFERRED_DMA_BUF_BUFFER, DMA_BUF_BUFFER),
            };
            assert_eq!(
                (count(deferred), count(whole)),
                (1, 0),
                "exportable {exportable}: {sels:?}"
            );
            assert!(r.dec.caches().identity(BLOCK).is_some(), "given the video");
            let memory = r.dec.caches().buffer(BLOCK).map(|b| b.memory_made());
            assert_eq!(memory, Some(false), "exportable {exportable}: no memory");
            assert!(
                !sels.contains(&"fillBuffer:range:value:"),
                "no zeros: {sels:?}"
            );
        }
    }

    /// A decode whose picture lands in a video cache's buffer, whose memory is not made, makes
    /// it as its command buffer begins and fills it with zeros on the transfer queue, as the
    /// buffer's make did when it made the memory: the host's video writes the memory itself,
    /// ahead of the platform's own zeros of it (`SurfaceCaches::zero_for_video`). The decode's op
    /// waits for those zeros, counted; a second decode makes and zeros nothing
    #[cfg(unix)]
    #[test]
    fn a_decode_into_a_video_caches_buffer_makes_its_memory_and_zeros() {
        use super::super::surface_cache::Reach;
        use reims_vgpu_vulkan::core::platform::resource::Buffer as _;
        use reims_vgpu_vulkan::core::testkit::impls::resource::MAKE_MEMORY;
        let r = video_shaped();
        let cache = cache_id(&r);
        assert_eq!(
            zeros_into(&r.p.call_log().since(0), cache),
            0,
            "none at its make"
        );
        let memory = |r: &Rig| (r.dec.caches().buffer(BLOCK)).is_some_and(|b| b.memory_made());
        assert!(!memory(&r));
        let queue = r.device.new_command_queue().unwrap();
        for (round, want) in [(1, 1), (0, 0)].into_iter().enumerate() {
            let cb = queue.command_buffer(ON).unwrap().unwrap();
            let decode = Reach {
                video: true,
                decoded: vec![(BLOCK, nv12_planes())],
                named: vec![(BLOCK, false)],
                ..Reach::default()
            };
            let writes = [RgfxOpSurfaceUse {
                flags: RGFX_SURFACE_USE_WRITES,
                ..reading(BLOCK, 0)[0]
            }];
            let mark = r.p.call_log().mark();
            let (_, uses) = (r.dec.surface_caches)
                .before(&r.device, &cb, &writes, &decode)
                .unwrap();
            let calls = r.p.call_log().since(mark);
            let made = calls.iter().filter(|c| c.sel == MAKE_MEMORY).count();
            assert_eq!(
                (made, zeros_into(&calls, cache)),
                want,
                "decode {round}: made, zeroed"
            );
            if round == 0 {
                assert!(
                    r.dec.surface_caches.surface_targets(BLOCK).is_none(),
                    "no surface: the picture lands in the buffer"
                );
                let mut short = || true;
                let now = || reims_vgpu_vulkan::core::platform::os::clock::MonoTime::from_nanos(0);
                assert!(
                    !r.dec.surface_caches.wait_filled(BLOCK, &mut short, &now),
                    "the zeros not done: the decode's op waits"
                );
                assert_eq!(r.dec.caches().counts().shapes.video_fill_waits, 1);
            }
            r.p.world().complete_all(CompleteOn::Caller);
            (r.dec.surface_caches)
                .finish(&r.device, 1, None, &cb, uses.unwrap(), || cb.commit(ON))
                .unwrap();
            r.p.world().complete_all(CompleteOn::Caller);
            assert!(memory(&r), "decode {round}: made");
        }
        assert_eq!(r.dec.caches().counts().shapes.video_zeroed, 1);
    }

    /// A video cache's buffer that holds no memory takes a fill from the pages straight into its
    /// shapes, as any deferred cache's does (the module doc of `surface_cache`, "Straight
    /// fills"), its memory left unmade. A decode whose picture lands in the buffer then makes the
    /// memory, zeros it, and fills it from the pages over the zeros, all before the decode's
    /// command buffer: the buffer the video writes holds the pages, as an eager cache's did
    #[cfg(unix)]
    #[test]
    fn a_video_caches_buffer_takes_the_pages_over_its_zeros_at_a_decode() {
        use super::super::surface_cache::Reach;
        use reims_vgpu_vulkan::core::platform::resource::Buffer as _;
        let mut r = video_shaped();
        let cache = cache_id(&r);
        r.p.world().give_exported_surfaces(false);
        let luma = plane_texture(&mut r, 21, 0);
        r.p.world().complete_all(CompleteOn::Caller);
        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 1, 0x101, &[(BLOCK, 1, false)]);
        r.p.world().complete_all(CompleteOn::Caller);
        let calls = r.p.call_log().since(mark);
        let memory = |r: &Rig| (r.dec.caches().buffer(BLOCK)).is_some_and(|b| b.memory_made());
        assert!(!memory(&r), "the pages' fill went straight");
        let filled = filled_images(&calls);
        assert!(
            filled
                .iter()
                .any(|&(from, image)| image == luma && from != cache),
            "into the shape from the pages: {filled:?}"
        );
        assert_eq!(zeros_into(&calls, cache), 0);

        let queue = r.device.new_command_queue().unwrap();
        let cb = queue.command_buffer(ON).unwrap().unwrap();
        let decode = Reach {
            video: true,
            decoded: vec![(BLOCK, nv12_planes())],
            named: vec![(BLOCK, false)],
            ..Reach::default()
        };
        let writes = [RgfxOpSurfaceUse {
            flags: RGFX_SURFACE_USE_WRITES,
            ..reading(BLOCK, 1)[0]
        }];
        let mark = r.p.call_log().mark();
        let (_, uses) = (r.dec.surface_caches)
            .before(&r.device, &cb, &writes, &decode)
            .unwrap();
        let calls = r.p.call_log().since(mark);
        assert!(memory(&r), "made for the decode");
        let id = |a: &Arg| a.as_obj().map_or(0, |o| o.0);
        let zeros = (calls.iter())
            .position(|c| c.sel == "fillBuffer:range:value:" && id(&c.args[0]) == cache)
            .expect("the memory's zeros");
        let pages = (calls.iter())
            .position(|c| c.sel == COPY && id(&c.args[2]) == cache)
            .expect("the pages into the buffer");
        assert!(zeros < pages, "the pages over the zeros");
        (r.dec.surface_caches)
            .finish(&r.device, 1, None, &cb, uses.unwrap(), || cb.commit(ON))
            .unwrap();
        r.p.world().complete_all(CompleteOn::Caller);
        assert_eq!(r.dec.caches().counts().shapes.video_zeroed, 1);
    }

    /// Library 30 and compute pipeline `obj_id` of a function in it.
    fn compute_pipeline(r: &mut Rig, obj_id: u64) {
        use reims_vgpu_graphics::wire::{
            RGFX_CMD_COMPUTE_PIPELINE_CREATE, RGFX_CMD_LIBRARY_CREATE, RGFX_LIBRARY_BYTES_OFFSET,
            RgfxComputePipelineCreate, RgfxLibraryCreate, RgfxMsgHeader,
        };
        use zerocopy::IntoBytes as _;
        let air = [1u8, 2, 3, 4];
        let mut library = RgfxLibraryCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_LIBRARY_CREATE,
                size: (RGFX_LIBRARY_BYTES_OFFSET as usize + air.len()).next_multiple_of(8) as u32,
                seqno: 0,
            },
            obj_id: 30,
            byte_length: air.len() as u64,
            ..RgfxLibraryCreate::default()
        }
        .as_bytes()
        .to_vec();
        library.extend_from_slice(&air);
        assert_eq!(r.dec.decode(&r.device, &library).unwrap(), Ok(()));
        let head = size_of::<RgfxComputePipelineCreate>() as u32;
        let name = b"cmain".as_slice();
        let total = (head as usize + name.len()).next_multiple_of(8);
        let mut pipeline = RgfxComputePipelineCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_COMPUTE_PIPELINE_CREATE,
                size: total as u32,
                seqno: 1,
            },
            obj_id,
            function_library_obj_id: 30,
            name_offset: head,
            name_len: name.len() as u32,
            ..RgfxComputePipelineCreate::default()
        }
        .as_bytes()
        .to_vec();
        pipeline.extend_from_slice(name);
        pipeline.resize(total, 0);
        assert_eq!(r.dec.decode(&r.device, &pipeline).unwrap(), Ok(()));
    }

    /// Texture `obj_id` over buffer 10, 32 by 8 BGRA8, whose usage lets a shader write it, and
    /// the image of its shape.
    fn writable_image(r: &mut Rig, obj_id: u64) -> u64 {
        let mark = r.p.call_log().mark();
        let message = surface_texture_as(obj_id, 10, 80, (32, 8), (0x2, 0));
        assert_eq!(r.dec.decode(&r.device, &message).unwrap(), Ok(()));
        views(&r.p.call_log().since(mark))[0].0
    }

    /// Command buffer `cb_obj_id` on queue 1, naming [`BLOCK`] written, of one compute encoder
    /// of `ops`.
    fn computes(r: &mut Rig, cb_obj_id: u64, ops: &[(u16, Vec<u8>)]) {
        use reims_vgpu_graphics::ops::RGFX_OP_ENCODER_BEGIN_COMPUTE;
        encoder_with(r, cb_obj_id, true, RGFX_OP_ENCODER_BEGIN_COMPUTE, ops);
    }

    /// A compute encoder's op that binds pipeline `obj_id`.
    fn bind_pipeline(obj_id: u64) -> (u16, Vec<u8>) {
        use reims_vgpu_graphics::ops::compute::{
            RGFX_OP_COMPUTE_SET_COMPUTE_PIPELINE_STATE, RgfxComputeOpSetComputePipelineState,
        };
        use zerocopy::IntoBytes as _;
        let m = RgfxComputeOpSetComputePipelineState {
            pipeline_obj_id: obj_id,
        };
        (
            RGFX_OP_COMPUTE_SET_COMPUTE_PIPELINE_STATE,
            m.as_bytes().to_vec(),
        )
    }

    /// A compute encoder's op that binds texture `obj_id` (0 for nil) at slot `index`.
    fn bind_texture(obj_id: u64, index: u64) -> (u16, Vec<u8>) {
        use reims_vgpu_graphics::ops::compute::{
            RGFX_OP_COMPUTE_SET_TEXTURE, RgfxComputeOpSetTexture,
        };
        use zerocopy::IntoBytes as _;
        let m = RgfxComputeOpSetTexture {
            texture_obj_id: obj_id,
            index,
        };
        (RGFX_OP_COMPUTE_SET_TEXTURE, m.as_bytes().to_vec())
    }

    /// The slot each pipeline was asked whether its function may write in `calls`.
    fn slots_asked(calls: &[Call]) -> Vec<u64> {
        (calls.iter())
            .filter(|c| c.sel == TEXTURE_SLOT_WRITES)
            .map(|c| c.args[0].as_u64().unwrap_or(u64::MAX))
            .collect()
    }

    /// Whether a download after `ops` in a compute encoder writes back texture 25, a shape's view
    /// whose usage lets a shader write it, with the shapes' usage writes and slot reads counted;
    /// and the slots pipeline 31 was asked of, its answer scripted as `answer` where given.
    fn slot_run(
        answer: Option<Option<bool>>,
        ops: &[(u16, Vec<u8>)],
    ) -> ((bool, u64, u64), Vec<u64>) {
        let mut r = shaped();
        let cache = cache_id(&r);
        compute_pipeline(&mut r, 31);
        if let Some(answer) = answer {
            r.p.script()
                .always(TEXTURE_SLOT_WRITES, Step::value(answer));
        }
        let image = writable_image(&mut r, 25);
        let mark = r.p.call_log().mark();
        computes(&mut r, 0x100, ops);
        let asked = slots_asked(&r.p.call_log().since(mark));
        r.p.world().complete_all(CompleteOn::Caller);
        let mark = r.p.call_log().mark();
        let downloaded = download(&r, BLOCK);
        let back = written_back(&r.p.call_log().since(mark));
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(downloaded.load(Ordering::SeqCst));
        let written = match back.as_slice() {
            [] => false,
            [only] => {
                assert_eq!(*only, (image, cache, 0));
                true
            }
            _ => panic!("written back more than once: {back:?}"),
        };
        let counts = r.dec.caches().counts().shapes;
        ((written, counts.usage_writes, counts.slot_reads), asked)
    }

    /// A texture whose usage lets a shader write it, bound at a slot that the encoder's pipeline
    /// only reads, leaves its shape clean: a slot bound before the pipeline is asked of once it
    /// is bound, one bound after as it is bound, and each is counted read as the encoder ends
    #[test]
    fn a_slot_its_pipeline_only_reads_leaves_its_shape_clean() {
        let ops = [bind_texture(25, 0), bind_pipeline(31), bind_texture(25, 1)];
        let (seen, asked) = slot_run(Some(Some(false)), &ops);
        assert_eq!(seen, (false, 0, 2));
        assert_eq!(asked, [0, 1]);
    }

    /// A slot that the encoder's pipeline may write writes its texture's shape, once the pipeline
    /// is bound after it; a texture bound at a slot and replaced there before any pipeline is
    /// counted read
    #[test]
    fn a_slot_its_pipeline_may_write_writes_its_shape() {
        let ops = [
            bind_texture(25, 1),
            bind_texture(0, 1),
            bind_texture(25, 0),
            bind_pipeline(31),
        ];
        let (seen, asked) = slot_run(Some(Some(true)), &ops);
        assert_eq!(seen, (true, 0, 1));
        assert_eq!(asked, [0]);
    }

    /// A pipeline that knows no slot's access leaves the texture's usage to decide: one that lets
    /// a shader write it writes its shape, counted; with no pipeline bound in the encoder, nothing
    /// writes it
    #[test]
    fn a_pipeline_that_knows_no_slots_access_leaves_the_usage_to_decide() {
        let (seen, asked) = slot_run(None, &[bind_pipeline(31), bind_texture(25, 0)]);
        assert_eq!(seen, (true, 1, 0));
        assert_eq!(asked, [0]);
        let (seen, asked) = slot_run(None, &[bind_texture(25, 0)]);
        assert_eq!(seen, (false, 0, 1));
        assert!(asked.is_empty());
    }

    /// In the per-shape form every surface takes a device cache: a block that asks whether it
    /// needs one is answered that it does, with no view made, and takes its cache when named
    /// cached; a block named uncached is refused, so the kext gives it a copy block, keeping no
    /// view, counted
    #[test]
    fn every_surface_takes_a_device_cache() {
        let mut r = rig(16);
        let page = r.page;
        take_shapes(&r);
        let layout = one_plane(page, page - 16, 2);
        assert_eq!(
            r.pages_asking(BLOCK, &[(1, 2)], 2 * page, Some(layout)),
            RGFX_STATUS_CACHE
        );
        assert!(r.dec.views_of_surfaces().view(BLOCK).is_none());
        assert_eq!(
            r.pages_cached(BLOCK, &[(1, 2)], 2 * page, true),
            RGFX_STATUS_OK
        );
        assert!(r.dec.caches().has(BLOCK));
        let second = BLOCK + 0x8000;
        assert_eq!(
            r.pages_cached(second, &[(3, 2)], 2 * page, false),
            RGFX_STATUS_REFUSED
        );
        assert!(r.dec.views_of_surfaces().view(second).is_none());
        assert!(!r.dec.caches().has(second));
        assert_eq!(r.dec.caches().counts().shapes.uncached_refused, 1);
    }

    /// In the per-shape form a block named cached whose pages the host makes no buffer over takes
    /// its device cache all the same, answered OK where it was refused, counted, with no view:
    /// the block's buffers are its cache, a texture over them a view of a shape's image, and the
    /// cache's first fill copies the pages into a staging buffer of its own, which the GPU copies
    /// straight into the shape's image, none into the cache. In the default form the block is
    /// refused still, and the kext gives the surface a copy block
    #[test]
    fn a_cached_block_with_no_view_takes_its_cache_copied() {
        let mut r = rig(8);
        let page = r.page;
        take_shapes(&r);
        r.dec.hold_resource_ids(0, 1);
        // The platform makes no buffer over the pages' view (`buffer_over`).
        r.p.script().push("newBufferWithDescriptor:", Step::Nil);
        assert_eq!(
            r.pages_cached(BLOCK, &[(1, 2)], 2 * page, true),
            RGFX_STATUS_OK
        );
        assert!(r.dec.caches().has(BLOCK));
        assert!(r.dec.views_of_surfaces().view(BLOCK).is_none());
        assert_eq!(r.dec.caches().counts().shapes.pages_copied, 1);
        let cache = cache_id(&r);
        queues(&mut r);
        let mark = r.p.call_log().mark();
        assert_eq!(r.buffer(10, BLOCK, 2 * page), Ok(()));
        let sels = r.p.call_log().sels_since(mark);
        assert!(
            !sels.iter().any(|s| s.starts_with("newBuffer")),
            "no buffer made: its buffers are the cache"
        );
        let mark = r.p.call_log().mark();
        texture(&mut r, 21, 80, (32, 8));
        let calls = r.p.call_log().since(mark);
        assert!(surface_textures(&calls).is_empty(), "none over the buffer");
        assert_eq!(views(&calls).len(), 1, "a view of its shape's image");
        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 2, false)]);
        let calls = r.p.call_log().since(mark);
        let filled = copies(&calls);
        assert!(filled.is_empty(), "none into the cache: {filled:?}");
        let shapes = fills(&calls);
        assert_eq!(shapes.len(), 1, "the cache's first fill");
        assert_ne!(
            shapes[0].0, cache,
            "the shape from a staging buffer of its own"
        );
        assert_eq!(r.dec.caches().counts().shapes.straight_fills, 1);

        let mut r = rig(8);
        r.p.script().push("newBufferWithDescriptor:", Step::Nil);
        assert_eq!(
            r.pages_cached(BLOCK, &[(1, 2)], 2 * page, true),
            RGFX_STATUS_REFUSED
        );
        assert!(!r.dec.caches().has(BLOCK));
    }

    /// A copy of one buffer into another ([`super::super::surface_cache::Way`]'s parts).
    const COPY: &str = "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:";

    /// The (destination, source offset, destination offset, size) of each copy out of `cache`
    /// into another buffer in `calls`.
    fn copies_out(calls: &[Call], cache: u64) -> Vec<(u64, u64, u64, u64)> {
        let id = |a: &Arg| a.as_obj().map_or(0, |o| o.0);
        (calls.iter())
            .filter(|c| c.sel == COPY && id(&c.args[0]) == cache)
            .map(|c| {
                let at = |i: usize| c.args[i].as_u64().unwrap_or(u64::MAX);
                (id(&c.args[2]), at(1), at(3), at(4))
            })
            .collect()
    }

    /// [`shaped`]'s rig with the caches' copies staged, its pages' bytes set first (from page 1,
    /// [`ram_bytes`]); pipeline 31's function writes every texture slot it is asked of.
    fn shaped_staged() -> Rig {
        let mut r = rig(8);
        r.dec.set_staged_surface_copies();
        let page = r.page;
        take_shapes(&r);
        r.dec.hold_resource_ids(0, 1);
        ram_bytes(&r, 1, 2 * page, Some(0x11));
        assert_eq!(
            r.pages_cached(BLOCK, &[(1, 2)], 2 * page, true),
            RGFX_STATUS_OK
        );
        queues(&mut r);
        assert_eq!(r.buffer(10, BLOCK, 2 * page), Ok(()));
        compute_pipeline(&mut r, 31);
        r.p.script()
            .always(TEXTURE_SLOT_WRITES, Step::value(Some(true)));
        r
    }

    /// Texture `obj_id` over buffer 10 at byte `offset` of the surface, 32 by 8 BGRA8 at a pitch
    /// of 128 (1 KiB from `offset`), whose usage lets a shader write it, and the image of its
    /// shape.
    fn writable_at(r: &mut Rig, obj_id: u64, offset: u64) -> u64 {
        use reims_vgpu_graphics::wire::RgfxTextureFromIOSurface;
        use zerocopy::{FromBytes as _, IntoBytes as _};
        let message = surface_texture_as(obj_id, 10, 80, (32, 8), (0x2, 0));
        let mut m = RgfxTextureFromIOSurface::read_from_bytes(&message).unwrap();
        m.offset = offset;
        let mark = r.p.call_log().mark();
        assert_eq!(r.dec.decode(&r.device, m.as_bytes()).unwrap(), Ok(()));
        views(&r.p.call_log().since(mark))[0].0
    }

    /// Command buffer `cb_obj_id`, naming [`BLOCK`] written, whose compute encoder writes texture
    /// `obj_id` through a slot pipeline 31 writes.
    fn shader_writes(r: &mut Rig, cb_obj_id: u64, obj_id: u64) {
        computes(r, cb_obj_id, &[bind_pipeline(31), bind_texture(obj_id, 0)]);
    }

    /// A download of [`BLOCK`] in `r`, its staging buffer's bytes set to `value` before the
    /// GPU's copy is done, answered once the pages hold them: the shapes it wrote back, each by
    /// (image, buffer, offset), and its copies out of `cache` ([`copies_out`]).
    #[allow(clippy::type_complexity)] // the two lists the tests read
    fn staged_download(
        r: &Rig,
        cache: u64,
        value: u8,
    ) -> (Vec<(u64, u64, u64)>, Vec<(u64, u64, u64, u64)>) {
        let mark = r.p.call_log().mark();
        let done = download(r, BLOCK);
        let calls = r.p.call_log().since(mark);
        let out = copies_out(&calls, cache);
        if let Some(&(staging, ..)) = out.first() {
            buffer_bytes(r, staging, Some(value));
        }
        complete_staged(r);
        assert!(done.load(Ordering::SeqCst), "answered");
        (written_back(&calls), out)
    }

    /// In the per-shape form a staged download moves only the bytes the host's copies wrote
    /// into the cache's buffer since its last copy out (the module doc of `surface_cache`,
    /// "Written ranges"): the shape written back, at its place, on the GPU and on the host's
    /// CPU. A shape written next outside the earlier one's bytes moves its own alone, so the
    /// bytes the guest's CPU wrote into the pages since, the earlier shape's among them, stay
    #[test]
    fn a_staged_download_moves_only_the_shapes_written_back() {
        let mut r = shaped_staged();
        let cache = cache_id(&r);
        let length = 2 * r.page;
        let first = writable_at(&mut r, 25, 0);
        let second = writable_at(&mut r, 26, 2048);
        let before = ram_bytes(&r, 1, length, None);

        shader_writes(&mut r, 0x100, 25);
        complete_staged(&r);
        let (back, out) = staged_download(&r, cache, 0xc3);
        assert_eq!(back, [(first, cache, 0)]);
        assert_eq!(out.len(), 1, "{out:?}");
        assert_eq!(
            (out[0].1, out[0].2, out[0].3),
            (0, 0, 1024),
            "the first shape's KiB alone"
        );
        let pages = ram_bytes(&r, 1, length, None);
        assert!(
            pages[..1024].iter().all(|&b| b == 0xc3),
            "the shape's bytes"
        );
        assert_eq!(
            pages[1024..],
            before[1024..],
            "the rest as the pages held them"
        );
        let counts = r.dec.caches().counts();
        assert_eq!(counts.narrowed, 1);
        assert_eq!(counts.unwritten_bytes, length - 1024);

        // The guest's CPU writes the pages, then the GPU the second shape alone.
        let poked = ram_bytes(&r, 1, length, Some(0x55));
        shader_writes(&mut r, 0x101, 26);
        complete_staged(&r);
        let (back, out) = staged_download(&r, cache, 0x3c);
        assert_eq!(back, [(second, cache, 2048)]);
        assert_eq!(out.len(), 1, "{out:?}");
        assert_eq!(
            (out[0].1, out[0].2, out[0].3),
            (2048, 0, 1024),
            "the second shape's KiB alone, at its piece's first byte"
        );
        let pages = ram_bytes(&r, 1, length, None);
        assert!(
            pages[2048..3072].iter().all(|&b| b == 0x3c),
            "the second shape's bytes"
        );
        assert_eq!(
            pages[..2048],
            poked[..2048],
            "the CPU's bytes over the first shape stay"
        );
        assert_eq!(pages[3072..], poked[3072..], "and past the second");
    }

    /// A command buffer that may write the block and names an object over the cache's buffer
    /// itself (here a blit from buffer 10, the buffer over the surface's pages) makes the next
    /// staged download move all of the surface's bytes, as the walk cannot tell what it wrote
    /// there; the one after, nothing but a shape written since, moves that shape's alone. The
    /// buffer holds its memory from before the first fill here ([`buffer_made`]): one that lacks
    /// the pages takes them at the next copy through it, over what the command wrote
    /// ([`a_command_naming_a_buffer_that_lacks_the_pages_reads_its_makes_zeros`])
    #[test]
    fn a_writer_over_the_caches_buffer_makes_the_download_whole() {
        let mut r = shaped_staged();
        buffer_made(&r);
        let cache = cache_id(&r);
        let length = 2 * r.page;
        let first = writable_at(&mut r, 25, 0);

        write_from(&mut r, 0x100, 25, (32, 8), 10);
        complete_staged(&r);
        let (back, out) = staged_download(&r, cache, 0xc3);
        assert_eq!(back, [(first, cache, 0)]);
        assert_eq!(out.len(), 1, "{out:?}");
        assert_eq!((out[0].1, out[0].2, out[0].3), (0, 0, length), "whole");
        let pages = ram_bytes(&r, 1, length, None);
        assert!(pages.iter().all(|&b| b == 0xc3), "every byte");
        assert_eq!(r.dec.caches().counts().narrowed, 0);

        shader_writes(&mut r, 0x101, 25);
        complete_staged(&r);
        let (back, out) = staged_download(&r, cache, 0x3c);
        assert_eq!(back, [(first, cache, 0)]);
        assert_eq!(
            (out[0].1, out[0].2, out[0].3),
            (0, 0, 1024),
            "the shape's alone"
        );
    }

    /// A staged download after a writer that wrote nothing the host copied into the cache's
    /// buffer moves nothing and makes no staging buffer, counted, and is answered; its buffer,
    /// which lacks the pages its shape's straight fill took, takes none of them, and makes no
    /// memory, so the next fill goes straight again. The next such writer of the surface, which
    /// reads its writes back, makes none for an early copy, which copied pages take none of;
    /// nor does its release
    #[test]
    fn a_staged_download_with_nothing_written_moves_nothing() {
        use reims_vgpu_vulkan::core::testkit::impls::resource::MAKE_MEMORY;
        let mut r = shaped_staged();
        let cache = cache_id(&r);
        let length = 2 * r.page;
        writable_at(&mut r, 25, 0);
        let before = ram_bytes(&r, 1, length, None);
        // A writer of the block whose encoder binds no texture.
        computes(&mut r, 0x100, &[bind_pipeline(31)]);
        complete_staged(&r);
        assert_eq!(r.dec.caches().counts().shapes.straight_fills, 1);
        let mark = r.p.call_log().mark();
        let (back, out) = staged_download(&r, cache, 0xc3);
        assert!(back.is_empty() && out.is_empty(), "{back:?} {out:?}");
        let sels = r.p.call_log().sels_since(mark);
        assert!(
            !sels.contains(&reims_vgpu_vulkan::core::testkit::impls::device::SCRATCH_BUFFER),
            "no staging buffer made"
        );
        assert_eq!(ram_bytes(&r, 1, length, None), before);
        let counts = r.dec.caches().counts();
        assert_eq!((counts.narrowed, counts.narrowed_none), (1, 1));
        assert_eq!(
            counts.buffer_refills, 0,
            "nor a fill of the buffer that lacks the pages"
        );
        assert_eq!(r.p.call_log().count(MAKE_MEMORY), 0, "no memory made");
        assert_eq!(r.dec.caches().live().buffer_made_bytes, 0);

        use_surfaces(&mut r, 1, 0x101, &[(BLOCK, 3, false)]);
        complete_staged(&r);
        let shapes = r.dec.caches().counts().shapes;
        assert_eq!(shapes.straight_fills, 2, "straight again");
        computes(&mut r, 0x102, &[bind_pipeline(31)]);
        complete_staged(&r);
        assert_eq!(
            r.p.call_log().count(MAKE_MEMORY),
            0,
            "nor for an early copy behind the writer, which copied pages take none of"
        );
        let released = release(&r, BLOCK);
        complete_staged(&r);
        assert!(released.load(Ordering::SeqCst));
        assert_eq!(r.p.call_log().count(MAKE_MEMORY), 0, "nor for the release");
    }

    /// Where the caches' copies are staged, a fill of a cache whose buffer holds no memory copies
    /// a shape's bytes straight from the pieces the host's CPU copied the pages into, none into
    /// the buffer: the shape's rows go in a run while they lie in one piece, and a row that
    /// crosses a piece's end goes in two parts split there, each at its own texels, counted. A
    /// later copy through the buffer takes the pages as they are then, not as the fill read
    /// them: bytes the guest's CPU wrote into the pages between the two, under no lock, reach the
    /// buffer, where a fill through the buffer kept the bytes it read (`surface_cache`'s module
    /// doc, "Straight fills")
    #[test]
    fn a_staged_straight_fill_splits_a_row_at_its_pieces_end() {
        use reims_vgpu_vulkan::core::platform::types::MtlOrigin;
        use reims_vgpu_vulkan::core::testkit::impls::resource::MAKE_MEMORY;
        let mut r = shaped_staged();
        r.dec.surface_caches.set_staging_piece(1024);
        let cache = cache_id(&r);
        let length = 2 * r.page;
        // 32 by 8 at a pitch of 128 from byte 64: row 7, bytes 960 to 1088, crosses 1024.
        let image = writable_at(&mut r, 25, 64);
        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 2, false)]);
        let calls = r.p.call_log().since(mark);
        complete_staged(&r);
        assert!(
            copies(&calls).iter().all(|&(_, to)| to != cache),
            "none into the buffer"
        );
        assert_eq!(r.p.call_log().count(MAKE_MEMORY), 0, "no memory made");
        let filled = fills(&calls);
        assert_eq!(filled.len(), 3, "{filled:?}");
        let (head, tail) = (filled[0].0, filled[2].0);
        assert_ne!(head, tail, "two pieces");
        assert_eq!(
            filled,
            [(head, 64, image), (head, 960, image), (tail, 0, image)]
        );
        let parts: Vec<(Arg, Arg)> = (calls.iter())
            .filter(|c| c.sel == FILL)
            .map(|c| (c.args[4].clone(), c.args[8].clone()))
            .collect();
        let origin = |x, y| Arg::val(&MtlOrigin { x, y, z: 0 });
        assert_eq!(
            parts,
            [
                (texels(32, 7), origin(0, 0)),
                (texels(16, 1), origin(0, 7)),
                (texels(16, 1), origin(16, 7))
            ]
        );
        let sh = r.dec.caches().counts().shapes;
        let straight = (
            sh.straight_fills,
            sh.straight_copies,
            sh.straight_split_rows,
        );
        assert_eq!(straight, (1, 3, 1));

        // The guest's CPU writes the pages with no lock, then a shader the shape.
        let poked = ram_bytes(&r, 1, length, Some(0x55));
        shader_writes(&mut r, 0x101, 25);
        complete_staged(&r);
        let mark = r.p.call_log().mark();
        let done = download(&r, BLOCK);
        let calls = r.p.call_log().since(mark);
        let into: Vec<u64> = (copies(&calls).into_iter())
            .filter(|&(_, to)| to == cache)
            .map(|(from, _)| from)
            .collect();
        assert_eq!(
            into.len() as u64,
            length / 1024,
            "the pages into the buffer, by piece"
        );
        let held: Vec<u8> = (into.iter())
            .flat_map(|&piece| buffer_bytes(&r, piece, None)[..1024].to_vec())
            .collect();
        assert!(held == poked, "the pages as they are now");
        assert_eq!(written_back(&calls), [(image, cache, 64)]);
        complete_staged(&r);
        assert!(done.load(Ordering::SeqCst), "answered");
        assert_eq!(r.dec.caches().counts().buffer_refills, 1);
    }
}
