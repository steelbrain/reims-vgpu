//! `TEXTURE_FROM_IOSURFACE` over a device cache in the per-shape form (`surface_cache::shapes`):
//! the guest's texture is a view, at the guest's resource index and in the guest's format, of
//! its shape's image, which the cache keeps for every texture of that shape over the surface.
//!
//! The usage the guest declares picks the shape's kind (`shapes`, "The shape"): a texture of
//! Metal's Unknown usage (0), `ShaderWrite` or `ShaderAtomic` takes storage, and views the storage
//! shape of its extent, made of the Unknown usage where the cache holds none; any other views the
//! storage shape of its extent where the cache holds one, else its plain shape, made of
//! `ShaderRead | RenderTarget` alone. A texture over a plain shape in a format neither the
//! shape's nor its sRGB sibling asks a view rt refuses, counted
//! ([`shapes::PLAIN_VIEW_REFUSED`], logged once), and lies over the buffer as any texture whose
//! view the platform does not make; one over a plain shape that declares `PixelFormatView` is
//! counted too.
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

use super::surface_cache::{self, CacheLayout, STORAGE_PRIVATE, ShapeMaker};
use super::surface_cache::shapes::{self, PLAIN_VIEW_REFUSED, ShapeCounts, ShapeKey, ShapeUse};
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

/// Whether a view refused over a plain shape was logged ([`PLAIN_VIEW_REFUSED`]): its line is
/// logged the first time alone, the rest counted (docs/conventions.md).
static PLAIN_LOGGED: AtomicBool = AtomicBool::new(false);

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
    /// Takes the per-shape form from now on (module doc), each shape's image made by `maker`:
    /// set once, at the device's make, before any surface is named.
    pub(crate) fn set_surface_shapes(&self, maker: ShapeMaker<P>) {
        self.surface_caches.set_shapes(maker);
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
        let Some((reference, image)) = caches.shape(device, block, key, &image) else {
            return unshaped(|_| {});
        };
        if image.usage() == PLAIN_USAGE {
            self.note_plain_view(m, image.pixel_format());
        }
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
        #[cfg(feature = "rt")]
        reims_vgpu_vulkan::rt_platform::diag::declared(&view, u64::from(d.usage));
        let shape = ShapeUse {
            block,
            writes: takes_storage(m),
            reference,
            #[cfg(feature = "rt")]
            diag: reims_vgpu_vulkan::rt_platform::diag::token(&view),
        };
        Ok(Some((view, shape)))
    }

    /// The texture `m` asks lies over a plain shape of `shape_format`: one that declares
    /// `PixelFormatView` is counted ([`ShapeCounts::plain_format_views`]), and one in a format
    /// neither `shape_format` nor its sRGB sibling asks a view rt refuses, counted
    /// ([`PLAIN_VIEW_REFUSED`], logged once). The view is asked all the same.
    fn note_plain_view(&self, m: &RgfxTextureFromIOSurface, shape_format: u64) {
        let caches = &self.surface_caches;
        if u64::from(m.desc.usage) & TEXTURE_USAGE_PIXEL_FORMAT_VIEW != 0 {
            caches.count_shapes(|c| c.plain_format_views += 1);
        }
        let pixel_format = u64::from(m.desc.pixel_format);
        if views_without_usage(shape_format, pixel_format) {
            return;
        }
        caches.count_shapes(|c| c.plain_views_refused += 1);
        reims_vgpu_vulkan::answer::count_kind(PLAIN_VIEW_REFUSED);
        #[cfg(feature = "rt")]
        reims_vgpu_vulkan::rt_platform::diag::plain_view_refused(shape_format, pixel_format);
        if !PLAIN_LOGGED.swap(true, Ordering::Relaxed) {
            log::warn!(
                target: "reims_vgpu::graphics", kind:% = PLAIN_VIEW_REFUSED;
                "the texture {:#x} over plane {} of the IOSurface {}: a view in format \
                 {pixel_format} of a plain shape of format {shape_format}, neither it nor its \
                 sRGB sibling, which rt refuses ({PLAIN_VIEW_REFUSED}; the rest are counted)",
                m.obj_id, m.plane, m.iosurface_id
            );
        }
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
            #[cfg(feature = "rt")]
            diag: None,
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

    use reims_vgpu_vulkan::core::platform::types::MtlSize;
    use reims_vgpu_vulkan::core::sync::Mutex;

    use super::super::surface_cache::tests::{
        NO_REACH, blit_with, blits_with, cache_id, cached, copies, download, encoder_with, queues,
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
        r.dec.set_surface_shapes(Arc::new(
            move |device: &<FakePlatform as Platform>::Device, desc: &TextureDesc| {
                kept.lock().push((desc.pixel_format, desc.usage));
                p.own_texture(device, desc)
            },
        ));
        made
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

    /// Command buffer `cb_obj_id` on queue 1, naming [`BLOCK`] written, that writes texture
    /// `obj_id` of `size` by a blit from buffer 10.
    fn write(r: &mut Rig, cb_obj_id: u64, obj_id: u64, size: (u32, u32)) {
        blit_with(r, cb_obj_id, true, write_op(obj_id, size));
    }

    /// A blit op that writes texture `obj_id` of `size` from buffer 10.
    fn write_op(obj_id: u64, size: (u32, u32)) -> (u16, Vec<u8>) {
        use reims_vgpu_graphics::ops::blit::{
            RGFX_OP_BLIT_COPY_BUFFER_TO_TEXTURE, RgfxBlitOpCopyBufferToTexture,
        };
        use zerocopy::IntoBytes as _;
        let (width, height) = (u64::from(size.0), u64::from(size.1));
        let m = RgfxBlitOpCopyBufferToTexture {
            source_buffer_obj_id: 10,
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
        assert_eq!(counts.plain_views_refused, 0);
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

    /// The usage a texture declares picks its shape's kind: one that declares Metal's Unknown
    /// usage (0), `ShaderWrite` or `ShaderAtomic` takes storage, its image made of the Unknown
    /// usage, and writes its shape where it is bound; any other takes a plain shape, its image
    /// made of `ShaderRead | RenderTarget` alone, and leaves it clean; each storage shape made
    /// over the plane while it holds the plain one is counted, whatever its extent. Over a plain
    /// shape a texture in the shape's format's sRGB sibling is a view of it; one in another
    /// format is a view rt refuses, counted, and one that declares `PixelFormatView` is counted
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
        let refused = texture_as_image(&mut r, 27, 70, (32, 8), 0x5);
        assert_eq!([sibling, format_view, refused], [plain; 3]);
        assert_eq!(made.lock().len(), 4);
        let counts = r.dec.caches().counts().shapes;
        assert_eq!((counts.made, counts.found), (4, 3));
        assert_eq!(counts.order_case, 3, "beside the plain one");
        let plain_counts = (counts.plain_views_refused, counts.plain_format_views);
        assert_eq!(plain_counts, (1, 1));
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
        assert_eq!(counts.plain_views_refused, 0, "any view of storage");
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

    /// A new shape owes its fill from the buffer, which the next command buffer that names the
    /// block makes, the cache's own first fill refilling every shape; a shape made later is filled
    /// alone in the next that names the block, with no copy from the pages; a newer generation
    /// fills the buffer from the pages and every shape from the buffer again
    #[test]
    fn a_new_shape_takes_the_buffers_bytes_before_its_first_use() {
        let mut r = shaped();
        let cache = cache_id(&r);
        texture(&mut r, 21, 80, (32, 8));
        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 1, 0x100, &[(BLOCK, 2, false)]);
        let calls = r.p.call_log().since(mark);
        assert_eq!(copies(&calls).len(), 1, "the cache's first fill");
        let first = fills(&calls);
        assert_eq!(first.len(), 1);
        assert_eq!((first[0].0, first[0].1), (cache, 0));
        // A second shape, filled alone.
        texture(&mut r, 22, 80, (16, 4));
        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 1, 0x101, &[(BLOCK, 2, false)]);
        let calls = r.p.call_log().since(mark);
        assert!(copies(&calls).is_empty(), "no copy from the pages");
        let second = fills(&calls);
        assert_eq!(second.len(), 1);
        assert_ne!(second[0].2, first[0].2, "the new shape's image");
        // Nothing owed: nothing filled.
        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 1, 0x102, &[(BLOCK, 2, false)]);
        assert!(fills(&r.p.call_log().since(mark)).is_empty());
        // A newer generation: the buffer, then every shape.
        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 1, 0x103, &[(BLOCK, 3, false)]);
        let calls = r.p.call_log().since(mark);
        assert_eq!(copies(&calls).len(), 1);
        assert_eq!(fills(&calls).len(), 2);
        let counts = r.dec.caches().counts().shapes;
        assert_eq!(counts.fills, 4);
        assert_eq!(counts.filled_bytes, 2 * (32 * 8 * 4 + 16 * 4 * 4));
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

    /// A shape a command buffer writes is dirty from its finish: a download copies it into the
    /// cache's buffer, at its place there, before it copies the buffer into the pages (rule 3),
    /// and once: the next download, nothing written since, writes nothing back. The surface reads
    /// back then, and the next writer's early copy into the pages writes the shape back first
    /// too. A release after a write writes it back before its copy
    #[test]
    fn a_download_or_release_writes_a_dirty_shape_back_before_its_copy() {
        let mut r = shaped();
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

    /// A new shape is filled from the buffer after the dirty shapes are written back into it, in
    /// one command buffer of the transfer queue, so it takes their writes (rule 2), with no copy
    /// from the pages; the shape written back is clean, so a download then copies the buffer,
    /// the write in it, into the pages alone
    #[test]
    fn a_new_shape_takes_a_dirty_shapes_writes() {
        let mut r = shaped();
        let cache = cache_id(&r);
        let written = texture_image(&mut r, 21, 80, (32, 8));
        write(&mut r, 0x100, 21, (32, 8));
        r.p.world().complete_all(CompleteOn::Caller);
        let new = texture_image(&mut r, 22, 80, (16, 4));
        let mark = r.p.call_log().mark();
        use_surfaces(&mut r, 1, 0x101, &[(BLOCK, 0, false)]);
        let calls = r.p.call_log().since(mark);
        assert!(copies(&calls).is_empty(), "no copy from the pages");
        assert_eq!(written_back(&calls), [(written, cache, 0)]);
        assert_eq!(fills(&calls), [(cache, 0, new)]);
        assert!(
            first(&calls, TO_BUFFER) < first(&calls, FILL),
            "written back first"
        );
        let counts = r.dec.caches().counts().shapes;
        assert_eq!((counts.fills_made, counts.written_back), (1, 1));
        r.p.world().complete_all(CompleteOn::Caller);

        let mark = r.p.call_log().mark();
        let downloaded = download(&r, BLOCK);
        let calls = r.p.call_log().since(mark);
        assert!(written_back(&calls).is_empty(), "clean");
        assert_eq!(copies(&calls).len(), 1, "the buffer into the pages");
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(downloaded.load(Ordering::SeqCst));
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
    /// into the cache before the shape's fill. In the default form the block is refused still,
    /// and the kext gives the surface a copy block
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
        assert_eq!(filled.len(), 1, "the cache's first fill");
        assert_eq!(filled[0].1, cache, "into the cache");
        assert_ne!(filled[0].0, cache, "from a staging buffer of its own");
        let shapes = fills(&calls);
        assert_eq!(shapes.len(), 1);
        assert_eq!(shapes[0].0, cache, "the shape from the cache");

        let mut r = rig(8);
        r.p.script().push("newBufferWithDescriptor:", Step::Nil);
        assert_eq!(
            r.pages_cached(BLOCK, &[(1, 2)], 2 * page, true),
            RGFX_STATUS_REFUSED
        );
        assert!(!r.dec.caches().has(BLOCK));
    }
}
