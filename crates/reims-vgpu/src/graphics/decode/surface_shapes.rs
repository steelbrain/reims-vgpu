//! `TEXTURE_FROM_IOSURFACE` over a device cache in the per-shape form (`surface_cache::shapes`):
//! the guest's texture is a view, at the guest's resource index and in the guest's format, of
//! its shape's image, which the cache keeps for every texture of that shape over the surface.
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
//! refused ([`UNCACHED_COPIED`]), so the kext gives the surface a copy block.

use std::sync::atomic::{AtomicBool, Ordering};

use reims_vgpu_graphics::wire::{RGFX_STATUS_REFUSED, RgfxTextureFromIOSurface};
use reims_vgpu_vulkan::core::platform::Platform;
use reims_vgpu_vulkan::core::platform::resource::{Buffer as _, Texture as _};
use reims_vgpu_vulkan::core::platform::types::{NsRange, PixelFormatInfo};
use reims_vgpu_vulkan::core::platform::values::TextureDesc;
use reims_vgpu_vulkan::core::raise::RaiseResult;

use super::surface_cache::STORAGE_PRIVATE;
use super::surface_cache::ShapeMaker;
use super::surface_cache::shapes::{self, ShapeCounts, ShapeKey, ShapeUse};
use super::{GraphicsDecoder, ObjHandle, TEXTURE_USAGE_WRITES, texture_desc};

/// `MTLTextureType2D`.
const TEXTURE_TYPE_2D: u64 = 2;

/// The degradation code of a block refused as uncached in the per-shape form, which the kext
/// gives a copy block instead (module doc).
pub(super) const UNCACHED_COPIED: &str = "rt-surface-uncached-copied";

/// Whether a block was refused as uncached ([`UNCACHED_COPIED`]): its line is logged the first
/// time alone, the rest counted (docs/conventions.md).
static UNCACHED_LOGGED: AtomicBool = AtomicBool::new(false);

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
        // The image: the guest's format and size, Private, of Metal's Unknown usage, which takes
        // every use and a view in any format of the texel's size, at no index of the guest's.
        let image = TextureDesc {
            texture_type: TEXTURE_TYPE_2D,
            pixel_format,
            width: key.width,
            height: key.height,
            usage: 0,
            resource_options: Some(STORAGE_PRIVATE),
            ..TextureDesc::default()
        };
        let Some((reference, image)) = caches.shape(device, block, key, &image) else {
            return unshaped(|_| {});
        };
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
        let shape = ShapeUse {
            block,
            writes: usage_writes(m),
            reference,
        };
        Ok(Some((view, shape)))
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
            writes: usage_writes(m),
            reference,
        })
    }
}

/// Whether the texture `m` asks may be written where it is bound: its usage is Unknown or holds
/// `ShaderWrite`.
fn usage_writes(m: &RgfxTextureFromIOSurface) -> bool {
    m.desc.usage == 0 || u64::from(m.desc.usage) & TEXTURE_USAGE_WRITES != 0
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

    use super::super::surface_cache::tests::{
        NO_REACH, blit_with, cache_id, cached, copies, download, queues, reading, release,
        surface_texture, surface_texture_as, surface_textures, use_surfaces,
    };
    use super::super::surface_pages::tests::{BLOCK, Rig, one_plane, rig};
    use super::*;
    use crate::platform::ApiPlatform as _;

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

    /// The rig's decoder takes the per-shape form, each image the platform's own texture.
    fn take_shapes(r: &Rig) {
        let p = Arc::clone(&r.p);
        r.dec.set_surface_shapes(Arc::new(
            move |device: &<FakePlatform as Platform>::Device, desc: &TextureDesc| {
                p.own_texture(device, desc)
            },
        ));
    }

    /// [`cached`]'s rig in the per-shape form, its texture resource ids held, with buffer 10 over
    /// the block's two pages.
    fn shaped() -> Rig {
        let mut r = rig(8);
        let page = r.page;
        take_shapes(&r);
        r.dec.hold_resource_ids(0, 1);
        assert_eq!(
            r.pages_cached(BLOCK, &[(1, 2)], 2 * page, true),
            RGFX_STATUS_OK
        );
        queues(&mut r);
        assert_eq!(r.buffer(10, BLOCK, 2 * page), Ok(()));
        r
    }

    /// Texture `obj_id` over buffer 10, of `format` and `size`.
    fn texture(r: &mut Rig, obj_id: u64, format: u32, size: (u32, u32)) {
        let message = surface_texture(obj_id, 10, format, size);
        assert_eq!(r.dec.decode(&r.device, &message).unwrap(), Ok(()));
    }

    /// [`texture`], and the image of its shape, which its view is of.
    fn texture_image(r: &mut Rig, obj_id: u64, format: u32, size: (u32, u32)) -> u64 {
        let mark = r.p.call_log().mark();
        texture(r, obj_id, format, size);
        views(&r.p.call_log().since(mark))[0].0
    }

    /// Command buffer `cb_obj_id` on queue 1, naming [`BLOCK`] written, that writes texture
    /// `obj_id` of `size` by a blit from buffer 10.
    fn write(r: &mut Rig, cb_obj_id: u64, obj_id: u64, size: (u32, u32)) {
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
        let op = (RGFX_OP_BLIT_COPY_BUFFER_TO_TEXTURE, m.as_bytes().to_vec());
        blit_with(r, cb_obj_id, true, op);
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
    /// texture of the same shape, in another format of the texel's size, is another view of the
    /// same image; one of another size has an image of its own. Each holds its shape, and writes
    /// it where its usage lets a shader write. In the default form the texture lies over the
    /// buffer, and holds only its shape in the cache's shadow, which makes no image
    #[test]
    fn a_texture_over_a_cache_is_a_view_of_its_shapes_image() {
        let mut r = shaped();
        let mark = r.p.call_log().mark();
        texture(&mut r, 21, 80, (32, 8));
        texture(&mut r, 22, 80, (32, 8));
        texture(&mut r, 23, 70, (32, 8));
        texture(&mut r, 24, 80, (16, 8));
        let message = surface_texture_as(25, 10, 80, (32, 8), (0x2, 0));
        assert_eq!(r.dec.decode(&r.device, &message).unwrap(), Ok(()));
        let calls = r.p.call_log().since(mark);
        assert!(surface_textures(&calls).is_empty(), "none over the buffer");
        assert_eq!(calls.iter().filter(|c| c.sel == IMAGE).count(), 2);
        let seen = views(&calls);
        let at = |n: u64| (seen[0].0, if n == 23 { 70 } else { 80 }, 0x40 + n);
        assert_eq!(seen.len(), 5);
        assert_eq!(seen[..3], [at(21), at(22), at(23)]);
        assert_ne!(seen[3].0, seen[0].0, "another shape, another image");
        assert_eq!((seen[3].1, seen[3].2), (80, 0x40 + 24));
        assert_eq!(seen[4], at(25));
        let counts = r.dec.caches().counts().shapes;
        assert_eq!((counts.made, counts.found, counts.unshaped), (2, 3, 0));
        assert_eq!(counts.made_bytes, 32 * 8 * 4 + 16 * 8 * 4);
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
}
