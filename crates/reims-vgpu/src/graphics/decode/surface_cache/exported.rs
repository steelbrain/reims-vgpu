//! A video cache's planes in a surface the platform exports (`shapes`, "Plane shapes").
//!
//! A cache made for the host's video (exportable for NVDEC, a dma-buf for VA-API:
//! [`super::CacheMaker::make`]) of a surface of two planes, a 4:2:0 picture's luma and chroma,
//! takes a surface of the platform's at the first texture the guest makes over one of its planes'
//! whole extent, in a plain shape: each plane an image of `R8Unorm` and `RG8Unorm` (8-bit samples)
//! or `R16Unorm` and `RG16Unorm` (16-bit), of `ShaderRead | RenderTarget | PixelFormatView`, so a
//! guest texture over the plane in another format of the texel's size (an `R8Uint`) is a view of
//! it, in the memory the host's video imports through the surface's handle
//! ([`GpuDevice::new_exported_surface`]). The planes are the cache's plane shapes, which the
//! bound never drops. Each plane holds room for a coded picture's rows below its own (a
//! multiple of 64 luma rows, 32 chroma), for VA-API's decodes into it; the pitch and each plane's
//! start are multiples of 256, as VA-API's imports and CUDA's copies take them.
//!
//! A decode whose luma and chroma textures show the planes whole is given the surface's planes
//! ([`Caches::surface_targets`]), and its picture lands there, by the hardware or by a copy into
//! the plane images: its command buffer leaves the plane shapes holding the picture and every
//! other shape owing a fill ([`owe_video`]). The cache's buffer stays a copy of its own, so the
//! plane shapes are dirty after such a write, and its copies out and the other shapes' fills take
//! the picture from them (rule 3); a command buffer of decodes alone into the surface writes
//! nothing back first, as it writes the planes whole.
//!
//! A platform that makes no such surface (the default, and one that exports no such memory)
//! answers `None`, counted, and the cache's shapes are made as any other's.

use std::sync::Arc;

#[cfg(unix)]
use reims_vgpu_video::core::device_memory::MemoryLife;
use reims_vgpu_vulkan::core::platform::Platform;
#[cfg(unix)]
use reims_vgpu_vulkan::core::platform::device::GpuDevice as _;
#[cfg(unix)]
use reims_vgpu_vulkan::core::platform::resource::{
    ExportedSurface, SurfaceDesc, SurfaceHandle, SurfaceMemory, SurfacePlaneDesc,
};
#[cfg(unix)]
use reims_vgpu_vulkan::core::platform::Handle as _;
use reims_vgpu_vulkan::core::platform::values::PagesPlane;
#[cfg(unix)]
use reims_vgpu_vulkan::core::raise::refusals;

#[cfg(unix)]
use super::super::video::SurfaceTarget;
#[cfg(unix)]
use super::shapes::Held;
use super::shapes::{ShapeCounts, ShapeKey, ShapeTable};
#[cfg(unix)]
use super::{Caches, ON, SurfaceCaches};

/// While a video cache's buffer is a copy of its own, apart from its surface's memory: a picture
/// that lands in the surface leaves its plane shapes dirty ([`owe_video`]).
const BUFFER_APART: bool = true;

/// `MTLPixelFormatR8Unorm`, `RG8Unorm`, `R16Unorm` and `RG16Unorm`: a plane image's format, by
/// the bytes of a luma sample.
const PLANE_FORMATS: [(u64, [u64; 2]); 2] = [(1, [10, 30]), (2, [20, 60])];

/// A plane image's usage: `ShaderRead | RenderTarget | PixelFormatView`.
const PLANE_USAGE: u64 = 0x1 | 0x4 | 0x10;

/// The pitch, and each plane's start after the first, a multiple of this many bytes: VA-API's
/// imports of a surface's planes take them on 256, CUDA's copies on any.
const PLANE_ALIGNMENT: u64 = 256;

/// The coded rows a plane holds: its rows rounded up to these, luma's and chroma's.
const CODED_ROWS: [u64; 2] = [64, 32];

/// How a cache made for the host's video shares its memory, as its make made it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum VideoMemory {
    /// Exportable, for CUDA to import (NVDEC's landing).
    OpaqueFd,
    /// A dma-buf, for VA-API to decode into.
    DmaBuf,
}

impl VideoMemory {
    /// The kind a make answered: exportable, else a dma-buf; `None` for neither.
    pub(super) fn of(exportable: bool, dma_buf: bool) -> Option<Self> {
        match (exportable, dma_buf) {
            (true, _) => Some(Self::OpaqueFd),
            (false, true) => Some(Self::DmaBuf),
            (false, false) => None,
        }
    }
}

/// A cache's surface (module doc): what it may take, and what it took.
pub(super) struct CacheSurface<P: Platform> {
    /// Where the cache was made for the host's video with a surface of two planes: its memory's
    /// kind and the planes' layout in the surface's bytes.
    video: Option<(VideoMemory, [PagesPlane; 2])>,
    /// A surface was asked of the platform ([`Self::ask`]), once a cache.
    asked: bool,
    /// The surface made.
    #[cfg(unix)]
    made: Option<PlaneSurface<P>>,
    /// A decode of the command buffer being walked was given the surface's planes
    /// ([`Caches::surface_targets`]); taken at its finish ([`take_landed`]).
    landed: bool,
    /// A decode given the planes answered no picture ([`skipped`]): the command buffer wrote
    /// nothing of the cache.
    skipped: bool,
    #[cfg(not(unix))]
    _platform: std::marker::PhantomData<fn() -> P>,
}

impl<P: Platform> CacheSurface<P> {
    /// A cache's, made for the host's video as `video` says (`None` for any other cache), its
    /// surface laid out as `layout` where its ask gave it.
    pub(super) fn new(video: Option<VideoMemory>, layout: Option<&[PagesPlane]>) -> Self {
        let planes = layout.and_then(|planes| <[PagesPlane; 2]>::try_from(planes).ok());
        CacheSurface {
            video: video.zip(planes),
            asked: false,
            #[cfg(unix)]
            made: None,
            landed: false,
            skipped: false,
            #[cfg(not(unix))]
            _platform: std::marker::PhantomData,
        }
    }

    /// Whether the cache has its surface.
    pub(super) fn is_made(&self) -> bool {
        #[cfg(unix)]
        return self.made.is_some();
        #[cfg(not(unix))]
        false
    }
}

/// The plane shapes of the planes `layout` lays out, of `luma_texel`-byte luma samples: each
/// plane's whole extent at its place and pitch, plain; `None` where a plane's rows are no whole
/// texels, or the chroma plane is not half the luma's each way, rounded up.
fn plane_keys(layout: &[PagesPlane; 2], luma_texel: u64) -> Option<[ShapeKey; 2]> {
    let key = |plane: u64| {
        let laid = layout[usize::try_from(plane).ok()?];
        let texel = luma_texel.checked_mul(plane + 1)?;
        let whole = laid.row_bytes > 0 && laid.rows > 0 && laid.row_bytes.is_multiple_of(texel);
        whole.then_some(ShapeKey {
            plane,
            offset: laid.offset,
            bytes_per_row: laid.bytes_per_row,
            width: laid.row_bytes / texel,
            height: laid.rows,
            texel_bytes: texel,
            storage: false,
        })
    };
    let keys = [key(0)?, key(1)?];
    let half = |n: u64| n.div_ceil(2);
    let halved = keys[1].width == half(keys[0].width) && keys[1].height == half(keys[0].height);
    halved.then_some(keys)
}

/// A cache's surface made ([`Caches::plane_shape`]).
#[cfg(unix)]
pub(super) struct PlaneSurface<P: Platform> {
    /// Each plane's image, luma's then chroma's: the cache's plane shapes'.
    planes: [P::Texture; 2],
    /// The memory they lie in.
    memory: Arc<dyn SurfaceMemory>,
    /// The surface's life: what the host's video makes of its memory (an import) is kept by its
    /// identity and goes as the cache does.
    life: MemoryLife,
    /// The rows each plane holds, its own and the room below them.
    rows: [u64; 2],
}

/// What a surface is asked of ([`CacheSurface::ask`]): its handle, each plane's description,
/// and the plane shapes they are.
#[cfg(unix)]
pub(super) struct SurfaceAsk {
    handle: SurfaceHandle,
    planes: [SurfacePlaneDesc; 2],
    keys: [ShapeKey; 2],
}

#[cfg(unix)]
impl SurfaceAsk {
    /// The description the platform makes the surface of.
    fn desc(&self) -> SurfaceDesc<'_> {
        SurfaceDesc {
            handle: self.handle,
            pitch_multiple: PLANE_ALIGNMENT,
            start_multiple: PLANE_ALIGNMENT,
            planes: &self.planes,
        }
    }
}

#[cfg(unix)]
impl<P: Platform> CacheSurface<P> {
    /// The surface to ask where `key`, a shape the cache's `shapes` hold none of, is a plane
    /// shape of the cache's surface (module doc), and no shape of either plane is held: asked
    /// once a cache. `None` otherwise.
    fn ask(&mut self, key: &ShapeKey, shapes: &ShapeTable<P::Texture>) -> Option<SurfaceAsk> {
        if self.asked || key.storage {
            return None;
        }
        let (memory, layout) = self.video?;
        let luma_texel = match key.plane {
            0 => key.texel_bytes,
            1 if key.texel_bytes.is_multiple_of(2) => key.texel_bytes / 2,
            _ => return None,
        };
        let (_, formats) = PLANE_FORMATS.iter().find(|(t, _)| *t == luma_texel)?;
        let keys = plane_keys(&layout, luma_texel)?;
        if keys.get(usize::try_from(key.plane).ok()?) != Some(key) {
            return None;
        }
        if (shapes.iter()).any(|s| keys.iter().any(|k| s.key.same_plane(k))) {
            return None;
        }
        self.asked = true;
        let plane = |n: usize| {
            let key = keys[n];
            let rows = key.height.next_multiple_of(CODED_ROWS[n]);
            SurfacePlaneDesc {
                pixel_format: formats[n],
                width: key.width,
                height: key.height,
                usage: PLANE_USAGE,
                rows_below: u32::try_from(rows - key.height).unwrap_or(0),
            }
        };
        Some(SurfaceAsk {
            handle: match memory {
                VideoMemory::OpaqueFd => SurfaceHandle::OpaqueFd,
                VideoMemory::DmaBuf => SurfaceHandle::DmaBuf,
            },
            planes: [plane(0), plane(1)],
            keys,
        })
    }

    /// The targets of a decode into the surface's planes, where the cache has its surface:
    /// counted as landed for the command buffer being walked.
    fn targets(&mut self) -> Option<[SurfaceTarget<P>; 2]> {
        let made = self.made.as_ref()?;
        let layout = made.memory.layout();
        let identity = made.life.identity();
        let target = |n: usize| {
            let laid = layout.get(n)?;
            Some(SurfaceTarget {
                memory: Arc::clone(&made.memory),
                identity: identity.clone(),
                texture: made.planes[n].clone_ref(),
                offset: laid.offset,
                bytes_per_row: laid.row_pitch,
                rows: made.rows[n],
            })
        };
        let targets = [target(0)?, target(1)?];
        self.landed = true;
        self.skipped = false;
        Some(targets)
    }
}

/// A video command buffer wrote the cache whose shapes are `shapes` and surface `surface`
/// ([`super::SurfaceCaches::unseen_writes`]): where its decodes landed in the surface
/// (`landed`, [`take_landed`]), every shape but the planes owes a fill, the planes holding the
/// picture ([`ShapeTable::owe_but_planes`]); else every shape owes one, the buffer holding it
/// ([`ShapeTable::owe_all`]).
pub(super) fn owe_video<T>(shapes: &mut ShapeTable<T>, landed: bool, counts: &mut ShapeCounts) {
    if landed {
        shapes.owe_but_planes(BUFFER_APART, counts);
    } else {
        shapes.owe_all(counts);
    }
}

/// Whether a decode of the command buffer just walked landed in `surface`, taken: `None` where
/// its decodes given the surface's planes answered no picture, so it wrote nothing of the cache
/// ([`skipped`]).
pub(super) fn take_landed<P: Platform>(surface: &mut CacheSurface<P>) -> Option<bool> {
    let landed = std::mem::take(&mut surface.landed);
    if std::mem::take(&mut surface.skipped) {
        return None;
    }
    Some(landed)
}

/// A decode into the cache whose surface is `surface` answered no picture (`picture` false):
/// where it was given the surface's planes, nothing landed there ([`take_landed`]).
pub(super) fn skipped<P: Platform>(surface: &mut CacheSurface<P>, picture: bool) {
    if !picture && surface.landed {
        surface.landed = false;
        surface.skipped = true;
    }
}

/// Whether a command buffer whose decodes write the planes of the cache whose surface is
/// `surface` whole (`decoded`) writes them into the surface: the plane shapes' writes the buffer
/// lacks are not copied into it first, as the picture replaces them (module doc).
pub(super) fn decodes_into<P: Platform>(surface: &CacheSurface<P>, decoded: bool) -> bool {
    decoded && surface.is_made()
}

/// The log's line of the video caches' planes and the fills a video write owed, from the shapes'
/// counts `sh`; `None` where nothing of either was counted.
pub(crate) fn planes_line(sh: &ShapeCounts) -> Option<String> {
    let any = sh.surfaces + sh.surfaces_refused + sh.planes_landed + sh.video_fills > 0;
    any.then(|| {
        format!(
            "IOSurface device caches' video planes: {} surfaces made ({} plane shapes), {} the \
             platform made none of, {} caches holding one at once at most; {} video command \
             buffers landed in them; of the fills, {} owed by a video write ({} MiB)",
            sh.surfaces,
            sh.planes_made,
            sh.surfaces_refused,
            sh.surfaces_live_most,
            sh.planes_landed,
            sh.video_fills,
            sh.video_filled_bytes >> 20
        )
    })
}

#[cfg(unix)]
impl<P: Platform> SurfaceCaches<P> {
    /// [`Caches::plane_shape`]'s ask, under the caches' lock: the surface to ask of the block at
    /// `block`'s cache for the shape `key` ([`CacheSurface::ask`]).
    fn surface_ask(&mut self, block: u64, key: &ShapeKey) -> Option<SurfaceAsk> {
        let cache = self.by_block.get_mut(&block)?;
        cache.surface.ask(key, &cache.shapes)
    }

    /// Adds `made`, the surface `ask` asked, to the block at `block`'s cache, its planes its
    /// plane shapes, each owing a fill from the buffer: a reference to the shape `key` and its
    /// image. Where the cache went, or holds a shape of either plane, since the ask, `made` is
    /// not taken. What goes (`made`'s where not taken, and the shapes past the bound), for the
    /// caller to let go of with the lock let go of.
    fn add_planes(
        &mut self,
        block: u64,
        key: &ShapeKey,
        ask: &SurfaceAsk,
        made: ExportedSurface<P>,
    ) -> (Option<Held<P::Texture>>, Gone<P>) {
        let ExportedSurface { planes, memory } = made;
        let unplaced = |planes: Vec<P::Texture>, memory| (None, (planes, Some(memory)));
        let Some(cache) = self.by_block.get_mut(&block) else {
            return unplaced(planes, memory);
        };
        let taken = cache.surface.asked && !cache.surface.is_made();
        let held = (cache.shapes.iter()).any(|s| ask.keys.iter().any(|k| s.key.same_plane(k)));
        if !taken || held || memory.layout().len() != 2 {
            return unplaced(planes, memory);
        }
        let Ok(images) = <[P::Texture; 2]>::try_from(planes) else {
            return (None, (Vec::new(), Some(memory)));
        };
        self.shape_clock += 1;
        let now = self.shape_clock;
        let mut found = None;
        let mut bytes = 0;
        for (plane, image) in ask.keys.iter().zip(&images) {
            let (reference, _) = cache.shapes.insert_plane(*plane, image.clone_ref(), true, now);
            if plane == key {
                found = Some((reference, image.clone_ref()));
            }
            bytes += plane.bytes();
        }
        let rows = ask.planes.map(|p| p.height + u64::from(p.rows_below));
        cache.surface.made = Some(PlaneSurface {
            planes: images,
            memory,
            life: MemoryLife::new(),
            rows,
        });
        let counts = &mut self.counts.shapes;
        counts.made += 2;
        counts.made_bytes += bytes;
        counts.planes_made += 2;
        counts.surfaces += 1;
        let live = (self.by_block.values()).filter(|c| c.surface.is_made()).count();
        let counts = &mut self.counts.shapes;
        counts.surfaces_live_most = counts.surfaces_live_most.max(live.try_into().unwrap_or(0));
        self.hold_bytes(bytes);
        let gone = self.sweep_shapes(block);
        self.note_shape_bound();
        (found, (gone, None))
    }
}

/// What [`SurfaceCaches::add_planes`] lets go of: images, and a surface's memory not taken.
#[cfg(unix)]
type Gone<P> = (Vec<<P as Platform>::Texture>, Option<Arc<dyn SurfaceMemory>>);

#[cfg(unix)]
impl<P: Platform> Caches<P> {
    /// The plane shape `key` of the block at `block`'s cache, where `key` asks its surface's
    /// (module doc): the surface made of both planes with the lock let go of, its planes added as
    /// the cache's plane shapes ([`SurfaceCaches::add_planes`]), and a reference to `key`'s and
    /// its image. `None` where the cache takes no surface for `key`, or the platform made none
    /// (counted), for the caller to make the shape's image of its own.
    pub(super) fn plane_shape(
        &self,
        device: &P::Device,
        block: u64,
        key: &ShapeKey,
    ) -> Option<Held<P::Texture>> {
        let ask = self.state.lock().surface_ask(block, key)?;
        let made = match device.new_exported_surface(ON, &ask.desc()) {
            Ok(made) => made.filter(|m| m.planes.len() == 2),
            Err(raise) => {
                log::warn!(
                    target: "reims_vgpu::graphics", kind:% = raise;
                    "the IOSurface block at {block:#x}: the platform raised at its video planes' \
                     surface ({raise:?})"
                );
                None
            }
        };
        let Some(made) = made else {
            self.state.lock().counts.shapes.surfaces_refused += 1;
            return None;
        };
        let (found, gone) = self.state.lock().add_planes(block, key, &ask, made);
        if let Err(refusal) = refusals::refusing(move || drop(gone)) {
            log::warn!(
                target: "reims_vgpu::graphics", kind:% = refusal;
                "the IOSurface block at {block:#x}'s video planes or shapes past their bound were \
                 let go of, and a release was refused: {refusal}"
            );
        }
        found
    }

    /// The planes of the block at `block`'s surface (module doc), as the targets of a decode that
    /// shows them whole, where its cache has one: the command buffer being walked lands its
    /// picture there ([`owe_video`]). `None` where it has none.
    pub(crate) fn surface_targets(&self, block: u64) -> Option<[SurfaceTarget<P>; 2]> {
        let mut state = self.state.lock();
        state.by_block.get_mut(&block)?.surface.targets()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// NV12's planes at 1920x1080, rows 2048 bytes apart, chroma from byte 0x220000.
    fn nv12() -> [PagesPlane; 2] {
        [
            PagesPlane {
                offset: 0,
                bytes_per_row: 2048,
                row_bytes: 1920,
                rows: 1080,
            },
            PagesPlane {
                offset: 0x22_0000,
                bytes_per_row: 2048,
                row_bytes: 1920,
                rows: 540,
            },
        ]
    }

    #[test]
    fn a_video_cache_s_plane_keys_are_its_planes_whole() {
        let keys = plane_keys(&nv12(), 1).expect("NV12's planes");
        assert_eq!((keys[0].width, keys[0].height, keys[0].texel_bytes), (1920, 1080, 1));
        assert_eq!((keys[1].width, keys[1].height, keys[1].texel_bytes), (960, 540, 2));
        assert_eq!((keys[1].offset, keys[1].bytes_per_row), (0x22_0000, 2048));
        assert!(!keys[0].storage && !keys[1].storage);
        // 16-bit samples: P010's planes of the same bytes are 960 pixels wide.
        let wide = plane_keys(&nv12(), 2).expect("16-bit planes");
        assert_eq!((wide[0].width, wide[1].width), (960, 480));
        // A chroma plane that is not half the luma's is no 4:2:0 picture's.
        let mut odd = nv12();
        odd[1].rows = 1080;
        assert!(plane_keys(&odd, 1).is_none());
    }

    #[test]
    fn a_landed_decode_leaves_the_planes_holding_it() {
        use reims_vgpu_vulkan::core::testkit::FakePlatform;

        let mut surface = CacheSurface::<FakePlatform>::new(Some(VideoMemory::DmaBuf), None);
        assert_eq!(take_landed(&mut surface), Some(false));
        surface.landed = true;
        assert_eq!(take_landed(&mut surface), Some(true));
        assert_eq!(take_landed(&mut surface), Some(false));
        // A decode given the planes that answered no picture wrote nothing of the cache.
        surface.landed = true;
        skipped(&mut surface, false);
        assert_eq!(take_landed(&mut surface), None);
        assert_eq!(take_landed(&mut surface), Some(false));
        // A cache with no surface takes none of it.
        assert!(!decodes_into(&surface, true));
    }
}
