//! A video cache's planes in a surface the platform exports (`shapes`, "Plane shapes").
//!
//! A cache made for the host's video (exportable for NVDEC, a dma-buf for VA-API:
//! [`super::CacheMaker::make`]) of a surface of two planes, a 4:2:0 picture's luma and chroma,
//! takes a surface of the platform's at the first texture the guest makes over one of its planes'
//! whole extent, in a plain shape: each plane an image of `R8Unorm` and `RG8Unorm` (8-bit samples)
//! or `R16Unorm` and `RG16Unorm` (16-bit), of `ShaderRead | RenderTarget | PixelFormatView`, so a
//! guest texture over the plane in another format of the texel's size (an `R8Uint`) is a view of
//! it, in the memory the host's video imports through the surface's handle
//! (`GpuDevice::new_exported_surface`, on Unix hosts). The planes are the cache's plane shapes,
//! which the bound never drops. Each plane holds room for a coded picture's rows below its own (a
//! multiple of 64 luma rows, 32 chroma), for VA-API's decodes into it; the pitch and each plane's
//! start are multiples of 256, as VA-API's imports and CUDA's copies take them.
//!
//! The surface's memory holds undefined bytes at its make, so the planes take their fill from the
//! cache's buffer at once, on the transfer queue, as their texture is made
//! (`SurfaceCaches::fill_planes_at_make`), where no command buffer begun and not finished uses
//! the cache and no copy of it waits, as a placed image's fill is made (`shapes`, "Placed
//! images"): the thread that makes the texture encodes and commits the fill and waits for
//! nothing. Where the cache's buffer holds no memory yet (`super`'s "Deferred memory"), the
//! fill clears each plane to the zeros it would read, a render pass of its own with no source,
//! so the buffer's memory stays unmade (`shapes`, "Zeros"). A decode's begin then
//! owes the planes no fill. The host's video writes them outside the platform's queues, so the
//! decode op waits on its thread for the cache's last fill (`Caches::wait_filled`), which that
//! fill, made ahead of it, has as a rule done by then; every command buffer that uses the cache
//! after it waits for it on the GPU. A busy cache's planes owe their fill to its next use.
//!
//! A decode whose luma and chroma textures show the planes whole is given the surface's planes
//! (`Caches::surface_targets`), and its picture lands there, by the hardware or by a copy into
//! the plane images: its command buffer leaves the plane shapes holding the picture and every
//! other shape owing a fill ([`owe_video`]). The cache's buffer stays a copy of its own, so the
//! plane shapes are dirty after such a write, and its copies out and the other shapes' fills take
//! the picture from them (rule 3); a command buffer of decodes alone into the surface writes
//! nothing back first, as it writes the planes whole.
//!
//! A platform that makes no such surface (the default, and one that exports no such memory)
//! answers `None`, counted, and the cache's shapes are made as any other's. A surface it makes in
//! its copy form (`SurfaceMemory::in_place` false), each plane its copy that it moves the bytes
//! into and back, is declined: let go of at once, counted (`surfaces_declined`,
//! `video-surface-copy-form-declined`), and the cache's shapes are made as any other's, so the
//! host's video writes the cache's buffer and the shapes take a fill a picture written. Kept,
//! such a surface's planes are copied in for every command buffer that samples them, written or
//! not. **Measured** (2026-11-21, an RX 550, whose driver holds no dma-buf surface in place):
//! before the decline, a macOS guest's walk (m3-max-63386) had rt copy such planes 15,602
//! times, 107.15 GB (rt's over-copies of cause `dedicated`), 2 a present on a scrolling desktop,
//! of 12.44 MB each, a 3840 by 2160 10-bit picture's two planes' sizes; with it (m3-max-63574),
//! the walk declined 137 surfaces and rt made no such copy.
//!
//! The platform decides a surface's form from the device's answers for its description and the
//! images it makes of it, so the description of each surface declined is kept (`CopyForms`, on
//! Unix), and a later cache whose surface would be made of the same one asks for none: it is
//! declined unmade, counted (`surfaces_unasked`, `video-surface-copy-form-unasked`), and its
//! shapes are made as a declined cache's are. Each new video cache otherwise made a surface
//! for the platform to let go of at once: 128 to 137 a walk on the RX 550 (m3-max-63574,
//! m3-max-63824), each an exported block of a picture's planes, about 25 MB at 3840 by 2160
//! 10-bit.
//!
//! On Windows the surface's planes lie over two Direct3D 11 textures that the host's video draws
//! its pictures into (`graphics::drawn`), made at the ask and imported by the platform
//! (`GpuDevice::import_d3d11_surface`), each plane the size of its plane shape with no rows of
//! room below it: a decode's targets are then the planes' images and those textures
//! (`Caches::drawn_targets`).
//!
//! The textures' bytes are undefined at their make, and the plane shapes owe a fill from the
//! buffer, as any new shape does. A decode that takes a new cache's first fill as met, the
//! buffer holding its make's zeros (`SurfaceCaches::take_decoded`), takes the plane shapes'
//! fills as met too (`take_drawn_fill`), and its command buffer waits for no fill: the
//! textures owe zeros instead (`graphics::drawn`), which the picture's draw writes wherever it
//! does not write the picture, as that fill would have left them, so a picture smaller than its
//! planes leaves the texels past it zero, as the module doc of `super` says of a picture that
//! lands short of the planes. A fill of the plane shapes made since leaves the textures owing
//! nothing (`planes_filled`), and a command buffer whose decode answered nothing over them
//! has them cleared to zeros at its finish (`SurfaceCaches::undrawn`).
//!
//! **Measured** (2026-11-19, Intel UHD 630, a macOS guest's 4K H.264 and HEVC playback, its
//! pictures held and released), before the fills were taken: the boot made 1977 such surfaces,
//! each for a first fill a decode took as met, and its fills of the shapes alone filled 4808
//! shapes of 23608 MiB, nearly all of them those surfaces' 3954 plane shapes, in 2522 fills on
//! the transfer queue; the caches' fills took 2502 ms of the GPU's time, and a held picture's
//! decode op waited 5.0 to 11.5 ms for its planes (each pass's mean), which waited for that fill.
//!
//! Such a surface is made on the ring-0 thread that carries the guest's texture make out, which
//! the guest's next messages wait for, and a player that holds each picture it decoded takes a
//! new IOSurface, and so a new surface, for every picture. So each ask of a surface takes a spare
//! made ahead for the same ask, where one is kept, and hands the makes of the next ones over to a
//! thread of the spares' own, `reims-vgpu.graphics-drawn-spares-<i>` (`spares`,
//! `DrawnSpares`), which nothing waits for; an ask that finds none makes its surface itself, as
//! before. An ask keeps as many spares, kept and being made, as the guest had decodes in flight
//! at its last decode on the device's video (`Caches::saw_decodes_in_flight`), at least one
//! and at most `spares::MOST_SPARES` of every ask, by the plane shapes, their formats and the
//! device's recoveries it was made at, so none made before a recovery is taken after it.
//! An ask that finds none while the spares' thread is making its spare waits for that make
//! (`spares`). Counted: `video-drawn-spares-used`, the asks that took a spare;
//! `video-drawn-spares-missed`, those that made their own; `video-drawn-spares-waited`, those
//! that waited for a make, with `video-drawn-spares-wait-ns` their nanoseconds waited and
//! `video-drawn-spares-wait-max-ns` the longest wait; `video-drawn-spares-made`, the spares made;
//! and `video-drawn-spares-dropped`, those let go of untaken.

use std::sync::Arc;

use reims_vgpu_video::core::device_memory::MemoryLife;
use reims_vgpu_vulkan::core::platform::Handle as _;
use reims_vgpu_vulkan::core::platform::Platform;
use reims_vgpu_vulkan::core::platform::device::GpuDevice as _;
#[cfg(windows)]
use reims_vgpu_vulkan::core::platform::resource::{
    D3d11PlaneDesc, ImportedSurface, ImportedSurfaceMemory,
};
#[cfg(unix)]
use reims_vgpu_vulkan::core::platform::resource::{
    ExportedSurface, SurfaceDesc, SurfaceHandle, SurfaceMemory, SurfacePlaneDesc,
};
use reims_vgpu_vulkan::core::platform::values::PagesPlane;
use reims_vgpu_vulkan::core::raise::refusals;

#[cfg(windows)]
use parking_lot::RwLock;
#[cfg(windows)]
use reims_vgpu_vulkan::core::raise::Raise;

#[cfg(windows)]
use super::super::super::drawn::{DrawnPlanes, DrawnTarget};
#[cfg(unix)]
use super::super::video::SurfaceTarget;
#[cfg(windows)]
use super::Now;
use super::shapes::{Held, ShapeCounts, ShapeKey, ShapeTable};
#[cfg(windows)]
use super::spares::{SpareCodes, Spares};
use super::{Caches, ON, SurfaceCaches};

/// While a video cache's buffer is a copy of its own, apart from its surface's memory: a picture
/// that lands in the surface leaves its plane shapes dirty ([`owe_video`]).
const BUFFER_APART: bool = true;

/// `MTLPixelFormatR8Unorm`, `RG8Unorm`, `R16Unorm` and `RG16Unorm`: a plane image's format, by
/// the bytes of a luma sample.
const PLANE_FORMATS: [(u64, [u64; 2]); 2] = [(1, [10, 30]), (2, [20, 60])];

/// `MTLPixelFormatR8Unorm`, `R16Uint`, `R16Unorm` and `R32Uint`: the element format of the
/// Direct3D 11 texture under a plane image on Windows (the module doc), by the bytes of a luma
/// sample, luma's then chroma's, whose element is one packed pair.
#[cfg(windows)]
const DRAWN_FORMATS: [(u64, [u64; 2]); 2] = [(1, [10, 23]), (2, [20, 53])];

/// A plane image's usage: `ShaderRead | RenderTarget | PixelFormatView`.
const PLANE_USAGE: u64 = 0x1 | 0x4 | 0x10;

/// Each plane's start after the first a multiple of this many bytes, and a dma-buf's pitch:
/// VA-API's imports of a surface's planes take them on 256. An exportable surface's pitch is a
/// multiple of 1 (its planes' texels alone): CUDA's copies take any pitch, and the platform makes
/// such a surface in place only at its driver's own pitch for the images. On an RTX 3090 that
/// pitch was off 256 for 1920 by 1080 NV12: at a pitch multiple of 256, a boot's first such
/// surface was made in the copy form for its first plane's pitch, and 60 of its 74 surfaces in
/// all (m3-max-53703, 2026-11-19).
#[cfg(unix)]
const PLANE_ALIGNMENT: u64 = 256;

/// The coded rows a plane holds: its rows rounded up to these, luma's and chroma's.
#[cfg(unix)]
const CODED_ROWS: [u64; 2] = [64, 32];

/// How a cache made for the host's video shares its memory, as its make made it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum VideoMemory {
    /// Exportable, for CUDA to import (NVDEC's landing).
    OpaqueFd,
    /// A dma-buf, for VA-API to decode into.
    DmaBuf,
    /// A buffer as any other's, whose surface's planes lie over Direct3D 11 textures that VA-API's
    /// pictures are drawn into (the module doc).
    #[cfg(windows)]
    Drawn,
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

    /// Whether the host's video writes the cache's memory itself, with no work of the platform's
    /// (CUDA's copies into an exportable cache, VA-API's decodes into a dma-buf), so that its make
    /// zeroes it ([`SurfaceCaches::zeros_of_new`]); a drawn cache's buffer is the platform's
    /// alone, the video's draws writing the Direct3D 11 textures under its planes.
    pub(super) fn written_outside(self) -> bool {
        match self {
            Self::OpaqueFd | Self::DmaBuf => true,
            #[cfg(windows)]
            Self::Drawn => false,
        }
    }
}

/// A cache's surface (module doc): what it may take, and what it took.
pub(super) struct CacheSurface<P: Platform> {
    /// Where the cache was made for the host's video with a surface of two planes: its memory's
    /// kind and the planes' layout in the surface's bytes.
    video: Option<(VideoMemory, [PagesPlane; 2])>,
    /// A surface was asked of the platform ([`Self::plane_keys_to_ask`]), once a cache.
    asked: bool,
    /// The surface made.
    #[cfg(unix)]
    made: Option<PlaneSurface<P>>,
    /// The surface made, over Direct3D 11 textures.
    #[cfg(windows)]
    made: Option<DrawnSurface<P>>,
    /// A decode of the command buffer being walked was given the surface's planes
    /// (`Caches::surface_targets`); taken at its finish ([`take_landed`]).
    landed: bool,
    /// A decode given the planes answered no picture ([`skipped`]): the command buffer wrote
    /// nothing of the cache.
    skipped: bool,
}

impl<P: Platform> CacheSurface<P> {
    /// A cache's, made for the host's video as `video` says (`None` for any other cache), its
    /// surface laid out as `layout` where its ask gave it.
    pub(super) fn new(video: Option<VideoMemory>, layout: Option<&[PagesPlane]>) -> Self {
        let planes = layout.and_then(|planes| <[PagesPlane; 2]>::try_from(planes).ok());
        CacheSurface {
            video: video.zip(planes),
            asked: false,
            made: None,
            landed: false,
            skipped: false,
        }
    }

    /// Whether the cache has its surface.
    pub(super) fn is_made(&self) -> bool {
        self.made.is_some()
    }

    /// Whether the host's video writes the cache's memory itself
    /// ([`VideoMemory::written_outside`]); `false` for a cache not made for it.
    pub(super) fn written_outside(&self) -> bool {
        self.video
            .is_some_and(|(memory, _)| memory.written_outside())
    }

    /// The plane shapes of the surface to ask where `key`, a shape the cache's `shapes` hold none
    /// of, is a plane shape of the cache's surface (module doc), and no shape of either plane is
    /// held: with the cache's memory's kind, the bytes of a luma sample and the planes' image
    /// formats. Asked once a cache; `None` otherwise.
    fn plane_keys_to_ask(
        &mut self,
        key: &ShapeKey,
        shapes: &ShapeTable<P::Texture>,
    ) -> Option<(VideoMemory, [ShapeKey; 2], u64, [u64; 2])> {
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
        Some((memory, keys, luma_texel, *formats))
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
    /// Each plane's clear to zeros, a render pass of its image alone
    /// ([`shapes::zeros_pass`](super::shapes::zeros_pass)), for the host's video to clear a
    /// plane a short picture lands in (`video_host`'s `Landing::writes_only`); `None` where the
    /// platform raised at one.
    clears: Option<[P::RenderPassDescriptor; 2]>,
}

/// What a surface is asked of ([`CacheSurface::ask`]): its handle, each plane's description,
/// and the plane shapes they are.
#[cfg(unix)]
pub(super) struct SurfaceAsk {
    handle: SurfaceHandle,
    planes: [SurfacePlaneDesc; 2],
    keys: [ShapeKey; 2],
}

/// What a surface's description is made of ([`SurfaceAsk::desc`]): its handle's kind, which
/// decides its pitch's and starts' multiples, and each plane's description.
#[cfg(unix)]
type SurfaceShape = (SurfaceHandle, [SurfacePlaneDesc; 2]);

/// The most descriptions [`CopyForms`] keeps: more than the picture sizes a few players decode
/// at once.
#[cfg(unix)]
const COPY_FORMS_KEPT: usize = 16;

/// The descriptions of the surfaces the platform made in its copy form (the module doc), the
/// latest [`COPY_FORMS_KEPT`]: a cache whose surface would be made of one of them asks for none
/// ([`Caches::plane_shape`]).
#[cfg(unix)]
#[derive(Default)]
pub(super) struct CopyForms(std::collections::VecDeque<SurfaceShape>);

#[cfg(unix)]
impl CopyForms {
    /// Whether the platform made a surface of `ask`'s description in its copy form.
    fn holds(&self, ask: &SurfaceAsk) -> bool {
        self.0.contains(&ask.shape())
    }

    /// The platform made a surface of `ask`'s description in its copy form: kept, the oldest
    /// let go of past [`COPY_FORMS_KEPT`].
    fn note(&mut self, ask: &SurfaceAsk) {
        if self.holds(ask) {
            return;
        }
        if self.0.len() >= COPY_FORMS_KEPT {
            self.0.pop_front();
        }
        self.0.push_back(ask.shape());
    }
}

#[cfg(unix)]
impl SurfaceAsk {
    /// What the description [`Self::desc`] is made of.
    fn shape(&self) -> SurfaceShape {
        (self.handle, self.planes)
    }

    /// The description the platform makes the surface of: a dma-buf's pitch on
    /// [`PLANE_ALIGNMENT`], an exportable surface's on its texels alone.
    fn desc(&self) -> SurfaceDesc<'_> {
        SurfaceDesc {
            handle: self.handle,
            pitch_multiple: match self.handle {
                SurfaceHandle::DmaBuf => PLANE_ALIGNMENT,
                SurfaceHandle::OpaqueFd => 1,
            },
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
        let (memory, keys, _, formats) = self.plane_keys_to_ask(key, shapes)?;
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
                clear: (made.clears.as_ref()).map(|clears| clears[n].clone_ref()),
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
/// counts `sh`, and the plane shapes' fills taken as met where any were (`take_drawn_fill`);
/// `None` where nothing of either was counted.
pub(crate) fn planes_line(sh: &ShapeCounts) -> Option<String> {
    let asked = sh.surfaces + sh.surfaces_refused + sh.surfaces_declined + sh.surfaces_unasked;
    let any = asked + sh.planes_landed + sh.video_fills > 0;
    any.then(|| {
        let taken = match sh.drawn_fills_taken {
            0 => String::new(),
            n => format!("; {n} plane shapes' fills taken as met by a decode drawn over zeros"),
        };
        format!(
            "IOSurface device caches' video planes: {} surfaces made ({} plane shapes), {} the \
             platform made none of, {} it made in its copy form declined, {} declined unmade as \
             it made their description so before, {} caches holding one at once at most; {} \
             video command buffers landed in them; of the fills, {} owed by a video write ({} \
             MiB); plane shapes filled {} at their make, {} at a later use, {} surfaces' planes \
             owing theirs past their make as their cache was busy; {} video ops waited on the \
             CPU for a cache's fill not done, the longest {} us{taken}",
            sh.surfaces,
            sh.planes_made,
            sh.surfaces_refused,
            sh.surfaces_declined,
            sh.surfaces_unasked,
            sh.surfaces_live_most,
            sh.planes_landed,
            sh.video_fills,
            sh.video_filled_bytes >> 20,
            sh.planes_filled_at_make,
            sh.plane_fills.saturating_sub(sh.planes_filled_at_make),
            sh.planes_owed_busy,
            sh.video_fill_waits,
            sh.video_fill_wait_max_us
        )
    })
}

/// A decode took the first fill of the cache whose surface is `surface` and shapes `shapes` as
/// met, the buffer holding its make's zeros ([`super::SurfaceCaches::take_decoded`]): where the
/// surface's planes lie over Direct3D 11 textures, the fills its plane shapes owe are taken as
/// met too, and the textures owe zeros instead (module doc); counted in `counts`.
#[cfg(windows)]
pub(super) fn take_drawn_fill<P: Platform>(
    surface: &CacheSurface<P>,
    shapes: &mut ShapeTable<P::Texture>,
    counts: &mut ShapeCounts,
) {
    let Some(made) = &surface.made else {
        return;
    };
    let taken = shapes.take_plane_fills();
    if taken > 0 {
        made.drawn.fill_taken();
        counts.drawn_fills_taken += taken;
    }
}

/// A fill of the plane shapes of the cache whose surface is `surface` from its buffer was made:
/// where the planes lie over Direct3D 11 textures, they owe no zeros (module doc).
#[cfg(windows)]
pub(super) fn planes_filled<P: Platform>(surface: &CacheSurface<P>) {
    if let Some(made) = &surface.made {
        made.drawn.filled();
    }
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
    /// plane shapes, each owing a fill from the buffer, and `clears` each plane's clear: a
    /// reference to the shape `key` and its image. Where the cache went, or holds a shape of either plane, since the ask, `made` is
    /// not taken. What goes (`made`'s where not taken, and the shapes past the bound), for the
    /// caller to let go of with the lock let go of.
    fn add_planes(
        &mut self,
        block: u64,
        key: &ShapeKey,
        ask: &SurfaceAsk,
        made: ExportedSurface<P>,
        clears: Option<[P::RenderPassDescriptor; 2]>,
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
        let now = super::tick(&mut self.shape_clock, self.shape_now.as_ref());
        let mut found = None;
        let mut bytes = 0;
        for (plane, image) in ask.keys.iter().zip(&images) {
            let (reference, _) = cache
                .shapes
                .insert_plane(*plane, image.clone_ref(), true, now);
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
            clears,
        });
        let counts = &mut self.counts.shapes;
        counts.made += 2;
        counts.made_bytes += bytes;
        counts.planes_made += 2;
        counts.surfaces += 1;
        let live = (self.by_block.values())
            .filter(|c| c.surface.is_made())
            .count();
        let counts = &mut self.counts.shapes;
        counts.surfaces_live_most = counts.surfaces_live_most.max(live.try_into().unwrap_or(0));
        self.hold_bytes(bytes);
        let gone = self.sweep_shapes(block);
        self.note_shape_bound();
        (found, (gone, None))
    }
}

#[cfg(unix)]
impl<P: Platform> SurfaceCaches<P> {
    /// The planes just added to the block at `block`'s cache ([`Self::add_planes`]) take their
    /// fill from its buffer at once (the module doc), with the cache's other shapes that owe one
    /// and its dirty shapes written back first ([`Self::fill_shapes`]), or by clears of
    /// `platform`'s while the buffer holds no memory ([`Self::fill_shapes_over`]), where no
    /// command buffer begun and not finished uses the cache and no copy of it waits; else they
    /// owe it to the cache's next use, counted ([`ShapeCounts::planes_owed_busy`]). A fill the
    /// platform raises at is logged, and the planes owe theirs still.
    fn fill_planes_at_make(&mut self, platform: &P, device: &P::Device, block: u64) {
        if self.pending(block, false) > 0 || self.deferring(block) {
            self.counts.shapes.planes_owed_busy += 1;
            return;
        }
        let before = self.counts.shapes.plane_fills;
        // A refusal of a buffer's memory is a begin's to fail its command buffer on, not this.
        let refused = self.memory_refused;
        if let Err(raise) = self.fill_shapes_over(device, block, Some(platform)) {
            log::warn!(
                target: "reims_vgpu::graphics", kind:% = raise;
                "the IOSurface block at {block:#x}: the platform raised at its video planes' fill \
                 at their make ({raise:?}); they owe it to the cache's next use"
            );
        }
        self.memory_refused = refused;
        let filled = self.counts.shapes.plane_fills - before;
        self.counts.shapes.planes_filled_at_make += filled;
    }
}

/// The counts' code of a surface the platform made in its copy form that a cache declined
/// ([`decline`]): in the process's counts line, which the boot reads at its end.
#[cfg(unix)]
const SURFACE_DECLINED: &str = "video-surface-copy-form-declined";

/// Whether the process has logged a declined surface ([`decline`]).
#[cfg(unix)]
static DECLINED_SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Lets go of `made`, a surface the platform made for the block at `block`'s cache in its copy
/// form (the module doc), with no lock held: counted ([`SURFACE_DECLINED`]), the process's first
/// logged, and a refusal of a release logged.
#[cfg(unix)]
fn decline<P: Platform>(block: u64, made: ExportedSurface<P>) {
    reims_vgpu_vulkan::answer::count_kind(SURFACE_DECLINED);
    if !DECLINED_SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
        log::info!(
            target: "reims_vgpu::graphics", kind:% = SURFACE_DECLINED;
            "the IOSurface block at {block:#x}: the platform made its video planes' surface in its \
             copy form, each plane its copy that it moves the bytes into and back; let go of, \
             the cache's shapes are made as any other's and the host's video writes its buffer \
             ({SURFACE_DECLINED}; the rest are counted)"
        );
    }
    if let Err(refusal) = refusals::refusing(move || drop(made)) {
        log::warn!(
            target: "reims_vgpu::graphics", kind:% = refusal;
            "the IOSurface block at {block:#x}'s video planes' surface in the copy form was let \
             go of, and a release was refused: {refusal}"
        );
    }
}

/// The counts' code of a cache whose surface was declined unmade, the platform having made its
/// description in the copy form before ([`CopyForms`]).
#[cfg(unix)]
const SURFACE_UNASKED: &str = "video-surface-copy-form-unasked";

/// Whether the process has logged a surface declined unmade ([`unasked`]).
#[cfg(unix)]
static UNASKED_SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The block at `block`'s cache asks for no surface, the platform having made its description in
/// the copy form before ([`CopyForms`]): counted ([`SURFACE_UNASKED`]), the process's first
/// logged.
#[cfg(unix)]
fn unasked(block: u64) {
    reims_vgpu_vulkan::answer::count_kind(SURFACE_UNASKED);
    if !UNASKED_SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
        log::info!(
            target: "reims_vgpu::graphics", kind:% = SURFACE_UNASKED;
            "the IOSurface block at {block:#x}: the platform made its video planes' surface's \
             description in its copy form before; none asked, the cache's shapes are made as \
             any other's and the host's video writes its buffer ({SURFACE_UNASKED}; the rest are \
             counted)"
        );
    }
}

/// Each of `planes`' clear to zeros ([`PlaneSurface::clears`]), the surface's that the block at
/// `block`'s cache takes; `None`, logged, where `platform` raised at one.
#[cfg(unix)]
fn plane_clears<P: Platform>(
    platform: &P,
    block: u64,
    planes: &[P::Texture],
) -> Option<[P::RenderPassDescriptor; 2]> {
    let [luma, chroma] = planes else {
        return None;
    };
    let made = super::shapes::zeros_pass(platform, luma)
        .and_then(|luma| Ok([luma, super::shapes::zeros_pass(platform, chroma)?]));
    match made {
        Ok(clears) => Some(clears),
        Err(raise) => {
            log::warn!(
                target: "reims_vgpu::graphics", kind:% = raise;
                "the IOSurface block at {block:#x}: the platform raised at its video planes' \
                 clears ({raise:?}); a short picture there fails to land"
            );
            None
        }
    }
}

/// What [`SurfaceCaches::add_planes`] lets go of: images, and a surface's memory not taken.
#[cfg(unix)]
type Gone<P> = (
    Vec<<P as Platform>::Texture>,
    Option<Arc<dyn SurfaceMemory>>,
);

#[cfg(unix)]
impl<P: Platform> Caches<P> {
    /// The plane shape `key` of the block at `block`'s cache, where `key` asks its surface's
    /// (module doc): the surface made of both planes with the lock let go of, its planes added as
    /// the cache's plane shapes ([`SurfaceCaches::add_planes`]), and a reference to `key`'s and
    /// its image. `None` where the cache takes no surface for `key`, or the platform made none
    /// (counted) or made it in its copy form, declined (counted, [`decline`]), or made its
    /// description so before, declined unmade (counted, [`unasked`]), for the caller to make the
    /// shape's image of its own. Where the cache's buffer holds no memory yet, the planes' fill
    /// at their make clears them by `platform`'s render passes
    /// ([`SurfaceCaches::fill_planes_at_make`]), with the buffer's memory left unmade.
    pub(super) fn plane_shape(
        &self,
        platform: &P,
        device: &P::Device,
        block: u64,
        key: &ShapeKey,
    ) -> Option<Held<P::Texture>> {
        let ask = {
            let mut state = self.state.lock();
            let ask = state.surface_ask(block, key)?;
            if state.copy_forms.holds(&ask) {
                state.counts.shapes.surfaces_unasked += 1;
                drop(state);
                unasked(block);
                return None;
            }
            ask
        };
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
        if !made.memory.in_place() {
            decline(block, made);
            let mut state = self.state.lock();
            state.counts.shapes.surfaces_declined += 1;
            state.copy_forms.note(&ask);
            return None;
        }
        let clears = plane_clears(platform, block, &made.planes);
        // The caches' lock, whose let-go commits the planes' fill (`Locked`).
        let (found, gone) = {
            let mut state = self.lock();
            let added = state.add_planes(block, key, &ask, made, clears);
            if added.0.is_some() {
                state.fill_planes_at_make(platform, device, block);
            }
            added
        };
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

/// A cache's surface made over Direct3D 11 textures ([`Caches::plane_shape`], on Windows).
#[cfg(windows)]
pub(super) struct DrawnSurface<P: Platform> {
    /// Each plane's image, luma's then chroma's: the cache's plane shapes'.
    planes: [P::Texture; 2],
    /// The memory they lie in.
    memory: Arc<dyn ImportedSurfaceMemory>,
    /// The Direct3D 11 textures under them, and their draws.
    drawn: Arc<DrawnPlanes>,
    /// The surface's life: its identity goes as the cache does.
    life: MemoryLife,
}

/// What a surface over Direct3D 11 textures is asked of ([`CacheSurface::drawn_ask`]): the
/// plane shapes, each plane's image format and its texture's element format, and whether its
/// samples are 16 bits.
#[cfg(windows)]
#[derive(Clone, PartialEq)]
pub(super) struct DrawnAsk {
    keys: [ShapeKey; 2],
    formats: [u64; 2],
    elements: [u64; 2],
    ten_bit: bool,
}

/// Whether the process has logged a cache's Direct3D 11 textures that were not made.
#[cfg(windows)]
static UNDRAWN_SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The spare surfaces over Direct3D 11 textures made ahead (the module doc), by their ask and the
/// device's recoveries they were made at.
#[cfg(windows)]
pub(super) type DrawnSpares<P> = Spares<(DrawnAsk, u64), (ImportedSurface<P>, DrawnPlanes)>;

/// The kinds [`DrawnSpares`] count (the module doc).
#[cfg(windows)]
static DRAWN_SPARE_CODES: SpareCodes = SpareCodes {
    used: "video-drawn-spares-used",
    missed: "video-drawn-spares-missed",
    waited: "video-drawn-spares-waited",
    wait_ns: "video-drawn-spares-wait-ns",
    wait_max_ns: "video-drawn-spares-wait-max-ns",
    made: "video-drawn-spares-made",
    dropped: "video-drawn-spares-dropped",
};

/// The spare surfaces made ahead for a device's caches (the module doc), each make under the
/// device's reset `gate` held shared, the asks' waits timed by `now`.
#[cfg(windows)]
pub(super) fn drawn_spares<P: Platform>(gate: Arc<RwLock<()>>, now: Now) -> DrawnSpares<P> {
    Spares::new(
        "reims-vgpu.graphics-drawn-spares",
        gate,
        &DRAWN_SPARE_CODES,
        now,
    )
}

/// Why [`DrawnAsk::make`] made no surface.
#[cfg(windows)]
enum Unmade {
    /// The textures were not made, and why.
    Textures(String),
    /// The platform raised at the surface.
    Raised(Raise),
    /// The platform made none, or the shapes are past a texture's size.
    None,
}

#[cfg(windows)]
impl DrawnAsk {
    /// The surface that `device` makes over new Direct3D 11 textures of the plane shapes' sizes,
    /// which `draw` makes, with those textures; why none was made where either makes none.
    fn make<P: Platform>(
        &self,
        device: &P::Device,
        draw: &Arc<reims_vgpu_video::vaapi::PlaneDraw>,
    ) -> Result<(ImportedSurface<P>, DrawnPlanes), Unmade> {
        let width = u32::try_from(self.keys[0].width).map_err(|_| Unmade::None)?;
        let height = u32::try_from(self.keys[0].height).map_err(|_| Unmade::None)?;
        let drawn =
            DrawnPlanes::new(draw, (width, height), self.ten_bit).map_err(Unmade::Textures)?;
        let targets = drawn.targets();
        let handles = [targets.luma_handle(), targets.chroma_handle()];
        let planes: [D3d11PlaneDesc<'_>; 2] = std::array::from_fn(|n| D3d11PlaneDesc {
            handle: handles[n],
            image_format: self.elements[n],
            pixel_format: self.formats[n],
            width: self.keys[n].width,
            height: self.keys[n].height,
            usage: PLANE_USAGE,
        });
        match device.import_d3d11_surface(ON, &planes) {
            Ok(made) => (made.filter(|m| m.planes.len() == 2))
                .map(|made| (made, drawn))
                .ok_or(Unmade::None),
            Err(raise) => Err(Unmade::Raised(raise)),
        }
    }

    /// [`Self::make`] for the block at `block`'s cache: `None` where no surface was made, the
    /// process's first miss of the textures logged, and each raise.
    fn make_for<P: Platform>(
        &self,
        device: &P::Device,
        draw: &Arc<reims_vgpu_video::vaapi::PlaneDraw>,
        block: u64,
    ) -> Option<(ImportedSurface<P>, DrawnPlanes)> {
        match self.make::<P>(device, draw) {
            Ok(made) => Some(made),
            Err(Unmade::Textures(why)) => {
                if !UNDRAWN_SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    log::warn!(
                        target: "reims_vgpu::graphics",
                        "the IOSurface block at {block:#x}: no Direct3D 11 textures for its video \
                         planes ({why}); they stay in images of their own, counted from here as \
                         surfaces the platform made none of"
                    );
                }
                None
            }
            Err(Unmade::Raised(raise)) => {
                log::warn!(
                    target: "reims_vgpu::graphics", kind:% = raise;
                    "the IOSurface block at {block:#x}: the platform raised at its video planes' \
                     surface over Direct3D 11 textures ({raise:?})"
                );
                None
            }
            Err(Unmade::None) => None,
        }
    }
}

#[cfg(windows)]
impl<P: Platform> CacheSurface<P> {
    /// The surface over Direct3D 11 textures to ask where `key` is a plane shape of the cache's
    /// surface ([`Self::plane_keys_to_ask`]): asked once a cache. `None` otherwise.
    fn drawn_ask(&mut self, key: &ShapeKey, shapes: &ShapeTable<P::Texture>) -> Option<DrawnAsk> {
        let (_, keys, luma_texel, formats) = self.plane_keys_to_ask(key, shapes)?;
        let (_, elements) = DRAWN_FORMATS.iter().find(|(t, _)| *t == luma_texel)?;
        Some(DrawnAsk {
            keys,
            formats,
            elements: *elements,
            ten_bit: luma_texel == 2,
        })
    }

    /// The targets of a decode into the surface's planes, where the cache has its surface:
    /// counted as landed for the command buffer being walked.
    fn drawn_targets(&mut self) -> Option<[DrawnTarget<P>; 2]> {
        let made = self.made.as_ref()?;
        let identity = made.life.identity();
        let target = |n: usize| DrawnTarget {
            memory: Arc::clone(&made.memory),
            planes: Arc::clone(&made.drawn),
            identity: identity.clone(),
            texture: made.planes[n].clone_ref(),
        };
        let targets = [target(0), target(1)];
        self.landed = true;
        self.skipped = false;
        Some(targets)
    }
}

#[cfg(windows)]
impl<P: Platform> SurfaceCaches<P> {
    /// The Direct3D 11 textures under the planes of the blocks `counts` names whose caches still
    /// hold their make's zeros where a decode took their first fill as met
    /// ([`super::SurfaceCache::zeros`]) as a command buffer that used them finishes, and that
    /// owe zeros: no decode answered a picture over them, so none is drawn, and the caller
    /// clears them to the zeros a fill would have left before the buffer's commit (module doc).
    pub(super) fn undrawn(&self, counts: &[super::Count]) -> Vec<Arc<DrawnPlanes>> {
        (counts.iter())
            .filter_map(|count| {
                let cache = self.by_block.get(&count.block)?;
                let made = cache.surface.made.as_ref().filter(|_| cache.zeros)?;
                made.drawn.owes_zeros().then(|| Arc::clone(&made.drawn))
            })
            .collect()
    }

    /// [`Caches::plane_shape`]'s ask, under the caches' lock: the surface over Direct3D 11
    /// textures to ask of the block at `block`'s cache for the shape `key`
    /// ([`CacheSurface::drawn_ask`]).
    fn drawn_ask(&mut self, block: u64, key: &ShapeKey) -> Option<DrawnAsk> {
        let cache = self.by_block.get_mut(&block)?;
        cache.surface.drawn_ask(key, &cache.shapes)
    }

    /// Adds `made`, the surface `ask` asked over the textures `drawn`, to the block at `block`'s
    /// cache, its planes its plane shapes, each owing a fill from the buffer: a reference to the
    /// shape `key` and its image. Where the cache went, or holds a shape of either plane, since
    /// the ask, `made` is not taken. What goes (`made` and `drawn` where not taken, and the shapes
    /// past the bound), for the caller to let go of with the lock let go of.
    fn add_drawn_planes(
        &mut self,
        block: u64,
        key: &ShapeKey,
        ask: &DrawnAsk,
        made: ImportedSurface<P>,
        drawn: DrawnPlanes,
    ) -> (Option<Held<P::Texture>>, DrawnGone<P>) {
        let ImportedSurface { planes, memory } = made;
        let Some(cache) = self.by_block.get_mut(&block) else {
            return (None, (planes, Some((memory, drawn))));
        };
        let taken = cache.surface.asked && !cache.surface.is_made();
        let held = (cache.shapes.iter()).any(|s| ask.keys.iter().any(|k| s.key.same_plane(k)));
        if !taken || held {
            return (None, (planes, Some((memory, drawn))));
        }
        let Ok(images) = <[P::Texture; 2]>::try_from(planes) else {
            return (None, (Vec::new(), Some((memory, drawn))));
        };
        let now = super::tick(&mut self.shape_clock, self.shape_now.as_ref());
        let mut found = None;
        let mut bytes = 0;
        for (plane, image) in ask.keys.iter().zip(&images) {
            let (reference, _) = cache
                .shapes
                .insert_plane(*plane, image.clone_ref(), true, now);
            if plane == key {
                found = Some((reference, image.clone_ref()));
            }
            bytes += plane.bytes();
        }
        cache.surface.made = Some(DrawnSurface {
            planes: images,
            memory,
            drawn: Arc::new(drawn),
            life: MemoryLife::new(),
        });
        let counts = &mut self.counts.shapes;
        counts.made += 2;
        counts.made_bytes += bytes;
        counts.planes_made += 2;
        counts.surfaces += 1;
        let live = (self.by_block.values())
            .filter(|c| c.surface.is_made())
            .count();
        let counts = &mut self.counts.shapes;
        counts.surfaces_live_most = counts.surfaces_live_most.max(live.try_into().unwrap_or(0));
        self.hold_bytes(bytes);
        let gone = self.sweep_shapes(block);
        self.note_shape_bound();
        (found, (gone, None))
    }
}

/// What [`SurfaceCaches::add_drawn_planes`] lets go of: images, and a surface's memory and
/// textures not taken.
#[cfg(windows)]
type DrawnGone<P> = (
    Vec<<P as Platform>::Texture>,
    Option<(Arc<dyn ImportedSurfaceMemory>, DrawnPlanes)>,
);

#[cfg(windows)]
impl<P: Platform> Caches<P> {
    /// The plane shape `key` of the block at `block`'s cache, where `key` asks its surface's
    /// (module doc): the spare made ahead for its ask, else the surface made over two new
    /// Direct3D 11 textures with the lock let go of, its planes added as the cache's plane shapes
    /// ([`SurfaceCaches::add_drawn_planes`]), and a reference to `key`'s and its image; the next
    /// spare's make handed over. `None` where the cache takes no surface for `key`, or no
    /// textures or surface were made (counted), for the caller to make the shape's image of its
    /// own. The planes take their fill at a use, never a clear, so `_platform` goes unused.
    pub(super) fn plane_shape(
        &self,
        _platform: &P,
        device: &P::Device,
        block: u64,
        key: &ShapeKey,
    ) -> Option<Held<P::Texture>> {
        let draw = self.drawn()?;
        let ask = self.state.lock().drawn_ask(block, key)?;
        let spares = self.drawn_spares.get();
        let spare_ask = (ask.clone(), device.recoveries());
        let spare = spares.and_then(|spares| spares.take(&spare_ask));
        let Some((made, drawn)) = spare.or_else(|| ask.make_for::<P>(device, &draw, block)) else {
            self.state.lock().counts.shapes.surfaces_refused += 1;
            return None;
        };
        if let Some(spares) = spares {
            let (device, draw, next) = (device.clone_ref(), Arc::clone(&draw), ask.clone());
            spares.want(spare_ask, move || next.make::<P>(&device, &draw).ok());
        }
        let (found, gone) = (self.state.lock()).add_drawn_planes(block, key, &ask, made, drawn);
        if let Err(refusal) = refusals::refusing(move || drop(gone)) {
            log::warn!(
                target: "reims_vgpu::graphics", kind:% = refusal;
                "the IOSurface block at {block:#x}'s video planes or shapes past their bound were \
                 let go of, and a release was refused: {refusal}"
            );
        }
        found
    }

    /// The planes of the block at `block`'s surface over Direct3D 11 textures (module doc), as
    /// the targets of a decode that shows them whole, where its cache has one: the command
    /// buffer being walked lands its picture there ([`owe_video`]). `None` where it has none.
    pub(crate) fn drawn_targets(&self, block: u64) -> Option<[DrawnTarget<P>; 2]> {
        let mut state = self.state.lock();
        state.by_block.get_mut(&block)?.surface.drawn_targets()
    }

    /// Makes `in_flight`, the guest's decodes in flight on the device's video at its last decode,
    /// the spare surfaces an ask keeps (module doc), where the caches make spares.
    pub(crate) fn saw_decodes_in_flight(&self, in_flight: u64) {
        if let Some(spares) = self.drawn_spares.get() {
            spares.set_depth(in_flight);
        }
    }
}

#[cfg(all(test, unix))]
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
        assert_eq!(
            (keys[0].width, keys[0].height, keys[0].texel_bytes),
            (1920, 1080, 1)
        );
        assert_eq!(
            (keys[1].width, keys[1].height, keys[1].texel_bytes),
            (960, 540, 2)
        );
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

    #[test]
    fn an_exportable_surface_takes_any_pitch_and_a_dma_buf_256() {
        let planes = [0, 1].map(|n| SurfacePlaneDesc {
            pixel_format: [10, 30][n],
            width: 1920 >> n,
            height: 1080 >> n,
            usage: PLANE_USAGE,
            rows_below: 0,
        });
        let keys = plane_keys(&nv12(), 1).expect("NV12's planes");
        for (handle, pitch_multiple) in [(SurfaceHandle::OpaqueFd, 1), (SurfaceHandle::DmaBuf, 256)]
        {
            let ask = SurfaceAsk {
                handle,
                planes,
                keys,
            };
            let desc = ask.desc();
            assert_eq!(desc.handle, handle);
            assert_eq!(
                (desc.pitch_multiple, desc.start_multiple),
                (pitch_multiple, 256),
                "{handle:?}"
            );
        }
    }
}

/// The rt platform's surfaces over the Direct3D 11 textures that reims-vgpu-video's plane draws
/// make, on the Windows host.
#[cfg(all(test, windows))]
mod rt_windows_tests {
    use reims_vgpu_vulkan::RtPlatform;
    use reims_vgpu_vulkan::core::platform::resource::Texture as _;

    use super::*;

    /// A video cache's planes over the plane draws' textures, of 8-bit and of 16-bit samples, are
    /// a surface the rt platform makes: two planes of the cache's plane formats, the luma's size
    /// and the chroma's half of it (the module doc).
    #[test]
    #[ignore = "needs Direct3D 11 on the rt device's GPU: run on the Windows host"]
    fn the_rt_platform_makes_a_surface_over_the_plane_draws_textures() {
        let platform = Arc::new(RtPlatform::new());
        crate::entry::run(|_| {
            let device = platform.system_default_device().expect("an rt device");
            let draw = reims_vgpu_video::vaapi::PlaneDraw::new(device.host_gpu().luid)
                .expect("the plane draws on the device's adapter");
            let draw = Arc::new(draw);
            let sizes: [(u64, u64); 2] = [(64, 48), (32, 24)];
            for ((texel, views), (_, elements)) in PLANE_FORMATS.iter().zip(&DRAWN_FORMATS) {
                let drawn = DrawnPlanes::new(&draw, (64, 48), *texel == 2).expect("the textures");
                let targets = drawn.targets();
                let handles = [targets.luma_handle(), targets.chroma_handle()];
                let planes: [D3d11PlaneDesc<'_>; 2] = std::array::from_fn(|n| D3d11PlaneDesc {
                    handle: handles[n],
                    image_format: elements[n],
                    pixel_format: views[n],
                    width: sizes[n].0,
                    height: sizes[n].1,
                    usage: PLANE_USAGE,
                });
                let made = (device.import_d3d11_surface(ON, &planes))
                    .expect("no raise")
                    .expect("a surface over the textures");
                assert_eq!(made.planes.len(), 2, "{texel}-byte samples");
                for (n, plane) in made.planes.iter().enumerate() {
                    let made_as = (plane.width(), plane.height(), plane.pixel_format());
                    let asked = (sizes[n].0, sizes[n].1, views[n]);
                    assert_eq!(made_as, asked, "plane {n} of {texel}-byte samples");
                }
            }
        });
    }
}

/// On Windows, a video cache's surface over the plane draws' Direct3D 11 textures in the caches'
/// rig: the plane shapes' fills a decode takes as met (the module doc).
#[cfg(test)]
#[cfg(windows)]
mod drawn_tests {
    use reims_vgpu_graphics::ops::{RGFX_SURFACE_USE_WRITES, RgfxOpSurfaceUse};
    use reims_vgpu_video::vaapi::PlaneDraw;
    use reims_vgpu_vulkan::core::platform::command::{CommandBuffer as _, CommandQueue as _};
    use reims_vgpu_vulkan::core::platform::values::TextureDesc;
    use reims_vgpu_vulkan::core::testkit::FakePlatform;

    use super::super::super::surface_pages::tests::{BLOCK, Rig};
    use super::super::Reach;
    use super::super::tests::cached;
    use super::*;

    /// The memory of the rig's surface, whose writes the tests make none of.
    struct Unwritten;

    impl ImportedSurfaceMemory for Unwritten {
        fn external_writes_done(&self) -> Result<(), String> {
            Ok(())
        }
    }

    /// The planes of the rig's 64 by 48 NV12 surface in its pages: the luma's rows, then the
    /// chroma's.
    const PLANES: [PagesPlane; 2] = [
        PagesPlane {
            offset: 0,
            bytes_per_row: 64,
            row_bytes: 64,
            rows: 48,
        },
        PagesPlane {
            offset: 64 * 48,
            bytes_per_row: 64,
            row_bytes: 64,
            rows: 24,
        },
    ];

    /// The rig's cache made for the host's video, its surface's planes over the Direct3D 11
    /// textures `draw` makes, its plane shapes owing their fills from the buffer, as
    /// [`SurfaceCaches::add_drawn_planes`] leaves a new surface's: the textures, and the CPU
    /// generation of the first fill the cache owes.
    fn drawn_cache(r: &Rig, draw: &Arc<PlaneDraw>) -> (Arc<DrawnPlanes>, u64) {
        let drawn = Arc::new(DrawnPlanes::new(draw, (64, 48), false).expect("the textures"));
        let images: [<FakePlatform as Platform>::Texture; 2] = std::array::from_fn(|n| {
            let desc = r.p.texture_descriptor(&TextureDesc {
                pixel_format: PLANE_FORMATS[0].1[n],
                width: 64 >> n,
                height: 48 >> n,
                usage: PLANE_USAGE,
                ..TextureDesc::default()
            });
            (r.device.new_texture(ON, &desc).unwrap()).expect("a plane's image")
        });
        let key = |plane, offset, width, height, texel_bytes| ShapeKey {
            plane,
            offset,
            bytes_per_row: 64,
            width,
            height,
            texel_bytes,
            storage: false,
        };
        let keys = [key(0, 0, 64, 48, 1), key(1, 64 * 48, 32, 24, 2)];
        let mut state = r.dec.surface_caches.lock();
        let cache = state.by_block.get_mut(&BLOCK).expect("the rig's cache");
        let owed = cache.first_fill.expect("the cache owes its first fill");
        cache.surface = CacheSurface::new(Some(VideoMemory::Drawn), Some(&PLANES));
        cache.surface.asked = true;
        for (key, image) in keys.into_iter().zip(&images) {
            drop(cache.shapes.insert_plane(key, image.clone_ref(), true, 0));
        }
        cache.surface.made = Some(DrawnSurface {
            planes: images,
            memory: Arc::new(Unwritten),
            drawn: Arc::clone(&drawn),
            life: MemoryLife::new(),
        });
        (drawn, owed)
    }

    /// The use of a decode that writes the rig's block at CPU generation `cpu_gen`.
    fn decodes(cpu_gen: u64) -> [RgfxOpSurfaceUse; 1] {
        [RgfxOpSurfaceUse {
            block_offset: BLOCK,
            cpu_gen,
            flags: RGFX_SURFACE_USE_WRITES,
            reserved_14: 0,
        }]
    }

    /// The reach of a command buffer of a decode alone that writes the rig's surface's planes
    /// whole.
    fn decoding() -> Reach {
        Reach {
            context: None,
            work: false,
            video: true,
            decoded: vec![(BLOCK, PLANES)],
            named: vec![(BLOCK, false)],
        }
    }

    /// The fills of a buffer or an image from a buffer that `r`'s platform was asked for since
    /// `mark`.
    fn fills_since(r: &Rig, mark: u64) -> usize {
        (r.p.call_log().since(mark).iter())
            .filter(|c| c.sel.starts_with("copyFromBuffer:"))
            .count()
    }

    /// A decode that writes the planes whole, into a new cache whose surface's planes lie over
    /// Direct3D 11 textures, takes the plane shapes' fills as met with the cache's first fill:
    /// its command buffer fills neither and waits for no fill, and the textures owe zeros
    /// instead, for the picture's draw to write. Nothing is filled at its finish either, its
    /// picture answered
    #[test]
    #[ignore = "needs Direct3D 11 on the default adapter: run on the Windows host"]
    fn a_decode_into_a_new_drawn_cache_fills_no_plane_shape_and_waits_for_none() {
        let r = cached();
        let draw = Arc::new(PlaneDraw::new(None).expect("the plane draws"));
        let (drawn, owed) = drawn_cache(&r, &draw);
        let queue = r.device.new_command_queue().unwrap();
        let cb = queue.command_buffer(ON).unwrap().unwrap();
        let caches = &r.dec.surface_caches;
        let mark = r.p.call_log().mark();
        let (waits, uses) = (caches.before(&r.device, &cb, &decodes(owed), &decoding())).unwrap();
        assert!(!waits, "no wait on the GPU");
        assert!(
            caches.lock().last_fill(BLOCK).is_none(),
            "no fill to wait for"
        );
        let counts = r.dec.caches().counts();
        assert_eq!(
            (counts.fills, counts.made_fills, counts.decode_firsts),
            (0, 0, 1)
        );
        let shapes = counts.shapes;
        assert_eq!(
            (shapes.fills, shapes.fills_made, shapes.drawn_fills_taken),
            (0, 0, 2)
        );
        assert!(drawn.owes_zeros(), "the textures owe zeros");
        drop(caches.drawn_targets(BLOCK).expect("the decode's targets"));
        caches.decoded(Some(&r.device), BLOCK, true).unwrap();
        (caches.finish(&r.device, 7, None, &cb, uses.unwrap(), || cb.commit(ON))).unwrap();
        assert_eq!(fills_since(&r, mark), 0, "nothing filled");
        assert!(drawn.owes_zeros(), "the picture's draw writes them");
    }

    /// A decode whose textures show the planes of a cache whose surface lies over Direct3D 11
    /// textures whole is given the surface's planes, which its picture is drawn into, and makes
    /// no memory of the cache's buffer, a deferred one whose memory is not made, as a Windows
    /// video cache's is (the module doc of `surface_cache`, "Deferred memory"). One whose
    /// textures do not show them whole lands its picture in the buffer, and makes it, once
    #[test]
    #[ignore = "needs Direct3D 11 on the default adapter: run on the Windows host"]
    fn a_decode_drawn_into_a_caches_surface_makes_no_memory_of_its_buffer() {
        use super::super::tests::{surface_texture_as, with_facts};
        use reims_vgpu_graphics::wire::RgfxTextureFromIOSurface;
        use reims_vgpu_vulkan::core::platform::resource::Buffer as _;
        use reims_vgpu_vulkan::core::platform::types::PixelFormatInfo;
        use reims_vgpu_vulkan::core::testkit::Step;
        use reims_vgpu_vulkan::core::testkit::impls::resource::MAKE_MEMORY;
        use reims_vgpu_vulkan::core::testkit::objects::{CompleteOn, Model};
        use zerocopy::{FromBytes as _, IntoBytes as _};
        let draw = Arc::new(PlaneDraw::new(None).expect("the plane draws"));
        for whole in [true, false] {
            let mut r = cached();
            let page = r.page;
            assert_eq!(r.buffer(10, BLOCK, 2 * page), Ok(()));
            // Textures 21 and 22 over buffer 10, each over its plane of [`PLANES`], as a decode's
            // luma (`R8Unorm`) and chroma (`RG8Unorm`) textures.
            for (obj_id, plane) in [(21, 0u32), (22, 1)] {
                let laid = PLANES[plane as usize];
                let texel = u64::from(plane + 1);
                let info = PixelFormatInfo {
                    block_bytes: texel,
                    block_width: 1,
                    block_height: 1,
                    block_depth: 1,
                    ..PixelFormatInfo::default()
                };
                r.p.script().clear();
                (r.p.script()).always("MTLPixelFormatGetInfoForDevice()", Step::value(info));
                let size = ((laid.row_bytes / texel) as u32, laid.rows as u32);
                let bytes = surface_texture_as(obj_id, 10, [10, 30][plane as usize], size, (1, 0));
                let mut m = RgfxTextureFromIOSurface::read_from_bytes(&bytes).unwrap();
                m.offset = laid.offset;
                m.bytes_per_row = laid.bytes_per_row;
                m.plane = plane;
                let mut message = m.as_bytes().to_vec();
                if whole {
                    let chroma = (PLANES[1].offset, PLANES[1].bytes_per_row);
                    message = with_facts(message, (64, 48), chroma);
                }
                assert_eq!(r.dec.decode(&r.device, &message).unwrap(), Ok(()));
            }
            drop(drawn_cache(&r, &draw));
            r.p.world().complete_all(CompleteOn::Caller);
            if let Some(buffer) = r.dec.caches().buffer(BLOCK)
                && let Model::Buffer(m) = &mut buffer.state().model
            {
                m.deferred = Some(false);
            }
            let mark = r.p.call_log().mark();
            let (luma, chroma) = r.dec.video_planes(0, 21, 22, true).unwrap();
            assert_eq!(
                (luma.drawn.is_some(), chroma.drawn.is_some()),
                (whole, whole),
                "whole {whole}: the surface's planes"
            );
            drop((luma, chroma));
            let calls = r.p.call_log().since(mark);
            let made = calls.iter().filter(|c| c.sel == MAKE_MEMORY).count();
            assert_eq!(
                made,
                usize::from(!whole),
                "whole {whole}: the buffer's memory"
            );
            let memory = r
                .dec
                .caches()
                .buffer(BLOCK)
                .is_some_and(|b| b.memory_made());
            assert_eq!(memory, !whole, "whole {whole}: made");
        }
    }

    /// Where such a decode answers nothing at all, its command buffer's finish clears the
    /// textures to the zeros they owe; where it answers no picture, the cache's first fill is
    /// made, which fills the plane shapes from the buffer: either way the textures owe nothing
    /// after the finish
    #[test]
    #[ignore = "needs Direct3D 11 on the default adapter: run on the Windows host"]
    fn a_drawn_cache_no_picture_lands_in_owes_nothing_after_the_finish() {
        let draw = Arc::new(PlaneDraw::new(None).expect("the plane draws"));
        for answer in [None, Some(false)] {
            let r = cached();
            let (drawn, owed) = drawn_cache(&r, &draw);
            let queue = r.device.new_command_queue().unwrap();
            let cb = queue.command_buffer(ON).unwrap().unwrap();
            let caches = &r.dec.surface_caches;
            let (_, uses) = (caches.before(&r.device, &cb, &decodes(owed), &decoding())).unwrap();
            assert!(drawn.owes_zeros(), "{answer:?}: owed at the begin");
            drop(caches.drawn_targets(BLOCK).expect("the decode's targets"));
            if let Some(picture) = answer {
                caches.decoded(Some(&r.device), BLOCK, picture).unwrap();
            }
            (caches.finish(&r.device, 7, None, &cb, uses.unwrap(), || cb.commit(ON))).unwrap();
            assert!(!drawn.owes_zeros(), "{answer:?}: owed after the finish");
            let filled = r.dec.caches().counts().shapes.fills;
            let expected = if answer.is_some() { 2 } else { 0 };
            assert_eq!(filled, expected, "{answer:?}: the plane shapes filled");
        }
    }
}
