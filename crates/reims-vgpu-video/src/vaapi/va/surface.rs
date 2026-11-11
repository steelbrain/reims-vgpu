//! Decode surfaces: a pool of surfaces of one format and size, lent out
//! one at a time, and their readback into caller-provided planes. 8-bit
//! 4:2:0 is NV12, and 10-bit 4:2:0 is P010.
//!
//! A frame's wait for its decode, and the staged copy back's run after it,
//! end at one deadline (`super::wait`). Past it the pool is stalled: that
//! frame fails, each later frame of the pool, and each surface asked of it,
//! fails at once, and the pool, with its surfaces, its staged copy back's
//! memory and the decoder's context once the decoder lets it go
//! ([`end_context`]), is kept until the GPU is done with every one of its
//! surfaces.

use std::sync::{Arc, OnceLock};
use std::time::Duration;

use crate::core::runtime::MonoTime;
use crate::core::sync::{Mutex, MutexGuard};

use super::caps::{Config, Context, Entrypoint, Profile};
use super::display::{Display, Error, check};
use super::export::{self, SurfaceExport};
use super::ffi;
use super::image::{self, Image};
use super::readback::{self, Mappings, ReadbackTimes, ReadbackWay, Staging, copy_rect, copy_row};
use super::target::{Target, TargetFormat, TargetShape};
use super::vpp_ffi as vpp;
use super::wait::{self, Unfinished, Work};

/// The most surfaces a pool makes. A DPB holds 16 frames at most (A.3.1),
/// so the rest are the current picture and frames the caller holds; past
/// this many, a caller that never drops its frames gets an error rather
/// than exhausting video memory.
const MAX_SURFACES: usize = 48;

/// What a pool's surfaces hold: 4:2:0 in two planes, luma then
/// interleaved Cb/Cr.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum SurfaceFormat {
    /// 8-bit samples (`VA_FOURCC_NV12`).
    Nv12,
    /// 10-bit samples in the top bits of little-endian 16-bit words
    /// (`VA_FOURCC_P010`).
    P010,
}

impl SurfaceFormat {
    /// The render-target format a config and its surfaces are made with.
    pub(crate) fn rt_format(self) -> core::ffi::c_uint {
        match self {
            Self::Nv12 => ffi::VA_RT_FORMAT_YUV420,
            Self::P010 => ffi::VA_RT_FORMAT_YUV420_10,
        }
    }

    pub(crate) fn fourcc(self) -> u32 {
        match self {
            Self::Nv12 => ffi::VA_FOURCC_NV12,
            Self::P010 => ffi::VA_FOURCC_P010,
        }
    }

    /// Bits a pixel takes, over both planes.
    pub(crate) fn bits_per_pixel(self) -> u32 {
        match self {
            Self::Nv12 => 12,
            Self::P010 => 24,
        }
    }

    /// The bytes one sample takes.
    pub(crate) fn bytes_per_sample(self) -> usize {
        match self {
            Self::Nv12 => 1,
            Self::P010 => 2,
        }
    }

    /// The FOURCC's name.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Nv12 => "NV12",
            Self::P010 => "P010",
        }
    }
}

/// A pool of surfaces of one format and coded size.
pub(crate) struct SurfacePool {
    inner: Arc<PoolInner>,
}

struct PoolInner {
    display: Display,
    format: SurfaceFormat,
    width: u32,
    height: u32,
    state: Mutex<PoolState>,
    readback: Mutex<Readback>,
    times: Mutex<ReadbackTimes>,
    /// The image [`Surface::copy_to`] copies through, made on first use:
    /// `None` until then, `Some(None)` where the driver makes none.
    copy_image: Mutex<Option<Option<Image>>>,
    /// The video-processing context [`Surface::copy_to`] copies a target through, made on first
    /// use: `None` until then, `Some(None)` where the driver makes none.
    processing: Mutex<Option<Option<Context>>>,
    /// The work the GPU did not finish by its deadline, once it has not
    /// (module docs).
    stalled: OnceLock<String>,
    /// Each surface's export as dma-bufs, made at its first frame asked
    /// so ([`Surface::export`]) and kept while the pool lives.
    exports: Mutex<Vec<(ffi::VASurfaceID, Arc<SurfaceExport>)>>,
}

struct PoolState {
    /// Every surface the pool made, lent or not.
    all: Vec<ffi::VASurfaceID>,
    free: Vec<ffi::VASurfaceID>,
}

/// How frames are read back (`super::readback`): staged through host
/// memory, through an image of the pool's format and size that
/// `vaGetImage` fills, or by `vaDeriveImage` where the driver cannot; each
/// made on first use.
enum Readback {
    Unknown,
    Staged(Staging),
    GetImage(Image),
    Derive,
    Mapped(Mappings),
}

impl Drop for PoolInner {
    fn drop(&mut self) {
        // The processing context goes before the surfaces it wrote into.
        drop(self.processing.get_mut().take());
        let mut all = std::mem::take(&mut self.state.get_mut().all);
        if !all.is_empty() {
            let count = i32::try_from(all.len()).unwrap_or(0);
            // SAFETY: every surface was made by vaCreateSurfaces on this
            // display, and none is lent, since each lent Surface holds the
            // pool; destroyed once, here, behind every call on them.
            self.display.release(move |raw| unsafe {
                ffi::vaDestroySurfaces(raw, all.as_mut_ptr(), count);
            });
        }
    }
}

impl SurfacePool {
    /// Makes `count` surfaces of `format`, `width` by `height`.
    pub(crate) fn new(
        display: &Display,
        format: SurfaceFormat,
        width: u32,
        height: u32,
        count: usize,
    ) -> Result<Self, Error> {
        let inner = Arc::new(PoolInner {
            display: display.clone_ref(),
            format,
            width,
            height,
            state: Mutex::new(PoolState {
                all: Vec::new(),
                free: Vec::new(),
            }),
            readback: Mutex::new(Readback::Unknown),
            times: Mutex::new(ReadbackTimes::default()),
            copy_image: Mutex::new(None),
            processing: Mutex::new(None),
            stalled: OnceLock::new(),
            exports: Mutex::new(Vec::new()),
        });
        let made = create_surfaces(display, format, width, height, count)?;
        {
            let mut state = inner.state.lock();
            state.all.extend(&made);
            state.free.extend(made.iter().rev());
        }
        Ok(Self { inner })
    }

    /// Every surface made so far, for a context's render targets.
    pub(crate) fn ids(&self) -> Vec<ffi::VASurfaceID> {
        self.inner.state.lock().all.clone()
    }

    /// A free surface, making one more when none is free; none from a
    /// stalled pool.
    pub(crate) fn get(&self) -> Result<Surface, Error> {
        self.inner.stalled()?;
        let mut state = self.inner.state.lock();
        let id = match state.free.pop() {
            Some(id) => id,
            None => {
                if state.all.len() >= MAX_SURFACES {
                    return Err(Error::OutOfSurfaces);
                }
                let inner = &self.inner;
                let made =
                    create_surfaces(&inner.display, inner.format, inner.width, inner.height, 1)?;
                state.all.extend(&made);
                made[0]
            }
        };
        Ok(Surface {
            id,
            pool: self.inner.clone(),
            target: None,
        })
    }

    /// `target` lent as a surface of the pool, a decode's picture: `None` where it is not of the
    /// pool's display, format and size, or the pool is stalled.
    pub(crate) fn lend_target(&self, target: Target) -> Option<Surface> {
        let inner = &self.inner;
        let shape = target.shape();
        let fits = target.is_on(&inner.display)
            && shape.format == TargetFormat::of(inner.format)
            && shape.coded == (inner.width, inner.height)
            && inner.stalled().is_ok();
        fits.then(|| Surface {
            id: target.id(),
            pool: Arc::clone(inner),
            target: Some(target),
        })
    }

    /// The format and coded size of its surfaces, which a decode's target is asked for.
    pub(crate) fn shape(&self) -> TargetShape {
        TargetShape {
            format: TargetFormat::of(self.inner.format),
            coded: (self.inner.width, self.inner.height),
        }
    }
}

/// Destroys `context`, which the surfaces of `pools` are decoded on, as its
/// decoder lets it go: now, or where one of the pools is stalled, once the
/// GPU is done with each of their surfaces (module docs).
pub(crate) fn end_context<'a>(
    context: Option<Context>,
    pools: impl IntoIterator<Item = &'a SurfacePool>,
) {
    let Some(context) = context else {
        return;
    };
    let pools: Vec<Arc<PoolInner>> = pools.into_iter().map(|p| Arc::clone(&p.inner)).collect();
    let Some(stalled) = pools.iter().find(|pool| pool.stalled.get().is_some()) else {
        return drop(context);
    };
    let display = stalled.display.clone_ref();
    let kept = Kept {
        _context: Some(context),
        pools,
    };
    wait::keep_work(&display, "a stalled decoder's context", Box::new(kept));
}

/// What makes a surface the probe's way, for tests: one surface of a format and size.
#[cfg(test)]
pub(crate) type MakeSurface =
    Box<dyn FnMut(&Display, SurfaceFormat, u32, u32) -> Result<ffi::VASurfaceID, Error>>;

#[cfg(test)]
thread_local! {
    /// What makes each surface a pool makes on this thread, for tests: where set, a surface over
    /// memory of the test's own (`super::target_probe`), in place of one the driver allocates.
    pub(crate) static MAKE_SURFACE: std::cell::RefCell<Option<MakeSurface>> =
        const { std::cell::RefCell::new(None) };
}

fn create_surfaces(
    display: &Display,
    surface_format: SurfaceFormat,
    width: u32,
    height: u32,
    count: usize,
) -> Result<Vec<ffi::VASurfaceID>, Error> {
    #[cfg(test)]
    {
        let made = MAKE_SURFACE.with(|make| {
            (make.borrow_mut().as_mut()).map(|make| {
                (0..count)
                    .map(|_| make(display, surface_format, width, height))
                    .collect::<Result<Vec<_>, Error>>()
            })
        });
        if let Some(made) = made {
            return made;
        }
    }
    let mut ids = vec![ffi::VA_INVALID_SURFACE; count];
    let mut format = ffi::VASurfaceAttrib::zeroed();
    format.type_ = ffi::VASurfaceAttribPixelFormat;
    format.flags = ffi::VA_SURFACE_ATTRIB_SETTABLE;
    format.value.type_ = ffi::VAGenericValueTypeInteger;
    format.value.value.i = i32::from_ne_bytes(surface_format.fourcc().to_ne_bytes());
    let count = u32::try_from(count).unwrap_or(0);
    check(
        "vaCreateSurfaces",
        // SAFETY: `ids` holds `count` entries, and the one attribute is
        // valid.
        display.call(|raw| unsafe {
            ffi::vaCreateSurfaces(
                raw,
                surface_format.rt_format(),
                width,
                height,
                ids.as_mut_ptr(),
                count,
                &mut format,
                1,
            )
        })?,
    )?;
    Ok(ids)
}

/// A surface on loan from its pool, back in the pool when dropped; or a target over the caller's
/// memory a decode was handed ([`SurfacePool::lend_target`]), which it holds, as the pool's
/// surfaces are held, in the pool's name.
pub(crate) struct Surface {
    id: ffi::VASurfaceID,
    pool: Arc<PoolInner>,
    /// The target the surface is, where it is one.
    target: Option<Target>,
}

impl Drop for Surface {
    fn drop(&mut self) {
        match self.target.take() {
            None => self.pool.state.lock().free.push(self.id),
            // A target of a stalled pool is kept, with its memory, until the GPU is done with it.
            Some(target) if self.pool.stalled.get().is_some() => wait::keep(
                &self.pool.display,
                "a stalled decoder's target",
                vec![Work::Surface(self.id)],
                target,
            ),
            Some(_) => {}
        }
    }
}

/// A rectangle of a surface, in luma samples.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Rect {
    pub(crate) x: u32,
    pub(crate) y: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
}

/// A surface read back where the CPU reaches it ([`Surface::read_back`]),
/// kept there until dropped: the pool's way back, locked so that no other
/// frame's read back reuses its staging surface, image or mapping, and the
/// bytes it left the frame in.
pub(crate) struct SurfaceReadBack<'a> {
    surface: &'a Surface,
    _readback: MutexGuard<'a, Readback>,
    read: Read,
    wait: Duration,
    transfer: Duration,
}

/// Where a read back left a frame: `len` bytes from `bytes`, its planes
/// at their offsets and pitches, read no further than `bounds`, each row
/// with `copy`; and what undoes the read back once it is copied.
struct Read {
    bytes: *const u8,
    len: usize,
    planes: [(usize, usize); 2],
    bounds: (u32, u32),
    copy: fn(&mut [u8], &[u8]),
    release: Release,
}

/// What a read back leaves to undo as it drops.
enum Release {
    Nothing,
    /// The pool's image, its buffer mapped.
    Unmap(ffi::VABufferID),
    /// A derived image, its buffer mapped, to unmap; it is destroyed as
    /// it drops.
    Derived(Image),
    /// A read of a decode surface's own dma-buf, to end.
    EndRead(std::os::fd::RawFd),
}

/// Why a read back goes another way, from now on, or fails.
enum Fallback {
    /// The staged copy back failed: images from now on.
    Staged(Error),
    /// The driver reads no surface into an image: derived images.
    Derive,
    /// The staged copy back's run did not finish by its deadline: what, in
    /// words. The pool is stalled.
    Stalled(String),
}

impl SurfaceReadBack<'_> {
    /// Copies `rect` of the surface into planes of its format: `y` with
    /// `y_stride` bytes a row, and the interleaved chroma `uv` with
    /// `uv_stride`. The rectangle's position must be even. An odd width or
    /// height takes the chroma sample its last column or row shares, as
    /// CoreVideo's planes and ffmpeg's frames have it: `uv` holds
    /// `(width + 1) / 2` sample pairs a row, `(height + 1) / 2` rows. The
    /// CPU's copy alone.
    pub(crate) fn copy_planes_to(
        &self,
        rect: Rect,
        y: &mut [u8],
        y_stride: usize,
        uv: &mut [u8],
        uv_stride: usize,
    ) -> Result<(), Error> {
        let pool = &self.surface.pool;
        let [(luma_bytes, luma_rows), (chroma_bytes, chroma_rows)] =
            readback::plane_extents(pool.format, rect);
        // Whether a plane of `len` bytes holds `rows` rows of `bytes`,
        // `stride` apart.
        let holds = |len: usize, stride: usize, rows: usize, bytes: usize| {
            stride >= bytes
                && stride
                    .checked_mul(rows.saturating_sub(1))
                    .and_then(|start| start.checked_add(bytes))
                    .is_some_and(|end| end <= len)
        };
        if !holds(y.len(), y_stride, luma_rows, luma_bytes) {
            return Err(Error::PlaneTooSmall("luma"));
        }
        if !holds(uv.len(), uv_stride, chroma_rows, chroma_bytes) {
            return Err(Error::PlaneTooSmall("chroma"));
        }
        let read = &self.read;
        let within = |at: u32, length: u32, bound: u32| {
            at.checked_add(length).is_some_and(|end| end <= bound)
        };
        if !within(rect.x, rect.width, read.bounds.0) || !within(rect.y, rect.height, read.bounds.1)
        {
            return Err(Error::Unsupported("a crop larger than its surface".into()));
        }
        let started = pool.display.runtime().now();
        // SAFETY: the read back's `len` bytes, readable until it drops:
        // the staging surface's memory, held by the pool's way back, which
        // this keeps locked, or a mapping undone only as it drops.
        let bytes = unsafe { core::slice::from_raw_parts(read.bytes, read.len) };
        let destinations = [(y, y_stride), (uv, uv_stride)];
        copy_rect(
            bytes,
            read.planes,
            pool.format,
            rect,
            destinations,
            read.copy,
        )?;
        pool.times.lock().add(
            self.wait,
            self.transfer,
            pool.display.runtime().elapsed_since(started),
        );
        Ok(())
    }
}

impl Drop for SurfaceReadBack<'_> {
    fn drop(&mut self) {
        let display = &self.surface.pool.display;
        match &self.read.release {
            Release::Nothing => {}
            // Mapped by the read back, unmapped once.
            Release::Unmap(buffer) => image::unmap_buffer(display, *buffer),
            // Mapped by the read back, unmapped once, before the image is
            // destroyed as the field drops after this.
            Release::Derived(image) => image::unmap_buffer(display, image.raw().buf),
            Release::EndRead(fd) => readback::end_read(*fd),
        }
    }
}

impl Surface {
    pub(crate) fn id(&self) -> ffi::VASurfaceID {
        self.id
    }

    /// The target over the caller's memory the surface is, where it is one.
    pub(crate) fn target(&self) -> Option<&Target> {
        self.target.as_ref()
    }

    /// What the surface holds.
    pub(crate) fn format(&self) -> SurfaceFormat {
        self.pool.format
    }

    /// Its width and height: its pool's, the decoded (coded) picture's.
    pub(crate) fn size(&self) -> (u32, u32) {
        (self.pool.width, self.pool.height)
    }

    /// The display it is on.
    pub(crate) fn display(&self) -> &Display {
        &self.pool.display
    }

    /// The surface as dma-bufs, for another device to read its planes (`super::export`):
    /// exported at its first ask and kept with its pool; none from a stalled pool.
    pub(crate) fn export(&self) -> Result<Arc<SurfaceExport>, Error> {
        self.pool.stalled()?;
        let mut exports = self.pool.exports.lock();
        if let Some((_, kept)) = exports.iter().find(|(id, _)| *id == self.id) {
            return Ok(Arc::clone(kept));
        }
        let made = Arc::new(export::export(self.display(), self.id, self.size())?);
        exports.push((self.id, Arc::clone(&made)));
        Ok(made)
    }

    /// Waits for the surface's decode to finish, until the deadline of the
    /// frame in hand (module docs).
    pub(crate) fn sync(&self) -> Result<(), Error> {
        self.sync_by(wait::current(self.display()))
    }

    /// Waits for the surface's decode to finish, until `deadline`: past it
    /// the pool is stalled, and a stalled pool's surfaces fail at once.
    pub(crate) fn sync_by(&self, deadline: MonoTime) -> Result<(), Error> {
        self.pool.stalled()?;
        let synced = wait::wait_for(
            self.display(),
            Work::Surface(self.id),
            deadline,
            "a decode in time",
        );
        self.stalled_if(synced)
    }

    /// `result`, the surface's pool stalled first where it is
    /// [`Error::Stalled`]: work on the surface, or from it, the GPU did not
    /// finish by its deadline.
    pub(crate) fn stalled_if<T>(&self, result: Result<T, Error>) -> Result<T, Error> {
        if let Err(Error::Stalled(what)) = &result {
            PoolInner::stall(&self.pool, what);
        }
        result
    }

    /// Reads the surface back where the CPU reaches it: waits for its
    /// decode, then has the driver put it there by the pool's way back
    /// (`super::readback`), both until one deadline. Every driver call a
    /// copy needs is made here or as the result drops, so that its copies
    /// run on the CPU alone. The pool's way back stays locked until the
    /// result drops.
    pub(crate) fn read_back(&self) -> Result<SurfaceReadBack<'_>, Error> {
        let deadline = wait::current(self.display());
        let started = self.display().runtime().now();
        self.sync_by(deadline)?;
        let read = self.read_back_synced(self.display().runtime().elapsed_since(started), deadline);
        self.stalled_if(read)
    }

    /// [`Surface::read_back`] of a surface whose decode took `wait`, its
    /// way back's run waited for until `deadline`.
    fn read_back_synced(
        &self,
        wait: Duration,
        deadline: MonoTime,
    ) -> Result<SurfaceReadBack<'_>, Error> {
        let pool = &self.pool;
        let mut readback = pool.readback.lock();
        if matches!(*readback, Readback::Unknown) {
            *readback = pool.first_readback();
        }
        let started = pool.display.runtime().now();
        let read = match &mut *readback {
            Readback::Staged(staging) => match staging.stage(self.id, deadline) {
                Ok(()) => {
                    let (bytes, planes) = staging.bytes();
                    Ok(Read {
                        bytes: bytes.as_ptr(),
                        len: bytes.len(),
                        planes,
                        bounds: (pool.width, pool.height),
                        copy: copy_row,
                        release: Release::Nothing,
                    })
                }
                // The run may still write the staging memory: no other way
                // back reads the frame meanwhile.
                Err(Error::Stalled(what)) => Err(Fallback::Stalled(what)),
                Err(error) => Err(Fallback::Staged(error)),
            },
            Readback::Mapped(mappings) => {
                let mapped = mappings.begin_read(&pool.display, self.id)?;
                Ok(Read {
                    bytes: mapped.bytes.as_ptr(),
                    len: mapped.bytes.len(),
                    planes: mapped.planes,
                    bounds: (pool.width, pool.height),
                    copy: readback::copy_row_streaming,
                    release: Release::EndRead(mapped.fd),
                })
            }
            // The region is the whole surface, which the image matches.
            Readback::GetImage(image) => match image.get_from(self.id, pool.width, pool.height) {
                Ok(()) => {
                    let raw = image.raw();
                    let mapped = map_image(&pool.display, pool.format, raw, false)?;
                    Ok(Read {
                        bytes: mapped.bytes,
                        len: mapped.len,
                        planes: mapped.planes,
                        bounds: (u32::from(raw.width), u32::from(raw.height)),
                        copy: copy_row,
                        release: Release::Unmap(raw.buf),
                    })
                }
                // The GPU did not finish the copy: no other way back reads
                // the frame meanwhile.
                Err(Error::Stalled(what)) => Err(Fallback::Stalled(what)),
                // The driver cannot read this surface into an image: derive
                // from now on, the image destroyed as the way back changes.
                Err(_) => Err(Fallback::Derive),
            },
            Readback::Derive | Readback::Unknown => {
                // The surface is synced; a failed map destroys the image as
                // it drops.
                let image = Image::derive(&pool.display, self.id)?;
                let raw = image.raw();
                let mapped = map_image(&pool.display, pool.format, raw, true)?;
                Ok(Read {
                    bytes: mapped.bytes,
                    len: mapped.len,
                    planes: mapped.planes,
                    bounds: (u32::from(raw.width), u32::from(raw.height)),
                    copy: copy_row,
                    release: Release::Derived(image),
                })
            }
        };
        let read = match read {
            Ok(read) => read,
            Err(Fallback::Staged(error)) => {
                log::warn!(
                    "the staged copy back failed, so this pool's frames come back through \
                     vaGetImage from now on: {error}"
                );
                *readback = pool.image_readback();
                drop(readback);
                return self.read_back_synced(wait, deadline);
            }
            Err(Fallback::Derive) => {
                *readback = Readback::Derive;
                drop(readback);
                return self.read_back_synced(wait, deadline);
            }
            Err(Fallback::Stalled(what)) => {
                drop(readback);
                return self.stalled_if(Err(Error::Stalled(what)));
            }
        };
        Ok(SurfaceReadBack {
            surface: self,
            _readback: readback,
            read,
            wait,
            transfer: pool.display.runtime().elapsed_since(started),
        })
    }

    /// What the pool's frames came back through so far, and how long it took.
    pub(crate) fn readback_times(&self) -> ReadbackTimes {
        *self.pool.times.lock()
    }
}

/// A stalled pool, or its decoder's context once the decoder has let it go
/// and the pools it decodes into, kept until the GPU is done with each of
/// the pools' surfaces (module docs): the context is destroyed first.
struct Kept {
    _context: Option<Context>,
    pools: Vec<Arc<PoolInner>>,
}

impl Unfinished for Kept {
    fn finished(&mut self) -> bool {
        (self.pools.iter()).all(|pool| wait::all_done(&pool.display, &pool.works()))
    }
}

impl PoolInner {
    /// `Err` where the pool is stalled (module docs).
    fn stalled(&self) -> Result<(), Error> {
        match self.stalled.get() {
            Some(what) => Err(Error::Stalled(format!(
                "{what}, of an earlier frame of the pool"
            ))),
            None => Ok(()),
        }
    }

    /// Stalls `pool`, the GPU not having finished `what` by its deadline,
    /// and keeps it until the GPU is done with each of its surfaces; a pool
    /// already stalled is kept already.
    fn stall(pool: &Arc<PoolInner>, what: &str) {
        if pool.stalled.set(what.to_owned()).is_err() {
            return;
        }
        let kept = format!(
            "a stalled decoder's {} {}x{} surfaces",
            pool.format.name(),
            pool.width,
            pool.height
        );
        let work = Kept {
            _context: None,
            pools: vec![Arc::clone(pool)],
        };
        wait::keep_work(&pool.display, kept, Box::new(work));
    }

    /// The work on the pool's surfaces the GPU may still be doing: on each
    /// surface it made, and on its staged copy back's. Asked on the thread
    /// that asks after kept work, which holds no lock of the pool's, so
    /// that a read back in hand, which holds the pool's way back, is waited
    /// for there.
    fn works(&self) -> Vec<Work> {
        let mut works: Vec<Work> = (self.state.lock().all.iter())
            .map(|&id| Work::Surface(id))
            .collect();
        if let Readback::Staged(staging) = &*self.readback.lock() {
            works.push(Work::Surface(staging.surface()));
        }
        works
    }

    /// The way the pool's frames come back, at its first readback: the one
    /// a check asked for (`readback::read_back_by`), or else staged through
    /// host memory, the fastest on the RX 7900 XTX and the Radeon 780M, or
    /// where that cannot be set up an image `vaGetImage` fills.
    fn first_readback(&self) -> Readback {
        match readback::asked() {
            Some(ReadbackWay::Mapped) => Readback::Mapped(Mappings::default()),
            Some(ReadbackWay::Derive) => Readback::Derive,
            Some(ReadbackWay::GetImage) => self.image_readback(),
            // The staged surface's planes are whole rows from its first byte at rows of 256
            // bytes, which a driver that writes rows of whole alignments alone takes too.
            None if let Err(why) = self.display.writes_imports(self.format) => {
                log::info!(
                    "no staged copy back for {} {}x{} frames, so they come back through \
                     vaGetImage: {why}",
                    self.format.name(),
                    self.width,
                    self.height
                );
                self.image_readback()
            }
            Some(ReadbackWay::Staged) | None => {
                match Staging::new(&self.display, self.format, self.width, self.height) {
                    Ok(staging) => Readback::Staged(staging),
                    Err(error) => {
                        log::info!(
                            "no staged copy back for {} {}x{} frames, so they come back \
                             through vaGetImage: {error}",
                            self.format.name(),
                            self.width,
                            self.height
                        );
                        self.image_readback()
                    }
                }
            }
        }
    }

    /// An image of the pool's format and size for `vaGetImage`, or a
    /// derived image where the driver makes none.
    fn image_readback(&self) -> Readback {
        match create_image(&self.display, self.format, self.width, self.height) {
            Ok(image) => Readback::GetImage(image),
            Err(_) => Readback::Derive,
        }
    }
}

impl Surface {
    /// Copies the whole surface into `to`, a surface of the same pool,
    /// through an image of the pool's format: `vaGetImage`, then
    /// `vaPutImage`. `false`, with nothing copied, where the driver makes no
    /// such image.
    ///
    /// A target over the caller's memory is copied by one video-processing run instead, which
    /// reads it on the GPU: the driver's CPU map of an import may fault (`super::target`), and the
    /// caller may have written over the picture since (the copy then holds what the memory holds).
    pub(crate) fn copy_to(&self, to: &Surface) -> Result<bool, Error> {
        if !Arc::ptr_eq(&self.pool, &to.pool) {
            return Err(Error::Unsupported("a copy between pools".into()));
        }
        let copied = match self.target {
            Some(_) => self.copy_by_processing(to),
            None => self.copy_within_pool(to),
        };
        self.stalled_if(copied)
    }

    /// [`Surface::copy_to`] of a target, by a video-processing run on the pool's context for it,
    /// made at its first copy; `false` where the driver makes no such context.
    fn copy_by_processing(&self, to: &Surface) -> Result<bool, Error> {
        self.sync()?;
        let pool = &self.pool;
        let mut processing = pool.processing.lock();
        let context = match &mut *processing {
            Some(context) => context,
            None => {
                let made = Config::new(
                    &pool.display,
                    Profile::NONE,
                    Entrypoint::VIDEO_PROC,
                    &mut [],
                )
                .and_then(|config| {
                    Context::new(&pool.display, config, pool.width, pool.height, &mut [])
                });
                match made {
                    Ok(context) => processing.insert(Some(context)),
                    Err(Error::Stalled(what)) => return Err(Error::Stalled(what)),
                    Err(_) => processing.insert(None),
                }
            }
        };
        let Some(context) = context else {
            return Ok(false);
        };
        let (Ok(width), Ok(height)) = (u16::try_from(pool.width), u16::try_from(pool.height))
        else {
            return Ok(false);
        };
        let whole = vpp::VARectangle {
            x: 0,
            y: 0,
            width,
            height,
        };
        let mut params = vpp::VAProcPipelineParameterBuffer::zeroed();
        params.surface = self.id;
        params.surface_region = &whole;
        params.output_region = &whole;
        // The same standard and range both sides: a plain copy.
        params.surface_color_standard = vpp::VAProcColorStandardBT709;
        params.output_color_standard = vpp::VAProcColorStandardBT709;
        let buffer = super::buffer::Buffer::with_value(
            &pool.display,
            context.id,
            vpp::VAProcPipelineParameterBufferType,
            &params,
        )?;
        super::buffer::render_picture(&pool.display, context.id, to.id, &[&[&buffer]])?;
        drop(processing);
        to.sync()?;
        Ok(true)
    }

    /// [`Surface::copy_to`] of a surface of the same pool.
    fn copy_within_pool(&self, to: &Surface) -> Result<bool, Error> {
        self.sync()?;
        let pool = &self.pool;
        let mut copy_image = pool.copy_image.lock();
        let image = match &mut *copy_image {
            Some(image) => image,
            None => match create_image(&pool.display, pool.format, pool.width, pool.height) {
                Ok(image) => copy_image.insert(Some(image)),
                Err(Error::Stalled(what)) => return Err(Error::Stalled(what)),
                Err(_) => copy_image.insert(None),
            },
        };
        let Some(image) = image else {
            return Ok(false);
        };
        // The region is the whole surface, which the image matches.
        image.get_from(self.id, pool.width, pool.height)?;
        image.put_into(to.id, pool.width, pool.height)?;
        Ok(true)
    }
}

fn create_image(
    display: &Display,
    surface_format: SurfaceFormat,
    width: u32,
    height: u32,
) -> Result<Image, Error> {
    let format = ffi::VAImageFormat {
        fourcc: surface_format.fourcc(),
        bits_per_pixel: surface_format.bits_per_pixel(),
        ..ffi::VAImageFormat::default()
    };
    Image::create(display, format, width, height)
}

/// Maps a filled image of `format`, `derived` or made by `vaCreateImage`
/// (`image::map_buffer`): its bytes, their length and its two planes'
/// offsets and pitches, readable until `vaUnmapBuffer` of its buffer.
fn map_image(
    display: &Display,
    format: SurfaceFormat,
    image: &ffi::VAImage,
    derived: bool,
) -> Result<MappedImage, Error> {
    if image.format.fourcc != format.fourcc() || image.num_planes < 2 {
        return Err(Error::Unsupported(format!(
            "reading back an image of {}",
            super::caps::Fourcc(image.format.fourcc)
        )));
    }
    let data = image::map_buffer(display, image.buf, derived)?;
    // A mapped image buffer is `data_size` readable bytes (va.h, VAImage),
    // until vaUnmapBuffer.
    let planes = [
        (to_usize(image.offsets[0]), to_usize(image.pitches[0])),
        (to_usize(image.offsets[1]), to_usize(image.pitches[1])),
    ];
    Ok(MappedImage {
        bytes: data.cast_const(),
        len: to_usize(image.data_size),
        planes,
    })
}

/// An image [`map_image`] mapped: `len` bytes from `bytes`, and its planes'
/// offsets and pitches.
struct MappedImage {
    bytes: *const u8,
    len: usize,
    planes: [(usize, usize); 2],
}

fn to_usize(value: u32) -> usize {
    usize::try_from(value).unwrap_or(usize::MAX)
}
