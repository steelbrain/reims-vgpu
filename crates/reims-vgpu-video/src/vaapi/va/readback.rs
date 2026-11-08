//! The copy back: how a decoded frame leaves its VA surface for the
//! caller's planes, and how long each part takes.
//!
//! A pool reads its frames back in one of these ways, the first that its
//! driver serves:
//! - **staged**: a video-processing run copies the whole surface on the GPU
//!   into a linear surface over [`HostMemory`] of ours, and the CPU copies
//!   the frame's rectangle out of that cached memory;
//! - **an image**: `vaGetImage` into an image of the pool's format, mapped
//!   and copied out;
//! - **a derived image**: `vaDeriveImage`, mapped and copied out, where the
//!   driver makes no image.
//!
//! A check that compares them on a driver picks one for every pool of the
//! process instead ([`read_back_by`]; the examples take it from
//! `RVGV_VAAPI_READBACK`). `mapped`, which is only for that, maps the decode
//! surface's own dma-buf and reads it with streaming loads.
//!
use std::collections::HashMap;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Duration;

use crate::core::runtime::MonoTime;

use super::caps::{Config, Context, Entrypoint, Profile};
use super::display::{Display, Error, check};
use super::ffi;
use super::host_memory::HostMemory;
use super::surface::{Rect, SurfaceFormat};
use super::vpp_ffi as vpp;
use super::wait::{self, Work};
use super::zero_copy::{PAGE, gcd};

/// What a pool's frames came back through, and the time each part took,
/// summed over the frames.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReadbackTimes {
    /// The frames read back.
    pub frames: u64,
    /// Waiting for their decodes (`super::wait`).
    pub wait: Duration,
    /// The driver's part: `vaGetImage`, or the video-processing run and its
    /// wait, or the map.
    pub transfer: Duration,
    /// The CPU's copy into the caller's planes.
    pub copy: Duration,
}

impl ReadbackTimes {
    pub(crate) fn add(&mut self, wait: Duration, transfer: Duration, copy: Duration) {
        self.frames += 1;
        self.wait += wait;
        self.transfer += transfer;
        self.copy += copy;
    }
}

/// A way frames come back, which a check can have every pool take
/// ([`read_back_by`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReadbackWay {
    /// Staged: a video-processing copy into host memory of ours.
    Staged,
    /// `vaGetImage` into an image of the pool's format.
    GetImage,
    /// `vaDeriveImage`, where the driver makes no image.
    Derive,
    /// The decode surface's own dma-buf, mapped and read with streaming
    /// loads: only to compare.
    Mapped,
}

impl ReadbackWay {
    /// The way `name` names: `staged`, `getimage`, `derive` or `mapped`.
    pub fn from_name(name: &str) -> Option<ReadbackWay> {
        match name {
            "staged" => Some(ReadbackWay::Staged),
            "getimage" => Some(ReadbackWay::GetImage),
            "derive" => Some(ReadbackWay::Derive),
            "mapped" => Some(ReadbackWay::Mapped),
            _ => None,
        }
    }
}

/// The way [`read_back_by`] set, 0 for none, else 1 + its index.
static ASKED: AtomicU8 = AtomicU8::new(0);

/// Has every pool of the process made from now on read its frames back
/// `way`, to compare the ways on a driver; `None` is the pools' own choice,
/// the first their driver serves (module doc), as in the product.
pub fn read_back_by(way: Option<ReadbackWay>) {
    let value = way.map_or(0, |way| 1 + way as u8);
    ASKED.store(value, Ordering::Relaxed);
}

/// The way [`read_back_by`] set, if any.
pub(crate) fn asked() -> Option<ReadbackWay> {
    match ASKED.load(Ordering::Relaxed) {
        1 => Some(ReadbackWay::Staged),
        2 => Some(ReadbackWay::GetImage),
        3 => Some(ReadbackWay::Derive),
        4 => Some(ReadbackWay::Mapped),
        _ => None,
    }
}

/// The bytes a row and the rows of `rect`'s two planes: luma, then the
/// interleaved chroma, whose last pair and row an odd width or height
/// shares with its last luma column or row.
pub(crate) fn plane_extents(format: SurfaceFormat, rect: Rect) -> [(usize, usize); 2] {
    let sample = format.bytes_per_sample();
    let (width, height) = (to_usize(rect.width), to_usize(rect.height));
    [
        (width * sample, height),
        (width.div_ceil(2) * 2 * sample, height.div_ceil(2)),
    ]
}

/// Copies `rect` out of `bytes`, whose planes are at `offsets` with
/// `pitches`, row by row, into the luma and chroma destinations, each with
/// its stride, with `copy` for each row.
pub(crate) fn copy_rect(
    bytes: &[u8],
    planes: [(usize, usize); 2],
    format: SurfaceFormat,
    rect: Rect,
    destinations: [(&mut [u8], usize); 2],
    copy: fn(&mut [u8], &[u8]),
) -> Result<(), Error> {
    let sample = format.bytes_per_sample();
    let short = || Error::Unsupported("an image smaller than its pitches".into());
    let x = to_usize(rect.x).checked_mul(sample).ok_or_else(short)?;
    let top = to_usize(rect.y);
    let [(luma_bytes, luma_rows), (chroma_bytes, chroma_rows)] = plane_extents(format, rect);
    let [(y, y_stride), (uv, uv_stride)] = destinations;
    let rows = [
        (planes[0], top, luma_rows, luma_bytes, y, y_stride),
        (planes[1], top / 2, chroma_rows, chroma_bytes, uv, uv_stride),
    ];
    for ((offset, pitch), first_row, rows, width, dst, dst_stride) in rows {
        for row in 0..rows {
            let src = first_row
                .checked_add(row)
                .and_then(|row| row.checked_mul(pitch))
                .and_then(|start| start.checked_add(offset)?.checked_add(x))
                .and_then(|start| bytes.get(start..start.checked_add(width)?))
                .ok_or_else(short)?;
            let dst = row
                .checked_mul(dst_stride)
                .and_then(|start| dst.get_mut(start..start.checked_add(width)?))
                .ok_or(Error::PlaneTooSmall("destination"))?;
            copy(dst, src);
        }
    }
    Ok(())
}

/// A plain row copy.
pub(crate) fn copy_row(dst: &mut [u8], src: &[u8]) {
    dst.copy_from_slice(src);
}

/// A row copy with SSE4.1's streaming loads (`MOVNTDQA`), which read
/// write-combined memory, as a mapping of video memory is, a cache line at
/// a time rather than a load at a time.
pub(crate) fn copy_row_streaming(dst: &mut [u8], src: &[u8]) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("sse4.1") {
        // SAFETY: SSE4.1 is there, checked above.
        unsafe { copy_streaming_sse41(dst, src) };
        return;
    }
    dst.copy_from_slice(src);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
fn copy_streaming_sse41(dst: &mut [u8], src: &[u8]) {
    use std::arch::x86_64::{__m128i, _mm_storeu_si128, _mm_stream_load_si128};
    let len = dst.len().min(src.len());
    // The streaming loads need 16-byte aligned sources: copy up to the
    // first aligned byte plainly.
    let head = src.as_ptr().align_offset(16).min(len);
    dst[..head].copy_from_slice(&src[..head]);
    let mut i = head;
    while i + 64 <= len {
        // SAFETY: `i + 64 <= len` bytes of both slices, `src + i` 16-byte
        // aligned; the stores are unaligned.
        unsafe {
            let from = src.as_ptr().add(i).cast::<__m128i>();
            let to = dst.as_mut_ptr().add(i).cast::<__m128i>();
            let a = _mm_stream_load_si128(from);
            let b = _mm_stream_load_si128(from.add(1));
            let c = _mm_stream_load_si128(from.add(2));
            let d = _mm_stream_load_si128(from.add(3));
            _mm_storeu_si128(to, a);
            _mm_storeu_si128(to.add(1), b);
            _mm_storeu_si128(to.add(2), c);
            _mm_storeu_si128(to.add(3), d);
        }
        i += 64;
    }
    dst[i..len].copy_from_slice(&src[i..len]);
}

/// A format a [`HostSurface`] is laid out in: the decoders' two.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HostFormat {
    /// `VA_FOURCC_NV12`: luma, then the chroma pairs.
    Nv12,
    /// `VA_FOURCC_P010`: as NV12, 16 bits a sample.
    P010,
}

impl HostFormat {
    /// The decoders' format's.
    pub(crate) fn of(format: SurfaceFormat) -> Self {
        match format {
            SurfaceFormat::Nv12 => Self::Nv12,
            SurfaceFormat::P010 => Self::P010,
        }
    }

    /// Its VA fourcc, render-target format and DRM format.
    fn codes(self) -> (u32, core::ffi::c_uint, u32) {
        match self {
            Self::Nv12 => (
                ffi::VA_FOURCC_NV12,
                ffi::VA_RT_FORMAT_YUV420,
                ffi::DRM_FORMAT_NV12,
            ),
            Self::P010 => (
                ffi::VA_FOURCC_P010,
                ffi::VA_RT_FORMAT_YUV420_10,
                ffi::DRM_FORMAT_P010,
            ),
        }
    }

    /// Where each plane of a [`HostSurface`] of `width` by `height` is, its
    /// offset and pitch, and the bytes of the whole: each plane's rows
    /// 256-byte aligned, and each plane at a page and a whole number of the
    /// first plane's rows from its first byte, as iHD writes a chroma plane
    /// whole rows after its luma plane (`super::import_check`). `None` past
    /// the address space.
    fn layout(self, width: u32, height: u32) -> Option<(Vec<(usize, usize)>, usize)> {
        let shapes = self.planes(width, height)?;
        let mut planes = Vec::with_capacity(shapes.len());
        let mut size = 0usize;
        let mut unit = PAGE;
        for (row, rows) in shapes {
            let offset = size.checked_next_multiple_of(unit)?;
            let pitch = row.checked_next_multiple_of(256)?;
            if planes.is_empty() {
                unit = (pitch / gcd(pitch, PAGE)).checked_mul(PAGE)?;
            }
            size = pitch.checked_mul(rows)?.checked_add(offset)?;
            planes.push((offset, pitch));
        }
        Some((planes, size))
    }

    /// The bytes a row and the rows of each of its planes, for a surface of
    /// `width` by `height`; `None` past the address space.
    fn planes(self, width: u32, height: u32) -> Option<Vec<(usize, usize)>> {
        let (w, h) = (to_usize(width), to_usize(height));
        let half = |v: usize| v.div_ceil(2);
        Some(match self {
            Self::Nv12 => vec![(w, h), (half(w) * 2, half(h))],
            Self::P010 => vec![(w.checked_mul(2)?, h), (half(w).checked_mul(4)?, half(h))],
        })
    }
}

/// A linear VA surface over host memory of ours ([`HostMemory`]), which
/// the GPU writes and the CPU reads cached: the staged copy back's. Each
/// plane's rows are 256-byte aligned, as AMD's linear surfaces need, and
/// each plane starts at a page.
pub(crate) struct HostSurface {
    display: Display,
    surface: ffi::VASurfaceID,
    /// The surface's bytes: its planes at `planes`' offsets and pitches.
    memory: HostMemory,
    planes: Vec<(usize, usize)>,
}

impl HostSurface {
    /// A surface of `format`, `width` by `height`, imported from host
    /// memory made for it: refused where the memory cannot be made, as
    /// with no `/dev/udmabuf` the process may open, or more bytes than
    /// udmabuf's `size_limit_mb` lets one dma-buf have (64 MiB at its
    /// stock value), or where the driver refuses the import.
    pub(crate) fn new(
        display: &Display,
        format: HostFormat,
        width: u32,
        height: u32,
    ) -> Result<Self, Error> {
        let too_large =
            || Error::Unsupported(format!("a host surface of {format:?} {width}x{height}"));
        let (planes, size) = format.layout(width, height).ok_or_else(too_large)?;
        let memory = HostMemory::new(size)?;
        let surface = import(
            display,
            format,
            (width, height),
            (memory.dma_buf(), memory.len()),
            &planes,
        )?;
        Ok(Self {
            display: display.clone_ref(),
            surface,
            memory,
            planes,
        })
    }

    /// Waits for the GPU's writes into the surface, a run's or a decode's,
    /// until `deadline` (`super::wait`): first on the memory's dma-buf,
    /// with no lock of the driver's held, so that the display's other calls
    /// go on meanwhile (`super::fence`), then by the surface's sync, which
    /// finds it done. Past the deadline, [`Error::Stalled`] with `what`;
    /// then the GPU may still write the memory.
    pub(crate) fn wait_for_writes(&self, deadline: MonoTime, what: &str) -> Result<(), Error> {
        super::fence::wait_for_writes(
            &**self.display.runtime(),
            deadline,
            [self.memory.dma_buf_fd()],
        );
        wait::wait_for(&self.display, Work::Surface(self.surface), deadline, what)
    }

    /// Its bytes, as the GPU last wrote them, and its planes' offsets and
    /// pitches. The caller has waited for the GPU's write
    /// ([`HostSurface::wait_for_writes`]).
    pub(crate) fn bytes(&self) -> (&[u8], &[(usize, usize)]) {
        (self.memory.bytes(), &self.planes)
    }
}

impl Drop for HostSurface {
    fn drop(&mut self) {
        // Made behind the display's calls in hand; the driver holds the
        // pages it imported until it destroys the surface.
        destroy_surface(&self.display, self.surface);
    }
}

/// Imports the `size` bytes of the dma-buf `fd` as a linear VA surface of
/// `format`, `width` by `height`, its planes at `planes`' offsets and
/// pitches: refused where the descriptor's 32-bit fields cannot hold them,
/// or where the driver refuses the import. The driver holds what it
/// imports until it destroys the surface, so `fd` may be closed after.
fn import(
    display: &Display,
    format: HostFormat,
    (width, height): (u32, u32),
    (fd, size): (RawFd, usize),
    planes: &[(usize, usize)],
) -> Result<ffi::VASurfaceID, Error> {
    let too_large = || {
        Error::Unsupported(format!(
            "an import of {format:?} {width}x{height} over {size} bytes"
        ))
    };
    // The descriptor's sizes and offsets are 32 bits.
    let u32_of = |v: usize| u32::try_from(v).map_err(|_| too_large());
    let (fourcc, rt_format, drm_format) = format.codes();
    let mut descriptor = ffi::VADRMPRIMESurfaceDescriptor {
        fourcc,
        width,
        height,
        num_objects: 1,
        num_layers: 1,
        ..Default::default()
    };
    descriptor.objects[0] = ffi::VADRMPRIMESurfaceObject {
        fd,
        size: u32_of(size)?,
        drm_format_modifier: ffi::DRM_FORMAT_MOD_LINEAR,
    };
    let mut layer = ffi::VADRMPRIMESurfaceLayer {
        drm_format,
        num_planes: u32_of(planes.len())?,
        object_index: [0; 4],
        offset: [0; 4],
        pitch: [0; 4],
    };
    for (n, &(offset, pitch)) in planes.iter().enumerate() {
        layer.offset[n] = u32_of(offset)?;
        layer.pitch[n] = u32_of(pitch)?;
    }
    descriptor.layers[0] = layer;
    let mut attributes = [
        ffi::VASurfaceAttrib::zeroed(),
        ffi::VASurfaceAttrib::zeroed(),
    ];
    attributes[0].type_ = ffi::VASurfaceAttribMemoryType;
    attributes[0].flags = ffi::VA_SURFACE_ATTRIB_SETTABLE;
    attributes[0].value.type_ = ffi::VAGenericValueTypeInteger;
    attributes[0].value.value.i =
        i32::from_ne_bytes(ffi::VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2.to_ne_bytes());
    attributes[1].type_ = ffi::VASurfaceAttribExternalBufferDescriptor;
    attributes[1].flags = ffi::VA_SURFACE_ATTRIB_SETTABLE;
    attributes[1].value.type_ = ffi::VAGenericValueTypePointer;
    attributes[1].value.value.p = (&raw mut descriptor).cast();
    let mut surface = ffi::VA_INVALID_SURFACE;
    check(
        "vaCreateSurfaces (import)",
        // SAFETY: one surface out; the attributes and the descriptor they
        // point at live for the call, and the driver holds the dma-buf it
        // imports for the surface's life.
        display.call(|raw| unsafe {
            ffi::vaCreateSurfaces(
                raw,
                rt_format,
                width,
                height,
                &mut surface,
                1,
                attributes.as_mut_ptr(),
                2,
            )
        })?,
    )?;
    Ok(surface)
}

/// `surface`'s memory, exported for reading (`vaExportSurfaceHandle`, its
/// layers composed into one): the descriptor, and its objects' dma-bufs,
/// owned.
fn export(
    display: &Display,
    surface: ffi::VASurfaceID,
) -> Result<(ffi::VADRMPRIMESurfaceDescriptor, Vec<OwnedFd>), Error> {
    let mut descriptor = ffi::VADRMPRIMESurfaceDescriptor::default();
    check(
        "vaExportSurfaceHandle",
        // SAFETY: a surface of this display, and a descriptor to fill.
        display.call(|raw| unsafe {
            ffi::vaExportSurfaceHandle(
                raw,
                surface,
                ffi::VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2,
                ffi::VA_EXPORT_SURFACE_READ_ONLY | ffi::VA_EXPORT_SURFACE_COMPOSED_LAYERS,
                (&raw mut descriptor).cast(),
            )
        })?,
    )?;
    let objects = usize::try_from(descriptor.num_objects).unwrap_or(0).min(4);
    // SAFETY: the export's descriptors, owned from here.
    let fds = descriptor.objects[..objects]
        .iter()
        .map(|object| unsafe { OwnedFd::from_raw_fd(object.fd) })
        .collect();
    Ok((descriptor, fds))
}

/// The staged way back: a [`HostSurface`] of the pool's format, and a
/// video-processing context that copies a decode surface into it.
pub(crate) struct Staging {
    /// Destroyed before the surface and the memory under it: the fields
    /// drop in their order.
    context: Option<Context>,
    host: HostSurface,
}

impl Staging {
    /// A staging surface for `width` by `height` frames of `format`.
    pub(crate) fn new(
        display: &Display,
        format: SurfaceFormat,
        width: u32,
        height: u32,
    ) -> Result<Self, Error> {
        let host = HostSurface::new(display, HostFormat::of(format), width, height)?;
        let context = Config::new(display, Profile::NONE, Entrypoint::VIDEO_PROC, &mut [])
            .and_then(|config| Context::new(display, config, width, height, &mut [host.surface]))?;
        Ok(Self {
            context: Some(context),
            host,
        })
    }

    /// The staging surface.
    pub(crate) fn surface(&self) -> ffi::VASurfaceID {
        self.host.surface
    }

    /// Copies the whole of `source` into the staging surface on the GPU and
    /// waits for it, until `deadline` (`super::wait`); then the frame is in
    /// [`Staging::bytes`].
    pub(crate) fn stage(&self, source: ffi::VASurfaceID, deadline: MonoTime) -> Result<(), Error> {
        let Some(context) = &self.context else {
            return Err(Error::Unsupported(
                "a staging surface with no context".into(),
            ));
        };
        copy(&self.host.display, context, source, self.host.surface, None)?;
        self.host
            .wait_for_writes(deadline, "the copy back of a decoded frame in time")
    }

    /// The staged frame's bytes, and its planes' offsets and pitches.
    pub(crate) fn bytes(&self) -> (&[u8], [(usize, usize); 2]) {
        let (bytes, planes) = self.host.bytes();
        let plane = |n: usize| planes.get(n).copied().unwrap_or((usize::MAX, 0));
        (bytes, [plane(0), plane(1)])
    }
}

/// Rows a run copies ([`copy`]): `rows` of them from `top`, `width` wide
/// from the left.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Band {
    pub(crate) width: u32,
    pub(crate) top: u32,
    pub(crate) rows: u32,
}

/// Submits a video-processing run on `context` that copies `source` into
/// `target`, a surface of the same format: the whole of it into the whole
/// of `target`, of the same size, where `band` is `None`, and else the
/// band's rows into as many at the top of `target`, 1:1. The same colour
/// standard, range and siting both sides, so that every sample is copied
/// as it is, as every frame of the decoders' checks was through the staged
/// copy back (docs/video/vaapi.md, "The copy back"). The run is not
/// waited for.
pub(crate) fn copy(
    display: &Display,
    context: &Context,
    source: ffi::VASurfaceID,
    target: ffi::VASurfaceID,
    band: Option<Band>,
) -> Result<(), Error> {
    let regions = match band {
        None => None,
        Some(band) => {
            let past = || Error::Unsupported(format!("{band:?}, past a video-processing region"));
            let width = u16::try_from(band.width).map_err(|_| past())?;
            let rows = u16::try_from(band.rows).map_err(|_| past())?;
            let top = i16::try_from(band.top).map_err(|_| past())?;
            let rectangle = |y| vpp::VARectangle {
                x: 0,
                y,
                width,
                height: rows,
            };
            Some((rectangle(top), rectangle(0)))
        }
    };
    let mut params = vpp::VAProcPipelineParameterBuffer::zeroed();
    params.surface = source;
    if let Some((from, to)) = &regions {
        params.surface_region = from;
        params.output_region = to;
    }
    params.surface_color_standard = vpp::VAProcColorStandardBT709;
    params.output_color_standard = vpp::VAProcColorStandardBT709;
    let buffer = super::buffer::Buffer::with_value(
        display,
        context.id,
        vpp::VAProcPipelineParameterBufferType,
        &params,
    )?;
    // The pipeline's two rectangles are `regions`', borrowed past the
    // render.
    super::buffer::render_picture(display, context.id, target, &[&[&buffer]])
}

/// Destroys `surface`, made by `vaCreateSurfaces` on `display` and used by
/// nothing that is not destroyed before it: now, or behind the display's
/// calls in hand.
pub(crate) fn destroy_surface(display: &Display, mut surface: ffi::VASurfaceID) {
    // SAFETY: a surface of the display, destroyed once, here.
    display.release(move |raw| unsafe {
        ffi::vaDestroySurfaces(raw, &mut surface, 1);
    });
}

/// The `mapped` way back, for comparison: each decode surface's own
/// dma-buf, mapped once, read with streaming loads.
#[derive(Default)]
pub(crate) struct Mappings {
    by_surface: HashMap<ffi::VASurfaceID, Mapping>,
}

struct Mapping {
    fd: OwnedFd,
    map: *mut u8,
    len: usize,
    planes: [(usize, usize); 2],
}

// SAFETY: a read-only mapping of the dma-buf, owned by this value.
unsafe impl Send for Mapping {}

unsafe extern "C" {
    fn mmap(
        address: *mut core::ffi::c_void,
        length: usize,
        protection: core::ffi::c_int,
        flags: core::ffi::c_int,
        fd: core::ffi::c_int,
        offset: i64,
    ) -> *mut core::ffi::c_void;
    fn munmap(address: *mut core::ffi::c_void, length: usize) -> core::ffi::c_int;
    fn ioctl(fd: core::ffi::c_int, request: core::ffi::c_ulong, ...) -> core::ffi::c_int;
}

/// `DMA_BUF_IOCTL_SYNC` (linux/dma-buf.h): `_IOW('b', 0, struct
/// dma_buf_sync)`, the struct being one `u64`; and its flags.
const DMA_BUF_IOCTL_SYNC: core::ffi::c_ulong =
    (1 << 30) | (8 << 16) | ((b'b' as core::ffi::c_ulong) << 8);
const DMA_BUF_SYNC_READ: u64 = 1;
const DMA_BUF_SYNC_START: u64 = 0;
const DMA_BUF_SYNC_END: u64 = 1 << 2;

impl Mapping {
    /// The surface's own dma-buf, mapped: refused, with its layout, where
    /// it is not one linear dma-buf.
    fn of(display: &Display, surface: ffi::VASurfaceID) -> Result<Self, Error> {
        let (descriptor, fds) = export(display, surface)?;
        let objects = fds.len();
        let layer = descriptor.layers[0];
        let object = descriptor.objects[0];
        if objects != 1 || object.drm_format_modifier != ffi::DRM_FORMAT_MOD_LINEAR {
            return Err(Error::Unsupported(format!(
                "the decode surface is {objects} objects, modifier {:#x}, pitches {:?}, \
                 offsets {:?}: not one linear dma-buf",
                object.drm_format_modifier, layer.pitch, layer.offset
            )));
        }
        let len = usize::try_from(object.size).unwrap_or(0);
        let fd = fds
            .into_iter()
            .next()
            .ok_or(Error::Unsupported("no dma-buf".into()))?;
        // SAFETY: a read-only shared mapping of the whole dma-buf.
        let map = unsafe { mmap(core::ptr::null_mut(), len, 0x1, 0x01, fd.as_raw_fd(), 0) };
        if map as usize == usize::MAX {
            return Err(Error::Unsupported(format!(
                "mapping the decode surface: {}",
                std::io::Error::last_os_error()
            )));
        }
        let planes = [
            (to_usize(layer.offset[0]), to_usize(layer.pitch[0])),
            (to_usize(layer.offset[1]), to_usize(layer.pitch[1])),
        ];
        Ok(Self {
            fd,
            map: map.cast(),
            len,
            planes,
        })
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: the mapping made in `Mappings::read`, unmapped once.
        unsafe { munmap(self.map.cast(), self.len) };
    }
}

impl Mappings {
    /// Begins a read of `surface` through its own mapping, made on first
    /// use: its bytes, its planes' offsets and pitches, and the dma-buf's
    /// descriptor, whose read [`end_read`] ends. A surface that is not one
    /// linear dma-buf is refused, with its layout.
    pub(crate) fn begin_read(
        &mut self,
        display: &Display,
        surface: ffi::VASurfaceID,
    ) -> Result<MappedRead<'_>, Error> {
        let mapping = match self.by_surface.entry(surface) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(Mapping::of(display, surface)?)
            }
        };
        let fd = mapping.fd.as_raw_fd();
        sync(fd, DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ);
        // SAFETY: `len` mapped bytes, alive while the mapping is; the
        // decode into them has been waited for.
        let bytes = unsafe { core::slice::from_raw_parts(mapping.map, mapping.len) };
        Ok(MappedRead {
            bytes,
            planes: mapping.planes,
            fd,
        })
    }
}

/// A read [`Mappings::begin_read`] began: the surface's bytes, its planes'
/// offsets and pitches, and the dma-buf whose read [`end_read`] ends.
pub(crate) struct MappedRead<'a> {
    pub(crate) bytes: &'a [u8],
    pub(crate) planes: [(usize, usize); 2],
    pub(crate) fd: RawFd,
}

/// Ends a read [`Mappings::begin_read`] began of the dma-buf `fd`.
pub(crate) fn end_read(fd: RawFd) {
    sync(fd, DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ);
}

/// `DMA_BUF_IOCTL_SYNC` of the dma-buf `fd` with `flags`.
fn sync(fd: RawFd, flags: u64) {
    // SAFETY: a dma-buf's descriptor and a `struct dma_buf_sync`.
    unsafe { ioctl(fd, DMA_BUF_IOCTL_SYNC, &raw const flags) };
}

fn to_usize(value: u32) -> usize {
    usize::try_from(value).unwrap_or(usize::MAX)
}

#[cfg(test)]
mod tests {
    use super::{HostFormat, Rect, SurfaceFormat, copy_rect, copy_row};
    use crate::vaapi::va::display::Error;

    /// An NV12 frame of 4x4 in 8 bytes a row: luma 0 to 31, then chroma
    /// from 100.
    fn frame() -> Vec<u8> {
        (0..32).chain(100..116).collect()
    }

    fn rect(x: u32, y: u32, width: u32, height: u32) -> Rect {
        Rect {
            x,
            y,
            width,
            height,
        }
    }

    #[test]
    fn a_rectangle_is_copied_row_by_row_into_each_plane() {
        let (mut y, mut uv) = (vec![0; 4], vec![0; 2]);
        let copied = copy_rect(
            &frame(),
            [(0, 8), (32, 8)],
            SurfaceFormat::Nv12,
            rect(2, 2, 2, 2),
            [(&mut y, 2), (&mut uv, 2)],
            copy_row,
        );
        assert!(copied.is_ok());
        assert_eq!(y, [18, 19, 26, 27]);
        assert_eq!(uv, [110, 111]);
    }

    #[test]
    fn strides_and_offsets_past_the_address_space_are_refused() {
        let bytes = frame();
        let copy = |planes: [(usize, usize); 2], stride: usize, at: Rect| {
            let (mut y, mut uv) = (vec![0; 16], vec![0; 8]);
            copy_rect(
                &bytes,
                planes,
                SurfaceFormat::Nv12,
                at,
                [(&mut y, stride), (&mut uv, stride)],
                copy_row,
            )
        };
        let short = |result: Result<(), Error>| {
            matches!(result, Err(Error::Unsupported(_) | Error::PlaneTooSmall(_)))
        };
        // A pitch whose rows wrap, an offset at the top, a destination
        // stride whose second row wraps, and a crop past the frame: each
        // refused, none indexing out of its slice.
        assert!(short(copy([(0, usize::MAX), (32, 8)], 4, rect(0, 0, 4, 2))));
        assert!(short(copy([(usize::MAX, 8), (32, 8)], 4, rect(0, 0, 4, 2))));
        assert!(short(copy([(0, 8), (32, 8)], usize::MAX, rect(0, 0, 4, 2))));
        assert!(short(copy([(0, 8), (32, 8)], 4, rect(u32::MAX, 0, 4, 2))));
        assert!(short(copy([(0, 8), (32, 8)], 4, rect(0, 0, 4, 8))));
    }

    /// A staged surface's planes each start a page and lie whole rows of the first plane's pitch
    /// from its first byte, at rows of 256 bytes: the sizes the decoders make, at 8 and 10 bits,
    /// among them iHD's staged P010 1920x1080, whose chroma plane a page past the luma's rows
    /// lay part of a row in.
    #[test]
    fn staged_planes_start_a_page_whole_rows_in() {
        for format in [HostFormat::Nv12, HostFormat::P010] {
            for (width, height) in [
                (1920, 1080),
                (1920, 1088),
                (1280, 720),
                (3840, 2160),
                (640, 368),
                (1366, 768),
                (8192, 4352),
                (16, 16),
            ] {
                let (planes, size) = format.layout(width, height).expect("a layout");
                let [(luma, pitch), (chroma, chroma_pitch)] = planes[..] else {
                    panic!("{width}x{height}: two planes");
                };
                let rows = height as usize;
                assert_eq!(luma, 0);
                assert_eq!(pitch, chroma_pitch);
                assert_eq!(pitch % 256, 0, "{width}x{height}");
                assert_eq!(chroma % 4096, 0, "{width}x{height}: a page");
                assert_eq!(chroma % pitch, 0, "{width}x{height}: whole rows");
                assert!(chroma >= pitch * rows, "{width}x{height}");
                assert!(chroma - pitch * rows < pitch / super::gcd(pitch, 4096) * 4096);
                assert_eq!(size, chroma + pitch * rows.div_ceil(2));
            }
        }
    }
}
