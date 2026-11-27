//! A decoded frame, for every codec: its surface, held until the frame is
//! dropped, what the decoder knows of it, and its readback into caller
//! planes, NV12 for 8-bit frames and P010 for 10-bit ones, and two planes of
//! the range extensions' packed frames.

use std::sync::Arc;

use crate::core::backend::{Chroma, DeviceWrite};
use crate::core::device_memory::GuestTarget;
use crate::core::surfaces::crop_within;

use super::display::{Display, Error};
use super::export::SurfaceExport;
use super::readback::ReadbackTimes;
use super::surface::{Rect, Surface, SurfaceFormat, SurfaceReadBack};
use super::wait::{self, Deadline};
use super::zero_copy::PackedTarget;

/// A decoded frame: its surface, held until the frame is dropped, and
/// what the decoder knows of it.
pub struct Frame {
    pub(crate) surface: Arc<Surface>,
    pub(crate) id: u64,
    pub(crate) poc: i32,
    pub(crate) idr: bool,
    pub(crate) reference: bool,
    pub(crate) rect: Rect,
    pub(crate) coded_size: (u32, u32),
    pub(crate) full_range: bool,
    /// The deadline of the decode call that gave the frame, which its
    /// waits and read back share (`super::wait`).
    pub(crate) deadline: Deadline,
}

impl Frame {
    /// The id of the sample that carried the frame.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// The frame's picture order count: its place in display order since
    /// the last IDR picture (H.264's `PicOrderCnt()`, HEVC's
    /// `PicOrderCntVal`).
    pub fn poc(&self) -> i32 {
        self.poc
    }

    /// Whether the frame is an IDR picture.
    pub fn is_idr(&self) -> bool {
        self.idr
    }

    /// Whether other frames may predict from this one.
    pub fn is_reference(&self) -> bool {
        self.reference
    }

    /// The width after the sequence's cropping.
    pub fn width(&self) -> u32 {
        self.rect.width
    }

    /// The height after the sequence's cropping.
    pub fn height(&self) -> u32 {
        self.rect.height
    }

    /// The decoded size before cropping.
    pub fn coded_size(&self) -> (u32, u32) {
        self.coded_size
    }

    /// Where the crop starts in the decoded picture: its first column and row, in luma samples
    /// (even, as 4:2:0's crops are), [`Frame::width`] by [`Frame::height`] from there.
    pub fn crop_origin(&self) -> (u32, u32) {
        (self.rect.x, self.rect.y)
    }

    /// Narrows the crop to the `width` by `height` samples from (`x`, `y`) of it: what its writes
    /// and copies take from then on, as they take an odd crop. `x` and `y` are even, as 4:2:0's
    /// crops start, and the rect lies inside the crop; [`Error::Unsupported`] otherwise, the crop
    /// left as it was.
    pub fn crop_within(&mut self, x: u32, y: u32, width: u32, height: u32) -> Result<(), Error> {
        let Rect {
            x: crop_x,
            y: crop_y,
            width: crop_width,
            height: crop_height,
        } = self.rect;
        let crop = [crop_x, crop_y, crop_width, crop_height];
        let [x, y, width, height] = crop_within(crop, [x, y, width, height]).ok_or_else(|| {
            Error::Unsupported(format!(
                "a crop of {width}x{height} at ({x}, {y}) of a frame of {crop_width}x{crop_height}"
            ))
        })?;
        self.rect = Rect {
            x,
            y,
            width,
            height,
        };
        Ok(())
    }

    /// The frame's surface as dma-bufs, for another device to read its planes
    /// (`super::export`): its pool's export of the surface, made at the surface's first ask.
    /// It reads the decode only once [`Frame::sync`] returned.
    pub fn export(&self) -> Result<Arc<SurfaceExport>, Error> {
        self.surface.export()
    }

    /// The frame's surface as the Direct3D 12 texture it is, on Windows, for the plane draws to
    /// read its planes through its resource (`super::d3d12`); `None` where its pool's surfaces
    /// are the driver's own. It reads the decode only once [`Frame::sync`] returned.
    #[cfg(windows)]
    pub fn texture(&self) -> Option<Arc<super::d3d12::SharedTexture>> {
        self.surface.texture()
    }

    /// A hold of the frame's surface, which keeps it out of its pool, so that no later frame is
    /// decoded into it, while the hold lives: for another device's read of the surface that may
    /// outlast the frame.
    pub fn hold(&self) -> FrameHold {
        FrameHold {
            _surface: Arc::clone(&self.surface),
        }
    }

    /// Whether the samples use the full range (0 to 255 at 8 bits) rather
    /// than the video range (16 to 235): the VUI's `video_full_range_flag`
    /// (H.264 E.2.1, HEVC E.3.1), false without a VUI.
    pub fn is_full_range(&self) -> bool {
        self.full_range
    }

    /// The bits a sample holds: 8 for an NV12, YUY2 or XYUV frame, 10 for a P010, Y210 or Y410
    /// one, 12 for a P012, Y212 or Y412 one.
    pub fn bit_depth(&self) -> u32 {
        u32::from(self.surface.format().bit_depth())
    }

    /// How the frame's chroma is sampled: 4:2:0, or the range extensions' 4:2:2 or 4:4:4
    /// ([`super::HevcDecoder::lands_range_extensions`]).
    pub fn chroma(&self) -> Chroma {
        self.surface.format().chroma()
    }

    /// Whether the frame is 4:4:4 ([`Self::chroma`]).
    pub fn is_4_4_4(&self) -> bool {
        self.chroma() == Chroma::Yuv444
    }

    /// Whether the frame's pixels are packed in one plane, which [`ReadBack::copy_packed_into`]
    /// reads back, rather than in two, which [`ReadBack::copy_into`] reads back.
    pub fn is_packed(&self) -> bool {
        self.surface.format().planes() == 1
    }

    /// Whether the frame is 8-bit 4:4:4 in XYUV, which [`Frame::write_4_4_4_to`] writes.
    pub fn is_xyuv(&self) -> bool {
        self.surface.format() == SurfaceFormat::Xyuv
    }

    /// The name of its surface's VA format (`NV12`, `P010`, `Y410` and the like).
    pub fn surface_format_name(&self) -> &'static str {
        self.surface.format().name()
    }

    /// What the frames of this frame's surface pool came back through so
    /// far, and how long each part took.
    pub fn readback_times(&self) -> ReadbackTimes {
        self.surface.readback_times()
    }

    /// Whether the frame was decoded into the target its decode was handed
    /// (`H264Decoder::decode_into`): its picture is then in the target's memory once it is
    /// decoded ([`Frame::sync`]), and nothing is to be copied.
    pub fn is_in_target(&self) -> bool {
        self.surface.target().is_some()
    }

    /// Waits until the frame is decoded.
    pub fn sync(&self) -> Result<(), Error> {
        wait::by(&self.deadline, || self.surface.sync())
    }

    /// Reads the frame back where the CPU reaches it, for the copies
    /// [`ReadBack`] makes: waits for the decode, and has the driver put the
    /// frame in host memory, so that those copies run on the CPU alone. The
    /// frame's pool reads no other frame back until the result drops.
    pub fn read_back(&self) -> Result<ReadBack<'_>, Error> {
        Ok(ReadBack {
            frame: self,
            surface: wait::by(&self.deadline, || self.surface.read_back())?,
        })
    }

    /// Copies an 8-bit frame, cropped, into NV12 planes, as
    /// [`ReadBack::copy_nv12_to`] does, reading it back first.
    pub fn copy_nv12_to(
        &self,
        y: &mut [u8],
        y_stride: usize,
        uv: &mut [u8],
        uv_stride: usize,
    ) -> Result<(), Error> {
        self.read_back()?.copy_nv12_to(y, y_stride, uv, uv_stride)
    }

    /// Copies a 10-bit frame, cropped, into P010 planes, as
    /// [`ReadBack::copy_p010_to`] does, reading it back first.
    pub fn copy_p010_to(
        &self,
        y: &mut [u8],
        y_stride: usize,
        uv: &mut [u8],
        uv_stride: usize,
    ) -> Result<(), Error> {
        self.read_back()?.copy_p010_to(y, y_stride, uv, uv_stride)
    }
}

/// A hold of a frame's surface ([`Frame::hold`]): the surface goes back to its pool once the
/// frame and every hold are dropped.
pub struct FrameHold {
    _surface: Arc<Surface>,
}

/// A frame read back where the CPU reaches it ([`Frame::read_back`]), kept
/// there until dropped; its copies run on the CPU alone.
pub struct ReadBack<'a> {
    frame: &'a Frame,
    surface: SurfaceReadBack<'a>,
}

impl ReadBack<'_> {
    /// Copies an 8-bit frame, cropped, into NV12 planes: `y` with
    /// `y_stride` bytes a row, and the interleaved Cb/Cr plane `uv` with
    /// `uv_stride`, [`Frame::height`] / 2 rows.
    pub fn copy_nv12_to(
        &self,
        y: &mut [u8],
        y_stride: usize,
        uv: &mut [u8],
        uv_stride: usize,
    ) -> Result<(), Error> {
        self.copy_rect(
            SurfaceFormat::Nv12,
            self.frame.rect,
            y,
            y_stride,
            uv,
            uv_stride,
        )
    }

    /// Copies a 10-bit frame, cropped, into P010 planes: each sample a
    /// little-endian 16-bit word with its 10 bits at the top, `y` with
    /// `y_stride` bytes a row and the interleaved Cb/Cr plane `uv` with
    /// `uv_stride`, [`Frame::height`] / 2 rows. CoreVideo's `x420` and
    /// `xf20` are laid out so.
    pub fn copy_p010_to(
        &self,
        y: &mut [u8],
        y_stride: usize,
        uv: &mut [u8],
        uv_stride: usize,
    ) -> Result<(), Error> {
        self.copy_rect(
            SurfaceFormat::P010,
            self.frame.rect,
            y,
            y_stride,
            uv,
            uv_stride,
        )
    }

    /// [`ReadBack::copy_nv12_to`] or [`ReadBack::copy_p010_to`], by the
    /// frame's depth, into planes of `y_rows` and `uv_rows` rows that may
    /// be smaller than the frame: they take its top left, as much as fits.
    /// A guest's format description may be smaller than its stream, and
    /// the guest's buffers are the description's size.
    pub fn copy_fitting_to(
        &self,
        (y, y_stride, y_rows): (&mut [u8], usize, usize),
        (uv, uv_stride, uv_rows): (&mut [u8], usize, usize),
    ) -> Result<(), Error> {
        self.copy_into(
            PlaneRows {
                bytes: y,
                bytes_per_row: y_stride,
                rows: y_rows,
                row_bytes: y_stride,
            },
            PlaneRows {
                bytes: uv,
                bytes_per_row: uv_stride,
                rows: uv_rows,
                row_bytes: uv_stride,
            },
        )
    }

    /// [`ReadBack::copy_fitting_to`] into planes whose rows hold `row_bytes` of the picture each,
    /// which may be fewer than their `bytes_per_row`: what lies past them in a row (another
    /// texture's bytes, inside a larger plane) is left as it was.
    pub fn copy_into(&self, y: PlaneRows<'_>, uv: PlaneRows<'_>) -> Result<(), Error> {
        let frame = self.frame;
        let format = frame.surface.format();
        let sample = format.bytes_per_sample();
        let fit = |bytes: usize| u32::try_from(bytes / sample.max(1)).unwrap_or(u32::MAX);
        // A chroma row holds a pair for every two columns, the last one too
        // where the width is odd, so an even width fits where its bytes do.
        let chroma_columns = fit(uv.row_bytes.min(uv.bytes_per_row)) & !1;
        let rect = Rect {
            width: frame
                .rect
                .width
                .min(fit(y.row_bytes.min(y.bytes_per_row)))
                .min(chroma_columns.max(1)),
            height: (frame.rect.height)
                .min(u32::try_from(y.rows).unwrap_or(u32::MAX))
                .min(u32::try_from(uv.rows).unwrap_or(u32::MAX).saturating_mul(2)),
            ..frame.rect
        };
        self.copy_rect(
            format,
            rect,
            y.bytes,
            y.bytes_per_row,
            uv.bytes,
            uv.bytes_per_row,
        )
    }

    /// Copies a packed frame ([`Frame::is_packed`]: the range extensions' 4:2:2 and 4:4:4),
    /// cropped, into two planes as CoreVideo's two-plane formats of its chroma format and depth
    /// (`422v`, `x422`, `tv22`, `444v`, `x444`, `tv44` and their full-range twins) and ffmpeg's
    /// `nv16`, `p210`, `p212`, `nv24`, `p410` and `p412` lay them out: `y`, and `cbcr`, each pair's
    /// Cb then Cr, as many rows as `y`, whose rows hold `row_bytes` of the picture each; a
    /// sample a byte at 8 bits, else a little-endian 16-bit word, its bits at the top. Planes of
    /// fewer rows or columns than the frame take its top left, as [`ReadBack::copy_into`]'s do.
    pub fn copy_packed_into(&self, y: PlaneRows<'_>, cbcr: PlaneRows<'_>) -> Result<(), Error> {
        let frame = self.frame;
        let format = frame.surface.format();
        let sample = format.bytes_per_sample();
        let fit = |n: usize| u32::try_from(n).unwrap_or(u32::MAX);
        let chroma_bytes = cbcr.row_bytes.min(cbcr.bytes_per_row) / (2 * sample);
        // A 4:2:2 row holds a pair for every two columns, the last one too where the width is
        // odd, so an even width fits where its pairs do.
        let chroma_columns = match format.chroma() {
            Chroma::Yuv422 => fit(chroma_bytes).saturating_mul(2),
            Chroma::Yuv420 | Chroma::Yuv444 => fit(chroma_bytes),
        };
        let rect = Rect {
            width: (frame.rect.width)
                .min(fit(y.row_bytes.min(y.bytes_per_row) / sample))
                .min(chroma_columns),
            height: (frame.rect.height).min(fit(y.rows)).min(fit(cbcr.rows)),
            ..frame.rect
        };
        self.surface.copy_packed_to(
            rect,
            (y.bytes, y.bytes_per_row),
            (cbcr.bytes, cbcr.bytes_per_row),
        )
    }

    fn copy_rect(
        &self,
        format: SurfaceFormat,
        rect: Rect,
        y: &mut [u8],
        y_stride: usize,
        uv: &mut [u8],
        uv_stride: usize,
    ) -> Result<(), Error> {
        if self.frame.surface.format() != format {
            return Err(Error::Unsupported(format!(
                "reading a {} frame back as {}",
                self.frame.surface.format().name(),
                format.name()
            )));
        }
        self.surface
            .copy_planes_to(rect, y, y_stride, uv, uv_stride)
    }
}

/// One plane's rows, which [`ReadBack::copy_into`] copies a frame into.
pub struct PlaneRows<'a> {
    /// The plane, from its first row's first byte.
    pub bytes: &'a mut [u8],
    /// Bytes from one of its rows to the next.
    pub bytes_per_row: usize,
    /// Its rows.
    pub rows: usize,
    /// The bytes of one of its rows that hold the picture.
    pub row_bytes: usize,
}

/// A display's imports of guest buffers, which its frames are written into by video-processing
/// runs ([`Frame::write_to_guest`]): kept by each buffer's identity until its mapping goes.
pub struct GuestWrites(Arc<super::zero_copy::GuestImports>);

impl GuestWrites {
    /// No imports yet, for frames of `display`.
    pub fn new(display: &Display) -> Self {
        Self(super::zero_copy::imports(display))
    }
}

impl Frame {
    /// Writes the frame into `target`, a buffer lent by its memory, by video-processing runs from
    /// its surface, with nothing for the CPU to copy (`super::zero_copy`), once its decode is
    /// done; [`DeviceWrite::Declined`] with why where the driver or the buffer's layout takes
    /// none, and the frame is to be copied by the CPU instead. `writes` keeps the buffer's import
    /// for the next frame written there.
    ///
    /// The run writes the buffer's whole width and height from the crop's first sample, as much
    /// of it as the decoded (coded) picture holds: the rows and columns past an odd crop that a
    /// buffer laid out to hold the coded picture's even rows and columns takes. Those of a frame
    /// past what the buffer holds are not written.
    pub fn write_to_guest(
        &self,
        target: &GuestTarget<'_>,
        writes: &GuestWrites,
    ) -> Result<DeviceWrite, Error> {
        self.write_to_guest_telling(target, writes, &mut || {})
    }

    /// [`Self::write_to_guest`], calling `decoded` once the frame's decode is done and the runs
    /// that write it are issued behind it, before they are waited for: from then on the write is
    /// the GPU's work alone. Not called where the decode failed, or the runs were not issued.
    pub fn write_to_guest_telling(
        &self,
        target: &GuestTarget<'_>,
        writes: &GuestWrites,
        decoded: &mut dyn FnMut(),
    ) -> Result<DeviceWrite, Error> {
        let rect = self.rect;
        let (coded_width, coded_height) = self.surface.size();
        let size = (
            target.width().min(coded_width.saturating_sub(rect.x)),
            target.height().min(coded_height.saturating_sub(rect.y)),
        );
        wait::by(&self.deadline, || {
            super::zero_copy::write(self, target, &writes.0, size, decoded)
        })
    }

    /// Writes the frame, an XYUV one ([`Frame::is_xyuv`]), into `target` as its XYUV surface
    /// holds it, by video-processing runs from its surface as [`Frame::write_to_guest`] writes a
    /// 4:2:0 frame (`super::zero_copy`), once its decode is done: `target`'s whole width and
    /// height from the crop's first sample, which the decoded (coded) picture holds.
    /// [`DeviceWrite::Declined`] with why where the driver or the buffer's layout takes none, or
    /// the target reaches past the decoded picture, and nothing was written.
    pub fn write_4_4_4_to(
        &self,
        target: &PackedTarget<'_>,
        writes: &GuestWrites,
    ) -> Result<DeviceWrite, Error> {
        let rect = self.rect;
        let (coded_width, coded_height) = self.surface.size();
        let size = (target.width(), target.height());
        if size.0 > coded_width.saturating_sub(rect.x)
            || size.1 > coded_height.saturating_sub(rect.y)
        {
            return Ok(DeviceWrite::Declined(format!(
                "a buffer of {}x{} from ({}, {}) of a decoded picture of {coded_width}x\
                 {coded_height}",
                size.0, size.1, rect.x, rect.y
            )));
        }
        wait::by(&self.deadline, || {
            super::zero_copy::write_packed(self, target, &writes.0, size, &mut || {})
        })
    }
}
