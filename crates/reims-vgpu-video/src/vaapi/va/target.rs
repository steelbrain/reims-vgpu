//! Render targets over memory of the caller's ([`Target`]): a decode's picture written by the
//! hardware straight into that memory, as a decoder's own surface is, where the caller hands the
//! decode a target for it ([`TargetFor`], `H264Decoder::decode_into`,
//! `HevcDecoder::decode_into`).
//!
//! - **The import.** The memory is one dma-buf, both planes in it at their own offsets and
//!   pitches, imported as a linear surface (`VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2`, one object,
//!   `DRM_FORMAT_MOD_LINEAR`), of the picture's format and coded width, and as tall as its planes
//!   need to be declared to span, from the luma plane's first byte, the chroma plane's rows of the
//!   picture ([`declared_rows`]): the coded height where the chroma plane lies right after the
//!   luma's rows, and more where it lies further on, the luma plane then declared over rows the
//!   decode does not write, the gap's and the chroma plane's first. The driver writes the coded
//!   picture's rows from each plane's first byte. A driver that refuses the import answers it at
//!   `vaCreateSurfaces`, and the decode then takes one of its own surfaces.
//!
//!   **Measured** on radeonsi (Mesa 26.2.6; a Radeon 780M and an RX 7900 XTX; libva 2.24.1,
//!   Linux 7.3.1; 2026-11-08; the target check and a libva program of our own): a decode writes
//!   an imported surface only within the bytes its planes as declared span from the luma plane's
//!   first byte, the luma's pitch times the import's height and the chroma's times half of it.
//!   Of a chroma plane further on than right after the luma's rows, the rows past that are never
//!   written and the luma plane's first row is overwritten, with no error (240x144 NV12 at
//!   256-byte rows: a 256-byte gap lost the chroma's last row, a 4096-byte gap its last 16).
//!   Imported as tall as [`declared_rows`] says, the luma plane declared over the gap and over
//!   the chroma plane's first rows, every sample was written and no byte outside the planes'
//!   rows changed; an import whose declared planes run past the dma-buf is refused. Its video
//!   processing writes the same surfaces at their offsets (docs/video/bugs.md, U28).
//! - **The references.** VA names a picture's references by their render targets. A driver that
//!   reads a reference from its target's memory would read whatever the caller wrote there since,
//!   so a caller hands targets only to a driver its check found keeping its references apart
//!   (`super::target_check`). The target itself, the surface, stays the reference's name while the
//!   decoder holds it: a decoder holds the target of each picture it references, and a target the
//!   decoder still holds ([`Target::is_held`]) is not handed a decode again, a new import of the
//!   same memory being made for it instead.
//! - **What it keeps.** A target holds its memory's descriptor and what the caller gives it to
//!   keep ([`Target::import`]), until the last of it and its pictures goes; the surface is then
//!   destroyed (`Display::release`).

use std::os::fd::{AsRawFd as _, BorrowedFd, OwnedFd};
use std::sync::Arc;

use super::display::{Display, Error, check};
use super::ffi;
use super::surface::SurfaceFormat;

/// The format of a decode's picture, which its target is imported as: 8-bit 4:2:0 in NV12, 10-bit
/// in P010 (the samples in the high bits of each 16-bit word).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetFormat {
    /// NV12: H.264's pictures and HEVC Main's.
    Nv12,
    /// P010: HEVC Main 10's.
    P010,
}

impl TargetFormat {
    /// The bytes of a sample of a plane.
    pub fn bytes_per_sample(self) -> u64 {
        match self {
            TargetFormat::Nv12 => 1,
            TargetFormat::P010 => 2,
        }
    }

    pub(crate) fn of(format: SurfaceFormat) -> TargetFormat {
        match format {
            SurfaceFormat::Nv12 => TargetFormat::Nv12,
            SurfaceFormat::P010 => TargetFormat::P010,
        }
    }

    fn surface(self) -> SurfaceFormat {
        match self {
            TargetFormat::Nv12 => SurfaceFormat::Nv12,
            TargetFormat::P010 => SurfaceFormat::P010,
        }
    }
}

/// Where a plane of a target lies in its dma-buf: its first byte, and the bytes from one row to
/// the next.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TargetPlane {
    /// The plane's first byte, from the dma-buf's.
    pub offset: u64,
    /// The bytes from one of its rows to the next.
    pub bytes_per_row: u64,
}

/// What a decode asks of its target ([`TargetFor`]): the picture's format and coded size, which
/// the target holds from each plane's first byte.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TargetShape {
    /// The picture's format.
    pub format: TargetFormat,
    /// The coded picture's width and height in luma samples: every row the hardware writes.
    pub coded: (u32, u32),
}

/// A target for a decode's picture, made as the picture's storage is: asked once, with the
/// picture's shape, as the decoder makes the decode's picture; `None` where the caller has none
/// for it, and the picture is decoded into a surface of the decoder's own. A target of another
/// shape is not used, as if `None` were answered.
pub type TargetFor = Box<dyn FnOnce(TargetShape) -> Option<Target> + Send>;

/// The height a target of a picture `height` rows tall, its planes at `luma` and `chroma`, is
/// imported as (module docs): the least even height, from the picture's, at which the planes as
/// declared, the luma plane's rows and the chroma plane's half as many, span as many bytes as lie
/// from the luma plane's first byte to the end of the chroma plane's rows of the picture. Where
/// the chroma plane lies right after the luma's rows, that is the picture's height made even.
/// The planes' offsets and pitches are each below 4 GiB.
fn declared_rows(height: u32, luma: TargetPlane, chroma: TargetPlane) -> u64 {
    let even = u64::from(height.next_multiple_of(2));
    let chroma_end = chroma.offset + chroma.bytes_per_row * u64::from(height.div_ceil(2));
    let reach = chroma_end.saturating_sub(luma.offset);
    let spanned = luma.bytes_per_row * even + chroma.bytes_per_row * (even / 2);
    // Each two rows more declare two of the luma plane's and one of the chroma's.
    let per_two = 2 * luma.bytes_per_row + chroma.bytes_per_row;
    even + 2 * reach.saturating_sub(spanned).div_ceil(per_two)
}

/// A render target over memory of the caller's (module docs), shared by the decoder's pictures
/// that are decoded into it.
#[derive(Clone)]
pub struct Target(Arc<Imported>);

struct Imported {
    display: Display,
    id: ffi::VASurfaceID,
    shape: TargetShape,
    /// The memory's descriptor, held for the surface's life.
    _dma_buf: OwnedFd,
    /// What the caller gave to keep while the surface lives.
    _keep: Box<dyn Send + Sync>,
}

impl Drop for Imported {
    fn drop(&mut self) {
        let mut id = self.id;
        // SAFETY: made by vaCreateSurfaces on this display, and no picture names it any more:
        // each holds the target. Destroyed once, here, behind every call on it.
        self.display.release(move |raw| unsafe {
            ffi::vaDestroySurfaces(raw, &raw mut id, 1);
        });
    }
}

impl std::fmt::Debug for Target {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Target")
            .field("id", &self.0.id)
            .field("shape", &self.0.shape)
            .finish_non_exhaustive()
    }
}

impl Target {
    /// `dma_buf`, of `size` bytes, imported on `display` as a linear surface of `shape` whose
    /// planes lie at `luma` and `chroma`, keeping `keep` while the surface lives. `Err` where the
    /// driver refuses the import, or the layout does not fit the dma-buf: a plane whose rows do not
    /// hold the coded picture's samples, whose rows run past the dma-buf, or which overlaps the
    /// other.
    pub fn import(
        display: &Display,
        dma_buf: BorrowedFd<'_>,
        size: u64,
        shape: TargetShape,
        [luma, chroma]: [TargetPlane; 2],
        keep: Box<dyn Send + Sync>,
    ) -> Result<Target, Error> {
        let (width, height) = shape.coded;
        let row = u64::from(width).saturating_mul(shape.format.bytes_per_sample());
        let end = |plane: TargetPlane, rows: u64| {
            (plane.bytes_per_row >= row)
                .then(|| {
                    plane
                        .bytes_per_row
                        .checked_mul(rows.checked_sub(1)?)?
                        .checked_add(row)?
                        .checked_add(plane.offset)
                })
                .flatten()
        };
        let luma_end = end(luma, u64::from(height));
        let chroma_end = end(chroma, u64::from(height.div_ceil(2)));
        let fits = match (luma_end, chroma_end) {
            (Some(luma_end), Some(chroma_end)) => {
                luma_end <= size
                    && chroma_end <= size
                    && (luma_end <= chroma.offset || chroma_end <= luma.offset)
            }
            _ => false,
        };
        let fields = (
            u32::try_from(size),
            u32::try_from(luma.offset),
            u32::try_from(chroma.offset),
            u32::try_from(luma.bytes_per_row),
            u32::try_from(chroma.bytes_per_row),
        );
        let (Ok(object_size), Ok(luma_at), Ok(chroma_at), Ok(luma_pitch), Ok(chroma_pitch)) =
            fields
        else {
            return Err(Error::Unsupported("a target past 4 GiB".into()));
        };
        if !fits || width == 0 || height == 0 {
            return Err(Error::Unsupported(format!(
                "a target of {width}x{height} {:?} whose planes do not fit its {size} bytes: \
                 {luma:?}, {chroma:?}",
                shape.format
            )));
        }
        let rows = declared_rows(height, luma, chroma);
        let declared_end = (luma.offset + luma.bytes_per_row * rows)
            .max(chroma.offset + chroma.bytes_per_row * (rows / 2));
        let (Ok(rows), true) = (u32::try_from(rows), declared_end <= size) else {
            return Err(Error::Unsupported(format!(
                "a target of {width}x{height} {:?} whose planes, declared {rows} rows tall to span \
                 its chroma plane, run past its {size} bytes: {luma:?}, {chroma:?}",
                shape.format
            )));
        };
        let format = shape.format.surface();
        let mut descriptor = ffi::VADRMPRIMESurfaceDescriptor {
            fourcc: format.fourcc(),
            width,
            height: rows,
            num_objects: 1,
            num_layers: 1,
            ..Default::default()
        };
        descriptor.objects[0] = ffi::VADRMPRIMESurfaceObject {
            fd: dma_buf.as_raw_fd(),
            size: object_size,
            drm_format_modifier: ffi::DRM_FORMAT_MOD_LINEAR,
        };
        descriptor.layers[0] = ffi::VADRMPRIMESurfaceLayer {
            drm_format: match format {
                SurfaceFormat::Nv12 => ffi::DRM_FORMAT_NV12,
                SurfaceFormat::P010 => ffi::DRM_FORMAT_P010,
            },
            num_planes: 2,
            object_index: [0, 0, 0, 0],
            offset: [luma_at, chroma_at, 0, 0],
            pitch: [luma_pitch, chroma_pitch, 0, 0],
        };
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
        // The surface's own descriptor, which it holds while it lives.
        let dma_buf = dma_buf
            .try_clone_to_owned()
            .map_err(|e| Error::Unsupported(format!("a target's dma-buf not duplicated: {e}")))?;
        let mut id = ffi::VA_INVALID_SURFACE;
        // SAFETY: one surface out; the attributes and the descriptor they point at live for the
        // call, and the dma-buf for the surface's life (the target holds it).
        let status = display.call(|raw| unsafe {
            ffi::vaCreateSurfaces(
                raw,
                format.rt_format(),
                width,
                rows,
                &raw mut id,
                1,
                attributes.as_mut_ptr(),
                2,
            )
        })?;
        check(
            "vaCreateSurfaces (a target over the caller's memory)",
            status,
        )?;
        Ok(Target(Arc::new(Imported {
            display: display.clone_ref(),
            id,
            shape,
            _dma_buf: dma_buf,
            _keep: keep,
        })))
    }

    /// The shape it was imported as.
    pub fn shape(&self) -> TargetShape {
        self.0.shape
    }

    /// Whether a decoder or a frame still holds it: a picture decoded into it that is a
    /// reference, or one not yet let go of. Such a target is not handed a decode again (module
    /// docs).
    pub fn is_held(&self) -> bool {
        Arc::strong_count(&self.0) > 1
    }

    pub(crate) fn id(&self) -> ffi::VASurfaceID {
        self.0.id
    }

    /// Whether it was imported on `display`.
    pub(crate) fn is_on(&self, display: &Display) -> bool {
        self.0.display.is_same(display)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plane(offset: u64, bytes_per_row: u64) -> TargetPlane {
        TargetPlane {
            offset,
            bytes_per_row,
        }
    }

    #[test]
    fn a_target_is_declared_as_tall_as_its_planes_need_to_span_its_chroma_rows() {
        // The shapes measured on radeonsi (module docs): 240x144 NV12 at 256-byte rows, the luma
        // plane 256 bytes in; the least height that was written whole at each gap.
        let luma = plane(256, 256);
        let after_luma = 256 + 256 * 144;
        assert_eq!(declared_rows(144, luma, plane(after_luma, 256)), 144);
        assert_eq!(declared_rows(144, luma, plane(after_luma + 256, 256)), 146);
        assert_eq!(declared_rows(144, luma, plane(after_luma + 512, 256)), 146);
        assert_eq!(declared_rows(144, luma, plane(after_luma + 4096, 256)), 156);
        // P010 of 240x136 at 512-byte rows, the chroma 256 bytes past the luma's rows.
        let luma = plane(256, 512);
        assert_eq!(
            declared_rows(136, luma, plane(256 + 512 * 136 + 256, 512)),
            138
        );
        // An odd height is made even; a chroma plane before the luma takes the picture's height.
        assert_eq!(declared_rows(9, plane(0, 64), plane(64 * 10, 64)), 10);
        assert_eq!(declared_rows(16, plane(4096, 64), plane(0, 64)), 16);
    }

    #[test]
    fn the_declared_planes_span_the_chroma_rows_and_no_two_rows_fewer_do() {
        for (height, pitch, gap) in [
            (144u32, 256u64, 0u64),
            (144, 256, 256),
            (136, 512, 4096),
            (1088, 1920, 4096),
            (2160, 7680, 4096),
            (1080, 1376, 3104),
        ] {
            let luma = plane(256, pitch);
            let chroma = plane(256 + pitch * u64::from(height) + gap, pitch);
            let rows = declared_rows(height, luma, chroma);
            let reach = chroma.offset + pitch * u64::from(height.div_ceil(2)) - luma.offset;
            let span = |rows: u64| pitch * rows + pitch * (rows / 2);
            assert_eq!(rows % 2, 0);
            assert!(span(rows) >= reach, "{height} {pitch} {gap}");
            assert!(
                rows == u64::from(height) || span(rows - 2) < reach,
                "{height} {pitch} {gap}"
            );
        }
    }
}
