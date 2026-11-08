//! Render targets over memory of the caller's ([`Target`]): a decode's picture written by the
//! hardware straight into that memory, as a decoder's own surface is, where the caller hands the
//! decode a target for it ([`TargetFor`], `H264Decoder::decode_into`,
//! `HevcDecoder::decode_into`).
//!
//! - **The import.** The memory is one dma-buf, both planes in it at their own offsets and
//!   pitches, imported as a linear surface (`VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2`, one object,
//!   `DRM_FORMAT_MOD_LINEAR`), of the picture's format and coded size: the driver writes the coded
//!   picture's rows from each plane's first byte. A driver that refuses the import answers it at
//!   `vaCreateSurfaces`, and the decode then takes one of its own surfaces.
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
        let format = shape.format.surface();
        let mut descriptor = ffi::VADRMPRIMESurfaceDescriptor {
            fourcc: format.fourcc(),
            width,
            height,
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
                height,
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
