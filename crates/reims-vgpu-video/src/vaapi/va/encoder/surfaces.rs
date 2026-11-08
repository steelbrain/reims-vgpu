//! The encoder's surfaces: a fixed pool of NV12 or P010 surfaces of the
//! coded size, lent one at a time, and the upload of a frame from host
//! memory into one, with the coded size's margin filled by repeating the
//! last column and row.

use std::sync::Arc;

use crate::core::sync::Mutex;

use super::super::display::{Display, Error, check};
use super::super::ffi::{self, encode as enc};
use super::super::image::Image;
use crate::codec::encode::{InputFormat, InputFrame};

/// The VA formats of an input format: the render-target format and the
/// surface FOURCC.
pub(crate) fn va_format(format: InputFormat) -> (u32, u32) {
    match format {
        InputFormat::Nv12 => (ffi::VA_RT_FORMAT_YUV420, ffi::VA_FOURCC_NV12),
        InputFormat::P010 => (enc::VA_RT_FORMAT_YUV420_10, enc::VA_FOURCC_P010),
    }
}

/// Surfaces of one size and format, made at once.
pub(crate) struct Pool {
    display: Display,
    ids: Vec<ffi::VASurfaceID>,
    free: Arc<Mutex<Vec<ffi::VASurfaceID>>>,
}

impl Pool {
    pub(crate) fn new(
        display: &Display,
        format: InputFormat,
        width: u32,
        height: u32,
        count: usize,
    ) -> Result<Self, Error> {
        let (rt_format, fourcc) = va_format(format);
        let mut ids = vec![ffi::VA_INVALID_SURFACE; count];
        let mut attrib = ffi::VASurfaceAttrib::zeroed();
        attrib.type_ = ffi::VASurfaceAttribPixelFormat;
        attrib.flags = ffi::VA_SURFACE_ATTRIB_SETTABLE;
        attrib.value.type_ = ffi::VAGenericValueTypeInteger;
        attrib.value.value.i = i32::from_ne_bytes(fourcc.to_ne_bytes());
        let count = u32::try_from(count).unwrap_or(0);
        check(
            "vaCreateSurfaces",
            // SAFETY: `ids` holds `count` entries, and the attribute is
            // valid.
            display.call(|raw| unsafe {
                ffi::vaCreateSurfaces(
                    raw,
                    rt_format,
                    width,
                    height,
                    ids.as_mut_ptr(),
                    count,
                    &mut attrib,
                    1,
                )
            })?,
        )?;
        let free = ids.iter().rev().copied().collect();
        Ok(Self {
            display: display.clone_ref(),
            ids,
            free: Arc::new(Mutex::new(free)),
        })
    }

    /// Every surface, for a context's render targets.
    pub(crate) fn ids(&self) -> &[ffi::VASurfaceID] {
        &self.ids
    }

    /// A free surface.
    pub(crate) fn get(&self) -> Result<Lent, Error> {
        let id = self.free.lock().pop().ok_or(Error::OutOfSurfaces)?;
        Ok(Lent {
            id,
            free: self.free.clone(),
        })
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        let mut ids = std::mem::take(&mut self.ids);
        let count = i32::try_from(ids.len()).unwrap_or(0);
        // SAFETY: made by vaCreateSurfaces on this display, destroyed once,
        // here. The session destroys its context first; a Lent outliving
        // the pool only pushes its ID onto the free list.
        self.display.release(move |raw| unsafe {
            ffi::vaDestroySurfaces(raw, ids.as_mut_ptr(), count);
        });
    }
}

/// A surface on loan, back in its pool when dropped.
pub(crate) struct Lent {
    id: ffi::VASurfaceID,
    free: Arc<Mutex<Vec<ffi::VASurfaceID>>>,
}

impl Lent {
    pub(crate) fn id(&self) -> ffi::VASurfaceID {
        self.id
    }
}

impl Drop for Lent {
    fn drop(&mut self) {
        self.free.lock().push(self.id);
    }
}

/// How frames reach their surfaces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UploadMethod {
    /// Mapping the surface itself (`vaDeriveImage`).
    Derive,
    /// Writing an image and copying it in (`vaPutImage`).
    PutImage,
}

/// Uploads frames of one format and size into surfaces of the coded size.
pub(crate) struct Uploader {
    display: Display,
    format: InputFormat,
    width: u32,
    height: u32,
    coded_width: u32,
    coded_height: u32,
    method: Option<UploadMethod>,
    /// The image `vaPutImage` copies from, made on first use.
    image: Option<Image>,
}

impl Uploader {
    pub(crate) fn new(
        display: &Display,
        format: InputFormat,
        size: (u32, u32),
        coded_size: (u32, u32),
    ) -> Self {
        Self {
            display: display.clone_ref(),
            format,
            width: size.0,
            height: size.1,
            coded_width: coded_size.0,
            coded_height: coded_size.1,
            method: None,
            image: None,
        }
    }

    pub(crate) fn method(&self) -> Option<UploadMethod> {
        self.method
    }

    /// Uses `method` from now on.
    pub(crate) fn force(&mut self, method: UploadMethod) {
        self.method = Some(method);
    }

    /// Checks that the frame's planes hold the picture.
    fn check_planes(&self, frame: &InputFrame<'_>) -> Result<(), Error> {
        let row = usize::try_from(self.width)
            .ok()
            .and_then(|width| width.checked_mul(self.format.bytes_per_sample()));
        let rows = [self.height, self.height / 2];
        for ((plane, rows), name) in frame.planes.iter().zip(rows).zip(["luma", "chroma"]) {
            let rows = usize::try_from(rows).unwrap_or(usize::MAX);
            let holds = row.is_some_and(|row| {
                plane.bytes_per_row >= row
                    && plane
                        .bytes_per_row
                        .checked_mul(rows.saturating_sub(1))
                        .and_then(|start| start.checked_add(row))
                        .is_some_and(|end| end <= plane.data.len())
            });
            if !holds {
                return Err(Error::PlaneTooSmall(name));
            }
        }
        Ok(())
    }

    /// Copies `frame` into `surface`. The first upload finds the method:
    /// `vaDeriveImage` where the driver maps the surface, else
    /// `vaPutImage`.
    pub(crate) fn upload(&mut self, frame: &InputFrame<'_>, surface: &Lent) -> Result<(), Error> {
        self.check_planes(frame)?;
        match self.method {
            Some(UploadMethod::Derive) => self.upload_derived(frame, surface),
            Some(UploadMethod::PutImage) => self.upload_put(frame, surface),
            None => match self.upload_derived(frame, surface) {
                Ok(()) => {
                    self.method = Some(UploadMethod::Derive);
                    Ok(())
                }
                // The GPU did not finish what the map waited for: no
                // method is found by it.
                Err(error @ Error::Stalled(_)) => Err(error),
                Err(_) => {
                    self.upload_put(frame, surface)?;
                    self.method = Some(UploadMethod::PutImage);
                    Ok(())
                }
            },
        }
    }

    fn upload_derived(&self, frame: &InputFrame<'_>, surface: &Lent) -> Result<(), Error> {
        // The surface is not being encoded: it is free until this upload.
        let mut image = Image::derive(&self.display, surface.id())?;
        self.fill(&mut image, frame)
    }

    fn upload_put(&mut self, frame: &InputFrame<'_>, surface: &Lent) -> Result<(), Error> {
        let mut image = match self.image.take() {
            Some(image) => image,
            None => {
                let (_, fourcc) = va_format(self.format);
                let format = ffi::VAImageFormat {
                    fourcc,
                    bits_per_pixel: if self.format == InputFormat::P010 {
                        24
                    } else {
                        12
                    },
                    ..ffi::VAImageFormat::default()
                };
                Image::create(&self.display, format, self.coded_width, self.coded_height)?
            }
        };
        // Both rectangles are the whole coded size, which the image and
        // the surface have.
        let result = self
            .fill(&mut image, frame)
            .and_then(|()| image.put_into(surface.id(), self.coded_width, self.coded_height));
        self.image = Some(image);
        result
    }

    /// Maps `image`, copies the frame into it with the margin filled, and
    /// unmaps it.
    fn fill(&self, image: &mut Image, frame: &InputFrame<'_>) -> Result<(), Error> {
        let (_, fourcc) = va_format(self.format);
        let raw = *image.raw();
        if raw.format.fourcc != fourcc || raw.num_planes < 2 {
            return Err(Error::Unsupported(format!(
                "uploading into an image of {}",
                super::super::caps::Fourcc(raw.format.fourcc)
            )));
        }
        if u32::from(raw.width) < self.coded_width || u32::from(raw.height) < self.coded_height {
            return Err(Error::Unsupported(
                "an image smaller than its surface".into(),
            ));
        }
        let mut mapping = image.map()?;
        self.copy_planes(&raw, mapping.bytes_mut(), frame)
    }

    fn copy_planes(
        &self,
        image: &ffi::VAImage,
        bytes: &mut [u8],
        frame: &InputFrame<'_>,
    ) -> Result<(), Error> {
        let sample = self.format.bytes_per_sample();
        let to = |v: u32| usize::try_from(v).unwrap_or(0);
        let planes = [
            (to(self.height), to(self.coded_height), sample),
            (to(self.height / 2), to(self.coded_height / 2), sample * 2),
        ];
        for (index, (rows, coded_rows, unit)) in planes.into_iter().enumerate() {
            let source = frame.planes[index];
            let pitch = to(image.pitches[index]);
            let offset = to(image.offsets[index]);
            let row_bytes = to(self.width) * sample;
            let coded_bytes = to(self.coded_width) * sample;
            let small = || Error::Unsupported("an image smaller than its pitches".into());
            for row in 0..coded_rows {
                let from_row = row.min(rows.saturating_sub(1));
                let src = from_row
                    .checked_mul(source.bytes_per_row)
                    .and_then(|start| source.data.get(start..start.checked_add(row_bytes)?))
                    .ok_or(Error::PlaneTooSmall("input"))?;
                let dst = row
                    .checked_mul(pitch)
                    .and_then(|start| start.checked_add(offset))
                    .and_then(|at| bytes.get_mut(at..at.checked_add(coded_bytes)?))
                    .ok_or_else(small)?;
                dst[..row_bytes].copy_from_slice(src);
                // The margin repeats the last sample (or Cb/Cr pair).
                if coded_bytes > row_bytes && row_bytes >= unit {
                    let (filled, margin) = dst.split_at_mut(row_bytes);
                    let last = &filled[row_bytes - unit..];
                    for chunk in margin.chunks_mut(unit) {
                        chunk.copy_from_slice(&last[..chunk.len()]);
                    }
                }
            }
        }
        Ok(())
    }
}
