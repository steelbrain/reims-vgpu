//! VA images: made with `vaCreateImage` or derived from a surface with
//! `vaDeriveImage`, each destroyed when dropped, and their buffers mapped
//! for as long as a [`Mapping`] lives.
//!
//! `vaGetImage`, and the map of a derived image, may wait for the GPU in
//! the driver, holding the display's lock (`super::calls`): each is made
//! on the display's call thread, until the deadline of the frame in hand.

use super::display::{Display, Error, check};
use super::ffi;

/// A VA image of one display, destroyed when dropped.
pub(crate) struct Image {
    display: Display,
    raw: ffi::VAImage,
    /// Whether `vaDeriveImage` made it: its map is of the surface itself.
    derived: bool,
}

impl Image {
    /// A new image of `format` and `width` by `height`.
    pub(crate) fn create(
        display: &Display,
        mut format: ffi::VAImageFormat,
        width: u32,
        height: u32,
    ) -> Result<Image, Error> {
        let size = |v: u32| {
            i32::try_from(v).map_err(|_| Error::Unsupported(format!("an image {v} pixels across")))
        };
        let (width, height) = (size(width)?, size(height)?);
        let mut raw = ffi::VAImage::default();
        check(
            "vaCreateImage",
            // SAFETY: valid in and out pointers, for the call alone.
            display.call(|va| unsafe {
                ffi::vaCreateImage(va, &mut format, width, height, &mut raw)
            })?,
        )?;
        Ok(Image {
            display: display.clone_ref(),
            raw,
            derived: false,
        })
    }

    /// The image of `surface`'s own memory, where the driver gives one:
    /// writes through its mapping are writes to the surface.
    pub(crate) fn derive(display: &Display, surface: ffi::VASurfaceID) -> Result<Image, Error> {
        let mut raw = ffi::VAImage::default();
        check(
            "vaDeriveImage",
            // SAFETY: a valid out-pointer, for the call alone; libva
            // refuses an ID that is not one of the display's surfaces.
            display.call(|va| unsafe { ffi::vaDeriveImage(va, surface, &mut raw) })?,
        )?;
        Ok(Image {
            display: display.clone_ref(),
            raw,
            derived: true,
        })
    }

    /// What the driver says of the image: its format, size, planes and
    /// buffer.
    pub(crate) fn raw(&self) -> &ffi::VAImage {
        &self.raw
    }

    /// Copies the top-left `width` by `height` pixels of `surface` into the
    /// image (`vaGetImage`), on the display's call thread, until the
    /// deadline of the frame in hand: past it [`Error::Stalled`], the call
    /// left to return there, and the image and the surface are destroyed
    /// behind it.
    pub(crate) fn get_from(
        &self,
        surface: ffi::VASurfaceID,
        width: u32,
        height: u32,
    ) -> Result<(), Error> {
        let image = self.raw.image_id;
        let got = self.display.wait_call(
            "the GPU's copy of a surface into an image (vaGetImage)",
            // SAFETY: the image lives until the call returns, its destroy
            // made behind it, and libva refuses a surface ID that is not
            // the display's, and a region past either.
            move |va| unsafe { ffi::vaGetImage(va, surface, 0, 0, width, height, image) },
            |_, _| {},
        )?;
        check("vaGetImage", got)
    }

    /// Copies the image's top-left `width` by `height` pixels into
    /// `surface`'s (`vaPutImage`).
    pub(crate) fn put_into(
        &self,
        surface: ffi::VASurfaceID,
        width: u32,
        height: u32,
    ) -> Result<(), Error> {
        let image = self.raw.image_id;
        check(
            "vaPutImage",
            // SAFETY: the image is alive while `self` is, and libva refuses
            // a surface ID that is not the display's, and a region past
            // either.
            self.display.call(|va| unsafe {
                ffi::vaPutImage(va, surface, image, 0, 0, width, height, 0, 0, width, height)
            })?,
        )
    }

    /// The image's bytes, mapped until the mapping drops. The mapping
    /// borrows the image mutably, so no other is made of it meanwhile, nor
    /// is it destroyed. A derived image is mapped on the display's call
    /// thread, until the deadline of the frame in hand: past it
    /// [`Error::Stalled`], and the map, once it returns, is undone there.
    pub(crate) fn map(&mut self) -> Result<Mapping<'_>, Error> {
        let data = map_buffer(&self.display, self.raw.buf, self.derived)?;
        let len = usize::try_from(self.raw.data_size).unwrap_or(0);
        let mapping = Mapping {
            image: self,
            data,
            len,
        };
        Ok(mapping)
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        let image = self.raw.image_id;
        // SAFETY: made by vaCreateImage or vaDeriveImage on this display,
        // destroyed once, here, behind every call on it; no mapping of it
        // lives, as each borrows it.
        self.display.release(move |va| unsafe {
            ffi::vaDestroyImage(va, image);
        });
    }
}

/// Maps `buffer`, an image's, of `display`: on the display's call thread
/// where `derived`, as the map of a derived image may wait for the GPU
/// (module docs), its unmap made there once a late map returns; else on
/// this thread. Not null.
pub(crate) fn map_buffer(
    display: &Display,
    buffer: ffi::VABufferID,
    derived: bool,
) -> Result<*mut u8, Error> {
    // SAFETY: the image's buffer, alive until the call returns, its destroy
    // made behind it; the out-pointer is the call's own.
    let map = move |va: ffi::VADisplay| unsafe {
        let mut data: *mut core::ffi::c_void = core::ptr::null_mut();
        let status = ffi::vaMapBuffer(va, buffer, &mut data);
        (status, data.expose_provenance())
    };
    let (status, data) = if derived {
        display.wait_call(
            "the GPU's copy of a surface for its derived image's map (vaMapBuffer)",
            map,
            move |va, (status, _)| {
                if status == ffi::VA_STATUS_SUCCESS {
                    // SAFETY: mapped by the late call, unmapped once, here.
                    unsafe { ffi::vaUnmapBuffer(va, buffer) };
                }
            },
        )?
    } else {
        display.call(map)?
    };
    check("vaMapBuffer", status)?;
    if data == 0 {
        unmap_buffer(display, buffer);
        return Err(Error::Unsupported("an image mapped at no address".into()));
    }
    Ok(std::ptr::with_exposed_provenance_mut(data))
}

/// Unmaps `buffer`, mapped by [`map_buffer`]: now, or behind the display's
/// calls in hand.
pub(crate) fn unmap_buffer(display: &Display, buffer: ffi::VABufferID) {
    // SAFETY: mapped by `map_buffer`, unmapped once, here.
    display.release(move |va| unsafe {
        ffi::vaUnmapBuffer(va, buffer);
    });
}

/// An image's buffer, mapped: unmapped when dropped.
pub(crate) struct Mapping<'i> {
    image: &'i Image,
    /// Not null: [`Image::map`] refuses a null address.
    data: *mut u8,
    len: usize,
}

impl Mapping<'_> {
    /// The mapped bytes, for writing.
    pub(crate) fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: a mapped image buffer is `data_size` bytes, readable and
        // writable, until vaUnmapBuffer (va.h, VAImage), which runs only
        // as this drops; `data` is not null, and the image is borrowed
        // mutably for the mapping's life, so nothing else maps it; and the
        // bytes are borrowed mutably through `self`.
        unsafe { core::slice::from_raw_parts_mut(self.data, self.len) }
    }
}

impl Drop for Mapping<'_> {
    fn drop(&mut self) {
        unmap_buffer(&self.image.display, self.image.raw.buf);
    }
}
