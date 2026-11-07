//! VA buffers: a decoder's parameters and data for one picture, copied
//! into libva and destroyed when dropped, and the renders of a picture.
//!
//! **A decoder's first picture waits.** radeonsi begins a decode
//! context's session at the context's first `vaEndPicture`: it writes a
//! session-create command's message into the first of the context's four
//! embedded buffers and submits it, and then maps the same buffer, for
//! write, to write the picture's own message, which waits on the CPU, with
//! the display's lock held, until the command is done (**read** in Mesa 26.2.4,
//! `src/gallium/drivers/radeonsi/mm/si_video_dec.c`, `si_dec_init_decoder`
//! and `decode_cmd_build`; docs/video/bugs.md, U24). After an unmap of the
//! process's GPU VM, a decoder let go among them, the command waits for
//! every job the VM had in flight at the unmap (U23), so the first
//! `vaEndPicture` waits as long. **Measured** on the RX 7900 XTX and the
//! Radeon 780M (Linux 7.2.9, Mesa 26.2.4, 2026-10-26), a new H.264 decoder's
//! first decode just after another decoder was let go, beside an 8192x8192
//! JPEG decode on the GPU: the first `vaEndPicture` took 43.5 and 127.6 ms,
//! in `drmSyncobjWait`, and returned once the command, which the scheduler
//! had held behind the unmap's page-table update and that update behind
//! the JPEG decode, was done, 0.04 ms after it on the RX 7900 XTX
//! (docs/video/bugs.md, B92). So a decoder's first `vaEndPicture` is a
//! call that may wait for the GPU, made on the display's call thread by the
//! frame's deadline ([`render_decode`], `super::calls`), and the decoder's
//! later ones on the caller's thread, as every other submission is.

use super::caps::Context;
use super::display::{Display, Error, check};
use super::ffi;

/// A VA buffer, destroyed on drop.
pub(crate) struct Buffer<'a> {
    display: &'a Display,
    pub(crate) id: ffi::VABufferID,
}

impl<'a> Buffer<'a> {
    /// A buffer holding one `T`, which libva copies.
    pub(crate) fn with_value<T: Copy>(
        display: &'a Display,
        context: ffi::VAContextID,
        type_: ffi::VABufferType,
        value: &T,
    ) -> Result<Self, Error> {
        Self::with_values(display, context, type_, core::slice::from_ref(value))
    }

    /// A buffer holding `values`, one element each, which libva copies:
    /// the form of a parameter buffer whose elements each describe a part
    /// of the data buffer that follows it, as AV1's tiles are.
    pub(crate) fn with_values<T: Copy>(
        display: &'a Display,
        context: ffi::VAContextID,
        type_: ffi::VABufferType,
        values: &[T],
    ) -> Result<Self, Error> {
        let size = u32::try_from(size_of::<T>()).unwrap_or(0);
        let count = u32::try_from(values.len())
            .map_err(|_| Error::Unsupported("over 4 Gi buffer elements".into()))?;
        Self::create(display, context, type_, size, count, values.as_ptr().cast())
    }

    /// A buffer holding `bytes`, which libva copies.
    pub(crate) fn with_bytes(
        display: &'a Display,
        context: ffi::VAContextID,
        type_: ffi::VABufferType,
        bytes: &[u8],
    ) -> Result<Self, Error> {
        let size = u32::try_from(bytes.len())
            .map_err(|_| Error::Unsupported("a slice over 4 GiB".into()))?;
        Self::create(display, context, type_, size, 1, bytes.as_ptr().cast())
    }

    fn create(
        display: &'a Display,
        context: ffi::VAContextID,
        type_: ffi::VABufferType,
        size: u32,
        count: u32,
        data: *const core::ffi::c_void,
    ) -> Result<Self, Error> {
        let mut id = ffi::VA_INVALID_ID;
        check(
            "vaCreateBuffer",
            // SAFETY: `data` points at `count` elements of `size` readable
            // bytes. libva copies them and does not write through the pointer
            // (va.h, vaCreateBuffer: "results the data being copied to the data
            // store").
            display.call(|raw| unsafe {
                ffi::vaCreateBuffer(raw, context, type_, size, count, data.cast_mut(), &mut id)
            })?,
        )?;
        Ok(Self { display, id })
    }
}

impl Drop for Buffer<'_> {
    fn drop(&mut self) {
        let id = self.id;
        // SAFETY: made by vaCreateBuffer; "the user must call
        // vaDestroyBuffer()" (va.h). Destroyed once, here.
        self.display.release(move |raw| unsafe {
            ffi::vaDestroyBuffer(raw, id);
        });
    }
}

/// Renders one picture into `target`, for every kind of context: decode,
/// encode and video processing. `vaBeginPicture`, `vaRenderPicture` for
/// each group of buffers in turn, then `vaEndPicture`.
///
/// A picture begun is ended whatever its renders answered, so that the
/// context takes its next picture as a new one rather than as more of this
/// one; the answer is the first failure, and the groups after a failed
/// render are not rendered.
pub(crate) fn render_picture(
    display: &Display,
    context: ffi::VAContextID,
    target: ffi::VASurfaceID,
    groups: &[&[&Buffer<'_>]],
) -> Result<(), Error> {
    render(display, context, target, groups, None)
}

/// What a decoder's first `vaEndPicture` is, where it holds the display past
/// the frame's deadline ([`Error::Stalled`]).
pub(crate) const FIRST_PICTURE: &str = "a new decoder's first picture (vaEndPicture)";

/// [`render_picture`] for a decoder's picture on `context`: the context's
/// first `vaEndPicture` is made on the display's call thread by the frame's
/// deadline, as a call that may wait for the GPU (module docs), and the
/// later ones on this thread. Past the deadline the picture fails as
/// [`Error::Stalled`], the call left to return there.
pub(crate) fn render_decode(
    display: &Display,
    context: &Context,
    target: ffi::VASurfaceID,
    groups: &[&[&Buffer<'_>]],
) -> Result<(), Error> {
    render(display, context.id, target, groups, Some(context))
}

/// [`render_picture`], where `decoder` is a decoder's context, its first
/// `vaEndPicture` on the display's call thread.
fn render(
    display: &Display,
    context: ffi::VAContextID,
    target: ffi::VASurfaceID,
    groups: &[&[&Buffer<'_>]],
    decoder: Option<&Context>,
) -> Result<(), Error> {
    check(
        "vaBeginPicture",
        // SAFETY: the context and the surface are this display's, alive for the
        // call: their owners outlive it.
        display.call(|raw| unsafe { ffi::vaBeginPicture(raw, context, target) })?,
    )?;
    let rendered = groups.iter().try_for_each(|group| {
        let mut ids: Vec<ffi::VABufferID> = group.iter().map(|b| b.id).collect();
        let count = i32::try_from(ids.len())
            .map_err(|_| Error::Unsupported("over 2 Gi buffers in a render".into()))?;
        check(
            "vaRenderPicture",
            // SAFETY: `ids` holds `count` IDs of this display's live buffers,
            // borrowed until this function returns, so past `vaEndPicture`.
            // Any pointer a buffer holds (a pipeline's rectangles) points at
            // the caller's values, which outlive the call.
            display.call(|raw| unsafe {
                ffi::vaRenderPicture(raw, context, ids.as_mut_ptr(), count)
            })?,
        )
    });
    let ended = if decoder.is_some_and(Context::first_end) {
        display.wait_call(
            FIRST_PICTURE,
            // SAFETY: a picture begun above on this context, ended once.
            // The context and the buffers outlive the call wherever it
            // returns: their destroys are made behind it.
            move |raw| unsafe { ffi::vaEndPicture(raw, context) },
            |_, _| {},
        )
    } else {
        // SAFETY: a picture begun above on this context, ended once.
        display.call(|raw| unsafe { ffi::vaEndPicture(raw, context) })
    }
    .and_then(|status| check("vaEndPicture", status));
    rendered.and(ended)
}
