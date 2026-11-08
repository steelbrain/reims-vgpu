//! The encoder core's accelerator on VA-API: a config and context on the
//! encode entrypoint, a pool of input surfaces and one of reconstructed
//! pictures, a coded buffer, and for each picture the buffers of
//! [`super::params`] and the packed headers, rendered in the order libva's
//! users render them: sequence, misc, picture, packed parameter sets,
//! packed slice header, slice.
//!
//! Each picture's driver calls and its coded buffer are waited for until
//! one deadline (`super::super::wait`). Past it the encoder is stalled:
//! that picture fails, each later call fails at once, and the session,
//! with its coded buffer, context and surfaces, is kept until the GPU is
//! done with the coded buffer and the display's call thread with what it
//! has (an upload's map of a derived image is made there).
//!
//! A dropped encoder's session that coded a picture is kept open, in
//! [`KEPT`], until a later session on the same render node is about to
//! submit a picture after its first: then it is destroyed, its close
//! followed on the VCN by that picture. radeonsi's VCN hangs otherwise
//! ([`super`]'s docs).

use crate::core::runtime::MonoTime;
use crate::core::sync::Mutex;

use super::super::buffer::{Buffer, render_picture};
use super::super::caps::{Config, Context, Entrypoint, Profile};
use super::super::display::{Display, Error, check};
use super::super::ffi::{self, encode as enc};
use super::super::wait::{self, Work};
use super::params;
use super::surfaces::{Lent, Pool, UploadMethod, Uploader};
use crate::codec::encode::encoder::{
    Accelerator, Headers, PackedKind, PackedUnit, PictureParams, StreamParams,
};
use crate::codec::encode::{EncoderSettings, FrameType, InputFrame, RateControl};

/// Surfaces beyond the references and B pictures in flight, so that one
/// is always free.
const SPARE_SURFACES: usize = 1;

/// The sessions of dropped encoders kept open, each with its display's
/// render node (module docs).
static KEPT: Mutex<Vec<(String, Session)>> = Mutex::new(Vec::new());

/// Destroys the kept sessions on render node `node`.
fn retire(node: &str) {
    let retired: Vec<(String, Session)> = {
        let mut kept = KEPT.lock();
        let (retired, left) = std::mem::take(&mut *kept)
            .into_iter()
            .partition(|(n, _)| n == node);
        *kept = left;
        retired
    };
    drop(retired);
}

/// What one stream's pictures are encoded with.
struct Session {
    display: Display,
    /// Destroyed after the coded buffer made on it, and before the
    /// surfaces it renders into; it holds its config.
    context: Option<Context>,
    coded_buffer: ffi::VABufferID,
    coded_buffer_size: usize,
    inputs: Pool,
    reconstructed: Pool,
    /// The references held, by number.
    references: Vec<(u64, Lent)>,
    uploader: Uploader,
    /// From [`StreamParams`].
    b_frames: u32,
    intra_period: u32,
    picture_qp: bool,
    /// The reconstruction of the picture last encoded and not yet kept,
    /// which a try of the same picture codes into again.
    attempt: Option<Lent>,
    /// Whether a picture was coded: the driver began the session on the
    /// device.
    coded: bool,
}

impl Session {
    fn context_id(&self) -> ffi::VAContextID {
        self.context.as_ref().map_or(ffi::VA_INVALID_ID, |c| c.id)
    }
}

/// VA-API's half of the encoder.
pub(crate) struct VaAccelerator {
    display: Display,
    profile: Profile,
    entrypoint: Entrypoint,
    /// The `VA_ENC_PACKED_HEADER_*` bits the config asks for.
    packed_headers: u32,
    session: Option<Session>,
    forced_upload: Option<UploadMethod>,
    /// The picture the GPU did not finish by its deadline, once it has not
    /// (module docs).
    stalled: Option<String>,
}

impl VaAccelerator {
    pub(crate) fn new(
        display: &Display,
        profile: Profile,
        entrypoint: Entrypoint,
        packed_headers: u32,
    ) -> Self {
        Self {
            display: display.clone_ref(),
            profile,
            entrypoint,
            packed_headers,
            session: None,
            forced_upload: None,
            stalled: None,
        }
    }

    /// `Err` where the encoder is stalled (module docs).
    fn unstalled(&self) -> Result<(), Error> {
        match &self.stalled {
            Some(what) => Err(Error::Stalled(format!("{what}, of an earlier picture"))),
            None => Ok(()),
        }
    }

    /// Stalls the encoder, the GPU not having finished `what` by its
    /// deadline, and keeps the session, with the picture's reconstruction,
    /// until the GPU is done with the picture's coded buffer and the
    /// display's call thread with what it has.
    fn stall(&mut self, what: &str, reconstructed: Option<Lent>) {
        self.stalled = Some(what.to_owned());
        if let Some(session) = self.session.take() {
            let works = vec![Work::Buffer(session.coded_buffer), Work::Calls];
            wait::keep(
                &self.display,
                "a stalled encoder's session",
                works,
                (reconstructed, session),
            );
        }
    }

    pub(crate) fn profile(&self) -> Profile {
        self.profile
    }

    pub(crate) fn entrypoint(&self) -> Entrypoint {
        self.entrypoint
    }

    pub(crate) fn packed_headers(&self) -> u32 {
        self.packed_headers
    }

    pub(crate) fn upload_method(&self) -> Option<UploadMethod> {
        self.session.as_ref().and_then(|s| s.uploader.method())
    }

    pub(crate) fn force_upload_method(&mut self, method: UploadMethod) {
        self.forced_upload = Some(method);
        if let Some(session) = &mut self.session {
            session.uploader.force(method);
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        // Inner objects first, as they nest: the coded buffer, then the
        // context it was made on, then the surfaces, each destroy made in
        // order.
        let buffer = self.coded_buffer;
        // SAFETY: made by vaCreateBuffer on this display, unmapped (each
        // map of it ends before its read returns), and destroyed once,
        // here, while its context lives.
        self.display.release(move |raw| unsafe {
            ffi::vaDestroyBuffer(raw, buffer);
        });
        drop(self.context.take());
        self.references.clear();
    }
}

fn rate_control_mode(rate_control: RateControl) -> u32 {
    match rate_control {
        RateControl::ConstantQp { .. } => enc::VA_RC_CQP,
        RateControl::ConstantBitRate { .. } => enc::VA_RC_CBR,
        RateControl::VariableBitRate { .. } => enc::VA_RC_VBR,
        RateControl::QualityVariableBitRate { .. } => enc::VA_RC_QVBR,
    }
}

impl Drop for VaAccelerator {
    /// Keeps the session open where it coded a picture (module docs).
    fn drop(&mut self) {
        if let Some(session) = self.session.take().filter(|s| s.coded) {
            KEPT.lock().push((self.display.path().to_owned(), session));
        }
    }
}

impl Accelerator for VaAccelerator {
    type Input = Lent;
    type Error = Error;

    fn configure(&mut self, stream: &StreamParams<'_>) -> Result<(), Error> {
        self.unstalled()?;
        let settings = stream.settings;
        let format = settings.input_format();
        let (rt_format, _) = super::surfaces::va_format(format);
        let mut attribs = [
            ffi::VAConfigAttrib {
                type_: ffi::VAConfigAttribRTFormat,
                value: rt_format,
            },
            ffi::VAConfigAttrib {
                type_: ffi::VAConfigAttribRateControl,
                value: if stream.picture_qp {
                    enc::VA_RC_CQP
                } else {
                    rate_control_mode(settings.rate_control)
                },
            },
            ffi::VAConfigAttrib {
                type_: ffi::VAConfigAttribEncPackedHeaders,
                value: self.packed_headers,
            },
        ];
        let config = Config::new(&self.display, self.profile, self.entrypoint, &mut attribs)?;
        let (width, height) = (stream.coded_width, stream.coded_height);
        let b_frames = usize::try_from(stream.b_frames).unwrap_or(0);
        let references = usize::try_from(stream.max_references).unwrap_or(1);
        // The frame being encoded and the B pictures waiting.
        let inputs = Pool::new(
            &self.display,
            format,
            width,
            height,
            b_frames + 1 + SPARE_SURFACES,
        )?;
        // The references and the picture being encoded.
        let reconstructed = Pool::new(
            &self.display,
            format,
            width,
            height,
            references + 1 + SPARE_SURFACES,
        )?;
        let mut targets: Vec<ffi::VASurfaceID> = inputs
            .ids()
            .iter()
            .chain(reconstructed.ids())
            .copied()
            .collect();
        let context = Context::new(&self.display, config, width, height, &mut targets)?;
        // Room for the raw picture and a megabyte: a picture larger than
        // its samples is an overflow the coded buffer reports.
        let raw = usize::try_from(u64::from(width) * u64::from(height) * 3 / 2).unwrap_or(0)
            * format.bytes_per_sample();
        let coded_buffer_size = raw + (1 << 20);
        let coded_buffer = create_coded_buffer(&self.display, context.id, coded_buffer_size)?;
        let mut uploader = Uploader::new(
            &self.display,
            format,
            (settings.width, settings.height),
            (width, height),
        );
        if let Some(method) = self.forced_upload {
            uploader.force(method);
        }
        self.session = Some(Session {
            display: self.display.clone_ref(),
            context: Some(context),
            coded_buffer,
            coded_buffer_size,
            inputs,
            reconstructed,
            references: Vec::new(),
            uploader,
            b_frames: stream.b_frames,
            intra_period: stream.intra_period,
            picture_qp: stream.picture_qp,
            attempt: None,
            coded: false,
        });
        Ok(())
    }

    fn upload(&mut self, frame: &InputFrame<'_>) -> Result<Lent, Error> {
        self.unstalled()?;
        let session = self.session.as_mut().ok_or_else(|| {
            Error::Unsupported("uploading before the stream is configured".into())
        })?;
        let surface = session.inputs.get()?;
        match session.uploader.upload(frame, &surface) {
            Ok(()) => Ok(surface),
            Err(Error::Stalled(what)) => {
                drop(surface);
                self.stall(&what, None);
                Err(Error::Stalled(what))
            }
            Err(error) => Err(error),
        }
    }

    fn encode(&mut self, input: &Lent, p: &PictureParams<'_>) -> Result<Vec<u8>, Error> {
        self.unstalled()?;
        let session = self
            .session
            .as_mut()
            .ok_or_else(|| Error::Unsupported("encoding before the stream is configured".into()))?;
        if session.coded {
            retire(self.display.path());
        }
        // A picture coded again codes into its last try's surface, on the
        // same context: Mesa keeps its DPB by surface.
        let reconstructed = match session.attempt.take() {
            Some(surface) => surface,
            None => session.reconstructed.get()?,
        };
        let coded = code_picture(
            &self.display,
            session,
            self.packed_headers,
            input,
            p,
            &reconstructed,
        );
        match coded {
            Ok(coded) => {
                if let Some(session) = self.session.as_mut() {
                    session.attempt = Some(reconstructed);
                    session.coded = true;
                }
                Ok(coded)
            }
            Err(Error::Stalled(what)) => {
                self.stall(&what, Some(reconstructed));
                Err(Error::Stalled(what))
            }
            Err(error) => Err(error),
        }
    }

    fn keep(&mut self, p: &PictureParams<'_>) {
        let Some(session) = self.session.as_mut() else {
            return;
        };
        let reconstructed = session.attempt.take();
        if let (Some(number), Some(reconstructed)) = (p.reference, reconstructed) {
            session.references.push((number, reconstructed));
        }
        session
            .references
            .retain(|(number, _)| p.retained.contains(number));
    }
}

/// Codes picture `p` of `input` into `reconstructed`, on `session`: its
/// parameter buffers, its submission, and its coded buffer's bytes, waited
/// for by the frame's deadline (`super::super::wait`).
fn code_picture(
    display: &Display,
    session: &Session,
    packed_headers: u32,
    input: &Lent,
    p: &PictureParams<'_>,
    reconstructed: &Lent,
) -> Result<Vec<u8>, Error> {
    let context = session.context_id();
    let surface_of = |number: u64| {
        session
            .references
            .iter()
            .find(|(n, _)| *n == number)
            .map(|(_, s)| s.id())
    };
    let stream = StreamParams {
        settings: p.settings,
        coded_width: 0,
        coded_height: 0,
        b_frames: session.b_frames,
        max_references: 0,
        intra_period: session.intra_period,
        picture_qp: session.picture_qp,
    };
    let mut buffers: Vec<Buffer<'_>> = Vec::with_capacity(16);
    let idr = p.frame_type == FrameType::Idr;
    match p.headers {
        Headers::H264 { sps, pps } => {
            if idr {
                let seq = params::h264_sequence(&stream, p.settings, sps);
                buffers.push(Buffer::with_value(
                    display,
                    context,
                    enc::VAEncSequenceParameterBufferType,
                    &seq,
                )?);
            }
            push_misc(
                &mut buffers,
                display,
                context,
                p.settings,
                idr,
                p.rate_control_changed,
                session.picture_qp,
            )?;
            let pic = params::h264_picture(
                p,
                pps,
                reconstructed.id(),
                session.coded_buffer,
                &surface_of,
            );
            buffers.push(Buffer::with_value(
                display,
                context,
                enc::VAEncPictureParameterBufferType,
                &pic,
            )?);
            push_packed(
                &mut buffers,
                display,
                context,
                packed_headers,
                p.parameter_sets,
                p.slice_header,
            )?;
            let slice = p.h264_slice.ok_or_else(|| {
                Error::Unsupported("an H.264 picture with no slice header".into())
            })?;
            let slice = params::h264_slice(p, sps, slice, &surface_of);
            buffers.push(Buffer::with_value(
                display,
                context,
                enc::VAEncSliceParameterBufferType,
                &slice,
            )?);
        }
        Headers::Hevc { sps, pps } => {
            let slice = p
                .hevc_slice
                .ok_or_else(|| Error::Unsupported("an HEVC picture with no slice header".into()))?;
            if idr {
                let seq = params::hevc_sequence(&stream, p.settings, sps);
                buffers.push(Buffer::with_value(
                    display,
                    context,
                    enc::VAEncSequenceParameterBufferType,
                    &seq,
                )?);
            }
            push_misc(
                &mut buffers,
                display,
                context,
                p.settings,
                idr,
                p.rate_control_changed,
                session.picture_qp,
            )?;
            let pic = params::hevc_picture(
                p,
                pps,
                slice,
                reconstructed.id(),
                session.coded_buffer,
                &surface_of,
            );
            buffers.push(Buffer::with_value(
                display,
                context,
                enc::VAEncPictureParameterBufferType,
                &pic,
            )?);
            push_packed(
                &mut buffers,
                display,
                context,
                packed_headers,
                p.parameter_sets,
                p.slice_header,
            )?;
            let slice = params::hevc_slice(p, sps, slice, &surface_of);
            buffers.push(Buffer::with_value(
                display,
                context,
                enc::VAEncSliceParameterBufferType,
                &slice,
            )?);
        }
    }
    let all: Vec<&Buffer<'_>> = buffers.iter().collect();
    render_picture(display, context, input.id(), &[&all])?;
    let deadline = wait::current(display);
    let coded = read_coded(
        display,
        session.coded_buffer,
        session.coded_buffer_size,
        deadline,
    );
    drop(buffers);
    coded
}

/// The misc parameters: rate control, HRD and frame rate on IDR pictures
/// and when they change, as libva's users send them, and the quality level
/// with them. Where the encoder picks each picture's QP, the driver's
/// constant QP takes no rate control or HRD.
fn push_misc<'a>(
    buffers: &mut Vec<Buffer<'a>>,
    display: &'a Display,
    context: ffi::VAContextID,
    settings: &EncoderSettings,
    idr: bool,
    changed: bool,
    picture_qp: bool,
) -> Result<(), Error> {
    if !idr && !changed {
        return Ok(());
    }
    let ty = enc::VAEncMiscParameterBufferType;
    if !picture_qp {
        if let Some(rc) = params::rate_control(settings, changed && !idr) {
            buffers.push(Buffer::with_value(display, context, ty, &rc)?);
        }
        if let Some(hrd) = params::hrd(settings) {
            buffers.push(Buffer::with_value(display, context, ty, &hrd)?);
        }
    }
    buffers.push(Buffer::with_value(
        display,
        context,
        ty,
        &params::frame_rate(settings),
    )?);
    if let Some(quality) = params::quality_level(settings) {
        buffers.push(Buffer::with_value(display, context, ty, &quality)?);
    }
    Ok(())
}

/// The packed parameter sets and slice header, each a parameter buffer
/// and a data buffer, for the kinds the config asked for.
fn push_packed<'a>(
    buffers: &mut Vec<Buffer<'a>>,
    display: &'a Display,
    context: ffi::VAContextID,
    packed_headers: u32,
    parameter_sets: &[PackedUnit],
    slice_header: &PackedUnit,
) -> Result<(), Error> {
    for unit in parameter_sets.iter().chain(core::iter::once(slice_header)) {
        let (bit, kind) = match unit.kind {
            PackedKind::Sequence => (
                enc::VA_ENC_PACKED_HEADER_SEQUENCE,
                enc::VAEncPackedHeaderSequence,
            ),
            PackedKind::Picture => (
                enc::VA_ENC_PACKED_HEADER_PICTURE,
                enc::VAEncPackedHeaderPicture,
            ),
            PackedKind::Slice => (enc::VA_ENC_PACKED_HEADER_SLICE, enc::VAEncPackedHeaderSlice),
        };
        if packed_headers & bit == 0 {
            continue;
        }
        let parameter = enc::VAEncPackedHeaderParameterBuffer {
            type_: kind,
            bit_length: u32::try_from(unit.bits).unwrap_or(0),
            has_emulation_bytes: 1,
            ..enc::VAEncPackedHeaderParameterBuffer::default()
        };
        buffers.push(Buffer::with_value(
            display,
            context,
            enc::VAEncPackedHeaderParameterBufferType,
            &parameter,
        )?);
        buffers.push(Buffer::with_bytes(
            display,
            context,
            enc::VAEncPackedHeaderDataBufferType,
            &unit.data,
        )?);
    }
    Ok(())
}

/// An empty coded buffer of `size` bytes, which the session owns and
/// destroys after its context.
fn create_coded_buffer(
    display: &Display,
    context: ffi::VAContextID,
    size: usize,
) -> Result<ffi::VABufferID, Error> {
    let size =
        u32::try_from(size).map_err(|_| Error::Unsupported("a 4 GiB coded buffer".into()))?;
    let mut id = ffi::VA_INVALID_ID;
    check(
        "vaCreateBuffer",
        // SAFETY: a null `data` asks libva for an uninitialised store, as a
        // coded buffer is (va.h, vaCreateBuffer).
        display.call(|raw| unsafe {
            ffi::vaCreateBuffer(
                raw,
                context,
                enc::VAEncCodedBufferType,
                size,
                1,
                core::ptr::null_mut(),
                &mut id,
            )
        })?,
    )?;
    Ok(id)
}

/// The most segments read of one coded buffer: ours, far more than a
/// picture is written in, so that a list that loops ends.
const MOST_SEGMENTS: usize = 64;

/// Waits for the coded buffer until `deadline`, and copies its segments
/// out, to `capacity` bytes at most.
fn read_coded(
    display: &Display,
    buffer: ffi::VABufferID,
    capacity: usize,
    deadline: MonoTime,
) -> Result<Vec<u8>, Error> {
    // A coded buffer of this display, with a picture ended into it.
    wait::wait_for(display, Work::Buffer(buffer), deadline, "an encode in time")?;
    let mut mapped: *mut core::ffi::c_void = core::ptr::null_mut();
    check(
        "vaMapBuffer",
        // SAFETY: mapping a coded buffer gives its first
        // VACodedBufferSegment (va.h, VACodedBufferSegment).
        display.call(|raw| unsafe { ffi::vaMapBuffer(raw, buffer, &mut mapped) })?,
    )?;
    let mut out = Vec::new();
    let mut status = 0;
    let mut segment = mapped.cast::<enc::VACodedBufferSegment>().cast_const();
    let mut result = Ok(());
    let mut segments = 0;
    // Past `capacity` the rest is not read: the overflow is refused below.
    while !segment.is_null() && out.len() <= capacity {
        segments += 1;
        if segments > MOST_SEGMENTS {
            result = Err(Error::Unsupported(format!(
                "a coded buffer of over {MOST_SEGMENTS} segments"
            )));
            break;
        }
        // SAFETY: the driver's segment list, valid while mapped.
        let s = unsafe { *segment };
        status |= s.status;
        if s.bit_offset != 0 {
            result = Err(Error::Unsupported(
                "a coded segment not starting on a byte".into(),
            ));
            break;
        }
        let size = usize::try_from(s.size).unwrap_or(0);
        if size > 0 && !s.buf.is_null() {
            // SAFETY: the segment's `size` bytes at `buf`, valid while
            // mapped.
            out.extend_from_slice(unsafe { core::slice::from_raw_parts(s.buf.cast::<u8>(), size) });
        }
        segment = s.next.cast::<enc::VACodedBufferSegment>().cast_const();
    }
    // SAFETY: mapped above, unmapped once, here.
    display.release(move |raw| unsafe {
        ffi::vaUnmapBuffer(raw, buffer);
    });
    result?;
    if status & enc::VA_CODED_BUF_STATUS_FRAME_SIZE_OVERFLOW != 0 || out.len() > capacity {
        return Err(Error::Unsupported(format!(
            "the picture overflowed its {capacity}-byte coded buffer"
        )));
    }
    if status & enc::VA_CODED_BUF_STATUS_BAD_BITSTREAM != 0 {
        return Err(Error::Unsupported(
            "the driver flagged a bad bitstream".into(),
        ));
    }
    Ok(out)
}
