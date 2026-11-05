//! The host session's backend on VA-API: core's [`DecoderBackend`] over
//! [`H264Decoder`], [`HevcDecoder`], [`Av1Decoder`] and [`Vp9Decoder`], each
//! driven by one [`Session`], so that the guest's VideoToolbox decodes
//! H.264, HEVC, AV1 and VP9 on the host's GPU.
//!
//! Core asks for capabilities when the guest lists decoders, opens a
//! decoder when the guest starts a decode session, and copies each frame
//! into the guest's pixel buffer through [`DecodedFrame::read_back`] and
//! [`FrameReadBack::copy_to`]. Frames come out in decode order, as
//! VideoToolbox's own decoders emit them.

mod av1_session;
mod vp9_session;

use std::path::Path;
use std::sync::{Arc, OnceLock};

use reims_vgpu_video_core::backend::{
    BackendDecoder, BackendError, Codec, CodecCapabilities, CompressedFrame, DecodedFrame,
    DecoderBackend, DecoderConfig, DeviceWrite, FrameDecoded, FrameReadBack, PixelFormat,
    ProfileSupport,
};
use reims_vgpu_video_core::device_memory::GuestTarget;
use reims_vgpu_video_core::runtime::Runtime;
use reims_vgpu_video_core::surfaces::PlaneDestination;

use super::av1::Av1Decoder;
use super::caps::{Entrypoint, Profile};
use super::display::{Display, Error};
use super::ffi;
use super::frame::{Frame, ReadBack};
use super::h264::H264Decoder;
use super::hevc::HevcDecoder;
use super::vp9::Vp9Decoder;
use super::wait;
use super::zero_copy::{self, GuestImports};
use crate::av1;
use crate::h264::{self, avcc::DecoderConfiguration, sps::Sps};
use crate::hevc;
use crate::vp9;

/// H.264's profiles as the guest names them (`profile_idc`), each with the
/// VA profiles that decode it, best first: a Baseline stream decodes on
/// Main or High as well, and a Main one on High (the order
/// [`H264Decoder`] picks a profile in).
const H264_PROFILES: [(u32, &[Profile]); 3] = [
    (
        66,
        &[
            Profile::H264_CONSTRAINED_BASELINE,
            Profile::H264_MAIN,
            Profile::H264_HIGH,
        ],
    ),
    (77, &[Profile::H264_MAIN, Profile::H264_HIGH]),
    (100, &[Profile::H264_HIGH]),
];

/// HEVC's profiles as the guest names them (`general_profile_idc`), each
/// with the VA profile that decodes it. Main Still Picture (3) is Main
/// limited to one picture, which VA-API has no profile of its own for: its
/// streams decode as Main (hevc-check.sh's mainstill-1080p, radeonsi).
/// Apple lists 1 to 4; the format range extensions (4) are refused.
const HEVC_PROFILES: [(u32, Profile); 3] = [
    (1, Profile::HEVC_MAIN),
    (2, Profile::HEVC_MAIN10),
    (3, Profile::HEVC_MAIN),
];

/// The widest VP9 or AV1 picture of more than 8 bits the backend refuses
/// ([`VaapiBackend::refusal`]): radeonsi decoded 10-bit inter frames 16,
/// 24 and 32 samples wide wrongly, and 33 to 48 and wider right, at any
/// height (the RX 7900 XTX and the Radeon 780M, Mesa 26.2.4, 2026-10-24).
const NARROWEST_DEEP: u32 = 32;

/// Decoders on one VA display.
pub struct VaapiBackend {
    display: Display,
    capabilities: Vec<CodecCapabilities>,
    /// The guest buffers its frames are written into, imported.
    imports: Arc<GuestImports>,
    /// The smallest picture each codec listed decodes ([`smallest`]).
    smallest: Vec<(Codec, (u32, u32))>,
}

impl VaapiBackend {
    /// A backend on the render node at `path`, such as
    /// `/dev/dri/renderD128`, on `runtime`.
    pub fn open(path: impl AsRef<Path>, runtime: Arc<dyn Runtime>) -> Result<Self, Error> {
        Ok(Self::new(Display::open(path, runtime)?))
    }

    /// A backend on `display`. What it decodes is read from the driver
    /// now, once.
    pub fn new(display: Display) -> Self {
        let capabilities = [
            h264_capabilities(&display),
            hevc_capabilities(&display),
            av1_session::capabilities(&display),
            vp9_session::capabilities(&display),
        ]
        .into_iter()
        .flatten()
        .collect();
        let h264 = H264_PROFILES
            .iter()
            .flat_map(|(_, profiles)| profiles.iter().copied());
        let smallest = vec![
            (Codec::H264, smallest(&display, h264)),
            (
                Codec::Hevc,
                smallest(&display, HEVC_PROFILES.map(|(_, profile)| profile)),
            ),
            (Codec::Av1, smallest(&display, [super::av1::AV1_MAIN])),
            (Codec::Vp9, smallest(&display, vp9_session::VA_PROFILES)),
        ];
        Self {
            imports: zero_copy::imports(&display),
            display,
            capabilities,
            smallest,
        }
    }

    /// Why the driver cannot start a session for `config`, before one is
    /// made: a picture under its smallest, which `vaCreateContext` would
    /// refuse at the first frame, so that every frame failed (radeonsi,
    /// 2026-10-24). It is kVTVideoDecoderMalfunctionErr at `dsta`, with no
    /// `Dpba`, as Apple's decoders refuse a size they do not decode; the
    /// guest then decodes the stream itself.
    ///
    /// So is VP9 or AV1 of more than 8 bits whose pictures are
    /// [`NARROWEST_DEEP`] or fewer samples wide: radeonsi decodes their
    /// inter frames wrongly, with no error, as ffmpeg's VA-API decode of
    /// the same streams does (the RX 7900 XTX and the Radeon 780M, Mesa
    /// 26.2.4, 2026-10-24), and a wrong picture is never shown for a
    /// capability's sake.
    fn refusal(&self, config: &DecoderConfig) -> Option<BackendError> {
        let (width, height) = session_size(config)?;
        let &(_, (min_width, min_height)) = self
            .smallest
            .iter()
            .find(|(codec, _)| *codec == config.codec)?;
        if width < min_width || height < min_height {
            return Some(BackendError::Malfunction(format!(
                "{:?} coded at {width}x{height}: the driver decodes from {min_width}x{min_height}",
                config.codec
            )));
        }
        let deep = match config.codec {
            Codec::Vp9 => vp9::vpcc::ConfigurationRecord::parse(&config.configuration)
                .is_ok_and(|record| record.bit_depth > 8),
            Codec::Av1 => av1::av1c::ConfigurationRecord::parse(&config.configuration)
                .is_ok_and(|record| record.bit_depth() > 8),
            Codec::H264 | Codec::Hevc => false,
        };
        (deep && width <= NARROWEST_DEEP).then(|| {
            BackendError::Malfunction(format!(
                "{:?} of more than 8 bits at {width}x{height}: radeonsi decodes inter frames \
                 {NARROWEST_DEEP} or fewer samples wide wrongly",
                config.codec
            ))
        })
    }
}

/// The smallest picture any of `profiles` decodes, as the driver says:
/// the largest `VASurfaceAttribMinWidth` and `VASurfaceAttribMinHeight`
/// among their decode configs, so that no size one of them refuses is
/// taken; 1x1 where the driver says none. Mesa's `vaCreateContext` refuses
/// a picture under them with `VA_STATUS_ERROR_RESOLUTION_NOT_SUPPORTED`:
/// radeonsi's are 64x64 for H.264 and HEVC and 16x16 for VP9 and AV1, on
/// the RX 7900 XTX and the Radeon 780M (2026-10-24).
fn smallest(display: &Display, profiles: impl IntoIterator<Item = Profile>) -> (u32, u32) {
    profiles
        .into_iter()
        .filter(|&profile| display.supports(profile, Entrypoint::VLD))
        .filter_map(|profile| display.surface_capabilities(profile, Entrypoint::VLD).ok())
        .fold((1, 1), |(width, height), surfaces| {
            (
                width.max(surfaces.min_width.unwrap_or(1)),
                height.max(surfaces.min_height.unwrap_or(1)),
            )
        })
}

/// The size a session for `config` decodes its pictures at, as the
/// driver's context would be made: H.264's and HEVC's coded size from the
/// record's first SPS, AV1's largest frame from its sequence header, and
/// else the description's. `None` where neither says.
fn session_size(config: &DecoderConfig) -> Option<(u32, u32)> {
    let coded = match config.codec {
        Codec::H264 => DecoderConfiguration::parse(&config.configuration)
            .ok()
            .and_then(|record| record.sequence_parameter_sets.first().copied())
            .and_then(|sps| Sps::parse(sps).ok())
            .map(|sps| (sps.coded_width(), sps.coded_height())),
        Codec::Hevc => hevc_sps(&config.configuration).map(|sps| {
            (
                sps.pic_width_in_luma_samples,
                sps.pic_height_in_luma_samples,
            )
        }),
        Codec::Av1 => av1::av1c::ConfigurationRecord::parse(&config.configuration)
            .ok()
            .and_then(|record| record.sequence_header_obu().ok().flatten())
            .and_then(|obu| av1::sequence::SequenceHeader::parse(obu).ok())
            .map(|sequence| sequence.max_frame_size()),
        Codec::Vp9 => None,
    };
    coded.or((config.width > 0 && config.height > 0).then_some((config.width, config.height)))
}

/// The largest picture `profile` decodes, as the driver says.
fn max_size(display: &Display, profile: Profile) -> (u32, u32) {
    let limits = display
        .config_attributes(
            profile,
            Entrypoint::VLD,
            &[
                ffi::VAConfigAttribMaxPictureWidth,
                ffi::VAConfigAttribMaxPictureHeight,
            ],
        )
        .unwrap_or_default();
    match limits[..] {
        [width, height] => (width.unwrap_or(0), height.unwrap_or(0)),
        _ => (0, 0),
    }
}

/// What the driver decodes of HEVC: Main and Main 10 where it has their VA
/// profiles, and the largest picture those take. None when it has neither.
fn hevc_capabilities(display: &Display) -> Option<CodecCapabilities> {
    let mut profiles = Vec::new();
    let (mut max_width, mut max_height) = (0, 0);
    for (general_profile_idc, profile) in HEVC_PROFILES {
        if !display.supports(profile, Entrypoint::VLD) {
            continue;
        }
        // VA-API reports no level: the host session lists Apple's.
        profiles.push(ProfileSupport {
            profile: general_profile_idc,
            max_level: None,
        });
        let (width, height) = max_size(display, profile);
        max_width = max_width.max(width);
        max_height = max_height.max(height);
    }
    (!profiles.is_empty()).then_some(CodecCapabilities {
        codec: Codec::Hevc,
        profiles,
        max_width,
        max_height,
    })
}

/// What the driver decodes of H.264: the profiles any of whose VA profiles
/// it has, and the largest picture those take. None when it has none.
fn h264_capabilities(display: &Display) -> Option<CodecCapabilities> {
    let mut profiles = Vec::new();
    let (mut max_width, mut max_height) = (0, 0);
    for (profile_idc, candidates) in H264_PROFILES {
        let supported: Vec<Profile> = candidates
            .iter()
            .copied()
            .filter(|&profile| display.supports(profile, Entrypoint::VLD))
            .collect();
        if supported.is_empty() {
            continue;
        }
        // VA-API reports no level: the host session lists what Apple's
        // decoder lists for a profile without one.
        profiles.push(ProfileSupport {
            profile: profile_idc,
            max_level: None,
        });
        for profile in supported {
            let (width, height) = max_size(display, profile);
            max_width = max_width.max(width);
            max_height = max_height.max(height);
        }
    }
    (!profiles.is_empty()).then_some(CodecCapabilities {
        codec: Codec::H264,
        profiles,
        max_width,
        max_height,
    })
}

impl DecoderBackend for VaapiBackend {
    fn capabilities(&self) -> Vec<CodecCapabilities> {
        self.capabilities.clone()
    }

    fn open(&self, config: &DecoderConfig) -> Result<Box<dyn BackendDecoder>, BackendError> {
        if let Some(refusal) = self.refusal(config) {
            return Err(refusal);
        }
        let (display, imports) = (&self.display, &self.imports);
        match config.codec {
            Codec::H264 => Session::<H264Decoder>::boxed(display, imports, config),
            Codec::Hevc => Session::<HevcDecoder>::boxed(display, imports, config),
            Codec::Av1 => Session::<Av1Decoder>::boxed(display, imports, config),
            Codec::Vp9 => Session::<Vp9Decoder>::boxed(display, imports, config),
        }
    }
}

/// One codec's decoder, as a [`Session`] drives it.
trait StreamDecoder: Sized + Send + 'static {
    /// The guest's codec it decodes.
    const CODEC: Codec;

    /// The decoder of `config`'s stream, and the pixel format its frames
    /// come in.
    fn open(display: &Display, config: &DecoderConfig)
    -> Result<(Self, PixelFormat), BackendError>;

    /// Whether a session of this codec whose frames come in `format` goes
    /// on with `config` without being reopened
    /// ([`BackendDecoder::can_accept`]).
    fn accepts(display: &Display, config: &DecoderConfig, format: PixelFormat) -> bool;

    /// Decodes one sample: the frames ready for output.
    fn decode(&mut self, sample: &[u8], id: u64) -> Result<Vec<Frame>, Error>;

    /// The frames held back for reordering, at the end of the stream: none
    /// where the codec holds none back.
    fn flush(&mut self) -> Result<Vec<Frame>, Error> {
        Ok(Vec::new())
    }

    /// [`BackendDecoder::frames_decoded`]: none where each sample is one
    /// frame.
    fn frames_decoded(&mut self) -> Vec<FrameDecoded> {
        Vec::new()
    }
}

/// One decode session, of any codec.
struct Session<D> {
    display: Display,
    imports: Arc<GuestImports>,
    decoder: D,
    configuration: Vec<u8>,
    format: PixelFormat,
    /// What the GPU did not finish by its deadline for a frame of the
    /// session, once it has not: each later sample fails at once
    /// (`super::wait`). Shared with the session's frames, which set it, and
    /// kept through a new record's decoder.
    stalled: Arc<OnceLock<String>>,
}

impl<D: StreamDecoder> Session<D> {
    fn new(
        display: &Display,
        imports: &Arc<GuestImports>,
        config: &DecoderConfig,
        stalled: Arc<OnceLock<String>>,
    ) -> Result<Self, BackendError> {
        let (decoder, format) = D::open(display, config)?;
        Ok(Self {
            display: display.clone_ref(),
            imports: Arc::clone(imports),
            decoder,
            configuration: config.configuration.clone(),
            format,
            stalled,
        })
    }

    fn boxed(
        display: &Display,
        imports: &Arc<GuestImports>,
        config: &DecoderConfig,
    ) -> Result<Box<dyn BackendDecoder>, BackendError> {
        let stalled = Arc::new(OnceLock::new());
        Ok(Box::new(Self::new(display, imports, config, stalled)?))
    }
}

impl<D: StreamDecoder> BackendDecoder for Session<D> {
    fn output_format(&self) -> PixelFormat {
        self.format
    }

    fn decode(
        &mut self,
        frame: &CompressedFrame<'_>,
    ) -> Result<Vec<Box<dyn DecodedFrame>>, BackendError> {
        if let Some(what) = self.stalled.get() {
            return Err(backend_error(Error::Stalled(format!(
                "{what}, of an earlier frame of the session"
            ))));
        }
        if let Some(config) = frame.new_config
            && config.configuration != self.configuration
        {
            // A new record: its parameter sets or sequence replace the old
            // ones from this frame on, which the guest starts at a random
            // access point.
            let stalled = Arc::clone(&self.stalled);
            *self = Self::new(&self.display, &self.imports, config, stalled)?;
        }
        let decoded = self
            .decoder
            .decode(frame.data, frame.frame_id.cast_unsigned());
        let frames = seen(&self.stalled, decoded)?;
        Ok(boxed(frames, &self.imports, &self.stalled))
    }

    fn flush(&mut self) -> Result<Vec<Box<dyn DecodedFrame>>, BackendError> {
        let flushed = self.decoder.flush();
        Ok(boxed(
            seen(&self.stalled, flushed)?,
            &self.imports,
            &self.stalled,
        ))
    }

    fn can_accept(&self, config: &DecoderConfig) -> bool {
        config.codec == D::CODEC && D::accepts(&self.display, config, self.format)
    }

    /// Each decode is handed to the GPU, and each frame's [`wait`] waits
    /// for its own: the next frame decodes while the last is copied back.
    /// Mesa's VA entry points take the driver's lock, and a frame holds its
    /// surface and its pool, so a reconfiguration's new context leaves the
    /// frame being copied whole.
    ///
    /// [`wait`]: reims_vgpu_video_core::backend::DecodedFrame::wait
    fn decodes_while_copying(&self) -> bool {
        true
    }

    fn frames_decoded(&mut self) -> Vec<FrameDecoded> {
        self.decoder.frames_decoded()
    }
}

/// H.264's frames are NV12 in the range of the record's first SPS, and a
/// session goes on with any record that opens.
impl StreamDecoder for H264Decoder {
    const CODEC: Codec = Codec::H264;

    fn open(
        display: &Display,
        config: &DecoderConfig,
    ) -> Result<(Self, PixelFormat), BackendError> {
        let decoder = H264Decoder::new(display, &config.configuration).map_err(backend_error)?;
        Ok((decoder, nv12(avcc_is_full_range(&config.configuration))))
    }

    fn accepts(display: &Display, config: &DecoderConfig, _format: PixelFormat) -> bool {
        H264Decoder::new(display, &config.configuration).is_ok()
    }

    fn decode(&mut self, sample: &[u8], id: u64) -> Result<Vec<Frame>, Error> {
        H264Decoder::decode(self, sample, id)
    }

    fn flush(&mut self) -> Result<Vec<Frame>, Error> {
        H264Decoder::flush(self)
    }
}

/// Whether the `avcC` record's first SPS says full range; video range when
/// it has none or the SPS does not read.
fn avcc_is_full_range(avcc: &[u8]) -> bool {
    DecoderConfiguration::parse(avcc)
        .ok()
        .and_then(|record| record.sequence_parameter_sets.first().copied())
        .and_then(|sps| Sps::parse(sps).ok())
        .is_some_and(|sps| sps.is_full_range())
}

fn nv12(full_range: bool) -> PixelFormat {
    if full_range {
        PixelFormat::Nv12FullRange
    } else {
        PixelFormat::Nv12VideoRange
    }
}

/// HEVC's frames come in the depth and range of the record's first SPS,
/// which a record the session goes on with keeps.
impl StreamDecoder for HevcDecoder {
    const CODEC: Codec = Codec::Hevc;

    fn open(
        display: &Display,
        config: &DecoderConfig,
    ) -> Result<(Self, PixelFormat), BackendError> {
        let decoder = HevcDecoder::new(display, &config.configuration).map_err(backend_error)?;
        Ok((decoder, hvcc_format(&config.configuration)))
    }

    fn accepts(display: &Display, config: &DecoderConfig, format: PixelFormat) -> bool {
        hvcc_format(&config.configuration) == format
            && HevcDecoder::new(display, &config.configuration).is_ok()
    }

    fn decode(&mut self, sample: &[u8], id: u64) -> Result<Vec<Frame>, Error> {
        HevcDecoder::decode(self, sample, id)
    }

    fn flush(&mut self) -> Result<Vec<Frame>, Error> {
        HevcDecoder::flush(self)
    }
}

/// The `hvcC` record's first SPS, where one reads.
fn hevc_sps(hvcc: &[u8]) -> Option<hevc::sps::Sps> {
    hevc::hvcc::DecoderConfiguration::parse(hvcc)
        .ok()
        .and_then(|record| {
            record.nal_units.iter().find_map(|unit| {
                hevc::nal::NalUnit::new(unit)
                    .ok()
                    .filter(|nal| nal.unit_type() == hevc::nal::nal_type::SPS)
                    .and_then(|_| hevc::sps::Sps::parse(unit).ok())
            })
        })
}

/// The pixel format the `hvcC` record's first SPS decodes into: P010 for
/// 10 bits, NV12 for 8, each in its range; 8-bit video range when the
/// record has no SPS that reads.
fn hvcc_format(hvcc: &[u8]) -> PixelFormat {
    match hevc_sps(hvcc) {
        Some(sps) => pixel_format(
            u32::from(sps.bit_depth_luma_minus8) + 8,
            sps.is_full_range(),
        ),
        None => nv12(false),
    }
}

/// The CoreVideo format of frames at `bit_depth`, in their range.
fn pixel_format(bit_depth: u32, full_range: bool) -> PixelFormat {
    match (bit_depth > 8, full_range) {
        (false, _) => nv12(full_range),
        (true, false) => PixelFormat::P010VideoRange,
        (true, true) => PixelFormat::P010FullRange,
    }
}

fn boxed(
    frames: Vec<Frame>,
    imports: &Arc<GuestImports>,
    stalled: &Arc<OnceLock<String>>,
) -> Vec<Box<dyn DecodedFrame>> {
    frames
        .into_iter()
        .map(|frame| {
            let frame = VaapiFrame(frame, Arc::clone(imports), Arc::clone(stalled));
            Box::new(frame) as Box<dyn DecodedFrame>
        })
        .collect()
}

/// `result` as the guest sees it, its session's `stalled` set where it is
/// [`Error::Stalled`].
fn seen<T>(stalled: &OnceLock<String>, result: Result<T, Error>) -> Result<T, BackendError> {
    result.map_err(|error| {
        if let Error::Stalled(what) = &error {
            // Set once: the first stall is the one named.
            let _ = stalled.set(what.clone());
        }
        backend_error(error)
    })
}

/// A decoded frame, still on its VA surface until copied, the backend's
/// imports of guest buffers it may be written into, and its session's
/// stall (`Session::stalled`).
struct VaapiFrame(Frame, Arc<GuestImports>, Arc<OnceLock<String>>);

impl DecodedFrame for VaapiFrame {
    fn frame_id(&self) -> i64 {
        self.0.id().cast_signed()
    }

    fn pixel_format(&self) -> PixelFormat {
        pixel_format(self.0.bit_depth(), self.0.is_full_range())
    }

    fn width(&self) -> u32 {
        self.0.width()
    }

    fn height(&self) -> u32 {
        self.0.height()
    }

    /// Waits for the decode and has the GPU copy the frame into host
    /// memory, by the pool's way back; the copy out of it is the CPU's.
    fn read_back(&self) -> Result<Box<dyn FrameReadBack + '_>, BackendError> {
        Ok(Box::new(VaapiReadBack(seen(&self.2, self.0.read_back())?)))
    }

    fn wait(&self) -> Result<(), BackendError> {
        seen(&self.2, self.0.sync())
    }

    /// One video-processing run from the decode surface into the guest's
    /// buffer, imported once per mapping.
    fn write_to_guest(&self, target: &GuestTarget<'_>) -> Result<DeviceWrite, BackendError> {
        let written = wait::by(&self.0.deadline, || {
            zero_copy::write(&self.0, target, &self.1)
        });
        seen(&self.2, written)
    }
}

/// A frame read back into host memory.
struct VaapiReadBack<'a>(ReadBack<'a>);

impl FrameReadBack for VaapiReadBack<'_> {
    fn copy_to(&self, planes: &mut [PlaneDestination<'_>]) -> Result<(), BackendError> {
        let [luma, chroma, ..] = planes else {
            return Err(BackendError::Malfunction(format!(
                "4:2:0 needs two destination planes, not {}",
                planes.len()
            )));
        };
        // A guest's buffers smaller than the frame take what fits, as the
        // trait says.
        self.0
            .copy_fitting_to(
                (
                    &mut *luma.bytes,
                    luma.bytes_per_row,
                    usize::try_from(luma.rows).unwrap_or(usize::MAX),
                ),
                (
                    &mut *chroma.bytes,
                    chroma.bytes_per_row,
                    usize::try_from(chroma.rows).unwrap_or(usize::MAX),
                ),
            )
            .map_err(backend_error)
    }
}

/// The status the guest sees: what the stream asks for that the decoder or
/// driver lacks is an unsupported format, a stream that does not decode is
/// bad data, a RASL picture skipped is Apple's skipped leading picture, and
/// the rest is the decoder's malfunction.
pub(super) fn backend_error(error: Error) -> BackendError {
    let text = error.to_string();
    match error {
        Error::HevcBitstream(hevc::Error::SkippedLeadingPicture) => {
            BackendError::SkippedLeadingPicture(text)
        }
        Error::Bitstream(h264::Error::Unsupported(_))
        | Error::HevcBitstream(hevc::Error::Unsupported(_))
        | Error::Av1Bitstream(av1::Error::Unsupported(_))
        | Error::Vp9Bitstream(vp9::Error::Unsupported(_))
        | Error::Unsupported(_) => BackendError::Unsupported(text),
        Error::Bitstream(_)
        | Error::HevcBitstream(_)
        | Error::Av1Bitstream(_)
        | Error::Vp9Bitstream(_) => BackendError::BadData(text),
        Error::Load(_)
        | Error::Open(_)
        | Error::NoDisplay
        | Error::Va { .. }
        | Error::OutOfSurfaces
        | Error::PlaneTooSmall(_)
        | Error::Stalled(_) => BackendError::Malfunction(text),
    }
}
