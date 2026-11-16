//! VideoToolbox's paravirtual decoders on the host's hardware video ([`VtBackend`]): the backend
//! of the decoder sessions of [`Host`](reims_vgpu_video::paravirt::Host), which an arm64 guest's
//! VideoToolbox runs over the device's `VT_MESSAGE`s. Each session's decoder is one of the video
//! arm's [`GuestDecoder`]s, and each of its pictures lands in the IOSurface the guest's kext lent
//! for it ([`GuestSurface`]), through a view of the surface's guest RAM pages
//! ([`LentViews::view`]), laid out as the kext read the surface ([`layout_of`]). The arm's decoder
//! answers once the picture is there; the session then emits the frame, and lets go of the
//! surface with a `rels`.
//!
//! Its paravirtual encoders are the arm's too ([`VtEncoderBackend`]): each session's encoder is
//! one of the arm's [`GuestEncoder`]s, which reads each source frame from the IOSurface the guest
//! lent for it, through a view of its pages laid out as the kext read the surface; the session
//! emits the samples, and lets go of each frame's surface once it is encoded and emitted.
//!
//! A decoder or encoder keeps the views of the last [`VIEWS_KEPT`] surfaces it used, by their
//! runs, as VideoToolbox's pool lends the same few surfaces again and again: a view maps the
//! guest's RAM, so a kept one never goes stale, and what the arm made of its memory (the driver's
//! import of it) is kept with it.

use std::collections::VecDeque;
use std::sync::Arc;

use reims_vgpu_graphics::wire::{
    RGFX_VIDEO_CODEC_H264, RGFX_VIDEO_CODEC_HEVC, RGFX_VT_SURFACE_PLANES, RgfxVtSurface,
};
use reims_vgpu_video::codec::encode::{Coded, FailedSample};
use reims_vgpu_video::core::backend::BackendError as VideoError;
use reims_vgpu_video::core::fourcc::FourCc;
use reims_vgpu_video::core::surfaces::{PlaneLayout, SurfaceLayout};
use reims_vgpu_video::paravirt::status::{self, OsStatus};
use reims_vgpu_video::paravirt::{
    Backend, BackendDecoder, BackendEncoder, Codec, Decoded, DecoderConfig, EncoderBackend,
    EncoderConfig, LentSurface, SourceFrame,
};
use reims_vgpu_vulkan::core::num::ToUsize as _;
use reims_vgpu_vulkan::core::platform::Platform;

use super::GuestSurface;
use super::decode::{GraphicsVideo, GuestDecoder, GuestEncoder, GuestOutput, LentView, LentViews};

/// The views a decoder keeps, of the surfaces it landed in last.
const VIEWS_KEPT: usize = 8;

/// The host's hardware video as VideoToolbox's decoder sessions take it.
pub(super) struct VtBackend<P: Platform> {
    video: Arc<dyn GraphicsVideo<P>>,
    views: LentViews<P>,
}

impl<P: Platform> VtBackend<P> {
    /// The backend of `video`'s guest decoders, landing in views `views` makes.
    pub(super) fn new(video: Arc<dyn GraphicsVideo<P>>, views: LentViews<P>) -> Self {
        Self { video, views }
    }
}

impl<P: Platform> Backend for VtBackend<P> {
    fn open(&self, config: &DecoderConfig) -> Result<Box<dyn BackendDecoder>, OsStatus> {
        let codec = match config.codec {
            Codec::H264 => RGFX_VIDEO_CODEC_H264,
            Codec::Hevc => RGFX_VIDEO_CODEC_HEVC,
        };
        let decoder =
            (self.video.guest_decoder(codec)).ok_or(status::COULD_NOT_FIND_VIDEO_DECODER)?;
        Ok(Box::new(VtDecoder {
            decoder,
            views: self.views.clone(),
            kept: VecDeque::new(),
            said: Vec::new(),
        }))
    }
}

/// One session's decoder, and the views of the surfaces it landed in last.
struct VtDecoder<P: Platform> {
    decoder: Box<dyn GuestDecoder>,
    views: LentViews<P>,
    kept: Kept,
    /// The reasons a decode failed, each logged once.
    said: Vec<String>,
}

/// The views a decoder keeps, by their runs and bytes, the newest last.
type Kept = VecDeque<(Vec<(u64, u64)>, u64, LentView)>;

/// The view of the surface of `length` bytes in `runs`: one of `kept`, or one `views` makes now,
/// kept.
fn view_of<'a, P: Platform>(
    kept: &'a mut Kept,
    views: &LentViews<P>,
    runs: &[(u64, u64)],
    length: u64,
) -> Result<&'a LentView, String> {
    let at = (kept.iter()).position(|(r, n, _)| r.as_slice() == runs && *n == length);
    let view = match at.and_then(|at| kept.remove(at)) {
        Some(found) => found,
        None => (runs.to_vec(), length, views.view(runs, length)?),
    };
    if kept.len() == VIEWS_KEPT {
        kept.pop_front();
    }
    kept.push_back(view);
    let (_, _, view) = kept.back().ok_or("no view kept")?;
    Ok(view)
}

impl<P: Platform> VtDecoder<P> {
    /// Logs `why` a decode failed with `status`, once a decoder for each reason.
    fn failed(&mut self, status: OsStatus, why: String) -> OsStatus {
        if !self.said.contains(&why) {
            log::warn!(
                target: "reims_vgpu::graphics::vt",
                "hardware video: a VideoToolbox decode failed ({status}): {why}"
            );
            self.said.push(why);
        }
        status
    }
}

impl<P: Platform> BackendDecoder for VtDecoder<P> {
    fn decode(
        &mut self,
        parameter_sets: &[u8],
        access_unit: &[u8],
        output: &dyn LentSurface,
    ) -> Result<Decoded, OsStatus> {
        let Some(GuestSurface(surface)) = output.as_any().downcast_ref::<GuestSurface>() else {
            return Err(self.failed(status::PARAMETER, "an output no guest lent".to_owned()));
        };
        let layout = match layout_of(&surface.surface) {
            Ok(layout) => layout,
            Err(why) => {
                return Err(self.failed(status::VIDEO_DECODER_UNSUPPORTED_DATA_FORMAT, why));
            }
        };
        let view = match view_of(
            &mut self.kept,
            &self.views,
            &surface.runs,
            surface.surface.length,
        ) {
            Ok(view) => view,
            Err(why) => return Err(self.failed(status::VIDEO_DECODER_MALFUNCTION, why)),
        };
        let output = GuestOutput {
            layout,
            memory: &**view.memory(),
        };
        let decoded = self.decoder.decode(parameter_sets, access_unit, &output);
        match decoded {
            Ok(true) => Ok(Decoded::Landed),
            Ok(false) | Err(VideoError::SkippedLeadingPicture(_)) => Ok(Decoded::Skipped),
            Err(VideoError::BadData(why)) => Err(self.failed(status::VIDEO_DECODER_BAD_DATA, why)),
            Err(VideoError::Unsupported(why)) => {
                Err(self.failed(status::VIDEO_DECODER_UNSUPPORTED_DATA_FORMAT, why))
            }
            Err(VideoError::Malfunction(why)) => {
                Err(self.failed(status::VIDEO_DECODER_MALFUNCTION, why))
            }
        }
    }
}

/// The host's hardware video as VideoToolbox's encoder sessions take it.
pub(super) struct VtEncoderBackend<P: Platform> {
    video: Arc<dyn GraphicsVideo<P>>,
    views: LentViews<P>,
}

impl<P: Platform> VtEncoderBackend<P> {
    /// The backend of `video`'s guest encoders, reading source frames through views `views`
    /// makes.
    pub(super) fn new(video: Arc<dyn GraphicsVideo<P>>, views: LentViews<P>) -> Self {
        Self { video, views }
    }
}

/// The status a VideoToolbox encoder answers for `error`: a source or setting the encoder does
/// not take is `kVTParameterErr`, anything else its malfunction.
fn encoder_status(error: &VideoError) -> OsStatus {
    match error {
        VideoError::Unsupported(_) | VideoError::BadData(_) => status::PARAMETER,
        VideoError::Malfunction(_) | VideoError::SkippedLeadingPicture(_) => {
            status::VIDEO_ENCODER_MALFUNCTION
        }
    }
}

impl<P: Platform> EncoderBackend for VtEncoderBackend<P> {
    fn open(&self, config: &EncoderConfig) -> Result<Box<dyn BackendEncoder>, OsStatus> {
        let encoder = self.video.guest_encoder(config).map_err(|error| {
            log::warn!(
                target: "reims_vgpu::graphics::vt",
                "hardware video: no VideoToolbox encoder of {:?} at {}x{}: {error}",
                config.settings.profile,
                config.settings.width,
                config.settings.height
            );
            encoder_status(&error)
        })?;
        Ok(Box::new(VtEncoder {
            encoder,
            views: self.views.clone(),
            kept: VecDeque::new(),
            said: Vec::new(),
        }))
    }
}

/// One session's encoder, and the views of the surfaces it read last.
struct VtEncoder<P: Platform> {
    encoder: Box<dyn GuestEncoder>,
    views: LentViews<P>,
    kept: Kept,
    /// The reasons an encode failed, each logged once.
    said: Vec<String>,
}

impl<P: Platform> VtEncoder<P> {
    /// Logs why `error` failed an encode, once an encoder for each reason, and its status.
    fn failed(&mut self, error: &VideoError) -> OsStatus {
        let why = error.to_string();
        if !self.said.contains(&why) {
            log::warn!(
                target: "reims_vgpu::graphics::vt",
                "hardware video: a VideoToolbox encode failed: {why}"
            );
            self.said.push(why);
        }
        encoder_status(error)
    }

    /// `coded` as the session takes it: each failed frame with its status.
    fn statuses(&mut self, coded: Vec<Coded<VideoError>>) -> Vec<Coded<OsStatus>> {
        (coded.into_iter())
            .map(|coded| {
                coded.map_err(|failed| FailedSample {
                    id: failed.id,
                    error: failed.error.map(|error| self.failed(&error)),
                })
            })
            .collect()
    }
}

impl<P: Platform> BackendEncoder for VtEncoder<P> {
    fn encode(
        &mut self,
        source: &dyn LentSurface,
        frame: &SourceFrame,
    ) -> Result<Vec<Coded<OsStatus>>, OsStatus> {
        let Some(GuestSurface(surface)) = source.as_any().downcast_ref::<GuestSurface>() else {
            return Err(self.failed(&VideoError::Malfunction(
                "a source no guest lent".to_owned(),
            )));
        };
        let layout = layout_of(&surface.surface)
            .map_err(|why| self.failed(&VideoError::Unsupported(why)))?;
        let view = view_of(
            &mut self.kept,
            &self.views,
            &surface.runs,
            surface.surface.length,
        )
        .map_err(VideoError::Malfunction);
        let coded = match view {
            Ok(view) => {
                let source = GuestOutput {
                    layout,
                    memory: &**view.memory(),
                };
                self.encoder.encode(&source, frame)
            }
            Err(error) => Err(error),
        };
        match coded {
            Ok(coded) => Ok(self.statuses(coded)),
            Err(error) => Err(self.failed(&error)),
        }
    }

    fn flush(&mut self) -> Result<Vec<Coded<OsStatus>>, OsStatus> {
        match self.encoder.flush() {
            Ok(coded) => Ok(self.statuses(coded)),
            Err(error) => Err(self.failed(&error)),
        }
    }

    fn reconfigure(&mut self, config: &EncoderConfig) -> Result<(), OsStatus> {
        self.encoder
            .reconfigure(config)
            .map_err(|error| match error {
                VideoError::Unsupported(_) => status::PROPERTY_NOT_SUPPORTED,
                error => self.failed(&error),
            })
    }
}

/// The layout of the surface `record` describes, as the kext read it of the IOSurface: its
/// format, size and two planes, each from its offset at its own pitch. `Err` says why a surface
/// is not one a picture lands in: not of two planes.
pub(super) fn layout_of(record: &RgfxVtSurface) -> Result<SurfaceLayout, String> {
    if record.plane_count as usize != RGFX_VT_SURFACE_PLANES {
        return Err(format!(
            "a surface of {} planes, '{}'",
            record.plane_count,
            FourCc(record.pixel_format)
        ));
    }
    let planes = (record.planes.iter())
        .map(|plane| -> Result<PlaneLayout, String> {
            Ok(PlaneLayout {
                offset: usize::try_from(plane.offset).map_err(|_| "a plane past 2^64")?,
                bytes_per_row: plane.bytes_per_row.to_usize(),
                width: plane.width,
                height: plane.height,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(SurfaceLayout {
        pixel_format: FourCc(record.pixel_format),
        width: record.width,
        height: record.height,
        planes,
        bytes_per_row: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use reims_vgpu_graphics::wire::RgfxVtSurfacePlane;

    /// A 64x48 `420v` surface of two planes, as the kext writes its record.
    fn nv12() -> RgfxVtSurface {
        RgfxVtSurface {
            lend: 3,
            length: 0x4000,
            surface_id: 9,
            pixel_format: u32::from_be_bytes(*b"420v"),
            width: 64,
            height: 48,
            plane_count: 2,
            run_count: 1,
            planes: [
                RgfxVtSurfacePlane {
                    offset: 0x400,
                    bytes_per_row: 64,
                    width: 64,
                    height: 48,
                    reserved_14: 0,
                },
                RgfxVtSurfacePlane {
                    offset: 0x1000,
                    bytes_per_row: 64,
                    width: 32,
                    height: 24,
                    reserved_14: 0,
                },
            ],
        }
    }

    #[test]
    fn a_two_plane_surface_lays_out_as_its_record_says() {
        let layout = layout_of(&nv12()).unwrap();
        assert_eq!(layout.pixel_format, FourCc::new(*b"420v"));
        assert_eq!((layout.width, layout.height), (64, 48));
        assert_eq!(
            layout.planes,
            vec![
                PlaneLayout {
                    offset: 0x400,
                    bytes_per_row: 64,
                    width: 64,
                    height: 48,
                },
                PlaneLayout {
                    offset: 0x1000,
                    bytes_per_row: 64,
                    width: 32,
                    height: 24,
                },
            ]
        );
    }

    /// An encode the arm refuses for its source or settings is a parameter error to the guest,
    /// any other failure the encoder's malfunction.
    #[test]
    fn an_encode_failure_is_a_parameter_error_or_a_malfunction() {
        let error = |make: fn(String) -> VideoError| encoder_status(&make(String::new()));
        assert_eq!(error(VideoError::Unsupported), status::PARAMETER);
        assert_eq!(error(VideoError::BadData), status::PARAMETER);
        assert_eq!(
            error(VideoError::Malfunction),
            status::VIDEO_ENCODER_MALFUNCTION
        );
    }

    #[test]
    fn a_surface_without_two_planes_takes_no_picture() {
        let mut record = nv12();
        record.plane_count = 0;
        assert_eq!(
            layout_of(&record),
            Err("a surface of 0 planes, '420v'".to_owned())
        );
    }
}
