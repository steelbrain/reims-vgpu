//! Encoding on VA-API's `EncSlice` entrypoint: H.264 and HEVC, with the
//! parameter sets and slice headers ours, as packed headers.
//! [`VideoEncoder`] is [`crate::codec::encode`]'s core over
//! [`accelerator::VaAccelerator`].
//!
//! A dropped encoder's session stays open until a later encoder on the
//! render node has coded a picture and is about to code its next
//! (`accelerator`'s docs). On radeonsi, an encode session begun on the VCN
//! after the last one was closed, with none open between, hangs the VCN at
//! its first picture, faulting on a write to an address not mapped, and the
//! kernel resets the ring, or the whole GPU where that fails. A pause of 3 s
//! after the close does not avoid it, nor does a display of the new
//! session's own while the process holds another display of the GPU, whose
//! address space it shares. Sessions never closed, or each closed once the
//! next has coded its first picture, do not hang. (Measured with a libva
//! program of our own, Mesa 26.2.6, Linux 7.3.1: the hang on a Radeon 780M
//! of Phoenix and one of Hawk Point, within 2 to 4 sessions; 40 sessions in
//! a row each way without it on the Phoenix one.) iHD (Iris Xe) codes 90
//! sessions in a row, each closed before the next.

mod accelerator;
mod caps;
mod params;
mod surfaces;

use std::sync::Arc;

pub use caps::{EncoderCapabilities, Feature, HevcBlockSizes, HevcFeatures};
pub use surfaces::UploadMethod;

use super::caps::{Entrypoint, Profile};
use super::display::{Display, Error};
use super::ffi::{self, encode as enc};
use super::wait::{self, Deadline};
use crate::codec::encode::encoder::{DriverLimits, Encoder};
use crate::codec::encode::{
    self, Codec, Coded, EncoderSettings, FrameRate, InputFrame, RateControl, Unsupported, hevc,
};
use accelerator::VaAccelerator;

impl From<Unsupported> for Error {
    fn from(e: Unsupported) -> Self {
        Self::Unsupported(e.0)
    }
}

/// The VA profiles that encode a profile, best first: a Constrained
/// Baseline stream is within Main's and High's tools too.
fn va_profiles(profile: encode::Profile) -> &'static [Profile] {
    const CB: Profile = Profile(ffi::VAProfileH264ConstrainedBaseline);
    const MAIN: Profile = Profile(ffi::VAProfileH264Main);
    const HIGH: Profile = Profile(ffi::VAProfileH264High);
    const HEVC_MAIN: Profile = Profile(enc::VAProfileHEVCMain);
    const HEVC_MAIN10: Profile = Profile(enc::VAProfileHEVCMain10);
    match profile {
        encode::Profile::H264ConstrainedBaseline => &[CB, MAIN, HIGH],
        encode::Profile::H264Main => &[MAIN, HIGH],
        encode::Profile::H264High => &[HIGH],
        encode::Profile::HevcMain => &[HEVC_MAIN],
        encode::Profile::HevcMain10 => &[HEVC_MAIN10],
    }
}

/// The encode entrypoints, full first, then low power.
const ENTRYPOINTS: [Entrypoint; 2] = [
    Entrypoint(enc::VAEntrypointEncSlice),
    Entrypoint(enc::VAEntrypointEncSliceLP),
];

/// The packed headers the encoder writes: parameter sets and slice
/// headers.
const PACKED_HEADERS: u32 = enc::VA_ENC_PACKED_HEADER_SEQUENCE
    | enc::VA_ENC_PACKED_HEADER_PICTURE
    | enc::VA_ENC_PACKED_HEADER_SLICE;

/// One profile the driver encodes, and its limits.
#[derive(Clone, Debug)]
pub struct EncoderSupport {
    /// The profile.
    pub profile: encode::Profile,
    /// The VA profile and entrypoint that encode it.
    pub capabilities: EncoderCapabilities,
    /// The largest picture.
    pub max_width: u32,
    /// The tallest picture.
    pub max_height: u32,
}

/// The driver's encoder for `profile`: the first VA profile and entrypoint
/// it has, with their attributes.
fn find(display: &Display, profile: encode::Profile) -> Result<Option<EncoderCapabilities>, Error> {
    for &va_profile in va_profiles(profile) {
        for entrypoint in ENTRYPOINTS {
            if let Some(caps) = EncoderCapabilities::query(display, va_profile, entrypoint)? {
                let rt = caps.rt_formats.unwrap_or(0);
                let wanted = surfaces::va_format(profile.input_format()).0;
                if rt & wanted != 0 {
                    return Ok(Some(caps));
                }
            }
        }
    }
    Ok(None)
}

/// The profiles the driver encodes, of the five the encoders write.
pub fn encoder_support(display: &Display) -> Vec<EncoderSupport> {
    [
        encode::Profile::H264ConstrainedBaseline,
        encode::Profile::H264Main,
        encode::Profile::H264High,
        encode::Profile::HevcMain,
        encode::Profile::HevcMain10,
    ]
    .into_iter()
    .filter_map(|profile| {
        let capabilities = find(display, profile).ok()??;
        let max_width = capabilities.max_width.unwrap_or(0);
        let max_height = capabilities.max_height.unwrap_or(0);
        Some(EncoderSupport {
            profile,
            capabilities,
            max_width,
            max_height,
        })
    })
    .collect()
}

/// HEVC's tools from the driver's features and block sizes: what it
/// requires, and what it allows that the encoder wants. A tool it requires
/// that the headers cannot say is refused.
fn hevc_tools(caps: &EncoderCapabilities) -> Result<hevc::Tools, Unsupported> {
    let features = caps
        .hevc_feature_set()
        .ok_or_else(|| Unsupported("the driver reports no HEVC features".into()))?;
    let sizes = caps
        .hevc_block_size_set()
        .ok_or_else(|| Unsupported("the driver reports no HEVC block sizes".into()))?;
    let required = [
        ("separate colour planes", features.separate_colour_planes),
        ("scaling lists", features.scaling_lists),
        ("PCM", features.pcm),
        ("temporal MVP", features.temporal_mvp),
        ("dependent slices", features.dependent_slices),
        ("sign data hiding", features.sign_data_hiding),
        ("weighted prediction", features.weighted_prediction),
        ("transquant bypass", features.transquant_bypass),
    ];
    for (name, feature) in required {
        if feature == Feature::Required {
            return Err(Unsupported(format!("the driver requires {name}")));
        }
    }
    Ok(hevc::Tools {
        log2_ctb_size: 3 + sizes.log2_max_coding_tree_block_size_minus3,
        log2_min_cb_size: 3 + sizes.log2_min_luma_coding_block_size_minus3,
        log2_min_tb_size: 2 + sizes.log2_min_luma_transform_block_size_minus2,
        log2_max_tb_size: 2 + sizes.log2_max_luma_transform_block_size_minus2,
        max_transform_hierarchy_depth_inter: sizes.max_max_transform_hierarchy_depth_inter,
        max_transform_hierarchy_depth_intra: sizes.max_max_transform_hierarchy_depth_intra,
        amp: features.amp.resolve(true),
        sao: features.sao.resolve(true),
        strong_intra_smoothing: features.strong_intra_smoothing.resolve(true),
        transform_skip: features.transform_skip.resolve(false),
        cu_qp_delta: features.cu_qp_delta.resolve(true),
        cu_qp_delta_required: features.cu_qp_delta == Feature::Required,
        constrained_intra_pred: features.constrained_intra_pred.resolve(false),
    })
}

fn limits(
    display: &Display,
    caps: &EncoderCapabilities,
    codec: Codec,
) -> Result<DriverLimits, Error> {
    let rc = caps.rate_control.unwrap_or(0);
    let (max_l0, max_l1) = caps.max_references();
    let surfaces = display.surface_capabilities(caps.profile, caps.entrypoint)?;
    Ok(DriverLimits {
        constant_qp: rc & enc::VA_RC_CQP != 0,
        constant_bit_rate: rc & enc::VA_RC_CBR != 0,
        variable_bit_rate: rc & enc::VA_RC_VBR != 0,
        quality_variable_bit_rate: rc & enc::VA_RC_QVBR != 0,
        // Ours: VA-API has no way to ask. radeonsi passes the HRD buffer it
        // is given, and our rate control holds it on any driver that takes
        // a constant QP.
        keeps_buffer: false,
        max_l0,
        max_l1,
        min_width: surfaces.min_width.unwrap_or(1),
        min_height: surfaces.min_height.unwrap_or(1),
        max_width: caps.max_width.or(surfaces.max_width).unwrap_or(u32::MAX),
        max_height: caps.max_height.or(surfaces.max_height).unwrap_or(u32::MAX),
        hevc_tools: match codec {
            Codec::Hevc => Some(hevc_tools(caps)?),
            Codec::H264 => None,
        },
    })
}

/// An H.264 or HEVC encoder on a VA display.
///
/// Frames go in in display order, as host-memory NV12 or P010 planes;
/// samples come out in decode order, as length-prefixed NAL units with
/// their time stamps and sync flags, and the configuration record with the
/// first.
pub struct VideoEncoder {
    inner: Encoder<VaAccelerator>,
}

// Encoders move between threads: libva is thread-safe (va.h,
// "Multithreading Guide"), and each encoder is used through `&mut`.
const _: fn() = || {
    fn send<T: Send>() {}
    send::<VideoEncoder>();
};

impl VideoEncoder {
    /// An encoder for `settings`, refused with [`Error::Unsupported`] when
    /// the driver lacks the profile, size or rate control they ask for.
    pub fn new(display: &Display, settings: EncoderSettings) -> Result<Self, Error> {
        let caps = find(display, settings.profile)?.ok_or_else(|| {
            Error::Unsupported(format!("the driver does not encode {:?}", settings.profile))
        })?;
        let limits = limits(display, &caps, settings.profile.codec())?;
        let packed = caps.packed_headers.unwrap_or(0) & PACKED_HEADERS;
        let accelerator = VaAccelerator::new(display, caps.profile, caps.entrypoint, packed);
        Ok(Self {
            inner: Encoder::new(accelerator, settings, &limits)?,
        })
    }

    /// Encodes one frame, and returns each frame that comes out, in decode
    /// order, with its own result: none while it waits to be a B picture,
    /// several when it ends a run of them. `Err` is a frame not taken.
    pub fn encode(&mut self, frame: &InputFrame<'_>) -> Result<Vec<Coded<Error>>, Error> {
        // One deadline for the call's driver calls (`super::wait`).
        wait::by(&Deadline::default(), || self.inner.encode(frame))
    }

    /// Encodes the frames still waiting, and returns each with its own
    /// result. More frames may follow.
    pub fn flush(&mut self) -> Vec<Coded<Error>> {
        wait::by(&Deadline::default(), || self.inner.flush())
    }

    /// The configuration record (`avcC` or `hvcC`) of the parameter sets in
    /// use; the first sample carries it too.
    pub fn configuration(&self) -> Arc<[u8]> {
        self.inner.configuration()
    }

    /// The settings, with the changes made since.
    pub fn settings(&self) -> &EncoderSettings {
        self.inner.settings()
    }

    /// Changes the bitrate, or the QPs, from the next picture, within the
    /// same mode.
    pub fn set_rate_control(&mut self, rate_control: RateControl) -> Result<(), Error> {
        self.inner.set_rate_control(rate_control)
    }

    /// Changes the frame rate rate control expects, from the next picture.
    pub fn set_frame_rate(&mut self, frame_rate: FrameRate) -> Result<(), Error> {
        self.inner.set_frame_rate(frame_rate)
    }

    /// Changes the key-frame interval, in frames and in ticks, from the
    /// next frame.
    pub fn set_key_frame_interval(&mut self, frames: Option<u32>, duration: Option<i64>) {
        self.inner.set_key_frame_interval(frames, duration);
    }

    /// The size the hardware encodes before the SPS crops it.
    pub fn coded_size(&self) -> (u32, u32) {
        self.inner.coded_size()
    }

    /// B pictures in a run, 0 when frames keep their order.
    pub fn b_frames(&self) -> u32 {
        self.inner.b_frames()
    }

    /// The level the SPS says.
    pub fn level_idc(&self) -> u8 {
        self.inner.level_idc()
    }

    /// The VA profile and entrypoint in use.
    pub fn va_profile(&self) -> (Profile, Entrypoint) {
        let a = self.inner.accelerator();
        (a.profile(), a.entrypoint())
    }

    /// The `VA_ENC_PACKED_HEADER_*` bits the driver takes from us.
    pub fn packed_headers(&self) -> u32 {
        self.inner.accelerator().packed_headers()
    }

    /// How frames reach the driver, once the first has.
    pub fn upload_method(&self) -> Option<UploadMethod> {
        self.inner.accelerator().upload_method()
    }

    /// Uploads frames by `method` from now on.
    pub fn force_upload_method(&mut self, method: UploadMethod) {
        self.inner.accelerator_mut().force_upload_method(method);
    }
}

#[cfg(test)]
mod tests {
    use crate::core::runtime::StdRuntime;

    use super::*;
    use crate::codec::encode::Plane;

    /// On the driver, 24 encoders made from one display one after another, each dropped before
    /// the next, the profiles the driver encodes in turn, code each of their 4 frames: a dropped
    /// encoder's session is closed only once the next one has coded a picture (module docs).
    #[test]
    #[ignore = "needs a VA-API GPU: run on the VA-API host"]
    fn encoders_made_one_after_another_from_one_display_code_every_frame() {
        let node = std::env::var("RVG_VAAPI_NODE").unwrap_or_else(|_| "/dev/dri/renderD128".into());
        let display = Display::open(&node, Arc::new(StdRuntime)).expect("a VA display");
        let profiles: Vec<encode::Profile> = (encoder_support(&display).into_iter())
            .map(|s| s.profile)
            .collect();
        assert!(!profiles.is_empty(), "the driver encodes nothing");
        let (width, height) = (320, 192);
        for made in 0..24 {
            let profile = profiles[made % profiles.len()];
            let settings = EncoderSettings::new(profile, width, height);
            let mut encoder = VideoEncoder::new(&display, settings)
                .unwrap_or_else(|e| panic!("encoder {made} ({profile:?}): {e}"));
            let row = width as usize * profile.input_format().bytes_per_sample();
            let luma = vec![0x40; row * height as usize];
            let chroma = vec![0x80; row * height as usize / 2];
            let mut coded = Vec::new();
            for id in 0..4 {
                let frame = InputFrame {
                    planes: [
                        Plane {
                            data: &luma,
                            bytes_per_row: row,
                        },
                        Plane {
                            data: &chroma,
                            bytes_per_row: row,
                        },
                    ],
                    pts: 3000 * id as i64,
                    duration: 3000,
                    force_key_frame: false,
                    id,
                };
                let out = (encoder.encode(&frame))
                    .unwrap_or_else(|e| panic!("encoder {made} ({profile:?}) frame {id}: {e}"));
                coded.extend(out);
            }
            coded.extend(encoder.flush());
            let samples = (coded.into_iter())
                .map(|c| c.map_err(|f| (f.id, f.error.map(|e| e.to_string()))))
                .collect::<Result<Vec<_>, _>>()
                .unwrap_or_else(|e| panic!("encoder {made} ({profile:?}): {e:?}"));
            assert_eq!(samples.len(), 4, "encoder {made} ({profile:?})");
        }
    }
}
