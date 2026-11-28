//! VA-API: decode, and encode where the driver has it, on the host's GPU
//! through libva alone, with no FFmpeg or other layer between, over the
//! codecs of [`crate::codec`].
//!
//! - The libva binding and the decoders on it: [`Display`],
//!   [`H264Decoder`], [`HevcDecoder`], [`Vp9Decoder`], [`Frame`].
//! - [`VideoEncoder`] runs [`crate::codec::encode`]'s encoders on VA-API.

mod va;

pub use va::{AV1_PROFILE0, Av1Decoder, av1_decodes_ten_bit, av1_limits};

pub use va::{
    Capabilities, ConfigCapabilities, Display, Entrypoint, Error, ExportedPlane, Fourcc, Frame,
    FrameHold, GuestWrites, H264_MAX_MACROBLOCKS, H264Decoder, HevcDecoder, ImportWrite,
    ImportWrites, LoadError, PackedTarget, PlaneRows, Profile, ReadBack, ReadbackTimes,
    ReadbackWay, SurfaceCapabilities, SurfaceExport, proc_filter_name, range_extensions_decoder,
    read_back_by, rt_format_names,
};

/// What the paravirtual route checks a stream's start against: the profiles a decoder of it
/// decodes on.
pub use va::{h264_decoder_profiles, hevc_main_decoder};

/// A pool's surface on Windows, a shared Direct3D 12 texture, which a frame names
/// (`Frame::texture`).
#[cfg(windows)]
pub use va::SharedTexture;

/// On Windows, a picture's planes drawn from its shared texture into two textures of one plane
/// each, which another device imports (`PlaneDraw::draw`).
#[cfg(windows)]
pub use va::{PlaneDraw, PlaneTargets};

pub use va::{Vp9Decoder, vp9_decoder};

pub use va::{
    TARGET_ALIGNMENT, Target, TargetChecks, TargetCodec, TargetFor, TargetFormat, TargetPlane,
    TargetShape, TargetTiling,
};

/// The probe of decodes into targets over memory of ours, which `crate::probes` runs by hand.
#[cfg(all(test, unix))]
pub(crate) use va::target_probe;

/// The tests' runtime with an exit gate, which the NVIDIA arm's tests run on too.
#[cfg(all(test, unix))]
pub(crate) use va::gated;

pub use va::{
    EncoderCapabilities, EncoderSupport, Feature, HevcBlockSizes, HevcFeatures, InputLayout,
    InputSurface, UploadMethod, VideoEncoder, encoder_support,
};
