//! The VA-API backend: decode (and, where the driver has it, encode and
//! JPEG decode) on the host's GPU through libva alone, with no FFmpeg or
//! other layer between.
//!
//! - [`h264`] and [`hevc`] are H.264's and HEVC's syntax and decoding
//!   process above the slice data, and [`vp9`] VP9's above the compressed
//!   header: plain Rust, on every host.
//! - The libva binding and the decoders on it ([`Display`],
//!   [`H264Decoder`], [`Frame`]) are Linux's alone, and absent elsewhere.
//! - [`VaapiBackend`], also Linux's, is the host session's backend on
//!   them: `reims_vgpu_video_core::backend::DecoderBackend` over VA-API.
//! - [`encode`] is the encoders' portable half: their settings and
//!   samples, the H.264 and HEVC headers they write, and the planning of
//!   each picture. [`VideoEncoder`] (Linux) runs them on VA-API, and
//!   [`VaapiEncoderBackend`] is the host session's encoder backend on it:
//!   `reims_vgpu_video_core::encoder_backend::EncoderBackend`.
//! - [`VaapiJpegDecoder`], Linux's, is the JPEG engine's decoder on
//!   VA-API's baseline JPEG decoder:
//!   `reims_vgpu_video_core::jpeg::JpegDecoder`, over the codec crate's
//!   [`jpeg`] parser.
//! - [`open_backend`], Linux's, is all three on one render node as core's
//!   `Backend`, which is what the video device and the glue take.

// The crate doc names this crate's Linux items, which other OSes have none of: its links
// resolve, and are checked, on Linux.
#![cfg_attr(not(target_os = "linux"), allow(rustdoc::broken_intra_doc_links))]

pub use reims_vgpu_video_codec::av1;
pub use reims_vgpu_video_codec::encode;
pub use reims_vgpu_video_codec::h264;
pub use reims_vgpu_video_codec::hevc;
pub use reims_vgpu_video_codec::jpeg;
pub use reims_vgpu_video_codec::vp9;

pub use h264::OutputOrder;

#[cfg(target_os = "linux")]
mod va;

#[cfg(target_os = "linux")]
pub use va::{
    AV1_MAIN, Av1Decoder, Capabilities, ConfigCapabilities, Display, Entrypoint, Error, Fourcc,
    Frame, H264Decoder, HevcDecoder, LoadError, Profile, ReadBack, ReadbackTimes, ReadbackWay,
    SurfaceCapabilities, VP9_PROFILE0, VP9_PROFILE2, VaapiBackend, VaapiJpegDecoder, Vp9Decoder,
    proc_filter_name, read_back_by, rt_format_names,
};

#[cfg(target_os = "linux")]
pub use va::{
    EncoderCapabilities, EncoderSupport, Feature, HevcBlockSizes, HevcFeatures, UploadMethod,
    VaapiEncoderBackend, VideoEncoder, encoder_support,
};

#[cfg(target_os = "linux")]
pub use va::{NAME, backend_on, open_backend};
