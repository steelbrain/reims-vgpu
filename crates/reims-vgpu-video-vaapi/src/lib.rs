//! The VA-API backend: decode, and encode where the driver has it, on the
//! host's GPU through libva alone, with no FFmpeg or other layer between.
//!
//! - [`h264`] and [`hevc`] are H.264's and HEVC's syntax and decoding
//!   process above the slice data: plain Rust, on every host.
//! - The libva binding and the decoders on it ([`Display`],
//!   [`H264Decoder`], [`HevcDecoder`], [`Frame`]) are Linux's alone, and
//!   absent elsewhere.
//! - [`encode`] is the encoders' portable half: their settings and
//!   samples, the H.264 and HEVC headers they write, and the planning of
//!   each picture. [`VideoEncoder`] (Linux) runs them on VA-API.

// The crate doc names this crate's Linux items, which other OSes have none of: its links
// resolve, and are checked, on Linux.
#![cfg_attr(not(target_os = "linux"), allow(rustdoc::broken_intra_doc_links))]

pub use reims_vgpu_video_codec::encode;
pub use reims_vgpu_video_codec::h264;
pub use reims_vgpu_video_codec::hevc;

pub use h264::OutputOrder;

#[cfg(target_os = "linux")]
mod va;

#[cfg(target_os = "linux")]
pub use va::{
    Capabilities, ConfigCapabilities, Display, Entrypoint, Error, Fourcc, Frame, GuestWrites,
    H264Decoder, HevcDecoder, ImportWrite, ImportWrites, LoadError, PlaneRows, Profile, ReadBack,
    ReadbackTimes, ReadbackWay, SurfaceCapabilities, proc_filter_name, read_back_by,
    rt_format_names,
};

#[cfg(target_os = "linux")]
pub use va::{
    EncoderCapabilities, EncoderSupport, Feature, HevcBlockSizes, HevcFeatures, UploadMethod,
    VideoEncoder, encoder_support,
};
