//! Video decode on Vulkan Video (`VK_KHR_video_queue`,
//! `VK_KHR_video_decode_queue`, `VK_KHR_video_decode_h264`,
//! `VK_KHR_video_decode_h265`): H.264 at 8 bits and 4:2:0, progressive,
//! and H.265 Main and Main 10, 4:2:0 at 8 and 10 bits, as the codec
//! crate's decoders take them, each picture landed in an output its caller
//! supplies.
//!
//! - [`VideoDevice`] is the device decoders run on, which need not be
//!   theirs: [`VideoDevice::from_raw`] takes another's instance, device and
//!   video-decode queue, as the product will hand it metal2vulkan's, and
//!   [`VideoDevice::create`] makes one for the tests and tools.
//! - [`h264_capabilities`] and [`h265_capabilities`] say what the device
//!   answers for each profile, which the guest's capability answers come
//!   from.
//! - [`H264Decoder`] and [`H265Decoder`] decode access units into DPB
//!   pictures of their own, and land each picture in a [`Surface`] their
//!   [`SurfaceSource`] gives: an image made as [`SurfaceSpec`] says (NV12,
//!   or P010 at 10 bits), which the decode writes where the device decodes
//!   into an output beside the DPB picture, else one GPU copy out of the
//!   DPB picture does; or two planes in a buffer ([`BufferPlanes`]),
//!   cropped, which one GPU copy writes. In the product each output is the
//!   GPU copy of a guest IOSurface, which the guest recycles once shown,
//!   reference or not. Each picture names the value of the decoder's
//!   timeline semaphore after which its output holds it
//!   ([`Picture::value`]), which graphics work and read backs wait on.
//! - [`DecodeImages`] and [`HostBuffers`] make outputs of their own, and
//!   [`Picture::read_back`] copies an image out, for the tests and tools.
//!
//! - [`H264Encoder`] and [`H265Encoder`] encode frames on an encode queue
//!   attached to the device ([`VideoDevice::attach_encode`]): sources read
//!   in place ([`SourceSpec`]) or through one GPU copy out of a buffer, the
//!   bitstream written straight into a buffer their [`BitstreamSource`]
//!   gives, host pages imported by their pointer among them. What each
//!   profile encodes is [`h264_encode_capabilities`] and
//!   [`h265_encode_capabilities`].
//!
//! Every `unsafe` block holds one operation and says why it is sound, and
//! every `unsafe fn` its contract (the lints below).

#![deny(
    clippy::undocumented_unsafe_blocks,
    clippy::multiple_unsafe_ops_per_block,
    clippy::missing_safety_doc
)]

mod device;
mod encode;
mod error;
mod h264;
mod h265;
mod import;
mod lent;
mod params;
mod picture;
mod profile;
mod session;
mod std_types;
mod surface;
mod timeline;

pub use device::{RawDevice, VideoDevice};
pub use encode::caps::{
    EncodeCapabilities, EncodeCodec, EncodeProfile, QualityPreference, Tuning,
    h264_encode_capabilities, h265_encode_capabilities,
};
pub use encode::device::{ENCODE_EXTENSIONS, RawEncodeQueue};
pub use encode::frame::{EncodeStatus, EncodedFrame, Feedback, GpuTime};
pub use encode::io::{
    BitstreamBuffer, BitstreamSource, BitstreamSpec, BitstreamTarget, EncodeSource, HostBitstream,
    HostBitstreams, ImportedBitstream, ImportedSource, SourcePicture, SourceSpec,
};
pub use encode::plan::{GopConfig, MAX_B_FRAMES, PictureKind};
pub use encode::rate::{DEFAULT_BUFFER_MS, DEFAULT_INITIAL_MS, RateControl, Vbv};
pub use encode::{
    EncodeConfig, FrameParams, H264Encoder, H265Encoder, ParameterSetMode, SourceCounts,
};
pub use error::Error;
pub use h264::{Frame, H264Decoder};
pub use h265::{H265Decoder, H265Frame};
pub use import::{HostImport, ImportedPlanes, SampleMemory};
pub use lent::LentQueue;
pub use picture::{Nv12, Picture};
pub use profile::{
    Capabilities, H264Capabilities, H264Profile, H265Capabilities, H265Profile, OUTPUT_FLAGS,
    OUTPUT_USAGE, P010_FORMAT, PICTURE_FORMAT, VideoProfile, h264_capabilities, h265_capabilities,
};
pub use reims_vgpu_video_codec::h264::{OutputOrder, avcc, nal};
/// The codec crate's HEVC syntax and decoding process: its `hvcC` record,
/// its NAL units and its errors, as the H.265 decoder's callers read them.
pub use reims_vgpu_video_codec::hevc;
pub use session::BitstreamCounts;
pub use surface::{
    BufferPlanes, DecodeImage, DecodeImages, HostBuffer, HostBuffers, Landing, Surface,
    SurfaceSource, SurfaceSpec,
};
