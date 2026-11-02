//! Video decode on Vulkan Video (`VK_KHR_video_queue`,
//! `VK_KHR_video_decode_queue`, `VK_KHR_video_decode_h264`): H.264 at 8
//! bits and 4:2:0, progressive, as the codec crate's decoder takes it,
//! decoded into images its caller supplies.
//!
//! - [`VideoDevice`] is the device decoders run on, which need not be
//!   theirs: [`VideoDevice::from_raw`] takes another's instance, device and
//!   video-decode queue, as the product will hand it metal2vulkan's, and
//!   [`VideoDevice::create`] makes one for the tests and tools.
//! - [`h264_capabilities`] says what the device answers for each H.264
//!   profile, which the guest's capability answers come from.
//! - [`H264Decoder`] decodes access units, each picture into a
//!   [`SurfaceImage`] its [`SurfaceSource`] gives, which is also the
//!   picture's DPB slot: the decoder never copies a picture. In the
//!   product each surface is the GPU copy of a guest IOSurface. Each
//!   decode signals the decoder's timeline semaphore
//!   ([`Picture::value`]), which graphics work and read backs wait on.
//! - [`DecodeImages`] makes surfaces of its own, and
//!   [`Picture::read_back`] copies one out, for the tests and tools.
//!
//! Every `unsafe` block holds one operation and says why it is sound, and
//! every `unsafe fn` its contract (the lints below).

#![deny(
    clippy::undocumented_unsafe_blocks,
    clippy::multiple_unsafe_ops_per_block,
    clippy::missing_safety_doc
)]

mod device;
mod error;
mod h264;
mod picture;
mod profile;
mod std_types;
mod surface;
mod timeline;

pub use device::{RawDevice, VideoDevice};
pub use error::Error;
pub use h264::{Frame, H264Decoder};
pub use picture::{Nv12, Picture};
pub use profile::{
    H264Capabilities, H264Profile, PICTURE_FORMAT, SURFACE_FLAGS, SURFACE_USAGE, h264_capabilities,
};
pub use reims_vgpu_video_codec::h264::{OutputOrder, avcc, nal};
pub use surface::{DecodeImage, DecodeImages, SurfaceImage, SurfaceSource};
