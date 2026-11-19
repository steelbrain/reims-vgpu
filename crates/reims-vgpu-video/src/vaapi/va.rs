//! VA-API through libva, with a binding of our own ([`ffi`]), and the
//! decoders on it. Linux only: lib.rs leaves this module out elsewhere.
//! libva is loaded at run time, not linked ([`libva`]), so a host without
//! it runs, and opens no display.
//!
//! Every `unsafe` block of the crate is here, each with the reason it is
//! sound, which the lints below hold it to; the binding's layouts and
//! constants are checked against libva's headers by `ffi`'s ignored tests.

#![deny(clippy::undocumented_unsafe_blocks, clippy::missing_safety_doc)]
// Windows makes no dma-buf: the module's dma-buf half is compiled out there where it names Unix's
// descriptors, and what is left of it, unused there, is let be.
#![cfg_attr(windows, allow(dead_code, unused_imports))]

mod buffer;
mod calls;
mod caps;
#[cfg(windows)]
mod d3d11;
#[cfg(windows)]
mod d3d12;
mod display;
mod encoder;
mod export;
#[cfg(unix)]
mod fence;
mod ffi;
mod frame;
#[cfg(test)]
#[cfg(unix)]
pub(crate) mod gated;
mod h264;
mod hevc;
#[cfg(unix)]
mod host_memory;
mod image;
mod import_check;
mod libva;
mod readback;
#[cfg(test)]
#[cfg(unix)]
mod spin;
mod surface;
mod target;
mod target_check;
#[cfg(test)]
#[cfg(unix)]
pub(crate) mod target_probe;
#[cfg(test)]
mod teardown;
mod vpp_ffi;
mod wait;
mod zero_copy;

pub use caps::{
    Capabilities, ConfigCapabilities, Entrypoint, Fourcc, Profile, SurfaceCapabilities,
    proc_filter_name, rt_format_names,
};
#[cfg(windows)]
pub use d3d11::{PlaneDraw, PlaneTargets};
#[cfg(windows)]
pub use d3d12::SharedTexture;
pub use display::{Display, Error};
pub use encoder::{
    EncoderCapabilities, EncoderSupport, Feature, HevcBlockSizes, HevcFeatures, InputSurface,
    UploadMethod, VideoEncoder, encoder_support,
};
pub use export::{ExportedPlane, SurfaceExport};
pub use frame::{Frame, FrameHold, GuestWrites, PlaneRows, ReadBack};
pub use h264::{H264_MAX_MACROBLOCKS, H264Decoder};
pub use hevc::HevcDecoder;
pub use import_check::{ImportWrite, ImportWrites};
pub use libva::LoadError;
pub use readback::{ReadbackTimes, ReadbackWay, read_back_by};
pub use target::{Target, TargetFor, TargetFormat, TargetPlane, TargetShape};
pub use target_check::{ALIGNMENT as TARGET_ALIGNMENT, TargetChecks, TargetCodec};
