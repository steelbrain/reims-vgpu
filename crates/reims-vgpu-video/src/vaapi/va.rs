//! VA-API through libva, with a binding of our own ([`ffi`]), and the
//! decoders on it. Linux only: lib.rs leaves this module out elsewhere.
//! libva is loaded at run time, not linked ([`libva`]), so a host without
//! it runs, and opens no display.
//!
//! Every `unsafe` block of the crate is here, each with the reason it is
//! sound, which the lints below hold it to; the binding's layouts and
//! constants are checked against libva's headers by `ffi`'s ignored tests.

#![deny(clippy::undocumented_unsafe_blocks, clippy::missing_safety_doc)]

mod buffer;
mod calls;
mod caps;
mod display;
mod encoder;
mod export;
mod fence;
mod ffi;
mod frame;
mod h264;
mod hevc;
mod host_memory;
mod image;
mod import_check;
mod libva;
mod readback;
#[cfg(test)]
mod spin;
mod surface;
mod vpp_ffi;
mod wait;
mod zero_copy;

pub use caps::{
    Capabilities, ConfigCapabilities, Entrypoint, Fourcc, Profile, SurfaceCapabilities,
    proc_filter_name, rt_format_names,
};
pub use display::{Display, Error};
pub use encoder::{
    EncoderCapabilities, EncoderSupport, Feature, HevcBlockSizes, HevcFeatures, UploadMethod,
    VideoEncoder, encoder_support,
};
pub use export::{ExportedPlane, SurfaceExport};
pub use frame::{Frame, FrameHold, GuestWrites, PlaneRows, ReadBack};
pub use h264::H264Decoder;
pub use hevc::HevcDecoder;
pub use import_check::{ImportWrite, ImportWrites};
pub use libva::LoadError;
pub use readback::{ReadbackTimes, ReadbackWay, read_back_by};
