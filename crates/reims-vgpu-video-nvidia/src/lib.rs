//! The NVIDIA backend: decode on NVDEC and encode on NVENC, over the CUDA
//! driver API, with nothing above them.
//!
//! The driver's libraries (libcuda, libnvcuvid, libnvidia-encode) are
//! loaded at run time, so that a host with no NVIDIA driver runs without
//! them: opening a GPU there fails, and nothing else does. Their
//! declarations are ours, checked against NVIDIA's headers on the test host
//! (`layout`'s test).
//!
//! - [`Gpu`] is a GPU and its CUDA context; [`devices`] lists the host's.
//! - [`decode_capabilities`] and [`encoder_report`] say what NVDEC and
//!   NVENC answer.
//! - [`NvencEncoder`] encodes raw frames to H.264 or HEVC.
//! - [`AnnexBDecoder`] decodes whole Annex B access units, one picture
//!   each, for a device that says where each picture goes, as the
//!   reims-graphics device's video does.
//!
//! Every `unsafe` block holds one operation and says why it is sound, and
//! every `unsafe fn` its contract (the lints below).

#![deny(
    clippy::undocumented_unsafe_blocks,
    clippy::multiple_unsafe_ops_per_block,
    clippy::missing_safety_doc
)]
// The reims-graphics device's NVIDIA video decodes on NVDEC's decoder, its accelerators, CUDA's
// copies, the bounded waits and zero copy (`annex_b`); AV1's bindings, CUDA's device memory and
// some helpers of the decoders are not called.
#![expect(dead_code, reason = "AV1's bindings and helpers nothing calls")]

mod annex_b;
mod cuda;
mod decode;
mod dl;
mod error;
#[cfg(test)]
mod layout;
mod nvdec;
mod nvenc;
mod probe;
mod wait;
mod zero_copy;

pub use annex_b::{
    AnnexBCodec, AnnexBDecoder, AnnexBPicture, Bitstreams, MemoryPlane, ReadBack, ReadPlane,
};
pub use cuda::{DeviceInfo, Gpu, GpuSelector};
pub use dl::LoadError;
pub use error::Error;
pub use nvdec::{ChromaFormat, DecodeCapabilities, NvdecCodec, SurfaceFormat, decode_capabilities};
pub use nvenc::encoder::{
    Bitstream, EncodeParams, EncodeProfile, EncodeProfileSupport, EncodeRateControl, NvencEncoder,
    Packet, PictureType, RawFrame, Tuning, encode_profiles,
};
pub use nvenc::max_api_version;
pub use probe::{EncodeCodecReport, EncoderReport, encoder_report};

/// `value` as a `usize`, which holds every `u32` on the 64-bit hosts the
/// crate runs on (asserted below).
pub(crate) fn usize_of(value: u32) -> usize {
    usize::try_from(value).unwrap_or(usize::MAX)
}

const _: () = assert!(usize::BITS >= u32::BITS);

/// The host's NVIDIA GPUs, as CUDA lists them; an error where the host has
/// no NVIDIA driver.
pub fn devices() -> Result<Vec<DeviceInfo>, Error> {
    cuda::Cuda::get()?.devices()
}
