//! NVDEC: libnvcuvid's decoder, loaded at run time, and what it decodes.
//!
//! Only the decoder half of libnvcuvid is used. The bitstream is parsed,
//! and the DPB kept, by our own codec code; NVDEC gets each picture's
//! parameters as `CUVIDPICPARAMS`, as VA-API gets its buffers.

pub(crate) mod decoder;
pub(crate) mod ffi;
pub(crate) mod h264;
pub(crate) mod hevc;

use std::sync::OnceLock;

use crate::core::sync::Mutex;
use crate::nvidia::cuda::Gpu;
use crate::nvidia::dl::Library;
use crate::nvidia::error::Error;
use ffi::{NvdecFunctions, Zeroed};

/// libnvcuvid, loaded once per process.
pub(crate) struct Nvdec {
    pub(crate) f: NvdecFunctions,
    _library: Library,
}

static NVDEC: OnceLock<Result<Nvdec, Error>> = OnceLock::new();

impl Nvdec {
    /// libnvcuvid, loading it on the first call.
    pub(crate) fn get() -> Result<&'static Self, Error> {
        NVDEC
            .get_or_init(|| {
                let library = Library::open(c"libnvcuvid.so.1")?;
                Ok(Self {
                    f: NvdecFunctions::load(&library)?,
                    _library: library,
                })
            })
            .as_ref()
            .map_err(Clone::clone)
    }
}

/// A codec NVDEC is asked about.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NvdecCodec {
    /// H.264.
    H264,
    /// HEVC.
    Hevc,
}

impl NvdecCodec {
    /// Every codec, in `cudaVideoCodec`'s order.
    pub const ALL: [Self; 2] = [Self::H264, Self::Hevc];

    pub(crate) fn raw(self) -> ffi::cudaVideoCodec {
        match self {
            Self::H264 => ffi::cudaVideoCodec_H264,
            Self::Hevc => ffi::cudaVideoCodec_HEVC,
        }
    }
}

/// A chroma format NVDEC is asked about.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ChromaFormat {
    /// Luma only.
    Monochrome,
    /// 4:2:0.
    Yuv420,
    /// 4:2:2.
    Yuv422,
    /// 4:4:4.
    Yuv444,
}

impl ChromaFormat {
    /// Every format, in `cudaVideoChromaFormat`'s order.
    pub const ALL: [Self; 4] = [Self::Monochrome, Self::Yuv420, Self::Yuv422, Self::Yuv444];

    pub(crate) fn raw(self) -> ffi::cudaVideoChromaFormat {
        match self {
            Self::Monochrome => ffi::cudaVideoChromaFormat_Monochrome,
            Self::Yuv420 => ffi::cudaVideoChromaFormat_420,
            Self::Yuv422 => ffi::cudaVideoChromaFormat_422,
            Self::Yuv444 => ffi::cudaVideoChromaFormat_444,
        }
    }
}

/// A surface format NVDEC can output.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SurfaceFormat {
    /// 8-bit 4:2:0, Y then interleaved CbCr.
    Nv12,
    /// 16-bit 4:2:0 (10 and 12 bits in the high bits), as P010.
    P016,
    /// 8-bit 4:4:4, planar.
    Yuv444,
    /// 16-bit 4:4:4, planar.
    Yuv444_16Bit,
    /// 8-bit 4:2:2, Y then interleaved CbCr.
    Nv16,
    /// 16-bit 4:2:2.
    P216,
}

impl SurfaceFormat {
    /// The formats `nOutputFormatMask`'s bits name, bit by bit.
    const BY_BIT: [(ffi::cudaVideoSurfaceFormat, Self); 6] = [
        (ffi::cudaVideoSurfaceFormat_NV12, Self::Nv12),
        (ffi::cudaVideoSurfaceFormat_P016, Self::P016),
        (ffi::cudaVideoSurfaceFormat_YUV444, Self::Yuv444),
        (ffi::cudaVideoSurfaceFormat_YUV444_16Bit, Self::Yuv444_16Bit),
        (ffi::cudaVideoSurfaceFormat_NV16, Self::Nv16),
        (ffi::cudaVideoSurfaceFormat_P216, Self::P216),
    ];

    pub(crate) fn raw(self) -> ffi::cudaVideoSurfaceFormat {
        Self::BY_BIT
            .iter()
            .find(|(_, format)| *format == self)
            .map_or(ffi::cudaVideoSurfaceFormat_NV12, |(raw, _)| *raw)
    }
}

/// What NVDEC answers for one codec, chroma format and bit depth
/// (`cuvidGetDecoderCaps`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodeCapabilities {
    /// The codec asked about.
    pub codec: NvdecCodec,
    /// The chroma format asked about.
    pub chroma: ChromaFormat,
    /// The bit depth asked about.
    pub bit_depth: u32,
    /// `bIsSupported`.
    pub supported: bool,
    /// `nNumNVDECs`: the decode engines.
    pub engines: u8,
    /// The output formats `nOutputFormatMask` lists.
    pub output_formats: Vec<SurfaceFormat>,
    /// `nMaxWidth`.
    pub max_width: u32,
    /// `nMaxHeight`.
    pub max_height: u32,
    /// `nMaxMBCount`: the most 16x16 blocks a picture may have.
    pub max_macroblocks: u32,
    /// `nMinWidth`.
    pub min_width: u32,
    /// `nMinHeight`.
    pub min_height: u32,
}

/// What NVDEC answered each GPU, by its PCI address, for each codec, chroma
/// format and bit depth asked ([`decode_capabilities`]).
static CAPABILITIES: Mutex<Vec<(String, DecodeCapabilities)>> = Mutex::new(Vec::new());

/// What NVDEC on `gpu` decodes of `codec` at `chroma` and `bit_depth`:
/// asked once a GPU, codec, chroma format and bit depth, and the answer kept
/// for the process; a query that failed is asked again where it is next
/// needed. The graphics device's NVDEC arm asks as it opens, and each
/// session's first decoder of a codec and bit depth asks again
/// (`decoder::Decoder::new`), so the kept answer spares that session the
/// query: `cuvidGetDecoderCaps` took 6-9 ms on an RTX 3090 (driver
/// 620.70.06, CachyOS), and 15-19 ms beside two running decoders or a busy
/// loop on each of 16 CPUs (**measured**, 2026-11-14, the session-start
/// probe, `crate::probes`).
pub fn decode_capabilities(
    gpu: &Gpu,
    codec: NvdecCodec,
    chroma: ChromaFormat,
    bit_depth: u32,
) -> Result<DecodeCapabilities, Error> {
    let on = &gpu.info().pci_bus_id;
    let same = |(at, caps): &&(String, DecodeCapabilities)| {
        at == on && (caps.codec, caps.chroma, caps.bit_depth) == (codec, chroma, bit_depth)
    };
    if let Some((_, caps)) = CAPABILITIES.lock().iter().find(same) {
        return Ok(caps.clone());
    }
    let caps = ask_capabilities(gpu, codec, chroma, bit_depth)?;
    CAPABILITIES.lock().push((on.clone(), caps.clone()));
    Ok(caps)
}

/// NVDEC's answer on `gpu` for `codec` at `chroma` and `bit_depth`, asked
/// now (`cuvidGetDecoderCaps`).
fn ask_capabilities(
    gpu: &Gpu,
    codec: NvdecCodec,
    chroma: ChromaFormat,
    bit_depth: u32,
) -> Result<DecodeCapabilities, Error> {
    let nvdec = Nvdec::get()?;
    let mut caps = ffi::CUVIDDECODECAPS::zeroed();
    caps.eCodecType = codec.raw();
    caps.eChromaFormat = chroma.raw();
    caps.nBitDepthMinus8 = bit_depth.saturating_sub(8);
    let current = gpu.push()?;
    // SAFETY: the struct is live and zeroed but for its three inputs; the
    // GPU's context is current, as the call needs.
    let result = unsafe { (nvdec.f.cuvidGetDecoderCaps)(&raw mut caps) };
    drop(current);
    gpu.cuda().check("cuvidGetDecoderCaps", result)?;
    let output_formats = SurfaceFormat::BY_BIT
        .iter()
        .filter(|(bit, _)| caps.nOutputFormatMask & (1 << bit) != 0)
        .map(|(_, format)| *format)
        .collect();
    Ok(DecodeCapabilities {
        codec,
        chroma,
        bit_depth,
        supported: caps.bIsSupported != 0,
        engines: caps.nNumNVDECs,
        output_formats,
        max_width: caps.nMaxWidth,
        max_height: caps.nMaxHeight,
        max_macroblocks: caps.nMaxMBCount,
        min_width: caps.nMinWidth.into(),
        min_height: caps.nMinHeight.into(),
    })
}

#[cfg(test)]
mod tests {
    use super::{ChromaFormat, NvdecCodec, decode_capabilities};
    use crate::nvidia::cuda::{Gpu, GpuSelector};
    use crate::probes::timeline;

    /// A GPU's capabilities are asked of NVDEC once a codec, chroma format and bit depth: the
    /// second ask answers as the first did, and makes no `cuvidGetDecoderCaps`.
    #[test]
    #[ignore = "needs an NVIDIA GPU: run on the NVIDIA host"]
    fn capabilities_are_asked_once_a_codec_chroma_format_and_bit_depth() {
        let gpu = Gpu::open(&GpuSelector::First).expect("an NVIDIA GPU");
        let ask = || decode_capabilities(&gpu, NvdecCodec::Hevc, ChromaFormat::Yuv420, 10);
        let first = ask().expect("NVDEC's answer");
        let (second, span) = timeline::record(ask);
        assert_eq!(second.expect("the kept answer"), first);
        let calls: Vec<&str> = span.steps.iter().map(|(call, _)| *call).collect();
        assert!(
            !calls.contains(&"cuvidGetDecoderCaps"),
            "asked again: {calls:?}"
        );
    }
}
