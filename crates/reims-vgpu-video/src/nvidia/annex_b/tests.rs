//! An [`AnnexBDecoder`] on the NVIDIA host: NVENC's access units decoded
//! in place, each picture written by the GPU into memory registered with
//! CUDA, as it reads back.

use std::alloc::Layout;
use std::ptr::NonNull;
use std::sync::Arc;

use crate::core::device_memory::{DeviceMemory, HostView, MemoryIdentity, MemoryLife};
use crate::core::runtime::Runtime;
use crate::core::surfaces::SurfaceError;

use super::*;
use crate::nvidia::cuda::GpuSelector;
use crate::nvidia::nvenc::encoder::tests::{noise, params};
use crate::nvidia::nvenc::encoder::{EncodeParams, EncodeProfile, NvencEncoder, RawFrame};
use crate::vaapi::gated::GatedRuntime;

/// The alignment of the test's memory: whole pages on every host's page size.
const PAGE: usize = 64 << 10;

/// What the GPU's writes must leave as it was: the bytes past each row.
const UNTOUCHED: u8 = 0xee;

/// Whole pages of host memory, as a device's view of a surface's pages.
struct Pages {
    first: NonNull<u8>,
    len: usize,
}

// SAFETY: the memory is the test's own allocation, which any thread may
// reach; the test reads it only once the GPU's writes into it are done.
unsafe impl Send for Pages {}
// SAFETY: as above.
unsafe impl Sync for Pages {}

impl Pages {
    fn layout(len: usize) -> Layout {
        Layout::from_size_align(len, PAGE).expect("a layout")
    }

    fn new(len: usize) -> Arc<Self> {
        // SAFETY: the layout's size is not zero.
        let at = unsafe { std::alloc::alloc(Self::layout(len)) };
        Arc::new(Self {
            first: NonNull::new(at).expect("the test's memory"),
            len,
        })
    }

    /// Sets every byte to `byte`, with no write of the GPU's in flight.
    fn fill(&self, byte: u8) {
        // SAFETY: `len` bytes allocated from `first`, which nothing borrows
        // and the GPU does not write meanwhile.
        unsafe { std::ptr::write_bytes(self.first.as_ptr(), byte, self.len) };
    }

    /// Its bytes, which nothing writes while they are borrowed: the GPU's
    /// writes are done when `write_into` returns.
    fn bytes(&self) -> &[u8] {
        // SAFETY: `len` bytes allocated from `first`, initialised by `fill`
        // first, with no write in flight while borrowed.
        unsafe { std::slice::from_raw_parts(self.first.as_ptr(), self.len) }
    }
}

impl Drop for Pages {
    fn drop(&mut self) {
        // SAFETY: allocated by `new` with this layout, freed once, after
        // the registration that held the view went.
        unsafe { std::alloc::dealloc(self.first.as_ptr(), Self::layout(self.len)) };
    }
}

/// The pages as a device takes them: their identity lives as long as this.
struct PagesMemory {
    life: MemoryLife,
    pages: Arc<Pages>,
}

impl DeviceMemory for PagesMemory {
    fn identity(&self) -> MemoryIdentity {
        self.life.identity()
    }

    fn length(&self) -> usize {
        self.pages.len
    }

    fn host_view(&self) -> Result<HostView, SurfaceError> {
        Ok(HostView {
            first: self.pages.first,
            size: self.pages.len,
            offset: 0,
            hold: Box::new(Arc::clone(&self.pages)),
        })
    }
}

/// Each picture of NVENC's H.264 and HEVC at 1080p, its access units with
/// the parameter sets before each IDR picture, reaches NVDEC in place, and
/// lands by the GPU in registered memory at a pitch of its own, its rows
/// alone, as the picture reads back.
#[test]
#[ignore = "needs an NVIDIA GPU: run on the NVIDIA host"]
fn access_units_decode_in_place_and_land_by_the_gpu_as_they_read_back() {
    let gpu = Arc::new(Gpu::open(&GpuSelector::First).expect("an NVIDIA GPU"));
    let runtime: Arc<dyn Runtime> = Arc::new(GatedRuntime);
    let (width, height, pitch) = (1920u32, 1080u32, 2048usize);
    let rows = usize::try_from(height).expect("rows");
    let chroma_at = pitch * rows;
    let len = (chroma_at + pitch * rows / 2).next_multiple_of(PAGE);
    for (profile, codec) in [
        (EncodeProfile::H264High, AnnexBCodec::H264),
        (EncodeProfile::HevcMain, AnnexBCodec::Hevc),
    ] {
        let params = EncodeParams {
            width,
            height,
            b_frames: 0,
            ..params(profile)
        };
        let mut encoder = NvencEncoder::new(Arc::clone(&gpu), &params).expect("an NVENC session");
        let mut decoder = AnnexBDecoder::new(&gpu, codec, &runtime);
        decoder
            .parameter_sets(encoder.sequence_parameters())
            .expect("the parameter sets");
        let memory = PagesMemory {
            life: MemoryLife::new(),
            pages: Pages::new(len),
        };
        let mut pictures = 0;
        for n in 0..8 {
            let (luma, chroma, row) = noise(&params, n);
            let packets = encoder
                .encode(&RawFrame {
                    luma: (&luma, row),
                    chroma: (&chroma, row),
                    index: n,
                    force_idr: n == 0,
                })
                .expect("a frame NVENC takes");
            for packet in packets {
                let packet = packet.expect("a packet");
                let picture = (decoder.decode(&packet.data))
                    .expect("a decode")
                    .expect("a picture");
                memory.pages.fill(UNTOUCHED);
                let plane = |offset, rows| MemoryPlane {
                    memory: &memory,
                    offset,
                    bytes_per_row: pitch,
                    row_bytes: 1920,
                    rows,
                };
                // SAFETY: the planes lie apart in the test's memory, which
                // nothing else writes until the call returns.
                let written = unsafe {
                    picture.write_into(
                        (width, height),
                        plane(0, height),
                        plane(chroma_at, height / 2),
                        Box::new(()),
                    )
                };
                written.expect("the decode").expect("the GPU's write");
                let read = picture.read_back((width, height)).expect("a read back");
                let bytes = memory.pages.bytes();
                for (read, at) in read.planes().iter().zip([0, chroma_at]) {
                    assert_eq!(read.bytes_per_row, 1920);
                    for row in 0..usize::try_from(read.rows).expect("rows") {
                        let written = &bytes[at + row * pitch..][..pitch];
                        assert_eq!(
                            &written[..1920],
                            &read.bytes[row * 1920..][..1920],
                            "{codec:?} picture {n}, row {row} of the plane at {at}"
                        );
                        assert!(written[1920..].iter().all(|&b| b == UNTOUCHED));
                    }
                }
                pictures += 1;
            }
        }
        assert_eq!(pictures, 8);
        assert_eq!(
            decoder.bitstreams(),
            Bitstreams {
                in_place: 8,
                gathered: 0
            },
            "{codec:?}"
        );
    }
}

/// An H.264 and an HEVC stream of NVENC's started at a P picture, as a session started or a seek
/// at one: each picture before the stream's next IDR picture is refused with one picture's error
/// ([`Error::is_picture_error`]), and the decoder decodes from the IDR picture on, each picture
/// read back whole.
#[test]
#[ignore = "needs an NVIDIA GPU: run on the NVIDIA host"]
fn a_stream_started_at_a_p_picture_answers_picture_errors_until_its_idr_picture() {
    let gpu = Arc::new(Gpu::open(&GpuSelector::First).expect("an NVIDIA GPU"));
    let runtime: Arc<dyn Runtime> = Arc::new(GatedRuntime);
    for (profile, codec) in [
        (EncodeProfile::H264High, AnnexBCodec::H264),
        (EncodeProfile::HevcMain, AnnexBCodec::Hevc),
    ] {
        let params = EncodeParams {
            width: 1920,
            height: 1080,
            b_frames: 0,
            ..params(profile)
        };
        let mut encoder = NvencEncoder::new(Arc::clone(&gpu), &params).expect("an NVENC session");
        let mut units = Vec::new();
        for n in 0..8 {
            let (luma, chroma, row) = noise(&params, n);
            let packets = encoder
                .encode(&RawFrame {
                    luma: (&luma, row),
                    chroma: (&chroma, row),
                    index: n,
                    force_idr: n == 0 || n == 5,
                })
                .expect("a frame NVENC takes");
            for packet in packets {
                units.push(packet.expect("a packet").data);
            }
        }
        assert_eq!(units.len(), 8, "{codec:?}");
        let mut decoder = AnnexBDecoder::new(&gpu, codec, &runtime);
        decoder
            .parameter_sets(encoder.sequence_parameters())
            .expect("the parameter sets");
        for (n, unit) in units.iter().enumerate().skip(2) {
            let decoded = decoder.decode(unit);
            if n < 5 {
                let error = decoded
                    .err()
                    .expect("a P picture with no reference is refused");
                assert!(error.is_picture_error(), "{codec:?} picture {n}: {error}");
                continue;
            }
            let picture = decoded.expect("a decode").expect("a picture");
            let read = picture
                .read_back((1920, 1080))
                .map(|read| read.planes()[0].rows);
            assert_eq!(read.ok(), Some(1080), "{codec:?} picture {n}");
        }
    }
}

/// An H.264 and an HEVC P picture whose slice is cut to half, which NVDEC answers failed: read
/// back as NVDEC left it, telling its status, with the pictures after it decoded, where the
/// decoder copies failed pictures; failed where it does not, an HEVC stream's next picture then
/// refused until an IRAP picture, and an H.264 stream's pictures after it decoded on from it, as
/// after any picture that does not decode (the codec module's H.264 decoder).
#[test]
#[ignore = "needs an NVIDIA GPU: run on the NVIDIA host"]
fn a_failed_picture_is_copied_and_decoded_on_from_or_refused() {
    let gpu = Arc::new(Gpu::open(&GpuSelector::First).expect("an NVIDIA GPU"));
    let runtime: Arc<dyn Runtime> = Arc::new(GatedRuntime);
    for (profile, codec) in [
        (EncodeProfile::H264High, AnnexBCodec::H264),
        (EncodeProfile::HevcMain, AnnexBCodec::Hevc),
    ] {
        let params = EncodeParams {
            width: 1920,
            height: 1080,
            b_frames: 0,
            ..params(profile)
        };
        let mut encoder = NvencEncoder::new(Arc::clone(&gpu), &params).expect("an NVENC session");
        let mut units = Vec::new();
        for n in 0..8 {
            let (luma, chroma, row) = noise(&params, n);
            let packets = encoder
                .encode(&RawFrame {
                    luma: (&luma, row),
                    chroma: (&chroma, row),
                    index: n,
                    force_idr: n == 0,
                })
                .expect("a frame NVENC takes");
            for packet in packets {
                units.push(packet.expect("a packet").data);
            }
        }
        assert_eq!(units.len(), 8, "{codec:?}");
        // The fourth picture, a P picture, its slice cut to half.
        let cut = units[3].len() / 2;
        units[3].truncate(cut);
        for copies in [true, false] {
            let mut decoder = AnnexBDecoder::new(&gpu, codec, &runtime);
            if copies {
                decoder.copy_failed_pictures();
            }
            decoder
                .parameter_sets(encoder.sequence_parameters())
                .expect("the parameter sets");
            for (n, unit) in units.iter().enumerate() {
                let decoded = decoder.decode(unit);
                if !copies && n == 4 && matches!(codec, AnnexBCodec::Hevc) {
                    assert!(
                        decoded.is_err(),
                        "{codec:?}: the picture after the failed one is refused"
                    );
                    break;
                }
                let picture = decoded.expect("a decode").expect("a picture");
                let read = picture
                    .read_back((1920, 1080))
                    .map(|read| read.planes()[0].rows);
                match (n, copies) {
                    (3, true) => {
                        assert_eq!(read.ok(), Some(1080), "{codec:?}: copied as NVDEC left it");
                        let status = picture.copied_failed();
                        assert!(
                            matches!(status, Some(8 | 9)),
                            "{codec:?}: its status: {status:?}"
                        );
                    }
                    (3, false) => {
                        assert!(
                            matches!(read, Err(Error::DecodeFailed(_))),
                            "{codec:?}: {read:?}"
                        );
                        assert_eq!(picture.copied_failed(), None);
                    }
                    // The pictures after it decode from it, as NVDEC left it.
                    _ => assert_eq!(
                        read.ok(),
                        Some(1080),
                        "{codec:?} picture {n}, copies {copies}"
                    ),
                }
            }
        }
    }
}

/// An H.264 P picture whose sample does not decode (its first NAL unit's `forbidden_zero_bit`
/// set) is one picture's error, and each picture after it comes out and reads back whole, decoded
/// on from the last reference's surface standing in for the lost frame (the codec module's
/// `Accelerator::conceal`), as a Mac's hardware decoder answers the pictures after a damaged one
/// with no error and an image (held85's damage run, potluck job m3-max-60715). As reims-vgpu's
/// arm makes it, the decoder copies failed pictures.
#[test]
#[ignore = "needs an NVIDIA GPU: run on the NVIDIA host"]
fn the_pictures_after_one_that_does_not_decode_come_out() {
    let gpu = Arc::new(Gpu::open(&GpuSelector::First).expect("an NVIDIA GPU"));
    let runtime: Arc<dyn Runtime> = Arc::new(GatedRuntime);
    let params = EncodeParams {
        width: 1920,
        height: 1080,
        b_frames: 0,
        ..params(EncodeProfile::H264High)
    };
    let mut encoder = NvencEncoder::new(Arc::clone(&gpu), &params).expect("an NVENC session");
    let mut units = Vec::new();
    for n in 0..8 {
        let (luma, chroma, row) = noise(&params, n);
        let packets = encoder
            .encode(&RawFrame {
                luma: (&luma, row),
                chroma: (&chroma, row),
                index: n,
                force_idr: n == 0,
            })
            .expect("a frame NVENC takes");
        for packet in packets {
            units.push(packet.expect("a packet").data);
        }
    }
    assert_eq!(units.len(), 8);
    // The fourth picture, a P picture: its first NAL unit's header, past its start code.
    let header = (units[3].windows(3))
        .position(|w| w == [0, 0, 1])
        .expect("an Annex B start code")
        + 3;
    units[3][header] |= 0x80;
    let mut decoder = AnnexBDecoder::new(&gpu, AnnexBCodec::H264, &runtime);
    decoder.copy_failed_pictures();
    decoder
        .parameter_sets(encoder.sequence_parameters())
        .expect("the parameter sets");
    for (n, unit) in units.iter().enumerate() {
        let decoded = decoder.decode(unit);
        if n == 3 {
            let error = decoded.err().expect("the damaged picture does not decode");
            assert!(error.is_picture_error(), "{error}");
            continue;
        }
        let picture = decoded
            .unwrap_or_else(|error| panic!("picture {n}: {error}"))
            .expect("a picture");
        let read = picture
            .read_back((1920, 1080))
            .map(|read| read.planes()[0].rows);
        assert_eq!(read.ok(), Some(1080), "picture {n}");
    }
}
