//! An [`AnnexBDecoder`] on the NVIDIA host: NVENC's access units decoded
//! in place, each picture written by the GPU into memory registered with
//! CUDA, as it reads back.

use std::alloc::Layout;
use std::ptr::NonNull;
use std::sync::Arc;

use reims_vgpu_video_core::device_memory::{DeviceMemory, HostView, MemoryIdentity, MemoryLife};
use reims_vgpu_video_core::runtime::{Runtime, StdRuntime};
use reims_vgpu_video_core::surfaces::SurfaceError;

use super::*;
use crate::cuda::GpuSelector;
use crate::nvenc::encoder::tests::{noise, params};
use crate::nvenc::encoder::{EncodeParams, EncodeProfile, NvencEncoder, RawFrame};

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
    let runtime: Arc<dyn Runtime> = StdRuntime::shared();
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
