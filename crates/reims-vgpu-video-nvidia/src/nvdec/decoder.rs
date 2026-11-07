//! One NVDEC decoder: its handle on the GPU, the decode surfaces pictures
//! are decoded into, and the copy of a decoded picture into host memory:
//! into the decoder's staging memory first, with every call into the
//! driver, then by the CPU into the caller's planes
//! ([`Picture::read_back`]).
//!
//! A [`Picture`] holds one decode surface, by its index, until it is
//! dropped: the codec crate's DPB keeps its references so, and a decoded
//! frame keeps its picture until it is copied. A stand-in for a lost frame
//! holds the same surface as the picture it stands in from
//! ([`Picture::stand_in`]), which is free once both have gone. A picture keeps its
//! decoder's handle alive, so a frame can be copied after its decoder was
//! replaced. Calls on one handle are made one at a time, under its lock,
//! with the GPU's context pushed: the decoder is made without NVDEC's
//! context lock (`vidLock`).
//!
//! **Its waits are bounded** (`crate::wait`). A picture's decode is asked
//! after with `cuvidGetDecodeStatus` until it is no longer in progress,
//! before the map, which then finds it done; the map's post-processing and
//! the copies run on the decoder's own stream, which is asked after with
//! `cuStreamQuery`. A frame's waits share one deadline, that of the decode
//! call that gave it (`crate::wait::Deadline`), which a VP9 picture's
//! decode waits until too. Past it the picture fails as stalled
//! ([`Error::Stalled`], the guest's malfunction),
//! and so does every later call on the decoder; the picture's surface, the
//! mapped frame, the decoder's handle and the memory its copies write
//! (the staging memory, or the guest's buffer, its registration and its
//! hold) are kept until the GPU is done ([`UnfinishedPicture`]).
//!
//! Two calls return only once the GPU is done, and are made on the
//! decoder's call thread (`crate::wait`'s `Calls`), their answer waited for
//! until the deadline, with the decoder's lock held for them:
//! - **a VP9 picture's `cuvidDecodePicture`**, which returns once the
//!   picture is decoded, and whose decode NVDEC keeps no status of
//!   (`cuvidGetDecodeStatus` answers an error), measured on the RTX 5080
//!   Laptop GPU, driver 615.71.09, 2026-10-25: 2.87 to 3.04 ms (median) of
//!   a 4K frame, 0.36 ms of a 720p one. It is given a copy of the
//!   bitstream and the slices' offsets, its own for as long as it runs;
//! - **the copies into pageable staging memory**, where `cuMemHostAlloc`
//!   gives no page-locked memory: a `cuMemcpy2DAsync` into pageable memory
//!   returns once the copy is done ([`copy_plane`]).
//!
//! Past the deadline such a call goes on on the thread, and the decoder
//! fails every later call at once, so nothing the call holds is touched
//! until it has returned: the decoder's handle, its stream and the mapped
//! frame, and the staging memory the copies write, are kept with it
//! ([`wait::Unreturned`], [`UnfinishedPicture`]). The other codecs'
//! `cuvidDecodePicture` returns as soon as the picture is queued (H.264,
//! HEVC and AV1 on that GPU), and is made on the caller's thread.

use std::any::Any;
use std::ffi::{c_int, c_ulong, c_ulonglong, c_void};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use reims_vgpu_video_core::runtime::{Answer, Unanswered};
use reims_vgpu_video_core::sync::Mutex;

use super::ffi::{self, CUvideodecoder, Zeroed};
use super::{ChromaFormat, Nvdec, NvdecCodec, SurfaceFormat, decode_capabilities};
use crate::cuda::ffi::{CU_MEMORYTYPE_DEVICE, CU_MEMORYTYPE_HOST, CUDA_MEMCPY2D, CUstream};
use crate::cuda::{Gpu, Pinned, Stream};
use crate::error::Error;
use crate::wait::{self, Calls, Unfinished};

/// The call thread's name (ours).
const CALL_THREAD_NAME: &str = "reims-vgpu-video.nvdec-calls";

/// NVDEC's limit on decode surfaces (`ulNumDecodeSurfaces`), as the SDK's
/// samples and ffmpeg hold it.
pub(crate) const MAX_SURFACES: u32 = 32;

/// Surfaces beyond the references and the picture being decoded, so that
/// a frame the session still holds does not stall the next decode.
pub(crate) const SPARE_SURFACES: usize = 2;

/// `pictures` surfaces as a [`DecoderSpec`] counts them, up to
/// [`MAX_SURFACES`].
pub(crate) fn surface_count(pictures: usize) -> u32 {
    u32::try_from(pictures)
        .unwrap_or(MAX_SURFACES)
        .min(MAX_SURFACES)
}

/// Where a picture's slices lie in its bitstream, as `pSliceDataOffsets`
/// takes them.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Slices<'a> {
    /// Each slice's start: H.264's slices, HEVC's slice segments, VP9's
    /// one frame.
    Starts(&'a [u32]),
    /// Each tile's start and end: AV1's.
    Tiles(&'a [[u32; 2]]),
}

impl Slices<'_> {
    /// `nNumSlices` and `pSliceDataOffsets` for a bitstream of `len`
    /// bytes; `Err` where a slice lies outside it, or there are more than
    /// NVDEC counts.
    fn offsets(self, len: u32) -> Result<(u32, *const u32), Error> {
        let (count, offsets, within) = match self {
            Slices::Starts(starts) => (
                starts.len(),
                starts.as_ptr(),
                starts.iter().all(|&start| start <= len),
            ),
            Slices::Tiles(tiles) => (
                tiles.len(),
                tiles.as_ptr().cast::<u32>(),
                tiles.iter().all(|&[start, end]| start <= end && end <= len),
            ),
        };
        if !within {
            return Err(Error::Bitstream(format!(
                "a slice outside the picture's {len} bytes"
            )));
        }
        let count = u32::try_from(count)
            .map_err(|_| Error::Unsupported(format!("{count} slices in one picture")))?;
        Ok((count, offsets))
    }

    /// The offsets as `pSliceDataOffsets` takes them, copied.
    fn to_vec(self) -> Vec<u32> {
        match self {
            Slices::Starts(starts) => starts.to_vec(),
            Slices::Tiles(tiles) => tiles.iter().flatten().copied().collect(),
        }
    }
}

/// A picture's decode the call thread makes ([`Decoder::decode`]): its
/// parameters, their lengths set, with its own copies of the bitstream and
/// the slices' offsets, checked within it, which the parameters are
/// pointed at for the call, and the decoder, held until the call returns.
struct Submission {
    inner: Arc<Inner>,
    params: ffi::CUVIDPICPARAMS,
    bitstream: Vec<u8>,
    offsets: Vec<u32>,
    /// What a test holds the call in flight with, before the driver's call.
    #[cfg(test)]
    before: Option<tests::BeforeSubmit>,
}

// SAFETY: the parameters' pointers are pointed at the order's own memory on
// the thread that makes the call; the decoder's handle is called there
// while the caller holds its lock for it, or, past the deadline, by that
// thread alone, as the decoder makes no other call once stalled.
unsafe impl Send for Submission {}

impl Submission {
    /// Makes the decode, with the GPU's context pushed.
    fn submit(self) -> Result<(), Error> {
        let Self {
            inner,
            mut params,
            bitstream,
            offsets,
            #[cfg(test)]
            before,
        } = self;
        // The lengths are the borrowed ones', which these copy.
        params.pBitstreamData = bitstream.as_ptr();
        params.pSliceDataOffsets = offsets.as_ptr();
        let _current = inner.gpu.push()?;
        let handle = inner.handle();
        #[cfg(test)]
        inner.in_flight.store(true, Ordering::SeqCst);
        #[cfg(test)]
        if let Some(before) = before {
            before();
        }
        // SAFETY: a live handle, which the caller's lock or the stall keeps
        // to this call alone; the parameters live for the call, and point
        // at the order's bitstream and offsets, each within it (checked
        // before the copy). NVDEC copies them.
        let result = unsafe { (inner.nvdec.f.cuvidDecodePicture)(handle, &raw mut params) };
        #[cfg(test)]
        inner.in_flight.store(false, Ordering::SeqCst);
        inner.gpu.cuda().check("cuvidDecodePicture", result)
    }
}

/// What a decoder is made for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DecoderSpec {
    pub(crate) codec: NvdecCodec,
    /// 4:2:0, or monochrome (AV1's), which NVDEC writes as 4:2:0.
    pub(crate) chroma: ChromaFormat,
    /// Bits a sample: 8, 10 or 12.
    pub(crate) bit_depth: u32,
    /// The coded picture's size, which every surface has.
    pub(crate) width: u32,
    pub(crate) height: u32,
    /// The surfaces: the DPB, the picture being decoded, and spares for
    /// frames waiting to be copied.
    pub(crate) surfaces: u32,
}

struct Inner {
    gpu: Arc<Gpu>,
    nvdec: &'static Nvdec,
    handle: CUvideodecoder,
    spec: DecoderSpec,
    output: SurfaceFormat,
    /// Held across each call on the handle, with the size the decoder's
    /// post-processing writes a mapped frame at: `ulTargetWidth` and
    /// `ulTargetHeight`, last set by creation or reconfiguration.
    calls: Mutex<(u32, u32)>,
    /// The surface indices no picture holds.
    free: Mutex<Vec<c_int>>,
    /// Host memory a mapped picture is copied into, its planes' rows packed
    /// ([`Picture::read_back`]): kept from frame to frame, and held while a
    /// picture read back into it is copied out.
    staging: Mutex<Staging>,
    /// The stream a picture's map and copies are queued on, asked after
    /// rather than waited for (ours).
    stream: Stream,
    /// Set once a picture's work was not done by its deadline: every later
    /// call fails at once.
    stalled: AtomicBool,
    /// The thread VP9's decodes and the copies into pageable memory are
    /// made on (module docs), started with the first; `None` inside where
    /// none starts, and those calls are made on the caller's thread.
    caller: OnceLock<Option<Calls>>,
    /// Set while a call of the call thread's is in flight, for the tests'
    /// count of calls that touch what it holds meanwhile.
    #[cfg(test)]
    in_flight: AtomicBool,
}

// SAFETY: the handle is used under `calls`, one call at a time, with the
// GPU's context pushed on the calling thread; NVDEC's handles are not tied
// to a thread.
unsafe impl Send for Inner {}
// SAFETY: as above.
unsafe impl Sync for Inner {}

impl Inner {
    /// The decoder's handle, for a call on it, which is made under `calls`
    /// with the GPU's context pushed, or as the decoder is destroyed.
    fn handle(&self) -> CUvideodecoder {
        self.touch();
        self.handle
    }

    /// The decoder's stream, for work queued on it or a question of it.
    fn stream(&self) -> &Stream {
        self.touch();
        &self.stream
    }

    /// Counts, in tests, a use of what a call of the call thread's holds
    /// while it is in flight, which the decoder never makes (module docs).
    fn touch(&self) {
        #[cfg(test)]
        if self.in_flight.load(Ordering::SeqCst) {
            tests::TOUCHED_IN_FLIGHT.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// The call thread, started the first time it is asked for: `None`
    /// where no thread starts.
    fn caller(&self) -> Option<&Calls> {
        self.caller
            .get_or_init(|| Calls::start(CALL_THREAD_NAME))
            .as_ref()
    }

    /// Marks the decoder's work as not done by its deadline: every later
    /// call fails at once.
    fn stall(&self) {
        self.stalled.store(true, Ordering::Release);
    }

    /// `Err` where an earlier picture's work was not done by its deadline.
    fn check_not_stalled(&self) -> Result<(), Error> {
        if self.stalled.load(Ordering::Acquire) {
            return Err(Error::Stalled(
                "an earlier picture of this decoder in time, which it may still be decoding or \
                 copying"
                    .to_owned(),
            ));
        }
        Ok(())
    }

    /// Sets the decoder's coded and target sizes to `size`, a picture's,
    /// so that the next map writes it unscaled. The caller holds `calls`,
    /// with the GPU's context pushed.
    fn reconfigure(&self, size: (u32, u32)) -> Result<(), Error> {
        let inner = self;
        let (width, height) = size;
        let mut info = ffi::CUVIDRECONFIGUREDECODERINFO::zeroed();
        info.ulWidth = width;
        info.ulHeight = height;
        info.ulTargetWidth = width;
        info.ulTargetHeight = height;
        info.ulNumDecodeSurfaces = inner.spec.surfaces;
        info.display_area.right = i16::try_from(width).unwrap_or(i16::MAX);
        info.display_area.bottom = i16::try_from(height).unwrap_or(i16::MAX);
        // SAFETY: a live handle, used under `calls`; the info is live and
        // filled.
        let result =
            unsafe { (inner.nvdec.f.cuvidReconfigureDecoder)(inner.handle(), &raw mut info) };
        inner.gpu.cuda().check("cuvidReconfigureDecoder", result)
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        let _current = match self.gpu.push() {
            Ok(current) => current,
            Err(e) => {
                log::warn!("an NVDEC decoder is left undestroyed: {e}");
                return;
            }
        };
        // SAFETY: the handle was made by cuvidCreateDecoder and is
        // destroyed once, when no picture holds it.
        let result = unsafe { (self.nvdec.f.cuvidDestroyDecoder)(self.handle()) };
        if let Err(e) = self.gpu.cuda().check("cuvidDestroyDecoder", result) {
            log::warn!("{e}");
        }
    }
}

/// An NVDEC decoder.
pub(crate) struct Decoder {
    inner: Arc<Inner>,
}

impl Decoder {
    /// A decoder on `gpu` for `spec`, once NVDEC says it decodes it: the
    /// codec at 4:2:0 and the bit depth, within its sizes and macroblock
    /// count.
    pub(crate) fn new(gpu: Arc<Gpu>, spec: &DecoderSpec) -> Result<Self, Error> {
        let nvdec = Nvdec::get()?;
        let caps = decode_capabilities(&gpu, spec.codec, spec.chroma, spec.bit_depth)?;
        if !caps.supported {
            return Err(Error::Unsupported(format!(
                "NVDEC does not decode {:?} at {:?}, {} bits",
                spec.codec, spec.chroma, spec.bit_depth
            )));
        }
        let macroblocks = u64::from(spec.width.div_ceil(16)) * u64::from(spec.height.div_ceil(16));
        if spec.width < caps.min_width || spec.height < caps.min_height {
            return Err(Error::TooSmall(format!(
                "NVDEC decodes {:?} from {}x{}, not {}x{}",
                spec.codec, caps.min_width, caps.min_height, spec.width, spec.height
            )));
        }
        if spec.width > caps.max_width
            || spec.height > caps.max_height
            || macroblocks > u64::from(caps.max_macroblocks)
        {
            return Err(Error::Unsupported(format!(
                "NVDEC decodes {:?} up to {}x{} ({} macroblocks), not {}x{}",
                spec.codec,
                caps.max_width,
                caps.max_height,
                caps.max_macroblocks,
                spec.width,
                spec.height
            )));
        }
        // 8 bits decode into NV12; deeper into P016, whose samples are
        // P010's for 10 bits: the value in the high bits.
        let output = if spec.bit_depth > 8 {
            SurfaceFormat::P016
        } else {
            SurfaceFormat::Nv12
        };
        if !caps.output_formats.contains(&output) {
            return Err(Error::Unsupported(format!(
                "NVDEC writes {:?} at {} bits as {:?}, not {output:?}",
                spec.codec, spec.bit_depth, caps.output_formats
            )));
        }
        let surfaces = spec.surfaces.clamp(1, MAX_SURFACES);
        let width = c_ulong::from(spec.width);
        let height = c_ulong::from(spec.height);
        let mut info = ffi::CUVIDDECODECREATEINFO::zeroed();
        info.ulWidth = width;
        info.ulHeight = height;
        info.ulNumDecodeSurfaces = c_ulong::from(surfaces);
        info.CodecType = spec.codec.raw();
        info.ChromaFormat = spec.chroma.raw();
        info.ulCreationFlags = ffi::cudaVideoCreate_PreferCUVID;
        info.bitDepthMinus8 = c_ulong::from(spec.bit_depth.saturating_sub(8));
        info.ulMaxWidth = width;
        info.ulMaxHeight = height;
        info.display_area.right = i16::try_from(spec.width).unwrap_or(i16::MAX);
        info.display_area.bottom = i16::try_from(spec.height).unwrap_or(i16::MAX);
        info.OutputFormat = output.raw();
        info.DeinterlaceMode = ffi::cudaVideoDeinterlaceMode_Weave;
        info.ulTargetWidth = width;
        info.ulTargetHeight = height;
        // A picture is mapped only while it is copied, one at a time.
        info.ulNumOutputSurfaces = 1;
        let stream = Stream::new(&gpu)?;
        let mut handle: CUvideodecoder = std::ptr::null_mut();
        {
            let _current = gpu.push()?;
            // SAFETY: the info is live and filled; the handle is written.
            let result = unsafe { (nvdec.f.cuvidCreateDecoder)(&raw mut handle, &raw mut info) };
            gpu.cuda().check("cuvidCreateDecoder", result)?;
        }
        let surface_count = c_int::try_from(surfaces).unwrap_or(1);
        Ok(Self {
            inner: Arc::new(Inner {
                gpu,
                nvdec,
                handle,
                spec: DecoderSpec { surfaces, ..*spec },
                output,
                calls: Mutex::new((spec.width, spec.height)),
                free: Mutex::new((0..surface_count).rev().collect()),
                staging: Mutex::new(Staging::None),
                stream,
                stalled: AtomicBool::new(false),
                caller: OnceLock::new(),
                #[cfg(test)]
                in_flight: AtomicBool::new(false),
            }),
        })
    }

    /// The size every surface has: the spec's.
    pub(crate) fn size(&self) -> (u32, u32) {
        (self.inner.spec.width, self.inner.spec.height)
    }

    /// `Err` where an earlier picture's work was not done by its deadline,
    /// as every later decode then fails ([`Error::Stalled`]).
    pub(crate) fn check_not_stalled(&self) -> Result<(), Error> {
        self.inner.check_not_stalled()
    }

    /// A free surface for a new picture of the spec's size.
    pub(crate) fn picture(&self) -> Result<Picture, Error> {
        self.picture_of_size(self.size())
    }

    /// A free surface for a new picture mapped at `size`, no larger than
    /// the spec's: a VP9 or AV1 frame of another size than the decoder's.
    /// NVDEC decodes it at its own size into the surface, and scales it to
    /// the decoder's target size when it is mapped, so it is mapped with
    /// the decoder reconfigured to `size` ([`Picture::read_back`]). An AV1
    /// frame's size is known only at its decode: [`Picture::set_size`].
    pub(crate) fn picture_of_size(&self, size: (u32, u32)) -> Result<Picture, Error> {
        let index = self.inner.free.lock().pop().ok_or(Error::OutOfSurfaces)?;
        Ok(Picture {
            index,
            size: Mutex::new(size),
            inner: Arc::clone(&self.inner),
            hold: Arc::new(SurfaceHold {
                index,
                inner: Arc::clone(&self.inner),
            }),
            stand_in: false,
        })
    }

    /// Decodes one picture of `bitstream`: `params` names its surface and
    /// its references, and `slices` where its slices lie, each within the
    /// bitstream. `params` is pointed at both for the call. A VP9 picture
    /// is decoded on the call thread, from copies of both, and waited for
    /// until a deadline of its own, past which it is [`Error::Stalled`]
    /// (module docs).
    pub(crate) fn decode(
        &self,
        params: &mut ffi::CUVIDPICPARAMS,
        bitstream: &[u8],
        slices: Slices<'_>,
    ) -> Result<(), Error> {
        self.decode_by(params, bitstream, slices, &wait::Deadline::default())
    }

    /// [`Decoder::decode`], where a VP9 picture's decode is waited for
    /// until `deadline`, its decode call's (`crate::decode`).
    pub(crate) fn decode_by(
        &self,
        params: &mut ffi::CUVIDPICPARAMS,
        bitstream: &[u8],
        slices: Slices<'_>,
        deadline: &wait::Deadline,
    ) -> Result<(), Error> {
        let len = u32::try_from(bitstream.len())
            .map_err(|_| Error::Unsupported("a picture over 4 GiB".into()))?;
        let (count, offsets) = slices.offsets(len)?;
        params.nBitstreamDataLen = len;
        params.nNumSlices = count;
        let inner = &self.inner;
        let _calls = inner.calls.lock();
        inner.check_not_stalled()?;
        if inner.spec.codec == NvdecCodec::Vp9
            && let Some(caller) = inner.caller()
        {
            let order = Submission {
                inner: Arc::clone(inner),
                params: *params,
                bitstream: bitstream.to_vec(),
                offsets: slices.to_vec(),
                #[cfg(test)]
                before: tests::BEFORE_SUBMIT.take(),
            };
            return self.submit_on(caller, order, deadline);
        }
        params.pBitstreamData = bitstream.as_ptr();
        params.pSliceDataOffsets = offsets;
        let _current = inner.gpu.push()?;
        // SAFETY: a live handle; the parameters live for the call, and
        // point at the bitstream and the slices' offsets, borrowed for it,
        // each offset within the bitstream (checked). NVDEC copies them.
        let result = unsafe { (inner.nvdec.f.cuvidDecodePicture)(inner.handle(), params) };
        inner.gpu.cuda().check("cuvidDecodePicture", result)
    }

    /// Makes `order`'s decode on the call thread `caller`, and waits for it
    /// until `deadline`, with the decoder's lock held by the caller. Past
    /// it the decoder is stalled, and the call is kept with what it holds
    /// until it has returned (module docs).
    fn submit_on(
        &self,
        caller: &Calls,
        order: Submission,
        deadline: &wait::Deadline,
    ) -> Result<(), Error> {
        let index = order.params.CurrPicIdx;
        let submitting = caller.call(move || order.submit());
        match wait::answer_by(&submitting, deadline.at()) {
            Ok(submitted) => submitted,
            Err(Unanswered::Gone) => Err(Error::Unsupported(
                "the decoder's call thread made no decode".to_owned(),
            )),
            Err(Unanswered::TimedOut) => {
                let inner = &self.inner;
                inner.stall();
                let what = format!(
                    "NVDEC's decode of picture {index} of a {:?} decoder in time",
                    inner.spec.codec
                );
                let held: Box<dyn Any + Send> = Box::new(Arc::clone(inner));
                wait::keep(
                    what.clone(),
                    Box::new(wait::Unreturned::new(submitting, vec![held])),
                );
                Err(Error::Stalled(what))
            }
        }
    }

    /// Decodes one picture of `nals`, its H.264 slices or HEVC slice
    /// segments, as [`Decoder::decode`] does: sent as an Annex B stream,
    /// each NAL unit after a three-byte start code, with
    /// `pSliceDataOffsets` giving where each starts. Where `in_place` holds
    /// the sample the units lie in, and they lie there as one run
    /// ([`one_run`]), that run of the sample is sent as it is, with no copy
    /// gathered; otherwise the units are gathered into one. Each is counted.
    pub(crate) fn decode_nals<'a>(
        &self,
        params: &mut ffi::CUVIDPICPARAMS,
        nals: impl Iterator<Item = &'a [u8]> + Clone,
        in_place: &mut InPlace,
    ) -> Result<(), Error> {
        match in_place
            .sample()
            .map(|sample| one_run(sample, nals.clone()))
        {
            Some(Some((run, starts))) => {
                let decoded = self.decode(params, run, Slices::Starts(&starts));
                in_place.in_place += 1;
                return decoded;
            }
            Some(None) => in_place.gathered += 1,
            None => {}
        }
        let (bitstream, starts) = annex_b(nals)?;
        self.decode(params, &bitstream, Slices::Starts(&starts))
    }
}

/// `decoder`, the one made for the active SPS; an error before one is.
pub(crate) fn active(decoder: Option<&Decoder>) -> Result<&Decoder, Error> {
    decoder.ok_or_else(|| Error::Unsupported("decoding before an SPS is active".into()))
}

/// The sample, an Annex B access unit, an accelerator's pictures are being
/// decoded from, for the decodes of its slices to send in place
/// ([`Decoder::decode_nals`]), and the decodes sent so and gathered.
#[derive(Debug, Default)]
pub(crate) struct InPlace {
    /// The sample's first byte's address and its length, while it is set.
    sample: Option<(usize, usize)>,
    /// Pictures sent as a run of their sample.
    pub(crate) in_place: u64,
    /// Pictures of a set sample whose units were gathered.
    pub(crate) gathered: u64,
}

impl InPlace {
    /// Takes `sample` as the one the next pictures are decoded from, until
    /// [`Self::clear`].
    ///
    /// # Safety
    ///
    /// `sample` stays borrowed, and as it is, until [`Self::clear`] is
    /// called: the codec decoder's one call on it.
    pub(crate) unsafe fn set(&mut self, sample: &[u8]) {
        self.sample = Some((sample.as_ptr().expose_provenance(), sample.len()));
    }

    /// No sample is set.
    pub(crate) fn clear(&mut self) {
        self.sample = None;
    }

    /// The sample set.
    fn sample(&self) -> Option<&[u8]> {
        let (at, len) = self.sample?;
        // SAFETY: the bytes `set` was given, still borrowed and as they
        // were until `clear` (its contract), which is the only way out of
        // the decode that reads them.
        Some(unsafe { std::slice::from_raw_parts(std::ptr::with_exposed_provenance(at), len) })
    }
}

/// `nals`, units of `sample`, as one run of it: the bytes from the first
/// unit's three-byte start code to the last unit's end, where each unit
/// lies in `sample` after a start code and nothing but zeros lies between
/// one unit's end and the next's start code (the zero byte a four-byte
/// start code begins with, or `trailing_zero_8bits`), and where each starts
/// in that run. `None` where they lie otherwise: in other memory, out of
/// order, or with another unit between them (an SEI amid the slices).
pub(crate) fn one_run<'s, 'a>(
    sample: &'s [u8],
    nals: impl Iterator<Item = &'a [u8]>,
) -> Option<(&'s [u8], Vec<u32>)> {
    let base = sample.as_ptr().addr();
    let mut first = None;
    let mut end = 0;
    let mut starts = Vec::new();
    for nal in nals {
        let at = nal.as_ptr().addr().checked_sub(base)?;
        let unit_end = at.checked_add(nal.len())?;
        let start = at.checked_sub(3)?;
        if nal.is_empty() || unit_end > sample.len() || sample.get(start..at)? != [0, 0, 1] {
            return None;
        }
        let run_start = *first.get_or_insert(start);
        if !starts.is_empty() && (start < end || sample.get(end..start)?.iter().any(|&b| b != 0)) {
            return None;
        }
        starts.push(u32::try_from(start - run_start).ok()?);
        end = unit_end;
    }
    Some((sample.get(first?..end)?, starts))
}

/// One decode surface, held until dropped.
pub(crate) struct Picture {
    index: c_int,
    /// The size it is mapped at: the frame's own, or that rounded up to
    /// even sides, as the codec maps it.
    size: Mutex<(u32, u32)>,
    inner: Arc<Inner>,
    /// The surface's hold, shared with the pictures that stand in for a
    /// lost frame from this one.
    hold: Arc<SurfaceHold>,
    /// Whether it stands in for a lost frame ([`Picture::stand_in`]).
    stand_in: bool,
}

/// A decode surface no new picture may take: its index goes back to the
/// free ones when the last picture of it goes.
struct SurfaceHold {
    index: c_int,
    inner: Arc<Inner>,
}

impl Drop for SurfaceHold {
    fn drop(&mut self) {
        self.inner.free.lock().push(self.index);
    }
}

/// A rectangle of a picture, in luma samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Crop {
    pub(crate) x: u32,
    pub(crate) y: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
}

/// A host plane to copy into.
pub(crate) struct HostPlane<'a> {
    pub(crate) bytes: &'a mut [u8],
    pub(crate) bytes_per_row: usize,
    pub(crate) rows: u32,
}

/// The staging memory a decoder copies mapped pictures into (ours): CUDA's
/// page-locked host memory, which a copy from the GPU fills by DMA, where
/// the driver copies into pageable memory through a bounce buffer of its
/// own; pageable memory where CUDA gives none, which the copies into return
/// only once done, and so are made on the call thread (module docs).
enum Staging {
    None,
    Pinned(Pinned),
    Pageable(Vec<u8>),
}

impl Staging {
    /// Makes it `needed` bytes or more, keeping what it is where it is
    /// that large already.
    fn make_room(&mut self, gpu: &Arc<Gpu>, needed: usize) {
        if self.bytes_mut().len() >= needed {
            return;
        }
        *self = Staging::None;
        #[cfg(test)]
        if tests::PAGEABLE.get() {
            *self = Staging::Pageable(vec![0; needed]);
            return;
        }
        *self = match Pinned::new(gpu, needed) {
            Ok(pinned) => Staging::Pinned(pinned),
            Err(error) => {
                log::warn!(
                    "no page-locked staging memory of {needed} bytes, so this decoder's \
                     pictures are copied through pageable memory: {error}"
                );
                Staging::Pageable(vec![0; needed])
            }
        };
    }

    /// Whether a copy into it returns only once done.
    fn is_pageable(&self) -> bool {
        matches!(self, Staging::Pageable(_))
    }

    fn bytes(&self) -> &[u8] {
        match self {
            Staging::None => &[],
            Staging::Pinned(pinned) => pinned.bytes(),
            Staging::Pageable(bytes) => bytes,
        }
    }

    fn bytes_mut(&mut self) -> &mut [u8] {
        match self {
            Staging::None => &mut [],
            Staging::Pinned(pinned) => pinned.bytes_mut(),
            Staging::Pageable(bytes) => bytes,
        }
    }
}

/// A picture read back into its decoder's staging memory
/// ([`Picture::read_back`]), held there until dropped.
pub(crate) struct PictureReadBack<'a> {
    staging: reims_vgpu_video_core::sync::MutexGuard<'a, Staging>,
    /// The luma and the chroma, where they lie in `staging`.
    planes: [StagedPlane; 2],
}

/// Where a plane lies in the staging memory: `rows` rows of `row_bytes`
/// each, packed, from `offset`.
#[derive(Clone, Copy, Debug)]
struct StagedPlane {
    offset: usize,
    row_bytes: usize,
    rows: u32,
}

impl StagedPlane {
    /// Its bytes.
    fn len(self) -> usize {
        self.row_bytes * usize::try_from(self.rows).unwrap_or(0)
    }

    /// It as a plane to copy into, in `bytes`, the staging memory from its
    /// offset on.
    fn host(self, bytes: &mut [u8]) -> HostPlane<'_> {
        HostPlane {
            bytes: &mut bytes[..self.len()],
            bytes_per_row: self.row_bytes,
            rows: self.rows,
        }
    }
}

impl PictureReadBack<'_> {
    /// The luma and the chroma as they lie in the staging memory: each
    /// one's bytes, its rows' bytes, packed, and its rows.
    pub(crate) fn planes(&self) -> [(&[u8], usize, u32); 2] {
        self.planes.map(|plane| {
            let bytes = &self.staging.bytes()[plane.offset..plane.offset + plane.len()];
            (bytes, plane.row_bytes, plane.rows)
        })
    }

    /// Copies the read back crop into `luma` and `chroma`, as much as fits,
    /// as [`copy_plane`] copies out of a mapped surface: the CPU's copy
    /// alone.
    pub(crate) fn copy_to(&self, luma: HostPlane<'_>, chroma: HostPlane<'_>) {
        for (plane, to) in self.planes.into_iter().zip([luma, chroma]) {
            let from = &self.staging.bytes()[plane.offset..plane.offset + plane.len()];
            let width = plane.row_bytes.min(to.bytes_per_row);
            let rows = rows_that_fit(
                plane.rows.min(to.rows),
                width,
                to.bytes.len(),
                to.bytes_per_row,
            );
            for row in 0..rows {
                let source = &from[row * plane.row_bytes..][..width];
                to.bytes[row * to.bytes_per_row..][..width].copy_from_slice(source);
            }
        }
    }
}

impl Picture {
    /// The surface's index, which NVDEC's parameters name it by.
    pub(crate) fn index(&self) -> c_int {
        self.index
    }

    /// The GPU its decoder runs on.
    pub(crate) fn gpu(&self) -> &Arc<Gpu> {
        &self.inner.gpu
    }

    /// Another picture of this one's surface, to stand in for a lost frame
    /// that later frames predict from (ours). NVDEC's surfaces are written
    /// by a decode alone, and a decoded surface is decoded into again only
    /// once no picture holds it, so the stand-in holds what this one holds
    /// for as long as it lives.
    pub(crate) fn stand_in(&self) -> Self {
        Self {
            index: self.index,
            size: Mutex::new(self.size()),
            inner: Arc::clone(&self.inner),
            hold: Arc::clone(&self.hold),
            stand_in: true,
        }
    }

    /// Whether it stands in for a lost frame.
    pub(crate) fn is_stand_in(&self) -> bool {
        self.stand_in
    }

    /// The size it is mapped at.
    pub(crate) fn size(&self) -> (u32, u32) {
        *self.size.lock()
    }

    /// Sets the size it is mapped at, before its decode.
    pub(crate) fn set_size(&self, size: (u32, u32)) {
        *self.size.lock() = size;
    }

    /// Bytes a sample of the output: 1 for NV12, 2 for P016.
    pub(crate) fn bytes_per_sample(&self) -> usize {
        if self.inner.output == SurfaceFormat::P016 {
            2
        } else {
            1
        }
    }

    /// Waits for the picture's decode, and copies `crop` of it into the
    /// decoder's staging memory: luma, then chroma (interleaved Cb and Cr,
    /// half as tall), each row packed. Every call into the driver a copy
    /// needs is made here: the map, the copies and the unmap, each wait
    /// bounded (module docs). What it returns copies the crop on the CPU
    /// alone ([`PictureReadBack::copy_to`]), and holds the staging memory
    /// until dropped. A decode NVDEC reports as failed is
    /// [`Error::DecodeFailed`]. Work not done by `deadline` is
    /// [`Error::Stalled`], and the staging memory, which its copies may
    /// still write, is kept with it.
    pub(crate) fn read_back(
        &self,
        crop: Crop,
        deadline: &wait::Deadline,
    ) -> Result<PictureReadBack<'_>, Error> {
        let inner = &self.inner;
        let mut staging = inner.staging.lock();
        // SAFETY: the destinations are slices of the staging memory, pageable
        // as `Staging` says, or page-locked: `staging`'s lock is held until
        // `with_mapped` has waited for the copies, or the memory is kept
        // with them where they are not done by its deadline (below).
        let mapped = unsafe {
            self.with_mapped(crop, deadline, |mapped| {
                let luma = StagedPlane {
                    offset: 0,
                    row_bytes: mapped.row_bytes[0],
                    rows: mapped.rows[0],
                };
                let chroma = StagedPlane {
                    offset: luma.len(),
                    row_bytes: mapped.row_bytes[1],
                    rows: mapped.rows[1],
                };
                let needed = chroma.offset + chroma.len();
                staging.make_room(&inner.gpu, needed);
                let pageable = staging.is_pageable();
                let (luma_bytes, chroma_bytes) = staging.bytes_mut().split_at_mut(chroma.offset);
                let into = |plane: StagedPlane, bytes: &mut [u8]| {
                    let to = plane.host(bytes);
                    Destination {
                        at: to.bytes.as_mut_ptr(),
                        len: to.bytes.len(),
                        bytes_per_row: to.bytes_per_row,
                        rows: to.rows,
                    }
                };
                let copies = Copies {
                    planes: mapped.planes(into(luma, luma_bytes), into(chroma, chroma_bytes)),
                    pageable,
                };
                (copies, [luma, chroma])
            })
        };
        let planes = match mapped {
            Ok(copied) => copied?,
            Err(Missed::Failed(error)) => return Err(error),
            Err(Missed::Stalled(mut unfinished)) => {
                // The copies may still write the staging memory: it goes
                // with the work, and the decoder makes no other, as it
                // fails every later call.
                unfinished.keep_also(Box::new(std::mem::replace(&mut *staging, Staging::None)));
                return Err(unfinished.keep());
            }
        };
        Ok(PictureReadBack { staging, planes })
    }

    /// Waits for the picture's decode, and copies `crop` of it straight
    /// into `luma` and `chroma`, host memory the caller registered with
    /// CUDA, as much as fits: the copy by DMA of the GPU alone, with no
    /// staging and no copy by the CPU (`crate::zero_copy`). The outer
    /// `Err` is the picture's, as [`Self::read_back`]'s: its map, a decode
    /// NVDEC reports as failed, or work not done by `deadline`, which is
    /// kept with `written` until the GPU is done. The inner is the copy's,
    /// after which the destinations may hold part of the picture.
    ///
    /// # Safety
    ///
    /// Each destination's `len` bytes from `at` are host memory the caller
    /// keeps mapped and writable through the call, which nothing else
    /// writes meanwhile, and which `written` keeps so for as long as it
    /// lives: the copies may write it after the call, where they are not
    /// done by its deadline.
    pub(crate) unsafe fn write_into(
        &self,
        crop: Crop,
        luma: Destination,
        chroma: Destination,
        written: Vec<Box<dyn Any + Send>>,
        deadline: &wait::Deadline,
    ) -> Result<Result<(), Error>, Error> {
        // SAFETY: the caller's destinations, registered with CUDA, which
        // `written` keeps, as the function's contract says.
        let mapped = unsafe {
            self.with_mapped(crop, deadline, |mapped| {
                let copies = Copies {
                    planes: mapped.planes(luma, chroma),
                    pageable: false,
                };
                (copies, ())
            })
        };
        match mapped {
            Ok(copied) => Ok(copied),
            Err(Missed::Failed(error)) => Err(error),
            Err(Missed::Stalled(mut unfinished)) => {
                for kept in written {
                    unfinished.keep_also(kept);
                }
                Err(unfinished.keep())
            }
        }
    }

    /// Waits for the picture's decode, maps it, gives `plan` where `crop` of
    /// it lies on the GPU, makes the copies `plan` gives into host memory
    /// (the luma's, then the chroma's) and waits for them, and unmaps it, on
    /// every path. The decoder's handle is held throughout, with the GPU's
    /// context pushed. Each wait asks after its work until `deadline`
    /// (module docs): copies into pageable memory are made on the
    /// call thread, and their answer waited for, the rest queued on the
    /// decoder's stream, and the stream asked after. The outer `Err` is the
    /// map's, the decode's, the unmap's, or the work's not done by its
    /// deadline; the inner the copies'.
    ///
    /// # Safety
    ///
    /// The destinations `plan` gives are host memory, writable until the
    /// copies are done, which nothing else reaches meanwhile, and which the
    /// caller keeps so with the work where it is not done by its deadline
    /// ([`Missed::Stalled`]). Those it calls pageable are, and the others
    /// are page-locked or registered with CUDA.
    unsafe fn with_mapped<T>(
        &self,
        crop: Crop,
        deadline: &wait::Deadline,
        plan: impl FnOnce(&Mapped) -> (Copies, T),
    ) -> Result<Result<T, Error>, Missed> {
        let inner = &self.inner;
        let mut target = inner.calls.lock();
        inner.check_not_stalled()?;
        let _current = inner.gpu.push()?;
        let deadline = deadline.at();
        // The decode is asked after here, so that the map finds it done.
        let status = wait::poll(deadline, || match decode_status(inner, self.index) {
            Some(ffi::cuvidDecodeStatus_InProgress) => None,
            status => Some(status),
        });
        let Some(status) = status else {
            return Err(self.stalled(Waiting::Decode, None));
        };
        check_status(status)?;
        let size = self.size();
        if *target != size {
            inner.reconfigure(size)?;
            *target = size;
        }
        #[cfg(test)]
        tests::before_map(inner.stream().raw());
        let frame = MappedFrame::map(inner, self.index)?;
        let mapped = Mapped::of(
            frame.address,
            frame.pitch,
            size,
            crop,
            self.bytes_per_sample(),
        )?;
        let (copies, planned) = plan(&mapped);
        let copied = match copies.pageable.then(|| inner.caller()).flatten() {
            Some(caller) => {
                let order = PageableCopies {
                    inner: Arc::clone(inner),
                    copies,
                };
                let copying = caller.call(move || order.make());
                match wait::answer_by(&copying, deadline) {
                    Ok(copied) => copied,
                    Err(Unanswered::Gone) => Err(Error::Unsupported(
                        "the decoder's call thread made no copy".to_owned(),
                    )),
                    Err(Unanswered::TimedOut) => {
                        return Err(self.stalled(Waiting::Copies(copying), Some(frame.forget())));
                    }
                }
            }
            // SAFETY: the plan's destinations, as the function's contract
            // says; the sources lie in the frame, mapped until the stream
            // has run the copies, which is waited for below, or kept mapped
            // with them.
            None => unsafe { copies.make(inner.gpu.cuda(), inner.stream().raw()) },
        };
        // Whatever was queued, before a copy failed too, is waited for
        // before the frame is unmapped.
        let done = wait::poll(deadline, || match inner.stream().is_done() {
            Ok(false) => None,
            Ok(true) => Some(Ok(())),
            Err(error) => Some(Err(error)),
        });
        let Some(done) = done else {
            return Err(self.stalled(Waiting::Stream, Some(frame.forget())));
        };
        let unmapped = frame.unmap();
        let copied = copied.and(done).map(|()| planned);
        unmapped?;
        Ok(copied)
    }

    /// The picture's work, not done by its deadline: the decoder fails each
    /// later call at once, and what the work may still write is to be kept
    /// with it ([`UnfinishedPicture`]): the picture's surface, the decoder's
    /// handle and the frame, where it is `mapped` at that address.
    fn stalled(&self, waiting: Waiting, mapped: Option<c_ulonglong>) -> Missed {
        self.inner.stall();
        Missed::Stalled(Box::new(UnfinishedPicture {
            kept: Vec::new(),
            mapped,
            waiting,
            index: self.index,
            _surface: Arc::clone(&self.hold),
            inner: Arc::clone(&self.inner),
        }))
    }
}

/// NVDEC's status of the decode into surface `index` of `inner`'s decoder:
/// `None` where it answers an error, as it does for a codec it keeps no
/// status for (its header: HEVC, H.264 and JPEG are kept). The caller holds
/// `calls`, with the GPU's context pushed.
fn decode_status(inner: &Inner, index: c_int) -> Option<ffi::cuvidDecodeStatus> {
    let mut status = ffi::CUVIDGETDECODESTATUS::zeroed();
    // SAFETY: a live handle and one of its surfaces, used under `calls`;
    // the status is live.
    let result =
        unsafe { (inner.nvdec.f.cuvidGetDecodeStatus)(inner.handle(), index, &raw mut status) };
    (result == crate::cuda::ffi::CUDA_SUCCESS).then_some(status.decodeStatus)
}

/// A decode's status as the copy takes it: NVDEC's failure is
/// [`Error::DecodeFailed`], and a picture with no status is taken as
/// decoded.
fn check_status(status: Option<ffi::cuvidDecodeStatus>) -> Result<(), Error> {
    match status {
        Some(status @ (ffi::cuvidDecodeStatus_Error | ffi::cuvidDecodeStatus_Error_Concealed)) => {
            Err(Error::DecodeFailed(format!(
                "NVDEC reports the picture's decode failed (status {status})"
            )))
        }
        _ => Ok(()),
    }
}

/// Why [`Picture::with_mapped`] gave no copy.
enum Missed {
    /// The picture's decode, map or unmap failed.
    Failed(Error),
    /// Its work was not done by its deadline, and is to be kept.
    Stalled(Box<UnfinishedPicture>),
}

impl From<Error> for Missed {
    fn from(error: Error) -> Self {
        Self::Failed(error)
    }
}

/// What the GPU was doing for a picture past its deadline.
enum Waiting {
    /// Its decode, which `cuvidGetDecodeStatus` says is in progress.
    Decode,
    /// Its map's post-processing or its copies, on the decoder's stream.
    Stream,
    /// Its copies into pageable memory, made on the call thread, which
    /// return once done, and this answers; then the stream.
    Copies(Answer<Result<(), Error>>),
}

/// A picture's work the GPU had not done by its deadline, kept with all it
/// may still write until the GPU is done ([`wait::keep`]), then let go in
/// its drop: the mapped frame unmapped, the memory the copies write (the
/// staging memory, or a guest buffer's registration and hold) freed or let
/// go, the surface free and the decoder's handle destroyed once nothing
/// else holds it.
pub(crate) struct UnfinishedPicture {
    /// What the copies write.
    kept: Vec<Box<dyn Any + Send>>,
    /// Where the frame is mapped, for copies still queued.
    mapped: Option<c_ulonglong>,
    waiting: Waiting,
    index: c_int,
    /// The picture's surface, which no new picture takes meanwhile.
    _surface: Arc<SurfaceHold>,
    inner: Arc<Inner>,
}

impl UnfinishedPicture {
    /// Keeps `what` too, until the GPU is done.
    fn keep_also(&mut self, what: Box<dyn Any + Send>) {
        self.kept.push(what);
    }

    /// Hands the work to [`wait::keep`], and gives the error its picture
    /// fails with.
    fn keep(self: Box<Self>) -> Error {
        let what = format!(
            "NVDEC's {} of picture {} of a {:?} decoder in time",
            match self.waiting {
                Waiting::Decode => "decode",
                Waiting::Stream => "map and copy",
                Waiting::Copies(_) => "map and copy into pageable memory",
            },
            self.index,
            self.inner.spec.codec
        );
        wait::keep(what.clone(), self);
        Error::Stalled(what)
    }
}

impl Unfinished for UnfinishedPicture {
    /// Done once NVDEC says the decode is not in progress, or the stream
    /// that its copies are on has run them, after the copies into pageable
    /// memory have returned, before which nothing of the decoder's is
    /// touched; never while an answer is an error, which says nothing of
    /// the GPU's writes.
    fn finished(&mut self) -> bool {
        if let Waiting::Copies(copying) = &self.waiting {
            if !wait::returned(copying) {
                return false;
            }
            self.waiting = Waiting::Stream;
        }
        let inner = &self.inner;
        let _calls = inner.calls.lock();
        let Ok(_current) = inner.gpu.push() else {
            return false;
        };
        match self.waiting {
            Waiting::Decode => decode_status(inner, self.index)
                .is_some_and(|status| status != ffi::cuvidDecodeStatus_InProgress),
            Waiting::Stream | Waiting::Copies(_) => inner.stream().is_done().unwrap_or(false),
        }
    }
}

impl Drop for UnfinishedPicture {
    fn drop(&mut self) {
        let Some(address) = self.mapped.take() else {
            return;
        };
        let inner = &self.inner;
        let _calls = inner.calls.lock();
        let Ok(_current) = inner.gpu.push() else {
            log::warn!("a mapped NVDEC frame is left mapped: its GPU's context is not to be had");
            return;
        };
        // SAFETY: the frame its picture's map mapped on this handle,
        // unmapped once, here (the map's guard was let go of it), under
        // `calls` with the context pushed, once the copies from it are done.
        let result = unsafe { (inner.nvdec.f.cuvidUnmapVideoFrame64)(inner.handle(), address) };
        if let Err(e) = inner.gpu.cuda().check("cuvidUnmapVideoFrame64", result) {
            log::warn!("{e}");
        }
    }
}

/// A picture mapped by `cuvidMapVideoFrame64`, at its surface's address
/// and pitch on the GPU, unmapped once: by [`MappedFrame::unmap`], or as it
/// drops on any other path, a panic's unwinding included, so that the
/// decoder's one output surface is free for its next map; or never, where
/// it is [`MappedFrame::forget`]ed for copies still queued. It is made and
/// dropped inside the decoder's call lock, with the GPU's context pushed
/// ([`Picture::with_mapped`]).
struct MappedFrame<'a> {
    inner: &'a Inner,
    address: c_ulonglong,
    pitch: u32,
    /// Whether it is still mapped.
    mapped: bool,
}

impl<'a> MappedFrame<'a> {
    /// Maps surface `index` of `inner`'s decoder, its post-processing
    /// queued on the decoder's stream. The caller holds `calls`, with the
    /// GPU's context pushed, and has waited for the decode.
    fn map(inner: &'a Inner, index: c_int) -> Result<Self, Error> {
        let mut proc = ffi::CUVIDPROCPARAMS::zeroed();
        proc.progressive_frame = 1;
        proc.output_stream = inner.stream().raw();
        let mut address: c_ulonglong = 0;
        let mut pitch: u32 = 0;
        // SAFETY: a live handle and one of its surfaces, used under
        // `calls` with the context pushed; the outputs are live, and the
        // stream is the decoder's, on the same context.
        let result = unsafe {
            (inner.nvdec.f.cuvidMapVideoFrame64)(
                inner.handle(),
                index,
                &raw mut address,
                &raw mut pitch,
                &raw mut proc,
            )
        };
        inner.gpu.cuda().check("cuvidMapVideoFrame64", result)?;
        Ok(Self {
            inner,
            address,
            pitch,
            mapped: true,
        })
    }

    /// Unmaps it, with the driver's answer.
    fn unmap(mut self) -> Result<(), Error> {
        self.release()
    }

    /// Leaves it mapped, for copies from it still queued, and gives its
    /// address, to unmap once they are done ([`UnfinishedPicture`]).
    fn forget(mut self) -> c_ulonglong {
        self.mapped = false;
        self.address
    }

    fn release(&mut self) -> Result<(), Error> {
        if !std::mem::take(&mut self.mapped) {
            return Ok(());
        }
        // SAFETY: the frame `map` mapped on this handle, unmapped once (the
        // flag), still under `calls` with the context pushed.
        let result = unsafe {
            (self.inner.nvdec.f.cuvidUnmapVideoFrame64)(self.inner.handle(), self.address)
        };
        self.inner
            .gpu
            .cuda()
            .check("cuvidUnmapVideoFrame64", result)
    }
}

impl Drop for MappedFrame<'_> {
    fn drop(&mut self) {
        if let Err(e) = self.release() {
            log::warn!("{e}");
        }
    }
}

/// Where a mapped picture's crop lies on the GPU ([`Picture::with_mapped`]):
/// each plane's source (address, pitch, x in bytes, first row), and its
/// rows' bytes and count, held to the surface.
#[derive(Debug, PartialEq, Eq)]
struct Mapped {
    planes: [(c_ulonglong, usize, usize, usize); 2],
    row_bytes: [usize; 2],
    rows: [u32; 2],
}

impl Mapped {
    /// Where `crop` of a picture mapped at `size` lies, its surface at
    /// `address` with rows `pitch` bytes apart and samples of `bytes`
    /// bytes: the crop cut to the picture, and each plane's rows to the
    /// pitch. The chroma follows the luma's rows, their count rounded up to
    /// even, as NVDEC lays a mapped frame out; an odd width's last chroma
    /// pair covers its last column. `Err` is a surface past the address
    /// space, which no map gives.
    fn of(
        address: c_ulonglong,
        pitch: u32,
        (width, height): (u32, u32),
        crop: Crop,
        bytes: usize,
    ) -> Result<Self, Error> {
        let chroma = u64::from(height)
            .checked_next_multiple_of(2)
            .and_then(|rows| rows.checked_mul(u64::from(pitch)))
            .and_then(|offset| address.checked_add(offset))
            .ok_or_else(|| {
                Error::Unsupported(format!(
                    "a picture of {width}x{height} mapped at {address:#x}, {pitch} bytes a row"
                ))
            })?;
        let pitch = usize::try_from(pitch).unwrap_or(0);
        let crop = Crop {
            width: crop.width.min(width.saturating_sub(crop.x)),
            height: crop.height.min(height.saturating_sub(crop.y)),
            ..crop
        };
        let samples = |n: u32| usize::try_from(n).unwrap_or(0).saturating_mul(bytes);
        let x_bytes = samples(crop.x);
        let within = |row_bytes: usize| row_bytes.min(pitch.saturating_sub(x_bytes));
        let y = usize::try_from(crop.y).unwrap_or(0);
        Ok(Self {
            planes: [
                (address, pitch, x_bytes, y),
                (chroma, pitch, x_bytes, y / 2),
            ],
            row_bytes: [
                within(samples(crop.width)),
                within(samples(crop.width).next_multiple_of(2 * bytes.max(1))),
            ],
            rows: [crop.height, crop.height.div_ceil(2)],
        })
    }
}

impl Mapped {
    /// The copies of its two planes into `luma` and `chroma`.
    fn planes(&self, luma: Destination, chroma: Destination) -> [PlaneCopy; 2] {
        let [luma_from, chroma_from] = self.planes;
        [
            PlaneCopy {
                from: luma_from,
                row_bytes: self.row_bytes[0],
                rows: self.rows[0],
                to: luma,
            },
            PlaneCopy {
                from: chroma_from,
                row_bytes: self.row_bytes[1],
                rows: self.rows[1],
                to: chroma,
            },
        ]
    }
}

/// The copy of a mapped picture's plane into host memory ([`copy_plane`]'s
/// arguments): from the device plane (address, pitch, x in bytes, first
/// row), `rows` rows of `row_bytes`, into `to`.
#[derive(Clone, Copy, Debug)]
struct PlaneCopy {
    from: (c_ulonglong, usize, usize, usize),
    row_bytes: usize,
    rows: u32,
    to: Destination,
}

/// A picture's copies into host memory ([`Picture::with_mapped`]): the
/// luma's, then the chroma's, and whether the memory is pageable, which a
/// copy into returns only once done.
struct Copies {
    planes: [PlaneCopy; 2],
    pageable: bool,
}

impl Copies {
    /// Queues them on `stream`, in order, until one fails.
    ///
    /// # Safety
    ///
    /// As [`copy_plane`]'s, for each: the destinations are host memory,
    /// writable until the stream has run the copies, which nothing else
    /// reaches meanwhile; the sources lie in a frame mapped as long.
    unsafe fn make(&self, cuda: &crate::cuda::Cuda, stream: CUstream) -> Result<(), Error> {
        for plane in &self.planes {
            // SAFETY: the function's contract.
            unsafe {
                copy_plane(
                    cuda,
                    stream,
                    plane.from,
                    plane.row_bytes,
                    plane.rows,
                    plane.to,
                )
            }?;
        }
        Ok(())
    }
}

/// A picture's copies into pageable memory, which the call thread makes
/// ([`Picture::with_mapped`]), with the decoder, its stream and its context
/// held until they return.
struct PageableCopies {
    inner: Arc<Inner>,
    copies: Copies,
}

// SAFETY: the destinations are the staging memory, which the picture's read
// back holds through the copies, or keeps with them past its deadline; the
// sources are the mapped frame, kept mapped as long; the stream is called
// with the GPU's context pushed on the thread that makes the copies.
unsafe impl Send for PageableCopies {}

impl PageableCopies {
    /// Makes the copies on the decoder's stream, with the GPU's context
    /// pushed: each returns once done.
    fn make(self) -> Result<(), Error> {
        let inner = &self.inner;
        let _current = inner.gpu.push()?;
        let stream = inner.stream().raw();
        #[cfg(test)]
        inner.in_flight.store(true, Ordering::SeqCst);
        // SAFETY: the order's contract (above).
        let copied = unsafe { self.copies.make(inner.gpu.cuda(), stream) };
        #[cfg(test)]
        inner.in_flight.store(false, Ordering::SeqCst);
        copied
    }
}

/// A plane of host memory to copy into: `len` bytes from `at`, its rows
/// `bytes_per_row` apart, `rows` of them.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Destination {
    pub(crate) at: *mut u8,
    pub(crate) len: usize,
    pub(crate) bytes_per_row: usize,
    pub(crate) rows: u32,
}

/// `nals` as one Annex B stream, each after a three-byte start code, and
/// where each starts in it.
fn annex_b<'a>(nals: impl Iterator<Item = &'a [u8]> + Clone) -> Result<(Vec<u8>, Vec<u32>), Error> {
    let mut bitstream = Vec::with_capacity(nals.clone().map(|nal| nal.len() + 3).sum());
    let mut starts = Vec::new();
    for nal in nals {
        starts.push(
            u32::try_from(bitstream.len())
                .map_err(|_| Error::Unsupported("a picture over 4 GiB".into()))?,
        );
        bitstream.extend_from_slice(&[0, 0, 1]);
        bitstream.extend_from_slice(nal);
    }
    Ok((bitstream, starts))
}

/// How many of `rows` rows of `width` bytes, `stride` bytes apart, fit in
/// `len` bytes: each row but the last needs its stride, the last its width.
fn rows_that_fit(rows: u32, width: usize, len: usize, stride: usize) -> usize {
    let rows = usize::try_from(rows).unwrap_or(0);
    match len.checked_sub(width) {
        Some(past_first) if width > 0 => rows.min(past_first / stride.max(1) + 1),
        _ => 0,
    }
}

/// Queues on `stream` the copy of `rows` rows of `row_bytes` from the
/// device plane at `source` (address, pitch, x in bytes, first row) into
/// `plane`, as much as fits: its rows, and its bytes, each row but the last
/// needing its stride and the last its width. Into page-locked or
/// registered memory the copy runs after the call returns; into pageable
/// memory CUDA copies before it returns (the driver API's reference,
/// "API synchronization behavior").
///
/// # Safety
///
/// `plane`'s `len` bytes from `at` are host memory, writable until the
/// stream has run the copy, which nothing else reaches meanwhile.
unsafe fn copy_plane(
    cuda: &crate::cuda::Cuda,
    stream: CUstream,
    (device, pitch, x_bytes, y): (c_ulonglong, usize, usize, usize),
    row_bytes: usize,
    rows: u32,
    plane: Destination,
) -> Result<(), Error> {
    // Within the surface's pitch too.
    let width = row_bytes
        .min(plane.bytes_per_row)
        .min(pitch.saturating_sub(x_bytes));
    let rows = rows_that_fit(rows.min(plane.rows), width, plane.len, plane.bytes_per_row);
    if rows == 0 {
        return Ok(());
    }
    let copy = CUDA_MEMCPY2D {
        srcXInBytes: x_bytes,
        srcY: y,
        srcMemoryType: CU_MEMORYTYPE_DEVICE,
        srcHost: std::ptr::null(),
        srcDevice: device,
        srcArray: std::ptr::null_mut(),
        srcPitch: pitch,
        dstXInBytes: 0,
        dstY: 0,
        dstMemoryType: CU_MEMORYTYPE_HOST,
        dstHost: plane.at.cast::<c_void>(),
        dstDevice: 0,
        dstArray: std::ptr::null_mut(),
        dstPitch: plane.bytes_per_row,
        WidthInBytes: width,
        Height: rows,
    };
    // SAFETY: the source is the mapped surface, whose rows hold the crop
    // (it was held to the surface's size), mapped until the stream has run
    // the copy; the destination holds `rows` rows of `width` bytes at its
    // stride, as counted above, in memory the caller keeps writable as
    // long; the stream is live, on the context current here. CUDA reads
    // the description during the call.
    cuda.check("cuMemcpy2DAsync", unsafe {
        (cuda.f.cuMemcpy2DAsync_v2)(&raw const copy, stream)
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use std::cell::{Cell, RefCell};
    use std::sync::atomic::AtomicUsize;

    use super::*;

    /// What a picture's map hands the decoder's stream first: a test's
    /// work, queued ahead of the map's.
    pub(crate) type BeforeMap = Box<dyn FnOnce(CUstream)>;

    /// What the call thread runs inside a decode's submission, before the
    /// driver's call: a test's hold of the call in flight.
    pub(crate) type BeforeSubmit = Box<dyn FnOnce() + Send>;

    thread_local! {
        /// The work the next map on this thread queues first, once.
        pub(crate) static BEFORE_MAP: RefCell<Option<BeforeMap>> = const { RefCell::new(None) };
        /// Whether the staging memory made on this thread is pageable, as
        /// where CUDA gives no page-locked memory.
        pub(crate) static PAGEABLE: Cell<bool> = const { Cell::new(false) };
        /// What the next decode submitted on the call thread from this
        /// thread runs inside its submission, once.
        pub(crate) static BEFORE_SUBMIT: Cell<Option<BeforeSubmit>> = const { Cell::new(None) };
    }

    /// How many times, process-wide, a decoder used what a call of its call
    /// thread's holds while the call was in flight: its handle or its
    /// stream (`Inner::touch`).
    pub(crate) static TOUCHED_IN_FLIGHT: AtomicUsize = AtomicUsize::new(0);

    /// Queues the test's work on `stream`, where it set any.
    pub(super) fn before_map(stream: CUstream) {
        if let Some(work) = BEFORE_MAP.with_borrow_mut(Option::take) {
            work(stream);
        }
    }

    fn crop(x: u32, y: u32, width: u32, height: u32) -> Crop {
        Crop {
            x,
            y,
            width,
            height,
        }
    }

    #[test]
    fn a_mapped_pictures_chroma_follows_its_luma_rows_rounded_to_even() -> Result<(), Error> {
        let mapped = Mapped::of(0x1000, 2048, (641, 361), crop(0, 0, 641, 361), 1)?;
        assert_eq!(mapped.planes[0], (0x1000, 2048, 0, 0));
        assert_eq!(mapped.planes[1], (0x1000 + 2048 * 362, 2048, 0, 0));
        // An odd width's last chroma pair covers its last column.
        assert_eq!(mapped.row_bytes, [641, 642]);
        assert_eq!(mapped.rows, [361, 181]);
        let deep = Mapped::of(0, 4096, (1920, 1080), crop(0, 0, 1920, 1080), 2)?;
        assert_eq!(deep.planes[1].0, 4096 * 1080);
        assert_eq!(deep.row_bytes, [3840, 3840]);
        Ok(())
    }

    #[test]
    fn a_crop_is_cut_to_the_picture_and_its_rows_to_the_pitch() -> Result<(), Error> {
        // H.264's 1920x1088 coded picture shown as 1920x1080 from (0, 4).
        let mapped = Mapped::of(0, 2048, (1920, 1088), crop(0, 4, 1920, 1080), 1)?;
        assert_eq!((mapped.planes[0].3, mapped.planes[1].3), (4, 2));
        assert_eq!(mapped.rows, [1080, 540]);
        // A crop past the picture keeps what is in it.
        let mapped = Mapped::of(0, 64, (64, 32), crop(8, 30, 100, 100), 1)?;
        assert_eq!((mapped.planes[0].2, mapped.rows), (8, [2, 1]));
        assert_eq!(mapped.row_bytes, [56, 56]);
        // A pitch shorter than the rows holds them to it.
        let mapped = Mapped::of(0, 40, (64, 32), crop(16, 0, 64, 32), 1)?;
        assert_eq!(mapped.row_bytes, [24, 24]);
        Ok(())
    }

    #[test]
    fn rows_fit_by_their_stride_and_the_last_by_its_width() {
        // Three rows of 4 bytes 8 apart need 20 bytes.
        assert_eq!(rows_that_fit(3, 4, 20, 8), 3);
        assert_eq!(rows_that_fit(3, 4, 19, 8), 2);
        assert_eq!(rows_that_fit(3, 4, 4, 8), 1);
        assert_eq!(rows_that_fit(3, 4, 3, 8), 0);
        assert_eq!(rows_that_fit(3, 0, 64, 8), 0);
        assert_eq!(rows_that_fit(0, 4, 64, 8), 0);
    }

    #[test]
    fn nal_units_go_after_start_codes_at_their_offsets() -> Result<(), Error> {
        let nals: [&[u8]; 2] = [&[0x65, 0xaa], &[0x41]];
        let (bitstream, starts) = annex_b(nals.into_iter())?;
        assert_eq!(bitstream, [0, 0, 1, 0x65, 0xaa, 0, 0, 1, 0x41]);
        assert_eq!(starts, [0, 5]);
        Ok(())
    }

    /// Units of a sample that follow one another after start codes, with
    /// zeros alone between them, are one run of it, a four-byte start code's
    /// zero byte among them; units with another between them, out of order,
    /// or not after a start code are not, nor are units of other memory.
    #[test]
    fn units_after_start_codes_with_zeros_between_are_one_run() {
        let sample = [
            0, 0, 1, 9, 0xf0, // an access unit delimiter
            0, 0, 0, 1, 0x65, 0xaa, 0xbb, // a slice after a four-byte start code
            0, 0, 0, 1, 0x41, 0xcc, // another
            0, 0, 1, 6, 5, // an SEI
            0, 0, 1, 0x41, 0xdd,
        ];
        let unit = |at: usize, len: usize| &sample[at..at + len];
        let (run, starts) = one_run(&sample, [unit(9, 3), unit(16, 2)].into_iter()).unwrap();
        assert_eq!(run, &sample[6..18]);
        assert_eq!(starts, [0, 7]);
        assert_eq!(
            one_run(&sample, [unit(9, 3)].into_iter()),
            Some((&sample[6..12], vec![0]))
        );
        // An SEI between them, the units out of order, or one not after a
        // start code.
        assert!(one_run(&sample, [unit(16, 2), unit(26, 2)].into_iter()).is_none());
        assert!(one_run(&sample, [unit(16, 2), unit(9, 3)].into_iter()).is_none());
        assert!(one_run(&sample, [unit(10, 2)].into_iter()).is_none());
        let other = [0u8, 0, 1, 0x65];
        assert!(one_run(&sample, [&other[3..]].into_iter()).is_none());
        assert!(one_run(&sample, std::iter::empty()).is_none());
    }

    /// A sample set for a decode is read back as it was given, and none once
    /// cleared.
    #[test]
    fn a_set_sample_is_the_one_given_until_cleared() {
        let sample = [0u8, 0, 1, 0x65, 1, 2];
        let mut in_place = InPlace::default();
        assert!(in_place.sample().is_none());
        // SAFETY: `sample` lives, unchanged, past the clear below.
        unsafe { in_place.set(&sample) };
        assert_eq!(in_place.sample(), Some(&sample[..]));
        in_place.clear();
        assert!(in_place.sample().is_none());
    }

    #[test]
    fn slices_lie_within_their_bitstream() {
        assert_eq!(Slices::Starts(&[0, 5, 10]).offsets(10).map(|o| o.0), Ok(3));
        assert!(Slices::Starts(&[0, 11]).offsets(10).is_err());
        assert_eq!(
            Slices::Tiles(&[[0, 4], [4, 10]]).offsets(10).map(|o| o.0),
            Ok(2)
        );
        assert!(Slices::Tiles(&[[0, 11]]).offsets(10).is_err());
        assert!(Slices::Tiles(&[[6, 4]]).offsets(10).is_err());
    }

    #[test]
    fn a_surface_past_the_address_space_is_an_error_not_a_wrap() {
        assert!(Mapped::of(u64::MAX - 10, 4096, (64, 64), crop(0, 0, 64, 64), 1).is_err());
        assert!(Mapped::of(0, u32::MAX, (u32::MAX, u32::MAX), crop(0, 0, 1, 1), 2).is_ok());
    }
}
