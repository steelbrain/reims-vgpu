//! The JPEG engine's decodes on nvJPEG's hardware decoder
//! (`NVJPEG_BACKEND_HARDWARE`, the GPU's JPEG engine): core's
//! `JpegDecoder`.
//!
//! Each decode reads the stream with the codec crate's parser, checks it
//! against the request ([`plan`]), walks its scan on the CPU (the hardware,
//! like VCN's, is not trusted to refuse damaged data as Apple's does), and
//! hands nvJPEG the image written again from what the parser read
//! ([`stream`]). nvJPEG decodes into device memory, in one of two forms:
//! - the components' planes at their own sampling (`NVJPEG_OUTPUT_YUV`),
//!   for every YCbCr layout, and for `BGRA` at a scale;
//! - interleaved BGR (`NVJPEG_OUTPUT_BGRI`), for `BGRA` at scale 1, with
//!   its chroma interpolated on the way.
//!
//! A scale of 1/2, 1/4 or 1/8 is nvJPEG's own (`nvjpegDecodeParamsSetScaleFactor`,
//! the hardware decoder's alone). A crop at scale 1 is its region of
//! interest. A crop at another scale is cut from the whole image's scaled
//! decode, which is the same thing, as each 8x8 block scales alone: nvJPEG
//! failed a region of interest with a scale (`NVJPEG_STATUS_EXECUTION_FAILED`
//! or `NVJPEG_STATUS_BAD_JPEG`).
//!
//! The part of each plane the output is made of is copied into page-locked
//! memory on the decoder's own stream, which is asked after until a
//! deadline, and then written into the host session's planes by core's
//! `write_samples`, or as BGRA. Where nvJPEG's samples are not the output
//! core's reference makes (an odd size at a scale, chroma at another
//! sampling than the stream's, `BGRA` at a scale), the CPU makes it of them
//! first ([`convert`]).
//!
//! **What was measured**, on the RTX 5080 Laptop GPU (libnvjpeg 13.2.3,
//! driver 615.71.09, 2026-10-25), against libjpeg-turbo 3.2.0's islow
//! decode of cjpeg's streams, planes compared sample for sample:
//! - every plane equal for 4:2:0, 4:2:2, 4:4:4 and grey, at qualities 10
//!   to 100, with restart markers every MCU, every three and every row,
//!   at 1x1, 17x9, 15x15, 65x33, 333x211, 4095x7, 7x4095, 16384x64,
//!   64x16384 and 16384x16000, and for crops at the image's edge;
//! - the hardware decoder refuses 4:1:1, progressive and multi-scan
//!   streams (`NVJPEG_STATUS_JPEG_NOT_SUPPORTED`): the last two the parser
//!   refuses first;
//! - scaled decodes within 0.15 of a sample on average of each plane
//!   box-filtered from libjpeg's, at most 1 off in luma at even sizes;
//! - times, a 4000x3000 4:2:0 decode into device memory 7.1 ms and a
//!   16384x16000 one 107 to 140 ms, with about 1 µs of the CPU's.
//!
//! **Waiting.** A decode is asked after with `cuStreamQuery` until
//! `crate::wait`'s deadline. Past it the engine is stalled: that decode
//! fails as a malfunction, and everything the GPU may still read or write
//! (nvJPEG's state and buffers, the device memory, the page-locked memory
//! and the stream nvJPEG reads) is kept until the stream is done. Each later
//! decode fails at once the same way until then, and decodes go on after:
//! the engine is the backend's for as long as the process runs.

mod convert;
mod nvjpeg;
mod plan;
mod stream;

use std::ffi::{c_int, c_uint, c_void};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use reims_vgpu_video_codec::jpeg as syntax;
use reims_vgpu_video_core::jpeg::{
    JpegCapabilities, JpegDecode, JpegDecoder, JpegDecodes, JpegError, JpegOutputFormat,
    JpegSubsampling, PixelRect,
};
use reims_vgpu_video_core::surfaces::PlaneDestination;
use reims_vgpu_video_core::sync::Mutex;

use crate::cuda::ffi::{CU_MEMORYTYPE_DEVICE, CU_MEMORYTYPE_HOST, CUDA_MEMCPY2D, CUdeviceptr};
use crate::cuda::{DeviceMemory, Gpu, Pinned, Stream};
use crate::dl::Zeroed;
use crate::error::Error;
use crate::wait::{self, Unfinished};
use nvjpeg::{Nvjpeg, nvjpegStatus_t};
use plan::Plan;

/// The device and page-locked memory a decode keeps for the next, at most:
/// a 4000x3000 4:2:0 decode needs 18 MB of each. Ours.
const KEPT_BYTES: usize = 64 << 20;

/// nvJPEG's hardware decoder on one GPU, as the JPEG engine's backend.
pub struct NvidiaJpegDecoder {
    gpu: Arc<Gpu>,
    session: Arc<Session>,
    /// What a decode works with, made at the first and after a stall; held
    /// for each decode, one at a time, as the engine says it has one core.
    work: Mutex<Option<Work>>,
    /// The decode the GPU did not finish by its deadline, and whether what
    /// it kept is let go (module docs).
    stalled: Mutex<Option<(String, Arc<AtomicBool>)>>,
}

impl NvidiaJpegDecoder {
    /// nvJPEG's hardware decoder on `gpu`, once a trial decode of a small
    /// image has come back right. An error where libnvjpeg is not on the
    /// loader's path, its hardware decoder does not open on the GPU, or the
    /// trial fails: the backend then has no JPEG engine.
    pub fn new(gpu: Arc<Gpu>) -> Result<Self, Error> {
        let nvjpeg = Nvjpeg::get()?;
        let session = Arc::new(Session::new(&gpu, nvjpeg)?);
        let decoder = Self {
            gpu,
            session,
            work: Mutex::new(None),
            stalled: Mutex::new(None),
        };
        decoder.trial()?;
        log::info!(
            "NVIDIA {}: JPEG engine on libnvjpeg {}'s hardware decoder ({} engine(s), {} core(s) \
             each)",
            decoder.gpu.info().pci_bus_id,
            nvjpeg.version(),
            decoder.session.engines.0,
            decoder.session.engines.1
        );
        Ok(decoder)
    }

    /// Decodes an 8x8 grey image of 128s into 4:2:0, and checks every
    /// sample: a hardware decoder that opens but decodes wrong is not used.
    fn trial(&self) -> Result<(), Error> {
        let data = stream::flat_grey();
        let request = JpegDecode {
            data: &data,
            width: 8,
            height: 8,
            subsampling: JpegSubsampling::Grey,
            region: PixelRect {
                x: 0,
                y: 0,
                width: 8,
                height: 8,
            },
            scale: 1,
            format: JpegOutputFormat::Biplanar420,
        };
        let mut luma = [0u8; 64];
        let mut chroma = [0u8; 32];
        let decoded = self.decode_logged(
            &request,
            &mut [
                PlaneDestination {
                    bytes: &mut luma,
                    bytes_per_row: 8,
                    rows: 8,
                },
                PlaneDestination {
                    bytes: &mut chroma,
                    bytes_per_row: 8,
                    rows: 4,
                },
            ],
            false,
        );
        let wrong = |what: String| Error::Unsupported(format!("the trial JPEG decode {what}"));
        decoded.map_err(|error| wrong(format!("failed: {error}")))?;
        if luma.iter().chain(&chroma).any(|&sample| sample != 128) {
            return Err(wrong(format!(
                "came back wrong: {luma:?} and {chroma:?}, not 128s"
            )));
        }
        Ok(())
    }

    /// The work the GPU did not finish by its deadline, where what it kept
    /// is not let go yet; once it is, `None`, and the stall is over.
    fn still_stalled(&self) -> Option<String> {
        let mut stalled = self.stalled.lock();
        let (what, let_go) = stalled.as_ref()?;
        if !let_go.load(Ordering::Acquire) {
            return Some(what.clone());
        }
        log::info!("the GPU is done with the stalled JPEG decode, and decodes go on");
        *stalled = None;
        None
    }

    fn decode_logged(
        &self,
        request: &JpegDecode<'_>,
        planes: &mut [PlaneDestination<'_>],
        log: bool,
    ) -> Result<(), JpegError> {
        let runtime = wait::runtime();
        let started = runtime.now();
        let image = syntax::parse(request.data).map_err(stream_error)?;
        let plan = Plan::of(&image, request)?;
        // The hardware is not trusted to refuse damaged data as Apple's
        // does, so the scan is walked on the CPU first.
        let walked = runtime.now();
        syntax::walk(&image).map_err(stream_error)?;
        let walk_time = runtime.elapsed_since(walked);
        let data = stream::rewrite(&image)?;
        let mut slot = self.work.lock();
        if let Some(what) = self.still_stalled() {
            return Err(JpegError::Malfunction(format!(
                "the GPU did not finish {what}, of an earlier decode"
            )));
        }
        if slot.is_none() {
            let work = Work::new(&self.gpu, &self.session)
                .map_err(|e| JpegError::Malfunction(e.to_string()))?;
            *slot = Some(work);
        }
        let work = slot
            .as_mut()
            .ok_or_else(|| JpegError::Malfunction("no decode state".into()))?;
        let gpu_started = runtime.now();
        match work.decode(data, &plan) {
            Ok(()) => {}
            Err(Failure::Decode(error)) => return Err(error),
            Err(Failure::Stalled) => {
                let what = format!("a {}x{} JPEG decode in time", request.width, request.height);
                let (let_go, setter) = LetGo::new();
                *self.stalled.lock() = Some((what.clone(), let_go));
                if let Some(work) = slot.take() {
                    wait::keep(
                        what.clone(),
                        Box::new(UnfinishedDecode {
                            work,
                            _let_go: setter,
                        }),
                    );
                }
                return Err(JpegError::Malfunction(format!(
                    "the GPU did not finish {what}"
                )));
            }
        }
        let gpu_time = runtime.elapsed_since(gpu_started);
        let staged = work
            .staging
            .as_ref()
            .ok_or_else(|| JpegError::Malfunction("no staging memory".into()))?;
        plan.write(staged.bytes(), request, planes)?;
        work.trim();
        if log {
            log::info!(
                "NVIDIA JPEG decode: {}x{} {:?} into {:?}, region {}x{} at ({}, {}) at 1/{}, \
                 by {:?}{} from nvJPEG's 1/{}, in {:.1} ms ({:.1} ms on the GPU and the copy \
                 back), the scan walked in {:.2} ms",
                request.width,
                request.height,
                request.subsampling,
                request.format,
                request.region.width,
                request.region.height,
                request.region.x,
                request.region.y,
                request.scale,
                plan.route,
                if plan.resampled() { ", resampled" } else { "" },
                plan.decode_scale(),
                runtime.elapsed_since(started).as_secs_f64() * 1000.0,
                gpu_time.as_secs_f64() * 1000.0,
                walk_time.as_secs_f64() * 1000.0
            );
        }
        Ok(())
    }
}

impl JpegDecoder for NvidiaJpegDecoder {
    /// The four samplings the module docs measure, at any size a frame
    /// header gives. nvJPEG's hardware decoder answers for each stream
    /// whether it takes it (`nvjpegDecoderJpegSupported`), which nothing
    /// here can say ahead, so a stream it refuses is within this answer.
    fn decodes(&self) -> JpegDecodes {
        JpegDecodes {
            subsamplings: vec![
                JpegSubsampling::Yuv420,
                JpegSubsampling::Yuv422,
                JpegSubsampling::Yuv444,
                JpegSubsampling::Grey,
            ],
            max_size: (u32::from(u16::MAX), u32::from(u16::MAX)),
        }
    }

    fn capabilities(&self) -> JpegCapabilities {
        // What the guest is told is the same on every backend: one core, and
        // none of the four features. CMPhoto then sends no stream that needs
        // them (devices.md, "The JPEG engine"); a quantization value over 255
        // is refused (`stream`), as on VA-API.
        JpegCapabilities {
            cores: 1,
            sixteen_bit_quantization_tables: false,
            missing_eoi: false,
            restart_logging: false,
            interchange_formats: false,
        }
    }

    fn decode(
        &self,
        request: &JpegDecode<'_>,
        planes: &mut [PlaneDestination<'_>],
    ) -> Result<(), JpegError> {
        self.decode_logged(request, planes, true)
    }
}

/// Core's error for the parser's or the walk's, as VA-API's decoder gives
/// it.
fn stream_error(error: syntax::Error) -> JpegError {
    match error {
        syntax::Error::Unsupported(what) => JpegError::UnsupportedStream(what.into()),
        syntax::Error::ScanEndsEarly => JpegError::ScanEndsEarly,
        other => JpegError::BadData(other.to_string()),
    }
}

/// libnvjpeg's handle on a GPU's context, and its hardware decoder.
struct Session {
    gpu: Arc<Gpu>,
    nvjpeg: &'static Nvjpeg,
    handle: nvjpeg::nvjpegHandle_t,
    decoder: nvjpeg::nvjpegJpegDecoder_t,
    /// The hardware decoder's engines, and the cores of each.
    engines: (c_uint, c_uint),
}

// SAFETY: nvJPEG's library handle is thread safe ("The library handle is
// thread safe, and can be used by multiple threads simultaneously", nvJPEG's
// documentation, thread safety); its decoder is used by one decode at a
// time, under `NvidiaJpegDecoder::work`'s lock, and by kept work, whose
// stream is done before anything of it is destroyed.
unsafe impl Send for Session {}
// SAFETY: as above.
unsafe impl Sync for Session {}

impl Session {
    fn new(gpu: &Arc<Gpu>, nvjpeg: &'static Nvjpeg) -> Result<Self, Error> {
        let _current = gpu.push()?;
        let f = &nvjpeg.f;
        let mut handle: nvjpeg::nvjpegHandle_t = std::ptr::null_mut();
        // SAFETY: null allocators are the library's own; the out pointer is
        // live; the GPU's context is current, which the handle binds to.
        check_open("nvjpegCreateEx", unsafe {
            (f.nvjpegCreateEx)(
                nvjpeg::NVJPEG_BACKEND_HARDWARE,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                nvjpeg::NVJPEG_FLAGS_UPSAMPLING_WITH_INTERPOLATION,
                &raw mut handle,
            )
        })?;
        let mut session = Self {
            gpu: Arc::clone(gpu),
            nvjpeg,
            handle,
            decoder: std::ptr::null_mut(),
            engines: (0, 0),
        };
        let (mut engines, mut cores) = (0, 0);
        // SAFETY: a live handle and two live out pointers.
        check_open("nvjpegGetHardwareDecoderInfo", unsafe {
            (f.nvjpegGetHardwareDecoderInfo)(handle, &raw mut engines, &raw mut cores)
        })?;
        if engines == 0 {
            return Err(Error::Unsupported(
                "nvJPEG lists no hardware JPEG decoder on this GPU".into(),
            ));
        }
        session.engines = (engines, cores);
        let mut decoder: nvjpeg::nvjpegJpegDecoder_t = std::ptr::null_mut();
        // SAFETY: a live handle and a live out pointer.
        check_open("nvjpegDecoderCreate", unsafe {
            (f.nvjpegDecoderCreate)(handle, nvjpeg::NVJPEG_BACKEND_HARDWARE, &raw mut decoder)
        })?;
        session.decoder = decoder;
        Ok(session)
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        let Ok(_current) = self.gpu.push() else {
            log::warn!("nvJPEG's handle is left undestroyed: its GPU's context is not to be had");
            return;
        };
        let f = &self.nvjpeg.f;
        if !self.decoder.is_null() {
            // SAFETY: made by nvjpegDecoderCreate, destroyed once, after
            // every state made on it (each `Work` holds the session).
            log_status("nvjpegDecoderDestroy", unsafe {
                (f.nvjpegDecoderDestroy)(self.decoder)
            });
        }
        // SAFETY: made by nvjpegCreateEx, destroyed once, last.
        log_status("nvjpegDestroy", unsafe { (f.nvjpegDestroy)(self.handle) });
    }
}

/// An open's error for an nvJPEG status.
fn check_open(call: &'static str, status: nvjpegStatus_t) -> Result<(), Error> {
    if status == nvjpeg::NVJPEG_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(Error::Unsupported(format!(
            "{call} failed: {} ({status})",
            nvjpeg::status_name(status)
        )))
    }
}

/// A decode's error for an nvJPEG status: a stream nvJPEG reads as bad or
/// does not take is the stream's, the rest a malfunction.
fn check_decode(call: &'static str, status: nvjpegStatus_t) -> Result<(), JpegError> {
    let what = || format!("{call}: {} ({status})", nvjpeg::status_name(status));
    match status {
        nvjpeg::NVJPEG_STATUS_SUCCESS => Ok(()),
        nvjpeg::NVJPEG_STATUS_BAD_JPEG | nvjpeg::NVJPEG_STATUS_INCOMPLETE_BITSTREAM => {
            Err(JpegError::BadData(what()))
        }
        nvjpeg::NVJPEG_STATUS_JPEG_NOT_SUPPORTED => Err(JpegError::UnsupportedStream(what())),
        _ => Err(JpegError::Malfunction(what())),
    }
}

/// Logs a teardown call's failure.
fn log_status(call: &'static str, status: nvjpegStatus_t) {
    if status != nvjpeg::NVJPEG_STATUS_SUCCESS {
        log::warn!("{call} failed: {} ({status})", nvjpeg::status_name(status));
    }
}

/// Why [`Work::decode`] has no output.
enum Failure {
    /// The stream's error, or a malfunction; the GPU has done all the
    /// decode queued.
    Decode(JpegError),
    /// The GPU did not finish by the deadline.
    Stalled,
}

/// What a decode works with: nvJPEG's parsed stream, state, buffers and
/// parameters, the CUDA stream its work and the copy back are queued on,
/// the device and page-locked memory, and the bytes nvJPEG reads.
struct Work {
    session: Arc<Session>,
    stream: Stream,
    jpeg: nvjpeg::nvjpegJpegStream_t,
    state: nvjpeg::nvjpegJpegState_t,
    pinned_buffer: nvjpeg::nvjpegBufferPinned_t,
    device_buffer: nvjpeg::nvjpegBufferDevice_t,
    params: nvjpeg::nvjpegDecodeParams_t,
    device: Option<DeviceMemory>,
    staging: Option<Pinned>,
    data: Vec<u8>,
}

// SAFETY: nvJPEG's objects here are used by one thread at a time, under
// `NvidiaJpegDecoder::work`'s lock, or by the kept work's owner alone, with
// the GPU's context pushed for each call (nvJPEG's documentation, thread
// safety: a state, its buffers and a stream are one thread's at a time).
unsafe impl Send for Work {}

impl Work {
    fn new(gpu: &Arc<Gpu>, session: &Arc<Session>) -> Result<Self, Error> {
        let stream = Stream::new(gpu)?;
        let _current = gpu.push()?;
        let f = &session.nvjpeg.f;
        let handle = session.handle;
        let mut work = Self {
            session: Arc::clone(session),
            stream,
            jpeg: std::ptr::null_mut(),
            state: std::ptr::null_mut(),
            pinned_buffer: std::ptr::null_mut(),
            device_buffer: std::ptr::null_mut(),
            params: std::ptr::null_mut(),
            device: None,
            staging: None,
            data: Vec::new(),
        };
        // SAFETY: a live handle and a live out pointer; this object, and
        // each made below with null allocators (the library's own), is
        // destroyed once by `Drop`, which skips the null ones.
        check_open("nvjpegJpegStreamCreate", unsafe {
            (f.nvjpegJpegStreamCreate)(handle, &raw mut work.jpeg)
        })?;
        // SAFETY: as above.
        check_open("nvjpegDecoderStateCreate", unsafe {
            (f.nvjpegDecoderStateCreate)(handle, session.decoder, &raw mut work.state)
        })?;
        // SAFETY: as above.
        check_open("nvjpegBufferPinnedCreate", unsafe {
            (f.nvjpegBufferPinnedCreate)(handle, std::ptr::null_mut(), &raw mut work.pinned_buffer)
        })?;
        // SAFETY: as above.
        check_open("nvjpegBufferDeviceCreate", unsafe {
            (f.nvjpegBufferDeviceCreate)(handle, std::ptr::null_mut(), &raw mut work.device_buffer)
        })?;
        // SAFETY: a live state and a live buffer, both of this handle.
        check_open("nvjpegStateAttachPinnedBuffer", unsafe {
            (f.nvjpegStateAttachPinnedBuffer)(work.state, work.pinned_buffer)
        })?;
        // SAFETY: as above.
        check_open("nvjpegStateAttachDeviceBuffer", unsafe {
            (f.nvjpegStateAttachDeviceBuffer)(work.state, work.device_buffer)
        })?;
        // SAFETY: as above.
        check_open("nvjpegDecodeParamsCreate", unsafe {
            (f.nvjpegDecodeParamsCreate)(handle, &raw mut work.params)
        })?;
        Ok(work)
    }

    /// Decodes `data` as `plan` says, into the device memory, and copies
    /// the output's rectangle back into the page-locked memory, waited for
    /// until `crate::wait`'s deadline. Whatever was queued on the stream is
    /// waited for before an error returns, so that an error leaves no work
    /// running.
    fn decode(&mut self, data: Vec<u8>, plan: &Plan) -> Result<(), Failure> {
        let deadline = wait::deadline();
        let malfunction = |error: Error| Failure::Decode(JpegError::Malfunction(error.to_string()));
        self.reserve(plan).map_err(malfunction)?;
        self.data = data;
        let queued = self.queue(plan);
        let done = wait::poll(deadline, || match self.stream.is_done() {
            Ok(false) => None,
            Ok(true) => Some(Ok(())),
            Err(error) => Some(Err(error)),
        });
        match done {
            None => Err(Failure::Stalled),
            Some(Err(error)) => Err(malfunction(error)),
            Some(Ok(())) => queued.map_err(Failure::Decode),
        }
    }

    /// Device and page-locked memory enough for `plan`, made anew where the
    /// last decode's is too small.
    fn reserve(&mut self, plan: &Plan) -> Result<(), Error> {
        let gpu = &self.session.gpu;
        if self
            .device
            .as_ref()
            .is_none_or(|d| d.len() < plan.device_bytes)
        {
            self.device = None;
            self.device = Some(DeviceMemory::new(gpu, plan.device_bytes)?);
        }
        if self
            .staging
            .as_ref()
            .is_none_or(|s| s.bytes().len() < plan.staged_bytes)
        {
            self.staging = None;
            self.staging = Some(Pinned::new(gpu, plan.staged_bytes.max(1))?);
        }
        Ok(())
    }

    /// Lets go of memory larger than [`KEPT_BYTES`], once a decode is done
    /// with it.
    fn trim(&mut self) {
        if self.device.as_ref().is_some_and(|d| d.len() > KEPT_BYTES) {
            self.device = None;
        }
        if self
            .staging
            .as_ref()
            .is_some_and(|s| s.bytes().len() > KEPT_BYTES)
        {
            self.staging = None;
        }
        if self.data.capacity() > KEPT_BYTES {
            self.data = Vec::new();
        }
    }

    /// Queues the decode and the copies back on the stream.
    fn queue(&mut self, plan: &Plan) -> Result<(), JpegError> {
        let session = &self.session;
        let f = &session.nvjpeg.f;
        let cuda = session.gpu.cuda();
        let _current = session
            .gpu
            .push()
            .map_err(|e| JpegError::Malfunction(e.to_string()))?;
        // SAFETY: a live handle and parsed-stream object; `data` lives in
        // `self` until the next decode, after this one's stream is done, or
        // with the kept work.
        check_decode("nvjpegJpegStreamParse", unsafe {
            (f.nvjpegJpegStreamParse)(
                session.handle,
                self.data.as_ptr(),
                self.data.len(),
                0,
                0,
                self.jpeg,
            )
        })?;
        // SAFETY: a live parameters object and a format nvjpeg.h names.
        check_decode("nvjpegDecodeParamsSetOutputFormat", unsafe {
            (f.nvjpegDecodeParamsSetOutputFormat)(self.params, plan.output)
        })?;
        let [x, y, w, h] = plan.roi.unwrap_or([0, 0, -1, -1]);
        // SAFETY: as above; (0, 0, -1, -1) is the whole image.
        check_decode("nvjpegDecodeParamsSetROI", unsafe {
            (f.nvjpegDecodeParamsSetROI)(self.params, x, y, w, h)
        })?;
        // SAFETY: as above, with a factor nvjpeg.h names.
        check_decode("nvjpegDecodeParamsSetScaleFactor", unsafe {
            (f.nvjpegDecodeParamsSetScaleFactor)(self.params, plan.scale_factor)
        })?;
        let mut supported: c_int = -1;
        // SAFETY: live decoder, parsed stream and parameters; a live out
        // pointer.
        check_decode("nvjpegDecoderJpegSupported", unsafe {
            (f.nvjpegDecoderJpegSupported)(
                session.decoder,
                self.jpeg,
                self.params,
                &raw mut supported,
            )
        })?;
        if supported != 0 {
            return Err(JpegError::UnsupportedStream(format!(
                "nvJPEG's hardware decoder does not take it ({supported})"
            )));
        }
        let device = self
            .device
            .as_ref()
            .ok_or_else(|| JpegError::Malfunction("no device memory".into()))?;
        let mut image = nvjpeg::nvjpegImage_t::zeroed();
        for ((channel, pitch), plane) in image
            .channel
            .iter_mut()
            .zip(image.pitch.iter_mut())
            .zip(&plan.planes)
        {
            let address = device_address(device, plane.device_offset);
            *channel = std::ptr::without_provenance_mut(usize::try_from(address).unwrap_or(0));
            *pitch = plane.pitch;
        }
        // SAFETY: live handle, decoder, state, parameters and parsed stream,
        // one thread's at a time (the lock).
        check_decode("nvjpegDecodeJpegHost", unsafe {
            (f.nvjpegDecodeJpegHost)(
                session.handle,
                session.decoder,
                self.state,
                self.params,
                self.jpeg,
            )
        })?;
        // SAFETY: as above, on the work's live stream, with its GPU's
        // context current.
        check_decode("nvjpegDecodeJpegTransferToDevice", unsafe {
            (f.nvjpegDecodeJpegTransferToDevice)(
                session.handle,
                session.decoder,
                self.state,
                self.jpeg,
                self.stream.raw(),
            )
        })?;
        // SAFETY: as above; each channel holds its plane's rows at its
        // pitch within the device memory `plan` sized, which lives until
        // the stream is done.
        check_decode("nvjpegDecodeJpegDevice", unsafe {
            (f.nvjpegDecodeJpegDevice)(
                session.handle,
                session.decoder,
                self.state,
                &raw mut image,
                self.stream.raw(),
            )
        })?;
        let staging = self
            .staging
            .as_mut()
            .ok_or_else(|| JpegError::Malfunction("no staging memory".into()))?;
        let staged = staging.bytes_mut();
        for plane in &plan.planes {
            let copied = plane.copied;
            let to = staged
                .get_mut(plane.staged_offset..)
                .ok_or_else(|| JpegError::Malfunction("staging memory too small".into()))?;
            let copy = CUDA_MEMCPY2D {
                srcXInBytes: copied.x,
                srcY: copied.y,
                srcMemoryType: CU_MEMORYTYPE_DEVICE,
                srcHost: std::ptr::null(),
                srcDevice: device_address(device, plane.device_offset),
                srcArray: std::ptr::null_mut(),
                srcPitch: plane.pitch,
                dstXInBytes: 0,
                dstY: 0,
                dstMemoryType: CU_MEMORYTYPE_HOST,
                dstHost: to.as_mut_ptr().cast::<c_void>(),
                dstDevice: 0,
                dstArray: std::ptr::null_mut(),
                dstPitch: copied.width.max(1),
                WidthInBytes: copied.width,
                Height: copied.height,
            };
            // SAFETY: the source rectangle lies in the plane nvJPEG writes
            // (`Plan::of`), and the destination's rows, packed, in the
            // page-locked memory `plan` sized; both live until the stream
            // is done, which is waited for or kept with them.
            cuda.check("cuMemcpy2DAsync", unsafe {
                (cuda.f.cuMemcpy2DAsync_v2)(&raw const copy, self.stream.raw())
            })
            .map_err(|e| JpegError::Malfunction(e.to_string()))?;
        }
        Ok(())
    }
}

impl Drop for Work {
    fn drop(&mut self) {
        let session = Arc::clone(&self.session);
        let f = &session.nvjpeg.f;
        if let Ok(_current) = session.gpu.push() {
            // Each object below was made by its creator in `Work::new` and
            // is destroyed once: the state before the buffers attached to
            // it, and none while the stream has work of theirs (the decode
            // waits for it, and kept work is dropped once it is done).
            if !self.params.is_null() {
                // SAFETY: as above.
                log_status("nvjpegDecodeParamsDestroy", unsafe {
                    (f.nvjpegDecodeParamsDestroy)(self.params)
                });
            }
            if !self.state.is_null() {
                // SAFETY: as above.
                log_status("nvjpegJpegStateDestroy", unsafe {
                    (f.nvjpegJpegStateDestroy)(self.state)
                });
            }
            if !self.device_buffer.is_null() {
                // SAFETY: as above.
                log_status("nvjpegBufferDeviceDestroy", unsafe {
                    (f.nvjpegBufferDeviceDestroy)(self.device_buffer)
                });
            }
            if !self.pinned_buffer.is_null() {
                // SAFETY: as above.
                log_status("nvjpegBufferPinnedDestroy", unsafe {
                    (f.nvjpegBufferPinnedDestroy)(self.pinned_buffer)
                });
            }
            if !self.jpeg.is_null() {
                // SAFETY: as above.
                log_status("nvjpegJpegStreamDestroy", unsafe {
                    (f.nvjpegJpegStreamDestroy)(self.jpeg)
                });
            }
        } else {
            log::warn!(
                "nvJPEG's decode state is left undestroyed: its GPU's context is not to be had"
            );
        }
        // Then the memory nvJPEG wrote and the copies read.
        self.device = None;
        self.staging = None;
    }
}

/// The device address `offset` bytes into `device`.
fn device_address(device: &DeviceMemory, offset: usize) -> CUdeviceptr {
    device
        .address()
        .saturating_add(u64::try_from(offset).unwrap_or(u64::MAX))
}

/// Sets its flag when dropped: what kept work lets go of says so.
struct LetGo(Arc<AtomicBool>);

impl LetGo {
    /// The flag, unset, and its setter.
    fn new() -> (Arc<AtomicBool>, Self) {
        let flag = Arc::new(AtomicBool::new(false));
        (Arc::clone(&flag), Self(flag))
    }
}

impl Drop for LetGo {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

/// A decode the GPU did not finish by its deadline, kept until its stream
/// is done.
struct UnfinishedDecode {
    work: Work,
    _let_go: LetGo,
}

impl Unfinished for UnfinishedDecode {
    /// Done once the stream has run everything queued on it; never while
    /// its answer is an error, which says nothing of the GPU's writes.
    fn finished(&mut self) -> bool {
        self.work.stream.is_done().unwrap_or(false)
    }
}

#[cfg(test)]
mod tests;
