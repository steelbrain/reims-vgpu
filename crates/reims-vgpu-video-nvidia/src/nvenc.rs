//! NVENC: libnvidia-encode, loaded at run time; its sessions on a GPU's
//! CUDA context, and what it encodes.
//!
//! The library hands out its entry points in a table
//! (`NvEncodeAPICreateInstance`). A session is opened on a CUDA context at
//! API 13.1, the version this binding declares, once the driver says it
//! supports that (`NvEncodeAPIGetMaxSupportedVersion`).

pub(crate) mod encoder;
pub(crate) mod ffi;
mod locks;

use std::cell::{OnceCell, RefCell};
use std::ffi::{CStr, c_int, c_void};
use std::mem::MaybeUninit;
use std::sync::{Arc, OnceLock};

use crate::cuda::Gpu;
use crate::dl::Library;
use crate::error::Error;
use crate::wait;
use ffi::{GUID, NV_ENCODE_API_FUNCTION_LIST, NVENCSTATUS, NvencEntryPoints, Zeroed};
use locks::{Finishing, Lock, LockCall, Locking, Locks};

/// libnvidia-encode and its entry points, loaded once per process.
pub(crate) struct Nvenc {
    pub(crate) api: NV_ENCODE_API_FUNCTION_LIST,
    /// `NvEncodeAPIGetMaxSupportedVersion`: major and minor.
    max_version: (u32, u32),
    _library: Library,
}

// SAFETY: the table is the driver's static entry points, read only after
// it is filled; the library handle is never closed.
unsafe impl Send for Nvenc {}
// SAFETY: as above.
unsafe impl Sync for Nvenc {}

static NVENC: OnceLock<Result<Nvenc, Error>> = OnceLock::new();

/// An entry point of the table, or an error naming it where the driver
/// left it null.
macro_rules! entry {
    ($nvenc:expr, $name:ident) => {
        $nvenc.api.$name.ok_or_else(|| {
            Error::Unsupported(format!("the driver's NVENC has no {}", stringify!($name)))
        })
    };
}

impl Nvenc {
    /// libnvidia-encode, loading it on the first call. Fails when the
    /// driver's NVENC is older than API 13.1.
    pub(crate) fn get() -> Result<&'static Self, Error> {
        NVENC.get_or_init(Self::load).as_ref().map_err(Clone::clone)
    }

    fn load() -> Result<Self, Error> {
        let library = Library::open(c"libnvidia-encode.so.1")?;
        let entry_points = NvencEntryPoints::load(&library)?;
        let mut version = 0u32;
        // SAFETY: the pointer is to a live u32.
        let status = unsafe { (entry_points.NvEncodeAPIGetMaxSupportedVersion)(&raw mut version) };
        check_status("NvEncodeAPIGetMaxSupportedVersion", status, None)?;
        // The answer is (major << 4) | minor.
        let max_version = (version >> 4, version & 0xf);
        if max_version < (ffi::NVENCAPI_MAJOR_VERSION, ffi::NVENCAPI_MINOR_VERSION) {
            return Err(Error::Unsupported(format!(
                "the driver's NVENC is API {}.{}, older than {}.{}",
                max_version.0,
                max_version.1,
                ffi::NVENCAPI_MAJOR_VERSION,
                ffi::NVENCAPI_MINOR_VERSION
            )));
        }
        let mut api = NV_ENCODE_API_FUNCTION_LIST::zeroed();
        api.version = ffi::NV_ENCODE_API_FUNCTION_LIST_VER;
        // SAFETY: the table is live and zeroed but for its version.
        let status = unsafe { (entry_points.NvEncodeAPICreateInstance)(&raw mut api) };
        check_status("NvEncodeAPICreateInstance", status, None)?;
        Ok(Self {
            api,
            max_version,
            _library: library,
        })
    }
}

/// The highest NVENC API version the driver supports, major and minor.
pub fn max_api_version() -> Result<(u32, u32), Error> {
    Ok(Nvenc::get()?.max_version)
}

/// `Ok` for `NV_ENC_SUCCESS`, else the error with the status's name and
/// the session's last error text where there is a session.
pub(crate) fn check_status(
    call: &'static str,
    status: NVENCSTATUS,
    encoder: Option<ffi::Encoder>,
) -> Result<(), Error> {
    if status == ffi::NV_ENC_SUCCESS {
        return Ok(());
    }
    let name = usize::try_from(status)
        .ok()
        .and_then(|index| ffi::NVENCSTATUS_NAMES.get(index))
        .copied()
        .unwrap_or("an unknown NVENCSTATUS");
    let detail = encoder
        .and_then(|encoder| {
            let nvenc = Nvenc::get().ok()?;
            let last = nvenc.api.nvEncGetLastErrorString?;
            // SAFETY: the encoder is a live session; the driver returns a
            // NUL-terminated string it owns, or NULL.
            let text = unsafe { last(encoder) };
            (!text.is_null()).then(|| {
                // SAFETY: non-null and NUL-terminated, read before the
                // session's next call.
                unsafe { CStr::from_ptr(text) }
                    .to_string_lossy()
                    .into_owned()
            })
        })
        .filter(|text| !text.is_empty());
    Err(Error::Nvenc {
        call,
        status,
        message: match detail {
            Some(detail) => format!("{name}: {detail}"),
            None => name.to_owned(),
        },
    })
}

/// An NVENC session on a GPU, destroyed when dropped.
///
/// It keeps the configuration it was initialised with
/// ([`Session::initialize`]) at the one address NVENC was given, changed in
/// place by a reconfiguration, and frees it only after the session is
/// destroyed: the header does not say NVENC copies it, and its SDK's
/// samples and ffmpeg keep it as long.
///
/// It keeps the buffers it makes ([`Session::create_input`] and the
/// others), which its owner names by [`Buffer`], and destroys each once,
/// before itself, as the header asks. The driver's handles never leave it:
/// each call that names a buffer is made here, from the buffer's own
/// handle, so that no other value reaches the driver as one.
///
/// **Its waits are bounded** (`crate::wait`). A bitstream or vector
/// buffer's lock, which waits for the picture or the estimate that fills
/// it, is made on a thread of the session's own, and waited for until the
/// deadline ([`locks`], which says why). Past the deadline the session is
/// stalled: that lock and every later call fail at once
/// ([`Error::Stalled`]), and when it drops, its lock thread finishes it
/// and destroys it once the lock returns.
pub(crate) struct Session {
    pub(crate) nvenc: &'static Nvenc,
    encoder: ffi::Encoder,
    /// The GPU, whose context the session runs on, held for as long.
    pub(crate) gpu: Arc<Gpu>,
    /// The initialisation and the configuration NVENC was given, once
    /// initialised, the first pointing at the second.
    config: Option<(ffi::NV_ENC_INITIALIZE_PARAMS, Box<ffi::NV_ENC_CONFIG>)>,
    /// The buffers it made, which [`Buffer`] indexes.
    buffers: Vec<Kept>,
    /// The output buffers submitted and not yet locked: what NVENC may
    /// still write, and what it may lock.
    submitted: RefCell<Submitted>,
    /// Its lock thread, started with its first lock; `None` where no thread
    /// starts, or in a session finished on the thread of the session it
    /// was.
    locks: OnceCell<Option<Locks>>,
    /// The buffer whose lock passed its deadline, and that lock's answer:
    /// the session is stalled.
    stuck: RefCell<Option<(*mut c_void, Locking)>>,
}

/// A buffer a [`Session`] made, by its place among the session's buffers.
/// A call given one of another kind refuses it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Buffer(usize);

/// A buffer a session keeps: the driver's handle, and what it is.
#[derive(Clone, Copy, Debug)]
struct Kept {
    handle: *mut c_void,
    kind: BufferKind,
}

/// What a session's buffer is, which also says what destroys it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BufferKind {
    /// An input buffer (`nvEncCreateInputBuffer`), of frames of this size
    /// and format.
    Input {
        width: u32,
        height: u32,
        format: ffi::NV_ENC_BUFFER_FORMAT,
    },
    /// A bitstream buffer (`nvEncCreateBitstreamBuffer`).
    Bitstream,
}

/// What [`Session::encode_picture`] submits.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Submission {
    /// A frame from an input buffer, uploaded at `pitch`, into a bitstream
    /// buffer: `index` numbers it (`frameIdx`), `timestamp` comes back with
    /// its packet, and `idr` makes it an IDR picture with its parameter
    /// sets before it.
    Frame {
        input: Buffer,
        output: Buffer,
        pitch: u32,
        index: u32,
        timestamp: u64,
        idr: bool,
    },
    /// The end of the stream: every frame NVENC holds comes out.
    End,
}

// SAFETY: an NVENC session may be driven from any thread, one call at a
// time; `&mut self` (or the caller's lock) gives that, since it is not
// `Sync`. The pointers it keeps are the driver's handles and the kept
// initialisation's to the kept configuration, which moves with it.
unsafe impl Send for Session {}

impl Session {
    /// Opens a session on `gpu`'s CUDA context.
    pub(crate) fn open(gpu: Arc<Gpu>) -> Result<Self, Error> {
        let nvenc = Nvenc::get()?;
        let open = entry!(nvenc, nvEncOpenEncodeSessionEx)?;
        let mut params = ffi::NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS::zeroed();
        params.version = ffi::NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
        params.deviceType = ffi::NV_ENC_DEVICE_TYPE_CUDA;
        params.device = gpu.context();
        params.apiVersion = ffi::NVENCAPI_VERSION;
        let mut encoder: ffi::Encoder = std::ptr::null_mut();
        let current = gpu.push()?;
        // SAFETY: the parameters are live and name the GPU's retained
        // context, current on this thread; the pointer is to a live handle.
        let status = unsafe { open(&raw mut params, &raw mut encoder) };
        drop(current);
        if status != ffi::NV_ENC_SUCCESS {
            // The header: on failure the client "must call
            // NvEncDestroyEncoder" on the handle it got, if any.
            if !encoder.is_null()
                && let Some(destroy) = nvenc.api.nvEncDestroyEncoder
            {
                // SAFETY: the handle the failed open returned, destroyed
                // once.
                unsafe { destroy(encoder) };
            }
            check_status("nvEncOpenEncodeSessionEx", status, None)?;
        }
        Ok(Self {
            nvenc,
            encoder,
            gpu,
            config: None,
            buffers: Vec::new(),
            submitted: RefCell::new(Submitted::default()),
            locks: OnceCell::new(),
            stuck: RefCell::new(None),
        })
    }

    /// `Err` where a lock of the session's passed its deadline.
    fn check_not_stalled(&self) -> Result<(), Error> {
        if self.stuck.borrow().is_some() {
            return Err(Error::Stalled(
                "an earlier picture of this NVENC session in time, which it may still be encoding"
                    .to_owned(),
            ));
        }
        Ok(())
    }

    /// Initialises the session with `init`, which this points at
    /// `config`, both kept for the session's life (type docs).
    pub(crate) fn initialize(
        &mut self,
        init: ffi::NV_ENC_INITIALIZE_PARAMS,
        config: ffi::NV_ENC_CONFIG,
    ) -> Result<(), Error> {
        let initialize = entry!(self.nvenc, nvEncInitializeEncoder)?;
        let _current = self.gpu.push()?;
        let encoder = self.encoder;
        let (init, config) = self.config.insert((init, Box::new(config)));
        init.encodeConfig = &raw mut **config;
        // SAFETY: a live session; the parameters are live, versioned, and
        // point at the configuration the session keeps until it is
        // destroyed.
        let status = unsafe { initialize(encoder, init) };
        self.check("nvEncInitializeEncoder", status)
    }

    /// Reconfigures the running session (`nvEncReconfigureEncoder`, with no
    /// reset and no forced IDR) to its initialisation and configuration as
    /// `change` changes them: they take the kept ones' places, the
    /// configuration at its address, and the old ones stay where NVENC
    /// refuses the new.
    pub(crate) fn reconfigure(
        &mut self,
        change: impl FnOnce(&mut ffi::NV_ENC_INITIALIZE_PARAMS, &mut ffi::NV_ENC_CONFIG),
    ) -> Result<(), Error> {
        let call = entry!(self.nvenc, nvEncReconfigureEncoder)?;
        self.check_not_stalled()?;
        let Some((kept_init, kept)) = self.config.as_mut() else {
            return Err(Error::Unsupported(
                "reconfiguring an NVENC session not initialised".into(),
            ));
        };
        let (mut init, mut config) = (*kept_init, **kept);
        change(&mut init, &mut config);
        let previous = std::mem::replace(&mut **kept, config);
        init.encodeConfig = &raw mut **kept;
        let mut params = ffi::NV_ENC_RECONFIGURE_PARAMS::zeroed();
        params.version = ffi::NV_ENC_RECONFIGURE_PARAMS_VER;
        params.reInitEncodeParams = init;
        let encoder = self.encoder;
        let result = self.gpu.push().and_then(|_current| {
            // SAFETY: a live, initialised session; the parameters are live,
            // versioned, and point at the kept configuration.
            let status = unsafe { call(encoder, &raw mut params) };
            check_status("nvEncReconfigureEncoder", status, Some(encoder))
        });
        match result {
            Ok(()) => *kept_init = init,
            Err(_) => **kept = previous,
        }
        result
    }

    /// Makes an input buffer of `width` by `height` frames in `format`.
    pub(crate) fn create_input(
        &mut self,
        width: u32,
        height: u32,
        format: ffi::NV_ENC_BUFFER_FORMAT,
    ) -> Result<Buffer, Error> {
        let create = entry!(self.nvenc, nvEncCreateInputBuffer)?;
        let mut input = ffi::NV_ENC_CREATE_INPUT_BUFFER::zeroed();
        input.version = ffi::NV_ENC_CREATE_INPUT_BUFFER_VER;
        input.width = width;
        input.height = height;
        input.memoryHeap = ffi::NV_ENC_MEMORY_HEAP_AUTOSELECT;
        input.bufferFmt = format;
        {
            let _current = self.gpu.push()?;
            // SAFETY: a live, initialised session; the parameters are live
            // and versioned.
            self.check("nvEncCreateInputBuffer", unsafe {
                create(self.encoder, &raw mut input)
            })?;
        }
        let kind = BufferKind::Input {
            width,
            height,
            format,
        };
        self.keep("nvEncCreateInputBuffer", input.inputBuffer, kind)
    }

    /// Makes a bitstream buffer, of NVENC's own size.
    pub(crate) fn create_bitstream(&mut self) -> Result<Buffer, Error> {
        let create = entry!(self.nvenc, nvEncCreateBitstreamBuffer)?;
        let mut output = ffi::NV_ENC_CREATE_BITSTREAM_BUFFER::zeroed();
        output.version = ffi::NV_ENC_CREATE_BITSTREAM_BUFFER_VER;
        {
            let _current = self.gpu.push()?;
            // SAFETY: a live, initialised session; the parameters are live
            // and versioned.
            self.check("nvEncCreateBitstreamBuffer", unsafe {
                create(self.encoder, &raw mut output)
            })?;
        }
        self.keep(
            "nvEncCreateBitstreamBuffer",
            output.bitstreamBuffer,
            BufferKind::Bitstream,
        )
    }

    /// Keeps `handle`, a buffer of `kind` that `call` just made, until the
    /// session is destroyed, and names it.
    fn keep(
        &mut self,
        call: &'static str,
        handle: *mut c_void,
        kind: BufferKind,
    ) -> Result<Buffer, Error> {
        if handle.is_null() {
            return Err(Error::Unsupported(format!("{call} gave no buffer")));
        }
        self.buffers.push(Kept { handle, kind });
        Ok(Buffer(self.buffers.len() - 1))
    }

    /// `input`'s handle, size and format, where it is an input buffer of
    /// this session's.
    fn input(
        &self,
        input: Buffer,
    ) -> Result<(*mut c_void, u32, u32, ffi::NV_ENC_BUFFER_FORMAT), Error> {
        match self.buffers.get(input.0) {
            Some(&Kept {
                handle,
                kind:
                    BufferKind::Input {
                        width,
                        height,
                        format,
                    },
            }) => Ok((handle, width, height, format)),
            _ => Err(not_for_the_call(input)),
        }
    }

    /// `output`'s handle, where it is one of `kinds` of this session's.
    fn output(&self, output: Buffer, kinds: &[BufferKind]) -> Result<*mut c_void, Error> {
        match self.buffers.get(output.0) {
            Some(kept) if kinds.contains(&kept.kind) => Ok(kept.handle),
            _ => Err(not_for_the_call(output)),
        }
    }

    /// Copies a 4:2:0 frame into `input`, one of the session's input
    /// buffers, not locked: `rows` luma rows of `row_bytes` from `luma`,
    /// then half as many chroma rows (rounded up) from `chroma`, each
    /// plane's bytes with its bytes from one row to the next, at NVENC's
    /// pitch, under the buffer's lock. Returns the pitch.
    pub(crate) fn upload(
        &self,
        input: Buffer,
        luma: (&[u8], usize),
        chroma: (&[u8], usize),
        row_bytes: usize,
        rows: usize,
    ) -> Result<u32, Error> {
        let lock = entry!(self.nvenc, nvEncLockInputBuffer)?;
        let unlock = entry!(self.nvenc, nvEncUnlockInputBuffer)?;
        let (input, ..) = self.input(input)?;
        self.check_not_stalled()?;
        let mut locked = ffi::NV_ENC_LOCK_INPUT_BUFFER::zeroed();
        locked.version = ffi::NV_ENC_LOCK_INPUT_BUFFER_VER;
        locked.inputBuffer = input;
        let _current = self.gpu.push()?;
        // SAFETY: a live session and one of its input buffers, not locked:
        // each lock here is unlocked before the call returns.
        self.check("nvEncLockInputBuffer", unsafe {
            lock(self.encoder, &raw mut locked)
        })?;
        let guard = Locked::new(self, input, unlock, "nvEncUnlockInputBuffer");
        let pitch = usize::try_from(locked.pitch).unwrap_or(0);
        let chroma_rows = rows.div_ceil(2);
        let bytes = (pitch >= row_bytes && !locked.bufferDataPtr.is_null())
            .then(|| pitch.checked_mul(rows.checked_add(chroma_rows)?))
            .flatten()
            .ok_or_else(|| {
                Error::Unsupported(format!("NVENC's input buffer has a pitch of {pitch}"))
            });
        let copied = bytes.and_then(|bytes| {
            // SAFETY: while locked, NVENC's buffer holds the luma's `rows`
            // rows at `pitch`, then the chroma's half as many: the buffer
            // it made for this size and a 4:2:0 format, whose chroma
            // follows the luma at `pitch * rows`, as the SDK's samples and
            // ffmpeg lay it out. Its bytes may be uninitialised, so they
            // are lent as `MaybeUninit` and only written.
            let buffer = unsafe {
                std::slice::from_raw_parts_mut(
                    locked.bufferDataPtr.cast::<MaybeUninit<u8>>(),
                    bytes,
                )
            };
            let (luma_rows, chroma_plane) = buffer.split_at_mut(pitch * rows);
            copy_rows(luma_rows, pitch, luma, row_bytes, rows, "luma")?;
            copy_rows(
                chroma_plane,
                pitch,
                chroma,
                row_bytes,
                chroma_rows,
                "chroma",
            )
        });
        let unlocked = guard.unlock();
        copied.and(unlocked)?;
        Ok(locked.pitch)
    }

    /// Locks `output`, a bitstream buffer of the session that a submitted
    /// picture fills, once NVENC has filled it, and gives
    /// what NVENC says of it to `read`; then unlocks it, on every path.
    /// `read`'s error comes first, then the unlock's. The lock waits on the
    /// session's lock thread, until the deadline, past which the session is
    /// stalled (type docs). A buffer with no picture submitted into it since
    /// its last lock is refused, since NVENC's lock of it would not return
    /// ([`Submitted`]).
    pub(crate) fn lock_bitstream<T>(
        &self,
        output: Buffer,
        read: impl FnOnce(&ffi::NV_ENC_LOCK_BITSTREAM) -> Result<T, Error>,
    ) -> Result<T, Error> {
        let call = entry!(self.nvenc, nvEncLockBitstream)?;
        let unlock = entry!(self.nvenc, nvEncUnlockBitstream)?;
        let output = self.output(output, &[BufferKind::Bitstream])?;
        self.check_not_stalled()?;
        self.submitted.borrow().check_pending(output)?;
        let Lock { status, locked } = self.wait_for_lock(call, output)?;
        // Made, whatever it answers: a buffer is locked once a submission.
        self.submitted.borrow_mut().locked(output);
        self.check("nvEncLockBitstream", status)?;
        let _current = self.gpu.push()?;
        let guard = Locked::new(self, output, unlock, "nvEncUnlockBitstream");
        let read = read(&locked);
        let unlocked = guard.unlock();
        let read = read?;
        unlocked?;
        Ok(read)
    }

    /// A waiting lock of `output` with `call`, made on the lock thread and
    /// waited for until the deadline, past which the session is stalled;
    /// made here where the session has no lock thread.
    fn wait_for_lock(&self, call: LockCall, output: *mut c_void) -> Result<Lock, Error> {
        let Some(locks) = self
            .locks
            .get_or_init(|| Locks::start(self.encoder, &self.gpu))
        else {
            return locks::lock_now(call, self.encoder, &self.gpu, output);
        };
        let locking = locks.lock(call, output);
        match wait::answer_by(&locking, wait::deadline()) {
            Ok(lock) => lock,
            Err(_) => {
                *self.stuck.borrow_mut() = Some((output, locking));
                Err(Error::Stalled("NVENC's picture in time".to_owned()))
            }
        }
    }

    /// Unlocks `output`, a buffer of the session's it has locked, with the
    /// GPU's context pushed; what fails is logged.
    fn unlock_bitstream(&self, output: *mut c_void) {
        let Some(unlock) = self.nvenc.api.nvEncUnlockBitstream else {
            return;
        };
        let current = match self.gpu.push() {
            Ok(current) => current,
            Err(e) => return log::warn!("{e}"),
        };
        // SAFETY: a live session and one of its buffers, which it locked,
        // unlocked once.
        let status = unsafe { unlock(self.encoder, output) };
        drop(current);
        if let Err(e) = self.check("nvEncUnlockBitstream", status) {
            log::warn!("{e}");
        }
    }

    /// Ends the stream, where frames are submitted, and locks each buffer
    /// still submitted, in order, unlocking it at once, so that nothing is
    /// in flight as the session is destroyed (`encoder`'s `Drop`): for a
    /// stalled session's lock thread, where each lock waits with no bound.
    fn drain(&self) {
        let submitted = self.submitted.take().0;
        let Some(call) = self.nvenc.api.nvEncLockBitstream else {
            return;
        };
        let frames = submitted.iter().any(|&output| {
            self.buffers
                .iter()
                .any(|kept| kept.handle == output && kept.kind == BufferKind::Bitstream)
        });
        if frames && let Err(e) = self.encode_picture(Submission::End) {
            log::warn!("{e}");
        }
        for output in submitted {
            match locks::lock_now(call, self.encoder, &self.gpu, output) {
                Ok(Lock { status, .. }) if status == ffi::NV_ENC_SUCCESS => {
                    self.unlock_bitstream(output);
                }
                Ok(Lock { status, .. }) => {
                    if let Err(e) = self.check("nvEncLockBitstream", status) {
                        log::warn!("{e}");
                    }
                }
                Err(e) => log::warn!("{e}"),
            }
        }
    }

    /// Submits `submission` (`nvEncEncodePicture`), its frame at its input
    /// buffer's size and format, as a progressive frame. "Need more input",
    /// while NVENC holds frames back for B pictures, is an answer rather
    /// than an error.
    pub(crate) fn encode_picture(&self, submission: Submission) -> Result<NVENCSTATUS, Error> {
        let encode = entry!(self.nvenc, nvEncEncodePicture)?;
        self.check_not_stalled()?;
        let mut picture = ffi::NV_ENC_PIC_PARAMS::zeroed();
        picture.version = ffi::NV_ENC_PIC_PARAMS_VER;
        match submission {
            Submission::Frame {
                input,
                output,
                pitch,
                index,
                timestamp,
                idr,
            } => {
                let (input, width, height, format) = self.input(input)?;
                let output = self.output(output, &[BufferKind::Bitstream])?;
                self.submitted.borrow().check_free(output)?;
                picture.inputWidth = width;
                picture.inputHeight = height;
                picture.inputPitch = pitch;
                picture.inputBuffer = input;
                picture.outputBitstream = output;
                picture.bufferFmt = format;
                picture.pictureStruct = ffi::NV_ENC_PIC_STRUCT_FRAME;
                picture.inputTimeStamp = timestamp;
                // The header: monotonically increasing from 0 where
                // lookahead levels or the ultra high quality tuning are
                // used.
                picture.frameIdx = index;
                if idr {
                    picture.encodePicFlags =
                        ffi::NV_ENC_PIC_FLAG_FORCEIDR | ffi::NV_ENC_PIC_FLAG_OUTPUT_SPSPPS;
                }
            }
            Submission::End => picture.encodePicFlags = ffi::NV_ENC_PIC_FLAG_EOS,
        }
        let _current = self.gpu.push()?;
        // SAFETY: a live session; the parameters are live and versioned,
        // and name two of this session's own buffers, not locked, or none
        // for the end.
        let status = unsafe { encode(self.encoder, &raw mut picture) };
        if status != ffi::NV_ENC_ERR_NEED_MORE_INPUT {
            self.check("nvEncEncodePicture", status)?;
        }
        if !picture.outputBitstream.is_null() {
            self.submitted.borrow_mut().submit(picture.outputBitstream);
        }
        Ok(status)
    }

    /// The parameter sets (`nvEncGetSequenceParams`), Annex B, at most
    /// `limit` bytes of them.
    pub(crate) fn sequence_parameters(&self, limit: usize) -> Result<Vec<u8>, Error> {
        let get = entry!(self.nvenc, nvEncGetSequenceParams)?;
        let mut buffer = vec![0u8; limit];
        let mut written = 0u32;
        let mut payload = ffi::NV_ENC_SEQUENCE_PARAM_PAYLOAD::zeroed();
        payload.version = ffi::NV_ENC_SEQUENCE_PARAM_PAYLOAD_VER;
        payload.inBufferSize = u32::try_from(buffer.len()).unwrap_or(0);
        payload.spsppsBuffer = buffer.as_mut_ptr().cast();
        payload.outSPSPPSPayloadSize = &raw mut written;
        let _current = self.gpu.push()?;
        // SAFETY: a live session; the buffer holds `inBufferSize` bytes and
        // the size pointer is live.
        self.check("nvEncGetSequenceParams", unsafe {
            get(self.encoder, &raw mut payload)
        })?;
        buffer.truncate(usize::try_from(written).unwrap_or(0).min(buffer.len()));
        Ok(buffer)
    }

    /// `Ok` for success, else the error with this session's last text.
    pub(crate) fn check(&self, call: &'static str, status: NVENCSTATUS) -> Result<(), Error> {
        check_status(call, status, Some(self.encoder))
    }

    /// Fills a GUID list by its count and list entry points.
    fn guids(
        &self,
        what: &'static str,
        count: impl FnOnce(&mut u32) -> NVENCSTATUS,
        list: impl FnOnce(&mut [GUID], &mut u32) -> NVENCSTATUS,
    ) -> Result<Vec<GUID>, Error> {
        let mut n = 0u32;
        self.check(what, count(&mut n))?;
        let mut guids = vec![GUID::default(); usize::try_from(n).unwrap_or(0)];
        let mut written = 0u32;
        self.check(what, list(&mut guids, &mut written))?;
        guids.truncate(usize::try_from(written).unwrap_or(0));
        Ok(guids)
    }

    /// The codecs the session encodes (`nvEncGetEncodeGUIDs`).
    pub(crate) fn codec_guids(&self) -> Result<Vec<GUID>, Error> {
        let count = entry!(self.nvenc, nvEncGetEncodeGUIDCount)?;
        let list = entry!(self.nvenc, nvEncGetEncodeGUIDs)?;
        self.guids(
            "nvEncGetEncodeGUIDs",
            // SAFETY: a live session and a pointer to a live u32.
            |n| unsafe { count(self.encoder, n) },
            // SAFETY: the array holds `len` GUIDs, which bounds the write.
            |guids, n| unsafe {
                list(
                    self.encoder,
                    guids.as_mut_ptr(),
                    u32::try_from(guids.len()).unwrap_or(0),
                    n,
                )
            },
        )
    }

    /// The profiles of `codec` (`nvEncGetEncodeProfileGUIDs`).
    pub(crate) fn profile_guids(&self, codec: GUID) -> Result<Vec<GUID>, Error> {
        let count = entry!(self.nvenc, nvEncGetEncodeProfileGUIDCount)?;
        let list = entry!(self.nvenc, nvEncGetEncodeProfileGUIDs)?;
        self.guids(
            "nvEncGetEncodeProfileGUIDs",
            // SAFETY: a live session and a pointer to a live u32.
            |n| unsafe { count(self.encoder, codec, n) },
            // SAFETY: the array holds `len` GUIDs, which bounds the write.
            |guids, n| unsafe {
                list(
                    self.encoder,
                    codec,
                    guids.as_mut_ptr(),
                    u32::try_from(guids.len()).unwrap_or(0),
                    n,
                )
            },
        )
    }

    /// The presets of `codec` (`nvEncGetEncodePresetGUIDs`).
    pub(crate) fn preset_guids(&self, codec: GUID) -> Result<Vec<GUID>, Error> {
        let count = entry!(self.nvenc, nvEncGetEncodePresetCount)?;
        let list = entry!(self.nvenc, nvEncGetEncodePresetGUIDs)?;
        self.guids(
            "nvEncGetEncodePresetGUIDs",
            // SAFETY: a live session and a pointer to a live u32.
            |n| unsafe { count(self.encoder, codec, n) },
            // SAFETY: the array holds `len` GUIDs, which bounds the write.
            |guids, n| unsafe {
                list(
                    self.encoder,
                    codec,
                    guids.as_mut_ptr(),
                    u32::try_from(guids.len()).unwrap_or(0),
                    n,
                )
            },
        )
    }

    /// The input formats of `codec` (`nvEncGetInputFormats`).
    pub(crate) fn input_formats(
        &self,
        codec: GUID,
    ) -> Result<Vec<ffi::NV_ENC_BUFFER_FORMAT>, Error> {
        let count = entry!(self.nvenc, nvEncGetInputFormatCount)?;
        let list = entry!(self.nvenc, nvEncGetInputFormats)?;
        let mut n = 0u32;
        // SAFETY: a live session and a pointer to a live u32.
        self.check("nvEncGetInputFormatCount", unsafe {
            count(self.encoder, codec, &raw mut n)
        })?;
        let mut formats = vec![0; usize::try_from(n).unwrap_or(0)];
        let mut written = 0u32;
        // SAFETY: the array holds `len` formats, which bounds the write.
        self.check("nvEncGetInputFormats", unsafe {
            list(
                self.encoder,
                codec,
                formats.as_mut_ptr(),
                u32::try_from(formats.len()).unwrap_or(0),
                &raw mut written,
            )
        })?;
        formats.truncate(usize::try_from(written).unwrap_or(0));
        Ok(formats)
    }

    /// One capability of `codec` (`nvEncGetEncodeCaps`).
    pub(crate) fn capability(
        &self,
        codec: GUID,
        capability: ffi::NV_ENC_CAPS,
    ) -> Result<i32, Error> {
        let get = entry!(self.nvenc, nvEncGetEncodeCaps)?;
        let mut param = ffi::NV_ENC_CAPS_PARAM::zeroed();
        param.version = ffi::NV_ENC_CAPS_PARAM_VER;
        param.capsToQuery = capability;
        let mut value: c_int = 0;
        // SAFETY: a live session; the parameter and the value are live.
        self.check("nvEncGetEncodeCaps", unsafe {
            get(self.encoder, codec, &raw mut param, &raw mut value)
        })?;
        Ok(value)
    }

    /// The configuration a preset and tuning make for `codec`
    /// (`nvEncGetEncodePresetConfigEx`).
    pub(crate) fn preset_config(
        &self,
        codec: GUID,
        preset: GUID,
        tuning: ffi::NV_ENC_TUNING_INFO,
    ) -> Result<ffi::NV_ENC_PRESET_CONFIG, Error> {
        let get = entry!(self.nvenc, nvEncGetEncodePresetConfigEx)?;
        let mut config = ffi::NV_ENC_PRESET_CONFIG::zeroed();
        config.version = ffi::NV_ENC_PRESET_CONFIG_VER;
        config.presetCfg.version = ffi::NV_ENC_CONFIG_VER;
        // SAFETY: a live session; the config is live and versioned.
        self.check("nvEncGetEncodePresetConfigEx", unsafe {
            get(self.encoder, codec, preset, tuning, &raw mut config)
        })?;
        Ok(config)
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        let locks = self.locks.take().flatten();
        if let Some((output, locking)) = self.stuck.take() {
            self.finish_on(locks, output, locking);
            return;
        }
        drop(locks);
        let current = self.gpu.push().inspect_err(|e| log::warn!("{e}"));
        let api = &self.nvenc.api;
        // Its buffers first, as the header asks; none is locked, since each
        // lock is undone before the call that made it returns.
        for kept in &self.buffers {
            let destroy = match kept.kind {
                BufferKind::Input { .. } => api.nvEncDestroyInputBuffer,
                BufferKind::Bitstream => api.nvEncDestroyBitstreamBuffer,
            };
            let Some(destroy) = destroy else { continue };
            // SAFETY: a buffer of this live session, of the kind this entry
            // point destroys, destroyed once, as the session goes.
            let status = unsafe { destroy(self.encoder, kept.handle) };
            if let Err(e) = self.check("destroying an NVENC buffer", status) {
                log::warn!("{e}");
            }
        }
        let Some(destroy) = api.nvEncDestroyEncoder else {
            return;
        };
        // SAFETY: the session is live and destroyed once, after its
        // buffers. Its configuration is freed after this, as its field
        // drops.
        let status = unsafe { destroy(self.encoder) };
        drop(current);
        if let Err(e) = check_status("nvEncDestroyEncoder", status, None) {
            log::warn!("{e}");
        }
    }
}

impl Session {
    /// Hands the stalled session, whose lock of `output` is `locking`, to
    /// its lock thread to finish (`locks`), and keeps a record of it in
    /// `crate::wait`'s list until the thread has destroyed it. With no lock
    /// thread, which a stalled session always has, it is leaked whole.
    fn finish_on(&mut self, locks: Option<Locks>, output: *mut c_void, locking: Locking) {
        let submitted = self.submitted.take();
        let what = format!(
            "an NVENC session, its lock of a picture and {} submitted after it",
            submitted.0.len().saturating_sub(1)
        );
        let session = Session {
            nvenc: self.nvenc,
            encoder: self.encoder,
            gpu: Arc::clone(&self.gpu),
            config: self.config.take(),
            buffers: std::mem::take(&mut self.buffers),
            submitted: RefCell::new(submitted),
            locks: OnceCell::from(None),
            stuck: RefCell::new(None),
        };
        let Some(locks) = locks else {
            log::error!("{what}: no thread finishes it, and it is leaked");
            std::mem::forget(session);
            return;
        };
        let finished = locks.finish(Finishing {
            session,
            output,
            locking,
        });
        wait::keep(what, Box::new(finished));
    }
}

/// The output buffers a session submitted pictures into and has not locked
/// since, in submission order. Each is in it once, so that each submission
/// is locked once at most: a picture is submitted into a buffer only once
/// the last submitted into it is locked ([`Submitted::check_free`]), and a
/// buffer is locked only while one is ([`Submitted::check_pending`]).
/// NVENC's second lock of a buffer already unlocked, with no picture
/// submitted into it since, does not return (docs/video/bugs.md, U19).
#[derive(Debug, Default)]
struct Submitted(Vec<*mut c_void>);

impl Submitted {
    /// `Err` where a picture submitted into `output` is not locked yet.
    fn check_free(&self, output: *mut c_void) -> Result<(), Error> {
        if self.0.contains(&output) {
            return Err(Error::Unsupported(
                "a picture into an NVENC buffer whose last picture is not locked yet".to_owned(),
            ));
        }
        Ok(())
    }

    /// `Err` where no picture submitted into `output` waits for its lock.
    fn check_pending(&self, output: *mut c_void) -> Result<(), Error> {
        if !self.0.contains(&output) {
            return Err(Error::Unsupported(
                "a lock of an NVENC buffer with no picture submitted into it since its last"
                    .to_owned(),
            ));
        }
        Ok(())
    }

    /// Records a picture submitted into `output`, which
    /// [`Submitted::check_free`] allowed.
    fn submit(&mut self, output: *mut c_void) {
        self.0.push(output);
    }

    /// Records the lock of `output` made.
    fn locked(&mut self, output: *mut c_void) {
        self.0.retain(|&submitted| submitted != output);
    }
}

/// The refusal of `buffer`, of another kind than a call takes.
fn not_for_the_call(buffer: Buffer) -> Error {
    Error::Unsupported(format!(
        "{buffer:?} is not a buffer of the NVENC session's of the kind the call takes"
    ))
}

/// One of a session's buffers, locked, unlocked once: by
/// [`Locked::unlock`], with NVENC's answer, or as it drops on any other
/// path, a panic's unwinding included.
struct Locked<'a> {
    session: &'a Session,
    buffer: *mut c_void,
    unlock: unsafe extern "C" fn(ffi::Encoder, *mut c_void) -> NVENCSTATUS,
    call: &'static str,
    locked: bool,
}

impl<'a> Locked<'a> {
    /// `buffer`, which `session` has just locked, and the entry point
    /// `call` that unlocks it.
    fn new(
        session: &'a Session,
        buffer: *mut c_void,
        unlock: unsafe extern "C" fn(ffi::Encoder, *mut c_void) -> NVENCSTATUS,
        call: &'static str,
    ) -> Self {
        Self {
            session,
            buffer,
            unlock,
            call,
            locked: true,
        }
    }

    /// Unlocks it, with NVENC's answer.
    fn unlock(mut self) -> Result<(), Error> {
        self.release()
    }

    fn release(&mut self) -> Result<(), Error> {
        if !std::mem::take(&mut self.locked) {
            return Ok(());
        }
        // SAFETY: the buffer the session locked, unlocked once (the flag)
        // by the entry point of its kind.
        let status = unsafe { (self.unlock)(self.session.encoder, self.buffer) };
        self.session.check(self.call, status)
    }
}

impl Drop for Locked<'_> {
    fn drop(&mut self) {
        if let Err(e) = self.release() {
            log::warn!("{e}");
        }
    }
}

/// Copies `rows` rows of `row_bytes` from `source`, its rows its stride
/// apart, into `destination` at `pitch`; `PlaneTooSmall` where either does
/// not hold them, before anything is copied.
fn copy_rows(
    destination: &mut [MaybeUninit<u8>],
    pitch: usize,
    (source, stride): (&[u8], usize),
    row_bytes: usize,
    rows: usize,
    plane: &'static str,
) -> Result<(), Error> {
    if rows == 0 {
        return Ok(());
    }
    let holds = |len: usize, stride: usize| {
        stride >= row_bytes
            && (rows - 1)
                .checked_mul(stride)
                .and_then(|n| n.checked_add(row_bytes))
                .is_some_and(|needed| len >= needed)
    };
    if !holds(source.len(), stride) || !holds(destination.len(), pitch) {
        return Err(Error::PlaneTooSmall(plane));
    }
    for row in 0..rows {
        let from = &source[row * stride..][..row_bytes];
        destination[row * pitch..][..row_bytes].write_copy_of_slice(from);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn written(destination: &[MaybeUninit<u8>]) -> Vec<u8> {
        destination
            .iter()
            // SAFETY: the test's buffers start initialised.
            .map(|byte| unsafe { byte.assume_init() })
            .collect()
    }

    #[test]
    fn rows_copy_at_the_pitch_and_short_planes_are_refused() {
        let source: Vec<u8> = (0..12).collect();
        let mut destination = [MaybeUninit::new(0u8); 16];
        assert_eq!(
            copy_rows(&mut destination, 8, (&source, 6), 4, 2, "luma"),
            Ok(())
        );
        let bytes = written(&destination);
        assert_eq!(bytes[..4], [0, 1, 2, 3]);
        assert_eq!(bytes[4..8], [0; 4]);
        assert_eq!(bytes[8..12], [6, 7, 8, 9]);
        assert_eq!(
            copy_rows(&mut destination, 8, (&source[..9], 6), 4, 2, "luma"),
            Err(Error::PlaneTooSmall("luma"))
        );
        assert_eq!(
            copy_rows(&mut destination, 8, (&source, 3), 4, 2, "chroma"),
            Err(Error::PlaneTooSmall("chroma"))
        );
    }

    #[test]
    fn a_destination_short_of_its_rows_is_refused_untouched() {
        let source = [7u8; 12];
        let mut destination = [MaybeUninit::new(0u8); 11];
        assert_eq!(
            copy_rows(&mut destination, 8, (&source, 4), 4, 2, "luma"),
            Err(Error::PlaneTooSmall("luma"))
        );
        assert_eq!(written(&destination), [0; 11]);
        let mut destination = [MaybeUninit::new(0u8); 12];
        assert_eq!(
            copy_rows(&mut destination, 8, (&source, 4), 4, 2, "luma"),
            Ok(())
        );
        assert_eq!(written(&destination), [7, 7, 7, 7, 0, 0, 0, 0, 7, 7, 7, 7]);
    }

    #[test]
    fn a_buffer_is_submitted_into_and_locked_once_a_picture() {
        let [first, second] = [0x10, 0x20].map(std::ptr::without_provenance_mut::<c_void>);
        let mut submitted = Submitted::default();
        assert!(submitted.check_pending(first).is_err(), "nothing submitted");
        assert_eq!(submitted.check_free(first), Ok(()));
        submitted.submit(first);
        assert!(submitted.check_free(first).is_err(), "submitted twice");
        assert_eq!(submitted.check_free(second), Ok(()));
        submitted.submit(second);
        assert_eq!(submitted.check_pending(first), Ok(()));
        submitted.locked(first);
        assert!(submitted.check_pending(first).is_err(), "locked twice");
        assert_eq!(submitted.check_free(first), Ok(()));
        assert_eq!(submitted.0, [second]);
    }

    #[test]
    fn statuses_are_named_by_value() {
        assert_eq!(
            ffi::NVENCSTATUS_NAMES.get(usize::try_from(ffi::NV_ENC_ERR_NEED_MORE_INPUT).unwrap()),
            Some(&"NV_ENC_ERR_NEED_MORE_INPUT")
        );
        assert_eq!(
            check_status("call", 21, None).map_err(|e| e.to_string()),
            Err("call failed: NV_ENC_ERR_INCOMPATIBLE_CLIENT_KEY (21)".to_owned())
        );
        assert!(check_status("call", ffi::NV_ENC_SUCCESS, None).is_ok());
    }
}
