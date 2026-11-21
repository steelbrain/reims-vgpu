//! A VA display on a DRM render node, or on Windows on a DXGI adapter, and
//! libva's errors.

use std::ffi::CStr;
use std::fs::File;
#[cfg(unix)]
use std::fs::OpenOptions;
#[cfg(unix)]
use std::os::fd::AsRawFd;
#[cfg(unix)]
use std::path::Path;
#[cfg(windows)]
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use crate::core::runtime::{ExitStep, MonoTime, Runtime, park_for_good};
use crate::core::sync::Mutex;

use super::calls::{Calls, Late};
use super::caps::Answers;
#[cfg(windows)]
use super::caps::{ContextShape, KeptContext};
use super::ffi;
#[cfg(windows)]
use super::held::{Asked, Held, Kept};
use super::import_check::{ImportWrite, ImportWrites};
use super::surface::SurfaceFormat;
use super::wait;

/// Why a VA-API operation failed.
#[derive(Debug)]
pub enum Error {
    /// libva or libva-drm is not on this host, or lacks a function the
    /// binding calls (`super::libva`).
    Load(super::libva::LoadError),
    /// The render node could not be opened.
    Open(std::io::Error),
    /// libva could not make a display on the render node.
    NoDisplay,
    /// A libva call failed, with its status and libva's text for it.
    Va {
        /// The function.
        call: &'static str,
        /// Its `VAStatus`.
        status: i32,
        /// `vaErrorStr` of the status.
        message: String,
    },
    /// The stream is malformed, or uses what the decoder refuses.
    Bitstream(crate::codec::h264::Error),
    /// The HEVC stream is malformed, or uses what the decoder refuses.
    HevcBitstream(crate::codec::hevc::Error),
    /// The driver cannot decode this stream: what is missing, in words.
    Unsupported(String),
    /// The caller holds every surface the decoder may make.
    OutOfSurfaces,
    /// A destination plane is too small for its stride and the frame.
    PlaneTooSmall(&'static str),
    /// The GPU did not finish work by its deadline, or an earlier frame's
    /// of the same decoder or encoder (`super::wait`): which,
    /// in words.
    Stalled(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Load(e) => write!(f, "{e}"),
            Self::Open(e) => write!(f, "cannot open the render node: {e}"),
            #[cfg(unix)]
            Self::NoDisplay => write!(f, "vaGetDisplayDRM returned no display"),
            #[cfg(windows)]
            Self::NoDisplay => write!(f, "vaGetDisplayWin32 returned no display"),
            Self::Va {
                call,
                status,
                message,
            } => write!(f, "{call} failed: {message} ({status:#x})"),
            Self::Bitstream(e) => write!(f, "{e}"),
            Self::HevcBitstream(e) => write!(f, "{e}"),
            Self::Unsupported(what) => write!(f, "the driver cannot: {what}"),
            Self::OutOfSurfaces => write!(f, "every decode surface is held"),
            Self::PlaneTooSmall(plane) => write!(f, "the {plane} plane is too small"),
            Self::Stalled(what) => write!(f, "the GPU did not finish {what}"),
        }
    }
}

impl Error {
    /// Whether the error is one picture's of the stream, after which the decoder goes on (the
    /// codec crate's `is_picture_error`, H.264's and HEVC's): the rest are the decoder's own or
    /// the driver's.
    pub fn is_picture_error(&self) -> bool {
        match self {
            Self::Bitstream(e) => e.is_picture_error(),
            Self::HevcBitstream(e) => e.is_picture_error(),
            _ => false,
        }
    }
}

impl std::error::Error for Error {}

impl From<crate::codec::h264::Error> for Error {
    fn from(e: crate::codec::h264::Error) -> Self {
        Self::Bitstream(e)
    }
}

impl From<crate::codec::hevc::Error> for Error {
    fn from(e: crate::codec::hevc::Error) -> Self {
        Self::HevcBitstream(e)
    }
}

/// `Ok` for `VA_STATUS_SUCCESS`, else the error with libva's text.
pub(crate) fn check(call: &'static str, status: ffi::VAStatus) -> Result<(), Error> {
    // The session-start probe's timeline of the calls (`crate::probes::timeline`).
    #[cfg(all(test, target_os = "linux"))]
    crate::probes::timeline::stamp(call);
    if status == ffi::VA_STATUS_SUCCESS {
        return Ok(());
    }
    // SAFETY: vaErrorStr takes any status and returns a static string
    // (va.h, "Returns a short english description of error_status").
    let text = unsafe { ffi::vaErrorStr(status) };
    let message = if text.is_null() {
        String::new()
    } else {
        // SAFETY: non-null, and NUL-terminated static storage.
        unsafe { CStr::from_ptr(text) }
            .to_string_lossy()
            .into_owned()
    };
    Err(Error::Va {
        call,
        status,
        message,
    })
}

/// An initialised VA display on a render node. Another handle to the
/// same display is taken with [`Display::clone_ref`]; the display is
/// terminated when the last handle, and everything made on it, is gone.
///
/// Every driver call on it is made through [`Display::call`] (or
/// [`Display::try_call`]), [`Display::wait_call`] or [`Display::release`],
/// each bounded by the deadline of the frame in hand ([`super::calls`]).
pub struct Display {
    inner: Arc<DisplayInner>,
}

struct DisplayInner {
    /// Its driver calls, and the thread that makes those that may wait for
    /// the GPU.
    calls: Calls,
    version: (i32, i32),
    vendor: String,
    path: String,
    /// What the driver answered to each query, kept from the opening
    /// thread on (`super::caps`, [`Display::prime`]).
    answers: Mutex<Answers>,
    /// The render node, open for as long as the display lives: until its
    /// `vaTerminate`.
    node: Option<File>,
    /// The host's threads and clock, which what is made on the display
    /// starts its threads on and times by.
    runtime: Arc<dyn Runtime>,
    /// What the driver's video processing writes into imported memory, as
    /// the check at the opening found it (`super::import_check`).
    import_writes: OnceLock<ImportWrites>,
    /// On Windows, the LUID of the DXGI adapter the display was opened on
    /// (`None`: the adapter the driver picked), and the Direct3D 12 device
    /// on it that the display's pools make their textures with
    /// (`super::d3d12`), made at its first use.
    #[cfg(windows)]
    adapter: Option<[u8; 8]>,
    #[cfg(windows)]
    d3d12: OnceLock<Result<super::d3d12::Device, String>>,
    /// On Windows, the display's live contexts, and the contexts' teardowns held and the
    /// decoders' contexts kept while one lives ([`Display::release_teardown`],
    /// [`Display::context_let_go`]).
    #[cfg(windows)]
    teardowns: Mutex<Held<HeldRelease, KeptContext>>,
}

/// A teardown held on Windows ([`Display::release_teardown`]).
#[cfg(windows)]
type HeldRelease = Box<dyn FnOnce(ffi::VADisplay) + Send>;

/// What a teardown ends ([`Display::release_teardown`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Teardown {
    /// A context (`vaDestroyContext`).
    Context,
    /// A config (`vaDestroyConfig`).
    Config,
    /// A pool's surfaces (`vaDestroySurfaces`), and on Windows the textures they are.
    Surfaces,
}

/// Whether a context's teardown was held, or a decoder's context kept, yet in this process,
/// which the log says once.
#[cfg(windows)]
static HELD_LOGGED: AtomicBool = AtomicBool::new(false);

// SAFETY: "All VAAPI functions implemented in libva are thread-safe" and
// a backend's must be too (va.h, "Multithreading Guide"). The handle is
// only passed to libva.
unsafe impl Send for DisplayInner {}
// SAFETY: as above.
unsafe impl Sync for DisplayInner {}

impl Drop for DisplayInner {
    fn drop(&mut self) {
        let node = self.node.take();
        // Behind every call the call thread has, the last of them.
        self.calls.release(move |raw| {
            // SAFETY: the display was initialised, and every object made on
            // it holds a handle, so none is left to use it but the releases
            // made before this.
            unsafe { ffi::vaTerminate(raw) };
            drop(node);
        });
    }
}

impl Display {
    /// Opens a DRM render node (`/dev/dri/renderD128`, say) and
    /// initialises VA-API on it, loading libva first where no display has
    /// yet: a host without it fails here, with [`Error::Load`]. Then asks
    /// the driver, on this thread, what a session will ask of it
    /// ([`Display::prime`]), and checks what its video processing writes
    /// into imported memory ([`Display::import_writes`]). libva's
    /// informational messages are silenced; its errors still go to stderr.
    /// What is made on it runs on `runtime`. Once the process's exit began,
    /// the thread parks for good before any call of libva's, as before each
    /// call of the display's (`super::calls`, "The process's exit").
    #[cfg(unix)]
    pub fn open(path: impl AsRef<Path>, runtime: Arc<dyn Runtime>) -> Result<Self, Error> {
        let path = path.as_ref();
        // The open's own calls of libva, `vaInitialize` and a failed one's `vaTerminate` among
        // them, are one step of the host's exit gate, as each call the display's `Calls` makes is
        // (`super::calls`, "The process's exit"): once the process's exit began, the thread
        // parks here for good, and an exit that begins during them waits for them to end.
        let Some(step) = runtime.exit_step() else {
            park_for_good()
        };
        super::libva::load().map_err(Error::Load)?;
        let node = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(Error::Open)?;
        // SAFETY: the descriptor is open, and stays open in `_node` for the
        // display's life.
        let raw = unsafe { ffi::vaGetDisplayDRM(node.as_raw_fd()) };
        if raw.is_null() {
            return Err(Error::NoDisplay);
        }
        Self::initialise(
            raw,
            path.display().to_string(),
            Some(node),
            None,
            runtime,
            step,
        )
    }

    /// Initialises VA-API on the DXGI adapter whose LUID is `luid` (the
    /// Vulkan device's `deviceLUID`), as `Display::open` does on a render
    /// node: libva-win32 picks the adapter's driver, Mesa's VA-on-D3D12
    /// (`vaon12`) where the adapter's registry names none. `None` is the
    /// adapter the driver picks, for the examples. No import check runs:
    /// Windows makes no dma-buf, so the driver is lent no import
    /// ([`Display::import_writes`] says so) and frames come back through
    /// images.
    #[cfg(windows)]
    pub fn open_adapter(luid: Option<[u8; 8]>, runtime: Arc<dyn Runtime>) -> Result<Self, Error> {
        let Some(step) = runtime.exit_step() else {
            park_for_good()
        };
        super::libva::load().map_err(Error::Load)?;
        let adapter = luid
            .as_ref()
            .map_or(core::ptr::null(), |luid| luid.as_ptr().cast());
        // SAFETY: `adapter` is null or a LUID's 8 bytes (`LowPart` then
        // `HighPart`, little-endian, as Vulkan's `deviceLUID` holds them),
        // read during the call alone.
        let raw = unsafe { ffi::vaGetDisplayWin32(adapter) };
        if raw.is_null() {
            return Err(Error::NoDisplay);
        }
        let name = match luid {
            Some(luid) => format!("the DXGI adapter of LUID {:#x}", u64::from_le_bytes(luid)),
            None => "the DXGI adapter the driver picks".to_owned(),
        };
        Self::initialise(raw, name, None, luid, runtime, step)
    }

    /// The Direct3D 12 device on the display's adapter, which its pools
    /// make their shared textures with (`super::d3d12`); `Err` with why
    /// where none is made.
    #[cfg(windows)]
    pub(crate) fn d3d12(&self) -> Result<&super::d3d12::Device, String> {
        super::d3d12::device_of(&self.inner.d3d12, self.inner.adapter)
    }

    /// The rest of an open, on the display `raw` libva made: `vaInitialize`,
    /// the driver's answers and, where Linux makes dma-bufs, the import
    /// check. `step` is the open's step of the host's exit gate, ended once
    /// libva's own calls of the open are made.
    fn initialise(
        raw: ffi::VADisplay,
        path: String,
        node: Option<File>,
        adapter: Option<[u8; 8]>,
        runtime: Arc<dyn Runtime>,
        step: ExitStep,
    ) -> Result<Self, Error> {
        #[cfg(unix)]
        let _ = adapter;
        // SAFETY: `raw` is a display; a null callback disables the
        // messages (va.h, vaSetInfoCallback).
        unsafe { ffi::vaSetInfoCallback(raw, None, core::ptr::null_mut()) };
        let (mut major, mut minor) = (0, 0);
        // SAFETY: `raw` is a display, and the out-pointers are valid.
        let status = unsafe { ffi::vaInitialize(raw, &mut major, &mut minor) };
        // vaInitialize loaded the driver, which may have registered exit-time destructors: the
        // host's exit gate runs its handler before them (`Runtime::arm_exit`), and so after
        // every call of the display's still in progress (`super::calls`).
        runtime.arm_exit();
        if let Err(e) = check("vaInitialize", status) {
            // SAFETY: vaTerminate also frees a display that failed to
            // initialise, as libva's own tools do.
            unsafe { ffi::vaTerminate(raw) };
            return Err(e);
        }
        // SAFETY: an initialised display; the string lives as long as it.
        let vendor = unsafe { ffi::vaQueryVendorString(raw) };
        let vendor = if vendor.is_null() {
            String::new()
        } else {
            // SAFETY: non-null and NUL-terminated.
            unsafe { CStr::from_ptr(vendor) }
                .to_string_lossy()
                .into_owned()
        };
        drop(step);
        let display = Self {
            inner: Arc::new(DisplayInner {
                calls: Calls::new(raw, Arc::clone(&runtime)),
                version: (major, minor),
                vendor,
                path,
                answers: Mutex::new(Answers::default()),
                node,
                runtime,
                import_writes: OnceLock::new(),
                #[cfg(windows)]
                adapter,
                #[cfg(windows)]
                d3d12: OnceLock::new(),
                #[cfg(windows)]
                teardowns: Mutex::new(Held::new()),
            }),
        };
        display.prime();
        let writes = super::import_check::learn(&display);
        let _ = display.inner.import_writes.set(writes);
        Ok(display)
    }

    /// The driver's answers kept so far (`super::caps`).
    pub(crate) fn answers_lock(&self) -> &Mutex<Answers> {
        &self.inner.answers
    }

    /// Another handle to this display.
    pub fn clone_ref(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }

    /// Whether `other` is a handle to this display.
    pub(crate) fn is_same(&self, other: &Display) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// The runtime it was opened on.
    pub fn runtime(&self) -> &Arc<dyn Runtime> {
        &self.inner.runtime
    }

    /// The driver's vendor string (`vaQueryVendorString`).
    pub fn vendor(&self) -> &str {
        &self.inner.vendor
    }

    /// What the driver's video processing writes into memory imported as a
    /// linear surface, as the display's check at its opening found it
    /// (`super::import_check`).
    pub fn import_writes(&self) -> ImportWrites {
        (self.inner.import_writes.get().cloned())
            .unwrap_or_else(|| ImportWrites::none("the display's import check has not run"))
    }

    /// What of an import of `format` the driver's video processing writes
    /// right, as the display's check found it: the staged copy back's
    /// ([`super::readback`]) and zero copy's ([`super::zero_copy`]) memory.
    /// `Err` says why the driver is lent no such import.
    pub(crate) fn writes_imports(&self, format: SurfaceFormat) -> Result<ImportWrite, String> {
        match self.inner.import_writes.get() {
            Some(writes) => writes.of(format),
            None => Err("the display's import check has not run".to_owned()),
        }
    }

    /// The VA-API version libva reports, major and minor.
    pub fn api_version(&self) -> (i32, i32) {
        self.inner.version
    }

    /// The render node the display is on.
    pub fn path(&self) -> &str {
        &self.inner.path
    }

    /// `call`, with the display's handle, made on this thread by the
    /// deadline of the frame in hand (`wait::current`), which a wait for
    /// the display sets where none has: [`Error::Stalled`] where a call
    /// that may wait for the GPU holds the display past it
    /// ([`super::calls`]).
    pub(crate) fn call<T>(&self, call: impl FnOnce(ffi::VADisplay) -> T) -> Result<T, Error> {
        self.inner
            .calls
            .call(|| wait::current(self), call)
            .map_err(|_| held())
    }

    /// [`Display::call`], by `deadline`.
    pub(crate) fn call_by<T>(
        &self,
        deadline: MonoTime,
        call: impl FnOnce(ffi::VADisplay) -> T,
    ) -> Result<T, Error> {
        self.inner.calls.call(|| deadline, call).map_err(|_| held())
    }

    /// `call`, made on this thread now, where no call that may wait for
    /// the GPU holds the display: `None` where one does.
    pub(crate) fn try_call<T>(&self, call: impl FnOnce(ffi::VADisplay) -> T) -> Option<T> {
        self.inner.calls.try_call(call)
    }

    /// `call`, a call that may wait for the GPU, made on the display's
    /// call thread; its answer waited for until the deadline of the frame
    /// in hand (`wait::current`). Past it [`Error::Stalled`], with `what`,
    /// and what the call gives once it returns goes to `late`, there.
    /// Whatever the call uses must outlive it: each destroy the display
    /// makes ([`Display::release`]) is made behind it.
    pub(crate) fn wait_call<T: Send + 'static>(
        &self,
        what: &str,
        call: impl FnOnce(ffi::VADisplay) -> T + Send + 'static,
        late: impl FnOnce(ffi::VADisplay, T) + Send + 'static,
    ) -> Result<T, Error> {
        self.wait_call_by(wait::current(self), what, call, late)
    }

    /// [`Display::wait_call`], until `deadline`.
    pub(crate) fn wait_call_by<T: Send + 'static>(
        &self,
        deadline: MonoTime,
        what: &str,
        call: impl FnOnce(ffi::VADisplay) -> T + Send + 'static,
        late: impl FnOnce(ffi::VADisplay, T) + Send + 'static,
    ) -> Result<T, Error> {
        self.inner
            .calls
            .wait_call(deadline, call, late)
            .map_err(|late| match late {
                Late::NotMade => held(),
                Late::InFlight => Error::Stalled(format!("{what} in time")),
            })
    }

    /// `release`, a destroy or an unmap: made now where no call that may
    /// wait for the GPU holds the display, else behind it, on the call
    /// thread. Never waited for.
    pub(crate) fn release(&self, release: impl FnOnce(ffi::VADisplay) + Send + 'static) {
        self.inner.calls.release(release);
    }

    /// Counts a context made on the display, which on Windows holds the contexts' teardowns
    /// asked for while it lives ([`Display::release_teardown`]).
    pub(crate) fn context_made(&self) {
        #[cfg(windows)]
        {
            self.inner.teardowns.lock().context_made();
        }
    }

    /// `release`, the teardown `kind`, made as [`Display::release`] makes a release.
    ///
    /// On Windows a context's is held while another context lives on the display, and made,
    /// behind the teardowns held and the contexts kept before it (`Display::context_let_go`),
    /// once the last live context's end is asked for. **Measured** on the UHD Graphics 630
    /// (vaon12 of Mesa 26.2.7 over Intel's Direct3D 12 video driver, 2026-11-17), two threads
    /// each decoding a 4K H.264 stream on one display, their decoders let go round after round
    /// or kept (`super::teardown`): a decoder let go while another lived ended the process in an
    /// access violation inside Intel's `igd12dxva64.dll` at the other's next `vaEndPicture`, or
    /// had that call fail, in every run of every shape tried: with the other decoder idle and its
    /// last picture done and read back; with no teardown between another context's
    /// `vaBeginPicture` and its `vaEndPicture`'s return; with each context's pictures submitted
    /// alone, once the other contexts' last pictures were done. With teardowns held until no
    /// context lived, and with the decoders kept until both threads ended, every run passed, as
    /// one thread making and letting go decoders one after another did. **Measured** again
    /// there (2026-11-20, each way run twice), with the context's teardown alone held: a pool's
    /// surfaces destroyed at once, or its surfaces and the config, passed as well. So a config's
    /// and a pool's surfaces' teardowns are made at once. `super::held` counts the live contexts
    /// and keeps what is held.
    pub(crate) fn release_teardown(
        &self,
        kind: Teardown,
        release: impl FnOnce(ffi::VADisplay) + Send + 'static,
    ) {
        #[cfg(windows)]
        {
            let asked = self.inner.teardowns.lock().ask(kind, Box::new(release));
            self.made(asked);
        }
        #[cfg(unix)]
        {
            let _ = kind;
            self.release(release);
        }
    }

    /// A decoder's context let go (`Context::new_decoder`): kept while another context lives
    /// on the display, for a decoder of its shape to take back ([`Display::take_context`]),
    /// else destroyed, behind what was held and kept before it, as a context's teardown is
    /// made ([`Display::release_teardown`]). A context kept holds what its decoder holds, and
    /// is taken back rather than another made beside it, so what the contexts kept hold is
    /// bounded by the most decoders of one shape that lived at once.
    #[cfg(windows)]
    pub(crate) fn context_let_go(&self, context: KeptContext) {
        let asked = self.inner.teardowns.lock().let_go(context);
        self.made(asked);
    }

    /// A decoder's context of `shape` kept on the display ([`Display::context_let_go`]), taken
    /// back and counted live.
    #[cfg(windows)]
    pub(crate) fn take_context(&self, shape: &ContextShape) -> Option<KeptContext> {
        self.inner
            .teardowns
            .lock()
            .take(|kept| kept.shape == *shape)
    }

    /// What [`Held`] made of what was let go: nothing yet where it was kept, else each release
    /// made and each context kept destroyed, in order.
    #[cfg(windows)]
    fn made(&self, asked: Asked<HeldRelease, KeptContext>) {
        match asked {
            Asked::Kept => {
                if !HELD_LOGGED.swap(true, Ordering::Relaxed) {
                    log::info!(
                        "hardware video: VA-API: a session's context is kept while another \
                         session lives on the display, and destroyed once none does"
                    );
                }
            }
            Asked::Now(now) => {
                for kept in now {
                    match kept {
                        Kept::Release(release) => self.release(release),
                        Kept::Context(context) => context.destroy(self),
                    }
                }
            }
        }
    }
}

/// The error of a call not made by its deadline, the display held by a
/// call that waits for the GPU.
fn held() -> Error {
    Error::Stalled("a driver call ahead of this one on the display in time".into())
}

#[cfg(test)]
#[cfg(unix)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc;
    use std::time::Duration;

    use super::super::gated::{GatedRuntime, close_the_gate};
    use crate::core::runtime::StdRuntime;

    use super::*;

    /// The child's role in [`an_open_once_the_exit_began_makes_no_call_of_libva`].
    const OPEN_CHILD: &str = "RVG_TEST_DISPLAY_OPEN_CHILD";

    /// One picture's errors are the codec crate's of a picture, H.264's and HEVC's; what the
    /// decoder does not take, a record's errors and the driver's are not.
    #[test]
    fn one_picture_s_errors_are_the_codec_s_of_a_picture() {
        use crate::codec::{h264, hevc};
        for error in [
            Error::Bitstream(h264::Error::MissingReference),
            Error::Bitstream(h264::Error::Unrecovered),
            Error::HevcBitstream(hevc::Error::MissingReference),
            Error::HevcBitstream(hevc::Error::SkippedLeadingPicture),
        ] {
            assert!(error.is_picture_error(), "{error}");
        }
        for error in [
            Error::Bitstream(h264::Error::Unsupported("data partitioning")),
            Error::Bitstream(h264::Error::AvccVersion),
            Error::HevcBitstream(hevc::Error::HvccVersion),
            Error::Unsupported("4:2:2".to_owned()),
            Error::OutOfSurfaces,
            Error::Stalled("a decode".to_owned()),
        ] {
            assert!(!error.is_picture_error(), "{error}");
        }
    }

    /// An open begun once the process's exit closed the host's gate makes no call of libva's:
    /// its thread parks before it loads libva, where it would otherwise fail at once, no node
    /// being at the path. Run in a child process of its own, which closes the gate.
    #[test]
    fn an_open_once_the_exit_began_makes_no_call_of_libva() {
        if std::env::var_os(OPEN_CHILD).is_some() {
            close_the_gate();
            let (done, opened) = mpsc::channel();
            StdRuntime::thread("test-exit-open", move || {
                let open = Display::open("/nonexistent/renderD128", Arc::new(GatedRuntime));
                let _ = done.send(open.is_ok());
            })
            .expect("a thread");
            let code = match opened.recv_timeout(Duration::from_millis(300)) {
                Err(mpsc::RecvTimeoutError::Timeout) => 0,
                _ => 3,
            };
            std::process::exit(code);
        }
        let name = "vaapi::va::display::tests::an_open_once_the_exit_began_makes_no_call_of_libva";
        let status = std::process::Command::new(std::env::current_exe().expect("the test"))
            .args(["--exact", name, "--nocapture", "--test-threads=1"])
            .env(OPEN_CHILD, "1")
            .status()
            .expect("a child");
        assert!(
            status.success(),
            "the open went on past the closed gate (3): {status:?}"
        );
    }

    /// The child's role in
    /// [`a_display_let_go_once_its_drivers_exit_time_destructors_ran_calls_nothing`].
    const LATE_CHILD: &str = "RVG_TEST_DISPLAY_LATE_CHILD";

    /// Set by the late exit handler, which has the display let go.
    static LET_GO: AtomicBool = AtomicBool::new(false);
    /// Set once the display's drop returned, its `vaTerminate` made.
    static DROPPED: AtomicBool = AtomicBool::new(false);

    /// An exit handler registered before the display opens, so that it runs after the exit-time
    /// destructors its driver registered as `vaInitialize` loaded it: has the display let go,
    /// and ends the process with 3 where the drop returned within a second.
    extern "C" fn let_go_late() {
        LET_GO.store(true, Ordering::SeqCst);
        for _ in 0..1000 {
            if DROPPED.load(Ordering::SeqCst) {
                // SAFETY: ends the process at once, which is the point.
                unsafe { libc::_exit(3) };
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    /// A display let go on another thread once the process's exit ran its driver's exit-time
    /// destructors, as a device's display was when a test process exited as the device's thread
    /// let it go: its last `vaTerminate` then aborted in iHD's frees ("double free or corruption
    /// (out)", "free(): invalid pointer") on the Iris Xe. With the host's gate armed as the
    /// display opened, the gate closed before those destructors ran, and the drop parks its
    /// thread before any call. Run in child processes of their own, which exit.
    #[test]
    #[ignore = "needs a VA-API GPU: run on the VA-API host"]
    fn a_display_let_go_once_its_drivers_exit_time_destructors_ran_calls_nothing() {
        let node = std::env::var("RVG_VAAPI_NODE").unwrap_or_else(|_| "/dev/dri/renderD128".into());
        if std::env::var_os(LATE_CHILD).is_some() {
            // SAFETY: a plain function that never unwinds.
            assert_eq!(unsafe { libc::atexit(let_go_late) }, 0);
            let display = Display::open(&node, Arc::new(GatedRuntime)).expect("a display");
            StdRuntime::thread("test-late-drop", move || {
                while !LET_GO.load(Ordering::SeqCst) {
                    std::thread::sleep(Duration::from_millis(1));
                }
                drop(display);
                DROPPED.store(true, Ordering::SeqCst);
            })
            .expect("a thread");
            std::process::exit(0);
        }
        let name = "vaapi::va::display::tests::\
                    a_display_let_go_once_its_drivers_exit_time_destructors_ran_calls_nothing";
        for _ in 0..5 {
            let child = std::process::Command::new(std::env::current_exe().expect("the test"))
                .args([
                    "--exact",
                    name,
                    "--nocapture",
                    "--test-threads=1",
                    "--ignored",
                ])
                .env(LATE_CHILD, "1")
                .output()
                .expect("a child");
            let stderr = String::from_utf8_lossy(&child.stderr);
            assert!(
                child.status.success(),
                "{node}: the display's drop made its vaTerminate after the driver's exit-time \
                 destructors (3), or the child ended otherwise: {:?}\n{stderr}",
                child.status
            );
        }
    }
}
