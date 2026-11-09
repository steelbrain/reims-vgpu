//! A VA display on a DRM render node, and libva's errors.

use std::ffi::CStr;
use std::fs::{File, OpenOptions};
use std::os::fd::AsRawFd;
use std::path::Path;
use std::sync::{Arc, OnceLock};

use crate::core::runtime::{MonoTime, Runtime, park_for_good};
use crate::core::sync::Mutex;

use super::calls::{Calls, Late};
use super::caps::Answers;
use super::ffi;
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
            Self::NoDisplay => write!(f, "vaGetDisplayDRM returned no display"),
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
}

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
                path: path.display().to_string(),
                answers: Mutex::new(Answers::default()),
                node: Some(node),
                runtime,
                import_writes: OnceLock::new(),
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
}

/// The error of a call not made by its deadline, the display held by a
/// call that waits for the GPU.
fn held() -> Error {
    Error::Stalled("a driver call ahead of this one on the display in time".into())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::mpsc;
    use std::time::Duration;

    use crate::core::runtime::StdRuntime;
    use super::super::gated::{GatedRuntime, close_the_gate};

    use super::*;

    /// The child's role in [`an_open_once_the_exit_began_makes_no_call_of_libva`].
    const OPEN_CHILD: &str = "RVG_TEST_DISPLAY_OPEN_CHILD";

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
}
