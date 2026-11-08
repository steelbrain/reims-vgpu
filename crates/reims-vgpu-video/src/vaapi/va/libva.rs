//! libva and libva-drm, loaded at run time with `dlopen` and `dlsym`, so
//! that a QEMU built with this crate starts on a host without them and
//! links nothing of libva's. The nvidia crate loads NVIDIA's libraries the
//! same way.
//!
//! Every libva function the binding calls has a wrapper here of the same
//! name and signature. `ffi` and `ffi::encode` re-export them, so no call
//! site names the loaded table.
//!
//! [`load`] loads both libraries once, for the life of the process, and
//! keeps its answer: a host without libva is told so at each open, without
//! a retry. [`super::Display::open`] calls it before any libva call, and
//! every other call needs a display that open made, so no wrapper runs
//! before a load that succeeded. The libraries are never unloaded: libva
//! loads the driver, whose threads and state outlive any one display.

use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};
use std::fmt;
use std::ptr::NonNull;
use std::sync::OnceLock;

use super::ffi::{
    VABufferID, VABufferType, VAConfigAttrib, VAConfigID, VAContextID, VADisplay, VAEntrypoint,
    VAImage, VAImageFormat, VAImageID, VAMessageCallback, VAProcFilterType, VAProfile, VAStatus,
    VASurfaceAttrib, VASurfaceID,
};

#[link(name = "dl")]
unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *mut c_char;
}

/// `RTLD_NOW`: resolve every symbol at load, so that a broken library
/// fails here rather than at its first call. glibc's value.
const RTLD_NOW: c_int = 2;
/// `RTLD_LOCAL`: keep the library's symbols out of the global scope.
/// glibc's value. libva-drm names libva as a dependency, and libva's
/// driver names what it needs, so neither looks for them globally.
const RTLD_LOCAL: c_int = 0;

/// libva's file name: its soname, as a host's loader finds it.
const LIBVA: &CStr = c"libva.so.2";
/// libva-drm's.
const LIBVA_DRM: &CStr = c"libva-drm.so.2";

/// Why libva or one of its functions could not be loaded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoadError {
    /// The library's file name, as it was asked for.
    pub library: &'static str,
    /// The function, when the library loaded and the function did not.
    pub symbol: Option<&'static str>,
    /// `dlerror`'s text.
    pub reason: String,
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.symbol {
            Some(symbol) => write!(f, "{} has no {symbol}: {}", self.library, self.reason),
            None => write!(f, "cannot load {}: {}", self.library, self.reason),
        }
    }
}

impl std::error::Error for LoadError {}

/// The text of the last `dlopen` or `dlsym` failure on this thread.
fn last_error() -> String {
    // SAFETY: dlerror takes no arguments; it returns NULL or a
    // NUL-terminated string that stays valid until the next dl call on
    // this thread, and it is copied before any.
    let text = unsafe { dlerror() };
    if text.is_null() {
        "no reason given".to_owned()
    } else {
        // SAFETY: non-null and NUL-terminated, as above.
        unsafe { CStr::from_ptr(text) }
            .to_string_lossy()
            .into_owned()
    }
}

/// A loaded library. Its handle is never closed.
struct Library {
    handle: NonNull<c_void>,
    name: &'static str,
}

impl Library {
    /// Loads the library `name` from the loader's search path.
    fn open(name: &'static CStr) -> Result<Self, LoadError> {
        let library = name.to_str().unwrap_or("?");
        // SAFETY: `name` is NUL-terminated; dlopen runs libva's
        // initialisers, which register nothing in this process's globals.
        let handle = unsafe { dlopen(name.as_ptr(), RTLD_NOW | RTLD_LOCAL) };
        match NonNull::new(handle) {
            Some(handle) => Ok(Self {
                handle,
                name: library,
            }),
            None => Err(LoadError {
                library,
                symbol: None,
                reason: last_error(),
            }),
        }
    }

    /// The address of the function `symbol`.
    fn symbol(&self, symbol: &'static str) -> Result<NonNull<c_void>, LoadError> {
        let missing = |reason: String| LoadError {
            library: self.name,
            symbol: Some(symbol),
            reason,
        };
        let name = CString::new(symbol).map_err(|e| missing(e.to_string()))?;
        // SAFETY: the handle is a live dlopen handle (never closed) and the
        // name is NUL-terminated.
        let address = unsafe { dlsym(self.handle.as_ptr(), name.as_ptr()) };
        NonNull::new(address).ok_or_else(|| missing(last_error()))
    }
}

/// Declares the table of one library's functions, each loaded by name, and
/// for each a wrapper of the same name and signature that calls it through
/// [`loaded`]'s `$part`. Each signature is va.h's or va_drm.h's.
macro_rules! functions {
    (
        $(#[$meta:meta])*
        struct $table:ident in $part:ident {
            $(
                fn $name:ident($($arg:ident: $ty:ty),* $(,)?) -> $ret:ty;
            )*
        }
    ) => {
        $(#[$meta])*
        #[allow(non_snake_case)]
        struct $table {
            $( $name: unsafe extern "C" fn($($ty),*) -> $ret, )*
        }

        impl $table {
            /// Loads every function from `library`, failing on the first
            /// that is missing.
            fn load(library: &Library) -> Result<Self, LoadError> {
                Ok(Self {
                    $(
                        // SAFETY: the symbol is the C function of that name,
                        // whose signature the field declares; a function
                        // pointer and a data pointer have one size here.
                        $name: unsafe {
                            std::mem::transmute::<
                                *mut c_void,
                                unsafe extern "C" fn($($ty),*) -> $ret,
                            >(library.symbol(stringify!($name))?.as_ptr())
                        },
                    )*
                })
            }
        }

        $(
            /// The libva function of this name, through the loaded library.
            ///
            /// # Safety
            ///
            /// As for the C function. Called only on a display that
            /// [`super::Display::open`] made, so after [`load`] succeeded.
            #[allow(non_snake_case, clippy::too_many_arguments)]
            pub(crate) unsafe fn $name($($arg: $ty),*) -> $ret {
                // SAFETY: the caller's, as for the C function.
                unsafe { (loaded().$part.$name)($($arg),*) }
            }
        )*
    };
}

functions! {
    /// libva's functions (va.h).
    struct Va in va {
        fn vaErrorStr(error_status: VAStatus) -> *const c_char;
        fn vaInitialize(
            dpy: VADisplay,
            major_version: *mut c_int,
            minor_version: *mut c_int,
        ) -> VAStatus;
        fn vaTerminate(dpy: VADisplay) -> VAStatus;
        fn vaQueryVendorString(dpy: VADisplay) -> *const c_char;
        fn vaSetInfoCallback(
            dpy: VADisplay,
            callback: VAMessageCallback,
            user_context: *mut c_void,
        ) -> VAMessageCallback;
        fn vaMaxNumProfiles(dpy: VADisplay) -> c_int;
        fn vaMaxNumEntrypoints(dpy: VADisplay) -> c_int;
        fn vaMaxNumImageFormats(dpy: VADisplay) -> c_int;
        fn vaQueryConfigProfiles(
            dpy: VADisplay,
            profile_list: *mut VAProfile,
            num_profiles: *mut c_int,
        ) -> VAStatus;
        fn vaQueryConfigEntrypoints(
            dpy: VADisplay,
            profile: VAProfile,
            entrypoint_list: *mut VAEntrypoint,
            num_entrypoints: *mut c_int,
        ) -> VAStatus;
        fn vaGetConfigAttributes(
            dpy: VADisplay,
            profile: VAProfile,
            entrypoint: VAEntrypoint,
            attrib_list: *mut VAConfigAttrib,
            num_attribs: c_int,
        ) -> VAStatus;
        fn vaCreateConfig(
            dpy: VADisplay,
            profile: VAProfile,
            entrypoint: VAEntrypoint,
            attrib_list: *mut VAConfigAttrib,
            num_attribs: c_int,
            config_id: *mut VAConfigID,
        ) -> VAStatus;
        fn vaDestroyConfig(dpy: VADisplay, config_id: VAConfigID) -> VAStatus;
        fn vaQuerySurfaceAttributes(
            dpy: VADisplay,
            config: VAConfigID,
            attrib_list: *mut VASurfaceAttrib,
            num_attribs: *mut c_uint,
        ) -> VAStatus;
        fn vaCreateSurfaces(
            dpy: VADisplay,
            format: c_uint,
            width: c_uint,
            height: c_uint,
            surfaces: *mut VASurfaceID,
            num_surfaces: c_uint,
            attrib_list: *mut VASurfaceAttrib,
            num_attribs: c_uint,
        ) -> VAStatus;
        fn vaDestroySurfaces(
            dpy: VADisplay,
            surfaces: *mut VASurfaceID,
            num_surfaces: c_int,
        ) -> VAStatus;
        fn vaCreateContext(
            dpy: VADisplay,
            config_id: VAConfigID,
            picture_width: c_int,
            picture_height: c_int,
            flag: c_int,
            render_targets: *mut VASurfaceID,
            num_render_targets: c_int,
            context: *mut VAContextID,
        ) -> VAStatus;
        fn vaDestroyContext(dpy: VADisplay, context: VAContextID) -> VAStatus;
        fn vaCreateBuffer(
            dpy: VADisplay,
            context: VAContextID,
            type_: VABufferType,
            size: c_uint,
            num_elements: c_uint,
            data: *mut c_void,
            buf_id: *mut VABufferID,
        ) -> VAStatus;
        fn vaDestroyBuffer(dpy: VADisplay, buffer_id: VABufferID) -> VAStatus;
        fn vaMapBuffer(
            dpy: VADisplay,
            buf_id: VABufferID,
            pbuf: *mut *mut c_void,
        ) -> VAStatus;
        fn vaUnmapBuffer(dpy: VADisplay, buf_id: VABufferID) -> VAStatus;
        fn vaBeginPicture(
            dpy: VADisplay,
            context: VAContextID,
            render_target: VASurfaceID,
        ) -> VAStatus;
        fn vaRenderPicture(
            dpy: VADisplay,
            context: VAContextID,
            buffers: *mut VABufferID,
            num_buffers: c_int,
        ) -> VAStatus;
        fn vaEndPicture(dpy: VADisplay, context: VAContextID) -> VAStatus;
        fn vaSyncSurface(dpy: VADisplay, render_target: VASurfaceID) -> VAStatus;
        fn vaSyncSurface2(dpy: VADisplay, surface: VASurfaceID, timeout_ns: u64) -> VAStatus;
        fn vaSyncBuffer(dpy: VADisplay, buf_id: VABufferID, timeout_ns: u64) -> VAStatus;
        fn vaQueryImageFormats(
            dpy: VADisplay,
            format_list: *mut VAImageFormat,
            num_formats: *mut c_int,
        ) -> VAStatus;
        fn vaCreateImage(
            dpy: VADisplay,
            format: *mut VAImageFormat,
            width: c_int,
            height: c_int,
            image: *mut VAImage,
        ) -> VAStatus;
        fn vaDestroyImage(dpy: VADisplay, image: VAImageID) -> VAStatus;
        fn vaGetImage(
            dpy: VADisplay,
            surface: VASurfaceID,
            x: c_int,
            y: c_int,
            width: c_uint,
            height: c_uint,
            image: VAImageID,
        ) -> VAStatus;
        fn vaPutImage(
            dpy: VADisplay,
            surface: VASurfaceID,
            image: VAImageID,
            src_x: c_int,
            src_y: c_int,
            src_width: c_uint,
            src_height: c_uint,
            dest_x: c_int,
            dest_y: c_int,
            dest_width: c_uint,
            dest_height: c_uint,
        ) -> VAStatus;
        fn vaDeriveImage(
            dpy: VADisplay,
            surface: VASurfaceID,
            image: *mut VAImage,
        ) -> VAStatus;
        fn vaExportSurfaceHandle(
            dpy: VADisplay,
            surface_id: VASurfaceID,
            mem_type: u32,
            flags: u32,
            descriptor: *mut c_void,
        ) -> VAStatus;
        fn vaQueryVideoProcFilters(
            dpy: VADisplay,
            context: VAContextID,
            filters: *mut VAProcFilterType,
            num_filters: *mut c_uint,
        ) -> VAStatus;
    }
}

functions! {
    /// libva-drm's function (va_drm.h).
    struct VaDrm in drm {
        fn vaGetDisplayDRM(fd: c_int) -> VADisplay;
    }
}

/// Both libraries' functions.
struct Loaded {
    va: Va,
    drm: VaDrm,
}

/// What [`load`] found, once for the process.
static LOADED: OnceLock<Result<Loaded, LoadError>> = OnceLock::new();

/// Loads libva and libva-drm, and every function the binding calls from
/// them, on the first call; later calls give the first one's answer.
pub(crate) fn load() -> Result<(), LoadError> {
    let loaded = LOADED.get_or_init(|| {
        let va = Library::open(LIBVA)?;
        let drm = Library::open(LIBVA_DRM)?;
        Ok(Loaded {
            va: Va::load(&va)?,
            drm: VaDrm::load(&drm)?,
        })
    });
    match loaded {
        Ok(_) => Ok(()),
        Err(error) => Err(error.clone()),
    }
}

/// The loaded functions. Only a wrapper calls this, and only on a display
/// [`super::Display::open`] made after [`load`] succeeded.
fn loaded() -> &'static Loaded {
    match LOADED.get() {
        Some(Ok(loaded)) => loaded,
        _ => unreachable!("a libva call before libva loaded: only Display::open makes a display"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_library_is_an_error_not_a_crash() {
        let error = Library::open(c"librvgv-no-such-library.so.1").err();
        assert_eq!(
            error.as_ref().map(|e| (e.library, e.symbol)),
            Some(("librvgv-no-such-library.so.1", None))
        );
        assert!(error.is_some_and(|e| !e.reason.is_empty()));
    }

    #[test]
    fn a_missing_function_names_itself() {
        // glibc's libc is in every process here, and has no libva function.
        let library = Library::open(c"libc.so.6");
        let error = library.and_then(|l| l.symbol("vaInitialize")).err();
        assert_eq!(
            error.as_ref().map(|e| (e.library, e.symbol)),
            Some(("libc.so.6", Some("vaInitialize")))
        );
    }
}
