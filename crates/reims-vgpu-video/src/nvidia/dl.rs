//! The driver's libraries, loaded at run time with `dlopen` and `dlsym`,
//! so that a QEMU built with this crate starts on a host with no NVIDIA
//! driver and links nothing of NVIDIA's; and what the three bindings
//! share: their function tables ([`functions!`]) and their all-zero
//! structs ([`Zeroed`]).
//!
//! A library, once loaded, stays loaded for the life of the process: the
//! driver's libraries start threads and keep state that outlives any one
//! handle, and nothing here unloads them.
//!
//! No function of ours is handed to the driver: the crate uses neither
//! NVIDIA's parser nor NVENC's events, so nothing of the driver's calls
//! back into Rust, and no panic can unwind into its frames.

use std::ffi::{CStr, c_char, c_int, c_void};
use std::fmt;
use std::ptr::NonNull;

#[cfg_attr(target_os = "linux", link(name = "dl"))]
unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *mut c_char;
}

/// `RTLD_NOW`: resolve every symbol at load, so that a broken library
/// fails here rather than at its first call. glibc's and Darwin's value.
const RTLD_NOW: c_int = 2;
/// `RTLD_LOCAL`: keep the library's symbols out of the global scope.
#[cfg(target_os = "macos")]
const RTLD_LOCAL: c_int = 4;
#[cfg(not(target_os = "macos"))]
const RTLD_LOCAL: c_int = 0;

/// Why a library or one of its functions could not be loaded.
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

/// A loaded library.
pub(crate) struct Library {
    handle: NonNull<c_void>,
    name: &'static str,
}

// SAFETY: a dlopen handle is a process-wide token; dlsym on it is
// thread-safe (POSIX: "dlsym() shall be thread-safe" since 2008), and the
// handle is never closed.
unsafe impl Send for Library {}
// SAFETY: as above: nothing about the handle is mutated after dlopen.
unsafe impl Sync for Library {}

impl Library {
    /// Loads the library `name` from the loader's search path.
    pub(crate) fn open(name: &'static CStr) -> Result<Self, LoadError> {
        let library = name.to_str().unwrap_or("?");
        // SAFETY: `name` is NUL-terminated; dlopen runs the library's
        // initialisers, which for the driver's libraries register nothing
        // in this process's globals.
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
    pub(crate) fn symbol(&self, symbol: &'static CStr) -> Result<NonNull<c_void>, LoadError> {
        // SAFETY: the handle is a live dlopen handle (never closed) and the
        // name is NUL-terminated.
        let address = unsafe { dlsym(self.handle.as_ptr(), symbol.as_ptr()) };
        NonNull::new(address).ok_or_else(|| LoadError {
            library: self.name,
            symbol: Some(symbol.to_str().unwrap_or("?")),
            reason: last_error(),
        })
    }
}

/// Declares a table of a library's functions, each loaded by name with
/// [`Library::symbol`]. Each field is the function's `unsafe extern "C"`
/// pointer, whose signature is the C declaration's (checked against
/// NVIDIA's headers where they are on the host).
macro_rules! functions {
    (
        $(#[$meta:meta])*
        $vis:vis struct $name:ident {
            $(
                $(#[$field_meta:meta])*
                $field:ident: fn($($arg:ty),* $(,)?) -> $ret:ty = $symbol:literal;
            )*
        }
    ) => {
        $(#[$meta])*
        #[allow(non_snake_case, missing_docs)]
        $vis struct $name {
            $(
                $(#[$field_meta])*
                pub(crate) $field: unsafe extern "C" fn($($arg),*) -> $ret,
            )*
        }

        impl $name {
            /// Loads every function from `library`, failing on the first
            /// that is missing.
            pub(crate) fn load(library: &$crate::nvidia::dl::Library) -> Result<Self, $crate::nvidia::dl::LoadError> {
                Ok(Self {
                    $(
                        // SAFETY: the symbol is the C function of that name,
                        // whose signature the field declares; a function
                        // pointer and a data pointer have one size here.
                        $field: unsafe {
                            std::mem::transmute::<
                                *mut std::ffi::c_void,
                                unsafe extern "C" fn($($arg),*) -> $ret,
                            >(library.symbol($symbol)?.as_ptr())
                        },
                    )*
                })
            }
        }

        #[cfg(test)]
        impl $name {
            /// Each function's C signature as declared here, for the
            /// check against the driver's headers.
            #[allow(
                dead_code,
                reason = "a table the headers do not declare has nothing to check it against"
            )]
            pub(crate) const SIGNATURES: &'static [$crate::nvidia::dl::Signature] = &[
                $(
                    $crate::nvidia::dl::Signature {
                        symbol: match $symbol.to_str() {
                            Ok(name) => name,
                            Err(_) => "?",
                        },
                        result: stringify!($ret),
                        arguments: &[$(stringify!($arg)),*],
                    },
                )*
            ];
        }
    };
}

pub(crate) use functions;

/// A C struct of the bindings that starts all-zero, as the headers' callers
/// fill them: `memset` to 0, then each input set.
///
/// # Safety
///
/// All-zero bytes are a valid value of the implementing type: it is
/// `repr(C)` and holds only integers, raw pointers, `Option`s of function
/// pointers (`None`), and arrays, structs and unions of those.
pub(crate) unsafe trait Zeroed: Copy {
    /// The all-zero value.
    fn zeroed() -> Self {
        // SAFETY: the trait's contract: all-zero bytes are a valid value.
        unsafe { std::mem::zeroed() }
    }
}

/// A function's signature as Rust declares it, in Rust's spelling: what
/// `crate::nvidia::layout`'s test turns into C and compares with the header's.
#[cfg(test)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Signature {
    /// The function's name in the library, or its field's in a table.
    pub(crate) symbol: &'static str,
    /// The return type.
    pub(crate) result: &'static str,
    /// The argument types, in order.
    pub(crate) arguments: &'static [&'static str],
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
}
