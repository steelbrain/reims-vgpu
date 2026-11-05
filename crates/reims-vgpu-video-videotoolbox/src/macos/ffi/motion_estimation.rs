//! VideoToolbox's motion-estimation session
//! (`VideoToolbox/VTMotionEstimationSession.h`,
//! `VTMotionEstimationSessionProperties.h`), new in macOS 26.
//!
//! The functions and keys are looked up at run time with `dlsym`, not
//! linked, so that the backend still loads on an older macOS, which has no
//! motion estimation to offer. Each function's type is declared once,
//! through `looked_up!` (below), which records its C spelling for the header
//! check in `ffi.rs`, as `declare!` does for the linked ones.
//!
//! The session's output handler is a block; `macos/motion.rs` makes one,
//! with the runtime's stack block class declared here.

use std::ffi::c_void;
use std::sync::OnceLock;

use super::core_foundation::{CFAllocatorRef, CFDictionaryRef, OSStatus};
use super::core_video::CVPixelBufferRef;
use super::private::{key, symbol};
use crate::macos::cf::Cf;

/// `VTMotionEstimationSessionRef`.
pub(crate) type VTMotionEstimationSessionRef = *mut c_void;
/// `VTMotionEstimationFrameFlags`: a `uint32_t` set.
pub(crate) type VTMotionEstimationFrameFlags = u32;
/// `VTMotionEstimationInfoFlags`: a `uint32_t` set.
pub(crate) type VTMotionEstimationInfoFlags = u32;
/// `VTMotionEstimationOutputHandler`: a block, here a pointer to a
/// `HandlerBlock` (`macos/motion.rs`).
pub(crate) type VTMotionEstimationOutputHandler = *mut c_void;

/// Declares each looked-up function's type under the function's own name,
/// and records it for the header check as `DECLARATIONS`.
macro_rules! looked_up {
    (
        functions {
            $( fn $name:ident ( $( $arg:ident : $ty:ty ),* $(,)? ) $( -> $ret:ty )? ; )*
        }
        statics {
            $( static $static_name:ident : $static_ty:ty ; )*
        }
    ) => {
        $(
            #[doc = concat!("`", stringify!($name), "`.")]
            pub(crate) type $name = unsafe extern "C" fn( $( $ty ),* ) $( -> $ret )?;
        )*

        #[cfg(test)]
        pub(crate) const DECLARATIONS: &[super::Declaration] = &[
            $(
                super::Declaration::Function {
                    name: stringify!($name),
                    returns: looked_up!(@returns $( $ret )?),
                    arguments: &[ $( stringify!($ty) ),* ],
                },
            )*
            $(
                super::Declaration::Static {
                    name: stringify!($static_name),
                    ty: stringify!($static_ty),
                },
            )*
        ];
    };
    (@returns) => { "void" };
    (@returns $ret:ty) => { stringify!($ret) };
}

looked_up! {
    functions {
        fn VTMotionEstimationSessionCreate(
            allocator: CFAllocatorRef,
            motion_vector_processor_selection_options: CFDictionaryRef,
            width: u32,
            height: u32,
            motion_estimation_session_out: *mut VTMotionEstimationSessionRef,
        ) -> OSStatus;
        fn VTMotionEstimationSessionCopySourcePixelBufferAttributes(
            motion_estimation_session: VTMotionEstimationSessionRef,
            attributes_out: *mut CFDictionaryRef,
        ) -> OSStatus;
        fn VTMotionEstimationSessionEstimateMotionVectors(
            session: VTMotionEstimationSessionRef,
            reference_image: CVPixelBufferRef,
            current_image: CVPixelBufferRef,
            motion_estimation_frame_flags: VTMotionEstimationFrameFlags,
            additional_frame_options: CFDictionaryRef,
            output_handler: VTMotionEstimationOutputHandler,
        ) -> OSStatus;
        fn VTMotionEstimationSessionCompleteFrames(
            session: VTMotionEstimationSessionRef,
        ) -> OSStatus;
        fn VTMotionEstimationSessionInvalidate(session: VTMotionEstimationSessionRef);
    }
    statics {
        static kVTMotionEstimationSessionCreationOption_MotionVectorSize: CFStringRef;
        static kVTMotionEstimationSessionCreationOption_UseMultiPassSearch: CFStringRef;
    }
}

/// The handler's type, for the header check: the typedef against the
/// block type `macos/motion.rs`'s handler is called as.
#[cfg(test)]
pub(crate) const C_ASSERTIONS: &str = "\
_Static_assert(__builtin_types_compatible_p(VTMotionEstimationOutputHandler, \
void (^)(OSStatus, VTMotionEstimationInfoFlags, CFDictionaryRef, CVPixelBufferRef)), \
\"VTMotionEstimationOutputHandler\");\n";

/// The session's functions and keys, where this macOS has them all.
#[derive(Clone, Copy)]
pub(crate) struct MotionEstimation {
    pub(crate) create: VTMotionEstimationSessionCreate,
    pub(crate) copy_source_attributes: VTMotionEstimationSessionCopySourcePixelBufferAttributes,
    pub(crate) estimate: VTMotionEstimationSessionEstimateMotionVectors,
    pub(crate) complete_frames: VTMotionEstimationSessionCompleteFrames,
    pub(crate) invalidate: VTMotionEstimationSessionInvalidate,
    /// `kVTMotionEstimationSessionCreationOption_MotionVectorSize`.
    pub(crate) motion_vector_size: Cf<'static>,
    /// `kVTMotionEstimationSessionCreationOption_UseMultiPassSearch`.
    pub(crate) use_multi_pass_search: Cf<'static>,
}

// SAFETY: the function pointers are plain code addresses, and the keys are
// immutable CFStrings that VideoToolbox's constants hold for the process's
// life; CFStrings may be read from any thread.
unsafe impl Send for MotionEstimation {}
// SAFETY: as for Send: nothing in it is ever written after the lookup.
unsafe impl Sync for MotionEstimation {}

/// The function at `name`, as the type `F` its declaration gives.
///
/// # Safety
///
/// `F` is the function pointer type `name`'s header declares.
unsafe fn function<F: Copy>(name: &std::ffi::CStr) -> Option<F> {
    let address = symbol(name)?;
    const { assert!(size_of::<F>() == size_of::<*mut c_void>()) };
    // SAFETY: as the caller says; a function pointer has a data pointer's
    // size on this platform, which the assertion holds us to.
    Some(unsafe { std::mem::transmute_copy::<*mut c_void, F>(&address) })
}

/// The session's functions and keys, looked up once: None on a macOS
/// without them.
pub(crate) fn motion_estimation() -> Option<MotionEstimation> {
    static LOOKED_UP: OnceLock<Option<MotionEstimation>> = OnceLock::new();
    *LOOKED_UP.get_or_init(|| {
        // SAFETY: each type is the one `looked_up!` declares for that
        // name, which the header check holds to the SDK's.
        unsafe {
            Some(MotionEstimation {
                create: function(c"VTMotionEstimationSessionCreate")?,
                copy_source_attributes: function(
                    c"VTMotionEstimationSessionCopySourcePixelBufferAttributes",
                )?,
                estimate: function(c"VTMotionEstimationSessionEstimateMotionVectors")?,
                complete_frames: function(c"VTMotionEstimationSessionCompleteFrames")?,
                invalidate: function(c"VTMotionEstimationSessionInvalidate")?,
                motion_vector_size: key(
                    c"kVTMotionEstimationSessionCreationOption_MotionVectorSize",
                )?,
                use_multi_pass_search: key(
                    c"kVTMotionEstimationSessionCreationOption_UseMultiPassSearch",
                )?,
            })
        }
    })
}

#[link(name = "System", kind = "dylib")]
unsafe extern "C" {
    /// `_NSConcreteStackBlock`, the class of a block made on the stack.
    pub(crate) static _NSConcreteStackBlock: [*const c_void; 32];
}

/// `BLOCK_HAS_COPY_DISPOSE` (`Block_private.h`; the ABI fixes it): the
/// block's descriptor has copy and dispose helpers.
pub(crate) const BLOCK_HAS_COPY_DISPOSE: i32 = 1 << 25;
