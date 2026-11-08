//! What NVDEC's decodes answer of themselves: a decoder's error as a
//! backend's ([`backend_error`]), and the failures NVDEC reports of the
//! pictures it decoded ([`Failures`]).
//!
//! **A frame NVDEC fails.** NVDEC's status of a picture's decode is read at
//! its copy, after the frame came out of the codec crate's decoder, and a
//! failed one is bad data. An H.264 or HEVC session then has its decoder
//! fail each picture after it until an I picture (H.264) or an IRAP picture
//! (HEVC) decodes ([`Failures`]).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use reims_vgpu_video_core::backend::BackendError;

use crate::error::Error;

/// What a decoder's error is as a backend's: what the stream asks for that
/// the decoder or driver lacks is unsupported, a stream that does not
/// decode, here or on NVDEC, is bad data, a RASL picture skipped is a
/// skipped leading picture, and the rest is the decoder's malfunction.
pub(crate) fn backend_error(error: Error) -> BackendError {
    match error {
        Error::Unsupported(text) => BackendError::Unsupported(text),
        Error::Bitstream(text) | Error::DecodeFailed(text) => BackendError::BadData(text),
        Error::SkippedLeadingPicture(text) => BackendError::SkippedLeadingPicture(text),
        other => BackendError::Malfunction(other.to_string()),
    }
}

/// Whether NVDEC failed the decode of a frame an H.264 or HEVC session
/// gave out ([`Error::DecodeFailed`], read at the frame's copy), since the
/// session's last decode: the frame sets it, and the session's next decode
/// takes it and has the codec crate's decoder fail the pictures after it
/// (module docs).
#[derive(Clone, Default)]
pub(crate) struct Failures(Arc<AtomicBool>);

impl Failures {
    /// `error`, noted where it is a decode NVDEC failed.
    pub(crate) fn noted(&self, error: Error) -> Error {
        if matches!(error, Error::DecodeFailed(_)) {
            self.0.store(true, Ordering::Release);
        }
        error
    }

    /// Whether a decode was noted as failed since the last call.
    fn take(&self) -> bool {
        self.0.swap(false, Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_decode_nvdec_failed_is_noted_once_and_answered_as_bad_data() {
        let failures = Failures::default();
        let error = failures.noted(Error::DecodeFailed("status 8".to_owned()));
        assert!(matches!(backend_error(error), BackendError::BadData(_)));
        assert!(failures.take());
        assert!(!failures.take(), "taken twice");
    }

    #[test]
    fn other_errors_are_not_noted() {
        let failures = Failures::default();
        for error in [
            Error::Bitstream("a slice header".to_owned()),
            Error::Stalled("a decode".to_owned()),
            Error::OutOfSurfaces,
            Error::Unsupported("4:2:2".to_owned()),
        ] {
            let _ = failures.noted(error);
        }
        assert!(!failures.take());
    }

    #[test]
    fn a_sessions_frames_note_into_its_failures() {
        let failures = Failures::default();
        let frame = failures.clone();
        let _ = frame.noted(Error::DecodeFailed("status 9".to_owned()));
        assert!(failures.take());
    }
}
