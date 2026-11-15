//! Probes, run by hand: ignored tests that print what they measure on the host's hardware, over
//! inputs a checkout does not hold, and check nothing but that they measured something. They sit
//! outside the `vaapi::` and `nvidia::` sets the hardware hosts' checks run
//! (docs/video/backends.md, "Tests"), as the fuzzers' long runs sit outside every check.

#[cfg(test)]
mod session_starts;
#[cfg(test)]
pub(crate) mod timeline;

/// The VA-API probe of decodes into targets over memory of ours (`vaapi::target_probe`), over the
/// conformance scripts' Annex B streams once they are written: `RVG_VA_STREAMS` names their
/// directories, absolute and `:`-separated (the checkout's `build/h264-check` and
/// `build/hevc-check`), and `RVG_VAAPI_NODE` the render node, `/dev/dri/renderD128` where unset.
/// Without `RVG_VA_STREAMS` it fails, naming it. From the checkout's root:
///
/// ```sh
/// RVG_VA_STREAMS=$PWD/build/h264-check:$PWD/build/hevc-check cargo nextest run \
///   -p reims-vgpu-video --run-ignored only --no-capture \
///   -E 'test(=probes::va_imported_targets_and_their_references)'
/// ```
#[test]
#[ignore = "a probe, run by hand: needs a VA-API GPU and the conformance scripts' streams"]
fn va_imported_targets_and_their_references() {
    let node = std::env::var("RVG_VAAPI_NODE").unwrap_or_else(|_| "/dev/dri/renderD128".into());
    let dirs = std::env::var("RVG_VA_STREAMS").expect("RVG_VA_STREAMS: the streams' directories");
    crate::vaapi::target_probe::probe(&node, &dirs);
}

/// The VA-API probe of a decoder's session start (`session_starts`'s module doc, which names its
/// inputs): each stream's sessions, the first decode of each split into its driver calls. From the
/// checkout's root, with streams written by hand:
///
/// ```sh
/// RVG_START_STREAMS=/abs/h264-2160p.264:/abs/hevc-2160p.265 cargo nextest run --release \
///   -p reims-vgpu-video --run-ignored only --no-capture -E 'test(=probes::va_session_starts)'
/// ```
#[test]
#[ignore = "a probe, run by hand: needs a VA-API GPU and streams written by hand"]
fn va_session_starts() {
    session_starts::va();
}

/// The NVDEC probe of a decoder's session start, as [`va_session_starts`] on VA-API
/// (`session_starts`'s module doc).
#[test]
#[ignore = "a probe, run by hand: needs an NVIDIA GPU and streams written by hand"]
fn nvdec_session_starts() {
    session_starts::nvdec();
}
