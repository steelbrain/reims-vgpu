//! The H.264 decoder on a VA-API host's GPU, holding it: on Windows on the adapter libva-win32
//! opens first, elsewhere on the render node `RVG_VAAPI_NODE` names (`super::super::teardown`'s
//! display and stream).

use super::super::teardown::{Stream, display};

/// The sample whose first NAL unit is damaged: a P picture, the stream's IDR picture its first.
const DAMAGED: usize = 3;

/// An H.264 P picture whose sample does not decode (its first NAL unit's `forbidden_zero_bit`
/// set) is one picture's error, and each picture after it comes out and reads back, decoded on
/// from a copy of the last reference standing in for the lost frame (the codec module's
/// `Accelerator::conceal`), as a Mac's hardware decoder answers the pictures after a damaged one
/// with no error and an image (held85's damage run, potluck job m3-max-60715).
#[test]
#[ignore = "needs a VA-API driver that decodes H.264, on its GPU: run on a VA-API host"]
fn the_pictures_after_one_that_does_not_decode_come_out() {
    let display = display();
    let stream = Stream::new(&display);
    let samples = stream.damaged(DAMAGED);
    assert!(
        samples.len() > DAMAGED + 2,
        "a stream of {} samples",
        samples.len()
    );
    let mut decoder = stream.decoder(&display);
    let mut out = 0;
    for (id, sample) in (0u64..).zip(&samples) {
        let decoded = decoder.decode(sample, id);
        if id == DAMAGED as u64 {
            let error = decoded.err().expect("the damaged sample does not decode");
            assert!(error.is_picture_error(), "{error}");
            continue;
        }
        let frames = decoded.unwrap_or_else(|error| panic!("sample {id}: {error}"));
        assert_eq!(frames.len(), 1, "sample {id}'s pictures");
        for frame in frames {
            frame.read_back().expect("a read back");
            out += 1;
        }
    }
    assert_eq!(out, samples.len() - 1, "the pictures that came out");
}
