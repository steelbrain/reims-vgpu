//! Command-stream framing decoder (port of `host/utils/reims-vgpu-stream-decode`).

use reims_vgpu_core::endian::ld32;
use reims_vgpu_protocol::size_fits_u32;

// Segment types and header length from `reims-vgpu-wire` (observed serializer
// surface). Re-exported so stream walkers share one path with the wire crate.
//
// `SEGMENT_TYPE_INFO` is 4, not the next integer in sequence — a guess would
// write 3, which is `SEGMENT_TYPE_EVENT` and stays local below. Protection
// options joined once the capture drove that envelope.
use reims_vgpu_wire::ops::segment as wire_segment;
pub use reims_vgpu_wire::ops::segment::{
    SEGMENT_HEADER_LEN, SEGMENT_TYPE_BLIT, SEGMENT_TYPE_COMPUTE, SEGMENT_TYPE_INFO,
    SEGMENT_TYPE_PROTECTION_OPTIONS, SEGMENT_TYPE_RENDER,
};

// The one type the wire crate deliberately does not name, because its capture
// has never driven the encoder that writes it. Keeping it here rather than
// pushing it upstream is the honest split: `reims-vgpu-wire` names what Apple's
// serializer was observed to emit, and an unobserved value has no place in it.
pub const SEGMENT_TYPE_EVENT: u8 = 3;

/// Segment-header field offsets, from the view that derived them.
///
/// The wire struct deliberately stops at seven bytes: the eighth is not written
/// and therefore is not a field the semantic decoder may read.
pub const SEGMENT_LENGTH_OFFSET: usize = core::mem::offset_of!(wire_segment::SegmentHeader, length);
pub const SEGMENT_TYPE_OFFSET: usize =
    core::mem::offset_of!(wire_segment::SegmentHeader, segment_type);
pub const SEGMENT_CONTINUES_PREVIOUS_OFFSET: usize =
    core::mem::offset_of!(wire_segment::SegmentHeader, continues_previous);
pub const SEGMENT_CONTINUES_NEXT_OFFSET: usize =
    core::mem::offset_of!(wire_segment::SegmentHeader, continues_next);

/// Record-header field offsets. This is the serializer's op header, a different
/// protocol level from the segment header above — see [`SEGMENT_HEADER_LEN`].
pub const RECORD_OPCODE_OFFSET: usize = core::mem::offset_of!(reims_vgpu_wire::OpHeader, opcode);
pub const RECORD_LENGTH_OFFSET: usize = core::mem::offset_of!(reims_vgpu_wire::OpHeader, length);
/// Serializer op-header length ([`reims_vgpu_wire::OP_HEADER_LEN`]). Distinct
/// from [`SEGMENT_HEADER_LEN`]: both are 8, but they frame different protocol
/// levels — do not treat them as interchangeable.
use reims_vgpu_wire::OP_HEADER_LEN;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeStatus {
    Ok,
    /// End of stream, or end of a segment's records. Control flow — the walkers
    /// terminate on it, so it is never a refusal and never reaches the log.
    Done,
    /// Refused; the payload is the registered slug naming which check refused.
    ///
    /// The payload is not decoration. This decoder frames *every* guest command,
    /// and a single coarse `ErrBadLength` covers seventeen checks here — a
    /// segment header disagreeing with the buffer, a record header disagreeing
    /// with its segment, and the re-validation of an already-parsed segment are
    /// three very different bugs that would otherwise arrive at the sink
    /// wearing one name.
    ErrArgs(&'static str),
    ErrShort(&'static str),
    ErrBadLength(&'static str),
}

impl crate::observe::Refusal for DecodeStatus {
    /// Slugs carry a `stream_` prefix: seven modules under `runtime/decode/`
    /// define a type called `DecodeStatus`, and five of them have an `ErrShort`
    /// meaning a different read. Without the prefix the crate-wide uniqueness
    /// gate could not tell this decoder's refusals from any other's.
    fn refusal(&self) -> Option<&'static str> {
        match self {
            Self::Ok | Self::Done => None,
            Self::ErrArgs(slug) | Self::ErrShort(slug) | Self::ErrBadLength(slug) => Some(slug),
        }
    }
}

/// One segment of the command stream, as decoded from its header.
///
/// The two bytes after `type_` are encoder-lifetime control.
/// `continues_previous` means this segment's records continue
/// the encoder the previous segment left open, and are a protocol error if that
/// encoder is absent or of a different type — the reader is required to refuse
/// rather than quietly open a fresh one. `continues_next` means the
/// encoder survives this segment and the next one may continue it; clear means
/// the encoder ends here. A render segment that does *not* continue a previous
/// one begins by decoding a render-pass descriptor out of its own records.
///
/// So one render command encoder — and therefore its pipeline, bind state and
/// render pass — may span an unbounded number of records across an unbounded
/// number of submitted child buffers. The executor carries the owning decoder
/// state until `continues_next` clears; resetting it at a child-buffer boundary
/// drops valid continuation draws that do not repeat their pipeline bind.
/// [`SEGMENT_CHAIN_ROUTES`] remains the workload instrument for how often each
/// lifetime form is used.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Segment {
    pub offset: u32,
    pub length: u32,
    pub type_: u8,
    /// This segment continues the open encoder from its predecessor.
    pub continues_previous: bool,
    /// This segment leaves its encoder open for its successor.
    pub continues_next: bool,
    pub command_offset: u32,
    pub command_length: u32,
    pub index: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Record {
    pub segment_index: u32,
    pub segment_type: u32,
    pub offset: u32,
    pub length: u32,
    pub opcode: u32,
    /// Absolute offset of the record header in the stream bytes.
    pub bytes_offset: u32,
}

pub fn segment_type_name(type_: u32) -> &'static str {
    match type_ {
        0 => "render",
        1 => "compute",
        2 => "blit",
        3 => "event",
        4 => "info",
        5 => "protection-options",
        _ => "unknown",
    }
}

/// What the stream walker should do with a segment family.
///
/// This exists so the walker's "everything else" arm is a decision rather than a
/// fallthrough. It used to be `_ => {}`, which gave the same silence to a IOSurface plane view
/// envelope the contract says to skip and to a segment family the host has never
/// seen — and the second of those is unknown wire format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SegmentDisposition {
    /// A family with a record walker: render, compute, blit, event, info.
    Walk,
    /// Type 5. `-beginSegment:protectionOptions:` emits a segment-level envelope
    /// *before* the real segment. Skipping it is contract-correct, so it is
    /// control flow and stays silent — logging it would put a line in the sink
    /// on every healthy frame that carries one.
    ///
    /// # Its command window is the protection value, and this doc used to deny
    /// that
    ///
    /// The wording here was "raw envelope bytes carrying no decodable protection
    /// value", which was a guess written where a measurement now sits. Driven
    /// under `-setSupportsProtectionOptionsEnvelope:` the burst is exactly three
    /// records: this IOSurface plane view header, then **eight fully-written bytes that are the
    /// `protectionOptions:` argument verbatim**, then the ordinary segment
    /// header. `reims_vgpu_wire::ops::segment::ProtectionOptionsEnvelope` is the
    /// view; `blit_begin_segment_protected` sends `0x44` and `..._alt` sends
    /// `0x33`, so it is the guest's value and not a constant.
    ///
    /// Skipping stays right — this device implements no protection domains, so
    /// there is nothing to do with the value — but "we cannot read it" and "we
    /// choose not to act on it" are different claims and only the second is
    /// true.
    ///
    /// The envelope needs **both** conditions: the `BOOL` argument clear *and*
    /// non-zero options. Either alone emits the ordinary single header, which is
    /// measured by `blit_begin_segment_protected_flag_set` and
    /// `blit_begin_segment_protection_zero` respectively.
    Envelope,
    /// A family this host has no contract for. MetalSerializer's deserializer
    /// constructs decoders for `0..3` and rejects new non-continuation types
    /// `>= 4`, so a type past the known set is not something to guess at:
    /// refuse it visibly instead of walking its bytes as records.
    Unknown,
}

impl crate::observe::Refusal for SegmentDisposition {
    fn refusal(&self) -> Option<&'static str> {
        match self {
            Self::Walk | Self::Envelope => None,
            Self::Unknown => Some("stream_segment_type_unknown"),
        }
    }
}

pub fn segment_disposition(type_: u8) -> SegmentDisposition {
    match type_ {
        SEGMENT_TYPE_RENDER | SEGMENT_TYPE_COMPUTE | SEGMENT_TYPE_BLIT | SEGMENT_TYPE_EVENT
        | SEGMENT_TYPE_INFO => SegmentDisposition::Walk,
        SEGMENT_TYPE_PROTECTION_OPTIONS => SegmentDisposition::Envelope,
        _ => SegmentDisposition::Unknown,
    }
}

fn validate_bytes(bytes: &[u8]) -> DecodeStatus {
    if !size_fits_u32(bytes.len()) {
        return DecodeStatus::ErrBadLength("stream_bytes_len_overflow");
    }
    DecodeStatus::Ok
}

fn segment_index_for_offset(bytes: &[u8], target_offset: u32) -> Result<u32, DecodeStatus> {
    let mut cursor = 0usize;
    let mut index = 0u32;
    while cursor < bytes.len() {
        if bytes.len() - cursor < SEGMENT_HEADER_LEN {
            return Err(DecodeStatus::ErrShort("stream_index_walk_short_header"));
        }
        if !size_fits_u32(cursor) {
            return Err(DecodeStatus::ErrBadLength(
                "stream_index_walk_cursor_overflow",
            ));
        }
        if cursor as u32 == target_offset {
            return Ok(index);
        }
        let segment_len = ld32(&bytes[cursor + SEGMENT_LENGTH_OFFSET..]) as usize;
        if segment_len < SEGMENT_HEADER_LEN || segment_len > bytes.len() - cursor {
            return Err(DecodeStatus::ErrBadLength("stream_index_walk_seg_len"));
        }
        cursor += segment_len;
        index += 1;
    }
    Err(DecodeStatus::ErrBadLength(
        "stream_index_target_offset_not_found",
    ))
}

/// Decode the next segment at `cursor`. On Ok advances cursor. Transactional: no partial out.
pub fn decode_next_segment(bytes: &[u8], cursor: &mut usize) -> Result<Segment, DecodeStatus> {
    let status = validate_bytes(bytes);
    if status != DecodeStatus::Ok {
        return Err(status);
    }
    if *cursor > bytes.len() {
        return Err(DecodeStatus::ErrArgs("stream_seg_cursor_past_end"));
    }
    if *cursor == bytes.len() {
        return Err(DecodeStatus::Done);
    }
    if bytes.len() - *cursor < SEGMENT_HEADER_LEN {
        return Err(DecodeStatus::ErrShort("stream_seg_short_header"));
    }
    let header = &bytes[*cursor..];
    let segment_len = ld32(&header[SEGMENT_LENGTH_OFFSET..]) as usize;
    if segment_len < SEGMENT_HEADER_LEN {
        return Err(DecodeStatus::ErrBadLength("stream_seg_len_below_header"));
    }
    if segment_len > bytes.len() - *cursor {
        return Err(DecodeStatus::ErrBadLength("stream_seg_len_past_buffer_end"));
    }
    if !size_fits_u32(*cursor) {
        return Err(DecodeStatus::ErrBadLength("stream_seg_cursor_overflow"));
    }
    let segment_index = segment_index_for_offset(bytes, *cursor as u32)?;
    let continues_previous = header[SEGMENT_CONTINUES_PREVIOUS_OFFSET] != 0;
    let continues_next = header[SEGMENT_CONTINUES_NEXT_OFFSET] != 0;
    crate::runtime::drain::note_store_route(segment_chain_route(
        continues_previous,
        continues_next,
    ));
    let out = Segment {
        offset: *cursor as u32,
        length: segment_len as u32,
        type_: header[SEGMENT_TYPE_OFFSET],
        continues_previous,
        continues_next,
        command_offset: (*cursor + SEGMENT_HEADER_LEN) as u32,
        command_length: (segment_len - SEGMENT_HEADER_LEN) as u32,
        index: segment_index,
    };
    *cursor += segment_len;
    Ok(out)
}

/// Every census route [`segment_chain_route`] can answer, in the order
/// `(continues_previous, continues_into_next)` counts up.
///
/// Exported so a reading is over a named set rather than over whichever names a
/// grep of the log happened to find, and so the four cannot be spelled twice.
pub const SEGMENT_CHAIN_ROUTES: [&str; 4] = [
    "seg_chain_none",
    "seg_chain_next",
    "seg_chain_prev",
    "seg_chain_both",
];

/// Which of [`SEGMENT_CHAIN_ROUTES`] a segment header's two encoder-lifetime
/// bytes select.
///
pub fn segment_chain_route(continues_previous: bool, continues_into_next: bool) -> &'static str {
    let index = usize::from(continues_previous) << 1 | usize::from(continues_into_next);
    SEGMENT_CHAIN_ROUTES[index]
}

fn validate_segment(bytes: &[u8], segment: &Segment) -> Result<usize, DecodeStatus> {
    let status = validate_bytes(bytes);
    if status != DecodeStatus::Ok {
        return Err(status);
    }
    if (segment.length as usize) < SEGMENT_HEADER_LEN {
        return Err(DecodeStatus::ErrBadLength("stream_reval_len_below_header"));
    }
    if (segment.offset as usize) > bytes.len()
        || (segment.length as usize) > bytes.len() - segment.offset as usize
    {
        return Err(DecodeStatus::ErrBadLength("stream_reval_span_oob"));
    }
    let header = &bytes[segment.offset as usize..];
    if ld32(&header[SEGMENT_LENGTH_OFFSET..]) != segment.length
        || header[SEGMENT_TYPE_OFFSET] != segment.type_
        || (header[SEGMENT_CONTINUES_PREVIOUS_OFFSET] != 0) != segment.continues_previous
        || (header[SEGMENT_CONTINUES_NEXT_OFFSET] != 0) != segment.continues_next
    {
        return Err(DecodeStatus::ErrBadLength("stream_reval_header_mismatch"));
    }
    if segment.command_offset != segment.offset + SEGMENT_HEADER_LEN as u32
        || segment.command_length != segment.length - SEGMENT_HEADER_LEN as u32
    {
        return Err(DecodeStatus::ErrBadLength(
            "stream_reval_command_span_mismatch",
        ));
    }
    if segment.command_offset < segment.offset
        || segment.command_length > u32::MAX - segment.command_offset
    {
        return Err(DecodeStatus::ErrBadLength(
            "stream_reval_command_offset_overflow",
        ));
    }
    let command_end = segment.command_offset as usize + segment.command_length as usize;
    if (segment.command_offset as usize) > command_end
        || command_end > segment.offset as usize + segment.length as usize
        || command_end > bytes.len()
    {
        return Err(DecodeStatus::ErrBadLength("stream_reval_command_end_oob"));
    }
    Ok(command_end)
}

pub fn decode_next_record(
    bytes: &[u8],
    segment: &Segment,
    cursor: &mut usize,
) -> Result<Record, DecodeStatus> {
    let command_end = validate_segment(bytes, segment)?;
    if *cursor < segment.command_offset as usize || *cursor > command_end {
        return Err(DecodeStatus::ErrArgs("stream_rec_cursor_out_of_segment"));
    }
    if segment.type_ == SEGMENT_TYPE_PROTECTION_OPTIONS {
        if *cursor != segment.command_offset as usize && *cursor != command_end {
            return Err(DecodeStatus::ErrArgs(
                "stream_rec_protection_cursor_misaligned",
            ));
        }
        *cursor = command_end;
        return Err(DecodeStatus::Done);
    }
    if *cursor == command_end {
        return Err(DecodeStatus::Done);
    }
    if command_end - *cursor < OP_HEADER_LEN {
        return Err(DecodeStatus::ErrShort("stream_rec_short_header"));
    }
    let header = &bytes[*cursor..];
    let opcode = ld32(&header[RECORD_OPCODE_OFFSET..]);
    let record_len = ld32(&header[RECORD_LENGTH_OFFSET..]) as usize;
    if record_len < OP_HEADER_LEN {
        return Err(DecodeStatus::ErrBadLength("stream_rec_len_below_header"));
    }
    if record_len > command_end - *cursor {
        return Err(DecodeStatus::ErrBadLength(
            "stream_rec_len_past_segment_end",
        ));
    }
    if !size_fits_u32(*cursor) {
        return Err(DecodeStatus::ErrBadLength("stream_rec_cursor_overflow"));
    }
    let out = Record {
        segment_index: segment.index,
        segment_type: segment.type_ as u32,
        offset: *cursor as u32,
        length: record_len as u32,
        opcode,
        bytes_offset: *cursor as u32,
    };
    *cursor += record_len;
    Ok(out)
}

pub fn decode_first_record(
    bytes: &[u8],
    segment: &Segment,
    cursor: &mut usize,
) -> Result<Record, DecodeStatus> {
    *cursor = segment.command_offset as usize;
    decode_next_record(bytes, segment, cursor)
}

/// Iterate all segments.
pub fn iter_segments(bytes: &[u8]) -> Result<Vec<Segment>, DecodeStatus> {
    let mut cursor = 0usize;
    let mut out = Vec::new();
    loop {
        match decode_next_segment(bytes, &mut cursor) {
            Ok(s) => out.push(s),
            Err(DecodeStatus::Done) => return Ok(out),
            Err(e) => return Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reims_vgpu_core::endian::st32;

    /// The four chain routes are distinct, and each pair of flags selects its
    /// own. A collision would fold two populations into one bucket, which is
    /// the failure that reads as an answer: `seg_chain_none` carrying the whole
    /// census is the reading that says this guest never chains encoders, and it
    /// must not also be what a mis-indexed table says.
    #[test]
    fn each_pair_of_chain_flags_selects_its_own_route() {
        let mut seen = std::collections::HashSet::new();
        for route in SEGMENT_CHAIN_ROUTES {
            assert!(
                seen.insert(route),
                "two chain routes share the name {route}"
            );
        }
        assert_eq!(segment_chain_route(false, false), "seg_chain_none");
        assert_eq!(segment_chain_route(false, true), "seg_chain_next");
        assert_eq!(segment_chain_route(true, false), "seg_chain_prev");
        assert_eq!(segment_chain_route(true, true), "seg_chain_both");
    }

    /// The bytes are guest-controlled, so the flag test is `!= 0` and not
    /// `== 1`. A guest writing any other truthy value must not be counted as
    /// "did not chain" — that would put a segment the reader's contract says
    /// continues an open encoder into the bucket whose emptiness is the whole
    /// question.
    #[test]
    fn decoder_normalizes_any_non_zero_chain_byte() {
        for v in [1u8, 2, 0x7f, 0x80, 0xff] {
            let mut bytes = [0u8; SEGMENT_HEADER_LEN];
            st32(&mut bytes[..4], SEGMENT_HEADER_LEN as u32);
            bytes[SEGMENT_CONTINUES_PREVIOUS_OFFSET] = v;
            bytes[SEGMENT_CONTINUES_NEXT_OFFSET] = v;
            let segment = decode_next_segment(&bytes, &mut 0).expect("segment");
            assert!(segment.continues_previous, "prev={v:#x}");
            assert!(segment.continues_next, "next={v:#x}");
        }
    }

    #[test]
    fn unwritten_header_byte_is_not_semantic_input() {
        let mut first = [0u8; SEGMENT_HEADER_LEN];
        st32(&mut first[..4], SEGMENT_HEADER_LEN as u32);
        first[SEGMENT_HEADER_LEN - 1] = 0xaa;
        let mut second = first;
        second[SEGMENT_HEADER_LEN - 1] = 0x55;

        let a = decode_next_segment(&first, &mut 0).expect("first segment");
        let b = decode_next_segment(&second, &mut 0).expect("second segment");
        assert_eq!(a, b);
        assert_eq!(validate_segment(&second, &a), Ok(SEGMENT_HEADER_LEN));
    }

    fn push_segment(buf: &mut Vec<u8>, type_: u8, payload: &[u8]) {
        let len = (SEGMENT_HEADER_LEN + payload.len()) as u32;
        let mut hdr = [0u8; 8];
        st32(&mut hdr[0..4], len);
        hdr[4] = type_;
        buf.extend_from_slice(&hdr);
        buf.extend_from_slice(payload);
    }

    fn push_record(buf: &mut Vec<u8>, opcode: u32, payload: &[u8]) {
        let len = (OP_HEADER_LEN + payload.len()) as u32;
        let mut hdr = [0u8; 8];
        st32(&mut hdr[0..4], opcode);
        st32(&mut hdr[4..8], len);
        buf.extend_from_slice(&hdr);
        buf.extend_from_slice(payload);
    }

    #[test]
    fn empty_stream_done() {
        let mut c = 0;
        assert_eq!(
            decode_next_segment(&[], &mut c).unwrap_err(),
            DecodeStatus::Done
        );
        assert_eq!(c, 0);
    }

    #[test]
    fn single_blit_segment_with_record() {
        let mut payload = Vec::new();
        push_record(&mut payload, 0x12d, &[0u8; 0x18]); // buffer-to-buffer shape
        let mut stream = Vec::new();
        push_segment(&mut stream, SEGMENT_TYPE_BLIT, &payload);

        let segs = iter_segments(&stream).unwrap();
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].type_, SEGMENT_TYPE_BLIT);
        assert_eq!(segs[0].index, 0);

        let mut rc = 0;
        let rec = decode_first_record(&stream, &segs[0], &mut rc).unwrap();
        assert_eq!(rec.opcode, 0x12d);
        assert_eq!(
            decode_next_record(&stream, &segs[0], &mut rc).unwrap_err(),
            DecodeStatus::Done
        );
    }

    #[test]
    fn short_and_bad_length_name_the_check_that_refused() {
        use crate::observe::Refusal;
        // Asserting the slug rather than the variant is the point: both of these
        // used to be one `ErrBadLength`/`ErrShort` shared with sixteen other
        // checks, so a passing test said nothing about *which* read disagreed.
        assert_eq!(
            decode_next_segment(&[1, 2, 3], &mut 0)
                .unwrap_err()
                .refusal(),
            Some("stream_seg_short_header")
        );
        let mut bad = [0u8; 8];
        st32(&mut bad[0..4], 4); // length < header
        assert_eq!(
            decode_next_segment(&bad, &mut 0).unwrap_err().refusal(),
            Some("stream_seg_len_below_header")
        );
        // A segment header that outruns the buffer is a different bug from one
        // that undershoots its own header, and now says so.
        let mut past = [0u8; 8];
        st32(&mut past[0..4], 64);
        assert_eq!(
            decode_next_segment(&past, &mut 0).unwrap_err().refusal(),
            Some("stream_seg_len_past_buffer_end")
        );
    }

    #[test]
    fn end_of_stream_and_end_of_segment_are_never_refusals() {
        use crate::observe::Refusal;
        // `Done` is how both walkers terminate. If it ever reported a reason the
        // sink would carry one line per segment per frame — the flood that the
        // speculative-return carve-out exists to prevent.
        assert_eq!(DecodeStatus::Done.refusal(), None);
        assert_eq!(DecodeStatus::Ok.refusal(), None);

        let mut stream = Vec::new();
        push_segment(&mut stream, SEGMENT_TYPE_RENDER, &[]);
        let segs = iter_segments(&stream).unwrap();
        let mut c = 0;
        assert_eq!(
            decode_first_record(&stream, &segs[0], &mut c)
                .unwrap_err()
                .refusal(),
            None
        );
        let mut sc = stream.len();
        assert_eq!(
            decode_next_segment(&stream, &mut sc).unwrap_err().refusal(),
            None
        );
    }

    #[test]
    fn every_refusal_in_this_decoder_carries_a_registered_slug() {
        use crate::observe::Refusal;
        // What this pins is that no site returns a refusal whose payload is
        // empty or absent, which would render `reason=` bare.
        for status in [
            DecodeStatus::ErrArgs("stream_seg_cursor_past_end"),
            DecodeStatus::ErrShort("stream_seg_short_header"),
            DecodeStatus::ErrBadLength("stream_bytes_len_overflow"),
        ] {
            let slug = status.refusal().expect("a refusal names its check");
            assert!(
                slug.starts_with("stream_"),
                "{slug} lacks the module prefix"
            );
        }
    }

    #[test]
    fn multi_segment_indices() {
        let mut stream = Vec::new();
        push_segment(&mut stream, SEGMENT_TYPE_RENDER, &[]);
        push_segment(&mut stream, SEGMENT_TYPE_COMPUTE, &[]);
        let segs = iter_segments(&stream).unwrap();
        assert_eq!(segs[0].index, 0);
        assert_eq!(segs[1].index, 1);
        assert_eq!(segment_type_name(0), "render");
    }

    #[test]
    fn property_fuzz_random_headers() {
        // Smoke: random-ish short buffers must not panic.
        for n in 0..32usize {
            let bytes = vec![0xAAu8; n];
            let mut c = 0;
            let _ = decode_next_segment(&bytes, &mut c);
        }
    }
}
