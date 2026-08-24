//! Golden-vector and cross-module smoke tests for protocol packages.
//!
//! The vectors were originally extracted from the C decoder unit-test matrices
//! under `host/utils`. That tree is deleted, so there is nothing left to run a
//! differential C↔Rust comparison against: these expectations are now the
//! source of truth for the values they cover, not a copy of one.

use reims_vgpu::runtime::decode::{blit, event, stream};
use reims_vgpu_core::endian::{st32, st64};
use reims_vgpu_core::pixel_format;

/// Never share the live product logs with a concurrent boot.
fn isolate_logs() {
    reims_vgpu::observe::redirect_logs_for_tests();
}

#[test]
fn pixel_format_c_matrix_rows() {
    isolate_logs();
    // IOSurface row expectations, from the deleted C pixel-format matrix, read
    // through the mapper rail's own row-bytes rule rather than a second copy of
    // it that only this vector reached.
    use reims_vgpu_protocol::packed_span_estimate;
    assert_eq!(
        packed_span_estimate(pixel_format::MTL_FORMAT_BGRA8_UNORM, 200, 1),
        Some(896)
    );
    assert_eq!(
        packed_span_estimate(pixel_format::MTL_FORMAT_RGBA16_FLOAT, 200, 1),
        Some(1664)
    );
}

#[test]
fn stream_segment_record_roundtrip_shape() {
    isolate_logs();
    let mut payload = Vec::new();
    // record: opcode 0x12d, length 0x28
    let mut rec = vec![0u8; 0x28];
    st32(&mut rec[0..], 0x12d);
    st32(&mut rec[4..], 0x28);
    payload.extend_from_slice(&rec);

    let mut stream_bytes = Vec::new();
    let seg_len = (8 + payload.len()) as u32;
    let mut hdr = [0u8; 8];
    st32(&mut hdr[0..], seg_len);
    hdr[4] = stream::SEGMENT_TYPE_BLIT;
    stream_bytes.extend_from_slice(&hdr);
    stream_bytes.extend_from_slice(&payload);

    let segs = stream::iter_segments(&stream_bytes).unwrap();
    assert_eq!(segs.len(), 1);
    let mut c = 0;
    let rec = stream::decode_first_record(&stream_bytes, &segs[0], &mut c).unwrap();
    assert_eq!(rec.opcode, 0x12d);
    let blit_cmd = blit::decode(&stream_bytes[rec.offset as usize..]).unwrap();
    assert_eq!(blit_cmd.opcode, 0x12d);
}

#[test]
fn event_signal_golden() {
    isolate_logs();
    let mut v = vec![0u8; 0x14];
    st32(&mut v[0..], 0x191);
    st32(&mut v[4..], 0x14);
    st32(&mut v[8..], 3);
    st64(&mut v[12..], 0x42);
    let cmd = event::decode(&v).unwrap();
    assert_eq!(cmd.event_ref, 3);
    assert_eq!(cmd.value, 0x42);
}

#[test]
fn corpus_property_random_decode_no_panic() {
    isolate_logs();
    // Smoke fuzz: random-ish buffers through all byte parsers.
    let seeds: &[&[u8]] = &[
        &[],
        &[0],
        &[0xff; 7],
        &[0x00; 64],
        &[0x12, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00],
    ];
    for s in seeds {
        let _ = stream::iter_segments(s);
        let _ = blit::decode(s);
        let _ = event::decode(s);
        let _ = reims_vgpu::runtime::decode::compute::decode(s);
        let _ = reims_vgpu::runtime::decode::render::decode(s);
        let _ = reims_vgpu::runtime::decode::resource::decode_list_object_entry(s);
    }
}
