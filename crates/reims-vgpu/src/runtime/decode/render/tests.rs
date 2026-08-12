/// A malformed render command used to be dropped at the dispatch site with no
/// log line at all — indistinguishable from a segment carrying no render
/// work. Each check names itself now, `Ok` still produces nothing, and the
/// prefix keeps them apart from the six sibling `DecodeStatus` enums.
#[test]
fn every_render_decode_failure_names_its_own_check() {
    use crate::observe::Refusal;
    const ERRS: &[DecodeStatus] = &[
        DecodeStatus::ErrShort,
        DecodeStatus::ErrUnknownOpcode,
        DecodeStatus::ErrUnsupportedOpcode,
        DecodeStatus::ErrBadLength,
    ];
    let mut slugs: Vec<&str> = ERRS.iter().filter_map(|s| s.refusal()).collect();
    assert_eq!(slugs.len(), ERRS.len(), "every error variant refuses");
    assert!(slugs.iter().all(|s| s.starts_with("render_decode_")));
    slugs.sort_unstable();
    let n = slugs.len();
    slugs.dedup();
    assert_eq!(slugs.len(), n, "two render decode checks share a slug");
}
use super::*;
use crate::contract::endian::st32;

fn hdr(op: u32, len: usize) -> Vec<u8> {
    let mut v = vec![0u8; len];
    st32(&mut v[0..4], op);
    st32(&mut v[4..8], len as u32);
    v
}

#[test]
fn pipeline_and_draw() {
    let mut v = hdr(wire::OPCODE_SET_RENDER_PIPELINE_STATE, 12);
    st32(&mut v[8..], 9);
    let c = decode(&v).unwrap();
    assert_eq!(c.pipeline_ref, 9);

    // The compact draw form gets its own test below, against captured
    // bytes rather than a fixture shaped like the code.
}

/// Opcode `0x1` is the COMPACT `drawPrimitives:vertexStart:vertexCount:` —
/// `alloc(1, 8)`, so wire sz `0x10` and an 8-byte payload of
/// `u32 primitiveType · u16 vertexStart · u16 vertexCount`.
///
/// These payload bytes are the contract's, from the encoder's field order
/// plus the checked-in corpus record: `03 00 00 00 00 00 06 00` = triangle list,
/// vertexStart 0, vertexCount 6.
///
/// This is the test that fails without the fix. The old fixture was a
/// synthetic 24-byte record with four u32s, which is neither the compact nor
/// the wide form — so the decoder rejected every real compact draw as
/// `ErrShort` and the test agreed with it.
#[test]
fn compact_draw_layout_is_the_contracts_eight_byte_payload() {
    let v: [u8; 16] = [
        0x01, 0x00, 0x00, 0x00, // opcode 0x1
        0x10, 0x00, 0x00, 0x00, // sz 0x10
        0x03, 0x00, 0x00, 0x00, // primitiveType 3 (triangle list)
        0x00, 0x00, // vertexStart 0
        0x06, 0x00, // vertexCount 6
    ];
    let c = decode(&v).expect("the contract's compact draw must decode");
    assert_eq!(c.kind, Kind::Draw);
    assert_eq!(c.primitive_type, 3);
    assert_eq!(c.vertex_start, 0);
    assert_eq!(c.vertex_count, 6);
    assert_eq!(c.instance_count, 1, "the non-instanced selector draws once");

    // A nonzero start must survive: the device offsets both the stage-in
    // fetch and `[[vertex_id]]` from it, so reading it from the wrong offset
    // renders the wrong vertices rather than nothing.
    let mut v2 = v;
    v2[12] = 0x02;
    v2[14] = 0x04;
    let c2 = decode(&v2).expect("nonzero start decodes");
    assert_eq!((c2.vertex_start, c2.vertex_count), (2, 4));
}

/// Any other length for opcode `0x1` is not a form this contract knows, so
/// it is refused by name rather than read at a guessed offset.
#[test]
fn a_compact_draw_of_the_wrong_length_is_refused_not_guessed() {
    let mut wide = hdr(wire::OPCODE_DRAW, 24);
    st32(&mut wide[8..], 3);
    assert_eq!(decode(&wide), Err(DecodeStatus::ErrBadLength));
    let short = hdr(wire::OPCODE_DRAW, 12);
    assert_eq!(decode(&short), Err(DecodeStatus::ErrBadLength));
}

/// The wide form is a *different opcode*, emitted when either count does
/// not fit 16 bits, and it keeps the compact form's field order rather than
/// the instanced forms': `primitiveType` leads and is 32-bit.
///
/// This arm used to decline by name, which lost the draw but said so. The
/// layout its comment proposed by analogy — two `u64`s then a trailing
/// `primitiveType` — was the wrong one of the two candidates, so decoding on
/// that reasoning would have drawn a wrong primitive from a wrong offset.
/// The bytes below are `reims-vgpu-wire`'s `render_draw_primitives_wide`
/// fixture shape: Apple's serializer emits exactly this for
/// `(Triangle, start 0x11111, count 0x22222)`.
#[test]
fn the_wide_draw_form_decodes_with_the_compact_forms_field_order() {
    use crate::contract::endian::st64;

    let mut v = hdr(wire::OPCODE_DRAW_WIDE, wire::DRAW_WIDE_TOTAL_LEN as usize);
    st32(&mut v[8..], 3); // primitiveType, 32-bit and FIRST
    st64(&mut v[12..], 0x11111); // vertexStart
    st64(&mut v[20..], 0x22222); // vertexCount
    let c = decode(&v).expect("the wide draw decodes");
    assert_eq!(c.kind, Kind::Draw);
    assert_eq!(c.primitive_type, 3);
    assert_eq!(c.vertex_start, 0x11111);
    assert_eq!(c.vertex_count, 0x22222);
    assert_eq!(c.instance_count, 1, "the non-instanced selector draws once");

    // Reading it with the instanced forms' order would have taken
    // `primitiveType` from the last four bytes, which are a count's high
    // half and read zero. A regression to that guess shows up here.
    assert_ne!(c.primitive_type, 0);

    let short = hdr(
        wire::OPCODE_DRAW_WIDE,
        wire::DRAW_WIDE_TOTAL_LEN as usize - 4,
    );
    assert_eq!(decode(&short), Err(DecodeStatus::ErrBadLength));
}

/// A wide count above 32 bits is refused rather than truncated.
///
/// `Command` carries 32-bit counts, and the wide encoding exists because a
/// value passed 16 bits, not 32 — so this cannot arise from a real draw.
/// Truncating would silently draw different geometry, which is the class
/// this decoder's named refusals exist to prevent.
#[test]
fn a_wide_count_that_cannot_fit_the_commands_field_is_refused_not_truncated() {
    use crate::contract::endian::st64;

    let mut v = hdr(wire::OPCODE_DRAW_WIDE, wire::DRAW_WIDE_TOTAL_LEN as usize);
    st32(&mut v[8..], 3);
    st64(&mut v[12..], 0);
    st64(&mut v[20..], 0x1_0000_0000);
    assert_eq!(decode(&v), Err(DecodeStatus::ErrCountOutOfRange));
}

/// The compact indexed draw carries its index type on the wire, in the two
/// bytes that used to be read as the upper half of `primitiveType`.
///
/// Both readings agree while the guest uses 16-bit indices, because
/// `MTLIndexTypeUInt16` is ordinal 0. With `MTLIndexTypeUInt32` the head
/// reads `04 00 01 00`, and the old arm produced `primitiveType = 0x10004`
/// — no such Metal primitive — while separately reporting UInt16 for a
/// 32-bit index buffer. Apple's serializer emits exactly these bytes for
/// `(TriangleStrip, count 0x1111, UInt32, ref, offset 0x2222)`; the wire
/// Every selector that carries an instance count leaves it non-zero, which
/// is the guarantee three `.max(1)`s downstream of here used to re-apply —
/// two in `runtime::exec`, one in `runtime::draw`. They are gone, so
/// this is now the only thing holding that property up. A decode arm added
/// without [`wire_instance_count`] fails here rather than in a boot.
#[test]
fn no_decoded_draw_leaves_a_zero_instance_count() {
    use crate::contract::endian::st16;

    // The two compact instanced forms, each with the wire carrying zero.
    let mut inst = hdr(
        wire::OPCODE_DRAW_INSTANCED,
        wire::DRAW_INSTANCED_TOTAL_LEN as usize,
    );
    st16(&mut inst[8..], 0); // vertexStart
    st16(&mut inst[10..], 3); // vertexCount
    st16(&mut inst[12..], 0); // instanceCount — the case under test
    st16(&mut inst[14..], 3); // primitiveType
    assert_eq!(
        decode(&inst).expect("instanced draw").instance_count,
        1,
        "a wire zero is clamped here or nowhere"
    );

    let mut ix = hdr(
        wire::OPCODE_DRAW_INDEXED_INSTANCED,
        wire::DRAW_INDEXED_INSTANCED_TOTAL_LEN as usize,
    );
    st16(&mut ix[8..], 3); // primitiveType
    st16(&mut ix[10..], 0); // indexType
    st32(&mut ix[12..], 1); // index buffer ref
    st16(&mut ix[16..], 3); // index count
    st16(&mut ix[18..], 0); // index buffer offset
    st16(&mut ix[20..], 0); // instanceCount — the case under test
    assert_eq!(
        decode(&ix).expect("indexed instanced draw").instance_count,
        1
    );

    // And the non-instanced selectors, which carry no count at all: the
    // API's own default is one instance, not zero.
    let mut compact = hdr(wire::OPCODE_DRAW, DRAW_COMPACT_CMD_LEN);
    st32(&mut compact[8..], 3);
    st16(&mut compact[12..], 0);
    st16(&mut compact[14..], 3);
    assert_eq!(decode(&compact).expect("compact draw").instance_count, 1);
}

/// crate's `render_draw_indexed_uint32` fixture is the same capture.
#[test]
fn a_compact_indexed_draw_reads_its_index_type_from_the_wire() {
    use crate::contract::endian::st16;

    let mut v = hdr(
        wire::OPCODE_DRAW_INDEXED,
        wire::DRAW_INDEXED_TOTAL_LEN as usize,
    );
    st16(&mut v[8..], 4); // primitiveType, 16-bit
    st16(&mut v[10..], 1); // indexType UInt32
    st32(&mut v[12..], 0x141f); // index buffer ref
    st16(&mut v[16..], 0x1111); // index count
    st16(&mut v[18..], 0x2222); // index buffer offset

    let c = decode(&v).expect("compact indexed draw");
    assert_eq!(c.kind, Kind::Draw);
    assert_eq!(
        c.primitive_type, 4,
        "primitiveType must not absorb indexType"
    );
    assert_eq!(c.index_type, 1, "UInt32 must survive rather than reading 0");
    assert_eq!(c.index_buffer_ref, 0x141f);
    assert_eq!(c.index_count, 0x1111);
    assert_eq!(c.index_buffer_offset, 0x2222);
    assert_eq!(c.instance_count, 1);

    // The instanced sibling appends a 16-bit instance count and changes
    // nothing before it.
    let mut w = hdr(
        wire::OPCODE_DRAW_INDEXED_INSTANCED,
        wire::DRAW_INDEXED_INSTANCED_TOTAL_LEN as usize,
    );
    w[8..20].copy_from_slice(&v[8..20]);
    st16(&mut w[20..], 0x3333);
    let ci = decode(&w).expect("compact instanced indexed draw");
    assert_eq!(ci.primitive_type, 4);
    assert_eq!(ci.index_type, 1);
    assert_eq!(ci.index_count, 0x1111);
    assert_eq!(ci.instance_count, 0x3333);
}

#[test]
fn compact_instanced_draw_layout_live_webkit_bytes() {
    // Live x86 WebKit content record (aneesiqbal.ai), boot serial-20260717-161608:
    //   03000000 10000000 | 00000400 0d000400
    // op 0x3, sz 0x10, payload = vs0 vc4 inst13 primTriStrip(4).
    let v: [u8; 16] = [
        0x03, 0x00, 0x00, 0x00, // opcode 0x3
        0x10, 0x00, 0x00, 0x00, // sz 0x10
        0x00, 0x00, // vertexStart 0
        0x04, 0x00, // vertexCount 4
        0x0d, 0x00, // instanceCount 13
        0x04, 0x00, // primitiveType 4 (triangle strip)
    ];
    let c = decode(&v).expect("compact instanced draw");
    assert_eq!(c.kind, Kind::Draw);
    assert_eq!(c.vertex_start, 0);
    assert_eq!(c.vertex_count, 4);
    assert_eq!(c.instance_count, 13);
    assert_eq!(c.primitive_type, 4);
    // Not misread as an indexed draw.
    assert_eq!(c.index_count, 0);
    assert_eq!(c.index_buffer_ref, 0);
}

/// The six draw forms that carry a base instance used to reach
/// `Kind::OtherAccepted` and execute nothing — two whole Metal selectors
/// dropped, plus the wide encodings of two more, each wearing the shape of
/// an accepted state-set.
///
/// This walks all twelve draw opcodes and asserts none of them lands in the
/// catch-all. A new opcode in `0x00..=0x0b` that nobody decodes fails here
/// rather than going quiet.
#[test]
fn no_draw_opcode_falls_through_to_the_accepted_catch_all() {
    for (opcode, total) in [
        (wire::OPCODE_DRAW_WIDE, wire::DRAW_WIDE_TOTAL_LEN),
        (wire::OPCODE_DRAW, wire::DRAW_TOTAL_LEN),
        (
            wire::OPCODE_DRAW_INSTANCED_WIDE,
            wire::DRAW_INSTANCED_WIDE_TOTAL_LEN,
        ),
        (wire::OPCODE_DRAW_INSTANCED, wire::DRAW_INSTANCED_TOTAL_LEN),
        (
            wire::OPCODE_DRAW_INSTANCED_BASE_WIDE,
            wire::DRAW_INSTANCED_BASE_WIDE_TOTAL_LEN,
        ),
        (
            wire::OPCODE_DRAW_INSTANCED_BASE,
            wire::DRAW_INSTANCED_BASE_TOTAL_LEN,
        ),
        (
            wire::OPCODE_DRAW_INDEXED_WIDE,
            wire::DRAW_INDEXED_WIDE_TOTAL_LEN,
        ),
        (wire::OPCODE_DRAW_INDEXED, wire::DRAW_INDEXED_TOTAL_LEN),
        (
            wire::OPCODE_DRAW_INDEXED_INSTANCED_WIDE,
            wire::DRAW_INDEXED_INSTANCED_WIDE_TOTAL_LEN,
        ),
        (
            wire::OPCODE_DRAW_INDEXED_INSTANCED,
            wire::DRAW_INDEXED_INSTANCED_TOTAL_LEN,
        ),
        (
            wire::OPCODE_DRAW_INDEXED_INSTANCED_BASE_WIDE,
            wire::DRAW_INDEXED_INSTANCED_BASE_WIDE_TOTAL_LEN,
        ),
        (
            wire::OPCODE_DRAW_INDEXED_INSTANCED_BASE,
            wire::DRAW_INDEXED_INSTANCED_BASE_TOTAL_LEN,
        ),
    ] {
        let v = hdr(opcode, total as usize);
        let c = decode(&v).unwrap_or_else(|e| panic!("opcode {opcode:#x} refused: {e:?}"));
        assert_eq!(
            c.kind,
            Kind::Draw,
            "opcode {opcode:#x} is a draw and must not decode as {:?}",
            c.kind
        );
    }
}

/// `drawPrimitives:…:instanceCount:baseInstance:`, the compact form.
///
/// Metal offsets `[[instance_id]]` and every per-instance vertex fetch from
/// `baseInstance`, so dropping it draws the same instance repeatedly rather
/// than drawing nothing — which is why this was invisible until the
/// selector was decoded at all.
#[test]
fn a_base_instance_draw_carries_its_base_instance() {
    use crate::contract::endian::st16;

    let mut v = hdr(
        wire::OPCODE_DRAW_INSTANCED_BASE,
        wire::DRAW_INSTANCED_BASE_TOTAL_LEN as usize,
    );
    st16(&mut v[8..], 1); // vertexStart
    st16(&mut v[10..], 2); // vertexCount
    st16(&mut v[12..], 3); // instanceCount
    st16(&mut v[14..], 4); // baseInstance
    st16(&mut v[16..], 3); // primitiveType, last and 16-bit
    let c = decode(&v).expect("base-instance draw");
    assert_eq!(c.kind, Kind::Draw);
    assert_eq!(c.vertex_start, 1);
    assert_eq!(c.vertex_count, 2);
    assert_eq!(c.instance_count, 3);
    assert_eq!(c.base_instance, 4);
    assert_eq!(c.primitive_type, 3);
}

/// The two indexed forms with a base vertex put the buffer offset BEFORE
/// the index count, which their four siblings do not.
///
/// Reading them with the siblings' order swaps the two, drawing the wrong
/// number of indices from the wrong place — and both are plausible values,
/// so nothing downstream would refuse it. The base vertex is signed, so a
/// small negative offset must not read as an index near 65535.
#[test]
fn the_full_indexed_draw_puts_its_offset_before_its_count_and_signs_its_base_vertex() {
    use crate::contract::endian::st16;

    let mut v = hdr(
        wire::OPCODE_DRAW_INDEXED_INSTANCED_BASE,
        wire::DRAW_INDEXED_INSTANCED_BASE_TOTAL_LEN as usize,
    );
    st16(&mut v[8..], 3); // primitiveType
    st16(&mut v[10..], 1); // indexType UInt32
    st32(&mut v[12..], 0x141f); // index buffer ref
    st16(&mut v[16..], 0x2222); // index buffer OFFSET first
    st16(&mut v[18..], 0x1111); // index count second
    st16(&mut v[20..], 0x3333); // instanceCount
    st16(&mut v[22..], 0xfffe); // baseVertex = -2, two's complement
    st16(&mut v[24..], 0x55); // baseInstance

    let c = decode(&v).expect("full indexed draw");
    assert_eq!(c.kind, Kind::Draw);
    assert_eq!(c.index_buffer_offset, 0x2222, "offset comes first here");
    assert_eq!(c.index_count, 0x1111, "count comes second here");
    assert_eq!(c.index_type, 1);
    assert_eq!(c.index_buffer_ref, 0x141f);
    assert_eq!(c.instance_count, 0x3333);
    assert_eq!(c.base_instance, 0x55);
    assert_eq!(
        c.base_vertex, -2,
        "a negative base vertex must stay negative"
    );
}

/// The wide form of the same record, whose base vertex is sign-extended to
/// 64 bits rather than truncated to 16.
#[test]
fn the_wide_full_indexed_draw_sign_extends_its_base_vertex() {
    use crate::contract::endian::{st16, st64};

    let mut v = hdr(
        wire::OPCODE_DRAW_INDEXED_INSTANCED_BASE_WIDE,
        wire::DRAW_INDEXED_INSTANCED_BASE_WIDE_TOTAL_LEN as usize,
    );
    st16(&mut v[8..], 3);
    st16(&mut v[10..], 0);
    st32(&mut v[12..], 0x141f);
    st64(&mut v[16..], 0x2222); // offset first
    st64(&mut v[24..], 0x1111); // count second
    st64(&mut v[32..], 0x10000); // instanceCount, the argument that widened it
    st64(&mut v[40..], (-70000i64) as u64); // baseVertex
    st64(&mut v[48..], 0x55); // baseInstance

    let c = decode(&v).expect("wide full indexed draw");
    assert_eq!(c.index_buffer_offset, 0x2222);
    assert_eq!(c.index_count, 0x1111);
    assert_eq!(c.instance_count, 0x10000);
    assert_eq!(c.base_vertex, -70000);
    assert_eq!(c.base_instance, 0x55);
}

#[test]
fn wide_indexed_draw_layout() {
    use crate::contract::endian::st16;

    let mut v = hdr(wire::OPCODE_DRAW_INDEXED_WIDE, 0x20);
    st16(&mut v[8..], 3); // triangle
    st16(&mut v[10..], 0); // UInt16
    st32(&mut v[12..], 0x3e); // index buffer ref
    st32(&mut v[16..], 6); // index count
    st32(&mut v[24..], 0x10100); // byte offset

    let c = decode(&v).expect("wide indexed draw");
    assert_eq!(c.kind, Kind::Draw);
    assert_eq!(c.primitive_type, 3);
    assert_eq!(c.index_type, 0);
    assert_eq!(c.index_buffer_ref, 0x3e);
    assert_eq!(c.index_count, 6);
    assert_eq!(c.index_buffer_offset, 0x10100);
    assert_eq!(c.instance_count, 1);
}

#[test]
fn execute_commands_range_and_indirect() {
    use crate::contract::endian::st64;
    // 0x15 withRange: ref + unaligned location/length
    let mut v = hdr(wire::OPCODE_EXECUTE_COMMANDS_RANGE, EXECUTE_RANGE_CMD_LEN);
    st32(&mut v[8..], 0x3333);
    st64(&mut v[12..], 5);
    st64(&mut v[20..], 7);
    let c = decode(&v).unwrap();
    assert_eq!(c.kind, Kind::ExecuteCommands);
    assert!(c.icb_is_range);
    assert_eq!(c.indirect_command_buffer_ref, 0x3333);
    assert_eq!(c.icb_range_location, 5);
    assert_eq!(c.icb_range_length, 7);
    // 0x14 indirect buffer form
    let mut v = hdr(
        wire::OPCODE_EXECUTE_COMMANDS_INDIRECT,
        EXECUTE_INDIRECT_CMD_LEN,
    );
    st32(&mut v[8..], 0x1111);
    st32(&mut v[12..], 0x2222);
    st64(&mut v[16..], 0x40);
    let c = decode(&v).unwrap();
    assert!(!c.icb_is_range);
    assert_eq!(c.indirect_command_buffer_ref, 0x1111);
    assert_eq!(c.icb_args_buffer_ref, 0x2222);
    assert_eq!(c.icb_args_buffer_offset, 0x40);
}

#[test]
fn depth_and_stencil_pass_slots() {
    use crate::contract::endian::{st16, st32, st64};
    use crate::contract::pass_action::{MTL_LOAD_ACTION_CLEAR, MTL_STORE_ACTION_STORE};
    let mut payload = vec![0u8; PASS_MIN_PAYLOAD];
    // depth @0
    st32(
        &mut payload[PASS_DEPTH_ATTACH_OFF + PASS_ATTACH_TEXREF..],
        77,
    );
    st16(
        &mut payload[PASS_DEPTH_ATTACH_OFF + PASS_ATTACH_LOAD_ACTION..],
        MTL_LOAD_ACTION_CLEAR,
    );
    st16(
        &mut payload[PASS_DEPTH_ATTACH_OFF + PASS_ATTACH_STORE_ACTION..],
        MTL_STORE_ACTION_STORE,
    );
    st64(
        &mut payload[PASS_DEPTH_ATTACH_OFF + PASS_DEPTH_ATTACH_CLEAR_DEPTH..],
        0.5f64.to_bits(),
    );
    // stencil @0x28
    st32(
        &mut payload[PASS_STENCIL_ATTACH_OFF + PASS_ATTACH_TEXREF..],
        88,
    );
    st32(
        &mut payload[PASS_STENCIL_ATTACH_OFF + PASS_STENCIL_ATTACH_CLEAR_STENCIL..],
        9,
    );
    let d = decode_depth_attachment(&payload);
    assert_eq!(d.texture_ref, 77);
    assert!((d.clear_depth - 0.5).abs() < 1e-9);
    let s = decode_stencil_attachment(&payload);
    assert_eq!(s.texture_ref, 88);
    assert_eq!(s.clear_stencil, 9);
}

/// Each of the first two records ends where the next one begins: depth
/// `[0x00, 0x28)`, stencil `[0x28, 0x4c)`. A payload that carries both in
/// full — and not one byte of the color section — must decode both.
///
/// A shared `PASS_DEPTH_STENCIL_ATTACH_STRIDE = 0x28` used to give the
/// stencil record the depth record's length, so the decoder demanded 0x50
/// bytes to read a 0x24-byte record and sliced 4 bytes past its end, over
/// color slot 0's texture ref. This payload is exactly `PASS_COLOR_ATTACH_OFF`
/// long, so the old guard rejected it and returned a defaulted attachment.
#[test]
fn depth_and_stencil_records_end_where_the_next_section_begins() {
    use crate::contract::endian::{st32, st64};
    let mut payload = vec![0u8; PASS_COLOR_ATTACH_OFF];
    st32(
        &mut payload[PASS_DEPTH_ATTACH_OFF + PASS_ATTACH_TEXREF..],
        31,
    );
    st64(
        &mut payload[PASS_DEPTH_ATTACH_OFF + PASS_DEPTH_ATTACH_CLEAR_DEPTH..],
        0.25f64.to_bits(),
    );
    st32(
        &mut payload[PASS_STENCIL_ATTACH_OFF + PASS_ATTACH_TEXREF..],
        32,
    );
    st32(
        &mut payload[PASS_STENCIL_ATTACH_OFF + PASS_STENCIL_ATTACH_CLEAR_STENCIL..],
        0xfe,
    );
    let d = decode_depth_attachment(&payload);
    assert_eq!(
        d.texture_ref, 31,
        "depth record is complete at {PASS_STENCIL_ATTACH_OFF} bytes"
    );
    assert!((d.clear_depth - 0.25).abs() < 1e-9);
    let s = decode_stencil_attachment(&payload);
    assert_eq!(
        s.texture_ref, 32,
        "stencil record is complete at {PASS_COLOR_ATTACH_OFF} bytes"
    );
    assert_eq!(s.clear_stencil, 0xfe);
}

/// Whether a scissor reaches the whole target is one rule, and it had been
/// written twice in opposite polarities.
///
/// The draw-coverage census asked
/// `x == 0 && y == 0 && w >= target_w && h >= target_h`; the partial-store
/// path asked `x > 0 || y > 0 || w < width || h < height` and acted on the
/// negation. Two spellings of one predicate, in two files, over four
/// numbers that travelled loose. Each row below is a case where a term
/// dropped from either spelling would change the answer.
#[test]
fn a_scissor_covers_its_target_only_when_every_term_says_so() {
    let full = ScissorRect {
        x: 0,
        y: 0,
        width: 800,
        height: 600,
    };
    assert!(full.covers(800, 600), "an exact fit covers");
    assert!(
        ScissorRect {
            width: 900,
            height: 700,
            ..full
        }
        .covers(800, 600),
        "a scissor larger than the target still covers it"
    );
    for (name, rect) in [
        ("offset x", ScissorRect { x: 1, ..full }),
        ("offset y", ScissorRect { y: 1, ..full }),
        ("narrow", ScissorRect { width: 799, ..full }),
        (
            "short",
            ScissorRect {
                height: 599,
                ..full
            },
        ),
    ] {
        assert!(
            !rect.covers(800, 600),
            "a scissor {name} must not read as covering the target"
        );
    }

    // A zero-extent rect draws nothing; the stream decode drops it and
    // keeps the previous scissor rather than binding an empty one.
    assert!(!full.is_empty());
    assert!(ScissorRect { width: 0, ..full }.is_empty());
    assert!(ScissorRect { height: 0, ..full }.is_empty());
}

/// All four fields of the prefix decide bindability, and both attachment
/// shapes hand all four to the rule.
///
/// The rule had two consumers and the second carried its own copy testing
/// `level` and `resolve_texture_ref` only — so a depth buffer bound at
/// slice 5 was refused by the stream decode and would have been accepted by
/// the Metal rail. This drives each field on its own, from both shapes, so
/// a consumer that reconstructs three of the four fails here rather than at
/// a guest that binds an array layer.
///
/// `level` is driven from both [`LevelSupport`] arms, because it is the one
/// field whose answer depends on which rail is asking: a colour attachment's
/// level resolves to its own plane in the guest allocation and a depth
/// attachment's does not. Every other field must refuse on both arms — an arm
/// that reads `AnyLevel` as "anything goes" fails here.
#[test]
fn every_field_of_the_attachment_prefix_decides_bindability() {
    for levels in [LevelSupport::LevelZeroOnly, LevelSupport::AnyLevel] {
        assert!(
            attachment_subresource_is_bindable(AttachSubresource::default(), levels),
            "{levels:?}: the whole texture at level 0, slice 0, plane 0 with no resolve is bindable"
        );
    }
    let mip = AttachSubresource {
        level: 1,
        ..AttachSubresource::default()
    };
    assert!(
        !attachment_subresource_is_bindable(mip, LevelSupport::LevelZeroOnly),
        "a rail that only renders level 0 must refuse a level the guest named"
    );
    assert!(
        attachment_subresource_is_bindable(mip, LevelSupport::AnyLevel),
        "a rail that resolves the named level's own plane must admit it"
    );

    for (name, sub) in [
        (
            "slice",
            AttachSubresource {
                slice: 5,
                ..AttachSubresource::default()
            },
        ),
        (
            "depth_plane",
            AttachSubresource {
                depth_plane: 2,
                ..AttachSubresource::default()
            },
        ),
        (
            "resolve_texture_ref",
            AttachSubresource {
                resolve_texture_ref: 99,
                ..AttachSubresource::default()
            },
        ),
    ] {
        for levels in [LevelSupport::LevelZeroOnly, LevelSupport::AnyLevel] {
            assert!(
                !attachment_subresource_is_bindable(sub, levels),
                "{levels:?}: a non-default {name} must refuse the attachment on its own"
            );
        }
    }

    // And both shapes hand all four to the rule: a field a conversion drops
    // arrives as 0, which is the value that admits.
    let all_four = AttachSubresource {
        level: 1,
        slice: 5,
        depth_plane: 2,
        resolve_texture_ref: 99,
    };
    assert_eq!(
        AttachSubresource::from(DepthAttachment {
            texture_ref: 77,
            level: 1,
            slice: 5,
            depth_plane: 2,
            resolve_texture_ref: 99,
            ..DepthAttachment::default()
        }),
        all_four
    );
    assert_eq!(
        AttachSubresource::from(StencilAttachment {
            texture_ref: 88,
            level: 1,
            slice: 5,
            depth_plane: 2,
            resolve_texture_ref: 99,
            ..StencilAttachment::default()
        }),
        all_four
    );
}

#[test]
fn blend_color_and_cull() {
    let mut v = hdr(wire::OPCODE_SET_BLEND_COLOR, 24);
    // RGBA as f32 bits
    st32(&mut v[8..], 1.0f32.to_bits());
    st32(&mut v[12..], 0.0f32.to_bits());
    st32(&mut v[16..], 0.0f32.to_bits());
    st32(&mut v[20..], 1.0f32.to_bits());
    let c = decode(&v).unwrap();
    assert_eq!(c.kind, Kind::SetBlendColor);
    assert!((c.blend_color[0] - 1.0).abs() < 1e-6);

    // Mode state is one NSUInteger on the wire (SET_MODE_TOTAL_LEN = 16).
    let mut v = hdr(
        wire::OPCODE_SET_CULL_MODE,
        wire::SET_MODE_TOTAL_LEN as usize,
    );
    st32(&mut v[8..], 2);
    let c = decode(&v).unwrap();
    assert_eq!(c.kind, Kind::SetCullMode);
    assert_eq!(c.cull_mode, 2);
}

/// Above the window is refused; inside it but unclaimed is not.
///
/// These are different answers and the difference is the whole reason the
/// bound moved. `0x99` used to be refused here because it was one past
/// `0x98`, the highest opcode this project had *seen* -- and `0xa5`/`0xa6`
/// were refused with it, which lost four real vertex binds. `0x99` is now
/// inside the encoder's range and unclaimed, so it reaches the catch-all
/// and is reported as a gap rather than denied.
#[test]
fn an_opcode_above_the_window_is_refused_and_one_inside_it_is_not() {
    assert!(opcode_above_the_encoder_window(0xff));
    assert_eq!(
        decode(&hdr(0xff, 16)).unwrap_err(),
        DecodeStatus::ErrUnsupportedOpcode
    );
    // One past the highest opcode Apple's serializer writes here.
    assert!(opcode_above_the_encoder_window(
        wire::OPCODE_SET_VERTEX_BUFFER_OFFSET_STRIDE + 1
    ));
    assert_eq!(
        decode(&hdr(wire::OPCODE_SET_VERTEX_BUFFER_OFFSET_STRIDE + 1, 16)).unwrap_err(),
        DecodeStatus::ErrUnsupportedOpcode
    );
    // Inside the range, claimed by no arm. `OtherAccepted` is what says so.
    //
    // Found rather than named. A literal here goes stale the moment that
    // opcode is decoded, which already happened once: `0x99` was the
    // unclaimed probe until `setVertexAmplificationMode:value:` turned out
    // to be exactly that number.
    let unclaimed = super::unclaimed_accepted_opcode();
    assert_eq!(
        decode(&hdr(unclaimed, 16))
            .unwrap_or_else(|e| panic!("op {unclaimed:#x}: {e:?}"))
            .kind,
        Kind::OtherAccepted
    );
}

/// Every bind opcode refuses `count == 0` and refuses more entries than the
/// slot table holds, so a decoded bind record ALWAYS carries at least one
/// entry.
///
/// `exec::apply_record` relies on this: it walks `buffer_binds` / `ref_binds`
/// directly, with no single-entry wire form to fall back to. If a zero-count
/// record ever decoded successfully, those loops would silently bind nothing.
#[test]
fn a_bind_record_never_decodes_to_zero_entries() {
    for (op, entry_size) in [
        (wire::OPCODE_SET_VERTEX_BUFFER, BUFFER_BIND_ENTRY_SIZE),
        (wire::OPCODE_SET_FRAGMENT_BUFFER, BUFFER_BIND_ENTRY_SIZE),
        (wire::OPCODE_SET_VERTEX_TEXTURE, REF_BIND_ENTRY_SIZE),
        (wire::OPCODE_SET_FRAGMENT_TEXTURE, REF_BIND_ENTRY_SIZE),
        (wire::OPCODE_SET_VERTEX_SAMPLER, REF_BIND_ENTRY_SIZE),
        (wire::OPCODE_SET_FRAGMENT_SAMPLER, REF_BIND_ENTRY_SIZE),
    ] {
        let hdr_len = 8;
        let body = |count: u32, entries: usize| {
            let mut v = hdr(op, hdr_len + BIND_ENTRIES + entries * entry_size);
            st32(&mut v[hdr_len + BIND_FIRST..], 0);
            st32(&mut v[hdr_len + BIND_COUNT..], count);
            v
        };
        assert_eq!(
            decode(&body(0, 0)).unwrap_err(),
            DecodeStatus::ErrBadLength,
            "op {op:#x} accepted count=0"
        );
        // A count with no entries behind it. The record is the guest's own
        // length claim, so this is the bound — and it must not wrap.
        assert_eq!(
            decode(&body(4, 3)).unwrap_err(),
            DecodeStatus::ErrShort,
            "op {op:#x} accepted a record one entry shorter than its count"
        );
        assert_eq!(
            decode(&body(u32::MAX, 1)).unwrap_err(),
            DecodeStatus::ErrShort,
            "op {op:#x} accepted a count whose entries cannot exist"
        );
        let c = decode(&body(1, 1)).unwrap_or_else(|e| panic!("op {op:#x}: {e:?}"));
        assert_eq!(
            c.buffer_binds.len() + c.ref_binds.len(),
            1,
            "op {op:#x} decoded one entry into neither list"
        );
        // Forty slots, which Apple produces (`setVertexTextures:withRange:`
        // over a range of 40) and a 32-entry cap used to refuse whole.
        let c = decode(&body(40, 40)).unwrap_or_else(|e| panic!("op {op:#x} refused 40: {e:?}"));
        assert_eq!(
            c.buffer_binds.len() + c.ref_binds.len(),
            40,
            "op {op:#x} did not decode all forty entries"
        );
    }
}

/// The six unapplied states decode to their own kinds, at their own widths.
///
/// Two are sixteen-byte records with a `u64`, two are twelve-byte records
/// with an `f32`, and the colour store action is the only one of the three
/// store forms that carries an index -- depth and stencil have one
/// attachment each. Reading a float record as a `f64` would take four bytes
/// of whatever followed it, which is why the length is asserted here rather
/// than assumed from the family.
#[test]
fn each_unapplied_state_decodes_at_its_own_width() {
    use crate::contract::endian::{st32, st64};
    use reims_vgpu_wire::ops::render as wire;

    for (op, wire_op) in [
        (
            wire::OPCODE_SET_DEPTH_CLIP_MODE,
            wire::OPCODE_SET_DEPTH_CLIP_MODE,
        ),
        (
            wire::OPCODE_SET_TRIANGLE_FILL_MODE,
            wire::OPCODE_SET_TRIANGLE_FILL_MODE,
        ),
        (wire::OPCODE_SET_LINE_WIDTH, wire::OPCODE_SET_LINE_WIDTH),
        (
            wire::OPCODE_SET_TESSELLATION_FACTOR_SCALE,
            wire::OPCODE_SET_TESSELLATION_FACTOR_SCALE,
        ),
        (
            wire::OPCODE_SET_COLOR_STORE_ACTION,
            wire::OPCODE_SET_COLOR_STORE_ACTION,
        ),
        (
            wire::OPCODE_SET_DEPTH_STORE_ACTION,
            wire::OPCODE_SET_DEPTH_STORE_ACTION,
        ),
        (
            wire::OPCODE_SET_STENCIL_STORE_ACTION,
            wire::OPCODE_SET_STENCIL_STORE_ACTION,
        ),
        (wire::OPCODE_TEXTURE_BARRIER, wire::OPCODE_TEXTURE_BARRIER),
    ] {
        assert_eq!(op, wire_op, "the serializer writes a different opcode");
    }

    // The `u64` mode records.
    for op in [
        wire::OPCODE_SET_DEPTH_CLIP_MODE,
        wire::OPCODE_SET_TRIANGLE_FILL_MODE,
    ] {
        let mut v = hdr(op, wire::SET_MODE_TOTAL_LEN as usize);
        st64(&mut v[OP_HEADER_LEN..], 1);
        let c = decode(&v).unwrap_or_else(|e| panic!("op {op:#x}: {e:?}"));
        assert_eq!(c.kind, Kind::SetRasterState, "op {op:#x}");
        assert_eq!(c.mode, 1, "op {op:#x}");
        assert_eq!(
            decode(&hdr(op, OP_HEADER_LEN + 4)).unwrap_err(),
            DecodeStatus::ErrShort,
            "op {op:#x} read a mode out of four bytes"
        );
    }

    // The `f32` records: twelve bytes, and the payload is four.
    for op in [
        wire::OPCODE_SET_LINE_WIDTH,
        wire::OPCODE_SET_TESSELLATION_FACTOR_SCALE,
    ] {
        let total = wire::SET_FLOAT_TOTAL_LEN as usize;
        assert_eq!(total, OP_HEADER_LEN + 4, "op {op:#x}");
        let mut v = hdr(op, total);
        st32(&mut v[OP_HEADER_LEN..], 2.5f32.to_bits());
        let c = decode(&v).unwrap_or_else(|e| panic!("op {op:#x}: {e:?}"));
        assert_eq!(c.kind, Kind::SetFloatState, "op {op:#x}");
        assert_eq!(c.float_value, 2.5, "op {op:#x}");
    }

    // The colour store action carries an index; the other two do not, and
    // the values are deliberately unequal so a swap is visible.
    let mut v = hdr(wire::OPCODE_SET_COLOR_STORE_ACTION, 16);
    st32(&mut v[OP_HEADER_LEN..], 2);
    st32(&mut v[OP_HEADER_LEN + 4..], 3);
    let c = decode(&v).expect("colour store action");
    assert_eq!(c.kind, Kind::SetStoreAction);
    assert_eq!((c.mode, c.first), (2, 3), "action and index are swapped");

    for op in [
        wire::OPCODE_SET_DEPTH_STORE_ACTION,
        wire::OPCODE_SET_STENCIL_STORE_ACTION,
    ] {
        let mut v = hdr(op, 16);
        st64(&mut v[OP_HEADER_LEN..], 1);
        let c = decode(&v).unwrap_or_else(|e| panic!("op {op:#x}: {e:?}"));
        assert_eq!(c.kind, Kind::SetStoreAction, "op {op:#x}");
        assert_eq!(c.mode, 1, "op {op:#x}");
        assert_eq!(c.first, 0, "op {op:#x} invented an attachment index");
    }

    // `textureBarrier` is the header alone and joins the barrier kind.
    let c = decode(&hdr(wire::OPCODE_TEXTURE_BARRIER, OP_HEADER_LEN)).expect("texture barrier");
    assert_eq!(c.kind, Kind::Barrier);
}

/// A vertex bind carrying an attribute stride binds the buffer.
///
/// It used to be refused outright: `0xa5`/`0xa6` are above the old
/// `wire::OPCODE_SET_VERTEX_BUFFER_OFFSET_STRIDE` of `0x98`, so `opcode_is_apple_rejected` called them
/// records Apple does not emit -- and Apple emits them whenever the guest
/// negotiates `supportsDynamicAttributeStride`. Every strided vertex bind
/// was dropped and the buffer never bound, which is the sampler-LOD bug
/// again with a worse consequence.
///
/// The plural case is the load-bearing one. Its two entries are twenty
/// bytes apart, and a decoder using the plain twelve would read the second
/// entry out of the middle of the first -- so both offsets are distinct and
/// both are asserted.
#[test]
fn a_vertex_bind_with_an_attribute_stride_binds_the_buffer_rather_than_being_refused() {
    use crate::contract::endian::{st32, st64};
    use reims_vgpu_wire::ops::render as wire;

    for (op, wire_op) in [
        (
            wire::OPCODE_SET_VERTEX_BUFFER_STRIDE,
            wire::OPCODE_SET_VERTEX_BUFFER_STRIDE,
        ),
        (
            wire::OPCODE_SET_VERTEX_BUFFER_OFFSET_STRIDE,
            wire::OPCODE_SET_VERTEX_BUFFER_OFFSET_STRIDE,
        ),
    ] {
        assert_eq!(op, wire_op, "the serializer writes a different opcode");
        assert!(
            !opcode_above_the_encoder_window(op),
            "op {op:#x} is still called a record Apple does not emit, and Apple emits it"
        );
    }

    // Two slots, twenty bytes apart: {ref u32, offset u64, stride u64}.
    let entries = 2usize;
    let total = OP_HEADER_LEN + BIND_ENTRIES + entries * BUFFER_STRIDE_BIND_ENTRY_SIZE;
    assert_eq!(total, 56, "the plural fixture is 56 bytes");
    let mut v = hdr(wire::OPCODE_SET_VERTEX_BUFFER_STRIDE, total);
    st32(&mut v[OP_HEADER_LEN + BIND_FIRST..], 9);
    st32(&mut v[OP_HEADER_LEN + BIND_COUNT..], entries as u32);
    for (i, (r, off, stride)) in [(5151u32, 0x3333u64, 0x5555u64), (5252, 0x4444, 0x6666)]
        .into_iter()
        .enumerate()
    {
        let e = OP_HEADER_LEN + BIND_ENTRIES + i * BUFFER_STRIDE_BIND_ENTRY_SIZE;
        st32(&mut v[e..], r);
        st64(&mut v[e + 4..], off);
        st64(&mut v[e + 12..], stride);
    }
    let c = decode(&v).expect("a strided vertex bind must decode");
    assert_eq!(c.kind, Kind::SetBuffer);
    assert_eq!(c.stage, Stage::Vertex, "there is no fragment stride form");
    assert!(c.has_attribute_stride);
    assert_eq!(c.first, 9);
    assert_eq!(
        c.buffer_binds,
        vec![
            DecodedBufferBind {
                buffer_ref: 5151,
                offset: 0x3333,
                attribute_stride: Some(0x5555),
            },
            DecodedBufferBind {
                buffer_ref: 5252,
                offset: 0x4444,
                attribute_stride: Some(0x6666),
            }
        ],
        "the entry stride is 20, not the plain bind's 12, and the third field of \
         each entry is the stride rather than padding stepped over"
    );

    // The plain bind must not be told it carries one, or every ordinary
    // vertex bind would report a loss it did not have.
    let plain_total = OP_HEADER_LEN + BIND_ENTRIES + BUFFER_BIND_ENTRY_SIZE;
    let mut p = hdr(wire::OPCODE_SET_VERTEX_BUFFER, plain_total);
    st32(&mut p[OP_HEADER_LEN + BIND_COUNT..], 1);
    st32(&mut p[OP_HEADER_LEN + BIND_ENTRIES..], 5151);
    let c = decode(&p).expect("plain vertex bind");
    assert!(!c.has_attribute_stride);

    // `0xa6` is the offset re-point with the stride appended: 28 bytes.
    let total = wire::SET_BUFFER_OFFSET_STRIDE_TOTAL_LEN as usize;
    let mut v = hdr(wire::OPCODE_SET_VERTEX_BUFFER_OFFSET_STRIDE, total);
    st32(&mut v[OP_HEADER_LEN..], 8);
    st64(&mut v[OP_HEADER_LEN + 4..], 0x4567);
    st64(&mut v[OP_HEADER_LEN + 12..], 0x5678);
    let c = decode(&v).expect("a strided offset re-point must decode");
    assert_eq!(c.kind, Kind::SetBufferOffset);
    assert_eq!(c.stage, Stage::Vertex);
    assert!(c.has_attribute_stride);
    assert_eq!((c.first, c.buffer_offset), (8, 0x4567));
    // Short of its own stride word it is refused, rather than read as the
    // twenty-byte record it is not.
    assert_eq!(
        decode(&hdr(
            wire::OPCODE_SET_VERTEX_BUFFER_OFFSET_STRIDE,
            total - 8
        ))
        .unwrap_err(),
        DecodeStatus::ErrShort
    );
}

/// Vertex amplification decodes at two widths the type encoding hides.
///
/// The mode record declares both arguments `Q` and puts both on the wire at
/// 32 bits, so a decoder written from the encoding reads a 24-byte record
/// that is 16. The count record's head is four bytes rather than the
/// eight-byte bind header every other counted record here uses, so reading
/// a `BindHeader` takes the first mapping's viewport offset as the count --
/// which is why the fixture's count and first offset are deliberately
/// unequal and this test asserts on both.
#[test]
fn vertex_amplification_decodes_at_the_widths_the_serializer_wrote() {
    use crate::contract::endian::st32;
    use reims_vgpu_wire::ops::render as wire;

    for (op, wire_op) in [
        (
            wire::OPCODE_SET_VERTEX_AMPLIFICATION_MODE,
            wire::OPCODE_SET_VERTEX_AMPLIFICATION_MODE,
        ),
        (
            wire::OPCODE_SET_VERTEX_AMPLIFICATION_COUNT,
            wire::OPCODE_SET_VERTEX_AMPLIFICATION_COUNT,
        ),
    ] {
        assert_eq!(op, wire_op, "the serializer writes a different opcode");
    }

    let total = wire::SET_VERTEX_AMPLIFICATION_MODE_TOTAL_LEN as usize;
    assert_eq!(total, OP_HEADER_LEN + 8, "two u32, not two u64");
    let mut v = hdr(wire::OPCODE_SET_VERTEX_AMPLIFICATION_MODE, total);
    st32(&mut v[OP_HEADER_LEN..], 0x5555);
    st32(&mut v[OP_HEADER_LEN + 4..], 0x6666);
    let c = decode(&v).expect("amplification mode");
    assert_eq!(c.kind, Kind::SetVertexAmplification);
    assert_eq!(
        (c.mode, c.amplification_value),
        (0x5555, 0x6666),
        "mode and value are crossed"
    );

    // Two mappings, four distinct offsets. The count is 2 and the first
    // viewport offset is 0x1111, so a four-byte head cannot be confused
    // with an eight-byte one.
    let mappings = 2usize;
    let total = OP_HEADER_LEN + AMPLIFICATION_COUNT_LEN + mappings * AMPLIFICATION_MAPPING_SIZE;
    assert_eq!(total, 28, "the fixture is 28 bytes");
    let mut v = hdr(wire::OPCODE_SET_VERTEX_AMPLIFICATION_COUNT, total);
    st32(&mut v[OP_HEADER_LEN..], mappings as u32);
    for (i, (vp, rt)) in [(0x1111u32, 0x2222u32), (0x3333, 0x4444)]
        .into_iter()
        .enumerate()
    {
        let e = OP_HEADER_LEN + AMPLIFICATION_COUNT_LEN + i * AMPLIFICATION_MAPPING_SIZE;
        st32(&mut v[e..], vp);
        st32(&mut v[e + 4..], rt);
    }
    let c = decode(&v).expect("amplification count");
    assert_eq!(c.kind, Kind::SetVertexAmplification);
    assert_eq!(c.count, 2, "the head was read as eight bytes");

    // A count with no mappings behind it. The record is the guest's own
    // length claim, so this is the bound, and it must not wrap.
    let mut short = hdr(wire::OPCODE_SET_VERTEX_AMPLIFICATION_COUNT, total);
    st32(&mut short[OP_HEADER_LEN..], 3);
    assert_eq!(decode(&short).unwrap_err(), DecodeStatus::ErrShort);
    let mut huge = hdr(wire::OPCODE_SET_VERTEX_AMPLIFICATION_COUNT, total);
    st32(&mut huge[OP_HEADER_LEN..], u32::MAX);
    assert_eq!(decode(&huge).unwrap_err(), DecodeStatus::ErrShort);
}

/// `0x0c` is two records, and only the length says which.
///
/// The plain wide patch draw is 56 bytes and the indexed one is 68, under
/// the same opcode — the one place in this family where a wide form does
/// not get its own number. The two bodies agree for nine bytes and diverge
/// at the tenth, so a decoder that dispatched on the opcode and read either
/// body unconditionally would misread the other half the time rather than
/// refuse it.
///
/// The third arm is the one that matters most: a `0x0c` at any *other*
/// length has no reading at all, and picking the nearer of the two would
/// take one record's buffer ref as the other's offset. It must be refused.
#[test]
fn the_wide_patch_draw_opcode_is_resolved_by_length_and_refused_without_one() {
    use reims_vgpu_wire::ops::render as wire;

    for (op, wire_op) in [
        (wire::OPCODE_DRAW_PATCHES, wire::OPCODE_DRAW_PATCHES),
        (
            wire::OPCODE_DRAW_PATCHES_WIDE,
            wire::OPCODE_DRAW_PATCHES_WIDE,
        ),
        (
            wire::OPCODE_DRAW_INDEXED_PATCHES,
            wire::OPCODE_DRAW_INDEXED_PATCHES,
        ),
        (
            wire::OPCODE_DRAW_PATCHES_INDIRECT,
            wire::OPCODE_DRAW_PATCHES_INDIRECT,
        ),
        (
            wire::OPCODE_DRAW_INDEXED_PATCHES_INDIRECT,
            wire::OPCODE_DRAW_INDEXED_PATCHES_INDIRECT,
        ),
    ] {
        assert_eq!(op, wire_op, "the serializer writes a different opcode");
    }

    // The two wide lengths really are different, which is what makes the
    // length usable as a discriminator at all.
    assert_ne!(
        wire::DRAW_PATCHES_WIDE_TOTAL_LEN,
        wire::DRAW_INDEXED_PATCHES_WIDE_TOTAL_LEN
    );

    for total in [
        wire::DRAW_PATCHES_WIDE_TOTAL_LEN as usize,
        wire::DRAW_INDEXED_PATCHES_WIDE_TOTAL_LEN as usize,
    ] {
        let c = decode(&hdr(wire::OPCODE_DRAW_PATCHES_WIDE, total))
            .unwrap_or_else(|e| panic!("0x0c at {total} bytes: {e:?}"));
        assert_eq!(c.kind, Kind::DrawPatches);
        assert_eq!(c.command_length as usize, total);
    }

    // Between the two, above both, and below both: none has a reading.
    for total in [
        wire::DRAW_PATCHES_WIDE_TOTAL_LEN as usize + 4,
        wire::DRAW_INDEXED_PATCHES_WIDE_TOTAL_LEN as usize + 4,
        wire::DRAW_PATCHES_WIDE_TOTAL_LEN as usize - 4,
    ] {
        assert_eq!(
            decode(&hdr(wire::OPCODE_DRAW_PATCHES_WIDE, total)).unwrap_err(),
            DecodeStatus::ErrBadLength,
            "0x0c at {total} bytes was given a reading; it has none"
        );
    }

    // The four single-length forms, each accepted at its own length and
    // refused four bytes short. A patch draw read short is invented
    // geometry, not a smaller draw.
    for (op, total) in [
        (
            wire::OPCODE_DRAW_PATCHES,
            wire::DRAW_PATCHES_TOTAL_LEN as usize,
        ),
        (
            wire::OPCODE_DRAW_INDEXED_PATCHES,
            wire::DRAW_INDEXED_PATCHES_TOTAL_LEN as usize,
        ),
        (
            wire::OPCODE_DRAW_PATCHES_INDIRECT,
            wire::DRAW_PATCHES_INDIRECT_TOTAL_LEN as usize,
        ),
        (
            wire::OPCODE_DRAW_INDEXED_PATCHES_INDIRECT,
            wire::DRAW_INDEXED_PATCHES_INDIRECT_TOTAL_LEN as usize,
        ),
    ] {
        let c = decode(&hdr(op, total)).unwrap_or_else(|e| panic!("op {op:#x}: {e:?}"));
        assert_eq!(c.kind, Kind::DrawPatches, "op {op:#x}");
        assert_eq!(
            decode(&hdr(op, total - 4)).unwrap_err(),
            DecodeStatus::ErrBadLength,
            "op {op:#x} accepted a record four bytes short"
        );
    }

    // A record whose header promises more bytes than are present is
    // refused by the framing, before any of the above runs.
    let mut truncated = hdr(
        wire::OPCODE_DRAW_PATCHES,
        wire::DRAW_PATCHES_TOTAL_LEN as usize,
    );
    truncated.truncate(truncated.len() - 4);
    assert!(decode(&truncated).is_err(), "a short buffer was accepted");
}

/// The store-action options and the tessellation factor buffer.
///
/// `0x67`/`0x6a`/`0x79` sit one opcode above the three store actions and
/// look like longer forms of them. They are not, and the difference is a
/// *width*: the options are a `u64` where the action is a `u32`, so the
/// colour form's attachment index moves from payload `+4` to `+8` and the
/// record grows from 16 bytes to 20. A decoder that reused
/// `ColorStoreAction` here would read the index out of the options' high
/// half and report attachment 0 for every one of them.
///
/// `0x7a` is checked in the same test because it makes the opposite
/// mistake available: it names a buffer with an offset, so it reads like a
/// bind, and a reader that took a `BindHeader` would call the ref `first`
/// and the low half of the offset `count`.
/// A colour attachment's `level` is sixteen bits, and `slice` is the
/// sixteen above it.
///
/// This arm read `ld32` at [`PASS_ATTACH_LEVEL`] for as long as it existed,
/// under a comment on the depth arm that stated the rule as "the archive
/// uses u16 for depth/stencil level (color uses u32)". Apple's own bytes say
/// all three attachment shapes are identical through their prefix, so the
/// wide read returned `level | (slice << 16)`: a pass rendering into array
/// slice 1 reported mip level 65536 and lost the slice.
///
/// The synthetic here is what a cube-face pass looks like — level 1, slice
/// 5 — and the two fields are read apart. The `0xffff` case is the same
/// claim at the boundary: a slice that fills its half must not reach the
/// level at all.
#[test]
fn a_colour_attachments_level_does_not_swallow_its_slice() {
    for (level, slice, plane) in [(1u16, 5u16, 2u16), (0, 0xffff, 0), (0xffff, 0, 0)] {
        let total = OP_HEADER_LEN + PASS_MIN_PAYLOAD;
        let mut cmd = vec![0u8; total];
        st32(&mut cmd[0..], wire_pass::OPCODE_RENDER_PASS);
        st32(&mut cmd[4..], total as u32);
        let slot = OP_HEADER_LEN + PASS_COLOR_ATTACH_OFF;
        st32(&mut cmd[slot + PASS_ATTACH_TEXREF..], 7);
        cmd[slot + PASS_ATTACH_LEVEL..slot + PASS_ATTACH_LEVEL + 2]
            .copy_from_slice(&level.to_le_bytes());
        cmd[slot + PASS_ATTACH_SLICE..slot + PASS_ATTACH_SLICE + 2]
            .copy_from_slice(&slice.to_le_bytes());
        cmd[slot + PASS_ATTACH_DEPTH_PLANE..slot + PASS_ATTACH_DEPTH_PLANE + 2]
            .copy_from_slice(&plane.to_le_bytes());
        let att = decode_color_attachment(&cmd[OP_HEADER_LEN..], 0);
        assert_eq!(att.level, u32::from(level), "level took the slice's bits");
        assert_eq!(att.slice, u32::from(slice), "slice went unread");
        assert_eq!(att.depth_plane, u32::from(plane), "depth plane went unread");
    }
}

/// The pass's tail is four fields this device decodes and does not apply.
///
/// A record short of the tail must leave all four at zero rather than
/// reading past its own payload — the decoder accepts a payload as small as
/// one colour slot, and Apple's own record is 584 bytes.
#[test]
fn the_render_pass_tail_is_read_only_when_the_record_carries_one() {
    use crate::contract::endian::st64;
    let full = OP_HEADER_LEN + PASS_TAIL_OFF + 0x1c;
    let mut cmd = vec![0u8; full];
    st32(&mut cmd[0..], wire_pass::OPCODE_RENDER_PASS);
    st32(&mut cmd[4..], full as u32);
    let t = OP_HEADER_LEN + PASS_TAIL_OFF;
    st32(&mut cmd[t + PASS_TAIL_VISIBILITY_BUFFER_REF..], 5151);
    st64(&mut cmd[t + PASS_TAIL_ARRAY_LENGTH..], 0x11);
    st64(&mut cmd[t + PASS_TAIL_TARGET_WIDTH..], 0x1234);
    st64(&mut cmd[t + PASS_TAIL_TARGET_HEIGHT..], 0x5678);
    let c = decode(&cmd).expect("well formed");
    assert_eq!(c.kind, Kind::RenderPass);
    assert_eq!(c.pass_visibility_result_buffer_ref, 5151);
    assert_eq!(c.pass_render_target_array_length, 0x11);
    assert_eq!(c.pass_render_target_width, 0x1234);
    assert_eq!(c.pass_render_target_height, 0x5678);

    let short = OP_HEADER_LEN + PASS_MIN_PAYLOAD;
    let mut cmd = vec![0u8; short];
    st32(&mut cmd[0..], wire_pass::OPCODE_RENDER_PASS);
    st32(&mut cmd[4..], short as u32);
    let c = decode(&cmd).expect("well formed");
    assert_eq!(c.kind, Kind::RenderPass);
    assert_eq!(c.pass_render_target_width, 0, "read past the payload");
}

/// The six pass-property records decode rather than reaching the catch-all.
///
/// Every one sits inside the accepted opcode window, so before they were
/// named they were accepted, dropped and silent — which is exactly the
/// shape that hid the sampler-LOD binds.
#[test]
fn every_pass_property_record_reaches_an_arm_of_its_own() {
    for (op, payload_len) in [
        (wire_pass::OPCODE_DEFAULT_RASTER_SAMPLE_COUNT, 4usize),
        (wire_pass::OPCODE_RASTERIZATION_RATE_MAP, 4),
        (wire_pass::OPCODE_IMAGEBLOCK_SAMPLE_LENGTH, 4),
        (wire_pass::OPCODE_THREADGROUP_MEMORY_LENGTH, 4),
        (wire_pass::OPCODE_TILE_SIZE, 4),
    ] {
        let total = OP_HEADER_LEN + payload_len;
        let mut cmd = vec![0u8; total];
        st32(&mut cmd[0..], op);
        st32(&mut cmd[4..], total as u32);
        st32(&mut cmd[OP_HEADER_LEN..], 4);
        let c = decode(&cmd).expect("well formed");
        assert_eq!(c.kind, Kind::RenderPassProperty, "opcode {op:#x}");
        if op == wire_pass::OPCODE_RASTERIZATION_RATE_MAP {
            assert_eq!(c.texture_ref, 4, "opcode {op:#x}: ref");
        } else {
            assert_eq!(c.mode, 4, "opcode {op:#x}: scalar");
        }
        // A header that claims to be the whole record carries no scalar,
        // and that is a refusal rather than a zero.
        let mut bare = vec![0u8; OP_HEADER_LEN];
        st32(&mut bare[0..], op);
        st32(&mut bare[4..], OP_HEADER_LEN as u32);
        assert!(
            matches!(decode(&bare), Err(DecodeStatus::ErrBadLength)),
            "opcode {op:#x}"
        );
    }

    // Sample positions are head plus `count` pairs, and the count is guest
    // data: one claiming more pairs than the record holds is refused.
    let total = OP_HEADER_LEN + 4 + 2 * 8;
    let mut cmd = vec![0u8; total];
    st32(&mut cmd[0..], wire_pass::OPCODE_SAMPLE_POSITIONS);
    st32(&mut cmd[4..], total as u32);
    st32(&mut cmd[OP_HEADER_LEN..], 2);
    let c = decode(&cmd).expect("well formed");
    assert_eq!(c.kind, Kind::RenderPassProperty);
    assert_eq!(c.count, 2);
    st32(&mut cmd[OP_HEADER_LEN..], 0xffff_ffff);
    assert!(matches!(decode(&cmd), Err(DecodeStatus::ErrBadLength)));
}

#[test]
fn the_store_action_options_are_not_wider_store_actions() {
    use crate::contract::endian::{st32, st64};
    use reims_vgpu_wire::ops::render as wire;

    for (op, wire_op) in [
        (
            wire::OPCODE_SET_COLOR_STORE_ACTION_OPTIONS,
            wire::OPCODE_SET_COLOR_STORE_ACTION_OPTIONS,
        ),
        (
            wire::OPCODE_SET_DEPTH_STORE_ACTION_OPTIONS,
            wire::OPCODE_SET_DEPTH_STORE_ACTION_OPTIONS,
        ),
        (
            wire::OPCODE_SET_STENCIL_STORE_ACTION_OPTIONS,
            wire::OPCODE_SET_STENCIL_STORE_ACTION_OPTIONS,
        ),
        (
            wire::OPCODE_SET_TESSELLATION_FACTOR_BUFFER,
            wire::OPCODE_SET_TESSELLATION_FACTOR_BUFFER,
        ),
    ] {
        assert_eq!(op, wire_op, "the serializer writes a different opcode");
    }

    // Each options opcode is exactly one above its store action, and the
    // two are different records at different lengths. Asserting the
    // adjacency keeps a future edit from collapsing them into one arm.
    for (action, options) in [
        (
            wire::OPCODE_SET_COLOR_STORE_ACTION,
            wire::OPCODE_SET_COLOR_STORE_ACTION_OPTIONS,
        ),
        (
            wire::OPCODE_SET_DEPTH_STORE_ACTION,
            wire::OPCODE_SET_DEPTH_STORE_ACTION_OPTIONS,
        ),
        (
            wire::OPCODE_SET_STENCIL_STORE_ACTION,
            wire::OPCODE_SET_STENCIL_STORE_ACTION_OPTIONS,
        ),
    ] {
        assert_eq!(options, action + 1);
    }

    // The colour form. The index is at +8; a `u32` read of the options
    // would leave it at +4 and find the options' own high half.
    let total = wire::SET_COLOR_STORE_ACTION_OPTIONS_TOTAL_LEN as usize;
    let mut v = hdr(wire::OPCODE_SET_COLOR_STORE_ACTION_OPTIONS, total);
    st64(&mut v[OP_HEADER_LEN..], 0x1111);
    st32(&mut v[OP_HEADER_LEN + 8..], 3);
    let c = decode(&v).expect("colour store action options");
    assert_eq!(c.kind, Kind::SetStoreActionOptions);
    assert_eq!(
        (c.mode, c.first),
        (0x1111, 3),
        "the options and the attachment index are crossed"
    );

    // Depth and stencil have one attachment each and carry no index, so
    // their record is four bytes shorter than the colour form's.
    let total = wire::SET_STORE_ACTION_OPTIONS_TOTAL_LEN as usize;
    assert_eq!(
        total + 4,
        wire::SET_COLOR_STORE_ACTION_OPTIONS_TOTAL_LEN as usize
    );
    for (op, options) in [
        (wire::OPCODE_SET_DEPTH_STORE_ACTION_OPTIONS, 0x2222u64),
        (wire::OPCODE_SET_STENCIL_STORE_ACTION_OPTIONS, 0x3333),
    ] {
        let mut v = hdr(op, total);
        st64(&mut v[OP_HEADER_LEN..], options);
        let c = decode(&v).unwrap_or_else(|e| panic!("op {op:#x}: {e:?}"));
        assert_eq!(c.kind, Kind::SetStoreActionOptions);
        assert_eq!(c.mode, options, "op {op:#x}");
        assert_eq!(c.first, 0, "op {op:#x} invented an attachment index");
    }

    // `0x7a`: ref, then two `u64` that differ, so a crossed pair shows.
    let total = wire::SET_TESSELLATION_FACTOR_BUFFER_TOTAL_LEN as usize;
    let mut v = hdr(wire::OPCODE_SET_TESSELLATION_FACTOR_BUFFER, total);
    st32(&mut v[OP_HEADER_LEN..], 5151);
    st64(&mut v[OP_HEADER_LEN + 4..], 0x3456);
    st64(&mut v[OP_HEADER_LEN + 12..], 0x4567);
    let c = decode(&v).expect("tessellation factor buffer");
    assert_eq!(c.kind, Kind::SetTessellationFactorBuffer);
    assert_eq!(
        (c.buffer_ref, c.buffer_offset),
        (5151, 0x3456),
        "read as a bind header rather than as a ref and an offset"
    );

    for (op, total) in [
        (
            wire::OPCODE_SET_COLOR_STORE_ACTION_OPTIONS,
            wire::SET_COLOR_STORE_ACTION_OPTIONS_TOTAL_LEN as usize,
        ),
        (
            wire::OPCODE_SET_DEPTH_STORE_ACTION_OPTIONS,
            wire::SET_STORE_ACTION_OPTIONS_TOTAL_LEN as usize,
        ),
        (
            wire::OPCODE_SET_STENCIL_STORE_ACTION_OPTIONS,
            wire::SET_STORE_ACTION_OPTIONS_TOTAL_LEN as usize,
        ),
        (
            wire::OPCODE_SET_TESSELLATION_FACTOR_BUFFER,
            wire::SET_TESSELLATION_FACTOR_BUFFER_TOTAL_LEN as usize,
        ),
    ] {
        assert_eq!(
            decode(&hdr(op, total - 4)).unwrap_err(),
            DecodeStatus::ErrBadLength,
            "op {op:#x} accepted a record four bytes short"
        );
    }
}

/// The nine tile-shader opcodes leave the catch-all.
///
/// All nine were `Kind::OtherAccepted` together, so a guest running a tile
/// shader produced one deduped line naming a number and nothing that said a
/// dispatch or a bind had been lost. They are checked here against
/// `reims_vgpu_wire::ops::tile`'s constants, which fixtures pin against
/// bytes Apple's serializer produced.
///
/// Every value is distinct, because four of the five bind opcodes share a
/// record shape and differ only in entry stride: a decoder that took
/// `0xa0`'s twelve-byte entry for `0x9f`'s four would accept a record it
/// should refuse, and one that read `0x9e` as a bind header would take the
/// low half of its 64-bit offset as a count.
#[test]
fn a_tile_record_is_decoded_rather_than_accepted_without_a_claim() {
    use crate::contract::endian::{st32, st64};
    use reims_vgpu_wire::ops::tile as wire_tile;

    // The local constants and the serializer's, held together so neither
    // can drift. This is the check that would have caught `0x86`/`0x87`.
    for (op, wire_op) in [
        (
            wire_tile::OPCODE_DISPATCH_THREADS_PER_TILE,
            wire_tile::OPCODE_DISPATCH_THREADS_PER_TILE,
        ),
        (
            wire_tile::OPCODE_SET_TILE_BUFFER,
            wire_tile::OPCODE_SET_TILE_BUFFER,
        ),
        (
            wire_tile::OPCODE_SET_TILE_BUFFER_OFFSET,
            wire_tile::OPCODE_SET_TILE_BUFFER_OFFSET,
        ),
        (
            wire_tile::OPCODE_SET_TILE_SAMPLER,
            wire_tile::OPCODE_SET_TILE_SAMPLER,
        ),
        (
            wire_tile::OPCODE_SET_TILE_SAMPLER_LOD,
            wire_tile::OPCODE_SET_TILE_SAMPLER_LOD,
        ),
        (
            wire_tile::OPCODE_SET_TILE_TEXTURE,
            wire_tile::OPCODE_SET_TILE_TEXTURE,
        ),
        (
            wire_tile::OPCODE_DISPATCH_THREADS_PER_TILE_IN_REGION,
            wire_tile::OPCODE_DISPATCH_THREADS_PER_TILE_IN_REGION,
        ),
        (
            wire_tile::OPCODE_DISPATCH_THREADS_PER_TILE_IN_REGION_RT_INDEX,
            wire_tile::OPCODE_DISPATCH_THREADS_PER_TILE_IN_REGION_RT_INDEX,
        ),
        (
            wire_tile::OPCODE_GET_TILE_DIMENSIONS,
            wire_tile::OPCODE_GET_TILE_DIMENSIONS,
        ),
        (
            wire_tile::OPCODE_SET_TILE_THREADGROUP_MEMORY,
            wire_tile::OPCODE_SET_TILE_THREADGROUP_MEMORY,
        ),
    ] {
        assert_eq!(op, wire_op, "the serializer writes a different opcode");
    }

    // `0x9c`: length, offset, index. Three distinct values, because a
    // decoder that took the compute encoder's two-field namesake would read
    // the offset's low half as the index.
    let total = wire_tile::SET_TILE_THREADGROUP_MEMORY_TOTAL_LEN as usize;
    let mut v = hdr(wire_tile::OPCODE_SET_TILE_THREADGROUP_MEMORY, total);
    st64(&mut v[OP_HEADER_LEN..], 0x1234);
    st64(&mut v[OP_HEADER_LEN + 8..], 0x2345);
    st32(&mut v[OP_HEADER_LEN + 16..], 5);
    let c = decode(&v).expect("tile threadgroup memory");
    assert_eq!(c.kind, Kind::TileBind);
    assert_eq!((c.first, c.count), (5, 1));

    // `0x9b`: three unnarrowed `u64`, none of them equal.
    let total = wire_tile::DISPATCH_THREADS_PER_TILE_TOTAL_LEN as usize;
    let mut v = hdr(wire_tile::OPCODE_DISPATCH_THREADS_PER_TILE, total);
    st64(&mut v[OP_HEADER_LEN..], 0x11);
    st64(&mut v[OP_HEADER_LEN + 8..], 0x22);
    st64(&mut v[OP_HEADER_LEN + 16..], 0x33);
    let c = decode(&v).expect("tile dispatch");
    assert_eq!(c.kind, Kind::TileDispatch);
    assert_eq!(c.tile_threads, [0x11, 0x22, 0x33]);

    // `0xa2`/`0xa3`: the same nine `u64`, the grid first and the region
    // origin-before-size. Only `0xa3` writes the trailing `u32`, so the
    // decoder must read neither -- set those four bytes and require the
    // answer not to move on either opcode.
    let total = wire_tile::DISPATCH_THREADS_PER_TILE_IN_REGION_TOTAL_LEN as usize;
    for op in [
        wire_tile::OPCODE_DISPATCH_THREADS_PER_TILE_IN_REGION,
        wire_tile::OPCODE_DISPATCH_THREADS_PER_TILE_IN_REGION_RT_INDEX,
    ] {
        let mut v = hdr(op, total);
        for (i, value) in [0x11u64, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99]
            .into_iter()
            .enumerate()
        {
            st64(&mut v[OP_HEADER_LEN + i * 8..], value);
        }
        let c = decode(&v).expect("tile region dispatch");
        assert_eq!(c.kind, Kind::TileDispatch);
        assert_eq!(
            c.tile_threads,
            [0x11, 0x22, 0x33],
            "op {op:#x} took the region's origin for the grid"
        );

        let mut noisy = v.clone();
        noisy[OP_HEADER_LEN + wire_tile::REGION_RT_INDEX_OFFSET..][..4].fill(0xff);
        assert_eq!(
            decode(&noisy).expect("tile region dispatch"),
            c,
            "op {op:#x} let the trailing render-target index reach a field; on \
             0xa2 those four bytes are the guest's ring"
        );
    }

    // `0xa4`: a ref then a 64-bit offset, and it is a readback rather than
    // a bind -- the buffer is where the *host* writes.
    let total = wire_tile::GET_TILE_DIMENSIONS_TOTAL_LEN as usize;
    let mut v = hdr(wire_tile::OPCODE_GET_TILE_DIMENSIONS, total);
    st32(&mut v[OP_HEADER_LEN..], 5151);
    st64(&mut v[OP_HEADER_LEN + 4..], 0x9999);
    let c = decode(&v).expect("tile dimensions query");
    assert_eq!(c.kind, Kind::TileDimensionsQuery);
    assert_eq!((c.buffer_ref, c.buffer_offset), (5151, 0x9999));

    // `0x9e`: index then a 64-bit offset. Not a bind header -- a decoder
    // that read one would take the offset's low half as a count.
    let total = wire_tile::SET_TILE_BUFFER_OFFSET_TOTAL_LEN as usize;
    let mut v = hdr(wire_tile::OPCODE_SET_TILE_BUFFER_OFFSET, total);
    st32(&mut v[OP_HEADER_LEN..], 4);
    st64(&mut v[OP_HEADER_LEN + 4..], 0x2345);
    let c = decode(&v).expect("tile buffer offset");
    assert_eq!(c.kind, Kind::TileBind);
    assert_eq!((c.first, c.count, c.buffer_offset), (4, 1, 0x2345));

    // The four bind opcodes, each at its own entry stride. A two-slot
    // record is built at the right size and accepted, and the same record
    // one entry short is refused -- which is what separates "knows the
    // stride" from "accepts anything with a plausible head".
    for (op, entry_size) in [
        (wire_tile::OPCODE_SET_TILE_BUFFER, BUFFER_BIND_ENTRY_SIZE),
        (wire_tile::OPCODE_SET_TILE_TEXTURE, REF_BIND_ENTRY_SIZE),
        (wire_tile::OPCODE_SET_TILE_SAMPLER, REF_BIND_ENTRY_SIZE),
        (
            wire_tile::OPCODE_SET_TILE_SAMPLER_LOD,
            SAMPLER_LOD_BIND_ENTRY_SIZE,
        ),
    ] {
        let total = OP_HEADER_LEN + BIND_ENTRIES + 2 * entry_size;
        let mut v = hdr(op, total);
        st32(&mut v[OP_HEADER_LEN + BIND_FIRST..], 7);
        st32(&mut v[OP_HEADER_LEN + BIND_COUNT..], 2);
        let c = decode(&v).unwrap_or_else(|e| panic!("op {op:#x}: {e:?}"));
        assert_eq!(c.kind, Kind::TileBind, "op {op:#x}");
        assert_eq!((c.first, c.count), (7, 2), "op {op:#x}");

        let mut short = hdr(op, total - entry_size);
        st32(&mut short[OP_HEADER_LEN + BIND_FIRST..], 7);
        st32(&mut short[OP_HEADER_LEN + BIND_COUNT..], 2);
        assert_eq!(
            decode(&short).unwrap_err(),
            DecodeStatus::ErrShort,
            "op {op:#x} accepted a two-slot bind holding one entry"
        );

        // A zero count is not a bind; it is a record whose head does not
        // describe itself, the same refusal the other stages give.
        let mut empty = hdr(op, OP_HEADER_LEN + BIND_ENTRIES);
        st32(&mut empty[OP_HEADER_LEN + BIND_COUNT..], 0);
        assert_eq!(
            decode(&empty).unwrap_err(),
            DecodeStatus::ErrBadLength,
            "op {op:#x} accepted a bind of no slots"
        );
    }

    // The fixed-length forms are refused at any other length rather than
    // read short.
    for (op, total) in [
        (
            wire_tile::OPCODE_DISPATCH_THREADS_PER_TILE,
            wire_tile::DISPATCH_THREADS_PER_TILE_TOTAL_LEN as usize,
        ),
        (
            wire_tile::OPCODE_DISPATCH_THREADS_PER_TILE_IN_REGION,
            wire_tile::DISPATCH_THREADS_PER_TILE_IN_REGION_TOTAL_LEN as usize,
        ),
        (
            wire_tile::OPCODE_DISPATCH_THREADS_PER_TILE_IN_REGION_RT_INDEX,
            wire_tile::DISPATCH_THREADS_PER_TILE_IN_REGION_TOTAL_LEN as usize,
        ),
        (
            wire_tile::OPCODE_SET_TILE_BUFFER_OFFSET,
            wire_tile::SET_TILE_BUFFER_OFFSET_TOTAL_LEN as usize,
        ),
        (
            wire_tile::OPCODE_GET_TILE_DIMENSIONS,
            wire_tile::GET_TILE_DIMENSIONS_TOTAL_LEN as usize,
        ),
        (
            wire_tile::OPCODE_SET_TILE_THREADGROUP_MEMORY,
            wire_tile::SET_TILE_THREADGROUP_MEMORY_TOTAL_LEN as usize,
        ),
    ] {
        assert_eq!(
            decode(&hdr(op, total - 4)).unwrap_err(),
            DecodeStatus::ErrBadLength,
            "op {op:#x} accepted a record four bytes short"
        );
    }
}

/// The two indirect draws and the visibility mode leave the catch-all.
///
/// All three were `Kind::OtherAccepted`, which is an `Ok` that means "no arm
/// claimed this" -- so a guest's indirect draw produced one deduped line
/// naming a raw opcode and nothing that said a draw had been lost.
///
/// Every field here is a distinct value on purpose. Two of these three
/// records put an argument on the wire in the reverse of its selector
/// order, and `0x11` names two buffers and two offsets that are all the
/// same widths, so a decoder that crossed any pair would read back
/// plausible numbers and draw the wrong thing. Only distinct values catch
/// that, and the layouts they are checked against are
/// `reims_vgpu_wire::ops::render`'s, pinned by fixtures.
#[test]
fn an_indirect_draw_is_decoded_rather_than_accepted_without_a_claim() {
    use crate::contract::endian::{st16, st32, st64};
    use reims_vgpu_wire::ops::render as wire;

    for (op, wire_op) in [
        (wire::OPCODE_DRAW_INDIRECT, wire::OPCODE_DRAW_INDIRECT),
        (
            wire::OPCODE_DRAW_INDEXED_INDIRECT,
            wire::OPCODE_DRAW_INDEXED_INDIRECT,
        ),
        (
            wire::OPCODE_SET_VISIBILITY_RESULT_MODE,
            wire::OPCODE_SET_VISIBILITY_RESULT_MODE,
        ),
    ] {
        assert_eq!(op, wire_op, "the serializer writes a different opcode");
    }

    // `0x10`: offset first, then the buffer, then a 16-bit primitive type.
    let total = wire::DRAW_INDIRECT_TOTAL_LEN as usize;
    let mut v = hdr(wire::OPCODE_DRAW_INDIRECT, total);
    st64(&mut v[OP_HEADER_LEN..], 0x1111);
    st32(&mut v[OP_HEADER_LEN + 8..], 5151);
    st16(&mut v[OP_HEADER_LEN + 12..], 3);
    let c = decode(&v).expect("indirect draw");
    assert_eq!(c.kind, Kind::DrawIndirect);
    assert_eq!(c.indirect_buffer_offset, 0x1111);
    assert_eq!(c.indirect_buffer_ref, 5151);
    assert_eq!(c.primitive_type, 3);
    // The two bytes above `primitive_type` are never written by the
    // serializer, so a wider read would take them. Set them and require the
    // answer not to move; `no_decoder_reads_a_bit_apples_serializer_never_wrote`
    // makes the same check against Apple's own measured mask.
    let mut noisy = v.clone();
    noisy[OP_HEADER_LEN + 14] = 0xff;
    noisy[OP_HEADER_LEN + 15] = 0xff;
    assert_eq!(
        decode(&noisy).expect("indirect draw"),
        c,
        "the record's unwritten tail reached a field"
    );

    // `0x11`: both types lead as `u16`, both refs follow as `u32`, both
    // offsets trail as `u64` -- the blit family's shape, not `0x10`'s.
    let total = wire::DRAW_INDEXED_INDIRECT_TOTAL_LEN as usize;
    let mut v = hdr(wire::OPCODE_DRAW_INDEXED_INDIRECT, total);
    st16(&mut v[OP_HEADER_LEN..], 4);
    st16(&mut v[OP_HEADER_LEN + 2..], 1);
    st32(&mut v[OP_HEADER_LEN + 4..], 5151);
    st32(&mut v[OP_HEADER_LEN + 8..], 5252);
    st64(&mut v[OP_HEADER_LEN + 12..], 0x1111);
    st64(&mut v[OP_HEADER_LEN + 20..], 0x2222);
    let c = decode(&v).expect("indexed indirect draw");
    assert_eq!(c.kind, Kind::DrawIndirect);
    assert_eq!(c.primitive_type, 4);
    assert_eq!(c.index_type, 1, "index type read out of the primitive type");
    assert_eq!((c.index_buffer_ref, c.indirect_buffer_ref), (5151, 5252));
    assert_eq!(
        (c.index_buffer_offset, c.indirect_buffer_offset),
        (0x1111, 0x2222),
        "the two offsets are crossed"
    );

    // `0x84`: offset first, mode second, reversing the selector.
    let total = wire::SET_VISIBILITY_RESULT_MODE_TOTAL_LEN as usize;
    let mut v = hdr(wire::OPCODE_SET_VISIBILITY_RESULT_MODE, total);
    st64(&mut v[OP_HEADER_LEN..], 0x1234);
    st64(&mut v[OP_HEADER_LEN + 8..], 2);
    let c = decode(&v).expect("visibility result mode");
    assert_eq!(c.kind, Kind::SetVisibilityResultMode);
    assert_eq!(
        (c.visibility_result_offset, c.mode),
        (0x1234, 2),
        "mode and offset are swapped"
    );

    // Each is a fixed length the serializer always writes, so a record that
    // is not that length is refused rather than read short.
    for (op, total) in [
        (
            wire::OPCODE_DRAW_INDIRECT,
            wire::DRAW_INDIRECT_TOTAL_LEN as usize,
        ),
        (
            wire::OPCODE_DRAW_INDEXED_INDIRECT,
            wire::DRAW_INDEXED_INDIRECT_TOTAL_LEN as usize,
        ),
        (
            wire::OPCODE_SET_VISIBILITY_RESULT_MODE,
            wire::SET_VISIBILITY_RESULT_MODE_TOTAL_LEN as usize,
        ),
    ] {
        assert_eq!(
            decode(&hdr(op, total - 4)).unwrap_err(),
            DecodeStatus::ErrBadLength,
            "op {op:#x} accepted a record four bytes short"
        );
    }
}

/// The plural viewport and scissor records are the singular ones behind a
/// count, and the two counts are not the same width.
///
/// Eight bytes for scissor, four for viewport -- from selectors that both
/// declare `Q`, so only the capture settles it. Borrowing either constant
/// for the other reads the first entry four bytes off, which for a scissor
/// is `x` taken from the high half of the count.
#[test]
fn a_plural_viewport_or_scissor_is_the_singular_record_behind_its_own_count() {
    use crate::contract::endian::{st32, st64};
    use reims_vgpu_wire::ops::render as wire;

    assert_eq!(
        wire::OPCODE_SET_SCISSOR_RECTS,
        wire::OPCODE_SET_SCISSOR_RECTS
    );
    assert_eq!(wire::OPCODE_SET_VIEWPORTS, wire::OPCODE_SET_VIEWPORTS);
    assert_ne!(
        SCISSOR_RECTS_COUNT_LEN, VIEWPORTS_COUNT_LEN,
        "the two counts are different widths; that is the whole hazard"
    );

    // Two rects, and the *first* is the one this rail keeps.
    let total = OP_HEADER_LEN + SCISSOR_RECTS_COUNT_LEN + 2 * SCISSOR_PAYLOAD_LEN;
    let mut v = hdr(wire::OPCODE_SET_SCISSOR_RECTS, total);
    st64(&mut v[OP_HEADER_LEN..], 2);
    let e0 = OP_HEADER_LEN + SCISSOR_RECTS_COUNT_LEN;
    for (i, val) in [0x11u64, 0x22, 0x33, 0x44].into_iter().enumerate() {
        st64(&mut v[e0 + i * 8..], val);
    }
    let e1 = e0 + SCISSOR_PAYLOAD_LEN;
    for (i, val) in [0x55u64, 0x66, 0x77, 0x88].into_iter().enumerate() {
        st64(&mut v[e1 + i * 8..], val);
    }
    let c = decode(&v).expect("two scissor rects");
    assert_eq!(c.kind, Kind::SetScissor);
    assert_eq!(c.count, 2);
    assert_eq!(
        c.scissors,
        vec![
            ScissorRect {
                x: 0x11,
                y: 0x22,
                width: 0x33,
                height: 0x44
            },
            ScissorRect {
                x: 0x55,
                y: 0x66,
                width: 0x77,
                height: 0x88
            }
        ],
        "both rects, in the guest's order, at the record's own count width"
    );

    // Two viewports, four-byte count.
    let total = OP_HEADER_LEN + VIEWPORTS_COUNT_LEN + 2 * 48;
    let mut v = hdr(wire::OPCODE_SET_VIEWPORTS, total);
    st32(&mut v[OP_HEADER_LEN..], 2);
    let e0 = OP_HEADER_LEN + VIEWPORTS_COUNT_LEN;
    for i in 0..6 {
        st64(&mut v[e0 + i * 8..], (1.0f64 + i as f64).to_bits());
    }
    for i in 0..6 {
        st64(&mut v[e0 + 48 + i * 8..], (100.0f64 + i as f64).to_bits());
    }
    let c = decode(&v).expect("two viewports");
    assert_eq!(c.kind, Kind::SetViewport);
    assert_eq!(c.count, 2);
    assert_eq!(
        c.viewports,
        vec![
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
        ],
        "the second viewport is the guest's second, not a copy of the first"
    );

    // A count of zero names no rect, and a record that cannot hold the
    // count it claims is refused rather than read short.
    let mut v = hdr(
        wire::OPCODE_SET_SCISSOR_RECTS,
        OP_HEADER_LEN + SCISSOR_RECTS_COUNT_LEN,
    );
    st64(&mut v[OP_HEADER_LEN..], 0);
    assert_eq!(decode(&v).unwrap_err(), DecodeStatus::ErrBadLength);
    let mut v = hdr(
        wire::OPCODE_SET_VIEWPORTS,
        OP_HEADER_LEN + VIEWPORTS_COUNT_LEN + 48,
    );
    st32(&mut v[OP_HEADER_LEN..], 2);
    assert_eq!(decode(&v).unwrap_err(), DecodeStatus::ErrShort);

    // The singular opcodes keep reading from offset zero and report one.
    let mut v = hdr(
        wire::OPCODE_SET_SCISSOR,
        OP_HEADER_LEN + SCISSOR_PAYLOAD_LEN,
    );
    st64(&mut v[OP_HEADER_LEN..], 0x99);
    let c = decode(&v).expect("singular scissor");
    assert_eq!(c.scissors.len(), 1);
    assert_eq!(c.scissors[0].x, 0x99);
    assert_eq!(c.count, 1);
}

/// A sampler bind carrying LOD clamps is a bind, at a wider entry stride.
///
/// `0x80` and `0x71` are not longer forms of `0x7f` and `0x70` — they are
/// separate opcodes, so this decoder knowing only the plain pair did not
/// lose the clamps, it lost the whole bind and left the slot empty. The
/// entry is twelve bytes here against four there, which is the assertion
/// that matters: reading a LOD record at the plain stride would take a
/// clamp for the next slot's sampler.
#[test]
fn a_sampler_bind_with_lod_clamps_is_still_a_sampler_bind() {
    use crate::contract::endian::st32;
    use reims_vgpu_wire::ops::render as wire;

    assert_eq!(
        wire::OPCODE_SET_VERTEX_SAMPLER_LOD,
        wire::OPCODE_SET_VERTEX_SAMPLER_LOD
    );
    assert_eq!(
        wire::OPCODE_SET_FRAGMENT_SAMPLER_LOD,
        wire::OPCODE_SET_FRAGMENT_SAMPLER_LOD
    );

    for (op, stage) in [
        (wire::OPCODE_SET_VERTEX_SAMPLER_LOD, Stage::Vertex),
        (wire::OPCODE_SET_FRAGMENT_SAMPLER_LOD, Stage::Fragment),
    ] {
        const COUNT: u32 = 2;
        let total = OP_HEADER_LEN + BIND_ENTRIES + (COUNT as usize) * SAMPLER_LOD_BIND_ENTRY_SIZE;
        let mut v = hdr(op, total);
        st32(&mut v[OP_HEADER_LEN + BIND_FIRST..], 3);
        st32(&mut v[OP_HEADER_LEN + BIND_COUNT..], COUNT);
        for i in 0..COUNT as usize {
            let e = OP_HEADER_LEN + BIND_ENTRIES + i * SAMPLER_LOD_BIND_ENTRY_SIZE;
            st32(&mut v[e..], 0x6363 + i as u32);
            // Clamps this decoder does not lift. They are here so a decoder
            // reading at the plain four-byte stride would pick one up as a
            // ref and fail the assertion below.
            st32(&mut v[e + 4..], 0x3e80_0000); // 0.25
            st32(&mut v[e + 8..], 0x3f40_0000); // 0.75
        }

        let c = decode(&v).unwrap_or_else(|e| panic!("op {op:#x}: {e:?}"));
        assert_eq!(c.kind, Kind::SetSampler, "op {op:#x}");
        assert_eq!(c.stage, stage, "op {op:#x}");
        assert!(c.has_sampler_lod, "op {op:#x}");
        assert_eq!(c.first, 3, "op {op:#x}");
        assert_eq!(
            c.ref_binds,
            vec![0x6363, 0x6364],
            "op {op:#x} read the entries at the wrong stride"
        );
        assert_eq!(c.sampler_ref, 0x6363, "op {op:#x}");
    }

    // The plain forms keep the four-byte stride and say they carry no
    // clamps, so the flag is the record's and not the family's.
    let total = OP_HEADER_LEN + BIND_ENTRIES + REF_BIND_ENTRY_SIZE;
    let mut v = hdr(wire::OPCODE_SET_VERTEX_SAMPLER, total);
    st32(&mut v[OP_HEADER_LEN + BIND_COUNT..], 1);
    st32(&mut v[OP_HEADER_LEN + BIND_ENTRIES..], 0x6363);
    let c = decode(&v).expect("plain sampler bind");
    assert!(!c.has_sampler_lod);
    assert_eq!(c.ref_binds, vec![0x6363]);
}

/// The accepted-opcode window ends exactly where Apple's render manifest
/// does, computed rather than transcribed.
///
/// The accepted window ends at the highest opcode in the wire render
/// manifest (`OPCODE_SET_VERTEX_BUFFER_OFFSET_STRIDE`). A capture that
/// adds a higher opcode fails here and names itself, rather than leaving
/// a stale bound that refuses real records as "Apple does not write".
#[test]
fn the_accepted_window_ends_where_apples_render_manifest_does() {
    let highest = reims_vgpu_wire::manifest::MANIFEST
        .iter()
        .filter(|e| e.class == "PGSerializerRenderCommandEncoder")
        .flat_map(|e| e.opcodes.iter().copied())
        .max()
        .expect("the render encoder has opcodes in the manifest");
    assert_eq!(
        wire::OPCODE_SET_VERTEX_BUFFER_OFFSET_STRIDE,
        highest,
        "a render opcode above the accepted window would be refused as one \
         Apple does not write"
    );
}

/// Every render opcode Apple's serializer emits has a constant here, and
/// this module names no opcode Apple does not emit.
///
/// [`wire::OPCODE_SET_VERTEX_BUFFER_OFFSET_STRIDE`] bounds the window and this fills it. The two catch
/// different failures, and the one this catches is the quieter: an opcode
/// *inside* the accepted window that no arm claims does not get refused, it
/// gets [`Kind::OtherAccepted`] — the catch-all whose `Ok` is not a decode,
/// and which is what hid the sampler-LOD binds `0x80`/`0x71` behind a
/// passing run. So a capture that adds `0x9b`..`0xa4` the way `TileShaders`
/// did lands below the window's end, breaks nothing, and loses every record
/// it names.
///
/// The roster below is references rather than numbers, so no entry can
/// carry a wrong *value*; what this test adds is that no entry can go
/// *missing*. The reverse direction matters as much and is the shape of the
/// `0x86`/`0x87` residency bug: an opcode named here and absent from the
/// manifest is a number no capture supports.
///
/// **`0x1a` used to be excluded from it, on a claim that was wrong.** This
/// doc said the render-pass descriptor was "the live descriptor this
/// device's own framing carries, not a record
/// `PGSerializerRenderCommandEncoder` writes, and the manifest agrees by
/// omitting it" — and the manifest omitted it only because no case had
/// driven `writeDescriptor`, which emits it. Six more opcodes arrived with
/// it, all of them pass properties behind a capability, and every one was
/// reaching the catch-all.
///
/// The general form is worth keeping: **"the manifest agrees" is not
/// evidence when the manifest's silence is what is being explained.** An
/// opcode absent from a capture-derived roster can be an opcode nobody
/// captured.
///
/// # Two decoded opcodes are deliberately outside this bijection
///
/// `OPCODE_USE_HEAPS_NO_STAGES` (`0x86`) and `OPCODE_USE_RESOURCES_NO_STAGES`
/// (`0x87`) are decoded by this module and named in no manifest row, which the
/// reverse direction above would ordinarily call a number no capture supports.
/// They are the exception the manifest cannot express: the selectors behind them
/// are declared on the encoder base class, and the manifest is built per class
/// from each class's own method list, so no `PGSerializerRenderCommandEncoder`
/// row can ever carry them however many captures run. See the inheritance caveat
/// in [`reims_vgpu_wire::manifest`]. Adding them to the roster would fail this
/// test for a reason that is about the instrument rather than about the device,
/// so they stay out of it and are covered by
/// `the_inherited_residency_opcodes_reach_the_residency_arm` instead.
#[test]
fn the_render_opcode_table_is_exactly_apples_render_manifest() {
    let device: &[(u32, &str)] = &[
        (wire::OPCODE_DRAW_WIDE, "wire::OPCODE_DRAW_WIDE"),
        (wire::OPCODE_DRAW, "wire::OPCODE_DRAW"),
        (
            wire::OPCODE_DRAW_INSTANCED_WIDE,
            "wire::OPCODE_DRAW_INSTANCED_WIDE",
        ),
        (wire::OPCODE_DRAW_INSTANCED, "wire::OPCODE_DRAW_INSTANCED"),
        (
            wire::OPCODE_DRAW_INSTANCED_BASE_WIDE,
            "wire::OPCODE_DRAW_INSTANCED_BASE_WIDE",
        ),
        (
            wire::OPCODE_DRAW_INSTANCED_BASE,
            "wire::OPCODE_DRAW_INSTANCED_BASE",
        ),
        (
            wire::OPCODE_DRAW_INDEXED_WIDE,
            "wire::OPCODE_DRAW_INDEXED_WIDE",
        ),
        (wire::OPCODE_DRAW_INDEXED, "wire::OPCODE_DRAW_INDEXED"),
        (
            wire::OPCODE_DRAW_INDEXED_INSTANCED_WIDE,
            "wire::OPCODE_DRAW_INDEXED_INSTANCED_WIDE",
        ),
        (
            wire::OPCODE_DRAW_INDEXED_INSTANCED,
            "wire::OPCODE_DRAW_INDEXED_INSTANCED",
        ),
        (
            wire::OPCODE_DRAW_INDEXED_INSTANCED_BASE_WIDE,
            "wire::OPCODE_DRAW_INDEXED_INSTANCED_BASE_WIDE",
        ),
        (
            wire::OPCODE_DRAW_INDEXED_INSTANCED_BASE,
            "wire::OPCODE_DRAW_INDEXED_INSTANCED_BASE",
        ),
        (
            wire::OPCODE_DRAW_PATCHES_WIDE,
            "wire::OPCODE_DRAW_PATCHES_WIDE",
        ),
        (wire::OPCODE_DRAW_PATCHES, "wire::OPCODE_DRAW_PATCHES"),
        (
            wire::OPCODE_DRAW_INDEXED_PATCHES,
            "wire::OPCODE_DRAW_INDEXED_PATCHES",
        ),
        (wire::OPCODE_DRAW_INDIRECT, "wire::OPCODE_DRAW_INDIRECT"),
        (
            wire::OPCODE_DRAW_INDEXED_INDIRECT,
            "wire::OPCODE_DRAW_INDEXED_INDIRECT",
        ),
        (
            wire::OPCODE_DRAW_PATCHES_INDIRECT,
            "wire::OPCODE_DRAW_PATCHES_INDIRECT",
        ),
        (
            wire::OPCODE_DRAW_INDEXED_PATCHES_INDIRECT,
            "wire::OPCODE_DRAW_INDEXED_PATCHES_INDIRECT",
        ),
        (
            wire::OPCODE_EXECUTE_COMMANDS_INDIRECT,
            "wire::OPCODE_EXECUTE_COMMANDS_INDIRECT",
        ),
        (
            wire::OPCODE_EXECUTE_COMMANDS_RANGE,
            "wire::OPCODE_EXECUTE_COMMANDS_RANGE",
        ),
        (
            wire::OPCODE_MEMORY_BARRIER_RESOURCES,
            "wire::OPCODE_MEMORY_BARRIER_RESOURCES",
        ),
        (
            wire::OPCODE_MEMORY_BARRIER_SCOPE,
            "wire::OPCODE_MEMORY_BARRIER_SCOPE",
        ),
        (wire::OPCODE_UPDATE_FENCE, "wire::OPCODE_UPDATE_FENCE"),
        (wire::OPCODE_WAIT_FOR_FENCE, "wire::OPCODE_WAIT_FOR_FENCE"),
        (wire::OPCODE_USE_HEAP, "wire::OPCODE_USE_HEAP"),
        (wire::OPCODE_SET_BLEND_COLOR, "wire::OPCODE_SET_BLEND_COLOR"),
        (
            wire::OPCODE_SET_COLOR_STORE_ACTION,
            "wire::OPCODE_SET_COLOR_STORE_ACTION",
        ),
        (
            wire::OPCODE_SET_COLOR_STORE_ACTION_OPTIONS,
            "wire::OPCODE_SET_COLOR_STORE_ACTION_OPTIONS",
        ),
        (
            wire::OPCODE_SET_DEPTH_STENCIL_STATE,
            "wire::OPCODE_SET_DEPTH_STENCIL_STATE",
        ),
        (
            wire::OPCODE_SET_DEPTH_STORE_ACTION,
            "wire::OPCODE_SET_DEPTH_STORE_ACTION",
        ),
        (
            wire::OPCODE_SET_DEPTH_STORE_ACTION_OPTIONS,
            "wire::OPCODE_SET_DEPTH_STORE_ACTION_OPTIONS",
        ),
        (wire::OPCODE_SET_CULL_MODE, "wire::OPCODE_SET_CULL_MODE"),
        (wire::OPCODE_SET_DEPTH_BIAS, "wire::OPCODE_SET_DEPTH_BIAS"),
        (
            wire::OPCODE_SET_DEPTH_CLIP_MODE,
            "wire::OPCODE_SET_DEPTH_CLIP_MODE",
        ),
        (
            wire::OPCODE_SET_FRAGMENT_BUFFER,
            "wire::OPCODE_SET_FRAGMENT_BUFFER",
        ),
        (
            wire::OPCODE_SET_FRAGMENT_BUFFER_OFFSET,
            "wire::OPCODE_SET_FRAGMENT_BUFFER_OFFSET",
        ),
        (
            wire::OPCODE_SET_FRAGMENT_SAMPLER,
            "wire::OPCODE_SET_FRAGMENT_SAMPLER",
        ),
        (
            wire::OPCODE_SET_FRAGMENT_SAMPLER_LOD,
            "wire::OPCODE_SET_FRAGMENT_SAMPLER_LOD",
        ),
        (
            wire::OPCODE_SET_FRAGMENT_TEXTURE,
            "wire::OPCODE_SET_FRAGMENT_TEXTURE",
        ),
        (
            wire::OPCODE_SET_FRONT_FACING,
            "wire::OPCODE_SET_FRONT_FACING",
        ),
        (
            wire::OPCODE_SET_RENDER_PIPELINE_STATE,
            "wire::OPCODE_SET_RENDER_PIPELINE_STATE",
        ),
        (wire::OPCODE_SET_SCISSOR, "wire::OPCODE_SET_SCISSOR"),
        (
            wire::OPCODE_SET_SCISSOR_RECTS,
            "wire::OPCODE_SET_SCISSOR_RECTS",
        ),
        (
            wire::OPCODE_SET_STENCIL_REFERENCE,
            "wire::OPCODE_SET_STENCIL_REFERENCE",
        ),
        (
            wire::OPCODE_SET_STENCIL_STORE_ACTION,
            "wire::OPCODE_SET_STENCIL_STORE_ACTION",
        ),
        (
            wire::OPCODE_SET_STENCIL_STORE_ACTION_OPTIONS,
            "wire::OPCODE_SET_STENCIL_STORE_ACTION_OPTIONS",
        ),
        (
            wire::OPCODE_SET_TESSELLATION_FACTOR_BUFFER,
            "wire::OPCODE_SET_TESSELLATION_FACTOR_BUFFER",
        ),
        (
            wire::OPCODE_SET_TESSELLATION_FACTOR_SCALE,
            "wire::OPCODE_SET_TESSELLATION_FACTOR_SCALE",
        ),
        (
            wire::OPCODE_SET_TRIANGLE_FILL_MODE,
            "wire::OPCODE_SET_TRIANGLE_FILL_MODE",
        ),
        (
            wire::OPCODE_SET_VERTEX_BUFFER,
            "wire::OPCODE_SET_VERTEX_BUFFER",
        ),
        (
            wire::OPCODE_SET_VERTEX_BUFFER_OFFSET,
            "wire::OPCODE_SET_VERTEX_BUFFER_OFFSET",
        ),
        (
            wire::OPCODE_SET_VERTEX_SAMPLER,
            "wire::OPCODE_SET_VERTEX_SAMPLER",
        ),
        (
            wire::OPCODE_SET_VERTEX_SAMPLER_LOD,
            "wire::OPCODE_SET_VERTEX_SAMPLER_LOD",
        ),
        (
            wire::OPCODE_SET_VERTEX_TEXTURE,
            "wire::OPCODE_SET_VERTEX_TEXTURE",
        ),
        (wire::OPCODE_SET_VIEWPORT, "wire::OPCODE_SET_VIEWPORT"),
        (wire::OPCODE_SET_VIEWPORTS, "wire::OPCODE_SET_VIEWPORTS"),
        (
            wire::OPCODE_SET_VISIBILITY_RESULT_MODE,
            "wire::OPCODE_SET_VISIBILITY_RESULT_MODE",
        ),
        (wire::OPCODE_TEXTURE_BARRIER, "wire::OPCODE_TEXTURE_BARRIER"),
        (wire::OPCODE_SET_LINE_WIDTH, "wire::OPCODE_SET_LINE_WIDTH"),
        (wire::OPCODE_USE_RESOURCE, "wire::OPCODE_USE_RESOURCE"),
        (
            wire::OPCODE_SET_VERTEX_AMPLIFICATION_MODE,
            "wire::OPCODE_SET_VERTEX_AMPLIFICATION_MODE",
        ),
        (
            wire::OPCODE_SET_VERTEX_AMPLIFICATION_COUNT,
            "wire::OPCODE_SET_VERTEX_AMPLIFICATION_COUNT",
        ),
        (
            wire_tile::OPCODE_DISPATCH_THREADS_PER_TILE,
            "wire_tile::OPCODE_DISPATCH_THREADS_PER_TILE",
        ),
        (
            wire_tile::OPCODE_SET_TILE_THREADGROUP_MEMORY,
            "wire_tile::OPCODE_SET_TILE_THREADGROUP_MEMORY",
        ),
        (
            wire_tile::OPCODE_SET_TILE_BUFFER,
            "wire_tile::OPCODE_SET_TILE_BUFFER",
        ),
        (
            wire_tile::OPCODE_SET_TILE_BUFFER_OFFSET,
            "wire_tile::OPCODE_SET_TILE_BUFFER_OFFSET",
        ),
        (
            wire_tile::OPCODE_SET_TILE_SAMPLER,
            "wire_tile::OPCODE_SET_TILE_SAMPLER",
        ),
        (
            wire_tile::OPCODE_SET_TILE_SAMPLER_LOD,
            "wire_tile::OPCODE_SET_TILE_SAMPLER_LOD",
        ),
        (
            wire_tile::OPCODE_SET_TILE_TEXTURE,
            "wire_tile::OPCODE_SET_TILE_TEXTURE",
        ),
        (
            wire_tile::OPCODE_DISPATCH_THREADS_PER_TILE_IN_REGION,
            "wire_tile::OPCODE_DISPATCH_THREADS_PER_TILE_IN_REGION",
        ),
        (
            wire_tile::OPCODE_DISPATCH_THREADS_PER_TILE_IN_REGION_RT_INDEX,
            "wire_tile::OPCODE_DISPATCH_THREADS_PER_TILE_IN_REGION_RT_INDEX",
        ),
        (
            wire_tile::OPCODE_GET_TILE_DIMENSIONS,
            "wire_tile::OPCODE_GET_TILE_DIMENSIONS",
        ),
        (
            wire::OPCODE_SET_VERTEX_BUFFER_STRIDE,
            "wire::OPCODE_SET_VERTEX_BUFFER_STRIDE",
        ),
        (
            wire::OPCODE_SET_VERTEX_BUFFER_OFFSET_STRIDE,
            "wire::OPCODE_SET_VERTEX_BUFFER_OFFSET_STRIDE",
        ),
        (
            wire_pass::OPCODE_RENDER_PASS,
            "wire_pass::OPCODE_RENDER_PASS",
        ),
        (
            wire_pass::OPCODE_DEFAULT_RASTER_SAMPLE_COUNT,
            "wire_pass::OPCODE_DEFAULT_RASTER_SAMPLE_COUNT",
        ),
        (
            wire_pass::OPCODE_SAMPLE_POSITIONS,
            "wire_pass::OPCODE_SAMPLE_POSITIONS",
        ),
        (
            wire_pass::OPCODE_RASTERIZATION_RATE_MAP,
            "wire_pass::OPCODE_RASTERIZATION_RATE_MAP",
        ),
        (
            wire_pass::OPCODE_IMAGEBLOCK_SAMPLE_LENGTH,
            "wire_pass::OPCODE_IMAGEBLOCK_SAMPLE_LENGTH",
        ),
        (
            wire_pass::OPCODE_THREADGROUP_MEMORY_LENGTH,
            "wire_pass::OPCODE_THREADGROUP_MEMORY_LENGTH",
        ),
        (wire_pass::OPCODE_TILE_SIZE, "wire_pass::OPCODE_TILE_SIZE"),
    ];

    let mut apple: Vec<u32> = reims_vgpu_wire::manifest::MANIFEST
        .iter()
        .filter(|e| e.class == "PGSerializerRenderCommandEncoder")
        .flat_map(|e| e.opcodes.iter().copied())
        .collect();
    apple.sort_unstable();
    apple.dedup();

    for (op, name) in device {
        assert!(
            apple.contains(op),
            "{name} = {op:#x} is an opcode Apple's render manifest does not \
             list, so no capture supports it"
        );
    }
    for op in &apple {
        assert!(
            device.iter().any(|(d, _)| d == op),
            "Apple's serializer emits render opcode {op:#x} and this module \
             names no constant for it, so it reaches Kind::OtherAccepted and \
             every record carrying it is lost without a refusal"
        );
    }
    assert_eq!(
        device.len(),
        apple.len(),
        "the roster has a duplicate entry"
    );
}

/// Residency is four opcodes in two pairs, and the pairs are not interchangeable.
///
/// `0x1b`/`0x89` are the `stages:`-qualified forms the render encoder declares
/// itself; `0x86`/`0x87` are the unqualified ones it inherits. The four numbers
/// must stay distinct, because their heads are three different sizes and reading
/// one record with another's layout starts the refs in the wrong place —
/// `a_residency_record_is_bounded_by_its_own_count` is what catches that, and it
/// can only catch it while the opcodes disagree.
#[test]
fn the_residency_opcodes_are_the_ones_apples_serializer_writes() {
    use reims_vgpu_wire::ops::render as wire;
    assert_eq!(wire::OPCODE_USE_HEAP, 0x1b);
    assert_eq!(wire::OPCODE_USE_RESOURCE, 0x89);
    assert_eq!(wire::OPCODE_USE_HEAPS_NO_STAGES, 0x86);
    assert_eq!(wire::OPCODE_USE_RESOURCES_NO_STAGES, 0x87);
    let all = [
        wire::OPCODE_USE_HEAP,
        wire::OPCODE_USE_RESOURCE,
        wire::OPCODE_USE_HEAPS_NO_STAGES,
        wire::OPCODE_USE_RESOURCES_NO_STAGES,
    ];
    for (i, a) in all.iter().enumerate() {
        for b in &all[i + 1..] {
            assert_ne!(a, b, "two residency forms share one opcode");
        }
    }
}

/// The refs of a residency record start at a different offset on each form,
/// and the count-led extent is checked rather than assumed.
///
/// `useHeap:` puts its array at `+6`, which is not a multiple of four: a
/// record read with `useResource:`'s `+8` accepts two bytes fewer than it
/// needs, so the length check is what separates the two layouts.
#[test]
fn a_residency_record_is_bounded_by_its_own_count() {
    for (op, refs_at, kind) in [
        (
            wire::OPCODE_USE_RESOURCE,
            USE_RESOURCE_REFS,
            Kind::UseResource,
        ),
        (wire::OPCODE_USE_HEAP, USE_HEAP_REFS, Kind::UseHeap),
        (
            wire::OPCODE_USE_RESOURCES_NO_STAGES,
            USE_RESOURCES_NO_STAGES_REFS,
            Kind::UseResource,
        ),
        (
            wire::OPCODE_USE_HEAPS_NO_STAGES,
            USE_HEAPS_NO_STAGES_REFS,
            Kind::UseHeap,
        ),
    ] {
        let body = |count: u32, entries: usize| {
            let mut v = hdr(op, OP_HEADER_LEN + refs_at + entries * REF_BIND_ENTRY_SIZE);
            st32(&mut v[OP_HEADER_LEN + RESIDENCY_COUNT..], count);
            v
        };

        let c = decode(&body(2, 2)).unwrap_or_else(|e| panic!("op {op:#x}: {e:?}"));
        assert_eq!(c.kind, kind, "op {op:#x}");
        assert_eq!(c.count, 2, "op {op:#x}");

        // One entry short of what the count claims.
        assert_eq!(
            decode(&body(2, 1)).unwrap_err(),
            DecodeStatus::ErrShort,
            "op {op:#x} accepted a record one ref shorter than its count"
        );
        // Past the bind cap and still well-formed. This record names no
        // table slot, so a bind-table cap is not its bound; refusing here
        // would drop a residency declaration a guest may legitimately make.
        let big = 40u32;
        let c = decode(&body(big, big as usize))
            .unwrap_or_else(|e| panic!("op {op:#x} refused {big} resources: {e:?}"));
        assert_eq!(c.count, big, "op {op:#x}");
        // A count whose byte length overflows `usize` must not wrap into a
        // bound the record satisfies.
        assert_eq!(
            decode(&body(u32::MAX, 1)).unwrap_err(),
            DecodeStatus::ErrShort,
            "op {op:#x} accepted a count whose array cannot exist"
        );
        // Shorter than the form's own head. Written against `refs_at` rather
        // than a literal, because the four forms have three different head
        // sizes and the smallest of them is four bytes — a literal 4 here is a
        // *valid* empty `useHeaps:count:` record, not a truncated one.
        assert_eq!(
            decode(&hdr(op, OP_HEADER_LEN + refs_at - 1)).unwrap_err(),
            DecodeStatus::ErrShort,
            "op {op:#x} accepted a record shorter than its own head"
        );
    }
}

/// `0x86` and `0x87` reach the residency arm, not the catch-all.
///
/// This test used to assert the opposite, and the reason it gave was sound at
/// the time: the residency kinds had no executor arm, so reading these two as
/// residency removed them from `Kind::OtherAccepted` — the one net that would
/// have named them on the failure channel. Both halves of that have since
/// changed. `runtime::exec` answers `UseResource`/`UseHeap` with
/// `render_noop_residency_hint`, so the kind is a counter rather than a hole;
/// and the numbers are not a guess but the two forms of residency a render
/// encoder inherits from the encoder base class, which
/// [`reims_vgpu_wire::ops::render::UseHeapsNoStages`] records.
///
/// So the net they belong in is the counter, and leaving them in the catch-all
/// reported an implemented command as unimplemented while
/// `render_noop_residency_hint` counted half its family.
#[test]
fn the_inherited_residency_opcodes_reach_the_residency_arm() {
    for (op, kind) in [
        (wire::OPCODE_USE_HEAPS_NO_STAGES, Kind::UseHeap),
        (wire::OPCODE_USE_RESOURCES_NO_STAGES, Kind::UseResource),
    ] {
        let c = decode(&hdr(op, OP_HEADER_LEN + 16)).unwrap_or_else(|e| panic!("{op:#x}: {e:?}"));
        assert_eq!(c.kind, kind, "{op:#x} did not reach the residency arm");
    }
}

#[test]
fn property_fuzz() {
    for op in 0u32..0x120 {
        for len in [8usize, 12, 16, 24, 32, 48, 64] {
            let _ = decode(&hdr(op, len));
        }
    }
}
