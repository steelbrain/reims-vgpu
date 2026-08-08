use crate::model::PAGE_SHIFT_ARM64E;

use super::*;
use crate::contract::endian::st32;

/// A `newSampler` record carrying `state`, at the layout the wire crate's
/// own fixtures use.
fn sampler_record(state: u32) -> Vec<u8> {
    use reims_vgpu_wire::ops::sampler::NEW_SAMPLER_TOTAL_LEN;
    let mut b = vec![0u8; NEW_SAMPLER_TOTAL_LEN as usize];
    st32(&mut b[0..], TYPE7_OBJECT_SAMPLER);
    st32(&mut b[4..], NEW_SAMPLER_TOTAL_LEN);
    st32(&mut b[8..], 7);
    st32(&mut b[12..], state);
    b
}

/// The five anisotropy bits are read verbatim, and only a record outside
/// Metal's range is repaired.
///
/// `MTLSamplerDescriptor.maxAnisotropy` runs 1 through 16 — exactly what
/// five bits hold — and the wire crate's oracle baseline (a sampler with
/// nothing set) already reports 1, so Apple's serializer writes the API
/// default rather than leaving the field clear. A zero is therefore a
/// malformed record, and the decoder is the one place that says what to do
/// about it: four consumers used to floor it again for themselves and a
/// fifth did not.
#[test]
fn a_sampler_carries_its_own_anisotropy_and_only_zero_is_repaired() {
    // 0x84000000 is the oracle baseline: anisotropy 1, everything else at
    // its default. Bits 26..31 hold the value.
    let baseline = decode_sampler_descriptor(&sampler_record(0x8400_0000)).expect("baseline");
    assert_eq!(baseline.max_anisotropy, 1);

    for declared in [2u32, 13, 16] {
        let state = 0x8400_0000 & !(0x1f << 26) | (declared << 26);
        let sd = decode_sampler_descriptor(&sampler_record(state)).expect("declared");
        assert_eq!(
            sd.max_anisotropy, declared,
            "a declared anisotropy is carried, not clamped"
        );
    }

    // Zero is out of Metal's range; the floor is the whole repair and it is
    // not applied to anything else.
    let zeroed = 0x8400_0000 & !(0x1f << 26);
    let sd = decode_sampler_descriptor(&sampler_record(zeroed)).expect("zeroed anisotropy");
    assert_eq!(sd.max_anisotropy, 1);
}

/// A one-attribute vertex-input block whose single buffer layout carries
/// exactly the step tags asked for.
///
/// `step_tags` is what the serializer wrote: it emits a tag only for a
/// property the guest set, so an empty slice is the record an app that
/// touched neither produces.
fn vertex_block(step_tags: &[(u8, u32)]) -> Vec<u8> {
    vertex_block_on_buffer(step_tags, 0)
}

/// [`vertex_block`] with the buffer index both the layout entry and the
/// attribute name, so a test can drive one at or above [`MAX_VERTEX_LAYOUTS`].
fn vertex_block_on_buffer(step_tags: &[(u8, u32)], buffer_index: u32) -> Vec<u8> {
    // Root entry: 1 count byte + two 6-byte (tag, len, u32) fields.
    const ROOT_LEN: usize = 1 + 2 * 6;
    let layout_rel = ROOT_LEN;
    let layout_entry_rel = 8usize; // past the count and the one offset word
    let layout_fields = 2 + step_tags.len();
    let layout_len = layout_entry_rel + 1 + layout_fields * 6;
    let attr_rel = layout_rel + layout_len;
    let mut b = vec![0u8; attr_rel + 8 + 1 + 4 * 6];

    b[0] = 2;
    b[1] = VERTEX_DESC_TAG_ATTRIBUTES;
    b[2] = 4;
    st32(&mut b[3..], attr_rel as u32);
    b[7] = VERTEX_DESC_TAG_LAYOUTS;
    b[8] = 4;
    st32(&mut b[9..], layout_rel as u32);

    st32(&mut b[layout_rel..], 1);
    st32(&mut b[layout_rel + 4..], layout_entry_rel as u32);
    let le = layout_rel + layout_entry_rel;
    b[le] = layout_fields as u8;
    let mut p = le + 1;
    for (tag, value) in [
        (VERTEX_LAYOUT_TAG_BUFFER_INDEX, buffer_index),
        (VERTEX_LAYOUT_TAG_STRIDE, 16),
    ]
    .iter()
    .chain(step_tags.iter())
    {
        b[p] = *tag;
        b[p + 1] = 4;
        st32(&mut b[p + 2..], *value);
        p += 6;
    }

    st32(&mut b[attr_rel..], 1);
    st32(&mut b[attr_rel + 4..], 8);
    let ae = attr_rel + 8;
    b[ae] = 4;
    let mut p = ae + 1;
    for (tag, value) in [
        (VERTEX_ATTR_TAG_LOCATION, 0u32),
        (VERTEX_ATTR_TAG_FORMAT, 31), // MTLVertexFormatFloat4
        (VERTEX_ATTR_TAG_OFFSET, 0),
        (VERTEX_ATTR_TAG_BUFFER_INDEX, buffer_index),
    ] {
        b[p] = tag;
        b[p + 1] = 4;
        st32(&mut b[p + 2..], value);
        p += 6;
    }
    b
}

/// A tag in a vertex-descriptor entry that no reader names is reported, and the
/// entries a guest actually sends report as complete.
///
/// The three vertex walks read through `entry_tag_u32`, which re-walks the entry
/// once per tag the caller asks for. A reader written that way never forms a
/// list of what the entry held, so it cannot notice a tag it does not ask for —
/// the same structural blind spot the pipeline block one level up had, where a
/// driven boot then found four unread tags.
#[test]
fn a_vertex_entry_tag_with_no_reader_is_reported() {
    // A tag no other test in this process uses, so `first_sight` cannot have
    // latched its shape already.
    const UNKNOWN_TAG: u8 = 0x5b;

    let cap = crate::observe::FailCapture::start();
    let plain = vertex_block(&[]);
    parse_vertex_block(&plain, 0, plain.len()).expect("a plain block decodes");
    let clean = cap.lines();
    assert!(
        clean
            .iter()
            .filter(|l| l.contains("type7_pipeline_shape"))
            .all(|l| l.contains("unconsumed=0")),
        "the entries a guest sends are read whole; that is the reading this \
         instrument exists to make, and it must not be an absence: {clean:?}"
    );
    assert!(
        clean
            .iter()
            .any(|l| l.contains("kind=vertex_layout") || l.contains("kind=vertex_attr")),
        "and the walk must have run, or `unconsumed=0` above is vacuous: \
         {clean:?}"
    );

    let cap2 = crate::observe::FailCapture::resume();
    let odd = vertex_block(&[(UNKNOWN_TAG, 9)]);
    parse_vertex_block(&odd, 0, odd.len()).expect("an unread tag is reported, not refused");
    let lines = cap2.lines();
    let shape: Vec<&String> = lines
        .iter()
        .filter(|l| l.contains("type7_pipeline_shape") && l.contains("kind=vertex_layout"))
        .collect();
    assert_eq!(
        shape.len(),
        1,
        "one shape line for the layout entry: {lines:?}"
    );
    assert!(
        shape[0].contains("5b:4*") && shape[0].contains("unconsumed=1"),
        "the unread tag must be starred and counted: {}",
        shape[0]
    );
    assert!(
        lines
            .iter()
            .any(|l| l.contains("reason=pipeline_descriptor_field_dropped")
                && l.contains("kind=vertex_layout")
                && l.contains("tag=0x5b")),
        "and the loss must reach the fail channel naming the entry kind: \
         {lines:?}"
    );
}

/// A layout that declared `stepRate` 0 means 0, and one that declared none
/// means 1.
///
/// `MTLVertexBufferLayoutDescriptor.stepRate` defaults to 1, and
/// `MTLVertexStepFunctionConstant` is the one step function that requires 0
/// — it fetches the attribute once for the whole draw. Two of the four
/// consumers of this record used to clamp the declared zero up to 1, which
/// turned that attribute into a per-instance stream that advanced, while a
/// third refused the zero outright and a fourth passed it through. The rule
/// lives on the record now, so there is one answer to compare against.
#[test]
fn a_declared_zero_step_rate_is_not_a_missing_one() {
    let declared = vertex_block(&[
        (VERTEX_LAYOUT_TAG_STEP_FUNCTION, 0),
        (VERTEX_LAYOUT_TAG_STEP_RATE, 0),
    ]);
    let attrs = parse_vertex_block(&declared, 0, declared.len()).expect("declared step state");
    assert_eq!(attrs.len(), 1);
    assert_eq!(attrs[0].declared_step_rate, Some(0));
    assert_eq!(attrs[0].step_rate(), 0);
    assert_eq!(attrs[0].declared_step_function, Some(0));
    assert_eq!(attrs[0].step_function_ordinal(7), 0);

    let absent = vertex_block(&[]);
    let attrs = parse_vertex_block(&absent, 0, absent.len()).expect("absent step state");
    assert_eq!(attrs.len(), 1);
    assert_eq!(attrs[0].declared_step_rate, None);
    assert_eq!(attrs[0].step_rate(), 1);
    assert_eq!(attrs[0].declared_step_function, None);
    // The absent step function has no one default: a plain vertex
    // descriptor wants `PerVertex` and a post-tessellation one wants
    // `PerPatchControlPoint`, so the caller names it and this returns it.
    assert_eq!(attrs[0].step_function_ordinal(7), 7);

    // A declared rate above the default is neither of the two above.
    let three = vertex_block(&[(VERTEX_LAYOUT_TAG_STEP_RATE, 3)]);
    let attrs = parse_vertex_block(&three, 0, three.len()).expect("declared rate");
    assert_eq!(attrs[0].step_rate(), 3);
}

/// A layout naming a buffer index past `MAX_VERTEX_LAYOUTS` refuses the whole
/// descriptor, and says which index did it.
///
/// `MTLVertexDescriptor.layouts` has no such subscript, so there is nothing to
/// build and every choice but refusing is a guess. The decoder used to keep the
/// block with that layout's stride dropped, which left the attributes naming it
/// at stride 0 — a *valid* pipeline drawing every vertex from element zero, and
/// indistinguishable downstream from a guest that asked for stride 0. The
/// decline line is what tells the two apart.
#[test]
fn a_layout_buffer_index_past_the_bound_refuses_the_descriptor() {
    let cap = crate::observe::FailCapture::start();

    let past = vertex_block_on_buffer(&[], MAX_VERTEX_LAYOUTS as u32);
    assert_eq!(
        parse_vertex_block(&past, 0, past.len()),
        Err(DecodeStatus::ErrUnsupported("res_vertex_layout_buffer_oob")),
        "an unbuildable layout refuses rather than dropping its stride"
    );
    assert_eq!(
        cap.one("res_vertex_block"),
        "res_vertex_block reason=vertex_descriptor_truncated \
         what=layout_buffer_index declared=31 max=31",
        "and names the index it refused on"
    );

    // The in-range control: the same block one index lower keeps its stride,
    // and says nothing.
    let quiet = crate::observe::FailCapture::start();
    let last = vertex_block_on_buffer(&[], MAX_VERTEX_LAYOUTS as u32 - 1);
    let attrs = parse_vertex_block(&last, 0, last.len()).expect("decodes at the bound");
    assert_eq!(attrs[0].stride, 16);
    assert!(
        quiet
            .lines()
            .iter()
            .all(|l| !l.contains("vertex_descriptor_truncated")),
        "a descriptor inside the bound is quiet: {:?}",
        quiet.lines()
    );
}

/// A descriptor naming more attributes than `MAX_VERTEX_ATTRS` refuses, and
/// says how many it was asked for.
///
/// `MTLVertexDescriptor.attributes` has 31 slots, so a count above that is a
/// descriptor that cannot be built. Keeping the first 31 left the shader
/// declaring stage inputs it would never receive, with the draw looking exactly
/// like one from a guest that declared fewer — so the surplus is refused, and
/// the line carries both numbers because a report naming only the bound cannot
/// say how much was asked for.
#[test]
fn an_attribute_count_past_the_bound_refuses_and_names_both_numbers() {
    let cap = crate::observe::FailCapture::start();

    // Overwrite the count word of a well-formed one-attribute block. The count
    // is read straight off the wire, so no surplus entries need to exist for the
    // decoder to be told about them — which is also the case that must not be
    // allowed to decode as "31 attributes, fine".
    let mut over = vertex_block_on_buffer(&[], 0);
    let attr_rel = u32::from_le_bytes(over[3..7].try_into().unwrap()) as usize;
    let declared = MAX_VERTEX_ATTRS as u32 + 1;
    st32(&mut over[attr_rel..], declared);
    assert_eq!(
        parse_vertex_block(&over, 0, over.len()),
        Err(DecodeStatus::ErrUnsupported("res_vertex_attr_count_over")),
        "a count past the array refuses rather than keeping the first 31"
    );
    assert_eq!(
        cap.one("res_vertex_block"),
        format!(
            "res_vertex_block reason=vertex_descriptor_truncated \
             what=attribute_count declared={declared} max={MAX_VERTEX_ATTRS}"
        ),
        "and both numbers reach the line"
    );

    // At the bound exactly, the same block decodes — the entries the count
    // promises past the first are absent, so this also pins that the refusal is
    // on the count and the short block is a separate (structural) error.
    let mut at = vertex_block_on_buffer(&[], 0);
    st32(&mut at[attr_rel..], MAX_VERTEX_ATTRS as u32);
    assert!(
        !matches!(
            parse_vertex_block(&at, 0, at.len()),
            Err(DecodeStatus::ErrUnsupported("res_vertex_attr_count_over"))
        ),
        "the bound itself is not over it"
    );
}

/// A well-formed heap-texture record, with the bytes the serializer does
/// not write left as the caller asks.
fn heap_texture_record(use_offset_byte: u8, ring: u8, offset: u64) -> Vec<u8> {
    let mut b = vec![ring; HEAP_TEXTURE_LEN];
    st32(&mut b[TEXTURE_VIEW_DESC_OPCODE..], HEAP_TEXTURE_OPCODE);
    st32(&mut b[4..], HEAP_TEXTURE_LEN as u32);
    st32(&mut b[8..], 48);
    st32(&mut b[HEAP_TEXTURE_HEAP_REF..], 6565);
    b[HEAP_TEXTURE_USE_OFFSET] = use_offset_byte;
    b[HEAP_TEXTURE_OFFSET..HEAP_TEXTURE_LEN].copy_from_slice(&offset.to_le_bytes());
    b
}

/// `useOffset` is one bit, and the rest of its slot is the guest's ring.
///
/// The bug this pins: the read used to be a `ld32` of the four bytes at
/// [`HEAP_TEXTURE_USE_OFFSET`] followed by a refusal of anything above 1.
/// `reims_vgpu_wire::ops::heap_texture` measures, against Apple's own bytes
/// under two arena fills, that the serializer writes bit 0 of the first
/// byte and nothing else in that slot — so on a real wire the other 31 bits
/// are whatever the ring last held, and the refusal fired on content the
/// guest never wrote. A dropped texture bind is the most severe loss class
/// in the device, and this one was invisible because a host capture arena
/// is zero-filled there.
#[test]
fn heap_texture_use_offset_ignores_the_ring_bytes_around_it() {
    for ring in [0x00u8, 0xaa, 0xff, 0x5a] {
        for (byte, expect) in [(0x00u8, false), (0x01, true), (0xfe, false), (0xff, true)] {
            let bytes = heap_texture_record(byte, ring, 0x0123_4ab0);
            let record = decode_heap_texture(&bytes)
                .unwrap_or_else(|e| panic!("ring {ring:#04x} byte {byte:#04x}: {e:?}"));
            assert_eq!(
                record.use_offset, expect,
                "ring {ring:#04x} byte {byte:#04x}: use_offset"
            );
            assert_eq!(
                record.offset, 0x0123_4ab0,
                "ring {ring:#04x} byte {byte:#04x}: offset"
            );
            assert_eq!(record.heap_ref, 6565);
            assert_eq!(record.descriptor.len(), 32);
        }
    }
}

/// The 40-byte descriptor body, laid out at the wire crate's own offsets.
///
/// Distinctive values throughout, and `usage` deliberately carries a bit
/// above its low byte: the narrow body packs usage into eight bits, so a
/// decoder that kept the narrow width would read `0x05` here.
fn wide_descriptor_body() -> Vec<u8> {
    use reims_vgpu_wire::ops::texture::WideTextureDescriptorBody as W;
    let mut d = vec![0u8; heap_query::WIDE_TEXTURE_BODY_LEN];
    d[offset_of!(W, type_and_flags)] = 0x42; // 2D, allowGPUOptimizedContents
    st16(&mut d[offset_of!(W, pixel_format)..], 80); // BGRA8Unorm
    st32(&mut d[offset_of!(W, usage)..], 0x0001_0005);
    st32(&mut d[offset_of!(W, width)..], 0x1111);
    st32(&mut d[offset_of!(W, height)..], 0x2222);
    st32(&mut d[offset_of!(W, depth)..], 1);
    st16(&mut d[offset_of!(W, mipmap_level_count)..], 3);
    st16(&mut d[offset_of!(W, sample_count)..], 1);
    st16(&mut d[offset_of!(W, array_length)..], 7);
    st16(&mut d[offset_of!(W, resource_options)..], 0x0020);
    d[offset_of!(W, swizzle_red)] = 5;
    d[offset_of!(W, swizzle_green)] = 0;
    d[offset_of!(W, swizzle_blue)] = 1;
    d[offset_of!(W, swizzle_alpha)] = 2;
    d
}

/// A well-formed wide heap-texture record.
fn heap_texture_wide_record(use_offset_byte: u8, ring: u8, offset: u64) -> Vec<u8> {
    let mut b = vec![ring; HEAP_TEXTURE_WIDE_LEN];
    st32(&mut b[TEXTURE_VIEW_DESC_OPCODE..], HEAP_TEXTURE_WIDE_OPCODE);
    st32(
        &mut b[TEXTURE_VIEW_DESC_LEN..],
        HEAP_TEXTURE_WIDE_LEN as u32,
    );
    st32(&mut b[TEXTURE_VIEW_DESC_TEXTURE_REF..], 48);
    st32(&mut b[HEAP_TEXTURE_WIDE_HEAP_REF..], 6565);
    b[HEAP_TEXTURE_WIDE_DESCRIPTOR..HEAP_TEXTURE_WIDE_USE_OFFSET]
        .copy_from_slice(&wide_descriptor_body());
    b[HEAP_TEXTURE_WIDE_USE_OFFSET] = use_offset_byte;
    b[HEAP_TEXTURE_WIDE_OFFSET..HEAP_TEXTURE_WIDE_LEN].copy_from_slice(&offset.to_le_bytes());
    b
}

/// The `TextureDescriptor2` heap record decodes at its own offsets.
///
/// Every field after the heap ref moves by the eight bytes the wide
/// descriptor adds, so decoding this at the narrow offsets would put
/// `useOffset` inside the descriptor and read the heap offset from the
/// swizzle. It is the *opcode* that says which, never the length.
#[test]
fn a_wide_heap_texture_record_decodes_at_its_own_offsets() {
    for ring in [0x00u8, 0xaa, 0xff] {
        let bytes = heap_texture_wide_record(0x01, ring, 0x0077_7000);
        let record = decode_heap_texture(&bytes).expect("wide heap texture");
        assert!(
            record.wide,
            "ring {ring:#04x}: record reports its body width"
        );
        assert_eq!(record.heap_ref, 6565);
        assert!(record.use_offset);
        assert_eq!(record.offset, 0x0077_7000);
        assert_eq!(record.descriptor.len(), heap_query::WIDE_TEXTURE_BODY_LEN);

        let desc = heap_query::decode_wide_serialized_texture_descriptor(record.descriptor)
            .expect("wide body");
        assert_eq!(desc.width, 0x1111);
        assert_eq!(desc.height, 0x2222);
        assert_eq!(desc.array_length, 7);
        assert_eq!(desc.pixel_format, 80);
        assert_eq!(desc.texture_type, 2);
        assert!(desc.allow_gpu_optimized_contents);
        // Thirty-two bits, not eight. The narrow body's `usage` is a byte
        // of the packed word; this one is a field of its own, and holding
        // it at the narrow width would silently drop bit 16.
        assert_eq!(desc.usage, 0x0001_0005);
        assert_eq!(desc.swizzle, Some([5, 0, 1, 2]));
    }
}

/// Neither heap record may be decoded at the other's length.
///
/// This is the invariant that makes the pair safe: the wide form is a
/// different opcode rather than a longer record, so a decoder that picked
/// its layout from the length would read one as the other the moment the
/// two ever agreed on a size.
#[test]
fn a_heap_texture_record_is_refused_at_the_other_forms_length() {
    let wide = heap_texture_wide_record(0x01, 0x00, 0);
    assert!(matches!(
        decode_heap_texture(&wide[..HEAP_TEXTURE_LEN]),
        Err(DecodeStatus::ErrShort("res_heap_texture_len"))
    ));

    let mut narrow_at_wide_len = wide.clone();
    st32(
        &mut narrow_at_wide_len[TEXTURE_VIEW_DESC_OPCODE..],
        HEAP_TEXTURE_OPCODE,
    );
    assert!(matches!(
        decode_heap_texture(&narrow_at_wide_len),
        Err(DecodeStatus::ErrShort("res_heap_texture_len"))
    ));
}

/// The narrow body has no swizzle field, and absent is not the identity.
///
/// A reader that turned `None` into `[R, G, B, A]` would be inventing a
/// contract for a record that never states one; a reader that turned it
/// into `[0, 0, 0, 0]` would swizzle every channel to zero.
#[test]
fn the_narrow_descriptor_body_carries_no_swizzle() {
    let narrow = vec![0u8; heap_query::TEXTURE_BODY_LEN];
    let desc = heap_query::decode_serialized_texture_descriptor(&narrow).expect("narrow body");
    assert_eq!(desc.swizzle, None);
}

/// A wide buffer-backed texture keeps its prefix and widens only its body.
#[test]
fn a_wide_buffer_texture_record_decodes_its_wide_descriptor() {
    let mut b = vec![0u8; BUF_TEX_WIDE_LEN];
    st32(
        &mut b[TEXTURE_VIEW_DESC_OPCODE..],
        TEXTURE_VIEW_OPCODE_BUFFER_TEXTURE_WIDE,
    );
    st32(&mut b[TEXTURE_VIEW_DESC_LEN..], BUF_TEX_WIDE_LEN as u32);
    st32(&mut b[TEXTURE_VIEW_DESC_TEXTURE_REF..], 99);
    st32(&mut b[BUF_TEX_DESC_BUFFER_REF..], 5151);
    b[BUF_TEX_DESC_OFFSET..BUF_TEX_DESC_OFFSET + 8].copy_from_slice(&0x2200u64.to_le_bytes());
    b[BUF_TEX_DESC_BYTES_PER_ROW..BUF_TEX_DESC_BYTES_PER_ROW + 8]
        .copy_from_slice(&0x4400u64.to_le_bytes());
    b[BUF_TEX_WIDE_DESC_BODY..].copy_from_slice(&wide_descriptor_body());

    let d = decode_buffer_texture_descriptor(&b).expect("wide buffer texture");
    assert_eq!(d.new_texture_ref, 99);
    assert_eq!(d.buffer_ref, 5151);
    assert_eq!(d.offset, 0x2200);
    assert_eq!(d.bytes_per_row, 0x4400);
    assert_eq!(d.desc.width, 0x1111);
    assert_eq!(d.desc.usage, 0x0001_0005);
    assert_eq!(d.desc.swizzle, Some([5, 0, 1, 2]));

    // The narrow opcode at this length is not this record. Before the wide
    // form existed this decoder took any length at or above the narrow one,
    // so these bytes decoded as a narrow record with its descriptor read
    // eight bytes short of where it lives.
    let mut mislabelled = b.clone();
    st32(
        &mut mislabelled[TEXTURE_VIEW_DESC_OPCODE..],
        TEXTURE_VIEW_OPCODE_BUFFER_TEXTURE,
    );
    assert!(matches!(
        decode_buffer_texture_descriptor(&mislabelled),
        Err(DecodeStatus::ErrShort("res_buffer_texture_short"))
    ));

    // And a record whose declared length disagrees with its own opcode is
    // refused by the other name, whichever way the disagreement runs.
    let mut wrong_declared = b.clone();
    st32(
        &mut wrong_declared[TEXTURE_VIEW_DESC_LEN..],
        BUF_TEX_MIN_LEN as u32,
    );
    assert!(matches!(
        decode_buffer_texture_descriptor(&wrong_declared),
        Err(DecodeStatus::ErrShort("res_buffer_texture_declared_len"))
    ));
}

#[test]
fn a_heap_texture_record_of_the_wrong_length_or_opcode_is_refused_by_name() {
    let good = heap_texture_record(0x01, 0x00, 0);
    assert!(matches!(
        decode_heap_texture(&good[..HEAP_TEXTURE_LEN - 1]),
        Err(DecodeStatus::ErrShort("res_heap_texture_len"))
    ));

    let mut wrong = good.clone();
    st32(
        &mut wrong[TEXTURE_VIEW_DESC_OPCODE..],
        TEXTURE_VIEW_OPCODE_RANGED,
    );
    assert!(matches!(
        decode_heap_texture(&wrong),
        Err(DecodeStatus::ErrUnsupported("res_heap_texture_opcode"))
    ));
}

/// The embedded descriptor is the one the shared decoder reads.
///
/// Two offsets have to agree for this record to work at all: the body
/// starts at [`HEAP_TEXTURE_DESCRIPTOR`] and ends where `useOffset` begins.
/// If either moves, this decodes a descriptor shifted by the difference,
/// which produces plausible-looking geometry rather than an error.
#[test]
fn the_embedded_descriptor_decodes_through_the_shared_reader() {
    let mut bytes = heap_texture_record(0x01, 0x00, 0);
    // packed: type 2, GPU-optimized contents, usage 5, format 80 — the
    // shape the serializer produced for the oracle's baseline.
    st32(&mut bytes[HEAP_TEXTURE_DESCRIPTOR..], 0x0050_05c2);
    st32(&mut bytes[HEAP_TEXTURE_DESCRIPTOR + 4..], 0x1111);
    st32(&mut bytes[HEAP_TEXTURE_DESCRIPTOR + 8..], 0x2222);
    st32(&mut bytes[HEAP_TEXTURE_DESCRIPTOR + 12..], 1);

    let record = decode_heap_texture(&bytes).expect("well formed");
    let descriptor =
        crate::runtime::heap_query::decode_serialized_texture_descriptor(record.descriptor)
            .expect("the shared decoder accepts the embedded body");
    assert_eq!(descriptor.texture_type, 2);
    assert_eq!(descriptor.usage, 5);
    assert_eq!(descriptor.pixel_format, 80);
    assert_eq!(descriptor.width, 0x1111);
    assert_eq!(descriptor.height, 0x2222);
    assert_eq!(descriptor.depth, 1);
}

/// Short reads on different descriptors name different checks.
///
/// This is the collapse the payload closes: **29 of the decoder's 40 sites
/// were `ErrShort`**, one name for twenty-nine different reads spanning a
/// 12-byte object-list entry, a type-7 TLV walk and a vertex-attribute
/// table offset. Asserting the class alone would pass on any of them.
#[test]
fn every_short_read_names_the_field_it_ran_out_on() {
    use crate::observe::Decline;
    let cases: &[(&str, &'static str)] = &[
        ("list entry", "res_list_entry_short"),
        ("buffer", "res_buffer_desc_short"),
        ("sampler", "res_sampler_short"),
        ("icb layout", "res_icb_layout_short"),
    ];
    let got = [
        decode_list_object_entry(&[0u8; 1]).unwrap_err(),
        decode_buffer_descriptor(&[0u8; 1]).unwrap_err(),
        decode_sampler_descriptor(&[0u8; 1]).unwrap_err(),
        decode_icb_command_layout(&[0u8; 1]).unwrap_err(),
    ];
    for ((what, want), e) in cases.iter().zip(got) {
        assert_eq!(e.slug(), *want, "{what} short read lost its name");
    }

    // The other three classes, so the whole vocabulary is exercised rather
    // than just the one that used to swallow everything.
    assert_eq!(
        decode_descriptor(0xfe, &[0u8; 64]).unwrap_err().slug(),
        "res_object_type_unknown"
    );
    let mut type7 = [0u8; 64];
    st32(&mut type7[0..], 0xdead_beef);
    assert_eq!(
        decode_type7_descriptor(&type7).unwrap_err().slug(),
        "res_type7_subtype_unknown"
    );
}

#[test]
fn icb_descriptor_from_serializer_fixture() {
    // PGSerializer emission: ConcurrentDispatch, maxKernel=4, maxCmd=8, options=0.
    let mut b = [0u8; ICB_DESC_LEN];
    st32(&mut b[0..], TYPE7_OBJECT_ICB);
    st32(&mut b[4..], ICB_DESC_LEN as u32);
    st32(&mut b[8..], MTL_INDIRECT_CMD_CONCURRENT_DISPATCH);
    // bind counts as u8: vertex, fragment, kernel, …
    b[ICB_DESC_MAX_VERTEX_BINDS] = 0;
    b[ICB_DESC_MAX_FRAGMENT_BINDS] = 0;
    b[ICB_DESC_MAX_KERNEL_BINDS] = 4;
    st16(&mut b[ICB_DESC_FLAGS..], 0); // no inherit
    let layout = compute_only_icb_layout(4);
    b[ICB_DESC_LAYOUT..ICB_DESC_LAYOUT + ICB_LAYOUT_LEN]
        .copy_from_slice(&encode_icb_command_layout(&layout));
    st32(&mut b[ICB_DESC_MAX_COMMAND_COUNT..], 8);
    st32(&mut b[ICB_DESC_OPTIONS..], 0);
    let icb = decode_icb_descriptor(&b).unwrap();
    assert_eq!(icb.command_types, MTL_INDIRECT_CMD_CONCURRENT_DISPATCH);
    assert_eq!(icb.max_kernel_buffer_bind_count, 4);
    assert_eq!(icb.max_command_count, 8);
    assert!(!icb.inherit_buffers());
    assert!(!icb.inherit_pipeline_state());
}

/// The two crates that read this flag word name the same bits.
///
/// `reims_vgpu_wire::ops::icb::flag` derived them from Apple's serializer,
/// one case per property, and this module restates them because the decoder
/// here is reached by object type rather than through a wire view. Two
/// declarations of one contract is exactly the drift this repository writes
/// ABI tests for, so they are compared rather than trusted.
#[test]
fn the_icb_flag_bits_agree_with_the_derivation_they_came_from() {
    use reims_vgpu_wire::ops::icb::flag;
    for (mine, theirs, name) in [
        (
            ICB_FLAG_INHERIT_PIPELINE_STATE,
            flag::INHERIT_PIPELINE_STATE,
            "inherit_pipeline_state",
        ),
        (
            ICB_FLAG_INHERIT_BUFFERS,
            flag::INHERIT_BUFFERS,
            "inherit_buffers",
        ),
        (
            ICB_FLAG_SUPPORT_RAY_TRACING,
            flag::SUPPORT_RAY_TRACING,
            "support_ray_tracing",
        ),
        (
            ICB_FLAG_SUPPORT_DYNAMIC_ATTRIBUTE_STRIDE,
            flag::SUPPORT_DYNAMIC_ATTRIBUTE_STRIDE,
            "support_dynamic_attribute_stride",
        ),
        (
            ICB_FLAG_INHERIT_DEPTH_STENCIL_STATE,
            flag::INHERIT_DEPTH_STENCIL_STATE,
            "inherit_depth_stencil_state",
        ),
        (
            ICB_FLAG_INHERIT_DEPTH_BIAS,
            flag::INHERIT_DEPTH_BIAS,
            "inherit_depth_bias",
        ),
        (
            ICB_FLAG_INHERIT_DEPTH_CLIP_MODE,
            flag::INHERIT_DEPTH_CLIP_MODE,
            "inherit_depth_clip_mode",
        ),
        (
            ICB_FLAG_INHERIT_CULL_MODE,
            flag::INHERIT_CULL_MODE,
            "inherit_cull_mode",
        ),
        (
            ICB_FLAG_INHERIT_FRONT_FACING_WINDING,
            flag::INHERIT_FRONT_FACING_WINDING,
            "inherit_front_facing_winding",
        ),
        (
            ICB_FLAG_INHERIT_TRIANGLE_FILL_MODE,
            flag::INHERIT_TRIANGLE_FILL_MODE,
            "inherit_triangle_fill_mode",
        ),
        (ICB_FLAG_UNIDENTIFIED, flag::UNIDENTIFIED, "unidentified"),
    ] {
        assert_eq!(mine, theirs, "{name} disagrees between the two crates");
    }
    // The wire side also names the bit the serializer never writes. This
    // decoder must not claim it in any group, because on a guest's ring it
    // is noise.
    assert_eq!(
        ICB_FLAG_UNIDENTIFIED & flag::NEVER_WRITTEN,
        0,
        "the unidentified group claims the bit the serializer never writes"
    );
    assert_eq!(ICB_FLAG_NEVER_WRITTEN, flag::NEVER_WRITTEN);
}

/// The decoded flag word holds no bit the serializer never wrote.
///
/// The word is stored raw now, and bit 15 is noise on a real wire, so a
/// descriptor read off a ring that last held `0xff` there would compare
/// unequal to the identical descriptor read off a zeroed one — and the host
/// ICB cache compares descriptors. Caught by the fixture instrument's
/// poison half; kept here as the unit-level gate.
#[test]
fn the_decoded_flag_word_holds_no_bit_the_serializer_never_wrote() {
    let mut seen = std::collections::BTreeSet::new();
    for ring in [0x00u8, 0x80, 0xff] {
        let mut b = [0u8; ICB_DESC_LEN];
        st32(&mut b[0..], TYPE7_OBJECT_ICB);
        st32(&mut b[4..], ICB_DESC_LEN as u32);
        st32(&mut b[8..], MTL_INDIRECT_CMD_DRAW);
        // Every written bit set, plus whatever the ring left in bit 15.
        st16(
            &mut b[ICB_DESC_FLAGS..],
            ICB_FLAGS_DEFAULT | ((ring as u16 & 0x80) << 8),
        );
        let layout = compute_only_icb_layout(0);
        b[ICB_DESC_LAYOUT..ICB_DESC_LAYOUT + ICB_LAYOUT_LEN]
            .copy_from_slice(&encode_icb_command_layout(&layout));
        st32(&mut b[ICB_DESC_MAX_COMMAND_COUNT..], 8);
        let icb = decode_icb_descriptor(&b).unwrap();
        assert_eq!(icb.flags & ICB_FLAG_NEVER_WRITTEN, 0, "ring {ring:#04x}");
        seen.insert(icb.flags);
    }
    assert_eq!(seen.len(), 1, "the flag word moved with the ring: {seen:?}");
}

/// Every flag the guest can ask for is either applied or counted as lost.
///
/// A descriptor left at its defaults must report nothing — that is what
/// makes each of these counters a healthy zero — and each of the eight this
/// device does not carry must name itself when the guest asks for it. A
/// single "some flag was dropped" count could not tell ray tracing from a
/// cull mode the guest did not want inherited.
#[test]
fn a_flag_this_device_drops_names_itself_and_a_default_descriptor_names_none() {
    const DEFAULT_FLAGS: u16 = ICB_FLAGS_DEFAULT;
    let at_default = IndirectCommandBufferDescriptor {
        flags: DEFAULT_FLAGS,
        ..Default::default()
    };
    assert!(
        at_default.unapplied_flags().is_empty(),
        "a descriptor at its defaults reports a loss: {:?}",
        at_default.unapplied_flags()
    );
    assert_eq!(at_default.unidentified_flags(), ICB_FLAG_UNIDENTIFIED);

    for (flags, want) in [
        (
            DEFAULT_FLAGS | ICB_FLAG_SUPPORT_RAY_TRACING,
            IcbUnappliedFlag::SupportRayTracing,
        ),
        (
            DEFAULT_FLAGS | ICB_FLAG_SUPPORT_DYNAMIC_ATTRIBUTE_STRIDE,
            IcbUnappliedFlag::SupportDynamicAttributeStride,
        ),
        (
            DEFAULT_FLAGS & !ICB_FLAG_INHERIT_DEPTH_STENCIL_STATE,
            IcbUnappliedFlag::InheritDepthStencilState,
        ),
        (
            DEFAULT_FLAGS & !ICB_FLAG_INHERIT_DEPTH_BIAS,
            IcbUnappliedFlag::InheritDepthBias,
        ),
        (
            DEFAULT_FLAGS & !ICB_FLAG_INHERIT_DEPTH_CLIP_MODE,
            IcbUnappliedFlag::InheritDepthClipMode,
        ),
        (
            DEFAULT_FLAGS & !ICB_FLAG_INHERIT_CULL_MODE,
            IcbUnappliedFlag::InheritCullMode,
        ),
        (
            DEFAULT_FLAGS & !ICB_FLAG_INHERIT_FRONT_FACING_WINDING,
            IcbUnappliedFlag::InheritFrontFacingWinding,
        ),
        (
            DEFAULT_FLAGS & !ICB_FLAG_INHERIT_TRIANGLE_FILL_MODE,
            IcbUnappliedFlag::InheritTriangleFillMode,
        ),
    ] {
        let desc = IndirectCommandBufferDescriptor {
            flags,
            ..Default::default()
        };
        assert_eq!(
            desc.unapplied_flags(),
            vec![want],
            "flags {flags:#06x} did not report exactly {want:?}"
        );
    }

    // The two this device *does* apply must never appear on the list,
    // whichever way they are set — otherwise a working path reports a loss.
    for flags in [
        DEFAULT_FLAGS | ICB_FLAG_INHERIT_BUFFERS | ICB_FLAG_INHERIT_PIPELINE_STATE,
        DEFAULT_FLAGS,
    ] {
        let desc = IndirectCommandBufferDescriptor {
            flags,
            ..Default::default()
        };
        assert!(desc.unapplied_flags().is_empty(), "flags {flags:#06x}");
    }

    // Every slug is distinct: eight losses that shared one name would be
    // the collapse this enum exists to prevent.
    let slugs: std::collections::BTreeSet<&str> = [
        IcbUnappliedFlag::SupportRayTracing,
        IcbUnappliedFlag::SupportDynamicAttributeStride,
        IcbUnappliedFlag::InheritDepthStencilState,
        IcbUnappliedFlag::InheritDepthBias,
        IcbUnappliedFlag::InheritDepthClipMode,
        IcbUnappliedFlag::InheritCullMode,
        IcbUnappliedFlag::InheritFrontFacingWinding,
        IcbUnappliedFlag::InheritTriangleFillMode,
    ]
    .iter()
    .map(|f| f.slug())
    .collect();
    assert_eq!(slugs.len(), 8, "two dropped flags share a slug");
}

/// `options` is sixteen bits, and the two bytes above it are the guest's
/// ring.
///
/// The serializer narrows the `Q` its selector declares and never touches
/// `+0x56`/`+0x57`. This decoder read a `u32` there, so a descriptor
/// allocated over a ring that last held anything non-zero produced
/// `MTLResourceOptions` with garbage in its top half — the same shape as
/// the `copyFromTexture:toBuffer:` `options` bug. Found by the oracle's
/// complementary-fill passes rather than by reading, which is why the two
/// fills are what this test drives.
#[test]
fn the_options_word_ignores_the_two_bytes_the_serializer_never_writes() {
    let mut decoded = Vec::new();
    for ring in [0x00u8, 0xaa, 0x55, 0xff] {
        let mut b = [0u8; ICB_DESC_LEN];
        st32(&mut b[0..], TYPE7_OBJECT_ICB);
        st32(&mut b[4..], ICB_DESC_LEN as u32);
        st32(&mut b[8..], MTL_INDIRECT_CMD_DRAW);
        b[ICB_DESC_MAX_VERTEX_BINDS] = 4;
        st16(&mut b[ICB_DESC_FLAGS..], 0);
        let layout = compute_only_icb_layout(0);
        b[ICB_DESC_LAYOUT..ICB_DESC_LAYOUT + ICB_LAYOUT_LEN]
            .copy_from_slice(&encode_icb_command_layout(&layout));
        st32(&mut b[ICB_DESC_MAX_COMMAND_COUNT..], 8);
        // MTLResourceStorageModePrivate, the value a real guest writes.
        st16(&mut b[ICB_DESC_OPTIONS..], 0x20);
        b[ICB_DESC_OPTIONS_UNWRITTEN] = ring;
        b[ICB_DESC_OPTIONS_UNWRITTEN + 1] = ring;
        decoded.push(decode_icb_descriptor(&b).unwrap().options);
    }
    assert!(
        decoded.iter().all(|&o| o == 0x20),
        "options moved with bytes the serializer never wrote: {decoded:?}"
    );
}

/// A dispatch-only `command_types` does not license discarding the
/// fragment bind count the descriptor states.
///
/// The decoder used to zero `max_fragment_buffer_bind_count` whenever the
/// command mask named a dispatch and no draw. That is an inference about
/// what the guest meant, overriding a byte the guest wrote at +0x0d, and it
/// was silent — so a descriptor built by a guest that reserves fragment
/// binds on a buffer it happens to fill with dispatches had the reservation
/// dropped with nothing recorded. Metal is handed this count directly
/// (`icb::materialize`), so the drop is guest-visible.
#[test]
fn a_dispatch_only_command_mask_keeps_the_stated_fragment_bind_count() {
    let mut b = [0u8; ICB_DESC_LEN];
    st32(&mut b[0..], TYPE7_OBJECT_ICB);
    st32(&mut b[4..], ICB_DESC_LEN as u32);
    // Dispatch bits only — no draw bit anywhere in the mask.
    st32(
        &mut b[8..],
        MTL_INDIRECT_CMD_CONCURRENT_DISPATCH | MTL_INDIRECT_CMD_CONCURRENT_DISPATCH_THREADS,
    );
    b[ICB_DESC_MAX_VERTEX_BINDS] = 0;
    b[ICB_DESC_MAX_FRAGMENT_BINDS] = 6;
    b[ICB_DESC_MAX_KERNEL_BINDS] = 4;
    st16(&mut b[ICB_DESC_FLAGS..], 0);
    let layout = compute_only_icb_layout(4);
    b[ICB_DESC_LAYOUT..ICB_DESC_LAYOUT + ICB_LAYOUT_LEN]
        .copy_from_slice(&encode_icb_command_layout(&layout));
    st32(&mut b[ICB_DESC_MAX_COMMAND_COUNT..], 8);
    st32(&mut b[ICB_DESC_OPTIONS..], 0);

    let icb = decode_icb_descriptor(&b).unwrap();
    assert_eq!(
        icb.max_fragment_buffer_bind_count, 6,
        "the wire byte at +0x0d is the answer, not the command mask"
    );
    assert_eq!(icb.max_kernel_buffer_bind_count, 4);
    assert_eq!(icb.layout.command_size, layout.command_size);
    assert_eq!(icb.layout.kernel_buffer_bind_offset, 0x64);
    match decode_type7_descriptor(&b).unwrap() {
        Descriptor::IndirectCommandBuffer(d) => assert_eq!(d.max_command_count, 8),
        _ => panic!("expected ICB"),
    }
    // inherit both (bit0 = pipeline, bit1 = buffers)
    st16(
        &mut b[ICB_DESC_FLAGS..],
        ICB_FLAG_INHERIT_BUFFERS | ICB_FLAG_INHERIT_PIPELINE_STATE,
    );
    let icb = decode_icb_descriptor(&b).unwrap();
    assert!(icb.inherit_buffers() && icb.inherit_pipeline_state());
}

/// Dedicated create-body max-count matrix: decode offsets +0x0f..+0x12 and
/// layout table sizing for object/mesh/objectTG/kernelTG.
#[test]
fn icb_create_body_max_count_matrix() {
    use crate::contract::endian::st16;

    // --- Decode: single-byte fields at RE offsets ---
    let mut b = [0u8; ICB_DESC_LEN];
    st32(&mut b[0..], TYPE7_OBJECT_ICB);
    st32(&mut b[4..], ICB_DESC_LEN as u32);
    st32(
        &mut b[8..],
        MTL_INDIRECT_CMD_DRAW_MESH_THREADGROUPS | MTL_INDIRECT_CMD_DRAW,
    );
    b[ICB_DESC_MAX_VERTEX_BINDS] = 2;
    b[ICB_DESC_MAX_FRAGMENT_BINDS] = 3;
    b[ICB_DESC_MAX_KERNEL_BINDS] = 0;
    b[ICB_DESC_MAX_OBJECT_BINDS] = 4; // +0x0f
    b[ICB_DESC_MAX_MESH_BINDS] = 5; // +0x10
    b[ICB_DESC_MAX_KERNEL_TG_BINDS] = 6; // +0x11
    b[ICB_DESC_MAX_OBJECT_TG_BINDS] = 7; // +0x12
    st16(&mut b[ICB_DESC_FLAGS..], 0);
    let layout = render_icb_layout_ex(
        2,
        3,
        4,
        5,
        7,
        MTL_INDIRECT_CMD_DRAW_MESH_THREADGROUPS | MTL_INDIRECT_CMD_DRAW,
    );
    b[ICB_DESC_LAYOUT..ICB_DESC_LAYOUT + ICB_LAYOUT_LEN]
        .copy_from_slice(&encode_icb_command_layout(&layout));
    st32(&mut b[ICB_DESC_MAX_COMMAND_COUNT..], 1);
    st32(&mut b[ICB_DESC_OPTIONS..], 0);

    let icb = decode_icb_descriptor(&b).unwrap();
    assert_eq!(icb.max_vertex_buffer_bind_count, 2);
    assert_eq!(icb.max_fragment_buffer_bind_count, 3);
    assert_eq!(icb.max_kernel_buffer_bind_count, 0);
    assert_eq!(icb.max_object_buffer_bind_count, 4);
    assert_eq!(icb.max_mesh_buffer_bind_count, 5);
    assert_eq!(icb.max_kernel_threadgroup_memory_bind_count, 6);
    assert_eq!(icb.max_object_threadgroup_memory_bind_count, 7);

    // --- Layout sizing: bind table order vertex → fragment → object → mesh ---
    // vertex @0x64, 2 × 0x14 = 0x28 → fragment @0x8c
    assert_eq!(layout.vertex_buffer_bind_offset, 0x64);
    assert_eq!(layout.fragment_buffer_bind_offset, 0x64 + 2 * 0x14);
    assert_eq!(
        layout.object_buffer_bind_offset,
        layout.fragment_buffer_bind_offset + 3 * 0x14
    );
    assert_eq!(
        layout.mesh_buffer_bind_offset,
        layout.object_buffer_bind_offset + 4 * 0x14
    );
    // After mesh: attribute stride max_vertex × 8
    let after_mesh = layout.mesh_buffer_bind_offset + 5 * 0x14;
    assert_eq!(layout.attribute_stride_offset, after_mesh);
    let after_stride = after_mesh + 2 * ICB_ATTRIBUTE_STRIDE_ENTRY_SIZE as u32;
    // Object TG table: max_object_tg × 8
    assert_eq!(layout.object_threadgroup_memory_length_offset, after_stride);
    assert_eq!(
        layout.threadgroup_memory_length_offset,
        after_stride + 7 * ICB_TG_MEMORY_STRIDE as u32
    );
    // Pure render: kernel TG slots empty (object TG ends at args)
    assert_eq!(
        layout.command_arguments_offset,
        layout.object_threadgroup_memory_length_offset + 7 * ICB_TG_MEMORY_STRIDE as u32
    );
    assert_eq!(
        layout.command_size,
        layout.command_arguments_offset + ICB_DRAW_MESH_ARGS_LEN
    );

    // --- Compute: kernelTG table size from max_kernel_tg ---
    let cl = compute_icb_layout(3, 2);
    assert_eq!(icb_layout_kernel_tg_slot_count(&cl), 2);
    assert_eq!(
        cl.threadgroup_memory_length_offset + 2 * ICB_TG_MEMORY_STRIDE as u32,
        cl.command_arguments_offset
    );

    // --- Zero counts: tables collapse (no object/mesh/objectTG) ---
    let zero = render_icb_layout_ex(1, 0, 0, 0, 0, MTL_INDIRECT_CMD_DRAW);
    assert_eq!(
        zero.object_buffer_bind_offset,
        zero.fragment_buffer_bind_offset
    );
    assert_eq!(zero.mesh_buffer_bind_offset, zero.object_buffer_bind_offset);
    assert_eq!(
        zero.object_threadgroup_memory_length_offset,
        zero.attribute_stride_offset + ICB_ATTRIBUTE_STRIDE_ENTRY_SIZE as u32
    );
    assert_eq!(
        zero.command_arguments_offset,
        zero.object_threadgroup_memory_length_offset
    );
}

/// A bit a guest sets in a packed stage-input word that this decoder has no
/// field for says so.
///
/// The layout entry's word names 10 of its 32 bits and the attribute entry's
/// names 16, so 22 and 16 respectively arrive with no reader. A field is
/// `(word >> shift) & mask` at its own site and no site is in a position to
/// notice that some bits are named by nothing — the same hole the tag
/// instruments above close for the TLV form, one level down and harder to read.
///
/// Its two headers are excluded because they tile their word exactly; that is
/// pinned by `const` assertion beside the masks, not here.
#[test]
fn a_packed_stage_input_bit_with_no_field_says_so() {
    use crate::contract::endian::st32;
    use crate::runtime::decode::resource::{
        COMPUTE_STAGE_INPUT_ATTR_BITS_FORMAT_MASK, COMPUTE_STAGE_INPUT_ATTR_BITS_FORMAT_SHIFT,
    };

    // The top bit of the attribute word, which is 16 above the highest bit any
    // field names. A value no other test in this process sets, so `first_sight`
    // cannot have latched it.
    const UNREAD_BIT: u32 = 1 << 31;
    // Every bit a field does name, so the line must report the unread one alone
    // rather than the whole word.
    let read: u32 =
        COMPUTE_STAGE_INPUT_ATTR_BITS_FORMAT_MASK << COMPUTE_STAGE_INPUT_ATTR_BITS_FORMAT_SHIFT;

    let mut b = [0u8; 4];
    st32(&mut b, read | UNREAD_BIT);

    let cap = crate::observe::FailCapture::start();
    super::note_unread_bits("compute_stage_input_attr", ld32(&b), read);
    let lines = cap.lines();
    assert!(
        lines.iter().any(|l| l.contains("packed_word_unread_bits")
            && l.contains("kind=compute_stage_input_attr")
            && l.contains("unread=0x80000000")),
        "the line must carry the unread bits alone, not the whole word: {lines:?}"
    );

    // A word inside the named bits is silent, which is what makes the line above
    // a reading rather than noise on every entry.
    let cap2 = crate::observe::FailCapture::resume();
    super::note_unread_bits("compute_stage_input_attr", read, read);
    assert!(
        cap2.lines().is_empty(),
        "a word whose every set bit has a field must stay quiet: {:?}",
        cap2.lines()
    );
}

#[test]
fn compute_pipeline_stage_input_fixture() {
    // Local MetalSerializer fixture: dynamic Float4 stage-input layout.
    // From reims_vgpu_resource_resolve_test make_compute_stage_input_pipeline.
    let fixture: [u8; 60] = [
        0x0b, 0x00, 0x00, 0x00, 0x3c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x2c, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x20, 0x00,
        0x40, 0x08, 0x08, 0x00, 0x18, 0x00, 0xa0, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x7c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ];
    let cp = decode_compute_pipeline_descriptor(&fixture).unwrap();
    // First TLV is empty (field_count=0 at +16); kernel not present in this fixture.
    assert_eq!(cp.kernel_func_ref, 0);
    let si = cp.stage_input.expect("stage-input block");
    assert_eq!(si.header0, 0x0840_0020);
    assert_eq!(si.header1, 0x0018_0008);
    assert_eq!(si.index_type, 0);
    assert_eq!(si.index_buffer_index, 0);
    assert_eq!(si.layouts.len(), 1);
    assert_eq!(si.layouts[0].raw_bits, 0xa0);
    assert_eq!(si.layouts[0].step_function, 5);
    assert_eq!(si.layouts[0].stride, u64::MAX);
    assert_eq!(si.attributes.len(), 1);
    assert_eq!(si.attributes[0].raw_bits, 0x7c00);
    // format bits 10..15 of 0x7c00 = 0x1f = 31 (Float4).
    assert_eq!(si.attributes[0].format, 31);
    assert_eq!(si.dropped_attributes, 0);
    assert_eq!(si.dropped_layouts, 0);
}

/// The widest stage-input the wire can state survives decode whole.
///
/// `header0` carries both counts in 5-bit fields, so 31 attributes and 31
/// layouts is the maximum a guest can ever declare — and the decoder's caps are
/// sized to exactly that, which is what makes `dropped_*` unreachable. At the
/// former caps of 16 this decoded 16 of each and set both drop counters to 15,
/// and the loader then discarded the whole stage-input for it.
#[test]
fn a_stage_input_at_the_wire_count_field_maximum_loses_nothing() {
    // The count field's whole range. Asserted against the mask rather than
    // spelled 31 so this drives what the wire can say, not what the caps admit —
    // the caps are pinned to the same mask at their declaration.
    let n = COMPUTE_STAGE_INPUT_HEADER0_COUNT_MASK;
    let ns = n as usize;

    // Same 24-byte type-7 prefix as the fixture above (tag, declared length,
    // then an empty first TLV record), so the stage-input block starts at 24.
    const BLOCK: usize = 24;
    let layout_section = BLOCK + COMPUTE_STAGE_INPUT_MIN_LEN;
    let attr_section = layout_section + ns * COMPUTE_STAGE_INPUT_LAYOUT_ENTRY_SIZE;
    let total = attr_section + ns * COMPUTE_STAGE_INPUT_ATTR_ENTRY_SIZE;
    let mut b = vec![0u8; total];
    st32(&mut b[0..], TYPE7_OBJECT_COMPUTE_PIPELINE);
    st32(&mut b[4..], total as u32);
    // word0, then header0: payload length (everything after word0) plus both
    // count fields at their maximum.
    st32(&mut b[BLOCK + COMPUTE_STAGE_INPUT_WORD0..], 1);
    st32(
        &mut b[BLOCK + COMPUTE_STAGE_INPUT_HEADER0..],
        (total - BLOCK - 4) as u32
            | (n << COMPUTE_STAGE_INPUT_HEADER0_ATTR_COUNT_SHIFT)
            | (n << COMPUTE_STAGE_INPUT_HEADER0_LAYOUT_COUNT_SHIFT),
    );
    // Both section offsets are relative to header0, not to word0.
    let base = BLOCK + COMPUTE_STAGE_INPUT_HEADER1_OFFSET_BASE;
    st32(
        &mut b[BLOCK + COMPUTE_STAGE_INPUT_HEADER1..],
        (layout_section - base) as u32
            | (((attr_section - base) as u32) << COMPUTE_STAGE_INPUT_HEADER1_ATTR_OFFSET_SHIFT),
    );
    for i in 0..n {
        // Buffer index i in the low 5 bits, so each layout is distinguishable.
        let e = layout_section + i as usize * COMPUTE_STAGE_INPUT_LAYOUT_ENTRY_SIZE;
        st32(&mut b[e..], i);
        st32(&mut b[e + COMPUTE_STAGE_INPUT_LAYOUT_STEP_RATE..], 1);
        let a = attr_section + i as usize * COMPUTE_STAGE_INPUT_ATTR_ENTRY_SIZE;
        st32(&mut b[a..], i);
        st32(&mut b[a + COMPUTE_STAGE_INPUT_ATTR_OFFSET..], i * 4);
    }

    let si = decode_compute_pipeline_descriptor(&b)
        .expect("type-7 decodes")
        .stage_input
        .expect("stage-input block");
    assert_eq!(si.dropped_attributes, 0, "no attribute may be dropped");
    assert_eq!(si.dropped_layouts, 0, "no layout may be dropped");
    assert_eq!(si.attributes.len(), ns);
    assert_eq!(si.layouts.len(), ns);
    for i in 0..ns {
        assert_eq!(si.layouts[i].buffer_index, i as u32);
        assert_eq!(si.attributes[i].location, i as u32);
        assert_eq!(si.attributes[i].offset, i as u32 * 4);
    }
}

#[test]
fn list_entry_and_buffer() {
    // Live list offset: ref * 12
    assert_eq!(list_object_entry_offset(3, 10), Some(36));

    let mut list = [0u8; 12];
    st32(&mut list[0..], 11u32 | (0x20u32 << 8));
    // desc_gva
    list[4] = 0x80;
    let le = decode_list_object_entry(&list).unwrap();
    assert_eq!(le.object_type, 11);
    assert_eq!(le.descriptor_length, 0x20);
    assert_eq!(le.descriptor_gva, 0x80);

    let mut buf = vec![0u8; LINEAR_DESC_MIN_LEN];
    // allocation_size = 256, handle = 0x1234
    buf[0] = 0;
    buf[1] = 1;
    buf[8] = 0x34;
    buf[9] = 0x12;
    let d = decode_buffer_descriptor(&buf).unwrap();
    assert_eq!(d.allocation_size, 256);
    assert_eq!(d.handle, 0x1234);
    assert_eq!(
        d.backing_gva_size(PAGE_SHIFT_ARM64E),
        Some(((0x1234u64) << RESOURCE_PAGE_SHIFT, 256))
    );
}

#[test]
fn iosurface_type11() {
    let mut b = [0u8; 0x20];
    b[0] = 2;
    b[0x16] = 0x50;
    b[0x18] = 64;
    b[0x1c] = 32;
    match decode_descriptor(11, &b).unwrap() {
        Descriptor::IOSurfaceTexture {
            mapping_id,
            width,
            height,
            pixel_format,
            ..
        } => {
            assert_eq!(mapping_id, 2);
            assert_eq!(width, 64);
            assert_eq!(height, 32);
            assert_eq!(pixel_format, 0x50);
        }
        _ => panic!("wrong kind"),
    }
}

#[test]
fn linear_texture_geometry() {
    use crate::contract::endian::{st16, st64};
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    let mut b = vec![0u8; TEXTURE_DESC_BASE_LEN];
    st64(&mut b[0..], 0x10000);
    st32(&mut b[8..], 0x10);
    st32(&mut b[TEXTURE_DESC_ROW_STRIDE..], 256);
    st32(&mut b[TEXTURE_DESC_WIDTH..], 64);
    st32(&mut b[TEXTURE_DESC_HEIGHT..], 32);
    st16(&mut b[TEXTURE_DESC_PIXEL_FORMAT..], MTL_FORMAT_BGRA8_UNORM);
    let d = decode_texture_descriptor(&b).unwrap();
    assert_eq!(d.width, 64);
    assert_eq!(d.height, 32);
    assert_eq!(d.row_stride, 256);
    assert_eq!(d.pixel_format, MTL_FORMAT_BGRA8_UNORM);
    assert_eq!(
        d.backing_gva_size(PAGE_SHIFT_ARM64E),
        Some(((0x10u64) << RESOURCE_PAGE_SHIFT, 0x10000))
    );
    assert_eq!(d.levels.len(), 1);
    assert_eq!(d.level(0).unwrap().width, 64);
}

/// A descriptor naming no extent is not a one-by-one texture, and the three
/// call sites that used to ask `has_width && has_height` now ask this.
///
/// The sampled-source path in `draw::vulkan` clamped both fields up
/// with `.max(1)`, which sized a four-byte payload — satisfied by almost any
/// buffer — and bound a single texel of it. Nothing above that could tell
/// the result from a real bind.
#[test]
fn a_descriptor_naming_no_extent_is_not_a_one_by_one_texture() {
    use crate::contract::endian::st64;
    let mut b = vec![0u8; TEXTURE_DESC_BASE_LEN];
    st64(&mut b[0..], 0x10000);
    st32(&mut b[8..], 0x10);
    st32(&mut b[TEXTURE_DESC_ROW_STRIDE..], 256);

    // A full-length record of zeroed geometry decodes; it names no extent.
    let d = decode_texture_descriptor(&b).unwrap();
    assert_eq!(d.extent(), None);
    assert!(
        d.levels.is_empty(),
        "no extent means no level layout to build one from"
    );

    // Either field alone leaves it no extent.
    st32(&mut b[TEXTURE_DESC_WIDTH..], 64);
    assert_eq!(decode_texture_descriptor(&b).unwrap().extent(), None);
    st32(&mut b[TEXTURE_DESC_HEIGHT..], 32);
    assert_eq!(
        decode_texture_descriptor(&b).unwrap().extent(),
        Some((64, 32))
    );

    // A record too short to carry the fields never reaches the extent
    // question: the decoder refuses it by name first, so the zero geometry
    // above is a record that was long enough and said nothing.
    let short = b[..TEXTURE_DESC_WIDTH].to_vec();
    assert!(matches!(
        decode_texture_descriptor(&short),
        Err(DecodeStatus::ErrShort("res_texture_desc_short"))
    ));
}

#[test]
fn multi_mip_level_layouts() {
    use crate::contract::endian::{st16, st32, st64};
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    // 2 mips: L0 64x32 + L1 record + format trailer shifted by 36.
    let levels = 2u32;
    let body = TEXTURE_DESC_BASE_LEN + TEXTURE_DESC_MIP_LEVEL_RECORD_LEN; // 116+36=152
    let mut b = vec![0u8; body];
    st64(&mut b[0..], 0x20000);
    st32(&mut b[8..], 0x20);
    st16(&mut b[TEXTURE_DESC_MIPMAP_LEVEL_COUNT..], levels as u16);
    st32(&mut b[TEXTURE_DESC_DATA_OFFSET..], 0);
    st32(&mut b[TEXTURE_DESC_USED_SIZE..], 64 * 32 * 4);
    st32(&mut b[TEXTURE_DESC_ROW_STRIDE..], 256);
    st32(&mut b[TEXTURE_DESC_WIDTH..], 64);
    st32(&mut b[TEXTURE_DESC_HEIGHT..], 32);
    // L1 record at +72
    let rec = TEXTURE_DESC_LEVEL_RECORDS;
    st64(&mut b[rec + TEXTURE_LEVEL_OFFSET..], 0x2000);
    st64(&mut b[rec + TEXTURE_LEVEL_SIZE..], 32 * 16 * 4);
    st64(&mut b[rec + TEXTURE_LEVEL_ROW_STRIDE..], 128);
    st32(&mut b[rec + TEXTURE_LEVEL_WIDTH..], 32);
    st32(&mut b[rec + TEXTURE_LEVEL_HEIGHT..], 16);
    st32(&mut b[rec + TEXTURE_LEVEL_DEPTH..], 1);
    // Format at 86 + 36
    let pf_off = TEXTURE_DESC_PIXEL_FORMAT + TEXTURE_DESC_MIP_LEVEL_RECORD_LEN;
    st16(&mut b[pf_off..], MTL_FORMAT_BGRA8_UNORM);
    let d = decode_texture_descriptor(&b).unwrap();
    assert_eq!(d.mipmap_level_count, 2);
    assert_eq!(d.levels.len(), 2);
    assert_eq!(d.pixel_format, MTL_FORMAT_BGRA8_UNORM);
    let l0 = d.level(0).unwrap();
    assert_eq!((l0.width, l0.height, l0.row_stride), (64, 32, 256));
    let l1 = d.level(1).unwrap();
    assert_eq!((l1.width, l1.height), (32, 16));
    assert_eq!(l1.offset, 0x2000);
    assert_eq!(l1.row_stride, 128);
    let (gva1, lay1) = d.level_gva(1, PAGE_SHIFT_ARM64E).unwrap();
    assert_eq!(gva1, ((0x20u64) << RESOURCE_PAGE_SHIFT) + 0x2000);
    assert_eq!(lay1.width, 32);
    assert!(d.level_gva(2, PAGE_SHIFT_ARM64E).is_none());
}

/// A mip level the descriptor named but the body does not reach is a drop,
/// and it says so.
///
/// `mipmap_level_count` keeps what the guest declared while `levels` holds
/// fewer, so `level(n)` answers `None` for a level that was named — the same
/// answer it gives for a level that was never named at all. Without a line
/// here the two are indistinguishable, and the first is a texture level this
/// device will not sample or blit.
///
/// The unshifted-format fallback that used to sit under this case is gone
/// too, so a body this short reports no format rather than reading bytes
/// 86..88 — which for a multi-mip body are inside level record 1, not the
/// format trailer.
#[test]
fn a_level_record_the_body_does_not_reach_is_reported_not_dropped() {
    use crate::contract::endian::{st16, st32, st64};
    // Declares 3 levels but carries only L0's geometry prefix and one
    // record's worth of room — L2's record runs past the end.
    let body = TEXTURE_DESC_LEVEL_RECORDS + TEXTURE_DESC_MIP_LEVEL_RECORD_LEN;
    let mut b = vec![0u8; body];
    st64(&mut b[0..], 0x20000);
    st32(&mut b[8..], 0x20);
    st16(&mut b[TEXTURE_DESC_MIPMAP_LEVEL_COUNT..], 3);
    st32(&mut b[TEXTURE_DESC_USED_SIZE..], 64 * 32 * 4);
    st32(&mut b[TEXTURE_DESC_ROW_STRIDE..], 256);
    st32(&mut b[TEXTURE_DESC_WIDTH..], 64);
    st32(&mut b[TEXTURE_DESC_HEIGHT..], 32);
    let rec = TEXTURE_DESC_LEVEL_RECORDS;
    st32(&mut b[rec + TEXTURE_LEVEL_WIDTH..], 32);
    st32(&mut b[rec + TEXTURE_LEVEL_HEIGHT..], 16);

    let cap = crate::observe::FailCapture::start();
    let d = decode_texture_descriptor(&b).unwrap();
    assert_eq!(d.mipmap_level_count, 3, "the declaration is preserved");
    assert_eq!(d.levels.len(), 2, "only two records are reachable");
    assert!(d.level(2).is_none());
    let short = cap
        .lines()
        .into_iter()
        .find(|l| l.starts_with("texture_desc_level_record_short"))
        .expect("a level the body does not reach must be reported");
    assert!(
        short.contains("declared=3") && short.contains("decoded=2"),
        "the line must name both counts: {short}"
    );
    // Same body: the format trailer sits past the end, so there is no
    // format rather than two bytes read out of a level record.
    assert_eq!(
        d.declared_pixel_format(),
        None,
        "no format is better than a wrong one"
    );
}

/// The type-7 header states its payload length twice, and a record where the
/// two disagree says so.
///
/// The declared length at `+4` covers the header plus the payload padded to
/// four; the fourth header word is the same payload unpadded. The second was
/// called `word3` and carried no doc, which made a field with a derivable
/// meaning look like one nobody had identified.
///
/// Most fixtures in this file leave the word zero and so trip the line — that is
/// why it reports rather than refuses, and why a future promotion to a refusal
/// has to fix the fixtures first rather than only measure a boot.
#[test]
fn a_type7_header_states_its_payload_length_twice_and_says_when_they_disagree() {
    use crate::contract::endian::st32;

    // A minimal classic pipeline: header, then a one-field TLV block of seven
    // bytes, padded to eight by the declared length.
    let mut b = vec![0u8; 16 + 8];
    let blen = b.len() as u32;
    st32(&mut b[0..], TYPE7_OBJECT_RENDER_PIPELINE);
    st32(&mut b[4..], blen);
    st32(&mut b[8..], 3);
    st32(&mut b[12..], 7);
    b[16] = 1;
    b[17] = PIPELINE_TAG_VERTEX_FUNC;
    b[18] = 4;
    st32(&mut b[19..], 5);

    let cap = crate::observe::FailCapture::start();
    let d = decode_render_pipeline_descriptor(&b).expect("decodes");
    assert_eq!(d.serialized_payload_len, 7);
    assert_eq!(d.vertex_func_ref, 5);
    assert!(
        !cap.lines()
            .iter()
            .any(|l| l.contains("type7_payload_len_disagrees")),
        "7 rounds up to 8 and 16 + 8 is the declared length, so the two agree: \
         {:?}",
        cap.lines()
    );

    // One byte too long to round up to the same padded length.
    st32(&mut b[12..], 9);
    let cap2 = crate::observe::FailCapture::resume();
    decode_render_pipeline_descriptor(&b).expect("still decodes: nothing is refused");
    let lines = cap2.lines();
    assert!(
        lines
            .iter()
            .any(|l| l.contains("type7_payload_len_disagrees")
                && l.contains("payload=9")
                && l.contains("declared=24")
                && l.contains("expected=28")),
        "a disagreement must name both lengths and what the declared one would \
         have to be: {lines:?}"
    );
}

#[test]
fn compact_render_pipeline_funcs() {
    use crate::contract::endian::st32;
    // Minimal type-7 render pipeline: header + fieldCount=2 with vert/frag refs.
    let mut b = vec![0u8; 16 + 1 + 6 + 6];
    let blen = b.len() as u32;
    st32(&mut b[0..], TYPE7_OBJECT_RENDER_PIPELINE);
    st32(&mut b[4..], blen);
    st32(&mut b[8..], 9);
    b[16] = 2;
    b[17] = PIPELINE_TAG_VERTEX_FUNC;
    b[18] = 4;
    st32(&mut b[19..], 2);
    b[23] = PIPELINE_TAG_FRAGMENT_FUNC;
    b[24] = 4;
    st32(&mut b[25..], 1);
    let p = decode_render_pipeline_descriptor(&b).unwrap();
    assert_eq!(p.vertex_func_ref, 2);
    assert_eq!(p.fragment_func_ref, 1);
    assert_eq!(p.object_func_ref, 0);
    assert_eq!(p.mesh_func_ref, 0);
    assert_eq!(p.object_id, 9);
}

/// A pipeline that renders through a depth-stencil buffer declares the two
/// attachment formats it is compiled against, and one asking for multisampling
/// declares its sample count. All three decode; none of them refuses the
/// pipeline.
///
/// The regression this pins is a whole application's canvas. While `0x04`, `0x09`
/// and `0x0a` were unidentified, `note_pipeline_tlv_fields` refused every
/// descriptor carrying one — so a map view, whose pipelines carry the depth and
/// stencil formats that a desktop compositor's do not, lost every pipeline it
/// built and fell through to the clear-only path, painting a flat rectangle
/// inside a window whose chrome composited normally.
#[test]
fn a_depth_stencil_pipeline_with_a_sample_count_decodes() {
    let mut b = vec![0u8; 16 + 1 + 5 * 6];
    let blen = b.len() as u32;
    st32(&mut b[0..], TYPE7_OBJECT_RENDER_PIPELINE);
    st32(&mut b[4..], blen);
    st32(&mut b[8..], 20);
    // Written in the order the guest's serializer emits them: the descriptor's
    // own properties first, the two function refs last.
    b[16] = 5;
    let mut p = 17;
    for (tag, value) in [
        (PIPELINE_TAG_RASTER_SAMPLE_COUNT, 4u32),
        (PIPELINE_TAG_DEPTH_ATTACH_FORMAT, 252),
        (PIPELINE_TAG_STENCIL_ATTACH_FORMAT, 253),
        (PIPELINE_TAG_VERTEX_FUNC, 7),
        (PIPELINE_TAG_FRAGMENT_FUNC, 8),
    ] {
        b[p] = tag;
        b[p + 1] = 4;
        st32(&mut b[p + 2..], value);
        p += 6;
    }

    let d = decode_render_pipeline_descriptor(&b)
        .expect("a depth-stencil pipeline is decoded, not refused");
    assert_eq!(d.vertex_func_ref, 7);
    assert_eq!(d.fragment_func_ref, 8);
    assert_eq!(
        d.raster_sample_count, 4,
        "the guest's requested sample count is read rather than defaulted"
    );
}

/// A sample count this device has no attachments for is a named degradation and
/// not a refusal: the draw still runs, at one sample.
///
/// Absence and an explicit single sample are one answer, so neither reports.
#[test]
fn an_unrasterizable_sample_count_degrades_rather_than_refusing() {
    let pipeline = |count: u32| {
        let mut b = vec![0u8; 16 + 1 + 6];
        let blen = b.len() as u32;
        st32(&mut b[0..], TYPE7_OBJECT_RENDER_PIPELINE);
        st32(&mut b[4..], blen);
        st32(&mut b[8..], 21);
        b[16] = 1;
        b[17] = PIPELINE_TAG_VERTEX_FUNC;
        b[18] = 4;
        st32(&mut b[19..], 3);
        if count != 0 {
            b[16] = 2;
            b.extend_from_slice(&[PIPELINE_TAG_RASTER_SAMPLE_COUNT, 4, 0, 0, 0, 0]);
            let blen = b.len() as u32;
            st32(&mut b[4..], blen);
            let end = b.len() - 4;
            st32(&mut b[end..], count);
        }
        b
    };

    // A count no attachment path here can meet: reported, and the descriptor
    // still decodes so the geometry is kept.
    let cap = crate::observe::FailCapture::start();
    let d = decode_render_pipeline_descriptor(&pipeline(8)).expect("decodes");
    assert_eq!(d.raster_sample_count, 8);
    let lines = cap.lines();
    let degraded: Vec<&String> = lines
        .iter()
        .filter(|l| l.contains("reason=pipeline_raster_sample_count_degraded"))
        .collect();
    assert_eq!(degraded.len(), 1, "one line per distinct count: {lines:?}");
    assert!(
        degraded[0].contains("count=8") && degraded[0].contains("built_at=1"),
        "the line names what was asked for and what was built: {}",
        degraded[0]
    );

    // Neither an absent property nor an explicit single sample is a loss.
    let cap2 = crate::observe::FailCapture::resume();
    for count in [0, 1] {
        let d = decode_render_pipeline_descriptor(&pipeline(count)).expect("decodes");
        assert!(d.raster_sample_count <= 1);
    }
    assert!(
        !cap2
            .lines()
            .iter()
            .any(|l| l.contains("pipeline_raster_sample_count_degraded")),
        "single-sampling is what this device does: {:?}",
        cap2.lines()
    );
}

/// A property the guest set on a pipeline descriptor that this decoder neither
/// reads nor has identified **refuses the pipeline**, and the shape line beside
/// it is what makes a boot with no such tag readable as a measurement rather
/// than as silence.
///
/// The colour-attachment walk has had both halves of this instrument for as
/// long as it has refused an unknown tag. The pipeline's own block had neither,
/// which is what let a property become `rasterizationEnabled = yes` or
/// `alphaToCoverageEnabled = no` without anything saying the guest had asked
/// otherwise.
#[test]
fn an_unidentified_pipeline_descriptor_field_refuses_the_pipeline() {
    use crate::contract::endian::st32;
    // Tags no other test in this process uses, so `first_sight` cannot have
    // latched either shape or either drop already.
    const UNKNOWN_TAG_A: u8 = 0x6d;
    const UNKNOWN_TAG_B: u8 = 0x6e;

    let mut b = vec![0u8; 16 + 1 + 6 + 6 + 6];
    let blen = b.len() as u32;
    st32(&mut b[0..], TYPE7_OBJECT_RENDER_PIPELINE);
    st32(&mut b[4..], blen);
    st32(&mut b[8..], 9);
    b[16] = 3;
    b[17] = PIPELINE_TAG_VERTEX_FUNC;
    b[18] = 4;
    st32(&mut b[19..], 2);
    b[23] = UNKNOWN_TAG_A;
    b[24] = 4;
    st32(&mut b[25..], 4);
    b[29] = UNKNOWN_TAG_B;
    b[30] = 4;
    st32(&mut b[31..], 1);

    let cap = crate::observe::FailCapture::start();
    assert_eq!(
        decode_render_pipeline_descriptor(&b).unwrap_err(),
        DecodeStatus::ErrUnsupported("res_pipeline_field_unread"),
        "a property this decoder cannot name is refused, not defaulted"
    );
    let lines = cap.lines();

    let shape: Vec<&String> = lines
        .iter()
        .filter(|l| l.contains("type7_pipeline_shape"))
        .collect();
    assert_eq!(
        shape.len(),
        1,
        "one shape line per distinct block: {lines:?}"
    );
    assert!(
        shape[0].contains("kind=render")
            && shape[0].contains("tags=[01:4,6d:4*!,6e:4*!]")
            && shape[0].contains("unconsumed=2")
            && shape[0].contains("unknown=2"),
        "the shape line stars every unread tag and bangs the unidentified \
         ones, and the two counts are separate: {}",
        shape[0]
    );

    let drops: Vec<&String> = lines
        .iter()
        .filter(|l| l.contains("reason=pipeline_descriptor_field_dropped"))
        .collect();
    assert_eq!(drops.len(), 2, "one decline per dropped field: {lines:?}");
    assert!(
        drops.iter().any(|l| l.contains("tag=0x6d"))
            && drops.iter().any(|l| l.contains("tag=0x6e")),
        "each decline names its own tag: {drops:?}"
    );
    assert!(
        drops[0].contains("kind=render"),
        "and which pipeline kind dropped it, because the two decoders read \
         different tag sets: {}",
        drops[0]
    );

    // The lines are latched; the refusal is not. `resume`, not `start`: the
    // claim above is what is under test, and `start` would clear the latch and
    // see them a second time.
    let cap2 = crate::observe::FailCapture::resume();
    assert_eq!(
        decode_render_pipeline_descriptor(&b).unwrap_err(),
        DecodeStatus::ErrUnsupported("res_pipeline_field_unread"),
        "the line names a tag once; the pipeline is refused every time"
    );
    assert!(
        cap2.lines().is_empty(),
        "a pipeline decoded once per distinct pipeline must not re-report its \
         shape: {:?}",
        cap2.lines()
    );
}

/// The counterweight to the refusal above, and the reason it is safe: a tag
/// this decoder has *identified* and deliberately does not apply still builds
/// the pipeline, and stays off the fail channel entirely.
///
/// Without this the refusal would decline every pipeline a live guest sends —
/// `label` arrives on most of them — which is why the benign list exists and
/// why widening it without an argument is the move that would quietly undo the
/// refusal.
#[test]
fn an_identified_but_unapplied_pipeline_field_still_builds_the_pipeline() {
    use crate::contract::endian::st32;

    let mut b = vec![0u8; 16 + 1 + 6 + 6 + 6];
    let blen = b.len() as u32;
    st32(&mut b[0..], TYPE7_OBJECT_RENDER_PIPELINE);
    st32(&mut b[4..], blen);
    st32(&mut b[8..], 9);
    b[16] = 3;
    b[17] = PIPELINE_TAG_VERTEX_FUNC;
    b[18] = 4;
    st32(&mut b[19..], 7);
    b[23] = PIPELINE_TAG_FRAGMENT_FUNC;
    b[24] = 4;
    st32(&mut b[25..], 8);
    // `label` — unread by decision, argued at RENDER_PIPELINE_TAGS_BENIGN.
    b[29] = RENDER_PIPELINE_TAG_LABEL;
    b[30] = 4;
    st32(&mut b[31..], 0x1234);

    let cap = crate::observe::FailCapture::start();
    let p = decode_render_pipeline_descriptor(&b)
        .expect("an identified property this device does not apply is not a refusal");
    assert_eq!(p.vertex_func_ref, 7);
    assert_eq!(p.fragment_func_ref, 8);

    let lines = cap.lines();
    let dropped: Vec<&String> = lines
        .iter()
        .filter(|l| l.contains("reason=pipeline_descriptor_field_dropped"))
        .collect();
    assert!(
        dropped.is_empty(),
        "expected control flow stays quiet: a benign drop has an argument \
         behind it and does not belong on the fail channel: {dropped:?}"
    );

    // It is still visible, on the channel that is for visibility rather than
    // for alarm — starred as unread, and not counted as unknown.
    let shape: Vec<&String> = lines
        .iter()
        .filter(|l| l.contains("type7_pipeline_shape"))
        .collect();
    assert_eq!(shape.len(), 1, "{lines:?}");
    assert!(
        shape[0].contains("tags=[01:4,02:4,00:4*]")
            && shape[0].contains("unconsumed=1")
            && shape[0].contains("unknown=0"),
        "a benign tag is starred but not banged, and the two counts differ: {}",
        shape[0]
    );
}

/// A classic render pipeline's vertex block is taken from the offset the
/// descriptor states, not from where the bytes after the TLV block happen to
/// begin.
///
/// Tag `0x03` is `vertexDescriptor`'s offset on this shape, in the same units as
/// the colour-attachment offset beside it. The decoder used to load it into
/// `tag03`, use that variable only on the mesh branch, and locate the vertex
/// block by assuming it was everything between the end of the TLV block and the
/// colour section — an assumption that needs `skip_optional_label_and_pad` to
/// step over a `label` string of unknown length first.
///
/// The fixture puts **two** vertex descriptors in the record: a decoy exactly
/// where the old assumption lands, and the real one where tag `0x03` points. The
/// two name different buffer indices, so which offset the decoder used is
/// readable off the result rather than argued.
#[test]
fn a_classic_pipeline_takes_its_vertex_block_from_the_stated_offset() {
    use crate::contract::endian::st32;

    const DECOY_BUFFER: u32 = 0;
    const REAL_BUFFER: u32 = 2;
    let decoy = vertex_block_on_buffer(&[], DECOY_BUFFER);
    let real = vertex_block_on_buffer(&[], REAL_BUFFER);

    // Four fields of six bytes each, plus the count byte. Offsets in tags 0x03
    // and 0x08 are from the end of the 16-byte header, which is where the TLV
    // block starts, so the decoy sits at exactly the offset the old assumption
    // would have computed.
    const TLV_LEN: usize = 1 + 4 * 6;
    let decoy_rel = TLV_LEN;
    let real_rel = decoy_rel + decoy.len();
    let color_rel = real_rel + real.len();

    let mut b = vec![0u8; 16 + color_rel + 8];
    let blen = b.len() as u32;
    st32(&mut b[0..], TYPE7_OBJECT_RENDER_PIPELINE);
    st32(&mut b[4..], blen);
    st32(&mut b[8..], 12);
    b[16] = 4;
    let mut p = 17;
    for (tag, value) in [
        (PIPELINE_TAG_VERTEX_DESCRIPTOR_OFFSET, real_rel as u32),
        (PIPELINE_TAG_COLOR_ATTACH_OFFSET, color_rel as u32),
        (PIPELINE_TAG_VERTEX_FUNC, 2),
        (PIPELINE_TAG_FRAGMENT_FUNC, 1),
    ] {
        b[p] = tag;
        b[p + 1] = 4;
        st32(&mut b[p + 2..], value);
        p += 6;
    }
    assert_eq!(
        p,
        16 + TLV_LEN,
        "the TLV block is the length the offsets assume"
    );
    b[16 + decoy_rel..16 + decoy_rel + decoy.len()].copy_from_slice(&decoy);
    b[16 + real_rel..16 + real_rel + real.len()].copy_from_slice(&real);

    let cap = crate::observe::FailCapture::start();
    let d = decode_render_pipeline_descriptor(&b).expect("a classic descriptor decodes");
    assert!(d.has_vertex_descriptor_offset);
    assert_eq!(d.vertex_descriptor_offset, real_rel as u32);
    assert_eq!(d.vertex_attributes.len(), 1);
    assert_eq!(
        d.vertex_attributes[0].buffer_index, REAL_BUFFER,
        "the block at the stated offset is the one decoded; the decoy at the \
         inferred start is not"
    );
    assert!(
        !cap.lines()
            .iter()
            .any(|l| l.contains("type7_vertex_block_inferred")),
        "and a descriptor that stated its offset must not report the fallback: \
         {:?}",
        cap.lines()
    );
}

/// A shape carrying no vertex-descriptor offset carries no vertex descriptor.
///
/// The reading that licenses this: on a driven boot, the classic shape
/// `[08,01,02]` produces no vertex-descriptor entry, and every shape carrying
/// `0x03` produces one. So an absent tag is "none", not "look for it" — which is
/// what makes the stated offset usable as the only route to the block.
#[test]
fn a_classic_pipeline_without_the_offset_reports_no_vertex_descriptor() {
    use crate::contract::endian::st32;

    let mut b = vec![0u8; 16 + 13 + 8];
    let blen = b.len() as u32;
    st32(&mut b[0..], TYPE7_OBJECT_RENDER_PIPELINE);
    st32(&mut b[4..], blen);
    st32(&mut b[8..], 13);
    b[16] = 2;
    b[17] = PIPELINE_TAG_COLOR_ATTACH_OFFSET;
    b[18] = 4;
    st32(&mut b[19..], 13);
    b[23] = PIPELINE_TAG_VERTEX_FUNC;
    b[24] = 4;
    st32(&mut b[25..], 2);

    let d = decode_render_pipeline_descriptor(&b).expect("decodes");
    assert!(!d.has_vertex_descriptor_offset);
    assert!(
        d.vertex_attributes.is_empty(),
        "no offset, no descriptor: the gap between the TLV block and the colour \
         section is not one"
    );
    assert_eq!(d.vertex_func_ref, 2, "the tags it does read are unaffected");
}

#[test]
fn compact_render_pipeline_object_mesh_funcs() {
    use crate::contract::endian::st32;
    // Mesh SPI shape: tag 0x14 section offset + 0x01 object / 0x02 mesh / 0x03 frag.
    // (Host serializeMeshRenderPipelineDescriptor differentials, 2026-07-12.)
    //
    // The colour section the tag points at is present and empty. It has to be:
    // an offset naming a section the descriptor does not contain loses an
    // unreadable number of attachments and `parse_color_attachments` refuses it,
    // which is the whole point of that refusal — so a fixture that declares the
    // offset must carry the eight header bytes a real descriptor carries.
    let mut b = vec![0u8; 16 + 24 + 8];
    let blen = b.len() as u32;
    st32(&mut b[0..], TYPE7_OBJECT_RENDER_PIPELINE);
    st32(&mut b[4..], blen);
    st32(&mut b[8..], 7);
    b[16] = 4;
    b[17] = PIPELINE_TAG_MESH_SECTION_OFFSET;
    b[18] = 4;
    st32(&mut b[19..], 24); // section offset from header end
    b[23] = PIPELINE_TAG_OBJECT_FUNC; // 0x01
    b[24] = 4;
    st32(&mut b[25..], 4); // object fn ref
    b[29] = PIPELINE_TAG_MESH_FUNC; // 0x02
    b[30] = 4;
    st32(&mut b[31..], 5); // mesh fn ref
    b[35] = PIPELINE_TAG_MESH_FRAGMENT_FUNC; // 0x03
    b[36] = 4;
    st32(&mut b[37..], 3); // frag fn ref
    let p = decode_render_pipeline_descriptor(&b).unwrap();
    assert_eq!(p.object_func_ref, 4);
    assert_eq!(p.mesh_func_ref, 5);
    assert_eq!(p.fragment_func_ref, 3);
    assert_eq!(p.vertex_func_ref, 0);
    assert!(p.has_color_attachment_offset);
    assert_eq!(p.color_attachment_offset, 24);
    assert_eq!(p.object_id, 7);
}

#[test]
fn depth_stencil_object_decode() {
    use crate::contract::endian::st32;
    let mut b = vec![0u8; DEPTH_STENCIL_DESC_LEN];
    st32(&mut b[0..], TYPE7_OBJECT_DEPTH_STENCIL);
    st32(&mut b[4..], DEPTH_STENCIL_DESC_LEN as u32);
    st32(&mut b[DEPTH_STENCIL_DESC_ID..], 5);
    // compare Less=1, write enabled, both stencil enabled
    let bits = 1u32
        | DEPTH_STENCIL_DEPTH_WRITE
        | DEPTH_STENCIL_FRONT_STENCIL_ENABLED
        | DEPTH_STENCIL_BACK_STENCIL_ENABLED;
    st32(&mut b[DEPTH_STENCIL_DESC_STATE_BITS..], bits);
    st32(&mut b[DEPTH_STENCIL_DESC_FRONT_FACE + 4..], 0xff);
    st32(&mut b[DEPTH_STENCIL_DESC_FRONT_FACE + 8..], 0xff);
    let d = decode_depth_stencil_descriptor(&b).unwrap();
    assert_eq!(d.depth_stencil_id, 5);
    assert_eq!(d.depth_compare_function, 1);
    assert!(d.depth_write_enabled);
    assert!(d.front_stencil_enabled);
    assert_eq!(d.front_face.read_mask, 0xff);
}

#[test]
fn color_attachment0_blend_section() {
    use crate::contract::endian::st32;
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    // Place section at off=16 (nonzero; 0 means "absent" for callers).
    // Section: count=1, entry_rel=8, entry with fieldCount + tags.
    let off = 16usize;
    let mut buf = vec![0u8; off + 8 + 1 + 6 * 3];
    st32(&mut buf[off..], 1);
    st32(&mut buf[off + 4..], 8);
    let entry = off + 8;
    buf[entry] = 3;
    buf[entry + 1] = COLOR_ATTACHMENT_TAG_PIXEL_FORMAT;
    buf[entry + 2] = 4;
    st32(&mut buf[entry + 3..], MTL_FORMAT_BGRA8_UNORM as u32);
    buf[entry + 7] = COLOR_ATTACHMENT_TAG_BLEND_ENABLE;
    buf[entry + 8] = 4;
    st32(&mut buf[entry + 9..], 1);
    buf[entry + 13] = COLOR_ATTACHMENT_TAG_DST_RGB;
    buf[entry + 14] = 4;
    st32(&mut buf[entry + 15..], 5); // OneMinusSourceAlpha
    let all = parse_color_attachments(&buf, buf.len(), off).expect("a well-formed table decodes");
    let c = all.first().copied().unwrap_or_default();
    assert!(c.has_pixel_format);
    assert_eq!(c.pixel_format, MTL_FORMAT_BGRA8_UNORM as u32);
    assert!(c.blending_enabled);
    assert_eq!(c.dst_rgb, 5);
    assert_eq!(c.src_rgb, BLEND_FACTOR_ONE);
    assert_eq!(c.slot, 0);
    let all = parse_color_attachments(&buf, buf.len(), off).expect("a well-formed table decodes");
    assert_eq!(all.len(), 1);
}

/// An entry that omits [`COLOR_ATTACHMENT_TAG_INDEX`] falls back to its
/// position, and still carries its own state.
///
/// This pins the fallback arm specifically: no entry here declares an
/// index, which is the only reason these slots come out as a dense prefix.
/// The arm where the guest does declare one is
/// `a_colour_attachment_takes_the_slot_the_guest_declared`.
///
/// Either way `slot` is what every consumer's `find(|a| a.slot == c.slot)`
/// rests on, and that is why an `or_else(first())` beside one of those is
/// not a harmless belt-and-braces: with an entry on slot 0, `find` cannot
/// miss for slot 0, so such a fallback is reachable *only* for a secondary
/// slot that has no entry — the one case where answering with slot 0's
/// state invents it. Each slot here carries a distinct `dst_rgb` so
/// borrowing entry 0's would be visible rather than coincidentally equal.
#[test]
fn colour_attachment_slots_are_their_own_index_and_carry_their_own_state() {
    use crate::contract::endian::st32;
    // [count][off0][off1][off2] then three 1-field entries, 7 bytes each.
    const ENTRY_LEN: usize = 7;
    let off = 16usize;
    let header = 4 + 4 * 3;
    let mut buf = vec![0u8; off + header + ENTRY_LEN * 3];
    st32(&mut buf[off..], 3);
    for i in 0..3 {
        let entry_rel = header + i * ENTRY_LEN;
        st32(&mut buf[off + 4 + i * 4..], entry_rel as u32);
        let entry = off + entry_rel;
        buf[entry] = 1;
        buf[entry + 1] = COLOR_ATTACHMENT_TAG_DST_RGB;
        buf[entry + 2] = 4;
        // Distinct per slot: 10, 11, 12.
        st32(&mut buf[entry + 3..], 10 + i as u32);
    }
    let all = parse_color_attachments(&buf, buf.len(), off).expect("a well-formed table decodes");
    assert_eq!(all.len(), 3, "all three entries are in range");
    for (i, a) in all.iter().enumerate() {
        assert_eq!(
            a.slot, i as u32,
            "an entry declaring no index keeps its position"
        );
        assert_eq!(a.dst_rgb, 10 + i as u32, "each slot keeps its own state");
    }
    // What a consumer's `find` must return, and what `first()` would.
    let by_slot = |s: u32| all.iter().find(|a| a.slot == s).map(|a| a.dst_rgb);
    assert_eq!(by_slot(2), Some(12));
    assert_ne!(
        by_slot(2),
        all.first().map(|a| a.dst_rgb),
        "slot 2 must not resolve to entry 0's state"
    );
    // A secondary slot the table does not describe has no state at all.
    assert_eq!(by_slot(5), None);
}

/// A section that declares three attachments and can deliver one refuses,
/// naming both numbers.
///
/// Delivering the one is what this used to do, and the result is a pipeline
/// whose other two slots take opaque `ONE`/`ZERO` defaults — downstream cannot
/// tell that from a guest that declared one, so the wrong blend or the absent
/// render target arrives with nothing but this line behind it. The line is not
/// enough on its own: a refusal is what stops the wrong pipeline from being
/// built at all.
///
/// The entry offset for slot 1 points past the descriptor, which is one of the
/// two ways the walk can stop short; the other is an offset word that does not
/// fit, and both take the same exit.
#[test]
fn a_colour_attachment_table_that_cannot_deliver_its_count_refuses() {
    use crate::contract::endian::st32;
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    let off = 16usize;
    // Header (count + 3 offset words) then one entry: one tag, 6 bytes.
    let mut buf = vec![0u8; off + 4 + 3 * 4 + 1 + 6];
    st32(&mut buf[off..], 3);
    st32(&mut buf[off + 4..], 16); // slot 0: entry at off+16, in range
    st32(&mut buf[off + 8..], 0xffff); // slot 1: resolves past the descriptor
    st32(&mut buf[off + 12..], 0xffff); // slot 2: never reached
    let entry = off + 16;
    buf[entry] = 1;
    buf[entry + 1] = COLOR_ATTACHMENT_TAG_PIXEL_FORMAT;
    buf[entry + 2] = 4;
    st32(&mut buf[entry + 3..], MTL_FORMAT_BGRA8_UNORM as u32);

    let cap = crate::observe::FailCapture::start();
    assert_eq!(
        parse_color_attachments(&buf, buf.len(), off),
        Err(DecodeStatus::ErrShort("res_color_entry_oob")),
        "a table that cannot deliver its count refuses the descriptor"
    );
    let lines = cap.lines();

    let truncated: Vec<&String> = lines
        .iter()
        .filter(|l| l.contains("reason=color_attachment_table_truncated"))
        .collect();
    assert_eq!(
        truncated.len(),
        1,
        "one decline for the truncated table: {lines:?}"
    );
    assert!(
        truncated[0].contains("declared=3") && truncated[0].contains("decoded=1"),
        "the decline names how many were promised and how many arrived: {}",
        truncated[0]
    );
}

/// `len` is a reach *within* `bytes`, and a reach past the end of `bytes` is
/// not a reach at all.
///
/// Every bound in this function and its helpers is checked against `len` while
/// the reads index `bytes`, so the two agreeing is what makes the whole family
/// total. [`parse_vertex_block`] has always carried the equivalent clause
/// (`block_end > bytes.len()`); this one did not, and a `len` past the end
/// therefore passed `section_off + 8 > len` and then panicked on the first
/// `ld32`. Two bytes were enough.
///
/// Not guest-reachable today — `decode_render_pipeline_descriptor` requires
/// `declared == bytes.len()` before it calls this — but this is a `pub fn` in a
/// `pub mod` whose totality rested on an invariant stated nowhere, and the next
/// caller does not inherit that check.
#[test]
fn a_colour_table_reach_past_the_record_refuses_instead_of_indexing_past_it() {
    let bytes = [0x36u8, 0xd2];
    assert_eq!(
        parse_color_attachments(&bytes, 9, 1),
        Err(DecodeStatus::ErrShort("res_color_reach_past_record")),
        "a reach past the record is refused, not read"
    );

    // The refusal is about the reach, not about the section: a reach that does
    // fit still gets the ordinary short-section verdict.
    assert_eq!(
        parse_color_attachments(&bytes, bytes.len(), 1),
        Err(DecodeStatus::ErrShort("res_color_section_oob")),
        "a reach inside the record still reports the section it cannot hold"
    );

    // And `section_off == 0` stays the quiet no-section case whatever the reach
    // says, because that arm returns before either check.
    assert_eq!(
        parse_color_attachments(&bytes, usize::MAX, 0),
        Ok(Vec::new()),
        "no colour section at all is still expected control flow"
    );
}

/// A count above `MTLRenderPipelineDescriptor.colorAttachments`' eight
/// subscripts refuses, rather than binding the first eight and dropping the
/// rest onto nothing.
#[test]
fn a_colour_attachment_count_past_the_eight_slot_array_refuses() {
    use crate::contract::endian::st32;
    let off = 16usize;
    let declared = MAX_COLOR_ATTACHMENTS + 1;
    // Only the header needs to be well formed: the count is refused before any
    // entry is read, which is the point — the surplus entries need not exist for
    // the descriptor to be unbuildable.
    let mut buf = vec![0u8; off + 4 + 4 * declared];
    st32(&mut buf[off..], declared as u32);
    let cap = crate::observe::FailCapture::start();
    assert_eq!(
        parse_color_attachments(&buf, buf.len(), off),
        Err(DecodeStatus::ErrUnsupported("res_color_count_over"))
    );
    assert!(
        cap.lines()
            .iter()
            .any(|l| l.contains("reason=color_attachment_table_truncated")
                && l.contains(&format!("declared={declared}"))),
        "and says which count it refused on: {:?}",
        cap.lines()
    );
}

/// An entry naming a slot the eight-slot array has no subscript for refuses,
/// because the fallback it used to take was itself the aliasing to avoid.
///
/// Falling back to the table position puts this entry's blend state, write mask
/// and pixel format on a slot 0..7 — a real attachment the guest named nothing
/// about. There is no correct slot to place it on, so there is no correct
/// pipeline to build.
#[test]
fn a_colour_attachment_slot_past_the_array_refuses_instead_of_taking_a_position() {
    use crate::contract::endian::st32;
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    let off = 16usize;
    let mut buf = vec![0u8; off + 8 + 1 + 2 * 6];
    st32(&mut buf[off..], 1);
    st32(&mut buf[off + 4..], 8);
    let entry = off + 8;
    buf[entry] = 2;
    buf[entry + 1] = COLOR_ATTACHMENT_TAG_INDEX;
    buf[entry + 2] = 4;
    // The first index with no subscript, so position 0 is what it would alias.
    st32(&mut buf[entry + 3..], MAX_COLOR_ATTACHMENTS as u32);
    buf[entry + 7] = COLOR_ATTACHMENT_TAG_PIXEL_FORMAT;
    buf[entry + 8] = 4;
    st32(&mut buf[entry + 9..], MTL_FORMAT_BGRA8_UNORM as u32);

    let cap = crate::observe::FailCapture::start();
    assert_eq!(
        parse_color_attachments(&buf, buf.len(), off),
        Err(DecodeStatus::ErrUnsupported("res_color_slot_over"))
    );
    assert!(
        cap.lines()
            .iter()
            .any(|l| l.contains("reason=color_attachment_index_out_of_range")),
        "the refusal keeps its existing line: {:?}",
        cap.lines()
    );
}

/// An entry whose `field_count` outruns the descriptor is refused, not decoded
/// down to the defaults for everything past the cut.
///
/// The walk used to `break` here and say nothing, so `entry_tag_u32` returned
/// the absent-field default for every tag the record ended before: opaque
/// `ONE`/`ZERO` blending and no pixel format, on an attachment the guest had
/// described. Build an entry that claims three fields and delivers one and a
/// half, and assert the refusal names the pair.
#[test]
fn a_colour_attachment_entry_shorter_than_its_field_count_is_refused() {
    use crate::contract::endian::st32;
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;

    let off = 16usize;
    // Header (count + one offset), then `[3][01:4 <fmt>][02:4 <trunc>]` — the
    // second field's length word is present and its four value bytes are not.
    let entry_len = 1 + 6 + 2;
    let mut buf = vec![0u8; off + 8 + entry_len];
    st32(&mut buf[off..], 1);
    st32(&mut buf[off + 4..], 8);
    let entry = off + 8;
    buf[entry] = 3;
    buf[entry + 1] = COLOR_ATTACHMENT_TAG_PIXEL_FORMAT;
    buf[entry + 2] = 4;
    st32(&mut buf[entry + 3..], MTL_FORMAT_BGRA8_UNORM as u32);
    buf[entry + 7] = COLOR_ATTACHMENT_TAG_BLEND_ENABLE;
    buf[entry + 8] = 4;

    let cap = crate::observe::FailCapture::start();
    assert_eq!(
        parse_color_attachments(&buf, buf.len(), off),
        Err(DecodeStatus::ErrShort("res_color_entry_fields_short")),
        "an entry that promises a field the record does not hold is refused"
    );
    let lines = cap.lines();
    let short: Vec<&String> = lines
        .iter()
        .filter(|l| l.contains("reason=color_attachment_entry_short"))
        .collect();
    assert_eq!(short.len(), 1, "one line for the entry: {lines:?}");
    assert!(
        short[0].contains("fields=1/3"),
        "the line separates an entry read part-way from one absent entirely: {}",
        short[0]
    );

    // The refusal is not latched even though the line is, the same split the
    // unread-tag sibling carries.
    let cap2 = crate::observe::FailCapture::resume();
    assert_eq!(
        parse_color_attachments(&buf, buf.len(), off),
        Err(DecodeStatus::ErrShort("res_color_entry_fields_short"))
    );
    assert!(
        cap2.lines().is_empty(),
        "the line is deduped: {:?}",
        cap2.lines()
    );
}

/// A colour-attachment field this decoder does not read refuses the pipeline,
/// and the shape line beside it is what makes a boot with *no* drops readable
/// as a measurement rather than as silence.
///
/// This test used to `.expect("a well-formed table decodes")` and assert only
/// that the drop was reported. Reporting is not the bar: the ten consumed tags
/// are the whole of `MTLRenderPipelineColorAttachmentDescriptor`, so an
/// eleventh is a property this decoder cannot place, and building the pipeline
/// without it puts Metal's default where the guest set its own.
#[test]
fn an_unconsumed_colour_attachment_field_refuses_the_pipeline() {
    use crate::contract::endian::st32;
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    // A tag no other test in this process uses, so `first_sight` cannot
    // have latched it already.
    const UNKNOWN_TAG: u8 = 0x7f;
    const UNKNOWN_VALUE: u32 = 13;
    let off = 16usize;
    let mut buf = vec![0u8; off + 8 + 1 + 2 * 6];
    st32(&mut buf[off..], 1);
    st32(&mut buf[off + 4..], 8);
    let entry = off + 8;
    buf[entry] = 2;
    buf[entry + 1] = COLOR_ATTACHMENT_TAG_PIXEL_FORMAT;
    buf[entry + 2] = 4;
    st32(&mut buf[entry + 3..], MTL_FORMAT_BGRA8_UNORM as u32);
    buf[entry + 7] = UNKNOWN_TAG;
    buf[entry + 8] = 4;
    st32(&mut buf[entry + 9..], UNKNOWN_VALUE);

    let cap = crate::observe::FailCapture::start();
    assert_eq!(
        parse_color_attachments(&buf, buf.len(), off),
        Err(DecodeStatus::ErrUnsupported("res_color_field_unread")),
        "a tag outside the descriptor is refused, not defaulted"
    );
    let lines = cap.lines();

    let shape: Vec<&String> = lines
        .iter()
        .filter(|l| l.contains("type7_color_attach_shape"))
        .collect();
    assert_eq!(
        shape.len(),
        1,
        "one shape line per distinct entry: {lines:?}"
    );
    assert!(
        shape[0].contains("tags=[01:4,7f:4*]") && shape[0].contains("unconsumed=1"),
        "the shape line names every tag and stars the unread ones: {}",
        shape[0]
    );

    let drop: Vec<&String> = lines
        .iter()
        .filter(|l| l.contains("reason=color_attachment_field_dropped"))
        .collect();
    assert_eq!(drop.len(), 1, "one decline per dropped field: {lines:?}");
    assert!(
        drop[0].contains("tag=0x7f") && drop[0].contains("value=13"),
        "the decline carries the tag and the value, which is what \
         identifies the field: {}",
        drop[0]
    );

    // The line is latched and the refusal is not. `resume`, not `start`: the
    // claim the window above made is the thing under test, and `start` would
    // clear it and see both lines a second time.
    let cap2 = crate::observe::FailCapture::resume();
    assert_eq!(
        parse_color_attachments(&buf, buf.len(), off),
        Err(DecodeStatus::ErrUnsupported("res_color_field_unread")),
        "every pipeline carrying the tag is refused, not just the first"
    );
    assert!(
        cap2.lines().is_empty(),
        "the census is deduped per shape and per (tag, len, value): {:?}",
        cap2.lines()
    );
}

/// A colour attachment binds to the slot the guest named, not to its
/// position in the section's offset table.
///
/// Every consumer selects the pipeline's blend state, write mask and pixel
/// format with `find(|a| a.slot == c.slot)`, so a slot derived from the
/// table position binds one attachment's state to another's slot the moment
/// the guest stops serializing a dense in-order prefix. Tag `0x00` is the
/// declared index — the same tag that carries `VERTEX_ATTR_TAG_LOCATION`
/// and `VERTEX_LAYOUT_TAG_BUFFER_INDEX` in the two sibling sections this
/// serializer emits in the identical shape, both of which this decoder
/// already read from the wire.
#[test]
fn a_colour_attachment_takes_the_slot_the_guest_declared() {
    use crate::contract::endian::st32;
    use crate::contract::pixel_format::{MTL_FORMAT_BGRA8_UNORM, MTL_FORMAT_RGBA8_UNORM};

    // Two entries, declared out of order: table position 0 names slot 3 and
    // position 1 names slot 1. Nothing but the declared index distinguishes
    // them from a dense prefix.
    let off = 16usize;
    let entry_len = 1 + 2 * 6;
    let mut buf = vec![0u8; off + 12 + 2 * entry_len];
    st32(&mut buf[off..], 2);
    st32(&mut buf[off + 4..], 12);
    st32(&mut buf[off + 8..], (12 + entry_len) as u32);
    let mut put = |entry: usize, index: u32, fmt: u32| {
        buf[entry] = 2;
        buf[entry + 1] = COLOR_ATTACHMENT_TAG_INDEX;
        buf[entry + 2] = 4;
        st32(&mut buf[entry + 3..], index);
        buf[entry + 7] = COLOR_ATTACHMENT_TAG_PIXEL_FORMAT;
        buf[entry + 8] = 4;
        st32(&mut buf[entry + 9..], fmt);
    };
    put(off + 12, 3, MTL_FORMAT_BGRA8_UNORM as u32);
    put(off + 12 + entry_len, 1, MTL_FORMAT_RGBA8_UNORM as u32);

    let got = parse_color_attachments(&buf, buf.len(), off).expect("a well-formed table decodes");
    assert_eq!(got.len(), 2);
    assert_eq!(
        (got[0].slot, got[1].slot),
        (3, 1),
        "the slot is the declared index; positions would read (0, 1)"
    );
    assert_eq!(
        got.iter().find(|a| a.slot == 3).map(|a| a.pixel_format),
        Some(MTL_FORMAT_BGRA8_UNORM as u32),
        "the lookup every consumer performs must reach this entry's own state"
    );
    assert_eq!(
        got.iter().find(|a| a.slot == 1).map(|a| a.pixel_format),
        Some(MTL_FORMAT_RGBA8_UNORM as u32)
    );

    // The index is a consumed field now, so it is no longer reported as a
    // field this decoder dropped.
    let cap = crate::observe::FailCapture::start();
    let _ = parse_color_attachments(&buf, buf.len(), off);
    assert!(
        !cap.lines()
            .iter()
            .any(|l| l.contains("reason=color_attachment_field_dropped")),
        "tag 0x00 is read, not dropped: {:?}",
        cap.lines()
    );
}

/// The ninth colour-attachment tag is `MTLColorWriteMask`, and an entry
/// that omits it left the property at `all`.
#[test]
fn a_colour_attachment_write_mask_decodes_and_defaults_to_all() {
    use crate::contract::endian::st32;
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;

    // `[fieldCount][01 pixelFormat][09 writeMask]`, the shape the live
    // guest sent (alpha-only, value 1).
    let off = 16usize;
    let mut buf = vec![0u8; off + 8 + 1 + 2 * 6];
    st32(&mut buf[off..], 1);
    st32(&mut buf[off + 4..], 8);
    let entry = off + 8;
    buf[entry] = 2;
    buf[entry + 1] = COLOR_ATTACHMENT_TAG_PIXEL_FORMAT;
    buf[entry + 2] = 4;
    st32(&mut buf[entry + 3..], MTL_FORMAT_BGRA8_UNORM as u32);
    buf[entry + 7] = COLOR_ATTACHMENT_TAG_WRITE_MASK;
    buf[entry + 8] = 4;
    st32(&mut buf[entry + 9..], MTL_COLOR_WRITE_MASK_ALPHA);
    let masked =
        parse_color_attachments(&buf, buf.len(), off).expect("a well-formed table decodes");
    assert_eq!(
        masked.first().map(|c| c.write_mask),
        Some(ColorWriteMask::new(MTL_COLOR_WRITE_MASK_ALPHA).unwrap())
    );
    assert_ne!(masked[0].write_mask.bits, MTL_COLOR_WRITE_MASK_ALL);

    // Same entry with the tag dropped: `all`, not `none`. This is the arm
    // a derived `Default` on a bare `u32` would have made a black
    // attachment, and every pipeline in the tree takes it.
    buf[entry] = 1;
    let plain = parse_color_attachments(&buf, buf.len(), off).expect("a well-formed table decodes");
    assert_eq!(plain[0].write_mask.bits, MTL_COLOR_WRITE_MASK_ALL);
    assert_eq!(plain[0].write_mask, ColorWriteMask::default());
}

/// The tag identification is an argument from `MTLRenderPipeline.h`'s
/// property order, so it needs a standing check that it still holds. A
/// value no four-bit mask can carry refuses the pipeline rather than
/// masking channels on a guess.
///
/// This used to keep `ColorWriteMask::default()` and log, on the argument
/// that leaving the pre-decode behaviour in place was safer than
/// introducing a new refusal. The argument does not survive reading what
/// the default *is*: `all`, the widest mask there is, so a guest that
/// masked a channel off got it written. And the value that gets here is
/// one `MTLColorWriteMask` cannot hold, which means the tag mapping this
/// entry was read through is wrong — the four bits are not a channel
/// selection at all, they are four bits of some other property. Writing
/// every channel from a field this decoder has misidentified is the guess;
/// refusing is what a device that does not know what it is holding does.
#[test]
fn a_write_mask_outside_the_four_bits_refuses_the_pipeline() {
    use crate::contract::endian::st32;
    let off = 16usize;
    let mut buf = vec![0u8; off + 8 + 1 + 6];
    st32(&mut buf[off..], 1);
    st32(&mut buf[off + 4..], 8);
    let entry = off + 8;
    buf[entry] = 1;
    buf[entry + 1] = COLOR_ATTACHMENT_TAG_WRITE_MASK;
    buf[entry + 2] = 4;
    st32(&mut buf[entry + 3..], 0x1234_5678);

    let cap = crate::observe::FailCapture::start();
    let status = parse_color_attachments(&buf, buf.len(), off)
        .expect_err("a mask outside the four bits is not a table this device can build");
    assert_eq!(
        status,
        DecodeStatus::ErrUnsupported("res_color_write_mask_over")
    );
    let lines = cap.lines();
    assert!(
        lines
            .iter()
            .any(|l| l.contains("reason=color_write_mask_out_of_range")
                && l.contains("value=305419896")),
        "the refusal names the value that refuted it: {lines:?}"
    );

    // The four-bit neighbour of the same value still decodes, so the refusal
    // is about the range and not about the tag being present.
    st32(&mut buf[entry + 3..], MTL_COLOR_WRITE_MASK_ALL);
    let ok = parse_color_attachments(&buf, buf.len(), off).expect("0xf is a mask Metal can hold");
    assert_eq!(ok[0].write_mask.bits, MTL_COLOR_WRITE_MASK_ALL);
}

/// The measured case, from an x86/Vulkan boot in Dark appearance: a 27x27
/// `RG8Unorm` corner mask at offset 0x850, 384-byte rows, in a 12 288-byte
/// allocation. `row_stride * height` scores 12 496 and refuses it; the
/// bytes actually read end at 12 166, with 122 to spare.
///
/// The guest's allocation is exactly right, and the old bound demanded
/// trailing padding that no row occupies — so the refusal dropped the
/// WindowServer's whole composite draw and the window rendered with square
/// corners and no shadow.
#[test]
fn a_levels_read_span_stops_at_the_last_row_not_a_last_stride() {
    const OFFSET: u64 = 0x850;
    const STRIDE: u64 = 384;
    const HEIGHT: u32 = 27;
    const TIGHT: u32 = 27 * 2;
    const ALLOCATION: u64 = 12288;

    let span = TextureLevelLayout {
        offset: OFFSET,
        size: 0,
        row_stride: STRIDE,
        width: 27,
        height: HEIGHT,
        depth: 1,
    }
    .read_span(TIGHT)
    .unwrap();
    assert_eq!(span, 26 * STRIDE + TIGHT as u64);
    assert_eq!(OFFSET + span, 12166);
    assert!(
        OFFSET + span <= ALLOCATION,
        "the guest sized this allocation for exactly this image"
    );
    // The bound this replaced, stated so the regression is visible here
    // rather than only on a live guest.
    assert!(OFFSET + STRIDE * HEIGHT as u64 > ALLOCATION);

    // A tight image (no padding) is unchanged: the two forms agree.
    let tight_span = TextureLevelLayout {
        offset: OFFSET,
        size: 0,
        row_stride: TIGHT as u64,
        width: 27,
        height: HEIGHT,
        depth: 1,
    }
    .read_span(TIGHT)
    .unwrap();
    assert_eq!(tight_span, (TIGHT as u64) * HEIGHT as u64);

    // A single row is its own tight length, with no stride charged at all.
    assert_eq!(
        TextureLevelLayout {
            offset: OFFSET,
            size: 0,
            row_stride: STRIDE,
            width: 27,
            height: 1,
            depth: 1
        }
        .read_span(TIGHT),
        Some(TIGHT as u64)
    );

    // Zero height has no rows and therefore no span; the caller rejects
    // that extent separately, and this must not underflow into a huge one.
    assert_eq!(
        TextureLevelLayout {
            offset: OFFSET,
            size: 0,
            row_stride: STRIDE,
            width: 27,
            height: 0,
            depth: 1
        }
        .read_span(TIGHT),
        None
    );
}

/// The array/volume form of the same rule. A slice is charged for every
/// plane below its last one in full, and for its last plane only as far as
/// the last row reaches — so an allocation sized exactly for N slices is
/// accepted rather than refused for the padding after the very last row.
#[test]
fn a_slice_read_span_charges_full_planes_and_a_tight_last_row() {
    const STRIDE: u64 = 384;
    const HEIGHT: u32 = 27;
    const TIGHT: u32 = 27 * 2;
    let layout = TextureLevelLayout {
        offset: 0,
        size: 0,
        row_stride: STRIDE,
        width: 27,
        height: HEIGHT,
        depth: 4,
    };

    // Depth 0 and 1 are both one plane, and then this is exactly `read_span`.
    // Read off the layout's own field now, so this asserts the encoding where
    // `planes()` applies it rather than where a caller used to.
    let flat = layout.read_span(TIGHT).unwrap();
    let at_depth = |depth: u32| {
        TextureLevelLayout { depth, ..layout }
            .slice_read_span(TIGHT)
            .unwrap()
    };
    assert_eq!(at_depth(1), flat);
    assert_eq!(at_depth(0), flat, "0 and 1 are both one plane");

    // Three whole planes, then the fourth's rows.
    let plane = STRIDE * HEIGHT as u64;
    assert_eq!(at_depth(4), 3 * plane + flat);

    // The stride form this replaced overcounts by exactly one row's padding,
    // whatever the plane count. `slice_stride` is that form, so the two are
    // compared against each other rather than against a re-spelled product.
    for depth in [1u32, 2, 4] {
        let level = TextureLevelLayout { depth, ..layout };
        assert_eq!(
            level.slice_stride().unwrap() - level.slice_read_span(TIGHT).unwrap(),
            STRIDE - TIGHT as u64
        );
    }

    // A zero depth strides as one plane, for the same encoding reason.
    assert_eq!(
        TextureLevelLayout { depth: 0, ..layout }.slice_stride(),
        Some(plane)
    );

    // Zero height has no rows, so no span — and must not underflow. Its stride
    // is zero rather than one invented row, which is the half a `height.max(1)`
    // used to get wrong.
    let no_rows = TextureLevelLayout {
        height: 0,
        ..layout
    };
    assert_eq!(no_rows.slice_read_span(TIGHT), None);
    assert_eq!(no_rows.slice_stride(), Some(0));
}

#[test]
fn texture_view_simple() {
    use crate::contract::endian::{st16, st32};
    let mut b = vec![0u8; TEXTURE_VIEW_MIN_SIMPLE];
    st32(
        &mut b[TEXTURE_VIEW_DESC_OPCODE..],
        TEXTURE_VIEW_OPCODE_SIMPLE,
    );
    st32(
        &mut b[TEXTURE_VIEW_DESC_LEN..],
        TEXTURE_VIEW_MIN_SIMPLE as u32,
    );
    st32(&mut b[TEXTURE_VIEW_DESC_TEXTURE_REF..], 10);
    st32(&mut b[TEXTURE_VIEW_DESC_BASE_REF..], 3);
    st16(&mut b[TEXTURE_VIEW_DESC_PIXEL_FORMAT..], 0x50);
    let v = decode_texture_view_descriptor(&b).unwrap();
    assert_eq!(v.base_texture_ref, 3);
    assert_eq!(v.view_texture_ref, 10);
    assert_eq!(v.pixel_format, 0x50);
    assert_eq!(v.declared_pixel_format(), Some(0x50));
    assert!(!v.carries_range());
    assert!(!v.carries_swizzle());

    // A view that states no format must not claim one. This used to be an
    // unconditional `true`, which disagreed with `decode_texture_descriptor`
    // — the decoder every current reader of this flag goes through — about
    // what the flag means. `MTLPixelFormatInvalid` is 0, so a zero here is
    // an absent format and the gates that fail closed on it must see that.
    st16(&mut b[TEXTURE_VIEW_DESC_PIXEL_FORMAT..], 0);
    let none = decode_texture_view_descriptor(&b).unwrap();
    assert_eq!(none.pixel_format, 0);
    assert_eq!(
        none.declared_pixel_format(),
        None,
        "format 0 is MTLPixelFormatInvalid, not a format the view named"
    );
}

#[test]
fn texture_view_swizzle_form() {
    use crate::contract::endian::{st16, st32, st64};
    let mut b = vec![0u8; TEXTURE_VIEW_MIN_SWIZZLE];
    st32(
        &mut b[TEXTURE_VIEW_DESC_OPCODE..],
        TEXTURE_VIEW_OPCODE_SWIZZLE,
    );
    st32(
        &mut b[TEXTURE_VIEW_DESC_LEN..],
        TEXTURE_VIEW_MIN_SWIZZLE as u32,
    );
    st32(&mut b[TEXTURE_VIEW_DESC_TEXTURE_REF..], 11);
    st32(&mut b[TEXTURE_VIEW_DESC_BASE_REF..], 4);
    st16(&mut b[TEXTURE_VIEW_DESC_PIXEL_FORMAT..], 0x46);
    st16(
        &mut b[TEXTURE_VIEW_DESC_TEXTURE_TYPE..],
        TEXTURE_VIEW_MTL_TYPE_2D,
    );
    st64(&mut b[TEXTURE_VIEW_DESC_LEVEL_BASE..], 1);
    st64(&mut b[TEXTURE_VIEW_DESC_LEVEL_COUNT..], 1);
    st64(&mut b[TEXTURE_VIEW_DESC_SLICE_BASE..], 0);
    st64(&mut b[TEXTURE_VIEW_DESC_SLICE_COUNT..], 1);
    // Selectors: B,G,R,A (4,3,2,5)
    b[TEXTURE_VIEW_DESC_SWIZZLE] = 4;
    b[TEXTURE_VIEW_DESC_SWIZZLE + 1] = 3;
    b[TEXTURE_VIEW_DESC_SWIZZLE + 2] = 2;
    b[TEXTURE_VIEW_DESC_SWIZZLE + 3] = 5;
    let v = decode_texture_view_descriptor(&b).unwrap();
    assert_eq!(v.view_opcode, TEXTURE_VIEW_OPCODE_SWIZZLE);
    assert_eq!(v.base_texture_ref, 4);
    assert!(v.carries_range());
    assert_eq!(v.level_base, 1);
    assert_eq!((v.slice_base, v.slice_count), (0, 1));
    assert_eq!(v.texture_type, TEXTURE_VIEW_MTL_TYPE_2D);
    assert!(v.carries_swizzle());
    assert_eq!(v.swizzle, [4, 3, 2, 5]);
}

#[test]
fn texture_view_ranged_form() {
    use crate::contract::endian::{st16, st32, st64};
    let mut b = vec![0u8; TEXTURE_VIEW_MIN_RANGED];
    st32(
        &mut b[TEXTURE_VIEW_DESC_OPCODE..],
        TEXTURE_VIEW_OPCODE_RANGED,
    );
    st32(
        &mut b[TEXTURE_VIEW_DESC_LEN..],
        TEXTURE_VIEW_MIN_RANGED as u32,
    );
    st32(&mut b[TEXTURE_VIEW_DESC_TEXTURE_REF..], 12);
    st32(&mut b[TEXTURE_VIEW_DESC_BASE_REF..], 5);
    st16(&mut b[TEXTURE_VIEW_DESC_PIXEL_FORMAT..], 0x50);
    st16(
        &mut b[TEXTURE_VIEW_DESC_TEXTURE_TYPE..],
        TEXTURE_VIEW_MTL_TYPE_2D,
    );
    st64(&mut b[TEXTURE_VIEW_DESC_LEVEL_BASE..], 2);
    st64(&mut b[TEXTURE_VIEW_DESC_LEVEL_COUNT..], 1);
    st64(&mut b[TEXTURE_VIEW_DESC_SLICE_BASE..], 0);
    st64(&mut b[TEXTURE_VIEW_DESC_SLICE_COUNT..], 1);
    let v = decode_texture_view_descriptor(&b).unwrap();
    assert_eq!(v.view_opcode, TEXTURE_VIEW_OPCODE_RANGED);
    assert_eq!(v.level_base, 2);
    assert_eq!(v.level_count, 1);
    assert!(v.carries_range());
    assert!(!v.carries_swizzle());
}

#[test]
fn decodes_opcode9_buffer_texture_live_blobs() {
    // Two real 64-byte opcode-9 descriptors captured from a live x86
    // reims-vgpu-pci boot (Notification Center widget-tile sampled inputs,
    // pipe=51/53). See journal 2026-07-17 Reims VGPU-VIEW-RESOLVE-OPCODE9.
    let b1 = hex_to_bytes(
        "0900000040000000090000000800000000000000000000000005000000000000\
         421150001c0100001c0100000100000001000100010010000000000000000000",
    );
    let d1 = decode_buffer_texture_descriptor(&b1).unwrap();
    assert_eq!(d1.new_texture_ref, 9);
    assert_eq!(d1.buffer_ref, 8);
    assert_eq!(d1.offset, 0);
    assert_eq!(d1.bytes_per_row, 1280);
    assert_eq!(d1.desc.pixel_format, 0x50); // BGRA8_UNORM
    assert_eq!(d1.desc.texture_type as u16, TEXTURE_VIEW_MTL_TYPE_2D);
    assert_eq!((d1.desc.width, d1.desc.height), (284, 284));
    assert_eq!(d1.desc.depth, 1);
    assert_eq!(d1.desc.mipmap_level_count, 1);
    assert_eq!(d1.desc.sample_count, 1);
    assert_eq!(d1.desc.array_length, 1);
    // The fields the inline reading dropped. `usage` is the byte the old
    // `flags & 0xf` / `flags >> 16` pair stepped straight over: the packed
    // word here is `0x00501142`, so `usage` is `0x11` —
    // `MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView`, which
    // is the guest saying it will sample this tile through a *different*
    // pixel format than the one it declared. This device discarded that on
    // every buffer-backed texture.
    assert_eq!(d1.desc.usage, 0x11);
    assert_eq!(d1.desc.resource_options, 0x0010);
    assert_eq!(d1.desc.protection_options, 0);
    assert!(d1.desc.allow_gpu_optimized_contents);
    assert!(!d1.desc.framebuffer_only);
    assert!(!d1.desc.is_drawable);

    let b2 = hex_to_bytes(
        "09000000400000004c0000004b000000000000000000000000010000000000004\
         211500040000000400000000100000001000100010010000000000000000000",
    );
    let d2 = decode_buffer_texture_descriptor(&b2).unwrap();
    assert_eq!(d2.new_texture_ref, 76);
    assert_eq!(d2.buffer_ref, 75);
    assert_eq!(d2.bytes_per_row, 256);
    assert_eq!(d2.desc.pixel_format, 0x50);
    assert_eq!((d2.desc.width, d2.desc.height), (64, 64));

    // A real texture-VIEW (opcode 8) is NOT a buffer texture.
    let mut view = vec![0u8; TEXTURE_VIEW_MIN_RANGED];
    crate::contract::endian::st32(
        &mut view[TEXTURE_VIEW_DESC_OPCODE..],
        TEXTURE_VIEW_OPCODE_RANGED,
    );
    crate::contract::endian::st32(
        &mut view[TEXTURE_VIEW_DESC_LEN..],
        TEXTURE_VIEW_MIN_RANGED as u32,
    );
    assert!(decode_buffer_texture_descriptor(&view).is_err());
    assert_eq!(
        texture_type8_opcode(&view),
        Some(TEXTURE_VIEW_OPCODE_RANGED)
    );
    assert_eq!(
        texture_type8_opcode(&b1),
        Some(TEXTURE_VIEW_OPCODE_BUFFER_TEXTURE)
    );
}

/// The header peek is bounded by the header, not by one variant's total.
///
/// A truncated type-8 blob still names which variant the guest meant, and
/// that name is what routes it to the decoder that can report the
/// truncation. Bounding the peek by the *simple view's* total length (20)
/// instead returned `None` for every blob of 8..20 bytes, so a short record
/// of any other variant read as "no opcode" and was refused by whichever
/// caller's fallback ran first, naming the wrong reason.
#[test]
fn the_type8_header_peek_is_bounded_by_the_header_it_reads() {
    // Exactly the header, declaring a 64-byte buffer-backed texture: the
    // shortest blob that carries an opcode at all.
    let mut short = vec![0u8; OP_HDR];
    crate::contract::endian::st32(
        &mut short[TEXTURE_VIEW_DESC_OPCODE..],
        TEXTURE_VIEW_OPCODE_BUFFER_TEXTURE,
    );
    crate::contract::endian::st32(&mut short[TEXTURE_VIEW_DESC_LEN..], BUF_TEX_MIN_LEN as u32);
    assert_eq!(
        texture_type8_header(&short),
        Some((TEXTURE_VIEW_OPCODE_BUFFER_TEXTURE, BUF_TEX_MIN_LEN as u32)),
        "a header-length blob must still name its variant and its declared length"
    );
    // And the checked reader is still the one that refuses it.
    assert!(decode_buffer_texture_descriptor(&short).is_err());

    // One byte short of a header is the only case with nothing to read.
    assert_eq!(texture_type8_header(&short[..OP_HDR - 1]), None);
    assert_eq!(texture_type8_opcode(&short[..OP_HDR - 1]), None);
}

fn hex_to_bytes(s: &str) -> Vec<u8> {
    let clean: String = s.chars().filter(|c| c.is_ascii_hexdigit()).collect();
    (0..clean.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&clean[i..i + 2], 16).unwrap())
        .collect()
}

#[test]
fn property_fuzz_types() {
    for t in 0u8..16 {
        let bytes = vec![0u8; 128];
        let _ = decode_descriptor(t, &bytes);
    }
}

/// A type-7 subtype **is** a `PGSerializer` opcode, and the two this module
/// still spells as numbers are exactly the two nothing has driven.
///
/// The type-7 object-list entry is reached by object *type* rather than off
/// the command stream, which is why its subtypes look like a private
/// enumeration and were written as one. They are not: `0x03` is
/// `newSamplerState`, `0x04` is `newDepthStencilState` and `0x36` is
/// `newIndirectCommandBuffer`, all three now taken from the crate that
/// derived them, and `decode_icb_descriptor` reads the identical 88 bytes
/// the fixture instrument feeds `ops::icb`.
///
/// `0x0b` and `0x0e` stay numbers because no capture has produced them.
/// Their selectors are the pipeline-creation family, which needs a
/// *serialized* descriptor rather than a Metal descriptor object to drive,
/// so they have no manifest row at all — they are the remainder behind
/// `counts()`, not an `Unimplemented` row, and driving them with malformed
/// input would prove nothing.
///
/// The class filter is load-bearing. `0x0b` is also
/// `drawIndexedPrimitives:…:baseVertex:baseInstance:` on the render
/// encoder, and reading that as support for this tag would be taking a
/// number from the wrong opcode space — the same trap `0x1b` sets, where
/// the texture-view creation and `useHeap:` share a value.
#[test]
fn the_undrivable_type7_subtypes_are_the_pipeline_pair_and_nothing_claims_them() {
    let serializer_opcodes = |op: u32| {
        reims_vgpu_wire::manifest::MANIFEST
            .iter()
            .filter(|e| e.class == "PGSerializer")
            .any(|e| e.opcodes.contains(&op))
    };

    // The three that are derived must still be, in the serializer's space.
    for (tag, name) in [
        (TYPE7_OBJECT_SAMPLER, "TYPE7_OBJECT_SAMPLER"),
        (TYPE7_OBJECT_DEPTH_STENCIL, "TYPE7_OBJECT_DEPTH_STENCIL"),
        (TYPE7_OBJECT_ICB, "TYPE7_OBJECT_ICB"),
    ] {
        assert!(
            serializer_opcodes(tag),
            "{name} = {tag:#x} is no longer an opcode Apple's PGSerializer                  manifest lists"
        );
    }

    // The two that are not must stay unclaimed. A capture that drives the
    // pipeline family gives them a row, and then the number here has a
    // derivation and must come from it rather than stay a literal.
    for (tag, name) in [
        (
            TYPE7_OBJECT_COMPUTE_PIPELINE,
            "TYPE7_OBJECT_COMPUTE_PIPELINE",
        ),
        (TYPE7_OBJECT_RENDER_PIPELINE, "TYPE7_OBJECT_RENDER_PIPELINE"),
    ] {
        assert!(
            !serializer_opcodes(tag),
            "{name} = {tag:#x} now has a PGSerializer row, so it is derived                  and must be read from reims-vgpu-wire rather than written here"
        );
    }
}

/// A layout table longer than `u16::MAX` entries reports its real length.
///
/// Every one of these counts used to narrow the quotient of two 32-bit layout
/// offsets to `u16` on the way out, which wraps: the 65 537-entry table below
/// answered 1. The callers then either read one entry of a table the guest
/// filled, or — at the attribute-stride readers, which bounds-check the guest's
/// index against this count — refused every index from 1 up as past the end.
///
/// A guest whose layout agrees with its own create body cannot get here; the
/// body declares each count in one byte. The layout blob is a separate copy and
/// nothing checks the two against each other, which is why the count is not
/// left resting on that agreement.
#[test]
fn a_layout_table_past_u16_reports_its_real_length() {
    let entries = u32::from(u16::MAX) + 2;

    // Kernel-TG lengths: [threadgroup_memory_length_offset, command_arguments_offset).
    let tg_start = 0x100u32;
    let mut layout = IcbCommandLayout {
        threadgroup_memory_length_offset: tg_start,
        command_arguments_offset: tg_start + entries * ICB_TG_MEMORY_STRIDE as u32,
        ..Default::default()
    };
    assert_eq!(
        icb_layout_kernel_tg_slot_count(&layout),
        entries,
        "kernel-TG count wrapped instead of reporting {entries}"
    );

    // Attribute strides: [attribute_stride_offset, earliest region after it).
    layout = IcbCommandLayout {
        attribute_stride_offset: tg_start,
        command_arguments_offset: tg_start + entries * ICB_ATTRIBUTE_STRIDE_ENTRY_SIZE as u32,
        ..Default::default()
    };
    assert_eq!(
        icb_layout_attribute_stride_slot_count(&layout),
        entries,
        "attribute-stride count wrapped instead of reporting {entries}"
    );

    // The shared rule underneath both, at each stride this crate uses.
    for stride in [
        ICB_TG_MEMORY_STRIDE,
        ICB_ATTRIBUTE_STRIDE_ENTRY_SIZE,
        ICB_BUFFER_BIND_STRIDE,
    ] {
        assert_eq!(
            icb_layout_table_len(tg_start, tg_start + entries * stride as u32, stride),
            entries,
            "stride {stride:#x} wrapped"
        );
        // An end at or below the start is an empty table, never a huge one from
        // a wrapped subtraction.
        assert_eq!(icb_layout_table_len(tg_start, tg_start, stride), 0);
        assert_eq!(icb_layout_table_len(tg_start, tg_start - 8, stride), 0);
    }
}
