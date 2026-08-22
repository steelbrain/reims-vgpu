//! Check the crate's views against Apple's serializer.
//!
//! This is the test that can find a *layout* error. The unit tests inside the
//! crate synthesize their buffers from the same constants the views read, so
//! they prove the view code is self-consistent and nothing more. Only these
//! fixtures — bytes Apple's serializer actually produced, checked against what
//! Metal was asked for — can say the layout is right.
//!
//! Fixtures are not committed; regenerate with `scripts/wire-oracle/wire-oracle.sh`.
//! With none present these tests are `ignored` rather than passing — the
//! decision is `build.rs`'s, so libtest prints one `ignored` line per test
//! instead of 34 green ones that measured nothing. See
//! `oracle/fixture_presence.rs`. `REIMS_WIRE_FIXTURES_REQUIRED=1` on hosts that
//! must have them (any Apple host, and CI) makes their absence fail the build.

use reims_vgpu_wire::manifest::{self, Coverage};
use reims_vgpu_wire::op::op;
use reims_vgpu_wire::ops::texture::{self, new_texture, NEW_TEXTURE_TOTAL_LEN, OPCODE_NEW_TEXTURE};
use serde_json::Value;

/// Serializer build these fixtures' layouts were derived against.
///
/// A different build is not automatically wrong, but it is a different
/// contract, and the layouts must be re-derived rather than assumed. Reported
/// rather than asserted, so a newer host runs the tests instead of refusing.
const DERIVED_AGAINST_BUNDLE_VERSION: &str = "64.4.7";

/// Apple's captured records.
///
/// Only reachable when `wire_fixtures` is set, so a failure to read here means
/// the capture was deleted between building this test and running it.
fn fixtures() -> Value {
    read_oracle_output("fixtures.json", "")
}

/// Read one of the oracle's outputs, or say what regenerates it.
///
/// `build.rs` already decided this file exists — every caller sits behind the
/// `cfg` that decision sets — so anything that goes wrong here is the capture
/// being removed mid-run, not the ordinary fixture-less checkout.
fn read_oracle_output(file: &str, regenerate_args: &str) -> Value {
    let dir = std::env::var("REIMS_WIRE_FIXTURES_DIR")
        .unwrap_or_else(|_| format!("{}/fixtures", env!("CARGO_MANIFEST_DIR")));
    let path = format!("{dir}/{file}");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "{path} was present when this test was built and is not now ({e}); \
             regenerate with scripts/wire-oracle/wire-oracle.sh{regenerate_args}"
        )
    });
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{path} is not valid JSON: {e}"))
}

fn unhex(s: &str) -> Vec<u8> {
    assert!(s.len().is_multiple_of(2), "hex buffer has an odd length");
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("hex digit"))
        .collect()
}

fn expect_u64(case: &Value, key: &str) -> u64 {
    case["expect"][key]
        .as_u64()
        .unwrap_or_else(|| panic!("case {} has no expect.{key}", case["name"]))
}

/// The one key of `candidates` this case carries.
///
/// Several selectors share one record shape, and each case names its own field
/// — so which key is present says which selector produced it. Exactly one must
/// match: none means the case checks nothing, and more than one means the
/// dispatch cannot tell which field is being asserted.
fn sole_key<'a>(case: &Value, candidates: &[&'a str]) -> &'a str {
    let present: Vec<&str> = candidates
        .iter()
        .copied()
        .filter(|k| !case["expect"][*k].is_null())
        .collect();
    assert_eq!(
        present.len(),
        1,
        "case {} carries {present:?} of {candidates:?}; expected exactly one",
        case["name"]
    );
    present[0]
}

fn expect_i64(case: &Value, key: &str) -> i64 {
    case["expect"][key]
        .as_i64()
        .unwrap_or_else(|| panic!("case {} has no expect.{key}", case["name"]))
}

/// The compact and wide opcodes a draw selector chooses between.
///
/// Selector-keyed rather than opcode-keyed on purpose: the point of the test
/// this feeds is that *Apple picked the one we predicted*, so it must not start
/// from the opcode Apple wrote.
fn draw_pair(selector: &str) -> Option<(u32, u32)> {
    use reims_vgpu_wire::ops::render as r;
    Some(match selector {
        "drawPrimitives:vertexStart:vertexCount:" => (r::OPCODE_DRAW, r::OPCODE_DRAW_WIDE),
        "drawPrimitives:vertexStart:vertexCount:instanceCount:" => {
            (r::OPCODE_DRAW_INSTANCED, r::OPCODE_DRAW_INSTANCED_WIDE)
        }
        "drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:" => (
            r::OPCODE_DRAW_INSTANCED_BASE,
            r::OPCODE_DRAW_INSTANCED_BASE_WIDE,
        ),
        "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:" => {
            (r::OPCODE_DRAW_INDEXED, r::OPCODE_DRAW_INDEXED_WIDE)
        }
        "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:\
         instanceCount:" => (
            r::OPCODE_DRAW_INDEXED_INSTANCED,
            r::OPCODE_DRAW_INDEXED_INSTANCED_WIDE,
        ),
        "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:\
         instanceCount:baseVertex:baseInstance:" => (
            r::OPCODE_DRAW_INDEXED_INSTANCED_BASE,
            r::OPCODE_DRAW_INDEXED_INSTANCED_BASE_WIDE,
        ),
        _ => return None,
    })
}

/// The record length the crate's constants claim for an opcode.
fn record_len_for(opcode: u32) -> Option<u32> {
    use reims_vgpu_wire::ops::render as r;
    Some(match opcode {
        r::OPCODE_DRAW => r::DRAW_TOTAL_LEN,
        r::OPCODE_DRAW_WIDE => r::DRAW_WIDE_TOTAL_LEN,
        r::OPCODE_DRAW_INSTANCED => r::DRAW_INSTANCED_TOTAL_LEN,
        r::OPCODE_DRAW_INSTANCED_WIDE => r::DRAW_INSTANCED_WIDE_TOTAL_LEN,
        r::OPCODE_DRAW_INSTANCED_BASE => r::DRAW_INSTANCED_BASE_TOTAL_LEN,
        r::OPCODE_DRAW_INSTANCED_BASE_WIDE => r::DRAW_INSTANCED_BASE_WIDE_TOTAL_LEN,
        r::OPCODE_DRAW_INDEXED => r::DRAW_INDEXED_TOTAL_LEN,
        r::OPCODE_DRAW_INDEXED_WIDE => r::DRAW_INDEXED_WIDE_TOTAL_LEN,
        r::OPCODE_DRAW_INDEXED_INSTANCED => r::DRAW_INDEXED_INSTANCED_TOTAL_LEN,
        r::OPCODE_DRAW_INDEXED_INSTANCED_WIDE => r::DRAW_INDEXED_INSTANCED_WIDE_TOTAL_LEN,
        r::OPCODE_DRAW_INDEXED_INSTANCED_BASE => r::DRAW_INDEXED_INSTANCED_BASE_TOTAL_LEN,
        r::OPCODE_DRAW_INDEXED_INSTANCED_BASE_WIDE => r::DRAW_INDEXED_INSTANCED_BASE_WIDE_TOTAL_LEN,
        r::OPCODE_SET_SCISSOR => r::SET_SCISSOR_TOTAL_LEN,
        r::OPCODE_SET_VIEWPORT => r::SET_VIEWPORT_TOTAL_LEN,
        r::OPCODE_SET_BLEND_COLOR => r::SET_BLEND_COLOR_TOTAL_LEN,
        r::OPCODE_SET_STENCIL_REFERENCE => r::SET_STENCIL_REFERENCE_TOTAL_LEN,
        r::OPCODE_SET_DEPTH_BIAS => r::SET_DEPTH_BIAS_TOTAL_LEN,
        r::OPCODE_SET_VISIBILITY_RESULT_MODE => r::SET_VISIBILITY_RESULT_MODE_TOTAL_LEN,
        r::OPCODE_DRAW_PATCHES => r::DRAW_PATCHES_TOTAL_LEN,
        r::OPCODE_DRAW_INDEXED_PATCHES => r::DRAW_INDEXED_PATCHES_TOTAL_LEN,
        r::OPCODE_DRAW_PATCHES_INDIRECT => r::DRAW_PATCHES_INDIRECT_TOTAL_LEN,
        r::OPCODE_DRAW_INDEXED_PATCHES_INDIRECT => r::DRAW_INDEXED_PATCHES_INDIRECT_TOTAL_LEN,
        r::OPCODE_SET_COLOR_STORE_ACTION => r::SET_COLOR_STORE_ACTION_TOTAL_LEN,
        r::OPCODE_SET_COLOR_STORE_ACTION_OPTIONS => r::SET_COLOR_STORE_ACTION_OPTIONS_TOTAL_LEN,
        r::OPCODE_SET_DEPTH_STORE_ACTION_OPTIONS | r::OPCODE_SET_STENCIL_STORE_ACTION_OPTIONS => {
            r::SET_STORE_ACTION_OPTIONS_TOTAL_LEN
        }
        r::OPCODE_SET_TESSELLATION_FACTOR_BUFFER => r::SET_TESSELLATION_FACTOR_BUFFER_TOTAL_LEN,
        r::OPCODE_DRAW_INDIRECT => r::DRAW_INDIRECT_TOTAL_LEN,
        r::OPCODE_DRAW_INDEXED_INDIRECT => r::DRAW_INDEXED_INDIRECT_TOTAL_LEN,
        r::OPCODE_EXECUTE_COMMANDS_INDIRECT => r::EXECUTE_COMMANDS_INDIRECT_TOTAL_LEN,
        r::OPCODE_EXECUTE_COMMANDS_RANGE => r::EXECUTE_COMMANDS_RANGE_TOTAL_LEN,
        r::OPCODE_MEMORY_BARRIER_SCOPE => r::MEMORY_BARRIER_SCOPE_TOTAL_LEN,
        r::OPCODE_TEXTURE_BARRIER => r::TEXTURE_BARRIER_TOTAL_LEN,
        op if r::is_mode_state(op) => r::SET_MODE_TOTAL_LEN,
        op if r::is_float_state(op) => r::SET_FLOAT_TOTAL_LEN,
        op if r::is_state_ref(op) => r::SET_STATE_TOTAL_LEN,
        op if r::is_buffer_offset(op) => r::SET_BUFFER_OFFSET_TOTAL_LEN,
        r::OPCODE_SET_VERTEX_BUFFER_OFFSET_STRIDE => r::SET_BUFFER_OFFSET_STRIDE_TOTAL_LEN,
        r::OPCODE_SET_VERTEX_AMPLIFICATION_MODE => r::SET_VERTEX_AMPLIFICATION_MODE_TOTAL_LEN,
        op if r::is_fence(op) => r::FENCE_TOTAL_LEN,
        // The pass descriptor and the two records it splits into under a
        // capability. Fixed length at every attachment count -- the record
        // always carries all eight colour slots, written or not.
        reims_vgpu_wire::ops::render_pass::OPCODE_RENDER_PASS => {
            reims_vgpu_wire::ops::render_pass::RENDER_PASS_TOTAL_LEN
        }
        reims_vgpu_wire::ops::render_pass::OPCODE_DEFAULT_RASTER_SAMPLE_COUNT => {
            reims_vgpu_wire::ops::render_pass::DEFAULT_RASTER_SAMPLE_COUNT_TOTAL_LEN
        }
        reims_vgpu_wire::ops::render_pass::OPCODE_RASTERIZATION_RATE_MAP => {
            reims_vgpu_wire::ops::render_pass::RASTERIZATION_RATE_MAP_TOTAL_LEN
        }
        reims_vgpu_wire::ops::render_pass::OPCODE_IMAGEBLOCK_SAMPLE_LENGTH
        | reims_vgpu_wire::ops::render_pass::OPCODE_THREADGROUP_MEMORY_LENGTH => {
            reims_vgpu_wire::ops::render_pass::TILE_MEMORY_TOTAL_LEN
        }
        reims_vgpu_wire::ops::render_pass::OPCODE_TILE_SIZE => {
            reims_vgpu_wire::ops::render_pass::TILE_SIZE_TOTAL_LEN
        }
        _ => return tile_record_len_for(opcode),
    })
}

/// The tile family's fixed-length records.
///
/// Separate from the render table above only because they live in their own
/// module; the `is_variable_length` split is the same one. Both region forms
/// share a length — `0xa2` allocates the four bytes `0xa3` writes and leaves
/// them alone, which the `PARTIALLY_WRITTEN` row for `0xa2` records.
fn tile_record_len_for(opcode: u32) -> Option<u32> {
    use reims_vgpu_wire::ops::tile as t;
    Some(match opcode {
        t::OPCODE_DISPATCH_THREADS_PER_TILE => t::DISPATCH_THREADS_PER_TILE_TOTAL_LEN,
        t::OPCODE_SET_TILE_BUFFER_OFFSET => t::SET_TILE_BUFFER_OFFSET_TOTAL_LEN,
        t::OPCODE_SET_TILE_THREADGROUP_MEMORY => t::SET_TILE_THREADGROUP_MEMORY_TOTAL_LEN,
        t::OPCODE_GET_TILE_DIMENSIONS => t::GET_TILE_DIMENSIONS_TOTAL_LEN,
        op if t::is_dispatch_threads_per_tile_in_region(op) => {
            t::DISPATCH_THREADS_PER_TILE_IN_REGION_TOTAL_LEN
        }
        _ => return None,
    })
}

/// Whether this record's length is head-plus-`count`-entries rather than fixed.
///
/// These have no single constant to check against. What replaces it is the
/// `entries.len() == count` assertion in the arm that reads them, which the
/// view's own bounds check has already had to satisfy.
fn is_variable_length(opcode: u32) -> bool {
    use reims_vgpu_wire::ops::render as r;
    r::is_ref_bind(opcode)
        || r::is_buffer_bind(opcode)
        || r::is_sampler_lod_bind(opcode)
        || r::is_buffer_stride_bind(opcode)
        || opcode == r::OPCODE_SET_VERTEX_AMPLIFICATION_COUNT
        || opcode == r::OPCODE_USE_RESOURCE
        || opcode == r::OPCODE_USE_HEAP
        || opcode == r::OPCODE_MEMORY_BARRIER_RESOURCES
        || opcode == r::OPCODE_SET_SCISSOR_RECTS
        || opcode == r::OPCODE_SET_VIEWPORTS
        || reims_vgpu_wire::ops::tile::is_tile_bind(opcode)
        // `0x0c` is two records at two lengths, so there is no one constant to
        // check it against; the arm that reads it asserts which it is.
        || opcode == r::OPCODE_DRAW_PATCHES_WIDE
        // Head plus `count` sample positions.
        || opcode == reims_vgpu_wire::ops::render_pass::OPCODE_SAMPLE_POSITIONS
}

/// Whether the arguments this case passed should have made the serializer
/// choose the wide encoding.
///
/// `base_vertex` is deliberately absent: it is the one draw argument that does
/// not widen a record. It is truncated to 16 bits instead, which
/// `render_draw_indexed_base_vertex_below_i16` shows and which adding it here
/// would hide by predicting a wide record Apple never emits.
fn expects_wide_encoding(case: &Value) -> bool {
    const WIDENS: &[&str] = &[
        "vertex_start",
        "vertex_count",
        "instance_count",
        "base_instance",
        "index_count",
        "index_buffer_offset",
    ];
    WIDENS
        .iter()
        .any(|k| case["expect"][k].as_u64().unwrap_or(0) > 0xffff)
}

#[test]
#[cfg_attr(not(wire_fixtures), ignore = "run scripts/wire-oracle/wire-oracle.sh")]
fn every_texture_fixture_reads_back_what_metal_was_asked_for() {
    let root = fixtures();

    let got = root["provenance"]["bundle_version"].as_str().unwrap_or("?");
    if got != DERIVED_AGAINST_BUNDLE_VERSION {
        eprintln!(
            "NOTE: fixtures came from serializer {got}, layouts were derived against \
             {DERIVED_AGAINST_BUNDLE_VERSION}. A failure below may be a contract \
             change rather than a bug."
        );
    }

    let cases = root["cases"].as_array().expect("cases array");
    assert!(!cases.is_empty(), "fixtures.json carries no cases");

    let mut checked = 0usize;
    for case in cases {
        let name = case["name"].as_str().expect("case name");
        if case["selector"] != "newTextureWithDescriptor:allocator:" {
            continue;
        }
        let bytes = unhex(case["buffer"].as_str().expect("buffer hex"));

        // The allocator's request is the operation's length; the header must
        // agree with it, or `length` means something other than we think.
        let allocated = case["allocated_len"].as_u64().expect("allocated_len");
        assert_eq!(
            bytes.len() as u64,
            allocated,
            "{name}: captured {} bytes for an allocation of {allocated}",
            bytes.len()
        );

        let o = op(&bytes, 0).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            o.length() as u64,
            allocated,
            "{name}: length disagrees with the allocator's request"
        );

        // One selector, two records. Under `-setSupportsSwizzledTextures:` this
        // selector emits the 40-byte descriptor at its own opcode, so the
        // dispatch here is on the opcode rather than on the case name — a case
        // that landed on the wrong form would then fail on a field rather than
        // be waved through.
        if o.opcode() == texture::OPCODE_NEW_TEXTURE_WIDE {
            assert_eq!(
                o.length(),
                texture::NEW_TEXTURE_WIDE_TOTAL_LEN,
                "{name}: wide record length drifted"
            );
            let w = texture::new_texture_wide(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
            let d = &w.desc;
            assert_eq!(
                d.texture_type() as u64,
                expect_u64(case, "texture_type"),
                "{name}: texture_type"
            );
            assert_eq!(
                d.usage.get() as u64,
                expect_u64(case, "usage"),
                "{name}: usage"
            );
            assert_eq!(
                d.pixel_format.get() as u64,
                expect_u64(case, "pixel_format"),
                "{name}: pixel_format"
            );
            assert_eq!(
                d.width.get() as u64,
                expect_u64(case, "width"),
                "{name}: width"
            );
            assert_eq!(
                d.height.get() as u64,
                expect_u64(case, "height"),
                "{name}: height"
            );
            assert_eq!(
                d.depth.get() as u64,
                expect_u64(case, "depth"),
                "{name}: depth"
            );
            assert_eq!(
                d.mipmap_level_count.get() as u64,
                expect_u64(case, "mipmap_level_count"),
                "{name}: mipmap_level_count"
            );
            assert_eq!(
                d.sample_count.get() as u64,
                expect_u64(case, "sample_count"),
                "{name}: sample_count"
            );
            assert_eq!(
                d.array_length.get() as u64,
                expect_u64(case, "array_length"),
                "{name}: array_length"
            );
            assert_eq!(
                d.storage_mode() as u64,
                expect_u64(case, "storage_mode"),
                "{name}: storage_mode"
            );
            assert_eq!(
                d.allow_gpu_optimized_contents() as u64,
                expect_u64(case, "allow_gpu_optimized_contents"),
                "{name}: allow_gpu_optimized_contents"
            );
            assert_eq!(
                d.cpu_cache_mode() as u64,
                expect_u64(case, "cpu_cache_mode"),
                "{name}: cpu_cache_mode"
            );
            assert_eq!(
                d.hazard_tracking_mode() as u64,
                expect_u64(case, "hazard_tracking_mode"),
                "{name}: hazard_tracking_mode"
            );
            // The four bytes only this form has. The permuted case is what makes
            // this an assertion about the guest's values rather than about a
            // constant: the default channels are (2,3,4,5) and it sends
            // (5,0,1,2).
            assert_eq!(
                d.swizzle_red as u64,
                expect_u64(case, "swizzle_red"),
                "{name}: swizzle_red"
            );
            assert_eq!(
                d.swizzle_green as u64,
                expect_u64(case, "swizzle_green"),
                "{name}: swizzle_green"
            );
            assert_eq!(
                d.swizzle_blue as u64,
                expect_u64(case, "swizzle_blue"),
                "{name}: swizzle_blue"
            );
            assert_eq!(
                d.swizzle_alpha as u64,
                expect_u64(case, "swizzle_alpha"),
                "{name}: swizzle_alpha"
            );
            // The fortieth byte is declared and never written, so it must be the
            // arena fill. A serializer that started writing it would mean a
            // field this crate does not name.
            let mask = unhex(case["written_mask"].as_str().expect("written_mask"));
            assert_eq!(
                *mask.last().expect("non-empty"),
                0,
                "{name}: the wide descriptor's last byte is now written; it is \
                 declared but has never carried a field"
            );
            checked += 1;
            continue;
        }

        assert_eq!(o.opcode(), OPCODE_NEW_TEXTURE, "{name}: opcode drifted");
        assert_eq!(
            o.length(),
            NEW_TEXTURE_TOTAL_LEN,
            "{name}: record length drifted"
        );

        let t = new_texture(&o).unwrap_or_else(|e| panic!("{name}: {e}"));

        assert_eq!(
            t.desc.texture_type() as u64,
            expect_u64(case, "texture_type"),
            "{name}: texture_type"
        );
        assert_eq!(
            t.desc.usage() as u64,
            expect_u64(case, "usage"),
            "{name}: usage"
        );
        assert_eq!(
            t.desc.pixel_format() as u64,
            expect_u64(case, "pixel_format"),
            "{name}: pixel_format"
        );
        assert_eq!(
            t.desc.width.get() as u64,
            expect_u64(case, "width"),
            "{name}: width"
        );
        assert_eq!(
            t.desc.height.get() as u64,
            expect_u64(case, "height"),
            "{name}: height"
        );
        assert_eq!(
            t.desc.depth.get() as u64,
            expect_u64(case, "depth"),
            "{name}: depth"
        );
        assert_eq!(
            t.desc.mipmap_level_count.get() as u64,
            expect_u64(case, "mipmap_level_count"),
            "{name}: mipmap_level_count"
        );
        assert_eq!(
            t.desc.sample_count.get() as u64,
            expect_u64(case, "sample_count"),
            "{name}: sample_count"
        );
        assert_eq!(
            t.desc.array_length.get() as u64,
            expect_u64(case, "array_length"),
            "{name}: array_length"
        );
        assert_eq!(
            t.desc.storage_mode() as u64,
            expect_u64(case, "storage_mode"),
            "{name}: storage_mode"
        );
        assert_eq!(
            t.desc.allow_gpu_optimized_contents() as u64,
            expect_u64(case, "allow_gpu_optimized_contents"),
            "{name}: allow_gpu_optimized_contents"
        );
        assert_eq!(
            t.desc.cpu_cache_mode() as u64,
            expect_u64(case, "cpu_cache_mode"),
            "{name}: cpu_cache_mode"
        );
        assert_eq!(
            t.desc.hazard_tracking_mode() as u64,
            expect_u64(case, "hazard_tracking_mode"),
            "{name}: hazard_tracking_mode"
        );

        checked += 1;
    }
    assert!(checked > 0, "no texture cases in fixtures.json");
    eprintln!("checked {checked} texture fixtures against Apple's serializer");
}

/// The heap sizing query is the texture creation record without its ref.
///
/// Both records embed one [`TextureDescriptorBody`], and the claim that they
/// embed the *same* one is what this checks — not by reading fields out of each
/// and comparing numbers, which two wrong-but-consistent readers would pass,
/// but by comparing the raw descriptor bytes and the measured written masks
/// directly. `0x01` writes `[ref][body]` and `0x16` writes `[body]`, so the
/// second is the first shifted four bytes.
///
/// Written before this record had a fixture, `ops::texture`'s doc already named
/// `reims-vgpu`'s heap size-and-align query as a third reader of that struct.
/// This is what stops the three drifting.
#[test]
#[cfg_attr(not(wire_fixtures), ignore = "run scripts/wire-oracle/wire-oracle.sh")]
fn the_heap_sizing_query_carries_the_texture_creation_record_minus_its_ref() {
    use reims_vgpu_wire::ops::texture;

    let root = fixtures();
    let cases = root["cases"].as_array().expect("cases array");

    let find = |name: &str| {
        cases
            .iter()
            .find(|c| c["name"] == name)
            .unwrap_or_else(|| panic!("no fixture {name}"))
    };
    let baseline = find("texture_baseline");
    let heap = find("serializer_heap_texture_size_and_align");

    let base_bytes = unhex(baseline["buffer"].as_str().expect("buffer"));
    let heap_bytes = unhex(heap["buffer"].as_str().expect("buffer"));
    let base_mask = unhex(baseline["written_mask"].as_str().expect("mask"));
    let heap_mask = unhex(heap["written_mask"].as_str().expect("mask"));

    let o = op(&heap_bytes, 0).expect("heap sizing record");
    assert_eq!(o.opcode(), texture::OPCODE_HEAP_TEXTURE_SIZE_AND_ALIGN);
    assert_eq!(o.length(), texture::HEAP_TEXTURE_SIZE_AND_ALIGN_TOTAL_LEN);
    assert_eq!(
        texture::NEW_TEXTURE_TOTAL_LEN - texture::HEAP_TEXTURE_SIZE_AND_ALIGN_TOTAL_LEN,
        4,
        "the two records differ by exactly the object ref"
    );

    // `0x01`: header, ref, then the body. `0x16`: header, then the body.
    let base_desc = reims_vgpu_wire::OP_HEADER_LEN + 4;
    let heap_desc = reims_vgpu_wire::OP_HEADER_LEN;
    let n = texture::TEXTURE_DESCRIPTOR_LEN;
    assert_eq!(
        &base_bytes[base_desc..base_desc + n],
        &heap_bytes[heap_desc..heap_desc + n],
        "the two records do not embed the same descriptor bytes"
    );
    assert_eq!(
        &base_mask[base_desc..base_desc + n],
        &heap_mask[heap_desc..heap_desc + n],
        "the serializer wrote different bits of the descriptor in the two records"
    );

    // And the view reads it, so the shared declaration is exercised from both
    // records rather than only from the creation one.
    let d = texture::heap_texture_size_and_align(&o).expect("descriptor body");
    let base_op = op(&base_bytes, 0).expect("texture record");
    let b = texture::new_texture(&base_op).expect("creation body");
    assert_eq!(d.width.get(), b.desc.width.get());
    assert_eq!(d.height.get(), b.desc.height.get());
    assert_eq!(d.packed.get(), b.desc.packed.get());

    eprintln!("heap sizing query and texture creation share one descriptor, {n} bytes");
}

#[test]
#[cfg_attr(not(wire_fixtures), ignore = "run scripts/wire-oracle/wire-oracle.sh")]
fn the_texture_record_carries_no_compression_type() {
    // A measured absence, and the reason it is a test rather than a comment: a
    // later serializer that starts carrying `compressionType` would otherwise
    // add a field nothing decodes, silently. The two cases differ only in that
    // property, so any byte that moves is the property moving.
    let root = fixtures();
    let mut baseline = None;
    let mut lossy = None;
    for case in root["cases"].as_array().expect("cases array") {
        match case["name"].as_str() {
            Some("texture_baseline") => baseline = Some(case),
            Some("texture_compression_lossy") => lossy = Some(case),
            _ => {}
        }
    }
    let (baseline, lossy) = (
        baseline.expect("texture_baseline fixture"),
        lossy.expect("texture_compression_lossy fixture"),
    );
    assert_ne!(
        expect_u64(baseline, "compression_type"),
        expect_u64(lossy, "compression_type"),
        "the two cases were asked for the same compression type, so this proves nothing"
    );

    let a = unhex(baseline["buffer"].as_str().expect("buffer hex"));
    let b = unhex(lossy["buffer"].as_str().expect("buffer hex"));
    assert_eq!(a.len(), b.len(), "the two records are different lengths");

    // The object ref is the one byte that must differ: each case allocates its
    // own. Everything else moving is the finding.
    let moved: Vec<usize> = (0..a.len()).filter(|i| a[*i] != b[*i]).collect();
    let object_ref = 8..12;
    assert!(
        moved.iter().all(|i| object_ref.contains(i)),
        "compressionType moved bytes {moved:?}; it reaches the wire after all, and \
         `ops::texture` says it does not"
    );
}

#[test]
#[cfg_attr(not(wire_fixtures), ignore = "run scripts/wire-oracle/wire-oracle.sh")]
fn texture_private_fields_read_back_the_properties_that_moved_them() {
    let root = fixtures();
    let mut narrow = 0;
    let mut wide = 0;
    for case in root["cases"].as_array().expect("cases array") {
        if case["selector"] != "newTextureWithDescriptor:allocator:" {
            continue;
        }
        let Some(framebuffer_only) = case["expect"]["framebuffer_only"].as_u64() else {
            continue;
        };
        let is_drawable = expect_u64(case, "is_drawable");
        let bytes = unhex(case["buffer"].as_str().expect("buffer hex"));
        let o = op(&bytes, 0).expect("well formed");
        let protection_options = expect_u64(case, "protection_options");
        let (got_framebuffer, got_drawable, got_write_swizzle, got_protection) =
            if o.opcode() == texture::OPCODE_NEW_TEXTURE_WIDE {
                wide += 1;
                let d = &texture::new_texture_wide(&o).expect("fits").desc;
                (
                    d.framebuffer_only(),
                    d.is_drawable(),
                    Some(d.write_swizzle_enabled()),
                    d.protection_options.get(),
                )
            } else {
                narrow += 1;
                let d = &new_texture(&o).expect("fits").desc;
                (
                    d.framebuffer_only(),
                    d.is_drawable(),
                    None,
                    d.protection_options.get(),
                )
            };
        assert_eq!(got_framebuffer as u64, framebuffer_only, "{}", case["name"]);
        assert_eq!(got_drawable as u64, is_drawable, "{}", case["name"]);
        assert_eq!(got_protection, protection_options, "{}", case["name"]);
        if let Some(got) = got_write_swizzle {
            assert_eq!(
                got as u64,
                expect_u64(case, "write_swizzle_enabled"),
                "{}",
                case["name"]
            );
        }
    }
    assert!(narrow > 0, "no narrow texture cases carried private flags");
    assert!(wide > 0, "no wide texture cases carried private flags");
}

/// Bits the serializer leaves alone, per record, measured rather than eyeballed.
///
/// The oracle captures every case twice under complementary arena fills and
/// records the bits that agreed. A bit that disagreed was never written, and on
/// a real wire it is whatever the guest's ring last held — so a view that reads
/// one is reading noise, and the crate's job is to know exactly where they are.
///
/// Each row is `(class, record, length, [(byte offset, written-bit mask)])`.
/// Only the bytes that are *not* fully written appear; a key absent from this
/// table must be written end to end, and the test says so if it is not. That is
/// what makes the table exhaustive without listing 146 all-ones rows.
///
/// `record` is the opcode, except for the segment header, which has none — its
/// first word is a length. Keying that on its leading bytes would file it with
/// `OPCODE_DRAW_WIDE`, which really is opcode zero, and the two have different
/// lengths, so the mistake shows up as an unmergeable union rather than as a
/// wrong answer. See [`record_key`].
///
/// The length is part of the key because a variable-length record's unwritten
/// tail moves with it: `useHeaps:count:` leaves two bytes past its last heap
/// ref, which is offset 18 in the one-heap form and 22 in the two-heap form.
/// Unioning those would claim both are written in both.
///
/// Sorted by class then key, and the masks are the union across every fixture
/// of that record: a bit set here was written by at least one case.
type UnwrittenBytes = &'static [(usize, u8)];
const PARTIALLY_WRITTEN: &[(&str, &str, usize, UnwrittenBytes)] = &[
    // The render pass descriptor's twelve two-byte holes, one per attachment
    // slot: the top half of every `store_action_options` word, plus the top half
    // of the depth and stencil resolve filters. Each of those three fields
    // occupies four bytes and the serializer writes sixteen bits, so a `u32`
    // read of any of them takes the guest's ring in its top half. See
    // `ops::render_pass`.
    (
        "PGSerializerRenderCommandEncoder",
        "opcode 0x001a",
        592,
        &[
            (34, 0x00),
            (35, 0x00),
            (46, 0x00),
            (47, 0x00),
            (74, 0x00),
            (75, 0x00),
            (82, 0x00),
            (83, 0x00),
            (110, 0x00),
            (111, 0x00),
            (170, 0x00),
            (171, 0x00),
            (230, 0x00),
            (231, 0x00),
            (290, 0x00),
            (291, 0x00),
            (350, 0x00),
            (351, 0x00),
            (410, 0x00),
            (411, 0x00),
            (470, 0x00),
            (471, 0x00),
            (530, 0x00),
            (531, 0x00),
        ],
    ),
    // The texture body's `packed` bit 7. See `ops::texture`.
    ("PGSerializer", "opcode 0x0001", 44, &[(12, 0x7f)]),
    // The same descriptor at the same bit, four bytes earlier because the heap
    // sizing query has no object ref ahead of it. Two records, one struct, one
    // hole — which is the check
    // `the_heap_sizing_query_carries_the_texture_creation_record_minus_its_ref`
    // makes directly rather than by inference.
    ("PGSerializer", "opcode 0x0016", 40, &[(8, 0x7f)]),
    // The wide descriptor's fortieth byte, and note what is *not* here: the
    // packed bit 7 above. The wide form writes that flag and the narrow one does
    // not, so the same selector's two records have holes in different places —
    // which is the whole reason this table is keyed by opcode and length rather
    // than by selector. See `ops::texture::WideTextureDescriptorBody`.
    ("PGSerializer", "opcode 0x0034", 52, &[(51, 0x00)]),
    // The same wide descriptor's tail in the three records that embed it, at
    // whatever offset each one puts it. The IOSurface form then carries a
    // two-byte plane and a one-byte rotation; its final byte is unwritten.
    ("PGSerializer", "opcode 0x0037", 72, &[(71, 0x00)]),
    (
        "PGSerializer",
        "opcode 0x0038",
        68,
        &[(55, 0x00), (56, 0x01), (57, 0x00), (58, 0x00), (59, 0x00)],
    ),
    (
        "PGSerializer",
        "opcode 0x0039",
        56,
        &[(51, 0x00), (55, 0x00)],
    ),
    // The fifth, and the one where the hole is not near the end: this is a
    // query, so the reply pair follows the descriptor and the unwritten byte
    // lands twelve bytes from the record's tail.
    (
        "PGSerializerInfoCommandEncoder",
        "opcode 0x01d5",
        60,
        &[(47, 0x00)],
    ),
    // The rate map's last sixteen bytes, at each of the three lengths captured.
    // The tail is the same size at every layer and sample count, which is what
    // makes it a property of the record rather than of a layer -- see
    // `ops::rate_map::UNWRITTEN_TAIL_LEN`, and note the length in the key: this
    // record is variable length, so one row per length is the rule.
    (
        "PGSerializer",
        "opcode 0x0032",
        64,
        &[
            (48, 0x00),
            (49, 0x00),
            (50, 0x00),
            (51, 0x00),
            (52, 0x00),
            (53, 0x00),
            (54, 0x00),
            (55, 0x00),
            (56, 0x00),
            (57, 0x00),
            (58, 0x00),
            (59, 0x00),
            (60, 0x00),
            (61, 0x00),
            (62, 0x00),
            (63, 0x00),
        ],
    ),
    (
        "PGSerializer",
        "opcode 0x0032",
        76,
        &[
            (60, 0x00),
            (61, 0x00),
            (62, 0x00),
            (63, 0x00),
            (64, 0x00),
            (65, 0x00),
            (66, 0x00),
            (67, 0x00),
            (68, 0x00),
            (69, 0x00),
            (70, 0x00),
            (71, 0x00),
            (72, 0x00),
            (73, 0x00),
            (74, 0x00),
            (75, 0x00),
        ],
    ),
    (
        "PGSerializer",
        "opcode 0x0032",
        96,
        &[
            (80, 0x00),
            (81, 0x00),
            (82, 0x00),
            (83, 0x00),
            (84, 0x00),
            (85, 0x00),
            (86, 0x00),
            (87, 0x00),
            (88, 0x00),
            (89, 0x00),
            (90, 0x00),
            (91, 0x00),
            (92, 0x00),
            (93, 0x00),
            (94, 0x00),
            (95, 0x00),
        ],
    ),
    // The ICB creation: a whole byte and one bit inside the descriptor half,
    // plus two bytes of tail. The holes sit *inside* the record rather than
    // only at its end, which is the shape that has cost this project two bugs.
    (
        "PGSerializer",
        "opcode 0x0036",
        88,
        &[(21, 0x00), (23, 0x7f), (86, 0x00), (87, 0x00)],
    ),
    // The sampler's argument-buffer nibble, then eight bytes of tail the
    // serializer allocates and never touches.
    (
        "PGSerializer",
        "opcode 0x0003",
        36,
        &[
            (16, 0x0f),
            (17, 0x00),
            (18, 0x00),
            (19, 0x00),
            (28, 0x00),
            (29, 0x00),
            (30, 0x00),
            (31, 0x00),
            (32, 0x00),
            (33, 0x00),
            (34, 0x00),
            (35, 0x00),
        ],
    ),
    // The depth-stencil state byte: six bits of a four-byte slot.
    (
        "PGSerializer",
        "opcode 0x0004",
        40,
        &[(12, 0x3f), (13, 0x00), (14, 0x00), (15, 0x00)],
    ),
    // The format-only texture view allocates room for the texture type its two
    // wider siblings carry, and never writes it.
    (
        "PGSerializer",
        "opcode 0x0007",
        20,
        &[(18, 0x00), (19, 0x00)],
    ),
    // Both backed textures embed the plain descriptor, so `packed` bit 7 comes
    // with it; the IOSurface form's plane index is a `u16` in a wider slot.
    ("PGSerializer", "opcode 0x0009", 64, &[(32, 0x7f)]),
    (
        "PGSerializer",
        "opcode 0x000c",
        48,
        &[(12, 0x7f), (46, 0x00), (47, 0x00)],
    ),
    // The heap texture: the embedded body's `packed` bit 7 again, and
    // `use_offset`, which is one bit rather than the word it sits in.
    (
        "PGSerializer",
        "opcode 0x0015",
        60,
        &[(16, 0x7f), (48, 0x01), (49, 0x00), (50, 0x00), (51, 0x00)],
    ),
    // Two-byte tails on records whose bodies stop short of their allocation.
    (
        "PGSerializerBlitCommandEncoder",
        "opcode 0x012e",
        96,
        &[(94, 0x00), (95, 0x00)],
    ),
    (
        "PGSerializerBlitCommandEncoder",
        "opcode 0x0132",
        32,
        &[(29, 0x00), (30, 0x00), (31, 0x00)],
    ),
    // The colour fill's format word is sixteen bits and its record is two bytes
    // longer. Note what is *absent*: `0x013f`, the pattern fill, has the same
    // length as `0x0132` above and no row here, because it writes all four
    // bytes of the word `0x0132` writes one of. That difference is the whole
    // distinction between the two selectors, and it is measured rather than
    // read off their names.
    (
        "PGSerializerBlitCommandEncoder",
        "opcode 0x0141",
        100,
        &[(98, 0x00), (99, 0x00)],
    ),
    (
        "PGSerializerComputeCommandEncoder",
        "opcode 0x00d7",
        12,
        &[(10, 0x00), (11, 0x00)],
    ),
    (
        "PGSerializerRenderCommandEncoder",
        "opcode 0x0002",
        36,
        &[(34, 0x00), (35, 0x00)],
    ),
    (
        "PGSerializerRenderCommandEncoder",
        "opcode 0x0004",
        44,
        &[(42, 0x00), (43, 0x00)],
    ),
    (
        "PGSerializerRenderCommandEncoder",
        "opcode 0x0005",
        20,
        &[(18, 0x00), (19, 0x00)],
    ),
    (
        "PGSerializerRenderCommandEncoder",
        "opcode 0x0009",
        24,
        &[(22, 0x00), (23, 0x00)],
    ),
    (
        "PGSerializerRenderCommandEncoder",
        "opcode 0x000b",
        28,
        &[(26, 0x00), (27, 0x00)],
    ),
    (
        "PGSerializerRenderCommandEncoder",
        "opcode 0x0010",
        24,
        &[(22, 0x00), (23, 0x00)],
    ),
    // `useHeaps:count:` is variable length, and its unwritten tail moves with
    // the record: two bytes past the last heap ref, wherever that lands. This
    // is why the key carries a length -- one row per length, not one per
    // opcode.
    (
        "PGSerializerRenderCommandEncoder",
        "opcode 0x001b",
        20,
        &[(18, 0x00), (19, 0x00)],
    ),
    (
        "PGSerializerRenderCommandEncoder",
        "opcode 0x001b",
        24,
        &[(22, 0x00), (23, 0x00)],
    ),
    // Five of the six patch draws allocate two bytes past `control_points` and
    // write neither. Only the compact plain form (`0x0d`, 24 bytes) is written
    // end to end, and `0x0c` needs one row per length because its two records
    // are different sizes.
    (
        "PGSerializerRenderCommandEncoder",
        "opcode 0x000c",
        56,
        &[(54, 0x00), (55, 0x00)],
    ),
    (
        "PGSerializerRenderCommandEncoder",
        "opcode 0x000c",
        68,
        &[(66, 0x00), (67, 0x00)],
    ),
    (
        "PGSerializerRenderCommandEncoder",
        "opcode 0x000f",
        32,
        &[(30, 0x00), (31, 0x00)],
    ),
    (
        "PGSerializerRenderCommandEncoder",
        "opcode 0x0012",
        36,
        &[(34, 0x00), (35, 0x00)],
    ),
    (
        "PGSerializerRenderCommandEncoder",
        "opcode 0x0013",
        48,
        &[(46, 0x00), (47, 0x00)],
    ),
    // `dispatchThreadsPerTile:inRegion:` allocates the four bytes its
    // `withRenderTargetArrayIndex:` sibling writes, and writes none of them.
    // The two records are otherwise identical, so this row is the whole
    // difference between them -- see `ops::tile`.
    (
        "PGSerializerRenderCommandEncoder",
        "opcode 0x00a2",
        84,
        &[(80, 0x00), (81, 0x00), (82, 0x00), (83, 0x00)],
    ),
    // The texture body's `packed` bit 7 for the third time, in the info
    // encoder's descriptor query. Same struct, same hole, a third offset.
    (
        "PGSerializerInfoCommandEncoder",
        "opcode 0x01c3",
        52,
        &[(8, 0x7f)],
    ),
    // The segment header's eighth byte, which neither `-beginSegment:` nor
    // `-endEncoding` writes. Every encoder class allocates one.
    (
        "PGSerializerBlitCommandEncoder",
        "segment header",
        8,
        &[(7, 0x00)],
    ),
    (
        "PGSerializerComputeCommandEncoder",
        "segment header",
        8,
        &[(7, 0x00)],
    ),
    (
        "PGSerializerInfoCommandEncoder",
        "segment header",
        8,
        &[(7, 0x00)],
    ),
    (
        "PGSerializerRenderCommandEncoder",
        "segment header",
        8,
        &[(7, 0x00)],
    ),
];

/// Which record a fixture's bytes are, for grouping written masks.
///
/// Not simply the leading `u32`. `-beginSegment:protectionOptions:` writes
/// framing rather than an operation, and its first word is a length that reads
/// zero until `-endEncoding` — while `OPCODE_DRAW_WIDE` really is opcode zero.
/// Keying both on their leading bytes files a 28-byte draw with an 8-byte
/// header.
///
/// The case *name* is needed as well as the selector, for the one record that
/// is neither: see the envelope note below.
fn record_key(selector: &str, name: &str, bytes: &[u8]) -> String {
    if selector == "beginSegment:protectionOptions:" {
        // The protection-options envelope's middle record is the same length as
        // a header and is not one — it is eight fully-written bytes of
        // `protectionOptions:`. Filing it as a header would union a
        // fully-written record with one that leaves its eighth byte alone, and
        // the hole would vanish from the table while a real guest still had it.
        if name.ends_with("_1") {
            return "protection options envelope".to_string();
        }
        return "segment header".to_string();
    }
    format!(
        "opcode {:#06x}",
        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    )
}

/// Render a hole list the way a reader needs it: decimal offset, hex mask.
///
/// A single `{:x?}` on the pair prints both in hex, so byte 18 reads as "12"
/// and sends whoever is adding the row to the wrong offset. That happened.
fn describe_holes(holes: &[(usize, u8)]) -> String {
    let each: Vec<String> = holes
        .iter()
        .map(|(offset, mask)| format!("(byte {offset}, written bits {mask:#04x})"))
        .collect();
    format!("[{}]", each.join(", "))
}

/// Every capability flag's untouched value is recorded, and they are all off.
///
/// This is the caveat on every `EMITS_NO_OPERATION` row in the manifest. A
/// selector family gated on a capability emits *nothing* with its flag off, and
/// the serializer starts with all sixteen off — so "ran and wrote nothing" is,
/// by default, a measurement of this harness rather than of Apple.
///
/// It is not hypothetical. The four Metal 3.1 attribute-stride vertex binds were
/// silent on their first capture and would have become four false rows; driven
/// through `withCapability` they emit `0xa5`/`0xa6`, and `reims-vgpu` was
/// refusing both as opcodes Apple does not produce.
///
/// The all-off assertion is deliberately strict. If a future serializer ships
/// with one defaulted on, that changes which silent rows are trustworthy and
/// which are not, and this test is where that should be noticed — not by a
/// reader assuming the map still looks the way this comment says.
/// The second texture-descriptor layout never reaches the wire.
///
/// A spent experiment, pinned so it is not re-run.
/// `-serializeTextureDescriptor2:textureDescriptor:` declares a **different**
/// struct from its unsuffixed sibling — `b4b1b1b1b1b16IIIISSSSQCCCC` against
/// `b4b1b1b1b1b8b16IIISSSSQ`. The `b8` usage leaves the packed word, a fourth
/// `I` appears and four `C` bytes trail: 40 bytes against 32. Read cold, that
/// looks like a live hazard — a guest negotiating `supportsTextureDescriptor2`
/// would send creation records this crate's 32-byte
/// [`reims_vgpu_wire::ops::texture::TextureDescriptorBody`] mis-reads field for
/// field, taking `width` where `usage` is.
///
/// It does not happen. Driven with the capability forced on, the creation record
/// is **byte-identical** apart from the object ref the allocator handed out. The
/// two `serializeTextureDescriptor*` selectors write into a caller-supplied
/// buffer — their first argument is a struct *pointer*, and their return is
/// `void` — so the second layout is a host-side serialization helper and the
/// wire is unaffected.
///
/// Worth keeping from it: the *unsuffixed* encoding is
/// `TextureDescriptorBody`'s layout exactly, field for field, which is an
/// independent derivation of a struct that was arrived at by perturbation.
#[test]
#[cfg_attr(not(wire_fixtures), ignore = "run scripts/wire-oracle/wire-oracle.sh")]
fn the_second_texture_descriptor_layout_does_not_reach_the_wire() {
    let root = fixtures();

    let find = |name: &str| -> Vec<u8> {
        let case = root["cases"]
            .as_array()
            .expect("cases")
            .iter()
            .find(|c| c["name"].as_str() == Some(name))
            .unwrap_or_else(|| panic!("no case {name}; regenerate the fixtures"));
        unhex(case["buffer"].as_str().expect("buffer hex"))
    };
    let plain = find("texture_baseline");
    let with2 = find("texture_baseline_descriptor2");

    assert_eq!(
        plain.len(),
        with2.len(),
        "the creation record changed length under supportsTextureDescriptor2, so the \
         40-byte descriptor does reach the wire and every reader of the 32-byte body \
         is wrong for such a guest"
    );
    // Every byte but the object ref, which is `[opcode][length][object_ref]` at
    // payload +0 and is whatever the allocator handed out next.
    const REF: std::ops::Range<usize> = 8..12;
    for (i, (a, b)) in plain.iter().zip(&with2).enumerate() {
        if REF.contains(&i) {
            continue;
        }
        assert_eq!(
            a, b,
            "byte {i} differs under supportsTextureDescriptor2: {plain:02x?} vs {with2:02x?}"
        );
    }
    assert_ne!(
        plain[REF], with2[REF],
        "the two cases were handed the same object ref, so this comparison is \
         between one record and itself"
    );
}

#[test]
#[cfg_attr(not(wire_fixtures), ignore = "run scripts/wire-oracle/wire-oracle.sh")]
fn the_capability_defaults_are_recorded_and_every_one_is_off() {
    let root = fixtures();

    let defaults = root["capability_defaults"]
        .as_object()
        .expect("capability_defaults; regenerate the fixtures");
    assert_eq!(
        defaults.len(),
        16,
        "the serializer's capability pairs changed count: {defaults:?}"
    );

    let on: Vec<&String> = defaults
        .iter()
        .filter(|(_, v)| v.as_bool() == Some(true))
        .map(|(k, _)| k)
        .collect();
    assert!(
        on.is_empty(),
        "these capabilities now default on: {on:?}. That is a real change in what \
         a `silent` capture means — families gated on them were emitting nothing \
         before and are not now. Re-read the EMITS_NO_OPERATION rows for them \
         before relaxing this assertion"
    );
    eprintln!(
        "all {} serializer capabilities default off; every `silent` row rests on that",
        defaults.len()
    );
}

#[test]
#[cfg_attr(not(wire_fixtures), ignore = "run scripts/wire-oracle/wire-oracle.sh")]
fn every_fixture_carries_a_measured_written_mask() {
    let root = fixtures();

    // A case with no mask is a case whose two passes disagreed about something
    // other than the fill, and the oracle says which. Reported rather than
    // skipped: a missing mask reads as "nothing was written", which is the one
    // conclusion it must never be allowed to suggest.
    let unmasked = root["unmasked"].as_array().expect("unmasked array");
    assert!(
        unmasked.is_empty(),
        "the oracle could not compare both passes for {} case(s): {}",
        unmasked.len(),
        unmasked
            .iter()
            .map(|u| format!("{} ({})", u["name"], u["reason"]))
            .collect::<Vec<_>>()
            .join(", ")
    );

    let poison = root["poison"].as_array().expect("poison array");
    assert_eq!(poison.len(), 2, "a written mask needs exactly two fills");
    assert_ne!(
        poison[0], poison[1],
        "the two fills must differ, or every bit agrees and everything reads written"
    );

    for case in root["cases"].as_array().expect("cases array") {
        let name = case["name"].as_str().expect("case name");
        let bytes = unhex(case["buffer"].as_str().expect("buffer hex"));
        let mask = unhex(
            case["written_mask"]
                .as_str()
                .unwrap_or_else(|| panic!("{name}: no written_mask")),
        );
        assert_eq!(
            mask.len(),
            bytes.len(),
            "{name}: the mask and the record are different lengths"
        );
    }
}

#[test]
#[cfg_attr(not(wire_fixtures), ignore = "run scripts/wire-oracle/wire-oracle.sh")]
fn no_view_reads_a_bit_the_serializer_never_wrote() {
    let root = fixtures();

    // Union the measured masks per (class, record): a bit set here was written
    // by at least one fixture of that record, which is the weakest claim that
    // still forbids a view from reading it.
    let mut union: std::collections::BTreeMap<(String, String, usize), Vec<u8>> =
        Default::default();
    for case in root["cases"].as_array().expect("cases array") {
        let bytes = unhex(case["buffer"].as_str().expect("buffer hex"));
        if bytes.len() < 8 {
            continue;
        }
        let mask = unhex(case["written_mask"].as_str().expect("written_mask"));
        let key = (
            case["class"].as_str().expect("class").to_string(),
            record_key(
                case["selector"].as_str().expect("selector"),
                case["name"].as_str().expect("name"),
                &bytes,
            ),
            bytes.len(),
        );
        match union.get_mut(&key) {
            None => {
                union.insert(key, mask);
            }
            Some(prev) => {
                for (p, m) in prev.iter_mut().zip(mask) {
                    *p |= m;
                }
            }
        }
    }

    let mut partial = 0usize;
    for ((class, record, len), mask) in &union {
        let holes: Vec<(usize, u8)> = mask
            .iter()
            .enumerate()
            .filter(|(_, m)| **m != 0xff)
            .map(|(i, m)| (i, *m))
            .collect();
        let declared = PARTIALLY_WRITTEN
            .iter()
            .find(|(c, r, l, _)| c == class && r == record && l == len)
            .map(|(_, _, _, h)| *h);
        match (holes.is_empty(), declared) {
            (true, None) => {}
            (true, Some(d)) => panic!(
                "-[{class}] {record} at {len} bytes is declared partially written as \
                 {d:?}, but every bit of every fixture is written now"
            ),
            (false, None) => panic!(
                "-[{class}] {record} at {len} bytes leaves bits unwritten at {} and \
                 PARTIALLY_WRITTEN does not say so. Those bytes are the guest's stale \
                 ring on a real wire; add the row and check no view reads them",
                describe_holes(&holes)
            ),
            (false, Some(d)) => {
                assert_eq!(
                    holes,
                    d.to_vec(),
                    "-[{class}] {record} at {len} bytes: the serializer's written bits moved"
                );
                partial += 1;
            }
        }
    }
    assert_eq!(
        partial,
        PARTIALLY_WRITTEN.len(),
        "PARTIALLY_WRITTEN has rows no fixture produced; a stale row claims a hole that \
         is not there"
    );
    eprintln!(
        "checked written masks for {} (class, record) pairs; {partial} are partially written",
        union.len()
    );
}

#[test]
#[cfg_attr(not(wire_inventory), ignore = "run wire-oracle.sh --inventory")]
fn the_selector_inventory_matches_what_the_manifest_believes() {
    // Guards against the manifest silently going stale when the host OS ships a
    // serializer with a different surface: coverage is only meaningful against
    // the right denominator.
    let root = inventory();

    for class in root["classes"].as_array().expect("classes array") {
        let name = class["class"].as_str().expect("class name");
        let count = class["instance_methods"].as_u64().expect("method count") as usize;
        let known = manifest::INVENTORY
            .iter()
            .find(|c| c.class == name)
            .unwrap_or_else(|| panic!("host serializer has class {name}, manifest does not"));
        assert_eq!(
            known.instance_methods, count,
            "{name}: host reports {count} selectors, manifest records {}. \
             Update manifest::INVENTORY and triage the difference.",
            known.instance_methods
        );
    }

    // Every row must name a selector Apple actually ships. The class check
    // above cannot catch a mistyped selector, and a row for a selector that
    // does not exist counts toward `covered` while covering nothing — the same
    // failure the manifest exists to prevent, pointing the other way.
    for e in manifest::MANIFEST {
        let class = root["classes"]
            .as_array()
            .expect("classes array")
            .iter()
            .find(|c| c["class"] == e.class)
            .unwrap_or_else(|| panic!("manifest names class {}, host does not", e.class));
        assert!(
            class["selectors"]
                .as_array()
                .expect("selectors array")
                .iter()
                .any(|s| s["selector"] == e.selector),
            "manifest row -[{} {}] names a selector this serializer does not ship",
            e.class,
            e.selector
        );
    }
}

/// The selector inventory, gated on its own `cfg`.
///
/// Separate from [`fixtures`] because the two outputs are regenerated by
/// different runs of the capture: having one without the other is a normal
/// intermediate state, and it should stand down only the tests that read the
/// half that is missing.
fn inventory() -> Value {
    read_oracle_output("inventory.json", " --inventory")
}

/// The Objective-C type encoding Apple ships for one selector.
fn type_encoding(root: &Value, class: &str, selector: &str) -> String {
    let c = root["classes"]
        .as_array()
        .expect("classes array")
        .iter()
        .find(|c| c["class"] == class)
        .unwrap_or_else(|| panic!("inventory has no class {class}"));
    let s = c["selectors"]
        .as_array()
        .expect("selectors array")
        .iter()
        .find(|s| s["selector"] == selector)
        .unwrap_or_else(|| panic!("inventory has no -[{class} {selector}]"));
    s["type_encoding"]
        .as_str()
        .unwrap_or_else(|| {
            panic!(
                "-[{class} {selector}] carries no type encoding; the inventory \
                 predates schema 2, regenerate it"
            )
        })
        .to_string()
}

/// Apple's own metadata, against the widths the views read.
///
/// This is the crate's *first* derivation source and it is independent of every
/// byte in `fixtures.json`: `method_getTypeEncoding` states each argument's
/// width and order before a single operation is captured. Where a record
/// carries its arguments verbatim, the two derivations must agree — and a
/// disagreement means a view is reading a width Apple's API does not have,
/// which the fixtures alone can miss whenever a test value is small enough to
/// look right at either width.
///
/// The draws are deliberately absent from the width table. Every draw argument
/// is declared 64-bit and none of them reaches the wire that way in the compact
/// form, because the serializer picks its encoding by magnitude. For those the
/// encoding settles *signedness* instead, which the second half of this test
/// uses.
#[test]
#[cfg_attr(not(wire_inventory), ignore = "run wire-oracle.sh --inventory")]
fn the_type_encodings_agree_with_the_widths_the_views_read() {
    use reims_vgpu_wire::ops::render;
    use std::mem::size_of;

    let root = inventory();
    const ENC: &str = "PGSerializerRenderCommandEncoder";

    for (selector, fragment, body, total) in [
        (
            "setScissorRect:",
            "{?=QQQQ}",
            size_of::<render::ScissorRect>(),
            render::SET_SCISSOR_TOTAL_LEN,
        ),
        (
            "setViewport:",
            "{?=dddddd}",
            size_of::<render::Viewport>(),
            render::SET_VIEWPORT_TOTAL_LEN,
        ),
        (
            "setBlendColorRed:green:blue:alpha:",
            "f16f20f24f28",
            size_of::<render::BlendColor>(),
            render::SET_BLEND_COLOR_TOTAL_LEN,
        ),
        (
            "setCullMode:",
            "Q16",
            size_of::<render::ModeState>(),
            render::SET_MODE_TOTAL_LEN,
        ),
        (
            "setFrontFacingWinding:",
            "Q16",
            size_of::<render::ModeState>(),
            render::SET_MODE_TOTAL_LEN,
        ),
    ] {
        let enc = type_encoding(&root, ENC, selector);
        assert!(
            enc.contains(fragment),
            "-[{ENC} {selector}] is declared {enc}, which no longer contains \
             {fragment} — the argument widths this view assumes have changed"
        );
        assert_eq!(
            body + reims_vgpu_wire::OP_HEADER_LEN,
            total as usize,
            "{selector}: the view does not fill its record"
        );
    }

    // Exactly one argument in the whole draw family is declared signed, and it
    // is `baseVertex` — which is why `DrawIndexedInstancedBase` reads it through
    // a signed scalar while every count beside it is unsigned. Reading it
    // unsigned would turn a small negative offset into a large positive one,
    // and no fixture value would look wrong.
    let with_base = type_encoding(
        &root,
        ENC,
        "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:\
         instanceCount:baseVertex:baseInstance:",
    );
    assert_eq!(
        with_base.matches('q').count(),
        1,
        "the indexed draw with a base vertex is declared {with_base}; expected \
         exactly one signed argument"
    );
    for selector in [
        "drawPrimitives:vertexStart:vertexCount:",
        "drawPrimitives:vertexStart:vertexCount:instanceCount:",
        "drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:",
        "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:",
        "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:\
         instanceCount:",
    ] {
        let enc = type_encoding(&root, ENC, selector);
        assert_eq!(
            enc.matches('q').count(),
            0,
            "-[{ENC} {selector}] is declared {enc}; a signed argument appeared \
             where the views read only unsigned counts"
        );
    }
}

/// An exclusion that cites Apple's refusal must still be getting one.
///
/// `Coverage::Excluded` is the one state that closes a selector without a view,
/// so it is the one a mistake hides behind permanently. For the seven that cite
/// a serializer assertion the oracle re-drives them every capture and lists
/// what refused, which turns those rows from a remembered claim into a measured
/// one — in both directions: a selector that starts serializing drops out of
/// `unsupported` and fails here, and a row that quietly loses its evidence
/// fails here too.
#[test]
#[cfg_attr(not(wire_fixtures), ignore = "run scripts/wire-oracle/wire-oracle.sh")]
fn every_excluded_row_that_claims_a_refusal_still_gets_one() {
    let root = fixtures();
    let refused: std::collections::BTreeSet<&str> = root["unsupported"]
        .as_array()
        .expect("fixtures.json has no `unsupported` list; it predates schema 2, regenerate it")
        .iter()
        .map(|u| u["selector"].as_str().expect("selector"))
        .collect();

    let mut claimed = std::collections::BTreeSet::new();
    for e in manifest::MANIFEST {
        if e.coverage
            != (manifest::Coverage::Excluded {
                reason: manifest::REFUSED_BY_SERIALIZER,
            })
        {
            continue;
        }
        assert!(
            refused.contains(e.selector),
            "-[{} {}] is excluded because the serializer refuses it, but this \
             capture drove it without a refusal. Either it serializes now and \
             needs a view, or the oracle stopped driving it.",
            e.class,
            e.selector
        );
        claimed.insert(e.selector);
    }

    for selector in &refused {
        assert!(
            claimed.contains(selector),
            "the serializer refused -[{selector}] this run and no manifest row \
             records it; a refusal with no row is indistinguishable from a \
             selector nobody looked at"
        );
    }
    eprintln!(
        "{} selectors refused by the serializer, all recorded",
        refused.len()
    );
}

/// An exclusion that cites silence must still be getting it.
///
/// The same bidirectional check as the refusal one above, for the third
/// outcome: the selector ran, returned, and wrote no record. That is the
/// quietest way for a manifest row to be wrong, because a selector that starts
/// emitting looks exactly like one that never did.
///
/// Matched on `(class, selector)` rather than on the selector alone. The
/// refusal test predates the blit rows and can get away with a bare selector
/// set; this one cannot — `getType` is silent on the blit encoder and is a
/// selector name several classes ship.
#[test]
#[cfg_attr(not(wire_fixtures), ignore = "run scripts/wire-oracle/wire-oracle.sh")]
fn every_silent_selector_is_silent_under_every_capability() {
    let root = fixtures();

    let set = |key: &str| -> std::collections::BTreeSet<(String, String)> {
        root[key]
            .as_array()
            .unwrap_or_else(|| {
                panic!("fixtures.json has no `{key}` list; regenerate it with the current oracle")
            })
            .iter()
            .map(|u| {
                (
                    u["class"].as_str().expect("class").to_string(),
                    u["selector"].as_str().expect("selector").to_string(),
                )
            })
            .collect()
    };
    let silent = set("silent");
    let with_every_capability = set("silent_with_every_capability");
    assert!(
        !with_every_capability.is_empty(),
        "the sweep pass produced no silent selectors at all, which means it did \
         not run rather than that everything emits"
    );

    // A selector silent at the default state but not under the sweep is one a
    // capability unlocks. Claiming EMITS_NO_OPERATION for it is a false
    // statement about Apple: the serializer does emit, this harness just never
    // asked. That is how three families were nearly recorded as absent.
    let unlocked: Vec<_> = silent.difference(&with_every_capability).collect();
    for (class, selector) in &unlocked {
        let row = manifest::MANIFEST
            .iter()
            .find(|e| e.class == class && e.selector == selector);
        let Some(row) = row else { continue };
        assert_ne!(
            row.coverage,
            manifest::Coverage::Excluded {
                reason: manifest::EMITS_NO_OPERATION,
            },
            "-[{class} {selector}] is excluded as emitting nothing, but it emits \
             once a capability is forced on. The row is a claim about Apple and \
             it is wrong; the selector needs driving under its flag."
        );
    }

    // The reverse is a finding too, and a stranger one: a capability that
    // *suppresses* a record would mean the manifest's opcode for it is
    // conditional on negotiation.
    let suppressed: Vec<_> = with_every_capability.difference(&silent).collect();
    assert!(
        suppressed.is_empty(),
        "these selectors emit at the default capability state and fall silent \
         with every flag on, which no manifest row can express: {suppressed:?}"
    );

    eprintln!(
        "{} selectors silent at default, {} under every capability; {} are \
         capability-gated",
        silent.len(),
        with_every_capability.len(),
        unlocked.len()
    );
}

/// Every capability-gated selector names the flag that unlocks it.
///
/// The sweep above answers "is this gated"; it cannot answer "on what", and
/// that second question is the one the next step needs, because
/// `withCapability` takes one flag name. Answering it by trying flags in turn
/// is a guess with a fast oracle behind it — the same shape as the guess the
/// sweep pass was written to delete, one level down. So the capture runs one
/// pass per flag with that flag alone on, and the difference from the default
/// silent list is what that flag unlocks.
///
/// This test is the gate on that map being complete, and it prints it, which
/// is the point: the printed table is the remaining work queue with the
/// argument each entry needs already filled in.
///
/// A gated selector attributed to *no* flag is not a failure — it would mean
/// the family needs two flags at once, which is a real thing a serializer can
/// do. It is reported, because a conjunction is worth knowing about before
/// someone spends an afternoon on the single-flag case.
#[test]
#[cfg_attr(not(wire_fixtures), ignore = "run scripts/wire-oracle/wire-oracle.sh")]
fn every_capability_gated_selector_names_the_flag_that_unlocks_it() {
    let root = fixtures();

    let set = |key: &str| -> std::collections::BTreeSet<(String, String)> {
        root[key]
            .as_array()
            .unwrap_or_else(|| {
                panic!("fixtures.json has no `{key}` list; regenerate it with the current oracle")
            })
            .iter()
            .map(|u| {
                (
                    u["class"].as_str().expect("class").to_string(),
                    u["selector"].as_str().expect("selector").to_string(),
                )
            })
            .collect()
    };
    let gated: std::collections::BTreeSet<_> = set("silent")
        .difference(&set("silent_with_every_capability"))
        .cloned()
        .collect();

    let attribution = root["capability_attribution"].as_array().expect(
        "fixtures.json has no `capability_attribution` list; regenerate it with \
         the current oracle",
    );
    assert!(
        !attribution.is_empty(),
        "the attribution passes produced no entries at all, which means they did \
         not run rather than that no flag unlocks anything"
    );

    // Which flags unlock a given selector, and — the stranger direction — which
    // flags stop one emitting. A suppression would mean the fixtures this crate
    // pins are conditional on a capability being *off*, which is a far larger
    // claim than "some families need a flag" and no manifest row can express it.
    let mut unlocked_by: std::collections::BTreeMap<(String, String), Vec<&str>> =
        std::collections::BTreeMap::new();
    let mut suppressions: Vec<String> = Vec::new();
    for entry in attribution {
        let flag = entry["flag"].as_str().expect("flag");
        for u in entry["unlocks"].as_array().expect("unlocks") {
            let key = (
                u["class"].as_str().expect("class").to_string(),
                u["selector"].as_str().expect("selector").to_string(),
            );
            assert!(
                gated.contains(&key),
                "the {flag} pass reports it unlocks -[{} {}], but that selector is \
                 not in the capability-gated set. The two measurements disagree, \
                 so one of the passes did not see the state it thought it did",
                key.0,
                key.1
            );
            unlocked_by.entry(key).or_default().push(flag);
        }
        for s in entry["suppresses"].as_array().expect("suppresses") {
            suppressions.push(format!(
                "{flag} silences -[{} {}]",
                s["class"].as_str().expect("class"),
                s["selector"].as_str().expect("selector")
            ));
        }
    }
    assert!(
        suppressions.is_empty(),
        "a capability stopped a record being emitted, which means a manifest \
         opcode is conditional on the flag being off: {suppressions:?}"
    );

    let unattributed: Vec<_> = gated
        .iter()
        .filter(|k| !unlocked_by.contains_key(*k))
        .collect();
    for (class, selector) in &gated {
        match unlocked_by.get(&(class.clone(), selector.clone())) {
            Some(flags) => eprintln!(
                "  {} -[{} {selector}]",
                flags.join("+"),
                class.trim_start_matches("PGSerializer")
            ),
            None => eprintln!(
                "  (no single flag) -[{} {selector}] — needs a conjunction",
                class.trim_start_matches("PGSerializer")
            ),
        }
    }
    eprintln!(
        "{} capability-gated selectors, {} attributed to a single flag, {} needing \
         more than one",
        gated.len(),
        unlocked_by.len(),
        unattributed.len()
    );
}

/// A capability delta this crate has seen, and what is known about it.
///
/// Keyed by `(flag, kind, case name)` — the case rather than the selector,
/// because a length change lands on every case a selector has and each one is a
/// separate measurement.
struct KnownDelta {
    flag: &'static str,
    kind: &'static str,
    name: &'static str,
    /// Why this crate's pinned layout is still right in the face of it.
    note: &'static str,
}

/// Every delta the capture currently reports, with its reading.
///
/// This is a roster, not a suppression list. A delta not in it fails the test,
/// and a row here that stops firing fails it too — because a delta that goes
/// away means either the finding was repaired (and this row should go with the
/// repair) or the capture stopped measuring it.
const KNOWN_CAPABILITY_DELTAS: &[KnownDelta] = &[
    KnownDelta {
        flag: "TileShaders",
        kind: "absent",
        name: "render_pass_tile_size",
        note: "DERIVED. `tileWidth` and `tileHeight` reach no byte of the pass \
               record at the default state -- measured, with Metal confirmed to \
               have kept the values the case set -- and under this flag they \
               leave the descriptor entirely and emit `0x24` beside it. So the \
               case emits one record here and two there, which is what `absent` \
               reports. Both halves are pinned: this case is the negative result \
               and `render_pass_tile_size_capable` is the record.",
    },
    KnownDelta {
        flag: "TileShaders",
        kind: "absent",
        name: "render_pass_imageblock",
        note: "DERIVED, same shape one property over: `imageblockSampleLength` \
               and `threadgroupMemoryLength` emit `0x22` and `0x23` under this \
               flag and nothing at all without it. Note the count -- three \
               records, not two, because each is its own opcode; \
               `render_pass_imageblock_capable` asserts that.",
    },
    KnownDelta {
        flag: "SwizzledTextures",
        kind: "length",
        name: "*",
        note: "DERIVED, and this row cannot go away: every \
               `newTextureWithDescriptor:allocator:` record switches from opcode 1 \
               at 44 bytes to opcode 0x34 at 52, so the 21 cases that drive that \
               selector differ in this pass by construction. The wide body is \
               `ops::texture::WideTextureDescriptorBody` and the two \
               `texture_swizzled` fixtures pin it. What is still open is the \
               device: `reims-vgpu` has no decoder for 0x34.",
    },
    KnownDelta {
        flag: "TextureDescriptor2",
        kind: "length",
        name: "*",
        note: "DERIVED, and this row cannot go away either: the other four \
               texture-creation records switch opcode to the same wide body — \
               buffer-backed 0x09 -> 0x37, heap-placed 0x15 -> 0x38, \
               IOSurface-backed 0x0c -> 0x39, and the heap size/align query \
               0x1c3 -> 0x1d5. All four have `_wide` fixtures pinning their \
               prefixes, which is where the one surprise was: the IOSurface \
               `plane` is written four bytes wide here and two in the narrow \
               form. The plain record does not move under this flag, which is \
               what `the_second_texture_descriptor_layout_does_not_reach_the_wire` \
               measured and is narrower than it reads.",
    },
    KnownDelta {
        flag: "ComputePassDescriptorDispatchType",
        kind: "absent",
        name: "*",
        note: "DERIVED, and this row stays because it cannot go away: six compute \
               selectors emit a *second* record with the flag on — every dispatch \
               form and both `executeCommandsInBuffer:` forms — so a case claiming \
               one operation lands on `multi` in that pass no matter what. The \
               second record is the 0xd7 memory barrier at scope \
               Buffers|Textures, emitted *after* the selector's own, and it is \
               the pass's dispatch type that decides: concurrent emits one. \
               Fixtures `compute_dispatch_threadgroups_serial`, \
               `compute_execute_commands_range_serial` and \
               `compute_dispatch_threadgroups_concurrent` pin all three arms at \
               the default state's capability forced on.",
    },
    KnownDelta {
        flag: "ProtectionOptionsEnvelope",
        kind: "absent",
        name: "blit_begin_segment_alt",
        note: "DERIVED, and this row cannot go away: \
               `beginSegment:protectionOptions:` emits three records instead of \
               one — a type-5 segment header, eight bytes that are the \
               `protectionOptions:` argument verbatim, then the ordinary header. \
               It needs the BOOL clear *and* non-zero options; either alone \
               emits one record. `blit_begin_segment_protected` and `..._alt` \
               pin the burst, `..._flag_set` and `..._protection_zero` pin the \
               two single-record arms. `reims-vgpu`'s decode::stream already \
               skipped type 5 correctly and its own doc's prediction was right.",
    },
];

/// No capability changes what a record contains — and it is not true.
///
/// The two capability tests above answer "does this selector emit" in both
/// directions. Neither can answer the third question, which is whether a flag
/// changes the record a selector *already* emits, and that is the one with
/// teeth: every fixture this crate pins was captured at the default state, and
/// nothing in `reims-vgpu` observes the guest negotiating a capability. A flag
/// that moves a field would make the pinned layout wrong for exactly the guests
/// that turned it on, silently.
///
/// The capture now diffs every capability pass's records against the default
/// pass's, case by case and byte by byte, and the answer is **no**. It found six
/// deltas across five flags; two are repaired and the rest are rostered above,
/// because each is its own piece of work and a roster keeps them from being
/// rediscovered one at a time.
///
/// The two that are gone were also a hole in the instrument beside this one, and
/// that is the part worth keeping. `capability_attribution` diffs the two
/// passes' **silent** lists, so a selector that *asserts* at the default state
/// is invisible to it — and both
/// `dispatchThreadsWithIndirectBuffer:indirectBufferOffset:` and
/// `insertCompressedTextureReinterpretationFlush` were exactly that: refused at
/// default, emitting under their own flags, and carrying a manifest row that
/// said Apple refuses them. Reading "0 capability-gated selectors" as "every
/// gated selector has been driven" was therefore true only of the silent kind.
///
/// The `(every flag)` sweep is excluded from the roster match on purpose: it is
/// the union of the sixteen single-flag passes and reports nothing they do not,
/// so rostering it twice would mean two rows to update per repair. It is still
/// checked — every sweep delta must be attributable to some single flag, and one
/// that is not would mean a *conjunction* of flags changes a record, which no
/// single-flag pass can find.
#[test]
#[cfg_attr(not(wire_fixtures), ignore = "run scripts/wire-oracle/wire-oracle.sh")]
fn no_capability_changes_what_a_record_contains() {
    let root = fixtures();

    let deltas = root["capability_content_deltas"].as_array().expect(
        "fixtures.json has no `capability_content_deltas` list; regenerate it \
         with the current oracle",
    );

    let matches = |k: &KnownDelta, flag: &str, kind: &str, name: &str| {
        k.flag == flag && k.kind == kind && (k.name == "*" || k.name == name)
    };

    let mut unexplained: Vec<String> = Vec::new();
    let mut fired: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
    // Every (flag, kind) the sweep reported, so a sweep-only delta — one that
    // needs several flags at once — cannot hide inside the union.
    let mut single_flag: std::collections::BTreeSet<(String, String)> =
        std::collections::BTreeSet::new();
    let mut sweep: Vec<(String, String)> = Vec::new();

    for d in deltas {
        let flag = d["flag"].as_str().expect("flag");
        let kind = d["kind"].as_str().expect("kind");
        let name = d["name"].as_str().expect("name");
        if flag == "(every flag)" {
            sweep.push((kind.to_string(), name.to_string()));
            continue;
        }
        single_flag.insert((kind.to_string(), name.to_string()));
        match KNOWN_CAPABILITY_DELTAS
            .iter()
            .position(|k| matches(k, flag, kind, name))
        {
            Some(i) => {
                fired.insert(i);
            }
            None => unexplained.push(format!(
                "{flag} / {kind} / {name}: -[{} {}] — {}",
                d["class"].as_str().unwrap_or("?"),
                d["selector"].as_str().unwrap_or("?"),
                d["reason"].as_str().unwrap_or("?")
            )),
        }
    }

    assert!(
        unexplained.is_empty(),
        "a capability changed a record and nothing in this crate accounts for it. \
         Every fixture here was captured with all sixteen flags off, so a guest \
         that negotiates one sends a record this crate's pinned layout does not \
         describe:\n  {}",
        unexplained.join("\n  ")
    );

    for (i, k) in KNOWN_CAPABILITY_DELTAS.iter().enumerate() {
        assert!(
            fired.contains(&i),
            "the roster says {} / {} / {} is a known capability delta and the \
             capture no longer reports it. Either it was repaired — in which case \
             this row goes with the repair — or the capture stopped measuring it, \
             which is worse.",
            k.flag,
            k.kind,
            k.name
        );
        // Printed, not merely stored: this list is the remaining work queue for
        // the one question the capability sweep could not answer, and a queue
        // nobody can read is a queue nobody works.
        eprintln!("  {} / {}: {}", k.flag, k.kind, k.note);
    }

    let conjunction_only: Vec<_> = sweep.iter().filter(|s| !single_flag.contains(*s)).collect();
    assert!(
        conjunction_only.is_empty(),
        "the sweep found record changes no single flag accounts for, so a \
         *conjunction* of capabilities changes these records and the per-flag \
         attribution cannot find them: {conjunction_only:?}"
    );

    eprintln!(
        "{} capability content deltas ({} from the sweep, {} attributed to a \
         single flag), all {} rostered",
        deltas.len(),
        sweep.len(),
        deltas.len() - sweep.len(),
        KNOWN_CAPABILITY_DELTAS.len()
    );
}

#[test]
#[cfg_attr(not(wire_fixtures), ignore = "run scripts/wire-oracle/wire-oracle.sh")]
fn every_excluded_row_that_claims_silence_still_gets_it() {
    let root = fixtures();
    let silent: std::collections::BTreeSet<(&str, &str)> = root["silent"]
        .as_array()
        .expect("fixtures.json has no `silent` list; regenerate it with the current oracle")
        .iter()
        .map(|u| {
            (
                u["class"].as_str().expect("class"),
                u["selector"].as_str().expect("selector"),
            )
        })
        .collect();

    let mut claimed = std::collections::BTreeSet::new();
    for e in manifest::MANIFEST {
        if e.coverage
            != (manifest::Coverage::Excluded {
                reason: manifest::EMITS_NO_OPERATION,
            })
        {
            continue;
        }
        assert!(
            silent.contains(&(e.class, e.selector)),
            "-[{} {}] is excluded because it emits nothing, but this capture did \
             not observe that. Either it emits a record now and needs a view, or \
             the oracle stopped driving it.",
            e.class,
            e.selector
        );
        claimed.insert((e.class, e.selector));
    }

    // Every silence needs a row, but not the same row. A selector that is
    // silent at the default state *and* under the sweep pass emits nothing
    // Apple can be asked for, and `EMITS_NO_OPERATION` says so. One that a
    // capability unlocks is a different thing entirely — it emits, and this
    // crate has no view for it — which is `Unimplemented`. Both are records of
    // having looked; only the absence of a row is not.
    let gated: std::collections::BTreeSet<(&str, &str)> = root["silent_with_every_capability"]
        .as_array()
        .expect("fixtures.json has no `silent_with_every_capability` list")
        .iter()
        .map(|u| {
            (
                u["class"].as_str().expect("class"),
                u["selector"].as_str().expect("selector"),
            )
        })
        .fold(silent.clone(), |mut acc, k| {
            acc.remove(&k);
            acc
        });

    // Selectors that produced a record somewhere in this run.
    //
    // A third state, and it took a real selector to find it. The two above are
    // "silent always" and "silent because its capability is off"; this one is
    // **silent sometimes**. `maybeEmitSerialBarrier` emits the scope barrier
    // with the pass's dispatch type left alone and emits nothing once the type
    // is concurrent, so it appears in `cases` and in `silent` at the same time
    // and neither of the rules below fits it.
    //
    // That is not a licence for any `Covered` row to sit beside a silence: the
    // row is honest only because a case *did* observe a record, which is
    // exactly what this set checks. A selector whose row claims `Covered` with
    // no case behind it still fails the last assertion.
    let emitted: std::collections::BTreeSet<(&str, &str)> = root["cases"]
        .as_array()
        .expect("cases array")
        .iter()
        .map(|c| {
            (
                c["class"].as_str().expect("class"),
                c["selector"].as_str().expect("selector"),
            )
        })
        .collect();

    let mut gated_rows = 0usize;
    let mut conditional_rows = 0usize;
    for (class, selector) in &silent {
        if emitted.contains(&(*class, *selector)) {
            let row = manifest::MANIFEST
                .iter()
                .find(|e| e.class == *class && e.selector == *selector);
            assert!(
                row.is_some_and(|e| matches!(e.coverage, manifest::Coverage::Covered { .. })),
                "-[{class} {selector}] emitted a record in one case and nothing in \
                 another, so it is a conditional emitter and its row must be \
                 `Covered`. An exclusion would say Apple never writes a record \
                 for it, which this run disproves."
            );
            conditional_rows += 1;
            continue;
        }
        if gated.contains(&(*class, *selector)) {
            let row = manifest::MANIFEST
                .iter()
                .find(|e| e.class == *class && e.selector == *selector);
            assert!(
                row.is_some_and(|e| e.coverage == manifest::Coverage::Unimplemented),
                "-[{class} {selector}] is silent only because its capability is \
                 off. That is not `EMITS_NO_OPERATION` and it is not covered \
                 either; the row must be `Unimplemented` until it is driven \
                 under its flag and given a view."
            );
            gated_rows += 1;
            continue;
        }
        assert!(
            claimed.contains(&(*class, *selector)),
            "-[{class} {selector}] emitted nothing this run and no manifest row \
             records it; a silence with no row is indistinguishable from a \
             selector nobody looked at"
        );
    }
    eprintln!(
        "{} selectors emitted no operation, all recorded ({gated_rows} of them \
         only because a capability is off, {conditional_rows} of them only in \
         one of their cases)",
        silent.len()
    );
}

#[test]
fn coverage_is_reported_every_run_so_the_gap_stays_visible() {
    let c = manifest::counts();
    let untriaged = manifest::untriaged();
    let surface: usize = manifest::INVENTORY.iter().map(|c| c.instance_methods).sum();
    eprintln!(
        "wire coverage: {} covered, {} unimplemented, {} excluded, {} untriaged of {} selectors",
        c.covered, c.unimplemented, c.excluded, untriaged, surface
    );
    assert_eq!(c.rows() + untriaged, surface);
}

/// Render encoder records, checked field by field against the arguments the
/// Metal call was given.
///
/// This is the surface `runtime::exec` decodes, so it is the one that has to be
/// right for the crate to be usable — and it is the one that caught the 12-byte
/// header, because these records carry no object ref to hide the boundary.
#[test]
#[cfg_attr(not(wire_fixtures), ignore = "run scripts/wire-oracle/wire-oracle.sh")]
fn every_render_fixture_reads_back_what_metal_was_asked_for() {
    use reims_vgpu_wire::ops::{render, render_pass, tile};

    let root = fixtures();
    let mut checked = 0usize;
    let mut draw_opcodes_seen = std::collections::BTreeSet::new();

    for case in root["cases"].as_array().expect("cases array") {
        if case["class"] != "PGSerializerRenderCommandEncoder" {
            continue;
        }
        let name = case["name"].as_str().expect("case name");
        let selector = case["selector"].as_str().expect("case selector");
        // A case whose expectations name `requested` rather than `count` is a
        // truncation witness: it asked for a range Apple's serializer refuses to
        // write whole, so "reads back what Metal was asked for" is the one thing
        // it must not do. `a_plural_bind_is_truncated_at_the_argument_table_size`
        // owns those, and asserts the relationship instead of the value.
        if case["expect"].get("requested").is_some() {
            continue;
        }
        let bytes = unhex(case["buffer"].as_str().expect("buffer hex"));
        let allocated = case["allocated_len"].as_u64().expect("allocated_len");

        // Segment framing rather than a command: no opcode, so it cannot go
        // through `op()`. Checked by
        // `every_segment_header_fixture_reads_back_what_the_encoder_wrote`.
        if selector == "beginSegment:protectionOptions:" {
            continue;
        }

        let o = op(&bytes, 0).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            o.length() as u64,
            allocated,
            "{name}: length disagrees with the serializer's own allocation"
        );
        if !is_variable_length(o.opcode()) {
            assert_eq!(
                record_len_for(o.opcode()),
                Some(o.length()),
                "{name}: opcode {:#x} came in at {} bytes; the crate's constant says otherwise",
                o.opcode(),
                o.length()
            );
        }

        // The compact/wide choice, predicted from the arguments alone and
        // checked against the opcode Apple actually wrote. This is the whole
        // selection rule as an executable claim: every draw fixture votes on it.
        if let Some((compact, wide)) = draw_pair(selector) {
            let want = if expects_wide_encoding(case) {
                wide
            } else {
                compact
            };
            assert_eq!(
                o.opcode(),
                want,
                "{name}: arguments predicted opcode {want:#x}, serializer emitted {:#x} \
                 -- the magnitude rule in ops::render is wrong",
                o.opcode()
            );
            draw_opcodes_seen.insert(o.opcode());
        }

        match o.opcode() {
            render::OPCODE_DRAW => {
                let d = render::draw(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_eq!(d.primitive_type.get() as u64, expect_u64(case, "primitive_type"), "{name}: primitive_type");
                assert_eq!(d.vertex_start.get() as u64, expect_u64(case, "vertex_start"), "{name}: vertex_start");
                assert_eq!(d.vertex_count.get() as u64, expect_u64(case, "vertex_count"), "{name}: vertex_count");
            }
            render::OPCODE_DRAW_WIDE => {
                let d = render::draw_wide(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_eq!(d.primitive_type.get() as u64, expect_u64(case, "primitive_type"), "{name}: primitive_type");
                assert_eq!(d.vertex_start.get(), expect_u64(case, "vertex_start"), "{name}: vertex_start");
                assert_eq!(d.vertex_count.get(), expect_u64(case, "vertex_count"), "{name}: vertex_count");
            }
            render::OPCODE_DRAW_INSTANCED => {
                let d = render::draw_instanced(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_eq!(d.primitive_type.get() as u64, expect_u64(case, "primitive_type"), "{name}: primitive_type");
                assert_eq!(d.vertex_start.get() as u64, expect_u64(case, "vertex_start"), "{name}: vertex_start");
                assert_eq!(d.vertex_count.get() as u64, expect_u64(case, "vertex_count"), "{name}: vertex_count");
                assert_eq!(d.instance_count.get() as u64, expect_u64(case, "instance_count"), "{name}: instance_count");
            }
            render::OPCODE_DRAW_INSTANCED_WIDE => {
                let d = render::draw_instanced_wide(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_eq!(d.primitive_type.get() as u64, expect_u64(case, "primitive_type"), "{name}: primitive_type");
                assert_eq!(d.vertex_start.get(), expect_u64(case, "vertex_start"), "{name}: vertex_start");
                assert_eq!(d.vertex_count.get(), expect_u64(case, "vertex_count"), "{name}: vertex_count");
                assert_eq!(d.instance_count.get(), expect_u64(case, "instance_count"), "{name}: instance_count");
            }
            render::OPCODE_DRAW_INSTANCED_BASE => {
                let d = render::draw_instanced_base(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_eq!(d.primitive_type.get() as u64, expect_u64(case, "primitive_type"), "{name}: primitive_type");
                assert_eq!(d.vertex_start.get() as u64, expect_u64(case, "vertex_start"), "{name}: vertex_start");
                assert_eq!(d.vertex_count.get() as u64, expect_u64(case, "vertex_count"), "{name}: vertex_count");
                assert_eq!(d.instance_count.get() as u64, expect_u64(case, "instance_count"), "{name}: instance_count");
                assert_eq!(d.base_instance.get() as u64, expect_u64(case, "base_instance"), "{name}: base_instance");
            }
            render::OPCODE_DRAW_INSTANCED_BASE_WIDE => {
                let d = render::draw_instanced_base_wide(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_eq!(d.primitive_type.get() as u64, expect_u64(case, "primitive_type"), "{name}: primitive_type");
                assert_eq!(d.vertex_start.get(), expect_u64(case, "vertex_start"), "{name}: vertex_start");
                assert_eq!(d.vertex_count.get(), expect_u64(case, "vertex_count"), "{name}: vertex_count");
                assert_eq!(d.instance_count.get(), expect_u64(case, "instance_count"), "{name}: instance_count");
                assert_eq!(d.base_instance.get(), expect_u64(case, "base_instance"), "{name}: base_instance");
            }
            render::OPCODE_DRAW_INDEXED => {
                let d = render::draw_indexed(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_eq!(d.primitive_type.get() as u64, expect_u64(case, "primitive_type"), "{name}: primitive_type");
                assert_eq!(d.index_type.get() as u64, expect_u64(case, "index_type"), "{name}: index_type");
                assert_eq!(d.index_buffer_ref.get() as u64, expect_u64(case, "index_buffer_ref"), "{name}: index_buffer_ref");
                assert_eq!(d.index_count.get() as u64, expect_u64(case, "index_count"), "{name}: index_count");
                assert_eq!(d.index_buffer_offset.get() as u64, expect_u64(case, "index_buffer_offset"), "{name}: index_buffer_offset");
            }
            render::OPCODE_DRAW_INDEXED_WIDE => {
                let d = render::draw_indexed_wide(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_eq!(d.primitive_type.get() as u64, expect_u64(case, "primitive_type"), "{name}: primitive_type");
                assert_eq!(d.index_type.get() as u64, expect_u64(case, "index_type"), "{name}: index_type");
                assert_eq!(d.index_buffer_ref.get() as u64, expect_u64(case, "index_buffer_ref"), "{name}: index_buffer_ref");
                assert_eq!(d.index_count.get(), expect_u64(case, "index_count"), "{name}: index_count");
                assert_eq!(d.index_buffer_offset.get(), expect_u64(case, "index_buffer_offset"), "{name}: index_buffer_offset");
            }
            render::OPCODE_DRAW_INDEXED_INSTANCED => {
                let d = render::draw_indexed_instanced(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_eq!(d.primitive_type.get() as u64, expect_u64(case, "primitive_type"), "{name}: primitive_type");
                assert_eq!(d.index_type.get() as u64, expect_u64(case, "index_type"), "{name}: index_type");
                assert_eq!(d.index_buffer_ref.get() as u64, expect_u64(case, "index_buffer_ref"), "{name}: index_buffer_ref");
                assert_eq!(d.index_count.get() as u64, expect_u64(case, "index_count"), "{name}: index_count");
                assert_eq!(d.index_buffer_offset.get() as u64, expect_u64(case, "index_buffer_offset"), "{name}: index_buffer_offset");
                assert_eq!(d.instance_count.get() as u64, expect_u64(case, "instance_count"), "{name}: instance_count");
            }
            render::OPCODE_DRAW_INDEXED_INSTANCED_WIDE => {
                let d = render::draw_indexed_instanced_wide(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_eq!(d.primitive_type.get() as u64, expect_u64(case, "primitive_type"), "{name}: primitive_type");
                assert_eq!(d.index_type.get() as u64, expect_u64(case, "index_type"), "{name}: index_type");
                assert_eq!(d.index_buffer_ref.get() as u64, expect_u64(case, "index_buffer_ref"), "{name}: index_buffer_ref");
                assert_eq!(d.index_count.get(), expect_u64(case, "index_count"), "{name}: index_count");
                assert_eq!(d.index_buffer_offset.get(), expect_u64(case, "index_buffer_offset"), "{name}: index_buffer_offset");
                assert_eq!(d.instance_count.get(), expect_u64(case, "instance_count"), "{name}: instance_count");
            }
            render::OPCODE_DRAW_INDEXED_INSTANCED_BASE => {
                let d = render::draw_indexed_instanced_base(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_eq!(d.primitive_type.get() as u64, expect_u64(case, "primitive_type"), "{name}: primitive_type");
                assert_eq!(d.index_type.get() as u64, expect_u64(case, "index_type"), "{name}: index_type");
                assert_eq!(d.index_buffer_ref.get() as u64, expect_u64(case, "index_buffer_ref"), "{name}: index_buffer_ref");
                assert_eq!(d.index_count.get() as u64, expect_u64(case, "index_count"), "{name}: index_count");
                assert_eq!(d.index_buffer_offset.get() as u64, expect_u64(case, "index_buffer_offset"), "{name}: index_buffer_offset");
                assert_eq!(d.instance_count.get() as u64, expect_u64(case, "instance_count"), "{name}: instance_count");
                assert_eq!(d.base_instance.get() as u64, expect_u64(case, "base_instance"), "{name}: base_instance");
                // Truncated to 16 bits by Apple's serializer, so the record can
                // only be expected to carry the low half of what Metal was
                // given -- see DrawIndexedInstancedBase::base_vertex.
                assert_eq!(
                    d.base_vertex.get(),
                    expect_i64(case, "base_vertex") as i16,
                    "{name}: base_vertex"
                );
            }
            render::OPCODE_DRAW_INDEXED_INSTANCED_BASE_WIDE => {
                let d = render::draw_indexed_instanced_base_wide(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_eq!(d.primitive_type.get() as u64, expect_u64(case, "primitive_type"), "{name}: primitive_type");
                assert_eq!(d.index_type.get() as u64, expect_u64(case, "index_type"), "{name}: index_type");
                assert_eq!(d.index_buffer_ref.get() as u64, expect_u64(case, "index_buffer_ref"), "{name}: index_buffer_ref");
                assert_eq!(d.index_count.get(), expect_u64(case, "index_count"), "{name}: index_count");
                assert_eq!(d.index_buffer_offset.get(), expect_u64(case, "index_buffer_offset"), "{name}: index_buffer_offset");
                assert_eq!(d.instance_count.get(), expect_u64(case, "instance_count"), "{name}: instance_count");
                assert_eq!(d.base_instance.get(), expect_u64(case, "base_instance"), "{name}: base_instance");
                // Sign-extended rather than truncated at this width.
                assert_eq!(d.base_vertex.get(), expect_i64(case, "base_vertex"), "{name}: base_vertex");
            }
            render::OPCODE_SET_SCISSOR => {
                let s = render::set_scissor(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_eq!(s.x.get(), expect_u64(case, "x"), "{name}: x");
                assert_eq!(s.y.get(), expect_u64(case, "y"), "{name}: y");
                assert_eq!(s.width.get(), expect_u64(case, "width"), "{name}: width");
                assert_eq!(s.height.get(), expect_u64(case, "height"), "{name}: height");
            }
            render::OPCODE_SET_VIEWPORT => {
                let v = render::set_viewport(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                for (got, key) in [
                    (v.origin_x.get(), "origin_x"),
                    (v.origin_y.get(), "origin_y"),
                    (v.width.get(), "width"),
                    (v.height.get(), "height"),
                    (v.znear.get(), "znear"),
                    (v.zfar.get(), "zfar"),
                ] {
                    let want = case["expect"][key].as_f64().unwrap_or_else(|| panic!("{name}: no expect.{key}"));
                    assert_eq!(got, want, "{name}: {key}");
                }
            }
            op if render::is_mode_state(op) => {
                let m = render::mode_state(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                // Six selectors, one record. Each case names its own field, so
                // the expectation key is whichever of these it carries — and a
                // case that carries none fails rather than checking nothing.
                let key = sole_key(case, &["cull_mode", "winding", "mode", "fill_mode", "store_action"]);
                assert_eq!(m.mode.get(), expect_u64(case, key), "{name}: {key}");
            }
            op if render::is_float_state(op) => {
                let f = render::float_state(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                let key = sole_key(case, &["width", "scale"]);
                let want = case["expect"][key].as_f64().unwrap_or_else(|| panic!("{name}: no expect.{key}"));
                assert_eq!(f.value.get() as f64, want, "{name}: {key}");
            }
            op if render::is_ref_bind(op) => {
                let (head, entries) = render::ref_binds(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                let key = sole_key(case, &["texture_ref", "sampler_ref"]);
                let want = expect_u64(case, key);
                // A singular selector is the `count == 1` case of the plural
                // one, and both write `first`. The `index`/`first` split in the
                // expectations is only which selector the case called.
                let first = case["expect"]["first"].as_u64().unwrap_or_else(|| expect_u64(case, "index"));
                let count = case["expect"]["count"].as_u64().unwrap_or(1);
                assert_eq!(head.first.get() as u64, first, "{name}: first");
                assert_eq!(head.count.get() as u64, count, "{name}: count");
                assert_eq!(entries.len() as u64, count, "{name}: entry count");
                for (i, e) in entries.iter().enumerate() {
                    assert_eq!(e.object_ref.get() as u64, want, "{name}: entry {i}");
                }
            }
            op if render::is_sampler_lod_bind(op) => {
                let (head, entries) =
                    render::sampler_lod_binds(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                let first = expect_u64(case, "first");
                let count = case["expect"]["count"].as_u64().unwrap_or(1);
                assert_eq!(head.first.get() as u64, first, "{name}: first");
                assert_eq!(head.count.get() as u64, count, "{name}: count");
                assert_eq!(entries.len() as u64, count, "{name}: entry count");
                // Per entry, not per record. The `_2` suffix only exists on the
                // plural cases, and those pass *different* clamps in each slot —
                // which is the whole reason the plural case is here, since a
                // one-entry record cannot tell a per-entry pair from a header.
                for (i, suffix) in ["", "_2"].iter().enumerate() {
                    if i >= entries.len() {
                        break;
                    }
                    assert_eq!(
                        entries[i].sampler_ref.get() as u64,
                        expect_u64(case, "sampler_ref"),
                        "{name}: entry {i} sampler_ref"
                    );
                    for (got, key) in [
                        (entries[i].lod_min_clamp.get(), "lod_min_clamp"),
                        (entries[i].lod_max_clamp.get(), "lod_max_clamp"),
                    ] {
                        let k = format!("{key}{suffix}");
                        let want = expect_f64(case, &k);
                        assert_eq!(got as f64, want, "{name}: {k}");
                    }
                }
                if entries.len() >= 2 {
                    assert_ne!(
                        (entries[0].lod_min_clamp.get(), entries[0].lod_max_clamp.get()),
                        (entries[1].lod_min_clamp.get(), entries[1].lod_max_clamp.get()),
                        "{name}: the two entries share their clamps, so this case \
                         cannot show they are per entry"
                    );
                }
            }
            op if render::is_buffer_bind(op) => {
                let (head, entries) = render::buffer_binds(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                let first = case["expect"]["first"].as_u64().unwrap_or_else(|| expect_u64(case, "index"));
                let count = case["expect"]["count"].as_u64().unwrap_or(1);
                assert_eq!(head.first.get() as u64, first, "{name}: first");
                assert_eq!(head.count.get() as u64, count, "{name}: count");
                assert_eq!(entries.len() as u64, count, "{name}: entry count");
                for (i, e) in entries.iter().enumerate() {
                    assert_eq!(e.buffer_ref.get() as u64, expect_u64(case, "buffer_ref"), "{name}: ref {i}");
                    // The plural case gives each slot a different offset, which
                    // is what shows the entry stride rather than assuming it.
                    let key = if count > 1 { format!("offset{i}") } else { "offset".to_string() };
                    assert_eq!(e.offset.get(), expect_u64(case, &key), "{name}: {key}");
                }
            }
            op if render::is_buffer_offset(op) => {
                let b = render::buffer_offset(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_eq!(b.index.get() as u64, expect_u64(case, "index"), "{name}: index");
                assert_eq!(b.offset.get(), expect_u64(case, "offset"), "{name}: offset");
            }
            op if render::is_buffer_stride_bind(op) => {
                let (head, entries) =
                    render::buffer_stride_binds(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                let first = case["expect"]["first"]
                    .as_u64()
                    .unwrap_or_else(|| expect_u64(case, "index"));
                let count = case["expect"]["count"].as_u64().unwrap_or(1);
                assert_eq!(head.first.get() as u64, first, "{name}: first");
                assert_eq!(head.count.get() as u64, count, "{name}: count");
                assert_eq!(entries.len() as u64, count, "{name}: entry count");
                for (i, e) in entries.iter().enumerate() {
                    // The bytes form has no `buffer_ref` expectation: the
                    // serializer stages the bytes into a buffer of its own
                    // choosing, so the ref is the staging buffer's rather than
                    // any value the case passed. Asserting a ref it never
                    // supplied would be reading the answer out of the bytes.
                    if let Some(want) = case["expect"]["buffer_ref"].as_u64() {
                        assert_eq!(e.buffer_ref.get() as u64, want, "{name}: ref {i}");
                    }
                    let suffix = if count > 1 { i.to_string() } else { String::new() };
                    assert_eq!(
                        e.offset.get(),
                        expect_u64(case, &format!("offset{suffix}")),
                        "{name}: offset{suffix}"
                    );
                    assert_eq!(
                        e.attribute_stride.get(),
                        expect_u64(case, &format!("attribute_stride{suffix}")),
                        "{name}: attribute_stride{suffix}"
                    );
                }
                // Per entry, not per record — the same claim the sampler LOD
                // arm makes, and it needs the same proof: a one-entry record
                // cannot tell a per-entry field from a trailing header word.
                if entries.len() >= 2 {
                    assert_ne!(
                        entries[0].attribute_stride.get(),
                        entries[1].attribute_stride.get(),
                        "{name}: the two entries share a stride, so this case \
                         cannot show it is per entry"
                    );
                }
            }
            render::OPCODE_SET_VERTEX_AMPLIFICATION_MODE => {
                let m = render::vertex_amplification_mode(&o)
                    .unwrap_or_else(|e| panic!("{name}: {e}"));
                // Both are `Q` in the type encoding and 32 bits here, so the
                // record cannot be read from the encoding alone.
                assert_eq!(m.mode.get() as u64, expect_u64(case, "mode"), "{name}: mode");
                assert_eq!(m.value.get() as u64, expect_u64(case, "value"), "{name}: value");
            }
            render::OPCODE_SET_VERTEX_AMPLIFICATION_COUNT => {
                let (head, entries) = render::vertex_amplification_count(&o)
                    .unwrap_or_else(|e| panic!("{name}: {e}"));
                let count = expect_u64(case, "count");
                assert_eq!(head.count.get() as u64, count, "{name}: count");
                assert_eq!(entries.len() as u64, count, "{name}: entry count");
                for (i, e) in entries.iter().enumerate() {
                    assert_eq!(
                        e.viewport_array_index_offset.get() as u64,
                        expect_u64(case, &format!("viewport_offset{i}")),
                        "{name}: viewport_offset{i}"
                    );
                    assert_eq!(
                        e.render_target_array_index_offset.get() as u64,
                        expect_u64(case, &format!("rt_offset{i}")),
                        "{name}: rt_offset{i}"
                    );
                }
                // The head is four bytes, not the eight-byte `BindHeader`. If it
                // were read as one, the count would come back as the first
                // mapping's viewport offset — so the two must differ.
                assert_ne!(
                    head.count.get(),
                    entries[0].viewport_array_index_offset.get(),
                    "{name}: this case cannot show the head is four bytes"
                );
            }
            render::OPCODE_SET_VERTEX_BUFFER_OFFSET_STRIDE => {
                let b = render::buffer_offset_stride(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_eq!(b.index.get() as u64, expect_u64(case, "index"), "{name}: index");
                assert_eq!(b.offset.get(), expect_u64(case, "offset"), "{name}: offset");
                assert_eq!(
                    b.attribute_stride.get(),
                    expect_u64(case, "attribute_stride"),
                    "{name}: attribute_stride"
                );
            }
            op if render::is_state_ref(op) => {
                let s = render::state_ref(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                let key = sole_key(case, &["pipeline_ref", "depth_stencil_ref"]);
                assert_eq!(s.object_ref.get() as u64, expect_u64(case, key), "{name}: {key}");
            }
            op if render::is_fence(op) => {
                let f = render::fence(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_eq!(f.fence_ref.get() as u64, expect_u64(case, "fence_ref"), "{name}: fence_ref");
                assert_eq!(f.stages.get() as u64, expect_u64(case, "stages"), "{name}: stages");
            }
            render::OPCODE_USE_RESOURCE => {
                let (head, refs) = render::use_resource(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                let count = case["expect"]["count"].as_u64().unwrap_or(1);
                assert_eq!(head.count.get() as u64, count, "{name}: count");
                assert_eq!(refs.len() as u64, count, "{name}: ref count");
                assert_eq!(head.usage.get() as u64, expect_u64(case, "usage"), "{name}: usage");
                assert_eq!(head.stages.get() as u64, expect_u64(case, "stages"), "{name}: stages");
                if let Some(r) = case["expect"]["resource_ref"].as_u64() {
                    assert_eq!(refs[0].object_ref.get() as u64, r, "{name}: resource_ref");
                }
            }
            render::OPCODE_USE_HEAP => {
                let (head, refs) = render::use_heap(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                // The singular selector is the plural one at `count == 1`, and
                // it shares this opcode — so the count comes from the case, not
                // from a constant. The plural fixture is what shows the refs at
                // `+6` really are an array rather than one ref at an odd offset.
                let count = case["expect"]["count"].as_u64().unwrap_or(1);
                assert_eq!(head.count.get() as u64, count, "{name}: count");
                assert_eq!(head.stages.get() as u64, expect_u64(case, "stages"), "{name}: stages");
                assert_eq!(refs.len() as u64, count, "{name}: ref count");
                assert_eq!(refs[0].object_ref.get() as u64, expect_u64(case, "heap_ref"), "{name}: heap_ref");
                if let Some(r2) = case["expect"]["heap_ref_2"].as_u64() {
                    assert_eq!(refs[1].object_ref.get() as u64, r2, "{name}: heap_ref_2");
                }
            }
            render::OPCODE_SET_COLOR_STORE_ACTION => {
                let s = render::set_color_store_action(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_eq!(s.store_action.get() as u64, expect_u64(case, "store_action"), "{name}: store_action");
                assert_eq!(s.index.get() as u64, expect_u64(case, "index"), "{name}: index");
                assert_ne!(
                    s.store_action.get(),
                    s.index.get(),
                    "{name}: this case exists to tell the two fields apart and its \
                     two values are equal again"
                );
            }
            render::OPCODE_SET_STENCIL_REFERENCE => {
                let s = render::set_stencil_reference(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                // The one-argument selector records only `reference`, and the
                // claim being checked is that it lands in BOTH fields.
                let (front, back) = match case["expect"]["reference"].as_u64() {
                    Some(r) => (r, r),
                    None => (expect_u64(case, "front"), expect_u64(case, "back")),
                };
                assert_eq!(s.front.get() as u64, front, "{name}: front");
                assert_eq!(s.back.get() as u64, back, "{name}: back");
            }
            render::OPCODE_SET_DEPTH_BIAS => {
                let d = render::set_depth_bias(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                for (got, key) in [
                    (d.bias.get(), "bias"),
                    (d.slope_scale.get(), "slope_scale"),
                    (d.clamp.get(), "clamp"),
                ] {
                    let want = case["expect"][key].as_f64().unwrap_or_else(|| panic!("{name}: no expect.{key}"));
                    assert_eq!(got as f64, want, "{name}: {key}");
                }
            }
            render::OPCODE_SET_VISIBILITY_RESULT_MODE => {
                let v = render::set_visibility_result_mode(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_eq!(v.mode.get(), expect_u64(case, "mode"), "{name}: mode");
                assert_eq!(v.offset.get(), expect_u64(case, "offset"), "{name}: offset");
            }
            render::OPCODE_DRAW_INDIRECT => {
                let d = render::draw_indirect(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_eq!(d.primitive_type.get() as u64, expect_u64(case, "primitive_type"), "{name}: primitive_type");
                assert_eq!(d.indirect_buffer_ref.get() as u64, expect_u64(case, "indirect_buffer_ref"), "{name}: indirect_buffer_ref");
                assert_eq!(d.indirect_buffer_offset.get(), expect_u64(case, "indirect_buffer_offset"), "{name}: indirect_buffer_offset");
                // Two bytes past `primitive_type` are never written.
                assert_eq!(&bytes[bytes.len() - 2..], &[0xAA, 0xAA], "{name}: the tail is no longer unwritten");
            }
            render::OPCODE_DRAW_INDEXED_INDIRECT => {
                let d = render::draw_indexed_indirect(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_eq!(d.primitive_type.get() as u64, expect_u64(case, "primitive_type"), "{name}: primitive_type");
                assert_eq!(d.index_type.get() as u64, expect_u64(case, "index_type"), "{name}: index_type");
                assert_eq!(d.index_buffer_ref.get() as u64, expect_u64(case, "index_buffer_ref"), "{name}: index_buffer_ref");
                assert_eq!(d.index_buffer_offset.get(), expect_u64(case, "index_buffer_offset"), "{name}: index_buffer_offset");
                assert_eq!(d.indirect_buffer_ref.get() as u64, expect_u64(case, "indirect_buffer_ref"), "{name}: indirect_buffer_ref");
                assert_eq!(d.indirect_buffer_offset.get(), expect_u64(case, "indirect_buffer_offset"), "{name}: indirect_buffer_offset");
                assert_ne!(d.index_buffer_ref.get(), d.indirect_buffer_ref.get(), "{name}: the two buffers are indistinguishable in this case");
            }
            render::OPCODE_EXECUTE_COMMANDS_INDIRECT => {
                let e = render::execute_commands_indirect(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_eq!(e.icb_ref.get() as u64, expect_u64(case, "icb_ref"), "{name}: icb_ref");
                assert_eq!(e.indirect_buffer_ref.get() as u64, expect_u64(case, "indirect_buffer_ref"), "{name}: indirect_buffer_ref");
                assert_eq!(e.indirect_buffer_offset.get(), expect_u64(case, "indirect_buffer_offset"), "{name}: indirect_buffer_offset");
            }
            render::OPCODE_EXECUTE_COMMANDS_RANGE => {
                let e = render::execute_commands_range(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_eq!(e.icb_ref.get() as u64, expect_u64(case, "icb_ref"), "{name}: icb_ref");
                assert_eq!(e.range_location.get(), expect_u64(case, "range_location"), "{name}: range_location");
                assert_eq!(e.range_length.get(), expect_u64(case, "range_length"), "{name}: range_length");
            }
            render::OPCODE_MEMORY_BARRIER_RESOURCES => {
                let (h, refs) = render::memory_barrier_resources(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_eq!(h.count.get() as u64, expect_u64(case, "count"), "{name}: count");
                assert_eq!(h.after_stages.get() as u64, expect_u64(case, "after_stages"), "{name}: after_stages");
                assert_eq!(h.before_stages.get() as u64, expect_u64(case, "before_stages"), "{name}: before_stages");
                assert_eq!(refs.len() as u64, h.count.get() as u64, "{name}: the ref array is not `count` long");
                assert_eq!(refs[0].object_ref.get() as u64, expect_u64(case, "resource_ref"), "{name}: resource_ref");
                assert_eq!(refs[1].object_ref.get() as u64, expect_u64(case, "resource_ref_2"), "{name}: resource_ref_2");
            }
            render::OPCODE_MEMORY_BARRIER_SCOPE => {
                let b = render::memory_barrier_scope(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_eq!(b.scope as u64, expect_u64(case, "scope"), "{name}: scope");
                assert_eq!(b.after_stages as u64, expect_u64(case, "after_stages"), "{name}: after_stages");
                assert_eq!(b.before_stages as u64, expect_u64(case, "before_stages"), "{name}: before_stages");
                assert_eq!(b.unidentified_u8, 0, "{name}: unidentified_u8 is no longer 0");
            }
            render::OPCODE_TEXTURE_BARRIER => {
                assert!(
                    render::texture_barrier_has_no_payload(&o),
                    "{name}: textureBarrier grew a payload"
                );
            }
            render::OPCODE_SET_SCISSOR_RECTS => {
                let (h, rects) = render::set_scissor_rects(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_eq!(h.count.get(), expect_u64(case, "count"), "{name}: count");
                assert_eq!(rects.len() as u64, h.count.get(), "{name}: the rect array is not `count` long");
                for (i, suffix) in ["", "_2"].iter().enumerate() {
                    for (got, key) in [
                        (rects[i].x.get(), "x"),
                        (rects[i].y.get(), "y"),
                        (rects[i].width.get(), "width"),
                        (rects[i].height.get(), "height"),
                    ] {
                        let k = format!("{key}{suffix}");
                        assert_eq!(got, expect_u64(case, &k), "{name}: {k}");
                    }
                }
            }
            render::OPCODE_SET_VIEWPORTS => {
                let (h, ports) = render::set_viewports(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_eq!(h.count.get() as u64, expect_u64(case, "count"), "{name}: count");
                assert_eq!(ports.len() as u64, h.count.get() as u64, "{name}: the viewport array is not `count` long");
                for (i, suffix) in ["", "_2"].iter().enumerate() {
                    for (got, key) in [
                        (ports[i].origin_x.get(), "origin_x"),
                        (ports[i].origin_y.get(), "origin_y"),
                        (ports[i].width.get(), "width"),
                        (ports[i].height.get(), "height"),
                        (ports[i].znear.get(), "znear"),
                        (ports[i].zfar.get(), "zfar"),
                    ] {
                        let k = format!("{key}{suffix}");
                        let want = case["expect"][&k].as_f64().unwrap_or_else(|| panic!("{name}: no expect.{k}"));
                        assert_eq!(got, want, "{name}: {k}");
                    }
                }
            }
            render::OPCODE_SET_BLEND_COLOR => {
                let b = render::set_blend_color(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                for (got, key) in [
                    (b.red.get(), "red"),
                    (b.green.get(), "green"),
                    (b.blue.get(), "blue"),
                    (b.alpha.get(), "alpha"),
                ] {
                    let want = case["expect"][key].as_f64().unwrap_or_else(|| panic!("{name}: no expect.{key}"));
                    assert_eq!(got as f64, want, "{name}: {key}");
                }
            }
            render::OPCODE_SET_COLOR_STORE_ACTION_OPTIONS => {
                let a = render::set_color_store_action_options(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_eq!(a.options.get(), expect_u64(case, "options"), "{name}: options");
                assert_eq!(a.index.get() as u64, expect_u64(case, "index"), "{name}: index");
            }
            render::OPCODE_SET_DEPTH_STORE_ACTION_OPTIONS
            | render::OPCODE_SET_STENCIL_STORE_ACTION_OPTIONS => {
                let a = render::set_store_action_options(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_eq!(a.options.get(), expect_u64(case, "options"), "{name}: options");
            }
            render::OPCODE_SET_TESSELLATION_FACTOR_BUFFER => {
                let t = render::set_tessellation_factor_buffer(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_eq!(t.buffer_ref.get() as u64, expect_u64(case, "buffer_ref"), "{name}: buffer_ref");
                assert_eq!(t.offset.get(), expect_u64(case, "offset"), "{name}: offset");
                assert_eq!(t.instance_stride.get(), expect_u64(case, "instance_stride"), "{name}: instance_stride");
            }
            render_pass::OPCODE_RENDER_PASS => check_render_pass(name, case, &o),
            render_pass::OPCODE_DEFAULT_RASTER_SAMPLE_COUNT => {
                let d = render_pass::default_raster_sample_count(&o)
                    .unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_eq!(
                    d.count.get() as u64,
                    expect_u64(case, "default_raster_sample_count"),
                    "{name}: count"
                );
            }
            render_pass::OPCODE_RASTERIZATION_RATE_MAP => {
                let m = render_pass::pass_rate_map(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_eq!(
                    m.rate_map_ref.get() as u64,
                    expect_u64(case, "rate_map_ref"),
                    "{name}: rate_map_ref"
                );
            }
            render_pass::OPCODE_IMAGEBLOCK_SAMPLE_LENGTH => {
                let t = render_pass::tile_memory(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_eq!(
                    t.length.get() as u64,
                    expect_u64(case, "imageblock_sample_length"),
                    "{name}: imageblock_sample_length"
                );
            }
            render_pass::OPCODE_THREADGROUP_MEMORY_LENGTH => {
                let t = render_pass::tile_memory(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_eq!(
                    t.length.get() as u64,
                    expect_u64(case, "threadgroup_memory_length"),
                    "{name}: threadgroup_memory_length"
                );
            }
            render_pass::OPCODE_TILE_SIZE => {
                let t = render_pass::tile_size(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_eq!(
                    t.width.get() as u64,
                    expect_u64(case, "tile_width"),
                    "{name}: tile_width"
                );
                assert_eq!(
                    t.height.get() as u64,
                    expect_u64(case, "tile_height"),
                    "{name}: tile_height"
                );
            }
            render_pass::OPCODE_SAMPLE_POSITIONS => {
                let (head, positions) =
                    render_pass::sample_positions(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_eq!(
                    head.count.get() as u64,
                    expect_u64(case, "sample_position_count"),
                    "{name}: sample_position_count"
                );
                assert_eq!(
                    o.length(),
                    render_pass::SAMPLE_POSITIONS_HEAD_LEN
                        + head.count.get() * render_pass::SAMPLE_POSITION_LEN,
                    "{name}: the record's length is not head plus count positions"
                );
                // The positions the case asked for, in order. Read as pairs so
                // an x/y swap fails rather than averaging out.
                let want: &[(f32, f32)] = &[(0.25, 0.75), (0.125, 0.375)];
                assert_eq!(positions.len(), want.len(), "{name}: position count");
                for (i, (x, y)) in want.iter().enumerate() {
                    assert_eq!((positions[i].x(), positions[i].y()), (*x, *y), "{name}: position {i}");
                }
            }
            op if render::is_patch_draw(op) => check_patch_draw(name, case, &o),
            op if tile::is_tile_opcode(op) => check_tile_record(name, case, &o),
            other => panic!("{name}: fixture carries opcode {other:#x} with no view; add one or mark it Unimplemented"),
        }
        checked += 1;
    }
    assert!(checked > 0, "no render encoder cases in fixtures.json");

    // Every one of the twelve draw opcodes has to have been produced by a real
    // Metal call in this run. A layout with no fixture behind it is an
    // inference, and the manifest would be claiming otherwise.
    let all: std::collections::BTreeSet<u32> = (0x00..=0x0bu32).collect();
    assert_eq!(
        draw_opcodes_seen,
        all,
        "the draw fixtures no longer cover every opcode in 0x00..=0x0b; \
         missing {:?}",
        all.difference(&draw_opcodes_seen).collect::<Vec<_>>()
    );

    eprintln!("checked {checked} render encoder fixtures against Apple's serializer");
}

/// Every field of the render pass descriptor, against the descriptor itself.
///
/// Each case moved one property off a common baseline and read its expectations
/// back off `MTLRenderPassDescriptor`, so the assertions here cover *every*
/// case rather than only the one that moved a given field — a field that
/// stopped tracking its property fails on eighteen fixtures, not one.
///
/// The colour slots are checked for slot 0 and slot 3 by name, which is what
/// pins the 60-byte stride: an off-by-one stride puts slot 3's texture ref
/// where nothing wrote one.
fn check_render_pass(name: &str, case: &Value, o: &reims_vgpu_wire::op::Op<'_>) {
    use reims_vgpu_wire::ops::render_pass as rp;

    let p = rp::render_pass(o).unwrap_or_else(|e| panic!("{name}: {e}"));

    let c0 = &p.color[0];
    assert_eq!(
        c0.prefix.texture_ref.get() as u64,
        expect_u64(case, "color0_texture_ref"),
        "{name}: color0 texture_ref"
    );
    for (got, key) in [
        (c0.prefix.level.get(), "color0_level"),
        (c0.prefix.slice.get(), "color0_slice"),
        (c0.prefix.depth_plane.get(), "color0_depth_plane"),
        (c0.prefix.resolve_level.get(), "color0_resolve_level"),
        (c0.prefix.resolve_slice.get(), "color0_resolve_slice"),
        (
            c0.prefix.resolve_depth_plane.get(),
            "color0_resolve_depth_plane",
        ),
        (c0.prefix.load_action.get(), "color0_load_action"),
        (c0.prefix.store_action.get(), "color0_store_action"),
        (
            c0.prefix.store_action_options.get(),
            "color0_store_action_options",
        ),
    ] {
        assert_eq!(got as u64, expect_u64(case, key), "{name}: {key}");
    }
    for (got, key) in [
        (c0.clear_color()[0], "color0_clear_red"),
        (c0.clear_color()[1], "color0_clear_green"),
        (c0.clear_color()[2], "color0_clear_blue"),
        (c0.clear_color()[3], "color0_clear_alpha"),
    ] {
        assert_eq!(got, expect_f64(case, key), "{name}: {key}");
    }
    // Set only by the slot-three case; absent means the slot is unattached, and
    // an unattached slot must read a zero ref rather than the previous slot's.
    let want3 = case["expect"]
        .get("color3_texture_ref")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    assert_eq!(
        p.color[3].prefix.texture_ref.get() as u64,
        want3,
        "{name}: color3 texture_ref -- the 60-byte stride"
    );

    for (got, key) in [
        (p.depth.prefix.load_action.get(), "depth_load_action"),
        (p.depth.prefix.store_action.get(), "depth_store_action"),
        (p.depth.prefix.level.get(), "depth_level"),
        (p.depth.resolve_filter.get(), "depth_resolve_filter"),
        (p.stencil.prefix.load_action.get(), "stencil_load_action"),
        (p.stencil.prefix.store_action.get(), "stencil_store_action"),
        (p.stencil.resolve_filter.get(), "stencil_resolve_filter"),
    ] {
        assert_eq!(got as u64, expect_u64(case, key), "{name}: {key}");
    }
    assert_eq!(
        p.depth.clear_depth(),
        expect_f64(case, "clear_depth"),
        "{name}: clear_depth"
    );
    assert_eq!(
        p.stencil.clear_stencil.get() as u64,
        expect_u64(case, "clear_stencil"),
        "{name}: clear_stencil"
    );
    for (got, key) in [
        (p.depth.prefix.texture_ref.get() as u64, "depth_texture_ref"),
        (
            p.stencil.prefix.texture_ref.get() as u64,
            "stencil_texture_ref",
        ),
        (
            p.visibility_result_buffer_ref.get() as u64,
            "visibility_buffer_ref",
        ),
        (
            c0.prefix.resolve_texture_ref.get() as u64,
            "color0_resolve_texture_ref",
        ),
    ] {
        let want = case["expect"]
            .get(key)
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        assert_eq!(got, want, "{name}: {key}");
    }

    for (got, key) in [
        (
            p.render_target_array_length.get(),
            "render_target_array_length",
        ),
        (p.render_target_width.get(), "render_target_width"),
        (p.render_target_height.get(), "render_target_height"),
    ] {
        assert_eq!(got, expect_u64(case, key), "{name}: {key}");
    }

    // The four properties measured *not* to reach this record. Metal kept the
    // values the cases set -- the expectations were read back off the
    // descriptor -- so a non-zero one here with no byte moving is the negative
    // result, and it is asserted rather than remembered.
    for key in [
        "tile_width",
        "tile_height",
        "imageblock_sample_length",
        "threadgroup_memory_length",
    ] {
        assert!(
            case["expect"].get(key).is_some(),
            "{name}: no expect.{key}; the case stopped reading it off the descriptor"
        );
    }
}

/// Every field of every patch draw, against what Metal was asked for.
///
/// The compact and wide forms are checked through one set of expectations, so a
/// field that moved between them shows up as a mismatch rather than as two
/// tables that happen to agree with themselves. The `0x0c` arm asserts *which*
/// of the two records it is from the length before reading either body — that
/// dispatch is the whole hazard of this family.
/// The fields a patch draw carries, lifted out of whichever of the six records
/// this one is so the assertions below can be written once.
///
/// `instancing` and `control_point_index` are `Option` because the indirect
/// forms have no patch counts (they are in the indirect buffer) and the plain
/// forms have no control-point index buffer. A field that is absent is not the
/// same as a field that is zero, and reading it as zero would let a decoder
/// that dropped one pass.
#[derive(Debug)]
struct PatchDrawFields {
    patch_index_buffer_ref: u64,
    patch_index_buffer_offset: u64,
    control_points: u64,
    /// `(patch_start, patch_count, instance_count, base_instance)`.
    instancing: Option<(u64, u64, u64, u64)>,
    /// `(ref, offset)`.
    control_point_index: Option<(u64, u64)>,
}

fn check_patch_draw(name: &str, case: &Value, o: &reims_vgpu_wire::op::Op<'_>) {
    use reims_vgpu_wire::ops::render as r;

    let f = match o.opcode() {
        r::OPCODE_DRAW_PATCHES => {
            let d = r::draw_patches(o).unwrap_or_else(|e| panic!("{name}: {e}"));
            PatchDrawFields {
                patch_index_buffer_ref: d.patch_index_buffer_ref.get() as u64,
                patch_index_buffer_offset: d.patch_index_buffer_offset.get() as u64,
                control_points: d.control_points.get() as u64,
                instancing: Some((
                    d.patch_start.get() as u64,
                    d.patch_count.get() as u64,
                    d.instance_count.get() as u64,
                    d.base_instance.get() as u64,
                )),
                control_point_index: None,
            }
        }
        r::OPCODE_DRAW_INDEXED_PATCHES => {
            let d = r::draw_indexed_patches(o).unwrap_or_else(|e| panic!("{name}: {e}"));
            PatchDrawFields {
                patch_index_buffer_ref: d.patch_index_buffer_ref.get() as u64,
                patch_index_buffer_offset: d.patch_index_buffer_offset.get() as u64,
                control_points: d.control_points.get() as u64,
                instancing: Some((
                    d.patch_start.get() as u64,
                    d.patch_count.get() as u64,
                    d.instance_count.get() as u64,
                    d.base_instance.get() as u64,
                )),
                control_point_index: Some((
                    d.control_point_index_buffer_ref.get() as u64,
                    d.control_point_index_buffer_offset.get() as u64,
                )),
            }
        }
        r::OPCODE_DRAW_PATCHES_WIDE => {
            // The length is the discriminator, and the crate must agree with
            // the serializer about which record this is before anything is read
            // out of it. That dispatch is the whole hazard of this family.
            let indexed = r::patch_draw_wide_is_indexed(o).unwrap_or_else(|| {
                panic!("{name}: 0x0c at {} bytes, neither wide form", o.length())
            });
            assert_eq!(
                indexed,
                name.contains("indexed"),
                "{name}: the length picked the wrong one of 0x0c's two records"
            );
            if indexed {
                let d = r::draw_indexed_patches_wide(o).unwrap_or_else(|e| panic!("{name}: {e}"));
                PatchDrawFields {
                    patch_index_buffer_ref: d.patch_index_buffer_ref.get() as u64,
                    patch_index_buffer_offset: d.patch_index_buffer_offset.get(),
                    control_points: d.control_points.get() as u64,
                    instancing: Some((
                        d.patch_start.get(),
                        d.patch_count.get(),
                        d.instance_count.get(),
                        d.base_instance.get(),
                    )),
                    control_point_index: Some((
                        d.control_point_index_buffer_ref.get() as u64,
                        d.control_point_index_buffer_offset.get(),
                    )),
                }
            } else {
                let d = r::draw_patches_wide(o).unwrap_or_else(|e| panic!("{name}: {e}"));
                PatchDrawFields {
                    patch_index_buffer_ref: d.patch_index_buffer_ref.get() as u64,
                    patch_index_buffer_offset: d.patch_index_buffer_offset.get(),
                    control_points: d.control_points.get() as u64,
                    instancing: Some((
                        d.patch_start.get(),
                        d.patch_count.get(),
                        d.instance_count.get(),
                        d.base_instance.get(),
                    )),
                    control_point_index: None,
                }
            }
        }
        r::OPCODE_DRAW_PATCHES_INDIRECT => {
            let d = r::draw_patches_indirect(o).unwrap_or_else(|e| panic!("{name}: {e}"));
            check_patch_indirect_buffer(
                name,
                case,
                d.indirect_buffer_ref.get() as u64,
                d.indirect_buffer_offset.get(),
            );
            PatchDrawFields {
                patch_index_buffer_ref: d.patch_index_buffer_ref.get() as u64,
                patch_index_buffer_offset: d.patch_index_buffer_offset.get(),
                control_points: d.control_points.get() as u64,
                instancing: None,
                control_point_index: None,
            }
        }
        r::OPCODE_DRAW_INDEXED_PATCHES_INDIRECT => {
            let d = r::draw_indexed_patches_indirect(o).unwrap_or_else(|e| panic!("{name}: {e}"));
            check_patch_indirect_buffer(
                name,
                case,
                d.indirect_buffer_ref.get() as u64,
                d.indirect_buffer_offset.get(),
            );
            PatchDrawFields {
                patch_index_buffer_ref: d.patch_index_buffer_ref.get() as u64,
                patch_index_buffer_offset: d.patch_index_buffer_offset.get(),
                control_points: d.control_points.get() as u64,
                instancing: None,
                control_point_index: Some((
                    d.control_point_index_buffer_ref.get() as u64,
                    d.control_point_index_buffer_offset.get(),
                )),
            }
        }
        other => panic!("{name}: patch opcode {other:#x} claimed by is_patch_draw with no arm"),
    };

    assert_eq!(
        f.control_points,
        expect_u64(case, "control_points"),
        "{name}: control_points -- it trails the record, reversing the selector"
    );
    assert_eq!(
        f.patch_index_buffer_ref,
        expect_u64(case, "patch_index_buffer_ref"),
        "{name}: patch_index_buffer_ref"
    );
    assert_eq!(
        f.patch_index_buffer_offset,
        expect_u64(case, "patch_index_buffer_offset"),
        "{name}: patch_index_buffer_offset"
    );
    if let Some((cp_ref, cp_off)) = f.control_point_index {
        assert_eq!(
            cp_ref,
            expect_u64(case, "control_point_index_buffer_ref"),
            "{name}: control_point_index_buffer_ref"
        );
        assert_eq!(
            cp_off,
            expect_u64(case, "control_point_index_buffer_offset"),
            "{name}: control_point_index_buffer_offset"
        );
    }
    if let Some((start, count, inst, base)) = f.instancing {
        assert_eq!(
            start,
            expect_u64(case, "patch_start"),
            "{name}: patch_start"
        );
        assert_eq!(
            count,
            expect_u64(case, "patch_count"),
            "{name}: patch_count"
        );
        assert_eq!(
            inst,
            expect_u64(case, "instance_count"),
            "{name}: instance_count"
        );
        assert_eq!(
            base,
            expect_u64(case, "base_instance"),
            "{name}: base_instance"
        );

        // The compact forms narrow all four to 16 bits, so the serializer must
        // switch to the wide opcode for anything above -- and must not use it
        // for anything that fits, or the pairing is not a rule.
        let wide = [start, count, inst, base].iter().any(|&v| v > 0xffff);
        assert_eq!(
            wide,
            o.opcode() == r::OPCODE_DRAW_PATCHES_WIDE,
            "{name}: the compact/wide choice does not follow the 16-bit rule"
        );
    }
}

/// The indirect pair's buffer, checked identically in both arms.
fn check_patch_indirect_buffer(name: &str, case: &Value, buffer_ref: u64, offset: u64) {
    assert_eq!(
        buffer_ref,
        expect_u64(case, "indirect_buffer_ref"),
        "{name}: indirect_buffer_ref"
    );
    assert_eq!(
        offset,
        expect_u64(case, "indirect_buffer_offset"),
        "{name}: indirect_buffer_offset"
    );
}

/// Every field of every tile record, against what Metal was asked for.
///
/// Split out of the render match rather than inlined because the tile family is
/// nine opcodes of its own module; folding it in would have put a second
/// hundred-line match inside the first.
fn check_tile_record(name: &str, case: &Value, o: &reims_vgpu_wire::op::Op<'_>) {
    use reims_vgpu_wire::ops::tile;

    match o.opcode() {
        tile::OPCODE_DISPATCH_THREADS_PER_TILE => {
            let d = tile::dispatch_threads_per_tile(o).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(d.width.get(), expect_u64(case, "width"), "{name}: width");
            assert_eq!(d.height.get(), expect_u64(case, "height"), "{name}: height");
            assert_eq!(d.depth.get(), expect_u64(case, "depth"), "{name}: depth");
        }
        op if tile::is_dispatch_threads_per_tile_in_region(op) => {
            let d = tile::dispatch_threads_per_tile_in_region(o)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            for (got, key) in [
                (d.width.get(), "width"),
                (d.height.get(), "height"),
                (d.depth.get(), "depth"),
                (d.origin_x.get(), "origin_x"),
                (d.origin_y.get(), "origin_y"),
                (d.origin_z.get(), "origin_z"),
                (d.region_width.get(), "region_width"),
                (d.region_height.get(), "region_height"),
                (d.region_depth.get(), "region_depth"),
            ] {
                assert_eq!(got, expect_u64(case, key), "{name}: {key}");
            }

            // The trailing index, present on `0xa3` and absent on `0xa2`. The
            // case's own expectations say which this should be, so a view that
            // handed the bytes back for both would fail here rather than pass
            // by reading a zero the arena happened to leave.
            let got = tile::dispatch_threads_per_tile_region_rt_index(o);
            match case["expect"]["render_target_array_index"].as_u64() {
                Some(want) => assert_eq!(
                    got.map(u64::from),
                    Some(want),
                    "{name}: render_target_array_index"
                ),
                None => assert_eq!(
                    got, None,
                    "{name}: the plain region form handed back a render-target array \
                     index; those four bytes are unwritten and are the guest's ring"
                ),
            }
        }
        tile::OPCODE_SET_TILE_THREADGROUP_MEMORY => {
            let m = tile::tile_threadgroup_memory(o).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(m.length.get(), expect_u64(case, "length"), "{name}: length");
            assert_eq!(m.offset.get(), expect_u64(case, "offset"), "{name}: offset");
            assert_eq!(
                m.index.get() as u64,
                expect_u64(case, "index"),
                "{name}: index"
            );
        }
        tile::OPCODE_SET_TILE_BUFFER_OFFSET => {
            let b = tile::tile_buffer_offset(o).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(
                b.index.get() as u64,
                expect_u64(case, "index"),
                "{name}: index"
            );
            assert_eq!(b.offset.get(), expect_u64(case, "offset"), "{name}: offset");
        }
        tile::OPCODE_GET_TILE_DIMENSIONS => {
            let g = tile::get_tile_dimensions(o).unwrap_or_else(|e| panic!("{name}: {e}"));
            // The ref and offset are the staging buffer's, which is what makes
            // this a readback rather than a bind — no argument of the call
            // supplied either.
            assert_eq!(
                g.buffer_ref.get(),
                8181,
                "{name}: getTileDimensions: no longer names the staging buffer"
            );
            assert_eq!(
                g.offset.get(),
                0x9999,
                "{name}: getTileDimensions: no longer names the staging offset"
            );
        }
        tile::OPCODE_SET_TILE_BUFFER => {
            let (h, entries) = tile::tile_buffer_binds(o).unwrap_or_else(|e| panic!("{name}: {e}"));
            let first = sole_key(case, &["first", "index"]);
            assert_eq!(
                h.first.get() as u64,
                expect_u64(case, first),
                "{name}: first"
            );
            assert_eq!(
                entries.len() as u64,
                h.count.get() as u64,
                "{name}: entry count"
            );
            for (i, suffix) in ["", "_2"].iter().enumerate().take(entries.len()) {
                assert_eq!(
                    entries[i].buffer_ref.get() as u64,
                    expect_u64(case, &format!("buffer_ref{suffix}")),
                    "{name}: buffer_ref{suffix}"
                );
                assert_eq!(
                    entries[i].offset.get(),
                    expect_u64(case, &format!("offset{suffix}")),
                    "{name}: offset{suffix}"
                );
            }
        }
        op @ (tile::OPCODE_SET_TILE_TEXTURE | tile::OPCODE_SET_TILE_SAMPLER) => {
            let (h, entries) = if op == tile::OPCODE_SET_TILE_TEXTURE {
                tile::tile_texture_binds(o)
            } else {
                tile::tile_sampler_binds(o)
            }
            .unwrap_or_else(|e| panic!("{name}: {e}"));
            let ref_key = if op == tile::OPCODE_SET_TILE_TEXTURE {
                "texture_ref"
            } else {
                "sampler_ref"
            };
            let first = sole_key(case, &["first", "index"]);
            assert_eq!(
                h.first.get() as u64,
                expect_u64(case, first),
                "{name}: first"
            );
            assert_eq!(
                entries.len() as u64,
                h.count.get() as u64,
                "{name}: entry count"
            );
            for (i, suffix) in ["", "_2"].iter().enumerate().take(entries.len()) {
                // The plural sampler case binds the same stub twice, so its
                // second entry has no `_2` expectation of its own.
                let key = format!("{ref_key}{suffix}");
                let want = case["expect"][&key]
                    .as_u64()
                    .unwrap_or_else(|| expect_u64(case, ref_key));
                assert_eq!(entries[i].object_ref.get() as u64, want, "{name}: {key}");
            }
        }
        tile::OPCODE_SET_TILE_SAMPLER_LOD => {
            let (h, entries) =
                tile::tile_sampler_lod_binds(o).unwrap_or_else(|e| panic!("{name}: {e}"));
            let first = sole_key(case, &["first", "index"]);
            assert_eq!(
                h.first.get() as u64,
                expect_u64(case, first),
                "{name}: first"
            );
            assert_eq!(
                entries.len() as u64,
                h.count.get() as u64,
                "{name}: entry count"
            );
            for (i, suffix) in ["", "_2"].iter().enumerate().take(entries.len()) {
                assert_eq!(
                    entries[i].sampler_ref.get() as u64,
                    expect_u64(case, "sampler_ref"),
                    "{name}: sampler_ref"
                );
                for (got, key) in [
                    (entries[i].lod_min_clamp.get(), "lod_min_clamp"),
                    (entries[i].lod_max_clamp.get(), "lod_max_clamp"),
                ] {
                    let k = format!("{key}{suffix}");
                    let want = case["expect"][&k]
                        .as_f64()
                        .unwrap_or_else(|| panic!("{name}: no expect.{k}"));
                    assert_eq!(got as f64, want, "{name}: {k}");
                }
            }
        }
        other => panic!("{name}: tile opcode {other:#x} claimed by is_tile_opcode with no arm"),
    }
}

/// The blit encoder's record length the crate's constants claim for an opcode.
///
/// Two of these lengths are *longer* than the body they name — `fillBuffer:`
/// and `copyFromTexture:toBuffer:` both leave bytes at the end of the record
/// unwritten — so this maps opcode to the serializer's allocation, which is
/// what the header declares, not to `size_of` the view.
fn blit_record_len_for(opcode: u32) -> Option<u32> {
    use reims_vgpu_wire::ops::blit as b;
    Some(match opcode {
        b::OPCODE_COPY_ICB => b::COPY_ICB_TOTAL_LEN,
        b::OPCODE_FILL_BUFFER => b::FILL_BUFFER_TOTAL_LEN,
        b::OPCODE_COPY_BUFFER_TO_BUFFER => b::COPY_BUFFER_TO_BUFFER_TOTAL_LEN,
        b::OPCODE_COPY_TEXTURE_SLICES => b::COPY_TEXTURE_SLICES_TOTAL_LEN,
        b::OPCODE_COPY_TEXTURE_REGION => b::COPY_TEXTURE_REGION_TOTAL_LEN,
        b::OPCODE_COPY_TEXTURE_REGION_OPTIONS => b::COPY_TEXTURE_REGION_OPTIONS_TOTAL_LEN,
        b::OPCODE_COPY_BUFFER_TO_TEXTURE => b::COPY_BUFFER_TO_TEXTURE_TOTAL_LEN,
        b::OPCODE_COPY_TEXTURE_TO_BUFFER => b::COPY_TEXTURE_TO_BUFFER_TOTAL_LEN,
        b::OPCODE_FILL_BUFFER_PATTERN4 => b::FILL_BUFFER_PATTERN4_TOTAL_LEN,
        b::OPCODE_FILL_TEXTURE_COLOR => b::FILL_TEXTURE_COLOR_TOTAL_LEN,
        b::OPCODE_FILL_TEXTURE_BYTES => b::FILL_TEXTURE_BYTES_TOTAL_LEN,
        op if b::is_ref(op) => b::REF_TOTAL_LEN,
        op if b::is_ref_slice_level(op) => b::REF_SLICE_LEVEL_TOTAL_LEN,
        op if b::is_icb_range(op) => b::ICB_RANGE_TOTAL_LEN,
        _ => return None,
    })
}

#[test]
#[cfg_attr(not(wire_fixtures), ignore = "run scripts/wire-oracle/wire-oracle.sh")]
fn every_blit_fixture_reads_back_what_metal_was_asked_for() {
    use reims_vgpu_wire::ops::blit;

    let root = fixtures();
    let mut checked = 0usize;
    // The `command:` argument of the three generic emitters, and the opcode
    // Apple wrote when it was passed. Collected rather than asserted per case,
    // because the claim is about the *relationship* between the two.
    let mut commanded: Vec<(u64, u32)> = Vec::new();

    for case in root["cases"].as_array().expect("cases array") {
        if case["class"] != "PGSerializerBlitCommandEncoder" {
            continue;
        }
        let name = case["name"].as_str().expect("case name");
        let selector = case["selector"].as_str().expect("case selector");
        let bytes = unhex(case["buffer"].as_str().expect("buffer hex"));
        let allocated = case["allocated_len"].as_u64().expect("allocated_len");

        // Segment framing rather than a command: no opcode, so it cannot go
        // through `op()`. Checked as a family by
        // `every_segment_header_fixture_reads_back_what_the_encoder_wrote`.
        if selector == "beginSegment:protectionOptions:" {
            continue;
        }

        let o = op(&bytes, 0).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            o.length() as u64,
            allocated,
            "{name}: length disagrees with the serializer's own allocation"
        );

        // The three `withCommand:` selectors put their argument in the opcode
        // field, so their records carry an opcode this crate never named. They
        // are checked below as a family instead of against a constant.
        if selector.contains("withCommand:") {
            commanded.push((expect_u64(case, "command"), o.opcode()));
            continue;
        }

        assert_eq!(
            blit_record_len_for(o.opcode()),
            Some(o.length()),
            "{name}: opcode {:#x} came in at {} bytes; the crate's constant says otherwise",
            o.opcode(),
            o.length()
        );

        macro_rules! eq {
            ($got:expr, $key:literal) => {
                assert_eq!($got as u64, expect_u64(case, $key), "{}: {}", name, $key)
            };
        }

        match o.opcode() {
            op if blit::is_ref(op) => {
                let r = blit::object_ref(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                // One shape, six opcodes, and the expectation key names which
                // kind of object the opcode implies — so a record decoded under
                // the wrong opcode fails on a missing key rather than passing.
                let key = sole_key(
                    case,
                    &["fence_ref", "resource_ref", "texture_ref", "buffer_ref"],
                );
                assert_eq!(
                    r.object_ref.get() as u64,
                    expect_u64(case, key),
                    "{name}: {key}"
                );
            }
            op if blit::is_ref_slice_level(op) => {
                let r = blit::ref_slice_level(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                eq!(r.texture_ref.get(), "texture_ref");
                eq!(r.slice.get(), "slice");
                eq!(r.level.get(), "level");
                assert_ne!(
                    r.slice.get(),
                    r.level.get(),
                    "{name}: slice and level are equal, so this case cannot tell them apart"
                );
            }
            op if blit::is_icb_range(op) => {
                let r = blit::icb_range(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                eq!(r.icb_ref.get(), "icb_ref");
                eq!(r.range_location.get(), "range_location");
                eq!(r.range_length.get(), "range_length");
            }
            blit::OPCODE_COPY_ICB => {
                let r = blit::copy_icb(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                eq!(r.source_ref.get(), "source_ref");
                eq!(r.dest_ref.get(), "dest_ref");
                eq!(r.range_location.get(), "range_location");
                eq!(r.range_length.get(), "range_length");
                eq!(r.dest_index.get(), "dest_index");
            }
            blit::OPCODE_FILL_BUFFER => {
                let r = blit::fill_buffer(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                eq!(r.buffer_ref.get(), "buffer_ref");
                eq!(r.range_location.get(), "range_location");
                eq!(r.range_length.get(), "range_length");
                eq!(r.value, "value");
            }
            blit::OPCODE_FILL_BUFFER_PATTERN4 => {
                let r = blit::fill_buffer_pattern4(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                eq!(r.buffer_ref.get(), "buffer_ref");
                eq!(r.range_location.get(), "range_location");
                eq!(r.range_length.get(), "range_length");
                eq!(r.pattern.get(), "pattern4");
            }
            blit::OPCODE_FILL_TEXTURE_COLOR => {
                let r = blit::fill_texture_color(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                eq!(r.texture_ref.get(), "texture_ref");
                eq!(r.level.get(), "level");
                eq!(r.slice.get(), "slice");
                eq!(r.size_width.get(), "size_width");
                eq!(r.size_height.get(), "size_height");
                eq!(r.size_depth.get(), "size_depth");
                eq!(r.origin_x.get(), "origin_x");
                eq!(r.origin_y.get(), "origin_y");
                eq!(r.origin_z.get(), "origin_z");
                // The clear colour is four `double`s and the only float field
                // in this class. Compared exactly: every component is a value
                // with an exact binary representation, so a rounding-tolerant
                // comparison here would hide a swapped pair rather than a
                // precision loss. Every case carries all four, and no two
                // components of any case are equal, so a record that wrote one
                // into all four slots could not read back correct.
                for (key, got) in [
                    ("color_red", r.color_red.get()),
                    ("color_green", r.color_green.get()),
                    ("color_blue", r.color_blue.get()),
                    ("color_alpha", r.color_alpha.get()),
                ] {
                    assert_eq!(got, expect_f64(case, key), "{name}: {key}");
                }
                // Two selectors write this record and the format word is the
                // only thing that differs: the `pixelFormat:` form carries the
                // argument, the plain form carries the *texture's* format. The
                // key names which, so a case that took it from the wrong place
                // fails on a missing key rather than passing on a coincidence.
                let key = sole_key(case, &["pixel_format", "texture_pixel_format"]);
                assert_eq!(
                    r.pixel_format.get() as u64,
                    expect_u64(case, key),
                    "{name}: {key}"
                );
            }
            blit::OPCODE_FILL_TEXTURE_BYTES => {
                let r = blit::fill_texture_bytes(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                eq!(r.texture_ref.get(), "texture_ref");
                eq!(r.level.get(), "level");
                eq!(r.slice.get(), "slice");
                eq!(r.size_width.get(), "size_width");
                eq!(r.size_height.get(), "size_height");
                eq!(r.size_depth.get(), "size_depth");
                eq!(r.origin_x.get(), "origin_x");
                eq!(r.origin_y.get(), "origin_y");
                eq!(r.origin_z.get(), "origin_z");
                // The pattern does not travel inline: the serializer stages it
                // and names the staging buffer, exactly as the five
                // `setBytes:` records do.
                eq!(r.bytes_ref.get(), "bytes_ref");
                eq!(r.bytes_offset.get(), "bytes_offset");
                eq!(r.length.get(), "length");
            }
            blit::OPCODE_COPY_BUFFER_TO_BUFFER => {
                let r = blit::copy_buffer_to_buffer(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                eq!(r.source_ref.get(), "source_ref");
                eq!(r.dest_ref.get(), "dest_ref");
                eq!(r.source_offset.get(), "source_offset");
                eq!(r.dest_offset.get(), "dest_offset");
                eq!(r.size.get(), "size");
            }
            blit::OPCODE_COPY_TEXTURE_SLICES => {
                let r = blit::copy_texture_slices(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                eq!(r.source_ref.get(), "source_ref");
                eq!(r.dest_ref.get(), "dest_ref");
                eq!(r.source_slice.get(), "source_slice");
                eq!(r.source_level.get(), "source_level");
                eq!(r.dest_slice.get(), "dest_slice");
                eq!(r.dest_level.get(), "dest_level");
                eq!(r.slice_count.get(), "slice_count");
                eq!(r.level_count.get(), "level_count");
            }
            blit::OPCODE_COPY_TEXTURE_REGION => {
                let r = blit::copy_texture_region(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                check_texture_region(name, r, case);
            }
            blit::OPCODE_COPY_TEXTURE_REGION_OPTIONS => {
                let r =
                    blit::copy_texture_region_options(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                check_texture_region(name, &r.region, case);
                eq!(r.options.get(), "options");
            }
            blit::OPCODE_COPY_BUFFER_TO_TEXTURE => {
                let r = blit::copy_buffer_to_texture(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                eq!(r.source_ref.get(), "source_ref");
                eq!(r.dest_ref.get(), "dest_ref");
                eq!(r.source_offset.get(), "source_offset");
                eq!(r.source_bytes_per_row.get(), "source_bytes_per_row");
                eq!(r.source_bytes_per_image.get(), "source_bytes_per_image");
                eq!(r.size_width.get(), "size_width");
                eq!(r.size_height.get(), "size_height");
                eq!(r.size_depth.get(), "size_depth");
                eq!(r.dest_origin_x.get(), "dest_origin_x");
                eq!(r.dest_origin_y.get(), "dest_origin_y");
                eq!(r.dest_origin_z.get(), "dest_origin_z");
                eq!(r.dest_slice.get(), "dest_slice");
                eq!(r.dest_level.get(), "dest_level");
                // Absent in the plain form's expectations, where the claim is
                // that the field is present in the record and reads zero.
                assert_eq!(
                    r.options.get() as u64,
                    case["expect"]["options"].as_u64().unwrap_or(0),
                    "{name}: options"
                );
            }
            blit::OPCODE_COPY_TEXTURE_TO_BUFFER => {
                let r = blit::copy_texture_to_buffer(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                eq!(r.source_ref.get(), "source_ref");
                eq!(r.dest_ref.get(), "dest_ref");
                eq!(r.source_origin_x.get(), "source_origin_x");
                eq!(r.source_origin_y.get(), "source_origin_y");
                eq!(r.source_origin_z.get(), "source_origin_z");
                eq!(r.size_width.get(), "size_width");
                eq!(r.size_height.get(), "size_height");
                eq!(r.size_depth.get(), "size_depth");
                eq!(r.dest_offset.get(), "dest_offset");
                eq!(r.dest_bytes_per_row.get(), "dest_bytes_per_row");
                eq!(r.dest_bytes_per_image.get(), "dest_bytes_per_image");
                eq!(r.source_slice.get(), "source_slice");
                eq!(r.source_level.get(), "source_level");
                assert_eq!(
                    r.options.get() as u64,
                    case["expect"]["options"].as_u64().unwrap_or(0),
                    "{name}: options"
                );
                // The two bytes after `options` are the serializer's, unwritten.
                // They must still be the oracle's poison, because a view that
                // grew into them would be reading a guest's stale ring bytes.
                assert_eq!(
                    &bytes[bytes.len() - 2..],
                    &[0xAA, 0xAA],
                    "{name}: the tail past `options` is no longer unwritten -- \
                     either the record grew a field or the arena stopped being poisoned"
                );
            }
            other => panic!(
                "{name}: fixture carries opcode {other:#x} with no view; \
                 add one or mark it Unimplemented"
            ),
        }
        checked += 1;
    }
    assert!(checked > 0, "no blit encoder cases in fixtures.json");

    // The `withCommand:` claim, as an executable statement: the opcode Apple
    // wrote equals the argument passed, and at least two distinct arguments
    // were passed. One value alone would be satisfied by a fixed opcode that
    // happened to equal it.
    assert!(
        commanded.len() >= 2,
        "the withCommand: family needs at least two different arguments to show \
         the opcode follows them; got {}",
        commanded.len()
    );
    for (command, opcode) in &commanded {
        assert_eq!(
            *command, *opcode as u64,
            "withCommand: {command:#x} produced opcode {opcode:#x}; the argument is \
             no longer the opcode"
        );
    }
    let distinct: std::collections::BTreeSet<u64> = commanded.iter().map(|(c, _)| *c).collect();
    assert!(
        distinct.len() >= 2,
        "every withCommand: case passed the same argument, so the opcode could be fixed"
    );

    eprintln!("checked {checked} blit encoder fixtures against Apple's serializer");
}

/// The nine `u64` and four `u16` shared by the plain and `options:` forms of the
/// region copy. Split out so both arms check the same fields rather than one of
/// them checking a subset.
fn check_texture_region(
    name: &str,
    r: &reims_vgpu_wire::ops::blit::CopyTextureRegion,
    case: &Value,
) {
    for (got, key) in [
        (r.source_origin_x.get(), "source_origin_x"),
        (r.source_origin_y.get(), "source_origin_y"),
        (r.source_origin_z.get(), "source_origin_z"),
        (r.size_width.get(), "size_width"),
        (r.size_height.get(), "size_height"),
        (r.size_depth.get(), "size_depth"),
        (r.dest_origin_x.get(), "dest_origin_x"),
        (r.dest_origin_y.get(), "dest_origin_y"),
        (r.dest_origin_z.get(), "dest_origin_z"),
    ] {
        assert_eq!(got, expect_u64(case, key), "{name}: {key}");
    }
    for (got, key) in [
        (r.source_ref.get(), "source_ref"),
        (r.dest_ref.get(), "dest_ref"),
    ] {
        assert_eq!(got as u64, expect_u64(case, key), "{name}: {key}");
    }
    for (got, key) in [
        (r.source_slice.get(), "source_slice"),
        (r.source_level.get(), "source_level"),
        (r.dest_slice.get(), "dest_slice"),
        (r.dest_level.get(), "dest_level"),
    ] {
        assert_eq!(got as u64, expect_u64(case, key), "{name}: {key}");
    }
}

/// The segment header, across every encoder class that has been driven.
///
/// This is where the claim "the byte at `+4` is a type" is executable. It is
/// derived from a *difference*: the same call with the same arguments on two
/// classes puts two different values there. So the test collects the value each
/// class produced and asserts they disagree — one class alone could not tell a
/// type from a constant, and if a future capture made them agree the derivation
/// would be gone while every per-field assertion still passed.
#[test]
#[cfg_attr(not(wire_fixtures), ignore = "run scripts/wire-oracle/wire-oracle.sh")]
fn every_segment_header_fixture_reads_back_what_the_encoder_wrote() {
    use reims_vgpu_wire::ops::segment;

    let root = fixtures();
    let mut by_class: std::collections::BTreeMap<&str, u8> = std::collections::BTreeMap::new();
    let mut checked = 0usize;
    let mut envelope_payloads: std::collections::BTreeSet<u64> = Default::default();
    let mut envelope_types: std::collections::BTreeSet<u8> = Default::default();
    let mut continuation_pair: std::collections::BTreeMap<&str, (bool, bool)> = Default::default();

    for case in root["cases"].as_array().expect("cases array") {
        if case["selector"] != "beginSegment:protectionOptions:" {
            continue;
        }
        let name = case["name"].as_str().expect("case name");
        let class = case["class"].as_str().expect("case class");
        let bytes = unhex(case["buffer"].as_str().expect("buffer hex"));

        assert_eq!(
            case["allocated_len"].as_u64().expect("allocated_len"),
            segment::SEGMENT_HEADER_LEN as u64,
            "{name}: the segment header is no longer {} bytes",
            segment::SEGMENT_HEADER_LEN
        );

        // The middle record of the protection-options envelope is not a header;
        // it is the eight bytes of `protectionOptions:`. It is the same length
        // as a header and would read as one whose `length` is the guest's
        // options, which is exactly the misreading the burst invites — so it is
        // recognised by position, `_1` of a three-record split, and asserted as
        // itself.
        if name.ends_with("_1") && case["expect"].get("flag").is_none() {
            let e = segment::protection_options_envelope(&bytes)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(
                e.protection_options.get(),
                expect_u64(case, "protection_options"),
                "{name}: protection_options"
            );
            envelope_payloads.insert(e.protection_options.get());
            checked += 1;
            continue;
        }

        let h = segment::segment_header(&bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            h.length.get(),
            0,
            "{name}: the length is backfilled by -endEncoding, so it must still \
             read 0 at -beginSegment: time"
        );
        let continues_previous = expect_u64(case, "flag") != 0;
        let continues_next = case["expect"]
            .get("continues_next")
            .and_then(serde_json::Value::as_u64)
            .is_some_and(|value| value != 0);
        assert_eq!(
            h.continues_previous(),
            continues_previous,
            "{name}: continues_previous"
        );
        assert_eq!(h.continues_next(), continues_next, "{name}: continues_next");
        if name.starts_with("blit_segment_continuation_pair_") {
            continuation_pair.insert(name, (h.continues_previous(), h.continues_next()));
        }

        // The eighth byte is never written. If a view ever grew into it, a real
        // guest would be reading whatever its ring last held.
        assert_eq!(
            bytes[segment::SEGMENT_HEADER_LEN - 1],
            0xAA,
            "{name}: the last byte of the header is no longer unwritten"
        );

        // `protectionOptions:` does not reach the header. Two cases pass
        // different values; if either ever appeared in these bytes this would
        // catch it.
        // Zero is skipped, and the reason is the same one the ICB ref scan
        // learned: a search for a byte value has to exclude the values it cannot
        // distinguish. Every header holds zero bytes by construction, so
        // `protectionOptions == 0` makes this assertion fire on a header that
        // carries the field no more than the others do.
        let protection = expect_u64(case, "protection_options");
        assert!(
            protection == 0 || !bytes.contains(&(protection as u8)),
            "{name}: protectionOptions {protection:#x} now appears in the header; \
             it used to reach no field"
        );

        // The envelope's leading header is on the blit class and carries type 5,
        // so it must not join the per-class fold below — that fold's whole claim
        // is "one class, one type", and this is one class writing a *second*
        // type deliberately.
        if h.segment_type == segment::SEGMENT_TYPE_PROTECTION_OPTIONS {
            envelope_types.insert(h.segment_type);
            checked += 1;
            continue;
        }
        if let Some(seen) = by_class.insert(class, h.segment_type) {
            assert_eq!(
                seen, h.segment_type,
                "{name}: two cases on {class} produced different segment types"
            );
        }
        checked += 1;
    }

    assert!(checked > 0, "no beginSegment: cases in fixtures.json");
    assert_eq!(
        continuation_pair.get("blit_segment_continuation_pair_0"),
        Some(&(false, true)),
        "the first header must leave its encoder open for the paired continuation"
    );
    assert_eq!(
        continuation_pair.get("blit_segment_continuation_pair_1"),
        Some(&(true, false)),
        "the second header must continue the first and then close the encoder"
    );

    // The envelope, and the reason it is asserted here rather than left to the
    // fixtures: two distinct payloads is what separates "the guest's options
    // reach the wire" from "a constant does".
    assert!(
        envelope_types.contains(&segment::SEGMENT_TYPE_PROTECTION_OPTIONS),
        "no protection-options envelope header was captured; the burst is driven \
         under `-setSupportsProtectionOptionsEnvelope:` with the BOOL clear and \
         non-zero options, and needs both"
    );
    assert!(
        envelope_payloads.len() > 1,
        "every envelope carried the same payload ({envelope_payloads:?}), so this \
         cannot tell the guest's options from a constant"
    );
    assert_eq!(
        by_class.get("PGSerializerRenderCommandEncoder"),
        Some(&segment::SEGMENT_TYPE_RENDER),
        "the render encoder no longer writes SEGMENT_TYPE_RENDER"
    );
    assert_eq!(
        by_class.get("PGSerializerBlitCommandEncoder"),
        Some(&segment::SEGMENT_TYPE_BLIT),
        "the blit encoder no longer writes SEGMENT_TYPE_BLIT"
    );
    assert_eq!(
        by_class.get("PGSerializerComputeCommandEncoder"),
        Some(&segment::SEGMENT_TYPE_COMPUTE),
        "the compute encoder no longer writes SEGMENT_TYPE_COMPUTE"
    );
    // The one type that is not the next number in sequence, which is why it is
    // worth naming separately: the class before it in this list writes 2 and
    // this one writes 4, so nothing about the value could have been guessed.
    assert_eq!(
        by_class.get("PGSerializerInfoCommandEncoder"),
        Some(&segment::SEGMENT_TYPE_INFO),
        "the info encoder no longer writes SEGMENT_TYPE_INFO"
    );
    let distinct: std::collections::BTreeSet<u8> = by_class.values().copied().collect();
    assert_eq!(
        distinct.len(),
        by_class.len(),
        "two encoder classes wrote the same value at +4, so nothing here shows \
         it is a type rather than a constant: {by_class:?}"
    );

    eprintln!(
        "checked {checked} segment headers across {} classes",
        by_class.len()
    );
}

/// A continuation is one edge represented in two adjacent headers. Testing
/// either byte alone cannot establish its direction, so this fixture requires
/// both ends and their opposite values.
#[test]
#[cfg_attr(not(wire_fixtures), ignore = "run scripts/wire-oracle/wire-oracle.sh")]
fn segment_continuation_pair_has_both_directional_ends() {
    use reims_vgpu_wire::ops::segment;

    let root = fixtures();
    let mut pair = std::collections::BTreeMap::new();
    for case in root["cases"].as_array().expect("cases array") {
        let Some(name) = case["name"].as_str() else {
            continue;
        };
        if !name.starts_with("blit_segment_continuation_pair_") {
            continue;
        }
        let bytes = unhex(case["buffer"].as_str().expect("buffer hex"));
        let written = unhex(case["written_mask"].as_str().expect("written mask"));
        let header = segment::segment_header(&bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            written,
            [0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0],
            "{name}: exactly the final header byte must remain unwritten"
        );
        assert_eq!(bytes[7], 0xaa, "{name}: unwritten poison changed");
        pair.insert(name, (header.continues_previous(), header.continues_next()));
    }

    assert_eq!(
        pair.get("blit_segment_continuation_pair_0"),
        Some(&(false, true))
    );
    assert_eq!(
        pair.get("blit_segment_continuation_pair_1"),
        Some(&(true, false))
    );
    assert_eq!(
        pair.len(),
        2,
        "unexpected continuation-pair fixtures: {pair:?}"
    );
}

/// The compute encoder's record length the crate's constants claim.
fn compute_record_len_for(opcode: u32) -> Option<u32> {
    use reims_vgpu_wire::ops::compute as c;
    Some(match opcode {
        c::OPCODE_DISPATCH_THREADGROUPS_INDIRECT => c::DISPATCH_THREADGROUPS_INDIRECT_TOTAL_LEN,
        c::OPCODE_DISPATCH_THREADS_INDIRECT => c::DISPATCH_THREADS_INDIRECT_TOTAL_LEN,
        c::OPCODE_INSERT_COMPRESSED_TEXTURE_FLUSH => c::INSERT_COMPRESSED_TEXTURE_FLUSH_TOTAL_LEN,
        c::OPCODE_SET_BUFFER_OFFSET => c::SET_BUFFER_OFFSET_TOTAL_LEN,
        c::OPCODE_SET_BUFFER_OFFSET_STRIDE => c::SET_BUFFER_OFFSET_STRIDE_TOTAL_LEN,
        c::OPCODE_SET_PIPELINE_STATE => c::SET_PIPELINE_STATE_TOTAL_LEN,
        c::OPCODE_SET_STAGE_IN_REGION => c::SET_STAGE_IN_REGION_TOTAL_LEN,
        c::OPCODE_SET_STAGE_IN_REGION_INDIRECT => c::SET_STAGE_IN_REGION_INDIRECT_TOTAL_LEN,
        c::OPCODE_SET_THREADGROUP_MEMORY_LENGTH => c::SET_THREADGROUP_MEMORY_LENGTH_TOTAL_LEN,
        c::OPCODE_MEMORY_BARRIER_SCOPE => c::MEMORY_BARRIER_SCOPE_TOTAL_LEN,
        c::OPCODE_SET_IMAGEBLOCK_SIZE => c::SET_IMAGEBLOCK_SIZE_TOTAL_LEN,
        c::OPCODE_WRITE_DESCRIPTOR => c::WRITE_DESCRIPTOR_TOTAL_LEN,
        c::OPCODE_EXECUTE_COMMANDS_RANGE => c::EXECUTE_COMMANDS_RANGE_TOTAL_LEN,
        c::OPCODE_EXECUTE_COMMANDS_INDIRECT => c::EXECUTE_COMMANDS_INDIRECT_TOTAL_LEN,
        op if c::is_dispatch(op) => c::DISPATCH_TOTAL_LEN,
        op if c::is_fence(op) => c::FENCE_TOTAL_LEN,
        op if c::is_control_flow_predicate(op) => c::CONTROL_FLOW_PREDICATE_TOTAL_LEN,
        op if c::is_control_flow_marker(op) => c::CONTROL_FLOW_MARKER_TOTAL_LEN,
        _ => return None,
    })
}

#[test]
#[cfg_attr(not(wire_fixtures), ignore = "run scripts/wire-oracle/wire-oracle.sh")]
fn every_compute_fixture_reads_back_what_metal_was_asked_for() {
    use reims_vgpu_wire::ops::compute;

    let root = fixtures();
    let mut checked = 0usize;

    for case in root["cases"].as_array().expect("cases array") {
        if case["class"] != "PGSerializerComputeCommandEncoder" {
            continue;
        }
        let name = case["name"].as_str().expect("case name");
        let selector = case["selector"].as_str().expect("case selector");
        if selector == "beginSegment:protectionOptions:" {
            continue; // framing; see the segment-header test
        }
        // A case whose expectations name `requested` rather than `count` is a
        // truncation witness: it asked for a range Apple's serializer refuses to
        // write whole, so "reads back what Metal was asked for" is the one thing
        // it must not do. `a_plural_bind_is_truncated_at_the_argument_table_size`
        // owns those, and asserts the relationship instead of the value.
        if case["expect"].get("requested").is_some() {
            continue;
        }
        let bytes = unhex(case["buffer"].as_str().expect("buffer hex"));
        let allocated = case["allocated_len"].as_u64().expect("allocated_len");

        let o = op(&bytes, 0).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            o.length() as u64,
            allocated,
            "{name}: length disagrees with the serializer's own allocation"
        );
        if let Some(want) = compute_record_len_for(o.opcode()) {
            assert_eq!(
                want,
                o.length(),
                "{name}: opcode {:#x} came in at {} bytes; the crate's constant says {want}",
                o.opcode(),
                o.length()
            );
        }

        macro_rules! eq {
            ($got:expr, $key:literal) => {
                assert_eq!($got as u64, expect_u64(case, $key), "{}: {}", name, $key)
            };
        }

        match o.opcode() {
            op if compute::is_dispatch(op) => {
                let d = compute::dispatch(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                eq!(d.groups_width.get(), "groups_width");
                eq!(d.groups_height.get(), "groups_height");
                eq!(d.groups_depth.get(), "groups_depth");
                eq!(d.threads_width.get(), "threads_width");
                eq!(d.threads_height.get(), "threads_height");
                eq!(d.threads_depth.get(), "threads_depth");
            }
            compute::OPCODE_DISPATCH_THREADGROUPS_INDIRECT => {
                let d = compute::dispatch_indirect(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                eq!(d.threads_width.get(), "threads_width");
                eq!(d.threads_height.get(), "threads_height");
                eq!(d.threads_depth.get(), "threads_depth");
                eq!(d.indirect_buffer_ref.get(), "indirect_buffer_ref");
                eq!(d.indirect_buffer_offset.get(), "indirect_buffer_offset");
            }
            compute::OPCODE_SET_BUFFER_OFFSET_STRIDE => {
                let b = compute::buffer_offset_stride(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                eq!(b.index.get(), "first");
                eq!(b.offset.get(), "offset");
                eq!(b.attribute_stride.get(), "attribute_stride");
            }
            op if compute::is_buffer_stride_bind(op) => {
                // Both the singular and the plural selector write this opcode,
                // so the plural case is what shows the leading word is a count
                // and that the offset and stride are per entry rather than once
                // at the head — its two entries differ in all three fields.
                let (h, entries) =
                    compute::buffer_stride_binds(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                eq!(h.first.get(), "first");
                assert_eq!(
                    entries.len() as u64,
                    h.count.get() as u64,
                    "{name}: the entry array is not `count` long"
                );
                for (i, suffix) in ["", "_2"].iter().enumerate().take(entries.len()) {
                    for (got, key) in [
                        (entries[i].buffer_ref.get() as u64, "buffer_ref"),
                        (entries[i].offset.get(), "offset"),
                        (entries[i].attribute_stride.get(), "attribute_stride"),
                    ] {
                        let k = format!("{key}{suffix}");
                        assert_eq!(got, expect_u64(case, &k), "{name}: {k}");
                    }
                }
            }
            compute::OPCODE_SET_BUFFER => {
                let (h, entries) =
                    compute::buffer_binds(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                eq!(h.first.get(), "first");
                eq!(h.count.get(), "count");
                assert_eq!(
                    entries.len() as u64,
                    h.count.get() as u64,
                    "{name}: entry count"
                );
                eq!(entries[0].buffer_ref.get(), "buffer_ref");
                eq!(entries[0].offset.get(), "offset");
                if let Some(r2) = case["expect"]["buffer_ref_2"].as_u64() {
                    assert_eq!(
                        entries[1].buffer_ref.get() as u64,
                        r2,
                        "{name}: buffer_ref_2"
                    );
                    assert_eq!(
                        entries[1].offset.get(),
                        expect_u64(case, "offset_2"),
                        "{name}: offset_2"
                    );
                    assert_ne!(
                        entries[0].offset.get(),
                        entries[1].offset.get(),
                        "{name}: the two slots share an offset"
                    );
                }
            }
            compute::OPCODE_SET_SAMPLER_LOD => {
                let (h, entries) =
                    compute::sampler_lod_binds(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                eq!(h.first.get(), "first");
                eq!(h.count.get(), "count");
                assert_eq!(
                    entries.len() as u64,
                    h.count.get() as u64,
                    "{name}: entry count"
                );
                eq!(entries[0].sampler_ref.get(), "sampler_ref");
                for (i, suffix) in ["", "_2"].iter().enumerate() {
                    if i >= entries.len() {
                        break;
                    }
                    for (got, key) in [
                        (entries[i].lod_min_clamp.get(), "lod_min_clamp"),
                        (entries[i].lod_max_clamp.get(), "lod_max_clamp"),
                    ] {
                        let k = format!("{key}{suffix}");
                        let want = case["expect"][&k]
                            .as_f64()
                            .unwrap_or_else(|| panic!("{name}: no expect.{k}"));
                        assert_eq!(got as f64, want, "{name}: {k}");
                    }
                }
            }
            op if compute::is_ref_bind(op) => {
                let (h, refs) = compute::ref_binds(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                eq!(h.first.get(), "first");
                eq!(h.count.get(), "count");
                assert_eq!(refs.len() as u64, h.count.get() as u64, "{name}: ref count");
                let key = sole_key(case, &["texture_ref", "sampler_ref"]);
                assert_eq!(
                    refs[0].object_ref.get() as u64,
                    expect_u64(case, key),
                    "{name}: {key}"
                );
                if let Some(r2) = case["expect"]["texture_ref_2"].as_u64() {
                    assert_eq!(refs[1].object_ref.get() as u64, r2, "{name}: texture_ref_2");
                }
            }
            compute::OPCODE_SET_BUFFER_OFFSET => {
                let b = compute::set_buffer_offset(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                eq!(b.index.get(), "first");
                eq!(b.offset.get(), "offset");
            }
            compute::OPCODE_SET_PIPELINE_STATE => {
                let r = compute::set_pipeline_state(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                eq!(r.object_ref.get(), "pipeline_ref");
            }
            compute::OPCODE_SET_STAGE_IN_REGION => {
                let r = compute::set_stage_in_region(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                eq!(r.size_width.get(), "size_width");
                eq!(r.size_height.get(), "size_height");
                eq!(r.size_depth.get(), "size_depth");
                eq!(r.origin_x.get(), "origin_x");
                eq!(r.origin_y.get(), "origin_y");
                eq!(r.origin_z.get(), "origin_z");
                // The claim this case exists to make: size leads. A record that
                // wrote origin first would read the origin into the size.
                assert_ne!(
                    r.size_width.get(),
                    r.origin_x.get(),
                    "{name}: this case cannot tell size from origin"
                );
            }
            compute::OPCODE_SET_STAGE_IN_REGION_INDIRECT => {
                let r = compute::set_stage_in_region_indirect(&o)
                    .unwrap_or_else(|e| panic!("{name}: {e}"));
                eq!(r.indirect_buffer_ref.get(), "indirect_buffer_ref");
                eq!(r.indirect_buffer_offset.get(), "indirect_buffer_offset");
            }
            compute::OPCODE_SET_THREADGROUP_MEMORY_LENGTH => {
                let t = compute::set_threadgroup_memory_length(&o)
                    .unwrap_or_else(|e| panic!("{name}: {e}"));
                eq!(t.length.get(), "length");
                eq!(t.index.get(), "index");
            }
            op if compute::is_fence(op) => {
                let f = compute::fence(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                eq!(f.object_ref.get(), "fence_ref");
            }
            compute::OPCODE_MEMORY_BARRIER_RESOURCES => {
                let (h, refs) =
                    compute::memory_barrier_resources(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                eq!(h.count.get(), "count");
                assert_eq!(refs.len() as u64, h.count.get() as u64, "{name}: ref count");
                eq!(refs[0].object_ref.get(), "resource_ref");
                eq!(refs[1].object_ref.get(), "resource_ref_2");
            }
            compute::OPCODE_SET_IMAGEBLOCK_SIZE => {
                let b = compute::set_imageblock_size(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                eq!(b.width.get(), "width");
                eq!(b.height.get(), "height");
                assert_ne!(
                    b.width.get(),
                    b.height.get(),
                    "{name}: the two dimensions are equal, so this case cannot tell \
                     which slot is which"
                );
            }
            compute::OPCODE_WRITE_DESCRIPTOR => {
                let d = compute::write_descriptor(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                eq!(d.dispatch_type.get(), "dispatch_type");
            }
            op if compute::is_control_flow_predicate(op) => {
                let p =
                    compute::control_flow_predicate(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                eq!(p.buffer_ref.get(), "buffer_ref");
                eq!(p.offset.get(), "offset");
                eq!(p.comparison.get(), "comparison");
                eq!(p.reference_value.get(), "reference_value");
            }
            // Four records that are the header alone. The claim is that there
            // is nothing after it: `op()` already checked the length against
            // the crate's constant above, so what is left to say is that the
            // payload is empty rather than a body this crate failed to name.
            op if compute::is_control_flow_marker(op) => {
                assert!(
                    o.payload.is_empty(),
                    "{name}: a control-flow marker carries {} payload bytes; \
                     the record is supposed to be its opcode and nothing else",
                    o.payload.len()
                );
            }
            compute::OPCODE_MEMORY_BARRIER_SCOPE => {
                let b = compute::memory_barrier_scope(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                eq!(b.scope.get(), "scope");
                // Two bytes of this record are never written. The device's
                // decoder reads a second `u16` here; these are the bytes it
                // would be reading.
                assert_eq!(
                    &bytes[bytes.len() - 2..],
                    &[0xAA, 0xAA],
                    "{name}: the two bytes past `scope` are no longer unwritten"
                );
            }
            compute::OPCODE_EXECUTE_COMMANDS_RANGE => {
                let e =
                    compute::execute_commands_range(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                eq!(e.icb_ref.get(), "icb_ref");
                eq!(e.range_location.get(), "range_location");
                eq!(e.range_length.get(), "range_length");
            }
            compute::OPCODE_EXECUTE_COMMANDS_INDIRECT => {
                let e = compute::execute_commands_indirect(&o)
                    .unwrap_or_else(|e| panic!("{name}: {e}"));
                eq!(e.icb_ref.get(), "icb_ref");
                eq!(e.indirect_buffer_ref.get(), "indirect_buffer_ref");
                eq!(e.indirect_buffer_offset.get(), "indirect_buffer_offset");
            }
            compute::OPCODE_DISPATCH_THREADS_INDIRECT => {
                let d = compute::dispatch_threads_indirect(&o)
                    .unwrap_or_else(|e| panic!("{name}: {e}"));
                eq!(d.indirect_buffer_offset.get(), "indirect_buffer_offset");
                eq!(d.indirect_buffer_ref.get(), "indirect_buffer_ref");
            }
            // The header alone, like the four control-flow markers, and asserted
            // the same way: the length check above says twenty-less-twelve, and
            // this says the remainder is empty rather than a body nobody named.
            op if compute::is_insert_compressed_texture_flush(op) => {
                assert!(
                    o.payload.is_empty(),
                    "{name}: the compressed-texture flush carries {} payload \
                     bytes; the record is supposed to be its opcode and nothing \
                     else",
                    o.payload.len()
                );
            }
            other => panic!(
                "{name}: fixture carries opcode {other:#x} with no view; \
                 add one or mark it Unimplemented"
            ),
        }
        checked += 1;
    }
    assert!(checked > 0, "no compute encoder cases in fixtures.json");
    eprintln!("checked {checked} compute encoder fixtures against Apple's serializer");
}

/// The info encoder's record length the crate's constants claim.
fn info_record_len_for(opcode: u32) -> Option<u32> {
    use reims_vgpu_wire::ops::info as i;
    Some(match opcode {
        i::OPCODE_RATE_MAP_INFO => i::RATE_MAP_INFO_TOTAL_LEN,
        i::OPCODE_COPY_RATE_PARAMETER_BUFFER => i::COPY_RATE_PARAMETER_BUFFER_TOTAL_LEN,
        i::OPCODE_HEAP_TEXTURE_DESCRIPTOR_SIZE_AND_ALIGN => {
            i::HEAP_TEXTURE_DESCRIPTOR_SIZE_AND_ALIGN_TOTAL_LEN
        }
        i::OPCODE_HEAP_TEXTURE_DESCRIPTOR_SIZE_AND_ALIGN_WIDE => {
            i::HEAP_TEXTURE_DESCRIPTOR_SIZE_AND_ALIGN_WIDE_TOTAL_LEN
        }
        op if i::is_query(op) => i::QUERY_TOTAL_LEN,
        op if i::is_imageblock_query(op) => i::IMAGEBLOCK_TOTAL_LEN,
        op if i::is_map_coordinate(op) => i::MAP_COORDINATE_TOTAL_LEN,
        _ => return None,
    })
}

fn expect_f64(case: &Value, key: &str) -> f64 {
    case["expect"][key]
        .as_f64()
        .unwrap_or_else(|| panic!("case {} has no expect.{key}", case["name"]))
}

/// Info encoder records, checked against what Metal was asked for.
///
/// The class's whole shape is a `(reply_buffer_ref, reply_offset)` pair naming
/// where the answer lands, so that pair is asserted on every query rather than
/// only the fields the selector took as arguments. The expectation for it comes
/// from `CaptureCommandStream`, which is what the encoder asked — and it is
/// checked non-zero as well as equal, because a stream that declined would hand
/// back `(nil, 0)` and produce a record that agreed with a zeroed expectation.
#[test]
#[cfg_attr(not(wire_fixtures), ignore = "run scripts/wire-oracle/wire-oracle.sh")]
fn every_info_fixture_reads_back_what_metal_was_asked_for() {
    use reims_vgpu_wire::ops::info;

    let root = fixtures();
    let mut checked = 0usize;
    // `mapCoordinateInternal:…command:` writes its `command:` argument into the
    // opcode field, so its records carry opcodes this crate never named.
    // Collected and checked as a family below, like the blit `withCommand:`
    // selectors.
    let mut commanded: Vec<(u64, u32)> = Vec::new();
    // Scratch refs the stream handed back, and the buffer the one *command* on
    // this class was given. The two sets must not meet: that difference is the
    // only thing separating a query's record from `copyRasterization…`'s, which
    // are byte-identical.
    let mut reply_refs: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
    let mut command_buffer_refs: Vec<(&str, u32)> = Vec::new();
    // The distinct objects the ten identical-shaped queries named. One shared
    // value would let a view reading the wrong field pass on all ten.
    let mut query_objects: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();

    for case in root["cases"].as_array().expect("cases array") {
        if case["class"] != "PGSerializerInfoCommandEncoder" {
            continue;
        }
        let name = case["name"].as_str().expect("case name");
        let selector = case["selector"].as_str().expect("case selector");
        if selector == "beginSegment:protectionOptions:" {
            continue; // framing; see the segment-header test
        }
        let bytes = unhex(case["buffer"].as_str().expect("buffer hex"));
        let allocated = case["allocated_len"].as_u64().expect("allocated_len");

        let o = op(&bytes, 0).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            o.length() as u64,
            allocated,
            "{name}: length disagrees with the serializer's own allocation"
        );

        macro_rules! eq {
            ($got:expr, $key:literal) => {
                assert_eq!($got as u64, expect_u64(case, $key), "{}: {}", name, $key)
            };
        }

        // Where the answer goes. Asserted for every query shape through one
        // path, so a record that dropped the pair cannot pass by being read
        // through a different arm.
        macro_rules! reply {
            ($buf:expr, $off:expr) => {{
                let (buf, off) = ($buf, $off);
                assert_ne!(
                    buf, 0,
                    "{name}: reply_buffer_ref is 0, so the stream declined to \
                     hand back scratch and this case checks nothing"
                );
                assert_eq!(
                    buf as u64,
                    expect_u64(case, "reply_buffer_ref"),
                    "{name}: reply_buffer_ref"
                );
                assert_eq!(
                    off,
                    expect_u64(case, "reply_offset"),
                    "{name}: reply_offset"
                );
                reply_refs.insert(buf);
            }};
        }

        // Dispatched by selector, not by opcode: the claim is that the argument
        // *became* the opcode, so starting from the opcode would assume it.
        if selector.starts_with("mapCoordinateInternal:") {
            assert_eq!(
                o.length(),
                info::MAP_COORDINATE_TOTAL_LEN,
                "{name}: the generic mapper no longer writes a coordinate record"
            );
            let m = info::map_coordinate(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
            eq!(m.rate_map_ref.get(), "object_ref");
            eq!(m.layer.get(), "layer");
            assert_eq!(m.x.get() as f64, expect_f64(case, "x"), "{name}: x");
            assert_eq!(m.y.get() as f64, expect_f64(case, "y"), "{name}: y");
            reply!(m.reply_buffer_ref.get(), m.reply_offset.get());
            commanded.push((expect_u64(case, "command"), o.opcode()));
            checked += 1;
            continue;
        }

        assert_eq!(
            info_record_len_for(o.opcode()),
            Some(o.length()),
            "{name}: opcode {:#x} came in at {} bytes; the crate's constant says otherwise",
            o.opcode(),
            o.length()
        );

        match o.opcode() {
            info::OPCODE_RATE_MAP_INFO => {
                let r = info::rate_map_info(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                eq!(r.rate_map_ref.get(), "object_ref");
                reply!(r.reply_buffer_ref.get(), r.reply_offset.get());
                // `layerCount:` does not reach the wire; the reply's byte
                // length does, and the count is recoverable from it. Asserting
                // the inverse rather than the field is the whole claim.
                assert_eq!(
                    info::rate_map_layer_count(r.reply_len.get()),
                    Some(expect_u64(case, "layer_count") as u32),
                    "{name}: reply_len {} does not invert to the layer count asked for",
                    r.reply_len.get()
                );
                assert_eq!(
                    r.unidentified_u32.get(),
                    0,
                    "{name}: unidentified_u32 is no longer 0, so the experiment \
                     its doc asks for may now be unnecessary"
                );
            }
            info::OPCODE_COPY_RATE_PARAMETER_BUFFER => {
                let c =
                    info::copy_rate_parameter_buffer(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                eq!(c.rate_map_ref.get(), "object_ref");
                // The one record on this class whose second and third fields
                // are the *caller's* buffer and offset rather than the stream's
                // scratch. Checked against the reply refs at the end of the run.
                eq!(c.buffer_ref.get(), "buffer_ref");
                eq!(c.buffer_offset.get(), "buffer_offset");
                command_buffer_refs.push((name, c.buffer_ref.get()));
            }
            op if info::is_imageblock_query(op) => {
                let q = info::imageblock_query(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                eq!(q.pipeline_ref.get(), "object_ref");
                eq!(q.width.get(), "width");
                eq!(q.height.get(), "height");
                eq!(q.depth.get(), "depth");
                reply!(q.reply_buffer_ref.get(), q.reply_offset.get());
            }
            op if info::is_map_coordinate(op) => {
                let m = info::map_coordinate(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                eq!(m.rate_map_ref.get(), "object_ref");
                eq!(m.layer.get(), "layer");
                assert_eq!(m.x.get() as f64, expect_f64(case, "x"), "{name}: x");
                assert_eq!(m.y.get() as f64, expect_f64(case, "y"), "{name}: y");
                assert_ne!(
                    m.x.get(),
                    m.y.get(),
                    "{name}: this case cannot tell x from y"
                );
                reply!(m.reply_buffer_ref.get(), m.reply_offset.get());
            }
            info::OPCODE_HEAP_TEXTURE_DESCRIPTOR_SIZE_AND_ALIGN => {
                // The one record on this class that carries a *descriptor*
                // rather than an object ref. Every field is read through the
                // shared `TextureDescriptorBody` accessors, so the same
                // declaration is exercised from a third record.
                let q = info::heap_texture_descriptor_size_and_align(&o)
                    .unwrap_or_else(|e| panic!("{name}: {e}"));
                let d = &q.descriptor;
                eq!(d.width.get(), "width");
                eq!(d.height.get(), "height");
                eq!(d.depth.get(), "depth");
                eq!(d.mipmap_level_count.get(), "mipmap_level_count");
                eq!(d.sample_count.get(), "sample_count");
                eq!(d.array_length.get(), "array_length");
                eq!(d.texture_type(), "texture_type");
                eq!(d.usage(), "usage");
                eq!(d.pixel_format(), "pixel_format");
                eq!(d.storage_mode(), "storage_mode");
                reply!(q.reply_buffer_ref.get(), q.reply_offset.get());
            }
            info::OPCODE_HEAP_TEXTURE_DESCRIPTOR_SIZE_AND_ALIGN_WIDE => {
                // The same query with the wide descriptor. The reply pair is
                // what makes this worth asserting separately: it sits *after*
                // the descriptor, so the wide body's unwritten fortieth byte is
                // in the middle of the record and the two fields behind it move
                // by eight.
                let q = info::heap_texture_descriptor_size_and_align_wide(&o)
                    .unwrap_or_else(|e| panic!("{name}: {e}"));
                let d = &q.descriptor;
                eq!(d.width.get(), "width");
                eq!(d.height.get(), "height");
                eq!(d.texture_type(), "texture_type");
                eq!(d.usage.get(), "usage");
                eq!(d.pixel_format.get(), "pixel_format");
                eq!(d.storage_mode(), "storage_mode");
                eq!(d.swizzle_blue, "swizzle_blue");
                reply!(q.reply_buffer_ref.get(), q.reply_offset.get());
            }
            op if info::is_query(op) => {
                let q = info::query(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                eq!(q.object_ref.get(), "object_ref");
                reply!(q.reply_buffer_ref.get(), q.reply_offset.get());
                query_objects.insert(q.object_ref.get());
            }
            other => panic!(
                "{name}: fixture carries opcode {other:#x} with no view; \
                 add one or mark it Unimplemented"
            ),
        }
        checked += 1;
    }
    assert!(checked > 0, "no info encoder cases in fixtures.json");

    // Ten selectors write the identical [`info::Query`] record and differ only
    // in opcode, so a view reading the wrong field would pass on all ten if
    // they all named one object. They name seven different stubs.
    assert!(
        query_objects.len() >= 5,
        "the query family is checked against only {} distinct object refs; \
         with too few, a view reading the wrong field would still pass",
        query_objects.len()
    );

    // A command's buffer is the caller's; a query's is the stream's. Nothing in
    // the bytes distinguishes them, so the cases must.
    assert!(!command_buffer_refs.is_empty(), "no info command case");
    for (name, r) in &command_buffer_refs {
        assert!(
            !reply_refs.contains(r),
            "{name}: the caller's buffer {r} is also a ref the stream handed \
             back, so nothing here shows this record is not a query"
        );
    }

    // The `command:` claim, as an executable statement.
    assert!(
        commanded.len() >= 2,
        "mapCoordinateInternal: needs at least two different command arguments \
         to show the opcode follows them; got {}",
        commanded.len()
    );
    for (command, opcode) in &commanded {
        assert_eq!(
            *command, *opcode as u64,
            "mapCoordinateInternal: command {command:#x} produced opcode \
             {opcode:#x}; the argument is no longer the opcode"
        );
    }
    let distinct: std::collections::BTreeSet<u64> = commanded.iter().map(|(c, _)| *c).collect();
    assert!(
        distinct.len() >= 2,
        "every mapCoordinateInternal: case passed the same command, so the \
         opcode could be fixed"
    );

    eprintln!("checked {checked} info encoder fixtures against Apple's serializer");
}

/// Destroy records, checked against the refs the serializer itself allocated.
///
/// Eleven selectors write one record shape and differ only in opcode, so the
/// thing worth asserting beyond the field is that the eleven are *distinct*: a
/// view reading the wrong four bytes would pass on all of them if they shared
/// an object, and a manifest that gave two kinds one opcode would send a delete
/// to the wrong table.
#[test]
#[cfg_attr(not(wire_fixtures), ignore = "run scripts/wire-oracle/wire-oracle.sh")]
fn every_destroy_fixture_reads_back_the_ref_the_serializer_allocated() {
    use reims_vgpu_wire::ops::destroy;

    let root = fixtures();
    let mut opcodes: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
    let mut refs: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
    let mut checked = 0usize;

    for case in root["cases"].as_array().expect("cases array") {
        let name = case["name"].as_str().expect("case name");
        if !name.starts_with("lifecycle_delete_") {
            continue;
        }
        let bytes = unhex(case["buffer"].as_str().expect("buffer hex"));
        let o = op(&bytes, 0).unwrap_or_else(|e| panic!("{name}: {e}"));

        assert!(
            destroy::is_delete(o.opcode()),
            "{name}: opcode {:#x} is not one this module claims",
            o.opcode()
        );
        assert_eq!(
            o.length(),
            destroy::DELETE_TOTAL_LEN,
            "{name}: a destroy record is no longer {} bytes",
            destroy::DELETE_TOTAL_LEN
        );
        assert_eq!(
            o.length() as u64,
            case["allocated_len"].as_u64().expect("allocated_len"),
            "{name}: length disagrees with the serializer's own allocation"
        );

        let d = destroy::delete(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            d.object_ref.get() as u64,
            expect_u64(case, "object_ref"),
            "{name}: object_ref"
        );
        assert_ne!(
            d.object_ref.get(),
            0,
            "{name}: ref 0 is what an unallocated object reads as, so this case \
             would pass against a record that named nothing"
        );

        assert!(
            opcodes.insert(o.opcode()),
            "{name}: opcode {:#x} was already written by another object kind",
            o.opcode()
        );
        assert!(
            refs.insert(d.object_ref.get()),
            "{name}: two kinds were deleted at the same ref, so this family \
             cannot show the record carries the ref rather than a constant"
        );
        checked += 1;
    }

    assert_eq!(
        checked, 11,
        "the serializer ships eleven -deleteXRef:allocator: selectors and this \
         capture drove {checked}"
    );
    eprintln!(
        "checked {checked} destroy fixtures, {} distinct opcodes",
        opcodes.len()
    );
}

/// Every opcode Apple wrote for a covered selector is one its row lists.
///
/// [`manifest::Entry::opcodes`] is a second transcription of the same
/// measurement the `OPCODE_*` constants carry, written as literals and compared
/// by nothing — the crate's own tests dispatch on the constants, so a row could
/// name any number at all and stay green. This closes that: the comparison runs
/// against the bytes rather than against the constants, so it catches a drift in
/// either transcription.
///
/// One direction only. A row is allowed to list an opcode this capture did not
/// produce — each draw names both its compact and its wide form and one case
/// picks one — but an opcode Apple *did* write and the row does not name is a
/// row describing a record that is not the one the selector emits.
#[test]
#[cfg_attr(not(wire_fixtures), ignore = "run scripts/wire-oracle/wire-oracle.sh")]
fn every_covered_row_lists_the_opcode_apple_wrote() {
    let root = fixtures();

    let mut observed: std::collections::BTreeMap<(&str, &str), std::collections::BTreeSet<u32>> =
        std::collections::BTreeMap::new();
    for case in root["cases"].as_array().expect("cases array") {
        let selector = case["selector"].as_str().expect("case selector");
        // The segment header is framing and carries no opcode field; its rows
        // are `CoveredNoFixedOpcode` and are skipped below anyway.
        if selector == "beginSegment:protectionOptions:" {
            continue;
        }
        let class = case["class"].as_str().expect("case class");
        let bytes = unhex(case["buffer"].as_str().expect("buffer hex"));
        let o = op(&bytes, 0).unwrap_or_else(|e| panic!("{}: {e}", case["name"]));
        observed
            .entry((class, selector))
            .or_default()
            .insert(o.opcode());
    }

    let mut checked = 0usize;
    for e in manifest::MANIFEST {
        let Coverage::Covered { .. } = e.coverage else {
            continue;
        };
        let Some(seen) = observed.get(&(e.class, e.selector)) else {
            continue; // covered by a capture this run did not take
        };
        for opcode in seen {
            assert!(
                e.opcodes.contains(opcode),
                "-[{} {}] emitted opcode {opcode:#x} and its manifest row lists \
                 {:#x?}; the row names a record the selector does not write",
                e.class,
                e.selector,
                e.opcodes
            );
        }
        checked += 1;
    }
    assert!(checked > 0, "no covered row was matched to a capture");
    eprintln!("checked {checked} covered rows against the opcodes Apple wrote");
}

/// A selector whose records were not all captured claims nothing.
///
/// The capture's fifth outcome, and the one that used to be no outcome at all:
/// a case that produced a different number of records than it claimed fell out
/// of `cases` without landing anywhere, which is indistinguishable from a
/// selector nobody drove. `setFragmentTexture:atTextureIndex:samplerState:
/// atSamplerIndex:` was exactly that — it writes **two** records, and the case
/// claiming one recorded neither.
///
/// The answer is a split case, so this list should stay empty. What it must
/// never contain is a selector the manifest calls `Covered`, because at least
/// one of that selector's records has no fixture behind it.
#[test]
#[cfg_attr(not(wire_fixtures), ignore = "run scripts/wire-oracle/wire-oracle.sh")]
fn a_selector_whose_records_went_uncaptured_is_not_claimed_covered() {
    let root = fixtures();
    let multi: std::collections::BTreeSet<(&str, &str)> = root["multi"]
        .as_array()
        .expect("fixtures.json has no `multi` list; regenerate it with the current oracle")
        .iter()
        .map(|m| {
            (
                m["class"].as_str().expect("class"),
                m["selector"].as_str().expect("selector"),
            )
        })
        .collect();

    for e in manifest::MANIFEST {
        if !multi.contains(&(e.class, e.selector)) {
            continue;
        }
        assert_eq!(
            e.coverage,
            manifest::Coverage::Unimplemented,
            "-[{} {}] emitted a record count no case claimed, so at least one of \
             its records has no fixture; the row must stay Unimplemented until a \
             split case pins every one",
            e.class,
            e.selector
        );
    }
    eprintln!(
        "{} selectors emitted an unclaimed record count",
        multi.len()
    );
}

/// A selector the harness could not drive is not a selector Apple refused.
///
/// The capture's four outcomes are four different kinds of evidence, and this
/// one is the only one that says nothing about Apple: `crashed` means our stub
/// faulted, so the record it would have written is unknown. A row that read a
/// fault as an exclusion would be claiming a measurement nobody made, and a row
/// that claimed `Covered` would be claiming a fixture that does not exist.
#[test]
#[cfg_attr(not(wire_fixtures), ignore = "run scripts/wire-oracle/wire-oracle.sh")]
fn a_selector_that_faulted_the_harness_claims_nothing() {
    let root = fixtures();
    let crashed: std::collections::BTreeSet<(&str, &str)> = root["crashed"]
        .as_array()
        .expect("fixtures.json has no `crashed` list; regenerate it with the current oracle")
        .iter()
        .map(|c| {
            (
                c["class"].as_str().expect("class"),
                c["selector"].as_str().expect("selector"),
            )
        })
        .collect();

    for e in manifest::MANIFEST {
        if !crashed.contains(&(e.class, e.selector)) {
            continue;
        }
        assert_eq!(
            e.coverage,
            manifest::Coverage::Unimplemented,
            "-[{} {}] faulted this capture, so nothing is known about the record \
             it writes; the row must stay Unimplemented",
            e.class,
            e.selector
        );
    }
    eprintln!(
        "{} selectors faulted the harness, none claimed",
        crashed.len()
    );
}

#[test]
#[cfg_attr(not(wire_fixtures), ignore = "run scripts/wire-oracle/wire-oracle.sh")]
fn every_sampler_fixture_reads_back_what_metal_was_asked_for() {
    use reims_vgpu_wire::ops::sampler::{self, new_sampler};

    let root = fixtures();
    let mut checked = 0usize;
    let private_candidate_names = [
        "sampler_force_seams_on_cubemap_filtering",
        "sampler_force_resource_index",
        "sampler_resource_index",
        "sampler_pixel_format",
        "sampler_reduction_mode",
        "sampler_lod_bias",
        "sampler_border_color_spi",
    ];
    let mut private_candidates = std::collections::BTreeSet::new();
    for case in root["cases"].as_array().expect("cases array") {
        let name = case["name"].as_str().expect("case name");
        if case["selector"] != "newSamplerStateWithDescriptor:allocator:" {
            continue;
        }
        let bytes = unhex(case["buffer"].as_str().expect("buffer hex"));
        let allocated = case["allocated_len"].as_u64().expect("allocated_len");
        assert_eq!(bytes.len() as u64, allocated, "{name}: captured length");

        let o = op(&bytes, 0).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            o.opcode(),
            sampler::OPCODE_NEW_SAMPLER,
            "{name}: opcode drifted"
        );
        assert_eq!(
            o.length(),
            sampler::NEW_SAMPLER_TOTAL_LEN,
            "{name}: record length drifted"
        );
        // The allocation is longer than the body, which is the finding rather
        // than an accident. Both numbers are checked so a serializer that
        // started filling the tail fails here rather than going unnoticed.
        assert_eq!(o.length() as u64, allocated, "{name}: length vs allocation");
        assert_eq!(
            sampler::NEW_SAMPLER_TOTAL_LEN - sampler::NEW_SAMPLER_WRITTEN_LEN,
            8,
            "{name}: the module no longer claims an eight-byte unwritten tail"
        );

        let s = new_sampler(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            s.object_ref.get() as u64,
            expect_u64(case, "object_ref"),
            "{name}: object_ref"
        );
        assert_eq!(
            s.min_filter() as u64,
            expect_u64(case, "min_filter"),
            "{name}: min_filter"
        );
        assert_eq!(
            s.mag_filter() as u64,
            expect_u64(case, "mag_filter"),
            "{name}: mag_filter"
        );
        assert_eq!(
            s.mip_filter() as u64,
            expect_u64(case, "mip_filter"),
            "{name}: mip_filter"
        );
        assert_eq!(
            s.s_address_mode() as u64,
            expect_u64(case, "s_address_mode"),
            "{name}: s_address_mode"
        );
        assert_eq!(
            s.t_address_mode() as u64,
            expect_u64(case, "t_address_mode"),
            "{name}: t_address_mode"
        );
        assert_eq!(
            s.r_address_mode() as u64,
            expect_u64(case, "r_address_mode"),
            "{name}: r_address_mode"
        );
        assert_eq!(
            s.max_anisotropy() as u64,
            expect_u64(case, "max_anisotropy"),
            "{name}: max_anisotropy"
        );
        assert_eq!(
            s.compare_function() as u64,
            expect_u64(case, "compare_function"),
            "{name}: compare_function"
        );
        assert_eq!(
            s.border_color() as u64,
            expect_u64(case, "border_color"),
            "{name}: border_color"
        );
        assert_eq!(
            s.normalized_coordinates() as u64,
            expect_u64(case, "normalized_coordinates"),
            "{name}: normalized_coordinates"
        );
        assert_eq!(
            s.lod_average() as u64,
            expect_u64(case, "lod_average"),
            "{name}: lod_average"
        );
        assert_eq!(
            s.support_argument_buffers() as u64,
            expect_u64(case, "support_argument_buffers"),
            "{name}: support_argument_buffers"
        );
        assert_eq!(
            s.unidentified_flag_bits(),
            0,
            "{name}: a private candidate moved an unidentified sampler flag"
        );
        if private_candidate_names.contains(&name) {
            private_candidates.insert(name);
        }
        // Metal keeps these as `float`, and the expectation is the `double`
        // JSON carries, so the comparison is made in the wire's own width.
        assert_eq!(
            s.lod_min_clamp.get(),
            case["expect"]["lod_min_clamp"]
                .as_f64()
                .expect("lod_min_clamp") as f32,
            "{name}: lod_min_clamp"
        );
        assert_eq!(
            s.lod_max_clamp.get(),
            case["expect"]["lod_max_clamp"]
                .as_f64()
                .expect("lod_max_clamp") as f32,
            "{name}: lod_max_clamp"
        );
        checked += 1;
    }
    assert!(checked > 0, "no sampler cases in fixtures.json");
    assert_eq!(
        private_candidates,
        private_candidate_names.into_iter().collect(),
        "the private sampler-property exclusion sweep is incomplete"
    );
    eprintln!("checked {checked} sampler fixtures against Apple's serializer");
}

#[test]
#[cfg_attr(not(wire_fixtures), ignore = "run scripts/wire-oracle/wire-oracle.sh")]
fn every_depth_stencil_fixture_reads_back_what_metal_was_asked_for() {
    use reims_vgpu_wire::ops::depth_stencil::{self, new_depth_stencil};

    let root = fixtures();
    let mut checked = 0usize;
    for case in root["cases"].as_array().expect("cases array") {
        let name = case["name"].as_str().expect("case name");
        if case["selector"] != "newDepthStencilStateWithDescriptor:allocator:" {
            continue;
        }
        let bytes = unhex(case["buffer"].as_str().expect("buffer hex"));
        assert_eq!(
            bytes.len() as u64,
            case["allocated_len"].as_u64().expect("allocated_len"),
            "{name}: captured length"
        );

        let o = op(&bytes, 0).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            o.opcode(),
            depth_stencil::OPCODE_NEW_DEPTH_STENCIL,
            "{name}: opcode drifted"
        );
        assert_eq!(
            o.length(),
            depth_stencil::NEW_DEPTH_STENCIL_TOTAL_LEN,
            "{name}: record length drifted"
        );

        let d = new_depth_stencil(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            d.object_ref.get() as u64,
            expect_u64(case, "object_ref"),
            "{name}: object_ref"
        );
        assert_eq!(
            d.depth_compare_function() as u64,
            expect_u64(case, "depth_compare_function"),
            "{name}: depth_compare_function"
        );
        assert_eq!(
            d.depth_write_enabled() as u64,
            expect_u64(case, "depth_write_enabled"),
            "{name}: depth_write_enabled"
        );

        for (face, side) in [(d.front_stencil(), "front"), (d.back_stencil(), "back")] {
            let expected_present = expect_u64(case, &format!("{side}_face_present")) != 0;
            assert_eq!(face.is_some(), expected_present, "{name}: {side} presence");
            let Some(face) = face else {
                continue;
            };
            assert_eq!(
                face.compare_function() as u64,
                expect_u64(case, &format!("{side}_stencil_compare_function")),
                "{name}: {side} compare_function"
            );
            assert_eq!(
                face.stencil_failure_operation() as u64,
                expect_u64(case, &format!("{side}_stencil_failure_operation")),
                "{name}: {side} stencil_failure_operation"
            );
            assert_eq!(
                face.depth_failure_operation() as u64,
                expect_u64(case, &format!("{side}_depth_failure_operation")),
                "{name}: {side} depth_failure_operation"
            );
            assert_eq!(
                face.depth_stencil_pass_operation() as u64,
                expect_u64(case, &format!("{side}_depth_stencil_pass_operation")),
                "{name}: {side} depth_stencil_pass_operation"
            );
            assert_eq!(
                face.read_mask.get() as u64,
                expect_u64(case, &format!("{side}_read_mask")),
                "{name}: {side} read_mask"
            );
            assert_eq!(
                face.write_mask.get() as u64,
                expect_u64(case, &format!("{side}_write_mask")),
                "{name}: {side} write_mask"
            );
            assert_eq!(
                face.unidentified_ops_bits(),
                0,
                "{name}: {side} moved bits above the four operations -- identify them"
            );
        }
        assert_eq!(
            d.front_stencil_present() as u64,
            expect_u64(case, "front_face_present"),
            "{name}: front-face presence"
        );
        assert_eq!(
            d.back_stencil_present() as u64,
            expect_u64(case, "back_face_present"),
            "{name}: back-face presence"
        );
        checked += 1;
    }
    assert!(checked > 0, "no depth-stencil cases in fixtures.json");
    eprintln!("checked {checked} depth-stencil fixtures against Apple's serializer");
}

#[test]
#[cfg_attr(not(wire_fixtures), ignore = "run scripts/wire-oracle/wire-oracle.sh")]
fn a_nil_stencil_face_produces_the_record_a_default_one_does() {
    // Publicly assigning nil is not a way to create an absent face: the
    // descriptor substitutes its default face before serialization. Keep that
    // API behavior distinct from the private-slot presence contract below.
    let root = fixtures();
    let mut by_name = std::collections::BTreeMap::new();
    for case in root["cases"].as_array().expect("cases array") {
        by_name.insert(case["name"].as_str().expect("case name"), case);
    }
    let baseline = by_name["depth_stencil_baseline"];
    let base_bytes = unhex(baseline["buffer"].as_str().expect("buffer hex"));

    for absent in [
        "depth_stencil_front_face_absent",
        "depth_stencil_back_face_absent",
        "depth_stencil_both_faces_absent",
    ] {
        let case = by_name
            .get(absent)
            .unwrap_or_else(|| panic!("{absent} fixture"));
        // The premise: the case set a face to nil and Metal handed the
        // serializer a face anyway. If a later Metal stops substituting, this
        // fires first and says the negative result below no longer holds.
        assert_eq!(
            (
                expect_u64(case, "front_face_present"),
                expect_u64(case, "back_face_present"),
            ),
            (1, 1),
            "{absent}: Metal kept the nil face this time, so the serializer was asked \
             something these fixtures have never asked it"
        );
        let bytes = unhex(case["buffer"].as_str().expect("buffer hex"));
        let moved: Vec<usize> = (0..bytes.len())
            .filter(|i| bytes[*i] != base_bytes[*i])
            .collect();
        let object_ref = 8..12;
        assert!(
            moved.iter().all(|i| object_ref.contains(i)),
            "{absent} moved bytes {moved:?}; a nil face reaches the wire after all, and \
             `ops::depth_stencil` says it does not"
        );
    }
}

#[test]
#[cfg_attr(not(wire_fixtures), ignore = "run scripts/wire-oracle/wire-oracle.sh")]
fn private_face_absence_controls_presence_and_hides_its_unwritten_body() {
    use reims_vgpu_wire::ops::depth_stencil::new_depth_stencil;

    let root = fixtures();
    let mut by_name = std::collections::BTreeMap::new();
    for case in root["cases"].as_array().expect("cases array") {
        by_name.insert(case["name"].as_str().expect("case name"), case);
    }

    for (name, front_present, back_present) in [
        ("depth_stencil_private_front_face_absent", false, true),
        ("depth_stencil_private_back_face_absent", true, false),
        ("depth_stencil_private_both_faces_absent", false, false),
    ] {
        let case = by_name
            .get(name)
            .unwrap_or_else(|| panic!("{name} fixture"));
        let bytes = unhex(case["buffer"].as_str().expect("buffer hex"));
        let op = op(&bytes, 0).unwrap_or_else(|error| panic!("{name}: {error}"));
        let descriptor = new_depth_stencil(&op).unwrap_or_else(|error| panic!("{name}: {error}"));

        assert_eq!(descriptor.front_stencil_present(), front_present, "{name}");
        assert_eq!(descriptor.back_stencil_present(), back_present, "{name}");
        assert_eq!(
            descriptor.front_stencil().is_some(),
            front_present,
            "{name}"
        );
        assert_eq!(descriptor.back_stencil().is_some(), back_present, "{name}");
    }
}

#[test]
#[cfg_attr(not(wire_fixtures), ignore = "run scripts/wire-oracle/wire-oracle.sh")]
fn the_fence_fixture_is_a_header_and_the_ref_the_serializer_allocated() {
    use reims_vgpu_wire::ops::fence::{self, new_fence};

    let root = fixtures();
    let mut checked = 0usize;
    for case in root["cases"].as_array().expect("cases array") {
        let name = case["name"].as_str().expect("case name");
        if case["selector"] != "newFenceWithAllocator:" {
            continue;
        }
        let bytes = unhex(case["buffer"].as_str().expect("buffer hex"));
        let o = op(&bytes, 0).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            o.opcode(),
            fence::OPCODE_NEW_FENCE,
            "{name}: opcode drifted"
        );
        assert_eq!(
            o.length(),
            fence::NEW_FENCE_TOTAL_LEN,
            "{name}: a fence record is no longer {} bytes -- it has grown a field",
            fence::NEW_FENCE_TOTAL_LEN
        );
        assert_eq!(
            o.length() as u64,
            case["allocated_len"].as_u64().expect("allocated_len"),
            "{name}: length disagrees with the serializer's own allocation"
        );
        let f = new_fence(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            f.object_ref.get() as u64,
            expect_u64(case, "object_ref"),
            "{name}: object_ref"
        );
        assert_ne!(f.object_ref.get(), 0, "{name}: ref 0 names no object");
        checked += 1;
    }
    assert!(checked > 0, "no fence cases in fixtures.json");
    eprintln!("checked {checked} fence fixtures against Apple's serializer");
}

#[test]
#[cfg_attr(not(wire_fixtures), ignore = "run scripts/wire-oracle/wire-oracle.sh")]
fn every_heap_texture_fixture_reads_back_what_metal_was_asked_for() {
    use reims_vgpu_wire::ops::heap_texture::{self, new_heap_texture};

    let root = fixtures();
    let mut checked = 0usize;
    let mut saw_both_flags = (false, false);
    for case in root["cases"].as_array().expect("cases array") {
        let name = case["name"].as_str().expect("case name");
        if case["selector"] != "newTextureWithDescriptor:heap:offset:useOffset:allocator:" {
            continue;
        }
        let bytes = unhex(case["buffer"].as_str().expect("buffer hex"));
        let o = op(&bytes, 0).unwrap_or_else(|e| panic!("{name}: {e}"));

        // Under `-setSupportsTextureDescriptor2:` this selector emits its own
        // opcode with the wide descriptor. Same prefix, same tail, same
        // `useOffset` bit in the same four-byte slot — which is the claim, so it
        // is asserted through the wide view rather than inferred from the narrow
        // one.
        if o.opcode() == heap_texture::OPCODE_NEW_HEAP_TEXTURE_WIDE {
            assert_eq!(
                o.length(),
                heap_texture::NEW_HEAP_TEXTURE_WIDE_TOTAL_LEN,
                "{name}: wide record length drifted"
            );
            let h =
                heap_texture::new_heap_texture_wide(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(
                h.object_ref.get() as u64,
                expect_u64(case, "object_ref"),
                "{name}: object_ref"
            );
            assert_eq!(
                h.heap_ref.get() as u64,
                expect_u64(case, "heap_ref"),
                "{name}: heap_ref"
            );
            assert_eq!(h.offset.get(), expect_u64(case, "offset"), "{name}: offset");
            assert_eq!(
                h.use_offset() as u64,
                expect_u64(case, "use_offset"),
                "{name}: use_offset"
            );
            assert_eq!(
                h.desc.width.get() as u64,
                expect_u64(case, "width"),
                "{name}: width"
            );
            assert_eq!(
                h.desc.swizzle_red as u64,
                expect_u64(case, "swizzle_red"),
                "{name}: swizzle_red"
            );
            if h.use_offset() {
                saw_both_flags.0 = true;
            } else {
                saw_both_flags.1 = true;
            }
            checked += 1;
            continue;
        }

        assert_eq!(
            o.opcode(),
            heap_texture::OPCODE_NEW_HEAP_TEXTURE,
            "{name}: opcode drifted"
        );
        assert_eq!(
            o.length(),
            heap_texture::NEW_HEAP_TEXTURE_TOTAL_LEN,
            "{name}: record length drifted"
        );

        let h = new_heap_texture(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            h.object_ref.get() as u64,
            expect_u64(case, "object_ref"),
            "{name}: object_ref"
        );
        assert_eq!(
            h.heap_ref.get() as u64,
            expect_u64(case, "heap_ref"),
            "{name}: heap_ref"
        );
        assert_eq!(h.offset.get(), expect_u64(case, "offset"), "{name}: offset");
        assert_eq!(
            h.use_offset() as u64,
            expect_u64(case, "use_offset"),
            "{name}: use_offset"
        );
        // The embedded descriptor is the same declaration the plain creation
        // record uses, so these assertions are what makes a shared-layout drift
        // fail twice instead of once.
        assert_eq!(
            h.desc.texture_type() as u64,
            expect_u64(case, "texture_type"),
            "{name}: texture_type"
        );
        assert_eq!(
            h.desc.pixel_format() as u64,
            expect_u64(case, "pixel_format"),
            "{name}: pixel_format"
        );
        assert_eq!(
            h.desc.usage() as u64,
            expect_u64(case, "usage"),
            "{name}: usage"
        );
        assert_eq!(
            h.desc.width.get() as u64,
            expect_u64(case, "width"),
            "{name}: width"
        );
        assert_eq!(
            h.desc.height.get() as u64,
            expect_u64(case, "height"),
            "{name}: height"
        );
        assert_eq!(
            h.desc.storage_mode() as u64,
            expect_u64(case, "storage_mode"),
            "{name}: storage_mode"
        );

        if h.use_offset() {
            saw_both_flags.0 = true;
        } else {
            saw_both_flags.1 = true;
        }
        checked += 1;
    }
    assert!(checked > 0, "no heap-texture cases in fixtures.json");
    assert_eq!(
        saw_both_flags,
        (true, true),
        "every heap-texture fixture asked for the same `useOffset`, so the flag is \
         pinned against one value and would pass if the view read a constant"
    );
    eprintln!("checked {checked} heap-texture fixtures against Apple's serializer");
}

#[test]
#[cfg_attr(not(wire_fixtures), ignore = "run scripts/wire-oracle/wire-oracle.sh")]
fn every_texture_view_fixture_reads_back_what_metal_was_asked_for() {
    use reims_vgpu_wire::ops::texture_view as tv;

    let root = fixtures();
    let mut seen: std::collections::BTreeSet<u32> = Default::default();
    for case in root["cases"].as_array().expect("cases array") {
        let name = case["name"].as_str().expect("case name");
        if !name.starts_with("texture_view_") {
            continue;
        }
        let bytes = unhex(case["buffer"].as_str().expect("buffer hex"));
        let o = op(&bytes, 0).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(
            tv::is_texture_view(o.opcode()),
            "{name}: opcode {:#x} is not one this module claims",
            o.opcode()
        );
        seen.insert(o.opcode());

        // Whichever form it is, the shared prefix reads the same way. This is
        // the assertion that would fail if a wider form's field crept into the
        // narrower one's offsets.
        let (object_ref, base_ref, format) = match o.opcode() {
            tv::OPCODE_TEXTURE_VIEW => {
                assert_eq!(o.length(), tv::TEXTURE_VIEW_TOTAL_LEN, "{name}: length");
                let v = tv::texture_view(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                (
                    v.object_ref.get(),
                    v.base_texture_ref.get(),
                    v.pixel_format.get(),
                )
            }
            tv::OPCODE_TEXTURE_VIEW_RANGED => {
                assert_eq!(
                    o.length(),
                    tv::TEXTURE_VIEW_RANGED_TOTAL_LEN,
                    "{name}: length"
                );
                let v = tv::texture_view_ranged(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_ranges(case, name, v);
                (
                    v.object_ref.get(),
                    v.base_texture_ref.get(),
                    v.pixel_format.get(),
                )
            }
            tv::OPCODE_TEXTURE_VIEW_SWIZZLE => {
                assert_eq!(
                    o.length(),
                    tv::TEXTURE_VIEW_SWIZZLE_TOTAL_LEN,
                    "{name}: length"
                );
                let v = tv::texture_view_swizzle(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_ranges(case, name, &v.ranged);
                assert_eq!(
                    v.swizzle.red as u64,
                    expect_u64(case, "swizzle_red"),
                    "{name}: swizzle red"
                );
                assert_eq!(
                    v.swizzle.green as u64,
                    expect_u64(case, "swizzle_green"),
                    "{name}: swizzle green"
                );
                assert_eq!(
                    v.swizzle.blue as u64,
                    expect_u64(case, "swizzle_blue"),
                    "{name}: swizzle blue"
                );
                assert_eq!(
                    v.swizzle.alpha as u64,
                    expect_u64(case, "swizzle_alpha"),
                    "{name}: swizzle alpha"
                );
                (
                    v.ranged.object_ref.get(),
                    v.ranged.base_texture_ref.get(),
                    v.ranged.pixel_format.get(),
                )
            }
            other => panic!("{name}: unclaimed view opcode {other:#x}"),
        };
        assert_eq!(
            object_ref as u64,
            expect_u64(case, "object_ref"),
            "{name}: object_ref"
        );
        assert_eq!(
            base_ref as u64,
            expect_u64(case, "base_texture_ref"),
            "{name}: base_texture_ref"
        );
        assert_eq!(
            format as u64,
            expect_u64(case, "pixel_format"),
            "{name}: pixel_format"
        );
    }
    assert_eq!(
        seen,
        [
            tv::OPCODE_TEXTURE_VIEW,
            tv::OPCODE_TEXTURE_VIEW_RANGED,
            tv::OPCODE_TEXTURE_VIEW_SWIZZLE
        ]
        .into_iter()
        .collect(),
        "all three view forms must be captured; a missing one leaves its opcode unpinned"
    );
    eprintln!(
        "checked texture-view fixtures across {} opcodes",
        seen.len()
    );
}

/// The four range numbers, each against its own expectation.
///
/// Split out because two forms carry them and the failure that matters is a
/// swap — `levels` for `slices`, or a base for a count. Every case gives the
/// four different values, so a swap reports a number the case never asked for
/// rather than a plausible one.
fn assert_ranges(
    case: &Value,
    name: &str,
    v: &reims_vgpu_wire::ops::texture_view::TextureViewRangedBody,
) {
    assert_eq!(
        v.texture_type.get() as u64,
        expect_u64(case, "texture_type"),
        "{name}: texture_type"
    );
    assert_eq!(
        v.level_base.get(),
        expect_u64(case, "level_base"),
        "{name}: level_base"
    );
    assert_eq!(
        v.level_count.get(),
        expect_u64(case, "level_count"),
        "{name}: level_count"
    );
    assert_eq!(
        v.slice_base.get(),
        expect_u64(case, "slice_base"),
        "{name}: slice_base"
    );
    assert_eq!(
        v.slice_count.get(),
        expect_u64(case, "slice_count"),
        "{name}: slice_count"
    );
    // The premise of the swap check above: if a case ever gives two of these
    // the same value, a swapped view passes and nobody notices.
    let all = [
        v.level_base.get(),
        v.level_count.get(),
        v.slice_base.get(),
        v.slice_count.get(),
    ];
    let mut distinct = all.to_vec();
    distinct.sort_unstable();
    distinct.dedup();
    assert_eq!(
        distinct.len(),
        4,
        "{name}: two of the four range numbers are equal ({all:?}), so this case cannot \
         tell a swapped view from a correct one"
    );
}

#[test]
#[cfg_attr(not(wire_fixtures), ignore = "run scripts/wire-oracle/wire-oracle.sh")]
fn every_backed_texture_fixture_reads_back_what_metal_was_asked_for() {
    use reims_vgpu_wire::ops::backed_texture as bt;

    let root = fixtures();
    let mut buffer_cases = 0usize;
    let mut planes: std::collections::BTreeSet<u64> = Default::default();

    for case in root["cases"].as_array().expect("cases array") {
        let name = case["name"].as_str().expect("case name");
        let bytes = unhex(case["buffer"].as_str().expect("buffer hex"));

        if case["selector"] == "newTextureWithBuffer:descriptor:offset:bytesPerRow:allocator:" {
            let o = op(&bytes, 0).unwrap_or_else(|e| panic!("{name}: {e}"));
            // The wide form under `-setSupportsTextureDescriptor2:`: same four
            // prefix fields at the same offsets, wide descriptor after them.
            if o.opcode() == bt::OPCODE_BUFFER_TEXTURE_WIDE {
                assert_eq!(
                    o.length(),
                    bt::BUFFER_TEXTURE_WIDE_TOTAL_LEN,
                    "{name}: length"
                );
                let t = bt::buffer_texture_wide(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_eq!(
                    t.object_ref.get() as u64,
                    expect_u64(case, "object_ref"),
                    "{name}: object_ref"
                );
                assert_eq!(
                    t.buffer_ref.get() as u64,
                    expect_u64(case, "buffer_ref"),
                    "{name}: buffer_ref"
                );
                assert_eq!(t.offset.get(), expect_u64(case, "offset"), "{name}: offset");
                assert_eq!(
                    t.bytes_per_row.get(),
                    expect_u64(case, "bytes_per_row"),
                    "{name}: bytes_per_row"
                );
                assert_eq!(
                    t.desc.width.get() as u64,
                    expect_u64(case, "width"),
                    "{name}: width"
                );
                assert_eq!(
                    t.desc.swizzle_alpha as u64,
                    expect_u64(case, "swizzle_alpha"),
                    "{name}: swizzle_alpha"
                );
                buffer_cases += 1;
                continue;
            }
            assert_eq!(o.opcode(), bt::OPCODE_BUFFER_TEXTURE, "{name}: opcode");
            assert_eq!(o.length(), bt::BUFFER_TEXTURE_TOTAL_LEN, "{name}: length");
            let t = bt::buffer_texture(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(
                t.object_ref.get() as u64,
                expect_u64(case, "object_ref"),
                "{name}: object_ref"
            );
            assert_eq!(
                t.buffer_ref.get() as u64,
                expect_u64(case, "buffer_ref"),
                "{name}: buffer_ref"
            );
            assert_eq!(t.offset.get(), expect_u64(case, "offset"), "{name}: offset");
            assert_eq!(
                t.bytes_per_row.get(),
                expect_u64(case, "bytes_per_row"),
                "{name}: bytes_per_row"
            );
            assert_ne!(
                t.offset.get(),
                t.bytes_per_row.get(),
                "{name}: the two u64s are equal, so a view that read one for the other passes"
            );
            assert_desc(case, name, &t.desc);
            buffer_cases += 1;
        }

        if case["selector"] == "newIOSurfaceTextureWithDescriptor:plane:allocator:" {
            let o = op(&bytes, 0).unwrap_or_else(|e| panic!("{name}: {e}"));
            // The wide form carries a two-byte plane followed by rotation. The
            // final byte is unwritten and must not affect either value.
            if o.opcode() == bt::OPCODE_IOSURFACE_TEXTURE_WIDE {
                assert_eq!(
                    o.length(),
                    bt::IOSURFACE_TEXTURE_WIDE_TOTAL_LEN,
                    "{name}: length"
                );
                let t = bt::iosurface_texture_wide(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_eq!(
                    t.object_ref.get() as u64,
                    expect_u64(case, "object_ref"),
                    "{name}: object_ref"
                );
                assert_eq!(
                    t.plane.get() as u64,
                    expect_u64(case, "plane"),
                    "{name}: plane"
                );
                if let Some(rotation) = case["expect"]["rotation"].as_u64() {
                    assert_eq!(t.rotation as u64, rotation, "{name}: rotation");
                }
                assert_eq!(
                    t.desc.protection_options.get(),
                    expect_u64(case, "protection_options"),
                    "{name}: protection_options"
                );
                planes.insert(expect_u64(case, "plane"));
                continue;
            }
            if o.opcode() == bt::OPCODE_IOSURFACE_TEXTURE_ROTATED {
                assert_eq!(
                    o.length(),
                    bt::IOSURFACE_TEXTURE_ROTATED_TOTAL_LEN,
                    "{name}: length"
                );
                let t = bt::iosurface_texture_rotated(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_eq!(
                    t.object_ref.get() as u64,
                    expect_u64(case, "object_ref"),
                    "{name}: object_ref"
                );
                assert_eq!(
                    t.plane.get() as u64,
                    expect_u64(case, "plane"),
                    "{name}: plane"
                );
                assert_eq!(
                    t.rotation as u64,
                    expect_u64(case, "rotation"),
                    "{name}: rotation"
                );
                assert_desc(case, name, &t.desc);
                planes.insert(expect_u64(case, "plane"));
                continue;
            }
            assert_eq!(o.opcode(), bt::OPCODE_IOSURFACE_TEXTURE, "{name}: opcode");
            assert_eq!(
                o.length(),
                bt::IOSURFACE_TEXTURE_TOTAL_LEN,
                "{name}: length"
            );
            let t = bt::iosurface_texture(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(
                t.object_ref.get() as u64,
                expect_u64(case, "object_ref"),
                "{name}: object_ref"
            );
            assert_eq!(
                t.plane.get() as u64,
                expect_u64(case, "plane"),
                "{name}: plane"
            );
            assert_desc(case, name, &t.desc);
            assert_eq!(
                t.desc.protection_options.get(),
                expect_u64(case, "protection_options"),
                "{name}: protection_options"
            );
            planes.insert(expect_u64(case, "plane"));
        }
    }

    assert!(
        buffer_cases > 0,
        "no buffer-backed texture cases in fixtures.json"
    );
    assert!(
        planes.len() > 1,
        "every IOSurface case asked for the same plane ({planes:?}), so the field is pinned \
         against one value and would pass if the view read a constant"
    );
    eprintln!(
        "checked {buffer_cases} buffer-backed and {} IOSurface texture fixtures",
        planes.len()
    );
}

/// The shared 32-byte descriptor, wherever a record embeds it.
fn assert_desc(case: &Value, name: &str, d: &reims_vgpu_wire::ops::texture::TextureDescriptorBody) {
    assert_eq!(
        d.texture_type() as u64,
        expect_u64(case, "texture_type"),
        "{name}: texture_type"
    );
    assert_eq!(d.usage() as u64, expect_u64(case, "usage"), "{name}: usage");
    assert_eq!(
        d.pixel_format() as u64,
        expect_u64(case, "pixel_format"),
        "{name}: pixel_format"
    );
    assert_eq!(
        d.width.get() as u64,
        expect_u64(case, "width"),
        "{name}: width"
    );
    assert_eq!(
        d.height.get() as u64,
        expect_u64(case, "height"),
        "{name}: height"
    );
    assert_eq!(
        d.depth.get() as u64,
        expect_u64(case, "depth"),
        "{name}: depth"
    );
    assert_eq!(
        d.storage_mode() as u64,
        expect_u64(case, "storage_mode"),
        "{name}: storage_mode"
    );
    assert_eq!(
        d.allow_gpu_optimized_contents() as u64,
        expect_u64(case, "allow_gpu_optimized_contents"),
        "{name}: allow_gpu_optimized_contents"
    );
}

/// The rasterization rate map, at every shape the capture drove.
///
/// This record is where a fixture test earns its keep twice over. It is
/// variable length in two dimensions at once — the layer count and, per layer,
/// two sample counts — so an arithmetic slip in one dimension is hidden by a
/// capture that only ever varies the other. And every number it carries is a
/// count of something else, which is why the cases use asymmetric sizes
/// throughout: a 4x3 layer inside a 320x200 screen cannot have its width read
/// as its height, or either read as the layer count.
#[test]
#[cfg_attr(not(wire_fixtures), ignore = "run scripts/wire-oracle/wire-oracle.sh")]
fn every_rate_map_fixture_reads_back_what_metal_was_asked_for() {
    use reims_vgpu_wire::ops::rate_map;

    let root = fixtures();
    let mut checked = 0usize;
    let mut lengths: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();

    for case in root["cases"].as_array().expect("cases array") {
        if case["class"] != "PGSerializer" {
            continue;
        }
        let name = case["name"].as_str().expect("case name");
        let bytes = unhex(case["buffer"].as_str().expect("buffer hex"));
        let o = op(&bytes, 0).unwrap_or_else(|e| panic!("{name}: {e}"));
        if o.opcode() != rate_map::OPCODE_NEW_RASTERIZATION_RATE_MAP {
            continue;
        }
        lengths.insert(o.length());

        let (head, layers) = rate_map::rate_map(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            head.screen_width.get() as u64,
            expect_u64(case, "screen_width"),
            "{name}: screen_width"
        );
        assert_eq!(
            head.screen_height.get() as u64,
            expect_u64(case, "screen_height"),
            "{name}: screen_height"
        );
        assert_eq!(
            head.layer_count.get() as u64,
            expect_u64(case, "layer_count"),
            "{name}: layer_count"
        );
        assert_eq!(
            layers.len() as u64,
            expect_u64(case, "layer_count"),
            "{name}: layer array"
        );
        // The `new` form allocates the ref and the `reset` form is handed one;
        // both land in the same field, which is the claim that lets the two
        // selectors share a manifest opcode.
        if case["expect"]["object_ref"].is_number() {
            assert_eq!(
                head.object_ref.get() as u64,
                expect_u64(case, "object_ref"),
                "{name}: object_ref"
            );
        }
        // Constants, asserted so a serializer that starts moving them fails
        // here rather than being read as the fields they are not.
        assert_eq!(
            head.unidentified_u32_a.get(),
            2,
            "{name}: unidentified_u32_a moved; the experiment in its doc may now be available"
        );
        assert_eq!(
            head.unidentified_u32_b.get(),
            0,
            "{name}: unidentified_u32_b moved"
        );

        for (index, layer) in layers.iter().enumerate() {
            let w = expect_u64(case, &format!("layer{index}_sample_width"));
            let h = expect_u64(case, &format!("layer{index}_sample_height"));
            assert_eq!(
                layer.sample_width.get() as u64,
                w,
                "{name}: layer{index} sample_width"
            );
            assert_eq!(
                layer.sample_height.get() as u64,
                h,
                "{name}: layer{index} sample_height"
            );

            let (horizontal, vertical) =
                rate_map::layer_qualities(&o, index).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(
                horizontal.len() as u64,
                w,
                "{name}: layer{index} horizontal count"
            );
            assert_eq!(
                vertical.len() as u64,
                h,
                "{name}: layer{index} vertical count"
            );
            for (q, value) in horizontal.iter().enumerate() {
                assert_eq!(
                    value.get() as f64,
                    expect_f64(case, &format!("layer{index}_horizontal{q}")),
                    "{name}: layer{index} horizontal{q}"
                );
            }
            for (q, value) in vertical.iter().enumerate() {
                assert_eq!(
                    value.get() as f64,
                    expect_f64(case, &format!("layer{index}_vertical{q}")),
                    "{name}: layer{index} vertical{q}"
                );
            }
        }

        // The declared length is longer than what the serializer wrote, at
        // every size. `PARTIALLY_WRITTEN` already proves those bytes are
        // untouched; this proves the crate computes where they start.
        assert_eq!(
            rate_map::quality_span(&o).unwrap_or_else(|e| panic!("{name}: {e}"))
                + rate_map::UNWRITTEN_TAIL_LEN
                + reims_vgpu_wire::OP_HEADER_LEN,
            o.length() as usize,
            "{name}: the written extent plus the tail is not the record"
        );
        checked += 1;
    }

    assert!(checked > 0, "no rate map cases in fixtures.json");
    assert!(
        lengths.len() >= 3,
        "the rate map is variable length and only {} length(s) were captured; \
         a single length cannot show the record grows with the layer count",
        lengths.len()
    );
    eprintln!("rate map: {checked} cases at lengths {lengths:?}");
}

/// The indirect-command-buffer creation, one property per case.
///
/// Two things are asserted that no single case could establish. Every named
/// flag bit is checked on the case that inverted *that* property and on every
/// other case as well, so a bit named from the wrong case fails on the rest.
/// And the record is checked to carry the object ref **nowhere**, which is this
/// record's one departure from every other creation record in the protocol —
/// stated as a test because a future serializer that starts carrying it would
/// otherwise be a silent contract change.
#[test]
#[cfg_attr(not(wire_fixtures), ignore = "run scripts/wire-oracle/wire-oracle.sh")]
fn every_icb_fixture_reads_back_what_metal_was_asked_for() {
    use reims_vgpu_wire::ops::icb;

    let root = fixtures();
    let mut checked = 0usize;
    let mut refs: std::collections::BTreeSet<u64> = std::collections::BTreeSet::new();
    let mut unidentified: Option<u16> = None;

    for case in root["cases"].as_array().expect("cases array") {
        if case["class"] != "PGSerializer" {
            continue;
        }
        let name = case["name"].as_str().expect("case name");
        let bytes = unhex(case["buffer"].as_str().expect("buffer hex"));
        let o = op(&bytes, 0).unwrap_or_else(|e| panic!("{name}: {e}"));
        if o.opcode() != icb::OPCODE_NEW_INDIRECT_COMMAND_BUFFER {
            continue;
        }
        assert_eq!(
            o.length(),
            icb::NEW_INDIRECT_COMMAND_BUFFER_TOTAL_LEN,
            "{name}: record length"
        );

        let b = icb::new_indirect_command_buffer(&o).unwrap_or_else(|e| panic!("{name}: {e}"));
        macro_rules! eq {
            ($got:expr, $key:literal) => {
                assert_eq!($got as u64, expect_u64(case, $key), "{name}: {}", $key)
            };
        }
        eq!(b.command_types.get(), "command_types");
        eq!(
            b.max_vertex_buffer_bind_count,
            "max_vertex_buffer_bind_count"
        );
        eq!(
            b.max_fragment_buffer_bind_count,
            "max_fragment_buffer_bind_count"
        );
        eq!(
            b.max_kernel_buffer_bind_count,
            "max_kernel_buffer_bind_count"
        );
        eq!(
            b.max_object_buffer_bind_count,
            "max_object_buffer_bind_count"
        );
        eq!(b.max_mesh_buffer_bind_count, "max_mesh_buffer_bind_count");
        eq!(
            b.max_kernel_threadgroup_memory_bind_count,
            "max_kernel_threadgroup_memory_bind_count"
        );
        eq!(
            b.max_object_threadgroup_memory_bind_count,
            "max_object_threadgroup_memory_bind_count"
        );
        eq!(b.max_command_count.get(), "max_command_count");
        eq!(b.options.get(), "options");

        for (bit, key) in [
            (icb::flag::INHERIT_PIPELINE_STATE, "inherit_pipeline_state"),
            (icb::flag::INHERIT_BUFFERS, "inherit_buffers"),
            (icb::flag::SUPPORT_RAY_TRACING, "support_ray_tracing"),
            (
                icb::flag::SUPPORT_DYNAMIC_ATTRIBUTE_STRIDE,
                "support_dynamic_attribute_stride",
            ),
            (
                icb::flag::INHERIT_DEPTH_STENCIL_STATE,
                "inherit_depth_stencil_state",
            ),
            (icb::flag::INHERIT_DEPTH_BIAS, "inherit_depth_bias"),
            (
                icb::flag::INHERIT_DEPTH_CLIP_MODE,
                "inherit_depth_clip_mode",
            ),
            (icb::flag::INHERIT_CULL_MODE, "inherit_cull_mode"),
            (
                icb::flag::INHERIT_FRONT_FACING_WINDING,
                "inherit_front_facing_winding",
            ),
            (
                icb::flag::INHERIT_TRIANGLE_FILL_MODE,
                "inherit_triangle_fill_mode",
            ),
        ] {
            // A host whose SDK predates a property produces no expectation for
            // it and no case for it either; the bit is then unchecked rather
            // than checked against a value nothing measured.
            let Some(want) = case["expect"][key].as_u64() else {
                continue;
            };
            assert_eq!(
                b.has_flag(bit) as u64,
                want,
                "{name}: {key} at bit {}",
                bit.trailing_zeros()
            );
        }

        // The bits no property moved read the same value in every case. A
        // difference means one of them is a real field this capture missed.
        match unidentified {
            None => unidentified = Some(b.unidentified_flags()),
            Some(prev) => assert_eq!(
                b.unidentified_flags(),
                prev,
                "{name} moved the unidentified flag bits from {prev:#018b} to {:#018b} -- \
                 identify them and give them names",
                b.unidentified_flags()
            ),
        }

        assert_eq!(
            b.unidentified_u32.get(),
            0,
            "{name}: the one slot that could have held the object ref is no longer zero"
        );

        // The layout struct: fifteen fields the type encoding names by width
        // and order only.
        for (index, word) in b.layout.words16.iter().enumerate() {
            assert_eq!(
                word.get() as u64,
                expect_u64(case, &format!("layout_s{index}")),
                "{name}: layout_s{index}"
            );
        }
        for (index, word) in b.layout.words32.iter().enumerate() {
            assert_eq!(
                word.get() as u64,
                expect_u64(case, &format!("layout_i{index}")),
                "{name}: layout_i{index}"
            );
        }

        // The finding this record is unusual for: the ref the serializer
        // returned is not in the record, at any offset, aligned or not.
        //
        // The search starts past the header. The opcode and the length are not
        // places a ref could be carried, and the length in particular *will*
        // collide: this record is 0x58 = 88 bytes, so the moment the ref
        // allocator hands out 88 the scan reports the length word as the ref
        // hiding in plain sight. That is not a hypothetical — adding four
        // capability cases ahead of these shifted the allocator by exactly
        // enough to hit it. A search for a value has to exclude the places the
        // value cannot mean what it matches.
        let object_ref = expect_u64(case, "object_ref") as u32;
        refs.insert(object_ref as u64);
        for at in reims_vgpu_wire::OP_HEADER_LEN..bytes.len().saturating_sub(3) {
            let word = u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]);
            assert_ne!(
                word, object_ref,
                "{name}: the allocated ref {object_ref} appears at byte {at}; this record \
                 was documented as not carrying it, and `ops::icb` should now say where it is"
            );
        }
        checked += 1;
    }

    assert!(
        checked > 0,
        "no indirect command buffer cases in fixtures.json"
    );
    assert!(
        refs.len() >= 8,
        "only {} distinct refs were allocated across the ICB cases; the \
         no-ref claim needs several, because one ref could be absent by chance",
        refs.len()
    );
    eprintln!(
        "indirect command buffer: {checked} cases, {} distinct refs, none on the wire",
        refs.len()
    );
}

/// A plural bind's range is truncated at the stage's argument-table size, and
/// the three resource classes do not share one.
///
/// This is the "silent clamping" trap this crate's `AGENTS.md` warns about, in
/// the one place where the argument is an `NSRange` and so there is no property
/// to read back afterwards. The expectations therefore carry `requested`, what
/// the case asked for, and never the count the record holds — writing the wire's
/// own answer into the expectation is what makes a fixture agree with itself.
///
/// What is asserted instead is the *relationship*: every one of these cases
/// asked for 200 and the record says otherwise, each class stopping at its own
/// number, and the offset case showing the bound falls on `first + count`
/// rather than on `count`. That last one is the load-bearing half — a reader
/// who took this for a cap on how many entries may be bound would size the
/// table at 20 for a range of 20 and read twelve entries of whatever followed.
#[test]
#[cfg_attr(not(wire_fixtures), ignore = "run scripts/wire-oracle/wire-oracle.sh")]
fn a_plural_bind_is_truncated_at_the_argument_table_size() {
    use reims_vgpu_wire::ops::bind_limit;

    let root = fixtures();
    let cases = root["cases"].as_array().expect("cases array");

    // Each case names the class it exercises and the limit that class stops at.
    let expected = |name: &str| -> Option<u32> {
        match name {
            "compute_set_textures_over_bind_limit"
            | "compute_set_textures_over_bind_limit_offset"
            | "render_set_vertex_textures_over_bind_limit" => Some(bind_limit::TEXTURE),
            "compute_set_buffers_over_bind_limit" | "render_set_vertex_buffers_over_bind_limit" => {
                Some(bind_limit::BUFFER)
            }
            "compute_set_samplers_over_bind_limit" => Some(bind_limit::SAMPLER),
            _ => None,
        }
    };

    let mut checked = 0usize;
    let mut seen_limits = std::collections::BTreeSet::new();
    for case in cases {
        let name = case["name"].as_str().expect("case name");
        let Some(limit) = expected(name) else {
            continue;
        };
        let bytes = unhex(case["buffer"].as_str().expect("buffer hex"));

        // The plural bind payload leads with `first` then `count`, both u32.
        let payload = &bytes[8..];
        let first = u32::from_le_bytes(payload[0..4].try_into().unwrap());
        let count = u32::from_le_bytes(payload[4..8].try_into().unwrap());
        let requested = expect_u64(case, "requested") as u32;
        let asked_first = expect_u64(case, "first") as u32;

        assert_eq!(
            first, asked_first,
            "{name}: the range's start was not clamped"
        );
        assert!(
            count < requested,
            "{name}: asked for {requested} and got {count}; this case exists to \
             witness a truncation and is no longer witnessing one"
        );
        assert_eq!(
            first + count,
            limit,
            "{name}: the record ends at {} and the argument table is {limit}; the \
             bound falls on first + count, so a case starting at {first} keeps \
             {} entries",
            first + count,
            limit - first
        );
        seen_limits.insert(limit);
        checked += 1;
    }

    assert_eq!(
        checked, 6,
        "expected all six over-limit bind cases; a missing one is a class whose \
         limit nothing pins"
    );
    // The whole point is that one number cannot serve all three. If these ever
    // collapse to a single value, `bind_limit`'s three constants are one.
    assert_eq!(
        seen_limits.len(),
        3,
        "the three resource classes read {seen_limits:?}; they are supposed to \
         differ, and one MAX_BIND_ENTRIES is what this measurement refutes"
    );
    eprintln!(
        "plural bind truncation: {checked} cases, limits {seen_limits:?} \
         (textures/buffers/samplers)"
    );
}
