//! Run Apple's own records through this device's decoders.
//!
//! `reims-vgpu-wire` captures what the serializer emits; this crate decodes what
//! the guest sends. Those are the same bytes, and until now nothing compared the
//! two — each divergence was found by a person reading one file beside the
//! other, and four of them were real:
//!
//! * the render encoder's residency opcodes named numbers Apple assigns to
//!   nothing (`0x86`/`0x87` for `0x1b`/`0x89`);
//! * the compute resource barrier demanded a record four bytes longer than the
//!   serializer writes, so **every** one was refused;
//! * `copyFromTexture:toBuffer:` read two bytes past its `options` field;
//! * three blit records the serializer emits were classified as records it
//!   refuses.
//!
//! Three of those four are visible from the bytes alone, and that is the first
//! test here. The claim it makes is narrow and mechanical: **a record Apple's
//! serializer produced is never refused by a decoder for a reason about its
//! shape.** A decoder may decline an opcode it does not implement — that is a
//! gap, it is reported below, and it is not a failure. What it may not do is
//! call a well-formed record short, over-long, or over-count, because that is
//! the guest's work lost to a layout this crate got wrong.
//!
//! The remaining one — `copyFromTexture:toBuffer:` — is not visible that way,
//! and neither was `useOffset`, which came later. Both are a field read *wider*
//! than the serializer writes, and decoding Apple's buffer once cannot see it:
//! the capture arena is zero-filled exactly where a guest's command ring is
//! not, so the fixture agrees with the over-wide read by accident. That is what
//! [`no_decoder_reads_a_bit_apples_serializer_never_wrote`] is for. It uses the
//! per-bit `written_mask` the oracle measures — every case captured twice under
//! complementary fills — to decode each record twice more, with the untouched
//! bits all-zero and then all-one, and requires the same answer both times.
//!
//! The two tests share one decoder dispatch ([`read_record`]) on purpose. They
//! ask different questions, but a second copy of the opcode table is how one of
//! them quietly stops covering a record the other learned about.
//!
//! Fixtures are not committed. With none present both tests are `ignored`,
//! decided at build time by `build.rs` — see `../../reims-vgpu-wire/oracle/
//! fixture_presence.rs` for why that is a `cfg` and not a runtime early-return.
//! Regenerate with `scripts/wire-oracle/wire-oracle.sh`;
//! `REIMS_WIRE_FIXTURES_REQUIRED=1` (any Apple host, and CI) makes their
//! absence fail the build rather than stand the tests down.

use reims_vgpu::runtime::decode::{blit, compute, render};
use serde_json::Value;

/// Apple's captured records.
///
/// Only reachable when `wire_fixtures` is set, so absence here means the
/// capture was deleted between building the test and running it.
fn fixtures() -> Value {
    let dir = std::env::var("REIMS_WIRE_FIXTURES_DIR")
        .unwrap_or_else(|_| format!("{}/../reims-vgpu-wire/fixtures", env!("CARGO_MANIFEST_DIR")));
    let path = format!("{dir}/fixtures.json");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "{path} was present when this test was built and is not now ({e}); \
             regenerate with scripts/wire-oracle/wire-oracle.sh"
        )
    });
    serde_json::from_str(&text).expect("fixtures.json is valid JSON")
}

fn unhex(s: &str) -> Vec<u8> {
    assert!(s.len().is_multiple_of(2), "hex buffer has an odd length");
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("hex digit"))
        .collect()
}

/// One decoder's answer for one record: what it made of the bytes, and what
/// its refusal — if it refused — says about whose fault that is.
///
/// The two tests in this file ask different questions of the same call, so they
/// share one dispatch rather than each carrying its own copy of the opcode
/// table. A copied dispatch is a divergence waiting to happen: the moment one
/// test learns a new opcode and the other does not, the second silently stops
/// covering it.
struct Reading {
    verdict: Verdict,
    /// The decoder's output, rendered. Nothing parses this — it is compared
    /// against the same decoder's output for a differently-poisoned copy of the
    /// same record, so all it has to be is faithful and stable.
    signature: String,
}

/// What a decoder's refusal says about whose fault it is.
enum Verdict {
    /// The record decoded, or decoded into the decoder's own "accepted but not
    /// executed" state. Nothing to report.
    Decoded,
    /// The decoder does not implement this opcode. A gap in this crate, named
    /// so the summary can list it — not a failure.
    NotImplemented(&'static str),
    /// The decoder refused a record the serializer produced, for a reason about
    /// the record's shape. That is a layout this crate has wrong.
    WrongShape(&'static str),
}

fn render_verdict(bytes: &[u8]) -> Reading {
    use render::DecodeStatus as S;
    let decoded = render::decode(bytes);
    let verdict = match &decoded {
        // `OtherAccepted` is an `Ok` that means "no arm claimed this". Counting
        // it as decoded is what hid `0x80`/`0x71` — the LOD-bearing sampler
        // binds — behind a passing run, because the record went through
        // `decode` and came back fine while the guest's bind was dropped.
        Ok(c) if c.kind == render::Kind::OtherAccepted => {
            Verdict::NotImplemented("render_other_accepted")
        }
        Ok(_) => Verdict::Decoded,
        Err(S::ErrUnknownOpcode) => Verdict::NotImplemented("render_decode_unknown_opcode"),
        Err(S::ErrUnsupportedOpcode) => Verdict::NotImplemented("render_decode_unsupported_opcode"),
        Err(S::ErrShort) => Verdict::WrongShape("render_decode_short"),
        Err(S::ErrBadLength) => Verdict::WrongShape("render_decode_bad_length"),
        Err(S::ErrCountOutOfRange) => Verdict::WrongShape("render_decode_count_out_of_range"),
    };
    Reading {
        verdict,
        signature: format!("{decoded:?}"),
    }
}

fn compute_verdict(bytes: &[u8]) -> Reading {
    use compute::DecodeStatus as S;
    let decoded = compute::decode(bytes);
    let verdict = match &decoded {
        Ok(_) => Verdict::Decoded,
        Err(S::ErrUnknownOpcode) => Verdict::NotImplemented("compute_decode_unknown_opcode"),
        Err(S::ErrUnsupportedOpcode) => {
            Verdict::NotImplemented("compute_decode_unsupported_opcode")
        }
        // `ErrShort` is now the compute arm's only shape failure, the way the
        // blit arm's became after `ErrUnimplementedOpcode` went. Every compute
        // refusal that is not an unknown or unsupported opcode is a layout this
        // project has wrong, which makes the arm strictly stronger than when a
        // second variant could absorb one.
        Err(S::ErrShort) => Verdict::WrongShape("compute_decode_short"),
    };
    Reading {
        verdict,
        signature: format!("{decoded:?}"),
    }
}

fn blit_verdict(bytes: &[u8]) -> Reading {
    use blit::DecodeStatus as S;
    let decoded = blit::decode(bytes);
    let verdict = match &decoded {
        Ok(_) => Verdict::Decoded,
        Err(S::ErrUnknownOpcode) => Verdict::NotImplemented("blit_decode_unknown_opcode"),
        Err(S::ErrShort) => Verdict::WrongShape("blit_decode_short"),
    };
    Reading {
        verdict,
        signature: format!("{decoded:?}"),
    }
}

/// The object-creation records, dispatched by opcode to their own decoders.
///
/// `PGSerializer` has no single `decode(&[u8])` the way the encoders do — each
/// creation record has its own reader in `decode::resource`, reached from the
/// object-list walk rather than from a command buffer. So this arm picks the
/// decoder by opcode itself, and that changes what an error means: the encoder
/// arms cannot tell "wrong decoder" from "wrong layout", but here the decoder
/// is chosen from the opcode Apple wrote, so **any** refusal is a refusal of a
/// record that decoder is supposed to read. All of them are `WrongShape`.
///
/// **A `Decoded` verdict on this arm means the reader this device has for these
/// bytes accepted them, which for four opcodes is the descriptor-*body* decoder
/// rather than a record reader.** Opcodes 1, `0x34`, `0x0c` and `0x39` create a
/// plain, swizzled or IOSurface-backed texture, and this device meets all four
/// through the kernel's object list instead of through the creation stream — so
/// nothing here reads the record, while `heap_query`'s body decoder reads the
/// descriptor inside it and is what those fixtures sign. Do not read those four
/// as "the creation record is implemented".
///
/// The opcodes with no reader at all are named individually rather than skipped
/// as a class. Two are worth their own line:
///
/// * `0x0c`, the IOSurface-backed texture. `decode_iosurface_texture_descriptor`
///   looks like its record reader and is not: that function reads the **type-11
///   object-list descriptor**, a structure the kernel and IOSurface write,
///   whose live blobs run `0x38`/`0x58` bytes against this record's 48. Its own
///   doc says so and says it does not run on the x86 pathway at all. Feeding a
///   `0x0c` record to it would not test it, it would test a coincidence — and
///   because that decoder has no opcode gate and only a `>= 0x20` length check,
///   the coincidence is that it *succeeds*, reading the opcode word as a
///   mapping id and four later fields off by whole struct members. That is the
///   finding; the fix is not to move its offsets, which are a Tier-2 format
///   with no oracle behind them.
/// * `13`, the fence, and `0x3e8`–`0x3f7`, the eleven destroy records: this
///   device tears objects down at the FIFO layer, so nothing reads either.
fn serializer_verdict(bytes: &[u8], opcode: u32) -> Reading {
    use reims_vgpu::runtime::decode::resource;
    use reims_vgpu::runtime::heap_query::{self, decode_serialized_texture_descriptor};

    // Turn a refusal into a failure. See the doc above for why every one
    // counts here when it does not on the encoder arms.
    fn shape<T: std::fmt::Debug, E: std::fmt::Debug>(
        r: Result<T, E>,
        slug: &'static str,
    ) -> Reading {
        Reading {
            verdict: match r {
                Ok(_) => Verdict::Decoded,
                Err(_) => Verdict::WrongShape(slug),
            },
            signature: format!("{r:?}"),
        }
    }

    fn gap(slug: &'static str) -> Reading {
        Reading {
            verdict: Verdict::NotImplemented(slug),
            signature: String::new(),
        }
    }

    /// Slice a record's embedded descriptor body out and hand it to the decoder
    /// that reads it.
    ///
    /// Three of these records have no reader for the *record* — the device
    /// meets a plain, IOSurface-backed or swizzled texture through the kernel's
    /// object list rather than through the creation stream — but all of them
    /// carry a body the device does read, at the same place: header, the new
    /// object's ref, then the descriptor. Slicing it is what keeps the body
    /// decoder meeting Apple's bytes, including the two fixtures that are the
    /// only ones pinning the wide form's swizzle order.
    ///
    /// The body's own length is not inferred from the record's; each caller
    /// passes the length its opcode implies.
    fn body(bytes: &[u8], len: usize, wide: bool) -> Reading {
        use reims_vgpu::runtime::heap_query::decode_wide_serialized_texture_descriptor;
        const REF_END: usize = 12; // 8-byte header, then the new object's ref
        if bytes.len() < REF_END + len {
            return Reading {
                verdict: Verdict::WrongShape("serializer_texture_record_short"),
                signature: String::new(),
            };
        }
        let body = &bytes[REF_END..REF_END + len];
        if wide {
            shape(
                decode_wide_serialized_texture_descriptor(body),
                "serializer_texture_body_wide",
            )
        } else {
            shape(
                decode_serialized_texture_descriptor(body),
                "serializer_texture_body",
            )
        }
    }

    let narrow = heap_query::TEXTURE_BODY_LEN;
    let wide = heap_query::WIDE_TEXTURE_BODY_LEN;

    match opcode {
        1 => body(bytes, narrow, false),
        // `newTextureWithDescriptor:allocator:` under `SwizzledTextures`. A
        // different opcode rather than a longer record, so it is dispatched
        // here rather than inferred from the length.
        0x34 => body(bytes, wide, true),
        3 => shape(
            resource::decode_sampler_descriptor(bytes),
            "serializer_sampler",
        ),
        4 => shape(
            resource::decode_depth_stencil_descriptor(bytes),
            "serializer_depth_stencil",
        ),
        7 | 8 | 0x1b => shape(
            resource::decode_texture_view_descriptor(bytes),
            "serializer_texture_view",
        ),
        // The buffer-backed texture and its `TextureDescriptor2` form. One
        // decoder reads both: it dispatches on the opcode and then requires the
        // length that opcode implies.
        9 | 0x37 => shape(
            resource::decode_buffer_texture_descriptor(bytes),
            "serializer_buffer_texture",
        ),
        // The heap record hands its embedded descriptor on as a slice rather
        // than reading it, so those bytes are not this decoder's reading and
        // must not be signed as if they were — `packed` bit 7 lives in there
        // and is never written, which would diverge on every poisoned pair for
        // a field nobody read. Compose the decoder that *does* read it, which
        // is what `compute_exec` does with the slice, and sign the pair.
        0x15 | 0x38 => {
            use reims_vgpu::runtime::heap_query::decode_wide_serialized_texture_descriptor;
            let record = resource::decode_heap_texture(bytes);
            let body = record.as_ref().map(|r| {
                if r.wide {
                    decode_wide_serialized_texture_descriptor(r.descriptor)
                } else {
                    decode_serialized_texture_descriptor(r.descriptor)
                }
            });
            Reading {
                verdict: match &record {
                    Ok(_) => Verdict::Decoded,
                    Err(_) => Verdict::WrongShape("serializer_heap_texture"),
                },
                signature: format!(
                    "{:?} body={body:?}",
                    record.map(|r| resource::HeapTextureRecord {
                        descriptor: &[],
                        ..r
                    })
                ),
            }
        }
        // The indirect-command-buffer creation. This device does have a reader
        // for it, and the gap map hid that for as long as the map was keyed on
        // how a record *arrives*: `decode_icb_descriptor` is reached by object
        // type through the type-7 object list rather than by opcode off the
        // command stream, so the 88 bytes never met their decoder here even
        // though the decoder existed. They are the same bytes — the type-7 tag
        // is the opcode and the declared length is the record length, which is
        // what the decoder itself checks first.
        0x36 => shape(resource::decode_icb_descriptor(bytes), "serializer_icb"),
        // `heapTextureSizeAndAlignWithDescriptor:`, and the second instance of
        // the shape `0x36` taught: a gap here can be a decoder nobody routed to
        // it. `runtime::heap_query` has read this record all along —
        // `SERIALIZED_TEXTURE_TAG` *is* this opcode, `SERIALIZED_TEXTURE_LEN`
        // *is* this record's length, and the 32 bytes behind the header are the
        // same `PGSerializedTextureDescriptor` body the `0x15` arm above hands
        // on. What hid it is that the device meets this record wrapped in a
        // request — `decode_request` reads a 24-byte task/reply header first —
        // so the bare serializer record never reached the decoder that parses
        // its payload.
        //
        // Only the descriptor body is signed. The two header words are the
        // opcode and the length, which `decode_request` checks against the same
        // two constants and which carry no field of their own.
        0x16 => {
            let want = heap_query::SERIALIZED_TEXTURE_LEN;
            if bytes.len() != want {
                return Reading {
                    verdict: Verdict::WrongShape("serializer_heap_texture_size_and_align_length"),
                    signature: String::new(),
                };
            }
            shape(
                decode_serialized_texture_descriptor(&bytes[8..want]),
                "serializer_heap_texture_size_and_align",
            )
        }
        // The IOSurface-backed texture and its `TextureDescriptor2` form. Only
        // the embedded descriptor is signed, for the reason in `body` above:
        // what reaches this device for an IOSurface texture is the type-11
        // object-list descriptor, a different structure entirely.
        0x0c => body(bytes, narrow, false),
        0x39 => body(bytes, wide, true),
        13 => gap("serializer_fence_no_reader"),
        0x3e8..=0x3f7 => gap("serializer_destroy_no_reader"),
        _ => gap("serializer_opcode_no_reader"),
    }
}

/// The info encoder's queries, which this device answers through two different
/// paths and mostly not at all.
///
/// This class used to be skipped wholesale through `UNCOVERED_CLASSES`, counted
/// as one number with one sentence. That is the weakest thing the map can do to
/// a class — `crates/reims-vgpu-wire/AGENTS.md` names it as this instrument's
/// blind spot, and the one live cross-crate disagreement is inside it — because
/// a class-level skip says nothing about *which* records are read. Per opcode:
///
/// * `0x1c3` and `0x1d5`, `heapTextureDescriptorSizeAndAlign:`, embed the same
///   `PGSerializedTextureDescriptor` body at payload `+0` that the creation
///   records carry, narrow and wide. `heap_query` reads both. This is the same
///   shape `0x36` and `0x16` taught — a record the device meets under another
///   name — so the body is sliced out and signed rather than being called a gap.
/// * `0x1d1`, the ICB host-resource query, is the disagreement: this crate's
///   `ops::info::Query` reads `object_ref`/`reply_buffer_ref`/`reply_offset`
///   where `runtime::icb` reads `buffer_ref`/`gpu_address` and *binds* them as
///   an ICB's command memory. The fixtures settled it in the wire crate's
///   favour and the repair is a deliberate behaviour change on a dormant rail,
///   not a rename. It is a **named** gap here so the map prints it every run
///   instead of folding it into a class total.
/// * The rest are queries with no reader on any rail. Named individually, so
///   one of them growing a reader is visible as one line changing.
fn info_verdict(bytes: &[u8], opcode: u32) -> Reading {
    use reims_vgpu::runtime::heap_query::{
        self, decode_serialized_texture_descriptor, decode_wide_serialized_texture_descriptor,
    };
    const HDR: usize = 8;

    fn body<T: std::fmt::Debug, E: std::fmt::Debug>(
        r: Result<T, E>,
        slug: &'static str,
    ) -> Reading {
        Reading {
            verdict: match r {
                Ok(_) => Verdict::Decoded,
                Err(_) => Verdict::WrongShape(slug),
            },
            signature: format!("{r:?}"),
        }
    }
    fn gap(slug: &'static str) -> Reading {
        Reading {
            verdict: Verdict::NotImplemented(slug),
            signature: String::new(),
        }
    }

    match opcode {
        0x1c3 => {
            let end = HDR + heap_query::TEXTURE_BODY_LEN;
            match bytes.get(HDR..end) {
                Some(b) => body(
                    decode_serialized_texture_descriptor(b),
                    "info_heap_texture_descriptor_body",
                ),
                None => Reading {
                    verdict: Verdict::WrongShape("info_heap_texture_descriptor_short"),
                    signature: String::new(),
                },
            }
        }
        0x1d5 => {
            let end = HDR + heap_query::WIDE_TEXTURE_BODY_LEN;
            match bytes.get(HDR..end) {
                Some(b) => body(
                    decode_wide_serialized_texture_descriptor(b),
                    "info_heap_texture_descriptor_body_wide",
                ),
                None => Reading {
                    verdict: Verdict::WrongShape("info_heap_texture_descriptor_wide_short"),
                    signature: String::new(),
                },
            }
        }
        0x1d1 => gap("info_icb_query_read_differently_by_runtime_icb"),
        _ => gap("info_query_no_reader"),
    }
}

/// Every decoder this file knows how to reach, keyed the way a fixture arrives.
///
/// Returns `None` for a class with no record decoder — the caller decides
/// whether that is a counted gap or a mapping bug, because the two tests
/// disagree about which.
fn read_record(class: &str, bytes: &[u8], opcode: u32) -> Option<Reading> {
    match class {
        "PGSerializer" => Some(serializer_verdict(bytes, opcode)),
        "PGSerializerRenderCommandEncoder" => Some(render_verdict(bytes)),
        "PGSerializerComputeCommandEncoder" => Some(compute_verdict(bytes)),
        "PGSerializerBlitCommandEncoder" => Some(blit_verdict(bytes)),
        "PGSerializerInfoCommandEncoder" => Some(info_verdict(bytes, opcode)),
        _ => None,
    }
}

/// Classes with no record decoder in this crate, and why.
///
/// The info encoder's queries are reached only through `runtime::icb`, for the
/// single opcode it decodes, so there is no `decode(&[u8])` to call. It is
/// counted and named rather than silently skipped — a class missing from this
/// summary would look like a class that passed.
///
/// `PGSerializer` used to be here too, which meant 69 fixtures were counted and
/// none was checked. It has its own dispatch now; see [`serializer_verdict`].
const UNCOVERED_CLASSES: &[(&str, &str)] = &[];

/// Whether a fixture's leading word is an opcode a guest would send.
///
/// Two families of case are not records, and both would test the oracle rather
/// than a decoder:
///
/// * `-beginSegment:protectionOptions:` writes segment framing, which has no
///   opcode field at all — its first word is a length.
/// * the generic emitters take the opcode as a `command:` argument and write it
///   straight through, so what a decoder would see is the case's own constant.
fn carries_a_guest_opcode(selector: &str) -> bool {
    selector != "beginSegment:protectionOptions:"
        && !selector.contains("withCommand:")
        && !selector.starts_with("mapCoordinateInternal:")
}

#[test]
#[cfg_attr(not(wire_fixtures), ignore = "run scripts/wire-oracle/wire-oracle.sh")]
fn no_record_apples_serializer_produced_is_refused_for_its_shape() {
    let root = fixtures();

    let mut decoded = 0usize;
    let mut gaps: std::collections::BTreeMap<(u32, &str), usize> =
        std::collections::BTreeMap::new();
    let mut uncovered: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    let mut wrong: Vec<String> = Vec::new();

    for case in root["cases"].as_array().expect("cases array") {
        let name = case["name"].as_str().expect("case name");
        let class = case["class"].as_str().expect("case class");
        let selector = case["selector"].as_str().expect("case selector");

        if !carries_a_guest_opcode(selector) {
            continue;
        }

        if let Some((_, why)) = UNCOVERED_CLASSES.iter().find(|(c, _)| *c == class) {
            *uncovered.entry(why).or_default() += 1;
            continue;
        }

        let bytes = unhex(case["buffer"].as_str().expect("buffer hex"));
        assert!(bytes.len() >= 8, "{name}: a record shorter than its header");
        let opcode = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);

        let Some(reading) = read_record(class, &bytes, opcode) else {
            panic!(
                "{name}: class {class} has no decoder mapping and is not in \
                 UNCOVERED_CLASSES; add one or the other"
            )
        };

        match reading.verdict {
            Verdict::Decoded => decoded += 1,
            Verdict::NotImplemented(slug) => *gaps.entry((opcode, slug)).or_default() += 1,
            Verdict::WrongShape(slug) => wrong.push(format!(
                "{name} ({class} {selector}, opcode {opcode:#x}, {} bytes) -> {slug}",
                bytes.len()
            )),
        }
    }

    assert!(
        decoded > 0,
        "no fixture reached a decoder; the class mapping is broken"
    );

    // The gap map, printed every run. These are opcodes Apple emits that this
    // crate does not decode — the honest distance to covering the encoders,
    // measured against the serializer rather than against a guest workload.
    for ((opcode, slug), n) in &gaps {
        eprintln!("gap: opcode {opcode:#x} -> {slug} ({n} fixture(s))");
    }
    for (why, n) in &uncovered {
        eprintln!("uncovered: {n} fixture(s) — {why}");
    }
    eprintln!(
        "{decoded} of Apple's records decoded, {} opcode(s) not implemented, {} uncovered",
        gaps.len(),
        uncovered.values().sum::<usize>()
    );

    assert!(
        wrong.is_empty(),
        "the serializer produced {} record(s) a decoder refused for their shape. \
         Each is a layout this crate has wrong, and each loses the guest's \
         command:\n  {}",
        wrong.len(),
        wrong.join("\n  ")
    );
}

/// Set every bit the serializer did not write, leaving every bit it did.
///
/// The mask is measured per case by capturing it twice under complementary
/// fills, so a clear bit here did not move with the fill and a set bit did:
/// this record leaves that bit exactly as it found it. On a host arena the
/// leftovers are the harness's; in a guest they are whatever the command ring
/// last held.
fn repaint_unwritten(bytes: &[u8], mask: &[u8], one: bool) -> Vec<u8> {
    bytes
        .iter()
        .zip(mask)
        .map(|(b, m)| if one { b | !m } else { b & m })
        .collect()
}

/// No decoder's answer may depend on a byte the serializer never wrote.
///
/// This is the other half of the divergence instrument, and it exists because
/// the shape test above cannot see the bug class that has now produced three
/// findings on its own:
///
/// * `copyFromTexture:toBuffer:` loaded `options` two bytes wide, so a plain
///   copy was declined whenever the ring held non-zero bytes past the field;
/// * `useOffset` in the heap-texture record is **one bit**, and `compute_exec`
///   loaded the four-byte slot around it, dropping the texture on a dirty ring;
/// * the sampler and the format-only texture view both stop short of their
///   allocation, and the tail is nobody's.
///
/// Every one is invisible to a test that decodes Apple's buffer once, because a
/// capture arena is zero-filled exactly where a guest's ring is not — the
/// fixture agrees with the over-wide read by accident. So each record is
/// decoded twice more, with its unwritten bits all-zero and then all-one, and
/// the two answers must be the same answer. They differ only if some decoder
/// read one of those bits, which is a decoder reading noise.
///
/// A record whose every bit is written is skipped, and the count of records
/// that had something to poison is asserted non-zero: a mask that arrived empty
/// would make this test pass by testing nothing.
#[test]
#[cfg_attr(not(wire_fixtures), ignore = "run scripts/wire-oracle/wire-oracle.sh")]
fn no_decoder_reads_a_bit_apples_serializer_never_wrote() {
    let root = fixtures();

    let mut poisoned = 0usize;
    let mut fully_written = 0usize;
    let mut noise: std::collections::BTreeMap<String, String> = Default::default();

    for case in root["cases"].as_array().expect("cases array") {
        let name = case["name"].as_str().expect("case name");
        let class = case["class"].as_str().expect("case class");
        let selector = case["selector"].as_str().expect("case selector");
        if !carries_a_guest_opcode(selector) {
            continue;
        }

        let bytes = unhex(case["buffer"].as_str().expect("buffer hex"));
        let mask = unhex(
            case["written_mask"]
                .as_str()
                .unwrap_or_else(|| panic!("{name}: no written_mask; regenerate the fixtures")),
        );
        assert_eq!(
            mask.len(),
            bytes.len(),
            "{name}: the mask and the record are different lengths"
        );
        if mask.iter().all(|m| *m == 0xff) {
            fully_written += 1;
            continue;
        }

        let opcode = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let Some(clean) = read_record(class, &bytes, opcode) else {
            continue;
        };
        // A gap carries no signature — there is no decoder to have read
        // anything. Poisoning it would only compare two empty strings.
        if matches!(clean.verdict, Verdict::NotImplemented(_)) {
            continue;
        }
        poisoned += 1;

        let zeros = repaint_unwritten(&bytes, &mask, false);
        let ones = repaint_unwritten(&bytes, &mask, true);
        let lo = read_record(class, &zeros, opcode)
            .expect("same class")
            .signature;
        let hi = read_record(class, &ones, opcode)
            .expect("same class")
            .signature;
        if lo != hi {
            // Keyed by record rather than by case: every fixture of one record
            // shares its layout, so a wide read reports once with the first
            // case that showed it rather than once per perturbation.
            noise
                .entry(format!(
                    "{class} opcode {opcode:#x} ({} bytes)",
                    bytes.len()
                ))
                .or_insert_with(|| {
                    format!("{name}\n      unwritten=0 -> {lo}\n      unwritten=1 -> {hi}")
                });
        }
    }

    assert!(
        poisoned > 0,
        "no record had an unwritten bit to poison; either the masks are empty \
         or every partially-written record lost its decoder"
    );
    eprintln!(
        "poisoned {poisoned} record(s) with unwritten bits; {fully_written} were written end to end"
    );

    assert!(
        noise.is_empty(),
        "{} record(s) decode differently depending on bits Apple's serializer \
         never wrote. On a real wire those bits are the guest's stale ring, so \
         each is a decoder reading noise:\n  {}",
        noise.len(),
        noise
            .iter()
            .map(|(record, detail)| format!("{record}\n    {detail}"))
            .collect::<Vec<_>>()
            .join("\n  ")
    );
}
