//! A fuzz of the decoder over the bytes a guest process controls: the messages its context sends
//! the decoder (creates, destroys, the region chunks they name) and the op streams of its command
//! buffers, which any process writes into its own rings. A panic there unwinds out of the host's
//! decode and ends the VM's process, and none of the decoder's refusals catches one
//! (`refusals::refusing` lets any other unwind go on), so the decoder must answer every input with
//! a result: carried out, refused, or a fault in the refusal scope (the decoder's `refusing`, as
//! `GraphicsDevice::handle_stage3` runs it).
//!
//! Each target is a plain `#[test]` over a seeded generator: inputs are made from words that lean
//! to the values the wire's fields take and the edges past them (object ids the decoder holds, ids
//! next to them, offsets at the context's block and at the region's end, sizes and counts around 0,
//! powers of two and the widths' maxima). A run is reproducible from its seed: case `i` of a
//! target is made from a seed of its own, a mix of the run's and `i`, and a failure prints both,
//! with the input's bytes. `RGFX_FUZZ_SEED` sets the run's seed (default fixed, so a check sees the
//! same inputs each run), `RGFX_FUZZ_CASES` the number of cases, `RGFX_FUZZ_INDEX` replays one
//! case.
//!
//! The command buffer target also checks what the decoder tells its guest of a command buffer: one
//! it carried out (`Ok(Ok(()))`) gets exactly one completion, and one it did not gets none from the
//! decoder (the device posts its failure: `GraphicsDevice::report`), so no buffer is answered twice
//! or left without an answer; and a stream holding an op of a tag no group decodes is never
//! carried out. The answer target checks that a command buffer answered on a doorbell's thread is
//! carried out as one its queue's thread takes unanswered: the same platform calls in the same
//! order, so the same commits, and the same outcomes and records, in the same order.

use std::collections::HashMap;
use std::sync::Arc;

use reims_graphics::ops::{
    RGFX_COMMAND_BUFFER_OPS_OFFSET, RGFX_OP_ENCODER_BEGIN_BLIT, RGFX_OP_ENCODER_BEGIN_COMPUTE,
    RGFX_OP_ENCODER_BEGIN_RENDER, RGFX_OP_ENCODER_END, RGFX_OP_SUBMITTED, RGFX_PASS_FLAG_RT_SIZE,
    RgfxCommandBuffer, RgfxOpHeader, RgfxRenderPassColorAttachment, RgfxRenderPassDesc,
};
use reims_graphics::region::Region;
use reims_graphics::wire::{
    RGFX_CMD_BUFFER_CREATE, RGFX_CMD_COMMAND_BUFFER, RGFX_CMD_EVENT_CREATE, RGFX_CMD_FENCE_CREATE,
    RGFX_CMD_QUEUE_CREATE, RGFX_CMD_SHARED_EVENT_CREATE, RGFX_CMD_TEXTURE_CREATE,
    RGFX_CONTEXT_ALLOCATIONS_OFFSET, RGFX_CURSOR_SPAN, RgfxBufferCreate, RgfxMsgHeader,
    RgfxObjectCreate, RgfxQueueCreate, RgfxSharedEventCreate, RgfxTextureCreate,
    RgfxTextureDescriptor,
};
use reims_vgpu_core::sync::Mutex;
use reims_vgpu_core::testkit::FakePlatform;
use reims_vgpu_core::testkit::objects::CompleteOn;
use zerocopy::{FromBytes, IntoBytes};

use super::{DecodeError, GraphicsDecoder, ObjHandle, refusing};

/// SplitMix64 (Steele, Lea and Flood, 2014): small, fast and seedable.
struct Rng(u64);

/// SplitMix64's output function, which mixes one word into another.
const fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        mix(self.0)
    }

    /// A value in `0..n`.
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    fn one_in(&mut self, n: u64) -> bool {
        self.below(n) == 0
    }
}

/// The context the decoder's messages come from, and the block of the region it holds.
const CONTEXT: u32 = 0;
const BLOCK: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
const BLOCK_SIZE: u64 = 0x10_0000;
const REGION_SIZE: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + BLOCK_SIZE + RGFX_CURSOR_SPAN;

/// The objects `setup` makes: queue 1, buffers 2 and 3, textures 4 and 5 (resource ids 4 and 5),
/// event 6, shared event 7, fence 8. A fuzzed create takes the ids after them.
const OBJECTS: u64 = 8;

/// A word of a fuzzed message or op: one of the values the wire's fields take, or an edge past
/// them, or any.
fn word(rng: &mut Rng) -> u64 {
    match rng.below(16) {
        // An object id, held or next to one.
        0..=3 => rng.below(OBJECTS + 4),
        // A small count, index or enum.
        4 => rng.below(64),
        // A power of two, and a mask or a width's maximum.
        5 => 1u64 << rng.below(64),
        6 => (1u64 << rng.below(64)).wrapping_sub(1),
        // An offset in (or just past) the context's block, and at the region's end.
        7 => BLOCK + 8 * rng.below(BLOCK_SIZE / 8 + 2),
        8 => REGION_SIZE.wrapping_sub(8 * rng.below(4)),
        // Near the top of 64 and of 32 bits.
        9 => u64::MAX - rng.below(4),
        10 => u64::from(u32::MAX) - rng.below(4),
        // A size or offset, and any 32-bit value.
        11 => rng.below(0x1_0000),
        12 => u64::from(rng.next() as u32),
        // An object id in the low of two 32-bit fields.
        13 => rng.below(OBJECTS + 4) | (rng.below(4) << 32),
        // A double's bits.
        14 => f64::from(rng.below(5000) as u32).to_bits(),
        _ => rng.next(),
    }
}

/// `n` fuzzed words, as bytes; one in four 32-bit halves is replaced by a small value, since many
/// fields are 32 bits wide.
fn words(rng: &mut Rng, n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n * 8);
    for _ in 0..n {
        let mut w = word(rng);
        if rng.one_in(4) {
            w = (w & !0xffff_ffff) | rng.below(OBJECTS + 4);
        }
        out.extend_from_slice(&w.to_le_bytes());
    }
    out
}

fn header(r#type: u32, size: usize) -> RgfxMsgHeader {
    RgfxMsgHeader {
        r#type,
        size: size as u32,
        seqno: 0,
    }
}

/// The record a decoder hands its sinks, kept in order.
type Seen = Arc<Mutex<Vec<(&'static str, u64)>>>;

/// A decoder for context 0, which holds its block of the region and resource-id chunk 0, with
/// `setup`'s objects made, and command buffers that complete at their commit; what its sinks were
/// handed, by kind and command buffer.
fn setup() -> (Arc<FakePlatform>, GraphicsDecoder<FakePlatform>, Seen) {
    let p = Arc::new(FakePlatform::new());
    p.world().set_auto_complete(Some(CompleteOn::Thread));
    let region = Arc::new(Region::new(REGION_SIZE as usize).unwrap());
    region.grant(CONTEXT, BLOCK, BLOCK_SIZE);
    let mut dec = GraphicsDecoder::<FakePlatform>::new(Arc::clone(&p), region);
    let seen: Seen = Arc::default();
    let s = Arc::clone(&seen);
    dec.set_completion_sink(Arc::new(move |r| s.lock().push(("complete", r.cb_obj_id))));
    let s = Arc::clone(&seen);
    dec.set_scheduled_sink(Arc::new(move |r| s.lock().push(("scheduled", r.cb_obj_id))));
    let s = Arc::clone(&seen);
    dec.set_submitted_sink(Arc::new(move |r| s.lock().push(("submitted", r.cb_obj_id))));
    dec.set_signal_sink(Arc::new(|_| {}));
    dec.set_context(Some(CONTEXT));
    dec.hold_resource_ids(CONTEXT, 0);
    let device = p.device();
    let mut made = |bytes: &[u8]| {
        assert_eq!(dec.decode(&device, bytes).unwrap(), Ok(()));
    };
    made(
        RgfxQueueCreate {
            header: header(RGFX_CMD_QUEUE_CREATE, size_of::<RgfxQueueCreate>()),
            obj_id: 1,
            ..RgfxQueueCreate::default()
        }
        .as_bytes(),
    );
    for obj_id in [2, 3] {
        made(
            RgfxBufferCreate {
                header: header(RGFX_CMD_BUFFER_CREATE, size_of::<RgfxBufferCreate>()),
                obj_id,
                length: 65536,
                ..RgfxBufferCreate::default()
            }
            .as_bytes(),
        );
    }
    for obj_id in [4, 5] {
        made(
            RgfxTextureCreate {
                header: header(RGFX_CMD_TEXTURE_CREATE, size_of::<RgfxTextureCreate>()),
                obj_id,
                resource_id: obj_id,
                desc: RgfxTextureDescriptor {
                    pixel_format: 80, // BGRA8Unorm
                    texture_type: 2,  // 2D
                    width: 256,
                    height: 256,
                    depth: 1,
                    mipmap_level_count: 1,
                    array_length: 1,
                    sample_count: 1,
                    usage: 5, // ShaderRead | RenderTarget
                    ..RgfxTextureDescriptor::default()
                },
                ..RgfxTextureCreate::default()
            }
            .as_bytes(),
        );
    }
    let create = |r#type, obj_id| RgfxObjectCreate {
        header: header(r#type, size_of::<RgfxObjectCreate>()),
        obj_id,
    };
    made(create(RGFX_CMD_EVENT_CREATE, 6).as_bytes());
    made(
        RgfxSharedEventCreate {
            header: header(
                RGFX_CMD_SHARED_EVENT_CREATE,
                size_of::<RgfxSharedEventCreate>(),
            ),
            obj_id: 7,
            ..RgfxSharedEventCreate::default()
        }
        .as_bytes(),
    );
    made(create(RGFX_CMD_FENCE_CREATE, 8).as_bytes());
    (p, dec, seen)
}

/// The types a context's messages to the decoder take (the kext passes a process's creates and
/// destroys, `OBJECT_DESTROY` through `ARGUMENT_LAYOUT`'s neighbours, and the queues' rings carry
/// `COMMAND_BUFFER`s), and one in sixteen any type.
fn message_type(rng: &mut Rng) -> u32 {
    const TYPES: [u32; 32] = [
        0x201, 0x202, 0x203, 0x204, 0x205, 0x206, 0x207, 0x208, 0x209, 0x20a, 0x20b, 0x20c, 0x20d,
        0x20e, 0x20f, 0x210, 0x211, 0x212, 0x213, 0x214, 0x215, 0x216, 0x217, 0x218, 0x219, 0x21a,
        0x21b, 0x21c, 0x21d, 0x2f1, 0x300, 0x301,
    ];
    if rng.one_in(16) {
        return rng.next() as u32;
    }
    TYPES[rng.below(TYPES.len() as u64) as usize]
}

/// A fuzzed message: a header of `type`, its size the message's (one in eight another), and
/// fuzzed words, from none to a few hundred, one in eight a byte count that is not whole words.
fn message(rng: &mut Rng) -> Vec<u8> {
    let r#type = message_type(rng);
    let n = match rng.below(8) {
        0 => rng.below(4) as usize,
        1..=5 => rng.below(48) as usize,
        _ => rng.below(400) as usize,
    };
    let mut body = words(rng, n);
    if rng.one_in(8) {
        body.truncate(body.len().saturating_sub(rng.below(8) as usize));
    }
    let size = if rng.one_in(8) {
        word(rng) as u32
    } else {
        (size_of::<RgfxMsgHeader>() + body.len()) as u32
    };
    let mut m = RgfxMsgHeader {
        r#type,
        size,
        seqno: rng.below(1 << 20),
    }
    .as_bytes()
    .to_vec();
    m.extend_from_slice(&body);
    m
}

/// `from`, a message the decoder carried out earlier in the case, mutated: one to four of its words
/// fuzzed (its header's among them, its type with them), its type another, or its end cut or
/// grown, so the creates the decoder took lead to messages over the objects they made and to the
/// edges of the values it took.
fn mutated(rng: &mut Rng, from: &[u8]) -> Vec<u8> {
    let mut m = from.to_vec();
    match rng.below(4) {
        0 => {
            let t = message_type(rng);
            m[..4].copy_from_slice(&t.to_le_bytes());
        }
        1 => {
            let keep = rng.below(m.len() as u64 / 8 + 1) as usize * 8;
            m.truncate(keep.max(size_of::<RgfxMsgHeader>()));
            let more = rng.below(8) as usize;
            m.extend(words(rng, more));
            let size = m.len() as u32;
            m[4..8].copy_from_slice(&size.to_le_bytes());
        }
        _ => {
            for _ in 0..1 + rng.below(4) {
                let at = rng.below(m.len() as u64 / 8) as usize * 8;
                let w = if rng.one_in(2) {
                    word(rng)
                } else {
                    // A nearby value: an off-by-one, a doubling, a sign flip.
                    let old = u64::from_le_bytes(m[at..at + 8].try_into().unwrap());
                    match rng.below(4) {
                        0 => old.wrapping_add(1),
                        1 => old.wrapping_sub(1),
                        2 => old.wrapping_mul(2),
                        _ => old ^ (1 << 31),
                    }
                };
                // Mostly the low half alone, since many fields are 32 bits wide.
                let bytes = w.to_le_bytes();
                let n = if rng.one_in(2) { 4 } else { 8 };
                m[at..at + n].copy_from_slice(&bytes[..n]);
            }
        }
    }
    m
}

/// The highest tag each op group decodes: framing, render, compute, blit, video.
const LAST_TAG: [u16; 5] = [0x0009, 0x104f, 0x201d, 0x3019, 0x4000];

/// One op: its header (its `len` the op's, one in sixteen another) and `body`.
fn op(rng: &mut Rng, tag: u16, body: &[u8]) -> Vec<u8> {
    let len = if rng.one_in(16) {
        word(rng) as u32
    } else {
        (8 + body.len()) as u32
    };
    let mut v = RgfxOpHeader { tag, flags: 0, len }.as_bytes().to_vec();
    v.extend_from_slice(body);
    v
}

/// A tag of `group` (0 framing, 1 render, ...): one it decodes, or one in sixteen past its last.
fn tag_of(rng: &mut Rng, group: u16) -> u16 {
    let base = group << 12;
    let last = LAST_TAG[group as usize] - base;
    if rng.one_in(16) {
        base + last + 1 + rng.below(0x40) as u16
    } else {
        base + rng.below(u64::from(last) + 1) as u16 + u16::from(group == 0)
    }
}

/// Whether `tag` is one no group decodes.
fn unknown(tag: u16) -> bool {
    let group = (tag >> 12) as usize;
    group >= LAST_TAG.len() || tag > LAST_TAG[group] || tag == 0
}

/// A render pass over texture 4 (colour, clear then store), 256 × 256.
fn render_pass() -> Vec<u8> {
    let mut body = RgfxRenderPassDesc {
        color_count: 1,
        flags: RGFX_PASS_FLAG_RT_SIZE,
        render_target_width: 256,
        render_target_height: 256,
        ..RgfxRenderPassDesc::default()
    }
    .as_bytes()
    .to_vec();
    body.extend_from_slice(
        RgfxRenderPassColorAttachment {
            texture_obj_id: 4,
            load_action: 2,
            store_action: 1,
            ..RgfxRenderPassColorAttachment::default()
        }
        .as_bytes(),
    );
    body
}

/// A fuzzed op stream. Most streams open
/// one encoder as the bundle does (a render pass over texture 4, a compute or a blit pass), then
/// carry fuzzed ops of its group, then end it; the rest are fuzzed ops of any group, framing ops
/// among them. No stream waits for an event or uses an IOSurface's device cache, so each buffer
/// the decoder carries out completes at its commit.
fn op_stream(rng: &mut Rng) -> Vec<u8> {
    let mut ops = Vec::new();
    let fuzzed = |rng: &mut Rng, group: u16, ops: &mut Vec<u8>| {
        let mut tag = tag_of(rng, group);
        // Framing's event and surface ops wait or fill, so a buffer that holds one need not
        // complete at its commit: none here.
        if group == 0 && matches!(tag, 5..=7) {
            tag = 8;
        }
        let n = match rng.below(4) {
            0 => rng.below(3),
            _ => rng.below(24),
        } as usize;
        let body = words(rng, n);
        ops.extend(op(rng, tag, &body));
    };
    if rng.one_in(4) {
        for _ in 0..rng.below(24) {
            let group = rng.below(5) as u16;
            fuzzed(rng, group, &mut ops);
        }
    } else {
        let group = 1 + rng.below(3) as u16;
        let (begin, body) = match group {
            1 => (RGFX_OP_ENCODER_BEGIN_RENDER, render_pass()),
            2 => (RGFX_OP_ENCODER_BEGIN_COMPUTE, Vec::new()),
            _ => (RGFX_OP_ENCODER_BEGIN_BLIT, Vec::new()),
        };
        ops.extend(op(rng, begin, &body));
        for _ in 0..1 + rng.below(16) {
            fuzzed(rng, group, &mut ops);
        }
        if !rng.one_in(16) {
            ops.extend(op(rng, RGFX_OP_ENCODER_END, &[]));
        }
    }
    ops
}

/// The tags of the ops `ops` frames, as the decoder's walk reads them (each op's `len` whole
/// words, its header's at least, within the stream), or `None` where the framing breaks.
fn framed_tags(mut ops: &[u8]) -> Option<Vec<u16>> {
    let mut tags = Vec::new();
    while !ops.is_empty() {
        let head = ops.get(..8)?;
        let tag = u16::from_le_bytes([head[0], head[1]]);
        let len = u32::from_le_bytes(head[4..8].try_into().unwrap()) as usize;
        if len < 8 || !len.is_multiple_of(8) || len > ops.len() {
            return None;
        }
        tags.push(tag);
        ops = &ops[len..];
    }
    Some(tags)
}

/// A `COMMAND_BUFFER` on queue 1 (one in sixteen another queue) of command buffer `cb_obj_id`,
/// carrying `ops`; its `op_bytes` theirs, one in sixteen another.
fn command_buffer(rng: &mut Rng, cb_obj_id: u64, ops: &[u8]) -> Vec<u8> {
    let op_bytes = if rng.one_in(16) {
        word(rng)
    } else {
        ops.len() as u64
    };
    let m = RgfxCommandBuffer {
        header: header(
            RGFX_CMD_COMMAND_BUFFER,
            RGFX_COMMAND_BUFFER_OPS_OFFSET as usize + ops.len(),
        ),
        queue_obj_id: if rng.one_in(16) { word(rng) } else { 1 },
        cb_obj_id,
        op_bytes,
    };
    let mut bytes = m.as_bytes().to_vec();
    bytes.extend_from_slice(ops);
    bytes
}

/// The run's seed and case count, and the one case to replay, from the environment.
fn run(name: &str, default_cases: u64) -> (u64, std::ops::Range<u64>) {
    let var = |key: &str| {
        std::env::var(key).ok().map(|v| {
            let v = v.trim();
            match v.strip_prefix("0x") {
                Some(hex) => u64::from_str_radix(hex, 16),
                None => v.parse(),
            }
            .unwrap_or_else(|_| panic!("{key}={v} is not a number"))
        })
    };
    let seed = var("RGFX_FUZZ_SEED").unwrap_or(0x5247_4658);
    let cases = var("RGFX_FUZZ_CASES").unwrap_or(default_cases);
    let seed = name.bytes().fold(seed, |s, b| mix(s ^ u64::from(b)));
    match var("RGFX_FUZZ_INDEX") {
        Some(i) => (seed, i..i + 1),
        None => (seed, 0..cases),
    }
}

/// The input being decoded, printed where the decode panics: the test's thread unwinds through
/// it, and the panic's own message and backtrace follow.
struct Input<'m> {
    case: u64,
    case_seed: u64,
    what: &'static str,
    index: u64,
    bytes: &'m [u8],
}

impl Drop for Input<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            eprintln!(
                "decoding case {} (RGFX_FUZZ_SEED's mix {:#x}), {} {}, panicked: {}",
                self.case,
                self.case_seed,
                self.what,
                self.index,
                hex(self.bytes)
            );
        }
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Every message a context sends the decoder, in sequences of up to 48 over its objects, half of
/// them mutated from those it carried out before, is answered with a result: none panics
#[test]
fn no_message_a_context_sends_ends_the_host() {
    let (seed, cases) = run("messages", 384);
    let mut outcomes = Outcomes::default();
    for case in cases {
        let case_seed = mix(seed ^ case);
        let mut rng = Rng(case_seed);
        let (p, mut dec, _) = setup();
        let device = p.device();
        // The case's messages the decoder carried out, which later ones are mutated from.
        let mut taken: Vec<Vec<u8>> = Vec::new();
        for i in 0..1 + rng.below(48) {
            let m = if !taken.is_empty() && rng.one_in(2) {
                let from = rng.below(taken.len() as u64) as usize;
                mutated(&mut rng, &taken[from])
            } else {
                message(&mut rng)
            };
            let input = Input {
                case,
                case_seed,
                what: "message",
                index: i,
                bytes: &m,
            };
            let outcome = refusing(|| dec.decode(&device, &m));
            drop(input);
            outcomes.add(u32::from_le_bytes(m[..4].try_into().unwrap()), &outcome);
            if matches!(outcome, Ok(Ok(()))) && m.len() >= size_of::<RgfxMsgHeader>() + 8 {
                taken.push(m);
            }
        }
    }
    outcomes.print("messages");
}

/// How many fuzzed inputs of each type were carried out, refused and faulted, printed with the
/// run's output (`--no-capture`), so a generator that stops reaching the decoder's work shows.
#[derive(Default)]
struct Outcomes(std::collections::BTreeMap<u32, [u64; 3]>);

impl Outcomes {
    fn add<E>(
        &mut self,
        r#type: u32,
        outcome: &reims_vgpu_core::raise::RaiseResult<Result<(), E>>,
    ) {
        let r#type = if (0x200..0x400).contains(&r#type) {
            r#type
        } else {
            0
        };
        let at = match outcome {
            Ok(Ok(())) => 0,
            Ok(Err(_)) => 1,
            Err(_) => 2,
        };
        self.0.entry(r#type).or_default()[at] += 1;
    }

    fn print(&self, what: &str) {
        let line: Vec<String> = self
            .0
            .iter()
            .map(|(t, [done, refused, faulted])| format!("{t:#x} {done}/{refused}/{faulted}"))
            .collect();
        eprintln!(
            "{what}, by type (carried out/refused/faulted): {}",
            line.join(", ")
        );
    }
}

/// Every command buffer a context sends is answered with a result, none panicking: one carried
/// out gets exactly one completion from the decoder, and one not carried out none, which the
/// device then posts; and a stream holding an op no group decodes is not carried out
#[test]
fn no_command_buffer_ends_the_host_or_goes_unanswered() {
    let (seed, cases) = run("command buffers", 768);
    let mut carried = 0;
    let mut outcomes = Outcomes::default();
    for case in cases.clone() {
        let case_seed = mix(seed ^ case);
        let mut rng = Rng(case_seed);
        let (p, mut dec, seen) = setup();
        let device = p.device();
        for i in 0..1 + rng.below(4) {
            let cb_obj_id = OBJECTS + 1 + i;
            let ops = op_stream(&mut rng);
            let m = command_buffer(&mut rng, cb_obj_id, &ops);
            // The ops the decoder walks: all of the message's, which its `op_bytes` must count
            // (`a_command_buffer_s_op_bytes_are_its_message_s_ops`).
            let op_bytes = RgfxCommandBuffer::read_from_prefix(&m).unwrap().0.op_bytes;
            let walked = &m[RGFX_COMMAND_BUFFER_OPS_OFFSET as usize..];
            let counted = op_bytes == walked.len() as u64;
            let holds_unknown =
                framed_tags(walked).is_some_and(|tags| tags.into_iter().any(unknown));
            let what = || {
                format!(
                    "case {case} (RGFX_FUZZ_SEED's mix {case_seed:#x}), command buffer {i}: {}",
                    hex(&m)
                )
            };
            let input = Input {
                case,
                case_seed,
                what: "command buffer",
                index: i,
                bytes: &m,
            };
            let outcome = refusing(|| dec.decode(&device, &m));
            drop(input);
            let done = matches!(outcome, Ok(Ok(())));
            carried += u64::from(done);
            outcomes.add(RGFX_CMD_COMMAND_BUFFER, &outcome);
            let mut answers: HashMap<&str, usize> = HashMap::new();
            for (kind, cb) in seen.lock().drain(..) {
                if cb == cb_obj_id {
                    *answers.entry(kind).or_default() += 1;
                }
            }
            let completes = answers.get("complete").copied().unwrap_or(0);
            assert_eq!(
                completes,
                usize::from(done),
                "{}: decoded {outcome:?}, completed {completes} times",
                what()
            );
            assert!(
                !(done && holds_unknown),
                "{}: carried out, though an op's tag is none a group decodes",
                what()
            );
            assert!(
                !(done && !counted),
                "{}: carried out, though its op_bytes are not its ops",
                what()
            );
            if let Ok(Err(e)) = &outcome {
                assert_ne!(*e, DecodeError::UnknownCommand, "{}", what());
            }
        }
    }
    outcomes.print("command buffers");
    // The generator reaches the walk's end often enough to check its answers.
    if cases.end - cases.start > 100 {
        assert!(
            carried > 0,
            "no command buffer of {} cases was carried out",
            cases.end
        );
    }
}

/// A `COMMAND_BUFFER` whose `op_bytes` is fewer than the ops its message carries is not carried
/// out: the bundle writes a buffer's message as its fixed part and its whole op stream, `op_bytes`
/// the stream's length (`RgfxEncoder::finish`), so a message that disagrees is not one the guest
/// made, and carrying out the ops its `op_bytes` covers drops the rest without a word, the buffer
/// answered completed. Found by `no_command_buffer_ends_the_host_or_goes_unanswered`: a buffer
/// whose `op_bytes` read 0 was carried out as an empty one and completed, its blit pass dropped.
#[test]
fn a_command_buffer_s_op_bytes_are_its_message_s_ops() {
    let (p, mut dec, seen) = setup();
    let device = p.device();
    let plain = |tag: u16| {
        RgfxOpHeader {
            tag,
            flags: 0,
            len: 8,
        }
        .as_bytes()
        .to_vec()
    };
    let mut ops = plain(RGFX_OP_ENCODER_BEGIN_BLIT);
    ops.extend(plain(RGFX_OP_ENCODER_END));
    for op_bytes in [0, 8] {
        let cb_obj_id = 20 + op_bytes;
        let mut m = RgfxCommandBuffer {
            header: header(
                RGFX_CMD_COMMAND_BUFFER,
                RGFX_COMMAND_BUFFER_OPS_OFFSET as usize + ops.len(),
            ),
            queue_obj_id: 1,
            cb_obj_id,
            op_bytes,
        }
        .as_bytes()
        .to_vec();
        m.extend_from_slice(&ops);
        let outcome = refusing(|| dec.decode(&device, &m));
        assert!(
            !matches!(outcome, Ok(Ok(()))),
            "op_bytes {op_bytes} of {} bytes of ops: carried out",
            ops.len()
        );
        assert!(!seen.lock().contains(&("complete", cb_obj_id)));
    }
}

/// What a run of [`an_answer_at_the_doorbell_keeps_the_order_of_a_take`]'s case saw: the
/// platform's calls, in order; each message's outcome, in order; and what the decoder posted.
type Run = (Vec<&'static str>, Vec<String>, Vec<(&'static str, u64)>);

/// A command buffer answered on a doorbell's thread before its begin is carried out as one its
/// queue's thread begins unanswered. Over sequences of up to eight command buffers of queue 1,
/// half of which ask for their `CB_SUBMITTED`, with the context's other messages between them, run
/// once with each buffer begun as the queue's thread begins it (`begin_taken`) and once with each
/// offered to the answer first, the platform sees the same calls in the same order (so the same
/// commits, in the same order), each message gets the same outcome, and the decoder posts the
/// same records in the same order, each buffer's `CB_SUBMITTED` once.
#[test]
fn an_answer_at_the_doorbell_keeps_the_order_of_a_take() {
    let (seed, cases) = run("answers", 256);
    let mut answered = 0;
    for case in cases.clone() {
        let case_seed = mix(seed ^ case);
        let mut run_case = |answering: bool| -> Run {
            let mut rng = Rng(case_seed);
            let (p, mut dec, seen) = setup();
            p.world().set_auto_complete(Some(CompleteOn::Caller));
            let device = p.device();
            let from = p.call_log().mark();
            let mut outcomes = Vec::new();
            for i in 0..1 + rng.below(8) {
                if rng.one_in(3) {
                    let m = message(&mut rng);
                    outcomes.push(format!("{:?}", refusing(|| dec.decode(&device, &m))));
                    continue;
                }
                let cb_obj_id = OBJECTS + 1 + i;
                let mut ops = op_stream(&mut rng);
                if rng.one_in(2) {
                    ops.extend(op(&mut rng, RGFX_OP_SUBMITTED, &[]));
                }
                let m = command_buffer(&mut rng, cb_obj_id, &ops);
                let queue = RgfxCommandBuffer::read_from_prefix(&m)
                    .unwrap()
                    .0
                    .queue_obj_id;
                // A buffer runs on the queue whose ring carried it: queue 1's alone here.
                if queue != 1 {
                    continue;
                }
                if answering {
                    let posted = Arc::clone(&seen);
                    let yes = dec.decoders().answer_at_doorbell(CONTEXT, &m, move |r| {
                        posted.lock().push(("submitted", r.cb_obj_id));
                        true
                    });
                    answered += u64::from(yes);
                }
                let begun = refusing(|| dec.begin_taken(&device, &m, Some(queue)));
                let outcome = match begun {
                    Ok(Ok(begun)) => refusing(|| dec.finish_command_buffer(&device, &m, begun)),
                    Ok(Err(e)) => Ok(Err(e)),
                    Err(raise) => Err(raise),
                };
                let stood_in = !matches!(outcome, Ok(Ok(())))
                    && matches!(refusing(|| dec.stand_in_for_failed(&device, &m)), Ok(true));
                dec.end_message();
                outcomes.push(format!("{outcome:?} {stood_in}"));
            }
            // What the answer kept on the queue, where its buffer was never begun, is let go of.
            let queue_left = matches!(dec.named(1), Some(ObjHandle::CommandQueue(_)));
            if queue_left {
                assert!(
                    dec.queue_counts(1).unwrap().answered.lock().is_none(),
                    "case {case}: an answer left on the queue"
                );
            }
            let seen = seen.lock().clone();
            (p.call_log().sels_since(from), outcomes, seen)
        };
        let taken = run_case(false);
        let answered_first = run_case(true);
        assert_eq!(
            taken, answered_first,
            "case {case} (RGFX_FUZZ_SEED's mix {case_seed:#x}): the answer changed what the \
             platform or the guest sees"
        );
    }
    if cases.end - cases.start > 100 {
        assert!(
            answered > 0,
            "no command buffer of {} cases was answered",
            cases.end
        );
    }
}
