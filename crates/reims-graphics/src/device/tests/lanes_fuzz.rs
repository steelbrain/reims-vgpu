//! A fuzz of ring 0's lanes: random interleavings of several processes' messages, each carried out
//! by a host that takes a random while over some of them, checked against an order worked out
//! here from the messages alone, in ring order (`device::lanes`' module doc):
//! - each context's messages are carried out in their ring order, and the presents in theirs;
//! - a present of a surface comes after every message published before it that changes the live
//!   textures over the surface's plane 0: one that makes one, destroys one, or closes a context
//!   that holds one;
//! - a shared block's release comes after every message published before it of each other context
//!   that shared the block since its last release;
//! - a resource id chunk or small chunk of ids another context held comes after that context's
//!   close published before it, and a new context's first message on its lane after every close
//!   published before it;
//! - a block's `SURFACE_PAGES` and `SURFACE_DOWNLOAD`s are carried out in their ring order; a
//!   context's message on its lane after the context's `SURFACE_PAGES` published before it; a
//!   buffer or heap over a block's pages, a buffer over a surface's pages and a block's release
//!   after every `SURFACE_PAGES` and `SURFACE_DOWNLOAD` of the block published before it; and a
//!   `SURFACE_PAGES` after its context's messages of those kinds published before it that name
//!   its block. The host takes longer over the blocks' messages than the rest, so the other kinds
//!   come while a block's lane is held;
//! - every message is carried out, once, and every answered message is answered once.
//!
//! A run is reproducible from its seed: case `i` is made from a mix of the run's seed and `i`, and
//! a failure prints both. `RGFX_LANES_FUZZ_SEED` sets the run's seed, `RGFX_LANES_FUZZ_CASES` the
//! number of cases.

use std::collections::{BTreeSet, HashMap, HashSet};

use super::*;

/// SplitMix64.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// When a message's carrying out began and ended, in one count over every message.
#[derive(Default)]
struct Seen {
    /// (seqno, began) in the order the host saw them.
    marks: Vec<(u64, bool)>,
    /// The textures over each surface's plane 0, by surface.
    named: HashMap<u32, BTreeSet<u64>>,
}

/// A host that records when it carries each message out ([`Seen`]), takes `slow`'s while over
/// some, answers a pipeline's create, a close, a release and a present at once, and names a
/// surface by the textures over its plane 0 it holds.
struct FuzzHost {
    seen: Arc<Mutex<Seen>>,
    /// The milliseconds the host takes over a message, by seqno.
    slow: Arc<HashMap<u64, u64>>,
}

impl FuzzHost {
    fn carry_out(&self, seqno: u64, body: impl FnOnce(&mut Seen)) {
        self.seen.lock().marks.push((seqno, true));
        if let Some(&ms) = self.slow.get(&seqno) {
            std::thread::sleep(Duration::from_millis(ms));
        }
        let mut seen = self.seen.lock();
        body(&mut seen);
        seen.marks.push((seqno, false));
    }
}

impl Host for FuzzHost {
    fn raise_interrupt(&self, _vector: u32) {}
    fn context_closed(&self, context: u32, answer: CompleteAnswer) {
        self.carry_out(answer.seqno, |seen| {
            for textures in seen.named.values_mut() {
                textures.retain(|&t| (t >> RGFX_CONTEXT_OBJ_ID_SHIFT) as u32 != context);
            }
        });
        answer.answer(RGFX_STATUS_OK);
    }
    fn block_released(&self, _release: &RgfxBlockRelease, answer: CompleteAnswer) {
        self.carry_out(answer.seqno, |_| {});
        answer.answer(RGFX_STATUS_OK);
    }
    fn surface_pages(
        &self,
        _pages: &RgfxSurfacePages,
        _runs: Vec<RgfxPageRun>,
        _flags: u32,
        answer: CompleteAnswer,
    ) {
        self.carry_out(answer.seqno, |_| {});
        answer.answer(RGFX_STATUS_OK);
    }
    fn surface_download(&self, _block_offset: u64, answer: CompleteAnswer) {
        self.carry_out(answer.seqno, |_| {});
        answer.answer(RGFX_STATUS_OK);
    }
    /// A download the host takes no while over is carried out where the control thread offers
    /// it; one it takes a while over (a copy to commit) is left to the block's lane.
    fn try_surface_download(
        &self,
        block_offset: u64,
        answer: CompleteAnswer,
    ) -> Result<(), CompleteAnswer> {
        if self.slow.contains_key(&answer.seqno) {
            return Err(answer);
        }
        self.surface_download(block_offset, answer);
        Ok(())
    }
    fn present_surface(&self, iosurface_id: u32, _generation: Option<u64>, answer: PresentAnswer) {
        let mut shown = iosurface_id == 0;
        self.carry_out(answer.seqno, |seen| {
            shown |= seen.named.get(&iosurface_id).is_some_and(|t| !t.is_empty());
        });
        answer.answer(if shown {
            RGFX_STATUS_OK
        } else {
            RGFX_STATUS_INVALID_ARGUMENT
        });
    }
    fn surface_textures(&self, iosurface_id: u32) -> Vec<u64> {
        let seen = self.seen.lock();
        (seen.named.get(&iosurface_id)).map_or_else(Vec::new, |t| t.iter().copied().collect())
    }
    fn stage3_message(&self, completion: &CompletionPoster, message: &[u8]) {
        let (header, _) = RgfxMsgHeader::read_from_prefix(message).unwrap();
        self.carry_out(header.seqno, |seen| match header.r#type {
            RGFX_CMD_TEXTURE_FROM_IOSURFACE => {
                let (m, _) = RgfxTextureFromIOSurface::read_from_prefix(message).unwrap();
                if m.plane == 0 {
                    seen.named
                        .entry(m.iosurface_id)
                        .or_default()
                        .insert(m.obj_id);
                }
            }
            RGFX_CMD_OBJECT_DESTROY => {
                let (m, _) = RgfxObjectDestroy::read_from_prefix(message).unwrap();
                for textures in seen.named.values_mut() {
                    textures.remove(&m.obj_id);
                }
            }
            _ => {}
        });
        if header.r#type == RGFX_CMD_RENDER_PIPELINE_CREATE {
            let mut reflection = RgfxPipelineReflection::new_zeroed();
            reflection.header = RgfxMsgHeader {
                r#type: RGFX_EVT_PIPELINE_REFLECTION,
                size: size_of::<RgfxPipelineReflection>() as u32,
                seqno: header.seqno,
            };
            completion.post(reflection.as_bytes());
        }
    }
}

/// The lane a message is carried out on, as the oracle works it out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lane {
    Context(u32),
    Presents,
    Block(u64),
}

/// The kinds on a context's lane that read what a block's lane makes of their block.
const OVER_BLOCK: [u32; 4] = [
    RGFX_CMD_BUFFER_OVER_PAGES,
    RGFX_CMD_HEAP_OVER_PAGES,
    RGFX_CMD_BUFFER_OVER_SURFACE_PAGES,
    RGFX_CMD_BLOCK_RELEASE,
];

/// One message of a case: its bytes, and what the order is worked out from.
#[derive(Clone, Debug)]
struct Sent {
    bytes: Vec<u8>,
    r#type: u32,
    /// The context it is of; 0 for a present or a download.
    context: u32,
    /// An object's id, a surface, a block's offset or a chunk, by type.
    what: u64,
    /// The block it names by offset, for the kinds a block's lane orders; 0 for none.
    block: u64,
    /// A texture's plane.
    plane: u32,
    /// Whether the host carries it out where it records it.
    recorded: bool,
    /// Whether ring 0's completion ring gets its answer.
    answered: bool,
}

/// The blocks the messages share and release.
const BLOCKS: [u64; 2] = [
    RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x10_0000,
    RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x20_0000,
];

/// A case's messages, in ring order.
fn case(rng: &mut Rng) -> Vec<Sent> {
    let mut sent = Vec::new();
    let mut live: Vec<u32> = Vec::new();
    let mut newest = 0u32;
    let mut objects: HashMap<u32, Vec<u64>> = HashMap::new();
    let mut next_obj = 1u64;
    let n = 16 + rng.below(48);
    for _ in 0..n {
        if live.is_empty() || (live.len() < 4 && rng.below(6) == 0) {
            newest += 1;
            live.push(newest);
        }
        let context = live[rng.below(live.len() as u64) as usize];
        let obj = |n: u64| (u64::from(context) << RGFX_CONTEXT_OBJ_ID_SHIFT) | n;
        let pick = rng.below(100);
        let message = |bytes: Vec<u8>, r#type, context, what, recorded, answered| Sent {
            bytes,
            r#type,
            context,
            what,
            block: 0,
            plane: 0,
            recorded,
            answered,
        };
        let pick = if rng.below(3) == 0 {
            100 + rng.below(30)
        } else {
            pick
        };
        let block = BLOCKS[rng.below(2) as usize];
        let one = match pick {
            0..20 => {
                let id = obj(next_obj);
                next_obj += 1;
                objects.entry(context).or_default().push(id);
                let r#type = [
                    RGFX_CMD_EVENT_CREATE,
                    RGFX_CMD_BUFFER_CREATE,
                    RGFX_CMD_RENDER_PIPELINE_CREATE,
                ][rng.below(3) as usize];
                message(
                    object_message(r#type, id),
                    r#type,
                    context,
                    id,
                    true,
                    r#type == RGFX_CMD_RENDER_PIPELINE_CREATE,
                )
            }
            20..40 => {
                let id = obj(next_obj);
                next_obj += 1;
                objects.entry(context).or_default().push(id);
                let (surface, plane) = (1 + rng.below(2) as u32, rng.below(3).min(1) as u32);
                Sent {
                    plane,
                    ..message(
                        texture_from_iosurface(id, surface, plane),
                        RGFX_CMD_TEXTURE_FROM_IOSURFACE,
                        context,
                        id,
                        true,
                        false,
                    )
                }
            }
            40..52 => match objects.get_mut(&context).filter(|o| !o.is_empty()) {
                Some(ids) => {
                    let id = ids.remove(rng.below(ids.len() as u64) as usize);
                    message(
                        object_message(RGFX_CMD_OBJECT_DESTROY, id),
                        RGFX_CMD_OBJECT_DESTROY,
                        context,
                        id,
                        true,
                        false,
                    )
                }
                None => continue,
            },
            52..66 => {
                let surface = rng.below(3);
                message(
                    present(surface as u32),
                    RGFX_CMD_PRESENT_SURFACE,
                    0,
                    surface,
                    true,
                    true,
                )
            }
            66..70 => {
                let chunk = 1 + rng.below(3);
                message(
                    id_chunk(context, chunk as u32),
                    RGFX_CMD_RESOURCE_ID_CHUNK,
                    context,
                    chunk,
                    true,
                    false,
                )
            }
            // Small chunks 4 to 15: those of chunks 1 to 3.
            70..74 => {
                let chunk = 4 + rng.below(12);
                message(
                    id_grant(RGFX_CMD_RESOURCE_ID_SMALL_CHUNK, context, chunk as u32),
                    RGFX_CMD_RESOURCE_ID_SMALL_CHUNK,
                    context,
                    chunk,
                    true,
                    false,
                )
            }
            74..84 => {
                let block = BLOCKS[rng.below(2) as usize];
                message(
                    block_message(RGFX_CMD_BLOCK_SHARE, context, block, 0x1000),
                    RGFX_CMD_BLOCK_SHARE,
                    context,
                    block,
                    false,
                    false,
                )
            }
            84..92 => Sent {
                block,
                ..message(
                    block_message(RGFX_CMD_BLOCK_RELEASE, context, block, 0x1000),
                    RGFX_CMD_BLOCK_RELEASE,
                    context,
                    block,
                    true,
                    true,
                )
            },
            100..110 => {
                // The lanes read no flag: any, one they do not know among them.
                let flags = [
                    0,
                    RGFX_SURFACE_PAGES_CACHED,
                    RGFX_SURFACE_PAGES_PRIVATE,
                    RGFX_SURFACE_PAGES_CACHED | (1 << 7),
                ][rng.below(4) as usize];
                Sent {
                    block,
                    ..message(
                        super::surface_pages_message(context, block, flags),
                        RGFX_CMD_SURFACE_PAGES,
                        context,
                        block,
                        true,
                        true,
                    )
                }
            }
            110..116 => Sent {
                block,
                ..message(
                    super::download_message(block),
                    RGFX_CMD_SURFACE_DOWNLOAD,
                    0,
                    block,
                    true,
                    true,
                )
            },
            116..124 => {
                let id = obj(next_obj);
                next_obj += 1;
                let r#type = OVER_BLOCK[rng.below(3) as usize];
                Sent {
                    block,
                    ..message(
                        super::over_block(r#type, id, block),
                        r#type,
                        context,
                        id,
                        true,
                        false,
                    )
                }
            }
            124..127 => message(
                block_message(RGFX_CMD_BLOCK_GRANT, context, block, 0x1000),
                RGFX_CMD_BLOCK_GRANT,
                context,
                block,
                false,
                false,
            ),
            127..130 => message(
                block_message(RGFX_CMD_BLOCK_UNSHARE, context, block, 0x1000),
                RGFX_CMD_BLOCK_UNSHARE,
                context,
                block,
                false,
                false,
            ),
            92..100 => {
                live.retain(|&c| c != context);
                message(
                    super::context_close(context),
                    RGFX_CMD_CONTEXT_CLOSE,
                    context,
                    u64::from(context),
                    true,
                    true,
                )
            }
            _ => unreachable!(),
        };
        sent.push(one);
    }
    sent
}

/// The pairs (earlier, later), by index in ring order, where the later message is carried out
/// only once the earlier one is (the module doc), worked out from the messages alone.
fn order(sent: &[Sent]) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    let mut first_of: HashMap<u32, usize> = HashMap::new();
    let mut holders: HashMap<u64, HashSet<u32>> = HashMap::new();
    // By small chunk: chunk `k` of 64 ids is small chunks `4k` to `4k + 3`.
    let mut chunk_holder: HashMap<u64, u32> = HashMap::new();
    // The live textures over each surface's plane 0, and the messages that changed them: a make,
    // a destroy of one, and the close of a context that held one.
    let mut over: HashMap<u64, HashSet<u64>> = HashMap::new();
    let mut changed: HashMap<u64, Vec<usize>> = HashMap::new();
    for (i, later) in sent.iter().enumerate() {
        let lane = later.lane();
        for (j, earlier) in sent[..i].iter().enumerate() {
            let same_lane = earlier.lane() == lane;
            // A context's message on its lane, after the context's pages on a block's.
            let after_pages = lane == Lane::Context(later.context)
                && earlier.r#type == RGFX_CMD_SURFACE_PAGES
                && earlier.context == later.context;
            // What reads a block's pages, after the block's lane.
            let reads_block =
                OVER_BLOCK.contains(&later.r#type) && earlier.lane() == Lane::Block(later.block);
            // A block's pages, after what their context made over the block.
            let made_over = later.r#type == RGFX_CMD_SURFACE_PAGES
                && OVER_BLOCK.contains(&earlier.r#type)
                && earlier.context == later.context
                && earlier.block == later.block;
            if same_lane || after_pages || reads_block || made_over {
                pairs.push((j, i));
            }
        }
        if matches!(lane, Lane::Context(_)) && !first_of.contains_key(&later.context) {
            first_of.insert(later.context, i);
            for (j, earlier) in sent[..i].iter().enumerate() {
                if earlier.r#type == RGFX_CMD_CONTEXT_CLOSE {
                    pairs.push((j, i));
                }
            }
        }
        match later.r#type {
            RGFX_CMD_TEXTURE_FROM_IOSURFACE if later.plane == 0 => {
                let surface = later.bytes_surface();
                over.entry(surface).or_default().insert(later.what);
                changed.entry(surface).or_default().push(i);
            }
            RGFX_CMD_OBJECT_DESTROY | RGFX_CMD_CONTEXT_CLOSE => {
                for (surface, textures) in &mut over {
                    let held = textures.len();
                    textures.retain(|&t| match later.r#type {
                        RGFX_CMD_OBJECT_DESTROY => t != later.what,
                        _ => (t >> RGFX_CONTEXT_OBJ_ID_SHIFT) as u32 != later.context,
                    });
                    if textures.len() != held {
                        changed.entry(*surface).or_default().push(i);
                    }
                }
            }
            RGFX_CMD_PRESENT_SURFACE if later.what != 0 => {
                for &j in changed.get(&later.what).into_iter().flatten() {
                    pairs.push((j, i));
                }
            }
            RGFX_CMD_BLOCK_SHARE => {
                holders.entry(later.what).or_default().insert(later.context);
            }
            RGFX_CMD_BLOCK_GRANT => {
                holders.remove(&later.what);
            }
            RGFX_CMD_BLOCK_RELEASE => {
                let held = holders.remove(&later.what).unwrap_or_default();
                for (j, earlier) in sent[..i].iter().enumerate() {
                    if earlier.context != later.context
                        && held.contains(&earlier.context)
                        && earlier.lane() == Lane::Context(earlier.context)
                    {
                        pairs.push((j, i));
                    }
                }
            }
            RGFX_CMD_RESOURCE_ID_CHUNK | RGFX_CMD_RESOURCE_ID_SMALL_CHUNK => {
                let small = if later.r#type == RGFX_CMD_RESOURCE_ID_CHUNK {
                    later.what * 4..later.what * 4 + 4
                } else {
                    later.what..later.what + 1
                };
                let mut held = HashSet::new();
                for unit in small {
                    let last = chunk_holder.insert(unit, later.context);
                    held.extend(last.filter(|&h| h != later.context));
                }
                for (j, earlier) in sent[..i].iter().enumerate() {
                    if earlier.r#type == RGFX_CMD_CONTEXT_CLOSE && held.contains(&earlier.context) {
                        pairs.push((j, i));
                    }
                }
            }
            _ => {}
        }
    }
    pairs
}

impl Sent {
    /// The lane it is carried out on.
    fn lane(&self) -> Lane {
        match self.r#type {
            RGFX_CMD_PRESENT_SURFACE => Lane::Presents,
            RGFX_CMD_SURFACE_PAGES | RGFX_CMD_SURFACE_DOWNLOAD => Lane::Block(self.block),
            _ => Lane::Context(self.context),
        }
    }

    /// A `TEXTURE_FROM_IOSURFACE`'s surface.
    fn bytes_surface(&self) -> u64 {
        let (m, _) = RgfxTextureFromIOSurface::read_from_prefix(&self.bytes).unwrap();
        u64::from(m.iosurface_id)
    }
}

/// The events on ring 0's completion ring now, waiting for none: each one's type and seqno.
fn events_now(rig: &mut Rig) -> Vec<(u32, u64)> {
    let data = RGFX_CONTROL_RING_DATA_SIZE as u64;
    let at = |count: u64| RGFX_CONTROL_COMPLETE_OFFSET + 256 + (count % data);
    let mut events = Vec::new();
    loop {
        let produced = (rig.region())
            .atomic_u64(RGFX_CONTROL_COMPLETE_OFFSET)
            .load(Ordering::SeqCst);
        if produced == rig.consumed {
            return events;
        }
        let head = rig.region().read(at(rig.consumed), 16).unwrap();
        let r#type = u32::from_le_bytes(head[0..4].try_into().unwrap());
        let size = u32::from_le_bytes(head[4..8].try_into().unwrap()) as u64;
        let seqno = u64::from_le_bytes(head[8..16].try_into().unwrap());
        rig.consumed += size;
        (rig.region())
            .atomic_u64(RGFX_CONTROL_COMPLETE_OFFSET + 0x40)
            .store(rig.consumed, Ordering::SeqCst);
        if r#type != RGFX_MSG_PAD {
            events.push((r#type, seqno));
        }
    }
}

/// Runs case `index` of the run seeded `seed`; a description of what went wrong, if anything did.
fn run_case(seed: u64, index: u64) -> Result<(), String> {
    let mut rng = Rng(seed ^ index.wrapping_mul(0xd1b5_4a32_d192_ed03));
    let sent = case(&mut rng);
    // The host takes 1 to 3 ms over about a third of the messages it records, and 2 to 6 ms over
    // two thirds of a block's, so the other kinds come while a block's lane is held.
    let mut slow = HashMap::new();
    for (seqno, message) in (1..).zip(&sent) {
        if matches!(message.lane(), Lane::Block(_)) {
            if rng.below(3) != 0 {
                slow.insert(seqno, 2 + rng.below(5));
            }
        } else if rng.below(3) == 0 {
            slow.insert(seqno, 1 + rng.below(3));
        }
    }
    let seen = Arc::new(Mutex::new(Seen::default()));
    let host = FuzzHost {
        seen: Arc::clone(&seen),
        slow: Arc::new(slow),
    };
    let mut rig = Rig::with_host(Box::new(host));
    rig.start();
    let mut events = Vec::new();
    for message in &sent {
        rig.send(message.bytes.clone());
        events.extend(events_now(&mut rig));
    }
    let recorded = sent.iter().filter(|m| m.recorded).count();
    let mut done = false;
    for _ in 0..5000 {
        events.extend(events_now(&mut rig));
        let ended = (seen.lock().marks.iter())
            .filter(|(_, began)| !began)
            .count();
        if ended == recorded && rig.device.shared.lanes.idle() {
            done = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    events.extend(events_now(&mut rig));
    let marks = seen.lock().marks.clone();
    let messages = || {
        let line = |(i, m): (usize, &Sent)| {
            format!(
                "{:3} type {:#x} context {} what {:#x} plane {} lane {:?}",
                i + 1,
                m.r#type,
                m.context,
                m.what,
                m.plane,
                m.lane()
            )
        };
        (sent.iter().enumerate().map(line))
            .collect::<Vec<_>>()
            .join("\n")
    };
    if !done {
        return Err(format!(
            "not every message was carried out: {marks:?}\n{}",
            messages()
        ));
    }
    // Where each message's carrying out began and ended, by seqno (seqno i + 1 is sent[i]).
    let (mut began, mut ended) = (HashMap::new(), HashMap::new());
    for (at, &(seqno, start)) in marks.iter().enumerate() {
        let into = if start { &mut began } else { &mut ended };
        if into.insert(seqno, at).is_some() {
            return Err(format!(
                "message {seqno} was carried out twice\n{}",
                messages()
            ));
        }
    }
    for (j, i) in order(&sent) {
        let (earlier, later) = (&sent[j], &sent[i]);
        if !earlier.recorded || !later.recorded {
            continue;
        }
        let (e, l) = (j as u64 + 1, i as u64 + 1);
        if ended[&e] > began[&l] {
            return Err(format!(
                "message {l} (type {:#x}) began before message {e} (type {:#x}) ended\n{}",
                later.r#type,
                earlier.r#type,
                messages()
            ));
        }
    }
    let mut answers: HashMap<u64, usize> = HashMap::new();
    for &(_, seqno) in &events {
        *answers.entry(seqno).or_default() += 1;
    }
    for (i, message) in sent.iter().enumerate() {
        let seqno = i as u64 + 1;
        let got = answers.get(&seqno).copied().unwrap_or(0);
        if got != usize::from(message.answered) {
            return Err(format!(
                "message {seqno} (type {:#x}) was answered {got} times\n{}",
                message.r#type,
                messages()
            ));
        }
    }
    Ok(())
}

/// Random interleavings of several processes' messages keep every order the guest relies on (the
/// module doc).
#[test]
fn lanes_keep_every_order_over_random_interleavings() {
    let seed = std::env::var("RGFX_LANES_FUZZ_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0x0005_eed1_a7e5_u64);
    let cases = std::env::var("RGFX_LANES_FUZZ_CASES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64u64);
    for index in 0..cases {
        if let Err(what) = run_case(seed, index) {
            panic!("seed {seed:#x} case {index}: {what}");
        }
    }
}
