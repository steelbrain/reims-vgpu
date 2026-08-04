//! Root/child FIFO drains, stamp writeback, and fail-visible command dispatch.
//!
//! Prefer structure correctness over full exec.c coverage: known root/child
//! control-plane ops update device state; unknown opcodes are recorded visibly.

use crate::contract::endian::{ld16, ld32, ld64, st16, st32};
use crate::model::*;
use crate::model::{DeviceState, ExecFault, FailEvent, PacketFault};
use crate::observe::Emit;
use crate::runtime::decode::fifo::{
    display_refresh_hz_1616, display_timing_entry_offset, encode_display_timing_entry,
    DisplayTimingEntry, DISPLAY_DESC_TIMING_STRIDE,
};
use crate::runtime::gpa_map;
use crate::runtime::heap_query::QueryError;
use crate::runtime::host::{HostAction, HostMemory, HostOps, MemError};
use crate::runtime::task_slot::{resolve_task_word, TaskWordSite};

/// apple-gfx `pending_frames >= 2`: hold further guest presents at FIFO head
/// until host paint consumes +0x188. Entry-side waitForPendingFrames — not
/// stamp-after-paint (that inverted PGDisplay completion and stacked tooltips).
pub const MAX_UNPAINTED_PRESENTS: u32 = 2;

/// Bit 0 in `translation_order_hold_mask` names the root FIFO. Child FIFOs use
/// their channel bit, matching `translation_deferred_mask`.
const TRANSLATION_ROOT_FIFO_BIT: u32 = 1;

fn note_translation_order_hold(state: &mut DeviceState, held_mask: u32) {
    let new_mask = held_mask & !state.translation_order_hold_mask;
    if new_mask == 0 {
        return;
    }
    let starts_episode = state.translation_order_hold_mask == 0;
    state.translation_order_hold_mask |= new_mask;
    if starts_episode {
        state.translation_order_holds = state.translation_order_holds.saturating_add(1);
    }
    // Census, not a failure: this is a resolver saying "not ready yet". The FIFO
    // is parked until the AIR module loads and `release_translation_order_holds`
    // takes the mask back down — and its release line was already `off`, so
    // logging the wait half as a failure made one control-flow pair straddle both
    // channels. Boot 87: 34 episodes started, 35 released, i.e. every one. A hold
    // that never releases is caught at `DeviceState::reset` instead, where the
    // guest's own teardown is the deadline and no age or depth has to be invented.
    crate::observe::off(format!(
        "translation_order_hold reason=air_loading held_mask={:#x} new_mask={new_mask:#x} producer_mask={:#x} count={}",
        state.translation_order_hold_mask,
        state.translation_deferred_mask,
        state.translation_order_holds
    ));
}

fn release_translation_order_holds(state: &mut DeviceState) {
    if state.translation_deferred_mask != 0 || state.translation_order_hold_mask == 0 {
        return;
    }
    let held_mask = std::mem::take(&mut state.translation_order_hold_mask);
    crate::observe::off(format!(
        "translation_order_release held_mask={held_mask:#x} producer_mask=0x0"
    ));
}

/// Install a `DefineTask2` payload and record the page-table identity it
/// resolved to.
///
/// The root ring and every child channel carry the same opcode with the same
/// payload layout. `site` names which ring so the two populations stay
/// separable on the census line; nothing else about them differs, and they
/// used to be two copies that could drift — the child arm reading a
/// four-byte length where the root read eight is exactly that having happened.
///
/// A short payload drops the task definition, and every later draw or resolve
/// on that task then fails downstream with no root cause. The child ring named
/// that; the root ring did not, and dropped silently. Both name it now.
/// True when a control packet is too short to carry the fields its opcode
/// needs, having said so.
///
/// Every arm that guards on payload length is acknowledged regardless of
/// whether it did anything: `drain_main_fifo` writes the root completion stamp
/// after the dispatch match, and `drain_child_fifo` calls `write_stamp` the
/// same way. So an arm that just skips on a short payload tells the guest its
/// command completed while nothing happened, and leaves the fail log empty —
/// the worst shape a loss can take, because the symptom surfaces arbitrarily
/// far downstream (a channel that never drains, an object list that never
/// binds) with no record of the cause.
///
/// `site` separates the two rings on the census line the way
/// [`apply_define_task2`] does, since the same opcode arrives on both.
fn packet_short(op: &'static str, channel: Option<u32>, have: usize, need: usize) -> bool {
    if have >= need {
        return false;
    }
    let site = match channel {
        Some(ch) => format!("ch{ch}"),
        None => "root".to_string(),
    };
    crate::observe::fail(format!(
        "packet_short reason={op}_short site={site} plen={have} need={need}"
    ));
    true
}

fn apply_define_task2<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    payload: &[u8],
    site: &str,
) {
    if payload.len() < DEFINE_TASK_LEN {
        crate::observe::fail(format!(
            "packet_short reason=define_task2_short site={site} plen={} need={DEFINE_TASK_LEN}",
            payload.len()
        ));
        return;
    }
    let raw_id = ld32(&payload[DEFINE_TASK_RAW_ID..]);
    let length = define_task_length(payload);
    let dir = ld32(&payload[DEFINE_TASK_DIRECTORY_PFN..]);
    // `raw_id` is `(task_id << 1) | is_kernel_task`: the guest's kernel-task and
    // user-task registrations differ only in that low bit, and the kernel task's
    // own id is 0, so `0x1` is the kernel task and not user task 1. Both halves
    // are decoded — the id to index the slot, the flag so the log says which
    // class registered rather than leaving the bit unaccounted for.
    let task_id = raw_id >> DEFINE_TASK_ID_SHIFT;
    let kernel_task = raw_id & 1 != 0;
    if !state.define_task(task_id, length, dir) || task_id as usize >= state.tasks.len() {
        return;
    }
    // Capture directory + root/depth so one boot shows the page-table identity.
    let slot = &state.tasks[task_id as usize];
    let walk =
        crate::runtime::gva_mem::diagnose_task_slot(host, slot, task_id, 0, state.page_shift);
    // `task=`/`dir=` are not repeated here: `walk` already carries the task id
    // as `tid=` and the directory as `dir=`, and a key printed twice in one line
    // is a field every log reader resolves arbitrarily.
    crate::observe::off(format!(
        "define_task {site} raw={raw_id:#x} kernel={} len={length:#x} page_shift={} {walk}",
        kernel_task as u8, state.page_shift
    ));
}

/// The task's address-space length from a `DefineTask2` payload.
///
/// The field is 64 bits wide, and the payload layout says so: it sits at
/// `DEFINE_TASK_LENGTH` (0x04) and the next field, `DEFINE_TASK_DIRECTORY_PFN`,
/// is at 0x0c — eight bytes later. Callers must have already checked
/// `payload.len() >= DEFINE_TASK_LEN` (16), which covers the whole field.
///
/// The root and child arms decode the same wire field, and read it the same
/// way here. The child arm used to take only the low 32 bits, which truncated
/// the length of any task spanning 4 GiB or more.
fn define_task_length(payload: &[u8]) -> u64 {
    ld64(&payload[DEFINE_TASK_LENGTH..])
}

/// Trailer size `submitTransaction` appends after serializing the transaction's
/// resource list: `[pipe][task][surface][gamma…]` for the gamma command,
/// `[pipe][surface][task]` otherwise.
fn display_txn_trailer_len(opcode: u16) -> usize {
    if opcode == CHILD_OP_PRESENT_GAMMA_X86 {
        0x24
    } else {
        0x0c
    }
}

/// Word slots of the surface id and the task field within the trailer, keyed on
/// the command that emitted it.
///
/// These are three different FIFO commands, not one shape with variations, so
/// nothing here may be assumed from one opcode to another:
///
/// - op6 `CmdDisplayTransaction3` — `[pipe][surface][task]`.
/// - op7, its gamma variant — `[pipe][task][surface][gamma…]`; the first two
///   words are swapped relative to op6.
/// - op8 `CmdDisplaySwapMapping` — `[display][_][mapping]`. This one names a
///   single mapping instead of serializing a transaction, so its surface word
///   is `DISPLAY_SWAP_MAPPING` (0x08), *not* op6's 0x04, and it has no known
///   task field at all. Reading it at op6's slot would return the unidentified
///   word between the display index and the mapping.
///
/// Slots are `regs.rs`'s `PRESENT_*` / `DISPLAY_SWAP_*` byte offsets divided by
/// four; this is the only place either reading of those offsets is spelled out,
/// so the present path and the payload census cannot decode the field
/// differently.
fn display_txn_trailer_slots(opcode: u16) -> (usize, Option<usize>) {
    match opcode {
        CHILD_OP_PRESENT_GAMMA_X86 => (PRESENT_GAMMA_X86_SURFACE_ID / 4, Some(1)),
        CHILD_OP_DISPLAY_SWAP => (DISPLAY_SWAP_MAPPING / 4, None),
        _ => (PRESENT_X86_SURFACE_ID / 4, Some(2)),
    }
}

/// The surface id a display-present payload names, read from offset zero.
///
/// The slot is the one `display_txn_trailer_slots` gives for the emitting
/// command, so this head-relative reading and the tail-relative reading the
/// census takes cannot drift apart.
///
/// `None` is a payload too short to hold the command's own trailer. That loses
/// the present outright, so callers must name it rather than presenting mapping
/// zero and completing the packet in silence.
fn present_surface_id(opcode: u16, payload: &[u8]) -> Option<u32> {
    let trailer = display_txn_trailer_len(opcode);
    (payload.len() >= trailer).then(|| {
        let off = display_txn_trailer_slots(opcode).0 * 4;
        ld32(&payload[off..])
    })
}

/// Alarm for a display-transaction payload longer than its command declares.
///
/// # The wire shape, and why there is nothing left to sample here
///
/// A display present is an `IOAccelDisplayPipeTransaction2` on the guest side —
/// a per-frame list of planes carrying source, destination and dirty rects — and
/// this device decodes only a single surface id from it. That reads like a
/// truncation, and a sampler used to sit here recording whether the rest of the
/// list rode inline in the payload.
///
/// It does not, and it never can: the guest's display pipe serializes the
/// transaction by taking **plane 0's** IOSurface and writing that one id into a
/// fixed-size command. There is no plane list on the wire, so decoding one
/// surface is the whole contract rather than a first approximation of it. The
/// command is 12 bytes for `CmdDisplayTransaction3` and 36 for its gamma
/// variant, which is what [`display_txn_trailer_len`] returns, and the field
/// order differs between them exactly as [`display_txn_trailer_slots`] says.
///
/// The same reading settles the third word. It is the id of the paravirt task
/// that owns the presented surface, taken from the display pipe's own resource
/// heap rather than from the transaction — so a zero there means the pipe has no
/// task bound yet, not that the field is unused.
///
/// One consequence worth stating because it is easy to go looking for: **the
/// guest's damage rects never reach this device.** They exist in the
/// transaction, and the serializer drops them. Any repair that wants per-frame
/// damage has to get it from somewhere other than the display path.
///
/// What survives is the one thing a static reading cannot promise for a guest
/// this device does not ship: that the payload keeps its declared size. A longer
/// one means this decode has become a truncation, so it is an always-on alarm
/// rather than a sample budget. Under-length is already refused by
/// [`present_surface_id`], which is where it costs a present.
///
/// # The plane-list reading is op6/op7's, and only op6/op7's
///
/// Everything above is about `submitTransaction`, which is what op6
/// `CmdDisplayTransaction3` and its gamma variant op7 emit. **op8
/// `CmdDisplaySwapMapping` does not serialize a transaction at all** — it names
/// a single mapping, as [`display_txn_trailer_slots`] says — so it has no plane
/// list to grow and "a plane list may have appeared" cannot be true of it.
///
/// That distinction is not hypothetical. op8 is the arm64 present path, and on
/// a driven arm64 boot this alarm fires on **every present**, 1 668 times in
/// 212 s, always as `op=0x8 plen=40 trailer=12`. The message it printed was an
/// explanation that structurally could not apply, on a first-class pathway's
/// normal traffic. What is true there is narrower and is what it says now: 28
/// bytes past the words this device knows, contents unnamed.
///
/// So the line carries the undecoded tail's bytes. It is latched per
/// `(opcode, length)` and therefore costs one line per shape per boot, and
/// without them a reader who sees this alarm has to rebuild and reboot before
/// learning anything at all — which is what closing the gap actually needs.
///
/// # What the arm64 guest actually puts there
///
/// Measured, and deliberately not interpreted. On two boots at 1920×1080 the
/// tail is byte-identical and there is exactly **one** `(opcode, length)` shape
/// in the whole boot — `op=0x8 plen=40` — so this is what every present on this
/// pathway carries:
///
/// ```text
/// +0x0c  00 00 00 00
/// +0x10  00 40 10 00
/// +0x14  00 00 00 00
/// +0x18  00 40 00 00
/// +0x1c  00 00 00 00
/// +0x20  01 01 00 00
/// +0x24  00 00 40 48
/// ```
///
/// No field here is named, and none should be until it has been made to move.
/// Read as little-endian `u32` the non-zero words are `0x00104000`, `0x00004000`,
/// `0x00000101` and `0x48400000`, and it is tempting to call the second one the
/// 16 KiB arm64 page size and the first a page-aligned length because
/// `0x104000 / 0x4000` is exactly 65. That is arithmetic agreeing with a guess,
/// which is not a derivation — every one of those words is also consistent with
/// a stride, an extent, a pair of `u16`s, or a `f32` (`0x48400000` is 196608.0).
///
/// **The experiment that would settle it is a display mode change.** Change the
/// guest's resolution, take the alarm's line again — the latch is keyed on
/// length, so a second mode at the same length needs the latch cleared or a
/// second boot — and diff the seven words. Whatever tracks width, height or
/// stride identifies itself immediately. Nothing short of that should turn
/// these into named fields.
fn note_display_txn_payload(state: &mut DeviceState, channel_id: u32, packet: &Packet) {
    let plen = packet.payload.len();
    let trailer = display_txn_trailer_len(packet.opcode);
    if plen <= trailer {
        return;
    }
    crate::runtime::drain::note_store_route("display_txn_payload_overlong");
    // Latched per (opcode, length): a guest that grew this command grew it for
    // every frame, and the thousandth line says nothing the first did not.
    if !state
        .display
        .txn_payload_samples
        .insert((packet.opcode, plen))
    {
        return;
    }
    // Bounded, because the length is the guest's. 64 bytes is four times the
    // largest trailer here and is enough to show the shape of whatever follows.
    const TAIL_DUMP_MAX: usize = 64;
    let tail = &packet.payload[trailer..];
    let shown = tail.len().min(TAIL_DUMP_MAX);
    let mut hex = String::with_capacity(shown * 3);
    for b in &tail[..shown] {
        hex.push_str(&format!("{b:02x}"));
    }
    let what = if packet.opcode == CHILD_OP_DISPLAY_SWAP {
        // No transaction is serialized here, so there is no plane list this
        // could be. Naming one would send the next reader looking for a
        // structure that cannot exist on this command.
        "this command names a single mapping and serializes no transaction, so these \
         bytes are not a plane list and nothing in this device names them"
    } else {
        "the command carries more than its declared trailer, so a plane list may have \
         appeared and decoding a single surface id would be dropping planes"
    };
    crate::observe::fail(format!(
        "display_txn_payload_overlong op={:#x} ch={channel_id} plen={plen} trailer={trailer} \
         undecoded={} tail=0x{hex}{} ({what})",
        packet.opcode,
        tail.len(),
        if shown < tail.len() { "..." } else { "" },
    ));
}

/// What the CPU-side capture can say about a present's content.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PresentContentVerdict {
    /// No CPU pixels exist for this present, so nothing can be claimed.
    Unsampled,
    /// Sampled, and every pixel's RGB is zero.
    Black,
    /// Sampled, and something is visible.
    Content,
}

/// Judge a present's captured frame.
///
/// An empty `frame_bgra` is **not** a black frame. When a dmabuf carries the
/// present (route B), `capture_present_frame` deliberately skips the full-frame
/// GPU→CPU readback and leaves the buffer empty, so a plain `max_rgb == 0` test
/// reports black on every such present — 1338 `present_black_retain` records
/// against 1312 presents on a live boot. That buries the always-on log under a
/// wolf-cry and hides the genuinely black frame the record exists to catch,
/// which is the opposite of what an always-on failure sink is for. With no
/// pixels there is no evidence either way, so the absence has its own verdict.
pub(crate) fn present_content_verdict(frame_bgra: &[u8], max_rgb: u8) -> PresentContentVerdict {
    if frame_bgra.is_empty() {
        PresentContentVerdict::Unsampled
    } else if max_rgb == 0 {
        PresentContentVerdict::Black
    } else {
        PresentContentVerdict::Content
    }
}

/// Parsed FIFO packet (main + child share framing).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Packet {
    pub opcode: u16,
    pub stamp_count: u16,
    pub total_size: u32,
    pub completion_stamp: u32,
    pub payload: Vec<u8>,
    pub next_head: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PacketError {
    ShortHeader,
    BadSize,
    Incomplete,
}

impl PacketError {
    /// The registered fault this error reports, or `None` when it is ring
    /// control flow rather than a refusal.
    ///
    /// This is I2's carve-out made mechanical instead of conventional. A partial
    /// packet in the ring is the *normal* state of a producer mid-write: the
    /// drain loop breaks and comes back. Logging it would flood the always-on
    /// sink on every healthy boot, which is why `ShortHeader` and `Incomplete`
    /// answer `None` here — and why a future variant cannot be added without its
    /// author deciding which side of that line it falls on. Both drain loops
    /// go through it, so that decision is made once rather than per ring.
    pub fn fault(self) -> Option<PacketFault> {
        match self {
            Self::ShortHeader | Self::Incomplete => None,
            Self::BadSize => Some(PacketFault::BadSize),
        }
    }
}

fn decode_packet(bytes: &[u8], head: u32, available: u32) -> Result<Packet, PacketError> {
    if available < PACKET_HEADER_LEN {
        return Err(PacketError::ShortHeader);
    }
    if bytes.len() < PACKET_HEADER_LEN as usize {
        return Err(PacketError::ShortHeader);
    }
    let opcode = ld16(&bytes[PACKET_OPCODE..]);
    let stamp_count = ld16(&bytes[PACKET_STAMP_COUNT..]);
    let total_size = ld32(&bytes[PACKET_TOTAL_SIZE..]);
    let completion_stamp = ld32(&bytes[PACKET_COMPLETION_STAMP..]);

    if total_size < PACKET_HEADER_LEN || total_size as usize > bytes.len() {
        return Err(PacketError::BadSize);
    }
    if available < total_size {
        return Err(PacketError::Incomplete);
    }
    let stamps_bytes = stamp_count as u32 * PACKET_STAMP_LEN;
    let min_payload_off = PACKET_HEADER_LEN + stamps_bytes;
    if total_size < min_payload_off {
        return Err(PacketError::BadSize);
    }
    let payload = bytes[min_payload_off as usize..total_size as usize].to_vec();
    Ok(Packet {
        opcode,
        stamp_count,
        total_size,
        completion_stamp,
        payload,
        next_head: head.wrapping_add(total_size),
    })
}

fn read_ring_bytes<M: HostMemory>(
    mem: &M,
    base_gpa: u64,
    ring_size: u32,
    absolute: u32,
    len: u32,
) -> Result<Vec<u8>, MemError> {
    let mut out = vec![0u8; len as usize];
    if ring_size == 0 || len == 0 {
        return Ok(out);
    }
    let mut copied = 0u32;
    while copied < len {
        let off = absolute.wrapping_add(copied) % ring_size;
        let chunk = (ring_size - off).min(len - copied);
        mem.read_gpa(
            base_gpa + off as u64,
            &mut out[copied as usize..(copied + chunk) as usize],
        )?;
        copied += chunk;
    }
    Ok(out)
}

/// Write stamp value to FIFO base page slot and set status bit.
pub fn write_stamp<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    stamp_index: u32,
    stamp_value: u32,
) {
    let index = stamp_slot_index(stamp_index);
    if state.gfx.fifo_base_page == 0 {
        return;
    }
    // Before the guest is told anything finished, everything this device still
    // owes guest RAM has to be in guest RAM. After this write the guest may free
    // the render targets and its allocator may hand those pages to anything, and
    // no later check can tell that memory apart from the target it used to be —
    // which is why the page-set guard passed on 810 of 810 landings and the heap
    // corruption continued. See `storage_flush::flush_all_windows_before_fence`,
    // which the root completion stamp in `drain_main_fifo` shares.
    crate::runtime::storage_flush::flush_all_windows_before_fence(state, host);
    let Some(off) = stamp_slot_offset(index, state.page_size()) else {
        return;
    };
    let gpa = state.pfn_gpa(state.gfx.fifo_base_page) + off;
    let page_size = state.page_size() as usize;
    if gpa_map::write_u32(host, gpa, stamp_value, page_size).is_ok() {
        // The guest's fence has moved. Everything it allocated for the work this
        // stamp completes may be freed from here on, so any deferred window
        // still holding bytes for guest RAM is now writing behind the guest's
        // back — see `GvaDeferredEntry::armed_stamp_seq`.
        state.completion_stamp_seq = state.completion_stamp_seq.wrapping_add(1);
        state
            .gfx
            .interrupt_status_gpu
            .fetch_or(1u32 << (index & 0x1f), std::sync::atomic::Ordering::AcqRel);
        host.enqueue(HostAction::irq_gfx());
    }
}

/// What the GPU behind this host can execute, for the device-info keys that
/// describe the GPU rather than the protocol.
///
/// The Metal backend serves an Apple GPU to an Apple guest, so the table's own
/// values already describe the executing device and there is nothing to reduce.
/// The Vulkan backend runs on anything from a discrete part to an iGPU at the
/// Vulkan floor, which is exactly the case a fixed table gets wrong.
fn device_info_limits() -> crate::model::DeviceInfoLimits {
    #[cfg(not(feature = "backend-vulkan"))]
    {
        crate::model::DeviceInfoLimits {
            max_sample_count: u32::MAX,
            d24_stencil8: true,
            max_threads_per_threadgroup: [u32::MAX; 3],
            max_threadgroup_memory_bytes: u32::MAX,
            native_fp16: true,
        }
    }
    #[cfg(feature = "backend-vulkan")]
    {
        crate::backend::vulkan::engine::device_info_limits()
    }
}

fn reply_device_info<H: HostMemory + HostOps>(
    host: &mut H,
    count: u32,
    reply_pfn: u32,
    page_shift: u32,
    version: u32,
) -> Result<(), MemError> {
    if reply_pfn == 0 {
        return Ok(());
    }
    let page_size = 1usize << page_shift;
    let gpa = pfn_to_gpa(reply_pfn, page_shift);
    // Contract: guest reply buffer is one page. Cap pairs so we never write past it.
    let max_pairs = (page_size / DEVICE_INFO_REPLY_PAIR_LEN) as u32;
    if max_pairs == 0 {
        crate::observe::fail(format!(
            "device_info fail reason=page_too_small page={page_size:#x}"
        ));
        return Err(MemError::BadArgs);
    }
    let limits = device_info_limits();
    let caps = crate::model::device_info_caps(&limits, version);
    // Printed on every reply, not only when something changed. A host that
    // already meets the table reduces nothing, and then silence would be
    // indistinguishable from the derivation never having run — which is exactly
    // the failure mode this line exists to rule out on a rig whose GPU exceeds
    // every entry. `derived` names every key the guest was told something other
    // than the table's value, whether the cause was the host GPU or the
    // negotiated version; `version` is printed so key 12's answer can be
    // checked against the rung without a second log line.
    let derived: Vec<String> = caps
        .iter()
        .zip(DEVICE_INFO_CAPS)
        .filter(|((_, served), (_, table))| served != table)
        .map(|((key, served), (_, table))| format!("key{key}={served}(was {table})"))
        .collect();
    crate::observe::off(format!(
        "device_info version={} dual_plane={} host_samples={} host_d24s8={} host_threads={}x{}x{} host_tg_mem={} host_fp16={} derived=[{}]",
        version,
        u8::from(crate::model::protocol_dual_plane_textures(version)),
        limits.max_sample_count,
        u8::from(limits.d24_stencil8),
        limits.max_threads_per_threadgroup[0],
        limits.max_threads_per_threadgroup[1],
        limits.max_threads_per_threadgroup[2],
        limits.max_threadgroup_memory_bytes,
        u8::from(limits.native_fp16),
        derived.join(" ")
    ));
    let n = (caps.len() as u32).min(count).min(max_pairs);
    // When guest asks for more than one page of pairs, still write at most a
    // page of caps + optional sentinel only if room remains.
    let write_sentinel = n < count && n.saturating_add(1) <= max_pairs;
    if count > max_pairs {
        crate::observe::fail(format!(
            "device_info cap reason=reply_page count={count} max_pairs={max_pairs} page={page_size:#x}"
        ));
    }
    for i in 0..n {
        let (key, value) = caps[i as usize];
        let mut pair = [0u8; DEVICE_INFO_REPLY_PAIR_LEN];
        st32(&mut pair[0..4], key);
        st32(&mut pair[4..8], value);
        gpa_map::write_bytes(
            host,
            gpa + (i as u64) * DEVICE_INFO_REPLY_PAIR_LEN as u64,
            &pair,
            page_size,
        )?;
    }
    if write_sentinel {
        let sentinel = [0u8; DEVICE_INFO_REPLY_PAIR_LEN];
        gpa_map::write_bytes(
            host,
            gpa + (n as u64) * DEVICE_INFO_REPLY_PAIR_LEN as u64,
            &sentinel,
            page_size,
        )?;
    }
    Ok(())
}

/// Wire keys of the `CmdGetComputeInfo` reply the guest reads back
/// (kb tahoe-x86 + texture-ref 29-06-26).
const COMPUTE_INFO_KEY_MAX_TOTAL_THREADS: u32 = 1;
const COMPUTE_INFO_KEY_THREAD_EXECUTION_WIDTH: u32 = 3;
const COMPUTE_INFO_KEY_STATIC_THREADGROUP_MEMORY: u32 = 4;

/// What this host answers for a compute pipeline's threadgroup limits.
///
/// The guest sizes its dispatches from these, so an over-promise is not
/// cosmetic: claim a `maxTotalThreadsPerThreadgroup` the host cannot run and
/// the threadgroup the guest builds from it is one the device rejects. They
/// used to be the fixed triple `(1, 1024), (3, 32), (4, 0)`, whose own comment
/// called itself conservative and deferred the real values to "once
/// metal2vulkan encode lands". That has landed, and both are device limits.
///
/// `staticThreadgroupMemoryLength` is a property of the *pipeline*, not the
/// device — the threadgroup memory the kernel declares — so no device limit
/// answers it and it stays 0 until pipeline reflection carries it.
fn compute_info_caps() -> [(u32, u32); 3] {
    // Apple GPUs report 1024 and 32 across every family the arm64 pathway
    // targets, and the Metal backend serves an Apple GPU to an Apple guest.
    #[cfg(not(feature = "backend-vulkan"))]
    let (max_total_threads, thread_execution_width) = (1024, 32);
    #[cfg(feature = "backend-vulkan")]
    let (max_total_threads, thread_execution_width) =
        crate::backend::vulkan::engine::compute_threadgroup_limits();
    [
        (COMPUTE_INFO_KEY_MAX_TOTAL_THREADS, max_total_threads),
        (
            COMPUTE_INFO_KEY_THREAD_EXECUTION_WIDTH,
            thread_execution_width,
        ),
        (COMPUTE_INFO_KEY_STATIC_THREADGROUP_MEMORY, 0),
    ]
}

/// Child `CmdGetComputeInfo` (0x3b): 24B payload
/// `[task_id@0][pipeline_ref@4][max_key@8][count@12][reply_gva@16]`.
/// Host writes key/value pairs at reply_gva before stamp (Apple host contract).
fn reply_compute_info<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    payload: &[u8],
) -> bool {
    if payload.len() < 24 {
        return false;
    }
    let raw_task = ld32(&payload[0..]);
    let pipeline_ref = ld32(&payload[4..]);
    let max_key = ld32(&payload[8..]);
    let count = ld32(&payload[12..]);
    let reply_gva = u64::from_le_bytes(payload[16..24].try_into().unwrap_or([0; 8]));
    if reply_gva == 0 || count == 0 {
        crate::observe::fail(format!(
            "get_compute_info empty task={raw_task} pipe={pipeline_ref} max_key={max_key} count={count} gva={reply_gva:#x}"
        ));
        return false;
    }
    // A live slot or nothing; `bad_task` now names the word the guest sent
    // rather than the halved id this used to have resolved to by then.
    let Some(task_id) = resolve_task_word(&state.tasks, TaskWordSite::ComputeInfo, raw_task) else {
        crate::observe::fail(format!(
            "get_compute_info bad_task task={raw_task} pipe={pipeline_ref}"
        ));
        return false;
    };
    let mut wrote = 0u32;
    for (key, value) in compute_info_caps() {
        if key > max_key {
            continue;
        }
        if wrote >= count {
            break;
        }
        let mut pair = [0u8; DEVICE_INFO_REPLY_PAIR_LEN];
        st32(&mut pair[0..4], key);
        st32(&mut pair[4..8], value);
        let off = (wrote as u64) * DEVICE_INFO_REPLY_PAIR_LEN as u64;
        if crate::runtime::gva_mem::write_task_gva_product(
            state,
            host,
            task_id,
            reply_gva + off,
            &pair,
        )
        .is_err()
        {
            crate::observe::fail(format!(
                "get_compute_info write_fail task={task_id} gva={reply_gva:#x} wrote={wrote}"
            ));
            return false;
        }
        wrote += 1;
    }
    if wrote < count {
        let sentinel = [0u8; DEVICE_INFO_REPLY_PAIR_LEN];
        let off = (wrote as u64) * DEVICE_INFO_REPLY_PAIR_LEN as u64;
        let _ = crate::runtime::gva_mem::write_task_gva_product(
            state,
            host,
            task_id,
            reply_gva + off,
            &sentinel,
        );
    }
    // Success census — the reply landed. Route to `off()` so it stays always-on
    // in the log but leaves the curated real-error view clean; the genuine
    // failures (`empty`/`bad_task`/`write_fail`/`short`) above stay `fail()`.
    crate::observe::off(format!(
        "get_compute_info ok task={task_id} pipe={pipeline_ref} max_key={max_key} count={count} wrote={wrote} gva={reply_gva:#x}"
    ));
    true
}

fn reply_heap_texture_size_and_align<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    payload: &[u8],
) -> bool {
    let request = match crate::runtime::heap_query::decode_request(payload) {
        Ok(request) => request,
        Err(error) => {
            Emit::decline("heap_texture_query", &error)
                .field("plen", payload.len())
                .fail();
            return false;
        }
    };
    // A live slot or nothing. `resolved_task` is gone with the arm that made the
    // two differ: the only slot this can act on is the one the guest named.
    let Some(task_id) = resolve_task_word(
        &state.tasks,
        TaskWordSite::HeapTextureQuery,
        request.task_id,
    ) else {
        Emit::decline("heap_texture_query", &QueryError::BadTask)
            .field("task", request.task_id)
            .field("gva", format!("{:#x}", request.reply_gva))
            .fail();
        return false;
    };
    let requirement = match crate::runtime::heap_query::query_size_and_align(&request.descriptor) {
        Ok(requirement) => requirement,
        Err(error) => {
            let desc = request.descriptor;
            Emit::decline("heap_texture_query", &error)
                .field("task", task_id)
                .field("type", desc.texture_type)
                .field("fmt", format!("{:#x}", desc.pixel_format))
                .field(
                    "dims",
                    format!("{}x{}x{}", desc.width, desc.height, desc.depth),
                )
                .field("mips", desc.mipmap_level_count)
                .field("samples", desc.sample_count)
                .field("array", desc.array_length)
                .field("usage", format!("{:#x}", desc.usage))
                .field("options", format!("{:#x}", desc.resource_options))
                .fail();
            return false;
        }
    };
    let reply = requirement.encode();
    if crate::runtime::gva_mem::write_task_gva_product(
        state,
        host,
        task_id,
        request.reply_gva,
        &reply,
    )
    .is_err()
    {
        crate::observe::fail(format!(
            "heap_texture_query fail reason=write_fail task={task_id} gva={:#x} reply_len={} size={:#x} align={:#x}",
            request.reply_gva,
            request.reply_len,
            requirement.size,
            requirement.align
        ));
        return false;
    }
    let desc = request.descriptor;
    crate::observe::off(format!(
        "heap_texture_query ok task={task_id} gva={:#x} type={} fmt={:#x} {}x{}x{} mips={} samples={} array={} usage={:#x} options={:#x} size={:#x} align={:#x}",
        request.reply_gva,
        desc.texture_type,
        desc.pixel_format,
        desc.width,
        desc.height,
        desc.depth,
        desc.mipmap_level_count,
        desc.sample_count,
        desc.array_length,
        desc.usage,
        desc.resource_options,
        requirement.size,
        requirement.align
    ));
    true
}

fn process_root_packet<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    packet: &Packet,
) {
    let op = packet.opcode;
    let effective = if op == ROOT_OP_WRAPPER {
        if packet.payload.len() >= 4 {
            ld32(&packet.payload[0..]) as u16
        } else {
            op
        }
    } else {
        op
    };

    match effective {
        ROOT_OP_DEVICE_INFO_TAHOE => {
            if !packet_short(
                "device_info_tahoe",
                None,
                packet.payload.len(),
                DEVICE_INFO_TAHOE_REPLY_PFN + 4,
            ) {
                let count = ld32(&packet.payload[DEVICE_INFO_TAHOE_COUNT..]);
                let pfn = ld32(&packet.payload[DEVICE_INFO_TAHOE_REPLY_PFN..]);
                let _ = reply_device_info(host, count, pfn, state.page_shift, state.gfx.version);
            }
        }
        ROOT_OP_DEVICE_INFO_MONTEREY => {
            if !packet_short(
                "device_info_monterey",
                None,
                packet.payload.len(),
                DEVICE_INFO_MONTEREY_REPLY_PFN + 4,
            ) {
                let count = ld32(&packet.payload[DEVICE_INFO_MONTEREY_COUNT..]);
                let pfn = ld32(&packet.payload[DEVICE_INFO_MONTEREY_REPLY_PFN..]);
                let _ = reply_device_info(host, count, pfn, state.page_shift, state.gfx.version);
            }
        }
        ROOT_OP_DEFINE_FIFO => {
            if !packet_short("define_fifo", None, packet.payload.len(), 4) {
                let ch = ld32(&packet.payload[0..]);
                if ch >= 1 && (ch as usize) < MAX_CHANNELS {
                    let bit = 1u32 << ch;
                    state.active_child_mask |= bit;
                    state.translation_deferred_mask &= !bit;
                    state.translation_order_hold_mask &= !bit;
                    state.present_translation_hold_mask &= !bit;
                    // Invalidate ring cache for this channel.
                    state.child_rings[ch as usize] = Default::default();
                }
            }
        }
        ROOT_OP_FREE_FIFO => {
            if !packet_short("free_fifo", None, packet.payload.len(), 4) {
                let ch = ld32(&packet.payload[0..]);
                if ch >= 1 && (ch as usize) < MAX_CHANNELS {
                    let bit = 1u32 << ch;
                    state.active_child_mask &= !bit;
                    state.pending.child_mask &= !bit;
                    state.translation_deferred_mask &= !bit;
                    state.translation_order_hold_mask &= !bit;
                    state.present_translation_hold_mask &= !bit;
                    state.child_rings[ch as usize] = Default::default();
                }
            }
        }
        ROOT_OP_DEFINE_TASK2 => apply_define_task2(state, host, &packet.payload, "root"),
        ROOT_OP_SET_OBJECT_LIST => {
            if !packet_short(
                "set_object_list",
                None,
                packet.payload.len(),
                SET_OBJECT_LIST_LEN,
            ) {
                let task_id = ld32(&packet.payload[SET_OBJECT_LIST_TASK_ID..]);
                let pfn = ld32(&packet.payload[SET_OBJECT_LIST_PFN..]);
                let count = ld32(&packet.payload[SET_OBJECT_LIST_COUNT..]);
                // A list with entries but no page is not a list: every resolve
                // on it computes an address out of the offset alone and the
                // read is refused. Carry the payload so the field the page
                // actually arrived in can be read off it.
                if pfn == 0 && count != 0 {
                    crate::observe::fail(format!(
                        "set_object_list_no_page task={task_id} count={count} payload={:02x?}",
                        &packet.payload[..packet.payload.len().min(32)]
                    ));
                }
                let _ = state.set_object_list(task_id, pfn, count);
            }
        }
        // PVG CmdDeleteTask (0x20). Live: top UnknownRootOpcode was op 32 total_size=16
        // (12-byte header + task_id u32). Guest reuses task ids — must clear.
        ROOT_OP_DELETE_TASK => {
            let task_id = if packet.payload.len() >= 4 {
                ld32(&packet.payload[0..])
            } else {
                0
            };
            let ok = state.delete_task(task_id);
            crate::observe::off(format!(
                "delete_task root task={task_id} ok={} plen={}",
                ok as u8,
                packet.payload.len()
            ));
        }
        _ => {
            state.record_fail(FailEvent::UnknownRootOpcode {
                opcode: effective,
                total_size: packet.total_size,
            });
        }
    }
}

/// Drain the main (root) FIFO while producer != consumer.
pub fn drain_main_fifo<H: HostMemory + HostOps>(state: &mut DeviceState, host: &mut H) {
    let ring_size = main_ring_data_size(state.gfx.fifo_length, state.gfx.fifo_start);
    if ring_size == 0 || state.gfx.fifo_base_page == 0 {
        state.pending.main_drain = false;
        return;
    }
    let base = state.pfn_gpa(state.gfx.fifo_base_page) + state.gfx.fifo_start as u64;
    let mut completed = false;

    while state
        .gfx
        .fifo_read
        .load(std::sync::atomic::Ordering::Acquire)
        != state.gfx.fifo_written
    {
        let Some(available) = published_byte_count(
            state
                .gfx
                .fifo_read
                .load(std::sync::atomic::Ordering::Acquire),
            state.gfx.fifo_written,
            ring_size,
        ) else {
            state.record_fail(FailEvent::MalformedRootPacket {
                fault: PacketFault::DesyncedHeadTail,
                head: state
                    .gfx
                    .fifo_read
                    .load(std::sync::atomic::Ordering::Acquire),
            });
            break;
        };
        if available < PACKET_HEADER_LEN {
            break;
        }
        // Snapshot up to min(available, ring_size) — header first then full packet.
        let header = match read_ring_bytes(
            host,
            base,
            ring_size,
            state
                .gfx
                .fifo_read
                .load(std::sync::atomic::Ordering::Acquire),
            PACKET_HEADER_LEN,
        ) {
            Ok(h) => h,
            Err(_) => {
                state.record_fail(FailEvent::MalformedRootPacket {
                    fault: PacketFault::RootHeaderRead,
                    head: state
                        .gfx
                        .fifo_read
                        .load(std::sync::atomic::Ordering::Acquire),
                });
                break;
            }
        };
        let total_size = ld32(&header[PACKET_TOTAL_SIZE..]);
        let snap_len = if total_size >= PACKET_HEADER_LEN
            && total_size <= ring_size
            && available >= total_size
        {
            total_size
        } else if available >= PACKET_HEADER_LEN {
            // incomplete or bad — try decode to classify
            PACKET_HEADER_LEN
        } else {
            break;
        };
        let snap = match read_ring_bytes(
            host,
            base,
            ring_size,
            state
                .gfx
                .fifo_read
                .load(std::sync::atomic::Ordering::Acquire),
            snap_len,
        ) {
            Ok(s) => s,
            Err(_) => {
                state.record_fail(FailEvent::MalformedRootPacket {
                    fault: PacketFault::RootSnapRead,
                    head: state
                        .gfx
                        .fifo_read
                        .load(std::sync::atomic::Ordering::Acquire),
                });
                break;
            }
        };
        match decode_packet(
            &snap,
            state
                .gfx
                .fifo_read
                .load(std::sync::atomic::Ordering::Acquire),
            available,
        ) {
            Ok(packet) => {
                process_root_packet(state, host, &packet);
                state
                    .gfx
                    .fifo_read
                    .store(packet.next_head, std::sync::atomic::Ordering::Release);
                // Root stamp = slot 0.
                if state.gfx.fifo_base_page != 0 {
                    if let Some(off) = stamp_slot_offset(0, state.page_size()) {
                        // The root stamp is a completion the guest waits on, so
                        // every deferred rail owes guest RAM its bytes here, not
                        // only at `write_stamp`'s child slots.
                        crate::runtime::storage_flush::flush_all_windows_before_fence(state, host);
                        let gpa = state.pfn_gpa(state.gfx.fifo_base_page) + off;
                        if gpa_map::write_u32(
                            host,
                            gpa,
                            packet.completion_stamp,
                            state.page_size() as usize,
                        )
                        .is_ok()
                        {
                            // A window armed after this point has outlived a fence
                            // the moment it is still armed at the next one. The
                            // counter is what `armed_stamp_seq` is compared
                            // against, so a rail that does not move it reads as
                            // punctual however long it actually waited.
                            state.completion_stamp_seq = state.completion_stamp_seq.wrapping_add(1);
                            completed = true;
                        } else {
                            // The guest waits on this root completion stamp; a
                            // silent writeback failure hangs it forever with no
                            // trace (drain.rs Rank-2 audit).
                            state.record_fail(FailEvent::MalformedRootPacket {
                                fault: PacketFault::RootStampWriteback,
                                head: state
                                    .gfx
                                    .fifo_read
                                    .load(std::sync::atomic::Ordering::Acquire),
                            });
                        }
                    }
                }
            }
            Err(err) => {
                // `fault()` decides which errors reach the log: a packet the
                // producer is still writing is ring control flow and answers
                // `None`. Either way the drain stops here and comes back.
                if let Some(fault) = err.fault() {
                    state.record_fail(FailEvent::MalformedRootPacket {
                        fault,
                        head: state
                            .gfx
                            .fifo_read
                            .load(std::sync::atomic::Ordering::Acquire),
                    });
                }
                break;
            }
        }
    }

    if completed {
        state
            .gfx
            .interrupt_status_gpu
            .fetch_or(1, std::sync::atomic::Ordering::AcqRel);
        host.enqueue(HostAction::irq_gfx());
    }
    state.pending.main_drain = false;
}

fn ensure_child_ring<M: HostMemory>(
    state: &mut DeviceState,
    mem: &M,
    channel_id: u32,
    base_pfn: u32,
) -> u32 {
    if channel_id == 0 || channel_id as usize >= MAX_CHANNELS || base_pfn == 0 {
        return 0;
    }
    let page_shift = state.page_shift;
    let page_size = state.page_size();
    let ring = &mut state.child_rings[channel_id as usize];
    if ring.valid && ring.base_pfn == base_pfn {
        return ring.length;
    }
    // Count leading non-zero PFNs in the page list (one page of u32 PFNs).
    let list_gpa = pfn_to_gpa(base_pfn, page_shift);
    let max_entries = (page_size / CHILD_RING_PFN_ENTRY_LEN) as u32;
    let mut page_gpas = Vec::new();
    for i in 0..max_entries {
        let mut b = [0u8; 4];
        if mem
            .read_gpa(list_gpa + i as u64 * CHILD_RING_PFN_ENTRY_LEN, &mut b)
            .is_err()
        {
            break;
        }
        let pfn = u32::from_le_bytes(b);
        if pfn == 0 {
            break;
        }
        page_gpas.push(pfn_to_gpa(pfn, page_shift));
    }
    let length = (page_gpas.len() as u32).saturating_mul(page_size as u32);
    *ring = crate::model::ChannelRing {
        valid: length != 0,
        base_pfn,
        length,
        page_gpas,
    };
    length
}

fn read_child_ring_bytes<M: HostMemory>(
    mem: &M,
    page_gpas: &[u64],
    ring_length: u32,
    absolute: u32,
    len: u32,
    page_shift: u32,
) -> Result<Vec<u8>, MemError> {
    let page_size = 1u64 << page_shift;
    let mut out = vec![0u8; len as usize];
    if ring_length == 0 || page_gpas.is_empty() {
        return Ok(out);
    }
    for i in 0..len {
        let off = absolute.wrapping_add(i) % ring_length;
        let page = (off as u64) >> page_shift;
        let page_off = (off as u64) & (page_size - 1);
        if page as usize >= page_gpas.len() {
            out[i as usize] = 0;
            continue;
        }
        let mut b = [0u8; 1];
        mem.read_gpa(page_gpas[page as usize] + page_off, &mut b)?;
        out[i as usize] = b[0];
    }
    Ok(out)
}

fn shared_w16<H: HostMemory + HostOps>(host: &mut H, gpa: u64, off: u64, v: u16, page_size: usize) {
    let mut b = [0u8; 2];
    st16(&mut b, v);
    let _ = gpa_map::write_bytes(host, gpa + off, &b, page_size);
}

fn shared_w32<H: HostMemory + HostOps>(host: &mut H, gpa: u64, off: u64, v: u32, page_size: usize) {
    let mut b = [0u8; 4];
    st32(&mut b, v);
    let _ = gpa_map::write_bytes(host, gpa + off, &b, page_size);
}

/// Fill the guest display descriptor page (archive `apple_pv_gpu_display_setup`).
///
///: `+0x208` is the timing-element **count**, not a
/// pixel width. Modes are 1920×1080, 1440×1080, 1280×1024 (apple-gfx A/B
/// reference geometry) plus 3840×2160 (4K UHD), each advertised at
/// `DISPLAY_REFRESH_HZ` (120 Hz), so the guest always latches the 120 Hz mode.
/// Element 0 (1920×1080) stays the native/preferred format (+0x210/+0x212 double
/// as NativeFormat*Pixels), so boot resolution is unchanged and 4K is an
/// additional selectable mode; the dynamic scanout/present/host-window geometry
/// follows the surface the guest actually presents at the selected mode.
fn fill_display_descriptor<H: HostMemory + HostOps>(
    host: &mut H,
    gpa: u64,
    index: u32,
    page_size: u64,
) {
    if gpa == 0 {
        return;
    }
    let Some(refresh) = display_refresh_hz_1616(DISPLAY_REFRESH_HZ) else {
        return;
    };
    let psz = page_size as usize;

    shared_w32(host, gpa, DISPLAY_DESC_SERIAL, DISPLAY_SERIAL_NUMBER, psz);
    let _ = gpa_map::write_bytes(
        host,
        gpa + DISPLAY_DESC_PRODUCT_NAME,
        DISPLAY_PRODUCT_NAME,
        psz,
    );
    shared_w16(host, gpa, DISPLAY_DESC_INDEX, index as u16, psz);
    shared_w16(host, gpa, DISPLAY_DESC_WIDTH_MM, DISPLAY_WIDTH_MM, psz);
    shared_w16(host, gpa, DISPLAY_DESC_HEIGHT_MM, DISPLAY_HEIGHT_MM, psz);
    shared_w32(host, gpa, DISPLAY_DESC_FEATURES, 0, psz);

    // HW cursor capability so the guest doorbells glyph/show/move.
    let max_wh = (CURSOR_MAX_DIM & 0xffff) | ((CURSOR_MAX_DIM & 0xffff) << 16);
    shared_w32(host, gpa, DISPLAY_SHARED_CURSOR_MAX_WH, max_wh, psz);
    shared_w32(
        host,
        gpa,
        DISPLAY_SHARED_CURSOR_FEATURES,
        DISPLAY_CURSOR_FEATURE_HW,
        psz,
    );

    const MODES: &[(u16, u16)] = &[
        (DISPLAY_MODE_EFI_W, DISPLAY_MODE_EFI_H),
        (DISPLAY_MODE1_W, DISPLAY_MODE1_H),
        (DISPLAY_MODE2_W, DISPLAY_MODE2_H),
        (DISPLAY_MODE3_W, DISPLAY_MODE3_H),
    ];
    shared_w16(
        host,
        gpa,
        DISPLAY_DESC_TIMING_COUNT,
        MODES.len() as u16,
        psz,
    );

    let mut encoded = [0u8; DISPLAY_DESC_TIMING_STRIDE as usize];
    for (i, &(width, height)) in MODES.iter().enumerate() {
        let Some(off) = display_timing_entry_offset(i as u32, page_size) else {
            return;
        };
        let entry = DisplayTimingEntry {
            width,
            height,
            refresh_1616: refresh,
            tail0: 0,
            tail1: 0,
        };
        encoded.fill(0);
        if !encode_display_timing_entry(&entry, &mut encoded) {
            return;
        }
        let _ = gpa_map::write_bytes(host, gpa + off, &encoded, psz);
    }
}

/// Sample cursor x/y/show from the display shared-state page (GPA +0xe00).
fn sample_cursor_position<M: HostMemory>(state: &mut DeviceState, mem: &M) {
    if state.display.shared_gpa == 0 {
        return;
    }
    let mut pos = [0u8; 4];
    if mem
        .read_gpa(
            state.display.shared_gpa + DISPLAY_SHARED_CURSOR_POS,
            &mut pos,
        )
        .is_err()
    {
        return;
    }
    let packed = ld32(&pos);
    if packed == 0xffff_ffff {
        state.cursor.show = false;
        return;
    }
    state.cursor.x = (packed & 0xffff) as u16;
    state.cursor.y = ((packed >> 16) & 0xffff) as u16;
    let mut show = [0u8; 4];
    if mem
        .read_gpa(
            state.display.shared_gpa + DISPLAY_SHARED_CURSOR_SHOW,
            &mut show,
        )
        .is_ok()
    {
        // Guest may only write a byte; treat non-zero low byte as show.
        state.cursor.show = show[0] != 0 || ld32(&show) != 0;
    }
}

/// Load CmdDisplayCursorGlyph pixels (BGRA guest → ARGB QEMUCursor).
/// Fail-visible, once per reason per boot, for the silent `load_cursor_glyph`
/// drop sites: a malformed cursor-glyph packet leaves the cursor stale/wrong
/// with no log. Cursor glyphs are infrequent (sent when the pointer *image*
/// changes, not per move) but a persistently-bad glyph could repeat, so latch
/// each reason once. Always returns `false` so callers stay `return cg_fail(..)`.
fn cursor_glyph_fail(reason: &'static str, detail: String) -> bool {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<&'static str>>> = Mutex::new(None);
    let mut guard = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if guard.get_or_insert_with(HashSet::new).insert(reason) {
        crate::observe::fail(detail);
    }
    false
}

fn load_cursor_glyph<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &H,
    packet: &Packet,
) -> bool {
    if packet.payload.len() < CURSOR_GLYPH_PAYLOAD_LEN {
        return cursor_glyph_fail(
            "cursor_glyph_short",
            format!(
                "cursor_glyph_fail reason=cursor_glyph_short plen={} need={CURSOR_GLYPH_PAYLOAD_LEN}",
                packet.payload.len()
            ),
        );
    }
    let task_id = ld32(&packet.payload[0x04..]);
    let virtual_offset = u64::from_le_bytes(packet.payload[0x08..0x10].try_into().unwrap());
    let mapped_length = u64::from_le_bytes(packet.payload[0x10..0x18].try_into().unwrap());
    let stride = u64::from_le_bytes(packet.payload[0x18..0x20].try_into().unwrap()) as u32;
    let width = ld16(&packet.payload[0x20..]) as u32;
    let height = ld16(&packet.payload[0x22..]) as u32;
    let hot_x = ld16(&packet.payload[0x24..]) as u32;
    let hot_y = ld16(&packet.payload[0x26..]) as u32;

    if width == 0
        || height == 0
        || width > CURSOR_MAX_DIM
        || height > CURSOR_MAX_DIM
        || stride < width.saturating_mul(CURSOR_GLYPH_BPP)
        || hot_x >= width
        || hot_y >= height
    {
        return cursor_glyph_fail(
            "cursor_glyph_geom",
            format!("cursor_glyph_fail reason=cursor_glyph_geom {width}x{height} stride={stride} hot=({hot_x},{hot_y}) max={CURSOR_MAX_DIM}"),
        );
    }
    let need = (height as u64 - 1)
        .saturating_mul(stride as u64)
        .saturating_add(width as u64 * CURSOR_GLYPH_BPP as u64);
    if mapped_length < need {
        return cursor_glyph_fail(
            "cursor_glyph_mapped_len",
            format!("cursor_glyph_fail reason=cursor_glyph_mapped_len mapped_length={mapped_length} need={need} {width}x{height}"),
        );
    }
    let Some(need_host) = crate::runtime::metal_draw::host_alloc_len(need) else {
        return cursor_glyph_fail(
            "cursor_glyph_alloc",
            format!("cursor_glyph_fail reason=cursor_glyph_alloc need={need}"),
        );
    };

    let mut src = vec![0u8; need_host];
    if crate::runtime::gva_mem::read_task_gva_by_id(
        host,
        &state.tasks,
        task_id,
        virtual_offset,
        &mut src,
        state.page_shift,
    )
    .is_err()
    {
        return cursor_glyph_fail(
            "cursor_glyph_read",
            format!("cursor_glyph_fail reason=cursor_glyph_read task={task_id} voff={virtual_offset:#x} need_host={need_host}"),
        );
    }

    let mut pixels = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        let row = (y as usize).saturating_mul(stride as usize);
        for x in 0..width {
            let px = row + (x as usize) * CURSOR_GLYPH_BPP as usize;
            if px + 4 > src.len() {
                return cursor_glyph_fail(
                    "cursor_glyph_bounds",
                    format!("cursor_glyph_fail reason=cursor_glyph_bounds px={px} src_len={} {width}x{height} stride={stride}", src.len()),
                );
            }
            let b = src[px];
            let g = src[px + 1];
            let r = src[px + 2];
            let a = src[px + 3];
            // QEMUCursor 0xAARRGGBB
            pixels.push(((a as u32) << 24) | ((r as u32) << 16) | ((g as u32) << 8) | (b as u32));
        }
    }

    state.cursor.width = width as u16;
    state.cursor.height = height as u16;
    state.cursor.hot_x = hot_x as u16;
    state.cursor.hot_y = hot_y as u16;
    state.cursor.pixels = pixels;
    state.cursor.glyph_ready = true;
    sample_cursor_position(state, host);
    true
}

/// Account one accepted present and request a worker→host action boundary.
///
/// Yielding here bounds how far the drain runs ahead of the display consumer.
/// Continuing to consume guest work can fill `pending_frames`, then hold
/// Display0 forever while its frame remains unconsumed.
fn enqueue_present_scanout<H: HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    width: u32,
    height: u32,
) {
    // Two presentation paths, selected by the live window link:
    //
    // - Window active (x86 default): no CPU `ScanoutUpdate`. QEMU runs
    //   `-display none`, no DisplayChangeListener ticks `gfx_update`, and the
    //   surface would be painted for nobody. The window is fed by
    //   `publish_window_frame` from this drain, and the present-completion ack
    //   is re-homed onto the drain tail (see `device_drain`).
    //
    // - No window (arm64 MMIO `-display cocoa`, or `REIMS_VGPU_WINDOW=0`): the QEMU
    //   console IS the display, so every present enqueues the CPU
    //   `ScanoutUpdate` (coalesced latest-wins in the action queue) and the
    //   ack comes from the console paint (`device_scanout_copy`), releasing
    //   `unpainted_presents` + `present_action_pending` there. Skipping the
    //   action here freezes the console at the last pre-boundary early-FB
    //   paint while the guest keeps presenting (live class: arm64 boot
    //   serial-20260723-221445, console stuck on the 15% progress bar while
    //   gen 38 presented the login wallpaper).
    if !state.present.window_active {
        host.enqueue(HostAction::scanout_gen(
            state.present.frame_mapping,
            width,
            height,
            state.present.frame_generation,
        ));
    }
    state.present.unpainted_presents = state.present.unpainted_presents.saturating_add(1);
    state.pending.host_action_yield = true;
}

fn present_page_identity_line(state: &DeviceState, mapping: u32, w: u32, h: u32) -> Option<String> {
    use std::collections::HashSet;
    let named = state.mappings.get(&mapping)?;
    let named_pfns: HashSet<u32> = named
        .page_entries
        .iter()
        .filter(|&&e| e & crate::contract::iosurface_pages::PAGE_ENTRY_VALID != 0)
        .map(|&e| e >> crate::contract::iosurface_pages::PAGE_ENTRY_PFN_SHIFT)
        .collect();
    let mut peers = String::new();
    for (&mid, m) in state.mappings.iter() {
        if mid == mapping
            || !m.has_geom
            || m.width != w
            || m.height != h
            || m.page_entries.is_empty()
        {
            continue;
        }
        let identical = m.page_entries == named.page_entries;
        let overlap = if identical {
            named_pfns.len()
        } else {
            m.page_entries
                .iter()
                .filter(|&&e| e & crate::contract::iosurface_pages::PAGE_ENTRY_VALID != 0)
                .filter(|&&e| {
                    named_pfns
                        .contains(&(e >> crate::contract::iosurface_pages::PAGE_ENTRY_PFN_SHIFT))
                })
                .count()
        };
        if !peers.is_empty() {
            peers.push(',');
        }
        peers.push_str(&format!(
            "mid{mid}:pages={}:overlap={overlap}:ident={}:kind={:?}",
            m.page_entries.len(),
            identical as u8,
            state.surface_write_kind(mid)
        ));
    }
    Some(format!(
        "present_page_identity mid={mapping} {w}x{h} pages={} valid={} map_gen={} kind={:?} peers=[{peers}]",
        named.page_entries.len(),
        named_pfns.len(),
        named.map_generation,
        state.surface_write_kind(mapping)
    ))
}

/// Which of the two present routes a present took, once per distinct route per
/// process.
///
/// Every present captures the surface the transaction names. This line splits
/// them on the named surface's write history anyway: `route=clear_only` is a
/// present whose named mid's most recent write was a `display_clear`/CLEAR
/// Store rather than a draw — the guest asking us to show a surface it has only
/// ever cleared. `route=named` is everything else. The split is the standing
/// measurement of whether that case occurs at all on a given rail; two lines per
/// process at most, which is what makes it safe to leave on.
///
/// **Measured: only `route=named write_kind=Composite`, on 104 x86/Vulkan boots
/// — every boot in the failure log since this line landed.** Not one
/// `route=clear_only`, including a 1766 s session driven through the
/// heavy-Safari residue repro. The dedup is per process, so one line per boot is
/// the whole reading for that boot.
///
/// That is an x86 statement only. `note_surface_clear` marks a mid ClearOnly
/// from a decoded `display_clear`/CLEAR Store, which is not rail-specific — what
/// the measurement shows is that on x86 the guest never *presents* a mid whose
/// most recent write was a Clear. An arm64 reading of this same line is what
/// would say whether that holds everywhere.
fn note_present_route(write_kind: crate::model::SurfaceWriteKind, is_clear_only: bool) {
    use std::sync::Mutex;
    static SEEN: Mutex<Option<std::collections::BTreeSet<bool>>> = Mutex::new(None);
    {
        let mut guard = SEEN.lock().unwrap_or_else(|p| p.into_inner());
        if !guard
            .get_or_insert_with(Default::default)
            .insert(is_clear_only)
        {
            return;
        }
    }
    crate::observe::fail(format!(
        "present_route route={} write_kind={write_kind:?}",
        if is_clear_only { "clear_only" } else { "named" },
    ));
}

fn log_present_page_identity(state: &DeviceState, mapping: u32, w: u32, h: u32) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<(u32, u32)>>> = Mutex::new(None);
    let Some(named) = state.mappings.get(&mapping) else {
        return;
    };
    let key = (mapping, named.map_generation);
    {
        let mut guard = SEEN.lock().unwrap_or_else(|p| p.into_inner());
        let seen = guard.get_or_insert_with(HashSet::new);
        if seen.len() > 1024 {
            seen.clear();
        }
        if !seen.insert(key) {
            return;
        }
    }
    if let Some(line) = present_page_identity_line(state, mapping, w, h) {
        crate::observe::fail(line);
    }
}

/// How long a display packet may be held for an earlier channel's shader
/// translation before the hold is worth less than what it costs.
///
/// The ordering itself is right: publishing a present's retain ahead of the
/// render packet it should follow shows the guest a frame that has not been
/// drawn yet. But the guest watchdogs the ring it wrote, and a stall it blames
/// on the device is a `GPU Reset` — measured on this rail as
/// `Name Display0 written: 17360 read: 17320`, one 40-byte packet short, with
/// `iconservicesagent`, `com.apple.dock.extra` and `Spotlight` aborting around
/// it as their in-flight work was discarded. A frame out of order costs one
/// frame; the reset costs every frame in flight and the processes waiting on
/// them.
///
/// The budget is wall-clock rather than a poll count because what has to fit
/// inside it is a translation: the compositor's uber shader takes seconds to
/// structure on first sight, and a count of drains says nothing about that.
/// Set well under a watchdog interval so the ring keeps moving while the
/// translation finishes in the background.
const PRESENT_ORDER_HOLD_BUDGET: std::time::Duration = std::time::Duration::from_millis(250);

/// Has the current display-order hold outlived [`PRESENT_ORDER_HOLD_BUDGET`]?
///
/// Starts the clock on the first call of a hold. Says so once when the budget
/// goes, because from then on presents pass in an order the guest did not ask
/// for and that has to be visible rather than inferred from a frame that looks
/// stale.
fn present_order_hold_budget_spent(state: &mut DeviceState) -> bool {
    let now = std::time::Instant::now();
    let since = *state.present_translation_hold_since.get_or_insert(now);
    if now.duration_since(since) < PRESENT_ORDER_HOLD_BUDGET {
        return false;
    }
    if state.present_translation_hold_mask != 0 {
        crate::observe::fail(format!(
            "present_order_hold_expired reason=translation_still_pending              held_ms={} pending_mask={:#x} hold_mask={:#x}              (the ring keeps moving; this present publishes ahead of the render              packet it should have followed)",
            now.duration_since(since).as_millis(),
            state.translation_deferred_mask,
            state.present_translation_hold_mask
        ));
        state.present_translation_hold_mask = 0;
    }
    true
}
/// Present a named mapping to the host console (DisplaySwap / x86 present op6/7).
fn present_named_mapping<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    channel_id: u32,
    mapping: u32,
) -> ChildPacketDisposition {
    if mapping == 0 {
        return ChildPacketDisposition::Complete;
    }
    // Archive apple_pv_gpu_display_swap:
    //   render_wait_surface(s, false, swap->mapping_id);
    //   scanout_present_boundary(...);
    //
    // Plus archive poll_tick Dekker rescue (apple_pv_gpu_poll_tick):
    // guest may publish child work without a doorbell while a drain
    // was in flight. Product has no separate host timer during the
    // DisplaySwap packet; drain **other** child FIFOs (skip mid-
    // packet channel) before and after wait_surface so body-layer
    // draws that land during the wait are frozen into the retain.
    // Never re-enter skip/draining_mask channels (boot wedge).
    // Not gen-stable multi-round; not surface_inflight invent.
    let skip = if state.draining_channel != 0 {
        state.draining_channel
    } else {
        channel_id
    };
    drain_other_child_fifos(state, host, skip);
    drain_other_child_fifos(state, host, skip);
    // Main-ring Dekker only (not full drain_stranded): guest may
    // publish root control work while child drains ran. Full
    // drain_stranded re-enters this child channel and wedged iBoot
    // (6720ce170). Main drain never re-enters a child mid-packet.
    if state.gfx.control_fifo != 0
        && state
            .gfx
            .fifo_read
            .load(std::sync::atomic::Ordering::Acquire)
            != state.gfx.fifo_written
    {
        drain_main_fifo(state, host);
        // Body-layer child work may be doorbell'd from main packets.
        drain_other_child_fifos(state, host, skip);
    }

    // Preflight translation keeps an EXEC packet at its channel head. If one
    // is still held after all rescue drains, accepting this display packet
    // would publish the prior +0x188 retain before the earlier render packet
    // executes. Leave the display head and stamp untouched. Poll-tick re-drives
    // all active channels; once translation is ready, the EXEC runs and this
    // packet is retried in order without blocking a vCPU or the QEMU main loop.
    let current_bit = 1u32.checked_shl(channel_id).unwrap_or(0);
    let deferred_other = state.translation_deferred_mask & !current_bit;
    if deferred_other != 0 && !present_order_hold_budget_spent(state) {
        if state.present_translation_hold_mask & current_bit == 0 {
            state.present_translation_holds = state.present_translation_holds.saturating_add(1);
            state.present_translation_hold_mask |= current_bit;
            crate::observe::fail(format!(
                "present_order_hold reason=translation_deferred ch={channel_id} mid={mapping} pending_mask={deferred_other:#x} frame_mapping={} early_front={} count={}",
                state.present.frame_mapping,
                state.present.early_front_mapping,
                state.present_translation_holds
            ));
        }
        return ChildPacketDisposition::Deferred;
    }
    if state.present_translation_hold_mask & current_bit != 0 {
        state.present_translation_hold_mask &= !current_bit;
        if state.present_translation_hold_mask == 0 {
            state.present_translation_hold_since = None;
        }
        crate::observe::off(format!(
            "present_order_release ch={channel_id} mid={mapping} pending_mask={:#x}",
            state.translation_deferred_mask
        ));
    }

    state.present.present_mapping = mapping;
    state.present.host_mapping = mapping;
    state.present.valid = true;
    // x86: present surface_id → type-4 object-list slot (heap index =
    // IOSurface getSurfaceID). Arm: MappingInternal page-table resolve.
    // Always attempt type-4 when pages empty; then iosfc/mapper path.
    let _ = crate::runtime::objects::ensure_surface_for_present(state, host, mapping);
    let force = state
        .mappings
        .get(&mapping)
        .map(|m| m.mapping_internal != 0)
        .unwrap_or(false);
    if force {
        // Same reason as the mapper-request path: a resolve that moves pages
        // strands any deferred window armed against the old ones.
        let _ = crate::runtime::storage_flush::flush_mapping_for_guest_read(state, host, mapping);
        let _ = crate::runtime::mapper::resolve_mapping_backing(state, host, mapping);
    }
    // Paint only from the presented surface's own geom — never the
    // previous console size fallback (that freezes mode switches).
    // Re-read gen after wait_surface (writebacks may have landed).
    let paint = state.mappings.get(&mapping).and_then(|m| {
        if m.has_geom && m.width > 0 && m.height > 0 {
            Some((m.width, m.height, m.content_generation))
        } else {
            None
        }
    });
    if let Some((w, h, gen)) = paint {
        state.present.width = w;
        state.present.height = h;
        state.present.generation = gen;
        log_present_page_identity(state, mapping, w, h);
        // Every present takes one route: capture the surface the transaction
        // named. A ClearOnly present — one whose named mid's most recent write
        // was a `display_clear`/CLEAR Store rather than a draw — used to take a
        // six-way resolver instead, choosing some *other* same-geometry surface
        // on the theory that the named one held nothing. `note_present_route`
        // still names which route each present takes, and its reading is why
        // there is only one left.
        let write_kind = state.surface_write_kind(mapping);
        let is_clear_only = matches!(write_kind, crate::model::SurfaceWriteKind::ClearOnly);
        note_present_route(write_kind, is_clear_only);

        // presentFrame names the front surface (leave-BAR1 boundary) once we
        // have a non-init present. Geom/capture may still fail after this.
        state.present.frame_flush_seen = true;
        // PGDisplay presentFrame **retains** the named surface into
        // +0x188 at present time; encodeCurrentFrame later re-shows
        // that retained surface (hostPresentCount). Guest may recycle
        // the mapping as soon as this packet's stamp completes — so
        // freeze guest pages **now**, after wait_surface drains, not
        // at BH after the stamp (that freezes mid-recycle partials:
        // toolbar-only dual-mid under app load). Mid-writeback Stores
        // must not recapture here — present boundary only.
        //
        // Always-on backing gate: a member presented twice with no full-frame
        // Store naming it in between is being displayed with content the guest
        // never sent for it. That is a real loss of guest work and belongs in
        // the log; nothing here papers over it.
        //
        // The line says "naming this mid" rather than "received", because that
        // is the whole of what `note_present_backing` read: decoded Store
        // bookkeeping, never the resident.
        //
        // WHICH IS WHY IT ALSO HAS TO READ THE CARRIER. The gate's witness is
        // `dense_frame_seq`, advanced only by `publish_surface_store` — i.e. when
        // a Store's pixels reached the mapping's GUEST PAGES. The resident rail
        // renders into the registry and skips that write, so "no full frame was
        // published for this mid" no longer implies "nothing can show one". A
        // 524 s boot measured four `reason=…never_stored` lines, each claiming
        // the surface was uninitialized and therefore black, against exactly one
        // `host_window_slate*` line in the whole run — a `covered=1` boot run at
        // t=22 s — with `presents == offered` and `direct_frac=1.00` in every
        // cadence window bracketing all four. A resident carried every one of
        // them. The message asserted a visual consequence the check cannot see,
        // which is "a reason the caller writes is not a reading" applied to an
        // outcome instead of a cause.
        //
        // So ask the presenter's own question, through the rule it shares
        // (`pools::slot_presentable`), and split on the answer the same way
        // `host_window_slate` / `host_window_slate_end` already split: a present
        // nothing can carry is a black frame and belongs on the failure channel;
        // one a resident carries cost no guest work and is a census. Reporting
        // both as black cries wolf every boot and — worse — leaves the real case
        // indistinguishable from the benign one, which is how a genuine
        // black-screen boot once produced zero lines here.
        //
        // Priced where it runs: one registry lookup under the engine lock, inside
        // the arm, so only on a present the structural gate has already refused —
        // four times in that boot, not 60 times a second.
        if let Some(backing) = state.note_present_backing(mapping) {
            let carried = present_resident_carries(state, mapping, w, h);
            let emit = crate::observe::Emit::decline("present_unbacked", &backing)
                .field("mid", mapping)
                .field("geom", format!("{w}x{h}"))
                .field("gen", gen)
                .field("carried", carrier_word(carried));
            if unbacked_present_is_a_loss(carried) {
                emit.fail();
            } else {
                emit.off();
            }
        }
        // The transaction payload carries exactly one field: plane 0's surface
        // id. So the capture source is the surface the guest named, and no
        // comparison between our own full-frame sequences may override it.
        // Presenting a "denser" same-geometry peer instead shows a buffer one
        // rotation step behind the one the guest asked for — residue when a
        // window closed in between, a stale region when one moved, thrash as
        // the choice oscillates.
        let encoded = crate::runtime::scanout::capture_present_frame(state, mapping, w, h, gen);
        if !encoded {
            // Retry encode at first host paint. Do **not** clear
            // frame_valid: PGDisplay keeps the prior presentFrame
            // (+0x188) for hostPresentCount until a new capture
            // succeeds. Invalidating the retain forced a black /
            // empty console when dual-mid page resolve raced.
            state.present.frame_encode_pending = true;
            let (pages, mapped, fmt) = state
                .mappings
                .get(&mapping)
                .map(|m| (m.page_entries.len(), m.mapped as u8, m.format))
                .unwrap_or((0, 0, 0));
            crate::observe::fail(format!(
                "present capture fail mid={mapping} {w}x{h} gen={gen} \
                 keep_prior={} pages={pages} mapped={mapped} fmt={fmt:#x}",
                state.present.frame_valid as u8
            ));
        } else {
            // One pass. `bgra_rgb_stats` already maxes the same
            // `px[0].max(px[1]).max(px[2])` per pixel, so a separate scan for
            // `max_rgb` was a second full 8 MiB walk of the frame, under the
            // device lock, for a value this call already returns.
            let (rgb_nz, max_rgb, px0) = crate::observe::bgra_rgb_stats(&state.present.frame_bgra);
            let verdict = present_content_verdict(&state.present.frame_bgra, max_rgb);
            if verdict == PresentContentVerdict::Unsampled {
                // Not a decline: the dmabuf rail carried the frame, so there are
                // no CPU pixels to judge and no guest work was lost.
                // `present_black` below is the alarm. On that rail this is the
                // normal outcome of every present.
                crate::observe::line(format!(
                    "present_content_unsampled mid={mapping} {w}x{h} gen={gen} \
                     (dmabuf carried the frame; no CPU pixels to judge)"
                ));
            } else if verdict == PresentContentVerdict::Black {
                // Both lines name the mapping the guest asked us to show and say
                // it came out black. They deliberately do not go looking for a
                // different surface that looks better: "which other host surface
                // has real content" is a judgement about observed pixels, and
                // `scanout::present_capture` already removed the same walk —
                // same undefended non-zero-pixel threshold — for that reason. A
                // black present is a decode or a writeback fault, and the mid,
                // geometry and generation here are what locate it.
                crate::observe::off(format!(
                    "present_black mid={mapping} {w}x{h} gen={gen} rgb_nz={rgb_nz} px0=[{},{},{},{}] (QMP will be black)",
                    px0[0], px0[1], px0[2], px0[3]
                ));
                crate::observe::fail(format!(
                    "present_black_retain mid={mapping} {w}x{h} gen={gen} (alpha-only/black +0x188)"
                ));
            } else {
                crate::observe::off(format!(
                    "present_content mid={mapping} {w}x{h} gen={gen} rgb_nz={rgb_nz} max_rgb={max_rgb} px0=[{},{},{},{}] encoded={}",
                    px0[0], px0[1], px0[2], px0[3], encoded as u8
                ));
            }
        }
        // No guest-page comparison here. The presented surface's guest window is
        // stale by construction on the Vulkan rail — `import_present` defers the
        // compositor front buffer's writeback on every present, so the pinned
        // resident is authoritative and those pages hold pre-dispatch bytes
        // until a host path reads them. Measured: ~99.5% of the frame differs at
        // full swing on every present, with a deferred window armed every time.
        // The guest's `screencapture` is an oracle because it makes the guest
        // re-execute the composite; its memory for a surface we render into is
        // not.
        // One line per accepted present, verbose-only. `present_enqueue` carried
        // the same fields through the always-on sink alongside it.
        crate::observe::line(format!(
            "present paint mid={mapping} {w}x{h} gen={gen} encoded={} retain={} unpainted={}",
            encoded as u8,
            state.present.frame_valid as u8,
            state.present.unpainted_presents.saturating_add(1)
        ));
        // Account the accepted present. The retain-vs-DisplaySwap (mapping,
        // generation) choice that used to be computed here addressed
        // `copy_to_bgra8`'s Unchanged/expected_generation checks on the QEMU
        // paint; with no paint action produced, the window resolves the frame
        // from `state.present` directly and the distinction has no consumer.
        enqueue_present_scanout(state, host, w, h);
        // Entry-side waitForPendingFrames / apple-gfx pending_frames:
        // count accepted presents until host paint. Stamp still
        // fires with this packet (below) — PGDisplay completion
        // after +0x188 retain, not after host encode.
    } else {
        // Named present without geom: still a product present attempt — leave
        // BAR1 (not a ClearOnly-init handoff defer, which requires geom).
        // Keep early_front peer tracker for dual-mid ClearOnly presents.
        state.present.frame_flush_seen = true;
    }
    // else: hold last painted console (no HostAction / no resize).

    // PGDisplay completion block runs for every present after the
    // +0x188 retain (also when geometry held the paint): display
    // shared-page present bit + conditional display IRQ.
    signal_display_present_complete(state, host);
    ChildPacketDisposition::Complete
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ChildPacketDisposition {
    Complete,
    Deferred,
}

/// One line per `ExecIndirect2` packet, naming everything the packet executed.
///
/// # Do not read these counters as a census
///
/// On the always-on sink this line is **failure-selected**: the caller emits it
/// through `observe::fail` only when `packet_failed`, and sends healthy packets
/// to the verbose-gated `observe::line` instead. So every copy of this line in
/// `/tmp/reims-vgpu-fail.log` is, by construction, a packet that failed a draw
/// or an ICB — which is why the accumulated log shows `draws_ok=0 draws_fail=1`
/// on all 414 of them. That ratio is the filter, not the draw path.
///
/// The same trap hides the ICB rail. `icb_ok=0` across those lines does **not**
/// mean the guest never runs indirect command buffers; a packet whose ICBs all
/// succeeded is exactly a packet that did not fail, so it never reached this
/// sink. Answering "does ICB run at all?" needs `REIMS_VGPU_DRAW_LOG`, or a
/// `note_store_route` counter that is not conditioned on failure.
fn exec_summary(channel_id: u32, result: &crate::runtime::exec::ExecResult, plen: usize) -> String {
    format!(
        "exec_indirect2 ch={channel_id} task={} streams={} saw_draw={} clears={} draws_ok={} draws_fail={} rt_resolves={} guest_stores={} icb_ok={} icb_fail={} compute_ctrl_fail={} compute_icb_fail={} render_unbinds={}/{}/{} total_us={} plen={plen}",
        result.task_id,
        result.streams_loaded,
        result.saw_draw as u8,
        result.clears_applied,
        result.metal_draws_ok,
        result.metal_draws_fail,
        result.render_attachment_resolves,
        result.render_guest_stores,
        result.render_icb_ok,
        result.render_icb_fail,
        result.compute_control_fail,
        result.compute_icb_fail,
        result.buffer_unbinds,
        result.texture_unbinds,
        result.sampler_unbinds,
        result.total_us,
    )
}

/// A synchronous ExecIndirect2 holding `DeviceInner` for this long starves the
/// guest's read-to-clear completion/status registers. This is a diagnostic
/// proxy only; it never changes packet ordering or completion behavior.
const SYNC_EXEC_STALL_US: u64 = 250_000;

#[inline]
fn sync_exec_stalled(total_us: u64) -> bool {
    total_us >= SYNC_EXEC_STALL_US
}

fn process_child_packet<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    channel_id: u32,
    packet: &Packet,
) -> ChildPacketDisposition {
    match packet.opcode {
        CHILD_OP_DEFINE_TASK2 => {
            apply_define_task2(
                state,
                host,
                &packet.payload,
                &format!("child ch={channel_id}"),
            );
        }
        CHILD_OP_SET_OBJECT_LIST => {
            // A short SET_OBJECT_LIST leaves the task's object list unbound —
            // every type-11 texture/object resolve on it then fails
            // (object_list_count==0). Never on a well-formed boot.
            if !packet_short(
                "set_object_list",
                Some(channel_id),
                packet.payload.len(),
                SET_OBJECT_LIST_LEN,
            ) {
                let task_id = ld32(&packet.payload[SET_OBJECT_LIST_TASK_ID..]);
                let pfn = ld32(&packet.payload[SET_OBJECT_LIST_PFN..]);
                let count = ld32(&packet.payload[SET_OBJECT_LIST_COUNT..]);
                let _ = state.set_object_list(task_id, pfn, count);
            }
        }
        CHILD_OP_DELETE_OBJECT => {
            if !packet_short("delete_object", Some(channel_id), packet.payload.len(), 8) {
                let task_id = ld32(&packet.payload[0..]);
                let id = ld32(&packet.payload[4..]);
                let _ = state.delete_object(task_id, id);
            }
        }
        // PVG CmdDeleteTask (0x20) on child channels too (was SMALL_ID alias only in decode).
        CHILD_OP_DELETE_TASK => {
            let task_id = if packet.payload.len() >= 4 {
                ld32(&packet.payload[0..])
            } else {
                0
            };
            let ok = state.delete_task(task_id);
            crate::observe::off(format!(
                "delete_task ch={channel_id} task={task_id} ok={} plen={}",
                ok as u8,
                packet.payload.len()
            ));
        }
        CHILD_OP_SETUP_SHARED_STATE => {
            // A short SETUP_SHARED_STATE drops display registration:
            // shared_gpa/index never latch, so the display NEVER onlines and the
            // boot wedges on a blank/console frame. The loudest of this class.
            if !packet_short(
                "setup_shared_state",
                Some(channel_id),
                packet.payload.len(),
                CHILD_SHARED_STATE_LEN,
            ) {
                let index = ld32(&packet.payload[CHILD_SHARED_STATE_INDEX..]);
                let pfn = ld32(&packet.payload[CHILD_SHARED_STATE_PFN..]);
                // reinit=1 means the guest tears down + re-registers the display
                // shared page while it was already ONLINE — the AppleParavirtDisplayPipe
                // setupSharedState/teardownSharedState re-init that makes WindowServer
                // rebuild display attributes (signalDisplay bit2 → process_online).
                // A reinit AFTER present_converge is the smoking gun for the intermittent
                // post-converge boot-progress overlay. Rare
                // event → always-on so a bad boot leaves a display-lifecycle timeline.
                let reinit = state.display.online_acked as u8;
                state.display.display_index = index;
                state.display.shared_gpa = state.pfn_gpa(pfn);
                state.display.online_acked = false;
                state.display.online_tries = 0;
                state.display.poll_ctr = 0;
                crate::observe::fail(format!(
                    "display_shared_state_setup index={index} gpa={:#x} reinit={reinit}",
                    state.display.shared_gpa
                ));
                // Archive apple_pv_gpu_display_setup: fill descriptor + modes
                // before completion so createDisplayAttributes sees TimingElements.
                // Do **not** pulse ONLINE here — enable() has not set +0x104 yet
                // (archive poll waits for mask bit 2, then pending+IRQ).
                fill_display_descriptor(host, state.display.shared_gpa, index, state.page_size());
            }
        }
        CHILD_OP_ONLINE_ACK => {
            state.display.online_acked = true;
            // The connectionChange-ack (process_online opcode 2) is believed to
            // echo the shared-descriptor `+0x200` token back to the host in its
            // payload. We consume the ack (online_acked)
            // but never inspect that token — capture it here (raw first words +
            // len, rare/once-per-online so no flood) so a bad boot records what
            // value the guest round-tripped. Measure-only.
            let w0 = if packet.payload.len() >= 4 {
                ld32(&packet.payload[0..])
            } else {
                0
            };
            let w1 = if packet.payload.len() >= 8 {
                ld32(&packet.payload[4..])
            } else {
                0
            };
            crate::observe::fail(format!(
                "display_online_ack index={} plen={} w0={:#x} w1={:#x}",
                state.display.display_index,
                packet.payload.len(),
                w0,
                w1
            ));
        }
        /*
         * Scanout policy:
         * - Early boot: front type-11 writebacks paint while !frame_flush_seen
         *   and job W×H matches established console (no mid-switch thrash).
         * - After first boundary: display presents paint (op8 DisplaySwap on
         *   arm ch4, **or** op6/7 on x86 Ventura/Tahoe display ch5).
         * - ch2 PRESENT_FRAME 0x28 / FLUSH 0x3b: bookkeeping only (mid-composite).
         */
        // The three display present commands. op8 `CmdDisplaySwapMapping` is
        // the arm/EFI-era path; x86 Ventura/Tahoe drives the display pipe with
        // op6 `CmdDisplayTransaction3` and its gamma variant op7. They differ
        // only in where the surface word sits, which
        // `display_txn_trailer_slots` owns for all three.
        opcode @ (CHILD_OP_DISPLAY_SWAP | CHILD_OP_PRESENT_X86 | CHILD_OP_PRESENT_GAMMA_X86) => {
            note_display_txn_payload(state, channel_id, packet);
            let Some(mapping) = present_surface_id(opcode, &packet.payload) else {
                crate::observe::fail(format!(
                    "packet_short reason=display_present_short ch={channel_id} op={opcode:#x} \
                     plen={} need={}",
                    packet.payload.len(),
                    display_txn_trailer_len(opcode)
                ));
                return ChildPacketDisposition::Complete;
            };
            // Per-present decode census (~30k/session under animation); the
            // present rate lives in the present_proxy summary, so gate the
            // per-packet line behind REIMS_VGPU_DRAW_LOG. `pipe` is the display
            // index for op8 and the pipe index for op6/7 — payload word 0 in
            // both. `task` is op6/7's task field, which is the submitting task's
            // and not a completion stamp; the packet's own stamp lives in the
            // FIFO header. op8 has no such word, and prints `-`.
            let (_, task_slot) = display_txn_trailer_slots(opcode);
            let task = task_slot
                .map(|slot| format!("{:#x}", ld32(&packet.payload[slot * 4..])))
                .unwrap_or_else(|| "-".to_string());
            crate::observe::line(format!(
                "present_txn op={opcode:#x} ch={channel_id} pipe={} sid={mapping} task={task} \
                 plen={} unpainted={} prior_present_mapping={}",
                ld32(&packet.payload[0..]),
                packet.payload.len(),
                state.present.unpainted_presents,
                state.present.present_mapping
            ));
            if present_named_mapping(state, host, channel_id, mapping)
                == ChildPacketDisposition::Deferred
            {
                return ChildPacketDisposition::Deferred;
            }
        }
        CHILD_OP_PRESENT_FRAME => {
            // PVG: CmdDeleteObject on some maps; arm misread as present. Never
            // paint. x86 present is op6/7 on display channel.
            let _ = packet;
        }
        // PVG / Monterey: 0x3b = CmdGetComputeInfo (query). Must write reply
        // before stamp or createComputePipeline stalls (texture-ref 29-06-26).
        // `CHILD_OP_PRESENT_FRAME_FLUSH` is the recovered legacy name for the
        // same wire opcode.
        CHILD_OP_GET_COMPUTE_INFO => {
            if packet.payload.len() >= 24 {
                let _ = reply_compute_info(state, host, &packet.payload);
            } else {
                crate::observe::fail(format!(
                    "get_compute_info short ch={channel_id} len={}",
                    packet.payload.len()
                ));
            }
        }
        CHILD_OP_CURSOR_SHOW => {
            if !packet_short("cursor_show", Some(channel_id), packet.payload.len(), 8) {
                let show = ld32(&packet.payload[4..]) != 0;
                state.cursor.show = show;
                sample_cursor_position(state, host);
                host.enqueue(HostAction::cursor(state.cursor.x, state.cursor.y, show));
            }
        }
        CHILD_OP_CURSOR_GLYPH => {
            if load_cursor_glyph(state, host, packet) {
                host.enqueue(HostAction::cursor_glyph());
                host.enqueue(HostAction::cursor(
                    state.cursor.x,
                    state.cursor.y,
                    state.cursor.show,
                ));
            }
        }
        CHILD_OP_EXEC_INDIRECT2 => {
            if packet.payload.len() < 12 {
                state.record_fail(FailEvent::UnsupportedExec {
                    channel: channel_id,
                    fault: ExecFault::Indirect2Short,
                });
            } else {
                // Process this channel's exec packet. Archive does not drain
                // other child FIFOs here; surface RAW is render_wait_surface on
                // the specific type-11/GVA key at sample/Load/swap sites.
                let result =
                    crate::runtime::exec::process_exec_indirect2(state, host, &packet.payload);
                let channel_bit = 1u32.checked_shl(channel_id).unwrap_or(0);
                if result.deferred {
                    if channel_bit != 0 && state.translation_deferred_mask & channel_bit == 0 {
                        state.translation_deferred_mask |= channel_bit;
                        // Census for the same reason as `translation_order_hold`:
                        // the packet is NOT consumed (`Deferred` leaves it at the
                        // FIFO head to be retried), and the matching
                        // `exec_translation_ready` below is already `off`. Boot 87:
                        // 55 deferrals, 56 readies.
                        crate::observe::off(format!(
                            "exec_translation_deferred reason=air_loading ch={channel_id} task={} pending_mask={:#x}",
                            result.task_id, state.translation_deferred_mask
                        ));
                    }
                    return ChildPacketDisposition::Deferred;
                }
                if channel_bit != 0 && state.translation_deferred_mask & channel_bit != 0 {
                    state.translation_deferred_mask &= !channel_bit;
                    crate::observe::off(format!(
                        "exec_translation_ready ch={channel_id} task={} pending_mask={:#x}",
                        result.task_id, state.translation_deferred_mask
                    ));
                }
                // Failure-carrying packets keep the full per-packet line on the
                // always-on sink (context for the per-site reason=<slug> lines).
                // Healthy packets are expected control flow and stay quiet
                // unless the draw log is on — the per-packet form ran ~1k
                // lines/s under Safari scroll.
                let packet_failed = result.metal_draws_fail > 0
                    || result.render_icb_fail > 0
                    || result.compute_control_fail > 0
                    || result.compute_icb_fail > 0;
                if packet_failed {
                    crate::observe::fail(exec_summary(channel_id, &result, packet.payload.len()));
                } else if crate::observe::draw_log_enabled() {
                    crate::observe::line(exec_summary(channel_id, &result, packet.payload.len()));
                }
                if sync_exec_stalled(result.total_us) {
                    crate::observe::fail(format!(
                        "TRANSPORT reason=sync_exec_lock_hold ch={channel_id} task={} total_us={} draws={} rt_resolves={} guest_stores={} threshold_us={SYNC_EXEC_STALL_US}",
                        result.task_id,
                        result.total_us,
                        result.metal_draws_ok.saturating_add(result.metal_draws_fail),
                        result.render_attachment_resolves,
                        result.render_guest_stores
                    ));
                }
            }
        }
        crate::runtime::decode::fifo::CHILD_OP_CONFIG_40 => {
            let _ = reply_heap_texture_size_and_align(state, host, &packet.payload);
        }
        // The one packet that says a cached page list has gone stale. It used to
        // sit in the stamp-and-forget family below, which is why
        // `mapping_page_drift` could report "the guest re-pointed this surface
        // and no packet said so" — the packet arrived and was discarded.
        CHILD_OP_REPLACE_PHYSICAL => {
            if !packet_short(
                "replace_physical",
                Some(channel_id),
                packet.payload.len(),
                crate::runtime::decode::fifo::CHILD_REPLACE_PHYSICAL_LEN as usize,
            ) {
                if let Some(cmd) =
                    crate::runtime::decode::fifo::decode_replace_physical(&packet.payload)
                {
                    crate::runtime::objects::replace_physical(state, cmd.task_id, cmd.object_id);
                }
            }
        }
        // PVG bookkeeping family: accept + stamp (already below). Full PT/map
        // semantics land with metal2vulkan encode; until then fail-visible
        // UnknownChildOpcode flooded /tmp/reims-vgpu-fail and hid draw telemetry.
        CHILD_OP_UNMAP_MEMORY
        | CHILD_OP_MAP_MEMORY2
        | CHILD_OP_INVALIDATE_RESOURCES
        | CHILD_OP_SYNCHRONIZE_RESOURCES
        | CHILD_OP_DELETE_IOSURFACE_BACKING2 => {
            // Stamp-complete for PT wire (no invent). Unmap/Map retire
            // gva_host_views; verbose-gated map_probe census for stage Unmapped.
            //
            // Live MapMemory2 plen=20 layout lead (not yet contract-final):
            //   task_id@0 u32, gva@4 u64, length@12 u64  (matches fifo MapMemoryCommand).
            let plen = packet.payload.len();
            let name = match packet.opcode {
                CHILD_OP_MAP_MEMORY2 => "MapMemory2",
                CHILD_OP_UNMAP_MEMORY => "UnmapMemory",
                CHILD_OP_INVALIDATE_RESOURCES => "InvalidateResources",
                CHILD_OP_SYNCHRONIZE_RESOURCES => "SynchronizeResources",
                CHILD_OP_DELETE_IOSURFACE_BACKING2 => "DeleteIOSurfaceBacking2",
                _ => "map_family",
            };
            if matches!(packet.opcode, CHILD_OP_MAP_MEMORY2 | CHILD_OP_UNMAP_MEMORY) && plen >= 20 {
                let task_id = crate::contract::endian::ld32(&packet.payload[0..]);
                let gva = crate::contract::endian::ld64(&packet.payload[4..]);
                let length = crate::contract::endian::ld64(&packet.payload[12..]);
                // Verbose-gated walk probe at map/unmap time. This runs a full
                // guest page-table walk (`diagnose_gva_walk`) purely to build the
                // log string, and fired ~9k times/boot on the drain path — a flood
                // and a real per-map cost. Gate it (and the periodic census) behind
                // `REIMS_VGPU_DRAW_LOG=1` so a normal boot pays neither; the functional
                // view-retire below stays always-on. Wire has no PPNs — the probe
                // asks whether the guest PT is already walkable under wire task_id.
                if crate::observe::draw_log_enabled() {
                    let walk = crate::runtime::gva_mem::diagnose_gva_walk(
                        host,
                        &state.tasks,
                        task_id,
                        gva,
                        state.page_shift,
                    );
                    crate::observe::line(format!(
                        "map_probe op={name} ch={channel_id} task={task_id} gva={gva:#x} len={length:#x} page_shift={} {walk}",
                        state.page_shift
                    ));
                    // Periodic active-task census (every 32 map/unmap) for boot overview.
                    state.map_family_events = state.map_family_events.saturating_add(1);
                    if state.map_family_events == 1 || state.map_family_events.is_multiple_of(32) {
                        let census = crate::runtime::gva_mem::format_active_tasks(&state.tasks);
                        crate::observe::line(format!(
                            "map_census n={} last_op={name} task={task_id} {census}",
                            state.map_family_events
                        ));
                    }
                }
                // RE (AppleParavirtMemoryMap): Unmap/Map only mutate the **task
                // page table** then notify — wire has no PPNs. Guest order is
                // deallocate/allocate **then** FIFO, so:
                // - Unmap notify: PTEs already gone → cannot GVA-write; retain
                //   host_gva_surfaces for sample (wallpaper wipe class).
                // - Map notify: PTEs already live → flush host_gva encode into
                //   **new** PFNs (not invent PTEs; not invent geom). Discrete
                //   type-2/3 content may live only in host_cache until this.
                // Samples still prefer host_cache GVA key on Load.
                //
                // HostOps **views** (gva_host_views) are the opposite of encode
                // cache: they alias the pages that were in the GPU PT. On Unmap
                // those pages are no longer mapped for the GPU — drop any host
                // view covering the range (Apple unmapMemory analogue). On Map
                // the PFNs may have changed under the same GVA — drop stale
                // views so the next ensure_gva_view re-walks. Does not invent
                // PTEs and does not destroy host_gva_surfaces content.
                if gva != 0 && length != 0 {
                    let n = crate::runtime::gva_view::retire_gva_views_overlapping(
                        state, task_id, gva, length,
                    );
                    let op = if packet.opcode == CHILD_OP_UNMAP_MEMORY {
                        "unmap_memory"
                    } else {
                        "map_memory2"
                    };
                    crate::runtime::gva_view::log_retire(op, task_id, gva, length, n);
                }
                // Deferred GVA render-Store windows overlapping the notified
                // VA range land **cache-only**: on Unmap the PTEs are already
                // gone; on Map the PFNs are fresh and the map-notify guest
                // flush is forbidden (PTE-corruption class). The encode cache
                // preserves the content for samples (wallpaper-retain).
                if gva != 0 && length != 0 && !state.gva_deferred_flush.is_empty() {
                    let hi = gva.saturating_add(length);
                    let overlapped: Vec<u64> = state
                        .gva_deferred_flush
                        .iter()
                        .filter(|(&wgva, e)| {
                            // This task's windows only. A GVA means nothing
                            // outside the address space that named it, so the
                            // overlap test is an overlap only once the task
                            // matches — and both sides are slot ids
                            // (`task_slot::resolve_task_word` on one, the
                            // unshifted `MapMemory2`/`UnmapMemory` word on the
                            // other). The `>> 1` arms this replaces also matched
                            // slots `task_id / 2`, `2 * task_id` and
                            // `2 * task_id + 1`: live, unrelated tasks whose
                            // pending frames were then landed cache-only and so
                            // never reached guest RAM.
                            e.task_id == task_id && wgva < hi && gva < wgva.saturating_add(e.span())
                        })
                        .map(|(&wgva, _)| wgva)
                        .collect();
                    for wgva in overlapped {
                        let trigger = if packet.opcode == CHILD_OP_UNMAP_MEMORY {
                            "unmap"
                        } else {
                            "remap"
                        };
                        crate::runtime::storage_flush::flush_gva_exact(
                            state, host, wgva, false, trigger,
                        );
                    }
                }
                // There is deliberately no host_cache→guest GVA flush on
                // MapMemory2. One existed and was disabled after
                // serial-20260714-035023: PTE Corruption (freelist-shaped
                // 0xff100000ff000000) ~135s into boot while it was writing —
                // one Map of len=0x1c3e000 alone drove 13 GVA rewrites. Samples
                // use the `host_gva_surfaces` retain on Unmap instead. Any
                // re-introduction has to be a *narrower* policy than that one
                // (exact-base only, no multi-key heap maps) and RE-justified, so
                // the broad implementation is not kept around to be switched
                // back on. See kb map-memory2 / xnu-pte-corruption-windowserver.
            } else if packet.opcode == CHILD_OP_DELETE_IOSURFACE_BACKING2 && plen >= 8 {
                // The live Ventura payload agrees with the resource contract:
                // `{objectID, taskID}`. This is the lifetime
                // boundary for the host IOSurface backing, not stamp-only
                // bookkeeping. Keeping page_entries after it lets later id
                // reuse/clear write pixels into pages the guest has recycled.
                let object_id = crate::contract::endian::ld32(&packet.payload[0..]);
                let task_id = crate::contract::endian::ld32(&packet.payload[4..]);
                // Never write guest pages here — the delete trails the guest's
                // CPU-side release asynchronously and the pages may already be
                // recycled (boot-16 PTE-corruption panic: a 14.7 MB delete-time
                // flush landed pixel bytes in a PTE page). But the id itself
                // may ALSO already be re-used by a live surface whose paint is
                // still deferred (~20 ms recycle under scroll — black-band
                // class), so content state must survive until the next page
                // resolve proves which incarnation this delete was for
                // (fingerprint compare in mapper::resolve). A second delete
                // with no resolve between is genuinely dead: tear down fully.
                let mode = if state.mapping_backing_condemned(object_id) {
                    crate::runtime::storage_flush::drop_windows(state, object_id, "delete_backing");
                    let _ = state.unmap_surface(object_id);
                    "dead"
                } else if state.condemn_surface_backing(object_id) {
                    "condemn"
                } else {
                    // No resolved pages ⇒ nothing a stale delete could hurt.
                    crate::runtime::storage_flush::drop_windows(state, object_id, "delete_backing");
                    let _ = state.unmap_surface(object_id);
                    "unmapped"
                };
                crate::runtime::mapper::flush_retired_views(state, host);
                if crate::observe::draw_log_enabled() {
                    crate::observe::line(format!(
                        "map_family op=DeleteIOSurfaceBacking2 ch={channel_id} object={object_id} task={task_id} plen={plen} mode={mode}"
                    ));
                }
            } else if packet.opcode == CHILD_OP_REPLACE_PHYSICAL && plen >= 8 {
                // Archived lead: {taskID, objectID}; live total_size=20 ⇒ header+payload.
                // Guest may rebind physical pages under the object — drop cached
                // page_entries / contig so the next Store re-resolves (safe
                // zero-copy / freelist-prevention). Object id is typically a
                // texture ref; also try as mapping_id when texture map misses.
                let task_id = crate::contract::endian::ld32(&packet.payload[0..]);
                let object_id = crate::contract::endian::ld32(&packet.payload[4..]);
                // Guest MAY rebind physical pages under the object — retire the
                // cached bindings so the next Store re-resolves (freelist
                // prevention). Like the trailing delete, this must not destroy
                // a live incarnation's deferred paint (a tile's only copy sits
                // in the GPU resident until writeback): condemn with a page
                // fingerprint and let the next resolve decide. A genuine rebind
                // resolves to different pages → bump + windows dropped there;
                // a revalidation/no-op resolves identical → content survives.
                let mut n_inv = 0u32;
                let mut n_cond = 0u32;
                let mut targets = vec![object_id];
                if let Some(&mid) = state.texture_to_mapping.get(&(task_id, object_id)) {
                    if mid != object_id {
                        targets.push(mid);
                    }
                }
                for id in targets {
                    if state.mapping_backing_condemned(id) {
                        // Decision already pending; the next resolve settles it.
                        continue;
                    }
                    if state.condemn_surface_backing(id) {
                        n_cond = n_cond.saturating_add(1);
                    } else {
                        // No resolved pages ⇒ nothing a stale replace could
                        // hurt; keep the old teardown semantics.
                        crate::runtime::storage_flush::drop_windows(state, id, "replace_physical");
                        if state.invalidate_mapping_pages(id) {
                            n_inv = n_inv.saturating_add(1);
                        }
                    }
                }
                // Per-op echo of a routine lifecycle op. Keep the per-op detail
                // (inv/condemn split) gated so it does not flood the always-on
                // sink; the `draw_log_enabled()` guard also skips the format
                // alloc on a healthy boot (mirrors the DeleteIOSurfaceBacking2
                // site above).
                if crate::observe::draw_log_enabled() {
                    crate::observe::line(format!(
                        "map_family op=ReplacePhysical ch={channel_id} task={task_id} object={object_id} plen={plen} inv_pages={n_inv} condemned={n_cond}"
                    ));
                }
            } else if packet.opcode == CHILD_OP_INVALIDATE_RESOURCES {
                // RE: {task_id, count} + count×{object_id, 4×u8 validity ops}.
                // Ops (PVG host layout): clr_host, set_host, clr_guest, set_guest.
                // Pageon hardcodes LE 01 00 00 01 = clr hostValid + set guestValid.
                //
                // The same four bytes the EXEC_INDIRECT2 resource table carries,
                // through the same consumer: this producer's records are 8 bytes
                // and that one's are 24, but the quad is one contract and must
                // not acquire two meanings.
                use crate::runtime::decode::fifo::{
                    decode_invalidate_resources, CHILD_INVALIDATE_PAGEON_FLAGS,
                };
                use crate::runtime::resource_validity::{apply, ValiditySite};
                match decode_invalidate_resources(&packet.payload) {
                    Some(cmd) => {
                        let mut bumped = 0u32;
                        let mut miss = 0u32;
                        let mut windows_dropped = 0u32;
                        for rec in &cmd.records {
                            let outcome = apply(
                                state,
                                cmd.task_id,
                                rec.object_id,
                                rec.ops,
                                ValiditySite::InvalidateResources,
                            );
                            bumped = bumped.saturating_add(outcome.bumped);
                            windows_dropped =
                                windows_dropped.saturating_add(outcome.windows_dropped);
                            if outcome.missed {
                                miss = miss.saturating_add(1);
                            }
                        }
                        // One counter here, two on the exec side: `pageBacking`
                        // names mapping ids, so a record this device holds no
                        // mapping for is already the surprising case. The exec
                        // table names task object refs, most of which have no
                        // surface state by construction.
                        note_store_route_n("validity_miss_inv", miss as u64);
                        let rec0 = cmd.records.first();
                        let oid = rec0.map(|r| r.object_id).unwrap_or(0);
                        let flags = rec0.map(|r| r.flags).unwrap_or(0);
                        let ops = rec0.map(|r| r.ops).unwrap_or_default();
                        let pageon = flags == CHILD_INVALIDATE_PAGEON_FLAGS;
                        // ~11k/boot of routine guest cache-coherence ops. The
                        // always-on rate is the `validity_*` family in the
                        // per-second `store_routes` line; gate the per-op decode
                        // detail so it does not bury the curated fail view. The
                        // `decode_fail` and `inv_multi` paths below stay
                        // fail-visible, and the guard also skips the format alloc
                        // on a healthy boot.
                        if crate::observe::draw_log_enabled() {
                            crate::observe::line(format!(
                            "map_family op=InvalidateResources opcode={:#x} ch={channel_id} plen={plen} task={} count={} oid={oid:#x} flags={flags:#x} clr_h={} set_h={} clr_g={} set_g={} pageon={pageon} bumped={bumped} miss={miss} windows_dropped={windows_dropped}",
                            packet.opcode,
                            cmd.task_id,
                            cmd.count,
                            ops.clear_host_valid,
                            ops.set_host_valid,
                            ops.clear_guest_valid,
                            ops.set_guest_valid
                        ));
                        }
                        if cmd.count > 1 {
                            let ids: Vec<String> = cmd
                                .records
                                .iter()
                                .map(|r| {
                                    format!(
                                        "{:#x}:clr_h={}/set_g={}",
                                        r.object_id, r.ops.clear_host_valid, r.ops.set_guest_valid
                                    )
                                })
                                .collect();
                            crate::observe::fail(format!(
                                "inv_multi ch={channel_id} task={} n={} recs=[{}]",
                                cmd.task_id,
                                cmd.count,
                                ids.join(",")
                            ));
                        }
                    }
                    None => {
                        let w0 = if plen >= 4 {
                            crate::contract::endian::ld32(&packet.payload[0..])
                        } else {
                            0
                        };
                        let w1 = if plen >= 8 {
                            crate::contract::endian::ld32(&packet.payload[4..])
                        } else {
                            0
                        };
                        crate::observe::fail(format!(
                            "map_family op=InvalidateResources opcode={:#x} ch={channel_id} plen={plen} decode_fail w0={w0:#x} w1={w1:#x}",
                            packet.opcode
                        ));
                    }
                }
            } else if packet.opcode == CHILD_OP_SYNCHRONIZE_RESOURCES {
                // RE synchronizeForUnwire → FIFO 0x35: {task,count}+{oid} only.
                // Guest contract is finish host GPU use before pageoff — not
                // "host invents pixels into guest pages." Discrete host_cache→
                // guest write was product invent (pre-change successful boots
                // were stamp-only). Keep decode + wait_surface; no guest write.
                use crate::runtime::decode::fifo::decode_synchronize_resources;
                match decode_synchronize_resources(&packet.payload) {
                    Some(cmd) => {
                        // The guest is about to CPU-read these resources
                        // (pageoff/unwire): land every deferred writeback
                        // (render/compute/linear-alias) into guest pages first
                        // — the only host-visible choke point for guest CPU
                        // reads (boot-25 black-wallpaper class).
                        let mut flushed = 0u32;
                        let mut flush_ok = true;
                        for &oid in &cmd.object_ids {
                            let (ok, n) =
                                crate::runtime::storage_flush::flush_mapping_for_guest_read(
                                    state, host, oid,
                                );
                            flush_ok &= ok;
                            flushed = flushed.saturating_add(n);
                            // The declaration itself is the read witness: the
                            // guest CPU load that follows leaves no trace the
                            // device can see, so a flush landed for a declared
                            // read must not be scored as unread.
                            crate::runtime::storage_flush::note_render_flush_pages_read(state, oid);
                        }
                        let oid = cmd.object_ids.first().copied().unwrap_or(0);
                        // Count into the always-on teardown-churn proxy; the
                        // per-event census floods to ~49k/session under a
                        // continuously-animating app, so it moves behind
                        // REIMS_VGPU_DRAW_LOG below.
                        // A deferred guest-read flush that did NOT land right
                        // before the guest CPU-reads these pages is a genuine
                        // black/stale-content drop — previously buried in the
                        // off() census (invisible in the curated fail view).
                        // Promote it to a reason-slugged fail line.
                        if !flush_ok {
                            crate::observe::fail(format!(
                                "map_family op=SynchronizeResources reason=guest_read_flush_incomplete ch={channel_id} task={} oid={oid:#x} deferred_flushed={flushed}",
                                cmd.task_id
                            ));
                        }
                        if crate::observe::draw_log_enabled() {
                            crate::observe::line(format!(
                                "map_family op=SynchronizeResources opcode={:#x} ch={channel_id} plen={plen} task={} count={} oid={oid:#x} deferred_flushed={flushed} flush_ok={flush_ok}",
                                packet.opcode, cmd.task_id, cmd.count
                            ));
                        }
                        if cmd.count > 1 {
                            let ids: Vec<String> =
                                cmd.object_ids.iter().map(|id| format!("{id:#x}")).collect();
                            crate::observe::fail(format!(
                                "sync_multi ch={channel_id} task={} n={} oids=[{}]",
                                cmd.task_id,
                                cmd.count,
                                ids.join(",")
                            ));
                        }
                    }
                    None => {
                        let w0 = if plen >= 4 {
                            crate::contract::endian::ld32(&packet.payload[0..])
                        } else {
                            0
                        };
                        let w1 = if plen >= 8 {
                            crate::contract::endian::ld32(&packet.payload[4..])
                        } else {
                            0
                        };
                        crate::observe::fail(format!(
                            "map_family op=SynchronizeResources opcode={:#x} ch={channel_id} plen={plen} decode_fail w0={w0:#x} w1={w1:#x}",
                            packet.opcode
                        ));
                    }
                }
            } else {
                let w0 = if plen >= 4 {
                    crate::contract::endian::ld32(&packet.payload[0..])
                } else {
                    0
                };
                let w1 = if plen >= 8 {
                    crate::contract::endian::ld32(&packet.payload[4..])
                } else {
                    0
                };
                crate::observe::off(format!(
                    "map_family op={name} opcode={:#x} ch={channel_id} plen={plen} w0={w0:#x} w1={w1:#x}",
                    packet.opcode
                ));
            }
        }
        // A fence with no payload. The guest emits it from a present's failure
        // and teardown legs to order work it is abandoning, and retiring its
        // stamps — which the drain does for every accepted packet — is the whole
        // contract. Named so it stops being reported as an unknown opcode.
        CHILD_OP_FLUSH_CHANNEL_EVENT => {
            crate::runtime::drain::note_store_route("child_flush_channel_event");
            // The command allocates no bytes, so payload is the one thing that
            // can falsify this reading. Bytes here would mean the command grew a
            // form this arm does not decode, and dropping them silently is what
            // the unknown-opcode arm was at least loud about.
            if !packet.payload.is_empty() {
                crate::observe::fail(format!(
                    "child_flush_channel_event fail reason=unexpected_payload ch={channel_id} \
                     plen={} (this command carries stamps only; a payload means it has grown \
                     a form this arm does not decode)",
                    packet.payload.len()
                ));
            }
        }
        _ => {
            state.record_fail(FailEvent::UnknownChildOpcode {
                channel: channel_id,
                opcode: packet.opcode,
                total_size: packet.total_size,
                stamp_count: packet.stamp_count,
                payload: packet.payload.clone(),
            });
        }
    }
    ChildPacketDisposition::Complete
}

/// Drain one child channel.
pub fn drain_child_fifo<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    channel_id: u32,
) {
    if state.gfx.root_page == 0 || channel_id == 0 || channel_id as usize >= MAX_CHANNELS {
        return;
    }
    let Some(regs_off) = child_reg_block_offset(channel_id) else {
        return;
    };
    let regs_gpa = state.pfn_gpa(state.gfx.root_page) + regs_off;

    let mut head = match crate::runtime::host::read_u32(host, regs_gpa + CHILD_REG_HEAD) {
        Ok(v) => v,
        Err(_) => {
            state.record_fail(FailEvent::MalformedChildPacket {
                channel: channel_id,
                fault: PacketFault::ChildRegsHeadRead,
                head: 0,
            });
            return;
        }
    };
    let stamp_index = match crate::runtime::host::read_u32(host, regs_gpa + CHILD_REG_STAMP_INDEX) {
        Ok(v) => v,
        Err(_) => {
            state.record_fail(FailEvent::MalformedChildPacket {
                channel: channel_id,
                fault: PacketFault::ChildRegsStampRead,
                head,
            });
            return;
        }
    };
    let base_pfn = match crate::runtime::host::read_u32(host, regs_gpa + CHILD_REG_BASE_PFN) {
        Ok(v) => v,
        Err(_) => {
            state.record_fail(FailEvent::MalformedChildPacket {
                channel: channel_id,
                fault: PacketFault::ChildRegsBaseRead,
                head,
            });
            return;
        }
    };

    let ring_length = ensure_child_ring(state, host, channel_id, base_pfn);
    if ring_length == 0 {
        return;
    }
    let page_gpas = state.child_rings[channel_id as usize].page_gpas.clone();

    // Nested drain_other must skip this channel (no re-enter head).
    // Use a bit mask so nested drains skip the full stack, not only the leaf.
    let prev_channel = state.draining_channel;
    let bit = 1u32 << channel_id;
    state.draining_channel = channel_id;
    state.draining_mask |= bit;

    loop {
        let tail = match crate::runtime::host::read_u32(host, regs_gpa + CHILD_REG_TAIL) {
            Ok(v) => v,
            Err(_) => {
                state.record_fail(FailEvent::MalformedChildPacket {
                    channel: channel_id,
                    fault: PacketFault::ChildTailRead,
                    head,
                });
                break;
            }
        };
        if head == tail {
            break;
        }
        let Some(available) = published_byte_count(head, tail, ring_length) else {
            state.record_fail(FailEvent::MalformedChildPacket {
                channel: channel_id,
                fault: PacketFault::DesyncedHeadTail,
                head,
            });
            break;
        };
        if available < PACKET_HEADER_LEN {
            break;
        }
        let header = match read_child_ring_bytes(
            host,
            &page_gpas,
            ring_length,
            head,
            PACKET_HEADER_LEN,
            state.page_shift,
        ) {
            Ok(h) => h,
            Err(_) => {
                state.record_fail(FailEvent::MalformedChildPacket {
                    channel: channel_id,
                    fault: PacketFault::ChildHeaderRead,
                    head,
                });
                break;
            }
        };
        let total_size = ld32(&header[PACKET_TOTAL_SIZE..]);
        let snap_len = if total_size >= PACKET_HEADER_LEN
            && total_size <= ring_length
            && available >= total_size
        {
            total_size
        } else {
            PACKET_HEADER_LEN
        };
        let snap = match read_child_ring_bytes(
            host,
            &page_gpas,
            ring_length,
            head,
            snap_len,
            state.page_shift,
        ) {
            Ok(s) => s,
            Err(_) => {
                state.record_fail(FailEvent::MalformedChildPacket {
                    channel: channel_id,
                    fault: PacketFault::ChildSnapRead,
                    head,
                });
                break;
            }
        };
        // Entry gate before decode of full payload: hold CmdDisplaySwap when
        // host paint is already two presents behind (apple-gfx pending_frames
        // >= 2). Leave head unmoved so body draws on other channels can still
        // land via drain_other; re-enter after note_present_paint_consumed.
        let peek_opcode = ld16(&header[PACKET_OPCODE..]);
        if matches!(
            peek_opcode,
            CHILD_OP_DISPLAY_SWAP | CHILD_OP_PRESENT_X86 | CHILD_OP_PRESENT_GAMMA_X86
        ) && state.present.unpainted_presents >= MAX_UNPAINTED_PRESENTS
        {
            note_present_backpressure_hold(state, channel_id, head, tail);
            // Paint will schedule the next worker slice. Preserve this channel
            // without self-waking the worker ahead of QEMU's action BH.
            state.pending.child_mask |= bit;
            break;
        }
        match decode_packet(&snap, head, available) {
            Ok(packet) => {
                if process_child_packet(state, host, channel_id, &packet)
                    == ChildPacketDisposition::Deferred
                {
                    // Translation owns only immutable AIR bytes. Keep head and
                    // stamp untouched so retry cannot duplicate any packet
                    // side effect; continue with sibling channels in the
                    // outer pending-drain loop.
                    break;
                }
                head = packet.next_head;
                if gpa_map::write_u32(
                    host,
                    regs_gpa + CHILD_REG_HEAD,
                    head,
                    state.page_size() as usize,
                )
                .is_err()
                {
                    // The packet was processed + stamped, but the consumer
                    // pointer never advanced: the next drain re-reads the stale
                    // head and RE-EXECUTES the same packets. Fail-visible so
                    // that silent replay is diagnosable (drain.rs Rank-1 audit).
                    state.record_fail(FailEvent::MalformedChildPacket {
                        channel: channel_id,
                        fault: PacketFault::ChildHeadWriteback,
                        head,
                    });
                }

                // Completion stamp. Execution is sync-per-packet: the packet's
                // work is done by the time control reaches here, so the stamp is
                // owed now. DisplaySwap included — PGDisplay present completion
                // follows the +0x188 retain, not host encode/paint.
                //
                // The archive orders stamps through a per-channel queue because
                // its draw jobs complete asynchronously (`ApplePVGPUDrawJob`,
                // `apple_pv_gpu_render_wait_surface`). If this device ever grows
                // an async execution path, that ordering has to come back with
                // it — and be written against the async model that then exists,
                // not inherited from an empty queue.
                write_stamp(state, host, stamp_index, packet.completion_stamp);
                if state.pending.host_action_yield {
                    if head != tail {
                        state.pending.child_mask |= bit;
                    }
                    break;
                }
            }
            Err(err) => {
                if let Some(fault) = err.fault() {
                    state.record_fail(FailEvent::MalformedChildPacket {
                        channel: channel_id,
                        fault,
                        head,
                    });
                }
                break;
            }
        }
    }

    state.draining_mask &= !bit;
    state.draining_channel = prev_channel;
}

/// Drain iosfc mapper producer→consumer handshake.
///
/// Prefer calling this on the **iosfc producer MMIO path** (publishing vCPU)
/// so `resolve_mapping_backing` KVA walks use `current_cpu`. BH-only resolve
/// with `cpu_memory_rw_debug(first_cpu)` deadlocks against MMIO holding
/// `DEVICES` (see reims-vgpu-mmio.c `read_kva`).
pub fn drain_iosfc<H: HostMemory + HostOps>(state: &mut DeviceState, host: &mut H) {
    let producer = state.iosfc.producer;
    let mut consumer = state.iosfc.consumer;
    if producer == consumer {
        state.pending.iosfc = false;
        return;
    }

    // Process requests between consumer and producer when ring is programmed.
    if state.iosfc.ring_base != 0 && producer > consumer {
        let start = consumer;
        let end = producer;
        for idx in start..end {
            let entry_off = (idx as u64) * MAPPER_REQUEST_ENTRY_LEN as u64;
            let mut e = [0u8; MAPPER_REQUEST_ENTRY_LEN];
            if host
                .read_gpa(state.iosfc.ring_base + entry_off, &mut e)
                .is_err()
            {
                break;
            }
            let rtype = ld32(&e[0..]);
            let mapping_id = ld32(&e[4..]);
            // Capture was taken at producer write for published entry (idx+1).
            let cap = match state.mapper_capture {
                Some(c) if c.producer == idx + 1 => state.mapper_capture.take(),
                _ => None,
            };
            match rtype {
                MAPPER_REQUEST_MAP => {
                    let _ = state.map_surface(mapping_id);
                    if let Some(c) = cap {
                        if c.request_type == MAPPER_REQUEST_MAP {
                            let _ = crate::runtime::mapper::apply_capture(state, &c, mapping_id);
                            // A resolve that re-points this surface at new
                            // pages retires the ones a deferred render window's
                            // pixels belong in, and the flush then drops the
                            // frame outright (`deferred_flush_lost
                            // reason=map_generation_drift`) rather than write a
                            // framebuffer into whatever owns that memory now.
                            // Measured losing icon-sized RGBA16Float surfaces
                            // and the menu-bar strip that way. Land what is
                            // pending first, while its pages are still ours; a
                            // mapping with nothing armed pays nothing.
                            let _ = crate::runtime::storage_flush::flush_mapping_for_guest_read(
                                state, host, mapping_id,
                            );
                            // Eager page-table + device-desc geometry when KVA works.
                            let _ = crate::runtime::mapper::resolve_mapping_backing(
                                state, host, mapping_id,
                            );
                        } else {
                            // Mismatched capture — put back for a later entry.
                            state.mapper_capture = Some(c);
                        }
                    }
                }
                MAPPER_REQUEST_UNMAP => {
                    // Deferred-writeback: DROP, never write — same recycled-page
                    // hazard as DeleteIOSurfaceBacking2 (the unmap request
                    // trails the guest release; writing risks PTE corruption).
                    crate::runtime::storage_flush::drop_windows(state, mapping_id, "mapper_unmap");
                    if let Some(c) = cap {
                        if c.request_type == MAPPER_REQUEST_UNMAP {
                            let _ = crate::runtime::mapper::apply_capture(state, &c, mapping_id);
                        } else {
                            state.mapper_capture = Some(c);
                            let _ = state.unmap_surface(mapping_id);
                        }
                    } else {
                        let _ = state.unmap_surface(mapping_id);
                    }
                }
                _ => {
                    if let Some(c) = cap {
                        state.mapper_capture = Some(c);
                    }
                    // Unknown mapper request: fail-visible, still advance. A
                    // mapper ring entry is not a FIFO packet — it carries no
                    // stamps and no payload span — so the two packet-shaped
                    // fields report the absence rather than borrowing the
                    // entry's bytes and implying a framing it does not have.
                    state.record_fail(FailEvent::UnknownChildOpcode {
                        channel: 0,
                        opcode: rtype as u16,
                        total_size: MAPPER_REQUEST_ENTRY_LEN as u32,
                        stamp_count: 0,
                        payload: Vec::new(),
                    });
                }
            }
            consumer = idx.wrapping_add(1);
        }
    } else {
        // No ring base: still catch consumer up (boot handshake).
        consumer = producer;
    }

    state.iosfc.consumer = consumer;
    if state.iosfc.consumer == state.iosfc.producer {
        host.enqueue(HostAction::irq_iosfc());
    }
    state.pending.iosfc = false;
}

/// Display-side present completion (PGDisplay `_presentMappedSurface`
/// completion block, live PVG binary RE in): after
/// `presentFrame` retains the surface into `+0x188`, the block sets pending
/// bit 1 on the display shared page, reads the enable mask, and pokes the
/// display IRQ when the guest asked for present notifications. This is the
/// guest's frame-done pacing edge — separate from the packet header stamp
/// (the swap fence). Without it the guest keeps swapping (fence releases)
/// but never receives the per-present display event.
pub fn signal_display_present_complete<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
) {
    let gpa = state.display.shared_gpa;
    if gpa == 0 {
        return;
    }
    let mut mask_le = [0u8; 4];
    if host
        .read_gpa(gpa + DISPLAY_SHARED_ENABLE_MASK, &mut mask_le)
        .is_err()
    {
        return;
    }
    let mask = ld32(&mask_le);
    // Pending word is atomic read-and-clear (ldclral) on the guest side; OR
    // the present bit so a not-yet-consumed ONLINE event is preserved.
    let mut pending_le = [0u8; 4];
    let pending = if host
        .read_gpa(gpa + DISPLAY_SHARED_PENDING, &mut pending_le)
        .is_ok()
    {
        ld32(&pending_le)
    } else {
        0
    };
    // A bit2 (ONLINE) still pending *after* online was acked is stale: the guest
    // already consumed that online event (`online_acked`), so re-delivering it
    // makes `signalDisplay` re-run process_online → connectionChange → a
    // boot-progress overlay rebuild (the host-driven strobe).
    // Preserving bit2 via the `pending |` write is only correct *pre-ack*; drop
    // it once acked so we don't hand the guest a redundant online. `stale` is 0
    // on healthy boots (bit2 clears at ack), so this is a no-op there — it only
    // suppresses the intermittent try_display_online/ack race leftover. A fresh
    // legitimate online (after a reinit) clears `online_acked` first, so it is
    // never masked here. Still logged (measure + fix together).
    let stale = state.display.online_acked && pending & DISPLAY_ONLINE_EVENT_MASK != 0;
    let base = if stale {
        pending & !DISPLAY_ONLINE_EVENT_MASK
    } else {
        pending
    };
    shared_w32(
        host,
        gpa,
        DISPLAY_SHARED_PENDING,
        base | DISPLAY_PRESENT_EVENT_MASK,
        state.page_size() as usize,
    );
    if stale {
        crate::runtime::census::present_proxy::note_stale_online_pending("present", pending);
    }
    if mask & DISPLAY_PRESENT_EVENT_MASK != 0 {
        let bit = 1u32 << (state.display.display_index & 0x1f);
        state
            .gfx
            .interrupt_status_disp
            .fetch_or(bit, std::sync::atomic::Ordering::AcqRel);
        host.enqueue(HostAction::irq_gfx());
    }
}

/// Minimum wall-clock interval shared by both display VBL signal paths, in
/// microseconds.
///
/// The x86 QEMU heartbeat oversamples this interval every
/// `REIMS_VGPU_PCI_HEARTBEAT_MS` (4 ms). The shared limiter caps heartbeat and
/// active-console polls at the rate we advertise, without aliasing a
/// heartbeat-only workload down to half rate.
///
/// **Derived from [`DISPLAY_REFRESH_HZ`], not written down.** This was a
/// millisecond grid with a hardcoded `8`, which is 125 Hz — so the device
/// advertised 120 Hz in its timing table and then delivered VBL 4.2% faster.
/// The guest honours what is delivered, not what is advertised: a driven
/// Safari measured its own `requestAnimationFrame` at exactly 125 Hz. 120 Hz is
/// 8333 µs and is simply not expressible on an integer-millisecond grid, so the
/// units are part of the fix rather than incidental to it.
pub(crate) const DISPLAY_VBL_MIN_INTERVAL_US: u64 =
    1_000_000 / crate::model::DISPLAY_REFRESH_HZ as u64;

/// Atomically claim the next display VBL for either the locked or lock-free
/// poll path. A single shared timestamp makes the cadence independent of device
/// lock contention and prevents both paths from signaling the same interval.
///
/// The claimed timestamp advances on a **fixed interval grid** (`last +
/// INTERVAL`), not to `now_ms`. Resetting to `now` lets poll jitter shift the
/// cadence phase permanently: a poll that lands slightly late pushes the *next*
/// deadline out another full interval, so the delivered VBL rate aliases down —
/// when the effective poll spacing sits in the danger zone (just under the
/// interval) it needs two polls per delivery and halves toward ~60 Hz. That is
/// the boot-to-boot 60-vs-120 split the user reports: on a boot where the poll
/// heartbeat jitters into that zone the guest latches 60 Hz. Advancing by exactly
/// one interval keeps delivery phase-locked to the grid and lets a late poll
/// "catch up" (each subsequent poll delivers until the grid is caught, then a
/// poll naturally skips and resyncs) so the *steady* rate converges to the grid
/// (~120 Hz) regardless of poll jitter, erring toward the ceiling the guest caps
/// at rather than latching 60. A long stall (≥2 intervals, e.g. the drain worker
/// held the lock) resyncs the phase to `now_ms` so we never unleash a burst of
/// back-dated VBLs.
pub(crate) fn claim_display_vbl(last_us: &std::sync::atomic::AtomicU64, now_us: u64) -> bool {
    let last = last_us.load(std::sync::atomic::Ordering::Acquire);
    let gap = now_us.saturating_sub(last);
    if gap < DISPLAY_VBL_MIN_INTERVAL_US {
        return false;
    }
    let next = if gap >= 2 * DISPLAY_VBL_MIN_INTERVAL_US {
        now_us
    } else {
        last + DISPLAY_VBL_MIN_INTERVAL_US
    };
    last_us
        .compare_exchange(
            last,
            next,
            std::sync::atomic::Ordering::AcqRel,
            std::sync::atomic::Ordering::Acquire,
        )
        .is_ok()
}

/// Pulse VBL at the phase-locked ~120 Hz cadence (grid interval
/// `DISPLAY_VBL_MIN_INTERVAL_MS`; see `claim_display_vbl`).
///
/// Writes pending bit 0, sets 0x1014 display bit, and raises MSI after ONLINE
/// has been acked. The limiter is owned outside `DeviceState` so this locked
/// path and `vbl_contended_pulse` use one time base. Without VBL the guest
/// compositor can stick on clear-only DisplaySwap of empty flip buffers.
pub fn signal_display_vbl<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    last_us: &std::sync::atomic::AtomicU64,
) {
    signal_display_vbl_at(state, host, last_us, crate::observe::elapsed_us());
}

/// Delivered-VBL rate, reported from the branch that decides it.
///
/// VBL is what paces the guest's compositor: WindowServer produces a frame off
/// its display-link callback, so whatever rate we deliver here is a ceiling on
/// guest frame rate no matter how fast the present path runs. Nothing measured
/// it. A driven boot emitted **zero** lines matching `vbl` anywhere in the
/// always-on channel, so "are we starving the display link" could not be
/// answered from a log, only guessed at from the constants.
///
/// The three arms are counted separately because a single "delivered" tally
/// cannot tell the two silences apart, and they have opposite meanings:
/// `not_online` is the display never having come up (no VBL is owed at all),
/// while `not_claimed` is the limiter doing its job at the advertised rate.
/// Reading a low delivered count without them would license both conclusions.
///
/// One line per 1024 deliveries — about 8 s at the grid rate, and it costs three
/// relaxed increments per poll otherwise.
/// Which way the VBL path went. Indices into [`VblCensus`].
pub(crate) const VBL_NOT_ONLINE: usize = 0;
pub(crate) const VBL_NOT_CLAIMED: usize = 1;
pub(crate) const VBL_DELIVERED: usize = 2;

/// One report per this many deliveries — about 8 s at the grid rate.
const VBL_REPORT_EVERY: u64 = 1024;

#[derive(Default)]
pub(crate) struct VblCensus {
    arms: [std::sync::atomic::AtomicU64; 3],
    last_report_ms: std::sync::atomic::AtomicU64,
    last_report_n: std::sync::atomic::AtomicU64,
}

impl VblCensus {
    /// Count one traversal and return the line to emit when a report is due.
    ///
    /// Returns the line rather than emitting it so the reporting rule is
    /// testable without a log sink: the interesting properties are "only
    /// deliveries report", "the rate is measured over the window and not the
    /// process lifetime", and "the two silent arms stay separable", and all
    /// three are assertions about this return value.
    pub(crate) fn note(&self, arm: usize, now_ms: u64) -> Option<String> {
        use std::sync::atomic::Ordering::Relaxed;
        let n = self.arms[arm].fetch_add(1, Relaxed) + 1;
        if arm != VBL_DELIVERED || !n.is_multiple_of(VBL_REPORT_EVERY) {
            return None;
        }
        let since_ms = now_ms.saturating_sub(self.last_report_ms.swap(now_ms, Relaxed));
        let since_n = n.saturating_sub(self.last_report_n.swap(n, Relaxed));
        // Window rate, not a lifetime average: the lifetime figure carries the
        // pre-online stretch forever and would read low long after the display
        // came up.
        let hz = if since_ms > 0 {
            (since_n * 1000) as f64 / since_ms as f64
        } else {
            0.0
        };
        Some(format!(
            "display_vbl delivered={n} not_claimed={} not_online={} window_hz={hz:.1} \
             grid_hz={:.1}",
            self.arms[VBL_NOT_CLAIMED].load(Relaxed),
            self.arms[VBL_NOT_ONLINE].load(Relaxed),
            1_000_000.0 / DISPLAY_VBL_MIN_INTERVAL_US as f64,
        ))
    }
}

pub(crate) fn note_vbl(arm: usize, now_ms: u64) {
    static VBL: std::sync::LazyLock<VblCensus> = std::sync::LazyLock::new(VblCensus::default);
    if let Some(line) = VBL.note(arm, now_ms) {
        crate::observe::off(line);
    }
}

/// Report at most this often. One line per second is bounded enough to leave on
/// for the life of the device and dense enough to see a stall move.
const DRAIN_DUTY_REPORT_MS: u64 = 1000;

/// Where the drain worker's wall clock goes.
///
/// The worker is the device's only executor: `device_drain` holds the device
/// lock for a whole tranche, so every guest FIFO packet, every GPU encode and
/// the host-window export are serialised behind it, and the guest's composite
/// rate cannot exceed the rate at which this thread finishes tranches.
///
/// Nothing else measures that. `sync_exec_lock_hold` is a per-packet threshold
/// line that only fires above `SYNC_EXEC_STALL_US`, so a worker pinned at 100%
/// by a steady stream of 200 ms tranches is completely silent — which is the
/// "an event count is not a state" trap, applied to a cost. This reads the
/// state: what fraction of wall clock the worker spends holding the lock, split
/// by the two phases that can own it.
///
/// The split is the point. `drain_us` is guest work (FIFO decode, draws, compute,
/// guest writeback); `publish_us` is our host-window export, which quiesces the
/// whole GPU twice per present. A duty near 1 says the ~2 Hz composite rate is
/// ours and names which half to attack; a duty near 0 says the worker is idle
/// and the guest is blocked on something upstream of us. No other line separates
/// those two readings.
///
/// `skipped` counts tranches that returned before taking the lock at all
/// (`present_action_pending`): a worker that keeps bailing looks identical to an
/// idle one in the duty figure alone, and it is not the same fault.
/// Which phase of guest work a slice of `drain_us` belongs to.
///
/// These are attributions inside `drain_us`, not a partition of it: a flush
/// reached from inside a draw is counted by both. That is deliberate and it is
/// self-checking — if the three sum to more than `drain_us` the phases nest, and
/// if they sum to much less the time is somewhere none of them names. Either
/// reading is useful and a single fused figure gives neither.
#[derive(Clone, Copy)]
pub enum DrainPhase {
    /// `encode_draw_chain`: metal2vulkan translate, encode, submit, readback.
    Draw,
    /// One compute record applied: bind bookkeeping for most kinds, encode +
    /// execute for a dispatch. Timed as a whole because "the binds are the cost"
    /// is exactly as interesting an answer as "the dispatch is".
    Compute,
    /// Deferred window flush: resident readback + guest writeback.
    Flush(FlushRail),
}

/// Which deferred-writeback rail a [`DrainPhase::Flush`] was spent on.
///
/// The aggregate `flush_us` is three quarters of the drain worker's wall clock
/// on a driven boot and had no owner, so every fix aimed at it was aimed by
/// guess. It is not one mechanism: four independent rails report as `Flush`, and
/// their counts are nowhere near proportional. One measured second read
/// `flushes=103` beside `surface_flush=15`, so the render rail the cost had been
/// attributed to is under a sixth of the *count* — and nothing said whose
/// microseconds those were.
///
/// Count and cost answer different questions here. A rail that flushes 71 times
/// at 50 µs and a rail that flushes 15 times at 7 ms are indistinguishable in
/// `flushes` and are opposite problems, so both are reported per rail.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FlushRail {
    /// `flush_render_one`: pinned resident readback, then guest writeback.
    Render,
    /// `flush_gva_one`: deferred GVA-addressed surface writeback.
    Gva,
    /// `flush_linear_one`: deferred linear-texture writeback.
    Linear,
    /// `flush_storage_one`: deferred compute-storage writeback.
    Storage,
}

/// The inside of a [`FlushRail::Render`] flush, which is 100% of the drain
/// worker's flush cost on a driven boot and is four very different things.
///
/// Live: `render_us=688003 render=100`, i.e. 6.9 ms per flush, ~69% of the
/// worker's entire second, with the other three rails at zero. Knowing that is
/// not yet enough to fix it, because the four parts below have opposite fixes.
/// A cost in [`Fence`](Self::Fence) is a GPU round trip and shrinking the copy
/// would not touch it; a cost in [`Map`](Self::Map) or [`Write`](Self::Write)
/// is bytes and a dirty rect would. Guessing between them is how the last
/// attempt picked its target.
///
/// Splitting it paid, and the record of what it bought belongs here rather than
/// only in a commit body. [`Write`](Self::Write) turned out to be the largest
/// phase and to be three whole-frame passes sharing one counter — see
/// [`SurfaceWritePhase`], which divides it again. Removing the two that were not
/// the guest's bytes took the render flush from **7.98 ms to 3.95 ms**, with the
/// drain worker's duty falling from 0.915 to ~0.72 and its worst tranche from
/// 46.5 ms to 18.5 ms. Those are device-side numbers and they reproduce: this
/// rail now measures 3.86 ms per flush.
///
/// **A Safari `requestAnimationFrame` figure was also attributed to that change
/// — "59.1 fps to 119.2 fps" — and that attribution does not hold.** rAF on this
/// pathway is bimodal at ~59 and ~118 with nothing in between, and *both* states
/// occur on one build within one boot: probing the same unchanged binary four
/// times in six minutes read 59.5, 117.3, 119.0 and 120.0, the low one being the
/// first probe after login. A single rAF number therefore cannot attribute
/// anything to a code change, in either direction — it nearly caused this rail's
/// BGRA8 upload change to be reverted as a regression when re-probing the same
/// build returned 117.3. Pair rAF with the device-side counters above, and see
/// `AGENTS.md` for the probe rule.
///
/// The phase left holding the flush is [`Fence`](Self::Fence) at ~45%, with
/// [`Write`](Self::Write) ~28% and [`Map`](Self::Map) ~22% — 94% of `flush_us`
/// accounted. What the rail moves is the headline: 116 flushes a second, each a
/// whole 1920x1080 frame, is **962 MB/s** read back from the GPU and landed in
/// guest pages, for ~62 presented frames. Every phase here is proportional to
/// that volume, so the next lever is reading back less than the whole
/// attachment, not making any one phase faster.
///
/// The obvious form of that lever does not pay, and the number is recorded here
/// so it is not re-derived. The guest already supplies a damage rect —
/// `OPCODE_SET_SCISSOR`, decoded verbatim into `req.scissor` — so a writeback could
/// land only the scissored region. A 30 s driven Safari probe on the
/// x86/PCI/Vulkan pathway bucketed every window-arming Store by the fraction of
/// its attachment the scissor covered: **99.34% of the texels a Store arms are
/// texels it covers**. Half the Stores carry no scissor at all and the other
/// half carry one spanning the whole attachment; the small ones were 0.8% of the
/// population and 0.66% of the area. The 35% of *all* draws that are scissored
/// are the small draws *inside* a pass — an icon, a glyph run, a window's own
/// layer — while the Store that ends a full-screen composite declares the full
/// screen. Reading back less has to find its evidence somewhere other than the
/// guest's scissor.
///
/// This paragraph used to end "[`Fence`](Self::Fence) is the GPU rendering the
/// frame rather than latency to reschedule — that is measured, not assumed",
/// and the measurement it pointed at does not say that. What
/// [`ResidentArmCensus`] measured is that submitting the copy *earlier* cannot
/// help, which rules out one explanation without establishing another. Holding
/// the host GPU at its top clock then moved the same wait from 2.55-2.83 ms to
/// **0.40 ms** with no code change, so roughly six sevenths of it is the
/// governor and only the last seventh is work. Read [`ResidentArmCensus`] for
/// the table, and record the host GPU's power state beside any number taken
/// from this phase.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ReadbackPhase {
    /// Record the copy command buffer and submit it. No GPU wait.
    Submit,
    /// Block on the readback fence: pure GPU round-trip latency, and the part
    /// no smaller copy can reduce.
    Fence,
    /// Make the staging buffer readable. On the leased arm that is the
    /// invalidate alone, because the mapping already exists for the slot's
    /// lifetime; on the fallback arm it is map, invalidate and a whole-frame
    /// memcpy into a host `Vec`. The two differ by ~8 MB, so this phase reads
    /// near zero exactly when every readback in the window was leased and
    /// climbs in proportion to the ones that were not.
    Map,
    /// Write the frame into the guest's pages (`write_bgra8_skipping`).
    Write,
}

impl ReadbackPhase {
    const ALL: [ReadbackPhase; 4] = [
        ReadbackPhase::Submit,
        ReadbackPhase::Fence,
        ReadbackPhase::Map,
        ReadbackPhase::Write,
    ];

    const fn index(self) -> usize {
        match self {
            ReadbackPhase::Submit => 0,
            ReadbackPhase::Fence => 1,
            ReadbackPhase::Map => 2,
            ReadbackPhase::Write => 3,
        }
    }

    const fn label(self) -> &'static str {
        match self {
            ReadbackPhase::Submit => "submit",
            ReadbackPhase::Fence => "fence",
            ReadbackPhase::Map => "map",
            ReadbackPhase::Write => "write",
        }
    }
}

impl FlushRail {
    const ALL: [FlushRail; 4] = [
        FlushRail::Render,
        FlushRail::Gva,
        FlushRail::Linear,
        FlushRail::Storage,
    ];

    const fn index(self) -> usize {
        match self {
            FlushRail::Render => 0,
            FlushRail::Gva => 1,
            FlushRail::Linear => 2,
            FlushRail::Storage => 3,
        }
    }

    const fn label(self) -> &'static str {
        match self {
            FlushRail::Render => "render",
            FlushRail::Gva => "gva",
            FlushRail::Linear => "linear",
            FlushRail::Storage => "storage",
        }
    }
}

/// Why a drain tranche did or did not hand the host window a new frame.
///
/// With the swapchain fixed, `host_window_cadence` reads `presents == offered`
/// with `busy_acquire=0` — the window shows every frame it is offered and drops
/// none. So the remaining deficit is entirely in the offer rate, which was 58/s
/// on a host panel at 120 Hz while the drain worker completed 110–132 render
/// flushes a second and the guest sustained ~117 fps. Something between the
/// composite and the window is halving it, and `publish_window_frame` has four
/// separate ways to return without publishing that the cadence census cannot see
/// from the other side.
///
/// The one that matters is [`SameKey`](Self::SameKey) against
/// [`Fresh`](Self::Fresh). A large `same_key` means the guest is presenting at
/// the offer rate and the window is being given everything there is — the
/// deficit would then be the guest's own present cadence, not ours. `fresh` near
/// the tranche rate with `same_key` small would mean the opposite. Those have
/// nothing in common as fixes, which is why this is measured before either is
/// attempted.
///
/// `fresh` counts a new key **reaching** the publish, not a frame landing in the
/// window's slot: the four ways the publish itself can still fail after that
/// point already have their own census in
/// [`crate::runtime::census::present_proxy::window_publish`], and duplicating
/// them here would give two counters that could disagree.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WindowPublish {
    /// A frame key not yet published reached the publish.
    Fresh,
    /// No window is attached to consume a frame.
    NoWindow,
    /// The device holds no valid captured frame yet.
    NoFrame,
    /// The captured frame is the one already published — same mapping,
    /// generation and present epoch.
    SameKey,
}

impl WindowPublish {
    const ALL: [WindowPublish; 4] = [
        WindowPublish::Fresh,
        WindowPublish::NoWindow,
        WindowPublish::NoFrame,
        WindowPublish::SameKey,
    ];

    const fn index(self) -> usize {
        match self {
            WindowPublish::Fresh => 0,
            WindowPublish::NoWindow => 1,
            WindowPublish::NoFrame => 2,
            WindowPublish::SameKey => 3,
        }
    }

    const fn label(self) -> &'static str {
        match self {
            WindowPublish::Fresh => "fresh",
            WindowPublish::NoWindow => "no_window",
            WindowPublish::NoFrame => "no_frame",
            WindowPublish::SameKey => "same_key",
        }
    }
}

#[derive(Default)]
pub(crate) struct WindowPublishCensus {
    arms: [std::sync::atomic::AtomicU64; 4],
}

impl WindowPublishCensus {
    fn note(&self, arm: WindowPublish) {
        self.arms[arm.index()].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    fn take(&self, win_ms: u64) -> Option<String> {
        use std::sync::atomic::Ordering::Relaxed;
        let counts: Vec<u64> = WindowPublish::ALL
            .iter()
            .map(|arm| self.arms[arm.index()].swap(0, Relaxed))
            .collect();
        if counts.iter().all(|&n| n == 0) {
            return None;
        }
        let body: String = WindowPublish::ALL
            .iter()
            .zip(&counts)
            .map(|(arm, n)| format!(" {}={n}", arm.label()))
            .collect();
        Some(format!("window_publish win_ms={win_ms}{body}"))
    }
}

/// The inside of [`ReadbackPhase::Write`], which is now the largest phase of the
/// largest rail and is three full-frame passes wearing one name.
///
/// `write_us=377356 write=95` is 3.97 ms per flush and 40% of the drain worker's
/// busy second, on an 8.29 MB frame — an effective 2.1 GB/s against ~9 GB/s for
/// the readback's own memcpy of the identical bytes. A previous attempt read
/// that gap as "cache-cold scattered writes into guest RAM, so only fewer bytes
/// help", removed a staging hop on that basis and measured no change.
///
/// The gap is a factor of four, which is the shape of doing the work four times,
/// not of doing it once badly. `write_bgra8_skipping` makes up to three
/// whole-frame passes and the name covers all of them, so none of them can be
/// ruled in or out:
///
/// - [`Stage`](Self::Stage) — the fragmented path's `frame` buffer: an 8 MB
///   allocation plus every row copied into it, before a single guest byte moves.
///   The contiguous path skips this entirely, so which path the composite takes
///   decides whether it exists at all.
/// - [`Land`](Self::Land) — the bytes actually reaching guest pages. This is the
///   only pass the guest needs and the only one a dirty rect would shrink.
/// - [`Cache`](Self::Cache) — a second 8 MB allocation holding a host-side
///   duplicate of the same frame for [`crate::runtime::surface_cache`], built
///   unconditionally on every non-skipping write.
///
/// Two of the three are freshly allocated multi-megabyte buffers per flush, ~95
/// times a second. A `vec![0u8; 8_290_000]` is not free even zeroed by the
/// allocator: the pages come back untouched and the fill faults every one of
/// them in. Whether that is the missing factor is exactly what this measures.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SurfaceWritePhase {
    /// Build the staged whole-frame buffer (fragmented path only).
    Stage,
    /// Move the bytes into the guest's pages.
    Land,
    /// Build and store the host-side [`crate::runtime::surface_cache`] copy.
    Cache,
}

impl SurfaceWritePhase {
    const ALL: [SurfaceWritePhase; 3] = [
        SurfaceWritePhase::Stage,
        SurfaceWritePhase::Land,
        SurfaceWritePhase::Cache,
    ];

    const fn index(self) -> usize {
        match self {
            SurfaceWritePhase::Stage => 0,
            SurfaceWritePhase::Land => 1,
            SurfaceWritePhase::Cache => 2,
        }
    }

    const fn label(self) -> &'static str {
        match self {
            SurfaceWritePhase::Stage => "stage",
            SurfaceWritePhase::Land => "land",
            SurfaceWritePhase::Cache => "cache",
        }
    }
}

/// [`SurfaceWritePhase`] totals over the census window, plus which of the two
/// landing paths the writes took.
///
/// `contig` and `frag` are counted because the split is not readable from the
/// phase totals alone: a `stage_us` of zero means the contiguous path, and a
/// reader with no path count cannot tell that from "the staging is free".
#[derive(Default)]
pub(crate) struct SurfaceWriteCensus {
    us: [std::sync::atomic::AtomicU64; 3],
    count: [std::sync::atomic::AtomicU64; 3],
    max_us: [std::sync::atomic::AtomicU64; 3],
    contig: std::sync::atomic::AtomicU64,
    frag: std::sync::atomic::AtomicU64,
    bytes: std::sync::atomic::AtomicU64,
}

impl SurfaceWriteCensus {
    fn note(&self, phase: SurfaceWritePhase, us: u64) {
        use std::sync::atomic::Ordering::Relaxed;
        let i = phase.index();
        self.us[i].fetch_add(us, Relaxed);
        self.count[i].fetch_add(1, Relaxed);
        self.max_us[i].fetch_max(us, Relaxed);
    }

    fn note_path(&self, contiguous: bool, bytes: u64) {
        use std::sync::atomic::Ordering::Relaxed;
        if contiguous {
            self.contig.fetch_add(1, Relaxed);
        } else {
            self.frag.fetch_add(1, Relaxed);
        }
        self.bytes.fetch_add(bytes, Relaxed);
    }

    fn take(&self, win_ms: u64) -> Option<String> {
        use std::sync::atomic::Ordering::Relaxed;
        let contig = self.contig.swap(0, Relaxed);
        let frag = self.frag.swap(0, Relaxed);
        if contig == 0 && frag == 0 {
            return None;
        }
        let bytes = self.bytes.swap(0, Relaxed);
        let mut body = String::new();
        for phase in SurfaceWritePhase::ALL {
            let i = phase.index();
            let us = self.us[i].swap(0, Relaxed);
            let n = self.count[i].swap(0, Relaxed);
            let max = self.max_us[i].swap(0, Relaxed);
            let label = phase.label();
            body.push_str(&format!(
                " {label}_us={us} {label}={n} {label}_max_us={max}"
            ));
        }
        Some(format!(
            "write_split win_ms={win_ms} contig={contig} frag={frag} bytes={bytes}{body}"
        ))
    }
}

/// How long a resident-backed render window sits armed before its flush reads
/// it, which is the only interval the readback's GPU round trip could hide in.
///
/// [`ReadbackPhase::Fence`] is 46% of the render rail and is paid at the flush,
/// because that is where the copy is submitted. Submitting it at the arm instead
/// — the guest's Store, where the window is created — would only shorten the
/// wait by however much wall clock separates the two: if the flush follows the
/// arm by less than the round trip, the fence still blocks and the move buys
/// nothing but complexity in the path that publishes composited pixels.
///
/// So this measures that separation before anything is built on it, rather than
/// assuming the GPU has idle time in between. **It refuted the proposal.** On a
/// driven boot: `arms=95 flushes=95 aged=95 age_us=33341 max_age_us=372`, i.e.
/// a mean arm→flush interval of **351 µs** with a **372 µs** worst case, beside
/// `fence_us=248748 fence=95` — a **2.6 ms** mean fence wait in the same second.
/// The interval is seven times shorter than the wait it would have to hide, and
/// the tight max says that is the whole distribution and not a mean concealing
/// a long tail. Submitting at the arm would leave ~2.2 ms of the 2.6 ms still to
/// wait, for a deferred readback slot and a second fence lifetime in the path
/// that publishes composited pixels.
///
/// What that also settles is *what the fence wait is not*. It is not scheduling
/// latency with slack to reclaim: the arm and the flush are 351 µs apart inside
/// one tranche, so the draws that produce the composite are still executing when
/// the copy is submitted, and the copy queues behind them however early it is
/// sent. Submitting earlier cannot help.
///
/// This paragraph used to continue "the 2.6 ms is the GPU rendering the frame.
/// Only cheaper draws can move it", and that inference does not follow from the
/// premise. Waiting on the GPU says the GPU is slow; it does not say the work is
/// large. **Measured, and it is not the work.** Same boot, same build, same
/// driven probe, with the only difference a synthetic load holding the host GPU
/// at its top clock instead of letting it choose:
///
/// | | host GPU at its own clock (P5, 800-1450 MHz) | held at P0, 2820 MHz |
/// |---|---|---|
/// | `fence_us`/`fence` | 2.55 - 2.83 ms | **0.40 ms** |
/// | total fence time per second | 265 - 341 ms | **35 ms** |
/// | `flush_us`/`flushes` | 4.0 ms | **0.83 - 1.75 ms** |
/// | Safari rAF long frames | 7 (0.39 %) | **0** |
/// | Safari rAF worst frame | 42 ms | **21 ms** |
///
/// So roughly six sevenths of the wait was the host GPU running at a third of
/// its clock or less, and the device's actual GPU cost per composited frame is
/// about **0.40 ms**. The governor is not misbehaving: this workload submits
/// ~0.4 ms of work per frame and then blocks, which reads as a few per cent
/// occupancy, and a few per cent occupancy is what a low clock is for.
///
/// Two consequences, and the second is the one that changes what to build:
///
/// - Any measurement of GPU-side latency here must record the host GPU's clock
///   and power state beside it, or it is a measurement of the governor. A number
///   taken at P5 against one taken at P0 is a 6x artefact with no code in it.
/// - The second consequence used to read "**this device is latency-bound on a
///   GPU that is usually downclocked, not throughput-bound**", and concluded
///   that removing a whole GPU round trip is worth about six times the flat GPU
///   cost while removing bytes is worth what it always was. **That does not
///   follow from the table above, and it is now measured false.** The premise is
///   that the wait shrinks with clock — and a copy moving 8 MB shrinks with
///   clock just as much as a latency does. The reading could not tell them
///   apart, and it picked one.
///
/// `readback_split`'s `bar_us` and `gpu_us` are the device's own timestamps
/// either side of the copy, and they settle it. Driven one-second windows on an
/// x86/PCI boot with the host GPU at P5:
///
/// ```text
/// fence 2.549 ms   copy 2.286 ms (89.7%)   draw-wait 0.0010 ms   ask 0.262 ms
/// fence 1.906 ms   copy 1.710 ms (89.7%)   draw-wait 0.0010 ms   ask 0.195 ms
/// fence 1.474 ms   copy 1.296 ms (87.9%)   draw-wait 0.0010 ms   ask 0.177 ms
/// ```
///
/// **87-91% of the fence wait is the copy executing.** 0.05% is the draw batch
/// it waits on — so the composite render is effectively free and the readback is
/// the device's whole GPU cost — and ~0.19 ms is the cost of asking. The copy
/// moves 8.29 MB at 3.6-6.4 GB/s in that power state. Two things follow, and
/// both are the opposite of what the old wording argued:
///
/// - **Removing bytes is worth ~1:1 against 90% of the largest cost in the
///   device.** The four levers the deferred-flush ledger in
///   [`crate::runtime::storage_flush::flush_mapping_windows_before_fence`] prices
///   in bytes are the ones that would pay, and they were not being weighed
///   against the right number.
/// - **Removing the second submission is worth the other ~11%** — a stable
///   0.18-0.26 ms per readback, and no more. That prices a step left queued as a
///   top item on the grounds that "round trips *are* the cost". They are not.
///
/// `multi` is not noise to be averaged away. The age of "the arm" is a single
/// number only when exactly one window was armed since the last flush; a window
/// that drifted out through one of `flush_render_one`'s refusals never reaches
/// the flush site at all, so the count self-heals on the next arm rather than
/// sticking at a wrong live population forever.
#[derive(Default)]
pub(crate) struct ResidentArmCensus {
    /// Arms since the last flush read the counter. Reset to 0 on every read.
    arms_since_flush: std::sync::atomic::AtomicU64,
    /// [`crate::observe::elapsed_us`] at the most recent arm.
    last_arm_us: std::sync::atomic::AtomicU64,
    arms: std::sync::atomic::AtomicU64,
    flushes: std::sync::atomic::AtomicU64,
    aged: std::sync::atomic::AtomicU64,
    age_us: std::sync::atomic::AtomicU64,
    max_age_us: std::sync::atomic::AtomicU64,
    /// Flushes reached with a count other than exactly one arm outstanding.
    multi: std::sync::atomic::AtomicU64,
}

impl ResidentArmCensus {
    fn note_arm(&self, now_us: u64) {
        use std::sync::atomic::Ordering::Relaxed;
        self.arms.fetch_add(1, Relaxed);
        self.arms_since_flush.fetch_add(1, Relaxed);
        self.last_arm_us.store(now_us, Relaxed);
    }

    fn note_flush(&self, now_us: u64) {
        use std::sync::atomic::Ordering::Relaxed;
        self.flushes.fetch_add(1, Relaxed);
        if self.arms_since_flush.swap(0, Relaxed) != 1 {
            self.multi.fetch_add(1, Relaxed);
            return;
        }
        let age = now_us.saturating_sub(self.last_arm_us.load(Relaxed));
        self.aged.fetch_add(1, Relaxed);
        self.age_us.fetch_add(age, Relaxed);
        self.max_age_us.fetch_max(age, Relaxed);
    }

    /// The line for the window that just closed, or `None` when no resident
    /// window was armed or flushed in it.
    fn take(&self, win_ms: u64) -> Option<String> {
        use std::sync::atomic::Ordering::Relaxed;
        let arms = self.arms.swap(0, Relaxed);
        let flushes = self.flushes.swap(0, Relaxed);
        if arms == 0 && flushes == 0 {
            return None;
        }
        let aged = self.aged.swap(0, Relaxed);
        let total = self.age_us.swap(0, Relaxed);
        let max = self.max_age_us.swap(0, Relaxed);
        let multi = self.multi.swap(0, Relaxed);
        Some(format!(
            "resident_arm_age win_ms={win_ms} arms={arms} flushes={flushes} aged={aged} \
             age_us={total} max_age_us={max} multi={multi}"
        ))
    }
}

#[derive(Default)]
pub(crate) struct DrainDutyCensus {
    tranches: std::sync::atomic::AtomicU64,
    skipped: std::sync::atomic::AtomicU64,
    drain_us: std::sync::atomic::AtomicU64,
    publish_us: std::sync::atomic::AtomicU64,
    draw_us: std::sync::atomic::AtomicU64,
    draws: std::sync::atomic::AtomicU64,
    compute_us: std::sync::atomic::AtomicU64,
    computes: std::sync::atomic::AtomicU64,
    flush_us: std::sync::atomic::AtomicU64,
    flushes: std::sync::atomic::AtomicU64,
    max_tranche_us: std::sync::atomic::AtomicU64,
    /// Longest single Flush in the window. `flush_us/flushes` is a mean, and a
    /// mean cannot tell "every flush costs 7.7 ms" from "most are free and one
    /// blocked 30 ms" — which are different defects with different fixes.
    max_flush_us: std::sync::atomic::AtomicU64,
    /// `flush_us`, `flushes` and `max_flush_us` again, split by [`FlushRail`]
    /// and indexed by [`FlushRail::index`].
    rail_us: [std::sync::atomic::AtomicU64; 4],
    rail_count: [std::sync::atomic::AtomicU64; 4],
    rail_max_us: [std::sync::atomic::AtomicU64; 4],
    /// The inside of the render rail, indexed by [`ReadbackPhase::index`].
    rb_us: [std::sync::atomic::AtomicU64; 4],
    rb_count: [std::sync::atomic::AtomicU64; 4],
    rb_max_us: [std::sync::atomic::AtomicU64; 4],
    /// GPU-side execution of the readback command buffer, from the device's own
    /// timestamp queries, split at the barrier. `rb_bar_us` is the copy command
    /// buffer waiting for the draw batch ahead of it to finish; `rb_gpu_us` is
    /// the copy itself. Together they divide [`ReadbackPhase::Fence`], which is
    /// CPU wall clock and cannot tell either from the cost of asking.
    rb_bar_us: std::sync::atomic::AtomicU64,
    rb_gpu_us: std::sync::atomic::AtomicU64,
    rb_gpu_count: std::sync::atomic::AtomicU64,
    rb_gpu_max_us: std::sync::atomic::AtomicU64,
    /// The window length `note` last reported, so `take_flush_rails` states the
    /// same denominator instead of deriving a second one.
    last_win_ms: std::sync::atomic::AtomicU64,
    /// Tranches that held the device lock for at least one whole guest frame.
    /// `max_tranche_us` is a max with no count, so it cannot distinguish one
    /// 38 ms tranche from three 20 ms ones; this is that count.
    slow_tranches: std::sync::atomic::AtomicU64,
    last_report_ms: std::sync::atomic::AtomicU64,
}

/// A tranche at or above this held the device lock for a whole guest frame.
///
/// Derived from the VBL cadence we actually deliver, because that *is* the
/// budget: the vCPU blocks on the same mutex this tranche holds, so a tranche
/// longer than one frame interval is one the guest cannot have serviced in
/// time. Deriving it also means it tracks the refresh rate instead of becoming
/// a stale constant beside it.
const DRAIN_TRANCHE_SLOW_US: u64 = DISPLAY_VBL_MIN_INTERVAL_US;

impl DrainDutyCensus {
    /// Count one skipped tranche (lock never taken).
    pub(crate) fn note_skipped(&self) {
        self.skipped
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Attribute `us` of the current tranche's `drain_us` to one phase.
    pub(crate) fn note_phase(&self, phase: DrainPhase, us: u64) {
        use std::sync::atomic::Ordering::Relaxed;
        let (total, count) = match phase {
            DrainPhase::Draw => (&self.draw_us, &self.draws),
            DrainPhase::Compute => (&self.compute_us, &self.computes),
            DrainPhase::Flush(_) => (&self.flush_us, &self.flushes),
        };
        total.fetch_add(us, Relaxed);
        count.fetch_add(1, Relaxed);
        if let DrainPhase::Flush(rail) = phase {
            self.max_flush_us.fetch_max(us, Relaxed);
            let i = rail.index();
            self.rail_us[i].fetch_add(us, Relaxed);
            self.rail_count[i].fetch_add(1, Relaxed);
            self.rail_max_us[i].fetch_max(us, Relaxed);
        }
    }

    /// The per-rail split of the window `drain_duty` just reported, or `None`
    /// when nothing flushed in it.
    ///
    /// A separate line rather than twelve more columns on `drain_duty`, and
    /// driven by that line's emitter rather than a cadence of its own, so the
    /// two divide against each other: the rails must sum to `flush_us` and their
    /// counts to `flushes`. Valid only immediately after `note` returns `Some`,
    /// which is the only place it is called.
    pub(crate) fn take_flush_rails(&self) -> Option<String> {
        use std::sync::atomic::Ordering::Relaxed;
        let win_ms = self.last_win_ms.load(Relaxed);
        let mut body = String::new();
        let mut any = false;
        for rail in FlushRail::ALL {
            let i = rail.index();
            let us = self.rail_us[i].swap(0, Relaxed);
            let n = self.rail_count[i].swap(0, Relaxed);
            let max = self.rail_max_us[i].swap(0, Relaxed);
            any |= n != 0;
            let label = rail.label();
            body.push_str(&format!(
                " {label}_us={us} {label}={n} {label}_max_us={max}"
            ));
        }
        any.then(|| format!("flush_rails win_ms={win_ms}{body}"))
    }

    /// The inside of the render rail over the window `drain_duty` just
    /// reported, or `None` when nothing was read back in it.
    ///
    /// Sits under `flush_rails`'s `render_us` and divides it. Read `gpu_us` and
    /// `bar_us` before concluding anything from `fence_us`: they are the GPU's
    /// own timestamps taken from inside that wait, so `fence_us` owning the line
    /// means latency only when `gpu_us` is a small part of it. When `gpu_us`
    /// owns `fence_us` the wait is the readback command buffer copying, which is
    /// bytes and a smaller copy does touch it; `bar_us` is the draw batch queued
    /// ahead of it, and only that part is a scheduling cost rather than a size
    /// one. `map_us`/`write_us` are host-side bytes either way.
    pub(crate) fn take_readback_split(&self) -> Option<String> {
        use std::sync::atomic::Ordering::Relaxed;
        let win_ms = self.last_win_ms.load(Relaxed);
        let mut body = String::new();
        let mut any = false;
        for phase in ReadbackPhase::ALL {
            let i = phase.index();
            let us = self.rb_us[i].swap(0, Relaxed);
            let n = self.rb_count[i].swap(0, Relaxed);
            let max = self.rb_max_us[i].swap(0, Relaxed);
            any |= n != 0;
            let label = phase.label();
            body.push_str(&format!(
                " {label}_us={us} {label}={n} {label}_max_us={max}"
            ));
        }
        let bar_us = self.rb_bar_us.swap(0, Relaxed);
        let gpu_us = self.rb_gpu_us.swap(0, Relaxed);
        let gpu = self.rb_gpu_count.swap(0, Relaxed);
        let gpu_max_us = self.rb_gpu_max_us.swap(0, Relaxed);
        body.push_str(&format!(
            " bar_us={bar_us} gpu_us={gpu_us} gpu={gpu} gpu_max_us={gpu_max_us}"
        ));
        any.then(|| format!("readback_split win_ms={win_ms}{body}"))
    }

    /// Record one readback command buffer's two GPU-side spans: `barrier_us`
    /// waiting for the draws, then `copy_us` moving the frame.
    fn note_readback_gpu(&self, barrier_us: u64, copy_us: u64) {
        use std::sync::atomic::Ordering::Relaxed;
        self.rb_bar_us.fetch_add(barrier_us, Relaxed);
        self.rb_gpu_us.fetch_add(copy_us, Relaxed);
        self.rb_gpu_count.fetch_add(1, Relaxed);
        self.rb_gpu_max_us.fetch_max(copy_us, Relaxed);
    }

    /// The window length [`Self::note`] last reported over, so a census emitted
    /// beside `drain_duty` states the same denominator rather than deriving a
    /// second one from a clock that has moved since.
    pub(crate) fn last_window_ms(&self) -> u64 {
        self.last_win_ms.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub(crate) fn note_readback(&self, phase: ReadbackPhase, us: u64) {
        use std::sync::atomic::Ordering::Relaxed;
        let i = phase.index();
        self.rb_us[i].fetch_add(us, Relaxed);
        self.rb_count[i].fetch_add(1, Relaxed);
        self.rb_max_us[i].fetch_max(us, Relaxed);
    }

    /// Accumulate one completed tranche and return the line when a report is
    /// due. Returns the line rather than emitting it so the reporting rule is
    /// testable without a log sink: that the window resets on report (so the
    /// figure is a rate over the window, not a lifetime average), and that duty
    /// is busy time over elapsed time.
    pub(crate) fn note(&self, drain_us: u64, publish_us: u64, now_ms: u64) -> Option<String> {
        use std::sync::atomic::Ordering::Relaxed;
        self.tranches.fetch_add(1, Relaxed);
        self.drain_us.fetch_add(drain_us, Relaxed);
        self.publish_us.fetch_add(publish_us, Relaxed);
        let tranche_us = drain_us.saturating_add(publish_us);
        self.max_tranche_us.fetch_max(tranche_us, Relaxed);
        if tranche_us >= DRAIN_TRANCHE_SLOW_US {
            self.slow_tranches.fetch_add(1, Relaxed);
        }
        let last = self.last_report_ms.load(Relaxed);
        // First call arms the window; it does not report a duty against a zero
        // origin, which would divide the whole boot's idle time into one tranche.
        if last == 0 {
            self.last_report_ms.store(now_ms, Relaxed);
            return None;
        }
        let win_ms = now_ms.saturating_sub(last);
        if win_ms < DRAIN_DUTY_REPORT_MS {
            return None;
        }
        self.last_report_ms.store(now_ms, Relaxed);
        self.last_win_ms.store(win_ms, Relaxed);
        let tranches = self.tranches.swap(0, Relaxed);
        let skipped = self.skipped.swap(0, Relaxed);
        let drain = self.drain_us.swap(0, Relaxed);
        let publish = self.publish_us.swap(0, Relaxed);
        let max = self.max_tranche_us.swap(0, Relaxed);
        let draw = self.draw_us.swap(0, Relaxed);
        let draws = self.draws.swap(0, Relaxed);
        let compute = self.compute_us.swap(0, Relaxed);
        let computes = self.computes.swap(0, Relaxed);
        let flush = self.flush_us.swap(0, Relaxed);
        let flushes = self.flushes.swap(0, Relaxed);
        let max_flush = self.max_flush_us.swap(0, Relaxed);
        let slow = self.slow_tranches.swap(0, Relaxed);
        let busy = drain.saturating_add(publish);
        let duty = busy as f64 / (win_ms as f64 * 1000.0);
        Some(format!(
            "drain_duty win_ms={win_ms} tranches={tranches} skipped={skipped} busy_us={busy} \
             duty={duty:.3} drain_us={drain} publish_us={publish} max_tranche_us={max} \
             draw_us={draw} draws={draws} compute_us={compute} computes={computes} \
             flush_us={flush} flushes={flushes} max_flush_us={max_flush} \
             slow_tranches={slow}/{tranches} slow_us={DRAIN_TRANCHE_SLOW_US}"
        ))
    }
}

/// How long the vCPU thread waited for the device lock, measured where it waits.
///
/// Every other figure about tranche length is taken from the side that *holds*
/// the lock, so the step from "the drain held it 38 ms" to "the guest missed a
/// frame" was an inference. This measures the stall from the side that suffers
/// it: the guest's MMIO access is stopped for exactly this long, on the vCPU
/// thread, inside `device_iosfc_read`/`device_iosfc_write`.
///
/// Those two are reached only from `reims_vgpu_qemu_iosfc_read`/`_write`, which
/// only `reims-vgpu-mmio` calls: the PCI device exposes no IOSFC region, so on
/// x86 this census is silent because the path does not exist, not because the
/// guest was never stalled. x86's own mechanism is [`DoorbellCensus`].
///
/// Only the contended path is timed: the uncontended path takes `try_lock` and
/// costs an atomic increment, so a fast access pays nothing for the measurement
/// itself. It does still drive the report, once per [`UNCONTENDED_POLL`]
/// acquisitions — without that, a window with zero waits emits nothing and
/// silence means both "the guest was never blocked" and "no IOSFC traffic
/// arrived". Reading the second as the first is how an instrument talks someone
/// out of a real stall.
#[derive(Default)]
pub(crate) struct VcpuLockCensus {
    waits: std::sync::atomic::AtomicU64,
    wait_us: std::sync::atomic::AtomicU64,
    max_wait_us: std::sync::atomic::AtomicU64,
    /// Waits that cost the guest at least a whole frame interval.
    frame_waits: std::sync::atomic::AtomicU64,
    uncontended: std::sync::atomic::AtomicU64,
    last_report_ms: std::sync::atomic::AtomicU64,
}

/// One in this many uncontended acquisitions reads the clock.
///
/// The uncontended path is the guest's hot MMIO path — hundreds of thousands of
/// acquisitions a second on a driven boot — so it cannot afford an
/// `Instant::now()` each time. It still has to reach the report, or a window
/// with no waits at all stays silent and "the guest was never blocked" is
/// indistinguishable from "no IOSFC traffic reached this device". Those are
/// opposite conclusions and the whole point of the census is to tell them apart.
const UNCONTENDED_POLL: u64 = 1024;

impl VcpuLockCensus {
    /// Count one free acquisition, returning the line when a report is due.
    ///
    /// The clock is read once per [`UNCONTENDED_POLL`] acquisitions, which puts
    /// the report's granularity at that many MMIO accesses rather than at the
    /// exact second boundary. `win_ms` is measured, not assumed, so a window
    /// that closes late reports its true length.
    pub(crate) fn note_uncontended(&self, now_ms: impl FnOnce() -> u64) -> Option<String> {
        let prior = self
            .uncontended
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if !prior.is_multiple_of(UNCONTENDED_POLL) {
            return None;
        }
        self.maybe_report(now_ms())
    }

    /// Record one contended wait and return the line when a report is due.
    pub(crate) fn note_wait(&self, us: u64, now_ms: u64) -> Option<String> {
        use std::sync::atomic::Ordering::Relaxed;
        self.waits.fetch_add(1, Relaxed);
        self.wait_us.fetch_add(us, Relaxed);
        self.max_wait_us.fetch_max(us, Relaxed);
        if us >= DRAIN_TRANCHE_SLOW_US {
            self.frame_waits.fetch_add(1, Relaxed);
        }
        self.maybe_report(now_ms)
    }

    /// The window logic both paths share.
    fn maybe_report(&self, now_ms: u64) -> Option<String> {
        use std::sync::atomic::Ordering::Relaxed;
        let last = self.last_report_ms.load(Relaxed);
        if last == 0 {
            self.last_report_ms.store(now_ms, Relaxed);
            return None;
        }
        let win_ms = now_ms.saturating_sub(last);
        if win_ms < DRAIN_DUTY_REPORT_MS {
            return None;
        }
        self.last_report_ms.store(now_ms, Relaxed);
        let waits = self.waits.swap(0, Relaxed);
        let total = self.wait_us.swap(0, Relaxed);
        let max = self.max_wait_us.swap(0, Relaxed);
        let frames = self.frame_waits.swap(0, Relaxed);
        let free = self.uncontended.swap(0, Relaxed);
        Some(format!(
            "vcpu_lock_wait win_ms={win_ms} waits={waits} uncontended={free} \
             wait_us={total} max_wait_us={max} frame_waits={frames} slow_us={DRAIN_TRANCHE_SLOW_US}"
        ))
    }
}

/// How long a guest MMIO doorbell sat queued before the host applied it.
///
/// This is the *other* half of the stall, and on the PCI pathway it is the only
/// half there is. `reims-vgpu-pci` exposes no IOSFC region — only
/// `reims-vgpu-mmio` calls `reims_vgpu_qemu_iosfc_read`/`_write` — so
/// [`VcpuLockCensus`], which instruments `lock_device_for_vcpu`, measures a code
/// path x86 does not have and is silent there by construction rather than by
/// result. Reading that silence as "the drain never stalled the guest" is
/// exactly the mistake it was rebuilt to prevent, so the x86 mechanism gets its
/// own census.
///
/// x86's vCPU never blocks: `device_gfx_write` takes `inner` with `try_lock` and
/// on failure pushes to `gfx_ingress` and returns, so the guest's store retires
/// immediately. The write is then applied by `lock_for_drain`, which takes
/// `inner` with a **blocking** lock and therefore cannot run until the drain
/// worker's current tranche ends. The cost is not a stopped vCPU, it is a
/// doorbell that the guest believes was accepted and whose work does not start
/// for up to a whole tranche — measured at `max_tranche_us` up to 43 ms while
/// `drain_duty` sat at 0.92.
///
/// `direct` counts writes that found the lock free and skipped the queue, and it
/// is load-bearing for the same reason `uncontended` is next door: `queued=0`
/// with a large `direct` is a working doorbell path, while both at zero is no
/// traffic at all.
///
/// # The delay is the tranche, and that is measured rather than inferred
///
/// The paragraph above said the cost was "up to a whole tranche". It is the
/// tranche, to within the measurement's own noise. Three windows of one driven
/// x86/PCI boot, each pairing this census against `drain_duty` at the same `t=`:
///
/// ```text
/// max_tranche_us  42563   42117   105308
/// max_age_us      41711   40627   103619
/// ```
///
/// Two consequences, and the second is the one that redirects the search.
///
/// **The rate is not marginal.** The same windows read `queued=71 direct=69`
/// and `queued=67 direct=74` — about half of the guest's register writes miss
/// the lock — with `age_us/queued` at 28.9 ms mean and `frame_late` 63 of 71.
/// Nine in ten deferred doorbells start their work more than a frame after the
/// guest was told the store retired.
///
/// **Lowering `duty` does not fix it.** The fourth window of the same run read
/// `duty=0.147` with `max_age_us=103619`: the worker was idle for 85 % of that
/// second and still held the guest's next submission for a tenth of it. A
/// doorbell does not wait for the device to be *busy*, it waits for the device
/// to be *holding the lock*, and one long tranche in an otherwise empty second
/// costs exactly as much as a full one. So the flush rail's cost and this stall
/// are separate problems that happen to share a cause, and the fix for one is
/// not the fix for the other.
///
/// What that leaves is the observation that a queued write does not need the
/// drain to stop — it needs its register applied, which costs microseconds and
/// adds work rather than interrupting any. `queued_offsets` is here to say which
/// registers those actually are, because "apply it sooner" is only safe for
/// registers whose effect is to publish more work.
///
/// # It is one register, and it is the one that only publishes work
///
/// `offsets=1` on every window of a driven boot that queued anything, with no
/// exceptions:
///
/// ```text
/// queued=106 direct=434  off_0x1020=106/25246
/// queued=92  direct=426  off_0x1020=92/20589
/// queued=110 direct=436  off_0x1020=110/25296
/// queued=40  direct=291  off_0x1020=40/45487
/// ```
///
/// `0x1020` is [`crate::model::GFX_REG_CHILD_DOORBELL`]. Every other register
/// the guest writes finds the lock free; the entire stall is one doorbell, rung
/// about a hundred times a second and applied up to 45 ms later.
///
/// That is the best case the paragraph above could have hoped for. A doorbell
/// carries no state the decode depends on — its whole effect is to say a child
/// channel has work — so there is nothing about it that has to be ordered
/// against a tranche in flight, and picking it up mid-tranche only lengthens the
/// work list. The two registers already served lock-free
/// (`GFX_REG_INTR_STATUS_DISP` / `_GPU`) are lock-free for exactly this reason.
///
/// Note the shape of the remaining risk, which is not this census's to answer:
/// recording the doorbell sooner is not the same as *acting* on it sooner. A bit
/// set in an atomic while `drain_pending` is midway through its channel loop
/// still waits for the next tranche unless that loop re-reads the mask. Making
/// it re-read is the un-refuted half of the budget experiment — that one
/// returned early and left `child_mask` set with nothing to re-arm it, and froze
/// a boot for 29 s; adding to the mask and continuing has no such gap.
///
/// # The residual is one other register, and it is *not* the same case
///
/// With `0x1020` served lock-free, two driven boots read 25 and 30 deferred
/// writes across their entire runs, against 348 in four windows before. All but
/// one are `0x1008` (`GFX_REG_FIFO_WRITTEN`), at roughly one a window with ages
/// of 3-28 ms; the odd one is `0x1220` once at boot.
///
/// `0x1008` looks like the obvious next application of the same trick, and
/// superficially it is even tidier: `fifo_read` is already an `Arc<AtomicU32>`
/// read lock-free in the other direction, so the producer counter would just be
/// its mirror. It is not tidier, and the difference is not about the register.
///
/// A doorbell only names work. `fifo_written` **bounds a loop**:
/// `drain_main_fifo` runs until `fifo_read` catches it, and today that comparand
/// cannot move mid-tranche because the guest's write needs the device lock. Make
/// it live and the loop follows a producer free to keep writing, so a guest
/// submitting steadily holds the device lock indefinitely — a hang, traded for a
/// delay that is now about one write per ten seconds. The refill above is
/// bounded precisely because it has the same hazard; a live comparand has
/// nowhere to put a bound without snapshotting, and a snapshot is what the
/// current code already is.
///
/// So the remaining 1 % is left where it is, deliberately. If it is ever worth
/// taking, the shape is a snapshot re-taken at a bounded number of points, not a
/// live read.
#[derive(Default)]
pub(crate) struct DoorbellCensus {
    queued: std::sync::atomic::AtomicU64,
    age_us: std::sync::atomic::AtomicU64,
    max_age_us: std::sync::atomic::AtomicU64,
    /// Queued writes whose apply was late by at least a whole frame interval.
    frame_late: std::sync::atomic::AtomicU64,
    direct: std::sync::atomic::AtomicU64,
    /// Rings served without asking for the device lock at all.
    lock_free: std::sync::atomic::AtomicU64,
    /// Which registers are actually being deferred: offset -> (count, max age).
    ///
    /// A lock on a census, which the atomics next door exist to avoid — and it
    /// is only reached on the *queued* path, which this census measures at ~70 a
    /// second against a `direct` path that never touches it. The alternative was
    /// a fixed offset table, which would have to be kept in step with the
    /// register map by hand and would silently drop whatever it did not list.
    queued_offsets: parking_lot::Mutex<std::collections::BTreeMap<u64, (u64, u64)>>,
    last_report_ms: std::sync::atomic::AtomicU64,
}

impl DoorbellCensus {
    /// Count one write applied straight from the vCPU thread, lock uncontended.
    ///
    /// Polled on the same one-in-[`UNCONTENDED_POLL`] rule as the free
    /// acquisitions next door, and for the same reason: this is the hot MMIO
    /// path, and a window with no queueing still has to be able to say so.
    pub(crate) fn note_direct(&self, now_ms: impl FnOnce() -> u64) -> Option<String> {
        let prior = self
            .direct
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if !prior.is_multiple_of(UNCONTENDED_POLL) {
            return None;
        }
        self.maybe_report(now_ms())
    }

    /// Count one ring taken with no device lock asked for.
    ///
    /// Polled on the same one-in-[`UNCONTENDED_POLL`] rule as `note_direct`,
    /// and for the same reason: this is the hot MMIO path.
    pub(crate) fn note_lock_free(&self, now_ms: impl FnOnce() -> u64) -> Option<String> {
        let prior = self
            .lock_free
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if !prior.is_multiple_of(UNCONTENDED_POLL) {
            return None;
        }
        self.maybe_report(now_ms())
    }

    /// Record the queue age of one applied doorbell, and which register it was.
    pub(crate) fn note_queued(&self, offset: u64, age_us: u64, now_ms: u64) -> Option<String> {
        use std::sync::atomic::Ordering::Relaxed;
        self.queued.fetch_add(1, Relaxed);
        self.age_us.fetch_add(age_us, Relaxed);
        self.max_age_us.fetch_max(age_us, Relaxed);
        if age_us >= DRAIN_TRANCHE_SLOW_US {
            self.frame_late.fetch_add(1, Relaxed);
        }
        {
            let mut by_offset = self.queued_offsets.lock();
            let slot = by_offset.entry(offset).or_insert((0, 0));
            slot.0 += 1;
            slot.1 = slot.1.max(age_us);
        }
        self.maybe_report(now_ms)
    }

    fn maybe_report(&self, now_ms: u64) -> Option<String> {
        use std::sync::atomic::Ordering::Relaxed;
        let last = self.last_report_ms.load(Relaxed);
        if last == 0 {
            self.last_report_ms.store(now_ms, Relaxed);
            return None;
        }
        let win_ms = now_ms.saturating_sub(last);
        if win_ms < DRAIN_DUTY_REPORT_MS {
            return None;
        }
        self.last_report_ms.store(now_ms, Relaxed);
        let queued = self.queued.swap(0, Relaxed);
        let total = self.age_us.swap(0, Relaxed);
        let max = self.max_age_us.swap(0, Relaxed);
        let late = self.frame_late.swap(0, Relaxed);
        let direct = self.direct.swap(0, Relaxed);
        let lockfree = self.lock_free.swap(0, Relaxed);
        // Descending by count, capped, and the cap is reported rather than
        // silently applied: a register that misses the list because three others
        // out-counted it must not read as a register that never deferred.
        let mut offsets: Vec<(u64, (u64, u64))> = std::mem::take(&mut *self.queued_offsets.lock())
            .into_iter()
            .collect();
        offsets.sort_by_key(|(off, (count, _))| (std::cmp::Reverse(*count), *off));
        let distinct = offsets.len();
        let mut body = String::new();
        for (off, (count, max_us)) in offsets.iter().take(DOORBELL_OFFSETS_REPORTED) {
            body.push_str(&format!(" off_{off:#x}={count}/{max_us}"));
        }
        Some(format!(
            "gfx_doorbell_delay win_ms={win_ms} queued={queued} direct={direct} \
             lockfree={lockfree} age_us={total} max_age_us={max} frame_late={late} \
             slow_us={DRAIN_TRANCHE_SLOW_US} offsets={distinct} shown={}{body}",
            distinct.min(DOORBELL_OFFSETS_REPORTED)
        ))
    }
}

/// How many deferred register offsets `gfx_doorbell_delay` names per window.
///
/// The line has to stay one line, and the question it answers — "which
/// registers are being held back" — is answered by the head of the
/// distribution: a register deferring twice a second is not what costs a frame.
/// `offsets=` states how many distinct ones there were, so a truncated tail is
/// visible rather than implied.
const DOORBELL_OFFSETS_REPORTED: usize = 4;

static DOORBELL: std::sync::LazyLock<DoorbellCensus> =
    std::sync::LazyLock::new(DoorbellCensus::default);

/// Count one child doorbell taken on the vCPU thread with no device lock at all.
///
/// Distinct from [`note_doorbell_direct`], which counts a write that *took* the
/// lock and found it free. This one never asks, so it can neither queue nor
/// contend — and the pair is what says so: `lockfree` rising while `queued`
/// falls to zero is the register leaving the contended path, whereas `queued`
/// staying up would mean something is still routing it through `gfx_ingress`.
pub fn note_doorbell_lock_free() {
    if let Some(line) = DOORBELL.note_lock_free(|| crate::observe::elapsed_ms() as u64) {
        crate::observe::off(line);
    }
}

/// Count one doorbell applied on the vCPU thread without queueing.
pub fn note_doorbell_direct() {
    if let Some(line) = DOORBELL.note_direct(|| crate::observe::elapsed_ms() as u64) {
        crate::observe::off(line);
    }
}

/// Record how long one doorbell sat in `gfx_ingress` before being applied.
pub fn note_doorbell_queued(offset: u64, age_us: u64) {
    if let Some(line) = DOORBELL.note_queued(offset, age_us, crate::observe::elapsed_ms() as u64) {
        crate::observe::off(line);
    }
}

static VCPU_LOCK: std::sync::LazyLock<VcpuLockCensus> =
    std::sync::LazyLock::new(VcpuLockCensus::default);

/// Count one uncontended device-lock acquisition from the vCPU thread.
///
/// Emits the same one-line census as the wait path, so a window that saw
/// traffic but never blocked still says so.
pub fn note_vcpu_lock_free() {
    if let Some(line) = VCPU_LOCK.note_uncontended(|| crate::observe::elapsed_ms() as u64) {
        crate::observe::off(line);
    }
}

/// Record one contended device-lock wait from the vCPU thread; emits at most
/// once per second.
pub fn note_vcpu_lock_wait(us: u64) {
    if let Some(line) = VCPU_LOCK.note_wait(us, crate::observe::elapsed_ms() as u64) {
        crate::observe::off(line);
    }
}

static DRAIN_DUTY: std::sync::LazyLock<DrainDutyCensus> =
    std::sync::LazyLock::new(DrainDutyCensus::default);

static RESIDENT_ARM: std::sync::LazyLock<ResidentArmCensus> =
    std::sync::LazyLock::new(ResidentArmCensus::default);

static SURFACE_WRITE: std::sync::LazyLock<SurfaceWriteCensus> =
    std::sync::LazyLock::new(SurfaceWriteCensus::default);

static WINDOW_PUBLISH: std::sync::LazyLock<WindowPublishCensus> =
    std::sync::LazyLock::new(WindowPublishCensus::default);

/// Record how one tranche's host-window publish attempt ended.
pub fn note_window_publish(arm: WindowPublish) {
    WINDOW_PUBLISH.note(arm);
}

/// Attribute `us` of one surface writeback to one of its whole-frame passes.
pub fn note_surface_write_phase(phase: SurfaceWritePhase, us: u64) {
    SURFACE_WRITE.note(phase, us);
}

/// Record which landing path one surface writeback took, and how many bytes of
/// frame it carried.
pub fn note_surface_write_path(contiguous: bool, bytes: u64) {
    SURFACE_WRITE.note_path(contiguous, bytes);
}

/// Stamp one resident-backed render window as armed.
pub fn note_resident_window_armed() {
    RESIDENT_ARM.note_arm(crate::observe::elapsed_us());
}

/// Record that a flush reached a resident-backed window's readback.
pub fn note_resident_window_flushed() {
    RESIDENT_ARM.note_flush(crate::observe::elapsed_us());
}

/// Accumulate one completed drain tranche; emits at most once per second.
pub fn note_drain_tranche(drain_us: u64, publish_us: u64) {
    if let Some(line) = DRAIN_DUTY.note(drain_us, publish_us, crate::observe::elapsed_ms() as u64) {
        crate::observe::off(line);
        // Immediately after `drain_duty`, so the two read as one record: the
        // rails must sum to its `flush_us` and their counts to its `flushes`.
        if let Some(rails) = DRAIN_DUTY.take_flush_rails() {
            crate::observe::off(rails);
        }
        // Under `flush_rails`, dividing its `render_us`.
        if let Some(split) = DRAIN_DUTY.take_readback_split() {
            crate::observe::off(split);
        }
        // Beside `readback_split`, because it is only readable against it: the
        // question is whether `age_us/aged` leaves room for `fence_us/fence`.
        if let Some(age) = RESIDENT_ARM.take(DRAIN_DUTY.last_window_ms()) {
            crate::observe::off(age);
        }
        // Under `readback_split`, dividing its `write_us` the same way it
        // divides `flush_rails`'s `render_us`.
        if let Some(write) = SURFACE_WRITE.take(DRAIN_DUTY.last_window_ms()) {
            crate::observe::off(write);
        }
        // The offer side of `host_window_cadence`, which can only see the
        // frames that reached it.
        if let Some(publish) = WINDOW_PUBLISH.take(DRAIN_DUTY.last_window_ms()) {
            crate::observe::off(publish);
        }
        // Under `window_publish`, which says how many frames were offered but
        // not why fewer reached the screen.
        emit_engine_lock(DRAIN_DUTY.last_window_ms());
        if let Some(routes) = take_store_routes() {
            crate::observe::off(routes);
        }
        // Onto the census cadence rather than a timer of its own, so a reader
        // pairing the footprint against `store_routes` is reading one clock.
        // The run dump rate-limits itself; this is the only caller.
        for line in crate::observe::footprint::census_lines(crate::observe::elapsed_ms() as u64) {
            crate::observe::off(line);
        }
        emit_engine_delta();
        // After `emit_engine_delta`, which emits `draw_phase`: the two divide
        // against each other and reading them in the other order invites
        // treating the engine's twelve phases as the whole draw, which is the
        // misreading this line exists to correct. Not gated on the backend —
        // the timer is runtime-side and the Metal arm can adopt it without a
        // second census.
        emit_chain_phase();
    }
}

/// The engine's own counters, over the window `drain_duty` just reported.
///
/// Two of them were tallied and never reported, and they are the two that price
/// the largest phase of a draw. `draw_phase` puts **70% of all timed draw work
/// in `Acquire`** — 193 µs per draw, 130 ms of a driven second — while
/// `creates=74 allocs=0` rules out the allocation churn that phase's own doc
/// names as its cost. What is left in there and scales with content is
/// [`crate::backend::vulkan::engine::pools`]'s sampled-cache lookup, which
/// fingerprints the **whole** incoming blob with two SipHash passes on every
/// call that does not take the identity fast path.
///
/// `sampled_cache_hit_bytes` is exactly the byte count fed to that fingerprint
/// on the hit path, and `sampled_identity_hits` is the count that skipped it
/// entirely. Together they turn "the cache is working, hits=122 misses=0" —
/// which is what the line said, and which reads as nothing to fix — into a
/// GB/s figure that can be compared against SipHash's throughput. Neither can
/// be derived from the counts alone: 122 hits over 4 KiB blobs and 122 over
/// 8 MiB blobs are three orders of magnitude apart and the line printed the
/// same number for both.
///
/// `drain_duty` established that 96-99% of the saturated drain second is
/// `draw_us`, at 1.5-7 ms per draw — orders of magnitude more than a draw's CPU
/// encode should cost. Which of the engine's per-draw costs that is was already
/// being counted and never reported: `engine::counter_snapshot` had no product
/// caller, so every one of these numbers existed and no boot had read one.
///
/// So this adds no instrumentation, only a window delta of what the engine
/// already tallies, chosen to separate the candidates that could each explain
/// milliseconds per draw:
///
/// - `batch_*` — whether draws coalesce into one submission or each takes its
///   own. Per-draw submission is a full CPU-GPU round trip.
/// - `readbacks` / `readback_bytes` — whether every draw drags its target back
///   to host memory, which is a fence wait plus a copy.
/// - `render_post_wait_skips` / `target_reads` — the two halves of the deferred
///   composite Store. The first counts draws that returned without a fence wait
///   because they kept their pixels on the GPU; the second counts the reads a
///   consumer later asked for. A rail that only *moves* the copy raises the
///   second by as much as it raises the first, and `readbacks` alone — which
///   pooled both until it was split — reported no change at all in that case.
/// - `creates` / `*_misses` — pipeline, shader and descriptor churn, where a
///   miss is a driver compile rather than a lookup.
/// - `sampled_reuploads` — re-staging texture content a cache hit should have
///   kept.
/// - `sampled_gathers` / `sampled_gather_bytes` — sampled binds served by
///   gathering scattered guest pages into staging. The sampled loop's only
///   byte-moving arm, and the last one of that loop to report itself.
/// - `ring_retire_blocks` / `target_evicts` — the engine waiting on itself.
///
/// One line per second, one atomic load per field. Emitted from the same window
/// as `drain_duty` so the two divide against each other; a delta on its own
/// clock would not.
/// Would a resident carry the present this mapping names, at this geometry?
///
/// `Some(true)` a presentable resident exists, `Some(false)` none does — so a
/// present with no guest-page frame behind it shows black — and `None` on a
/// backend with no target registry to ask, where the honest answer is that this
/// build cannot tell.
///
/// It asks through [`crate::backend::vulkan::engine::resident_presentable`],
/// which shares `pools::slot_presentable` with the window presenter's own
/// selection. Sharing the rule is the point rather than tidiness: a looser
/// predicate here would report a frame as carried that the presenter then
/// refuses, which is a disagreement neither call site can see on its own — the
/// same shape as the publish/present split that once blanked the window.
#[cfg(feature = "backend-vulkan")]
fn present_resident_carries(
    state: &crate::model::DeviceState,
    mapping: u32,
    width: u32,
    height: u32,
) -> Option<bool> {
    let identity =
        crate::runtime::present_identity::surface_identity(state, mapping, width, height);
    Some(crate::backend::vulkan::engine::resident_presentable(
        &identity, width, height,
    ))
}

#[cfg(not(feature = "backend-vulkan"))]
fn present_resident_carries(
    _state: &crate::model::DeviceState,
    _mapping: u32,
    _width: u32,
    _height: u32,
) -> Option<bool> {
    None
}

/// Which channel an unbacked present belongs on: `true` is the failure channel.
///
/// A separate function because the `None` arm is the whole content of the rule
/// and it is one character away from being wrong. `carried != Some(true)` and
/// `carried == Some(false)` differ only when the build cannot answer, and that is
/// exactly the case where a possible black frame would be downgraded to a census
/// with nothing to notice it. Fail-closed: only a resident that positively
/// carries the frame demotes the line.
fn unbacked_present_is_a_loss(carried: Option<bool>) -> bool {
    carried != Some(true)
}

/// The `carried=` field: what answered for this present, or that nothing could.
fn carrier_word(carried: Option<bool>) -> &'static str {
    match carried {
        Some(true) => "resident",
        Some(false) => "nothing",
        None => "unknown",
    }
}

#[cfg(feature = "backend-vulkan")]
fn emit_engine_delta() {
    use crate::backend::vulkan::engine::CounterSnapshot;
    static PREV: std::sync::Mutex<Option<CounterSnapshot>> = std::sync::Mutex::new(None);
    let now = crate::backend::vulkan::engine::counter_snapshot();
    let Ok(mut prev) = PREV.lock() else {
        return;
    };
    let d = now.delta_since(&prev.unwrap_or_default());
    *prev = Some(now);
    crate::observe::off(format!(
        "engine_delta creates={} allocs={} batch_opens={} batch_joins={} batch_flushes={} \
         batch_flush_draws={} readbacks={} readback_bytes={} render_post_wait_skips={} \
         target_reads={} target_read_bytes={} pipeline_misses={} \
         shader_misses={} pass_misses={} layout_misses={} sampler_misses={} \
         sampled_cache_hits={} sampled_identity_hits={} sampled_cache_hit_bytes={} \
         sampled_cache_misses={} sampled_reuploads={} \
         sampled_reupload_bytes={} sampled_gathers={} sampled_gather_bytes={} \
         sampled_gather_skips={} sampled_gather_skip_bytes={} \
         seed_uploads={} seed_upload_bytes={} \
         ring_retire_blocks={} target_evicts={} desc_pool_grow={} gen_mismatch={}",
        d.creates,
        d.allocs,
        d.batch_opens,
        d.batch_joins,
        d.batch_flushes,
        d.batch_flush_draws,
        d.readbacks,
        d.readback_bytes,
        d.render_post_wait_skips,
        d.target_reads,
        d.target_read_bytes,
        d.pipeline_misses,
        d.shader_misses,
        d.pass_misses,
        d.layout_misses,
        d.sampler_misses,
        d.sampled_cache_hits,
        d.sampled_identity_hits,
        d.sampled_cache_hit_bytes,
        d.sampled_cache_misses,
        d.sampled_reuploads,
        d.sampled_reupload_bytes,
        d.sampled_gathers,
        d.sampled_gather_bytes,
        d.sampled_gather_skips,
        d.sampled_gather_skip_bytes,
        d.seed_uploads,
        d.seed_upload_bytes,
        d.ring_retire_blocks,
        d.target_evicts,
        d.desc_pool_grow,
        d.gen_mismatch,
    ));
    emit_draw_phase();
}

/// The split of `drain_duty`'s `draw_us` that actually covers it, over the same
/// window.
///
/// `draw_phase` divides the engine and `chain_phase` divides everything around
/// it, so this line is emitted immediately after that one and the two are read
/// together: `chain_phase`'s `engine_us` must equal `draw_phase`'s twelve
/// summed, and `chain_phase`'s eight must equal `drain_duty`'s `draw_us`.
/// Whatever `draw_phase` does not account for is the other seven bars here, and
/// on the boot that motivated this line that was 82% of the draw.
///
/// Silent when no chain ran, so an idle desktop costs nothing.
///
/// The split of `chain_phase`'s `binds_us`, over the same window.
///
/// Emitted immediately after it, in the same relationship `draw_phase` has to
/// `engine_us`: divide the three against the column above. They are not claimed
/// to sum to it — see [`crate::runtime::bind_phase`] for why a computed
/// remainder was left out.
fn emit_bind_phase() {
    let Some(w) = crate::runtime::bind_phase::take_window() else {
        return;
    };
    crate::observe::off(format!(
        "bind_phase binds={} vertex_us={} fragment_us={} attrs_us={}",
        w.binds, w.vertex_us, w.fragment_us, w.attrs_us,
    ));
}

fn emit_chain_phase() {
    let Some(w) = crate::runtime::chain_phase::take_window() else {
        return;
    };
    crate::observe::off(format!(
        "chain_phase chains={} prep_us={} pipeline_us={} binds_us={} sampled_us={} \
         seed_us={} assemble_us={} engine_us={} store_us={} max_us={}",
        w.chains,
        w.prep_us,
        w.pipeline_us,
        w.binds_us,
        w.sampled_us,
        w.seed_us,
        w.assemble_us,
        w.engine_us,
        w.store_us,
        w.max_us,
    ));
    // Under `chain_phase`, dividing its largest column the same way
    // `draw_phase` divides its `engine_us`.
    emit_bind_phase();
}

/// The split of `drain_duty`'s `draw_us`, over the same window.
///
/// `drain_duty` says a saturated second is 93-99% `draw_us` and `engine_delta`
/// says ~450 MB/s crosses the bus each way. Those two are consistent with
/// opposite fixes — moving fewer bytes, or stopping the per-draw GPU round trip
/// — and neither line can tell them apart. This one can: `readback_us` and the
/// staging half of `setup_us` scale with bytes, `wait_us` does not.
///
/// Silent when no draw ran, so an idle desktop costs nothing.
#[cfg(feature = "backend-vulkan")]
fn emit_draw_phase() {
    let Some(w) = crate::backend::vulkan::engine::draw_phase_window() else {
        return;
    };
    crate::observe::off(format!(
        "draw_phase draws={} prep_us={} pipeline_us={} stage_us={} stage_pass_us={} \
         acquire_us={} acquire_sampled_us={} sampled_upload_us={} acquire_readback_us={} \
         descriptors_us={} \
         record_us={} submit_us={} wait_us={} readback_us={} max_us={} stalls={}",
        w.draws,
        w.prep_us,
        w.pipeline_us,
        w.stage_us,
        w.stage_pass_us,
        w.acquire_us,
        w.acquire_sampled_us,
        w.sampled_upload_us,
        w.acquire_readback_us,
        w.descriptors_us,
        w.record_us,
        w.submit_us,
        w.wait_us,
        w.readback_us,
        w.max_us,
        w.stalls,
    ));
    emit_stage_phase();
}

/// Under `draw_phase`, dividing its largest column — `stage_us` is 83 % of that
/// phase's second on a driven drag, and the five parts want opposite fixes.
#[cfg(feature = "backend-vulkan")]
fn emit_stage_phase() {
    let Some(w) = crate::backend::vulkan::engine::stage_phase::take_window() else {
        return;
    };
    crate::observe::off(format!(
        "stage_phase acquire_us={} acquires={} bytes_us={} bytes_n={} bytes_b={} \
         runs_us={} runs_n={} runs_b={} swap_us={} swap_n={} swap_b={} \
         shift_us={} shift_n={} shift_b={}",
        w.acquire_us,
        w.acquires,
        w.bytes_us,
        w.bytes_n,
        w.bytes_b,
        w.runs_us,
        w.runs_n,
        w.runs_b,
        w.swap_us,
        w.swap_n,
        w.swap_b,
        w.shift_us,
        w.shift_n,
        w.shift_b,
    ));
}

#[cfg(not(feature = "backend-vulkan"))]
fn emit_engine_delta() {}

/// The engine mutex's wait and hold time over the same window, split by which
/// thread class asked for it.
///
/// Emitted beside `window_publish` because it divides the gap that line opens:
/// `window_publish fresh` is what the device offered the window and
/// `host_window_cadence presents` is what reached the screen, and when the two
/// disagree the first candidate is that the window thread could not have the
/// engine while the worker held it.
#[cfg(feature = "backend-vulkan")]
fn emit_engine_lock(win_ms: u64) {
    if let Some(line) = crate::backend::vulkan::engine::take_engine_lock_census(win_ms) {
        crate::observe::off(line);
    }
}

#[cfg(not(feature = "backend-vulkan"))]
fn emit_engine_lock(_win_ms: u64) {}

/// Count a drain wake-up that returned before taking the device lock.
pub fn note_drain_skipped() {
    DRAIN_DUTY.note_skipped();
}

/// Attribute elapsed time since `started` to one phase of the current tranche.
pub fn note_drain_phase(phase: DrainPhase, started: std::time::Instant) {
    DRAIN_DUTY.note_phase(phase, started.elapsed().as_micros() as u64);
}

/// Attribute one slice of a render-rail flush to the part of it that was spent.
pub fn note_readback_phase(phase: ReadbackPhase, us: u64) {
    DRAIN_DUTY.note_readback(phase, us);
}

/// Record the two GPU-side spans of one readback command buffer, read from its
/// own timestamp queries after the fence signalled.
///
/// Reported as `readback_split bar_us`/`gpu_us` beside `fence_us`, which they
/// divide. `fence_us` is CPU wall clock across `vkWaitForFences` and therefore
/// contains three different things with three different fixes: the draw batch
/// still executing (`bar_us`, the copy's barrier waiting on it), the copy
/// itself (`gpu_us`), and the cost of asking (the remainder). Both spans are
/// deltas between two points on the GPU's own timeline, so no clock correlation
/// is involved and the subtraction is exact.
pub fn note_readback_gpu_us(barrier_us: u64, copy_us: u64) {
    DRAIN_DUTY.note_readback_gpu(barrier_us, copy_us);
}

/// Count one guest-Store routing decision, by route name.
///
/// The routes are the attribution for `engine_delta`'s readback bytes: only
/// `cpu_portability` reads a full frame back and CPU-copies it into the guest's
/// pages, and only it is forced to — `gva_store_defer_eligible` refuses any
/// target with a nonzero `mapping_id`, so a type-11 composite Store has no
/// deferred rail to take. Whether that is 2 Stores a second or 20 decides
/// whether building one is worth it, and the route's own first-appearance line
/// is deduplicated per process and cannot say.
static STORE_ROUTES: std::sync::Mutex<Option<std::collections::BTreeMap<&'static str, u64>>> =
    std::sync::Mutex::new(None);

/// # This census cannot find the Finder icon defect, and that is now measured
///
/// Several sessions have concluded "no counter separates a corrupt icon round
/// from a clean one" by printing six or eight hand-picked columns. That is a
/// statement about the columns someone thought to print, not about the census.
/// It has now been asked of the whole census at once.
///
/// Three 14-round `icon-composite.sh` boots, x86 / Vulkan, pooled: **42 scored
/// rounds, 9 corrupt, 33 clean**. Every counter in this map present in at least
/// 80% of rounds was normalised per 1000 `draw_scissor_full` — round length
/// varies ~40% on this rig and almost every draw-path counter is proportional
/// to it — and ranked by AUC, the probability that a random corrupt round
/// scores above a random clean one. The best column in the entire census:
///
/// ```text
/// AUC 0.75  surface_flush             permutation p = 0.021 raw
/// AUC 0.73  load_seed_ok                            p = 0.914 Bonferroni
/// AUC 0.72  type11_seed_uploaded      (43 columns tested)
/// AUC 0.72  type11_seed_guest_wrote
/// AUC 0.71  t11_gw_ref_moved
/// ```
///
/// Corrected for having looked at 43 columns, nothing is distinguishable from
/// noise. The leaders are also largely one quantity wearing different names — a
/// type-11 seed upload is a `load_seed_ok_mapping` — so they are one weak
/// signal, not five.
///
/// The reason is structural rather than a gap to be filled by adding counters.
/// A round runs ~11 000 draws and the defect is **one** icon: a single
/// operation going wrong is a ~1e-4 perturbation of any population, which no
/// aggregate can resolve. Adding a counter to this map cannot change that, and
/// a session that adds one and reads it per round is repeating a measurement
/// that has now been shown to have no power.
///
/// What would have power is a *screen-to-resource join*: name the 64x64 target
/// backing the cell that is blank in the capture, then dump that one target's
/// history. A distinct-texel content summary would be one half of it — a
/// correct icon carries hundreds of distinct texels and a blank one collapses
/// to one — and the other half is the mapping from a screen rectangle to a
/// target identity. Neither exists today; `observe::bgra_present_stats` is the
/// nearest thing and it summarises a whole frame, not one target.
///
/// Settled by the same three boots, so nobody re-runs it: the Vulkan
/// synchronization repairs are not the producer either. Corruption rates were
/// 3/14 before them, 4/14 after the first, 2/14 after all five. The hazards
/// they closed were real undefined behaviour and those fixes stand on that
/// ground alone — see
/// [`crate::backend::vulkan::engine::exec::resident_read_source_scope`] — but
/// they do not move this class.
///
/// # A scoring flaw that inverts verdicts, recorded here because the harness is not tracked
///
/// The repro scripts live under `.agents/`, which is gitignored, so a fix made
/// there does not survive to the next session and this warning would vanish
/// with it.
///
/// `iconscore.py` scores a capture by counting blue blobs in a horizontal band
/// and comparing the count to `--expect`. Its own description defines the
/// population as "blue blobs of near-identical area", but it only ever policed
/// the *small* side of that (a `shrunk` class). On 2026-07-31 an unrelated blue
/// object of area 3247, against an icon median of 1235, entered the band and
/// was counted toward `expect`. That **inverted the verdict of all fourteen
/// rounds of a probe boot**: a round showing all seven icons counted 8 and read
/// CORRUPT, and a round genuinely missing one counted 7 and read CLEAN.
///
/// It was caught only by re-deriving each round's verdict from the *positions*
/// of the blobs rather than their number. Any conclusion of the form "n corrupt
/// rounds out of m" is worth exactly as much as the assumption that nothing
/// else blue and icon-sized was on screen, and that assumption is not
/// self-checking. A symmetric `outsized` exclusion, reported on the output line
/// rather than applied silently, is the fix; if the harness in front of you
/// does not print `outsized=` when something is excluded, it predates this and
/// its verdicts should be re-derived positionally before they are believed.
pub fn note_store_route(route: &'static str) {
    note_store_route_n(route, 1);
}

/// Add `n` to a named count in the same per-second window as [`note_store_route`].
///
/// For events that arrive in batches — one notify marking many cache entries —
/// where the number that matters is the entries, not the notifies, and taking
/// the lock once per entry would cost more than the census is worth.
pub fn note_store_route_n(route: &'static str, n: u64) {
    if n == 0 {
        return;
    }
    if let Ok(mut g) = STORE_ROUTES.lock() {
        *g.get_or_insert_with(Default::default)
            .entry(route)
            .or_default() += n;
    }
}

/// Accumulate microseconds against a named cost, into the same per-second window
/// as the route counts above.
///
/// The same map on purpose. `store_routes` is already drained once a second
/// beside `drain_duty`, so a cost reported here divides into that window's
/// `draw_us` with no join and no cross-boot comparison. `draw_phase` cannot
/// carry these: it brackets the *engine's* internals, and this is the runtime
/// work on either side of them — which is where **28 % of `draw_us`** was
/// going unattributed (~245 ms per second, stable across 200 windows of the
/// 2026-07-30 boot, larger than `stage_us` and `readback_us` and second only to
/// `wait_us`). A phase table that sums to 72 % of the thing it decomposes
/// cannot be used to choose what to fix.
pub fn note_store_route_us(name: &'static str, us: u64) {
    if let Ok(mut g) = STORE_ROUTES.lock() {
        *g.get_or_insert_with(Default::default)
            .entry(name)
            .or_default() += us;
    }
}

/// Read one route's count out of the live window, for tests that assert a
/// census fired rather than trusting that it was wired up.
///
/// A counter nobody reads back is a counter that can be deleted, mistyped, or
/// placed on the wrong side of an early return without any test noticing — and
/// several of this crate's readings have turned on exactly which side of a
/// branch a `note_store_route` sat on.
#[cfg(test)]
pub(crate) fn store_route_count(route: &str) -> u64 {
    STORE_ROUTES
        .lock()
        .ok()
        .and_then(|g| g.as_ref().and_then(|m| m.get(route).copied()))
        .unwrap_or(0)
}

/// Drain and format the window's route counts, or `None` if none were taken.
fn take_store_routes() -> Option<String> {
    let mut g = STORE_ROUTES.lock().ok()?;
    let routes = g.as_mut()?;
    if routes.is_empty() {
        return None;
    }
    let mut out = String::from("store_routes");
    for (route, n) in routes.iter() {
        out.push_str(&format!(" {route}={n}"));
    }
    routes.clear();
    Some(out)
}

fn signal_display_vbl_at<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    last_us: &std::sync::atomic::AtomicU64,
    now_us: u64,
) {
    // The limiter paces in microseconds because 120 Hz is not expressible in
    // whole milliseconds; the census windows in milliseconds so its `t=` stays
    // on the same scale as every other always-on line.
    let now_ms = now_us / 1_000;
    if state.display.shared_gpa == 0 || !state.display.online_acked {
        note_vbl(VBL_NOT_ONLINE, now_ms);
        return;
    }
    if !claim_display_vbl(last_us, now_us) {
        note_vbl(VBL_NOT_CLAIMED, now_ms);
        return;
    }
    note_vbl(VBL_DELIVERED, now_ms);
    let gpa = state.display.shared_gpa;
    let mut pending_le = [0u8; 4];
    let pending = if host
        .read_gpa(gpa + DISPLAY_SHARED_PENDING, &mut pending_le)
        .is_ok()
    {
        ld32(&pending_le)
    } else {
        0
    };
    // Drop a stale (already-acked) ONLINE bit so we don't re-deliver it and make
    // the guest re-run process_online → connectionChange → overlay rebuild (see
    // signal_display_present_complete). signal_display_vbl only runs post-ack, so
    // online_acked is already true here; `stale` is 0 on healthy boots (no-op).
    let stale = state.display.online_acked && pending & DISPLAY_ONLINE_EVENT_MASK != 0;
    let base = if stale {
        pending & !DISPLAY_ONLINE_EVENT_MASK
    } else {
        pending
    };
    shared_w32(
        host,
        gpa,
        DISPLAY_SHARED_PENDING,
        base | DISPLAY_VBL_EVENT_MASK,
        state.page_size() as usize,
    );
    if stale {
        crate::runtime::census::present_proxy::note_stale_online_pending("vbl", pending);
    }
    let bit = 1u32 << (state.display.display_index & 0x1f);
    state
        .gfx
        .interrupt_status_disp
        .fetch_or(bit, std::sync::atomic::Ordering::AcqRel);
    host.enqueue(HostAction::irq_gfx());
}

/// Assert display ONLINE once the guest has published the enable mask.
///
/// Archive `apple_pv_gpu_display_signal_online` + poll_tick gate:
/// write shared `+0x100` pending bit 2, then pulse display IRQ. Only after
/// `enable()` sets `+0x104` bit 2 — earlier IRQs wedge an unregistered display.
/// createDisplayAttributes then consumes TimingElements (incl. 1440 mode).
pub fn try_display_online<H: HostMemory + HostOps>(state: &mut DeviceState, host: &mut H) {
    if state.display.shared_gpa == 0 || state.display.online_acked {
        return;
    }
    if state.display.online_tries >= DISPLAY_ONLINE_MAX_TRIES {
        return;
    }
    // Cadence: skip most ticks (archive divisor); still run often enough via
    // gfx_update / drain that enable() is observed within seconds.
    let ctr = state.display.poll_ctr.wrapping_add(1);
    state.display.poll_ctr = ctr;
    if !ctr.is_multiple_of(DISPLAY_ONLINE_POLL_DIVISOR) {
        return;
    }
    let gpa = state.display.shared_gpa;
    let mut mask_le = [0u8; 4];
    if host
        .read_gpa(gpa + DISPLAY_SHARED_ENABLE_MASK, &mut mask_le)
        .is_err()
    {
        return;
    }
    let mask = ld32(&mask_le);
    if mask & DISPLAY_ONLINE_EVENT_MASK == 0 {
        return;
    }
    // pending word is atomic read-and-clear on the guest side.
    shared_w32(
        host,
        gpa,
        DISPLAY_SHARED_PENDING,
        DISPLAY_ONLINE_EVENT_MASK,
        state.page_size() as usize,
    );
    let bit = 1u32 << (state.display.display_index & 0x1f);
    state
        .gfx
        .interrupt_status_disp
        .fetch_or(bit, std::sync::atomic::Ordering::AcqRel);
    host.enqueue(HostAction::irq_gfx());
    // Always-on on the first ONLINE pulse per shared-state generation (rare, not a
    // flood): the display-lifecycle timeline entry point. A second pass through here
    // after a reinit setup pairs with display_shared_state_setup reinit=1 to show a
    // post-converge display rebuild.
    if state.display.online_tries == 0 {
        crate::observe::fail(format!(
            "display_online_signal index={}",
            state.display.display_index
        ));
    }
    state.display.online_tries = state.display.online_tries.saturating_add(1);
}

/// Drain active/pending child FIFOs other than channels mid-drain.
///
/// Used by DisplaySwap Dekker rescue and stranded paths — **not** by
/// `render_wait_surface` (archive wait is surface-keyed async completion only).
///
/// Mask matches archive `poll_tick`: `active_child_mask | pending_child_mask`
/// so work doorbell'd while a drain was in flight is not skipped. Skips
/// `skip_channel` and every bit in `state.draining_mask` so nested drains
/// cannot re-enter a mid-packet channel (same head re-process).
pub fn drain_other_child_fifos<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    skip_channel: u32,
) {
    let mask = state.active_child_mask | state.pending.child_mask;
    let nested = state.draining_mask;

    // A cold-translation EXEC is already the oldest accepted item in the host
    // scheduler timeline. Retry its channel(s) first. Sibling FIFO packets may
    // tear down tasks, mappings, objects, or surfaces referenced by that EXEC,
    // so none may pass the artificial translation boundary. Translation owns
    // immutable AIR and completes independently of all FIFO drains.
    let deferred = state.translation_deferred_mask;
    if deferred != 0 {
        for ch in 1..MAX_CHANNELS as u32 {
            let bit = 1u32 << ch;
            if deferred & bit == 0 || ch == skip_channel || nested & bit != 0 {
                continue;
            }
            state.pending.child_mask &= !bit;
            drain_child_fifo(state, host, ch);
        }
        if state.translation_deferred_mask != 0 {
            let held = mask & !deferred & !nested & !(1u32 << skip_channel);
            state.pending.child_mask |= held | state.translation_deferred_mask;
            note_translation_order_hold(state, held);
            return;
        }
        release_translation_order_holds(state);
    }

    let mut remaining = mask;
    for ch in 1..MAX_CHANNELS as u32 {
        if state.pending.host_action_yield {
            break;
        }
        if ch == skip_channel {
            continue;
        }
        if nested & (1u32 << ch) != 0 {
            continue;
        }
        if mask & (1u32 << ch) == 0 {
            continue;
        }
        remaining &= !(1u32 << ch);
        // Clear pending bit for channels we actually drain (archive poll_tick
        // consumes pending when it drains). Leave skip/nested bits alone.
        state.pending.child_mask &= !(1u32 << ch);
        drain_child_fifo(state, host, ch);
        if state.translation_deferred_mask != 0 {
            let held = remaining & !nested & !(1u32 << skip_channel);
            state.pending.child_mask |= held | state.translation_deferred_mask;
            note_translation_order_hold(state, held);
            break;
        }
    }
}

/// Host paint consumed the current +0x188 retain (Painted or Unchanged).
///
/// Clears the entry-side present backpressure counter so a DisplaySwap held
/// at channel head can run on the next drain (schedule_bh from scanout path).
pub fn note_present_paint_consumed(state: &mut DeviceState) {
    state.present.unpainted_presents = 0;
    state.present.backpressure_hold_active = false;
    state.pending.host_action_yield = false;
}

fn note_present_backpressure_hold(state: &mut DeviceState, channel: u32, head: u32, tail: u32) {
    if state.present.backpressure_hold_active
        && state.present.backpressure_hold_channel == channel
        && state.present.backpressure_hold_head == head
    {
        return;
    }
    state.present.backpressure_hold_active = true;
    state.present.backpressure_hold_channel = channel;
    state.present.backpressure_hold_head = head;
    state.present.backpressure_hold_count = state.present.backpressure_hold_count.saturating_add(1);
    crate::observe::fail(format!(
        "THRASH present_action_starvation reason=pending_frames_cap ch={channel} head={head} tail={tail} unpainted={} episode={}",
        state.present.unpainted_presents, state.present.backpressure_hold_count
    ));
}

/// Publish the poll-tick/Dekker rescue to the asynchronous drain owner.
///
/// This performs no guest-memory reads and no command execution. QEMU may call
/// it from its display/main-loop context without accidentally translating or
/// submitting GPU work under the BQL. Active child channels are intentionally
/// coalesced into one mask; the worker's normal ring checks make idle channels
/// cheap no-ops.
pub fn publish_stranded_fifos<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
) -> bool {
    let mut published = false;
    if state.gfx.control_fifo != 0 {
        if state
            .gfx
            .fifo_read
            .load(std::sync::atomic::Ordering::Acquire)
            != state.gfx.fifo_written
        {
            state.pending.main_drain = true;
            published = true;
        }
        if state.active_child_mask != 0 {
            state.pending.child_mask |= state.active_child_mask;
            published = true;
        }
    }
    if state.iosfc.consumer != state.iosfc.producer {
        state.pending.iosfc = true;
        published = true;
    }
    // A translation finishing is a producer advance with no doorbell behind it,
    // which is the case this whole function exists for — only the producer is
    // ours, not the guest's, so none of the tests above can see it.
    //
    // The packet that deferred is still at its channel's FIFO head, waiting to
    // be retried. The guest has nothing left to send: it is blocked on the very
    // work that packet represents, so it rings no doorbell. The async
    // translation worker stores its result and returns, with no way to say the
    // packet became runnable. If the channel is not republished here, the retry
    // never happens and neither side moves again.
    //
    // Measured on the Windows rail: a 1.18 MB fragment shader took 9.3 s to
    // translate, the FIFOs were parked meanwhile, and the guest then sat idle
    // for thirteen minutes with the translation it was waiting for long since
    // complete. Republishing costs one drain per poll while a deferral is
    // outstanding, and stops as soon as the retry clears the mask.
    if state.translation_deferred_mask != 0 {
        state.pending.child_mask |= state.translation_deferred_mask;
        published = true;
    }
    // Parked FIFOs need the same rescue, and for the same reason one step later.
    // `release_translation_order_holds` runs only at the top of a drain, so a
    // hold that outlives the deferral that caused it needs a drain scheduled to
    // take it back down — and if the guest is quiet there is nothing to schedule
    // one. Measured: 14 hold episodes against 7 releases, with the deferred mask
    // already clear and every FIFO still parked.
    //
    // Bit 0 names the root FIFO rather than a channel, so it is routed to
    // `main_drain`; putting it in `child_mask` would arm channel 0, which is not
    // what that bit means.
    if state.translation_order_hold_mask != 0 {
        let held = state.translation_order_hold_mask;
        if held & TRANSLATION_ROOT_FIFO_BIT != 0 {
            state.pending.main_drain = true;
        }
        state.pending.child_mask |= held & !TRANSLATION_ROOT_FIFO_BIT;
        published = true;
    }
    // A DisplaySwap parked on present backpressure is the third way to strand a
    // FIFO, and on the host-window path it is a closed cycle.
    //
    // `enqueue_present_scanout` sets `host_action_yield`, which makes
    // `drain_pending` return at its first line. On the window path the ack that
    // clears it -- `note_present_paint_consumed` -- runs only inside
    // `device_drain`, which is the very thing that will not run. There is no
    // `ScanoutUpdate` on this path either, so `device_scanout_copy`, the other
    // caller, never fires. Nothing else can lower the flag, and the guest cannot
    // help: it is waiting for the present it already queued.
    //
    // Measured on the Windows rail: guest-side `GPU hang: Name Display0
    // written: 504 read: 432` repeated 196 times with both ring pointers frozen
    // at the same values, while `display_vbl` kept ticking and `drain_duty` had
    // stopped entirely. The 72-byte gap is the DisplaySwap the entry gate is
    // holding.
    //
    // Republishing the channel gets a drain scheduled; the ack inside it clears
    // the yield and the hold, and the gate re-opens. Costs one drain per poll
    // while a present is outstanding, which is bounded by
    // `MAX_UNPAINTED_PRESENTS`.
    if state.present.backpressure_hold_active || state.pending.host_action_yield {
        state.pending.child_mask |= state.active_child_mask;
        published = true;
    }
    if published {
        host.schedule_bh();
    }
    published
}

/// How long a woken-less worker must sit on outstanding work before it is a
/// stall. Present + GPU encode of one heavy frame stays well under this; every
/// wedge measured so far sat for minutes.
const STALL_REPORT_AFTER_MS: u64 = 5_000;
/// One snapshot per this interval while the stall persists.
const STALL_REPORT_EVERY_MS: u64 = 10_000;

/// Emit a wait-state snapshot when the drain worker has not woken for
/// [`STALL_REPORT_AFTER_MS`] while something is visibly outstanding.
///
/// Every wedge this device has had on the bring-up rail was silent: the worker
/// stopped waking, the periodic censuses stopped with it (they are emitted
/// *from* the worker), and the only signal left was a guest watchdog line on a
/// serial console — or nothing, when the guest waited without a watchdog. A
/// stall diagnosis needed a human to correlate three logs after the fact. This
/// line is the device saying it itself, from the poll path, which keeps running
/// precisely because it is not the thing that wedged.
///
/// A quiet worker with clean host-side state is still reported, because the
/// disagreement that matters may live in guest memory: the guest advances a
/// child ring's tail with a plain store, and if no doorbell follows, no host
/// counter moves. The snapshot therefore reads each channel's guest-visible
/// head/tail out of the root page's register blocks — two 4-byte reads per
/// channel, the same reads a drain does, nowhere near executing work under the
/// caller's lock. A guest watchdog said `written: 1240 read: 1208` while every
/// host counter said "nothing outstanding"; the ring cursors are where that
/// conflict is visible from our side.
pub fn report_stall_if_wedged<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &H,
    now_ms: u64,
) {
    if state.last_drain_wake_ms == 0
        || now_ms.saturating_sub(state.last_drain_wake_ms) < STALL_REPORT_AFTER_MS
        || now_ms.saturating_sub(state.last_stall_report_ms) < STALL_REPORT_EVERY_MS
    {
        return;
    }
    let fifo_read = state
        .gfx
        .fifo_read
        .load(std::sync::atomic::Ordering::Acquire);
    // Guest-visible child ring cursors, for every channel whose register block
    // shows a live cursor pair. `tail != head` is guest work no host counter
    // knows about.
    let mut ring_lag = Vec::new();
    if state.gfx.root_page != 0 {
        for ch in 1..MAX_CHANNELS as u32 {
            let Some(regs_off) = child_reg_block_offset(ch) else {
                continue;
            };
            let regs_gpa = state.pfn_gpa(state.gfx.root_page) + regs_off;
            let (Ok(tail), Ok(head)) = (
                crate::runtime::host::read_u32(host, regs_gpa + CHILD_REG_TAIL),
                crate::runtime::host::read_u32(host, regs_gpa + CHILD_REG_HEAD),
            ) else {
                continue;
            };
            if tail != head {
                ring_lag.push(format!("ch{ch}:head={head} tail={tail}"));
            }
        }
    }
    let outstanding = fifo_read != state.gfx.fifo_written
        || !ring_lag.is_empty()
        || state.pending.main_drain
        || state.pending.child_mask != 0
        || state.pending.iosfc
        || state.pending.host_action_yield
        || state.present.backpressure_hold_active
        || state.present.unpainted_presents != 0
        || state.translation_deferred_mask != 0
        || state.translation_order_hold_mask != 0;
    if !outstanding {
        return;
    }
    state.last_stall_report_ms = now_ms;
    crate::observe::fail(format!(
        "STALL drain_wedged idle_ms={} fifo={}..{} control_fifo={:#x} root_page={:#x} \
         pending_main={} pending_child={:#x} iosfc={} yield={} backpressure={} unpainted={} \
         deferred={:#x} held={:#x} active_children={:#x} rings=[{}]",
        now_ms.saturating_sub(state.last_drain_wake_ms),
        fifo_read,
        state.gfx.fifo_written,
        state.gfx.control_fifo,
        state.gfx.root_page,
        state.pending.main_drain,
        state.pending.child_mask,
        state.pending.iosfc,
        state.pending.host_action_yield,
        state.present.backpressure_hold_active,
        state.present.unpainted_presents,
        state.translation_deferred_mask,
        state.translation_order_hold_mask,
        state.active_child_mask,
        ring_lag.join(" ")
    ));
}

/// Run all pending drains (BH body).
///
/// # This runs to completion on purpose, and a wall-clock budget was tried
///
/// This holds the device lock for its whole duration, and on x86 the vCPU does
/// not block on that lock — `device_gfx_write` takes it with `try_lock` and on
/// failure queues the write, so the guest's store retires while the work it rang
/// for does not start until this returns. A driven boot measures tranches of
/// 18-43 ms and ~105 doorbells a second applied at least one whole frame late
/// (`gfx_doorbell_delay`), so capping the tranche is an obvious thing to reach
/// for. It was reached for, measured, and reverted.
///
/// A budget of one frame interval, checked between child channels using the same
/// requeue the translation-hold and `host_action_yield` arms below already use,
/// **made the delay worse**: mean doorbell age 19.0 ms against 10.8 ms before,
/// and `max_tranche_us` 34-37 ms against 18-22 ms. It fired only ~22 times a
/// second against ~200 tranches, because the cost is not spread across channels
/// — a single channel's flush run holds the lock for tens of milliseconds and a
/// between-channel check cannot reach inside it.
///
/// It also introduced a stall. Returning with `child_mask` still set leaves work
/// that nothing re-arms: a doorbell would, but the guest has no reason to ring
/// one for work it already submitted, and the 4 ms poll publishes *producer*
/// state rather than noticing an already-set mask. One boot froze for 29 s with
/// the census silent and the guest reporting a single 30 263 ms frame. The same
/// gap exists in principle for the translation-hold arm below, which has never
/// been observed to hit it — worth knowing if one ever does.
///
/// The cost is inside the render flush, not in the scheduling around it. See
/// [`crate::runtime::storage_flush::flush_mapping_windows_before_fence`].
///
/// # What that refutation does *not* cover, and the measurement that separates them
///
/// Read as "the doorbell delay is the flush rail's cost, so shrink the flush",
/// the paragraph above overstates itself, and [`DoorbellCensus`] now says by how
/// much. `max_age_us` tracks `max_tranche_us` to within 3 % across a driven
/// boot — 41711/42563, 40627/42117, 103619/105308 — and one of those windows
/// read `duty=0.147`. The worker was idle for 85 % of that second and still held
/// the guest's next submission for a tenth of it, because a doorbell does not
/// wait for the device to be *busy*, it waits for the device to be *holding the
/// lock*. A tranche whose work halves still costs a doorbell half a tranche.
///
/// The thing that was tried and reverted was **pausing** this function: return
/// early on a budget, requeue the rest. That is what made the delay worse and
/// what introduced the 29 s freeze, and it stays refuted.
///
/// A queued write does not need this to pause. It needs its register applied,
/// which costs microseconds and *adds* available work rather than interrupting
/// any — and the queue is drained today only by `lock_for_drain`, once, before
/// the tranche starts. Nothing has measured which registers are in it; the
/// `off_*` breakdown on `gfx_doorbell_delay` is there to answer that first,
/// because "apply it sooner" is only safe for registers whose effect is to
/// publish more work, and unsafe for any the decode below depends on not
/// changing mid-tranche.
/// How many times one tranche will pick up newly rung child channels.
///
/// Not a time budget and not a work budget — a bound on how many times the
/// drain will go back for doorbells that arrived while it was running. Three
/// covers the measured shape: `gfx_doorbell_delay` reads about a hundred rings
/// a second against tranches of tens of milliseconds, so at most a handful of
/// channels are rung during any one pass and a channel already served is
/// excluded from the refill. The cap exists so a guest that rings continuously
/// cannot hold the device lock indefinitely, not because any run has needed it.
const CHILD_DOORBELL_REFILLS: u32 = 3;

/// Move child channels the guest rang lock-free into the drain's pending mask.
///
/// The guest's doorbell write does not take the device lock — see
/// [`crate::model::GfxRegs::child_doorbell_rung`] for why that register can be
/// taken that way and no other can. This is where the bits become work.
///
/// `active_child_mask` is set as well as `pending.child_mask`, because that is
/// what the locked handler in `runtime::mmio` does for the same register and the
/// two must not disagree: `publish_stranded_fifos` re-publishes from
/// `active_child_mask`, so a channel that only ever rang lock-free would be
/// invisible to the stranded-FIFO rescue.
pub(crate) fn fold_rung_child_doorbells(state: &mut DeviceState) {
    let rung = state
        .gfx
        .child_doorbell_rung
        .swap(0, std::sync::atomic::Ordering::AcqRel);
    if rung == 0 {
        return;
    }
    state.active_child_mask |= rung;
    state.pending.child_mask |= rung;
}

pub fn drain_pending<H: HostMemory + HostOps>(state: &mut DeviceState, host: &mut H) {
    // Recorded before the yield check on purpose: the stall reporter separates
    // "the worker never wakes" from "the worker wakes and cannot proceed", and
    // a yield-parked wake is the second shape.
    state.last_drain_wake_ms = crate::observe::elapsed_ms() as u64;
    // A queued present action is part of the ordered device timeline. QEMU
    // cannot paint it while this worker owns the device lock, so later worker
    // wakeups must leave guest work queued until scanout consumes the action.
    if state.pending.host_action_yield {
        return;
    }
    release_translation_order_holds(state);
    // Retry an already translation-held EXEC before allowing either the root
    // FIFO or a sibling child FIFO to overtake it. The guest is free to queue
    // Unmap/Delete immediately after submission; without this boundary the
    // retried EXEC can stage successfully and then write back after its task
    // mapping has been destroyed.
    let deferred = state.translation_deferred_mask;
    if deferred != 0 {
        if state.pending.main_drain {
            note_translation_order_hold(state, TRANSLATION_ROOT_FIFO_BIT);
        }
        let sibling_pending = state.pending.child_mask & !deferred;
        note_translation_order_hold(state, sibling_pending);
        for ch in 1..MAX_CHANNELS as u32 {
            let bit = 1u32 << ch;
            if deferred & bit == 0 {
                continue;
            }
            state.pending.child_mask &= !bit;
            drain_child_fifo(state, host, ch);
        }
        if state.translation_deferred_mask != 0 {
            state.pending.child_mask |= state.translation_deferred_mask;
            return;
        }
        release_translation_order_holds(state);
    }
    if state.pending.main_drain {
        drain_main_fifo(state, host);
    }
    fold_rung_child_doorbells(state);
    let mut mask = state.pending.child_mask;
    state.pending.child_mask = 0;
    // Channels this pass has already run, so a refill cannot re-run one.
    let mut served = 0u32;
    // Bounded refills. Each pass picks up channels the guest rang *while the
    // previous pass was running, under the device lock it could not take* —
    // which is the whole point, and is why the doorbell was worth making
    // lock-free. Serving them here rather than next tranche is what turns a
    // ring into work that starts now.
    //
    // Bounded because the guest can ring faster than this drains, and an
    // unbounded refill would hold the device lock for as long as it kept
    // ringing. Leaving the remainder is safe here in a way it was NOT for the
    // reverted tranche budget: that one returned with `child_mask` set and
    // nothing to re-arm it, and froze a boot for 29 s. Every bit that arrives
    // here arrives with its own `schedule_bh` already rung by the vCPU, so the
    // worker is guaranteed another wakeup for whatever this pass leaves.
    for _ in 0..CHILD_DOORBELL_REFILLS {
        let mut remaining = mask;
        for ch in 1..MAX_CHANNELS as u32 {
            let bit = 1u32 << ch;
            if mask & bit != 0 {
                remaining &= !bit;
                served |= bit;
                drain_child_fifo(state, host, ch);
                if state.translation_deferred_mask != 0 {
                    state.pending.child_mask |= remaining | state.translation_deferred_mask;
                    note_translation_order_hold(state, remaining);
                    return;
                }
                if state.pending.host_action_yield {
                    state.pending.child_mask |= remaining;
                    return;
                }
            }
        }
        fold_rung_child_doorbells(state);
        // Only channels this pass has not already run: a channel rung again
        // while its own drain was in flight has had that work seen, and
        // re-running it here would spin on one busy channel while the others
        // wait.
        mask = std::mem::take(&mut state.pending.child_mask) & !served;
        if mask == 0 {
            break;
        }
        note_store_route("child_doorbell_refill");
    }
    // Whatever the refill cap left, handed back to the next wakeup.
    state.pending.child_mask |= mask;
    if state.pending.iosfc {
        drain_iosfc(state, host);
    }
    try_display_online(state, host);
    // Unmap contiguous views retired by MAP/UNMAP/page-table changes (their
    // Metal objects were dropped at retire time; execution is sync-per-packet
    // so nothing aliases them anymore).
    crate::runtime::mapper::flush_retired_views(state, host);
    // Unpin engine residents of linear cache entries dropped by task/object
    // deletes this drain (they become LRU-evictable instead of leaking).
    crate::runtime::storage_flush::retire_linear_residents(state);
    // Land GVA render-Store windows whose task died this drain (cache-only —
    // the GVA walk went with the task) and unpin their residents.
    crate::runtime::storage_flush::retire_gva_windows(state, host);
}

#[cfg(test)]
mod tests;
