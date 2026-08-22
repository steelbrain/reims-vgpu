//! Child FIFO record layouts, and the two child commands decoded here.
//!
//! This module holds the parts of the FIFO contract the live drain path reads:
//! the resource-list / invalidate / synchronize record layout, the
//! display-descriptor timing entries, and the `EXEC_INDIRECT2` header offsets.
//!
//! The **opcodes** those records belong to are not here. [`crate::model::regs`]
//! holds one table for the whole device — root and child together — and this
//! module used to restate five of its child entries, four of them byte-identical
//! declarations that nothing imported while the drain's dispatch matched on
//! `regs`. A second table of the same numbers cannot be kept honest by anything
//! in the toolchain: a correction to an opcode the RE got wrong reaches whichever
//! copy its author was reading, and the other one keeps compiling.
//!
//! Packet framing itself — reading a ring, walking headers and stamps, writing
//! back head — lives in `runtime/drain/mod.rs`, which does it against live guest
//! memory and reports each failure as a `PacketFault`.

use reims_vgpu_core::endian::{ld32, st16, st32};

// --- child record layout, as the PVG command table numbers them ---

/// CmdInvalidateResources / CmdSynchronizeResources shared header.
pub const CHILD_RESOURCE_LIST_TASK_ID: u32 = 0x00;
pub const CHILD_RESOURCE_LIST_COUNT: u32 = 0x04;
pub const CHILD_RESOURCE_LIST_HEADER_LEN: u32 = 8;
/// Per-object record on Invalidate: `{object_id u32}` + 4 validity-op bytes.
pub const CHILD_INVALIDATE_RECORD_LEN: u32 = 8;
/// Per-object record on Synchronize: `{object_id u32}` only (no validity ops).
pub const CHILD_SYNCHRONIZE_RECORD_LEN: u32 = 4;
/// CmdReplacePhysical (`0x3c`): a fixed `{task_id, object_id}` pair, no list.
pub const CHILD_REPLACE_PHYSICAL_TASK_ID: u32 = 0x00;
pub const CHILD_REPLACE_PHYSICAL_OBJECT_ID: u32 = 0x04;
pub const CHILD_REPLACE_PHYSICAL_LEN: u32 = 8;
/// Hardcoded pageon second dword from `pageBacking` (LE bytes `01 00 00 01`).
///
/// Not a free-form bitfield. PVG host `invalidateResources:` treats the four
/// bytes after `ref` as:
/// `clear_host_valid | set_host_valid | clear_guest_valid | set_guest_valid`.
/// Pageon = clear hostValid + set guestValid (host cache stale; guest pages live).
pub const CHILD_INVALIDATE_PAGEON_FLAGS: u32 = 0x0100_0001;
/// Why a `CmdInvalidateResources` / `CmdSynchronizeResources` payload did not
/// decode.
///
/// These two decoders returned a bare `Option`, and the drain's `None` arm could
/// only say `decode_fail` with the first two words of the payload. Three
/// different conditions reached it and a reader could not tell them apart —
/// which matters here more than in most decoders, because of what each refusal
/// costs. A dropped Invalidate leaves this device serving host-cached pixels for
/// a resource the guest has just CPU-written; a dropped Synchronize lets the
/// guest CPU-read pages whose deferred writeback never landed, which is the
/// class the sibling handler still names "boot-25 black-wallpaper".
///
/// One of those three conditions was **not a malformed payload at all**. A
/// `CHILD_RESOURCE_LIST_MAX_COUNT` of 256 sat above the length check and refused
/// any list longer than that, on the stated basis that "guest pageBacking
/// hardcodes count=1". It bounded nothing: the `Vec::with_capacity(count)` it
/// appeared to protect runs *after* `payload.len() < need`, so the payload's own
/// length is what caps the allocation, and `checked_mul` already covers the
/// arithmetic. Its only effect was to refuse a well-formed 257-entry list the
/// guest is entitled to send. It is gone, and this enum is what remains: the two
/// conditions that are genuinely the payload disagreeing with itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResourceListDecodeError {
    /// Shorter than the `{task_id, count}` header, so the count cannot be read.
    ShortHeader { plen: usize },
    /// The declared count needs more bytes than the payload carries. `need` is
    /// header + `count` × record length, so `need` against `plen` says by how
    /// much, and the record length distinguishes the two commands (8 vs 4).
    Truncated {
        count: u32,
        plen: usize,
        need: usize,
    },
}

impl crate::observe::Decline for ResourceListDecodeError {
    fn slug(&self) -> &'static str {
        match self {
            Self::ShortHeader { .. } => "resource_list_short_header",
            Self::Truncated { .. } => "resource_list_truncated",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match *self {
            Self::ShortHeader { plen } => vec![("plen", plen.to_string())],
            Self::Truncated { count, plen, need } => vec![
                ("count", count.to_string()),
                ("plen", plen.to_string()),
                ("need", need.to_string()),
            ],
        }
    }
}

pub const DISPLAY_DESC_TIMING_BASE: u64 = 0x210;
pub const DISPLAY_DESC_TIMING_STRIDE: u32 = 0x10;
pub const DISPLAY_TIMING_WIDTH: u32 = 0x00;
pub const DISPLAY_TIMING_HEIGHT: u32 = 0x02;
pub const DISPLAY_TIMING_REFRESH: u32 = 0x04;
pub const DISPLAY_TIMING_TAIL0: u32 = 0x08;
pub const DISPLAY_TIMING_TAIL1: u32 = 0x0c;
pub const DISPLAY_TIMING_REFRESH_FRAC_BITS: u32 = 16;

pub const CHILD_EXEC_INDIRECT_TASK_ID: u32 = 0x00;
pub const CHILD_EXEC_INDIRECT_RESOURCE_COUNT: u32 = 0x04;
pub const CHILD_EXEC_INDIRECT_CMDBUF_COUNT: u32 = 0x08;
pub const CHILD_EXEC_INDIRECT_HEADER_LEN: u32 = 12;
pub const CHILD_EXEC_INDIRECT_RESOURCE_DESC_LEN: u32 = 24;
pub const CHILD_EXEC_INDIRECT_CMDBUF_DESC_LEN: u32 = 16;
pub const CHILD_EXEC_INDIRECT_CMDBUF_GVA: u32 = 0x00;
pub const CHILD_EXEC_INDIRECT_CMDBUF_LENGTH: u32 = 0x08;

/// Per-resource descriptor offsets inside the EXEC_INDIRECT2 resource table.
///
/// The queue writes one 24-byte record per live list entry: `{object_id u32}`
/// followed by the same four validity-op bytes a `CmdInvalidateResources`
/// record carries, then 16 trailing bytes it zeroes.
pub const CHILD_EXEC_RESOURCE_OBJECT_ID: u32 = 0x00;
pub const CHILD_EXEC_RESOURCE_VALIDITY_OPS: u32 = 0x04;
pub const CHILD_EXEC_RESOURCE_TAIL: u32 = 0x08;
pub const CHILD_EXEC_RESOURCE_TAIL_LEN: u32 = 16;

// --- display-descriptor timing entries ---

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DisplayTimingEntry {
    pub width: u16,
    pub height: u16,
    pub refresh_1616: u32,
    pub tail0: u32,
    pub tail1: u32,
}

fn bounded_entry_offset_from_base(
    index: u32,
    base: u64,
    byte_capacity: u64,
    entry_len: u32,
) -> Option<u64> {
    if entry_len == 0 {
        return None;
    }
    let entry_offset = index as u64 * entry_len as u64;
    let offset = base.checked_add(entry_offset)?;
    if offset > byte_capacity || byte_capacity - offset < entry_len as u64 {
        None
    } else {
        Some(offset)
    }
}

pub fn display_refresh_hz_1616(refresh_hz: u32) -> Option<u32> {
    if refresh_hz > (u32::MAX >> DISPLAY_TIMING_REFRESH_FRAC_BITS) {
        return None;
    }
    Some(refresh_hz << DISPLAY_TIMING_REFRESH_FRAC_BITS)
}

pub fn encode_display_timing_entry(entry: &DisplayTimingEntry, dst: &mut [u8]) -> bool {
    if dst.len() < DISPLAY_DESC_TIMING_STRIDE as usize {
        return false;
    }
    st16(&mut dst[DISPLAY_TIMING_WIDTH as usize..], entry.width);
    st16(&mut dst[DISPLAY_TIMING_HEIGHT as usize..], entry.height);
    st32(
        &mut dst[DISPLAY_TIMING_REFRESH as usize..],
        entry.refresh_1616,
    );
    st32(&mut dst[DISPLAY_TIMING_TAIL0 as usize..], entry.tail0);
    st32(&mut dst[DISPLAY_TIMING_TAIL1 as usize..], entry.tail1);
    true
}

pub fn display_timing_entry_offset(index: u32, byte_capacity: u64) -> Option<u64> {
    bounded_entry_offset_from_base(
        index,
        DISPLAY_DESC_TIMING_BASE,
        byte_capacity,
        DISPLAY_DESC_TIMING_STRIDE,
    )
}

pub use reims_vgpu_protocol::ResourceValidityOps as InvalidateValidityOps;

/// One CmdInvalidateResources object record (RE: `pageBacking` second `getCommandBytes(8)`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InvalidateResourceRecord {
    pub object_id: u32,
    /// LE dword form of the four validity-op bytes (see [`InvalidateValidityOps`]).
    pub flags: u32,
    pub ops: InvalidateValidityOps,
}

/// FIFO CmdInvalidateResources (0x34) payload (RE pageBacking + live plen=16).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct InvalidateResourcesCommand {
    pub task_id: u32,
    pub count: u32,
    pub records: Vec<InvalidateResourceRecord>,
}

/// FIFO CmdReplacePhysical (0x3c) payload — `{task_id, object_id}`, 8 bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReplacePhysicalCommand {
    pub task_id: u32,
    pub object_id: u32,
}

/// FIFO CmdSynchronizeResources (0x35) payload (RE synchronizeForUnwire + live plen=12).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SynchronizeResourcesCommand {
    pub task_id: u32,
    pub count: u32,
    pub object_ids: Vec<u32>,
}

/// Decode CmdInvalidateResources: header `{task_id, count}` + `count × {object_id, flags}`.
///
/// Guest `pageBacking` always writes count=1 and flags=`0x1000001` on the observed
/// guest driver; the decoder accepts any count the payload is long enough to
/// carry, which is what makes it forward-compatible with a serializer that
/// batches. The payload's length is the only bound — see
/// [`ResourceListDecodeError`] for the cap that used to sit above it and what
/// that cap was actually doing.
pub fn decode_invalidate_resources(
    payload: &[u8],
) -> Result<InvalidateResourcesCommand, ResourceListDecodeError> {
    let plen = payload.len();
    if plen < CHILD_RESOURCE_LIST_HEADER_LEN as usize {
        return Err(ResourceListDecodeError::ShortHeader { plen });
    }
    let task_id = ld32(&payload[CHILD_RESOURCE_LIST_TASK_ID as usize..]);
    let count = ld32(&payload[CHILD_RESOURCE_LIST_COUNT as usize..]);
    let need = resource_list_need(count, CHILD_INVALIDATE_RECORD_LEN);
    if plen < need {
        return Err(ResourceListDecodeError::Truncated { count, plen, need });
    }
    // Bounded by `need <= plen`, which was just checked: a declared count the
    // payload cannot carry never reaches here, so this reserves at most one
    // record per record's worth of bytes the guest actually sent.
    let mut records = Vec::with_capacity(count as usize);
    let mut off = CHILD_RESOURCE_LIST_HEADER_LEN as usize;
    for _ in 0..count {
        let object_id = ld32(&payload[off..]);
        let flags = ld32(&payload[off + 4..]);
        records.push(InvalidateResourceRecord {
            object_id,
            flags,
            ops: InvalidateValidityOps::from_le_dword(flags),
        });
        off += CHILD_INVALIDATE_RECORD_LEN as usize;
    }
    Ok(InvalidateResourcesCommand {
        task_id,
        count,
        records,
    })
}

/// Bytes a resource-list payload must carry for the count it declares.
///
/// Saturating rather than checked: `count` is a `u32` and `record_len` is 4 or
/// 8, so the product fits `u64` and the sum cannot wrap — but the `as usize`
/// that follows would truncate on a 32-bit host, and a truncated `need` reads as
/// a payload that is long enough. Saturating there means an impossible length
/// stays impossible instead of becoming a small one.
///
/// One spelling for both commands, because the only thing that differs between
/// them is `record_len`, and a second copy of this arithmetic is how the two
/// would come to disagree about what "long enough" means.
fn resource_list_need(count: u32, record_len: u32) -> usize {
    let bytes = u64::from(count).saturating_mul(u64::from(record_len));
    let total = bytes.saturating_add(u64::from(CHILD_RESOURCE_LIST_HEADER_LEN));
    usize::try_from(total).unwrap_or(usize::MAX)
}

/// One entry of the per-resource table an `EXEC_INDIRECT2` payload carries
/// between its 12-byte header and its command-buffer descriptors.
///
/// The guest builds this in `writeInvalidates`, one record per live entry of the
/// submission's `AppleParavirtSegmentResourceList`. The first eight bytes are
/// byte-identical in layout to a `CmdInvalidateResources` record, so [`ops`]
/// comes off the same [`InvalidateValidityOps`] decoder — the record *lengths*
/// differ (8 vs 24), the quad does not.
///
/// `clear_host_valid` is sourced from `AppleParavirtResource::shouldInvalidateHost()`,
/// which is a `lock btr` test-and-clear of the resource's dirty bit plus a sticky
/// flag it also clears. `writeInvalidates` is its only caller, so the guest's
/// statement that it CPU-wrote a resource is delivered here exactly once and is
/// never resent.
///
/// [`ops`]: Self::ops
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExecResourceDesc {
    pub object_id: u32,
    pub ops: InvalidateValidityOps,
    /// Bytes `+0x08..0x18`. Zeroed by the Ventura 13.7.8 x86 guest driver; kept
    /// raw rather than dropped so a build that populates them is visible instead
    /// of silently discarded.
    pub tail: [u8; CHILD_EXEC_RESOURCE_TAIL_LEN as usize],
}

impl ExecResourceDesc {
    /// How many of the 16 trailing bytes this record actually sets.
    pub fn tail_nonzero_bytes(&self) -> u32 {
        self.tail.iter().filter(|b| **b != 0).count() as u32
    }
}

/// Decode the `EXEC_INDIRECT2` resource table: `resource_count` records of
/// [`CHILD_EXEC_INDIRECT_RESOURCE_DESC_LEN`] bytes, starting right after the
/// header.
///
/// `None` when the payload is shorter than the count it declares — the same
/// refusal shape as the other list decoders here, so a malformed or truncated
/// submission is a caller-visible failure rather than a partial table. The
/// bound is checked before the allocation, so a hostile `resource_count`
/// cannot reserve memory the payload does not back.
pub fn decode_exec_resource_table(payload: &[u8]) -> Option<Vec<ExecResourceDesc>> {
    if payload.len() < CHILD_EXEC_INDIRECT_HEADER_LEN as usize {
        return None;
    }
    let count = ld32(&payload[CHILD_EXEC_INDIRECT_RESOURCE_COUNT as usize..]);
    let table_len = (count as u64).checked_mul(CHILD_EXEC_INDIRECT_RESOURCE_DESC_LEN as u64)?;
    let need = (CHILD_EXEC_INDIRECT_HEADER_LEN as u64).checked_add(table_len)?;
    if need > payload.len() as u64 {
        return None;
    }
    let mut descs = Vec::with_capacity(count as usize);
    let mut off = CHILD_EXEC_INDIRECT_HEADER_LEN as usize;
    for _ in 0..count {
        let object_id = ld32(&payload[off + CHILD_EXEC_RESOURCE_OBJECT_ID as usize..]);
        let flags = ld32(&payload[off + CHILD_EXEC_RESOURCE_VALIDITY_OPS as usize..]);
        let tail_off = off + CHILD_EXEC_RESOURCE_TAIL as usize;
        let mut tail = [0u8; CHILD_EXEC_RESOURCE_TAIL_LEN as usize];
        tail.copy_from_slice(&payload[tail_off..tail_off + CHILD_EXEC_RESOURCE_TAIL_LEN as usize]);
        descs.push(ExecResourceDesc {
            object_id,
            ops: InvalidateValidityOps::from_le_dword(flags),
            tail,
        });
        off += CHILD_EXEC_INDIRECT_RESOURCE_DESC_LEN as usize;
    }
    Some(descs)
}

/// Decode CmdReplacePhysical (`0x3c`): `{task_id, object_id}`, 8 bytes.
///
/// The guest emits this once per attached resource at the tail of a re-commit
/// into the GPU page table — that is, after the range was released, its pages
/// were wired to different host frames, and the new PFNs were written back at
/// the *same* GPU-VA. It therefore carries no address of its own: the GVA is
/// unchanged, and only the translation behind it moved.
///
/// `task_id` is a plain slot id, as the other resource-list commands carry it,
/// and not the doubled `DefineTask2` word.
pub fn decode_replace_physical(payload: &[u8]) -> Option<ReplacePhysicalCommand> {
    if payload.len() < CHILD_REPLACE_PHYSICAL_LEN as usize {
        return None;
    }
    Some(ReplacePhysicalCommand {
        task_id: ld32(&payload[CHILD_REPLACE_PHYSICAL_TASK_ID as usize..]),
        object_id: ld32(&payload[CHILD_REPLACE_PHYSICAL_OBJECT_ID as usize..]),
    })
}

/// Decode CmdSynchronizeResources: header `{task_id, count}` + `count × {object_id}`.
///
/// Guest `synchronizeForUnwire` uses `getCommandBytes(4)` for the object cell (no flags).
pub fn decode_synchronize_resources(
    payload: &[u8],
) -> Result<SynchronizeResourcesCommand, ResourceListDecodeError> {
    let plen = payload.len();
    if plen < CHILD_RESOURCE_LIST_HEADER_LEN as usize {
        return Err(ResourceListDecodeError::ShortHeader { plen });
    }
    let task_id = ld32(&payload[CHILD_RESOURCE_LIST_TASK_ID as usize..]);
    let count = ld32(&payload[CHILD_RESOURCE_LIST_COUNT as usize..]);
    let need = resource_list_need(count, CHILD_SYNCHRONIZE_RECORD_LEN);
    if plen < need {
        return Err(ResourceListDecodeError::Truncated { count, plen, need });
    }
    // Bounded by the length check above; see `decode_invalidate_resources`.
    let mut object_ids = Vec::with_capacity(count as usize);
    let mut off = CHILD_RESOURCE_LIST_HEADER_LEN as usize;
    for _ in 0..count {
        object_ids.push(ld32(&payload[off..]));
        off += CHILD_SYNCHRONIZE_RECORD_LEN as usize;
    }
    Ok(SynchronizeResourcesCommand {
        task_id,
        count,
        object_ids,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use reims_vgpu_core::endian::st32;

    /// RE pageBacking: plen=16 = header + one 8-byte record; LE `01 00 00 01`
    /// = clear_host_valid + set_guest_valid (PVG validity-op bytes).
    #[test]
    fn decode_invalidate_pageon_shape() {
        let mut p = [0u8; 16];
        st32(&mut p[0..], 0); // task
        st32(&mut p[4..], 1); // count
        st32(&mut p[8..], 0x2a); // object_id
        st32(&mut p[12..], CHILD_INVALIDATE_PAGEON_FLAGS);
        let c = decode_invalidate_resources(&p).expect("decode");
        assert_eq!(c.task_id, 0);
        assert_eq!(c.count, 1);
        assert_eq!(c.records.len(), 1);
        assert_eq!(c.records[0].object_id, 0x2a);
        assert_eq!(c.records[0].flags, CHILD_INVALIDATE_PAGEON_FLAGS);
        assert_eq!(c.records[0].ops, InvalidateValidityOps::PAGE_ON);
        assert_eq!(c.records[0].ops.clear_host_valid, 1);
        assert_eq!(c.records[0].ops.set_host_valid, 0);
        assert_eq!(c.records[0].ops.clear_guest_valid, 0);
        assert_eq!(c.records[0].ops.set_guest_valid, 1);
        // LE memory: not bit0|bit24 as independent product bits.
        assert_eq!(CHILD_INVALIDATE_PAGEON_FLAGS.to_le_bytes(), [1, 0, 0, 1]);
    }

    #[test]
    fn invalidate_validity_ops_roundtrip() {
        let ops = InvalidateValidityOps {
            clear_host_valid: 1,
            set_host_valid: 0,
            clear_guest_valid: 1,
            set_guest_valid: 0,
        };
        assert_eq!(InvalidateValidityOps::from_le_dword(ops.to_le_dword()), ops);
    }

    /// RE synchronizeForUnwire: plen=12 = header + one u32 object_id.
    #[test]
    fn decode_synchronize_unwire_shape() {
        let mut p = [0u8; 12];
        st32(&mut p[0..], 4); // task
        st32(&mut p[4..], 1);
        st32(&mut p[8..], 99);
        let c = decode_synchronize_resources(&p).expect("decode");
        assert_eq!(c.task_id, 4);
        assert_eq!(c.count, 1);
        assert_eq!(c.object_ids, vec![99]);
    }

    #[test]
    fn decode_invalidate_multi_object_when_payload_long() {
        let mut p = [0u8; 8 + 16];
        st32(&mut p[0..], 1);
        st32(&mut p[4..], 2);
        st32(&mut p[8..], 10);
        st32(&mut p[12..], 0x1000001);
        st32(&mut p[16..], 11);
        st32(&mut p[20..], 0x1000001);
        let c = decode_invalidate_resources(&p).expect("decode");
        assert_eq!(c.count, 2);
        assert_eq!(c.records[0].object_id, 10);
        assert_eq!(c.records[1].object_id, 11);
    }

    /// The three descriptor offsets must tile the stride `exec` uses to skip
    /// the table. If they ever disagree, the decoded records and the cmdbuf
    /// section would be read from different places in the same payload.
    #[test]
    fn exec_resource_desc_offsets_tile_the_stride() {
        assert_eq!(CHILD_EXEC_RESOURCE_OBJECT_ID, 0);
        assert_eq!(CHILD_EXEC_RESOURCE_VALIDITY_OPS, 4);
        assert_eq!(
            CHILD_EXEC_RESOURCE_TAIL + CHILD_EXEC_RESOURCE_TAIL_LEN,
            CHILD_EXEC_INDIRECT_RESOURCE_DESC_LEN
        );
    }

    /// Build an EXEC_INDIRECT2 payload with `descs` resource records and no
    /// command buffers.
    fn exec_payload_with_table(descs: &[(u32, u32, [u8; 16])]) -> Vec<u8> {
        let mut p = vec![
            0u8;
            CHILD_EXEC_INDIRECT_HEADER_LEN as usize
                + descs.len() * CHILD_EXEC_INDIRECT_RESOURCE_DESC_LEN as usize
        ];
        st32(
            &mut p[CHILD_EXEC_INDIRECT_RESOURCE_COUNT as usize..],
            descs.len() as u32,
        );
        for (i, (id, flags, tail)) in descs.iter().enumerate() {
            let off = CHILD_EXEC_INDIRECT_HEADER_LEN as usize
                + i * CHILD_EXEC_INDIRECT_RESOURCE_DESC_LEN as usize;
            st32(&mut p[off + CHILD_EXEC_RESOURCE_OBJECT_ID as usize..], *id);
            st32(
                &mut p[off + CHILD_EXEC_RESOURCE_VALIDITY_OPS as usize..],
                *flags,
            );
            let t = off + CHILD_EXEC_RESOURCE_TAIL as usize;
            p[t..t + CHILD_EXEC_RESOURCE_TAIL_LEN as usize].copy_from_slice(tail);
        }
        p
    }

    /// RE writeInvalidates: `{object_id}` + the same validity quad an
    /// invalidate record carries, at stride 24 with 16 trailing bytes.
    #[test]
    fn decode_exec_resource_table_reads_id_quad_and_tail() {
        let mut tail = [0u8; 16];
        tail[0] = 0xaa;
        tail[15] = 0x01;
        let p =
            exec_payload_with_table(&[(0x2a, 0x0000_0001, [0u8; 16]), (0x2b, 0x0000_0100, tail)]);
        let descs = decode_exec_resource_table(&p).expect("decode");
        assert_eq!(descs.len(), 2);
        assert_eq!(descs[0].object_id, 0x2a);
        assert_eq!(descs[0].ops.clear_host_valid, 1);
        assert_eq!(descs[0].ops.set_host_valid, 0);
        assert_eq!(descs[0].tail_nonzero_bytes(), 0);
        assert_eq!(descs[1].object_id, 0x2b);
        assert_eq!(descs[1].ops.clear_host_valid, 0);
        assert_eq!(descs[1].ops.set_host_valid, 1);
        assert_eq!(descs[1].tail_nonzero_bytes(), 2);
    }

    /// The quad decoder is shared with `CmdInvalidateResources`; only the record
    /// length differs. A second decoder for the same four bytes would be a
    /// second place for the field order to drift.
    #[test]
    fn exec_table_and_invalidate_record_decode_the_same_quad() {
        let p = exec_payload_with_table(&[(7, CHILD_INVALIDATE_PAGEON_FLAGS, [0u8; 16])]);
        let descs = decode_exec_resource_table(&p).expect("decode");
        assert_eq!(descs[0].ops, InvalidateValidityOps::PAGE_ON);
    }

    #[test]
    fn decode_exec_resource_table_empty_when_count_zero() {
        let p = exec_payload_with_table(&[]);
        assert_eq!(decode_exec_resource_table(&p).expect("decode").len(), 0);
    }

    /// `resource_count` is guest-controlled. A count the payload cannot back
    /// must refuse, not read past the buffer and not reserve for records that
    /// are not there.
    #[test]
    fn decode_exec_resource_table_rejects_count_the_payload_cannot_back() {
        let mut p = exec_payload_with_table(&[(1, 0, [0u8; 16])]);
        st32(&mut p[CHILD_EXEC_INDIRECT_RESOURCE_COUNT as usize..], 2);
        assert!(decode_exec_resource_table(&p).is_none());
        st32(
            &mut p[CHILD_EXEC_INDIRECT_RESOURCE_COUNT as usize..],
            u32::MAX,
        );
        assert!(decode_exec_resource_table(&p).is_none());
    }

    #[test]
    fn decode_exec_resource_table_rejects_short_header() {
        assert!(decode_exec_resource_table(&[0u8; 4]).is_none());
    }

    #[test]
    fn decode_invalidate_rejects_short_for_count() {
        let mut p = [0u8; 12]; // header claims count=2 but only 4 payload bytes
        st32(&mut p[0..], 0);
        st32(&mut p[4..], 2);
        st32(&mut p[8..], 1);
        assert_eq!(
            decode_invalidate_resources(&p),
            Err(ResourceListDecodeError::Truncated {
                count: 2,
                plen: 12,
                need: 24,
            }),
            "a count the payload cannot carry must name itself, not share a \
             verdict with a short header"
        );
        assert_eq!(
            decode_invalidate_resources(&[0u8; 4]),
            Err(ResourceListDecodeError::ShortHeader { plen: 4 })
        );
    }

    /// A list longer than the retired `CHILD_RESOURCE_LIST_MAX_COUNT` decodes.
    ///
    /// That cap was 256, it sat above the length check, and it bounded nothing
    /// the length check did not already bound — so its only effect was to refuse
    /// a well-formed list. Refusing an Invalidate leaves this device serving
    /// host-cached pixels for a resource the guest has just CPU-written, so the
    /// cost of that refusal was not a lost log line.
    ///
    /// 257 rather than a round number: one past the retired bound, so a
    /// reinstated cap of any plausible size fails here.
    #[test]
    fn a_list_longer_than_the_retired_cap_decodes_when_the_payload_carries_it() {
        const COUNT: u32 = 257;
        let mut p = vec![
            0u8;
            CHILD_RESOURCE_LIST_HEADER_LEN as usize
                + COUNT as usize * CHILD_INVALIDATE_RECORD_LEN as usize
        ];
        st32(&mut p[0..], 3);
        st32(&mut p[4..], COUNT);
        for i in 0..COUNT as usize {
            let at =
                CHILD_RESOURCE_LIST_HEADER_LEN as usize + i * CHILD_INVALIDATE_RECORD_LEN as usize;
            st32(&mut p[at..], 0x100 + i as u32);
            st32(&mut p[at + 4..], CHILD_INVALIDATE_PAGEON_FLAGS);
        }
        let c = decode_invalidate_resources(&p).expect("the payload carries every record");
        assert_eq!(c.task_id, 3);
        assert_eq!(c.records.len(), COUNT as usize);
        assert_eq!(c.records[256].object_id, 0x100 + 256);

        // Same for the sibling command, whose records are 4 bytes rather than 8
        // — the one thing that differs between the two decoders.
        let mut s = vec![
            0u8;
            CHILD_RESOURCE_LIST_HEADER_LEN as usize
                + COUNT as usize * CHILD_SYNCHRONIZE_RECORD_LEN as usize
        ];
        st32(&mut s[4..], COUNT);
        let c = decode_synchronize_resources(&s).expect("the payload carries every record");
        assert_eq!(c.object_ids.len(), COUNT as usize);
    }

    /// The length a count needs cannot wrap into a small number.
    ///
    /// `u32::MAX` records of 8 bytes is 32 GiB, which no payload carries — the
    /// point is that it stays impossible. Computed in `u64` and saturated into
    /// `usize`, so a 32-bit host cannot truncate `need` into something a short
    /// payload satisfies, which would turn a bound into an out-of-range read.
    #[test]
    fn a_count_that_cannot_fit_reports_truncated_rather_than_wrapping() {
        let mut p = [0u8; 16];
        st32(&mut p[4..], u32::MAX);
        match decode_invalidate_resources(&p) {
            Err(ResourceListDecodeError::Truncated { count, plen, need }) => {
                assert_eq!((count, plen), (u32::MAX, 16));
                assert!(need > plen, "need={need} must exceed the payload");
            }
            other => panic!("expected Truncated, got {other:?}"),
        }
    }
}
