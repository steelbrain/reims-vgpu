//! Host-neutral IOSurface mapper-internal page-plan resolution.

use crate::geometry::{
    mapper_entry_gpa as entry_gpa_shift, span_page_count as span_page_count_shift,
};
use alloc::vec::Vec;

fn ld32(bytes: &[u8]) -> u32 {
    u32::from_le_bytes(
        bytes[..4]
            .try_into()
            .expect("the caller supplies four bytes"),
    )
}

fn ld64(bytes: &[u8]) -> u64 {
    u64::from_le_bytes(
        bytes[..8]
            .try_into()
            .expect("the caller supplies eight bytes"),
    )
}

fn checked_add_u64(a: u64, b: u64) -> Option<u64> {
    a.checked_add(b)
}

pub const U32_SIZE: usize = 4;

pub const MAPPING_INTERNAL_BACKPTR: u64 = 0x18;
pub const MAPPING_INTERNAL_ID: u64 = 0x30;
pub const MAPPING_INTERNAL_DESC_PTR: u64 = 0x38;
pub const MAPPING_INTERNAL_SIZE: u64 = 0x40;
pub const MAPPING_INTERNAL_EXPECTED_SIZE: u32 = 0x200;
pub const MAPPING_INTERNAL_PAGE_FIELD_48: u64 = 0x48;
pub const MAPPING_INTERNAL_PAGE_FIELD_50: u64 = 0x50;
pub const MAPPING_INTERNAL_PAGE_COUNT: u64 = 0x70;
pub const MAPPING_PAGE_TABLE_FROM_F48: u64 = 0xb8;
pub const MAPPING_PAGE_TABLE_FROM_F50: u64 = 0x28;

pub const ARM_KERNEL_VA_MASK: u64 = 0xffffff00_00000000;
pub const ARM_KERNEL_VA_BASE: u64 = 0xfffffe00_00000000;
/// x86_64 Darwin canonical kernel half (bits 63:47 all ones in 48-bit VA).
pub const X86_KERNEL_VA_MIN: u64 = 0xffff8000_00000000;

/// Why a mapper/page-table resolve refused, or [`Status::Ok`] if it did not.
///
/// Every variant here names a check this walker actually performs. `ErrArgs`,
/// `ErrOverflow` and `ErrSpanRange` used to sit alongside them and were never
/// constructed: the argument checks are `ErrShortDescriptor`, the arithmetic
/// goes through `checked_*` helpers that fold overflow into the length and
/// page-count refusals, and the span walk refuses as `ErrPageCount`. A refusal
/// class the code cannot reach is a claim the fail log can never substantiate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Ok,
    ErrShortDescriptor(&'static str),
    ErrNotKernelVa(&'static str),
    ErrInternalRead(&'static str),
    ErrInternalOwner(&'static str),
    ErrInternalMappingId(&'static str),
    ErrInternalSize(&'static str),
    ErrInternalFields(&'static str),
    ErrPageCount(&'static str),
    ErrPageTableRead(&'static str),
    ErrPageEntry(&'static str),
    ErrNoPageTable(&'static str),
}

/// Memory access callbacks for mapper/page-table reads.
pub trait PagesMemory {
    fn read(&self, address: u64, dst: &mut [u8]) -> bool;
    fn is_kernel_va(&self, address: u64) -> bool {
        guest_kernel_va(address)
    }
    fn is_ram_gpa(&self, _address: u64) -> bool {
        true
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MapperInternalFields {
    pub internal_kva: u64,
    pub has_mapper_device: bool,
    pub mapper_device_kva: u64,
    pub owner_kva: u64,
    pub mapping_id: u32,
    pub internal_size: u32,
    pub page_field_48: u64,
    pub page_field_50: u64,
    pub raw_page_count: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PageTablePlan {
    pub entries: Vec<u32>,
    pub page_table_kva: u64,
    pub min_size: u64,
    pub required_pages: u64,
    /// Whether the `MappingInternal` field this plan did *not* come through was
    /// populated as well.
    ///
    /// [`build_table_plan`] used to chase `+0x48` then `+0xb8`, and `+0x50`
    /// then `+0x28`, returning the entries of whichever parsed first — the
    /// classic "try both, keep the one that works" ladder. Two driven arm64
    /// boots retired it; see that function. This is what is left of the
    /// measurement, and it stays because it is free: `contract` stays clear of
    /// the observability dependency, so the fact travels in the plan rather
    /// than being emitted here.
    pub candidates: CandidateOutcome,
}

/// What the unused `MappingInternal` page field held on one successful plan.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CandidateOutcome {
    /// `+0x50` held a kernel VA as well as `+0x48`, which the plan came
    /// through.
    ///
    /// Measured `true` on every one of 223 successful resolves across two
    /// driven arm64 workloads, which is *why* the second chase could go: the
    /// field is populated essentially always, so it never discriminated
    /// anything. A run where this turned mostly `false` would mean the two
    /// fields really are two layouts and the deletion was wrong.
    pub other_field_populated: bool,
}

pub fn arm_kernel_va(address: u64) -> bool {
    (address & ARM_KERNEL_VA_MASK) == ARM_KERNEL_VA_BASE
}

pub fn x86_kernel_va(address: u64) -> bool {
    address >= X86_KERNEL_VA_MIN
}

/// Guest kernel VA: arm64e TTBR1 window **or** x86_64 Darwin high half.
pub fn guest_kernel_va(address: u64) -> bool {
    arm_kernel_va(address) || x86_kernel_va(address)
}

pub fn required_entry_count(
    fields: &MapperInternalFields,
    min_size: u64,
    page_shift: u32,
) -> Result<u32, Status> {
    let pages64 = fields.raw_page_count;
    let required_pages = span_page_count_shift(min_size, page_shift);
    // Guest page count is authoritative — no product 4096-page ceiling.
    // Fail only on zero, span coverage, or host-unaddressable entry vectors
    // (process addressability for `Vec<u32>` of entries — not a MiB budget).
    if pages64 == 0 || pages64 < required_pages || pages64 > u32::MAX as u64 {
        return Err(Status::ErrPageCount("iosurface_page_count_invalid"));
    }
    let entry_bytes = pages64.saturating_mul(4);
    if usize::try_from(entry_bytes)
        .ok()
        .filter(|&n| n <= isize::MAX as usize)
        .is_none()
    {
        return Err(Status::ErrPageCount(
            "iosurface_page_count_host_addressability",
        ));
    }
    Ok(pages64 as u32)
}

fn read_u32(mem: &dyn PagesMemory, address: u64) -> Option<u32> {
    let mut bytes = [0u8; 4];
    if address > u64::MAX - 3 {
        return None;
    }
    if !mem.read(address, &mut bytes) {
        return None;
    }
    Some(ld32(&bytes))
}

fn read_u64(mem: &dyn PagesMemory, address: u64) -> Option<u64> {
    let mut bytes = [0u8; 8];
    if address > u64::MAX - 7 {
        return None;
    }
    if !mem.read(address, &mut bytes) {
        return None;
    }
    Some(ld64(&bytes))
}

fn read_u32_at(mem: &dyn PagesMemory, base: u64, offset: u64) -> Option<u32> {
    let address = checked_add_u64(base, offset)?;
    read_u32(mem, address)
}

fn read_u64_at(mem: &dyn PagesMemory, base: u64, offset: u64) -> Option<u64> {
    let address = checked_add_u64(base, offset)?;
    read_u64(mem, address)
}

pub fn read_mapper_identity(
    mem: &dyn PagesMemory,
    internal_kva: u64,
    has_mapper_device: bool,
    mapper_device_kva: u64,
) -> Result<MapperInternalFields, Status> {
    if !mem.is_kernel_va(internal_kva) {
        return Err(Status::ErrNotKernelVa(
            "iosurface_mapper_internal_kva_invalid",
        ));
    }
    if has_mapper_device && !mem.is_kernel_va(mapper_device_kva) {
        return Err(Status::ErrNotKernelVa(
            "iosurface_mapper_device_kva_invalid",
        ));
    }
    let owner_kva = read_u64_at(mem, internal_kva, MAPPING_INTERNAL_BACKPTR).ok_or(
        Status::ErrInternalRead("iosurface_mapper_internal_owner_read"),
    )?;
    let mapping_id = read_u32_at(mem, internal_kva, MAPPING_INTERNAL_ID).ok_or(
        Status::ErrInternalRead("iosurface_mapper_internal_mapping_id_read"),
    )?;
    let internal_size = read_u32_at(mem, internal_kva, MAPPING_INTERNAL_SIZE).ok_or(
        Status::ErrInternalRead("iosurface_mapper_internal_size_read"),
    )?;
    Ok(MapperInternalFields {
        internal_kva,
        has_mapper_device,
        mapper_device_kva,
        owner_kva,
        mapping_id,
        internal_size,
        page_field_48: 0,
        page_field_50: 0,
        raw_page_count: 0,
    })
}

pub fn read_mapper_internal(
    mem: &dyn PagesMemory,
    internal_kva: u64,
    has_mapper_device: bool,
    mapper_device_kva: u64,
) -> Result<MapperInternalFields, Status> {
    let mut fields = read_mapper_identity(mem, internal_kva, has_mapper_device, mapper_device_kva)?;
    fields.page_field_48 = read_u64_at(mem, internal_kva, MAPPING_INTERNAL_PAGE_FIELD_48).ok_or(
        Status::ErrInternalRead("iosurface_mapper_page_field_48_read"),
    )?;
    fields.page_field_50 = read_u64_at(mem, internal_kva, MAPPING_INTERNAL_PAGE_FIELD_50).ok_or(
        Status::ErrInternalRead("iosurface_mapper_page_field_50_read"),
    )?;
    fields.raw_page_count = read_u64_at(mem, internal_kva, MAPPING_INTERNAL_PAGE_COUNT)
        .ok_or(Status::ErrInternalRead("iosurface_mapper_page_count_read"))?;
    Ok(fields)
}

pub fn read_internal_desc_ptr(mem: &dyn PagesMemory, internal_kva: u64) -> Result<u64, Status> {
    let desc_kva = read_u64_at(mem, internal_kva, MAPPING_INTERNAL_DESC_PTR).ok_or(
        Status::ErrInternalRead("iosurface_mapper_device_desc_pointer_read"),
    )?;
    if desc_kva == 0 {
        return Err(Status::ErrInternalFields(
            "iosurface_mapper_device_desc_pointer_zero",
        ));
    }
    if !mem.is_kernel_va(desc_kva) {
        return Err(Status::ErrInternalFields(
            "iosurface_mapper_device_desc_pointer_invalid",
        ));
    }
    Ok(desc_kva)
}

pub fn validate_mapper_internal(
    mem: &dyn PagesMemory,
    expected_mapping_id: u32,
    fields: &MapperInternalFields,
) -> Status {
    if !mem.is_kernel_va(fields.internal_kva) {
        return Status::ErrNotKernelVa("iosurface_validate_internal_kva_invalid");
    }
    if fields.mapping_id != expected_mapping_id {
        return Status::ErrInternalMappingId("iosurface_validate_mapping_id_mismatch");
    }
    if fields.internal_size != MAPPING_INTERNAL_EXPECTED_SIZE {
        return Status::ErrInternalSize("iosurface_validate_internal_size_mismatch");
    }
    if fields.has_mapper_device {
        if !mem.is_kernel_va(fields.mapper_device_kva) {
            return Status::ErrNotKernelVa("iosurface_validate_mapper_device_kva_invalid");
        }
        if fields.owner_kva != fields.mapper_device_kva {
            return Status::ErrInternalOwner("iosurface_validate_internal_owner_mismatch");
        }
    }
    Status::Ok
}

fn read_table_entries(
    mem: &dyn PagesMemory,
    table_kva: u64,
    pages: u32,
    page_shift: u32,
) -> Result<Vec<u32>, Status> {
    let mut entries = Vec::with_capacity(pages as usize);
    for i in 0..pages {
        let entry = read_u32_at(mem, table_kva, (i as u64) * U32_SIZE as u64)
            .ok_or(Status::ErrPageTableRead("iosurface_page_table_entry_read"))?;
        let gpa = entry_gpa_shift(entry, page_shift)
            .ok_or(Status::ErrPageEntry("iosurface_page_table_entry_invalid"))?;
        if !mem.is_ram_gpa(gpa) {
            return Err(Status::ErrPageEntry("iosurface_page_table_gpa_not_ram"));
        }
        entries.push(entry);
    }
    Ok(entries)
}

pub fn build_table_plan(
    mem: &dyn PagesMemory,
    expected_mapping_id: u32,
    fields: &MapperInternalFields,
    min_size: u64,
    page_shift: u32,
) -> Result<PageTablePlan, Status> {
    let st = validate_mapper_internal(mem, expected_mapping_id, fields);
    if st != Status::Ok {
        return Err(st);
    }
    let field_48_populated = mem.is_kernel_va(fields.page_field_48);
    let field_50_populated = mem.is_kernel_va(fields.page_field_50);
    if !field_48_populated && !field_50_populated {
        return Err(Status::ErrInternalFields(
            "iosurface_page_table_fields_invalid",
        ));
    }
    let required_pages = span_page_count_shift(min_size, page_shift);
    let pages = required_entry_count(fields, min_size, page_shift)?;

    // One chase, `+0x48` then `+0xb8`.
    //
    // There used to be a second, `+0x50` then `+0x28`, with the entries of
    // whichever parsed first being returned — a "try both, keep the one that
    // works" ladder, which this project refuses on principle but could not
    // refuse here without evidence, because the alternative reading was that
    // the two fields are two layouts handled side by side.
    //
    // Two driven arm64 boots settled it, over **223 successful resolves** on
    // deliberately different workloads (Safari plus window drags; Finder view
    // switching, System Settings panes, Mission Control and window resizes).
    // Both fields held a kernel VA on every one of them, so the branch really
    // did choose rather than dispatch — and `+0x48` won every one, with the
    // second chase never once carrying a resolve that the first had failed.
    // A branch that chooses, and always chooses the same way, is a fallback.
    //
    // What replaces the fallback is loudness. Every way the `+0x48` chase can
    // fail now reaches `mapper_resolve_fail` under its own slug instead of
    // being silently rescued by a rail nothing has confirmed, and
    // [`CandidateOutcome::other_field_populated`] keeps measuring the premise.
    // The field test above stays: a mapping with only `+0x50` set is still
    // *detected*, and refused by name rather than resolved through the
    // unconfirmed path.
    if !field_48_populated {
        return Err(Status::ErrNoPageTable("iosurface_page_table_only_field_50"));
    }
    let table_kva = match read_u64_at(mem, fields.page_field_48, MAPPING_PAGE_TABLE_FROM_F48) {
        Some(v) if mem.is_kernel_va(v) => v,
        Some(_) => {
            return Err(Status::ErrNoPageTable(
                "iosurface_page_table_pointer_48_invalid",
            ))
        }
        None => {
            return Err(Status::ErrPageTableRead(
                "iosurface_page_table_pointer_48_read",
            ))
        }
    };
    let entries = read_table_entries(mem, table_kva, pages, page_shift)?;
    Ok(PageTablePlan {
        entries,
        page_table_kva: table_kva,
        min_size,
        required_pages,
        candidates: CandidateOutcome {
            other_field_populated: field_50_populated,
        },
    })
}
