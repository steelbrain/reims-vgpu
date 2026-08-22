//! Cross-owner IOSurface fixtures retained while their implementations live in
//! protocol and paging crates.

#[cfg(test)]
use reims_vgpu_paging::geometry::{
    mapper_entry_gpa as entry_gpa_shift, span_page_count as span_page_count_shift,
    MAPPER_PAGE_ENTRY_PFN_SHIFT as PAGE_ENTRY_PFN_SHIFT,
    MAPPER_PAGE_ENTRY_VALID as PAGE_ENTRY_VALID,
};
#[cfg(test)]
use reims_vgpu_paging::mapper::*;
#[cfg(test)]
use reims_vgpu_protocol::iosurface::*;
#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{PAGE_SHIFT_ARM64E, PAGE_SIZE_ARM64E};
    use crate::runtime::mapper::MapperStatusRefusal;
    use reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM;

    use std::collections::HashMap;

    fn refusal(status: &Status) -> Option<&'static str> {
        crate::observe::Refusal::refusal(&MapperStatusRefusal(status))
    }

    struct MapMem {
        map: HashMap<u64, u8>,
    }
    impl MapMem {
        fn new() -> Self {
            Self {
                map: HashMap::new(),
            }
        }
        fn put_u32(&mut self, a: u64, v: u32) {
            for (i, b) in v.to_le_bytes().iter().enumerate() {
                self.map.insert(a + i as u64, *b);
            }
        }
        fn put_u64(&mut self, a: u64, v: u64) {
            for (i, b) in v.to_le_bytes().iter().enumerate() {
                self.map.insert(a + i as u64, *b);
            }
        }
    }
    impl PagesMemory for MapMem {
        fn read(&self, address: u64, dst: &mut [u8]) -> bool {
            for (i, s) in dst.iter_mut().enumerate() {
                match self.map.get(&(address + i as u64)) {
                    Some(b) => *s = *b,
                    None => return false,
                }
            }
            true
        }
        fn is_kernel_va(&self, address: u64) -> bool {
            arm_kernel_va(address)
        }
    }

    #[test]
    fn status_refusal_separates_control_flow_from_exact_failures() {
        assert_eq!(refusal(&Status::Ok), None);
        assert!(
            crate::observe::Emit::refusal(
                "mapper_resolve_fail",
                &MapperStatusRefusal(&Status::Ok),
            )
            .is_none(),
            "success must not be representable as a failure line"
        );

        let request = Status::ErrShortDescriptor("iosurface_mapper_request_short");
        assert_eq!(refusal(&request), Some("iosurface_mapper_request_short"));
        assert_eq!(
            crate::observe::Emit::refusal("mapper_resolve_fail", &MapperStatusRefusal(&request),)
                .unwrap()
                .field("mapping", 7)
                .render(),
            "mapper_resolve_fail reason=iosurface_mapper_request_short \
             class=short_descriptor mapping=7"
        );
    }

    /// An unreadable `+0x48` pointer is refused by its own name, however good
    /// the other field looks.
    ///
    /// This case used to be the interesting one for a different reason: with
    /// two candidates the question was which failure to *attribute* the refusal
    /// to, and the answer was "the candidate actually walked". With one
    /// candidate there is nothing to outrank, and the case becomes the alarm
    /// instead. A well-formed table sits behind `+0x50` here and this device
    /// deliberately does not go and get it, so if a driven arm64 boot ever
    /// shows this slug the deletion of that chase is what to reconsider.
    #[test]
    fn an_unreadable_chased_pointer_is_refused_by_its_own_name() {
        let internal = ARM_KERNEL_VA_BASE + 0x10_000;
        let field_48 = ARM_KERNEL_VA_BASE + 0x20_000;
        let field_50 = ARM_KERNEL_VA_BASE + 0x30_000;
        let table = ARM_KERNEL_VA_BASE + 0x40_000;
        let mut mem = MapMem::new();

        mem.put_u64(field_50 + MAPPING_PAGE_TABLE_FROM_F50, table);
        mem.put_u32(table, 0);
        let fields = MapperInternalFields {
            internal_kva: internal,
            mapping_id: 3,
            internal_size: MAPPING_INTERNAL_EXPECTED_SIZE,
            page_field_48: field_48,
            page_field_50: field_50,
            raw_page_count: 1,
            ..MapperInternalFields::default()
        };

        let error =
            build_table_plan(&mem, 3, &fields, PAGE_SIZE_ARM64E, PAGE_SHIFT_ARM64E).unwrap_err();
        assert_eq!(
            refusal(&error),
            Some("iosurface_page_table_pointer_48_read")
        );
    }

    /// The page table comes through `+0x48` or it does not come at all.
    ///
    /// The `+0x50` chase that used to stand behind it was retired on 223
    /// successful resolves across two driven arm64 workloads, on which both
    /// fields were always populated and `+0x48` always won. The two cases that
    /// used to be rescued by it are now refusals **by name**, which is the
    /// whole trade: a rail nothing has confirmed no longer answers silently,
    /// and if either refusal ever appears in a driven boot's log it says the
    /// deletion was wrong and names which reading was right.
    #[test]
    fn the_page_table_comes_through_field_48_or_is_refused_by_name() {
        let internal = ARM_KERNEL_VA_BASE + 0x10_000;
        let field_48 = ARM_KERNEL_VA_BASE + 0x20_000;
        let field_50 = ARM_KERNEL_VA_BASE + 0x30_000;
        let table_a = ARM_KERNEL_VA_BASE + 0x40_000;
        let table_b = ARM_KERNEL_VA_BASE + 0x50_000;
        let good_entry = 1u32; // frame 1, which `entry_gpa_shift` accepts
        let base = |page_field_48, page_field_50| MapperInternalFields {
            internal_kva: internal,
            mapping_id: 3,
            internal_size: MAPPING_INTERNAL_EXPECTED_SIZE,
            page_field_48,
            page_field_50,
            raw_page_count: 1,
            ..MapperInternalFields::default()
        };

        // Only `+0x48` populated: a plan, and the census says the other field
        // was empty. On the two measured workloads this never happened.
        let mut mem = MapMem::new();
        mem.put_u64(field_48 + MAPPING_PAGE_TABLE_FROM_F48, table_a);
        mem.put_u32(table_a, good_entry);
        let plan = build_table_plan(
            &mem,
            3,
            &base(field_48, 0),
            PAGE_SIZE_ARM64E,
            PAGE_SHIFT_ARM64E,
        )
        .expect("the chased field alone is a plan");
        assert_eq!(plan.page_table_kva, table_a);
        assert!(!plan.candidates.other_field_populated);

        // Both populated and both parseable: the plan comes through `+0x48`,
        // and `table_b` is never read. This is the shape all 223 measured
        // resolves had.
        let mut mem = MapMem::new();
        mem.put_u64(field_48 + MAPPING_PAGE_TABLE_FROM_F48, table_a);
        mem.put_u64(field_50 + MAPPING_PAGE_TABLE_FROM_F50, table_b);
        mem.put_u32(table_a, good_entry);
        mem.put_u32(table_b, good_entry);
        let plan = build_table_plan(
            &mem,
            3,
            &base(field_48, field_50),
            PAGE_SIZE_ARM64E,
            PAGE_SHIFT_ARM64E,
        )
        .expect("both good is a plan");
        assert_eq!(plan.page_table_kva, table_a, "the chase is `+0x48`");
        assert!(plan.candidates.other_field_populated);

        // Only `+0x50` populated. The field test still sees it — this is not
        // `iosurface_page_table_fields_invalid` — but the chase that used to
        // answer it is gone, so it is refused under a name that says exactly
        // which reading of the two fields it would take to make that wrong.
        let mut mem = MapMem::new();
        mem.put_u64(field_50 + MAPPING_PAGE_TABLE_FROM_F50, table_b);
        mem.put_u32(table_b, good_entry);
        let error = build_table_plan(
            &mem,
            3,
            &base(0, field_50),
            PAGE_SIZE_ARM64E,
            PAGE_SHIFT_ARM64E,
        )
        .expect_err("the deleted chase does not answer this");
        assert_eq!(refusal(&error), Some("iosurface_page_table_only_field_50"));

        // Both populated, `+0x48`'s table unparseable. This is the one shape
        // the fallback was ever load-bearing for, and `earlier_failed` read
        // zero over both driven boots — so it is now a refusal carrying the
        // reason the table failed, rather than a silent rescue.
        let mut mem = MapMem::new();
        mem.put_u64(field_48 + MAPPING_PAGE_TABLE_FROM_F48, table_a);
        mem.put_u64(field_50 + MAPPING_PAGE_TABLE_FROM_F50, table_b);
        mem.put_u32(table_a, 0); // a zero entry is refused
        mem.put_u32(table_b, good_entry);
        let error = build_table_plan(
            &mem,
            3,
            &base(field_48, field_50),
            PAGE_SIZE_ARM64E,
            PAGE_SHIFT_ARM64E,
        )
        .expect_err("no second candidate rescues this any more");
        assert_eq!(
            refusal(&error),
            Some("iosurface_page_table_entry_invalid"),
            "the refusal names why the chased table failed, not that a \
             fallback was missing"
        );
    }

    #[test]
    fn packed_span_estimate_rounds_the_row_up() {
        // 200 BGRA = 800 tight, rounded to 896; the estimate is that row times
        // the height, and it is a byte count with no offset or pitch to bind.
        assert_eq!(
            packed_span_estimate(MTL_FORMAT_BGRA8_UNORM, 200, 100),
            Some(896 * 100)
        );
    }

    #[test]
    fn entry_gpa_and_span() {
        assert!(entry_gpa_shift(0, PAGE_SHIFT_ARM64E).is_none());
        let e = (5 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
        assert_eq!(
            entry_gpa_shift(e, PAGE_SHIFT_ARM64E).unwrap(),
            (5u64) << PAGE_SHIFT_ARM64E
        );
        assert_eq!(span_page_count_shift(0, PAGE_SHIFT_ARM64E), 1);
        assert_eq!(span_page_count_shift(1, PAGE_SHIFT_ARM64E), 1);
        assert_eq!(
            span_page_count_shift(PAGE_SIZE_ARM64E + 1, PAGE_SHIFT_ARM64E),
            2
        );
    }

    #[test]
    fn kernel_va_and_identity() {
        assert!(arm_kernel_va(ARM_KERNEL_VA_BASE + 0x1000));
        assert!(!arm_kernel_va(0x1000));
        assert!(x86_kernel_va(X86_KERNEL_VA_MIN + 0x1000));
        assert!(!x86_kernel_va(0x1000));
        assert!(guest_kernel_va(ARM_KERNEL_VA_BASE + 1));
        assert!(guest_kernel_va(X86_KERNEL_VA_MIN + 1));
        let mut m = MapMem::new();
        let kva = ARM_KERNEL_VA_BASE + 0x10000;
        m.put_u64(kva + MAPPING_INTERNAL_BACKPTR, kva);
        m.put_u32(kva + MAPPING_INTERNAL_ID, 1);
        m.put_u32(kva + MAPPING_INTERNAL_SIZE, MAPPING_INTERNAL_EXPECTED_SIZE);
        let f = read_mapper_identity(&m, kva, false, 0).unwrap();
        assert_eq!(f.mapping_id, 1);
        assert_eq!(validate_mapper_internal(&m, 1, &f), Status::Ok);
    }

    #[test]
    fn property_fuzz_packed_span_estimate() {
        for w in [1u32, 2, 64, 200, 1920] {
            for h in [1u32, 2, 100] {
                if let Some(end) = packed_span_estimate(MTL_FORMAT_BGRA8_UNORM, w, h) {
                    // Errs long: never below the tight extent it stands in for.
                    assert!(end >= w as u64 * 4 * h as u64);
                    assert_eq!(end % 128, 0);
                }
            }
        }
    }

    /// A texture bind takes its window from the guest's descriptor or not at
    /// all. With no descriptor there is nothing to derive a base offset or a
    /// pitch from, and supplying one would be a wrong bind that reads as success
    /// at every layer above — so this declines, and only the page-sizing
    /// estimate answers.
    #[test]
    fn a_bind_without_a_descriptor_declines_where_page_sizing_still_answers() {
        assert!(
            sample_window_from_device_desc(None, None, MTL_FORMAT_BGRA8_UNORM, 200, 100).is_none()
        );
        assert!(
            sample_window_from_device_desc(Some(&[0u8; 4]), None, MTL_FORMAT_BGRA8_UNORM, 200, 100)
                .is_none(),
            "a descriptor shorter than a full record is no descriptor"
        );
        assert_eq!(
            mapping_span_bound(None, MTL_FORMAT_BGRA8_UNORM, 200, 100),
            Some(896 * 100)
        );
    }

    /// The inverse of [`dims_extent`], for building a record to decode.
    ///
    /// Test-only inverse of the protocol decoder. The literal layout here is
    /// deliberately independent, so the round trip fails if the decoder moves.
    fn pack_plane_dims(width: u32, height: u32) -> u64 {
        ((width as u64 & 0x00ff_ffff) << 8) | ((height as u64 & 0x00ff_ffff) << 40)
    }

    /// The `dims` layout, against a word written out by hand.
    ///
    /// A round trip through [`pack_plane_dims`] cannot see a shift that moved,
    /// because it moves with it; this literal cannot. 1280 is `0x500` at byte 1
    /// and 720 is `0x2d0` at byte 5, which is the whole claim — and the guard
    /// bytes are the other half of it, since the element sizes share the word
    /// and a mask one bit too wide would read one of them into an extent.
    #[test]
    fn the_dims_word_puts_width_at_bit_eight_and_height_at_bit_forty() {
        assert_eq!(dims_extent(0x0002_d000_0005_0000), (1280, 720));
        assert_eq!(pack_plane_dims(1280, 720), 0x0002_d000_0005_0000);
        // Element width at byte 0 and element height at byte 4, both set to
        // every bit they have; neither may reach an extent.
        assert_eq!(dims_extent(0x0002_d0ff_0005_00ff), (1280, 720));
        // An extent that fills its 24 bits does not spill into the byte above.
        assert_eq!(dims_extent(0xffff_ff00_ffff_ff00), (0xff_ffff, 0xff_ffff));
    }

    /// The page-sizing estimate must not reach past the wire `alloc_size`
    /// (surface backing `length`). RE: allocateBackingHandle writes length@0
    /// independently of plane dims.
    #[test]
    fn mapping_span_bound_rejects_a_span_past_alloc_size() {
        use reims_vgpu_core::endian::{st32, st64};
        use reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM;

        // Device desc: 1024×1024 dims, alloc only 384*4096 = 0x180000.
        let mut desc = vec![0u8; DEVICE_DESC_LEN];
        st32(&mut desc[DEVICE_DESC_ALLOC_SIZE..], 0x18_0000);
        st32(
            &mut desc[DEVICE_DESC_PIXEL_FORMAT..],
            MTL_FORMAT_BGRA8_UNORM as u32,
        );
        let dims = pack_plane_dims(1024, 1024);
        st64(&mut desc[DEVICE_DESC_DIMS..], dims);
        // bpr too small for 1024 BGRA → the device-surface path rejects, so the
        // estimate is what answers and the allocation is what bounds it.
        st32(&mut desc[DEVICE_DESC_BPR..], 64);
        desc[DEVICE_DESC_PLANE_COUNT] = 0;

        // 1024*4096 > alloc → None (fail closed, no height lie).
        assert!(mapping_span_bound(Some(&desc), MTL_FORMAT_BGRA8_UNORM, 1024, 1024).is_none());
        // Within alloc: the estimate stands.
        assert_eq!(
            mapping_span_bound(Some(&desc), MTL_FORMAT_BGRA8_UNORM, 1024, 384),
            Some(384 * 4096)
        );
        // A descriptor this texture cannot be placed against sizes pages but
        // never sources a bind.
        assert!(sample_window_from_device_desc(
            Some(&desc),
            None,
            MTL_FORMAT_BGRA8_UNORM,
            1024,
            384
        )
        .is_none());
    }

    #[test]
    fn the_geometry_scan_picks_a_plane_only_when_exactly_one_matches() {
        use reims_vgpu_core::endian::{st16, st32, st64};
        use reims_vgpu_core::pixel_format::{MTL_FORMAT_R8_UNORM, MTL_FORMAT_RG8_UNORM};

        let mut desc = vec![0u8; DEVICE_DESC_LEN];
        st32(&mut desc[DEVICE_DESC_ALLOC_SIZE..], 0x20000);
        desc[DEVICE_DESC_PLANE_COUNT] = 2;
        // Plane 0: Y 16×8 R8 bpr=64 offset=512 size=512
        let p0 = DEVICE_DESC_PLANES;
        st32(&mut desc[p0 + DEVICE_PLANE_OFFSET..], 512);
        st32(&mut desc[p0 + DEVICE_PLANE_SIZE..], 512);
        st64(&mut desc[p0 + DEVICE_PLANE_DIMS..], pack_plane_dims(16, 8));
        st32(&mut desc[p0 + DEVICE_PLANE_BPR..], 64);
        st16(&mut desc[p0 + DEVICE_PLANE_BPE..], 1);
        // Plane 1: UV 8×4 RG8 bpr=64 offset=1024 size=256
        let p1 = DEVICE_DESC_PLANES + DEVICE_PLANE_DESC_LEN;
        st32(&mut desc[p1 + DEVICE_PLANE_OFFSET..], 1024);
        st32(&mut desc[p1 + DEVICE_PLANE_SIZE..], 256);
        st64(&mut desc[p1 + DEVICE_PLANE_DIMS..], pack_plane_dims(8, 4));
        st32(&mut desc[p1 + DEVICE_PLANE_BPR..], 64);
        st16(&mut desc[p1 + DEVICE_PLANE_BPE..], 2);

        let (off_y, bpr_y, end_y) =
            sample_window_from_device_desc(Some(&desc), None, MTL_FORMAT_R8_UNORM, 16, 8).unwrap();
        assert_eq!(off_y, 512);
        assert_eq!(bpr_y, 64);
        // exclusive last-row end: 512 + 7*64 + 16
        assert_eq!(end_y, 512 + 7 * 64 + 16);

        let (off_uv, bpr_uv, end_uv) =
            sample_window_from_device_desc(Some(&desc), None, MTL_FORMAT_RG8_UNORM, 8, 4).unwrap();
        assert_eq!(off_uv, 1024);
        assert_eq!(bpr_uv, 64);
        assert_eq!(end_uv, 1024 + 3 * 64 + 16);

        // Dims that hit no plane record: zero matches, so nothing is bound.
        assert!(
            sample_window_from_device_desc(Some(&desc), None, MTL_FORMAT_R8_UNORM, 4, 4).is_none()
        );
    }

    /// v0a8 (biplanar video + alpha) shape from the live apple.com hero: the
    /// Y and alpha planes share geometry and bpe, so the geometry scan is
    /// ambiguous by construction — only an explicit wire plane index (IOSurface plane view
    /// record `+0x20`) separates them.
    #[test]
    fn sample_window_plane_index_selects_among_same_geometry_planes() {
        use reims_vgpu_core::endian::{st16, st32, st64};
        use reims_vgpu_core::pixel_format::{MTL_FORMAT_R8_UNORM, MTL_FORMAT_RG8_UNORM};

        // Live shape (scaled): Y 946×350 @32 bpr 960; UV 473×175 @336032
        // bpr 960 bpe 2; alpha 946×350 @504992 bpr 960 bpe 1.
        let mut desc = vec![0u8; DEVICE_DESC_LEN];
        st32(&mut desc[DEVICE_DESC_ALLOC_SIZE..], 843_776);
        desc[DEVICE_DESC_PLANE_COUNT] = 3;
        let planes = [
            (32u32, 336_000u32, 946u32, 350u32, 960u32, 1u16),
            (336_032, 168_000, 473, 175, 960, 2),
            (504_992, 336_000, 946, 350, 960, 1),
        ];
        for (i, (off, size, w, h, bpr, bpe)) in planes.iter().enumerate() {
            let base = DEVICE_DESC_PLANES + i * DEVICE_PLANE_DESC_LEN;
            st32(&mut desc[base + DEVICE_PLANE_OFFSET..], *off);
            st32(&mut desc[base + DEVICE_PLANE_SIZE..], *size);
            st64(
                &mut desc[base + DEVICE_PLANE_DIMS..],
                pack_plane_dims(*w, *h),
            );
            st32(&mut desc[base + DEVICE_PLANE_BPR..], *bpr);
            st16(&mut desc[base + DEVICE_PLANE_BPE..], *bpe);
        }

        // Indexed selection: each plane record by its wire index.
        let y = sample_window_from_device_desc(Some(&desc), Some(0), MTL_FORMAT_R8_UNORM, 946, 350)
            .unwrap();
        assert_eq!((y.0, y.1), (32, 960));
        let uv =
            sample_window_from_device_desc(Some(&desc), Some(1), MTL_FORMAT_RG8_UNORM, 473, 175)
                .unwrap();
        assert_eq!((uv.0, uv.1), (336_032, 960));
        let a = sample_window_from_device_desc(Some(&desc), Some(2), MTL_FORMAT_R8_UNORM, 946, 350)
            .unwrap();
        assert_eq!((a.0, a.1), (504_992, 960));

        // No index: Y geometry matches plane 0 AND plane 2. Two matches is not
        // "pick the first" — the scan cannot tell them apart, so it declines and
        // the caller reports a lost bind rather than sampling luma for alpha.
        assert!(
            sample_window_from_device_desc(Some(&desc), None, MTL_FORMAT_R8_UNORM, 946, 350)
                .is_none()
        );

        // An index past the plane count names no record, so it resolves nothing
        // — not the geometry scan's answer, and not plane 0's bytes.
        assert!(sample_window_from_device_desc(
            Some(&desc),
            Some(7),
            MTL_FORMAT_R8_UNORM,
            946,
            350
        )
        .is_none());
    }
}
