//! Live guest-visible state remembered by the host.
//!
//! Registers, rings, tasks/objects, mapper, present/cursor, stamps — the
//! ApplePV-shaped model. No parsing of wire bytes here; no backend execution.

mod lru_memo;
mod regs;
mod state;

pub(crate) use lru_memo::LruBytesMemo;
pub(crate) use regs::*;
pub use reims_vgpu_core::{
    ChannelRing, CursorState, DisplayHandshake, DisplayOnlinePoll, DisplaySharedPage,
    GfxRegisters as GfxRegs, IosfcRegisters as IosfcRegs, MapperCapture, PendingWork,
    TargetIdentity, TargetKeyDivergence, TaskEntry, TaskTable, GFX_MMIO_SIZE,
};
// `GfxRegs` has no in-crate importer and is here for the five doc comments
// that link `model::GfxRegs::child_doorbell_rung`. `state` is a private
// `mod`, so this is the only path those links can name — and rustc's
// unused-import lint cannot see a doc link, so it will call this dead.
#[cfg(test)]
pub(crate) use state::SurfaceMappingLifecycle;
pub use state::{
    ComputeStorageOrigin, ComputeStorageResidencyKey, DeviceId, DeviceResetEffect, DeviceState,
    ExecFault, FailEvent, GuestLinearMemo, GvaBacking, GvaEvictionWitness, GvaHostView,
    HostLinearTexture, HostReleaseEffect, HostSurface, LinearMaterializeDecline,
    LinearReplicaWindow, MappingInvalidationEffect, PacketFault, PresentBacking, PresentState,
    ResourceValidity, SurfaceBackingWalk, SurfaceMappingEntry, SurfaceWriteKind,
    TaskDefinitionEffect, TaskDefinitionKind, TaskNamespaceRetirement, TaskResource,
    TaskResourceLifetimeRef, TranslationHoldAtReset, UnimplementedCommand,
    GVA_ENCODE_CACHE_BYTE_CAP, GVA_EVICTION_WITNESS_KEYS,
};
pub(crate) use state::{
    HostPageView, LoadedComputePipeline, LoadedFunction, StateMutationDecline, SurfaceHostView,
};

#[cfg(test)]
mod tests {
    use crate::model::{PAGE_SHIFT_ARM64E, PAGE_SIZE_ARM64E, PAGE_SIZE_X86};

    use super::*;
    use crate::runtime::host::{HostActionKind, HostMemory};
    use crate::runtime::Device;
    use crate::runtime::FakeHost;
    use reims_vgpu_core::endian::st32;

    #[test]
    fn stamp_slot_offset_respects_guest_page_size() {
        assert_eq!(stamp_slot_offset(0, PAGE_SIZE_X86), Some(0));
        assert_eq!(stamp_slot_offset(1023, PAGE_SIZE_X86), Some(1023 * 4));
        assert_eq!(stamp_slot_offset(1024, PAGE_SIZE_X86), None);
        assert_eq!(stamp_slot_offset(1024, PAGE_SIZE_ARM64E), Some(1024 * 4));
        assert_eq!(stamp_slot_offset(4095, PAGE_SIZE_ARM64E), Some(4095 * 4));
        assert_eq!(stamp_slot_offset(4096, PAGE_SIZE_ARM64E), None);
    }

    fn dev() -> Device {
        Device::new(DeviceId(1), PAGE_SHIFT_ARM64E)
    }

    #[test]
    fn reset_view_collection_detaches_every_guest_alias() {
        let mut d = dev();
        let import = std::sync::Arc::new(
            crate::runtime::guest_ram::GuestRamImport::new_host_allocation(0x3000, 0x4000, 0x1000)
                .expect("aligned test import"),
        );
        let import_id = import.id();
        d.state.host_materializations.retire_view((0x1000, 0x2000));
        let mut view = SurfaceHostView::new(
            0x3000,
            0x4000,
            reims_vgpu_memory::GuestPageFootprint::new(
                [0x3000, 0x4000, 0x5000, 0x6000].into(),
                0x1000,
            )
            .expect("test footprint"),
        )
        .expect("valid host view");
        assert!(view.replace_import(import).is_none());
        let mut materialization = super::state::SurfaceMaterialization::default();
        materialization.install(view);
        d.state.surfaces.mappings.insert(
            7,
            SurfaceMappingEntry {
                materialization,
                ..Default::default()
            },
        );
        d.state.host_materializations.publish_gva_view(GvaHostView {
            task_id: 1,
            gva: 0x8000,
            length: 0x1000,
            host_view: HostPageView::new(0x5000, 0x6000),
            ..Default::default()
        });

        let effects = d.state.take_all_host_release_effects();
        assert!(matches!(
            effects.first(),
            Some(HostReleaseEffect::RetireImportedView { import, .. }) if *import == import_id
        ));
        let mut views: Vec<_> = effects
            .iter()
            .filter_map(|effect| match effect {
                HostReleaseEffect::ReleaseView { ptr, len }
                | HostReleaseEffect::RetireImportedView { ptr, len, .. } => Some((*ptr, *len)),
                HostReleaseEffect::RetireGuestImport(_)
                | HostReleaseEffect::RetireLinearResident(_) => None,
            })
            .collect();
        views.sort_unstable();
        assert_eq!(
            views,
            vec![(0x1000, 0x2000), (0x3000, 0x4000), (0x5000, 0x6000)]
        );
        assert!(d.state.host_materializations.queued_views().is_empty());
        assert!(d.state.host_materializations.views().is_empty());
        assert!(!d.state.surfaces.mappings[&7].materialization.has_view());
    }

    #[test]
    fn a_mapping_host_view_cannot_split_its_pointer_length_and_footprint() {
        let footprint = reims_vgpu_memory::GuestPageFootprint::new([0x1000, 0x2000].into(), 0x1000)
            .expect("two-page footprint");
        assert!(SurfaceHostView::new(0x3000, 0x1000, footprint.clone()).is_none());
        assert!(SurfaceHostView::new(0, 0x2000, footprint.clone()).is_none());
        assert!(SurfaceHostView::new(0x3000, 0, footprint).is_none());
    }

    #[test]
    fn replacing_a_mapping_import_retires_the_previous_identity() {
        let footprint = reims_vgpu_memory::GuestPageFootprint::new([0x1000].into(), 0x1000)
            .expect("one-page footprint");
        let mut view = SurfaceHostView::new(0x3000, 0x1000, footprint).expect("valid view");
        let first = std::sync::Arc::new(
            reims_vgpu_memory::GuestRamImport::new_host_allocation(0x3000, 0x1000, 0x1000)
                .expect("first import"),
        );
        let first_id = first.id();
        assert!(view.replace_import(std::sync::Arc::clone(&first)).is_none());

        let replacement = std::sync::Arc::new(
            reims_vgpu_memory::GuestRamImport::new_host_allocation(0x3000, 0x1000, 0x1000)
                .expect("replacement import"),
        );
        assert_eq!(view.replace_import(replacement), Some(first_id));
        assert!(first.is_retired());
    }

    fn setup_boot_regs(d: &mut Device, h: &mut FakeHost) {
        d.gfx_write(h, GFX_REG_VERSION, 0x3e, MMIO_U32);
        assert_eq!(d.gfx_read(GFX_REG_VERSION, MMIO_U32), 0x3e);
        d.gfx_write(h, GFX_REG_FIFO_BASE_PAGE, 0x10, MMIO_U32);
        d.gfx_write(h, GFX_REG_FIFO_START, 0x4000, MMIO_U32);
        d.gfx_write(h, GFX_REG_FIFO_LENGTH, 0x10000, MMIO_U32);
        d.gfx_write(h, GFX_REG_CONTROL_FIFO, 1, MMIO_U32);
        let stamp = pfn_to_gpa(0x10, PAGE_SHIFT_ARM64E);
        h.map_range(stamp, PAGE_SIZE_ARM64E as usize, 0);
        let ring = stamp + 0x4000;
        h.map_range(ring, 0xc000, 0);
    }

    fn write_main_packet(h: &mut FakeHost, abs_off: u32, opcode: u16, stamp: u32, payload: &[u8]) {
        let ring_base = pfn_to_gpa(0x10, PAGE_SHIFT_ARM64E) + 0x4000;
        let ring_size = 0xc000u32;
        let total = PACKET_HEADER_LEN + payload.len() as u32;
        let mut raw = vec![0u8; total as usize];
        raw[0..2].copy_from_slice(&opcode.to_le_bytes());
        raw[2..4].copy_from_slice(&0u16.to_le_bytes());
        raw[4..8].copy_from_slice(&total.to_le_bytes());
        raw[8..12].copy_from_slice(&stamp.to_le_bytes());
        raw[12..].copy_from_slice(payload);
        for (i, b) in raw.iter().enumerate() {
            let off = abs_off.wrapping_add(i as u32) % ring_size;
            let _ = h.write_gpa(ring_base + off as u64, &[*b]);
        }
    }

    #[test]
    fn version_handshake_echo() {
        let mut d = dev();
        let mut h = FakeHost::new();
        d.gfx_write(&mut h, GFX_REG_VERSION, 0x3e, MMIO_U32);
        assert_eq!(d.gfx_read(GFX_REG_VERSION, MMIO_U32), 0x3e);
    }

    /// A version this host does not implement must come back as the newest one
    /// it does, not as itself.
    ///
    /// The guest switches on what it reads back, and its switch has no arm
    /// above `PROTOCOL_VERSION_MAX` — everything past the top rung lands in a
    /// default that turns object tables, the child doorbell, heaps and
    /// buffer-from-IOSurface all off at once. Echoing the request is therefore
    /// not a harmless pass-through: it is how a guest newer than this host gets
    /// silently degraded to a near-empty device.
    #[test]
    fn version_handshake_clamps_above_what_this_host_implements() {
        let mut d = dev();
        let mut h = FakeHost::new();
        d.gfx_write(
            &mut h,
            GFX_REG_VERSION,
            PROTOCOL_VERSION_MAX as u64 + 1,
            MMIO_U32,
        );
        assert_eq!(
            d.gfx_read(GFX_REG_VERSION, MMIO_U32),
            PROTOCOL_VERSION_MAX as u64
        );
        // A guest older than this host keeps its own version: the host must not
        // answer with features the guest never asked to speak.
        d.reset();
        d.gfx_write(&mut h, GFX_REG_VERSION, 4, MMIO_U32);
        assert_eq!(d.gfx_read(GFX_REG_VERSION, MMIO_U32), 4);
    }

    #[test]
    fn both_register_windows() {
        let mut d = dev();
        let mut h = FakeHost::new();
        d.gfx_write(&mut h, GFX_REG_ROOT_PAGE, 0x42, MMIO_U32);
        assert_eq!(d.gfx_read(GFX_REG_ROOT_PAGE, MMIO_U32), 0x42);
        d.iosfc_write(&mut h, IOSFC_REG_RING_BASE, 0x7000_0000, MMIO_U64);
        d.iosfc_write(&mut h, IOSFC_REG_CAPACITY, 0x400, MMIO_U32);
        assert_eq!(d.iosfc_read(IOSFC_REG_RING_BASE, MMIO_U64), 0x7000_0000);
        assert_eq!(d.iosfc_read(IOSFC_REG_CAPACITY, MMIO_U32), 0x400);
        d.state
            .registers
            .gfx
            .interrupt_status_gpu
            .store(0x5, std::sync::atomic::Ordering::Release);
        assert_eq!(d.gfx_read(GFX_REG_INTR_STATUS_GPU, MMIO_U32), 0x5);
        assert_eq!(d.gfx_read(GFX_REG_INTR_STATUS_GPU, MMIO_U32), 0);
    }

    #[test]
    fn efi_display_constants() {
        let mut d = dev();
        assert_eq!(
            d.gfx_read(GFX_REG_EFI_MODE_COUNT, MMIO_U32),
            EFI_MODE_COUNT as u64
        );
        assert_eq!(
            d.gfx_read(GFX_REG_EFI_DISPLAY_PORTS, MMIO_U32),
            EFI_DISPLAY_PORT_COUNT as u64
        );
        assert_eq!(
            d.gfx_read(GFX_REG_EFI_BUILTIN_CONNECTED, MMIO_U32),
            EFI_BUILTIN_CONNECTED as u64
        );
        let size = d.gfx_read(GFX_REG_EFI_MODE_SIZE, MMIO_U32);
        assert_eq!(size >> EFI_MODE_WIDTH_SHIFT, EFI_BOOT_WIDTH as u64);
        assert_eq!(size & 0xffff, EFI_BOOT_HEIGHT as u64);
    }

    #[test]
    fn main_fifo_wraparound() {
        let mut d = dev();
        let mut h = FakeHost::new();
        setup_boot_regs(&mut d, &mut h);
        let ring_size = 0xc000u32;
        let payload = 1u32.to_le_bytes();
        let total = PACKET_HEADER_LEN + 4;
        let start = ring_size - 8;
        write_main_packet(&mut h, start, ROOT_OP_DEFINE_FIFO, 0x11, &payload);
        d.state
            .registers
            .gfx
            .fifo_read
            .store(start, std::sync::atomic::Ordering::Release);
        d.state.registers.gfx.fifo_written = start.wrapping_add(total);
        d.state.scheduling.pending.request_main();
        d.drain(&mut h);
        assert_eq!(
            d.state
                .registers
                .gfx
                .fifo_read
                .load(std::sync::atomic::Ordering::Acquire),
            start.wrapping_add(total)
        );
        assert!(d.state.scheduling.pending.active_child_mask() & (1 << 1) != 0);
        assert_eq!(h.get_u32(pfn_to_gpa(0x10, PAGE_SHIFT_ARM64E)), 0x11);
        assert!(h.action_count(HostActionKind::IrqGfxPulse) >= 1);
    }

    #[test]
    fn device_info_reply() {
        let mut d = dev();
        let mut h = FakeHost::new();
        setup_boot_regs(&mut d, &mut h);
        let reply_pfn = 0x20u32;
        h.map_range(
            pfn_to_gpa(reply_pfn, PAGE_SHIFT_ARM64E),
            PAGE_SIZE_ARM64E as usize,
            0xee,
        );
        let mut payload = vec![0u8; 12];
        // The word at 0 is the guest's parse ceiling, exclusive; this is what a
        // 13.7.8 guest writes. It read 0 here for as long as the offset was
        // thought to be unused, and a ceiling of 0 admits no key at all.
        st32(
            &mut payload[DEVICE_INFO_TAHOE_KEY_TABLE_LEN..],
            DEVICE_INFO_KEY_BUFFER_WITH_IOSURFACE + 1,
        );
        // A whole page of pair slots, as the guest sends. A short count here
        // would make this test emit the reply-truncated alarm every run, and a
        // standing false alarm in the suite's own log is one nobody reads.
        st32(
            &mut payload[DEVICE_INFO_TAHOE_COUNT..],
            (PAGE_SIZE_ARM64E as usize / DEVICE_INFO_REPLY_PAIR_LEN) as u32,
        );
        st32(&mut payload[DEVICE_INFO_TAHOE_REPLY_PFN..], reply_pfn);
        write_main_packet(&mut h, 0, ROOT_OP_DEVICE_INFO_TAHOE, 3, &payload);
        d.state
            .registers
            .gfx
            .fifo_read
            .store(0, std::sync::atomic::Ordering::Release);
        d.state.registers.gfx.fifo_written = PACKET_HEADER_LEN + 12;
        d.state.scheduling.pending.request_main();
        d.drain(&mut h);
        assert_eq!(
            h.get_u32(pfn_to_gpa(reply_pfn, PAGE_SHIFT_ARM64E)),
            DEVICE_INFO_CAPS[0].0
        );
        // The value is asserted as a bound, not as the table entry. The
        // GPU-dependent keys are reduced to what the host can execute, and in a
        // test there is no resolved device, so the answer is the backend's
        // floor. Pinning the table entry here would pin the over-promise the
        // reduction exists to prevent.
        let served = h.get_u32(pfn_to_gpa(reply_pfn, PAGE_SHIFT_ARM64E) + 4);
        assert!(
            served >= 1 && served <= DEVICE_INFO_CAPS[0].1,
            "served {served} must be a reduction of {}",
            DEVICE_INFO_CAPS[0].1
        );
    }

    /// The AIR version the guest actually receives sits inside the window the
    /// guest honours, read out of the reply page rather than out of the table.
    ///
    /// The `const` assertion beside [`DEVICE_INFO_AIR_VERSION`] already binds
    /// the table entry. This binds the other end: key 18 is served through the
    /// same reduction every other key goes through, so a future change that
    /// routes it through a host-dependent floor — as the GPU-dependent keys are
    /// — could deliver a value the constant never sees. The guest's driver
    /// rewrites an out-of-window value silently in both directions (undefined
    /// becomes 2.2, at-or-above 2.8 clamps to 2.7), so a wrong value here does
    /// not fail, it just stops being what this table says it is.
    ///
    /// Driven at macOS 26's declared ceiling of 45, which is the rail that has a
    /// consumer for this key.
    #[test]
    fn the_air_version_the_guest_receives_is_inside_the_window_the_guest_honours() {
        let mut d = dev();
        let mut h = FakeHost::new();
        setup_boot_regs(&mut d, &mut h);
        let reply_pfn = 0x20u32;
        h.map_range(
            pfn_to_gpa(reply_pfn, PAGE_SHIFT_ARM64E),
            PAGE_SIZE_ARM64E as usize,
            0xee,
        );
        let mut payload = vec![0u8; 12];
        st32(&mut payload[DEVICE_INFO_TAHOE_KEY_TABLE_LEN..], 45);
        st32(
            &mut payload[DEVICE_INFO_TAHOE_COUNT..],
            (PAGE_SIZE_ARM64E as usize / DEVICE_INFO_REPLY_PAIR_LEN) as u32,
        );
        st32(&mut payload[DEVICE_INFO_TAHOE_REPLY_PFN..], reply_pfn);
        write_main_packet(&mut h, 0, ROOT_OP_DEVICE_INFO_TAHOE, 3, &payload);
        d.state
            .registers
            .gfx
            .fifo_read
            .store(0, std::sync::atomic::Ordering::Release);
        d.state.registers.gfx.fifo_written = PACKET_HEADER_LEN + 12;
        d.state.scheduling.pending.request_main();
        d.drain(&mut h);

        // Walk the reply the way the guest does: pairs until the zero
        // terminator. Reading a fixed slot instead would pin the key's position
        // in the table, which is not what is under test here.
        let base = pfn_to_gpa(reply_pfn, PAGE_SHIFT_ARM64E);
        let mut served = None;
        for slot in 0..(PAGE_SIZE_ARM64E as usize / DEVICE_INFO_REPLY_PAIR_LEN) {
            let at = base + (slot * DEVICE_INFO_REPLY_PAIR_LEN) as u64;
            let key = h.get_u32(at);
            if key == 0 {
                break;
            }
            if key == DEVICE_INFO_KEY_MAX_MSL_VERSION {
                served = Some(h.get_u32(at + 4));
                break;
            }
        }

        let served = served.expect("key 18 is inside macOS 26's ceiling, so the reply carries it");
        assert!(
            (DEVICE_INFO_AIR_VERSION_MIN..=DEVICE_INFO_AIR_VERSION_MAX).contains(&served),
            "the guest was told AIR {served:#x}, outside the window \
             {DEVICE_INFO_AIR_VERSION_MIN:#x}..={DEVICE_INFO_AIR_VERSION_MAX:#x} it honours — \
             it will rewrite the value and hold something this table does not describe"
        );
    }

    /// A key the guest parses and this device never answers is counted, and the
    /// two ways that happens are counted apart.
    ///
    /// The reply already reported `above_ceiling` — keys this device sends that
    /// the guest discards, which costs nothing. The opposite direction was
    /// silent, and it is the one that can cost guest work: the guest's walker
    /// has an arm per key, so a key that never arrives leaves that field at
    /// whatever the capability struct was initialised to, and
    /// [`DEVICE_INFO_CAPS`]'s doc is explicit that a value here is an
    /// instruction to the guest about what it may build.
    ///
    /// Driven at the two ceilings real rails declare, measured on driven boots:
    /// macOS 26 sends 45 and macOS 15 sends 42. The rails disagreeing is the
    /// point — a guest that parses further must report more holes, which is
    /// what fails if the ceiling stops feeding the computation.
    #[test]
    fn the_keys_a_guest_parses_and_this_device_never_answers_are_counted() {
        use crate::runtime::drain::store_route_count;

        /// Drive one device-info request at `key_table_len` and return the
        /// `(holes, tail)` this reply contributed.
        fn ask(key_table_len: u32) -> (u64, u64) {
            let mut d = dev();
            let mut h = FakeHost::new();
            setup_boot_regs(&mut d, &mut h);
            let reply_pfn = 0x20u32;
            h.map_range(
                pfn_to_gpa(reply_pfn, PAGE_SHIFT_ARM64E),
                PAGE_SIZE_ARM64E as usize,
                0xee,
            );
            let mut payload = vec![0u8; 12];
            st32(
                &mut payload[DEVICE_INFO_TAHOE_KEY_TABLE_LEN..],
                key_table_len,
            );
            st32(
                &mut payload[DEVICE_INFO_TAHOE_COUNT..],
                (PAGE_SIZE_ARM64E as usize / DEVICE_INFO_REPLY_PAIR_LEN) as u32,
            );
            st32(&mut payload[DEVICE_INFO_TAHOE_REPLY_PFN..], reply_pfn);
            write_main_packet(&mut h, 0, ROOT_OP_DEVICE_INFO_TAHOE, 3, &payload);
            d.state
                .registers
                .gfx
                .fifo_read
                .store(0, std::sync::atomic::Ordering::Release);
            d.state.registers.gfx.fifo_written = PACKET_HEADER_LEN + 12;
            d.state.scheduling.pending.request_main();
            let before = (
                store_route_count("device_info_key_holes"),
                store_route_count("device_info_key_tail"),
            );
            d.drain(&mut h);
            (
                store_route_count("device_info_key_holes") - before.0,
                store_route_count("device_info_key_tail") - before.1,
            )
        }

        let table_top = DEVICE_INFO_CAPS
            .iter()
            .map(|&(key, _)| key)
            .max()
            .expect("the table is not empty");

        // The three sets partition the guest's parse range, and the partition is
        // derived here rather than written down twice. Adding a key to the table
        // without dropping it from the unanswered list — or the reverse — fails
        // this, which is the only thing stopping that list from decaying into a
        // comment that used to be true.
        let answered: std::collections::BTreeSet<u32> =
            DEVICE_INFO_CAPS.iter().map(|&(key, _)| key).collect();
        let derived_unanswered: Vec<u32> = (1..=table_top)
            .filter(|k| !answered.contains(k))
            .filter(|k| !DEVICE_INFO_DEAD_KEYS.contains(k))
            .collect();
        assert_eq!(
            derived_unanswered, DEVICE_INFO_UNANSWERED_KEYS,
            "DEVICE_INFO_UNANSWERED_KEYS must be exactly the keys below the \
             table top that this device neither answers nor knows to be dead"
        );
        for dead in DEVICE_INFO_DEAD_KEYS {
            assert!(
                !answered.contains(dead),
                "key {dead} is sent and also declared dead — the guest would \
                 discard it, so one of the two is wrong"
            );
        }

        // A guest that parses nothing has no unanswered key of either kind:
        // key 0 terminates the walk and is not a key, so a ceiling of 1 admits
        // none. This is what catches an off-by-one that counts key 0 as a hole
        // on every boot of every rail.
        assert_eq!(ask(1), (0, 0), "a ceiling of 1 admits no key at all");

        // Tail is pure arithmetic and independently checkable: keys beyond
        // anything this device has ever been asked for.
        assert_eq!(
            ask(table_top + 1).1,
            0,
            "a guest that stops at the table's top asks nothing new"
        );
        assert_eq!(
            ask(table_top + 9).1,
            8,
            "eight keys past the table's top are eight tail keys"
        );

        // The two real rails. The holes are the keys below each ceiling that
        // the table skips; deriving the expectation from `DEVICE_INFO_CAPS`
        // rather than writing the numbers keeps this from pinning today's gaps.
        let gaps_below = |ceiling: u32| -> u64 {
            (1..ceiling.min(table_top + 1))
                .filter(|key| !DEVICE_INFO_CAPS.iter().any(|&(k, _)| k == *key))
                .filter(|key| !DEVICE_INFO_DEAD_KEYS.contains(key))
                .count() as u64
        };
        let macos_15 = ask(42).0;
        let macos_26 = ask(45).0;
        assert_eq!(macos_15, gaps_below(42), "macOS 15 parses keys 1..=41");
        assert_eq!(macos_26, gaps_below(45), "macOS 26 parses keys 1..=44");

        // **The two rails have the same holes, and that is the finding.** The
        // only key macOS 26 parses that macOS 15 does not and this device does
        // not answer is 43, which the guest's own walker has no arm for. Once it
        // stops being counted, the hole sets are identical — so no hole can
        // explain a defect that appears on macOS 26 and not on macOS 15. This
        // assertion used to read `macos_26 > macos_15` and passed for exactly
        // the wrong reason: it was counting the dead key.
        assert_eq!(
            macos_26, macos_15,
            "the extra keys macOS 26 parses are answered (42, 44) or dead (43), \
             so parsing further must find no additional hole"
        );

        // Ceiling sensitivity still has to hold, or the report could be a
        // constant. Key 22 is a real hole, so a ceiling above it must find one
        // more than a ceiling at it.
        assert_eq!(
            ask(23).0,
            ask(22).0 + 1,
            "raising the ceiling past the hole at key 22 must report it"
        );
    }

    #[test]
    fn reset_clears_state() {
        let mut d = dev();
        let mut h = FakeHost::new();
        d.gfx_write(&mut h, GFX_REG_VERSION, 0x3e, MMIO_U32);
        d.state.define_task(1, 0x1000, 5);
        d.state.map_surface(7);
        d.record_fail(FailEvent::UnknownRootOpcode {
            opcode: 0xff,
            total_size: 12,
        });
        d.reset();
        assert_eq!(d.gfx_read(GFX_REG_VERSION, MMIO_U32), 0);
        // A reset removes the task entirely rather than clearing a slot in
        // place, so the question is liveness and not a flag on a resident entry.
        assert!(!d.state.tasks.is_active(1));
        assert!(d.state.surfaces.mappings.is_empty());
        assert!(d.fails().is_empty());
    }

    #[test]
    fn resource_lifecycle() {
        let mut d = dev();
        d.state.define_task(2, 0x2000, 9);
        assert!(d.state.set_object_list(2, 3, 64));
        assert!(d.state.insert_object(2, 10));
        assert!(d.state.fixtures.objects.contains(&(2, 10)));
        assert!(d.state.delete_object(2, 10));
        assert!(!d.state.fixtures.objects.contains(&(2, 10)));
        d.state.insert_object(2, 1);
        d.state.define_task(2, 0x2000, 9);
        assert!(d.state.fixtures.objects.is_empty());
    }

    #[test]
    fn mapper_handshake() {
        let mut d = dev();
        let mut h = FakeHost::new();
        let base = 0x8000_0000u64;
        h.map_range(base, 0x1000, 0);
        d.iosfc_write(&mut h, IOSFC_REG_RING_BASE, base, MMIO_U64);
        d.iosfc_write(&mut h, IOSFC_REG_CAPACITY, 0x40, MMIO_U32);
        let _ = h.write_gpa(base, &1u32.to_le_bytes());
        let _ = h.write_gpa(base + 4, &7u32.to_le_bytes());
        // PRODUCER write sync-drains the mapper ring on the publishing vCPU
        // (pending.iosfc is set then cleared inside iosfc_write).
        d.iosfc_write(&mut h, IOSFC_REG_PRODUCER, 0x10, MMIO_U32);
        assert!(
            !d.state.scheduling.pending.iosfc_requested(),
            "iosfc producer path drains before return"
        );
        assert!(
            d.state.surfaces.mappings.contains_key(&7)
                || !d.fails().is_empty()
                || d.state.registers.iosfc.consumer > 0
                || h.bh_scheduled
        );
    }

    #[test]
    fn present_and_cursor_model() {
        let mut d = dev();
        d.state
            .presentation
            .present
            .set_console_geometry_for_test(1440, 900);
        d.state.presentation.cursor.set_visible(true);
        d.state
            .presentation
            .cursor
            .publish_glyph(2, 2, 1, 1, vec![0xff00_0000; 4]);
        assert_eq!(d.state.presentation.present.console_width(), 1440);
        assert!(d.state.presentation.cursor.position().visible);
    }

    #[test]
    fn fail_visible_unknown_root() {
        let mut d = dev();
        let mut h = FakeHost::new();
        setup_boot_regs(&mut d, &mut h);
        write_main_packet(&mut h, 0, 0xeeee, 1, &[]);
        d.state
            .registers
            .gfx
            .fifo_read
            .store(0, std::sync::atomic::Ordering::Release);
        d.state.registers.gfx.fifo_written = PACKET_HEADER_LEN;
        d.state.scheduling.pending.request_main();
        d.drain(&mut h);
        assert!(!d.fails().is_empty());
    }

    #[test]
    fn fail_visible_malformed_packet() {
        let mut d = dev();
        let mut h = FakeHost::new();
        setup_boot_regs(&mut d, &mut h);
        // total_size smaller than header
        let ring_base = pfn_to_gpa(0x10, PAGE_SHIFT_ARM64E) + 0x4000;
        let _ = h.write_gpa(ring_base, &0u16.to_le_bytes());
        let _ = h.write_gpa(ring_base + 2, &0u16.to_le_bytes());
        let _ = h.write_gpa(ring_base + 4, &4u32.to_le_bytes());
        d.state
            .registers
            .gfx
            .fifo_read
            .store(0, std::sync::atomic::Ordering::Release);
        d.state.registers.gfx.fifo_written = 12;
        d.state.scheduling.pending.request_main();
        let before = d
            .state
            .registers
            .gfx
            .fifo_read
            .load(std::sync::atomic::Ordering::Acquire);
        d.drain(&mut h);
        // malformed: head must not advance (or fail recorded)
        assert!(
            d.state
                .registers
                .gfx
                .fifo_read
                .load(std::sync::atomic::Ordering::Acquire)
                == before
                || !d.fails().is_empty()
        );
    }

    #[test]
    fn child_fifo_drain_define_task() {
        let mut d = dev();
        let mut h = FakeHost::new();
        setup_boot_regs(&mut d, &mut h);
        let ch = 1u32;
        d.state.scheduling.pending.activate_children(1 << ch);
        // Minimal child ring setup is covered by main DEFINE_FIFO + drain path.
        let payload = ch.to_le_bytes();
        write_main_packet(&mut h, 0, ROOT_OP_DEFINE_FIFO, 2, &payload);
        d.state
            .registers
            .gfx
            .fifo_read
            .store(0, std::sync::atomic::Ordering::Release);
        d.state.registers.gfx.fifo_written = PACKET_HEADER_LEN + 4;
        d.state.scheduling.pending.request_main();
        d.drain(&mut h);
        assert!(d.state.scheduling.pending.active_child_mask() & (1 << ch) != 0);
    }

    #[test]
    fn doorbell_sets_pending_child() {
        let mut d = dev();
        let mut h = FakeHost::new();
        d.state.scheduling.pending.replace_active_children(1 << 3);
        // Child doorbells publish work and schedule the BH; the vCPU MMIO path
        // must not synchronously consume render work.
        d.gfx_write(&mut h, GFX_REG_CHILD_DOORBELL, 3, MMIO_U32);
        assert_eq!(
            d.state.scheduling.pending.child_mask(),
            1 << 3,
            "doorbell leaves pending work for the BH"
        );
        assert!(
            d.state.scheduling.pending.active_child_mask() & (1 << 3) != 0,
            "doorbell keeps channel active"
        );
        assert!(
            h.bh_scheduled,
            "doorbell schedules BH for HostAction delivery"
        );
    }
}
