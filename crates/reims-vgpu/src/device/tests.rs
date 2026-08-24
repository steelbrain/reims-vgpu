//! Registry and QEMU-entry-surface tests for the crate root.
//!
//! Out of line for the reason the four `runtime/` modules that already do this
//! have: colocated, these 490 lines were a quarter of `lib.rs` and sat between
//! a reader and the entry points they document.

use super::*;
use crate::model::PAGE_SHIFT_ARM64E;

#[test]
fn lifecycle() {
    let id = device_create(None, PAGE_SHIFT_ARM64E).expect("create");
    assert_ne!(id, 0);
    assert!(device_reset(id));
    assert!(device_destroy(id));
    assert!(!device_destroy(id));
}

#[test]
fn backend_name_is_vulkan() {
    assert_eq!(backend_name(), "vulkan");
}

#[test]
fn panic_does_not_escape() {
    let v = unwind_safe("reims_vgpu_qemu_device_tests", || panic!("boom"), 42i32);
    assert_eq!(v, 42);
}

#[test]
fn mmio_hooks() {
    let id = device_create(None, PAGE_SHIFT_ARM64E).expect("create");
    assert!(device_gfx_write(id, 0x1034, 0x3e, 4));
    assert_eq!(device_gfx_read(id, 0x1034, 4), Some(0x3e));
    assert!(device_iosfc_write(id, 0x1008, 0x400, 4));
    assert_eq!(device_iosfc_read(id, 0x1008, 4), Some(0x400));
    assert!(device_destroy(id));
}

/// A copy that fails still owes the consumption when a present action was
/// pending, whichever host ran it.
///
/// The headless arm used to clear `present_action_pending` and leave
/// `unpainted_presents` standing, so the flag said "nothing outstanding"
/// while the backpressure counter said the opposite — the stranding
/// [`note_scanout_copy_consumed`] exists to prevent, on the one arm nobody
/// was reading.
#[test]
fn a_failed_copy_still_frees_a_consumed_present_action() {
    let id = device_create(None, PAGE_SHIFT_ARM64E).expect("create");
    let slot = device_slot(id).expect("slot");
    slot.inner
        .lock()
        .device
        .state
        .presentation
        .present
        .set_unpainted_presents_for_test(3);
    slot.present_action_pending.store(true, Ordering::Release);

    let mut dst = [0u8; 4];
    let rc = device_scanout_copy(id, 7, &mut dst, 4, 1, 1, 0);
    assert_eq!(
        rc,
        crate::runtime::scanout::ScanoutCopyResult::Failed,
        "a headless device has no guest memory to paint from"
    );

    assert!(
        !slot.present_action_pending.load(Ordering::Acquire),
        "the host consumed the action; C cannot replay it"
    );
    assert_eq!(
        slot.inner
            .lock()
            .device
            .state
            .presentation
            .present
            .unpainted_presents(),
        0,
        "and the backpressure counter must agree with the flag"
    );
    assert!(device_destroy(id));
}

#[test]
fn drain_without_ops_is_ok() {
    let id = device_create(None, PAGE_SHIFT_ARM64E).expect("create");
    assert!(device_drain(id));
    assert!(device_pop_action(id).is_none());
    assert!(device_destroy(id));
}

/// The device drain owns one backend transaction, including work performed
/// after the FIFO body. A session scope around `Device::drain` alone ends too
/// early: the submission-tail flush then runs against another device's/default
/// backend state and the window cannot see the resident just rendered.
#[test]
fn drain_tail_remains_in_the_device_executor_session() {
    use crate::runtime::executor::*;
    use reims_vgpu_core::{
        CapabilityService, ComputeResidencyService, ExecutionPort, GuestWriteService,
        PresentationService, ReadbackService, ResidentService,
    };
    use std::sync::atomic::{AtomicBool, AtomicUsize};

    #[derive(Debug)]
    struct ScopeGuard(Arc<AtomicUsize>);

    impl Drop for ScopeGuard {
        fn drop(&mut self) {
            self.0.fetch_sub(1, Ordering::AcqRel);
        }
    }

    #[derive(Debug)]
    struct ScopeProbe {
        depth: Arc<AtomicUsize>,
        tail_was_scoped: AtomicBool,
    }

    impl ExecutionPort for ScopeProbe {
        type Submission = ResolvedSubmission;
        type Completion = ExecutionCompletion;
        type Error = DrawError;

        fn execute(&self, _submission: Self::Submission) -> Result<Self::Completion, Self::Error> {
            unreachable!("an empty drain executes no command buffer")
        }
    }
    impl ResidentService for ScopeProbe {}
    impl GuestWriteService for ScopeProbe {}
    impl ComputeResidencyService for ScopeProbe {}
    impl CapabilityService for ScopeProbe {}
    impl PresentationService for ScopeProbe {}
    impl ReadbackService for ScopeProbe {
        type Error = DrawError;

        fn read_target(
            &self,
            _identity: &crate::model::TargetIdentity,
        ) -> Result<reims_vgpu_core::TargetReadback, Self::Error> {
            unreachable!("an empty drain reads no target")
        }
    }
    impl GuestPageTransferService for ScopeProbe {}
    impl ResidentCopyService for ScopeProbe {}
    impl CompletionService for ScopeProbe {}
    impl SubmissionBatchService for ScopeProbe {
        fn flush_submission_tail(&self) {
            self.tail_was_scoped
                .store(self.depth.load(Ordering::Acquire) != 0, Ordering::Release);
        }
    }
    impl GuestImportService for ScopeProbe {}
    impl MaintenanceService for ScopeProbe {}
    impl SessionService for ScopeProbe {
        fn enter(&self) -> ExecutionScope {
            self.depth.fetch_add(1, Ordering::AcqRel);
            ExecutionScope::test(ScopeGuard(Arc::clone(&self.depth)))
        }
    }
    impl ObservationService for ScopeProbe {}
    impl ShaderTranslationService for ScopeProbe {}
    impl RenderBufferPlanningService for ScopeProbe {}
    impl crate::runtime::executor::GuestImagePlanningService for ScopeProbe {}
    impl WindowPresentationService for ScopeProbe {}
    impl Executor for ScopeProbe {}

    let probe = Arc::new(ScopeProbe {
        depth: Arc::new(AtomicUsize::new(0)),
        tail_was_scoped: AtomicBool::new(false),
    });
    let id = device_create(Some(ReimsVgpuHostOps::null()), PAGE_SHIFT_ARM64E).expect("create");
    let slot = device_slot(id).expect("slot");
    slot.inner.lock().device.executor = probe.clone();

    assert!(device_drain(id));
    assert!(
        probe.tail_was_scoped.load(Ordering::Acquire),
        "the submission tail must observe the same session as the drained work"
    );
    assert_eq!(
        probe.depth.load(Ordering::Acquire),
        0,
        "the complete drain transaction releases its session before returning"
    );
    assert!(device_destroy(id));
}

#[cfg(all(feature = "host-window", target_os = "macos"))]
#[test]
fn window_publish_key_advances_for_in_place_present() {
    use super::window_publish::window_frame_key;
    let mut state = crate::model::DeviceState::new(crate::model::DeviceId(1), PAGE_SHIFT_ARM64E);
    state
        .presentation
        .present
        .set_frame_identity_for_test(7, 0, 0, 11, 0);
    let first = window_frame_key(&state.presentation.present);

    state.advance_present_epoch();
    assert_ne!(
        window_frame_key(&state.presentation.present),
        first,
        "a repeated resource generation still represents a new DisplaySwap"
    );
}

/// A host publication can advance without a guest-page generation change, so
/// the host window's publish key includes the content epoch.
#[cfg(feature = "host-window")]
#[test]
fn window_publish_key_moves_for_a_host_publication() {
    use super::window_publish::window_frame_key;
    let mut state = crate::model::DeviceState::new(crate::model::DeviceId(1), PAGE_SHIFT_ARM64E);
    state.set_mapping_geom(7, 8, 4, 0x1e);

    fn publish(state: &mut crate::model::DeviceState) {
        let epoch = state.note_surface_content_published(7);
        let generation = state
            .surfaces
            .mappings
            .get(&7)
            .expect("mapping 7")
            .content
            .guest_page_generation;
        state
            .presentation
            .present
            .set_frame_identity_for_test(7, 0, 0, generation, epoch);
    }

    state
        .presentation
        .present
        .set_frame_identity_for_test(7, 0, 0, 0, 0);
    publish(&mut state);
    let first = window_frame_key(&state.presentation.present);
    let generation = state.presentation.present.frame().generation();

    publish(&mut state);

    assert_eq!(
        state.presentation.present.frame().generation(),
        generation,
        "host publication does not itself change the guest-page generation"
    );
    assert_ne!(
        window_frame_key(&state.presentation.present),
        first,
        "two host publications into one surface are two frames and must publish twice"
    );
}

/// The guest ISR read of the read-to-clear interrupt-status registers
/// must observe live bits (and clear them) even while the drain worker
/// owns the device lock — a stale cached mask loses stamp signals.
#[test]
fn intr_status_reads_are_live_while_device_lock_held() {
    let id = device_create(None, PAGE_SHIFT_ARM64E).expect("create");
    let slot = device_slot(id).expect("slot");
    let _drain_guard = slot.inner.lock();
    // Drain-side signal lands while the lock is held.
    slot.intr_gpu.fetch_or(0x21, Ordering::AcqRel);
    slot.intr_disp.fetch_or(0x1, Ordering::AcqRel);
    // ISR sees live bits; second read is clear (read-to-clear).
    assert_eq!(device_gfx_read(id, 0x1018, 4), Some(0x21));
    assert_eq!(device_gfx_read(id, 0x1018, 4), Some(0));
    assert_eq!(device_gfx_read(id, 0x1014, 4), Some(0x1));
    assert_eq!(device_gfx_read(id, 0x1014, 4), Some(0));
    drop(_drain_guard);
    assert!(device_destroy(id));
}

/// The main-FIFO consumer counter (0x100c) must show drain progress live
/// while the device lock is held — the guest writeFifo producer spins on
/// it and a cached pre-tranche snapshot stalls the producer for the whole
/// tranche.
#[test]
fn fifo_read_counter_is_live_while_device_lock_held() {
    let id = device_create(None, PAGE_SHIFT_ARM64E).expect("create");
    let slot = device_slot(id).expect("slot");
    let _drain_guard = slot.inner.lock();
    slot.fifo_read_live.store(0x1234, Ordering::Release);
    assert_eq!(device_gfx_read(id, 0x100c, 4), Some(0x1234));
    slot.fifo_read_live.store(0x1300, Ordering::Release);
    assert_eq!(device_gfx_read(id, 0x100c, 4), Some(0x1300));
    drop(_drain_guard);
    assert!(device_destroy(id));
}

/// Interrupt-status mask-clear writes apply lock-free too.
#[test]
fn intr_status_write_clears_mask_while_device_lock_held() {
    let id = device_create(None, PAGE_SHIFT_ARM64E).expect("create");
    let slot = device_slot(id).expect("slot");
    let _drain_guard = slot.inner.lock();
    slot.intr_gpu.fetch_or(0x7, Ordering::AcqRel);
    assert!(device_gfx_write(id, 0x1018, 0x2, 4));
    assert_eq!(device_gfx_read(id, 0x1018, 4), Some(0x5));
    drop(_drain_guard);
    assert!(device_destroy(id));
}

/// Prompt actions (IRQ pulses) pop without the device lock so the BH can
/// deliver MSIs mid-drain; lock-owning actions still wait for the lock.
#[test]
fn prompt_actions_pop_while_device_lock_held() {
    let id = device_create(None, PAGE_SHIFT_ARM64E).expect("create");
    let slot = device_slot(id).expect("slot");
    slot.prompt_actions.lock().push_back(HostAction::irq_gfx());
    let _drain_guard = slot.inner.lock();
    let a = device_pop_action(id).expect("prompt action pops mid-drain");
    assert_eq!(a.kind, crate::runtime::host::HostActionKind::IrqGfxPulse);
    assert!(device_pop_action(id).is_none());
    drop(_drain_guard);
    assert!(device_destroy(id));
}

/// The interrupt-status atomics stay wired to the same slot across reset
/// ([`crate::model::DeviceState::reset`] must preserve the shared `Arc`s and only
/// zero the values they hold).
#[test]
fn intr_status_atomics_survive_reset() {
    let id = device_create(None, PAGE_SHIFT_ARM64E).expect("create");
    let slot = device_slot(id).expect("slot");
    slot.intr_gpu.fetch_or(0xff, Ordering::AcqRel);
    assert!(device_reset(id));
    // Reset cleared pending bits.
    assert_eq!(device_gfx_read(id, 0x1018, 4), Some(0));
    // Post-reset signals still reach the lock-free read rail.
    {
        let d = slot.inner.lock();
        d.device
            .state
            .registers
            .gfx
            .interrupt_status_gpu
            .fetch_or(0x9, Ordering::AcqRel);
    }
    assert_eq!(device_gfx_read(id, 0x1018, 4), Some(0x9));
    assert!(device_destroy(id));
}

/// Pre-boundary without early front: BAR1/efi. Boundary or early latch: leave.
#[test]
fn host_console_bar1_until_present_boundary() {
    // (frame_flush, early_latched)
    assert!(host_console_uses_bar1(false, false));
    assert!(!host_console_uses_bar1(true, false));
    assert!(!host_console_uses_bar1(false, true));
    assert!(!host_console_uses_bar1(true, true));

    let id = device_create(None, PAGE_SHIFT_ARM64E).expect("create");
    assert_eq!(device_console_feed(id), Some(ConsoleFeed::Firmware));

    // Present bookkeeping without boundary must not leave the firmware feed
    // by itself — only frame_flush_seen or an early front latch.
    {
        let slot = device_slot(id).expect("device");
        let mut d = slot.inner.lock();
        d.device
            .state
            .presentation
            .present
            .establish_console(1920, 1080, 0);
        d.device
            .state
            .presentation
            .present
            .note_present_candidate(3);
    }
    assert_eq!(device_console_feed(id), Some(ConsoleFeed::Firmware));

    {
        let slot = device_slot(id).expect("device");
        let mut d = slot.inner.lock();
        d.device.state.presentation.present.cross_content_boundary();
        publish_present_boundary(
            &slot,
            d.device
                .state
                .presentation
                .present
                .content_boundary_crossed(),
        );
    }
    assert_eq!(device_console_feed(id), Some(ConsoleFeed::Product));
    assert!(device_destroy(id));
}

/// The paint verdict, over all three console feeds.
///
/// This is the rule the x86 shim used to assemble from `console_feed`'s kind
/// and mapping out-params while the arm64 shim assembled nothing and painted
/// unconditionally. The `Early` arm is the one that differed: a present
/// naming a mapping other than the latched front is a pre-boundary steal of
/// the firmware console, and only one pathway refused it.
#[test]
fn only_the_latched_front_may_paint_before_the_present_boundary() {
    use reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM;

    let id = device_create(None, PAGE_SHIFT_ARM64E).expect("create");

    // _FIRMWARE: the guest is still on BAR1 / efi_fb. Nothing presented
    // paints, whichever mapping it names.
    assert_eq!(device_console_feed(id), Some(ConsoleFeed::Firmware));
    assert_eq!(device_scanout_may_paint(id, 0), Some(false));
    assert_eq!(device_scanout_may_paint(id, 7), Some(false));

    // _EARLY: latch mapping 7 as the composited front.
    {
        let slot = device_slot(id).expect("device");
        let mut d = slot.inner.lock();
        let state = &mut d.device.state;
        state.presentation.present.establish_console(1920, 1080, 0);
        assert!(state.map_surface(7));
        let m = state.surfaces.mappings.get_mut(&7).unwrap();
        m.lifecycle.active = true;
        m.publish_geometry_for_test(1920, 1080, MTL_FORMAT_BGRA8_UNORM);
        m.content.guest_page_generation = 4;
        m.pages.entries = vec![1];
        state.note_surface_composite(7);
        state.presentation.present.note_early_composite(7);
    }
    assert!(matches!(
        device_console_feed(id),
        Some(ConsoleFeed::Early { mapping_id: 7, .. })
    ));
    assert_eq!(
        device_scanout_may_paint(id, 7),
        Some(true),
        "the latched early front is exactly what the pre-boundary console shows"
    );
    assert_eq!(
        device_scanout_may_paint(id, 8),
        Some(false),
        "a present naming any other mapping must not steal the surface from \
         the firmware console underneath it"
    );

    // _PRODUCT: the compositor owns the console, so every present paints —
    // including the mapping the early arm just refused.
    {
        let slot = device_slot(id).expect("device");
        publish_present_boundary(&slot, true);
    }
    assert_eq!(device_console_feed(id), Some(ConsoleFeed::Product));
    assert_eq!(device_scanout_may_paint(id, 7), Some(true));
    assert_eq!(device_scanout_may_paint(id, 8), Some(true));

    assert_eq!(
        device_scanout_may_paint(id.wrapping_add(1_000_000), 7),
        None,
        "an unknown device has no verdict to give; the shim must not paint"
    );
    assert!(device_destroy(id));
}

/// The kinds the shims switch on are a wire contract, not an internal
/// numbering: a shim built against a header whose `_EARLY` is 1 and a
/// staticlib whose `Early` is 2 would paint the firmware framebuffer for the
/// whole boot and report nothing. Nothing else compares the two.
#[test]
fn the_abi_header_agrees_on_the_console_feed_kinds() {
    use crate::qemu::abi::header_define as define;
    assert_eq!(
        define("REIMS_VGPU_CONSOLE_FEED_FIRMWARE"),
        ConsoleFeed::Firmware.kind()
    );
    assert_eq!(
        define("REIMS_VGPU_CONSOLE_FEED_EARLY"),
        ConsoleFeed::Early {
            mapping_id: 1,
            width: 1,
            height: 1,
            generation: 1,
        }
        .kind()
    );
    assert_eq!(
        define("REIMS_VGPU_CONSOLE_FEED_PRODUCT"),
        ConsoleFeed::Product.kind()
    );
}

#[test]
fn present_boundary_query_is_monotonic_and_lock_free() {
    let id = device_create(None, PAGE_SHIFT_ARM64E).expect("create");
    let slot = device_slot(id).expect("device");
    let inner = slot.inner.lock();

    // The lock is held for all of this. Before the boundary the answer is
    // `Firmware` — the contended device reports the pre-boundary source
    // rather than an error, because the caller is a display tick and making
    // it invent a policy for "no answer" is what moving the rule here
    // removes.
    assert_eq!(device_console_feed(id), Some(ConsoleFeed::Firmware));
    publish_present_boundary(&slot, true);
    assert_eq!(
        device_console_feed(id),
        Some(ConsoleFeed::Product),
        "QEMU refresh must read the boundary while the worker owns device state"
    );
    publish_present_boundary(&slot, false);
    assert_eq!(
        device_console_feed(id),
        Some(ConsoleFeed::Product),
        "the per-boot product-console boundary must not regress to firmware"
    );

    drop(inner);
    assert!(device_reset(id));
    assert_eq!(device_console_feed(id), Some(ConsoleFeed::Firmware));
    assert!(device_destroy(id));
}

/// Regression proxy for the IPI-timeout class: a doorbell arriving while
/// the render worker owns device state never waits for that state lock.
///
/// It used to queue, and the queue is what this asserted. That was the
/// weaker half of the guarantee: the guest's store retired, but the work it
/// rang for did not start until the worker's tranche ended, measured at up
/// to 45 ms and about a hundred rings a second
/// (`gfx_doorbell_delay off_0x1020`, which read `offsets=1` — this register
/// was the entire queueing stall on the PCI pathway). The ring is now taken
/// with no device lock asked for at all, so `gfx_ingress` must stay empty.
///
/// Both halves are asserted. An empty ingress alone would also be what a
/// dropped doorbell looks like, so the bit has to be shown arriving and then
/// shown becoming a pending channel.
#[test]
fn a_child_doorbell_never_queues_behind_the_render_worker() {
    let id = device_create(None, PAGE_SHIFT_ARM64E).expect("create");
    let slot = device_slot(id).expect("device");
    let inner = slot.inner.lock();

    assert!(device_gfx_write(
        id,
        crate::model::GFX_REG_CHILD_DOORBELL,
        4,
        crate::model::MMIO_U32,
    ));
    assert_eq!(
        slot.gfx_ingress.lock().len(),
        0,
        "the ring must not queue behind a held device lock"
    );
    assert_ne!(
        slot.child_doorbell_rung.load(Ordering::Acquire) & (1 << 4),
        0,
        "and must be recorded, or an empty queue is just a lost doorbell"
    );
    drop(inner);

    assert!(device_drain(id));
    let inner = slot.inner.lock();
    assert_ne!(
        inner.device.state.scheduling.pending.child_mask() & (1 << 4),
        0,
        "the fold must turn the ring into pending work"
    );
    assert_ne!(
        inner.device.state.scheduling.pending.active_child_mask() & (1 << 4),
        0,
        "and into an active channel, or the stranded-FIFO rescue cannot see it"
    );
    assert_eq!(
        inner
            .device
            .state
            .registers
            .gfx
            .child_doorbell_rung
            .load(Ordering::Acquire),
        0,
        "the fold consumes the bit rather than replaying it every drain"
    );
    drop(inner);
    assert!(device_destroy(id));
}

/// A ring naming no channel is dropped, not shifted by — and says so.
///
/// `1u32 << channel` is undefined past the word, and the locked handler in
/// `crate::runtime::mmio` has always range-checked before shifting. The lock-free
/// path is a second implementation of that same guard, so it gets its own
/// assertion rather than inheriting the first one's.
///
/// Ringing nothing is the correct action and is not the whole contract. The
/// guest is not told, and the commands it queued on that channel sit in the ring
/// forever — a stalled channel, which from the guest's side does not look like a
/// dropped record at all. So the refusal has to reach the fail channel, and it
/// has to name the channel, or no boot can say whether a guest has ever crossed
/// `MAX_CHANNELS` — a bound this device imposes and the protocol never states.
///
/// Fails without the fix: all three sites answered `is_child_channel` and said
/// nothing, so the capture is empty.
#[test]
fn a_child_doorbell_outside_the_channel_range_rings_nothing_and_reports_it() {
    let cap = crate::observe::FailCapture::start();
    let id = device_create(None, PAGE_SHIFT_ARM64E).expect("create");
    let slot = device_slot(id).expect("device");
    for channel in [0u64, crate::model::MAX_CHANNELS as u64, 0xffff_ffff] {
        assert!(device_gfx_write(
            id,
            crate::model::GFX_REG_CHILD_DOORBELL,
            channel,
            crate::model::MMIO_U32,
        ));
    }
    assert_eq!(
        slot.child_doorbell_rung.load(Ordering::Acquire),
        0,
        "channel 0 is the main FIFO and the rest name nothing"
    );
    assert_eq!(slot.gfx_ingress.lock().len(), 0, "and none of them queue");

    let reported: Vec<String> = cap
        .lines()
        .into_iter()
        .filter(|l| l.split_whitespace().next() == Some("child_channel_out_of_range"))
        .collect();
    assert_eq!(
        reported.len(),
        3,
        "one line per distinct refused channel — the latch is per channel id, \
         not per reason, so three ids are three lines: {reported:?}"
    );
    for (channel, line) in [0u32, crate::model::MAX_CHANNELS as u32, 0xffff_ffff]
        .iter()
        .zip(&reported)
    {
        assert!(
            line.contains(&format!("channel={channel}")),
            "the line must name the channel that was refused: {line}"
        );
        assert!(
            line.contains("reason=channel_outside_device_range"),
            "and carry a reason= so it ranks in the fail-channel queue: {line}"
        );
    }

    // Re-ringing the same channels is latched: the magnitude belongs to the
    // census route, not to a repeated line.
    for channel in [0u64, crate::model::MAX_CHANNELS as u64, 0xffff_ffff] {
        assert!(device_gfx_write(
            id,
            crate::model::GFX_REG_CHILD_DOORBELL,
            channel,
            crate::model::MMIO_U32,
        ));
    }
    assert_eq!(
        cap.lines()
            .iter()
            .filter(|l| l.split_whitespace().next() == Some("child_channel_out_of_range"))
            .count(),
        3,
        "a guest hammering an out-of-range doorbell costs one line per channel"
    );
    assert!(device_destroy(id));
}

#[test]
fn present_action_owns_worker_boundary_until_scanout_copy() {
    // Still valid as a test of `device_scanout_copy`'s own contract, which is
    // reachable for pre-boundary console paints and QMP screendump. Note the
    // production present path no longer depends on it: `device_drain` acks
    // each present itself after publishing to the host window, since no
    // per-present `ScanoutUpdate` is enqueued for QEMU to apply.
    let id = device_create(Some(ReimsVgpuHostOps::null()), PAGE_SHIFT_ARM64E).expect("create");
    let slot = device_slot(id).expect("device");
    {
        let mut inner = slot.inner.lock();
        let present = &mut inner.device.state.presentation.present;
        present.set_frame_identity_for_test(4, 2, 2, 7, 0);
        present.replace_frame_pixels_for_test(vec![0x55; 16]);
        present.validate_frame_for_test();
        present.set_unpainted_presents_for_test(1);
        inner
            .device
            .state
            .scheduling
            .pending
            .yield_for_host_action();
    }
    slot.present_action_pending.store(true, Ordering::Release);
    slot.gfx_ingress.lock().push_back(QueuedGfxWrite {
        offset: crate::model::GFX_REG_CHILD_DOORBELL,
        data: 4,
        size: crate::model::MMIO_U32,
        queued_at: Some(std::time::Instant::now()),
    });

    assert!(device_drain(id));
    assert_eq!(
        slot.gfx_ingress.lock().len(),
        1,
        "a newly woken worker must not overtake the queued scanout action"
    );

    let mut dst = vec![0u8; 16];
    assert_eq!(
        device_scanout_copy(id, 4, &mut dst, 8, 2, 2, 7),
        crate::runtime::scanout::ScanoutCopyResult::Painted
    );
    assert!(!slot.present_action_pending.load(Ordering::Acquire));
    assert!(!slot
        .inner
        .lock()
        .device
        .state
        .scheduling
        .pending
        .host_action_yielded());

    assert!(device_drain(id));
    assert_eq!(slot.gfx_ingress.lock().len(), 0);
    assert!(device_destroy(id));
}
