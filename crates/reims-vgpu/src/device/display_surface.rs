//! What QEMU's display front-end asks the device for: which source owns the
//! console, whether a mapping may paint, the scanout and EFI-console copies,
//! and the cursor glyph.
//!
//! One of the four jobs the crate root used to hold. Every item here is an
//! entry point `crate::qemu::abi` calls or a type it passes across the C boundary, so
//! the whole module is the display half of the QEMU ABI surface and nothing
//! else. It reaches back into the root for the registry only — `device_slot`,
//! `BoundDevice`, `DeviceInner` — which is why the seam is here: three symbols
//! out, and, before the move, three prose mentions of `device_scanout_copy` in.
//!
//! `use super::*` rather than a named list, for the reason
//! `crate::runtime::draw::execution` gives and `window_publish` repeats: a chapter of the
//! root lifted out whole should not read as a redesign.

use super::*;

/// Which source owns the host console right now.
///
/// The three arms are exhaustive and ordered: see [`device_console_feed`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsoleFeed {
    /// BAR1 UEFI GOP, or the guest-programmed `efi_fb_start` when it has one.
    Firmware,
    /// A latched same-geometry early front writeback — the Apple logo and
    /// progress pill, before the compositor's first present.
    Early {
        mapping_id: u32,
        width: u32,
        height: u32,
        generation: u32,
    },
    /// The compositor has presented; product present owns the console.
    Product,
}

impl ConsoleFeed {
    /// The `REIMS_VGPU_CONSOLE_FEED_*` discriminant the QEMU shims switch on.
    /// Must match `include/reims_vgpu_qemu_abi.h`; see
    /// `the_abi_header_agrees_on_the_console_feed_kinds`.
    pub fn kind(&self) -> u32 {
        match self {
            Self::Firmware => 0,
            Self::Early { .. } => 1,
            Self::Product => 2,
        }
    }
}

/// The whole console-ownership decision, from protocol state only.
///
/// **Not** content, sparsity, boot stage, or any screenshot heuristic.
///
/// Both QEMU shims used to reach this answer themselves, by calling
/// `present_boundary_seen` and `early_scanout_target` and branching on the pair.
/// That put the rule in C twice over, beside a third copy here
/// (`host_console_uses_bar1`, which the host-window pump uses — not a doc
/// link, because that predicate is gated with the pump and so does not exist
/// on a build without `host-window`) whose doc
/// admitted as much — it said "shipped C mirrors this". Three copies of one
/// display-ownership rule is three chances for a pathway to disagree about who
/// owns the screen. This is the one copy; the shims paint what it returns.
///
/// The boundary check is deliberately the **monotonic** latch and not
/// `state.presentation.present.content_boundary_crossed()`. The latter is not monotonic across reset — a flush-less
/// (ClearOnly) present clears it — and on the arm64 MMIO console that re-armed
/// the early paint mid-session, flickering stale pre-boundary GOP content (the
/// Apple logo at the old geometry) against live product presents.
///
/// Returns `None` only when `id` names no device.
pub fn device_console_feed(id: u64) -> Option<ConsoleFeed> {
    let slot = device_slot(id)?;
    if slot.present_boundary_seen.load(Ordering::Acquire) {
        return Some(ConsoleFeed::Product);
    }
    // A contended device is reported as `Firmware` rather than as an error: the
    // caller is a display tick, the early console is the pre-boundary source by
    // definition, and the alternative — failing the call — makes the shim invent
    // a policy for "no answer", which is the thing being removed.
    let Some(d) = slot.inner.try_lock() else {
        return Some(ConsoleFeed::Firmware);
    };
    Some(
        match crate::runtime::scanout::early_scanout_target(&d.device) {
            // `Early` guarantees a paintable target, so the shims do not re-test
            // the geometry — both used to, which is a decode rule living in C.
            // A zero here is not a target, it is the absence of one.
            Some((mapping_id, width, height, generation))
                if mapping_id != 0 && width != 0 && height != 0 =>
            {
                ConsoleFeed::Early {
                    mapping_id,
                    width,
                    height,
                    generation,
                }
            }
            _ => ConsoleFeed::Firmware,
        },
    )
}

/// Whether a present naming `mapping_id` may paint the host console.
///
/// This is the second half of the console-ownership rule, and it belonged in the
/// shims for exactly as long as it took one of them to disagree. The x86 PCI
/// shim held it — `kind == FIRMWARE || (kind == EARLY && latched != mapping_id)`,
/// assembled from [`device_console_feed`]'s kind and mapping out-params — and the
/// arm64 MMIO shim held nothing at all and painted every present it was handed.
/// One rule, two pathways, one of them missing it: the same shape as the GPA
/// attrs drift, on the question of who owns the screen.
///
/// The three arms, from protocol state only:
///
/// - [`ConsoleFeed::Product`]: the compositor owns the console, so every present
///   paints.
/// - [`ConsoleFeed::Early`]: only the latched front may paint. A clear-only
///   present naming some other mapping must not steal the surface from the
///   firmware console underneath it.
/// - [`ConsoleFeed::Firmware`]: the guest is still on BAR1 / `efi_fb`; nothing
///   presented here paints.
///
/// Returns `None` only when `id` names no device.
pub fn device_scanout_may_paint(id: u64, mapping_id: u32) -> Option<bool> {
    Some(match device_console_feed(id)? {
        ConsoleFeed::Product => true,
        ConsoleFeed::Early {
            mapping_id: latched,
            ..
        } => latched == mapping_id,
        ConsoleFeed::Firmware => false,
    })
}

/// [`ConsoleFeed::Firmware`], for a caller that already holds device state.
///
/// From **protocol state only** — not content, sparsity, boot stage, or any
/// screenshot heuristic:
///
/// - `frame_flush_seen` (DisplaySwap / x86 present): product owns the console.
/// - Else if `early_front_latched` (same-geom IOSurface texture front writeback): the
///   product early paint, logo + pill, before the swap.
/// - Else: BAR1 / guest-programmed `efi_fb_start` (UEFI + PE log console).
///
/// This doc used to say "shipped C mirrors this", and it did — that mirroring is
/// what [`device_console_feed`] removed, so the QEMU shims now ask rather than
/// reconstruct. What is left here is not a second copy of the rule for them; it
/// is the form the host-window early pump needs, which runs inside the drain
/// with `&DeviceState` in hand and so cannot take `device_console_feed`'s
/// by-id, lock-free path.
///
/// The two are deliberately **not** identical, and the difference is the
/// `frame_flush_seen` argument: `device_console_feed` reads the monotonic
/// boundary latch instead, because a flush-less (ClearOnly) present clears
/// `frame_flush_seen` and a console that re-armed on that flickered stale
/// pre-boundary content. This predicate takes whatever its caller passes, and
/// its caller passes the non-monotonic flag.
///
/// Gated with its caller: the pump lives in `window_publish`, which is behind
/// `host-window`, so a build without that feature has nobody to ask.
#[cfg(any(test, feature = "host-window"))]
#[inline]
pub fn host_console_uses_bar1(frame_flush_seen: bool, early_front_latched: bool) -> bool {
    !frame_flush_seen && !early_front_latched
}

/// Copy guest EFI console FB (programmed at 0x1210) into a host BGRA8 surface.
///
/// `false` if no efi_fb_start or GPA read fails — C uses BAR1 then.
pub fn device_efi_console_copy(
    id: u64,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
) -> bool {
    let Some(slot) = device_slot(id) else {
        return false;
    };
    let Some(mut d) = slot.inner.try_lock() else {
        return false;
    };
    let executor = Arc::clone(&d.device.executor);
    let _scope = executor.enter();
    if d.device.state.registers.gfx.efi_fb_start == 0 {
        return false;
    }
    let Some(ops) = slot.ops else {
        return false;
    };
    let DeviceInner { device, actions } = &mut *d;
    let host = QemuHost::new(&ops, actions, &slot.prompt_actions);
    crate::runtime::scanout::paint_efi_console(device, &host, dst, dst_stride, width, height)
}

/// Fill a host BGRA8 framebuffer from the named guest mapping (or EFI FB).
///
/// `dst` is a row-major buffer with `dst_stride` bytes per row. QEMU passes
/// `surface_data` / `surface_stride` from the DisplaySurface.
/// `generation` is the HostAction stamp (0 = always paint).
pub fn device_scanout_copy(
    id: u64,
    mapping_id: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    generation: u32,
) -> crate::runtime::scanout::ScanoutCopyResult {
    use crate::runtime::scanout::ScanoutCopyResult;
    let Some(slot) = device_slot(id) else {
        return ScanoutCopyResult::Failed;
    };
    let present_action = slot.present_action_pending.load(Ordering::Acquire);
    let mut d = if present_action {
        // The worker observes `present_action_pending` before taking this lock
        // and returns immediately. Blocking here is therefore bounded to that
        // short handoff; pre-boundary periodic scanout remains nonblocking.
        slot.inner.lock()
    } else {
        let Some(d) = slot.inner.try_lock() else {
            return ScanoutCopyResult::Unchanged;
        };
        d
    };
    let executor = Arc::clone(&d.device.executor);
    let _scope = executor.enter();
    if let Some(ops) = slot.ops {
        let DeviceInner { device, actions } = &mut *d;
        let mut host = QemuHost::new(&ops, actions, &slot.prompt_actions);
        // `frame_encode_pending` also means a valid +0x188 snapshot was just
        // installed and must be blitted once. `copy_to_bgra8` distinguishes
        // that case from capture-fail retry without draining guest commands in
        // this QEMU display context; do not discard the only scanout action.
        let rc = crate::runtime::scanout::copy_to_bgra8(
            device, &mut host, mapping_id, dst, dst_stride, width, height, generation,
        );
        note_scanout_copy_consumed(device, &mut host, &slot, rc, present_action);
        rc
    } else {
        // No QEMU ops table is bound (unit tests / headless create), so every
        // guest read fails and the copy falls back to a black clear.
        let mut host = NullHost;
        let rc = crate::runtime::scanout::copy_to_bgra8(
            &mut d.device,
            &mut host,
            mapping_id,
            dst,
            dst_stride,
            width,
            height,
            generation,
        );
        note_scanout_copy_consumed(&mut d.device, &mut host, &slot, rc, present_action);
        rc
    }
}

/// Entry-side `waitForPendingFrames`: what one `copy_to_bgra8` owes the device
/// once it returns.
///
/// Two conditions, one rule. A copy that painted — or that found nothing to
/// redo — frees the DisplaySwap packet held at channel head. A copy that
/// *failed* owes exactly the same whenever `present_action` was set, because
/// the host consumed the action either way and C cannot replay it; leaving it
/// outstanding strands every later channel behind it.
///
/// The stamp for accepted presents fires at retain, not here.
///
/// This lives apart from [`device_scanout_copy`] because that function picks
/// between two hosts and the rule is the same under both. It was written out
/// once per arm before, and the two copies had already drifted: the headless
/// arm cleared the pending flag after a failed copy without recording the
/// consumption, which is the stranding the paragraph above forbids.
fn note_scanout_copy_consumed<H: HostOps>(
    state: &mut crate::runtime::Device,
    host: &mut H,
    slot: &BoundDevice,
    rc: crate::runtime::scanout::ScanoutCopyResult,
    present_action: bool,
) {
    use crate::runtime::scanout::ScanoutCopyResult;
    let painted = matches!(
        rc,
        ScanoutCopyResult::Painted | ScanoutCopyResult::Unchanged
    );
    if !painted && !present_action {
        return;
    }
    crate::runtime::drain::note_present_paint_consumed(state);
    slot.present_action_pending.store(false, Ordering::Release);
    host.schedule_bh();
}

/// Cursor glyph metadata for the QEMU console.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CursorGlyphInfo {
    pub width: u32,
    pub height: u32,
    pub hot_x: u32,
    pub hot_y: u32,
    pub pixel_count: u32,
}

pub fn device_cursor_glyph_info(id: u64) -> Option<CursorGlyphInfo> {
    let slot = device_slot(id)?;
    let d = slot.inner.try_lock()?;
    let c = d.device.state.presentation.cursor.glyph()?;
    Some(CursorGlyphInfo {
        width: c.width as u32,
        height: c.height as u32,
        hot_x: c.hot_x as u32,
        hot_y: c.hot_y as u32,
        pixel_count: c.pixels.len() as u32,
    })
}

/// Copy QEMUCursor ARGB pixels. Returns number of pixels written.
pub fn device_cursor_glyph_copy(id: u64, out: &mut [u32]) -> Option<usize> {
    let slot = device_slot(id)?;
    let d = slot.inner.try_lock()?;
    let c = d.device.state.presentation.cursor.glyph()?;
    let n = c.pixels.len().min(out.len());
    out[..n].copy_from_slice(&c.pixels[..n]);
    Some(n)
}
