//! Gfx and iosfc MMIO read/write handlers.
//!
//! Handlers only update bounded device state, set pending-work flags, and
//! schedule a host BH. No heavy decode or GPU work on this path.

use crate::model::FailEvent;
use crate::model::*;
use crate::runtime::host::{HostAction, HostMemory, HostOps};
use crate::runtime::mapper;
use crate::runtime::Device;

/// Gfx MMIO read (device-owned registers).
pub fn gfx_read(state: &mut Device, offset: u64, size: u32) -> u64 {
    if offset < REG_BASE {
        return 0;
    }

    if size == MMIO_U64 && offset == GFX_REG_EFI_FB_START {
        return state.registers.gfx.efi_fb_start;
    }
    if size == MMIO_U64 && offset == GFX_REG_FIFO_BASE_PAGE {
        return state.registers.gfx.fifo_base_page as u64;
    }
    if size == MMIO_U64 {
        let lo = gfx_read(state, offset, MMIO_U32);
        let hi = gfx_read(state, offset + MMIO_U32 as u64, MMIO_U32);
        return lo | (hi << 32);
    }
    if size != MMIO_U32 {
        state.record_fail(FailEvent::BadMmioAccess { offset, size });
        return 0;
    }

    match offset {
        GFX_REG_CONTROL_FIFO => state.registers.gfx.control_fifo as u64,
        GFX_REG_FIFO_LENGTH => state.registers.gfx.fifo_length as u64,
        GFX_REG_FIFO_WRITTEN => state.registers.gfx.fifo_written as u64,
        GFX_REG_FIFO_READ => state
            .registers
            .gfx
            .fifo_read
            .load(std::sync::atomic::Ordering::Acquire) as u64,
        GFX_REG_FIFO_START => state.registers.gfx.fifo_start as u64,
        GFX_REG_INTR_STATUS_DISP => state
            .registers
            .gfx
            .interrupt_status_disp
            .swap(0, std::sync::atomic::Ordering::AcqRel)
            as u64,
        GFX_REG_INTR_STATUS_GPU => state
            .registers
            .gfx
            .interrupt_status_gpu
            .swap(0, std::sync::atomic::Ordering::AcqRel) as u64,
        GFX_REG_ROOT_PAGE => state.registers.gfx.root_page as u64,
        GFX_REG_INTR_FAULT => state
            .registers
            .gfx
            .interrupt_fault
            .load(std::sync::atomic::Ordering::Acquire) as u64,
        GFX_REG_FIFO_BASE_PAGE => state.registers.gfx.fifo_base_page as u64,
        GFX_REG_VERSION => state.registers.gfx.version as u64,
        GFX_REG_EFI_DISPLAY => state.registers.gfx.efi_display as u64,
        GFX_REG_EFI_MODE_COUNT => EFI_MODE_COUNT as u64,
        GFX_REG_EFI_MODE_SELECT => state.registers.gfx.efi_mode_select as u64,
        GFX_REG_EFI_MODE_SIZE => {
            ((EFI_BOOT_WIDTH as u64) << EFI_MODE_WIDTH_SHIFT) | EFI_BOOT_HEIGHT as u64
        }
        GFX_REG_EFI_FB_START => state.registers.gfx.efi_fb_start & 0xffff_ffff,
        GFX_REG_EFI_FB_LENGTH => state.registers.gfx.efi_fb_length as u64,
        GFX_REG_EFI_FB_DEPTH => state.registers.gfx.efi_fb_depth as u64,
        GFX_REG_EFI_FB_MODE => state.registers.gfx.efi_fb_mode as u64,
        GFX_REG_EFI_STRIDE_ALIGN => EFI_STRIDE_ALIGNMENT as u64,
        GFX_REG_EFI_FB_STRIDE => state.registers.gfx.efi_fb_stride as u64,
        GFX_REG_EFI_DISPLAY_PORTS => EFI_DISPLAY_PORT_COUNT as u64,
        GFX_REG_EFI_BUILTIN_CONNECTED => EFI_BUILTIN_CONNECTED as u64,
        _ => state.registers.gfx.sparse_get(offset) as u64,
    }
}

/// Gfx MMIO write.
pub fn gfx_write<H: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut H,
    offset: u64,
    data: u64,
    size: u32,
) {
    if offset < REG_BASE {
        return;
    }

    if size == MMIO_U64 && offset == GFX_REG_EFI_FB_START {
        let prev = state.registers.gfx.efi_fb_start;
        state.registers.gfx.efi_fb_start = data;
        if data != prev {
            crate::observe::off(format!("efi_fb_start {prev:#x} -> {data:#x} (u64 write)"));
        }
        return;
    }
    if size == MMIO_U64 && offset == GFX_REG_FIFO_BASE_PAGE {
        state.registers.gfx.fifo_base_page = data as u32;
        return;
    }
    if size == MMIO_U64 {
        gfx_write(state, host, offset, data & 0xffff_ffff, MMIO_U32);
        gfx_write(state, host, offset + MMIO_U32 as u64, data >> 32, MMIO_U32);
        return;
    }
    if size != MMIO_U32 {
        state.record_fail(FailEvent::BadMmioAccess { offset, size });
        return;
    }

    let val = data as u32;
    match offset {
        GFX_REG_CONTROL_FIFO => {
            state.registers.gfx.control_fifo = val;
            if state.registers.gfx.control_fifo != 0 {
                state.scheduling.pending.request_main();
                // Doorbells only publish work. QEMU's one-shot BH drains after
                // this MMIO callback releases the device lock, keeping shader
                // translation/GPU waits off the guest vCPU and BQL path.
                host.schedule_bh();
            }
        }
        GFX_REG_FIFO_LENGTH => state.registers.gfx.fifo_length = val,
        GFX_REG_FIFO_WRITTEN => {
            state.registers.gfx.fifo_written = val;
            if state.registers.gfx.control_fifo != 0 {
                state.scheduling.pending.request_main();
                host.schedule_bh();
            }
        }
        GFX_REG_FIFO_START => state.registers.gfx.fifo_start = val,
        GFX_REG_INTR_STATUS_DISP => {
            state
                .registers
                .gfx
                .interrupt_status_disp
                .fetch_and(!val, std::sync::atomic::Ordering::AcqRel);
        }
        GFX_REG_INTR_STATUS_GPU => {
            state
                .registers
                .gfx
                .interrupt_status_gpu
                .fetch_and(!val, std::sync::atomic::Ordering::AcqRel);
        }
        GFX_REG_ROOT_PAGE => state.registers.gfx.root_page = val,
        GFX_REG_CHILD_DOORBELL | GFX_REG_CHILD_REPLAY_DOORBELL => {
            if crate::runtime::accept_child_channel(val, "mmio_child_doorbell") {
                state
                    .scheduling
                    .pending
                    .activate_and_request_children(1u32 << val);
                // Decode/execute belongs to the host BH, never the producer
                // vCPU's MMIO callback (ack-fast/render-async invariant).
                host.schedule_bh();
            }
        }
        GFX_REG_MAIN_KICK => {
            state.scheduling.pending.request_main();
            host.schedule_bh();
        }
        GFX_REG_INTR_FAULT => state
            .registers
            .gfx
            .interrupt_fault
            .store(val, std::sync::atomic::Ordering::Release),
        GFX_REG_FIFO_BASE_PAGE => state.registers.gfx.fifo_base_page = val,
        GFX_REG_VERSION => {
            // The guest writes the highest protocol version it speaks and then
            // reads this register back; the value it reads selects, in the
            // guest driver, an entire feature set — object tables, the child
            // doorbell, heaps, buffer-from-IOSurface, the FIFO depth. It is the
            // single widest-reaching thing the guest asks this device, and
            // until now it was an unobserved echo, so no log could say which
            // feature set the guest ended up configuring.
            let negotiated = negotiate_protocol_version(val);
            if negotiated != state.registers.gfx.version {
                crate::observe::off(format!(
                    "protocol_version requested={val} negotiated={negotiated} max={PROTOCOL_VERSION_MAX}"
                ));
            }
            state.registers.gfx.version = negotiated;
            // Take the guest-RAM import here rather than letting the first
            // gather take it. This handshake is the guest driver's first act
            // and its display pipe does not exist yet, so the seconds the
            // import costs on a multi-gigabyte RAMBlock are spent where nothing
            // is timing them. Left lazy it lands inside the first draw, inside
            // a display transaction the guest abandons after 1000 ms — see
            // `guest_ram_map::warm` for both halves of the cost and why neither
            // may run before the backend has published a granularity.
            crate::runtime::guest_ram_map::warm(host, state.executor.as_ref());
        }
        GFX_REG_EFI_DISPLAY => state.registers.gfx.efi_display = val,
        GFX_REG_EFI_MODE_SELECT => state.registers.gfx.efi_mode_select = val,
        GFX_REG_EFI_FB_START => {
            let prev = state.registers.gfx.efi_fb_start;
            state.registers.gfx.efi_fb_start =
                (state.registers.gfx.efi_fb_start & !0xffff_ffff) | (val as u64);
            if state.registers.gfx.efi_fb_start != prev {
                crate::observe::off(format!(
                    "efi_fb_start {prev:#x} -> {:#x} (u32 lo write)",
                    state.registers.gfx.efi_fb_start
                ));
            }
        }
        GFX_REG_EFI_FB_LENGTH => state.registers.gfx.efi_fb_length = val,
        GFX_REG_EFI_FB_DEPTH => state.registers.gfx.efi_fb_depth = val,
        GFX_REG_EFI_FB_MODE => state.registers.gfx.efi_fb_mode = val,
        GFX_REG_EFI_DISPLAY_IRQ => {
            // Dual-use: cursor sample doorbell + display IRQ bit.
            if val < 32 {
                state
                    .registers
                    .gfx
                    .interrupt_status_disp
                    .fetch_or(1u32 << val, std::sync::atomic::Ordering::AcqRel);
                // Sample cursor position from display shared page (GPA).
                if let Some(page) = state.presentation.display.shared_page() {
                    let mut pos = [0u8; 4];
                    if host
                        .read_gpa(page.gpa + DISPLAY_SHARED_CURSOR_POS, &mut pos)
                        .is_ok()
                    {
                        let packed = u32::from_le_bytes(pos);
                        if packed != 0xffff_ffff {
                            state
                                .presentation
                                .cursor
                                .set_position((packed & 0xffff) as u16, (packed >> 16) as u16);
                        }
                    }
                    let cursor = state.presentation.cursor.position();
                    host.enqueue(HostAction::cursor(cursor.x, cursor.y, cursor.visible));
                }
                host.enqueue(HostAction::irq_gfx());
                host.schedule_bh();
            }
        }
        GFX_REG_EFI_FB_STRIDE => {
            state.registers.gfx.efi_fb_stride = val;
            crate::observe::off(format!("efi_fb_stride -> {val:#x}"));
        }
        _ => state.registers.gfx.sparse_set(offset, val),
    }
}

/// Iosfc MMIO read.
pub fn iosfc_read(state: &Device, offset: u64, size: u32) -> u64 {
    let mut val = match offset {
        IOSFC_REG_RING_BASE => state.registers.iosfc.ring_base,
        IOSFC_REG_CAPACITY => state.registers.iosfc.capacity as u64,
        IOSFC_REG_DESC_TABLE => state.registers.iosfc.desc_table,
        IOSFC_REG_PRODUCER => state.registers.iosfc.producer as u64,
        IOSFC_REG_CONSUMER => state.registers.iosfc.consumer as u64,
        _ => 0,
    };
    if size < MMIO_U64 && size > 0 {
        let bits = (size as u64).saturating_mul(8).min(64);
        if bits < 64 {
            val &= (1u64 << bits) - 1;
        }
    }
    val
}

/// Iosfc MMIO write. Producer bump: capture directed handoff, then **drain
/// the mapper ring on this vCPU** (KVA page-table walks need `current_cpu`).
/// BH is only for HostAction delivery (IRQ / residual); never the sole place
/// that resolves MappingInternal — that deadlocks with `cpu_memory_rw_debug`
/// from the main loop while a vCPU holds the DEVICES lock in MMIO.
pub fn iosfc_write<H: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut H,
    offset: u64,
    data: u64,
    _size: u32,
) {
    match offset {
        IOSFC_REG_RING_BASE => state.registers.iosfc.ring_base = data,
        IOSFC_REG_CAPACITY => state.registers.iosfc.capacity = data as u32,
        IOSFC_REG_DESC_TABLE => state.registers.iosfc.desc_table = data,
        IOSFC_REG_PRODUCER => {
            let producer = data as u32;
            state.registers.iosfc.producer = producer;
            if state.registers.iosfc.consumer != producer {
                // Capture MappingInternal* while x19/x21/x22 still hold the
                // publishing vCPU's directed handoff.
                if let Some(cap) = mapper::capture_at_producer(state, host, producer) {
                    state.publish_mapper_capture(cap);
                }
                state.scheduling.pending.request_iosfc();
                // Sync drain on the publishing vCPU (historical product path;
                // rust rewrite had regressed to BH-only resolve).
                crate::runtime::drain::drain_iosfc(state, host);
                // Deliver enqueued IRQs / scanout actions on the main loop.
                host.schedule_bh();
            }
        }
        IOSFC_REG_CONSUMER => state.registers.iosfc.consumer = data as u32,
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{DeviceId, PAGE_SHIFT_X86};
    use crate::runtime::host::FakeHost;

    #[test]
    fn gfx_doorbell_only_publishes_work_for_bh() {
        let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();

        gfx_write(&mut state, &mut host, GFX_REG_CHILD_DOORBELL, 4, MMIO_U32);

        assert_ne!(state.scheduling.pending.child_mask() & (1 << 4), 0);
        assert_ne!(state.scheduling.pending.active_child_mask() & (1 << 4), 0);
        assert!(host.bh_scheduled);
    }
}
