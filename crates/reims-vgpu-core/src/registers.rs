//! Guest-visible register-bank state.

use std::collections::BTreeMap;
use std::sync::atomic::AtomicU32;
use std::sync::Arc;

/// Gfx MMIO window size. Sparse register state is bounded by the region the
/// transport exposes to the guest.
pub const GFX_MMIO_SIZE: u64 = 0x4000;

/// Gfx named registers plus sparse backing for unnamed offsets.
#[derive(Clone, Debug)]
pub struct GfxRegisters {
    pub version: u32,
    pub control_fifo: u32,
    pub fifo_length: u32,
    pub fifo_written: u32,
    pub fifo_read: Arc<AtomicU32>,
    pub fifo_start: u32,
    pub root_page: u32,
    pub fifo_base_page: u32,
    pub interrupt_status_disp: Arc<AtomicU32>,
    pub interrupt_status_gpu: Arc<AtomicU32>,
    pub interrupt_fault: Arc<AtomicU32>,
    pub child_doorbell_rung: Arc<AtomicU32>,
    pub efi_display: u32,
    pub efi_mode_select: u32,
    pub efi_fb_start: u64,
    pub efi_fb_length: u32,
    pub efi_fb_depth: u32,
    pub efi_fb_mode: u32,
    pub efi_fb_stride: u32,
    sparse: BTreeMap<u32, u32>,
}

impl Default for GfxRegisters {
    fn default() -> Self {
        Self {
            version: 0,
            control_fifo: 0,
            fifo_length: 0,
            fifo_written: 0,
            fifo_read: Arc::new(AtomicU32::new(0)),
            fifo_start: 0,
            root_page: 0,
            fifo_base_page: 0,
            interrupt_status_disp: Arc::new(AtomicU32::new(0)),
            interrupt_status_gpu: Arc::new(AtomicU32::new(0)),
            interrupt_fault: Arc::new(AtomicU32::new(0)),
            child_doorbell_rung: Arc::new(AtomicU32::new(0)),
            efi_display: 0,
            efi_mode_select: 0,
            efi_fb_start: 0,
            efi_fb_length: 0,
            efi_fb_depth: 0,
            efi_fb_mode: 0,
            efi_fb_stride: 0,
            sparse: BTreeMap::new(),
        }
    }
}

impl GfxRegisters {
    pub fn sparse_get(&self, offset: u64) -> u32 {
        let index = (offset / 4) as u32;
        self.sparse.get(&index).copied().unwrap_or(0)
    }

    pub fn sparse_set(&mut self, offset: u64, value: u32) {
        if offset < GFX_MMIO_SIZE {
            self.sparse.insert((offset / 4) as u32, value);
        }
    }
}

/// IOSurface command-ring register bank.
#[derive(Clone, Debug, Default)]
pub struct IosfcRegisters {
    pub ring_base: u64,
    pub capacity: u32,
    pub desc_table: u64,
    pub producer: u32,
    pub consumer: u32,
}

/// Both transport register banks owned by one device instance.
#[derive(Clone, Debug, Default)]
pub struct DeviceRegisters {
    pub gfx: GfxRegisters,
    pub iosfc: IosfcRegisters,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_gfx_state_cannot_escape_the_exposed_window() {
        let mut registers = GfxRegisters::default();
        registers.sparse_set(GFX_MMIO_SIZE - 4, 7);
        registers.sparse_set(GFX_MMIO_SIZE, 9);
        assert_eq!(registers.sparse_get(GFX_MMIO_SIZE - 4), 7);
        assert_eq!(registers.sparse_get(GFX_MMIO_SIZE), 0);
    }
}
