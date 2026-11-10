//! Windows/NVIDIA swapchain initialization.
//!
//! On RTX 3060 / NVIDIA 595.71, opening directly in MAILBOX selected the
//! legacy GDI-copy presentation path. One FIFO present before remaking the
//! same window's MAILBOX swapchain selected DXGI Flip instead (verified by
//! PresentMon). FIFO is universally supported; steady-state image count,
//! preferred mode and synchronization remain the swapchain's normal choices.

use ash::vk;

/// Select the first mode only. Swapchain::present restores the preferred mode
/// after the first successful present, through its existing retirement path.
pub(super) fn initial_present_mode(
    vendor_id: u32,
    preferred: vk::PresentModeKHR,
) -> vk::PresentModeKHR {
    if vendor_id == 0x10de && preferred == vk::PresentModeKHR::MAILBOX {
        vk::PresentModeKHR::FIFO
    } else {
        preferred
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// This is the actual policy at swapchain creation, not a second model of
    /// it: Windows NVIDIA must initialize FIFO before steady-state MAILBOX.
    #[test]
    fn nvidia_mailbox_initializes_fifo() {
        assert_eq!(
            initial_present_mode(0x10de, vk::PresentModeKHR::MAILBOX).as_raw(),
            vk::PresentModeKHR::FIFO.as_raw()
        );
    }

    /// A FIFO-only surface and other vendors retain the requested mode.
    #[test]
    fn other_initial_modes_stay_selected() {
        for vendor in [0x10de, 0x1002, 0x8086] {
            assert_eq!(
                initial_present_mode(vendor, vk::PresentModeKHR::FIFO).as_raw(),
                vk::PresentModeKHR::FIFO.as_raw()
            );
        }
        for vendor in [0x1002, 0x8086] {
            assert_eq!(
                initial_present_mode(vendor, vk::PresentModeKHR::MAILBOX).as_raw(),
                vk::PresentModeKHR::MAILBOX.as_raw()
            );
        }
    }
}
