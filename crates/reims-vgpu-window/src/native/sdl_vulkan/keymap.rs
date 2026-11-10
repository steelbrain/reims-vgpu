//! SDL's scancodes to the USB HID keyboard usages (page 0x07) the guest's keyboard sends.
//!
//! SDL names a key by its position (its scancode, from the host's evdev code on Linux), and its
//! scancodes up to 0xE7 are the USB usages themselves: a key reaches the guest by position, so
//! the guest's own layout applies, and the ISO key left of 1 and the one right of left Shift are
//! the guest's own. The scancodes past 0xE7 (media and application keys) are SDL's own, with no
//! usage on the keyboard page: they are dropped, as are 0 to 3, which name no key.

/// The first usage of a key (`a`).
const FIRST: u16 = 0x04;
/// The last usage of a key (right GUI, which a Mac guest reads as right ⌘).
const LAST: u16 = 0xE7;

/// The usage of the key SDL's `scancode` names, or `None` for one with no usage on the keyboard
/// page.
pub(crate) fn usb_usage(scancode: i32) -> Option<u16> {
    u16::try_from(scancode)
        .ok()
        .filter(|usage| (FIRST..=LAST).contains(usage))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scancode up to 0xE7 is its USB usage: letters, Return, the ISO key, the modifiers
    #[test]
    fn scancodes_are_usb_usages() {
        assert_eq!(usb_usage(4), Some(0x04), "a");
        assert_eq!(usb_usage(40), Some(0x28), "Return");
        assert_eq!(
            usb_usage(100),
            Some(0x64),
            "the ISO key right of left Shift"
        );
        assert_eq!(usb_usage(53), Some(0x35), "the key left of 1");
        assert_eq!(usb_usage(224), Some(0xE0), "left Control");
        assert_eq!(
            usb_usage(227),
            Some(0xE3),
            "left GUI: the guest's left Command"
        );
        assert_eq!(usb_usage(231), Some(0xE7), "right GUI");
    }

    /// SDL's own scancodes, and those that name no key, are dropped
    #[test]
    fn keys_off_the_keyboard_page_are_dropped() {
        for scancode in [-1, 0, 1, 3, 232, 257, 258, 290, i32::MAX] {
            assert_eq!(usb_usage(scancode), None, "{scancode}");
        }
    }
}
