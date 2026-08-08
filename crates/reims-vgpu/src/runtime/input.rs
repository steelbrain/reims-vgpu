//! Host-window input contract — the neutral codes the host-owned window
//! ([[host-window]]) uses to drive guest input through the QEMU C shim.
//!
//! Input flows window-thread → Rust → `Input*` [`crate::runtime::host::HostAction`]s
//! → the thread-safe action-delivery BH → thin C trampolines that call
//! `qemu_input_*`. The split keeps QEMU keycode/button constants out of Rust:
//!
//! - **Keys** ride the **Linux evdev** keycode space (`KEY_*`). The window
//!   thread maps its platform key to evdev; the C shim forwards the code
//!   verbatim to `qemu_input_event_send_key_linux`, which owns evdev→qcode. So
//!   neither side hard-codes a QEMU `QKeyCode`.
//! - **Buttons** ride [`ReimsVgpuButton`], a small stable enum owned by this crate.
//!   The C shim maps it to QEMU's `InputButton` with an explicit switch, so
//!   QEMU's `InputButton` discriminants never leak into Rust (they are a QAPI
//!   ABI, not ours to bake in).
//! - **Pointer moves** are absolute pixels within the current surface; the C
//!   shim scales them into the abs axis range.

/// Neutral pointer/wheel button code carried in `InputPointerButton`'s `a0`.
///
/// This is **our** stable wire contract, independent of QEMU's `InputButton`
/// QAPI enum — the C shim translates it. Do not renumber existing variants
/// (the discriminant is the wire value); append new ones at the end.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReimsVgpuButton {
    Left = 0,
    Middle = 1,
    Right = 2,
    /// Wheel notch up — emitted as a down+up pair per notch.
    WheelUp = 3,
    /// Wheel notch down — emitted as a down+up pair per notch.
    WheelDown = 4,
    /// Back / navigate-back side button.
    Side = 5,
    /// Forward / navigate-forward extra button.
    Extra = 6,
    /// Horizontal wheel left — down+up pair per notch.
    WheelLeft = 7,
    /// Horizontal wheel right — down+up pair per notch.
    WheelRight = 8,
}

impl ReimsVgpuButton {
    /// Reconstruct a button from its wire value (`a0`). Used by tests and any
    /// consumer that needs to round-trip the packed action; returns `None` for
    /// an unknown code rather than inventing a fallback button.
    #[cfg(test)]
    pub fn from_wire(v: u32) -> Option<Self> {
        Some(match v {
            0 => Self::Left,
            1 => Self::Middle,
            2 => Self::Right,
            3 => Self::WheelUp,
            4 => Self::WheelDown,
            5 => Self::Side,
            6 => Self::Extra,
            7 => Self::WheelLeft,
            8 => Self::WheelRight,
            _ => return None,
        })
    }

    /// True for the wheel codes, which the window thread emits as a momentary
    /// down+up pair (a wheel button has no held state).
    #[cfg(test)]
    pub fn is_wheel(self) -> bool {
        matches!(
            self,
            Self::WheelUp | Self::WheelDown | Self::WheelLeft | Self::WheelRight
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::host::{HostAction, HostActionKind};

    /// The ABI header's button table agrees with this enum, entry for entry.
    ///
    /// `shim.h` states the hazard in its own words — "a duplicated button table
    /// is a table that can drift, and a drift between the two shims is a bug the
    /// guest sees on exactly one pathway" — and this table is duplicated exactly
    /// that way: Rust packs the code into `HostAction.a0`, the header names the
    /// same nine numbers, and `reims_vgpu_shim_input_button` switches on them.
    /// Nothing compared the two until this test.
    ///
    /// A drift does not fail anywhere. It sends a middle-click where the guest
    /// user right-clicked, or a wheel notch in the wrong direction, and the only
    /// symptom is a desktop that behaves oddly under the mouse.
    #[test]
    fn the_abi_header_agrees_on_the_button_table() {
        use crate::qemu::abi::header_define as define;
        for (name, button) in [
            ("REIMS_VGPU_BUTTON_LEFT", ReimsVgpuButton::Left),
            ("REIMS_VGPU_BUTTON_MIDDLE", ReimsVgpuButton::Middle),
            ("REIMS_VGPU_BUTTON_RIGHT", ReimsVgpuButton::Right),
            ("REIMS_VGPU_BUTTON_WHEEL_UP", ReimsVgpuButton::WheelUp),
            ("REIMS_VGPU_BUTTON_WHEEL_DOWN", ReimsVgpuButton::WheelDown),
            ("REIMS_VGPU_BUTTON_SIDE", ReimsVgpuButton::Side),
            ("REIMS_VGPU_BUTTON_EXTRA", ReimsVgpuButton::Extra),
            ("REIMS_VGPU_BUTTON_WHEEL_LEFT", ReimsVgpuButton::WheelLeft),
            ("REIMS_VGPU_BUTTON_WHEEL_RIGHT", ReimsVgpuButton::WheelRight),
        ] {
            assert_eq!(
                define(name),
                button as u32,
                "{name} has drifted from ReimsVgpuButton::{button:?}; the shim \
                 would forward the wrong QEMU InputButton"
            );
        }
    }

    /// Every `ReimsVgpuButton` round-trips through its wire value, and `from_wire`
    /// rejects an out-of-range code (no invented fallback button).
    #[test]
    fn reims_vgpu_button_wire_roundtrip() {
        let all = [
            ReimsVgpuButton::Left,
            ReimsVgpuButton::Middle,
            ReimsVgpuButton::Right,
            ReimsVgpuButton::WheelUp,
            ReimsVgpuButton::WheelDown,
            ReimsVgpuButton::Side,
            ReimsVgpuButton::Extra,
            ReimsVgpuButton::WheelLeft,
            ReimsVgpuButton::WheelRight,
        ];
        for b in all {
            assert_eq!(ReimsVgpuButton::from_wire(b as u32), Some(b));
        }
        assert_eq!(ReimsVgpuButton::from_wire(9), None);
        assert_eq!(ReimsVgpuButton::from_wire(u32::MAX), None);
    }

    /// The wheel codes (and only those) report `is_wheel`.
    #[test]
    fn only_wheel_codes_are_wheel() {
        assert!(ReimsVgpuButton::WheelUp.is_wheel());
        assert!(ReimsVgpuButton::WheelDown.is_wheel());
        assert!(ReimsVgpuButton::WheelLeft.is_wheel());
        assert!(ReimsVgpuButton::WheelRight.is_wheel());
        assert!(!ReimsVgpuButton::Left.is_wheel());
        assert!(!ReimsVgpuButton::Middle.is_wheel());
        assert!(!ReimsVgpuButton::Right.is_wheel());
        assert!(!ReimsVgpuButton::Side.is_wheel());
        assert!(!ReimsVgpuButton::Extra.is_wheel());
    }

    /// `InputKey` packs the evdev code in `a0` and up/down in `a1`, exactly as
    /// the C shim unpacks it for `qemu_input_event_send_key_linux`. `KEY_A` in
    /// the Linux evdev space is 30.
    #[test]
    fn input_key_packs_evdev_and_state() {
        const KEY_A: u32 = 30;
        let down = HostAction::input_key(KEY_A, true);
        assert_eq!(down.kind, HostActionKind::InputKey);
        assert_eq!(down.a0 as u32, KEY_A);
        assert_eq!(down.a1, 1);
        let up = HostAction::input_key(KEY_A, false);
        assert_eq!(up.a1, 0);
    }

    /// `InputPointerMove` carries absolute x/y plus the surface dims the C shim
    /// scales against (min_in = 0, max_in = dim).
    #[test]
    fn input_pointer_move_packs_abs_coords_and_dims() {
        let m = HostAction::input_pointer_move(640, 360, 1920, 1080);
        assert_eq!(m.kind, HostActionKind::InputPointerMove);
        assert_eq!(m.a0, 640);
        assert_eq!(m.a1, 360);
        assert_eq!(m.a2, 1920);
        assert_eq!(m.a3, 1080);
    }

    /// `InputPointerButton` packs the neutral button code in `a0` and up/down in
    /// `a1`; the code round-trips back to the same `ReimsVgpuButton`.
    #[test]
    fn input_pointer_button_packs_neutral_code_and_state() {
        let a = HostAction::input_pointer_button(ReimsVgpuButton::Right, true);
        assert_eq!(a.kind, HostActionKind::InputPointerButton);
        assert_eq!(
            ReimsVgpuButton::from_wire(a.a0 as u32),
            Some(ReimsVgpuButton::Right)
        );
        assert_eq!(a.a1, 1);
        let up = HostAction::input_pointer_button(ReimsVgpuButton::WheelUp, false);
        assert_eq!(
            ReimsVgpuButton::from_wire(up.a0 as u32),
            Some(ReimsVgpuButton::WheelUp)
        );
        assert_eq!(up.a1, 0);
    }
}
