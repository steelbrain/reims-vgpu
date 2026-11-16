//! A pointer for games (docs/graphics-kext/design.md, "A pointer for games"): the offer of
//! `POINTER_MOTION`, the guest's cursor hidden on every display it sees, and the
//! `POINTER_MOTION`s a window's captured pointer sends, in their order, joined while the ring has
//! no room, a press or a release never lost.

use super::*;

/// A host that offers `POINTER_MOTION`, as one with a window does, and records what the device
/// told it of the control completion ring.
struct MotionHost(Arc<Mutex<Vec<RingDegradation>>>);

impl Host for MotionHost {
    fn raise_interrupt(&self, _vector: u32) {}
    fn features(&self) -> u64 {
        RGFX_FEATURE_POINTER_MOTION
    }
    fn ring_degraded(&self, kind: RingDegradation) {
        self.0.lock().push(kind);
    }
}

/// A rig of `count` displays whose host offers `POINTER_MOTION`.
fn motion_rig(count: usize) -> Rig {
    let degraded = Arc::new(Mutex::new(Vec::new()));
    let host = MotionHost(Arc::clone(&degraded));
    let mut rig = Rig::with_displays(Box::new(host), Lanes::default(), count);
    rig.degraded = degraded;
    rig
}

/// The guest's driver takes the bits `bits` of `FEATURES_HI` and starts the device.
fn take_and_start(rig: &Rig, bits: u64) {
    rig.device
        .mmio_write(RGFX_REG_DRIVER_FEATURES_HI.into(), (bits >> 32) as u32);
    rig.start();
}

/// A `MOVE_CURSOR` of `display` to (`x`, `y`), shown or hidden.
fn cursor_of(display: u32, x: i32, y: i32, visible: bool) -> Vec<u8> {
    RgfxMoveCursor {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_MOVE_CURSOR,
            ..Default::default()
        },
        display,
        x,
        y,
        visible: u32::from(visible),
    }
    .as_bytes()
    .to_vec()
}

/// Sends `command`, a `MOVE_CURSOR`, and waits for the control thread to have carried it out: it
/// carries out the answered `GET_DISPLAYS` after it.
fn moved(rig: &mut Rig, command: Vec<u8>) {
    rig.send(command);
    rig.send(get_displays());
    rig.event().expect("DISPLAYS");
}

/// The next event, as a `POINTER_MOTION`: its motion, wheel and buttons.
fn motion(rig: &mut Rig) -> (i32, i32, i32, u32) {
    let m = RgfxPointerMotion::read_from_bytes(&rig.event().expect("a POINTER_MOTION")).unwrap();
    assert_eq!(
        (m.header.r#type, m.header.size, m.header.seqno),
        (RGFX_EVT_POINTER_MOTION, 32, 0)
    );
    (m.dx, m.dy, m.wheel, m.buttons)
}

/// `POINTER_MOTION` (`FEATURES_HI` bit 19) is offered where the host offers it, on a device of one
/// display as of several, and only there can the guest take it
#[test]
fn pointer_motion_is_offered_where_the_host_offers_it() {
    let rig = motion_rig(1);
    let hi = rig.device.mmio_read(RGFX_REG_FEATURES_HI.into());
    assert_eq!(hi, (RGFX_FEATURE_POINTER_MOTION >> 32) as u32);
    let plain = Rig::new();
    assert_eq!(plain.device.mmio_read(RGFX_REG_FEATURES_HI.into()), 0);
    take_and_start(&plain, RGFX_FEATURE_POINTER_MOTION);
    assert_eq!(
        plain.device.mmio_read(RGFX_REG_DRIVER_FEATURES_HI.into()),
        0,
        "a host that does not offer it gives it to no guest"
    );
}

/// A guest that did not take `POINTER_MOTION` (a kext built before it, the option ROM,
/// Ambrosia's module) is sent no `POINTER_MOTION`, and its hidden cursor drives nothing: the
/// window's pointer stays the tablet's, as before
#[test]
fn a_guest_that_did_not_take_pointer_motion_gets_none() {
    let mut rig = motion_rig(1);
    rig.start();
    let link = rig.device.pointer_link();
    moved(&mut rig, cursor_of(0, 10, 10, false));
    assert_eq!(link.cursor_hidden_for(), None);
    assert!(!link.motion(5, 6, 0, 0), "the guest takes no motion");
    assert!(rig.quiet_for(200), "nothing sent");
}

/// The guest's hidden cursor drives the motion only once it hides it on every display it sees:
/// not before it has moved it, not while any display shows it, and from the first show on, a
/// hide starts the count again, so a hide and show in a row (WindowServer at its start, or the
/// cursor going from one display to the next) never holds for long
#[test]
fn the_cursor_is_hidden_once_every_display_it_sees_hides_it() {
    let mut rig = motion_rig(2);
    take_and_start(
        &rig,
        RGFX_FEATURE_POINTER_MOTION | RGFX_FEATURE_MULTI_DISPLAY,
    );
    let link = rig.device.pointer_link();
    assert_eq!(link.cursor_hidden_for(), None, "never moved");
    moved(&mut rig, cursor_of(0, 10, 10, true));
    assert_eq!(link.cursor_hidden_for(), None, "shown on display 0");
    // Onto display 1: hidden on 0, shown on 1.
    moved(&mut rig, cursor_of(0, 10, 10, false));
    let flicker = link.cursor_hidden_for();
    assert!(flicker.is_some(), "hidden between the two");
    moved(&mut rig, cursor_of(1, 5, 5, true));
    assert_eq!(link.cursor_hidden_for(), None, "shown on display 1");
    // A game hides it.
    moved(&mut rig, cursor_of(1, 5, 5, false));
    std::thread::sleep(Duration::from_millis(60));
    let hidden = link.cursor_hidden_for().expect("hidden on both");
    assert!(hidden >= 60_000_000, "counted from the hide: {hidden} ns");
    // Hidden again on display 1 keeps the count.
    moved(&mut rig, cursor_of(1, 6, 6, false));
    assert!(link.cursor_hidden_for().unwrap() >= hidden);
    moved(&mut rig, cursor_of(0, 1, 1, true));
    assert_eq!(link.cursor_hidden_for(), None, "shown on display 0");
    // Hidden again after the show: counted from this hide.
    moved(&mut rig, cursor_of(0, 1, 1, false));
    let again = link.cursor_hidden_for().expect("hidden on both again");
    assert!(again < hidden, "counted from the new hide: {again} ns");
}

/// A guest that sees display 0 alone (it did not take `MULTI_DISPLAY`) hides its cursor where
/// display 0 hides it
#[test]
fn a_guest_that_sees_one_display_hides_its_cursor_there() {
    let mut rig = motion_rig(2);
    take_and_start(&rig, RGFX_FEATURE_POINTER_MOTION);
    let link = rig.device.pointer_link();
    moved(&mut rig, cursor_of(0, 10, 10, false));
    assert!(link.cursor_hidden_for().is_some());
}

/// Each `POINTER_MOTION` the window sends reaches the guest that took it as it was sent: motion,
/// wheel and buttons
#[test]
fn motion_reaches_a_guest_that_took_it() {
    let mut rig = motion_rig(1);
    take_and_start(&rig, RGFX_FEATURE_POINTER_MOTION);
    let link = rig.device.pointer_link();
    assert!(link.motion(5, -6, 0, 0));
    assert_eq!(motion(&mut rig), (5, -6, 0, 0));
    assert!(link.motion(0, 0, 0, RGFX_POINTER_BUTTON_LEFT));
    assert_eq!(motion(&mut rig), (0, 0, 0, RGFX_POINTER_BUTTON_LEFT));
    assert!(link.motion(-3, 2, 1, RGFX_POINTER_BUTTON_LEFT));
    assert_eq!(motion(&mut rig), (-3, 2, 1, RGFX_POINTER_BUTTON_LEFT));
    assert!(link.motion(0, 0, -2, 0));
    assert_eq!(motion(&mut rig), (0, 0, -2, 0));
}

/// Motion that finds the control completion ring full is owed: motion of the same buttons joins
/// the last owed, a press or a release is owed after it, and the next refresh, once the guest has
/// taken room, sends them in their order, so no press or release is lost
#[test]
fn owed_motion_joins_and_a_press_never_does() {
    let mut rig = motion_rig(1);
    take_and_start(&rig, RGFX_FEATURE_POINTER_MOTION);
    // 24-byte answers: the 32 KiB ring holds 1365 of them.
    let count = 1400u64;
    for _ in 0..count {
        rig.send(vblank(0));
    }
    assert!(
        eventually(|| !rig.degraded.lock().is_empty()),
        "the ring never filled"
    );
    let link = rig.device.pointer_link();
    assert!(link.motion(1, 2, 0, 0));
    assert!(link.motion(3, 4, 1, 0));
    assert!(link.motion(0, 0, 0, RGFX_POINTER_BUTTON_LEFT));
    assert!(link.motion(5, 0, 0, RGFX_POINTER_BUTTON_LEFT));
    assert!(link.motion(0, 0, 0, 0));
    assert!(link.motion(0, 7, 0, 0));
    for seqno in 1..=count {
        assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));
    }
    rig.refresh();
    assert_eq!(motion(&mut rig), (4, 6, 1, 0));
    assert_eq!(motion(&mut rig), (5, 0, 0, RGFX_POINTER_BUTTON_LEFT));
    assert_eq!(motion(&mut rig), (0, 7, 0, 0));
    assert!(rig.quiet_for(100), "nothing more");
}

/// A reset forgets the hidden cursor and the owed motion, and sends the next driver none until it
/// takes `POINTER_MOTION` again
#[test]
fn a_reset_forgets_the_hidden_cursor_and_the_motion() {
    let mut rig = motion_rig(1);
    take_and_start(&rig, RGFX_FEATURE_POINTER_MOTION);
    let link = rig.device.pointer_link();
    moved(&mut rig, cursor_of(0, 1, 1, false));
    assert!(link.cursor_hidden_for().is_some());
    rig.device
        .mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_RESET);
    (rig.produced, rig.consumed, rig.seqno) = (0, 0, 0);
    assert_eq!(link.cursor_hidden_for(), None);
    rig.start();
    assert!(!link.motion(1, 1, 0, 0), "not taken since the reset");
    assert!(rig.quiet_for(100));
    rig.device
        .mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_RESET);
    take_and_start(&rig, RGFX_FEATURE_POINTER_MOTION);
    assert_eq!(link.cursor_hidden_for(), None, "never moved since");
    assert!(link.motion(2, 3, 0, 0));
    assert_eq!(motion(&mut rig), (2, 3, 0, 0));
}
