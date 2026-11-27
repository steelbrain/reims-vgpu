//! A pointer for games (docs/graphics-kext/design.md, "A pointer for games"): the offer of
//! `POINTER_MOTION`, the guest's cursor hidden on every display it sees (but on its way to the
//! next), and the `POINTER_MOTION`s a window's captured pointer sends, in their order, joined
//! while the ring has no room, a press or a release never lost; and the `SCROLL`s a window sends
//! the guest's devices that scroll (docs/graphics-kext/design.md, "Scrolling").

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

/// A crossing's hide is the cursor going from one display to the next: while a `POINTER_OVER`
/// has yet to show the cursor on the pointer's display, the cursor hidden on every display counts
/// from a second after it at the earliest, so a slow guest's crossing (an arm64 guest under TCG
/// shows it up to half a second later) never drives the guest's mouse mid-drag; once the cursor
/// shows there, a hide counts from itself again; and a crossing that never shows it holds the
/// count off for that second alone
#[test]
fn a_crossing_s_hide_is_not_a_game_s() {
    let mut rig = motion_rig(2);
    take_and_start(
        &rig,
        RGFX_FEATURE_POINTER_MOTION | RGFX_FEATURE_MULTI_DISPLAY | RGFX_FEATURE_POINTER_OVER,
    );
    let link = rig.device.pointer_link();
    moved(&mut rig, cursor_of(0, 10, 10, true));
    let over = |rig: &mut Rig| {
        let event = rig.event().expect("a POINTER_OVER");
        let p = RgfxPointerOver::read_from_bytes(&event).unwrap();
        assert_eq!(p.header.r#type, RGFX_EVT_POINTER_OVER);
    };
    // The pointer over display 1: the guest hides its cursor on display 0 on its way there.
    assert!(!link.pointer_at(1, 5, 5, 0), "held for the crossing");
    over(&mut rig);
    moved(&mut rig, cursor_of(0, 10, 10, false));
    std::thread::sleep(Duration::from_millis(60));
    assert_eq!(
        link.cursor_hidden_for(),
        Some(0),
        "hidden, but on its way to display 1"
    );
    moved(&mut rig, cursor_of(1, 5, 5, true));
    assert_eq!(link.cursor_hidden_for(), None, "shown on display 1");
    // A game hides it there, no crossing under way: counted from the hide.
    moved(&mut rig, cursor_of(1, 5, 5, false));
    std::thread::sleep(Duration::from_millis(60));
    assert!(
        link.cursor_hidden_for().unwrap() >= 60_000_000,
        "a game's hide counts from itself"
    );
    // A crossing onto display 0 that never shows the cursor there: the count waits a second.
    assert!(!link.pointer_at(0, 7, 7, 0), "held for the crossing");
    over(&mut rig);
    assert_eq!(link.cursor_hidden_for(), Some(0), "a crossing under way");
    std::thread::sleep(Duration::from_millis(1060));
    assert!(
        link.cursor_hidden_for().unwrap() >= 50_000_000,
        "counted from a second after the POINTER_OVER"
    );
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

/// A `SCROLL_DEVICES` naming `devices`.
fn scroll_devices(devices: u32) -> Vec<u8> {
    RgfxScrollDevices {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_SCROLL_DEVICES,
            ..Default::default()
        },
        devices,
        reserved_14: 0,
    }
    .as_bytes()
    .to_vec()
}

/// The next event, as a `SCROLL`: its notches up and right, then its distance up and right.
fn scrolled(rig: &mut Rig) -> (i32, i32, i32, i32) {
    let s = RgfxScroll::read_from_bytes(&rig.event().expect("a SCROLL")).unwrap();
    assert_eq!(
        (s.header.r#type, s.header.size, s.header.seqno),
        (RGFX_EVT_SCROLL, 32, 0)
    );
    (s.notches_up, s.notches_right, s.fine_up, s.fine_right)
}

/// Scrolling goes to the guest's own devices from its `SCROLL_DEVICES` on, and until one that
/// names none or a reset: a wheel's notches as notches, a precise device's distance where the
/// guest has a precise device, else the notches in it, one per 10 points counted on. Before it, or
/// after, nothing is sent, and the host's own pointing device scrolls
#[test]
fn scrolling_goes_to_the_devices_the_guest_names() {
    let mut rig = motion_rig(1);
    take_and_start(&rig, RGFX_FEATURE_POINTER_MOTION);
    let link = rig.device.pointer_link();
    assert!(!link.scroll((1, 0), false), "no devices named");
    moved(&mut rig, scroll_devices(RGFX_SCROLL_DEVICE_WHEEL));
    assert!(link.scroll((2, -3), false));
    assert_eq!(scrolled(&mut rig), (2, -3, 0, 0));
    moved(
        &mut rig,
        scroll_devices(RGFX_SCROLL_DEVICE_WHEEL | RGFX_SCROLL_DEVICE_PRECISE),
    );
    assert!(link.scroll((100, -200), true));
    assert_eq!(scrolled(&mut rig), (0, 0, 100, -200));
    assert!(link.scroll((0, 1), false), "a wheel's notch stays a notch");
    assert_eq!(scrolled(&mut rig), (0, 1, 0, 0));
    moved(&mut rig, scroll_devices(0));
    assert!(!link.scroll((1, 0), false), "none named");
    moved(&mut rig, scroll_devices(RGFX_SCROLL_DEVICE_WHEEL));
    rig.device
        .mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_RESET);
    (rig.produced, rig.consumed, rig.seqno) = (0, 0, 0);
    take_and_start(&rig, RGFX_FEATURE_POINTER_MOTION);
    assert!(!link.scroll((1, 0), false), "none named since the reset");
    assert!(rig.quiet_for(100));
}

/// A guest with a mouse and no precise device gets a precise device's distance as notches, one
/// per 10 points (a line of SDL's) each way, the rest counted on and counted afresh where the
/// scrolling turns back; a distance of i32's range overruns nothing
#[test]
fn a_mouse_alone_gets_a_notch_per_10_points() {
    let mut rig = motion_rig(1);
    take_and_start(&rig, RGFX_FEATURE_POINTER_MOTION);
    moved(&mut rig, scroll_devices(RGFX_SCROLL_DEVICE_WHEEL));
    let link = rig.device.pointer_link();
    assert_eq!(FINE_PER_NOTCH, 2560);
    assert!(
        link.scroll((1280, -640), true),
        "taken, though no notch yet"
    );
    assert!(rig.quiet_for(100), "no notch yet");
    assert!(link.scroll((1280, -1920), true));
    assert_eq!(scrolled(&mut rig), (1, -1, 0, 0));
    assert!(link.scroll((6000, 0), true));
    assert_eq!(scrolled(&mut rig), (2, 0, 0, 0));
    assert!(
        link.scroll((-1000, 0), true),
        "back the other way: counted afresh"
    );
    assert!(link.scroll((-1600, 0), true));
    assert_eq!(scrolled(&mut rig), (-1, 0, 0, 0));
    assert!(link.scroll((i32::MAX, i32::MIN), true));
    assert_eq!(scrolled(&mut rig), (i32::MAX / 2560, i32::MIN / 2560, 0, 0));
}

/// Scrolling that finds the control completion ring full is owed, joined into one `SCROLL` to
/// the field, saturating, and the next refresh sends it after the owed motion
#[test]
fn owed_scrolling_joins_after_the_motion() {
    let mut rig = motion_rig(1);
    take_and_start(&rig, RGFX_FEATURE_POINTER_MOTION);
    moved(
        &mut rig,
        scroll_devices(RGFX_SCROLL_DEVICE_WHEEL | RGFX_SCROLL_DEVICE_PRECISE),
    );
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
    assert!(link.scroll((1, 2), false));
    assert!(link.scroll((300, -5), true));
    assert!(link.scroll((3, 0), false));
    assert!(link.scroll((i32::MAX, 0), false));
    for seqno in 3..=count + 2 {
        assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));
    }
    rig.refresh();
    assert_eq!(motion(&mut rig), (1, 2, 0, 0));
    assert_eq!(scrolled(&mut rig), (i32::MAX, 2, 300, -5));
    assert!(rig.quiet_for(100), "nothing more");
}
