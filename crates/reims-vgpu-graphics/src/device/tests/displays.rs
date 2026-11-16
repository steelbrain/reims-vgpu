//! A device of several displays (docs/graphics-kext/design.md, "Multiple displays"): the offer
//! of `MULTI_DISPLAY`, what a guest that takes it and one that does not see, each display's room,
//! mode, events, cursor, panel, presents and colour, and the reset.

use super::*;

/// 64 MiB, each display's room on a device of two.
const ROOM_OF_TWO: u64 = 64 << 20;

/// A `SET_SCANOUT` of `display`.
fn scanout_on(display: u32, width: u32, height: u32, stride: u32, offset: u64) -> Vec<u8> {
    let mut c = RgfxSetScanout::read_from_bytes(&scanout(width, height, stride, offset)).unwrap();
    c.display = display;
    c.as_bytes().to_vec()
}

/// A `SET_VBLANK` of `display`.
fn vblank_on(display: u32, enabled: u32) -> Vec<u8> {
    let mut c = RgfxSetVblank::read_from_bytes(&vblank(enabled)).unwrap();
    c.display = display;
    c.as_bytes().to_vec()
}

/// A `SET_CURSOR` of `display`: a 1 x 1 image at `offset`, its hot spot at its pixel.
fn cursor_on(display: u32, offset: u64) -> Vec<u8> {
    RgfxSetCursor {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_SET_CURSOR,
            ..Default::default()
        },
        display,
        width: 1,
        height: 1,
        hot_x: 0,
        hot_y: 0,
        reserved_24: 0,
        offset,
    }
    .as_bytes()
    .to_vec()
}

/// A `SET_CURSOR` of `display`: a `width` x `height` image at `offset`, its hot spot at `hot`.
fn shape_on(display: u32, width: u32, height: u32, hot: (u32, u32), offset: u64) -> Vec<u8> {
    RgfxSetCursor {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_SET_CURSOR,
            ..Default::default()
        },
        display,
        width,
        height,
        hot_x: hot.0,
        hot_y: hot.1,
        reserved_24: 0,
        offset,
    }
    .as_bytes()
    .to_vec()
}

/// A `MOVE_CURSOR` of `display` to (`x`, `y`), shown.
fn move_on(display: u32, x: i32, y: i32) -> Vec<u8> {
    RgfxMoveCursor {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_MOVE_CURSOR,
            ..Default::default()
        },
        display,
        x,
        y,
        visible: 1,
    }
    .as_bytes()
    .to_vec()
}

/// The guest's driver takes `MULTI_DISPLAY`, before it starts the device.
fn take_multi_display(rig: &Rig) {
    rig.device.mmio_write(
        RGFX_REG_DRIVER_FEATURES_HI.into(),
        (RGFX_FEATURE_MULTI_DISPLAY >> 32) as u32,
    );
}

/// The guest's driver takes `MULTI_DISPLAY` and `POINTER_OVER`, before it starts the device.
fn take_pointer_over(rig: &Rig) {
    rig.device.mmio_write(
        RGFX_REG_DRIVER_FEATURES_HI.into(),
        ((RGFX_FEATURE_MULTI_DISPLAY | RGFX_FEATURE_POINTER_OVER) >> 32) as u32,
    );
}

/// The guest's driver takes `MULTI_DISPLAY`, `POINTER_OVER` and `DISPLAY_BOUNDS`, before it
/// starts the device.
fn take_bounds(rig: &Rig) {
    rig.device.mmio_write(
        RGFX_REG_DRIVER_FEATURES_HI.into(),
        ((RGFX_FEATURE_MULTI_DISPLAY | RGFX_FEATURE_POINTER_OVER | RGFX_FEATURE_DISPLAY_BOUNDS)
            >> 32) as u32,
    );
}

/// A `DISPLAY_BOUNDS` of `display`: at (`x`, `y`), `w` by `h` points.
fn bounds_of(display: u32, x: i32, y: i32, w: i32, h: i32) -> Vec<u8> {
    RgfxDisplayBounds {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_DISPLAY_BOUNDS,
            ..Default::default()
        },
        display,
        min_x: x,
        min_y: y,
        max_x: x + w,
        max_y: y + h,
        reserved_24: 0,
    }
    .as_bytes()
    .to_vec()
}

/// A `DISPLAYS` answer's words after its header.
fn words(bytes: &[u8]) -> Vec<u32> {
    let header = RgfxMsgHeader::read_from_prefix(bytes).unwrap().0;
    assert_eq!(header.r#type, RGFX_EVT_DISPLAYS);
    assert_eq!(header.size as usize, bytes.len());
    bytes[16..]
        .chunks(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// What display `i`'s screen was told since the last look, display 0's included.
fn told_on(rig: &Rig, i: usize) -> Vec<Told> {
    match i {
        0 => rig.told(),
        _ => std::mem::take(&mut *rig.more[i - 1].1.lock()),
    }
}

/// `MULTI_DISPLAY`, `POINTER_OVER` and `DISPLAY_BOUNDS` (`FEATURES_HI` bits 16 to 18) are
/// offered only by a device of two displays or more, and only there can the guest take them: a
/// device of one reads the feature words it read before displays had rooms of their own
#[test]
fn multi_display_is_offered_only_by_a_device_of_two_displays_or_more() {
    for bit in [
        (RGFX_FEATURE_MULTI_DISPLAY >> 32) as u32,
        (RGFX_FEATURE_POINTER_OVER >> 32) as u32,
        (RGFX_FEATURE_DISPLAY_BOUNDS >> 32) as u32,
    ] {
        for count in 1..=RGFX_DISPLAYS_MAX as usize {
            let rig = Rig::of_displays(count);
            let offered = rig.device.mmio_read(RGFX_REG_FEATURES_HI.into()) & bit != 0;
            assert_eq!(offered, count > 1, "bit {bit:#x}, {count} displays");
            take_bounds(&rig);
            let taken = rig.device.mmio_read(RGFX_REG_DRIVER_FEATURES_HI.into()) & bit != 0;
            assert_eq!(taken, count > 1, "bit {bit:#x}, {count} displays");
        }
    }
    let one = Rig::new();
    assert_eq!(one.device.mmio_read(RGFX_REG_FEATURES_HI.into()), 0);
    assert_eq!(one.device.mmio_read(RGFX_REG_FEATURES_LO.into()), 0);
}

/// A device makes 1 to `RGFX_DISPLAYS_MAX` displays, and refuses none or more
#[test]
fn a_device_has_one_to_four_displays() {
    let region = || Arc::new(Region::new(REGION_SIZE as usize).unwrap());
    let screens = |n: usize| -> Vec<Box<dyn Screen>> {
        (0..n)
            .map(|_| Box::new(crate::display::FixedScreen(MODE)) as Box<dyn Screen>)
            .collect()
    };
    struct Quiet;
    impl Host for Quiet {
        fn raise_interrupt(&self, _vector: u32) {}
    }
    let host = || -> Box<dyn Host> { Box::new(Quiet) };
    assert!(Device::with_screens(region(), host(), screens(0)).is_none());
    assert!(Device::with_screens(region(), host(), screens(4)).is_some());
    assert!(Device::with_screens(region(), host(), screens(5)).is_none());
}

/// A guest that did not take `MULTI_DISPLAY` on a device of three displays sees display 0 alone,
/// as from a device of one: `DISPLAYS` lists it alone, every command of another display is
/// refused `NO_DISPLAY` (a `MOVE_CURSOR` and a `DISPLAY_COLOR` dropped), and another display's
/// refreshes send it no event, its mode changed or not
#[test]
fn a_guest_that_did_not_take_multi_display_sees_display_0_alone() {
    let mut rig = Rig::of_displays(3);
    rig.start();
    rig.send(get_displays());
    let bytes = rig.event().unwrap();
    assert_eq!(bytes.len(), 48);
    assert_eq!(words(&bytes), [1, 0, 1, 0, 1, 1280, 800, NEVER_MHZ]);
    let room = scanout_room(3);
    let image = RGFX_ALLOCATIONS_OFFSET + 0x1000;
    rig.region().write(image, &[0, 0, 0, 255]);
    for display in [1, 2, 3, 7] {
        for bytes in [
            scanout_on(display, 16, 4, 128, RGFX_ALLOCATIONS_OFFSET + room),
            cursor_on(display, image),
            vblank_on(display, 1),
            display_power(display, 0),
            present_surface(display, 0),
        ] {
            let s = rig.send(bytes);
            assert_eq!(
                rig.complete(),
                (s, RGFX_STATUS_NO_DISPLAY),
                "display {display}"
            );
        }
        rig.send(move_on(display, 5, 6));
        rig.send(display_color(display, 0, 1, 0, 1));
    }
    // Answered once the control thread took everything before it.
    rig.send(get_displays());
    rig.event().unwrap();
    rig.lanes_done();
    *rig.more[0].0.lock() = Mode { width: 640, ..MODE };
    for display in 1..3 {
        rig.device.shared.refresh(display, 99);
    }
    assert!(rig.quiet_for(500), "no event of another display");
    for i in 1..3 {
        assert_eq!(told_on(&rig, i), [], "display {i}'s screen");
    }
    assert!(
        rig.by_display
            .lock()
            .iter()
            .all(|(d, seen)| seen.is_none() || *d == 0),
        "the host was handed nothing of another display"
    );
    // Display 0 works as on a device of one, in its own room.
    let s = rig.send(scanout(16, 4, 128, RGFX_ALLOCATIONS_OFFSET));
    assert_eq!(rig.complete(), (s, RGFX_STATUS_OK));
    let s = rig.send(vblank(1));
    assert_eq!(rig.complete(), (s, RGFX_STATUS_OK));
    rig.refresh();
    let v = RgfxVblank::read_from_bytes(&rig.event().unwrap()).unwrap();
    assert_eq!((v.header.r#type, v.display), (RGFX_EVT_VBLANK, 0));
}

/// A guest that took `MULTI_DISPLAY` is told every display, ids 0 up, each connected, with the
/// mode its own screen offers as its one, preferred mode
#[test]
fn a_guest_that_took_multi_display_is_told_every_display_s_own_mode() {
    let mut rig = Rig::of_displays(3);
    *rig.more[0].0.lock() = Mode {
        width: 1920,
        height: 1080,
        refresh_mhz: 60_000,
        scale_120ths: 120,
    };
    *rig.more[1].0.lock() = Mode {
        width: 2560,
        height: 1440,
        ..MODE
    };
    take_multi_display(&rig);
    rig.start();
    rig.send(get_displays());
    let bytes = rig.event().unwrap();
    assert_eq!(bytes.len(), 104);
    assert_eq!(
        words(&bytes),
        [
            3, 0, 1, 0, 1, 1280, 800, NEVER_MHZ, 1, 1, 0, 1, 1920, 1080, 60_000, 2, 1, 0, 1, 2560,
            1440, NEVER_MHZ
        ]
    );
}

/// Each display's scanout, cursor and panel reach its own screen alone, and its state is its own
#[test]
fn each_display_s_commands_reach_its_own_screen() {
    let mut rig = Rig::of_displays(2);
    take_multi_display(&rig);
    rig.start();
    let at = RGFX_ALLOCATIONS_OFFSET + ROOM_OF_TWO;
    let s = rig.send(scanout_on(1, 16, 4, 128, at));
    assert_eq!(rig.complete(), (s, RGFX_STATUS_OK));
    let shown = Scanout {
        width: 16,
        height: 4,
        stride: 128,
        offset: at,
    };
    assert_eq!(told_on(&rig, 1), [Told::Scanout(Some(shown))]);
    assert_eq!(told_on(&rig, 0), []);
    let image = RGFX_ALLOCATIONS_OFFSET + 0x1000;
    rig.region().write(image, &[0, 0, 0, 255]);
    let s = rig.send(cursor_on(1, image));
    assert_eq!(rig.complete(), (s, RGFX_STATUS_OK));
    rig.send(move_on(1, 7, 8));
    let s = rig.send(display_power(1, 0));
    assert_eq!(rig.complete(), (s, RGFX_STATUS_OK));
    let told = told_on(&rig, 1);
    assert!(matches!(told[0], Told::Cursor(Some(_))), "{told:?}");
    assert_eq!(told[1..], [Told::Moved(7, 8, true), Told::Power(false)]);
    assert_eq!(told_on(&rig, 0), []);
    let displays = &rig.device.shared.displays;
    assert_eq!(displays[0].state.lock().scanout, None);
    assert!(!displays[0].state.lock().asleep);
    assert_eq!(displays[1].state.lock().scanout, Some(shown));
    assert!(displays[1].state.lock().asleep);
    // A refresh shows each display's own scanout.
    rig.device.shared.refresh(1, 5);
    assert_eq!(told_on(&rig, 1), [Told::Present]);
    rig.refresh();
    assert_eq!(told_on(&rig, 0), []);
}

/// On a device of two, each display's scanout keeps to its own 64 MiB room, whether or not the
/// guest took `MULTI_DISPLAY`: one that reaches into the next display's room, or lies in another
/// display's, is refused, changing nothing
#[test]
fn each_display_s_scanout_keeps_to_its_own_room() {
    let span = 2560 * 479 + 2560;
    let first = RGFX_ALLOCATIONS_OFFSET;
    let second = first + ROOM_OF_TWO;
    let end = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
    for took in [false, true] {
        let mut rig = Rig::of_displays(2);
        if took {
            take_multi_display(&rig);
        }
        rig.start();
        let s = rig.send(scanout_on(0, 640, 480, 2560, second - span));
        assert_eq!(rig.complete(), (s, RGFX_STATUS_OK));
        for bytes in [
            scanout_on(0, 640, 480, 2560, second - span + 4),
            scanout_on(0, 640, 480, 2560, second),
            scanout_on(0, 3840, 4400, 15360, first),
        ] {
            let s = rig.send(bytes);
            assert_eq!(rig.complete(), (s, RGFX_STATUS_INVALID_ARGUMENT));
        }
        if !took {
            continue;
        }
        let s = rig.send(scanout_on(1, 640, 480, 2560, end - span));
        assert_eq!(rig.complete(), (s, RGFX_STATUS_OK));
        for bytes in [
            scanout_on(1, 640, 480, 2560, first),
            scanout_on(1, 640, 480, 2560, second - 4096),
            scanout_on(1, 640, 480, 2560, end - span + 4),
            scanout_on(1, 640, 480, 2560, end),
        ] {
            let s = rig.send(bytes);
            assert_eq!(rig.complete(), (s, RGFX_STATUS_INVALID_ARGUMENT));
        }
    }
}

/// On a device of several displays each offers the largest mode of its screen's ratio that its
/// room holds, to a guest that took `MULTI_DISPLAY` and to one that did not (so the firmware's
/// scanout in display 0's preferred mode fits display 0's room); one display offers its screen's
/// mode, 8K included
#[test]
fn a_display_offers_the_largest_mode_its_room_holds() {
    let eight_k = Mode {
        width: 7680,
        height: 4320,
        ..MODE
    };
    let mut one = Rig::new();
    *one.mode.lock() = eight_k;
    one.start();
    one.send(get_displays());
    assert_eq!(words(&one.event().unwrap())[5..8], [7680, 4320, NEVER_MHZ]);

    let mut four = Rig::of_displays(4);
    *four.mode.lock() = eight_k;
    *four.more[0].0.lock() = eight_k;
    four.start();
    four.send(get_displays());
    assert_eq!(words(&four.event().unwrap())[5..8], [3840, 2160, NEVER_MHZ]);

    let mut four = Rig::of_displays(4);
    *four.more[0].0.lock() = eight_k;
    take_multi_display(&four);
    four.start();
    four.send(get_displays());
    let words = words(&four.event().unwrap());
    assert_eq!(words[5..8], [1280, 800, NEVER_MHZ]);
    assert_eq!(words[12..15], [3840, 2160, NEVER_MHZ]);
}

/// Each display's `VBLANK`s and `DISPLAY_CHANGED` carry its id, its `VBLANK`s on while the guest
/// has them on for it and counted by its own refreshes, and the host is told each display's
/// refresh by its id
#[test]
fn each_display_s_vblanks_and_changes_carry_its_id() {
    let mut rig = Rig::of_displays(2);
    take_multi_display(&rig);
    rig.start();
    rig.send(get_displays());
    rig.event().unwrap();
    let s = rig.send(vblank_on(1, 1));
    assert_eq!(rig.complete(), (s, RGFX_STATUS_OK));
    rig.refresh();
    assert!(rig.quiet_for(500), "display 0's VBLANKs are off");
    rig.device.shared.refresh(1, 77);
    let v = RgfxVblank::read_from_bytes(&rig.event().unwrap()).unwrap();
    assert_eq!(
        (v.header.r#type, v.display, v.count, v.host_time_ns),
        (RGFX_EVT_VBLANK, 1, 1, 77)
    );
    *rig.more[0].0.lock() = Mode { width: 640, ..MODE };
    rig.device.shared.refresh(1, 78);
    let v = RgfxVblank::read_from_bytes(&rig.event().unwrap()).unwrap();
    assert_eq!((v.display, v.count), (1, 2));
    let c = RgfxDisplayChanged::read_from_bytes(&rig.event().unwrap()).unwrap();
    assert_eq!((c.header.r#type, c.display), (RGFX_EVT_DISPLAY_CHANGED, 1));
    rig.refresh();
    assert!(rig.quiet_for(500), "display 0's mode did not change");
    let refreshed: Vec<u32> = (rig.by_display.lock().iter())
        .filter(|(_, seen)| seen.is_none())
        .map(|&(d, _)| d)
        .collect();
    assert_eq!(refreshed, [0, 1, 1, 0]);
}

/// A present, a kept present and a colour part of display 1 reach the host naming display 1
#[test]
fn presents_and_colours_name_their_display_to_the_host() {
    let mut rig = Rig::of_displays(2);
    take_multi_display(&rig);
    rig.start();
    rig.send(display_color(1, 2, 1, 0, 1));
    let s = rig.send(present_surface(1, HELD_SURFACE));
    assert_eq!(rig.complete(), (s, RGFX_STATUS_OK));
    let s = rig.send(present_with_flags(0, RGFX_PRESENT_KEEP));
    assert_eq!(rig.complete(), (s, RGFX_STATUS_OK));
    let handed: Vec<(u32, Option<Seen>)> = rig.by_display.lock().clone();
    assert_eq!(
        handed,
        [
            (
                1,
                Some(Seen::Color {
                    stage: 2,
                    first: 0,
                    entries: 1,
                    red: 0
                })
            ),
            (1, Some(Seen::Present(HELD_SURFACE))),
            (0, Some(Seen::Keep)),
        ]
    );
}

/// A reset lets go of the take, so the next driver sees display 0 alone unless it takes the bit
/// again, and leaves each display as display 0 is left: its scanout kept, its cursor image gone,
/// its panel on, and no `VBLANK`
#[test]
fn a_reset_lets_go_of_the_take_and_leaves_each_display_as_display_0() {
    let mut rig = Rig::of_displays(2);
    take_multi_display(&rig);
    rig.start();
    let at = RGFX_ALLOCATIONS_OFFSET + ROOM_OF_TWO;
    let s = rig.send(scanout_on(1, 16, 4, 128, at));
    assert_eq!(rig.complete(), (s, RGFX_STATUS_OK));
    let s = rig.send(vblank_on(1, 1));
    assert_eq!(rig.complete(), (s, RGFX_STATUS_OK));
    let s = rig.send(display_power(1, 0));
    assert_eq!(rig.complete(), (s, RGFX_STATUS_OK));
    told_on(&rig, 1);
    rig.device
        .mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_RESET);
    assert_eq!(
        told_on(&rig, 1),
        [
            Told::Cursor(None),
            Told::Moved(0, 0, true),
            Told::Power(true)
        ]
    );
    assert_eq!(
        told_on(&rig, 0),
        [Told::Cursor(None), Told::Moved(0, 0, true)]
    );
    assert_eq!(rig.device.mmio_read(RGFX_REG_DRIVER_FEATURES_HI.into()), 0);
    (rig.produced, rig.consumed, rig.seqno) = (0, 0, 0);
    rig.start();
    rig.send(get_displays());
    assert_eq!(words(&rig.event().unwrap())[0], 1);
    // The scanout stays, shown at each refresh, and no event of display 1 comes.
    rig.device.shared.refresh(1, 9);
    assert_eq!(told_on(&rig, 1), [Told::Present]);
    assert!(rig.quiet_for(500));
}

/// The next event, as a `POINTER_OVER`: its display, point and buttons.
fn pointer_over(rig: &mut Rig) -> (u32, i32, i32, u32) {
    let p = RgfxPointerOver::read_from_bytes(&rig.event().expect("a POINTER_OVER")).unwrap();
    assert_eq!(
        (p.header.r#type, p.header.size, p.header.seqno),
        (RGFX_EVT_POINTER_OVER, 32, 0)
    );
    (p.display, p.x, p.y, p.buttons)
}

/// Sends `command`, a `MOVE_CURSOR`, and waits for the control thread to have carried it out: it
/// carries out the answered `GET_DISPLAYS` after it.
fn moved(rig: &mut Rig, command: Vec<u8>) {
    rig.send(command);
    rig.send(get_displays());
    rig.event().expect("DISPLAYS");
}

/// The host's pointer over a display the guest's cursor is not on sends the guest that took
/// `POINTER_OVER` a `POINTER_OVER` of that display at the pointer's point, with its buttons, once
/// while the cursor stays where it was, and none while the cursor is on the pointer's display;
/// once the cursor has shown on the pointer's display, the pointer's next display is told anew
#[test]
fn the_pointer_over_another_display_tells_the_guest_once() {
    let mut rig = Rig::of_displays(2);
    take_pointer_over(&rig);
    rig.start();
    rig.send(get_displays());
    rig.event().unwrap();
    let link = rig.device.pointer_link();
    // The cursor starts on display 0.
    link.pointer_at(0, 10, 20, 0);
    assert!(rig.quiet_for(200), "the cursor is on the pointer's display");
    link.pointer_at(1, 100, 200, 0);
    assert_eq!(pointer_over(&mut rig), (1, 100, 200, 0));
    link.pointer_at(1, 101, 201, 0);
    assert!(
        rig.quiet_for(200),
        "told once while the cursor stays on display 0"
    );
    // The guest moved its cursor onto display 1: the pointer there tells nothing.
    moved(&mut rig, move_on(1, 101, 201));
    link.pointer_at(1, 102, 202, 0);
    assert!(rig.quiet_for(200), "the cursor is on display 1");
    // Back over display 0's window, a button held: the guest is told it, to drag there.
    link.pointer_at(0, 5, 6, RGFX_POINTER_BUTTON_LEFT);
    assert_eq!(pointer_over(&mut rig), (0, 5, 6, RGFX_POINTER_BUTTON_LEFT));
    moved(&mut rig, move_on(0, 5, 6));
    link.pointer_at(1, 7, 8, 0);
    assert_eq!(pointer_over(&mut rig), (1, 7, 8, 0));
    // A hidden cursor's moves are not where it shows.
    let mut hidden = RgfxMoveCursor::read_from_bytes(&move_on(1, 7, 8)).unwrap();
    hidden.visible = 0;
    moved(&mut rig, hidden.as_bytes().to_vec());
    link.pointer_at(1, 9, 9, 0);
    assert!(
        rig.quiet_for(200),
        "told once while the cursor shows on display 0"
    );
}

/// A cursor near the edge between two displays shows on both on an arm64 guest, each display
/// given the part of its image on it, the one not holding the hot spot a part whose hot spot is
/// held to the image's column at the display's edge (measured on arm64 macOS 27.0.1, potluck job
/// m3-max-35746): that display's `MOVE_CURSOR` leaves the cursor on the display showing the hot
/// spot, so the pointer over the other display is told the guest; where both show the image's
/// edge (a hot spot exactly at it), a crossing's own display takes the cursor; and the part over
/// the edge takes it once the display it was on hides it
#[test]
fn the_image_over_the_edge_leaves_the_cursor_on_the_display_of_its_hot_spot() {
    let mut rig = Rig::of_displays(2);
    take_pointer_over(&rig);
    rig.start();
    rig.send(get_displays());
    rig.event().unwrap();
    for display in 0..2 {
        let at = RGFX_ALLOCATIONS_OFFSET + u64::from(display) * ROOM_OF_TWO;
        let s = rig.send(scanout_on(display, 1920, 1080, 1920 * 4, at));
        assert_eq!(rig.complete(), (s, RGFX_STATUS_OK));
    }
    let image = RGFX_ALLOCATIONS_OFFSET + 0x1000;
    rig.region().write(image, &[0; 28 * 40 * 4]);
    let shape = |rig: &mut Rig, display: u32, width: u32, hot: (u32, u32)| {
        let s = rig.send(shape_on(display, width, 40, hot, image));
        assert_eq!(rig.complete(), (s, RGFX_STATUS_OK));
    };
    let link = rig.device.pointer_link();
    // The arrow at 1915 of display 0: its left 10 columns there, holding the hot spot, its right
    // 18 on display 1, the hot spot held to their first column, at pixel 0.
    shape(&mut rig, 0, 10, (5, 5));
    moved(&mut rig, move_on(0, 1915, 650));
    shape(&mut rig, 1, 18, (0, 5));
    moved(&mut rig, move_on(1, 0, 650));
    assert!(
        link.pointer_at(0, 1915, 650, 0),
        "the cursor is on display 0"
    );
    assert!(rig.quiet_for(100), "nothing told");
    assert!(
        !link.pointer_at(1, 0, 650, 0),
        "the crossing onto display 1"
    );
    assert_eq!(pointer_over(&mut rig), (1, 0, 650, 0));
    // The crossing's post at display 1's pixel 0: display 0 shows the arrow's first 5 columns at
    // its 1919, the hot spot held to their last; display 1 the rest, the hot spot on their first.
    shape(&mut rig, 0, 5, (4, 5));
    moved(&mut rig, move_on(0, 1919, 650));
    shape(&mut rig, 1, 23, (0, 5));
    moved(&mut rig, move_on(1, 0, 650));
    assert!(
        link.pointer_at(1, 1, 650, 0),
        "display 1, told, took the cursor"
    );
    moved(&mut rig, move_on(0, 1919, 651));
    assert!(
        link.pointer_at(1, 2, 650, 0),
        "display 0's part over the edge left it there"
    );
    // The arrow whole on display 1, then its part over the edge on display 0 once display 1
    // hides it: display 0 takes the cursor.
    shape(&mut rig, 1, 28, (5, 5));
    moved(&mut rig, move_on(1, 700, 500));
    let mut hidden = RgfxMoveCursor::read_from_bytes(&move_on(1, 700, 500)).unwrap();
    hidden.visible = 0;
    moved(&mut rig, hidden.as_bytes().to_vec());
    shape(&mut rig, 0, 4, (3, 5));
    moved(&mut rig, move_on(0, 1919, 500));
    assert!(
        link.pointer_at(0, 1919, 500, 0),
        "the cursor is on display 0"
    );
    assert!(rig.quiet_for(100), "nothing told");
}

/// The guest's cursor moved off the pointer's display by the guest itself, while the pointer
/// stays over that display's window, is sent back: the pointer's display is told again
#[test]
fn the_cursor_moved_away_under_the_pointer_is_told_the_pointer_s_display() {
    let mut rig = Rig::of_displays(3);
    take_pointer_over(&rig);
    rig.start();
    rig.send(get_displays());
    rig.event().unwrap();
    let link = rig.device.pointer_link();
    link.pointer_at(1, 1, 2, 0);
    assert_eq!(pointer_over(&mut rig), (1, 1, 2, 0));
    moved(&mut rig, move_on(1, 1, 2));
    moved(&mut rig, move_on(2, 30, 40));
    link.pointer_at(1, 3, 4, 0);
    assert_eq!(pointer_over(&mut rig), (1, 3, 4, 0));
}

/// No `POINTER_OVER` goes to a guest that did not take `MULTI_DISPLAY`, which sees display 0
/// alone, to one that took `MULTI_DISPLAY` alone (a driver that knows no `POINTER_OVER`), to one
/// whose device has one display, before the driver starts the device, nor of a display past the
/// count
#[test]
fn the_pointer_tells_only_a_guest_that_took_pointer_over() {
    let mut one = Rig::new();
    one.start();
    one.device.pointer_link().pointer_at(0, 1, 1, 0);
    one.device.pointer_link().pointer_at(1, 1, 1, 0);
    assert!(one.quiet_for(200), "a device of one display");

    let mut untaken = Rig::of_displays(2);
    untaken.start();
    untaken.device.pointer_link().pointer_at(1, 1, 1, 0);
    assert!(untaken.quiet_for(200), "a guest that sees display 0 alone");

    let mut displays_only = Rig::of_displays(2);
    take_multi_display(&displays_only);
    displays_only.start();
    displays_only.send(get_displays());
    assert_eq!(
        words(&displays_only.event().unwrap())[0],
        2,
        "it sees both displays"
    );
    displays_only.device.pointer_link().pointer_at(1, 1, 1, 0);
    displays_only.refresh();
    assert!(
        displays_only.quiet_for(200),
        "a guest that took MULTI_DISPLAY alone"
    );

    let mut early = Rig::of_displays(2);
    take_pointer_over(&early);
    early.device.pointer_link().pointer_at(1, 1, 1, 0);
    early.start();
    early.device.pointer_link().pointer_at(2, 1, 1, 0);
    assert!(early.quiet_for(200), "before the start, and past the count");
    early.device.pointer_link().pointer_at(1, 1, 1, 0);
    assert_eq!(pointer_over(&mut early), (1, 1, 1, 0));
}

/// A reset forgets what the guest was told: the next driver, its cursor on display 0, is told
/// of the pointer over display 1 again
#[test]
fn a_reset_forgets_what_the_pointer_told() {
    let mut rig = Rig::of_displays(2);
    take_pointer_over(&rig);
    rig.start();
    let link = rig.device.pointer_link();
    link.pointer_at(1, 1, 1, 0);
    assert_eq!(pointer_over(&mut rig), (1, 1, 1, 0));
    rig.device
        .mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_RESET);
    (rig.produced, rig.consumed, rig.seqno) = (0, 0, 0);
    take_pointer_over(&rig);
    rig.start();
    link.pointer_at(1, 2, 2, 0);
    assert_eq!(pointer_over(&mut rig), (1, 2, 2, 0));
}

/// A `POINTER_OVER` that finds the control completion ring full is owed, and goes at the next
/// refresh once the guest has taken room
#[test]
fn a_pointer_over_the_full_ring_has_no_room_for_goes_at_the_next_refresh() {
    let mut rig = Rig::of_displays(2);
    take_pointer_over(&rig);
    rig.start();
    // 24-byte answers: the 32 KiB ring holds 1365 of them.
    let count = 1400u64;
    for _ in 0..count {
        rig.send(vblank_on(0, 0));
    }
    assert!(
        eventually(|| !rig.degraded.lock().is_empty()),
        "the ring never filled"
    );
    rig.device.pointer_link().pointer_at(1, 3, 4, 0);
    for seqno in 1..=count {
        assert_eq!(rig.complete(), (seqno, RGFX_STATUS_OK));
    }
    assert!(rig.quiet_for(100), "owed until a refresh");
    rig.refresh();
    assert_eq!(pointer_over(&mut rig), (1, 3, 4, 0));
}

/// A crossing holds the pointer's positions back from the guest's pointing device until the
/// guest's cursor shows on the pointer's display, so that the guest never takes one as a place
/// on the display its cursor is still on, where its tablet's place would land at the same spot
/// of that display: the position after the cursor came is handed on; so is one whose buttons
/// changed, and any once 100 ms have gone with no cursor there
#[test]
fn a_crossing_holds_the_pointer_until_the_guest_s_cursor_is_there() {
    let mut rig = Rig::of_displays(2);
    take_pointer_over(&rig);
    rig.start();
    rig.send(get_displays());
    rig.event().unwrap();
    let link = rig.device.pointer_link();
    assert!(link.pointer_at(0, 10, 20, 0), "the cursor is on display 0");
    assert!(
        !link.pointer_at(1, 100, 200, 0),
        "the crossing's own position"
    );
    assert_eq!(pointer_over(&mut rig), (1, 100, 200, 0));
    assert!(
        !link.pointer_at(1, 101, 201, 0),
        "held while the cursor stays"
    );
    assert!(
        link.pointer_at(1, 101, 201, RGFX_POINTER_BUTTON_LEFT),
        "a button's change goes on"
    );
    assert!(!link.pointer_at(1, 102, 202, RGFX_POINTER_BUTTON_LEFT));
    moved(&mut rig, move_on(1, 102, 202));
    assert!(link.pointer_at(1, 103, 203, RGFX_POINTER_BUTTON_LEFT));
    // Back to display 0, the guest never showing its cursor there: held 100 ms at most.
    assert!(!link.pointer_at(0, 5, 5, RGFX_POINTER_BUTTON_LEFT));
    assert_eq!(pointer_over(&mut rig), (0, 5, 5, RGFX_POINTER_BUTTON_LEFT));
    std::thread::sleep(Duration::from_millis(110));
    assert!(link.pointer_at(0, 6, 6, RGFX_POINTER_BUTTON_LEFT));
    assert!(rig.quiet_for(100), "told once");
    // A device of one display, and a guest that did not take POINTER_OVER, hold nothing.
    let one = Rig::new();
    one.start();
    assert!(one.device.pointer_link().pointer_at(1, 1, 1, 0));
    let untaken = Rig::of_displays(2);
    take_multi_display(&untaken);
    untaken.start();
    assert!(untaken.device.pointer_link().pointer_at(1, 1, 1, 0));
}

/// A guest of two displays, each 1920 x 1080 scanned out at `scale` pixels a point, that took
/// `DISPLAY_BOUNDS`, with display 1 at (`x1`, `y1`) of its desktop.
fn placed(scale: u32, x1: i32, y1: i32) -> Rig {
    let mut rig = Rig::of_displays(2);
    take_bounds(&rig);
    rig.start();
    let (w, h) = (1920 * scale, 1080 * scale);
    for display in 0..2 {
        let at = RGFX_ALLOCATIONS_OFFSET + u64::from(display) * ROOM_OF_TWO;
        let s = rig.send(scanout_on(display, w, h, w * 4, at));
        assert_eq!(rig.complete(), (s, RGFX_STATUS_OK));
    }
    rig.send(bounds_of(0, 0, 0, 1920, 1080));
    rig.send(bounds_of(1, x1, y1, 1920, 1080));
    rig.send(get_displays());
    rig.event().expect("DISPLAYS");
    rig
}

/// The captured pointer held at an edge of display 0's window where the guest's desktop puts
/// display 1 goes on onto display 1, as the guest arranged them, its moves scaled as over display
/// 0's window, and comes back across the same edge; edges no display meets hold it
#[test]
fn a_pointer_past_a_shared_edge_goes_onto_the_display_there() {
    let rig = placed(1, 1920, 0);
    let link = rig.device.pointer_link();
    let mode = (1920, 1080);
    assert!(link.adjoins(0, (1919, 500), mode, (1, 0)));
    assert!(
        !link.adjoins(0, (1919, 500), mode, (0, 1)),
        "the bottom edge"
    );
    assert!(!link.adjoins(0, (0, 500), mode, (-1, 0)), "the left edge");
    assert!(!link.adjoins(0, (1919, 0), mode, (0, -1)), "the top edge");
    assert!(link.adjoins(1, (0, 10), mode, (-1, 0)), "and back");
    assert!(link.leave(0, (1919, 500), mode));
    let spot = |display, x, y| Spot {
        display,
        x,
        y,
        width: 1920,
        height: 1080,
    };
    assert_eq!(link.away_moved(0.4, 0.0), Some(spot(0, 1919, 500)));
    assert_eq!(link.away_moved(1.0, 0.0), Some(spot(1, 0, 500)));
    assert_eq!(link.away_moved(300.0, -20.0), Some(spot(1, 300, 480)));
    // Up past display 1's top, which no display meets: held there.
    assert_eq!(link.away_moved(0.0, -1000.0), Some(spot(1, 300, 0)));
    assert_eq!(link.away_moved(-301.0, 10.0), Some(spot(0, 1919, 10)));
    link.come_back();
    assert_eq!(link.away_moved(1.0, 0.0), None, "the window's again");
}

/// The guest's arrangement decides: display 1 left of display 0 meets display 0's left edge,
/// and a HiDPI guest's points carry the pointer as far as its pixels show
#[test]
fn the_guest_s_arrangement_decides_which_edge_meets_which_display() {
    let rig = placed(2, -1920, 300);
    let link = rig.device.pointer_link();
    let mode = (3840, 2160);
    assert!(
        !link.adjoins(0, (3839, 1000), mode, (1, 0)),
        "the right edge"
    );
    assert!(!link.adjoins(0, (0, 100), mode, (-1, 0)), "above display 1");
    assert!(link.adjoins(0, (0, 1000), mode, (-1, 0)));
    assert!(link.leave(0, (0, 1000), mode));
    // Two pixels of display 0 are a point of the desktop, and four pixels of display 1.
    let spot = link.away_moved(-4.0, 0.0).unwrap();
    assert_eq!((spot.display, spot.x, spot.y), (1, 3836, 400));
    assert_eq!((spot.width, spot.height), (3840, 2160));
}

/// Nothing goes past an edge for a guest that did not take `DISPLAY_BOUNDS` or gave no
/// rectangles, nor for a display the guest does not see, and a reset forgets the rectangles and
/// the pointer gone onto the desktop
#[test]
fn only_a_guest_that_placed_its_displays_carries_the_pointer_past_an_edge() {
    let mode = (1920, 1080);
    let mut untaken = Rig::of_displays(2);
    take_pointer_over(&untaken);
    untaken.start();
    let at = RGFX_ALLOCATIONS_OFFSET + ROOM_OF_TWO;
    for (display, at) in [(0, RGFX_ALLOCATIONS_OFFSET), (1, at)] {
        let s = untaken.send(scanout_on(display, 1920, 1080, 7680, at));
        assert_eq!(untaken.complete(), (s, RGFX_STATUS_OK));
    }
    untaken.send(bounds_of(0, 0, 0, 1920, 1080));
    untaken.send(bounds_of(1, 1920, 0, 1920, 1080));
    untaken.send(get_displays());
    untaken.event().unwrap();
    let link = untaken.device.pointer_link();
    assert!(!link.adjoins(0, (1919, 500), mode, (1, 0)));
    assert!(!link.leave(0, (1919, 500), mode));

    let mut unplaced = Rig::of_displays(2);
    take_bounds(&unplaced);
    unplaced.start();
    unplaced.send(bounds_of(0, 0, 0, 1920, 1080));
    unplaced.send(bounds_of(1, 1920, 0, 1920, 1080));
    unplaced.send(bounds_of(2, 0, 1080, 1920, 1080));
    unplaced.send(get_displays());
    unplaced.event().unwrap();
    assert!(
        !unplaced
            .device
            .pointer_link()
            .adjoins(0, (1919, 500), mode, (1, 0)),
        "no scanout"
    );

    let mut rig = placed(1, 1920, 0);
    let link = rig.device.pointer_link();
    assert!(!link.adjoins(2, (0, 0), mode, (1, 0)), "past the count");
    // Display 1's rectangle of no area forgets it.
    rig.send(bounds_of(1, 1920, 0, 0, 1080));
    rig.send(get_displays());
    rig.event().unwrap();
    assert!(!link.adjoins(0, (1919, 500), mode, (1, 0)));
    rig.send(bounds_of(1, 1920, 0, 1920, 1080));
    rig.send(get_displays());
    rig.event().unwrap();
    assert!(link.leave(0, (1919, 500), mode));
    rig.device
        .mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_RESET);
    (rig.produced, rig.consumed, rig.seqno) = (0, 0, 0);
    take_bounds(&rig);
    rig.start();
    assert_eq!(link.away_moved(1.0, 0.0), None, "a reset ends it");
    assert!(
        !link.adjoins(0, (1919, 500), mode, (1, 0)),
        "a reset forgets the rectangles"
    );
}
