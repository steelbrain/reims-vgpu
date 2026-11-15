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

/// `MULTI_DISPLAY` (`FEATURES_HI` bit 16) is offered only by a device of two displays or more,
/// and only there can the guest take it: a device of one reads the feature words it read before
/// displays had rooms of their own
#[test]
fn multi_display_is_offered_only_by_a_device_of_two_displays_or_more() {
    let bit = (RGFX_FEATURE_MULTI_DISPLAY >> 32) as u32;
    for count in 1..=RGFX_DISPLAYS_MAX as usize {
        let rig = Rig::of_displays(count);
        let offered = rig.device.mmio_read(RGFX_REG_FEATURES_HI.into()) & bit != 0;
        assert_eq!(offered, count > 1, "{count} displays");
        take_multi_display(&rig);
        let taken = rig.device.mmio_read(RGFX_REG_DRIVER_FEATURES_HI.into()) & bit != 0;
        assert_eq!(taken, count > 1, "{count} displays");
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
