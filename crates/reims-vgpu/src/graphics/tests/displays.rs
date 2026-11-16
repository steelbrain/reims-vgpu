//! A device of several displays ([`GraphicsConfig::display_count`]): the counts and windows it
//! takes, each display's read-back, and each display's cursor at the host's callbacks; and a
//! window's captured pointer driving the guest's own mouse (`POINTER_MOTION`).

use reims_vgpu_window::Landed;

use super::*;

/// A headless device of `count` displays on the testkit's platform, with `callbacks`.
fn headless_of(count: u32, callbacks: GraphicsCallbacks) -> GraphicsDevice<FakePlatform> {
    let config = GraphicsConfig {
        callbacks,
        display_count: count,
        ..GraphicsConfig::default()
    };
    GraphicsDevice::with_platform(Arc::new(FakePlatform::new()), config).unwrap()
}

/// Whether `device` offers `MULTI_DISPLAY`, in `FEATURES_HI`.
fn offers_multi_display(device: &GraphicsDevice<FakePlatform>) -> bool {
    let hi = u64::from(device.mmio_read(RGFX_REG_FEATURES_HI.into())) << 32;
    hi & RGFX_FEATURE_MULTI_DISPLAY != 0
}

/// A count of 0 makes one display, as 1 does, and a device of one display offers no
/// `MULTI_DISPLAY`; 2 to 4 make that many, each offering it; 5 is refused, and so are display
/// windows not given exactly where display 0 has one: one for each display from 1
#[test]
fn the_display_count_makes_one_to_four_displays() {
    for (count, made) in [(0, 1), (1, 1), (2, 2), (3, 3), (4, 4)] {
        let device = headless_of(count, GraphicsCallbacks::default());
        assert_eq!(device.display_count(), made, "count {count}");
        assert_eq!(offers_multi_display(&device), made > 1, "count {count}");
    }
    let refused = |config: GraphicsConfig| {
        GraphicsDevice::with_platform(Arc::new(FakePlatform::new()), config).err()
    };
    let five = GraphicsConfig {
        display_count: 5,
        ..GraphicsConfig::default()
    };
    assert_eq!(refused(five), Some(Error::InvalidArgument));
    let windows_without_window = GraphicsConfig {
        display_count: 2,
        display_windows: vec![WindowConfig::default()],
        ..GraphicsConfig::default()
    };
    assert_eq!(
        refused(windows_without_window),
        Some(Error::InvalidArgument)
    );
    let one_window_short = GraphicsConfig {
        display_count: 3,
        window: Some(WindowConfig::default()),
        display_windows: vec![WindowConfig::default()],
        ..GraphicsConfig::default()
    };
    assert_eq!(refused(one_window_short), Some(Error::InvalidArgument));
}

/// Each display of a headless device reads back at 1920 × 1080, black with no scanout, and its
/// cursor is at (0, 0); a display past the count reads as (0, 0), refused, with nothing written,
/// and display 0's read and cursor are `read_display`'s and `cursor_position`'s
#[test]
fn each_display_reads_back_and_one_past_the_count_is_refused() {
    let device = headless_of(2, GraphicsCallbacks::default());
    let mut dst = vec![0x55u8; 1920 * 1080 * 4];
    for display in 0..2 {
        dst.fill(0x55);
        assert_eq!(
            device.read_display_at(display, &mut dst),
            ((1920, 1080), Ok(())),
            "display {display}"
        );
        assert!(
            dst.chunks(4).all(|px| px == [0, 0, 0, 0xff]),
            "display {display}"
        );
        assert_eq!(device.cursor_position_at(display), (0, 0));
    }
    dst.fill(0x55);
    assert_eq!(
        device.read_display_at(2, &mut dst),
        ((0, 0), Err(Error::InvalidArgument))
    );
    assert!(dst.iter().all(|&b| b == 0x55));
    assert_eq!(device.cursor_position_at(2), (0, 0));
    assert_eq!(device.read_display(&mut dst), ((1920, 1080), Ok(())));
    assert_eq!(device.cursor_position(), device.cursor_position_at(0));
}

/// What the host's cursor callbacks of every display were told, in order: the display, then
/// what.
type ToldAt = Arc<Mutex<Vec<(u32, Told)>>>;

/// Display 0's three cursor callbacks and the `_at` three of displays 1 and up, recording what
/// they are told.
fn telling_at(told: &ToldAt) -> GraphicsCallbacks {
    let at = |told: &ToldAt| Arc::clone(told);
    let (glyph, show, moved) = (at(told), at(told), at(told));
    let (glyph_at, show_at, moved_at) = (at(told), at(told), at(told));
    let glyph_of =
        |g: &CursorGlyph<'_>| Told::Glyph(g.width, g.height, g.hot_spot, g.rgba.to_vec());
    GraphicsCallbacks {
        cursor_glyph: Some(Arc::new(move |g: &CursorGlyph<'_>| {
            glyph.lock().push((0, glyph_of(g)));
        })),
        cursor_show: Some(Arc::new(move |s| show.lock().push((0, Told::Show(s))))),
        cursor_move: Some(Arc::new(move || moved.lock().push((0, Told::Move)))),
        cursor_glyph_at: Some(Arc::new(move |display, g: &CursorGlyph<'_>| {
            glyph_at.lock().push((display, glyph_of(g)));
        })),
        cursor_show_at: Some(Arc::new(move |display, s| {
            show_at.lock().push((display, Told::Show(s)));
        })),
        cursor_move_at: Some(Arc::new(move |display| {
            moved_at.lock().push((display, Told::Move));
        })),
        ..GraphicsCallbacks::default()
    }
}

/// `SET_CURSOR` of `display`, a 1 × 1 image at `slot`.
fn set_cursor(display: u32, slot: u64) -> Vec<u8> {
    RgfxSetCursor {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_SET_CURSOR,
            size: 0,
            seqno: 0,
        },
        display,
        width: 1,
        height: 1,
        hot_x: 0,
        hot_y: 0,
        reserved_24: 0,
        offset: slot,
    }
    .as_bytes()
    .to_vec()
}

/// `MOVE_CURSOR` of `display` to (`x`, `y`), shown.
fn move_cursor(display: u32, x: i32, y: i32) -> Vec<u8> {
    RgfxMoveCursor {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_MOVE_CURSOR,
            size: 0,
            seqno: 0,
        },
        display,
        x,
        y,
        visible: 1,
    }
    .as_bytes()
    .to_vec()
}

/// Display 1's cursor, set and moved by a guest that took `MULTI_DISPLAY`, reaches the host by
/// the `_at` callbacks naming display 1, and `cursor_position_at(1)` reads it, while display 0's
/// callbacks are told nothing and its cursor stays at (0, 0); a reset puts display 1's at
/// (0, 0) too
#[test]
fn display_1_s_cursor_reaches_the_host_by_the_at_callbacks() {
    let told: ToldAt = Arc::new(Mutex::new(Vec::new()));
    let device = headless_of(2, telling_at(&told));
    let wait = |count: usize| {
        eventually(|| told.lock().len() >= count);
        let told = told.lock();
        assert_eq!(told.len(), count, "{told:?}");
        told.iter()
            .map(|(display, t)| format!("{display} {t:?}"))
            .collect::<Vec<_>>()
    };
    device.mmio_write(
        RGFX_REG_DRIVER_FEATURES_HI.into(),
        (RGFX_FEATURE_MULTI_DISPLAY >> 32) as u32,
    );
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    let slot = device.region_size() - RGFX_CURSOR_SPAN;
    // SAFETY: the slot lies in the device's region, mapped while the device lives.
    unsafe {
        guest.write(slot as usize, &[10, 20, 30, 255]);
    }
    assert_eq!(
        answered(&device, &mut guest, set_cursor(1, slot)),
        RGFX_STATUS_OK
    );
    guest.send(&device, move_cursor(1, 100, 50));
    let glyph = Told::Glyph(1, 1, (0, 0), vec![30, 20, 10, 255]);
    assert_eq!(
        wait(3),
        [format!("1 {glyph:?}"), "1 Show(1)".into(), "1 Move".into()]
    );
    assert_eq!(device.cursor_position_at(1), (100, 50));
    assert_eq!(device.cursor_position(), (0, 0));

    device.reset();
    // Each display's cursor goes to (0, 0), display 0's first, and display 1's hides with its
    // image gone.
    assert_eq!(wait(6)[3..], ["0 Move", "1 Show(0)", "1 Move"]);
    assert_eq!(device.cursor_position_at(1), (0, 0));
}

/// A guest that did not take `MULTI_DISPLAY` sees display 0 alone: its `SET_CURSOR` of display 1
/// is answered `NO_DISPLAY`, and no callback of display 1's is told anything
#[test]
fn a_guest_that_did_not_take_multi_display_has_no_display_1_cursor() {
    let told: ToldAt = Arc::new(Mutex::new(Vec::new()));
    let device = headless_of(2, telling_at(&told));
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    let slot = device.region_size() - RGFX_CURSOR_SPAN;
    // SAFETY: the slot lies in the device's region, mapped while the device lives.
    unsafe {
        guest.write(slot as usize, &[10, 20, 30, 255]);
    }
    assert_eq!(
        answered(&device, &mut guest, set_cursor(1, slot)),
        RGFX_STATUS_NO_DISPLAY
    );
    assert_eq!(
        answered(&device, &mut guest, set_cursor(0, slot)),
        RGFX_STATUS_OK
    );
    eventually(|| !told.lock().is_empty());
    assert!(told.lock().iter().all(|(display, _)| *display == 0));
}

/// A window's pointer on a device of several displays tells a guest that took `POINTER_OVER`
/// that the host's pointer is over the window's display: `POINTER_OVER` of that display at the
/// event's point, with its buttons, is published, and the host's pointer callback, which moves
/// the guest's tablet, waits until the guest's cursor shows on the window's display, so the
/// tablet never puts the cursor at the event's spot of the display it is still on; a button's
/// change goes on at once, and the guest is told once
#[test]
fn a_window_s_pointer_tells_the_guest_its_display_and_waits_for_its_cursor() {
    let device = headless_of(2, GraphicsCallbacks::default());
    device.mmio_write(
        RGFX_REG_DRIVER_FEATURES_HI.into(),
        ((RGFX_FEATURE_MULTI_DISPLAY | RGFX_FEATURE_POINTER_OVER) >> 32) as u32,
    );
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    let producer = device.region_base() as usize + RGFX_CONTROL_COMPLETE_OFFSET as usize;
    // Each call's point, buttons and the completion ring's producer as the call saw it.
    type Calls = Mutex<Vec<(u16, u16, u32, u64)>>;
    let seen: Arc<Calls> = Arc::default();
    let at = Arc::clone(&seen);
    let mut config = WindowConfig::default();
    config.callbacks.pointer = Some(Arc::new(move |x, y, _width, _height, buttons| {
        // SAFETY: the control completion ring's producer, in the device's region, which stays
        // mapped while the device lives.
        let produced = unsafe { AtomicU64::from_ptr(producer as *mut u64).load(Ordering::SeqCst) };
        at.lock().push((x, y, buttons, produced));
    }));
    let link = Arc::new(OnceLock::new());
    let _ = link.set(device.device.pointer_link());
    crate::window::tell_pointer(&mut config, 1, &link);
    let pointer = config.callbacks.pointer.clone().unwrap();
    pointer(100, 50, 1920, 1080, 0);
    let p = RgfxPointerOver::read_from_bytes(&guest.event().unwrap()).unwrap();
    assert_eq!(
        (p.header.r#type, p.display, p.x, p.y, p.buttons),
        (RGFX_EVT_POINTER_OVER, 1, 100, 50, 0)
    );
    pointer(101, 51, 1920, 1080, 0);
    assert!(
        seen.lock().is_empty(),
        "held while the cursor is on display 0"
    );
    pointer(101, 51, 1920, 1080, reims_vgpu_window::BUTTON_LEFT);
    // The button's change went on, after the event was published.
    assert_eq!(
        *seen.lock(),
        [(101, 51, reims_vgpu_window::BUTTON_LEFT, 32)]
    );
    guest.send(&device, move_cursor(1, 101, 51));
    assert_eq!(answered(&device, &mut guest, vblank(0, 0)), RGFX_STATUS_OK);
    pointer(102, 52, 1920, 1080, reims_vgpu_window::BUTTON_LEFT);
    assert_eq!(seen.lock().len(), 2, "the cursor is on display 1");
    assert_eq!(seen.lock()[1].0, 102);
    assert!(guest.quiet_for(100), "told once");
}

/// `SET_VBLANK` of `display`.
fn vblank(display: u32, enabled: u32) -> Vec<u8> {
    RgfxSetVblank {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_SET_VBLANK,
            size: 0,
            seqno: 0,
        },
        display,
        enabled,
    }
    .as_bytes()
    .to_vec()
}

/// A command `command` of no answer, then one answered, so the first is carried out.
fn carried<P: ApiPlatform>(device: &GraphicsDevice<P>, guest: &mut Guest, command: Vec<u8>) {
    guest.send(device, command);
    assert_eq!(answered(device, guest, vblank(0, 0)), RGFX_STATUS_OK);
}

/// A captured pointer held at display 0's window's edge where the guest placed display 1 goes on
/// through display 1's window: the guest is told `POINTER_OVER` of display 1 with the drag's
/// button, display 1's host callback gets the positions once the guest's cursor shows there, and
/// back over display 0 the pointer is display 0's window's again, the guest told it
#[test]
fn a_captured_pointer_past_the_shared_edge_goes_on_through_the_other_display_s_window() {
    let told: Arc<Mutex<Vec<String>>> = Arc::default();
    let window = |display: u32| {
        let told = Arc::clone(&told);
        let mut config = WindowConfig::default();
        config.callbacks.pointer = Some(Arc::new(move |x, y, w, h, buttons| {
            told.lock()
                .push(format!("{display}: {x},{y} of {w}x{h} {buttons}"));
        }));
        config
    };
    let config = GraphicsConfig {
        display_count: 2,
        window: Some(window(0)),
        display_windows: vec![window(1)],
        ..GraphicsConfig::default()
    };
    let device = GraphicsDevice::with_platform(Arc::new(FakePlatform::new()), config).unwrap();
    device.mmio_write(
        RGFX_REG_DRIVER_FEATURES_HI.into(),
        ((RGFX_FEATURE_MULTI_DISPLAY | RGFX_FEATURE_POINTER_OVER | RGFX_FEATURE_DISPLAY_BOUNDS)
            >> 32) as u32,
    );
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    for display in 0..2u32 {
        let scanout = RgfxSetScanout {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_SET_SCANOUT,
                size: 0,
                seqno: 0,
            },
            display,
            width: 1920,
            height: 1080,
            stride: 7680,
            offset: RGFX_ALLOCATIONS_OFFSET + u64::from(display) * (64 << 20),
        };
        assert_eq!(
            answered(&device, &mut guest, scanout.as_bytes().to_vec()),
            RGFX_STATUS_OK
        );
        let bounds = RgfxDisplayBounds {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_DISPLAY_BOUNDS,
                size: 0,
                seqno: 0,
            },
            display,
            min_x: 1920 * display as i32,
            min_y: 0,
            max_x: 1920 * (display as i32 + 1),
            max_y: 1080,
            reserved_24: 0,
        };
        carried(&device, &mut guest, bounds.as_bytes().to_vec());
    }
    // Display 0's window as the device gives it its desktop.
    let link = Arc::new(OnceLock::new());
    let _ = link.set(device.device.pointer_link());
    let desktop = crate::window::Desktop::new(link);
    desktop.set_windows(device.windows.iter().map(|w| Some(w.weak())).collect());
    let mut config = WindowConfig::default();
    desktop.give(&mut config, 0);
    let beyond = config.callbacks.beyond.unwrap();
    let windows: Vec<_> = device.windows.iter().map(|w| w.weak()).collect();
    let left = reims_vgpu_window::BUTTON_LEFT;
    let mode = (1920, 1080);

    // The drag's click and its move to the edge, through display 0's window.
    let zero = windows[0].upgrade().unwrap();
    zero.pointer_from_beyond(1919, 500, 1920, 1080, left);
    assert!(beyond.adjoins((1919, 500), mode, (1, 0)));
    assert!(!beyond.adjoins((1919, 500), mode, (0, 1)));
    assert!(beyond.leave((1919, 500), mode));
    assert_eq!(beyond.moved((2.0, 0.0), left), Landed::Elsewhere);
    let p = RgfxPointerOver::read_from_bytes(&guest.event().unwrap()).unwrap();
    assert_eq!((p.display, p.x, p.y, p.buttons), (1, 1, 500, left));
    assert_eq!(beyond.moved((3.0, 0.0), left), Landed::Elsewhere);
    assert_eq!(
        *told.lock(),
        ["0: 1919,500 of 1920x1080 1"],
        "held until the cursor is on 1"
    );
    carried(&device, &mut guest, move_cursor(1, 4, 500));
    assert_eq!(beyond.moved((1.0, 2.0), left), Landed::Elsewhere);
    assert_eq!(told.lock()[1], "1: 5,502 of 1920x1080 1");
    // The button comes up on display 1; back over display 0, the guest is told it.
    assert_eq!(beyond.moved((0.0, 0.0), 0), Landed::Elsewhere);
    assert_eq!(told.lock()[2], "1: 5,502 of 1920x1080 0");
    assert_eq!(
        beyond.moved((-10.0, 0.0), 0),
        Landed::Here {
            at: (1915, 502),
            mode
        }
    );
    let p = RgfxPointerOver::read_from_bytes(&guest.event().unwrap()).unwrap();
    assert_eq!((p.display, p.x, p.y, p.buttons), (0, 1915, 502, 0));
    beyond.come_back();
    assert_eq!(
        beyond.moved((1.0, 0.0), 0),
        Landed::Lost,
        "the window's again"
    );
}

/// A device with a window offers `POINTER_MOTION`, and a headless one does not; a window the
/// device gives the guest's mouse ([`crate::window::give_motion`]), one with a pointer callback,
/// is told how long the guest has hidden its cursor, and its motion reaches a guest that took
/// `POINTER_MOTION` as the event; a window with no pointer callback is given no mouse
#[test]
fn a_window_s_device_offers_pointer_motion_and_drives_the_guest_s_mouse() {
    let offers = |device: &GraphicsDevice<FakePlatform>| {
        let hi = u64::from(device.mmio_read(RGFX_REG_FEATURES_HI.into())) << 32;
        hi & RGFX_FEATURE_POINTER_MOTION != 0
    };
    assert!(!offers(&headless_of(1, GraphicsCallbacks::default())));
    let with_pointer = || {
        let mut config = WindowConfig::default();
        config.callbacks.pointer = Some(Arc::new(|_x, _y, _width, _height, _buttons| {}));
        config
    };
    let config = GraphicsConfig {
        window: Some(with_pointer()),
        ..GraphicsConfig::default()
    };
    let device = GraphicsDevice::with_platform(Arc::new(FakePlatform::new()), config).unwrap();
    assert!(offers(&device));
    device.mmio_write(
        RGFX_REG_DRIVER_FEATURES_HI.into(),
        (RGFX_FEATURE_POINTER_MOTION >> 32) as u32,
    );
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    let link = Arc::new(OnceLock::new());
    let _ = link.set(device.device.pointer_link());
    let mut no_pointer = WindowConfig::default();
    crate::window::give_motion(&mut no_pointer, &link);
    assert!(
        no_pointer.callbacks.motion.is_none(),
        "no pointer, no mouse"
    );
    let mut config = with_pointer();
    crate::window::give_motion(&mut config, &link);
    let mouse = config.callbacks.motion.unwrap();
    assert_eq!(
        mouse.hidden_for(),
        None,
        "the guest has not hidden its cursor"
    );
    let hide = RgfxMoveCursor {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_MOVE_CURSOR,
            size: 0,
            seqno: 0,
        },
        display: 0,
        x: 10,
        y: 10,
        visible: 0,
    };
    carried(&device, &mut guest, hide.as_bytes().to_vec());
    assert!(mouse.hidden_for().is_some(), "hidden on its one display");
    assert!(mouse.moved((5, -3), 1, reims_vgpu_window::BUTTON_LEFT));
    let m = RgfxPointerMotion::read_from_bytes(&guest.event().unwrap()).unwrap();
    assert_eq!(
        (m.header.r#type, m.dx, m.dy, m.wheel, m.buttons),
        (RGFX_EVT_POINTER_MOTION, 5, -3, 1, RGFX_POINTER_BUTTON_LEFT)
    );
    carried(&device, &mut guest, move_cursor(0, 10, 10));
    assert_eq!(mouse.hidden_for(), None, "shown again");
}
