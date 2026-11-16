//! A device of several displays ([`GraphicsConfig::display_count`]): the counts and windows it
//! takes, each display's read-back, and each display's cursor at the host's callbacks.

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
/// event's point, with its buttons, is published before the host's pointer callback, which moves
/// the guest's tablet, is called with the event; told once, the next event goes to the host alone
#[test]
fn a_window_s_pointer_tells_the_guest_its_display_before_the_host() {
    let device = headless_of(2, GraphicsCallbacks::default());
    device.mmio_write(
        RGFX_REG_DRIVER_FEATURES_HI.into(),
        ((RGFX_FEATURE_MULTI_DISPLAY | RGFX_FEATURE_POINTER_OVER) >> 32) as u32,
    );
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    let producer = device.region_base() as usize + RGFX_CONTROL_COMPLETE_OFFSET as usize;
    let seen: Arc<Mutex<Vec<(u16, u16, u64)>>> = Arc::default();
    let at = Arc::clone(&seen);
    let mut config = WindowConfig::default();
    config.callbacks.pointer = Some(Arc::new(move |x, y, _width, _height, _buttons| {
        // SAFETY: the control completion ring's producer, in the device's region, which stays
        // mapped while the device lives.
        let produced = unsafe { AtomicU64::from_ptr(producer as *mut u64).load(Ordering::SeqCst) };
        at.lock().push((x, y, produced));
    }));
    let link = Arc::new(OnceLock::new());
    let _ = link.set(device.device.pointer_link());
    crate::window::tell_pointer(&mut config, 1, &link);
    let pointer = config.callbacks.pointer.clone().unwrap();
    pointer(100, 50, 1920, 1080, reims_vgpu_window::BUTTON_LEFT);
    let p = RgfxPointerOver::read_from_bytes(&guest.event().unwrap()).unwrap();
    assert_eq!(
        (p.header.r#type, p.display, p.x, p.y, p.buttons),
        (RGFX_EVT_POINTER_OVER, 1, 100, 50, RGFX_POINTER_BUTTON_LEFT)
    );
    // The host's callback saw the event's 32 bytes already published.
    assert_eq!(*seen.lock(), [(100, 50, 32)]);
    pointer(101, 51, 1920, 1080, 0);
    assert!(
        guest.quiet_for(200),
        "told once while the cursor stays on display 0"
    );
    assert_eq!(seen.lock().len(), 2);
}
