use std::ffi::c_void;

use parking_lot::Mutex;

use super::*;

/// What the C functions saw: one line per call.
#[derive(Default)]
struct Rec {
    lines: Mutex<Vec<String>>,
}

impl Rec {
    fn new() -> Arc<Rec> {
        Arc::new(Rec::default())
    }

    /// The recorder as a ctx (the test keeps the `Arc`).
    fn ctx(self: &Arc<Rec>) -> *mut c_void {
        Arc::as_ptr(self).cast_mut().cast()
    }

    fn push(ctx: *mut c_void, line: String) {
        // SAFETY: every recorder ctx is `Arc::as_ptr` of a `Rec` the test keeps alive.
        unsafe { &*ctx.cast::<Rec>() }.lines.lock().push(line);
    }

    fn lines(&self) -> Vec<String> {
        self.lines.lock().clone()
    }
}

unsafe extern "C-unwind" fn t_pointer(ctx: *mut c_void, x: u16, y: u16, w: u16, h: u16, b: u32) {
    Rec::push(ctx, format!("pointer {x},{y} in {w}x{h} buttons {b:#x}"));
}
unsafe extern "C-unwind" fn t_wheel(ctx: *mut c_void, dy: i32, dx: i32) {
    Rec::push(ctx, format!("wheel {dy},{dx}"));
}
unsafe extern "C-unwind" fn t_key(ctx: *mut c_void, usage: u16, down: bool) {
    Rec::push(ctx, format!("key {usage:#x} {down}"));
}
unsafe extern "C-unwind" fn t_close(ctx: *mut c_void) {
    Rec::push(ctx, "close".to_owned());
}
unsafe extern "C-unwind" fn t_scroll(ctx: *mut c_void, scroll: *const RvgWindowScroll) {
    // SAFETY: the library's event, live for the call.
    let s = unsafe { &*scroll };
    Rec::push(
        ctx,
        format!(
            "scroll {} bytes {},{} phase {} momentum {} notches {},{}",
            s.struct_size, s.dy, s.dx, s.phase, s.momentum, s.notches_dy, s.notches_dx
        ),
    );
}

fn window_table(rec: &Arc<Rec>) -> RvgWindowCallbacks {
    RvgWindowCallbacks {
        struct_size: sized::size_u32::<RvgWindowCallbacks>(),
        ctx: rec.ctx(),
        destroy: None,
        pointer: Some(t_pointer),
        wheel: Some(t_wheel),
        key: Some(t_key),
        close: Some(t_close),
        scroll: Some(t_scroll),
    }
}

/// The config's mode is the mode the window opens at, none where either side is 0
#[test]
fn the_window_config_s_mode() {
    let to = |mode_width, mode_height| {
        RvgWindowConfig {
            struct_size: sized::size_u32::<RvgWindowConfig>(),
            callbacks: std::ptr::null(),
            mode_width,
            mode_height,
        }
        .to_config(WindowCallbacks::default())
        .mode
    };
    assert_eq!(to(2560, 1600), Some((2560, 1600)));
    assert_eq!(to(0, 1600), None);
    assert_eq!(to(2560, 0), None);
}

/// Each closure calls its C function with the table's ctx and the arguments the window passes;
/// an absent function is no closure
#[test]
fn window_callbacks_forward() {
    let rec = Rec::new();
    let t = Table::new(rec.ctx(), None);
    let cb = window_callbacks(&t, &window_table(&rec));
    (cb.pointer.unwrap())(
        10,
        20,
        640,
        480,
        RVG_WINDOW_BUTTON_LEFT | RVG_WINDOW_BUTTON_EXTRA,
    );
    (cb.wheel.unwrap())(-2, 1);
    (cb.key.as_ref().unwrap())(0x04, true);
    (cb.key.unwrap())(0x04, false);
    (cb.close.unwrap())();
    let scroll = cb.scroll.unwrap();
    let event = |dy, dx, phase, momentum, notches| ScrollEvent {
        dy,
        dx,
        phase,
        momentum,
        notches,
    };
    scroll(event(
        1536,
        -256,
        ScrollPhase::Began,
        MomentumPhase::None,
        (0, 0),
    ));
    scroll(event(0, 0, ScrollPhase::Ended, MomentumPhase::None, (0, 0)));
    scroll(event(
        10240,
        0,
        ScrollPhase::None,
        MomentumPhase::Changed,
        (1, -2),
    ));
    assert_eq!(
        rec.lines(),
        [
            "pointer 10,20 in 640x480 buttons 0x11",
            "wheel -2,1",
            "key 0x4 true",
            "key 0x4 false",
            "close",
            "scroll 28 bytes 1536,-256 phase 2 momentum 0 notches 0,0",
            "scroll 28 bytes 0,0 phase 4 momentum 0 notches 0,0",
            "scroll 28 bytes 10240,0 phase 0 momentum 2 notches 1,-2",
        ]
    );
    let none = RvgWindowCallbacks {
        pointer: None,
        wheel: None,
        key: None,
        close: None,
        scroll: None,
        ..window_table(&rec)
    };
    let cb = window_callbacks(&t, &none);
    assert!(cb.pointer.is_none() && cb.wheel.is_none() && cb.key.is_none());
    assert!(cb.close.is_none() && cb.scroll.is_none());
}

/// Each phase is its RVG_WINDOW_SCROLL_PHASE_* and each momentum its
/// RVG_WINDOW_SCROLL_MOMENTUM_*
#[test]
fn scroll_phases_are_their_constants() {
    let phases = [
        (ScrollPhase::None, RVG_WINDOW_SCROLL_PHASE_NONE),
        (ScrollPhase::MayBegin, RVG_WINDOW_SCROLL_PHASE_MAY_BEGIN),
        (ScrollPhase::Began, RVG_WINDOW_SCROLL_PHASE_BEGAN),
        (ScrollPhase::Changed, RVG_WINDOW_SCROLL_PHASE_CHANGED),
        (ScrollPhase::Ended, RVG_WINDOW_SCROLL_PHASE_ENDED),
        (ScrollPhase::Cancelled, RVG_WINDOW_SCROLL_PHASE_CANCELLED),
    ];
    for (phase, value) in phases {
        assert_eq!(phase_value(phase), value, "{phase:?}");
    }
    let momenta = [
        (MomentumPhase::None, RVG_WINDOW_SCROLL_MOMENTUM_NONE),
        (MomentumPhase::Began, RVG_WINDOW_SCROLL_MOMENTUM_BEGAN),
        (MomentumPhase::Changed, RVG_WINDOW_SCROLL_MOMENTUM_CHANGED),
        (MomentumPhase::Ended, RVG_WINDOW_SCROLL_MOMENTUM_ENDED),
    ];
    for (momentum, value) in momenta {
        assert_eq!(momentum_value(momentum), value, "{momentum:?}");
    }
}
