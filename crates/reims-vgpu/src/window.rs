//! A graphics device's window, as the device's last reference ends it ([`WindowLink`]), its
//! pointer as it tells the guest which display it is over ([`tell_pointer`]), its captured
//! pointer as it goes past the image's edge onto the guest's other displays ([`Desktop`]), and as
//! it drives the guest's own mouse while the guest hides its cursor ([`give_motion`]).

use std::sync::{Arc, OnceLock, Weak};

use reims_vgpu_graphics::PointerLink;
use reims_vgpu_graphics::wire::{
    RGFX_POINTER_BUTTON_EXTRA, RGFX_POINTER_BUTTON_LEFT, RGFX_POINTER_BUTTON_MIDDLE,
    RGFX_POINTER_BUTTON_RIGHT, RGFX_POINTER_BUTTON_SIDE,
};
use reims_vgpu_window::{
    BUTTON_EXTRA, BUTTON_LEFT, BUTTON_MIDDLE, BUTTON_RIGHT, BUTTON_SIDE, Landed, PointerBeyond,
    PointerMotion, Window, WindowConfig,
};

use crate::entry;

// The window's buttons are `POINTER_OVER`'s, bit for bit.
const _: () = assert!(
    BUTTON_LEFT == RGFX_POINTER_BUTTON_LEFT
        && BUTTON_RIGHT == RGFX_POINTER_BUTTON_RIGHT
        && BUTTON_MIDDLE == RGFX_POINTER_BUTTON_MIDDLE
        && BUTTON_SIDE == RGFX_POINTER_BUTTON_SIDE
        && BUTTON_EXTRA == RGFX_POINTER_BUTTON_EXTRA
);

/// Has the pointer of display `display`'s window, `config`'s, tell the guest which display it is
/// over: each of its events tells the device, through `link` once the device is made, that the
/// host's pointer is over display `display` at the event's point with its buttons
/// ([`PointerLink::pointer_at`]), before the event goes to the host's callback, which moves the
/// guest's pointing device. The guest scales the one USB tablet over the display its cursor is
/// on, so a window's pointer reaches its own display once the guest has moved its cursor there
/// (docs/graphics-kext/design.md, "Multiple displays", X1): until then, for up to a second, the
/// device holds the event back from the host, but for one whose buttons changed, so the tablet
/// never puts the cursor at the event's spot of the display it is still on. A window with no
/// pointer callback has nothing to tell.
pub(crate) fn tell_pointer(
    config: &mut WindowConfig,
    display: u32,
    link: &Arc<OnceLock<PointerLink>>,
) {
    let Some(pointer) = config.callbacks.pointer.take() else {
        return;
    };
    let link = Arc::clone(link);
    config.callbacks.pointer = Some(Arc::new(move |x, y, width, height, buttons| {
        let goes_on = link
            .get()
            .is_none_or(|link| link.pointer_at(display, i32::from(x), i32::from(y), buttons));
        if goes_on {
            pointer(x, y, width, height, buttons);
        }
    }));
}

/// Has a window, `config`'s, drive the guest's own mouse with its captured pointer while the guest
/// hides its cursor (docs/graphics-kext/design.md, "A pointer for games"): through `link`, once
/// the device is made, the device's `POINTER_MOTION`, which a guest that took it hands its mouse.
/// A window with no pointer callback gives the guest no pointer, and so no mouse either.
pub(crate) fn give_motion(config: &mut WindowConfig, link: &Arc<OnceLock<PointerLink>>) {
    if config.callbacks.pointer.is_none() {
        return;
    }
    config.callbacks.motion = Some(Arc::new(GuestMouse {
        link: Arc::clone(link),
    }));
}

/// The guest's own mouse, through the device's link once it is made ([`give_motion`]).
struct GuestMouse {
    /// The device's link.
    link: Arc<OnceLock<PointerLink>>,
}

impl PointerMotion for GuestMouse {
    fn hidden_for(&self) -> Option<u64> {
        self.link.get()?.cursor_hidden_for()
    }

    fn moved(&self, by: (i32, i32), wheel: i32, buttons: u32) -> bool {
        self.link
            .get()
            .is_some_and(|link| link.motion(by.0, by.1, wheel, buttons))
    }
}

/// The guest's desktop, as a device of several displays' windows carry a captured pointer across
/// it (docs/graphics-kext/design.md, "Multiple displays", "A drag across displays"): the device,
/// which knows where the guest placed each display (`DISPLAY_BOUNDS`) and where the pointer is on
/// it ([`PointerLink::leave`]), and each display's window, through whose pointer (and so through
/// [`tell_pointer`]'s `POINTER_OVER`) the pointer goes to the guest wherever it lands.
pub(crate) struct Desktop {
    /// The device's link, once it is made.
    link: Arc<OnceLock<PointerLink>>,
    /// Each display's window, by its id, once they are made: `None` for a display without one.
    windows: OnceLock<Vec<Option<Weak<Window>>>>,
}

impl Desktop {
    /// The desktop over the device that `link` will name.
    pub(crate) fn new(link: Arc<OnceLock<PointerLink>>) -> Arc<Desktop> {
        Arc::new(Desktop {
            link,
            windows: OnceLock::new(),
        })
    }

    /// The displays' windows, display `i`'s at `i`.
    pub(crate) fn set_windows(&self, windows: Vec<Option<Weak<Window>>>) {
        let _ = self.windows.set(windows);
    }

    /// Display `display`'s window's side of the desktop, which `config` takes as its
    /// [`PointerBeyond`].
    pub(crate) fn give(self: &Arc<Self>, config: &mut WindowConfig, display: u32) {
        config.callbacks.beyond = Some(Arc::new(DisplayBeyond {
            desktop: Arc::clone(self),
            display,
        }));
    }

    /// Display `display`'s window, while it lives.
    fn window(&self, display: u32) -> Option<Arc<Window>> {
        self.windows
            .get()?
            .get(display as usize)?
            .as_ref()?
            .upgrade()
    }
}

/// One display's window's side of the [`Desktop`].
struct DisplayBeyond {
    /// The desktop.
    desktop: Arc<Desktop>,
    /// The display's id.
    display: u32,
}

impl PointerBeyond for DisplayBeyond {
    fn adjoins(&self, at: (u16, u16), mode: (u16, u16), outward: (i8, i8)) -> bool {
        self.desktop
            .link
            .get()
            .is_some_and(|link| link.adjoins(self.display, at, mode, outward))
    }

    fn leave(&self, at: (u16, u16), mode: (u16, u16)) -> bool {
        self.desktop
            .link
            .get()
            .is_some_and(|link| link.leave(self.display, at, mode))
    }

    fn moved(&self, by: (f64, f64), buttons: u32) -> Landed {
        let Some(spot) = self
            .desktop
            .link
            .get()
            .and_then(|link| link.away_moved(by.0, by.1))
        else {
            return Landed::Lost;
        };
        if let Some(window) = self.desktop.window(spot.display) {
            window.pointer_from_beyond(spot.x, spot.y, spot.width, spot.height, buttons);
        }
        if spot.display == self.display {
            Landed::Here {
                at: (spot.x, spot.y),
                mode: (spot.width, spot.height),
            }
        } else {
            Landed::Elsewhere
        }
    }

    fn come_back(&self) {
        if let Some(link) = self.desktop.link.get() {
            link.come_back();
        }
    }
}

/// A device's window, as every reference to the device shares it: its drop, at the device's last
/// reference, is the window's end ([`Window::close`]).
pub(crate) struct WindowLink {
    window: Arc<Window>,
}

impl WindowLink {
    /// The link to `window`, which closes it when it drops.
    pub(crate) fn new(window: Arc<Window>) -> WindowLink {
        WindowLink { window }
    }

    /// The window, for the tests' look at it.
    #[cfg(test)]
    pub(crate) fn window(&self) -> &Window {
        &self.window
    }

    /// The window, held weakly: the [`Desktop`]'s hold, which keeps it alive no longer.
    pub(crate) fn weak(&self) -> Weak<Window> {
        Arc::downgrade(&self.window)
    }
}

impl Drop for WindowLink {
    /// Closes the window, as an entry: it waits for nothing, and what the window lets go of here
    /// (the host's window callbacks, the sink) is released on this thread.
    fn drop(&mut self) {
        entry::run(|_| self.window.close());
    }
}
