//! A graphics device's window, as the device's last reference ends it ([`WindowLink`]), and its
//! pointer as it tells the guest which display it is over ([`tell_pointer`]).

use std::sync::{Arc, OnceLock};

use reims_vgpu_graphics::PointerLink;
use reims_vgpu_graphics::wire::{
    RGFX_POINTER_BUTTON_EXTRA, RGFX_POINTER_BUTTON_LEFT, RGFX_POINTER_BUTTON_MIDDLE,
    RGFX_POINTER_BUTTON_RIGHT, RGFX_POINTER_BUTTON_SIDE,
};
use reims_vgpu_window::{
    BUTTON_EXTRA, BUTTON_LEFT, BUTTON_MIDDLE, BUTTON_RIGHT, BUTTON_SIDE, Window, WindowConfig,
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
/// guest's pointing device. The
/// guest scales the one USB tablet over the display its cursor is on, so a window's pointer
/// reaches its own display once the guest has moved its cursor there (docs/graphics-kext/
/// design.md, "Multiple displays", X1). A window with no pointer callback has nothing to tell.
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
        if let Some(link) = link.get() {
            link.pointer_at(display, i32::from(x), i32::from(y), buttons);
        }
        pointer(x, y, width, height, buttons);
    }));
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
}

impl Drop for WindowLink {
    /// Closes the window, as an entry: it waits for nothing, and what the window lets go of here
    /// (the host's window callbacks, the sink) is released on this thread.
    fn drop(&mut self) {
        entry::run(|_| self.window.close());
    }
}
