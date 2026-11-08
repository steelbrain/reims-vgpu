//! A graphics device's window, as the device's last reference ends it ([`WindowLink`]).

use std::sync::Arc;

use reims_vgpu_window::Window;

use crate::entry;

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
