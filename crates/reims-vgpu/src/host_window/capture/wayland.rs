//! Wayland shortcut capture via `keyboard-shortcuts-inhibit-unstable-v1`.
//!
//! # Contract
//!
//! `zwp_keyboard_shortcuts_inhibit_manager_v1.inhibit_shortcuts(surface, seat)`
//! yields a `zwp_keyboard_shortcuts_inhibitor_v1`. While that object lives the
//! compositor must forward every key event from `seat` to `surface` instead of
//! acting on its own shortcuts. Destroying it restores compositor handling.
//! The compositor confirms with `active`, or declines with `inactive` — the
//! protocol permits it to refuse, so "the request was sent" is not the same
//! claim as "the shortcuts are ours" and this file does not conflate them.
//!
//! # Why this borrows the window's display rather than opening its own
//!
//! The inhibitor is bound to a `wl_surface`, and the only surface that matters
//! is the one the window already created. A second connection to the compositor
//! could not name it — Wayland object ids are per-connection — so the request
//! has to travel on the window's own connection.
//!
//! `Backend::from_foreign_display` wraps the existing `wl_display` **without**
//! taking ownership of it: it allocates its own libwayland event queue and
//! leaves the window's teardown of the display alone. All of this file's objects
//! live on that private queue, so dispatching it can never consume an event the
//! window's own loop was waiting for.

use wayland_client::protocol::{wl_registry, wl_seat};
use wayland_client::{Connection, Dispatch, EventQueue, Proxy, QueueHandle};
use wayland_protocols::wp::keyboard_shortcuts_inhibit::zv1::client::{
    zwp_keyboard_shortcuts_inhibit_manager_v1::ZwpKeyboardShortcutsInhibitManagerV1,
    zwp_keyboard_shortcuts_inhibitor_v1::{self, ZwpKeyboardShortcutsInhibitorV1},
};

use super::{Capture, CaptureError};

/// Globals collected from the registry, plus what the compositor said about the
/// inhibitor it was last asked for.
#[derive(Default)]
struct State {
    manager: Option<ZwpKeyboardShortcutsInhibitManagerV1>,
    seat: Option<wl_seat::WlSeat>,
    /// `Some(true)` after an `active`, `Some(false)` after an `inactive`. `None`
    /// until the compositor has answered for the current inhibitor.
    honoured: Option<bool>,
}

impl Dispatch<wl_registry::WlRegistry, ()> for State {
    fn event(
        state: &mut Self,
        registry: &wl_registry::WlRegistry,
        event: wl_registry::Event,
        _: &(),
        _: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        let wl_registry::Event::Global {
            name,
            interface,
            version,
        } = event
        else {
            return;
        };
        match interface.as_str() {
            "zwp_keyboard_shortcuts_inhibit_manager_v1" if state.manager.is_none() => {
                state.manager = Some(registry.bind(name, version.min(1), qh, ()));
            }
            "wl_seat" if state.seat.is_none() => {
                state.seat = Some(registry.bind(name, version.min(7), qh, ()));
            }
            _ => {}
        }
    }
}

impl Dispatch<wl_seat::WlSeat, ()> for State {
    fn event(
        _: &mut Self,
        _: &wl_seat::WlSeat,
        _: wl_seat::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        // The seat is an argument to `inhibit_shortcuts` and nothing else; its
        // capability and name events carry nothing this file decides on.
    }
}

impl Dispatch<ZwpKeyboardShortcutsInhibitManagerV1, ()> for State {
    fn event(
        _: &mut Self,
        _: &ZwpKeyboardShortcutsInhibitManagerV1,
        _: <ZwpKeyboardShortcutsInhibitManagerV1 as Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        // The manager is a factory; the protocol gives it no events.
    }
}

impl Dispatch<ZwpKeyboardShortcutsInhibitorV1, ()> for State {
    fn event(
        state: &mut Self,
        _: &ZwpKeyboardShortcutsInhibitorV1,
        event: zwp_keyboard_shortcuts_inhibitor_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        // The compositor is allowed to decline. Record which answer came back so
        // `set` can report a refusal rather than assume the request took.
        state.honoured = Some(matches!(
            event,
            zwp_keyboard_shortcuts_inhibitor_v1::Event::Active
        ));
    }
}

/// Shortcut capture over the window's own Wayland connection.
pub struct WaylandCapture {
    conn: Connection,
    queue: EventQueue<State>,
    state: State,
    surface: wayland_client::protocol::wl_surface::WlSurface,
    /// The live inhibitor, when capture is on. Dropping capture destroys it,
    /// which is the protocol's own way of handing the shortcuts back.
    inhibitor: Option<ZwpKeyboardShortcutsInhibitorV1>,
}

impl WaylandCapture {
    /// Bind the inhibit manager and a seat on the window's existing connection.
    ///
    /// # Safety of the two foreign pointers
    ///
    /// `display.display` and `window.surface` come from the window's own
    /// `raw-window-handle`, so they are live for as long as the window is, and
    /// this type is owned by the window. The display is wrapped unowned, so this
    /// type never disconnects it; the surface is wrapped by id and this type
    /// issues no request that would destroy it.
    pub fn new(
        display: raw_window_handle::WaylandDisplayHandle,
        window: raw_window_handle::WaylandWindowHandle,
    ) -> Result<Self, CaptureError> {
        use wayland_client::backend::{Backend, ObjectId};
        use wayland_client::protocol::wl_surface::WlSurface;

        // SAFETY: a live `wl_display` from the window this capture belongs to,
        // wrapped unowned — see the type doc.
        let backend = unsafe { Backend::from_foreign_display(display.display.as_ptr().cast()) };
        let conn = Connection::from_backend(backend);

        let mut queue: EventQueue<State> = conn.new_event_queue();
        let qh = queue.handle();
        let mut state = State::default();
        let _registry = conn.display().get_registry(&qh, ());
        queue
            .roundtrip(&mut state)
            .map_err(|e| CaptureError::Protocol(e.to_string()))?;

        if state.manager.is_none() {
            return Err(CaptureError::NoInhibitManager);
        }
        if state.seat.is_none() {
            return Err(CaptureError::NoSeat);
        }

        // SAFETY: a live `wl_surface` from the same window and connection.
        let surface_id = unsafe {
            ObjectId::from_ptr(WlSurface::interface(), window.surface.as_ptr().cast())
                .map_err(|e| CaptureError::Protocol(e.to_string()))?
        };
        let surface = WlSurface::from_id(&conn, surface_id)
            .map_err(|e| CaptureError::Protocol(e.to_string()))?;

        Ok(Self {
            conn,
            queue,
            state,
            surface,
            inhibitor: None,
        })
    }
}

impl Capture for WaylandCapture {
    fn set(&mut self, active: bool) -> Result<(), CaptureError> {
        match (active, self.inhibitor.is_some()) {
            (true, false) => {
                let (Some(manager), Some(seat)) = (&self.state.manager, &self.state.seat) else {
                    return Err(CaptureError::NoInhibitManager);
                };
                self.state.honoured = None;
                let qh = self.queue.handle();
                self.inhibitor = Some(manager.inhibit_shortcuts(&self.surface, seat, &qh, ()));
                // The compositor may answer `inactive`; one round trip turns
                // "asked" into "took", which is the difference between a log
                // that says capture is on and a guest that actually gets the
                // keys.
                self.queue
                    .roundtrip(&mut self.state)
                    .map_err(|e| CaptureError::Protocol(e.to_string()))?;
                if self.state.honoured == Some(false) {
                    return Err(CaptureError::PartialOnly("compositor_declined"));
                }
            }
            (false, true) => {
                if let Some(inhibitor) = self.inhibitor.take() {
                    inhibitor.destroy();
                }
                self.state.honoured = None;
                self.conn
                    .flush()
                    .map_err(|e| CaptureError::Protocol(e.to_string()))?;
            }
            // Already in the requested state; the protocol has nothing to do.
            (true, true) | (false, false) => {}
        }
        Ok(())
    }

    fn describe(&self) -> &'static str {
        "wayland_shortcuts_inhibit"
    }
}

impl Drop for WaylandCapture {
    fn drop(&mut self) {
        // Hand the shortcuts back explicitly. A compositor cleans up on client
        // disconnect, but this client outlives the window: the display belongs
        // to the window and stays connected.
        if let Some(inhibitor) = self.inhibitor.take() {
            inhibitor.destroy();
            let _ = self.conn.flush();
        }
    }
}
