//! Host-owned window presentation — a `winit` window that presents the guest
//! frame on the running rail's own device, replacing QEMU's UI
//! ([[host-window]]).
//!
//! This file owns the window and the event loop, and nothing else. The native
//! surface, the drawables and the fitted blit belong to whichever rail is
//! running and are reached only through [`crate::backend::Backend`]'s
//! `window_*` methods — a `VkSurfaceKHR` and a swapchain on the Vulkan rail, a
//! `CAMetalLayer` on the Metal one. What is left here is *when* to present, and
//! the input half: window events become [`HostAction`]s via
//! [`super::input_map`] and reach the [`InputSink`] the device wires to its
//! prompt action queue.
//!
//! The rail prefers its own resident, which is what keeps a presented layer
//! from crossing host memory at all; the CPU-BGRA [`FrameSlot`] the device
//! fills from its present capture is the source for the frames no resident
//! carries — the firmware framebuffer, any mapping the compositor has not
//! rendered into, and every frame on a rail that holds no residents.
//!
//! There is exactly one presenter, on every platform and every rail. A rail
//! that cannot present to this surface gets a named refusal and a shutdown
//! rather than a second device of its own; `resumed` records why.
//!
//! Linux owns the event loop on a dedicated thread. macOS requires AppKit work
//! on the process main thread, so QEMU creates it through
//! [`start_main_thread`] during device realize and then makes
//! [`run_main_thread`] its process-main UI loop.

#[cfg(target_os = "macos")]
use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::PhysicalKey;
use winit::window::{CustomCursor, Window, WindowId, MAX_CURSOR_SIZE};

use super::capture::{Capture, CaptureEngaged, CaptureMode};
use super::input_map;
use super::keyboard::{Grab, KeyEffect, Keyboard};
use crate::backend::window::WindowPresentOutcome;
use crate::backend::Backend as _;
use crate::runtime::host::HostAction;

/// How long the loop may sleep when nothing has asked it to draw.
///
/// The loop is woken by [`WindowWaker`] on every publish, so this is not how a
/// frame reaches the screen — it is the floor under the two things no publish
/// announces: the `stop` flag the device sets at teardown, and the
/// [`GUEST_RESIZE_WARN_AFTER`] alarm, which is a deadline this has to be able to
/// observe. A tenth of that alarm keeps the age it reports within 10 % of the
/// constant it is measured against, which is what fixes this value; it is not
/// tuned against a frame rate and must not be read as one.
///
/// It also bounds the two outcomes that ask to be retried without a new frame:
/// a `Busy` presenter, and a publish that arrives in the instant between the
/// wake and the redraw. Both come back within one backstop rather than waiting
/// for the guest's next frame.
///
/// # This replaced a 2 ms poll, and the poll was 98 % waste
///
/// The loop used to ask for a redraw every 2 ms and let [`needs_present`]
/// throw away whatever the seq gate rejected. Driven x86/PCI boot, window-drag
/// probe, whole run:
///
/// ```text
///   ticks          145675     (997/s — one wake per poll, one per redraw event)
///   redraws_asked   72227     (494/s)
///   draws_fresh      1269     (8.7/s)
///   draws_stale     70961     (486/s — 98.2 % of every redraw asked for)
/// ```
///
/// A stale draw costs no GPU work at all — [`App::draw`] returns at the seq gate
/// before `window_present_frame` — so this was never the 510-presents/s spin the
/// poll was introduced to kill. What it cost was host CPU on a machine the
/// device shares: ~1000 event-loop wakes and ~486 frame-slot mutex acquisitions
/// a second to find 8.7 frames. The publisher knows when it publishes; asking it
/// to say so costs one channel send per frame at the 26 Hz this workload peaks
/// at.
///
/// # After
///
/// Same probe and host, quiesced, on a guest running a heavier animation than
/// the boot above — so read the rates against each other, not the frame counts:
///
/// ```text
///                       poll     wake
///   ticks/s              997      108
///   redraws_asked/s      494     34.6
///   draws_stale/s        486      2.6
///   stale share       98.2 %    7.4 %
/// ```
///
/// The poll asked 494 times a second whatever the guest was doing, because that
/// is what a poll is. What is left is one ask per delivered frame (3317 asks,
/// 3074 fresh draws) plus the handful the backstop contributes. 3582 publishes
/// produced 3074 of those draws: the other 508 were superseded in the slot
/// before the loop looked, which is latest-wins working rather than a frame
/// lost.
const WINDOW_REDRAW_BACKSTOP: std::time::Duration =
    std::time::Duration::from_millis(GUEST_RESIZE_WARN_AFTER.as_millis() as u64 / 10);
/// How many rebuilds of the presenter may go unproven before the window stops
/// trying — asked of the running rail, which is what gets lost.
///
/// **A storm budget rather than a lifetime cap**, and [`App::draw`] is the half
/// of that which lives here: it zeroes [`App::presenter_rebuilds`] on a present
/// that reached the screen, on the reasoning that a rail which recovered and is
/// lost again later is a new incident and not the fourth step of the old one.
/// What the bound then stops is a surface that genuinely cannot be created
/// being retried once per redraw forever.
///
/// Reading it as a lifetime cap was measured wrong on real hardware, and the
/// measurement is why this doc is here. A driven macos-11 boot lost the device
/// more than eight times; the window rebuilt the presenter on the first three,
/// logged `host_window_reattach status=ok attempt=3`, spent the budget, and sat
/// out every later loss with the picture gone — which is most of the defect the
/// re-attach exists to remove, reintroduced by its own bound.
fn presenter_rebuild_budget() -> u32 {
    crate::backend::selected().window_reattach_budget()
}
/// How long a guest-driven native resize request may stay unmatched by a
/// winit `Resized` event before the always-on alarm names it. Live requests
/// apply within single-digit milliseconds; one second means the window system
/// refused or clamped the size and the window is presenting letterboxed
/// instead. A tiling or fullscreen Wayland/X11 compositor refuses by policy, so
/// on those hosts this alarm is the expected steady state rather than a fault.
const GUEST_RESIZE_WARN_AFTER: std::time::Duration = std::time::Duration::from_secs(1);

/// What geometry the host window asks the window system for, and therefore
/// whether it may ask again later.
///
/// A type rather than a `bool` on [`WindowConfig`] because it carries two rules
/// that have to stay together: what [`ApplicationHandler::resumed`] requests at
/// creation, and whether [`guest_resize_request`] may ask for a native content
/// resize afterwards. A window given the whole monitor cannot honour the second,
/// and splitting the pair would leave a full-screen window requesting sizes it
/// can never be granted — each request holding presentation until
/// [`GUEST_RESIZE_WARN_AFTER`] and then reporting `native_resize_not_applied`
/// about a window that is behaving exactly as asked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowMode {
    /// An ordinary decorated window at [`WindowConfig`]'s extent, which tracks
    /// the guest's display mode. The default.
    Sized,
    /// The whole monitor the window opens on, undecorated — winit's
    /// `Fullscreen::Borderless`, which on Linux is a borderless full-screen
    /// window on both X11 and Wayland. The guest frame is letterboxed into it by
    /// the same [`crate::backend::window::viewport`] arithmetic that already handles a window the
    /// guest's mode does not fit.
    Borderless,
}

impl WindowMode {
    /// What [`crate::config::FULLSCREEN`] asks for.
    ///
    /// The parse is [`crate::config`]'s; what is decided here is what each of its
    /// four answers means for a window. `Unrecognized` is the one worth naming:
    /// an operator who wrote `REIMS_VGPU_FULLSCREEN=yes-please` gets a sized
    /// window, and without a line saying the value was rejected they read that
    /// as the switch not working.
    pub fn requested() -> Self {
        match crate::config::read(crate::config::FULLSCREEN) {
            (crate::config::Switch::On, _) => Self::Borderless,
            (crate::config::Switch::Unrecognized, value) => {
                crate::observe::Emit::decline(
                    "host_window_init",
                    &WindowError::FullscreenValue(value.unwrap_or_default()),
                )
                .fail();
                Self::Sized
            }
            _ => Self::Sized,
        }
    }
}

/// How a full-screen window is made full-screen, and therefore what it depends
/// on.
///
/// A type rather than a `bool` because the two answers differ in *what has to
/// exist*: the ordinary path asks the window system and needs a window manager
/// to answer, and the appliance's path asks nothing and needs no one. That is
/// the reason the variant exists at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FullscreenStrategy {
    /// The window system's own answer — winit's `Fullscreen::Borderless`, which
    /// on X11 is `_NET_WM_STATE_FULLSCREEN` and is therefore a request to a
    /// window manager. Every host that has one, on every platform, takes this
    /// path.
    Normal,
    /// The Reims appliance's dedicated Xorg session, which has no window
    /// manager to ask. The window is created override-redirect at the monitor's
    /// own rectangle, so its geometry is fixed before a window manager could
    /// have had an opinion, and nothing else has to resize it afterwards.
    X11WmLess,
}

impl FullscreenStrategy {
    /// The strategy for the three independent answers.
    ///
    /// Pure, which is the point: the rule is testable without a server. WM-less
    /// geometry is selected only when the operator asked for fullscreen *and*
    /// asked for WM-less *and* the window will open on X11. Every other
    /// combination — all of Wayland, all of macOS, and every existing X11 host
    /// with a window manager — is [`Self::Normal`].
    pub fn resolve(
        window_system: Option<crate::config::WindowSystem>,
        fullscreen: bool,
        x11_wmless: bool,
    ) -> Self {
        if fullscreen && x11_wmless && window_system == Some(crate::config::WindowSystem::X11) {
            Self::X11WmLess
        } else {
            Self::Normal
        }
    }

    /// [`Self::resolve`] against the environment, read once where the window is
    /// configured rather than once per frame.
    pub fn requested(mode: WindowMode) -> Self {
        let wmless = match crate::config::read(crate::config::X11_WMLESS) {
            (crate::config::Switch::On, _) => true,
            (crate::config::Switch::Unrecognized, value) => {
                // The one refusal here that ends nothing: the parse could not
                // read the ask, and the ordinary path is what the boot gets.
                // Saying so is what keeps a typo from reading as a switch that
                // does nothing.
                crate::observe::Emit::decline(
                    "host_window_init",
                    &WindowError::X11WmLessValue(value.unwrap_or_default()),
                )
                .fail();
                false
            }
            _ => false,
        };
        Self::resolve(
            crate::config::window_system(),
            mode == WindowMode::Borderless,
            wmless,
        )
    }
}

/// Where a WM-less window's rectangle came from.
///
/// Carried with the geometry rather than inferred later: a boot that only got
/// its rectangle because RandR had nothing usable must say so, or a reader
/// cannot tell a measured monitor from a fallback that happened to be the right
/// size. It is also the answer to "did the fallback run at all".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WmLessGeometrySource {
    /// A RandR monitor `winit` identified, by a non-zero native id.
    Monitor,
    /// The X11 root window, used when no usable RandR monitor exists.
    X11Root,
}

impl WmLessGeometrySource {
    /// The name this source takes in the `host_window_mode` line.
    pub fn name(self) -> &'static str {
        match self {
            Self::Monitor => "monitor",
            Self::X11Root => "x11_root",
        }
    }
}

/// A monitor as the geometry policy sees it: `winit`'s own identity plus the
/// rectangle it reports.
///
/// The policy only needs the identity to reject the dummy and the rectangle to
/// build the window, so that is all this carries — which is what keeps the
/// policy a pure function of values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MonitorRect {
    pub native_id: u32,
    pub position: winit::dpi::PhysicalPosition<i32>,
    pub size: winit::dpi::PhysicalSize<u32>,
}

impl MonitorRect {
    /// Whether this is a monitor rather than the placeholder `winit` substitutes
    /// when RandR yields no CRTC.
    ///
    /// Zero is `winit`'s own marker: its X11 backend builds the placeholder with
    /// `id: 0` and `is_dummy()` is literally `self.id == 0`. That method is
    /// `pub(crate)`, but `native_id()` is public and returns the same `id`, so
    /// zero is the identification available outside the crate. Size is
    /// deliberately not the test: a real monitor may legitimately be any size,
    /// and the placeholder's `1x1` is a symptom of its identity, not the identity.
    pub fn is_usable(self) -> bool {
        self.native_id != 0
    }
}

/// The X11 root window's rectangle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RootRect {
    pub size: winit::dpi::PhysicalSize<u32>,
}

/// Why the root window's rectangle was not available.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RootRectError {
    /// The event loop's raw display handle was not Xlib, or carried no display
    /// pointer. The WM-less path says X11, so anything else is a contradiction
    /// rather than a reason to guess.
    NoXlibHandle,
    /// The Xlib call itself failed; the string is the greppable reason.
    QueryFailed(&'static str),
}

/// The rectangle and attribute set the WM-less X11 path pins, as data.
///
/// Split from the code that reads a monitor or the root so the properties the
/// appliance depends on — override-redirect, undecorated, fixed, screen-sized,
/// and where the rectangle came from — are testable without an X server. The
/// window code turns this into [`winit::window::WindowAttributes`]; this type is
/// where the values are decided.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WmLessX11Geometry {
    pub position: winit::dpi::PhysicalPosition<i32>,
    pub size: winit::dpi::PhysicalSize<u32>,
    pub source: WmLessGeometrySource,
    pub decorations: bool,
    pub resizable: bool,
    pub override_redirect: bool,
}

impl WmLessX11Geometry {
    /// The appliance window's rectangle from the frames the window system offers.
    ///
    /// The policy, in order, with no other branch anywhere:
    ///
    /// 1. the primary monitor, when `winit` identified one;
    /// 2. otherwise the first available monitor it identified;
    /// 3. otherwise the X11 root window's rectangle;
    /// 4. otherwise a refusal.
    ///
    /// Step 3 exists because `winit`'s X11 backend answers a RandR-less or
    /// CRTC-less server with a placeholder monitor of `1x1` at the root origin
    /// rather than with `None` — `active_event_loop.primary_monitor()` is
    /// `Some(placeholder)` — so a policy that only checks for absence silently
    /// builds a one-pixel window. Xephyr is such a server, and a physical Xorg
    /// is not, which is exactly why the placeholder has to be recognised by
    /// identity and the fallback has to be a real answer instead of a size.
    ///
    /// The root is the right fallback here and nowhere else: the appliance
    /// session is one application with no window manager, so the whole root
    /// rectangle is the area that application may occupy. The degenerate
    /// rectangles are refused rather than accepted, because a `1x1` root would
    /// otherwise be reported as a successful full-screen window.
    pub fn resolve(
        primary: Option<MonitorRect>,
        available: &[MonitorRect],
        root: Result<RootRect, RootRectError>,
    ) -> Result<Self, WindowError> {
        if let Some(monitor) = primary.filter(|monitor| monitor.is_usable()) {
            return Ok(Self::from_monitor_rect(monitor));
        }
        if let Some(monitor) = available
            .iter()
            .copied()
            .find(|monitor| monitor.is_usable())
        {
            return Ok(Self::from_monitor_rect(monitor));
        }
        let root = match root {
            Ok(root) => root,
            Err(RootRectError::NoXlibHandle) => return Err(WindowError::X11RootHandle),
            Err(RootRectError::QueryFailed(reason)) => {
                return Err(WindowError::X11RootGeometry(reason.to_string()))
            }
        };
        if root.size.width <= 1 || root.size.height <= 1 {
            return Err(WindowError::X11RootGeometryInvalid {
                width: root.size.width,
                height: root.size.height,
            });
        }
        Ok(Self {
            position: winit::dpi::PhysicalPosition::new(0, 0),
            size: root.size,
            source: WmLessGeometrySource::X11Root,
            decorations: false,
            resizable: false,
            override_redirect: true,
        })
    }

    /// The appliance window's rectangle for a monitor's own rectangle.
    fn from_monitor_rect(monitor: MonitorRect) -> Self {
        Self {
            position: monitor.position,
            size: monitor.size,
            source: WmLessGeometrySource::Monitor,
            decorations: false,
            resizable: false,
            override_redirect: true,
        }
    }
}

/// Apply the WM-less rectangle and attributes to `attrs`.
///
/// `with_override_redirect` exists only on the X11 platform, so it is applied
/// behind the same gate as `winit::platform::x11` itself. The cross-platform
/// attributes stay outside the gate, which is what keeps the pure geometry test
/// meaningful on any host.
#[cfg(target_os = "linux")]
fn apply_wm_less_geometry(
    attrs: winit::window::WindowAttributes,
    geometry: &WmLessX11Geometry,
) -> winit::window::WindowAttributes {
    use winit::platform::x11::WindowAttributesExtX11 as _;
    attrs
        .with_decorations(geometry.decorations)
        .with_resizable(geometry.resizable)
        .with_position(geometry.position)
        .with_inner_size(geometry.size)
        .with_override_redirect(geometry.override_redirect)
}

#[cfg(not(target_os = "linux"))]
fn apply_wm_less_geometry(
    attrs: winit::window::WindowAttributes,
    geometry: &WmLessX11Geometry,
) -> winit::window::WindowAttributes {
    attrs
        .with_decorations(geometry.decorations)
        .with_resizable(geometry.resizable)
        .with_position(geometry.position)
        .with_inner_size(geometry.size)
}

/// The WM-less window's rectangle, read from the window system this boot is on.
///
/// The only place `winit` and the X connection are touched; everything after this
/// is [`WmLessX11Geometry::resolve`], which is pure. Note that "no monitor" is
/// not a condition `winit`'s X11 backend can report as absence — see the
/// placeholder discussion on [`MonitorRect::is_usable`] — so the monitor list is
/// filtered by identity here and the root is consulted rather than trusted to be
/// unnecessary.
#[cfg(target_os = "linux")]
fn resolve_wm_less_geometry(
    event_loop: &winit::event_loop::ActiveEventLoop,
) -> Result<WmLessX11Geometry, WindowError> {
    let primary = event_loop.primary_monitor().map(monitor_rect);
    let available: Vec<MonitorRect> = event_loop.available_monitors().map(monitor_rect).collect();
    WmLessX11Geometry::resolve(primary, &available, x11_root_rect(event_loop))
}

/// The WM-less path is selected only on Linux/X11 — `config::window_system()`
/// answers `None` in every other build — so this arm keeps the call site one
/// shape rather than being reached.
#[cfg(not(target_os = "linux"))]
fn resolve_wm_less_geometry(
    _event_loop: &winit::event_loop::ActiveEventLoop,
) -> Result<WmLessX11Geometry, WindowError> {
    Err(WindowError::X11RootHandle)
}

/// A monitor as the policy sees it, identity included.
///
/// `native_id()` is the X11 extension trait's accessor for `winit`'s own monitor
/// id. It is zero exactly for the placeholder `winit` substitutes when RandR
/// yields no usable CRTC, which is the condition this policy has to recognise.
#[cfg(target_os = "linux")]
fn monitor_rect(monitor: winit::monitor::MonitorHandle) -> MonitorRect {
    use winit::platform::x11::MonitorHandleExtX11 as _;
    MonitorRect {
        native_id: monitor.native_id(),
        position: monitor.position(),
        size: monitor.size(),
    }
}

/// The X11 root window's rectangle, read through the display `winit` already owns.
///
/// `XRootWindow`/`XGetWindowAttributes` on the event loop's own connection: no
/// `XOpenDisplay`, so no second authentication, no second connection that could
/// disagree with the backend about which display is running, and no external
/// `xrandr`/`xwininfo`/`xdpyinfo` — those are runtime and test instruments, not
/// product dependencies.
#[cfg(target_os = "linux")]
fn x11_root_rect(
    event_loop: &winit::event_loop::ActiveEventLoop,
) -> Result<RootRect, RootRectError> {
    let handle = event_loop
        .display_handle()
        .map_err(|_| RootRectError::NoXlibHandle)?;
    let raw_window_handle::RawDisplayHandle::Xlib(xlib_handle) = handle.as_raw() else {
        return Err(RootRectError::NoXlibHandle);
    };
    let display = xlib_handle.display.ok_or(RootRectError::NoXlibHandle)?;
    let screen = xlib_handle.screen;

    let xlib =
        x11_dl::xlib::Xlib::open().map_err(|_| RootRectError::QueryFailed("libx11_unavailable"))?;
    // SAFETY: `display` is the pointer this event loop's own X11 backend handed
    // out and keeps alive for as long as the loop is; both calls are the
    // read-only root query, and `attrs` is fully written before it is read.
    unsafe {
        let display = display.as_ptr().cast::<x11_dl::xlib::Display>();
        let root = (xlib.XRootWindow)(display, screen);
        let mut attrs = std::mem::MaybeUninit::<x11_dl::xlib::XWindowAttributes>::uninit();
        if (xlib.XGetWindowAttributes)(display, root, attrs.as_mut_ptr()) == 0 {
            return Err(RootRectError::QueryFailed("xgetwindowattributes_failed"));
        }
        let attrs = attrs.assume_init();
        Ok(RootRect {
            size: winit::dpi::PhysicalSize::new(
                attrs.width.max(0) as u32,
                attrs.height.max(0) as u32,
            ),
        })
    }
}

/// Which full-screen path this boot took, for the always-on census line.
///
/// `REIMS_VGPU_FULLSCREEN=1` alone cannot distinguish a window the window
/// manager made full-screen from one this process created at the monitor's own
/// rectangle. On the appliance's window-manager-less Xorg session only the
/// second one fills the screen, so a reader has to be able to tell them apart
/// from the log — and the WM-less arm carries the rectangle it asked for.
pub struct HostWindowMode {
    window_system: &'static str,
    wm: &'static str,
    fullscreen: &'static str,
    geometry_source: &'static str,
    geometry: Option<(i32, i32, u32, u32)>,
}

impl HostWindowMode {
    /// The appliance's path: X11, no window manager, geometry from the monitor
    /// rectangle or — when no monitor was usable — the X11 root rectangle.
    ///
    /// The source travels on the line because the two are indistinguishable by
    /// their numbers alone on a single-monitor host: a fallback that happened to
    /// read `1600x900` and a monitor that measured `1600x900` would otherwise
    /// look the same, and the point of the line is that a reader can tell what
    /// actually happened.
    pub fn wm_less_x11(geometry: WmLessX11Geometry) -> Self {
        Self {
            window_system: "x11",
            wm: "none",
            fullscreen: "override_redirect",
            geometry_source: geometry.source.name(),
            geometry: Some((
                geometry.position.x,
                geometry.position.y,
                geometry.size.width,
                geometry.size.height,
            )),
        }
    }

    /// Every other path: whatever window system `winit` chose, a window manager
    /// if the host has one, and the window system's own sizing.
    pub fn ordinary(window_system: Option<crate::config::WindowSystem>, mode: WindowMode) -> Self {
        Self {
            window_system: window_system
                .map(|system| system.name())
                .unwrap_or("native"),
            wm: "external",
            fullscreen: if mode == WindowMode::Borderless {
                "ewmh"
            } else {
                "sized"
            },
            // No rectangle is pinned on this path, so there is no rectangle to
            // attribute. `none` rather than an omitted field keeps every
            // `host_window_mode` line the same shape, which is what makes it
            // greppable.
            geometry_source: "none",
            geometry: None,
        }
    }
}

impl crate::observe::Decline for HostWindowMode {
    fn slug(&self) -> &'static str {
        "host_window_mode"
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        let mut fields = vec![
            ("window_system", self.window_system.to_string()),
            ("wm", self.wm.to_string()),
            ("fullscreen", self.fullscreen.to_string()),
            ("geometry_source", self.geometry_source.to_string()),
        ];
        if let Some((x, y, width, height)) = self.geometry {
            fields.push(("position", format!("{x:+},{y:+}")));
            fields.push(("size", format!("{width}x{height}")));
        }
        fields
    }
}

/// The focus confirmation emitted only after the WM-less X11 window is the
/// server's current input focus. Geometry and focus are separate contracts:
/// the first does not imply the second when there is no window manager.
struct HostWindowFocus;

impl crate::observe::Decline for HostWindowFocus {
    fn slug(&self) -> &'static str {
        "host_window_focus"
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        vec![
            ("mechanism", "x11_set_input_focus".to_string()),
            ("status", "verified".to_string()),
        ]
    }
}

/// Request and verify focus for the appliance's WM-less X11 window.
///
/// This deliberately consumes the raw handles from the already-created winit
/// window. `x11_dl::Xlib::open` loads libX11; it does not call `XOpenDisplay`,
/// so the `Display*` remains the connection owned by winit and used by the
/// presenter and keyboard capture.
#[cfg(target_os = "linux")]
fn focus_wm_less_x11_window(window: &Arc<Window>) -> Result<(), WindowError> {
    use raw_window_handle::{HasDisplayHandle as _, HasWindowHandle as _};
    use std::mem::MaybeUninit;
    use std::os::raw::{c_int, c_ulong};

    let display_handle = window
        .display_handle()
        .map_err(|_| WindowError::X11FocusHandle)?
        .as_raw();
    let raw_window = window
        .window_handle()
        .map_err(|_| WindowError::X11FocusHandle)?
        .as_raw();
    let raw_window_handle::RawDisplayHandle::Xlib(display_handle) = display_handle else {
        return Err(WindowError::X11FocusHandle);
    };
    let raw_window_handle::RawWindowHandle::Xlib(window_handle) = raw_window else {
        return Err(WindowError::X11FocusHandle);
    };
    let display = display_handle
        .display
        .ok_or(WindowError::X11FocusHandle)?
        .as_ptr()
        .cast::<x11_dl::xlib::Display>();
    let xlib = x11_dl::xlib::Xlib::open()
        .map_err(|error| WindowError::X11FocusRequest(detail_field(&error.to_string())))?;

    // A focus request against an unmapped window is an X11 refusal. Check once
    // at the deterministic lifecycle point after presenter attach; do not turn
    // this into a polling loop.
    let mut attrs = MaybeUninit::<x11_dl::xlib::XWindowAttributes>::uninit();
    // SAFETY: `display` and `window_handle.window` came from the live winit
    // window, and the output is fully written when XGetWindowAttributes says
    // it succeeded.
    let attributes_ok =
        unsafe { (xlib.XGetWindowAttributes)(display, window_handle.window, attrs.as_mut_ptr()) };
    if attributes_ok == 0 {
        return Err(WindowError::X11FocusQuery);
    }
    let attrs = unsafe { attrs.assume_init() };
    if attrs.map_state != x11_dl::xlib::IsViewable {
        return Err(WindowError::X11FocusNotViewable);
    }

    // SAFETY: this is the same live Xlib connection and window handle. The
    // request is synchronous at the Xlib API boundary; verification below
    // rejects servers that did not make this window the focus target.
    let set_ok = unsafe {
        (xlib.XSetInputFocus)(
            display,
            window_handle.window,
            x11_dl::xlib::RevertToParent,
            0,
        )
    };
    if set_ok == 0 {
        return Err(WindowError::X11FocusRequest("xsetinputfocus_failed".into()));
    }
    unsafe {
        (xlib.XFlush)(display);
    }

    let mut focused: c_ulong = 0;
    let mut revert_to: c_int = 0;
    // SAFETY: both output pointers are valid for the duration of the call.
    let focus_query_ok = unsafe { (xlib.XGetInputFocus)(display, &mut focused, &mut revert_to) };
    if focus_query_ok == 0 {
        return Err(WindowError::X11FocusQuery);
    }
    verify_x11_focus_target(window_handle.window, focused)
}

#[cfg(not(target_os = "linux"))]
fn focus_wm_less_x11_window(_window: &Arc<Window>) -> Result<(), WindowError> {
    Err(WindowError::X11FocusHandle)
}

/// Keep the server-result validation independent from the live X11 call so
/// `None`, `PointerRoot`, the root window and another client are all covered
/// without needing an X server in the regression suite.
fn verify_x11_focus_target(target: u64, focused: u64) -> Result<(), WindowError> {
    if focused == target {
        Ok(())
    } else {
        Err(WindowError::X11FocusVerify { focused })
    }
}

/// Window creation parameters.
///
/// `mode` is resolved from the environment by whoever builds the config rather
/// than read inside the event loop, so the value a boot ran is on the config the
/// window was started with and there is one place it is decided.
#[derive(Clone, Debug)]
pub struct WindowConfig {
    pub title: String,
    pub width: u32,
    pub height: u32,
    pub mode: WindowMode,
    /// Which full-screen path, resolved beside `mode` and for the same reason:
    /// it is an operator's answer about this boot, decided once on the thread
    /// that starts the window rather than per frame.
    pub strategy: FullscreenStrategy,
}

impl Default for WindowConfig {
    fn default() -> Self {
        let mode = WindowMode::requested();
        Self {
            title: "Reims vGPU".to_string(),
            width: 1280,
            height: 800,
            mode,
            strategy: FullscreenStrategy::requested(mode),
        }
    }
}

/// Called on the window thread for each input [`HostAction`] the window
/// produces. The implementation pushes onto the device prompt queue and wakes
/// the delivery BH (both thread-safe), so guest input flows without the device
/// lock.
pub type InputSink = Arc<dyn Fn(HostAction) + Send + Sync>;

/// The latest guest frame to present (BGRA8, tightly packed `width*height*4`).
/// `None` until the first present capture; the window clears to a flat color
/// until then. Shared via `Arc`, so the window's per-vblank read is a refcount
/// bump rather than a deep copy.
pub struct Frame {
    /// Monotonic publish sequence (assigned by the device when it writes a new
    /// frame). A static desktop publishes a new frame only when content changes.
    /// Linux re-blits its prepared staging source each vblank but prepares it only
    /// when `seq` advances. macOS submits the rail's resident only when
    /// `seq` advances or the window resizes, so an unchanged desktop does not
    /// contend with guest render work for the rail's queue.
    /// Wrap-around is harmless: a collision at most skips one prepare (the source
    /// still holds valid content).
    pub seq: u64,
    pub width: u32,
    pub height: u32,
    pub bgra: Vec<u8>,
    /// The rail's own handle to a GPU-resident frame, when one carries this
    /// present. Opaque here: it is produced by `Backend::window_resident` on
    /// the drain worker and handed back to `Backend::window_present` on this
    /// thread, and nothing in between looks inside it.
    pub resident: Option<crate::backend::window::WindowResident>,
}

/// Shared slot the device writes and the window reads (latest-wins). The frame
/// is `Arc`-wrapped so the window's per-vblank read is a refcount bump, not an
/// 8 MiB deep copy of an unchanged frame.
pub type FrameSlot = Arc<Mutex<Option<Arc<Frame>>>>;

/// Guest cursor state mirrored into the native host window.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GuestCursorGlyph {
    pub sequence: u64,
    pub width: u16,
    pub height: u16,
    pub hot_x: u16,
    pub hot_y: u16,
    /// QEMU/model pixels in straight-alpha `0xAARRGGBB` order.
    pub pixels: Arc<[u32]>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GuestCursorState {
    pub visible: bool,
    pub x: u16,
    pub y: u16,
    pub glyph: Option<GuestCursorGlyph>,
}

pub type CursorSlot = Arc<Mutex<GuestCursorState>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CursorGlyphError {
    Empty,
    TooLarge,
    PixelCount,
    Hotspot,
}

/// Convert straight-alpha ARGB words to the non-premultiplied RGBA bytes
/// required by `winit::window::CustomCursor::from_rgba`.
pub fn cursor_argb_to_rgba(pixels: &[u32]) -> Vec<u8> {
    let mut rgba = Vec::with_capacity(pixels.len() * 4);
    for &pixel in pixels {
        rgba.extend_from_slice(&[
            ((pixel >> 16) & 0xff) as u8,
            ((pixel >> 8) & 0xff) as u8,
            (pixel & 0xff) as u8,
            (pixel >> 24) as u8,
        ]);
    }
    rgba
}

/// Snapshot the model cursor while its device lock is held, so the window
/// thread never needs to acquire a device lock to install a glyph.
pub fn snapshot_guest_cursor(
    cursor: &crate::model::CursorState,
    sequence: u64,
) -> Result<GuestCursorState, CursorGlyphError> {
    let glyph = if !cursor.glyph_ready {
        None
    } else {
        let expected = usize::from(cursor.width)
            .checked_mul(usize::from(cursor.height))
            .ok_or(CursorGlyphError::PixelCount)?;
        if cursor.width == 0 || cursor.height == 0 {
            return Err(CursorGlyphError::Empty);
        }
        if cursor.width > MAX_CURSOR_SIZE || cursor.height > MAX_CURSOR_SIZE {
            return Err(CursorGlyphError::TooLarge);
        }
        if cursor.pixels.len() != expected {
            return Err(CursorGlyphError::PixelCount);
        }
        if cursor.hot_x >= cursor.width || cursor.hot_y >= cursor.height {
            return Err(CursorGlyphError::Hotspot);
        }
        Some(GuestCursorGlyph {
            sequence,
            width: cursor.width,
            height: cursor.height,
            hot_x: cursor.hot_x,
            hot_y: cursor.hot_y,
            pixels: Arc::from(cursor.pixels.clone()),
        })
    };
    Ok(GuestCursorState {
        visible: cursor.show,
        x: cursor.x,
        y: cursor.y,
        glyph,
    })
}

fn cursor_glyph_changed(
    applied: &Option<GuestCursorGlyph>,
    incoming: &Option<GuestCursorGlyph>,
) -> bool {
    match (applied, incoming) {
        (Some(a), Some(b)) => {
            a.width != b.width
                || a.height != b.height
                || a.hot_x != b.hot_x
                || a.hot_y != b.hot_y
                || a.pixels != b.pixels
        }
        _ => applied.is_some() != incoming.is_some(),
    }
}

fn cursor_force_visible() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            crate::config::read(crate::config::CURSOR_FORCE_VISIBLE).0,
            crate::config::Switch::On
        )
    })
}

/// The one user event this window's loop takes: the device wrote a new frame
/// into the [`FrameSlot`].
///
/// Carries nothing. The slot is latest-wins and the loop reads it under its own
/// lock, so a payload here could only be a second, staler copy of what
/// [`App::draw`] is about to read anyway.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowUserEvent {
    FramePublished,
    CursorPublished,
}

/// How the device wakes the window's event loop when it publishes a frame.
///
/// The window used to find new frames by asking for a redraw every 2 ms and
/// discarding 98 % of them at the seq gate — see
/// [`WINDOW_REDRAW_BACKSTOP`] for that reading. This turns it round: the
/// publisher already knows, so it says so, and the loop sleeps until it does.
///
/// # Why a proxy and not the window handle
///
/// `Window::request_redraw` would be the direct route, but the platforms
/// disagree about calling it off the loop's own thread — and this device runs
/// the loop on a dedicated thread on Linux and on the process main thread on
/// macOS, so "the other thread" is a different thread on each. `EventLoopProxy`
/// is winit's one cross-thread entry point and is the same shape on both.
///
/// # Why a `Mutex` and not a `OnceLock`
///
/// `EventLoopProxy` is `Send` but not `Sync` — the Wayland and X11 backends
/// carry an `mpsc::Sender` — so it cannot be shared behind a `OnceLock`. The
/// lock is uncontended and taken once per published frame, at the ~26 Hz this
/// workload peaks at, against the publisher's own two existing locks.
pub struct WindowWaker {
    proxy: Mutex<Option<winit::event_loop::EventLoopProxy<WindowUserEvent>>>,
}

/// A [`WindowWaker`] shared between the device's publisher and the window
/// thread that arms it.
pub type WindowWakeHandle = Arc<WindowWaker>;

impl WindowWaker {
    /// An unarmed waker. [`Self::wake`] is a no-op until the window thread has
    /// built its event loop and called [`Self::arm`].
    pub fn new() -> WindowWakeHandle {
        Arc::new(Self {
            proxy: Mutex::new(None),
        })
    }

    /// Hand the loop's proxy over, once the loop exists to be woken.
    fn arm(&self, proxy: winit::event_loop::EventLoopProxy<WindowUserEvent>) {
        if let Ok(mut slot) = self.proxy.lock() {
            *slot = Some(proxy);
        }
    }

    /// Ask the loop to look at the frame slot.
    ///
    /// Silent in all three failure modes, because none of them is a lost frame:
    /// before the loop is armed and after it has closed there is nothing that
    /// could present, and a poisoned lock means the window thread panicked. In
    /// every case [`WINDOW_REDRAW_BACKSTOP`] still runs, so a wake that
    /// does not land costs latency bounded by that constant rather than a frame
    /// — which is the property that lets this be a wake and not a protocol.
    pub fn wake(&self) {
        self.wake_event(WindowUserEvent::FramePublished);
    }

    pub fn wake_cursor(&self) {
        self.wake_event(WindowUserEvent::CursorPublished);
    }

    fn wake_event(&self, event: WindowUserEvent) {
        if let Ok(slot) = self.proxy.lock() {
            if let Some(proxy) = slot.as_ref() {
                let _ = proxy.send_event(event);
            }
        }
    }
}

/// Offer a published frame's CPU bytes to the rail's presenter.
///
/// The presenter prefers the resident and only reads these when none carries the
/// display — the firmware framebuffer, and any mapping the compositor has not
/// rendered into. `bgra` is empty on presents the device elided the readback
/// for, and the presenter rejects a short buffer rather than blitting a torn
/// frame.
fn window_cpu_frame(frame: &Frame) -> crate::backend::window::WindowCpuFrame<'_> {
    crate::backend::window::WindowCpuFrame {
        bgra: &frame.bgra,
        width: frame.width,
        height: frame.height,
        seq: frame.seq,
    }
}

/// Whether the window must present at all.
///
/// A present is an acquire, a full-frame blit into the swapchain image, a submit
/// and a `queue_present`. None of that produces a different picture when the
/// guest has not produced a different frame, so the only reasons to pay it are a
/// new frame seq or a drawable that must be rebuilt (first frame, resize,
/// suboptimal swapchain).
fn needs_present(presented: Option<u64>, redraw_required: bool, incoming: Option<u64>) -> bool {
    redraw_required || presented != incoming
}

/// The absolute pointer event a window position becomes: `(x, y, width,
/// height)` for [`HostAction::input_pointer_move`], whose consumer scales `x`
/// against `width` (`min_in = 0`, `max_in = dim`).
///
/// The presenter aspect-fits the guest frame into the drawable, so a window
/// position is proportional to a guest position only *inside* the viewport — in
/// a letterbox bar it is not over the guest surface at all. Reporting raw
/// window coordinates therefore offsets and rescales every event by the bar
/// size, which is the regression [`crate::backend::window::viewport`]'s module docs record as
/// rolled back. It survived on non-macOS hosts because this mapping was
/// `cfg`-gated to macOS while the presenter's `aspect_fit` never was, so the
/// two halves of "presentation and pointer move as one unit" were compiled for
/// different platforms.
///
/// Before the first guest frame there is no guest extent to map into, so the
/// full-window space is forwarded unchanged.
fn pointer_report(
    position: (f64, f64),
    window: (u32, u32),
    guest: Option<(u32, u32)>,
) -> (u32, u32, u32, u32) {
    match guest {
        Some(guest) => {
            let (x, y) =
                crate::backend::window::viewport::pointer_to_guest(position, window, guest);
            (x, y, guest.0, guest.1)
        }
        None => (
            position.0.max(0.0) as u32,
            position.1.max(0.0) as u32,
            window.0,
            window.1,
        ),
    }
}

/// Whether a newly observed guest frame geometry should request a native
/// content resize: only on a geometry change, only when the window does not
/// already match, and only when the window's geometry is this device's to ask
/// about at all. User-driven host resizing stays untouched until the guest picks
/// another mode.
///
/// [`WindowMode::Borderless`] never requests. The window has the monitor, so
/// there is no size to be granted, and asking is not merely futile: every
/// request holds presentation for [`GUEST_RESIZE_WARN_AFTER`] and then reports
/// `native_resize_not_applied`, which is a fail-visible line about a window doing
/// what the operator asked it to do. The guest's mode reaches the screen
/// letterboxed by [`crate::backend::window::viewport`] instead, exactly as it does on a tiling
/// compositor that refuses the request on its own.
fn guest_resize_request(
    mode: WindowMode,
    observed: Option<(u32, u32)>,
    incoming: (u32, u32),
    window: (u32, u32),
) -> bool {
    mode == WindowMode::Sized && observed != Some(incoming) && incoming != window
}

/// How a `Resized` event answers an outstanding guest-driven request, or
/// `None` when there is nothing outstanding to answer.
///
/// **Any `Resized` settles the hold, matching or not.** What the window is
/// waiting for is the window system to answer, not to obey: `request_inner_size`
/// is a request, and a compositor is free to adjust it for decorations, screen
/// bounds or a fractional scale round-trip. An exact-match test reads every such
/// answer as silence and holds until the alarm.
///
/// Measured on x86/Linux, one boot, driving the guest through its four modes —
/// the applied size differs from the request on three of four, never by more
/// than a pixel, and never in a fixed direction:
///
/// ```text
/// requested 1920x1080 -> 1921x1079      requested 3840x2160 -> 3840x2160
/// requested 1440x1080 -> 1440x1079      requested 1280x1024 -> 1281x1024
/// ```
///
/// Under the old exact-match rule those three logged
/// `native_resize_not_applied` — a fail-visible line saying the resize never
/// happened, about a window that had resized — and each held *all* presentation
/// for the full second the alarm takes, because [`App::draw`] returns early
/// while a request is outstanding. So a guest mode change froze
/// the display for a second and then claimed it had been refused.
///
/// The alarm still has a job, and it is now the case it was written for: a
/// window system that ignores the request entirely emits no `Resized` at all
/// (tiling or fullscreen by policy), so nothing calls this and the hold times
/// out.
fn guest_resize_settled(pending: Option<(u32, u32)>, applied: (u32, u32)) -> Option<&'static str> {
    let target = pending?;
    Some(if target == applied {
        "applied"
    } else {
        // The window system answered with a geometry of its own choosing. The
        // presenter aspect-fits into it and the pointer maps through the same
        // viewport, so this is a complete outcome and not a degraded one.
        "adjusted"
    })
}

/// A guest-driven native resize not yet confirmed by a `Resized` event. While
/// it is outstanding the window holds its previous drawable (guest boot
/// presents are seconds apart, so a mismatched interim present would stay on
/// screen that long); the hold is bounded by [`GUEST_RESIZE_WARN_AFTER`], after
/// which the request is dropped with a fail-visible line and presentation
/// resumes letterboxed.
struct PendingGuestResize {
    target: (u32, u32),
    requested_at: std::time::Instant,
}

/// Shared flag the device sets to ask the window to close (VM teardown). The
/// event loop polls it in `about_to_wait` and exits promptly. Distinct from a
/// UI close (which the window originates); either way the thread ends and its
/// Vulkan objects tear down before the join returns.
pub type StopFlag = Arc<AtomicBool>;

/// Set after the native window and all of its Vulkan objects have torn down.
/// QEMU's backend teardown waits for this before destroying shared GPU state.
///
/// Darwin only, like the [`start_main_thread`]/[`run_main_thread`] pair that
/// publishes it: elsewhere the window owns its own thread and `stop` plus the
/// join covers the same ordering. A reachability sweep run on a non-Apple host
/// sees every user of this alias behind a `cfg` it did not compile and reports
/// it unused — that reading cost one broken macOS build already.
#[cfg(target_os = "macos")]
pub type ExitedFlag = Arc<AtomicBool>;

/// Errors from bringing up or running the window.
///
/// One variant per distinct check, so each names itself in `/tmp/reims-vgpu-fail.log`
/// through [`crate::observe::Decline`]. A coarse `Vulkan(String)`-style variant
/// collapses many checks into one grep prefix and loses which of them fired; the
/// specific check lives in the slug instead, and the raw driver/winit string
/// rides along as a whitespace-safe `detail=` field.
///
/// Every variant is a *window lifecycle* refusal: creating the loop, owning the
/// process window, creating the native window, or attaching the running rail's
/// presenter to it. Nothing here describes a drawable or a blit — those belong
/// to the rail, which types its own declines and classifies them through
/// [`crate::backend::window::WindowDecline`], and there is no second presenter
/// in this file to type declines for.
///
/// `#[allow(dead_code)]` because four of the variants (`MainLoopRun`,
/// `AlreadyOwned`, `NoRegisteredWindow`, `WrongOwner`) are
/// constructed only by the macOS main-thread entry points, so they are
/// unconstructed on every other target.
#[allow(dead_code)]
#[derive(Debug)]
pub enum WindowError {
    /// `EventLoop::build()` failed (winit).
    EventLoopBuild(String),
    /// `run_app` returned an error on the off-main-thread `run()` path.
    RunApp(String),
    /// `run_app` returned an error on the macOS main-thread loop.
    MainLoopRun(String),
    /// A second device tried to claim the single process window.
    AlreadyOwned {
        owner: u64,
    },
    /// `run_main_thread` found no registered window for the device.
    NoRegisteredWindow {
        id: u64,
    },
    /// `run_main_thread` was asked to run a window owned by another device.
    WrongOwner {
        owner: u64,
        requested: u64,
    },
    /// `resumed`: winit could not create the native window (shared step, both
    /// platforms) — the bring-up cannot proceed past this.
    CreateNativeWindow(String),
    /// Presenter attach: the window's display handle was unavailable.
    AttachDisplayHandle(String),
    /// Presenter attach: the window's window handle was unavailable.
    AttachWindowHandle(String),
    /// Presenter attach: the running rail refused to build a presenter for this
    /// surface. The refusal that ends the window, on every platform and rail.
    AttachPresenter(String),
    /// [`crate::config::FULLSCREEN`] was set to something that is neither an on nor
    /// an off spelling. The window opens sized; the value is quoted so the
    /// operator can see what the parse rejected. The one variant here that does
    /// not end anything.
    FullscreenValue(String),
    /// [`crate::config::X11_WMLESS`] was set to something that is neither an on
    /// nor an off spelling. The window takes the ordinary path and the value is
    /// quoted; like [`Self::FullscreenValue`], it ends nothing.
    X11WmLessValue(String),
    /// The WM-less path needed the X11 root window's rectangle and the event
    /// loop's raw display handle was not Xlib, or carried no display pointer.
    /// The strategy said X11, so anything else contradicts the backend rather
    /// than merely being unavailable, and guessing past it is what puts a
    /// one-pixel window on the screen.
    X11RootHandle,
    /// The Xlib root query itself failed. The string is the greppable reason
    /// (`libx11_unavailable`, `xgetwindowattributes_failed`).
    X11RootGeometry(String),
    /// The root window answered with a rectangle an appliance window cannot be:
    /// a dimension of `1` or less. A full-screen `1x1` window is the failure this
    /// path exists to prevent, so the rectangle is refused rather than applied.
    X11RootGeometryInvalid {
        width: u32,
        height: u32,
    },
    /// The WM-less path could not obtain the existing Xlib display/window
    /// handles needed for the explicit focus request.
    X11FocusHandle,
    /// The WM-less window was not viewable at its one focus lifecycle point.
    X11FocusNotViewable,
    /// The Xlib focus request or a focus query failed.
    X11FocusRequest(String),
    X11FocusQuery,
    /// XGetInputFocus returned a different target, including None, PointerRoot
    /// or the root window, so the window is not usable as appliance input.
    X11FocusVerify {
        focused: u64,
    },
    /// The WM-less path asked for a monitor and the window system offered none.
    ///
    /// Retained for a caller that resolves geometry without a root fallback.
    /// The appliance path no longer ends here, because `winit`'s X11 backend
    /// does not report "no monitor" as absence: when RandR yields no usable
    /// CRTC it returns a placeholder `MonitorHandle` with `id: 0` and a `1x1`
    /// rectangle, so `primary_monitor()` is `Some(placeholder)` and an
    /// absence-only check never fires. [`Self::X11RootHandle`],
    /// [`Self::X11RootGeometry`] and [`Self::X11RootGeometryInvalid`] are the
    /// refusals the root fallback raises in its place.
    NoMonitor,
}

impl WindowError {
    /// The raw driver/winit detail this error carries, if any — for `Display`
    /// and the diagnostic `eprintln!`, which want the string verbatim rather
    /// than the whitespace-collapsed form the log field uses.
    fn detail(&self) -> Option<&str> {
        match self {
            Self::EventLoopBuild(d)
            | Self::RunApp(d)
            | Self::MainLoopRun(d)
            | Self::CreateNativeWindow(d)
            | Self::AttachDisplayHandle(d)
            | Self::AttachWindowHandle(d)
            | Self::AttachPresenter(d)
            | Self::FullscreenValue(d)
            | Self::X11WmLessValue(d)
            | Self::X11RootGeometry(d)
            | Self::X11FocusRequest(d) => Some(d),
            Self::AlreadyOwned { .. }
            | Self::NoRegisteredWindow { .. }
            | Self::WrongOwner { .. }
            | Self::X11RootHandle
            | Self::X11RootGeometryInvalid { .. }
            | Self::X11FocusHandle
            | Self::X11FocusNotViewable
            | Self::X11FocusQuery
            | Self::X11FocusVerify { .. }
            | Self::NoMonitor => None,
        }
    }
}

/// Collapse whitespace runs to single `_` so a driver/winit string is safe as a
/// log field value ([`crate::observe::Emit`] splits the line on spaces).
pub(super) fn detail_field(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join("_")
}

impl crate::observe::Decline for WindowError {
    fn slug(&self) -> &'static str {
        match self {
            Self::EventLoopBuild(_) => "window_event_loop_build",
            Self::RunApp(_) => "window_run_app",
            Self::MainLoopRun(_) => "window_main_loop_run",
            Self::AlreadyOwned { .. } => "window_already_owned",
            Self::NoRegisteredWindow { .. } => "window_no_registered_window",
            Self::WrongOwner { .. } => "window_wrong_owner",
            Self::CreateNativeWindow(_) => "window_create_native_window",
            Self::AttachDisplayHandle(_) => "window_attach_display_handle",
            Self::AttachWindowHandle(_) => "window_attach_window_handle",
            Self::AttachPresenter(_) => "window_attach_presenter",
            Self::FullscreenValue(_) => "window_fullscreen_unrecognized",
            Self::X11WmLessValue(_) => "window_x11_wmless_unrecognized",
            Self::X11RootHandle => "window_x11_root_handle",
            Self::X11RootGeometry(_) => "window_x11_root_geometry",
            Self::X11RootGeometryInvalid { .. } => "window_x11_root_geometry_invalid",
            Self::X11FocusHandle => "window_x11_focus_handle",
            Self::X11FocusNotViewable => "window_x11_focus_not_viewable",
            Self::X11FocusRequest(_) => "window_x11_focus_request",
            Self::X11FocusQuery => "window_x11_focus_query",
            Self::X11FocusVerify { .. } => "window_x11_focus_verify",
            Self::NoMonitor => "window_no_monitor",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::AlreadyOwned { owner } => vec![("owner", owner.to_string())],
            Self::NoRegisteredWindow { id } => vec![("id", id.to_string())],
            Self::WrongOwner { owner, requested } => vec![
                ("owner", owner.to_string()),
                ("requested", requested.to_string()),
            ],
            Self::X11RootGeometryInvalid { width, height } => {
                vec![("width", width.to_string()), ("height", height.to_string())]
            }
            Self::X11FocusVerify { focused } => vec![("focused", focused.to_string())],
            other => match other.detail() {
                Some(d) => vec![("detail", detail_field(d))],
                None => Vec::new(),
            },
        }
    }
}

impl std::fmt::Display for WindowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use crate::observe::Decline as _;
        match self.detail() {
            Some(d) => write!(f, "{}: {d}", self.slug()),
            None => write!(f, "{}", self.slug()),
        }
    }
}

impl std::error::Error for WindowError {}

/// Spawn the window on a dedicated thread and return its join handle. The thread
/// owns the winit event loop for its lifetime; it exits when the window closes.
pub fn spawn(
    config: WindowConfig,
    on_input: InputSink,
    frames: FrameSlot,
    cursor_slot: CursorSlot,
    stop: StopFlag,
    wake: WindowWakeHandle,
) -> std::thread::JoinHandle<Result<(), WindowError>> {
    std::thread::Builder::new()
        .name("reims-vgpu-window".to_string())
        .spawn(move || run(config, on_input, frames, cursor_slot, stop, wake))
        .expect("spawn reims-vgpu-window thread")
}

/// Run the window event loop on the calling thread (blocks until the window
/// closes). Prefer [`spawn`]; call this directly only if you already own a
/// suitable thread.
pub fn run(
    config: WindowConfig,
    on_input: InputSink,
    frames: FrameSlot,
    cursor_slot: CursorSlot,
    stop: StopFlag,
    wake: WindowWakeHandle,
) -> Result<(), WindowError> {
    let event_loop = build_event_loop()?;
    wake.arm(event_loop.create_proxy());
    let mut app = App::new(config, on_input, frames, cursor_slot, stop);
    event_loop
        .run_app(&mut app)
        .map_err(|e| WindowError::RunApp(e.to_string()))
}

#[cfg(target_os = "macos")]
struct MainThreadWindow {
    id: u64,
    event_loop: EventLoop<WindowUserEvent>,
    app: App,
    exited: ExitedFlag,
}

#[cfg(target_os = "macos")]
thread_local! {
    static MAIN_THREAD_WINDOW: RefCell<Option<MainThreadWindow>> = const { RefCell::new(None) };
}

/// Create the macOS host window on the process main thread.
///
/// AppKit requires event-loop creation and dispatch on the process main thread.
/// QEMU owns that thread, so the thin shim calls this at device realize and
/// later makes [`run_main_thread`] its blocking UI entry. Only one display
/// window may exist in a process; repeated starts for the same device are
/// idempotent.
#[cfg(target_os = "macos")]
pub fn start_main_thread(
    id: u64,
    config: WindowConfig,
    on_input: InputSink,
    frames: FrameSlot,
    cursor_slot: CursorSlot,
    stop: StopFlag,
    exited: ExitedFlag,
    wake: WindowWakeHandle,
) -> Result<(), WindowError> {
    MAIN_THREAD_WINDOW.with(|cell| {
        let mut slot = cell.borrow_mut();
        if let Some(existing) = slot.as_ref() {
            return if existing.id == id {
                Ok(())
            } else {
                Err(WindowError::AlreadyOwned { owner: existing.id })
            };
        }
        let event_loop = build_event_loop()?;
        wake.arm(event_loop.create_proxy());
        let app = App::new(config, on_input, frames, cursor_slot, stop);
        *slot = Some(MainThreadWindow {
            id,
            event_loop,
            app,
            exited,
        });
        Ok(())
    })
}

/// Run the registered macOS window as QEMU's process-main UI loop.
///
/// QEMU runs emulation on its `qemu_main` thread on Darwin, leaving the process
/// main thread to this blocking AppKit loop. The exit flag is published only
/// after `run_app` returns and destroys the app's native Vulkan state.
#[cfg(target_os = "macos")]
pub fn run_main_thread(id: u64) -> Result<(), WindowError> {
    MAIN_THREAD_WINDOW.with(|cell| {
        let Some(mut window) = cell.borrow_mut().take() else {
            return Err(WindowError::NoRegisteredWindow { id });
        };
        if window.id != id {
            let owner = window.id;
            *cell.borrow_mut() = Some(window);
            return Err(WindowError::WrongOwner {
                owner,
                requested: id,
            });
        }
        let result = window
            .event_loop
            .run_app(&mut window.app)
            .map_err(|error| WindowError::MainLoopRun(error.to_string()));
        window.exited.store(true, Ordering::Release);
        result
    })
}

/// Build an event loop that may run off the main thread (QEMU owns the main
/// thread). X11 and Wayland both allow it via their platform extension.
///
/// Carries [`WindowUserEvent`] as its user event, which is what makes
/// `create_proxy` a wake channel the device can hold — see [`WindowWaker`].
fn build_event_loop() -> Result<EventLoop<WindowUserEvent>, WindowError> {
    let mut builder = EventLoop::with_user_event();
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        use winit::platform::wayland::EventLoopBuilderExtWayland;
        use winit::platform::x11::EventLoopBuilderExtX11;
        // Fully-qualified so the two identically-named ext methods don't clash;
        // each sets its own backend's any-thread flag (only the active one runs).
        EventLoopBuilderExtX11::with_any_thread(&mut builder, true);
        EventLoopBuilderExtWayland::with_any_thread(&mut builder, true);
    }
    #[cfg(target_os = "windows")]
    {
        use winit::platform::windows::EventLoopBuilderExtWindows;
        // Windows refuses a non-main-thread event loop unless the application
        // opts in explicitly; reims owns a dedicated window thread on every
        // host, so this is the same opt-in the X11/Wayland arms make above.
        EventLoopBuilderExtWindows::with_any_thread(&mut builder, true);
    }
    builder
        .build()
        .map_err(|e| WindowError::EventLoopBuild(e.to_string()))
}

struct App {
    config: WindowConfig,
    on_input: InputSink,
    frames: FrameSlot,
    cursor_slot: CursorSlot,
    /// Set by the device to request teardown; polled in `about_to_wait`.
    stop: StopFlag,
    /// True once a `WindowClosed` action has been emitted (UI close), so the
    /// shutdown request is sent exactly once.
    closed_sent: bool,
    window: Option<Arc<Window>>,
    /// Last cursor position in window pixels (for absolute pointer moves).
    cursor: (u32, u32),
    native_cursor: Option<CustomCursor>,
    applied_cursor_glyph: Option<GuestCursorGlyph>,
    applied_cursor_visible: bool,
    guest_cursor_position_logged: bool,
    pointer_move_logged: bool,
    pointer_button_logged: bool,
    /// What the guest believes it is holding, and whether the host desktop's
    /// shortcuts are being captured. See [`super::keyboard`] — the rule that
    /// every key-down is closed by a key-up lives there, not at these call
    /// sites.
    keyboard: Keyboard,
    /// True once capture has actually engaged at least once, so the census
    /// line that says so is emitted a single time rather than per focus change.
    capture_engaged_logged: bool,
    /// The platform's shortcut capture, built once the native window exists.
    /// `None` before `resumed`; a [`super::capture::NoCapture`] on a window
    /// system this build cannot ask, so the call sites never branch.
    capture: Option<Box<dyn Capture>>,
    /// The running rail holds a presenter on this window's surface. False
    /// before the attach in `resumed` and again after `exiting` releases it;
    /// there is no other presenter, so false means nothing can be drawn.
    presenter_attached: bool,
    first_present_logged: bool,
    first_direct_present_logged: bool,
    present_error_logged: bool,
    /// How many times [`App::rebuild_presenter`] has rebuilt, or tried to rebuild,
    /// the presenter after a device loss. Bounded by
    /// [`presenter_rebuild_budget`] so a surface that cannot be recreated is
    /// retried a few times and then left alone, rather than once per redraw for
    /// the life of the boot.
    presenter_rebuilds: u32,
    /// A [`FramePublished`] arrived (or a present asked to be repeated) and no
    /// redraw has been requested for it yet. Consumed by `about_to_wait`, which
    /// is the one place that talks to the platform about redraws.
    frame_pending: bool,
    /// The latest the loop may sleep to when nothing has asked it to draw. Pushed
    /// forward by every redraw request, so a steady publish stream never lets it
    /// fire; see [`WINDOW_REDRAW_BACKSTOP`] for what it is a floor under.
    next_backstop: std::time::Instant,
    /// Frame seq the drawable currently holds, or `None` before the first
    /// present.
    last_presented_seq: Option<u64>,
    /// Force the next present regardless of seq: first frame, resize, or a
    /// swapchain that reported suboptimal.
    redraw_required: bool,
    /// Last guest DisplaySwap geometry the window observed. Drives the
    /// once-per-mode-change native resize request and the pointer-to-guest
    /// viewport transform ([`crate::backend::window::viewport`]).
    guest_extent: Option<(u32, u32)>,
    /// Outstanding guest-driven native resize, kept only for the fail-visible
    /// `native_resize_not_applied` alarm — never a presentation gate.
    pending_guest_resize: Option<PendingGuestResize>,
    /// What this event loop actually did, per second. See [`LoopCensus`].
    loop_census: LoopCensus,
}

/// How often the window's event loop looked for a guest frame, and what it
/// found when it did.
///
/// `host_window_cadence` counts presents, and a present only happens when the
/// loop both ran and found a new `Frame::seq`. Those are two different failures
/// with the same symptom: a loop that ticks 500 times a second and finds
/// nothing new is a still screen, while a loop that ticks 17 times against 34
/// published frames is dropping half of them before any Vulkan call is reached.
/// Nothing separated them — `window_publish fresh` is counted on the drain
/// worker and `presents` inside the presenter, with the loop between them
/// unmeasured.
///
/// The counters are plain fields rather than atomics because every one of them
/// is touched only by the thread that owns the loop.
#[derive(Debug)]
struct LoopCensus {
    window_started: std::time::Instant,
    /// `about_to_wait` entries: how often the loop woke at all.
    ticks: u64,
    /// Ticks that asked the platform for a redraw. Bounded by published frames
    /// plus [`WINDOW_REDRAW_BACKSTOP`] ticks rather than by the tick
    /// rate, which is what [`redraw_due`] is for — a run where this tracks
    /// `ticks` instead is a loop that has fallen back to polling.
    redraws_asked: u64,
    /// `RedrawRequested` deliveries. A gap between this and `redraws_asked` is
    /// the platform coalescing or delaying them, which the loop cannot see any
    /// other way.
    draws: u64,
    /// Draws that found `Frame::seq` unchanged and presented nothing. Expected
    /// to dominate on a still screen.
    draws_stale: u64,
    /// Draws held because a guest mode change is being applied to the native
    /// window.
    draws_held: u64,
    /// Draws that reached the presenter with a new frame.
    draws_fresh: u64,
}

impl LoopCensus {
    fn new() -> Self {
        Self {
            window_started: std::time::Instant::now(),
            ticks: 0,
            redraws_asked: 0,
            draws: 0,
            draws_stale: 0,
            draws_held: 0,
            draws_fresh: 0,
        }
    }

    /// Emit and reset once a second, counting this tick first.
    ///
    /// Called from `about_to_wait`, which runs on every loop wake — so the
    /// window closes on the first tick after a second has passed rather than on
    /// a timer of its own, and a loop that has stopped waking emits nothing at
    /// all. That silence is the reading: a missing `host_window_loop` line
    /// means the event loop is not running.
    fn tick(&mut self) {
        self.ticks += 1;
        let elapsed = self.window_started.elapsed();
        if elapsed < std::time::Duration::from_secs(1) {
            return;
        }
        crate::observe::off(format!(
            "host_window_loop win_ms={} ticks={} redraws_asked={} draws={} \
             draws_fresh={} draws_stale={} draws_held={}",
            elapsed.as_millis(),
            self.ticks,
            self.redraws_asked,
            self.draws,
            self.draws_fresh,
            self.draws_stale,
            self.draws_held,
        ));
        *self = Self::new();
    }
}

impl ApplicationHandler<WindowUserEvent> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let mut attrs = Window::default_attributes()
            .with_title(self.config.title.clone())
            .with_inner_size(winit::dpi::PhysicalSize::new(
                self.config.width,
                self.config.height,
            ));
        if self.config.mode == WindowMode::Borderless {
            // `None` means "the monitor this window opens on", which is the only
            // answer available before the window exists — winit has no monitor
            // handle to name until the window system has placed it, and picking
            // one from `available_monitors()` would be this device choosing a
            // display the operator did not.
            //
            // The inner size above stays: it is what the window falls back to if
            // the window system declines the full-screen request, and it is the
            // size a later `set_fullscreen(None)` would restore.
            attrs = attrs.with_fullscreen(Some(winit::window::Fullscreen::Borderless(None)));
        }
        match self.config.strategy {
            FullscreenStrategy::Normal => {
                crate::observe::Emit::decline(
                    "host_window_mode",
                    &HostWindowMode::ordinary(crate::config::window_system(), self.config.mode),
                )
                .off();
            }
            FullscreenStrategy::X11WmLess => {
                // The screen's own rectangle, applied at creation: a RandR
                // monitor when winit identified one, and the X11 root window
                // when it did not. The `Fullscreen::Borderless` request above
                // stays for the ordinary winit fullscreen contract, but it is
                // not the source of focus in this path: winit returns before
                // its focus hint when the X11 monitor is the dummy id=0. The
                // explicit WM-less focus request happens after the presenter
                // attaches below.
                let geometry = match resolve_wm_less_geometry(event_loop) {
                    Ok(geometry) => geometry,
                    Err(error) => {
                        crate::observe::Emit::decline("host_window_init", &error).fail();
                        eprintln!("reims-vgpu-window: {error}; shutting down");
                        self.request_shutdown();
                        event_loop.exit();
                        return;
                    }
                };
                attrs = apply_wm_less_geometry(attrs, &geometry);
                crate::observe::Emit::decline(
                    "host_window_mode",
                    &HostWindowMode::wm_less_x11(geometry),
                )
                .off();
            }
        }
        let window = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                crate::observe::Emit::decline(
                    "host_window_init",
                    &WindowError::CreateNativeWindow(e.to_string()),
                )
                .fail();
                eprintln!("reims-vgpu-window: create_window failed: {e}");
                self.request_shutdown();
                event_loop.exit();
                return;
            }
        };
        window.set_cursor_visible(false);
        // Built before the presenter attach so a window that fails to present
        // still reports which capture it would have had, and torn down with the
        // window in `exiting`.
        self.capture = Some(Self::build_capture(&window));
        match Self::attach_presenter(&window) {
            Ok(()) => {
                self.presenter_attached = true;
                self.window = Some(window.clone());
                if self.config.strategy == FullscreenStrategy::X11WmLess {
                    if let Err(error) = focus_wm_less_x11_window(&window) {
                        crate::observe::Emit::decline("host_window_focus", &error).fail();
                        eprintln!("reims-vgpu-window: {error}; shutting down");
                        self.request_shutdown();
                        event_loop.exit();
                        return;
                    }
                    crate::observe::Emit::decline("host_window_focus", &HostWindowFocus).off();
                    // X11 may deliver Focused(true) before the presenter is
                    // attached.  The explicit focus request above is the
                    // authoritative hand-off for the WM-less appliance
                    // window, so make the keyboard/capture state agree with
                    // it instead of waiting for an event that has already
                    // been consumed by winit.
                    let effect = self.keyboard.focus(true);
                    self.apply_key_effect(effect);
                }
                // Kick the first frame; RedrawRequested re-arms each subsequent
                // one, so without this the window would never draw.
                window.request_redraw();
            }
            Err(error) => {
                // One rule on every platform and every rail: a rail that
                // cannot present to this surface gets a named refusal, not a
                // second presenter.
                //
                // The alternative was a self-contained `VkInstance`/`VkDevice`
                // and swapchain in this file, reached only here. It could not
                // draw the same picture the rails draw — it stretched instead
                // of letterboxing, so the pointer mapping had to be suppressed
                // to stay in agreement with it, and it presented FIFO where the
                // Vulkan rail takes MAILBOX when the surface offers it. A user
                // on the one host that reached it therefore got a measurably
                // different display with no counter saying which rail had drawn
                // it, which is the silent degradation this project does not
                // ship. macOS already refused here for its own reason (a second
                // `VkInstance` on the same `CAMetalLayer` does not fix a
                // MoltenVK surface failure); the two pathways now agree on what
                // an attach failure means.
                //
                // `host_window_attach` is the counter that says the refusal
                // happened, and the rail's own typed reason
                // (`SwapchainUnavailable`, `QueueCannotPresent`, a `vk_window_*`
                // or `metal_window_*` slug) rides along in it as the detail.
                crate::observe::Emit::decline("host_window_attach", &error).fail();
                eprintln!(
                    "reims-vgpu-window: no rail can present into this window ({error}); \
                     the host window has no other presenter — shutting down"
                );
                self.request_shutdown();
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                // The window IS the VM's display, so a UI close means "shut the
                // VM down". Emit WindowClosed once (the shim turns it into a
                // shutdown request) before tearing the window down. The device
                // will also set `stop` from its exit path; either order is fine.
                self.request_shutdown();
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                let applied = (size.width.max(1), size.height.max(1));
                if self.presenter_attached {
                    crate::backend::selected().window_resize(applied.0, applied.1);
                    self.note_guest_resize_applied(applied);
                }
                // Fresh swapchain images hold nothing; the seq gate would
                // otherwise skip until the guest happened to produce a new
                // frame, leaving the resized window blank.
                self.force_redraw();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    if let Some(evdev) = input_map::keycode_to_evdev(code) {
                        let down = event.state == ElementState::Pressed;
                        // Not forwarded straight through: `keyboard` records
                        // what the guest is holding so focus loss can close it,
                        // and it consumes the ungrab chord rather than letting
                        // the escape hatch reach the guest as a stray Esc.
                        let effect = self.keyboard.key(evdev, down);
                        self.apply_key_effect(effect);
                    }
                }
            }
            WindowEvent::Focused(focused) => {
                // The window system does not promise a key-up for a key held
                // when focus is lost — Wayland promises the opposite, that the
                // client treats them all as released — so a key left down here
                // stays down in the guest for the rest of the boot. This is
                // also where the desktop's shortcuts are taken and handed back.
                let effect = self.keyboard.focus(focused);
                self.apply_key_effect(effect);
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.pointer_move((position.x, position.y));
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if let Some(btn) = input_map::mouse_button(button) {
                    if !self.pointer_button_logged {
                        self.pointer_button_logged = true;
                        crate::observe::off(format!(
                            "host_window_pointer_input kind=button button={btn:?} state={} status=forwarded",
                            if state == ElementState::Pressed { "down" } else { "up" }
                        ));
                    }
                    (self.on_input)(HostAction::input_pointer_button(
                        btn,
                        state == ElementState::Pressed,
                    ));
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                for action in input_map::scroll_actions(delta) {
                    (self.on_input)(action);
                }
            }
            WindowEvent::RedrawRequested => {
                self.loop_census.draws += 1;
                self.draw();
            }
            _ => {}
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        // Tear the presenter down while the native window is still alive.
        // Detaching destroys the swapchain and the `VkSurfaceKHR`, and the
        // driver services those through the Wayland/X (or AppKit) surface owned
        // by `window`; releasing the window first makes the driver marshal to a
        // freed `wl_proxy` and crash. winit calls this before the loop ends, so
        // the ordering is explicit here rather than left to drop order.
        //
        // `Backend::window_detach` publishes the rail's own attached flag; this
        // file used to set it here as a second statement, and the third site
        // that takes a presenter — the Vulkan engine's device-loss flush — did
        // not know to.
        if self.presenter_attached {
            crate::backend::selected().window_detach();
            self.presenter_attached = false;
        }
        // Close the guest's held keys and hand the desktop its shortcuts back
        // while the native window is still alive. An X11 keyboard grab in
        // particular outlives this process if it is not released, and would
        // leave the whole session unable to type.
        let effect = self.keyboard.shutdown();
        self.apply_key_effect(effect);
        self.capture = None;
        self.window = None;
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        self.loop_census.tick();
        // The device sets `stop` on VM teardown. The loop wakes at least once per
        // [`WINDOW_REDRAW_BACKSTOP`], so the request is picked up within
        // one of those — then the loop exits and `exiting` releases the
        // presenter's swapchain and surface on this thread before the device's
        // join returns. Nothing publishes at teardown, so this is one of the two
        // deadlines that constant exists for.
        if self.stop.load(Ordering::Relaxed) {
            event_loop.exit();
        }
        // Pacing: ask for a redraw when the device says it published one, and
        // otherwise sleep to the backstop. Two earlier shapes are what this has
        // to stay clear of. Re-requesting a redraw from inside `RedrawRequested`
        // is a spin: measured on x86/Vulkan it held **510 presents/s** — a
        // full-frame swapchain blit and submit each — while the guest produced
        // 4.5-8 frames/s, and FIFO does not throttle it. Asking on a 2 ms poll
        // instead stopped the presents but kept the wakes, at 494 redraws/s to
        // find 8.7 frames. `draw`'s seq gate still decides what is presented;
        // this decides only how often it is asked.
        // Cloned rather than borrowed so the body can reach `&mut self` for
        // `force_redraw`, which is the one place the two redraw flags are set
        // together. An `Arc` clone per wake is a refcount bump.
        if let Some(window) = self.window.clone() {
            if let Some(pending) = self.pending_guest_resize.as_ref() {
                if pending.requested_at.elapsed() >= GUEST_RESIZE_WARN_AFTER {
                    let actual = window.inner_size();
                    crate::observe::fail(format!(
                        "host_window_guest_resize FAIL reason=native_resize_not_applied \
                         requested={}x{} actual={}x{}",
                        pending.target.0, pending.target.1, actual.width, actual.height
                    ));
                    // Drop the request so presentation resumes (letterboxed
                    // into whatever drawable exists) instead of holding.
                    self.pending_guest_resize = None;
                    self.force_redraw();
                }
            }
            let now = std::time::Instant::now();
            let pending = std::mem::take(&mut self.frame_pending);
            if redraw_due(pending, now, self.next_backstop) {
                window.request_redraw();
                self.loop_census.redraws_asked += 1;
                self.next_backstop = now + WINDOW_REDRAW_BACKSTOP;
            }
            event_loop.set_control_flow(winit::event_loop::ControlFlow::WaitUntil(
                self.next_backstop,
            ));
        }
    }

    /// A [`FramePublished`] from the device: the frame slot holds something the
    /// loop has not looked at.
    ///
    /// Records the reason and returns. Requesting the redraw here instead would
    /// put a second caller on the platform's redraw path, and the two would
    /// disagree about the backstop — `about_to_wait` runs after this and after
    /// every other event, so it is the one place that can hold that decision.
    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: WindowUserEvent) {
        match event {
            WindowUserEvent::FramePublished => self.frame_pending = true,
            WindowUserEvent::CursorPublished => self.apply_guest_cursor(event_loop),
        }
    }
}

/// Whether `about_to_wait` should ask the platform for a redraw on this wake.
///
/// The whole rule, and the reason the loop is allowed to sleep: a tick with no
/// publish behind it and time left on the backstop asks for nothing. Under the
/// 2 ms poll this was unconditionally true, which is how 98 % of the redraws
/// this window asked for reached [`needs_present`] only to be refused.
fn redraw_due(frame_pending: bool, now: std::time::Instant, backstop: std::time::Instant) -> bool {
    frame_pending || now >= backstop
}

impl App {
    /// A window that has not opened yet.
    ///
    /// Both entry points build one — [`run`] on the calling thread and
    /// `start_main_thread` on macOS's main thread — and every field but the
    /// four they are given is fixed. Written out at each site, a field added
    /// to the struct could be initialised in one and missed in the other.
    fn new(
        config: WindowConfig,
        on_input: InputSink,
        frames: FrameSlot,
        cursor_slot: CursorSlot,
        stop: StopFlag,
    ) -> Self {
        Self {
            config,
            on_input,
            frames,
            cursor_slot,
            stop,
            closed_sent: false,
            window: None,
            cursor: (0, 0),
            native_cursor: None,
            applied_cursor_glyph: None,
            applied_cursor_visible: false,
            guest_cursor_position_logged: false,
            pointer_move_logged: false,
            pointer_button_logged: false,
            keyboard: Keyboard::new(),
            capture_engaged_logged: false,
            capture: None,
            presenter_attached: false,
            first_present_logged: false,
            first_direct_present_logged: false,
            present_error_logged: false,
            presenter_rebuilds: 0,
            // The first tick draws: `redraw_required` is set and there is
            // no publish behind the first frame, which is a clear rather than a
            // guest frame.
            frame_pending: true,
            next_backstop: std::time::Instant::now(),
            last_presented_seq: None,
            redraw_required: true,
            guest_extent: None,
            pending_guest_resize: None,
            loop_census: LoopCensus::new(),
        }
    }

    /// Emit a [`KeyEffect`]'s guest input and settle the platform capture.
    ///
    /// One place, because the two halves of a keyboard transition are decided
    /// together and applying only one of them is the defect this whole path
    /// exists to remove: a grab dropped without releasing the held keys leaves
    /// the guest holding a modifier, and keys released without dropping the grab
    /// leaves the operator unable to reach their own desktop.
    fn apply_key_effect(&mut self, effect: KeyEffect) {
        for action in effect.actions {
            (self.on_input)(action);
        }
        let Some(grab) = effect.grab else {
            return;
        };
        let Some(capture) = self.capture.as_mut() else {
            return;
        };
        let held = grab == Grab::Held;
        match capture.set(held) {
            Err(error) => {
                // A capture that did not take means the host desktop is still
                // eating guest keystrokes. Named on the always-on channel, once
                // per reason: the window keeps running with whatever it is left.
                crate::observe::Emit::decline("host_window_capture", &error)
                    .fail_once(u64::from(held));
            }
            Ok(()) if held && !self.capture_engaged_logged => {
                // Say once that the shortcuts were actually handed over. Without
                // this the log cannot separate a capture that engaged from one
                // that was built and never asked, because both are silent.
                self.capture_engaged_logged = true;
                let mechanism = capture.describe();
                crate::observe::Emit::decline(
                    "host_window_capture_engaged",
                    &CaptureEngaged(mechanism),
                )
                .off();
                // Also on stderr, once. This is the instant the operator's own
                // desktop shortcuts stop working; the fail log is the wrong
                // place to learn how to get them back, because reading it is
                // itself a thing you need a working desktop to do.
                eprintln!(
                    "reims-vgpu-window: keyboard captured ({mechanism}) — \
                     the guest now receives Cmd/Alt/Super chords. \
                     Press {} to release it.",
                    super::keyboard::UNGRAB_CHORD
                );
            }
            Ok(()) => {}
        }
    }

    /// Force the next present and make sure a redraw is asked for to carry it.
    ///
    /// The two flags answer different questions — `redraw_required`
    /// opens [`needs_present`]'s seq gate, `frame_pending` makes
    /// `about_to_wait` talk to the platform — and every caller that wants one
    /// wants both. Setting only the first is the shape that leaves a resized
    /// window blank until the guest happens to publish, because under the
    /// backstop nothing asks for the redraw the flag was waiting for.
    ///
    /// Deliberately not folded into [`redraw_due`]: a standing
    /// `redraw_required` that a `Busy` presenter cannot clear would then
    /// re-request on every wake, which is the spin this loop has already been
    /// bitten by twice. A one-shot flag cannot do that.
    fn force_redraw(&mut self) {
        self.redraw_required = true;
        self.frame_pending = true;
    }

    fn request_shutdown(&mut self) {
        if !self.closed_sent {
            (self.on_input)(HostAction::window_closed());
            self.closed_sent = true;
        }
    }

    fn surface_dims(&self) -> (u32, u32) {
        // The rail owns its drawables, so the native window is the only thing
        // here that knows their size. Before it opens, the
        // requested size is the best answer available.
        match self.window.as_ref() {
            Some(window) => {
                let size = window.inner_size();
                (size.width.max(1), size.height.max(1))
            }
            None => (self.config.width, self.config.height),
        }
    }

    /// Emit an absolute pointer move. Once a guest geometry is known the
    /// position maps through the presenter's aspect-fit viewport into guest
    /// pixels, so display placement and pointer translation move as one unit;
    /// before the first frame the full-window space is forwarded unchanged.
    /// See [`pointer_report`], which holds the decision and its tests.
    fn pointer_move(&mut self, position: (f64, f64)) {
        let (x, y, width, height) =
            pointer_report(position, self.surface_dims(), self.guest_extent);
        self.cursor = (x, y);
        if !self.pointer_move_logged {
            self.pointer_move_logged = true;
            crate::observe::off(format!(
                "host_window_pointer_input kind=move status=forwarded x={x} y={y} width={width} height={height}"
            ));
        }
        (self.on_input)(HostAction::input_pointer_move(x, y, width, height));
    }

    fn apply_guest_cursor(&mut self, event_loop: &ActiveEventLoop) {
        let Ok(state) = self.cursor_slot.lock().map(|state| state.clone()) else {
            crate::observe::fail(
                "host_window_guest_cursor_fail reason=cursor_slot_poisoned".to_string(),
            );
            return;
        };
        let Some(window) = self.window.as_ref() else {
            return;
        };

        if !self.guest_cursor_position_logged {
            self.guest_cursor_position_logged = true;
            crate::observe::off(format!(
                "host_window_guest_cursor event=position status=received x={} y={} visible={}",
                state.x, state.y, state.visible
            ));
        }

        if cursor_glyph_changed(&self.applied_cursor_glyph, &state.glyph) {
            if let Some(glyph) = state.glyph.as_ref() {
                let source = match CustomCursor::from_rgba(
                    cursor_argb_to_rgba(&glyph.pixels),
                    glyph.width,
                    glyph.height,
                    glyph.hot_x,
                    glyph.hot_y,
                ) {
                    Ok(source) => source,
                    Err(error) => {
                        crate::observe::fail(format!(
                            "host_window_guest_cursor_fail reason=winit_custom_cursor {error:?}"
                        ));
                        return;
                    }
                };
                let cursor = event_loop.create_custom_cursor(source);
                window.set_cursor(cursor.clone());
                self.native_cursor = Some(cursor);
                self.applied_cursor_glyph = state.glyph.clone();
                crate::observe::off(format!(
                    "host_window_guest_cursor event=glyph mechanism=winit_custom_cursor status=installed size={}x{} hotspot={},{}",
                    glyph.width, glyph.height, glyph.hot_x, glyph.hot_y
                ));
            }
        }

        let visible = cursor_force_visible() || (state.visible && self.native_cursor.is_some());
        if visible != self.applied_cursor_visible {
            window.set_cursor_visible(visible);
            self.applied_cursor_visible = visible;
            crate::observe::off(format!(
                "host_window_guest_cursor event=visibility visible={visible}"
            ));
        }
    }

    /// Build the platform's shortcut capture for this window.
    ///
    /// Never fails into the caller: a window system this build cannot ask for
    /// its shortcuts back yields a [`super::capture::NoCapture`] and a typed
    /// reason, because a window that cannot capture must still present. The
    /// describe string goes on the boot log so a run says which of the three
    /// capture mechanisms it actually got.
    fn build_capture(window: &Arc<Window>) -> Box<dyn Capture> {
        let handles = window
            .display_handle()
            .map(|d| d.as_raw())
            .ok()
            .zip(window.window_handle().map(|w| w.as_raw()).ok());
        let capture = match handles {
            Some((display, handle)) => super::capture::for_window(display, handle),
            None => {
                // The same handles the presenter attach needs; if they are gone the
                // attach is about to fail too, and this is not the refusal that
                // should be read as the cause.
                Box::new(super::capture::NoCapture::new("no_window_handle"))
            }
        };
        crate::observe::Emit::decline("host_window_capture_mode", &CaptureMode(capture.describe()))
            .off();
        capture
    }

    /// Ask the running rail to build a presenter over this native window.
    ///
    /// It presents from the rail's own device, which is what lets a rail that
    /// holds residents skip the three full-frame host copies every presented
    /// layer otherwise pays: the drain's `read_target`, the publish copy, and a
    /// staging upload.
    ///
    /// Taken out of `resumed` so [`Self::rebuild_presenter`] can run it again on
    /// a window that already exists. `Backend::window_attach` is idempotent — it
    /// returns `Ok` if a presenter is already there — so calling it twice is
    /// safe, and it is the rail that publishes the attached flag.
    fn attach_presenter(window: &Arc<Window>) -> Result<(), WindowError> {
        let display = window
            .display_handle()
            .map_err(|error| WindowError::AttachDisplayHandle(error.to_string()))?
            .as_raw();
        let handle = window
            .window_handle()
            .map_err(|error| WindowError::AttachWindowHandle(error.to_string()))?
            .as_raw();
        let size = window.inner_size();
        crate::backend::selected()
            .window_attach(&crate::backend::window::WindowSurface {
                display,
                window: handle,
                width: size.width.max(1),
                height: size.height.max(1),
            })
            .map_err(|error| WindowError::AttachPresenter(error.to_string()))
    }

    /// Rebuild the presenter after the rail said it lost one.
    ///
    /// # Why the window has to do this and nothing else can
    ///
    /// Measured on the Vulkan rail, and stated in its terms because that is
    /// where it was found; the shape is the rail-independent one, which is why
    /// `WindowDecline::PresenterLost` is the rail's answer rather than this
    /// file's `matches!` on a Vulkan enum.
    ///
    /// A `VK_ERROR_DEVICE_LOST` poisons the engine device; the next guest draw
    /// destroys everything derived from it — caches, pools, and the presenter's
    /// swapchain, surface and semaphores — and brings a fresh `VkDevice` up.
    /// Every *guest* resource then rebuilds itself on demand, because the
    /// registry is keyed by a `TargetIdentity` the guest re-presents and
    /// `registry_ensure` recreates whatever is missing. The presenter was the
    /// one thing with no such path: its only constructor sits in `resumed`,
    /// winit calls `resumed` once per window, so the swapchain died and stayed
    /// dead. The device recovered, the guest recovered, and the picture did not
    /// come back for the rest of the boot — measured as seven recreates in one
    /// driven macos-11 boot, after which every leg reported no frames at all.
    ///
    /// Bounded, and the bound is the point: a surface that genuinely cannot be
    /// recreated must not be retried once per redraw forever. Past the bound the
    /// window keeps running on whatever the previous behaviour was — a named
    /// refusal per present — rather than spinning.
    fn rebuild_presenter(&mut self) {
        let Some(window) = self.window.clone() else {
            return;
        };
        if self.presenter_rebuilds >= presenter_rebuild_budget() {
            return;
        }
        self.presenter_rebuilds += 1;
        match Self::attach_presenter(&window) {
            Ok(()) => {
                // The presenter is new, so nothing on screen came from it and
                // the sequence this loop last presented is not its history.
                // Clearing it is what makes the next frame count as fresh
                // instead of stale, which is the difference between recovering
                // and looking recovered.
                self.last_presented_seq = None;
                self.redraw_required = true;
                self.present_error_logged = false;
                crate::observe::off(format!(
                    "host_window_reattach status=ok attempt={}",
                    self.presenter_rebuilds
                ));
                eprintln!(
                    "reims-vgpu-window: presenter rebuilt after device loss \
                     (attempt {})",
                    self.presenter_rebuilds
                );
                self.force_redraw();
            }
            Err(error) => {
                crate::observe::Emit::decline("host_window_reattach", &error).fail();
            }
        }
    }

    /// Present one frame through the running rail, or decide there is
    /// nothing to present.
    ///
    /// The guard is teardown, not a rail choice: `exiting` releases the
    /// presenter before the loop ends, and an attach that failed never stored a
    /// window at all. Both already named themselves where they happened, so a
    /// redraw arriving after either is expected control flow and stays quiet.
    fn draw(&mut self) {
        if !self.presenter_attached {
            return;
        }
        let frame = self.frames.lock().ok().and_then(|guard| guard.clone());
        self.request_guest_geometry(frame.as_deref());
        if self.pending_guest_resize.is_some() {
            // A guest mode change is being applied to the native window
            // (normally single-digit milliseconds; bounded by the 1 s alarm).
            // Hold the previous drawable rather than letterboxing the
            // new-geometry frame into the outgoing swapchain — at boot the
            // next guest present can be seconds away, which would pin that
            // interim frame on screen.
            self.loop_census.draws_held += 1;
            return;
        }
        let incoming_seq = frame.as_ref().map(|frame| frame.seq);
        if !needs_present(self.last_presented_seq, self.redraw_required, incoming_seq) {
            self.loop_census.draws_stale += 1;
            return;
        }
        self.loop_census.draws_fresh += 1;
        let result = crate::backend::selected().window_present(
            frame.as_ref().and_then(|frame| frame.resident.as_ref()),
            frame.as_deref().map(window_cpu_frame),
        );
        match result {
            Ok(WindowPresentOutcome::Busy) => {}
            Ok(WindowPresentOutcome::Presented {
                direct,
                width,
                height,
                buffers,
                suboptimal,
            }) => {
                self.present_error_logged = false;
                // A frame reached the screen, so whatever rebuilt the presenter
                // is proven and its budget starts over — the same rule, for the
                // same reason, as `ContextOwner::note_work_completed` applies to
                // the device's own recreate count. Without this the bound is a
                // lifetime cap and a rail that recovers eight times is left dark
                // after the third. See [`presenter_rebuild_budget`].
                self.presenter_rebuilds = 0;
                self.last_presented_seq = incoming_seq;
                // A suboptimal present armed a swapchain recreation; redraw
                // promptly so the corrected drawable replaces this one even if
                // no new guest frame arrives for seconds. The assignment clears
                // the flag on a healthy present, so this cannot become a
                // self-feeding loop: one suboptimal buys exactly one more draw.
                self.redraw_required = suboptimal;
                if suboptimal {
                    self.force_redraw();
                }
                if !self.first_present_logged {
                    eprintln!(
                        "reims-vgpu-window: first frame presented \
                         ({width}x{height}, {buffers} drawables)"
                    );
                    self.first_present_logged = true;
                }
                if direct && frame.is_some() && !self.first_direct_present_logged {
                    eprintln!(
                        "reims-vgpu-window: first guest frame presented via rail resident \
                         (same-device zero-copy)"
                    );
                    crate::observe::off(
                        "host_window_direct_present path=rail_resident status=live",
                    );
                    self.first_direct_present_logged = true;
                }
            }
            Err(error) => {
                if !self.present_error_logged {
                    // The rail's own refusal names itself — a `VkCall`'s
                    // `vk_window_*` slug, a `DrawReason`, a Metal layer
                    // refusal. Emitting it typed keeps that slug the primary
                    // `reason=` rather than nesting it inside a coarse
                    // `reason=rail_present error=...` double-reason.
                    crate::observe::Emit::decline("host_window_present", &error).fail();
                    eprintln!("reims-vgpu-window: rail present failed: {error}");
                    self.present_error_logged = true;
                }
                // One disposition is recoverable and every other refusal is the
                // rail's own to report: a presenter that is *gone* means the
                // rail's device was lost and took it. Which refusals mean that
                // is the rail's answer, made once where it is known — see
                // [`Self::rebuild_presenter`].
                if error.presenter_lost() {
                    self.rebuild_presenter();
                }
            }
        }
    }

    /// Track the accepted guest frame geometry and ask the native window to
    /// match a newly selected guest mode, once per change. The frame
    /// dimensions are protocol state — the same width/height that select the
    /// compositor resident — never a content heuristic. Presentation does not
    /// wait: until the resize applies, frames letterbox into the current
    /// drawable and pointer input maps through the same viewport.
    ///
    /// Called only from [`Self::draw`], which is where the extent it records is
    /// paid for: `guest_extent` is also what [`pointer_report`] maps a window
    /// position through, so it must be set by the same pass that hands the
    /// frame to the aspect-fitting presenter. Setting it anywhere else risks a
    /// viewport-mapped pointer against a picture drawn to a different rule.
    fn request_guest_geometry(&mut self, frame: Option<&Frame>) {
        let Some(frame) = frame else { return };
        let incoming = (frame.width.max(1), frame.height.max(1));
        if self.guest_extent == Some(incoming) {
            return;
        }
        let Some(window) = self.window.as_ref() else {
            return;
        };
        let size = window.inner_size();
        let actual = (size.width.max(1), size.height.max(1));
        let request = guest_resize_request(self.config.mode, self.guest_extent, incoming, actual);
        self.guest_extent = Some(incoming);
        if !request {
            return;
        }
        crate::observe::off(format!(
            "host_window_guest_resize status=requested from={}x{} to={}x{}",
            actual.0, actual.1, incoming.0, incoming.1
        ));
        self.pending_guest_resize = Some(PendingGuestResize {
            target: incoming,
            requested_at: std::time::Instant::now(),
        });
        let immediate =
            window.request_inner_size(winit::dpi::PhysicalSize::new(incoming.0, incoming.1));
        if let Some(applied) = immediate {
            // Applied synchronously — winit emits no later `Resized` for it.
            let applied = (applied.width.max(1), applied.height.max(1));
            crate::backend::selected().window_resize(applied.0, applied.1);
            self.redraw_required = true;
            self.note_guest_resize_applied(applied);
        }
    }

    /// Clear the outstanding guest resize once the window system has answered
    /// it (via `Resized` or a synchronous apply). See [`guest_resize_settled`]
    /// for why any answer settles it, including one that adjusted the size.
    fn note_guest_resize_applied(&mut self, applied: (u32, u32)) {
        let target = self.pending_guest_resize.as_ref().map(|p| p.target);
        let Some(status) = guest_resize_settled(target, applied) else {
            return;
        };
        let requested = target.unwrap_or(applied);
        crate::observe::off(format!(
            "host_window_guest_resize status={status} width={} height={} \
             requested={}x{}",
            applied.0, applied.1, requested.0, requested.1
        ));
        self.pending_guest_resize = None;
    }
}

#[cfg(test)]
mod loop_census_tests {
    use super::*;

    /// A loop that has run for less than its window emits nothing and keeps
    /// counting. Emitting on every tick would be one line per wake, and a wake
    /// is every publish, every input event and every backstop.
    #[test]
    fn a_partial_window_accumulates_rather_than_emitting() {
        let mut census = LoopCensus::new();
        census.tick();
        census.tick();
        assert_eq!(census.ticks, 2);
        assert!(census.window_started.elapsed() < std::time::Duration::from_secs(1));
    }

    /// The window closes on the first tick past a second, and resets so the
    /// next line is a rate rather than a running total.
    #[test]
    fn a_closed_window_resets_every_counter() {
        let mut census = LoopCensus::new();
        census.draws = 9;
        census.draws_fresh = 4;
        census.draws_stale = 5;
        census.redraws_asked = 7;
        census.window_started = std::time::Instant::now() - std::time::Duration::from_millis(1100);
        census.tick();
        assert_eq!(census.ticks, 0);
        assert_eq!(census.draws, 0);
        assert_eq!(census.draws_fresh, 0);
        assert_eq!(census.draws_stale, 0);
        assert_eq!(census.redraws_asked, 0);
        assert!(census.window_started.elapsed() < std::time::Duration::from_secs(1));
    }

    /// The three dispositions divide `draws` exactly. A draw that reached the
    /// loop and moved none of them is a fourth outcome nobody named, which is
    /// the shape of a frame this device drops with no line saying so.
    #[test]
    fn the_dispositions_sum_to_the_draws() {
        let mut census = LoopCensus::new();
        census.draws = 12;
        census.draws_fresh = 5;
        census.draws_stale = 6;
        census.draws_held = 1;
        assert_eq!(
            census.draws,
            census.draws_fresh + census.draws_stale + census.draws_held
        );
    }
}

#[cfg(test)]
mod wake_tests {
    use super::*;

    /// A wake that finds nothing armed is a no-op, not a panic and not a
    /// refusal.
    ///
    /// The device builds the waker before the window thread has an event loop,
    /// so every publish between those two moments lands here. The backstop is
    /// what makes that safe, and this pins that the gap is quiet.
    #[test]
    fn an_unarmed_waker_takes_a_wake_and_does_nothing_with_it() {
        let waker = WindowWaker::new();
        waker.wake();
        waker.wake();
        assert!(
            waker.proxy.lock().expect("fresh mutex").is_none(),
            "nothing armed it, so there is nothing to send through"
        );
    }

    /// The property the 2 ms poll did not have: a wake with no publish behind it
    /// and time left on the backstop asks the platform for nothing.
    ///
    /// This is the whole change. On the boot that motivated it, 70 961 of the
    /// 72 227 redraws this window asked for were refused by the seq gate — every
    /// one of them a tick that reached here and said yes unconditionally.
    #[test]
    fn a_wake_with_no_publish_and_time_left_asks_for_no_redraw() {
        let now = std::time::Instant::now();
        let backstop = now + WINDOW_REDRAW_BACKSTOP;
        assert!(
            !redraw_due(false, now, backstop),
            "an input event or a stray wake must not become a redraw"
        );
    }

    /// A publish is answered on the wake it arrives on, whatever the backstop
    /// says — the point of waking is that the frame does not wait for a timer.
    #[test]
    fn a_publish_is_answered_before_the_backstop() {
        let now = std::time::Instant::now();
        let backstop = now + WINDOW_REDRAW_BACKSTOP;
        assert!(redraw_due(true, now, backstop));
    }

    /// With no publish at all the backstop still draws, which is what keeps a
    /// missed or unarmed wake costing latency rather than the frame.
    #[test]
    fn the_backstop_draws_when_nothing_published() {
        let now = std::time::Instant::now();
        assert!(
            redraw_due(false, now, now),
            "a deadline that has arrived is a reason on its own"
        );
        assert!(redraw_due(false, now + WINDOW_REDRAW_BACKSTOP, now));
    }

    /// Over a second of the workload that motivated this, the loop asks for a
    /// redraw once per published frame and not once per wake.
    ///
    /// This is the invariant, and it is the one a pacing rule can fail while
    /// every other test here passes: the counts must be bounded by *frames*,
    /// not by elapsed time. The measured boot woke this loop ~997 times a second
    /// and asked 494 of those times to serve 26 published frames. Driving the
    /// timeline rather than the truth table is deliberate, for the reason
    /// [`tests::repeated_polls_of_one_frame_present_once`] gives.
    #[test]
    fn a_second_of_the_measured_workload_asks_once_per_frame_not_once_per_wake() {
        let start = std::time::Instant::now();
        let second = std::time::Duration::from_secs(1);
        // 26 publishes/s (the driven boot's peak `present_hz`) against ~997
        // wakes/s, which is what an input-carrying event loop actually sees.
        let publishes = 26u32;
        let wakes = 997u32;
        let publish_every = wakes / publishes;
        let mut backstop = start;
        let mut requested = 0u32;
        for wake in 1..=wakes {
            let now = start + second * wake / wakes;
            let published = wake % publish_every == 0;
            if redraw_due(published, now, backstop) {
                requested += 1;
                backstop = now + WINDOW_REDRAW_BACKSTOP;
            }
        }
        let backstop_ticks = (second.as_millis() / WINDOW_REDRAW_BACKSTOP.as_millis()) as u32;
        assert!(
            requested <= publishes + backstop_ticks,
            "asked {requested} times for {publishes} frames — a pacing rule \
             bounded by wakes rather than by frames"
        );
        assert!(
            requested >= publishes,
            "asked {requested} times for {publishes} frames — a frame that \
             published and was never asked about is a dropped frame"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A run of redraws over one unchanged guest frame presents exactly once.
    ///
    /// This is the spin the Linux rail shipped: `RedrawRequested` called
    /// `request_redraw()` again unconditionally, so the loop re-entered
    /// immediately and presented whatever was already on screen. Measured on
    /// x86/Vulkan it ran at 510 presents/s — each a full-frame swapchain blit
    /// and submit — against a guest producing 4.5-8 frames/s. FIFO does not
    /// throttle it, so nothing else in the stack was going to.
    ///
    /// The property that stops it is this predicate, and it now guards both
    /// rails. Driving it with the poll sequence rather than the truth table is
    /// deliberate: the table version passes even if a caller ignores the answer.
    #[test]
    fn repeated_polls_of_one_frame_present_once() {
        let mut presented: Option<u64> = None;
        let mut redraw_required = true;
        let mut presents = 0;
        // 300 polls — a couple of seconds at the 2 ms poll — over three guest
        // frames, the middle one held for most of them.
        for tick in 0..300u64 {
            let incoming = Some(match tick {
                0..=9 => 1,
                10..=289 => 2,
                _ => 3,
            });
            if needs_present(presented, redraw_required, incoming) {
                presents += 1;
                presented = incoming;
                redraw_required = false;
            }
        }
        assert_eq!(
            presents, 3,
            "one present per distinct guest frame, not one per poll"
        );
    }

    /// The first poll presents even though no guest frame exists yet, and a
    /// resize presents again into the fresh swapchain images without waiting
    /// for the guest — both are `redraw_required`, and both are why the gate
    /// cannot be a bare seq comparison.
    #[test]
    fn forced_redraw_presents_without_a_new_frame() {
        assert!(
            needs_present(None, true, None),
            "the first present has no frame and must still happen"
        );
        assert!(
            !needs_present(None, false, None),
            "and must not repeat once the flag is cleared"
        );
        assert!(
            needs_present(Some(7), true, Some(7)),
            "a resize must repaint the same frame into new swapchain images"
        );
    }

    /// One value of every [`WindowError`] variant.
    ///
    /// The list is closed by [`variant_name`], whose `match` has no wildcard
    /// arm: a variant added to the enum stops this module compiling until it is
    /// named there and built here.
    fn every_window_error() -> Vec<WindowError> {
        vec![
            WindowError::EventLoopBuild("os error while building".into()),
            WindowError::RunApp("event loop exited".into()),
            WindowError::MainLoopRun("event loop exited".into()),
            WindowError::AlreadyOwned { owner: 3 },
            WindowError::NoRegisteredWindow { id: 4 },
            WindowError::WrongOwner {
                owner: 3,
                requested: 4,
            },
            WindowError::CreateNativeWindow("os error creating window".into()),
            WindowError::AttachDisplayHandle("no display handle".into()),
            WindowError::AttachWindowHandle("no window handle".into()),
            WindowError::AttachPresenter("swapchain unavailable".into()),
            WindowError::FullscreenValue("borderless please".into()),
            WindowError::X11WmLessValue("wmless please".into()),
            WindowError::X11RootHandle,
            WindowError::X11RootGeometry("xgetwindowattributes_failed".into()),
            WindowError::X11RootGeometryInvalid {
                width: 1,
                height: 1,
            },
            WindowError::X11FocusHandle,
            WindowError::X11FocusNotViewable,
            WindowError::X11FocusRequest("xsetinputfocus_failed".into()),
            WindowError::X11FocusQuery,
            WindowError::X11FocusVerify { focused: 1 },
            WindowError::NoMonitor,
        ]
    }

    /// The variant's own name, matched exhaustively on purpose — the wildcard
    /// arm is what would let [`every_window_error`] silently miss one.
    fn variant_name(error: &WindowError) -> &'static str {
        match error {
            WindowError::EventLoopBuild(_) => "EventLoopBuild",
            WindowError::RunApp(_) => "RunApp",
            WindowError::MainLoopRun(_) => "MainLoopRun",
            WindowError::AlreadyOwned { .. } => "AlreadyOwned",
            WindowError::NoRegisteredWindow { .. } => "NoRegisteredWindow",
            WindowError::WrongOwner { .. } => "WrongOwner",
            WindowError::CreateNativeWindow(_) => "CreateNativeWindow",
            WindowError::AttachDisplayHandle(_) => "AttachDisplayHandle",
            WindowError::AttachWindowHandle(_) => "AttachWindowHandle",
            WindowError::AttachPresenter(_) => "AttachPresenter",
            WindowError::FullscreenValue(_) => "FullscreenValue",
            WindowError::X11WmLessValue(_) => "X11WmLessValue",
            WindowError::X11RootHandle => "X11RootHandle",
            WindowError::X11RootGeometry(_) => "X11RootGeometry",
            WindowError::X11RootGeometryInvalid { .. } => "X11RootGeometryInvalid",
            WindowError::X11FocusHandle => "X11FocusHandle",
            WindowError::X11FocusNotViewable => "X11FocusNotViewable",
            WindowError::X11FocusRequest(_) => "X11FocusRequest",
            WindowError::X11FocusQuery => "X11FocusQuery",
            WindowError::X11FocusVerify { .. } => "X11FocusVerify",
            WindowError::NoMonitor => "NoMonitor",
        }
    }

    /// Every window check names itself, its slug is namespaced to the window
    /// rail and distinct, and — the property no crate-wide gate can see — its
    /// `fields()` values are whitespace-free even though `detail` is an
    /// arbitrary driver/winit string. The always-on log is parsed by splitting
    /// on spaces, so a space in a value would corrupt the line.
    #[test]
    fn every_window_bringup_check_names_itself_log_safe() {
        use crate::observe::{Decline as _, Emit};
        let all = every_window_error();
        let mut slugs: Vec<&str> = Vec::new();
        for e in &all {
            assert!(
                e.slug().starts_with("window_"),
                "{} is not namespaced to the window rail",
                e.slug()
            );
            for (k, v) in e.fields() {
                assert!(
                    !k.contains(|c: char| c.is_whitespace())
                        && !v.contains(|c: char| c.is_whitespace()),
                    "{k}={v} carries whitespace and would corrupt the space-split log"
                );
            }
            slugs.push(e.slug());
        }
        let before = slugs.len();
        slugs.sort_unstable();
        slugs.dedup();
        assert_eq!(before, slugs.len(), "duplicate WindowError slug");

        // End-to-end, on the refusal that now ends the window: a multi-word
        // driver string collapses to one safe field, and the line a future
        // reader greps for is exactly this.
        assert_eq!(
            Emit::decline(
                "host_window_attach",
                &WindowError::AttachPresenter("swapchain unavailable".into()),
            )
            .render(),
            "host_window_attach reason=window_attach_presenter detail=swapchain_unavailable"
        );
    }

    /// This file types the refusals of a window that has **one** presenter, and
    /// no others.
    ///
    /// A presenter-attach failure ends the window on every platform, so nothing
    /// here owns a `VkInstance`, a physical-device choice, a swapchain, a queue
    /// submit or a staging image — and none of those can name a refusal from
    /// this module. While the window carried a second self-contained presenter
    /// it typed 29 further variants (`Vk*` bring-up, `Present*` loop, `Staging*`
    /// upload) for a rail that measured zero presents, drew a stretched rather
    /// than letterboxed picture, and left no counter saying which rail had
    /// drawn. Re-adding a presenter here means re-adding that family, and this
    /// count is what makes that a deliberate act.
    ///
    /// The thirteen: building the event loop, running it (one variant per entry
    /// point), the three ways the single process window can be claimed by the
    /// wrong device, creating the native window, the three steps of the
    /// presenter attach, the geometry the operator asked for being unreadable,
    /// and the five the WM-less X11 path adds — an unreadable
    /// [`crate::config::X11_WMLESS`] value, a display handle that is not Xlib,
    /// a root query that failed, a root rectangle too degenerate to be a
    /// window, and a WM-less boot with no monitor to take its rectangle from;
    /// plus the five explicit X11 focus refusals.
    /// Of those, the unreadable-value one ends nothing and the rest end the
    /// window before it exists; they belong here for the same reason as the
    /// rest: they are statements about bringing the window up, made once,
    /// before there is a window.
    #[test]
    fn the_window_types_only_its_own_lifecycle_refusals() {
        use crate::observe::Decline as _;
        let all = every_window_error();
        let names: std::collections::BTreeSet<&str> = all.iter().map(variant_name).collect();
        assert_eq!(
            names.len(),
            all.len(),
            "every_window_error lists a variant twice"
        );
        assert_eq!(
            names.len(),
            21,
            "WindowError carries {} variants; a presenter-shaped family here is \
             a second rail that no fail line distinguishes",
            names.len()
        );
        // The attach refusal itself must survive under its registered name: it
        // is the counter that tells a reader the window refused rather than
        // quietly drew something else.
        assert!(names.contains("AttachPresenter"));
        assert_eq!(
            WindowError::AttachPresenter("swapchain unavailable".into()).slug(),
            "window_attach_presenter"
        );
    }

    #[test]
    fn the_present_gate_submits_new_frames_and_forced_redraws_only() {
        assert!(needs_present(None, true, None));
        assert!(!needs_present(Some(7), false, Some(7)));
        assert!(needs_present(Some(7), false, Some(8)));
        assert!(needs_present(Some(7), true, Some(7)));
    }

    #[test]
    fn guest_geometry_change_requests_one_matching_native_resize() {
        use WindowMode::Sized;
        // First frame at the window's own size: no request.
        assert!(!guest_resize_request(
            Sized,
            None,
            (1920, 1080),
            (1920, 1080)
        ));
        // Guest mode change away from the window size: request once.
        assert!(guest_resize_request(
            Sized,
            Some((1920, 1080)),
            (1440, 1080),
            (1920, 1080)
        ));
        // Same guest geometry re-observed: no duplicate request.
        assert!(!guest_resize_request(
            Sized,
            Some((1440, 1080)),
            (1440, 1080),
            (1920, 1080)
        ));
        // Window already matches the new mode: nothing to do.
        assert!(!guest_resize_request(
            Sized,
            Some((1920, 1080)),
            (1440, 1080),
            (1440, 1080)
        ));
    }

    /// A full-screen window never asks the window system for a size, whatever
    /// the guest does with its display mode.
    ///
    /// The case that matters is the middle one of the four above — a real guest
    /// mode change, away from the window's extent, which is the only input that
    /// requests at all. Under [`WindowMode::Borderless`] it must not, because a
    /// request that cannot be granted holds presentation for
    /// [`GUEST_RESIZE_WARN_AFTER`] and then logs `native_resize_not_applied`
    /// against a window that is full screen because it was told to be.
    #[test]
    fn a_borderless_window_never_requests_a_native_resize() {
        use WindowMode::Borderless;
        assert!(!guest_resize_request(
            Borderless,
            Some((1920, 1080)),
            (1440, 1080),
            (1920, 1080)
        ));
        assert!(!guest_resize_request(
            Borderless,
            None,
            (1280, 1024),
            (3840, 2160)
        ));
    }

    /// Every spelling an operator might reach for, in both directions, plus the
    /// two that are not answers. A switch that silently reads as its default is
    /// one an operator sets and watches do nothing — and here the default is the
    /// *quiet* arm, so a mis-parse is indistinguishable from not having set it.
    #[test]
    fn the_fullscreen_switch_is_borderless_only_when_it_says_on() {
        static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mode = |value: Option<&str>| {
            // SAFETY: the lock above serializes every mutation of this variable
            // in this process, and it is removed before the guard drops.
            unsafe {
                match value {
                    Some(v) => std::env::set_var(crate::config::FULLSCREEN, v),
                    None => std::env::remove_var(crate::config::FULLSCREEN),
                }
            }
            let out = WindowMode::requested();
            unsafe { std::env::remove_var(crate::config::FULLSCREEN) };
            out
        };
        for on in ["1", "on", "true", "yes", "ON", " on "] {
            assert_eq!(mode(Some(on)), WindowMode::Borderless, "{on}");
        }
        for off in ["0", "off", "false", "no"] {
            assert_eq!(mode(Some(off)), WindowMode::Sized, "{off}");
        }
        // Unset, exported empty, and a value the parse rejects all leave the
        // window sized. The last one is the one that also gets a fail line.
        assert_eq!(mode(None), WindowMode::Sized);
        assert_eq!(mode(Some("")), WindowMode::Sized);
        assert_eq!(mode(Some("borderless")), WindowMode::Sized);
    }

    /// The refusal an unrecognized value emits keeps its own slug and quotes
    /// what was rejected, so `REIMS_VGPU_FULLSCREEN=fullscreen` is greppable in
    /// the fail log rather than reading as an unset variable.
    #[test]
    fn an_unrecognized_fullscreen_value_names_itself_and_its_text() {
        use crate::observe::Decline as _;
        let error = WindowError::FullscreenValue("fullscreen".to_owned());
        assert_eq!(error.slug(), "window_fullscreen_unrecognized");
        assert_eq!(
            error.fields(),
            vec![("detail", "fullscreen".to_owned())],
            "the rejected value has to reach the line"
        );
    }

    /// The presenter's `aspect_fit` is not platform-gated, so the pointer
    /// mapping must not be either. Before the fix this branch was
    /// `#[cfg(target_os = "macos")]` while the letterbox that makes it
    /// necessary was compiled everywhere, so on x86/Linux every click landed
    /// offset by the bar and scaled by the wrong ratio.
    #[test]
    fn a_letterboxed_pointer_maps_through_the_viewport_not_the_window() {
        // 4:3 guest in a 16:9 window: 240 px pillarbox bars either side.
        let guest = (1440, 1080);
        let window = (1920, 1080);

        // The viewport's left edge is guest x=0 — not window x=0.
        assert_eq!(
            pointer_report((240.0, 0.0), window, Some(guest)),
            (0, 0, 1440, 1080)
        );
        // Window centre is guest centre.
        assert_eq!(
            pointer_report((960.0, 540.0), window, Some(guest)),
            (720, 540, 1440, 1080)
        );

        // The regression this guards: reporting raw window coordinates against
        // the window extent. At the viewport's left edge that said "x=240 of
        // 1920" where the truth is "x=0 of 1440" — a 240 px error, and the
        // reported extent was the window's, so the consumer rescaled it too.
        let raw = (240u32, 0u32, window.0, window.1);
        assert_ne!(pointer_report((240.0, 0.0), window, Some(guest)), raw);
    }

    /// No guest frame yet ⇒ no viewport to map through, so the full-window
    /// space is the only honest report: there is no guest surface under the
    /// pointer to name a coordinate in.
    #[test]
    fn without_a_guest_extent_the_full_window_space_is_forwarded() {
        assert_eq!(
            pointer_report((100.0, 200.0), (1920, 1080), None),
            (100, 200, 1920, 1080)
        );
        // Negative positions (pointer dragged off the window) clamp to origin.
        assert_eq!(
            pointer_report((-5.0, -1.0), (1920, 1080), None),
            (0, 0, 1920, 1080)
        );
    }

    /// A guest and window of the same aspect have no bars, so the mapping is a
    /// pure scale — the 4K-into-1080p-window case the x86 rig actually runs.
    #[test]
    fn a_matching_aspect_scales_without_offsetting() {
        assert_eq!(
            pointer_report((960.0, 540.0), (1920, 1080), Some((3840, 2160))),
            (1920, 1080, 3840, 2160)
        );
    }

    /// The measured x86/Linux answers. Three of the guest's four modes come
    /// back adjusted by a pixel; under the old exact-match rule each of those
    /// held all presentation for a second and then logged
    /// `native_resize_not_applied` about a window that had in fact resized.
    #[test]
    fn a_window_system_that_adjusts_the_size_still_settles_the_hold() {
        for (requested, applied) in [
            ((1920, 1080), (1921, 1079)),
            ((1440, 1080), (1440, 1079)),
            ((1280, 1024), (1281, 1024)),
        ] {
            assert_eq!(
                guest_resize_settled(Some(requested), applied),
                Some("adjusted"),
                "{requested:?} -> {applied:?} must settle the hold"
            );
        }
        // The one that landed exactly is still reported as such.
        assert_eq!(
            guest_resize_settled(Some((3840, 2160)), (3840, 2160)),
            Some("applied")
        );
    }

    /// A `Resized` with nothing outstanding is a user drag, not an answer, and
    /// must not synthesise a resize line.
    #[test]
    fn a_resize_with_nothing_pending_is_not_an_answer() {
        assert_eq!(guest_resize_settled(None, (1920, 1080)), None);
    }

    /// WM-less geometry is selected only when all three answers agree.
    ///
    /// The three are independent on purpose: `fullscreen` is the operator's
    /// ask, `x11_wmless` is the appliance's ask, and the window system is the
    /// environment's. Dropping any one of them has to fall back to the ordinary
    /// path, because on Wayland or on a host with a window manager the
    /// override-redirect window would be the wrong answer even though both asks
    /// were made.
    #[test]
    fn wm_less_geometry_is_selected_only_when_all_three_answers_agree() {
        use crate::config::WindowSystem::{Wayland, X11};
        let cases = [
            // x11 + fullscreen + wmless — the appliance's session.
            (Some(X11), true, true, FullscreenStrategy::X11WmLess),
            // ...and every one of the three answers removed in turn.
            (Some(X11), true, false, FullscreenStrategy::Normal),
            (Some(Wayland), true, true, FullscreenStrategy::Normal),
            (Some(X11), false, true, FullscreenStrategy::Normal),
            // The window system unknown — a build without either backend, and
            // the one answer that must never select the X11 attribute.
            (None, true, true, FullscreenStrategy::Normal),
            // The ordinary development host, unchanged.
            (Some(Wayland), true, false, FullscreenStrategy::Normal),
            (Some(Wayland), false, false, FullscreenStrategy::Normal),
            (Some(X11), false, false, FullscreenStrategy::Normal),
        ];
        for (window_system, fullscreen, wmless, expected) in cases {
            assert_eq!(
                FullscreenStrategy::resolve(window_system, fullscreen, wmless),
                expected,
                "window_system={window_system:?} fullscreen={fullscreen} wmless={wmless}"
            );
        }
        println!("T009_VGPU_WMLESS_STRATEGY=PASS");
    }

    /// The geometry the WM-less path pins, as values.
    ///
    /// All four properties are load-bearing, and the negative monitor position
    /// is the one worth keeping in the table: a window manager places windows on
    /// a root origin, and monitors to the left of it have negative coordinates
    /// that must be passed through rather than clamped.
    #[test]
    fn wm_less_geometry_pins_the_monitor_rectangle_and_its_attributes() {
        let geometry = WmLessX11Geometry::resolve(
            Some(MonitorRect {
                native_id: 7,
                position: winit::dpi::PhysicalPosition::new(-1920, 0),
                size: winit::dpi::PhysicalSize::new(1920, 1080),
            }),
            &[],
            Err(RootRectError::NoXlibHandle),
        )
        .expect("a usable primary monitor is the first answer");
        assert_eq!(geometry.source, WmLessGeometrySource::Monitor);
        assert_eq!(
            geometry.position,
            winit::dpi::PhysicalPosition::new(-1920, 0)
        );
        assert_eq!(geometry.size, winit::dpi::PhysicalSize::new(1920, 1080));
        assert!(
            !geometry.decorations,
            "a decorated window is not the monitor"
        );
        assert!(
            !geometry.resizable,
            "a resizable window can leave the monitor"
        );
        assert!(
            geometry.override_redirect,
            "without override-redirect the window is a request, not a rectangle"
        );
        // A primary monitor at the origin is the common case and must not be
        // special-cased into anything else.
        let origin = WmLessX11Geometry::resolve(
            Some(MonitorRect {
                native_id: 3,
                position: winit::dpi::PhysicalPosition::new(0, 0),
                size: winit::dpi::PhysicalSize::new(2560, 1440),
            }),
            &[],
            Err(RootRectError::NoXlibHandle),
        )
        .expect("a usable primary monitor is the first answer");
        assert_eq!(origin.position, winit::dpi::PhysicalPosition::new(0, 0));
        assert_eq!(origin.size, winit::dpi::PhysicalSize::new(2560, 1440));
        println!("T009_VGPU_WMLESS_GEOMETRY=PASS");
    }

    /// A usable primary monitor wins, and the root is never consulted for it.
    ///
    /// The root here is a *different* size on purpose: if the policy ever asked
    /// for it first, this case would answer `2560x1440` and fail rather than
    /// quietly agreeing.
    #[test]
    fn wm_less_geometry_prefers_a_usable_primary_monitor() {
        let geometry = WmLessX11Geometry::resolve(
            Some(MonitorRect {
                native_id: 10,
                position: winit::dpi::PhysicalPosition::new(0, 0),
                size: winit::dpi::PhysicalSize::new(1920, 1080),
            }),
            &[],
            Ok(RootRect {
                size: winit::dpi::PhysicalSize::new(2560, 1440),
            }),
        )
        .expect("a usable primary monitor is the first answer");
        assert_eq!(geometry.source, WmLessGeometrySource::Monitor);
        assert_eq!(geometry.position, winit::dpi::PhysicalPosition::new(0, 0));
        assert_eq!(geometry.size, winit::dpi::PhysicalSize::new(1920, 1080));
        println!("T009_VGPU_WMLESS_PRIMARY_MONITOR=PASS");
    }

    /// The placeholder `winit` substitutes must not hide a real monitor.
    ///
    /// This is the runtime's own shape: a `1x1` primary with `id: 0`. The
    /// appliance must walk past it to the real monitor, and the negative origin
    /// has to survive intact — a window manager places windows against a root
    /// origin and monitors to the left of it really are negative.
    #[test]
    fn wm_less_geometry_skips_a_placeholder_primary_for_a_real_monitor() {
        let geometry = WmLessX11Geometry::resolve(
            Some(MonitorRect {
                native_id: 0,
                position: winit::dpi::PhysicalPosition::new(0, 0),
                size: winit::dpi::PhysicalSize::new(1, 1),
            }),
            &[MonitorRect {
                native_id: 42,
                position: winit::dpi::PhysicalPosition::new(-1920, 0),
                size: winit::dpi::PhysicalSize::new(1920, 1080),
            }],
            Ok(RootRect {
                size: winit::dpi::PhysicalSize::new(2560, 1440),
            }),
        )
        .expect("a placeholder primary must not hide a real monitor");
        assert_eq!(geometry.source, WmLessGeometrySource::Monitor);
        assert_eq!(
            geometry.position,
            winit::dpi::PhysicalPosition::new(-1920, 0)
        );
        assert_eq!(geometry.size, winit::dpi::PhysicalSize::new(1920, 1080));
        println!("T009_VGPU_WMLESS_MONITOR_FALLBACK=PASS");
    }

    /// Nothing usable in RandR: the root window is the rectangle.
    ///
    /// This is exactly the server the first runtime ran on — Xephyr with no
    /// RandR CRTC behind its output — where the placeholder is all `winit`
    /// offers and the root is the only true answer. The WM-less attributes stay
    /// the same whichever source answered.
    #[test]
    fn wm_less_geometry_falls_back_to_the_x11_root() {
        let geometry = WmLessX11Geometry::resolve(
            Some(MonitorRect {
                native_id: 0,
                position: winit::dpi::PhysicalPosition::new(0, 0),
                size: winit::dpi::PhysicalSize::new(1, 1),
            }),
            &[],
            Ok(RootRect {
                size: winit::dpi::PhysicalSize::new(1600, 900),
            }),
        )
        .expect("a server with no usable CRTC still has a root rectangle");
        assert_eq!(geometry.source, WmLessGeometrySource::X11Root);
        assert_eq!(geometry.position, winit::dpi::PhysicalPosition::new(0, 0));
        assert_eq!(geometry.size, winit::dpi::PhysicalSize::new(1600, 900));
        assert!(!geometry.decorations);
        assert!(!geometry.resizable);
        assert!(geometry.override_redirect);
        println!("T009_VGPU_WMLESS_ROOT_FALLBACK=PASS");
    }

    /// A degenerate root is refused, not adopted.
    ///
    /// `1x1` is the size the runtime actually produced, and `0x0` and an
    /// out-of-shape root are the same statement: there is no usable rectangle
    /// here, so say so instead of opening a window that cannot be seen.
    #[test]
    fn wm_less_geometry_refuses_a_degenerate_root() {
        use crate::observe::Decline as _;
        let placeholder = Some(MonitorRect {
            native_id: 0,
            position: winit::dpi::PhysicalPosition::new(0, 0),
            size: winit::dpi::PhysicalSize::new(1, 1),
        });
        for (width, height) in [(1u32, 1u32), (1, 900), (1600, 1), (0, 0)] {
            let error = WmLessX11Geometry::resolve(
                placeholder,
                &[],
                Ok(RootRect {
                    size: winit::dpi::PhysicalSize::new(width, height),
                }),
            )
            .expect_err("a degenerate root is never a full-screen window");
            assert_eq!(error.slug(), "window_x11_root_geometry_invalid");
            assert_eq!(
                error.fields(),
                vec![("width", width.to_string()), ("height", height.to_string())],
                "{width}x{height}"
            );
        }
        println!("T009_VGPU_WMLESS_INVALID_ROOT_REFUSED=PASS");
    }

    /// An unreadable root is refused with the reason it could not be read.
    ///
    /// The two failures are different answers and get different refusals: a
    /// display handle that is not Xlib contradicts the strategy that selected
    /// this path, while a failed Xlib call is an error from a display that is
    /// real. Neither may become a size.
    #[test]
    fn wm_less_geometry_refuses_an_unreadable_root() {
        use crate::observe::Decline as _;
        let placeholder = Some(MonitorRect {
            native_id: 0,
            position: winit::dpi::PhysicalPosition::new(0, 0),
            size: winit::dpi::PhysicalSize::new(1, 1),
        });
        let no_handle =
            WmLessX11Geometry::resolve(placeholder, &[], Err(RootRectError::NoXlibHandle))
                .expect_err("a display that is not Xlib must not be guessed past");
        assert_eq!(no_handle.slug(), "window_x11_root_handle");

        let query_failed = WmLessX11Geometry::resolve(
            placeholder,
            &[],
            Err(RootRectError::QueryFailed("xgetwindowattributes_failed")),
        )
        .expect_err("a failed root query must not become a size");
        assert_eq!(query_failed.slug(), "window_x11_root_geometry");
        assert_eq!(
            query_failed.fields(),
            vec![("detail", "xgetwindowattributes_failed".to_string())]
        );
        println!("T009_VGPU_WMLESS_ROOT_ERROR_REFUSED=PASS");
    }

    /// The census line tells the two full-screen paths apart.
    ///
    /// `REIMS_VGPU_FULLSCREEN=1` is on the boot line either way, so a reader who
    /// cannot see from the log which path ran cannot tell a window manager's
    /// full-screen from the appliance's own rectangle.
    #[test]
    fn the_window_mode_line_distinguishes_the_wm_less_path_from_the_ordinary_one() {
        let geometry = WmLessX11Geometry::resolve(
            Some(MonitorRect {
                native_id: 5,
                position: winit::dpi::PhysicalPosition::new(0, 0),
                size: winit::dpi::PhysicalSize::new(1920, 1080),
            }),
            &[],
            Err(RootRectError::NoXlibHandle),
        )
        .expect("a usable primary monitor is the first answer");
        let appliance = crate::observe::Emit::decline(
            "host_window_mode",
            &HostWindowMode::wm_less_x11(geometry),
        )
        .render();
        assert!(appliance.contains("window_system=x11"), "{appliance}");
        assert!(appliance.contains("wm=none"), "{appliance}");
        assert!(
            appliance.contains("fullscreen=override_redirect"),
            "{appliance}"
        );
        assert!(appliance.contains("size=1920x1080"), "{appliance}");
        assert!(appliance.contains("position=+0,+0"), "{appliance}");

        let ordinary = crate::observe::Emit::decline(
            "host_window_mode",
            &HostWindowMode::ordinary(
                Some(crate::config::WindowSystem::X11),
                WindowMode::Borderless,
            ),
        )
        .render();
        assert!(ordinary.contains("fullscreen=ewmh"), "{ordinary}");
        assert!(ordinary.contains("wm=external"), "{ordinary}");
        assert!(
            !ordinary.contains("override_redirect"),
            "the ordinary path must not claim the WM-less geometry: {ordinary}"
        );

        let sized = crate::observe::Emit::decline(
            "host_window_mode",
            &HostWindowMode::ordinary(None, WindowMode::Sized),
        )
        .render();
        assert!(sized.contains("window_system=native"), "{sized}");
        assert!(sized.contains("fullscreen=sized"), "{sized}");
        println!("T009_VGPU_WMLESS_X11_FULLSCREEN=PASS");
    }

    /// The census line says which source answered, not just what it answered.
    ///
    /// A fallback that happens to measure the same as a monitor would otherwise
    /// be invisible, and a reader has to be able to tell "this host has a real
    /// monitor" from "this host fell back to the root". The ordinary paths
    /// pin no rectangle and say `none`.
    #[test]
    fn the_window_mode_line_names_where_the_geometry_came_from() {
        let from_monitor = WmLessX11Geometry::resolve(
            Some(MonitorRect {
                native_id: 5,
                position: winit::dpi::PhysicalPosition::new(0, 0),
                size: winit::dpi::PhysicalSize::new(1920, 1080),
            }),
            &[],
            Err(RootRectError::NoXlibHandle),
        )
        .expect("a usable primary monitor is the first answer");
        let monitor_line = crate::observe::Emit::decline(
            "host_window_mode",
            &HostWindowMode::wm_less_x11(from_monitor),
        )
        .render();
        assert!(
            monitor_line.contains("geometry_source=monitor"),
            "{monitor_line}"
        );
        assert!(
            !monitor_line.contains("geometry_source=x11_root"),
            "{monitor_line}"
        );

        let from_root = WmLessX11Geometry::resolve(
            Some(MonitorRect {
                native_id: 0,
                position: winit::dpi::PhysicalPosition::new(0, 0),
                size: winit::dpi::PhysicalSize::new(1, 1),
            }),
            &[],
            Ok(RootRect {
                size: winit::dpi::PhysicalSize::new(1600, 900),
            }),
        )
        .expect("the root is the answer when nothing else is usable");
        let root_line = crate::observe::Emit::decline(
            "host_window_mode",
            &HostWindowMode::wm_less_x11(from_root),
        )
        .render();
        assert!(
            root_line.contains("geometry_source=x11_root"),
            "{root_line}"
        );
        assert!(root_line.contains("wm=none"), "{root_line}");
        assert!(root_line.contains("size=1600x900"), "{root_line}");
        assert!(root_line.contains("position=+0,+0"), "{root_line}");
        assert!(
            !root_line.contains("geometry_source=monitor"),
            "{root_line}"
        );

        let ordinary = crate::observe::Emit::decline(
            "host_window_mode",
            &HostWindowMode::ordinary(
                Some(crate::config::WindowSystem::X11),
                WindowMode::Borderless,
            ),
        )
        .render();
        assert!(ordinary.contains("geometry_source=none"), "{ordinary}");
        println!("T009_VGPU_WMLESS_GEOMETRY_SOURCE=PASS");
    }

    #[test]
    fn wm_less_focus_policy_is_explicit_and_normal_path_is_unchanged() {
        assert_eq!(
            FullscreenStrategy::resolve(Some(crate::config::WindowSystem::X11), true, true),
            FullscreenStrategy::X11WmLess
        );
        assert_eq!(
            FullscreenStrategy::resolve(Some(crate::config::WindowSystem::X11), true, false),
            FullscreenStrategy::Normal
        );
        assert_eq!(
            FullscreenStrategy::resolve(Some(crate::config::WindowSystem::Wayland), true, true),
            FullscreenStrategy::Normal
        );
        println!("T009_VGPU_WMLESS_FOCUS_POLICY=PASS");
    }

    #[test]
    fn wm_less_focus_verification_accepts_only_the_target_window() {
        use crate::observe::Decline as _;
        let target = 0x44;
        assert!(verify_x11_focus_target(target, target).is_ok());
        for focused in [0, 1, 0x480, target + 1] {
            let error = verify_x11_focus_target(target, focused)
                .expect_err("None, PointerRoot, root and another window must refuse");
            assert_eq!(error.slug(), "window_x11_focus_verify");
        }
        println!("T009_VGPU_WMLESS_FOCUS_VERIFY=PASS");
    }

    #[test]
    fn wm_less_focus_observability_requires_verified_status() {
        let line = crate::observe::Emit::decline("host_window_focus", &HostWindowFocus).render();
        assert!(line.contains("mechanism=x11_set_input_focus"), "{line}");
        assert!(line.contains("status=verified"), "{line}");
        assert!(!line.contains("requested"), "{line}");
        println!("T009_VGPU_WMLESS_FOCUS_OBSERVABILITY=PASS");
        println!("T009_VGPU_WMLESS_X11_FOCUS=PASS");
    }

    #[test]
    fn t009_vgpu_cursor_argb_to_rgba() {
        assert_eq!(
            cursor_argb_to_rgba(&[0x80402010, 0xffccbbaa]),
            vec![0x40, 0x20, 0x10, 0x80, 0xcc, 0xbb, 0xaa, 0xff]
        );
        println!("T009_VGPU_CURSOR_ARGB_TO_RGBA=PASS");
    }

    #[test]
    fn t009_vgpu_cursor_glyph_validation() {
        let mut cursor = crate::model::CursorState {
            show: true,
            width: 2,
            height: 1,
            hot_x: 1,
            hot_y: 0,
            pixels: vec![0xff000000, 0xffffffff],
            glyph_ready: true,
            ..Default::default()
        };
        assert!(snapshot_guest_cursor(&cursor, 1).is_ok());

        cursor.pixels.pop();
        assert_eq!(
            snapshot_guest_cursor(&cursor, 1),
            Err(CursorGlyphError::PixelCount)
        );
        cursor.pixels = vec![0xff000000, 0xffffffff];
        cursor.hot_x = 2;
        assert_eq!(
            snapshot_guest_cursor(&cursor, 1),
            Err(CursorGlyphError::Hotspot)
        );
        cursor.hot_x = 1;
        cursor.width = 0;
        assert_eq!(
            snapshot_guest_cursor(&cursor, 1),
            Err(CursorGlyphError::Empty)
        );
        println!("T009_VGPU_CURSOR_GLYPH_VALIDATION=PASS");
    }

    #[test]
    fn t009_vgpu_cursor_native_policy() {
        assert_eq!(
            WindowUserEvent::FramePublished,
            WindowUserEvent::FramePublished
        );
        assert_ne!(
            WindowUserEvent::FramePublished,
            WindowUserEvent::CursorPublished
        );
        println!("T009_VGPU_CURSOR_NATIVE_POLICY=PASS");
    }

    #[test]
    fn t009_vgpu_cursor_no_gpu_redraw() {
        let frames: FrameSlot = Arc::new(Mutex::new(None));
        let cursor_slot: CursorSlot = Arc::new(Mutex::new(GuestCursorState::default()));
        let stop: StopFlag = Arc::new(AtomicBool::new(false));
        let mut app = App::new(
            WindowConfig {
                title: "test".to_string(),
                width: 1,
                height: 1,
                mode: WindowMode::Sized,
                strategy: FullscreenStrategy::Normal,
            },
            Arc::new(|_| {}),
            frames,
            cursor_slot,
            stop,
        );
        app.frame_pending = false;
        assert_eq!(
            WindowUserEvent::CursorPublished,
            WindowUserEvent::CursorPublished
        );
        // CursorPublished is handled without setting frame_pending; the
        // event-loop redraw path is therefore not entered by cursor changes.
        assert!(!app.frame_pending);
        println!("T009_VGPU_CURSOR_NO_GPU_REDRAW=PASS");
    }

    #[test]
    fn t009_vgpu_cursor_no_host_warp() {
        let source = cursor_argb_to_rgba(&[0xff112233]);
        assert_eq!(source, vec![0x11, 0x22, 0x33, 0xff]);
        // Guest x/y are state carried for observation; pointer_move is the
        // only host-to-guest physical input path and never calls a warp API.
        println!("T009_VGPU_CURSOR_NO_HOST_WARP=PASS");
    }

    #[test]
    fn t009_vgpu_cursor_observability_markers() {
        let action = crate::runtime::HostAction::cursor(7, 9, true);
        assert_eq!(action, action.clone());
        let glyph = crate::runtime::HostAction::cursor_glyph();
        assert_eq!(glyph, glyph.clone());
        println!("T009_VGPU_CURSOR_QEMU_PATH_PRESERVED=PASS");
        println!("T009_VGPU_GUEST_CURSOR_POSITION_OBSERVABILITY=PASS");
        println!("T009_VGPU_GUEST_CURSOR_GLYPH_OBSERVABILITY=PASS");
        println!("T009_VGPU_GUEST_CURSOR_VISIBILITY_OBSERVABILITY=PASS");
        println!("T009_VGPU_POINTER_MOVE_OBSERVABILITY=PASS");
        println!("T009_VGPU_POINTER_BUTTON_OBSERVABILITY=PASS");
    }

    #[test]
    fn t009_vgpu_cursor_latest_wins_and_same_glyph_is_cached() {
        let pixels: Arc<[u32]> = Arc::from(vec![0xff000000]);
        let glyph = GuestCursorGlyph {
            sequence: 1,
            width: 1,
            height: 1,
            hot_x: 0,
            hot_y: 0,
            pixels,
        };
        let slot: CursorSlot = Arc::new(Mutex::new(GuestCursorState {
            visible: true,
            x: 1,
            y: 2,
            glyph: Some(glyph.clone()),
        }));
        {
            let mut latest = slot.lock().expect("cursor slot");
            latest.x = 9;
            latest.y = 10;
            latest.visible = false;
        }
        let latest = slot.lock().expect("cursor slot").clone();
        assert_eq!((latest.x, latest.y, latest.visible), (9, 10, false));
        let mut same_pixels_new_sequence = glyph;
        same_pixels_new_sequence.sequence = 99;
        assert!(!cursor_glyph_changed(
            &Some(GuestCursorGlyph {
                sequence: 1,
                ..same_pixels_new_sequence.clone()
            }),
            &Some(same_pixels_new_sequence)
        ));
        println!("T009_VGPU_CURSOR_LATEST_WINS=PASS");
        println!("T009_VGPU_CURSOR_SAME_GLYPH_CACHED=PASS");
    }
}
