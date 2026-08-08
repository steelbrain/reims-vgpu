//! Host-owned window presentation — a `winit` window that presents the guest
//! frame on the engine's own `VkDevice`, replacing QEMU's UI ([[host-window]]).
//!
//! This file owns the window and the event loop. The surface, the swapchain and
//! the acquire → clear/blit → present sequence live in
//! [`crate::backend::vulkan::engine::window_present`], on the device that
//! rendered the frame; this file drives them and decides *when* to present. It
//! also translates window input via [`super::input_map`] and hands each
//! [`HostAction`] to the [`InputSink`] (the device wires that to the prompt
//! action queue).
//!
//! The presenter prefers the engine-resident image, which is what keeps a
//! presented layer from crossing host memory at all; the CPU-BGRA [`FrameSlot`]
//! the device fills from its present capture is the source for the frames no
//! resident carries — the firmware framebuffer, and any mapping the compositor
//! has not rendered into.
//!
//! There is exactly one presenter, on every platform. A host whose engine
//! device cannot present to this surface gets a named refusal and a shutdown
//! rather than a second `VkDevice`; `resumed` records why.
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
use winit::window::{Window, WindowId};

use super::input_map;
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
/// The loop used to ask for a redraw every 2 ms and let [`needs_engine_present`]
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
const ENGINE_WINDOW_REDRAW_BACKSTOP: std::time::Duration =
    std::time::Duration::from_millis(GUEST_RESIZE_WARN_AFTER.as_millis() as u64 / 10);
/// How long a guest-driven native resize request may stay unmatched by a
/// winit `Resized` event before the always-on alarm names it. Live requests
/// apply within single-digit milliseconds; one second means the window system
/// refused or clamped the size and the window is presenting letterboxed
/// instead. A tiling or fullscreen Wayland/X11 compositor refuses by policy, so
/// on those hosts this alarm is the expected steady state rather than a fault.
const GUEST_RESIZE_WARN_AFTER: std::time::Duration = std::time::Duration::from_secs(1);

/// Window creation parameters.
#[derive(Clone, Debug)]
pub struct WindowConfig {
    pub title: String,
    pub width: u32,
    pub height: u32,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            title: "Reims vGPU".to_string(),
            width: 1280,
            height: 800,
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
    /// when `seq` advances. macOS submits the engine resident only when
    /// `seq` advances or the window resizes, so an unchanged desktop does not
    /// contend with guest render work for the engine queue.
    /// Wrap-around is harmless: a collision at most skips one prepare (the source
    /// still holds valid content).
    pub seq: u64,
    pub width: u32,
    pub height: u32,
    pub bgra: Vec<u8>,
    /// Engine-resident source for same-device MoltenVK presentation.
    pub resident: Option<crate::backend::vulkan::engine::WindowPresentSource>,
}

/// Shared slot the device writes and the window reads (latest-wins). The frame
/// is `Arc`-wrapped so the window's per-vblank read is a refcount bump, not an
/// 8 MiB deep copy of an unchanged frame.
pub type FrameSlot = Arc<Mutex<Option<Arc<Frame>>>>;

/// The one user event this window's loop takes: the device wrote a new frame
/// into the [`FrameSlot`].
///
/// Carries nothing. The slot is latest-wins and the loop reads it under its own
/// lock, so a payload here could only be a second, staler copy of what
/// [`App::draw`] is about to read anyway.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FramePublished;

/// How the device wakes the window's event loop when it publishes a frame.
///
/// The window used to find new frames by asking for a redraw every 2 ms and
/// discarding 98 % of them at the seq gate — see
/// [`ENGINE_WINDOW_REDRAW_BACKSTOP`] for that reading. This turns it round: the
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
    proxy: Mutex<Option<winit::event_loop::EventLoopProxy<FramePublished>>>,
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
    fn arm(&self, proxy: winit::event_loop::EventLoopProxy<FramePublished>) {
        if let Ok(mut slot) = self.proxy.lock() {
            *slot = Some(proxy);
        }
    }

    /// Ask the loop to look at the frame slot.
    ///
    /// Silent in all three failure modes, because none of them is a lost frame:
    /// before the loop is armed and after it has closed there is nothing that
    /// could present, and a poisoned lock means the window thread panicked. In
    /// every case [`ENGINE_WINDOW_REDRAW_BACKSTOP`] still runs, so a wake that
    /// does not land costs latency bounded by that constant rather than a frame
    /// — which is the property that lets this be a wake and not a protocol.
    pub fn wake(&self) {
        if let Ok(slot) = self.proxy.lock() {
            if let Some(proxy) = slot.as_ref() {
                let _ = proxy.send_event(FramePublished);
            }
        }
    }
}

/// Offer a published frame's CPU bytes to the engine presenter.
///
/// The presenter prefers the resident and only reads these when none carries the
/// display — the firmware framebuffer, and any mapping the compositor has not
/// rendered into. `bgra` is empty on presents the device elided the readback
/// for, and the presenter rejects a short buffer rather than blitting a torn
/// frame.
fn window_cpu_frame(frame: &Frame) -> crate::backend::vulkan::engine::WindowCpuFrame<'_> {
    crate::backend::vulkan::engine::WindowCpuFrame {
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
fn needs_engine_present(
    presented: Option<u64>,
    redraw_required: bool,
    incoming: Option<u64>,
) -> bool {
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
/// size, which is the regression [`super::viewport`]'s module docs record as
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
            let (x, y) = super::viewport::pointer_to_guest(position, window, guest);
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
/// content resize: only on a geometry change, and only when the window does
/// not already match. User-driven host resizing stays untouched until the
/// guest picks another mode.
fn guest_resize_request(
    observed: Option<(u32, u32)>,
    incoming: (u32, u32),
    window: (u32, u32),
) -> bool {
    observed != Some(incoming) && incoming != window
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
/// process window, creating the native window, or attaching the engine
/// presenter to it. Nothing here describes a swapchain or a blit — those belong
/// to the engine presenter, which types its own declines
/// ([`crate::backend::vulkan::engine`]'s `DrawError`), and there is no second
/// presenter in this file to type declines for.
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
    AlreadyOwned { owner: u64 },
    /// `run_main_thread` found no registered window for the device.
    NoRegisteredWindow { id: u64 },
    /// `run_main_thread` was asked to run a window owned by another device.
    WrongOwner { owner: u64, requested: u64 },
    /// `resumed`: winit could not create the native window (shared step, both
    /// platforms) — the bring-up cannot proceed past this.
    CreateNativeWindow(String),
    /// Engine-attach: the window's display handle was unavailable.
    AttachDisplayHandle(String),
    /// Engine-attach: the window's window handle was unavailable.
    AttachWindowHandle(String),
    /// Engine-attach: `window_present_attach` (engine swapchain) failed. The
    /// refusal that ends the window, on every platform.
    AttachEngine(String),
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
            | Self::AttachEngine(d) => Some(d),
            Self::AlreadyOwned { .. }
            | Self::NoRegisteredWindow { .. }
            | Self::WrongOwner { .. } => None,
        }
    }
}

/// Collapse whitespace runs to single `_` so a driver/winit string is safe as a
/// log field value ([`crate::observe::Emit`] splits the line on spaces).
fn detail_field(s: &str) -> String {
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
            Self::AttachEngine(_) => "window_attach_engine",
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
    stop: StopFlag,
    wake: WindowWakeHandle,
) -> std::thread::JoinHandle<Result<(), WindowError>> {
    std::thread::Builder::new()
        .name("reims-vgpu-window".to_string())
        .spawn(move || run(config, on_input, frames, stop, wake))
        .expect("spawn reims-vgpu-window thread")
}

/// Run the window event loop on the calling thread (blocks until the window
/// closes). Prefer [`spawn`]; call this directly only if you already own a
/// suitable thread.
pub fn run(
    config: WindowConfig,
    on_input: InputSink,
    frames: FrameSlot,
    stop: StopFlag,
    wake: WindowWakeHandle,
) -> Result<(), WindowError> {
    let event_loop = build_event_loop()?;
    wake.arm(event_loop.create_proxy());
    let mut app = App::new(config, on_input, frames, stop);
    event_loop
        .run_app(&mut app)
        .map_err(|e| WindowError::RunApp(e.to_string()))
}

#[cfg(target_os = "macos")]
struct MainThreadWindow {
    id: u64,
    event_loop: EventLoop<FramePublished>,
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
        let app = App::new(config, on_input, frames, stop);
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
/// Carries [`FramePublished`] as its user event, which is what makes
/// `create_proxy` a wake channel the device can hold — see [`WindowWaker`].
fn build_event_loop() -> Result<EventLoop<FramePublished>, WindowError> {
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
    builder
        .build()
        .map_err(|e| WindowError::EventLoopBuild(e.to_string()))
}

struct App {
    config: WindowConfig,
    on_input: InputSink,
    frames: FrameSlot,
    /// Set by the device to request teardown; polled in `about_to_wait`.
    stop: StopFlag,
    /// True once a `WindowClosed` action has been emitted (UI close), so the
    /// shutdown request is sent exactly once.
    closed_sent: bool,
    window: Option<Arc<Window>>,
    /// Last cursor position in window pixels (for absolute pointer moves).
    cursor: (u32, u32),
    /// The engine presenter holds a swapchain on this window's surface. False
    /// before the attach in `resumed` and again after `exiting` releases it;
    /// there is no other presenter, so false means nothing can be drawn.
    engine_attached: bool,
    first_engine_present_logged: bool,
    first_engine_guest_logged: bool,
    engine_error_logged: bool,
    /// A [`FramePublished`] arrived (or a present asked to be repeated) and no
    /// redraw has been requested for it yet. Consumed by `about_to_wait`, which
    /// is the one place that talks to the platform about redraws.
    frame_pending: bool,
    /// The latest the loop may sleep to when nothing has asked it to draw. Pushed
    /// forward by every redraw request, so a steady publish stream never lets it
    /// fire; see [`ENGINE_WINDOW_REDRAW_BACKSTOP`] for what it is a floor under.
    next_backstop: std::time::Instant,
    /// Frame seq the drawable currently holds, or `None` before the first
    /// present.
    last_engine_seq: Option<u64>,
    /// Force the next present regardless of seq: first frame, resize, or a
    /// swapchain that reported suboptimal.
    engine_redraw_required: bool,
    /// Last guest DisplaySwap geometry the window observed. Drives the
    /// once-per-mode-change native resize request and the pointer-to-guest
    /// viewport transform ([`super::viewport`]).
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
    /// plus [`ENGINE_WINDOW_REDRAW_BACKSTOP`] ticks rather than by the tick
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

impl ApplicationHandler<FramePublished> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title(self.config.title.clone())
            .with_inner_size(winit::dpi::PhysicalSize::new(
                self.config.width,
                self.config.height,
            ));
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
        // Present from the engine's own device. Presenting the compositor
        // resident from the device that rendered it is what removes the three
        // full-frame host copies every presented layer otherwise pays: the
        // drain's `read_target`, the publish copy, and a staging upload.
        let attach = window
            .display_handle()
            .map_err(|error| WindowError::AttachDisplayHandle(error.to_string()))
            .and_then(|display| {
                window
                    .window_handle()
                    .map_err(|error| WindowError::AttachWindowHandle(error.to_string()))
                    .map(|handle| (display.as_raw(), handle.as_raw()))
            })
            .and_then(|(display, handle)| {
                let size = window.inner_size();
                crate::backend::vulkan::engine::window_present_attach(
                    display,
                    handle,
                    size.width.max(1),
                    size.height.max(1),
                )
                .map_err(|error| WindowError::AttachEngine(error.to_string()))
            });
        match attach {
            Ok(()) => {
                self.engine_attached = true;
                crate::backend::vulkan::engine::note_window_present_attached(true);
                // Kick the first frame; RedrawRequested re-arms each subsequent
                // one, so without this the window would never draw.
                window.request_redraw();
                self.window = Some(window);
            }
            Err(error) => {
                // One rule on every platform: a host whose engine device cannot
                // present to this surface gets a named refusal, not a second
                // presenter.
                //
                // The alternative was a self-contained `VkInstance`/`VkDevice`
                // and swapchain in this file, reached only here. It could not
                // draw the same picture the engine rail draws — it stretched
                // instead of letterboxing, so the pointer mapping had to be
                // suppressed to stay in agreement with it, and it presented
                // FIFO where the engine rail takes MAILBOX when the surface
                // offers it. A user on the one host that reached it therefore
                // got a measurably different display with no counter saying
                // which rail had drawn it, which is the silent degradation this
                // project does not ship. macOS already refused here for its own
                // reason (a second `VkInstance` on the same `CAMetalLayer` does
                // not fix a MoltenVK surface failure); the two pathways now
                // agree on what an attach failure means.
                //
                // `host_window_engine_attach` is the counter that says the
                // refusal happened, and the presenter's own typed reason
                // (`SwapchainUnavailable`, `QueueCannotPresent`, a `vk_window_*`
                // slug) rides along in it as the detail.
                crate::observe::Emit::decline("host_window_engine_attach", &error).fail();
                eprintln!(
                    "reims-vgpu-window: engine present unavailable ({error}); \
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
                if self.engine_attached {
                    crate::backend::vulkan::engine::window_present_resize(applied.0, applied.1);
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
                        (self.on_input)(HostAction::input_key(evdev, down));
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.pointer_move((position.x, position.y));
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if let Some(btn) = input_map::mouse_button(button) {
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
        if self.engine_attached {
            crate::backend::vulkan::engine::window_present_detach();
            crate::backend::vulkan::engine::note_window_present_attached(false);
            self.engine_attached = false;
        }
        self.window = None;
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        self.loop_census.tick();
        // The device sets `stop` on VM teardown. The loop wakes at least once per
        // [`ENGINE_WINDOW_REDRAW_BACKSTOP`], so the request is picked up within
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
                self.next_backstop = now + ENGINE_WINDOW_REDRAW_BACKSTOP;
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
    fn user_event(&mut self, _event_loop: &ActiveEventLoop, _event: FramePublished) {
        self.frame_pending = true;
    }
}

/// Whether `about_to_wait` should ask the platform for a redraw on this wake.
///
/// The whole rule, and the reason the loop is allowed to sleep: a tick with no
/// publish behind it and time left on the backstop asks for nothing. Under the
/// 2 ms poll this was unconditionally true, which is how 98 % of the redraws
/// this window asked for reached [`needs_engine_present`] only to be refused.
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
    fn new(config: WindowConfig, on_input: InputSink, frames: FrameSlot, stop: StopFlag) -> Self {
        Self {
            config,
            on_input,
            frames,
            stop,
            closed_sent: false,
            window: None,
            cursor: (0, 0),
            engine_attached: false,
            first_engine_present_logged: false,
            first_engine_guest_logged: false,
            engine_error_logged: false,
            // The first tick draws: `engine_redraw_required` is set and there is
            // no publish behind the first frame, which is a clear rather than a
            // guest frame.
            frame_pending: true,
            next_backstop: std::time::Instant::now(),
            last_engine_seq: None,
            engine_redraw_required: true,
            guest_extent: None,
            pending_guest_resize: None,
            loop_census: LoopCensus::new(),
        }
    }

    /// Force the next present and make sure a redraw is asked for to carry it.
    ///
    /// The two flags answer different questions — `engine_redraw_required`
    /// opens [`needs_engine_present`]'s seq gate, `frame_pending` makes
    /// `about_to_wait` talk to the platform — and every caller that wants one
    /// wants both. Setting only the first is the shape that leaves a resized
    /// window blank until the guest happens to publish, because under the
    /// backstop nothing asks for the redraw the flag was waiting for.
    ///
    /// Deliberately not folded into [`redraw_due`]: a standing
    /// `engine_redraw_required` that a `Busy` presenter cannot clear would then
    /// re-request on every wake, which is the spin this loop has already been
    /// bitten by twice. A one-shot flag cannot do that.
    fn force_redraw(&mut self) {
        self.engine_redraw_required = true;
        self.frame_pending = true;
    }

    fn request_shutdown(&mut self) {
        if !self.closed_sent {
            (self.on_input)(HostAction::window_closed());
            self.closed_sent = true;
        }
    }

    fn surface_dims(&self) -> (u32, u32) {
        // The engine presenter owns the swapchain, so the native window is the
        // only thing here that knows the drawable size. Before it opens, the
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
        (self.on_input)(HostAction::input_pointer_move(x, y, width, height));
    }

    /// Present one frame through the engine presenter, or decide there is
    /// nothing to present.
    ///
    /// The guard is teardown, not a rail choice: `exiting` releases the
    /// presenter before the loop ends, and an attach that failed never stored a
    /// window at all. Both already named themselves where they happened, so a
    /// redraw arriving after either is expected control flow and stays quiet.
    fn draw(&mut self) {
        if !self.engine_attached {
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
        if !needs_engine_present(
            self.last_engine_seq,
            self.engine_redraw_required,
            incoming_seq,
        ) {
            self.loop_census.draws_stale += 1;
            return;
        }
        self.loop_census.draws_fresh += 1;
        let result = crate::backend::vulkan::engine::window_present_frame(
            frame.as_ref().and_then(|frame| frame.resident.as_ref()),
            frame.as_deref().map(window_cpu_frame),
        );
        match result {
            Ok(crate::backend::vulkan::engine::WindowPresentOutcome::Busy) => {}
            Ok(crate::backend::vulkan::engine::WindowPresentOutcome::Presented {
                direct,
                width,
                height,
                swapchain_images,
                suboptimal,
            }) => {
                self.engine_error_logged = false;
                self.last_engine_seq = incoming_seq;
                // A suboptimal present armed a swapchain recreation; redraw
                // promptly so the corrected drawable replaces this one even if
                // no new guest frame arrives for seconds. The assignment clears
                // the flag on a healthy present, so this cannot become a
                // self-feeding loop: one suboptimal buys exactly one more draw.
                self.engine_redraw_required = suboptimal;
                if suboptimal {
                    self.force_redraw();
                }
                if !self.first_engine_present_logged {
                    eprintln!(
                        "reims-vgpu-window: first frame presented \
                         ({width}x{height}, {swapchain_images} swapchain images)"
                    );
                    self.first_engine_present_logged = true;
                }
                if direct && frame.is_some() && !self.first_engine_guest_logged {
                    eprintln!(
                        "reims-vgpu-window: first guest frame presented via engine resident \
                         (same-device zero-copy)"
                    );
                    crate::observe::off(
                        "host_window_direct_present path=engine_resident status=live",
                    );
                    self.first_engine_guest_logged = true;
                }
            }
            Err(error) => {
                if !self.engine_error_logged {
                    // The engine present rail's `DrawError` names its own reason
                    // — a `VkCall`'s `vk_window_*` slug, a `DrawReason` refusal,
                    // or `vk_engine_*_untyped` for the not-yet-typed variants.
                    // Emitting it typed keeps that slug the primary `reason=`
                    // rather than nesting it inside a coarse
                    // `reason=engine_resident_present error=...` double-reason.
                    crate::observe::Emit::decline("host_window_present", &error).fail();
                    eprintln!("reims-vgpu-window: engine resident present failed: {error}");
                    self.engine_error_logged = true;
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
        let request = guest_resize_request(self.guest_extent, incoming, actual);
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
            crate::backend::vulkan::engine::window_present_resize(applied.0, applied.1);
            self.engine_redraw_required = true;
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
        let backstop = now + ENGINE_WINDOW_REDRAW_BACKSTOP;
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
        let backstop = now + ENGINE_WINDOW_REDRAW_BACKSTOP;
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
        assert!(redraw_due(false, now + ENGINE_WINDOW_REDRAW_BACKSTOP, now));
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
                backstop = now + ENGINE_WINDOW_REDRAW_BACKSTOP;
            }
        }
        let backstop_ticks = (second.as_millis() / ENGINE_WINDOW_REDRAW_BACKSTOP.as_millis()) as u32;
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
            if needs_engine_present(presented, redraw_required, incoming) {
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
            needs_engine_present(None, true, None),
            "the first present has no frame and must still happen"
        );
        assert!(
            !needs_engine_present(None, false, None),
            "and must not repeat once the flag is cleared"
        );
        assert!(
            needs_engine_present(Some(7), true, Some(7)),
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
            WindowError::AttachEngine("swapchain unavailable".into()),
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
            WindowError::AttachEngine(_) => "AttachEngine",
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
                "host_window_engine_attach",
                &WindowError::AttachEngine("swapchain unavailable".into()),
            )
            .render(),
            "host_window_engine_attach reason=window_attach_engine detail=swapchain_unavailable"
        );
    }

    /// This file types the refusals of a window that has **one** presenter, and
    /// no others.
    ///
    /// An engine-attach failure ends the window on every platform, so nothing
    /// here owns a `VkInstance`, a physical-device choice, a swapchain, a queue
    /// submit or a staging image — and none of those can name a refusal from
    /// this module. While the window carried a second self-contained presenter
    /// it typed 29 further variants (`Vk*` bring-up, `Present*` loop, `Staging*`
    /// upload) for a rail that measured zero presents, drew a stretched rather
    /// than letterboxed picture, and left no counter saying which rail had
    /// drawn. Re-adding a presenter here means re-adding that family, and this
    /// count is what makes that a deliberate act.
    ///
    /// The ten: building the event loop, running it (one variant per entry
    /// point), the three ways the single process window can be claimed by the
    /// wrong device, creating the native window, and the three steps of the
    /// engine attach.
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
            10,
            "WindowError carries {} variants; a presenter-shaped family here is \
             a second rail that no fail line distinguishes",
            names.len()
        );
        // The attach refusal itself must survive under its registered name: it
        // is the counter that tells a reader the window refused rather than
        // quietly drew something else.
        assert!(names.contains("AttachEngine"));
        assert_eq!(
            WindowError::AttachEngine("swapchain unavailable".into()).slug(),
            "window_attach_engine"
        );
    }

    #[test]
    fn engine_present_gate_submits_new_frames_and_forced_redraws_only() {
        assert!(needs_engine_present(None, true, None));
        assert!(!needs_engine_present(Some(7), false, Some(7)));
        assert!(needs_engine_present(Some(7), false, Some(8)));
        assert!(needs_engine_present(Some(7), true, Some(7)));
    }

    #[test]
    fn guest_geometry_change_requests_one_matching_native_resize() {
        // First frame at the window's own size: no request.
        assert!(!guest_resize_request(None, (1920, 1080), (1920, 1080)));
        // Guest mode change away from the window size: request once.
        assert!(guest_resize_request(
            Some((1920, 1080)),
            (1440, 1080),
            (1920, 1080)
        ));
        // Same guest geometry re-observed: no duplicate request.
        assert!(!guest_resize_request(
            Some((1440, 1080)),
            (1440, 1080),
            (1920, 1080)
        ));
        // Window already matches the new mode: nothing to do.
        assert!(!guest_resize_request(
            Some((1920, 1080)),
            (1440, 1080),
            (1440, 1080)
        ));
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
}
