//! The UI thread on Linux, `reims-vgpu.ui`: SDL's thread, the work posted to it, and the host's
//! screens as SDL lists them.
//!
//! It starts at the first window's opening or the first look at the host's screens, and runs for
//! the process: SDL's video is initialized there (so SDL takes it for its main thread) and every
//! SDL call but a render thread's Vulkan surface is made there. Work is posted to it ([`post`]):
//! queued, then run in the order posted, between its events, with an event of its own pushed to
//! wake it. It waits in SDL for the next event, so it costs nothing while nothing happens. A
//! window's input reaches the host's callbacks here, one at a time.
//!
//! **SDL's settings**, before its video starts: no signal handlers of its own (the host's stay
//! the host's), the screen saver allowed (the windows keep the host from sleeping no more than
//! any other window), `reims-vgpu` the windows' application ID (the task bar's entry on Wayland),
//! cursors given for the screen's scale ([`crate::sdl::cursor`]), and no quit event when the last
//! window closes, as the windows' end is the display's.
//!
//! Where SDL's video does not start (no Wayland compositor and no X server to reach), the reason
//! is logged once, no window opens, the work posted is dropped, and the host's screens are none:
//! a display matching its host screen then offers what `crate::policy` gives without a screen.
//!
//! **The host's screens** ([`screens`]): a snapshot the UI thread takes once SDL's video starts
//! and again when SDL reports a screen added, removed, moved or changed. The main screen (SDL's
//! primary display) comes first, then the others top to bottom, left to right. A look before the
//! first snapshot starts the UI thread and waits for it, up to [`FIRST_SNAPSHOT`]; on the UI
//! thread itself it never waits.

use std::cell::Cell;
use std::collections::VecDeque;
use std::ffi::CStr;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use reims_vgpu_core::rt::spawn_named;
use reims_vgpu_core::sync::{Condvar, Mutex};
use reims_vgpu_vulkan::os::OsThreads;
use sdl3_sys::everything::*;

use crate::screens::HostScreen;
use crate::sdl::sdl_error;

/// The UI thread's stack.
const STACK: usize = 1024 * 1024;

/// How long a look at the host's screens waits for the UI thread's first snapshot.
const FIRST_SNAPSHOT: Duration = Duration::from_secs(1);

/// How long the UI thread waits in SDL at most before it looks at its work again, in ms.
const WAIT_MS: i32 = 1000;

/// Work for the UI thread.
type Job = Box<dyn FnOnce() + Send>;

/// Where SDL's video is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    /// No UI thread yet.
    Idle,
    /// The UI thread is starting SDL's video.
    Starting,
    /// SDL's video runs: work posted wakes the UI thread.
    Running,
    /// SDL's video did not start: work posted is dropped.
    Failed,
}

/// The work posted and not run yet, with where SDL's video is.
struct Queue {
    phase: Phase,
    jobs: VecDeque<Job>,
}

/// The queue.
static QUEUE: Mutex<Queue> = Mutex::new(Queue {
    phase: Phase::Idle,
    jobs: VecDeque::new(),
});

/// The event type the UI thread is woken with, once SDL has given one.
static WAKE: AtomicU32 = AtomicU32::new(0);

/// The last snapshot of the host's screens, once taken.
static SCREENS: Mutex<Option<Vec<HostScreen>>> = Mutex::new(None);
/// Signalled when a snapshot is taken.
static SCREENS_TAKEN: Condvar = Condvar::new();

thread_local! {
    /// Whether this thread is the UI thread.
    static ON_UI: Cell<bool> = const { Cell::new(false) };
}

/// Whether the caller runs on the UI thread.
pub(crate) fn on_ui_thread() -> bool {
    ON_UI.with(Cell::get)
}

/// Starts the UI thread, once.
pub(crate) fn start() {
    {
        let mut queue = QUEUE.lock();
        if queue.phase != Phase::Idle {
            return;
        }
        queue.phase = Phase::Starting;
    }
    spawn_named(
        &OsThreads::without_hints(),
        "reims-vgpu.ui",
        STACK,
        Box::new(()),
        run,
    );
}

/// Posts `job` to the UI thread, starting it where it has not started: it runs after the work
/// posted before it. Dropped where SDL's video did not start.
pub(crate) fn post(job: impl FnOnce() + Send + 'static) {
    start();
    let mut queue = QUEUE.lock();
    match queue.phase {
        Phase::Failed => {}
        Phase::Idle | Phase::Starting => queue.jobs.push_back(Box::new(job)),
        Phase::Running => {
            queue.jobs.push_back(Box::new(job));
            drop(queue);
            wake();
        }
    }
}

/// Wakes the UI thread from its wait in SDL.
fn wake() {
    let kind = WAKE.load(Ordering::Acquire);
    if kind == 0 {
        return;
    }
    // SAFETY: an all-zero event is a valid one to fill; SDL copies it into its queue, from any
    // thread.
    unsafe {
        let mut event: SDL_Event = std::mem::zeroed();
        event.r#type = kind;
        SDL_PushEvent(&mut event);
    }
}

/// The host's screens (the module doc): the snapshot, waited for up to [`FIRST_SNAPSHOT`] where
/// none is taken yet, off the UI thread. Empty where none comes.
pub(crate) fn screens() -> Vec<HostScreen> {
    start();
    let mut screens = SCREENS.lock();
    if screens.is_none() && !on_ui_thread() {
        SCREENS_TAKEN.wait_for(&mut screens, FIRST_SNAPSHOT);
    }
    screens.clone().unwrap_or_default()
}

/// Sets SDL's settings (the module doc).
fn set_hints() {
    let hints: [(&CStr, &CStr); 6] = [
        (c"SDL_NO_SIGNAL_HANDLERS", c"1"),
        (c"SDL_VIDEO_ALLOW_SCREENSAVER", c"1"),
        (c"SDL_APP_ID", c"reims-vgpu"),
        (c"SDL_MOUSE_DPI_SCALE_CURSORS", c"1"),
        (c"SDL_QUIT_ON_LAST_WINDOW_CLOSE", c"0"),
        (c"SDL_MOUSE_FOCUS_CLICKTHROUGH", c"1"),
    ];
    for (name, value) in hints {
        // SAFETY: both are NUL-terminated strings SDL copies.
        unsafe { SDL_SetHint(name.as_ptr(), value.as_ptr()) };
    }
}

/// Starts SDL's video: whether it started.
fn start_video() -> bool {
    set_hints();
    // SAFETY: SDL's video starts on this thread, which becomes its main thread; only the hints,
    // which SDL takes on any thread before it starts, came first.
    if !unsafe { SDL_Init(SDL_INIT_VIDEO) } {
        log::warn!(
            target: "reims_vgpu_window",
            "SDL's video does not start ({}): no window opens",
            sdl_error()
        );
        return false;
    }
    // SAFETY: SDL's video runs.
    let kind = unsafe { SDL_RegisterEvents(1) };
    WAKE.store(kind, Ordering::Release);
    // SAFETY: SDL's video runs; the driver's name is a string SDL keeps.
    let driver = unsafe {
        let name = SDL_GetCurrentVideoDriver();
        if name.is_null() {
            String::new()
        } else {
            CStr::from_ptr(name).to_string_lossy().into_owned()
        }
    };
    log::info!(
        target: "reims_vgpu_window",
        "windows open through SDL's {driver} driver{}",
        driver_note(&driver, std::env::var_os("WAYLAND_DISPLAY").is_some())
    );
    true
}

/// What the log adds to SDL's `driver`: where a Wayland compositor runs (`wayland_session`) and SDL
/// took X11, that the windows are XWayland's, and why (`super::sdl`'s module doc).
fn driver_note(driver: &str, wayland_session: bool) -> &'static str {
    if driver == "x11" && wayland_session {
        ", through XWayland: SDL takes it where the Wayland compositor has no fifo-v1 \
         (SDL_VIDEO_DRIVER=wayland asks for Wayland)"
    } else {
        ""
    }
}

/// The UI thread's body.
fn run() {
    ON_UI.with(|on| on.set(true));
    if !start_video() {
        QUEUE.lock().phase = Phase::Failed;
        drop(std::mem::take(&mut QUEUE.lock().jobs));
        *SCREENS.lock() = Some(Vec::new());
        SCREENS_TAKEN.notify_all();
        return;
    }
    snapshot_screens();
    QUEUE.lock().phase = Phase::Running;
    loop {
        run_jobs();
        // SAFETY: an all-zero event is a valid one for SDL to fill.
        let mut event: SDL_Event = unsafe { std::mem::zeroed() };
        // SAFETY: SDL's video runs on this thread.
        if unsafe { SDL_WaitEventTimeout(&mut event, WAIT_MS) } {
            dispatch(&event);
            // SAFETY: as above.
            while unsafe { SDL_PollEvent(&mut event) } {
                dispatch(&event);
            }
            crate::sdl::shell::batch_over();
        }
    }
}

/// Runs the work posted, in order, until there is none.
fn run_jobs() {
    loop {
        let Some(job) = QUEUE.lock().jobs.pop_front() else {
            return;
        };
        job();
    }
}

/// Hands `event` to whatever it is about.
fn dispatch(event: &SDL_Event) {
    // SAFETY: every event starts with its type.
    let kind = unsafe { event.r#type };
    if kind == WAKE.load(Ordering::Relaxed) {
        return;
    }
    let display = SDL_EVENT_DISPLAY_FIRST.0..=SDL_EVENT_DISPLAY_LAST.0;
    if display.contains(&kind) {
        snapshot_screens();
        crate::sdl::shell::screens_changed();
        return;
    }
    crate::sdl::shell::event(event);
}

/// Takes a snapshot of the host's screens (the module doc).
fn snapshot_screens() {
    let screens = list_screens();
    log::debug!(target: "reims_vgpu_window", "host screens: {screens:?}");
    *SCREENS.lock() = Some(screens);
    SCREENS_TAKEN.notify_all();
}

/// The host's screens as SDL lists them, the main one first (the module doc).
fn list_screens() -> Vec<HostScreen> {
    let mut count = 0;
    // SAFETY: SDL's video runs on this thread; the list is SDL's to free.
    let list = unsafe { SDL_GetDisplays(&mut count) };
    if list.is_null() {
        return Vec::new();
    }
    let count = usize::try_from(count).unwrap_or(0);
    // SAFETY: SDL returned `count` display IDs at `list`.
    let ids: Vec<SDL_DisplayID> = unsafe { std::slice::from_raw_parts(list, count) }.to_vec();
    // SAFETY: the list is SDL's allocation, copied out above.
    unsafe { SDL_free(list.cast()) };
    // SAFETY: SDL's video runs on this thread.
    let primary = unsafe { SDL_GetPrimaryDisplay() };
    let mut screens: Vec<(bool, (i32, i32), HostScreen)> = ids
        .into_iter()
        .filter_map(|id| {
            let screen = host_screen(id)?;
            let mut bounds = SDL_Rect {
                x: 0,
                y: 0,
                w: 0,
                h: 0,
            };
            // SAFETY: as above.
            unsafe { SDL_GetDisplayBounds(id, &mut bounds) };
            Some((id != primary, (bounds.y, bounds.x), screen))
        })
        .collect();
    screens.sort_by_key(|&(other, at, _)| (other, at));
    screens.into_iter().map(|(_, _, s)| s).collect()
}

/// Screen `id`'s size in pixels, scale and refresh rate (its current mode's,
/// `crate::screens::screen_refresh`), or `None` where SDL gives no current mode.
fn host_screen(id: SDL_DisplayID) -> Option<HostScreen> {
    // SAFETY: SDL's video runs on this thread; the mode is SDL's, valid until the display
    // changes, and read at once.
    let mode = unsafe { SDL_GetCurrentDisplayMode(id).as_ref() }?;
    // SAFETY: as above.
    let content = unsafe { SDL_GetDisplayContentScale(id) };
    let density = f64::from(mode.pixel_density);
    let scale = crate::sdl::units::pixels_per_point(mode.pixel_density, content);
    let pixels = (mode_pixels(mode.w, density), mode_pixels(mode.h, density));
    let mut fullscreen = Vec::new();
    let mut count = 0;
    // SAFETY: as above; the list is SDL's to free.
    let modes = unsafe { SDL_GetFullscreenDisplayModes(id, &mut count) };
    if !modes.is_null() {
        let count = usize::try_from(count).unwrap_or(0);
        // SAFETY: SDL returned `count` mode pointers at `modes`.
        for &m in unsafe { std::slice::from_raw_parts(modes, count) } {
            // SAFETY: each is a mode SDL keeps until the list is freed.
            if let Some(m) = unsafe { m.as_ref() } {
                let density = f64::from(m.pixel_density);
                let size = (mode_pixels(m.w, density), mode_pixels(m.h, density));
                fullscreen.push((size, f64::from(m.refresh_rate)));
            }
        }
        // SAFETY: the list is SDL's allocation.
        unsafe { SDL_free(modes.cast()) };
    }
    Some(HostScreen {
        id: id.0,
        pixels,
        scale,
        refresh_hz: crate::screens::screen_refresh(
            pixels,
            f64::from(mode.refresh_rate),
            fullscreen,
        ),
    })
}

/// The pixels of a side of an SDL display mode, `points` long at `density` pixels per point
/// (SDL's points are pixels where the density is 1 or less).
fn mode_pixels(points: i32, density: f64) -> u32 {
    (f64::from(points) * density.max(1.0)).round() as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// X11 in a Wayland session is named XWayland, with why SDL took it; otherwise the driver
    /// stands alone
    #[test]
    fn x11_in_a_wayland_session_says_xwayland() {
        assert!(driver_note("x11", true).starts_with(", through XWayland"));
        assert_eq!(driver_note("x11", false), "");
        assert_eq!(driver_note("wayland", true), "");
    }
}
