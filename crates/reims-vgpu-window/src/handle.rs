//! [`Window`]: a display's window as the display holds it, its [`title`], and the [`FrameSink`]
//! it draws through.
//!
//! A `Window` is `Send` and `Sync`, and holds no AppKit or SDL object: the display's callbacks
//! tell it what the guest did from the display thread, and its last reference may go on any
//! thread. Its AppKit side (the `NSWindow`, its view and layer, the cursor) lives on the main
//! thread and is released there (`shell`); the render thread's reference to the layer, in the
//! window's surface, is taken back and released there too. On Linux SDL's window lives on the UI
//! thread (`sdl::shell`), and goes once the render thread has let its swapchain go. [`Window::close`] is its end: from then on no
//! callback of the host's is called, and the window closes on the UI thread (the main thread on
//! macOS, the library's `reims-vgpu.ui` on Linux). Nothing in it waits for the UI thread, for its
//! render thread or for the GPU.
//!
//! [`Window::attach`] has the OS's side (`native`) open the window: on macOS it starts the
//! window's render thread (`render`), starts the application once per process (`app`) and posts
//! the window's opening to the main thread; on Linux it starts the window's render thread and
//! posts the opening to the library's UI thread, which SDL runs on (`sdl`). The window opens sized
//! for the guest's mode, else for the first mode the display offers ([`Window::offered_mode`]),
//! which the display tells it before the attach and whenever its modes change. What the guest
//! does later reaches the window's two sides: a new frame wakes the render thread; a mode, a
//! cursor glyph or its visibility is stored here and posted to the UI thread, which resizes
//! the window and lays the view out again, or sets the cursor again. A cursor change posts at
//! most one refresh at a time, which reads the latest state.
//!
//! **The opening's handoff.** What the guest does while the opening is being posted reaches the
//! window either way: the attach marks the opening posted before it posts it, so a mode or a
//! cursor change from then on is posted too. One posted before the opening runs finds no window
//! on the UI thread, and the opening reads the guest's state itself; one posted after it runs
//! behind it. The opening takes the cursor refresh ([`Window::opening`]) before it reads the
//! state, so a refresh that found no window does not keep the next change from posting one.
//!
//! **Who moves the guest's cursor.** The window's pointer events move it, through the host, and
//! over the image the host's cursor takes the guest's glyph, at the host's pointer. At each of
//! the guest's cursor doorbells the window reads the guest's cursor position through its sink
//! ([`Window::cursor_moved`]); a doorbell that leaves it where it was (one rung for a show or a
//! hide, say) moves nothing, and so does the first position read. Where the guest moves its
//! cursor with no pointer event of the window's in the [`POINTER_LEADS_FOR`] before, something
//! else moves it: a mouse the host passed through to the guest, which the host's pointer never
//! sees, or the guest placing its own. The window then draws the guest's cursor into its image
//! itself, at the position it read ([`DrawnCursor`], which the render thread hands the sink with
//! each draw), each move drawing again, and the host's cursor is invisible over the image. The
//! window's next pointer event gives the cursor back to the host's pointer. A move the guest
//! reports soon after a pointer event of the window's is the guest following it, however late its
//! report comes within that time.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use reims_vgpu_core::platform::os::MonoTime;
use reims_vgpu_core::sync::Mutex;

pub use crate::native::FrameSink;

use crate::config::{WindowCallbacks, WindowConfig};
use crate::native::{self, Surface};
use crate::policy;

/// The next window's id.
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// What a window's title starts with.
pub const TITLE: &str = "Reims vGPU";

/// The title of the window of the display on `port`: [`TITLE`], then " (port N)" for a port N
/// other than 0. The display's name is not in it.
pub fn title(port: u64) -> String {
    if port == 0 {
        TITLE.to_owned()
    } else {
        format!("{TITLE} (port {port})")
    }
}

/// A rectangle in a drawable's pixels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PixelRect {
    /// Pixels from the left.
    pub x: u64,
    /// Pixels from the top.
    pub y: u64,
    /// Pixels wide.
    pub width: u64,
    /// Pixels high.
    pub height: u64,
}

/// How long after a pointer event of the window's the guest's cursor moves are taken as following
/// it: a move of the guest's later than that, with none of the window's between, is the guest's
/// own (the module doc). Well past the guest's answer to a move under load, and short enough that
/// a passed-through mouse is followed from its first move once the host's pointer rests.
pub(crate) const POINTER_LEADS_FOR: Duration = Duration::from_secs(1);

/// The guest's side of the window, as the display's callbacks last reported it.
#[derive(Clone, Debug, Default)]
pub(crate) struct GuestState {
    /// The guest's mode, once it has set one.
    pub(crate) mode: Option<(u16, u16)>,
    /// The guest's cursor glyph: width, height, hot spot and RGBA pixels.
    pub(crate) cursor: Option<Arc<Glyph>>,
    /// Whether the guest shows its cursor.
    pub(crate) cursor_visible: bool,
    /// Who moves the guest's cursor.
    pub(crate) lead: CursorLead,
}

impl GuestState {
    /// Whether the host's cursor shows the guest's over the image: while the guest shows its
    /// cursor and the window does not draw it itself.
    pub(crate) fn host_cursor_visible(&self) -> bool {
        self.cursor_visible && !self.lead.guest_leads()
    }
}

/// Who moves the guest's cursor: the window's pointer events, or the guest on its own, and then
/// the window draws the cursor itself (the module doc).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CursorLead {
    /// When the window last sent the host a pointer event, on the monotonic clock.
    pointer_sent: Option<MonoTime>,
    /// The guest moved its cursor on its own, since the window's last pointer event.
    guest_leads: bool,
    /// The guest's cursor position last read, in its mode's pixels.
    at: Option<(u16, u16)>,
}

impl CursorLead {
    /// Whether the guest moves its cursor on its own: the window draws it.
    pub(crate) fn guest_leads(&self) -> bool {
        self.guest_leads
    }

    /// The window sent the host a pointer event at `now`: the host's pointer leads. Whether that
    /// changed who does.
    pub(crate) fn pointer_sent(&mut self, now: MonoTime) -> bool {
        self.pointer_sent = Some(now);
        std::mem::replace(&mut self.guest_leads, false)
    }

    /// The guest's cursor position last read.
    pub(crate) fn at(&self) -> Option<(u16, u16)> {
        self.at
    }

    /// The guest's cursor was read at `at` at `now`: a move where the position last read was
    /// another, which makes the guest lead unless the window sent a pointer event within
    /// [`POINTER_LEADS_FOR`] before. `None` where it did not move (the first read included);
    /// else whether who leads changed.
    pub(crate) fn guest_at(&mut self, now: MonoTime, at: (u16, u16)) -> Option<bool> {
        if self.at.replace(at).is_none_or(|before| before == at) {
            return None;
        }
        let following = self
            .pointer_sent
            .is_some_and(|sent| now.saturating_duration_since(sent) < POINTER_LEADS_FOR);
        if following || self.guest_leads {
            return Some(false);
        }
        self.guest_leads = true;
        Some(true)
    }
}

/// A cursor glyph, copied from the guest's.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Glyph {
    /// Pixels wide.
    pub width: u32,
    /// Pixels high.
    pub height: u32,
    /// The hot spot, in pixels from the top left.
    pub hot_spot: (u16, u16),
    /// `width * 4` bytes a row, `height` rows, RGBA, the alpha straight. A row the bytes do not
    /// reach is transparent.
    pub rgba: Vec<u8>,
}

/// The guest's cursor as the window draws it into its image, where the guest moves it on its own
/// (the module doc): the sink draws `glyph` over the frame, its hot spot at `at` in `mode`'s
/// pixels, a glyph pixel to a guest pixel as the image scales the mode, clipped to the image.
#[derive(Clone, Debug)]
pub struct DrawnCursor {
    /// The guest's glyph.
    pub glyph: Arc<Glyph>,
    /// The guest's mode, whose pixels `at` is in.
    pub mode: (u16, u16),
    /// The guest's cursor position, as the window last read it.
    pub at: (u16, u16),
}

/// What a window posts to its UI thread: `native`'s posts, or a test's.
pub(crate) struct UiPosts {
    /// Starts the window's render thread and posts its opening.
    open: fn(&Arc<Window>),
    /// Posts that the guest of window `id` set a mode.
    guest_mode_changed: fn(u64),
    /// Posts a cursor refresh of window `id`.
    cursor_changed: fn(u64),
    /// Posts the close of window `id`.
    close: fn(u64),
}

/// The OS's posts.
static NATIVE_POSTS: UiPosts = UiPosts {
    open: native::open,
    guest_mode_changed: native::guest_mode_changed,
    cursor_changed: native::cursor_changed,
    close: native::close,
};

/// A display's window.
pub struct Window {
    /// The window's id among the process's, which the UI thread keeps its OS side by.
    id: u64,
    /// The display's port: which host screen the window opens on (`screens`).
    port: u64,
    /// The window's title.
    title: String,
    /// The config's flag.
    match_host_screen: bool,
    /// The config's chosen mode.
    chosen_mode: Option<(u16, u16)>,
    /// The config's density for the chosen mode.
    chosen_ppi: Option<f64>,
    /// The config's `keeps_its_size`.
    keeps_its_size: bool,
    /// The host's callbacks, until the window's end takes them out.
    callbacks: Mutex<Option<WindowCallbacks>>,
    /// What the guest last reported.
    guest: Mutex<GuestState>,
    /// The first mode the display offers the guest, once it offers one.
    offered: Mutex<Option<(u16, u16)>>,
    /// The size in millimetres the display tells the guest, once it is known: what the guest's
    /// scale follows ([`policy::guest_scale`]).
    told_size: Mutex<Option<(f64, f64)>>,
    /// What draws the display's frames, once the display exists.
    sink: Mutex<Option<Arc<dyn FrameSink>>>,
    /// Set at the window's end.
    closed: AtomicBool,
    /// Set at the attach: the window has a render thread, and opens on the UI thread.
    attached: AtomicBool,
    /// Set as the opening is about to be posted to the UI thread: what is posted from then on
    /// finds the window open there, or runs before the opening, which reads it itself.
    open_posted: AtomicBool,
    /// A cursor refresh is posted to the UI thread and has not run yet.
    cursor_posted: AtomicBool,
    /// What the render thread draws with.
    surface: Surface,
    /// What it posts to the UI thread.
    posts: &'static UiPosts,
}

impl std::fmt::Debug for Window {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Window")
            .field("port", &self.port)
            .field("title", &self.title)
            .field("closed", &self.closed.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

impl Window {
    /// The window of the display on `port`, titled `title`. It opens at [`Window::attach`];
    /// until then it only records what the guest reports.
    pub fn new(config: WindowConfig, port: u64, title: String) -> Arc<Window> {
        Window::posting_to(config, port, title, &NATIVE_POSTS)
    }

    /// The window of [`Window::new`], posting through `posts`.
    fn posting_to(
        config: WindowConfig,
        port: u64,
        title: String,
        posts: &'static UiPosts,
    ) -> Arc<Window> {
        Arc::new(Window {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            port,
            title,
            match_host_screen: config.match_host_screen,
            chosen_mode: config.mode,
            chosen_ppi: config.mode_ppi,
            keeps_its_size: config.keeps_its_size,
            callbacks: Mutex::new(Some(config.callbacks)),
            guest: Mutex::new(GuestState::default()),
            offered: Mutex::new(None),
            told_size: Mutex::new(None),
            sink: Mutex::new(None),
            closed: AtomicBool::new(false),
            attached: AtomicBool::new(false),
            open_posted: AtomicBool::new(false),
            cursor_posted: AtomicBool::new(false),
            surface: Surface::new(),
            posts,
        })
    }

    /// The display's port.
    pub fn port(&self) -> u64 {
        self.port
    }

    /// The window's title.
    pub fn title(&self) -> &str {
        &self.title
    }

    /// Whether the config asked for modes and a density matching the host's screen.
    pub fn match_host_screen(&self) -> bool {
        self.match_host_screen
    }

    /// Opens the window over `sink`, the display's: its render thread starts, and the window
    /// opens on the UI thread (on macOS, when the main thread runs its run loop). Does nothing
    /// after the end; a second attach only replaces the sink.
    pub fn attach(self: &Arc<Self>, sink: Arc<dyn FrameSink>) {
        if self.is_closed() {
            return;
        }
        *self.sink.lock() = Some(sink);
        // An end that raced the store may have missed the sink.
        if self.is_closed() {
            drop(self.sink.lock().take());
            return;
        }
        self.open();
    }

    /// Has the OS's side open the window, the first time: its render thread starts, and its
    /// opening is posted to the UI thread, marked posted first (the module doc), so what the
    /// guest does meanwhile is posted too.
    fn open(self: &Arc<Self>) {
        if self.attached.swap(true, Ordering::AcqRel) {
            return;
        }
        self.open_posted.store(true, Ordering::Release);
        (self.posts.open)(self);
    }

    /// The guest presented a frame: the window draws it.
    pub fn new_frame(&self) {
        self.surface.request_draw();
    }

    /// The display offers the guest `width` by `height` first: the window opens sized for it
    /// until the guest sets a mode. It sizes a window that has opened already for nothing.
    pub fn offered_mode(&self, width: u16, height: u16) {
        *self.offered.lock() = Some((width, height));
    }

    /// The guest set a mode `width` by `height`: the image takes its aspect ratio, and a window
    /// that follows the guest's mode is resized for it (`shell`).
    pub fn guest_mode(&self, width: u16, height: u16) {
        self.guest.lock().mode = Some((width, height));
        if self.is_open() {
            (self.posts.guest_mode_changed)(self.id);
        }
    }

    /// The guest set a cursor glyph: `rgba` holds `height` rows of `width * 4` bytes. A cursor
    /// the window draws is drawn again.
    pub fn cursor_glyph(&self, width: u32, height: u32, hot_spot: (u16, u16), rgba: &[u8]) {
        let drawn = {
            let mut guest = self.guest.lock();
            guest.cursor = Some(Arc::new(Glyph {
                width,
                height,
                hot_spot,
                rgba: rgba.to_vec(),
            }));
            guest.lead.guest_leads()
        };
        self.post_cursor_refresh();
        if drawn {
            self.surface.request_draw();
        }
    }

    /// The guest showed (`true`) or hid its cursor. A cursor the window draws is drawn again.
    pub fn cursor_show(&self, show: bool) {
        let drawn = {
            let mut guest = self.guest.lock();
            guest.cursor_visible = show;
            guest.lead.guest_leads()
        };
        self.post_cursor_refresh();
        if drawn {
            self.surface.request_draw();
        }
    }

    /// The guest rang the doorbell that reports its cursor's moves: the window reads its cursor
    /// position through the sink ([`FrameSink::guest_cursor_position`]), on this thread, and
    /// takes it as `Window::cursor_at` does. Nothing before the sink is attached, after the
    /// display has ended, without a guest display, or while it is paused.
    pub fn cursor_moved(&self) {
        let sink = self.sink.lock().clone();
        if let Some(at) = sink.and_then(|sink| sink.guest_cursor_position()) {
            self.cursor_at(at);
        }
    }

    /// The guest's cursor is at `at`: where it moved, with no pointer event of the window's in
    /// the [`POINTER_LEADS_FOR`] before, the window draws the guest's cursor there, and hides the
    /// host's over the image (the module doc). Each move then draws again.
    pub(crate) fn cursor_at(&self, at: (u16, u16)) {
        let (changed, drawn) = {
            let mut guest = self.guest.lock();
            let Some(changed) = guest.lead.guest_at(native::now(), at) else {
                return;
            };
            (changed, guest.lead.guest_leads())
        };
        if changed {
            log::debug!(
                target: "reims_vgpu_window",
                "the guest moves its cursor on its own: the window draws it"
            );
            self.post_cursor_refresh();
        }
        if drawn {
            self.surface.request_draw();
        }
    }

    /// Posts a cursor refresh to the UI thread, unless one is posted and has not run yet: it
    /// reads the latest glyph and visibility when it runs.
    fn post_cursor_refresh(&self) {
        if self.is_open() && !self.cursor_posted.swap(true, Ordering::AcqRel) {
            (self.posts.cursor_changed)(self.id);
        }
    }

    /// The window's end: no callback of the host's runs after it returns, other than one
    /// already running on the UI thread, and the window closes there. Its render thread ends
    /// by itself. Waits for nothing. Later calls do nothing.
    pub fn close(&self) {
        if self.closed.swap(true, Ordering::AcqRel) {
            return;
        }
        let callbacks = self.callbacks.lock().take();
        let sink = self.sink.lock().take();
        self.surface.close();
        if self.attached.load(Ordering::Acquire) {
            (self.posts.close)(self.id);
        }
        drop((callbacks, sink));
    }

    /// Whether the window has ended.
    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }

    /// Whether the window's opening is posted, or about to be, and the window has not ended:
    /// what is posted to the UI thread now runs after the opening and finds the window open, or
    /// before it, finding no window, when the opening reads what it would have told.
    fn is_open(&self) -> bool {
        self.open_posted.load(Ordering::Acquire) && !self.is_closed()
    }

    /// The first mode the display offers, once it offers one.
    pub fn offered(&self) -> Option<(u16, u16)> {
        *self.offered.lock()
    }

    /// The config's chosen mode (`WindowConfig::mode`).
    pub fn chosen_mode(&self) -> Option<(u16, u16)> {
        self.chosen_mode
    }

    /// The config's density for its chosen mode (`WindowConfig::mode_ppi`).
    pub fn chosen_ppi(&self) -> Option<f64> {
        self.chosen_ppi
    }

    /// Whether the window never resizes for the guest's mode (`WindowConfig::keeps_its_size`).
    pub fn keeps_its_size(&self) -> bool {
        self.keeps_its_size
    }

    /// The display tells the guest it is `size_in_millimeters`: the window opens and resizes at
    /// a guest point per host point, the guest's scale for each mode following it.
    pub fn told_size(&self, size_in_millimeters: (f64, f64)) {
        *self.told_size.lock() = Some(size_in_millimeters);
    }
}

/// What the window's backend reads of it.
impl Window {
    /// The window's id among the process's.
    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    /// The host's callbacks, while the window lives.
    pub(crate) fn callbacks(&self) -> Option<WindowCallbacks> {
        self.callbacks.lock().clone()
    }

    /// Hands each of `events`, in order, to `send` with the host's callbacks, while the window
    /// lives: the window's end ([`Window::close`]) drops the events not yet sent, so no callback
    /// of the host's starts after the end has returned, but the one the UI thread was making.
    /// Nothing is sent after the end.
    pub(crate) fn send_each<E>(&self, events: Vec<E>, mut send: impl FnMut(&WindowCallbacks, E)) {
        if events.is_empty() {
            return;
        }
        let Some(callbacks) = self.callbacks() else {
            return;
        };
        for event in events {
            if self.is_closed() {
                return;
            }
            send(&callbacks, event);
        }
    }

    /// The window is about to send the host a pointer event, which moves the guest's cursor: the
    /// host's pointer leads it again, and a cursor the window drew goes (the module doc). On the
    /// UI thread, as the event's callback.
    pub(crate) fn pointer_sent(&self) {
        if self.guest.lock().lead.pointer_sent(native::now()) {
            log::debug!(
                target: "reims_vgpu_window",
                "the window moves the guest's cursor: the host's cursor shows it again"
            );
            self.post_cursor_refresh();
            self.surface.request_draw();
        }
    }

    /// The cursor the render thread has the sink draw into the image: the guest's, where the
    /// guest moves it on its own and shows it, once it has a glyph and a mode; else none.
    pub(crate) fn drawn_cursor(&self) -> Option<DrawnCursor> {
        let guest = self.guest.lock();
        if !(guest.lead.guest_leads() && guest.cursor_visible) {
            return None;
        }
        Some(DrawnCursor {
            glyph: Arc::clone(guest.cursor.as_ref()?),
            mode: guest.mode?,
            at: guest.lead.at()?,
        })
    }

    /// What the guest last reported.
    pub(crate) fn guest(&self) -> GuestState {
        self.guest.lock().clone()
    }

    /// The guest's mode, once it has set one.
    pub(crate) fn mode(&self) -> Option<(u16, u16)> {
        self.guest.lock().mode
    }

    /// The scale the guest runs `mode` at: 2 or 1 by the size the display tells it
    /// ([`policy::guest_scale`]), 1 before the display says.
    pub(crate) fn guest_scale(&self, mode: (u16, u16)) -> u8 {
        self.told_size
            .lock()
            .map_or(1, |size| policy::guest_scale(mode, size))
    }

    /// The sink, copied out of its lock, until the window's end.
    pub(crate) fn sink(&self) -> Option<Arc<dyn FrameSink>> {
        self.sink.lock().clone()
    }

    /// What the render thread draws with.
    pub(crate) fn surface(&self) -> &Surface {
        &self.surface
    }

    /// A posted cursor refresh runs: a later change posts another.
    pub(crate) fn take_cursor_refresh(&self) {
        self.cursor_posted.store(false, Ordering::Release);
    }

    /// The UI thread runs the window's opening, before the window opens there and reads the
    /// guest's state: the opening is a cursor refresh too, so a refresh posted before it, which
    /// found no window, does not keep a later change from posting one (the module doc).
    pub(crate) fn opening(&self) {
        self.take_cursor_refresh();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// "Reims vGPU", then the port when it is not 0
    #[test]
    fn a_window_s_title_is_reims_vgpu_and_its_port() {
        assert_eq!(title(0), "Reims vGPU");
        assert_eq!(title(1), "Reims vGPU (port 1)");
        assert_eq!(title(7), "Reims vGPU (port 7)");
    }

    /// Events go to the host's callbacks in order while the window lives; its end, inside the
    /// callback of one of them, drops the rest, and after it nothing is sent
    #[test]
    fn nothing_is_sent_after_the_window_s_end() {
        let window = Window::new(WindowConfig::default(), 0, title(0));
        let mut sent = Vec::new();
        window.send_each(vec![1, 2], |_, event| sent.push(event));
        assert_eq!(sent, [1, 2]);
        window.send_each(vec![3, 4, 5], |_, event| {
            sent.push(event);
            window.close();
        });
        assert_eq!(sent, [1, 2, 3]);
        window.send_each(vec![6], |_, event| sent.push(event));
        assert_eq!(sent, [1, 2, 3]);
    }

    /// What the display thread does, in a test.
    type Act = Box<dyn FnOnce()>;

    /// Work posted to a test's UI thread.
    enum Job {
        Open(std::sync::Weak<Window>),
        Mode(u64),
        Cursor(u64),
    }

    /// A test's UI thread: what is posted to it, run when the test says, and what the window
    /// open there last read of the guest, as the shells read it.
    #[derive(Default)]
    struct FakeUi {
        /// What is posted and has not run.
        jobs: std::collections::VecDeque<Job>,
        /// The window open there.
        open: Option<Arc<Window>>,
        /// The guest's mode as the window last read it.
        mode: Option<(u16, u16)>,
        /// The guest's cursor glyph as the window last read it.
        cursor: Option<Arc<Glyph>>,
        /// What the display thread does while the opening is posted: before it is queued, and
        /// once the UI thread has run it, before the attach returns.
        meanwhile: Option<(Act, Act)>,
    }

    thread_local! {
        static UI: std::cell::RefCell<FakeUi> = std::cell::RefCell::default();
    }

    /// Runs what is posted to the test's UI thread, in order, as the shells would.
    fn run_ui() {
        while let Some(job) = UI.with(|ui| ui.borrow_mut().jobs.pop_front()) {
            let open = UI.with(|ui| ui.borrow().open.clone());
            let open = |id| open.filter(|w| w.id() == id);
            match job {
                Job::Open(window) => {
                    let Some(window) = window.upgrade() else {
                        continue;
                    };
                    window.opening();
                    let (mode, cursor) = (window.mode(), window.guest().cursor);
                    UI.with(|ui| {
                        let mut ui = ui.borrow_mut();
                        (ui.mode, ui.cursor) = (mode, cursor);
                        ui.open = Some(window);
                    });
                }
                Job::Mode(id) => {
                    if let Some(window) = open(id) {
                        UI.with(|ui| ui.borrow_mut().mode = window.mode());
                    }
                }
                Job::Cursor(id) => {
                    if let Some(window) = open(id) {
                        window.take_cursor_refresh();
                        UI.with(|ui| ui.borrow_mut().cursor = window.guest().cursor);
                    }
                }
            }
        }
    }

    fn post(job: Job) {
        UI.with(|ui| ui.borrow_mut().jobs.push_back(job));
    }

    /// The test's posts: the opening runs at once, with the display thread's work around it.
    static TEST_POSTS: UiPosts = UiPosts {
        open: |window| {
            let meanwhile = UI.with(|ui| ui.borrow_mut().meanwhile.take());
            let (before, after) = meanwhile.unzip();
            if let Some(before) = before {
                before();
            }
            post(Job::Open(Arc::downgrade(window)));
            run_ui();
            if let Some(after) = after {
                after();
            }
        },
        guest_mode_changed: |id| post(Job::Mode(id)),
        cursor_changed: |id| post(Job::Cursor(id)),
        close: |_| {},
    };

    /// A glyph told apart by its width.
    fn glyph(width: u32) -> Glyph {
        Glyph {
            width,
            height: 1,
            hot_spot: (0, 0),
            rgba: vec![0; width as usize * 4],
        }
    }

    /// The guest's state as the open window last read it.
    fn read() -> (Option<(u16, u16)>, Option<u32>) {
        UI.with(|ui| {
            let ui = ui.borrow();
            (ui.mode, ui.cursor.as_ref().map(|g| g.width))
        })
    }

    /// A mode or a cursor the guest sets while the opening is being posted reaches the window:
    /// before it runs, the opening reads it; after, it is posted behind the opening; and a
    /// cursor refresh that ran before the opening, finding no window, leaves the next change
    /// its own refresh
    #[test]
    fn what_the_guest_does_during_the_opening_reaches_the_window() {
        let window = Window::posting_to(WindowConfig::default(), 0, title(0), &TEST_POSTS);
        let set = |window: &Arc<Window>, mode: (u16, u16), cursor: u32| {
            let g = glyph(cursor);
            window.cursor_glyph(g.width, g.height, g.hot_spot, &g.rgba);
            window.guest_mode(mode.0, mode.1);
        };
        let (before, after) = (Arc::clone(&window), Arc::clone(&window));
        UI.with(|ui| {
            ui.borrow_mut().meanwhile = Some((
                Box::new(move || set(&before, (800, 600), 1)),
                Box::new(move || set(&after, (1280, 800), 2)),
            ));
        });
        window.open();
        run_ui();
        assert_eq!(read(), (Some((1280, 800)), Some(2)));
        set(&window, (1024, 768), 3);
        run_ui();
        assert_eq!(read(), (Some((1024, 768)), Some(3)));
        window.close();
    }

    /// The guest leads its cursor when it moves it with no pointer event of the window's in the
    /// time before; a move soon after one is the guest following it; the window's next pointer
    /// event leads again. The first position read and a read of the same position are no moves.
    /// Each move's answer says whether who leads changed
    #[test]
    fn the_guest_leads_its_cursor_when_nothing_of_the_window_s_moved_it() {
        let t0 = MonoTime::from_nanos(1_000);
        let mut lead = CursorLead::default();
        assert_eq!(
            lead.guest_at(t0, (5, 5)),
            None,
            "the first read moves nothing"
        );
        assert_eq!(lead.guest_at(t0, (5, 5)), None, "nor does the same place");
        assert!(!lead.guest_leads());
        assert_eq!(
            lead.guest_at(t0, (6, 5)),
            Some(true),
            "no pointer event yet: the guest's own"
        );
        let later = t0.saturating_add(POINTER_LEADS_FOR);
        assert_eq!(lead.guest_at(later, (7, 5)), Some(false));
        assert!(lead.guest_leads() && lead.at() == Some((7, 5)));
        let t1 = t0.saturating_add(2 * POINTER_LEADS_FOR);
        assert!(lead.pointer_sent(t1), "the window's pointer leads again");
        assert!(!lead.guest_leads() && !lead.pointer_sent(t1));
        let soon = t1.saturating_add(POINTER_LEADS_FOR - Duration::from_millis(1));
        assert_eq!(
            lead.guest_at(soon, (8, 5)),
            Some(false),
            "the guest following the pointer"
        );
        assert!(!lead.guest_leads());
        let idle = t1.saturating_add(2 * POINTER_LEADS_FOR);
        assert_eq!(
            lead.guest_at(idle, (8, 5)),
            None,
            "a doorbell that moves nothing"
        );
        assert!(!lead.guest_leads());
        assert_eq!(
            lead.guest_at(idle, (9, 5)),
            Some(true),
            "the guest's own again"
        );
        assert!(lead.guest_leads());
    }

    /// A sink that answers only the guest's cursor position, as the test sets it.
    struct PositionSink(Mutex<Option<(u16, u16)>>);

    #[cfg(target_os = "macos")]
    impl FrameSink for PositionSink {
        fn device(
            &self,
        ) -> Option<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>>>
        {
            None
        }

        fn draw(
            &self,
            _drawable: &objc2::runtime::ProtocolObject<dyn objc2_quartz_core::CAMetalDrawable>,
            _region: PixelRect,
            _cursor: Option<&DrawnCursor>,
        ) -> bool {
            false
        }

        fn guest_cursor_position(&self) -> Option<(u16, u16)> {
            *self.0.lock()
        }
    }

    #[cfg(target_os = "linux")]
    impl FrameSink for PositionSink {
        fn presenter(
            &self,
            _target: &dyn crate::VulkanTarget,
        ) -> Option<Box<dyn crate::Presenter>> {
            None
        }

        fn guest_cursor_position(&self) -> Option<(u16, u16)> {
            *self.0.lock()
        }
    }

    /// At each doorbell the window reads the guest's cursor position through its sink: none
    /// reads nothing, the first position is no move, and a new one is
    #[test]
    fn a_doorbell_reads_the_guest_s_cursor_through_the_sink() {
        let window = Window::new(WindowConfig::default(), 0, title(0));
        let sink = Arc::new(PositionSink(Mutex::new(None)));
        *window.sink.lock() = Some(Arc::clone(&sink) as Arc<dyn FrameSink>);
        window.cursor_moved();
        assert_eq!(window.guest().lead.at(), None);
        *sink.0.lock() = Some((3, 4));
        window.cursor_moved();
        assert_eq!(window.guest().lead.at(), Some((3, 4)));
        assert!(!window.guest().lead.guest_leads());
        *sink.0.lock() = Some((5, 4));
        window.cursor_moved();
        assert!(window.guest().lead.guest_leads());
        window.close();
    }

    /// While the guest moves its cursor on its own, the window draws it, with its glyph, in the
    /// guest's mode, and the host's cursor is invisible over the image; a hidden guest cursor is
    /// drawn by neither; the window's pointer event gives it back to the host's cursor
    #[test]
    fn a_cursor_the_guest_moves_on_its_own_is_the_window_s_to_draw() {
        let window = Window::new(WindowConfig::default(), 0, title(0));
        let g = glyph(4);
        window.cursor_show(true);
        window.cursor_moved();
        assert_eq!(window.guest().lead.at(), None, "no sink, no position");
        window.cursor_at((10, 20));
        window.cursor_at((11, 20));
        assert!(window.drawn_cursor().is_none(), "no glyph or mode yet");
        window.cursor_glyph(g.width, g.height, g.hot_spot, &g.rgba);
        window.guest_mode(800, 600);
        let drawn = window.drawn_cursor().unwrap();
        assert_eq!((*drawn.glyph).clone(), g);
        assert_eq!((drawn.mode, drawn.at), ((800, 600), (11, 20)));
        assert!(!window.guest().host_cursor_visible());
        window.cursor_show(false);
        assert!(window.drawn_cursor().is_none());
        assert!(!window.guest().host_cursor_visible());
        window.cursor_show(true);
        window.pointer_sent();
        assert!(window.drawn_cursor().is_none());
        assert!(window.guest().host_cursor_visible());
        window.cursor_at((12, 20));
        assert!(
            window.drawn_cursor().is_none(),
            "the guest following the pointer"
        );
        window.close();
    }
}
