//! [`Display`]: a display on one of a device's ports, made from a [`DisplayConfig`].
//!
//! A display's callbacks run on the process's display thread ([`crate::platform`]), one at a
//! time across every display of the process, in the order core queued them. A job still reads
//! its callback when it runs, as it did on the host's queue: a display released with a job
//! queued dies at that job. The last drop of a display is its unplug, run on the releasing
//! thread; it may also be the device's last reference.
//!
//! A display created with a window ([`DisplayConfig::window`]) has all five callbacks as the
//! guest reads them, each telling the window before it calls the host's; its last reference
//! closes the window before it lets the display go ([`crate::window`]). Without a window, the
//! display is as its config says.

use std::sync::Arc;

use reims_vgpu_core::display::descriptor::DisplayDescriptor as CoreDescriptor;
use reims_vgpu_core::display::host_calls::CursorGlyph as CoreGlyph;
use reims_vgpu_core::display::mode::DisplayMode as CoreMode;
use reims_vgpu_core::display::{DisplayCoord, DisplayHandle, DisplayName};
use reims_vgpu_core::platform::os::Preferences;
use reims_vgpu_core::platform::types::{MtlOrigin, MtlRegion, MtlSize};
use reims_vgpu_core::platform::{Handle as _, Platform};
use reims_vgpu_core::raise::{OnFault, RaiseResult};
use reims_vgpu_core::rt::Hold;
use reims_vgpu_window::{Window, WindowConfig};

use crate::DefaultPlatform;
use crate::device::{Device, GuestMemoryLease};
use crate::entry;
use crate::error::Error;
use crate::frame::{Frame, FrameTarget, Region};
use crate::platform::{ApiPlatform, api_state};
use crate::retained::Ours;
use crate::stats;
use crate::window::{self, WindowLink};

/// The settings [`DisplayConfig::apply_preferences`] asks its reader for.
const MIN_NITS_KEY: &str = "PGDisplayMinNits";
const MAX_NITS_KEY: &str = "PGDisplayMaxNits";
const MAX_SDR_NITS_KEY: &str = "PGDisplayMaxSDRNits";
/// The settings [`DisplayMode::with_preferences`] asks its reader for.
const SUPPORTS_EDR_KEY: &str = "PGDisplaySupportsEDR";
const SUPPORTS_10BPC_KEY: &str = "PGDisplaySupports10bpc";

/// A raise that reaches an entry point: nothing catches it there, so it ends the process.
pub(crate) fn no_catch<T>(r: RaiseResult<T>) -> T {
    r.unwrap_or_else(|raised| raised.fatal())
}

/// A display mode offered to the guest.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DisplayMode {
    /// Pixels wide.
    pub width: u16,
    /// Pixels high.
    pub height: u16,
    /// The refresh rate, as the guest reads it.
    pub refresh_rate_hz: f64,
    /// The guest reads that the mode supports EDR.
    pub supports_edr: bool,
    /// The guest reads that the mode supports 10 bits per channel.
    pub supports_10bpc: bool,
}

impl DisplayMode {
    /// The size and rate, both flags off.
    pub fn new(width: u16, height: u16, refresh_rate_hz: f64) -> DisplayMode {
        DisplayMode {
            width,
            height,
            refresh_rate_hz,
            supports_edr: false,
            supports_10bpc: false,
        }
    }

    /// The size and rate, then the flags `prefs` answers when the mode is made:
    /// `PGDisplaySupportsEDR`, then `PGDisplaySupports10bpc`, each set when the key exists and
    /// is true.
    pub fn with_preferences(
        prefs: &dyn Preferences,
        width: u16,
        height: u16,
        refresh_rate_hz: f64,
    ) -> DisplayMode {
        let set = |key| prefs.app_bool(key) == (true, true);
        DisplayMode {
            supports_edr: set(SUPPORTS_EDR_KEY),
            supports_10bpc: set(SUPPORTS_10BPC_KEY),
            ..DisplayMode::new(width, height, refresh_rate_hz)
        }
    }

    /// Core's mode.
    fn to_core(self) -> CoreMode {
        CoreMode {
            size_in_pixels: DisplayCoord {
                x: self.width,
                y: self.height,
            },
            refresh_rate_in_hz: self.refresh_rate_hz,
            supports_edr: self.supports_edr,
            supports_10bpc: self.supports_10bpc,
        }
    }
}

/// The cursor glyph, borrowed for the callback.
#[derive(Clone, Copy, Debug)]
pub struct CursorGlyph<'a> {
    /// Pixels wide.
    pub width: u32,
    /// Pixels high.
    pub height: u32,
    /// The hot spot, in pixels from the top left.
    pub hot_spot: (u16, u16),
    /// `width * 4` bytes a row, `height` rows: the guest's BGRA with red and blue swapped.
    pub rgba: &'a [u8],
}

/// `(width, height, pixel_format)`: the guest set a mode.
pub type ModeChangeFn = Arc<dyn Fn(u16, u16, u32) + Send + Sync>;
/// The guest presented a frame since the last call (calls coalesce).
pub type NewFrameFn = Arc<dyn Fn() + Send + Sync>;
/// The guest set a cursor glyph.
pub type CursorGlyphFn = Arc<dyn Fn(&CursorGlyph<'_>) + Send + Sync>;
/// The guest showed or hid the cursor: the guest's byte as written (0 hides).
pub type CursorShowFn = Arc<dyn Fn(u8) + Send + Sync>;
/// The guest moved the cursor.
pub type CursorMoveFn = Arc<dyn Fn() + Send + Sync>;

/// A display's callbacks, each on the display thread. `None` is a callback the host does not
/// give (each C typedef's comment says what the guest then sees).
#[derive(Clone, Default)]
pub struct DisplayCallbacks {
    /// The guest set a mode.
    pub mode_change: Option<ModeChangeFn>,
    /// The guest presented a frame. Without it, the display's first present ends the process.
    pub new_frame: Option<NewFrameFn>,
    /// The guest set a cursor glyph. Without it, the guest reads a glyph size limit of 0.
    pub cursor_glyph: Option<CursorGlyphFn>,
    /// The guest showed or hid the cursor.
    pub cursor_show: Option<CursorShowFn>,
    /// The guest moved the cursor.
    pub cursor_move: Option<CursorMoveFn>,
}

impl std::fmt::Debug for DisplayCallbacks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DisplayCallbacks")
            .field("mode_change", &self.mode_change.is_some())
            .field("new_frame", &self.new_frame.is_some())
            .field("cursor_glyph", &self.cursor_glyph.is_some())
            .field("cursor_show", &self.cursor_show.is_some())
            .field("cursor_move", &self.cursor_move.is_some())
            .finish()
    }
}

/// A display's config.
#[derive(Clone)]
pub struct DisplayConfig {
    /// The port, below the device's port count.
    pub port: u64,
    /// The guest's serial number for the display, unique in the process.
    pub serial_number: u32,
    /// `None` is "Apple Virtual".
    pub name: Option<String>,
    /// Width, then height.
    pub size_in_millimeters: (f64, f64),
    /// The callbacks.
    pub callbacks: DisplayCallbacks,
    /// `None` is no window. Otherwise the library opens a window for the display, draws each
    /// new frame into it and sends its input to the config's callbacks. The display then has
    /// every callback as the guest reads them, and calls those of `callbacks` it has. With
    /// `match_host_screen`, the display offers the modes, size and scale factor of the window's
    /// host screen, where the platform names one, in place of `size_in_millimeters`.
    pub window: Option<WindowConfig>,
    /// What core's descriptor holds beyond the fields above, at its defaults (the colour
    /// space, the origin, the nits).
    hidden: CoreDescriptor,
}

impl std::fmt::Debug for DisplayConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DisplayConfig")
            .field("port", &self.port)
            .field("serial_number", &self.serial_number)
            .field("name", &self.name)
            .field("size_in_millimeters", &self.size_in_millimeters)
            .field("callbacks", &self.callbacks)
            .field("window", &self.window)
            .finish_non_exhaustive()
    }
}

impl DisplayConfig {
    /// A display on `port` with `serial_number`, no name ("Apple Virtual"), a zero size, no
    /// callbacks and no window.
    pub fn new(port: u64, serial_number: u32) -> DisplayConfig {
        DisplayConfig {
            port,
            serial_number,
            name: None,
            size_in_millimeters: (0.0, 0.0),
            callbacks: DisplayCallbacks::default(),
            window: None,
            hidden: CoreDescriptor::sdk_defaults(),
        }
    }

    /// Reads the nits settings `prefs` answers now: `PGDisplayMinNits`, `PGDisplayMaxNits` and
    /// `PGDisplayMaxSDRNits`, in that order, each stored when it is a number.
    pub fn apply_preferences(&mut self, prefs: &dyn Preferences) {
        let h = &mut self.hidden;
        for (key, nits) in [
            (MIN_NITS_KEY, &mut h.min_nits),
            (MAX_NITS_KEY, &mut h.max_nits),
            (MAX_SDR_NITS_KEY, &mut h.max_sdr_nits),
        ] {
            if let Some(value) = prefs.any_user_f32(key) {
                *nits = value;
            }
        }
    }

    /// Core's descriptor, with `queue` as the display's queue: closures that call the host's.
    fn into_core(self, queue: Arc<dyn reims_vgpu_core::host::HostQueue>) -> CoreDescriptor {
        let DisplayCallbacks {
            mode_change,
            new_frame,
            cursor_glyph,
            cursor_show,
            cursor_move,
        } = self.callbacks;
        let mut d = self.hidden;
        if let Some(name) = self.name {
            d.name = Some(DisplayName::new(name));
        }
        d.size_in_millimeters = self.size_in_millimeters;
        d.queue = Some(queue);
        d.mode_change_handler = mode_change.map(|f| {
            let h: reims_vgpu_core::display::host_calls::ModeChangeHandler =
                Arc::new(move |size: DisplayCoord, format: u32| f(size.x, size.y, format));
            h
        });
        d.new_frame_event_handler = new_frame;
        d.cursor_glyph_handler = cursor_glyph.map(|f| {
            let h: reims_vgpu_core::display::host_calls::CursorGlyphHandler =
                Arc::new(move |g: &CoreGlyph, hot: DisplayCoord| {
                    f(&CursorGlyph {
                        width: g.width,
                        height: g.height,
                        hot_spot: (hot.x, hot.y),
                        rgba: &g.rgba,
                    });
                });
            h
        });
        d.cursor_glyph_image_maker = None;
        d.cursor_show_handler = cursor_show;
        d.cursor_move_handler = cursor_move;
        d
    }
}

/// One reference to a display.
pub struct Display<P: Platform = DefaultPlatform> {
    /// The display's window, which every reference shares: the last reference's drop closes it
    /// before that reference's hold on the display goes (the `Drop` below).
    window: Option<Arc<WindowLink>>,
    inner: Ours<DisplayHandle<P>>,
    /// The frame queue of the device's Metal device.
    frame_queue: Option<P::CommandQueue>,
    /// The device's lease on the guest's RAM ([`Device::guest_memory_lease`]): core's display
    /// keeps core's device alive, so the lease goes after `inner`.
    guest_memory: Option<GuestMemoryLease>,
}

impl<P: Platform> Drop for Display<P> {
    /// The window's share first, so that the display's last reference closes the window before
    /// it releases the display; then the reference itself, then the frame queue, then the
    /// device's lease on the guest's RAM.
    fn drop(&mut self) {
        drop(self.window.take());
    }
}

impl<P: Platform> std::fmt::Debug for Display<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Display")
            .field("port", &self.inner.port())
            .field("serial_number", &self.inner.serial_num())
            .finish_non_exhaustive()
    }
}

impl<P: ApiPlatform> Display<P> {
    /// A new display on `device` ([`Device::new_display`]). The display thread is made at the
    /// process's first display. The config's closures are dropped before a refusal returns,
    /// except the new-frame closure of a display refused after its source was made, which the
    /// source's cancel frees on the display thread.
    ///
    /// With a window, the callbacks are the window's tee ([`window::tee`]). Matching the host
    /// screen the platform names for the port, the scale factor and the size in millimetres are
    /// its density, and the display is made with its modes, as [`Display::set_modes`] sets them;
    /// the window opens sized for the first. The window is given the display's sink once the
    /// display exists, and a refusal closes it before it returns. Where the platform names no
    /// screen, the density and the modes are those [`window::host_matched`] gives without one.
    ///
    /// With the frame statistics on (`REIMS_VGPU_STATS`, [`stats`]), the new-frame callback, the
    /// tee's for a window display, counts each of the guest's presents before it runs, and the
    /// display's stats are registered once it exists, before its window gets its sink.
    pub(crate) fn new(device: &Device<P>, mut config: DisplayConfig) -> Result<Display<P>, Error> {
        let platform = device.platform();
        entry::run(|_| {
            let state = api_state(platform);
            let queue = state.display_thread(platform);
            let frame_queue = state.frame_queue(device.handle.mtl_device_ref());
            let port = config.port;
            let serial = config.serial_number;
            let mut offered = Vec::new();
            let link = match config.window.take() {
                None => None,
                Some(wcfg) => {
                    let w = Window::new(wcfg, port, reims_vgpu_window::handle::title(port));
                    config.callbacks = window::tee(&w, std::mem::take(&mut config.callbacks));
                    if w.match_host_screen()
                        && let Some((modes, density)) = window::host_matched(
                            window::host_screen::<P>(port).as_ref(),
                            w.chosen_mode(),
                            w.chosen_ppi(),
                            reims_vgpu_window::env::settings().refresh_hz,
                        )
                    {
                        config.hidden.scale_factor = density.scale_factor;
                        config.size_in_millimeters = density.size_in_millimeters;
                        if let Some(first) = modes.first() {
                            w.offered_mode(first.width, first.height);
                        }
                        offered = modes;
                    }
                    w.told_size(config.size_in_millimeters);
                    Some(Arc::new(WindowLink::new(w)))
                }
            };
            let stats = stats::observe(
                platform.threads(),
                port,
                link.is_some(),
                &mut config.callbacks.new_frame,
            );
            let core = config.into_core(queue);
            let made = device.handle.new_display(&core, port, serial);
            drop(core);
            let made = match made {
                Ok(made) => made,
                Err(refusal) => {
                    drop(link);
                    return Err(refusal.into());
                }
            };
            if let Some(stats) = &stats {
                stats::register(stats, &made.downgrade());
            }
            if !offered.is_empty() {
                let modes: Vec<CoreMode> = offered
                    .iter()
                    .map(|m| DisplayMode::new(m.width, m.height, m.refresh_hz).to_core())
                    .collect();
                no_catch(made.set_mode_list(&modes));
            }
            if let Some(link) = &link
                && let Some(sink) = window::window_sink(
                    platform,
                    made.downgrade(),
                    frame_queue.as_ref().map(|q| q.clone_ref()),
                )
            {
                link.window().attach(sink);
            }
            Ok(Display {
                window: link,
                inner: Ours::new(made),
                frame_queue,
                guest_memory: device.guest_memory_lease(),
            })
        })
    }
}

impl<P: Platform> Display<P> {
    /// Another reference to this display. It is unplugged when the last one drops, and its
    /// window, if it has one, closes then.
    #[must_use]
    pub fn clone_ref(&self) -> Display<P> {
        Display {
            window: self.window.clone(),
            inner: Ours::new(self.inner.clone_ref()),
            frame_queue: self
                .frame_queue
                .as_ref()
                .map(reims_vgpu_core::platform::Handle::clone_ref),
            guest_memory: self.guest_memory.clone(),
        }
    }

    /// Whether both are references to one display.
    pub fn ptr_eq(&self, other: &Display<P>) -> bool {
        self.inner.ptr_eq(&other.inner)
    }

    /// Core's display (tests).
    #[cfg(any(test, feature = "testkit"))]
    pub fn core(&self) -> &DisplayHandle<P> {
        &self.inner
    }

    /// The display's window (tests).
    #[cfg(test)]
    pub(crate) fn window(&self) -> Option<&Arc<Window>> {
        self.window.as_deref().map(WindowLink::window)
    }

    /// Runs `body` as an entry ([`entry::run`]), with a hold of the entry's holder.
    fn entry<T>(&self, body: impl for<'u> FnOnce(Hold<'u>, &DisplayHandle<P>) -> T) -> T {
        entry::run(|hold| body(hold, &self.inner))
    }

    /// Offers the guest `modes`, in order (copied). An empty list changes nothing. The change may
    /// plug the display in and queue a mode change. A raise from the display ends the process.
    /// The display's window, if it has one and has not opened yet, opens sized for the first.
    pub fn set_modes(&self, modes: &[DisplayMode]) {
        if let (Some(link), Some(first)) = (&self.window, modes.first()) {
            link.window().offered_mode(first.width, first.height);
        }
        self.entry(|_, d| {
            let modes: Vec<CoreMode> = modes.iter().map(|m| m.to_core()).collect();
            no_catch(d.set_mode_list(&modes));
        });
    }

    /// The cursor's position; `(65535, 65535)` before the guest has linked the display.
    pub fn cursor_position(&self) -> (u16, u16) {
        let c = self.entry(|_, d| d.cursor_position());
        (c.x, c.y)
    }

    /// The guest's presents counted so far.
    pub fn guest_present_count(&self) -> u64 {
        self.inner.guest_present_count()
    }

    /// The host's encodes of a new frame counted so far.
    pub fn host_present_count(&self) -> u64 {
        self.inner.host_present_count()
    }

    /// Encodes the current frame into `target` over `region` (`None`: the whole target), in a
    /// command buffer of the process's frame queue, and answers the uncommitted frame. The
    /// display counts the frame in flight until it completes; its presents wait for frames in
    /// flight. The first encode after each new guest frame counts one host present.
    /// [`Error::InvalidArgument`] for a target on another Metal device.
    pub fn encode_frame(
        &self,
        target: &FrameTarget<P>,
        region: Option<Region>,
    ) -> Result<Frame<P>, Error> {
        self.entry(|_, d| {
            let Some(queue) = &self.frame_queue else {
                return Err(Error::Metal);
            };
            let region = region.unwrap_or(Region {
                x: 0,
                y: 0,
                width: u64::from(target.width()),
                height: u64::from(target.height()),
            });
            let region = MtlRegion::new(
                MtlOrigin::new(region.x, region.y, 0),
                MtlSize::new(region.width, region.height, 1),
            );
            // Metal's autorelease of the command buffer goes at the entry's pool, popped before
            // the return: only the frame's own reference is left.
            use reims_vgpu_core::platform::command::CommandQueue as _;
            let Some(command_buffer) = no_catch(queue.command_buffer(OnFault::Die)) else {
                return Err(Error::Metal);
            };
            // The frame itself for a CPU read where the pass would leave it as it is (behind its
            // switch), else the pass.
            let source =
                no_catch(d.encode_current_frame_by_copy(&command_buffer, target.texture(), region));
            let copied = source.is_some();
            target.set_source(source);
            if !copied {
                let encoded =
                    no_catch(d.encode_current_frame(&command_buffer, target.texture(), region));
                if !encoded {
                    return Err(Error::InvalidArgument);
                }
            }
            Ok(Frame::new(command_buffer, target.clone_ref()))
        })
    }
}
