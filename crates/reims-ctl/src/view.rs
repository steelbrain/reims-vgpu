//! The terminal viewer (`reims-ctl view`, and `v` in the interactive interface): a running VM's
//! display drawn in the terminal, with the terminal's keys, mouse and pasted text sent to the
//! guest. It is for VMs on hosts with no screen, to set the guest up before it has screen
//! sharing or remote login.
//!
//! Everything goes through the VM's QMP socket, so it works on any running VM:
//! - A frame is QMP's `screendump` of the display's console: a PPM, written into a directory of
//!   the viewer's own under `$XDG_RUNTIME_DIR` (or `/dev/shm`, or the temporary directory),
//!   removed when it ends, or by the next viewer where it was killed (`FrameDir`). For
//!   it, apple-gfx draws the guest's newest frame into the console's surface
//!   (vendor/qemu hw/display/apple-gfx.c, `apple_gfx_fb_update_display`), rendering and reading
//!   it back only when the guest has presented a frame since the last.
//! - Input is `input-send-event`: the USB tablet's absolute position and buttons, and the USB
//!   keyboard's keys (keys.rs).
//!
//! QMP's socket serves one client at a time, so the viewer lets its connection go after each
//! frame, and `reims-ctl stop` gets in while it runs.
//!
//! QEMU's USB keyboard queues 16 scancodes, drops a key whose scancodes do not fit, and gives the
//! guest one each time it polls the keyboard (hw/input/hid.c). Keys go out at most 12 scancodes
//! ahead of one each 10 ms, a tap's events together, so typing and pasting never overrun the
//! queue. Its tablet turns each `input-send-event` into one report, so a button change ends the
//! command it is in: a press and its release in one would leave the buttons as they were, and
//! the guest would see no click.
//!
//! Frames come as fast as the terminal takes them, at most `fps` a second: a capture waits until
//! the frame before it is written. QEMU writes the PPM under its big lock, so captures are at
//! least four screendumps' time apart; a screen that has not changed in five captures, with no
//! input for two seconds, is captured four times a second.
//!
//! Four threads: input (terminal events to QMP's), QMP (input out, frames in), the encoder
//! (encode.rs), and this one, which alone writes to the terminal.

mod encode;
mod keys;
mod term;

pub use term::Protocol;

use crate::host::Profile;
use crate::qmp::Qmp;
use crate::runtime;
use crate::vm::Vm;
use anyhow::{Context, Result, bail};
use encode::{Encoded, Frame, Payload, Place};
use keys::{Ev, Key, Mods};
use ratatui::DefaultTerminal;
use ratatui::buffer::CellDiffOption;
use ratatui::crossterm::event::{
    self, DisableBracketedPaste, DisableFocusChange, DisableMouseCapture, EnableBracketedPaste,
    EnableFocusChange, EnableMouseCapture, Event, KeyEventKind, KeyboardEnhancementFlags,
    MouseButton, MouseEvent, MouseEventKind, PopKeyboardEnhancementFlags,
    PushKeyboardEnhancementFlags,
};
use ratatui::crossterm::execute;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use serde_json::{Value, json};
use std::collections::VecDeque;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub struct Options {
    /// The display, from 1.
    pub display: u32,
    /// The most frames a second.
    pub fps: u32,
    /// How to draw frames; asked of the terminal when `None`.
    pub protocol: Option<Protocol>,
}

impl Default for Options {
    fn default() -> Options {
        Options {
            display: 1,
            fps: 15,
            protocol: None,
        }
    }
}

/// A cell's size where the terminal does not say: kitty's and iTerm2's frames are scaled to
/// their cells by the terminal, so only the ratio matters to them.
const ASSUMED_CELL: (u16, u16) = (8, 16);

/// Scancodes the guest's keyboard may have queued, of QEMU's 16.
const KEY_QUEUE: f64 = 12.0;
/// Scancodes the guest takes from the queue a second, at most.
const KEYS_PER_SEC: f64 = 100.0;

/// Shows `vm`'s display until Control-] or the VM stops; what to tell the user after, if
/// anything.
pub fn run(vm: &Vm, opts: &Options) -> Result<Option<String>> {
    let name = vm.name().to_string();
    if runtime::pid(vm).is_none() {
        bail!("{name} is not running: reims-ctl start {name}");
    }
    // SAFETY: isatty only reads its descriptor's state.
    if unsafe { libc::isatty(0) == 0 || libc::isatty(1) == 0 } {
        bail!("reims-ctl view draws in a terminal: run it in one");
    }
    let target = Target::for_vm(vm, opts.display)?;
    let dir = FrameDir::new()?;
    let file = dir.0.join("frame.ppm");
    // A screendump before the terminal is taken, so that what stops one shows plainly.
    Qmp::connect(&target.socket, Duration::from_secs(5))
        .and_then(|mut q| q.execute("screendump", Some(target.screendump(&file))))
        .with_context(|| format!("take a screendump of {name}"))?;

    let mut tg = TermGuard::new()?;
    let caps = term::query(Duration::from_secs(1));
    let proto = opts
        .protocol
        .unwrap_or_else(|| term::choose(&caps, |k| std::env::var(k).ok()));
    let (cols, rows) = ratatui::crossterm::terminal::size()?;
    let measured = term::cell_size(&caps, cols, rows);
    let pixel_mouse = caps.sgr_pixels && measured.is_some();
    tg.enable(caps.kitty_keyboard, pixel_mouse, proto == Protocol::Kitty)?;

    let sh = Arc::new(Shared {
        stop: AtomicBool::new(false),
        busy: AtomicBool::new(false),
        screen: Mutex::new(Screen {
            cols,
            rows,
            cell: measured.unwrap_or(ASSUMED_CELL),
            pixel_mouse,
            place: None,
            latch: Mods::NONE,
            layout: 0,
        }),
    });
    let (ui_tx, ui_rx) = channel();
    let (input_tx, input_rx) = channel();
    let (enc_tx, enc_rx) = channel();
    let (done_tx, done_rx) = channel();
    {
        let c = Capture {
            sh: sh.clone(),
            target,
            file,
            vm: vm.clone(),
            rx: input_rx,
            enc: enc_tx.clone(),
            ui: ui_tx.clone(),
            gap: Duration::from_secs_f64(1.0 / f64::from(opts.fps.clamp(1, 60))),
            done: done_tx,
        };
        std::thread::Builder::new()
            .name("view-qmp".into())
            .spawn(move || qmp_thread(c))?;
    }
    {
        let (sh, ui) = (sh.clone(), ui_tx.clone());
        std::thread::Builder::new()
            .name("view-encode".into())
            .spawn(move || encoder_thread(sh, proto, enc_rx, ui))?;
    }
    let input = {
        let (sh, name) = (sh.clone(), name.clone());
        let enhanced = caps.kitty_keyboard;
        std::thread::Builder::new()
            .name("view-input".into())
            .spawn(move || input_thread(sh, name, enhanced, input_tx, ui_tx))?
    };

    let mut v = View {
        name,
        display: opts.display,
        proto,
        caps,
        blocks: None,
        frame: None,
        shown: VecDeque::new(),
        note: None,
    };
    let result = main_loop(&mut tg.terminal, &sh, &mut v, &ui_rx, &enc_tx);
    tg.quiet();
    sh.stop.store(true, Ordering::SeqCst);
    let _ = input.join();
    // The QMP thread sends the input thread's releases of buttons held down, then ends.
    let _ = done_rx.recv_timeout(Duration::from_secs(2));
    drop(tg);
    drop(dir);
    result
}

/// Where QMP finds the display.
struct Target {
    socket: PathBuf,
    /// The display device's id and the display's head. vmapple's apple-gfx-mmio has no id; its
    /// first display is console 0, QMP's default.
    device: Option<(&'static str, u32)>,
}

impl Target {
    fn for_vm(vm: &Vm, display: u32) -> Result<Target> {
        let count = vm.config.display_count();
        if display == 0 || display > count {
            bail!(
                "{} has {count} display{}: --display takes 1 to {count}",
                vm.name(),
                if count == 1 { "" } else { "s" }
            );
        }
        let device = match vm.config.profile {
            // qemu.rs gives reims-graphics-pci, apple-gfx-pci or vmware-svga the id gfx.
            Profile::Q35X86_64 => Some(("gfx", display - 1)),
            Profile::VmappleArm64 if display == 1 => None,
            Profile::VmappleArm64 => {
                bail!("a vmapple VM's displays past the first have no name QMP can take")
            }
        };
        Ok(Target {
            socket: vm.qmp_socket(),
            device,
        })
    }

    fn with_device(&self, mut v: Value) -> Value {
        if let Some((d, head)) = self.device {
            v["device"] = json!(d);
            v["head"] = json!(head);
        }
        v
    }

    fn screendump(&self, file: &Path) -> Value {
        self.with_device(json!({ "filename": file, "format": "ppm" }))
    }
}

/// The directory QEMU writes frames into, removed when the viewer ends. A viewer that a signal
/// ended (killed, or its terminal closed) leaves its directory, a frame of up to some MB, in
/// memory where it is under `/dev/shm`; the next viewer removes it ([`sweep_frame_dirs`]).
struct FrameDir(PathBuf);

impl FrameDir {
    fn new() -> Result<FrameDir> {
        use std::os::unix::fs::DirBuilderExt;
        let base = std::env::var_os("XDG_RUNTIME_DIR")
            .map(PathBuf::from)
            .filter(|p| p.is_dir())
            .or_else(|| Some(PathBuf::from("/dev/shm")).filter(|p| p.is_dir()))
            .unwrap_or_else(std::env::temp_dir);
        sweep_frame_dirs(&base);
        let dir = base.join(format!("{FRAME_DIR_PREFIX}{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::DirBuilder::new()
            .mode(0o700)
            .create(&dir)
            .with_context(|| format!("create {}", dir.display()))?;
        Ok(FrameDir(dir))
    }
}

impl Drop for FrameDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A frame directory's name, before its viewer's pid.
const FRAME_DIR_PREFIX: &str = "reims-ctl-view-";

/// Removes the frame directories in `base` that viewers which no longer run left: those of this
/// user's named for a pid no process has. A running viewer's stays, as does any other entry.
fn sweep_frame_dirs(base: &Path) {
    use std::os::unix::fs::MetadataExt;
    let Ok(entries) = std::fs::read_dir(base) else {
        return;
    };
    // SAFETY: getuid has no preconditions.
    let uid = unsafe { libc::getuid() };
    for e in entries.flatten() {
        let pid = e.file_name().to_str().and_then(|n| {
            n.strip_prefix(FRAME_DIR_PREFIX)
                .and_then(|p| p.parse::<libc::pid_t>().ok())
        });
        let Some(pid) = pid.filter(|p| *p > 0) else {
            continue;
        };
        // The entry itself, not what a link names.
        if !e.metadata().is_ok_and(|m| m.is_dir() && m.uid() == uid) {
            continue;
        }
        // SAFETY: signal 0 sends nothing; it asks only whether the process exists.
        let gone = unsafe { libc::kill(pid, 0) } == -1
            && std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH);
        if gone {
            let _ = std::fs::remove_dir_all(e.path());
        }
    }
}

/// The terminal while the viewer has it: ratatui's (raw, the alternate screen), and the modes
/// the viewer turns on, which it turns off again however it ends.
struct TermGuard {
    terminal: DefaultTerminal,
    /// The mouse and the rest are reported.
    reporting: bool,
    keyboard: bool,
    pixel_mouse: bool,
    kitty: bool,
}

impl TermGuard {
    fn new() -> Result<TermGuard> {
        Ok(TermGuard {
            terminal: ratatui::try_init()?,
            reporting: false,
            keyboard: false,
            pixel_mouse: false,
            kitty: false,
        })
    }

    /// Stops the terminal reporting the mouse, focus, pastes and keys the kitty way, as soon as
    /// the viewer stops reading them: reports sent after would reach the shell as text.
    fn quiet(&mut self) {
        if !self.reporting {
            return;
        }
        self.reporting = false;
        let mut out = std::io::stdout();
        if self.pixel_mouse {
            let _ = out.write_all(b"\x1b[?1016l");
        }
        if self.keyboard {
            let _ = execute!(out, PopKeyboardEnhancementFlags);
        }
        let _ = execute!(
            out,
            DisableFocusChange,
            DisableBracketedPaste,
            DisableMouseCapture
        );
        let _ = out.flush();
    }

    fn enable(&mut self, keyboard: bool, pixel_mouse: bool, kitty: bool) -> Result<()> {
        let mut out = std::io::stdout();
        self.reporting = true;
        execute!(
            out,
            EnableMouseCapture,
            EnableBracketedPaste,
            EnableFocusChange
        )?;
        if keyboard {
            execute!(
                out,
                PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES)
            )?;
            self.keyboard = true;
        }
        if pixel_mouse {
            out.write_all(b"\x1b[?1016h")?;
            self.pixel_mouse = true;
        }
        self.kitty = kitty;
        out.flush()?;
        Ok(())
    }
}

impl Drop for TermGuard {
    fn drop(&mut self) {
        self.quiet();
        let mut out = std::io::stdout();
        if self.kitty {
            let _ = out.write_all(&encode::kitty_delete_all());
        }
        let _ = out.flush();
        // Reports that came in before the terminal stopped sending them go unread.
        // SAFETY: tcflush only discards stdin's queued input.
        unsafe { libc::tcflush(0, libc::TCIFLUSH) };
        ratatui::restore();
    }
}

struct Shared {
    stop: AtomicBool,
    /// A frame is captured and not yet on the screen: the next capture waits for it.
    busy: AtomicBool,
    screen: Mutex<Screen>,
}

impl Shared {
    fn screen(&self) -> Screen {
        *self.screen.lock().unwrap()
    }

    fn set_screen(&self, f: impl FnOnce(&mut Screen)) {
        f(&mut self.screen.lock().unwrap())
    }
}

#[derive(Clone, Copy)]
struct Screen {
    cols: u16,
    rows: u16,
    /// A cell's size in pixels, measured or assumed.
    cell: (u16, u16),
    /// The terminal reports the mouse in pixels.
    pixel_mouse: bool,
    /// Where the frame on the screen is.
    place: Option<Place>,
    /// The toolbar's modifiers, held for the next key or click.
    latch: Mods,
    /// Counts resizes: a frame encoded for an earlier size is not shown.
    layout: u64,
}

/// The cells frames may fill: all but the toolbar's row and the status row.
fn image_area(cols: u16, rows: u16) -> Place {
    Place {
        col: 0,
        row: 1,
        cols,
        rows: rows.saturating_sub(2),
    }
}

/// A frame's sizes, captured and sent, and its bytes (none for half blocks).
#[derive(Clone, Copy)]
struct Shown {
    frame: (u32, u32),
    sent: (u32, u32),
    bytes: usize,
}

enum Ui {
    Frame(Encoded, u64),
    Resize,
    Redraw,
    Note(String),
    Quit,
    /// The VM stopped: the viewer ends, saying this.
    Gone(String),
}

struct View {
    name: String,
    display: u32,
    proto: Protocol,
    caps: term::Caps,
    /// Half blocks on the screen, where they draw the frame.
    blocks: Option<(Place, Vec<u8>)>,
    /// The last frame on the screen.
    frame: Option<Shown>,
    /// When frames reached the screen, over the last two seconds.
    shown: VecDeque<Instant>,
    note: Option<(String, Instant)>,
}

fn main_loop(
    terminal: &mut DefaultTerminal,
    sh: &Shared,
    v: &mut View,
    ui: &Receiver<Ui>,
    enc: &Sender<Option<Frame>>,
) -> Result<Option<String>> {
    let mut dirty = true;
    let mut pending: Option<Vec<u8>> = None;
    let mut tick = Instant::now();
    loop {
        if dirty {
            let s = sh.screen();
            terminal.draw(|f| draw(f, v, &s))?;
            dirty = false;
        }
        if let Some(bytes) = pending.take() {
            let mut out = std::io::stdout().lock();
            out.write_all(&bytes)?;
            out.flush()?;
            sh.busy.store(false, Ordering::SeqCst);
        }
        let msg = match ui.recv_timeout(Duration::from_millis(250)) {
            Ok(m) => Some(m),
            Err(RecvTimeoutError::Timeout) => None,
            Err(RecvTimeoutError::Disconnected) => return Ok(None),
        };
        if tick.elapsed() >= Duration::from_secs(1) {
            // The status row's rate, and a note's expiry.
            tick = Instant::now();
            dirty = true;
        }
        match msg {
            None => {}
            Some(Ui::Frame(e, layout)) => {
                let mut s = sh.screen.lock().unwrap();
                if layout != s.layout {
                    drop(s);
                    sh.busy.store(false, Ordering::SeqCst);
                    continue;
                }
                if s.place != Some(e.place) {
                    // A frame of a new size or place: the old one goes first.
                    s.place = Some(e.place);
                    drop(s);
                    clear(terminal, v)?;
                } else {
                    drop(s);
                }
                let now = Instant::now();
                v.shown.push_back(now);
                while v
                    .shown
                    .front()
                    .is_some_and(|t| now.duration_since(*t) > Duration::from_secs(2))
                {
                    v.shown.pop_front();
                }
                let mut shown = Shown {
                    frame: e.frame,
                    sent: e.sent,
                    bytes: 0,
                };
                match e.payload {
                    Payload::Escapes(b) => {
                        shown.bytes = b.len();
                        v.frame = Some(shown);
                        pending = Some(b);
                    }
                    Payload::Blocks(px) => {
                        v.frame = Some(shown);
                        v.blocks = Some((e.place, px));
                        sh.busy.store(false, Ordering::SeqCst);
                    }
                }
                dirty = true;
            }
            Some(Ui::Resize) => {
                let (cols, rows) = ratatui::crossterm::terminal::size()?;
                let cell = term::cell_size(&v.caps, cols, rows);
                sh.set_screen(|s| {
                    s.cols = cols;
                    s.rows = rows;
                    if let Some(c) = cell {
                        s.cell = c;
                    }
                    s.place = None;
                    s.layout += 1;
                });
                clear(terminal, v)?;
                // The encoder draws the last frame again at the new size.
                let _ = enc.send(None);
                dirty = true;
            }
            Some(Ui::Redraw) => dirty = true,
            Some(Ui::Note(n)) => {
                v.note = Some((n, Instant::now()));
                dirty = true;
            }
            Some(Ui::Quit) => return Ok(None),
            Some(Ui::Gone(m)) => return Ok(Some(m)),
        }
    }
}

/// Clears the screen, kitty's images with it, and has the next draw draw every cell. Not
/// `Terminal::clear`, which asks the terminal where its cursor is and waits for the answer (a
/// round trip over SSH, and an error after two seconds without one); a fullscreen terminal's
/// `resize` clears without asking.
fn clear(terminal: &mut DefaultTerminal, v: &mut View) -> Result<()> {
    if v.proto == Protocol::Kitty {
        let mut out = std::io::stdout().lock();
        out.write_all(&encode::kitty_delete_all())?;
        out.flush()?;
    }
    v.blocks = None;
    let size = terminal.size()?;
    terminal.resize(Rect::new(0, 0, size.width, size.height))?;
    Ok(())
}

const BAR: Style = Style::new().fg(Color::White).bg(Color::DarkGray);
const KEY: Style = Style::new()
    .fg(Color::Black)
    .bg(Color::Yellow)
    .add_modifier(Modifier::BOLD);
const BUTTON: Style = Style::new().fg(Color::Black).bg(Color::Gray);
const HELD: Style = Style::new()
    .fg(Color::Black)
    .bg(Color::Cyan)
    .add_modifier(Modifier::BOLD);

/// The top row, and the columns of each modifier button on it.
fn toolbar(name: &str, latch: Mods) -> (Line<'static>, Vec<(u16, u16, Mods)>) {
    let mut spans = Vec::new();
    let mut buttons = Vec::new();
    let mut x = 0u16;
    let mut push = |s: String, style: Style, spans: &mut Vec<Span<'static>>| {
        let w = s.chars().count() as u16;
        spans.push(Span::styled(s, style));
        x += w;
        (x - w, x)
    };
    push(" Ctrl-] disconnect ".into(), KEY, &mut spans);
    push("  next key or click with ".into(), BAR, &mut spans);
    for (i, (m, label, _)) in Mods::EACH.into_iter().enumerate() {
        if i > 0 {
            push(" ".into(), BAR, &mut spans);
        }
        let style = if latch.contains(m) { HELD } else { BUTTON };
        let (x0, x1) = push(format!(" {label} "), style, &mut spans);
        buttons.push((x0, x1, m));
    }
    push("   Alt+key is Cmd+key   ".into(), BAR, &mut spans);
    push(
        name.to_string(),
        BAR.add_modifier(Modifier::BOLD),
        &mut spans,
    );
    (Line::from(spans), buttons)
}

fn status(v: &View) -> Line<'static> {
    let mut s = format!(" display {}", v.display);
    if let Some(Shown {
        frame: (fw, fh),
        sent: (sw, sh),
        bytes,
    }) = v.frame
    {
        s.push_str(&format!(" · {fw}×{fh}"));
        match v.proto {
            Protocol::Blocks => s.push_str(" · half blocks"),
            p => s.push_str(&format!(" · {} {sw}×{sh}", p.name())),
        }
        s.push_str(&format!(" · {:.0} fps", v.shown.len() as f64 / 2.0));
        if bytes > 0 {
            s.push_str(&format!(" · {} KB a frame", bytes.div_ceil(1024)));
        }
    } else {
        s.push_str(&format!(" · {}", v.proto.name()));
    }
    let mut spans = vec![Span::styled(s, BAR)];
    if let Some((n, at)) = &v.note
        && at.elapsed() < Duration::from_secs(6)
    {
        spans.push(Span::styled(
            format!("   {n}"),
            BAR.fg(Color::Yellow).add_modifier(Modifier::BOLD),
        ));
    }
    Line::from(spans)
}

fn draw(f: &mut ratatui::Frame, v: &View, s: &Screen) {
    let a = f.area();
    if a.height < 4 || a.width < 20 {
        f.render_widget(Paragraph::new("too small"), a);
        return;
    }
    let (bar, _) = toolbar(&v.name, s.latch);
    f.render_widget(Paragraph::new(bar).style(BAR), Rect::new(0, 0, a.width, 1));
    f.render_widget(
        Paragraph::new(status(v)).style(BAR),
        Rect::new(0, a.height - 1, a.width, 1),
    );
    let inside = |x: u16, y: u16| x < a.width && y >= 1 && y < a.height - 1;
    let buf = f.buffer_mut();
    match (&v.blocks, s.place) {
        (Some((p, px)), _) => {
            // A cell's upper half is the foreground of ▀, its lower half the background.
            let w = p.cols as usize;
            for r in 0..p.rows {
                for c in 0..p.cols {
                    let (x, y) = (p.col + c, p.row + r);
                    if !inside(x, y) {
                        continue;
                    }
                    let at = |row: usize| {
                        let i = (row * w + c as usize) * 3;
                        Color::Rgb(px[i], px[i + 1], px[i + 2])
                    };
                    if let Some(cell) = buf.cell_mut((x, y)) {
                        cell.set_char('▀')
                            .set_fg(at(2 * r as usize))
                            .set_bg(at(2 * r as usize + 1));
                    }
                }
            }
        }
        (None, Some(p)) if v.proto != Protocol::Blocks => {
            // The frame's cells are the image's: ratatui leaves them alone.
            for y in p.row..p.row + p.rows {
                for x in p.col..p.col + p.cols {
                    if inside(x, y)
                        && let Some(cell) = buf.cell_mut((x, y))
                    {
                        cell.set_diff_option(CellDiffOption::Skip);
                    }
                }
            }
        }
        _ => {
            let msg = format!("waiting for {}'s screen…", v.name);
            let w = msg.chars().count() as u16;
            let r = Rect::new(
                a.width.saturating_sub(w) / 2,
                a.height / 2,
                w.min(a.width),
                1,
            );
            f.render_widget(Paragraph::new(msg), r);
        }
    }
}

// ------------------------------------------------------------ input

fn input_thread(
    sh: Arc<Shared>,
    name: String,
    enhanced: bool,
    qmp: Sender<Vec<Ev>>,
    ui: Sender<Ui>,
) {
    // Buttons down in the guest, with the modifiers held for them.
    let mut held: Vec<(MouseButton, Mods)> = Vec::new();
    let note = |n: String| {
        let _ = ui.send(Ui::Note(n));
    };
    while !sh.stop.load(Ordering::SeqCst) {
        match event::poll(Duration::from_millis(50)) {
            Ok(true) => {}
            Ok(false) => continue,
            Err(_) => break,
        }
        let Ok(ev) = event::read() else { break };
        match ev {
            Event::Key(k) if k.kind != KeyEventKind::Release => {
                if keys::is_disconnect(&k, enhanced) {
                    let _ = ui.send(Ui::Quit);
                    continue;
                }
                let latch = sh.screen().latch;
                match keys::key(&k, latch) {
                    Key::Tap(evs) => {
                        let _ = qmp.send(evs);
                        if !latch.is_empty() {
                            sh.set_screen(|s| s.latch = Mods::NONE);
                            let _ = ui.send(Ui::Redraw);
                        }
                    }
                    Key::Untypable(c) => note(format!("{c} has no key on a US keyboard")),
                    Key::Ignored => {}
                }
            }
            Event::Paste(text) => {
                let (taps, untypable) = keys::paste(&text);
                for t in taps {
                    let _ = qmp.send(t);
                }
                if !untypable.is_empty() {
                    let s: String = untypable.iter().take(8).collect();
                    note(format!(
                        "left out of the paste, with no key on a US keyboard: {s}"
                    ));
                }
            }
            Event::Mouse(m) => mouse(&sh, &name, &mut held, m, &qmp, &ui),
            Event::Resize(..) => {
                let _ = ui.send(Ui::Resize);
            }
            Event::FocusLost => release(&mut held, &qmp),
            _ => {}
        }
    }
    release(&mut held, &qmp);
}

/// Lets go of the buttons held down in the guest.
fn release(held: &mut Vec<(MouseButton, Mods)>, qmp: &Sender<Vec<Ev>>) {
    for (b, mods) in held.drain(..).rev() {
        let mut evs = vec![Ev::Btn(keys::button(b), false)];
        evs.extend(keys::hold(mods, false));
        let _ = qmp.send(evs);
    }
}

fn mouse(
    sh: &Shared,
    name: &str,
    held: &mut Vec<(MouseButton, Mods)>,
    m: MouseEvent,
    qmp: &Sender<Vec<Ev>>,
    ui: &Sender<Ui>,
) {
    let s = sh.screen();
    let pixel = s.pixel_mouse.then_some(s.cell);
    let (col, row) = match pixel {
        Some((cw, ch)) => (m.column / cw.max(1), m.row / ch.max(1)),
        None => (m.column, m.row),
    };
    let at = s.place.map(|p| keys::point(m.column, m.row, pixel, p));
    let pos = at.map(|(fx, fy, _)| keys::abs(fx, fy));
    let on = at.is_some_and(|(_, _, on)| on);
    let send = |evs: Vec<Ev>| {
        let _ = qmp.send(evs);
    };
    match m.kind {
        // A release lands wherever the pointer is; one the terminal does not name the button of
        // releases the last pressed.
        MouseEventKind::Up(b) if !held.is_empty() => {
            let i = held
                .iter()
                .rposition(|(h, _)| *h == b)
                .unwrap_or(held.len() - 1);
            let (b, mods) = held.remove(i);
            let mut evs: Vec<Ev> = pos.into_iter().collect();
            evs.push(Ev::Btn(keys::button(b), false));
            evs.extend(keys::hold(mods, false));
            send(evs);
        }
        MouseEventKind::Down(MouseButton::Left) if row == 0 => {
            let (_, buttons) = toolbar(name, s.latch);
            if let Some(&(_, _, b)) = buttons.iter().find(|(x0, x1, _)| (*x0..*x1).contains(&col)) {
                sh.set_screen(|s| s.latch.toggle(b));
                let _ = ui.send(Ui::Redraw);
            }
        }
        MouseEventKind::Moved if on => send(pos.into_iter().collect()),
        MouseEventKind::Drag(_) if !held.is_empty() => send(pos.into_iter().collect()),
        MouseEventKind::Down(b) if on => {
            let mods = Mods::from_terminal(m.modifiers) | s.latch;
            if !s.latch.is_empty() {
                sh.set_screen(|s| s.latch = Mods::NONE);
                let _ = ui.send(Ui::Redraw);
            }
            let mut evs: Vec<Ev> = pos.into_iter().collect();
            evs.extend(keys::hold(mods, true));
            evs.push(Ev::Btn(keys::button(b), true));
            held.push((b, mods));
            send(evs);
        }
        MouseEventKind::ScrollUp
        | MouseEventKind::ScrollDown
        | MouseEventKind::ScrollLeft
        | MouseEventKind::ScrollRight
            if on =>
        {
            let wheel = match m.kind {
                MouseEventKind::ScrollUp => "wheel-up",
                MouseEventKind::ScrollDown => "wheel-down",
                MouseEventKind::ScrollLeft => "wheel-left",
                _ => "wheel-right",
            };
            let mut evs: Vec<Ev> = pos.into_iter().collect();
            evs.push(Ev::Btn(wheel, true));
            evs.push(Ev::Btn(wheel, false));
            send(evs);
        }
        _ => {}
    }
}

// ------------------------------------------------------------ QMP and frames

struct Capture {
    sh: Arc<Shared>,
    target: Target,
    file: PathBuf,
    vm: Vm,
    rx: Receiver<Vec<Ev>>,
    enc: Sender<Option<Frame>>,
    ui: Sender<Ui>,
    /// The shortest time between captures.
    gap: Duration,
    done: Sender<()>,
}

/// Queues a group of input events; a move after a move replaces it.
fn queue(pending: &mut VecDeque<Vec<Ev>>, g: Vec<Ev>) {
    let is_move = |g: &[Ev]| matches!(g, [Ev::Abs(..)]);
    if is_move(&g) && pending.back().is_some_and(|b| is_move(b)) {
        *pending.back_mut().unwrap() = g;
    } else if !g.is_empty() {
        pending.push_back(g);
    }
}

/// The next `input-send-event`'s events, from the front of `pending`: as many groups as the
/// keyboard's queue takes on top of the `queued` scancodes, which they add to, up to and with
/// the first that presses or releases a button.
fn take_batch(pending: &mut VecDeque<Vec<Ev>>, queued: &mut f64) -> Vec<Ev> {
    let mut batch = Vec::new();
    while let Some(g) = pending.front() {
        let k = f64::from(g.iter().map(|e| e.slots()).sum::<u32>());
        // A tap with every modifier (14) goes alone into an empty queue.
        if *queued + k > KEY_QUEUE && *queued > 0.0 {
            break;
        }
        *queued += k;
        let g = pending.pop_front().unwrap();
        let click = g.iter().any(|e| e.is_click());
        batch.extend(g);
        if click {
            break;
        }
    }
    batch
}

fn connect<'a>(conn: &'a mut Option<Qmp>, socket: &Path) -> Result<&'a mut Qmp> {
    if conn.is_none() {
        *conn = Some(Qmp::connect(socket, Duration::from_secs(5))?);
    }
    Ok(conn.as_mut().unwrap())
}

fn send_input(conn: &mut Option<Qmp>, t: &Target, evs: Vec<Ev>) -> Result<()> {
    let mut events = Vec::new();
    for e in evs {
        e.json(&mut events);
    }
    let r = connect(conn, &t.socket).and_then(|q| {
        q.execute(
            "input-send-event",
            Some(t.with_device(json!({ "events": events }))),
        )
    });
    if r.is_err() {
        *conn = None;
    }
    r.map(|_| ())
}

fn qmp_thread(c: Capture) {
    let mut conn: Option<Qmp> = None;
    let mut pending: VecDeque<Vec<Ev>> = VecDeque::new();
    // Scancodes the guest has yet to take from its queue, by estimate.
    let mut queued = 0.0f64;
    let mut queued_at = Instant::now();
    let mut next = Instant::now();
    let mut last: Option<Arc<Vec<u8>>> = None;
    let mut same = 0u32;
    let mut input_at: Option<Instant> = None;
    let note = |e: &anyhow::Error| {
        let _ = c.ui.send(Ui::Note(format!("{e:#}")));
    };
    let gone = || {
        let stopped = runtime::pid(&c.vm).is_none();
        if stopped {
            let _ = c.ui.send(Ui::Gone(format!("{} stopped", c.vm.name())));
        }
        stopped
    };
    loop {
        let stopping = c.sh.stop.load(Ordering::SeqCst);
        let now = Instant::now();
        let wait = if stopping {
            Duration::ZERO
        } else if !pending.is_empty() || now >= next {
            Duration::from_millis(10)
        } else {
            (next - now).min(Duration::from_millis(100))
        };
        match c.rx.recv_timeout(wait) {
            Ok(g) => queue(&mut pending, g),
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) if pending.is_empty() => break,
            Err(RecvTimeoutError::Disconnected) => {}
        }
        while let Ok(g) = c.rx.try_recv() {
            queue(&mut pending, g);
        }
        if stopping {
            // The input thread releases the buttons it holds down as it ends, which drops its
            // sender. Releases go out; a tap not yet sent is dropped whole.
            let end = Instant::now() + Duration::from_secs(1);
            while let Ok(g) =
                c.rx.recv_timeout(end.saturating_duration_since(Instant::now()))
            {
                queue(&mut pending, g);
            }
            pending.retain(|g| !g.iter().any(|e| matches!(e, Ev::Key(_, true))));
            if !pending.is_empty() {
                let _ = send_input(&mut conn, &c.target, pending.drain(..).flatten().collect());
            }
            break;
        }

        let t = Instant::now();
        queued = (queued - t.duration_since(queued_at).as_secs_f64() * KEYS_PER_SEC).max(0.0);
        queued_at = t;
        let batch = take_batch(&mut pending, &mut queued);
        if !batch.is_empty() {
            input_at = Some(t);
            // Show what the input did soon.
            next = next.min(t + Duration::from_millis(30));
            if let Err(e) = send_input(&mut conn, &c.target, batch) {
                if gone() {
                    break;
                }
                note(&e);
            }
        }

        if Instant::now() < next || c.sh.busy.load(Ordering::SeqCst) {
            continue;
        }
        let t0 = Instant::now();
        let dumped = connect(&mut conn, &c.target.socket)
            .and_then(|q| q.execute("screendump", Some(c.target.screendump(&c.file))))
            .and_then(|_| Ok(std::fs::read(&c.file)?));
        // Between frames QMP's socket is free for other clients.
        conn = None;
        let data = match dumped {
            Ok(d) => d,
            Err(e) => {
                if gone() {
                    break;
                }
                note(&e);
                next = Instant::now() + Duration::from_secs(1);
                continue;
            }
        };
        let dump = t0.elapsed();
        if last.as_deref() == Some(&data) {
            same += 1;
        } else {
            // Compared as QEMU wrote it: a frame of padded rows holds its own copy.
            let data = Arc::new(data);
            match encode::parse_ppm(data.clone()) {
                Ok(f) => {
                    same = 0;
                    last = Some(data);
                    c.sh.busy.store(true, Ordering::SeqCst);
                    let _ = c.enc.send(Some(f));
                }
                Err(e) => note(&e),
            }
        }
        let quiet = same >= 5 && input_at.is_none_or(|t| t.elapsed() > Duration::from_secs(2));
        let gap = c.gap.max(dump * 4);
        next = Instant::now()
            + if quiet {
                gap.max(Duration::from_millis(250))
            } else {
                gap
            };
    }
    let _ = c.done.send(());
}

fn encoder_thread(sh: Arc<Shared>, proto: Protocol, rx: Receiver<Option<Frame>>, ui: Sender<Ui>) {
    let mut frame: Option<Frame> = None;
    let mut turn = 0;
    while let Ok(m) = rx.recv() {
        // The newest frame; None draws the last one again.
        frame = m.or(frame);
        while let Ok(m) = rx.try_recv() {
            frame = m.or(frame);
        }
        let Some(f) = &frame else { continue };
        let s = sh.screen();
        let area = image_area(s.cols, s.rows);
        if area.cols == 0 || area.rows == 0 {
            sh.busy.store(false, Ordering::SeqCst);
            continue;
        }
        match encode::encode(proto, f, area, s.cell, &mut turn) {
            Ok(e) => {
                if ui.send(Ui::Frame(e, s.layout)).is_err() {
                    break;
                }
            }
            Err(e) => {
                sh.busy.store(false, Ordering::SeqCst);
                let _ = ui.send(Ui::Note(format!("{e:#}")));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The next viewer removes the frame directory a killed one left, and keeps a running
    /// viewer's and every other entry
    #[test]
    fn a_killed_viewers_frame_dir_is_removed_by_the_next() {
        let base = std::env::temp_dir().join(format!("reims-ctl-sweep-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        let mut ended = std::process::Command::new("true").spawn().unwrap();
        ended.wait().unwrap();
        let dir = |name: String| {
            let d = base.join(name);
            std::fs::create_dir(&d).unwrap();
            std::fs::write(d.join("frame.ppm"), "P6").unwrap();
            d
        };
        let killed = dir(format!("{FRAME_DIR_PREFIX}{}", ended.id()));
        let running = dir(format!("{FRAME_DIR_PREFIX}{}", std::process::id()));
        let others = [
            dir(format!("{FRAME_DIR_PREFIX}x")),
            dir(format!("{FRAME_DIR_PREFIX}0")),
            dir(format!("{FRAME_DIR_PREFIX}-1")),
            dir("reims-ctl-qmp".into()),
        ];
        sweep_frame_dirs(&base);
        let left = (
            killed.exists(),
            running.exists(),
            others.iter().all(|d| d.exists()),
        );
        let _ = std::fs::remove_dir_all(&base);
        assert_eq!(left, (false, true, true), "(killed, running, others)");
    }

    #[test]
    fn the_toolbar_shows_disconnect_first_and_finds_its_buttons() {
        let (line, buttons) = toolbar("mac", Mods::CMD);
        let text: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(text.starts_with(" Ctrl-] disconnect "), "{text}");
        assert_eq!(buttons.len(), 4);
        for (x0, x1, m) in buttons {
            let label: String = text
                .chars()
                .skip(x0 as usize)
                .take((x1 - x0) as usize)
                .collect();
            let want = Mods::EACH.iter().find(|e| e.0 == m).unwrap().1;
            assert_eq!(label.trim(), want);
        }
        let held = line
            .spans
            .iter()
            .find(|s| s.content.trim() == "Cmd")
            .unwrap();
        assert_eq!(held.style, HELD);
    }

    #[test]
    fn moves_coalesce_and_taps_do_not() {
        let mut q = VecDeque::new();
        queue(&mut q, vec![Ev::Abs(1, 1)]);
        queue(&mut q, vec![Ev::Abs(2, 2)]);
        queue(&mut q, keys::tap(Mods::NONE, "a"));
        queue(&mut q, vec![Ev::Abs(3, 3)]);
        queue(&mut q, vec![Ev::Abs(4, 4)]);
        queue(&mut q, vec![]);
        assert_eq!(
            Vec::from(q),
            vec![
                vec![Ev::Abs(2, 2)],
                keys::tap(Mods::NONE, "a"),
                vec![Ev::Abs(4, 4)]
            ]
        );
    }

    #[test]
    fn a_click_ends_its_batch_and_keys_fill_the_queue() {
        let press = vec![Ev::Abs(5, 5), Ev::Btn("left", true)];
        let release = vec![Ev::Abs(5, 5), Ev::Btn("left", false)];
        let mut q = VecDeque::from(vec![press.clone(), release.clone(), vec![Ev::Abs(6, 6)]]);
        let mut queued = 0.0;
        assert_eq!(take_batch(&mut q, &mut queued), press);
        assert_eq!(take_batch(&mut q, &mut queued), release);
        assert_eq!(take_batch(&mut q, &mut queued), vec![Ev::Abs(6, 6)]);
        assert_eq!(queued, 0.0);
        // Two Cmd taps of 6 scancodes fill the 12; the third waits.
        let cmd_z = keys::tap(Mods::CMD, "z");
        let mut q = VecDeque::from(vec![cmd_z.clone(), cmd_z.clone(), cmd_z.clone()]);
        assert_eq!(take_batch(&mut q, &mut queued).len(), 8);
        assert_eq!((queued, q.len()), (12.0, 1));
        assert!(take_batch(&mut q, &mut queued).is_empty());
        // A tap bigger than the allowance goes once the queue is empty.
        let all = keys::tap(Mods::CMD | Mods::OPT | Mods::CTRL | Mods::SHIFT, "left");
        let mut q = VecDeque::from(vec![all.clone()]);
        queued = 0.0;
        assert_eq!(take_batch(&mut q, &mut queued), all);
    }

    #[test]
    fn the_image_takes_the_rows_between_the_bars() {
        assert_eq!(
            image_area(120, 40),
            Place {
                col: 0,
                row: 1,
                cols: 120,
                rows: 38
            }
        );
    }
}
