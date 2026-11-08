//! The interactive interface: the VMs, their details, and what can be done with them.
//!
//! Slow work (starting, stopping, creating, a bug report) runs on a thread of its own, which
//! reports its steps; the list shows each VM's step until it ends. Each VM's state is read
//! again every second, so VMs started or stopped elsewhere (the CLI, a printed script, closing
//! a VM's window) show as they are.
//!
//! A form whose submit starts slow work (a new VM's create, an edit's new installer) stays open
//! while it runs, showing its step, and closes once it succeeds: a failure shows in it, to fix
//! and submit again. Esc closes it meanwhile, and it opens again if the work fails.

use crate::cli::{self, delete_vm, describe, set_installer};
use crate::create::{self, DiskSource, ImportMode, Spec};
use crate::form::{self, Field, Form, opts};
use crate::gpu::GpuInfo;
use crate::host::{Host, Os, Profile};
use crate::update::Channel;
use crate::vm::{DiskController, Graphics, Registry, Vm, format_mib, parse_size_mib};
use crate::{install, logs, opencore, paths, reims_util, runtime, update, usb};
use anyhow::{Context, Result, bail};
use ratatui::DefaultTerminal;
use ratatui::Frame;
use ratatui::crossterm::event::{
    self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseButton, MouseEvent,
    MouseEventKind,
};
use ratatui::crossterm::execute;
use ratatui::crossterm::terminal::{EnterAlternateScreen, enable_raw_mode};
use ratatui::layout::{Constraint, Layout, Margin, Position, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Cell, Clear, Paragraph, Row, Table, TableState, Wrap};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::{Duration, Instant};

pub fn run() -> Result<()> {
    let host = Host::current();
    // The first run after an update brings the VMs an older reims-ctl made up to date: what it
    // changed shows first.
    let mut updated = Vec::new();
    crate::migrate::run(&mut |s| updated.push(s.to_string()));
    let mut app = App::new(host)?;
    if !updated.is_empty() {
        app.overlay = Some(Overlay::Text {
            title: "reims-ctl is updated  (Esc close)".into(),
            lines: updated,
            scroll: 0,
            kind: TextKind::Help,
        });
    }
    let mut terminal = ratatui::init();
    app.kitty_keys = ratatui::crossterm::terminal::supports_keyboard_enhancement().unwrap_or(false);
    let kitty_keys = app.kitty_keys;
    // ratatui's panic hook restores the terminal but leaves the mouse reported, and reports
    // would reach the shell as text; keys reported the kitty way would reach the next program
    // to use the alternate screen.
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = report_mouse(false);
        if kitty_keys {
            let _ = report_keys(false);
        }
        hook(info);
    }));
    let result = app.run(&mut terminal);
    let _ = app.hand_over();
    ratatui::restore();
    result?;
    if app.update_requested {
        return run_update(app.update_channel);
    }
    Ok(())
}

/// Runs the update in the plain terminal, then the new reims-ctl in this process's place.
fn run_update(channel: Option<Channel>) -> Result<()> {
    println!("Updating reims-ctl and its QEMU…\n");
    update::update(channel)?;
    println!("\nUpdated. Starting the new reims-ctl…");
    use std::os::unix::process::CommandExt;
    let exe = update::restart_path(std::env::current_exe()?);
    let err = std::process::Command::new(&exe).exec();
    Err(err).with_context(|| format!("run {}", exe.display()))
}

struct VmRow {
    name: String,
    dir: PathBuf,
    vm: Result<Vm, String>,
    pid: Option<i32>,
}

/// A work's messages carry its VM and the id `spawn` gave it.
enum Msg {
    Progress(String, u64, String),
    Done(String, u64, Result<String, String>),
    Gpus(Vec<GpuInfo>),
}

enum Overlay {
    Form(FormKind, Form),
    Text {
        title: String,
        lines: Vec<String>,
        scroll: usize,
        kind: TextKind,
    },
    Confirm {
        title: String,
        body: String,
        options: Vec<(&'static str, Action)>,
        choice: usize,
    },
    /// The host's USB devices, to give the VM `vm` or take back (usb.rs): `choice` is the
    /// selected row of [`usb::rows`].
    Usb {
        vm: String,
        devices: Vec<usb::HostDevice>,
        choice: usize,
    },
    /// A VM's menu, hanging from a cell: the one right-clicked, or the VM's name in the list.
    /// `all` holds every command, as Ctrl with Enter or the right click opens it.
    Menu {
        vm: String,
        choice: VmCommand,
        at: Position,
        all: bool,
    },
}

#[derive(Clone)]
enum FormKind {
    Create,
    Edit(String),
}

#[derive(Clone)]
enum TextKind {
    Command { vm: String, script: String },
    Log { vm: String, serial: bool },
    Help,
}

#[derive(Clone)]
enum Action {
    None,
    Delete { vm: String, keep_files: bool },
    ForceStop(String),
    UpdateOpencore(String),
    Update(Option<Channel>),
    Quit,
}

/// What can be done with the selected VM: each has a key on the main screen, and an item in
/// the VM's menu, which shows the key.
#[derive(Clone, Copy, Debug, PartialEq)]
enum VmCommand {
    Start,
    View,
    Stop,
    PowerOff,
    Edit,
    Usb,
    ShowCommand,
    ToggleLogs,
    ViewLogs,
    Report,
    UpdateOpencore,
    Delete,
}

impl VmCommand {
    /// Each with its key, in the menu's order.
    const ALL: [(VmCommand, char); 12] = [
        (VmCommand::Start, 's'),
        (VmCommand::View, 'v'),
        (VmCommand::Stop, 'x'),
        (VmCommand::PowerOff, 'K'),
        (VmCommand::Edit, 'e'),
        (VmCommand::Usb, 'p'),
        (VmCommand::ShowCommand, 'c'),
        (VmCommand::ToggleLogs, 'l'),
        (VmCommand::ViewLogs, 'L'),
        (VmCommand::Report, 'b'),
        (VmCommand::UpdateOpencore, 'U'),
        (VmCommand::Delete, 'd'),
    ];

    /// Its key in ALL, or a blank.
    fn key(self) -> char {
        Self::ALL.iter().find(|c| c.0 == self).map_or(' ', |c| c.1)
    }

    fn of_key(k: char) -> Option<VmCommand> {
        Self::ALL.iter().find(|c| c.1 == k).map(|c| c.0)
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Level {
    Info,
    Ok,
    Warn,
    Error,
}

struct App {
    host: Host,
    rows: Vec<VmRow>,
    table: TableState,
    overlay: Option<Overlay>,
    status: Option<(String, Level)>,
    activity: BTreeMap<String, String>,
    tx: Sender<Msg>,
    rx: Receiver<Msg>,
    gpus: Option<Vec<GpuInfo>>,
    qemu: Option<String>,
    last_state: Instant,
    last_registry: Instant,
    quit: bool,
    update_requested: bool,
    update_channel: Option<Channel>,
    pending_select: Option<String>,
    /// A VM's logs for the run loop to show in less, which needs the terminal.
    pending_page: Option<(String, Vec<PathBuf>)>,
    /// A VM for the run loop to show in the terminal viewer, which needs the terminal.
    pending_view: Option<Vm>,
    /// The macOS versions reims-util downloads, read when the first x86-64 VM's form opens.
    macos_versions: Option<Result<Vec<reims_util::Version>, String>>,
    /// The works `spawn` started: the last one's id.
    spawned: u64,
    /// The forms closed while the work they started runs, by its id: one whose work fails
    /// opens again.
    closed_forms: BTreeMap<u64, (FormKind, Form)>,
    /// The forms whose work failed while another view was open, to open once none is.
    failed_forms: Vec<(FormKind, Form)>,
    /// The terminal reports the mouse (report_mouse).
    mouse_reported: bool,
    /// The terminal speaks kitty's keyboard protocol, which tells Ctrl-Enter from Enter, and
    /// reports keys its way (report_keys).
    kitty_keys: bool,
    keys_reported: bool,
    /// Where the last frame drew the VM list's rows, and the VM menu, for the mouse.
    list_rows: Rect,
    menu_area: Rect,
}

impl App {
    /// The interface with no VMs read yet.
    fn with(host: Host) -> App {
        let (tx, rx) = channel();
        App {
            host,
            rows: Vec::new(),
            table: TableState::default(),
            overlay: None,
            status: None,
            activity: BTreeMap::new(),
            tx,
            rx,
            gpus: None,
            qemu: None,
            last_state: Instant::now(),
            last_registry: Instant::now(),
            quit: false,
            update_requested: false,
            update_channel: None,
            pending_select: None,
            pending_page: None,
            pending_view: None,
            macos_versions: None,
            spawned: 0,
            closed_forms: BTreeMap::new(),
            failed_forms: Vec::new(),
            mouse_reported: false,
            kitty_keys: false,
            keys_reported: false,
            list_rows: Rect::default(),
            menu_area: Rect::default(),
        }
    }

    fn new(host: Host) -> Result<App> {
        let mut app = App::with(host);
        app.qemu = host
            .default_profile()
            .and_then(|p| install::qemu_binary(p).ok())
            .map(|q| {
                install::qemu_version(&q)
                    .map(|v| v.replace("QEMU emulator version ", "QEMU "))
                    .unwrap_or_else(|| paths::tilde(&q))
            });
        app.reload()?;
        if !app.rows.is_empty() {
            app.table.select(Some(0));
        }
        if host.os == Os::Linux {
            let tx = app.tx.clone();
            std::thread::spawn(move || {
                let _ = tx.send(Msg::Gpus(crate::gpu::list()));
            });
        }
        if host.default_profile().is_none() {
            app.say(
                &format!("reims-ctl runs no VMs on {host} yet"),
                Level::Error,
            );
        } else if app.qemu.is_none() {
            app.say(
                "QEMU is not installed: run scripts/install-ctl.sh in your reims-vgpu checkout",
                Level::Warn,
            );
        }
        Ok(app)
    }

    fn say(&mut self, s: &str, level: Level) {
        self.status = Some((s.to_string(), level));
    }

    fn reload(&mut self) -> Result<()> {
        let reg = Registry::load()?;
        let selected = self.selected_name();
        self.rows = reg
            .vms
            .iter()
            .map(|e| {
                let vm = Vm::load(&e.dir).map_err(|err| format!("{err:#}"));
                let pid = vm.as_ref().ok().and_then(runtime::pid);
                VmRow {
                    name: e.name.clone(),
                    dir: e.dir.clone(),
                    vm,
                    pid,
                }
            })
            .collect();
        let idx = selected.and_then(|n| self.rows.iter().position(|r| r.name == n));
        self.table.select(match (idx, self.rows.is_empty()) {
            (_, true) => None,
            (Some(i), _) => Some(i),
            (None, _) => Some(self.table.selected().unwrap_or(0).min(self.rows.len() - 1)),
        });
        self.last_registry = Instant::now();
        self.last_state = Instant::now();
        Ok(())
    }

    fn refresh_state(&mut self) {
        for r in &mut self.rows {
            r.pid = r.vm.as_ref().ok().and_then(runtime::pid);
        }
        // Devices plugged in or out show as they are.
        if let Some(Overlay::Usb { devices, .. }) = &mut self.overlay {
            *devices = usb::host_devices();
        }
        self.last_state = Instant::now();
    }

    fn selected_name(&self) -> Option<String> {
        self.table
            .selected()
            .and_then(|i| self.rows.get(i))
            .map(|r| r.name.clone())
    }

    fn selected(&self) -> Option<&VmRow> {
        self.table.selected().and_then(|i| self.rows.get(i))
    }

    fn selected_vm(&mut self) -> Option<Vm> {
        let row = self.selected()?;
        match &row.vm {
            Ok(vm) => Some(vm.clone()),
            Err(e) => {
                let e = e.clone();
                self.say(&e, Level::Error);
                None
            }
        }
    }

    /// Runs `work` on a thread of its own and returns its id, or None, saying so, if the VM is
    /// busy.
    fn spawn<F>(&mut self, vm: &str, first: &str, work: F) -> Option<u64>
    where
        F: FnOnce(&mut dyn FnMut(&str)) -> Result<String> + Send + 'static,
    {
        if self.activity.contains_key(vm) {
            self.say(&format!("{vm} is busy: {}", self.activity[vm]), Level::Warn);
            return None;
        }
        self.activity.insert(vm.to_string(), first.to_string());
        self.spawned += 1;
        let id = self.spawned;
        let tx = self.tx.clone();
        let name = vm.to_string();
        std::thread::spawn(move || {
            let t = tx.clone();
            let n = name.clone();
            let mut progress = move |s: &str| {
                let _ = t.send(Msg::Progress(n.clone(), id, s.to_string()));
            };
            let r = work(&mut progress).map_err(|e| format!("{e:#}"));
            let _ = tx.send(Msg::Done(name, id, r));
        });
        Some(id)
    }

    /// `spawn`, for a form's submit: the form shows the work's steps until it ends.
    fn spawn_for<F>(&mut self, f: &mut Form, vm: &str, first: &str, work: F) -> Result<()>
    where
        F: FnOnce(&mut dyn FnMut(&str)) -> Result<String> + Send + 'static,
    {
        let id = self
            .spawn(vm, first, work)
            .with_context(|| format!("{vm} is busy"))?;
        f.busy = Some((id, first.to_string()));
        Ok(())
    }

    fn run(&mut self, terminal: &mut DefaultTerminal) -> Result<()> {
        while !self.quit {
            // The main screen and the VM menu take the mouse. The views and forms leave it to
            // the terminal, whose selection copies their text.
            let mouse = matches!(self.overlay, None | Some(Overlay::Menu { .. }));
            if mouse != self.mouse_reported {
                report_mouse(mouse)?;
                self.mouse_reported = mouse;
            }
            if self.kitty_keys != self.keys_reported {
                report_keys(self.kitty_keys)?;
                self.keys_reported = self.kitty_keys;
            }
            terminal.draw(|f| self.draw(f))?;
            if event::poll(Duration::from_millis(250))? {
                match event::read()? {
                    Event::Key(k) if k.kind == KeyEventKind::Press => self.key(k),
                    Event::Mouse(m) => self.mouse(m),
                    _ => {}
                }
            }
            if let Some((vm, files)) = self.pending_page.take() {
                self.page(terminal, &vm, &files)?;
            }
            if let Some(vm) = self.pending_view.take() {
                self.show_screen(terminal, &vm)?;
            }
            while let Ok(m) = self.rx.try_recv() {
                self.message(m);
            }
            if self.last_registry.elapsed() > Duration::from_secs(5) {
                let _ = self.reload();
            } else if self.last_state.elapsed() > Duration::from_secs(1) {
                self.refresh_state();
            }
        }
        Ok(())
    }

    /// Shows a VM's logs in less, on the plain terminal: its own selection then copies their
    /// text, a long line whole. Without less, the QEMU log shows in a view of reims-ctl's own.
    fn page(&mut self, terminal: &mut DefaultTerminal, vm: &str, files: &[PathBuf]) -> Result<()> {
        self.hand_over()?;
        ratatui::restore();
        let status = run_less(files);
        enable_raw_mode()?;
        execute!(std::io::stdout(), EnterAlternateScreen)?;
        terminal.clear()?;
        match status {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => self.open_log(vm, false),
            Err(e) => self.say(&format!("less: {e}"), Level::Error),
        }
        Ok(())
    }

    /// Shows a running VM's screen in the terminal (view.rs) until its Control-].
    fn show_screen(&mut self, terminal: &mut DefaultTerminal, vm: &Vm) -> Result<()> {
        self.hand_over()?;
        ratatui::restore();
        let result = crate::view::run(vm, &crate::view::Options::default());
        enable_raw_mode()?;
        execute!(std::io::stdout(), EnterAlternateScreen)?;
        terminal.clear()?;
        match result {
            Ok(Some(m)) => self.say(&m, Level::Info),
            Ok(None) => {}
            Err(e) => self.say(&format!("{}: {e:#}", vm.name()), Level::Error),
        }
        let _ = self.reload();
        Ok(())
    }

    fn message(&mut self, m: Msg) {
        match m {
            Msg::Progress(vm, id, s) => {
                self.status = Some((format!("{vm}: {s}"), Level::Info));
                if let Some(Overlay::Form(_, f)) = &mut self.overlay
                    && let Some((work, step)) = &mut f.busy
                    && *work == id
                {
                    *step = s.clone();
                }
                self.activity.insert(vm, s);
            }
            Msg::Done(vm, id, r) => {
                self.activity.remove(&vm);
                // The form that started the work: open, or closed while it ran.
                let form = match self.overlay.take() {
                    Some(Overlay::Form(kind, f)) if matches!(f.busy, Some((w, _)) if w == id) => {
                        Some((kind, f))
                    }
                    other => {
                        self.overlay = other;
                        self.closed_forms.remove(&id)
                    }
                };
                match r {
                    Ok(s) => self.say(&s, Level::Ok),
                    Err(e) => {
                        self.say(&format!("{vm}: {e}"), Level::Error);
                        // It opens again as it was submitted, with the error to fix.
                        if let Some((kind, mut f)) = form {
                            f.busy = None;
                            f.error = Some(e);
                            self.failed_forms.push((kind, f));
                        }
                    }
                }
                let _ = self.reload();
            }
            Msg::Gpus(list) => {
                let options = gpu_options(&list);
                self.gpus = Some(list);
                if let Some(Overlay::Form(_, f)) = &mut self.overlay {
                    f.set_options("gpu", options);
                }
            }
        }
        self.open_failed_form();
    }

    // ------------------------------------------------------------ keys

    fn key(&mut self, k: KeyEvent) {
        if k.code == KeyCode::Char('c')
            && k.modifiers.contains(KeyModifiers::CONTROL)
            && self.overlay.is_none()
        {
            self.try_quit();
            return;
        }
        match self.overlay.take() {
            Some(Overlay::Form(kind, mut f)) => match f.handle_key(k) {
                form::Event::None => self.overlay = Some(Overlay::Form(kind, f)),
                form::Event::Cancel => {
                    if let Some((id, _)) = f.busy {
                        self.closed_forms.insert(id, (kind, f));
                    }
                }
                form::Event::Submit => match self.submit(&kind, &mut f) {
                    Ok(()) if f.busy.is_none() => {}
                    // The work it started runs: it stays open, until that succeeds.
                    Ok(()) => self.overlay = Some(Overlay::Form(kind, f)),
                    Err(e) => {
                        f.error = Some(format!("{e:#}"));
                        self.overlay = Some(Overlay::Form(kind, f));
                    }
                },
            },
            Some(Overlay::Text {
                title,
                lines,
                scroll,
                kind,
            }) => self.text_key(k, title, lines, scroll, kind),
            Some(Overlay::Confirm {
                title,
                body,
                options,
                choice,
            }) => match k.code {
                KeyCode::Left | KeyCode::Up | KeyCode::BackTab => {
                    let c = (choice + options.len() - 1) % options.len();
                    self.overlay = Some(Overlay::Confirm {
                        title,
                        body,
                        options,
                        choice: c,
                    });
                }
                KeyCode::Right | KeyCode::Down | KeyCode::Tab => {
                    let c = (choice + 1) % options.len();
                    self.overlay = Some(Overlay::Confirm {
                        title,
                        body,
                        options,
                        choice: c,
                    });
                }
                KeyCode::Enter => self.act(options[choice].1.clone()),
                KeyCode::Esc | KeyCode::Char('q') | KeyCode::Char('n') => {}
                _ => {
                    self.overlay = Some(Overlay::Confirm {
                        title,
                        body,
                        options,
                        choice,
                    })
                }
            },
            Some(Overlay::Menu {
                vm,
                choice,
                at,
                all,
            }) => self.menu_key(k, vm, choice, at, all),
            Some(Overlay::Usb {
                vm,
                devices,
                choice,
            }) => self.usb_key(k, vm, devices, choice),
            None => self.main_key(k),
        }
        self.open_failed_form();
    }

    /// Opens a form whose work failed, once no other view is open.
    fn open_failed_form(&mut self) {
        if self.overlay.is_none()
            && let Some((kind, f)) = self.failed_forms.pop()
        {
            self.overlay = Some(Overlay::Form(kind, f));
        }
    }

    fn menu_key(&mut self, k: KeyEvent, vm: String, choice: VmCommand, at: Position, all: bool) {
        let items = self.menu_items(&vm, all);
        if items.is_empty() {
            return;
        }
        let i = menu_index(&items, choice);
        let n = items.len();
        let ctrl = k.modifiers.contains(KeyModifiers::CONTROL);
        let choice = match k.code {
            KeyCode::Esc | KeyCode::Char('q') => return,
            KeyCode::Char('c') if ctrl => return,
            KeyCode::Down | KeyCode::Char('j') | KeyCode::Tab => items[(i + 1) % n].0,
            KeyCode::Up | KeyCode::Char('k') | KeyCode::BackTab => items[(i + n - 1) % n].0,
            KeyCode::Home => items[0].0,
            KeyCode::End => items[n - 1].0,
            KeyCode::Enter => return self.menu_command(&vm, items[i].0),
            KeyCode::Char(c) if !ctrl => match VmCommand::of_key(c) {
                Some(c) if items.iter().any(|it| it.0 == c) => return self.menu_command(&vm, c),
                _ => choice,
            },
            _ => choice,
        };
        self.overlay = Some(Overlay::Menu {
            vm,
            choice,
            at,
            all,
        });
    }

    fn mouse(&mut self, m: MouseEvent) {
        match self.overlay.take() {
            Some(Overlay::Menu {
                vm,
                choice,
                at,
                all,
            }) => self.menu_mouse(m, vm, choice, at, all),
            None => self.main_mouse(m),
            // The terminal reports the mouse only on the main screen and in the menu.
            other => self.overlay = other,
        }
    }

    /// A click selects a VM, a right click opens its menu there (with Ctrl, the menu of
    /// everything), and the wheel moves down and up the list, as the arrow keys do (as most
    /// terminals' wheels do on a screen of its own that leaves the mouse to them).
    fn main_mouse(&mut self, m: MouseEvent) {
        let at = Position::new(m.column, m.row);
        match m.kind {
            MouseEventKind::Down(b @ (MouseButton::Left | MouseButton::Right)) => {
                let Some(i) = self.row_at(at) else { return };
                self.table.select(Some(i));
                if b == MouseButton::Right {
                    self.open_menu(Some(at), m.modifiers.contains(KeyModifiers::CONTROL));
                }
            }
            MouseEventKind::ScrollDown => self.main_key(KeyCode::Down.into()),
            MouseEventKind::ScrollUp => self.main_key(KeyCode::Up.into()),
            _ => {}
        }
    }

    /// A click on an item does it, and one outside the menu closes it: a right click on
    /// another VM opens that VM's menu. The wheel moves through the items, as ↓ and ↑ do.
    fn menu_mouse(
        &mut self,
        m: MouseEvent,
        vm: String,
        choice: VmCommand,
        at: Position,
        all: bool,
    ) {
        let items = self.menu_items(&vm, all);
        let pos = Position::new(m.column, m.row);
        let inner = self.menu_area.inner(Margin::new(1, 1));
        let row = inner.contains(pos).then(|| usize::from(pos.y - inner.y));
        match (m.kind, row.and_then(|i| items.get(i))) {
            (MouseEventKind::Down(_), _) if !self.menu_area.contains(pos) => self.main_mouse(m),
            (MouseEventKind::Down(MouseButton::Left | MouseButton::Right), Some(&(c, _))) => {
                self.menu_command(&vm, c)
            }
            (MouseEventKind::ScrollDown, _) => {
                self.menu_key(KeyCode::Down.into(), vm, choice, at, all)
            }
            (MouseEventKind::ScrollUp, _) => self.menu_key(KeyCode::Up.into(), vm, choice, at, all),
            _ => {
                self.overlay = Some(Overlay::Menu {
                    vm,
                    choice,
                    at,
                    all,
                })
            }
        }
    }

    /// The VM whose row of the list holds the cell `at`.
    fn row_at(&self, at: Position) -> Option<usize> {
        let r = self.list_rows;
        let i = self.table.offset() + r.contains(at).then(|| usize::from(at.y - r.y))?;
        (i < self.rows.len()).then_some(i)
    }

    /// Opens the selected VM's menu, hanging from `at` or else from the VM's name in the list:
    /// with `all`, the menu of every command.
    fn open_menu(&mut self, at: Option<Position>, all: bool) {
        let Some(i) = self.table.selected() else {
            return;
        };
        let Some(row) = self.rows.get(i) else { return };
        let Some(&(first, _)) = vm_menu(row, all, self.host).first() else {
            return;
        };
        // The name shows on the VM's row, past the highlight symbol.
        let r = self.list_rows;
        let y = r.y + i.saturating_sub(self.table.offset()) as u16;
        self.overlay = Some(Overlay::Menu {
            vm: row.name.clone(),
            choice: first,
            at: at.unwrap_or(Position::new(r.x + 2, y.min(r.bottom().saturating_sub(1)))),
            all,
        });
    }

    /// The menu of the VM named `name`, with each item's label, of every command with `all`;
    /// empty when the list no longer has it.
    fn menu_items(&self, name: &str, all: bool) -> Vec<(VmCommand, &'static str)> {
        self.rows
            .iter()
            .find(|r| r.name == name)
            .map(|r| vm_menu(r, all, self.host))
            .unwrap_or_default()
    }

    /// Does `c` to the VM named `vm`, the one a menu was opened on, which another VM may have
    /// replaced as the selected one since (a VM created selects itself when it appears).
    fn menu_command(&mut self, vm: &str, c: VmCommand) {
        let Some(i) = self.rows.iter().position(|r| r.name == vm) else {
            return;
        };
        self.table.select(Some(i));
        self.command(c);
    }

    fn text_key(
        &mut self,
        k: KeyEvent,
        title: String,
        lines: Vec<String>,
        scroll: usize,
        kind: TextKind,
    ) {
        // Scrolling counts the rows the lines wrap to in the terminal as it is now.
        let (w, h) = ratatui::crossterm::terminal::size().unwrap_or((120, 40));
        let (width, height) = text_inner(Rect::new(0, 0, w, h));
        let max = wrap_rows(&lines, width).len().saturating_sub(height);
        let page = height.max(1);
        let mut scroll = scroll.min(max);
        match k.code {
            KeyCode::Esc | KeyCode::Char('q') => return,
            KeyCode::Down | KeyCode::Char('j') => scroll = (scroll + 1).min(max),
            KeyCode::Up | KeyCode::Char('k') => scroll = scroll.saturating_sub(1),
            KeyCode::PageDown | KeyCode::Char(' ') => scroll = (scroll + page).min(max),
            KeyCode::PageUp => scroll = scroll.saturating_sub(page),
            KeyCode::Home | KeyCode::Char('g') => scroll = 0,
            KeyCode::End | KeyCode::Char('G') => scroll = max,
            KeyCode::Char('w') => {
                if let TextKind::Command { vm, script } = &kind {
                    let r = self
                        .rows
                        .iter()
                        .find(|r| &r.name == vm)
                        .map(|r| r.dir.join("launch.sh"));
                    if let Some(p) = r {
                        use std::os::unix::fs::PermissionsExt;
                        let res = std::fs::write(&p, script).and_then(|_| {
                            std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755))
                        });
                        match res {
                            Ok(()) => self.say(&format!("wrote {}", paths::tilde(&p)), Level::Ok),
                            Err(e) => self.say(&format!("{}: {e}", p.display()), Level::Error),
                        }
                    }
                }
            }
            KeyCode::Char('y') => {
                if let TextKind::Command { script, .. } = &kind {
                    match copy_to_clipboard(script) {
                        Some(tool) => self.say(&format!("copied the command ({tool})"), Level::Ok),
                        None => self.say("no clipboard tool found (pbcopy, wl-copy, xclip or xsel): press w to write it to a file", Level::Warn),
                    }
                }
            }
            KeyCode::Char('s') => {
                if let TextKind::Log { vm, serial } = &kind {
                    let vm = vm.clone();
                    let serial = !*serial;
                    self.open_log(&vm, serial);
                    return;
                }
            }
            KeyCode::Char('r') => {
                if let TextKind::Log { vm, serial } = &kind {
                    let (vm, serial) = (vm.clone(), *serial);
                    self.open_log(&vm, serial);
                    return;
                }
            }
            _ => {}
        }
        self.overlay = Some(Overlay::Text {
            title,
            lines,
            scroll,
            kind,
        });
    }

    fn try_quit(&mut self) {
        if self.activity.is_empty() {
            self.quit = true;
            return;
        }
        let busy = self
            .activity
            .iter()
            .map(|(k, v)| format!("{k}: {v}"))
            .collect::<Vec<_>>()
            .join("\n");
        self.overlay = Some(Overlay::Confirm {
            title: "Quit while work is running?".into(),
            body: format!(
                "{busy}\n\nQuitting stops this work where it is: a guest asked to shut down still does, \
                 but is no longer powered off if it does not; a VM being created is left unfinished, \
                 though a download of macOS picks up where it stopped when you create it again."
            ),
            options: vec![("Keep working", Action::None), ("Quit", Action::Quit)],
            choice: 0,
        });
    }

    fn main_key(&mut self, k: KeyEvent) {
        if let KeyCode::Char(c) = k.code
            && let Some(c) = VmCommand::of_key(c)
        {
            return self.command(c);
        }
        let n = self.rows.len();
        match k.code {
            KeyCode::Char('q') | KeyCode::Esc => self.try_quit(),
            KeyCode::Down | KeyCode::Char('j') if n > 0 => self
                .table
                .select(Some((self.table.selected().unwrap_or(0) + 1).min(n - 1))),
            KeyCode::Up | KeyCode::Char('k') if n > 0 => self
                .table
                .select(Some(self.table.selected().unwrap_or(0).saturating_sub(1))),
            KeyCode::Char('?') | KeyCode::Char('h') | KeyCode::F(1) => self.overlay = Some(help()),
            KeyCode::Char('n') => self.open_create(),
            KeyCode::Enter => self.open_menu(None, k.modifiers.contains(KeyModifiers::CONTROL)),
            KeyCode::Delete => self.ask_delete(),
            KeyCode::Char('u') => self.ask_update(),
            KeyCode::Char('r') => {
                let _ = self.reload();
                self.say("reloaded", Level::Info);
            }
            _ => {}
        }
    }

    fn command(&mut self, c: VmCommand) {
        match c {
            VmCommand::Start => self.start(),
            VmCommand::View => self.view_screen(),
            VmCommand::Stop => self.stop(false),
            VmCommand::PowerOff => self.ask_force_stop(),
            VmCommand::Edit => self.open_edit(),
            VmCommand::Usb => self.open_usb(),
            VmCommand::ShowCommand => self.show_command(),
            VmCommand::ToggleLogs => self.toggle_logs(),
            VmCommand::ViewLogs => self.view_logs(),
            VmCommand::Report => self.report(),
            VmCommand::UpdateOpencore => self.ask_update_opencore(),
            VmCommand::Delete => self.ask_delete(),
        }
    }

    /// Stops the mouse being reported, whatever the tracked state says, before less, the viewer
    /// or the shell takes the terminal. What the terminal reported before it stopped, such as a
    /// click's release or the wheel still turning, comes before its answer to where its cursor
    /// is, so it is read here, not by them. Keys reported the kitty way stop too.
    fn hand_over(&mut self) -> Result<()> {
        report_mouse(false)?;
        self.mouse_reported = false;
        if self.keys_reported {
            report_keys(false)?;
            self.keys_reported = false;
        }
        let _ = ratatui::crossterm::cursor::position();
        Ok(())
    }

    fn act(&mut self, a: Action) {
        match a {
            Action::None => {}
            Action::Quit => self.quit = true,
            Action::Update(channel) => {
                self.update_requested = true;
                self.update_channel = channel;
                self.quit = true;
            }
            Action::Delete { vm, keep_files } => match delete_vm(&vm, keep_files) {
                Ok(s) => {
                    self.say(&s, Level::Ok);
                    let _ = self.reload();
                }
                Err(e) => self.say(&format!("{e:#}"), Level::Error),
            },
            Action::UpdateOpencore(name) => {
                self.spawn(&name.clone(), "making a new OpenCore image", move |p| {
                    opencore::renew(&mut Registry::load()?.open(&name)?, None, p)
                });
            }
            Action::ForceStop(name) => {
                let Some(vm) = self
                    .rows
                    .iter()
                    .find(|r| r.name == name)
                    .and_then(|r| r.vm.as_ref().ok())
                    .cloned()
                else {
                    return;
                };
                self.activity.remove(&name);
                self.spawn(&name, "powering off", move |p| {
                    runtime::stop(&vm, true, None, |s| p(s))?;
                    Ok(format!("{}: powered off", vm.name()))
                });
            }
        }
    }

    // ------------------------------------------------------------ actions

    fn start(&mut self) {
        let Some(vm) = self.selected_vm() else { return };
        if runtime::running(&vm) {
            self.say(&format!("{} is already running", vm.name()), Level::Info);
            return;
        }
        // What runtime::warnings finds shows in the VM's details, not here.
        let host = self.host;
        self.spawn(&vm.config.name.clone(), "starting", move |_| {
            // What the start did to its settings first, and why, follows the start's line.
            let mut vm = vm;
            let mut said = Vec::new();
            let warned = runtime::prepare(&mut vm, &mut |s| said.push(s.to_string()));
            let pid = runtime::start(&vm, host)?;
            let mut s = format!("{}: running (pid {pid}); its window is open", vm.name());
            for n in said.iter().chain(&warned) {
                s.push_str("; ");
                s.push_str(n);
            }
            Ok(s)
        });
    }

    fn stop(&mut self, force: bool) {
        let Some(vm) = self.selected_vm() else { return };
        if !runtime::running(&vm) {
            self.say(&format!("{} is not running", vm.name()), Level::Info);
            return;
        }
        let timeout = vm.config.stop_timeout();
        let first = if force {
            "powering off"
        } else {
            "shutting down"
        };
        self.spawn(&vm.config.name.clone(), first, move |p| {
            runtime::stop(&vm, force, timeout, |s| p(s))?;
            Ok(format!("{}: stopped", vm.name()))
        });
    }

    fn ask_force_stop(&mut self) {
        let Some(row) = self.selected() else { return };
        if row.pid.is_none() {
            let n = row.name.clone();
            self.say(&format!("{n} is not running"), Level::Info);
            return;
        }
        let name = row.name.clone();
        self.overlay = Some(Overlay::Confirm {
            title: format!("Power off {name}?"),
            body: "Powering off is pulling the plug: whatever the guest has not written to its disk is lost. \
                   Stop (x) asks the guest to shut down first."
                .into(),
            options: vec![("Cancel", Action::None), ("Power off", Action::ForceStop(name))],
            choice: 0,
        });
    }

    fn ask_delete(&mut self) {
        let Some(row) = self.selected() else { return };
        let name = row.name.clone();
        if row.pid.is_some() {
            self.say(&format!("{name} is running: stop it first"), Level::Warn);
            return;
        }
        let dir = paths::tilde(&row.dir);
        self.overlay = Some(Overlay::Confirm {
            title: format!("Delete {name}?"),
            body: format!(
                "Deleting the VM and its files removes {dir}, with its disks, for good. Keeping its \
                 files only takes it off this list. A disk used in place, outside that directory, stays either way."
            ),
            options: vec![
                ("Cancel", Action::None),
                (
                    "Delete it and its files",
                    Action::Delete {
                        vm: name.clone(),
                        keep_files: false,
                    },
                ),
                (
                    "Delete it, keep its files",
                    Action::Delete {
                        vm: name,
                        keep_files: true,
                    },
                ),
            ],
            choice: 0,
        });
    }

    fn ask_update_opencore(&mut self) {
        let Some(vm) = self.selected_vm() else { return };
        let name = vm.name().to_string();
        let Some(q) = &vm.config.q35 else {
            self.say(
                &format!("{name} boots no OpenCore: only x86-64 VMs do"),
                Level::Info,
            );
            return;
        };
        if runtime::running(&vm) {
            self.say(&format!("{name} is running: stop it first"), Level::Warn);
            return;
        }
        let image = vm.path(&q.opencore);
        let id = match opencore::Config::read(&image, q.opencore_format).and_then(|c| c.identity())
        {
            Ok(id) => id,
            Err(e) => {
                let image = paths::tilde(&image);
                self.say(
                    &format!("{name}: read the Mac {image} presents: {e:#}"),
                    Level::Error,
                );
                return;
            }
        };
        self.overlay = Some(Overlay::Confirm {
            title: format!("Make {name} a new OpenCore image?"),
            body: format!(
                "reims-util makes it a new OpenCore image for its graphics, {}, with what it \
                 carries now, and on reims-vgpu the macOS drivers of its GPU device that this \
                 reims-ctl carries. It presents the same Mac, a {} with serial {}, and the VM \
                 keeps its OVMF variables, which hold its boot entries, and its boot selection \
                 (e changes it).\n\nThe image it replaces is deleted, with anything else \
                 changed in it.",
                vm.config.graphics().id(),
                id.model,
                id.serial
            ),
            options: vec![
                ("Cancel", Action::None),
                ("Make it", Action::UpdateOpencore(name)),
            ],
            choice: 0,
        });
    }

    fn ask_update(&mut self) {
        let Some(i) = install::Install::load() else {
            self.say(
                "no install record: run scripts/install-ctl.sh in your reims-vgpu checkout once",
                Level::Error,
            );
            return;
        };
        let running = self.rows.iter().filter(|r| r.pid.is_some()).count();
        let mut note = if running > 0 {
            format!("\n\n{running} VM(s) running keep their current QEMU until they restart.")
        } else {
            String::new()
        };
        if !self.activity.is_empty() {
            let busy = self
                .activity
                .iter()
                .map(|(k, v)| format!("{k}: {v}"))
                .collect::<Vec<_>>()
                .join("\n");
            note.push_str(&format!(
                "\n\nThe restart stops this work where it is, as quitting does (a download of \
                 macOS picks up where it stopped):\n{busy}"
            ));
        }
        let current = update::current_branch(&i.repo)
            .as_deref()
            .and_then(Channel::of);
        // __record-install installs reims-util where the host makes x86-64 VMs, or where one is
        // beside reims-ctl (cli.rs).
        let what = if self.host.default_profile() == Some(Profile::Q35X86_64)
            || reims_util::beside().is_some()
        {
            "reims-ctl, the library, QEMU and reims-util"
        } else {
            "reims-ctl, the library and QEMU"
        };
        let mut options = vec![("Cancel", Action::None), ("Update", Action::Update(None))];
        for c in Channel::ALL.into_iter().filter(|&c| Some(c) != current) {
            let label = match c {
                Channel::Stable => "Switch to stable",
                Channel::Development => "Switch to development",
            };
            options.push((label, Action::Update(Some(c))));
        }
        self.overlay = Some(Overlay::Confirm {
            title: "Update reims-ctl?".into(),
            body: format!(
                "This pulls {}, which follows {}, and rebuilds {what} (a few minutes; the build's \
                 output shows here), then restarts reims-ctl.\n\nA switch checks out that channel's branch first, and \
                 later updates stay on it: stable is master, and development gets changes first.{note}",
                paths::tilde(&i.repo),
                update::following(&i.repo)
            ),
            options,
            choice: 1,
        });
    }

    /// Opens the selected VM's USB devices, where its host passes them through.
    fn open_usb(&mut self) {
        let Some(vm) = self.selected_vm() else { return };
        if let Err(why) = usb_available(self.host, &vm) {
            return self.say(&why, Level::Warn);
        }
        self.overlay = Some(Overlay::Usb {
            vm: vm.config.name.clone(),
            devices: usb::host_devices(),
            choice: 0,
        });
    }

    /// The rows of the USB devices' view of the VM named `vm`: the host's `devices`, each
    /// with what of the VM's settings, as the list last read them, gives it.
    fn usb_rows(&self, vm: &str, devices: &[usb::HostDevice]) -> Vec<usb::Row> {
        let chosen = self
            .rows
            .iter()
            .find(|r| r.name == vm)
            .and_then(|r| r.vm.as_ref().ok())
            .map(|vm| vm.config.usb.clone())
            .unwrap_or_default();
        usb::rows(&chosen, devices)
    }

    fn usb_key(
        &mut self,
        k: KeyEvent,
        vm: String,
        mut devices: Vec<usb::HostDevice>,
        choice: usize,
    ) {
        let rows = self.usb_rows(&vm, &devices);
        let last = rows.len().saturating_sub(1);
        let mut choice = choice.min(last);
        match k.code {
            KeyCode::Esc | KeyCode::Char('q') => return,
            KeyCode::Down | KeyCode::Char('j') | KeyCode::Tab => choice = (choice + 1).min(last),
            KeyCode::Up | KeyCode::Char('k') | KeyCode::BackTab => {
                choice = choice.saturating_sub(1)
            }
            KeyCode::Home => choice = 0,
            KeyCode::End => choice = last,
            KeyCode::Char('r') => {
                devices = usb::host_devices();
                self.say("rescanned this host's USB devices", Level::Info);
            }
            KeyCode::Char(' ') | KeyCode::Enter => {
                if let Some(row) = rows.get(choice) {
                    self.toggle_usb(&vm, &devices, row);
                }
            }
            _ => {}
        }
        self.overlay = Some(Overlay::Usb {
            vm,
            devices,
            choice,
        });
    }

    /// Gives the VM named `name` the device of `row`, or takes it back: at once while it runs
    /// (usb::apply).
    fn toggle_usb(&mut self, name: &str, devices: &[usb::HostDevice], row: &usb::Row) {
        let Some(vm) = self
            .rows
            .iter()
            .find(|r| r.name == name)
            .and_then(|r| r.vm.as_ref().ok())
            .cloned()
        else {
            return;
        };
        match usb::apply(&vm, |cur| usb::toggle(cur, devices, row)) {
            Ok((_, said)) => {
                let failed = said
                    .iter()
                    .any(|s| s.contains("failed") || s.contains("can't"));
                self.say(
                    &said.join("; "),
                    if failed { Level::Warn } else { Level::Ok },
                );
            }
            Err(e) => self.say(&format!("{e:#}"), Level::Error),
        }
        let _ = self.reload();
    }

    fn toggle_logs(&mut self) {
        let Some(vm) = self.selected_vm() else {
            return;
        };
        // As saved now, not as the list last read it: the toggle saves the whole of it.
        let mut vm = match Vm::load(&vm.dir) {
            Ok(vm) => vm,
            Err(e) => return self.say(&format!("{e:#}"), Level::Error),
        };
        vm.config.logs.enabled = !vm.config.logs.enabled;
        match vm.save() {
            Ok(()) => {
                let on = vm.config.logs.enabled;
                let when = if runtime::running(&vm) {
                    " from its next start"
                } else {
                    ""
                };
                self.say(
                    &format!(
                        "{}: logs {}{when}{}",
                        vm.name(),
                        if on { "on" } else { "off" },
                        if on {
                            " (b makes a bug report of them)"
                        } else {
                            ""
                        }
                    ),
                    Level::Ok,
                );
                let _ = self.reload();
            }
            Err(e) => self.say(&format!("{e:#}"), Level::Error),
        }
    }

    fn report(&mut self) {
        let Some(vm) = self.selected_vm() else { return };
        let host = self.host;
        let logs_on = vm.config.logs.enabled;
        self.spawn(&vm.config.name.clone(), "writing a bug report", move |_| {
            let p = logs::bundle(&vm, host)?;
            let mut s = format!("wrote {}", paths::tilde(&p));
            if !logs_on {
                s.push_str(
                    " · logs were off: turn them on (l), reproduce the problem, and report again",
                );
            }
            Ok(s)
        });
    }

    fn show_command(&mut self) {
        let Some(vm) = self.selected_vm() else { return };
        match runtime::invocation_for_print(&vm, self.host) {
            Ok(inv) => {
                let script = inv.shell_script(&cli::script_comment(&vm));
                self.overlay = Some(Overlay::Text {
                    title: format!(
                        "{}: the QEMU command  (w write launch.sh · y copy · Esc close)",
                        vm.name()
                    ),
                    lines: script.lines().map(String::from).collect(),
                    scroll: 0,
                    kind: TextKind::Command {
                        vm: vm.config.name.clone(),
                        script,
                    },
                });
            }
            Err(e) => self.say(&format!("{e:#}"), Level::Error),
        }
    }

    /// The selected VM's QEMU log, then its serial console, for the run loop to show in less.
    fn view_screen(&mut self) {
        let Some(r) = self.selected() else { return };
        if r.pid.is_none() {
            let name = r.name.clone();
            self.say(&format!("{name} is not running: s starts it"), Level::Warn);
            return;
        }
        self.pending_view = self.selected_vm();
    }

    fn view_logs(&mut self) {
        let Some(vm) = self.selected_vm() else { return };
        let files: Vec<PathBuf> = [vm.qemu_log(), vm.serial_log()]
            .into_iter()
            .filter(|f| f.exists())
            .collect();
        if files.is_empty() {
            let p = paths::tilde(&vm.qemu_log());
            self.say(
                &format!("{}: {p} does not exist yet", vm.name()),
                Level::Info,
            );
            return;
        }
        self.pending_page = Some((vm.config.name.clone(), files));
    }

    fn open_log(&mut self, name: &str, serial: bool) {
        let Some(vm) = self
            .rows
            .iter()
            .find(|r| r.name == name)
            .and_then(|r| r.vm.as_ref().ok())
            .cloned()
        else {
            return;
        };
        let file = if serial {
            vm.serial_log()
        } else {
            vm.qemu_log()
        };
        let text = if file.exists() {
            runtime::tail(&file, 2000)
        } else {
            format!("{} does not exist yet", paths::tilde(&file))
        };
        let lines: Vec<String> = text.lines().map(String::from).collect();
        // Its end: drawing and the keys clamp this to the last screenful.
        let scroll = usize::MAX;
        let which = if serial { "serial console" } else { "QEMU log" };
        self.overlay = Some(Overlay::Text {
            title: format!(
                "{name}: {which}  (s {} · r reload · Esc close)",
                if serial { "QEMU log" } else { "serial console" }
            ),
            lines,
            scroll,
            kind: TextKind::Log {
                vm: name.to_string(),
                serial,
            },
        });
    }

    // ------------------------------------------------------------ forms

    fn gpu_field(&self, current: &crate::vm::Gpu) -> Field {
        let mut options = gpu_options(self.gpus.as_deref().unwrap_or(&[]));
        let selected = gpu_id(current);
        if !options.iter().any(|o| o.0 == selected) {
            options.push((
                selected.clone(),
                current.name.clone().unwrap_or_else(|| selected.clone()),
            ));
        }
        let help = if self.gpus.is_none() {
            "Looking for the host's GPUs…"
        } else {
            "The GPU the library draws on. Automatic takes the first discrete GPU, then the first integrated one."
        };
        Field::choice("gpu", "GPU", options, &selected, help)
    }

    fn open_create(&mut self) {
        let Some(profile) = self.host.default_profile() else {
            self.say(crate::host::UNSUPPORTED, Level::Error);
            return;
        };
        let mut fields = vec![
            Field::text(
                "name",
                "Name",
                "",
                "Letters, digits, '-', '_' and '.'. Empty names a VM of downloaded macOS after it.",
            )
            .placeholder("the macOS version's, such as tahoe"),
        ];
        if !matches!(self.macos_versions, Some(Ok(_))) {
            self.macos_versions = Some(reims_util::versions().map_err(|e| format!("{e:#}")));
        }
        let (options, version_help) = match &self.macos_versions {
            Some(Ok(list)) if !list.is_empty() => (
                list.iter()
                    .map(|v| (v.slug.clone(), v.name.clone()))
                    .collect(),
                "The macOS to download from Apple. x86-64 macOS 26 is the one we've tested."
                    .to_string(),
            ),
            Some(Ok(_)) => (
                opts(&[("", "none")]),
                "reims-util lists no versions.".to_string(),
            ),
            Some(Err(e)) => (opts(&[("", "unavailable")]), e.clone()),
            None => (opts(&[("", "unavailable")]), String::new()),
        };
        let first = options.first().map(|o| o.0.clone()).unwrap_or_default();
        fields.extend([
            Field::choice("source", "macOS from", opts(&[("download", "Download it from Apple"), ("installer", "An installer image of my own"), ("disk", "A disk that has macOS already")]), "download", "reims-ctl downloads the installer, makes the VM's OpenCore image and a new disk, and the VM boots into the installer."),
            Field::choice("macos", "macOS version", options, &first, &version_help).show_if("source", &["download"]),
            Field::choice("installer_kind", "Installer", opts(&[("online", "Online, about 1 GB"), ("offline", "Offline, 13 to 20 GB")]), "online", "Online is Apple's recovery image: macOS downloads the newest release as it installs. Offline holds one release whole, and installs with no network (macOS 11 and later). Either is kept for the next VM, and a stopped download picks up where it stopped.").show_if("source", &["download"]),
            Field::text("release", "Release", "", "Optional: an exact release of that macOS, such as 26.6.2; empty for the newest. reims-util offline-installer list lists them.").placeholder("the newest").show_if("installer_kind", &["offline"]),
            Field::path("installer", "Installer image", "", "An installer to boot (Tab completes): a raw image, such as a recovery image's BaseSystem.img, or a DMG, converted to a raw image in the VM's directory. Eject it once macOS is installed (e).").show_if("source", &["installer"]),
            Field::text("disk_size", "Disk size", "128G", "The new disk's size; it takes only what the guest writes.").show_if("source", &["download", "installer"]),
            Field::path("disk_path", "Disk image", "", "A qcow2 or raw image (Tab completes).").show_if("source", &["disk"]),
            Field::choice("disk_mode", "Use the disk by", opts(&[("copy", "Copying it (instant where the file system reflinks)"), ("in-place", "Using it in place")]), "copy", "A copy leaves the image as it was; in place, the VM writes the image itself.")
                .show_if("source", &["disk"]),
            Field::path("opencore", "OpenCore image", "", "Empty makes one for this VM, with OVMF, presenting the Mac of the image beside an existing disk (opencore.qcow2, as a VM's directory holds them), unless the disk is copied from a VM reims-ctl has, else a Mac identity of its own. Or an OpenCore image of yours to copy, with OVMF_CODE.fd and OVMF_VARS.fd beside it, such as the one a disk of macOS booted from, which keeps its Mac's identity.")
                .placeholder("made for this VM"),
        ]);
        fields.extend([
            Field::path(
                "dir",
                "Storage directory",
                "",
                "Where its disks and settings go (Tab completes). Empty for the default.",
            )
            .placeholder(&format!(
                "{}/<name>",
                paths::tilde(&paths::default_vms_dir())
            )),
            Field::text(
                "cpus",
                "CPU cores",
                &create::default_cpus().to_string(),
                &format!("vCPUs; this host has {}.", create::host_cpus()),
            ),
            Field::text(
                "memory",
                "Memory",
                &format_mib(create::default_memory_mib()),
                &format!(
                    "Such as 8G; this host has {}.",
                    format_mib(create::host_memory_mib() / 1024 * 1024)
                ),
            ),
        ]);
        fields.push(graphics_field(Graphics::default()));
        if profile.gpu_choice(self.host) {
            fields.push(
                self.gpu_field(&crate::vm::Gpu::default())
                    .show_if("graphics", LIBRARY),
            );
        }
        fields.extend([
            Field::text("resolution", "Resolution", "1920x1080", RESOLUTION_HELP)
                .show_if("graphics", LIBRARY),
            Field::toggle(
                "logs",
                "Logs",
                false,
                "Keep the library's log and the serial console, for bug reports.",
            ),
            Field::text(
                "ssh_port",
                "SSH port",
                "",
                "Optional: a port on 127.0.0.1 forwarded to the guest's SSH (Remote Login).",
            )
            .placeholder("optional"),
        ]);
        let form = Form::new("New VM", "create", fields);
        self.overlay = Some(Overlay::Form(FormKind::Create, form));
    }

    fn open_edit(&mut self) {
        let Some(vm) = self.selected_vm() else { return };
        let c = &vm.config;
        let mut fields = vec![
            Field::text(
                "cpus",
                "CPU cores",
                &c.cpus.to_string(),
                &format!("vCPUs; this host has {}.", create::host_cpus()),
            ),
            Field::text("memory", "Memory", &format_mib(c.memory_mib), "Such as 8G."),
        ];
        // The resolution's, borderless's and the GPU's fields are reims-vgpu's.
        fields.push(graphics_field(c.graphics()));
        fields.extend([
            Field::text(
                "resolution",
                "Resolution",
                &c.display.resolution,
                RESOLUTION_HELP,
            )
            .show_if("graphics", LIBRARY),
            Field::toggle(
                "borderless",
                "Borderless",
                c.display.borderless,
                "A window without borders that fills its screen.",
            )
            .show_if("graphics", LIBRARY),
        ]);
        if c.profile.gpu_choice(self.host) {
            fields.push(self.gpu_field(&c.gpu).show_if("graphics", LIBRARY));
        }
        fields.extend([
            Field::toggle(
                "audio",
                "Audio",
                c.audio.enabled,
                "A virtio sound device on the host's sound, with a microphone, and the HDA \
                 controller OpenCore's boot chime plays on.",
            ),
            Field::toggle(
                "logs",
                "Logs",
                c.logs.enabled,
                "The library's full log and the serial console, in the VM's logs/. Off, QEMU's \
                 log still keeps the library's warnings.",
            ),
            Field::text(
                "log_level",
                "Log level",
                &c.logs.level,
                "REIMS_VGPU_LOG's filter: error, warn, info, debug, trace, or target=level pairs.",
            )
            .show_if("logs", &["yes"]),
            Field::toggle(
                "network",
                "Network",
                c.network.enabled,
                "User-mode networking (NAT).",
            ),
            Field::text(
                "ssh_port",
                "SSH port",
                &c.network
                    .ssh_port
                    .map(|p| p.to_string())
                    .unwrap_or_default(),
                "A port on 127.0.0.1 forwarded to the guest's SSH; empty or 0 for none.",
            )
            .placeholder("none")
            .show_if("network", &["yes"]),
            Field::text(
                "stop_timeout",
                "Shutdown timeout",
                &c.stop_timeout_secs.to_string(),
                "Seconds stop (x) waits for the guest to shut down before it powers the VM off; \
                 0 for no limit: stop waits until the guest shuts down, and only power off \
                 (K) pulls the plug.",
            ),
        ]);
        if let Some(q) = &c.q35 {
            let options =
                DiskController::ALL.map(|d| (d.id().to_string(), d.describe().to_string()));
            fields.push(Field::choice(
                "disk_controller",
                "Disk controller",
                options.to_vec(),
                q.disk_controller.id(),
                "SATA, which macOS's installers see, or NVMe, for macOS once installed: an \
                 installer may find no disk on NVMe. macOS finds its disk on either; once it is \
                 installed, eject the installer before a move, or OpenCore may start it instead.",
            ));
            fields.push(
                Field::path(
                    "installer",
                    "Installer image",
                    &installer_shown(&vm),
                    "A raw image, or a DMG, converted to a raw image in the VM's directory (Tab completes). Empty to eject.",
                )
                .placeholder("none (ejected)"),
            );
        }
        if let Some(on) = opencore::shows_picker(&vm) {
            fields.push(Field::toggle(
                "boot_selection",
                "Boot selection",
                on,
                "OpenCore's boot picker at each start, to pick Recovery or another disk. Off, it \
                 boots macOS's startup disk at once. A change makes the VM a new OpenCore image \
                 presenting the same Mac, so the VM must be stopped.",
            ));
        }
        let extra = c
            .extra_args
            .iter()
            .map(|a| crate::qemu::quote(a))
            .collect::<Vec<_>>()
            .join(" ");
        let env = c
            .extra_env
            .iter()
            .map(|(k, v)| format!("{k}={}", crate::qemu::quote(v)))
            .collect::<Vec<_>>()
            .join(" ");
        fields.extend([
            Field::text(
                "extra_args",
                "Extra QEMU args",
                &extra,
                "More arguments for QEMU, shell-quoted.",
            )
            .placeholder("none"),
            Field::text(
                "extra_env",
                "Extra environment",
                &env,
                "KEY=VALUE pairs for QEMU's environment, shell-quoted; a KEY is letters, digits \
                 and _, not starting with a digit.",
            )
            .placeholder("none"),
        ]);
        let note = if runtime::running(&vm) {
            " (applies at its next start)"
        } else {
            ""
        };
        let form = Form::new(&format!("Edit {}{note}", c.name), "save", fields);
        self.overlay = Some(Overlay::Form(FormKind::Edit(c.name.clone()), form));
    }

    fn submit(&mut self, kind: &FormKind, f: &mut Form) -> Result<()> {
        match kind {
            FormKind::Create => {
                let spec = self.spec_from(f)?;
                let host = self.host;
                let name = spec.name.clone();
                self.spawn_for(f, &name, "creating", move |p| {
                    let vm = create::create(spec, host, p)?;
                    let next = if runtime::install_steps(&vm).is_some() {
                        "; its details say how to install macOS"
                    } else {
                        ""
                    };
                    Ok(format!(
                        "created {} in {} · s starts it{next}",
                        vm.name(),
                        paths::tilde(&vm.dir)
                    ))
                })?;
                // Select it once it appears.
                self.pending_select = Some(name);
                Ok(())
            }
            FormKind::Edit(name) => {
                let mut vm = Registry::load()?.open(name)?;
                let installer = installer_change(&vm, f)?;
                let picker = picker_change(&vm, f)?;
                let graphics = vm.config.graphics();
                apply_edit(&mut vm, f, self.gpus.as_deref())?;
                // A change of graphics makes the OpenCore image for them, where the VM is
                // stopped; while it runs, its next start makes one without reims-vgpu's kexts on
                // another device, and warns of an image that lacks them on reims-vgpu.
                let image = picker.is_some()
                    || (vm.config.graphics() != graphics
                        && !runtime::running(&vm)
                        && opencore::made_for_other_graphics(&vm));
                if (installer.is_some() || image)
                    && let Some(a) = self.activity.get(name)
                {
                    bail!("{name} is busy: {a}");
                }
                vm.save()?;
                let note = if runtime::running(&vm) {
                    " (at its next start)"
                } else {
                    ""
                };
                self.say(&format!("{name}: saved{note}"), Level::Ok);
                self.reload()?;
                // After the rest is saved: a DMG takes seconds to convert, and a new OpenCore image
                // about as long. The image first: the form, submitted again after a failure, finds
                // it made, where it would convert a DMG again.
                if installer.is_some() || image {
                    let first = if image {
                        "making a new OpenCore image"
                    } else {
                        "changing the installer"
                    };
                    self.spawn_for(f, name, first, move |p| {
                        let mut done = Vec::new();
                        if image {
                            done.push(opencore::renew(&mut vm, picker, p)?);
                        }
                        if let Some(path) = installer {
                            done.push(set_installer(&mut vm, path, p)?);
                        }
                        Ok(done.join("; "))
                    })?;
                }
                Ok(())
            }
        }
    }

    fn spec_from(&self, f: &Form) -> Result<Spec> {
        let mut name = f.text("name");
        if name.is_empty() && f.is_shown("macos") && f.choice("source") == "download" {
            // Its error first: a version that cannot be had names nothing.
            self.download_from(f)?;
            name = self.free_name(&f.choice("macos"));
        }
        crate::vm::check_name(&name)?;
        if self.rows.iter().any(|r| r.name == name) {
            bail!("a VM named {name} already exists");
        }
        let mut s = Spec::new(&name);
        let dir = f.text("dir");
        s.dir = (!dir.is_empty()).then(|| paths::absolute(&dir));
        s.cpus = Some(f.text("cpus").parse().context("CPU cores: a number")?);
        s.memory_mib = Some(parse_size_mib(&f.text("memory")).context("memory")?);
        if f.is_shown("gpu") {
            s.gpu = gpu_from_choice(&f.choice("gpu"), self.gpus.as_deref().unwrap_or(&[]));
        }
        let opt_path = |key: &str| {
            let t = f.text(key);
            (!t.is_empty()).then(|| paths::absolute(&t))
        };
        match self.host.default_profile() {
            Some(Profile::Q35X86_64) => {
                let size = || -> Result<DiskSource> {
                    Ok(DiskSource::New {
                        size_mib: parse_size_mib(&f.text("disk_size")).context("disk size")?,
                    })
                };
                match f.choice("source").as_str() {
                    "download" => {
                        s.macos = Some(self.download_from(f)?);
                        s.disk = Some(size()?);
                    }
                    "installer" => {
                        s.installer =
                            Some(opt_path("installer").context("give the installer image")?);
                        s.disk = Some(size()?);
                    }
                    _ => {
                        s.disk = Some(DiskSource::Existing {
                            path: opt_path("disk_path").context("give the disk image's path")?,
                            mode: ImportMode::parse(&f.choice("disk_mode"))?,
                        });
                    }
                }
                s.opencore = opt_path("opencore");
            }
            None => bail!(crate::host::UNSUPPORTED),
        }
        if f.is_shown("graphics") {
            s.graphics = Some(Graphics::parse(&f.choice("graphics"))?);
        }
        // Hidden on VMware SVGA, which opens at the firmware's resolution: the default.
        if f.is_shown("resolution") {
            s.display.resolution = f.text("resolution");
        }
        s.display.check()?;
        s.logs = f.toggle("logs");
        s.ssh_port = ssh_port_field(f)?;
        Ok(s)
    }

    /// `base`, or `base-2`, `base-3`… where a VM has the name, or its default directory holds
    /// files (a VM deleted with its files kept, or one whose create stopped).
    fn free_name(&self, base: &str) -> String {
        let taken = |n: &str| {
            self.rows.iter().any(|r| r.name == n)
                || self.activity.contains_key(n)
                || std::fs::read_dir(create::default_dir(n)).is_ok_and(|mut d| d.next().is_some())
        };
        (1..)
            .map(|i| {
                if i == 1 {
                    base.to_string()
                } else {
                    format!("{base}-{i}")
                }
            })
            .find(|n| !taken(n))
            .unwrap_or_default()
    }

    /// The download the form asks for: its version, and an offline installer's release.
    fn download_from(&self, f: &Form) -> Result<reims_util::Download> {
        let slug = f.choice("macos");
        let version = match &self.macos_versions {
            Some(Ok(list)) => list.iter().find(|v| v.slug == slug),
            Some(Err(e)) => bail!("{e}"),
            None => None,
        }
        .context("choose the macOS version")?;
        let offline = f.choice("installer_kind") == "offline";
        if !offline {
            return Ok(reims_util::Download {
                version: slug,
                offline,
            });
        }
        if !version.has_offline() {
            bail!(
                "{} has no offline installer (macOS 11 and later do): choose Online",
                version.name
            );
        }
        let release = f.text("release");
        if release.is_empty() {
            return Ok(reims_util::Download {
                version: slug,
                offline,
            });
        }
        if release.split('.').next() != Some(version.number()) {
            bail!(
                "Release: a release of {}, such as {}.1, or empty for the newest",
                version.name,
                version.number()
            );
        }
        Ok(reims_util::Download {
            version: release,
            offline,
        })
    }

    // ------------------------------------------------------------ drawing

    fn draw(&mut self, f: &mut Frame) {
        if let Some(name) = self.pending_select.clone()
            && let Some(i) = self.rows.iter().position(|r| r.name == name)
        {
            self.table.select(Some(i));
            self.pending_select = None;
        }
        let area = f.area();
        let width = area.width as usize;
        // A row each, unless the status or the hints are wider than the terminal.
        let status_rows = match &self.status {
            Some((s, _)) if s.chars().count() + 1 > width => 2,
            _ => 1,
        };
        let hints = key_hints();
        let hint_rows = if hints.width() > width { 2 } else { 1 };
        let [header, body, status, keys] = Layout::vertical([
            Constraint::Length(1),
            Constraint::Min(5),
            Constraint::Length(status_rows),
            Constraint::Length(hint_rows),
        ])
        .areas(area);
        self.draw_header(f, header);
        if area.width >= 100 {
            let [list, detail] =
                Layout::horizontal([Constraint::Percentage(45), Constraint::Percentage(55)])
                    .areas(body);
            self.draw_list(f, list);
            self.draw_detail(f, detail);
        } else {
            let [list, detail] =
                Layout::vertical([Constraint::Percentage(45), Constraint::Percentage(55)])
                    .areas(body);
            self.draw_list(f, list);
            self.draw_detail(f, detail);
        }
        self.draw_status(f, status);
        f.render_widget(Paragraph::new(hints).wrap(Wrap { trim: false }), keys);
        match &self.overlay {
            Some(Overlay::Form(_, form)) => form.render(f, area),
            Some(Overlay::Text {
                title,
                lines,
                scroll,
                ..
            }) => draw_text(f, area, title, lines, *scroll),
            Some(Overlay::Confirm {
                title,
                body,
                options,
                choice,
            }) => draw_confirm(f, area, title, body, options, *choice),
            Some(Overlay::Usb {
                vm,
                devices,
                choice,
            }) => {
                let rows = self.usb_rows(vm, devices);
                let running = self.rows.iter().any(|r| &r.name == vm && r.pid.is_some());
                draw_usb(f, area, vm, running, &rows, devices, *choice);
            }
            // A menu whose VM has left the list closes.
            Some(Overlay::Menu {
                vm,
                choice,
                at,
                all,
            }) => match self.menu_items(vm, *all) {
                items if items.is_empty() => self.overlay = None,
                items => self.menu_area = draw_menu(f, area, vm, &items, *choice, *at),
            },
            None => {}
        }
    }

    fn draw_header(&self, f: &mut Frame, area: Rect) {
        let profile = self
            .host
            .default_profile()
            .map(|p| p.describe().to_string())
            .unwrap_or_else(|| "no VMs on this host".into());
        let qemu = match &self.qemu {
            Some(q) => Span::styled(q.clone(), Style::new().fg(Color::Green)),
            None => Span::styled("QEMU not installed", Style::new().fg(Color::Red).bold()),
        };
        let line = Line::from(vec![
            Span::styled(
                " reims-ctl ",
                Style::new().fg(Color::Black).bg(Color::Cyan).bold(),
            ),
            Span::raw(format!("  {} · {profile} · ", self.host)),
            qemu,
        ]);
        f.render_widget(Paragraph::new(line), area);
    }

    fn draw_list(&mut self, f: &mut Frame, area: Rect) {
        // Its rows show inside the border, below the header row.
        let [_, list_rows] = Layout::vertical([Constraint::Length(1), Constraint::Min(0)])
            .areas(area.inner(Margin::new(1, 1)));
        self.list_rows = list_rows;
        let mut rows: Vec<Row> = self
            .rows
            .iter()
            .map(|r| {
                let (state, style) = match (self.activity.get(&r.name), r.pid, &r.vm) {
                    (Some(a), _, _) => (a.clone(), Style::new().fg(Color::Yellow)),
                    (None, _, Err(_)) => ("broken".into(), Style::new().fg(Color::Red)),
                    (None, Some(_), _) => ("running".into(), Style::new().fg(Color::Green)),
                    (None, None, _) => ("stopped".into(), Style::new().dim()),
                };
                let (cpus, mem) = match &r.vm {
                    Ok(vm) => (vm.config.cpus.to_string(), format_mib(vm.config.memory_mib)),
                    Err(_) => (String::new(), String::new()),
                };
                Row::new(vec![
                    Cell::from(r.name.clone()),
                    Cell::from(state).style(style),
                    Cell::from(cpus),
                    Cell::from(mem),
                ])
            })
            .collect();
        for (name, a) in &self.activity {
            if !self.rows.iter().any(|r| &r.name == name) {
                rows.push(Row::new(vec![
                    Cell::from(name.clone()),
                    Cell::from(a.clone()).style(Style::new().fg(Color::Yellow)),
                    Cell::from(""),
                    Cell::from(""),
                ]));
            }
        }
        let empty = rows.is_empty();
        let table = Table::new(
            rows,
            [
                Constraint::Min(12),
                Constraint::Min(12),
                Constraint::Length(5),
                Constraint::Length(7),
            ],
        )
        .header(Row::new(vec!["NAME", "STATE", "CPUS", "MEMORY"]).style(Style::new().bold().dim()))
        .block(Block::bordered().title(" VMs "))
        .row_highlight_style(
            Style::new()
                .bg(Color::Indexed(237))
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("› ");
        f.render_stateful_widget(table, area, &mut self.table);
        if empty {
            let inner = Rect {
                x: area.x + 2,
                y: area.y + 3,
                width: area.width.saturating_sub(4),
                height: 2,
            };
            f.render_widget(
                Paragraph::new("No VMs yet. Press n to create one.").dim(),
                inner,
            );
        }
    }

    fn draw_detail(&self, f: &mut Frame, area: Rect) {
        let block = Block::bordered().title(match self.selected() {
            Some(r) => format!(" {} ", r.name),
            None => " details ".into(),
        });
        let lines: Vec<Line> = match self.selected() {
            None => vec![Line::from("")],
            Some(r) => match &r.vm {
                Err(e) => vec![
                    Line::from(Span::styled(e.clone(), Style::new().fg(Color::Red))),
                    Line::from(format!("directory {}", paths::tilde(&r.dir))),
                ],
                Ok(vm) => {
                    let mut lines: Vec<Line> = describe(vm, self.host)
                        .into_iter()
                        .map(|(k, v)| {
                            Line::from(vec![
                                Span::styled(format!("{k:<13} "), Style::new().dim()),
                                Span::raw(v),
                            ])
                        })
                        .collect();
                    if r.pid.is_some() {
                        lines.push(Line::from(""));
                        lines.push(Line::from(Span::styled(
                            "v shows its screen here, in the terminal",
                            Style::new().dim(),
                        )));
                    }
                    if let Some(steps) = runtime::install_steps(vm) {
                        lines.push(Line::from(""));
                        lines.push(Line::from(steps));
                    }
                    // Below its settings, what in them its guest does not work with, then
                    // what starting it warns of on this host.
                    let incompatible = vm.config.incompatibilities();
                    for w in incompatible
                        .into_iter()
                        .chain(runtime::warnings(vm, self.host))
                        .chain(usb::problems(vm))
                    {
                        lines.push(Line::from(""));
                        lines.push(Line::from(Span::styled(w, Style::new().fg(Color::Yellow))));
                    }
                    lines
                }
            },
        };
        f.render_widget(
            Paragraph::new(lines)
                .block(block)
                .wrap(Wrap { trim: false }),
            area,
        );
    }

    fn draw_status(&self, f: &mut Frame, area: Rect) {
        let Some((s, level)) = &self.status else {
            return;
        };
        let style = match level {
            Level::Info => Style::new(),
            Level::Ok => Style::new().fg(Color::Green),
            Level::Warn => Style::new().fg(Color::Yellow),
            Level::Error => Style::new().fg(Color::Red),
        };
        f.render_widget(
            Paragraph::new(Span::styled(format!(" {s}"), style)).wrap(Wrap { trim: true }),
            area,
        );
    }
}

/// The main screen's key hints: the everyday keys, the VM's menu, and the help, which lists
/// the rest.
fn key_hints() -> Line<'static> {
    let keys = [
        ("n", "new"),
        ("Enter", "menu"),
        ("s", "start"),
        ("x", "stop"),
        ("u", "update"),
        ("?", "help"),
    ];
    // Non-breaking spaces inside each hint, so a narrow terminal wraps between hints only.
    let mut spans = Vec::new();
    for (k, what) in keys {
        if !spans.is_empty() {
            spans.push(Span::raw("  "));
        }
        spans.push(Span::styled(
            format!("\u{a0}{k}\u{a0}"),
            Style::new().fg(Color::Black).bg(Color::DarkGray),
        ));
        spans.push(Span::raw(format!("\u{a0}{what}")));
    }
    Line::from(spans)
}

/// A VM's menu: what can be done with it as it is now, each with its label. A VM whose
/// settings do not load can only be deleted. With `all`, every command, whatever the VM's
/// state: one that cannot be done now says why on the status line.
fn vm_menu(row: &VmRow, all: bool, host: Host) -> Vec<(VmCommand, &'static str)> {
    let vm = row.vm.as_ref().ok();
    if vm.is_none() && !all {
        return vec![(VmCommand::Delete, "Delete…")];
    }
    let running = row.pid.is_some();
    let x86_64 = vm.is_some_and(|vm| vm.config.q35.is_some());
    VmCommand::ALL
        .into_iter()
        .filter_map(|(c, _)| {
            let (label, now) = match c {
                VmCommand::Start => ("Start", !running),
                VmCommand::View => ("View screen in terminal", running),
                VmCommand::Stop => ("Stop", running),
                VmCommand::PowerOff => ("Power off…", running),
                VmCommand::Edit => ("Edit settings…", true),
                VmCommand::Usb => (
                    "USB devices…",
                    vm.is_some_and(|vm| usb_available(host, vm).is_ok()),
                ),
                VmCommand::ShowCommand => ("Show QEMU command", true),
                VmCommand::ToggleLogs => match vm.map(|vm| vm.config.logs.enabled) {
                    Some(true) => ("Turn logs off", true),
                    Some(false) => ("Turn logs on", true),
                    None => ("Turn logs on or off", true),
                },
                VmCommand::ViewLogs => ("View logs", true),
                VmCommand::Report => ("Write a bug report", true),
                VmCommand::UpdateOpencore => ("Make a new OpenCore image…", !running && x86_64),
                VmCommand::Delete => ("Delete…", !running),
            };
            (all || now).then_some((c, label))
        })
        .collect()
}

/// Whether the VM can be given the host's USB devices: an x86-64 VM on a Linux host. Else why
/// not, to say.
fn usb_available(host: Host, vm: &Vm) -> Result<(), String> {
    if host.default_profile().is_none() {
        return Err(crate::host::UNSUPPORTED.into());
    }
    if vm.config.q35.is_none() {
        return Err(format!(
            "{}: USB passthrough is for x86-64 VMs on Linux hosts",
            vm.name()
        ));
    }
    Ok(())
}

fn gpu_id(g: &crate::vm::Gpu) -> String {
    match (&g.driver_file, g.device_id) {
        (None, _) => "auto".into(),
        (Some(f), Some(id)) => format!("{}#{id}", f.display()),
        (Some(f), None) => f.display().to_string(),
    }
}

fn gpu_options(list: &[GpuInfo]) -> Vec<(String, String)> {
    let mut v = vec![("auto".to_string(), "Automatic (discrete first)".to_string())];
    for g in list {
        v.push((gpu_id(&g.to_config()), g.label()));
    }
    v
}

fn gpu_from_choice(id: &str, list: &[GpuInfo]) -> crate::vm::Gpu {
    if id == "auto" {
        return crate::vm::Gpu::default();
    }
    if let Some(g) = list.iter().find(|g| gpu_id(&g.to_config()) == id) {
        return g.to_config();
    }
    let (file, dev) = match id.split_once('#') {
        Some((f, d)) => (f, d.parse().ok()),
        None => (id, None),
    };
    crate::vm::Gpu {
        driver_file: Some(file.into()),
        device_id: dev,
        name: None,
    }
}

const RESOLUTION_HELP: &str = "The size the window opens at, WIDTHxHEIGHT in the desktop's \
     points, shrunk to fit the screen; the guest takes the window's size in pixels, and follows \
     it as you resize it. It starts a HiDPI desktop by itself in a window at least 3840 pixels \
     wide.";

/// The graphics field's values that show a field of the library's.
const LIBRARY: &[&str] = &["reims-vgpu"];

/// The graphics choice of a VM, at `current`.
fn graphics_field(current: Graphics) -> Field {
    let options = Graphics::ALL.map(|g| (g.id().to_string(), g.describe().to_string()));
    Field::choice(
        "graphics",
        "Graphics",
        options.to_vec(),
        current.id(),
        "reims-vgpu is its own GPU device, which macOS drives with the drivers reims-ctl puts in \
         the VM's OpenCore image (after a change to it, U makes the image again): one display, \
         in a window whose size the guest follows. VMware SVGA is QEMU's: one \
         display at the firmware's resolution, with no GPU acceleration (macOS has no driver \
         for it), in QEMU's own window.",
    )
}

/// The VM's installer as the edit form shows it: its full path, or empty.
fn installer_shown(vm: &Vm) -> String {
    vm.config
        .q35
        .as_ref()
        .and_then(|q| q.installer.as_ref())
        .map(|p| vm.path(p).to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// The installer the edit form changes the VM's to, if it changes it; `Some(None)` ejects it.
fn installer_change(vm: &Vm, f: &Form) -> Result<Option<Option<PathBuf>>> {
    if !f.is_shown("installer") {
        return Ok(None);
    }
    let t = f.text("installer");
    if t == installer_shown(vm) {
        return Ok(None);
    }
    let path = (!t.is_empty()).then(|| paths::absolute(&t));
    if let Some(p) = &path
        && !p.is_file()
    {
        bail!("installer image: {} does not exist", p.display());
    }
    Ok(Some(path))
}

/// The boot selection the edit form changes the VM's OpenCore image to, if it changes it: the
/// image is made again, which the VM must be stopped for.
fn picker_change(vm: &Vm, f: &Form) -> Result<Option<bool>> {
    let on = f.toggle("boot_selection");
    if !f.is_shown("boot_selection") || opencore::shows_picker(vm) == Some(on) {
        return Ok(None);
    }
    if runtime::running(vm) {
        bail!(
            "{} is running: stop it to change its boot selection, which makes it a new OpenCore \
             image",
            vm.name()
        );
    }
    Ok(Some(on))
}

fn apply_edit(vm: &mut Vm, f: &Form, gpus: Option<&[GpuInfo]>) -> Result<()> {
    let c = &mut vm.config;
    c.cpus = f.text("cpus").parse().context("CPU cores: a number")?;
    c.memory_mib = parse_size_mib(&f.text("memory")).context("memory")?;
    if f.is_shown("graphics") {
        c.graphics = Some(Graphics::parse(&f.choice("graphics"))?);
    }
    if f.is_shown("disk_controller")
        && let Some(q) = &mut c.q35
    {
        q.disk_controller = DiskController::parse(&f.choice("disk_controller"))?;
    }
    // Hidden on VMware SVGA, the resolution, the window's first size, and borderless stay as
    // they were.
    if f.is_shown("resolution") {
        c.display.resolution = f.text("resolution");
        c.display.check()?;
    }
    if f.is_shown("borderless") {
        c.display.borderless = f.toggle("borderless");
    }
    if f.is_shown("gpu") {
        let id = f.choice("gpu");
        if id != gpu_id(&c.gpu) {
            c.gpu = gpu_from_choice(&id, gpus.unwrap_or(&[]));
        }
    }
    c.audio.enabled = f.toggle("audio");
    c.logs.enabled = f.toggle("logs");
    if f.is_shown("log_level") && !f.text("log_level").is_empty() {
        c.logs.level = f.text("log_level");
    }
    c.network.enabled = f.toggle("network");
    if f.is_shown("ssh_port") {
        c.network.ssh_port = ssh_port_field(f)?;
    }
    c.stop_timeout_secs = f
        .text("stop_timeout")
        .parse()
        .context("shutdown timeout: seconds")?;
    c.extra_args =
        shlex::split(&f.text("extra_args")).context("extra QEMU args: unbalanced quotes")?;
    c.extra_env.clear();
    for kv in shlex::split(&f.text("extra_env")).context("extra environment: unbalanced quotes")? {
        let (k, v) = kv
            .split_once('=')
            .with_context(|| format!("extra environment takes KEY=VALUE, not {kv:?}"))?;
        c.extra_env.insert(k.into(), v.into());
    }
    c.check()
}

/// The SSH port field's forward: none where it is empty or 0, as `set --ssh-port 0` has it.
fn ssh_port_field(f: &Form) -> Result<Option<u16>> {
    let p = f.text("ssh_port");
    if p.is_empty() {
        return Ok(None);
    }
    let p = p.parse().context("SSH port: a number")?;
    Ok(crate::vm::ssh_port(p))
}

/// Runs less on `files`, each at its end, as system(3) runs a command: Ctrl+C and Ctrl+\ reach
/// the terminal's whole foreground process group, so reims-ctl ignores them until less exits,
/// and less gets them as it would from a shell.
fn run_less(files: &[PathBuf]) -> std::io::Result<std::process::ExitStatus> {
    use std::os::unix::process::CommandExt;
    let mut cmd = std::process::Command::new("less");
    cmd.arg("++G").args(files);
    // SAFETY: only signal, which is async-signal-safe, between fork and exec.
    unsafe {
        cmd.pre_exec(|| {
            libc::signal(libc::SIGINT, libc::SIG_DFL);
            libc::signal(libc::SIGQUIT, libc::SIG_DFL);
            Ok(())
        });
    }
    // SAFETY: dispositions set to ignore, then back to what they were.
    let (int, quit) = unsafe {
        (
            libc::signal(libc::SIGINT, libc::SIG_IGN),
            libc::signal(libc::SIGQUIT, libc::SIG_IGN),
        )
    };
    let status = cmd.status();
    unsafe {
        libc::signal(libc::SIGINT, int);
        libc::signal(libc::SIGQUIT, quit);
    }
    status
}

/// Has the terminal report the mouse's buttons and wheel, or stop: X11's reports (mode 1000)
/// in SGR's form (1006), which holds any column. Not the mouse moving, which nothing here
/// reads.
fn report_mouse(on: bool) -> std::io::Result<()> {
    use std::io::Write;
    let mut out = std::io::stdout();
    out.write_all(if on {
        b"\x1b[?1000h\x1b[?1006h"
    } else {
        b"\x1b[?1006l\x1b[?1000l"
    })?;
    out.flush()
}

/// Has a terminal that speaks kitty's keyboard protocol report keys its way, or stop: pushes
/// its flag 1 (disambiguate escape codes), which only tells keys apart, as Ctrl-Enter from
/// Enter; or pops it. Plain keys still come as they did, and text as text.
fn report_keys(on: bool) -> std::io::Result<()> {
    use std::io::Write;
    let mut out = std::io::stdout();
    out.write_all(if on { b"\x1b[>1u" } else { b"\x1b[<1u" })?;
    out.flush()
}

fn copy_to_clipboard(s: &str) -> Option<&'static str> {
    use std::io::Write;
    use std::process::{Command, Stdio};
    let tools: &[(&'static str, &[&str])] = &[
        ("pbcopy", &[]),
        ("wl-copy", &[]),
        ("xclip", &["-selection", "clipboard"]),
        ("xsel", &["-b", "-i"]),
    ];
    for (tool, args) in tools {
        let Ok(mut child) = Command::new(tool)
            .args(*args)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
        else {
            continue;
        };
        if let Some(mut stdin) = child.stdin.take() {
            let _ = stdin.write_all(s.as_bytes());
        }
        if child.wait().map(|st| st.success()).unwrap_or(false) {
            return Some(tool);
        }
    }
    None
}

const HELP: &str = "\
reims-ctl runs x86-64 macOS VMs on QEMU with reims-vgpu: each VM's display opens in a window of
its own. A VM runs reims-vgpu's own GPU device unless its settings choose another (e, graphics).

  ↑ ↓ / j k   choose a VM, or click it
  Enter       open its menu: what can be done with it now, each with its key. A right
              click on a VM opens its menu too. Hold Ctrl as you press Enter or right click
              for a menu of everything, whether it can be done now or not (Ctrl-Enter needs
              a terminal that tells it from Enter, such as kitty, foot or Ghostty on a Mac)
  n           create a VM. Choose a macOS version: reims-ctl downloads its
              installer from Apple and makes the VM's OpenCore image and disk, and the VM
              boots into the installer; its details then say how to install macOS. The form
              stays open until the VM is made, and shows what failed, to fix and create again;
              Esc closes it meanwhile, and it opens again if the create fails
  s           start it
  x           stop it: the guest is asked to shut down, and powered off if it has not
              when its shutdown timeout ends (edit e to change it)
  K           kill it: power it off at once
  e           edit its settings: CPUs, memory, resolution, borderless, GPU, audio,
              network, logs, extra QEMU arguments (they apply at its next start), its disk
              controller (SATA, or NVMe once macOS is installed), its installer image, its
              boot selection (OpenCore's picker, which makes it a new OpenCore image) and
              graphics: reims-vgpu, its own GPU device, on one display in a window whose
              size the guest follows, with the macOS drivers reims-ctl puts in its OpenCore
              image (made again on a change); or QEMU's VMware SVGA, one display with no GPU
              acceleration, in QEMU's own window. Resolution, borderless and GPU are
              reims-vgpu's
  p           give it this host's USB devices: Space gives the selected one to
              the guest or takes it back, at once while the VM runs; a device you gave it
              joins the guest when it is plugged in. More in reims-ctl usb --help
  c           show the QEMU command it runs, to write as launch.sh (w) or copy (y):
              run it yourself, without reims-ctl
  v           show a running VM's screen here, in the terminal, and use its keyboard and
              mouse from here: for a host with no screen, over SSH too. Ctrl-] comes back
  l           turn its logs on or off
  L           view its QEMU log in less: :n shows the serial console, / searches,
              F follows the log as it grows, q comes back
  b           write a bug report: its logs, settings, command and this host's details,
              zipped into your Downloads folder
  d           delete it
  U           make it a new OpenCore image with the installed reims-util, presenting the
              same Mac, for its graphics: on reims-vgpu with the macOS drivers of its GPU
              device that this reims-ctl carries, which an image an older reims-ctl made
              lacks
  u           update reims-ctl and QEMU from your reims-vgpu checkout
  r           reload the list
  ?           this help
  q           quit; running VMs keep running

The main screen takes the mouse: to select text on it, hold Shift as you drag (Option in
iTerm2; in Terminal, ⌘R turns its mouse reporting off and on).

The same from a shell: reims-ctl help.";

fn help() -> Overlay {
    Overlay::Text {
        title: "help  (Esc close)".into(),
        lines: HELP.lines().map(String::from).collect(),
        scroll: 0,
        kind: TextKind::Help,
    }
}

/// The columns and rows of a text view's text: the whole screen below its title row, with no
/// border or margin beside the text, so the terminal's selection copies the text alone.
fn text_inner(area: Rect) -> (usize, usize) {
    (
        area.width.max(1) as usize,
        area.height.saturating_sub(1) as usize,
    )
}

/// `lines` cut into rows of at most `width` characters, so a long line (a command's drive
/// paths, a log's message) shows whole rather than past the screen's edge.
fn wrap_rows(lines: &[String], width: usize) -> Vec<&str> {
    let mut rows = Vec::new();
    for l in lines {
        let mut rest = l.as_str();
        loop {
            match rest.char_indices().nth(width) {
                Some((i, _)) => {
                    rows.push(&rest[..i]);
                    rest = &rest[i..];
                }
                None => {
                    rows.push(rest);
                    break;
                }
            }
        }
    }
    rows
}

fn draw_text(f: &mut Frame, area: Rect, title: &str, lines: &[String], scroll: usize) {
    f.render_widget(Clear, area);
    let [top, body] = Layout::vertical([Constraint::Length(1), Constraint::Min(0)]).areas(area);
    f.render_widget(
        Paragraph::new(format!(" {title}")).style(Style::new().fg(Color::Black).bg(Color::Cyan)),
        top,
    );
    let (width, height) = text_inner(area);
    let rows = wrap_rows(lines, width);
    let start = scroll.min(rows.len().saturating_sub(height));
    let shown: Vec<Line> = rows
        .into_iter()
        .skip(start)
        .take(height)
        .map(Line::from)
        .collect();
    f.render_widget(Paragraph::new(shown), body);
}

fn draw_confirm(
    f: &mut Frame,
    area: Rect,
    title: &str,
    body: &str,
    options: &[(&str, Action)],
    choice: usize,
) {
    let width = area.width.saturating_sub(4).min(76);
    let body_lines = body
        .lines()
        .map(|l| {
            (l.chars().count() as u16)
                .div_ceil(width.saturating_sub(4).max(10))
                .max(1)
        })
        .sum::<u16>();
    let popup = area.centered(
        Constraint::Length(width),
        Constraint::Length((body_lines + 5).min(area.height)),
    );
    f.render_widget(Clear, popup);
    let block = Block::bordered()
        .title(Line::from(format!(" {title} ")).bold())
        .border_style(Style::new().fg(Color::Yellow));
    let inner = block.inner(popup);
    f.render_widget(block, popup);
    let [b, o] = Layout::vertical([Constraint::Min(1), Constraint::Length(1)]).areas(inner);
    f.render_widget(Paragraph::new(body).wrap(Wrap { trim: false }), b);
    let mut spans = Vec::new();
    for (i, (label, _)) in options.iter().enumerate() {
        let style = if i == choice {
            Style::new().fg(Color::Black).bg(Color::Yellow).bold()
        } else {
            Style::new()
        };
        spans.push(Span::styled(format!(" {label} "), style));
        spans.push(Span::raw("  "));
    }
    f.render_widget(Paragraph::new(Line::from(spans)), o);
}

/// Draws the USB devices' view of the VM `vm`: each of `rows` with a box checked where the VM
/// has the device, its name, IDs and notes, `choice` highlighted, and what lets QEMU open the
/// devices this user can't.
fn draw_usb(
    f: &mut Frame,
    area: Rect,
    vm: &str,
    running: bool,
    rows: &[usb::Row],
    devices: &[usb::HostDevice],
    choice: usize,
) {
    let width = area.width.saturating_sub(4).min(100);
    let inner_w = width.saturating_sub(4) as usize;
    let mut lines: Vec<Line> = Vec::new();
    let when = if running {
        "at once, while it runs"
    } else {
        "from its next start"
    };
    lines.push(
        Line::from(format!(
            "This host's USB devices. One you give {vm} is the guest's, {when}, until the VM \
             stops or you take it back; one not plugged in joins the guest when it is."
        ))
        .dim(),
    );
    lines.push(Line::from(""));
    let first_row = lines.len();
    if rows.is_empty() {
        lines.push(Line::from(
            "No USB devices are plugged into this host, other than hubs.",
        ));
    }
    let name_w = rows
        .iter()
        .map(|r| r.name().chars().count())
        .max()
        .unwrap_or(0)
        .min(inner_w.saturating_sub(24).clamp(12, 44));
    let mut closed = false;
    for (i, r) in rows.iter().enumerate() {
        let mut name: String = r.name().chars().take(name_w).collect();
        if r.name().chars().count() > name_w {
            name.pop();
            name.push('…');
        }
        let notes = r.notes(devices);
        closed |= r.device.as_ref().is_some_and(|d| !d.usable());
        let text = format!(
            "{} [{}] {name:<name_w$}  {}{}",
            if i == choice { "›" } else { " " },
            if r.chosen.is_some() { "x" } else { " " },
            r.id(),
            if notes.is_empty() {
                String::new()
            } else {
                format!("  {}", notes.join(", "))
            }
        );
        lines.push(if i == choice {
            Line::from(text).style(Style::new().fg(Color::Black).bg(Color::Cyan))
        } else if r.device.is_none() {
            Line::from(text).dim()
        } else {
            Line::from(text)
        });
    }
    lines.push(Line::from(""));
    if closed {
        lines.push(
            Line::from(format!("QEMU can't open what you can't: {}.", usb::fix()))
                .style(Style::new().fg(Color::Yellow)),
        );
    }
    lines.push(
        Line::from(format!(
            "Space give or take back · r rescan · Esc close · up to {} devices a VM",
            crate::vm::USB_DEVICES_MAX
        ))
        .dim(),
    );
    // Rows wrap to the popup's width; the highlighted one stays in sight.
    let rows_of = |l: &Line| (l.width().max(1)).div_ceil(inner_w.max(1)) as u16;
    let need: u16 = lines.iter().map(rows_of).sum::<u16>() + 2;
    let height = need.min(area.height);
    let popup = area.centered(Constraint::Length(width), Constraint::Length(height));
    f.render_widget(Clear, popup);
    let block = Block::bordered()
        .title(Line::from(format!(" {vm}: USB devices ")).bold())
        .border_style(Style::new().fg(Color::Cyan));
    let body = block.inner(popup);
    let above: u16 = lines[..first_row + choice.min(rows.len().saturating_sub(1))]
        .iter()
        .map(rows_of)
        .sum();
    let scroll = (above + 1).saturating_sub(body.height);
    f.render_widget(
        Paragraph::new(lines)
            .block(block)
            .wrap(Wrap { trim: false })
            .scroll((scroll, 0)),
        popup,
    );
}

/// Draws a VM's menu below the cell `at`, or above it where the screen has no room below,
/// and as far left or up as the screen needs; returns where. Each item shows its key at its
/// right.
fn draw_menu(
    f: &mut Frame,
    area: Rect,
    vm: &str,
    items: &[(VmCommand, &str)],
    choice: VmCommand,
    at: Position,
) -> Rect {
    let label_w = items
        .iter()
        .map(|it| it.1.chars().count())
        .max()
        .unwrap_or(0);
    // A space, the label, three spaces, the key and a space; or the title, in its border.
    let inner_w = (label_w + 6).max(vm.chars().count() + 4);
    let width = (inner_w as u16 + 2).min(area.width);
    let height = (items.len() as u16 + 2).min(area.height);
    let y = if at.y + 1 + height <= area.bottom() {
        at.y + 1
    } else if at.y >= area.y + height {
        at.y - height
    } else {
        area.bottom() - height
    };
    let popup = Rect {
        x: at.x.min(area.right() - width),
        y: y.min(area.bottom() - height),
        width,
        height,
    };
    f.render_widget(Clear, popup);
    let block = Block::bordered()
        .title(Line::from(format!(" {vm} ")).bold())
        .border_style(Style::new().fg(Color::Cyan));
    let chosen = menu_index(items, choice);
    let key_w = inner_w - 2 - label_w;
    let lines: Vec<Line> = items
        .iter()
        .enumerate()
        .map(|(i, &(c, label))| {
            let label = format!(" {label:<label_w$}");
            let key = format!("{:>key_w$} ", c.key());
            if i == chosen {
                Line::from(label + &key).style(Style::new().fg(Color::Black).bg(Color::Cyan))
            } else {
                Line::from(vec![
                    Span::raw(label),
                    Span::styled(key, Style::new().dim()),
                ])
            }
        })
        .collect();
    f.render_widget(Paragraph::new(lines).block(block), popup);
    popup
}

/// Where `choice` is in a menu's `items`: at its first item where it is no longer one, as when
/// the VM has started or stopped since.
fn menu_index(items: &[(VmCommand, &str)], choice: VmCommand) -> usize {
    items.iter().position(|it| it.0 == choice).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::Arch;

    fn press(app: &mut App, codes: impl IntoIterator<Item = KeyCode>) {
        for code in codes {
            app.key(KeyEvent::new(code, KeyModifiers::NONE));
        }
    }

    fn form(app: &App) -> &Form {
        match &app.overlay {
            Some(Overlay::Form(_, f)) => f,
            _ => panic!("no form is open"),
        }
    }

    #[test]
    fn a_create_that_fails_keeps_its_form_open_with_the_error() {
        // The create and the list it reads again use these VMs, not the user's.
        let _env = crate::testenv::Env::new("tui-create");
        let host = Host {
            os: Os::Linux,
            arch: Arch::X86_64,
        };
        let mut app = App::with(host);
        app.macos_versions = Some(Ok(Vec::new()));
        app.open_create();
        let name = format!("form-test-{}", std::process::id());
        let disk = std::env::temp_dir().join(format!("{name}/disk.qcow2"));
        let disk = disk.to_string_lossy().into_owned();
        press(&mut app, name.chars().map(KeyCode::Char));
        press(
            &mut app,
            [KeyCode::Down, KeyCode::Right, KeyCode::Right, KeyCode::Down],
        );
        press(&mut app, disk.chars().map(KeyCode::Char));
        app.key(KeyEvent::new(KeyCode::Char('s'), KeyModifiers::CONTROL));
        assert_eq!(form(&app).busy.as_ref().unwrap().1, "creating");
        // The create fails on the disk, before any slow step.
        let m = app.rx.recv_timeout(Duration::from_secs(30)).unwrap();
        app.message(m);
        let f = form(&app);
        assert_eq!(f.busy, None);
        assert_eq!(f.error, Some(format!("the disk {disk} does not exist")));
        assert_eq!((f.text("name"), f.text("disk_path")), (name, disk));
    }

    #[test]
    fn a_busy_form_closes_once_its_work_succeeds_and_opens_again_if_it_fails() {
        // The list each work's end reads again is of these VMs, not the user's.
        let _env = crate::testenv::Env::new("tui-busy");
        let mut app = App::with(Host::current());
        let busy = |app: &mut App, id| {
            let mut f = Form::new("t", "save", Vec::new());
            f.busy = Some((id, "working".into()));
            app.overlay = Some(Overlay::Form(FormKind::Create, f));
        };
        let done = |app: &mut App, id, r: Result<&str, &str>| {
            let r = r.map(String::from).map_err(String::from);
            app.message(Msg::Done("vm".into(), id, r));
        };
        let shown = |app: &App| match &app.overlay {
            Some(Overlay::Form(_, f)) => Some((f.busy.as_ref().map(|b| b.0), f.error.clone())),
            _ => None,
        };
        busy(&mut app, 1);
        done(&mut app, 1, Ok("done"));
        assert_eq!(shown(&app), None);
        // Another work's end leaves it be; its own failure shows in it.
        busy(&mut app, 2);
        done(&mut app, 3, Err("other"));
        assert_eq!(shown(&app), Some((Some(2), None)));
        done(&mut app, 2, Err("boom"));
        assert_eq!(shown(&app), Some((None, Some("boom".into()))));
        // Closed while its work runs, it opens again if that fails, once no other view is open.
        busy(&mut app, 4);
        press(&mut app, [KeyCode::Esc, KeyCode::Char('?')]);
        done(&mut app, 4, Err("bang"));
        assert!(matches!(app.overlay, Some(Overlay::Text { .. })));
        press(&mut app, [KeyCode::Esc]);
        assert_eq!(shown(&app), Some((None, Some("bang".into()))));
    }

    #[test]
    fn wrap_rows_cuts_at_the_width() {
        let lines = vec!["abcdef".to_string(), String::new(), "ééé".to_string()];
        assert_eq!(wrap_rows(&lines, 3), ["abc", "def", "", "ééé"]);
        assert_eq!(wrap_rows(&lines, 2), ["ab", "cd", "ef", "", "éé", "é"]);
    }

    #[test]
    fn key_hints_fit_an_80_column_terminal() {
        assert!(key_hints().width() <= 80);
    }

    /// A stopped VM's row, whose files do not exist.
    fn vm_row(name: &str) -> VmRow {
        let config = format!(
            r#"
            name = "{name}"
            profile = "q35-x86_64"
            cpus = 2
            memory_mib = 4096
            disk = "disk.qcow2"
            [q35]
            opencore = "opencore.qcow2"
            firmware_code = "OVMF_CODE.fd"
            firmware_vars = "OVMF_VARS.fd"
            "#
        );
        let dir = std::env::temp_dir().join(format!("reims-ctl-tui-{}-{name}", std::process::id()));
        let vm = Vm {
            dir: dir.clone(),
            config: toml::from_str(&config).unwrap(),
        };
        VmRow {
            name: name.into(),
            dir,
            vm: Ok(vm),
            pid: None,
        }
    }

    fn test_app(rows: Vec<VmRow>) -> App {
        let mut app = App::with(Host::current());
        app.rows = rows;
        app.table.select(Some(0));
        app.gpus = Some(Vec::new());
        app
    }

    /// Where `s` first shows inside `area`, row by row.
    fn find(buf: &ratatui::buffer::Buffer, area: Rect, s: &str) -> Option<Position> {
        area.positions().find(|p| {
            s.chars().enumerate().all(|(i, c)| {
                let x = p.x + i as u16;
                x < area.right() && buf[(x, p.y)].symbol() == c.to_string()
            })
        })
    }

    fn mouse(kind: MouseEventKind, at: Position) -> MouseEvent {
        MouseEvent {
            kind,
            column: at.x,
            row: at.y,
            modifiers: KeyModifiers::NONE,
        }
    }

    #[test]
    fn each_vm_command_has_a_key_of_its_own() {
        for (c, k) in VmCommand::ALL {
            assert_eq!((c.key(), VmCommand::of_key(k)), (k, Some(c)));
            // The main screen's other keys.
            assert!(!"nqjkh?ur".contains(k), "{c:?}");
        }
    }

    const LINUX: Host = Host {
        os: Os::Linux,
        arch: Arch::X86_64,
    };

    #[test]
    fn a_vms_menu_holds_what_can_be_done_with_it_now() {
        use VmCommand::*;
        let commands = |r: &VmRow| {
            vm_menu(r, false, LINUX)
                .iter()
                .map(|it| it.0)
                .collect::<Vec<_>>()
        };
        let every = [ShowCommand, ToggleLogs, ViewLogs, Report];
        let stopped = [&[Start, Edit, Usb][..], &every, &[UpdateOpencore, Delete]].concat();
        assert_eq!(commands(&vm_row("t")), stopped);
        let mut running = vm_row("t");
        running.pid = Some(1);
        let running_menu = [&[View, Stop, PowerOff, Edit, Usb][..], &every].concat();
        assert_eq!(commands(&running), running_menu);
        // A Mac passes no USB device through.
        let mac = Host {
            os: Os::Mac,
            arch: Arch::Aarch64,
        };
        assert!(
            !vm_menu(&vm_row("t"), false, mac)
                .iter()
                .any(|it| it.0 == Usb)
        );
        let broken = VmRow {
            vm: Err("parse vm.toml".into()),
            ..vm_row("t")
        };
        assert_eq!(commands(&broken), [Delete]);

        let mut row = vm_row("t");
        assert!(vm_menu(&row, false, LINUX).contains(&(ToggleLogs, "Turn logs on")));
        row.vm.as_mut().unwrap().config.logs.enabled = true;
        assert!(vm_menu(&row, false, LINUX).contains(&(ToggleLogs, "Turn logs off")));

        // The menu of everything holds every command, whatever the VM's state.
        let each = VmCommand::ALL.map(|c| c.0);
        for row in [&vm_row("t"), &running, &broken] {
            let all = vm_menu(row, true, LINUX);
            assert_eq!(all.iter().map(|it| it.0).collect::<Vec<_>>(), each);
            // Each item is labelled as in the menu of what can be done now.
            for it in vm_menu(row, false, LINUX) {
                assert!(all.contains(&it), "{it:?}");
            }
        }
        assert!(vm_menu(&broken, true, LINUX).contains(&(ToggleLogs, "Turn logs on or off")));
    }

    /// p opens a VM's USB devices: a device it was given that is not plugged in shows checked,
    /// after the host's devices, and Space takes it back, in its settings as they are now. A
    /// Mac host says why it has none.
    #[test]
    fn the_usb_view_takes_back_a_device_not_plugged_in() {
        let env = crate::testenv::Env::new("tui-usb");
        let dir = env.root.join("vms/t");
        std::fs::create_dir_all(&dir).unwrap();
        let mut vm = vm_row("t").vm.unwrap();
        vm.dir = dir.clone();
        vm.config.usb = vec![crate::vm::UsbDevice {
            id: "fffe:0001".into(),
            port: None,
            name: "Gadget Away".into(),
        }];
        vm.save().unwrap();
        let mut reg = Registry::default();
        reg.add("t", &dir).unwrap();
        reg.save().unwrap();

        let mut app = App::with(Host {
            os: Os::Mac,
            arch: Arch::Aarch64,
        });
        app.reload().unwrap();
        press(&mut app, [KeyCode::Char('p')]);
        assert!(app.overlay.is_none());
        assert_eq!(app.status.as_ref().unwrap().0, crate::host::UNSUPPORTED);

        let mut app = App::with(LINUX);
        app.reload().unwrap();
        press(&mut app, [KeyCode::Char('p')]);
        let Some(Overlay::Usb { devices, .. }) = &app.overlay else {
            panic!("no USB view");
        };
        let devices = devices.clone();
        let rows = app.usb_rows("t", &devices);
        let away = rows.len() - 1;
        assert_eq!(rows[away].name(), "Gadget Away");
        assert!(rows[away].chosen.is_some() && rows[away].device.is_none());
        let mut terminal =
            ratatui::Terminal::new(ratatui::backend::TestBackend::new(120, 30)).unwrap();
        terminal.draw(|f| app.draw(f)).unwrap();
        let buf = terminal.backend().buffer().clone();
        let at = find(&buf, buf.area, "[x] Gadget Away").expect("the device's row");
        let row: String = (at.x..buf.area.right())
            .map(|x| buf[(x, at.y)].symbol().to_string())
            .collect();
        assert!(row.contains("fffe:0001  not plugged in"), "{row}");

        // Edited meanwhile, as by reims-ctl set: the change keeps it.
        let mut now = Vm::load(&dir).unwrap();
        now.config.cpus = 3;
        now.save().unwrap();
        press(&mut app, [KeyCode::End, KeyCode::Char(' ')]);
        let saved = Vm::load(&dir).unwrap().config;
        assert!(saved.usb.is_empty(), "{:?}", saved.usb);
        assert_eq!(saved.cpus, 3);
        assert!(
            app.status
                .as_ref()
                .unwrap()
                .0
                .contains("Gadget Away is no longer chosen"),
            "{:?}",
            app.status.as_ref().map(|s| &s.0)
        );
        assert!(matches!(app.overlay, Some(Overlay::Usb { .. })));
        press(&mut app, [KeyCode::Esc]);
        assert!(app.overlay.is_none());
    }

    /// A VM's edit form offers its disk's controller as its settings have it, and saves the one
    /// picked.
    #[test]
    fn the_edit_form_picks_a_vms_disk_controller() {
        let mut nvme = vm_row("n");
        let q = nvme.vm.as_mut().unwrap().config.q35.as_mut().unwrap();
        q.disk_controller = DiskController::Nvme;
        let mut app = test_app(vec![vm_row("t"), nvme]);
        app.table.select(Some(1));
        app.open_edit();
        assert_eq!(form(&app).choice("disk_controller"), "nvme");
        app.overlay = None;
        app.table.select(Some(0));
        app.open_edit();
        assert_eq!(form(&app).choice("disk_controller"), "ahci");
        for _ in 0..form(&app).fields.len() {
            if form(&app).focused_key() == "disk_controller" {
                break;
            }
            press(&mut app, [KeyCode::Down]);
        }
        press(&mut app, [KeyCode::Right]);
        let mut vm = vm_row("t").vm.unwrap();
        apply_edit(&mut vm, form(&app), None).unwrap();
        assert_eq!(vm.config.q35.unwrap().disk_controller, DiskController::Nvme);
    }

    /// A VM's edit form shows the resolution and borderless on reims-vgpu, and hides them on
    /// VMware SVGA, whose settings keep them as they were.
    #[test]
    fn the_edit_form_shows_reims_vgpu_its_display_settings() {
        let mut row = vm_row("t");
        let c = &mut row.vm.as_mut().unwrap().config;
        c.graphics = None;
        c.display.resolution = "3840x2160".into();
        let mut app = test_app(vec![row]);
        app.open_edit();
        assert_eq!(form(&app).choice("graphics"), "reims-vgpu");
        assert!(form(&app).is_shown("borderless") && form(&app).is_shown("resolution"));
        let mut vm = app.rows[0].vm.as_ref().unwrap().clone();
        apply_edit(&mut vm, form(&app), None).unwrap();
        assert_eq!(vm.config.graphics, Some(Graphics::ReimsVgpu));
        for _ in 0..form(&app).fields.len() {
            if form(&app).focused_key() == "graphics" {
                break;
            }
            press(&mut app, [KeyCode::Down]);
        }
        press(&mut app, [KeyCode::Right]);
        assert_eq!(form(&app).choice("graphics"), "vmware-svga");
        assert!(!form(&app).is_shown("borderless") && !form(&app).is_shown("resolution"));
        vm.config.display.resolution = "1280x720".into();
        apply_edit(&mut vm, form(&app), None).unwrap();
        assert_eq!(vm.config.graphics, Some(Graphics::VmwareSvga));
        assert_eq!(
            vm.config.display.resolution, "1280x720",
            "a hidden field changes nothing"
        );
    }

    #[test]
    fn ctrl_with_enter_or_a_right_click_opens_the_menu_of_everything() {
        let mut app = test_app(vec![vm_row("alpha"), vm_row("beta")]);
        app.rows[0].pid = Some(1);
        let mut terminal =
            ratatui::Terminal::new(ratatui::backend::TestBackend::new(120, 30)).unwrap();
        let mut draw = |app: &mut App| {
            terminal.draw(|f| app.draw(f)).unwrap();
            terminal.backend().buffer().clone()
        };
        let screen = draw(&mut app);
        let beta = find(&screen, app.list_rows, "beta").unwrap();

        // Ctrl-Enter: a running VM's menu holds Start too, shown chosen, and Delete, whose key
        // says why it cannot be done now and closes the menu.
        app.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::CONTROL));
        assert!(matches!(&app.overlay, Some(Overlay::Menu { vm, all: true, .. }) if vm == "alpha"));
        let screen = draw(&mut app);
        let start = find(&screen, app.menu_area, "Start").unwrap();
        assert_eq!(screen[start].bg, Color::Cyan);
        app.key(KeyEvent::from(KeyCode::Char('d')));
        assert!(app.overlay.is_none());
        let said = app.status.as_ref().map(|s| s.0.as_str());
        assert_eq!(said, Some("alpha is running: stop it first"));

        // A Ctrl right click: a stopped VM's menu holds a running VM's items too. A right click
        // without Ctrl, on the same VM, opens its menu of what can be done now.
        let mut ctrl_click = mouse(MouseEventKind::Down(MouseButton::Right), beta);
        ctrl_click.modifiers = KeyModifiers::CONTROL;
        app.mouse(ctrl_click);
        assert!(matches!(&app.overlay, Some(Overlay::Menu { vm, all: true, .. }) if vm == "beta"));
        let screen = draw(&mut app);
        assert!(find(&screen, app.menu_area, "View screen in terminal").is_some());
        app.mouse(mouse(MouseEventKind::Down(MouseButton::Right), beta));
        assert!(matches!(&app.overlay, Some(Overlay::Menu { vm, all: false, .. }) if vm == "beta"));
        let screen = draw(&mut app);
        assert_eq!(
            find(&screen, app.menu_area, "View screen in terminal"),
            None
        );
    }

    #[test]
    fn a_right_click_or_enter_opens_a_vms_menu_whose_items_do_their_commands() {
        let mut app = test_app(vec![vm_row("alpha"), vm_row("beta")]);
        // A running VM's menu has no OpenCore image item.
        app.rows[1].pid = Some(1);
        let mut terminal =
            ratatui::Terminal::new(ratatui::backend::TestBackend::new(120, 30)).unwrap();
        let mut draw = |app: &mut App| {
            terminal.draw(|f| app.draw(f)).unwrap();
            terminal.backend().buffer().clone()
        };
        let screen = draw(&mut app);
        let list = app.list_rows;

        // A right click selects the VM and opens its menu below the cell clicked, each item
        // with its key at its right.
        let beta = find(&screen, list, "beta").unwrap();
        app.mouse(mouse(MouseEventKind::Down(MouseButton::Right), beta));
        assert!(matches!(&app.overlay, Some(Overlay::Menu { vm, .. }) if vm == "beta"));
        assert_eq!(app.selected_name().as_deref(), Some("beta"));
        let screen = draw(&mut app);
        let menu = app.menu_area;
        assert_eq!((menu.x, menu.y), (beta.x, beta.y + 1));
        assert_eq!(find(&screen, menu, "Make a new OpenCore"), None);
        let edit = find(&screen, menu, "Edit settings…").unwrap();
        assert_eq!(screen[(menu.right() - 3, edit.y)].symbol(), "e");

        // A click outside closes it; the wheel moves up the list.
        app.mouse(mouse(
            MouseEventKind::Down(MouseButton::Left),
            Position::new(0, 0),
        ));
        assert!(app.overlay.is_none());
        app.mouse(mouse(MouseEventKind::ScrollUp, beta));
        assert_eq!(app.selected_name().as_deref(), Some("alpha"));

        // Enter opens the selected VM's menu below its name. Power off is a running VM's: its
        // key leaves the menu open, and Esc closes it.
        app.key(KeyEvent::from(KeyCode::Enter));
        let screen = draw(&mut app);
        let alpha = find(&screen, list, "alpha").unwrap();
        assert_eq!((app.menu_area.x, app.menu_area.y), (alpha.x, alpha.y + 1));
        assert!(find(&screen, app.menu_area, "Make a new OpenCore image…").is_some());
        app.key(KeyEvent::from(KeyCode::Char('K')));
        assert!(matches!(&app.overlay, Some(Overlay::Menu { vm, .. }) if vm == "alpha"));
        app.key(KeyEvent::from(KeyCode::Esc));
        assert!(app.overlay.is_none());

        // Once the VM runs, Start leaves its menu, and Enter does the item shown chosen instead.
        app.key(KeyEvent::from(KeyCode::Enter));
        app.rows[0].pid = Some(1);
        let screen = draw(&mut app);
        let view = find(&screen, app.menu_area, "View screen in terminal").unwrap();
        assert_eq!(screen[view].bg, Color::Cyan);
        app.key(KeyEvent::from(KeyCode::Enter));
        assert!(app.overlay.is_none() && app.pending_view.is_some());

        // A click on an item does it, to the VM the menu is of.
        app.mouse(mouse(MouseEventKind::Down(MouseButton::Right), beta));
        let screen = draw(&mut app);
        let edit = find(&screen, app.menu_area, "Edit settings…").unwrap();
        app.table.select(Some(0));
        app.mouse(mouse(MouseEventKind::Down(MouseButton::Left), edit));
        assert!(matches!(&app.overlay, Some(Overlay::Form(FormKind::Edit(n), _)) if n == "beta"));
        assert_eq!(app.selected_name().as_deref(), Some("beta"));
        // So does its key.
        app.overlay = None;
        app.key(KeyEvent::from(KeyCode::Enter));
        app.key(KeyEvent::from(KeyCode::Char('e')));
        assert!(matches!(&app.overlay, Some(Overlay::Form(FormKind::Edit(n), _)) if n == "beta"));
    }

    #[test]
    fn a_menu_near_the_screens_edges_stays_on_the_screen() {
        let items = [
            (VmCommand::Start, "Start"),
            (VmCommand::Edit, "Edit settings…"),
            (VmCommand::Delete, "Delete…"),
        ];
        let mut terminal =
            ratatui::Terminal::new(ratatui::backend::TestBackend::new(40, 12)).unwrap();
        let mut at = |p: Position| {
            let mut r = Rect::default();
            terminal
                .draw(|f| {
                    let area = f.area();
                    r = draw_menu(f, area, "t", &items, VmCommand::Start, p);
                })
                .unwrap();
            r
        };
        // Below the cell, from it.
        assert_eq!(at(Position::new(2, 1)), Rect::new(2, 2, 22, 5));
        // Above it where there is no room below, and left where there is none to the right.
        assert_eq!(at(Position::new(35, 10)), Rect::new(18, 5, 22, 5));
        // Up where the cell is below the screen, as when the terminal shrinks.
        assert_eq!(at(Position::new(35, 20)), Rect::new(18, 7, 22, 5));
    }

    /// The logs toggle saves the VM's settings as they are saved now, not as the list last read
    /// them (it reads them every 5 seconds): a reims-ctl set from a shell just before stays.
    #[test]
    fn toggling_logs_keeps_what_was_saved_since_the_list_was_read() {
        let env = crate::testenv::Env::new("tui-logs");
        let dir = env.root.join("vms/t");
        std::fs::create_dir_all(&dir).unwrap();
        let config = r#"
            name = "t"
            profile = "q35-x86_64"
            cpus = 2
            memory_mib = 4096
            disk = "disk.qcow2"
            [q35]
            opencore = "opencore.qcow2"
            firmware_code = "OVMF_CODE.fd"
            firmware_vars = "OVMF_VARS.fd"
        "#;
        std::fs::write(dir.join(crate::vm::CONFIG_FILE), config).unwrap();
        let mut reg = Registry::default();
        reg.add("t", &dir).unwrap();
        reg.save().unwrap();
        let mut app = App::with(Host::current());
        app.reload().unwrap();
        let mut vm = Vm::load(&dir).unwrap();
        vm.config.memory_mib = 8192;
        vm.save().unwrap();
        press(&mut app, [KeyCode::Char('l')]);
        let saved = Vm::load(&dir).unwrap().config;
        assert!(saved.logs.enabled);
        assert_eq!(saved.memory_mib, 8192);
    }
}
