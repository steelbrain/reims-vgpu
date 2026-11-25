//! The interactive interface: the VMs, their details, and what can be done with them.
//!
//! Slow work (starting, stopping, creating, a bug report) runs on a thread of its own, which
//! reports its steps; the list shows each VM's step until it ends. Each VM's state is read
//! again every second, so VMs started or stopped elsewhere (the CLI, a printed script, closing
//! a VM's window, the guest shutting down) show as they are, and a status line that states a
//! VM's state says it again once it is another.
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
use crate::{disks, install, logs, opencore, paths, reims_util, runtime, update, usb};
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
    stop_restores();
    result?;
    if app.update_requested {
        return run_update(app.update_channel);
    }
    Ok(())
}

/// Stops the restores of arm64 VMs still running as the interface ends, and waits while each
/// removes its VM's files, so none is left half made.
fn stop_restores() {
    if crate::restore::running() == 0 {
        return;
    }
    println!("Stopping the restore of each arm64 VM being made, which removes its files…");
    crate::restore::cancel();
    if !crate::restore::wait_stopped(Duration::from_secs(120)) {
        eprintln!(
            "warning: a restore did not stop within 2 minutes: its VM's directory may be left; \
             delete it before you create the VM again"
        );
    }
}

/// Runs the update in the plain terminal, then reims-ctl again in this process's place: the new
/// one, or the same where the install was already up to date.
fn run_update(channel: Option<Channel>) -> Result<()> {
    println!("Updating reims-ctl and its QEMU…\n");
    let exe = update::restart_path(std::env::current_exe()?);
    match update::update(channel)? {
        update::Outcome::Updated => println!("\nUpdated. Starting the new reims-ctl…"),
        update::Outcome::Current => println!("\nStarting reims-ctl again…"),
    }
    let err = crate::os::exec(std::process::Command::new(&exe));
    Err(err).with_context(|| format!("run {}", exe.display()))
}

struct VmRow {
    name: String,
    dir: PathBuf,
    vm: Result<Vm, String>,
    pid: Option<i32>,
}

/// A VM's state as a status line states it: the VM, and its QEMU's pid, or None where it is
/// stopped, as [`VmRow::pid`] reads it.
struct Stated {
    vm: String,
    pid: Option<i32>,
}

/// What a work says as it succeeds: its line, and the state of a VM where the line states it.
struct Said {
    line: String,
    stated: Option<Stated>,
}

impl Said {
    /// `line`, which states that `vm` runs as the QEMU of `pid`, or, with None, that it is
    /// stopped.
    fn stating(line: String, vm: &str, pid: Option<i32>) -> Said {
        let stated = Some(Stated {
            vm: vm.to_string(),
            pid,
        });
        Said { line, stated }
    }
}

impl From<String> for Said {
    fn from(line: String) -> Said {
        Said { line, stated: None }
    }
}

/// A work's messages carry its VM and the id `spawn` gave it.
enum Msg {
    Progress(String, u64, String),
    Done(String, u64, Result<Said, String>),
    Gpus(Vec<GpuInfo>),
    /// Apple's catalog of IPSWs, read as the interface starts on an arm64 host, and as the first
    /// arm64 VM's form opens on another (ask_ipsws).
    Ipsws(Result<Vec<reims_restore::catalog::Entry>, String>),
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
    /// What's downloaded for new VMs (downloads.rs): `choice` is the selected row.
    Downloads {
        items: Vec<crate::downloads::Item>,
        choice: usize,
    },
    /// The host's USB devices, to give the VM `vm` or take back (usb.rs): `choice` is the
    /// selected row of [`usb::rows`].
    Usb {
        vm: String,
        devices: Vec<usb::HostDevice>,
        choice: usize,
    },
    /// The host's disks, to give the VM `vm` or take back (disks.rs): `found` as last read,
    /// `own` the devices that hold the VM's images, and `choice` the selected row of
    /// [`disks::rows`].
    Disks {
        vm: String,
        found: Vec<disks::HostDisk>,
        own: Vec<disks::Dev>,
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
    /// A new VM of the profile.
    Create(Profile),
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
    Delete {
        vm: String,
        keep_files: bool,
    },
    ForceStop(String),
    UpdateOpencore(String),
    Update(Option<Channel>),
    /// Deletes a download by its name, then shows the downloads again.
    DeleteDownload(String),
    /// Shows the downloads again.
    Downloads,
    /// Opens the form of a new VM of the profile.
    Create(Profile),
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
    Disks,
    ShowCommand,
    ToggleLogs,
    ViewLogs,
    Report,
    UpdateOpencore,
    Delete,
}

impl VmCommand {
    /// Each with its key, in the menu's order.
    const ALL: [(VmCommand, char); 13] = [
        (VmCommand::Start, 's'),
        (VmCommand::View, 'v'),
        (VmCommand::Stop, 'x'),
        (VmCommand::PowerOff, 'K'),
        (VmCommand::Edit, 'e'),
        (VmCommand::Usb, 'p'),
        (VmCommand::Disks, 'P'),
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

#[derive(Clone, Copy, Debug, PartialEq)]
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
    /// The VM's state the status line states, where it states one: each read of the VMs' states
    /// says the state again once it is another (`restate`).
    stated: Option<Stated>,
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
    /// Apple's catalog of IPSWs, once read.
    ipsw_catalog: Option<Result<Vec<reims_restore::catalog::Entry>, String>>,
    /// Whether Apple's catalog of IPSWs was asked for (ask_ipsws).
    ipsws_asked: bool,
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
            stated: None,
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
            ipsw_catalog: None,
            ipsws_asked: false,
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
        if host.default_profile() == Some(Profile::VmappleArm64) {
            app.ask_ipsws();
        }
        if matches!(host.os, Os::Linux | Os::Windows) {
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
                &format!(
                    "QEMU is not installed: run {} in your reims-vgpu checkout",
                    install::INSTALL_SCRIPT
                ),
                Level::Warn,
            );
        }
        Ok(app)
    }

    fn say(&mut self, s: &str, level: Level) {
        self.status = Some((s.to_string(), level));
        self.stated = None;
    }

    /// `say`, for a line that states that `vm` runs as the QEMU of `pid`, or, with None, that
    /// it is stopped.
    fn say_state(&mut self, s: &str, level: Level, vm: &str, pid: Option<i32>) {
        self.say(s, level);
        self.stated = Some(Stated {
            vm: vm.to_string(),
            pid,
        });
    }

    /// Says a VM's state again where the status line states it and the VMs' states, as just
    /// read, have it another: the VM stopped, ran again as another QEMU, or was started. A
    /// stop with reims-vgpu's fatal error in QEMU's log since the start names it, and where the
    /// log is. Work of this interface for the VM is left to say what it did as it ends, and a
    /// VM gone from the list, or whose settings do not read, is left as the line says.
    fn restate(&mut self) {
        let Some(stated) = &self.stated else { return };
        if self.activity.contains_key(&stated.vm) {
            return;
        }
        let Some(row) = self.rows.iter().find(|r| r.name == stated.vm) else {
            return;
        };
        let Ok(vm) = &row.vm else { return };
        if row.pid == stated.pid {
            return;
        }
        let (name, pid, log) = (stated.vm.clone(), row.pid, vm.qemu_log());
        if let Some(p) = pid {
            let s = format!("{name}: running (pid {p})");
            return self.say_state(&s, Level::Info, &name, pid);
        }
        match fatal_line(&log) {
            // An error, which stays until another line replaces it.
            Some(fatal) => {
                let log = paths::tilde(&log);
                let s = format!("{name}: stopped: {fatal} (its QEMU log: {log})");
                self.say(&s, Level::Error);
            }
            None => self.say_state(&format!("{name}: stopped"), Level::Info, &name, None),
        }
    }

    /// Reads Apple's catalog of IPSWs in the background, once, for the arm64 form's choices
    /// (Msg::Ipsws): as the interface starts where arm64 VMs are the host's own, and as the first
    /// arm64 VM's form opens where they are its second profile.
    fn ask_ipsws(&mut self) {
        if self.ipsws_asked || self.ipsw_catalog.is_some() {
            return;
        }
        self.ipsws_asked = true;
        let tx = self.tx.clone();
        std::thread::spawn(move || {
            let client = reims_restore::http::Client::apple();
            let r = reims_restore::catalog::fetch(&client, reims_restore::catalog::URL);
            let _ = tx.send(Msg::Ipsws(r.map_err(|e| format!("{e:#}"))));
        });
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
        self.restate();
        Ok(())
    }

    fn refresh_state(&mut self) {
        for r in &mut self.rows {
            r.pid = r.vm.as_ref().ok().and_then(runtime::pid);
        }
        self.restate();
        // Devices plugged in or out show as they are.
        if let Some(Overlay::Usb { devices, .. }) = &mut self.overlay {
            *devices = usb::host_devices();
        }
        if let Some(Overlay::Disks { found, .. }) = &mut self.overlay {
            *found = disks::host_disks();
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
    fn spawn<F, S>(&mut self, vm: &str, first: &str, work: F) -> Option<u64>
    where
        F: FnOnce(&mut dyn FnMut(&str)) -> Result<S> + Send + 'static,
        S: Into<Said>,
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
            let r = work(&mut progress)
                .map(Into::into)
                .map_err(|e| format!("{e:#}"));
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
            // The viewer says something after only as the VM stops (view::run).
            Ok(Some(m)) => self.say_state(&m, Level::Info, vm.name(), None),
            Ok(None) => {}
            Err(e) => self.say(&format!("{}: {e:#}", vm.name()), Level::Error),
        }
        let _ = self.reload();
        Ok(())
    }

    fn message(&mut self, m: Msg) {
        match m {
            Msg::Progress(vm, id, s) => {
                self.say(&format!("{vm}: {s}"), Level::Info);
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
                    Ok(said) => {
                        self.say(&said.line, Level::Ok);
                        self.stated = said.stated;
                    }
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
            Msg::Ipsws(r) => {
                self.ipsw_catalog = Some(r);
                let options = ipsw_options(self.ipsw_catalog.as_ref());
                if let Some(Overlay::Form(FormKind::Create(p), f)) = &mut self.overlay
                    && *p == Profile::VmappleArm64
                    && f.is_shown("source")
                {
                    f.set_options("source", options);
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
            Some(Overlay::Disks {
                vm,
                found,
                own,
                choice,
            }) => self.disks_key(k, vm, found, own, choice),
            Some(Overlay::Downloads { items, choice }) => self.downloads_key(k, items, choice),
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
                        let res =
                            std::fs::write(&p, script).and_then(|_| crate::os::set_mode(&p, 0o755));
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
                 though a download of macOS picks up where it stopped when you create it again; an \
                 arm64 VM being restored is stopped, and its files removed."
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
            KeyCode::Char('D') => self.open_downloads(),
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
            VmCommand::Disks => self.open_disks(),
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
            Action::Downloads => self.open_downloads(),
            Action::DeleteDownload(name) => {
                match crate::downloads::delete(&name) {
                    Ok(s) => self.say(&s, Level::Ok),
                    Err(e) => self.say(&format!("{e:#}"), Level::Error),
                }
                self.open_downloads();
            }
            Action::Quit => self.quit = true,
            Action::Create(profile) => {
                if profile == Profile::VmappleArm64 {
                    self.ask_ipsws();
                }
                self.open_create_form(profile);
            }
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
                    let s = format!("{}: powered off", vm.name());
                    Ok(Said::stating(s, vm.name(), None))
                });
            }
        }
    }

    // ------------------------------------------------------------ actions

    fn start(&mut self) {
        let Some(vm) = self.selected_vm() else { return };
        if let Some(p) = runtime::pid(&vm) {
            let s = format!("{} is already running", vm.name());
            return self.say_state(&s, Level::Info, vm.name(), Some(p));
        }
        // What runtime::warnings finds shows in the VM's details, not here.
        let host = self.host;
        self.spawn(&vm.config.name.clone(), "starting", move |p| {
            // What the start did to its settings first, and why, follows the start's line.
            let mut vm = vm;
            let mut said = Vec::new();
            let warned = runtime::prepare(&mut vm, &mut |s| said.push(s.to_string()));
            let started = runtime::start(&vm, host, p)?;
            let mut s = started_line(vm.name(), started.pid, started.window);
            for n in said.iter().chain(&warned).chain(&started.note) {
                s.push_str("; ");
                s.push_str(n);
            }
            Ok(Said::stating(s, vm.name(), Some(started.pid)))
        });
    }

    fn stop(&mut self, force: bool) {
        let Some(vm) = self.selected_vm() else { return };
        if !runtime::running(&vm) {
            let s = format!("{} is not running", vm.name());
            return self.say_state(&s, Level::Info, vm.name(), None);
        }
        let timeout = vm.config.stop_timeout();
        let first = if force {
            "powering off"
        } else {
            "shutting down"
        };
        self.spawn(&vm.config.name.clone(), first, move |p| {
            runtime::stop(&vm, force, timeout, |s| p(s))?;
            let s = format!("{}: stopped", vm.name());
            Ok(Said::stating(s, vm.name(), None))
        });
    }

    fn ask_force_stop(&mut self) {
        let Some(row) = self.selected() else { return };
        if row.pid.is_none() {
            let n = row.name.clone();
            return self.say_state(&format!("{n} is not running"), Level::Info, &n, None);
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
        if let Some(p) = row.pid {
            let s = format!("{name} is running: stop it first");
            return self.say_state(&s, Level::Warn, &name, Some(p));
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
        if let Some(p) = runtime::pid(&vm) {
            let s = format!("{name} is running: stop it first");
            return self.say_state(&s, Level::Warn, &name, Some(p));
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
                &format!(
                    "no install record: run {} in your reims-vgpu checkout once",
                    install::INSTALL_SCRIPT
                ),
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

    fn open_downloads(&mut self) {
        self.overlay = Some(Overlay::Downloads {
            items: crate::downloads::list(),
            choice: 0,
        });
    }

    fn downloads_key(&mut self, k: KeyEvent, items: Vec<crate::downloads::Item>, choice: usize) {
        let last = items.len().saturating_sub(1);
        let mut choice = choice.min(last);
        match k.code {
            KeyCode::Esc | KeyCode::Char('q') => return,
            KeyCode::Down | KeyCode::Char('j') | KeyCode::Tab => choice = (choice + 1).min(last),
            KeyCode::Up | KeyCode::Char('k') | KeyCode::BackTab => {
                choice = choice.saturating_sub(1)
            }
            KeyCode::Home => choice = 0,
            KeyCode::End => choice = last,
            KeyCode::Char('r') => return self.open_downloads(),
            KeyCode::Char('d') | KeyCode::Delete => {
                if let Some(i) = items.get(choice) {
                    let name = i.name();
                    self.overlay = Some(Overlay::Confirm {
                        title: "Delete a download".into(),
                        body: format!(
                            "Delete {name}, {}? A later VM of its macOS downloads it again.",
                            crate::downloads::size(i.bytes)
                        ),
                        options: vec![
                            ("Delete", Action::DeleteDownload(name)),
                            ("Keep", Action::Downloads),
                        ],
                        choice: 1,
                    });
                    return;
                }
            }
            _ => {}
        }
        self.overlay = Some(Overlay::Downloads { items, choice });
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

    /// Opens the selected VM's host disks, where its host passes them through.
    fn open_disks(&mut self) {
        let Some(vm) = self.selected_vm() else { return };
        if let Err(why) = disks_available(self.host, &vm) {
            return self.say(&why, Level::Warn);
        }
        let own = disks::own_devs(&vm, &disks::HostUse::current());
        self.overlay = Some(Overlay::Disks {
            vm: vm.config.name.clone(),
            found: disks::host_disks(),
            own,
            choice: 0,
        });
    }

    /// The rows of the disks' view of the VM named `vm`: the host's disks `found`, each with
    /// what of the VM's settings, as the list last read them, gives it.
    fn disks_rows(&self, vm: &str, found: &[disks::HostDisk]) -> Vec<disks::Row> {
        let chosen = self
            .rows
            .iter()
            .find(|r| r.name == vm)
            .and_then(|r| r.vm.as_ref().ok())
            .map(|vm| vm.config.host_disks.clone())
            .unwrap_or_default();
        disks::rows(&chosen, found)
    }

    fn disks_key(
        &mut self,
        k: KeyEvent,
        vm: String,
        mut found: Vec<disks::HostDisk>,
        own: Vec<disks::Dev>,
        choice: usize,
    ) {
        let rows = self.disks_rows(&vm, &found);
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
                found = disks::host_disks();
                self.say("rescanned this host's disks", Level::Info);
            }
            KeyCode::Char(' ') | KeyCode::Enter => {
                if let Some(row) = rows.get(choice) {
                    self.toggle_disk(&vm, row, &own);
                }
            }
            _ => {}
        }
        self.overlay = Some(Overlay::Disks {
            vm,
            found,
            own,
            choice,
        });
    }

    /// Gives the VM named `name` the disk of `row`, or takes it back, from its next start
    /// (disks::apply): a disk the host uses is refused, saying why. `own` are the devices that
    /// hold the VM's images.
    fn toggle_disk(&mut self, name: &str, row: &disks::Row, own: &[disks::Dev]) {
        let Some(vm) = self
            .rows
            .iter()
            .find(|r| r.name == name)
            .and_then(|r| r.vm.as_ref().ok())
            .cloned()
        else {
            return;
        };
        match disks::apply(&vm, |cur| disks::toggle(cur, row, own)) {
            Ok((_, said)) => self.say(&said.join("; "), Level::Ok),
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
            let s = format!("{name} is not running: s starts it");
            return self.say_state(&s, Level::Warn, &name, None);
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
            concat!(
                "The GPU the library draws on. Automatic takes the only GPU, or, of several, a \
                 discrete GPU before an integrated one. ",
                // cli::CROSSING, which concat! cannot name.
                "If your screen is plugged into a different GPU than the VM's, the window system \
                 copies every frame across to it over the VM's GPU's link: an RTX 3090 in a PCIe \
                 3.0 x4 slot spends about 16 ms a frame on it at 3648 x 2039."
            )
        };
        Field::choice("gpu", "GPU", options, &selected, help)
    }

    /// The form of a new VM, or, where the host runs VMs of two profiles (x86-64 and arm64 VMs on
    /// x86-64 Linux), first the choice of one.
    fn open_create(&mut self) {
        let host = self.host;
        let profiles = Profile::supported_on(host);
        match profiles {
            [] => self.say(crate::host::UNSUPPORTED, Level::Error),
            [profile] => self.open_create_form(*profile),
            _ => {
                let mut body = String::from("This host runs:");
                for p in profiles {
                    body += &format!("\n- {}", p.describe(host));
                }
                if profiles.iter().any(|p| p.accel(host) == "tcg") {
                    body += "\n\nTCG, QEMU's emulation, runs an arm64 guest several times slower \
                             than an arm64 host does.";
                }
                let mut options: Vec<(&'static str, Action)> = profiles
                    .iter()
                    .map(|&p| (profile_label(p), Action::Create(p)))
                    .collect();
                options.push(("Cancel", Action::None));
                self.overlay = Some(Overlay::Confirm {
                    title: "New VM".into(),
                    body,
                    options,
                    choice: 0,
                });
            }
        }
    }

    /// The form of a new VM of `profile`, one the host runs.
    fn open_create_form(&mut self, profile: Profile) {
        if profile == Profile::VmappleArm64 {
            let options = ipsw_options(self.ipsw_catalog.as_ref());
            let first = options[0].0.clone();
            let mut fields = vec![
                Field::text("name", "Name", "", "Letters, digits, '-', '_' and '.'."),
                Field::choice(
                    "source",
                    "macOS from",
                    options,
                    &first,
                    "The macOS restored onto the VM's new disk, with tickets Apple signs for \
                     this VM, as a Mac restores one. Apple offers only the current macOS to \
                     download: reims-ctl downloads its IPSW once (about 25 GB) and keeps it for \
                     the next VM (D lists what's kept); a stopped download continues. For \
                     another macOS, give an IPSW of your own.",
                ),
                Field::path(
                    "ipsw",
                    "IPSW",
                    "",
                    "A Mac's IPSW (Tab completes), such as \
                     UniversalMac_14.6.1_23G93_Restore.ipsw, read where it is.",
                )
                .show_if("source", &["path"]),
                Field::text(
                    "disk_size",
                    "Disk size",
                    &format_mib(create::DEFAULT_DISK_MIB),
                    "The new disk's size; it takes only what macOS writes, about the IPSW's size.",
                ),
            ];
            fields.extend(place_fields());
            let form = Form::new("New arm64 VM", "create", fields);
            self.overlay = Some(Overlay::Form(FormKind::Create(profile), form));
            return;
        }
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
            Field::path("installer", "Installer image", "", "An installer to boot (Tab completes): a raw image, such as a recovery image's BaseSystem.img, or a DMG, converted to a raw image in the VM's directory. Once macOS is installed, eject it in the edit form (e), with the VM running or not: it goes at the VM's next start.").show_if("source", &["installer"]),
            Field::text("disk_size", "Disk size", "128G", "The new disk's size; it takes only what the guest writes.").show_if("source", &["download", "installer"]),
            Field::path("disk_path", "Disk image", "", "A qcow2 or raw image (Tab completes).").show_if("source", &["disk"]),
            Field::choice("disk_mode", "Use the disk by", opts(&[("copy", "Copying it (instant where the file system reflinks)"), ("in-place", "Using it in place")]), "copy", "A copy leaves the image as it was; in place, the VM writes the image itself. A copy is instant where the image and the VM's directory are on one file system that makes reflinks, such as btrfs or XFS; else it is a full copy, which says so, with its size, before it starts.")
                .show_if("source", &["disk"]),
            Field::path("opencore", "OpenCore image", "", "Empty makes one for this VM, with OVMF, presenting the Mac of the image beside an existing disk (opencore.qcow2, as a VM's directory holds them), unless the disk is copied from a VM reims-ctl has, else a Mac identity of its own. Or an OpenCore image of yours to copy, with OVMF_CODE.fd and OVMF_VARS.fd beside it, such as the one a disk of macOS booted from, which keeps its Mac's identity.")
                .placeholder("made for this VM"),
        ]);
        fields.extend(place_fields());
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
            Field::toggle(
                "network",
                "Network",
                true,
                "User-mode networking (NAT). Installing from an offline installer, turn it off: \
                 with a network, its installer can download an update as it installs, a second \
                 download of the release. Turn it on in the VM's edit form once macOS is set up.",
            ),
            Field::text(
                "ssh_port",
                "SSH port",
                "",
                "Optional: a port on 127.0.0.1 forwarded to the guest's SSH (Remote Login).",
            )
            .placeholder("optional")
            .show_if("network", &["yes"]),
        ]);
        let form = Form::new("New VM", "create", fields);
        self.overlay = Some(Overlay::Form(FormKind::Create(profile), form));
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
        // An arm64 VM's form has these two and capture input alone: its displays are reims-ctl
        // set's.
        if c.profile == Profile::VmappleArm64 {
            fields.push(capture_field(c.display.capture));
            let form = Form::new(&format!("Edit {}", c.name), "save", fields);
            self.overlay = Some(Overlay::Form(FormKind::Edit(c.name.clone()), form));
            return;
        }
        // The displays', the resolution's, borderless's, capture input's and the GPU's fields are
        // reims-vgpu's.
        fields.push(graphics_field(c.graphics()));
        fields.extend([
            displays_field(c.display.count).show_if("graphics", LIBRARY),
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
                "Windows without borders that fill their screens, one for each display.",
            )
            .show_if("graphics", LIBRARY),
            capture_field(c.display.capture).show_if("graphics", LIBRARY),
        ]);
        if c.profile.gpu_choice(self.host) {
            fields.push(self.gpu_field(&c.gpu).show_if("graphics", LIBRARY));
        }
        fields.extend([
            Field::toggle(
                "audio",
                "Audio",
                c.audio.enabled,
                "Two virtio sound devices on the host's sound, a speaker and a microphone, and \
                 the HDA controller OpenCore's boot chime plays on.",
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
                "SATA, NVMe or virtio. macOS's installers and macOS find the disk on SATA or \
                 NVMe. On virtio an installed macOS 26 starts; other releases and the installers \
                 have not been tried there. Once macOS is installed, eject the installer before \
                 a move, or OpenCore may start it instead. Both can change while the VM runs, \
                 and take effect at its next start.",
            ));
            fields.push(
                Field::path(
                    "installer",
                    "Installer image",
                    &installer_shown(&vm),
                    "A raw image, or a DMG, converted to a raw image in the VM's directory (Tab completes). Empty to eject, with the VM running or not: the installer goes at its next start.",
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
            FormKind::Create(profile) => {
                let spec = self.spec_from(*profile, f)?;
                let host = self.host;
                let name = spec.name.clone();
                self.spawn_for(f, &name, "creating", move |p| {
                    let vm = create::create(spec, host, p)?;
                    if vm.config.profile == Profile::VmappleArm64 {
                        return Ok(format!(
                            "created {} in {}: its macOS is restored · s starts it; after its \
                             first boot, which makes its Data volume, stop it and run \
                             reims-restore finish-eacs {}",
                            vm.name(),
                            paths::tilde(&vm.dir),
                            paths::tilde(&vm.dir)
                        ));
                    }
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

    fn spec_from(&self, profile: Profile, f: &Form) -> Result<Spec> {
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
        s.profile = Some(profile);
        let dir = f.text("dir");
        create::check_dir(std::path::Path::new(&dir))?;
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
        match profile {
            Profile::Q35X86_64 => {
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
            Profile::VmappleArm64 => {
                s.ipsw = Some(match f.choice("source").as_str() {
                    "path" => opt_path("ipsw")
                        .context("give the IPSW to restore")?
                        .display()
                        .to_string(),
                    // latest, or the path of an IPSW downloaded before
                    other => other.to_string(),
                });
                s.disk = Some(DiskSource::New {
                    size_mib: parse_size_mib(&f.text("disk_size")).context("disk size")?,
                });
            }
        }
        if f.is_shown("graphics") {
            s.graphics = Some(Graphics::parse(&f.choice("graphics"))?);
        }
        // Hidden on VMware SVGA, which opens at the firmware's resolution: the default.
        if f.is_shown("resolution") {
            s.display.resolution = f.text("resolution");
        }
        s.display.check()?;
        // The arm64 form has neither: its VM takes the defaults.
        if f.is_shown("logs") {
            s.logs = f.toggle("logs");
        }
        if f.is_shown("network") {
            s.network = f.toggle("network");
            if s.network {
                s.ssh_port = ssh_port_field(f)?;
            }
        }
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
            Some(Overlay::Disks {
                vm,
                found,
                own,
                choice,
            }) => {
                let rows = self.disks_rows(vm, found);
                let of = self.rows.iter().find(|r| &r.name == vm);
                let running = of.is_some_and(|r| r.pid.is_some());
                let most = of
                    .and_then(|r| r.vm.as_ref().ok())
                    .and_then(|vm| vm.config.q35.as_ref())
                    .map_or(0, |q| q.host_disk_ports().len());
                draw_disks(f, area, vm, running, &rows, own, most, *choice);
            }
            Some(Overlay::Downloads { items, choice }) => draw_downloads(f, area, items, *choice),
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
            .map(|p| p.describe(self.host).to_string())
            .unwrap_or_else(|| "no VMs on this host".into());
        let mut spans = vec![
            Span::styled(
                " reims-ctl ",
                Style::new().fg(Color::Black).bg(Color::Cyan).bold(),
            ),
            Span::raw(format!("  {} · {profile}", self.host)),
        ];
        if self.host.default_profile().is_some() {
            spans.push(Span::raw(" · "));
            spans.push(match &self.qemu {
                Some(q) => Span::styled(q.clone(), Style::new().fg(Color::Green)),
                None => Span::styled("QEMU not installed", Style::new().fg(Color::Red).bold()),
            });
        }
        f.render_widget(Paragraph::new(Line::from(spans)), area);
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
                        .chain(disks::problems(vm))
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
                VmCommand::Disks => (
                    "Host disks…",
                    vm.is_some_and(|vm| disks_available(host, vm).is_ok()),
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

/// Whether the VM can be given the host's USB devices: any VM on a Linux host, and an x86-64
/// VM on a Windows one ([`crate::usb::available`]). Else why not, to say.
fn usb_available(host: Host, vm: &Vm) -> Result<(), String> {
    crate::usb::available(host)?;
    if vm.config.profile == Profile::VmappleArm64 && host.os != Os::Linux {
        return Err(format!("{}: {}", vm.name(), crate::usb::LINUX_ONLY));
    }
    Ok(())
}

/// Whether the VM can be given the host's disks: an x86-64 VM on a Linux host. Else why not,
/// to say.
fn disks_available(host: Host, vm: &Vm) -> Result<(), String> {
    if host.os != Os::Linux
        || host.default_profile() != Some(Profile::Q35X86_64)
        || vm.config.q35.is_none()
    {
        return Err(disks::ONLY.into());
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
        select: None,
        name: None,
    }
}

/// The edit form's capture input field: whether a click into the VM's windows captures the
/// mouse and the keyboard ([`crate::vm::Display::capture`]).
fn capture_field(on: bool) -> Field {
    Field::toggle(
        "capture",
        "Capture input",
        on,
        "On: a click into the VM's window captures the mouse and the keyboard, until the keys \
         its title names let them go. Off: the window never captures them: the mouse works over \
         the guest's screen and leaves the window freely, keys go to the guest while its window \
         is in front, and the host keeps its own shortcuts, such as Alt+Tab and \u{2318}Tab.",
    )
}

const RESOLUTION_HELP: &str = "The size each window opens at, WIDTHxHEIGHT in the desktop's \
     points, shrunk to fit the screen; the guest takes the window's size in pixels, and follows \
     it as you resize it. It starts a HiDPI desktop by itself on a display whose window is at \
     least 3840 pixels wide.";

/// The choice of a VM's displays on reims-vgpu, at `current`.
fn displays_field(current: u32) -> Field {
    let options = (1..=crate::vm::DISPLAYS_MAX)
        .map(|n| (n.to_string(), n.to_string()))
        .collect();
    Field::choice(
        "displays",
        "Displays",
        options,
        &current.to_string(),
        "How many displays the guest has, each in a window of its own, where macOS shows its \
         desktop; closing any of the windows powers the VM off. VMware SVGA has one.",
    )
}

/// The graphics field's values that show a field of the library's.
const LIBRARY: &[&str] = &["reims-vgpu"];

/// A new VM's directory, vCPUs and memory: the create forms' fields.
fn place_fields() -> Vec<Field> {
    vec![
        Field::path(
            "dir",
            "Storage directory",
            "",
            "Where its disks and settings go (Tab completes). Empty for the default.",
        )
        .placeholder(&format!(
            "default: {}/{}",
            paths::tilde(&paths::default_vms_dir()),
            create::NAME_PLACEHOLDER
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
    ]
}

/// The graphics choice of a VM, at `current`.
fn graphics_field(current: Graphics) -> Field {
    let options = Graphics::ALL.map(|g| (g.id().to_string(), g.describe().to_string()));
    Field::choice(
        "graphics",
        "Graphics",
        options.to_vec(),
        current.id(),
        "reims-vgpu is its own GPU device, which macOS drives with the drivers reims-ctl puts in \
         the VM's OpenCore image (after a change to it, U makes the image again): one to four \
         displays, each in a window whose size the guest follows. VMware SVGA is QEMU's: one \
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
    // An arm64 VM's form has these two and capture input alone (open_edit).
    if c.profile == Profile::VmappleArm64 {
        c.display.capture = f.toggle("capture");
        return c.check();
    }
    if f.is_shown("graphics") {
        c.graphics = Some(Graphics::parse(&f.choice("graphics"))?);
    }
    if f.is_shown("disk_controller")
        && let Some(q) = &mut c.q35
    {
        q.disk_controller = DiskController::parse(&f.choice("disk_controller"))?;
    }
    // Hidden on VMware SVGA, which has one display, the displays, the resolution, the windows'
    // first size, borderless and capture input stay as they were.
    if f.is_shown("displays") {
        c.display.count = f.choice("displays").parse().context("displays: a number")?;
    }
    if f.is_shown("resolution") {
        c.display.resolution = f.text("resolution");
        c.display.check()?;
    }
    if f.is_shown("borderless") {
        c.display.borderless = f.toggle("borderless");
    }
    if f.is_shown("capture") {
        c.display.capture = f.toggle("capture");
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

/// The line a start from this interface ends with: QEMU's pid, and that the VM's window is open
/// where the start opened one ([`runtime::Started::window`]); else that it runs without one, as
/// on a host where no window can open, and the key that shows its screen in this terminal.
fn started_line(name: &str, pid: i32, window: bool) -> String {
    if window {
        format!("{name}: running (pid {pid}); its window is open")
    } else {
        format!(
            "{name}: running (pid {pid}) without a window; {} shows its screen in this terminal",
            VmCommand::View.key()
        )
    }
}

/// How reims-vgpu's fatal line starts: the line the library writes to QEMU's stderr, in one
/// write, as it ends the process (`abort`), and at no other time.
const FATAL: &str = "reims-vgpu: fatal: ";

/// reims-vgpu's fatal line in the QEMU log `log` since the VM's start: after the `----` that
/// ends the command a start writes to the log ([`runtime::start`]), each start's log a new file.
fn fatal_line(log: &std::path::Path) -> Option<String> {
    let text = runtime::tail(log, usize::MAX);
    let lines: Vec<&str> = text.lines().collect();
    let run = match lines.iter().rposition(|l| l.trim() == "----") {
        Some(i) => &lines[i + 1..],
        None => &lines[..],
    };
    run.iter()
        .rev()
        .find_map(|l| l.find(FATAL).map(|i| l[i..].trim_end().to_string()))
}

/// Runs less on `files`, each at its end, in the terminal ([`crate::os::run_in_terminal`]).
fn run_less(files: &[PathBuf]) -> std::io::Result<std::process::ExitStatus> {
    let mut cmd = std::process::Command::new("less");
    cmd.arg("++G").args(files);
    crate::os::run_in_terminal(cmd)
}

/// Has the terminal report the mouse's buttons and wheel, or stop: X11's reports (mode 1000)
/// in SGR's form (1006), which holds any column. Not the mouse moving, which nothing here
/// reads.
#[cfg(unix)]
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

/// Has the console report the mouse, or stop: on Windows crossterm reads the mouse from the
/// console's mouse records, which come only while its mouse input is on (crossterm's mouse
/// capture), the mouse moving among them. A stop where it was never on fails, which is no
/// matter.
#[cfg(windows)]
fn report_mouse(on: bool) -> std::io::Result<()> {
    use ratatui::crossterm::event::{DisableMouseCapture, EnableMouseCapture};
    if on {
        execute!(std::io::stdout(), EnableMouseCapture)
    } else {
        let _ = execute!(std::io::stdout(), DisableMouseCapture);
        Ok(())
    }
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
reims-ctl runs x86-64 macOS VMs on QEMU with reims-vgpu: each of a VM's displays opens in a window
of its own. A VM runs reims-vgpu's own GPU device unless its settings choose another (e, graphics).

  ↑ ↓ / j k   choose a VM, or click it
  Enter       open its menu: what can be done with it now, each with its key. A right
              click on a VM opens its menu too. Hold Ctrl as you press Enter or right click
              for a menu of everything, whether it can be done now or not (Ctrl-Enter needs
              a terminal that tells it from Enter, such as kitty, foot or Ghostty on a Mac)
  n           create a VM. Choose a macOS version: reims-ctl downloads its
              installer from Apple and makes the VM's OpenCore image and disk, and the VM
              boots into the installer; its details then say how to install macOS. The form
              stays open until the VM is made, and shows what failed, to fix and create again;
              Esc closes it meanwhile, and it opens again if the create fails. On an arm64
              Mac or arm64 Linux, it makes an arm64 VM: from the current macOS, which
              reims-ctl downloads from Apple (about 25 GB) and keeps for the next, or an IPSW
              you have, restored onto its new disk. On x86-64 Linux, which runs both, it asks
              which first; there an arm64 VM is emulated by TCG, several times slower
  s           start it
  x           stop it: the guest is asked to shut down, and powered off if it has not
              when its shutdown timeout ends (edit e to change it)
  K           kill it: power it off at once
  e           edit its settings: CPUs, memory, displays, resolution, borderless, capture
              input, GPU, audio, network, logs, extra QEMU arguments (they apply at its next
              start), its disk controller (SATA, NVMe or virtio), its installer image, its boot
              selection (OpenCore's picker, which makes it a new OpenCore image) and
              graphics: reims-vgpu, its own GPU device, on one to four displays, each in a
              window whose size the guest follows, with the macOS drivers reims-ctl puts in
              its OpenCore image (made again on a change); or QEMU's VMware SVGA, one display
              with no GPU acceleration, in QEMU's own window. Displays, resolution,
              borderless, capture input and GPU are reims-vgpu's. Capture input off, a click
              into the VM's window never captures the mouse and the keyboard
  p           give it this host's USB devices: Space gives the selected one to
              the guest or takes it back, at once while the VM runs; a device you gave it
              joins the guest when it is plugged in. More in reims-ctl usb --help
  P           give it whole disks of this host's: Space gives the selected one to the
              guest or takes it back, from the VM's next start; a disk this host uses can't
              be given. More in reims-ctl disks --help
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
  D           downloads: the macOS installers and IPSWs kept for new VMs, and the space
              each takes; d deletes the selected one (a later VM of its macOS downloads it
              again). Also reims-ctl downloads
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

/// Draws the disks' view of the VM `vm`: each of `rows` with a box checked where the VM has
/// the disk, its name and size, its file and notes, `choice` highlighted, and what lets QEMU
/// open the selected disk where this user can't. `own` are the devices that hold the VM's
/// images, and `most` the disks it takes.
#[allow(clippy::too_many_arguments)]
fn draw_disks(
    f: &mut Frame,
    area: Rect,
    vm: &str,
    running: bool,
    rows: &[disks::Row],
    own: &[disks::Dev],
    most: usize,
    choice: usize,
) {
    let width = area.width.saturating_sub(4).min(100);
    let inner_w = width.saturating_sub(4) as usize;
    let mut lines: Vec<Line> = Vec::new();
    let when = if running {
        "from its next start (until it stops, it keeps the ones it started with)"
    } else {
        "from its next start"
    };
    lines.push(
        Line::from(format!(
            "This host's disks. One you give {vm} is the guest's, whole, {when}. A disk this \
             host uses can't be given: a guest writing it would destroy it."
        ))
        .dim(),
    );
    lines.push(Line::from(""));
    let first_row = lines.len();
    if rows.is_empty() {
        lines.push(Line::from("This host has no disks to list."));
    }
    let name_w = rows
        .iter()
        .map(|r| r.name().chars().count())
        .max()
        .unwrap_or(0)
        .min(inner_w.saturating_sub(24).clamp(12, 44));
    for (i, r) in rows.iter().enumerate() {
        let mut name: String = r.name().chars().take(name_w).collect();
        if r.name().chars().count() > name_w {
            name.pop();
            name.push('…');
        }
        let notes = r.notes(own);
        let text = format!(
            "{} [{}] {name:<name_w$}  {}{}",
            if i == choice { "›" } else { " " },
            if r.chosen.is_some() { "x" } else { " " },
            r.path().display(),
            if notes.is_empty() {
                String::new()
            } else {
                format!("  {}", notes.join(", "))
            }
        );
        lines.push(if i == choice {
            Line::from(text).style(Style::new().fg(Color::Black).bg(Color::Cyan))
        } else if r.disk.is_none() {
            Line::from(text).dim()
        } else {
            Line::from(text)
        });
    }
    lines.push(Line::from(""));
    // What lets QEMU open the selected disk, where this user can't and the host leaves it be.
    if let Some(d) = rows
        .get(choice)
        .and_then(|r| r.disk.as_ref())
        .filter(|d| !d.usable() && disks::refusal(d, own).is_none())
    {
        let fix = format!("QEMU can't open what you can't: {}", disks::fix(d));
        for l in fix.lines() {
            lines.push(Line::from(l.to_string()).style(Style::new().fg(Color::Yellow)));
        }
    }
    lines.push(
        Line::from(format!(
            "Space give or take back · r rescan · Esc close · up to {most} disks a VM"
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
        .title(Line::from(format!(" {vm}: host disks ")).bold())
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

/// A profile's choice in the new VM's question (open_create).
fn profile_label(p: Profile) -> &'static str {
    match p {
        Profile::Q35X86_64 => "x86-64 macOS",
        Profile::VmappleArm64 => "arm64 macOS",
    }
}

/// The arm64 form's choices of macOS: the current one from Apple (`latest`), as Apple's catalog
/// names it once read, unless it is downloaded already; each IPSW downloaded, by its path; or
/// an IPSW of the user's own (`path`).
fn ipsw_options(
    catalog: Option<&Result<Vec<reims_restore::catalog::Entry>, String>>,
) -> Vec<(String, String)> {
    let have = crate::ipsw::on_disk();
    let mut v = Vec::new();
    match catalog
        .and_then(|c| c.as_ref().ok())
        .and_then(|l| l.first())
    {
        Some(e) if have.iter().any(|o| o.build == e.build) => {}
        Some(e) => v.push((
            "latest".into(),
            format!("macOS {} ({}), downloaded from Apple", e.version, e.build),
        )),
        None => v.push((
            "latest".into(),
            "The current macOS, downloaded from Apple".into(),
        )),
    }
    for o in &have {
        v.push((
            o.path.display().to_string(),
            format!("macOS {} ({}), downloaded", o.version, o.build),
        ));
    }
    v.push(("path".into(), "An IPSW of my own".into()));
    v
}

/// Draws what's downloaded for new VMs: each of `items` with the space it takes and what it is,
/// `choice` highlighted, and the space they take in all.
fn draw_downloads(f: &mut Frame, area: Rect, items: &[crate::downloads::Item], choice: usize) {
    let width = area.width.saturating_sub(4).min(110);
    let inner_w = width.saturating_sub(2) as usize;
    let mut lines: Vec<Line> = vec![
        Line::from(
            "What reims-ctl keeps downloaded for new VMs: macOS installers, and IPSWs for arm64 \
             VMs. Each is used by the next VM of its macOS; a deleted one is downloaded again \
             when a VM needs it.",
        )
        .dim(),
        Line::from(""),
    ];
    let first_row = lines.len();
    if items.is_empty() {
        lines.push(Line::from("Nothing is downloaded."));
    }
    let name_w = items
        .iter()
        .map(|i| i.name().chars().count())
        .max()
        .unwrap_or(0)
        .min(inner_w.saturating_sub(14).max(12));
    for (n, i) in items.iter().enumerate() {
        let mut name: String = i.name().chars().take(name_w).collect();
        if i.name().chars().count() > name_w {
            name.pop();
            name.push('…');
        }
        let mut text = format!(
            "{} {name:<name_w$}  {:>8}  {}",
            if n == choice { "›" } else { " " },
            crate::downloads::size(i.bytes),
            i.what
        );
        // A row a line, cut where it is too long.
        if text.chars().count() > inner_w {
            text = text.chars().take(inner_w.saturating_sub(1)).collect();
            text.push('…');
        }
        lines.push(if n == choice {
            Line::from(text).style(Style::new().fg(Color::Black).bg(Color::Cyan))
        } else {
            Line::from(text)
        });
    }
    lines.push(Line::from(""));
    let total: u64 = items.iter().map(|i| i.bytes).sum();
    lines.push(
        Line::from(format!(
            "{} in all · d delete · r rescan · Esc close",
            crate::downloads::size(total)
        ))
        .dim(),
    );
    let rows_of = |l: &Line| (l.width().max(1)).div_ceil(inner_w.max(1)) as u16;
    let need: u16 = lines.iter().map(rows_of).sum::<u16>() + 2;
    let height = need.min(area.height);
    let popup = area.centered(Constraint::Length(width), Constraint::Length(height));
    f.render_widget(Clear, popup);
    let block = Block::bordered()
        .title(Line::from(" Downloads ").bold())
        .border_style(Style::new().fg(Color::Cyan));
    let body = block.inner(popup);
    let above: u16 = lines[..first_row + choice.min(items.len().saturating_sub(1))]
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

    /// D shows the downloads; d asks, and Delete deletes the selected one, and the view shows
    /// what is left.
    #[test]
    fn the_downloads_view_deletes_what_is_chosen() {
        let _env = crate::testenv::Env::new("tui-downloads");
        let dir = paths::installers_dir();
        std::fs::create_dir_all(&dir).unwrap();
        let a = "UniversalMac_14.6.1_23G93_Restore.ipsw";
        let b = "UniversalMac_27.2.1_26B131_Restore.ipsw";
        std::fs::write(dir.join(a), b"a").unwrap();
        std::fs::write(dir.join(b), b"b").unwrap();
        let mut app = App::with(Host {
            os: Os::Linux,
            arch: Arch::Aarch64,
        });
        press(
            &mut app,
            [KeyCode::Char('D'), KeyCode::Down, KeyCode::Char('d')],
        );
        assert!(matches!(app.overlay, Some(Overlay::Confirm { .. })));
        // Keep, the default, goes back to the list
        press(&mut app, [KeyCode::Enter]);
        assert!(matches!(app.overlay, Some(Overlay::Downloads { .. })));
        press(
            &mut app,
            [
                KeyCode::Down,
                KeyCode::Char('d'),
                KeyCode::Left,
                KeyCode::Enter,
            ],
        );
        let Some(Overlay::Downloads { items, .. }) = &app.overlay else {
            panic!("the downloads view closed");
        };
        let left: Vec<_> = items.iter().map(|i| i.name()).collect();
        assert_eq!(left, [a]);
        assert!(!dir.join(b).exists());
        assert!(
            app.status
                .as_ref()
                .unwrap()
                .0
                .starts_with(&format!("deleted {b}"))
        );
    }

    /// The arm64 form offers the current macOS from Apple once its catalog is read, each IPSW
    /// downloaded, and an IPSW of the user's own.
    #[test]
    fn the_arm64_form_offers_apples_current_macos_and_those_downloaded() {
        let _env = crate::testenv::Env::new("tui-ipsws");
        let dir = paths::installers_dir();
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("UniversalMac_14.6.1_23G93_Restore.ipsw"), b"a").unwrap();
        let entry = |v: &str, b: &str| reims_restore::catalog::Entry {
            version: v.into(),
            build: b.into(),
            url: format!("https://updates.cdn-apple.com/x/UniversalMac_{v}_{b}_Restore.ipsw"),
            sha1: None,
        };
        let labels = |c| -> Vec<String> { ipsw_options(c).into_iter().map(|o| o.1).collect() };
        assert_eq!(
            labels(None),
            [
                "The current macOS, downloaded from Apple",
                "macOS 14.6.1 (23G93), downloaded",
                "An IPSW of my own"
            ]
        );
        let read = Ok(vec![entry("27.2.1", "26B131")]);
        assert_eq!(
            labels(Some(&read))[0],
            "macOS 27.2.1 (26B131), downloaded from Apple"
        );
        // downloaded already: offered once, as downloaded
        let read = Ok(vec![entry("14.6.1", "23G93")]);
        assert_eq!(
            labels(Some(&read)),
            ["macOS 14.6.1 (23G93), downloaded", "An IPSW of my own"]
        );
        let mut app = App::with(Host {
            os: Os::Mac,
            arch: Arch::Aarch64,
        });
        app.open_create();
        assert_eq!(form(&app).choice("source"), "latest");
        app.message(Msg::Ipsws(Ok(vec![entry("27.2.1", "26B131")])));
        assert_eq!(form(&app).choice("source"), "latest");
    }

    /// On x86-64 Linux, which runs x86-64 VMs and arm64 ones, n asks which to make: x86-64 opens
    /// its form, and arm64 the form of an arm64 VM restored from an IPSW, as on an arm64 host,
    /// whose create makes an arm64 VM. An arm64 Mac, with one profile, opens its form at once.
    #[test]
    fn a_host_of_two_profiles_asks_which_vm_to_create() {
        let _env = crate::testenv::Env::new("tui-profiles");
        let x86 = Host {
            os: Os::Linux,
            arch: Arch::X86_64,
        };
        let mut app = App::with(x86);
        app.macos_versions = Some(Ok(Vec::new()));
        press(&mut app, [KeyCode::Char('n')]);
        let Some(Overlay::Confirm { body, options, .. }) = &app.overlay else {
            panic!("n asks nothing");
        };
        let labels: Vec<_> = options.iter().map(|o| o.0).collect();
        assert_eq!(labels, ["x86-64 macOS", "arm64 macOS", "Cancel"]);
        assert!(
            body.contains("arm64 macOS guest (vmapple and Ambrosia, emulated by TCG)"),
            "{body}"
        );
        press(&mut app, [KeyCode::Esc]);
        assert!(app.overlay.is_none());

        let created = |app: &App| match &app.overlay {
            Some(Overlay::Form(FormKind::Create(p), f)) => Some((*p, f.title.clone())),
            _ => None,
        };
        press(&mut app, [KeyCode::Char('n'), KeyCode::Enter]);
        let x86_form = (Profile::Q35X86_64, "New VM".to_string());
        assert_eq!(created(&app), Some(x86_form));
        press(&mut app, [KeyCode::Esc]);

        // The catalog read, as the form's first opening reads it.
        app.ipsw_catalog = Some(Ok(vec![reims_restore::catalog::Entry {
            version: "27.2.1".into(),
            build: "26B131".into(),
            url: "https://updates.cdn-apple.com/x.ipsw".into(),
            sha1: None,
        }]));
        press(
            &mut app,
            [KeyCode::Char('n'), KeyCode::Right, KeyCode::Enter],
        );
        let arm64_form = (Profile::VmappleArm64, "New arm64 VM".to_string());
        assert_eq!(created(&app), Some(arm64_form));
        assert!(!app.ipsws_asked);
        assert_eq!(form(&app).choice("source"), "latest");
        assert!(!form(&app).is_shown("ipsw"));
        press(&mut app, "arm".chars().map(KeyCode::Char));
        let spec = app.spec_from(Profile::VmappleArm64, form(&app)).unwrap();
        assert_eq!(spec.profile, Some(Profile::VmappleArm64));
        assert_eq!(spec.ipsw.as_deref(), Some("latest"));

        let mut mac = App::with(Host {
            os: Os::Mac,
            arch: Arch::Aarch64,
        });
        mac.ipsws_asked = true;
        press(&mut mac, [KeyCode::Char('n')]);
        assert_eq!(created(&mac).map(|c| c.0), Some(Profile::VmappleArm64));
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
        // x86-64, the first of the host's two
        press(&mut app, [KeyCode::Char('n'), KeyCode::Enter]);
        let name = format!("form-test-{}", std::process::id());
        let disk = std::env::temp_dir().join(&name).join("disk.qcow2");
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

    /// The create form's storage directory typed in as its placeholder shows it, `<name>` and
    /// all, is refused in the form, which says to name the folder or leave the field empty
    #[test]
    fn a_storage_directory_that_still_says_name_is_refused_in_the_form() {
        let _env = crate::testenv::Env::new("tui-placeholder");
        let mut app = App::with(Host {
            os: Os::Linux,
            arch: Arch::X86_64,
        });
        app.macos_versions = Some(Ok(Vec::new()));
        press(&mut app, [KeyCode::Char('n'), KeyCode::Enter]);
        press(&mut app, "mac".chars().map(KeyCode::Char));
        for _ in 0..20 {
            if form(&app).focused_key() == "dir" {
                break;
            }
            press(&mut app, [KeyCode::Down]);
        }
        assert_eq!(form(&app).focused_key(), "dir");
        press(&mut app, "~/vms/<name>".chars().map(KeyCode::Char));
        app.key(KeyEvent::new(KeyCode::Char('s'), KeyModifiers::CONTROL));
        let f = form(&app);
        assert_eq!(f.busy, None);
        assert_eq!(
            f.error.as_deref(),
            Some(
                "the storage directory still says <name>: replace <name> with a name for the \
                 folder, or leave it empty for the default"
            )
        );
    }

    #[test]
    fn a_busy_form_closes_once_its_work_succeeds_and_opens_again_if_it_fails() {
        // The list each work's end reads again is of these VMs, not the user's.
        let _env = crate::testenv::Env::new("tui-busy");
        let mut app = App::with(Host::current());
        let busy = |app: &mut App, id| {
            let mut f = Form::new("t", "save", Vec::new());
            f.busy = Some((id, "working".into()));
            app.overlay = Some(Overlay::Form(FormKind::Create(Profile::Q35X86_64), f));
        };
        let done = |app: &mut App, id, r: Result<&str, &str>| {
            let r = r.map(|s| Said::from(s.to_string())).map_err(String::from);
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
        let stopped = [
            &[Start, Edit, Usb, Disks][..],
            &every,
            &[UpdateOpencore, Delete],
        ]
        .concat();
        assert_eq!(commands(&vm_row("t")), stopped);
        let mut running = vm_row("t");
        running.pid = Some(1);
        let running_menu = [&[View, Stop, PowerOff, Edit, Usb, Disks][..], &every].concat();
        assert_eq!(commands(&running), running_menu);
        // A Mac passes no USB device or disk through.
        let mac = Host {
            os: Os::Mac,
            arch: Arch::Aarch64,
        };
        assert!(
            !vm_menu(&vm_row("t"), false, mac)
                .iter()
                .any(|it| it.0 == Usb || it.0 == Disks)
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
            arch: Arch::X86_64,
        });
        app.reload().unwrap();
        press(&mut app, [KeyCode::Char('p')]);
        assert!(app.overlay.is_none());
        assert_eq!(app.status.as_ref().unwrap().0, crate::host::UNSUPPORTED);
        // An arm64 Mac makes arm64 VMs, which take USB devices on Linux hosts alone.
        let mut app = App::with(Host {
            os: Os::Mac,
            arch: Arch::Aarch64,
        });
        app.reload().unwrap();
        press(&mut app, [KeyCode::Char('p')]);
        assert!(app.overlay.is_none());
        let said = &app.status.as_ref().unwrap().0;
        assert_eq!(said, crate::usb::LINUX_ONLY);

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

    /// P opens a VM's host disks, on a Linux host alone: a disk it was given that is not
    /// attached shows checked, after the host's disks, and Space takes it back, in its settings
    /// as they are now.
    #[test]
    fn the_disks_view_takes_back_a_disk_not_attached() {
        let env = crate::testenv::Env::new("tui-disks");
        let dir = env.root.join("vms/t");
        std::fs::create_dir_all(&dir).unwrap();
        let mut vm = vm_row("t").vm.unwrap();
        vm.dir = dir.clone();
        vm.config.host_disks = vec![crate::vm::HostDiskChoice {
            path: "/dev/disk/by-id/ata-Away_Disk_0001".into(),
            name: "Away Disk, 2 TB".into(),
        }];
        vm.save().unwrap();
        let mut reg = Registry::default();
        reg.add("t", &dir).unwrap();
        reg.save().unwrap();

        let mut app = App::with(Host {
            os: Os::Mac,
            arch: Arch::X86_64,
        });
        app.reload().unwrap();
        press(&mut app, [KeyCode::Char('P')]);
        assert!(app.overlay.is_none());
        assert_eq!(app.status.as_ref().unwrap().0, disks::ONLY);

        let mut app = App::with(LINUX);
        app.reload().unwrap();
        press(&mut app, [KeyCode::Char('P'), KeyCode::End]);
        let Some(Overlay::Disks { found, .. }) = &app.overlay else {
            panic!("no disks view");
        };
        let found = found.clone();
        let rows = app.disks_rows("t", &found);
        let away = rows.len() - 1;
        assert_eq!(rows[away].name(), "Away Disk, 2 TB");
        assert!(rows[away].chosen.is_some() && rows[away].disk.is_none());
        assert_eq!(rows[away].notes(&[]), ["not attached"]);
        let mut terminal =
            ratatui::Terminal::new(ratatui::backend::TestBackend::new(120, 30)).unwrap();
        terminal.draw(|f| app.draw(f)).unwrap();
        let buf = terminal.backend().buffer().clone();
        assert!(find(&buf, buf.area, "[x] Away Disk, 2 TB").is_some());
        assert!(find(&buf, buf.area, "t: host disks").is_some());

        // Edited meanwhile, as by reims-ctl set: the change keeps it.
        let mut now = Vm::load(&dir).unwrap();
        now.config.cpus = 3;
        now.save().unwrap();
        press(&mut app, [KeyCode::Char(' ')]);
        let saved = Vm::load(&dir).unwrap().config;
        assert!(saved.host_disks.is_empty(), "{:?}", saved.host_disks);
        assert_eq!(saved.cpus, 3);
        assert!(
            app.status
                .as_ref()
                .unwrap()
                .0
                .contains("Away Disk, 2 TB is no longer the VM's"),
            "{:?}",
            app.status.as_ref().map(|s| &s.0)
        );
        assert!(matches!(app.overlay, Some(Overlay::Disks { .. })));
        press(&mut app, [KeyCode::Esc]);
        assert!(app.overlay.is_none());
    }

    /// A draw of `app`: whether its VM pane tells that the VM's OpenCore image lacks reims-vgpu's
    /// macOS drivers, and the reads of the image the draw made, and the bytes they read.
    fn draw_lacks(
        terminal: &mut ratatui::Terminal<ratatui::backend::TestBackend>,
        app: &mut App,
    ) -> (bool, (u64, u64)) {
        let (reads, bytes) = opencore::tests::READS.get();
        terminal.draw(|f| app.draw(f)).unwrap();
        let (now_reads, now_bytes) = opencore::tests::READS.get();
        let buf = terminal.backend().buffer();
        let lacks = find(buf, buf.area, "Its OpenCore image lacks").is_some();
        (lacks, (now_reads - reads, now_bytes - bytes))
    }

    /// The VM pane, drawn a few times a second, reads a VM's OpenCore image again only once it
    /// may have changed: unchanged, a draw reads none of it, and the next draw after a rewrite
    /// tells what the image then holds, whether another program rewrote it in place or reims-ctl
    /// made it a new one (U), of the same length and time of last write.
    #[test]
    fn the_vm_pane_reads_the_opencore_image_again_only_once_it_changes() {
        use opencore::tests::{carried_a, enabled, image_loading, on_reims_vgpu, qcow2};
        let env = crate::testenv::Env::new("tui-opencore-reads");
        let dir = env.root.join("vms/t");
        std::fs::create_dir_all(&dir).unwrap();
        // config.plist loading A.kext then B.kext, or B.kext alone, padded to the same length: an
        // image holding the guest's kexts, and one lacking A.kext, of one length.
        let both = enabled("A.kext") + &enabled("B.kext");
        let b_alone = format!("{:<1$}", enabled("B.kext"), both.len());
        let holding = qcow2(&image_loading(&both, carried_a(), &[]));
        let lacking = qcow2(&image_loading(&b_alone, carried_a(), &[]));
        assert_eq!(holding.len(), lacking.len());
        on_reims_vgpu(&dir, &lacking, crate::vm::ImageFormat::Qcow2);
        let image = dir.join("opencore.qcow2");
        // Last written an hour ago, as a VM's image is by the time the pane shows it.
        let an_hour_ago = std::time::SystemTime::now() - Duration::from_secs(3600);
        let written_long_ago = |p: &std::path::Path| {
            let f = std::fs::File::options().write(true).open(p).unwrap();
            f.set_modified(an_hour_ago).unwrap();
        };
        written_long_ago(&image);
        let mut reg = Registry::default();
        reg.add("t", &dir).unwrap();
        reg.save().unwrap();
        let mut app = App::with(Host {
            os: Os::Mac,
            arch: Arch::X86_64,
        });
        app.reload().unwrap();
        let mut terminal =
            ratatui::Terminal::new(ratatui::backend::TestBackend::new(200, 50)).unwrap();
        let read_lacking = draw_lacks(&mut terminal, &mut app);
        assert!(read_lacking.0 && read_lacking.1.0 > 0, "{read_lacking:?}");
        // Unchanged, the next draw reads none of it, and tells the same.
        let kept = draw_lacks(&mut terminal, &mut app);
        assert_eq!(kept, (true, (0, 0)));

        // Rewritten in place by another program, holding the drivers, of the same length.
        std::fs::write(&image, &holding).unwrap();
        let read_holding = draw_lacks(&mut terminal, &mut app);
        assert!(!read_holding.0 && read_holding.1.0 > 0, "{read_holding:?}");
        eprintln!(
            "a draw's reads of the image, and bytes read: {:?} lacking A.kext and {:?} holding \
             the set where it may have changed, {:?} where it has not",
            read_lacking.1, read_holding.1, kept.1
        );
        // Its time of last write set back to the hour before: read again, then kept.
        written_long_ago(&image);
        assert!(!draw_lacks(&mut terminal, &mut app).0);
        assert_eq!(draw_lacks(&mut terminal, &mut app), (false, (0, 0)));

        // reims-ctl makes it a new image (U), here one lacking A.kext, of the same length and
        // time of last write as the image it replaces.
        let made = env.root.join("made.qcow2");
        std::fs::write(&made, &lacking).unwrap();
        written_long_ago(&made);
        let util = crate::testenv::program(
            &env.root.join("reims-util"),
            &format!(
                "#!/bin/sh\nwhile [ \"$1\" != -o ]; do shift; done\ncp -p '{}' \"$2\"\n\
                 echo '{{\"event\":\"done\",\"result\":{{}}}}'\n",
                made.display()
            ),
        );
        env.set("REIMS_CTL_UTIL", &util);
        let mut vm = Registry::load().unwrap().open("t").unwrap();
        opencore::renew(&mut vm, None, &mut |_| {}).unwrap();
        assert!(draw_lacks(&mut terminal, &mut app).0);
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
        // virtio is the next, and SATA again after it.
        press(&mut app, [KeyCode::Right]);
        assert_eq!(form(&app).choice("disk_controller"), "virtio");
        let mut vm = vm_row("t").vm.unwrap();
        apply_edit(&mut vm, form(&app), None).unwrap();
        assert_eq!(
            vm.config.q35.unwrap().disk_controller,
            DiskController::Virtio
        );
        press(&mut app, [KeyCode::Right]);
        assert_eq!(form(&app).choice("disk_controller"), "ahci");
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

    /// A VM's edit form shows capture input on reims-vgpu as the VM's settings have it, and
    /// saves it as toggled; on VMware SVGA it hides it, and the setting stays as it was. An
    /// arm64 VM's form has it beside its CPUs and memory.
    #[test]
    fn the_edit_form_turns_capture_input_off_and_on() {
        let mut row = vm_row("t");
        row.vm.as_mut().unwrap().config.graphics = None;
        let mut arm = vm_row("a");
        let c = &mut arm.vm.as_mut().unwrap().config;
        c.profile = Profile::VmappleArm64;
        c.q35 = None;
        c.vmapple = Some(crate::vm::Vmapple {
            aux: "aux.img".into(),
            ecid: 0x9e35_f01e_dc0c_5536,
        });
        let mut app = test_app(vec![row, arm]);
        let focus = |app: &mut App, key: &str| {
            for _ in 0..form(app).fields.len() {
                if form(app).focused_key() == key {
                    return;
                }
                press(app, [KeyCode::Down]);
            }
            panic!("no field {key}");
        };
        app.open_edit();
        assert!(form(&app).is_shown("capture"));
        assert_eq!(form(&app).toggle("capture"), crate::vm::DEFAULT_CAPTURE);
        focus(&mut app, "capture");
        press(&mut app, [KeyCode::Char('n')]);
        let mut vm = app.rows[0].vm.as_ref().unwrap().clone();
        apply_edit(&mut vm, form(&app), None).unwrap();
        assert!(!vm.config.display.capture);
        press(&mut app, [KeyCode::Char(' ')]);
        apply_edit(&mut vm, form(&app), None).unwrap();
        assert!(vm.config.display.capture);
        press(&mut app, [KeyCode::Char('n')]);
        focus(&mut app, "graphics");
        press(&mut app, [KeyCode::Right]);
        assert_eq!(form(&app).choice("graphics"), "vmware-svga");
        assert!(!form(&app).is_shown("capture"));
        apply_edit(&mut vm, form(&app), None).unwrap();
        assert!(vm.config.display.capture, "a hidden field changes nothing");
        // An arm64 VM's form: its CPUs, its memory and capture input.
        app.overlay = None;
        app.table.select(Some(1));
        app.open_edit();
        let keys: Vec<_> = form(&app).fields.iter().map(|f| f.key).collect();
        assert_eq!(keys, ["cpus", "memory", "capture"]);
        focus(&mut app, "capture");
        press(&mut app, [KeyCode::Char('n')]);
        let mut vm = app.rows[1].vm.as_ref().unwrap().clone();
        apply_edit(&mut vm, form(&app), None).unwrap();
        assert!(!vm.config.display.capture);
    }

    /// A VM's edit form shows reims-vgpu's displays, 1 to 4, and saves the one chosen; on
    /// VMware SVGA, which has one display, it hides them, and the count stays as it was.
    #[test]
    fn the_edit_form_sets_reims_vgpu_s_displays() {
        let mut row = vm_row("t");
        let c = &mut row.vm.as_mut().unwrap().config;
        c.graphics = None;
        c.display.count = 2;
        let mut app = test_app(vec![row]);
        app.open_edit();
        assert!(form(&app).is_shown("displays"));
        assert_eq!(form(&app).choice("displays"), "2");
        let focus = |app: &mut App, key: &str| {
            for _ in 0..form(app).fields.len() {
                if form(app).focused_key() == key {
                    return;
                }
                press(app, [KeyCode::Down]);
            }
            panic!("no field {key}");
        };
        focus(&mut app, "displays");
        press(&mut app, [KeyCode::Right]);
        assert_eq!(form(&app).choice("displays"), "3");
        let mut vm = app.rows[0].vm.as_ref().unwrap().clone();
        apply_edit(&mut vm, form(&app), None).unwrap();
        assert_eq!(vm.config.display.count, 3);
        press(&mut app, [KeyCode::Right]);
        press(&mut app, [KeyCode::Right]);
        assert_eq!(form(&app).choice("displays"), "1");
        apply_edit(&mut vm, form(&app), None).unwrap();
        assert_eq!(vm.config.display.count, 1);
        focus(&mut app, "graphics");
        press(&mut app, [KeyCode::Right]);
        assert_eq!(form(&app).choice("graphics"), "vmware-svga");
        assert!(!form(&app).is_shown("displays"));
        vm.config.display.count = 2;
        apply_edit(&mut vm, form(&app), None).unwrap();
        assert_eq!(vm.config.graphics, Some(Graphics::VmwareSvga));
        assert_eq!(vm.config.display.count, 2, "a hidden field changes nothing");
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

    /// A start's line says the VM's window is open where the start opened one, and otherwise
    /// that it runs without one and which key shows its screen here.
    #[test]
    fn a_start_says_whether_its_window_opened() {
        assert_eq!(
            started_line("t", 42, true),
            "t: running (pid 42); its window is open"
        );
        assert_eq!(
            started_line("t", 42, false),
            "t: running (pid 42) without a window; v shows its screen in this terminal"
        );
    }

    /// A status line that states a VM's state says it again once each read of the VMs' states
    /// finds it another: a start's "running" line says the VM stopped once its QEMU ends by
    /// itself (the guest shut down, or its window was closed), with reims-vgpu's fatal error and
    /// where the log is where QEMU's log since the start holds one; a line that it is not
    /// running says it runs once it is started elsewhere. A line about another VM, an error, and
    /// a VM this interface's work runs for stay as they are.
    #[test]
    fn a_line_stating_a_vms_state_says_it_again_once_the_state_changes() {
        let env = crate::testenv::Env::new("tui-restate");
        let mut reg = Registry::default();
        let mut vms = Vec::new();
        for name in ["t", "u"] {
            let dir = env.root.join("vms").join(name);
            std::fs::create_dir_all(dir.join("logs")).unwrap();
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
            std::fs::write(dir.join(crate::vm::CONFIG_FILE), config).unwrap();
            reg.add(name, &dir).unwrap();
            vms.push(Vm::load(&dir).unwrap());
        }
        reg.save().unwrap();
        let (t, u) = (&vms[0], &vms[1]);
        let mut app = App::with(LINUX);
        let said = |app: &App| app.status.clone().unwrap();
        let select = |app: &mut App, name: &str| {
            let i = app.rows.iter().position(|r| r.name == name);
            app.table.select(i);
        };

        // t runs, as its start says, until its QEMU ends by itself.
        let mut qemu = crate::runtime::test_qemu(&t.pidfile(), 30);
        let pid = qemu.id() as i32;
        app.reload().unwrap();
        let running = format!("t: running (pid {pid}); its window is open");
        let start = Said::stating(running.clone(), "t", Some(pid));
        app.message(Msg::Done("t".into(), 1, Ok(start)));
        app.refresh_state();
        assert_eq!(said(&app), (running, Level::Ok));
        let _ = qemu.kill();
        let _ = qemu.wait();
        app.refresh_state();
        assert_eq!(said(&app), ("t: stopped".into(), Level::Info));

        // Started elsewhere, as by the CLI: the line says it runs, and, said again, it stopped.
        let mut qemu = crate::runtime::test_qemu(&t.pidfile(), 30);
        let pid = qemu.id() as i32;
        app.refresh_state();
        let now = (format!("t: running (pid {pid})"), Level::Info);
        assert_eq!(said(&app), now);
        let _ = qemu.kill();
        let _ = qemu.wait();
        app.reload().unwrap();
        assert_eq!(said(&app), ("t: stopped".into(), Level::Info));

        // While this interface's work runs for t, its end says what happened.
        app.activity.insert("t".into(), "starting".into());
        let mut qemu = crate::runtime::test_qemu(&t.pidfile(), 30);
        app.refresh_state();
        assert_eq!(said(&app).0, "t: stopped");
        app.activity.clear();

        // A line about u stays as t's state changes.
        select(&mut app, "u");
        press(&mut app, [KeyCode::Char('x')]);
        let u_line = ("u is not running".to_string(), Level::Info);
        assert_eq!(said(&app), u_line);
        let _ = qemu.kill();
        let _ = qemu.wait();
        app.refresh_state();
        assert_eq!(said(&app), u_line);

        // A line that u is not running says it runs once it does.
        let mut other = crate::runtime::test_qemu(&u.pidfile(), 30);
        app.refresh_state();
        let now = (format!("u: running (pid {})", other.id()), Level::Info);
        assert_eq!(said(&app), now);
        let _ = other.kill();
        let _ = other.wait();

        // reims-vgpu's fatal error in QEMU's log since the start is named, with where the log
        // is: an error, which stays as the VM's state changes.
        let mut qemu = crate::runtime::test_qemu(&t.pidfile(), 30);
        app.refresh_state();
        select(&mut app, "t");
        press(&mut app, [KeyCode::Char('d')]);
        assert_eq!(said(&app).0, "t is running: stop it first");
        let log = "reims-ctl 0.1.0 started t at 2026-11-24 00:00:00 +0000\n\
                   qemu-system-x86_64 \\\n  -name t\n----\n\
                   reims-vgpu: fatal: display.pipeline.no_library\n";
        std::fs::write(t.qemu_log(), log).unwrap();
        let _ = qemu.kill();
        let _ = qemu.wait();
        app.refresh_state();
        let fatal = format!(
            "t: stopped: reims-vgpu: fatal: display.pipeline.no_library (its QEMU log: {})",
            paths::tilde(&t.qemu_log())
        );
        assert_eq!(said(&app), (fatal.clone(), Level::Error));
        let mut qemu = crate::runtime::test_qemu(&t.pidfile(), 30);
        app.refresh_state();
        let _ = qemu.kill();
        let _ = qemu.wait();
        assert_eq!(said(&app), (fatal, Level::Error));
    }
}
