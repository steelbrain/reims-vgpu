//! The commands. With none, reims-ctl opens its interactive interface (tui.rs).

use crate::create::{self, DiskSource, ImportMode, Spec};
use crate::host::{Host, Profile};
use crate::vm::{self, Registry, Vm, format_mib, parse_size_mib};
use crate::{gpu, install, logs, opencore, paths, qemu, reims_util, runtime, update, view};
use anyhow::{Context, Result, anyhow, bail};
use clap::{Args, Parser, Subcommand};
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(
    name = "reims-ctl",
    version,
    about = "Create, boot and manage macOS VMs on QEMU with reims-vgpu",
    long_about = "Create, boot and manage macOS VMs on QEMU with reims-vgpu.\n\n\
        Run with no command for the interactive interface. Every option takes its value as \
        `--name value` or `--name=value`; yes/no options also take a bare `--name` for yes.",
    after_help = "VMs live in ~/.local/share/reims-ctl/vms unless created elsewhere (--dir); \
        `reims-ctl command NAME` prints the QEMU command a VM runs, for scripts of your own."
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Cmd>,
}

#[derive(Subcommand)]
pub enum Cmd {
    /// List the VMs and whether they run
    List {
        /// Machine-readable output
        #[arg(long)]
        json: bool,
    },
    /// Create a VM
    #[command(after_help = create_examples())]
    Create(Box<CreateArgs>),
    /// Delete a VM (its directory too, unless --keep-files)
    Delete {
        name: String,
        /// Leave the VM's directory where it is
        #[arg(long)]
        keep_files: bool,
        /// Do not ask
        #[arg(long, short)]
        yes: bool,
    },
    /// Start a VM: a window opens for its display
    Start { name: String },
    /// Shut a VM down; if the guest has not shut down when the timeout ends, power it off
    Stop {
        name: String,
        /// Power off at once
        #[arg(long)]
        force: bool,
        /// Seconds to wait for the guest before powering it off; 0 for no limit, waiting until
        /// the guest shuts down (the VM's own setting by default)
        #[arg(long)]
        timeout: Option<u64>,
    },
    /// Show a VM's settings and state
    Show { name: String },
    /// Change a VM's settings (they apply at its next start)
    Set(Box<SetArgs>),
    /// Show a running VM's screen in this terminal, and use its keyboard and mouse from here
    #[command(after_help = VIEW_HELP)]
    View {
        name: String,
        /// The most frames a second
        #[arg(long, default_value_t = 15)]
        fps: u32,
        /// How to draw the screen (by default, the best the terminal answers it can show)
        #[arg(long, value_enum)]
        protocol: Option<view::Protocol>,
    },
    /// Print the QEMU command that runs a VM, as a shell script
    Command {
        name: String,
        /// Write it to this file (executable) instead
        #[arg(long, short)]
        output: Option<PathBuf>,
    },
    /// Give a VM an installer image for its next start; a DMG is converted to a raw
    /// image in the VM's directory
    Insert { name: String, path: PathBuf },
    /// Take the installer out of a VM
    Eject { name: String },
    /// Replace a VM's OpenCore image with one the installed reims-util makes, presenting
    /// the same Mac
    #[command(name = "update-opencore", after_help = UPDATE_OPENCORE_HELP)]
    UpdateOpencore {
        name: String,
        /// OpenCore's boot picker at each start, to pick Recovery or another disk (default: as
        /// the old image has it)
        #[arg(long, num_args = 0..=1, default_missing_value = "true")]
        boot_selection: Option<bool>,
    },
    /// List the GPUs a VM can be given (Linux)
    Gpus,
    /// List this host's USB devices, and give a VM some or take them back (Linux)
    #[command(after_help = usb_help())]
    Usb {
        /// The VM whose devices to show or change (without one, this host's devices)
        name: Option<String>,
        /// Give the VM this device (repeatable)
        #[arg(long, value_name = "DEVICE")]
        add: Vec<String>,
        /// Take this device back from the VM (repeatable)
        #[arg(long, value_name = "DEVICE")]
        remove: Vec<String>,
    },
    /// Print the end of a VM's QEMU log, or its serial console's
    Logs {
        name: String,
        /// Lines to print
        #[arg(long, short = 'n', default_value_t = 40)]
        lines: usize,
        /// The serial console's log instead
        #[arg(long)]
        serial: bool,
    },
    /// Zip a VM's logs, settings and the host's details into Downloads, for a bug report
    Report { name: String },
    /// Update reims-ctl and its QEMU from the checkout it was installed from
    #[command(after_help = UPDATE_HELP)]
    Update {
        /// Switch the checkout to this channel first; later updates stay on it
        #[arg(long, value_enum)]
        channel: Option<update::Channel>,
    },
    /// Check the host and the install
    Doctor,
    /// The interactive interface (what reims-ctl runs with no command)
    Tui,
    #[command(name = "__vulkan-probe", hide = true)]
    VulkanProbe,
    #[command(name = "__record-install", hide = true)]
    RecordInstall {
        #[arg(long)]
        repo: PathBuf,
        #[arg(long)]
        qemu: PathBuf,
        #[arg(long, default_value = "")]
        commit: String,
    },
}

const VIEW_HELP: &str = "\
For a VM on a host with no screen, to set macOS up before it has screen sharing or remote \
login. The screen is drawn with the terminal's graphics (kitty's, iTerm2's or sixel; half-block \
characters where it has none), over SSH as well, and the terminal's keys, mouse and pasted text \
go to the guest as a US keyboard's and a tablet's.

Ctrl-] disconnects; the VM keeps running. Alt+key is Cmd+key. The Cmd, Opt, Ctrl and Shift \
buttons on the top row hold that key for the next key or click.";

const UPDATE_HELP: &str = "\
After an update, the new reims-ctl's first run (its restart from the interface, or the next \
command) updates the VMs an older reims-ctl made, once, with a line for each VM. A VM whose \
settings name no graphics, as those a reims-ctl from before reims-vgpu's own GPU device made, \
runs that device, and one whose settings name apple-gfx, Apple's paravirtualized GPU, which \
reims-vgpu no longer has, moves to it. Either's OpenCore image is made again for the device, \
presenting the same Mac, with its macOS drivers and without the ported drivers older images \
held, and the old image is kept beside it as <image>.pre-migration-1, or -3 for one that was on \
apple-gfx. A VM that is running then is updated at its next start. One that can't be updated \
stays as it was, and its line says why. A VM on VMware SVGA keeps it; where its OpenCore image \
still loads the ported drivers, the image is made again without them, and the old one is kept \
as <image>.pre-migration-2.";

const UPDATE_OPENCORE_HELP: &str = "\
The new image is made for the VM's graphics. On reims-vgpu it holds the macOS drivers of \
reims-vgpu's own GPU device that this reims-ctl carries. A start makes the image again where it \
holds other builds of them, as one an older reims-ctl made does after an update of reims-ctl; an \
image that lacks them, as one of your own may, stays as it is, which reims-ctl start warns of: \
make it again here. A change of graphics with reims-ctl set or the interface's edit form makes it \
again where the VM is stopped, and a start on VMware SVGA makes one without the drivers where the \
image holds them. The new image presents the same Mac (its model, \
serial number, MLB, UUID and ROM, read from the old image's config.plist), so macOS sees no new \
machine, and the VM keeps its OVMF variables, which hold its boot entries. Its boot selection, \
OpenCore's picker, is carried over unless --boot-selection gives it; hidden, OpenCore boots \
macOS's startup disk at once. Whatever else was changed in the old image is not carried over. The \
VM must be stopped; reims-ctl update installs the reims-util this reims-ctl needs.";

/// `usb --help`'s account of passthrough, with the udev rule the install adds.
fn usb_help() -> String {
    format!(
        "\
DEVICE is a device's number in the list, its ID (1050:0407), its port (3-2.1) or a word of its \
name; --add takes the ID of a device that isn't plugged in, too.

A device a VM has is the guest's while the VM runs: QEMU takes it from this host's drivers, and \
macOS in the guest drives it, as a Mac drives a device plugged into it. It comes back to this \
host when the VM stops, or when you take it back. While the VM runs, --add and --remove attach \
and detach it at once; either way the VM keeps it for its next start. A VM takes up to {max} \
devices.

A device is known by its vendor and product IDs, so the VM has it in any port. Where this host \
has two devices with the same IDs, the one you give is known by its port too, and the VM has it \
only in that port. A device that isn't plugged in joins the guest when it is, and one unplugged \
comes back when it is plugged in again.

QEMU opens the device's file in /dev/bus/usb, which only root may write unless a udev rule \
says otherwise. scripts/install-ctl.sh, which reims-ctl update runs, adds {file}, which gives \
USB devices other than hubs to the kvm group and to the user at this host's screen:

  {rule}

Over SSH, without a session at the screen, you need to be in the kvm group (sudo usermod -aG \
kvm $USER, then log in again).",
        max = vm::USB_DEVICES_MAX,
        file = crate::usb::UDEV_RULE_FILE,
        rule = crate::usb::UDEV_RULE,
    )
}

/// `--flag`, `--flag=true`, `--flag false`.
fn bool_arg() -> clap::Arg {
    clap::Arg::new("x")
        .num_args(0..=1)
        .default_missing_value("true")
}

/// `create --help`'s examples, under a header styled as clap styles its own.
fn create_examples() -> String {
    let h = *clap::builder::Styles::default().get_header();
    format!("{h}Examples:{h:#}{CREATE_EXAMPLES}")
}

/// `e`, followed by the examples: for a create that was not given enough to make a VM.
fn with_create_examples(e: impl std::fmt::Display) -> anyhow::Error {
    anyhow!("{e}\n\nExamples:{CREATE_EXAMPLES}")
}

const CREATE_EXAMPLES: &str = r"
  An x86-64 guest on Linux:

    # Download macOS Tahoe from Apple, make the VM's OpenCore image and a new
    # 128G disk, and start it: the installer boots into macOS Recovery. Once
    # macOS is installed, reims-ctl eject mac takes the installer out
    reims-ctl create mac --macos tahoe --start

    # One exact release, which installs with no network (13 to 20 GB)
    reims-ctl create mac --macos 26.6.2 --offline --start

    # An installer of your own, such as a recovery image (BaseSystem.dmg)
    reims-ctl create mac --installer ~/Downloads/BaseSystem.dmg

    # A disk that already has macOS installed (copied; see --disk-mode): the
    # new OpenCore image presents the Mac of the one beside it, opencore.qcow2,
    # as a VM's directory holds them, so macOS sees no new Mac. A copy of the
    # disk of a VM reims-ctl has is a Mac of its own, as it runs beside it
    reims-ctl create mac --disk ~/vms/old/disk.qcow2

    # Or with the OpenCore image it booted from, copied, which keeps its Mac too
    reims-ctl create mac --disk ~/vms/mac.qcow2 --opencore ~/vms/OpenCore.qcow2";

#[derive(Args)]
pub struct CreateArgs {
    /// The VM's name (or --name)
    #[arg(value_name = "NAME")]
    pub name_pos: Option<String>,
    /// The VM's name
    #[arg(long)]
    pub name: Option<String>,
    /// Its directory (default ~/.local/share/reims-ctl/vms/NAME)
    #[arg(long)]
    pub dir: Option<PathBuf>,
    /// vCPUs
    #[arg(long)]
    pub cpus: Option<u32>,
    /// RAM, such as 8G
    #[arg(long)]
    pub memory: Option<String>,
    /// A new disk of this size, such as 128G
    #[arg(long)]
    pub disk_size: Option<String>,
    /// An existing disk image
    #[arg(long, conflicts_with = "disk_size")]
    pub disk: Option<PathBuf>,
    /// How an existing disk is used: copy (instant, a reflink, where the disk and the VM's
    /// directory are on one file system that makes them, else a full copy) or in-place
    #[arg(long, default_value = "copy")]
    pub disk_mode: String,
    /// Download this macOS from Apple to install: a name (tahoe), a number (26) or latest; with
    /// --offline, also a release (26.6.2) or a build (25G83)
    #[arg(long, value_name = "VERSION", conflicts_with = "installer")]
    pub macos: Option<String>,
    /// With --macos, an offline installer of one release (13 to 20 GB), which installs with no
    /// network, rather than Apple's recovery image (about 1 GB), which downloads the newest
    /// release as it installs
    #[arg(long, requires = "macos")]
    pub offline: bool,
    /// An installer image of your own, attached as a disk: a raw image, such as a recovery
    /// image's BaseSystem.img, or a DMG, converted to a raw image in the VM's directory
    #[arg(long)]
    pub installer: Option<PathBuf>,
    /// An OpenCore image of your own to copy, with OVMF_CODE.fd and OVMF_VARS.fd beside it
    /// (default: a new one made for the VM, with OVMF, presenting the Mac of the image beside
    /// --disk, opencore.qcow2, where there is one and --disk is no copy of the disk of a VM
    /// reims-ctl has, else a Mac identity of its own)
    #[arg(long)]
    pub opencore: Option<PathBuf>,
    /// The network card's MAC, such as 52:54:00:12:34:56 (default: a random one)
    #[arg(long)]
    pub mac: Option<String>,
    /// reims-vgpu (the default) or vmware-svga (see reims-ctl set --help)
    #[arg(long)]
    pub graphics: Option<String>,
    /// The size reims-vgpu's window opens at, WIDTHxHEIGHT in the desktop's points, which the
    /// guest follows; it starts a HiDPI desktop by itself in a window at least 3840 pixels
    /// wide, and offers one in its Displays settings from 2880
    #[arg(long)]
    pub resolution: Option<String>,
    /// A borderless window that fills its screen
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    pub borderless: Option<bool>,
    /// The GPU: auto, a number or name from `reims-ctl gpus`, or a Vulkan ICD JSON file
    #[arg(long)]
    pub gpu: Option<String>,
    /// Keep the library's log and the serial console
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    pub logs: Option<bool>,
    /// A virtio sound device on the host's sound, with a microphone, and the HDA controller
    /// OpenCore's boot chime plays on
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    pub audio: Option<bool>,
    /// Forward this port on 127.0.0.1 to the guest's SSH port; 0 for none
    #[arg(long)]
    pub ssh_port: Option<u16>,
    /// Start it once made
    #[arg(long)]
    pub start: bool,
}

#[derive(Args)]
pub struct SetArgs {
    pub name: String,
    #[arg(long)]
    pub cpus: Option<u32>,
    /// RAM, such as 8G
    #[arg(long)]
    pub memory: Option<String>,
    /// reims-vgpu (the default), its own GPU device, which the guest drives with the kexts
    /// reims-ctl puts in its OpenCore image (a change of graphics makes the image again for
    /// them, as update-opencore does, where the VM is stopped), on one display in a window whose
    /// size the guest follows; or vmware-svga, QEMU's VMware SVGA: one display with no GPU
    /// acceleration, in QEMU's own window
    #[arg(long)]
    pub graphics: Option<String>,
    /// The system disk's controller: ahci (SATA), or nvme once macOS is installed (an installer
    /// may find no disk on NVMe). Once it is installed, eject the installer before a
    /// move, or OpenCore may start it
    #[arg(long)]
    pub disk_controller: Option<String>,
    /// The size reims-vgpu's window opens at, WIDTHxHEIGHT in the desktop's points
    #[arg(long)]
    pub resolution: Option<String>,
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    pub borderless: Option<bool>,
    /// auto, a number or name from `reims-ctl gpus`, or a Vulkan ICD JSON file
    #[arg(long)]
    pub gpu: Option<String>,
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    pub logs: Option<bool>,
    /// REIMS_VGPU_LOG's filter, such as info or warn,reims_vgpu_vulkan::core=debug
    #[arg(long)]
    pub log_level: Option<String>,
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    pub audio: Option<bool>,
    /// pipewire, pa, alsa, none, or auto
    #[arg(long)]
    pub audio_backend: Option<String>,
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    pub network: Option<bool>,
    /// 0 for none
    #[arg(long)]
    pub ssh_port: Option<u16>,
    /// The network card's MAC, such as 52:54:00:12:34:56
    #[arg(long)]
    pub mac: Option<String>,
    /// Seconds stop waits for the guest to shut down before it powers the VM off; 0 for no
    /// limit: stop waits until the guest shuts down, and only stop --force powers it off
    #[arg(long)]
    pub stop_timeout: Option<u64>,
    /// More QEMU arguments, as one shell-quoted string; empty for none
    #[arg(long, allow_hyphen_values = true)]
    pub extra_args: Option<String>,
    /// KEY=VALUE for QEMU's environment, KEY letters, digits and _, not starting with a digit;
    /// KEY= removes it (repeatable)
    #[arg(long = "env")]
    pub env: Vec<String>,
}

pub fn main(cli: Cli) -> Result<i32> {
    let host = Host::current();
    let cmd = match cli.command {
        None | Some(Cmd::Tui) => return crate::tui::run().map(|_| 0),
        Some(c) => c,
    };
    // The first run after an update brings the VMs an older reims-ctl made up to date, telling
    // of it on stderr, so a command's output stays as it is; reims-ctl's own commands, which the
    // installer and reims-ctl run, leave it to the user's.
    if !matches!(cmd, Cmd::VulkanProbe | Cmd::RecordInstall { .. }) {
        crate::migrate::run(&mut |s| eprintln!("{s}"));
    }
    match cmd {
        Cmd::List { json } => list(json)?,
        Cmd::Create(a) => create_cmd(*a, host)?,
        Cmd::Delete {
            name,
            keep_files,
            yes,
        } => delete(&name, keep_files, yes)?,
        Cmd::Start { name } => {
            let mut vm = Registry::load()?.open(&name)?;
            let first = runtime::prepare(&mut vm, &mut |p| println!("{name}: {p}"));
            for w in first
                .into_iter()
                .chain(runtime::warnings(&vm, host))
                .chain(crate::usb::problems(&vm))
            {
                eprintln!("warning: {w}");
            }
            let pid = runtime::start(&vm, host)?;
            println!("{name}: running, pid {pid}");
        }
        Cmd::Stop {
            name,
            force,
            timeout,
        } => {
            let vm = Registry::load()?.open(&name)?;
            let t = timeout.map_or(vm.config.stop_timeout(), vm::stop_timeout);
            runtime::stop(&vm, force, t, |s| println!("{name}: {s}"))?;
        }
        Cmd::Show { name } => show(&Registry::load()?.open(&name)?, host),
        Cmd::Set(a) => set(*a, host)?,
        Cmd::View {
            name,
            fps,
            protocol,
        } => {
            let vm = Registry::load()?.open(&name)?;
            let opts = view::Options { fps, protocol };
            if let Some(m) = view::run(&vm, &opts)? {
                println!("{m}");
            }
        }
        Cmd::Command { name, output } => {
            let vm = Registry::load()?.open(&name)?;
            let script =
                runtime::invocation_for_print(&vm, host)?.shell_script(&script_comment(&vm));
            match output {
                Some(p) => {
                    std::fs::write(&p, script)?;
                    use std::os::unix::fs::PermissionsExt;
                    std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755))?;
                    println!("wrote {}", p.display());
                }
                None => print!("{script}"),
            }
        }
        Cmd::Insert { name, path } => {
            let mut vm = Registry::load()?.open(&name)?;
            let path = paths::absolute(&path.to_string_lossy());
            println!(
                "{}",
                set_installer(&mut vm, Some(path), &mut |p| println!("{name}: {p}"))?
            );
        }
        Cmd::Eject { name } => {
            let mut vm = Registry::load()?.open(&name)?;
            println!("{}", set_installer(&mut vm, None, &mut |_| {})?);
        }
        Cmd::UpdateOpencore {
            name,
            boot_selection,
        } => {
            let mut vm = Registry::load()?.open(&name)?;
            println!(
                "{}",
                opencore::renew(&mut vm, boot_selection, &mut |p| println!("{name}: {p}"))?
            );
        }
        Cmd::Gpus => gpus(host),
        Cmd::Usb { name, add, remove } => usb(host, name.as_deref(), &add, &remove)?,
        Cmd::Logs {
            name,
            lines,
            serial,
        } => {
            let vm = Registry::load()?.open(&name)?;
            let f = if serial {
                vm.serial_log()
            } else {
                vm.qemu_log()
            };
            if !f.exists() {
                bail!("{} does not exist yet", f.display());
            }
            println!("{}", runtime::tail(&f, lines));
        }
        Cmd::Report { name } => {
            let vm = Registry::load()?.open(&name)?;
            let p = logs::bundle(&vm, host)?;
            println!("wrote {}", p.display());
            if !vm.config.logs.enabled {
                println!(
                    "logs were off for {name}: turn them on (reims-ctl set {name} --logs), reproduce, and report again for the library's log"
                );
            }
        }
        Cmd::Update { channel } => update::update(channel)?,
        Cmd::Doctor => return doctor(host),
        Cmd::Tui => unreachable!(),
        Cmd::VulkanProbe => return Ok(gpu::probe_main()),
        Cmd::RecordInstall { repo, qemu, commit } => {
            install::Install {
                repo: repo.clone(),
                commit,
                qemu,
                installed_at: runtime::now_stamp("%Y-%m-%d %H:%M:%S %z"),
            }
            .save()?;
            // What makes a new x86-64 VM's OpenCore image and installer, and on any host one
            // already beside reims-ctl, moved to the commit reims-ctl pins (reims_util.rs).
            if host.default_profile() == Some(Profile::Q35X86_64) || reims_util::beside().is_some()
            {
                println!(
                    "==> Installing reims-util at {} (cargo install)",
                    &reims_util::REV[..7]
                );
                reims_util::install(&repo)?;
            }
        }
    }
    let _ = bool_arg;
    Ok(0)
}

pub fn script_comment(vm: &Vm) -> String {
    let screen = match vm.config.graphics() {
        vm::Graphics::ReimsVgpu => "Its window opens under -display none",
        vm::Graphics::VmwareSvga if crate::host::has_screen(&vm.config.extra_env) => {
            "Its window is QEMU's own, under -display gtk"
        }
        vm::Graphics::VmwareSvga => "With no screen here, it runs headless under -display none",
    };
    format!(
        "{}: the QEMU command reims-ctl {} runs for this VM.\n{screen}; QMP is at the -qmp socket.",
        vm.name(),
        env!("CARGO_PKG_VERSION")
    )
}

fn list(json: bool) -> Result<()> {
    let reg = Registry::load()?;
    if json {
        let mut out = Vec::new();
        for e in &reg.vms {
            let (state, pid) = match Vm::load(&e.dir) {
                Ok(vm) => match runtime::pid(&vm) {
                    Some(p) => ("running", Some(p)),
                    None => ("stopped", None),
                },
                Err(_) => ("broken", None),
            };
            out.push(
                serde_json::json!({ "name": e.name, "dir": e.dir, "state": state, "pid": pid }),
            );
        }
        println!("{}", serde_json::to_string_pretty(&out)?);
        return Ok(());
    }
    if reg.vms.is_empty() {
        println!("No VMs yet: reims-ctl create, or reims-ctl for the interactive interface.");
        return Ok(());
    }
    println!(
        "{:<20} {:<9} {:>5} {:>7} {:<11} DIR",
        "NAME", "STATE", "CPUS", "MEMORY", "GRAPHICS"
    );
    for e in &reg.vms {
        match Vm::load(&e.dir) {
            Ok(vm) => {
                let c = &vm.config;
                let state = if runtime::running(&vm) {
                    "running"
                } else {
                    "stopped"
                };
                println!(
                    "{:<20} {:<9} {:>5} {:>7} {:<11} {}",
                    e.name,
                    state,
                    c.cpus,
                    format_mib(c.memory_mib),
                    c.graphics().name(),
                    paths::tilde(&e.dir)
                );
            }
            Err(err) => println!("{:<20} {:<9} {err:#}", e.name, "broken"),
        }
    }
    Ok(())
}

pub fn resolve_gpu(arg: &str) -> Result<vm::Gpu> {
    let a = arg.trim();
    if a.is_empty() || a == "auto" || a == "default" {
        return Ok(vm::Gpu::default());
    }
    if a.ends_with(".json") && std::path::Path::new(a).is_file() {
        return Ok(vm::Gpu {
            driver_file: Some(paths::absolute(a)),
            device_id: None,
            name: None,
        });
    }
    let list = gpu::list();
    if let Ok(n) = a.parse::<usize>()
        && n >= 1
        && n <= list.len()
    {
        return Ok(list[n - 1].to_config());
    }
    let low = a.to_ascii_lowercase();
    let hits: Vec<&gpu::GpuInfo> = list
        .iter()
        .filter(|g| g.name.to_ascii_lowercase().contains(&low))
        .collect();
    match hits.len() {
        1 => Ok(hits[0].to_config()),
        0 => bail!("no GPU matches {a:?}: reims-ctl gpus lists them"),
        _ => bail!("{a:?} matches several GPUs: give its number from reims-ctl gpus"),
    }
}

fn create_cmd(a: CreateArgs, host: Host) -> Result<()> {
    let name = a
        .name
        .clone()
        .or(a.name_pos.clone())
        .ok_or_else(|| with_create_examples("give the VM a name: reims-ctl create NAME"))?;
    // Nothing a VM is made from: whatever create says is missing, the examples show the ways.
    let source =
        a.macos.is_some() || a.installer.is_some() || a.disk.is_some() || a.opencore.is_some();
    let mut s = Spec::new(&name);
    s.dir = a
        .dir
        .as_ref()
        .map(|d| paths::absolute(&d.to_string_lossy()));
    s.cpus = a.cpus;
    s.memory_mib = a.memory.as_deref().map(parse_size_mib).transpose()?;
    s.disk = match (&a.disk_size, &a.disk) {
        (Some(sz), _) => Some(DiskSource::New {
            size_mib: parse_size_mib(sz)?,
        }),
        (None, Some(p)) => Some(DiskSource::Existing {
            path: paths::absolute(&p.to_string_lossy()),
            mode: ImportMode::parse(&a.disk_mode)?,
        }),
        (None, None) => Some(DiskSource::New {
            size_mib: 128 * 1024,
        }),
    };
    let abs = |p: &Option<PathBuf>| p.as_ref().map(|p| paths::absolute(&p.to_string_lossy()));
    s.macos = a.macos.clone().map(|version| reims_util::Download {
        version,
        offline: a.offline,
    });
    s.opencore = abs(&a.opencore);
    s.installer = abs(&a.installer);
    s.mac = a.mac.clone();
    s.graphics = a.graphics.as_deref().map(vm::Graphics::parse).transpose()?;
    if let Some(r) = &a.resolution {
        s.display.resolution = r.clone();
    }
    s.display.borderless = a.borderless.unwrap_or(false);
    s.display.check()?;
    if let Some(g) = &a.gpu {
        s.gpu = resolve_gpu(g)?;
    }
    s.logs = a.logs.unwrap_or(false);
    s.audio = a.audio.unwrap_or(true);
    s.ssh_port = a.ssh_port;
    let vm = create::create(s, host, &mut |p| println!("{name}: {p}")).map_err(|e| {
        if source {
            e
        } else {
            with_create_examples(format!("{e:#}"))
        }
    })?;
    println!("created {name} in {}", paths::tilde(&vm.dir));
    if let Some(steps) = runtime::install_steps(&vm) {
        println!("{steps}");
    }
    if a.start {
        let mut vm = vm;
        for w in runtime::prepare(&mut vm, &mut |p| println!("{name}: {p}")) {
            eprintln!("warning: {w}");
        }
        let pid = runtime::start(&vm, host)?;
        println!("{name}: running, pid {pid}");
    }
    Ok(())
}

fn confirm(question: &str) -> bool {
    print!("{question} [y/N] ");
    let _ = std::io::stdout().flush();
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).is_ok() && matches!(line.trim(), "y" | "Y" | "yes")
}

/// On a terminal, offers to write 0 to MGLRU's thrashing guard, `file`, with sudo; then how to
/// keep it at 0, or to turn it off where it is not. Without a terminal it only tells.
fn offer_min_ttl_off(file: &Path) -> String {
    use std::io::IsTerminal;
    let terminal = std::io::stdin().is_terminal() && std::io::stdout().is_terminal();
    if terminal
        && confirm(&format!(
            "     Write 0 to {} now with sudo?",
            file.display()
        ))
    {
        let wrote = std::process::Command::new("sudo")
            .arg("tee")
            .arg(file)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::null())
            .spawn()
            .and_then(|mut c| {
                c.stdin.take().expect("piped").write_all(b"0\n")?;
                c.wait()
            });
        if let Err(e) = &wrote {
            eprintln!("     sudo didn't run: {e}");
        }
    }
    runtime::min_ttl_fix(runtime::min_ttl_ms(file) == Some(0))
}

pub fn delete_vm(name: &str, keep_files: bool) -> Result<String> {
    let mut reg = Registry::load()?;
    let entry = reg
        .find(name)
        .cloned()
        .with_context(|| format!("no VM named {name}"))?;
    // From its pid file alone: a VM whose vm.toml does not read (a typo, or a downgrade's) may
    // still run.
    if let Some(p) = runtime::pid_at(&vm::pidfile_in(&entry.dir)) {
        bail!("{name} is running (pid {p}): stop it first");
    }
    if !keep_files
        && entry.dir.join(vm::CONFIG_FILE).is_file()
        && let Some((other, file)) = used_inside(&reg, name, &entry.dir)
    {
        bail!(
            "{} also holds {other}'s files ({}), which deleting {name} with its files would \
             delete too: delete {other} first, or delete {name} and keep its files",
            paths::tilde(&entry.dir),
            paths::tilde(&file)
        );
    }
    reg.remove(name);
    reg.save()?;
    if keep_files {
        return Ok(format!(
            "{name} deleted; its files stay in {}",
            paths::tilde(&entry.dir)
        ));
    }
    if entry.dir.join(vm::CONFIG_FILE).is_file() {
        std::fs::remove_dir_all(&entry.dir)
            .with_context(|| format!("remove {}", entry.dir.display()))?;
        Ok(format!("{name} deleted, with {}", paths::tilde(&entry.dir)))
    } else {
        Ok(format!(
            "{name} deleted; {} holds no vm.toml, so it stays",
            paths::tilde(&entry.dir)
        ))
    }
}

/// A VM other than `name` that `dir` holds, or one of whose files it holds (a disk used in
/// place, say): the VM, and its directory or that file.
fn used_inside(reg: &Registry, name: &str, dir: &Path) -> Option<(String, PathBuf)> {
    let real = |p: &Path| std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
    let dir = real(dir);
    reg.vms.iter().filter(|e| e.name != name).find_map(|e| {
        let files = Vm::load(&e.dir).map(|vm| vm.files()).unwrap_or_default();
        std::iter::once(e.dir.clone())
            .chain(files)
            .find(|p| real(p).starts_with(&dir))
            .map(|p| (e.name.clone(), p))
    })
}

fn delete(name: &str, keep_files: bool, yes: bool) -> Result<()> {
    let reg = Registry::load()?;
    let entry = reg
        .find(name)
        .cloned()
        .with_context(|| format!("no VM named {name}"))?;
    if !yes {
        let what = if keep_files {
            "keeping its files".to_string()
        } else {
            format!("and {} with its disks", paths::tilde(&entry.dir))
        };
        if !confirm(&format!("Delete {name}, {what}?")) {
            println!("nothing deleted");
            return Ok(());
        }
    }
    println!("{}", delete_vm(name, keep_files)?);
    Ok(())
}

/// Lines describing a VM, for `show` and the interface's details.
pub fn describe(vm: &Vm, host: Host) -> Vec<(String, String)> {
    let c = &vm.config;
    let mut v: Vec<(String, String)> = Vec::new();
    let mut add = |k: &str, val: String| v.push((k.into(), val));
    add(
        "state",
        match runtime::pid(vm) {
            Some(p) => format!("running (pid {p})"),
            None => "stopped".into(),
        },
    );
    add("kind", c.profile.describe().into());
    add("directory", paths::tilde(&vm.dir));
    add("cpus", c.cpus.to_string());
    add("memory", format_mib(c.memory_mib));
    // A disk in the VM's directory by its file name: the directory is the line above.
    let disk = vm.disk();
    let disk_name = match disk.strip_prefix(&vm.dir) {
        Ok(rel) => rel.display().to_string(),
        Err(_) => paths::tilde(&disk),
    };
    match &c.q35 {
        Some(q) => add(
            "disk",
            format!(
                "{disk_name} ({}, {})",
                c.disk_format.as_str(),
                q.disk_controller.describe()
            ),
        ),
        None => add("disk", format!("{disk_name} ({})", c.disk_format.as_str())),
    }
    let graphics = c.graphics();
    match graphics {
        vm::Graphics::VmwareSvga => add(
            "graphics",
            "VMware SVGA, no GPU acceleration, in QEMU's window".into(),
        ),
        g => add("graphics", g.describe().into()),
    }
    // reims-vgpu's one display follows its window: of the settings, only the window's first size
    // and borderless reach it.
    if graphics == vm::Graphics::ReimsVgpu {
        add(
            "display",
            if c.display.borderless {
                "one, filling its screen, borderless".into()
            } else {
                let (w, h) = c.display.window_size().unwrap_or((1920, 1080));
                format!("one, in a window opening at {w}x{h}, whose size the guest follows")
            },
        );
    }
    if graphics.library() && c.profile.gpu_choice(host) {
        add(
            "gpu",
            match (&c.gpu.name, &c.gpu.driver_file) {
                (Some(n), _) => n.clone(),
                (None, Some(f)) => paths::tilde(f),
                _ => "automatic (discrete first)".into(),
            },
        );
    }
    if let Some(q) = &c.q35 {
        add("opencore", paths::tilde(&vm.path(&q.opencore)));
        add(
            "installer",
            match &q.installer {
                Some(i) => paths::tilde(&vm.path(i)),
                None => "none".into(),
            },
        );
        add("cpu model", q.cpu_model.clone());
    }
    add(
        "audio",
        if c.audio.enabled {
            c.audio.backend.clone().unwrap_or_else(|| "on".into())
        } else {
            "off".into()
        },
    );
    add(
        "network",
        if !c.network.enabled {
            "off".into()
        } else {
            match c.network.ssh_port {
                Some(p) => format!("user, mac {}, ssh on 127.0.0.1:{p}", c.network.mac),
                None => format!("user, mac {}", c.network.mac),
            }
        },
    );
    if !c.usb.is_empty() {
        add(
            "usb",
            c.usb
                .iter()
                .map(|u| match u.name.as_str() {
                    "" => u.id.clone(),
                    n => format!("{n} ({})", u.id),
                })
                .collect::<Vec<_>>()
                .join(", "),
        );
    }
    add(
        "logs",
        if c.logs.enabled {
            format!("on ({})", c.logs.level)
        } else {
            "off".into()
        },
    );
    add(
        "stop timeout",
        match c.stop_timeout() {
            Some(t) => format!("{} s", t.as_secs()),
            None => "none: stop waits until the guest shuts down".into(),
        },
    );
    if !c.extra_args.is_empty() {
        add(
            "extra args",
            c.extra_args
                .iter()
                .map(|a| qemu::quote(a))
                .collect::<Vec<_>>()
                .join(" "),
        );
    }
    for (k, val) in &c.extra_env {
        add("env", format!("{k}={val}"));
    }
    v
}

fn show(vm: &Vm, host: Host) {
    println!("{}", vm.name());
    for (k, v) in describe(vm, host) {
        println!("  {k:<13} {v}");
    }
}

/// Sets the installer, for the VM's next start. Returns what happened. A DMG is converted into
/// the VM's directory first (`create::installer_media`), and the VM's copy of the installer it
/// replaces is deleted. The installer is set in the VM's settings as they are once that is
/// done, which `vm` becomes: a DMG takes seconds to convert, in which others may be saved (the
/// interface's edit form or logs toggle, reims-ctl set).
pub fn set_installer(
    vm: &mut Vm,
    path: Option<PathBuf>,
    progress: &mut dyn FnMut(&str),
) -> Result<String> {
    let running = runtime::running(vm);
    let name = vm.name().to_string();
    let dir = vm.dir.clone();
    let q = vm
        .config
        .q35
        .as_ref()
        .context("only x86-64 VMs take installer media")?;
    if let Some(p) = &path
        && !p.is_file()
    {
        bail!("{} does not exist", p.display());
    }
    // The VM's copy, named again by its path (the TUI shows it so), stays as it is. Any other
    // installer is the user's, though written relative to the VM's directory by hand.
    let copy = |old: &Option<PathBuf>| {
        old.as_ref()
            .filter(|o| *o == Path::new(create::INSTALLER_COPY))
            .map(|o| dir.join(o))
    };
    let installer = match &path {
        Some(p) if copy(&q.installer).is_some_and(|c| same_file(&c, p)) => q.installer.clone(),
        Some(p) => Some(create::installer_media(p, &dir, progress)?),
        None => None,
    };
    let mut now = Vm::load(&dir)?;
    let q = now
        .config
        .q35
        .as_mut()
        .context("only x86-64 VMs take installer media")?;
    let old = std::mem::replace(&mut q.installer, installer.clone());
    now.save()?;
    *vm = now;
    if let Some(c) = copy(&old)
        && installer != old
    {
        let _ = std::fs::remove_file(c);
    }
    Ok(match (&path, running) {
        (Some(p), false) => format!("{name}: {} attached for the next start", paths::tilde(p)),
        (Some(_), true) => format!("{name}: the installer changes at the next start"),
        (None, false) => format!("{name}: installer removed"),
        (None, true) => format!("{name}: the installer goes at the next start"),
    })
}

/// Whether `a` and `b` name the same existing file.
fn same_file(a: &Path, b: &Path) -> bool {
    matches!(
        (std::fs::canonicalize(a), std::fs::canonicalize(b)),
        (Ok(a), Ok(b)) if a == b
    )
}

fn set(a: SetArgs, host: Host) -> Result<()> {
    let mut vm = Registry::load()?.open(&a.name)?;
    let c = &mut vm.config;
    if let Some(n) = a.cpus {
        c.cpus = n;
    }
    if let Some(m) = &a.memory {
        c.memory_mib = parse_size_mib(m)?;
    }
    if let Some(g) = &a.graphics {
        c.graphics = Some(vm::Graphics::parse(g)?);
    }
    if let Some(d) = &a.disk_controller {
        let d = vm::DiskController::parse(d)?;
        match &mut c.q35 {
            Some(q) => q.disk_controller = d,
            None => bail!("{}: --disk-controller is for x86-64 VMs", c.name),
        }
    }
    if let Some(r) = &a.resolution {
        c.display.resolution = r.clone();
    }
    if let Some(b) = a.borderless {
        c.display.borderless = b;
    }
    if let Some(g) = &a.gpu {
        c.gpu = resolve_gpu(g)?;
    }
    if let Some(b) = a.logs {
        c.logs.enabled = b;
    }
    if let Some(l) = &a.log_level {
        c.logs.level = l.clone();
    }
    if let Some(b) = a.audio {
        c.audio.enabled = b;
    }
    if let Some(b) = &a.audio_backend {
        c.audio.backend = (b != "auto" && !b.is_empty()).then(|| b.clone());
    }
    if let Some(b) = a.network {
        c.network.enabled = b;
    }
    if let Some(p) = a.ssh_port {
        c.network.ssh_port = vm::ssh_port(p);
    }
    if let Some(m) = &a.mac {
        c.network.mac = m.clone();
    }
    if let Some(t) = a.stop_timeout {
        c.stop_timeout_secs = t;
    }
    if let Some(x) = &a.extra_args {
        c.extra_args = shlex::split(x).context("--extra-args: unbalanced quotes")?;
    }
    for kv in &a.env {
        let (k, v) = kv
            .split_once('=')
            .with_context(|| format!("--env takes KEY=VALUE, not {kv:?}"))?;
        if v.is_empty() {
            c.extra_env.remove(k);
        } else {
            c.extra_env.insert(k.into(), v.into());
        }
    }
    vm.save()?;
    let note = if runtime::running(&vm) {
        " (at its next start)"
    } else {
        ""
    };
    println!("{}: saved{note}", vm.name());
    // A graphics given makes the OpenCore image for it, where the VM is stopped; while it runs,
    // its next start makes one without reims-vgpu's kexts on another device, and warns of an
    // image that lacks them on reims-vgpu.
    if a.graphics.is_some() && opencore::made_for_other_graphics(&vm) {
        let name = vm.name().to_string();
        let running = runtime::running(&vm);
        if !running {
            match opencore::renew(&mut vm, None, &mut |p| println!("{name}: {p}")) {
                Ok(s) => println!("{s}"),
                Err(e) => eprintln!(
                    "warning: {name}: its OpenCore image is made for other graphics, and a new \
                     one couldn't be made ({e:#}): reims-ctl update-opencore {name} makes it"
                ),
            }
        } else if let Some(w) = opencore::missing_guest_kexts(&vm) {
            println!("{name}: {w}");
        } else {
            println!(
                "{name}: its OpenCore image loads reims-vgpu's macOS drivers: its next start \
                 makes it a new one without them"
            );
        }
    }
    show(&vm, host);
    Ok(())
}

fn gpus(host: Host) {
    if host.default_profile().is_none() {
        println!("{}", crate::host::UNSUPPORTED);
        return;
    }
    let list = gpu::list();
    if list.is_empty() {
        println!(
            "No Vulkan GPU found (is the Vulkan loader, libvulkan.so.1, installed with your GPU's driver?)"
        );
        return;
    }
    println!("By default the library takes the first discrete GPU, then the first integrated one.");
    for (i, g) in list.iter().enumerate() {
        println!(
            "{:>2}. {}  [{:04x}:{:04x}]  {}",
            i + 1,
            g.name,
            g.vendor_id,
            g.device_id,
            g.driver_file.display()
        );
    }
    println!("Give a VM one with: reims-ctl set NAME --gpu NUMBER");
}

/// `reims-ctl usb`: this host's USB devices, or a VM's choice of them, changed by `add` and
/// `remove` first.
fn usb(host: Host, name: Option<&str>, add: &[String], remove: &[String]) -> Result<()> {
    use crate::usb;
    if host.default_profile().is_none() {
        bail!("{}", crate::host::UNSUPPORTED);
    }
    let devices = usb::host_devices();
    let Some(name) = name else {
        if !add.is_empty() || !remove.is_empty() {
            bail!("name the VM to give the devices to: reims-ctl usb NAME --add DEVICE");
        }
        if devices.is_empty() {
            println!("No USB devices are plugged into this host, other than hubs.");
            return Ok(());
        }
        for (i, r) in usb::rows(&[], &devices).iter().enumerate() {
            println!("{}", usb_line(i, None, r, &devices));
        }
        println!("Give a VM one with: reims-ctl usb NAME --add NUMBER");
        return Ok(());
    };
    let mut vm = Registry::load()?.open(name)?;
    if !add.is_empty() || !remove.is_empty() {
        // Each named in the list as it was, then turned over in the settings as they are.
        let rows = usb::rows(&vm.config.usb, &devices);
        let mut picks = Vec::new();
        // A device not plugged in, given by its IDs.
        let mut absent = Vec::new();
        for (arg, on) in add
            .iter()
            .map(|a| (a, true))
            .chain(remove.iter().map(|a| (a, false)))
        {
            let found = usb::find(&rows, arg);
            let by_id = vm::UsbDevice {
                id: arg.trim().to_ascii_lowercase(),
                port: None,
                name: String::new(),
            };
            let r = match found {
                Ok(i) => &rows[i],
                Err(_) if on && by_id.ids().is_ok() => {
                    absent.push(by_id);
                    continue;
                }
                Err(e) => return Err(e),
            };
            match (r.chosen.is_some(), on) {
                (true, true) => println!("{name}: {} has it already", r.name()),
                (false, false) => println!("{name}: {} isn't the VM's", r.name()),
                _ => picks.push(r.clone()),
            }
        }
        if !picks.is_empty() || !absent.is_empty() {
            let (saved, said) = usb::apply(&vm, |cur| {
                let mut c = picks
                    .iter()
                    .fold(cur.to_vec(), |c, r| usb::toggle(&c, &devices, r));
                for u in &absent {
                    if !c.contains(u) {
                        c.push(u.clone());
                    }
                }
                c
            })?;
            for s in said {
                println!("{s}");
            }
            vm = saved;
        }
    }
    let rows = usb::rows(&vm.config.usb, &devices);
    if rows.is_empty() {
        println!("No USB devices are plugged into this host, other than hubs.");
        return Ok(());
    }
    for (i, r) in rows.iter().enumerate() {
        println!("{}", usb_line(i, Some(r.chosen.is_some()), r, &devices));
    }
    if add.is_empty() && remove.is_empty() {
        println!(
            "Give {name} one with: reims-ctl usb {name} --add NUMBER; take one back with --remove NUMBER"
        );
    }
    Ok(())
}

/// A line of `usb`'s list: its number, whether the VM has it, its name, IDs and notes.
fn usb_line(
    i: usize,
    chosen: Option<bool>,
    r: &crate::usb::Row,
    devices: &[crate::usb::HostDevice],
) -> String {
    let mark = match chosen {
        Some(true) => "[x] ",
        Some(false) => "[ ] ",
        None => "",
    };
    let mut line = format!("{:>2}. {mark}{}  [{}]", i + 1, r.name(), r.id());
    let notes = r.notes(devices);
    if !notes.is_empty() {
        line.push_str("  ");
        line.push_str(&notes.join(", "));
    }
    line
}

fn doctor(host: Host) -> Result<i32> {
    let mut bad = 0;
    let mut line = |ok: bool, what: &str, detail: String| {
        if !ok {
            bad += 1;
        }
        println!("{} {what}: {detail}", if ok { "ok  " } else { "FAIL" });
    };
    let profile = host.default_profile();
    line(
        profile.is_some(),
        "host",
        format!(
            "{host}{}",
            profile
                .map(|p| format!(", runs {p} VMs"))
                .unwrap_or_else(|| format!(": {}", crate::host::UNSUPPORTED))
        ),
    );
    line(
        true,
        "files",
        format!(
            "settings in {}, VMs and QEMU in {}",
            paths::tilde(&paths::config_dir()),
            paths::tilde(&paths::data_dir())
        ),
    );
    let Some(profile) = profile else { return Ok(1) };
    match install::qemu_binary(profile) {
        Ok(q) => {
            line(
                true,
                "qemu",
                format!(
                    "{} ({})",
                    q.display(),
                    install::qemu_version(&q).unwrap_or_default()
                ),
            );
            let b = install::audio_backends(&q);
            line(
                true,
                "audio backends",
                format!(
                    "{} (default {})",
                    b.join(", "),
                    install::default_audio_backend(&b)
                ),
            );
        }
        Err(e) => line(false, "qemu", format!("{e:#}")),
    }
    match install::Install::load() {
        Some(i) => line(
            true,
            "checkout",
            format!(
                "{} at {}, following {}",
                i.repo.display(),
                i.commit,
                update::following(&i.repo)
            ),
        ),
        None => line(
            false,
            "checkout",
            "no install record: reims-ctl update needs scripts/install-ctl.sh to have run".into(),
        ),
    }
    match host.os {
        crate::host::Os::Linux => {
            let kvm = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open("/dev/kvm");
            line(
                kvm.is_ok(),
                "/dev/kvm",
                match kvm {
                    Ok(_) => "usable".into(),
                    Err(e) => format!(
                        "{e}: add yourself to the kvm group (sudo usermod -aG kvm $USER) and log in again"
                    ),
                },
            );
            let udmabuf = runtime::udmabuf_problem();
            line(
                udmabuf.is_none(),
                "/dev/udmabuf",
                udmabuf.clone().unwrap_or_else(|| "usable".into()),
            );
            if let Some(n) = runtime::host_max_map_count() {
                // A speed-up only, on NVIDIA where udmabuf opens (runtime::max_map_count): for
                // the largest VM, or a new one.
                let memory_mib = Registry::load()
                    .map(|reg| {
                        reg.vms
                            .iter()
                            .filter_map(|e| Vm::load(&e.dir).ok())
                            .map(|vm| vm.config.memory_mib)
                            .max()
                            .unwrap_or(0)
                    })
                    .unwrap_or(0)
                    .max(create::default_memory_mib());
                let note = (udmabuf.is_none() && runtime::host_has_nvidia())
                    .then(|| runtime::map_count_note(n, memory_mib))
                    .flatten();
                line(
                    true,
                    "vm.max_map_count",
                    match note {
                        Some(note) => format!("{n}: {note}"),
                        None => n.to_string(),
                    },
                );
            }
            // MGLRU's thrashing guard, told of above 0 and offered off on a terminal; never a
            // failure, as VMs run with it.
            let min_ttl = Path::new(runtime::MIN_TTL_FILE);
            if let Some(ms) = runtime::min_ttl_ms(min_ttl) {
                let note = runtime::min_ttl_note(ms);
                line(
                    true,
                    "lru_gen min_ttl_ms",
                    note.clone().unwrap_or(ms.to_string()),
                );
                if note.is_some() {
                    println!("     {}", offer_min_ttl_off(min_ttl));
                }
            }
            let gpus = gpu::list();
            line(
                !gpus.is_empty(),
                "vulkan",
                if gpus.is_empty() {
                    "no GPU found".into()
                } else {
                    gpus.iter()
                        .map(|g| g.label())
                        .collect::<Vec<_>>()
                        .join("; ")
                },
            );
            // USB passthrough: QEMU's usb-host, and the devices this user may open, which
            // fail the check only where a VM is given USB devices.
            let devices = crate::usb::host_devices();
            let closed: Vec<String> = devices
                .iter()
                .filter(|d| !d.usable())
                .map(|d| d.file().display().to_string())
                .collect();
            let used = Registry::load().is_ok_and(|reg| {
                reg.vms
                    .iter()
                    .filter_map(|e| Vm::load(&e.dir).ok())
                    .any(|vm| !vm.config.usb.is_empty())
            });
            let usb_host = install::qemu_binary(profile)
                .ok()
                .and_then(|q| install::has_device(&q, "usb-host"));
            let (ok, detail) = if usb_host == Some(false) {
                (
                    !used,
                    "this QEMU has no usb-host, which passes USB devices through: reims-ctl \
                     update builds one with it"
                        .to_string(),
                )
            } else if devices.is_empty() {
                (true, "no devices plugged in, other than hubs".to_string())
            } else if closed.is_empty() {
                (
                    true,
                    format!(
                        "{} to pass through (reims-ctl usb)",
                        plural(devices.len(), "device")
                    ),
                )
            } else {
                (
                    !used,
                    format!(
                        "you can't open {} of the {} plugged in ({}): {}",
                        closed.len(),
                        plural(devices.len(), "device"),
                        closed.join(", "),
                        crate::usb::fix()
                    ),
                )
            };
            line(ok, "usb", detail);
            let session = std::env::var_os("WAYLAND_DISPLAY").is_some()
                || std::env::var_os("DISPLAY").is_some();
            line(
                session,
                "desktop session",
                if session {
                    "found".into()
                } else {
                    "neither WAYLAND_DISPLAY nor DISPLAY is set: a VM's window needs a desktop session".into()
                },
            );
        }
        crate::host::Os::Mac | crate::host::Os::Other => {}
    }
    if profile == Profile::Q35X86_64 {
        // What makes a new VM's OpenCore image, with its OVMF, and downloads its installer.
        match reims_util::binary() {
            Ok(b) => {
                let v = reims_util::version(&b);
                line(
                    v.is_some(),
                    "reims-util",
                    match v {
                        Some(v) => format!("{} ({v})", b.display()),
                        None => format!(
                            "{} does not run: reinstall it with reims-ctl update",
                            b.display()
                        ),
                    },
                );
            }
            Err(e) => line(false, "reims-util", format!("{e:#}")),
        }
    }
    Ok(if bad == 0 { 0 } else { 1 })
}

/// `n` of `what`: 1 device, 2 devices.
fn plural(n: usize, what: &str) -> String {
    if n == 1 {
        format!("1 {what}")
    } else {
        format!("{n} {what}s")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testenv::Env;

    /// `name`, registered with `dir`, its settings `config`.
    fn register(name: &str, dir: &Path, config: &str) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join(vm::CONFIG_FILE), config).unwrap();
        let mut reg = Registry::load().unwrap();
        reg.add(name, dir).unwrap();
        reg.save().unwrap();
    }

    #[test]
    fn a_running_vm_whose_settings_do_not_read_is_not_deleted() {
        let env = Env::new("delete-running");
        let dir = env.root.join("vms/x");
        register("x", &dir, "name = ");
        let pidfile = vm::pidfile_in(&dir);
        let mut qemu = crate::runtime::test_qemu(&pidfile, 30);
        let r = delete_vm("x", false);
        let _ = qemu.kill();
        let _ = qemu.wait();
        let e = format!("{:#}", r.unwrap_err());
        assert!(e.contains("x is running"), "{e}");
        assert!(dir.join(vm::CONFIG_FILE).is_file());
        assert!(Registry::load().unwrap().find("x").is_some());
    }

    /// A VM deleted with its files takes no other VM's with it: one made in a directory inside
    /// its own (`--dir ~/VMs` for the first VM, `--dir ~/VMs/b` for the next), or one using a
    /// disk in it in place.
    #[test]
    fn deleting_a_vms_files_spares_other_vms() {
        let env = Env::new("delete-nested");
        let config = |name: &str, disk: &str| {
            format!(
                r#"
                name = "{name}"
                profile = "q35-x86_64"
                cpus = 2
                memory_mib = 4096
                disk = "{disk}"
                [q35]
                opencore = "opencore.qcow2"
                firmware_code = "OVMF_CODE.fd"
                firmware_vars = "OVMF_VARS.fd"
                "#
            )
        };
        let a = env.root.join("VMs");
        register("a", &a, &config("a", "disk.img"));
        let b = a.join("b");
        register("b", &b, &config("b", "disk.img"));
        let shared = a.join("shared.img");
        std::fs::write(&shared, "c's disk").unwrap();
        register(
            "c",
            &env.root.join("c"),
            &config("c", &shared.to_string_lossy()),
        );

        let e = format!("{:#}", delete_vm("a", false).unwrap_err());
        assert!(e.contains("b's files"), "{e}");
        assert!(b.join(vm::CONFIG_FILE).is_file());
        assert!(Registry::load().unwrap().find("a").is_some());
        delete_vm("b", false).unwrap();
        assert!(!b.exists());
        let e = format!("{:#}", delete_vm("a", false).unwrap_err());
        assert!(e.contains("c's files"), "{e}");
        assert_eq!(std::fs::read_to_string(&shared).unwrap(), "c's disk");
        // Kept, its files may hold others'.
        delete_vm("a", true).unwrap();
        assert!(Registry::load().unwrap().find("a").is_none());
        assert!(shared.is_file());
    }

    #[test]
    fn a_change_of_installer_deletes_only_the_vms_own_copy() {
        let d = std::env::temp_dir().join(format!("reims-ctl-installer-{}", std::process::id()));
        std::fs::create_dir_all(d.join("vm")).unwrap();
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
            installer = "installer.img"
        "#;
        let mut vm = Vm {
            dir: d.join("vm"),
            config: toml::from_str(config).unwrap(),
        };
        vm.save().unwrap();
        let copy = d.join("vm").join(create::INSTALLER_COPY);
        let user = d.join("BaseSystem.img");
        std::fs::write(&copy, [0u8; 4096]).unwrap();
        std::fs::write(&user, [0u8; 4096]).unwrap();
        let installer = |vm: &Vm| vm.config.q35.as_ref().unwrap().installer.clone();
        let mut quiet = |_: &str| {};
        // The copy, named by its path (as the TUI shows it), stays the VM's copy.
        set_installer(&mut vm, Some(copy.clone()), &mut quiet).unwrap();
        assert_eq!(installer(&vm), Some(create::INSTALLER_COPY.into()));
        assert!(copy.is_file());
        // Another installer replaces it, and the copy goes.
        set_installer(&mut vm, Some(user.clone()), &mut quiet).unwrap();
        assert_eq!(installer(&vm), Some(user.clone()));
        assert!(!copy.exists());
        // An installer used in place is never deleted.
        set_installer(&mut vm, None, &mut quiet).unwrap();
        assert_eq!(installer(&vm), None);
        assert!(user.is_file());

        // What was saved while it ran stays: here, while a DMG would convert.
        let mut stale = vm.clone();
        vm.config.memory_mib = 8192;
        vm.save().unwrap();
        set_installer(&mut stale, Some(user.clone()), &mut quiet).unwrap();
        let saved = Vm::load(&d.join("vm")).unwrap();
        assert_eq!(saved.config.memory_mib, 8192);
        assert_eq!(installer(&saved), Some(user.clone()));
        assert_eq!(stale.config, saved.config);

        // An installer of the user's, named relative to the VM's directory by hand, is the
        // user's, not the VM's copy: neither a change nor an eject deletes it.
        let other = d.join("Other.img");
        std::fs::write(&other, [0u8; 4096]).unwrap();
        for own in ["BaseSystem.img", "../BaseSystem.img"] {
            let file = vm.dir.join(own);
            std::fs::write(&file, [0u8; 4096]).unwrap();
            for to in [Some(other.clone()), None] {
                vm.config.q35.as_mut().unwrap().installer = Some(own.into());
                vm.save().unwrap();
                set_installer(&mut vm, to.clone(), &mut quiet).unwrap();
                assert!(file.is_file(), "{own} to {to:?}");
            }
        }
        std::fs::remove_dir_all(&d).unwrap();
    }
}
