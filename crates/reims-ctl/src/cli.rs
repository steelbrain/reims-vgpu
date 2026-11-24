//! The commands. With none, reims-ctl opens its interactive interface (tui.rs).

use crate::create::{self, DiskSource, ImportMode, Spec};
use crate::host::{Host, Profile};
use crate::vm::{self, Registry, Vm, format_mib, parse_size_mib};
use crate::{gpu, install, logs, opencore, paths, qemu, reims_util, runtime, update, view};
use anyhow::{Context, Result, anyhow, bail};
use clap::{Args, Parser, Subcommand};
use std::io::Write;
use std::path::{Path, PathBuf};

/// What `reims-ctl --help` ends with: where VMs live, in the host's own form (paths.rs).
const AFTER_HELP: &str = if cfg!(windows) {
    "VMs live in ~\\.local\\share\\reims-ctl\\vms unless created elsewhere (--dir); \
     `reims-ctl command NAME` prints the QEMU command a VM runs, for scripts of your own."
} else {
    "VMs live in ~/.local/share/reims-ctl/vms unless created elsewhere (--dir); \
     `reims-ctl command NAME` prints the QEMU command a VM runs, for scripts of your own."
};

#[derive(Parser)]
#[command(
    name = "reims-ctl",
    version,
    about = "Create, boot and manage macOS VMs on QEMU with reims-vgpu",
    long_about = "Create, boot and manage macOS VMs on QEMU with reims-vgpu.\n\n\
        Run with no command for the interactive interface. Every option takes its value as \
        `--name value` or `--name=value`; yes/no options also take a bare `--name` for yes.",
    after_help = AFTER_HELP
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
    /// Start a VM: a window opens for each of its displays
    #[command(after_help = START_HELP)]
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
    /// Take the installer out of a VM; a running VM need not stop, and drops it at its next
    /// start
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
    /// List the GPUs a VM can be given (Linux and Windows)
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
    /// List this host's disks, and give a VM whole ones or take them back (Linux)
    #[command(after_help = DISKS_HELP)]
    Disks {
        /// The VM whose disks to show or change (without one, this host's disks)
        name: Option<String>,
        /// Give the VM this disk (repeatable)
        #[arg(long, value_name = "DISK")]
        add: Vec<String>,
        /// Take this disk back from the VM (repeatable)
        #[arg(long, value_name = "DISK")]
        remove: Vec<String>,
    },
    /// List what's downloaded for new VMs (macOS installers and IPSWs) and the space each
    /// takes, or delete one
    Downloads {
        #[command(subcommand)]
        action: Option<DownloadsCmd>,
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
    #[command(after_help = update_help())]
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
        /// The QEMU build installed.
        #[arg(long)]
        qemu: Option<PathBuf>,
        #[arg(long, default_value = "")]
        commit: String,
        /// The bash that ran the install, which `update` runs it in again: on Windows, MSYS2's.
        #[arg(long)]
        shell: Option<PathBuf>,
    },
}

const START_HELP: &str = "\
On Linux, the VM's QEMU runs in a systemd user unit of its own, reims-ctl-NAME.scope, so it gets \
a share of the CPU of its own beside other work, a build in the terminal that started it among \
it; systemctl --user status reims-ctl-NAME.scope shows it. Where no systemd user manager \
answers, or from an SSH login whose manager does not linger (loginctl enable-linger), QEMU runs \
as part of the terminal's work instead, and the start warns so.";

const VIEW_HELP: &str = "\
For a VM on a host with no screen, to set macOS up before it has screen sharing or remote \
login. The screen is drawn with the terminal's graphics (kitty's, iTerm2's or sixel; half-block \
characters where it has none), over SSH as well, and the terminal's keys, mouse and pasted text \
go to the guest as a US keyboard's and a tablet's. On a VM with several displays it shows the \
first.

Ctrl-] disconnects; the VM keeps running. Alt+key is Cmd+key. The Cmd, Opt, Ctrl and Shift \
buttons on the top row hold that key for the next key or click.";

/// What `reims-ctl update --help` ends with: when it builds, then [`UPDATE_MIGRATIONS`].
fn update_help() -> String {
    format!(
        "Where the checkout is already at the latest commit of the branch it follows, and \
         reims-ctl and its QEMU were installed from that commit, update builds nothing and says \
         so; {} in the checkout builds them again anyway.\n\n{UPDATE_MIGRATIONS}",
        install::INSTALL_SCRIPT
    )
}

const UPDATE_MIGRATIONS: &str = "\
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

/// `disks --help`'s account of disk passthrough.
const DISKS_HELP: &str = "\
DISK is a disk's number in the list, its file (/dev/sdb, or its link in /dev/disk/by-id), or a \
word of its name.

A disk a VM has is the guest's while the VM runs, a SATA disk to macOS in the guest, which reads \
and writes it as a Mac does a disk inside it. A VM keeps it for its next start: a disk given or \
taken back while the VM runs is the guest's, or this host's again, from then. An x86-64 VM \
takes up to three, or four where its own disk is on NVMe or virtio.

A disk is known by its link in /dev/disk/by-id, named from its model and serial number, so the \
VM has it in any port and after this host restarts. A disk with no link there, as behind some \
USB enclosures, can't be given. A loop device is known by its file in /dev, which may lead to \
another one after this host restarts.

A disk this host uses can't be given, since a guest writing it would destroy it: one that is \
mounted, or has a partition mounted (this host's system disk among them), swap, or part of LVM, \
LUKS or RAID, and the disk that holds the VM's own image. A start checks again, and asks the \
kernel whether anything holds the disk.

QEMU opens the disk's file, which only root and the disk group may open, and the disk group \
may write every disk of this host. Where you can't open a disk, reims-ctl gives the udev rule \
that makes that disk alone yours, and the commands that add it.";

/// `--flag`, `--flag=true`, `--flag false`.
fn bool_arg() -> clap::Arg {
    clap::Arg::new("x")
        .num_args(0..=1)
        .default_missing_value("true")
}

/// `create --help`'s examples, under a header styled as clap styles its own.
fn create_examples() -> String {
    let h = *clap::builder::Styles::default().get_header();
    format!("{h}Examples:{h:#}{}", create_examples_on(Host::current()))
}

/// `e`, followed by the examples: for a create that was not given enough to make a VM.
fn with_create_examples(e: impl std::fmt::Display) -> anyhow::Error {
    anyhow!("{e}\n\nExamples:{}", create_examples_on(Host::current()))
}

/// The examples of a create on `host`. Each guest's are headed by the hosts that make it: the
/// x86-64 guest's by `host`'s own system where it makes that guest ([`Profile::supported_on`]).
/// Their paths are in `host`'s own form: on Windows with `\` between their parts, as reims-ctl
/// prints paths there (paths.rs).
fn create_examples_on(host: Host) -> String {
    let x86 = if !Profile::supported_on(host).contains(&Profile::Q35X86_64) {
        "x86-64 Linux or Windows"
    } else if host.os == crate::host::Os::Windows {
        "Windows"
    } else {
        "Linux"
    };
    let path = |p: &str| {
        if host.os == crate::host::Os::Windows {
            p.replace('/', "\\")
        } else {
            p.to_string()
        }
    };
    format!(
        r"
  An x86-64 guest on {x86}:

    # Download macOS Tahoe from Apple, make the VM's OpenCore image and a new
    # 128G disk, and start it: the installer boots into macOS Recovery. Once
    # macOS is installed, reims-ctl eject mac takes the installer out at the
    # VM's next start, so the VM can keep running
    reims-ctl create mac --macos tahoe --start

    # One exact release, which installs with no network (13 to 20 GB); with the VM's
    # network off, its installer downloads no update as it installs
    reims-ctl create mac --macos 26.6.2 --offline --network false --start

    # An installer of your own, such as a recovery image (BaseSystem.dmg)
    reims-ctl create mac --installer {installer}

    # A disk that already has macOS installed (copied; see --disk-mode): the
    # new OpenCore image presents the Mac of the one beside it, opencore.qcow2,
    # as a VM's directory holds them, so macOS sees no new Mac. A copy of the
    # disk of a VM reims-ctl has is a Mac of its own, as it runs beside it
    reims-ctl create mac --disk {old_disk}

    # Or with the OpenCore image it booted from, copied, which keeps its Mac too
    reims-ctl create mac --disk {disk} --opencore {opencore}

  An arm64 guest on an arm64 Mac, arm64 Linux or x86-64 Linux:

    # Download the current macOS's IPSW from Apple (about 25 GB, kept for the next VM; a
    # stopped download continues), restore it onto a new 128G disk, with tickets Apple
    # signs for the VM, and start it. On x86-64 Linux the VM runs emulated (QEMU's TCG),
    # far slower than on an arm64 host
    reims-ctl create mac --ipsw latest --start

    # An IPSW of your own, read where it is: Apple offers only the current macOS to download.
    # macOS makes its Data volume at the VM's first boot and then resets the VM: after that
    # boot, reims-restore finish-eacs DIR makes it ready for Setup Assistant
    reims-ctl create mac --ipsw {ipsw}",
        installer = path("~/Downloads/BaseSystem.dmg"),
        old_disk = path("~/vms/old/disk.qcow2"),
        disk = path("~/vms/mac.qcow2"),
        opencore = path("~/vms/OpenCore.qcow2"),
        ipsw = path("~/Downloads/UniversalMac_14.6.1_23G93_Restore.ipsw"),
    )
}

#[derive(Subcommand)]
pub enum DownloadsCmd {
    /// Delete downloads, by the names the list shows: a later VM of their macOS downloads them
    /// again
    Delete {
        #[arg(required = true, value_name = "NAME")]
        names: Vec<String>,
        /// Do not ask
        #[arg(long, short)]
        yes: bool,
    },
}

/// `downloads`: what's kept for new VMs, and the space it takes.
fn downloads() {
    let items = crate::downloads::list();
    let dirs: Vec<String> = crate::downloads::dirs()
        .iter()
        .filter(|d| d.is_dir())
        .map(|d| paths::tilde(d))
        .collect();
    if items.is_empty() {
        println!(
            "Nothing is downloaded: a new VM downloads its macOS, and it's kept here for the next."
        );
        return;
    }
    let w = items
        .iter()
        .map(|i| i.name().len())
        .max()
        .unwrap_or(4)
        .max(4);
    println!("{:<w$}  {:>8}  WHAT", "NAME", "SIZE");
    for i in &items {
        println!(
            "{:<w$}  {:>8}  {}",
            i.name(),
            crate::downloads::size(i.bytes),
            i.what
        );
    }
    let total: u64 = items.iter().map(|i| i.bytes).sum();
    println!(
        "\n{} in {}. reims-ctl downloads delete NAME deletes one.",
        crate::downloads::size(total),
        dirs.join(" and ")
    );
}

/// `downloads delete`: each of `names`, asking first unless `yes`.
fn delete_downloads(names: &[String], yes: bool) -> Result<()> {
    let items = crate::downloads::list();
    for n in names {
        let Some(i) = items.iter().find(|i| &i.name() == n) else {
            bail!("no download is named {n}: reims-ctl downloads lists them");
        };
        if !yes && !confirm(&format!("Delete {n}, {}?", crate::downloads::size(i.bytes))) {
            println!("{n}: not deleted");
            continue;
        }
        println!("{}", crate::downloads::delete(n)?);
    }
    Ok(())
}

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
    /// release as it installs. Install it with the VM's network off (--network false), so its
    /// installer downloads no update
    #[arg(long, requires = "macos")]
    pub offline: bool,
    /// An installer image of your own, attached as a disk: a raw image, such as a recovery
    /// image's BaseSystem.img, or a DMG, converted to a raw image in the VM's directory
    #[arg(long)]
    pub installer: Option<PathBuf>,
    /// On an arm64 Mac, arm64 Linux or x86-64 Linux: the IPSW to restore an arm64 VM's macOS
    /// from: latest,
    /// the current macOS, downloaded from Apple once and kept (reims-ctl downloads lists it),
    /// or an IPSW's path, read where it is. The VM needs one boot, which makes its Data volume,
    /// and then `reims-restore finish-eacs DIR`
    #[arg(long, value_name = "latest|PATH", conflicts_with_all = ["macos", "installer", "disk", "opencore"])]
    pub ipsw: Option<String>,
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
    /// How many displays the guest has on reims-vgpu, 1 to 4 (default 1), each in a window of
    /// its own, where macOS shows its desktop. VMware SVGA has one
    #[arg(long, value_name = "N")]
    pub displays: Option<u32>,
    /// The size each of reims-vgpu's windows opens at, WIDTHxHEIGHT in the desktop's points,
    /// which its display follows; the guest starts a HiDPI desktop by itself on a display whose
    /// window is at least 3840 pixels wide, and offers one in its Displays settings from 2880
    #[arg(long)]
    pub resolution: Option<String>,
    /// Borderless windows that fill their screens, one for each display
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    pub borderless: Option<bool>,
    /// The GPU: auto, a number or name from `reims-ctl gpus`, or a Vulkan ICD JSON file
    #[arg(long)]
    pub gpu: Option<String>,
    /// Keep the library's log and the serial console
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    pub logs: Option<bool>,
    /// Two virtio sound devices on the host's sound, a speaker and a microphone, and the HDA
    /// controller OpenCore's boot chime plays on
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    pub audio: Option<bool>,
    /// Its network card (user-mode NAT); --network false makes it with none, as an offline
    /// installer's install is best done
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    pub network: Option<bool>,
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
    /// them, as update-opencore does, where the VM is stopped), on 1 to 4 displays (--displays),
    /// each in a window whose size the guest follows; or vmware-svga, QEMU's VMware SVGA: one
    /// display with no GPU acceleration, in QEMU's own window
    #[arg(long)]
    pub graphics: Option<String>,
    /// How many displays the guest has on reims-vgpu, 1 to 4, each in a window of its own, where
    /// macOS shows its desktop. VMware SVGA has one
    #[arg(long, value_name = "N")]
    pub displays: Option<u32>,
    /// The system disk's controller: ahci (SATA), nvme or virtio (virtio-blk, tried with an
    /// installed macOS 26 alone). Once macOS is installed, eject the installer before a move, or
    /// OpenCore may start it
    #[arg(long)]
    pub disk_controller: Option<String>,
    /// The size each of reims-vgpu's windows opens at, WIDTHxHEIGHT in the desktop's points
    #[arg(long)]
    pub resolution: Option<String>,
    /// Borderless windows that fill their screens, one for each display
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
    // On Windows, the programs an update set aside go where nothing runs them now; not while the
    // installer records its install, between its set-aside of reims-util and the put-back.
    if !matches!(cli.command, Some(Cmd::RecordInstall { .. })) {
        crate::os::remove_set_aside_programs();
    }
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
            let started = runtime::start(&vm, host)?;
            if let Some(n) = &started.note {
                eprintln!("warning: {n}");
            }
            println!("{name}: running, pid {}", started.pid);
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
                    crate::os::set_mode(&p, 0o755)?;
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
        Cmd::Downloads { action } => match action {
            None => downloads(),
            Some(DownloadsCmd::Delete { names, yes }) => delete_downloads(&names, yes)?,
        },
        Cmd::Usb { name, add, remove } => usb(host, name.as_deref(), &add, &remove)?,
        Cmd::Disks { name, add, remove } => disks(host, name.as_deref(), &add, &remove)?,
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
        Cmd::Update { channel } => {
            update::update(channel)?;
        }
        Cmd::Doctor => return doctor(host),
        Cmd::Tui => unreachable!(),
        Cmd::VulkanProbe => return Ok(gpu::probe_main()),
        Cmd::RecordInstall {
            repo,
            qemu,
            commit,
            shell,
        } => {
            install::Install {
                repo: repo.clone(),
                commit,
                qemu: qemu.unwrap_or_default(),
                installed_at: runtime::now_stamp("%Y-%m-%d %H:%M:%S %z"),
                shell: shell.unwrap_or_default(),
            }
            .save()?;
            // What an arm64 VM boots on, so its first start needs no network (ambrosia.rs).
            if Profile::supported_on(host).contains(&Profile::VmappleArm64)
                && !crate::ambrosia::is_kept()
            {
                println!(
                    "==> Downloading Ambrosia {}, the arm64 VMs' firmware",
                    crate::ambrosia::VERSION
                );
                crate::ambrosia::image()?;
            }
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
        vm::Graphics::ReimsVgpu if vm.config.display.count > 1 => format!(
            "Its {} windows, one for each display, open under -display none",
            vm.config.display.count
        ),
        vm::Graphics::ReimsVgpu => "Its window opens under -display none".into(),
        vm::Graphics::VmwareSvga if crate::host::has_screen(&vm.config.extra_env) => {
            "Its window is QEMU's own, under -display gtk".into()
        }
        vm::Graphics::VmwareSvga => {
            "With no screen here, it runs headless under -display none".into()
        }
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
            select: None,
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
    let source = a.macos.is_some()
        || a.installer.is_some()
        || a.disk.is_some()
        || a.opencore.is_some()
        || a.ipsw.is_some();
    let arm64 = a.ipsw.is_some() || host.default_profile() == Some(Profile::VmappleArm64);
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
            size_mib: create::DEFAULT_DISK_MIB,
        }),
    };
    let abs = |p: &Option<PathBuf>| p.as_ref().map(|p| paths::absolute(&p.to_string_lossy()));
    s.macos = a.macos.clone().map(|version| reims_util::Download {
        version,
        offline: a.offline,
    });
    s.opencore = abs(&a.opencore);
    s.installer = abs(&a.installer);
    s.ipsw = a.ipsw.as_deref().map(|i| {
        if crate::ipsw::is_path(i) {
            paths::absolute(i).display().to_string()
        } else {
            i.to_string()
        }
    });
    s.mac = a.mac.clone();
    s.graphics = a.graphics.as_deref().map(vm::Graphics::parse).transpose()?;
    if let Some(n) = a.displays {
        s.display.count = n;
    }
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
    s.network = a.network.unwrap_or(true);
    s.ssh_port = a.ssh_port;
    if arm64 {
        // Ctrl-C stops the restore, which removes what it made.
        crate::restore::stop_at_interrupt();
    }
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
        let started = runtime::start(&vm, host)?;
        if let Some(n) = &started.note {
            eprintln!("warning: {n}");
        }
        println!("{name}: running, pid {}", started.pid);
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
    add("kind", c.profile.describe(host).into());
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
    if let Some(a) = &c.vmapple {
        add("aux", paths::tilde(&vm.path(&a.aux)));
        add("ecid", format!("{:#018x}", a.ecid));
    }
    let graphics = c.graphics();
    match graphics {
        vm::Graphics::VmwareSvga => add(
            "graphics",
            "VMware SVGA, no GPU acceleration, in QEMU's window".into(),
        ),
        g => add("graphics", g.describe().into()),
    }
    // reims-vgpu's displays follow their windows: of the settings, only their count, the
    // windows' first size and borderless reach them.
    if graphics == vm::Graphics::ReimsVgpu {
        let n = c.display.count;
        let (w, h) = c.display.window_size().unwrap_or((1920, 1080));
        match (n, c.display.borderless) {
            (1, true) => add("display", "one, filling its screen, borderless".into()),
            (1, false) => add(
                "display",
                format!("one, in a window opening at {w}x{h}, whose size the guest follows"),
            ),
            (n, true) => add(
                "displays",
                format!("{n}, in {n} borderless windows, each filling its screen"),
            ),
            (n, false) => add(
                "displays",
                format!("{n}, in {n} windows opening at {w}x{h}, whose sizes the guest follows"),
            ),
        }
    }
    if graphics.library() && c.profile.gpu_choice(host) {
        add(
            "gpu",
            match (&c.gpu.name, &c.gpu.driver_file) {
                (Some(n), _) => n.clone(),
                (None, Some(f)) => paths::tilde(f),
                _ => "automatic: the only GPU, or a discrete one before an integrated one".into(),
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
    if !c.host_disks.is_empty() {
        add(
            "host disks",
            c.host_disks
                .iter()
                .map(|d| match d.name.as_str() {
                    "" => d.path.display().to_string(),
                    n => format!("{n} ({})", d.path.display()),
                })
                .collect::<Vec<_>>()
                .join(", "),
        );
    }
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
/// replaces is deleted, or what happened says it could not be (on Windows, while the VM runs).
/// The installer is set in the VM's settings as they are once that is done, which `vm` becomes:
/// a DMG takes seconds to convert, in which others may be saved (the interface's edit form or
/// logs toggle, reims-ctl set).
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
    // On Windows the VM's QEMU, while it runs, holds the copy so that it cannot be deleted.
    let mut kept = String::new();
    if let Some(c) = copy(&old)
        && installer != old
        && let Err(e) = std::fs::remove_file(&c)
        && e.kind() != std::io::ErrorKind::NotFound
    {
        kept = format!(
            "; its old copy {} could not be deleted: {e}",
            paths::tilde(&c)
        );
        if running {
            kept.push_str(". Delete it once the VM stops");
        }
    }
    let said = match (&path, running) {
        (Some(p), false) => format!("{name}: {} attached for the next start", paths::tilde(p)),
        (Some(_), true) => format!("{name}: the installer changes at the next start"),
        (None, false) => format!("{name}: installer removed"),
        (None, true) => format!("{name}: the installer goes at the next start"),
    };
    Ok(said + &kept)
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
    // A count given is checked against the graphics; one kept from reims-vgpu waits unused on
    // VMware SVGA, as the resolution does.
    if let Some(n) = a.displays {
        c.display.count = n;
        c.display.check_on(c.graphics())?;
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
    match host.default_profile() {
        None => return println!("{}", crate::host::UNSUPPORTED),
        Some(p) if !p.gpu_choice(host) => {
            return println!(
                "A VM runs on this Mac's GPU, through Mesa's KosmicKrisp: there is none to choose"
            );
        }
        Some(_) => {}
    }
    let list = gpu::list();
    if list.is_empty() {
        let loader = if cfg!(windows) {
            "vulkan-1.dll"
        } else {
            "libvulkan.so.1"
        };
        println!(
            "No Vulkan GPU found (is the Vulkan loader, {loader}, installed with your GPU's driver?)"
        );
        return;
    }
    println!(
        "By default a VM gets the only GPU, or, of several, a discrete GPU before an integrated \
         one."
    );
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
    println!("{CROSSING}");
}

/// What a VM's window costs on a GPU that does not drive the screen, for `gpus` and the TUI's GPU
/// field (the library's `rt_window` logs the route each window takes).
pub const CROSSING: &str = "If your screen is plugged into a different GPU than the VM's, the \
                            window system copies every frame across to it over the VM's GPU's \
                            link: an RTX 3090 in a PCIe 3.0 x4 slot spends about 16 ms a frame \
                            on it at 3648 x 2039.";

/// `doctor`'s USB passthrough check, on Linux: QEMU's usb-host, and the devices this user may
/// open, which fail the check only where a VM is given USB devices.
fn usb_check(profile: Profile) -> (bool, String) {
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
    if usb_host == Some(false) {
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
    }
}

/// `reims-ctl usb`: this host's USB devices, or a VM's choice of them, changed by `add` and
/// `remove` first.
fn usb(host: Host, name: Option<&str>, add: &[String], remove: &[String]) -> Result<()> {
    use crate::usb;
    if let Err(why) = usb::available(host) {
        bail!("{why}");
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

/// `reims-ctl disks`: this host's disks, or a VM's choice of them, changed by `add` and
/// `remove` first.
fn disks(host: Host, name: Option<&str>, add: &[String], remove: &[String]) -> Result<()> {
    use crate::disks;
    if host.os != crate::host::Os::Linux || host.default_profile() != Some(Profile::Q35X86_64) {
        bail!("{}", disks::ONLY);
    }
    let found = disks::host_disks();
    let Some(name) = name else {
        if !add.is_empty() || !remove.is_empty() {
            bail!("name the VM to give the disks to: reims-ctl disks NAME --add DISK");
        }
        let rows = disks::rows(&[], &found);
        if rows.is_empty() {
            println!("This host has no disks to list.");
            return Ok(());
        }
        for (i, r) in rows.iter().enumerate() {
            println!("{}", disk_line(i, None, r, &[]));
        }
        println!("Give a VM one with: reims-ctl disks NAME --add NUMBER");
        return Ok(());
    };
    let mut vm = Registry::load()?.open(name)?;
    if vm.config.q35.is_none() {
        bail!("{}", disks::ONLY);
    }
    let used = disks::HostUse::current();
    let own = disks::own_devs(&vm, &used);
    if !add.is_empty() || !remove.is_empty() {
        // Each named in the list as it was, then turned over in the settings as they are.
        let rows = disks::rows(&vm.config.host_disks, &found);
        let mut picks = Vec::new();
        for (arg, on) in add
            .iter()
            .map(|a| (a, true))
            .chain(remove.iter().map(|a| (a, false)))
        {
            let r = &rows[disks::find(&rows, arg)?];
            match (r.chosen.is_some(), on) {
                (true, true) => println!("{name}: {} is the VM's already", r.name()),
                (false, false) => println!("{name}: {} isn't the VM's", r.name()),
                _ => picks.push(r.clone()),
            }
        }
        if !picks.is_empty() {
            let (saved, said) = disks::apply(&vm, |cur| {
                picks
                    .iter()
                    .try_fold(cur.to_vec(), |c, r| disks::toggle(&c, r, &own))
            })?;
            for s in said {
                println!("{s}");
            }
            vm = saved;
        }
    }
    let rows = disks::rows(&vm.config.host_disks, &found);
    if rows.is_empty() {
        println!("This host has no disks to list.");
        return Ok(());
    }
    for (i, r) in rows.iter().enumerate() {
        println!("{}", disk_line(i, Some(r.chosen.is_some()), r, &own));
    }
    for p in disks::problems(&vm) {
        eprintln!("warning: {p}");
    }
    if add.is_empty() && remove.is_empty() {
        println!(
            "Give {name} one with: reims-ctl disks {name} --add NUMBER; take one back with --remove NUMBER"
        );
    }
    Ok(())
}

/// A line of `disks`' list: its number, whether the VM has it, its name and size, its file and
/// notes. `own` are the devices that hold the VM's images.
fn disk_line(i: usize, chosen: Option<bool>, r: &crate::disks::Row, own: &[(u32, u32)]) -> String {
    let mark = match chosen {
        Some(true) => "[x] ",
        Some(false) => "[ ] ",
        None => "",
    };
    let mut line = format!("{:>2}. {mark}{}  {}", i + 1, r.name(), r.path().display());
    let notes = r.notes(own);
    if !notes.is_empty() {
        line.push_str("  (");
        line.push_str(&notes.join(", "));
        line.push(')');
    }
    line
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
                .map(|_| {
                    let profiles: Vec<_> =
                        Profile::supported_on(host).iter().map(|p| p.id()).collect();
                    format!(", runs {} VMs", profiles.join(" and "))
                })
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
    for &p in Profile::supported_on(host) {
        // The default profile's QEMU is "qemu", as it was before a host ran two.
        let what = if p == profile {
            "qemu".to_string()
        } else {
            format!("qemu for {} VMs", p.id())
        };
        let what = what.as_str();
        match install::qemu_binary(p) {
            Ok(q) => {
                line(
                    true,
                    what,
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
            Err(e) => line(false, what, format!("{e:#}")),
        }
    }
    if Profile::supported_on(host).contains(&Profile::VmappleArm64) {
        // What an arm64 VM boots on: Ambrosia, which a start downloads where it is missing, and
        // the configuration bundle of what this reims-ctl carries.
        line(
            true,
            "ambrosia",
            if crate::ambrosia::is_kept() {
                format!(
                    "{} {}",
                    crate::ambrosia::VERSION,
                    paths::tilde(&crate::ambrosia::path())
                )
            } else {
                format!(
                    "{} is not downloaded yet: an arm64 VM's start downloads it",
                    crate::ambrosia::VERSION
                )
            },
        );
        let carried = crate::prebuilt::carries_arm64();
        line(
            carried,
            "arm64 bundle",
            if carried {
                "the arm64 guest's kexts and the device's module for Ambrosia".into()
            } else {
                "this reims-ctl was built without the arm64 guest's kexts or the device's module \
                 for Ambrosia, which clang builds: install clang, then run reims-ctl update"
                    .to_string()
            },
        );
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
            format!(
                "no install record: reims-ctl update needs {} to have run",
                install::INSTALL_SCRIPT
            ),
        ),
    }
    // KVM and the library's needs, for the x86-64 VMs Linux runs; KVM alone for the arm64 VMs
    // arm64 Linux runs.
    let vms = registered_vms();
    if host.os == crate::host::Os::Linux
        && profile.accel(host) == "kvm"
        && let Some(v) = runtime::host_kernel()
    {
        let problem = runtime::kernel_problem(v);
        line(
            problem.is_none(),
            "kernel",
            problem.unwrap_or_else(|| format!("Linux {}.{}", v.0, v.1)),
        );
    }
    match host.os {
        crate::host::Os::Linux if profile.accel(host) == "kvm" && profile != Profile::Q35X86_64 => {
            let (ok, detail) = kvm_check(host, &vms);
            line(ok, "/dev/kvm", detail);
            let (ok, detail) = usb_check(profile);
            line(ok, "usb", detail);
        }
        crate::host::Os::Linux if profile == Profile::Q35X86_64 => {
            let (ok, detail) = kvm_check(host, &vms);
            line(ok, "/dev/kvm", detail);
            let udmabuf = runtime::udmabuf_problem();
            line(
                udmabuf.is_none(),
                "/dev/udmabuf",
                udmabuf.clone().unwrap_or_else(|| "usable".into()),
            );
            if let Some(n) = runtime::host_max_map_count() {
                // A speed-up only, on NVIDIA where udmabuf opens (runtime::max_map_count): for
                // the largest VM, or a new one.
                let memory_mib = vms
                    .iter()
                    .map(|vm| vm.config.memory_mib)
                    .max()
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
            let (ok, detail) = usb_check(profile);
            line(ok, "usb", detail);
            let (ok, detail) = desktop_check(
                crate::host::has_screen(&Default::default()),
                "found",
                "neither WAYLAND_DISPLAY nor DISPLAY is set",
                &vms,
            );
            line(ok, "desktop session", detail);
        }
        // The Windows Hypervisor Platform, the GPUs, the sound devices and the desktop, for the
        // x86-64 VMs Windows runs.
        #[cfg(windows)]
        crate::host::Os::Windows if profile == Profile::Q35X86_64 => windows_checks(&mut line),
        crate::host::Os::Linux
        | crate::host::Os::Mac
        | crate::host::Os::Windows
        | crate::host::Os::Other => {}
    }
    if profile == Profile::Q35X86_64 {
        // What makes a new VM's OpenCore image, with its OVMF, and downloads its installer.
        match reims_util::binary() {
            Ok(b) => {
                let v = reims_util::version(&b);
                let stale = reims_util::stale(&b);
                line(
                    v.is_some() && stale.is_none(),
                    "reims-util",
                    match (v, stale) {
                        (Some(_), Some(stale)) => stale,
                        (Some(v), None) => format!("{} ({v})", b.display()),
                        (None, _) => format!(
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

/// Whether this user can open /dev/kvm, and what doctor says of it.
fn kvm_check(host: Host, vms: &[Vm]) -> (bool, String) {
    let kvm = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/kvm");
    kvm_line(host, kvm.map(drop), vms)
}

/// doctor's /dev/kvm line, `opened` how its open went: a failure only where a VM this host runs
/// needs KVM ([`needs_kvm`]). On x86-64 Linux arm64 VMs run on TCG, which needs none.
fn kvm_line(host: Host, opened: std::io::Result<()>, vms: &[Vm]) -> (bool, String) {
    const FIX: &str = "add yourself to the kvm group (sudo usermod -aG kvm $USER) and log in again";
    match opened {
        Ok(()) => (true, "usable".into()),
        Err(e) if needs_kvm(host, vms) => (false, format!("{e}: {FIX}")),
        Err(e) => {
            let on_kvm: Vec<&str> = Profile::supported_on(host)
                .iter()
                .filter(|p| p.accel(host) == "kvm")
                .map(|p| p.id())
                .collect();
            (
                true,
                format!(
                    "{e}, and no VM here runs on KVM: {} VMs would, so for one, {FIX}",
                    on_kvm.join(" and ")
                ),
            )
        }
    }
}

/// Whether a VM this host runs needs KVM: one of `vms` whose profile runs on it here, or, with no
/// VMs yet, the one a create makes, of the host's default profile.
fn needs_kvm(host: Host, vms: &[Vm]) -> bool {
    let kvm = |p: Profile| p.accel(host) == "kvm";
    if vms.is_empty() {
        return host.default_profile().is_some_and(kvm);
    }
    vms.iter().any(|vm| kvm(vm.config.profile))
}

/// doctor's checks of a Windows host, each a `line`.
#[cfg(windows)]
fn windows_checks(line: &mut dyn FnMut(bool, &str, String)) {
    let whpx = crate::os::whpx_present();
    line(
        whpx == Some(true),
        "hypervisor platform",
        match whpx {
            Some(true) => "on, with a hypervisor running".into(),
            Some(false) => "on, but it answers that no hypervisor runs: turn virtualization \
                            (VT-x or AMD-V) on in the PC's firmware settings"
                .into(),
            None => "off, and VMs need it: run scripts\\install.ps1, or turn on Windows \
                     Hypervisor Platform in Windows Features (optionalfeatures), then restart"
                .into(),
        },
    );
    let gpus = gpu::list();
    line(
        !gpus.is_empty(),
        "vulkan",
        if gpus.is_empty() {
            "no GPU found: install your GPU's own driver, whose Vulkan the VMs use".into()
        } else {
            gpus.iter()
                .map(|g| g.label())
                .collect::<Vec<_>>()
                .join("; ")
        },
    );
    let sound = crate::os::sound_devices();
    line(
        true,
        "sound",
        match (sound.output, sound.input) {
            (true, true) => "an output device and a recording device".into(),
            (true, false) => "no recording device (a microphone): VMs record silence".into(),
            (false, _) => "no output device: VMs run without sound".into(),
        },
    );
    let (ok, detail) = desktop_check(
        crate::host::has_screen(&Default::default()),
        "a signed-in desktop, where a VM's window opens",
        "this is no signed-in desktop's session (a service's or an SSH login's)",
        &registered_vms(),
    );
    line(ok, "desktop", detail);
}

/// The VMs this user has whose settings read.
fn registered_vms() -> Vec<Vm> {
    Registry::load()
        .map(|reg| {
            reg.vms
                .iter()
                .filter_map(|e| Vm::load(&e.dir).ok())
                .collect()
        })
        .unwrap_or_default()
}

/// Whether the VM's start opens the library's window: on reims-vgpu's own device, unless its
/// variables give `REIMS_VGPU_WINDOW=0`, which keeps QEMU's device headless.
fn opens_window(vm: &Vm) -> bool {
    let env = &vm.config.extra_env;
    let headless = env.get("REIMS_VGPU_WINDOW").is_some_and(|v| v == "0");
    vm.config.graphics().library() && !headless
}

/// doctor's line on the desktop a VM's window opens on, `session` whether this process has one:
/// a failure only where a VM of `vms` would open a window and finds no screen (its own
/// variables can name one). A host whose VMs run headless, or that has none yet, needs none.
fn desktop_check(session: bool, found: &str, missing: &str, vms: &[Vm]) -> (bool, String) {
    if session {
        return (true, found.into());
    }
    let blind: Vec<&str> = vms
        .iter()
        .filter(|vm| opens_window(vm) && !crate::host::has_screen(&vm.config.extra_env))
        .map(|vm| vm.name())
        .collect();
    match blind.as_slice() {
        [] if vms.is_empty() => (
            true,
            format!("{missing}: a VM's window needs a desktop session, and there are no VMs yet"),
        ),
        [] => (
            true,
            format!(
                "{missing}, and no VM here needs one: each runs headless or names a screen in its \
                 variables"
            ),
        ),
        _ => (
            false,
            format!(
                "{missing}, and {} open{} a window, which needs a desktop session: start {} from \
                 one, or headless (reims-ctl set NAME --env REIMS_VGPU_WINDOW=0)",
                blind.join(", "),
                if blind.len() == 1 { "s" } else { "" },
                if blind.len() == 1 { "it" } else { "them" },
            ),
        ),
    }
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

    /// `create --help`'s examples head the x86-64 guest's by the host's own system where it
    /// makes that guest, Windows on Windows, and by the hosts that do elsewhere; on Windows their
    /// paths have `\` alone. The examples are the same on every host.
    #[test]
    fn create_examples_name_the_host_that_makes_each_guest() {
        use crate::host::{Arch, Os};
        let on = |os, arch| create_examples_on(Host { os, arch });
        let windows = on(Os::Windows, Arch::X86_64);
        assert!(windows.contains("An x86-64 guest on Windows:"), "{windows}");
        assert!(!windows.contains("on Linux:"), "{windows}");
        assert!(!windows.contains('/'), "{windows}");
        assert!(
            windows.contains(r"--installer ~\Downloads\BaseSystem.dmg"),
            "{windows}"
        );
        let linux = on(Os::Linux, Arch::X86_64);
        assert!(linux.contains("An x86-64 guest on Linux:"), "{linux}");
        assert!(
            linux.contains("--installer ~/Downloads/BaseSystem.dmg"),
            "{linux}"
        );
        assert_eq!(
            windows
                .replace('\\', "/")
                .replace("on Windows:", "on Linux:"),
            linux
        );
        for (os, arch) in [(Os::Mac, Arch::Aarch64), (Os::Linux, Arch::Aarch64)] {
            let e = on(os, arch);
            assert!(
                e.contains("An x86-64 guest on x86-64 Linux or Windows:"),
                "{e}"
            );
            assert!(
                e.contains("An arm64 guest on an arm64 Mac, arm64 Linux or x86-64 Linux:"),
                "{e}"
            );
        }
    }

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
            &config("c", &crate::testenv::toml_path(&shared)),
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

    /// A VM's settings, for tests: `graphics` its graphics line, or empty for none.
    fn config(graphics: &str) -> String {
        format!(
            r#"
            name = "t"
            profile = "q35-x86_64"
            cpus = 2
            memory_mib = 4096
            disk = "disk.qcow2"
            {graphics}
            [q35]
            opencore = "opencore.qcow2"
            firmware_code = "OVMF_CODE.fd"
            firmware_vars = "OVMF_VARS.fd"
            "#
        )
    }

    fn set_args(argv: &[&str]) -> SetArgs {
        match Cli::try_parse_from(argv).unwrap().command {
            Some(Cmd::Set(a)) => *a,
            _ => unreachable!(),
        }
    }

    /// `set --displays` takes 1 to 4 on reims-vgpu, and refuses more than one on VMware SVGA,
    /// which has one; a refused set saves nothing.
    #[test]
    fn set_takes_reims_vgpu_s_displays_and_refuses_vmware_svga_more_than_one() {
        let env = Env::new("set-displays");
        let dir = env.root.join("vms/t");
        register("t", &dir, &config(""));
        let count = || Vm::load(&dir).unwrap().config.display.count;
        let host = Host::current();
        set(
            set_args(&["reims-ctl", "set", "t", "--displays", "2"]),
            host,
        )
        .unwrap();
        assert_eq!(count(), 2);
        for bad in ["0", "5"] {
            let e = set(
                set_args(&["reims-ctl", "set", "t", "--displays", bad]),
                host,
            );
            let e = format!("{:#}", e.unwrap_err());
            assert!(e.contains("1 to 4 displays"), "{bad}: {e}");
            assert_eq!(count(), 2, "{bad}");
        }
        std::fs::write(
            dir.join(vm::CONFIG_FILE),
            config("graphics = \"vmware-svga\""),
        )
        .unwrap();
        let e = set(set_args(&["reims-ctl", "set", "t", "--displays=3"]), host);
        let e = format!("{:#}", e.unwrap_err());
        assert!(e.contains("VMware SVGA has one display"), "{e}");
        assert_eq!(count(), 1);
        set(
            set_args(&["reims-ctl", "set", "t", "--displays", "1"]),
            host,
        )
        .unwrap();
        assert_eq!(count(), 1);
    }

    /// A VM's summary and its printed command say how many windows it opens: one, as before
    /// the setting, or one for each display.
    #[test]
    fn a_vms_summary_says_how_many_windows_it_opens() {
        let mut vm = Vm {
            dir: "/nonexistent/reims-ctl/t".into(),
            config: toml::from_str(&config("")).unwrap(),
        };
        let host = Host::current();
        let line = |vm: &Vm| {
            describe(vm, host)
                .into_iter()
                .find(|(k, _)| k.starts_with("display"))
                .unwrap()
        };
        let pair = |k: &str, v: &str| (k.to_string(), v.to_string());
        assert_eq!(
            line(&vm),
            pair(
                "display",
                "one, in a window opening at 1920x1080, whose size the guest follows"
            )
        );
        assert!(script_comment(&vm).contains("\nIts window opens under -display none;"));
        vm.config.display.count = 2;
        vm.config.display.resolution = "2560x1440".into();
        assert_eq!(
            line(&vm),
            pair(
                "displays",
                "2, in 2 windows opening at 2560x1440, whose sizes the guest follows"
            )
        );
        assert!(
            script_comment(&vm)
                .contains("\nIts 2 windows, one for each display, open under -display none;"),
            "{}",
            script_comment(&vm)
        );
        vm.config.display.borderless = true;
        assert_eq!(
            line(&vm),
            pair(
                "displays",
                "2, in 2 borderless windows, each filling its screen"
            )
        );
        vm.config.display.count = 1;
        assert_eq!(
            line(&vm),
            pair("display", "one, filling its screen, borderless")
        );
        // VMware SVGA's one display is QEMU's window, which the graphics line names.
        vm.config.display.count = 2;
        vm.config.graphics = Some(vm::Graphics::VmwareSvga);
        assert!(
            !describe(&vm, host)
                .iter()
                .any(|(k, _)| k.starts_with("display"))
        );
        assert!(!script_comment(&vm).contains("windows"));
    }

    /// A VM, `name`, of `graphics` (a settings line, or empty), whose variables name no screen
    /// whatever this process's are, with `env` over that.
    #[cfg(unix)]
    fn screenless_vm(name: &str, graphics: &str, env: &[(&str, &str)]) -> Vm {
        let mut config: vm::VmConfig = toml::from_str(&config(graphics)).unwrap();
        config.name = name.into();
        config.extra_env = [("WAYLAND_DISPLAY", ""), ("DISPLAY", "")]
            .iter()
            .chain(env)
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        Vm {
            dir: format!("/nonexistent/reims-ctl/{name}").into(),
            config,
        }
    }

    /// doctor fails on the desktop only where a VM would open a window and finds no screen: one
    /// on reims-vgpu's own device without REIMS_VGPU_WINDOW=0. VMs started headless, VMware
    /// SVGA's (QEMU's window, left out without a screen), one whose own variables name a screen,
    /// and no VMs at all need no session of this process's.
    #[cfg(unix)]
    #[test]
    fn doctor_wants_a_desktop_only_for_a_vms_window() {
        let check = |session, vms: &[Vm]| desktop_check(session, "found", "no session", vms);
        let windowed = screenless_vm("w", "", &[]);
        let headless = screenless_vm("h", "", &[("REIMS_VGPU_WINDOW", "0")]);
        let svga = screenless_vm("s", "graphics = \"vmware-svga\"", &[]);
        let shown = screenless_vm("d", "", &[("DISPLAY", ":1")]);
        assert!(opens_window(&windowed) && opens_window(&shown));
        assert!(!opens_window(&headless) && !opens_window(&svga));

        let found = check(true, std::slice::from_ref(&windowed));
        assert_eq!(found, (true, "found".to_string()));
        let (ok, detail) = check(false, &[]);
        assert!(ok && detail.contains("there are no VMs yet"), "{detail}");
        let (ok, detail) = check(false, &[headless.clone(), svga.clone(), shown.clone()]);
        assert!(ok && detail.contains("no VM here needs one"), "{detail}");
        let (ok, detail) = check(false, &[headless.clone(), windowed.clone()]);
        assert!(!ok, "{detail}");
        assert!(
            detail.starts_with("no session, and w opens a window, which needs a desktop session"),
            "{detail}"
        );
        assert!(detail.contains("REIMS_VGPU_WINDOW=0"), "{detail}");
        let (ok, detail) = check(false, &[windowed, screenless_vm("x", "", &[])]);
        assert!(!ok && detail.contains("w, x open a window"), "{detail}");
    }

    /// doctor fails on /dev/kvm only where a VM this host runs needs KVM: on x86-64 Linux an
    /// x86-64 VM, or the x86-64 VM a create makes where there are none yet; arm64 VMs run on TCG
    /// there. On arm64 Linux every VM runs on KVM.
    #[test]
    fn doctor_wants_kvm_only_where_a_vm_runs_on_it() {
        use crate::host::{Arch, Os};
        let x86 = Host {
            os: Os::Linux,
            arch: Arch::X86_64,
        };
        let arm = Host {
            os: Os::Linux,
            arch: Arch::Aarch64,
        };
        let vm_of = |profile: Profile| {
            let mut config: vm::VmConfig = toml::from_str(&config("")).unwrap();
            config.profile = profile;
            Vm {
                dir: "/nonexistent/reims-ctl/t".into(),
                config,
            }
        };
        let (q35, arm64) = (vm_of(Profile::Q35X86_64), vm_of(Profile::VmappleArm64));
        assert!(needs_kvm(x86, &[]) && needs_kvm(arm, &[]));
        assert!(needs_kvm(x86, &[arm64.clone(), q35.clone()]));
        assert!(!needs_kvm(x86, std::slice::from_ref(&arm64)));
        assert!(needs_kvm(arm, std::slice::from_ref(&arm64)));

        let denied = || Err(std::io::Error::from(std::io::ErrorKind::PermissionDenied));
        assert_eq!(kvm_line(x86, Ok(()), &[]), (true, "usable".to_string()));
        let (ok, detail) = kvm_line(x86, denied(), &[q35]);
        assert!(
            !ok && detail.ends_with("kvm $USER) and log in again"),
            "{detail}"
        );
        let (ok, detail) = kvm_line(x86, denied(), std::slice::from_ref(&arm64));
        assert!(ok, "{detail}");
        assert!(
            detail.contains("no VM here runs on KVM: q35-x86_64 VMs would, so for one, add"),
            "{detail}"
        );
        assert!(!kvm_line(arm, denied(), &[arm64]).0);
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

    /// On Windows a running VM's QEMU holds the VM's copy of its installer open without
    /// FILE_SHARE_DELETE (block/file-win32.c), so that no new copy replaces it and it cannot be
    /// deleted: a DMG is refused before its conversion, which writes gigabytes, and a change
    /// that would delete the copy says that it stays.
    #[cfg(windows)]
    #[test]
    fn a_held_installer_copy_refuses_a_dmg_and_is_said_to_stay() {
        use std::os::windows::fs::OpenOptionsExt;
        use windows_sys::Win32::Storage::FileSystem::{FILE_SHARE_READ, FILE_SHARE_WRITE};
        let d = std::env::temp_dir().join(format!("reims-ctl-held-{}", std::process::id()));
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
        std::fs::write(&copy, [0u8; 4096]).unwrap();
        // A UDIF image: it ends in a 512-byte trailer starting `koly`.
        let dmg = d.join("BaseSystem.dmg");
        let mut udif = vec![0u8; 4096];
        udif[4096 - 512..4096 - 508].copy_from_slice(b"koly");
        std::fs::write(&dmg, &udif).unwrap();
        // Opened as QEMU opens a drive's file.
        let held = std::fs::OpenOptions::new()
            .read(true)
            .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE)
            .open(&copy)
            .unwrap();
        let installer = |vm: &Vm| vm.config.q35.as_ref().unwrap().installer.clone();
        let mut quiet = |_: &str| {};
        let e = set_installer(&mut vm, Some(dmg), &mut quiet)
            .unwrap_err()
            .to_string();
        assert!(
            e.ends_with("so a new copy cannot replace it: stop the VM first"),
            "{e}"
        );
        assert!(!d.join("vm").join("installer.img.part").exists());
        let saved = Vm::load(&d.join("vm")).unwrap();
        assert_eq!(installer(&saved), Some(create::INSTALLER_COPY.into()));
        // An eject takes it out of the settings, and says the copy stays.
        let said = set_installer(&mut vm, None, &mut quiet).unwrap();
        assert!(
            said.starts_with("t: installer removed; its old copy "),
            "{said}"
        );
        assert!(said.contains(" could not be deleted: "), "{said}");
        assert_eq!(installer(&vm), None);
        assert!(copy.is_file());
        drop(held);
        std::fs::remove_dir_all(&d).unwrap();
    }
}
