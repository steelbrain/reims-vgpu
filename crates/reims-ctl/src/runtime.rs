//! Starting, finding and stopping a VM's QEMU.
//!
//! QEMU runs detached: in a session of its own (so closing the terminal does not end it), and
//! reparented away from reims-ctl by a double fork (so a TUI that stays open never holds an
//! exited QEMU as a zombie). Its pid is the one QEMU writes to the VM's `qemu.pid`, counted only
//! while that process runs and its command line names the pid file, so a stale file or a reused
//! pid is never taken for the VM. A VM started from a printed script is found the same way.
//!
//! On Linux, QEMU runs in a systemd user scope of its own, `reims-ctl-NAME.scope`, where the
//! user's manager answers and outlasts the start ([`launch`]); elsewhere, and where none does, in
//! the cgroup of the process that starts it, with a warning on Linux.
//!
//! QEMU's stdout and stderr go to `logs/qemu.log`, kept for the last five starts
//! (`qemu.1.log`, ...), the serial console's likewise.

use crate::host::{Host, Os, Profile};
use crate::qemu::{self, Ctx, Invocation};
use crate::qmp;
use crate::vm::{DEFAULT_STOP_TIMEOUT, DiskController, Graphics, Vm};
use anyhow::{Context, Result, bail};
use serde_json::json;
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::time::{Duration, Instant};

/// QEMU's pid, while it runs.
pub fn pid(vm: &Vm) -> Option<i32> {
    pid_at(&vm.pidfile())
}

/// The pid of the QEMU that writes `pidfile`, while it runs: what [`pid`] reads, for a VM whose
/// settings may not read.
pub fn pid_at(pidfile: &Path) -> Option<i32> {
    let pid: i32 = std::fs::read_to_string(pidfile).ok()?.trim().parse().ok()?;
    if pid <= 0 || !alive(pid) {
        return None;
    }
    let needle = pidfile.as_os_str().as_encoded_bytes();
    let cmd = cmdline(pid)?;
    cmd.windows(needle.len())
        .any(|w| w == needle)
        .then_some(pid)
}

fn alive(pid: i32) -> bool {
    // SAFETY: signal 0 only checks the pid.
    let r = unsafe { libc::kill(pid, 0) };
    r == 0 || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

#[cfg(target_os = "linux")]
fn cmdline(pid: i32) -> Option<Vec<u8>> {
    let b = std::fs::read(format!("/proc/{pid}/cmdline")).ok()?;
    // A zombie's command line is empty.
    (!b.is_empty()).then_some(b)
}

#[cfg(target_os = "macos")]
fn cmdline(pid: i32) -> Option<Vec<u8>> {
    let mut mib = [libc::CTL_KERN, libc::KERN_PROCARGS2, pid];
    let mut size: libc::size_t = 0;
    // SAFETY: a size query; mib names a valid sysctl.
    if unsafe {
        libc::sysctl(
            mib.as_mut_ptr(),
            3,
            std::ptr::null_mut(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    } != 0
    {
        return None;
    }
    let mut buf = vec![0u8; size];
    // SAFETY: buf holds size bytes.
    if unsafe {
        libc::sysctl(
            mib.as_mut_ptr(),
            3,
            buf.as_mut_ptr().cast(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    } != 0
    {
        return None;
    }
    buf.truncate(size);
    Some(buf)
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn cmdline(_pid: i32) -> Option<Vec<u8>> {
    None
}

pub fn running(vm: &Vm) -> bool {
    pid(vm).is_some()
}

/// What is wrong with the host or the VM's files for a start, if anything.
pub fn preflight(vm: &Vm, host: Host) -> Result<()> {
    let c = &vm.config;
    c.check()?;
    if !crate::host::Profile::supported_on(host).contains(&c.profile) {
        bail!("{}", crate::host::UNSUPPORTED);
    }
    let need = |p: &Path, what: &str| -> Result<()> {
        if !p.exists() {
            bail!("{what} {} is missing", p.display());
        }
        Ok(())
    };
    need(&vm.disk(), "the disk")?;
    match c.profile {
        Profile::Q35X86_64 => {
            let q = c.q35.as_ref().unwrap();
            need(&vm.path(&q.opencore), "the OpenCore image")?;
            need(&vm.path(&q.firmware_code), "the firmware")?;
            need(&vm.path(&q.firmware_vars), "the firmware variables")?;
            if let Some(i) = q.installer.as_ref().map(|i| vm.path(i))
                && !i.exists()
            {
                bail!(
                    "the installer {} is missing: reims-ctl eject {} takes it out (in the \
                     interface, clear Installer image in its edit form, e)",
                    i.display(),
                    c.name
                );
            }
        }
    }
    if host.os == Os::Linux {
        let kvm = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open("/dev/kvm");
        if let Err(e) = kvm {
            bail!(
                "/dev/kvm cannot be opened ({e}): enable virtualization in the firmware, and add \
                 yourself to the kvm group (sudo usermod -aG kvm $USER, then log in again)"
            );
        }
    }
    Ok(())
}

/// The host's `vm.max_map_count` (Linux).
pub fn host_max_map_count() -> Option<u64> {
    std::fs::read_to_string("/proc/sys/vm/max_map_count")
        .ok()
        .and_then(|s| s.trim().parse().ok())
}

/// The file of MGLRU's thrashing guard (Linux; the kernel's
/// Documentation/admin-guide/mm/multigen_lru.rst, "Thrashing prevention"): the pages used in the
/// last that many milliseconds are kept from eviction, and where they can't be kept in memory the
/// kernel's OOM killer runs. Measured on this project's CachyOS hosts, whose kernel defaults it to
/// 100: kswapd invoked the global OOM killer with most RAM free, killing a compiler with 19 GiB
/// free on one, and making 19 of a day's 22 OOM kills on another. A VM's QEMU can be the one
/// killed; scripts/install-ctl.sh and `reims-ctl doctor` offer to set 0, which turns the guard
/// off, while memory that does run out still ends in the OOM killer.
pub const MIN_TTL_FILE: &str = "/sys/kernel/mm/lru_gen/min_ttl_ms";

/// The systemd-tmpfiles line that writes 0 to [`MIN_TTL_FILE`] at each boot, which
/// scripts/install-ctl.sh shows too.
pub const MIN_TTL_TMPFILES: &str = "w /sys/kernel/mm/lru_gen/min_ttl_ms - - - - 0";

/// The thrashing guard's milliseconds in `file` ([`MIN_TTL_FILE`]); None where it isn't there.
pub fn min_ttl_ms(file: &Path) -> Option<u64> {
    std::fs::read_to_string(file)
        .ok()
        .and_then(|s| s.trim().parse().ok())
}

/// What the thrashing guard at `ms` does to VMs, for a value above 0; None at 0.
pub fn min_ttl_note(ms: u64) -> Option<String> {
    (ms > 0).then(|| {
        format!(
            "{ms}: where the kernel can't keep the pages used in the last {ms} ms in memory, it \
             kills a process, which can be a VM's QEMU, even while RAM is free (measured on hosts \
             at 100, killing builds with most RAM free); 0 turns this off"
        )
    })
}

/// How to keep the thrashing guard off after a restart, and, where it is not `off` yet, how to
/// turn it off now, as scripts/install-ctl.sh says it.
pub fn min_ttl_fix(off: bool) -> String {
    let keep = format!(
        "the line {MIN_TTL_TMPFILES} in a file in /etc/tmpfiles.d, such as \
         /etc/tmpfiles.d/reims-ctl-min-ttl.conf"
    );
    if off {
        format!("it's 0 now; {keep}, keeps it at 0 after a restart")
    } else {
        format!(
            "echo 0 | sudo tee {MIN_TTL_FILE} turns it off now, and {keep}, keeps it at 0 after \
             a restart"
        )
    }
}

/// The `vm.max_map_count` at which an NVIDIA GPU takes all of a guest of `memory_mib` without
/// copies, and the value to suggest raising it to; scripts/install-ctl.sh keeps the same rule.
/// docs/qemu.md, "Linux hosts: vm.max_map_count": a view of an x86-64 guest's memory mapped from
/// its files takes a mapping per 4 KiB page, which a GPU that pins a pointer's pages (NVIDIA's)
/// needs; past the library's budget, the limit less 16,384 kept for the rest of the process and
/// 8,192 more, its buffers are copied instead, which works and is slower. Other GPUs take the
/// views through udmabuf, a mapping per 4 MiB, so the limit is only a speed-up, and only there.
pub fn max_map_count(memory_mib: u64) -> (u64, u64) {
    let want = memory_mib * 256 + 16_384 + 8_192;
    (want, want.next_power_of_two().max(1 << 20))
}

/// What keeps the library from opening `/dev/udmabuf`, which every Linux host needs: the guest's
/// memory reaches the GPU through it, and without it the library's views of that memory take a
/// kernel mapping per 4 KiB page, past `vm.max_map_count` at 3840 x 2160 even at the guest's
/// desktop (docs/qemu.md, "Linux hosts: vm.max_map_count"). None where it opens.
pub fn udmabuf_problem() -> Option<String> {
    use std::os::unix::fs::MetadataExt;
    const NODE: &str = "/dev/udmabuf";
    let e = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(NODE)
        .err()?;
    let group = std::fs::metadata(NODE).ok().map(|m| m.gid());
    let over_ssh = std::env::var_os("SSH_CONNECTION").is_some();
    Some(udmabuf_message(
        &e,
        group.map(|gid| (gid, group_name(gid))),
        over_ssh,
    ))
}

/// The name of group `gid`, from /etc/group, else its number.
fn group_name(gid: u32) -> String {
    std::fs::read_to_string("/etc/group")
        .ok()
        .and_then(|groups| group_in(&groups, gid))
        .unwrap_or_else(|| gid.to_string())
}

/// The name `groups`, in /etc/group's form, gives group `gid`.
fn group_in(groups: &str, gid: u32) -> Option<String> {
    groups.lines().find_map(|l| {
        let mut f = l.split(':');
        let name = f.next()?;
        (f.nth(1)?.parse() == Ok(gid)).then(|| name.to_owned())
    })
}

/// [`udmabuf_problem`]'s message for `e`, the node's group (its id and name) where it exists, and
/// whether reims-ctl runs over ssh.
fn udmabuf_message(e: &std::io::Error, group: Option<(u32, String)>, over_ssh: bool) -> String {
    const WHY: &str = "the guest's memory reaches the GPU through it, and without it the guest \
                       can crash at higher resolutions, at 4K even at its desktop";
    let load = "Load it with sudo modprobe udmabuf, and put udmabuf in a file in \
                /etc/modules-load.d to keep it";
    let Some((gid, name)) = group else {
        return format!("There's no /dev/udmabuf: {WHY}. {load}");
    };
    if e.kind() != std::io::ErrorKind::PermissionDenied {
        return format!("/dev/udmabuf doesn't open ({e}): {WHY}. {load}");
    }
    if gid == 0 {
        return format!(
            "Only root can open /dev/udmabuf ({e}): {WHY}. Give it to the kvm group with a udev \
             rule, a file in /etc/udev/rules.d holding KERNEL==\"udmabuf\", GROUP=\"kvm\", \
             MODE=\"0660\", add yourself to kvm (sudo usermod -aG kvm $USER), and restart"
        );
    }
    let ssh = if over_ssh {
        ". Over ssh, the access a desktop session gets to it doesn't apply, so you need the \
         group there"
    } else {
        ""
    };
    format!(
        "You can't open /dev/udmabuf ({e}): {WHY}. Add yourself to the {name} group that owns it \
         (sudo usermod -aG {name} $USER) and log in again{ssh}"
    )
}

/// Whether the host has an NVIDIA GPU (Linux): a card of its PCI vendor id.
pub fn host_has_nvidia() -> bool {
    std::fs::read_dir("/sys/class/drm")
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|card| std::fs::read_to_string(card.path().join("device/vendor")).ok())
        .any(|v| v.trim() == "0x10de")
}

/// What `vm.max_map_count` costs a host whose GPU is NVIDIA's, for `reims-ctl doctor`: where
/// `/dev/udmabuf` opens (else [`udmabuf_problem`] says what to do) and the limit falls short of
/// [`max_map_count`] for a guest of `memory_mib`, how much of it stays uncopied and the value
/// that keeps all of it so. A speed-up only: the guest works either way.
pub fn map_count_note(n: u64, memory_mib: u64) -> Option<String> {
    let (want, raise) = max_map_count(memory_mib);
    if n >= want {
        return None;
    }
    let uncopied = n.saturating_sub(16_384 + 8_192) / 256;
    Some(format!(
        "on an NVIDIA GPU, past about {uncopied} MiB of a {} GiB guest's memory the GPU gets \
         copies of it, which is slower; sudo sysctl -w vm.max_map_count={raise}, and a file in \
         /etc/sysctl.d to keep it, avoids them",
        memory_mib / 1024
    ))
}

/// How to install macOS in an x86-64 VM that has an installer, for the CLI and the interface's
/// details; None for other VMs. An offline installer, which reims-util names
/// `…-offline-installer.img`, installs from its script; any other installer boots Recovery,
/// whose Reinstall macOS installs it. For a disk on NVMe, they first say to move it to SATA
/// ([`DiskController`]). An offline installer holds its release whole, so its steps say to
/// install with the VM's network off, which keeps macOS's installer from downloading an update
/// as it installs, and to turn the network back on afterwards.
pub fn install_steps(vm: &Vm) -> Option<String> {
    let q = vm.config.q35.as_ref()?;
    let installer = q.installer.as_ref()?;
    let offline = installer
        .file_name()
        .is_some_and(|n| n.to_string_lossy().ends_with("-offline-installer.img"));
    let last = if offline {
        "open Utilities > Terminal and run: sh /Volumes/Installer/install-macos.sh"
    } else {
        "pick Reinstall macOS"
    };
    let name = vm.name();
    let first = match q.disk_controller {
        DiskController::Ahci => String::new(),
        DiskController::Nvme => format!(
            "Its disk is on NVMe, where macOS's installer may find no disk: first move it to \
             SATA (reims-ctl set {name} --disk-controller ahci, or Disk controller in its edit \
             form). "
        ),
    };
    let network_off = if offline && vm.config.network.enabled {
        format!(
            "Its installer holds the whole release, so turn the VM's network off before you \
             start it (reims-ctl set {name} --network false, or Network in its edit form): with \
             a network, macOS's installer can download an update as it installs, a second \
             download of the release. "
        )
    } else {
        String::new()
    };
    let network_on = if offline {
        format!(" and turn its network back on (reims-ctl set {name} --network)")
    } else {
        String::new()
    };
    Some(format!(
        "{first}{network_off}To install macOS, start {name}: OpenCore boots its installer into \
         macOS Recovery. If the VM's disk is new, open Disk Utility, erase it (QEMU HARDDISK \
         Media) as APFS and quit Disk Utility. Then {last}. Once macOS is installed, eject the \
         installer (reims-ctl eject {name}){network_on}."
    ))
}

/// Warnings about the host, or the guest's last boot, for a start that would still work. Each is
/// about the library, which a VM on VMware SVGA does not run.
pub fn warnings(vm: &Vm, host: Host) -> Vec<String> {
    let mut w = Vec::new();
    let graphics = vm.config.graphics();
    if !graphics.library() {
        return w;
    }
    if graphics == Graphics::ReimsVgpu {
        w.extend(crate::opencore::missing_guest_kexts(vm));
    }
    if host.os == Os::Linux
        && let Some(problem) = udmabuf_problem()
    {
        w.push(problem);
    }
    w
}

/// Keeps the last five of a log: `x.log` becomes `x.1.log`, and so on.
fn rotate(path: &Path) {
    let stem = path.file_stem().unwrap().to_string_lossy().into_owned();
    let dir = path.parent().unwrap();
    let nth = |n: u32| dir.join(format!("{stem}.{n}.log"));
    let _ = std::fs::remove_file(nth(4));
    for n in (1..4).rev() {
        let _ = std::fs::rename(nth(n), nth(n + 1));
    }
    let _ = std::fs::rename(path, nth(1));
}

pub fn context(vm: &Vm, host: Host) -> Result<Ctx> {
    let qemu = crate::install::qemu_binary(vm.config.profile)?;
    let backends = crate::install::audio_backends(&qemu);
    let audio_backend = crate::install::default_audio_backend(&backends);
    if vm.config.audio.enabled
        && let Some(b) = &vm.config.audio.backend
        && !backends.is_empty()
        && !backends.contains(b)
    {
        bail!(
            "this QEMU has no {b} audio backend; it has {}",
            backends.join(", ")
        );
    }
    let screen = crate::host::has_screen(&vm.config.extra_env);
    if vm.config.graphics() == Graphics::VmwareSvga && screen {
        let displays = crate::install::display_backends(&qemu);
        if !displays.is_empty() && !displays.iter().any(|d| d == "gtk") {
            bail!(
                "this QEMU has no window of its own, which VMware SVGA shows the display in: \
                 reims-ctl update builds one with GTK"
            );
        }
    }
    // QEMU's configure leaves usb-host out where it finds no libusb, as an install before
    // scripts/install-ctl.sh named it did.
    if !vm.config.usb.is_empty() && crate::install::has_device(&qemu, "usb-host") == Some(false) {
        bail!(
            "this QEMU has no usb-host, which passes USB devices through: reims-ctl update builds \
             one with it"
        );
    }
    Ok(Ctx {
        host,
        qemu,
        audio_backend,
        nested_vmx: crate::host::kvm_nested_vmx(),
        screen,
        graphics_rom: graphics_rom(vm)?,
    })
}

/// The option ROM of reims-vgpu's own device, written into the data directory for a VM on it
/// ([`crate::prebuilt::graphics_rom`]); an empty path, which no command names, for another.
fn graphics_rom(vm: &Vm) -> Result<std::path::PathBuf> {
    if vm.config.graphics() == Graphics::ReimsVgpu {
        crate::prebuilt::graphics_rom()
    } else {
        Ok(std::path::PathBuf::new())
    }
}

/// The invocation `start` would run, with QEMU's real path where it is installed.
pub fn invocation_for_print(vm: &Vm, host: Host) -> Result<Invocation> {
    let ctx = context(vm, host).unwrap_or_else(|_| {
        let qemu = crate::install::qemu_binary_or_default(vm.config.profile);
        Ctx {
            host,
            qemu,
            audio_backend: crate::install::default_audio_backend(&[]),
            nested_vmx: crate::host::kvm_nested_vmx(),
            screen: crate::host::has_screen(&vm.config.extra_env),
            graphics_rom: graphics_rom(vm).unwrap_or_else(|_| crate::prebuilt::graphics_rom_path()),
        }
    });
    qemu::invocation(vm, &ctx)
}

pub fn now_stamp(fmt: &str) -> String {
    // SAFETY: time and localtime_r fill the structs given; strftime writes at most buf's size.
    unsafe {
        let t = libc::time(std::ptr::null_mut());
        let mut tm: libc::tm = std::mem::zeroed();
        libc::localtime_r(&t, &mut tm);
        let mut buf = [0u8; 64];
        let f = std::ffi::CString::new(fmt).unwrap();
        let n = libc::strftime(buf.as_mut_ptr().cast(), buf.len(), f.as_ptr(), &tm);
        String::from_utf8_lossy(&buf[..n]).into_owned()
    }
}

/// Another VM that runs on `vm`'s disk, which both use in place. QEMU on Linux would refuse the
/// second's start on its lock of the image; on macOS QEMU locks no image, and both would write
/// the disk.
fn running_on_disk(vm: &Vm) -> Option<String> {
    let real = |p: &Path| std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
    let disk = real(&vm.disk());
    let reg = crate::vm::Registry::load().ok()?;
    reg.vms
        .iter()
        .filter(|e| e.name != vm.name())
        .filter_map(|e| Vm::load(&e.dir).ok())
        .find(|other| real(&other.disk()) == disk && pid(other).is_some())
        .map(|other| other.name().to_string())
}

/// Readies the VM's directory for a run of QEMU: its logs rotated, and the QMP socket's
/// directory made. Returns the socket.
///
/// The pid file and the socket an earlier run left stay where they are. QEMU locks its pid file
/// before it opens anything else, so a second QEMU of a VM that already runs, from a start
/// racing this one, ends there, before it opens the VM's disks or replaces the first one's
/// socket; and QEMU replaces a stale socket itself. Deleted here, the second QEMU would lock a
/// new file and run on the disks the first one writes, which on macOS, where QEMU locks no
/// image, nothing else stops.
fn prepare_run(vm: &Vm) -> Result<std::path::PathBuf> {
    std::fs::create_dir_all(vm.logs_dir())?;
    rotate(&vm.qemu_log());
    if vm.config.logs.enabled {
        rotate(&vm.serial_log());
    }
    let sock = vm.qmp_socket();
    if let Some(d) = sock.parent().filter(|d| *d != vm.dir) {
        private_dir(d)?;
    }
    Ok(sock)
}

/// Makes `d`, a directory for QMP sockets outside the VMs' own, the user's alone (0700) where it
/// is new, and refuses one that is not a directory, another user owns, or others may write into:
/// whoever could write there could put a socket of theirs in QMP's place and drive the VM. Some
/// of them are under `/tmp`, which every user writes into (`paths::short_runtime_dir`).
fn private_dir(d: &Path) -> Result<()> {
    use std::os::unix::fs::{DirBuilderExt, MetadataExt};
    std::fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(d)
        .with_context(|| format!("create {}", d.display()))?;
    let m = std::fs::symlink_metadata(d).with_context(|| format!("read {}", d.display()))?;
    // SAFETY: getuid has no preconditions.
    let uid = unsafe { libc::getuid() };
    if !m.is_dir() || m.uid() != uid || m.mode() & 0o022 != 0 {
        bail!(
            "{} is not a directory of this user's that others cannot write into, so it holds no \
             QMP socket: remove it, or make it so",
            d.display()
        );
    }
    Ok(())
}

/// What a start of a stopped VM does to its settings and OpenCore image first, before [`start`]:
/// an update's change that waited while it ran ([`crate::migrate::at_start`]); then a new image
/// ([`crate::opencore::renew`]) for one on another device whose image loads reims-vgpu's guest kexts, as an image made for
/// reims-vgpu does after a change of graphics, without them; and for one on reims-vgpu whose
/// image holds other builds of them, as one an older reims-ctl made does after an update, with
/// this reims-ctl's ([`crate::opencore::holds_other_guest_kexts`]). An image reims-vgpu's kexts
/// lack is warned of instead (`warnings`): the image may be one the user gave, which a new one
/// would not carry over. Notes go to `progress`, the new image's with what the kexts left in the
/// guest; warnings are returned, for a start that still runs.
pub fn prepare(vm: &mut Vm, progress: &mut dyn FnMut(&str)) -> Vec<String> {
    let mut w = Vec::new();
    if running(vm) {
        return w;
    }
    // An update's change that waited while the VM ran (crate::migrate).
    crate::migrate::at_start(vm, progress);
    let name = vm.name().to_string();
    if crate::opencore::holds_unused_guest_kexts(vm) {
        let graphics = vm.config.graphics().name();
        progress(&format!(
            "its OpenCore image loads reims-vgpu's macOS drivers, which {graphics} has no use \
             for: making it a new one without them, presenting the same Mac"
        ));
        match crate::opencore::renew(vm, None, progress) {
            Ok(s) => progress(s.strip_prefix(&format!("{name}: ")).unwrap_or(&s)),
            Err(e) => w.push(format!(
                "{name}'s OpenCore image loads reims-vgpu's macOS drivers, which {graphics} has no \
                 use for, and a new one without them couldn't be made ({e:#}): reims-ctl \
                 update-opencore {name} makes it"
            )),
        }
    } else if crate::opencore::holds_other_guest_kexts(vm) {
        progress(
            "its OpenCore image holds other builds of reims-vgpu's macOS drivers than this \
             reims-ctl carries: making it a new one with these, presenting the same Mac",
        );
        match crate::opencore::renew(vm, None, progress) {
            Ok(s) => progress(s.strip_prefix(&format!("{name}: ")).unwrap_or(&s)),
            Err(e) => w.push(format!(
                "{name}'s OpenCore image holds other builds of reims-vgpu's macOS drivers than \
                 this reims-ctl carries, and a new one with these couldn't be made ({e:#}): \
                 reims-ctl update-opencore {name} makes it"
            )),
        }
    }
    w
}

/// The systemd user unit a start runs a VM's QEMU in ([`launch`]): one per VM, so a start of a
/// VM whose QEMU still runs in it finds the name taken.
pub fn scope_unit(name: &str) -> String {
    format!("reims-ctl-{name}.scope")
}

/// What a start reads of the user's systemd service manager ([`launch`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UserManager {
    /// `systemd-run` is on the PATH and the user's manager answers on its bus.
    pub answers: bool,
    /// The caller runs under that manager (its cgroup is below `user@UID.service`), as a
    /// desktop's terminal does: a scope there lasts as long as the caller's own processes.
    pub caller_inside: bool,
    /// The manager stays after the user's last session ends (`loginctl enable-linger`).
    pub lingers: bool,
}

impl UserManager {
    /// This host's, read now: on Linux by `systemctl --user` and `loginctl`, and none
    /// elsewhere.
    pub fn current() -> UserManager {
        if !cfg!(target_os = "linux") {
            return UserManager::default();
        }
        let ok = |program: &str, args: &[&str]| {
            std::process::Command::new(program)
                .args(args)
                .stdin(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .output()
                .ok()
                .filter(|o| o.status.success())
                .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        };
        let answers = ok("systemd-run", &["--version"]).is_some()
            && ok(
                "systemctl",
                &["--user", "show", "--property=Version", "--value"],
            )
            .is_some_and(|v| !v.is_empty());
        // SAFETY: getuid has no preconditions.
        let uid = unsafe { libc::getuid() };
        let caller_inside = std::fs::read_to_string("/proc/self/cgroup")
            .is_ok_and(|c| c.contains(&format!("/user@{uid}.service/")));
        let lingers = ok(
            "loginctl",
            &[
                "show-user",
                &uid.to_string(),
                "--property=Linger",
                "--value",
            ],
        )
        .is_some_and(|v| v == "yes");
        UserManager {
            answers,
            caller_inside,
            lingers,
        }
    }
}

/// How a start runs a VM's QEMU ([`launch`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Launch {
    /// In a systemd user scope of its own, the unit named.
    Scope(String),
    /// In the cgroup of the process that starts it, as its descendant; on Linux with why no
    /// scope of its own was taken.
    Plain(Option<&'static str>),
}

/// How a start runs `name`'s QEMU on a host whose user manager is `m`. On Linux, in a systemd
/// user scope of its own ([`scope_unit`]), so the cgroup v2 CPU controller weighs the VM against
/// other cgroups, not against the processes of the cgroup that started it: there, a build at
/// nice 0 beside a QEMU that ananicy-cpp gives nice 9 (its `Heavy_CPU` type, as CachyOS has it)
/// takes nine shares of the CPU for each of QEMU's, and starved an x86-64 macOS guest until it
/// could not answer an SSH login (docs/qemu.md, "Beside a build"). The scope's weight is the
/// manager's default, and QEMU keeps its nice and policy, so no host policy changes.
///
/// No scope where no user manager answers, nor where the caller runs outside it (an SSH or
/// console login) and the manager does not linger: that manager stops its units when the user's
/// last session ends, which would end a VM the login started.
pub fn launch(name: &str, m: UserManager) -> Launch {
    if !cfg!(target_os = "linux") {
        Launch::Plain(None)
    } else if !m.answers {
        Launch::Plain(Some("no systemd user manager answers"))
    } else if !m.caller_inside && !m.lingers {
        Launch::Plain(Some(
            "this login runs outside the user's systemd manager, which stops a scope at the \
             user's last logout unless it lingers (loginctl enable-linger)",
        ))
    } else {
        Launch::Scope(scope_unit(name))
    }
}

/// The warning a start under `launch` gives, where QEMU runs in the cgroup that started it on
/// Linux.
pub fn launch_note(name: &str, launch: &Launch) -> Option<String> {
    match launch {
        Launch::Plain(Some(why)) => Some(format!(
            "{name}'s QEMU runs in the cgroup of the process that started it, not in a systemd \
             user scope of its own ({why}): CPU-heavy work in that cgroup, a build among it, \
             takes the guest's share of the CPU"
        )),
        _ => None,
    }
}

/// `inv` run as `launch` says: in a scope, through `systemd-run --user --scope`, which moves
/// itself into the new unit and then executes QEMU in its own process, with the environment it
/// was given, so QEMU keeps its pid, its environment and its detachment.
pub fn launched(inv: Invocation, name: &str, launch: &Launch) -> Invocation {
    let Launch::Scope(unit) = launch else {
        return inv;
    };
    let mut args = vec![
        "--user".to_owned(),
        "--scope".to_owned(),
        "--quiet".to_owned(),
        "--collect".to_owned(),
        format!("--unit={unit}"),
        format!("--description=reims-ctl: {name}'s QEMU"),
        "--".to_owned(),
        inv.program.to_string_lossy().into_owned(),
    ];
    args.extend(inv.args);
    Invocation {
        program: "systemd-run".into(),
        args,
        env: inv.env,
        unset: inv.unset,
    }
}

/// Whether the user's systemd unit `unit` is active.
fn unit_active(unit: &str) -> bool {
    std::process::Command::new("systemctl")
        .args(["--user", "is-active", "--quiet", unit])
        .stdin(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

/// What [`start`] started.
#[derive(Debug)]
pub struct Started {
    /// QEMU's pid.
    pub pid: i32,
    /// A warning about how QEMU runs ([`launch_note`]), for the start's output.
    pub note: Option<String>,
}

/// Starts the VM, and returns QEMU's pid once QMP answers.
pub fn start(vm: &Vm, host: Host) -> Result<Started> {
    if let Some(p) = pid(vm) {
        bail!("{} already runs (pid {p})", vm.name());
    }
    if let Some(other) = running_on_disk(vm) {
        bail!(
            "{other} is running on {}'s disk, {}: stop it first. Two VMs on one disk corrupt it",
            vm.name(),
            crate::paths::tilde(&vm.disk())
        );
    }
    preflight(vm, host)?;
    let ctx = context(vm, host)?;
    let inv = qemu::invocation(vm, &ctx)?;
    let launch = launch(vm.name(), UserManager::current());
    if let Launch::Scope(unit) = &launch
        && unit_active(unit)
    {
        bail!(
            "{}'s QEMU still runs in its systemd user scope {unit}, though its pid file names no \
             running QEMU: systemctl --user stop {unit} ends it",
            vm.name()
        );
    }
    let note = launch_note(vm.name(), &launch);
    let sock = prepare_run(vm)?;

    let mut log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(vm.qemu_log())?;
    writeln!(
        log,
        "reims-ctl {} started {} at {}",
        env!("CARGO_PKG_VERSION"),
        vm.name(),
        now_stamp("%Y-%m-%d %H:%M:%S %z")
    )?;
    match (&launch, &note) {
        (Launch::Scope(unit), _) => writeln!(log, "in the systemd user scope {unit}")?,
        (_, Some(n)) => writeln!(log, "warning: {n}")?,
        _ => {}
    }
    log.write_all(inv.shell_script("the command").as_bytes())?;
    writeln!(log, "----")?;
    let err = log.try_clone()?;

    let run = launched(inv, vm.name(), &launch);
    let mut cmd = run.command();
    cmd.current_dir(&vm.dir)
        .stdin(std::process::Stdio::null())
        .stdout(log)
        .stderr(err);
    // SAFETY: only async-signal-safe calls between fork and exec: setsid, fork, _exit.
    unsafe {
        cmd.pre_exec(|| {
            if libc::setsid() < 0 {
                return Err(std::io::Error::last_os_error());
            }
            match libc::fork() {
                -1 => Err(std::io::Error::last_os_error()),
                0 => Ok(()),
                _ => libc::_exit(0),
            }
        });
    }
    let mut child = cmd
        .spawn()
        .with_context(|| format!("run {}", run.program.display()))?;
    let _ = child.wait();
    let started = |pid| Started {
        pid,
        note: note.clone(),
    };

    let deadline = Instant::now() + Duration::from_secs(60);
    let mut seen = false;
    loop {
        if let Some(p) = pid(vm) {
            seen = true;
            if qmp::Qmp::connect(&sock, Duration::from_secs(2)).is_ok() {
                return Ok(started(p));
            }
        } else if seen || Instant::now() > deadline - Duration::from_secs(50) {
            // QEMU wrote its pid file and exited, or never wrote it.
            std::thread::sleep(Duration::from_millis(200));
            if pid(vm).is_none() {
                let log = tail(&vm.qemu_log(), 15);
                let silent = log.lines().last().is_some_and(|l| l.trim() == "----");
                let note = match mem_available_mib(Path::new("/proc/meminfo")) {
                    Some(a) if silent && vm.config.profile.memfd_ram(host) => {
                        format!("\n{}", memory_note(vm.name(), vm.config.memory_mib, a))
                    }
                    _ => String::new(),
                };
                bail!("QEMU exited:\n{log}{note}");
            }
        }
        if Instant::now() > deadline {
            match pid(vm) {
                Some(p) => return Ok(started(p)),
                None => bail!("QEMU did not start:\n{}", tail(&vm.qemu_log(), 15)),
            }
        }
        std::thread::sleep(Duration::from_millis(150));
    }
}

/// Stops the VM: macOS's Control-Option-Command-Power on the USB keyboard ("quit all apps,
/// then shut down", which asks about unsaved documents), then, a third of `timeout` in, the
/// machine's power button (ACPI's on q35), and at `timeout` a power-off; with
/// `force`, the power-off at once. With no `timeout` it waits until the guest shuts down, the
/// button a third of [`DEFAULT_STOP_TIMEOUT`] in, and never powers the VM off: where neither the
/// keys nor the button reach the guest, it ends with an error instead. `progress` hears each
/// step.
///
/// The keys come first: they shut a macOS 26.7.2 guest down within 20 s (2026-11-07, Radeon 780M
/// host). Neither they nor the button shut down a guest whose Setup Assistant is open, which macOS
/// opens once on a Mac it has not run on (`create`'s note, `new_mac_note`), so that stop ends in
/// the power-off, saying why it may have come.
pub fn stop(
    vm: &Vm,
    force: bool,
    timeout: Option<Duration>,
    mut progress: impl FnMut(&str),
) -> Result<()> {
    let Some(p) = pid(vm) else {
        progress("not running");
        return Ok(());
    };
    let sock = vm.qmp_socket();
    if !force {
        let chord: Vec<_> = ["ctrl", "alt", "meta_l", "power"]
            .iter()
            .map(|k| json!({ "type": "qcode", "data": k }))
            .collect();
        let keys = json!({ "keys": chord, "hold-time": 200 });
        let asked = match qmp::run(&sock, "send-key", Some(keys)) {
            Ok(_) => {
                progress("asked the guest to shut down (Control-Option-Command-Power)");
                true
            }
            Err(e) => {
                progress(&format!("{e}"));
                false
            }
        };
        let start = Instant::now();
        let button_at = timeout.unwrap_or(DEFAULT_STOP_TIMEOUT) / 3;
        let mut button = false;
        while timeout.is_none_or(|t| start.elapsed() < t) {
            if !alive(p) {
                progress("shut down");
                return Ok(());
            }
            if !button && (!asked || start.elapsed() >= button_at) {
                button = true;
                match qmp::run(&sock, "system_powerdown", None) {
                    Ok(_) => progress("still running: pressed its power button"),
                    Err(e) if !asked && timeout.is_none() => bail!(
                        "{e:#}: neither the keys nor the power button reach {}, which runs on; \
                         a forced stop powers it off",
                        vm.name()
                    ),
                    Err(_) => {}
                }
            }
            std::thread::sleep(Duration::from_millis(300));
        }
        progress(NO_SHUTDOWN);
    }
    power_off(vm, p, &mut progress)
}

/// What a stop says as it powers off a guest that did not shut down in its timeout.
pub const NO_SHUTDOWN: &str = "the guest did not shut down in time (a question about an unsaved \
     document, or Setup Assistant, which macOS opens once on a new Mac, holds it up: answer it in \
     the VM's window): powering off";

fn power_off(vm: &Vm, p: i32, progress: &mut impl FnMut(&str)) -> Result<()> {
    let _ = qmp::run(&vm.qmp_socket(), "quit", None);
    if wait_gone(p, Duration::from_secs(10)) {
        progress("powered off");
        return Ok(());
    }
    // SAFETY: p is our QEMU's pid, checked by pid().
    unsafe { libc::kill(p, libc::SIGTERM) };
    if wait_gone(p, Duration::from_secs(5)) {
        progress("powered off (SIGTERM)");
        return Ok(());
    }
    // SAFETY: as above.
    unsafe { libc::kill(p, libc::SIGKILL) };
    if wait_gone(p, Duration::from_secs(5)) {
        progress("killed (SIGKILL)");
        return Ok(());
    }
    bail!("QEMU (pid {p}) did not exit")
}

fn wait_gone(p: i32, t: Duration) -> bool {
    let deadline = Instant::now() + t;
    while Instant::now() < deadline {
        if !alive(p) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    !alive(p)
}

/// The host's available memory, `MemAvailable` in `file` (`/proc/meminfo`), in MiB; None where it
/// isn't there.
pub fn mem_available_mib(file: &Path) -> Option<u64> {
    std::fs::read_to_string(file)
        .ok()?
        .lines()
        .find_map(|l| l.strip_prefix("MemAvailable:"))
        .and_then(|v| v.split_whitespace().next()?.parse::<u64>().ok())
        .map(|kib| kib / 1024)
}

/// What a start adds where QEMU ended with no error of its own after the command reims-ctl logged
/// (the log's last line is that command's `----`), as when the kernel ends a process it cannot
/// give memory: QEMU allocates the VM's RAM whole as it starts on Linux (`qemu`'s module doc), so
/// a host without that much free memory ends it there.
pub fn memory_note(name: &str, memory_mib: u64, available_mib: u64) -> String {
    let gib = |mib: u64| match mib % 1024 {
        0 => format!("{} GiB", mib / 1024),
        _ => format!("{:.1} GiB", mib as f64 / 1024.0),
    };
    format!(
        "QEMU ended without an error of its own, as the kernel ends a process it cannot give \
         memory: {name} needs {} of free memory to start, since QEMU allocates all of its RAM \
         as it starts, and this host has {} available now. Free some, or give it less with \
         reims-ctl set {name} --memory SIZE",
        gib(memory_mib),
        gib(available_mib)
    )
}

/// The last `n` lines of a text file.
pub fn tail(path: &Path, n: usize) -> String {
    let Ok(mut f) = std::fs::File::open(path) else {
        return String::new();
    };
    let len = f.metadata().map(|m| m.len()).unwrap_or(0);
    let start = len.saturating_sub(64 * 1024);
    let _ = f.seek(SeekFrom::Start(start));
    let mut buf = Vec::new();
    let _ = f.read_to_end(&mut buf);
    let text = String::from_utf8_lossy(&buf);
    let lines: Vec<&str> = text.lines().collect();
    lines[lines.len().saturating_sub(n)..].join("\n")
}

/// A QEMU for tests: a process whose command line names `pidfile` and which writes its pid
/// there once it runs, as QEMU's `-pidfile` does, running `secs` seconds. It returns once the
/// pid file names it. A pid file the test wrote itself after `spawn` could name a process
/// whose command line still reads empty: Linux wakes the parent once the child's exec has
/// replaced its memory, and sets the new program's command line after that.
#[cfg(test)]
pub(crate) fn test_qemu(pidfile: &Path, secs: u32) -> std::process::Child {
    let mut qemu = std::process::Command::new("sh")
        .args(["-c", &format!("echo $$ > \"$0\"; sleep {secs}; :")])
        .arg(pidfile)
        .spawn()
        .unwrap();
    let (id, deadline) = (qemu.id(), Instant::now() + Duration::from_secs(10));
    let written = || {
        let s = std::fs::read_to_string(pidfile).unwrap_or_default();
        s.trim().parse::<u32>().ok() == Some(id)
    };
    while !written() {
        if Instant::now() > deadline || qemu.try_wait().unwrap().is_some() {
            let _ = qemu.kill();
            panic!("the test QEMU wrote no pid file at {}", pidfile.display());
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    qemu
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The limit at which an NVIDIA GPU takes a guest uncopied is a mapping per 4 KiB page of it
    /// past the library's 16,384 and 8,192, as the library's own line names it
    #[test]
    fn the_suggested_max_map_count_covers_a_guest_in_4k_pages() {
        assert_eq!(max_map_count(16 * 1024), (4_218_880, 8_388_608));
        assert_eq!(max_map_count(4096), (1_073_152, 2_097_152));
        assert_eq!(max_map_count(2048), (548_864, 1_048_576));
        assert!(map_count_note(8_388_608, 16 * 1024).is_none());
        let note = map_count_note(65_530, 16 * 1024).unwrap();
        assert!(
            note.contains("past about 159 MiB of a 16 GiB guest's memory"),
            "{note}"
        );
        assert!(note.contains("vm.max_map_count=8388608"), "{note}");
    }

    /// A start QEMU did not survive, with no error of its own, names the memory the VM's RAM
    /// needs and what the host has available, read from `MemAvailable` as the kernel writes it.
    #[test]
    fn a_vm_whose_qemu_ends_silently_is_told_what_its_ram_needs() {
        let env = crate::testenv::Env::new("meminfo");
        let file = env.root.join("meminfo");
        assert_eq!(mem_available_mib(&file), None);
        std::fs::write(
            &file,
            "MemTotal:       32000000 kB\nMemFree:         1000000 kB\nMemAvailable:    3145728 kB\n",
        )
        .unwrap();
        assert_eq!(mem_available_mib(&file), Some(3072));
        let note = memory_note("mac", 8192, 3584);
        assert!(
            note.starts_with("QEMU ended without an error of its own")
                && note.contains("mac needs 8 GiB of free memory to start")
                && note.contains("this host has 3.5 GiB available now")
                && note.contains("reims-ctl set mac --memory SIZE"),
            "{note}"
        );
    }

    /// The thrashing guard is told of above 0 only, read from its file as the kernel writes it,
    /// and the line that keeps it off is the one scripts/install-ctl.sh shows, for the same file.
    #[test]
    fn the_thrashing_guard_is_told_of_above_0_with_its_fix() {
        let env = crate::testenv::Env::new("min-ttl");
        let file = env.root.join("min_ttl_ms");
        assert_eq!(min_ttl_ms(&file), None);
        std::fs::write(&file, "100\n").unwrap();
        assert_eq!(min_ttl_ms(&file), Some(100));
        std::fs::write(&file, "0\n").unwrap();
        assert_eq!(min_ttl_ms(&file), Some(0));
        assert_eq!(min_ttl_note(0), None);
        let note = min_ttl_note(100).unwrap();
        assert!(note.starts_with("100: "), "{note}");
        assert!(
            note.contains("last 100 ms") && note.contains("QEMU"),
            "{note}"
        );
        let fix = min_ttl_fix(false);
        assert!(fix.contains(&format!("sudo tee {MIN_TTL_FILE}")), "{fix}");
        assert!(fix.contains(MIN_TTL_TMPFILES), "{fix}");
        assert!(min_ttl_fix(true).starts_with("it's 0 now"));
        let script = include_str!("../../../scripts/install-ctl.sh");
        assert!(script.contains(&format!("min_ttl_file={MIN_TTL_FILE}\n")));
        assert!(script.contains(&format!("min_ttl_line='{MIN_TTL_TMPFILES}'\n")));
    }

    /// The udmabuf message names the group that owns the node, says what to do where root alone
    /// may open it or there is none, and says why the group is needed over ssh
    #[test]
    fn the_udmabuf_message_names_the_fix() {
        let denied = std::io::Error::from(std::io::ErrorKind::PermissionDenied);
        let m = udmabuf_message(&denied, Some((992, "kvm".into())), false);
        assert!(m.contains("crash at higher resolutions"), "{m}");
        assert!(
            m.contains("sudo usermod -aG kvm $USER) and log in again"),
            "{m}"
        );
        assert!(!m.contains("ssh"), "{m}");
        let m = udmabuf_message(&denied, Some((78, "render".into())), true);
        assert!(m.contains("usermod -aG render"), "{m}");
        assert!(m.contains("Over ssh"), "{m}");
        let m = udmabuf_message(&denied, Some((0, "root".into())), false);
        assert!(
            m.contains("Only root") && m.contains("GROUP=\"kvm\""),
            "{m}"
        );
        let missing = std::io::Error::from(std::io::ErrorKind::NotFound);
        assert!(udmabuf_message(&missing, None, false).contains("modprobe udmabuf"));
        let no_module = std::io::Error::from_raw_os_error(libc::ENODEV);
        let m = udmabuf_message(&no_module, Some((992, "kvm".into())), false);
        assert!(
            m.contains("doesn't open") && m.contains("modprobe udmabuf"),
            "{m}"
        );
    }

    /// A group's name is read from /etc/group's form by its id
    #[test]
    fn a_group_is_named_by_its_id() {
        let groups = "root:x:0:\nkvm:x:992:anees\nrender:x:989:\n";
        assert_eq!(group_in(groups, 992).as_deref(), Some("kvm"));
        assert_eq!(group_in(groups, 0).as_deref(), Some("root"));
        assert_eq!(group_in(groups, 5), None);
    }

    /// QEMU's invocation for the launch tests: a program, its arguments and what it sets and
    /// removes of the environment.
    fn qemu_invocation() -> Invocation {
        Invocation {
            program: "/opt/qemu/bin/qemu-system-x86_64".into(),
            args: ["-name", "kv-26", "-pidfile", "/vms/kv-26/qemu.pid"]
                .map(String::from)
                .to_vec(),
            env: vec![("REIMS_VGPU_LOG".to_owned(), "info".to_owned())],
            unset: vec!["REIMS_VGPU_WINDOW".to_owned()],
        }
    }

    /// The program, arguments and environment changes `inv`'s command runs with.
    fn command_of(inv: &Invocation) -> (String, Vec<String>, Vec<(String, Option<String>)>) {
        let cmd = inv.command();
        let s = |o: &std::ffi::OsStr| o.to_string_lossy().into_owned();
        let mut envs: Vec<_> = cmd.get_envs().map(|(k, v)| (s(k), v.map(s))).collect();
        envs.sort();
        (s(cmd.get_program()), cmd.get_args().map(s).collect(), envs)
    }

    /// In a scope, the command is `systemd-run --user --scope` with the VM's own unit, then
    /// QEMU's program and arguments as they were, and QEMU's environment changes on it, which
    /// systemd-run's own process carries into QEMU; without one, it is QEMU's command as it was.
    #[test]
    fn a_scope_runs_qemu_through_systemd_run_under_the_vms_unit() {
        let plain = command_of(&qemu_invocation());
        let env = vec![
            ("REIMS_VGPU_LOG".to_owned(), Some("info".to_owned())),
            ("REIMS_VGPU_WINDOW".to_owned(), None),
        ];
        assert_eq!(plain.0, "/opt/qemu/bin/qemu-system-x86_64");
        assert_eq!(plain.2, env);
        let scope = Launch::Scope(scope_unit("kv-26"));
        let (program, args, envs) = command_of(&launched(qemu_invocation(), "kv-26", &scope));
        assert_eq!(program, "systemd-run");
        let mut want: Vec<String> = [
            "--user",
            "--scope",
            "--quiet",
            "--collect",
            "--unit=reims-ctl-kv-26.scope",
            "--description=reims-ctl: kv-26's QEMU",
            "--",
            "/opt/qemu/bin/qemu-system-x86_64",
        ]
        .map(String::from)
        .to_vec();
        want.extend(plain.1.iter().cloned());
        assert_eq!(args, want);
        assert_eq!(envs, env);
        for no_scope in [Launch::Plain(None), Launch::Plain(Some("why"))] {
            assert_eq!(
                command_of(&launched(qemu_invocation(), "kv-26", &no_scope)),
                plain
            );
        }
    }

    /// On Linux a start takes a scope of its own where the user's manager answers and outlasts
    /// the VM's start: the caller runs under it, or it lingers. Otherwise QEMU runs as before,
    /// with a warning that says why; elsewhere as before, with none.
    #[test]
    fn a_start_takes_a_scope_where_a_user_manager_keeps_it() {
        let m = |answers, caller_inside, lingers| UserManager {
            answers,
            caller_inside,
            lingers,
        };
        let cases = [
            (m(true, true, false), true),
            (m(true, false, true), true),
            (m(true, true, true), true),
            (m(true, false, false), false),
            (m(false, true, true), false),
            (m(false, false, false), false),
        ];
        for (manager, scoped) in cases {
            let launch = launch("kv-26", manager);
            let note = launch_note("kv-26", &launch);
            if !cfg!(target_os = "linux") {
                assert_eq!(launch, Launch::Plain(None), "{manager:?}");
                assert_eq!(note, None);
            } else if scoped {
                assert_eq!(launch, Launch::Scope("reims-ctl-kv-26.scope".to_owned()));
                assert_eq!(note, None);
            } else {
                assert!(matches!(launch, Launch::Plain(Some(_))), "{manager:?}");
                let note = note.unwrap();
                assert!(
                    note.starts_with(
                        "kv-26's QEMU runs in the cgroup of the process that started it, not in a \
                         systemd user scope of its own ("
                    ),
                    "{note}"
                );
                let why = if manager.answers {
                    "this login runs outside the user's systemd manager"
                } else {
                    "no systemd user manager answers"
                };
                assert!(note.contains(why), "{note}");
            }
        }
    }

    /// A VM does not start on a disk another VM, using it in place too, runs on.
    #[test]
    fn a_vm_does_not_start_on_a_disk_another_runs_on() {
        let env = crate::testenv::Env::new("start-shared");
        let disk = env.root.join("disk.img");
        std::fs::write(&disk, "").unwrap();
        let mut reg = crate::vm::Registry::default();
        let mut vms = Vec::new();
        for name in ["a", "b"] {
            let dir = env.root.join(name);
            std::fs::create_dir_all(&dir).unwrap();
            let config = format!(
                r#"
                name = "{name}"
                profile = "q35-x86_64"
                cpus = 2
                memory_mib = 4096
                disk = "{}"
                [q35]
                opencore = "opencore.qcow2"
                firmware_code = "OVMF_CODE.fd"
                firmware_vars = "OVMF_VARS.fd"
                "#,
                disk.display()
            );
            std::fs::write(dir.join(crate::vm::CONFIG_FILE), config).unwrap();
            reg.add(name, &dir).unwrap();
            vms.push(Vm::load(&dir).unwrap());
        }
        reg.save().unwrap();
        // a runs.
        let mut qemu = test_qemu(&vms[0].pidfile(), 30);
        let r = start(&vms[1], Host::current());
        let _ = qemu.kill();
        let _ = qemu.wait();
        let e = format!("{:#}", r.unwrap_err());
        assert!(e.starts_with("a is running on b's disk"), "{e}");
    }

    /// A start leaves the pid file and the QMP socket where they are, a running QEMU's among
    /// them: QEMU's lock on its pid file is what ends a second QEMU of the VM, from a start
    /// racing this one, before it opens the disks the first one writes.
    #[test]
    fn a_start_leaves_the_pid_file_and_socket_to_qemu() {
        let d = std::env::temp_dir().join(format!("reims-ctl-prepare-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let config = r#"
            name = "t"
            profile = "q35-x86_64"
            cpus = 2
            memory_mib = 4096
            disk = "disk.img"
            [q35]
            opencore = "opencore.qcow2"
            firmware_code = "OVMF_CODE.fd"
            firmware_vars = "OVMF_VARS.fd"
        "#;
        let vm = Vm {
            dir: d.clone(),
            config: toml::from_str(config).unwrap(),
        };
        let sock = vm.qmp_socket();
        std::fs::create_dir_all(sock.parent().unwrap()).unwrap();
        std::fs::write(&sock, "").unwrap();
        std::fs::write(vm.pidfile(), "1234\n").unwrap();
        assert_eq!(prepare_run(&vm).unwrap(), sock);
        assert_eq!(std::fs::read_to_string(vm.pidfile()).unwrap(), "1234\n");
        assert!(sock.exists());
        assert!(vm.logs_dir().is_dir());
        let _ = std::fs::remove_file(&sock);
        std::fs::remove_dir_all(&d).unwrap();
    }

    /// A QMP server at `sock` that greets, answers every command, and records each command but
    /// qmp_capabilities.
    fn fake_qmp(sock: &Path) -> std::sync::Arc<std::sync::Mutex<Vec<String>>> {
        use std::io::BufRead;
        let listener = std::os::unix::net::UnixListener::bind(sock).unwrap();
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let record = seen.clone();
        std::thread::spawn(move || {
            for mut conn in listener.incoming().map_while(Result::ok) {
                let _ = conn.write_all(b"{\"QMP\": {}}\n");
                let reader = std::io::BufReader::new(conn.try_clone().unwrap());
                for line in reader.lines().map_while(Result::ok) {
                    let v: serde_json::Value = serde_json::from_str(&line).unwrap();
                    let command = v["execute"].as_str().unwrap_or_default().to_string();
                    if command != "qmp_capabilities" {
                        record.lock().unwrap().push(command);
                    }
                    let _ = conn.write_all(b"{\"return\": {}}\n");
                }
            }
        });
        seen
    }

    /// A stop timeout of 0 is no limit, as an SSH port of 0 is no forward: stop asks the guest
    /// to shut down, waits for it and powers nothing off. Where nothing reaches the guest, it
    /// ends with an error rather than wait for good.
    /// A QMP socket's directory is made the user's alone, and one others may write into is
    /// refused: one made 0755 by an earlier reims-ctl is taken as it is
    #[test]
    fn a_qmp_socket_directory_others_may_write_into_is_refused() {
        use std::os::unix::fs::PermissionsExt;
        let d = std::env::temp_dir().join(format!("reims-ctl-private-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        let new = d.join("made");
        private_dir(&new).unwrap();
        let mode = |p: &Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode(&new), 0o700);
        let earlier = d.join("earlier");
        std::fs::create_dir(&earlier).unwrap();
        std::fs::set_permissions(&earlier, std::fs::Permissions::from_mode(0o755)).unwrap();
        private_dir(&earlier).unwrap();
        let open = d.join("open");
        std::fs::create_dir(&open).unwrap();
        std::fs::set_permissions(&open, std::fs::Permissions::from_mode(0o777)).unwrap();
        let refused = private_dir(&open);
        let file = d.join("file");
        std::fs::write(&file, "").unwrap();
        let not_dir = private_dir(&file);
        let _ = std::fs::remove_dir_all(&d);
        assert!(refused.is_err(), "{refused:?}");
        assert!(not_dir.is_err(), "{not_dir:?}");
    }

    #[test]
    fn a_stop_timeout_of_0_waits_for_the_guest() {
        let d = std::env::temp_dir().join(format!("reims-ctl-stop-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let config = r#"
            name = "t"
            profile = "q35-x86_64"
            cpus = 2
            memory_mib = 4096
            disk = "disk.img"
            stop_timeout_secs = 0
            [q35]
            opencore = "opencore.qcow2"
            firmware_code = "OVMF_CODE.fd"
            firmware_vars = "OVMF_VARS.fd"
        "#;
        let vm = Vm {
            dir: d.clone(),
            config: toml::from_str(config).unwrap(),
        };
        let qemu = || test_qemu(&vm.pidfile(), 10);
        let mut running = qemu();
        let asked = fake_qmp(&vm.qmp_socket());
        let (tx, rx) = std::sync::mpsc::channel();
        let stopping = {
            let vm = vm.clone();
            std::thread::spawn(move || {
                stop(&vm, false, vm.config.stop_timeout(), |s| {
                    let _ = tx.send(s.to_string());
                })
            })
        };
        std::thread::sleep(Duration::from_secs(1));
        let commands = asked.lock().unwrap().clone();
        let waited = alive(running.id() as i32);
        // The guest shuts down.
        let _ = running.kill();
        let _ = running.wait();
        let stopped = stopping.join().unwrap();
        let steps: Vec<String> = rx.try_iter().collect();

        std::fs::remove_file(vm.qmp_socket()).unwrap();
        let mut running = qemu();
        let unreached = stop(&vm, false, vm.config.stop_timeout(), |_| {});
        let _ = running.kill();
        let _ = running.wait();
        std::fs::remove_dir_all(&d).unwrap();

        assert_eq!(commands, ["send-key"]);
        assert!(waited);
        stopped.unwrap();
        assert_eq!(
            steps.last().map(String::as_str),
            Some("shut down"),
            "{steps:?}"
        );
        let e = format!("{:#}", unreached.unwrap_err());
        assert!(e.ends_with("a forced stop powers it off"), "{e}");
    }

    /// An x86-64 VM with an installer has install steps, which for a disk on NVMe first say to
    /// move it to SATA.
    #[test]
    fn install_steps_move_a_disk_on_nvme_to_sata_first() {
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
            dir: "/vms/t".into(),
            config: toml::from_str(config).unwrap(),
        };
        let steps = install_steps(&vm).unwrap();
        assert!(steps.starts_with("To install macOS, start t:"), "{steps}");
        let q = vm.config.q35.as_mut().unwrap();
        q.disk_controller = DiskController::Nvme;
        let steps = install_steps(&vm).unwrap();
        assert!(
            steps.starts_with("Its disk is on NVMe")
                && steps.contains("reims-ctl set t --disk-controller ahci"),
            "{steps}"
        );
        vm.config.q35.as_mut().unwrap().installer = None;
        assert_eq!(install_steps(&vm), None);
    }

    /// An x86-64 VM's install steps end with ejecting the installer.
    #[test]
    fn install_steps_end_with_ejecting_the_installer() {
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
        let vm = Vm {
            dir: "/vms/t".into(),
            config: toml::from_str(config).unwrap(),
        };
        let steps = install_steps(&vm).unwrap();
        assert!(
            steps.ends_with("eject the installer (reims-ctl eject t).")
                && !steps.contains("error mode")
                && !steps.contains("--network"),
            "{steps}"
        );
    }

    /// An offline installer's steps say to install with the network off, before the VM starts,
    /// and to turn it back on with the installer ejected; with the network already off, only the
    /// second.
    #[test]
    fn offline_install_steps_install_with_the_network_off() {
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
            installer = "macos-26.6.2-offline-installer.img"
        "#;
        let mut vm = Vm {
            dir: "/vms/t".into(),
            config: toml::from_str(config).unwrap(),
        };
        let off = "turn the VM's network off before you start it (reims-ctl set t --network false";
        let on = "eject the installer (reims-ctl eject t) and turn its network back on (reims-ctl \
                  set t --network).";
        let steps = install_steps(&vm).unwrap();
        assert!(
            steps.contains(off)
                && steps.find(off) < steps.find("To install macOS, start t")
                && steps.ends_with(on),
            "{steps}"
        );
        vm.config.network.enabled = false;
        let steps = install_steps(&vm).unwrap();
        assert!(
            !steps.contains("--network false") && steps.ends_with(on),
            "{steps}"
        );
    }
}
