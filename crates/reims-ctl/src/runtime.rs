//! Starting, finding and stopping a VM's QEMU.
//!
//! QEMU runs detached: in a session of its own (so closing the terminal does not end it), and
//! reparented away from reims-ctl by a double fork (so a TUI that stays open never holds an
//! exited QEMU as a zombie). Its pid is the one QEMU writes to the VM's `qemu.pid`, counted only
//! while that process runs and its command line names the pid file, so a stale file or a reused
//! pid is never taken for the VM. A VM started from a printed script is found the same way.
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
        bail!("{} is a {} VM, which {host} cannot run", c.name, c.profile);
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
        Profile::VmappleArm64 => {
            need(
                Path::new(qemu::AVPBOOTER),
                "Virtualization.framework's boot ROM",
            )?;
            let v = c.vmapple.as_ref().unwrap();
            need(&vm.path(&v.aux), "the auxiliary storage")?;
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

/// The reminder where an x86-64 VM's macOS gets installed.
pub const SET_METAL_ERROR_MODE: &str = "set Metal's error mode once macOS is installed (README)";

/// The host's `vm.max_map_count` (Linux).
pub fn host_max_map_count() -> Option<u64> {
    std::fs::read_to_string("/proc/sys/vm/max_map_count")
        .ok()
        .and_then(|s| s.trim().parse().ok())
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
/// ([`DiskController`]).
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
    Some(format!(
        "{first}To install macOS, start {name}: OpenCore boots its installer into macOS \
         Recovery. If the VM's disk is new, open Disk Utility, erase it (QEMU HARDDISK Media) as \
         APFS and quit Disk Utility. Then {last}. Once macOS is installed, eject the installer \
         (reims-ctl eject {name}) and set Metal's error mode (README)."
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
    if graphics == Graphics::AppleGfx && vm.config.profile == Profile::Q35X86_64 {
        w.extend(crate::opencore::missing_kext(vm));
        // docs/qemu.md, "x86-64 macOS guests: Metal's error mode": without it, macOS 26's icon
        // service ends as it starts, again and again, and the library logs each command it left
        // queued, which QEMU's log keeps with logs off too. From Recovery, the `cd` expands the
        // Data volume's name, and `defaults` leaves the file at mode 0600, which the user's
        // processes cannot read.
        let n = emptied_entry_lines(&vm.qemu_log());
        if n >= 3 {
            w.push(format!(
                "QEMU's latest log has {n} lines on GPU work left by crashed macOS processes. On \
                 x86-64 macOS 26 that's most likely its icon service, crashing because Metal's \
                 error mode isn't set: the desktop can stay on its wallpaper with no menu bar, or \
                 come up gray with a menu bar but no Dock. To set it, turn its boot selection \
                 on (press e in reims-ctl, or run reims-ctl update-opencore {} --boot-selection), \
                 boot Recovery from OpenCore's picker, open Utilities > Terminal and run: cd \
                 /Volumes/*Data/Library/Preferences && defaults write \"$PWD/.GlobalPreferences\" \
                 METAL_ERROR_MODE -int 5 && chmod 644 .GlobalPreferences.plist && reboot",
                vm.name()
            ));
        }
    }
    if host.os == Os::Linux
        && let Some(problem) = udmabuf_problem()
    {
        w.push(problem);
    }
    w
}

/// How many lines of a QEMU log, since the guest's last reset, name an object entry the guest
/// had emptied: the library logs them at its warn level when a command of a guest process that
/// has ended runs, and `device reset` at its info level; reims-ctl turns both on with logs off
/// too. The TUI asks at every draw, so the count carries on from the bytes already read while the
/// file is the same (by inode) and has not shrunk.
fn emptied_entry_lines(log: &Path) -> usize {
    use std::os::unix::fs::MetadataExt;
    const EMPTIED: &[u8] = b"has an empty entry";
    static READ: std::sync::Mutex<(std::path::PathBuf, u64, u64, usize)> =
        std::sync::Mutex::new((std::path::PathBuf::new(), 0, 0, 0));
    let Ok(mut f) = std::fs::File::open(log) else {
        return 0;
    };
    let Ok(meta) = f.metadata() else { return 0 };
    let mut r = READ.lock().unwrap_or_else(|e| e.into_inner());
    if r.0 != log || r.1 != meta.ino() || r.2 > meta.len() {
        *r = (log.to_path_buf(), meta.ino(), 0, 0);
    }
    let mut buf = Vec::new();
    if f.seek(SeekFrom::Start(r.2)).is_ok() {
        let _ = f.take(meta.len() - r.2).read_to_end(&mut buf);
    }
    // Whole lines only: the rest is read again once its line ends.
    let end = buf.iter().rposition(|&b| b == b'\n').map_or(0, |i| i + 1);
    for l in buf[..end].split(|&b| b == b'\n') {
        if l.ends_with(b"device reset") {
            r.3 = 0;
        } else if l.windows(EMPTIED.len()).any(|w| w == EMPTIED) {
            r.3 += 1;
        }
    }
    r.2 += end as u64;
    r.3
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

/// Writes the trimmed auxiliary storage a vmapple VM boots, if missing: the image without its
/// first 16 KiB block (docs/system/arm/vmapple.rst in vendor/qemu).
pub fn ensure_aux_trimmed(vm: &Vm) -> Result<()> {
    let Some(trimmed) = vm.aux_trimmed() else {
        return Ok(());
    };
    if trimmed.exists() {
        return Ok(());
    }
    let aux = vm.path(&vm.config.vmapple.as_ref().unwrap().aux);
    let mut src = std::fs::File::open(&aux).with_context(|| format!("open {}", aux.display()))?;
    src.seek(SeekFrom::Start(0x4000))?;
    let tmp = trimmed.with_extension("trimmed.tmp");
    let mut dst = std::fs::File::create(&tmp)?;
    std::io::copy(&mut src, &mut dst)?;
    dst.sync_all()?;
    std::fs::rename(&tmp, &trimmed)?;
    Ok(())
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

/// Starts the VM, and returns QEMU's pid once QMP answers.
pub fn start(vm: &Vm, host: Host) -> Result<i32> {
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
    if vm.config.profile == Profile::VmappleArm64 {
        ensure_aux_trimmed(vm)?;
    }
    let ctx = context(vm, host)?;
    let inv = qemu::invocation(vm, &ctx)?;
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
    log.write_all(inv.shell_script("the command").as_bytes())?;
    writeln!(log, "----")?;
    let err = log.try_clone()?;

    let mut cmd = inv.command();
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
        .with_context(|| format!("run {}", inv.program.display()))?;
    let _ = child.wait();

    let deadline = Instant::now() + Duration::from_secs(60);
    let mut seen = false;
    loop {
        if let Some(p) = pid(vm) {
            seen = true;
            if qmp::Qmp::connect(&sock, Duration::from_secs(2)).is_ok() {
                return Ok(p);
            }
        } else if seen || Instant::now() > deadline - Duration::from_secs(50) {
            // QEMU wrote its pid file and exited, or never wrote it.
            std::thread::sleep(Duration::from_millis(200));
            if pid(vm).is_none() {
                bail!("QEMU exited:\n{}", tail(&vm.qemu_log(), 15));
            }
        }
        if Instant::now() > deadline {
            match pid(vm) {
                Some(p) => return Ok(p),
                None => bail!("QEMU did not start:\n{}", tail(&vm.qemu_log(), 15)),
            }
        }
        std::thread::sleep(Duration::from_millis(150));
    }
}

/// Stops the VM: macOS's Control-Option-Command-Power on the USB keyboard ("quit all apps,
/// then shut down", which asks about unsaved documents), then, a third of `timeout` in, the
/// machine's power button (ACPI's on q35, a GPIO on vmapple), and at `timeout` a power-off; with
/// `force`, the power-off at once. With no `timeout` it waits until the guest shuts down, the
/// button a third of [`DEFAULT_STOP_TIMEOUT`] in, and never powers the VM off: where neither the
/// keys nor the button reach the guest, it ends with an error instead. `progress` hears each
/// step.
///
/// The keys come first because the power button alone does not shut a guest down: a macOS 26
/// vmapple guest did nothing with vmapple's in 90 s, where the keys shut it down in under 6 s
/// (2026-10-11, M3 Max).
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
        progress("the guest did not shut down in time: powering off");
    }
    power_off(vm, p, &mut progress)
}

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
                profile = "vmapple-arm64"
                cpus = 2
                memory_mib = 4096
                disk = "{}"
                [vmapple]
                aux = "aux.img"
                ecid = 1
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
            profile = "vmapple-arm64"
            cpus = 2
            memory_mib = 4096
            disk = "disk.img"
            [vmapple]
            aux = "aux.img"
            ecid = 1
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
            profile = "vmapple-arm64"
            cpus = 2
            memory_mib = 4096
            disk = "disk.img"
            stop_timeout_secs = 0
            [vmapple]
            aux = "aux.img"
            ecid = 1
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

    #[test]
    fn emptied_entry_lines_count_on_from_what_was_read() {
        let d = std::env::temp_dir().join(format!("reims-ctl-emptied-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let log = d.join("qemu.log");
        let fault = "reims-vgpu WARN reims_vgpu_core::tasks::local::exec: task 2: exec faulted: \
                     resource 3 not found: object 3 has an empty entry\n";
        let append = |s: &str| {
            let mut f = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&log)
                .unwrap();
            f.write_all(s.as_bytes()).unwrap();
        };
        assert_eq!(emptied_entry_lines(&log), 0);
        append("reims-ctl 0.1.0 started t\n");
        append(fault);
        assert_eq!(emptied_entry_lines(&log), 1);
        // A line still being written counts once it ends.
        append(&fault[..40]);
        assert_eq!(emptied_entry_lines(&log), 1);
        append(&fault[40..]);
        append(fault);
        assert_eq!(emptied_entry_lines(&log), 3);
        // The guest's reboot in the same run (as after setting the key from Recovery) counts
        // from the device's reset.
        append("reims-vgpu INFO reims_vgpu_core::device::lifecycle: device reset\n");
        assert_eq!(emptied_entry_lines(&log), 0);
        append(fault);
        assert_eq!(emptied_entry_lines(&log), 1);
        // A start's rotation leaves a new file, counted from its start.
        std::fs::rename(&log, d.join("qemu.1.log")).unwrap();
        append(&"x".repeat(4096));
        append("\n");
        append(fault);
        assert_eq!(emptied_entry_lines(&log), 1);
        std::fs::remove_dir_all(&d).unwrap();
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
}
