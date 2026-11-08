//! Making a VM: its directory, disks, firmware and `vm.toml`, then its registry entry.
//!
//! An x86-64 VM (`q35-x86_64`) boots OpenCore from an OpenCore image of its own, with its own OVMF
//! code and variables: made for it by reims-util, holding the guest's kexts for reims-vgpu's own
//! device where the VM is on it ([`reims_util::kext_args`]), and presenting the Mac an existing
//! disk's macOS ran on where the image it booted from is beside it ([`disks_mac`]), or copies of an
//! image given and the OVMF beside it ([`firmware_beside`]). Its disk is a new one or an existing
//! one, copied (a reflink where the file system can make one, else a full copy, which says so
//! first: copy.rs) or used in place. A new disk needs an installer: one reims-util downloads from
//! Apple (reims_util.rs), before the VM's directory is made, so a download stopped part way leaves
//! no VM half made, or one given. An installer is used in place, except a DMG, which qemu-img
//! converts to a raw image in the VM's directory ([`installer_media`]). A VM's settings name its
//! graphics: reims-vgpu's own device unless the create chose another.

use crate::copy;
use crate::host::{Host, Profile};
use crate::install;
use crate::opencore::{self, Identity};
use crate::paths;
use crate::reims_util::{self, Download};
use crate::vm::*;
use anyhow::{Context, Result, bail};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImportMode {
    /// A copy in the VM's directory: a reflink (btrfs, XFS) or clone (APFS) where the image and
    /// the directory are on one file system that makes them, else a full copy, which says so
    /// first where it is long ([`copy`]).
    Copy,
    /// The image itself.
    InPlace,
}

impl ImportMode {
    pub fn parse(s: &str) -> Result<ImportMode> {
        Ok(match s {
            "copy" | "clone" => ImportMode::Copy,
            "in-place" | "inplace" | "link" => ImportMode::InPlace,
            _ => bail!("import mode is copy or in-place, not {s:?}"),
        })
    }
}

#[derive(Clone, Debug)]
pub enum DiskSource {
    New { size_mib: u64 },
    Existing { path: PathBuf, mode: ImportMode },
}

#[derive(Clone, Debug)]
pub struct Spec {
    pub name: String,
    pub dir: Option<PathBuf>,
    pub profile: Option<Profile>,
    pub cpus: Option<u32>,
    pub memory_mib: Option<u64>,
    pub disk: Option<DiskSource>,
    /// The graphics device; none for the default, reims-vgpu's own.
    pub graphics: Option<Graphics>,
    pub display: Display,
    pub gpu: Gpu,
    pub logs: bool,
    pub audio: bool,
    pub ssh_port: Option<u16>,
    pub mac: Option<String>,
    // q35
    /// macOS to download from Apple, for the installer.
    pub macos: Option<Download>,
    pub opencore: Option<PathBuf>,
    pub installer: Option<PathBuf>,
}

impl Spec {
    pub fn new(name: &str) -> Spec {
        Spec {
            name: name.into(),
            dir: None,
            profile: None,
            cpus: None,
            memory_mib: None,
            disk: None,
            graphics: None,
            display: Display::default(),
            gpu: Gpu::default(),
            logs: false,
            audio: true,
            ssh_port: None,
            mac: None,
            macos: None,
            opencore: None,
            installer: None,
        }
    }
}

/// The host's logical CPUs.
pub fn host_cpus() -> u32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(4)
}

/// The host's RAM in MiB.
pub fn host_memory_mib() -> u64 {
    #[cfg(target_os = "macos")]
    {
        let mut v: u64 = 0;
        let mut size = std::mem::size_of::<u64>();
        let name = c"hw.memsize";
        // SAFETY: hw.memsize is a u64.
        if unsafe {
            libc::sysctlbyname(
                name.as_ptr(),
                (&mut v as *mut u64).cast(),
                &mut size,
                std::ptr::null_mut(),
                0,
            )
        } == 0
        {
            return v / (1024 * 1024);
        }
        16 * 1024
    }
    #[cfg(not(target_os = "macos"))]
    {
        std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|m| {
                m.lines()
                    .find(|l| l.starts_with("MemTotal:"))
                    .and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok())
            })
            .map(|kib| kib / 1024)
            .unwrap_or(16 * 1024)
    }
}

pub fn default_cpus() -> u32 {
    (host_cpus() / 2).clamp(2, 8)
}

pub fn default_memory_mib() -> u64 {
    ((host_memory_mib() / 2) / 1024 * 1024).clamp(4096, 16384)
}

pub fn default_dir(name: &str) -> PathBuf {
    paths::default_vms_dir().join(name)
}

/// Makes the VM, calling `progress` with each step, and registers it.
pub fn create(spec: Spec, host: Host, progress: &mut dyn FnMut(&str)) -> Result<Vm> {
    check_name(&spec.name)?;
    if let Some(m) = &spec.mac {
        check_mac(m)?;
    }
    if Registry::load()?.find(&spec.name).is_some() {
        bail!("a VM named {} already exists", spec.name);
    }
    let profile = match spec.profile {
        Some(p) => p,
        None => host.default_profile().context(crate::host::UNSUPPORTED)?,
    };
    if !Profile::supported_on(host).contains(&profile) {
        bail!("{}", crate::host::UNSUPPORTED);
    }
    let graphics = spec.graphics.unwrap_or_default();
    spec.display.check()?;
    // The image reims-util makes holds them.
    if graphics == Graphics::ReimsVgpu && spec.opencore.is_none() {
        crate::prebuilt::set()?;
    }
    let dir = spec.dir.clone().unwrap_or_else(|| default_dir(&spec.name));
    // Checked here, before a download, and again as the directory is taken, after it (claim).
    if dir.exists() && std::fs::read_dir(&dir)?.next().is_some() {
        return Err(not_empty(&dir));
    }
    // What was given is checked before the slow steps: a download, a disk's copy.
    if let Some(DiskSource::Existing { path, .. }) = &spec.disk
        && !path.is_file()
    {
        bail!("the disk {} does not exist", path.display());
    }
    let mut spec = spec;
    // The macOS version, where an installer was downloaded for it.
    let mut macos = None;
    if spec.macos.is_some() && spec.installer.is_some() {
        bail!("give macOS to download or an installer, not both");
    }
    if let Some(i) = &spec.installer
        && !i.exists()
    {
        bail!("the installer {} does not exist", i.display());
    }
    match &spec.opencore {
        Some(oc) => {
            own_firmware(oc)?;
        }
        // reims-util makes one, after the disk's copy.
        None => {
            reims_util::binary()?;
        }
    }
    if spec.macos.is_none()
        && spec.installer.is_none()
        && matches!(spec.disk, Some(DiskSource::New { .. }))
    {
        bail!(
            "a new disk needs macOS installed on it: give the macOS to download \
             (--macos tahoe), or an installer"
        );
    }
    if let Some(d) = spec.macos.take() {
        let i = reims_util::installer(&d, progress)?;
        spec.installer = Some(i.image);
        macos = Some(i.version);
    }
    let made = claim(&dir)?;
    let result = match profile {
        Profile::Q35X86_64 => make_q35(&spec, graphics, &dir, macos.as_deref(), progress),
    };
    let vm = match result {
        Ok(vm) => vm,
        Err(e) => {
            if made {
                let _ = std::fs::remove_dir_all(&dir);
            } else if let Ok(rd) = std::fs::read_dir(&dir) {
                for e in rd.flatten() {
                    let _ = if e.path().is_dir() {
                        std::fs::remove_dir_all(e.path())
                    } else {
                        std::fs::remove_file(e.path())
                    };
                }
            }
            return Err(e);
        }
    };
    vm.save()?;
    let _ = std::fs::remove_file(dir.join(CREATING));
    // Read again: VMs may have come and gone while this one was made, over a download.
    let mut reg = Registry::load()?;
    reg.add(&spec.name, &dir)?;
    reg.save()?;
    progress("done");
    Ok(vm)
}

/// Marks a directory a VM is being made in, until its vm.toml is written.
const CREATING: &str = ".reims-ctl-creating";

fn not_empty(dir: &Path) -> anyhow::Error {
    anyhow::anyhow!(
        "{} is not empty: choose a new or empty directory",
        dir.display()
    )
}

/// Takes `dir` for a VM being made: makes it, or takes it as it is where it is empty, and marks
/// it, so that another create into it finds it taken, though its own check came before this
/// one's download. Returns whether it was made, which a failed create removes whole.
fn claim(dir: &Path) -> Result<bool> {
    let made = !dir.exists();
    std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    if std::fs::read_dir(dir)?.next().is_some() {
        return Err(not_empty(dir));
    }
    // Of two creates that both found it empty, the second finds the first's mark.
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(dir.join(CREATING))
        .map_err(|_| not_empty(dir))?;
    Ok(made)
}

fn base_config(spec: &Spec, profile: Profile, disk: PathBuf, disk_format: ImageFormat) -> VmConfig {
    // Saved by name: an image made for the one it is on may not suit another.
    VmConfig {
        name: spec.name.clone(),
        profile,
        cpus: spec.cpus.unwrap_or_else(default_cpus),
        memory_mib: spec.memory_mib.unwrap_or_else(default_memory_mib),
        disk,
        disk_format,
        stop_timeout_secs: 90,
        extra_args: vec![],
        graphics: Some(spec.graphics.unwrap_or_default()),
        display: spec.display.clone(),
        gpu: spec.gpu.clone(),
        audio: Audio {
            enabled: spec.audio,
            backend: None,
        },
        network: Network {
            enabled: true,
            mac: spec.mac.clone().unwrap_or_else(random_mac),
            ssh_port: spec.ssh_port.and_then(ssh_port),
        },
        logs: Logs {
            enabled: spec.logs,
            ..Logs::default()
        },
        q35: None,
        extra_env: BTreeMap::new(),
        usb: vec![],
    }
}

fn file_name(p: &Path) -> String {
    p.file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

fn qemu_img(profile: Profile, args: &[&std::ffi::OsStr]) -> Result<()> {
    let img = install::qemu_img(profile)?;
    let out = Command::new(&img)
        .args(args)
        .output()
        .with_context(|| format!("run {}", img.display()))?;
    if !out.status.success() {
        bail!(
            "qemu-img failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(())
}

/// What a create that gives macOS on an existing disk a new OpenCore image presenting a Mac of
/// its own says, where it keeps no Mac of the disk's ([`disks_mac`]): the image presents a new
/// serial number, MLB and UUID, so the disk's macOS boots on a Mac it has not run on, and runs
/// Setup Assistant once at that boot. While Setup Assistant is open, macOS shuts down at neither
/// the keys nor the power button [`crate::runtime::stop`] sends, so a stop then ends in a
/// power-off at its timeout.
/// Measured with a macOS 26.7.2 (25G332) disk on a Radeon 780M host, 2026-11-07: on a new image,
/// Setup Assistant (`-MiniBuddyYes`) ran from about 10 s after boot, and the guest ignored the
/// keys and the button for 180 s, and System Events' `shut down` for 120 s, while `sudo shutdown
/// -h now` over SSH, which no stop can send without the guest's password, powered it off in 6 s;
/// on the image it booted from before (`--opencore`), Setup Assistant did not run, and the keys
/// shut it down within 20 s. None where the disk is new, or an image is given.
fn new_mac_note(spec: &Spec) -> Option<&'static str> {
    (matches!(spec.disk, Some(DiskSource::Existing { .. })) && spec.opencore.is_none()).then_some(
        "macOS on this disk will boot on a new Mac (a new serial number and UUID), so it opens \
         Setup Assistant once: finish it in the VM's window, since until then it does not shut \
         down when asked. --opencore with the image the disk booted from keeps its Mac",
    )
}

/// The OpenCore image beside an existing disk, as a VM's directory holds them, reims-ctl's or one
/// reims-util made an image into: `opencore.qcow2`, else `opencore.img`, in the disk's directory.
fn image_beside(disk: &Path) -> Option<PathBuf> {
    let dir = disk.parent()?;
    ["opencore.qcow2", "opencore.img"]
        .into_iter()
        .map(|n| dir.join(n))
        .find(|p| p.is_file())
}

/// The Mac macOS on an existing disk ran on, where the create is given no image: the one the
/// image beside the disk presents ([`image_beside`]), with that image. The VM's image is made to
/// present it, so the disk's macOS boots on the Mac it knows and opens no Setup Assistant, which
/// would hold up its shutdowns ([`new_mac_note`]). None for a copy of the disk of a VM reims-ctl
/// has ([`vm_with_disk`]): that copy runs beside the VM, so it is a Mac of its own, as two VMs
/// running as one Mac would be one Mac to Apple's services, which know a Mac by its serial
/// number, MLB and ROM. A VM given that disk in place runs the VM's own macOS, never beside it
/// (runtime.rs refuses a start on a disk another VM runs on), so it keeps the Mac. `progress`
/// hears why an image beside the disk is not taken.
fn disks_mac(spec: &Spec, progress: &mut dyn FnMut(&str)) -> Option<(PathBuf, Identity)> {
    let Some(DiskSource::Existing { path, mode }) = &spec.disk else {
        return None;
    };
    if spec.opencore.is_some() {
        return None;
    }
    let image = image_beside(path)?;
    if *mode == ImportMode::Copy
        && let Some(vm) = vm_with_disk(path)
    {
        progress(&format!(
            "this is a copy of {vm}'s disk, so it runs beside {vm} as a Mac of its own"
        ));
        return None;
    }
    match ImageFormat::detect(&image)
        .and_then(|f| opencore::Config::read(&image, f))
        .and_then(|c| c.identity())
    {
        Ok(id) => Some((image, id)),
        Err(e) => {
            progress(&format!(
                "could not read the Mac {} presents: {e:#}",
                paths::tilde(&image)
            ));
            None
        }
    }
}

/// The VM of those reims-ctl has whose disk `disk` is, by name.
fn vm_with_disk(disk: &Path) -> Option<String> {
    let disk = std::fs::canonicalize(disk).ok()?;
    Registry::load().ok()?.vms.iter().find_map(|e| {
        let vm = Vm::load(&e.dir).ok()?;
        let theirs = std::fs::canonicalize(vm.path(&vm.config.disk)).ok()?;
        (theirs == disk).then(|| e.name.clone())
    })
}

/// The network card address a Mac's ROM is (12 hex digits), as a Mac's ROM is its first network
/// card's address: none for a multicast address, which no card has.
fn mac_of_rom(rom: &str) -> Option<String> {
    let b = (0..6)
        .map(|i| u8::from_str_radix(rom.get(i * 2..i * 2 + 2)?, 16).ok())
        .collect::<Option<Vec<u8>>>()?;
    (rom.len() == 12 && (b[0] & 1) == 0).then(|| {
        b.iter()
            .map(|x| format!("{x:02x}"))
            .collect::<Vec<_>>()
            .join(":")
    })
}

/// Makes an x86-64 VM on `graphics`, which its settings save by name.
fn make_q35(
    spec: &Spec,
    graphics: Graphics,
    dir: &Path,
    macos: Option<&str>,
    progress: &mut dyn FnMut(&str),
) -> Result<Vm> {
    // The disk.
    let (disk, disk_format) = match spec.disk.clone() {
        Some(DiskSource::New { size_mib }) => {
            progress("creating the disk");
            let d = dir.join("disk.qcow2");
            qemu_img(
                Profile::Q35X86_64,
                &[
                    "create".as_ref(),
                    "-q".as_ref(),
                    "-f".as_ref(),
                    "qcow2".as_ref(),
                    d.as_os_str(),
                    format!("{size_mib}M").as_ref(),
                ],
            )?;
            (PathBuf::from("disk.qcow2"), ImageFormat::Qcow2)
        }
        Some(DiskSource::Existing { path, mode }) => {
            import_disk(&path, mode, dir, "disk", progress)?
        }
        None => bail!("give the VM a disk: a new one's size, or an existing image"),
    };

    // OpenCore and OVMF: made for the VM, or copies of what was given. A made image presents the
    // Mac an existing disk's macOS ran on where the image it booted from is beside it, else a Mac
    // of its own. A Mac's ROM is its first network card's address: a new Mac's ROM is the VM's
    // card's, and a VM presenting the disk's Mac takes that Mac's ROM for its card, unless a MAC
    // was given.
    let kept = disks_mac(spec, progress);
    let mac = spec
        .mac
        .clone()
        .or_else(|| kept.as_ref().and_then(|(_, id)| mac_of_rom(&id.rom)))
        .unwrap_or_else(random_mac);
    let (oc_name, oc_format) = match &spec.opencore {
        None => {
            let kexts = reims_util::kext_args(graphics)?;
            let image = dir.join("opencore.qcow2");
            match &kept {
                Some((beside, id)) => {
                    progress(&format!(
                        "presenting the Mac macOS on this disk ran on, which {} presents",
                        paths::tilde(beside)
                    ));
                    reims_util::opencore_presenting(&image, id, &kexts, progress)?;
                }
                None => {
                    reims_util::opencore(&image, &mac, macos, &kexts, progress)?;
                    if let Some(note) = new_mac_note(spec) {
                        progress(note);
                    }
                }
            }
            (PathBuf::from("opencore.qcow2"), ImageFormat::Qcow2)
        }
        Some(src) => {
            progress("copying the OpenCore image");
            let format = ImageFormat::detect(src)?;
            let name = format!(
                "opencore.{}",
                if format == ImageFormat::Qcow2 {
                    "qcow2"
                } else {
                    "img"
                }
            );
            copy::file(src, &dir.join(&name), progress)?;
            (PathBuf::from(name), format)
        }
    };
    // An image given brings its OVMF beside it, as a made one's is written beside it.
    if let Some(src) = &spec.opencore {
        let (code, vars) = own_firmware(src)?;
        progress("copying the firmware");
        copy::file(&code, &dir.join("OVMF_CODE.fd"), progress)?;
        copy::file(&vars, &dir.join("OVMF_VARS.fd"), progress)?;
    }
    // A read-only source's copy is read-only too; the variables are written.
    let _ = make_writable(&dir.join("OVMF_VARS.fd"));

    let installer = match spec.installer.clone() {
        Some(i) => Some(installer_media(&i, dir, progress)?),
        None => None,
    };
    let mut c = base_config(spec, Profile::Q35X86_64, disk, disk_format);
    c.graphics = Some(graphics);
    c.network.mac = mac;
    c.q35 = Some(Q35 {
        opencore: oc_name,
        opencore_format: oc_format,
        firmware_code: "OVMF_CODE.fd".into(),
        firmware_vars: "OVMF_VARS.fd".into(),
        installer,
        cpu_model: default_cpu_model(),
        cpu_flags: default_cpu_flags(),
        disk_controller: DiskController::Ahci,
        disk_port: 3,
    });
    Ok(Vm {
        dir: dir.to_path_buf(),
        config: c,
    })
}

fn make_writable(p: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let mut perm = std::fs::metadata(p)?.permissions();
    perm.set_mode(perm.mode() | 0o600);
    std::fs::set_permissions(p, perm)
}

fn import_disk(
    src: &Path,
    mode: ImportMode,
    dir: &Path,
    stem: &str,
    progress: &mut dyn FnMut(&str),
) -> Result<(PathBuf, ImageFormat)> {
    if !src.is_file() {
        bail!("the disk {} does not exist", src.display());
    }
    let src = std::fs::canonicalize(src)?;
    let format = ImageFormat::detect(&src)?;
    Ok(match mode {
        ImportMode::InPlace => (src, format),
        ImportMode::Copy => {
            progress(&format!("copying {}", file_name(&src)));
            let ext = if format == ImageFormat::Qcow2 {
                "qcow2"
            } else {
                "img"
            };
            let name = format!("{stem}.{ext}");
            copy::file(&src, &dir.join(&name), progress)?;
            let _ = make_writable(&dir.join(&name));
            (PathBuf::from(name), format)
        }
    })
}

/// The VM's raw copy of an installer DMG, in its directory.
pub const INSTALLER_COPY: &str = "installer.img";

/// Whether `p` is a DMG QEMU reads only through its dmg driver: a UDIF image, which ends in a
/// 512-byte trailer starting `koly`. A read/write DMG (UDRW) has none: it is a raw image.
pub fn is_dmg(p: &Path) -> Result<bool> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(p).with_context(|| format!("open {}", p.display()))?;
    let len = f.metadata()?.len();
    if len < 512 {
        return Ok(false);
    }
    f.seek(SeekFrom::Start(len - 512))?;
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    Ok(magic == *b"koly")
}

/// The installer a VM in `dir` attaches for `src`: `src` itself, used in place, or, for a DMG
/// (a recovery image, BaseSystem.dmg, which is what reims-util's online installer is), a raw
/// copy qemu-img writes into `dir`, named relative to it. QEMU could read the DMG itself, but
/// decompresses a whole chunk for each read outside the last one: 4 KiB reads 1 MiB apart took
/// 2.5 ms each from a 2.5 GiB recovery image against 0.05 ms from its raw copy, which took 10 s
/// to make (measured on a Ryzen 7 7840HS).
///
/// The installed QEMU reads zlib DMGs (Apple's recovery images, as osrecovery.apple.com serves
/// them, are); bzip2 and LZFSE ones only where it was built with their libraries, and LZMA and
/// ADC ones never.
pub fn installer_media(src: &Path, dir: &Path, progress: &mut dyn FnMut(&str)) -> Result<PathBuf> {
    if !is_dmg(src)? {
        return Ok(src.to_path_buf());
    }
    progress(&format!("converting {} to a raw image", file_name(src)));
    // Written beside the copy and renamed over it: a running VM may have it open.
    let part = dir.join(format!("{INSTALLER_COPY}.part"));
    let converted = qemu_img(
        Profile::Q35X86_64,
        &[
            "convert".as_ref(),
            "-q".as_ref(),
            "-f".as_ref(),
            "dmg".as_ref(),
            "-O".as_ref(),
            "raw".as_ref(),
            src.as_os_str(),
            part.as_os_str(),
        ],
    );
    if let Err(e) = converted {
        let _ = std::fs::remove_file(&part);
        let e = format!("{e:#}");
        // qemu-img's warning names the compression: "dmg-lzfse module is missing", "chunks
        // of unknown type".
        if e.contains("module is missing") || e.contains("unknown type") {
            bail!(
                "{e}\n{} is compressed in a way this QEMU does not read: on a Mac, `hdiutil \
                 convert {} -format UDZO -o <new>.dmg` makes a zlib copy, which it does",
                file_name(src),
                crate::qemu::quote(&src.to_string_lossy())
            );
        }
        bail!("{e}");
    }
    std::fs::rename(&part, dir.join(INSTALLER_COPY))
        .with_context(|| format!("rename {}", part.display()))?;
    Ok(PathBuf::from(INSTALLER_COPY))
}

/// OVMF's code and variables beside an OpenCore image of the user's, as reims-util writes them
/// beside the images it makes and a VM's directory holds them.
pub fn firmware_beside(opencore: &Path) -> Option<(PathBuf, PathBuf)> {
    let dir = opencore.parent()?;
    let (code, vars) = (dir.join("OVMF_CODE.fd"), dir.join("OVMF_VARS.fd"));
    (code.is_file() && vars.is_file()).then_some((code, vars))
}

/// The OVMF beside `opencore`, an OpenCore image of the user's, or why there is none.
fn own_firmware(opencore: &Path) -> Result<(PathBuf, PathBuf)> {
    if !opencore.is_file() {
        bail!("the OpenCore image {} does not exist", opencore.display());
    }
    firmware_beside(opencore).with_context(|| {
        format!(
            "no OVMF_CODE.fd and OVMF_VARS.fd beside {}: copy the OVMF code and variables it \
             boots with beside it under those names, or give no OpenCore image to have one made \
             for the VM",
            opencore.display()
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A MAC QEMU would refuse stops a create before it makes anything.
    #[test]
    fn a_mac_qemu_refuses_stops_a_create_at_once() {
        let env = crate::testenv::Env::new("create-mac");
        let mut spec = Spec::new("t");
        spec.dir = Some(env.root.join("t"));
        spec.mac = Some("52:54:00:12:34:56,romfile=/x".into());
        let e = create(spec, Host::current(), &mut |_| {}).unwrap_err();
        assert!(format!("{e:#}").starts_with("a MAC is six pairs"), "{e:#}");
        assert!(!env.root.join("t").exists());
    }

    /// `create --ssh-port 0`, and 0 in the interface's field, mean no forward, as `set
    /// --ssh-port 0` does; QEMU's hostfwd on port 0 would forward a port nobody is told.
    #[test]
    fn a_new_vms_ssh_port_of_0_is_no_forward() {
        let mut spec = Spec::new("t");
        let config = |spec: &Spec| {
            base_config(
                spec,
                Profile::Q35X86_64,
                "disk.qcow2".into(),
                ImageFormat::Qcow2,
            )
        };
        spec.ssh_port = Some(0);
        assert_eq!(config(&spec).network.ssh_port, None);
        spec.ssh_port = Some(2222);
        assert_eq!(config(&spec).network.ssh_port, Some(2222));
    }

    #[test]
    fn a_dmg_is_known_by_its_udif_trailer() {
        let d = std::env::temp_dir().join(format!("reims-ctl-dmg-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let mut udif = vec![0u8; 4096];
        udif[4096 - 512..4096 - 508].copy_from_slice(b"koly");
        std::fs::write(d.join("udzo.dmg"), &udif).unwrap();
        // A read/write DMG is a raw image, which ends in the backup GPT header.
        let mut raw = vec![0u8; 4096];
        raw[4096 - 512..4096 - 504].copy_from_slice(b"EFI PART");
        std::fs::write(d.join("udrw.dmg"), &raw).unwrap();
        std::fs::write(d.join("short.dmg"), b"koly").unwrap();
        assert!(is_dmg(&d.join("udzo.dmg")).unwrap());
        assert!(!is_dmg(&d.join("udrw.dmg")).unwrap());
        assert!(!is_dmg(&d.join("short.dmg")).unwrap());
        // One that is not is used in place, without qemu-img.
        assert_eq!(
            installer_media(&d.join("udrw.dmg"), &d, &mut |_| {}).unwrap(),
            d.join("udrw.dmg")
        );
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn a_disk_given_a_new_opencore_image_is_told_of_setup_assistant() {
        let spec = |disk, opencore: Option<&str>| Spec {
            disk,
            opencore: opencore.map(PathBuf::from),
            ..Spec::new("mac")
        };
        let existing = || {
            Some(DiskSource::Existing {
                path: "/vms/mac/disk.qcow2".into(),
                mode: ImportMode::Copy,
            })
        };
        let note = new_mac_note(&spec(existing(), None)).expect("a note");
        assert!(
            note.contains("Setup Assistant") && note.contains("--opencore"),
            "{note}"
        );
        assert_eq!(
            new_mac_note(&spec(existing(), Some("/vms/mac/opencore.qcow2"))),
            None
        );
        let new = Some(DiskSource::New { size_mib: 131_072 });
        assert_eq!(new_mac_note(&spec(new, None)), None);
    }

    /// A create given an existing disk and no image makes the VM an image presenting the Mac the
    /// image beside the disk presents, the one its macOS ran on, with OVMF beside it, and gives
    /// the VM's network card that Mac's ROM: macOS on the disk sees no new Mac, so it opens no
    /// Setup Assistant, which held up every stop (`new_mac_note`). Where no image is beside the
    /// disk, the image presents a Mac of its own, with the VM's card as its ROM, and the create
    /// says what that does. A MAC given is the VM's either way. A copy of the disk of a VM
    /// reims-ctl has runs beside that VM, so it is a Mac of its own; given in place, that disk
    /// keeps its Mac.
    #[test]
    fn a_disk_keeps_the_mac_of_the_image_beside_it() {
        use crate::host::{Arch, Os};
        use crate::opencore::tests::{image, plist, qcow2};
        use std::os::unix::fs::PermissionsExt;
        let env = crate::testenv::Env::new("create-disks-mac");
        let args = env.root.join("args");
        let made = env.root.join("made.qcow2");
        std::fs::write(&made, qcow2(&image(plist("").as_bytes()))).unwrap();
        let util = env.root.join("reims-util");
        let script = format!(
            "#!/bin/sh\nprintf '%s\\n' \"$@\" > '{args}'\n\
             while [ \"$1\" != -o ]; do shift; done\ncp '{made}' \"$2\"\n\
             echo '{{\"event\":\"done\",\"result\":{{}}}}'\n",
            args = args.display(),
            made = made.display(),
        );
        std::fs::write(&util, script).unwrap();
        std::fs::set_permissions(&util, std::fs::Permissions::from_mode(0o755)).unwrap();
        env.set("REIMS_CTL_UTIL", &util);
        let host = Host {
            os: Os::Linux,
            arch: Arch::X86_64,
        };
        // A VM's directory as another VM left it: its disk, and the image it booted from.
        let old = env.root.join("old");
        std::fs::create_dir_all(&old).unwrap();
        std::fs::write(old.join("disk.qcow2"), b"a disk").unwrap();
        std::fs::write(
            old.join("opencore.qcow2"),
            qcow2(&image(plist("").as_bytes())),
        )
        .unwrap();
        let alone = env.root.join("alone");
        std::fs::create_dir_all(&alone).unwrap();
        std::fs::write(alone.join("disk.img"), b"a disk").unwrap();
        let create_in = |name: &str, disk: PathBuf, mode: ImportMode, mac: Option<&str>| {
            let mut spec = Spec::new(name);
            spec.dir = Some(env.root.join("vms").join(name));
            spec.disk = Some(DiskSource::Existing { path: disk, mode });
            spec.mac = mac.map(String::from);
            let mut said = Vec::new();
            let vm = create(spec, host, &mut |s| said.push(s.to_string())).unwrap();
            let given = std::fs::read_to_string(&args).unwrap();
            let given: Vec<String> = given.lines().map(String::from).collect();
            (vm, given, said.join("\n"))
        };
        let create_from = |name: &str, disk: PathBuf, mac: Option<&str>| {
            create_in(name, disk, ImportMode::Copy, mac)
        };
        let value = |given: &[String], flag: &str| {
            given
                .windows(2)
                .find(|w| w[0] == flag)
                .map(|w| w[1].clone())
        };

        let (vm, given, said) = create_from("kept", old.join("disk.qcow2"), None);
        assert_eq!(value(&given, "--model").as_deref(), Some("iMac20,1"));
        assert_eq!(value(&given, "--serial").as_deref(), Some("C02D40VEPN5T"));
        assert_eq!(value(&given, "--mlb").as_deref(), Some("C02029100GU0000JA"));
        assert_eq!(
            value(&given, "--uuid").as_deref(),
            Some("4A2B8C1D-0E3F-4A5B-8C6D-7E8F9A0B1C2D")
        );
        assert_eq!(value(&given, "--rom").as_deref(), Some("525400010203"));
        assert!(!given.iter().any(|a| a == "--no-ovmf"), "{given:?}");
        assert_eq!(vm.config.network.mac, "52:54:00:01:02:03");
        assert!(!said.contains("Setup Assistant"), "{said}");
        let q = vm.config.q35.as_ref().unwrap();
        assert_eq!(q.opencore, Path::new("opencore.qcow2"));
        assert!(vm.dir.join("opencore.qcow2").is_file());
        // The image beside the disk is read, not used or changed.
        assert_eq!(
            std::fs::read(old.join("opencore.qcow2")).unwrap(),
            qcow2(&image(plist("").as_bytes()))
        );

        let (vm, given, _) =
            create_from("given", old.join("disk.qcow2"), Some("52:54:00:aa:bb:cc"));
        assert_eq!(value(&given, "--serial").as_deref(), Some("C02D40VEPN5T"));
        assert_eq!(value(&given, "--rom").as_deref(), Some("525400010203"));
        assert_eq!(vm.config.network.mac, "52:54:00:aa:bb:cc");

        let (vm, given, said) = create_from("new", alone.join("disk.img"), None);
        assert_eq!(value(&given, "--serial"), None);
        assert_eq!(value(&given, "--rom"), Some(vm.config.network.mac.clone()));
        assert!(said.contains("Setup Assistant"), "{said}");

        // kept's disk, beside the image made for it, which presents the disk's Mac.
        let kepts = env.root.join("vms/kept/disk.img");
        let (_, given, said) = create_from("clone", kepts.clone(), None);
        assert_eq!(value(&given, "--serial"), None);
        assert!(
            said.contains("copy of kept's disk") && said.contains("Setup Assistant"),
            "{said}"
        );
        let (_, given, said) = create_in("same", kepts, ImportMode::InPlace, None);
        assert_eq!(value(&given, "--serial").as_deref(), Some("C02D40VEPN5T"));
        assert!(!said.contains("Setup Assistant"), "{said}");
    }

    #[test]
    fn a_roms_network_card_is_its_six_bytes_unless_multicast() {
        assert_eq!(
            mac_of_rom("525400010203").as_deref(),
            Some("52:54:00:01:02:03")
        );
        assert_eq!(
            mac_of_rom("A4B1C2D3E4F5").as_deref(),
            Some("a4:b1:c2:d3:e4:f5")
        );
        assert_eq!(mac_of_rom("015400010203"), None);
        assert_eq!(mac_of_rom("5254000102"), None);
        assert_eq!(mac_of_rom("52540001020304"), None);
        assert_eq!(mac_of_rom("52540001020G"), None);
    }

    #[test]
    fn firmware_comes_from_beside_the_opencore_image() {
        let d = std::env::temp_dir().join(format!("reims-ctl-ovmf-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let oc = d.join("OpenCore.qcow2");
        std::fs::write(d.join("OVMF_CODE.fd"), b"").unwrap();
        assert_eq!(firmware_beside(&oc), None);
        std::fs::write(d.join("OVMF_VARS.fd"), b"").unwrap();
        assert_eq!(
            firmware_beside(&oc),
            Some((d.join("OVMF_CODE.fd"), d.join("OVMF_VARS.fd")))
        );
        std::fs::remove_dir_all(&d).unwrap();
    }

    /// A create whose directory another create filled during its download (the interface
    /// makes VMs side by side) leaves that VM's files be.
    #[test]
    fn a_create_finds_its_directory_taken_during_its_download() {
        use crate::host::{Arch, Os};
        use std::os::unix::fs::PermissionsExt;
        let env = crate::testenv::Env::new("create-taken");
        let dir = env.root.join("vms/b");
        // reims-util's download, during which another create makes its VM there.
        let util = env.root.join("reims-util");
        let done = r#"{"event":"done","result":{"title":"macOS Tahoe 26","image":"/x.dmg"}}"#;
        let script = format!(
            "#!/bin/sh\nmkdir -p '{d}' && echo theirs > '{d}/disk.qcow2'\necho '{done}'\n",
            d = dir.display()
        );
        std::fs::write(&util, script).unwrap();
        std::fs::set_permissions(&util, std::fs::Permissions::from_mode(0o755)).unwrap();
        env.set("REIMS_CTL_UTIL", &util);
        let mut spec = Spec::new("a");
        spec.dir = Some(dir.clone());
        spec.profile = Some(Profile::Q35X86_64);
        spec.disk = Some(DiskSource::New { size_mib: 1024 });
        spec.macos = Some(Download {
            version: "tahoe".into(),
            offline: false,
        });
        let host = Host {
            os: Os::Linux,
            arch: Arch::X86_64,
        };
        let r = create(spec, host, &mut |_| {});
        assert_eq!(
            std::fs::read_to_string(dir.join("disk.qcow2"))
                .ok()
                .as_deref(),
            Some("theirs\n")
        );
        let e = format!("{:#}", r.unwrap_err());
        assert!(e.contains("is not empty"), "{e}");
        assert!(Registry::load().unwrap().find("a").is_none());
    }

    /// Of two creates into one directory, the one that takes it second finds it taken.
    #[test]
    fn a_directory_is_claimed_once() {
        let d = std::env::temp_dir().join(format!("reims-ctl-claim-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        assert!(claim(&d).unwrap());
        assert!(claim(&d).is_err());
        std::fs::remove_file(d.join(CREATING)).unwrap();
        assert!(!claim(&d).unwrap());
        std::fs::remove_dir_all(&d).unwrap();
    }
}
