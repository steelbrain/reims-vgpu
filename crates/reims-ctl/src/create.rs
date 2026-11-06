//! Making a VM: its directory, disks, firmware and `vm.toml`, then its registry entry.
//!
//! An x86-64 VM (`q35-x86_64`) boots OpenCore from an OpenCore image of its own, with its own
//! OVMF code and variables: made for it by reims-util, holding the guest's kexts for
//! reims-vgpu's own device where the VM is on it ([`reims_util::kext_args`]), or copies of an
//! image given and the OVMF beside it ([`firmware_beside`]). Its disk is a new one or an
//! existing one (copied, reflinked where the file system can, or used in place). A new disk
//! needs an installer: one reims-util downloads from Apple (reims_util.rs), before the VM's
//! directory is made, so a download stopped part way leaves no VM half made, or one given. An
//! installer is used in place, except a DMG, which qemu-img converts to a raw image in the VM's
//! directory ([`installer_media`]).
//!
//! An arm64 VM (`vmapple-arm64`) is one Virtualization.framework made: installed here from an
//! IPSW (vz.rs), or imported from macosvm's or tart's directory (an APFS clone of its disk and
//! auxiliary storage, or used in place).

use crate::host::{Host, Profile};
use crate::install;
use crate::paths;
use crate::reims_util::{self, Download};
use crate::vm::*;
use anyhow::{Context, Result, bail};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImportMode {
    /// A copy in the VM's directory: reflinked (btrfs, XFS) or cloned (APFS) where it can be.
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
    /// The graphics device; none for the profile's default ([`Graphics::default_for`]).
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
    // vmapple
    pub ipsw: Option<String>,
    pub bundle: Option<PathBuf>,
    pub bundle_mode: ImportMode,
    pub ecid: Option<u64>,
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
            ipsw: None,
            bundle: None,
            bundle_mode: ImportMode::Copy,
            ecid: None,
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
        None => host
            .default_profile()
            .with_context(|| format!("reims-ctl runs no VMs on {host} yet"))?,
    };
    if !Profile::supported_on(host).contains(&profile) {
        bail!("{host} cannot run {profile} VMs");
    }
    let graphics = spec
        .graphics
        .unwrap_or_else(|| Graphics::default_for(profile));
    if graphics == Graphics::ReimsVgpu {
        if profile != Profile::Q35X86_64 {
            bail!(
                "reims-vgpu's own GPU device is for x86-64 VMs: an arm64 macOS guest drives only \
                 apple-gfx"
            );
        }
        let d = &spec.display;
        check_reims_vgpu_display(Some(d.count), Some(&d.resolution), Some(d.hidpi), d.refresh)?;
        // The image reims-util makes holds them.
        if spec.opencore.is_none() {
            crate::prebuilt::set()?;
        }
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
    if profile == Profile::Q35X86_64 {
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
    } else if spec.macos.is_some() {
        bail!("an arm64 VM installs macOS from an IPSW (--ipsw latest)");
    }
    let made = claim(&dir)?;
    let result = match profile {
        Profile::Q35X86_64 => make_q35(&spec, graphics, &dir, macos.as_deref(), progress),
        Profile::VmappleArm64 => make_vmapple(&spec, host, &dir, progress),
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
        graphics: Some(
            spec.graphics
                .unwrap_or_else(|| Graphics::default_for(profile)),
        ),
        video_acceleration: true,
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
        vmapple: None,
        extra_env: BTreeMap::new(),
    }
}

fn file_name(p: &Path) -> String {
    p.file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Copies `src` to `dst`: a clone on APFS, a reflink where Linux's file system can, else a
/// sparse copy.
pub fn copy_file(src: &Path, dst: &Path) -> Result<()> {
    let ok = if cfg!(target_os = "macos") {
        Command::new("cp")
            .arg("-c")
            .arg(src)
            .arg(dst)
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
            || Command::new("cp")
                .arg(src)
                .arg(dst)
                .status()
                .map(|s| s.success())
                .unwrap_or(false)
    } else {
        Command::new("cp")
            .args(["--reflink=auto", "--sparse=always"])
            .arg(src)
            .arg(dst)
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    };
    if !ok {
        bail!("could not copy {} to {}", src.display(), dst.display());
    }
    Ok(())
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

    // OpenCore and OVMF: made for the VM, or copies of what was given. A Mac's ROM is its first
    // network card's address, so a made image's is the VM's.
    let mac = spec.mac.clone().unwrap_or_else(random_mac);
    let (oc_name, oc_format) = match &spec.opencore {
        None => {
            let kexts = reims_util::kext_args(graphics)?;
            reims_util::opencore(&dir.join("opencore.qcow2"), &mac, macos, &kexts, progress)?;
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
            copy_file(src, &dir.join(&name))?;
            (PathBuf::from(name), format)
        }
    };
    // An image given brings its OVMF beside it, as a made one's is written beside it.
    if let Some(src) = &spec.opencore {
        let (code, vars) = own_firmware(src)?;
        progress("copying the firmware");
        copy_file(&code, &dir.join("OVMF_CODE.fd"))?;
        copy_file(&vars, &dir.join("OVMF_VARS.fd"))?;
    }
    // A read-only source's copy is read-only too; the variables are written.
    let _ = make_writable(&dir.join("OVMF_VARS.fd"));

    let installer = match spec.installer.clone() {
        Some(i) => Some(installer_media(&i, dir, progress)?),
        None => None,
    };
    let mut c = base_config(spec, Profile::Q35X86_64, disk, disk_format);
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
            progress(&format!(
                "copying {} (a reflink or clone where the file system can)",
                file_name(&src)
            ));
            let ext = if format == ImageFormat::Qcow2 {
                "qcow2"
            } else {
                "img"
            };
            let name = format!("{stem}.{ext}");
            copy_file(&src, &dir.join(&name))?;
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

fn make_vmapple(spec: &Spec, host: Host, dir: &Path, progress: &mut dyn FnMut(&str)) -> Result<Vm> {
    if let Some(bundle) = &spec.bundle {
        let b = Bundle::read(bundle)?;
        let ecid = spec
            .ecid
            .or(b.ecid)
            .context("the bundle's machine identifier was not found: give its ECID (--ecid)")?;
        let (disk, aux) = match spec.bundle_mode {
            ImportMode::InPlace => (b.disk.clone(), b.aux.clone()),
            ImportMode::Copy => {
                progress(
                    "cloning the disk and auxiliary storage (APFS clones where the volume allows)",
                );
                copy_file(&b.disk, &dir.join("disk.img"))?;
                copy_file(&b.aux, &dir.join("aux.img"))?;
                (PathBuf::from("disk.img"), PathBuf::from("aux.img"))
            }
        };
        let mut c = base_config(spec, Profile::VmappleArm64, disk, ImageFormat::Raw);
        if spec.cpus.is_none()
            && let Some(n) = b.cpus
        {
            c.cpus = n;
        }
        if spec.memory_mib.is_none()
            && let Some(m) = b.memory_mib
        {
            c.memory_mib = m;
        }
        c.vmapple = Some(Vmapple { aux, ecid });
        let vm = Vm {
            dir: dir.to_path_buf(),
            config: c,
        };
        crate::runtime::ensure_aux_trimmed(&vm)?;
        return Ok(vm);
    }
    if let Some(ipsw) = &spec.ipsw {
        let size = match &spec.disk {
            Some(DiskSource::New { size_mib }) => *size_mib,
            None => 64 * 1024,
            Some(DiskSource::Existing { .. }) => {
                bail!("an IPSW installs onto a new disk: give its size, not an image")
            }
        };
        let mut c = base_config(
            spec,
            Profile::VmappleArm64,
            "disk.img".into(),
            ImageFormat::Raw,
        );
        let ecid = crate::vz::install(ipsw, dir, size, c.cpus, c.memory_mib, progress)?;
        c.vmapple = Some(Vmapple {
            aux: "aux.img".into(),
            ecid,
        });
        let vm = Vm {
            dir: dir.to_path_buf(),
            config: c,
        };
        crate::runtime::ensure_aux_trimmed(&vm)?;
        let _ = host;
        return Ok(vm);
    }
    bail!(
        "an arm64 VM is installed from an IPSW (--ipsw PATH, or --ipsw latest), or imported from a Virtualization.framework VM's directory (--bundle)"
    )
}

// ---------------------------------------------------------------- Virtualization.framework VMs

/// A Virtualization.framework VM's files, as macosvm or tart keep them.
#[derive(Clone, Debug)]
pub struct Bundle {
    pub disk: PathBuf,
    pub aux: PathBuf,
    pub ecid: Option<u64>,
    pub cpus: Option<u32>,
    pub memory_mib: Option<u64>,
}

impl Bundle {
    pub fn read(dir: &Path) -> Result<Bundle> {
        let dir = std::fs::canonicalize(dir)
            .with_context(|| format!("{} does not exist", dir.display()))?;
        let disk = dir.join("disk.img");
        if !disk.is_file() {
            bail!(
                "{} has no disk.img: not a macosvm or tart VM",
                dir.display()
            );
        }
        // macosvm: vm.json, aux.img; its machineId is the identifier's data, in base64.
        let vm_json = dir.join("vm.json");
        if vm_json.is_file() && dir.join("aux.img").is_file() {
            let j: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&vm_json)?)?;
            let ecid = j
                .get("machineId")
                .and_then(|v| v.as_str())
                .and_then(|b| ecid_from_identifier_base64(b).ok());
            let cpus = j.get("cpus").and_then(|v| v.as_u64()).map(|n| n as u32);
            let memory_mib = j
                .get("ram")
                .and_then(|v| v.as_u64())
                .map(|b| b / (1024 * 1024));
            return Ok(Bundle {
                disk,
                aux: dir.join("aux.img"),
                ecid,
                cpus,
                memory_mib,
            });
        }
        // tart: config.json with "ecid" (the identifier's data, in base64), nvram.bin.
        let config = dir.join("config.json");
        if config.is_file() && dir.join("nvram.bin").is_file() {
            let j: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&config)?)?;
            let ecid = j
                .get("ecid")
                .and_then(|v| v.as_str())
                .and_then(|b| ecid_from_identifier_base64(b).ok());
            let cpus = j.get("cpuCount").and_then(|v| v.as_u64()).map(|n| n as u32);
            let memory_mib = j
                .get("memorySize")
                .and_then(|v| v.as_u64())
                .map(|b| b / (1024 * 1024));
            return Ok(Bundle {
                disk,
                aux: dir.join("nvram.bin"),
                ecid,
                cpus,
                memory_mib,
            });
        }
        if dir.join("aux.img").is_file() {
            return Ok(Bundle {
                disk,
                aux: dir.join("aux.img"),
                ecid: None,
                cpus: None,
                memory_mib: None,
            });
        }
        bail!(
            "{} has a disk.img but no aux.img (macosvm) or nvram.bin (tart)",
            dir.display()
        )
    }
}

/// The ECID in a `VZMacMachineIdentifier`'s data representation (a property list), given in
/// base64.
pub fn ecid_from_identifier_base64(b64: &str) -> Result<u64> {
    let data = base64_decode(b64)?;
    ecid_from_identifier(&data)
}

pub fn ecid_from_identifier(data: &[u8]) -> Result<u64> {
    let mut child = Command::new("plutil")
        .args(["-extract", "ECID", "raw", "-o", "-", "-"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .context("run plutil")?;
    use std::io::Write;
    child.stdin.take().unwrap().write_all(data)?;
    let out = child.wait_with_output()?;
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    s.parse()
        .with_context(|| format!("no ECID in the machine identifier ({s:?})"))
}

pub fn base64_decode(s: &str) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    let mut acc: u32 = 0;
    let mut bits = 0;
    for c in s.bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            b'=' | b'\n' | b'\r' | b' ' => continue,
            _ => bail!("not base64"),
        } as u32;
        acc = (acc << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    Ok(out)
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
                Profile::VmappleArm64,
                "disk.img".into(),
                ImageFormat::Raw,
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

    #[test]
    fn base64() {
        assert_eq!(base64_decode("aGVsbG8=").unwrap(), b"hello");
        assert_eq!(base64_decode("aGVsbG8gd29ybGQ").unwrap(), b"hello world");
    }
}
