//! A VM: its directory, and the settings in its `vm.toml`.
//!
//! A VM's directory holds its `vm.toml`, its disks (unless a disk was used in place), its
//! firmware copies, `qemu.pid` while QEMU runs, the QMP socket where the path is short enough,
//! and `logs/`. The registry (`vms.toml` in the config directory) lists each VM by its name and
//! directory, so a VM may live anywhere.

use crate::host::Profile;
use crate::paths;
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub const CONFIG_FILE: &str = "vm.toml";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VmConfig {
    pub name: String,
    pub profile: Profile,
    pub cpus: u32,
    pub memory_mib: u64,
    /// The system disk, relative to the VM's directory unless absolute.
    pub disk: PathBuf,
    #[serde(default)]
    pub disk_format: ImageFormat,
    /// How long `stop` waits for the guest to shut down before it powers the VM off; 0 for no
    /// limit ([`VmConfig::stop_timeout`]).
    #[serde(default = "default_stop_timeout")]
    pub stop_timeout_secs: u64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub extra_args: Vec<String>,
    /// The guest's graphics device ([`VmConfig::graphics`]): None, as settings an older
    /// reims-ctl saved may have it, is reims-vgpu's own device, the default. A create saves it
    /// by name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graphics: Option<Graphics>,
    #[serde(default)]
    pub display: Display,
    #[serde(default)]
    pub gpu: Gpu,
    #[serde(default)]
    pub audio: Audio,
    #[serde(default)]
    pub network: Network,
    #[serde(default)]
    pub logs: Logs,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub q35: Option<Q35>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extra_env: BTreeMap<String, String>,
    /// The host's USB devices the guest is given while the VM runs (usb.rs), x86-64 VMs'.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub usb: Vec<UsbDevice>,
}

fn default_stop_timeout() -> u64 {
    DEFAULT_STOP_TIMEOUT.as_secs()
}

/// How long `stop` waits for the guest by default.
pub const DEFAULT_STOP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(90);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ImageFormat {
    #[default]
    Qcow2,
    Raw,
}

impl ImageFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            ImageFormat::Qcow2 => "qcow2",
            ImageFormat::Raw => "raw",
        }
    }

    /// The format of the image at `path`, from its first bytes: qcow2's magic, else raw.
    pub fn detect(path: &Path) -> Result<ImageFormat> {
        use std::io::Read;
        let mut magic = [0u8; 4];
        let mut f =
            std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
        let n = f.read(&mut magic)?;
        Ok(if n == 4 && magic == *b"QFI\xfb" {
            ImageFormat::Qcow2
        } else {
            ImageFormat::Raw
        })
    }
}

/// The guest's graphics device.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Graphics {
    /// reims-vgpu's own GPU device, QEMU's `reims-graphics-pci`: the guest drives it with
    /// reims-vgpu's kexts, which reims-ctl puts in the VM's OpenCore image
    /// ([`crate::prebuilt`]), on one display in a window of the library's. The default. Settings
    /// an older reims-ctl saved for apple-gfx, Apple's paravirtualized GPU, which reims-vgpu no
    /// longer has, read as it ([`crate::migrate`] makes their OpenCore image again for it).
    #[default]
    #[serde(alias = "apple-gfx")]
    ReimsVgpu,
    /// QEMU's VMware SVGA II: one display, the firmware's framebuffer, with no GPU acceleration
    /// (macOS has no driver for the device), in QEMU's own window.
    VmwareSvga,
}

impl Graphics {
    pub const ALL: [Graphics; 2] = [Graphics::ReimsVgpu, Graphics::VmwareSvga];

    pub fn id(self) -> &'static str {
        match self {
            Graphics::ReimsVgpu => "reims-vgpu",
            Graphics::VmwareSvga => "vmware-svga",
        }
    }

    /// Its name, short, for `list`.
    pub fn name(self) -> &'static str {
        match self {
            Graphics::ReimsVgpu => "reims-vgpu",
            Graphics::VmwareSvga => "VMware SVGA",
        }
    }

    /// Its name for the interface's choice and `show`.
    pub fn describe(self) -> &'static str {
        match self {
            Graphics::ReimsVgpu => "reims-vgpu, its own GPU device and macOS drivers",
            Graphics::VmwareSvga => "VMware SVGA, no GPU acceleration",
        }
    }

    pub fn parse(s: &str) -> Result<Graphics> {
        Graphics::ALL
            .into_iter()
            .find(|g| g.id() == s.trim())
            .with_context(|| format!("graphics is reims-vgpu or vmware-svga, not {s:?}"))
    }

    /// Whether the library runs the device, and draws its display in its window.
    pub fn library(self) -> bool {
        self != Graphics::VmwareSvga
    }
}

/// The controller an x86-64 VM's system disk is on, which its settings choose.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DiskController {
    /// A SATA disk on ICH9's AHCI controller, at [`Q35::disk_port`]: the disk of every VM
    /// reims-ctl makes, whose settings name no controller. macOS 26's installer, as reims-ctl
    /// downloads it, may find no disk on NVMe.
    #[default]
    Ahci,
    /// An NVMe controller of its own, for a VM with macOS installed: macOS finds its APFS
    /// container on either controller.
    Nvme,
}

impl DiskController {
    pub const ALL: [DiskController; 2] = [DiskController::Ahci, DiskController::Nvme];

    pub fn id(self) -> &'static str {
        match self {
            DiskController::Ahci => "ahci",
            DiskController::Nvme => "nvme",
        }
    }

    pub fn describe(self) -> &'static str {
        match self {
            DiskController::Ahci => "SATA",
            DiskController::Nvme => "NVMe",
        }
    }

    pub fn parse(s: &str) -> Result<DiskController> {
        DiskController::ALL
            .into_iter()
            .find(|d| d.id() == s.trim())
            .with_context(|| format!("the disk controller is ahci or nvme, not {s:?}"))
    }

    fn is_ahci(&self) -> bool {
        *self == DiskController::Ahci
    }
}

/// reims-vgpu's one display: its window opens at the resolution's size
/// ([`Display::window_size`]) and keeps the size the person gives it, which the guest follows, at
/// the refresh rate of the host screen it is on. The guest starts a HiDPI desktop by itself in a
/// window at least 3840 pixels wide, and offers one in its Displays settings from 2880 (the
/// kext's HiDPI modes, whose pixels halve exactly: the guest is offered the window's pixels, the
/// width rounded down to a multiple of 16 and the height to even). Settings an older reims-ctl saved for apple-gfx's displays (their
/// count, HiDPI and refresh rate) are read past, and left out of the next save.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Display {
    /// The size the window opens at, `<width>x<height>`.
    pub resolution: String,
    /// A borderless window that fills its screen.
    pub borderless: bool,
}

impl Default for Display {
    fn default() -> Display {
        Display {
            resolution: "1920x1080".into(),
            borderless: false,
        }
    }
}

impl Display {
    /// The size reims-vgpu's window opens at, the resolution's, in the host's points (a guest
    /// point per host point), shrunk to fit the screen, which its guest takes in pixels and
    /// follows: measured on a Linux desktop at 2.1x, 1280x720 opened 1280x720 points and the
    /// guest ran 2688x1512. None for the device's own, 1920x1080.
    pub fn window_size(&self) -> Option<(u32, u32)> {
        parse_resolution(self.resolution.trim())
            .ok()
            .filter(|&size| size != (1920, 1080))
    }

    pub fn check(&self) -> Result<()> {
        parse_resolution(self.resolution.trim()).map(|_| ())
    }
}

pub fn parse_resolution(s: &str) -> Result<(u32, u32)> {
    let (w, h) = s
        .split_once(['x', 'X'])
        .with_context(|| format!("resolution must be <width>x<height>, not {s:?}"))?;
    let w: u32 = w
        .trim()
        .parse()
        .with_context(|| format!("bad width in {s:?}"))?;
    let h: u32 = h
        .trim()
        .parse()
        .with_context(|| format!("bad height in {s:?}"))?;
    if !(1..=65535).contains(&w) || !(1..=65535).contains(&h) {
        bail!("each side of a resolution is 1 to 65535 pixels, not {s:?}");
    }
    Ok((w, h))
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Gpu {
    /// The Vulkan driver's manifest (ICD JSON) the loader is limited to: `VK_DRIVER_FILES`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub driver_file: Option<PathBuf>,
    /// The PCI device ID the loader keeps, where the driver drives several GPUs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device_id: Option<u32>,
    /// The GPU's name, for display.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Audio {
    pub enabled: bool,
    /// QEMU's audio backend (`pipewire`, `pa`, `alsa`, `none`); the host's best
    /// by default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,
}

impl Default for Audio {
    fn default() -> Audio {
        Audio {
            enabled: true,
            backend: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Network {
    pub enabled: bool,
    pub mac: String,
    /// A host port on 127.0.0.1 forwarded to the guest's SSH port ([`ssh_port`]).
    #[serde(
        skip_serializing_if = "Option::is_none",
        deserialize_with = "saved_ssh_port"
    )]
    pub ssh_port: Option<u16>,
}

/// The SSH forward from host port `p`: none for 0, wherever a port is given, as QEMU's
/// `hostfwd` on port 0 forwards a port nobody is told.
pub fn ssh_port(p: u16) -> Option<u16> {
    (p != 0).then_some(p)
}

/// A saved SSH port: 0, which create saved for `--ssh-port 0` before it meant none there too,
/// is none.
fn saved_ssh_port<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> std::result::Result<Option<u16>, D::Error> {
    Ok(Option::<u16>::deserialize(d)?.and_then(ssh_port))
}

impl Default for Network {
    fn default() -> Network {
        Network {
            enabled: true,
            mac: random_mac(),
            ssh_port: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Logs {
    /// The library's log at `level`, the guest's serial console and QEMU's timestamps, into
    /// `logs/`. Off, QEMU's log still keeps QEMU's own errors, and the library's warnings and
    /// device resets, which reims-ctl's warnings read.
    pub enabled: bool,
    /// `REIMS_VGPU_LOG`'s filter: a level, and `target=level` pairs.
    pub level: String,
}

impl Default for Logs {
    fn default() -> Logs {
        Logs {
            enabled: false,
            level: "info".into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Q35 {
    /// The OpenCore image, in the VM's directory: made for it, or a copy of one given.
    pub opencore: PathBuf,
    #[serde(default)]
    pub opencore_format: ImageFormat,
    /// OVMF's code and variables (copies in the VM's directory).
    pub firmware_code: PathBuf,
    pub firmware_vars: PathBuf,
    /// The installer, a raw image (reims-util's offline installer, or a recovery image's raw
    /// copy) attached as a disk, whose writes go to a temporary overlay: used in place where
    /// absolute; `installer.img`, the VM's raw copy of a DMG (`create::installer_media`), which a
    /// change of installer deletes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub installer: Option<PathBuf>,
    #[serde(default = "default_cpu_model")]
    pub cpu_model: String,
    #[serde(default = "default_cpu_flags")]
    pub cpu_flags: String,
    /// The system disk's controller: SATA for every VM reims-ctl makes, which its settings may
    /// move to NVMe once macOS is installed. OpenCore's saved boot entry names the controller
    /// and the port, so after a move an attached installer may boot instead.
    #[serde(default, skip_serializing_if = "DiskController::is_ahci")]
    pub disk_controller: DiskController,
    /// The system disk's AHCI port, on [`DiskController::Ahci`]: 3, after OpenCore's 2.
    #[serde(default = "default_disk_port")]
    pub disk_port: u32,
}

/// Skylake-Client without TSX: macOS boots on it without an OpenCore CPUID spoof, and it has no
/// AVX-512, whose state XNU mishandles on AMD hosts (docs/qemu.md, "x86-64 macOS guests: the CPU
/// model").
pub fn default_cpu_model() -> String {
    "Skylake-Client-noTSX-IBRS".into()
}

pub fn default_cpu_flags() -> String {
    "+ssse3,+sse4.2,+popcnt,+avx,+aes,+xsave,+xsaveopt,check".into()
}

fn default_disk_port() -> u32 {
    3
}

/// A USB device of the host's that the guest is given while the VM runs (usb.rs): one
/// `[[usb]]` table of the settings.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct UsbDevice {
    /// Its vendor and product IDs, `vvvv:pppp` in hex, as lsusb shows them: the device wherever
    /// it is plugged in.
    pub id: String,
    /// Where it is plugged in, for a device chosen while the host had another with its IDs:
    /// Linux's name for the port, such as `3-2.1` (bus 3, the hub on port 2, its port 1). With
    /// it the guest gets only the device in that port.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub port: Option<String>,
    /// Its name when it was chosen, shown while it is not plugged in.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub name: String,
}

/// The most USB devices a VM takes: its xHCI controller's seven ports, less the keyboard's and
/// the tablet's, and one left free, where QEMU would chain a full-speed hub (`qemu`'s module
/// doc).
pub const USB_DEVICES_MAX: usize = 4;

impl UsbDevice {
    /// Its vendor and product IDs.
    pub fn ids(&self) -> Result<(u16, u16)> {
        let bad = || {
            anyhow::anyhow!(
                "a USB device's ID is its vendor and product IDs in hex, as lsusb shows them, \
                 such as 1050:0407, not {:?}",
                self.id
            )
        };
        let (v, p) = self.id.split_once(':').ok_or_else(bad)?;
        let hex = |s: &str| {
            (s.len() == 4 && s.bytes().all(|b| b.is_ascii_hexdigit()))
                .then(|| u16::from_str_radix(s, 16).ok())
                .flatten()
        };
        Ok((hex(v).ok_or_else(bad)?, hex(p).ok_or_else(bad)?))
    }

    /// Its port's bus and the port numbers from there, as QEMU's `hostbus` and `hostport` take
    /// them: `3-2.1` is bus 3 and `2.1`.
    pub fn bus_port(&self) -> Result<Option<(u32, &str)>> {
        let Some(p) = self.port.as_deref() else {
            return Ok(None);
        };
        parse_usb_port(p)
            .map(Some)
            .with_context(|| format!("{}: {:?} names no USB port", self.id, p))
    }

    pub fn check(&self) -> Result<()> {
        self.ids()?;
        self.bus_port()?;
        Ok(())
    }
}

/// A USB port's name as Linux gives it (`3-2.1`): its bus, and the port numbers from the bus's
/// root hub, at most 7 (USB's tiers).
pub fn parse_usb_port(p: &str) -> Option<(u32, &str)> {
    let (bus, ports) = p.split_once('-')?;
    let number = |s: &str| {
        (1..=3).contains(&s.len()) && s.bytes().all(|b| b.is_ascii_digit()) && !s.starts_with('0')
    };
    if !number(bus) || ports.split('.').count() > 7 || !ports.split('.').all(number) {
        return None;
    }
    Some((bus.parse().ok()?, ports))
}

impl VmConfig {
    pub fn check(&self) -> Result<()> {
        check_name(&self.name)?;
        if self.cpus == 0 || self.cpus > 256 {
            bail!("cpus must be 1 to 256, not {}", self.cpus);
        }
        if self.memory_mib < 2048 {
            bail!("memory must be at least 2 GiB, not {} MiB", self.memory_mib);
        }
        self.display.check()?;
        for k in self.extra_env.keys() {
            check_env_name(k)?;
        }
        check_mac(&self.network.mac)?;
        self.check_usb()?;
        if self.q35.is_none() {
            bail!("{}: no [q35] settings", self.name);
        }
        Ok(())
    }

    /// Refuses USB devices the VM cannot be given: more than [`USB_DEVICES_MAX`]; one named
    /// twice, which QEMU would refuse by its id; and an ID or port that names no device.
    fn check_usb(&self) -> Result<()> {
        if self.usb.len() > USB_DEVICES_MAX {
            bail!(
                "{}: a VM takes {USB_DEVICES_MAX} USB devices at most, not {}",
                self.name,
                self.usb.len()
            );
        }
        for (i, u) in self.usb.iter().enumerate() {
            u.check().with_context(|| self.name.clone())?;
            if self.usb[..i]
                .iter()
                .any(|o| crate::usb::qemu_id(o) == crate::usb::qemu_id(u))
            {
                bail!("{}: USB device {} is chosen twice", self.name, u.id);
            }
        }
        Ok(())
    }

    /// How long `stop` waits for the guest to shut down before it powers the VM off: `None`
    /// for a timeout of 0, no limit, as an SSH port of 0 is no forward. A stop then waits until
    /// the guest shuts down, which may ask about unsaved documents for as long as they stay
    /// open; only a forced stop powers the VM off.
    pub fn stop_timeout(&self) -> Option<std::time::Duration> {
        stop_timeout(self.stop_timeout_secs)
    }

    /// The guest's graphics device: the settings', else reims-vgpu's own.
    pub fn graphics(&self) -> Graphics {
        self.graphics.unwrap_or_default()
    }
}

impl VmConfig {
    /// What in these settings the VM's guest does not work with, each a sentence to show the
    /// user: audio off, without which Safari's streaming players fail (docs/qemu.md, "x86-64
    /// macOS guests: audio").
    pub fn incompatibilities(&self) -> Vec<String> {
        let mut w = Vec::new();
        if !self.audio.enabled {
            w.push(format!(
                "Audio is off, and without an audio device an x86-64 macOS guest's Safari can't \
                 play YouTube or other streaming video. Turn it on; to keep the guest silent, \
                 also run reims-ctl set {} --audio-backend none.",
                self.name
            ));
        }
        w
    }
}

/// A stop timeout of `secs` seconds: `None` for 0, no limit ([`VmConfig::stop_timeout`]).
pub fn stop_timeout(secs: u64) -> Option<std::time::Duration> {
    (secs != 0).then(|| std::time::Duration::from_secs(secs))
}

pub fn check_name(name: &str) -> Result<()> {
    let ok = !name.is_empty()
        && name.len() <= 64
        && !name.starts_with(['-', '.'])
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'));
    if !ok {
        bail!(
            "a VM's name is 1 to 64 letters, digits, '-', '_' or '.', not starting with '-' or '.': {name:?}"
        );
    }
    Ok(())
}

/// An environment variable's name for QEMU's environment: letters, digits and '_', not
/// starting with a digit, as the shell's `export` in a printed command takes it.
pub fn check_env_name(k: &str) -> Result<()> {
    let ok = k.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_')
        && k.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
    if !ok {
        bail!(
            "an environment variable's name is letters, digits and '_', not starting with a \
             digit: {k:?}"
        );
    }
    Ok(())
}

/// A MAC as QEMU's `mac` property takes it: six pairs of hex digits, each pair after the first
/// following a ':' or a '-' (hw/core/qdev-properties-system.c, set_mac).
pub fn check_mac(mac: &str) -> Result<()> {
    let b = mac.as_bytes();
    let ok = b.len() == 17
        && b.chunks(3).all(|p| {
            p[0].is_ascii_hexdigit()
                && p[1].is_ascii_hexdigit()
                && p.get(2).is_none_or(|s| matches!(s, b':' | b'-'))
        });
    if !ok {
        bail!(
            "a MAC is six pairs of hex digits joined by ':' or '-', such as \
             52:54:00:12:34:56, not {mac:?}"
        );
    }
    Ok(())
}

/// A locally administered MAC in QEMU's 52:54:00 prefix.
pub fn random_mac() -> String {
    let b = random_bytes::<3>();
    format!("52:54:00:{:02x}:{:02x}:{:02x}", b[0], b[1], b[2])
}

fn random_bytes<const N: usize>() -> [u8; N] {
    use std::io::Read;
    let mut b = [0u8; N];
    if std::fs::File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(&mut b))
        .is_err()
    {
        let t = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();
        let seed = t.as_nanos() ^ (std::process::id() as u128) << 64;
        for (i, x) in b.iter_mut().enumerate() {
            *x = (seed >> (i * 8)) as u8;
        }
    }
    b
}

/// A size such as `8G`, `8GiB`, `512M` or `1.5T`, in MiB; a bare number is GiB.
pub fn parse_size_mib(s: &str) -> Result<u64> {
    let t = s.trim();
    let split = t
        .find(|c: char| !(c.is_ascii_digit() || c == '.'))
        .unwrap_or(t.len());
    let (num, unit) = t.split_at(split);
    let n: f64 = num
        .parse()
        .with_context(|| format!("not a size: {s:?} (try 8G or 512M)"))?;
    let mul = match unit.trim().to_ascii_lowercase().as_str() {
        "" | "g" | "gb" | "gib" => 1024.0,
        "m" | "mb" | "mib" => 1.0,
        "t" | "tb" | "tib" => 1024.0 * 1024.0,
        _ => bail!("not a size: {s:?} (try 8G or 512M)"),
    };
    let mib = (n * mul).round();
    if !(1.0..=1e12).contains(&mib) {
        bail!("not a size: {s:?}");
    }
    Ok(mib as u64)
}

pub fn format_mib(mib: u64) -> String {
    if mib >= 1024 && mib.is_multiple_of(1024) {
        format!("{}G", mib / 1024)
    } else {
        format!("{mib}M")
    }
}

#[derive(Clone, Debug)]
pub struct Vm {
    pub dir: PathBuf,
    pub config: VmConfig,
}

impl Vm {
    pub fn load(dir: &Path) -> Result<Vm> {
        let file = dir.join(CONFIG_FILE);
        let text =
            std::fs::read_to_string(&file).with_context(|| format!("read {}", file.display()))?;
        let config: VmConfig =
            toml::from_str(&text).with_context(|| format!("parse {}", file.display()))?;
        Ok(Vm {
            dir: dir.to_path_buf(),
            config,
        })
    }

    pub fn save(&self) -> Result<()> {
        self.config.check()?;
        let text = toml::to_string_pretty(&self.config)?;
        let file = self.dir.join(CONFIG_FILE);
        let tmp = self.dir.join(format!(".{CONFIG_FILE}.tmp"));
        std::fs::write(&tmp, text).with_context(|| format!("write {}", tmp.display()))?;
        std::fs::rename(&tmp, &file).with_context(|| format!("write {}", file.display()))?;
        Ok(())
    }

    pub fn name(&self) -> &str {
        &self.config.name
    }

    /// `p` against the VM's directory, unless absolute.
    pub fn path(&self, p: &Path) -> PathBuf {
        if p.is_absolute() {
            p.to_path_buf()
        } else {
            self.dir.join(p)
        }
    }

    pub fn disk(&self) -> PathBuf {
        self.path(&self.config.disk)
    }

    pub fn logs_dir(&self) -> PathBuf {
        self.dir.join("logs")
    }

    pub fn qemu_log(&self) -> PathBuf {
        self.logs_dir().join("qemu.log")
    }

    pub fn serial_log(&self) -> PathBuf {
        self.logs_dir().join("serial.log")
    }

    pub fn pidfile(&self) -> PathBuf {
        pidfile_in(&self.dir)
    }

    /// The files the VM runs on, wherever they are: its disk, and its OpenCore image, firmware
    /// and installer.
    pub fn files(&self) -> Vec<PathBuf> {
        let mut v = vec![self.disk()];
        if let Some(q) = &self.config.q35 {
            v.extend([&q.opencore, &q.firmware_code, &q.firmware_vars].map(|p| self.path(p)));
            v.extend(q.installer.as_deref().map(|p| self.path(p)));
        }
        v
    }

    /// The QMP socket ([`qmp_socket_in`] the runtime directory).
    pub fn qmp_socket(&self) -> PathBuf {
        qmp_socket_in(
            &self.dir,
            &self.config.name,
            &paths::runtime_dir(),
            &paths::short_runtime_dir(),
        )
    }
}

/// The pid file QEMU writes in a VM's directory while it runs.
pub fn pidfile_in(dir: &Path) -> PathBuf {
    dir.join("qemu.pid")
}

/// The longest path a Unix socket of reims-ctl's takes: std refuses one of sun_path's size or
/// more, which on macOS is 104 bytes (Linux's is 108), though macOS and QEMU take 104 (measured
/// 2026-10-23, M3 Max).
const SOCKET_PATH_MAX: usize = 103;

/// The QMP socket of the VM named `name` in `dir`: `qmp.sock` in `dir` where that path is
/// shorter than 100 bytes; else `<name>.qmp` in `runtime`, where that fits a socket; else, for
/// a long name, `<hash>.qmp` there, 16 hex digits of the name's FNV-1a hash; else, where a long
/// `runtime` leaves no room even for that, `<hash>.qmp` in `short`, which always fits. A VM's
/// socket stays where earlier reims-ctl put it wherever that path fits, so a VM it started is
/// found.
pub fn qmp_socket_in(dir: &Path, name: &str, runtime: &Path, short: &Path) -> PathBuf {
    let own = dir.join("qmp.sock");
    if own.as_os_str().len() < 100 {
        return own;
    }
    let named = runtime.join(format!("{name}.qmp"));
    if named.as_os_str().len() <= SOCKET_PATH_MAX {
        return named;
    }
    let hash = name.bytes().fold(0xcbf2_9ce4_8422_2325_u64, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    });
    let hashed = runtime.join(format!("{hash:016x}.qmp"));
    if hashed.as_os_str().len() <= SOCKET_PATH_MAX {
        return hashed;
    }
    short.join(format!("{hash:016x}.qmp"))
}

// ---------------------------------------------------------------- the registry

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Registry {
    #[serde(default, rename = "vm")]
    pub vms: Vec<Entry>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Entry {
    pub name: String,
    pub dir: PathBuf,
}

impl Registry {
    pub fn load() -> Result<Registry> {
        let file = paths::registry_file();
        match std::fs::read_to_string(&file) {
            Ok(text) => toml::from_str(&text).with_context(|| format!("parse {}", file.display())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Registry::default()),
            Err(e) => Err(e).with_context(|| format!("read {}", file.display())),
        }
    }

    pub fn save(&self) -> Result<()> {
        let file = paths::registry_file();
        std::fs::create_dir_all(file.parent().unwrap())?;
        let tmp = file.with_extension("toml.tmp");
        std::fs::write(&tmp, toml::to_string_pretty(self)?)?;
        std::fs::rename(&tmp, &file)?;
        Ok(())
    }

    pub fn find(&self, name: &str) -> Option<&Entry> {
        self.vms.iter().find(|e| e.name == name)
    }

    pub fn add(&mut self, name: &str, dir: &Path) -> Result<()> {
        if self.find(name).is_some() {
            bail!("a VM named {name} already exists");
        }
        self.vms.push(Entry {
            name: name.into(),
            dir: dir.to_path_buf(),
        });
        self.vms.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(())
    }

    pub fn remove(&mut self, name: &str) -> Option<Entry> {
        let i = self.vms.iter().position(|e| e.name == name)?;
        Some(self.vms.remove(i))
    }

    /// The VM named `name`.
    pub fn open(&self, name: &str) -> Result<Vm> {
        let e = self
            .find(name)
            .with_context(|| format!("no VM named {name} (reims-ctl list shows them)"))?;
        Vm::load(&e.dir)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A resolution is the window's first size, `<width>x<height>`; reims-vgpu's window opens
    /// at it, and the device's own 1920x1080 names none.
    #[test]
    fn reims_vgpu_s_window_opens_at_the_resolution_s_size() {
        let d = |resolution: &str| Display {
            resolution: resolution.into(),
            ..Default::default()
        };
        assert_eq!(d("2560x1440").window_size(), Some((2560, 1440)));
        assert_eq!(d(" 3840x2160").window_size(), Some((3840, 2160)));
        assert_eq!(d("1920x1080").window_size(), None);
        assert!(d("2560x1600").check().is_ok());
        for bad in ["host", "2560", "19x", "0x1080", "70000x10"] {
            assert!(d(bad).check().is_err(), "{bad}");
            assert_eq!(d(bad).window_size(), None, "{bad}");
        }
    }

    #[test]
    fn sizes() {
        assert_eq!(parse_size_mib("8G").unwrap(), 8192);
        assert_eq!(parse_size_mib("8").unwrap(), 8192);
        assert_eq!(parse_size_mib("512M").unwrap(), 512);
        assert_eq!(parse_size_mib("1.5GiB").unwrap(), 1536);
        assert_eq!(parse_size_mib("1T").unwrap(), 1024 * 1024);
        assert!(parse_size_mib("eight").is_err());
        assert!(parse_size_mib("8Q").is_err());
        assert_eq!(format_mib(8192), "8G");
        assert_eq!(format_mib(1536), "1536M");
    }

    /// A MAC is what QEMU's mac property takes, and nothing more: a ',' would add options to
    /// the network card's.
    #[test]
    fn macs_are_what_qemu_takes() {
        for good in [
            "52:54:00:12:34:56",
            "52-54-00-12-34-56",
            "52:54-00:12-34:56",
            "AA:bb:CC:dd:EE:ff",
        ] {
            assert!(check_mac(good).is_ok(), "{good:?}");
        }
        for bad in [
            "",
            "52:54:00:12:34",
            "52:54:00:12:34:56:78",
            "52:54:00:12:34:5g",
            "52.54.00.12.34.56",
            "525400123456",
            "5254.0012.3456",
            " 52:54:00:12:34:56",
            "52:54:00:12:34:56 ",
            "2:54:00:12:34:567",
            "52:54:00:12:34:56,romfile=/x",
        ] {
            assert!(check_mac(bad).is_err(), "{bad:?}");
        }
        assert!(check_mac(&random_mac()).is_ok());
        let mut c = x86_64_config();
        assert!(c.check().is_ok());
        c.network.mac = "52:54:00:12:34:56,romfile=/x".into();
        assert!(c.check().is_err());
    }

    #[test]
    fn names() {
        assert!(check_name("macos-26.1_b").is_ok());
        for bad in ["", "-a", ".a", "a b", "a/b", &"x".repeat(65)] {
            assert!(check_name(bad).is_err(), "{bad:?}");
        }
    }

    /// Every VM name has a QMP socket that binds, with a runtime directory as long as macOS's
    /// `$TMPDIR/reims-ctl-<uid>` and a VM directory too long for its own; and a VM keeps the
    /// socket earlier reims-ctl gave it wherever that fits.
    #[test]
    fn every_name_has_a_qmp_socket_that_binds() {
        let dir = Path::new("/Users/someone/VMs").join("d".repeat(80));
        let runtime = Path::new("/var/folders/rk/50k022t15gq2j9zcy6wcfkh00000gn/T/reims-ctl-501");
        let short = Path::new("/tmp/reims-ctl-501");
        for n in 1..=64 {
            let s = qmp_socket_in(&dir, &"n".repeat(n), runtime, short);
            assert!(s.as_os_str().len() <= SOCKET_PATH_MAX, "{}", s.display());
            assert_eq!(s.parent(), Some(runtime));
        }
        assert_eq!(
            qmp_socket_in(&dir, "macos-26", runtime, short),
            runtime.join("macos-26.qmp")
        );
        assert_eq!(
            qmp_socket_in(Path::new("/vms/a"), "a", runtime, short),
            Path::new("/vms/a/qmp.sock")
        );
        let long = "a".repeat(64);
        let other = format!("{}b", "a".repeat(63));
        assert_ne!(
            qmp_socket_in(&dir, &long, runtime, short),
            qmp_socket_in(&dir, &other, runtime, short)
        );
        // Bound where it falls, in the runtime directory or the short one: under a `$TMPDIR`
        // too long for even a hashed name, as a CI job's can be, the short one.
        let tmp = std::env::temp_dir().join(format!("reims-ctl-qmp-{}", std::process::id()));
        let short_tmp = Path::new("/tmp").join(format!("reims-ctl-qmp-{}", std::process::id()));
        for d in [&tmp, &short_tmp] {
            std::fs::create_dir_all(d).unwrap();
        }
        let s = qmp_socket_in(&dir, &long, &tmp, &short_tmp);
        let bound = std::os::unix::net::UnixListener::bind(&s)
            .and_then(|_l| std::os::unix::net::UnixStream::connect(&s));
        for d in [&tmp, &short_tmp] {
            let _ = std::fs::remove_dir_all(d);
        }
        assert!(s.as_os_str().len() <= SOCKET_PATH_MAX, "{}", s.display());
        bound.unwrap_or_else(|e| panic!("{}: {e}", s.display()));
    }

    /// A `$TMPDIR` or `$XDG_RUNTIME_DIR` so long that not even a hashed name fits there puts the
    /// socket in the short runtime directory, where every name's fits; one that fits stays
    #[test]
    fn a_long_runtime_dir_puts_the_qmp_socket_in_the_short_one() {
        let dir = Path::new("/Users/someone/VMs").join("d".repeat(80));
        let runtime = Path::new("/Users/someone").join("t".repeat(84));
        let short = Path::new("/tmp/reims-ctl-4294967295");
        for n in 1..=64 {
            let s = qmp_socket_in(&dir, &"n".repeat(n), &runtime, short);
            assert!(s.as_os_str().len() <= SOCKET_PATH_MAX, "{}", s.display());
            assert_eq!(s.parent(), Some(short), "{n}");
        }
        let fits = Path::new("/Users/someone").join("t".repeat(60));
        assert_eq!(
            qmp_socket_in(&dir, &"n".repeat(64), &fits, short).parent(),
            Some(fits.as_path())
        );
        assert!(
            paths::short_runtime_dir()
                .join("0123456789abcdef.qmp")
                .as_os_str()
                .len()
                <= 46
        );
    }

    fn x86_64_config() -> VmConfig {
        VmConfig {
            name: "t".into(),
            profile: Profile::Q35X86_64,
            cpus: 4,
            memory_mib: 8192,
            disk: "disk.qcow2".into(),
            disk_format: ImageFormat::Qcow2,
            stop_timeout_secs: 90,
            extra_args: vec!["-boot".into(), "menu=on".into()],
            graphics: None,
            display: Display::default(),
            gpu: Gpu {
                driver_file: Some("/usr/share/vulkan/icd.d/nvidia_icd.json".into()),
                device_id: Some(0x2204),
                name: None,
            },
            audio: Audio::default(),
            network: Network {
                enabled: true,
                mac: "52:54:00:01:02:03".into(),
                ssh_port: Some(2222),
            },
            logs: Logs::default(),
            q35: Some(Q35 {
                opencore: "opencore.qcow2".into(),
                opencore_format: ImageFormat::Qcow2,
                firmware_code: "OVMF_CODE.fd".into(),
                firmware_vars: "OVMF_VARS.fd".into(),
                installer: Some("/isos/BaseSystem.img".into()),
                cpu_model: default_cpu_model(),
                cpu_flags: default_cpu_flags(),
                disk_controller: DiskController::Nvme,
                disk_port: 3,
            }),
            extra_env: BTreeMap::from([("A".into(), "b".into())]),
            usb: vec![],
        }
    }

    /// The USB devices a VM is given are kept as `[[usb]]` tables, checked as QEMU takes them:
    /// an x86-64 VM's, at most four, each once, by IDs and a port that name a device.
    #[test]
    fn usb_devices_are_checked_as_qemu_takes_them() {
        let mut c = x86_64_config();
        c.usb = vec![UsbDevice {
            id: "1050:0407".into(),
            port: Some("3-2.1".into()),
            name: "Yubico YubiKey".into(),
        }];
        assert!(c.check().is_ok());
        let text = toml::to_string_pretty(&c).unwrap();
        assert_eq!(toml::from_str::<VmConfig>(&text).unwrap(), c);
        assert!(
            text.ends_with(
                "\n[[usb]]\nid = \"1050:0407\"\nport = \"3-2.1\"\nname = \"Yubico YubiKey\"\n"
            ),
            "{text}"
        );
        let u = |id: &str, port: Option<&str>| UsbDevice {
            id: id.into(),
            port: port.map(String::from),
            name: String::new(),
        };
        // Saved without a port or a name, it reads back so.
        c.usb = vec![u("0403:6001", None)];
        let text = toml::to_string_pretty(&c).unwrap();
        assert!(text.ends_with("\n[[usb]]\nid = \"0403:6001\"\n"), "{text}");
        assert_eq!(toml::from_str::<VmConfig>(&text).unwrap(), c);
        let refused = |c: &VmConfig| format!("{:#}", c.check().unwrap_err());
        c.usb = (1..=5).map(|i| u(&format!("1050:000{i}"), None)).collect();
        assert!(
            refused(&c).contains("4 USB devices at most"),
            "{}",
            refused(&c)
        );
        c.usb.pop();
        assert!(c.check().is_ok());
        c.usb = vec![u("1050:0407", None), u("1050:0407", None)];
        assert!(refused(&c).contains("chosen twice"), "{}", refused(&c));
        // Two alike, each in its port, are two devices.
        c.usb = vec![u("0403:6001", Some("1-4")), u("0403:6001", Some("2-1.2"))];
        assert!(c.check().is_ok());
        for bad in [
            "1050:407",
            "1050-0407",
            "10500407",
            "1050:04g7",
            ":0407",
            "1050:0407 ",
        ] {
            c.usb = vec![u(bad, None)];
            assert!(refused(&c).contains("such as 1050:0407"), "{bad:?}");
        }
        for bad in [
            "3",
            "3-",
            "0-1",
            "3-0",
            "3-2.1,x=y",
            "3-2.1.1.1.1.1.1.1",
            "3-02",
            "a-1",
        ] {
            c.usb = vec![u("1050:0407", Some(bad))];
            assert!(refused(&c).contains("names no USB port"), "{bad:?}");
        }
        assert_eq!(parse_usb_port("3-2.1"), Some((3, "2.1")));
        assert_eq!(parse_usb_port("1-10"), Some((1, "10")));
    }

    #[test]
    fn config_round_trips_through_toml() {
        let mut c = x86_64_config();
        let text = toml::to_string_pretty(&c).unwrap();
        assert_eq!(toml::from_str::<VmConfig>(&text).unwrap(), c);
        for g in Graphics::ALL {
            c.graphics = Some(g);
            let text = toml::to_string_pretty(&c).unwrap();
            assert!(
                text.contains(&format!("graphics = \"{}\"\n", g.id())),
                "{text}"
            );
            assert_eq!(toml::from_str::<VmConfig>(&text).unwrap(), c);
        }
    }

    /// A VM whose settings name no graphics runs reims-vgpu's own device. Settings an older
    /// reims-ctl saved for apple-gfx read as reims-vgpu, and save naming it; the names parse
    /// back to their devices.
    #[test]
    fn a_vm_naming_no_graphics_or_apple_gfx_runs_reims_vgpu() {
        let mut c = x86_64_config();
        let text = toml::to_string_pretty(&c).unwrap();
        assert!(!text.contains("graphics"), "{text}");
        let loaded = toml::from_str::<VmConfig>(&text).unwrap();
        assert_eq!(loaded.graphics(), Graphics::ReimsVgpu);
        assert_eq!(toml::to_string_pretty(&loaded).unwrap(), text);
        c.graphics = Some(Graphics::ReimsVgpu);
        let ours = toml::to_string_pretty(&c).unwrap();
        let apple = ours.replace("graphics = \"reims-vgpu\"", "graphics = \"apple-gfx\"");
        assert_ne!(apple, ours);
        let loaded = toml::from_str::<VmConfig>(&apple).unwrap();
        assert_eq!(loaded.graphics, Some(Graphics::ReimsVgpu));
        assert_eq!(toml::to_string_pretty(&loaded).unwrap(), ours);
        for g in Graphics::ALL {
            assert_eq!(Graphics::parse(g.id()).unwrap(), g);
        }
        for gone in ["apple-gfx", "vga"] {
            assert!(Graphics::parse(gone).is_err(), "{gone}");
        }
    }

    /// A VM whose settings name no controller has its disk on SATA, and one on SATA saves
    /// naming none.
    #[test]
    fn a_vm_naming_no_disk_controller_has_its_disk_on_sata() {
        let mut c = x86_64_config();
        let text = toml::to_string_pretty(&c).unwrap();
        assert!(text.contains("disk_controller = \"nvme\"\n"), "{text}");
        let sata = text.replace("disk_controller = \"nvme\"\n", "");
        let q = toml::from_str::<VmConfig>(&sata).unwrap().q35.unwrap();
        assert_eq!(q.disk_controller, DiskController::Ahci);
        c.q35.as_mut().unwrap().disk_controller = DiskController::Ahci;
        assert_eq!(toml::to_string_pretty(&c).unwrap(), sata);
        // The settings name each controller by the id `set --disk-controller` takes.
        for d in DiskController::ALL {
            assert_eq!(DiskController::parse(d.id()).unwrap(), d);
            let named = text.replace("\"nvme\"", &format!("\"{}\"", d.id()));
            let q = toml::from_str::<VmConfig>(&named).unwrap().q35.unwrap();
            assert_eq!(q.disk_controller, d);
        }
    }

    /// What an older reims-ctl saved for apple-gfx alone (video acceleration, the displays'
    /// count, HiDPI and refresh rate) is read past, and left out of the next save.
    #[test]
    fn apple_gfx_s_settings_are_read_past() {
        let c = x86_64_config();
        let text = toml::to_string_pretty(&c).unwrap();
        let old = text.replace(
            "[display]\n",
            "video_acceleration = false\n\n[display]\ncount = 2\nhidpi = true\nrefresh = 120\n",
        );
        assert_ne!(old, text);
        let loaded = toml::from_str::<VmConfig>(&old).unwrap();
        assert_eq!(loaded, c);
        assert_eq!(toml::to_string_pretty(&loaded).unwrap(), text);
    }

    /// A VM an earlier create saved with an SSH port of 0 has no forward, as `set --ssh-port 0`
    /// gives.
    #[test]
    fn a_saved_ssh_port_of_0_is_no_forward() {
        let mut c = x86_64_config();
        let text = toml::to_string_pretty(&c).unwrap();
        assert!(text.contains("ssh_port = 2222"), "{text}");
        let zero = text.replace("ssh_port = 2222", "ssh_port = 0");
        assert_eq!(
            toml::from_str::<VmConfig>(&zero).unwrap().network.ssh_port,
            None
        );
        c.network.ssh_port = None;
        let none = toml::to_string_pretty(&c).unwrap();
        assert_eq!(toml::from_str::<VmConfig>(&none).unwrap(), c);
    }

    /// A VM is an x86-64 one, with its [q35] settings; VMware SVGA is one of its devices.
    #[test]
    fn a_vm_has_its_q35_settings() {
        let mut c = x86_64_config();
        c.graphics = Some(Graphics::parse("vmware-svga").unwrap());
        assert!(c.check().is_ok());
        c.q35 = None;
        let e = format!("{:#}", c.check().unwrap_err());
        assert!(e.contains("no [q35] settings"), "{e}");
    }

    #[test]
    fn incompatibilities_are_an_x86_64_vms_silence() {
        let mut c = x86_64_config();
        assert!(c.incompatibilities().is_empty());
        // A wide window is none.
        c.display.resolution = "2560x1600".into();
        assert!(c.incompatibilities().is_empty());
        c.audio.enabled = false;
        let w = c.incompatibilities();
        assert_eq!(w.len(), 1);
        assert!(
            w[0].contains("reims-ctl set t --audio-backend none"),
            "{w:?}"
        );
    }
}
