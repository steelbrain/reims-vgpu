//! An x86-64 VM's OpenCore image: the Mac it presents, and making the VM a new one.
//!
//! reims-util makes a VM's image when the VM is made (reims_util.rs). A VM on reims-vgpu's own
//! GPU device needs the guest's kexts this reims-ctl carries ([`crate::prebuilt`]), which an
//! image of the user's own may lack ([`missing_guest_kexts`]), and one an older reims-ctl made
//! holds other builds of, until the VM's next start makes it again
//! ([`holds_other_guest_kexts`]). [`renew`] replaces a VM's image with one the
//! installed reims-util makes for the VM's graphics, presenting the same Mac, so macOS sees no
//! new machine. reims-ctl keeps no copy of that identity: it is read from the old image's
//! config.plist.
//!
//! An image is read as reims-util writes it: qcow2 (or raw), a GPT, and a FAT32 EFI system
//! partition holding EFI/OC/config.plist.

use crate::os::FileExt;
use crate::prebuilt::Kext;
use crate::vm::{Graphics, ImageFormat, Vm};
use crate::{paths, prebuilt, reims_util, runtime};
use anyhow::{Context, Result, bail, ensure};
use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError};
use std::time::{Duration, SystemTime};

/// Lilu, the old graphics loader and its two shims, which the guest's set loaded through before
/// Graft: an image reims-ctl makes lists none of them (migrate.rs, Migration 4).
pub(crate) const LEGACY_KEXTS: [&str; 4] = [
    "Lilu.kext",
    "ReimsGraphicsLoader.kext",
    "RgfxIOGraphicsFamilyBridge.kext",
    "RgfxAppleGraphicsDeviceControlBridge.kext",
];

/// The Mac an OpenCore image presents: its config.plist's PlatformInfo > Generic, which
/// reims-util takes as `--model`, `--serial`, `--mlb`, `--uuid` and `--rom`.
#[derive(Clone, Debug, PartialEq)]
pub struct Identity {
    pub model: String,
    pub serial: String,
    pub mlb: String,
    pub uuid: String,
    /// Six bytes, as 12 hex digits.
    pub rom: String,
}

/// An OpenCore image's config.plist.
pub struct Config(plist::Dictionary);

impl Config {
    pub fn read(path: &Path, format: ImageFormat) -> Result<Config> {
        let bytes = read_file(path, format, "EFI/OC/config.plist")?
            .context("its EFI system partition has no EFI/OC/config.plist")?;
        Config::parse(&bytes)
    }

    fn parse(bytes: &[u8]) -> Result<Config> {
        match plist::Value::from_reader(std::io::Cursor::new(bytes)).context("read config.plist")? {
            plist::Value::Dictionary(d) => Ok(Config(d)),
            _ => bail!("config.plist holds no dictionary"),
        }
    }

    pub fn identity(&self) -> Result<Identity> {
        let generic = self
            .0
            .get("PlatformInfo")
            .and_then(plist::Value::as_dictionary)
            .and_then(|p| p.get("Generic"))
            .and_then(plist::Value::as_dictionary)
            .context("config.plist has no PlatformInfo > Generic")?;
        let text = |key: &str| {
            generic
                .get(key)
                .and_then(plist::Value::as_string)
                .filter(|s| !s.trim().is_empty())
                .map(String::from)
                .with_context(|| format!("config.plist's PlatformInfo > Generic has no {key}"))
        };
        let rom = generic
            .get("ROM")
            .and_then(plist::Value::as_data)
            .filter(|d| d.len() == 6)
            .context("config.plist's PlatformInfo > Generic has no six-byte ROM")?;
        Ok(Identity {
            model: text("SystemProductName")?,
            serial: text("SystemSerialNumber")?,
            mlb: text("MLB")?,
            uuid: text("SystemUUID")?,
            rom: rom.iter().map(|b| format!("{b:02X}")).collect(),
        })
    }

    fn kext_entries(&self) -> impl Iterator<Item = &plist::Dictionary> {
        self.0
            .get("Kernel")
            .and_then(plist::Value::as_dictionary)
            .and_then(|k| k.get("Add"))
            .and_then(plist::Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(plist::Value::as_dictionary)
    }

    /// Every bundle listed in Kernel > Add, including disabled entries.
    pub fn configured_kexts(&self) -> Vec<&str> {
        self.kext_entries()
            .filter_map(|e| e.get("BundlePath").and_then(plist::Value::as_string))
            .collect()
    }

    /// The kexts OpenCore loads, in its order: the bundle each enabled entry of Kernel > Add
    /// names (its BundlePath, under EFI/OC/Kexts).
    pub fn loaded_kexts(&self) -> Vec<&str> {
        self.kext_entries()
            .filter(|e| e.get("Enabled").and_then(plist::Value::as_boolean) == Some(true))
            .filter_map(|e| e.get("BundlePath").and_then(plist::Value::as_string))
            .collect()
    }

    /// Whether OpenCore shows its boot picker at each start: Misc > Boot > ShowPicker. Hidden,
    /// it boots the default entry, macOS's startup disk.
    pub fn shows_picker(&self) -> bool {
        self.0
            .get("Misc")
            .and_then(plist::Value::as_dictionary)
            .and_then(|m| m.get("Boot"))
            .and_then(plist::Value::as_dictionary)
            .and_then(|b| b.get("ShowPicker"))
            .and_then(plist::Value::as_boolean)
            == Some(true)
    }
}

/// Whether a VM's OpenCore image shows its boot picker; `None` for an image reims-ctl cannot
/// read, or a VM with no OpenCore.
pub fn shows_picker(vm: &Vm) -> Option<bool> {
    let q = vm.config.q35.as_ref()?;
    Config::read(&vm.path(&q.opencore), q.opencore_format)
        .ok()
        .map(|c| c.shows_picker())
}

/// How an image holds the guest's kexts for reims-vgpu's own device.
#[derive(Clone, Debug, PartialEq)]
enum GuestKexts {
    /// Each, in the set's order, with the executable reims-ctl carries.
    Held,
    /// Not these, which it does not load.
    Lacks(Vec<&'static str>),
    /// Each, but these are other builds than reims-ctl's, or out of the set's order.
    Other(Vec<&'static str>),
}

/// How an image whose OpenCore loads `loaded`, in its order, holds `set`; `same` tells whether
/// it holds a kext's executable as reims-ctl carries it, which is asked where the kext has one.
fn guest_kexts(
    set: &'static [Kext],
    loaded: &[&str],
    same: &mut dyn FnMut(&Kext) -> bool,
) -> GuestKexts {
    let at = |k: &Kext| loaded.iter().position(|b| b.eq_ignore_ascii_case(k.bundle));
    let lacks: Vec<_> = set
        .iter()
        .filter(|k| at(k).is_none())
        .map(|k| k.bundle)
        .collect();
    if !lacks.is_empty() {
        return GuestKexts::Lacks(lacks);
    }
    let mut last = None;
    let other: Vec<_> = set
        .iter()
        .filter(|k| {
            let i = at(k);
            let in_order = last < i;
            last = i;
            let other_build = k.executable.is_some() && !same(k);
            !in_order || other_build
        })
        .map(|k| k.bundle)
        .collect();
    if other.is_empty() {
        GuestKexts::Held
    } else {
        GuestKexts::Other(other)
    }
}

/// How a VM's OpenCore image holds the guest's kexts this reims-ctl carries; None where the
/// image cannot be read, an error where reims-ctl carries no set. What a read of the image finds
/// is kept while its file stays as it was ([`KEPT`]): the interface's VM pane asks at each draw,
/// a few times a second, and a read reads the image's config.plist and the kexts' executables,
/// megabytes.
fn image_guest_kexts(vm: &Vm) -> Option<Result<GuestKexts>> {
    let q = vm.config.q35.as_ref()?;
    let image = vm.path(&q.opencore);
    let format = q.opencore_format;
    let now = SystemTime::now();
    let stamp = Stamp::of(&image);
    {
        let mut kept = KEPT.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(k) = kept.get(&image)
            && Some(k.stamp) == stamp
            && k.format == format
        {
            return Some(Ok(k.kexts.clone()));
        }
        kept.remove(&image);
    }
    let read = guest_kexts_in(&image, format, prebuilt::set);
    if let (Some(Ok(kexts)), Some(stamp)) = (&read, stamp)
        && stamp.settled(now)
    {
        let kexts = kexts.clone();
        let mut kept = KEPT.lock().unwrap_or_else(PoisonError::into_inner);
        kept.insert(
            image,
            Kept {
                format,
                stamp,
                kexts,
            },
        );
    }
    read
}

/// What a read of an image found of the guest's kexts, and its file as the read found it.
struct Kept {
    format: ImageFormat,
    stamp: Stamp,
    kexts: GuestKexts,
}

/// What reads of images found, by the image's path ([`image_guest_kexts`]). A read's answer is
/// kept only where its file was last written more than [`SETTLED`] before the read began, and
/// given again only while the file has the stamp it had then ([`Stamp`]): a write after the read
/// gives the file a later time of last write. Any other stamp drops the answer. A new image
/// reims-ctl makes ([`renew`]) drops its path's answer whatever its stamp.
static KEPT: Mutex<BTreeMap<PathBuf, Kept>> = Mutex::new(BTreeMap::new());

/// How long before a read of an image its file must have been last written for the read's answer
/// to be kept ([`KEPT`]). FAT keeps a file's time of last write to two seconds, coarser than the
/// other file systems a host may keep a VM on, and a kernel may take that time from a clock some
/// milliseconds behind: a second more covers it, so a write after the read cannot leave the file
/// the time it had.
const SETTLED: Duration = Duration::from_secs(3);

/// An image's file as a read of it found it: its length, when it was last written, and on macOS
/// and Linux when its inode last changed ([`crate::os::changed`]), which a program that sets the
/// file's time of last write back, as `cp -p` and `rsync -t` do, moves to the present.
#[derive(Clone, Copy, PartialEq)]
struct Stamp {
    len: u64,
    modified: SystemTime,
    changed: Option<SystemTime>,
}

impl Stamp {
    fn of(path: &Path) -> Option<Stamp> {
        let m = std::fs::metadata(path).ok()?;
        Some(Stamp {
            len: m.len(),
            modified: m.modified().ok()?,
            changed: crate::os::changed(&m),
        })
    }

    /// Whether the file was last written more than [`SETTLED`] before `now`.
    fn settled(&self, now: SystemTime) -> bool {
        let since = now.duration_since(self.modified);
        since.is_ok_and(|d| d > SETTLED)
    }
}

/// How the image at `image` holds the guest's kexts `set` gives; None where the image cannot be
/// read, an error where `set` gives one. Each executable is compared whole, however large
/// ([`Fat32::holds`]).
fn guest_kexts_in(
    image: &Path,
    format: ImageFormat,
    set: impl FnOnce() -> Result<&'static [Kext]>,
) -> Option<Result<GuestKexts>> {
    let disk = Disk::open(image, format).ok()?;
    let fat = Fat32::open(&disk, esp_start(&disk).ok()?).ok()?;
    let config = fat.read_file("EFI/OC/config.plist").ok()??;
    let config = Config::parse(&config).ok()?;
    let set = match set() {
        Ok(s) => s,
        Err(e) => return Some(Err(e)),
    };
    let mut same = |k: &Kext| {
        let (Some(exe), Some(carried)) = (k.executable, k.executable_bytes()) else {
            return false;
        };
        let file = format!("EFI/OC/Kexts/{}/{exe}", k.bundle);
        matches!(fat.holds(&file, carried), Ok(Some(true)))
    };
    Some(Ok(guest_kexts(set, &config.loaded_kexts(), &mut same)))
}

/// What to tell of a VM on reims-vgpu's own GPU device whose OpenCore image lacks the guest's
/// kexts this reims-ctl carries, as an image of the user's own may, which no start makes again;
/// None where it holds them, or holds other builds of them, as the image of a VM an older
/// reims-ctl made does after an update, which the VM's next start makes again with these
/// ([`holds_other_guest_kexts`]); None also where the VM is on another device, or the image
/// cannot be read.
pub fn missing_guest_kexts(vm: &Vm) -> Option<String> {
    if vm.config.graphics() != Graphics::ReimsVgpu {
        return None;
    }
    match image_guest_kexts(vm)? {
        Err(e) => Some(format!("{e:#}")),
        Ok(GuestKexts::Held | GuestKexts::Other(_)) => None,
        Ok(GuestKexts::Lacks(b)) => Some(format!(
            "Its OpenCore image lacks reims-vgpu's macOS drivers ({}), without which macOS has \
             no driver for its GPU device. To make it a new image that has this reims-ctl's, \
             presenting the same Mac, press U in reims-ctl or run reims-ctl update-opencore {}.",
            b.join(", "),
            vm.name()
        )),
    }
}

/// Whether a VM on reims-vgpu's own GPU device has an OpenCore image that does not hold the
/// guest's kexts this reims-ctl carries as it carries them, lacking them or holding other builds,
/// or reims-ctl carries none; false where the image cannot be read.
fn guest_kexts_differ(vm: &Vm) -> bool {
    vm.config.graphics() == Graphics::ReimsVgpu
        && matches!(
            image_guest_kexts(vm),
            Some(Err(_) | Ok(GuestKexts::Lacks(_) | GuestKexts::Other(_)))
        )
}

/// Whether a VM on reims-vgpu's own device has an OpenCore image that holds the guest's kexts this
/// reims-ctl carries, each in the set's order and of its build.
pub fn holds_guest_kexts(vm: &Vm) -> bool {
    vm.config.graphics() == Graphics::ReimsVgpu
        && matches!(image_guest_kexts(vm), Some(Ok(GuestKexts::Held)))
}

/// Whether a VM on reims-vgpu's own GPU device has an OpenCore image that loads each of the
/// guest's kexts, but other builds of some than this reims-ctl carries, or out of the set's
/// order: an image an older reims-ctl made, which a start makes again ([`crate::runtime::prepare`])
/// so the guest runs the kexts that came with this reims-ctl.
pub fn holds_other_guest_kexts(vm: &Vm) -> bool {
    vm.config.graphics() == Graphics::ReimsVgpu
        && matches!(image_guest_kexts(vm), Some(Ok(GuestKexts::Other(_))))
}

/// Whether a VM's OpenCore image has OpenCore load any of the guest's kexts for reims-vgpu's own
/// device; None where the image can't be read, or reims-ctl carries no set.
fn loads_guest_kexts(vm: &Vm) -> Option<bool> {
    let q = vm.config.q35.as_ref()?;
    let config = Config::read(&vm.path(&q.opencore), q.opencore_format).ok()?;
    let loaded = config.loaded_kexts();
    let set = prebuilt::set().ok()?;
    Some(
        set.iter()
            .any(|k| loaded.iter().any(|b| b.eq_ignore_ascii_case(k.bundle))),
    )
}

/// Whether a VM on another device than reims-vgpu's own has an OpenCore image that loads the
/// guest's kexts for it, as one made for reims-vgpu does. On a macOS they load on, OpenCore loads
/// them with no device of theirs there (measured on apple-gfx with a macOS 26.7.2 guest:
/// ReimsGraphicsLoader and RgfxGate loaded), and RgfxGate installs its Metal bundle in
/// /Library/GPUBundles and sets AppleGVA's gvaForceAMDKE at each boot (guest/rgfxgate).
pub fn holds_unused_guest_kexts(vm: &Vm) -> bool {
    vm.config.graphics() != Graphics::ReimsVgpu && loads_guest_kexts(vm) == Some(true)
}

/// Whether a VM's OpenCore image is made for other graphics than the VM runs: on reims-vgpu,
/// lacking the guest's kexts or holding other builds ([`guest_kexts_differ`]); on another
/// device, loading them ([`holds_unused_guest_kexts`]).
pub fn made_for_other_graphics(vm: &Vm) -> bool {
    match vm.config.graphics() {
        Graphics::ReimsVgpu => guest_kexts_differ(vm),
        _ => holds_unused_guest_kexts(vm),
    }
}

/// Replaces a stopped VM's OpenCore image with one the installed reims-util makes, presenting
/// the same Mac, with its boot picker shown where `show_picker` says, or as the old image shows
/// it when `None`. The new image is made beside the old one, which stays as it was if that fails.
/// It holds the kexts for the VM's graphics ([`reims_util::kext_args`]): on reims-vgpu's own
/// device, the guest's that this reims-ctl carries. The VM keeps its OVMF: its variables hold
/// its boot entries. The image is set in the VM's settings as they are once it is made, which
/// `vm` becomes: others may be saved meanwhile. Returns what happened, and, where the old image
/// loaded reims-vgpu's kexts on other graphics, what they left in the guest
/// ([`crate::macos::gva_note`]).
pub fn renew(
    vm: &mut Vm,
    show_picker: Option<bool>,
    progress: &mut dyn FnMut(&str),
) -> Result<String> {
    let name = vm.name().to_string();
    if runtime::running(vm) {
        bail!("{name} is running: stop it first");
    }
    let q = vm
        .config
        .q35
        .clone()
        .context("only x86-64 VMs boot OpenCore")?;
    let old = vm.path(&q.opencore);
    let read = || format!("read the Mac {} presents", paths::tilde(&old));
    let config = Config::read(&old, q.opencore_format).with_context(read)?;
    let id = config.identity().with_context(read)?;
    let picker = show_picker.unwrap_or_else(|| config.shows_picker());
    let image = vm.dir.join("opencore.qcow2");
    let new = vm.dir.join("opencore.new.qcow2");
    let dropped = holds_unused_guest_kexts(vm);
    let kexts = reims_util::kext_args(vm.config.graphics())?;
    let made = reims_util::opencore_as(&new, &id, picker, &kexts, progress).and_then(|()| {
        let config = Config::read(&new, ImageFormat::Qcow2)
            .context("read the new OpenCore image before replacing the old one")?;
        ensure!(
            !config
                .configured_kexts()
                .iter()
                .any(|k| LEGACY_KEXTS.iter().any(|old| k.eq_ignore_ascii_case(old))),
            "the installed reims-util still includes obsolete guest kexts: run reims-ctl update, \
             which installs the reims-util it needs"
        );
        // reims-util hides the picker unless --show-picker asks for it; an older one shows it.
        ensure!(
            config.shows_picker() == picker,
            "the installed reims-util did not {} OpenCore's picker: run reims-ctl update, which \
             installs the reims-util it needs",
            if picker { "show" } else { "hide" }
        );
        Ok(())
    });
    if let Err(e) = made {
        let _ = std::fs::remove_file(&new);
        return Err(e);
    }
    // The file the old image's path names, where it is in the VM's directory and not the image
    // made: compared with links, '.' and '..' resolved, so a path given as "../x" or
    // "sub/../opencore.qcow2" is neither taken for one in the directory nor for another file.
    let replaced = std::fs::canonicalize(&vm.dir)
        .ok()
        .and_then(|dir| Some((entry(&old)?, dir)))
        .filter(|(e, dir)| e.starts_with(dir) && *e != dir.join("opencore.qcow2"))
        .map(|(e, _)| e);
    std::fs::rename(&new, &image).with_context(|| format!("replace {}", image.display()))?;
    // The new image is read at the next ask, whatever its stamp: a file system may keep it with
    // the old one's length and time of last write, as one keeping times to two seconds may.
    KEPT.lock()
        .unwrap_or_else(PoisonError::into_inner)
        .remove(&image);
    *vm = Vm::load(&vm.dir)?;
    if let Some(q) = vm.config.q35.as_mut() {
        q.opencore = "opencore.qcow2".into();
        q.opencore_format = ImageFormat::Qcow2;
    }
    vm.save()?;
    // An image the VM was given under another name, in its directory, is replaced too.
    if let Some(e) = replaced {
        let _ = std::fs::remove_file(e);
    }
    let mut s = format!(
        "{name}: a new OpenCore image, presenting the same {} (serial {}), its boot selection {}",
        id.model,
        id.serial,
        if picker { "on" } else { "off" }
    );
    if guest_kexts_differ(vm) {
        s.push_str(
            "; the installed reims-util did not put reims-vgpu's macOS drivers in it as given: \
             run reims-ctl update, which installs the reims-util it needs, and make it again",
        );
    }
    // What the kexts left in the guest stays there.
    if dropped && let Some(n) = crate::macos::gva_note(vm) {
        s.push_str("; ");
        s.push_str(&n);
    }
    Ok(s)
}

/// The directory entry `p` names, its directory's path resolved (links, '.' and '..'): the file
/// a removal of `p` unlinks, a link itself where `p` names one. None where `p` ends in '..', or
/// its directory does not resolve.
fn entry(p: &Path) -> Option<std::path::PathBuf> {
    let name = p.file_name()?;
    Some(std::fs::canonicalize(p.parent()?).ok()?.join(name))
}

/// The file at `path` (as [`Fat32::read_file`] takes it) on the EFI system partition of the
/// image at `image`; None where there is none.
fn read_file(image: &Path, format: ImageFormat, path: &str) -> Result<Option<Vec<u8>>> {
    let disk = Disk::open(image, format)?;
    Fat32::open(&disk, esp_start(&disk)?)?.read_file(path)
}

const SECTOR: u64 = 512;

/// The EFI system partition's type GUID, C12A7328-F81F-11D2-BA4B-00A0C93EC93B, as GPT stores
/// it.
const ESP_TYPE: [u8; 16] = [
    0x28, 0x73, 0x2a, 0xc1, 0x1f, 0xf8, 0xd2, 0x11, 0xba, 0x4b, 0x00, 0xa0, 0xc9, 0x3e, 0xc9, 0x3b,
];

/// Where the GPT's EFI system partition starts, in bytes.
fn esp_start(disk: &Disk) -> Result<u64> {
    let mut header = [0u8; SECTOR as usize];
    disk.read(SECTOR, &mut header)?;
    ensure!(&header[..8] == b"EFI PART", "it has no GPT");
    partitions(disk, &ESP_TYPE)?
        .into_iter()
        .next()
        .context("its GPT has no EFI system partition")
}

/// Where each partition of type `kind` (its GUID, as GPT stores it) starts, in bytes, in the
/// GPT's order; none on a disk with no GPT.
pub(crate) fn partitions(disk: &Disk, kind: &[u8; 16]) -> Result<Vec<u64>> {
    let mut header = [0u8; SECTOR as usize];
    disk.read(SECTOR, &mut header)?;
    if &header[..8] != b"EFI PART" {
        return Ok(Vec::new());
    }
    let size = u64::from(le32(&header, 84));
    ensure!((128..=4096).contains(&size), "GPT entries of {size} bytes");
    let count = u64::from(le32(&header, 80).min(128));
    let mut entries = vec![0u8; (count * size) as usize];
    disk.read(le64(&header, 72) * SECTOR, &mut entries)?;
    Ok(entries
        .chunks_exact(size as usize)
        .filter(|e| e[..16] == *kind)
        .map(|e| le64(e, 32) * SECTOR)
        .collect())
}

/// An image's bytes as the guest sees them.
pub(crate) enum Disk {
    Raw { file: File, size: u64 },
    Qcow2(Qcow2),
}

impl Disk {
    pub(crate) fn open(path: &Path, format: ImageFormat) -> Result<Disk> {
        let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        Ok(match format {
            ImageFormat::Raw => Disk::Raw {
                size: file.metadata()?.len(),
                file,
            },
            ImageFormat::Qcow2 => Disk::Qcow2(Qcow2::open(file)?),
        })
    }

    pub(crate) fn read(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        #[cfg(test)]
        {
            let (reads, bytes) = tests::READS.get();
            tests::READS.set((reads + 1, bytes + buf.len() as u64));
        }
        let size = match self {
            Disk::Raw { size, .. } => *size,
            Disk::Qcow2(q) => q.size,
        };
        ensure!(
            offset.saturating_add(buf.len() as u64) <= size,
            "a read at {offset:#x}, past the image's end"
        );
        match self {
            Disk::Raw { file, .. } => file.read_exact_at(buf, offset)?,
            Disk::Qcow2(q) => q.read(offset, buf)?,
        }
        Ok(())
    }
}

/// Bits 9 to 55 of a qcow2 L1 or L2 entry: where the L2 table or the cluster is in the file.
const QCOW2_OFFSET: u64 = 0x00ff_ffff_ffff_fe00;

/// A qcow2 image (QEMU's docs/interop/qcow2.txt) with no backing file: each guest cluster
/// found through the L1 and L2 tables, unallocated and zero clusters reading as zeros.
pub(crate) struct Qcow2 {
    file: File,
    cluster_bits: u32,
    size: u64,
    l1: Vec<u64>,
}

impl Qcow2 {
    fn open(file: File) -> Result<Qcow2> {
        let mut h = [0u8; 48];
        file.read_exact_at(&mut h, 0)?;
        ensure!(&h[..4] == b"QFI\xfb", "not a qcow2 image");
        ensure!(be64(&h, 8) == 0, "a qcow2 image with a backing file");
        // Version 3's incompatible features: an external data file (bit 2) or extended L2
        // entries (bit 4) lay the clusters out otherwise.
        if be32(&h, 4) >= 3 {
            let mut f = [0u8; 8];
            file.read_exact_at(&mut f, 72)?;
            ensure!(
                u64::from_be_bytes(f) & (1 << 2 | 1 << 4) == 0,
                "a qcow2 image with an external data file or extended L2 entries"
            );
        }
        let cluster_bits = be32(&h, 20);
        ensure!(
            (9..=21).contains(&cluster_bits),
            "qcow2 clusters of 2^{cluster_bits} bytes"
        );
        let mut l1 = vec![0u8; be32(&h, 36).min(1 << 20) as usize * 8];
        file.read_exact_at(&mut l1, be64(&h, 40))?;
        Ok(Qcow2 {
            file,
            cluster_bits,
            size: be64(&h, 24),
            l1: l1.as_chunks::<8>().0.iter().map(|e| be64(e, 0)).collect(),
        })
    }

    /// Where guest cluster `index` is in the file; None where it reads as zeros.
    fn cluster(&self, index: u64) -> Result<Option<u64>> {
        let per_table = 1u64 << (self.cluster_bits - 3);
        let l1 = self
            .l1
            .get((index / per_table) as usize)
            .copied()
            .unwrap_or(0);
        if l1 & QCOW2_OFFSET == 0 {
            return Ok(None);
        }
        let mut e = [0u8; 8];
        self.file
            .read_exact_at(&mut e, (l1 & QCOW2_OFFSET) + index % per_table * 8)?;
        let e = u64::from_be_bytes(e);
        ensure!(e & 1 << 62 == 0, "a qcow2 image with compressed clusters");
        // Bit 0: a zero cluster.
        Ok((e & 1 == 0 && e & QCOW2_OFFSET != 0).then_some(e & QCOW2_OFFSET))
    }

    fn read(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        let cluster = 1u64 << self.cluster_bits;
        let mut done = 0;
        while done < buf.len() {
            let at = offset + done as u64;
            let within = at & (cluster - 1);
            let n = ((cluster - within) as usize).min(buf.len() - done);
            let part = &mut buf[done..done + n];
            match self.cluster(at >> self.cluster_bits)? {
                Some(c) => self.file.read_exact_at(part, c + within)?,
                None => part.fill(0),
            }
            done += n;
        }
        Ok(())
    }
}

/// The most of a directory, or of a file read whole, reims-ctl reads. The volume is the guest's to
/// write, and a corrupt or hostile one may give a directory a chain that loops, which only the
/// FAT's end mark would end, as a directory's entry gives no size, or give a file's entry a size
/// of up to 4 GiB. The one file read whole is config.plist, of tens of KiB; the guest's kexts are
/// compared as they are read ([`Fat32::holds`]), and none of this bounds them.
const MOST: u64 = 16 << 20;

/// A FAT32 volume (Microsoft's FAT specification, 1.03).
struct Fat32<'a> {
    disk: &'a Disk,
    /// Where its first FAT and its cluster 2 are in the image.
    fat: u64,
    data: u64,
    cluster_size: u64,
    root: u32,
}

impl<'a> Fat32<'a> {
    fn open(disk: &'a Disk, start: u64) -> Result<Fat32<'a>> {
        let mut b = [0u8; SECTOR as usize];
        disk.read(start, &mut b)?;
        let sector = u64::from(le16(&b, 11));
        let per_cluster = u64::from(b[13]);
        let reserved = u64::from(le16(&b, 14));
        let fat_sectors = u64::from(le32(&b, 36));
        ensure!(
            b[510..] == [0x55, 0xaa] && le16(&b, 22) == 0 && fat_sectors > 0,
            "its EFI system partition is not FAT32"
        );
        ensure!(
            sector >= 512 && per_cluster > 0,
            "a FAT32 volume of no clusters"
        );
        Ok(Fat32 {
            disk,
            fat: start + reserved * sector,
            data: start + (reserved + u64::from(b[16]) * fat_sectors) * sector,
            cluster_size: per_cluster * sector,
            root: le32(&b, 44),
        })
    }

    /// The entry at `path`, `/` between its names, matched without regard to case as FAT
    /// matches them; None where there is none.
    fn find(&self, path: &str) -> Result<Option<Entry>> {
        let mut dir = self.root;
        let mut names = path.split('/').peekable();
        while let Some(name) = names.next() {
            let bytes = self.read_chain(dir, MOST)?;
            let Some(e) = entries(&bytes).into_iter().find(|e| e.is(name)) else {
                return Ok(None);
            };
            if names.peek().is_none() {
                return Ok(Some(e));
            }
            dir = e.cluster;
        }
        Ok(None)
    }

    /// The file at `path` (as [`Fat32::find`] takes it), read whole; None where there is none.
    fn read_file(&self, path: &str) -> Result<Option<Vec<u8>>> {
        let Some(e) = self.find(path)? else {
            return Ok(None);
        };
        let size = u64::from(e.size);
        ensure!(
            size <= MOST,
            "{path} is of {size} bytes, more than reims-ctl reads"
        );
        let bytes = self.read_chain(e.cluster, size)?;
        ensure!(bytes.len() as u64 == size, "{path} is cut short");
        Ok(Some(bytes))
    }

    /// Whether the file at `path` (as [`Fat32::find`] takes it) holds `bytes`, however many: its
    /// entry's size first, then its clusters, a cluster at a time, to the first that differs, so
    /// none of it is read where its size differs, and no more than `bytes` holds where it does;
    /// None where there is no such file.
    fn holds(&self, path: &str, bytes: &[u8]) -> Result<Option<bool>> {
        let Some(e) = self.find(path)? else {
            return Ok(None);
        };
        if u64::from(e.size) != bytes.len() as u64 {
            return Ok(Some(false));
        }
        let mut rest = bytes;
        self.chain(e.cluster, bytes.len() as u64, &mut |part| {
            let same = rest.starts_with(part);
            if same {
                rest = &rest[part.len()..];
            }
            same
        })?;
        // A chain that ends before the file's size does not hold the rest.
        Ok(Some(rest.is_empty()))
    }

    /// The bytes of the clusters from `first`, at most `most`.
    fn read_chain(&self, first: u32, most: u64) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        self.chain(first, most, &mut |part| {
            out.extend_from_slice(part);
            true
        })?;
        Ok(out)
    }

    /// Gives `part` the bytes of the clusters from `first`, a cluster at a time, until the chain
    /// ends, `most` bytes are given (the last cluster's cut to them), or `part` returns false.
    fn chain(&self, first: u32, most: u64, part: &mut dyn FnMut(&[u8]) -> bool) -> Result<()> {
        let mut buf = vec![0u8; self.cluster_size as usize];
        let mut c = first;
        let mut given = 0;
        while (2..0x0fff_fff8).contains(&c) && given < most {
            let n = (most - given).min(self.cluster_size) as usize;
            let offset = self.data + u64::from(c - 2) * self.cluster_size;
            self.disk.read(offset, &mut buf[..n])?;
            if !part(&buf[..n]) {
                break;
            }
            given += n as u64;
            if given == most {
                break;
            }
            let mut next = [0u8; 4];
            self.disk.read(self.fat + u64::from(c) * 4, &mut next)?;
            c = u32::from_le_bytes(next) & 0x0fff_ffff;
        }
        Ok(())
    }
}

/// A directory entry: its long name where one precedes it, and its 8.3 name.
struct Entry {
    long: Option<String>,
    short: String,
    cluster: u32,
    size: u32,
}

impl Entry {
    fn is(&self, name: &str) -> bool {
        self.long
            .as_deref()
            .is_some_and(|l| l.eq_ignore_ascii_case(name))
            || self.short.eq_ignore_ascii_case(name)
    }
}

/// A directory's entries, deleted ones and the volume label left out.
fn entries(bytes: &[u8]) -> Vec<Entry> {
    // The long name's parts, in the order they are stored: its last part first.
    let mut parts: Vec<[u16; 13]> = Vec::new();
    let mut out = Vec::new();
    for e in bytes.as_chunks::<32>().0 {
        let attr = e[11];
        match e[0] {
            0 => break,
            0xe5 => parts.clear(),
            _ if attr & 0x3f == 0x0f => {
                if e[0] & 0x40 != 0 {
                    parts.clear();
                }
                let offsets = [1, 3, 5, 7, 9, 14, 16, 18, 20, 22, 24, 28, 30];
                parts.push(offsets.map(|o| le16(e, o)));
            }
            _ if attr & 0x08 != 0 => parts.clear(),
            _ => {
                let long = (!parts.is_empty()).then(|| {
                    let units: Vec<u16> = parts
                        .iter()
                        .rev()
                        .flatten()
                        .copied()
                        .take_while(|&u| u != 0)
                        .collect();
                    String::from_utf16_lossy(&units)
                });
                parts.clear();
                let base = String::from_utf8_lossy(&e[..8]).trim_end().to_string();
                let ext = String::from_utf8_lossy(&e[8..11]).trim_end().to_string();
                out.push(Entry {
                    long,
                    short: format!("{base}.{ext}").trim_end_matches('.').to_string(),
                    cluster: u32::from(le16(e, 20)) << 16 | u32::from(le16(e, 26)),
                    size: le32(e, 28),
                });
            }
        }
    }
    out
}

pub(crate) fn le16(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}

pub(crate) fn le32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

pub(crate) fn le64(b: &[u8], o: usize) -> u64 {
    u64::from(le32(b, o)) | u64::from(le32(b, o + 4)) << 32
}

fn be32(b: &[u8], o: usize) -> u32 {
    u32::from_be_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

fn be64(b: &[u8], o: usize) -> u64 {
    u64::from(be32(b, o)) << 32 | u64::from(be32(b, o + 4))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::cell::Cell;

    thread_local! {
        /// The reads of images this thread made ([`Disk::read`]), and the bytes they read.
        pub(crate) static READS: Cell<(u64, u64)> = const { Cell::new((0, 0)) };
    }

    /// A directory entry of `name`, an 8.3 name, with no long name before it.
    fn short(name: &[u8; 11], attr: u8, cluster: u32, size: u32) -> [u8; 32] {
        let mut e = [0u8; 32];
        e[..11].copy_from_slice(name);
        e[11] = attr;
        e[20..22].copy_from_slice(&((cluster >> 16) as u16).to_le_bytes());
        e[26..28].copy_from_slice(&(cluster as u16).to_le_bytes());
        e[28..32].copy_from_slice(&size.to_le_bytes());
        e
    }

    /// The long-name entries of `name`, last part first, as FAT stores them.
    fn long(name: &str) -> Vec<[u8; 32]> {
        let mut units: Vec<u16> = name.encode_utf16().collect();
        units.push(0);
        while !units.len().is_multiple_of(13) {
            units.push(0xffff);
        }
        let n = units.len() / 13;
        (0..n)
            .rev()
            .map(|i| {
                let mut e = [0u8; 32];
                e[0] = (i + 1) as u8 | if i == n - 1 { 0x40 } else { 0 };
                e[11] = 0x0f;
                let offsets = [1, 3, 5, 7, 9, 14, 16, 18, 20, 22, 24, 28, 30];
                for (o, u) in offsets.into_iter().zip(&units[i * 13..]) {
                    e[o..o + 2].copy_from_slice(&u.to_le_bytes());
                }
                e
            })
            .collect()
    }

    #[test]
    fn entries_have_their_long_names() {
        let mut dir: Vec<[u8; 32]> = vec![short(b"EFI        ", 0x08, 0, 0)];
        dir.push(short(b"OC         ", 0x10, 0x1_0005, 0));
        dir.extend(long("config.plist"));
        dir.push(short(b"CONFIG~1PLI", 0x20, 7, 12345));
        let mut deleted = short(b"OLD     TXT", 0x20, 9, 1);
        deleted[0] = 0xe5;
        dir.push(deleted);
        dir.push([0; 32]);
        dir.push(short(b"AFTER      ", 0x20, 10, 1));
        let e = entries(&dir.concat());
        // The volume label, the deleted entry and what follows the end are not entries.
        assert_eq!(e.len(), 2);
        assert!(e[0].is("oc") && e[0].long.is_none() && e[0].cluster == 0x1_0005);
        assert!(e[1].is("config.plist") && e[1].is("CONFIG~1.PLI"));
        assert_eq!((e[1].cluster, e[1].size), (7, 12345));
    }

    fn config(kexts: &str) -> Config {
        Config::parse(plist(kexts).as_bytes()).unwrap()
    }

    /// A config.plist, Kernel > Add holding `kexts`.
    pub(crate) fn plist(kexts: &str) -> String {
        format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Kernel</key><dict><key>Add</key><array>{kexts}</array></dict>
  <key>PlatformInfo</key><dict>
    <key>Automatic</key><true/>
    <key>Generic</key><dict>
      <key>MLB</key><string>C02029100GU0000JA</string>
      <key>ROM</key><data>UlQAAQID</data>
      <key>SystemProductName</key><string>iMac20,1</string>
      <key>SystemSerialNumber</key><string>C02D40VEPN5T</string>
      <key>SystemUUID</key><string>4A2B8C1D-0E3F-4A5B-8C6D-7E8F9A0B1C2D</string>
    </dict>
  </dict>
</dict></plist>"#
        )
    }

    /// A raw OpenCore image laid out as reims-util's: a GPT whose one partition, the EFI system
    /// partition, is FAT32, a sector a cluster, holding EFI/OC/`config_plist`.
    pub(crate) fn image(config_plist: &[u8]) -> Vec<u8> {
        image_of(&[("EFI/OC/config.plist", config_plist)])
    }

    /// [`image`]'s layout holding `files`, each at its path from the volume's root, `/` between
    /// its names: each directory and file in a chain of its own, in the order first named.
    pub(crate) fn image_of(files: &[(&str, &[u8])]) -> Vec<u8> {
        enum Node {
            Dir(Vec<(String, usize)>),
            File(Vec<u8>),
        }
        let mut nodes = vec![Node::Dir(Vec::new())];
        for (path, bytes) in files {
            let names: Vec<&str> = path.split('/').collect();
            let mut at = 0;
            for (i, name) in names.iter().enumerate() {
                let Node::Dir(children) = &nodes[at] else {
                    panic!("{path}: a file on its way")
                };
                if let Some((_, c)) = children.iter().find(|(n, _)| n == name) {
                    at = *c;
                    continue;
                }
                nodes.push(if i + 1 == names.len() {
                    Node::File(bytes.to_vec())
                } else {
                    Node::Dir(Vec::new())
                });
                let c = nodes.len() - 1;
                if let Node::Dir(children) = &mut nodes[at] {
                    children.push((name.to_string(), c));
                }
                at = c;
            }
        }
        // A directory's entries: each child's long name, then its 8.3 entry; then the end.
        let entries_of = |children: &[(String, usize)], first: &[usize]| -> Vec<u8> {
            let mut d = Vec::new();
            for (name, c) in children {
                d.extend(long(name).concat());
                let short_name: [u8; 11] = format!("N{c:07}   ").into_bytes().try_into().unwrap();
                let (attr, size) = match &nodes[*c] {
                    Node::Dir(_) => (0x10, 0),
                    Node::File(b) => (0x20, b.len() as u32),
                };
                d.extend(short(&short_name, attr, first[*c] as u32, size));
            }
            d.extend([0u8; 32]);
            d
        };
        // Each node's clusters, 16 entries to a directory's, and its first (0 for an empty
        // file), the root's 2.
        let sizes: Vec<usize> = nodes
            .iter()
            .map(|n| match n {
                Node::Dir(children) => (children
                    .iter()
                    .map(|(name, _)| long(name).len() + 1)
                    .sum::<usize>()
                    + 1)
                .div_ceil(16),
                Node::File(b) => b.len().div_ceil(512),
            })
            .collect();
        let mut first = Vec::new();
        let mut next = 2;
        for s in &sizes {
            first.push(if *s == 0 { 0 } else { next });
            next += s;
        }
        const ESP: usize = 64 * 512;
        let put = |img: &mut Vec<u8>, at: usize, bytes: &[u8]| {
            img[at..at + bytes.len()].copy_from_slice(bytes)
        };
        // The boot sector, one FAT, then cluster 2.
        let fat_sectors = (next * 4).div_ceil(512);
        let data = ESP + (1 + fat_sectors) * 512;
        let cluster = |c: usize| data + (c - 2) * 512;
        let mut img = vec![0u8; cluster(next)];
        put(&mut img, 512, b"EFI PART");
        put(&mut img, 512 + 72, &2u64.to_le_bytes());
        put(&mut img, 512 + 80, &1u32.to_le_bytes());
        put(&mut img, 512 + 84, &128u32.to_le_bytes());
        put(&mut img, 1024, &ESP_TYPE);
        put(&mut img, 1024 + 32, &(ESP as u64 / 512).to_le_bytes());
        put(&mut img, ESP + 11, &512u16.to_le_bytes());
        img[ESP + 13] = 1;
        put(&mut img, ESP + 14, &1u16.to_le_bytes());
        img[ESP + 16] = 1;
        put(&mut img, ESP + 36, &(fat_sectors as u32).to_le_bytes());
        put(&mut img, ESP + 44, &2u32.to_le_bytes());
        put(&mut img, ESP + 510, &[0x55, 0xaa]);
        for (n, (at, size)) in nodes.iter().zip(first.iter().zip(&sizes)) {
            for c in *at..*at + *size {
                let link = if c + 1 == at + size {
                    0x0fff_ffff
                } else {
                    c as u32 + 1
                };
                put(&mut img, ESP + 512 + c * 4, &link.to_le_bytes());
            }
            let bytes = match n {
                Node::Dir(children) => entries_of(children, &first),
                Node::File(b) => b.clone(),
            };
            if *size > 0 {
                put(&mut img, cluster(*at), &bytes);
            }
        }
        img
    }

    /// `raw` as a qcow2 image, the format reims-util writes: version 2, clusters of 512 bytes,
    /// one L1 table, then the L2 tables, then each cluster holding a byte other than zero; a
    /// cluster of zeros is left unallocated.
    pub(crate) fn qcow2(raw: &[u8]) -> Vec<u8> {
        const BITS: u32 = 9;
        const CLUSTER: usize = 1 << BITS;
        const PER_TABLE: usize = CLUSTER / 8;
        let clusters = raw.len().div_ceil(CLUSTER);
        let tables = clusters.div_ceil(PER_TABLE).max(1);
        let l1_clusters = (tables * 8).div_ceil(CLUSTER);
        let l2_at = |t: usize| (1 + l1_clusters + t) * CLUSTER;
        let mut img = vec![0u8; l2_at(tables)];
        img[..4].copy_from_slice(b"QFI\xfb");
        img[4..8].copy_from_slice(&2u32.to_be_bytes());
        img[20..24].copy_from_slice(&BITS.to_be_bytes());
        img[24..32].copy_from_slice(&(raw.len() as u64).to_be_bytes());
        img[36..40].copy_from_slice(&(tables as u32).to_be_bytes());
        img[40..48].copy_from_slice(&(CLUSTER as u64).to_be_bytes());
        // Bit 63 of an entry: the cluster is used once, as QEMU marks the clusters it writes.
        for t in 0..tables {
            let e = l2_at(t) as u64 | 1 << 63;
            img[CLUSTER + t * 8..CLUSTER + t * 8 + 8].copy_from_slice(&e.to_be_bytes());
        }
        for (c, data) in raw.chunks(CLUSTER).enumerate() {
            if data.iter().all(|&b| b == 0) {
                continue;
            }
            let at = img.len();
            img.extend_from_slice(data);
            img.resize(at + CLUSTER, 0);
            let e = (at as u64 | 1 << 63).to_be_bytes();
            let slot = l2_at(c / PER_TABLE) + c % PER_TABLE * 8;
            img[slot..slot + 8].copy_from_slice(&e);
        }
        img
    }

    #[test]
    fn an_image_reads_as_reims_util_lays_it_out() {
        let d = std::env::temp_dir().join(format!("reims-ctl-oc-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let p = d.join("opencore.img");
        std::fs::write(&p, image(plist("").as_bytes())).unwrap();
        let c = Config::read(&p, ImageFormat::Raw).unwrap();
        assert_eq!(c.identity().unwrap().serial, "C02D40VEPN5T");
        let p = d.join("opencore.qcow2");
        std::fs::write(&p, qcow2(&image(plist("").as_bytes()))).unwrap();
        let c = Config::read(&p, ImageFormat::Qcow2).unwrap();
        assert_eq!(c.identity().unwrap().serial, "C02D40VEPN5T");
        std::fs::remove_dir_all(&d).unwrap();
    }

    /// The new image is set in the VM's settings as they are once it is made: what was saved
    /// while reims-util made it stays.
    #[test]
    fn renew_keeps_settings_saved_while_the_image_is_made() {
        let env = crate::testenv::Env::new("renew");
        let dir = env.root.join("vm");
        std::fs::create_dir_all(&dir).unwrap();
        let made = env.root.join("made.img");
        std::fs::write(&made, qcow2(&image(plist("").as_bytes()))).unwrap();
        std::fs::write(dir.join("opencore.img"), image(plist("").as_bytes())).unwrap();
        let config = |memory: u64| {
            format!(
                r#"
                name = "t"
                profile = "q35-x86_64"
                cpus = 2
                memory_mib = {memory}
                disk = "disk.qcow2"
                [q35]
                opencore = "opencore.img"
                opencore_format = "raw"
                firmware_code = "OVMF_CODE.fd"
                firmware_vars = "OVMF_VARS.fd"
                "#
            )
        };
        std::fs::write(dir.join(crate::vm::CONFIG_FILE), config(4096)).unwrap();
        let mut vm = Vm::load(&dir).unwrap();
        // reims-util makes the image, and meanwhile the VM's memory is saved (reims-ctl set).
        let util = env.root.join("reims-util");
        let script = format!(
            "#!/bin/sh\nwhile [ \"$1\" != -o ]; do shift; done\ncp '{made}' \"$2\"\n\
             cat > '{toml}' <<'EOF'\n{config}\nEOF\necho '{{\"event\":\"done\",\"result\":{{}}}}'\n",
            made = made.display(),
            toml = dir.join(crate::vm::CONFIG_FILE).display(),
            config = config(8192),
        );
        let util = crate::testenv::program(&util, &script);
        env.set("REIMS_CTL_UTIL", &util);

        let said = renew(&mut vm, None, &mut |_| {}).unwrap();
        assert!(said.contains("iMac20,1"), "{said}");
        let saved = Vm::load(&dir).unwrap();
        assert_eq!(saved.config.memory_mib, 8192);
        let q = saved.config.q35.as_ref().unwrap();
        assert_eq!(q.opencore, Path::new("opencore.qcow2"));
        assert_eq!(q.opencore_format, ImageFormat::Qcow2);
        assert_eq!(vm.config, saved.config);
        // The image it replaced, the VM's own under another name, is gone.
        assert!(dir.join("opencore.qcow2").is_file());
        assert!(!dir.join("opencore.img").exists());
    }

    /// The old image goes only where its path, resolved, names a file in the VM's directory
    /// other than the new image: an OpenCore path edited to `../other/oc.img` leaves that file
    /// be, and one spelling the new image's place another way, `sub/../opencore.qcow2`, leaves
    /// the new image.
    #[test]
    fn renew_deletes_no_file_but_an_old_one_in_the_vms_directory() {
        let env = crate::testenv::Env::new("renew-outside");
        let made = env.root.join("made.img");
        std::fs::write(&made, qcow2(&image(plist("").as_bytes()))).unwrap();
        let util = env.root.join("reims-util");
        let script = format!(
            "#!/bin/sh\nwhile [ \"$1\" != -o ]; do shift; done\ncp '{}' \"$2\"\n\
             echo '{{\"event\":\"done\",\"result\":{{}}}}'\n",
            made.display()
        );
        let util = crate::testenv::program(&util, &script);
        env.set("REIMS_CTL_UTIL", &util);
        let renewed = |vm_dir: &str, opencore: &str, old: &Path| {
            let dir = env.root.join(vm_dir);
            std::fs::create_dir_all(dir.join("sub")).unwrap();
            std::fs::create_dir_all(old.parent().unwrap()).unwrap();
            std::fs::write(old, image(plist("").as_bytes())).unwrap();
            let config = format!(
                r#"
                name = "{vm_dir}"
                profile = "q35-x86_64"
                cpus = 2
                memory_mib = 4096
                disk = "disk.qcow2"
                [q35]
                opencore = "{opencore}"
                opencore_format = "raw"
                firmware_code = "OVMF_CODE.fd"
                firmware_vars = "OVMF_VARS.fd"
                "#
            );
            std::fs::write(dir.join(crate::vm::CONFIG_FILE), config).unwrap();
            let mut vm = Vm::load(&dir).unwrap();
            renew(&mut vm, None, &mut |_| {}).unwrap();
            dir
        };
        let outside = env.root.join("other/oc.img");
        let dir = renewed("a", "../other/oc.img", &outside);
        assert!(dir.join("opencore.qcow2").is_file());
        assert!(
            outside.is_file(),
            "a file outside the VM's directory was deleted"
        );
        let dir = renewed(
            "b",
            "sub/../opencore.qcow2",
            &env.root.join("b/opencore.qcow2"),
        );
        assert!(
            dir.join("opencore.qcow2").is_file(),
            "the new image was deleted"
        );
    }

    /// The image of a VM is made for its graphics: on reims-vgpu with the guest's kexts the
    /// tests carry, in their order, written to the data directory; on VMware SVGA with none.
    #[test]
    fn renew_makes_the_image_for_the_vms_graphics() {
        let env = crate::testenv::Env::new("renew-graphics");
        let dir = env.root.join("vm");
        std::fs::create_dir_all(&dir).unwrap();
        let made = env.root.join("made.img");
        std::fs::write(&made, qcow2(&image(plist("").as_bytes()))).unwrap();
        std::fs::write(dir.join("opencore.img"), image(plist("").as_bytes())).unwrap();
        let args = env.root.join("args");
        let util = env.root.join("reims-util");
        let script = format!(
            "#!/bin/sh\nprintf '%s\\n' \"$@\" > '{args}'\n\
             while [ \"$1\" != -o ]; do shift; done\ncp '{made}' \"$2\"\n\
             echo '{{\"event\":\"done\",\"result\":{{\"kexts\":[]}}}}'\n",
            args = args.display(),
            made = made.display(),
        );
        let util = crate::testenv::program(&util, &script);
        env.set("REIMS_CTL_UTIL", &util);
        let config = |graphics: &str| {
            format!(
                r#"
                name = "t"
                profile = "q35-x86_64"
                cpus = 2
                memory_mib = 4096
                disk = "disk.qcow2"
                {graphics}
                [q35]
                opencore = "opencore.img"
                opencore_format = "raw"
                firmware_code = "OVMF_CODE.fd"
                firmware_vars = "OVMF_VARS.fd"
                "#
            )
        };
        std::fs::write(dir.join(crate::vm::CONFIG_FILE), config("")).unwrap();
        let mut vm = Vm::load(&dir).unwrap();
        renew(&mut vm, None, &mut |_| {}).unwrap();
        let given = std::fs::read_to_string(&args).unwrap();
        let given: Vec<&str> = given.lines().collect();
        let added: Vec<&str> = given
            .windows(2)
            .filter(|w| w[0] == "--add-kext")
            .map(|w| w[1])
            .collect();
        let kexts = env.root.join("data").join("prebuilt");
        assert_eq!(added.len(), 2, "{given:?}");
        let sep = std::path::MAIN_SEPARATOR;
        assert!(
            added[0].starts_with(kexts.to_str().unwrap())
                && added[0].ends_with(&format!("{sep}A.kext,25.4.0,25.99.99")),
            "{added:?}"
        );
        assert!(added[1].ends_with(&format!("{sep}B.kext")), "{added:?}");
        let a = added[0].split(',').next().unwrap();
        assert!(Path::new(a).join("Contents/MacOS/A").is_file());
        assert!(!given.contains(&"--no-kexts"), "{given:?}");
        std::fs::write(dir.join("opencore.img"), image(plist("").as_bytes())).unwrap();
        std::fs::write(
            dir.join(crate::vm::CONFIG_FILE),
            config("graphics = \"vmware-svga\""),
        )
        .unwrap();
        let mut vm = Vm::load(&dir).unwrap();
        renew(&mut vm, None, &mut |_| {}).unwrap();
        let given = std::fs::read_to_string(&args).unwrap();
        assert!(!given.contains("--add-kext"), "{given}");
    }

    /// A VM on VMware SVGA whose image loads a kext of the guest's set, as one made for
    /// reims-vgpu does, has its image made for other graphics; one whose image loads none does
    /// not. On reims-vgpu, an image lacking them is made for other graphics.
    #[test]
    fn an_image_follows_the_graphics_by_the_guest_kexts_it_loads() {
        let env = crate::testenv::Env::new("follow");
        let dir = env.root.join("vm");
        std::fs::create_dir_all(&dir).unwrap();
        let entry = |bundle: &str| {
            format!(
                "<dict><key>BundlePath</key><string>{bundle}</string>\
                 <key>Enabled</key><true/></dict>"
            )
        };
        let vm = |graphics: &str, kexts: &str| {
            std::fs::write(dir.join("opencore.img"), image(plist(kexts).as_bytes())).unwrap();
            let config = format!(
                r#"
                name = "t"
                profile = "q35-x86_64"
                cpus = 2
                memory_mib = 4096
                disk = "disk.img"
                disk_format = "raw"
                graphics = "{graphics}"
                [q35]
                opencore = "opencore.img"
                opencore_format = "raw"
                firmware_code = "OVMF_CODE.fd"
                firmware_vars = "OVMF_VARS.fd"
                "#
            );
            std::fs::write(dir.join(crate::vm::CONFIG_FILE), config).unwrap();
            Vm::load(&dir).unwrap()
        };
        let lilu = entry("Lilu.kext");
        let ours = lilu.clone() + &entry("A.kext") + &entry("B.kext");
        let v = vm("vmware-svga", &ours);
        assert!(holds_unused_guest_kexts(&v) && made_for_other_graphics(&v));
        let v = vm("vmware-svga", &lilu);
        assert!(!holds_unused_guest_kexts(&v) && !made_for_other_graphics(&v));
        let v = vm("reims-vgpu", &ours);
        assert!(!holds_unused_guest_kexts(&v));
        let v = vm("reims-vgpu", &lilu);
        assert!(!holds_unused_guest_kexts(&v) && made_for_other_graphics(&v));
    }

    /// A start of a VM on VMware SVGA whose image loads the guest's kexts, as one made for
    /// reims-vgpu does, makes it a new image for VMware SVGA, without them, first, telling of the
    /// AppleGVA setting they left in its guest; a start on reims-vgpu leaves an image lacking
    /// them be, which its warnings tell of.
    #[test]
    fn a_start_makes_an_image_without_the_guest_kexts_for_other_graphics() {
        let env = crate::testenv::Env::new("start-follow");
        let dir = env.root.join("vm");
        std::fs::create_dir_all(&dir).unwrap();
        let entry = |bundle: &str| {
            format!(
                "<dict><key>BundlePath</key><string>{bundle}</string>\
                 <key>Enabled</key><true/></dict>"
            )
        };
        let lilu = entry("Lilu.kext");
        let made = env.root.join("made.img");
        std::fs::write(&made, qcow2(&image(plist("").as_bytes()))).unwrap();
        let args = env.root.join("args");
        let util = env.root.join("reims-util");
        let script = format!(
            "#!/bin/sh\nprintf '%s\\n' \"$@\" > '{args}'\n\
             while [ \"$1\" != -o ]; do shift; done\ncp '{made}' \"$2\"\n\
             echo '{{\"event\":\"done\",\"result\":{{}}}}'\n",
            args = args.display(),
            made = made.display(),
        );
        let util = crate::testenv::program(&util, &script);
        env.set("REIMS_CTL_UTIL", &util);
        let vm = |graphics: &str, kexts: &str| {
            std::fs::write(dir.join("opencore.img"), image(plist(kexts).as_bytes())).unwrap();
            let config = format!(
                r#"
                name = "t"
                profile = "q35-x86_64"
                cpus = 2
                memory_mib = 4096
                disk = "disk.img"
                disk_format = "raw"
                graphics = "{graphics}"
                [q35]
                opencore = "opencore.img"
                opencore_format = "raw"
                firmware_code = "OVMF_CODE.fd"
                firmware_vars = "OVMF_VARS.fd"
                "#
            );
            std::fs::write(dir.join(crate::vm::CONFIG_FILE), config).unwrap();
            Vm::load(&dir).unwrap()
        };
        // Its guest keeps what RgfxGate wrote, which the new image's note tells of.
        crate::macos::tests::gva_disk(&dir.join("disk.img"));
        let mut v = vm(
            "vmware-svga",
            &(lilu.clone() + &entry("A.kext") + &entry("B.kext")),
        );
        let mut said = Vec::new();
        let warned = crate::runtime::prepare(&mut v, &mut |s| said.push(s.to_string()));
        assert!(warned.is_empty(), "{warned:?}");
        assert!(said.iter().any(|s| s.contains("without them")), "{said:?}");
        assert!(said.iter().any(|s| s.contains("AppleGVA")), "{said:?}");
        assert!(
            !std::fs::read_to_string(&args)
                .unwrap()
                .contains("--add-kext")
        );
        let q = v.config.q35.as_ref().unwrap();
        assert_eq!(q.opencore, Path::new("opencore.qcow2"));
        assert!(!holds_unused_guest_kexts(&v));
        std::fs::remove_file(&args).unwrap();
        // Started again, it makes no image, and tells nothing again.
        let mut said = Vec::new();
        crate::runtime::prepare(&mut v, &mut |s| said.push(s.to_string()));
        assert!(said.is_empty() && !args.exists(), "{said:?}");
        // Lacking them on reims-vgpu, the image stays.
        let mut v = vm("reims-vgpu", &lilu);
        crate::runtime::prepare(&mut v, &mut |_| {});
        assert!(!args.exists());
        assert_eq!(
            v.config.q35.as_ref().unwrap().opencore,
            Path::new("opencore.img")
        );
    }

    /// A start of a VM on reims-vgpu whose image loads the guest's kexts in other builds than
    /// reims-ctl carries, as one an older reims-ctl made does, makes it a new image with these
    /// first; one lacking them stays.
    #[test]
    fn a_start_makes_an_image_with_the_guest_kexts_carried_for_other_builds() {
        let env = crate::testenv::Env::new("start-builds");
        let dir = env.root.join("vm");
        std::fs::create_dir_all(&dir).unwrap();
        let entry = |bundle: &str| {
            format!(
                "<dict><key>BundlePath</key><string>{bundle}</string>\
                 <key>Enabled</key><true/></dict>"
            )
        };
        let lilu = entry("Lilu.kext");
        let made = env.root.join("made.img");
        std::fs::write(&made, qcow2(&image(plist("").as_bytes()))).unwrap();
        let args = env.root.join("args");
        let util = env.root.join("reims-util");
        let script = format!(
            "#!/bin/sh\nprintf '%s\\n' \"$@\" > '{args}'\n\
             while [ \"$1\" != -o ]; do shift; done\ncp '{made}' \"$2\"\n\
             echo '{{\"event\":\"done\",\"result\":{{}}}}'\n",
            args = args.display(),
            made = made.display(),
        );
        let util = crate::testenv::program(&util, &script);
        env.set("REIMS_CTL_UTIL", &util);
        let config = r#"
            name = "t"
            profile = "q35-x86_64"
            cpus = 2
            memory_mib = 4096
            disk = "disk.img"
            disk_format = "raw"
            graphics = "reims-vgpu"
            [q35]
            opencore = "opencore.img"
            opencore_format = "raw"
            firmware_code = "OVMF_CODE.fd"
            firmware_vars = "OVMF_VARS.fd"
            "#;
        std::fs::write(dir.join(crate::vm::CONFIG_FILE), config).unwrap();
        // Each kext loaded, in order, but none with the executable reims-ctl carries.
        let ours = lilu.clone() + &entry("A.kext") + &entry("B.kext");
        std::fs::write(dir.join("opencore.img"), image(plist(&ours).as_bytes())).unwrap();
        let mut v = Vm::load(&dir).unwrap();
        assert!(holds_other_guest_kexts(&v) && !holds_unused_guest_kexts(&v));
        let mut said = Vec::new();
        let warned = crate::runtime::prepare(&mut v, &mut |s| said.push(s.to_string()));
        assert!(warned.is_empty(), "{warned:?}");
        assert!(said.iter().any(|s| s.contains("other builds")), "{said:?}");
        assert!(
            std::fs::read_to_string(&args)
                .unwrap()
                .contains("--add-kext")
        );
        let q = v.config.q35.as_ref().unwrap();
        assert_eq!(q.opencore, Path::new("opencore.qcow2"));
        std::fs::remove_file(&args).unwrap();
        // Lacking them, as the image the stand-in reims-util made does, it stays.
        assert!(!holds_other_guest_kexts(&v));
        let mut said = Vec::new();
        crate::runtime::prepare(&mut v, &mut |s| said.push(s.to_string()));
        assert!(said.is_empty() && !args.exists(), "{said:?}");
    }

    /// A Kernel > Add entry loading `bundle`.
    pub(crate) fn enabled(bundle: &str) -> String {
        format!(
            "<dict><key>BundlePath</key><string>{bundle}</string><key>Enabled</key><true/></dict>"
        )
    }

    /// An image as reims-util makes it with the set the tests carry: config.plist loading A.kext
    /// then B.kext, and each bundle's files under EFI/OC/Kexts, A's executable `a`; then `more`.
    fn image_with_set(a: &[u8], more: &[(&str, &[u8])]) -> Vec<u8> {
        image_loading(&(enabled("A.kext") + &enabled("B.kext")), a, more)
    }

    /// [`image_with_set`]'s files, its config.plist's Kernel > Add holding `kexts`.
    pub(crate) fn image_loading(kexts: &str, a: &[u8], more: &[(&str, &[u8])]) -> Vec<u8> {
        let config = plist(kexts);
        let mut files: Vec<(&str, &[u8])> = vec![
            ("EFI/OC/config.plist", config.as_bytes()),
            (
                "EFI/OC/Kexts/A.kext/Contents/Info.plist",
                &b"<plist>a</plist>"[..],
            ),
            ("EFI/OC/Kexts/A.kext/Contents/MacOS/A", a),
            (
                "EFI/OC/Kexts/B.kext/Contents/Info.plist",
                &b"<plist>b</plist>"[..],
            ),
        ];
        files.extend(more);
        image_of(&files)
    }

    /// A VM on reims-vgpu in `dir` whose OpenCore image is `image`, in `format`.
    pub(crate) fn on_reims_vgpu(dir: &Path, image: &[u8], format: ImageFormat) -> Vm {
        let (file, format) = match format {
            ImageFormat::Raw => ("opencore.img", "raw"),
            ImageFormat::Qcow2 => ("opencore.qcow2", "qcow2"),
        };
        std::fs::write(dir.join(file), image).unwrap();
        let config = format!(
            r#"
            name = "t"
            profile = "q35-x86_64"
            cpus = 2
            memory_mib = 4096
            disk = "disk.img"
            disk_format = "raw"
            graphics = "reims-vgpu"
            [q35]
            opencore = "{file}"
            opencore_format = "{format}"
            firmware_code = "OVMF_CODE.fd"
            firmware_vars = "OVMF_VARS.fd"
            "#
        );
        std::fs::write(dir.join(crate::vm::CONFIG_FILE), config).unwrap();
        Vm::load(dir).unwrap()
    }

    /// The executable of A.kext the tests carry.
    pub(crate) fn carried_a() -> &'static [u8] {
        crate::prebuilt::set().unwrap()[0]
            .executable_bytes()
            .unwrap()
    }

    /// An image reims-util made with the guest's kexts this reims-ctl carries holds them, read
    /// through qcow2 as through raw, whatever else its volume comes to hold beside them (what
    /// OpenCore or a guest's macOS may write to it while it runs: a log in its root, a directory
    /// of its own, a directory of many entries). One whose executable is another build's holds
    /// other builds, of another size or of the same.
    #[test]
    fn an_image_holds_the_guest_kexts_it_was_made_with() {
        let env = crate::testenv::Env::new("made-with");
        let dir = env.root.join("vm");
        std::fs::create_dir_all(&dir).unwrap();
        let names: Vec<String> = (0..40)
            .map(|i| format!("a file of a long name {i}"))
            .collect();
        let mut more: Vec<(&str, &[u8])> = vec![
            ("opencore-2026-11-23-120000.txt", &[b'x'; 3000][..]),
            ("EFI/APPLE/UPDATERS/MULTIUPDATER/flash.bin", &[1u8; 700][..]),
        ];
        let paths: Vec<String> = names.iter().map(|n| format!(".fseventsd/{n}")).collect();
        more.extend(paths.iter().map(|p| (p.as_str(), &b"event"[..])));
        for format in [ImageFormat::Raw, ImageFormat::Qcow2] {
            let as_made = |raw: Vec<u8>| match format {
                ImageFormat::Raw => raw,
                ImageFormat::Qcow2 => qcow2(&raw),
            };
            let v = on_reims_vgpu(&dir, &as_made(image_with_set(carried_a(), &[])), format);
            assert!(holds_guest_kexts(&v), "{format:?}");
            let v = on_reims_vgpu(&dir, &as_made(image_with_set(carried_a(), &more)), format);
            assert!(holds_guest_kexts(&v), "{format:?}");
            let mut same_size = carried_a().to_vec();
            *same_size.last_mut().unwrap() ^= 1;
            for other in [&b"\xcf\xfa\xed\xfe an older a"[..], &same_size[..]] {
                let v = on_reims_vgpu(&dir, &as_made(image_with_set(other, &more)), format);
                assert!(
                    !holds_guest_kexts(&v) && holds_other_guest_kexts(&v),
                    "{format:?}"
                );
            }
        }
    }

    /// A kext's executable is compared whole, however large: one of more than 16 MiB, as
    /// RgfxGate's may come to be with the Metal bundle it embeds, is held where each of its bytes
    /// is the carried one's, read through qcow2 as through raw, and another build where only its
    /// last byte differs.
    #[test]
    fn an_image_holds_an_executable_of_more_than_16_mib_as_carried() {
        let big: &'static [u8] =
            Vec::leak((0..(16 << 20) + 4099u32).map(|i| (i % 251) as u8).collect());
        let files: &'static [(&'static str, &'static [u8])] =
            Vec::leak(vec![("Contents/MacOS/Big", big)]);
        let set: &'static [Kext] = Vec::leak(vec![Kext {
            bundle: "Big.kext",
            executable: Some("Contents/MacOS/Big"),
            min_kernel: None,
            max_kernel: None,
            files,
        }]);
        let d = std::env::temp_dir().join(format!("reims-ctl-big-kext-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let p = d.join("opencore.img");
        let config = plist(&enabled("Big.kext"));
        let mut other = big.to_vec();
        *other.last_mut().unwrap() ^= 1;
        for format in [ImageFormat::Raw, ImageFormat::Qcow2] {
            let held = |exe: &[u8]| {
                let raw = image_of(&[
                    ("EFI/OC/config.plist", config.as_bytes()),
                    ("EFI/OC/Kexts/Big.kext/Contents/MacOS/Big", exe),
                ]);
                let image = match format {
                    ImageFormat::Raw => raw,
                    ImageFormat::Qcow2 => qcow2(&raw),
                };
                std::fs::write(&p, image).unwrap();
                guest_kexts_in(&p, format, || Ok(set)).unwrap().unwrap()
            };
            assert_eq!(held(big), GuestKexts::Held, "{format:?}");
            assert_eq!(
                held(&other),
                GuestKexts::Other(vec!["Big.kext"]),
                "{format:?}"
            );
        }
        std::fs::remove_dir_all(&d).unwrap();
    }

    /// After an update of reims-ctl, the image of a VM an older reims-ctl made holds other builds
    /// of the guest's kexts until the VM's next start makes it again with these
    /// (`runtime::prepare`); a VM running since before the update keeps it, as a macOS install,
    /// which keeps its VM running for an hour or more, does. Its warnings tell nothing of it, and a
    /// change of its graphics to reims-vgpu makes it again all the same. An image that lacks them,
    /// as one of the user's own may, which no start makes again, is warned of.
    #[test]
    fn an_image_an_older_reims_ctl_made_is_not_warned_of() {
        let env = crate::testenv::Env::new("older-build");
        let dir = env.root.join("vm");
        std::fs::create_dir_all(&dir).unwrap();
        let host = crate::host::Host {
            os: crate::host::Os::Windows,
            arch: crate::host::Arch::X86_64,
        };
        let of_drivers = |v: &Vm| -> Vec<String> {
            crate::runtime::warnings(v, host)
                .into_iter()
                .filter(|w| w.contains("drivers"))
                .collect()
        };
        let older = image_with_set(b"\xcf\xfa\xed\xfe an older a", &[]);
        let v = on_reims_vgpu(&dir, &qcow2(&older), ImageFormat::Qcow2);
        assert!(holds_other_guest_kexts(&v) && made_for_other_graphics(&v));
        assert_eq!(missing_guest_kexts(&v), None);
        assert_eq!(of_drivers(&v), Vec::<String>::new());
        // Lacking A.kext, it is warned of, with how to make it again.
        let lacking = image_of(&[("EFI/OC/config.plist", plist(&enabled("B.kext")).as_bytes())]);
        let v = on_reims_vgpu(&dir, &qcow2(&lacking), ImageFormat::Qcow2);
        let said = missing_guest_kexts(&v).unwrap();
        assert!(
            said.contains("lacks reims-vgpu's macOS drivers (A.kext)")
                && said.contains("update-opencore t"),
            "{said}"
        );
        assert_eq!(of_drivers(&v), [said]);
        assert!(!holds_other_guest_kexts(&v) && made_for_other_graphics(&v));
    }

    /// An image holds the guest's kexts where OpenCore loads each, in the set's order, with
    /// the executable reims-ctl carries; one it does not load is lacking, and one out of order
    /// or of another build is another.
    #[test]
    fn an_image_holds_the_guest_kexts_loaded_in_order_as_carried() {
        let set = crate::prebuilt::set().unwrap();
        let held = guest_kexts(
            set,
            &["Lilu.kext", "ReimsVgpu.kext", "a.KEXT", "B.kext"],
            &mut |_| true,
        );
        assert_eq!(held, GuestKexts::Held);
        assert_eq!(
            guest_kexts(set, &["Lilu.kext", "B.kext"], &mut |_| true),
            GuestKexts::Lacks(vec!["A.kext"])
        );
        assert_eq!(
            guest_kexts(set, &["B.kext", "A.kext"], &mut |_| true),
            GuestKexts::Other(vec!["B.kext"])
        );
        // An executable the image holds otherwise than carried, or not at all.
        let mut asked = Vec::new();
        let other = guest_kexts(set, &["A.kext", "B.kext"], &mut |k| {
            asked.push(k.bundle);
            false
        });
        assert_eq!(other, GuestKexts::Other(vec!["A.kext"]));
        // B.kext has no executable to compare.
        assert_eq!(asked, ["A.kext"]);
        let entries = "<dict><key>BundlePath</key><string>A.kext</string><key>Enabled</key><true/></dict>\
             <dict><key>BundlePath</key><string>B.kext</string><key>Enabled</key><false/></dict>";
        assert_eq!(config(entries).loaded_kexts(), ["A.kext"]);
    }

    #[test]
    fn identity_is_platform_infos_generic() {
        assert_eq!(
            config("").identity().unwrap(),
            Identity {
                model: "iMac20,1".into(),
                serial: "C02D40VEPN5T".into(),
                mlb: "C02029100GU0000JA".into(),
                uuid: "4A2B8C1D-0E3F-4A5B-8C6D-7E8F9A0B1C2D".into(),
                rom: "525400010203".into(),
            }
        );
    }

    #[test]
    fn the_picker_shows_where_misc_boot_show_picker_is_true() {
        let with = |boot: &str| {
            let plist = format!(
                "<plist><dict><key>Misc</key><dict><key>Boot</key><dict>{boot}</dict></dict>\
                 </dict></plist>"
            );
            Config::parse(plist.as_bytes()).unwrap()
        };
        assert!(!config("").shows_picker());
        assert!(!with("").shows_picker());
        assert!(!with("<key>ShowPicker</key><false/>").shows_picker());
        assert!(with("<key>ShowPicker</key><true/>").shows_picker());
    }
}
