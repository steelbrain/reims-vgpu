//! What reims-ctl carries for reims-vgpu's own GPU device ([`crate::vm::Graphics::ReimsVgpu`]),
//! built into its binary from the repository:
//! - the device's option ROM, `firmware/prebuilt/ReimsGraphics.rom`, which QEMU's
//!   `reims-graphics-pci` has no default for: its `romfile` ([`graphics_rom`]);
//! - the guest's kexts, the set `guest/prebuilt/set.toml` lists, which build.rs embeds in the
//!   set's order, and which reims-ctl adds to the OpenCore image of a VM on reims-vgpu
//!   ([`guest_kexts`]): OpenCore loads them in that order;
//! - for an arm64 VM, which boots on Ambrosia: the arm64 guest's kexts, the set
//!   `guest/prebuilt-arm64e/set.toml` lists, and the device's driver module for Ambrosia, which
//!   build.rs builds from `firmware/ambrosia/ReimsGraphicsModule.c`. Its QEMU loads them into
//!   the guest as Ambrosia's configuration bundle ([`ambrosia_bundle`]), the bundle
//!   `make -C firmware/ambrosia config-bundle` makes of the same files; under KVM with
//!   `firmware/ambrosia/kvm.plist`'s patches added, as `config-bundle KVM=1` adds them.
//!
//! Embedded, they are the ones of the commit reims-ctl was built from, however
//! scripts/install-ctl.sh installed it (`cargo install --path`, with or without the QEMU of that
//! commit), and need nothing installed beside them.
//!
//! Each is written where it is used, into the data directory's `prebuilt`, under a name of its
//! content's hash: the ROM and the bundle where a VM's QEMU command is made, the set where
//! reims-util makes an image with it. One already there is used as it is where its bytes are the embedded ones, and
//! written again otherwise, into a file or a directory beside it that is renamed into its place,
//! so neither two reims-ctl writing at once nor one of another commit leaves one half-written or
//! changes one a running QEMU reads.

use crate::paths;
use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};

/// The option ROM of reims-vgpu's own device: `make -C firmware prebuilt` writes it, and
/// `firmware/prebuilt/ReimsGraphics.toml` records the sources it was built from.
pub static GRAPHICS_ROM: &[u8] = include_bytes!("../../../firmware/prebuilt/ReimsGraphics.rom");

/// A kext of the guest's set: its bundle's name, which is its `BundlePath` in OpenCore's
/// `Kernel > Add`, its executable's path in the bundle (none for a kext without code), the
/// Darwin versions OpenCore loads it on (`MinKernel` and `MaxKernel`, X.Y.Z; none is no bound),
/// and each of its files, by its path in the bundle.
#[derive(Debug)]
pub struct Kext {
    pub bundle: &'static str,
    pub executable: Option<&'static str>,
    pub min_kernel: Option<&'static str>,
    pub max_kernel: Option<&'static str>,
    pub files: &'static [(&'static str, &'static [u8])],
}

impl Kext {
    /// Its executable's bytes, where it has one.
    pub fn executable_bytes(&self) -> Option<&'static [u8]> {
        let exe = self.executable?;
        self.files.iter().find(|(p, _)| *p == exe).map(|(_, b)| *b)
    }
}

// What build.rs embeds: the sets, `SET` and `SET_ARM64E`, each None where the checkout it was
// built from had none, and `AMBROSIA_MODULE`, None where clang could not build it; for Windows,
// neither `SET_ARM64E` nor `AMBROSIA_MODULE`.
include!(concat!(env!("OUT_DIR"), "/set.rs"));

/// The config.plist of the device's bundle for Ambrosia, `firmware/ambrosia/config.plist`: it
/// names the module and the vtables Ambrosia hands Graft.
pub static AMBROSIA_CONFIG: &[u8] = include_bytes!("../../../firmware/ambrosia/config.plist");

/// What an arm64 VM's configuration adds under KVM, `firmware/ambrosia/kvm.plist`: its
/// `Kernel > Patch` alone, which rewrites the pre-indexed store macOS's kernel sets up each GIC
/// redistributor with, which KVM's in-kernel vGIC can't take (the file says why and where it was
/// measured).
pub static AMBROSIA_KVM: &[u8] = include_bytes!("../../../firmware/ambrosia/kvm.plist");

/// Each kext of the arm64 guest's configuration, in `Kernel > Add` order, Graft before the two
/// kexts that run on it: its bundle's name in the bundle and its `CFBundleIdentifier`, by which
/// it is found in the set. They are `firmware/ambrosia/tools/stage-config.py`'s `KEXTS`.
const AMBROSIA_KEXTS: [(&str, &str); 3] = [
    ("Graft.kext", "ai.aneesiqbal.graft"),
    ("ReimsGraphics.kext", "ai.aneesiqbal.ReimsGraphics"),
    ("RgfxGate.kext", "ai.aneesiqbal.RgfxGate"),
];

/// How many displays' nodes the bundle names (`Graphics > DisplayNode`, `reims-display0` on),
/// one for each display a device can have: stage-config.py's `DISPLAY_NODES`.
const AMBROSIA_DISPLAY_NODES: usize = 4;

/// The guest's kexts, in the order OpenCore loads them; an error where this reims-ctl was built
/// from a checkout without them.
pub fn set() -> Result<&'static [Kext]> {
    carried().context(
        "this reims-ctl was built without reims-vgpu's guest kexts (guest/prebuilt/set.toml): \
         run reims-ctl update",
    )
}

/// The set reims-ctl carries: the embedded one.
#[cfg(not(test))]
fn carried() -> Option<&'static [Kext]> {
    SET
}

/// The set reims-ctl's tests carry: one of their own, whatever the checkout holds.
#[cfg(test)]
fn carried() -> Option<&'static [Kext]> {
    Some(tests::SET)
}

/// The arm64 guest's kexts and the module reims-ctl carries: the embedded ones.
#[cfg(not(test))]
fn carried_arm64() -> (Option<&'static [Kext]>, Option<&'static [u8]>) {
    (SET_ARM64E, AMBROSIA_MODULE)
}

/// The arm64 guest's kexts and the module reims-ctl's tests carry: their own.
#[cfg(test)]
fn carried_arm64() -> (Option<&'static [Kext]>, Option<&'static [u8]>) {
    (Some(tests::SET_ARM64E), Some(b"\x7fELF module"))
}

/// The directory the embedded files are written into.
fn dir() -> PathBuf {
    paths::data_dir().join("prebuilt")
}

/// The ROM's file, written there if it is not: its `romfile`.
pub fn graphics_rom() -> Result<PathBuf> {
    rom_in(&dir(), GRAPHICS_ROM)
}

/// Where [`graphics_rom`] writes it.
pub fn graphics_rom_path() -> PathBuf {
    rom_file(&dir(), GRAPHICS_ROM)
}

/// `rom`'s file in `dir`: `ReimsGraphics-<hash>.rom`.
fn rom_file(dir: &Path, rom: &[u8]) -> PathBuf {
    content_file(dir, "ReimsGraphics", "rom", rom)
}

/// `bytes`' file in `dir`: `<stem>-<hash>.<ext>`.
fn content_file(dir: &Path, stem: &str, ext: &str, bytes: &[u8]) -> PathBuf {
    dir.join(format!("{stem}-{}.{ext}", hash([bytes])))
}

/// An arm64 VM's configuration bundle for Ambrosia, written there if it is not: QEMU loads it
/// into the guest's memory, where Ambrosia reads it. It is ours, with the VM's own patches where
/// it has them, its `vm_config` ([`crate::vm::Vm::ambrosia_config`]), and where the VM runs on
/// KVM (`kvm`), [`AMBROSIA_KVM`]'s after them.
pub fn ambrosia_bundle(vm_config: &Path, kvm: bool) -> Result<PathBuf> {
    let (set, module) = carried_arm64();
    let set = set.context(
        "this reims-ctl was built without reims-vgpu's arm64 guest kexts \
         (guest/prebuilt-arm64e/set.toml): run reims-ctl update",
    )?;
    let module = module.context(
        "this reims-ctl was built without the device's driver module for Ambrosia, which clang \
         builds as reims-ctl is built: install clang, then run reims-ctl update",
    )?;
    let bundle = config_bundle(set, module, patches(vm_config, kvm)?)?;
    file_in(&dir(), "Config", "cpio", &bundle)
}

/// Whether this reims-ctl carries what an arm64 VM's bundle is made of: the arm64 guest's kexts
/// and the device's module for Ambrosia.
pub fn carries_arm64() -> bool {
    matches!(carried_arm64(), (Some(_), Some(_)))
}

/// Where [`ambrosia_bundle`] writes it, or None where this reims-ctl can't make one.
pub fn ambrosia_bundle_path(vm_config: &Path, kvm: bool) -> Option<PathBuf> {
    let (Some(set), Some(module)) = carried_arm64() else {
        return None;
    };
    let bundle = config_bundle(set, module, patches(vm_config, kvm).ok()?).ok()?;
    Some(content_file(&dir(), "Config", "cpio", &bundle))
}

/// The `Kernel > Patch` entries of an arm64 VM's bundle: its own ([`vm_patches`]), then, where
/// it runs on KVM, [`kvm_patches`].
fn patches(vm_config: &Path, kvm: bool) -> Result<Vec<plist::Value>> {
    let mut patches = vm_patches(vm_config)?;
    if kvm {
        patches.extend(kvm_patches()?);
    }
    Ok(patches)
}

/// [`AMBROSIA_KVM`]'s `Kernel > Patch` entries.
fn kvm_patches() -> Result<Vec<plist::Value>> {
    let config = plist::Value::from_reader(std::io::Cursor::new(AMBROSIA_KVM))
        .context("read firmware/ambrosia/kvm.plist")?;
    match config
        .as_dictionary()
        .and_then(|d| d.get("Kernel"))
        .and_then(plist::Value::as_dictionary)
        .and_then(|k| k.get("Patch"))
    {
        Some(plist::Value::Array(patches)) => Ok(patches.clone()),
        _ => bail!("firmware/ambrosia/kvm.plist has no Kernel > Patch array"),
    }
}

/// The `Kernel > Patch` entries of a VM's own configuration, `vm_config`
/// ([`crate::vm::Vm::ambrosia_config`]), which the VM's bundle takes after its own: none where
/// the VM has none. reims-restore writes it with those entries alone, and one with anything else
/// is refused, rather than booted without it.
fn vm_patches(vm_config: &Path) -> Result<Vec<plist::Value>> {
    let bytes = match std::fs::read(vm_config) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e).with_context(|| format!("read {}", vm_config.display())),
    };
    let what = || vm_config.display().to_string();
    let plist = newc_config_plist(&bytes).with_context(what)?;
    let config = plist::Value::from_reader(std::io::Cursor::new(plist))
        .with_context(|| format!("{}'s config.plist", what()))?;
    let only = |v: &plist::Value, key: &str| -> Result<plist::Value> {
        let d = v
            .as_dictionary()
            .with_context(|| format!("{}: its configuration is not a dictionary", what()))?;
        if let Some(k) = d.keys().find(|k| *k != key) {
            bail!(
                "{}: its configuration has {k}, and reims-ctl takes only its Kernel > Patch",
                what()
            );
        }
        d.get(key)
            .cloned()
            .with_context(|| format!("{}: its configuration has no Kernel > Patch", what()))
    };
    match only(&only(&config, "Kernel")?, "Patch")? {
        plist::Value::Array(patches) => Ok(patches),
        _ => bail!("{}: its Kernel > Patch is not an array", what()),
    }
}

/// The data of the first member of the "newc" archive `bytes`, which must be `config.plist`, as
/// in every Ambrosia configuration bundle. The header's hex digits may be of either case.
fn newc_config_plist(bytes: &[u8]) -> Result<&[u8]> {
    let header = bytes
        .get(..110)
        .and_then(|h| std::str::from_utf8(h).ok())
        .filter(|h| h.starts_with("070701"))
        .context("not a configuration bundle: no newc header at its start")?;
    let field = |i: usize| {
        usize::from_str_radix(&header[6 + 8 * i..14 + 8 * i], 16)
            .context("not a configuration bundle: a newc header field is not hex")
    };
    let (size, name_size) = (field(6)?, field(11)?);
    let name = bytes
        .get(110..110 + name_size)
        .context("not a configuration bundle: it ends in its first name")?;
    if name != b"config.plist\0" {
        bail!("not a configuration bundle: its first file is not config.plist");
    }
    let at = (110 + name_size).next_multiple_of(4);
    bytes
        .get(at..at + size)
        .context("not a configuration bundle: it ends in its config.plist")
}

/// The configuration bundle of `set` and `module`, as `make -C firmware/ambrosia config-bundle`
/// makes it: a cpio archive in the "newc" format, Ambrosia's, of `config.plist`, which is
/// [`AMBROSIA_CONFIG`] with a `Kernel > Add` entry for each kext and the displays' nodes, then
/// the module as `Drivers/ReimsGraphics.elf`, then each kext's files under
/// `Kexts/<its name>.kext/`, all of them in byte order of their paths. A VM's own `patches`
/// ([`vm_patches`]) are its `Kernel > Patch`, where it has any.
fn config_bundle(set: &[Kext], module: &[u8], patches: Vec<plist::Value>) -> Result<Vec<u8>> {
    let mut config = match plist::Value::from_reader(std::io::Cursor::new(AMBROSIA_CONFIG))
        .context("read firmware/ambrosia/config.plist")?
    {
        plist::Value::Dictionary(d) => d,
        _ => bail!("firmware/ambrosia/config.plist is not a dictionary"),
    };
    let mut kexts = Vec::new();
    for (name, identifier) in AMBROSIA_KEXTS {
        let kext = set
            .iter()
            .find(|k| bundle_identifier(k).as_deref() == Some(identifier))
            .with_context(|| format!("the arm64 guest's kexts have no {identifier} for {name}"))?;
        kexts.push((name, kext));
    }
    let section = |config: &mut plist::Dictionary, key: &str| -> Result<plist::Dictionary> {
        match config.remove(key) {
            None => Ok(plist::Dictionary::new()),
            Some(plist::Value::Dictionary(d)) => Ok(d),
            Some(_) => bail!("firmware/ambrosia/config.plist's {key} is not a dictionary"),
        }
    };
    let mut kernel = section(&mut config, "Kernel")?;
    let add = kexts
        .iter()
        .map(|(name, _)| {
            let mut entry = plist::Dictionary::new();
            entry.insert("BundlePath".into(), format!("Kexts/{name}").into());
            plist::Value::Dictionary(entry)
        })
        .collect();
    kernel.insert("Add".into(), plist::Value::Array(add));
    if !patches.is_empty() {
        kernel.insert("Patch".into(), plist::Value::Array(patches));
    }
    config.insert("Kernel".into(), plist::Value::Dictionary(kernel));
    let mut graphics = section(&mut config, "Graphics")?;
    let nodes = (0..AMBROSIA_DISPLAY_NODES)
        .map(|i| format!("reims-display{i}").into())
        .collect();
    graphics.insert("DisplayNode".into(), plist::Value::Array(nodes));
    config.insert("Graphics".into(), plist::Value::Dictionary(graphics));
    let mut plist_bytes = Vec::new();
    plist::Value::Dictionary(config)
        .to_writer_xml(&mut plist_bytes)
        .context("write the bundle's config.plist")?;

    let mut files: Vec<(String, bool, &[u8])> = Vec::new();
    for (name, kext) in &kexts {
        for (path, bytes) in kext.files {
            let executable = path.starts_with("Contents/MacOS/");
            files.push((format!("Kexts/{name}/{path}"), executable, *bytes));
        }
    }
    files.sort_by(|a, b| a.0.cmp(&b.0));
    let mut out = Vec::new();
    let mut ino = 0;
    let mut member = |out: &mut Vec<u8>, path: &str, mode: u32, data: &[u8]| -> Result<()> {
        ino += 1;
        newc_member(out, ino, mode, path, data)
    };
    member(&mut out, "config.plist", NEWC_FILE, &plist_bytes)?;
    member(
        &mut out,
        "Drivers/ReimsGraphics.elf",
        NEWC_EXECUTABLE,
        module,
    )?;
    for (path, executable, data) in files {
        let mode = if executable {
            NEWC_EXECUTABLE
        } else {
            NEWC_FILE
        };
        member(&mut out, &path, mode, data)?;
    }
    newc_member(&mut out, 0, 0, NEWC_TRAILER, &[])?;
    Ok(out)
}

/// The `CFBundleIdentifier` of `kext`'s `Contents/Info.plist`.
fn bundle_identifier(kext: &Kext) -> Option<String> {
    let (_, info) = kext
        .files
        .iter()
        .find(|(p, _)| *p == "Contents/Info.plist")?;
    let info = plist::Value::from_reader(std::io::Cursor::new(*info)).ok()?;
    info.as_dictionary()?
        .get("CFBundleIdentifier")?
        .as_string()
        .map(str::to_string)
}

/// A regular file's mode in the bundle, `rw-r--r--`, and an executable's, `rwxr-xr-x`: the
/// modes Ambrosia's own bundle writer gives them.
const NEWC_FILE: u32 = 0o100_644;
const NEWC_EXECUTABLE: u32 = 0o100_755;
/// The name of the member that ends a "newc" archive.
const NEWC_TRAILER: &str = "TRAILER!!!";

/// Appends to `out` a "newc" member: a 110-byte header of the magic and 13 fields of 8 lowercase
/// hex digits (inode, mode, uid, gid, links, mtime, size, the device numbers, the name's size
/// with its NUL, a checksum of 0), its NUL-terminated name padded to 4 bytes, then its data
/// padded to 4 bytes. Nothing of the host goes in: owners, times and devices are 0.
fn newc_member(out: &mut Vec<u8>, ino: u32, mode: u32, path: &str, data: &[u8]) -> Result<()> {
    let size = u32::try_from(data.len()).with_context(|| format!("{path} is over 4 GiB"))?;
    let name_size = u32::try_from(path.len() + 1).context("a path over 4 GiB")?;
    let fields = [ino, mode, 0, 0, 1, 0, size, 0, 0, 0, 0, name_size, 0];
    out.extend_from_slice(b"070701");
    for f in fields {
        out.extend_from_slice(format!("{f:08x}").as_bytes());
    }
    out.extend_from_slice(path.as_bytes());
    out.push(0);
    out.resize(out.len().next_multiple_of(4), 0);
    out.extend_from_slice(data);
    out.resize(out.len().next_multiple_of(4), 0);
    Ok(())
}

/// The bundles of the guest's kexts, written there if they are not, in the set's order.
pub fn guest_kexts() -> Result<Vec<PathBuf>> {
    kexts_in(&dir(), set()?)
}

/// 16 hex digits of the FNV-1a hash of `parts`, each its length then its bytes: a name for
/// content, whose bytes are compared before it is used.
fn hash<'a>(parts: impl IntoIterator<Item = &'a [u8]>) -> String {
    let mut h = 0xcbf2_9ce4_8422_2325_u64;
    let mut eat = |bytes: &[u8]| {
        for b in bytes {
            h = (h ^ u64::from(*b)).wrapping_mul(0x0100_0000_01b3);
        }
    };
    for p in parts {
        eat(&(p.len() as u64).to_le_bytes());
        eat(p);
    }
    format!("{h:016x}")
}

/// A name for a file or directory written beside its place, apart from every other writer's.
fn beside(place: &Path) -> PathBuf {
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let mut name = place.file_name().unwrap_or_default().to_os_string();
    name.push(format!(".{}-{t}.tmp", std::process::id()));
    place.with_file_name(name)
}

/// `rom` in `dir`, as [`rom_file`] names it.
fn rom_in(dir: &Path, rom: &[u8]) -> Result<PathBuf> {
    file_in(dir, "ReimsGraphics", "rom", rom)
}

/// `bytes` in `dir`, as [`content_file`] names them.
fn file_in(dir: &Path, stem: &str, ext: &str, bytes: &[u8]) -> Result<PathBuf> {
    let file = content_file(dir, stem, ext, bytes);
    if std::fs::read(&file).is_ok_and(|b| b == bytes) {
        return Ok(file);
    }
    std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    let tmp = beside(&file);
    let written = std::fs::write(&tmp, bytes).and_then(|()| std::fs::rename(&tmp, &file));
    if let Err(e) = written {
        let _ = std::fs::remove_file(&tmp);
        return Err(e).with_context(|| format!("write {}", file.display()));
    }
    Ok(file)
}

/// Whether `bundle` holds `kext`'s files with their bytes (other files there aside).
fn holds(bundle: &Path, kext: &Kext) -> bool {
    kext.files
        .iter()
        .all(|(p, b)| std::fs::read(bundle.join(p)).is_ok_and(|on| on == *b))
}

/// `set` in `dir`, as the bundles of `kexts-<hash>`.
fn kexts_in(dir: &Path, set: &[Kext]) -> Result<Vec<PathBuf>> {
    let parts = set.iter().flat_map(|k| {
        [
            k.bundle.as_bytes(),
            k.executable.unwrap_or_default().as_bytes(),
        ]
        .into_iter()
        .chain(k.files.iter().flat_map(|(p, b)| [p.as_bytes(), *b]))
    });
    let place = dir.join(format!("kexts-{}", hash(parts)));
    let bundles = |at: &Path| -> Vec<PathBuf> { set.iter().map(|k| at.join(k.bundle)).collect() };
    let whole = |at: &Path| set.iter().all(|k| holds(&at.join(k.bundle), k));
    if whole(&place) {
        return Ok(bundles(&place));
    }
    let tmp = beside(&place);
    let write = || -> std::io::Result<()> {
        for k in set {
            for (p, b) in k.files {
                let file = tmp.join(k.bundle).join(p);
                std::fs::create_dir_all(file.parent().unwrap_or(&tmp))?;
                std::fs::write(&file, b)?;
            }
        }
        Ok(())
    };
    if let Err(e) = write() {
        let _ = std::fs::remove_dir_all(&tmp);
        return Err(e).with_context(|| format!("write {}", place.display()));
    }
    // One that is not whole, left by an earlier write that stopped, goes aside first; one a writer
    // at once with this one renamed into place first is whole, and stays.
    if place.exists() && !whole(&place) {
        let old = beside(&place);
        if std::fs::rename(&place, &old).is_ok() {
            let _ = std::fs::remove_dir_all(&old);
        }
    }
    let renamed = std::fs::rename(&tmp, &place);
    let _ = std::fs::remove_dir_all(&tmp);
    if !whole(&place) {
        match renamed {
            Err(e) => return Err(e).with_context(|| format!("write {}", place.display())),
            Ok(()) => bail!("{} changed as it was written", place.display()),
        }
    }
    Ok(bundles(&place))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The set the tests carry ([`carried`]).
    pub static SET: &[Kext] = &[
        Kext {
            bundle: "A.kext",
            executable: Some("Contents/MacOS/A"),
            min_kernel: Some("25.4.0"),
            max_kernel: Some("25.99.99"),
            files: &[
                ("Contents/Info.plist", b"<plist>a</plist>"),
                ("Contents/MacOS/A", b"\xcf\xfa\xed\xfe a"),
            ],
        },
        Kext {
            bundle: "B.kext",
            executable: None,
            min_kernel: None,
            max_kernel: None,
            files: &[("Contents/Info.plist", b"<plist>b</plist>")],
        },
    ];

    /// An Info.plist that names `$id` as its bundle's identifier.
    macro_rules! info {
        ($id:literal) => {
            concat!(
                "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<plist version=\"1.0\"><dict>",
                "<key>CFBundleIdentifier</key><string>",
                $id,
                "</string></dict></plist>\n"
            )
            .as_bytes()
        };
    }

    /// The arm64 set the tests carry ([`carried_arm64`]): out of the bundle's order, and Graft
    /// under its arm64e build's name, as `guest/prebuilt-arm64e` has it.
    pub static SET_ARM64E: &[Kext] = &[
        Kext {
            bundle: "RgfxGate.kext",
            executable: Some("Contents/MacOS/RgfxGate"),
            min_kernel: Some("23.0.0"),
            max_kernel: None,
            files: &[
                ("Contents/Info.plist", info!("ai.aneesiqbal.RgfxGate")),
                ("Contents/MacOS/RgfxGate", b"rgfxgate"),
            ],
        },
        Kext {
            bundle: "Graft-arm64e.kext",
            executable: Some("Contents/MacOS/Graft"),
            min_kernel: Some("23.0.0"),
            max_kernel: None,
            files: &[
                ("Contents/MacOS/Graft", b"graft"),
                ("Contents/Info.plist", info!("ai.aneesiqbal.graft")),
            ],
        },
        Kext {
            bundle: "ReimsGraphics.kext",
            executable: Some("Contents/MacOS/ReimsGraphics"),
            min_kernel: Some("23.0.0"),
            max_kernel: None,
            files: &[
                ("Contents/Info.plist", info!("ai.aneesiqbal.ReimsGraphics")),
                ("Contents/MacOS/ReimsGraphics", b"reims graphics"),
            ],
        },
    ];

    /// The members of a "newc" archive, its trailer's aside, as Ambrosia reads them: each one's
    /// name, mode and data. Every header field but those is checked: 0, or 1 links.
    fn members(a: &[u8]) -> Vec<(String, u32, Vec<u8>)> {
        let mut out = Vec::new();
        let mut at = 0;
        let mut ino = 0;
        loop {
            let h = std::str::from_utf8(&a[at..at + 110]).unwrap();
            assert_eq!(&h[..6], "070701");
            let field = |i: usize| u32::from_str_radix(&h[6 + 8 * i..14 + 8 * i], 16).unwrap();
            assert!(
                h[6..]
                    .bytes()
                    .all(|b| b.is_ascii_digit() || b.is_ascii_lowercase())
            );
            let (mode, size, name_size) = (field(1), field(6) as usize, field(11) as usize);
            for i in [2, 3, 5, 7, 8, 9, 10, 12] {
                assert_eq!(field(i), 0, "field {i} of {h}");
            }
            assert_eq!(field(4), 1);
            let name = &a[at + 110..at + 110 + name_size];
            assert_eq!(name.last(), Some(&0));
            let name = std::str::from_utf8(&name[..name_size - 1])
                .unwrap()
                .to_string();
            let data_at = (at + 110 + name_size).next_multiple_of(4);
            if name == NEWC_TRAILER {
                assert_eq!((field(0), mode, size), (0, 0, 0));
                assert_eq!(a.len(), data_at);
                return out;
            }
            ino += 1;
            assert_eq!(field(0), ino);
            out.push((name, mode, a[data_at..data_at + size].to_vec()));
            at = (data_at + size).next_multiple_of(4);
        }
    }

    /// A member's header and padding are Ambrosia's own writer's, field by field
    /// (ambrosia-bundle's newc.rs, `header_fields`).
    #[test]
    fn a_bundle_member_is_ambrosias_newc() {
        let mut a = Vec::new();
        newc_member(&mut a, 1, NEWC_EXECUTABLE, "config.plist", b"abc").unwrap();
        newc_member(&mut a, 0, 0, NEWC_TRAILER, &[]).unwrap();
        let h = std::str::from_utf8(&a[..110]).unwrap();
        let field = |i: usize| &h[6 + 8 * i..14 + 8 * i];
        assert_eq!(&h[..6], "070701");
        assert_eq!(field(0), "00000001");
        assert_eq!(field(1), "000081ed");
        assert_eq!(field(6), "00000003");
        assert_eq!(field(11), "0000000d");
        assert_eq!(&a[110..124], b"config.plist\0\0");
        assert_eq!(&a[124..128], b"abc\0");
        assert_eq!(&a[238..249], b"TRAILER!!!\0");
        assert_eq!(a.len(), 252);
    }

    /// The bundle holds config.plist first, with an entry for each kext in Graft's, ReimsGraphics'
    /// and RgfxGate's order and the four displays' nodes, then the module, then the kexts' files
    /// in byte order under their roles' names, executables 0755.
    #[test]
    fn the_ambrosia_bundle_holds_the_config_module_and_kexts() {
        let a = config_bundle(SET_ARM64E, b"\x7fELF module", Vec::new()).unwrap();
        let m = members(&a);
        let names: Vec<_> = m.iter().map(|(n, ..)| n.as_str()).collect();
        assert_eq!(
            names,
            [
                "config.plist",
                "Drivers/ReimsGraphics.elf",
                "Kexts/Graft.kext/Contents/Info.plist",
                "Kexts/Graft.kext/Contents/MacOS/Graft",
                "Kexts/ReimsGraphics.kext/Contents/Info.plist",
                "Kexts/ReimsGraphics.kext/Contents/MacOS/ReimsGraphics",
                "Kexts/RgfxGate.kext/Contents/Info.plist",
                "Kexts/RgfxGate.kext/Contents/MacOS/RgfxGate",
            ]
        );
        let modes: Vec<_> = m.iter().map(|(_, mode, _)| *mode).collect();
        assert_eq!(
            modes,
            [
                0o100_644, 0o100_755, 0o100_644, 0o100_755, 0o100_644, 0o100_755, 0o100_644,
                0o100_755
            ]
        );
        assert_eq!(m[1].2, b"\x7fELF module");
        assert_eq!(m[3].2, b"graft");
        let config = plist::Value::from_reader(std::io::Cursor::new(&m[0].2)).unwrap();
        let config = config.as_dictionary().unwrap();
        let section = |k: &str| config.get(k).unwrap().as_dictionary().unwrap();
        let strings = |v: &plist::Value| -> Vec<String> {
            v.as_array()
                .unwrap()
                .iter()
                .map(|s| s.as_string().unwrap().to_string())
                .collect()
        };
        let add: Vec<_> = section("Kernel")["Add"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| {
                e.as_dictionary().unwrap()["BundlePath"]
                    .as_string()
                    .unwrap()
            })
            .collect();
        assert_eq!(
            add,
            [
                "Kexts/Graft.kext",
                "Kexts/ReimsGraphics.kext",
                "Kexts/RgfxGate.kext"
            ]
        );
        assert_eq!(
            strings(&section("Kernel")["ExportVtableDiscriminators"]).len(),
            3
        );
        assert_eq!(
            section("Graphics")["Driver"].as_string(),
            Some("Drivers/ReimsGraphics.elf")
        );
        assert_eq!(
            strings(&section("Graphics")["DisplayNode"]),
            [
                "reims-display0",
                "reims-display1",
                "reims-display2",
                "reims-display3"
            ]
        );
        // The same files make the same bytes, written once under their hash.
        assert_eq!(
            config_bundle(SET_ARM64E, b"\x7fELF module", Vec::new()).unwrap(),
            a
        );
        let d = scratch("bundle");
        let f = file_in(&d, "Config", "cpio", &a).unwrap();
        assert_eq!(f, content_file(&d, "Config", "cpio", &a));
        assert_eq!(std::fs::read(&f).unwrap(), a);
        std::fs::remove_dir_all(&d).unwrap();
    }

    /// A VM reims-restore restored boots with its own patches: the `Kernel > Patch` of the
    /// `config.cpio` reims-restore writes beside its disks is the bundle's, after the rest of our
    /// configuration. A VM without one gets none, and one whose configuration holds anything
    /// else is refused.
    #[test]
    fn a_restored_vms_patches_are_its_bundles() {
        let d = scratch("vm-config");
        std::fs::create_dir_all(&d).unwrap();
        let file = d.join("config.cpio");
        assert!(vm_patches(&file).unwrap().is_empty());
        let mut patch = plist::Dictionary::new();
        patch.insert("Identifier".into(), "com.example.kext".into());
        patch.insert("Find".into(), plist::Value::Data(vec![1, 2, 3]));
        patch.insert("Replace".into(), plist::Value::Data(vec![4, 5, 6]));
        let patches = vec![plist::Value::Dictionary(patch)];
        let mut kernel = plist::Dictionary::new();
        kernel.insert("Patch".into(), plist::Value::Array(patches.clone()));
        let mut config = plist::Dictionary::new();
        config.insert("Kernel".into(), plist::Value::Dictionary(kernel));
        std::fs::write(
            &file,
            reims_restore::anchors::config_bundle(&config).unwrap(),
        )
        .unwrap();
        assert_eq!(vm_patches(&file).unwrap(), patches);
        let bundle = members(&config_bundle(SET_ARM64E, b"m", vm_patches(&file).unwrap()).unwrap());
        let plist = plist::Value::from_reader(std::io::Cursor::new(&bundle[0].2)).unwrap();
        let kernel = plist.as_dictionary().unwrap()["Kernel"]
            .as_dictionary()
            .unwrap();
        assert_eq!(kernel["Patch"], plist::Value::Array(patches));
        assert!(kernel.contains_key("Add") && kernel.contains_key("ExportVtableDiscriminators"));

        config.insert(
            "Misc".into(),
            plist::Value::Dictionary(plist::Dictionary::new()),
        );
        std::fs::write(
            &file,
            reims_restore::anchors::config_bundle(&config).unwrap(),
        )
        .unwrap();
        let e = vm_patches(&file).unwrap_err();
        assert!(e.to_string().contains("has Misc"), "{e:#}");
        std::fs::write(&file, b"not a bundle").unwrap();
        assert!(vm_patches(&file).is_err());
        std::fs::remove_dir_all(&d).unwrap();
    }

    /// A VM on KVM boots with kvm.plist's patch for macOS's GIC setup, after the VM's own
    /// patches; on HVF and TCG its bundle has none of it.
    #[test]
    fn a_vm_on_kvm_boots_with_the_gic_patch() {
        let gic = kvm_patches().unwrap();
        assert_eq!(gic.len(), 1);
        let p = gic[0].as_dictionary().unwrap();
        assert_eq!(p["Identifier"].as_string(), Some("com.apple.kernel"));
        assert_eq!(
            p["Find"].as_data(),
            Some(
                &[
                    0x09, 0xc0, 0xaf, 0x12, 0x09, 0x0d, 0x08, 0xb8, 0x09, 0x00, 0xa1, 0x52, 0x09,
                    0x81, 0x00, 0xb9
                ][..]
            )
        );
        assert_eq!(
            p["Replace"].as_data(),
            Some(
                &[
                    0x09, 0xc0, 0xaf, 0x12, 0x09, 0x81, 0x00, 0xb9, 0x09, 0x00, 0xa1, 0x52, 0x09,
                    0x01, 0x01, 0xb9
                ][..]
            )
        );
        assert_eq!(p["Expect"].as_signed_integer(), Some(1));

        let d = scratch("kvm-patches");
        std::fs::create_dir_all(&d).unwrap();
        let none = d.join("config.cpio");
        assert!(patches(&none, false).unwrap().is_empty());
        assert_eq!(patches(&none, true).unwrap(), gic);

        let mut own = plist::Dictionary::new();
        own.insert("Identifier".into(), "com.example.kext".into());
        own.insert("Find".into(), plist::Value::Data(vec![1, 2, 3]));
        own.insert("Replace".into(), plist::Value::Data(vec![4, 5, 6]));
        let mut kernel = plist::Dictionary::new();
        kernel.insert(
            "Patch".into(),
            plist::Value::Array(vec![plist::Value::Dictionary(own.clone())]),
        );
        let mut config = plist::Dictionary::new();
        config.insert("Kernel".into(), plist::Value::Dictionary(kernel));
        let file = d.join("restored.cpio");
        std::fs::write(
            &file,
            reims_restore::anchors::config_bundle(&config).unwrap(),
        )
        .unwrap();
        let mut want = vec![plist::Value::Dictionary(own)];
        assert_eq!(patches(&file, false).unwrap(), want);
        want.extend(gic);
        assert_eq!(patches(&file, true).unwrap(), want);
        let bundle =
            members(&config_bundle(SET_ARM64E, b"m", patches(&file, true).unwrap()).unwrap());
        let plist = plist::Value::from_reader(std::io::Cursor::new(&bundle[0].2)).unwrap();
        let kernel = plist.as_dictionary().unwrap()["Kernel"]
            .as_dictionary()
            .unwrap();
        assert_eq!(kernel["Patch"], plist::Value::Array(want));
        std::fs::remove_dir_all(&d).unwrap();
    }

    /// A set without a kext of a role makes no bundle, and says which.
    #[test]
    fn a_set_without_a_role_makes_no_bundle() {
        let e = config_bundle(&SET_ARM64E[..2], b"m", Vec::new()).unwrap_err();
        assert!(
            e.to_string().contains("ai.aneesiqbal.ReimsGraphics"),
            "{e:#}"
        );
    }

    /// A reims-ctl for Windows, which runs no arm64 VM, carries neither the arm64 set nor the
    /// module (build.rs).
    #[cfg(windows)]
    #[test]
    fn a_reims_ctl_for_windows_carries_no_arm64_guest() {
        assert!(super::SET_ARM64E.is_none());
        assert!(AMBROSIA_MODULE.is_none());
    }

    /// The module build.rs built is an aarch64 ELF static PIE, the module format Ambrosia loads:
    /// a checkout whose clang can't build it fails here, not at an arm64 VM's start.
    #[cfg(not(windows))]
    #[test]
    fn the_module_is_built() {
        let elf = AMBROSIA_MODULE.expect(
            "build.rs built no module for Ambrosia: its warning says why (is clang installed?)",
        );
        assert_eq!(&elf[..4], b"\x7fELF");
        assert_eq!(elf[4], 2, "ELFCLASS64");
        assert_eq!(u16::from_le_bytes([elf[16], elf[17]]), 3, "ET_DYN");
        assert_eq!(u16::from_le_bytes([elf[18], elf[19]]), 183, "EM_AARCH64");
    }

    /// The bundle of the embedded set and module is the one `make -C firmware/ambrosia
    /// config-bundle` makes of them: the same files, in the same order, with the same bytes, and
    /// a config.plist of the same values (tools/stage-config.py writes it, with Python's plist
    /// writer).
    #[cfg(not(windows))]
    #[test]
    fn the_embedded_bundle_is_the_makefiles() {
        let set = super::SET_ARM64E.expect("guest/prebuilt-arm64e/set.toml");
        let module = AMBROSIA_MODULE.expect("the module (the_module_is_built)");
        let ours = members(&config_bundle(set, module, Vec::new()).unwrap());
        let d = scratch("stage-config");
        let bundles = kexts_in(&d.join("set"), set).unwrap();
        std::fs::write(d.join("ReimsGraphics.elf"), module).unwrap();
        let dir = std::env::var_os("CARGO_MANIFEST_DIR").unwrap();
        let ambrosia = Path::new(&dir).join("../../firmware/ambrosia");
        let mut cmd = std::process::Command::new("python3");
        cmd.args(["-I", "-B"])
            .arg(ambrosia.join("tools/stage-config.py"))
            .arg(d.join("out"))
            .arg(ambrosia.join("config.plist"))
            .arg(d.join("ReimsGraphics.elf"));
        for (_, identifier) in AMBROSIA_KEXTS {
            let i = set
                .iter()
                .position(|k| bundle_identifier(k).as_deref() == Some(identifier))
                .unwrap();
            cmd.arg(&bundles[i]);
        }
        let out = cmd.output().unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        // The Makefile's order: config.plist, the module, then `find Kexts -type f | LC_ALL=C
        // sort`.
        fn files(root: &Path, dir: &Path, out: &mut Vec<String>) {
            for e in std::fs::read_dir(dir).unwrap() {
                let p = e.unwrap().path();
                if p.is_dir() {
                    files(root, &p, out);
                } else {
                    let rel: Vec<_> = p
                        .strip_prefix(root)
                        .unwrap()
                        .iter()
                        .map(|n| n.to_str().unwrap().to_string())
                        .collect();
                    out.push(rel.join("/"));
                }
            }
        }
        let staged = d.join("out");
        let mut kexts = Vec::new();
        files(&staged, &staged.join("Kexts"), &mut kexts);
        kexts.sort();
        let theirs: Vec<_> = ["config.plist", "Drivers/ReimsGraphics.elf"]
            .into_iter()
            .map(str::to_string)
            .chain(kexts)
            .collect();
        let ours_names: Vec<_> = ours.iter().map(|(n, ..)| n.clone()).collect();
        assert_eq!(ours_names, theirs);
        for (name, _, data) in &ours[1..] {
            assert_eq!(std::fs::read(staged.join(name)).unwrap(), *data, "{name}");
        }
        let parse = |b: &[u8]| plist::Value::from_reader(std::io::Cursor::new(b)).unwrap();
        assert_eq!(
            parse(&ours[0].2),
            parse(&std::fs::read(staged.join("config.plist")).unwrap())
        );
        std::fs::remove_dir_all(&d).unwrap();
    }

    fn scratch(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("reims-ctl-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    /// The ROM is written once under its content's name, written again where its file's bytes
    /// changed, and a ROM of other bytes gets a file of its own.
    #[test]
    fn the_rom_is_written_by_its_content() {
        let d = scratch("rom");
        let f = rom_in(&d, b"rom one").unwrap();
        assert_eq!(std::fs::read(&f).unwrap(), b"rom one");
        assert_eq!(rom_in(&d, b"rom one").unwrap(), f);
        std::fs::write(&f, b"rom on").unwrap();
        assert_eq!(rom_in(&d, b"rom one").unwrap(), f);
        assert_eq!(std::fs::read(&f).unwrap(), b"rom one");
        let other = rom_in(&d, b"rom two").unwrap();
        assert_ne!(other, f);
        assert!(f.is_file());
        // Nothing is left beside them.
        assert_eq!(std::fs::read_dir(&d).unwrap().count(), 2);
        std::fs::remove_dir_all(&d).unwrap();
    }

    /// The set is written whole, its bundles in its order, and written again where a file of it
    /// is missing or changed.
    #[test]
    fn the_set_is_written_whole_in_its_order() {
        let d = scratch("kexts");
        let bundles = kexts_in(&d, SET).unwrap();
        let names: Vec<_> = bundles.iter().map(|b| b.file_name().unwrap()).collect();
        assert_eq!(names, ["A.kext", "B.kext"]);
        assert_eq!(
            std::fs::read(bundles[0].join("Contents/MacOS/A")).unwrap(),
            b"\xcf\xfa\xed\xfe a"
        );
        assert_eq!(SET[0].executable_bytes(), Some(&b"\xcf\xfa\xed\xfe a"[..]));
        assert_eq!(SET[1].executable_bytes(), None);
        assert_eq!(kexts_in(&d, SET).unwrap(), bundles);
        std::fs::remove_file(bundles[1].join("Contents/Info.plist")).unwrap();
        std::fs::write(bundles[0].join("Contents/MacOS/A"), b"stale").unwrap();
        assert_eq!(kexts_in(&d, SET).unwrap(), bundles);
        assert!(holds(&bundles[0], &SET[0]) && holds(&bundles[1], &SET[1]));
        assert_eq!(
            std::fs::read_dir(&d).unwrap().count(),
            1,
            "nothing beside it"
        );
        std::fs::remove_dir_all(&d).unwrap();
    }

    /// The set embedded, where the checkout has one, is whole: each kext's executable is one of
    /// its files, and its bounds are X.Y.Z.
    #[test]
    fn the_embedded_set_is_whole() {
        let Some(set) = super::SET else { return };
        assert!(!set.is_empty());
        for k in set {
            assert!(k.bundle.ends_with(".kext"), "{k:?}");
            assert!(
                k.executable.is_none() || k.executable_bytes().is_some(),
                "{}: its executable is not in the bundle",
                k.bundle
            );
            assert!(
                k.files.iter().any(|(p, _)| *p == "Contents/Info.plist"),
                "{}",
                k.bundle
            );
        }
    }

    /// The ROM embedded is the one committed, which firmware/prebuilt/ReimsGraphics.toml
    /// records, and the record's sources are the tree's: a change to what the ROM is built
    /// from without `make -C firmware prebuilt` fails here (firmware/prebuilt.sh --check).
    #[test]
    fn the_prebuilt_rom_is_built_from_the_trees_sources() {
        // Read at run time, as install.rs's tests read it, so the test is the same binary in
        // every worktree.
        let dir = std::env::var_os("CARGO_MANIFEST_DIR")
            .expect("CARGO_MANIFEST_DIR: run the tests with cargo");
        let firmware = Path::new(&dir).join("../../firmware");
        assert_eq!(
            std::fs::read(firmware.join("prebuilt/ReimsGraphics.rom")).unwrap(),
            GRAPHICS_ROM
        );
        let out = std::process::Command::new("sh")
            .arg(firmware.join("prebuilt.sh"))
            .arg("--check")
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
}
