//! What reims-ctl carries for reims-vgpu's own GPU device ([`crate::vm::Graphics::ReimsVgpu`]),
//! built into its binary from the repository:
//! - the device's option ROM, `firmware/prebuilt/ReimsGraphics.rom`, which QEMU's
//!   `reims-graphics-pci` has no default for: its `romfile` ([`graphics_rom`]);
//! - the guest's kexts, the set `guest/prebuilt/set.toml` lists, which build.rs embeds in the
//!   set's order, and which reims-ctl adds to the OpenCore image of a VM on reims-vgpu
//!   ([`guest_kexts`]): OpenCore loads them in that order.
//!
//! Embedded, they are the ones of the commit reims-ctl was built from, however
//! scripts/install-ctl.sh installed it (`cargo install --path`, on macOS and on Linux, with or
//! without the QEMU of that commit), and need nothing installed beside them.
//!
//! Each is written where it is used, into the data directory's `prebuilt`, under a name of its
//! content's hash: the ROM where a VM's QEMU command is made, the set where reims-util makes an
//! image with it. One already there is used as it is where its bytes are the embedded ones, and
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

// The set build.rs embeds: `SET`, None where the checkout it was built from had none.
include!(concat!(env!("OUT_DIR"), "/set.rs"));

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

/// A Darwin version bound, X.Y.Z, as its major, minor and patch.
fn version(v: &str) -> Option<(u32, u32, u32)> {
    let mut p = v.trim().split('.').map(|n| n.parse::<u32>().ok());
    let v = (p.next()??, p.next()??, p.next()??);
    p.next().is_none().then_some(v)
}

/// The Darwin versions, major and minor, that OpenCore loads every kext of the set on: the
/// latest of their MinKernel and the earliest of their MaxKernel. None where reims-ctl carries
/// no set, or the bounds leave none.
pub fn bounds() -> Option<((u32, u32), (u32, u32))> {
    let (mut min, mut max) = ((0, 0, 0), (u32::MAX, u32::MAX, u32::MAX));
    for k in carried()? {
        if let Some(m) = k.min_kernel {
            min = min.max(version(m)?);
        }
        if let Some(m) = k.max_kernel {
            max = max.min(version(m)?);
        }
    }
    (min <= max).then_some(((min.0, min.1), (max.0, max.1)))
}

/// Whether OpenCore loads every kext of the set on Darwin `major`.`minor` (its patch level taken
/// as 0, as no bound names one but 99); None where reims-ctl carries no set.
pub fn loads_on((major, minor): (u32, u32)) -> Option<bool> {
    let mut loads = true;
    for k in carried()? {
        let at = (major, minor, 0);
        loads &= k.min_kernel.and_then(version).is_none_or(|m| m <= at)
            && k.max_kernel.and_then(version).is_none_or(|m| at <= m);
    }
    Some(loads)
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
    dir.join(format!("ReimsGraphics-{}.rom", hash([rom])))
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
    let file = rom_file(dir, rom);
    if std::fs::read(&file).is_ok_and(|b| b == rom) {
        return Ok(file);
    }
    std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    let tmp = beside(&file);
    let written = std::fs::write(&tmp, rom).and_then(|()| std::fs::rename(&tmp, &file));
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

    /// The set loads on the Darwin versions its bounds cover, a whole Darwin version for a
    /// bound of 99: the tests' set, Darwin 25.4.0 to 25.99.99, on macOS 26.4 and later releases
    /// of macOS 26.
    #[test]
    fn the_set_loads_within_its_bounds() {
        assert_eq!(bounds(), Some(((25, 4), (25, 99))));
        assert_eq!(loads_on((25, 4)), Some(true));
        assert_eq!(loads_on((25, 99)), Some(true));
        assert_eq!(loads_on((25, 3)), Some(false));
        assert_eq!(loads_on((24, 6)), Some(false));
        assert_eq!(loads_on((26, 0)), Some(false));
        assert_eq!(version("25.4.0"), Some((25, 4, 0)));
        assert_eq!(version("25.4"), None);
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
