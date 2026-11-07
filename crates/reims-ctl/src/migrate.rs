//! What an update of reims-ctl changes in the VMs an older one made: each migration runs once, at
//! the first run of the new reims-ctl (the restart after `update`, or any later command), on every
//! VM it is for that is stopped then; a VM running then gets it at its next start
//! ([`at_start`]). Each change is told in one line a VM, and a VM it can't change stays as it was,
//! told why and how to change it by hand.
//!
//! The record is `migrations.toml` in the config directory: `done`, the last migration run, and
//! `deferred`, the directories of the VMs that were running then. A lock beside it,
//! `migrations.lock`, keeps two reims-ctl starting at once from migrating a VM twice.
//!
//! Migration 1 is for the x86-64 VMs a reims-ctl from before reims-vgpu's own GPU device made:
//! their settings name no graphics, their OpenCore image has no kexts for that device, and may
//! hold the ported drivers images no longer carry. Each moves to reims-vgpu's own device where
//! the macOS its disk holds runs the device's drivers, as a start would choose
//! ([`crate::macos::graphics_for`]), and stays on apple-gfx otherwise; its image is made again for
//! that, presenting the same Mac ([`crate::opencore::renew`]), the old one kept beside it as
//! `<image>.pre-migration-1`. A VM whose settings name graphics had them set by its user, or by
//! a reims-ctl that knew of the device, and keeps them.
//!
//! Migration 2 is for the x86-64 VMs whose OpenCore image still loads the ported drivers
//! ([`PORTED`]), whatever graphics their settings name: an image reims-util made before
//! reims-ctl's graphics option, which migration 1 left be where the VM's settings named graphics.
//! Each image is made again for the graphics the VM has, presenting the same Mac, the old one
//! kept as `<image>.pre-migration-2`; one whose settings name no graphics is migration 1's case,
//! and gets its plan.

use crate::macos::{self, Macos};
use crate::opencore::{self, Config};
use crate::vm::{Graphics, Registry, Vm};
use crate::{host::Profile, paths, runtime};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};

/// The migrations this reims-ctl carries: 1 to this.
pub const CURRENT: u32 = 2;

/// The ported drivers older OpenCore images load, by bundle name: the seven kexts that run Apple's
/// arm64 GPU driver on x86-64, which reims-util carried until its fe9bfe1 (reims_util::REV).
const PORTED: [&str; 7] = [
    "AppleParavirtGPULoader.kext",
    "IOGPUFamilyBridge.kext",
    "IOGraphicsFamilyBridge.kext",
    "AppleGraphicsDeviceControlBridge.kext",
    "AppleParavirtIOSurface.kext",
    "AppleParavirtGPUIOGPUFamily.kext",
    "APVDisplayPort.kext",
];

#[derive(Debug, Default, PartialEq, Serialize, Deserialize)]
struct State {
    /// The last migration run.
    #[serde(default)]
    done: u32,
    /// The VMs that were running then, by their directories.
    #[serde(default)]
    deferred: Vec<PathBuf>,
}

fn state_file() -> PathBuf {
    paths::config_dir().join("migrations.toml")
}

fn load() -> State {
    std::fs::read_to_string(state_file())
        .ok()
        .and_then(|t| toml::from_str(&t).ok())
        .unwrap_or_default()
}

fn save(state: &State) -> Result<()> {
    let file = state_file();
    std::fs::create_dir_all(file.parent().unwrap())?;
    let tmp = file.with_extension("toml.tmp");
    std::fs::write(&tmp, toml::to_string_pretty(state)?)?;
    std::fs::rename(&tmp, &file).with_context(|| format!("write {}", file.display()))
}

/// The migrations' lock, held until it drops; None where it can't be taken, which migrates
/// nothing this run.
fn lock() -> Option<File> {
    let file = paths::config_dir().join("migrations.lock");
    std::fs::create_dir_all(file.parent()?).ok()?;
    let f = File::create(&file).ok()?;
    // SAFETY: flock on a descriptor this File owns.
    (unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_EX) } == 0).then_some(f)
}

/// What a migration does to a VM.
#[derive(Debug, PartialEq)]
enum Plan {
    /// Nothing: a VM the migration is not for.
    Skip,
    /// It runs: at its next start.
    Defer,
    /// Its disk can't be read, so its macOS is not known: it stays as it was.
    Unknown(String),
    /// Its image made again for its macOS: on the graphics saved in its settings, or, where its
    /// disk holds no macOS yet, on the default, with none saved; the macOS that decided.
    Image(Option<Graphics>, Option<Macos>),
    /// Its image made again for the graphics its settings name, which stay.
    Again(Graphics),
}

/// Migration 1's plan for `vm`, `running` or not, its disk's macOS read by `found`.
fn plan(vm: &Vm, running: bool, found: impl FnOnce() -> Result<Vec<Macos>>) -> Plan {
    if vm.config.profile != Profile::Q35X86_64 || vm.config.graphics.is_some() {
        return Plan::Skip;
    }
    if running {
        return Plan::Defer;
    }
    match found() {
        Err(e) => Plan::Unknown(format!("{e:#}")),
        Ok(found) => {
            let graphics = macos::graphics_for(&found);
            // The one its graphics follow: on apple-gfx, the first the drivers don't run on.
            let decided = found
                .iter()
                .find(|m| {
                    graphics == Some(Graphics::AppleGfx)
                        && m.darwin().and_then(crate::prebuilt::loads_on) == Some(false)
                })
                .or(found.first())
                .cloned();
            Plan::Image(graphics, decided)
        }
    }
}

/// Migration 2's plan for `vm`, `running` or not, from the kexts its image loads; one whose
/// settings name no graphics gets migration 1's, its disk's macOS read by `found`.
fn plan_2(
    vm: &Vm,
    running: bool,
    kexts: &[String],
    found: impl FnOnce() -> Result<Vec<Macos>>,
) -> Plan {
    let ported = kexts
        .iter()
        .any(|k| PORTED.iter().any(|p| k.eq_ignore_ascii_case(p)));
    if vm.config.profile != Profile::Q35X86_64 || !ported {
        return Plan::Skip;
    }
    match vm.config.graphics {
        None => plan(vm, running, found),
        Some(_) if running => Plan::Defer,
        Some(g) => Plan::Again(g),
    }
}

/// Migration `n`'s plan for `vm`, `running` or not.
fn plan_of(n: u32, vm: &Vm, running: bool) -> Plan {
    let found = || macos::on_disk(&vm.disk(), vm.config.disk_format);
    match n {
        1 => plan(vm, running, found),
        _ => {
            let kexts = vm
                .config
                .q35
                .as_ref()
                .map(|q| loaded(&vm.path(&q.opencore), q.opencore_format))
                .unwrap_or_default();
            plan_2(vm, running, &kexts, found)
        }
    }
}

/// Runs the migrations this reims-ctl carries that have not run, on every VM in the registry,
/// telling `say` what each did. Where none has run yet and there are VMs, the first line says the
/// update worked. A record that can't be written is told, and the migrations run again next time.
pub fn run(say: &mut dyn FnMut(&str)) {
    if load().done >= CURRENT {
        return;
    }
    let Some(_lock) = lock() else { return };
    let mut state = load();
    if state.done >= CURRENT {
        return;
    }
    let registry = match Registry::load() {
        Ok(r) => r,
        Err(e) => return say(&format!("reims-ctl's update of its VMs waits: {e:#}")),
    };
    if !registry.vms.is_empty() {
        say("reims-ctl is updated. What the update changes in the VMs an older one made:");
    }
    let mut told = false;
    for e in &registry.vms {
        let mut vm = match Vm::load(&e.dir) {
            Ok(vm) => vm,
            Err(err) => {
                say(&format!("{}: left as it was: {err:#}", e.name));
                told = true;
                continue;
            }
        };
        // A VM that waited for its start since an earlier update still has every migration to run.
        let waited = state.deferred.iter().any(|d| same_dir(d, &vm.dir));
        let first = if waited { 1 } else { state.done + 1 };
        let running = runtime::running(&vm);
        let mut defer = false;
        for n in first..=CURRENT {
            let p = plan_of(n, &vm, running);
            defer = p == Plan::Defer;
            if let Some(line) = carry_out(&mut vm, p, n) {
                say(&line);
                told = true;
            }
            if defer {
                break;
            }
        }
        if defer && !waited {
            state.deferred.push(vm.dir.clone());
        } else if !running {
            state.deferred.retain(|d| !same_dir(d, &vm.dir));
        }
    }
    if !registry.vms.is_empty() && !told {
        say("nothing: each VM is as this reims-ctl makes them.");
    }
    state.done = CURRENT;
    if let Err(e) = save(&state) {
        say(&format!(
            "reims-ctl couldn't record its update of the VMs ({e:#}): it runs again next time"
        ));
    }
}

/// At a start of a stopped VM, a migration that waited while it ran, now, telling `progress`.
pub fn at_start(vm: &mut Vm, progress: &mut dyn FnMut(&str)) {
    if !load().deferred.iter().any(|d| same_dir(d, &vm.dir)) {
        return;
    }
    let Some(_lock) = lock() else { return };
    let mut state = load();
    let before = state.deferred.len();
    state.deferred.retain(|d| !same_dir(d, &vm.dir));
    if state.deferred.len() == before || save(&state).is_err() {
        return;
    }
    let name = format!("{}: ", vm.name());
    for n in 1..=CURRENT {
        let p = plan_of(n, vm, false);
        if let Some(line) = carry_out(vm, p, n) {
            progress(line.strip_prefix(&name).unwrap_or(&line));
        }
    }
}

fn same_dir(a: &Path, b: &Path) -> bool {
    a == b || std::fs::canonicalize(a).ok() == std::fs::canonicalize(b).ok()
}

/// Carries out migration `n`'s `plan` on `vm`: the line telling what happened, or None where
/// nothing did.
fn carry_out(vm: &mut Vm, plan: Plan, n: u32) -> Option<String> {
    let name = vm.name().to_string();
    let manual = format!(
        "reims-ctl set {name} --graphics reims-vgpu (for macOS 26.4 or later) or apple-gfx \
         makes its OpenCore image for one"
    );
    match plan {
        Plan::Skip => None,
        Plan::Defer => Some(format!(
            "{name} is running: it is updated at its next start, which makes its OpenCore image \
             again"
        )),
        Plan::Unknown(why) => Some(format!(
            "{name}: left as it was: reims-ctl can't tell which macOS its disk holds ({why}); \
             {manual}"
        )),
        Plan::Image(graphics, decided) => {
            Some(match image(vm, graphics, decided.as_ref(), None, n) {
                Ok(line) => format!("{name}: {line}"),
                Err(e) => format!("{name}: left as it was: {e:#}; {manual}"),
            })
        }
        Plan::Again(graphics) => Some(match image(vm, None, None, Some(graphics), n) {
            Ok(line) => format!("{name}: {line}"),
            Err(e) => format!(
                "{name}: left as it was: {e:#}; reims-ctl update-opencore {name} makes its \
                 OpenCore image again"
            ),
        }),
    }
}

/// Saves `graphics` in `vm`'s settings and makes its OpenCore image again for them, or, with
/// `kept`, for the graphics its settings name, keeping the old image beside it as migration `n`'s
/// backup; what changed, or an error with the VM as it was.
fn image(
    vm: &mut Vm,
    graphics: Option<Graphics>,
    decided: Option<&Macos>,
    kept: Option<Graphics>,
    n: u32,
) -> Result<String> {
    let q = vm.config.q35.clone().context("it boots no OpenCore")?;
    let old = vm.path(&q.opencore);
    let before = loaded(&old, q.opencore_format);
    // A copy of an image in the VM's directory, which the new one replaces; one elsewhere stays.
    let backup = old
        .starts_with(&vm.dir)
        .then(|| backup_of(&old, n))
        .flatten()
        .filter(|b| !b.exists());
    if let Some(b) = &backup {
        std::fs::copy(&old, b).with_context(|| format!("copy its image to {}", b.display()))?;
    }
    let undo = |vm: &mut Vm, e: anyhow::Error| {
        if let Some(b) = &backup {
            let _ = std::fs::remove_file(b);
        }
        if graphics.is_some()
            && let Ok(mut now) = Vm::load(&vm.dir)
        {
            now.config.graphics = None;
            if now.save().is_ok() {
                *vm = now;
            }
        }
        e
    };
    if let Some(g) = graphics {
        vm.config.graphics = Some(g);
        if let Err(e) = vm.save() {
            return Err(undo(vm, e));
        }
    }
    let made = match opencore::renew(vm, None, &mut |_| {}) {
        Ok(s) => s,
        Err(e) => return Err(undo(vm, e.context("a new OpenCore image couldn't be made"))),
    };
    let q = vm.config.q35.clone().context("it boots no OpenCore")?;
    let after = loaded(&vm.path(&q.opencore), q.opencore_format);
    let mut line = match (graphics, decided) {
        _ if kept.is_some() => format!(
            "its OpenCore image is made again for {}, without the ported drivers older images \
             held",
            kept.map(Graphics::name).unwrap_or_default()
        ),
        (Some(Graphics::ReimsVgpu), Some(m)) => {
            format!(
                "now on reims-vgpu's own GPU device, whose drivers {} runs",
                m.describe()
            )
        }
        (Some(Graphics::AppleGfx), Some(m)) => format!(
            "stays on apple-gfx: its {} is {}",
            m.describe(),
            macos::too_old_or_new(m)
        ),
        _ => "on reims-vgpu's own GPU device, the default, as its disk holds no macOS yet".into(),
    };
    line.push_str("; ");
    line.push_str(
        made.strip_prefix(&format!("{}: ", vm.name()))
            .unwrap_or(&made),
    );
    let dropped: Vec<&str> = before
        .iter()
        .filter(|k| !after.contains(k))
        .map(String::as_str)
        .collect();
    let added: Vec<&str> = after
        .iter()
        .filter(|k| !before.contains(k))
        .map(String::as_str)
        .collect();
    if !dropped.is_empty() {
        line.push_str(&format!(", without {}", some(&dropped)));
    }
    if !added.is_empty() {
        line.push_str(&format!(
            "{} with {}",
            if dropped.is_empty() { "," } else { " and" },
            some(&added)
        ));
    }
    if let Some(b) = &backup {
        line.push_str(&format!(
            "; the old image is kept as {}",
            b.file_name().unwrap_or_default().to_string_lossy()
        ));
    }
    Ok(line)
}

/// Up to three kexts by name; more as their count and the first two.
fn some(kexts: &[&str]) -> String {
    match kexts {
        [_, _, _, _, ..] => format!(
            "{} kexts ({}, {} and {} more)",
            kexts.len(),
            kexts[0],
            kexts[1],
            kexts.len() - 2
        ),
        _ => kexts.join(", "),
    }
}

/// The kexts an image's OpenCore loads; none where it can't be read.
fn loaded(image: &Path, format: crate::vm::ImageFormat) -> Vec<String> {
    Config::read(image, format)
        .map(|c| c.loaded_kexts().iter().map(|k| k.to_string()).collect())
        .unwrap_or_default()
}

/// Where migration `n` keeps a VM's old image: beside it, `.pre-migration-<n>` after its name.
fn backup_of(image: &Path, n: u32) -> Option<PathBuf> {
    let name = image.file_name()?.to_str()?;
    Some(image.with_file_name(format!("{name}.pre-migration-{n}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::opencore::tests::{image as oc_image, plist, qcow2};
    use std::os::unix::fs::PermissionsExt;

    fn macos(version: &str, build: &str) -> Macos {
        Macos {
            version: version.into(),
            build: Some(build.into()),
        }
    }

    fn entry(bundle: &str) -> String {
        format!(
            "<dict><key>BundlePath</key><string>{bundle}</string>\
             <key>Enabled</key><true/></dict>"
        )
    }

    /// A VM in `dir` of `profile`, with `graphics` where Some, whose image loads `kexts`.
    fn vm(dir: &Path, profile: &str, graphics: Option<&str>, kexts: &str) -> Vm {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("opencore.img"), oc_image(plist(kexts).as_bytes())).unwrap();
        let graphics = graphics
            .map(|g| format!("graphics = \"{g}\""))
            .unwrap_or_default();
        let config = format!(
            r#"
            name = "{name}"
            profile = "{profile}"
            cpus = 2
            memory_mib = 4096
            disk = "disk.img"
            disk_format = "raw"
            {graphics}
            [q35]
            opencore = "opencore.img"
            opencore_format = "raw"
            firmware_code = "OVMF_CODE.fd"
            firmware_vars = "OVMF_VARS.fd"
            "#,
            name = dir.file_name().unwrap().to_string_lossy()
        );
        std::fs::write(dir.join(crate::vm::CONFIG_FILE), config).unwrap();
        Vm::load(dir).unwrap()
    }

    /// A stand-in reims-util that makes each image it is asked for as `made`, recording its
    /// arguments in `args`; one that fails where `fails`.
    fn util(env: &crate::testenv::Env, made: &Path, args: &Path, fails: bool) {
        let util = env.root.join("reims-util");
        let script = if fails {
            "#!/bin/sh\necho 'no room' >&2\nexit 1\n".to_string()
        } else {
            format!(
                "#!/bin/sh\nprintf '%s\\n' \"$@\" > '{args}'\n\
                 while [ \"$1\" != -o ]; do shift; done\ncp '{made}' \"$2\"\n\
                 echo '{{\"event\":\"done\",\"result\":{{}}}}'\n",
                args = args.display(),
                made = made.display(),
            )
        };
        std::fs::write(&util, script).unwrap();
        std::fs::set_permissions(&util, std::fs::Permissions::from_mode(0o755)).unwrap();
        env.set("REIMS_CTL_UTIL", &util);
    }

    /// Migration 1 is for an x86-64 VM whose settings name no graphics, stopped: on the graphics
    /// its macOS runs, the test set's kexts loading on Darwin 25.4 and later (macOS 26.4); an
    /// arm64 VM, or one whose settings name graphics, as its user chose, is left be.
    #[test]
    fn migration_1_is_for_x86_64_vms_with_no_graphics_chosen() {
        let env = crate::testenv::Env::new("migrate-plan");
        let new = || Ok(vec![macos("26.7.2", "25G332")]);
        let old = || Ok(vec![macos("15.7.2", "24G325")]);
        let v = vm(&env.root.join("a"), "q35-x86_64", None, "");
        assert_eq!(
            plan(&v, false, new),
            Plan::Image(Some(Graphics::ReimsVgpu), Some(macos("26.7.2", "25G332")))
        );
        assert_eq!(
            plan(&v, false, old),
            Plan::Image(Some(Graphics::AppleGfx), Some(macos("15.7.2", "24G325")))
        );
        // A disk with an old and a new macOS: apple-gfx, for the old one.
        assert_eq!(
            plan(&v, false, || Ok(vec![
                macos("26.7.2", "25G332"),
                macos("15.7.2", "24G325")
            ])),
            Plan::Image(Some(Graphics::AppleGfx), Some(macos("15.7.2", "24G325")))
        );
        assert_eq!(plan(&v, false, || Ok(vec![])), Plan::Image(None, None));
        assert!(matches!(
            plan(&v, false, || anyhow::bail!("an encrypted volume")),
            Plan::Unknown(why) if why.contains("encrypted")
        ));
        assert_eq!(plan(&v, true, new), Plan::Defer);
        for g in ["apple-gfx", "reims-vgpu", "vmware-svga"] {
            let v = vm(&env.root.join(g), "q35-x86_64", Some(g), "");
            assert_eq!(plan(&v, false, new), Plan::Skip, "{g}");
        }
        let v = vm(&env.root.join("arm"), "vmapple-arm64", None, "");
        assert_eq!(plan(&v, false, new), Plan::Skip);
    }

    /// A migration makes the image again for the graphics it saves, the old image kept beside
    /// it, and tells what changed in one line; where the image can't be made, the VM's settings
    /// and image stay as they were, and the line says why and what to run.
    #[test]
    fn migration_1_makes_the_image_again_or_leaves_the_vm_as_it_was() {
        let env = crate::testenv::Env::new("migrate-image");
        let made = env.root.join("made.img");
        let args = env.root.join("args");
        // What reims-util makes: Lilu and the set's kexts (by name; the stand-in copies them).
        let fresh = entry("Lilu.kext") + &entry("A.kext") + &entry("B.kext");
        std::fs::write(&made, qcow2(&oc_image(plist(&fresh).as_bytes()))).unwrap();
        util(&env, &made, &args, false);
        let ported = entry("Lilu.kext") + &entry("AppleParavirtGPU.kext");
        let dir = env.root.join("old26");
        let mut v = vm(&dir, "q35-x86_64", None, &ported);
        let old_bytes = std::fs::read(dir.join("opencore.img")).unwrap();
        let line = carry_out(
            &mut v,
            Plan::Image(Some(Graphics::ReimsVgpu), Some(macos("26.7.2", "25G332"))),
            1,
        )
        .unwrap();
        assert!(
            line.starts_with("old26: now on reims-vgpu's own GPU device"),
            "{line}"
        );
        assert!(line.contains("macOS 26.7.2 (25G332)"), "{line}");
        assert!(
            line.contains("without AppleParavirtGPU.kext and with A.kext, B.kext"),
            "{line}"
        );
        assert!(
            line.contains("kept as opencore.img.pre-migration-1"),
            "{line}"
        );
        assert!(
            std::fs::read_to_string(&args)
                .unwrap()
                .contains("--add-kext")
        );
        let saved = Vm::load(&dir).unwrap();
        assert_eq!(saved.config.graphics, Some(Graphics::ReimsVgpu));
        assert_eq!(
            saved.config.q35.as_ref().unwrap().opencore,
            Path::new("opencore.qcow2")
        );
        assert_eq!(
            std::fs::read(dir.join("opencore.img.pre-migration-1")).unwrap(),
            old_bytes
        );

        // On apple-gfx, an image without the set's kexts, as reims-util makes for it.
        std::fs::write(
            &made,
            qcow2(&oc_image(plist(&entry("Lilu.kext")).as_bytes())),
        )
        .unwrap();
        let dir = env.root.join("old15");
        let mut v = vm(&dir, "q35-x86_64", None, &ported);
        let line = carry_out(
            &mut v,
            Plan::Image(Some(Graphics::AppleGfx), Some(macos("15.7.2", "24G325"))),
            1,
        )
        .unwrap();
        assert!(
            line.starts_with("old15: stays on apple-gfx: its macOS 15.7.2"),
            "{line}"
        );
        assert!(line.contains("older than macOS 26.4"), "{line}");
        assert!(line.contains("without AppleParavirtGPU.kext"), "{line}");
        assert!(
            !std::fs::read_to_string(&args)
                .unwrap()
                .contains("--add-kext")
        );
        assert_eq!(
            Vm::load(&dir).unwrap().config.graphics,
            Some(Graphics::AppleGfx)
        );

        // reims-util fails: the VM as it was.
        util(&env, &made, &args, true);
        let dir = env.root.join("stuck");
        let mut v = vm(&dir, "q35-x86_64", None, &ported);
        let before = std::fs::read(dir.join("opencore.img")).unwrap();
        let line = carry_out(
            &mut v,
            Plan::Image(Some(Graphics::ReimsVgpu), Some(macos("26.7.2", "25G332"))),
            1,
        )
        .unwrap();
        assert!(line.starts_with("stuck: left as it was:"), "{line}");
        assert!(line.contains("reims-ctl set stuck --graphics"), "{line}");
        let saved = Vm::load(&dir).unwrap();
        assert_eq!(saved.config.graphics, None);
        assert_eq!(
            saved.config.q35.as_ref().unwrap().opencore,
            Path::new("opencore.img")
        );
        assert_eq!(std::fs::read(dir.join("opencore.img")).unwrap(), before);
        assert!(!dir.join("opencore.img.pre-migration-1").exists());
    }

    /// The ported drivers' kexts, as an image from before reims-util's fe9bfe1 loads them.
    fn ported() -> String {
        std::iter::once("Lilu.kext")
            .chain(["ReimsVgpu.kext"])
            .chain(PORTED)
            .map(entry)
            .collect()
    }

    fn names(kexts: &[&str]) -> Vec<String> {
        kexts.iter().map(|k| k.to_string()).collect()
    }

    /// Migration 2 is for an x86-64 VM whose image loads the ported drivers, whatever graphics
    /// its settings name, which stay; one whose settings name none gets migration 1's plan. An
    /// image reims-util makes since fe9bfe1, or one migration 2 made, holds none, and is left be.
    #[test]
    fn migration_2_is_for_images_with_the_ported_drivers() {
        let env = crate::testenv::Env::new("migrate-plan-2");
        let new = || Ok(vec![macos("26.7.2", "25G332")]);
        let old = names(&[
            "Lilu.kext",
            "ReimsVgpu.kext",
            "AppleParavirtGPUIOGPUFamily.kext",
        ]);
        let fresh = names(&["Lilu.kext", "ReimsVgpu.kext"]);
        let ours = names(&["Lilu.kext", "ReimsGraphicsLoader.kext", "RgfxGate.kext"]);
        for (g, graphics) in [
            ("apple-gfx", Graphics::AppleGfx),
            ("reims-vgpu", Graphics::ReimsVgpu),
            ("vmware-svga", Graphics::VmwareSvga),
        ] {
            let v = vm(&env.root.join(g), "q35-x86_64", Some(g), "");
            assert_eq!(plan_2(&v, false, &old, new), Plan::Again(graphics), "{g}");
            assert_eq!(plan_2(&v, true, &old, new), Plan::Defer, "{g}");
            assert_eq!(plan_2(&v, false, &fresh, new), Plan::Skip, "{g}");
            assert_eq!(plan_2(&v, false, &ours, new), Plan::Skip, "{g}");
            assert_eq!(plan_2(&v, false, &[], new), Plan::Skip, "{g}");
        }
        // Names compare as FAT32 compares them, case aside.
        let shouted = names(&["APPLEPARAVIRTGPULOADER.KEXT"]);
        let v = vm(&env.root.join("case"), "q35-x86_64", Some("apple-gfx"), "");
        assert_eq!(
            plan_2(&v, false, &shouted, new),
            Plan::Again(Graphics::AppleGfx)
        );
        // No graphics named: migration 1's plan, from its disk's macOS.
        let v = vm(&env.root.join("none"), "q35-x86_64", None, "");
        assert_eq!(
            plan_2(&v, false, &old, new),
            Plan::Image(Some(Graphics::ReimsVgpu), Some(macos("26.7.2", "25G332")))
        );
        assert_eq!(plan_2(&v, false, &fresh, new), Plan::Skip);
        let v = vm(
            &env.root.join("arm"),
            "vmapple-arm64",
            Some("apple-gfx"),
            "",
        );
        assert_eq!(plan_2(&v, false, &old, new), Plan::Skip);
        // The plan reads the kexts from the VM's image.
        let v = vm(
            &env.root.join("img"),
            "q35-x86_64",
            Some("apple-gfx"),
            &ported(),
        );
        assert_eq!(plan_of(2, &v, false), Plan::Again(Graphics::AppleGfx));
        let v = vm(
            &env.root.join("img2"),
            "q35-x86_64",
            Some("apple-gfx"),
            &entry("Lilu.kext"),
        );
        assert_eq!(plan_of(2, &v, false), Plan::Skip);
    }

    /// Migration 2 makes the image again for the graphics the VM's settings name, which stay,
    /// keeping the old image as `.pre-migration-2`; an update from a reims-ctl that ran
    /// migration 1 runs migration 2 alone, once, and leaves the image it made be.
    #[test]
    fn migration_2_makes_the_image_again_for_the_graphics_set() {
        let env = crate::testenv::Env::new("migrate-image-2");
        let made = env.root.join("made.img");
        let args = env.root.join("args");
        std::fs::write(
            &made,
            qcow2(&oc_image(plist(&entry("Lilu.kext")).as_bytes())),
        )
        .unwrap();
        util(&env, &made, &args, false);
        let dir = env.root.join("gfx");
        let mut v = vm(&dir, "q35-x86_64", Some("apple-gfx"), &ported());
        let old_bytes = std::fs::read(dir.join("opencore.img")).unwrap();
        let line = carry_out(&mut v, Plan::Again(Graphics::AppleGfx), 2).unwrap();
        assert!(
            line.starts_with("gfx: its OpenCore image is made again for apple-gfx"),
            "{line}"
        );
        assert!(
            line.contains(
                "without 8 kexts (ReimsVgpu.kext, AppleParavirtGPULoader.kext and 6 more)"
            ),
            "{line}"
        );
        assert!(
            line.contains("kept as opencore.img.pre-migration-2"),
            "{line}"
        );
        let saved = Vm::load(&dir).unwrap();
        assert_eq!(saved.config.graphics, Some(Graphics::AppleGfx));
        assert_eq!(
            saved.config.q35.as_ref().unwrap().opencore,
            Path::new("opencore.qcow2")
        );
        assert_eq!(
            std::fs::read(dir.join("opencore.img.pre-migration-2")).unwrap(),
            old_bytes
        );
        assert_eq!(plan_of(2, &saved, false), Plan::Skip);

        // reims-util fails: the VM as it was.
        util(&env, &made, &args, true);
        let dir = env.root.join("stuck");
        let mut v = vm(&dir, "q35-x86_64", Some("apple-gfx"), &ported());
        let before = std::fs::read(dir.join("opencore.img")).unwrap();
        let line = carry_out(&mut v, Plan::Again(Graphics::AppleGfx), 2).unwrap();
        assert!(line.starts_with("stuck: left as it was:"), "{line}");
        assert!(line.contains("reims-ctl update-opencore stuck"), "{line}");
        assert_eq!(std::fs::read(dir.join("opencore.img")).unwrap(), before);
        assert!(!dir.join("opencore.img.pre-migration-2").exists());

        // After migration 1: migration 2 alone, on the VM whose image holds the ported drivers.
        util(&env, &made, &args, false);
        let mut reg = Registry::default();
        for (name, kexts) in [("ported", ported()), ("fresh", entry("Lilu.kext"))] {
            let dir = env.root.join(name);
            vm(&dir, "q35-x86_64", Some("apple-gfx"), &kexts);
            reg.add(name, &dir).unwrap();
        }
        reg.save().unwrap();
        save(&State {
            done: 1,
            deferred: vec![],
        })
        .unwrap();
        let mut said = Vec::new();
        run(&mut |s| said.push(s.to_string()));
        assert!(said[0].starts_with("reims-ctl is updated"), "{said:?}");
        assert_eq!(said.len(), 2, "{said:?}");
        assert!(
            said[1].starts_with("ported: its OpenCore image is made again"),
            "{said:?}"
        );
        assert!(
            env.root
                .join("ported/opencore.img.pre-migration-2")
                .exists()
        );
        assert!(!env.root.join("fresh/opencore.img.pre-migration-2").exists());
        assert_eq!(load().done, 2);
        let mut said = Vec::new();
        run(&mut |s| said.push(s.to_string()));
        assert!(said.is_empty(), "{said:?}");
    }

    #[test]
    fn a_line_names_up_to_three_kexts() {
        assert_eq!(some(&["A.kext"]), "A.kext");
        assert_eq!(
            some(&["A.kext", "B.kext", "C.kext"]),
            "A.kext, B.kext, C.kext"
        );
        assert_eq!(
            some(&["A.kext", "B.kext", "C.kext", "D.kext", "E.kext"]),
            "5 kexts (A.kext, B.kext and 3 more)"
        );
    }

    /// The first run records the migrations, and later ones do nothing; a VM running then waits
    /// for its next start, which migrates it once.
    #[test]
    fn migrations_run_once_and_a_running_vm_waits_for_its_start() {
        let env = crate::testenv::Env::new("migrate-run");
        let made = env.root.join("made.img");
        let args = env.root.join("args");
        let fresh = entry("Lilu.kext") + &entry("A.kext") + &entry("B.kext");
        std::fs::write(&made, qcow2(&oc_image(plist(&fresh).as_bytes()))).unwrap();
        util(&env, &made, &args, false);
        // No VMs: recorded, nothing said.
        let mut said = Vec::new();
        run(&mut |s| said.push(s.to_string()));
        assert!(said.is_empty(), "{said:?}");
        assert_eq!(load().done, CURRENT);
        let _ = std::fs::remove_file(state_file());

        // Three VMs: two whose settings name no graphics, one of them running, and one its user
        // set to apple-gfx.
        let mut reg = Registry::default();
        for (name, graphics) in [
            ("idle", None),
            ("busy", None),
            ("chosen", Some("apple-gfx")),
        ] {
            let dir = env.root.join(name);
            vm(&dir, "q35-x86_64", graphics, &entry("Lilu.kext"));
            reg.add(name, &dir).unwrap();
        }
        reg.save().unwrap();
        // A process whose command line names the pidfile, as QEMU's does (runtime::pid_at).
        let pidfile = crate::vm::pidfile_in(&env.root.join("busy"));
        let mut qemu = std::process::Command::new("sh")
            .args(["-c", "sleep 30"])
            .arg(&pidfile)
            .spawn()
            .unwrap();
        std::fs::write(&pidfile, qemu.id().to_string()).unwrap();
        let mut said = Vec::new();
        run(&mut |s| said.push(s.to_string()));
        assert!(said[0].starts_with("reims-ctl is updated"), "{said:?}");
        assert!(
            said.iter().any(|s| s.starts_with("busy is running")),
            "{said:?}"
        );
        // The idle one's disk can't be read here (there is none): left as it was.
        assert!(
            said.iter().any(|s| s.starts_with("idle: left as it was")),
            "{said:?}"
        );
        assert!(!said.iter().any(|s| s.starts_with("chosen")), "{said:?}");
        let state = load();
        assert_eq!(state.done, CURRENT);
        assert_eq!(state.deferred, vec![env.root.join("busy")]);
        let mut said = Vec::new();
        run(&mut |s| said.push(s.to_string()));
        assert!(said.is_empty(), "{said:?}");

        // Stopped, its start migrates it, and only once.
        let _ = qemu.kill();
        let _ = qemu.wait();
        std::fs::remove_file(&pidfile).unwrap();
        let mut v = Vm::load(&env.root.join("busy")).unwrap();
        let mut said = Vec::new();
        at_start(&mut v, &mut |s| said.push(s.to_string()));
        assert_eq!(said.len(), 1, "{said:?}");
        assert!(said[0].starts_with("left as it was"), "{said:?}");
        assert!(load().deferred.is_empty());
        let mut said = Vec::new();
        at_start(&mut v, &mut |s| said.push(s.to_string()));
        assert!(said.is_empty(), "{said:?}");
    }
}
