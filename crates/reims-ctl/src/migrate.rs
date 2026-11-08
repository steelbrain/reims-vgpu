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
//! Each makes a VM's OpenCore image again, presenting the same Mac ([`crate::opencore::renew`]),
//! for the graphics it runs, the old image kept beside it as `<image>.pre-migration-<n>`:
//! - Migration 1 is for the x86-64 VMs a reims-ctl from before reims-vgpu's own GPU device made:
//!   their settings name no graphics, which is that device, and their image has no kexts for it,
//!   and may hold the ported drivers images no longer carry.
//! - Migration 2 is for the VMs whose image still loads the ported drivers ([`PORTED`]), whatever
//!   graphics their settings name, which stay: an image reims-util made before reims-ctl's
//!   graphics option, which migration 1 left be where the VM's settings named graphics. One
//!   whose settings name no graphics is migration 1's, and one whose settings name apple-gfx
//!   migration 3's.
//! - Migration 3 is for the VMs whose settings name apple-gfx, Apple's paravirtualized GPU, which
//!   reims-vgpu no longer has: they read as reims-vgpu's own device ([`Graphics::ReimsVgpu`]),
//!   and its image, made for apple-gfx, lacks the device's kexts. The new image's save names the
//!   device.
//! - Migration 4 makes again the image of every x86-64 VM that lists Lilu, the old graphics
//!   loader or one of its two shims, enabled or not, whatever its device: the guest's set now
//!   loads through Graft. A VM on reims-vgpu's own device gets Graft and the set; one on another
//!   device gets no kext.

use crate::opencore::{self, Config};
use crate::vm::{CONFIG_FILE, Graphics, Registry, Vm};
use crate::{host::Profile, paths, runtime};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};

/// The migrations this reims-ctl carries: 1 to this.
pub const CURRENT: u32 = 4;

/// The ported drivers older OpenCore images load, by bundle name: the seven kexts that run Apple's
/// arm64 GPU driver on x86-64, which reims-util carried until its fe9bfe1.
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
    /// Its image made again for the graphics it runs, for this reason.
    Image(Why),
}

/// Why a VM's image is made again.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Why {
    /// Migration 1: its settings name no graphics, which is reims-vgpu's own device.
    Default,
    /// Migration 2: its image loads the ported drivers; its settings name these graphics.
    Ported(Graphics),
    /// Migration 3: its settings name apple-gfx.
    AppleGfx,
    /// Migration 4: its image lists Lilu, the old graphics loader or one of its shims.
    Graft,
}

/// Migration 1's plan for `vm`, `running` or not.
fn plan(vm: &Vm, running: bool) -> Plan {
    if vm.config.profile != Profile::Q35X86_64 || vm.config.graphics.is_some() {
        return Plan::Skip;
    }
    if running {
        return Plan::Defer;
    }
    Plan::Image(Why::Default)
}

/// Migration 2's plan for `vm`, `running` or not, from the kexts its image loads, and whether its
/// settings name apple-gfx (migration 3's); one whose settings name no graphics gets migration
/// 1's.
fn plan_2(vm: &Vm, running: bool, kexts: &[String], apple_gfx: bool) -> Plan {
    let ported = kexts
        .iter()
        .any(|k| PORTED.iter().any(|p| k.eq_ignore_ascii_case(p)));
    if vm.config.profile != Profile::Q35X86_64 || !ported || apple_gfx {
        return Plan::Skip;
    }
    match vm.config.graphics {
        None => plan(vm, running),
        Some(_) if running => Plan::Defer,
        Some(g) => Plan::Image(Why::Ported(g)),
    }
}

/// Migration 3's plan for `vm`, `running` or not, whose settings name apple-gfx or not.
fn plan_3(vm: &Vm, running: bool, apple_gfx: bool) -> Plan {
    match (vm.config.profile, apple_gfx, running) {
        (Profile::Q35X86_64, true, true) => Plan::Defer,
        (Profile::Q35X86_64, true, false) => Plan::Image(Why::AppleGfx),
        _ => Plan::Skip,
    }
}

/// Migration 4's plan for `vm`, `running` or not, whose image lists `kexts` (disabled entries
/// too): an x86-64 VM listing any of [`opencore::LEGACY_KEXTS`] has its image made again,
/// whatever its device.
fn plan_4(vm: &Vm, running: bool, kexts: &[String]) -> Plan {
    if vm.config.profile != Profile::Q35X86_64
        || !kexts.iter().any(|k| {
            opencore::LEGACY_KEXTS
                .iter()
                .any(|old| k.eq_ignore_ascii_case(old))
        })
    {
        return Plan::Skip;
    }
    if running {
        Plan::Defer
    } else {
        Plan::Image(Why::Graft)
    }
}

/// Whether `vm`'s settings, as saved, name apple-gfx: they read as reims-vgpu's own device.
fn names_apple_gfx(vm: &Vm) -> bool {
    std::fs::read_to_string(vm.dir.join(CONFIG_FILE))
        .ok()
        .and_then(|t| t.parse::<toml::Table>().ok())
        .is_some_and(|t| t.get("graphics").and_then(toml::Value::as_str) == Some("apple-gfx"))
}

/// Migration `n`'s plan for `vm`, `running` or not.
fn plan_of(n: u32, vm: &Vm, running: bool) -> Plan {
    match n {
        1 => plan(vm, running),
        2 => {
            let kexts = vm
                .config
                .q35
                .as_ref()
                .map(|q| loaded(&vm.path(&q.opencore), q.opencore_format))
                .unwrap_or_default();
            plan_2(vm, running, &kexts, names_apple_gfx(vm))
        }
        3 => plan_3(vm, running, names_apple_gfx(vm)),
        4 => {
            let kexts = vm
                .config
                .q35
                .as_ref()
                .and_then(|q| Config::read(&vm.path(&q.opencore), q.opencore_format).ok())
                .map(|c| {
                    c.configured_kexts()
                        .iter()
                        .map(|k| k.to_string())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            plan_4(vm, running, &kexts)
        }
        _ => Plan::Skip,
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
    match plan {
        Plan::Skip => None,
        Plan::Defer => Some(format!(
            "{name} is running: it is updated at its next start, which makes its OpenCore image \
             again"
        )),
        Plan::Image(why) => Some(match image(vm, why, n) {
            Ok(line) => format!("{name}: {line}"),
            Err(e) => format!(
                "{name}: left as it was: {e:#}; reims-ctl update-opencore {name} makes its \
                 OpenCore image again"
            ),
        }),
    }
}

/// Makes `vm`'s OpenCore image again for the graphics it runs, keeping the old image beside it as
/// migration `n`'s backup; what changed, or an error with the VM as it was.
fn image(vm: &mut Vm, why: Why, n: u32) -> Result<String> {
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
    let made = match opencore::renew(vm, None, &mut |_| {}) {
        Ok(s) => s,
        Err(e) => {
            if let Some(b) = &backup {
                let _ = std::fs::remove_file(b);
            }
            return Err(e.context("a new OpenCore image couldn't be made"));
        }
    };
    let q = vm.config.q35.clone().context("it boots no OpenCore")?;
    let after = loaded(&vm.path(&q.opencore), q.opencore_format);
    let mut line = match why {
        Why::Default => "on reims-vgpu's own GPU device, the default".to_string(),
        Why::Ported(g) => format!(
            "its OpenCore image is made again for {}, without the ported drivers older images \
             held",
            g.name()
        ),
        Why::AppleGfx => "moved from apple-gfx, which reims-vgpu no longer has, to reims-vgpu's \
                          own GPU device"
            .to_string(),
        Why::Graft => "its OpenCore image is made again without Lilu or the old graphics loader \
                       and shims"
            .to_string(),
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

    /// A stand-in for a running VM's QEMU: a shell whose command line names `pidfile`, as QEMU's
    /// does, and which writes its own pid there, as QEMU does. Only then can runtime::pid_at take
    /// it for running: until then its pid holds the test's command line, then `sh`'s, which may
    /// exec another shell, and none during each exec. It loops, since a shell may exec a lone
    /// command in its own place, dropping `pidfile` from its command line. It is returned once
    /// runtime::pid_at sees it.
    fn running_qemu(pidfile: &Path) -> std::process::Child {
        let child = std::process::Command::new("sh")
            .args(["-c", r#"echo $$ > "$1"; while :; do sleep 1; done"#, "sh"])
            .arg(pidfile)
            .spawn()
            .unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while runtime::pid_at(pidfile).is_none() {
            assert!(
                std::time::Instant::now() < deadline,
                "the stand-in never wrote its pidfile"
            );
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        child
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

    /// Migration 1 is for an x86-64 VM whose settings name no graphics, stopped; one whose
    /// settings name graphics, as its user chose (apple-gfx among them, migration 3's), is left
    /// be.
    #[test]
    fn migration_1_is_for_x86_64_vms_with_no_graphics_chosen() {
        let env = crate::testenv::Env::new("migrate-plan");
        let v = vm(&env.root.join("a"), "q35-x86_64", None, "");
        assert_eq!(plan(&v, false), Plan::Image(Why::Default));
        assert_eq!(plan(&v, true), Plan::Defer);
        for g in ["apple-gfx", "reims-vgpu", "vmware-svga"] {
            let v = vm(&env.root.join(g), "q35-x86_64", Some(g), "");
            assert_eq!(plan(&v, false), Plan::Skip, "{g}");
        }
    }

    /// A migration makes the image again for the graphics the VM runs, the old image kept beside
    /// it, and tells what changed in one line; where the image can't be made, the VM's settings
    /// and image stay as they were, and the line says why and what to run.
    #[test]
    fn migration_1_makes_the_image_again_or_leaves_the_vm_as_it_was() {
        let env = crate::testenv::Env::new("migrate-image");
        let made = env.root.join("made.img");
        let args = env.root.join("args");
        // What reims-util makes: the set's kexts (by name; the stand-in copies them).
        let fresh = entry("A.kext") + &entry("B.kext");
        std::fs::write(&made, qcow2(&oc_image(plist(&fresh).as_bytes()))).unwrap();
        util(&env, &made, &args, false);
        let ported = entry("Lilu.kext") + &entry("AppleParavirtGPU.kext");
        let dir = env.root.join("old26");
        let mut v = vm(&dir, "q35-x86_64", None, &ported);
        let old_bytes = std::fs::read(dir.join("opencore.img")).unwrap();
        let line = carry_out(&mut v, Plan::Image(Why::Default), 1).unwrap();
        assert!(
            line.starts_with("old26: on reims-vgpu's own GPU device, the default"),
            "{line}"
        );
        assert!(
            line.contains("without Lilu.kext, AppleParavirtGPU.kext and with A.kext, B.kext"),
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
        assert_eq!(saved.config.graphics, None);
        assert_eq!(
            saved.config.q35.as_ref().unwrap().opencore,
            Path::new("opencore.qcow2")
        );
        assert_eq!(
            std::fs::read(dir.join("opencore.img.pre-migration-1")).unwrap(),
            old_bytes
        );

        // reims-util fails: the VM as it was.
        util(&env, &made, &args, true);
        let dir = env.root.join("stuck");
        let mut v = vm(&dir, "q35-x86_64", None, &ported);
        let before = std::fs::read(dir.join("opencore.img")).unwrap();
        let line = carry_out(&mut v, Plan::Image(Why::Default), 1).unwrap();
        assert!(line.starts_with("stuck: left as it was:"), "{line}");
        assert!(line.contains("reims-ctl update-opencore stuck"), "{line}");
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
    /// its settings name, which stay; one whose settings name none gets migration 1's plan, and
    /// one whose settings name apple-gfx is migration 3's. An image reims-util makes since
    /// fe9bfe1, or one migration 2 made, holds none, and is left be.
    #[test]
    fn migration_2_is_for_images_with_the_ported_drivers() {
        let env = crate::testenv::Env::new("migrate-plan-2");
        let old = names(&[
            "Lilu.kext",
            "ReimsVgpu.kext",
            "AppleParavirtGPUIOGPUFamily.kext",
        ]);
        let fresh = names(&["Lilu.kext"]);
        let ours = names(&["Lilu.kext", "ReimsGraphicsLoader.kext", "RgfxGate.kext"]);
        for (g, graphics) in [
            ("reims-vgpu", Graphics::ReimsVgpu),
            ("vmware-svga", Graphics::VmwareSvga),
        ] {
            let v = vm(&env.root.join(g), "q35-x86_64", Some(g), "");
            assert_eq!(
                plan_2(&v, false, &old, false),
                Plan::Image(Why::Ported(graphics)),
                "{g}"
            );
            assert_eq!(plan_2(&v, true, &old, false), Plan::Defer, "{g}");
            assert_eq!(plan_2(&v, false, &fresh, false), Plan::Skip, "{g}");
            assert_eq!(plan_2(&v, false, &ours, false), Plan::Skip, "{g}");
            assert_eq!(plan_2(&v, false, &[], false), Plan::Skip, "{g}");
        }
        // Names compare as FAT32 compares them, case aside.
        let shouted = names(&["APPLEPARAVIRTGPULOADER.KEXT"]);
        let v = vm(
            &env.root.join("case"),
            "q35-x86_64",
            Some("vmware-svga"),
            "",
        );
        assert_eq!(
            plan_2(&v, false, &shouted, false),
            Plan::Image(Why::Ported(Graphics::VmwareSvga))
        );
        // No graphics named: migration 1's plan.
        let v = vm(&env.root.join("none"), "q35-x86_64", None, "");
        assert_eq!(plan_2(&v, false, &old, false), Plan::Image(Why::Default));
        assert_eq!(plan_2(&v, false, &fresh, false), Plan::Skip);
        // apple-gfx named: migration 3's, read from the settings as saved.
        let v = vm(
            &env.root.join("apple"),
            "q35-x86_64",
            Some("apple-gfx"),
            &ported(),
        );
        assert_eq!(plan_2(&v, false, &old, true), Plan::Skip);
        assert_eq!(plan_of(2, &v, false), Plan::Skip);
        // The plan reads the kexts from the VM's image.
        let v = vm(
            &env.root.join("img"),
            "q35-x86_64",
            Some("vmware-svga"),
            &ported(),
        );
        assert_eq!(
            plan_of(2, &v, false),
            Plan::Image(Why::Ported(Graphics::VmwareSvga))
        );
        let v = vm(
            &env.root.join("img2"),
            "q35-x86_64",
            Some("vmware-svga"),
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
        std::fs::write(&made, qcow2(&oc_image(plist("").as_bytes()))).unwrap();
        util(&env, &made, &args, false);
        let dir = env.root.join("gfx");
        let mut v = vm(&dir, "q35-x86_64", Some("vmware-svga"), &ported());
        let old_bytes = std::fs::read(dir.join("opencore.img")).unwrap();
        let line = carry_out(&mut v, Plan::Image(Why::Ported(Graphics::VmwareSvga)), 2).unwrap();
        assert!(
            line.starts_with("gfx: its OpenCore image is made again for VMware SVGA"),
            "{line}"
        );
        assert!(
            line.contains("without 9 kexts (Lilu.kext, ReimsVgpu.kext and 7 more)"),
            "{line}"
        );
        assert!(
            line.contains("kept as opencore.img.pre-migration-2"),
            "{line}"
        );
        let saved = Vm::load(&dir).unwrap();
        assert_eq!(saved.config.graphics, Some(Graphics::VmwareSvga));
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
        let mut v = vm(&dir, "q35-x86_64", Some("vmware-svga"), &ported());
        let before = std::fs::read(dir.join("opencore.img")).unwrap();
        let line = carry_out(&mut v, Plan::Image(Why::Ported(Graphics::VmwareSvga)), 2).unwrap();
        assert!(line.starts_with("stuck: left as it was:"), "{line}");
        assert!(line.contains("reims-ctl update-opencore stuck"), "{line}");
        assert_eq!(std::fs::read(dir.join("opencore.img")).unwrap(), before);
        assert!(!dir.join("opencore.img.pre-migration-2").exists());

        // After migration 1: migrations 2 and 3, on the VM whose image holds the ported drivers.
        util(&env, &made, &args, false);
        let mut reg = Registry::default();
        for (name, kexts) in [("ported", ported()), ("fresh", String::new())] {
            let dir = env.root.join(name);
            vm(&dir, "q35-x86_64", Some("vmware-svga"), &kexts);
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
        assert_eq!(load().done, CURRENT);
        let mut said = Vec::new();
        run(&mut |s| said.push(s.to_string()));
        assert!(said.is_empty(), "{said:?}");
    }

    /// Migration 3 is for an x86-64 VM whose settings, as saved, name apple-gfx: its image is made
    /// again for reims-vgpu's own device, which its settings then name, the old one kept as
    /// `.pre-migration-3`; one running waits for its start.
    #[test]
    fn migration_3_moves_apple_gfx_vms_to_reims_vgpu() {
        let env = crate::testenv::Env::new("migrate-apple");
        let made = env.root.join("made.img");
        let args = env.root.join("args");
        let fresh = entry("A.kext") + &entry("B.kext");
        std::fs::write(&made, qcow2(&oc_image(plist(&fresh).as_bytes()))).unwrap();
        util(&env, &made, &args, false);
        for g in [None, Some("reims-vgpu"), Some("vmware-svga")] {
            let v = vm(&env.root.join(g.unwrap_or("none")), "q35-x86_64", g, "");
            assert_eq!(plan_of(3, &v, false), Plan::Skip, "{g:?}");
        }
        let dir = env.root.join("apple");
        let mut v = vm(&dir, "q35-x86_64", Some("apple-gfx"), &entry("Lilu.kext"));
        assert_eq!(v.config.graphics, Some(Graphics::ReimsVgpu));
        assert_eq!(plan_of(3, &v, true), Plan::Defer);
        let p = plan_of(3, &v, false);
        assert_eq!(p, Plan::Image(Why::AppleGfx));
        let line = carry_out(&mut v, p, 3).unwrap();
        assert!(
            line.starts_with(
                "apple: moved from apple-gfx, which reims-vgpu no longer has, to reims-vgpu's \
                 own GPU device"
            ),
            "{line}"
        );
        assert!(
            line.contains("without Lilu.kext and with A.kext, B.kext"),
            "{line}"
        );
        assert!(
            line.contains("kept as opencore.img.pre-migration-3"),
            "{line}"
        );
        assert!(
            std::fs::read_to_string(&args)
                .unwrap()
                .contains("--add-kext")
        );
        let saved = std::fs::read_to_string(dir.join(CONFIG_FILE)).unwrap();
        assert!(saved.contains("graphics = \"reims-vgpu\""), "{saved}");
        assert_eq!(plan_of(3, &Vm::load(&dir).unwrap(), false), Plan::Skip);
    }

    #[test]
    fn migration_4_finds_each_legacy_component_for_every_graphics_choice() {
        let env = crate::testenv::Env::new("migrate-graft-plan");
        for graphics in [None, Some("reims-vgpu"), Some("vmware-svga")] {
            let v = vm(
                &env.root.join(graphics.unwrap_or("none")),
                "q35-x86_64",
                graphics,
                "",
            );
            for legacy in opencore::LEGACY_KEXTS {
                let listed = names(&[&legacy.to_ascii_uppercase()]);
                assert_eq!(plan_4(&v, false, &listed), Plan::Image(Why::Graft));
                assert_eq!(plan_4(&v, true, &listed), Plan::Defer);
            }
            assert_eq!(
                plan_4(
                    &v,
                    false,
                    &names(&["Graft.kext", "RgfxGate.kext", "ReimsGraphics.kext"])
                ),
                Plan::Skip
            );
        }
        let disabled = entry("Lilu.kext").replace("<true/>", "<false/>");
        let v = vm(
            &env.root.join("disabled"),
            "q35-x86_64",
            Some("vmware-svga"),
            &disabled,
        );
        assert_eq!(plan_of(4, &v, false), Plan::Image(Why::Graft));
    }

    #[test]
    fn migration_4_preserves_identity_and_graphics_and_replaces_the_legacy_image() {
        let env = crate::testenv::Env::new("migrate-graft-image");
        let made = env.root.join("made.img");
        let args = env.root.join("args");
        let legacy = opencore::LEGACY_KEXTS
            .iter()
            .map(|k| entry(k))
            .collect::<String>();
        for graphics in ["reims-vgpu", "vmware-svga"] {
            let dir = env.root.join(graphics);
            let mut v = vm(&dir, "q35-x86_64", Some(graphics), &legacy);
            let before = std::fs::read(dir.join("opencore.img")).unwrap();
            let identity = Config::read(&dir.join("opencore.img"), crate::vm::ImageFormat::Raw)
                .unwrap()
                .identity()
                .unwrap();
            let fresh = if graphics == "reims-vgpu" {
                entry("Graft.kext") + &entry("RgfxGate.kext") + &entry("ReimsGraphics.kext")
            } else {
                String::new()
            };
            std::fs::write(&made, qcow2(&oc_image(plist(&fresh).as_bytes()))).unwrap();
            util(&env, &made, &args, false);
            let p = plan_of(4, &v, false);
            let line = carry_out(&mut v, p, 4).unwrap();
            assert!(
                line.contains("without Lilu or the old graphics loader and shims"),
                "{line}"
            );
            assert!(
                line.contains("kept as opencore.img.pre-migration-4"),
                "{line}"
            );
            let saved = Vm::load(&dir).unwrap();
            assert_eq!(saved.config.graphics, v.config.graphics);
            assert_eq!(
                std::fs::read(dir.join("opencore.img.pre-migration-4")).unwrap(),
                before
            );
            let q = saved.config.q35.as_ref().unwrap();
            let after = Config::read(&saved.path(&q.opencore), q.opencore_format).unwrap();
            assert_eq!(after.identity().unwrap(), identity);
            let given = std::fs::read_to_string(&args).unwrap();
            for value in [
                &identity.model,
                &identity.serial,
                &identity.mlb,
                &identity.uuid,
                &identity.rom,
            ] {
                assert!(given.lines().any(|arg| arg == value), "{given}");
            }
            assert_eq!(given.contains("--add-kext"), graphics == "reims-vgpu");
            assert_eq!(plan_of(4, &saved, false), Plan::Skip);
            assert_eq!(
                after.loaded_kexts(),
                if graphics == "reims-vgpu" {
                    vec!["Graft.kext", "RgfxGate.kext", "ReimsGraphics.kext"]
                } else {
                    vec![]
                }
            );
        }
    }

    #[test]
    fn migration_4_rejects_an_old_util_without_replacing_the_image() {
        let env = crate::testenv::Env::new("migrate-graft-old-util");
        let made = env.root.join("made.img");
        let args = env.root.join("args");
        let dir = env.root.join("old-util");
        let mut v = vm(&dir, "q35-x86_64", Some("vmware-svga"), &entry("Lilu.kext"));
        let before = std::fs::read(dir.join("opencore.img")).unwrap();
        let settings = std::fs::read(dir.join(CONFIG_FILE)).unwrap();
        for legacy in opencore::LEGACY_KEXTS {
            let disabled = entry(legacy).replace("<true/>", "<false/>");
            std::fs::write(&made, qcow2(&oc_image(plist(&disabled).as_bytes()))).unwrap();
            util(&env, &made, &args, false);
            let line = carry_out(&mut v, Plan::Image(Why::Graft), 4).unwrap();
            assert!(
                line.contains("still includes obsolete guest kexts"),
                "{line}"
            );
            assert_eq!(std::fs::read(dir.join("opencore.img")).unwrap(), before);
            assert_eq!(std::fs::read(dir.join(CONFIG_FILE)).unwrap(), settings);
            assert!(!dir.join("opencore.img.pre-migration-4").exists());
            assert!(!dir.join("opencore.new.qcow2").exists());
        }
    }

    #[test]
    fn migration_4_defers_a_running_vm_and_runs_at_its_next_start() {
        let env = crate::testenv::Env::new("migrate-graft-running");
        let dir = env.root.join("busy");
        let mut v = vm(&dir, "q35-x86_64", Some("vmware-svga"), &entry("Lilu.kext"));
        let before = std::fs::read(dir.join("opencore.img")).unwrap();
        let made = env.root.join("made.img");
        let args = env.root.join("args");
        std::fs::write(&made, qcow2(&oc_image(plist("").as_bytes()))).unwrap();
        util(&env, &made, &args, false);
        let mut reg = Registry::default();
        reg.add("busy", &dir).unwrap();
        reg.save().unwrap();
        save(&State {
            done: 3,
            deferred: vec![],
        })
        .unwrap();
        let pidfile = crate::vm::pidfile_in(&dir);
        let mut process = running_qemu(&pidfile);
        let mut said = Vec::new();
        run(&mut |line| said.push(line.to_string()));
        assert!(
            said.iter().any(|line| line.starts_with("busy is running")),
            "{said:?}"
        );
        assert_eq!(load().deferred, vec![dir.clone()]);
        assert_eq!(load().done, CURRENT);
        assert_eq!(std::fs::read(dir.join("opencore.img")).unwrap(), before);
        assert!(!args.exists());
        process.kill().unwrap();
        process.wait().unwrap();
        std::fs::remove_file(pidfile).unwrap();
        at_start(&mut v, &mut |_| {});
        assert!(load().deferred.is_empty());
        assert_eq!(plan_of(4, &v, false), Plan::Skip);
        assert!(dir.join("opencore.img.pre-migration-4").exists());
        assert!(
            !std::fs::read_to_string(args)
                .unwrap()
                .contains("--add-kext")
        );
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
        let fresh = entry("A.kext") + &entry("B.kext");
        std::fs::write(&made, qcow2(&oc_image(plist(&fresh).as_bytes()))).unwrap();
        util(&env, &made, &args, false);
        // No VMs: recorded, nothing said.
        let mut said = Vec::new();
        run(&mut |s| said.push(s.to_string()));
        assert!(said.is_empty(), "{said:?}");
        assert_eq!(load().done, CURRENT);
        let _ = std::fs::remove_file(state_file());

        // Three VMs: two whose settings name no graphics, one of them running, and one its user
        // set to VMware SVGA.
        let mut reg = Registry::default();
        for (name, graphics) in [
            ("idle", None),
            ("busy", None),
            ("chosen", Some("vmware-svga")),
        ] {
            let dir = env.root.join(name);
            let kexts = if graphics.is_some() {
                String::new()
            } else {
                entry("Lilu.kext")
            };
            vm(&dir, "q35-x86_64", graphics, &kexts);
            reg.add(name, &dir).unwrap();
        }
        reg.save().unwrap();
        let pidfile = crate::vm::pidfile_in(&env.root.join("busy"));
        let mut qemu = running_qemu(&pidfile);
        let mut said = Vec::new();
        run(&mut |s| said.push(s.to_string()));
        assert!(said[0].starts_with("reims-ctl is updated"), "{said:?}");
        assert!(
            said.iter().any(|s| s.starts_with("busy is running")),
            "{said:?}"
        );
        assert!(
            said.iter()
                .any(|s| s.starts_with("idle: on reims-vgpu's own GPU device")),
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
        assert!(
            said[0].starts_with("on reims-vgpu's own GPU device"),
            "{said:?}"
        );
        assert!(load().deferred.is_empty());
        let mut said = Vec::new();
        at_start(&mut v, &mut |s| said.push(s.to_string()));
        assert!(said.is_empty(), "{said:?}");
    }
}
