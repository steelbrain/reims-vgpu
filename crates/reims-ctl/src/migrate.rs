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
//! Migrations 1 to 4 each make a VM's OpenCore image again, presenting the same Mac
//! ([`crate::opencore::renew`]), for the graphics it runs, the old image kept beside it as
//! `<image>.pre-migration-<n>`; migration 5 changes only a VM's settings:
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
//! - Migration 5 is for the VMs given a GPU from `reims-ctl gpus` by a reims-ctl that named it by
//!   its driver's manifest alone, which hid the host's other GPUs from the library: where the
//!   host's list has that GPU, by its manifest, name and any device ID, its settings name it as a
//!   pick from the list does now ([`crate::gpu::GpuInfo::to_config`]), by its IDs where those
//!   name it. A running VM's too: QEMU reads them at the VM's next start. The host's GPUs are
//!   listed once a run, and only for a VM that has such a GPU.
//! - Migration 6 is for the VMs whose GPU a reims-ctl saved under a driver's manifest whose probe
//!   finds GPUs of other drivers too ([`crate::gpu`], "A probe past its manifest"), which does
//!   not name that GPU: as a user's Windows host saved a GTX 1650 Ti under Intel's manifest,
//!   with no IDs. Where that manifest's probe now goes past it and finds one GPU of the
//!   settings' name and any device ID, they name it as a pick of that GPU from the list does
//!   now, by its IDs, as migration 5's do; a running VM's too, and the probes run as migration
//!   5's list does.

use crate::gpu::GpuInfo;
use crate::opencore::{self, Config};
use crate::vm::{CONFIG_FILE, Graphics, Registry, Vm};
use crate::{host::Profile, paths, runtime};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::path::{Path, PathBuf};

/// The migrations this reims-ctl carries: 1 to this.
pub const CURRENT: u32 = 6;

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
    f.lock().is_ok().then_some(f)
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
    /// Migrations 5 and 6: its GPU's settings replaced by these.
    Gpu(crate::vm::Gpu),
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

/// Migration 5's plan for `vm`, of the host's GPUs `gpus`, asked only where its settings name a
/// GPU of the list by its driver's manifest and name with no IDs: the settings a pick of that
/// GPU from the list gives, where they name it by its IDs and the list has one GPU of that
/// manifest, name and device ID (where the settings keep one).
fn plan_5(vm: &Vm, gpus: impl FnOnce() -> Vec<GpuInfo>) -> Plan {
    let g = &vm.config.gpu;
    let (Some(file), Some(name), None) = (&g.driver_file, &g.name, &g.select) else {
        return Plan::Skip;
    };
    let gpus = gpus();
    let listed: Vec<&GpuInfo> = (gpus.iter())
        .filter(|l| {
            &l.driver_file == file
                && &l.name == name
                && g.device_id.is_none_or(|id| id == l.device_id)
        })
        .collect();
    match listed.as_slice() {
        [only] if only.to_config().select.is_some() => Plan::Gpu(only.to_config()),
        _ => Plan::Skip,
    }
}

/// Migration 6's plan for `vm`, of what each of the host's manifests' probes found, `probes`,
/// asked only where its settings name a GPU by its driver's manifest and name with no IDs: where
/// the probe of that manifest went past it ([`crate::gpu::went_past`]) and found one GPU of that
/// name and device ID (where the settings keep one), the settings a pick of the GPU of its IDs
/// from the list gives, where they name it by its IDs.
fn plan_6(vm: &Vm, probes: impl FnOnce() -> Vec<Vec<GpuInfo>>) -> Plan {
    let g = &vm.config.gpu;
    let (Some(file), Some(name), None) = (&g.driver_file, &g.name, &g.select) else {
        return Plan::Skip;
    };
    let probes = probes();
    let found: Vec<(u32, u32)> = (probes.iter())
        .filter(|p| p.iter().any(|f| &f.driver_file == file) && crate::gpu::went_past(p))
        .flatten()
        .filter(|f| &f.name == name && g.device_id.is_none_or(|id| id == f.device_id))
        .map(|f| (f.vendor_id, f.device_id))
        .collect();
    let [ids] = found.as_slice() else {
        return Plan::Skip;
    };
    let gpus = crate::gpu::from_probes(probes);
    let listed: Vec<&GpuInfo> = (gpus.iter())
        .filter(|l| (l.vendor_id, l.device_id) == *ids)
        .collect();
    match listed.as_slice() {
        [only] if only.to_config().select.is_some() => Plan::Gpu(only.to_config()),
        _ => Plan::Skip,
    }
}

/// What each of the host's manifests' probes found ([`crate::gpu::probes`]), probed at the first
/// ask of a run.
fn host_probes() -> Vec<Vec<GpuInfo>> {
    static PROBED: std::sync::OnceLock<Vec<Vec<GpuInfo>>> = std::sync::OnceLock::new();
    PROBED.get_or_init(crate::gpu::probes).clone()
}

/// The host's GPUs ([`crate::gpu::list`]), from [`host_probes`].
fn host_gpus() -> Vec<GpuInfo> {
    crate::gpu::from_probes(host_probes())
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
        5 => plan_5(vm, host_gpus),
        6 => plan_6(vm, host_probes),
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
        Plan::Gpu(gpu) => Some(match pick_gpu(vm, gpu, n) {
            Ok(line) => format!("{name}: {line}"),
            Err(e) => format!(
                "{name}: left as it was: {e:#}; reims-ctl set {name} --gpu NUMBER picks its GPU \
                 again, from reims-ctl gpus"
            ),
        }),
    }
}

/// Gives `vm` the GPU settings `gpu` and saves them (migration `n`, 5 or 6); what changed, or an
/// error with the VM as it was.
fn pick_gpu(vm: &mut Vm, gpu: crate::vm::Gpu, n: u32) -> Result<String> {
    let before = std::mem::replace(&mut vm.config.gpu, gpu);
    if let Err(e) = vm.save() {
        vm.config.gpu = before;
        return Err(e);
    }
    let g = &vm.config.gpu;
    let why = match n {
        5 => "instead of hiding the host's other GPUs, so its window can be drawn on the GPU \
              your desktop runs on"
            .to_string(),
        _ => format!(
            "instead of by the Vulkan driver file it was saved with, {}, which finds GPUs of \
             other drivers too",
            (before.driver_file.as_deref())
                .and_then(Path::file_name)
                .unwrap_or_default()
                .to_string_lossy()
        ),
    };
    Ok(format!(
        "its GPU, {}, is now picked by its IDs ({}) {why}",
        g.name.as_deref().unwrap_or("?"),
        g.select.as_deref().unwrap_or("?"),
    ))
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

    /// A stand-in for a running VM's QEMU: a shell whose command line names `pidfile`, as QEMU's
    /// does, and which writes its own pid there, as QEMU does. Only then can runtime::pid_at take
    /// it for running: until then its pid holds the test's command line, then `sh`'s, which may
    /// exec another shell, and none during each exec. It loops, since a shell may exec a lone
    /// command in its own place, dropping `pidfile` from its command line. It is returned once
    /// runtime::pid_at sees it.
    #[cfg(unix)]
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

    /// A stand-in for a running VM's QEMU on Windows ([`runtime::test_qemu`]).
    #[cfg(windows)]
    fn running_qemu(pidfile: &Path) -> runtime::TestQemu {
        runtime::test_qemu(pidfile, 600)
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
        let util = crate::testenv::program(&util, &script);
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

    /// A VM whose settings, as a reims-ctl before migration 5 saved a pick from `reims-ctl gpus`,
    /// end with `gpu`, its `[gpu]` table.
    fn vm_with_gpu(dir: &Path, gpu: &str) -> Vm {
        vm(dir, "q35-x86_64", None, "");
        let file = dir.join(CONFIG_FILE);
        let text = std::fs::read_to_string(&file).unwrap() + gpu;
        std::fs::write(&file, text).unwrap();
        Vm::load(dir).unwrap()
    }

    /// A GPU of the host's list.
    fn listed(name: &str, ids: (u32, u32), kind: &str, file: &str) -> GpuInfo {
        GpuInfo {
            name: name.into(),
            vendor_id: ids.0,
            device_id: ids.1,
            kind: kind.into(),
            driver_file: file.into(),
            shares_driver: false,
            shares_ids: false,
            driver_id: 0,
            past_manifest: false,
        }
    }

    const NVIDIA: &str = "/usr/share/vulkan/icd.d/nvidia_icd.json";
    const RADEON: &str = "/usr/share/vulkan/icd.d/radeon_icd.json";
    const LVP: &str = "/usr/share/vulkan/icd.d/lvp_icd.json";
    const AMDVLK: &str = "/usr/share/vulkan/icd.d/amd_icd64.json";

    /// A host's list: an RTX 3090, an RX 7900 XTX and a Radeon 780M on RADV, which drives both,
    /// and lavapipe.
    fn host() -> Vec<GpuInfo> {
        let mut xtx = listed(
            "AMD Radeon RX 7900 XTX (RADV NAVI31)",
            (0x1002, 0x744c),
            "discrete",
            RADEON,
        );
        let mut igpu = listed(
            "AMD Radeon 780M Graphics (RADV PHOENIX)",
            (0x1002, 0x15bf),
            "integrated",
            RADEON,
        );
        xtx.shares_driver = true;
        igpu.shares_driver = true;
        vec![
            listed(
                "NVIDIA GeForce RTX 3090",
                (0x10de, 0x2204),
                "discrete",
                NVIDIA,
            ),
            xtx,
            igpu,
            listed("llvmpipe (LLVM 23.1.4, 256 bits)", (0x10005, 0), "cpu", LVP),
        ]
    }

    /// Migration 5 names a GPU an older reims-ctl gave a VM from the list by its IDs, its device
    /// ID kept where its driver drives several, and tells it in one line; the settings saved,
    /// the next run leaves the VM be without listing the host's GPUs.
    #[test]
    fn migration_5_names_a_gpu_from_the_list_by_its_ids() {
        let env = crate::testenv::Env::new("migrate-gpu");
        let unasked = || -> Vec<GpuInfo> { panic!("the host's GPUs were listed") };
        let dir = env.root.join("nv");
        let mut v = vm_with_gpu(
            &dir,
            "[gpu]\ndriver_file = \"/usr/share/vulkan/icd.d/nvidia_icd.json\"\n\
             name = \"NVIDIA GeForce RTX 3090\"\n",
        );
        let plan = plan_5(&v, host);
        let want = crate::vm::Gpu {
            driver_file: Some(NVIDIA.into()),
            device_id: None,
            select: Some("10de:2204".into()),
            name: Some("NVIDIA GeForce RTX 3090".into()),
        };
        assert_eq!(plan, Plan::Gpu(want.clone()));
        let line = carry_out(&mut v, plan, 5).unwrap();
        assert_eq!(
            line,
            "nv: its GPU, NVIDIA GeForce RTX 3090, is now picked by its IDs (10de:2204) instead \
             of hiding the host's other GPUs, so its window can be drawn on the GPU your desktop \
             runs on"
        );
        let saved = Vm::load(&dir).unwrap();
        assert_eq!(saved.config.gpu, want);
        assert_eq!(plan_of(5, &saved, false), Plan::Skip);
        assert_eq!(plan_5(&saved, unasked), Plan::Skip);

        let dir = env.root.join("xtx");
        let v = vm_with_gpu(
            &dir,
            "[gpu]\ndriver_file = \"/usr/share/vulkan/icd.d/radeon_icd.json\"\n\
             device_id = 29772\nname = \"AMD Radeon RX 7900 XTX (RADV NAVI31)\"\n",
        );
        assert_eq!(
            plan_5(&v, host),
            Plan::Gpu(crate::vm::Gpu {
                driver_file: Some(RADEON.into()),
                device_id: Some(0x744c),
                select: Some("1002:744c".into()),
                name: Some("AMD Radeon RX 7900 XTX (RADV NAVI31)".into()),
            })
        );
    }

    /// Migration 5 leaves a VM's GPU be where IDs cannot name it or the list does not have it:
    /// lavapipe, a GPU another manifest reaches too, a manifest given by its path, a GPU of
    /// another host, and a device ID the list does not have for that GPU; the host's GPUs are
    /// listed only for a VM whose settings name a GPU of the list.
    #[test]
    fn migration_5_leaves_gpus_their_ids_cannot_name() {
        let env = crate::testenv::Env::new("migrate-gpu-kept");
        let unasked = || -> Vec<GpuInfo> { panic!("the host's GPUs were listed") };
        let gpu = |file: &str, extra: &str, name: &str| {
            format!("[gpu]\ndriver_file = \"{file}\"\n{extra}name = \"{name}\"\n")
        };
        let lvp = vm_with_gpu(
            &env.root.join("lvp"),
            &gpu(LVP, "", "llvmpipe (LLVM 23.1.4, 256 bits)"),
        );
        assert_eq!(plan_5(&lvp, host), Plan::Skip);
        let twin = || {
            let mut list = host();
            list[1].shares_ids = true;
            let mut amdvlk = list[1].clone();
            amdvlk.driver_file = AMDVLK.into();
            list.push(amdvlk);
            list
        };
        let xtx = vm_with_gpu(
            &env.root.join("xtx"),
            &gpu(
                RADEON,
                "device_id = 29772\n",
                "AMD Radeon RX 7900 XTX (RADV NAVI31)",
            ),
        );
        assert!(matches!(plan_5(&xtx, host), Plan::Gpu(_)));
        assert_eq!(plan_5(&xtx, twin), Plan::Skip);
        let elsewhere = vm_with_gpu(
            &env.root.join("elsewhere"),
            &gpu(NVIDIA, "", "NVIDIA GeForce RTX 2080"),
        );
        assert_eq!(plan_5(&elsewhere, host), Plan::Skip);
        let other_id = vm_with_gpu(
            &env.root.join("other-id"),
            &gpu(
                RADEON,
                "device_id = 5567\n",
                "AMD Radeon RX 7900 XTX (RADV NAVI31)",
            ),
        );
        assert_eq!(plan_5(&other_id, host), Plan::Skip);
        let path = vm_with_gpu(
            &env.root.join("path"),
            "[gpu]\ndriver_file = \"/opt/mesa/share/vulkan/icd.d/radeon_icd.x86_64.json\"\n",
        );
        assert_eq!(plan_5(&path, unasked), Plan::Skip);
        let auto = vm(&env.root.join("auto"), "q35-x86_64", None, "");
        assert_eq!(plan_5(&auto, unasked), Plan::Skip);
    }

    /// Migration 6 names a GPU saved under a manifest whose probe goes past it by its IDs, as on
    /// a user's Windows host: an Intel GPU and a GTX 1650 Ti, the GTX saved under Intel's
    /// igvk64.json with its device ID and no IDs to pick it, where the probe of Intel's manifest
    /// finds both GPUs. The VM then starts on the GTX's IDs, the loader limited to no driver; the
    /// settings saved, the next run leaves it be without probing. Where NVIDIA's probe found the
    /// GTX too, the pick is from that probe, as the list has it.
    #[test]
    fn migration_6_names_a_gpu_its_manifest_probe_went_past_by_its_ids() {
        let env = crate::testenv::Env::new("migrate-gpu-past");
        let unasked = || -> Vec<Vec<GpuInfo>> { panic!("the host's GPUs were probed") };
        const INTEL: &str = "C:/drivers/intel/igvk64.json";
        const NV: &str = "C:/drivers/nvidia/nv-vk64.json";
        const GTX: &str = "NVIDIA GeForce GTX 1650 Ti";
        // VkDriverId: 4 is NVIDIA's, 5 Intel's on Windows.
        let found = |name: &str, ids: (u32, u32), kind: &str, file: &str, driver_id: u32| GpuInfo {
            driver_id,
            ..listed(name, ids, kind, file)
        };
        let uhd = |file| found("Intel(R) UHD Graphics", (0x8086, 0x9bc4), "integrated", file, 5);
        let gtx = |file| found(GTX, (0x10de, 0x1f95), "discrete", file, 4);
        let windows = crate::host::Host {
            os: crate::host::Os::Windows,
            arch: crate::host::Arch::X86_64,
        };
        let start_env = |v: &Vm| -> std::collections::BTreeMap<String, String> {
            let ctx = crate::qemu::tests::ctx(windows);
            let inv = crate::qemu::invocation(v, &ctx).unwrap();
            inv.env.into_iter().collect()
        };
        let saved_shape = format!(
            "[gpu]\ndriver_file = \"{INTEL}\"\ndevice_id = 8085\nname = \"{GTX}\"\n"
        );

        let dir = env.root.join("mac");
        let mut v = vm_with_gpu(&dir, &saved_shape);
        let before = start_env(&v);
        assert_eq!(before["VK_DRIVER_FILES"], INTEL);
        assert_eq!(before["VK_LOADER_DEVICE_ID_FILTER"], "0x1f95");
        // Intel's probe found both GPUs; NVIDIA's, the GTX alone.
        let host = || vec![vec![uhd(INTEL), gtx(INTEL)], vec![gtx(NV)]];
        let plan = plan_6(&v, host);
        let want = crate::vm::Gpu {
            driver_file: Some(NV.into()),
            device_id: None,
            select: Some("10de:1f95".into()),
            name: Some(GTX.into()),
        };
        assert_eq!(plan, Plan::Gpu(want.clone()));
        assert_eq!(plan_5(&v, || crate::gpu::from_probes(host())), Plan::Skip);
        let line = carry_out(&mut v, plan, 6).unwrap();
        assert_eq!(
            line,
            "mac: its GPU, NVIDIA GeForce GTX 1650 Ti, is now picked by its IDs (10de:1f95) \
             instead of by the Vulkan driver file it was saved with, igvk64.json, which finds \
             GPUs of other drivers too"
        );
        let saved = Vm::load(&dir).unwrap();
        assert_eq!(saved.config.gpu, want);
        let after = start_env(&saved);
        assert_eq!(after["MESA_VK_DEVICE_SELECT"], "10de:1f95");
        assert_eq!(after["VK_LOADER_DEVICE_SELECT"], "0x10de:0x1f95");
        for limit in [
            "VK_DRIVER_FILES",
            "VK_ICD_FILENAMES",
            "VK_LOADER_DEVICE_ID_FILTER",
        ] {
            assert!(!after.contains_key(limit), "{limit} in {after:?}");
        }
        assert_eq!(plan_of(6, &saved, false), Plan::Skip);
        assert_eq!(plan_6(&saved, unasked), Plan::Skip);

        // NVIDIA's probe found both GPUs too: the GTX is listed from Intel's, by its IDs; and so
        // where the settings keep no device ID.
        let v = vm_with_gpu(&env.root.join("every"), &saved_shape);
        let every = || vec![vec![uhd(INTEL), gtx(INTEL)], vec![gtx(NV), uhd(NV)]];
        assert_eq!(
            plan_6(&v, every),
            Plan::Gpu(crate::vm::Gpu {
                driver_file: Some(INTEL.into()),
                device_id: Some(0x1f95),
                select: Some("10de:1f95".into()),
                name: Some(GTX.into()),
            })
        );
        let no_id = vm_with_gpu(
            &env.root.join("no-id"),
            &saved_shape.replace("device_id = 8085\n", ""),
        );
        assert_eq!(plan_6(&no_id, every), plan_6(&v, every));
        // Each probe keeps to its manifest: Intel's finds no GTX, and the VM is left be.
        assert_eq!(
            plan_6(&v, || vec![vec![uhd(INTEL)], vec![gtx(NV)]]),
            Plan::Skip
        );
        // A device ID Intel's probe did not find, and a VM with no GPU chosen, are left be.
        let other = vm_with_gpu(
            &env.root.join("other"),
            &saved_shape.replace("8085", "8086"),
        );
        assert_eq!(plan_6(&other, host), Plan::Skip);
        let auto = vm(&env.root.join("auto"), "q35-x86_64", None, "");
        assert_eq!(plan_6(&auto, unasked), Plan::Skip);
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
