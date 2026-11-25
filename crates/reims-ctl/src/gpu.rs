//! The host's GPUs a VM can be given.
//!
//! On Linux and Windows the library runs on Vulkan. Of the host's GPUs it opens the one the host's
//! user picked, else the only one, else the first in the runtime's order of kinds, discrete
//! before integrated; it times none (reims-vgpu-vulkan's `device_choice`, `chosen_untimed`). The
//! list comes from asking each driver's manifest (an ICD JSON file) in a child process
//! (`reims-ctl __vulkan-probe`), so a driver that hangs or crashes costs only that one. The
//! manifests are the loader's: on Linux in its search paths, on Windows in the registry
//! ([`icd_files`]).
//!
//! **A VM's GPU.** A GPU picked from the list is named by its vendor and device IDs, in
//! `MESA_VK_DEVICE_SELECT` and `VK_LOADER_DEVICE_SELECT` ([`crate::vm::Gpu::select`]): both list
//! that GPU first without hiding the others, and the library opens the one they name. Its window
//! presents from that GPU, the process's one device, wherever the desktop composites it: where
//! that is another GPU, each frame crosses to it through the window system (reims-vgpu's
//! `graphics/rt_window.rs`). Mesa's device-select layer reads the first; it sorts the GPUs after
//! the loader does and puts its own default first, so the loader's variable alone leaves that
//! default first. The loader reads the second, on a host without the layer. **Measured** on a host
//! with an RTX 3090 and a Radeon 780M (loader 1.4.363, Mesa 26.2.6's layer, 2026-11-14): the
//! loader's variable alone listed the 780M first, the layer's variable or both the 3090, and
//! both `10005:0` lavapipe. **Read** in the loader's v1.3.204 source: it reads
//! `VK_LOADER_DEVICE_SELECT`.
//!
//! The loader is limited to the GPU's driver's manifest instead (`VK_DRIVER_FILES`, and, where
//! that driver drives several GPUs, the PCI device ID, `VK_LOADER_DEVICE_ID_FILTER`) where IDs
//! cannot name the pick ([`GpuInfo::to_config`]):
//! - a CPU implementation such as lavapipe, which keeps every GPU's driver out of the VM's
//!   process;
//! - a GPU another manifest reaches too (RADV and AMDVLK on one card): its IDs do not say which
//!   driver;
//! - a manifest given by its path (`--gpu FILE.json`), which may be a driver outside the
//!   loader's search paths, or a second driver for a GPU the loader reaches by its own.
//!
//! **A probe past its manifest.** A manifest names one driver, so a probe that finds GPUs of
//! several drivers (by `VkPhysicalDeviceDriverProperties::driverID`, else by vendor) shows a
//! loader that did not keep to it, nor would keep a VM's process to it: its GPUs are named by
//! their IDs alone ([`GpuInfo::past_manifest`]), and one another manifest's probe found is listed
//! from that probe. **Seen** in a user's report from a Windows host with an Intel GPU and a GTX
//! 1650 Ti: the probe of Intel's manifest found both GPUs, so the GTX was saved under that
//! manifest, and QEMU's run limited to it found both too. Why is not confirmed; a loader that
//! ignores `VK_DRIVER_FILES` in an elevated process would do it. The library reads
//! `MESA_VK_DEVICE_SELECT` itself (reims-vgpu-vulkan's `device_choice`), so IDs name the pick
//! wherever the loader runs. A VM an older reims-ctl gave a GPU so, under a manifest whose probe
//! now goes past it, is named by that GPU's IDs once ([`crate::migrate`], migration 6).

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GpuInfo {
    pub name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    /// discrete, integrated, virtual, cpu or other.
    pub kind: String,
    pub driver_file: PathBuf,
    /// Whether another GPU shares the driver (the device ID then picks it).
    #[serde(default)]
    pub shares_driver: bool,
    /// Whether another manifest reaches a GPU of the same vendor and device IDs.
    #[serde(default)]
    pub shares_ids: bool,
    /// The driver's `VkDriverId`, where it says (Vulkan 1.2): 0 where it does not.
    #[serde(default)]
    pub driver_id: u32,
    /// Whether the probe of [`GpuInfo::driver_file`] found GPUs of other drivers too, so that
    /// manifest is not known to be this GPU's (the module doc): only its IDs name it.
    #[serde(default)]
    pub past_manifest: bool,
}

impl GpuInfo {
    pub fn label(&self) -> String {
        if self.past_manifest {
            return format!("{} ({})", self.name, self.kind);
        }
        let file = self
            .driver_file
            .file_name()
            .map(|f| f.to_string_lossy().into_owned())
            .unwrap_or_default();
        format!("{} ({}, {file})", self.name, self.kind)
    }

    /// The VM's settings for this GPU: named by its IDs, unless it is a CPU implementation or
    /// another manifest reaches a GPU of the same IDs, which its driver's manifest names alone;
    /// by its IDs wherever its probe went past its manifest (the module doc).
    pub fn to_config(&self) -> crate::vm::Gpu {
        let by_ids = self.past_manifest || (self.kind != "cpu" && !self.shares_ids);
        crate::vm::Gpu {
            driver_file: Some(self.driver_file.clone()),
            device_id: self.shares_driver.then_some(self.device_id),
            select: by_ids.then(|| format!("{:04x}:{:04x}", self.vendor_id, self.device_id)),
            name: Some(self.name.clone()),
        }
    }
}

/// The Vulkan driver manifests the loader would read, in its search order (the Khronos loader's
/// Windows registry keys, [`crate::os::vulkan_driver_files`]).
#[cfg(windows)]
pub fn icd_files() -> Vec<PathBuf> {
    crate::os::vulkan_driver_files()
}

/// The Vulkan driver manifests the loader would read, in its search order (the Khronos loader's
/// Linux search paths).
#[cfg(not(windows))]
pub fn icd_files() -> Vec<PathBuf> {
    let home = crate::paths::home();
    let env_list = |var: &str, default: &str| -> Vec<PathBuf> {
        std::env::var(var)
            .ok()
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| default.to_string())
            .split(':')
            .filter(|s| !s.is_empty())
            .map(PathBuf::from)
            .collect()
    };
    let mut dirs = Vec::new();
    dirs.push(
        std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| home.join(".config")),
    );
    dirs.extend(env_list("XDG_CONFIG_DIRS", "/etc/xdg"));
    dirs.push("/etc".into());
    dirs.push(
        std::env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| home.join(".local/share")),
    );
    dirs.extend(env_list("XDG_DATA_DIRS", "/usr/local/share:/usr/share"));
    let mut files = Vec::new();
    for d in dirs {
        let Ok(rd) = std::fs::read_dir(d.join("vulkan/icd.d")) else {
            continue;
        };
        let mut here: Vec<PathBuf> = rd
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "json"))
            .collect();
        here.sort();
        for p in here {
            if !files.contains(&p) {
                files.push(p);
            }
        }
    }
    files
}

/// The host's GPUs, each with the driver manifest that reaches it.
pub fn list() -> Vec<GpuInfo> {
    from_probes(probes())
}

/// What each driver's manifest's probe found, in the loader's order of manifests: the GPUs, each
/// with that manifest as its [`GpuInfo::driver_file`].
pub fn probes() -> Vec<Vec<GpuInfo>> {
    if !cfg!(any(target_os = "linux", windows)) {
        return Vec::new();
    }
    let Ok(exe) = std::env::current_exe() else {
        return Vec::new();
    };
    let handles: Vec<_> = icd_files()
        .into_iter()
        .map(|file| {
            let exe = exe.clone();
            std::thread::spawn(move || probe(&exe, &file))
        })
        .collect();
    handles.into_iter().filter_map(|h| h.join().ok()).collect()
}

/// Whether `probe`, what one manifest's probe found, holds GPUs of several drivers (by
/// `driverID`, else by vendor), so it went past its manifest (the module doc).
pub fn went_past(probe: &[GpuInfo]) -> bool {
    let driver = |g: &GpuInfo| match g.driver_id {
        0 => (0, g.vendor_id),
        id => (id, 0),
    };
    probe.iter().any(|g| driver(g) != driver(&probe[0]))
}

/// The host's GPUs from each manifest's probe, in the loader's order of manifests: a probe that
/// found several drivers' GPUs marked [`GpuInfo::past_manifest`], and each GPU listed once,
/// discrete before integrated.
pub fn from_probes(probes: Vec<Vec<GpuInfo>>) -> Vec<GpuInfo> {
    let mut found: Vec<GpuInfo> = Vec::new();
    for mut probe in probes {
        let past = went_past(&probe);
        for g in &mut probe {
            g.past_manifest = past;
        }
        found.extend(probe);
    }
    // Those found through their own manifest first. One GPU can be reached through two manifests
    // (a 32-bit and a 64-bit one, say): keep the first. One found past its manifest is kept only
    // where none kept has its IDs, which alone name it.
    found.sort_by_key(|g| g.past_manifest);
    let mut gpus: Vec<GpuInfo> = Vec::new();
    for g in found {
        let ids = (g.vendor_id, g.device_id);
        let seen = gpus.iter().any(|k| {
            (k.vendor_id, k.device_id) == ids
                && (g.past_manifest
                    || (k.name == g.name && k.driver_file.file_name() == g.driver_file.file_name()))
        });
        if !seen {
            gpus.push(g);
        }
    }
    mark_shared(&mut gpus);
    let rank = |k: &str| match k {
        "discrete" => 0,
        "integrated" => 1,
        "virtual" => 2,
        "other" => 3,
        _ => 4,
    };
    gpus.sort_by_key(|g| rank(&g.kind));
    gpus
}

/// Marks each of `gpus` that shares its driver with another ([`GpuInfo::shares_driver`]), and each
/// whose IDs another manifest reaches too ([`GpuInfo::shares_ids`]).
fn mark_shared(gpus: &mut [GpuInfo]) {
    let shared: Vec<(bool, bool)> = gpus
        .iter()
        .map(|g| {
            let others = || gpus.iter().filter(|o| !std::ptr::eq(*o, g));
            (
                others().any(|o| o.driver_file == g.driver_file),
                others().any(|o| {
                    o.driver_file != g.driver_file
                        && (o.vendor_id, o.device_id) == (g.vendor_id, g.device_id)
                }),
            )
        })
        .collect();
    for (g, (driver, ids)) in gpus.iter_mut().zip(shared) {
        g.shares_driver = driver;
        g.shares_ids = ids;
    }
}

fn probe(exe: &std::path::Path, file: &std::path::Path) -> Vec<GpuInfo> {
    use std::process::{Command, Stdio};
    let child = Command::new(exe)
        .arg("__vulkan-probe")
        .env("VK_DRIVER_FILES", file)
        .env("VK_ICD_FILENAMES", file)
        .env("VK_LOADER_LAYERS_DISABLE", "~all~")
        .env_remove("VK_LOADER_DEVICE_ID_FILTER")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn();
    let Ok(mut child) = child else {
        return Vec::new();
    };
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if std::time::Instant::now() < deadline => {
                std::thread::sleep(std::time::Duration::from_millis(50))
            }
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return Vec::new();
            }
        }
    }
    let mut out = String::new();
    use std::io::Read;
    let _ = child.stdout.take().map(|mut s| s.read_to_string(&mut out));
    out.lines()
        .filter_map(|l| serde_json::from_str::<GpuInfo>(l).ok())
        .map(|mut g| {
            g.driver_file = file.to_path_buf();
            g
        })
        .collect()
}

/// `reims-ctl __vulkan-probe`: one JSON line per physical device the loader finds.
#[cfg(any(target_os = "linux", windows))]
pub fn probe_main() -> i32 {
    use ash::vk;
    // SAFETY: loads the system's Vulkan loader; the calls below follow the API's rules.
    unsafe {
        let Ok(entry) = ash::Entry::load() else {
            return 1;
        };
        let app = vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 2, 0));
        let ci = vk::InstanceCreateInfo::default().application_info(&app);
        let Ok(instance) = entry.create_instance(&ci, None) else {
            return 1;
        };
        if let Ok(devices) = instance.enumerate_physical_devices() {
            for pd in devices {
                let p = instance.get_physical_device_properties(pd);
                let name = p
                    .device_name_as_c_str()
                    .map(|c| c.to_string_lossy().into_owned())
                    .unwrap_or_default();
                let kind = match p.device_type {
                    vk::PhysicalDeviceType::DISCRETE_GPU => "discrete",
                    vk::PhysicalDeviceType::INTEGRATED_GPU => "integrated",
                    vk::PhysicalDeviceType::VIRTUAL_GPU => "virtual",
                    vk::PhysicalDeviceType::CPU => "cpu",
                    _ => "other",
                };
                // VkPhysicalDeviceDriverProperties is Vulkan 1.2's.
                let v = p.api_version;
                let driver_id = if vk::api_version_major(v) > 1 || vk::api_version_minor(v) >= 2 {
                    let mut driver = vk::PhysicalDeviceDriverProperties::default();
                    let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut driver);
                    instance.get_physical_device_properties2(pd, &mut p2);
                    driver.driver_id.as_raw() as u32
                } else {
                    0
                };
                let info = GpuInfo {
                    name,
                    vendor_id: p.vendor_id,
                    device_id: p.device_id,
                    kind: kind.into(),
                    driver_file: PathBuf::new(),
                    shares_driver: false,
                    shares_ids: false,
                    driver_id,
                    past_manifest: false,
                };
                println!("{}", serde_json::to_string(&info).unwrap());
            }
        }
        instance.destroy_instance(None);
    }
    0
}

#[cfg(not(any(target_os = "linux", windows)))]
pub fn probe_main() -> i32 {
    1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu(name: &str, ids: (u32, u32), kind: &str, file: &str) -> GpuInfo {
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

    /// RADV drives two GPUs, and AMDVLK reaches one of them too; NVIDIA's driver and lavapipe
    /// one each.
    #[test]
    fn a_shared_driver_and_shared_ids_are_marked() {
        let mut gpus = vec![
            gpu("RTX 3090", (0x10de, 0x2204), "discrete", "/icd/nvidia.json"),
            gpu(
                "XTX (RADV)",
                (0x1002, 0x744c),
                "discrete",
                "/icd/radeon.json",
            ),
            gpu(
                "780M (RADV)",
                (0x1002, 0x15bf),
                "integrated",
                "/icd/radeon.json",
            ),
            gpu(
                "XTX (AMDVLK)",
                (0x1002, 0x744c),
                "discrete",
                "/icd/amd.json",
            ),
            gpu("llvmpipe", (0x10005, 0), "cpu", "/icd/lvp.json"),
        ];
        mark_shared(&mut gpus);
        let marks: Vec<(bool, bool)> = gpus
            .iter()
            .map(|g| (g.shares_driver, g.shares_ids))
            .collect();
        assert_eq!(
            marks,
            [
                (false, false),
                (true, true),
                (true, false),
                (false, true),
                (false, false)
            ]
        );
    }

    /// A GPU of the list is named by its IDs, its device ID kept where its driver drives several;
    /// a CPU implementation and a GPU whose IDs another manifest reaches too, by their driver's
    /// manifest alone (the module doc).
    #[test]
    fn a_pick_is_named_by_its_ids_where_they_name_it() {
        let mut g = gpu(
            "NVIDIA GeForce RTX 3090",
            (0x10de, 0x2204),
            "discrete",
            "/icd/nvidia.json",
        );
        assert_eq!(
            g.to_config(),
            crate::vm::Gpu {
                driver_file: Some("/icd/nvidia.json".into()),
                device_id: None,
                select: Some("10de:2204".into()),
                name: Some("NVIDIA GeForce RTX 3090".into()),
            }
        );
        g.shares_driver = true;
        assert_eq!(g.to_config().device_id, Some(0x2204));
        assert_eq!(g.to_config().select.as_deref(), Some("10de:2204"));
        g.shares_ids = true;
        assert_eq!(g.to_config().select, None);
        assert_eq!(
            g.to_config().driver_file.as_deref(),
            Some(std::path::Path::new("/icd/nvidia.json"))
        );
        let lvp = gpu("llvmpipe", (0x10005, 0), "cpu", "/icd/lvp.json");
        assert_eq!(lvp.to_config().select, None);
        let small = gpu("virtio", (0x1af4, 0x50), "virtual", "/icd/virtio.json");
        assert_eq!(small.to_config().select.as_deref(), Some("1af4:0050"));
        assert_eq!(small.to_config().selected().unwrap(), Some((0x1af4, 0x50)));
    }

    /// A probe that found several drivers' GPUs went past its manifest: its GPUs are named by
    /// their IDs, and one a probe of its own manifest found is listed from that probe. Seen on a
    /// Windows host, where the probe of Intel's manifest found the GTX too and saved it under
    /// Intel's manifest.
    #[test]
    fn gpus_a_probe_found_past_its_manifest_are_named_by_their_ids() {
        let found = |name: &str, ids: (u32, u32), kind: &str, file: &str, driver_id: u32| GpuInfo {
            driver_id,
            ..gpu(name, ids, kind, file)
        };
        let (intel, nvidia) = ("/icd/igvk64.json", "/icd/nv-vk64.json");
        let uhd = |file| found("Intel UHD", (0x8086, 0x9bc4), "integrated", file, 5);
        let gtx = |file| found("GTX 1650 Ti", (0x10de, 0x1f95), "discrete", file, 4);
        let picks = |gpus: &[GpuInfo]| {
            gpus.iter()
                .map(|g| {
                    let c = g.to_config();
                    (g.name.clone(), c.select, c.driver_file, g.past_manifest)
                })
                .collect::<Vec<_>>()
        };
        let pick = |name: &str, select: &str, file: &str, past| {
            (
                name.to_owned(),
                Some(select.to_owned()),
                Some(PathBuf::from(file)),
                past,
            )
        };
        // Every probe found both GPUs: each is listed once, by its IDs.
        let every = from_probes(vec![
            vec![uhd(intel), gtx(intel)],
            vec![gtx(nvidia), uhd(nvidia)],
        ]);
        assert_eq!(
            picks(&every),
            [
                pick("GTX 1650 Ti", "10de:1f95", intel, true),
                pick("Intel UHD", "8086:9bc4", intel, true)
            ]
        );
        assert_eq!(every[0].label(), "GTX 1650 Ti (discrete)");
        assert_ne!(
            every[0].to_config(),
            every[1].to_config(),
            "the picks differ"
        );
        // NVIDIA's probe kept to its manifest: the GTX is listed from it.
        let one = from_probes(vec![vec![uhd(intel), gtx(intel)], vec![gtx(nvidia)]]);
        assert_eq!(
            picks(&one),
            [
                pick("GTX 1650 Ti", "10de:1f95", nvidia, false),
                pick("Intel UHD", "8086:9bc4", intel, true)
            ]
        );
        // Each probe kept to its manifest: as before.
        let both = from_probes(vec![vec![uhd(intel)], vec![gtx(nvidia)]]);
        assert_eq!(
            picks(&both),
            [
                pick("GTX 1650 Ti", "10de:1f95", nvidia, false),
                pick("Intel UHD", "8086:9bc4", intel, false)
            ]
        );
        assert_eq!(both[0].label(), "GTX 1650 Ti (discrete, nv-vk64.json)");
        // Drivers that do not say their ID are told apart by vendor; two manifests of one name
        // (a 32-bit and a 64-bit one) list a GPU once.
        let unsaid = from_probes(vec![
            vec![found("A", (0x1002, 1), "discrete", "/a/x.json", 0)],
            vec![found("A", (0x1002, 1), "discrete", "/b/x.json", 0)],
            vec![
                found("B", (0x10de, 2), "discrete", "/c/y.json", 0),
                found("C", (0x8086, 3), "integrated", "/c/y.json", 0),
            ],
        ]);
        assert_eq!(
            picks(&unsaid),
            [
                pick("A", "1002:0001", "/a/x.json", false),
                pick("B", "10de:0002", "/c/y.json", true),
                pick("C", "8086:0003", "/c/y.json", true)
            ]
        );
    }
}
