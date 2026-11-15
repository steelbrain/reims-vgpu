//! The host's GPUs a VM can be given.
//!
//! On Linux the library runs on Vulkan. Of the host's GPUs it opens the only one, or, of several,
//! the one that ran a short workload fastest, timed once for each set of GPUs and drivers
//! (reims-vgpu-vulkan's `device_choice`), unless the host's user picked one. The list comes from
//! asking each driver's manifest (an ICD JSON file) in a child process (`reims-ctl
//! __vulkan-probe`), so a driver that hangs or crashes costs only that one.
//!
//! **A VM's GPU.** A GPU picked from the list is named by its vendor and device IDs, in
//! `MESA_VK_DEVICE_SELECT` and `VK_LOADER_DEVICE_SELECT` ([`crate::vm::Gpu::select`]): both list
//! that GPU first without hiding the others, and the library opens the first. Every driver still
//! loads, so the library also sees the GPU the desktop composites the window on, and where that
//! is another GPU it presents the window from a device of its own there, instead of sending every
//! frame across from the pick. Mesa's device-select layer reads the first; it sorts the GPUs after
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
}

impl GpuInfo {
    pub fn label(&self) -> String {
        let file = self
            .driver_file
            .file_name()
            .map(|f| f.to_string_lossy().into_owned())
            .unwrap_or_default();
        format!("{} ({}, {file})", self.name, self.kind)
    }

    /// The VM's settings for this GPU: named by its IDs, unless it is a CPU implementation or
    /// another manifest reaches a GPU of the same IDs, which its driver's manifest names alone
    /// (the module doc).
    pub fn to_config(&self) -> crate::vm::Gpu {
        let by_ids = self.kind != "cpu" && !self.shares_ids;
        crate::vm::Gpu {
            driver_file: Some(self.driver_file.clone()),
            device_id: self.shares_driver.then_some(self.device_id),
            select: by_ids.then(|| format!("{:04x}:{:04x}", self.vendor_id, self.device_id)),
            name: Some(self.name.clone()),
        }
    }
}

/// The Vulkan driver manifests the loader would read, in its search order (the Khronos loader's
/// Linux search paths).
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
    if !cfg!(target_os = "linux") {
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
    let mut gpus: Vec<GpuInfo> = handles
        .into_iter()
        .filter_map(|h| h.join().ok())
        .flatten()
        .collect();
    // One GPU can be reached through two manifests (a 32-bit and a 64-bit one, say): keep the
    // first.
    let mut seen = Vec::new();
    gpus.retain(|g| {
        let key = (
            g.vendor_id,
            g.device_id,
            g.name.clone(),
            g.driver_file.file_name().map(|f| f.to_os_string()),
        );
        if seen.contains(&key) {
            false
        } else {
            seen.push(key);
            true
        }
    });
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
#[cfg(target_os = "linux")]
pub fn probe_main() -> i32 {
    use ash::vk;
    // SAFETY: loads the system's Vulkan loader; the calls below follow the API's rules.
    unsafe {
        let Ok(entry) = ash::Entry::load() else {
            return 1;
        };
        let app = vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 1, 0));
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
                let info = GpuInfo {
                    name,
                    vendor_id: p.vendor_id,
                    device_id: p.device_id,
                    kind: kind.into(),
                    driver_file: PathBuf::new(),
                    shares_driver: false,
                    shares_ids: false,
                };
                println!("{}", serde_json::to_string(&info).unwrap());
            }
        }
        instance.destroy_instance(None);
    }
    0
}

#[cfg(not(target_os = "linux"))]
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
}
