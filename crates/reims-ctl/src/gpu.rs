//! The host's GPUs a VM can be given.
//!
//! On Linux the library runs on Vulkan, and picks the first discrete GPU, then the first
//! integrated one, in the loader's order. A VM can name another: its Vulkan driver's manifest
//! (an ICD JSON file, which `VK_DRIVER_FILES` limits the loader to) and, where that driver
//! drives several GPUs, the PCI device ID (`VK_LOADER_DEVICE_ID_FILTER`). The list comes from
//! asking each manifest's driver in a child process (`reims-ctl __vulkan-probe`), so a driver
//! that hangs or crashes costs only that one.

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

    pub fn to_config(&self) -> crate::vm::Gpu {
        crate::vm::Gpu {
            driver_file: Some(self.driver_file.clone()),
            device_id: self.shares_driver.then_some(self.device_id),
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
    let counts: Vec<usize> = gpus
        .iter()
        .map(|g| {
            gpus.iter()
                .filter(|o| o.driver_file == g.driver_file)
                .count()
        })
        .collect();
    for (g, n) in gpus.iter_mut().zip(counts) {
        g.shares_driver = n > 1;
    }
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
