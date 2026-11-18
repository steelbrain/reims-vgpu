//! The host reims-ctl runs on, and the machine profiles it can run there.
//!
//! A profile is one QEMU machine for one guest architecture: its binary, accelerator, devices
//! and what it can attach. Each host supports the profiles [`Profile::supported_on`] lists:
//! - Linux and Windows on x86-64: [`Profile::Q35X86_64`], `q35` booted through OpenCore, with
//!   reims-vgpu's own GPU device, on KVM on Linux and on the Windows Hypervisor Platform (WHPX)
//!   on Windows;
//! - an arm64 Mac, arm64 Linux and Linux on x86-64: [`Profile::VmappleArm64`], QEMU's `vmapple`
//!   booted by Ambrosia (ambrosia.rs), with reims-vgpu's own GPU device, whose VMs reims-ctl
//!   makes by restoring macOS from an IPSW (restore.rs): on Hypervisor.framework (HVF) on a Mac,
//!   on KVM on arm64 Linux, and emulated by QEMU's TCG on x86-64, where an arm64 guest runs
//!   several times slower than on an arm64 host. On Linux on x86-64 it is the second profile, an
//!   IPSW's, and q35 the default.
//!
//! Any other host runs none ([`UNSUPPORTED`]). Windows on x86-64 runs no arm64 VM: its QEMU is
//! built for x86-64 guests alone (scripts/install-ctl.sh).
//!
//! Another host or guest architecture is another profile (x86-64 guests on an Intel Mac would be
//! `q35` on HVF, without the memfd RAM only Linux has): add it here and in `qemu.rs`.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

/// What reims-ctl tells a host it runs no VMs on, for a create or a start there.
pub const UNSUPPORTED: &str = "reims-ctl runs x86-64 VMs on Linux and Windows x86-64 hosts, \
                               and arm64 VMs on arm64 Macs, arm64 Linux and Linux on x86-64";

/// What reims-ctl tells of the work only x86-64 VMs have (USB devices, VMware SVGA).
pub const X86_64_ONLY: &str = "that's for x86-64 VMs, which run on Linux and Windows x86-64 hosts";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Os {
    Mac,
    Linux,
    Windows,
    Other,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arch {
    Aarch64,
    X86_64,
    Other,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Host {
    pub os: Os,
    pub arch: Arch,
}

impl Host {
    pub fn current() -> Host {
        let os = if cfg!(target_os = "macos") {
            Os::Mac
        } else if cfg!(target_os = "linux") {
            Os::Linux
        } else if cfg!(windows) {
            Os::Windows
        } else {
            Os::Other
        };
        let arch = if cfg!(target_arch = "aarch64") {
            Arch::Aarch64
        } else if cfg!(target_arch = "x86_64") {
            Arch::X86_64
        } else {
            Arch::Other
        };
        Host { os, arch }
    }

    pub fn default_profile(self) -> Option<Profile> {
        Profile::supported_on(self).first().copied()
    }
}

/// Whether KVM here offers VMX to its guests: `kvm_intel` is loaded, which only a CPU with VMX
/// allows, and its `nested` is on, Linux's default since 4.20. AMD's KVM offers SVM instead,
/// which macOS's Hypervisor.framework does not drive, and HVF offers neither.
pub fn kvm_nested_vmx() -> bool {
    std::fs::read_to_string("/sys/module/kvm_intel/parameters/nested")
        .is_ok_and(|s| matches!(s.trim(), "Y" | "1"))
}

/// Whether a window can open for a QEMU run with `env` over this process's environment: it names
/// a Wayland compositor or an X server. Without one, QEMU's GTK window ends QEMU as it starts,
/// where the library's runs headless.
#[cfg(unix)]
pub fn has_screen(env: &BTreeMap<String, String>) -> bool {
    ["WAYLAND_DISPLAY", "DISPLAY"]
        .iter()
        .any(|v| match env.get(*v) {
            Some(s) => !s.is_empty(),
            None => std::env::var_os(v).is_some_and(|s| !s.is_empty()),
        })
}

/// Whether a window can open for a QEMU run: on Windows, this process's window station is the
/// one a signed-in user's desktop shows, as it is not for a service's or an SSH login's.
#[cfg(windows)]
pub fn has_screen(_env: &BTreeMap<String, String>) -> bool {
    crate::os::desktop_shown()
}

impl fmt::Display for Host {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let os = match self.os {
            Os::Mac => "macOS",
            Os::Linux => "Linux",
            Os::Windows => "Windows",
            Os::Other => std::env::consts::OS,
        };
        let arch = match self.arch {
            Arch::Aarch64 => "arm64",
            Arch::X86_64 => "x86-64",
            Arch::Other => std::env::consts::ARCH,
        };
        write!(f, "{os} on {arch}")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Profile {
    #[serde(rename = "q35-x86_64")]
    Q35X86_64,
    /// An arm64 macOS guest on QEMU's `vmapple`, the id the arm64 VMs of an earlier reims-ctl
    /// and the guests Ambrosia's boot check reads have in their settings.
    #[serde(rename = "vmapple-arm64")]
    VmappleArm64,
}

impl Profile {
    pub fn supported_on(host: Host) -> &'static [Profile] {
        match (host.os, host.arch) {
            (Os::Linux, Arch::X86_64) => &[Profile::Q35X86_64, Profile::VmappleArm64],
            (Os::Windows, Arch::X86_64) => &[Profile::Q35X86_64],
            (Os::Mac | Os::Linux, Arch::Aarch64) => &[Profile::VmappleArm64],
            _ => &[],
        }
    }

    pub fn id(self) -> &'static str {
        match self {
            Profile::Q35X86_64 => "q35-x86_64",
            Profile::VmappleArm64 => "vmapple-arm64",
        }
    }

    /// What it is, with its accelerator on `host`.
    pub fn describe(self, host: Host) -> &'static str {
        match (self, self.accel(host)) {
            (Profile::Q35X86_64, "whpx") => "x86-64 macOS guest (q35 and OpenCore, WHPX)",
            (Profile::Q35X86_64, _) => "x86-64 macOS guest (q35 and OpenCore, KVM)",
            (Profile::VmappleArm64, "hvf") => "arm64 macOS guest (vmapple and Ambrosia, HVF)",
            (Profile::VmappleArm64, "kvm") => "arm64 macOS guest (vmapple and Ambrosia, KVM)",
            (Profile::VmappleArm64, _) => {
                "arm64 macOS guest (vmapple and Ambrosia, emulated by TCG)"
            }
        }
    }

    /// The QEMU binary's name in the install's `bin`: with `.exe` on Windows.
    pub fn qemu_binary(self) -> &'static str {
        match self {
            Profile::Q35X86_64 if cfg!(windows) => "qemu-system-x86_64.exe",
            Profile::Q35X86_64 => "qemu-system-x86_64",
            Profile::VmappleArm64 => "qemu-system-aarch64",
        }
    }

    /// The accelerator on `host`: for an x86-64 guest KVM, and on Windows the Windows Hypervisor
    /// Platform; for an arm64 guest Hypervisor.framework on a Mac, KVM on arm64 Linux, and TCG,
    /// QEMU's emulation, on an x86-64 host.
    pub fn accel(self, host: Host) -> &'static str {
        match (self, host.os, host.arch) {
            (Profile::Q35X86_64, Os::Windows, _) => "whpx",
            (Profile::Q35X86_64, ..) => "kvm",
            (Profile::VmappleArm64, Os::Mac, _) => "hvf",
            (Profile::VmappleArm64, _, Arch::Aarch64) => "kvm",
            (Profile::VmappleArm64, ..) => "tcg",
        }
    }

    /// QEMU's `-accel` for it on `host`: [`Profile::accel`] with its options. An arm64 guest runs
    /// with the GIC in the kernel on HVF, and on TCG a thread for each vCPU; these are the options
    /// Ambrosia's boot check runs it with (firmware/ambrosia/check-boot.sh).
    pub fn accel_arg(self, host: Host) -> &'static str {
        match (self, self.accel(host)) {
            (Profile::VmappleArm64, "hvf") => "hvf,kernel-irqchip=on",
            (Profile::VmappleArm64, "tcg") => "tcg,thread=multi",
            (_, accel) => accel,
        }
    }

    /// Whether the guest's RAM must be a shared memfd: the library on Linux builds its views of
    /// guest pages from the file behind the RAM, and refuses RAM without one.
    pub fn memfd_ram(self, host: Host) -> bool {
        host.os == Os::Linux
    }

    /// Whether the guest's RAM must be shared memory of QEMU's RAM backend (`share=on`), which
    /// reims-vgpu's QEMU makes a pagefile-backed section on Windows: the library maps its views
    /// of guest pages from that section, as it does from the memfd on Linux.
    pub fn section_ram(self, host: Host) -> bool {
        host.os == Os::Windows
    }

    /// Whether a Vulkan GPU can be chosen: the library runs on the host's Vulkan GPUs on Linux
    /// and Windows, and on a Mac on MoltenVK, over the Mac's one GPU.
    pub fn gpu_choice(self, host: Host) -> bool {
        matches!(host.os, Os::Linux | Os::Windows)
    }
}

impl fmt::Display for Profile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Linux on x86-64 runs x86-64 guests on KVM in memfd RAM, and arm64 guests on vmapple under
    /// TCG; an arm64 Mac and arm64 Linux run arm64 guests on vmapple, under HVF and KVM; an Intel
    /// Mac, Windows on arm64 and any other host run none.
    #[test]
    fn each_host_has_its_profile() {
        let linux = Host {
            os: Os::Linux,
            arch: Arch::X86_64,
        };
        assert_eq!(linux.default_profile(), Some(Profile::Q35X86_64));
        assert_eq!(
            Profile::supported_on(linux),
            [Profile::Q35X86_64, Profile::VmappleArm64]
        );
        assert!(Profile::Q35X86_64.memfd_ram(linux));
        assert!(!Profile::Q35X86_64.section_ram(linux));
        assert_eq!(Profile::Q35X86_64.accel(linux), "kvm");
        assert_eq!(Profile::Q35X86_64.accel_arg(linux), "kvm");
        let arm = Profile::VmappleArm64;
        assert_eq!(arm.accel(linux), "tcg");
        assert_eq!(arm.accel_arg(linux), "tcg,thread=multi");
        assert!(arm.memfd_ram(linux) && arm.gpu_choice(linux));
        assert!(arm.describe(linux).ends_with("emulated by TCG)"));
        for os in [Os::Mac, Os::Linux] {
            let host = Host {
                os,
                arch: Arch::Aarch64,
            };
            assert_eq!(
                Profile::supported_on(host),
                [Profile::VmappleArm64],
                "{os:?}"
            );
        }
        let (mac, arm_linux) = (
            Host {
                os: Os::Mac,
                arch: Arch::Aarch64,
            },
            Host {
                os: Os::Linux,
                arch: Arch::Aarch64,
            },
        );
        assert_eq!(arm.accel(mac), "hvf");
        assert_eq!(arm.accel_arg(mac), "hvf,kernel-irqchip=on");
        assert!(!arm.memfd_ram(mac) && !arm.gpu_choice(mac));
        assert!(arm.describe(mac).ends_with("HVF)"));
        assert_eq!(arm.accel(arm_linux), "kvm");
        assert_eq!(arm.accel_arg(arm_linux), "kvm");
        assert!(arm.memfd_ram(arm_linux) && arm.gpu_choice(arm_linux));
        assert!(arm.describe(arm_linux).ends_with("KVM)"));
        assert_eq!(arm.qemu_binary(), "qemu-system-aarch64");
        for (os, arch) in [
            (Os::Mac, Arch::X86_64),
            (Os::Windows, Arch::Aarch64),
            (Os::Other, Arch::Aarch64),
        ] {
            assert_eq!(Host { os, arch }.default_profile(), None, "{os:?} {arch:?}");
        }
    }

    /// Windows on x86-64 runs x86-64 guests on WHPX, in RAM of a shared section rather than a
    /// memfd, with a Vulkan GPU to choose.
    #[test]
    fn windows_on_x86_64_runs_vms_on_whpx() {
        let windows = Host {
            os: Os::Windows,
            arch: Arch::X86_64,
        };
        let q35 = Profile::Q35X86_64;
        assert_eq!(Profile::supported_on(windows), [q35]);
        assert_eq!(q35.accel(windows), "whpx");
        assert!(q35.section_ram(windows) && !q35.memfd_ram(windows));
        assert!(q35.gpu_choice(windows));
        assert!(q35.describe(windows).ends_with("WHPX)"));
        assert_eq!(windows.to_string(), "Windows on x86-64");
    }

    #[cfg(unix)]
    #[test]
    fn a_vms_environment_decides_its_screen_over_this_process() {
        let env = |v: &str| {
            BTreeMap::from(["WAYLAND_DISPLAY", "DISPLAY"].map(|k| (k.to_string(), v.to_string())))
        };
        assert!(has_screen(&env(":0")));
        assert!(!has_screen(&env("")));
    }
}
