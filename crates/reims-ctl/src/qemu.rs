//! The QEMU command line a VM runs: one [`Invocation`], which `start` runs and `command`
//! prints as a shell script, so a printed script runs the VM exactly as reims-ctl does.
//!
//! Every VM gets:
//! - on reims-vgpu's own device, the library's window for its display: `-display none` with
//!   `REIMS_VGPU_WINDOW` unset (`0` would keep the device headless), never a window of QEMU's
//!   own;
//! - a QMP socket (`-qmp`), through which reims-ctl stops the VM;
//! - QEMU's pid file;
//! - user networking with virtio-net, and an optional SSH port forward; with networking off,
//!   `-nic none`, without which QEMU gives the machine its default NIC on user networking (on
//!   q35 an e1000e in the first free slot, 1, where the root port's address is);
//! - a virtio sound device on the host's audio backend, one output stream and one input stream,
//!   which x86-64 macOS's AppleVirtIOSound drives (docs/qemu.md, "x86-64 macOS guests: audio"):
//!   Safari's Media Source players fail without an output device; and an ICH9 Intel HDA
//!   controller with a line-out codec on that backend, for OpenCore's boot chime: its AudioDxe
//!   drives HDA, not virtio sound, and x86-64 macOS 26 has no HDA driver, so the guest's only
//!   output is the virtio device;
//! - the library's log (`REIMS_VGPU_LOG`) in QEMU's: with logs on at their level, with the
//!   serial console and timestamps; with logs off its warnings alone ([`LOGS_OFF`]).
//!
//! The machine, `q35-x86_64`, follows the command line the x86-64 guests were measured on
//! (docs/qemu.md): RAM as a shared memfd, which the library on Linux requires, allocated whole as
//! QEMU starts (`prealloc`), so no guest-RAM fault runs on a vCPU: where the host backs shmem with
//! transparent huge pages, such a fault allocates a huge page, compacting memory for it under the
//! process's mmap lock, and stalled vCPUs for up to 1.8 s, the others queued behind it ("Linux
//! hosts: guest RAM resident from the start" there); the keyboard and
//! tablet on a qemu-xhci controller, since on ICH9's EHCI the guest stops re-arming a device's
//! interrupt transfers under GPU load and its input freezes, with the host's USB devices the VM's
//! settings give it beside them ([`crate::usb`]); fixed PCI addresses (the AHCI controller, which
//! holds OpenCore's image, the installer and, on SATA, the system disk, at slot 2, where
//! OpenCore's boot entries name it; on NVMe, the system disk's controller at slot 10,
//! [`DiskController`]); `isa-applesmc` with Apple's SMC key, [`OSK`];
//! VMX in the guest's CPU where the host's KVM offers it, for the guest's own VMs;
//! `dump-guest-core=off`, which leaves guest RAM out of a core dump of QEMU: with it in, a
//! crash's core takes the process tens of seconds to end. QEMU refuses the
//! option where the OS cannot leave memory out of a core (macOS has no `MADV_DONTDUMP`); and no
//! ACPI S3 (`ICH9-LPC.disable_s3`): a guest in S3 never wakes again with its display. Only PS/2
//! input wakes QEMU from S3, not the USB keyboard and tablet, and QEMU's wake resets every
//! device, where the guest's GPU driver restores none of the device's state, so the display
//! stays dark and WindowServer waits on the GPU for good. Without S3, macOS's sleep stops at a
//! dark wake, from which input wakes the display (docs/qemu.md).
//!
//! A VM's graphics ([`Graphics`]) are one of two, each at slot 4 with the id `gfx`, which the
//! terminal viewer's screendumps name:
//! - reims-vgpu's own GPU device, `reims-graphics-pci`, with the option ROM reims-ctl carries as
//!   its `romfile` ([`Ctx::graphics_rom`], [`crate::prebuilt`]), which the device has no default
//!   for. The guest drives it with the kexts reims-ctl puts in the VM's OpenCore image. Its
//!   BAR 2, the library's region, is 1 GiB below 4 GiB, which macOS gives an address only where
//!   q35 leaves room there, so the machine keeps 1 GiB of RAM below 4 GiB
//!   (`max-ram-below-4g=1G`): with more, the guest's kext finds no region. Its one display opens
//!   in the library's window at the resolution's size ([`crate::vm::Display::window_size`],
//!   the device's `resolution`; 1920x1080 by default) and follows the window's size at the host
//!   screen's rate; borderless reaches it as `REIMS_VGPU_WINDOW_BORDERLESS`.
//! - QEMU's `vmware-svga` (`graphics = "vmware-svga"`). The library then draws nothing, so the
//!   window's and the GPU's variables are left out, and QEMU's own window, GTK's (`-display
//!   gtk`), shows the display. Where QEMU's environment names no Wayland compositor or X server,
//!   GTK's window would end QEMU as it starts, so it is `-display none` there, and the guest runs
//!   headless, as on the library's window.
//!
//! The xHCI controller has seven ports (`p2=7,p3=7`), each taking a USB 2 and a USB 3 device:
//! the keyboard, the tablet, and up to [`crate::vm::USB_DEVICES_MAX`] host devices, at their
//! own speeds. Its 14 ports are within the 15 macOS numbers on one controller. QEMU's default of
//! four would leave two: where a device takes the last free port, QEMU chains a full-speed hub
//! there first (hw/usb/bus.c, `usb_claim_port`), on which a high-speed device does not attach,
//! so one port stays free. With as many USB 2 ports as USB 3 ones no port is USB 2 alone: QEMU
//! gives a device the first free port, and one freed goes to the back of the line.

use crate::host::{Host, Profile};
use crate::vm::{DiskController, Graphics, Vm};
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

/// The SMC key `isa-applesmc` gives every x86-64 guest.
pub const OSK: &str = "ourhardworkbythesewordsguardedpleasedontsteal(c)AppleComputerInc";

#[derive(Clone, Debug, PartialEq)]
pub struct Invocation {
    pub program: PathBuf,
    pub args: Vec<String>,
    /// Variables set for QEMU.
    pub env: Vec<(String, String)>,
    /// Variables removed from QEMU's environment.
    pub unset: Vec<String>,
}

pub struct Ctx {
    pub host: Host,
    pub qemu: PathBuf,
    /// The audio backend for a VM that names none.
    pub audio_backend: String,
    /// Whether the host's KVM offers VMX to guests ([`crate::host::kvm_nested_vmx`]).
    pub nested_vmx: bool,
    /// Whether QEMU's own window can open ([`crate::host::has_screen`]).
    pub screen: bool,
    /// The option ROM of reims-vgpu's own device, which its `romfile` names
    /// ([`crate::prebuilt::graphics_rom`]).
    pub graphics_rom: PathBuf,
}

/// A value inside a QEMU option list: commas doubled.
pub fn opt(s: &str) -> String {
    s.replace(',', ",,")
}

fn optp(p: &Path) -> String {
    opt(&p.to_string_lossy())
}

pub fn invocation(vm: &Vm, ctx: &Ctx) -> Result<Invocation> {
    let c = &vm.config;
    let mut a: Vec<String> = Vec::new();
    let mut push = |xs: &[&str]| a.extend(xs.iter().map(|s| s.to_string()));
    push(&["-name", &c.name]);
    match c.profile {
        Profile::Q35X86_64 => q35(vm, ctx, &mut a)?,
    }
    common(vm, ctx, &mut a)?;
    a.extend(c.extra_args.iter().cloned());

    let mut env = Vec::new();
    let mut unset = vec!["REIMS_VGPU_WINDOW".to_string()];
    let graphics = c.graphics();
    let library = graphics.library();
    if c.logs.enabled {
        env.push(("REIMS_VGPU_LOG".into(), c.logs.level.clone()));
    } else if library {
        env.push(("REIMS_VGPU_LOG".into(), LOGS_OFF.into()));
    } else {
        unset.push("REIMS_VGPU_LOG".into());
    }
    if library && c.display.borderless {
        env.push(("REIMS_VGPU_WINDOW_BORDERLESS".into(), "1".into()));
    }
    if library
        && c.profile.gpu_choice(ctx.host)
        && let Some(f) = &c.gpu.driver_file
    {
        // The loader reads VK_DRIVER_FILES; loaders before 1.3.207 read VK_ICD_FILENAMES.
        let f = f.to_string_lossy().into_owned();
        env.push(("VK_DRIVER_FILES".into(), f.clone()));
        env.push(("VK_ICD_FILENAMES".into(), f));
        if let Some(id) = c.gpu.device_id {
            env.push(("VK_LOADER_DEVICE_ID_FILTER".into(), format!("{id:#06x}")));
        }
    }
    for (k, v) in &c.extra_env {
        // A name the shell's export takes, so the printed script runs as start does.
        crate::vm::check_env_name(k)?;
        env.retain(|(ek, _)| ek != k);
        unset.retain(|u| u != k);
        env.push((k.clone(), v.clone()));
    }
    Ok(Invocation {
        program: ctx.qemu.clone(),
        args: a,
        env,
        unset,
    })
}

/// `REIMS_VGPU_LOG` with logs off: the library's warnings, into QEMU's log beside QEMU's own
/// errors.
const LOGS_OFF: &str = "warn";

fn q35(vm: &Vm, ctx: &Ctx, a: &mut Vec<String>) -> Result<()> {
    let c = &vm.config;
    let q = c.q35.as_ref().context("no [q35] settings")?;
    let graphics = c.graphics();
    let mem = format!("{}M", c.memory_mib);
    let mut push = |xs: &[&str]| a.extend(xs.iter().map(|s| s.to_string()));
    let mut machine = String::from("q35");
    if c.profile.memfd_ram(ctx.host) {
        machine.push_str(",memory-backend=ram,dump-guest-core=off");
    }
    // Room below 4 GiB for reims-vgpu's 1 GiB BAR 2 (the module doc).
    if graphics == Graphics::ReimsVgpu {
        machine.push_str(",max-ram-below-4g=1G");
    }
    push(&["-machine", &machine]);
    if c.profile.memfd_ram(ctx.host) {
        // Resident from the start, made by as many threads as the guest has vCPUs (the module
        // doc).
        let backend = format!(
            "memory-backend-memfd,id=ram,size={mem},share=on,prealloc=on,prealloc-threads={}",
            c.cpus
        );
        push(&["-object", &backend]);
    }
    push(&["-accel", c.profile.accel(), "-m", &mem]);
    let mut cpu = format!(
        "{},kvm=on,vendor=GenuineIntel,+invtsc,vmware-cpuid-freq=on",
        q.cpu_model
    );
    // VMX lets the guest run VMs of its own (Hypervisor.framework, Docker). It is given only where
    // KVM offers it, so `check` never drops it with a warning. It is `vmx=on`, not `+vmx`, so a
    // VM's `-vmx` or `vmx=off` in `cpu_flags` takes it away: QEMU applies `name=value` flags in
    // order, and `+` and `-` ones after them, so a `+vmx` here would outlive a `vmx=off`.
    if ctx.nested_vmx {
        cpu.push_str(",vmx=on");
    }
    if !q.cpu_flags.trim().is_empty() {
        cpu.push(',');
        cpu.push_str(q.cpu_flags.trim());
    }
    push(&["-cpu", &cpu]);
    push(&[
        "-smp",
        &format!("{0},sockets=1,cores={0},threads=1", c.cpus),
    ]);
    push(&[
        "-global",
        "ICH9-LPC.acpi-pci-hotplug-with-bridge-support=off",
    ]);
    push(&["-global", "ICH9-LPC.disable_s3=1"]);
    push(&[
        "-device",
        "pcie-root-port,bus=pcie.0,addr=0x1,slot=1,x-speed=16,x-width=32",
    ]);
    push(&["-device", &format!("isa-applesmc,osk={OSK}")]);
    push(&[
        "-drive",
        &format!(
            "if=pflash,format=raw,readonly=on,file={}",
            optp(&vm.path(&q.firmware_code))
        ),
    ]);
    push(&[
        "-drive",
        &format!(
            "if=pflash,format=raw,file={}",
            optp(&vm.path(&q.firmware_vars))
        ),
    ]);
    push(&["-smbios", "type=2"]);
    push(&["-device", "ich9-ahci,id=sata,addr=0x2"]);
    push(&[
        "-drive",
        &format!(
            "id=opencore,if=none,format={},file={}",
            q.opencore_format.as_str(),
            optp(&vm.path(&q.opencore))
        ),
    ]);
    push(&["-device", "ide-hd,bus=sata.2,drive=opencore,bootindex=1"]);
    let disk = format!(
        "id=disk,if=none,format={},file={}",
        c.disk_format.as_str(),
        optp(&vm.disk())
    );
    match q.disk_controller {
        // `discard=unmap` passes the guest's TRIMs to the image, which frees their space. QEMU's
        // NVMe controller needs a serial, and without an address takes slot 3, virtio-net's.
        DiskController::Nvme => push(&[
            "-drive",
            &format!("{disk},discard=unmap"),
            "-device",
            "nvme,drive=disk,serial=reims-ctl,addr=0xa",
        ]),
        DiskController::Ahci => push(&[
            "-drive",
            &disk,
            "-device",
            &format!("ide-hd,bus=sata.{},drive=disk,rotation_rate=1", q.disk_port),
        ]),
    }
    // The installer, on port 4, past OpenCore's, or on 5 past an AHCI disk on 4.
    if let Some(p) = q.installer.as_deref().map(|p| vm.path(p)) {
        let port = if q.disk_controller == DiskController::Ahci && q.disk_port == 4 {
            5
        } else {
            4
        };
        push(&[
            "-drive",
            &format!(
                "id=installer,if=none,format=raw,snapshot=on,file={}",
                optp(&p)
            ),
        ]);
        push(&[
            "-device",
            &format!("ide-hd,bus=sata.{port},drive=installer"),
        ]);
    }
    if c.network.enabled {
        push(&["-netdev", &netdev(vm)]);
        push(&[
            "-device",
            &format!(
                "virtio-net-pci,netdev=net0,id=net0,mac={},addr=0x3",
                c.network.mac
            ),
        ]);
    }
    push(&["-vga", "none"]);
    let gfx = match graphics {
        // The window's first size only where it is not the device's own, 1920x1080: a QEMU
        // built before the property refuses it.
        Graphics::ReimsVgpu => format!(
            "reims-graphics-pci,id=gfx,bus=pcie.0,addr=0x4,romfile={}{}",
            optp(&ctx.graphics_rom),
            c.display
                .window_size()
                .map_or(String::new(), |(w, h)| format!(",resolution={w}x{h}"))
        ),
        Graphics::VmwareSvga => "vmware-svga,id=gfx,bus=pcie.0,addr=0x4".into(),
    };
    push(&["-device", &gfx]);
    push(&[
        "-device",
        "qemu-xhci,id=xhci,addr=0x5,conditional-intr-mapping=on,p2=7,p3=7",
    ]);
    push(&[
        "-device",
        "usb-kbd,id=kbd,bus=xhci.0",
        "-device",
        "usb-tablet,id=tablet,bus=xhci.0",
    ]);
    for u in &c.usb {
        push(&["-device", &crate::usb::device_arg(u)?]);
    }
    if c.audio.enabled {
        push(&["-audiodev", &audiodev(vm, ctx)]);
        push(&[
            "-device",
            "virtio-sound-pci,id=audio,audiodev=snd0,addr=0x7",
        ]);
        // At ICH9's own slot for it.
        push(&["-device", "ich9-intel-hda,id=hda,addr=0x1b"]);
        push(&["-device", "hda-output,bus=hda.0,audiodev=snd0"]);
    }
    Ok(())
}

fn netdev(vm: &Vm) -> String {
    let mut n = "user,id=net0,ipv6=off".to_string();
    if let Some(p) = vm.config.network.ssh_port {
        n.push_str(&format!(",hostfwd=tcp:127.0.0.1:{p}-:22"));
    }
    n
}

fn audiodev(vm: &Vm, ctx: &Ctx) -> String {
    let backend = vm
        .config
        .audio
        .backend
        .as_deref()
        .unwrap_or(&ctx.audio_backend);
    format!("{},id=snd0", opt(backend))
}

fn common(vm: &Vm, ctx: &Ctx, a: &mut Vec<String>) -> Result<()> {
    let c = &vm.config;
    let mut push = |xs: &[&str]| a.extend(xs.iter().map(|s| s.to_string()));
    if !c.network.enabled {
        push(&["-nic", "none"]);
    }
    let qemu_window = !c.graphics().library() && ctx.screen;
    push(&["-display", if qemu_window { "gtk" } else { "none" }]);
    if c.logs.enabled {
        push(&["-serial", &format!("file:{}", vm.serial_log().display())]);
        push(&["-msg", "timestamp=on"]);
    } else {
        push(&["-serial", "none"]);
    }
    push(&["-monitor", "none"]);
    push(&[
        "-qmp",
        &format!("unix:{},server=on,wait=off", optp(&vm.qmp_socket())),
    ]);
    push(&["-pidfile", &vm.pidfile().to_string_lossy()]);
    Ok(())
}

impl Invocation {
    /// A `std::process::Command` running it.
    pub fn command(&self) -> std::process::Command {
        let mut cmd = std::process::Command::new(&self.program);
        cmd.args(&self.args);
        for k in &self.unset {
            cmd.env_remove(k);
        }
        for (k, v) in &self.env {
            cmd.env(k, v);
        }
        cmd
    }

    /// A POSIX shell script that runs it, an option and its value to a line.
    pub fn shell_script(&self, comment: &str) -> String {
        let mut s = String::from("#!/bin/sh\n");
        for line in comment.lines() {
            s.push_str("# ");
            s.push_str(line);
            s.push('\n');
        }
        for k in &self.unset {
            s.push_str(&format!("unset {k}\n"));
        }
        for (k, v) in &self.env {
            s.push_str(&format!("export {k}={}\n", quote(v)));
        }
        s.push_str("exec ");
        s.push_str(&quote(&self.program.to_string_lossy()));
        let mut i = 0;
        while i < self.args.len() {
            s.push_str(" \\\n  ");
            s.push_str(&quote(&self.args[i]));
            if self.args[i].starts_with('-')
                && i + 1 < self.args.len()
                && !self.args[i + 1].starts_with('-')
            {
                s.push(' ');
                s.push_str(&quote(&self.args[i + 1]));
                i += 1;
            }
            i += 1;
        }
        s.push('\n');
        s
    }
}

pub fn quote(s: &str) -> String {
    shlex::try_quote(s)
        .map(|c| c.into_owned())
        .unwrap_or_else(|_| format!("'{}'", s.replace('\'', "'\\''")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::{Arch, Os};
    use crate::vm::*;
    use std::collections::BTreeMap;

    fn linux() -> Host {
        Host {
            os: Os::Linux,
            arch: Arch::X86_64,
        }
    }

    fn mac() -> Host {
        Host {
            os: Os::Mac,
            arch: Arch::Aarch64,
        }
    }

    fn q35_vm() -> Vm {
        Vm {
            dir: "/vms/t".into(),
            config: VmConfig {
                name: "t".into(),
                profile: Profile::Q35X86_64,
                cpus: 6,
                memory_mib: 12288,
                disk: "disk.qcow2".into(),
                disk_format: ImageFormat::Qcow2,
                stop_timeout_secs: 90,
                extra_args: vec![],
                graphics: None,
                display: Display::default(),
                gpu: Gpu::default(),
                audio: Audio::default(),
                network: Network {
                    enabled: true,
                    mac: "52:54:00:aa:bb:cc".into(),
                    ssh_port: Some(2326),
                },
                logs: Logs::default(),
                q35: Some(Q35 {
                    opencore: "opencore.qcow2".into(),
                    opencore_format: ImageFormat::Qcow2,
                    firmware_code: "OVMF_CODE.fd".into(),
                    firmware_vars: "OVMF_VARS.fd".into(),
                    installer: None,
                    cpu_model: default_cpu_model(),
                    cpu_flags: default_cpu_flags(),
                    disk_controller: DiskController::Ahci,
                    disk_port: 3,
                }),
                extra_env: BTreeMap::new(),
                usb: vec![],
            },
        }
    }

    fn ctx(host: Host) -> Ctx {
        Ctx {
            host,
            qemu: "/q/bin/qemu".into(),
            audio_backend: "pipewire".into(),
            nested_vmx: false,
            screen: true,
            graphics_rom: "/d/firmware/ReimsGraphics-0123456789abcdef.rom".into(),
        }
    }

    fn has(args: &[String], pair: &[&str]) -> bool {
        args.windows(pair.len())
            .any(|w| w.iter().zip(pair).all(|(a, b)| a == b))
    }

    #[test]
    fn q35_on_linux_has_the_measured_devices() {
        let inv = invocation(&q35_vm(), &ctx(linux())).unwrap();
        let a = &inv.args;
        for pair in [
            &[
                "-machine",
                "q35,memory-backend=ram,dump-guest-core=off,max-ram-below-4g=1G",
            ][..],
            &[
                "-object",
                "memory-backend-memfd,id=ram,size=12288M,share=on,prealloc=on,prealloc-threads=6",
            ],
            &["-accel", "kvm", "-m", "12288M"],
            &["-smp", "6,sockets=1,cores=6,threads=1"],
            &["-device", &format!("isa-applesmc,osk={OSK}")],
            &[
                "-drive",
                "if=pflash,format=raw,readonly=on,file=/vms/t/OVMF_CODE.fd",
            ],
            &["-device", "ich9-ahci,id=sata,addr=0x2"],
            &["-device", "ide-hd,bus=sata.2,drive=opencore,bootindex=1"],
            &[
                "-drive",
                "id=disk,if=none,format=qcow2,file=/vms/t/disk.qcow2",
                "-device",
                "ide-hd,bus=sata.3,drive=disk,rotation_rate=1",
            ],
            &[
                "-netdev",
                "user,id=net0,ipv6=off,hostfwd=tcp:127.0.0.1:2326-:22",
            ],
            &["-vga", "none"],
            &[
                "-device",
                "reims-graphics-pci,id=gfx,bus=pcie.0,addr=0x4,\
                 romfile=/d/firmware/ReimsGraphics-0123456789abcdef.rom",
            ],
            &[
                "-device",
                "qemu-xhci,id=xhci,addr=0x5,conditional-intr-mapping=on,p2=7,p3=7",
            ],
            &["-audiodev", "pipewire,id=snd0"],
            &[
                "-device",
                "virtio-sound-pci,id=audio,audiodev=snd0,addr=0x7",
            ],
            &["-device", "ich9-intel-hda,id=hda,addr=0x1b"],
            &["-device", "hda-output,bus=hda.0,audiodev=snd0"],
            &["-display", "none"],
            &["-qmp", "unix:/vms/t/qmp.sock,server=on,wait=off"],
            &["-pidfile", "/vms/t/qemu.pid"],
        ] {
            assert!(has(a, pair), "missing {pair:?} in {a:?}");
        }
        assert!(!a.iter().any(|x| x == "-usb"), "no EHCI");
        assert!(
            !a.iter().any(|x| x.starts_with("nvme,")),
            "the disk on SATA alone"
        );
        assert!(!a.iter().any(|x| x.contains("usb-audio")));
        assert!(!a.iter().any(|x| x.contains("installer")), "no installer");
        assert!(inv.unset.contains(&"REIMS_VGPU_WINDOW".to_string()));
        assert!(!inv.unset.contains(&"REIMS_VGPU_LOG".to_string()));
    }

    /// The host's USB devices a VM is given are on its xHCI controller after the keyboard and
    /// the tablet, each by its IDs, and by its port where it was chosen beside a twin.
    #[test]
    fn usb_devices_are_passed_through_after_the_keyboard_and_tablet() {
        let mut vm = q35_vm();
        vm.config.usb = vec![
            UsbDevice {
                id: "1050:0407".into(),
                port: None,
                name: "Yubico YubiKey".into(),
            },
            UsbDevice {
                id: "0403:6001".into(),
                port: Some("3-2.1".into()),
                name: "FTDI FT232R".into(),
            },
        ];
        let a = invocation(&vm, &ctx(linux())).unwrap().args;
        let tablet = a
            .iter()
            .position(|x| x == "usb-tablet,id=tablet,bus=xhci.0")
            .unwrap();
        assert_eq!(
            a[tablet + 1..tablet + 5],
            [
                "-device",
                "usb-host,id=usb-1050-0407,bus=xhci.0,vendorid=0x1050,productid=0x0407",
                "-device",
                "usb-host,id=usb-0403-6001-3-2.1,bus=xhci.0,vendorid=0x0403,productid=0x6001,\
                 hostbus=3,hostport=2.1",
            ]
        );
        // None chosen: none passed through, on the same controller.
        let a = invocation(&q35_vm(), &ctx(linux())).unwrap().args;
        assert!(!a.iter().any(|x| x.starts_with("usb-host")), "{a:?}");
    }

    /// With logs off the library still logs its warnings; with logs on, at their level.
    #[test]
    fn the_librarys_warnings_are_logged_with_logs_off() {
        let mut vm = q35_vm();
        let level = |vm: &Vm| {
            let inv = invocation(vm, &ctx(linux())).unwrap();
            assert!(!inv.unset.contains(&"REIMS_VGPU_LOG".to_string()));
            let env: BTreeMap<_, _> = inv.env.into_iter().collect();
            env["REIMS_VGPU_LOG"].clone()
        };
        assert!(!vm.config.logs.enabled);
        assert_eq!(level(&vm), "warn");
        vm.config.logs = Logs {
            enabled: true,
            level: "debug".into(),
        };
        assert_eq!(level(&vm), "debug");
    }

    /// A guest that enters S3 keeps a dark display (the module doc), so q35 offers it none.
    #[test]
    fn q35_offers_the_guest_no_s3_sleep() {
        let inv = invocation(&q35_vm(), &ctx(linux())).unwrap();
        assert!(has(&inv.args, &["-global", "ICH9-LPC.disable_s3=1"]));
    }

    #[test]
    fn network_off_gives_the_machine_no_nic() {
        let mut q35 = q35_vm();
        q35.config.network.enabled = false;
        let a = invocation(&q35, &ctx(linux())).unwrap().args;
        assert!(has(&a, &["-nic", "none"]), "{a:?}");
        assert!(!a.iter().any(|x| x == "-netdev" || x.contains("net-pci")));
        let a = invocation(&q35_vm(), &ctx(linux())).unwrap().args;
        assert!(!a.iter().any(|x| x == "-nic"));
    }

    #[test]
    fn vmx_only_where_kvm_offers_it() {
        let cpu = |nested_vmx| {
            let inv = invocation(
                &q35_vm(),
                &Ctx {
                    nested_vmx,
                    ..ctx(linux())
                },
            )
            .unwrap();
            let i = inv.args.iter().position(|x| x == "-cpu").unwrap();
            inv.args[i + 1].clone()
        };
        assert_eq!(
            cpu(true),
            format!(
                "{},kvm=on,vendor=GenuineIntel,+invtsc,vmware-cpuid-freq=on,vmx=on,{}",
                default_cpu_model(),
                default_cpu_flags()
            )
        );
        assert!(!cpu(false).contains("vmx"));
    }

    #[test]
    fn q35_without_audio_has_no_sound_devices() {
        let mut vm = q35_vm();
        vm.config.audio.enabled = false;
        let inv = invocation(&vm, &ctx(linux())).unwrap();
        for x in ["-audiodev", "usb-audio", "ich9-intel-hda", "hda-output"] {
            assert!(
                !inv.args.iter().any(|a| a.contains(x)),
                "{x} in {:?}",
                inv.args
            );
        }
        assert!(!inv.args.iter().any(|a| a.contains("virtio-sound")));
    }

    /// A disk moved to NVMe is on a controller of its own, off the AHCI controller.
    #[test]
    fn a_disk_on_nvme_has_a_controller_of_its_own() {
        let mut vm = q35_vm();
        vm.config.q35.as_mut().unwrap().disk_controller = DiskController::Nvme;
        let a = invocation(&vm, &ctx(linux())).unwrap().args;
        assert!(has(
            &a,
            &[
                "-drive",
                "id=disk,if=none,format=qcow2,file=/vms/t/disk.qcow2,discard=unmap",
                "-device",
                "nvme,drive=disk,serial=reims-ctl,addr=0xa",
            ]
        ));
        assert!(
            !a.iter().any(|x| x.ends_with("drive=disk,rotation_rate=1")),
            "{a:?}"
        );
        // The installer stays on AHCI, on port 4.
        vm.config.q35.as_mut().unwrap().installer = Some("/isos/BaseSystem.img".into());
        let a = invocation(&vm, &ctx(linux())).unwrap().args;
        assert!(has(&a, &["-device", "ide-hd,bus=sata.4,drive=installer"]));
    }

    #[test]
    fn installer_disk_goes_on_its_own_port_through_an_overlay() {
        let mut vm = q35_vm();
        let q = vm.config.q35.as_mut().unwrap();
        q.installer = Some("/isos/Base,System.img".into());
        let a = invocation(&vm, &ctx(linux())).unwrap().args;
        assert!(has(
            &a,
            &[
                "-drive",
                "id=installer,if=none,format=raw,snapshot=on,file=/isos/Base,,System.img"
            ]
        ));
        assert!(has(&a, &["-device", "ide-hd,bus=sata.4,drive=installer"]));
        // Past an AHCI disk on port 4.
        let q = vm.config.q35.as_mut().unwrap();
        q.disk_port = 4;
        let a = invocation(&vm, &ctx(linux())).unwrap().args;
        assert!(has(&a, &["-device", "ide-hd,bus=sata.5,drive=installer"]));
        // The VM's copy of a DMG, in its directory.
        let q = vm.config.q35.as_mut().unwrap();
        q.installer = Some("installer.img".into());
        let a = invocation(&vm, &ctx(linux())).unwrap().args;
        assert!(has(
            &a,
            &[
                "-drive",
                "id=installer,if=none,format=raw,snapshot=on,file=/vms/t/installer.img"
            ]
        ));
    }

    #[test]
    fn logs_gpu_and_window_settings_reach_the_environment() {
        let mut vm = q35_vm();
        vm.config.logs = Logs {
            enabled: true,
            level: "debug".into(),
        };
        vm.config.display.borderless = true;
        vm.config.gpu = Gpu {
            driver_file: Some("/icd/nvidia.json".into()),
            device_id: Some(0x2204),
            name: None,
        };
        vm.config
            .extra_env
            .insert("REIMS_VGPU_LOG".into(), "trace".into());
        let inv = invocation(&vm, &ctx(linux())).unwrap();
        let env: BTreeMap<_, _> = inv.env.iter().cloned().collect();
        assert_eq!(env["REIMS_VGPU_LOG"], "trace");
        assert_eq!(env["REIMS_VGPU_WINDOW_BORDERLESS"], "1");
        assert_eq!(env["VK_DRIVER_FILES"], "/icd/nvidia.json");
        assert_eq!(env["VK_LOADER_DEVICE_ID_FILTER"], "0x2204");
        assert!(has(&inv.args, &["-serial", "file:/vms/t/logs/serial.log"]));
        assert!(has(&inv.args, &["-msg", "timestamp=on"]));
        assert!(!inv.unset.contains(&"REIMS_VGPU_LOG".to_string()));
    }

    #[test]
    fn vmware_svga_takes_the_gfx_slot_in_qemus_window() {
        let mut vm = q35_vm();
        vm.config.graphics = Some(Graphics::VmwareSvga);
        vm.config.display.borderless = true;
        vm.config.gpu.driver_file = Some("/icd/nvidia.json".into());
        let inv = invocation(&vm, &ctx(linux())).unwrap();
        let a = &inv.args;
        assert!(has(a, &["-vga", "none"]));
        assert!(has(
            a,
            &["-device", "vmware-svga,id=gfx,bus=pcie.0,addr=0x4"]
        ));
        assert!(has(a, &["-display", "gtk"]));
        assert!(!a.iter().any(|x| x.contains("reims-graphics")), "{a:?}");
        assert!(inv.env.is_empty(), "{:?}", inv.env);
        // Without a screen for the window, headless.
        let headless = Ctx {
            screen: false,
            ..ctx(linux())
        };
        let a = invocation(&vm, &headless).unwrap().args;
        assert!(has(&a, &["-display", "none"]));
        // On reims-vgpu's device, the library's window, never QEMU's.
        let a = invocation(&q35_vm(), &ctx(linux())).unwrap().args;
        assert!(has(&a, &["-display", "none"]));
    }

    /// reims-vgpu's own device, with the ROM reims-ctl carries, on a machine that leaves its BAR
    /// room below 4 GiB: the default of a VM whose settings name no graphics. The resolution, as
    /// its window's first size, and borderless reach it; the GPU and the logs are the library's.
    #[test]
    fn reims_vgpu_has_its_own_device() {
        let mut vm = q35_vm();
        vm.config.display.resolution = "2560x1600".into();
        vm.config.display.borderless = true;
        vm.config.gpu.driver_file = Some("/icd/nvidia.json".into());
        let inv = invocation(&vm, &ctx(linux())).unwrap();
        let a = &inv.args;
        for pair in [
            &[
                "-machine",
                "q35,memory-backend=ram,dump-guest-core=off,max-ram-below-4g=1G",
            ][..],
            &["-vga", "none"],
            &[
                "-device",
                "reims-graphics-pci,id=gfx,bus=pcie.0,addr=0x4,\
                 romfile=/d/firmware/ReimsGraphics-0123456789abcdef.rom,resolution=2560x1600",
            ],
            &["-display", "none"],
        ] {
            assert!(has(a, pair), "missing {pair:?} in {a:?}");
        }
        assert!(!a.iter().any(|x| x.contains("vmware-svga")), "{a:?}");
        let env: BTreeMap<_, _> = inv.env.iter().cloned().collect();
        assert_eq!(env["REIMS_VGPU_WINDOW_BORDERLESS"], "1");
        assert_eq!(env["VK_DRIVER_FILES"], "/icd/nvidia.json");
        assert!(env.contains_key("REIMS_VGPU_LOG"));
        assert!(inv.unset.contains(&"REIMS_VGPU_WINDOW".to_string()));
        // A comma in the ROM's path stays in the option's value; without memfd RAM the machine
        // still leaves the BAR its room.
        let c = Ctx {
            graphics_rom: "/a,b/ReimsGraphics.rom".into(),
            ..ctx(mac())
        };
        let a = invocation(&vm, &c).unwrap().args;
        assert!(has(&a, &["-machine", "q35,max-ram-below-4g=1G"]), "{a:?}");
        assert!(
            a.iter()
                .any(|x| x.ends_with(",romfile=/a,,b/ReimsGraphics.rom,resolution=2560x1600")),
            "{a:?}"
        );
        // At the device's own size, 1920x1080, the window's size is left to it.
        vm.config.display.resolution = "1920x1080".into();
        let a = invocation(&vm, &ctx(linux())).unwrap().args;
        assert!(
            a.iter()
                .any(|x| x.starts_with("reims-graphics-pci,") && !x.contains("resolution")),
            "{a:?}"
        );
        // VMware SVGA keeps the machine's RAM below 4 GiB as it was.
        vm.config.graphics = Some(Graphics::VmwareSvga);
        let a = invocation(&vm, &ctx(linux())).unwrap().args;
        assert!(!a.iter().any(|x| x.contains("max-ram-below-4g")), "{a:?}");
    }

    /// A variable for QEMU's environment is named as the printed script's export takes it: a
    /// space or a `;` in its name would break the script or run more than QEMU.
    #[test]
    fn an_environment_variable_is_named_as_the_shell_takes_it() {
        let mut vm = q35_vm();
        let ctx = ctx(linux());
        for bad in ["A B", "X;touch /tmp/x;Y", "1A", "", "A-B", "A=B"] {
            vm.config.extra_env = BTreeMap::from([(bad.to_string(), "1".to_string())]);
            assert!(vm.config.check().is_err(), "{bad:?}");
            assert!(invocation(&vm, &ctx).is_err(), "{bad:?}");
        }
        vm.config.extra_env = BTreeMap::from([("_Ab_9".to_string(), "1 2".to_string())]);
        assert!(vm.config.check().is_ok());
        let script = invocation(&vm, &ctx).unwrap().shell_script("");
        assert!(script.contains("\nexport _Ab_9='1 2'\n"), "{script}");
    }

    #[test]
    fn shell_script_quotes_and_pairs() {
        let inv = Invocation {
            program: "/q/qemu".into(),
            args: vec![
                "-name".into(),
                "a b".into(),
                "-S".into(),
                "-m".into(),
                "8G".into(),
            ],
            env: vec![("X".into(), "1 2".into())],
            unset: vec!["Y".into()],
        };
        let s = inv.shell_script("hello");
        assert_eq!(
            s,
            "#!/bin/sh\n# hello\nunset Y\nexport X='1 2'\nexec /q/qemu \\\n  -name 'a b' \\\n  -S \\\n  -m 8G\n"
        );
    }
}
