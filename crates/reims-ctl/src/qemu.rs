//! The QEMU command line a VM runs: one [`Invocation`], which `start` runs and `command`
//! prints as a shell script, so a printed script runs the VM as reims-ctl does, which on Linux
//! runs it in a systemd user scope of its own ([`crate::runtime::launch`]).
//!
//! Every VM gets:
//! - on reims-vgpu's own device, the library's window for each of its displays: `-display none`
//!   with `REIMS_VGPU_WINDOW` unset (`0` would keep the device headless), never a window of
//!   QEMU's own;
//! - a QMP socket (`-qmp`), through which reims-ctl stops the VM;
//! - QEMU's pid file;
//! - user networking with virtio-net, and an optional SSH port forward; with networking off,
//!   `-nic none`, without which QEMU gives the machine its default NIC on user networking (on
//!   q35 an e1000e in the first free slot, 1, where the root port's address is);
//! - two virtio sound devices on the host's audio backend, which x86-64 macOS's AppleVirtIOSound
//!   drives (docs/qemu.md, "x86-64 macOS guests: audio"): one with an output stream alone, since
//!   Safari's Media Source players fail without an output device, and one with an input stream
//!   alone, so a recording starts the guest's input whether or not an output plays (on one device
//!   with both, the guest starts the input with each output and stops reading it, and a recording
//!   made while an output plays gets nothing); and an ICH9 Intel HDA
//!   controller with a line-out codec on that backend, for OpenCore's boot chime: its AudioDxe
//!   drives HDA, not virtio sound, and x86-64 macOS 26 has no HDA driver, so the guest's only
//!   output is the virtio device;
//! - the library's log (`REIMS_VGPU_LOG`) in QEMU's: with logs on at their level, with the
//!   serial console and timestamps; with logs off its warnings alone ([`LOGS_OFF`]).
//!
//! On Windows the machine runs on WHPX, the Windows Hypervisor Platform, and its RAM is shared
//! memory of QEMU's RAM backend (`memory-backend-ram,share=on`), which reims-vgpu's QEMU makes a
//! pagefile-backed section there, as the library needs; the rest of its command line is Linux's.
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
//! OpenCore's boot entries name it; on NVMe, the system disk's controller at slot 10, and on
//! virtio, its virtio-blk device at slot 6, [`DiskController`]); `isa-applesmc` with Apple's SMC
//! key, [`OSK`]; VMX in the guest's CPU where the host's KVM offers it, for the guest's own VMs;
//! `dump-guest-core=off`, which leaves guest RAM out of a core dump of QEMU: with it in, a
//! crash's core takes the process tens of seconds to end. QEMU refuses the
//! option where the OS cannot leave memory out of a core (macOS has no `MADV_DONTDUMP`, and
//! QEMU on Windows says "Dumping guest memory cannot be disabled on this host"); and no
//! ACPI S3 (`ICH9-LPC.disable_s3`): a guest in S3 never wakes again with its display. Only PS/2
//! input wakes QEMU from S3, not the USB keyboard and tablet, and QEMU's wake resets every
//! device, where the guest's GPU driver restores none of the device's state, so the display
//! stays dark and WindowServer waits on the GPU for good. Without S3, macOS's sleep stops at a
//! dark wake, from which input wakes the display (docs/qemu.md).
//!
//! While the installer is attached, the system disk drops the guest's flushes (`cache.no-flush=on`,
//! which with QEMU's other cache settings at their defaults is its `cache=unsafe`). An install
//! writes a whole system and syncs it as it goes, and each guest flush is a flush of the image file
//! to the host's disk (`fdatasync` on Linux, `FlushFileBuffers` on Windows), which the write waits
//! behind. The writes still reach the image file as they come, and the host's OS writes them to its
//! disk in its own time. A host crash or power loss while the installer is attached can damage the
//! system disk, and the install then starts over. `reims-ctl eject` takes the installer out at the
//! VM's next start, and from that start the guest's flushes reach the host's disk again. Measured
//! with `qemu-img bench` on a new 128 GiB qcow2 image as [`crate::create`] makes it, 512 MiB of 64
//! KiB writes with a flush after each MiB (2026-11-22): on Windows 11's NTFS, 3.7 s with the
//! flushes against 0.37 s without; on Linux's btrfs (`compress=zstd:1`), 2.4 s against 0.09 s.
//!
//! A VM's graphics ([`Graphics`]) are one of two, each at slot 4 with the id `gfx`, which the
//! terminal viewer's screendumps name:
//! - reims-vgpu's own GPU device, `reims-graphics-pci`, with the option ROM reims-ctl carries as
//!   its `romfile` ([`Ctx::graphics_rom`], [`crate::prebuilt`]), which the device has no default
//!   for. The guest drives it with the kexts reims-ctl puts in the VM's OpenCore image. Its
//!   BAR 2, the library's region, is 1 GiB below 4 GiB, which macOS gives an address only where
//!   q35 leaves room there, so the machine keeps 1 GiB of RAM below 4 GiB
//!   (`max-ram-below-4g=1G`): with more, the guest's kext finds no region. It has as many
//!   displays as the VM's settings give, its `displays`, named only above one, the device's own
//!   count, since a QEMU built before the property refuses it. Each display opens in a window of
//!   the library's at the resolution's size ([`crate::vm::Display::window_size`], the device's
//!   `resolution`, every window's; 1920x1080 by default) and follows its window's size at the
//!   rate of the host screen the window is on; borderless reaches every window as
//!   `REIMS_VGPU_WINDOW_BORDERLESS`.
//! - QEMU's `vmware-svga` (`graphics = "vmware-svga"`). The library then draws nothing, so the
//!   window's and the GPU's variables are left out, and QEMU's own window, GTK's (`-display
//!   gtk`), shows the display. Where QEMU's environment names no Wayland compositor or X server,
//!   GTK's window would end QEMU as it starts, so it is `-display none` there, and the guest runs
//!   headless, as on the library's window.
//!
//! The arm64 machine, `vmapple-arm64` ([`vmapple`]), is QEMU's `vmapple`, with the command line
//! Ambrosia's boot check boots arm64 guests on (firmware/ambrosia/check-boot.sh, docs/qemu.md
//! "arm64 guests"): Ambrosia's image as its firmware (`-bios`, ambrosia.rs); the VM's auxiliary
//! storage without its header and its disk, each as a pflash drive and as the machine's
//! `vmapple-virtio-blk-pci` of its variant; virtio-net; reims-vgpu's own device,
//! `reims-graphics-pci`, which has no option ROM there; and the configuration bundle reims-ctl
//! carries ([`crate::prebuilt::ambrosia_bundle`]), loaded at 0x120000000, where Ambrosia reads
//! it: its config.plist starts the device's driver module and adds the guest's kexts. The machine
//! has its own xHCI controller with a keyboard and a tablet, and its `uuid` is the VM's ECID, which
//! the tickets of its macOS are bound to. It runs on HVF on a Mac, with the GIC in the kernel, on
//! KVM on arm64 Linux, and on TCG, a thread for each vCPU, on x86-64 ([`Profile::accel_arg`]). On
//! Linux its RAM is a shared memfd, as the library there needs, not allocated as QEMU starts.
//! It takes no host USB device, no VMware SVGA and no sound device.
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
use anyhow::{Context, Result, bail};
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
    /// The host's sound devices ([`sound_routing`]).
    pub sound: crate::os::SoundDevices,
    /// Whether the host's KVM offers VMX to guests ([`crate::host::kvm_nested_vmx`]).
    pub nested_vmx: bool,
    /// Whether the guest's CPU lists the AVX-512 state components in CPUID leaf 0xD, QEMU's
    /// `x-avx512-xstate` ([`avx512_xstate`]).
    pub avx512_xstate: bool,
    /// Whether QEMU's own window can open ([`crate::host::has_screen`]).
    pub screen: bool,
    /// The option ROM of reims-vgpu's own device, which its `romfile` names
    /// ([`crate::prebuilt::graphics_rom`]).
    pub graphics_rom: PathBuf,
    /// What an arm64 VM boots on; None for an x86-64 VM.
    pub arm64: Option<Arm64Boot>,
}

/// Whether an x86-64 guest's CPU lists the AVX-512 state components in CPUID leaf 0xD while leaf
/// 7 keeps AVX-512F clear, QEMU's `x-avx512-xstate`. macOS then keeps AVX-512 state, and the
/// guest's RgfxGate turns it on in each user thread at the thread's first FP use
/// (guest/rgfxgate/src/RgfxGateXstate.cpp): a Zen 4 core leaves its op cache unused while a
/// thread's XCR0 lacks the AVX-512 components. The guest still sees no AVX-512 in leaf 7. Given
/// where each holds, asked in this order:
/// - `kvm`: QEMU runs on KVM, which gives the guest leaf 0xD's components from the host's;
/// - `host_offsets`: the host's OS keeps the components, at whatever offsets its CPU lays them
///   out ([`crate::host::avx512_xstate_offsets`]): AMD's or Intel's. On Intel's layout its gain
///   is unmeasured, since no host it was measured on has one;
/// - `kexts_held`: the VM is on reims-vgpu's own device and its OpenCore image holds the guest's
///   kexts this reims-ctl carries ([`crate::opencore::holds_guest_kexts`]), whose RgfxGate copies
///   the state at the CPU's offsets. macOS alone copies it at Intel's, and a thread that runs
///   AVX-512 instructions loses its AVX-512 registers over a signal;
/// - `qemu_knows`: the QEMU it runs has the property ([`crate::install::cpu_has_property`]).
pub fn avx512_xstate(
    kvm: bool,
    host_offsets: Option<[u32; 3]>,
    kexts_held: impl FnOnce() -> bool,
    qemu_knows: impl FnOnce() -> bool,
) -> bool {
    kvm && host_offsets.is_some() && kexts_held() && qemu_knows()
}

/// What an arm64 VM boots on: Ambrosia's image ([`crate::ambrosia::image`]) and the
/// configuration bundle it boots with ([`crate::prebuilt::ambrosia_bundle`]).
#[derive(Clone, Debug, PartialEq)]
pub struct Arm64Boot {
    pub ambrosia: PathBuf,
    pub bundle: PathBuf,
}

/// Where QEMU loads the configuration bundle in an arm64 guest's memory, and Ambrosia reads it.
pub const BUNDLE_ADDR: &str = "0x120000000";

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
        Profile::VmappleArm64 => vmapple(vm, ctx, &mut a)?,
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
    if library && c.profile.gpu_choice(ctx.host) {
        if let Some((vendor, device)) = c.gpu.selected()? {
            // The GPU first, every other GPU and driver kept: Mesa's device-select layer reads
            // MESA_VK_DEVICE_SELECT, the loader VK_LOADER_DEVICE_SELECT (crate::gpu).
            env.push((
                "MESA_VK_DEVICE_SELECT".into(),
                format!("{vendor:04x}:{device:04x}"),
            ));
            env.push((
                "VK_LOADER_DEVICE_SELECT".into(),
                format!("{vendor:#06x}:{device:#06x}"),
            ));
        } else if let Some(f) = &c.gpu.driver_file {
            // The loader reads VK_DRIVER_FILES; loaders before 1.3.207 read VK_ICD_FILENAMES.
            let f = f.to_string_lossy().into_owned();
            env.push(("VK_DRIVER_FILES".into(), f.clone()));
            env.push(("VK_ICD_FILENAMES".into(), f));
            if let Some(id) = c.gpu.device_id {
                env.push(("VK_LOADER_DEVICE_ID_FILTER".into(), format!("{id:#06x}")));
            }
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
    } else if c.profile.section_ram(ctx.host) {
        // QEMU on Windows leaves no memory out of a core and refuses `dump-guest-core=off`.
        machine.push_str(",memory-backend=ram");
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
    } else if c.profile.section_ram(ctx.host) {
        // A pagefile-backed section in reims-vgpu's QEMU on Windows (the profile's section_ram).
        push(&[
            "-object",
            &format!("memory-backend-ram,id=ram,size={mem},share=on"),
        ]);
    }
    push(&["-accel", c.profile.accel(ctx.host), "-m", &mem]);
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
    // `name=value` as well, so a VM's `x-avx512-xstate=off` in `cpu_flags`, which follows it,
    // takes it away.
    if ctx.avx512_xstate {
        cpu.push_str(",x-avx512-xstate=on");
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
        "pcie-root-port,bus=pcie.0,addr=0x1,slot=1,speed=16,width=32",
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
    let mut disk = format!(
        "id=disk,if=none,format={},file={}",
        c.disk_format.as_str(),
        optp(&vm.disk())
    );
    // An install's flushes, which the module's doc measures, are dropped while the installer is
    // attached.
    if q.installer.is_some() {
        disk.push_str(",cache.no-flush=on");
    }
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
        // SATA's drive options, at the first slot no other device takes.
        DiskController::Virtio => push(&[
            "-drive",
            &disk,
            "-device",
            "virtio-blk-pci,drive=disk,addr=0x6",
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
        // The displays only above one, and the windows' first size only where it is not the
        // device's own, 1920x1080: a QEMU built before either property refuses it.
        Graphics::ReimsVgpu => format!(
            "reims-graphics-pci,id=gfx,bus=pcie.0,addr=0x4,romfile={}{}{}",
            optp(&ctx.graphics_rom),
            c.display
                .qemu_displays()
                .map_or(String::new(), |n| format!(",displays={n}")),
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
        let (backend, input_apart) = sound_routing(audio_backend(vm, ctx), ctx.sound);
        push(&["-audiodev", &format!("{},id=snd0", opt(backend))]);
        if input_apart {
            push(&["-audiodev", "none,id=sndin"]);
        }
        push(&[
            "-device",
            "virtio-sound-pci,id=audio,audiodev=snd0,addr=0x7,streams=1",
            "-device",
            if input_apart {
                "virtio-sound-pci,id=audioin,audiodev=sndin,addr=0x8,streams=1,input-streams=1"
            } else {
                "virtio-sound-pci,id=audioin,audiodev=snd0,addr=0x8,streams=1,input-streams=1"
            },
        ]);
        // At ICH9's own slot for it.
        push(&["-device", "ich9-intel-hda,id=hda,addr=0x1b"]);
        push(&["-device", "hda-output,bus=hda.0,audiodev=snd0"]);
    }
    Ok(())
}

/// The arm64 machine's options (the module doc), in the order Ambrosia's boot check gives them,
/// which is the order QEMU gives the devices their PCI slots in.
fn vmapple(vm: &Vm, ctx: &Ctx, a: &mut Vec<String>) -> Result<()> {
    let c = &vm.config;
    let v = c.vmapple.as_ref().context("no [vmapple] settings")?;
    let boot = ctx.arm64.as_ref().context("no Ambrosia for an arm64 VM")?;
    if c.graphics() != Graphics::ReimsVgpu {
        bail!(
            "an arm64 VM runs on reims-vgpu's own device, not {}",
            c.graphics().name()
        );
    }
    if !c.usb.is_empty() {
        bail!("USB passthrough: {}", crate::host::X86_64_ONLY);
    }
    let mem = format!("{}M", c.memory_mib);
    let mut push = |xs: &[&str]| a.extend(xs.iter().map(|s| s.to_string()));
    push(&["-accel", c.profile.accel_arg(ctx.host)]);
    let mut machine = format!("vmapple,uuid=0x{:016x}", v.ecid);
    if c.profile.memfd_ram(ctx.host) {
        machine.push_str(",memory-backend=ram,dump-guest-core=off");
    }
    push(&["-machine", &machine, "-m", &mem]);
    if c.profile.memfd_ram(ctx.host) {
        push(&[
            "-object",
            &format!("memory-backend-memfd,id=ram,size={mem},share=on"),
        ]);
    }
    push(&["-smp", &c.cpus.to_string()]);
    push(&["-bios", &boot.ambrosia.to_string_lossy()]);
    let aux = optp(&vm.aux_trimmed());
    let disk = optp(&vm.disk());
    let format = c.disk_format.as_str();
    push(&[
        "-drive",
        &format!("file={aux},if=pflash,format=raw"),
        "-drive",
        &format!("file={disk},if=pflash,format={format}"),
        "-drive",
        &format!("file={aux},if=none,id=aux,format=raw"),
        "-drive",
        &format!("file={disk},if=none,id=root,format={format}"),
        "-device",
        "vmapple-virtio-blk-pci,variant=aux,drive=aux",
        "-device",
        "vmapple-virtio-blk-pci,variant=root,drive=root",
    ]);
    if c.network.enabled {
        push(&["-netdev", &netdev(vm)]);
        push(&[
            "-device",
            &format!(
                "virtio-net-pci,netdev=net0,id=net0,mac={},romfile=",
                c.network.mac
            ),
        ]);
    }
    push(&[
        "-device",
        &format!(
            "reims-graphics-pci,id=gfx{}{}",
            c.display
                .qemu_displays()
                .map_or(String::new(), |n| format!(",displays={n}")),
            c.display
                .window_size()
                .map_or(String::new(), |(w, h)| format!(",resolution={w}x{h}"))
        ),
    ]);
    push(&[
        "-device",
        &format!(
            "loader,file={},addr={BUNDLE_ADDR},force-raw=on",
            optp(&boot.bundle)
        ),
    ]);
    Ok(())
}

fn netdev(vm: &Vm) -> String {
    let mut n = "user,id=net0,ipv6=off".to_string();
    if let Some(p) = vm.config.network.ssh_port {
        n.push_str(&format!(",hostfwd=tcp:127.0.0.1:{p}-:22"));
    }
    n
}

/// The VM's audio backend: its own, else the host's default.
fn audio_backend<'a>(vm: &'a Vm, ctx: &'a Ctx) -> &'a str {
    vm.config
        .audio
        .backend
        .as_deref()
        .unwrap_or(&ctx.audio_backend)
}

/// QEMU's DirectSound (`dsound`) opens Windows' output and recording devices as QEMU starts, and
/// QEMU ends where there is no output device. Where Windows has none, the VM's sound goes to
/// `none`; where it has no recording device, the input device alone gets an audiodev of its own,
/// `none`, so the guest records silence. What it returns: the backend of `snd0`, and whether the
/// input device is apart. Every other backend is as the VM names it.
pub fn sound_routing(backend: &str, devices: crate::os::SoundDevices) -> (&str, bool) {
    match backend {
        "dsound" if !devices.output => ("none", false),
        "dsound" => (backend, !devices.input),
        _ => (backend, false),
    }
}

/// What a start says of the VM's sound where [`sound_routing`] changes it.
pub fn sound_note(backend: &str, devices: crate::os::SoundDevices) -> Option<String> {
    match sound_routing(backend, devices) {
        ("none", _) if backend != "none" => Some(
            "this PC has no sound output device, so the VM runs without sound: connect speakers \
             or headphones and start it again for sound"
                .into(),
        ),
        (_, true) => Some(
            "this PC has no recording device (a microphone), so the VM's microphone records \
             silence"
                .into(),
        ),
        _ => None,
    }
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
                vmapple: None,
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
            sound: crate::os::SoundDevices {
                output: true,
                input: true,
            },
            nested_vmx: false,
            avx512_xstate: false,
            screen: true,
            graphics_rom: "/d/firmware/ReimsGraphics-0123456789abcdef.rom".into(),
            arm64: None,
        }
    }

    fn arm64_vm() -> Vm {
        let mut vm = q35_vm();
        let c = &mut vm.config;
        c.profile = Profile::VmappleArm64;
        c.cpus = 4;
        c.memory_mib = 8192;
        c.disk = "disk.img".into();
        c.disk_format = ImageFormat::Raw;
        c.graphics = Some(Graphics::ReimsVgpu);
        c.q35 = None;
        c.vmapple = Some(Vmapple {
            aux: "aux.img".into(),
            ecid: 0x9e35_f01e_dc0c_5536,
        });
        vm
    }

    fn arm64_ctx(host: Host) -> Ctx {
        Ctx {
            graphics_rom: std::path::PathBuf::new(),
            arm64: Some(Arm64Boot {
                ambrosia: "/d/ambrosia/ambrosia-0.1.3.bin".into(),
                bundle: "/d/prebuilt/Config-0123456789abcdef.cpio".into(),
            }),
            ..ctx(host)
        }
    }

    /// The machine's options on each host, as Ambrosia's boot check runs them
    /// (firmware/ambrosia/check-boot.sh): `head` is the accelerator and the RAM, which differ by
    /// host; the rest is every host's.
    fn arm64_command_line(head: &[&str]) -> Vec<String> {
        let rest = [
            "-smp",
            "4",
            "-bios",
            "/d/ambrosia/ambrosia-0.1.3.bin",
            "-drive",
            "file=/vms/t/aux.img.trimmed,if=pflash,format=raw",
            "-drive",
            "file=/vms/t/disk.img,if=pflash,format=raw",
            "-drive",
            "file=/vms/t/aux.img.trimmed,if=none,id=aux,format=raw",
            "-drive",
            "file=/vms/t/disk.img,if=none,id=root,format=raw",
            "-device",
            "vmapple-virtio-blk-pci,variant=aux,drive=aux",
            "-device",
            "vmapple-virtio-blk-pci,variant=root,drive=root",
            "-netdev",
            "user,id=net0,ipv6=off,hostfwd=tcp:127.0.0.1:2326-:22",
            "-device",
            "virtio-net-pci,netdev=net0,id=net0,mac=52:54:00:aa:bb:cc,romfile=",
            "-device",
            "reims-graphics-pci,id=gfx",
            "-device",
            "loader,file=/d/prebuilt/Config-0123456789abcdef.cpio,addr=0x120000000,force-raw=on",
            "-display",
            "none",
            "-serial",
            "none",
            "-monitor",
            "none",
            "-qmp",
            "unix:/vms/t/qmp.sock,server=on,wait=off",
            "-pidfile",
            "/vms/t/qemu.pid",
        ];
        ["-name", "t"]
            .iter()
            .chain(head)
            .chain(&rest)
            .map(|s| s.to_string())
            .collect()
    }

    /// An arm64 VM on an x86-64 Linux host: vmapple under TCG, a thread for each vCPU, in memfd
    /// RAM, which the library on Linux needs.
    #[test]
    fn arm64_on_x86_64_linux_runs_on_tcg() {
        let inv = invocation(&arm64_vm(), &arm64_ctx(linux())).unwrap();
        let want = arm64_command_line(&[
            "-accel",
            "tcg,thread=multi",
            "-machine",
            "vmapple,uuid=0x9e35f01edc0c5536,memory-backend=ram,dump-guest-core=off",
            "-m",
            "8192M",
            "-object",
            "memory-backend-memfd,id=ram,size=8192M,share=on",
        ]);
        assert_eq!(unix_paths(&inv.args), want);
    }

    /// On arm64 Linux the same machine runs on KVM.
    #[test]
    fn arm64_on_arm64_linux_runs_on_kvm() {
        let arm_linux = Host {
            os: Os::Linux,
            arch: Arch::Aarch64,
        };
        let inv = invocation(&arm64_vm(), &arm64_ctx(arm_linux)).unwrap();
        let want = arm64_command_line(&[
            "-accel",
            "kvm",
            "-machine",
            "vmapple,uuid=0x9e35f01edc0c5536,memory-backend=ram,dump-guest-core=off",
            "-m",
            "8192M",
            "-object",
            "memory-backend-memfd,id=ram,size=8192M,share=on",
        ]);
        assert_eq!(unix_paths(&inv.args), want);
    }

    /// On a Mac it runs on HVF, the GIC in the kernel, in QEMU's own RAM: macOS has no memfd,
    /// and QEMU there refuses `dump-guest-core=off`.
    #[test]
    fn arm64_on_a_mac_runs_on_hvf() {
        let inv = invocation(&arm64_vm(), &arm64_ctx(mac())).unwrap();
        let want = arm64_command_line(&[
            "-accel",
            "hvf,kernel-irqchip=on",
            "-machine",
            "vmapple,uuid=0x9e35f01edc0c5536",
            "-m",
            "8192M",
        ]);
        assert_eq!(unix_paths(&inv.args), want);
        // The library's window is its own, as for an x86-64 VM.
        assert!(inv.unset.contains(&"REIMS_VGPU_WINDOW".to_string()));
    }

    /// The device has the VM's displays and their windows' first size; the VM's logs, its
    /// network and its ECID reach the command; and what an arm64 VM can't have is refused.
    #[test]
    fn arm64_takes_the_vms_settings() {
        let mut vm = arm64_vm();
        vm.config.display.count = 2;
        vm.config.display.resolution = "2560x1440".into();
        vm.config.logs.enabled = true;
        vm.config.network.ssh_port = None;
        vm.config.vmapple.as_mut().unwrap().ecid = 7;
        let inv = invocation(&vm, &arm64_ctx(mac())).unwrap();
        assert!(has(
            &inv.args,
            &[
                "-device",
                "reims-graphics-pci,id=gfx,displays=2,resolution=2560x1440"
            ]
        ));
        assert!(has(
            &inv.args,
            &["-machine", "vmapple,uuid=0x0000000000000007"]
        ));
        assert!(has(&inv.args, &["-netdev", "user,id=net0,ipv6=off"]));
        assert!(has(&inv.args, &["-serial", "file:/vms/t/logs/serial.log"]));
        assert!(inv.env.iter().any(|(k, _)| k == "REIMS_VGPU_LOG"));
        vm.config.network.enabled = false;
        let inv = invocation(&vm, &arm64_ctx(mac())).unwrap();
        assert!(has(&inv.args, &["-nic", "none"]));
        assert!(!inv.args.iter().any(|a| a.starts_with("virtio-net-pci")));

        let mut svga = arm64_vm();
        svga.config.graphics = Some(Graphics::VmwareSvga);
        assert!(invocation(&svga, &arm64_ctx(mac())).is_err());
        let mut usb = arm64_vm();
        usb.config.usb = vec![UsbDevice {
            id: "1234:5678".into(),
            port: None,
            name: String::new(),
        }];
        assert!(invocation(&usb, &arm64_ctx(mac())).is_err());
        assert!(invocation(&arm64_vm(), &ctx(mac())).is_err(), "no Ambrosia");
    }

    fn has(args: &[String], pair: &[&str]) -> bool {
        let args = unix_paths(args);
        args.windows(pair.len())
            .any(|w| w.iter().zip(pair).all(|(a, b)| a == b))
    }

    /// `args` with the `/` these tests' paths have where Windows joins the VM's paths with `\`.
    fn unix_paths(args: &[String]) -> Vec<String> {
        args.iter()
            .map(|a| {
                if cfg!(windows) {
                    a.replace('\\', "/")
                } else {
                    a.clone()
                }
            })
            .collect()
    }

    /// A q35 VM's whole command line on Linux, in order: what another host's profile leaves as
    /// it is.
    #[test]
    fn q35_on_linux_is_this_command_line() {
        let inv = invocation(&q35_vm(), &ctx(linux())).unwrap();
        let osk = format!("isa-applesmc,osk={OSK}");
        let want = [
            "-name",
            "t",
            "-machine",
            "q35,memory-backend=ram,dump-guest-core=off,max-ram-below-4g=1G",
            "-object",
            "memory-backend-memfd,id=ram,size=12288M,share=on,prealloc=on,prealloc-threads=6",
            "-accel",
            "kvm",
            "-m",
            "12288M",
            "-cpu",
            "Skylake-Client-noTSX-IBRS,kvm=on,vendor=GenuineIntel,+invtsc,vmware-cpuid-freq=on,\
             +ssse3,+sse4.2,+popcnt,+avx,+aes,+xsave,+xsaveopt,check",
            "-smp",
            "6,sockets=1,cores=6,threads=1",
            "-global",
            "ICH9-LPC.acpi-pci-hotplug-with-bridge-support=off",
            "-global",
            "ICH9-LPC.disable_s3=1",
            "-device",
            "pcie-root-port,bus=pcie.0,addr=0x1,slot=1,speed=16,width=32",
            "-device",
            &osk,
            "-drive",
            "if=pflash,format=raw,readonly=on,file=/vms/t/OVMF_CODE.fd",
            "-drive",
            "if=pflash,format=raw,file=/vms/t/OVMF_VARS.fd",
            "-smbios",
            "type=2",
            "-device",
            "ich9-ahci,id=sata,addr=0x2",
            "-drive",
            "id=opencore,if=none,format=qcow2,file=/vms/t/opencore.qcow2",
            "-device",
            "ide-hd,bus=sata.2,drive=opencore,bootindex=1",
            "-drive",
            "id=disk,if=none,format=qcow2,file=/vms/t/disk.qcow2",
            "-device",
            "ide-hd,bus=sata.3,drive=disk,rotation_rate=1",
            "-netdev",
            "user,id=net0,ipv6=off,hostfwd=tcp:127.0.0.1:2326-:22",
            "-device",
            "virtio-net-pci,netdev=net0,id=net0,mac=52:54:00:aa:bb:cc,addr=0x3",
            "-vga",
            "none",
            "-device",
            "reims-graphics-pci,id=gfx,bus=pcie.0,addr=0x4,\
             romfile=/d/firmware/ReimsGraphics-0123456789abcdef.rom",
            "-device",
            "qemu-xhci,id=xhci,addr=0x5,conditional-intr-mapping=on,p2=7,p3=7",
            "-device",
            "usb-kbd,id=kbd,bus=xhci.0",
            "-device",
            "usb-tablet,id=tablet,bus=xhci.0",
            "-audiodev",
            "pipewire,id=snd0",
            "-device",
            "virtio-sound-pci,id=audio,audiodev=snd0,addr=0x7,streams=1",
            "-device",
            "virtio-sound-pci,id=audioin,audiodev=snd0,addr=0x8,streams=1,input-streams=1",
            "-device",
            "ich9-intel-hda,id=hda,addr=0x1b",
            "-device",
            "hda-output,bus=hda.0,audiodev=snd0",
            "-display",
            "none",
            "-serial",
            "none",
            "-monitor",
            "none",
            "-qmp",
            "unix:/vms/t/qmp.sock,server=on,wait=off",
            "-pidfile",
            "/vms/t/qemu.pid",
        ];
        assert_eq!(unix_paths(&inv.args), want);
    }

    /// On Windows the machine runs on WHPX, its RAM shared memory of QEMU's RAM backend, which
    /// reims-vgpu's QEMU makes a pagefile-backed section there, without `dump-guest-core=off`,
    /// which QEMU on Windows refuses; the rest of the command line is Linux's.
    #[test]
    fn q35_on_windows_runs_on_whpx_in_shared_ram() {
        let windows = Host {
            os: Os::Windows,
            arch: Arch::X86_64,
        };
        let w = unix_paths(&invocation(&q35_vm(), &ctx(windows)).unwrap().args);
        let mut want = unix_paths(&invocation(&q35_vm(), &ctx(linux())).unwrap().args);
        for (option, value) in [
            ("-machine", "q35,memory-backend=ram,max-ram-below-4g=1G"),
            ("-object", "memory-backend-ram,id=ram,size=12288M,share=on"),
            ("-accel", "whpx"),
        ] {
            let i = want.iter().position(|a| a == option).unwrap();
            want[i + 1] = value.to_string();
        }
        assert_eq!(w, want);
        assert!(!w.iter().any(|a| a.contains("dump-guest-core")));
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
                "virtio-sound-pci,id=audio,audiodev=snd0,addr=0x7,streams=1",
                "-device",
                "virtio-sound-pci,id=audioin,audiodev=snd0,addr=0x8,streams=1,input-streams=1",
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

    /// QEMU's `x-avx512-xstate` only where each fact holds, asked in their order: KVM, the host's
    /// AVX-512 state at any layout, AMD's or Intel's, the guest's kexts in the image, then a QEMU
    /// that knows the property; neither the image nor QEMU is read where a fact before fails.
    #[test]
    fn avx512_xstate_only_where_each_fact_holds() {
        const ZEN4: Option<[u32; 3]> = Some([832, 896, 1408]);
        const INTEL: Option<[u32; 3]> = Some([1088, 1152, 1664]);
        let asked = std::cell::RefCell::new(Vec::new());
        let decide = |kvm, offsets, kexts: bool, knows: bool| {
            asked.borrow_mut().clear();
            avx512_xstate(
                kvm,
                offsets,
                || {
                    asked.borrow_mut().push("kexts");
                    kexts
                },
                || {
                    asked.borrow_mut().push("qemu");
                    knows
                },
            )
        };
        for offsets in [ZEN4, INTEL] {
            assert!(decide(true, offsets, true, true), "{offsets:?}");
            assert_eq!(*asked.borrow(), ["kexts", "qemu"]);
        }
        // On WHPX, and on a host whose OS keeps no AVX-512 state.
        for (kvm, offsets) in [(false, ZEN4), (false, INTEL), (true, None)] {
            assert!(!decide(kvm, offsets, true, true), "{kvm} {offsets:?}");
            assert!(asked.borrow().is_empty());
        }
        // An image without the guest's kexts, and an older QEMU.
        assert!(!decide(true, ZEN4, false, true));
        assert_eq!(*asked.borrow(), ["kexts"]);
        assert!(!decide(true, ZEN4, true, false));
    }

    /// `x-avx512-xstate=on` comes after `vmx=on` and before the VM's `cpu_flags`, so a VM's
    /// `x-avx512-xstate=off` there takes it away: QEMU applies `name=value` flags in order, the
    /// last one winning. Without it, the line is the one reims-ctl gave before it had the
    /// property, byte for byte, as on a host without AVX-512 state, WHPX or an older QEMU.
    #[test]
    fn avx512_xstate_comes_before_the_vms_cpu_flags() {
        let cpu = |vm: &Vm, nested_vmx, avx512_xstate| {
            let c = Ctx {
                nested_vmx,
                avx512_xstate,
                ..ctx(linux())
            };
            let inv = invocation(vm, &c).unwrap();
            let i = inv.args.iter().position(|x| x == "-cpu").unwrap();
            inv.args[i + 1].clone()
        };
        let head = format!(
            "{},kvm=on,vendor=GenuineIntel,+invtsc,vmware-cpuid-freq=on",
            default_cpu_model()
        );
        let flags = default_cpu_flags();
        let vm = q35_vm();
        assert_eq!(
            cpu(&vm, true, true),
            format!("{head},vmx=on,x-avx512-xstate=on,{flags}")
        );
        assert_eq!(
            cpu(&vm, false, true),
            format!("{head},x-avx512-xstate=on,{flags}")
        );
        assert_eq!(cpu(&vm, true, false), format!("{head},vmx=on,{flags}"));
        assert_eq!(cpu(&vm, false, false), format!("{head},{flags}"));
        let mut off = q35_vm();
        off.config.q35.as_mut().unwrap().cpu_flags = format!("{flags},x-avx512-xstate=off");
        assert_eq!(
            cpu(&off, false, true),
            format!("{head},x-avx512-xstate=on,{flags},x-avx512-xstate=off")
        );
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

    /// On DirectSound, which opens Windows' devices as QEMU starts: with no recording device the
    /// input device records from an audiodev of its own, `none`, and with no output device the
    /// VM's sound is all `none`, each said at the start; the other backends, and `none` itself,
    /// are as the VM names them whatever the host has.
    #[test]
    fn dsound_goes_to_none_where_windows_has_no_sound_device() {
        let devices = |output, input| crate::os::SoundDevices { output, input };
        let mut vm = q35_vm();
        vm.config.audio.backend = Some("dsound".into());
        let args = |sound| {
            invocation(
                &vm,
                &Ctx {
                    sound,
                    ..ctx(linux())
                },
            )
            .unwrap()
            .args
        };
        let a = args(devices(true, true));
        assert!(has(&a, &["-audiodev", "dsound,id=snd0"]), "{a:?}");
        assert!(!a.iter().any(|x| x.contains("sndin")), "{a:?}");
        let a = args(devices(true, false));
        assert!(has(&a, &["-audiodev", "dsound,id=snd0"]), "{a:?}");
        assert!(has(&a, &["-audiodev", "none,id=sndin"]), "{a:?}");
        assert!(has(
            &a,
            &[
                "-device",
                "virtio-sound-pci,id=audioin,audiodev=sndin,addr=0x8,streams=1,input-streams=1"
            ]
        ));
        assert!(has(
            &a,
            &[
                "-device",
                "virtio-sound-pci,id=audio,audiodev=snd0,addr=0x7,streams=1"
            ]
        ));
        let a = args(devices(false, false));
        assert!(has(&a, &["-audiodev", "none,id=snd0"]), "{a:?}");
        assert!(!a.iter().any(|x| x.contains("sndin")), "{a:?}");
        assert!(sound_note("dsound", devices(true, true)).is_none());
        assert!(
            sound_note("dsound", devices(true, false))
                .unwrap()
                .contains("records silence")
        );
        assert!(
            sound_note("dsound", devices(false, true))
                .unwrap()
                .contains("without sound")
        );
        for other in ["pipewire", "wav", "none"] {
            assert_eq!(sound_routing(other, devices(false, false)), (other, false));
            assert!(sound_note(other, devices(false, false)).is_none());
        }
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

    /// A disk moved to virtio is a virtio-blk device of its own, off the AHCI controller, with
    /// SATA's drive options.
    #[test]
    fn a_disk_on_virtio_is_a_device_of_its_own() {
        let mut vm = q35_vm();
        vm.config.q35.as_mut().unwrap().disk_controller = DiskController::Virtio;
        let a = invocation(&vm, &ctx(linux())).unwrap().args;
        assert!(has(
            &a,
            &[
                "-drive",
                "id=disk,if=none,format=qcow2,file=/vms/t/disk.qcow2",
                "-device",
                "virtio-blk-pci,drive=disk,addr=0x6",
            ]
        ));
        assert!(
            !a.iter().any(|x| x.ends_with("drive=disk,rotation_rate=1")),
            "{a:?}"
        );
        // No other device takes its slot.
        let mut slots: Vec<&str> = a
            .iter()
            .flat_map(|x| x.split(','))
            .filter_map(|o| o.strip_prefix("addr="))
            .collect();
        assert!(slots.contains(&"0x6"), "{a:?}");
        let n = slots.len();
        slots.sort_unstable();
        slots.dedup();
        assert_eq!(slots.len(), n, "{a:?}");
        // The installer stays on AHCI, on port 4, and the disk drops its flushes meanwhile.
        vm.config.q35.as_mut().unwrap().installer = Some("/isos/BaseSystem.img".into());
        let a = invocation(&vm, &ctx(linux())).unwrap().args;
        assert!(has(&a, &["-device", "ide-hd,bus=sata.4,drive=installer"]));
        assert!(has(
            &a,
            &[
                "-drive",
                "id=disk,if=none,format=qcow2,file=/vms/t/disk.qcow2,cache.no-flush=on",
                "-device",
                "virtio-blk-pci,drive=disk,addr=0x6",
            ]
        ));
    }

    /// The system disk drops the guest's flushes while the installer is attached, on SATA and on
    /// NVMe, and keeps them once it is ejected.
    #[test]
    fn the_disk_drops_flushes_only_while_the_installer_is_attached() {
        let mut vm = q35_vm();
        let a = invocation(&vm, &ctx(linux())).unwrap().args;
        assert!(has(
            &a,
            &[
                "-drive",
                "id=disk,if=none,format=qcow2,file=/vms/t/disk.qcow2"
            ]
        ));
        assert!(!a.iter().any(|x| x.contains("no-flush")), "{a:?}");
        vm.config.q35.as_mut().unwrap().installer = Some("/isos/BaseSystem.img".into());
        let a = invocation(&vm, &ctx(linux())).unwrap().args;
        assert!(has(
            &a,
            &[
                "-drive",
                "id=disk,if=none,format=qcow2,file=/vms/t/disk.qcow2,cache.no-flush=on"
            ]
        ));
        // Only the system disk: the installer's overlay and OpenCore's image keep theirs.
        assert_eq!(
            a.iter().filter(|x| x.contains("no-flush")).count(),
            1,
            "{a:?}"
        );
        vm.config.q35.as_mut().unwrap().disk_controller = DiskController::Nvme;
        let a = invocation(&vm, &ctx(linux())).unwrap().args;
        assert!(has(
            &a,
            &[
                "-drive",
                "id=disk,if=none,format=qcow2,file=/vms/t/disk.qcow2,cache.no-flush=on,\
                 discard=unmap"
            ]
        ));
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
        // On port 4 again where the disk is on a controller of its own, whatever its AHCI port.
        for d in [DiskController::Nvme, DiskController::Virtio] {
            vm.config.q35.as_mut().unwrap().disk_controller = d;
            let a = invocation(&vm, &ctx(linux())).unwrap().args;
            assert!(has(&a, &["-device", "ide-hd,bus=sata.4,drive=installer"]), "{d:?}");
        }
        vm.config.q35.as_mut().unwrap().disk_controller = DiskController::Ahci;
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
            select: Some("10de:2204".into()),
            name: Some("NVIDIA GeForce RTX 3090".into()),
        };
        vm.config
            .extra_env
            .insert("REIMS_VGPU_LOG".into(), "trace".into());
        let inv = invocation(&vm, &ctx(linux())).unwrap();
        let env: BTreeMap<_, _> = inv.env.iter().cloned().collect();
        assert_eq!(env["REIMS_VGPU_LOG"], "trace");
        assert_eq!(env["REIMS_VGPU_WINDOW_BORDERLESS"], "1");
        // The GPU first, by its IDs, the loader limited to no driver.
        assert_eq!(env["MESA_VK_DEVICE_SELECT"], "10de:2204");
        assert_eq!(env["VK_LOADER_DEVICE_SELECT"], "0x10de:0x2204");
        for limit in [
            "VK_DRIVER_FILES",
            "VK_ICD_FILENAMES",
            "VK_LOADER_DEVICE_ID_FILTER",
        ] {
            assert!(!env.contains_key(limit), "{limit} in {env:?}");
        }
        assert!(has(&inv.args, &["-serial", "file:/vms/t/logs/serial.log"]));
        assert!(has(&inv.args, &["-msg", "timestamp=on"]));
        assert!(!inv.unset.contains(&"REIMS_VGPU_LOG".to_string()));
    }

    /// A GPU its settings name by its driver's manifest alone (a CPU implementation, a GPU
    /// another manifest reaches too, or a manifest given by its path; crate::gpu) limits the
    /// loader to that driver, keeping the GPU's device ID where the driver drives several. A
    /// select that names no IDs is refused; on the Mac no GPU is chosen.
    #[test]
    fn a_gpu_named_by_its_driver_alone_limits_the_loader_to_it() {
        let mut vm = q35_vm();
        vm.config.gpu = Gpu {
            driver_file: Some("/icd/radeon.json".into()),
            device_id: Some(0x744c),
            select: None,
            name: Some("AMD Radeon RX 7900 XTX (RADV NAVI31)".into()),
        };
        let env: BTreeMap<_, _> = invocation(&vm, &ctx(linux()))
            .unwrap()
            .env
            .into_iter()
            .collect();
        assert_eq!(env["VK_DRIVER_FILES"], "/icd/radeon.json");
        assert_eq!(env["VK_ICD_FILENAMES"], "/icd/radeon.json");
        assert_eq!(env["VK_LOADER_DEVICE_ID_FILTER"], "0x744c");
        assert!(!env.contains_key("MESA_VK_DEVICE_SELECT"), "{env:?}");
        assert!(!env.contains_key("VK_LOADER_DEVICE_SELECT"), "{env:?}");
        vm.config.gpu.device_id = None;
        let env: BTreeMap<_, _> = invocation(&vm, &ctx(linux()))
            .unwrap()
            .env
            .into_iter()
            .collect();
        assert_eq!(env["VK_DRIVER_FILES"], "/icd/radeon.json");
        assert!(!env.contains_key("VK_LOADER_DEVICE_ID_FILTER"), "{env:?}");
        vm.config.gpu.select = Some("radeon".into());
        assert!(invocation(&vm, &ctx(linux())).is_err());
        vm.config.gpu.select = Some("1002:744c".into());
        let env = invocation(&vm, &ctx(mac())).unwrap().env;
        assert!(
            !env.iter()
                .any(|(k, _)| k.starts_with("VK_") || k.starts_with("MESA_")),
            "{env:?}"
        );
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
        // One display, the device's own, names none.
        assert!(!a.iter().any(|x| x.contains("displays")), "{a:?}");
        // VMware SVGA keeps the machine's RAM below 4 GiB as it was.
        vm.config.graphics = Some(Graphics::VmwareSvga);
        let a = invocation(&vm, &ctx(linux())).unwrap().args;
        assert!(!a.iter().any(|x| x.contains("max-ram-below-4g")), "{a:?}");
    }

    /// A VM of several displays gives reims-vgpu's device their count, and nothing else of the
    /// command changes: the resolution is every window's first size, and borderless every
    /// window's, as for one display. VMware SVGA has one display, whatever the count.
    #[test]
    fn reims_vgpu_has_the_vms_displays() {
        let one = invocation(&q35_vm(), &ctx(linux())).unwrap();
        let mut vm = q35_vm();
        for n in 2..=DISPLAYS_MAX {
            vm.config.display.count = n;
            let inv = invocation(&vm, &ctx(linux())).unwrap();
            let mut want = one.clone();
            for a in &mut want.args {
                if a.starts_with("reims-graphics-pci,") {
                    a.push_str(&format!(",displays={n}"));
                }
            }
            assert_eq!(inv, want, "{n}");
        }
        vm.config.display.resolution = "2560x1600".into();
        vm.config.display.borderless = true;
        let inv = invocation(&vm, &ctx(linux())).unwrap();
        assert!(
            has(
                &inv.args,
                &[
                    "-device",
                    "reims-graphics-pci,id=gfx,bus=pcie.0,addr=0x4,\
                     romfile=/d/firmware/ReimsGraphics-0123456789abcdef.rom,displays=4,\
                     resolution=2560x1600"
                ]
            ),
            "{:?}",
            inv.args
        );
        let env: BTreeMap<_, _> = inv.env.iter().cloned().collect();
        assert_eq!(env["REIMS_VGPU_WINDOW_BORDERLESS"], "1");
        vm.config.graphics = Some(Graphics::VmwareSvga);
        let a = invocation(&vm, &ctx(linux())).unwrap().args;
        assert!(
            has(&a, &["-device", "vmware-svga,id=gfx,bus=pcie.0,addr=0x4"]),
            "{a:?}"
        );
        assert!(!a.iter().any(|x| x.contains("displays")), "{a:?}");
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
