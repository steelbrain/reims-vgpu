# Running QEMU on the library

`vendor/qemu` is QEMU with `reims-graphics-pci`, the GPU device our own
kexts drive (`guest/`), in C on this library's C API. `make qemu` builds
its `qemu-system-x86_64` on the checkout's library, linked statically
through pkg-config. On Linux it runs x86-64 macOS guests under KVM, and on
Windows under WHPX ("Windows hosts", below), and users run them with
`reims-ctl`, whose installer, `scripts/install-ctl.sh` (on Windows run by
`scripts/install.ps1`), builds the same QEMU on the library's Vulkan
platform. This file says how QEMU is built and uses the library, and the
measured facts QEMU's side and the guests rely on.

`make qemu` builds no `qemu-system-aarch64`. QEMU's `vmapple` machine,
without the Apple paravirtualized GPU it once had, builds again in an
`aarch64-softmmu` build, under HVF or TCG, for Ambrosia, our own arm64
firmware ("arm64 guests", below). On an arm64 Mac or arm64 Linux,
`reims-ctl` makes arm64 VMs by restoring macOS from an IPSW onto their disk
(`crates/reims-restore`), which runs no QEMU, and doesn't start them;
`scripts/install-ctl.sh` builds no QEMU there. Each VM needs one boot
before it is ready: macOS makes its Data volume at its first boot and resets
the VM, and `reims-restore finish-eacs DIR` then makes it bootable for Setup
Assistant (`crates/reims-restore/docs/design.md`, "The Data volume, by
EACS").

No test runs QEMU, so a VM boot is the only check of QEMU's side of the
API ([lessons.md](lessons.md)). Each measurement below names its host and
guest. Anything not measured is marked **unknown**.

## The QEMU build

```sh
make qemu          # cargo xtask dist, then build/qemu/qemu-system-x86_64
```

`make qemu` runs `cargo xtask dist`, which writes the library into the dist
tree, `<target>/dist-vulkan` (`target`, or `CARGO_TARGET_DIR`): on Linux on
Vulkan, and on a Mac over MoltenVK. Then it configures `vendor/qemu` out of tree in
`build/qemu`, on Linux:

```sh
PKG_CONFIG_PATH=$PWD/target/dist-vulkan/lib/pkgconfig vendor/qemu/configure \
  --target-list=x86_64-softmmu --enable-reims-vgpu --disable-docs
```

It configures once, and again whenever `QEMU_CONFIGURE` (extra configure
options) is set, or `config.status` came from another source or another
`PKG_CONFIG_PATH`, or names an option vendor/qemu's configure no longer
takes (`--enable-pvg`, `--apple-gfx-backend`, `--pvg-framework`) or the
`aarch64-softmmu` target an earlier `make qemu` built, or the dist tree's
`reims-vgpu.pc` differs from the copy it keeps of the one it configured
with (`build/qemu/reims-vgpu.pc.configured`), or vendor/qemu's
`configure`, `scripts/meson-buildoptions.sh`, `pythondeps.toml` or `VERSION`
is newer than `build/qemu/config-host.mak`, as QEMU's own Makefile
configures again. Then ninja builds `qemu-system-x86_64`, at its own default
parallelism or a caller's `-jN` (QEMU's Makefile, called with no `-j`, hands
ninja `-j1`), and has meson regenerate `build.ninja` where a meson file
changed. **Measured** on um870 (16 threads, vendor/qemu 1422502b,
2026-11-13): a clean `make qemu` after the library's dist built
QEMU's 1951 steps in 310 s through QEMU's Makefile at `-j1`, and in 59 s
through ninja at its default, to the same binary. configure fetches meson subprojects (`dtc`,
`keycodemapdb`, ...) into the source's `subprojects/`, which its own
`.gitignore` covers.

- **The library.** configure records `PKG_CONFIG_PATH` in `config.status`,
  and meson keeps it as its `pkg_config_path`, so ninja's own reconfigures
  find the package too. The `reims_vgpu` dependency (`--enable-reims-vgpu`)
  is `dependency('reims-vgpu', method: 'pkg-config', static: true)`, looked
  for on Linux, macOS and Windows hosts, so QEMU links `libreims_vgpu.a` and what its
  `Libs.private` names, and records no path to the library: nothing of it
  is loaded from a file at run time.
- **The package's flags are configure's.** meson reads `reims-vgpu.pc` at
  configure and keeps what it read, so a build configured against an older
  package relinks a newer archive without the libraries it added to
  `Libs.private`, and fails to link; hence the reconfigure when the package
  changes ([lessons.md](lessons.md)).
- **The device.** Where the library is found, `CONFIG_REIMS_GRAPHICS` is
  set for every target, whatever its accelerator, and
  `hw/display/meson.build` builds `reims-graphics-pci.c`
  (`CONFIG_REIMS_GRAPHICS_PCI`). configure's summary says whether it found
  reims-vgpu. The device has no default option ROM: `reims-ctl` gives it
  `firmware/prebuilt/ReimsGraphics.rom` as its `romfile`.
- **Hosts.** On Linux the guest runs under KVM. On a Mac the same
  `qemu-system-x86_64` runs it under TCG, over MoltenVK, which realizes the
  device and checks it paused ("The reims-graphics device", below). On
  Windows QEMU builds in MSYS2's CLANG64 environment and runs the guest
  under WHPX ("Windows hosts", below).

## What vendor/qemu carries

`vendor/qemu` is upstream QEMU with these commits on top, on branch
`steelbrain/reims-vgpu-rebase-2`: upstream master at `e9fa1f66f0` (a merge
into staging of 2026-11-13) with the commits of branch `steelbrain/reims-vgpu`
rebased onto it, in their order. That branch, which stays at `1f6f4ea455`,
started from upstream's `b833716681` (a merge into staging of 2026-06-19) and
merged upstream master into it (`049fab62bb`, then `49873ce25e`, at
`9c88811b81` of 2026-09-30); the measurements and the history further down
name its commits. Seven of its commits are not on the rebased branch, upstream
master having them: five fixes to `audio/pwaudio.c`; the thread pool's floor
of one thread when a job finished before `thread_pool_submit_immediate`
resized it (`850b5a4bb4`, upstream's `2566b6e902`); and the fix of QMP's
`x-query-virtio-status` and HMP's `info virtio-status`, which crashed on a
virtio device (`d151bcd9`, upstream's `b501f21312`).

| what | commits |
|---|---|
| the reims-graphics device's transport, for the library's own guest driver: `reims-graphics-pci` (1234:5247, class 0x0380), on the library's graphics device (`rvg_graphics_device_*`); BAR 0 its registers with MSI-X from 0x800, the doorbell an ioeventfd, BAR 2 the library's region as RAM; built for every target where the library is found (`CONFIG_REIMS_GRAPHICS`), x86-64 under TCG on a Mac among them ("The reims-graphics device", below) | `a6795a99`, `03d4af6f` |
| the device gives the library the guest RAM's map and the files behind it | `e3e5ef1d` |
| the device tells the library the guest's architecture (`guest_arch`: arm64 where `target_aarch64()`, else x86-64), which the device's `isFramebufferReadSupported` answer follows, and the class of Mac whose GPU results the library's devices give (`GuestArch::mac_class`) | `308b5fc6` |
| the device raises its interrupt through an irqfd where KVM's in-kernel irqchip takes MSI-X through one | `967400b3` |
| on WHPX with the hypervisor's APIC (kernel-irqchip), the raising thread asks the hypervisor for the vector's MSI itself (`whpx_send_msi_direct`), with no BQL, so a hold of the main loop (a PNG screendump's 120 ms) no longer holds the guest's completions back | `243971f2` |
| on WHPX with kernel-irqchip, a vCPU with no NMI, SMI, INIT, TPR or HARD request pending runs on without the BQL (`whpx_vcpu_pre_run`, as `kvm_arch_pre_run` with KVM's in-kernel irqchip), so an exit to the device's lockless BARs no longer waits for the BQL's holder | `0fdbf381` |
| a command buffer is taken, and answered, on the vCPU that rings its queue's doorbell, with the plain register write | `945a86b7`, `f16ca588`, `142c8e72` |
| a command queue's plain doorbell, which the library routes where the guest took `DOORBELL_ASKS` (`route_doorbell`), is an ioeventfd matching the ring's id that signals the queue's eventfd, added and dropped under the BQL as the library asks, at most 128 at once | `db9cb35f` |
| at QEMU's quit the device's release runs with the BQL let go, so a `route_doorbell` waiting for it finds the routes closed and returns, holding the release up no more | `92db49c1` |
| BAR 4, the doorbell pages: 4 MiB of 32-bit MMIO, 256 pages of 16 KiB, whose writes go to the library (`rvg_graphics_device_doorbell_write`) and whose reads answer 0; the device offers them (`doorbell_pages`), and a queue of a context with a page has its plain doorbell's ioeventfd at the page's first byte (`route_doorbell_page`), not on DOORBELL | `83b7255c` |
| the console shows the display (`rvg_graphics_device_read_display`), so QMP's `screendump` saves the guest's frame, and has the guest's cursor | `aebdef13`, `00d11f7e` |
| a screendump waits for the console's read of the display at most 3 s from the read's start, then is answered with what the console shows, so QMP goes on while the GPU never answers the read | `fa600694` |
| the device's realize warns where BAR 2 would get no address: on a PC machine with RAM at 1 GiB ("The reims-graphics device", below) | `b7d7fdaf` |
| the window opens at the `resolution` property's size, 1920x1080 without it | `9bf62d84` |
| the `displays` property, 1 to 4 (1 by default, others refused at realize): display i is head i, with a console, a window under `-display none` and a cursor of its own; console i reads display i (`rvg_graphics_device_read_display_at`), so QMP's `screendump` of head i saves it ("The reims-graphics device", below) | `010bf827` |
| file-backed guest RAM (the memfd of `reims-ctl`'s VMs) mapped 2 MiB-aligned on Linux, as anonymous RAM is, so transparent huge pages can back it where the host allows them ("Linux hosts: huge pages for guest RAM", below) | `5e83e90dc2` |
| `isa-applesmc` answers GET_KEY_BY_INDEX, listing its keys by name and 0xb8 past the last, so the key walk macOS 15.4 and later make at boot ends ("x86-64 macOS guests: the SMC", below) | `83de566436` |
| QMP's `screendump` in PPM, its default format, writes rows of width × 3 bytes. Upstream's wrote each row with its line buffer's stride, which pixman rounds up to 4 bytes, so a display whose width is not a multiple of 4 (a window of 3662 × 2039 pixels) read back sheared, its colours rotating from row to row; PNG's rows were right. `tests/qtest/display-screendump-test.c` dumps `bochs-display` at widths of 64, 66 and 67 | `e1360002ad` |
| the input layer's smooth `scroll` events (`qemu_input_queue_scroll`, `qemu_input_scrolls_smoothly`, QMP `input-send-event`); a device without them (`usb-tablet`, which every x86-64 VM has) gets the event's wheel notches | `9017d7e47e` |
| `vmapple` ("arm64 guests", below): patches 1 to 6 of the RFC v7 series "vmapple: make it work on the latest macOS release" (Mohamed Mediouni), and the branch's commits for HVF's ISV=0 data aborts and in-kernel vGIC, the secondary CPUs' pointer authentication keys, macOS 14 and later and macOS 27 guests, the AES engine's refusal without a crypto library, and `usb-vmapple-tablet`, Virtualization.framework's USB pointing device | `00eb197889`, `4d4daab1a0`, `1161969f05`, `d24cde8e6e`, `ab4c3018f4`, `d10ee1dd7b`, `1d0776ac53`, `3deaf0f735`, `74d5134fb9`, `29043c9b34`, `eb67b81577`, `743bbba8f9`, `ffca79f534`, `6a54a097bd`, `eae5dd4a5c`, `4a39ef8d54`, `d01f28c9ed`, `93f9f40cdc` |
| apple-gfx, Apple's paravirtualized GPU device on the library, with its windows, its option ROM, its IOSurface device and the VideoToolbox and M2 scaler devices, removed: `vmapple`, whose display was `apple-gfx-mmio`, stayed in the tree unbuilt until the next row's commits; the option that finds the library is `reims_vgpu` (`--enable-reims-vgpu`) | `96e1b470` |
| `vmapple` for Ambrosia, built by default in an `aarch64-softmmu` build again: the machine without apple-gfx's display; no kernel-entry hook (the firmware makes macOS's kernel-entry changes, so AVPBooter boots only macOS 12 and 13 guests); RAM at 4 GiB, with the PCIe MMIO window from 0x50000000 below it for 1 GiB BARs (AVPBooter does not boot on this layout); TCG with the `max` CPU and a 24 MHz generic timer; the guest's write to its DMA configuration doorbell at 0x4ffff8 ignored, which panicked a macOS 27.0.1 kernel under TCG; and `virtio-net-pci` without its control queue in the machine's defaults, since one boot in four to six of a macOS 26 or 27 guest missed the completion of its first control command and never started DHCP | `15989559`, `b6821b78`, `981994e8`, `c629d147`, `42e7766d`, `1b3509f3` |
| BAR 2 is one RAM region with no dirty log and no `scanout_written` callback: the library's window draws the scanout again at each refresh while it shows no presented surface | `7365f20e` |
| `vmapple`'s second clock, `vmapple-rtc` at 0x30240000 as Virtualization.framework has it: a 64-bit microsecond count from the Unix epoch that follows `-rtc`, an ID register reading 0xf001 and no interrupt, beside the PL031. macOS 14 and later keep their RTC offset in it; without it each boot logs "RTC reset likely" and steps the clock back to the disk's last write until a time server answers. macOS 12 and 13 have no driver for it | `665035aa` |
| `vmapple`'s `max` CPU offers no SME by default under TCG (an optional compat default: under HVF the CPU gets its `sme` property only as its vCPU starts, after the default would apply, so HVF's CPU keeps what the host gives; `-cpu max,sme=on` turns it back on): TCG's streaming vectors reach 2048 bits, and a macOS 14.6.1 kernel sizes each thread's SME state zone from the longest, 74328 bytes, and panics with "zone_create: element size too large". 15.6.1, 26.6.2 and 27.0.1 boot either way; an M1 to M3 under HVF offers no SME | `a25cd13a`, `d4489ead` |
| `vmapple` under KVM on an arm64 Linux host: KVM passes the guest's PSCI CPU_ON calls to QEMU (an SMCCC filter, `KVM_ARM_VM_SMCCC_FILTER`, Linux 6.4 and later; at `-smp` above 1 a kernel without it is fatal), which starts the CPU through `arm_set_cpu_on()` as under HVF and TCG, so its secondary CPUs take the boot CPU's pointer authentication keys (`cpu-on-copies-pauth-keys`) instead of KVM's reset values; and its CPUs get no PMU (vmapple initializes none, and KVM runs no vCPU with one uninitialized) and PSCI 1.1, which macOS's ApplePSCI takes where it refuses the 1.3 current kernels give. Tested by the Asahi Linux experiment on an M2 Pro (Linux 6.19.14, macOS 27.2.1, `-smp 4`); under HVF and TCG nothing changes | `3d06c211`, `8dda3627`, `055cc6c4` |
| each HVF vCPU thread runs at the user-initiated QoS, above the default QoS a host build runs at and below the host's user-interactive work, so a guest keeps its share of the CPU beside a build (HVF is built on macOS alone) | `2c3a3650` |
| HVF's in-kernel GIC writes each SPI's enable, pending, active and group bits, trigger and priority to that SPI's own distributor registers, and clears the set/clear pairs with ones, as the KVM GIC does, so a guest reboot clears the last boot's SPIs: kept, the MSIs a panicking macOS left pending reached AppleVirtIO in the next boot before it had set up its queues (a data abort in `AppleVirtIOQueue::handleEvent`) | `47098463` |
| `vmapple`'s AES engine resets as at power-on, its FIFO, keys, IVs and mode with its registers, and lowers its interrupt: the words a panicking macOS left in its FIFO otherwise led the next boot's first command, which failed as invalid ("AES Error: IntStatus 0x4") | `06688caa` |
| the PL011's and the PL061's resets lower their interrupt lines: an in-kernel GIC (HVF's, KVM's) keeps a line's level across its own reset, so an interrupt raised in the last boot was otherwise pending again in the next, with nothing in the device behind it | `553e4c60` |
| `vmapple` under KVM answers macOS's pointer authentication calls (SMCCC 0xC1000000 to 0xC10000FF) as HVF and TCG do: KVM passes them to QEMU, which writes NOT_SUPPORTED to x0 alone, where KVM's own answer also zeroes x1 to x3 and XNU keeps x3 of GET_DEFAULT_KEYS (launchd then died of a pointer authentication failure); and KVM's data aborts with no instruction syndrome (pair, writeback and SIMD&FP accesses, `KVM_EXIT_ARM_NISV`) are emulated by the ISV=0 emulator HVF used, now `target/arm/isv0-mmio.c`, in a strict mode that changes no register and leaves the external abort where QEMU has no device at the address (the first was our Ambrosia module's ECAM read `ldr w9, [x8, #16]!`, before `8e7e0aa71` wrote the module's and the arm64e kext's device accesses as plain loads and stores). HVF calls the moved emulator as before. Tested by the Asahi Linux experiment on an M2 Pro: macOS 27.2.1 under KVM passes check-boot's 14 checks with `84f164fff`'s module and kexts, and does so on a QEMU without the ISV=0 commits too, so neither they nor macOS's own code through the desktop and a Metal app took an ISV=0 abort | `d5c89ec9`, `2b067363`, `af7af708`, `4445c480` |
| `usb-host` sets an interface's alternate setting (`libusb_set_interface_alt_setting`) on a thread-pool worker and completes the guest's SET_INTERFACE when the call returns, where it made the call holding the BQL; the device's other control requests, a reset, closing it and QEMU's exit wait for a call in flight ("x86-64 macOS guests: USB passthrough", below) | `3e24b0f5` |
| `qemu-xhci` on a PCI Express bus sets HCCPARAMS1's LTC, with PCIe's LTR capability, and completes the Set Latency Tolerance Value command with Success, keeping its BELT; on conventional PCI it refuses the command with TRB Error as before. HCCPARAMS2 (offset 0x1c, xHCI 1.1) reads 0 | `aa598d74`, `c16cb1f8`, `5412dd06` |
| `qemu-xhci`'s interrupters moderate their interrupts as IMOD asks (xHCI 4.17.2): an interrupt asserts when a non-blocking event awaits, IMODC is 0 and the event handler is not busy, and loads IMODC with IMODI, 4000 (1 ms) from a reset | `9fa8e2bc`, `26c17020` |
| `qemu-xhci` completes each command with a Command Completion Event of its own (xHCI 6.4.2.2): No Op, Set LTV, Get Port Bandwidth, a command it does not support and an Enable Slot that finds no free slot complete with Slot ID 0, and a command for a slot completes with that Slot ID whether or not the slot is enabled. Upstream's kept the previous command's Slot ID across one doorbell's commands, gave an NEC firmware revision reply the previous command's completion code, and gave every command after an NEC reply the reply's type and fields; `tests/qtest/usb-hcd-xhci-test.c` queues both kinds at one doorbell | `347a2204`, `3b327429` |
| while a `usb-host` device's SET_INTERFACE call runs on its worker, the main loop makes no usbfs call on the device, each of which would wait for the call: it handles no libusb events, so every usb-host device's completions wait for the call, and the device's submits and cancels wait in order in a queue of its own, made when the call returns ("x86-64 macOS guests: USB passthrough", below) | `db125fda` |
| `qemu-xhci` posts each Transfer Event a TD's TRBs owe after a Short Packet (xHCI 4.10.1.1.2): the TRB the packet is met on, where it has ISP or IOC, and each later TRB of the TD with IOC, with the same residue. Upstream's posted one event per TD, so macOS's isoch IN TDs of an Isoch TRB chained to a Normal TRB never completed, and a passed-through camera's stream stopped before its first frame ("x86-64 macOS guests: USB passthrough", below); `tests/qtest/usb-hcd-xhci-test.c` runs TDs of two TRBs into usb-kbd's reports | `ae97f3f5`, `1890afed` |
| WHPX answers CPUID leaf 4 from the CPU model: its exit list lacked the leaf, so Hyper-V answered it with no cache levels, and macOS panicked 1.3 s into boot ("Windows hosts", below) | `cc128f0e` |
| on Windows, shared RAM (`memory-backend-ram,share=on`, any RAM_SHARED block) is a view of a section the paging file backs, whose handle the RAMBlock keeps (`qemu_ram_get_section`); `qemu_win32_map_alloc` passes a size's high 32 bits ("Windows hosts: guest RAM in a section", below) | `6c2cae16` |
| `reims-graphics-pci` on Windows: meson looks for the library there too, the memory walk passes each range of RAM that is a view of a section with the section's handle and the range's offset in it, and the control ring's doorbell is the EventNotifier's event | `a86885f8` |
| the `dsound` audiodev plays where Windows has no recording device: its capture object is optional, where realize failed without one and QEMU exited ("Could not initialize DirectSoundCapture"); an input voice then fails to initialize, and reims-ctl gives such a host's input device a `none` audiodev of its own ("Windows hosts: sound", below) | `e435673c` |

The branch's other commits are the apple-gfx work `96e1b470` removes, and
`cfaadbfa09`, an HVF workaround that `478c28ff70` reverts, upstream having
fixed the same assertion, and `2c9e6ed5` and `fa464041`, which correct
comments and the machine's guide. `docs/system/arm/vmapple.rst` describes the
machine and how to run it on Ambrosia.

## The reims-graphics device

`reims-graphics-pci` is the device of the library's own guest driver for
x86-64 macOS guests (`docs/graphics-kext/design.md`): its own PCI ID,
1234:5247, class 0x0380, which Apple's paravirtualized graphics kext never
matches. Its host side is the library's graphics device
(`rvg_graphics_device_*`, docs/api.md); `hw/display/reims-graphics-pci.c`
is its transport:
- **BAR 0**, 4 KiB: the library's 32-bit registers below 0x800, the MSI-X
  table at 0x800 and its PBA at 0xc00, one vector. The vCPU calls the
  library itself, with no BQL: an access waits at most for the command the
  control thread is taking. A write of 0, the control ring's id, to
  DOORBELL (0x20) is taken by an ioeventfd (an eventfd on Linux, a pipe on
  a Mac), whose file the library's control thread polls. So is a write of
  a command queue's ring the library routes (`route_doorbell`, on Linux,
  where the guest took `DOORBELL_ASKS`): an ioeventfd matching the ring's id
  exactly signals the queue's eventfd, which its thread polls, so the vCPU
  goes back to the guest with no exit to QEMU. The device adds and drops
  these under the BQL as the library's queues are made and end, at most 128
  at once. Every other register write goes to the library through
  `rvg_graphics_device_mmio_write_taking`: a write of a command queue's ring
  (with bit 0 set, where the guest took `DOORBELL_ASKS`) may answer the
  queue's next command buffer on the vCPU, posting its `CB_SUBMITTED` and
  raising the interrupt inside the call, and wakes the queue's thread, which
  takes the buffer either way.
- **BAR 4**, 4 MiB of 32-bit MMIO: the doorbell pages, 256 of 16 KiB, which
  the guest's kext binds to its contexts and maps into their processes
  (design.md, "BAR4: the doorbell pages"). Reads answer 0, and every write
  goes to the library (`rvg_graphics_device_doorbell_write`), on the vCPU
  with no BQL, but a write the library routes: a command queue whose context
  has a page has its plain doorbell's ioeventfd at the page's first byte
  (`route_doorbell_page`), matching the ring's id, in place of DOORBELL's.
  The device always has it and offers the pages (`doorbell_pages`), and the
  routes on the pages count among the 128.
- **BAR 2**, 64-bit and prefetchable: the library's region
  (`rvg_graphics_device_region`), 1 GiB of memory of QEMU's process (on
  Linux a shared mapping of a memfd of the library's, on macOS private
  anonymous memory), as one RAM region over the mapping: the control rings,
  below `RVG_GRAPHICS_ALLOCATIONS_OFFSET` (0x400000), and the guest's
  allocations from there. It has no dirty log: the library's window draws
  the guest's scanout again at each of the display's refreshes while it
  shows no presented surface.
- **The guest's RAM**: on Linux the device gives the library the map of the
  RAM ranges that are a shared file in pages of the host's size (the memfd of
  `reims-ctl`'s VMs), with the files behind them, and the GPU uses an
  IOSurface's own pages in place. A range with no such file (anonymous or huge
  pages) is left out of the map, and a surface whose pages fall there is copied
  as every surface is without a map; since IOSurfaces live only in the main
  RAM, which the memfd backs, every surface maps in place. On macOS the device
  gives every RAM range in pages of the host's size, with no file (an fd of
  -1): the library views the process's own mapping of it (`mach_vm_remap`),
  which shares the pages HVF maps into the guest. There the library uses a
  surface in place only for a guest that lays its pages out in whole 16 KiB
  pages (`RGFX_FEATURE_PAGES_16K`, which the arm64e kext takes); an x86-64
  guest's surfaces are copied. QEMU gives the
  library no dirty log of the guest's RAM: with one (`2243023e`, which
  `926401c4` reverts), resolving the pages behind each IOSurface's import cost
  its thread about 5 ms a command buffer on a discrete GPU, more than the whole
  copies it saved.
- An interrupt the library raises, on a thread of its own, is written to
  the vector's irqfd where KVM's in-kernel irqchip injects MSI-X through one
  and MSI-X is enabled, and KVM injects it on the raising thread. On WHPX
  with the hypervisor's APIC (kernel-irqchip) the raising thread requests the
  enabled, unmasked vector's MSI from the hypervisor itself
  (`whpx_send_msi_direct`, WHvRequestInterrupt). Elsewhere (HVF, TCG, a
  masked vector, WHPX without kernel-irqchip) a BH sends it under the BQL.
  The BH waits for the main loop, which a PNG screendump of 2880x1620 holds
  for about 120 ms (a busy desktop's: 1.4 s) on a Mac mini 2018.
- On WHPX the vCPU loop took the BQL before each run of a vCPU
  (`whpx_vcpu_pre_run`), so a vCPU that exited to the device's lockless
  BARs, to ring a doorbell, waited for the BQL's holder too. With
  kernel-irqchip and no interrupt request pending it now runs on without it,
  as KVM's does. Both changes are needed for a screendump to cost the guest
  no frame. With neither, an on-screen app's frames had a 7-refresh gap at
  each PNG screendump on the mini (gap max 128-148 ms). With the interrupts
  alone the gap stayed at 116.6-133.7 ms, with the vCPUs waiting 111-123 ms
  for the BQL. With both, gap max was 33.7-34.0 ms (measured 11-16). Under KVM, the
  same screendump costs no frame (ms01's Iris Xe, 33.3-33.7 ms).
- **Displays:** the `displays` property gives the device 1 to 4 displays
  (`RvgGraphicsDeviceConfig`'s `display_count`), 1 by default; realize
  refuses 0 and anything above 4. Display i is head i of QEMU's UI, with a
  console, a window and a cursor of its own (below), each the same as display
  0's. A guest whose driver takes `MULTI_DISPLAY`, which the device offers
  only with two displays or more, drives them all. Other guests, and the
  option ROM, drive display 0 alone, and the other displays show black
  (docs/graphics-kext/design.md, "Multiple displays"). A window's pointer
  goes to its own display's console in that display's pixels, and every
  window's keys reach the one keyboard. A VM has one `usb-tablet`, so a
  position over display 1's window reaches the guest as the same position
  over display 0's would; how the guest should place it (the design's X1)
  is still open. With `displays=1` QEMU makes the same console, window,
  reads and callbacks as before.
- Under `-display none` the library opens a window for each display, unless
  `REIMS_VGPU_WINDOW` is `0`. Each opens at the `resolution` property's size,
  `<width>x<height>` (1920x1080 without it), a guest pixel per host point,
  shrunk to fit the screen; then the window keeps the size the person gives
  it, and the display offers the guest that size in pixels at the host
  screen's rate. On a KDE Wayland desktop of 3840x2160 at 2.1x, 1829x1029
  points, `resolution=1280x720` opened the window at 1280x720 points, and a
  macOS 26.7.2 guest ran 2688x1512; `2560x1440` and the default opened it
  shrunk to 1744x981 points, the guest at 3662x2039 (**measured**). Its
  input is queued for a BH and reaches the guest's tablet and keyboard
  through its display's console. Closing any window powers the VM off.
- **The console** of each display shows what the display shows, with a
  window or without one: at each of its updates (a display change listener's
  refresh, or a screendump of its head) a thread of the device's pool reads
  it back from the library (`rvg_graphics_device_read_display_at`), the
  scanout or the IOSurface the guest presents, through the display's colour,
  at the display's mode, and a BH copies it into the console's surface and
  says the update is done. So QMP's `screendump` of head i saves display i's
  frame (head 0 by default). One read of a display runs at a time; an update
  that comes while one is under way is done with it. The read waits
  for its draw on the GPU, never under the BQL, and the guest sees nothing
  of it. A `screendump` answers once the read is done, and QMP runs its
  commands one at a time, on every monitor, so none answers meanwhile; HMP's
  do. A read still under way 3 s after its start is late: a `screendump`
  waiting for it is answered then with what the console shows, the frame of
  the last read that ended, or QEMU's placeholder ("Guest has not initialized
  the display (yet).") where none has, and so is every update until the read
  ends, which starts no other read. The read's end updates the console. The
  3 s leave `reims-ctl view`, which waits 5 s for a QMP answer, 2 s of
  margin. Before that bound, a read waiting for a GPU that stopped answering
  (NVIDIA's Xid 109) held every QMP monitor to the quit, while HMP's `info
  status` answered in under 1 ms throughout a 150 s wait, the vCPUs ran and
  the guest answered ssh (**measured**, RTX 2080 and RTX 3090, 2026-11-12).
- **The cursor** is each console's too: the library tells the device each
  display's guest cursor on its display thread (`cursor_glyph`,
  `cursor_show` and `cursor_move` of `RvgGraphicsCallbacks` for display 0,
  and for displays 1 and up `cursor_glyph_at`, `cursor_show_at` and
  `cursor_move_at`, which name the display), and the BH, in order with the
  windows' input, gives the display's console its image and hot spot
  (`qemu_console_set_cursor`) and its place and whether it shows
  (`qemu_console_set_mouse`). The library gives the hot spot's place
  (`rvg_graphics_device_cursor_position_at`), and the console takes the image's
  top-left corner, as virtio-gpu gives it, so the device takes the hot spot
  off. QEMU's own displays (SPICE, VNC, GTK, D-Bus) draw the cursor from it.
  The image the console reads back has no
  cursor, and under `-display none` the console has no display: the window
  shows the cursor as the host's pointer.
- It has no default option ROM: `romfile=` names one.
- **QEMU's quit:** once `vm_shutdown` stopped the VM for good, a VM-state
  handler tears the device down, BAR 2 out of the guest's address space
  first, so that the library's release of its last device lets go of the
  GPU device before the process exits (docs/api.md, "The process's exit").
  A shutdown whose action pauses the VM (`-action shutdown=pause`,
  `-no-shutdown`), and a quit of a VM stopped before it, keep the device to
  the exit, as before. So does a quit while a read of a display (a
  console's, above) still waits for its draw on the GPU: the teardown waits
  for every display's read at most 5 s, under the BQL, as the reads use the
  device. A
  screendump still waiting for the read is answered first, with what the
  console last showed: QEMU's quit waits for QMP's dispatcher, which the
  screendump holds (`monitor_cleanup`), and a read that ends after the
  console closed updates nothing. A GPU that stopped answering otherwise held
  the quit for good: after NVIDIA's Xid 109 (a context switch timeout) a read
  never ended, and QMP's and HMP's `quit` left QEMU in `thread_pool_free`
  until SIGKILL (**measured**, RTX 2080 and RTX 3090, macOS 26.7.2,
  2026-11-12).
  The release itself runs with the BQL let go, and nothing else of the
  teardown does: `route_doorbell`, which the library calls on its threads as
  a command queue is made or ends, takes the BQL, and one waiting for it as
  the release began held the release up to its 5 s bound under a BQL the
  teardown held. Now it takes the BQL, finds the routes the teardown closed
  and returns. No other callback takes the BQL, and no BH of the device's
  runs inside the release, which runs on the main loop's thread. A window
  or cursor callback can still run after the release (one already running
  on the window's thread as the release closes the window, or one of a
  device thread its 5 s left running), so the BH those callbacks schedule
  stays to the exit and drops their events once the console is closed;
  until `baa655fe` the teardown deleted it, and such a callback ended QEMU
  with SIGSEGV at its quit (read from the code; no run reached it).
- **The machine:** macOS places every 64-bit prefetchable BAR below 4 GiB,
  where a 1 GiB BAR needs the window to start at or below 0x4000_0000. So a
  q35 machine with the device keeps at most 1 GiB of the guest's RAM below
  4 GiB (`-machine q35,...,max-ram-below-4g=1G`): with q35's default of
  2 GiB there, BAR 2 gets no address and the kext finds no region
  (docs/graphics-kext/design.md, the region; `REGION_SIZE` in
  `crates/reims-vgpu-graphics/src/device.rs`). `reims-ctl` gives its x86-64 VMs
  the device by default and adds the option with it
  (`crates/reims-ctl/src/qemu.rs`). Another host that adds the device adds
  the option too. Where it does not, the device's realize warns, naming the
  option: on a PC machine (one with the `max-ram-below-4g` property) with
  RAM at 1 GiB.

`CONFIG_REIMS_GRAPHICS` is set for every target when the library is found,
whatever its accelerator, so `make qemu` on a Mac builds the device into
`qemu-system-x86_64` too, where it runs under TCG.

**Measured** on the M3 Max, under TCG on q35, paused (`-S`), when the region
was 256 MiB:
- headless (`REIMS_VGPU_WINDOW=0`), QMP's `query-pci` shows 1234:5247,
  class 0x0380, BAR 0 4 KiB of 32-bit memory and BAR 2 256 MiB of 64-bit
  prefetchable memory; `system_reset` and `quit` exit 0;
- under the qtest accelerator, with the BARs programmed by hand, MAGIC reads
  0x58464752, VERSION 1 and the region size 0x10000000; a GET_DISPLAYS
  written into the submission ring and rung through DOORBELL's ioeventfd is
  answered with DISPLAYS, one display of one mode, 1920 x 1080 at 60000
  mHz; STATUS 0 empties the completion ring;
- under `-display none`, the window opens and the library imports the
  region for it with no refusal.

**Measured** on an RTX 3090 (NVIDIA 620.70.06) under KVM, macOS 26.7.2 with
the window open: the guest's `shutdown -h now` ended QEMU 11 s later with
exit code 0, the library's let-go of its one GPU device logged
(`reims_vgpu_vulkan::let_go`, at debug), no `reims-vgpu: fatal` line and no
core. Realized paused (`-S`) and quit through QMP, QEMU exited 0 with the
device kept to the exit.

### Measuring WindowServer's composites

dtrace in an x86-64 guest adds to what it measures, so WindowServer's
composite time (`CompositorMetal::CompositeLayersToDestination`, CLTD) is
read with that function's own entry and return probes and no others, and
QEMU's CPU time is read with no dtrace in the guest. A script that also
probes what CLTD calls splits its time up, but its total is no longer
CLTD's. On-CPU samples of a guest thread key their state on `tid`: a
`profile` probe does not see a thread's `self->` variables.

**Measured** on the RTX 5080 Laptop GPU host with a macOS 26.6.1 x86-64
guest, under KVM:
- each pid or objc probe that fires costs the guest about 10 µs. A script
  of about 50 of them a composite adds 0.4 to 0.5 ms to CLTD, with Apple's
  own driver as much as with ours;
- `profile-997` on the guest's eight vCPUs adds about 30% of a host core to
  QEMU's CPU time.

On the host, QEMU is never traced with a per-task perf recorder: `perf record
-p` or `-t`, or tracepoint or uprobe events its new threads inherit. Each
thread carries the events, so **measured** on the same host, a thread's exit
took 0.12 to 12 s and at least 0.13 s of CPU in `perf_event_exit_task`, and
as the host's bursts of graphics threads exited, those exits filled 22 to 23
of its 24 CPUs, starved the vCPUs and froze the guest for 1.3 to 10 s: frames
of up to 9.5 s. A host trace records system-wide (`perf record -a`), filtered
by pid, and measures only after perf's first sample, since attaching its
uprobes made present gaps of 378 and 521 ms.

### fbt probes in the guest's kexts

`dtrace -l -P fbt -m ai.aneesiqbal.ReimsGraphics` lists no probes. The
guest's kernel decides that, not the kext's build. **Read** from xnu's
public source (its main branch, `bsd/dev/dtrace` and
`libkern/c++/OSKext.cpp`, not checked against macOS 26's):
- While SIP restricts dtrace (`CSR_ALLOW_UNRESTRICTED_DTRACE` not allowed),
  the kernel makes no fbt probe in any module
  (`dtrace_fbt_probes_restricted`). The lane's guests run with SIP on, so
  fbt needs an image whose `csr-active-config` allows dtrace.
- OpenCore injects our kext into the Boot kernel collection
  ([docs/graphics-kext/design.md](graphics-kext/design.md), item 13). For a
  Boot-collection kext, the kernel defers fbt's probes to the dtrace command,
  which sends the symbols it finds for the kext's UUID
  (`dtrace_module_loaded`). It makes them from the kext's own symbol table
  at its load only where the kext's `Info.plist` sets
  `OSBundleForceDTraceInit`, which `OSKext::registerWithDTrace` ignores for a
  Boot-collection kext unless the guest boots with `keepsyms=1`. From that
  table fbt takes only global functions.

**Inferred**, not measured: the dtrace command finds Apple's Boot-collection
kexts in the collection on the guest's disk, but OpenCore injects ours only
into the copy in memory, and no file on the guest's disk has its UUID.

**Measured** on the kext as `make -C guest/kext` builds it: its symbol table
keeps every function (398 symbols in `__text`, 324 of them global), it has an
`LC_UUID`, and 384 of the 392 functions `otool -tv` lists (314 of the 319
global ones) start with `push %rbp; mov %rsp,%rbp`, the prologue fbt patches
on x86-64. `keepsyms=1` alone, which the lane's trace images of 11-01 had in
their boot-args beside `csr-active-config` 0x7f, does not change where the
probes' symbols come from (above).

So an image for tracing our kext sets three things: a `csr-active-config`
that allows dtrace, `keepsyms=1`, and `OSBundleForceDTraceInit` (true) in the
`Info.plist` of the ReimsGraphics.kext that OpenCore injects. `make -C
guest/kext DTRACE=1 BUILD=build/dtrace` builds that kext; a plain `make`
leaves the key out. Without `keepsyms=1` the kernel ignores the key for a
Boot-collection kext, logging `Ignoring OSBundleForceDTraceInit for Boot KC
Kext ai.aneesiqbal.ReimsGraphics` (**read** from xnu-12377.1.9 and
xnu-12377.121.6, macOS 26's sources; macOS 11's, xnu-7195, panics there
instead). With SIP on, dtrace makes no fbt probe whatever the kext sets
(`dtrace_fbt_probes_restricted`). Whether the three give our global
functions their probes is not yet measured. To check, as root in the guest
on such an image:
- `dtrace -l -P fbt -m ai.aneesiqbal.ReimsGraphics | wc -l` prints more
  than 0: about 314 entry probes, and their return probes;
- `log show --last boot --predicate 'eventMessage CONTAINS
  "OSBundleForceDTraceInit"'` prints no line.

## Linux hosts

`reims-ctl` runs its VMs on Linux x86-64 hosts, under KVM, with the library
on its Vulkan platform. These are the facts of such hosts the library and
QEMU rely on.

### Linux hosts: huge pages for guest RAM

QEMU maps file-backed guest RAM, the memfd of `reims-ctl`'s VMs among it, at
a 2 MiB-aligned address (`system/physmem.c`), as it maps anonymous RAM.
The kernel backs a shmem mapping with a transparent huge page only where the
mapping's address keeps the file's offset in one, so at the 4 KiB alignment
before, the guest ran on 4 KiB pages alone. Whether huge pages back it now
is the host's choice: `/sys/kernel/mm/transparent_hugepage/shmem_enabled`
at `advise` (CachyOS ships it), with QEMU's `MADV_HUGEPAGE`, or at `always`
or `within_size`; at `never` or `deny`, upstream's default, nothing changes.

Measured on msi-5080 (macos-26, 16 GiB, a drag and a UFO test,
2026-10-27): KVM maps about 10 GiB of the guest at 2 MiB, none
before, and the vCPUs' time a frame falls 6-7% at the same frame rate.

### Linux hosts: guest RAM resident from the start

`reims-ctl` has QEMU allocate the guest's RAM whole as it starts
(`prealloc=on` on the memfd backend), with as many threads as the guest has
vCPUs (`prealloc-threads`). Without it, the guest's first touch of a page
faults on a vCPU, and KVM resolves the fault through `get_user_pages`, which
holds the process's mmap lock for read. Where the host backs the memfd with
transparent huge pages (above), that fault allocates a 2 MiB page, and with
`defrag` at `madvise` (the kernel's default) and QEMU's `MADV_HUGEPAGE` it
reclaims and compacts memory for it inside the fault. On a host whose free
memory is fragmented, compaction mostly fails and can take over a second. A
thread that asks for the lock for write meanwhile (glibc growing a heap by
`mprotect`, a view's `mmap`) waits behind the fault, and every later reader
waits behind it, the other vCPUs' faults among them.

Measured on msi-5080 (CachyOS, Linux 7.2.8; transparent huge pages
`enabled` always, `defrag` madvise, `shmem_enabled` advise;
`compaction_proactiveness` 0; macos-26, 16 GiB, 8 vCPUs; Safari playing the
4K HEVC clips one, two and three at once; 2026-11-08), two boots each way:
- without prealloc, vCPUs stood in uninterruptible sleep, waiting for the
  lock, for 15 and 22 s of the boot in all, up to 1.0 s at a time; vCPUs'
  faults held it for up to 1.8 s, and 94 and 95% of the boot's direct
  compactions failed; the clips presented 1.1 to 57 frames a second and
  dropped up to 1,570 frames each;
- with prealloc, no vCPU took the mmap lock, the longest hold of it was
  1.9 ms, and the clips presented 50 to 58 frames a second, dropping 0 to 4;
- QEMU took 5.3 and 6.5 s from its start to its first vCPU thread, against
  0.05 s, and the process held 17.1 to 17.2 GiB at the guest's first Safari
  page, against 10.9 GiB (14.5 to 15.0 GiB at the end of the playback
  without prealloc).

Through `reims-ctl`, installed by `scripts/install-ctl.sh` (macos-26.7.2,
8 GiB, 8 vCPUs; 2026-11-08): on um870 (Radeon 780M) QEMU reached its first
vCPU thread 0.53 s after its start and `reims-ctl start` returned in 0.8 s;
on ms01 (Iris Xe) `start` returned in 8.9 s. Both reached the desktop with
all of the guest's RAM in 2 MiB pages.

A host without the free memory for a VM's RAM now refuses the VM as QEMU
starts, where before it overcommitted: the kernel kills QEMU during the
allocation, and `reims-ctl start` says what the VM needs and what the host
has available (seen with an 8 GiB VM in a 3 GiB memory cgroup on um870).

### Linux hosts: `vm.max_map_count`

On Linux the library's views of guest memory take kernel mappings, which
count against `vm.max_map_count` (the kernel's default is 65,530; some
distributions raise it). An x86-64 macOS guest hands out its pages
in 16 KiB clumps laid backwards, and a mapping of a file runs only forwards,
so a view mapped from the guest's files takes a mapping per 4 KiB page.
Measured on an RTX 5080 Laptop GPU host (Linux 7.2.8, `vm.max_map_count`
1,048,576) with a 16 GiB macOS 26.6.1 guest at 3840x2160 HiDPI on 2026-10-27:
- 99.97% of the views' mappings were single 4 KiB pages;
- within 190 s of a window drag, a resize and Safari scrolling, the views held
  1.03 million mappings over about 4 GiB of guest pages, each page in about
  one view. That passed the budget, and the process died.

So the library maps a view of scattered pages as dma-bufs of those pages,
made with `/dev/udmabuf`: one mapping per 4 MiB at udmabuf's stock
`list_limit` of 1,024 (`crates/reims-vgpu-vulkan/src/os/linux/vm.rs`, "Views
through dma-bufs"). How the GPU then takes those pages depends on its driver:
- **RADV** imports such a view through the dma-bufs, so the views stay small.
- **NVIDIA's driver** (615) pins a pointer's pages, and refused both a pointer
  into a udmabuf mapping and the dma-buf itself. A probe finds this as the
  device is made: its import fails, and the copy the runtime makes in its
  place counts as the refusal. From then on:
  - The device is handed the guest's RAM whole (metal2vulkan's `HostRam`,
    `crates/reims-vgpu-vulkan/src/os/host_ram.rs`). An object whose
    pages are one run of the RAM inside one 64 MiB chunk of it is bound in
    the device's import of that chunk, made the first time an object over it
    is made: it takes no mapping and asks no gate. A device without
    `VK_EXT_external_memory_host` is not handed the RAM, said once in the
    log, and its objects take the ways below.
  - Where the probe also found the device takes the pages one dma-buf holds
    in the pointer's place (`import.rs`, `ONE_DMA_BUF`; NVIDIA's driver and
    ANV) and makes textures over them in place (`ONE_DMA_BUF_VIEWS`; ANV at
    metal2vulkan fb33065288, not NVIDIA's driver), a view one dma-buf holds (at
    udmabuf's defaults 1,024 runs and 64 MiB) is mapped through that dma-buf
    from the first, one mapping, and a no-copy buffer over the whole view is
    imported through it, given a duplicate of its descriptor; a heap's
    import, and CUDA's registration of a video surface, map its pages from
    their files (`vm.rs`, "One dma-buf a view"). The device learns this once,
    as it is made: in a host's first process after a driver update or a
    change of the runtime's in-place check, before the runtime keeps the
    check's verdicts, its views are mapped as below for that whole process,
    and the route is taken from the next process on.
  - Other views are mapped from their files while they fit 8,192 mappings
    short of the budget (`budget.rs`, `HEADROOM`), and through dma-bufs past
    that. Any other import of a view's pages (a no-copy buffer's or a heap's)
    maps them from their files while they fit short of those 8,192 mappings
    too.
    Past that line the library's import gate refuses the pages, and
    the runtime makes the object over a copy of them, which takes no mapping
    (metal2vulkan's `HostMemory::with_import_gate`, announced
    `copied-host-pages`; with `REIMS_VGPU_STRICT=1` it ends the process),
    but pages one dma-buf holds (at udmabuf's defaults 1,024 runs and
    64 MiB), where the probe found the device takes such a dma-buf in the
    pointer's place: NVIDIA's driver imports it dedicated to the buffer, so
    the gate lets them through and they take no mapping (`import.rs`,
    `ONE_DMA_BUF`). The copy lies in host-cached system memory, compared
    whole on the CPU at each commit and each completion, and it reads the
    pages with the CPU, which reads pages left in views through dma-bufs
    too.
- **Before anything would die or be refused at the whole budget**, the
  library folds back into dma-bufs the views' pages mapped from their files
  that no live import a device may pin holds, and weighs again (`vm.rs`,
  "Folding back"). A live import holds its pages only while it is being
  made where the GPU keeps the pages it pinned as it imported them, which
  another probe finds as the device is made (`import.rs`, `probe_fold`),
  once for each GPU, driver and kernel, its verdict kept in the library's
  cache (`device-checks/fold-after-import-UUID`). It maps other pages over
  an import's view and checks that the GPU still reads the imported ones,
  then folds the view, writes the pages through their own mapping, and
  checks that the GPU reads them and that its writes show through both. A
  driver that reads the new pages fails it before the fold: ANV on i915
  does, and lost its device when the fold came first (docs/lessons.md).
  NVIDIA's driver passes, so its made imports' views are folded too. Only
  where that frees too little is a view or an import refused. A buffer's or
  heap's import short of the headroom folds the same way before it is
  copied, where the views grew or an import was freed since the last such
  fold. A map the kernel itself refuses at `vm.max_map_count`,
  when the rest of the process holds more than the 16,384 kept for it, folds
  the same way and is tried once more.
- **What is refused for want of mappings** answers as the same call does on
  Metal when memory runs out: a remap of guest pages, or a range
  a view is made of, answers `KERN_NO_SPACE` with nothing mapped, and the
  library's own memory none. Each is counted and
  logged at the first and at each power of two, those after the first at
  debug (docs/conventions.md). With `REIMS_VGPU_STRICT=1` each ends the
  process at the call, naming the numbers.
- **Where `/dev/udmabuf` is not the process's** (the device node is the `kvm`
  group's on Ubuntu), every view is mapped from its files, so views past the
  budget are refused as above. At the kernel's default that is about 192 MiB
  of an x86-64 guest's views, so such a host needs udmabuf, not a higher
  limit.

The budget is `vm.max_map_count` less 16,384 mappings kept for the rest of
the process (QEMU, the C library, the driver, the runtime). On the same host,
with a 12 GiB guest on 2026-10-10, the process held 1.7k to 4.6k mappings
besides its views. The library's own count is exact at every map and unmap,
and views and imports that choose between files and dma-bufs are weighed
together as they are made, so threads making them at once cannot pass the
line between them.

What gets logged:
- Where udmabuf is not the process's, a warning when the guest's RAM is
  leased, if the RAM at a mapping per 4 KiB page would not fit.
- Each answer given for want of mappings, counted, at the first and at each
  power of two (after the first at debug), and in the degradation counts as
  `out-of-host-memory`.
- On a device that needs pages mapped from their files, an information line
  if imports of all of the guest's RAM, a mapping per 4 KiB page, would come
  within the headroom of the budget, with the `sysctl -w
  vm.max_map_count=...` value that covers them (2^23, 8,388,608, for a 16 GiB
  guest): past that, buffers and heaps not bound in a chunk are copied.
- Each buffer or heap copied for want of mappings is counted with
  its bytes, logged at the first and at each power of two (after the first
  at debug), and written to
  the trace (`nrcopy`: first byte, length, mappings it would have taken).
- On a device that needs pages mapped from their files but takes no host
  memory by its pointer, an information line, once, that the guest's RAM is
  not used whole.

### Linux hosts: AMD GPUs and udmabuf

RADV refuses to import pages with a file behind them as host memory, and the
guest's RAM and the library's views are all memfd pages. Where the pointer
import is refused, the runtime imports dma-bufs of the same pages, which the
library makes with `/dev/udmabuf`, cut into chunks within udmabuf's limits as
the host has them (`crates/reims-vgpu-vulkan/src/os/linux/dma_buf.rs`). The
device node is the `kvm` group's on Ubuntu. A guest whose QEMU cannot open it
dies at boot on an AMD GPU, at its first no-copy buffer.

What the chunks must hold, **measured** with a 14 GiB macOS 26.6.1 guest at
one 1920x1080 display on an RTX 3090 host, where every import is host
memory: at the idle desktop, 829 imports, the largest 63.3 MiB, 19 of more
than 1024 runs of file pages and the most 16,201 (4 KiB pieces); with Safari
open on wikipedia.org, 4,551 imports and 29 of more than 1024 runs. At
udmabuf's defaults (1024 runs, 64 MiB) the 16,201-run import is 16 chunks. On
a Radeon 780M at those defaults the same guest cold-boots to its desktop, and
Safari loads wikipedia.org and quits, with none of 5,119 dma-buf requests
refused: 5,095 took one dma-buf, and the rest 2 to 16.

### Linux hosts: Intel GPUs without a shared last-level cache

Arrow Lake-S's iGPU (`deviceID` 0x7d67, Xe-LPG; ANV of Mesa 26.2.4 on i915,
Linux 7.2) shares no last-level cache with the host, and ANV's one
host-cached memory type there (`DEVICE_LOCAL | HOST_VISIBLE | HOST_CACHED`)
is not coherent: the GPU reads stale 64-byte lines of what the host wrote
into it. A texture placed there for its host copy shows sparse, fully
transparent 4x4-pixel blocks (one line of a Tile4 32-bit image) in an x86-64
macOS 26 guest's boot logo, Dock icons and widgets (**measured**).
metal2vulkan keeps a host-copied texture out of memory the host maps without
coherence (since `edb5ffd8cc`), and the guest draws them whole.

### Linux hosts: Intel GPUs on i915: a reset of QEMU's context

i915 resets a context on the render engine (`rcs0`) that the GPU does not
preempt within the engine's `preempt_timeout_ms`, 7500 ms on Gen12 (Linux's
`CONFIG_DRM_I915_PREEMPT_TIMEOUT_COMPUTE`: Gen12 has no mid-thread
preemption), and logs `qemu-system-x86[PID] context reset due to GPU hang`.
With GuC submission, i915's default on Alder Lake-P, Raptor Lake-S and later
(`intel_uc.c`), GuC applies that timeout, which a context takes into its GuC
policy when it registers (`register_context`). The heartbeat
(`heartbeat_interval_ms`, 2500) raises its pulse's priority each interval
and waits twice the preempt timeout at the highest before i915 resets the
engine itself. The GPU preempts between commands and thread groups: i915
sets Gen12's compute preemption to thread groups
(`WaDisableGPGPUMidThreadPreemption:gen12`), and ANV (Mesa 26.0.8 and
26.2.6, `init_render_queue_state`) keeps compute there, sets 3D preemption
to mid-command-buffer, and on Gen12.0 turns preemption off during a draw. So
one draw or one thread group that runs past the timeout is reset, where a
command buffer of many runs on (**read from code**, Linux 7.0).

**Measured** on `homelab-ms01-1` (Raptor Lake-P's Iris Xe, 96 EUs, GuC
70.49.4, Linux 7.3.1, Mesa 26.2.6, iHD 26.3.5; KDE Plasma on the same GPU),
2026-11-16, x86-64 macOS 26.7.2 booted cold at master 818868564 with 8 vCPUs
and 8 GiB, through Safari's scroll, WebGL and video pages, H.264 and HEVC in
QuickTime Player, and three idle minutes (jobs m3-max-35697, m3-max-35741):
- with i915's defaults, nothing was reset. The longest runtime command buffer
  took 128 ms of GPU time (a render pass of Safari's scroll), and the two of
  452 and 794 dispatches media analysis sends at boot
  ([video/bugs.md](video/bugs.md), B26) 102 and 98 ms;
- with `rcs0`'s `preempt_timeout_ms` at 50 while QEMU made its contexts (set
  before QEMU started, restored at the QuickTime step), through the boot and
  Safari's pages, nothing was reset either;
- a reset of the whole GT through debugfs (`i915_wedged`, `rcs0`'s mask:
  with GuC, i915 resets the whole GT) while QuickTime played logged QEMU's
  context reset as above. The runtime made its `VkDevice` again in 209 ms
  (one command buffer failed as `GpuHang`), and the guest ran on with no
  fatal: QuickTime's decode session landed all its 1171 pictures, the next one
  1162. The desktop picture showed flat green from then to the run's end; the
  recovery's notice counted 449 textures whose contents were lost, 421 of them
  `Private`. KWin kept compositing.

### Linux hosts: NVIDIA's pipeline import budget

**Measured** on an RTX 3090 and an RTX 5080 Laptop GPU, NVIDIA driver
615.71.09: once a process has loaded about 29 MB of pipeline code from a
cache, `vkCreateGraphicsPipelines` answers `VK_PIPELINE_COMPILE_REQUIRED` for
the pipelines after it, so each of those compiles, 14 to 24 s for each of the
big glass pipelines (Xgc), on every boot: on the 5080 about 10 of them fit in
a process. The budget is the same whether the code comes from
`VkPipelineCache` data or from pipeline binaries (`VK_KHR_pipeline_binary`),
with the driver's own cache on or off, in one cache or twelve, and destroying
a cache gives none of it back. Only fewer bytes help: smaller shaders, fewer
pipelines per state.

### Linux hosts: the guest's disk

`reims-ctl` gives a VM its system disk as an image file in the VM's
directory (qcow2 for a new disk), on QEMU's defaults (`cache=writeback`,
`aio=threads`), on AHCI unless the VM's settings choose NVMe, which adds
`discard=unmap`. On AHCI
the guest's flush (FLUSH CACHE EXT) is not queued: an x86-64 macOS 26.7.2
guest started no other disk command while one ran, over 1,069 to 1,368
flushes a boot. On NVMe it went on reading and writing during its flushes
(4,870 to 17,485 commands a boot).

**Measured** on two CachyOS hosts (Linux 7.3.3; `/home` on btrfs,
`compress=zstd:1,ssd,discard=async`), each cold-booting macOS 26.7.2 on the
Radeon 780M from a reflink clone of the same disk, through a Safari page
load, a window resize and a scroll (2026-11-11 and 12; on um790 jobs
m3-max-19441, 19474, 19602, 19604 and 19714, on um870 19679 and 19697). A
wait below is a span of 300 ms or more in which none of the guest's
outstanding disk commands completed, given per boot as a count and the
total time:
- On `homelab-um790-xtx-1`'s Lexar NM620 (DRAM-less), a single read, of the
  image's data or of btrfs's metadata, took 0.3 to 2.0 s now and then, and
  once every request on the device took 1.66 s at the same time. A guest
  command waits behind such a read, and a guest flush also waits behind
  btrfs's transaction commit, which comes every 30 s and took up to 2.5 s
  there. On QEMU's defaults six boots had 0, 0, 1 (0.44 s), 2 (0.94 s),
  2 (2.76 s) and 6 waits (3.91 s).
- On `homelab-um870-1`'s Samsung 970 EVO Plus, the SSD took at most 9 ms
  over a read, and no guest write took over 69 ms nor a flush over 177 ms.
  Six boots on AHCI, two of them on qcow2 overlays, had no wait but one
  0.33 s read 2 s into a boot, in the firmware; two on NVMe had none.

None of the other choices did better on the Lexar than the defaults in the
same jobs:
- `aio=io_uring`: no wait, and one of 1.01 s, within the defaults' spread;
  that one was a flush writing back about 1 GB the guest had just written
  into the host's page cache.
- A nodatacow image (made in a `chattr +C` directory): 3 waits (2.66 s) and
  6 (6.16 s). Before writing in place, btrfs checks that no other file
  shares the extent, reading the extent tree, and the flushes' fsyncs
  waited on those reads, up to 1.95 s.
- `cache=none`: 6 waits (5.92 s) and 6 (3.64 s). Btrfs writes a direct
  write to a checksummed file through the page cache, under the inode's
  lock, which held every other read and write of the image for up to
  729 ms; and the guest's reads
  no longer hit the host's page cache, so more of them met the SSD's slow
  reads: 29 and 16 of the guest's reads took over 500 ms in a boot, against
  0 or 1 on the defaults.
- NVMe: 4 waits (2.51 s) and 7 (7.72 s). The guest's TRIMs reach btrfs as
  punched holes, one of which waited 831 ms for the image's inode lock.

What NVMe does gain is the boot: the guest answered SSH 9 to 10 s after
QEMU started in three NVMe boots of four, on both hosts, against 17 to
19 s in every AHCI boot; the fourth, on the Lexar, took 22 s.

### Linux hosts: KVM halt polling

QEMU sets no halt polling for a VM (`KVM_CAP_HALT_POLL`), and `reims-ctl`
gives no `-overcommit cpu-pm=on`, so a halted vCPU is the host's to wake.
KVM polls it for up to the module parameter `kvm.halt_poll_ns`, which it
reads at each halt of a VM that set no limit of its own (200 µs on x86
unless the host sets another), grows each vCPU's window from 10 µs up to
that limit and shrinks it again to fit how long its halts last, and stops
polling as soon as another task wants the CPU. A vCPU halted longer than its window sleeps, and its
host CPU may go to its deepest idle state. The measurements below are why
both stay so.

**Where a cold wake's time goes** (measured 2026-11-14; x86-64 macOS 26.7.2
on reims-graphics-pci, 8 vCPUs, `Skylake-Client-noTSX-IBRS`; exclusive jobs
on reflink clones with `restrict=on`, `m3-max-29907` and `m3-max-29908`). A
probe in the guest sleeps 2 ms, then times a syscall, a CPUID or its wake
of a thread blocked as long, and KVM's and the scheduler's tracepoints on
the host time the same halts. Each figure is a p50.
- On `homelab-um870-1` (Ryzen 7 8745H, Linux 7.3.3; `acpi_idle`, whose C3
  is entered by I/O port 0x415 with a declared exit latency of 350 µs;
  kvm_amd's `avic` off), a vCPU halted for 2.8 ms reached VM entry 48 to
  52 µs after the host's `sched_waking`, from C3 in 85 and 93% of the
  wakes. A wake by IPI, half to three quarters of them, took 28 µs for the
  target CPU to leave its idle state and 18 µs more to switch the vCPU
  thread in; a vCPU thread queued on the waking CPU itself reached VM entry
  in 39 to 43 µs; and VM entry came 16 to 17 µs after the switch-in. In the
  guest the thread's wake took 25 to 29 µs, and the first CPUID 4.2 to
  4.6 µs, against 1.9 µs after 200 µs asleep and 0.24 µs on the host itself.
- On `homelab-ms01-1` (Core i9-13900H, Linux 7.3.1; `intel_idle`, with C2
  at MWAIT 0x21 and C3 at 0x60; kvm_intel's APICv on, IPI virtualization
  off), the same halts reached VM entry 20 to 23 µs after `sched_waking`,
  30 and 48% of them from C3. In the guest the wake took 5.1 to 5.9 µs and
  the first CPUID 1.0 to 1.25 µs, as after 200 µs asleep.
- The guest's first syscall after 2 ms asleep costs no more than the host's
  own: 3.0 to 4.1 µs on um870, where Linux's takes 4.2 to 4.4 µs (1.0 µs
  back to back), and 0.6 to 0.7 µs on ms01, where Linux's takes 1.7 µs
  (0.25 µs). macOS on a MacBook Pro's Core i7-9750H takes 3.3 µs (1.2 µs;
  job `m3-max-29818`).

**A longer poll.** The jobs below set `kvm.halt_poll_ns` live inside the
job and put it back at its end. In one boot per host, at 200 µs, 1 ms, 4 ms
and 200 µs again: 30 s of the idle desktop, 40 s of Safari playing H.264
1080p60, then the probe, and the `single.sleep` and `tex1.sleep` cases of
kcall_split, which times the guest's commit of a Metal command buffer after
its thread slept (medians of 500). Then Geekbench 7.1.0's Metal timed pass,
one cold boot per value, in the order 200 µs, 4 ms, 1 ms, 200 µs. Jobs
`m3-max-30079` on um870's Radeon 780M and `m3-max-29946` on ms01's Iris
Xe, 2026-11-14. Package W is RAPL's package energy over the window or the
pass; poll CPUs is KVM's own halt-poll time.

| host | `halt_poll_ns` | GB total s | vCPU CPUs | poll CPUs | package W |
|---|---|---|---|---|---|
| um870 | 200 µs | 173.1 | 0.53 | 0.026 | 16.8 |
| um870 | 4 ms | 170.1 | 1.02 | 0.562 | 21.1 |
| um870 | 1 ms | 172.5 | 0.66 | 0.179 | 18.4 |
| um870 | 200 µs | 174.1 | 0.52 | 0.025 | 17.1 |
| ms01 | 200 µs | 286.6 | 0.48 | 0.025 | 37.4 |
| ms01 | 4 ms | 285.7 | 0.96 | 0.533 | 38.3 |
| ms01 | 1 ms | 286.3 | 0.56 | 0.105 | 38.1 |
| ms01 | 200 µs | 286.3 | 0.47 | 0.025 | 34.7 |

| host | `halt_poll_ns` | video W | vCPU CPUs | poll CPUs | wake after 2 ms, µs | CPUID after 2 ms, µs | `single.sleep` / `tex1.sleep` commit, µs |
|---|---|---|---|---|---|---|---|
| um870 | 200 µs | 7.74 | 0.50 | 0.055 | 29.1 | 4.18 | 19.8 / 47.6 |
| um870 | 1 ms | 7.53 | 0.65 | 0.267 | 28.5 | 4.04 | 24.4 / 49.6 |
| um870 | 4 ms | 24.72 | 2.95 | 2.107 | 4.9 | 1.16 | 15.5 / 34.9 |
| um870 | 200 µs | 7.38 | 0.50 | 0.053 | 25.0 | 4.50 | 24.0 / 52.0 |
| ms01 | 200 µs | 23.4 | 0.38 | 0.057 | 5.72 | 1.25 | 13.4 / 37.6 |
| ms01 | 1 ms | 24.0 | 0.52 | 0.274 | 6.15 | 1.20 | 12.0 / 27.4 |
| ms01 | 4 ms | 41.2 | 4.27 | 2.651 | 4.46 | 1.10 | 8.0 / 18.7 |
| ms01 | 200 µs | 25.0 | 0.44 | 0.084 | 5.86 | 1.00 | 12.8 / 29.1 |

Safari presented 474 to 491 frames in each 10 s file on um870 at every
value, and 487 to 491 on ms01, but for the boot's first file. The idle
desktop drew 3.2 to 3.8 W on um870 at every value, but for the second
200 µs window's 9.8 W, in which the guest kept 0.64 vCPUs busy. On ms01 it
drew 18.5 and 19.5 W at 200 µs, 22.3 W at 1 ms and 23.7 W at 4 ms, with
0.25 CPUs polling.

No value beats KVM's default on either host. At 4 ms the guest's cold wake
is gone, a slept commit is 4 to 19 µs shorter, and um870's Geekbench pass
is 1.7 to 2.3% shorter (in its Background Blur, Face Tracking, Super
Resolution and Feature Matching workloads), ms01's no shorter. For that the
host polls half a CPU more over the pass, 4 W more on um870, and 2.1 to
2.7 CPUs while Safari plays a video, 16 to 18 W more on both hosts. At
1 ms the probe's 2 ms halts outlast the window, so its wake is unchanged;
the Geekbench totals fall inside or 0.6 s beside the two 200 µs passes'
spread; and the host polls 0.2 CPUs more during the video, and draws 1.3 to
1.6 W more over um870's pass.

`-overcommit cpu-pm=on` lets the guest's HLT, MWAIT and PAUSE run without
exits, so each vCPU keeps a host CPU busy while the guest idles: in boots
of jobs `m3-max-29907` and `m3-max-29908`, 7.98 CPUs on um870, its package
at 5.97 W where the idle windows without it drew 3.2 to 4.0 W, and 8.0 CPUs
on ms01, at 20.1 W where they drew 16.8 to 19.5 W but for one of 23.8 W.
On um870 the cold costs stay: the guest's 2 ms wake took 22 µs and its
CPUID after 2 ms 4.1 µs.

## Windows hosts

Windows hosts are being brought up on homelab-mac-mini-2018 (Windows 11
26H2, i7-8700B, 64 GiB, Intel UHD 630), where `scripts/install.ps1`
installs reims-ctl as README.md says ("Windows hosts: installing", below).
QEMU builds there in MSYS2's CLANG64 environment (clang and lld; QEMU builds on
Windows only with MinGW, `vendor/qemu/docs/about/build-platforms.rst`), and
runs x86-64 guests under WHPX, the Windows Hypervisor Platform
(`-accel whpx`):

```sh
../configure --target-list=x86_64-softmmu --enable-whpx --enable-sdl \
  --enable-gtk --enable-slirp --disable-rust --disable-docs --disable-werror
```

**Measured** on the mini (2026-11-15, clang 23.1.5): configure took 169 s
and `ninja -j12` 349 s for a clean build; the binary's accelerators are tcg
and whpx. Without a `reims-vgpu.pc` on its `PKG_CONFIG_PATH`, configure finds
no library ("Run-time dependency reims-vgpu found: NO") and leaves
`reims-graphics-pci` out.

The library builds there too: `cargo xtask dist`, run in CLANG64 with
Rust's `x86_64-pc-windows-gnullvm` target, writes the dist tree
([architecture.md](architecture.md) 11.3), and configure takes it as on
Linux, with `PKG_CONFIG_PATH=<target>/dist-vulkan/lib/pkgconfig` and
`--enable-reims-vgpu`. meson's static lookup links `libreims_vgpu.a` by its
path, and the binary imports no DLL of the library. **Measured** on the
mini (2026-11-15): the library's release build took 14 min 47 s from an
empty target (`m3-max-32508`); renaming the archive's globals and linking
the DLL then took 30 s, and a dist of the same build again 12 s, leaving
the tree's files as they were (`m3-max-32649`). configure took 113 s with
the library and `ninja -j12 qemu-system-x86_64.exe` 293 s, beside SDL
2.32.10's DLL (`m3-max-32660`).

- **CPUID leaf 4.** WHPX takes from QEMU's CPU model only the CPUID leaves
  in its exit list, and Hyper-V answers the rest. Before `cc128f0e` the list
  had no leaf 4, so the guest saw no cache levels: macOS counted 0 CPUs
  sharing the last-level cache and panicked 1.3 s into boot
  (`osfmk/i386/cpu_threads.c:198`, zero divisor), under `-cpu host` and
  `kernel-irqchip=off` alike.
- **The host's CPUID.** WHPX offers the guest no invariant TSC (CPUID
  80000007h EDX bit 8) and no ARAT (06h EAX bit 2) on the mini, so `check`
  warns about both. macOS boots without them: its kernel takes the TSC
  deadline timer, at 3.19 GHz.
- **No `dump-guest-core=off`.** QEMU refuses it on Windows ("Dumping guest
  memory cannot be disabled on this host") and exits, so a Windows host's
  `-machine` leaves it out.

### Windows hosts: installing

A Windows user runs `scripts/install.ps1` in PowerShell, from GitHub
(`irm .../install.ps1 | iex`) or from a clone. It installs MSYS2 and Git
for Windows with winget where they are missing, turns the Windows
Hypervisor Platform on from an elevated PowerShell where it is off (Windows
then needs a restart), clones the repository into `%USERPROFILE%\reims-vgpu`
where it runs from no checkout, brings MSYS2's packages up to date
(`pacman -Syuu`, twice), and runs `scripts/install-ctl.sh --yes` in MSYS2's
CLANG64 shell. That installs its MSYS2 packages with pacman and, where
there is no rustup, rustup with the `x86_64-pc-windows-gnullvm` host, whose
builds link with CLANG64's clang and lld; it builds the library's dist tree
and QEMU (WHPX, GTK, slirp, libusb, no guest agent), copies the CLANG64
DLLs QEMU and qemu-img load beside them, and installs reims-ctl and
reims-util into `%USERPROFILE%\.cargo\bin`. `install.toml` records the
MSYS2 bash the install ran in, and `reims-ctl update` runs the script there,
with `MSYSTEM=CLANG64`. Whatever rustup's default host, the script builds
with `rust-toolchain.toml`'s channel for the `x86_64-pc-windows-gnullvm` host
(`RUSTUP_TOOLCHAIN`), which it installs with `rustup toolchain install`
where it is missing, leaving the user's default as it was: with Windows'
usual default host, `x86_64-pc-windows-msvc`, xtask, the build scripts,
reims-ctl and reims-util would build for MSVC, which links only with Visual
Studio's `link.exe`.

**Measured** on the mini (2026-11-16, `m3-max-36191`), in a scratch MSYS2
extracted from MSYS2's base archive (its package list of 2026-09-27, so the
update step had work to do) and a scratch user profile with rustup's
`rustup-init` run there beforehand:
- MSYS2's extraction and first login took 40 s, `rustup-init` 29 s, and
  install.ps1 1,083 s: 19 min in all. Of install.ps1's time, the library's
  release build took 6 min 4 s. QEMU's install has 68 CLANG64 DLLs beside
  it.
- install.ps1 again took 62 s, and `reims-ctl update` from PowerShell, with
  the checkout unchanged, 52 s, replacing its own running binary.
- `doctor` then passed every line: the hypervisor platform on and a
  hypervisor running, the UHD 630, no recording device, a signed-in
  desktop.
- The user's PATH and rustup's uninstall entry in the registry were as
  before.
- Before the Windows build turned QEMU's guest agent off, its configure
  stopped there for want of `windmc` (`m3-max-36094`); before install-ctl.sh
  tried it for a minute, the move of QEMU's staged install failed once with
  "Permission denied" (`m3-max-36159`).

**Measured** on the mini (2026-11-16, `m3-max-36662`), with a scratch user
profile whose rustup `rustup-init` made with the `x86_64-pc-windows-msvc`
default host and the stable toolchain as its default, as a Windows user's
usually is: install-ctl.sh took 943 s (the library's release build 6 min
9 s), installing `nightly-2026-11-03-x86_64-pc-windows-gnullvm` beside the
user's toolchains. `rustup default` still named
`stable-x86_64-pc-windows-msvc` afterwards, and the user's PATH and rustup's
uninstall entry were as before. reims-ctl.exe and reims-util.exe import the
Universal CRT's `api-ms-win-crt-*` and no `VCRUNTIME140.dll`, which an MSVC
build imports. `doctor` passed every line. The mini's own rustup has the
MSVC default host too, with Visual Studio 18's build tools, so builds for
that host link there; no host here lacks them.

install.ps1's winget and hypervisor-platform steps ran only as dry runs
(`-DryRun`, with `REIMS_INSTALL_PRETEND_MISSING` naming each step's check as
failed): no clean PC has run them.

### Windows hosts: guest RAM in a section

QEMU's anonymous RAM on Windows comes from VirtualAlloc, private to the
process: no second mapping can share its pages, as `mach_vm_remap` shares a
Mac's, so the library could not view a guest buffer's pages in place over
it. Shared RAM (`memory-backend-ram,share=on`, as every RAM_SHARED block) is
a view of a section the paging file backs (`CreateFileMapping` with
SEC_COMMIT, then `MapViewOfFile`, `6c2cae16`), committed whole as
VirtualAlloc's memory is. The RAMBlock keeps the section's handle
(`qemu_ram_get_section`), which `reims-graphics-pci` hands the library with
each range's offset in the section, and the library maps views of the
section's pages (`MapViewOfFile3`) that share the pages WHPX maps into the
guest. The guest's RAM is one block, so
`-machine q35,memory-backend=ram -object memory-backend-ram,id=ram,size=4096M,share=on`
makes one section of it.

**Measured** on the mini (2026-11-15), booting Sequoia's recovery image
(Darwin 24.4.0) on a 15.8.2 VM's OpenCore and OVMF under WHPX, with
vmware-svga, 6 vCPUs, 4 GiB and `-display gtk`, each run with the machine
to itself, QMP screendumps every 15 s, then QEMU's CPU time and memory read
60 s apart (`m3-max-32230` share=on, `m3-max-32232` share=off):
- share=on: QEMU's trace shows one 4 GiB section (`win32_map_alloc
  size:4294967296`), and WHPX maps its view (`WHvMapGpaRange`) as it maps
  VirtualAlloc's memory. The Recovery window was up 2:09 to 2:24 after
  QEMU's start, as with share=off.
- Idle at the Recovery window, 7 minutes in: 0.45 CPUs with share=on and
  0.48 without; working set 0.81 and 0.78 GB, peak working set 4.26 and
  4.27 GB. Private bytes were 0.78 GB with share=on, where the section holds
  the guest's RAM, and 5.03 GB without.
- QEMU quit over QMP and exited 0 both ways.

### Windows hosts: sound

Of the Windows build's audio backends (none, dbus, dsound, wav, as
`reims-ctl doctor` lists them), `dsound` plays on the host's own devices,
and is reims-ctl's default there. Its realize, as QEMU starts, opens
Windows' default output and recording devices; QEMU exits where the output
device fails, and before `e435673c` where the recording device fails too
(read from code: `audio/dsoundaudio.c`, `audio_dsound_realize`).
**Measured** on the mini (2026-11-16, `m3-max-36142`, QEMU at `979d171c`,
realized paused): it has two output devices and no recording device
(`waveOutGetNumDevs` 2, `waveInGetNumDevs` 0), and QEMU exited with "Could
not initialize DirectSoundCapture" on `dsound`, on `dsound` with
`in.voices=0`, and on `dsound` beside a `none` audiodev for the input
device alike; on `none` it realized. So none of these options avoided it,
and the walk's VM started only on `wav` (`m3-max-35999`, where three
`afplay` pings in the guest grew `qemu.wav` from 442 KB to 1.66 MB).

From `e435673c` the recording device is optional: where it can't be
opened, `dsound` plays and logs the failure, and an input voice on it then
fails to initialize (read from code). `reims-ctl` counts Windows' devices
as `waveOutGetNumDevs` and `waveInGetNumDevs` do at each start
(`os::sound_devices`), and routes a VM on `dsound` by them
(`qemu::sound_routing`): with no recording device the input device,
`audioin`, gets an audiodev of its own, `none`, and the guest records
silence; with no output device the VM's sound is all `none`. The start
says so, and `doctor` says which devices there are.
**Measured** on the mini (2026-11-16, `m3-max-36258`, QEMU at
`e435673c`): the three runs above that exited all realized, and a VM on
`dsound` started with its input device on `none`, the start warning so. The
guest listed an output and an input "Apple Virtual Sound Device", and
`afplay` of a system sound exited 0 (no one listened at the host).

## x86-64 macOS guests

What `reims-ctl`'s q35 command gives an x86-64 macOS guest, and why
(`crates/reims-ctl/src/qemu.rs`).

### x86-64 macOS guests: USB input on xHCI

Put an x86-64 macOS guest's keyboard and tablet on an xHCI controller, not
on the USB 2.0 controller `-usb` gives a q35 machine (ICH9's EHCI):

```sh
-device qemu-xhci,id=xhci,addr=0x5,conditional-intr-mapping=on \
-device usb-kbd,bus=xhci.0 -device usb-tablet,bus=xhci.0
```

in place of `-usb -device usb-kbd -device usb-tablet`. The explicit slot
leaves the other devices' PCI addresses where they were, so the boot
entries that name them still match.

On EHCI the guest's input freezes under heavy guest GPU and WindowServer
work (app launches, Geekbench: about one freeze a minute), the keyboard
first, which macOS has report every 25 ms with no key pressed. After a
completed interrupt transfer the guest never activates the next transfer
descriptor of that device's queue: the queue head's overlay is done with its
interrupt-on-complete bit set, and the next descriptor is an inactive dummy,
so QEMU has nowhere to put the device's reports, though it completed the
transfer and raised the interrupt, which the guest took and cleared. The
port stays enabled, the controller keeps running, and the other device keeps
working. Input comes back only when the device is unplugged and plugged
again (QMP `device_del`, then `device_add`) or the guest restarts. The GPU
device shares no interrupt with EHCI: `reims-graphics-pci` raises only MSI-X
(`msix_notify`, or its vector's irqfd). On xHCI the guest binds
`AppleUSBXHCIPCI`, on MSI, the driver of every Mac that runs macOS 26, and
the same load gave no freeze in 20 min (**measured**, macOS 26.6.1 guest,
RTX 5080 Laptop GPU host).

**Unknown**: guests before macOS 26 on `qemu-xhci`.

### x86-64 macOS guests: USB passthrough

reims-ctl gives a q35 VM the host's USB devices with QEMU's `usb-host`, on
the same controller, which has seven ports:

```sh
-device qemu-xhci,id=xhci,addr=0x5,conditional-intr-mapping=on,p2=7,p3=7 \
-device usb-kbd,bus=xhci.0 -device usb-tablet,bus=xhci.0 \
-device usb-host,id=usb-1050-0407,bus=xhci.0,vendorid=0x1050,productid=0x0407
```

Read from QEMU's code (hw/usb/hcd-xhci.c, hw/usb/bus.c,
hw/usb/host-libusb.c):
- The controller has `max(p2, p3)` ports for devices, each a USB 2 and a
  USB 3 port where both counts reach it, and shows the guest `p2 + p3`
  ports, USB 3 ones first. macOS numbers at most 15 ports on a controller,
  so 7 and 7 is the most with every port taking either speed.
- A USB device takes the first free port, and a port freed goes to the back
  of the free ones, so with fewer USB 2 ports than USB 3 ones a device
  attached and detached again can come to a port of one speed alone.
- A device that takes the last free port gets a full-speed `usb-hub`
  chained there first, which a high-speed device does not attach to
  ("speed mismatch"). With the keyboard and the tablet, four devices leave
  one port free, so reims-ctl gives a VM four at most. QEMU's default of
  four ports would leave room for one.
- `usb-host` takes a port as it is created, whether its device is plugged
  in or not. Every 2 s QEMU opens the match of each `usb-host` device
  without one, by `vendorid` and `productid`, and by `hostbus` and
  `hostport` where they are given, and closes a device unplugged; so a
  device plugged in after the VM started joins the guest, and one plugged in
  again comes back. A device it fails to open three times (`libusb_open:
  -3 [ACCESS]` in QEMU's log, where the user may not write its file in
  `/dev/bus/usb`) is tried again only once it is unplugged.

**Measured** on a CachyOS host (kernel 7.3.1) with a macOS 26.7.2 (25G332)
guest on reims-graphics-pci, given a USB stick the host's kernel makes
(`dummy_hcd`'s bus, and `g_mass_storage` on a 64 MB FAT image: 0525:a4a5,
high speed):
- `system_profiler SPUSBHostDataType` lists it ("Mass Storage Gadget", 480
  Mb/s) under the controller's driver, AppleUSBXHCIPCI, and `ioreg -p
  IOUSB` as an IOUSBHostDevice. On macOS 26, `SPUSBDataType` lists nothing.
- macOS mounts its volume, and a file the guest writes there is in the
  host's image once the VM stops.
- Taken back while the VM runs (QMP's `device_del`), it leaves the guest's
  USB tree and its disk list; given again (`device_add`), it comes back at
  another location and macOS mounts it again.
- While the guest has it, QEMU takes it from the host's driver: its disk
  (`/dev/sda`) leaves the host, and comes back once the VM stops.

**Measured** on a CachyOS host (kernel 7.3.3, Ryzen 7 7840HS) with the same
guest, given a USB audio device served on the host's `vhci_hcd` (UAC2 at
high speed, 1d6b:0101, two inputs and four outputs at 48 kHz, its output
asynchronous with a feedback endpoint), on which a Core Audio IOProc ran
three times for 5 s; QEMU at `8b804b45` (before the three rows above) and
at `b8c465b8` (after), one boot each:
- macOS sets the audio interfaces' alternate setting 1 as a stream starts
  and 0 as it stops. Before, `usb-host` made each call holding the BQL: 0.2
  to 2 ms for setting 1, and 0.2 to 6.9 ms for setting 0, the longest on
  the output interface. QMP's round trip, sent 2 ms after each answer, took
  5.5 ms while one of those calls ran, and 6.0 ms at most over 10,000 of
  them. After, the calls take as long on a worker (0.2 to 4.5 ms), and the
  longest of 10,000 round trips was 3.2 ms.
- Between the two calls of each pair macOS sends Set Latency Tolerance
  Value, with BELT 0x47a (about 125 µs) as a stream starts and 0x7d0 (about
  1 ms) as it stops. Before, `qemu-xhci` refused each (an unimplemented command,
  0x14, in its trace), and the guest's kernel logged
  `AppleUSBXHCICommandRing::setLatencyTolerance: completed with result code
  5`, TRB Error, six times; its read of HCCPARAMS2 was traced as
  unimplemented too. After, every one completes with Success, and the guest
  logs no such line.
- macOS sets IMODI to 320 (80 µs) on interrupters 0, 1 and 3 to 7 and to
  500 (125 µs) on interrupter 2, with IMODC 0, twice as its driver starts.
- The IOProc ran 472, 471 and 471 times before, and 457, 472 and 470 times
  after. 2.7 s into the after boot's first stream, the guest's `usbaudiod`
  could not enqueue its feedback endpoint's transfers (0xe00002ee, a request
  for a frame already past; `safeguardAgainstDelays timeDiff 43, threshold
  64`) and restarted the stream, which stopped the IOProc for about 160 ms.
  No call was in flight then and QEMU's main loop shows no stall; one boot of
  each does not tell whether these changes have a part in it.

A second boot of each, with every event QEMU posted to the guest traced
(`usb_xhci_queue_event`; m3-max-22668, the same host and guest):
- The first stream restarted with both QEMUs: the IOProc ran 457 times and
  the guest logged 22 IsoTooOld lines at `8b804b45`, 460 times and 31 lines
  at `b8c465b8`. The restart is not these changes'.
- After, the main loop stalled 5.6, 5.5 and 11.3 ms while the worker set the
  output interface's alternate setting 0 (calls of 5.0 to 7.9 ms), each
  stall the length of the call it overlapped. The main loop was meanwhile
  taking the input stream's transfers from the same device, which still ran;
  that waits for the call, as usbfs takes a lock of the device for both
  (inferred from the timing, not traced in the kernel). Before, the same
  calls took at most 4.1 ms, and no round trip passed 5 ms. So the worker
  frees the main loop only where no other endpoint of the device completes
  during the call.

`78b3270c` keeps the main loop off the device while the call runs: it
handles no libusb events, and the device's submits and cancels wait in
order for the call to return. A boot at `104d7b18` and one with the change
on it (m3-max-26476, the same guest and device, on a CachyOS host with
kernel 7.3.1, a Core i9-13900H and Iris Xe graphics):
- Before, each of the eight calls of 2.9 to 7.7 ms that set the output
  interface's alternate setting 0 while the input stream ran overlapped a
  QMP round trip about as long (2.3 to 7.3 ms), and six of the eight round
  trips over 5 ms overlapped such a call.
- After, the four such calls (4.9 to 6.9 ms) overlapped no round trip over
  1 ms. Three round trips passed 5 ms (5.3, 9.0 and 19.7 ms), none during a
  call, as two did before (7.8 and 12.3 ms): another stall of the main
  loop, which this boot did not trace.
- The IOProc ran 390, 472 and 471 times before, its first stream
  restarting, and 458, 471 and 470 times after.

**Measured** with the same guest, given a USB Video Class camera served on
the host's `vhci_hcd` (1d6b:0102, high speed, YUY2 at 320x240 and 30 fps,
its streaming interface's alternate settings an isochronous IN endpoint of
one to three packets of 1024 bytes each microframe, each frame's number
drawn in its first rows):
- macOS queues each of the endpoint's TDs that crosses a page as an Isoch
  TRB with ISP and Chain, then a Normal TRB with ISP and IOC, and a frame's
  packets end short: the header-only packets between frames, and its last.
  Before `ae97f3f5`, `qemu-xhci` posted the short packet's event at the
  Isoch TRB and none for the Normal TRB (`usb_xhci_queue_event`, m3-max-31655).
  AppleUSBXHCIIsochronousRequest logged `mismatched event TRB` for each later
  event of the endpoint, its requests never completed, and the stream
  stopped within about 100 ms: AVCaptureSession gave no frame in any
  format, on an Iris Xe host (kernel 7.3.1) and with a real high speed
  camera (5986:1193, MJPEG and YUY2) on the RTX 5080 Laptop host.
- After, at `1890afed` on homelab-gem10 (kernel 7.3.3, Ryzen 7 7840HS;
  m3-max-31888), four captures of 4 s each gave 117 to 122 frames, 30 a
  second, none dropped, on `vmware-svga` and on `reims-graphics-pci` (its
  Radeon 780M). Of the frames in the device's own format ('yuvs'), 117 of
  117 and 117 of 118 held their numbers in order, none missed; of the
  frames macOS converted to '2vuy' and '420v', every one but the first,
  which was black on both. The guest logged no `mismatched event TRB`.
- Photo Booth, with the camera at 640x480, showed its frames on
  `reims-graphics-pci` (homelab-um790-xtx-1's Radeon 780M, at `1890afed`;
  m3-max-31967). On `vmware-svga`, for which system_profiler lists no
  Metal support, its viewfinder stayed black while the camera sent 589
  frames (m3-max-31926).

**Unknown**: a guest's view of a device on a USB 3 port; whether macOS
guests before 26 drive the controller with 14 ports as with 8.

### x86-64 macOS guests: sleep

Give an x86-64 macOS guest no ACPI S3:

```sh
-global ICH9-LPC.disable_s3=1
```

**Read** from QEMU's code: a guest in S3 wakes only on PS/2 input
(`hw/input/ps2.c`), not on the USB keyboard and tablet on `qemu-xhci`, nor
on input through the window, and a wake (QMP `system_wakeup`, or an RTC
alarm, which macOS schedules before it sleeps) resets every device
(`pc_machine_wakeup`, `RESET_TYPE_WAKEUP`), the GPU among them. Whether our
kexts bring the display back after that reset is **unknown**. A guest of
Apple's paravirtualized GPU driver, which the tree had before, never did:
its window stayed black and WindowServer waited on the GPU for good.

With S3 disabled, the DSDT has no `\_S3`. `pmset sleepnow` stops at a dark
wake (`sleepASAP 1`): the display turns off (`IODisplayWrangler` power state
1) and the VM keeps running; idle sleep stops at the same dark wake. Input
through the window turns the display on again (power state 4). **Measured**
with a macOS 26.6.1 guest on an RX 7900 XTX host, on that earlier GPU;
on `reims-graphics-pci` it is **unknown**.

### x86-64 macOS guests: audio

reims-ctl gives every VM two virtio sound devices, one with an output
stream alone and one with an input stream alone (vendor/qemu's
`input-streams` property):

```sh
-audiodev pipewire,id=snd0
-device virtio-sound-pci,id=audio,audiodev=snd0,addr=0x7,streams=1
-device virtio-sound-pci,id=audioin,audiodev=snd0,addr=0x8,streams=1,input-streams=1
```

macOS's `AppleVirtIO` kext has `AppleVirtIOSound`, matching virtio device
0x19. **Measured**
with a macOS 26.7.2 (25G332) x86-64 guest on reims-ctl's q35 command with
one device of both streams (`virtio-sound-pci,audiodev=snd0,addr=0x7`), on a
Radeon 780M host:
- `AppleVirtIOPCITransport` binds the device (`pci1af4,1059`), with
  `AppleVirtIOSound` and its user client on it. Core Audio lists one
  device, "Apple Virtual Sound Device", with 2 output and 2 input channels
  at 48 kHz: the default output, system output and input.
- `afplay` of a 3 s 1 kHz tone reaches QEMU. With the audiodev made
  `wav,path=...`, the file holds 3.1 s at 1 kHz. With `pipewire`, as
  reims-ctl runs it, QEMU's stream runs, linked to the host's default sink,
  and `pw-record` of that sink's monitor holds the same tone.
- Safari's Media Source player plays H.264 with AAC (fragmented MP4) and
  VP9 with Opus (WebM) to the end, and their audio reaches QEMU.

On one device of both streams, the guest starts its input stream with each
output stream, and reads it only at first. **Measured** with the 26.7.2
guest on that device, on a host whose PipeWire default source is an analog input and
whose sink is the dummy `auto_null` (homelab-um790's Radeon 780M,
2026-11-12, QEMU's `virtio_snd_*` traces):
- Each `afplay` sets up and starts stream 1, the input, with stream 0,
  and stops both at its end. The guest posts receive buffers for the
  input's first 0.73 s, a queue kick every 90 ms, then none while the
  output plays on.
- QEMU captures only while the guest has receive buffers posted (vendor/qemu
  9a03bc4c): virtio-snd turns the input voice on at the input's START when
  buffers are posted, and at the first buffer posted after a time with
  none, and off when audio comes with none posted. With each `afplay` the
  voice ran 1.05 to 1.08 s, the PipeWire capture stream then paused, and
  the host's default source stayed `SUSPENDED` through plays of 45 s and
  75 s.
- A recording alone (AVAudioRecorder through `osascript`, 8 s) starts
  both streams, posts a receive buffer about every 88 ms throughout, and
  stops both at its end: the voice ran from START to STOP, the default
  source ran with it, and the first captured audio came 40 ms after START. A
  restarted capture reads from then on (pwaudio drops what its ring held
  as the voice turns on), so the largest backlog a read found in the boot
  was 2,820 bytes.
- A recording started while an output plays gets no audio from the
  device: the guest posts no receive buffer for it, its input stream
  having run since the output's START, though AVAudioRecorder reports 10 s
  recorded. QEMU fills only buffers the guest posts, so it was so before
  9a03bc4c too.
- Before 9a03bc4c the capture ran from START to STOP, a cycle every
  5.3 ms (176 kB a second), so a play longer than about 25 s filled the
  4 MiB ring with audio nothing read. QEMU drops what does not fit, says
  so once per stream (`pipewire: capture stream snd0: its 4194304-byte
  buffer is full of audio nothing has read`), and gives the count as the
  stream closes: a 100 s play dropped 13,375,572 bytes in 14,218 cycles
  after the warning 24.9 s in. A reader that finds the ring full now reads
  its newest audio, counting what it skipped (vendor/qemu 24753894; no
  ring filled in these boots). Before 8abc8492 each such cycle wrote an
  `overrun write` line, about 190 a second, and wrote over the ring's
  unread bytes.
- The first recording over ssh raises the guest's microphone prompt for
  `sshd-keygen-wrapper`; a boot answers it over QMP (`input-send-event`
  on the usb-tablet at its Allow button).
- A host with no source (um870, ms01) runs no capture cycles.

Why, read from the code of AppleVirtIOSound's HAL plug-in (the x86_64
slice of macOS 27.0.1's `AppleVirtIOSound.driver`, disassembled for this
and not committed): the plug-in makes one Core Audio device of each
virtio-snd device, with all of its streams, and starts them together. It
posts an input stream's receive buffers at the stream's start (four) and
again at each one's completion while its ring buffer has room, and a
client's read takes from that ring. On a device started for an output,
nothing reads the input's ring, which fills, and the posting ends until
the device stops.

So reims-ctl gives the input a device of its own. **Measured** with the
26.7.2 guest on the two devices above and, as the control, on the one
device of both streams, one cold boot each on the same host with the same
phases (homelab-um790's Radeon 780M, 2026-11-13, vendor/qemu 1422502b,
QEMU's `virtio_snd_*` traces; the harness named the sink as the capture's
target, and the default source, an analog input, ran in its place):
- Core Audio lists two devices named "Apple Virtual Sound Device": one with
  2 input channels, the default input, and one with 2 output channels,
  the default output and system output. Each `AppleVirtIOSound` has one
  stream.
- An `afplay` starts the output device's stream alone: the input stays
  stopped, and the host's source suspended, through plays of 25 and 50 s.
- A 10 s recording started 20 s into a 50 s play starts the input
  device's stream 0.65 s after its command, and the guest posts receive
  buffers to its end, 117 of them, at most 0.13 s apart. The host's
  default source runs with it, no read found more than 2,824 bytes
  waiting, and the recording holds the source's audio in each of its 10
  seconds (98.7% of its samples non-zero; that host's analog input carries
  full-scale noise). On the one device the same recording got no receive
  buffer and no captured read: its first 2 s hold samples from before it,
  and its other 8 s are zeros.
- A recording alone runs as on the one device: 94 receive buffers in 8 s,
  and the source's audio in each second.

On one device of both streams, macOS 11's Core Audio daemon can die at any
sound. **Measured** with the 11.7.11 (20G1443) guest booted with
`vmware-svga` in place of the graphics device, on homelab-um870 (a host
with no PipeWire source: the capture stream ends in `no target node
available`), vendor/qemu 1422502b with the stream traces of 87729d88,
2026-11-13, playing alert beeps (`osascript -e beep`) and system sounds
(`afplay`) in rounds a few seconds apart:
- On the one device, each sound sets up both streams (393,216-byte buffers
  of 65,536-byte periods, stereo float at 48 kHz), posts four receive
  buffers to the input, starts both, and at its end stops and releases
  both. At each of the input's stops the four receive buffers were still
  posted, and the RELEASE completed them, with no audio in them.
- `coreaudiod` died at the 12th round (m3-max-25374), on its
  `com.apple.AVIOStream.data` queue in AppleVirtIOSound's
  `AVIOStreamRingBuffer::writeOutputFrames`, at the assertion `bytesToWrite
  <= _bufferSize`. appwalk8's walk w-780m-11 died at the same place on the
  same plug-in, its `memmove` of 0x1b280000 bytes faulting at an address in
  a shared library's `__TEXT`.
- On vendor/qemu e829b256, before the capture changes of 8abc8492 to
  1422502b, the same rounds killed it at the 12th as well (m3-max-25417),
  in the same function, dividing by a buffer size of 0.
- On the two devices reims-ctl gives, the same harness's 60 rounds started
  the output device alone, its stream with nothing posted at each stop, and
  the daemon lived (m3-max-25414).

Why, read from the code of 11.7.11's AppleVirtIOSound HAL plug-in (74.120.4)
and of its `AppleVirtIO` kext (both disassembled for this and not
committed): each receive buffer's completion block holds a pointer to the
input stream's ring. The plug-in's `stopStream` frees an input ring at once,
because the ring counts data in flight only for output, and then sends STOP
and RELEASE. The kext completes every stream message as a success, passing
the plug-in only `latency_bytes`, so each receive buffer the device completes
at RELEASE, as virtio's sound device must, has its block write into the
freed ring. It dies only where something has taken that memory meanwhile.
macOS 27.0.1's plug-in keeps a stopped ring until its buffers come back, and
sends STOP only then. No answer of the device's avoids it: the guest frees
the ring before it stops the stream. On reims-ctl's two devices a sound
never starts an input stream, so only a recording's end reaches it.


`pa` in place of `pipewire` uses PulseAudio, and `none` gives the guest a
working output whose samples QEMU drops. macOS 26 for x86-64 ships no
Intel HDA driver (no `AppleHDA`; 26.7.2 loads no HDA kext), so `-device
ich9-intel-hda -device hda-duplex` leaves the guest with no audio device;
reims-ctl's x86-64 VMs have `ich9-intel-hda` with an `hda-output` for
OpenCore's boot chime alone, which plays on the same audiodev: the 26.7.2
guest's `wav` capture starts with 2.5 s of it. `AppleUSBAudio`, which
macOS does ship, drives a USB audio device on the xHCI controller above
(`-device usb-audio,audiodev=snd0,bus=xhci.0`), output only.

reims-ctl's default backend is `pa` where QEMU was built without PipeWire:
QEMU's meson takes libpipewire-0.3 0.3.60 or later, and the installer
leaves the feature to configure, so Ubuntu 22.04 (0.3.48) and Debian 11
(0.3.19) get `pa`. Its capture voice keeps a record stream only while the
voice is on (vendor/qemu 1f6f4ea4), so the host's sound server lists QEMU
as a recording client only while the guest records. **Measured** with the
26.7.2 guest on reims-ctl's two devices and `-audiodev pa`, on homelab-ms01's
Iris Xe against pipewire-pulse 1.6.10, the capture named
`auto_null.monitor` (that host has no default source), 2026-11-14
(m3-max-30994):
- No source output of QEMU's while the guest was idle, while it played an
  output alone and after each recording.
- Each of three recordings (3, 8 and 6 s, AVAudioRecorder over ssh) had one
  source output, uncorked, made 2 to 4 ms after the input device's START
  and removed 2 ms after its STOP. The guest sends that device's
  SET_PARAMS, PREPARE and START together, and STOP with RELEASE.
- The recording made during an output holds the guest's own speech from
  the null sink's monitor (83.4% of its samples non-zero, in each second);
  the one alone holds zeros, the monitor of a sink playing nothing.

Before 1f6f4ea4 the record stream ran from the stream's PREPARE to its
RELEASE (read from code). With this guest on the capture-only device that is
the same window, to the millisecond. It is longer wherever the voice is off
inside a prepared stream: while the guest posts no receive buffer
(9a03bc4c), or while the VM is stopped, as the audio core turns voices off
then.

Without an output device, Safari's Media Source players fail. **Measured**
with the 26.7.2 guest above with audio off (`audio.enabled = false`, no
sound device):
- A Media Source stream with an audio track fails with `MEDIA_ERR_DECODE`
  ("Media failed to decode") within 0.1 s of its data's append, before it
  plays: H.264 with AAC, muted or not, and VP9 with Opus. The muted one
  and VP9 never start; the unmuted H.264 with AAC keeps its error while
  its clock runs on after `play()`.
- A video-only stream plays to the end.

**Measured** with a macOS 26.6.1 guest on an RTX 5080 Laptop GPU host:
- YouTube streams VP9 video and Opus audio. The video decodes (WebKit's GPU
  process queued its first frames), then the audio renderer finds no
  device (`err=-101`), and WebKit reports a `DecodeError`. The page keeps
  loading, then shows YouTube's "An error occurred".
- A plain MP4 file plays, without sound.
- With `usb-audio` the guest lists "QEMU" as its default output, and the
  same YouTube video plays.

### x86-64 macOS guests: the CPU model

`reims-ctl` gives a new VM `Skylake-Client-noTSX-IBRS`. Under KVM the model
sets only what CPUID reports; with `check`, QEMU drops a feature the host
lacks and warns once per vCPU. A VM keeps the model it was made with in its
`vm.toml` (`cpu_model` and `cpu_flags` under `[q35]`; VMs `reims-ctl` made
before 2026-10-12 have `Haswell-noTSX`): edit them there, or remove both for
the default, and the VM runs the new model from its next start.

- **Haswell-noTSX** is model 60, stepping 1. XNU panics on model 60 below
  stepping 3 ("Haswell pre-C0 steppings are not supported",
  `osfmk/i386/machine_check.c`, read from code). It boots only behind an
  OpenCore that replaces CPUID leaf 1 (`Kernel > Emulate > Cpuid1Data`):
  reims-util's image sets `0x00050654`, Skylake-SP stepping 4, unless made
  with `--no-cpuid-spoof` (read from code), so its guests see model 85 under
  any `-cpu`.
- **Cascadelake-Server** (v1) adds TSX, which no host here has, and AVX-512,
  which Intel's hybrid parts (12th generation on) lack. On an AMD host with
  AVX-512 the guest gets it, and XNU saves its registers at Intel's XSAVE
  offsets only (`fp_avx512_offset`, `osfmk/i386/fpu.c`, read from code).
- **Skylake-Client-noTSX-IBRS** is model 94, which XNU knows, so it boots
  with or without the leaf 1 spoof; it is Haswell plus RDSEED, ADX, SMAP,
  XSAVEC, PREFETCHW and IBRS, with no AVX-512.

**Measured** with an x86-64 macOS 26.6.1 guest (25G76) behind an OpenCore
whose `Cpuid1Data` is `0x00050654`:
- `query-cpu-definitions` under KVM: on a Ryzen 7 7840HS (Zen 4) every
  model above lacks only `pcid` (and Cascadelake-Server v1 `hle` and `rtm`);
  on a Core i9-13900H Haswell-noTSX and Skylake-Client-noTSX-IBRS lack
  nothing, and Cascadelake-Server lacks TSX and all of its AVX-512.
- On the Ryzen, Haswell-noTSX and Cascadelake-Server each reach the
  desktop; both report model 85 stepping 4 and `hw.cpufamily` `0x37fc219f`
  (Skylake), and only Cascadelake-Server sets `hw.optional.avx512f`.
- Cascadelake-Server's guest there reads AMD's layout from CPUID 0xD:
  opmask at 832, ZMM_Hi256 at 896, Hi16_ZMM at 1408 (Intel's are 1088, 1152
  and 1664). `k3` reads 0 after a signal handler returns, and
  `thread_get_state(x86_AVX512_STATE64)` gives wrong zmm16, upper zmm1 and
  `k3`; setting that state back zeroes `k3` in the thread. Single-threaded,
  Accelerate's sgemm and dgemm run about a third faster than on
  Haswell-noTSX.

**Unknown**: a boot on Skylake-Client-noTSX-IBRS itself, and a boot behind
an OpenCore that leaves leaf 1 as it is.

### x86-64 macOS guests: VMX

`reims-ctl` adds `vmx=on` to `-cpu` where the host's KVM offers VMX to guests:
`kvm_intel` is loaded and its `nested` parameter
(`/sys/module/kvm_intel/parameters/nested`) reads `Y` or `1`, Linux's
default since 4.20. With it the guest's Hypervisor.framework works, so the
guest can run VMs of its own (Docker, UTM). AMD's KVM offers SVM instead,
which macOS does not drive, so there the command is unchanged and `check`
has nothing to drop. QEMU's CPU models leave VMX off, though they
carry its capability MSRs for when it is on. A VM's `-vmx` or `vmx=off` in
`cpu_flags` takes it away: QEMU applies `name=value` flags in order, as
global properties when the vCPU is made, and `+` and `-` ones after them,
at realize (`x86_cpu_parse_featurestr`, `x86_cpu_expand_features`, read
from code). A `+vmx` would outlive a `vmx=off`, the form QEMU's own docs
show, with only a warning of an ambiguous model string.

**Measured** on a Core i9-13900H host (`nested` Y) with an x86-64 macOS
26.6.1 guest: QEMU warns of no missing feature under `check`, the vCPU has
`vmx`, `vmx-ept` and `vmx-unrestricted-guest` (QMP `qom-get`), and the guest
reaches its desktop reading `kern.hv_support: 1`, with `VMX` in
`machdep.cpu.features`. A paused QEMU on Skylake-Client-noTSX-IBRS gives the
vCPU `vmx` with `vmx=on` and with `vmx=on,+ssse3`, and none with
`vmx=on,vmx=off` or `vmx=on,-vmx`, with no warning.

**Unknown**: a VM run inside the guest.

### x86-64 macOS guests: the SMC

An x86-64 guest's SMC is QEMU's `isa-applesmc`: `reims-ctl` gives every
q35 VM one, with Apple's key, and reims-util's OpenCore images carry no
VirtualSMC. It has six keys, `REV `, `OSK0`, `OSK1`, `NATJ`, `MSSP` and
`MSSD`, and takes two commands at its I/O ports (0x300, 0x304 and the
result at 0x31e):
- READ (0x10): a key and a length byte; a key it lacks gets 0x84.
- GET_KEY_BY_INDEX (0x12): a big-endian index of 4 bytes, with no length
  byte, answered with a key's name. It lists `MSSD`, `MSSP`, `NATJ` and
  `REV `, in the order of their names, without `OSK0` and `OSK1`, which a
  Mac finds by name only, and answers 0xb8 from index 4. It has no
  `#KEY`.

Every other command gets 0x82 (bad command), GET_KEY_TYPE (0x13) and WRITE
(0x11) among them. Upstream's device, at the master vendor/qemu is rebased
onto (`e9fa1f66f0`), answers READ alone, and 0x82 to the rest (read from code).
macOS 15.4 and later list the SMC's keys at boot: PerfPowerServices'
`-[PLSMCMetricsAgent getAllKeys]` calls `SMCGetKeyFromIndex` for index
after index until it gets 0xb8, so on upstream's device the walk never
ends, and PerfPowerServices runs a vCPU at 100% for as long as the guest
runs. VirtualSMC 1.3.6 fixed the same walk on its side by answering 0xb8
past the last key
([acidanthera/bugtracker#2480](https://github.com/acidanthera/bugtracker/issues/2480)).
GET_KEY_TYPE and WRITE stay refused: macOS gives each refusal up at once,
and answering them changes nothing PerfPowerServices does (below).

**Measured** with the x86-64 macOS 26.6.1 guest (25G76) behind reims-util's
OpenCore on a Ryzen 7 7840HS host, the guest's port I/O counted with
bpftrace on KVM's `kvm_pio` tracepoint. On upstream's device, from about
30 s after boot, the guest sends about 22,000 GET_KEY_BY_INDEX a second,
PerfPowerServices runs at 100% in `getAllKeys`, and QEMU takes 119% of a
host core with the desktop idle. On vendor/qemu's, each boot walks the
index once or twice, each walk ending at 0xb8, the SMC sees no I/O once the
guest is idle, and QEMU takes 15% to 25% of a host core once the guest's
daemons have settled. The guest sends GET_KEY_TYPE about 60 times a boot
and WRITE a few times at boot and shutdown, and gives each refusal up after
four status reads; a device that answered them, and had `#KEY`, changed
nothing PerfPowerServices does, and the guest never read `#KEY`. macOS
sends no length byte after GET_KEY_BY_INDEX's index, nor after
GET_KEY_TYPE's key: a device that waits for one leaves AppleSMC reading the
status port about 33,000 times a second until its timeout.

### x86-64 macOS guests: the disk on NVMe

A VM whose settings choose NVMe has its system disk on QEMU's `nvme`
device of its own at slot 10, with `discard=unmap` on its drive; every
other VM has it on ICH9's AHCI controller (`crates/reims-ctl/src/qemu.rs`):

```sh
-drive id=disk,if=none,format=qcow2,file=disk.qcow2,discard=unmap \
-device nvme,drive=disk,serial=reims-ctl,addr=0xa
```

macOS's IONVMeFamily drives it in Recovery and once installed. Disk Utility
names the disk `QEMU NVMe Ctrl Media` there, and `QEMU HARDDISK Media` on
AHCI.

**Measured** on 2026-11-12, with the guest set and option ROM of
`a379a5a45`, each guest given 6 GiB:
- Recovery 26.6.2 (25G83), the image `reims-ctl create --macos tahoe`
  downloads, reached its menu 20 to 32 s after QEMU started with a new disk
  on NVMe, and in 30 s on AHCI (jobs m3-max-19896 on `homelab-um870-1`,
  20026 and 20013 on `homelab-gem10-1`, each on a Radeon 780M). On NVMe,
  Disk Utility erased the disk as APFS, and macOS 26.7.3 installed onto it
  and reached its desktop through Setup Assistant (jobs 20026 and 20047).
- Recovery 11.6 (20G165), 12.5.1 (21G83), 13.4.1 (22F82), 14.6.1 (23G93)
  and 15.4.1 (24E263), the images `--macos 11` to `15` download, each
  reached its menu with a new disk on NVMe, and Disk Utility listed the disk
  (137.44 GB, PCI-Express, `disk0`) and erased it as APFS (jobs m3-max-20246
  and 20278, the Radeon 780M on `homelab-um870-1`).
- Installed 11.7.11, 12.7.6, 13.7.8, 14.8.9, 15.8.2 and 26.7.2, each booted
  from overlays of its disks with the disk on NVMe, answered SSH 12 to 23 s
  after QEMU started and presented their desktops, with `disk0` on
  `QEMU NVMe Ctrl` (job m3-max-20061, the RTX 3090 on `homelab-um870-1`;
  not timing runs).

On 2026-10-27 a new macOS 26 VM on NVMe booted Recovery slowly and listed
no disk, and NVMe was not the cause. Two video drivers of that day's guest
set (reims-util `cbde8611`, for the paravirtualized GPU since deleted)
waited in their start, with no timeout, for an IOSurface mapper that
macOS 26's Recovery leaves unbound. With that set, Recovery 26.6.2 stopped
at the Apple logo's progress bar on NVMe (44% at 8 min, job m3-max-20142)
and on AHCI (46% at 13 min, job 20158), on `homelab-ms01-1`'s Iris Xe; the
set of its next commit, `8a8a852`, which bounds only those two waits,
reached the menu in 40 s and listed the NVMe disk in Disk Utility (job
20143). No boot with that set was watched past 13 minutes, so whether its
Recovery came up at last, and without the disk, as reported, was not seen.

**Measured** on 2026-11-12, six cold boots of macOS 26.7.2 (25G332) on
`homelab-um790-xtx-1`, three with the disk on NVMe and three on AHCI, in
turn, each from a reflink clone of the same disk, with reims-ctl's two disk
lines above, on the Radeon 780M and 8 GiB of guest RAM (jobs m3-max-21073,
21194, 21195 and 21196; the host's disk as in "Linux hosts: the guest's
disk"):
- The guest booted sooner on NVMe. It answered SSH 9.9 s after QEMU
  started on NVMe and 12.9 to 13.0 s on AHCI. loginwindow started at 5.6 to
  5.9 s on NVMe and 8.9 to 9.0 s on AHCI, and Finder at 14.4 to 14.7 s and
  18.1 to 18.2 s.
- Two of the three NVMe boots had waits in their first 150 s (a wait as in
  "Linux hosts: the guest's disk", here of 300 ms or more): 3, 7.76 s in
  all, and 7, 13.67 s in all. The third had none, and no AHCI boot had one.
  Each NVMe wait began with a guest flush or write outstanding and 28 to 192
  commands behind it, while the host's SSD had a single request in flight.
  Most came within 2 s of the guest's TRIMs (300 to 390 a boot), which
  reach the image only on NVMe, through `discard=unmap`.
- A 2 GiB file written from `/dev/urandom` (about 500 MB/s in the guest)
  reached QEMU in 8.3 to 12.0 s on NVMe, with 1 to 4 waits, and in 4.8 s on
  AHCI, with none. Read back after a `purge`, it took 0.56 to 0.60 s on NVMe
  and 0.83 to 0.87 s on AHCI. The first GiB of a `tar` of
  `/System/Library/Frameworks` took 7.4 to 8.5 s on either.

**Unknown**: whether NVMe without `discard=unmap` waits as it does with it
there, and how NVMe and AHCI compare on a host whose SSD shows no waits
(`homelab-um870-1`'s, above).

### x86-64 macOS guests: VMware SVGA

reims-ctl runs an x86-64 VM with `graphics = "vmware-svga"` on QEMU's VMware
SVGA II in the slot `reims-graphics-pci` takes otherwise, which the
library takes no part in, in QEMU's own window:

```sh
-vga none -device vmware-svga,id=gfx,bus=pcie.0,addr=0x4 -display gtk
```

scripts/install-ctl.sh builds QEMU on Linux with GTK for it, and reims-ctl
refuses to start such a VM on a QEMU whose `-display help` lists no `gtk`,
until `reims-ctl update` builds one.
Where the environment names no Wayland compositor or X server, GTK's window
ends QEMU as it starts, so reims-ctl gives such a VM `-display none` there.
Closing the window ends QEMU at once (`gd_window_close`, `qmp_quit`), as
closing the library's window powers the VM off.

**Measured** with a macOS 26.6.1 guest on reims-ctl's command line, on an
RTX 5080 Laptop GPU host:
- OVMF and OpenCore drive the device, and macOS boots to its desktop at
  1280 × 720 on the framebuffer the firmware left, with no GPU acceleration:
  macOS has no driver for the device.
- The window shows the guest's desktop on Plasma's Wayland session. With
  the host's screen idle, KWin sends no frame callbacks, and the window
  keeps its first frame ("Guest has not initialized the display (yet)")
  until the screen wakes.
- QMP's `screendump` and `input-send-event` on device `gfx`, head 0, show
  the display and reach the guest's USB tablet, as `reims-ctl view` uses
  them.

**Unknown**: what sets the 1280 × 720 mode (OVMF's platform configuration
or OpenCore's `UEFI > Output > Resolution`), whether a larger one fits the
device's default 16 MiB of VRAM, and the window on GNOME or under X11.

### x86-64 macOS guests: a test base whose apps never open

A guest's disk whose user home carries the BSD flag `restricted`
(SF_RESTRICTED, the flag System Integrity Protection guards) boots and
answers SSH, but no sandboxed app opens a window: Safari, Preview and Maps
start and show nothing. The user's containermanagerd cannot make
`~/Library/Containers` or set its ACLs there ("Failed creating working
directories: Operation not permitted", "Failed to create deathrow"),
exits, and launchd starts it again every 10 s, while the apps wait on it.
Nothing of ours takes part: the same disk does this with no kexts of ours
loaded.

Spot it with `ls -lOd ~ ~/Library`: the flags column reads `restricted`
where a user's own directories show `-`. The owner can be right, the user
itself, all the same. With SIP on, nothing clears the flag: `sudo chown -R`
fails with EPERM, and `diskutil resetUserPermissions / 501` reports success
and leaves it.

The repair runs on a copy, never the base. Boot the copy once behind an
OpenCore image with SIP off and no kexts (`reims-util opencore
--disable-sip`; an image with none of the set's kexts in it, reims-ctl
starts the VM on as it is), run `sudo chflags -R norestricted ~` and `sudo
chown -R <user>:staff ~`, and power it off. Then boot it again on its own
image, SIP on, and check that the apps open. SIP's state is in the VM's
NVRAM, not on its disk, so the repaired disk boots with SIP on as before.

**Measured** on 2026-11-08 with the macOS 13.7.8 (22H730) test base, whose
copies on Linux hosts with a Radeon 780M and with an RTX 5080 Laptop GPU
came from one disk: its home and `~/Library` read `restricted`,
containermanagerd exited every 10 s, and Safari, Preview and Maps opened no
window, with our kexts and without them. After the repair, on a Radeon 780M
host, with SIP on, the three apps opened windows (Safari 5, Preview 4,
Maps 5) and containermanagerd did not exit in 3 minutes.

## Signing

On a Mac, QEMU's build adds entitlements only to an emulator whose target
has HVF (`accel/hvf/entitlements.plist`), which `qemu-system-x86_64` on an
arm64 Mac has not, so the binary carries the linker's ad hoc signature
alone. The library is linked into it statically, so that signature covers
it. `cargo xtask dist` signs `libreims_vgpu.dylib` ad hoc, for hosts that
link it.

Apple's own GPU task service carries a private entitlement for protected
content, `com.apple.private.ggdsw.GPUProcessProtectedContent`, which only
Apple's signature can carry: a trivial ad hoc binary signed with it is
killed at exec (SIGKILL, exit 137; **measured** on the M3 Max). QEMU's
signature carries none, so Metal makes no protected content for it.

## Booting a guest to check a change

A boot checks what no test can: QEMU's side of the API, and what guests do
with the device's answers (AGENTS.md, "Checking a change"). It is run so
that what it shows is the change's:

- **The boot's job holds its GPU.** A boot runs as a potluck job holding
  the GPU it boots on (`--gpu`), so no test that opens a Metal or Vulkan
  device shares it. A boot whose timings are compared with another's takes
  the whole machine (`--exclusive --hold-line`), since a cold boot compiles
  its pipelines on the CPU and a build beside it would skew them.
- **Copies only.** A boot runs on an APFS clone, a copy or a `snapshot=on`
  overlay of the guest's disk, and a step that changes the guest's state (a
  `defaults write`, an app relaunched with new settings, a file written)
  runs only there; a harness refuses it on a guest's own disk, naming the
  disk. On a persistent disk such a step can leave a fault the guest never
  repairs: an x86-64 macOS 26 guest whose Dock was relaunched with new
  settings while its icon service was failing saved an icon cache of blank
  icons, and its Dock never came back. A damaged disk is copied again from
  its source. Before blaming the guest or the platform for a broken disk,
  read what our own steps did on it.
- **A timing boot runs on a clone.** A boot whose timings are read (a
  frame's gaps, a page load, a stall) runs on a clone of the guest's disk, a
  reflink on Linux (`cp --reflink=always`, the copy `reims-ctl` makes) or an
  APFS clone on a Mac, not on a qcow2 overlay (`qemu-img create -b`, or
  `snapshot=on`). The guest's first write to each 64 KiB cluster of an
  overlay reads the rest of the cluster from the base image, and on a Linux
  host's btrfs `/home` (CachyOS's `compress=zstd:1`) that read is of cold,
  compressed and checksummed data. Measured on `homelab-um790-xtx-1`, two
  cold boots of macOS 26.7.2 on each layout of the same disk (2026-11-11,
  jobs m3-max-19441 and m3-max-19474): the overlay's guest writes took up to
  1.90 and 1.96 s, and the guest's commands waited with none completing for
  100 ms or more 19 and 18 times (8.5 and 4.8 s in all); on the reflink
  clone no write took over 28 ms, and such waits came 6 and 3 times (1.6
  and 0.5 s). The host's SSD took 1.8 to 2.0 s over a single read now and
  then on both layouts; only the overlay put a guest write behind one
  ([lessons.md](lessons.md)).
- **The screen is checked.** Each boot checks a screendump of its desktop:
  the Dock shown, the wallpaper not flat, the menu bar there, and a fixed
  alert's corners and background against a known-good one. A boot can pass
  desktop and SSH checks, frame counts and crash reports with its Dock
  missing. A build someone else will use passes this boot first. Builds
  that differ in what they show are bisected on `snapshot=on` overlays, so
  the disk cannot vary.
- **Guest images come through QEMU**, by QMP `screendump`, never from a
  capture of the host's screen, which reaches into the session of the
  person at the host: a single-instance screen recorder's command line
  attaches to the recorder already running and ends its recording.
- **A control.** A guest that misbehaves boots beside a control that
  changes one thing back (the library of the branch's base, or the last
  good kext set), so the difference is known to be the change's.
- **The steps cover what the change reaches.** A change to pixel formats
  boots to the desktop and plays video in Safari; the window's walk
  (resize, zoom, full screen, minimise, close) runs once for a change to
  the window, not on every boot.
- **An app that shows no window**: capture its own log and a `sample` of
  it in the guest before the VM goes down, and before blaming the host.
- **The host is set up as users' are** (AGENTS.md, "Measure what users
  get"). A policy users' hosts carry stays, such as a priority daemon's
  rule for QEMU (ananicy-cpp's nice +9 and low I/O priority on CachyOS). A
  harness's own artifact goes: QEMU started over SSH inherits sshd's
  `SCHED_IDLE` class, and the launch puts it back to `SCHED_OTHER` at nice
  0, as a terminal would.
- **Out of scope**, noted in one line and left: a crash in QEMU's quit or
  teardown after the run's work is done (exit 139), which is QEMU's own,
  and a lone GPU hang the driver reports (an NVIDIA Xid) that the same boot
  run again does not show. A crash while the guest runs is in scope.

**The x86-64 guests** for a check's boots are on `homelab-um870-1` (Radeon
780M), one of each release, in `~/.cache/rvg-vms`: `macos-11.7.11`
(20G1443), `macos-12.7.6` (21H1320), `macos-13.7.8` (22H730),
`macos-14.8.9` (23J631), `macos-15.8.2` (24H232) and `macos-26.7.2-shared`
(25G332), each a `disk.qcow2` with its `opencore.qcow2` and `OVMF_VARS.fd`.
`macos-14.8.9` and `macos-15.8.2` are copies of the guests of the same names
in `~/.cache/rvg-vms` on `msi-laptop`, equal by SHA-256 on both hosts and
clean by `qemu-img check` (2026-11-09). Each is an 8 GiB guest; one job
booting `macos-14.8.9` and `macos-15.8.2` in turn, with the device, held
`--ram 18`, and their QEMUs peaked at 11.8 and 16.7 GiB of anonymous and
shared memory.

## arm64 guests

arm64 macOS guests ran on QEMU's `vmapple` machine on an Apple silicon Mac,
and the machine's display was Apple's paravirtualized GPU. That device is
gone from the library and from QEMU: `make qemu` builds no
`qemu-system-aarch64`, and `reims-ctl` starts no arm64 VMs (it makes them,
by a restore from an IPSW, above). `vmapple` builds
again, without the device, in an `aarch64-softmmu` build, and runs under HVF
or TCG on Ambrosia, our own arm64 firmware, which enters macOS's kernel
itself. Virtualization.framework's firmware, AVPBooter, does not boot on the
machine's layout now. The branch's commits for it ("What vendor/qemu
carries", above) say what each one changes and why.

### arm64 guests: the device on Ambrosia

`firmware/ambrosia` is the device's driver module for Ambrosia
([architecture.md](architecture.md) 11.5), which gives an arm64 guest an
early console on the device. `make qemu` builds no `qemu-system-aarch64`,
and runs configure again when its build directory holds an `aarch64-softmmu`
build, so one with the device is configured by hand in a directory of its
own, on the same dist tree, with `make qemu`'s options and `vmapple`'s:

```sh
make                                    # cargo xtask dist: target/dist-vulkan
mkdir -p build/qemu-aarch64 && cd build/qemu-aarch64
PKG_CONFIG_PATH=$PWD/../../target/dist-vulkan/lib/pkgconfig ../../vendor/qemu/configure \
  --target-list=aarch64-softmmu --enable-hvf --enable-reims-vgpu --enable-gnutls \
  --disable-docs --disable-bsd-user --disable-linux-user --disable-tools
make qemu-system-aarch64
```

configure's summary says `reims-vgpu support: YES` and `HVF support: YES`,
and QEMU's build signs the binary with the hypervisor entitlement. A guest
boots on Ambrosia as `vendor/qemu`'s `docs/system/arm/vmapple.rst` says,
with the device and a configuration bundle that names the module
(`make -C firmware/ambrosia bundle`) added; `REIMS_VGPU_WINDOW=0` keeps the
library's window shut, and QMP's `screendump` shows the display:

```
-device reims-graphics-pci
-device loader,file=firmware/ambrosia/build/ReimsGraphics.cpio,addr=0x120000000,force-raw=on
```

**Measured** on the M3 Max under HVF on 2026-11-08, with `vendor/qemu` at
`aef8a7cf` configured as above on the library of `61b1aac9e`, Ambrosia's
`44f0dac` and APFS clones of a macOS 26.6.2 (25G83) guest's disks:
- With the module, in each of two boots, Ambrosia logged `reims-graphics
  at 00:05.0: 1920x1080, framebuffer 0x80400000, registers 0x50000000` and
  started its console there. The screendumps showed its log from the
  console's start to the kernel's entry, and the kernel's progress bar
  beneath it, advancing; the guest has no driver for the device (our kexts
  are x86-64), so that is all the display showed. The guest answered SSH
  12 s after QEMU started, with System Integrity Protection enabled and
  `kern.bootargs` empty, and QEMU logged no `reims-vgpu: fatal` line. Once
  the second boot answered SSH, QEMU's `info pci` had BAR 0 at 0x50000000
  and BAR 2 at 0x80000000, where the module put them, and the other
  devices' BARs from 0x50001000 up, where the kernel put them.
- With the device and no module, the guest answered SSH 11 s after QEMU
  started, with SIP enabled and no boot arguments; the display stayed
  black, and `info pci` listed both of the device's BARs as not mapped.
- The device as it was before `7aee20f4`, with the scanout's dirty log
  (`c7909d60`'s `reims-graphics-pci.c`, on the library of `a851db6a7`),
  stopped QEMU under HVF as soon as the module set the scanout:
  `accel/hvf/hvf-all.c:74:do_hv_vm_protect: assertion failed:
  (!(size & ~page_mask))`.

### arm64 guests: the Neural Engine

`vmapple` gives a guest no Neural Engine. Virtualization.framework gives
its macOS guests a virtio one: the guest's AppleNeuralEngine sends compile,
load and evaluate requests to the host through
`AppleVirtIONeuralEngineDevice.kext` (`_ANEVirtualClient`, read from
AppleNeuralEngine's strings in macOS 27.0.1, which ships the kext), and the
host's Neural Engine runs the models. `vendor/qemu` has no such device. A
host with no Neural Engine would have to run the guest's models itself.

**Measured** on the M3 Max under HVF on 2026-11-14 (development
`51b141b58`'s arm64e set), with a probe that runs `VNRecognizeTextRequest`
on a rendered line of text and VisionKit's `ImageAnalyzer`, the API Live
Text uses:
- macOS 14.6.1, 15.6.1 and 26.6.2 read the text at both recognition levels,
  and so does `ImageAnalyzer`. TextRecognition runs Espresso models
  (`cr_td_model_v3_eir` or `cr_td_model_v3`, `cr_tr_model_latincyrillic_v3`)
  on the CPU (`eng=0`): on 15.6.1 and 26.6.2 Espresso logs `VM detected:
  Adjusted requested context from (eng=5 dev=-1) to (eng=0 dev=-1)`, and on
  14.6.1 it asks for the CPU from the start.
- macOS 27.0.1 (26A434) and 27.2.1 (26B131) read the text at the fast level
  only. At the accurate level, the default, `performRequests:error:` answers
  NO with no error and no results, whatever compute device the request
  names, and `ImageAnalyzer` finds no text. macOS 27 ships only E5 models
  for Latin text (`cr_td_model_v3_e5`, `cr_tr_model_latincyrillic_v3_e5`),
  and TextRecognition compiles them for the Neural Engine alone: Espresso
  throws `No available backends on platform [universal] for compute units
  ane`. Espresso compiles for nothing but the CPU in a VM (`On-device
  compilation within a VM only supports CPU currently`), and through Core
  ML the detector compiles for no CPU backend, in the guest or natively
  (`Input, 'img_input' is not of type tensor`). Vision's
  `VNDetectLensSmudgeRequest` refuses as well (`com.apple.Vision` error 22,
  which names the Neural Engine). Each of Vision's 24 other image requests
  answered as it does natively.
- An x86-64 macOS 26.7.2 guest on a Linux host, on a Radeon 780M, reads the
  text at both levels, and so does `ImageAnalyzer`: x86-64 macOS's
  TextRecognition runs Espresso models on the GPU (`eng=5`), our device.

### arm64 guests: the boot check of our configuration

Ambrosia's done list has 10 boots of our configuration: macOS 13.6, 14.6.1,
15.6.1, 26.6.2 and 27.0.1, each under HVF on the M3 Max and under TCG on a
Linux member. The boot check covers four of them, macOS 14.6.1, 15.6.1,
26.6.2 and 27.0.1, and macOS 27.2.1 (26B131) beside them: arm64 macOS 11 to 13
are not supported for now (owner, 2026-11-09), since macOS 13.6 offers the
device no MSI and the kexts have no INTx path, and the arm64e set is placed
from Darwin 23. 27.2.1 boots on Ambrosia's development from its `0e6489b` on,
not on Ambrosia 0.1.1: there its kernel is entered and no login comes (job
`m3-max-14589`), as 27.2.1's kernel checks a memory size in its boot arguments
that 0.1.1 leaves at 0. Our configuration is the module's early console with Graft and
our two kexts added to the kernel collection. Its bundle is built from the
kexts' arm64e builds, and `check-boot.sh` checks one boot of it:

```sh
make -C firmware/ambrosia config-bundle GRAFT=<Graft.kext> \
  REIMS_GRAPHICS=<ReimsGraphics.kext> RGFXGATE=<RgfxGate.kext>
make -C firmware/ambrosia guest-tools          # on a Mac: bench and present_probe for arm64
firmware/ambrosia/check-boot.sh -f <Ambrosia>/target/ambrosia.bin \
  -q <qemu-system-aarch64 with the device> ~/.local/share/reims-ctl/vms/macos-26
```

- **The bundle.** `config-bundle` writes `build/Config.cpio`: the module's
  `config.plist` with a `Kernel` > `Add` entry for each kext, Graft first,
  `Graphics` > `DisplayNode` set to the list `reims-display0` to
  `reims-display3`, and the kexts under
  `Kexts/`. It checks each kext's `CFBundleIdentifier`
  (`ai.aneesiqbal.graft`, `ai.aneesiqbal.ReimsGraphics`,
  `ai.aneesiqbal.RgfxGate`) and that its executable has an arm64e slice,
  and names every kext that is not given or fails, writing no bundle.
- **The Ambrosia it needs.** `Graphics` > `DisplayNode` as a list needs
  Ambrosia 0.1.3 (tag `v0.1.3`, commit `1b924d9`) or later (Ambrosia's
  `docs/configuration.md`, "Graphics > DisplayNode"); an older Ambrosia
  refuses the bundle (`Graphics.DisplayNode is not a string`). Where the module drives a
  display, Ambrosia adds `/arm-io/reims-display0`, a `display-subsystem`
  node, to the device tree before the kernel starts, and one for each other
  display the module reports, and logs `ambrosia: display node
  /arm-io/reims-display<i>, a display-subsystem` for each; ReimsGraphics uses
  each node as its display's ([architecture.md](architecture.md) 11.5).
- **Several displays need Ambrosia 0.1.3.** On a device of two displays or
  more (`reims-graphics-pci,displays=N`), the module takes `MULTI_DISPLAY`
  where Ambrosia gives its graphics driver interface's revision 2, which
  Ambrosia 0.1.3 is the first release to give, and reports the count; Ambrosia
  then adds a node for each display `Graphics` > `DisplayNode` names, and the
  arm64e kext drives each display with a node ([graphics-kext/design.md](graphics-kext/design.md),
  "Multiple displays", A1 and A2). Our configuration names a node for each of
  the four displays a device can have, so one bundle serves every count, and
  Ambrosia adds those of the displays the module reports, logging none past
  them.
- **Measured at two displays and one.** On Ambrosia 0.1.3's image
  (`ambrosia.bin` 617ac54c) with our bundle, arm64 14.6.1, 15.6.1 and 26.6.2
  under TCG on homelab-um870-1's Radeon 780M, and 27.0.1 under HVF on the M3
  with no window, each booted at `displays=2` to the desktop on both heads:
  Ambrosia logged a node for display 0 and display 1, WindowServer logged
  `FB Count = 2`, CoreGraphics listed two displays with display 0 the main
  one at (0, 0) and display 1 at (1920, 0), and no `reims-vgpu: fatal` line
  came (potluck jobs m3-max-35037 and m3-max-35038, 15.6.1 twice). At
  `displays=1` each booted with one node, one display and the feature word
  of a boot before multiple displays (m3-max-35037, 35039, 35254, 35255).
  Before the release registered the displays in turn, 15.6.1's WindowServer
  took one of two services in one boot (m3-max-34657, `Timeout: expect: 2
  current: 1`) and made display 1 the main one in another (m3-max-34945).
  `system_profiler SPDisplaysDataType` lists no display on an arm64 guest at
  either count, so the boots count displays through CoreGraphics.
- **The boot.** The script boots APFS clones of the guest's disks (`cp -c`;
  `cp --reflink=always` on Linux) in its log directory and removes them at
  the end, with 8 GiB, 4 CPUs, the device, `REIMS_VGPU_WINDOW=0` and
  `REIMS_VGPU_LOG=info`, and forwards SSH from port 10090 (`-p`). Under TCG
  the guest's RAM is a shared memfd, as `reims-ctl` gives its VMs on Linux:
  `-machine vmapple,...,memory-backend=ram,dump-guest-core=off` and
  `-object memory-backend-memfd,id=ram,size=8G,share=on`. Under HVF the RAM
  stays QEMU's own, which the device maps with no file on macOS ("The
  reims-graphics device", above).
- **The guest's RAM on `vmapple` is not all memfd.** Besides the guest's main
  RAM, `vmapple` has two small RAM regions with no file behind them:
  `firmware` (1 MiB at 0x100000, where Ambrosia runs) and `VMApple Config`
  (64 KiB at 0x400000), each made by `memory_region_init_ram`, and no
  command-line option gives them one. On Linux the device maps the
  file-backed ranges and leaves these two out ("The reims-graphics device",
  above), so with the memfd it maps the main RAM in place and prints no copy
  warning; an IOSurface never lies in those two regions, so every surface
  still maps in place. (With the main RAM not a memfd either, no range is
  file-backed, and the device warns `no guest RAM is a shared memfd in pages
  of the host's size: IOSurfaces are copied for the GPU` and copies every
  surface.) On macOS it maps all three, with no file.
- **The checks**, each a line with PASS or FAIL and what it found, then the
  verdict; the script's header says what each one reads. `firmware`:
  Ambrosia entered the kernel; where it stops the boot, its line, and QEMU
  is stopped at once. `console`: the module's early console. `ssh`: a login
  within 90 s of QEMU's start under HVF, 450 s under TCG. `sip`, `boot-args`:
  `csrutil status` enabled and `kern.bootargs` empty. `kexts`: `kmutil
  showloaded` lists the three identifiers. `windowserver-metal`: the lane
  gate's reading of it, the desktop's processes up, `bench -l` listing a
  Metal device that is not Apple's paravirtual one, WindowServer's pid kept,
  and the host's log showing an IOSurface the guest presented.
  `metal-app`: `present_probe` full-screen in the user's session, under
  `report.py`'s bar, measured from a second after its window is first shown,
  which it waits up to 60 s for (`-v 60`, the line's `visible_s`): under TCG
  the guest's post-login launches can hold a new window back for seconds
  ([lessons.md](lessons.md)), and a window not shown in that time fails,
  saying so. Under TCG it passes on frames presented with at most 3
  answers missing, and its drops are printed but not judged, as the emulated
  CPU sets WindowServer's composite rate (pacing is judged under HVF only).
  Its line also gives the host's display flips during the probe,
  WindowServer's own present rate, and on Linux the host's pressure over it
  (`/proc/pressure`: the time a second some task waited for a CPU, for
  memory and for IO, and all did for memory; `/proc/vmstat`: direct
  compactions and reclaims). The probe's window is full-screen at the status
  level, which needs no right to be in front ([lessons.md](lessons.md)). A
  probe that falls short while Setup
  Assistant runs fails as unread, naming it, as its full-screen window
  covers the probe's. `bundle-install`: RgfxGate's Graft client state reads
  installed, tagged or present, and `/Library/GPUBundles/reims-graphics.bundle`
  carries the restricted flag and the `KernelExtensionManagement` rootless
  class. `amfi-map`: WindowServer maps the bundle's executable (`sudo lsof`
  on its pid) and the boot's log has no AMFI or library-validation denial
  naming it. `page-count`: the kext's count of a surface's pages
  (`mach_vm_region` over `kernel_map`, inside IOKit's pageable maps by
  `IOIteratePageableMaps`) did not take the guest down: after the Metal work,
  its `kern.bootsessionuuid` is the login's (each boot sets a new one;
  `kern.boottime` is not compared, as macOS corrects it while its clock
  settles) and no panic report was written in the run; its line gives the
  host's last count of IOSurface device caches (made, deferred, carrying
  pages, refused) and the kext's count lines (counted, not counted). Zero
  deferred while caches were made, no count line, or no count that counted
  is printed under the verdict as a `check-boot: finding:` line, not a
  failure. `ax-matrix`: the boot's log has no `not able to set accessibility
  matrix` line from WindowServer. `icc`: the boot's log has no `Failed to
  set ICC curve` line and at most one `Failed to set ICC matrix`, the
  cursor's layer 1 (ReimsGraphics' `set_block`, RgfxMobileICC.hpp); the
  read counts only where the log printed its header. `amfi-refuse`: last of the guest's checks,
  a copy of `/usr/bin/true` signed ad hoc with `com.apple.rootless.install`
  is killed by AMFI (exit 137), its refusal passing through RgfxGate's
  signature hook, and 5 s later `kern.bootsessionuuid` is still the
  login's. `screen`: QMP's screendump at the end (`screen.png`, else
  `screen.ppm`) is not one colour or nearly so: its commonest colour covers
  at most 90% of it, and it has 1024 distinct colours or more. A desktop
  that WindowServer composites empty is one colour while it still presents,
  which `metal-app` passes, as it measures presentation and not pixels. The
  bounds are well clear of the screendumps of 69 passing boots under HVF
  on 2026-11-11 (14.6.1, 15.6.1, 26.6.2, 27.0.1 and 27.2.1): the commonest
  colour on 3.7% to 21.0% of the screen, and 240,414 to 404,446 distinct
  colours.
  `no-fatal`: no `reims-vgpu: fatal` line in QEMU's log or the serial log. A reading in the guest that printed nothing fails its check as
  unread (a desktop poll that never answered is no desktop), and under TCG
  each command in the guest, each SSH login the boot waits on, and the wait
  for the desktop have 5 times their HVF limits.

**Measured** on the M3 Max under HVF on 2026-11-08, with our QEMU at
`bc378624` built as above on the library of `61b1aac9e`, Ambrosia 0.1.0's
released image (`ea2987b`) and APFS clones of the 26.6.2 (25G83) guest
`macos-26`:
- With the module's bundle alone (`make bundle`, no `Kernel` > `Add`), six
  checks passed: `firmware`, `console` (`reims-graphics at 00:05.0:
  1920x1080, framebuffer 0x80400000`), `ssh` at 13.5 s, `sip`, `boot-args`
  and `no-fatal`. Three failed, as they must with no kexts: `kexts` (none of
  the three loaded), `windowserver-metal` (WindowServer up with no Dock or
  Finder after 60 s, `bench -l` listing no Metal device, no IOSurface shown)
  and `metal-app` (not run, with no Metal device). The screendump showed
  Ambrosia's log on the early console and the kernel's progress bar. The
  run took 76 s, 60 s of it the wait for a desktop.
- With a bundle by hand holding `Kernel` > `Add` entries for the x86-64
  builds of the three kexts (which `config-bundle` refuses), Ambrosia read
  the three, then logged `boot failed: KextsNotPlaced` after reading the
  disks, on the console too; `firmware` failed with that line, the guest's
  checks failed as not run, and the run took 1.2 s.
- With a login over SSH for each guest command, one boot's last login was
  refused ("Permission denied") after about 20 had succeeded; the guest's
  commands now share the first login's connection and retry a failed login.

Graft's arm64e build and the arm64e builds of `ReimsGraphics.kext` and
`RgfxGate.kext`, with the Metal bundle inside RgfxGate for arm64e and arm64,
are `guest/prebuilt-arm64e` (`make -C guest prebuilt-arm64e`):
`GRAFT=guest/prebuilt-arm64e/Graft-arm64e.kext`,
`REIMS_GRAPHICS=guest/prebuilt-arm64e/ReimsGraphics.kext` and
`RGFXGATE=guest/prebuilt-arm64e/RgfxGate.kext`.

**A pass** (measured 2026-11-09 on the M3 Max under HVF, development
`58a5bf9df`, Ambrosia `3d094f9`, jobs `m3-max-12987` and `12993`): every
check passes on 14.6.1, 15.6.1, 26.6.2 and 27.0.1. Graft binds the arm64e
ReimsGraphics, its display and accelerator together; the display is an
IOMobileFramebuffer ([graphics-kext/design.md](graphics-kext/design.md),
"Decided", item 13).

**27.2.1** (measured 2026-11-11 on the M3 Max under HVF, Ambrosia development
`8a05c40`, development `87e04846c`'s QEMU): every check passes with the arm64e
set that knows 27.2.1's IOMobileFramebufferLegacy (job `m3-max-14598`). The
host's display switched to an IOSurface the guest presented 879 times, and
`metal-app` presented 601 frames, none dropped. With the set before it (job
`m3-max-14599`), the kext begins no swap: `windowserver-metal` finds no
desktop in 60 s and the host's log no IOSurface the guest presented. 27.0.1
passes on Ambrosia 0.1.1 with the same set (job `m3-max-14590`).

**Beside a build** (measured 2026-11-11 on the M3 Max under HVF, 27.0.1
(26A434), each probe's 10 s beside N spinning processes started 2 s before
it, jobs `m3-max-14119`, `14149` and `14347`): at the default QoS, before
`5658f96c`, `metal-app` held 59.8 to 60.0 frames a second beside 16
processes, one per core, and fell to 22.5 and 24.4 beside 64 (CVDisplayLink
at 29.9 and 36.2 Hz) and to 13.4 beside 128 (21.4 Hz), with few drops. A
loop in the guest sleeping 5 ms at a time woke up to 1.9 s late, while a host
thread sleeping to a 60 Hz grid, as the device's refresh thread does, ran
every tick, at most 12.7 ms late: the guest's vCPUs, CPU-bound threads at the
default QoS, got a share like each spinning process. With `5658f96c`'s
user-initiated vCPUs beside 64 the probe ran at 59.9 frames a second with
CVDisplayLink at 60.0 Hz, gaps of at most 33 ms and the guest's sleeps at
most 9.4 ms late; its commit-to-`presentedTime` latency was 48.7 ms at the
median, against 32.5 ms on an idle host, a frame later, as the host's own
threads stay at the default QoS. A gate's 27.0.1 probe ran at 39.1 frames a
second (CVDisplayLink 45.4 Hz) beside a full-core `cargo xtask dist` on the
same Mac for this reason.

**Beside a build on Linux** (measured 2026-11-11 on `homelab-um870-1`, a
Ryzen 7 8745H with 16 threads on CachyOS, x86-64 macOS 26.7.2 (25G332) on
KVM with 8 vCPUs and the Radeon 780M, `present_probe` full screen for 10 s
beside N `yes` processes at nice 0, jobs `m3-max-14318`, `14617` and
`14702`): ananicy-cpp runs `qemu-system-x86_64` at nice 9 (its `Heavy_CPU`
type), and within one cgroup nice alone divides the CPU, so each of QEMU's
threads weighs 110 against a spinning process's 1024. With QEMU and the
processes in one cgroup, as a VM reims-ctl left in the cgroup of the
terminal that started it shares that terminal's build, the guest ran the
probe at 60.0 frames a second alone, and beside 16 processes it did not
finish the probe's 10 s within 2 minutes; beside 64 it did not answer a new
SSH login. With the 64 processes in a systemd user scope of their own and
QEMU where it was, it ran at 29.0 and 50.2 frames a second in two boots.
reims-ctl now starts QEMU in a systemd user scope of its own,
`reims-ctl-NAME.scope` (`runtime::launch`); with the processes in the
starting cgroup, the guest ran the probe at 39.7 frames a second beside 16
(CVDisplayLink at 51.1 Hz, gaps of at most 399 ms, the guest's 5 ms sleeps
at most 245 ms late) and at 32.2 beside 64 (53.1 Hz, 293 ms, 169 ms): its
cgroup and the build's weigh the same, and the guest's frame rate follows
its vCPUs' share. A host thread sleeping to a 60 Hz grid, as the device's
refresh thread does, kept every tick in every run, at most 5.9 ms late.

**Beside a terminal's build on Linux** (measured 2026-11-13 on
`homelab-um870-1`, x86-64 macOS 26.7.2 on KVM with the Radeon 780M, the boot
harness's walk of the desktop and Safari's pages, jobs `m3-max-27804` and
`m3-max-27882`): with a cold build of this repository running beside the VM,
QEMU in its job's cgroup and the build in a systemd user scope of its own,
each at CPU weight 100 as `reims-ctl-NAME.scope` and a terminal's scope are,
the guest's pages stop for up to 2.6 s. ananicy-cpp gives every `bash` nice
-4 (its `Doc-View` type), so a build typed into a bash terminal runs its
compilers at nice -4, against QEMU's nice 9. Beside a cold `cargo xtask
dist` started that way, with 8 vCPUs, Safari scrolled at 38.3 frames a
second with a largest frame gap of 1097 ms, its autoscroll ran at 23.9 with
2582 ms and a WebGL scene at 41.4 with 116 ms, against 60 frames a second
and gaps of at most 37 ms on an idle host. QEMU got 1.7 to 1.9 cores against
the build's 9 to 15. In each host present gap of 100 ms or more, the vCPUs
spent 51 to 87% of their runnable time waiting on a run queue, while the
device's threads were idle, QEMU's memory pressure stayed at most 63 ms a
gap and the vCPUs took no major fault. The gaps are 2.4 to 28 times what
that lost share accounts for (the idle host's gap divided by 1 - L, L the
vCPUs' share on a run queue); the guest's own waits across its vCPUs would
multiply it, which no counter shows (inferred: KVM's pause-loop yields
stayed at 0, at kvm_amd's stock `pause_filter_count` 3000 and
`pause_filter_thresh` 128, so it is no spin-wait). A cold `cargo build` of
the workspace at nice 0 stopped the pages for at most 167 ms. With 4 vCPUs
beside the nice -4 build, the scroll's largest gap fell to 365 ms at 44.8
frames a second, still 5.3 times its bound, and on the idle host 8 guest
workers each piping 3000 MiB through `md5` took 9.67 s a round, against 5.34
s with 8 vCPUs: reims-ctl's default stays at half the host's threads, at
most 8. Nothing of ours moves the stall: the device's threads wait on
nothing in it; KVM's pause filter is a module parameter, which QEMU does not
set; KVM polls a halted vCPU only while no other task wants its CPU, as the
build's tasks do, and QEMU leaves a VM's polling to the host ("Linux hosts:
KVM halt polling"); QEMU's `-overcommit cpu-pm=on` keeps idle vCPUs on the
host's CPUs; KVM's paravirtual spinlocks and TLB flushes need a guest
kernel that uses them, and macOS's does not; and a higher CPU weight or nice
for QEMU's scope would raise it over the user's own work.

**A TCG run** (`-t`) runs on a Linux member. What it needs:
- **A node** with the arm64 guests: `homelab-um870-1` holds reflink copies of
  all five in `~/.cache/rvg-vms` (`macos-13-arm64` to `macos-27-0-arm64`,
  each with its `vm.toml`). The script's log directory (`-o`) must be on that
  filesystem, so that its clones are reflinks. Another member needs copies of
  the guests first.
- **A GPU** for the device: the library runs the guest's Metal calls on the
  host's GPU over Vulkan, so the potluck job holds one, `--gpu 780m` on
  `homelab-um870-1` (its Radeon 780M, on RADV). The script runs the device
  on the GPU `MESA_VK_DEVICE_SELECT` names, else on the one GPU the job
  holds (`POTLUCK_GPUS`), which it then names there; it stops where neither
  names one GPU, since on a host with several GPUs the library takes its
  own pick ([lessons.md](lessons.md)). `homelab-um870-1` also has an RTX
  3090: a run with no lease (job `m3-max-17037`) put the device on it, and
  so did three `--exclusive` runs that took the first GPU of their lease,
  which lists every GPU of the node, the 3090 first (`m3-max-17679`, `17862`,
  `17863`). An `--exclusive` run names its card: `--env
  MESA_VK_DEVICE_SELECT=1002:1900!` for the 780M.
- **The whole machine**, for a run that gates a change: `--exclusive
  --hold-line`, as for any boot whose timings decide its verdict ("Booting a
  guest to check a change", above). Under TCG the emulated guest gets
  through its work at the pace of the CPU its vCPUs get. Beside a build that
  used 34 minutes of CPU in 4 min 18 s (job `m3-max-17260`), 14.6.1's probe
  showed no frame in its 10 s, before `metal-app` waited for its window (job
  `m3-max-17111`). The probe's line gives the host's pressure over its run,
  so a run beside other work shows it.
- **The device's library and our QEMU, built on the node**, in one potluck
  workspace that the boots then reuse (`--workspace`): `cargo xtask dist`
  (the library on Vulkan, `target/dist-vulkan`), then our QEMU at
  `vendor/qemu`'s commit, `bc378624`, configured as above against that dist
  tree but without `--enable-hvf`, for TCG: `--target-list=aarch64-softmmu
  --enable-reims-vgpu --enable-gnutls --disable-docs --disable-bsd-user
  --disable-linux-user --disable-tools`, then `make qemu-system-aarch64`.
  The dispatch leaves out `vendor/qemu/roms/*` and
  `vendor/qemu/tests/lcitool/libvirt-ci` (`--without-submodule`). On
  `homelab-um870-1` the job took 9 min and peaked at 12.0 GiB (`--ram 16`).
- **Copies from the Mac**: Ambrosia's `target/ambrosia.bin` and
  `build/guest-tools`. The node has clang, lld, cpio and python3, so it can
  build the bundle itself.
- **On the node's PATH**: `sshpass`, `python3`, `perl` and GNU `timeout`;
  `homelab-um870-1` has all four.
- **RAM**: `--ram 18`. A boot with the device peaked at 11.8 GiB by `potluck
  why`, page cache included, and at 9.4 GiB of anonymous and shared memory,
  QEMU's own high-water mark 9.5 GiB.
- **Time**: the script's limits are 450 s to SSH and 300 s more for the
  desktop; a boot with the module's bundle alone took 5 min 20 s. The SSH
  login of macOS 26 on `homelab-um870-1`'s 780M comes at one of two times:
  122-132 s (m3-max-32336, 32792) or 301-315 s (32613, 33045, and 33301,
  which failed at the old 300 s limit with the desktop presenting), so the
  limit sits well past the slow one.

**Measured** on `homelab-um870-1` under TCG on 2026-11-08, with our QEMU at
`bc378624` built there as above on the library of the same tree, Ambrosia
0.1.0's released image (`ea2987b`), the module's bundle alone (`make
bundle`), and reflink clones of the guests in `~/.cache/rvg-vms`. Each boot
gave the six checks HVF gives and failed the three that need kexts, as it
must: `kexts` (none of the three loaded), `windowserver-metal` (WindowServer
up with no Dock or Finder after 180 s, `bench -l` listing no Metal device, no
IOSurface shown) and `metal-app` (not run, with no Metal device):
- 26.6.2 (25G83), the guest's RAM plain: `ssh` at 112.1 s. WindowServer
  restarted every 20 to 30 s while the script waited for a desktop, 7 pids
  in 180 s.
- 27.0.1 (26A434), the guest's RAM plain: `ssh` at 101.2 s; WindowServer
  kept its pid.
- 27.0.1, the guest's RAM a shared memfd (the script as it is): `ssh` at
  114.7 s; WindowServer restarted once. Ambrosia found RAM at 4 GiB as
  before (`RAM 0x100000000-0x300000000`), QEMU mapped the guest's RAM from
  its `memory-backend-memfd`, and QEMU still warned that IOSurfaces are
  copied (above). `info mtree -f` listed the address space's RAM as
  `firmware`, `VMApple Config`, the device's region and `ram`.

In each, Ambrosia logged `reims-graphics at 00:05.0: 1920x1080, framebuffer
0x80400000` and started its console there; the screendump showed its log
and the kernel's progress bar. QEMU logged no `reims-vgpu: fatal` line.
Each job peaked at 11.5 to 11.8 GiB by `potluck why`, page cache included,
and at 9.4 to 9.5 GiB of anonymous and shared memory; QEMU's own high-water
mark was 9.5 to 9.6 GiB. Each boot took about 5 min 20 s.
