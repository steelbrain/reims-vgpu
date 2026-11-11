# Running QEMU on the library

`vendor/qemu` is QEMU with `reims-graphics-pci`, the GPU device our own
kexts drive (`guest/`), in C on this library's C API. `make qemu` builds
its `qemu-system-x86_64` on the checkout's library, linked statically
through pkg-config. On Linux it runs x86-64 macOS guests under KVM, and
users run them with `reims-ctl`, whose installer,
`scripts/install-ctl.sh`, builds the same QEMU on the library's Vulkan
platform. This file says how QEMU is built and uses the library, and the
measured facts QEMU's side and the guests rely on.

`make qemu` builds no `qemu-system-aarch64`, and `reims-ctl` makes no arm64
VMs. QEMU's `vmapple` machine, without the Apple paravirtualized GPU it once
had, builds again in an `aarch64-softmmu` build, under HVF or TCG, for
Ambrosia, our own arm64 firmware ("arm64 guests", below).

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
with (`build/qemu/reims-vgpu.pc.configured`). Then it builds
`qemu-system-x86_64`. configure fetches meson subprojects (`dtc`,
`keycodemapdb`, ...) into the source's `subprojects/`, which its own
`.gitignore` covers.

- **The library.** configure records `PKG_CONFIG_PATH` in `config.status`,
  and meson keeps it as its `pkg_config_path`, so ninja's own reconfigures
  find the package too. The `reims_vgpu` dependency (`--enable-reims-vgpu`)
  is `dependency('reims-vgpu', method: 'pkg-config', static: true)`, looked
  for on Linux and macOS hosts, so QEMU links `libreims_vgpu.a` and what its
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
  device and checks it paused ("The reims-graphics device", below).

## What vendor/qemu carries

`vendor/qemu` is upstream QEMU with these commits on top, on branch
`steelbrain/reims-vgpu`. The branch started from upstream's `b833716681` (a
merge into staging of 2026-06-19), and upstream master is merged into it
(`049fab62bb`, then `49873ce25e`, at `9c88811b81` of 2026-09-30).

| what | commits |
|---|---|
| the reims-graphics device's transport, for the library's own guest driver: `reims-graphics-pci` (1234:5247, class 0x0380), on the library's graphics device (`rvg_graphics_device_*`); BAR 0 its registers with MSI-X from 0x800, the doorbell an ioeventfd, BAR 2 the library's region as RAM; built for every target where the library is found (`CONFIG_REIMS_GRAPHICS`), x86-64 under TCG on a Mac among them ("The reims-graphics device", below) | `77ee6c6b`, `bf69f959` |
| the device gives the library the guest RAM's map and the files behind it | `abb94a98` |
| the device tells the library the guest's architecture (`guest_arch`: arm64 where `target_aarch64()`, else x86-64), which the device's `isFramebufferReadSupported` answer follows | `c2fb9b92` |
| the device raises its interrupt through an irqfd where KVM's in-kernel irqchip takes MSI-X through one | `36deb281` |
| a command buffer is taken, and answered, on the vCPU that rings its queue's doorbell, with the plain register write | `f2c7e2a5`, `67cc08ad`, `5f459785` |
| the console shows the display (`rvg_graphics_device_read_display`), so QMP's `screendump` saves the guest's frame, and has the guest's cursor | `70a9a466`, `c119b802` |
| the device's realize warns where BAR 2 would get no address: on a PC machine with RAM at 1 GiB ("The reims-graphics device", below) | `22ae1fd1` |
| the window opens at the `resolution` property's size, 1920x1080 without it | `93fb0692` |
| file-backed guest RAM (the memfd of `reims-ctl`'s VMs) mapped 2 MiB-aligned on Linux, as anonymous RAM is, so transparent huge pages can back it where the host allows them ("Linux hosts: huge pages for guest RAM", below) | `1b41f5f5db` |
| `isa-applesmc` answers GET_KEY_BY_INDEX, listing its keys by name and 0xb8 past the last, so the key walk macOS 15.4 and later make at boot ends ("x86-64 macOS guests: the SMC", below) | `8d795445f2` |
| QMP's `screendump` in PPM, its default format, writes rows of width × 3 bytes. Upstream's wrote each row with its line buffer's stride, which pixman rounds up to 4 bytes, so a display whose width is not a multiple of 4 (a window of 3662 × 2039 pixels) read back sheared, its colours rotating from row to row; PNG's rows were right. `tests/qtest/display-screendump-test.c` dumps `bochs-display` at widths of 64, 66 and 67 | `23198e4b79` |
| the input layer's smooth `scroll` events (`qemu_input_queue_scroll`, `qemu_input_scrolls_smoothly`, QMP `input-send-event`); a device without them (`usb-tablet`, which every x86-64 VM has) gets the event's wheel notches | `840feb5ced` |
| the thread pool keeps at least one thread when a job finished before `thread_pool_submit_immediate` resized it: a backport of upstream's `2566b6e902`, which the merge of upstream master brings too | `850b5a4bb4` |
| QMP's `x-query-virtio-status` and HMP's `info virtio-status` no longer crash on a virtio device: a patch from upstream's list (Vladimir Sementsov-Ogievskiy) that fixes upstream's `1ee0bc0c9bf0` | `d151bcd9` |
| `vmapple` ("arm64 guests", below): patches 1 to 6 of the RFC v7 series "vmapple: make it work on the latest macOS release" (Mohamed Mediouni), and the branch's commits for HVF's ISV=0 data aborts and in-kernel vGIC, the secondary CPUs' pointer authentication keys, macOS 14 and later and macOS 27 guests, the AES engine's refusal without a crypto library, and `usb-vmapple-tablet`, Virtualization.framework's USB pointing device | `e181197cc2`, `663b9b9c0b`, `4a2e2b3967`, `450e03eed2`, `f116a9d560`, `c912618391`, `c2f5832f32`, `674af72fe2`, `da640879f0`, `e0ce93da92`, `d46ebaa252`, `61f7de5fc1`, `fc46f8bf02`, `a24c8eadfa`, `e1c37ea4d4`, `d235e51310`, `fbda9c8b28`, `c7a4f0700f` |
| apple-gfx, Apple's paravirtualized GPU device on the library, with its windows, its option ROM, its IOSurface device and the VideoToolbox and M2 scaler devices, removed: `vmapple`, whose display was `apple-gfx-mmio`, stayed in the tree unbuilt until the next row's commits; the option that finds the library is `reims_vgpu` (`--enable-reims-vgpu`) | `eb36a099` |
| `vmapple` for Ambrosia, built by default in an `aarch64-softmmu` build again: the machine without apple-gfx's display; no kernel-entry hook (the firmware makes macOS's kernel-entry changes, so AVPBooter boots only macOS 12 and 13 guests); RAM at 4 GiB, with the PCIe MMIO window from 0x50000000 below it for 1 GiB BARs (AVPBooter does not boot on this layout); TCG with the `max` CPU and a 24 MHz generic timer; the guest's write to its DMA configuration doorbell at 0x4ffff8 ignored, which panicked a macOS 27.0.1 kernel under TCG; and `virtio-net-pci` without its control queue in the machine's defaults, since one boot in four to six of a macOS 26 or 27 guest missed the completion of its first control command and never started DHCP | `15989559`, `cd2ecf44`, `bd408aea`, `a52f4a20`, `58a6f887`, `c7909d60` |
| BAR 2 is one RAM region with no dirty log and no `scanout_written` callback: the library's window draws the scanout again at each refresh while it shows no presented surface | `7aee20f4` |
| `vmapple`'s second clock, `vmapple-rtc` at 0x30240000 as Virtualization.framework has it: a 64-bit microsecond count from the Unix epoch that follows `-rtc`, an ID register reading 0xf001 and no interrupt, beside the PL031. macOS 14 and later keep their RTC offset in it; without it each boot logs "RTC reset likely" and steps the clock back to the disk's last write until a time server answers. macOS 12 and 13 have no driver for it | `aef8a7cf` |
| `vmapple`'s `max` CPU offers no SME by default under TCG (an optional compat default: under HVF the CPU gets its `sme` property only as its vCPU starts, after the default would apply, so HVF's CPU keeps what the host gives; `-cpu max,sme=on` turns it back on): TCG's streaming vectors reach 2048 bits, and a macOS 14.6.1 kernel sizes each thread's SME state zone from the longest, 74328 bytes, and panics with "zone_create: element size too large". 15.6.1, 26.6.2 and 27.0.1 boot either way; an M1 to M3 under HVF offers no SME | `d2bd3447`, `634c3b8f` |
| `vmapple` under KVM on an arm64 Linux host: KVM passes the guest's PSCI CPU_ON calls to QEMU (an SMCCC filter, `KVM_ARM_VM_SMCCC_FILTER`, Linux 6.4 and later; at `-smp` above 1 a kernel without it is fatal), which starts the CPU through `arm_set_cpu_on()` as under HVF and TCG, so its secondary CPUs take the boot CPU's pointer authentication keys (`cpu-on-copies-pauth-keys`) instead of KVM's reset values; and its CPUs get no PMU (vmapple initializes none, and KVM runs no vCPU with one uninitialized) and PSCI 1.1, which macOS's ApplePSCI takes where it refuses the 1.3 current kernels give. Tested by the Asahi Linux experiment on an M2 Pro (Linux 6.19.14, macOS 27.2.1, `-smp 4`); under HVF and TCG nothing changes | `570daa78`, `da466bd8`, `5093524e` |
| each HVF vCPU thread runs at the user-initiated QoS, above the default QoS a host build runs at and below the host's user-interactive work, so a guest keeps its share of the CPU beside a build (HVF is built on macOS alone) | `5658f96c` |
| `vmapple` under KVM answers macOS's pointer authentication calls (SMCCC 0xC1000000 to 0xC10000FF) as HVF and TCG do: KVM passes them to QEMU, which writes NOT_SUPPORTED to x0 alone, where KVM's own answer also zeroes x1 to x3 and XNU keeps x3 of GET_DEFAULT_KEYS (launchd then died of a pointer authentication failure); and KVM's data aborts with no instruction syndrome (pair, writeback and SIMD&FP accesses, `KVM_EXIT_ARM_NISV`) are emulated by the ISV=0 emulator HVF used, now `target/arm/isv0-mmio.c`, in a strict mode that changes no register and leaves the external abort where QEMU has no device at the address (Ambrosia's ECAM read `ldr w9, [x8, #16]!` was the first). HVF calls the moved emulator as before. Tested by the Asahi Linux experiment on an M2 Pro (macOS 27.2.1 under KVM to check-boot's ssh, 12 of 14) | `bd5ac1ce`, `4dca14b4`, `89e32906`, `3892a92f` |

The branch's other commits are the apple-gfx work `eb36a099` removes, and
`cbb749d421`, an HVF workaround that `481e881b5c` reverts, upstream having
fixed the same assertion, and `75723e48` and `bc378624`, which correct
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
  a Mac), whose file the library's control thread polls. Every other
  register write goes to the library through
  `rvg_graphics_device_mmio_write_taking`: a write of a command queue's ring
  may answer the queue's next command buffer on the vCPU, posting its
  `CB_SUBMITTED` and raising the interrupt inside the call, and wakes the
  queue's thread, which takes the buffer either way.
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
  and MSI-X is enabled, and KVM injects it on the raising thread; elsewhere
  (HVF, TCG) a BH sends it under the BQL. Neither waits for a lock.
- Under `-display none` the library opens a window for the display, unless
  `REIMS_VGPU_WINDOW` is `0`. It opens at the `resolution` property's size,
  `<width>x<height>` (1920x1080 without it), a guest pixel per host point,
  shrunk to fit the screen; then the window keeps the size the person gives
  it, and the display offers the guest that size in pixels at the host
  screen's rate. On a KDE Wayland desktop of 3840x2160 at 2.1x, 1829x1029
  points, `resolution=1280x720` opened the window at 1280x720 points, and a
  macOS 26.7.2 guest ran 2688x1512; `2560x1440` and the default opened it
  shrunk to 1744x981 points, the guest at 3662x2039 (**measured**). Its
  input is queued for a BH and reaches the guest's tablet and keyboard
  through the device's console. Closing the window powers the VM off.
- **The console** shows what the display shows, with a window or without
  one: at each of its updates (a display change listener's refresh, or a
  screendump) a thread of the device's pool reads it back from the library
  (`rvg_graphics_device_read_display`), the scanout or the IOSurface the
  guest presents, through the display's colour, at the display's mode, and
  a BH copies it into the console's surface and says the update is done. So
  QMP's `screendump` saves the guest's frame. One read runs at a time; an
  update that comes while one is under way is done with it. The read waits
  for its draw on the GPU, never under the BQL, and the guest sees nothing
  of it.
- **The cursor** is the console's too: the library tells the device the
  guest's cursor on its display thread (`cursor_glyph`, `cursor_show`,
  `cursor_move` of `RvgGraphicsCallbacks`), and the BH, in order with the
  window's input, gives the console its image and hot spot
  (`qemu_console_set_cursor`) and its place and whether it shows
  (`qemu_console_set_mouse`). The library gives the hot spot's place
  (`rvg_graphics_device_cursor_position`), and the console takes the image's
  top-left corner, as virtio-gpu gives it, so the device takes the hot spot
  off. QEMU's own displays (SPICE, VNC, GTK, D-Bus) draw the cursor from it.
  The image the console reads back has no
  cursor, and under `-display none` the console has no display: the window
  shows the cursor as the host's pointer.
- It has no default option ROM: `romfile=` names one.
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
    `crates/reims-vgpu-vulkan/src/os/linux/host_ram.rs`). An object whose
    pages are one run of the RAM inside one 64 MiB chunk of it is bound in
    the device's import of that chunk, made the first time an object over it
    is made: it takes no mapping and asks no gate. A device without
    `VK_EXT_external_memory_host` is not handed the RAM, said once in the
    log, and its objects take the ways below.
  - Views are mapped from their files while they fit 8,192 mappings short of
    the budget (`budget.rs`, `HEADROOM`), and through dma-bufs past that. Any
    other import of a view's pages (a no-copy buffer's, a heap's or a
    rebind's) maps them from their files while they fit short of those 8,192
    mappings too, as does the writable twin an import of read-only pages
    needs. Past that line the library's import gate refuses the pages, and
    the runtime makes the object over a copy of them, which takes no mapping
    (metal2vulkan's `HostMemory::with_import_gate`, announced
    `copied-host-pages`; with `REIMS_VGPU_STRICT=1` it ends the process). The
    copy lies in host-cached system memory, compared whole on the CPU at each
    commit and each completion, and it reads the pages with the CPU, which
    reads pages left in views through dma-bufs too.
- **Before anything would die or be refused at the whole budget**, the
  library folds back into dma-bufs the views' pages mapped from their files
  that no live import a device may pin holds, and weighs again (`vm.rs`,
  "Folding back"). Only where that frees too little is a view or an import
  refused. A buffer's or heap's import short of the headroom folds the same
  way before it is copied, where the views grew or an import was freed since
  the last such fold. A map the kernel itself refuses at `vm.max_map_count`,
  when the rest of the process holds more than the 16,384 kept for it, folds
  the same way and is tried once more.
- **What is refused for want of mappings** answers as the same call does on
  Metal when memory runs out: a remap of guest pages, or a range
  a view is made of, answers `KERN_NO_SPACE` with nothing mapped, an
  IOSurface stand-in's memory is none, an import's twin is none, and a buffer
  over a surface or a heap over address ranges is nil. Each is counted and
  logged at the first and at each power of two. A replace whose alias is
  refused still ends the process. With `REIMS_VGPU_STRICT=1` each ends the
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
  power of two.
- On a device that needs pages mapped from their files, an information line
  if imports of all of the guest's RAM, a mapping per 4 KiB page, would come
  within the headroom of the budget, with the `sysctl -w
  vm.max_map_count=...` value that covers them (2^23, 8,388,608, for a 16 GiB
  guest): past that, buffers and heaps not bound in a chunk are copied.
- Each buffer, heap or rebind copied for want of mappings is counted with
  its bytes, logged at the first and at each power of two, and written to
  the trace (`nrcopy` for pages left in dma-bufs, `notwin` for read-only
  pages with no twin: first byte, length, mappings it would have taken).
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

reims-ctl gives every VM a virtio sound device, with one output and one
input stream:

```sh
-audiodev pipewire,id=snd0 -device virtio-sound-pci,audiodev=snd0,addr=0x7
```

macOS's `AppleVirtIO` kext has `AppleVirtIOSound`, matching virtio device
0x19. **Measured**
with a macOS 26.7.2 (25G332) x86-64 guest on reims-ctl's q35 command, on a
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


`pa` in place of `pipewire` uses PulseAudio, and `none` gives the guest a
working output whose samples QEMU drops. macOS 26 for x86-64 ships no
Intel HDA driver (no `AppleHDA`; 26.7.2 loads no HDA kext), so `-device
ich9-intel-hda -device hda-duplex` leaves the guest with no audio device;
reims-ctl's x86-64 VMs have `ich9-intel-hda` with an `hda-output` for
OpenCore's boot chime alone, which plays on the same audiodev: the 26.7.2
guest's `wav` capture starts with 2.5 s of it. `AppleUSBAudio`, which
macOS does ship, drives a USB audio device on the xHCI controller above
(`-device usb-audio,audiodev=snd0,bus=xhci.0`), output only.

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
(0x11) among them. Upstream's device, at the master vendor/qemu merged
(`9c88811b81`), answers READ alone, and 0x82 to the rest (read from code).
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
`qemu-system-aarch64`, and `reims-ctl` makes no arm64 VMs. `vmapple` builds
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

### arm64 guests: the boot check of our configuration

Ambrosia's done list has 10 boots of our configuration: macOS 13.6, 14.6.1,
15.6.1, 26.6.2 and 27.0.1, each under HVF on the M3 Max and under TCG on a
Linux member. The boot check covers four of them, macOS 14.6.1, 15.6.1,
26.6.2 and 27.0.1: arm64 macOS 11 to 13 are not supported for now (owner,
2026-11-09), since macOS 13.6 offers the device no MSI and the kexts have no
INTx path, and the arm64e set is placed from Darwin 23. Our configuration is the module's early console with Graft and
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
  `Graphics` > `DisplayNode` set to `reims-display0`, and the kexts under
  `Kexts/`. It checks each kext's `CFBundleIdentifier`
  (`ai.aneesiqbal.graft`, `ai.aneesiqbal.ReimsGraphics`,
  `ai.aneesiqbal.RgfxGate`) and that its executable has an arm64e slice,
  and names every kext that is not given or fails, writing no bundle.
- **The Ambrosia it needs.** `Graphics` > `DisplayNode` needs Ambrosia
  `8f2655b` or later (Ambrosia's `docs/configuration.md`, "Graphics >
  DisplayNode"). Where the module drives a display, Ambrosia adds
  `/arm-io/reims-display0`, a `display-subsystem` node, to the device tree
  before the kernel starts, and logs `ambrosia: display node
  /arm-io/reims-display0, a display-subsystem`; ReimsGraphics uses that node
  as display 0's ([architecture.md](architecture.md) 11.5). An older
  Ambrosia ignores the key, as it ignores every key it does not know, and
  boots with no node: ReimsGraphics then attaches its own node when it
  starts, after WindowServer has counted the displays, so WindowServer
  counts none and stays headless, while the processes that start later find
  the display.
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
  within 90 s of QEMU's start under HVF, 300 s under TCG. `sip`, `boot-args`:
  `csrutil status` enabled and `kern.bootargs` empty. `kexts`: `kmutil
  showloaded` lists the three identifiers. `windowserver-metal`: the lane
  gate's reading of it, the desktop's processes up, `bench -l` listing a
  Metal device that is not Apple's paravirtual one, WindowServer's pid kept,
  and the host's log showing an IOSurface the guest presented.
  `metal-app`: `present_probe` full-screen in the user's session, under
  `report.py`'s bar; under TCG it passes on frames presented with at most 3
  answers missing, and its drops are printed but not judged, as the emulated
  CPU sets WindowServer's composite rate (pacing is judged under HVF only).
  Its line also gives the host's display flips during the probe,
  WindowServer's own present rate. A probe that falls short while Setup
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
  matrix` line from WindowServer. `amfi-refuse`: last of the guest's checks,
  a copy of `/usr/bin/true` signed ad hoc with `com.apple.rootless.install`
  is killed by AMFI (exit 137), its refusal passing through RgfxGate's
  signature hook, and 5 s later `kern.bootsessionuuid` is still the
  login's. `no-fatal`: no `reims-vgpu: fatal` line in QEMU's log or the
  serial log. A reading in the guest that printed nothing fails its check as
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

**A TCG run** (`-t`) runs on a Linux member. What it needs:
- **A node** with the arm64 guests: `homelab-um870-1` holds reflink copies of
  all five in `~/.cache/rvg-vms` (`macos-13-arm64` to `macos-27-0-arm64`,
  each with its `vm.toml`). The script's log directory (`-o`) must be on that
  filesystem, so that its clones are reflinks. Another member needs copies of
  the guests first.
- **A GPU** for the device: the library runs the guest's Metal calls on the
  host's GPU over Vulkan, so the potluck job holds one, `--gpu 780m` on
  `homelab-um870-1` (its Radeon 780M, on RADV).
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
- **Time**: the script's limits are 300 s to SSH and 300 s more for the
  desktop; a boot with the module's bundle alone took 5 min 20 s.

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
