# Running QEMU on the library

`vendor/qemu` is QEMU with `apple-gfx` in C on this library's C API, and
with the `vmapple` machine working on a macOS 26 host. `make qemu` builds
its `qemu-system-aarch64` on the checkout's library, linked statically
through pkg-config, and `make vm` boots a macOS guest on it. On Linux the
same `apple-gfx` builds for KVM, where x86-64 macOS guests drive
`apple-gfx-pci`. This file says how QEMU is built and uses the library, how
a VM is run and driven, and the measured facts QEMU's side relies on.

The suite never runs QEMU, so a VM boot is the only check of QEMU's side of
the API ([lessons.md](lessons.md)). Each measurement below names its host
and its date or commit. Anything not measured is marked **unknown**.

## The QEMU build

```sh
make qemu          # cargo xtask dist, then build/qemu/qemu-system-aarch64
```

`make qemu` runs `cargo xtask dist`, which writes the library into
`<target>/dist` (`target/dist`, or under `CARGO_TARGET_DIR`;
[architecture.md](architecture.md) 11.3), then configures `vendor/qemu` out
of tree in `build/qemu`:

```sh
PKG_CONFIG_PATH=$PWD/target/dist/lib/pkgconfig vendor/qemu/configure \
  --target-list=aarch64-softmmu --enable-hvf --enable-pvg --disable-docs
```

It configures once, and again whenever `QEMU_CONFIGURE` (extra configure
options) is set, or `config.status` came from another source or another
`PKG_CONFIG_PATH`, or names `--apple-gfx-backend` or `--pvg-framework`,
which vendor/qemu's configure does not take, or the dist tree's
`reims-vgpu.pc` differs from the copy it keeps of the one it configured with
(`build/qemu/reims-vgpu.pc.configured`). Then it builds
`qemu-system-aarch64`, which QEMU's build signs ad hoc with
`accel/hvf/entitlements.plist` (`com.apple.security.hypervisor`) and no
hardened runtime. configure fetches meson subprojects (`dtc`,
`keycodemapdb`, ...) into the source's `subprojects/`, which its own
`.gitignore` covers.

- **The library.** configure records `PKG_CONFIG_PATH` in `config.status`,
  and meson keeps it as its `pkg_config_path`, so ninja's own reconfigures
  find the package too. The `pvg` dependency (`--enable-pvg`) is
  `dependency('reims-vgpu', method: 'pkg-config', static: true)`, so QEMU
  links `libreims_vgpu.a` and the system frameworks its `Libs.private`
  names, and records no path to the library: dyld loads nothing of it.
  meson warns that `-lSystem`, `-liconv` and `-lobjc` have no static
  archive; they stay the system's dylibs.
- **The package's flags are configure's.** meson reads `reims-vgpu.pc` at
  configure and keeps what it read. A QEMU build configured before the
  library came to need Carbon (the window's ISO keyboard check) relinked the
  new archive without it, and failed on `KBGetLayoutType` and
  `LMGetKbdType`; hence the reconfigure when the package changes
  ([lessons.md](lessons.md)).
- **The option ROM.** meson installs the file the package's `optionrom`
  variable names into QEMU's data directory as `ReimsVgpu.rom`,
  `apple-gfx-pci`'s default `romfile`, and configure links it into
  `build/qemu/qemu-bundle`, where a QEMU run from the build tree finds it.
  `make qemu` fails if it is not there. A package that names no such file
  gets a warning from meson, and `apple-gfx-pci` then needs `romfile=` to
  realize.
- **The machines.** `hw/display/meson.build` builds `apple-gfx.c` with
  `apple-gfx-pci.c` (`CONFIG_MAC_PVG_PCI`) or with `apple-gfx-mmio.c`
  (`CONFIG_MAC_PVG_MMIO`, Arm targets only). `hw/vmapple/vmapple.c`
  creates `apple-gfx-mmio`, and the machine depends on `MAC_PVG`
  (`hw/vmapple/Kconfig`), so a build without `apple-gfx` leaves it out.
- **Linux hosts.** configure looks for the package on Linux too, and sets
  `MAC_PVG` for the targets KVM runs, as it does for HVF's on macOS: on an
  x86-64 host that is `apple-gfx-pci`. `apple-gfx-activity.m` is macOS's;
  Linux builds its stub, `apple-gfx-activity-stub.c`.

A predecessor's QEMU, for a pair of boots ("Running the vmapple VM on
reims-vgpu", below), is `make qemu` in a worktree of its own. That worktree
gets its own clone of `vendor/qemu`, never a `git submodule update` of the
main checkout's module ([lessons.md](lessons.md)).

## What vendor/qemu carries

`vendor/qemu` is upstream QEMU with these commits on top, on branch
`steelbrain/reims-vgpu`. The branch started from upstream's `b833716681` (a
merge into staging of 2026-06-19), and upstream master at `9c88811b81`
(2026-09-30) is merged into it (`49873ce25e`).

| what | commits |
|---|---|
| `vmapple` on a macOS 26 host: patches 1 to 6 of the RFC v7 series "vmapple: make it work on the latest macOS release" (Mohamed Mediouni); patch 7, which needs SIP disabled, is left out | `e181197cc2` |
| HVF: emulate the data aborts that report ISV=0 (load and store pairs, SIMD&FP, indexed loads and stores), which macOS 13 and later guests make at MMIO during early boot | `663b9b9c0b` |
| HVF: place the in-kernel vGIC at the machine's own GIC bases, so `kernel-irqchip=on` works for `vmapple` | `4a2e2b3967` |
| `vmapple`'s secondary CPUs start with the pointer authentication keys of the CPU that powers them on (an opt-in Arm CPU property, `cpu-on-copies-pauth-keys`, which `vmapple` sets), so a macOS guest runs on every vCPU ("Every vCPU of the guest", below) | `f116a9d560`, `c912618391` |
| `vmapple` depends on `MAC_PVG`, so it is built only with `apple-gfx` | `f7a1765826` |
| `apple-gfx` in C on the library's C API (`hw/display/apple-gfx.c`, `apple-gfx-mmio.c`, `apple-gfx-pci.c`), found through pkg-config | `acdcc169e9`, `e4fedf2708` |
| `apple-gfx-pci`'s default `romfile`, `ReimsVgpu.rom`, installed from the package's `optionrom` | `08d0d0aca7`, `27c55651b3` |
| `apple-gfx`'s `displays` property: up to 8 displays a device, one console each; the cursor callbacks' work through one ordered queue and one BH | `b8842e1a50` |
| window mode: under `-display none`, each display gets a window of the library's, whose input reaches the guest's USB tablet and keyboard; closing it powers the VM off ("Window mode", below); the guest's mode 1920x1080 by default; `resolution=<w>x<h>@<ppi>` tells the guest that density for the mode, and a density at which the guest's scale is not known from it, or a width told over 65535 mm, fails the realize ("A chosen density", below) | `54fd32d1c3`, `4ec8f35655`, `ffb8ae8b17`, `c720eb298d`, `e4228ee079`, `c1e0d866ba`, `1dc0b093c9`, `ffbf1f3cee` |
| smooth scrolling: the input layer's `scroll` events (`qemu_input_queue_scroll`, `qemu_input_scrolls_smoothly`, QMP `input-send-event`); `usb-vmapple-tablet`, Virtualization.framework's USB pointing device, which a macOS guest scrolls as a trackpad; `vmapple`'s `tablet` property, `usb-vmapple-tablet` by default; the windows' scroll callback sent as smooth scroll events, or as notches to a device without them ("Measured facts about driving it", below) | `840feb5ced`, `c2f5832f32`, `674af72fe2`, `3dd8c62445`, `da640879f0`, `e0ce93da92`, `d46ebaa252`, `61f7de5fc1`, `fc46f8bf02`, `797c366c0f` |
| vmapple boots macOS 14 and later guests, with no option: when iBoot enters the kernel, which the machine sees through `hvf_arm_exec_fault`, a hook HVF calls on an instruction fetch's stage 2 fault, it takes the kernel's waits on failed pointer authentication hypercalls out of its code, and adds `/filesystems` with `DoNotUseSEP` to a device tree that has no `/filesystems` ("macOS 14 and later guests", below); `vmapple.rst`'s "Guest versions" | `a24c8eadfa`, `e1c37ea4d4` |
| `apple-gfx` names each display "Reims vGPU" to the guest, which saw "QEMU display" before | `3ca83dca38` |
| `apple-gfx` holds a user-initiated activity from its realize to its finalize, so App Nap never slows a VM whose window is hidden or behind another application's (`hw/display/apple-gfx-activity.m`; "App Nap", below) | `1e9e1915cb` |
| `apple-gfx`'s MMIO accesses call the library on the vCPU's thread, with no hop to the device's pool, and the GPU device's region takes them without the BQL under KVM ("MMIO", below) | `93afbf1f1e`, `8d81cebd5f`, `2eec63f528` |
| `apple-gfx` builds on Linux hosts, for KVM: each range of the device's memory map has its own host address (q35's RAM above 4 GiB is an alias partway into its block) and names the file behind it, which the realize requires on Linux; the trace lists each range ("apple-gfx on reims-vgpu", below) | `c312e13367`, `2b8150f1b8`, `3ad617fb01`, `c83a31e78a` |
| `apple-gfx` reads each frame's pixels on the pool thread, and renders the console's surface only while a display change listener shows it, or when the display system asks ("Frames", below) | `06437bde6f`, `add4929158` |
| `apple-gfx-pci`'s BAR 2, a window onto the EFI framebuffer: one alias of the machine's RAM, moved as the guest writes the framebuffer's start, so that a macOS installer or recovery image, which has no driver of the device's own, takes the boot framebuffer over (the `efi-framebuffer-bar` property, on by default; "BAR 2" in "apple-gfx on reims-vgpu", and "x86-64 macOS guests: installers and recovery", below) | `89fc1f56ba`, `d6710a2850`, `440da86693` |
| `isa-applesmc` answers GET_KEY_BY_INDEX, listing its keys by name and 0xb8 past the last, so the key walk macOS 15.4 and later make at boot ends ("x86-64 macOS guests: the SMC", below) | `8d795445f2` |
| vmapple boots macOS 27 guests: `vmapple-virtio-blk-pci` links its capabilities in address order and carries Apple's vendor capability, and `virtio-net-pci` links them in address order with `x-ascending-capabilities`, which `vmapple` turns on (Jesus Martínez, co-author; "macOS 27 guests", below) | `d235e51310`, `fbda9c8b28` |
| the IOSurface device over PCI for x86-64 guests that run Apple's arm64 GPU driver: `apple-gfx-iosurface-pci` (106b:eeef, ours), whose presence is the only switch: `apple-gfx-pci` isolates a task, and offers argument buffers, only while the guest's mapper has a descriptor table, and has no option for either; and a reset of the IOSurface device's registers ("The IOSurface device over PCI", below) | `bd95509a` |
| the VideoToolbox device's transport for macOS guests' AppleVideoToolboxParavirtualization: `apple-videotoolbox-pci` (106b:1a03, Apple's IDs; subsystem 106b:1a03 and class 0x120000 ours), modern virtio, two queues, on the library's video device (`rvg_video_device_*`), its IOSurfaces resolved through the process's latest IOSurface device, with no properties: the library picks what it decodes and encodes with, on the GPU its GPU device runs on; the guest-to-host queue holds 1024 descriptors, where Apple's hold 256, so that an x86-64 guest's largest message fits; a readable part over 16 MiB is answered -12904 before it is copied; each of its elements answered on a worker thread of the device's, in the guest's order, each status before the messages sent while it was answered ("The VideoToolbox device", below) | `ff2ac115`, `f7fdf62dab` |
| the M2 scaler device's transport for macOS guests' AppleM2ScalerParavirtDriver: `apple-m2scaler-pci` (106b:1a0b, subsystem 106b:1a0b and class 0x1200, Apple's), modern virtio, one queue, on the library's scaler device (`rvg_scaler_device_*`), the guest's memory reached through the device's DMA address space, with no properties, as the VideoToolbox device; realized on every host, it refuses the guest's features where the host has no GPU for it; a readable part over 64 KiB is a needs-reset before it is copied; each element answered on a worker thread of the device's, in the guest's order ("The M2 scaler device", below) | `ff2ac115` |
| comments, coding style and dead code, with no functional change: the window's wheel callback, which the library never calls beside `scroll`, is gone; configure's summary says whether it found reims-vgpu on Linux too | `f3148adebd`, `e4ca06d9f7`, `5b13eebbf1`, `ef73c3c6bd`, `f624140283`, `d25256b8b0`, `6855e95b7a`, `56a956fbec` |

QEMU does not release the device at its exit: the release waits for the
guest's FIFO work, which never completes once the vCPUs have stopped, so
with that release (`6a039ed39e`, reverted by `20e1b3ca18`) a QMP `quit` of a
running guest did not exit.

`docs/system/arm/vmapple.rst` says how to build QEMU with `apple-gfx` on
the library, and describes the `displays` property and window mode.

**Measured** on 2026-10-11 on this host (an Apple M3 Max, macOS 26.7,
25G229), with no guest booted, at `56a956fbec` built with
`--enable-werror`:
- `otool -L` lists no reims-vgpu library and no `LC_RPATH` to the dist
  tree, and QEMU's binary defines the `rvg_` functions.
- Paused (`-S`, no guest code run) and headless (`REIMS_VGPU_WINDOW=0`),
  `vmapple` with 8 vCPUs and blank aux and root images realizes
  `apple-gfx-mmio`, its memory map one range (0x70000000, 4 GiB, no file);
  HMP `xp` reads the GPU device's registers and the IOSurface registers
  0x30211000 and 0x30211008 as 0, `system_reset` resets, and QMP `quit`
  exits 0.
- Paused, `virt` with two `apple-gfx-pci` devices, one with `displays=2`
  and `resolution=2560x1600@148`, the other with two `display-modes`,
  realizes both as 106b:eeee with BAR 0 and the ROM's BAR 6 and romfile
  `ReimsVgpu.rom` (5,632 bytes, built from `firmware/`), resets, and exits
  0 at `quit`. `displays=9`, `resolution=12x` and `resolution=1280x800@220`
  each fail the realize with its message.

At `440da86693` (BAR 2), on 2026-10-12, paused: on this host, `virt` with
`apple-gfx-pci` realizes it with BAR 0 (16 KiB), BAR 2 (64 MiB, 32-bit,
prefetchable) and the ROM's BAR 6 (8 KiB), and with `efi-framebuffer-bar=off`
without BAR 2; each exits 0 at `quit`. The same holds for `q35` under KVM on
the RTX 5080 Laptop GPU host. Under `-accel qtest` (no guest code: the test
writes the PCI configuration and the device's registers itself), on `virt`, in
16 KiB pages: a write to 0x1210 alone makes BAR 2 show 64 MiB of the RAM
there, and a write through BAR 2 lands in that RAM; a start inside BAR 2 moves
the window on in RAM, and one 64 KiB before the end of RAM gives a 64 KiB
window; a start at the UART, in the flash, past the end of RAM or of 0, and
any start under NUMA, leave BAR 2 empty; a 0 to 0x1200 keeps the window, and
`system_reset` empties it. On `q35`, in 4 KiB pages, the window ends where
TSEG or the SMBASE window begins once the test turns them on as firmware does,
and a start inside either leaves BAR 2 empty, as does a start of 0, which is
RAM there.

`scripts/internal/qtest_iosurface_pci.py` makes 27 checks against
`qemu-system-x86_64` built with `--enable-pvg` (`q35`, `-accel qtest`, 5 GiB
of `memory-backend-memfd` RAM, no guest code; the script writes the PCI
configuration and the registers itself). On 2026-10-16, on an AMD Ryzen 7
7840HS host (CachyOS, Linux 7.2.9, Mesa 26.2.4), with the library's device
on lavapipe (llvmpipe, LLVM 23.1.1), an earlier form of the script passed
the 21 checks this one keeps:
- the mapper's realize fails for a second instance, with its message;
- `apple-gfx-pci` is 106b:eeee in its configuration and in its ROM's PCI
  data structure, read through BAR 6;
- the mapper is 106b:eeef, class 0x0880, with no interrupt pin, an MSI
  capability, and BAR 0 alone (64 KiB of 32-bit memory; no other BAR, no
  ROM); its command register takes memory space and bus mastering;
- 0x1028 reads 0 on this 4 KiB host; an 8-byte read past the registers
  reads all ones; ring and descriptor-table bases above 4 GiB, written in 8
  bytes, read back whole, and a 4-byte read gives the low half;
- with a 256-command ring above 4 GiB, an unknown command completes with
  `kIOReturnUnsupported` (0xe00002c7) and advances 0x1020, and each
  completion writes the MSI's data at its address while MSI and bus
  mastering are on, and nothing while either is off.

The other six came with later changes and have not run: the mapper's
realize without a `gpu` link beside the machine's only `apple-gfx-pci`, and
its refusals, each with its message, when there is none or more than one;
that `apple-gfx-pci` has no `use-iosurface-mapper` or `process-isolation`
property; and that a machine reset (0xcf9) returns the mapper's ring base,
table base, descriptor enable and counters to 0 and leaves its ring size.

The same day `apple-gfx-pci.c` compiled without a warning under Apple clang
with `-Werror`, in a macOS 27 build's `aarch64-softmmu` configuration, and
on 2026-10-17 so did its form at `bd95509a`, against this tree's header; no
QEMU for this Mac was built from either.

## The VideoToolbox device

`apple-videotoolbox-pci` (`hw/display/apple-videotoolbox-pci.c`) is the
transport of the paravirtualized VideoToolbox device, 106b:1a03, which a
macOS guest's AppleVideoToolboxParavirtualization drives; the library's
video device (`rvg_video_device_*`, `crates/reims-vgpu/src/video.rs`) is
its host side, whose session is reims-vgpu-video-core's host session
(`crates/reims-vgpu-video-core`). The device has no properties: the
library picks what it decodes and encodes with
([video/backends.md](video/backends.md), "Picking a backend"). On macOS
that is VideoToolbox. On Linux it is the GPU the library's GPU device
(`apple-gfx-pci`) runs on, the one the Vulkan loader's environment picks:
NVIDIA's NVDEC and NVENC on an NVIDIA GPU where CUDA opens it, else VA-API
on its render node, and the host's first hardware GPU that offers video,
discrete ones first, where that GPU is a software rasterizer or offers
none. The library logs its choice and why; where no backend opens, the
guest is offered no codec and decodes in software. A VM takes it as
`-device apple-videotoolbox-pci`; reims-ctl gives every VM on reims-vgpu
one, an x86-64 VM's at slot 8.
Queue 0's elements go to `rvg_video_device_deliver`, gathered across
descriptors, and its status is scattered into the writable part; one whose
readable part is over 16 MiB (`RVG_VIDEO_MAX_MESSAGE_SIZE`) is answered
`RVG_VIDEO_ALLOCATION_FAILED`, -12904, from its descriptors' sizes and is
never copied, as the library answers it, since a guest's descriptors may
add up to many times the host's memory. Queue 1
takes the session's messages from a ring as the guest posts buffers.
DRIVER_OK starts the session, and a reset ends it.
`REIMS_VGPU_LOG=warn,reims_vgpu::video=debug` shows each message both ways.

- **The worker.** The library answers each element on a thread of the
  device's own, as Apple's host answers them on the device's own queue,
  so the main loop runs through it: a `jdec` waits for its JPEG decode,
  and a session's end for the session's thread, 10 s at most each. As the
  scaler's worker does ("The M2 scaler device", below), the notification
  pops one element and hands it over, and a bottom half writes the status
  back on the main loop and only then pops the next, so the elements are
  answered one at a time, in the guest's order. The ring of host messages
  is not drained while an element is with the worker, so a message the
  session sends while answering one reaches the guest after that
  element's status, as Apple's replies, queued behind the handler on the
  device's queue, do. A reset and the unrealize wait, under the BQL, for
  the element with the worker, and discard its status; the worker never
  takes the BQL, and neither `send_to_guest` nor the IOSurface device's
  queue waits for anything that holds it.

The guest's kext waits for its IOSurface mapper's service before it sets
DRIVER_OK, so on q35 the device needs `apple-gfx-iosurface-pci`.

Queue 0 holds 1024 descriptors, QEMU's most (`VIRTQUEUE_MAX_SIZE`), and
queue 1 holds 256. Apple's device holds 256 on each, the most its virtio
base takes. The guest's AppleVirtIO sends each message as a direct chain,
one descriptor per physical segment, sizes the queue by what the device
reports, and refuses a message of more segments than that with
kIOReturnBadArgument, so the message never reaches the device.
VideoToolbox cuts a sample into messages of up to 0x200000 bytes: 128 of
an arm64 guest's 16 KiB pages, but 512 of an x86-64 guest's 4 KiB, where
256 descriptors carry 1 MiB at worst. Queue 1's buffers are 0x4000 bytes,
so it keeps Apple's size.

**Measured** on homelab-gem10 on 2026-10-24, `x86_64-softmmu`, with no
guest: realized paused on q35 under KVM, alone and beside `apple-gfx-pci`
and `apple-gfx-iosurface-pci`, it is 106b:1a03, class 0x120000, subsystem
106b:1a03, with BAR 1 (4 KiB, MSI-X, 3 vectors) and BAR 4 (16 KiB, 64-bit
prefetchable: common, ISR, device and notify at 0x0 to 0x3000), its
capabilities linked 0x40 to 0x98 in ascending order. A qtest driver brought
it to DRIVER_OK offering `VERSION_1` and no device feature, read queue 0's
size as 1024 and queue 1's as 256, and had a message in three readable
descriptors answered with status 0 across two writable descriptors of 2
bytes, a `subs` atom's IDs resolved to no surface with no IOSurface device,
and a writable part of 2 bytes left as it was with a used length of 0. On
the worker, three messages on one notification were answered in their
order, each status before its reply, and after a reset made at once after
a notification nothing of the element reached the new queue.

**Measured** on homelab-gem10 on 2026-10-24, with x86-64 macOS 26.7.1
guests running the x86-64 port of the guest's kext, on renderD128 (the
RX 7900 XTX):
- with 256 descriptors on queue 0, each of three AV1 samples of 1,886,022
  to 2,101,248 bytes failed in the guest with -536870206 (0xe00002c2,
  kIOReturnBadArgument), and no message larger than 1,332,549 bytes
  reached the device;
- with 1024, all nine AV1 streams and four H.264 and HEVC streams decoded
  in hardware sessions and matched their references frame for frame. The
  device took one message of 2,093,408 bytes, and a 2,101,248-byte sample
  as a fragment of 0x200000 bytes (flags 0x10) and a last part of 4,512;
- the first JPEG decode after the GPU had idled took 880 to 890 ms on the
  host. With the elements answered on the main loop, QMP's `query-status`
  went unanswered 879 and 883 ms for two of them; on the worker, no
  exchange during the decodes took over 3.3 ms.

## The M2 scaler device

`apple-m2scaler-pci` (`hw/display/apple-m2scaler-pci.c`) is the transport
of the paravirtualized M2 scaler device, 106b:1a0b, which a macOS guest's
AppleM2ScalerParavirtDriver drives for IOSurfaceAccelerator; the library's
scaler device (`rvg_scaler_device_*`, `crates/reims-vgpu/src/scaler.rs`)
is its host side, over reims-vgpu-video-core's scaler, transforming with
the library's own kernel on the GPU it draws with: Metal's on macOS, the
Vulkan platform's on Linux, any GPU ([video/scaler.md](video/scaler.md),
"The kernel"). It has no properties. Where that opens no device, or the
device refuses the kernel, the device is realized all the same, so a VM
given it starts on every host: `rvg_scaler_device_scales` answers false,
and the device's `validate_features` refuses the guest's features
(`-ENOTSUP`, traced as `apple_m2scaler_features_refused`), so FEATURES_OK
stays unset. The guest kext reads its status back, fails its start before
DRIVER_OK, writes FAILED and registers no service, and the guest's
IOSurfaceAccelerator does without it ([video/scaler.md](video/scaler.md),
"A host without a GPU"). A VM takes it as `-device apple-m2scaler-pci`;
reims-ctl gives every VM on reims-vgpu one, an x86-64 VM's at slot 9.
Each queue element's readable part, at most 64 KiB, is copied into one
buffer for `rvg_scaler_device_process`, which answers in place. A longer
one is a needs-reset before anything is allocated or copied, as the
library refuses it too (`scaler::MAX_REQUEST_SIZE`); Apple's host copies
any size, and the bound is ours, so that a guest's descriptors cannot have
QEMU allocate and copy gigabytes. The guest kext's largest request is 0x608
bytes. A reply goes into the writable part with a used length of its size;
a needs-reset is `virtio_error`, with the element detached and no reply; an
end of the VM aborts, as Apple's host ends its process. DRIVER_OK starts
the device and a reset resets it.
`REIMS_VGPU_LOG=info,reims_vgpu::scaler=debug,reims_vgpu_video_core::scaler=debug`
logs each request and transform.

- **The worker.** The library answers each element on a thread of the
  device's own, so the main loop runs through a transform: a 16384x16384
  turn takes 3 to 4 s of the GPU. The notification pops one element and
  hands it over; a bottom half returns the worker's answer on the main
  loop and only then pops the next, so the elements are answered one at a
  time, in the guest's order, as Apple's handler answers them. A
  needs-reset is set from that bottom half; an end of the VM aborts on the
  worker. A reset and the unrealize wait, under the BQL, for the element
  with the worker, and discard its answer; the worker never takes the BQL.
- **The guest's memory.** The capabilities' address and the side
  buffers' lists are guest-physical. The device answers the library's
  `is_ram`, `read` and `write` on the worker, on `VirtIODevice.dma_as`:
  each piece of the range mapped with a `MemoryRegionCache`, every one
  RAM and not a RAM device or none of it, and the bytes copied through
  the same caches, so the check and the copy see the same memory and no
  access dispatches to a region that would take the BQL. At most eight
  pieces, and a range whose end wraps is refused before the walk, which
  only adds what each cache took and subtracts it from what is left, so a
  guest's 64-bit length cannot overflow it (reviewed 2026-10-24). An
  IOMMU's address space is not translated by the caches: behind one every
  request naming the guest's memory would be refused; the guests have
  none.
- **A needs-reset at DRIVER_OK.** `virtio_set_status` writes the status
  after the device's `set_status` returns, so a `virtio_error` from inside
  it is overwritten and lost. Where the start fails, a bottom half sets
  DEVICE_NEEDS_RESET once the status is written; the reset cancels it.
- **The virtio ID** is QEMU's alone: `VIRTIO_ID_VIDEO_ENCODER`, one
  `hw/virtio/virtio-qmp.c` lists (an ID it does not list reaches its
  `g_assert_not_reached`), with Apple's PCI IDs written over the one
  virtio-pci derives, as `apple-videotoolbox-pci` does with
  `VIRTIO_ID_VIDEO_DECODER`.

The guest kext waits for its IOSurface mapper's service, as the video
kext does, so on q35 the device needs `apple-gfx-iosurface-pci`. The
guest's IOSurfaceAccelerator looks the kext up by its own class only
where `kern.hv_vmm_present` reads 1, and by Apple's hardware scaler's,
`AppleM2ScalerCSCDriver`, where an OpenCore hides the VM (reims-util's
default). The kext's x86-64 port answers to both names, so either guest
uses the device; Apple's build of it is bound and unused in a guest that
hides its VM.

**Measured** on homelab-gem10 on 2026-10-24, `x86_64-softmmu`:
- Realized paused on q35 under KVM, alone and beside `apple-gfx-pci`,
  `apple-gfx-iosurface-pci` and `apple-videotoolbox-pci`: 106b:1a0b,
  class 0x120000, subsystem 106b:1a0b, BAR 1 (4 KiB, MSI-X) and
  BAR 4 (16 KiB, 64-bit prefetchable).
- A qtest driver had the capabilities written into guest RAM, an unknown
  surface's transform refused with its completion cleared, side buffers
  gathered and scattered across descriptors, an unknown type replied
  unchanged, at 64 KiB too, and a readable part of 64 KiB and a byte, a
  short request and one naming the IOAPIC's page each a needs-reset,
  served again after the driver's reset; a list of nine entries ended
  QEMU. On the worker, three requests on one notification were answered in
  their order, and a reset at once after a notification put nothing on the
  new ring.
- With x86-64 macOS 26.7.1 guests running the kext's x86-64 port, the VM
  hidden and shown, on the RX 7900 XTX: a probe's 39 transforms ran on
  VA-API's video processing, the backend before the kernel, 34 done and 5
  refused past the offered ratios with
  kIOReturnBadArgument, as Apple's scaler refuses them. A 16384x16384 BGRA
  quarter turn took 3.3 to 4.9 s on the host; with the elements answered
  on the main loop QMP's `query-status` went unanswered that long, and on
  the worker it answered within 23 ms through every turn.
- On an RTX 5080 Laptop GPU (driver 615.71.09, 2026-10-25), on the
  library's kernel: realized paused under KVM, alone and beside
  `apple-videotoolbox-pci`, the device transforming on that GPU, and a
  qtest driver's ACKNOWLEDGE, DRIVER, `VERSION_1` and FEATURES_OK read
  back as 0xb, FEATURES_OK kept, where VA-API's video processing had left
  that GPU without a scaler.

## apple-gfx on reims-vgpu

`vendor/qemu`'s `apple-gfx` is three C files, `hw/display/apple-gfx.c`,
`apple-gfx-mmio.c` and `apple-gfx-pci.c`, on the library's C API,
`reims_vgpu.h`. They use no Objective-C, Metal, IOSurface, dispatch, blocks
or autorelease pools, and call 21 of the header's 48 functions. Beside them,
`apple-gfx-activity.m` holds Foundation's activity against App Nap while a
device is realized, and calls nothing of the library ("App Nap", below).
How it uses the API:

- **Realize** fills an `RvgDeviceConfig`: the `displays` property's port
  count (1 to 8, default 1; any other value fails the realize), the
  system's default Metal device (`metal_registry_id` 0) and the guest's
  memory map, walked from the flat view into an array freed after the
  create, since the library copies it. The map is the flat view at the
  realize: RAM mapped later is not in it. The PCI variant gives
  `raise_interrupt`, and always sets `use_iosurface_mapper`,
  `process_isolation` and `process_isolation_follows_guest`; it has no
  option for them. An `apple-gfx-iosurface-pci` makes the IOSurface host
  device, and without one every task runs in process ("The IOSurface
  device over PCI", below). The MMIO one sets the first two always,
  and makes the IOSurface host device after the device, from a second walk. Display i takes port i, the next serial
  number, "Reims vGPU", 400 x 300 mm and a callback table, then the modes
  (the `display-modes` property, or 1920 x 1080, 1440 x 1080 and 1280 x
  1024 at 60 Hz), and console head i. A refused create fails the realize
  with `rvg_status_message`'s text, after releasing the displays made so
  far and the device. A migration blocker stands in for suspend and resume,
  which QEMU does not call.
- **The files behind guest RAM.** Each range of the map also names the file
  behind it (`RvgDeviceConfig.memory_files`), where its RAM is a shared
  mapping of a file in pages of the host's size: the RAM block's descriptor,
  and the range's offset in the block plus the block's offset in the file.
  On Linux the library builds its views of guest pages from these files, and
  a range without one fails the realize, which says to back the machine's
  RAM with `-object memory-backend-memfd,share=on,id=<id>` and `-machine
  memory-backend=<id>`, without hugetlb. On macOS QEMU passes the files only
  where every range has one, and the library checks them without using
  them; `vmapple`'s anonymous RAM has none. The `apple_gfx_memory_range`
  trace lists each range with its host address, descriptor and offset.
- **Frames.** A mode change makes an `RvgFrameTarget` of the surface's size
  in BGRA8, the byte order of pixman's x8r8g8b8. A render retains the
  target, and a pool job encodes the display's frame into it, waits, and
  reads the target's pixels into a buffer of the job's own with
  `rvg_frame_target_read_pixels`; a bottom half then copies them into the
  surface, if the size is unchanged. The read stays off the main loop: on
  the Linux laptop's RTX 5080 a 1920 x 1080 read took 1.7 s on average
  (2026-10-10), and made in the bottom half it starved the guest's AHCI
  disk. A display renders one frame at a time.
- **Callbacks** come on the library's display thread, one at a time in the
  guest's order: a mode change takes the BQL and sets the mode
  synchronously, and a new frame schedules a bottom half. The cursor
  callbacks queue their work, with the show value and a copy of the glyph's
  pixels and hot spot, on the device's event queue, which one bottom half
  runs under the BQL in the order it was queued; one-shot bottom halves
  scheduled together run in the reverse order, and let a cursor glyph be
  replaced by the one set before it. Without window mode, the console's
  surface follows the guest's frames only while a display change listener
  (VNC) or a screendump asked for it within the last 6 s, twice the slowest
  refresh: one render at a time, of the newest frame, the frames presented
  meanwhile skipped. Otherwise nothing is rendered at a present, and the
  surface is drawn when the display system asks for it (`gfx_update`), if
  the guest presented a frame since the last render began or the mode
  changed; that render's completion signals the update done, so a
  screendump reads the newest frame.
- **MMIO.** The vCPU's thread calls the library for the reads and writes of
  both MMIO regions, and waits only on the library's threads, which never
  wait on QEMU: the device's one callback, `raise_interrupt`, schedules a
  bottom half and returns, even on the vCPU's thread inside the access, and
  `mode_change`, the one callback that takes the BQL, runs on the display
  thread, which no access waits for. The GPU device's region is lockless
  (`memory_region_enable_lockless_io`): under KVM a vCPU makes its access
  without the BQL; HVF takes the BQL at every exit, and TCG at every access to
  I/O, so under them the accesses run one at a time. The one access that
  takes the BQL itself is a write to 0x1210 on `apple-gfx-pci`, after the
  library's call has returned ("BAR 2", below). Until `93afbf1f1e` each
  access went to the device's `ThreadPool` while the vCPU waited in
  `AIO_WAIT_WHILE`, holding the BQL, and one vmapple run hung there for good
  ([lessons.md](lessons.md)). The pool now runs only the renders, which it
  takes with `thread_pool_submit_immediate`. That call needs upstream's
  `2566b6e902`: a job that completed before the call resized the pool to the
  work left let the pool be resized to 0 threads, and QEMU aborted on
  `thread_pool_set_max_threads`'s `assert(max_threads > 0)`; both of two
  vmapple boots without the fix, when the accesses went through the pool,
  aborted 3 min 29 s in. `apple-gfx-mmio`'s IOSurface registers are read
  and written as 32 bits: values zero-extend and truncate, where the
  library's functions take and return 64 bits. `apple-gfx-iosurface-pci`
  passes an 8-byte access whole, and a 4-byte one as `apple-gfx-mmio` does.
- **Reset.** Each bus front-end's reset calls `rvg_device_reset` in its hold
  phase. It may call back, and the device's one callback, `raise_interrupt`,
  only schedules a bottom half.
- **BAR 2** (`apple-gfx-pci`, with its `efi-framebuffer-bar` property, on by
  default): a 64 MiB, 32-bit, prefetchable window onto the EFI framebuffer,
  for guests with no driver of the device's own ("x86-64 macOS guests:
  installers and recovery", below). It holds one alias of the machine's RAM,
  disabled at realize. After the library has taken a guest write to 0x1210
  (the framebuffer's start), `apple_gfx_write` calls the PCI variant's
  `mmio_written` hook, which takes the BQL (under KVM the access came without
  it) and reads the start back from the library, since vCPUs' hooks may reach
  the BQL in another order than their writes reached the library. It looks the
  start up in the system memory's flat view, in the library's pages
  (`APV_PAGE_SHIFT`: 4 KiB on an x86-64 host, 16 KiB otherwise). Where the
  machine's RAM is there, BAR 2 shows the run of it from the start, up to
  64 MiB, at the BAR's offset 0: only RAM the guest reaches at those addresses
  then, so the window ends where SMRAM, a ROM or a device's region lies over
  the RAM. A start of 0, or anything else there (a device's region, a NUMA
  node's memory), disables the alias. The update is done before the access
  returns, since the option ROM reads its buffer back through BAR 2 right
  after its writes, and a repeated start changes nothing. Only 0x1210 writes
  and a reset change BAR 2, so it shows the framebuffer from the ROM's write
  until the next one or a reset: after EFI goes off the kernel's draws to its
  boot framebuffer still land in RAM. QEMU leaves BAR 2 out where the machine
  has no RAM region, or its RAM is a guest_memfd's, which KVM refuses a second
  memory slot onto. The library never sees BAR 2: the ROM writes the RAM page
  as before, and only the address its Graphics Output Protocol gives is
  BAR 2's ([architecture.md](architecture.md) 11.4).
- **The IOSurface device over PCI** (`apple-gfx-iosurface-pci`): the
  device `apple-gfx-mmio` has as its second region, for x86-64 guests that
  run Apple's arm64 GPU driver, ported (`AppleParavirtGPUIOGPUFamily`). That
  driver's start waits for the mapper driver's service with no timeout, and
  makes every texture over an IOSurface a mapper texture (object type 11).
  The mapper driver takes memory index 0 and interrupt index 0, which the
  GPU's function, with its one MSI vector, cannot give it beside the GPU
  driver, so it is a PCI device of its own:
  - vendor 0x106b, device 0xeeef (ours: no Apple device has it), class
    0x0880 (system peripheral);
  - BAR 0, 64 KiB of 32-bit memory, with the IOSurface host device's
    registers at 0x1000 to 0x1030, as in vmapple's region; an 8-byte
    access passes whole, since the ring's and the descriptor table's bases
    are guest physical addresses that may lie above 4 GiB;
  - one MSI vector, notified from a bottom half the IOSurface device's
    worker schedules once per completed command, while the guest has MSI
    on.

  Its `gpu` link names the `apple-gfx-pci` it maps IOSurfaces for, which
  must come before it on the command line. Unset, it is the machine's only
  `apple-gfx-pci`, and realize refuses when there is none before it or
  more than one. Realize refuses a second instance, since a device maps
  through the process's latest IOSurface device
  (`rvg_iosurface_device_create`). It makes the IOSurface device from the
  flat view, as `apple-gfx-mmio` does, after the GPU's device. The
  IOSurface device's config has no files: on Linux the library maps guest
  pages from the process's backing table, where the GPU's device
  registered the files behind guest RAM
  (`crates/reims-vgpu-vulkan/src/os/linux/files.rs`). The library has no
  reset of the IOSurface device. A reset of `apple-gfx-iosurface-pci`
  writes 0 to its registers through the library's MMIO writes, as a guest
  would: the ring base (which unmaps the ring), the descriptor enable and
  the descriptor table base (which tear every surface down and drop the
  table), the ring size, and the read and write counters, in that order.
  The library ignores a ring size of 0, so that register keeps its last
  size, which maps nothing until a guest writes a base and a size again; a
  write counter's doorbell with no ring runs nothing. A guest that reboots
  into Apple's x86-64 driver so finds no descriptor table left from the
  port. A PCI reset also turns MSI off, and the guest's mapper writes every
  register again at its start. Adding the device is the only switch:

  ```sh
  -device apple-gfx-pci,id=gpu0 -device apple-gfx-iosurface-pci
  ```

  Isolation is the setup in which the library writes info replies into
  guest memory: in process each one ends QEMU ([testing.md](testing.md)
  7.4), and Apple's arm64 driver sends them from binary version 101 (it
  asks for 245). `apple-gfx-pci` isolates a task only when, as the guest
  defines it, the `apple-gfx-iosurface-pci` it maps through has a
  descriptor table base other than 0 (`process_isolation_follows_guest`);
  otherwise the task runs in process, and each task made says which in an
  info line (`task N runs isolated` or `runs in process`, with the
  reason). Device info offers argument buffers, which only isolated tasks
  serve, by the same rule, read when the guest asks, and logs one line
  each time its answer changes (`device info: argument buffers on` or
  `off`, with the reason). The port's mapper
  driver writes the base at its start, and Apple's arm64 driver, ported,
  waits for the mapper's service before it makes a task. Apple's x86-64
  AppleParavirtGPU never touches 106b:eeef, and references IOSurfaces
  through ref textures, whose backing is in task 0, which an isolated task
  does not serve ([testing.md](testing.md) 6.20). With every task
  isolated, macOS 26.7.1 (25G241) under that driver, on homelab-gem10's
  780M, had each such task end at `gpu_task.unsupported.backing_for_id` (11
  times in one boot), every later command on it fault, and the guest's GPU
  hang through its restarts, with WindowServer crashing; with isolation off
  the same boot passed (paravirt-x86-guest's runs `20261017-004627` and
  `-013359`, against `-012431`, 2026-10-17). The driver asks for binary
  version 6 there. Under reims-util's route (Lilu and ReimsVgpu.kext) it
  asks for 245 and still makes ref textures: with every task isolated, run
  `20261017-022711` had the same 11 deaths, a WindowServer crash and none
  of metalwin's 60 frames. So the version cannot tell the drivers apart,
  and the mapper can: Apple's x86-64 driver runs in process, stock or
  patched, and Apple's arm64 one, ported, isolated. On an x86-64 host the IOSurface
  device offers only the first descriptor form at 0x1028
  ([testing.md](testing.md) 6.23), which the mapper driver needs on 4 KiB
  pages. A VM with no `apple-gfx-iosurface-pci` runs every task in process
  and is told no argument buffers, as before the device existed.
- **Not used**, since QEMU's deployment target is 26.0: the task list and
  the memory callbacks macOS before 15.4 needed. QEMU gives the device
  `raise_interrupt` and a memory map only. The PCI IDs, BAR 0 and MSI count
  come from the header (`RVG_PCI_*`); BAR 2 is QEMU's own, and so are the
  IOSurface device's PCI ID, BAR and MSI count.

### Window mode

Under `-display none` on macOS, QEMU runs its main loop on a thread named
`qemu_main` and leaves the main thread in `CFRunLoopRun()`, where the
library can run AppKit (`reims-vgpu-window`'s `native/macos/main_thread.rs`).
`apple-gfx` then has the library open a window for each of its displays:

- **When.** At realize, when `query-display-options` gives the display type
  none and QEMU's environment does not set `REIMS_VGPU_WINDOW=0`. The type
  is none under `-display none` and `-nographic`, and under `-vnc` without
  `-display`. QEMU says so once, on stderr: `apple-gfx: -display none: each
  display opens a window of its own; REIMS_VGPU_WINDOW=0 keeps QEMU
  headless`. Any other display type, or `REIMS_VGPU_WINDOW=0`, keeps
  `apple-gfx` headless.
- **Modes.** Each display is offered 1920x1080 at 1x, then the standard
  sizes within it, at the rate the host screen is set to
  (`RVG_WINDOW_MATCH_HOST_SCREEN` with a mode), and the window scales the
  guest's frames to its size on the GPU. The `resolution` property picks
  another mode; `<w>x<h>@<ppi>` also tells the guest that density, and the
  guest runs a HiDPI desktop at the mode's pixels where it reads 146.667
  ppi or more across the width in whole centimetres, the told millimetres
  rounded to nearest and then up (`2560x1600@200`, the display `macosvm`
  gave the guest in its `vm.json`, looks like 1280x800; `@148` reads 44 cm,
  147.8 ppi, and `@146` 45 cm, 144.5 ppi, a 1x desktop). A windowed window
  on a Retina screen shows a HiDPI guest one guest pixel per screen pixel
  where half the mode fits. QEMU refuses a density that does not tell the
  guest's scale for the mode, the only one it checks ("A chosen density's
  boundary", below); or `host` for the host screen's own modes at its
  density:
  `-global apple-gfx-mmio.resolution=2560x1600` on `vmapple`, or
  `-device apple-gfx-pci,resolution=host`; `run_vmapple.sh` passes
  `RESOLUTION=`. A malformed value fails the realize. `apple-gfx` sets no
  modes of its own; with the `display-modes` property, it passes no flag
  and sets those. The library's `REIMS_VGPU_WINDOW_*` variables pick each
  window's screen and style and the offered modes' rate ([docs/api.md](api.md)
  section 2, and `crates/reims-vgpu-window/features.md`). With
  `resolution=host` on the M3 Max's built-in Retina screen the macOS 13.6
  guest boots to a HiDPI desktop, 3456x2160 looking like 1728x1080: it
  picks that from the size in millimetres, read in whole centimetres, and
  not from the scale factor, and took the screen as 1x while it was told
  144 ppi (`crates/reims-vgpu-window/src/policy.rs` has the measurements).
  At that mode it boots much slower than at 1920x1080, which is why
  1920x1080 is the default.
- **Several displays.** `-global apple-gfx-mmio.displays=N` on `vmapple`,
  or `-device apple-gfx-pci,displays=N`, gives the device N display ports,
  up to 8, each with a display, a console and, in window mode, a window of
  its own.
- **The windows.** Each opens once QEMU's main thread runs its run loop,
  titled "Reims vGPU", with " (port N)" for port N above 0; the guest sees
  each display named "Reims vGPU". QEMU makes no
  `NSApplication`, so the library makes one and runs it on the main thread
  for good: a Dock icon, a ⌘Tab entry and a menu named after the process.
  A window is windowed by default, sized at one guest point per host point
  (a 1x guest's pixel is a point, scaled on the GPU; a HiDPI guest's is a
  screen pixel) for the first mode the display offers, then for each mode
  the guest sets until the person sizes it, and windows on one screen open
  24 points apart; `REIMS_VGPU_WINDOW_BORDERLESS=1` makes each cover its
  screen.
  [docs/api.md](api.md) section 2 describes a window, and
  `crates/reims-vgpu-window/features.md` lists what it does for the person
  at the host.
- **Frames.** The library draws the guest's frames into the windows; QEMU
  does not render each frame into its console's surface. It draws the
  surface only when its display system asks: a `screendump`, or a listener
  such as VNC. Where the guest presented nothing since the last render
  began, and the mode is unchanged, the surface has the newest frame, and
  the update is done at once without a render
  (`apple_gfx_render_new_frame`). Otherwise it renders the display's
  current frame once, and is done whether or not the render drew, so a
  screendump never waits for a frame that will not come, and one before the
  guest's first mode returns at once.
- **Input.** The window callbacks come on the main thread on macOS, and on
  the library's `reims-vgpu.ui` thread on Linux, without the BQL, and queue
  their events on the same event queue as the cursor's work, with the
  display's console as the source. On `vmapple` they reach the USB tablet
  and keyboard the machine adds: the pointer as absolute X and Y clamped to
  the mode, with the left, right, middle and both side buttons; each scroll
  event as smooth scroll events, the distance in 1/256 point with the
  gesture's phases, where the tablet takes them (the machine's default
  `usb-vmapple-tablet`), and else as wheel button presses and releases, one
  per notch (`tablet=usb-tablet`, which has no horizontal wheel, so
  horizontal notches go nowhere on it); each key's USB usage mapped to
  QEMU's Linux key codes and sent at once.
- **Closing a window**, or choosing Quit, powers the VM off: a host-UI
  shutdown request with the power-off action, as closing QEMU's own windows
  does, and QEMU exits.

`scripts/internal/run_vmapple.sh` runs QEMU with `-display none` and
`REIMS_VGPU_WINDOW=0`, so its runs stay headless for `vm_drive.py`'s tours;
`WINDOW=1` opens the windows ("Steps", below).

**Measured** on this host (an Apple M3 Max, macOS 26.7, 25G229) with the
macOS 13.6 guest ("The guest", below):
- **Paused** (2026-09-30, `-S`, no guest code run), `virt` with
  `apple-gfx-pci` and `displays=2` opens two windows, the second 24 points
  right of and below the first. Borderless, a window covers the built-in
  screen, with the image below its 32-point camera housing. QMP `quit`, and
  Quit from the Dock, each end QEMU with exit 0.
- **Booted with `WINDOW=1`** (2026-09-30): QEMU logs the one line above and
  no error line. The guest sets the host-matched mode 3456×2160
  (`apple_gfx_mode_change x=3456 y=2160`), and a `screendump` returns a
  3456×2160 image through the render on demand. Told 144 ppi, the guest
  showed that desktop at 1×; told 220 ppi, it boots to HiDPI, looking like
  1728×1080 ("Window mode", above).
- **With the host's screen locked**, the window reports itself hidden and
  draws nothing, and `capture-app`, a window capture tool on the host,
  cannot capture any window.
- **A chosen density** (2026-10-01, `RESOLUTION=2560x1600@200`, 8 vCPUs,
  from a fresh clone): the guest sets 2560x1600 (`apple_gfx_mode_change`)
  3.3 s after QEMU starts. The window opens with 1280x800 points of content
  on the built-in 2x screen, 1280x832 with its title bar, and `capture-app
  --native-resolution` returns 2560x1664: one screen pixel per guest pixel.
  The guest runs HiDPI: its 13-point body text is drawn about 26 pixels
  high.
- **A chosen density's boundary** (2026-10-01, `vendor/qemu` from
  `ffb8ae8b17`, 4 vCPUs, 8G, `WINDOW=1`, each boot from a fresh clone): the
  guest's scale, from its menu bar's height in a screendump (24 pixels at
  1x, 48 at HiDPI) and the window's `capture-app --native-resolution`
  capture. Each density is the told width in millimetres, the whole
  centimetres the guest reads (the millimetres rounded to nearest, then
  up) and the pixels per inch across them:
  - 1x: 2560x1600@146 (446.42 mm, 45 cm, 144.5 ppi), 3456x2160@147 (598.57,
    60, 146.30) and @149 (590.54, 60 from 591 mm, 146.30), 2944x1840@148
    (506.45, 51, 146.62), 2656x1660@149 (453.84, 46, 146.657),
    1790x1119@148 (307.93, 31, 146.665), 3811x2382@148 (655.60, 66,
    146.666), and 1280x800@220 (148.13, 15, 216.7);
  - HiDPI: 2021x1263@149 (345.33, 35, 146.667), 3176x1984@148 (546.36, 55,
    146.673), 2080x1300@149 (355.41, 36, 146.76), 3008x1880@148 (517.46, 52,
    146.93), 2560x1600@148 (440.39, 44 from 440 mm, 147.8) and @149 (437.43,
    44, 147.8), 1920x1200@149 (328.08, 33, 147.8), 2624x1640@149 (448.37,
    45, 148.1), 2048x1536@149 (349.95, 35, 148.6), 5120x2880@218 (597.96,
    60, 216.7), 2560x1600@227 (287.13, 29, 224.2), and 5832x3645@148
    (1003.26, 101, 146.666) and @130 (1142.17, 115, 128.8);
  - so from 1920 to 3811 pixels wide the width's reading alone decides,
    whatever the height: 3456x2160@149 reads 148.3 ppi down its 37 cm and
    ran 1x. 5832 pixels wide ran HiDPI at 128.8 ppi, and 1280x800 1x at
    216.7; where between those widths, or whether the 1280x800's 15 cm
    rather than its pixels, decided it was not measured. apple-gfx refuses
    the densities these leave unknown, and a told width over 65535 mm, more
    than the settings page's 16 bits hold (`hw/display/apple-gfx.c`,
    `apple_gfx_guest_scale_unknown`), and `reims-vgpu-window`'s `policy`
    says what the window takes the guest to run at each.
- **Two displays** (2026-10-01, `vendor/qemu` at `c912618391`, 8 vCPUs,
  `-global apple-gfx-mmio.displays=2`, `WINDOW=1`): both windows open, and
  port 1's stays black while port 0 shows the desktop. The guest runs its
  device at binary version 62: it writes `0x3e` to 0x1034 0.3 s after QEMU
  starts, before its kernel runs, so from its boot firmware, and nothing
  writes 0x1034 again. It then reads the port count at 0x122c once, which
  gives 1 below version 245 (`supportsMultipleDisplays`; docs/testing.md
  6.21), and never again. The trace has one mode change, 1920x1080, so
  port 1 never gets a mode.

### Isolated tasks

`apple-gfx-mmio` sets `process_isolation`, and so does `apple-gfx-pci`,
with `process_isolation_follows_guest`: there a task is isolated when the
guest defines it while its `apple-gfx-iosurface-pci` has a descriptor
table ("The IOSurface device over PCI", above). The library then runs each
isolated guest task on threads of its own in QEMU's process
([architecture.md](architecture.md) 5.1). QEMU starts no other process: an
isolated task's CPU, and the Metal lines it logs, are QEMU's.

### Linux hosts: `vm.max_map_count`

On Linux the library's views of guest memory take a kernel mapping for each
run of guest pages. An x86-64 guest's GPU memory comes in scattered 16 KiB
pieces, so a view takes about 64 mappings per MiB. Measured on an RTX 5080
Laptop GPU host (CachyOS, whose `vm.max_map_count` is 1,048,576) with a
12 GiB macOS 26.6.1 guest on 2026-10-10:
- 2.2k to 3.3k guest-memory mappings at an idle desktop, and 22k to 29k
  with Safari and Maps open;
- back to the idle count after they quit, in each of three rounds;
- 1.7k to 4.6k other mappings in the process.

The views stop 16,384 mappings short of `vm.max_map_count`, and a view past
that is refused. Past the limit itself the kernel refuses QEMU's and the
GPU driver's maps too. A stock limit of 65,530 holds about 770 MiB of live
views. When a guest's RAM is leased, the library warns if the RAM, in 16 KiB
pieces, would not fit. `sysctl -w vm.max_map_count=1048576` covers a 12 GiB
guest.

### Linux hosts: AMD GPUs and udmabuf

RADV refuses to import pages with a file behind them as host memory, and the
guest's RAM and the library's views are all memfd pages, so a macOS 26 guest
on an AMD GPU died at its first no-copy buffer: `reims-vgpu: fatal: platform
call newBufferWithBytesNoCopy:length:options:deallocator:: the Vulkan runtime
refused it: unsupported: Unsupported { feature: HostMemory, ... }` (Radeon
780M, RADV of Mesa 26.0.8, Linux 7.0, 2026-10-11). Where the pointer import is
refused, the runtime now imports dma-bufs of the same pages, which the library
makes with `/dev/udmabuf`, cut into chunks within udmabuf's limits as the host
has them (`crates/reims-vgpu-vulkan/src/os/linux/dma_buf.rs`). The device node
is the `kvm` group's on Ubuntu.

What the chunks must hold, measured on an RTX 3090 host (where every import is
host memory) with that guest, 14 GiB, one 1920x1080 display, on 2026-10-11:
at the idle desktop, 829 imports, the largest 63.3 MiB, 19 of more than 1024
runs of file pages and the most 16,201 (4 KiB pieces); with Safari open on
wikipedia.org, 4,551 imports and 29 of more than 1024 runs. At udmabuf's
defaults (1024 runs, 64 MiB) the 16,201-run import is 16 chunks.

On the Radeon 780M with udmabuf at its defaults and metal2vulkan `11715e2a3f`
(2026-10-12), the guest cold-booted to its desktop, and Safari loaded
wikipedia.org and quit with no death: 5,119 dma-buf requests, textures' own
among them, none refused; 5,095 took one dma-buf, 19 two, 2 three, 2 seven and
1 sixteen.

### Linux hosts: a 4K display on a Radeon 780M

At a 3840 × 2160 display an x86-64 macOS guest on a Radeon 780M drops about
a quarter of a 30 fps video's frames: its WindowServer's command buffers take
about 25 ms of the GPU a frame, two thirds of it in metal2vulkan's shadow
copies of guest surfaces and its conversion of the video's frames, and the
frames no longer fit. At 1920 × 1080 they take 12 ms, and about 1% of the
frames drop. B73 in [video/bugs.md](video/bugs.md) holds what is left and
whose it is.

Measured on homelab-gem10 on 2026-10-26 (Linux 7.2.9, RADV of Mesa 26.2.4),
with x86-64 macOS 26.7.1 (25G241) guests of 4 vCPUs, headless
(`display-modes`; at 3840 × 2160 the guest looks like 1920 × 1080), each a
cold boot from fresh overlays: 60 s after login, the Aerial screen saver for
120 s, its 3840 × 2160 HEVC Main 10 video decoded on VA-API at both display
sizes, counted by its layer's CoreMedia statistics in 6 s periods.

| reims-vgpu, metal2vulkan | display | enqueued | displayed | dropped | over 16 ms late | presents a second | GPU busy |
|---|---|---|---|---|---|---|---|
| f9c7a5811, 67c64f99d8 | 3840 × 2160 | 3,628 | 2,905 | 688 | 879 | 24.5 | 60% |
| f9c7a5811, 67c64f99d8 | 1920 × 1080 | 3,630 | 3,563 | 47 | 990 | 30.0 | not sampled |
| 221176d06, 27935d575e | 3840 × 2160 | 3,630 | 2,742 | 853 | 950 | 23.0 | 54% |
| 221176d06, 27935d575e | 1920 × 1080 | 3,628 | 3,562 | 45 | 929 | 30.0 | 35% |
| 73e48ad5a, 6c9cad047a | 3840 × 2160 | 3,628 | 2,785 | 810 | 1,014 | 23.0 | not sampled |

Boots of 60 s under reims-vgpu's trace (`REIMS_VGPU_TRACE`), which costs some
frames of its own, at 221176d06 and 27935d575e. Two switches in a copy of the
runtime, not committed, measure what its copies cost: the first skips the copy
in of every shadow of a million texels or more over a buffer's bytes, the
second also skips the YCbCr conversion, so the guest shows stale texels.
WindowServer's GPU time is the sum of its command buffers' GPU legs between two
presents (median, then p90); "starts" is its first commit after the last
present, at the median.

| boot | enqueued | dropped | over 16 ms late | presents a second | WindowServer's GPU time | starts | GPU busy |
|---|---|---|---|---|---|---|---|
| 3840 × 2160 | 1,826 | 416 | 497 | 23.0 | 25.2, 27.5 ms | 21.4 ms | 54% |
| without the large shadows' copies | 1,826 | 137 | 807 | 28.0 | 16.8, 19.8 ms | 29.1 ms | 43% |
| without those and the conversion | 1,826 | 39 | 38 | 30.0 | 8.1, 10.1 ms | 24.0 ms | 28% |
| 1920 × 1080 | 1,826 | 100 | 454 | 30.0 | 12.0, 14.3 ms | 22.5 ms | 40% |
| 3840 × 2160, at 73e48ad5a and 6c9cad047a | 1,826 | 413 | 511 | 23.4 | 24.8, 27.2 ms | 21.7 ms | not sampled |

At 67c64f99d8 the first three took 27.2, 16.8 and 8.2 ms. So of a 4K frame's
25.2 ms, the shadows take 8.4 and the conversion 8.7.

The boots at 73e48ad5a and 6c9cad047a ran a build that logs the runtime's
`WorkCounts` every 5 s, otherwise as above. Over the 120 s of the 4K screen
saver every texel copied in from those surfaces was an encoder's first copy
in of its texture: 68.7 billion, 572 million a second, and none of a texture
an earlier encoder of the same command buffer copied in, or after the
surface changed (B73).

- **What the copies are.** WindowServer commits three command buffers a frame
  on one task thread, the second the longest (18 ms at 4K). Every texture over
  a guest surface is a shadow the runtime copies in from the surface's bytes
  before each encoder that uses it (only a changed rectangle, where the
  runtime bounds the change), and keeps and merges back where commands write
  it (B26 in [video/bugs.md](video/bugs.md)); a two-plane YCbCr
  surface's texture, the video frame's, has its planes copied to a staging
  buffer and converted by a dispatch at each copy in (**read**, metal2vulkan
  27935d575e, `shadow.rs`, `ShadowSource::Converted`). In a build counting the
  runtime's work at 67c64f99d8, 115 s of the 4K screen saver copied in 500
  million texels a second of surfaces of a million texels or more, about two
  and a half 3840 × 2160 surfaces a frame, kept 333 million for 40 merges a
  second, and refreshed 20 YCbCr textures a second; at 1920 × 1080, 186
  million, 124 million for 60 merges, and 30.
- **Not the library's.** In the 4K boot at 27935d575e QEMU's threads took:
  the vCPUs 38 to 51% of a CPU each, the main threads 10.1%, the task threads
  3.1% in all, the FIFO threads 0.7%, the video device's 3.0% and the
  runtime's scheduler 2.3%. The swap presented 0.17 ms after the frame's last
  command buffer completed at the median, 0.26 ms at p99.
- **No waits in the kernel.** `perf record` of the GPU scheduler's and
  amdgpu's tracepoints, read only, for 8 s from 25 s into the traced screen
  savers: QEMU's graphics jobs, 69 a second at 4K and 65 at 1080p, ran 0.01 ms
  after their submission at the median, 0.67 and 0.02 ms at p90, 3.6 and 0.65
  ms at the most. QEMU unmapped 13 and 4 buffers (`amdgpu_vm_bo_unmap`), and 4
  graphics jobs at 4K waited on the page-table updates after them. DRM's
  fdinfo gave QEMU's graphics work 554 ms of each second at 4K and 339 at
  1080p, its decodes on VCN 235 to 237 ms, and VA-API's compute work 84 to 85
  ms.
- **The clock.** The 780M's shader clock (`pp_dpm_sclk`: 800, 1100 and 2700
  MHz, `power_dpm_force_performance_level` `auto`) read 800 MHz in 107 of 110
  one-second samples of the 4K screen saver, at 54% busy
  (`gpu_busy_percent`), and in all 114 at 1080p. In an earlier boot it rose to
  2.2 to 2.5 GHz only at 84% busy, as the guest shut down. The boots left the
  host's power policy as users have it.

Not measured: a window (`-display` other than `none`), where the host draws
each presented frame through a texture over the surface too, and other GPUs.

### Linux hosts: Intel GPUs without a shared last-level cache

Arrow Lake-S's iGPU (`deviceID` 0x7d67, Xe-LPG; ANV of Mesa 26.2.4 on i915,
Linux 7.2) shares no last-level cache with the host, and ANV's one host-cached
memory type there is not coherent (`DEVICE_LOCAL | HOST_VISIBLE |
HOST_CACHED`). metal2vulkan placed `Shared` and `Managed` textures in it for
their host copies, and the GPU read stale 64-byte lines of them: an x86-64
macOS 26 guest showed sparse, fully transparent 4x4-pixel blocks (one line of a
Tile4 32-bit image) in its boot logo, Dock icons and widgets, kept in each
texture (2026-10-12). Measured as the dark share of the boot logo's interior
once the GPU draws it, at 57 and 60 s of a cold boot: 34% at metal2vulkan
`11715e2a3f`; 6 and 17% with `INTEL_DEBUG=noccs`, with stale bright blocks
outside the logo too; 14 and 29% with `INTEL_DEBUG=stall`. Since `edb5ffd8cc`,
which keeps a host-copied texture out of memory the host maps without
coherence, there are none, and the Dock and widgets are whole.

### Linux hosts: NVIDIA's pipeline import budget

Measured on an RTX 3090 and an RTX 5080 Laptop GPU, NVIDIA driver
615.71.09, 2026-10-10: once a process has loaded about 29 MB of pipeline
code from a cache, `vkCreateGraphicsPipelines` answers
`VK_PIPELINE_COMPILE_REQUIRED` for the pipelines after it, so each of those
compiles, 14 to 24 s for each of the big glass pipelines (Xgc), on every
boot: on the 5080 about 10 of them fit in a process. The budget is the same
whether the code comes from `VkPipelineCache` data or from pipeline binaries
(`VK_KHR_pipeline_binary`), with the driver's own cache on or off, in one
cache or twelve, and destroying a cache gives none of it back. Only fewer
bytes help: smaller shaders, fewer pipelines per state. The driver's
disk-cache cap, a cap of the runtime's, a binary's identity and binary
records were each suspected first and ruled out.

### x86-64 macOS guests: Metal's error mode

An x86-64 macOS guest needs one setting, made once inside the guest. On a
desktop that comes up:

```sh
sudo defaults write /Library/Preferences/.GlobalPreferences METAL_ERROR_MODE -int 5
```

A new install's desktop does not come up without it, so there it is made
from macOS Recovery (OpenCore's Recovery entry, then Utilities > Terminal),
on the Data volume, followed by `reboot`:

```sh
cd /Volumes/*Data/Library/Preferences
defaults write "$PWD/.GlobalPreferences" METAL_ERROR_MODE -int 5
chmod 644 .GlobalPreferences.plist
```

Without it the guest's icon service ends as it starts, again and again:
apps wait minutes for their icons, and a guest with no icon cache yet, as a
new install has, shows only its wallpaper. Mode 5, the same as
`MTL_ERROR_MODE=nslog`, has Metal log an error report and go on rather than
end the process. Every process of every user reads the key, root's
included, from its next launch on, so a boot needs nothing else.
`sudo defaults delete /Library/Preferences/.GlobalPreferences
METAL_ERROR_MODE` undoes it.

Measured on an RTX 5080 Laptop GPU host with a macOS 26.6.1 guest on 2026-10-11:
- The guest's Metal plugin for the PCI device answers `nil` for the
  device's `targetDeviceArchitecture`. Loading a binary archive asks for the
  device's slice (`+[MTLLoader sliceIDForDevice:...]`), and Metal reports
  "Target device architecture is nil", twice for each load. With no error
  mode set, the report ends the process.
- iconservicesagent (root's and the user's), Spotlight and
  `com.apple.dock.extra` load IconRendering's binary archive, so each ends
  as it starts. launchd spaces out the restarts, doubling the wait up to
  320 s.
- An app's first icon request waits on its main thread, with no timeout,
  for iconservicesagent's reply. Calendar's launch waited 102.7 s, and the
  icons came back blank (error 4097) in Calendar, System Settings and the
  Dock.
- With the key, the report is logged and the load goes on. No slice
  matches, so the archive is empty, and each pipeline compiles from
  IconRendering's own library. The icons are correct, nothing ends, and
  Calendar launches in 0.41 s, also after a reboot of the guest. The report
  is the only Metal error report any process logs.
- Nothing of the archive reaches the host: QEMU's log has no line of it,
  with or without the key.

A new install, and the key made from Recovery. Measured on an RTX 3090 host
(Ubuntu 26.04, NVIDIA 615.71.09) on 2026-10-12, with a macOS 26.6.1 guest on
a qcow2 overlay of its disk, and the library and QEMU at `9b2c5452`:
- With the key, the desktop was up 60 s after the picker, with no exec
  fault.
- With the key deleted and the icon caches kept, the desktop was up by
  80 s. iconservicesagent, `com.apple.dock.extra` and Spotlight each ended
  every 10-20 s, and the host logged an exec fault for most of the ends.
- With the key deleted and the icon caches deleted too, as a new install
  has none (`/Library/Caches/com.apple.iconservices.store`, and the user's
  `com.apple.dock.iconcache`, `com.apple.dock.extra`,
  `com.apple.iconservices` and `com.apple.iconservicesagent` under `getconf
  DARWIN_USER_CACHE_DIR`), and the host's shader cache empty: only the
  wallpaper, with no menu bar, Dock or widgets, still at 5 min. The Dock's
  main thread waited in `-[ISConcreteIcon generateImageWithDescriptor:]`,
  in `xpc_connection_send_message_with_reply_sync` to iconservicesagent,
  which had ended 15 times. A beta tester's new install on an RTX 3050
  Laptop GPU host (NVIDIA 615.71.09) showed the same: the wallpaper only,
  and the same exec faults in QEMU's log.
- Recovery boots on BAR 2 ("x86-64 macOS guests: installers and recovery",
  below) to its window by 60 s after the picker, with the Data volume at
  `/Volumes/<name> - Data` and no password asked. The commands above
  brought that guest's desktop back by 80 s after the picker, with no exec
  fault.
- Two traps on the way there. `defaults write
  /Volumes/*\ -\ Data/Library/Preferences/.GlobalPreferences ...` expands
  no glob, since no file has that name without `.plist`: it made a literal
  `/Volumes/* - Data` in Recovery's RAM disk and wrote the key there. The
  `cd` expands the glob through the directory. And `defaults` leaves the
  file at mode 0600, where the user's processes read no key (`defaults
  read` as the user finds no such pair, as root it finds 5), so the icon
  service went on ending until `chmod 644`.
- Finder's Dock icon, asked for while the icon service was failing, stayed
  blank after the fix, also after `killall Dock`. Deleting the user's
  `com.apple.dock.iconcache` and then `killall Dock` brought it back.

No value the host supplies decides any of this, so Apple's own host would
end an x86-64 macOS 26 guest's icon service the same way. Read from the
guest's binaries and registry, and from the x86-64 code of macOS 26.7:
- The PCI device (`106b:eeee`) matches `AppleParavirtGPU.kext`, which names
  the `AppleParavirtGPUMetal` plugin (`MetalPluginName` in the registry).
  That plugin's device has no `targetDeviceArchitecture` of its own, so it
  answers Metal's base `nil`. The IOGPU-family plugin, which has one, is
  matched only by vmapple's device-tree node, never by a PCI device, and
  its kext is not in x86-64 macOS 26.6.2's kernel collections.
- Metal's base `targetDeviceArchitecture` answers `nil` for every device,
  and nothing reaches it from the host: the device info the plugin fetches
  (method 261, synchronously, as its device is made) feeds only its shader
  compiler. The binary archive's load reads the architecture alone.
- Nothing on the path asks the device first. IconRendering attaches its
  archive whenever the file is in its bundle, RenderBox makes it the first
  time one of IconRendering's functions is used, asking the device nothing
  (no family, no `supportsBinaryArchives`), and Metal's base
  `newBinaryArchiveWithDescriptor:` loads it without asking
  `supportsBinaryArchives` (which this device answers NO). RenderBox's
  `RB_USE_BINARY_ARCHIVE` covers only RenderBox's own library, and does not
  help.
- The device info the library sends comes from the same builder as Apple's
  x86-64 framework's, key for key, in the same order. Only the family keys
  differ from what an Intel Mac would send, since they carry the host GPU's
  families: key 28's bit 0 (Apple5), key 37 (1009 here, 1005 on a GPU of no
  Apple family) and key 44 (Apple5 to Apple9, none on an Intel or AMD
  GPU). The archive's path reads none of them. RenderBox asks the device's
  families for other choices (its depth buffer default, framebuffer reads);
  which of them follow from these keys is not traced.
- Metal takes its error mode from `METAL_ERROR_MODE` in the environment,
  then from the defaults (arguments, the app's domain, the global domain),
  then from `MTL_ERROR_MODE`. With none set it is a constant 6, which logs
  and asserts. No build variant, SIP, boot-args or AMFI condition changes
  it, and nothing a host or OpenCore supplies is among the sources.
- A command buffer's error report, which a GPU channel restart makes, never
  ends the process, in any error mode: the aborts do not cascade through
  the restarts.
- IconRendering's archive holds only Apple GPU slices, so a device that
  did answer an architecture would miss them too and compile from the
  library, as it does with the key.

Not measured: the key's effect on guests before macOS 26.

Without the key, the aborts also fault the GPU. Measured on an RTX 5080
Laptop GPU host on 2026-10-11 (macOS 26.6.1, one library build):
- An icon service process aborts 40-150 ms after it starts, with a command
  buffer it committed still queued on the host or being decoded. The host
  runs that exec (opcode 55) after the process has gone. By then the
  object-list entries it names read as empty, and the host's first lookup
  of one fails ("object N has an empty entry"). Objects the task had cached
  before still resolve.
- The guest answers each such fault with a channel restart ("Fault on
  channel: Exec", "GPU hang"), a `gpuRestart` report and a tailspin. In one
  cold boot, 5 aborts at login gave 5 faults, each 0.5-10 ms after its
  process ended (the crash reports' `procExitAbsTime` against the restart
  reports' stamps). That boot had no other restart.
- With the key on the disk before login, a boot of the same macOS and build
  had no exec fault. A guest whose key is written only after login faults
  at that login, and not after the key is written.
- The host makes an object the first time a command names it, from the
  entry it reads then (section 6.11 of docs/testing.md), as Apple's
  framework does. So Apple's host faults the same way whenever it runs the
  exec after the guest's teardown. A host that decoded faster would only
  narrow the window: some faulting execs started after their process had
  ended.
- Such an exec makes its pipelines from the emptied entries too. A
  function whose entry is empty raises ("pipeline function N not found:
  object N has an empty entry"), so the exec faults like the others, where
  a nil function would have Metal abort at a render descriptor with no
  vertex function, ending QEMU (docs/testing.md, section 6.18). So does a
  resource whose entry is empty in a concurrent compute encoder's
  UseResources, where Metal ends the process at a nil entry; the other
  resource lists take it as nil, which Metal skips (section 6.16). Answered
  nil, it ended QEMU in the RTX 3050 beta tester's log of a boot before the
  key was set, and at Setup Assistant on a new install on the RTX 5080
  Laptop GPU host.
- The guest empties the list in place, mid-exec. Measured on the RTX 5080
  Laptop GPU host on 2026-10-12, in 10 boots of a new macOS 26.6.2 install
  to Setup Assistant (no key), with the library logging each lookup: the
  task's whole list read as type 0 with offset all ones at the fault, and
  the same 1, 10 and 100 ms later, through the host's mapping and a fresh
  page-table walk alike; the list's pages had not moved. Of the first 64
  entries, 33 were filled when the exec started, 39-40 once it had made its
  resources, and 0-8 at the fault, 1-5 ms in. Execs (opcode 55) ran on
  child FIFO 1 and object deletes (40) on child FIFO 2, the deletes with no
  barriers; each exec there took 20-80 ms while its pipelines compiled
  cold, so FIFO 1 reached a process's last exec after its end. Each boot
  faulted 9-21 execs; in 2 of the 10 a render pipeline's function resolved
  to nil and QEMU ended.

### x86-64 macOS guests: thumbnail extensions

On an x86-64 macOS 26 guest, Quick Look's image and video thumbnail
extensions end when they first enumerate Metal's devices, and Finder shows
the file type's icon where the thumbnail would be. No setting inside the
guest avoids it, `METAL_ERROR_MODE` included, and nothing the host supplies
takes part.

Measured on an RTX 5080 Laptop GPU host with a macOS 26.6.1 guest on
2026-10-11, in three boots:
- `ImageThumbnailExtension` (under ImageIO's JPEG decode) and
  `AudiovisualThumbnailExtension` (under VideoToolbox) make the PCI
  device's `AppleParavirtGPUMetal` device, which asks the kext for the
  device info: method 261 of the shared user client. The kernel's sandbox
  refuses it (`deny(1) iokit-external-method AppleParavirtSharedUserClient
  261`), the plugin logs "Failed to get device info: e00002e2"
  (`kIOReturnNotPermitted`), goes on without the info, and its own
  assertion on the info's shader version aborts the process.
- The thumbnail agent answers "Could not generate a thumbnail" for each
  file the process held, and Finder shows the type's icon (an MP4 in
  Recents). Each boot's first Finder window listing a JPEG and two videos
  ended one of each extension.
- Across the app inventory (Safari, Photos, Preview, QuickTime, Messages,
  Mail and the rest) no other process had method 261 refused. The host's
  log has no line at that time.

Read from the guest's binaries and sandbox profiles:
- The extensions' profile, `quicklook-thumbnail-secure`, lets a user
  client of class `IOAccelSharedUserClient2`, which AppleParavirtGPU's
  shared user client is, call only methods 0 1 4 7 8 9 10 12 259 268. The
  kext's own methods are 256 to 262. `blastdoor`, `airlock`,
  `com.apple.CoreGraphics.CGPDFService` and
  `com.apple.audio.AudioConverterService.Hardened` carry the same list, so
  a process under one of them that enumerates Metal's devices ends the same
  way.
- The kernel applies the sandbox's filter before it calls the driver
  (XNU's `is_io_connect_method`), so the refused call reaches neither the
  kext nor the host, and Apple's own host ends these extensions the same
  way. The assertion is not `MTLReportFailure`, so Metal's error mode does
  not cover it.
- The profiles are on the sealed system volume. Letting the methods
  through means unsealing it, and the plugin would need 256 to 262 there,
  not 261 alone.

Not measured: Quick Look's preview panel for images and videos, and image
types whose decode does not reach Metal.

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

On EHCI the guest's input freezes. Measured on an RTX 5080 Laptop GPU host
with a macOS 26.6.1 guest on 2026-10-11:
- One device's input stops. After a completed interrupt transfer the guest
  never activates the next transfer descriptor of that device's queue: the
  queue head's overlay is done with its interrupt-on-complete bit set, and
  the next descriptor is an inactive dummy, so QEMU has nowhere to put the
  device's reports. QEMU completed the transfer and raised the interrupt,
  and the guest took it and cleared it. The port stays enabled, the
  controller keeps running, and the other device keeps working. Input comes
  back only when the device is unplugged and plugged again (QMP
  `device_del`, then `device_add`) or the guest restarts.
- The keyboard goes first. macOS has it report every 25 ms with no key
  pressed, about 40 completed transfers a second.
- It comes with heavy guest GPU and WindowServer work: app launches,
  Geekbench (about one freeze a minute), and a loop of the TV, Maps, Photos
  and screen saver apps (two freezes in 47 s). Input injected at 100 Hz with
  four busy guest processes gave none in 10 min (about 87,000 completed
  transfers). The host's load made no difference.
- This library's device shares no interrupt with EHCI: `apple-gfx-pci`
  raises only MSI (`msi_notify`), and EHCI was alone on its pin (GSI 19).
- On xHCI the guest binds `AppleUSBXHCIPCI`, on MSI, the driver of every
  Mac that runs macOS 26. The same app loop gave no freeze in 20 min (about
  50,000 keyboard reports).

Not measured: guests before macOS 26 on `qemu-xhci`.

### x86-64 macOS guests: sleep

Give an x86-64 macOS guest no ACPI S3:

```sh
-global ICH9-LPC.disable_s3=1
```

A guest that enters S3 does not come back with its display. Measured on
the RX 7900 XTX host with a macOS 26.6.1 guest (ReimsVgpu.kext, binary
version 245) on 2026-10-13:
- `pmset sleepnow` puts the guest in S3 about 30 s later, and idle sleep,
  with the system's sleep timer at 1 min, within 160 s of setting it:
  QMP's `query-status` says `suspended`. After idle sleep the window shows
  black. macOS schedules RTC alarms of its own before it sleeps
  (`MaintenanceWakeCalendarDate`, here 30 min ahead).
- No input wakes it. Mouse and key events through the window, and through
  QMP's `input-send-event`, leave it suspended: in QEMU only PS/2 input
  requests a system wakeup (`hw/input/ps2.c`); the USB keyboard and tablet
  on `qemu-xhci` do not.
- A wake (QMP `system_wakeup`, as an RTC alarm would) resets every device
  (`pc_machine_wakeup`, `RESET_TYPE_WAKEUP`): the library logs `device
  reset` and the guest display's destruction. The guest's kernel logs
  `ACPI S3 WAKE` and only unpauses the GPU's configuration space
  (`kConfigOpUnpaused`, `setPCIPowerState(2->2)`): its driver restores
  none of the device's state, asks for no binary version, and creates no
  guest display. `IOAccelEventMachine2::waitForStamp` then waits for good,
  the window stays black, and SSH times out in its banner exchange.

With S3 disabled, the DSDT has no `\_S3`. `pmset sleepnow` stops at a dark
wake (`sleepASAP 1`): the display turns off (`IODisplayWrangler` power state
1) and the VM keeps running; idle sleep stops at the same dark wake. Input
through the window turns the display on again (power state 4): in all 8
rounds measured, 7 of `pmset sleepnow` and 1 of idle sleep. While the display
is off the window keeps the last frame: the x86-64 guest sends no sleep
command (opcode 9), and WindowServer still presents the occasional frame.

Not measured: a host other than um780's.

### x86-64 macOS guests: audio

reims-ctl gives every macOS guest, x86-64 and arm64, a virtio sound device,
with one output and one input stream:

```sh
-audiodev pipewire,id=snd0 -device virtio-sound-pci,audiodev=snd0,addr=0x7
```

arm64 macOS's `AppleVirtIO` kext has `AppleVirtIOSound`, matching virtio
device 0x19 (read from macOS 27's `Info.plist`). Not yet checked: that
x86-64 macOS 26 has it too, and how either guest runs on QEMU's device.

`pa` in place of `pipewire` uses PulseAudio, and `none` gives the guest a
working output whose samples QEMU drops. macOS 26 for x86-64 ships no
Intel HDA driver (no `AppleHDA`), so `-device ich9-intel-hda -device
hda-duplex` leaves the guest with no audio device; `AppleUSBAudio`, which
it does ship, drives a USB audio device on the xHCI controller above
(`-device usb-audio,audiodev=snd0,bus=xhci.0`), output only.

Without an output device, Safari's Media Source players fail. Measured on
an RTX 5080 Laptop GPU host with a macOS 26.6.1 guest on 2026-10-11:
- YouTube streams VP9 video and Opus audio. The video decodes (WebKit's GPU
  process queued its first frames), then the audio renderer finds no
  device (`err=-101`), and WebKit reports a `DecodeError`. The page keeps
  loading, then shows YouTube's "An error occurred".
- A plain MP4 file plays, without sound.
- With `usb-audio` the guest lists "QEMU" as its default output, and the
  same YouTube video plays.

### x86-64 macOS guests: installers and recovery

A macOS installer or recovery image boots to its Recovery UI on
`apple-gfx-pci` because BAR 2 shows the EFI framebuffer ("BAR 2" in
"apple-gfx on reims-vgpu", above, and [architecture.md](architecture.md)
11.4). Without BAR 2 it stops on the Apple logo.

Measured on an RTX 5080 Laptop GPU host (an Intel Core Ultra 9 275HX,
CachyOS, KVM, `q35` with OVMF and OpenCore, headless) on 2026-10-11, with an
x86-64 macOS 26.6.2 BaseSystem (25G83), before BAR 2:
- The image ships no AppleParavirtGPU, so the one driver that binds the
  device (106b:eeee, class 0x0380) is IOGraphics' generic
  `IONDRVFramebuffer`, which takes the EFI boot framebuffer over only when
  it lies inside one of the device's memory BARs.
- The kernel's boot framebuffer was the option ROM's buffer in RAM
  (`initialize_screen: b=7DED4000` on the serial console), and the device
  had only its 16 KiB register BAR. The console logged `S20: Not usable`,
  then `newUserClient failed: 0 0xe00002be ... IONDRVFramebuffer` over and
  over: WindowServer aborted and launchd started it 18 times in 4 min,
  while the screen stayed on the Apple logo's progress bar.
- With `vmware-svga` in the device's place, whose framebuffer is its BAR 1
  (`b=80000000`), the same image reached its Recovery UI about 63 s after
  QEMU started.

With BAR 2 (`440da86693`, and the option ROM of `0b8914ff7d`), on the same
host, image and OpenCore, whose picker waits 45 s before it boots, so each
time below includes that wait, with QEMU started at nice 0:
- OVMF put BAR 2 at 0x80000000. From the ROM's write on, `info mtree` showed
  the alias as 0x80000000-0x8212bfff onto the RAM at 0x7ded4000-0x7fffffff:
  33 MiB, from the framebuffer to the end of the RAM below 4 GiB, since this
  OVMF turns on neither TSEG nor the SMBASE window. It stayed until the end of
  each run, 120 s after QEMU started.
- The installer's kernel took `initialize_screen: b=80000000`, and its
  Recovery UI was up at the 65 s screendump. The console has no `Not usable`
  and no `newUserClient failed`, and launchd started WindowServer once.
- The installed macOS 26 guest, on its disk behind `snapshot=on`, also booted
  on BAR 2 (`b=80000000`) and reached its desktop by the 65 s screendump, its
  Dock, wallpaper and menu bar shown. With `-d invalid_mem` QEMU logged no
  access to BAR 2: the guest's draws to its boot framebuffer after EFI goes
  off land in RAM. At `89fc1f56ba`, which emptied BAR 2 when EFI went off, the
  same boot made 90 writes to it in 0.7 s, at the rows of the boot progress
  bar, which QEMU dropped.
- With `efi-framebuffer-bar=off` the device has no BAR 2, so the ROM gives the
  RAM address, as before BAR 2 (`b=7DED4000`), and the same guest reached the
  same desktop by the 65 s screendump too.

Not measured: an installation run to its end, and installers and guests
before macOS 26.

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

Measured on 2026-10-12 on homelab-um780 (a Ryzen 7 7840HS, Zen 4, QEMU at
`871e0931`) with an x86-64 macOS 26.6.1 guest (25G76) on `snapshot=on`
overlays, behind an OpenCore whose `Cpuid1Data` is `0x00050654`, on the
RTX 3090:
- `query-cpu-definitions` under KVM: every model above lacks only `pcid`
  here (plus `hle` and `rtm` for Cascadelake-Server v1); on homelab-ms01 (a
  Core i9-13900H) Haswell-noTSX and Skylake-Client-noTSX-IBRS lack nothing,
  and Cascadelake-Server lacks TSX and all of its AVX-512.
- Haswell-noTSX and Cascadelake-Server each reached the desktop, SSH up at
  66 s; both report model 85 stepping 4 and `hw.cpufamily` `0x37fc219f`
  (Skylake), and only Cascadelake-Server sets `hw.optional.avx512f`.
- Cascadelake-Server's guest reads AMD's layout from CPUID 0xD: opmask at
  832, ZMM_Hi256 at 896, Hi16_ZMM at 1408 (Intel's are 1088, 1152 and 1664).
  In each of three runs, `k3` read 0 after a signal handler returned, and
  `thread_get_state(x86_AVX512_STATE64)` gave wrong zmm16, upper zmm1 and
  `k3`; setting that state back zeroed `k3` in the thread.
- Single-thread, Cascadelake-Server over Haswell-noTSX: Accelerate's sgemm
  and dgemm (1024) 37% and 34% faster, SHA-256 and SHA-512 about 20%, LZFSE
  encode 30% and decode 12%; FFT, vImage, memcpy and memset the same.

Not measured: a boot on Skylake-Client-noTSX-IBRS itself, and a boot behind
an OpenCore that leaves leaf 1 as it is.

### x86-64 macOS guests: VMX

`reims-ctl` adds `vmx=on` to `-cpu` where the host's KVM offers VMX to guests:
`kvm_intel` is loaded and its `nested` parameter
(`/sys/module/kvm_intel/parameters/nested`) reads `Y` or `1`, Linux's
default since 4.20. With it the guest's Hypervisor.framework works, so the
guest can run VMs of its own (Docker, UTM). AMD's KVM offers SVM instead,
which macOS does not drive, and HVF offers neither (`hvf_get_supported_cpuid`
masks VMX out, read from code), so there the command is unchanged and
`check` has nothing to drop. QEMU's CPU models leave VMX off, though they
carry its capability MSRs for when it is on. A VM's `-vmx` or `vmx=off` in
`cpu_flags` takes it away: QEMU applies `name=value` flags in order, as
global properties when the vCPU is made, and `+` and `-` ones after them,
at realize (`x86_cpu_parse_featurestr`, `x86_cpu_expand_features`, read
from code). A `+vmx` would outlive a `vmx=off`, the form QEMU's own docs
show, with only a warning of an ambiguous model string.

Measured on 2026-10-12 on homelab-ms01 (a Core i9-13900H, CachyOS's kernel
7.2.9, `nested` Y), with the x86-64 macOS 26.6.1 guest from qcow2 overlays
of its disks, booted twice: on Haswell-noTSX with `+vmx`, and on
Skylake-Client-noTSX-IBRS with `vmx=on`:
- QEMU warned of no missing feature under `check`, and the kernel logged
  nothing of KVM's.
- The vCPU had `vmx`, `vmx-ept` and `vmx-unrestricted-guest` (QMP
  `qom-get`).
- The guest reached its desktop, read `kern.hv_support: 1` with `VMX` in
  `machdep.cpu.features`, and shut down when asked.
- A paused QEMU on Skylake-Client-noTSX-IBRS gave the vCPU `vmx` with
  `vmx=on` and with `vmx=on,+ssse3`, and none with `vmx=on,vmx=off` or
  `vmx=on,-vmx`, with no warning.

Not measured: a VM run inside the guest.

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
and answering them changed nothing PerfPowerServices does (below).

Measured on 2026-10-13 on homelab-um780 (a Ryzen 7 7840HS, Ubuntu 26.04,
KVM), QEMU at `d080baa32` with and without `8d795445f2`, with the x86-64
macOS 26.6.1 guest (25G76) behind reims-util's OpenCore (`5216d98`), on
`snapshot=on` overlays, the guest's port I/O counted with bpftrace on
KVM's `kvm_pio` tracepoint:
- Without it: from about 30 s after boot, about 22,000 GET_KEY_BY_INDEX
  a second, each answered 0x82 after four status reads. PerfPowerServices
  ran at 100% (7 min 9 s of CPU in 9 min of uptime), every spindump sample
  in `getAllKeys`, `SMCGetKeyFromIndex`, `IOConnectCallStructMethod` and
  AppleSMC; QEMU took 119% of a host core with the desktop idle.
- With it, in two runs of a cold boot and a reboot in the same QEMU: each
  boot walked the index once or twice, 5 commands a walk, each walk ending
  at 0xb8, and the SMC saw no I/O once the guest was idle.
  PerfPowerServices, at 0.0%, used 0.59 to 0.81 s of CPU in its first 3
  to 4 minutes, and about 1.05 s in the 3 minutes after the reboot. QEMU
  took 15% to 25% of a host core once the guest's daemons had settled, and
  up to 50% in the first minutes after a boot. The guest sent GET_KEY_TYPE
  about 60 times a boot and WRITE a few times at boot and shutdown, and
  gave each refusal up after four status reads, as without it. It reached
  its desktop at each boot (a QMP screendump showed the Dock, the menu bar
  and the wallpaper), reading `NATJ` 0 and `MSSD` 3, and QEMU exited 8.3 s
  after a shutdown. The runs' builds differed from `8d795445f2` only in an
  include and a store of 0 over 0, which change nothing the guest sees.
- With a build that also answered GET_KEY_TYPE and WRITE, as an SMC does,
  and had `#KEY`: the walk ended the same way, and PerfPowerServices used
  0.6 s of CPU in its first 3 minutes. At boot the guest asked GET_KEY_TYPE
  of 55 to 63 keys the device lacks (0x84), and of `REV `, which it then
  read; it wrote `HE0N`, `MSDW`, `NTOK` and `QENA` (0x84), and at shutdown
  `NATi` (0x84) and `NATJ` (0, taken). It never read `#KEY`, nor asked the
  type of `#KEY`, `MSSD`, `MSSP` or `NATJ`, though the walk listed them.
- macOS sends no length byte after GET_KEY_BY_INDEX's index, nor after
  GET_KEY_TYPE's key: a build that waited for one left AppleSMC reading the
  status port about 33,000 times a second, each GET_KEY_TYPE until its
  timeout.

### x86-64 macOS guests: displays wider than 2048 pixels

Without ReimsVgpu.kext, an x86-64 macOS guest's WindowServer dies at boot,
often enough to matter, on a display wider than 2048 pixels. The guest then
comes up at the login window instead of logging in on its own, and later
shows "WindowServer quit unexpectedly". METAL_ERROR_MODE (above) does not
change it.

The OpenCore image reims-util makes carries ReimsVgpu.kext, which loads on
macOS 26 alone. Among its patches, it stops the plugin refusing the tiles
below (its guard in `-[AppleParavirtTexture initTiledWithBuffer:...]`), and
the host accepts them by the bytes they read: the slices before the last at
full strides, and the last to its furthest mip's last pixel, not a full
stride past it (`slice_footprint_fits` in core's
`resources/texture/dimension.rs`). Apple's framework counts full strides, as
the plugin does, so a guest with the kext on it would still die. An image
from an older reims-util has no kext: reims-ctl says so in the VM's details,
and `reims-ctl update-opencore` (`U` in its interface) replaces it with a new
one presenting the same Mac.

The cause is in the guest, between its WindowServer and its Metal plugin for
the PCI device, read from the code of macOS 26.6.1's (25G76) x86-64
SkyLight and plugin:
- SkyLight draws a window of at least 1024 × 1024 pixels' area in tiles of
  1024 × 1024 pixels, from the left and the top. Each tile is a texture
  made with `newTiledTextureWithDescriptor:offset:bytesPerRow:` over a
  no-copy buffer of the window's pixels, the buffer ending within a page of
  the window's last row. Its client storage debug option, off by default,
  would make private textures instead; arm64 SkyLight makes private
  textures.
- The plugin counts a tile's bytes as its offset plus its rows times the
  window's bytes per row, and refuses a tile that ends past the buffer. So
  every tile of the bottom row but the first is refused: the count runs
  past the buffer's end by the tile's offset in the row.
- SkyLight keeps the textures it got in a list, skipping the refused ones,
  and draws tile *n* with the list's *n*th entry, unchecked. With two tiles
  across (2048 pixels or narrower) the one missing entry falls inside the
  list's allocation; with three or more, the reads land past it, and
  WindowServer dies on a texture pointer into freed memory, in the plugin's
  `IOAccelResourceListAddResource`. Where its first command buffer is still
  queued, the guest kernel frees WindowServer's objects under it, and the
  host logs `task 1: exec faulted: resource 14 not found`, then sees a GPU
  restart; a death before it leaves no line on the host.

No answer of the host's reaches the plugin's count, so a guest on Apple's
framework would meet it too (not measured). With the kext, the tiles reach
the host, which counted them as Apple's framework does until its check took
their real extent.

Measured on 2026-10-11 and 12 with a macOS 26.6.1 guest, from fresh
overlays:
- WindowServer's log holds the plugin's refusal at each boot: at 2560 ×
  1600, allocations of 16388096 and 16392192 bytes against a buffer of
  16384000 (the bottom row's second and third tiles); at 1920 × 1080, one
  of 8298496 against 8294400.
- On homelab-um780's RTX 3090, windowed, counted by the host's line (so a
  death before the first command buffer is missed): 2560 × 1600 died at 4
  boots of 4, 2304 × 1440 at 2 of 2, 2176 × 1360 at none of 2, 2048 × 1280
  at none of 2; headless (`display-modes`), 2560 × 1600 at 1 of 3. On
  homelab-ms01's Iris Xe, windowed, counted by the guest's crash reports,
  2560 × 1600 died at about half its boots (13 of 29). 1920 × 1080 died at
  none of 14 across both hosts, windowed and headless.
- A tester's 2736 × 1704 (`resolution = "host"`, an Iris Xe under Linux)
  left the host's line in 2 of its 5 boots' logs.

With the kext, on 2026-10-13, on msi-laptop's RTX 5080 with the macOS
26.6.1 guest at `resolution=2560x1600` (it ran 1920 × 1200): on the host's
full-stride count, the host refused 533 textures, 530 of them tiles, and
WindowServer died; on the real extent, it refused none, and WindowServer ran
13 minutes with no crash report.

Not measured: what the guest shows of a refused tile at 2048 pixels or
narrower without the kext (the bottom-right tile of a large window), and an
x86-64 guest older than macOS 26, which the kext does not load on.

### x86-64 macOS guests: VMware SVGA

reims-ctl runs an x86-64 VM with `graphics = "vmware-svga"` on QEMU's VMware
SVGA II in apple-gfx's slot, which the library takes no part in, in QEMU's
own window:

```sh
-vga none -device vmware-svga,id=gfx,bus=pcie.0,addr=0x4 -display gtk
```

scripts/install-ctl.sh builds QEMU on Linux with GTK for it (before, with
neither GTK nor SDL, its `-display help` listed only `none` and `dbus`;
reims-ctl refuses to start such a VM on a QEMU whose list has no `gtk`,
until `reims-ctl update` builds one).
Where the environment names no Wayland compositor or X server, GTK's window
ends QEMU as it starts, so reims-ctl gives such a VM `-display none` there.
Closing the window ends QEMU at once (`gd_window_close`, `qmp_quit`), as
closing the library's does (apple-gfx's `apple_gfx_input_close`,
`qemu_system_shutdown_request`).

Measured on msi-laptop on 2026-10-13, with the macOS 26.6.1 guest on
reims-ctl's command line otherwise unchanged, and QEMU configured as
install-ctl.sh does (`GTK support: YES`, `-display help` adding `gtk`):
- OVMF and OpenCore drive the device, and macOS boots to its desktop at
  1280 × 720 on the framebuffer the firmware left, with no GPU acceleration:
  macOS has no driver for the device.
- The window, "QEMU (svga-test)", opened on Plasma's Wayland session and
  showed the guest's desktop. With the laptop's screen idle, KWin sends no
  frame callbacks, and the window keeps its first frame ("Guest has not
  initialized the display (yet)") until the screen wakes.
- QMP's `screendump` and `input-send-event` on device `gfx`, head 0, show
  the display and reach the guest's USB tablet, as `reims-ctl view` uses
  them.
- `system_powerdown` shut the guest down in 9 s.

Not measured: what sets the 1280 × 720 mode (OVMF's platform configuration
or OpenCore's `UEFI > Output > Resolution`), whether a larger one fits the
device's default 16 MiB of VRAM, and the window on GNOME or under X11.

### App Nap

macOS naps an application none of whose windows the person sees: QEMU's
window behind another application's, or hidden. While an `apple-gfx` device
is realized, QEMU holds a user-initiated activity, which keeps App Nap away
(`hw/display/apple-gfx-activity.m`, `1e9e1915cb`). The first device's
realize begins it and the last device's finalize ends it. Its options are
`NSActivityUserInitiatedAllowingIdleSystemSleep`, not
`NSActivityUserInitiated`, which would also keep the host from idle sleep
for as long as QEMU runs ([lessons.md](lessons.md)). In window mode the
library holds an activity of its own for the process's life
(`reims-vgpu-window`'s `native/macos/app.rs`). Under `-display cocoa`, or any display
whose windows are QEMU's, nothing else holds one.

**Measured** on 2026-10-08 on this host (an Apple M3 Max, macOS 26.7,
25G229), with the Vulkan platform, `-display cocoa` and macOS 13.6 to 26.6.2
guests, before the activity (`3ca83dca38`). The lines are runningboardd's
and launchservicesd's, from `log show`:
- RunningBoard names QEMU `anon<qemu-system-aarch64>` and starts it
  unnapped: `Set AppNap state: <... enabled:Y active:N socket:N disk:N
  priority:N cpu:N timer:Tier0>`.
- 30 to 50 s after QEMU last lost the focus (`Set darwin role to:
  UserInteractiveNonFocal`), it naps it: `Set AppNap state: <... enabled:Y
  active:Y socket:Y disk:Y priority:Y cpu:Y timer:Tier5>`.
- The focus ends the nap within the millisecond: `SETFRONT: pid=<QEMU's>`,
  then `Set darwin role to: UserInteractiveFocal` and `Set AppNap state:
  <... active:N ...>`.
- Napped, every named QEMU thread runs at priority 4 or below, base and
  current: the vCPUs (`CPU n/HVF`), `qemu_main`, the library's
  `reims-vgpu.fifo`, `reims-vgpu.display` and task threads, and
  metal2vulkan's. Metal's compiler service runs QEMU's compiles at priority
  0 to 4 (thread priorities from libproc's `PROC_PIDTHREADINFO`, every
  5 s).
- With the window in the background: a 26.6.2 guest booted with the Vulkan
  platform's shader cache empty napped from 32 s to 348 s and reached its
  desktop at 215 s. A 13.6 guest napped from 31 s to the end of its run,
  and a 14.6.1 guest from 76 s to 142 s. The runs the person kept bringing
  QEMU forward never napped.

With the activity (`1e9e1915cb`), on the same host, build and 26.6.2 guest,
booted with the Vulkan platform's shader cache empty and QEMU's window in
the background the whole run:
- RunningBoard never napped QEMU. 46 and 56 s after QEMU lost the focus it
  logged `Set AppNap state: <... enabled:Y active:N socket:N disk:Y
  priority:N cpu:Y timer:Tier0>`, and no `active:Y`.
- Of 5,633 thread samples, one was at priority 4. It was an unnamed thread
  10 s in, which later took the name `reims-vgpu.task-service` and ran at 31.
- The desktop came at 53 s, as it did in a boot of the guest kept in front.
  The pipeline states and calls that took 100 ms or more to make took 67 s
  in all, where the napped boot's took 482 s, and its longest gap between
  frames was 7.4 s, where the napped boot's was 46.5 s. Safari's video
  played. There was no GPU Timeout, where the napped boot had 10.
- QEMU came to the front four times, for under 3 s each: at its window's
  opening, 27 s in (a napped boot of the guest did the same 28 s in), and
  twice later. Each time the focus went back to the application in front
  before.

## Signing

QEMU's build signs `qemu-system-aarch64` ad hoc with
`com.apple.security.hypervisor` and no hardened runtime. The library is
linked into it statically, so that signature covers it, and dyld loads
nothing of the library from a file. Isolated tasks run in QEMU's process,
under the same signature.

- `cargo xtask dist` signs `libreims_vgpu.dylib` ad hoc, for hosts that link
  it (the suite's runner). An ad hoc signature is enough for a process
  without library validation. A host under the hardened runtime that loads
  the dylib by an `@rpath` name is **not measured**.
- Apple's own GPU task service carries a private entitlement for protected
  content, `com.apple.private.ggdsw.GPUProcessProtectedContent`, which only
  Apple's signature can carry: a trivial ad hoc binary signed with it is
  killed at exec (SIGKILL, exit 137; measured on this host, 2026-09-23).
  QEMU enables no protected content.
- Every Rust binary is arm64, as QEMU is ([architecture.md](architecture.md)
  D32).

## Booting a guest to check a change

A boot checks what no test can: QEMU's side of the API, and what guests do
with the device's answers (AGENTS.md, "Checking a change"). It is run so
that what it shows is the change's:

- **GPU work holds first.** Before the boot starts, everyone who may run
  GPU work is told to hold it until the boot ends: any test that opens a
  Metal or Vulkan device, and, when the boot's timings are compared with
  another's, local builds too. A cold boot compiles its pipelines on the
  CPU: a macOS 15.6.1 guest reached its desktop in 74 s beside other
  builds, and in 32 s alone (measured on this host, 2026-10-08).
- **Copies only.** A boot runs on an APFS clone, a copy or a `snapshot=on`
  overlay of the guest's disk, and a step that changes the guest's state (a
  `defaults write`, an app relaunched with new settings, a file written)
  runs only there; a harness refuses it on a guest's own disk, naming the
  disk. On a persistent disk such a step can leave a fault the guest never
  repairs: an x86-64 macOS 26 guest whose Dock was relaunched with new
  settings while its icon service was failing ("x86-64 macOS guests:
  Metal's error mode", above) saved an icon cache of blank icons, and its
  Dock never came back. A damaged disk is copied again from its source.
  Before blaming the guest or the platform for a broken disk, read what our
  own steps did on it.
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
- **A Metal control.** On a Mac, a guest that misbehaves on the Vulkan
  platform boots beside a Metal-platform control of the same guest and
  build, so the difference is known to be the platform's.
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

## Running a vmapple VM

What the machine needs (`docs/system/arm/vmapple.rst` in vendor/qemu):

- **A guest**: a VM installed with Virtualization.framework, for example with
  `macosvm`: its `disk.img`, its `aux.img` trimmed of its first 16 KiB block
  (`dd if=aux.img of=aux.img.trimmed bs=$((0x4000)) skip=1`), and the ECID
  from the base64 `machineId` plist in `vm.json` (`contrib/vmapple/uuid.sh`),
  which `-M vmapple,uuid=` takes in decimal. macOS 13 boots with
  `663b9b9c0b` (a boot without it is **unknown**), and macOS 14 and later
  boot too: the machine changes their kernel's memory and device tree as
  iBoot enters the kernel ("macOS 14 and later guests", below).
- **Boot ROM**: `-bios
  /System/Library/Frameworks/Virtualization.framework/Resources/AVPBooter.vmapple2.bin`,
  Virtualization.framework's, not this library's.
- **Interrupts**: `-accel hvf,kernel-irqchip=on`, which `4a2e2b3967` makes
  work. `kernel-irqchip=off` (the default) is **unknown**.
- **Option ROM**: none. `vmapple`'s `apple-gfx-mmio` has no ROM; the
  `ReimsVgpu.rom` QEMU installs is `apple-gfx-pci`'s, an EFI option ROM
  holding an x86-64 boot-service driver (`file`: "BIOS (ia32) ROM Ext.
  (11*512)"), built from `firmware/` ([architecture.md](architecture.md)
  11.4).

### The guest

The measurements of this file at `vmapple` name a guest by its version.
"The macOS 13.6 guest" is **macOS 13.6 Ventura, 22G120**, set up with
`macosvm` at 2560x1600: its System volume's `SystemVersion.plist` says
`ProductVersion 13.6`, `ProductBuildVersion 22G120`, and About This Mac in
the booted guest says "macOS Ventura 13.6", "Apple M1 (Virtual)", "Apple
Virtual Machine 1". It has finished Setup Assistant and logs its owner's
account in automatically. Its `hardwareModel` says `PlatformVersion 2`,
minimum OS 12.0.0. It is `/Users/aneesiqbal/Projects/macos-gpu-re/guest-13`
(`disk.img`, `aux.img`, `vm.json`), beside the 14.6.1, 15.6.1 and 26.6.2
guests in `guest-14`, `guest-15` and `guest-26`. `vm_drive.py`'s default
`GUEST` still names the directory it was in before, `guest`, which this
host no longer has: give a run its guest with `--vm`.

### Driving a guest over SSH

The first time something run over SSH controls an app (Apple Events:
`osascript`, System Events, UI scripting, screen actions), macOS shows a
privacy prompt on the guest's own screen, "sshd-keygen-wrapper wants access
to control <app>", with Don't Allow and OK (or Allow). The command blocks,
or fails, until the prompt is answered, and SSH cannot answer it. It comes
once per controlled app (System Events, Finder, Safari, ...), on every
guest version, x86 and arm64 alike, and again on a fresh clone. Answer it on
the screen through QMP: `send-key` `ret` when OK is the default button, or
an absolute pointer move and click on it (`input-send-event` through the
tablet), then take a screendump to see it gone. Where `open -a <App>` or
`open <URL>` does the job, use it: it sends no Apple Events and raises no
prompt. A script that drives a guest polls for the prompt after each
first-use call; an unanswered prompt shows only as a step that stalls.

### Steps

```sh
# 1. QEMU, on the checkout's library
make qemu

# 2. the VM: APFS-clones disk.img and aux.img into the clone directory (the
#    VM's own files are never written), trims aux, reads the ECID, and runs
#    QEMU with no DYLD variable
FRESH=1 scripts/internal/run_vmapple.sh path/to/macosvm-vm /tmp/vm
# or: make vm VM=path/to/macosvm-vm CLONE=/tmp/vm
# or, with a window for each guest display (window mode):
# WINDOW=1 scripts/internal/run_vmapple.sh path/to/macosvm-vm /tmp/vm
# in another shell, while it runs:
echo 'screendump /tmp/shot.png -f png' | nc -U /tmp/vm/monitor.sock
```

`make vm` builds QEMU first, then runs the script
(`scripts/internal/run_vmapple.sh`'s header has the details). It runs
`build/qemu/qemu-system-aarch64`, which holds the checkout's library.
`QEMU=` runs another build, such as a predecessor's. The script refuses a
QEMU that links `ParavirtualizedGraphics` at all, and one that neither
links the library by name nor defines `rvg_device_create`. The clone
directory must be on the VM's volume, for the APFS clones.

The script's header lists its options: `FRESH=1` re-clones, `PRINT_LIBS=1`
sets `DYLD_PRINT_LIBRARIES` for QEMU, `DISPLAY_OPT=cocoa` opens QEMU's own
window, and `MEM`, `CPUS` and `ACCEL` set the rest (default 16 GiB, 8 vCPUs,
`hvf,kernel-irqchip=on`); `RESOLUTION=` sets the guest's mode in window
mode ("Window mode", above). `WINDOW=1` has the library open a window for each
guest display under the default `-display none` ("Window mode", above);
otherwise the script runs QEMU with `REIMS_VGPU_WINDOW=0`, and a run stays
headless, as `vm_drive.py`'s tours need (it starts QEMU only through the
script). The library's `REIMS_VGPU_WINDOW_*` variables pass through. It
writes the serial console to `serial.log`, QEMU's stderr with the
`apple_gfx*` trace to `qemu.log`, and opens an HMP monitor
(`monitor.sock`) and QMP (`qmp.sock`), in `OUT` (default the clone
directory).

Measured facts about driving it (on this host, macOS 26.7, 25G229):
- **The pointer** is a USB tablet, which takes absolute events only: QMP
  `input-send-event` with `abs` axes (0 to 32767) moves and clicks it, and
  HMP's `mouse_move`, which sends relative motion, does nothing. HMP
  `sendkey` reaches the USB keyboard.
- **Scrolling** (2026-10-01, macOS 13.6 guest, the machine's default
  `usb-vmapple-tablet`, QMP `input-send-event` as the window's callbacks
  would send it, measured by an AppKit window in the guest logging its
  scroll events and its scroll offset). macOS binds its
  AppleVirtualPlatformHIDEventDriver to the device and turns its smooth
  scrolling on (feature report 3 set to 1). A swipe of 30 events of 6
  points at 120 Hz with its phases, then 40 momentum events: 73 precise
  scroll events in the guest at 121 per second, with the gesture's and the
  momentum's phases; the text view's offset moves 57 times in 0.57 s, a
  median 6 points every 8.4 ms, 284 points in all for the host's 269 (the
  momentum's 89 exactly; the gesture's 180 as 191). The same swipe as the
  notches `usb-tablet` gets, one per 40 points: 6 wheel events at 11 per
  second, offset jumps of 1 to 33.5 points 50 to 183 ms apart, 97.5 points
  in all, and no momentum. Horizontally, the same swipe moves the offset
  291 points in 58 steps. Five wheel notches 60 ms apart scroll 20 points
  each (`usb-tablet`: 5.5 to 38.5, as macOS accelerates them). Dragging is
  the same on both: 60 moves at 120 Hz give 59 drag events (`usb-tablet`)
  or 60, a median 8.3 ms apart, 5 points each.
- **The first report after a reset**, and the first report after a scroll
  report that returns the scroll fields to 0, are scrolls to macOS, whose
  pointer and buttons it ignores: `usb-vmapple-tablet` sends a report of
  nothing first, and a report that returns the fields to 0 before a pointer
  report. Without them, the pointer's first move after the guest starts, or
  a click right after a scroll, was lost.
- **`DYLD_PRINT_LIBRARIES`** has to be set on QEMU's own exec: set in the
  calling shell it is dropped, since the script runs under `/bin/bash`, a
  platform binary, which starts with `DYLD_*` variables removed from its
  environment (`DYLD_PRINT_LIBRARIES=1 FOO=1 /bin/bash -c env` prints `FOO`
  alone).
- **Beside the suite.** A VM never runs beside the behaviour-lock suite. The
  suite's cases fault the GPU and crash processes on purpose, and a VM beside
  them gives spoiled results (runs that overlapped a suite run on 2026-09-26
  were thrown away). Two VMs can run at once: they do not disturb each other.

### Running the vmapple VM on reims-vgpu

`scripts/internal/vm_drive.py` boots the guest from fresh APFS clones and
drives it through a fixed tour on its own, with nothing else on the GPU
("Beside the suite", above). Clicks go through QMP `input-send-event` at text
Vision finds on the screendump (`scripts/internal/vm_ocr.swift`), or at the
Dock's icons. Keys go through HMP `sendkey`. The tour has 22 steps:

- the desktop, reached by auto-login;
- Keyboard Setup Assistant's Quit;
- Launchpad;
- Maps: its "What's New" and notifications sheets, the map of Estonia, the
  globe (ten Cmd-minus), a 200x60 drag, two Cmd-equals, two Cmd-minus, each
  left 15 s to load tiles;
- Photos and its "What's New";
- Safari's start page;
- System Settings, searched for Displays, then 1920x1080 and back to
  1440x1080;
- a Finder window dragged by its toolbar;
- Mission Control.

Each step is timed, and counts the frames (`apple_gfx_new_frame`), doorbells
and interrupts in QEMU's trace. The step waits for what it expects on
screen. If the screen stays still with no new frame for 180 s, or for the
whole of its timeout, that is a hang, and the driver saves `sample` of
QEMU, the trace's tail and a screendump under `hang/`. QEMU's os_log lines
are collected by its PID: they are the system frameworks' (Metal's among
them), since the library logs through the `log` crate and writes none there.
Crash reports are matched by the PID in the report.

`scripts/internal/vm_drive.py run --out DIR [--label L]` boots one VM. `pair
--out-base DIR --index I LABEL[=QEMU] LABEL[=QEMU]` boots two
at once, each into `DIR/LABEL-I` and each driven on its own: a change's QEMU
against its predecessor's. A QEMU defaults to `make qemu`'s build; a
predecessor's comes from `make qemu` in a worktree of its own ("The QEMU
build", above). A run's label (default: `vm`) names it in
`result.json`, and `summary`, `perf` and `logdiff` tell runs apart by it;
`perf`, given runs of two labels, compares them. `compare DIR_A DIR_B`
diffs two runs' screendumps step by step: the pixels that differ by more
than 16 in any channel below the menu bar, and Maps' placeholder grid in
each settled Maps frame. For example:

```sh
scripts/internal/vm_drive.py pair --out-base /tmp/vm-pairs --index 1 --perf \
  change parent=/tmp/parent-wt/build/qemu/qemu-system-aarch64
scripts/internal/vm_drive.py compare /tmp/vm-pairs/change-1 /tmp/vm-pairs/parent-1
scripts/internal/vm_drive.py perf /tmp/vm-pairs/change-* /tmp/vm-pairs/parent-*
```

**`--perf`** adds `perf_tour` after the tour, bursts of interactive input:

- a Finder window dragged by its toolbar in two circles (pointer events at
  60 Hz for 3 s, three times), then resized from its bottom-right corner,
  300x200 px in and out three times;
- Mission Control in and out six times (Ctrl-Up every 0.9 s);
- Launchpad: three swipes each way, 0.6 s apart (Launchpad has one page on
  this guest, so each swipe rubber-bands), then a search typed and erased
  and the Other folder opened and closed;
- Safari on a long page served from the host: `long_page`, 400 sections of
  text over gradient blocks with shadows, at `http://10.0.2.2:<port>/`
  through QEMU's user network, so it needs no Internet. It is scrolled by
  150 wheel clicks at 60 Hz down and back up, twice, then by ten Space and
  ten Shift-Space at 0.3 s;
- Maps, reopened on the tour's globe: six 300 px pans at 60 Hz, 20 wheel
  clicks in and out at 60 Hz, twice, and its zoom buttons four times each
  way.

Each burst records the wall-clock time of its first and last input. The
continuous bursts (the drags, the resize, the scrolls, the pans and the
wheel zooms) send pointer events at 60 Hz, and every event should bring a
frame. The keyed ones (Mission Control, Launchpad, Space, the zoom buttons)
leave the screen still between animations.

A `--perf` run also records:
- QEMU's trace with `-msg timestamp=on`, which prefixes every `apple_gfx*`
  line with an ISO 8601 UTC time in microseconds (`util/log.c`,
  `qemu_log_trylock_with_context`); the trace backend is `log`, which
  prints no time of its own. QEMU runs with `-name
  vm-MODE,debug-threads=on`, which names its threads ("CPU 0/HVF",
  "qemu_main", "worker"; the library's are `reims-vgpu.*`).
- CPU three ways: QEMU's `rusage` at exit (`wait4`); the CPU of each live
  thread of QEMU just before quit (libproc `PROC_PIDTHREADINFO`), which
  separates the vCPU threads, QEMU's own and the library's; and `ps -o
  time` of QEMU every 2 s.
- QEMU's os_log lines, by PID (ndjson), and the guest's network traffic
  (`filter-dump`).

### How the figures are defined

`vm_drive.py perf` reads each run with `analyse`:

- *Doorbell to interrupt.* A FIFO doorbell is a write of `n` to 0x1020,
  which wakes child FIFO `n` (1 to 4 on this guest). The host publishes a
  stamp by setting bit `stampIndex` in the read-to-clear mask at 0x1018 and
  raising the interrupt. After every interrupt the guest's handler reads
  0x1018, 0x1014 and 0x102c. The guest gives child `n` stamp index `n`, so
  every bit it reads belongs to a FIFO it rang.

  The rule: a doorbell of FIFO `n` is matched to the first read of 0x1018
  after its line with bit `n` set. Its latency ends at the last
  `apple_gfx_raise_irq` before that read. Doorbells rung again before the
  match share it.

  QEMU prints a write's line only after `rvg_device_mmio_write` has
  returned (`apple_gfx_write`, on the vCPU's thread), while a FIFO's thread
  may raise the interrupt, and print its line, before that. So the
  interrupt can be printed before the doorbell it completes, when the host
  finished first. Two rules handle that:
  - *lenient* (the default figures): such a doorbell counts 0. The trace has
    no stamp values, so this is the time to the next stamp the host
    publishes for that FIFO, which may be the FIFO's earlier work. It is a
    lower bound.
  - *strict*: such a doorbell waits for an interrupt printed after its own
    line. This is an upper bound, and its tail is an artifact: a FIFO that
    goes idle leaves the doorbell waiting for its next work, seconds later.

  "From the desktop" figures count only the doorbells after the boot step
  reached the desktop.
- *Frame interval.* The time between consecutive `apple_gfx_new_frame`
  events inside a burst, from its first input to its last. The event comes
  from the new-frame handler of display 0, the only display. Launchpad's
  swipes pause, so they count with the keyed bursts.
- *Host silent* (a stall). A doorbell of FIFO `n` followed, for more than
  100 ms, by no interrupt at all and no read with bit `n`. Overlapping
  intervals are merged. Each is listed with its tour step, the trace events
  inside it and QEMU's os_log events within 0.5 s. A *slow FIFO* interval
  is one where the matched latency exceeds 100 ms while other FIFOs may
  progress.
- *MMIO round trip.* The spacing of adjacent trace lines that the guest
  issues back to back: two doorbells in a row (the guest rings a batch), and
  0x1018 then 0x1014 then 0x102c in the interrupt handler. The guest's own
  instructions between the two accesses are the same whatever the build, so
  a difference between two builds is the host's MMIO path and its return to
  the vCPU.
- *Quiet seconds.* The guest downloads on its own ("What the guest does on
  its own", below), which loads the guest and QEMU whatever the build. The
  *quiet* figures keep only the seconds, ±2 s, in which neither guest of the
  pair moved more than 1 MB. The pcap has its own clock (QEMU's virtual
  clock plus a start time, `net/dump.c`); it is put on the wall clock by the
  guest's first connection to the page server, which the driver makes at a
  known time.
- *Statistics.* The per-boot values of two labels are compared with an
  exact two-sided Mann-Whitney test over the boots. For the pooled doorbell
  latencies, a 95 % bootstrap interval of the difference resamples whole
  boots, 400 times.

### What the guest does on its own

Measured on the macOS 13.6 guest over the tour's boots of several builds
(at `85023009`, 20 boots; `aaede98d`, 18; `e7581f16`, 10; `0342ccf72`, 6).
None of these is the build's doing: each came on more than one build, or
with the order two QEMUs started in.

- **Frames that differ between any two boots**: the desktop's first frame
  (the wallpaper not yet drawn, the Dock still rising); Ventura's wallpaper,
  the day picture on one VM and the dusk one on the other until the switch
  to 1920x1080 redraws it; Software Update's badge on System Settings' Dock
  icon (about 406 to 410 px), which comes to one VM before the other;
  Launchpad's live icons (Calendar, Clock) and its search field's caret;
  Keyboard Setup Assistant's Quit button highlight; Safari's scroll position
  in a burst's screendump; and Maps' globe, whose tiles stream in at
  different levels of detail and which sits a few pixels apart after the
  zoom keys. Two boots of one build differ in 18 137 to 162 278 pixels at
  `maps-globe-loaded` (four pairs of boots at `85023009`).
- **Maps' placeholder grid.** On 13.6 Maps draws a tile it has no data for
  as a grid: a cream fill (249, 245, 237), fine lines of (236, 232, 224) and
  black meridians. `vm_drive.py` counts those colours in the map area of
  each settled Maps frame. At `85023009` every frame whose map drew held
  1 001 or fewer; a gap is 5 000 or more. A globe drawn all in placeholder
  held 284 097, in the stall below.
- **A stall with the host idle**, in 1 of 20 boots at `85023009`. After ten
  Cmd-minus the globe came up all grid, and nothing on screen changed
  again: in the next 122 s the trace counted no `apple_gfx_new_frame`, no
  FIFO doorbell (`apple_gfx_write offset=0x1020`) and no interrupt, and the
  menu bar clock stopped. The guest was alive: it went on ringing the
  display doorbell (`offset=0x1220`) with `apple_gfx_cursor_move` as the
  pointer moved. The trace's last GPU exchange was a normal one: the guest
  woke FIFOs 0 and 1, QEMU raised the interrupt, and the guest read the
  stamp mask (stamp 3), the display mask (0) and the fault flag (0). On the
  host every FIFO thread was waiting for guest bytes in its read of the
  ring, so the host had processed everything the guest had queued. The same
  signature, no frame and no FIFO doorbell while the cursor still moves,
  came in Setup Assistant on a macOS 12 guest, on another build. What the
  guest waits on is **unknown**.
- **Software Update.** The guest runs it in the background (DNS for `gdmf`,
  `mesu`, `swscan` and `swcdn.apple.com`), and at a random time starts
  fetching about 8 GB from Apple's CDN (`cdn-h3.g.aaplimg.com`,
  `get-bx.g.aaplimg.com`), at about 40 MB/s: in 12 of 18 boots at
  `aaede98d`, and 2 of 6 at `0342ccf72`. It loads the guest and QEMU (slirp
  in `qemu_main`, the block layer's "worker" threads, vCPU 0): QEMU's CPU
  besides the vCPUs was 108 to 120 s in the boots that downloaded and 61 to
  71 s in those that did not, at `aaede98d`.
- **A log line of the second QEMU.** When two QEMUs start together, the one
  that reaches Metal's shader cache second logs `list file already locked
  (<private>)` (libCoreFSCache, no subsystem), 2 to 6 ms after Metal opens
  the cache: in every pair at `85023009` and `aaede98d`, and never in a
  QEMU started alone. `logdiff` shows it for whichever VM of a pair started
  second.

### Every vCPU of the guest (2026-10-01)

**Measured on the macOS 13.6 guest at `-smp 8`, on the M3 Max (macOS 26.7,
25G229), with `-accel hvf,kernel-irqchip=on`. Without `vendor/qemu`'s
`c912618391`, the guest runs on vCPU 0 alone, and vCPUs 1 to 7 each use a
host core for the whole run. With it, all eight come up, share the guest's
work and idle near 0.**

**What went wrong** (at `450e03eed2`):
- vCPUs 1 to 7 each used 100 % of a host core (`ps -M`), and vCPU 0 about
  9 %. CPU 1 was parked in XNU's exception-stack check (`b.lt .`, DAIF
  masked) after a panic, and CPUs 2 to 7 in `ldxr; wfe` behind it. Neither
  loop exits to QEMU.
- CPU 1's first exception frame was an instruction abort (ESR `0x86000004`)
  at a corrupted pointer: the `retaa` that ends `machine_load_context()`
  failed to authenticate the `thread_continue` that `machine_stack_attach()`
  had signed on CPU 0 with the IA key (FEAT_PAuth2 corrupts the pointer on
  a failure; the host has FEAT_FPAC but not FEAT_FPACCOMBINE, so `retaa`
  does not trap).
- SCTLR_EL1 enabled all four keys on every vCPU. The ten AP*Key registers
  held one value on vCPU 0 and zero on vCPUs 1 to 7 (read through the
  gdbstub). vCPU 0's keys went from 0 to that value while AVPBooter ran and
  never changed again; the secondaries kept QEMU's reset zeros; QEMU never
  wrote a stale key back. Traced from reset, QEMU saw only PSCI's HVCs and
  two of `0x8300feff` (OEM FEATURES): none of the PAC calls reached it.

**Hypervisor.framework answers vmapple's PAC calls itself.** A probe (one
vCPU at EL1, MMU off, `hvc #0` then a store to an unmapped IPA, so a call
HVF answers itself shows only as the store's data abort, with the registers
as HVF left them) measured, under the public API:
- `hv_vcpu_run()` does not return for an `hvc #0` with x0 in `0xC1000000`
  to `0xC10000FF`.
- The PAC calls (`0xC1000000` to `0xC1000006`, and also `0xC1000007` and
  `0xC10000FF`) return x0 = `0xffffffff`. They change neither x1 to x5 nor
  any AP*Key register.
- `PAC_NOP` (`0xC10000F0`) returns 0.
- It also answers Apple's OEM UID (`0x8300FF01`, which returns the UUID
  `3B878185-AA62-4E1F-9DC9-D6799CBB6EBB` in x0 to x3) and REVISION
  (`0x8300FF03`, 1.0).
- OEM FEATURES (`0x8300FEFF`), `GET_MABS_OFFSET` (`0xC3000003`),
  `0x8100FF01` and PSCI exit to the VMM.

So QEMU cannot implement XNU's paravirtualized PAC interface
(`doc/vmapple_pac.md` and `osfmk/arm64/hv_hvc.h` at xnu-8796.141.3). The
guest runs on whatever keys each vCPU holds. Measure it again after a macOS
update.

**AVPBooter seeds the keys, on the boot vCPU only.**
`AVPBooter.vmapple2.bin` writes all ten AP*Key registers, with its `msr`s
at `0x102ff8` to `0x10304c`, using one 32-bit value repeated. The value
differed between QEMU processes, and within one process stayed the same
across HMP `system_reset`. XNU on vmapple never wrote them.

**The fix.**
- `f116a9d560` gives Arm CPUs an opt-in property,
  `cpu-on-copies-pauth-keys`. A CPU that has it starts with the AP*Key
  registers of the CPU that powers it on through PSCI CPU_ON, instead of
  the reset value. The caller's keys are read, after its state is
  synchronized, when it makes the call.
- `c912618391` sets the property on `vmapple`'s CPUs.

Both commits apply unchanged to upstream master at `9c88811b81`. The calls
themselves stay unimplemented, as at `-smp 1`: the per-process B keys and
the EL0 diversifier have no effect, and the kernel and every process share
the firmware's one set of keys. Only Hypervisor.framework's private mode,
which needs an Apple-private entitlement, provides the calls.

**Results**, before at `450e03eed2` and after at `c912618391`, one
`vm_drive.py run --perf` each, with nothing else on the GPU:

| | before | after |
|---|---|---|
| guest `hw.ncpu` / `hw.activecpu` | 8 / 2 | 8 / 8 |
| idle guest, per vCPU thread (after the guest settled) | 7 at 100 % | 2 % or less |
| eight shell busy loops in the guest | not run | 100 % per vCPU thread |
| tour and bursts | done, no hang, 410 s | done, no hang, 391 s |
| desktop reached | 18 s | 7 s |
| QEMU's CPU over the run (`rusage`) | 3 056 s | 247 s |
| `-smp 1` | | desktop; `hw.ncpu` 1 |

- With the fix, HMP `system_reset` and a restart from the guest's Apple
  menu each brought all eight CPUs back up, in one QEMU process, with the
  keys equal on every vCPU. The CPU that powers a secondary on is not
  always vCPU 0: in one boot vCPU 3 started vCPU 7, which then shared the
  keys.
- XNU made no PSCI CPU_OFF in any of these boots, idle or loaded, so no
  CPU was powered on a second time without a reset.
- **Eight `yes > /dev/null` in the guest** gave each vCPU thread only
  about 39 % of a host core, although the guest's `top` showed each `yes`
  at 99 %. Eight pure shell loops gave 100 % each. The eight `yes` write
  to one vnode; what holds their vCPUs back is **unknown**.

**Unknowns.**
- A PSCI CPU_OFF followed by CPU_ON under HVF. `arm_set_cpu_on()` resets
  the target's `CPUARMState` but marks nothing dirty, so the reset state,
  keys included, may not reach HVF when the vCPU's state was not dirty. The
  guest never did this, so it is not measured.
- Guests other than macOS 13.6, and `-smp` values other than 1 and 8.

### macOS 14 and later guests (2026-10-01)

**Measured on macOS 14.6.1 (23G93), 15.6.1 (24G90) and 26.6.2 (25G83)
guests, installed with `macosvm` and booted at `-smp 4` with 8 GiB from fresh
APFS clones, on the M3 Max (macOS 26.7, 25G229, SIP enabled). Without
vendor/qemu's `e1c37ea4d4` none of them boots: each stops in its kernel's
first instructions, in a loop that the public Hypervisor.framework API gives
QEMU no way to end. With it, 13.6, 14.6.1, 15.6.1 and 26.6.2 reach their
desktops within 25 s of QEMU's start, in screenshots taken every 5 s, while
other work loaded the host's CPUs during some of those boots. The hardware
model is not the cause.**

**How it fails.**
- `serial.log` ends at `End of iBootStage2 serial output`, as the 13.6
  guest's does, whose kernel writes nothing to the serial console. iBoot's
  banner names the same platform as on 13.6: `vma2ap`, board 0x20, CPID
  FE00. Every guest runs the same `AVPBooter.vmapple2.bin`, the only macOS
  booter Virtualization.framework ships.
- vCPU 0 then uses a whole host core and never exits. The other vCPUs stay
  powered off, and the display stays at "Guest has not initialized the
  display (yet)".
- HMP `info registers` and `xp /i` show vCPU 0 at the `cbnz`, with
  x0 = `0xffffffff` and the MMU still off:

  ```
  msr  vbar_el1, x0
  mov  x0, #0xc1000000
  hvc  #0
  cbnz x0, .
  ```

  15.6.1 and 26.6.2 stop at the same four instructions.

**Why.**
- `0xC1000000` is SET_INITIAL_STATE, the first of vmapple's
  paravirtualized PAC calls, which Hypervisor.framework answers itself,
  with x0 = `0xffffffff` and no exit ("Every vCPU of the guest", above).
- The vmapple kernel of 14.6.1 (`kernelcache.release.vma2`) checks the
  result of every PAC call and spins if it fails. It has 19
  `hvc #0; cbnz x0, .` sites, found in the kernelcache and in guest RAM:
  - SET_INITIAL_STATE in every CPU's start code, with no condition;
  - GET_DEFAULT_KEYS (`0xC1000001`), inlined 8 times behind a once-only
    flag;
  - SET_B_KEYS (`0xC1000003`) and SET_EL0_DIVERSIFIER (`0xC1000004`) in
    the context switch, called when the incoming thread's key differs from
    the CPU's;
  - SET_EL0_DIVERSIFIER_AT_EL1 (`0xC1000005`) in the helpers that enable
    and disable the user JOP key.
- The 13.6 (22G120) kernel has 18 PAC call sites and checks none of them.
  It runs on whatever keys each CPU holds.
- **Nothing gates the calls.** No site reads a device-tree property, an ID
  register, a `vmapple-cfg` field or a boot-arg. The only conditions are the
  once-only flag and the changed key. The `DeviceTree.vma2macosap` template
  has no PAC property.
- **The hardware model does not decide it.** The 13.6 guest's
  `hardwareModel` is `DataRepresentationVersion 1`, `PlatformVersion 2`,
  minimum OS 12.0.0, and 14.6.1's is the same with minimum OS 13.0.0. Their
  `aux.img` headers match: version 1, with the same `HUFA` block at
  16 KiB. 15.6.1 and 26.6.2 (`DataRepresentationVersion 2`, an `aux.img`
  header of version 2) stop at the same spin. Virtualization.framework
  boots all of them (`macosvm -g`), so it gets the calls answered. **How**
  it does so is **unknown**: it was not measured.

**What the public API allows**, measured with probes:
- `_hv_vm_config_set_isa()`, a private export of Hypervisor.framework,
  takes 1, the default. With 2, 3 or 4, `hv_vm_create()` fails with
  `HV_DENIED` (`0xfae94007`) under `com.apple.security.hypervisor`. The
  "enable private ISA" patch of the RFC series that boots macOS 26 guests
  uses ISA 3 and the `com.apple.private.hypervisor` entitlement. Only
  Apple-signed binaries hold that entitlement, so it needs SIP disabled and
  AMFI bypassed. That series marks the patch as not for merging.
- With EL2 enabled (`hv_vm_config_set_el2_enabled()`, supported on this
  host), HVF still answers an EL1 `hvc #0` that carries a PAC function ID,
  with `0xffffffff`. Other HVCs, PSCI_VERSION for one, reach the guest's
  EL2 as EC 0x16.
- `smc #0` with a PAC function ID exits to the VMM (EC 0x17).
- `_hv_vcpu_config_set_vmkey()` changes nothing about the calls.

**The calls are the only blocker.** In one experiment, done through QEMU's
gdbstub with `Qqemu.PhyMemMode:1` and not something the scripts do, the
`cbnz x0, .` after each of the 19 `hvc`s was replaced with a `nop` while
vCPU 0 spun at the first one. 14.6.1 reached its desktop about 15 s later.
`hw.activecpu` was 4 and `kern.osversion` 23G93, and apple-gfx drew on
reims-vgpu. The calls then fail as they do on 13.6:
- every process and the kernel share the firmware's one set of keys;
- the per-process B keys and the EL0 diversifier have no effect;
- GET_DEFAULT_KEYS leaves the kernel the caller's x2 and x3.

In that boot, `top -c d -stats csw` counted the context switches over 10 s.
This is an upper bound on the SET_B_KEYS and SET_EL0_DIVERSIFIER calls,
which run only when the key changes:
- the idle desktop, with Terminal: 13 940, about 1 400 a second;
- `yes | cat > /dev/null`: 2 405 380, about 240 000 a second.

**Booting them.** `e1c37ea4d4` makes two changes to every vmapple guest
when iBoot enters the kernel, with no option to turn them on or off. QEMU
catches that moment by keeping guest RAM non-executable from reset: each
page iBoot runs is made executable on its first fetch, until a fetch lands
between the boot arguments' `physBase` and `topOfKernelData` with x0 at
those boot arguments, as iBoot enters the kernel. iBoot's fetch faults
before that were 32 on 13.6, 33 on 14.6.1 and 15.6.1 and 44 on 26.6.2, all
outside that range, which on one boot of 15.6.1 ran from 0x70a04000 to
0x785ac000.
- Every `cbnz x0, .` that follows an `hvc #0` in that range becomes a
  `nop`: 19 on 14.6.1, 15.6.1 and 26.6.2, none on 13.6. The guests then run
  on the firmware's keys, as above.
- If the device tree that the boot arguments point to has no `/filesystems`
  node, the machine appends one, holding `DoNotUseSEP` = 1, as the root's
  last child. 15.6.1 and 26.6.2 need it. Without it, each panics with
  `"unencrypted data volume is not allowed" @apfs_vfsops.c:2292` (2372 on
  26.6.2) within 3 s of QEMU's start, before its display is initialized,
  then resets and boots again, over and over. With `-action
  reboot=shutdown,shutdown=pause`, QEMU stopped at that reset, and the
  panic string was in guest RAM. 13.6's tree has a `/filesystems` node,
  without the property, and is left as it is; 14.6.1's, 15.6.1's and
  26.6.2's have none.
- The node, 96 bytes, goes in the zeroes after the tree within its last
  16 KiB page; the machine walks the tree to find where it ends. On 13.6,
  14.6.1 and 15.6.1 the boot arguments' `deviceTreeLength` ends where the
  tree does, and grows by the node. On 26.6.2 it is rounded up to the page,
  0x8c000 for a tree that ends at 0x89c8c, and the page after it holds
  data: the node goes inside that length, which stays as it was. The guests
  boot with `/chosen/memory-map`'s `DeviceTree` entry left at the old
  length.

**Unknowns.**
- Whether ISA 3 boots these guests under QEMU. It needs SIP disabled, so it
  was not tried.
- What else in the guest relies on the per-process PAC keys that the
  `nop`s leave it without, or on the Secure Enclave that `DoNotUseSEP` tells
  APFS is absent.
- Whether macOS 12 guests still boot. The changes apply to every guest, and
  none was tried.
- What a VM exit per PAC call would cost under the context-switch rates
  above.

### macOS 27 guests (2026-10-14)

**Measured on a macOS 27.0.1 (26A434) guest, installed by reims-ctl and
booted with its command (4 vCPUs, 8 GiB) on the M3 Max (macOS 27, 26A434).
Without vendor/qemu's `d235e51310` and `fbda9c8b28` the guest shows a black
screen; with them it reaches Setup Assistant, then its login window, on the
library's Metal platform. A macOS 26 guest still reaches its desktop.**

**How it fails.**
- iBoot drives the display as on macOS 26, and the kernel starts: the
  machine's changes for macOS 14 and later ("macOS 14 and later guests",
  above) apply at the same 19 sites.
- QEMU links each virtio device's capabilities from high offsets to low:
  MSI-X at 0x98, then 0x84, 0x70 and on down. Each virtio device of the
  guest, both disks and the network card, reads the PCI_CFG capability at
  0x84, then the notify capability at 0x70 and a byte of BAR 4, then starts
  over, about 100 000 config reads a second, for as long as the VM runs.
- No device's status leaves 0, the root disk sees no I/O, and the kernel
  never reaches its GPU driver. `page-per-vq=on` changes nothing.
- macOS 26 walks the same list once and goes on.

**Why**, as the change's co-author found it, not measured here: macOS 27's
virtio PCI transport has a cached and a fallback capability lookup, which
disagree on a list linked from high offsets to low.

**Booting them.** `d235e51310` relinks the capabilities in ascending
address order, keeping each one's offset and contents: always on
`vmapple-virtio-blk-pci`, and on `virtio-net-pci` when its
`x-ascending-capabilities` property is on. `fbda9c8b28` turns that property
on in `vmapple`'s defaults, so a VM's command line needs no option.
`vmapple-virtio-blk-pci` also gains Apple's vendor capability: virtio
cfg_type 9, vendor 0x106b, 14 bytes, with its transport encryption disabled
flag set. The co-author found a root disk without FileVault needs it.

**Unknowns.**
- Whether the vendor capability is needed beyond the order: it was not
  tried apart from it.
- The co-author's patch also mapped an AES device at 0x30240000 and a stub
  it names RRTC at 0x30260000, and widened `vmapple-cfg`'s window to 1 MiB.
  vendor/qemu carries none of them: the guest booted without them, and
  macOS 26 makes the same accesses to those addresses.

## Unknowns

- A VM workload that reaches an isolated task's XR10 blit pipelines, a
  two-plane XR10+A8 texture, which the library's own shaders draw.
- Whether an isolated task fails a guest request when Maps opens. Earlier
  task services, and the system framework's, logged `Failed to create
  Fence` there, 3 to 12 times a boot, and the guest resumed a faulted child
  FIFO as many times. Which guest request fails is **unknown**, and so is
  whether it matters to what Maps draws.
- The stall above ("What the guest does on its own"): what the guest waits
  on.
- A host under the hardened runtime that loads `libreims_vgpu.dylib` by an
  `@rpath` name.
- On `vmapple`, guests other than macOS 13.6 under the tour, and workloads
  heavier than the tour `scripts/internal/vm_drive.py` drives.
- `kernel-irqchip=off`, and a boot without vendor/qemu's `663b9b9c0b`.
- Several displays: the macOS 13.6 guest does not drive port 1 at version
  62 ("Window mode", above). Whether it would with 0x122c reporting both
  ports, which guests negotiate version 245, and whether the one USB
  tablet's pointer reaches a second display.
- The `manual` entries of `crates/reims-vgpu-window/features.md` that
  "Window mode" does not record as measured, the guest's input, capture
  and the system's hot keys among them.
