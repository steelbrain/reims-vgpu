# Known issues

This page lists what we know is wrong right now, and what you can do about it.

## With several GPUs, a VM can get the slower one

A VM runs on one of your host's GPUs. If you picked one, it uses that. If the host has only one
GPU, it uses that. Otherwise it takes a discrete GPU over an integrated one. That's usually the
faster choice, but not always, since some integrated GPUs beat small discrete cards. On one of our
test hosts, with a Radeon 780M (integrated) and a Radeon RX 550 (discrete), VMs get the RX 550.
Drawing the guest's display at 3840x2400 takes about 40 ms a frame on the RX 550, and 24 ms on
the 780M.

On Linux and Windows you can pick the GPU yourself. `reims-ctl gpus` lists your GPUs, each with a
number and its vendor and device IDs, and `--gpu` gives a VM one of them:

```sh
reims-ctl gpus
reims-ctl set mac --gpu 2      # the second GPU on the list
reims-ctl set mac --gpu auto   # back to the default
```

You can also set the variable yourself, in QEMU's environment. On that host, this gives the VM the
780M, whose vendor and device IDs are `1002:15bf`:

```sh
reims-ctl set mac --env MESA_VK_DEVICE_SELECT=1002:15bf
```

`DRI_PRIME` and `VK_LOADER_DEVICE_SELECT` work too, with the same `vendor:device` in hex, and
`DRI_PRIME` also takes a PCI address, such as `pci-0000_03_00_0`. If what you set doesn't name a
GPU on your host, the VM gets the first GPU Vulkan lists. A Mac has one GPU, so there's nothing to
pick there.

## Video on Windows

We've found three bugs when a guest plays 4K H.264 video on a Windows host. We test that on an
Intel UHD 630.

- **Wrong pictures.** In 2 of 78 runs of our playback check, the guest got wrong pictures. Once
  it was a single bad picture in an otherwise good run. The other time a whole run was wrong,
  right after a run that had failed to decode.
- **"Cannot Decode".** A video can stop at about frame 85 with "Cannot Decode". Before a recent
  fix that happened in 2 of 28 runs, and since then in 1 of 63.
- **A slow first frame.** When a video starts, its first frame can take 0.1 to 0.5 s to show up.
  Before that same fix it took 0.8 to 0.9 s.

We're working on all three. We don't know of a workaround yet.

## QEMU uses more memory than you gave the VM

Graphics memory that the guest and the GPU share comes from the host, on top of the VM's RAM.
QEMU takes it as the guest's apps need it, and keeps 64 MiB of it spare so the guest doesn't wait
when it asks for more. When the guest lets go of some, QEMU gives it back once it has gone unused
for 5 seconds. On a Radeon 780M, a desktop with a browser and video used at most 73 MiB of it at
a time, and 3D apps use more. So leave your host some memory beyond what you give the VM.

## A Mac needs macOS 26 or later to run VMs

On a Mac, reims-vgpu runs on Mesa's KosmicKrisp Vulkan driver, which comes with Homebrew's `mesa`.
Homebrew builds KosmicKrisp only on macOS 26 and later, so on a Mac running macOS 15 or earlier,
VMs don't start: QEMU finds no GPU it can use and stops. `scripts/install-ctl.sh` installs `mesa`
for you on macOS 26 and later, and tells you when your Mac's macOS is too old. We've booted arm64
macOS 27.0.1 to its desktop this way on an M3 Max.
