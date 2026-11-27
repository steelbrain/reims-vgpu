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

Task Manager's shared GPU memory for QEMU also climbs while a guest decodes video, and it stays up
until QEMU exits. How far it climbs follows how many decoded pictures an app holds at once. One of
our tests keeps every picture of a 4K HEVC clip, and each play added 5.5 to 8.2 GiB, to 37 GiB
after six plays. An H.264 export in our tests added about 97 MiB a round. On our Intel UHD 630, no
memory is used behind that figure: over two of those 4K plays it grew by 11.5 GiB, while the host's
available memory didn't drop and its committed memory didn't rise. We haven't measured this on an
NVIDIA or AMD card on a Windows host.

## Some video formats don't use your GPU

Your GPU may decode or encode video formats that the guest doesn't get from it. For these, we
found no way in macOS to hand the work to our virtual GPU, so the guest uses its own software, or
has no player for them at all:

- **VP9 decoding on Intel Mac guests.** macOS 26.7.2 sends VP9 to a GPU driver like ours, but
  never tells it where to put each decoded picture.
- **AV1 decoding on Intel Mac guests.** Intel macOS has no AV1 decoding that a GPU can provide.
- **HEVC encoding on Intel Mac guests.** macOS 26.7.2 sends hardware HEVC encoding to an Intel
  GPU, even when an app asks for ours, so apps encode HEVC in software. On Linux hosts, H.264
  encoding uses your GPU.
- **HEVC at 4:2:2, 4:4:4 or 12 bits on Intel Mac guests.** macOS 26.7.2 would only take these
  from a GPU as 4:2:0 at 10 bits, which loses colour detail, so the guest keeps decoding them in
  software at full quality.
- **HEVC at 4:2:2, 4:4:4 or 12 bits on arm64 guests.** These decode on your GPU where its own
  driver offers them, and in the guest's software where it doesn't. The AMD GPUs we test on, a
  Radeon 780M and an RX 7900 XTX, offer none of them. NVIDIA GPUs decode 4:2:0 and 4:4:4 up to
  12 bits but not 4:2:2, even on a card whose decoder handles it, such as the RTX 5080 Laptop GPU
  we test on: we haven't built that part yet. An Intel Iris Xe decodes all of them, except 4:2:0
  video at 8 or 10 bits that uses HEVC's range extensions profiles, which we don't take from
  Intel's driver yet.
- **JPEG decoding on arm64 guests.** macOS 14.6.1 and 26.6.2 look for Apple's own JPEG and scaler
  hardware by name before they hand a photo or a Motion JPEG video to a GPU. Our virtual
  machines have neither, so they decode JPEG in software. Our virtual GPU decodes JPEG when
  asked directly, but no app's route asks it.
- **MPEG-1, MPEG-2, MPEG-4 Part 2, VC-1 and VP8 decoding, and VP9 and AV1 encoding.** We found no
  way in macOS for a GPU to take these.

## Sound comes after video on Windows

On a Windows host, a VM's sound reaches your speakers about 0.1 s after the guest sends it, and
macOS in the VM is told it plays within about 2 ms. So a video player in the guest shows each
frame about 0.1 s before you hear the sound that goes with it. We measured this on our Windows
test host, and haven't measured other hosts yet.

macOS gets that 2 ms from its own driver for the VM's sound device, which reports the same figure
whatever the host does. Telling it the real delay needs a driver of ours in the guest, which we
don't have yet. We don't know of a workaround.

## QEMU uses more memory than you gave the VM

Graphics memory that the guest and the GPU share comes from the host, on top of the VM's RAM.
QEMU takes it as the guest's apps need it, and keeps 64 MiB of it spare so the guest doesn't wait
when it asks for more. When the guest lets go of some, QEMU gives it back once it has gone unused
for 5 seconds. On a Radeon 780M, a desktop with a browser and video used at most 73 MiB of it at
a time, and 3D apps use more. So leave your host some memory beyond what you give the VM.

## An app can pause when it asks for a lot of graphics memory at once

When an app in the guest asks for more graphics memory at once than QEMU has spare, the host's GPU
driver has to allocate that much, and the app waits until it's done. Other apps in the guest don't
wait for it. On an RTX 2080, an app that asked for 547 MiB waited 271.5 ms, and 139.5 ms of that
was the driver's allocation. On a Radeon RX 550, an app waited 65 ms, 64.4 ms of it the driver's.
We measured these before a change to how QEMU handles such a request, and we'll update them once
we've measured again. The driver's part stays either way. We don't know of a workaround yet.

## A macOS debug kernel may not start on an AMD host

On a Linux host whose CPU has AVX-512, such as a Zen 4 Ryzen, `reims-ctl` turns on AVX-512's
registers for the threads of a macOS VM's apps. Apps still see no AVX-512, but the CPU runs their
code faster: on a Ryzen 7 8745H, a CPU benchmark's scores went up about 10%. macOS's usual kernels
work with this. A DEVELOPMENT or DEBUG kernel, from a Kernel Debug Kit, checks at startup that the
saved registers take as much room as on Intel's CPUs. On AMD's they take less, so the check fails
and the kernel stops; on Intel's it passes. We found this in the kernel's source and haven't
started one this way.

To start such a kernel, turn this off for that VM. In its `vm.toml`
(`~/.local/share/reims-ctl/vms/NAME/vm.toml` unless you made it elsewhere), add
`,x-avx512-xstate=off` to the end of `cpu_flags`, under `[q35]`:

```toml
cpu_flags = "+ssse3,+sse4.2,+popcnt,+avx,+aes,+xsave,+xsaveopt,check,x-avx512-xstate=off"
```

## Animations can run late on an Intel integrated GPU

On a Linux host with an Intel Iris Xe Graphics (Core i9-13900H), a web page that animates once per
display refresh shows about 30% of its frames a refresh late, because under the host's default
power policy its CPU runs the guest's and the host's graphics work at about 1.3 GHz, too slow to
finish each frame within one refresh.

## The pointer moves differently while the guest shares its screen

While an app in the guest records or shares the screen, such as a video call's screen share or
macOS's own screen recording, macOS hides the mouse cursor and draws it into the picture itself.
To the window, that looks like a game hiding the cursor to steer by the mouse. So while the window
has captured input, the pointer switches to the way it works for games 50 ms after you first move
the mouse, and stays that way until the recording or share stops. The guest then moves the cursor
by your mouse's own movement and the guest's mouse settings rather than your host's pointer speed,
so it can feel slower. Your host's pointer also stays hidden inside the window, and can't move on
to another of the VM's displays. To get the normal pointer back, press the release chord once
(Ctrl+Alt+Super, Ctrl+Alt+Win on Windows, or ⌃⌥⌘ on a Mac, where Window ▸ Release Input does the
same). The pointer then stays normal until macOS shows its cursor again, which it does when the
recording or share stops, even if you click back into the guest's picture meanwhile.

## Touchpad scrolling moves in steps of 10 points on Intel Mac guests

In an x86-64 macOS guest, a touchpad on your host scrolls the way a touchpad does, by as far as
your fingers move, but in steps of 10 points: movement shorter than that waits until more of it
adds up. macOS's own driver counts scrolling in whole steps, and nothing a device can tell it makes
a step smaller. The guest also gets the distance only, not the gesture: macOS doesn't see a swipe
begin or end. arm64 guests scroll through their own virtual tablet instead, which gets the
gesture too.

## On a Mac

### You need macOS 26 or later to run VMs

On a Mac, reims-vgpu runs on Mesa's KosmicKrisp Vulkan driver, which comes with Homebrew's `mesa`.
Homebrew builds KosmicKrisp only on macOS 26 and later, so on a Mac running macOS 15 or earlier,
VMs don't start: QEMU finds no GPU it can use and stops. `scripts/install-ctl.sh` installs `mesa`
for you on macOS 26 and later, and tells you when your Mac's macOS is too old. We've booted arm64
macOS 27.0.1 to its desktop this way on an M3 Max.

### Opening apps copies more than on Linux

On a Mac, the guest's memory reaches the GPU through its address on the host. That way of sharing
memory can't hand the GPU an image's bytes as they are, so we copy them. Most of the copying happens
while apps open. On our M3 Max, we copied from about a quarter of a 1080p frame's bytes to about one
and a half frames' worth while apps opened, and under 1% of a frame's bytes during steady
animation. Windows hosts share the guest's memory the same way, so this applies there too, though
we've only measured it on a Mac. Linux hosts share it another way, and we're building support
there for using an image's bytes in place.
