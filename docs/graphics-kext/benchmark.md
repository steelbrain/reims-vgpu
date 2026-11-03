# GPU-heavy benchmark: the graphics lane against the paravirt lane

Desktop compositing was measured first: on an RTX 5080 laptop host, the graphics lane composites
the guest's desktop at 60 a second (60 Hz cap) or 119.6 (120 Hz) with QEMU at 62-73% CPU, and the
Apple paravirt lane at 27 a second with QEMU at 297%. This note measures GPU-heavy work in both
lanes on the same host, guest and host build, beside real Macs. Measured 2026-11-03.

The tool is `guest/bundle/tools/bench/bench.m` (`make bench`, `make bench ARCH=arm64`). Its
header says what each case does and what each number means. In short:
- `frag.alu.*`: a full-screen heavy fragment shader (a 256-step FMA loop), offscreen, one
  command buffer a frame;
- `frag.fill.2160p`: 16 blended full-screen layers a frame;
- `compute.fma`: FMA-bound compute;
- `blit.256m`: a 256 MiB Private-to-Private copy;
- `sample.tex`: 16 bilinear samples a thread from a 4096 x 4096 texture;
- `draws.10k`: 10,000 small draws a command buffer, each with its own `setVertexBytes`;
- `upload.64m` and `readback.64m`: a 64 MiB Shared buffer the CPU writes and the GPU reads, and
  the reverse, end to end;
- `mps.sgemm.*`: MPSMatrixMultiplication, FP32;
- `onscreen`: a full-screen CAMetalLayer animating the heavy shader for 20 s, presented frames
  counted from the drawables' `presentedTime`.

## Machines

- **msi**: RTX 5080 Laptop GPU (NVIDIA 615.71.09), Core Ultra 9 275HX, Linux 7.2.8 (CachyOS). The
  host library and QEMU are built from this branch at 0dbc3f40e on the Vulkan platform
  (metal2vulkan ef392b28b1, QEMU 67cc08a). Both lanes ran this one build.
- **The guest**: macOS 26.6.1 (25G76) x86-64, 8 vCPUs, 16 GiB, SIP on, 3840 x 2400 at 120 Hz.
  Both lanes boot one disk image through throwaway overlays.
  - **ours**: ReimsGraphics, its loader and the bundle at a3d308cec (unchanged to 0dbc3f40e), on
    `reims-graphics-pci`. Metal names the device "NVIDIA GeForce RTX 5080 Laptop GPU".
  - **paravirt**: the guest's own Apple paravirt GPU driver, on `apple-gfx-pci` and
    `apple-gfx-iosurface-pci` (display mode `3840x2400@120`). Metal names the device "Apple
    Paravirtual device", with unified memory.
- **M3 Max**: Mac15,9, macOS 27.0.1 (26A434), native arm64. On-screen ran on the built-in
  display, 3456 x 2234 at 120 Hz.
- **MBP 2019**: MacBookPro16,1, macOS 26.7.1 (25G241), x86_64.
  - AMD Radeon Pro 5300M and Intel UHD Graphics 630, picked with `-d`.
  - On-screen ran on the Intel GPU only (it drives the display, 3584 x 2240 at 60 Hz), lid
    closed.
  - Each GPU was checked for a new `*.gpuRestart` after its run; none appeared.

## How it was run

In each lane, one boot ran the cases over SSH, each case in its own process for at least 6 s
(`-m 6`). On-screen ran in the logged-in user's session through `launchctl asuser`. Each boot
held the 5080's GPU lock exclusively, so nothing else used the GPU during it.

The host sampled once a second, and each case's numbers are the means over its window:
- QEMU's CPU by thread from `/proc`, where 100% is one core;
- `nvidia-smi dmon`: SM utilization, clocks, power and PCIe traffic.

Neither lane logged a `reims-vgpu: fatal` line, and WindowServer kept its pid through every run.

**Timing.** The msi numbers are wall time, from commit to completion: on the Vulkan platform a
command buffer's GPU start and end are host-clock windows (metal2vulkan's
`gpu_times_from_host`), not the GPU's own times.

**Clock ramp.** The 5080's driver holds about 1440 MHz core and 9001 MHz memory for 1 to 8 s
of sustained full load before ramping to its power cap. The delay varied run to run in *both*
lanes, the same way: paravirt held 8 s in one run and 1 s in another. So the frame cases come
from a 20 s run (`-m 20 -v`) in each lane, with repetitions split by the per-second clock:
- `frag.alu.2160p`: 81 of 89 repetitions in our run, and 97 of 103 in paravirt's, ran after
  the ramp (~2270 MHz, 125 W);
- `frag.fill.2160p`: 156 of 161 and 159 of 160 (~2770 MHz);
- `frag.alu.1080p`: mostly before the ramp in both lanes, so it is a low-clock number.

The hold is the driver's power-capped state. Measured 2026-11-03 with nvidia-smi sampling every
100 ms on the host, and its 1 s samples during the boots above:
- **What it is.**
  - The core clock is held at 1207 to 1477 MHz while the SM is 90 to 100% busy, at 31 to 103 W.
  - Memory runs at 9001 MHz (the driver's P4), and nvidia-smi gives the "SW power cap" reason.
    The part's cap is 115 W.
  - Outside it, the core runs at 2.1 to 2.8 GHz under the same load.
- **When it comes.**
  - Started from idle, 2 of 8 host processes went into it 2.1 to 2.4 s into the load and stayed
    6.0 to 6.8 s. In the other 6, the driver's P4 kept the core at 1755 to 2377 MHz.
  - Under sustained load it came in each of 3 runs. It began 10.2 to 10.7 s into the load and
    lasted 11 to 12 s.
  - In the boots it lasted 1 to 8 s, at 1425 to 1447 MHz and 31 to 33 W.
- **No VM is needed.** Those host processes ran no VM. They ran the frame cases on the Vulkan
  platform and made their targets as the decoder makes a guest's.
- **What it costs.** Both frame cases follow the core clock, not the memory clock.
  - `frag.fill.2160p` runs:
    - 769 to 858 fps (102 to 114 Gpix/s) at 1215 to 1267 MHz;
    - about 1000 fps (134 Gpix/s) at 1425 to 1447 MHz in the boots;
    - 1910 to 1950 fps (254 to 259 Gpix/s) at 2767 to 2782 MHz, with memory at 9001, 11001 or
      14001 MHz alike.
  - `frag.alu.2160p` runs 254 to 303 fps at 1215 to 1477 MHz, and 470 to 503 at 2.1 to 2.8 GHz.
  - At the high clock, fill was the same whatever the target's image: fresh or recycled, at any
    offset in a pooled block, in a block of its own, or without the usages the runtime adds
    where they are free.
- **What a 3 s case shows.**
  - A case of 3 s (`-m 3`) lands in either state.
  - In the boots that ran the three frame cases in a row for 3 s each, the `frag.alu` cases ran
    at the low clock's rates in every sample. `frag.fill.2160p` landed in either state.
  - Both lanes show it: ours at 132.76 (low clock throughout), 134.23 (low for 8 s) and 255.63
    Gpix/s (high); paravirt at 257 (low for 1 s), 133.67 (low for 8 s) and 255.
  - A warm-up does not avoid it. After 12 s of load, the fill case started inside the low clock
    in 3 of 3 runs. Only the core clock sampled during a repetition says which state it ran in.

## Results

Higher is better everywhere. Ratios are ours over paravirt. QEMU CPU, SM and clock are the
lanes' means over each case.

| case | unit | ours | paravirt | ours / paravirt | QEMU CPU % ours / paravirt | 5080 SM % ours / paravirt | M3 Max | MBP AMD 5300M | MBP Intel 630 |
|---|---|---|---|---|---|---|---|---|---|
| frag.alu.1080p | fps | 1172 | 986 | 1.19 | 52 / 63 | 99 / 99 | 576 | 181 | 26.2 |
| frag.alu.2160p | fps | 485 | 484 | 1.00 | 23 / 26 | 97 / 100 | 155 | 45.7 | 6.6 |
| frag.fill.2160p | Gpix/s | 256 | 255 | 1.00 | 67 / 90 | 98 / 99 | 412 | 36.9 | 7.8 |
| compute.fma | TFLOPS | 32.9 | 31.4 | 1.05 | 11 / 9 | 99 / 96 | 12.0 | 3.1 | 0.18 |
| blit.256m | GB/s | 308 | 6.5 | 47 | 18 / 48 | 80 / 60 | 162 | 76.8 | 9.0 |
| sample.tex | Gsamples/s | 330 | 76.8 | 4.3 | 13 / 22 | 88 / 87 | 217 | 78.9 | 10.4 |
| draws.10k | k draws/s | 456 | 404 | 1.13 | 108 / 106 | 2 / 18 | 5902 | 8964 | 4132 |
| upload.64m | GB/s | 7.7 | 3.8 | 2.0 | 56 / 58 | 45 / 34 | 18.4 | 3.5 | 3.8 |
| readback.64m | GB/s | 7.7 | 3.7 | 2.1 | 99 / 102 | 61 / 44 | 26.9 | 3.5 | 6.1 |
| mps.sgemm.2048 | TFLOPS | 11.4 | 0.59 | 19 | 22 / 18 | 88 / 94 | 11.1 | 2.7 | 0.19 |
| mps.sgemm.4096 | TFLOPS | 10.2 | 0.98 | 10 | 16 / 22 | 95 / 96 | 11.6 | 2.3 | 0.14 |
| onscreen | presented fps | not reported (see below) | 42.4, 41.1 | | 23 / 282 | 59 / 78 | 119.8 | not run | 6.7 |

Idle desktop before the cases: QEMU 8-12% in ours, 7% in paravirt. The 1080p fragment case's
post-ramp repetitions were few: ours 1748 fps (13), paravirt 1991 fps (6).

## What it shows

- **Where the GPU does the work, both lanes match.** The 2160p fragment ALU, fill and compute
  FMA cases are GPU-bound in both lanes. They reach the same rates at the same clocks, so neither
  lane adds a cost the GPU sees.
  - Against the M3 Max, ALU and FMA run 2.7-3.1x faster; fill runs 0.6x, since the M3 blends in
    tile memory.
  - Against the 5300M: 7-11x.
- **Memory placement is where ours wins.**
  - Paravirt's device has unified memory, so its Private resources are guest RAM, and the 5080
    reaches them across PCIe. Its 256 MiB blit moves ~7.8 GB/s each way over the bus (6.5 GB/s
    copied); ours keeps Private memory in VRAM and copies at 308 GB/s.
  - Texture sampling is 4.3x for the same reason.
  - MPS SGEMM is 10-19x, at 20 GB/s of PCIe reads in paravirt. Ours matches the M3 Max.
- **Shared memory moves 2x faster in ours**: 7.7 against 3.8 GB/s upload and readback.
- **Draw calls are host-bound in both lanes.** 10,000 draws a command buffer run at 456k (ours)
  and 404k (paravirt) draws a second, with the GPU nearly idle. Each lane's host decoder thread
  is at a full core: our graphics thread at 96%, paravirt's FIFO thread at 93%. The guest
  encodes 10,000 draws in 0.89 ms in ours (1.43 ms in paravirt), so the host's decode and
  replay, ~2 us a draw, is the lane's limit. The real Macs do 4-9M draws a second, which makes
  this the largest gap to native.
- **On-screen, paravirt is CPU-bound and ours gives no present feedback.**
  - **Paravirt** presents 41-42 fps at 4K with QEMU at 270-282% CPU: its FIFO thread at 85-87%,
    metal2vulkan's handler at 82-85% and its queue at 62-64%. It moves 16 GB/s of PCIe reads, so
    it is bound by the host CPU, not the GPU (SM 74-78%).
  - **Ours**: QEMU at 23-25%. But every drawable comes back with `presentedTime` 0, counted as
    dropped, and `nextDrawable` never throttles: the app submitted 254 and 320 frames a second
    at a 120 Hz display, with the GPU at 59-69% SM.
  - With the present trace on (`REIMS_VGPU_LOG=info,reims_graphics=trace`), the host saw one
    `PRESENT_SURFACE` in the 20 s window. So ours gives apps neither present feedback nor
    display-rate back pressure, and its displayed rate is unmeasured here. That is a lane bug,
    handed on to be fixed.
  - Fixed since: the bundle's `presentDrawable:` presented nothing. With the fix, `onscreen -t
    10` on msi presents 120.0 frames a second, 1201 presented and 0 dropped, frame p50 8.33 ms,
    with QEMU at 40% and SM at 41% (measured 2026-11-03).
  - Paravirt and both real Macs report `presentedTime` and throttle to what is displayed:
    paravirt 848 presented / 0 dropped; Intel 630 6.7 of 6.8; M3 119.8 at 120 Hz.
