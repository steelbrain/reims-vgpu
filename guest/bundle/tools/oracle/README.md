# The dGPU oracle probes

Small Metal programs of our own that measure what a real Mac GPU does where Metal's documented
contract leaves room: when CPU and GPU writes meet, what render passes keep, what the device
orders on its own, its timing, its errors and its answers. Run them on an Intel Mac with an AMD
discrete GPU and an Intel iGPU (the x86-64 guest's frame of reference), and on an Apple-silicon
Mac where it shows the gap. What they found is in
[docs/graphics-kext/dgpu-oracle.md](../../../../docs/graphics-kext/dgpu-oracle.md).

| probe | answers |
|---|---|
| `oracle_iosurface` | when a texture over an IOSurface sees CPU writes, and when the surface's pages see GPU writes, with and without `IOSurfaceLock`, in one process and across two; the cost of each sync |
| `oracle_storage` | Managed buffers and textures with and without `didModifyRange:` / `synchronizeResource:`; Shared textures; linear textures over Shared and Managed buffers; Shared buffers written while the GPU runs |
| `oracle_passes` | what load and store actions (`DontCare`, `Clear`) leave in color, depth and MSAA attachments; memoryless textures, framebuffer fetch, raster order groups and tile pipelines |
| `oracle_cmdbuf` | the order the device keeps for tracked and untracked resources, fences, barriers and command buffers; scheduled and completed timing, handler threads and order, GPU times, asynchronous pipeline creators |
| `oracle_caps` | limits, feature answers (public and the SPI getters core reads) and per-pixel-format answers |
| `oracle_errors` | the errors and refusals for purged resources, oversized creates, oversubscribed memory and a GPU timeout |

Each probe picks its device by a substring of its name from `MTLCopyAllDevices` (`AMD`,
`Intel`, `Apple`). It never calls `MTLCreateSystemDefaultDevice`, which switches a dual-GPU
MacBook's display to its discrete GPU. A case that may end the process (Metal's validation
asserts on some creates) runs in a child process of its own, and the line says how it died.
The header of each `.m` lists its cases and the meaning of every column.

## Build

On a Mac with Xcode's toolchain, never on the oracle:

```sh
guest/bundle/tools/oracle/build.sh /tmp/rgk-oracle/x86 x86_64   # for an Intel Mac
guest/bundle/tools/oracle/build.sh /tmp/rgk-oracle/arm arm64     # for an Apple-silicon Mac
```

Each probe that runs shaders loads `oracle.metallib` from its own directory, so copy it along.

## Run

On the oracle (`ssh macbook-pro-2019`), from `~/oracle/bin/rgk-oracle`, holding the shared GPU
lock and keeping the machine awake:

```sh
/usr/bin/lockf -k ~/oracle/gpu.lock caffeinate -s ./oracle_iosurface AMD single
/usr/bin/lockf -k ~/oracle/gpu.lock caffeinate -s ./oracle_iosurface Intel timing 3840 2400
/usr/bin/lockf -k ~/oracle/gpu.lock caffeinate -s ./oracle_storage AMD
```

On an Apple-silicon Mac of this repository, hold the GPU lock:
`scripts/internal/gpu-locks.sh -- /tmp/rgk-oracle/arm/oracle_passes Apple`.

`oracle_iosurface` also takes a script of its own steps in place of a scenario name (the steps
are listed in its header). `oracle_errors DEVICE timeout SECONDS` runs one long kernel to meet
the GPU's watchdog (about 16 s on the Intel 630, 20 s on the 5300M): run it only on a GPU that
does not drive a display. The other kernels stay near a second.
