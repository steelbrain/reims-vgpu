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
| `oracle_storage` | Managed buffers and textures with and without `didModifyRange:` / `synchronizeResource:`; Shared textures; linear textures over Shared and Managed buffers, their `replaceRegion:` and `didModifyData` among them, and the offsets and pitches each form of a texture over a buffer takes; no-copy Managed buffers; argument encoders writing Shared and Managed argument buffers; Shared buffers written while the GPU runs |
| `oracle_passes` | what load and store actions (`DontCare`, `Clear`) leave in color, depth, stencil and MSAA attachments; memoryless textures, framebuffer fetch, raster order groups and tile pipelines |
| `oracle_cmdbuf` | the order the device keeps for tracked and untracked resources, fences, barriers and command buffers; scheduled and completed timing, handler threads and order, GPU times, asynchronous pipeline creators; when the CPU sees a shared event's signal from the middle of a buffer |
| `oracle_caps` | limits, feature answers (public and the SPI getters core reads) and per-pixel-format answers |
| `oracle_errors` | the errors and refusals for purged resources, oversized creates, oversubscribed memory and a GPU timeout |
| `oracle_source` | what a runtime source compile makes (`newLibraryWithSource:options:error:`): the library, its bytes (a `.metallib` of AIR) and its errors, through the device's public selector, its Metal base class's, and that class's compiler-task form, which reaches Metal's compiler service without the device's `compiler` |
| `oracle_surftex` | which textures `newTextureWithDescriptor:iosurface:plane:` makes over a surface's plane, answers nil for, or ends the process at, by the descriptor's type, size, depth, array length, levels, samples, format, usage, storage and swizzle and the surface's OSType and planes |
| `oracle_ycbcr` | what a two-plane Y'CbCr texture (`MTLPixelFormat` 500) over plane 0 of a 420v or 420f surface reads, by the surface's `kIOSurfaceYCbCrMatrix` (709, 601 or none, set by CoreVideo or on the surface), or whether Metal refuses it |
| `oracle_ca_ycbcr` | which matrix CoreAnimation (a CARenderer drawing the surface as a layer's contents into a texture of the device) and VideoToolbox (`VTPixelTransferSession` to 32BGRA) convert an `x420` or `420v` surface by, tagged 709, 601 or untagged |
| `oracle_sampler` | what a sampler's `lodBias` and `reductionMode` do to its samples, by the LOD's source (`level()`, `gradient2d()`, a fragment's implicit LOD) and the mip filter, and the device's SPI answers about them |
| `oracle_order` | when a command buffer that runs after another queue's or another process's pending access to the same IOSurface is answered submitted (`commitAndWaitUntilSubmitted`), scheduled and completed, and whether the GPU ran it after that access |
| `oracle_names` | the words Metal gives where a function is made with a constant value of another type, for every `MTLDataType` (a bfloat constant's among them), and what a binary archive answers for a mesh pipeline added to it, with no mesh function and with one |
| `oracle_mesh` | what a mesh render pipeline answers for each descriptor shape an app sets (object functions, the maxima, the payload, required sizes, no fragment or no mesh function, functions of another stage, archive misses, indirect command buffer support), what each mesh draw form draws, and the mesh commands of indirect command buffers |
| `oracle_bundle` | what the device answers where the guest bundle answered otherwise: buffers' `allocatedSize` (plain, heap and no-copy), event creators with options, binary archives over Metal's and other fat files and a pipeline's hit there, `newFunctionWithDescriptor:` failing on an archive miss, each function stage's TYPE tag and type, a dynamic library's type and install name, `minLOD`, `bufferClass`, heap sizes by `allowGPUOptimizedContents`, the libraries of a function array or a package, textures' `gpuAddress` and `virtualAddress` by storage, shape, heap, buffer, view and IOSurface (`texaddr`) and what a CPU write at a texture's `virtualAddress` shows to `getBytes:` (`texmem`), what a no-copy buffer over read-only memory, anonymous or a file, does with GPU writes (`nocopy.ro`) and how many copies a writable no-copy Managed buffer keeps (`nocopy.copies`), `minimumLinearTextureAlignmentForPixelFormat:` for one raw value (`linalign.N`), depth/stencil states in argument buffers and encoders made from a buffer binding or with their reflection, the IOSurface SPI (`copyIOSurfaceSharedTextureProperties:`, `newBufferWithIOSurface:`), a shared event's port and coding, and which slices of a binary archive Metal reads |
| `oracle_events` | a shared event made in one process and used in another: its Mach port (`eventPort`) and handle, the events made back from either, their coding (keyed and NSXPC), and the value, listeners and waits two holders share, in one process and across two; it runs the same on the guest's device |
| `oracle_reflection` | the reflection a render, mesh or compute pipeline is made with (`reflection:` and the completion handlers' argument), by options, over `oracle_reflection.metallib`: every getter of its bindings, built-in arguments, types and struct members, recursively, and with `ORACLE_IVARS` set their instance variables, then the reflection classes' members (shared with `../pipeline_reflection_check.mm` through `oracle_walk.h`) |

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
