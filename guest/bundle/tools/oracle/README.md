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
| `oracle_cmdbuf` | the order the device keeps for tracked and untracked resources, fences, barriers and command buffers; scheduled and completed timing, handler threads and order, GPU times (one long buffer's among them), asynchronous pipeline creators; when the CPU sees a shared event's signal from the middle of a buffer |
| `oracle_caps` | limits, feature answers (public and the SPI getters core reads) and per-pixel-format answers |
| `oracle_errors` | the errors and refusals for purged resources, oversized creates, oversubscribed memory and a GPU timeout |
| `oracle_source` | what a runtime source compile makes (`newLibraryWithSource:options:error:`): the library, its bytes (a `.metallib` of AIR) and its errors, through the device's public selector, its Metal base class's, and that class's compiler-task form, which reaches Metal's compiler service without the device's `compiler` |
| `oracle_surftex` | which textures `newTextureWithDescriptor:iosurface:plane:` makes over a surface's plane, answers nil for, or ends the process at, by the descriptor's type, size, depth, array length, levels, samples, format, usage, storage and swizzle and the surface's OSType and planes; its fourth and fifth rounds ask what an Apple GPU answers otherwise (its own compressed formats, cubes over a plane, two-plane textures of odd sizes); its aliasing cases (`--alias N`) what one command buffer's second encoder reads through one texture over a plane of what its first wrote through another, and what each reads and leaves past the plane's texels |
| `oracle_texdesc` | which texture descriptors the device's `newTextureWithDescriptor:`, a Private heap's and a Shared buffer's `newTextureWithDescriptor:offset:bytesPerRow:` make, answer nil for, or end the process at, by the descriptor alone: its type, size, depth, array length, levels, samples, format, usage and storage |
| `oracle_nilbind` | what a command buffer does with nil where an object belongs: a draw or dispatch reading a texture, sampler or buffer bound nil or never bound, a texture written bound nil, a render pass with a nil attachment, a pipeline state set nil, a blit from or to nil and a use of nil |
| `oracle_ycbcr` | what a two-plane Y'CbCr texture (`MTLPixelFormat` 500) over plane 0 of a 420v or 420f surface reads, by the surface's `kIOSurfaceYCbCrMatrix` (709, 601 or none, set by CoreVideo or on the surface), or whether Metal refuses it |
| `oracle_ca_ycbcr` | which matrix CoreAnimation (a CARenderer drawing the surface as a layer's contents into a texture of the device) and VideoToolbox (`VTPixelTransferSession` to 32BGRA) convert an `x420` or `420v` surface by, tagged 709, 601 or untagged |
| `oracle_sampler` | what a sampler's `lodBias` and `reductionMode` do to its samples, by the LOD's source (`level()`, `gradient2d()`, a fragment's implicit LOD) and the mip filter, and the device's SPI answers about them |
| `oracle_order` | when a command buffer that runs after another queue's or another process's pending access to the same IOSurface is answered submitted (`commitAndWaitUntilSubmitted`), scheduled and completed, and whether the GPU ran it after that access |
| `oracle_complete` | what a texture's and a buffer's resource SPI `isComplete`, `isWriteComplete` and `waitUntilComplete` answer while a command buffer that renders to, writes, blits into or only reads it is in flight, right after its commit and once it is scheduled, before its commit, and for a resource only another queue's buffer runs beside; and whether a command buffer of a second queue (empty, a blit, a synchronize of the texture the first writes, a compute dispatch) completes before the first queue's earlier one |
| `oracle_names` | the words Metal gives where a function is made with a constant value of another type, for every `MTLDataType` (a bfloat constant's among them), and what a binary archive answers for a mesh pipeline added to it, with no mesh function and with one |
| `oracle_mesh` | what a mesh render pipeline answers for each descriptor shape an app sets (object functions, the maxima, the payload, required sizes, no fragment or no mesh function, functions of another stage, archive misses, indirect command buffer support), what each mesh draw form draws, and the mesh commands of indirect command buffers |
| `oracle_spi_members` | which members of each Metal protocol and its SPI counterpart (`MTLLibrarySPI` for `MTLLibrary`, ...) the device's own objects answer (`respondsToSelector:`), for each kind of object the guest bundle makes: the verdicts `tests/protocol_members.txt` gives as `%spi-both`, `%spi-neither` and `%spi-differ`, and the class that answers each; with `answers`, what a fresh object answers to each getter that does not act (run it once: a getter that asserts ends the process, and is listed in the tool) |
| `oracle_bundle` | what the device answers where the guest bundle answered otherwise: buffers' `allocatedSize` (plain, heap and no-copy), event creators with options, binary archives over Metal's and other fat files and a pipeline's hit there, `newFunctionWithDescriptor:` failing on an archive miss, each function stage's TYPE tag and type, a dynamic library's type and install name, `minLOD`, `bufferClass`, heap sizes by `allowGPUOptimizedContents`, the libraries of a function array or a package, textures' `gpuAddress` and `virtualAddress` by storage, shape, heap, buffer, view and IOSurface (`texaddr`) and what a CPU write at a texture's `virtualAddress` shows to `getBytes:` (`texmem`), what a no-copy buffer over read-only memory, anonymous or a file, does with GPU writes (`nocopy.ro`) and how many copies a writable no-copy Managed buffer keeps (`nocopy.copies`), `minimumLinearTextureAlignmentForPixelFormat:` for one raw value (`linalign.N`), depth/stencil states in argument buffers and encoders made from a buffer binding or with their reflection, the IOSurface SPI (`copyIOSurfaceSharedTextureProperties:`, `newBufferWithIOSurface:`), a shared event's port and coding, which slices of a binary archive Metal reads, a Shared heap (`heap.shared`), buffers and textures of options holding bit 17, 18 or 19, and whether buffers answer the tiled and linear texture SPI (`tiled.responds`) |
| `oracle_events` | a shared event made in one process and used in another: its Mach port (`eventPort`) and handle, the events made back from either, their coding (keyed and NSXPC), and the value, listeners and waits two holders share, in one process and across two; it runs the same on the guest's device |
| `oracle_ratemap` | what a rasterization rate map answers and draws: the layer counts the device supports, its maps' and descriptors' classes and members, a map's answers and parameter data, a triangle drawn through one, decoded in the shader, and which calls end the process |
| `oracle_reflection` | the reflection a render, mesh or compute pipeline is made with (`reflection:` and the completion handlers' argument), by options, over `oracle_reflection.metallib`: every getter of its bindings, built-in arguments, types and struct members, recursively, and with `ORACLE_IVARS` set their instance variables, then the reflection classes' members (shared with `../pipeline_reflection_check.mm` through `oracle_walk.h`) |
| `oracle_urls` | what each call of a device, a library and a binary archive that takes a URL answers for each form of URL: nil, a path of no scheme, a file URL, another scheme's URL of a path, a URL of no path, the file there and not: the object made, the error, whether a write left a file, or how the process ended |
| `oracle_residency` | what a residency set (`MTLResidencySet`) answers: whether the device makes one, its classes and members, its count, contents and size after each add, remove and commit of buffers, textures and heaps, its residency requests, what holds what (a set its allocations, a queue and a command buffer their sets), a queue's, a command buffer's and the encoders' residency calls, a `CAMetalLayer`'s set, and the calls on bad inputs, each in a child process |
| `oracle_rows` | the bytes per row, plane offsets and sizes IOSurface works out for a surface made with none named, by OSType and size, and the alignments it answers (`IOSurfaceGetPropertyAlignment`, `IOSurfaceAlignProperty`); the rows and plane offsets CoreVideo gives an IOSurface-backed pixel buffer; the device's linear and IOSurface texture alignments |
| `oracle_overlap` | which function constant values Metal refuses where they put two present arguments of one argument table (buffers, constant and argument buffers, threadgroup memory, textures and texture arrays, samplers) at overlapping locations, by predicate, use and index, a kernel's, a vertex function's and a fragment function's, at which call (by values set by index, name or range, a copy, a descriptor, a copy of it, a specialized name, each completion handler form) and with which error |
| `oracle_profile` | what Metal's base device (`_MTLDevice`) answers under each feature profile asked: its getters of no argument, `supportsFamily:`, `supportsFeatureSet:` and its feature queries' answers, from a subclass that answers `featureProfile` alone, with no GPU driver object made |
| `oracle_readonly` | what each GPU and CPU write into read-only memory does (a no-copy buffer or a heap made with `MTLResourceOptions` bit 24, or over pages the process maps read-only, against a writable control): a render pass's stores and resolves of every action into it, colour, depth and stencil, by the descriptor and late; a draw's, a kernel's and a blit's texture writes; the CPU's `replaceRegion:`; the command buffer's status and error, the pages after it, and whether the queue runs on and its second fault revokes it; `oracle_readonly.sh DEVICE` runs every shape |
| `oracle_threads` | what a compute pipeline's threadgroup fields make, each step a first compile in a child process of its own: `requiredThreadsPerThreadgroup` alone, against the kernel's `[[max_total_threads_per_threadgroup(n)]]`, the device's limit and the descriptor's `maxTotalThreadsPerThreadgroup`, beside `threadGroupSizeIsMultipleOfThreadExecutionWidth`, through each creator and after the kernel compiled once; `maxTotalThreadsPerThreadgroup` against the kernel's and the device's; macOS 27's `forwardProgressUsage`, `contentionRelief` and `optimizeForPersistentKernel`; what each made pipeline answers |
| `oracle_waits` | whether `waitUntilCompleted` and `waitUntilScheduled` return before, while or after a command buffer's completed and scheduled handlers run, whether a handler that blocks holds them, and whether a later buffer of the queue, with a handler or none, waits for an earlier buffer's handler that blocks; commit to wait on an idle GPU with and without a handler |

Each probe picks its device by a substring of its name from `MTLCopyAllDevices` (`AMD`,
`Intel`, `Apple`). It never calls `MTLCreateSystemDefaultDevice`, which switches a dual-GPU
MacBook's display to its discrete GPU. A case that may end the process (Metal's validation
asserts on some creates) runs in a child process of its own, and the line says how it died.
The header of each `.m` lists its cases and the meaning of every column.

## Beside the probes: gameload

`../gameload` is a load, not a probe: a game-shaped frame of our own, rendered offscreen for a fixed
number of frames, which prints each frame's CPU, wait, frame-to-frame and GPU times in ms and their
p50, p90 and mean. A frame holds what AAA Metal ports make: textures and a buffer placed in a
placement heap and uploaded each frame, transient textures from an automatic heap made aliasable as
they are done with, render targets placed in a heap, a 4x MSAA scene with depth resolved and
composited through an sRGB view, a UI pass into a BGRA view, an indirect command buffer of draws a
kernel encodes, a compute indirect command buffer of concurrent dispatches a kernel encodes (their
sum checked by the CPU every frame), 4000 CPU draws and 1000 shadow draws over argument buffers,
compute post-processing and a blit stream with a read-back. Each of its nine shapes (`heap`,
`rtheap`, `targets`, `views`, `icb`, `cicb`, `draws`, `post`, `blit`) turns off with
`--off NAME,...`, so a run with one off measures that shape's share; `gameload.m`'s header lists
every option and line.
`make -C guest/bundle/tools/gameload` builds one universal binary (arm64 and x86_64),
`guest/bundle/build/gameload`, which runs on an Apple-silicon Mac, an Intel Mac and both guests.
The measured runs were these. Each of the first four ran once with every shape on and then once
with each shape off (`--off heap`, `--off rtheap`, and so on), the native ones each in a potluck job
holding the GPU it measures; the last ran once:

```sh
caffeinate -s ./gameload -d Apple -f 1500 -w 150 -q             # the M3 Max (--gpu m3max --ram 2)
caffeinate -s ./gameload -d AMD -f 600 -w 60 -q                 # the oracle's 5300M (--gpu 5300m --ram 1)
caffeinate -s ./gameload -d Intel -f 600 -w 60 -q               # the oracle's 630 (--gpu 630 --ram 1)
~/walk/gl/gameload -f 300 -w 30                                 # a guest, in its GUI session
~/walk/gl/gameload -f 300 -w 30 -b                              # a guest, the ICB inheriting no buffers
```

## Build

On a Mac with Xcode's toolchain, never on the oracle:

```sh
guest/bundle/tools/oracle/build.sh /tmp/rgk-oracle/x86 x86_64   # for an Intel Mac
guest/bundle/tools/oracle/build.sh /tmp/rgk-oracle/arm arm64     # for an Apple-silicon Mac
```

Each probe that runs shaders loads `oracle.metallib` from its own directory, so copy it along.

## Run

On the oracle, in a potluck job that holds the GPU it measures (`--ram 1 --gpu 5300m`, or
`--gpu 630` for the Intel GPU), keeping the machine awake:

```sh
caffeinate -s ./oracle_iosurface AMD single
caffeinate -s ./oracle_iosurface Intel timing 3840 2400
caffeinate -s ./oracle_storage AMD
```

On an Apple-silicon Mac of this repository, in a potluck job holding its GPU (`--gpu apple`):
`/tmp/rgk-oracle/arm/oracle_passes Apple`.

`oracle_iosurface` also takes a script of its own steps in place of a scenario name (the steps
are listed in its header). `oracle_errors DEVICE timeout SECONDS` runs one long kernel to meet
the GPU's watchdog (about 16 s on the Intel 630, 20 s on the 5300M): run it only on a GPU that
does not drive a display. The other kernels stay near a second.
