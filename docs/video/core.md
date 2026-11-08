# Core

`crates/reims-vgpu-video-core` is what the host's video arms share, independent of the host,
the driver and QEMU. It links nothing of the library's: the library hands it a runtime instead.
Each module's doc says what it holds; this file is the overview.

## The runtime and the locks

- **`runtime`**: the threads and the clock the video crates run on, a `Runtime` of three calls
  (a thread's start, the monotonic clock's time, one timed wait). The library's is its
  platform's (reims-vgpu's `video/runtime.rs`, `PlatformRuntime`): a thread is a worker of the
  platform's, named as the crate asks and marked as the library's, so that a panic on it ends
  the process at once, and every bound is kept on the platform's monotonic clock, which a test
  makes virtual. `StdRuntime` is std's threads and clock, for the examples and the hardware
  tests. A `Bell` is the one place a video crate waits on a condition variable with a
  deadline.
- **`sync`**: the locks, `parking_lot`'s, the types reims-vgpu-core's `sync` re-exports.

## A guest surface's memory

- **`surfaces`**: a guest surface's layout (`SurfaceLayout`, `PlaneLayout`); **`fourcc`** the
  four-character codes that name its pixel format.
- **`device_memory`**: a surface's memory as a device reaches it, for an arm whose GPU writes
  a decoded picture into the guest's pages itself.
  - `DeviceMemory` gives the pages as a dma-buf of byte ranges (`dma_buf_of`), for a GPU that
    imports one (VA-API), or as one contiguous view in the host's address space
    (`host_view`), for one that registers host memory (CUDA's `cuMemHostRegister`). Each is
    the same pages the guest reads. The reims-graphics device's planes answer it over the
    library's own view of the guest's RAM (reims-vgpu's `PlanePages`, in
    `graphics/decode/video.rs`).
  - `MemoryIdentity` names one mapping of the pages: the same for every hold on it, and never
    the same for another mapping, so that an import kept by it is never used for pages mapped
    since. `MemoryLife` goes with the mapping, and tells what kept something by its identity.
  - `DeviceImports` keeps an arm's imports by identity, made once per mapping and dropped on a
    thread of the runtime's as soon as the mapping goes.
  - `GuestTarget` is a surface's layout checked against its memory, so that a device told to
    write its planes writes nowhere else.

## Bounded waits

**`device_wait`**: no driver's blocking wait has a bound of its own (NVDEC's map, NVENC's lock,
CUDA's synchronous copy, VA-API's sync), so each backend asks the driver's own non-blocking
question until the work is done or `DEVICE_WAIT` (5 s) has passed (`poll_until`). Past it, the
work fails as the backend's malfunction, and the decoder or encoder fails each later call at
once. What the device may still write is kept until it is done: the backend's memory and
handle, an import of a guest buffer the device writes itself among them, go to `Lingering`,
whose thread asks each kept work's question every `LINGER_POLL` (100 ms) and drops it once the
device is done. Work a device never finishes is kept until the process ends, since freeing it
would hand memory the device may still write back to the allocator or to the guest.

## What a backend answers

**`backend`**: the pixel formats a backend writes (`PixelFormat`), whether its device wrote a
picture into a guest surface itself (`DeviceWrite`, `Declined` with why where it took the CPU's
copy instead), and why it could not do what it was asked (`BackendError`: unsupported, bad
data, a skipped leading picture, or the backend's malfunction).
