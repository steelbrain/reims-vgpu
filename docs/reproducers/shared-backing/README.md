# Z13 DMA-BUF GPU data test

Linux x86_64 only. Selects Radeon 8060S PCI 1002:1586. Requires Vulkan/RADV, Rust, accessible /dev/udmabuf and a 4096-byte host page size. No guest or modified driver required.

```sh
cargo build --locked
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/radeon_icd.json timeout 45 target/debug/z13-dmabuf-gpu-probe 10
```

The argument is 1–20 iterations. Each iteration tests contiguous 64 KiB, 1 MiB, 16 MiB exports and reordered/discontiguous memfd offsets of 64 KiB (16 list entries) and 1 MiB (256 entries). Scattered cases use twice as much backing, leaving unexported guard pages. Physical fragmentation is not measured.

Pattern values depend on word offset, iteration and phase. CPU writes through the DMA-BUF mapping within DMA_BUF_IOCTL_SYNC brackets. Vulkan acquires foreign queue ownership, copies the imported bytes to independent coherent staging memory, then overwrites the imported bytes from a different staging region, releases ownership and signals a fence. After fence completion and CPU synchronization, every 32-bit word is checked in the readback, packed mapping and original backing; all unexported guard pages must retain their original pattern. Vulkan memory is destroyed only after the fence. After each case the test verifies no named test-memfd mapping or descriptor remains in its process.

This validates a serialized transfer workload, not shader semantics, concurrent CPU/GPU access, Reims guest-page recycling, a live VM, long-run driver pin accounting, or performance. Kernel list limit on the Z13 is 1024, so the probe deliberately stays below it. No validation layer was enabled; build and clippy with warnings denied passed. A failure aborts this disposable process; explicit GPU waits are bounded at five seconds and the outer timeout bounds the run.

The source hardcodes Linux x86_64 ioctl encodings checked against installed headers. Vulkan receives a CLOEXEC duplicate of the DMA-BUF fd; the probe retains the original for CPU synchronization. Test name, source and results contain no guest material.
