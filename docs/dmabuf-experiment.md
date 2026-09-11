# Serialized Linux DMA-BUF experiment

Build with `scripts/qemu-build/qemu-build.sh --target x86_64 --backend vulkan --experimental-dmabuf`. This requires the matching QEMU submodule patch, which adds the Meson option and ABI v21 backing callback. Default builds do not enable the experimental Rust feature. The option changes compiled availability; Vulkan extension and buffer-capability checks still gate actual imports.

Scope: packed resource imports on Linux x86_64. HostOps resolves the file and offset for each guest page; the memory owner coalesces adjacent ranges without reordering, then uses udmabuf to export sealed shared backing. The import bound checks that the exported length matches. Kernel exporter errors are typed refusals and retain existing fallback behavior. RAMBlock imports keep their existing path. No vendor-ID gate or userptr flag modification is used.

The Vulkan importer first tries the existing host-pointer path, then the supplied DMA-BUF payload when available. It transfers fd ownership only on successful allocation, uses the common memory-type selector, and binds a dedicated buffer. Such an allocation cannot be used as a host-pointer child image; that path explicitly refuses it.

This first implementation disables the asynchronous queue worker when DMA-BUF capability is enabled. Submissions involving the live DMA-BUF registry acquire/release foreign ownership and run to completion synchronously, with CPU cache maintenance before and afterward. Only then is the guest's fence/timeline completion submitted. The registry currently brackets all live DMA-BUF buffers, not a minimal per-submission set. This is a correctness experiment, not a demonstrated throughput improvement.

Packed resource replacement, unmap, task retirement and reset enqueue their owned aliases for the existing fence-safe host cleanup. This includes the earlier local packed-alias lifetime fix, which overlaps the work in upstream PR #77. RAMBlock imports retain their VM lifetime.

Normal retirement marks a lease unavailable for future submissions before Vulkan destruction and alias release. A failed submission/wait does not prove GPU completion: the transport is poisoned, its payloads are quarantined, and their aliases are released only after device destruction. Temporary Vulkan objects from an unconfirmed submission are also left for device destruction.

Validation on the Z13: the ignored `backend::vulkan::engine::dmabuf::tests::packed_dma_buffers_preserve_data_order_and_retire` test is explicitly run with the experimental feature and RADV ICD. It covers fallback, reads, writes, scattered offset order, untouched pages, and non-resurrection after retirement. It is opt-in because it specifically requires an accessible exporter and a driver that refuses the shared host-pointer control. General memory geometry and ABI tests remain ordinary tests.

Remaining qualification: precise per-submission access tracking; broader failure injection; pinned-memory/resource-budget behavior; host/guest concurrency and long-run validation; full conformance against native and repeated guest controls; matched playback measurements. Standalone and boot success must not be presented as evidence that this fixes audio or video skips. Do not enable this by default or replace a known working VM with it.

## Upstream history and scope of this proposal

Reims previously implemented guest-page DMA-BUF export in commit
`089280a93fecc1c1397119a2ce6f055c0805e401`; commit
`159fde84a72458dfd3dc7e4296913a7751eda58f` replaced that path with bounded
RAMBlock host-pointer imports, and ABI v18 removed the old export callback.
This proposal is not the first DMA-BUF implementation in Reims. It adds a
narrow, opt-in packed-buffer fallback to the current memory owner model,
motivated by reproducible shared-backing host-pointer import refusal on RADV.
Issue #29 includes older DMA-BUF experiments and the move to host-pointer
imports. PR #79 concerns outstanding guest writes at page release; this
proposal does not claim to replace that work or fix page-table corruption.

The exporter requires shared, fd-backed guest RAM with suitable seals and
access to `/dev/udmabuf`. The standard upstream plain `-m` allocation has no
exportable backing. For an existing QEMU launch, use a matching-size
`-object memory-backend-memfd,id=ram,size=16G,share=on` and select it through
`-machine memory-backend=ram` (merge the property into any existing machine
argument). Keep `-m` consistent with the memory object's size. No host
permission or guest-memory configuration is changed by the build option.

`docs/reproducers/shared-backing` contains the standalone Rust GPU probe and
recorded results. It requires no macOS guest or extracted shader material.
