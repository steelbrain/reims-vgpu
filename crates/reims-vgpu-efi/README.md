# reims-vgpu-efi

UEFI **Graphics Output Protocol** driver for product PCI device
`reims-vgpu-pci` (vendor `0x106B`, device `0xEEEE`).

This is **not** a second QEMU display and **not** `-vga std`. OpenCore and
`boot.efi` get a normal `EFI_GRAPHICS_OUTPUT_PROTOCOL` whose framebuffer is
**BAR1 on the same PCI function** the host accelerator already uses. The C
device stays a thin shim (BAR1 RAM + pre-product scanout); this crate owns the
firmware-side business logic.

## Relationship to `reims-vgpu`

| Crate | Role | Target |
|-------|------|--------|
| `crates/reims-vgpu` | Host composition + QEMU staticlib, linking semantic and Vulkan sibling crates | host (Linux or macOS) |
| **`crates/reims-vgpu-efi`** | UEFI option-ROM PE (GOP install + Blt) | `x86_64-unknown-uefi` |

Sibling packages, separate Cargo workspaces. Do **not** add this crate as a
member of `Cargo.toml` — the UEFI `no_std` graph must not pollute host /
QEMU builds.

## What it does

1. Finds PCI `0x106B:0xEEEE` via `EFI_PCI_IO_PROTOCOL`, enables MEM/IO/BM.
2. Reads **BAR1** as a linear BGRA8 framebuffer (1920×1080 contract, matching
   `reims-vgpu-pci.c`).
3. Pool-allocates a permanent `GopCtx` and installs the standard UEFI
   **`EFI_GRAPHICS_OUTPUT_PROTOCOL`** GUID
   (`9042a9de-23dc-4a38-96fb-7aded080516a` — spec GUID, not a private ID).
4. Implements QueryMode / SetMode / Blt; fills a dark slate so the FB is
   visibly non-black before OpenCore paints.
5. Prints a stable scrape line on COM1 (`0x3F8`) for product boot gates.

## Source layout

| File | Role |
|------|------|
| `src/lib.rs` + `src/paint.rs` | **Shipped Blt/fill/copy** (host `cargo test --lib`) |
| `src/main.rs` | Entry: BAR1 → build GOP → install protocol |
| `src/pci.rs` | Product VID/DID + BAR1 base |
| `src/gop.rs` | UEFI protocol adapters → `paint` |
| `src/serial.rs` | COM1 polled UART scrape rail |
| `.cargo/config.toml` | Link as `efi_boot_service_driver` (subsystem 11) |

Host unit tests (no boot services):

```sh
cd crates/reims-vgpu-efi && cargo test --lib
```

## Build

Requires the UEFI target:

```sh
rustup target add x86_64-unknown-uefi
```

Preferred path — compile PE and wrap as a PCI expansion ROM:

```sh
crates/reims-vgpu-efi/scripts/reims-vgpu-efi-rom/reims-vgpu-efi-rom.sh
# → crates/reims-vgpu-efi/out/reims-vgpu-efi.efi
# → crates/reims-vgpu-efi/out/reims-vgpu-gop.rom
```

Manual PE only:

```sh
cd crates/reims-vgpu-efi
cargo build --release --target x86_64-unknown-uefi
```

`out/` and `target/` are gitignored; rebuild after every source change.

## Boot

`vm/boot-x86.sh --device reims-vgpu-pci` attaches the ROM when present:

```text
-device reims-vgpu-pci,...,romfile=.../reims-vgpu-gop.rom,rombar=1
```

- `REIMS_VGPU_GOP_ROM=/path/to.rom` — override
- `REIMS_VGPU_GOP_ROM=` — disable option ROM

## Packaging invariants (load-bearing)

The ROM wrapper (`scripts/reims-vgpu-efi-rom/` under this crate) asserts:

1. `55 AA` + EFI signature **`0x0EF1`** + PCIR code type **`0x03`** + PE `MZ`
2. PE **`OptionalHeader.Subsystem = 11` (BOOT_SERVICE_DRIVER)** — `efi_app` (10)
   is unloaded when `StartImage` returns, so OpenCore reports Missing GOP

## Serial gate (do not rename)

```text
reims-vgpu efi-gop: GOP installed
```

Live check also expects no `OCC: Missing compatible GOP` and an
`initialize_screen` handoff with non-zero width/height matching BAR1.

## Further reading

- Host BAR1 / scanout: `vendor/qemu/hw/display/reims-vgpu-pci.c`
- ROM packager details:
  `crates/reims-vgpu-efi/scripts/reims-vgpu-efi-rom/README.md`
