# qemu-build.sh

Builds the vendored `vendor/qemu` submodule into `vendor/qemu/build/qemu-system-<arch>`, linking the
thin device shim(s) to **`crates/reims-vgpu`**.

| Target | Typical host | Output | Pathway |
|--------|--------------|--------|---------|
| `aarch64` | Darwin / Apple Silicon | `qemu-system-aarch64` | arm64 macOS guest on macOS host (`vm/boot-arm64.sh`) |
| `x86_64` | Linux | `qemu-system-x86_64` | x86 macOS guest on Linux host (`vm/boot-x86.sh`) |

Both targets use Vulkan: the native ICD on Linux and MoltenVK on macOS. The target defaults by host
OS (`aarch64` on Darwin, `x86_64` on Linux).

## What it does

`vendor/qemu` already carries the project patches — this script does **not** clone or patch, it
builds. Steps:

1. Populates the submodule if needed (`git submodule update --init vendor/qemu`).
2. Resolves `--target` (or `QEMU_TARGET`).
3. Builds `crates/reims-vgpu` as a staticlib and links it into the device shim:
   - the staticlib composes `reims-vgpu-core` with the `reims-vgpu-vulkan` executor (native Vulkan
     on Linux, MoltenVK on macOS).
4. **aarch64:** expects `CONFIG_VMAPPLE`, HVF/Cocoa configure, verifies `-M vmapple`.
5. **x86_64:** `x86_64-softmmu`, HVF/Cocoa off; lists PCI/sysbus device help as applicable.

Re-runs are idempotent (skips configure when the target stamp matches). Switching target forces
reconfigure. Patch record: `vendor/qemu-patches/`.

## Run

```sh
# Explicit pathway builds
scripts/qemu-build/qemu-build.sh --target aarch64
scripts/qemu-build/qemu-build.sh --target x86_64

# Point the matching boot script at the binary
QEMU_BIN=$PWD/vendor/qemu/build/qemu-system-aarch64 vm/boot-arm64.sh --device reims-vgpu-mmio --testing
QEMU_BIN=$PWD/vendor/qemu/build/qemu-system-x86_64 vm/boot-x86.sh --device reims-vgpu-pci --testing
```

### Requirements

- **Both:** cargo (`crates/reims-vgpu`), ninja, meson, pkg-config, glib, pixman.
- **aarch64:** macOS, Xcode CLT, HVF/Cocoa, Vulkan loader, and MoltenVK ICD.
- **x86_64:** Linux QEMU build deps; KVM for boots.
