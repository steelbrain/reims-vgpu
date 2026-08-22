# feature-matrix.sh

Compile the Vulkan backend for both supported host pathways and fail if either does not build.

## Why

The same feature set uses MoltenVK on macOS and a native Vulkan ICD on Linux:

| Pathway | Feature set | Host |
|---|---|---|
| Vulkan / MoltenVK | `--no-default-features --features host-window` | macOS |
| Vulkan / native | same | Linux |

`vendor/qemu/hw/display/meson.build` always builds that feature set. The matrix checks all targets,
the separate UEFI option-ROM workspace, and formatting for both workspaces.

## Run

```sh
scripts/feature-matrix/feature-matrix.sh              # check + native test counts
scripts/feature-matrix/feature-matrix.sh --no-counts  # compile gate only
scripts/feature-matrix/feature-matrix.sh --build      # link-level build
```

The cross target defaults to the other supported host pathway. Override it with `CROSS_TARGET=...`.
Cross-compiled binaries are checked but cannot be enumerated or run on the current host.

Warnings are reported but do not fail a cell. This is a compile gate, not runtime verification; a
backend change still needs a live boot on each affected host pathway.
