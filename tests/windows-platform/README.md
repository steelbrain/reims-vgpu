# Windows platform contracts

These checks use the production Windows modules without enabling core's Unix-only
process/death testkit. They do not start a guest, create a window or capture input.
Use a 64-bit Windows host with the repository's Rust version and a matching MinGW
UCRT toolchain. Keep UCRT's `bin` before Rustup's bundled MinGW DLLs on `PATH` when
building SDL3 with GCC; mixing those DLLs can make GCC fail before compilation.

```powershell
cargo test --manifest-path tests/windows-platform/Cargo.toml
cargo test -p reims-vgpu-window --lib --no-default-features
cargo clippy -p reims-vgpu-c --lib --no-default-features --features vulkan -- -D warnings
cargo build -p reims-vgpu-c --no-default-features --features vulkan
python tests/windows-platform/smoke.py --library target/debug --output target/windows-smoke
```

The standalone Rust tests cover real section aliases, sparse reservations, page
permissions, RAM lease lifetimes, clocks and CRT descriptor ownership. The window
tests exercise input and presentation state without opening SDL.

The smoke check compiles the repository's authored `xtask/src/smoke_guest.m`
workload as C with Windows allocation primitives and x86-64 guest page geometry.
It sends the same protocol-245 FIFO commands through the actual C API DLL, checks
GPU fill writeback, and checks two small EFI frames after changing the live RAM
source. It requires a working Vulkan driver and matching DLL/import library in
`--library`; `--output` contains temporary generated source and logs. It does not
validate native-window presentation, guest boot or desktop performance.

The workspace-wide Unix death/process tests and Linux/macOS regression gates
remain separate checks before merge.
