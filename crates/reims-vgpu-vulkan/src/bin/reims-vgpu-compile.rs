//! `reims-vgpu-compile`: the helper process that compiles a kernel's pipeline past rt's stage
//! bound for the library in the VM's process, which starts it beside its own executable, caps it
//! and reads its reply (`reims_vgpu_vulkan::rt_platform::stage`). It first sets its own
//! core-file limit to none, soft and hard, on Linux and macOS (`RLIMIT_CORE`), so its crash
//! writes no core file; Windows has no such limit of a process's own.

fn main() -> std::process::ExitCode {
    reims_vgpu_vulkan::rt_platform::stage::helper_main()
}
