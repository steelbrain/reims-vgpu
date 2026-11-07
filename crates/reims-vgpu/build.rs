//! Sets `cfg(vulkan_platform)` where the library has the Vulkan platform, `VulkanPlatform`
//! (architecture 2.2): on Linux, whose only platform it is, with or without the `vulkan` feature,
//! and on macOS with that feature, beside the Metal platform. The library's code is gated on the
//! cfg, as Cargo's features cannot depend on the target.

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rustc-check-cfg=cfg(vulkan_platform)");
    let linux = std::env::var("CARGO_CFG_TARGET_OS").is_ok_and(|os| os == "linux");
    if linux || std::env::var_os("CARGO_FEATURE_VULKAN").is_some() {
        println!("cargo::rustc-cfg=vulkan_platform");
    }
}
