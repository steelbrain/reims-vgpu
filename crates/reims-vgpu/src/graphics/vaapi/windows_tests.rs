//! The arm on Windows, on the rt platform's device: `vaon12` on its DXGI adapter, where each
//! picture lands by the CPU's copy (`rt_windows`).

use std::sync::Arc;

/// On the rt platform on Windows the arm opens on the DXGI adapter by the LUID Vulkan answers for
/// the device, answers what the driver decodes and no encoder, and lands its pictures in device
/// caches, by the CPU's copy, decoding into none (`super::rt_windows`).
#[test]
#[ignore = "needs vaon12 on the rt device's GPU: run on the Windows host"]
fn the_rt_platforms_arm_opens_on_the_devices_adapter_and_decodes_alone() {
    use reims_vgpu_vulkan::RtPlatform;

    let platform = Arc::new(RtPlatform::new());
    crate::entry::run(|_| {
        let device = platform.system_default_device().expect("an rt device");
        assert!(
            device.host_gpu().luid.is_some(),
            "the device names no adapter"
        );
        let runtime = crate::video::runtime::PlatformRuntime::of(&platform);
        let video = super::open_on_rt(&device, runtime).expect("VA-API on the rt device's GPU");
        let caps = video.caps();
        assert!(!caps.is_empty());
        assert!(
            caps.iter().all(|c| c.direction == super::RGFX_VIDEO_DECODE),
            "{caps:?}"
        );
        assert!(video.guest_encoders().is_empty());
        assert!(video.guest_decodes());
        assert!(video.lands_in_device_caches());
        assert!(!video.writes_into_device_caches());
    });
}
