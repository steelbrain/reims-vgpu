//! Windows OS contracts using the production implementations and core traits.
//! These tests allocate local memory; they do not start a VM or capture input.
#![cfg(all(target_os = "windows", test))]

#[path = "../../crates/reims-vgpu-vulkan/src/os/windows/clock.rs"]
pub mod clock;
pub mod device;
#[path = "../../crates/reims-vgpu-vulkan/src/os/windows/failure.rs"]
pub mod failure;
#[path = "../../crates/reims-vgpu-vulkan/src/os/windows/mapping.rs"]
pub mod mapping;
#[path = "../../crates/reims-vgpu-vulkan/src/os/windows/reservation.rs"]
pub mod reservation;
#[path = "../../crates/reims-vgpu-vulkan/src/os/windows/threads.rs"]
pub mod threads;
#[path = "../../crates/reims-vgpu-vulkan/src/os/windows/vm.rs"]
pub mod vm;

/// Exercise the owner used by IOSurface, not only the raw VM allocation entry.
#[test]
fn owned_surface_storage_is_zeroed_and_writable() {
    let allocation = vm::OwnMemory::new(8192).expect("allocate surface pages");
    let pointer = allocation.first() as *mut u8;
    // SAFETY: the owner retains both pages until the end of this test.
    unsafe {
        assert!(
            std::slice::from_raw_parts(pointer, 8192)
                .iter()
                .all(|byte| *byte == 0)
        );
        pointer.add(8191).write(0x5a);
        assert_eq!(pointer.add(8191).read(), 0x5a);
    }
}
