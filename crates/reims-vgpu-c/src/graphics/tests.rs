use std::sync::atomic::{AtomicUsize, Ordering};

use super::*;
use crate::status::RVG_ERR_UNSUPPORTED;

static DESTROYS: AtomicUsize = AtomicUsize::new(0);

unsafe extern "C-unwind" fn destroy(_ctx: *mut c_void) {
    DESTROYS.fetch_add(1, Ordering::SeqCst);
}

unsafe extern "C-unwind" fn raise(_ctx: *mut c_void, _vector: u32) {}

fn callbacks() -> RvgGraphicsCallbacks {
    RvgGraphicsCallbacks {
        struct_size: size_of::<RvgGraphicsCallbacks>() as u32,
        ctx: std::ptr::null_mut(),
        destroy: Some(destroy),
        raise_interrupt: Some(raise),
        scanout_written: None,
    }
}

fn config(callbacks: &RvgGraphicsCallbacks) -> RvgGraphicsDeviceConfig {
    RvgGraphicsDeviceConfig {
        struct_size: size_of::<RvgGraphicsDeviceConfig>() as u32,
        callbacks,
        window: std::ptr::null(),
        memory_map: std::ptr::null(),
        memory_files: std::ptr::null(),
    }
}

/// A headless device: its region is mapped, its registers answer, a second doorbell or a closed
/// descriptor is refused, and its table's destroy runs once its threads end after the release
#[test]
fn a_headless_graphics_device_lives_and_ends() {
    let cb = callbacks();
    let c = config(&cb);
    let mut device = std::ptr::null_mut();
    // SAFETY: a live config and out-pointer.
    assert_eq!(
        unsafe { rvg_graphics_device_create(&c, &mut device) },
        RVG_OK
    );
    let mut size = 0;
    // SAFETY: a live device; the size is writable.
    let base = unsafe { rvg_graphics_device_region(device, &mut size) };
    assert!(!base.is_null() && base.addr() % 4096 == 0);
    assert_eq!(size, reims_graphics::REGION_SIZE);
    // SAFETY: a live device.
    unsafe {
        assert_eq!(
            rvg_graphics_device_mmio_read(device, 0),
            reims_graphics::wire::RGFX_MAGIC
        );
        rvg_graphics_device_mmio_write(device, 0x10, 1);
        assert_eq!(rvg_graphics_device_mmio_read(device, 0x10), 1);
        rvg_graphics_device_reset(device);
        assert_eq!(rvg_graphics_device_mmio_read(device, 0x10), 0);
        assert_eq!(
            rvg_graphics_device_set_doorbell(device, -1),
            RVG_ERR_INVALID_ARGUMENT
        );
        assert_eq!(rvg_graphics_device_set_doorbell(device, 0), RVG_OK);
        assert_eq!(
            rvg_graphics_device_set_doorbell(device, 0),
            RVG_ERR_INVALID_ARGUMENT
        );
    }
    let before = DESTROYS.load(Ordering::SeqCst);
    // SAFETY: the device, not used again.
    unsafe { rvg_graphics_device_release(device) };
    for _ in 0..2000 {
        if DESTROYS.load(Ordering::SeqCst) > before {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert_eq!(DESTROYS.load(Ordering::SeqCst), before + 1);
}

/// A config the library refuses for its size runs no destroy; NULL config and out-pointer are
/// refused
#[test]
fn refused_graphics_configs() {
    let cb = callbacks();
    let mut c = config(&cb);
    let mut device = std::ptr::null_mut();
    // SAFETY: NULLs and live pointers.
    unsafe {
        assert_eq!(
            rvg_graphics_device_create(std::ptr::null(), &mut device),
            RVG_ERR_INVALID_ARGUMENT
        );
        let before = DESTROYS.load(Ordering::SeqCst);
        assert_eq!(
            rvg_graphics_device_create(&c, std::ptr::null_mut()),
            RVG_ERR_INVALID_ARGUMENT
        );
        assert_eq!(
            DESTROYS.load(Ordering::SeqCst),
            before + 1,
            "a refusal's destroy"
        );
        c.struct_size = 4;
        assert_eq!(
            rvg_graphics_device_create(&c, &mut device),
            RVG_ERR_INVALID_ARGUMENT
        );
        let mut big = [0u8; 64];
        std::ptr::copy_nonoverlapping(
            std::ptr::from_ref(&config(&cb)).cast::<u8>(),
            big.as_mut_ptr(),
            size_of::<RvgGraphicsDeviceConfig>(),
        );
        big[0] = 64;
        big[60] = 1;
        assert_eq!(
            rvg_graphics_device_create(big.as_ptr().cast(), &mut device),
            RVG_ERR_UNSUPPORTED
        );
        assert_eq!(
            DESTROYS.load(Ordering::SeqCst),
            before + 1,
            "no destroy for a size"
        );
        // Files without a map are refused, after the tables are the library's.
        let file = RvgMemoryRangeFile {
            fd: -1,
            flags: 0,
            offset: 0,
        };
        let mut c = config(&cb);
        c.memory_files = &file;
        assert_eq!(
            rvg_graphics_device_create(&c, &mut device),
            RVG_ERR_INVALID_ARGUMENT
        );
        assert_eq!(
            DESTROYS.load(Ordering::SeqCst),
            before + 2,
            "a refusal's destroy"
        );
    }
    assert!(device.is_null());
}

/// A host built against the first version passes a config that ends at `window`: it makes a
/// device with no memory map
#[test]
fn a_first_version_graphics_config_makes_a_device() {
    let cb = callbacks();
    let mut c = config(&cb);
    c.struct_size = RvgGraphicsDeviceConfig::V1_SIZE as u32;
    // Past the first version's size, as a host built against it leaves its memory.
    c.memory_files = std::ptr::dangling();
    let mut device = std::ptr::null_mut();
    // SAFETY: a live config and out-pointer; the device is released.
    unsafe {
        assert_eq!(rvg_graphics_device_create(&c, &mut device), RVG_OK);
        assert!(!device.is_null());
        rvg_graphics_device_release(device);
    }
}
