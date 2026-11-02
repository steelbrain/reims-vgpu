use std::sync::atomic::{AtomicUsize, Ordering};

use super::*;
use crate::device::RvgMemoryRange;
use crate::status::RVG_ERR_UNSUPPORTED;
use crate::testkit::{Rec, t_destroy, test_platform};

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
        dirty_source: std::ptr::null(),
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

unsafe extern "C-unwind" fn t_slots(_ctx: *mut c_void, _slot_ctx: *mut c_void) -> bool {
    true
}

/// A dirty source over `rec` with only the functions that make it one.
fn dirty_table(rec: &Arc<Rec>, fd: i32) -> RvgDirtySource {
    RvgDirtySource {
        struct_size: crate::sized::size_u32::<RvgDirtySource>(),
        kvm_vm_fd: fd,
        ctx: rec.ctx(),
        destroy: Some(t_destroy),
        start: None,
        flush_vcpus: None,
        for_each_slot: Some(t_slots),
        take_host_writes: None,
    }
}

/// A graphics device over a memory map keeps its dirty source until its end; one without a map,
/// a source that is none and a refusal end it before the create returns; a source the create
/// cannot read is refused, ending no table; a second-version config has no source
#[test]
fn a_graphics_dirty_source_lives_with_its_device() {
    let p = test_platform();
    let src = Rec::new();
    let cb = callbacks();
    let host = reims_vgpu_core::platform::Platform::vm(&*p)
        .allocate_anywhere(0x10_0000)
        .unwrap();
    let range = RvgMemoryRange {
        physical_address: 0x10_0000,
        physical_length: 0x10_0000,
        host_address: host.as_ptr(),
    };
    let map = RvgMemoryMap {
        ranges: &range,
        range_count: 1,
    };
    let d = dirty_table(&src, 9);
    let mut c = config(&cb);
    c.memory_map = &map;
    c.dirty_source = &d;
    let create = |c: &RvgGraphicsDeviceConfig| {
        let mut device = std::ptr::null_mut();
        // SAFETY: a live config and out-pointer.
        let s = unsafe { rvg_graphics_device_create(c, &mut device) };
        (s, device)
    };
    let (s, dev) = create(&c);
    assert_eq!(s, RVG_OK);
    assert!(src.destroys().is_empty());
    // SAFETY: the device made above, not used again.
    unsafe { rvg_graphics_device_release(dev) };
    assert_eq!(src.wait_destroys(1).len(), 1);
    assert!(src.lines().is_empty(), "no harvest was wanted");

    c.memory_map = std::ptr::null();
    let (s, dev) = create(&c);
    assert_eq!(s, RVG_OK);
    assert_eq!(
        src.destroys().len(),
        2,
        "no map: ended before the create returned"
    );
    // SAFETY: the device made above, not used again.
    unsafe { rvg_graphics_device_release(dev) };

    c.memory_map = &map;
    let none = RvgDirtySource { kvm_vm_fd: -1, ..d };
    c.dirty_source = &none;
    let (s, dev) = create(&c);
    assert_eq!(s, RVG_OK);
    assert_eq!(
        src.destroys().len(),
        3,
        "none: ended before the create returned"
    );
    // SAFETY: the device made above, not used again.
    unsafe { rvg_graphics_device_release(dev) };

    // Files without a map are refused after the tables are the library's.
    let file = RvgMemoryRangeFile {
        fd: -1,
        flags: 0,
        offset: 0,
    };
    c.dirty_source = &d;
    c.memory_map = std::ptr::null();
    c.memory_files = &file;
    assert_eq!(create(&c).0, RVG_ERR_INVALID_ARGUMENT);
    assert_eq!(src.destroys().len(), 4, "a refusal ends it");
    c.memory_map = &map;
    c.memory_files = std::ptr::null();

    let small = RvgDirtySource {
        struct_size: 4,
        ..d
    };
    c.dirty_source = &small;
    let before = DESTROYS.load(Ordering::SeqCst);
    assert_eq!(create(&c).0, RVG_ERR_INVALID_ARGUMENT);
    assert_eq!(src.destroys().len(), 4);
    assert_eq!(
        DESTROYS.load(Ordering::SeqCst),
        before,
        "no destroy for a size"
    );

    c.dirty_source = &d;
    c.struct_size = std::mem::offset_of!(RvgGraphicsDeviceConfig, dirty_source) as u32;
    let (s, dev) = create(&c);
    assert_eq!(s, RVG_OK, "a second-version config has no source");
    assert_eq!(src.destroys().len(), 4);
    // SAFETY: the device made above, not used again.
    unsafe { rvg_graphics_device_release(dev) };
}
