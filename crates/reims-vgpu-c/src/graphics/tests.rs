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
        cursor_glyph: None,
        cursor_show: None,
        cursor_move: None,
        route_doorbell: None,
        route_doorbell_page: None,
        cursor_glyph_at: None,
        cursor_show_at: None,
        cursor_move_at: None,
        pool_add: None,
        pool_remove: None,
    }
}

fn config(callbacks: &RvgGraphicsCallbacks) -> RvgGraphicsDeviceConfig {
    RvgGraphicsDeviceConfig {
        struct_size: size_of::<RvgGraphicsDeviceConfig>() as u32,
        callbacks,
        window: std::ptr::null(),
        memory_map: std::ptr::null(),
        memory_files: std::ptr::null(),
        guest_arch: RVG_GUEST_ARCH_X86_64,
        doorbell_pages: 0,
        display_count: 0,
        display_windows: std::ptr::null(),
        pool_physical_address: 0,
        pool_physical_length: 0,
    }
}

/// A doorbell no host has open, and one it has, which the test owns: on Linux and macOS -1 and
/// stdin's descriptor.
#[cfg(not(windows))]
fn doorbells() -> (i32, i32, ()) {
    (-1, 0, ())
}

/// [`doorbells`] on Windows: NULL, which names no handle, and a manual-reset event, as QEMU's
/// EventNotifier makes one, closed with the handle the test keeps.
#[cfg(windows)]
fn doorbells() -> (*mut c_void, *mut c_void, std::os::windows::io::OwnedHandle) {
    use std::os::windows::io::{AsRawHandle as _, FromRawHandle as _, OwnedHandle};

    use windows_sys::Win32::System::Threading::CreateEventW;
    // SAFETY: a new unnamed manual-reset event, not set.
    let event = unsafe { CreateEventW(std::ptr::null(), 1, 0, std::ptr::null()) };
    assert!(!event.is_null(), "{}", std::io::Error::last_os_error());
    // SAFETY: `CreateEventW` made it, and nothing else owns it.
    let event = unsafe { OwnedHandle::from_raw_handle(event) };
    (std::ptr::null_mut(), event.as_raw_handle(), event)
}

/// A headless device: its region is mapped, its registers answer, a second doorbell or a closed
/// one is refused, and its table's destroy runs once its threads end after the release
#[test]
fn a_headless_graphics_device_lives_and_ends() {
    let cb = callbacks();
    let c = config(&cb);
    let (closed, open, _kept) = doorbells();
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
    assert_eq!(size, reims_vgpu_graphics::REGION_SIZE);
    // SAFETY: a live device.
    unsafe {
        assert_eq!(
            rvg_graphics_device_mmio_read(device, 0),
            reims_vgpu_graphics::wire::RGFX_MAGIC
        );
        rvg_graphics_device_mmio_write_taking(device, 0x10, 1);
        assert_eq!(rvg_graphics_device_mmio_read(device, 0x10), 1);
        rvg_graphics_device_reset(device);
        assert_eq!(rvg_graphics_device_mmio_read(device, 0x10), 0);
        assert_eq!(
            rvg_graphics_device_set_doorbell(device, closed),
            RVG_ERR_INVALID_ARGUMENT
        );
        assert_eq!(rvg_graphics_device_set_doorbell(device, open), RVG_OK);
        assert_eq!(
            rvg_graphics_device_set_doorbell(device, open),
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
        let mut big = [0u8; 96];
        std::ptr::copy_nonoverlapping(
            std::ptr::from_ref(&config(&cb)).cast::<u8>(),
            big.as_mut_ptr(),
            size_of::<RvgGraphicsDeviceConfig>(),
        );
        big[0] = 96;
        big[88] = 1;
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
            #[cfg(not(windows))]
            fd: -1,
            #[cfg(windows)]
            handle: std::ptr::null_mut(),
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
        // A guest architecture the library does not know is refused, after the tables are the
        // library's.
        let mut c = config(&cb);
        c.guest_arch = RVG_GUEST_ARCH_ARM64 + 1;
        assert_eq!(
            rvg_graphics_device_create(&c, &mut device),
            RVG_ERR_INVALID_ARGUMENT
        );
        assert_eq!(
            DESTROYS.load(Ordering::SeqCst),
            before + 3,
            "a refusal's destroy"
        );
    }
    assert!(device.is_null());
}

/// The device's answer of `isFramebufferReadSupported` in the blob the guest reads, of a device
/// made from `c`.
fn framebuffer_read(c: &RvgGraphicsDeviceConfig) -> u8 {
    use reims_vgpu_graphics::wire::{RGFX_DEVICE_INFO_OFFSET, RgfxDeviceInfo};
    // is_framebuffer_read_supported, third of the blob's flags.
    let at = RGFX_DEVICE_INFO_OFFSET as usize + std::mem::offset_of!(RgfxDeviceInfo, flags) + 2;
    let mut device = std::ptr::null_mut();
    // SAFETY: a live config and out-pointer; then a live device, whose region holds the blob
    // while it lives, released.
    unsafe {
        assert_eq!(rvg_graphics_device_create(c, &mut device), RVG_OK);
        let answer = rvg_graphics_device_region(device, std::ptr::null_mut())
            .cast::<u8>()
            .add(at)
            .read();
        rvg_graphics_device_release(device);
        answer
    }
}

/// The guest's architecture decides the device's `isFramebufferReadSupported`: an arm64 guest's
/// is the platform's (the fake's yes), an x86-64 guest's no; a config of the first version, whose
/// `struct_size` ends before `guest_arch`, serves an x86-64 guest whatever lies past its size
#[test]
fn the_guest_s_architecture_decides_framebuffer_read() {
    let cb = RvgGraphicsCallbacks {
        destroy: None,
        ..callbacks()
    };
    let mut c = config(&cb);
    assert_eq!(framebuffer_read(&c), 0);
    c.guest_arch = RVG_GUEST_ARCH_ARM64;
    assert_eq!(framebuffer_read(&c), 1);
    // The first version's size, with an arm64 guest past it, which the library does not read.
    c.struct_size = std::mem::offset_of!(RvgGraphicsDeviceConfig, guest_arch) as u32;
    assert_eq!(framebuffer_read(&c), 0);
}

/// A callback table of the first version, whose `struct_size` ends before `route_doorbell`, is
/// taken, whatever lies past its size: the host routes no doorbell
#[test]
fn a_callback_table_without_route_doorbell_is_taken() {
    let cb = RvgGraphicsCallbacks {
        struct_size: std::mem::offset_of!(RvgGraphicsCallbacks, route_doorbell) as u32,
        destroy: None,
        ..callbacks()
    };
    let c = config(&cb);
    let mut device = std::ptr::null_mut();
    // SAFETY: a live config and out-pointer; then a live device, released.
    unsafe {
        assert_eq!(rvg_graphics_device_create(&c, &mut device), RVG_OK);
        rvg_graphics_device_release(device);
    }
}

/// The doorbell pages the host maps decide the wire's `DOORBELL_PAGES` (`FEATURES_HI` bit 14): a
/// config with `doorbell_pages` 1 offers it, one with 0 does not, and so does a config whose
/// `struct_size` ends before the field, whatever lies past it; any other value is refused. A
/// write of BAR 4 that names no bound page is dropped. A callback table whose `struct_size` ends
/// before `route_doorbell_page` is taken.
#[test]
fn the_doorbell_pages_the_host_maps_decide_the_feature() {
    const PAGES: u32 = 1 << 14;
    let offered = |c: &RvgGraphicsDeviceConfig| {
        let mut device = std::ptr::null_mut();
        // SAFETY: a live config and out-pointer; then a live device, released.
        unsafe {
            assert_eq!(rvg_graphics_device_create(c, &mut device), RVG_OK);
            let hi = rvg_graphics_device_mmio_read(device, 0x00c);
            rvg_graphics_device_doorbell_write(device, 0, 0x1000);
            rvg_graphics_device_release(device);
            hi & PAGES != 0
        }
    };
    let cb = RvgGraphicsCallbacks {
        struct_size: std::mem::offset_of!(RvgGraphicsCallbacks, route_doorbell_page) as u32,
        destroy: None,
        ..callbacks()
    };
    let mut c = config(&cb);
    assert!(!offered(&c));
    c.doorbell_pages = 1;
    assert!(offered(&c));
    c.struct_size = std::mem::offset_of!(RvgGraphicsDeviceConfig, doorbell_pages) as u32;
    assert!(!offered(&c));
    c.struct_size = size_of::<RvgGraphicsDeviceConfig>() as u32;
    c.doorbell_pages = 2;
    let mut device = std::ptr::null_mut();
    // SAFETY: a live config and out-pointer.
    assert_eq!(
        unsafe { rvg_graphics_device_create(&c, &mut device) },
        RVG_ERR_INVALID_ARGUMENT
    );
}

/// `MULTI_DISPLAY`, `FEATURES_HI` bit 16, read from a device made from `c`, with what reading
/// each display and one past them answers.
fn displays_of(c: &RvgGraphicsDeviceConfig) -> (bool, Vec<(RvgStatus, u32, u32)>, RvgDisplayCoord) {
    const MULTI_DISPLAY: u32 = 1 << 16;
    let mut device = std::ptr::null_mut();
    // SAFETY: a live config and out-pointer; then a live device, its reads given NULL buffers
    // and writable sizes, released.
    unsafe {
        assert_eq!(rvg_graphics_device_create(c, &mut device), RVG_OK);
        let hi = rvg_graphics_device_mmio_read(device, 0x00c);
        let reads = (0..=RVG_GRAPHICS_DISPLAYS_MAX)
            .map(|display| {
                let (mut width, mut height) = (7, 7);
                let status = rvg_graphics_device_read_display_at(
                    device,
                    display,
                    std::ptr::null_mut(),
                    0,
                    &mut width,
                    &mut height,
                );
                (status, width, height)
            })
            .collect();
        let past = rvg_graphics_device_cursor_position_at(device, RVG_GRAPHICS_DISPLAYS_MAX);
        rvg_graphics_device_release(device);
        (hi & MULTI_DISPLAY != 0, reads, past)
    }
}

/// A config's `display_count` makes that many displays, 0 and 1 one, as a config whose
/// `struct_size` ends before the field: two or more offer `MULTI_DISPLAY`, and each display reads
/// back at its mode, 1920 x 1080 with no window, where a display past them is refused with a
/// size of 0 x 0 and its cursor at (0, 0); a callback table whose `struct_size` ends before
/// `cursor_glyph_at` is taken
#[test]
fn the_display_count_makes_the_displays() {
    let cb = RvgGraphicsCallbacks {
        struct_size: std::mem::offset_of!(RvgGraphicsCallbacks, cursor_glyph_at) as u32,
        destroy: None,
        ..callbacks()
    };
    let shown = (RVG_ERR_INVALID_ARGUMENT, 1920, 1080);
    let past = (RVG_ERR_INVALID_ARGUMENT, 0, 0);
    let mut c = config(&cb);
    for count in 0..=RVG_GRAPHICS_DISPLAYS_MAX {
        c.display_count = count;
        let (multi, reads, cursor) = displays_of(&c);
        let made = count.max(1) as usize;
        assert_eq!(multi, made > 1, "{count} displays");
        assert!(
            reads[..made].iter().all(|r| *r == shown),
            "{count}: {reads:?}"
        );
        assert!(
            reads[made..].iter().all(|r| *r == past),
            "{count}: {reads:?}"
        );
        assert_eq!(cursor, RvgDisplayCoord { x: 0, y: 0 });
    }
    // A host built before the field reads one display whatever lies past its size.
    c.display_count = 3;
    c.struct_size = std::mem::offset_of!(RvgGraphicsDeviceConfig, display_count) as u32;
    let (multi, reads, _) = displays_of(&c);
    assert!(!multi);
    assert_eq!(reads[1], past);
}

/// A display count past `RVG_GRAPHICS_DISPLAYS_MAX`, and display windows not given exactly where
/// display 0's window is, are refused once the tables are the library's, running their destroys
#[test]
fn refused_display_configs() {
    let cb = callbacks();
    let window = RvgWindowConfig {
        struct_size: size_of::<RvgWindowConfig>() as u32,
        callbacks: std::ptr::null(),
        mode_width: 0,
        mode_height: 0,
    };
    let one: [*const RvgWindowConfig; 1] = [&window];
    let none: [*const RvgWindowConfig; 1] = [std::ptr::null()];
    let refused = |c: &RvgGraphicsDeviceConfig| {
        let mut device = std::ptr::null_mut();
        let before = DESTROYS.load(Ordering::SeqCst);
        // SAFETY: a live config and out-pointer.
        let status = unsafe { rvg_graphics_device_create(c, &mut device) };
        assert!(device.is_null());
        (status, DESTROYS.load(Ordering::SeqCst) - before)
    };
    let mut c = config(&cb);
    c.display_count = RVG_GRAPHICS_DISPLAYS_MAX + 1;
    assert_eq!(refused(&c), (RVG_ERR_INVALID_ARGUMENT, 1));
    // Windows of displays 1 and up with no window of display 0's.
    c.display_count = 2;
    c.display_windows = one.as_ptr();
    assert_eq!(refused(&c), (RVG_ERR_INVALID_ARGUMENT, 1));
    // Display 0's window with none of display 1's, or a NULL one.
    c.window = &window;
    c.display_windows = std::ptr::null();
    assert_eq!(refused(&c), (RVG_ERR_INVALID_ARGUMENT, 1));
    c.display_windows = none.as_ptr();
    assert_eq!(refused(&c), (RVG_ERR_INVALID_ARGUMENT, 1));
    // Windows given for a device of one display.
    c.display_count = 1;
    c.display_windows = one.as_ptr();
    assert_eq!(refused(&c), (RVG_ERR_INVALID_ARGUMENT, 1));
}

static MOVES: AtomicUsize = AtomicUsize::new(0);

unsafe extern "C-unwind" fn cursor_moved(_ctx: *mut c_void) {
    MOVES.fetch_add(1, Ordering::SeqCst);
}

/// The host is told of each move of the cursor, a reset's among them, and reads where the hot
/// spot is
#[test]
fn the_host_is_told_of_the_cursor_s_moves() {
    let moves = || {
        let cb = RvgGraphicsCallbacks {
            destroy: None,
            cursor_move: Some(cursor_moved),
            ..callbacks()
        };
        let c = config(&cb);
        let mut device = std::ptr::null_mut();
        let before = MOVES.load(Ordering::SeqCst);
        // SAFETY: a live config and out-pointer; then a live device, released.
        let at = unsafe {
            assert_eq!(rvg_graphics_device_create(&c, &mut device), RVG_OK);
            rvg_graphics_device_reset(device);
            for _ in 0..2000 {
                if MOVES.load(Ordering::SeqCst) > before {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            // Past a move the display thread would run late.
            std::thread::sleep(std::time::Duration::from_millis(20));
            let at = rvg_graphics_device_cursor_position(device);
            rvg_graphics_device_release(device);
            at
        };
        assert_eq!(at, RvgDisplayCoord { x: 0, y: 0 });
        MOVES.load(Ordering::SeqCst) - before
    };
    assert_eq!(moves(), 1);
}

/// The read of the display: with no buffer it is refused, with the mode's size given, 1920 x
/// 1080 with no window; a buffer short of the mode's pixels is refused with nothing written; a
/// buffer of them is read, black where the guest has no scanout, the size pointers NULL
#[test]
fn a_graphics_device_s_display_reads_back() {
    let cb = RvgGraphicsCallbacks {
        destroy: None,
        ..callbacks()
    };
    let c = config(&cb);
    let mut device = std::ptr::null_mut();
    let (mut width, mut height) = (0, 0);
    let mut dst = vec![7u8; 1920 * 1080 * 4];
    // SAFETY: a live config and out-pointer; then a live device, `dst` holding the lengths
    // given, the size pointers writable or NULL; the device is released.
    unsafe {
        assert_eq!(rvg_graphics_device_create(&c, &mut device), RVG_OK);
        assert_eq!(
            rvg_graphics_device_read_display(
                device,
                std::ptr::null_mut(),
                0,
                &mut width,
                &mut height
            ),
            RVG_ERR_INVALID_ARGUMENT
        );
        assert_eq!((width, height), (1920, 1080));
        assert_eq!(
            rvg_graphics_device_read_display(
                device,
                dst.as_mut_ptr(),
                dst.len() - 1,
                &mut width,
                &mut height
            ),
            RVG_ERR_INVALID_ARGUMENT
        );
        assert!(dst.iter().all(|b| *b == 7), "a refused read wrote");
        assert_eq!(
            rvg_graphics_device_read_display(
                device,
                dst.as_mut_ptr(),
                dst.len(),
                std::ptr::null_mut(),
                std::ptr::null_mut()
            ),
            RVG_OK
        );
        rvg_graphics_device_release(device);
    }
    assert!(dst.chunks(4).all(|p| p == [0, 0, 0, 0xff]), "not black");
}

/// What the queue's service thread spends copying an out-of-line command buffer's message of each
/// length, under the allocator the library ships with (this crate's global one): the copy out of
/// its block (`Region::read_held`, the one copy the host makes), and that copy copied once more,
/// as the hand-off to the buffer's lane did before it kept the device's copy
/// (`reims_vgpu::graphics`, `Host::command_buffer`). Prints each one's p50 over 200 rounds from
/// the same block, and the second copy's share.
#[test]
#[ignore = "a measurement, run by hand in a release build: prints a command buffer's copy cost"]
#[allow(clippy::disallowed_methods)] // a measurement: the wall clock's time
fn command_buffer_copy_cost() {
    use std::hint::black_box;
    use std::time::Instant;

    use reims_vgpu_graphics::region::Region;
    use reims_vgpu_graphics::wire::RGFX_CONTEXT_ALLOCATIONS_OFFSET;

    const BLOCK: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
    const SPAN: usize = 8 << 20;
    let region = Region::new(BLOCK as usize + SPAN).unwrap();
    region.grant(1, BLOCK, SPAN as u64);
    assert!(region.write(BLOCK, &vec![0x5a; SPAN]));
    let p50 = |f: &dyn Fn() -> usize| {
        let mut us: Vec<f64> = (0..200)
            .map(|_| {
                let t0 = Instant::now();
                black_box(f());
                t0.elapsed().as_secs_f64() * 1e6
            })
            .collect();
        us.sort_by(f64::total_cmp);
        us[us.len() / 2]
    };
    for len in [32usize << 10, 64 << 10, 256 << 10, 1 << 20, 5 << 20] {
        let one = p50(&|| region.read_held(1, BLOCK, len).unwrap().len());
        let two = p50(&|| {
            let copy = region.read_held(1, BLOCK, len).unwrap();
            black_box(copy.to_vec()).len()
        });
        println!(
            "command_buffer_copy_cost: {} KiB: one copy p50 {one:.1} us, two {two:.1} us \
             (the second {:.1} us, {:.0}%)",
            len >> 10,
            two - one,
            100.0 * (two - one) / two
        );
    }
}

/// What the pool's host callbacks were asked, for [`a_pool_s_pieces_reach_the_host_and_back`].
static POOL_ASKS: parking_lot::Mutex<Vec<(bool, u64, u64)>> = parking_lot::Mutex::new(Vec::new());

unsafe extern "C-unwind" fn pool_add(
    _ctx: *mut c_void,
    physical_address: u64,
    host_address: *mut c_void,
    length: u64,
) {
    assert!(!host_address.is_null());
    POOL_ASKS.lock().push((true, physical_address, length));
}

unsafe extern "C-unwind" fn pool_remove(_ctx: *mut c_void, physical_address: u64, length: u64) {
    POOL_ASKS.lock().push((false, physical_address, length));
}

/// A pool needs both of its callbacks and a range of whole 2 MiB; a device without one refuses
/// the pool's calls; a device with one grows on the host's ask, on its pool thread, takes the
/// host's word that it added the piece once, and its removal; a config of the fourth version,
/// whose `struct_size` ends before the pool's fields, has no pool whatever lies past its size
#[test]
fn a_pool_s_pieces_reach_the_host_and_back() {
    const BASE: u64 = 52 << 30;
    let plain = RvgGraphicsCallbacks {
        destroy: None,
        ..callbacks()
    };
    let with_pool = RvgGraphicsCallbacks {
        pool_add: Some(pool_add),
        pool_remove: Some(pool_remove),
        ..plain
    };
    let mut device = std::ptr::null_mut();
    // SAFETY: live configs and out-pointers; then live devices, released.
    unsafe {
        for (cb, address, length) in [
            (&plain, BASE, 64 << 30),
            (&with_pool, BASE + 4096, 64 << 30),
            (&with_pool, BASE, (64 << 30) + 4096),
            (&with_pool, BASE, 0),
        ] {
            let c = RvgGraphicsDeviceConfig {
                pool_physical_address: address,
                pool_physical_length: length,
                ..config(cb)
            };
            assert_eq!(
                rvg_graphics_device_create(&c, &mut device),
                RVG_ERR_INVALID_ARGUMENT,
                "{address:#x} {length:#x}"
            );
        }
        // The fourth version: no pool, whatever its fields hold.
        let c = RvgGraphicsDeviceConfig {
            struct_size: std::mem::offset_of!(RvgGraphicsDeviceConfig, pool_physical_address)
                as u32,
            pool_physical_address: BASE,
            pool_physical_length: 64 << 30,
            ..config(&plain)
        };
        assert_eq!(rvg_graphics_device_create(&c, &mut device), RVG_OK);
        assert_eq!(
            rvg_graphics_device_pool_grow(device, 1),
            RVG_ERR_INVALID_ARGUMENT
        );
        assert_eq!(
            rvg_graphics_device_pool_added(device, BASE),
            RVG_ERR_INVALID_ARGUMENT
        );
        rvg_graphics_device_release(device);

        let c = RvgGraphicsDeviceConfig {
            pool_physical_address: BASE,
            pool_physical_length: 64 << 30,
            ..config(&with_pool)
        };
        assert_eq!(rvg_graphics_device_create(&c, &mut device), RVG_OK);
        assert_eq!(
            rvg_graphics_device_pool_grow(device, 0),
            RVG_ERR_INVALID_ARGUMENT
        );
        assert_eq!(rvg_graphics_device_pool_grow(device, 1), RVG_OK);
        for _ in 0..10_000 {
            if !POOL_ASKS.lock().is_empty() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert_eq!(*POOL_ASKS.lock(), [(true, BASE, 256 << 20)]);
        assert_eq!(rvg_graphics_device_pool_added(device, BASE), RVG_OK);
        assert_eq!(
            rvg_graphics_device_pool_added(device, BASE),
            RVG_ERR_INVALID_ARGUMENT
        );
        assert_eq!(
            rvg_graphics_device_pool_removed(device, BASE + (256 << 20)),
            RVG_ERR_INVALID_ARGUMENT
        );
        // The host removes its pieces before the release, and the release asks it nothing.
        assert_eq!(rvg_graphics_device_pool_removed(device, BASE), RVG_OK);
        rvg_graphics_device_release(device);
    }
    assert_eq!(POOL_ASKS.lock().len(), 1);
}
