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
        cursor_glyph: None,
        cursor_show: None,
        cursor_move: None,
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
        rvg_graphics_device_mmio_write_taking(device, 0x10, 1);
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

    use reims_graphics::region::Region;
    use reims_graphics::wire::RGFX_CONTEXT_ALLOCATIONS_OFFSET;

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
