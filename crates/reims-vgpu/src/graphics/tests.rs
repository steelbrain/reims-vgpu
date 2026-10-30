use std::sync::atomic::{AtomicU32, Ordering};

use reims_graphics::wire::*;
use reims_vgpu_core::testkit::FakePlatform;

use super::*;

/// A graphics device on the testkit's platform, with no window, raising through `raised`.
fn headless(raised: &Arc<AtomicU32>) -> GraphicsDevice<FakePlatform> {
    let raised = Arc::clone(raised);
    let config = GraphicsConfig {
        callbacks: GraphicsCallbacks {
            raise_interrupt: Some(Arc::new(move |vector| {
                assert_eq!(vector, 0);
                raised.fetch_add(1, Ordering::SeqCst);
            })),
            scanout_written: None,
        },
        window: None,
    };
    GraphicsDevice::with_platform(Arc::new(FakePlatform::new()), config).unwrap()
}

/// A headless device's region is the device's, its registers read as the wire says, and the
/// guest's armed completion reaches the host's interrupt callback; its display offers 1920 ×
/// 1080 at 60 Hz
#[test]
fn a_headless_device_answers_its_guest() {
    let raised = Arc::new(AtomicU32::new(0));
    let device = headless(&raised);
    assert_eq!(device.region_size(), reims_graphics::REGION_SIZE);
    assert!(!device.region_base().is_null());
    assert_eq!(device.mmio_read(RGFX_REG_MAGIC.into()), RGFX_MAGIC);
    assert_eq!(
        device.mmio_read(RGFX_REG_REGION_SIZE_LO.into()),
        (reims_graphics::REGION_SIZE & 0xffff_ffff) as u32
    );
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    // SAFETY: the region is the device's, mapped while it lives, and the rings' fields are at
    // their offsets; nothing else writes the guest's side.
    unsafe {
        let base = device.region_base();
        let ring = base.add(RGFX_CONTROL_SUBMIT_OFFSET as usize + 256);
        let header = RgfxMsgHeader {
            r#type: RGFX_CMD_GET_DISPLAYS,
            size: 16,
            seqno: 1,
        };
        std::ptr::copy_nonoverlapping(
            std::ptr::from_ref(&header).cast::<u8>(),
            ring,
            size_of::<RgfxMsgHeader>(),
        );
        let armed = base.add(RGFX_CONTROL_COMPLETE_OFFSET as usize + 0xc0);
        std::sync::atomic::AtomicU32::from_ptr(armed.cast()).store(1, Ordering::SeqCst);
        let producer = base.add(RGFX_CONTROL_SUBMIT_OFFSET as usize);
        std::sync::atomic::AtomicU64::from_ptr(producer.cast()).store(16, Ordering::SeqCst);
    }
    device.mmio_write(RGFX_REG_DOORBELL.into(), RGFX_CONTROL_RING_ID);
    for _ in 0..2000 {
        if raised.load(Ordering::SeqCst) > 0 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert_eq!(raised.load(Ordering::SeqCst), 1);
    // The answer: DISPLAYS, its one mode 1920 × 1080 at 60 Hz.
    // SAFETY: as above; the device published the answer before it raised.
    let mode = unsafe {
        let answer = device
            .region_base()
            .add(RGFX_CONTROL_COMPLETE_OFFSET as usize + 256 + 36);
        std::slice::from_raw_parts(answer, 12).to_vec()
    };
    let words: Vec<u32> = mode
        .chunks(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(words, [1920, 1080, 60_000]);
}

/// The host screen's rate in thousandths of a hertz, rounded; 60 Hz for a rate that is none
#[test]
fn a_screen_s_rate_is_kept_in_millihertz() {
    assert_eq!(hz_to_mhz(120.0), 120_000);
    assert_eq!(hz_to_mhz(59.94), 59_940);
    assert_eq!(hz_to_mhz(0.0), 60_000);
    assert_eq!(hz_to_mhz(f64::NAN), 60_000);
}

/// The window shows the guest's cursor once it has an image and while the guest shows it,
/// telling the window only of a change
#[test]
fn the_cursor_shows_with_an_image_while_the_guest_shows_it() {
    let window = Window::new(WindowConfig::default(), 0, "test".to_owned());
    let mut cursor = CursorState {
        image: false,
        visible: true,
        shown: None,
    };
    cursor.update(&window);
    assert_eq!(cursor.shown, Some(false));
    cursor.image = true;
    cursor.update(&window);
    assert_eq!(cursor.shown, Some(true));
    cursor.visible = false;
    cursor.update(&window);
    assert_eq!(cursor.shown, Some(false));
    window.close();
}
