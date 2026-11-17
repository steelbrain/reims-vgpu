//! The cursor's colour chain (`CURSOR_COLOR`, `display::cursor_color`): the screen is handed the
//! cursor's image converted through the chain the guest set, the last image again at a new chain,
//! and the image as it is once the chain is off, after a reset, and for a guest that did not take
//! `CURSOR_COLOR`.

use super::*;

/// A host that offers `CURSOR_COLOR`, as every host does.
struct ColorHost;

impl Host for ColorHost {
    fn raise_interrupt(&self, _vector: u32) {}
    fn features(&self) -> u64 {
        RGFX_FEATURE_CURSOR_COLOR
    }
}

/// A started rig whose host offers `CURSOR_COLOR`, which its guest takes where `take`.
fn color_rig(take: bool) -> Rig {
    let rig = Rig::with_host(Box::new(ColorHost));
    take_color(&rig, take);
    rig.start();
    rig
}

/// The guest's driver takes `CURSOR_COLOR` where `take`, else nothing.
fn take_color(rig: &Rig, take: bool) {
    let bits = if take { RGFX_FEATURE_CURSOR_COLOR } else { 0 };
    rig.device
        .mmio_write(RGFX_REG_DRIVER_FEATURES_HI.into(), (bits >> 32) as u32);
}

/// Where the guest's cursor image lies.
const IMAGE: u64 = RGFX_ALLOCATIONS_OFFSET + 0x1000;

/// The image, premultiplied BGRA: a half-transparent red, then an opaque (30, 20, 10).
const BGRA: [u8; 8] = [0, 0, 128, 128, 10, 20, 30, 255];

/// [`BGRA`] as the screen is handed it with no chain: straight RGBA.
const STRAIGHT: [u8; 8] = [255, 0, 0, 128, 30, 20, 10, 255];

/// [`BGRA`] through [`SWAP`]'s chain: red and blue swapped, the alphas kept.
const SWAPPED: [u8; 8] = [0, 0, 255, 128, 10, 20, 30, 255];

/// A matrix from XYZ that swaps red and blue, so the chain's result is exact.
const SWAP: [f32; 9] = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0];

/// The identity.
const ONE: [f32; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

/// A `SET_CURSOR` of display 0's 2 x 1 image at [`IMAGE`].
fn set_cursor() -> Vec<u8> {
    RgfxSetCursor {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_SET_CURSOR,
            ..Default::default()
        },
        display: 0,
        width: 2,
        height: 1,
        hot_x: 0,
        hot_y: 0,
        reserved_24: 0,
        offset: IMAGE,
    }
    .as_bytes()
    .to_vec()
}

/// A `CURSOR_COLOR` of display `display`, on or off, its curves `Bypass` and its matrix from XYZ
/// `from_xyz`.
fn cursor_color(display: u32, on: bool, from_xyz: [f32; 9]) -> RgfxCursorColor {
    RgfxCursorColor {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_CURSOR_COLOR,
            ..Default::default()
        },
        display,
        on: u32::from(on),
        to_xyz: ONE.map(f32::to_bits),
        from_xyz: from_xyz.map(f32::to_bits),
        ..Default::default()
    }
}

/// The image the screen is handed, of `rgba`'s pixels.
fn image(rgba: [u8; 8]) -> Told {
    Told::Cursor(Some(CursorImage {
        width: 2,
        height: 1,
        hot_spot: (0, 0),
        rgba: rgba.to_vec(),
    }))
}

/// Sets the cursor's image and waits for its answer.
fn cursor(rig: &mut Rig) {
    let s = rig.send(set_cursor());
    assert_eq!(rig.complete(), (s, RGFX_STATUS_OK));
}

/// Sends `command`, unanswered, and waits for the control thread to have carried it out: it
/// carries out the answered `GET_DISPLAYS` after it.
fn unanswered(rig: &mut Rig, command: &RgfxCursorColor) {
    rig.send(command.as_bytes().to_vec());
    rig.send(get_displays());
    rig.event().expect("DISPLAYS");
}

/// A chain hands the screen the last image again, converted, and converts each image set after
/// it; the same chain again hands it nothing; a chain turned off hands it the image as it is
#[test]
fn the_cursor_shows_its_image_through_its_chain() {
    let mut rig = color_rig(true);
    rig.region().write(IMAGE, &BGRA);
    cursor(&mut rig);
    assert_eq!(rig.told(), [image(STRAIGHT)]);
    unanswered(&mut rig, &cursor_color(0, true, SWAP));
    assert_eq!(rig.told(), [image(SWAPPED)]);
    unanswered(&mut rig, &cursor_color(0, true, SWAP));
    assert_eq!(rig.told(), []);
    cursor(&mut rig);
    assert_eq!(rig.told(), [image(SWAPPED)]);
    unanswered(&mut rig, &cursor_color(0, false, SWAP));
    assert_eq!(rig.told(), [image(STRAIGHT)]);
}

/// A chain set before the first image converts that image; one of a curve the device cannot read
/// is dropped and the cursor keeps its chain; a removed image is not handed again at a new chain
#[test]
fn a_chain_before_the_image_converts_it_and_one_unread_is_dropped() {
    let mut rig = color_rig(true);
    rig.region().write(IMAGE, &BGRA);
    unanswered(&mut rig, &cursor_color(0, true, SWAP));
    assert_eq!(rig.told(), []);
    cursor(&mut rig);
    assert_eq!(rig.told(), [image(SWAPPED)]);
    let mut unread = cursor_color(0, true, ONE);
    unread.blend_curves[1].form = 9;
    unanswered(&mut rig, &unread);
    assert_eq!(rig.told(), []);
    cursor(&mut rig);
    assert_eq!(rig.told(), [image(SWAPPED)]);
    let mut removed = set_cursor();
    removed[20..24].copy_from_slice(&0u32.to_le_bytes());
    let s = rig.send(removed);
    assert_eq!(rig.complete(), (s, RGFX_STATUS_OK));
    unanswered(&mut rig, &cursor_color(0, true, ONE));
    assert_eq!(rig.told(), [Told::Cursor(None)]);
}

/// A guest that did not take `CURSOR_COLOR`, as the x86-64 kext, has its `CURSOR_COLOR`s dropped
/// and its cursor shown as it is; so does a chain of a display the guest does not see
#[test]
fn a_chain_from_a_guest_that_did_not_take_it_is_dropped() {
    let mut rig = color_rig(false);
    rig.region().write(IMAGE, &BGRA);
    cursor(&mut rig);
    unanswered(&mut rig, &cursor_color(0, true, SWAP));
    cursor(&mut rig);
    assert_eq!(rig.told(), [image(STRAIGHT), image(STRAIGHT)]);
    let mut rig = color_rig(true);
    rig.region().write(IMAGE, &BGRA);
    unanswered(&mut rig, &cursor_color(1, true, SWAP));
    cursor(&mut rig);
    assert_eq!(rig.told(), [image(STRAIGHT)]);
}

/// A reset turns the chain off: the image set after it shows as it is
#[test]
fn a_reset_turns_the_chain_off() {
    let mut rig = color_rig(true);
    rig.region().write(IMAGE, &BGRA);
    unanswered(&mut rig, &cursor_color(0, true, SWAP));
    rig.device
        .mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_RESET);
    (rig.produced, rig.consumed, rig.seqno) = (0, 0, 0);
    take_color(&rig, true);
    rig.start();
    rig.told();
    assert_eq!(
        rig.device.shared.displays[0].state.lock().cursor_color,
        None
    );
    cursor(&mut rig);
    assert_eq!(rig.told(), [image(STRAIGHT)]);
}
