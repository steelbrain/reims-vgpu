use metal2vulkan_rt::{
    CommandBuffer, CommandBufferStatus, Device, MemoryClass, PixelFormat, TextureDescriptor,
    TextureType, TextureUsage,
};

use super::*;
use crate::seam::test_device;

/// The test texture's side, in texels.
const SIDE: u32 = 4;

/// Its bytes: RGBA8, 4 a texel.
const BYTES: usize = (SIDE * SIDE * 4) as usize;

/// The options other than none that Metal names: depth from a depth/stencil texture, stencil from
/// one, and a PVRTC texture's rows; and a value of no option's.
const OPTIONS: [u64; 4] = [1, 2, 4, 0x100];

fn device() -> Device {
    let (device, name) = test_device().unwrap();
    println!("rt's device: {name}");
    device
}

/// A texture the copies reach, with no shader usage, so rt takes no index for it.
fn texture(device: &Device) -> Texture {
    let mut descriptor =
        TextureDescriptor::new(TextureType::Type2D, PixelFormat::RGBA8Unorm, SIDE, SIDE);
    descriptor.usage = TextureUsage::RENDER_TARGET;
    device.new_texture(&descriptor, None).unwrap()
}

/// The whole texture's bytes from a buffer's first, row after row.
fn layout() -> BufferLayout {
    BufferLayout {
        offset: 0,
        bytes_per_row: u64::from(SIDE) * 4,
        bytes_per_image: BYTES as u64,
    }
}

const SIZE: [u32; 3] = [SIDE, SIDE, 1];

/// A Shared buffer's bytes, as the host reads them.
fn bytes(buffer: &Buffer) -> Vec<u8> {
    let contents = buffer.contents().unwrap();
    // SAFETY: the buffer is alive, Shared and `BYTES` long, and no work in flight writes it.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), BYTES) }.to_vec()
}

fn run(command_buffer: &CommandBuffer) {
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed().unwrap();
    assert_eq!(
        command_buffer.status(),
        CommandBufferStatus::Completed,
        "{:?}",
        command_buffer.error()
    );
}

/// Options 0 is rt's copy, both ways: a texture filled from one buffer reads back into another
/// byte for byte.
#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "MoltenVK: rt's floor refuses it (no nullDescriptor), so no rt device on the Mac before KosmicKrisp at the cutover; ledger: rt on the Mac"
)]
fn on_vulkan_a_copy_with_no_option_is_rts_copy() {
    let device = device();
    let block = device.take_block(MemoryClass::Shared, 1).unwrap();
    let source = device.new_buffer(block.id, 0, BYTES as u64, None).unwrap();
    let destination = (device.new_buffer(block.id, 4096, BYTES as u64, Some(0))).unwrap();
    let want: Vec<u8> = (0..BYTES).map(|i| (i * 7 + 3) as u8).collect();
    let contents = source.contents().unwrap();
    // SAFETY: the buffer is alive, Shared and `BYTES` long, and no work uses it yet.
    unsafe {
        contents
            .as_ptr()
            .copy_from_nonoverlapping(want.as_ptr(), BYTES)
    };
    let texture = texture(&device);
    let queue = device.new_command_queue(4).unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.blit_command_encoder().unwrap();
    let place = TexturePlace::default();
    copy_from_buffer_with_options(&mut encoder, &source, layout(), SIZE, &texture, place, 0)
        .unwrap();
    copy_from_texture_to_buffer_with_options(
        &mut encoder,
        &texture,
        place,
        SIZE,
        &destination,
        layout(),
        0,
    )
    .unwrap();
    encoder.end_encoding();
    run(&command_buffer);
    assert_eq!(bytes(&destination), want);
}

/// Any options but none are refused, both ways, with nothing recorded: the command buffer
/// completes, and the bytes the refused copies name are untouched.
#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "MoltenVK: rt's floor refuses it (no nullDescriptor), so no rt device on the Mac before KosmicKrisp at the cutover; ledger: rt on the Mac"
)]
fn on_vulkan_a_copy_with_an_option_is_refused_with_nothing_recorded() {
    let device = device();
    let block = device.take_block(MemoryClass::Shared, 1).unwrap();
    let source = device
        .new_buffer(block.id, 0, BYTES as u64, Some(0))
        .unwrap();
    let destination = (device.new_buffer(block.id, 4096, BYTES as u64, None)).unwrap();
    let contents = destination.contents().unwrap();
    // SAFETY: the buffer is alive, Shared and `BYTES` long, and no work uses it yet.
    unsafe { contents.as_ptr().write_bytes(0xa5, BYTES) };
    let texture = texture(&device);
    let queue = device.new_command_queue(4).unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.blit_command_encoder().unwrap();
    let place = TexturePlace::default();
    for options in OPTIONS {
        let to = copy_from_buffer_with_options(
            &mut encoder,
            &source,
            layout(),
            SIZE,
            &texture,
            place,
            options,
        );
        assert!(
            matches!(to, Err(BlitError::Options(o)) if o == options),
            "{to:?}"
        );
        let from = copy_from_texture_to_buffer_with_options(
            &mut encoder,
            &texture,
            place,
            SIZE,
            &destination,
            layout(),
            options,
        );
        assert!(
            matches!(from, Err(BlitError::Options(o)) if o == options),
            "{from:?}"
        );
    }
    encoder.end_encoding();
    run(&command_buffer);
    assert_eq!(bytes(&destination), vec![0xa5; BYTES]);
}
