//! The blit copies between a buffer and a texture that take an `MTLBlitOption`
//! (`copyFromBuffer:...toTexture:...options:` and `copyFromTexture:...toBuffer:...options:`), on
//! metal2vulkan-rt, whose copies take none: its copy with no option is the options form's with
//! none (rt's README, "Blit encoders").
//!
//! Options 0, `MTLBlitOptionNone`, is rt's copy. Any other value, the depth or the stencil of a
//! combined depth/stencil texture or a PVRTC texture's rows among them, is
//! [`BlitError::Options`], with nothing recorded, for the caller to fail the command buffer on
//! as it does for rt's refusals.

use metal2vulkan_rt::{BlitCommandEncoder, Buffer, BufferLayout, Error, Texture, TexturePlace};

/// A copy that was not recorded.
#[derive(Debug)]
pub enum BlitError {
    /// `MTLBlitOption` bits other than none, as the guest sent them.
    Options(u64),
    /// rt's answer.
    Runtime(Error),
}

/// `copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:`
/// `toTexture:destinationSlice:destinationLevel:destinationOrigin:options:` on `encoder`: rt's
/// `copy_from_buffer` where `options` is 0.
///
/// Err: [`BlitError::Options`] for any other; rt's answer ([`BlitError::Runtime`]).
pub fn copy_from_buffer_with_options(
    encoder: &mut BlitCommandEncoder,
    source: &Buffer,
    layout: BufferLayout,
    size: [u32; 3],
    destination: &Texture,
    to: TexturePlace,
    options: u64,
) -> Result<(), BlitError> {
    none(options)?;
    (encoder.copy_from_buffer(source, layout, size, destination, to)).map_err(BlitError::Runtime)
}

/// `copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:`
/// `destinationOffset:destinationBytesPerRow:destinationBytesPerImage:options:` on `encoder`:
/// rt's `copy_from_texture_to_buffer` where `options` is 0.
///
/// Err: [`BlitError::Options`] for any other; rt's answer ([`BlitError::Runtime`]).
pub fn copy_from_texture_to_buffer_with_options(
    encoder: &mut BlitCommandEncoder,
    source: &Texture,
    from: TexturePlace,
    size: [u32; 3],
    destination: &Buffer,
    layout: BufferLayout,
    options: u64,
) -> Result<(), BlitError> {
    none(options)?;
    (encoder.copy_from_texture_to_buffer(source, from, size, destination, layout))
        .map_err(BlitError::Runtime)
}

/// Whether `options` is `MTLBlitOptionNone`, the only options rt copies with.
fn none(options: u64) -> Result<(), BlitError> {
    match options {
        0 => Ok(()),
        other => Err(BlitError::Options(other)),
    }
}

#[cfg(test)]
mod tests;
