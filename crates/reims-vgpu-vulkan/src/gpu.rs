//! What the rt platform takes of Metal's documented facts and of the M3 Max's answers: the stand-ins
//! ([`caps`]) and the identity answers' relations ([`identity`]), the descriptors' resource options
//! and the facts a texture over a surface's plane takes ([`device`]), the formats Metal makes no
//! view in ([`no_texture`]), the copy of the guest's bytes a write takes ([`resource`]), the
//! library's own shaders ([`shaders`]) and the trace's clock ([`timing`]); and Metal's raw pixel
//! formats and swizzles as metal2vulkan's types.

pub mod caps;
pub mod device;
pub mod identity;
pub mod no_texture;
pub mod resource;
pub mod shaders;
pub mod timing;

use crate::core::platform::types::Swizzle;
use metal2vulkan::metal::{PixelFormat, TextureSwizzle, TextureSwizzleChannels};

/// metal2vulkan's format of a raw `MTLPixelFormat`: `None` for a value it has no enumerator of.
pub(crate) fn runtime_pixel_format(raw: u64) -> Option<PixelFormat> {
    u32::try_from(raw)
        .ok()
        .and_then(|raw| PixelFormat::try_from(raw).ok())
}

/// metal2vulkan's swizzle of core's: `None` where a channel is a value it has no enumerator of.
pub(crate) fn runtime_swizzle(s: Swizzle) -> Option<TextureSwizzleChannels> {
    let channel = |raw: u8| TextureSwizzle::try_from(u32::from(raw)).ok();
    Some(TextureSwizzleChannels {
        red: channel(s.red)?,
        green: channel(s.green)?,
        blue: channel(s.blue)?,
        alpha: channel(s.alpha)?,
    })
}
