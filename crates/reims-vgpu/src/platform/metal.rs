//! `MetalPlatform` as an `ApiPlatform`: the product's platform, on macOS.

use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCreateSystemDefaultDevice, MTLOrigin, MTLRegion, MTLSize, MTLTexture};
use reims_vgpu_core::platform::Platform;
use reims_vgpu_metal::MetalPlatform;
use reims_vgpu_metal::msg;
use reims_vgpu_metal::platform::Obj;

use super::*;

impl ApiPlatform for MetalPlatform {
    fn process() -> Arc<MetalPlatform> {
        crate::singleton::platform()
    }

    /// `MTLCreateSystemDefaultDevice()`, or the device of `MTLCopyAllDevices()` whose
    /// `registryID` matches (`Platform::open_device`). Each send runs in the message layer,
    /// which dies at the send if Metal raises.
    fn metal_device(&self, registry_id: Option<u64>) -> Option<<Self as Platform>::Device> {
        match registry_id {
            None => msg::get("MTLCreateSystemDefaultDevice()", || {
                MTLCreateSystemDefaultDevice()
            })
            .map(Obj::from_retained),
            Some(id) => self.open_device(id),
        }
    }

    /// `getBytes:bytesPerRow:fromRegion:mipmapLevel:` over the whole texture.
    fn read_pixels(
        &self,
        texture: &<Self as Platform>::Texture,
        dst: &mut [u8],
        bytes_per_row: usize,
    ) {
        let t: &ProtocolObject<dyn MTLTexture> = texture.get();
        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize {
                width: msg::get("width", || t.width()),
                height: msg::get("height", || t.height()),
                depth: 1,
            },
        };
        let Some(bytes) = std::ptr::NonNull::new(dst.as_mut_ptr().cast()) else {
            return;
        };
        msg::get("getBytes:bytesPerRow:fromRegion:mipmapLevel:", || {
            // SAFETY: the caller checked that `dst` holds `height` rows `bytes_per_row`
            // apart, each `width * 4` bytes (a 4-byte format); Metal writes only there.
            unsafe {
                t.getBytes_bytesPerRow_fromRegion_mipmapLevel(bytes, bytes_per_row, region, 0)
            }
        });
    }
}
