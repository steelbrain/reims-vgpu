//! The rt platform's surfaces over Direct3D 11 textures (`GpuDevice::import_d3d11_surface`), on
//! Windows. Each plane of a picture is rt's texture over a shared texture that the host's video
//! draws the picture into (reims-vgpu-video's `vaapi::PlaneDraw`). The IOSurface device caches
//! keep a two-plane surface's planes in such textures, so a picture lands in the images that the
//! guest's textures are views of (reims-vgpu's `surface_cache::exported`).
//!
//! **The call** is rt's `Device::import_d3d11_surface`, one `rt::D3d11Plane` a plane. Where rt
//! refuses it (a device that imports no Direct3D 11 texture, a plane whose formats or usage its
//! import does not take), the surface is refused and counted, and the caller keeps the planes in
//! images of its own, where the host's video lands its pictures by the CPU's copy.
//!
//! **The planes.** Each plane is a 2D texture of the plane's format and size, at no resource
//! index, over the Direct3D texture its handle names. rt imports each one as a dedicated
//! `D3D11_TEXTURE` image of the texture's element format (`image_format`). A chroma plane's
//! element is one packed pair (`R16Uint`, `R32Uint`), and its texture is viewed in the plane's
//! two-channel format.
//!
//! **The acquire.** `ImportedSurfaceMemory::external_writes_done` is rt's
//! `Surface::external_writes_done`, the planes' acquire from `VK_QUEUE_FAMILY_EXTERNAL`. rt takes
//! no keyed mutex. The writer holds the textures' keyed mutexes across its write, waits for the
//! write, releases them, and only then calls it.
//!
//! **Refusals.** Where rt makes no surface, the call answers `None`, counted
//! ([`D3D11_SURFACE_REFUSED_CODE`]). The process's first refusal also writes a line naming rt's
//! error and the planes.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use metal2vulkan_rt as rt;

use crate::core::platform::resource::{D3d11PlaneDesc, ImportedSurface, ImportedSurfaceMemory};
use crate::core::platform::values::TextureDesc;

use super::descriptor::rt_texture_descriptor;
use super::device::RtDevice;
use super::{RtObj, RtPlatform, texture};

/// The log target of the surfaces' lines.
const TARGET: &str = "reims_vgpu_vulkan::rt_platform";

/// The code of a surface over Direct3D 11 textures that rt made, counted per make.
pub const D3D11_SURFACES_CODE: &str = "rt-d3d11-surfaces";

/// The code of a surface over Direct3D 11 textures that rt refused (the module doc's
/// "Refusals"), counted per refusal.
pub const D3D11_SURFACE_REFUSED_CODE: &str = "rt-refused-d3d11-surface";

/// `MTLTextureType2D`.
const TEXTURE_TYPE_2D: u64 = 2;

/// `MTLResourceStorageModePrivate`, in `MTLResourceOptions`: the planes' storage.
const STORAGE_PRIVATE: u64 = 2 << 4;

/// Whether the process has refused a surface yet.
static REFUSED_SAID: AtomicBool = AtomicBool::new(false);

/// What a surface over Direct3D 11 textures keeps: rt's surface, which holds the planes'
/// images.
struct RtImportedMemory {
    /// rt's surface.
    surface: rt::Surface,
}

impl ImportedSurfaceMemory for RtImportedMemory {
    fn external_writes_done(&self) -> Result<(), String> {
        (self.surface.external_writes_done()).map_err(|error| error.to_string())
    }
}

/// The texture of `plane`: 2D and Private, of the plane's format, size and usage.
fn plane_texture(plane: &D3d11PlaneDesc<'_>) -> TextureDesc {
    TextureDesc {
        texture_type: TEXTURE_TYPE_2D,
        pixel_format: plane.pixel_format,
        width: plane.width,
        height: plane.height,
        usage: plane.usage,
        resource_options: Some(STORAGE_PRIVATE),
        ..TextureDesc::default()
    }
}

/// rt's pixel format of the `MTLPixelFormat` `format`, as `what`.
fn rt_format(what: &str, format: u64) -> Result<rt::PixelFormat, String> {
    let word = u32::try_from(format).map_err(|_| format!("{what} {format}"))?;
    rt::PixelFormat::try_from(word).map_err(|_| format!("{what} {format}"))
}

/// rt's plane of `plane`, whose texture is `texture`; `Err` says why rt takes none.
fn plane_spec<'a>(
    plane: &D3d11PlaneDesc<'a>,
    texture: &TextureDesc,
) -> Result<rt::D3d11Plane<'a>, String> {
    Ok(rt::D3d11Plane {
        handle: plane.handle,
        image_format: rt_format("element format", plane.image_format)?,
        descriptor: rt_texture_descriptor(texture)?,
        index: None,
    })
}

/// Counts rt's refusal `why` of a surface of `planes` ([`D3D11_SURFACE_REFUSED_CODE`]). The
/// process's first refusal also writes its line.
fn refused(planes: &[D3d11PlaneDesc<'_>], why: &dyn std::fmt::Display) {
    crate::answer::count_kind(D3D11_SURFACE_REFUSED_CODE);
    if !REFUSED_SAID.swap(true, Ordering::Relaxed) {
        log::warn!(
            target: TARGET,
            kind:% = D3D11_SURFACE_REFUSED_CODE;
            "a surface over Direct3D 11 textures of {planes:?}: metal2vulkan-rt makes none \
             ({why}); its planes stay in images of their own, and the host's video lands its \
             pictures another way ({D3D11_SURFACE_REFUSED_CODE}), counted from here"
        );
    }
}

/// The surface over the Direct3D 11 textures `planes` name, on `device` (the module doc).
/// `None` where rt makes none, counted, or where a plane's texture is not one rt takes.
pub(super) fn imported_surface(
    device: &RtObj<RtDevice>,
    planes: &[D3d11PlaneDesc<'_>],
) -> Option<ImportedSurface<RtPlatform>> {
    let sel = "importD3D11Surface (a plane)";
    let mut textures = Vec::with_capacity(planes.len());
    let mut specs = Vec::with_capacity(planes.len());
    for plane in planes {
        let texture = plane_texture(plane);
        match plane_spec(plane, &texture) {
            Ok(spec) => specs.push(spec),
            Err(why) => {
                refused(planes, &why);
                return None;
            }
        }
        textures.push(texture);
    }
    let surface = match device.rt().import_d3d11_surface(&specs) {
        Ok(surface) => surface,
        Err(error) => {
            refused(planes, &error);
            return None;
        }
    };
    crate::answer::count_kind(D3D11_SURFACES_CODE);
    let made = (surface.planes().iter().zip(&textures))
        .map(|(plane, texture)| texture::surface_plane(device, sel, texture, plane.clone()))
        .collect::<Option<Vec<_>>>()?;
    Some(ImportedSurface {
        planes: made,
        memory: Arc::new(RtImportedMemory { surface }),
    })
}

#[cfg(test)]
mod tests {
    use std::os::windows::io::BorrowedHandle;

    use super::*;
    use crate::core::platform::device::GpuDevice as _;
    use crate::core::raise::OnFault;

    fn counted(code: &str) -> u64 {
        (crate::answer::degradation_counts().into_iter())
            .find(|(c, _)| c == code)
            .map_or(0, |(_, n)| n)
    }

    /// A surface rt refuses is counted, for the host's video to land its pictures another way
    /// (the module doc): here a luma plane of 2-byte elements viewed as `R8Unorm`, whose texel
    /// is 1 byte, which rt refuses before it reads the plane's handle.
    #[test]
    fn a_surface_rt_refuses_is_counted_for_another_landing_on_vulkan() {
        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        // SAFETY: -1 is the current process's pseudo-handle, which is never closed; rt refuses
        // the luma plane's formats, or a device that imports no Direct3D 11 texture, before it
        // reads a handle.
        let handle =
            unsafe { BorrowedHandle::borrow_raw(std::ptr::without_provenance_mut(usize::MAX)) };
        let plane = |image_format, pixel_format, width, height| D3d11PlaneDesc {
            handle,
            image_format,
            pixel_format,
            width,
            height,
            usage: 0x1 | 0x4 | 0x10,
        };
        let planes = [plane(20, 10, 64, 64), plane(23, 30, 32, 32)];
        let before = counted(D3D11_SURFACE_REFUSED_CODE);
        let made = device.import_d3d11_surface(OnFault::Return, &planes);
        assert!(matches!(made, Ok(None)));
        assert_eq!(counted(D3D11_SURFACE_REFUSED_CODE), before + 1);
    }
}
