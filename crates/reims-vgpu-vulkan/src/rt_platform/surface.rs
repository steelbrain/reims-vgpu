//! The rt platform's exported surfaces (`GpuDevice::new_exported_surface`): a picture's planes as
//! rt's textures over memory another API imports, rt's `Device::new_exported_surface`. The
//! IOSurface device caches keep a two-plane surface's planes in them, so that the host's video
//! writes its pictures into the images the guest's textures are views of (reims-vgpu's
//! `surface_cache::exported`).
//!
//! **The planes.** Each plane is a 2D texture of the plane's format and size at no resource
//! index: rt's plane texture, which the guest's textures over the plane are views of at their
//! own indices. Its usage is the caller's (`PixelFormatView` among it for the views of another
//! format of the texel's size, which rt's planes in place list where their tiling samples them).
//!
//! **The forms.** rt decides at the make: in place, each plane a linear image in the shared
//! memory, or the copy form, each plane rt's texture over a buffer of an exported block, which
//! rt's submissions copy into and back. Each make is counted by its form
//! ([`SURFACES_IN_PLACE_CODE`], [`SURFACES_COPIED_CODE`]); the process's first copy form writes a
//! line naming rt's reason and the spec.
//!
//! **Each export a new fd.** rt gives one fd of the memory as it makes the surface, which the
//! surface keeps; each export answers a duplicate of it (`dup`), the caller's own, as an import
//! takes it, of the whole allocation from its first byte. An opaque fd's allocation in place
//! holds every plane, so it is not a dedicated one; the copy form's block's is.
//!
//! **The acquire.** `SurfaceMemory::external_writes_done` is rt's `Surface::external_writes_done`:
//! the planes' acquire from the other API's queue family, recorded ahead of every command buffer
//! committed after it, with the contract of an exported block's (`super::export`, "The acquire").
//!
//! **Refusals.** Where rt makes no surface the call answers `None`, counted
//! ([`SURFACE_REFUSED_CODE`]), the process's first with a line naming rt's error and the spec,
//! and the caller keeps the planes in images of its own.

use std::os::fd::OwnedFd;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use metal2vulkan_rt as rt;

use crate::core::platform::resource::{
    ExportedMemory, ExportedSurface, SurfaceDesc, SurfaceHandle, SurfaceMemory, SurfacePlaneLayout,
};
use crate::core::platform::values::TextureDesc;

use super::descriptor::rt_texture_descriptor;
use super::device::RtDevice;
use super::{RtObj, RtPlatform, texture};

/// The log target of the surfaces' lines.
const TARGET: &str = "reims_vgpu_vulkan::rt_platform";

/// The code of a surface rt made in place, counted per make.
pub const SURFACES_IN_PLACE_CODE: &str = "rt-surfaces-in-place";

/// The code of a surface rt made in the copy form, counted per make.
pub const SURFACES_COPIED_CODE: &str = "rt-surfaces-copied";

/// The code of a surface rt made none of (the module's docs, "Refusals"), counted per refusal.
pub const SURFACE_REFUSED_CODE: &str = "rt-refused-surface";

/// `MTLTextureType2D`.
const TEXTURE_TYPE_2D: u64 = 2;

/// `MTLResourceStorageModePrivate`, in `MTLResourceOptions`: the planes' storage.
const STORAGE_PRIVATE: u64 = 2 << 4;

/// Whether the process refused a surface yet.
static REFUSED_SAID: AtomicBool = AtomicBool::new(false);

/// Whether the process made a surface in the copy form yet.
static COPIED_SAID: AtomicBool = AtomicBool::new(false);

/// What an exported surface keeps of its export (the module's docs).
struct RtSurfaceMemory {
    /// rt's surface, which holds the planes' memory.
    surface: rt::Surface,
    /// rt's fd of the memory, which each export duplicates.
    fd: OwnedFd,
    /// The kind of `fd`.
    handle: SurfaceHandle,
    /// The allocation's bytes, which an import names.
    size: u64,
    /// Whether rt made the planes in place.
    in_place: bool,
    /// Each plane's place in the memory.
    layout: Vec<SurfacePlaneLayout>,
}

/// rt's handle of `handle`'s kind.
fn rt_handle(handle: SurfaceHandle) -> rt::ExportHandle {
    match handle {
        SurfaceHandle::OpaqueFd => rt::ExportHandle::OpaqueFd,
        SurfaceHandle::DmaBuf => rt::ExportHandle::DmaBuf,
    }
}

/// The texture of plane `plane` of `desc`: 2D, Private, of the plane's format, size and usage.
fn plane_texture(desc: &SurfaceDesc<'_>, plane: usize) -> Option<TextureDesc> {
    let p = desc.planes.get(plane)?;
    Some(TextureDesc {
        texture_type: TEXTURE_TYPE_2D,
        pixel_format: p.pixel_format,
        width: p.width,
        height: p.height,
        usage: p.usage,
        resource_options: Some(STORAGE_PRIVATE),
        ..TextureDesc::default()
    })
}

/// Counts rt's refusal `why` of the surface `desc` ([`SURFACE_REFUSED_CODE`]); the process's
/// first writes its line.
fn refused(desc: &SurfaceDesc<'_>, why: &dyn std::fmt::Display) {
    crate::answer::count_kind(SURFACE_REFUSED_CODE);
    if !REFUSED_SAID.swap(true, Ordering::Relaxed) {
        log::warn!(
            target: TARGET,
            kind:% = SURFACE_REFUSED_CODE;
            "an exported surface of {desc:?}: metal2vulkan-rt makes none ({why}); its planes stay \
             in images of their own, and the host's video lands its pictures another way \
             ({SURFACE_REFUSED_CODE}), counted from here"
        );
    }
}

/// The exported surface `desc` names on `device` (the module's docs); `None` where rt makes none,
/// counted, or a plane's texture is not one rt takes.
pub(super) fn exported_surface(
    device: &RtObj<RtDevice>,
    desc: &SurfaceDesc<'_>,
) -> Option<ExportedSurface<RtPlatform>> {
    let sel = "newExportedSurface (a plane)";
    let mut textures = Vec::with_capacity(desc.planes.len());
    let mut specs = Vec::with_capacity(desc.planes.len());
    for (at, plane) in desc.planes.iter().enumerate() {
        let Some(texture) = plane_texture(desc, at) else {
            refused(desc, &format!("no plane {at}"));
            return None;
        };
        let descriptor = match rt_texture_descriptor(&texture) {
            Ok(d) => d,
            Err(why) => {
                refused(desc, &why);
                return None;
            }
        };
        specs.push(rt::PlaneSpec {
            descriptor,
            index: None,
            rows_below: plane.rows_below,
        });
        textures.push(texture);
    }
    let spec = rt::SurfaceSpec {
        handle: rt_handle(desc.handle),
        pitch_multiple: desc.pitch_multiple,
        start_multiple: desc.start_multiple,
        planes: &specs,
    };
    let (surface, export) = match device.rt().new_exported_surface(&spec) {
        Ok(made) => made,
        Err(error) => {
            refused(desc, &error);
            return None;
        }
    };
    let in_place = match export.form {
        rt::SurfaceForm::InPlace => {
            crate::answer::count_kind(SURFACES_IN_PLACE_CODE);
            true
        }
        rt::SurfaceForm::Copy(reason) => {
            crate::answer::count_kind(SURFACES_COPIED_CODE);
            if !COPIED_SAID.swap(true, Ordering::Relaxed) {
                log::warn!(
                    target: TARGET,
                    kind:% = SURFACES_COPIED_CODE;
                    "an exported surface of {desc:?}: metal2vulkan-rt made it in the copy form \
                     ({reason:?}), its planes copied into and back ({SURFACES_COPIED_CODE}), \
                     counted from here"
                );
            }
            false
        }
    };
    let planes = (surface.planes().iter().zip(&textures))
        .map(|(plane, texture)| texture::surface_plane(device, sel, texture, plane.clone()))
        .collect::<Option<Vec<_>>>()?;
    let layout = (export.planes.iter())
        .map(|p| SurfacePlaneLayout {
            offset: p.offset,
            row_pitch: p.row_pitch,
        })
        .collect();
    let memory = RtSurfaceMemory {
        surface,
        fd: export.export.handle,
        handle: desc.handle,
        size: export.export.size,
        in_place,
        layout,
    };
    Some(ExportedSurface {
        planes,
        memory: Arc::new(memory),
    })
}

impl SurfaceMemory for RtSurfaceMemory {
    fn export(&self) -> Result<ExportedMemory, String> {
        let fd =
            (self.fd.try_clone()).map_err(|error| format!("a new fd of the surface: {error}"))?;
        Ok(ExportedMemory {
            fd,
            allocation_size: self.size,
            offset: 0,
            length: self.size,
            dedicated: !(self.in_place && self.handle == SurfaceHandle::OpaqueFd),
        })
    }

    fn external_writes_done(&self) -> Result<(), String> {
        (self.surface.external_writes_done()).map_err(|error| error.to_string())
    }

    fn handle(&self) -> SurfaceHandle {
        self.handle
    }

    fn in_place(&self) -> bool {
        self.in_place
    }

    fn layout(&self) -> &[SurfacePlaneLayout] {
        &self.layout
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::platform::device::GpuDevice as _;
    use crate::core::platform::resource::{SurfacePlaneDesc, Texture as _};
    use crate::core::raise::OnFault;
    use crate::rt_platform::RtPlatform;

    fn counted(code: &str) -> u64 {
        (crate::answer::degradation_counts().into_iter())
            .find(|(c, _)| c == code)
            .map_or(0, |(_, n)| n)
    }

    /// An NV12 picture's two planes, R8Unorm and RG8Unorm of half its size each way, sampled,
    /// rendered and viewed in another format, as the IOSurface device caches ask them.
    fn nv12(width: u64, height: u64) -> [SurfacePlaneDesc; 2] {
        let usage = 0x1 | 0x4 | 0x10;
        [
            SurfacePlaneDesc {
                pixel_format: 10,
                width,
                height,
                usage,
                rows_below: 0,
            },
            SurfacePlaneDesc {
                pixel_format: 30,
                width: width.div_ceil(2),
                height: height.div_ceil(2),
                usage,
                rows_below: 0,
            },
        ]
    }

    /// Exported surfaces of a 1080p NV12 picture's planes on the GPU the test holds, of each
    /// handle its device exports: two plane textures of the planes' formats and sizes, the layout
    /// plane 0 at 0 and plane 1 past its rows at one pitch on the multiples; each export a new fd
    /// of the whole allocation, dedicated but for an opaque fd in place; the acquire taken; each
    /// make counted by its form. An R8Uint view of plane 0 and an RG8Uint view of plane 1 are
    /// made where the surface is in place. A handle the device does not export is refused,
    /// counted, and answers `None`.
    #[test]
    fn exported_surfaces_on_vulkan() {
        use std::os::fd::AsRawFd as _;
        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let planes = nv12(1920, 1080);
        for handle in [SurfaceHandle::OpaqueFd, SurfaceHandle::DmaBuf] {
            let desc = SurfaceDesc {
                handle,
                pitch_multiple: 256,
                start_multiple: 256,
                planes: &planes,
            };
            let counts = [
                SURFACES_IN_PLACE_CODE,
                SURFACES_COPIED_CODE,
                SURFACE_REFUSED_CODE,
            ]
            .map(counted);
            let made = device.new_exported_surface(OnFault::Return, &desc);
            let Some(surface) = made.expect("never a raise") else {
                assert_eq!(counted(SURFACE_REFUSED_CODE), counts[2] + 1);
                println!("{handle:?}: refused, counted");
                continue;
            };
            let memory = &surface.memory;
            let form = if memory.in_place() { 0 } else { 1 };
            assert_eq!(
                counted([SURFACES_IN_PLACE_CODE, SURFACES_COPIED_CODE][form]),
                counts[form] + 1
            );
            assert_eq!(memory.handle(), handle);
            let [luma, chroma] = surface.planes.as_slice() else {
                panic!("two planes");
            };
            assert_eq!(
                (luma.pixel_format(), luma.width(), luma.height()),
                (10, 1920, 1080)
            );
            assert_eq!(
                (chroma.pixel_format(), chroma.width(), chroma.height()),
                (30, 960, 540)
            );
            let layout = memory.layout();
            assert_eq!(layout.len(), 2);
            let pitch = layout[0].row_pitch;
            assert_eq!(layout[0].offset, 0);
            assert_eq!(layout[1].row_pitch, pitch);
            assert!(pitch >= 1920 && pitch % 256 == 0, "pitch {pitch}");
            assert!(layout[1].offset >= 1080 * pitch && layout[1].offset % 256 == 0);
            let first = memory.export().expect("an export");
            let second = memory.export().expect("another");
            assert_ne!(first.fd.as_raw_fd(), second.fd.as_raw_fd());
            assert_eq!((first.offset, first.length), (0, first.allocation_size));
            assert!(first.allocation_size >= layout[1].offset + 540 * pitch);
            let opaque_in_place = memory.in_place() && handle == SurfaceHandle::OpaqueFd;
            assert_eq!(first.dedicated, !opaque_in_place);
            assert_eq!(memory.external_writes_done(), Ok(()));
            if memory.in_place() {
                let luma_view = luma.new_texture_view_at_index(13, 0);
                let chroma_view = chroma.new_texture_view_at_index(33, 0);
                assert!(
                    luma_view.expect("never a raise").is_some(),
                    "an R8Uint view"
                );
                assert!(
                    chroma_view.expect("never a raise").is_some(),
                    "an RG8Uint view"
                );
            }
            println!("{handle:?}: in place {}, pitch {pitch}", memory.in_place());
        }
    }
}
