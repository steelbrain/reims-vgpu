//! A decode surface's memory as dma-bufs, for another device to read its planes
//! ([`super::Frame::export`]): `vaExportSurfaceHandle`, read only, each plane a layer of its
//! own, so each plane is one image of its own format (R8 and RG8 for NV12, R16 and RG16 for P010)
//! in its object, at its offset and pitch, laid out as the object's DRM format modifier says.
//!
//! A pool exports each of its surfaces once, at its first frame asked so, and keeps the export
//! with the surface until the pool goes: the surfaces are lent again frame after frame, and the
//! other device then imports each once.
//!
//! What the drivers export (**measured** with a libva and Vulkan program written from scratch,
//! surfaces made as the pools make theirs, 2026-11-08):
//! - radeonsi (Mesa 26.2.6) on a Radeon 780M: one object, modifier `0x200000010401b04` (a
//!   64 KiB swizzle, no compression); 3840x2160 NV12 with luma rows of 3840 bytes and chroma rows
//!   of 4096 at 8,847,360, P010 with rows of 7680; 1920x1080 NV12 rows of 2048, P010 4096;
//! - iHD 26.3.5 on an Iris Xe: one object, modifier `0x100000000000002` (Y tiling), rows of the
//!   frame's bytes.
//!
//! Vulkan's driver of the same GPU (RADV, ANV) imported each plane as an image of that modifier
//! and copied it byte for byte.

use std::os::fd::{AsFd, BorrowedFd, FromRawFd, OwnedFd};
use std::sync::{Arc, Weak};

use super::display::{Display, Error, check};
use super::ffi;

/// One plane of a [`SurfaceExport`]: where it lies in its dma-buf, and how.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExportedPlane {
    /// The dma-buf it lies in: an index of the export's objects.
    pub object: usize,
    /// That dma-buf's size in bytes.
    pub object_size: u64,
    /// The dma-buf's DRM format modifier: how the plane's texels are laid out in it.
    pub modifier: u64,
    /// The plane's first byte in the dma-buf.
    pub offset: u64,
    /// Bytes from one of its rows to the next, as the modifier counts them.
    pub pitch: u64,
}

/// A decode surface's memory as dma-bufs (module docs), kept while its pool lives.
#[derive(Debug)]
pub struct SurfaceExport {
    objects: Vec<OwnedFd>,
    planes: [ExportedPlane; 2],
    size: (u32, u32),
    /// Its life, which another device's imports of it are let go by ([`SurfaceExport::alive`]).
    life: Arc<()>,
}

impl SurfaceExport {
    /// The luma plane (0) or the interleaved chroma plane (1).
    pub fn plane(&self, plane: usize) -> ExportedPlane {
        self.planes[plane.min(1)]
    }

    /// The dma-buf `plane` lies in, owned by the export.
    pub fn fd(&self, plane: usize) -> BorrowedFd<'_> {
        self.objects[self.plane(plane).object].as_fd()
    }

    /// The surface's width and height in luma samples: the decoded (coded) picture's.
    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    /// A reference that lives as long as the export: an import of its dma-bufs is let go once it
    /// no longer upgrades.
    pub fn alive(&self) -> Weak<()> {
        Arc::downgrade(&self.life)
    }
}

/// Exports `surface`, of `size`, on `display` (module docs); `Err` where the driver exports none,
/// or not as one plane a layer.
pub(crate) fn export(
    display: &Display,
    surface: ffi::VASurfaceID,
    size: (u32, u32),
) -> Result<SurfaceExport, Error> {
    let mut descriptor = ffi::VADRMPRIMESurfaceDescriptor::default();
    check(
        "vaExportSurfaceHandle",
        // SAFETY: a surface of this display, and a descriptor to fill.
        display.call(|raw| unsafe {
            ffi::vaExportSurfaceHandle(
                raw,
                surface,
                ffi::VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2,
                ffi::VA_EXPORT_SURFACE_READ_ONLY | ffi::VA_EXPORT_SURFACE_SEPARATE_LAYERS,
                (&raw mut descriptor).cast(),
            )
        })?,
    )?;
    let count = usize::try_from(descriptor.num_objects).unwrap_or(0).min(4);
    // SAFETY: the export's descriptors, owned from here, each closed once as its `OwnedFd` drops.
    let objects: Vec<OwnedFd> = (descriptor.objects[..count].iter())
        .map(|object| unsafe { OwnedFd::from_raw_fd(object.fd) })
        .collect();
    planes_of(&descriptor).map(|planes| SurfaceExport {
        objects,
        planes,
        size,
        life: Arc::new(()),
    })
}

/// The two planes of an export of separate layers: each layer one plane in an object the export
/// holds; `Err` with the export's shape otherwise.
fn planes_of(descriptor: &ffi::VADRMPRIMESurfaceDescriptor) -> Result<[ExportedPlane; 2], Error> {
    let shape = || {
        let layers =
            &descriptor.layers[..usize::try_from(descriptor.num_layers).unwrap_or(0).min(4)];
        Error::Unsupported(format!(
            "the decode surface exported as {} objects and {} layers of {:?} planes: not a layer \
             of one plane for each of its two planes",
            descriptor.num_objects,
            descriptor.num_layers,
            layers.iter().map(|l| l.num_planes).collect::<Vec<_>>(),
        ))
    };
    if descriptor.num_layers != 2 {
        return Err(shape());
    }
    let plane = |layer: &ffi::VADRMPRIMESurfaceLayer| -> Result<ExportedPlane, Error> {
        if layer.num_planes != 1 || layer.object_index[0] >= descriptor.num_objects.min(4) {
            return Err(shape());
        }
        let object = usize::try_from(layer.object_index[0]).map_err(|_| shape())?;
        let found = descriptor.objects[object];
        Ok(ExportedPlane {
            object,
            object_size: u64::from(found.size),
            modifier: found.drm_format_modifier,
            offset: u64::from(layer.offset[0]),
            pitch: u64::from(layer.pitch[0]),
        })
    };
    Ok([plane(&descriptor.layers[0])?, plane(&descriptor.layers[1])?])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layer(object: u32, offset: u32, pitch: u32) -> ffi::VADRMPRIMESurfaceLayer {
        ffi::VADRMPRIMESurfaceLayer {
            num_planes: 1,
            object_index: [object, 0, 0, 0],
            offset: [offset, 0, 0, 0],
            pitch: [pitch, 0, 0, 0],
            ..Default::default()
        }
    }

    /// An export of separate layers gives each plane its object, the object's size and modifier,
    /// and the layer's offset and pitch: radeonsi's 4K NV12 export as the 780M answered it
    #[test]
    fn each_layer_is_a_plane_in_its_object() {
        let mut d = ffi::VADRMPRIMESurfaceDescriptor {
            num_objects: 1,
            num_layers: 2,
            ..Default::default()
        };
        d.objects[0].size = 13_565_952;
        d.objects[0].drm_format_modifier = 0x0200_0000_1040_1b04;
        d.layers[0] = layer(0, 0, 3840);
        d.layers[1] = layer(0, 8_847_360, 4096);
        let planes = planes_of(&d).expect("two planes");
        assert_eq!(
            planes[1],
            ExportedPlane {
                object: 0,
                object_size: 13_565_952,
                modifier: 0x0200_0000_1040_1b04,
                offset: 8_847_360,
                pitch: 4096,
            }
        );
        assert_eq!(planes[0].pitch, 3840);
    }

    /// A composed export (one layer of two planes), a layer of two planes, or a layer naming an
    /// object the export does not hold, is refused
    #[test]
    fn an_export_not_of_a_layer_a_plane_is_refused() {
        let mut d = ffi::VADRMPRIMESurfaceDescriptor {
            num_objects: 1,
            num_layers: 1,
            ..Default::default()
        };
        d.layers[0] = layer(0, 0, 3840);
        d.layers[0].num_planes = 2;
        assert!(planes_of(&d).is_err(), "one layer");
        d.num_layers = 2;
        d.layers[1] = layer(0, 8_847_360, 4096);
        assert!(planes_of(&d).is_err(), "a layer of two planes");
        d.layers[0].num_planes = 1;
        d.layers[1].object_index[0] = 1;
        assert!(planes_of(&d).is_err(), "an object past the export's");
        d.layers[1].object_index[0] = 0;
        assert!(planes_of(&d).is_ok());
    }
}
