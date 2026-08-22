//! IOSurface texture / texture geometry registration onto the IOSurface mapping table.
//!
//! Archive: `apple_pv_gpu_register_iosurface_texture` latches width/height/format
//! on the mapping when a texture descriptor is resolved. Device-desc geom
//! (mapper path) and texture-path geom share [`Device::set_mapping_geom`].

use crate::runtime::decode::resource::{decode_descriptor, DecodeStatus, Descriptor, ObjectKind};
use crate::runtime::Device;
use reims_vgpu_protocol::MapperSurfaceRef;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MapperSurfaceResolveError {
    Unresolved,
    Geometry,
}

/// Register geometry from a decoded IOSurface texture / IOSurface texture descriptor.
pub fn register_iosurface_texture_geom(
    state: &mut Device,
    mapper_surface: MapperSurfaceRef,
    width: u32,
    height: u32,
    format: u16,
) -> Result<u32, MapperSurfaceResolveError> {
    let mapping_id = state
        .resolve_mapper_surface(mapper_surface)
        .ok_or(MapperSurfaceResolveError::Unresolved)?
        .get();
    if let Some(m) = state.surfaces.mappings.get(&mapping_id) {
        if m.has_geometry()
            && m.width_or_zero() == width
            && m.height_or_zero() == height
            && m.format_or_zero() == format
        {
            return Ok(mapping_id);
        }
    }
    state
        .set_mapping_geom(mapping_id, width, height, format)
        .then_some(mapping_id)
        .ok_or(MapperSurfaceResolveError::Geometry)
}

/// Decode descriptor bytes and, if IOSurface texture, latch mapping geometry.
pub fn register_from_descriptor_bytes(state: &mut Device, object_type: u8, desc: &[u8]) -> bool {
    let kind = ObjectKind::from_wire_tag(object_type)
        .ok_or(DecodeStatus::ErrUnknownType("res_object_type_unknown"));
    let decoded = kind.and_then(|kind| decode_descriptor(kind, desc));
    match decoded {
        Ok(Descriptor::MapperIOSurfaceTextureView(view)) => register_iosurface_texture_geom(
            state,
            view.mapper_surface,
            view.declaration.width,
            view.declaration.height,
            view.declaration.pixel_format,
        )
        .is_ok(),
        // A buffer, a function, a pipeline: not a refusal. This is called for
        // every object type and only IOSurface texture carries mapping geometry, so the
        // decoder answering "that is a different object" is the normal case and
        // must stay out of the log.
        Ok(_) => false,
        Err(e) => {
            if ObjectKind::from_wire_tag(object_type) == Some(ObjectKind::IOSurfaceTexture) {
                crate::observe::Emit::decline(
                    "iosurface_texture_register",
                    &crate::runtime::decode::resource::DecodeDecline(e),
                )
                .field("obj_type", object_type)
                .field("len", desc.len())
                .fail_once(object_type as u64);
            }
            false
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::model::{DeviceId, PAGE_SHIFT_ARM64E};
    use reims_vgpu_core::endian::{st16, st32, st64};

    #[test]
    fn iosurface_texture_geom_and_generation() {
        let mut s = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        assert!(s.map_mapper_surface(
            MapperSurfaceRef::new(5),
            reims_vgpu_protocol::MapperResolvedSurfaceId::new(5)
        ));
        assert_eq!(
            register_iosurface_texture_geom(&mut s, MapperSurfaceRef::new(5), 640, 480, 0x50),
            Ok(5)
        );
        let m = s.surfaces.mappings.get(&5).unwrap();
        assert!(m.has_geometry());
        assert_eq!(
            (m.width_or_zero(), m.height_or_zero(), m.format_or_zero()),
            (640, 480, 0x50)
        );
        assert_eq!(s.mark_mapping_written(5), 1);
        assert_eq!(s.mark_mapping_written(5), 2);

        assert!(s.map_mapper_surface(
            MapperSurfaceRef::new(9),
            reims_vgpu_protocol::MapperResolvedSurfaceId::new(9)
        ));
        let desc = mapper_texture_descriptor(9, 0x73, 100, 50);
        assert!(register_from_descriptor_bytes(&mut s, 11, &desc));
        let m = s.surfaces.mappings.get(&9).unwrap();
        assert_eq!(m.width_or_zero(), 100);
        assert_eq!(m.height_or_zero(), 50);
        assert_eq!(m.format_or_zero(), 0x73);
    }

    #[test]
    fn mapper_resolution_uses_an_explicit_edge_without_narrowing() {
        let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mapper = MapperSurfaceRef::new(0x1_0000_0001);
        assert!(
            state.map_mapper_surface(mapper, reims_vgpu_protocol::MapperResolvedSurfaceId::new(7))
        );
        assert_eq!(
            register_iosurface_texture_geom(&mut state, mapper, 8, 4, 0x73),
            Ok(7)
        );
        assert_eq!(
            state.resolve_mapper_surface(mapper),
            Some(reims_vgpu_protocol::MapperResolvedSurfaceId::new(7))
        );
        assert_eq!(state.resolve_mapper_surface(MapperSurfaceRef::new(1)), None);
    }

    fn mapper_texture_descriptor(
        mapper_ref: u64,
        pixel_format: u16,
        width: u32,
        height: u32,
    ) -> [u8; 0x38] {
        let mut desc = [0u8; 0x38];
        st64(&mut desc[0..], mapper_ref);
        st32(&mut desc[0x08..], 0x0c);
        st32(&mut desc[0x0c..], 0x30);
        st32(&mut desc[0x10..], 1);
        st16(&mut desc[0x16..], pixel_format);
        st32(&mut desc[0x18..], width);
        st32(&mut desc[0x1c..], height);
        desc
    }

    /// A truncated or unknown nested operation cannot latch geometry through
    /// the old headerless-prefix interpretation.
    #[test]
    fn incomplete_or_unknown_mapper_texture_is_refused() {
        let mut s = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let exact = mapper_texture_descriptor(9, 0x73, 8, 4);

        assert!(!register_from_descriptor_bytes(&mut s, 11, &exact[..0x37]));
        assert!(!s.surfaces.mappings.contains_key(&9));

        let mut unknown = exact;
        st32(&mut unknown[0x08..], 0x58);
        assert!(!register_from_descriptor_bytes(&mut s, 11, &unknown));
        assert!(!s.surfaces.mappings.contains_key(&9));
    }
}
