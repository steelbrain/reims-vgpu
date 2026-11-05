//! Metal's GPU objects: the device's queries and creators ([`device`]), resources and state
//! objects ([`resource`](mod@resource)), rasterization rate maps ([`rate_map`]), the descriptors
//! ([`desc`]), the encoders, the indirect command buffers, the command queue and buffer, the
//! events, and our own shaders ([`shaders`]).
//!
//! Every Metal method core asks for is one typed call (objc2-metal's, or `spi`'s declaration),
//! run through the message layer ([`crate::msg`]), named by its selector. The encoder families
//! (`BlitCmd`, `ComputeCmd`, `RenderCmd`, `IcbRenderCmd`, `IcbComputeCmd`) are each one
//! exhaustive `match` in their trait's `encode`, one call per variant.

mod argument;
pub mod command;
pub mod desc;
pub mod device;
pub mod encode_blit;
pub mod encode_compute;
pub mod encode_render;
pub mod event;
pub mod icb;
pub mod rate_map;
pub mod resource;
pub mod scaler;
pub mod shaders;

use std::ptr::NonNull;

use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLEvent, MTLResource};
use reims_vgpu_core::num::ToU64;
use reims_vgpu_core::platform::blit::ResourceRef;
use reims_vgpu_core::platform::event::EventRef;
use reims_vgpu_core::platform::types::NsRange;
use reims_vgpu_core::raise::death::DeathKind;

use crate::platform::MetalPlatform;

/// The `id<MTLResource>` a resource argument is: a buffer, a texture or an indirect command
/// buffer.
pub(crate) fn resource(r: ResourceRef<'_, MetalPlatform>) -> &ProtocolObject<dyn MTLResource> {
    match r {
        ResourceRef::Buffer(b) => ProtocolObject::from_ref(b.get()),
        ResourceRef::Texture(t) => ProtocolObject::from_ref(t.get()),
        ResourceRef::Icb(i) => ProtocolObject::from_ref(i.get()),
    }
}

/// The event a command buffer waits on or signals, as an `id<MTLEvent>`.
pub(crate) fn event(e: EventRef<'_, MetalPlatform>) -> &ProtocolObject<dyn MTLEvent> {
    match e {
        EventRef::Event(e) => e.get(),
        EventRef::Shared(e) => ProtocolObject::from_ref(e.get()),
    }
}

/// A C array of objects (nil entries included, as NULL), built before the call.
pub(crate) fn objects<'a, T: 'a>(objs: impl Iterator<Item = Option<&'a T>>) -> Vec<*const T> {
    objs.map(|o| o.map_or(std::ptr::null(), std::ptr::from_ref))
        .collect()
}

/// The start of a C array as the typed pointer a call takes (dangling but non-null for an empty
/// one, which Metal does not read). Metal reads a nil entry as NULL whatever the element type
/// objc2-metal names; nothing on the Rust side reads the elements.
pub(crate) fn array<T, U>(values: &[T]) -> NonNull<U> {
    NonNull::from(values).cast::<U>()
}

/// A `…:withRange:` send reads `range.length` entries of each array it is passed. An array
/// shorter than that is a bug in the caller: Metal would read past it, so the process dies at
/// the send (`OutOfBounds` at `sel`) instead of sending.
pub(crate) fn check_len(sel: &'static str, len: usize, range: NsRange) {
    if len.to_u64() < range.length {
        reims_vgpu_core::raise::death::fatal(reims_vgpu_core::raise::death::DeathId::Platform {
            kind: DeathKind::OutOfBounds,
            call: sel,
        })
    }
}

#[cfg(test)]
mod tests {
    use reims_vgpu_core::raise::death::DeathKind;
    use reims_vgpu_core::testkit::capture::catch_death;

    use super::*;
    use crate::gpu::resource::testkit::*;

    /// check_len passes arrays at least as long as the range, and dies otherwise, at the
    /// selector
    #[test]
    fn check_len_dies_on_short_arrays() {
        check_len("setTextures:withRange:", 2, NsRange::new(5, 2));
        check_len("setTextures:withRange:", 3, NsRange::new(0, 2));
        let site = catch_death(|| check_len("setVertexTextures:withRange:", 1, NsRange::new(0, 2)))
            .unwrap_err();
        assert_eq!(
            (site.kind, site.file, site.line),
            (DeathKind::OutOfBounds, "setVertexTextures:withRange:", 0)
        );
    }

    /// objects keeps nil entries as NULL in order; event and resource are the objects
    /// themselves
    #[test]
    fn glue_helpers() {
        let k = kit();
        let v = objects([Some(k.buffer.get()), None].into_iter());
        assert_eq!(v.len(), 2);
        assert_eq!(v[0], std::ptr::from_ref(k.buffer.get()));
        assert!(v[1].is_null());
        let p: NonNull<*const u8> = array(&v);
        assert_eq!(p.as_ptr().cast::<*const u8>(), v.as_ptr().cast_mut().cast());
        let raw = |o: &ProtocolObject<dyn MTLEvent>| std::ptr::from_ref(o).cast::<u8>();
        assert_eq!(
            raw(event(EventRef::Shared(&k.shared_event))),
            std::ptr::from_ref(k.shared_event.get()).cast::<u8>()
        );
        let raw = |o: &ProtocolObject<dyn MTLResource>| std::ptr::from_ref(o).cast::<u8>();
        assert_eq!(
            raw(resource(ResourceRef::Texture(&k.texture))),
            std::ptr::from_ref(k.texture.get()).cast::<u8>()
        );
        assert_eq!(
            raw(resource(ResourceRef::Icb(&k.icb_render))),
            std::ptr::from_ref(k.icb_render.get()).cast::<u8>()
        );
        assert_eq!(
            raw(resource(ResourceRef::Buffer(&k.buffer))),
            std::ptr::from_ref(k.buffer.get()).cast::<u8>()
        );
    }
}
