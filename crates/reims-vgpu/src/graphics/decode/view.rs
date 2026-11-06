//! `TEXTURE_VIEW`: a view of a texture the guest made, by the selector the guest called.
//!
//! The host makes the view with the platform's selector of the same form, at the guest-picked
//! resource id: `newTextureViewWithPixelFormat:resourceIndex:` for a view in a format alone,
//! `...textureType:levels:slices:resourceIndex:` for one that names its type, levels and slices,
//! and `...slices:swizzle:resourceIndex:` for one that names its swizzle too. The view lies over
//! the region span its texture lies over, if any, so a block's release finds it.
//!
//! What the view may be is the platform's to decide, as Metal decides it: a view Metal refuses
//! (a format of another class, a type of another family, levels or slices outside the texture,
//! a slice count the type does not take) ends the process in the platform, which the refusal
//! scope the decode runs in turns into this op's refusal, and a view Metal answers nil for is
//! refused too. Either way nothing is made, and the guest's later use of the id fails as a use
//! of any refused create does. A view of a pixel format the device describes none for is
//! refused before the platform is asked, as Metal asserts there ([`super::described`]).

use reims_graphics::wire::{
    RGFX_TEXTURE_VIEW_SUBRANGE, RGFX_TEXTURE_VIEW_SWIZZLE, RgfxTextureView,
};
use reims_vgpu_core::platform::Platform;
use reims_vgpu_core::platform::resource::Texture;
use reims_vgpu_core::platform::types::{NsRange, Swizzle};
use reims_vgpu_core::raise::RaiseResult;

use super::{DecodeError, GraphicsDecoder, ObjHandle, read};

impl<P: Platform> GraphicsDecoder<P> {
    /// `TEXTURE_VIEW`: the platform's view of the named texture (module doc). Malformed for flags
    /// the wire does not define, or a swizzle without its type, levels and slices; a bad reference
    /// for a parent that is no texture the sender holds; a bad resource id as for any texture.
    pub(super) fn texture_view(
        &self,
        device: &P::Device,
        message: &[u8],
    ) -> RaiseResult<Result<(), DecodeError>> {
        let Some(m) = read::<RgfxTextureView>(message) else {
            return Ok(Err(DecodeError::Malformed));
        };
        let subrange = m.flags & RGFX_TEXTURE_VIEW_SUBRANGE != 0;
        let swizzled = m.flags & RGFX_TEXTURE_VIEW_SWIZZLE != 0;
        if m.flags & !(RGFX_TEXTURE_VIEW_SUBRANGE | RGFX_TEXTURE_VIEW_SWIZZLE) != 0
            || (swizzled && !subrange)
        {
            return Ok(Err(DecodeError::Malformed));
        }
        if !self.holds_resource_id(m.obj_id, m.resource_id) {
            return Ok(Err(DecodeError::BadResourceId));
        }
        if !super::described::<P>(device, m.pixel_format.into()) {
            return Ok(Err(DecodeError::Refused));
        }
        let made = {
            let Some(ObjHandle::Texture(parent)) = self.named(m.parent_obj_id) else {
                return Ok(Err(DecodeError::BadReference));
            };
            let format = u64::from(m.pixel_format);
            let texture_type = u64::from(m.texture_type);
            let levels = NsRange::new(m.level_start.into(), m.level_count.into());
            let slices = NsRange::new(m.slice_start.into(), m.slice_count.into());
            match (subrange, swizzled) {
                (false, _) => parent.new_texture_view_at_index(format, m.resource_id)?,
                (true, false) => parent.new_texture_subview_at_index(
                    format,
                    texture_type,
                    levels,
                    slices,
                    m.resource_id,
                )?,
                (true, true) => parent.new_swizzled_texture_subview_at_index(
                    format,
                    texture_type,
                    levels,
                    slices,
                    swizzle(m.swizzle),
                    m.resource_id,
                )?,
            }
        };
        let span = self.span_of(m.parent_obj_id);
        match made {
            Some(view) => self.insert_over(m.obj_id, ObjHandle::Texture(view), span),
            None => Ok(Err(DecodeError::Refused)),
        }
    }
}

/// The four `MTLTextureSwizzle` channels packed one byte each, red lowest.
fn swizzle(packed: u32) -> Swizzle {
    let [red, green, blue, alpha] = packed.to_le_bytes();
    Swizzle {
        red,
        green,
        blue,
        alpha,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use reims_graphics::wire::{
        RGFX_CMD_RESOURCE_ID_CHUNK, RGFX_CMD_TEXTURE_CREATE, RGFX_CMD_TEXTURE_VIEW,
        RGFX_CONTEXT_ALLOCATIONS_OFFSET, RGFX_CURSOR_SPAN, RgfxMsgHeader, RgfxResourceIdChunk,
        RgfxTextureCreate, RgfxTextureDescriptor,
    };
    use reims_vgpu_core::testkit::{Arg, FakePlatform};
    use zerocopy::IntoBytes;

    use super::*;

    fn header(r#type: u32, size: usize) -> RgfxMsgHeader {
        RgfxMsgHeader {
            r#type,
            size: size as u32,
            seqno: 0,
        }
    }

    /// A decoder whose context (0) holds the resource ids 65 to 128, with a 2D array texture 1
    /// of four slices and seven levels at resource id 65.
    fn decoder() -> (Arc<FakePlatform>, GraphicsDecoder<FakePlatform>) {
        let p = Arc::new(FakePlatform::new());
        let size = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x10_0000 + RGFX_CURSOR_SPAN;
        let region = Arc::new(reims_graphics::region::Region::new(size as usize).unwrap());
        let mut dec = GraphicsDecoder::<FakePlatform>::new(Arc::clone(&p), region);
        let device = p.device();
        let chunk = RgfxResourceIdChunk {
            header: header(RGFX_CMD_RESOURCE_ID_CHUNK, size_of::<RgfxResourceIdChunk>()),
            context: 0,
            chunk: 1,
        };
        assert_eq!(dec.decode(&device, chunk.as_bytes()).unwrap(), Ok(()));
        let texture = RgfxTextureCreate {
            header: header(RGFX_CMD_TEXTURE_CREATE, size_of::<RgfxTextureCreate>()),
            obj_id: 1,
            gpu_address: 0x5_0000,
            resource_id: 65,
            desc: RgfxTextureDescriptor {
                pixel_format: 70, // RGBA8Unorm
                texture_type: 3,  // 2DArray
                width: 64,
                height: 32,
                depth: 1,
                mipmap_level_count: 7,
                array_length: 4,
                sample_count: 1,
                usage: 0x11, // ShaderRead | PixelFormatView
                ..RgfxTextureDescriptor::default()
            },
            ..RgfxTextureCreate::default()
        };
        assert_eq!(dec.decode(&device, texture.as_bytes()).unwrap(), Ok(()));
        (p, dec)
    }

    fn view(obj_id: u64, parent: u64, resource_id: u64, flags: u32) -> RgfxTextureView {
        RgfxTextureView {
            header: header(RGFX_CMD_TEXTURE_VIEW, size_of::<RgfxTextureView>()),
            obj_id,
            parent_obj_id: parent,
            resource_id,
            pixel_format: 71, // RGBA8Unorm_sRGB
            flags,
            ..RgfxTextureView::default()
        }
    }

    /// Each form of view is made by the platform's selector of that form, at the view's resource
    /// id, with the levels, slices and swizzle the message names; a view of a view is made of it
    #[test]
    fn a_view_is_made_by_the_selector_of_its_form() {
        let (p, mut dec) = decoder();
        let device = p.device();
        assert_eq!(
            dec.decode(&device, view(2, 1, 66, 0).as_bytes()).unwrap(),
            Ok(())
        );
        let sub = RgfxTextureView {
            texture_type: 2, // 2D
            level_start: 1,
            level_count: 2,
            slice_start: 3,
            slice_count: 1,
            ..view(3, 1, 67, RGFX_TEXTURE_VIEW_SUBRANGE)
        };
        assert_eq!(dec.decode(&device, sub.as_bytes()).unwrap(), Ok(()));
        let swizzled = RgfxTextureView {
            swizzle: 0x0502_0202, // red, red, red, one
            ..RgfxTextureView {
                obj_id: 4,
                resource_id: 68,
                parent_obj_id: 3,
                flags: RGFX_TEXTURE_VIEW_SUBRANGE | RGFX_TEXTURE_VIEW_SWIZZLE,
                level_start: 0,
                ..sub
            }
        };
        assert_eq!(dec.decode(&device, swizzled.as_bytes()).unwrap(), Ok(()));
        assert_eq!(dec.object_count(), 4);

        let log = p.call_log();
        let format_only = log.calls_to("newTextureViewWithPixelFormat:resourceIndex:");
        assert_eq!(format_only.len(), 1);
        assert_eq!(format_only[0].args, vec![Arg::U64(71), Arg::U64(66)]);
        let subview =
            log.calls_to("newTextureViewWithPixelFormat:textureType:levels:slices:resourceIndex:");
        assert_eq!(subview.len(), 1);
        assert_eq!(
            subview[0].args,
            vec![
                Arg::U64(71),
                Arg::U64(2),
                Arg::val(&NsRange::new(1, 2)),
                Arg::val(&NsRange::new(3, 1)),
                Arg::U64(67),
            ]
        );
        let swizzled = log.calls_to(
            "newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:resourceIndex:",
        );
        assert_eq!(swizzled.len(), 1);
        assert_eq!(
            swizzled[0].args,
            vec![
                Arg::U64(71),
                Arg::U64(2),
                Arg::val(&NsRange::new(0, 2)),
                Arg::val(&NsRange::new(3, 1)),
                Arg::val(&Swizzle {
                    red: 2,
                    green: 2,
                    blue: 2,
                    alpha: 5,
                }),
                Arg::U64(68),
            ]
        );
    }

    /// A view of no texture the sender holds is a bad reference, one at a resource id its
    /// context holds no chunk of a bad resource id, and flags the wire does not define, or a
    /// swizzle with no levels and slices, malformed: the platform is not called for any of them
    #[test]
    fn a_view_the_message_cannot_name_is_refused_before_the_platform() {
        let (p, mut dec) = decoder();
        let device = p.device();
        let cases = [
            (view(2, 9, 66, 0), DecodeError::BadReference),
            (view(2, 1, 200, 0), DecodeError::BadResourceId),
            (view(2, 1, 66, 4), DecodeError::Malformed),
            (
                view(2, 1, 66, RGFX_TEXTURE_VIEW_SWIZZLE),
                DecodeError::Malformed,
            ),
        ];
        for (m, e) in cases {
            assert_eq!(dec.decode(&device, m.as_bytes()).unwrap(), Err(e), "{m:?}");
        }
        assert_eq!(dec.object_count(), 1);
        let views = p
            .call_log()
            .calls()
            .into_iter()
            .filter(|c| c.sel.starts_with("newTextureView"))
            .count();
        assert_eq!(views, 0);
    }
}
