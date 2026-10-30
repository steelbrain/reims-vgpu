//! The runtime's GPU objects: the device's queries and creators ([`device`]), capabilities
//! ([`caps`]) and pixel format descriptions ([`format_info`]), resources and state objects
//! ([`resource`]) and the guest's pages under them ([`backing`]), the descriptors ([`desc`]) and
//! Apple's serialized pipeline descriptors ([`serialized`]), the
//! encoders, the indirect command buffers, rate maps, the command queue and buffer, and the
//! events; Metal's GPU address space, which buffers and heaps are placed by ([`metal_space`]); the
//! two-plane YCbCr formats ([`ycbcr`]) and the formats it makes no texture of
//! ([`no_texture`]); what GPU work may write, where a write that never reaches a surface's memory
//! dies ([`writes`]); and the debug lines that time the runtime's compiles ([`timing`]).
//!
//! Every Metal method core asks for is one runtime call, run through the answer layer
//! ([`crate::answer`]) and named by its selector. The encoder families (`BlitCmd`, `ComputeCmd`,
//! `RenderCmd`, `IcbRenderCmd`, `IcbComputeCmd`) are each one exhaustive `match` in their trait's
//! `encode`, one call per variant.
//!
//! The raw values core passes (`u64` pixel formats, texture types, options) become the runtime's
//! typed values here: a value the runtime's type cannot hold dies at the selector that took it
//! ([`crate::answer::unrepresentable`]).

pub mod backing;
pub mod caps;
pub mod command;
pub mod desc;
pub mod device;
pub mod encode_blit;
pub mod encode_compute;
pub mod encode_render;
pub mod event;
pub mod fill;
pub mod format_info;
#[cfg(test)]
mod frame_cost;
pub mod icb;
pub mod identity;
pub mod metal_space;
pub mod no_texture;
#[cfg(test)]
mod paging_discard;
pub mod predication;
pub mod rate_map;
pub mod resource;
#[cfg(test)]
#[cfg(target_os = "macos")]
mod rgbx;
pub mod scaler;
pub mod serialized;
pub mod shaders;
#[cfg(test)]
#[cfg(target_os = "macos")]
mod slice_write_repro;
#[cfg(test)]
mod spin;
pub mod states;
pub mod timing;
#[cfg(test)]
#[cfg(target_os = "macos")]
mod undefined;
pub mod writes;
pub mod ycbcr;

use std::ops::Range;

use metal2vulkan::metal::{
    PixelFormat, TextureSwizzle, TextureSwizzleChannels, TextureType, TextureUsage,
};
use metal2vulkan_runtime as rt;
use reims_vgpu_core::platform::types::{MtlOrigin, MtlRegion, MtlSize, NsRange, Swizzle};

use crate::answer::unrepresentable;

/// The most elements a bind's list holds on the stack ([`listed`]): Metal's 31 buffer slots of a
/// stage, and as many textures or samplers.
const LISTED: usize = 31;

/// Runs `call` with each element of `list` as `map` makes it, in order: a bind's list of core's
/// objects as the runtime's, which every bind of several slots makes, on the stack where it holds
/// at most [`LISTED`], else in a `Vec`. The runtime reads the slice alone, so where it lies
/// changes nothing it answers.
pub(crate) fn listed<A, B: Copy + Default, R>(
    list: &[A],
    mut map: impl FnMut(&A) -> B,
    call: impl FnOnce(&[B]) -> R,
) -> R {
    if list.len() > LISTED {
        let all: Vec<B> = list.iter().map(map).collect();
        return call(&all);
    }
    let mut on_stack = [B::default(); LISTED];
    for (to, from) in on_stack.iter_mut().zip(list) {
        *to = map(from);
    }
    call(&on_stack[..list.len()])
}

/// `MTLSize`.
pub(crate) fn size(s: MtlSize) -> rt::Size {
    rt::Size::new(s.width, s.height, s.depth)
}

/// `MTLOrigin`.
pub(crate) fn origin(o: MtlOrigin) -> rt::Origin {
    rt::Origin::new(o.x, o.y, o.z)
}

/// `MTLRegion`.
pub(crate) fn region(r: MtlRegion) -> rt::Region {
    rt::Region::new_3d(
        r.origin.x,
        r.origin.y,
        r.origin.z,
        r.size.width,
        r.size.height,
        r.size.depth,
    )
}

/// `NSRange` as the range of indices it names; `None` where its end passes `u64::MAX`.
pub(crate) fn range(r: NsRange) -> Option<Range<u64>> {
    r.location.checked_add(r.length).map(|end| r.location..end)
}

/// A raw `MTLPixelFormat`, sent with `sel`.
pub(crate) fn pixel_format(sel: &'static str, raw: u64) -> PixelFormat {
    runtime_pixel_format(raw)
        .unwrap_or_else(|| unrepresentable(sel, &format!("pixel format {raw}")))
}

/// The runtime's format of a raw `MTLPixelFormat`: `None` for a value it has no enumerator of.
pub(crate) fn runtime_pixel_format(raw: u64) -> Option<PixelFormat> {
    u32::try_from(raw)
        .ok()
        .and_then(|raw| PixelFormat::try_from(raw).ok())
}

/// The runtime's format of a raw `MTLPixelFormat` Metal takes, or the raw value, where the runtime
/// has no format of it or Metal describes none for it (`Depth24Unorm_Stencil8` and `X24_Stencil8`,
/// which the runtime has), but `MTLPixelFormatInvalid`: a texture or a state of a raw value dies
/// where it is made, or is nil, as Metal's ([`crate::gpu::desc::lacked`],
/// [`crate::gpu::serialized::RawFormat`]). `sel` asks Metal's description
/// ([`crate::gpu::format_info::described`]).
pub(crate) fn held_format(sel: &'static str, raw: u64) -> Result<PixelFormat, u64> {
    held_by(raw, |raw| crate::gpu::format_info::described(sel, raw))
}

/// The runtime's format of `raw` where `described` says Metal describes it, or `raw` itself.
fn held_by(raw: u64, described: impl FnOnce(u64) -> bool) -> Result<PixelFormat, u64> {
    match runtime_pixel_format(raw) {
        Some(pf) if raw == 0 || described(raw) => Ok(pf),
        _ => Err(raw),
    }
}

/// A raw `MTLTextureType`, sent with `sel`.
pub(crate) fn texture_type(sel: &'static str, raw: u64) -> TextureType {
    runtime_texture_type(raw)
        .unwrap_or_else(|| unrepresentable(sel, &format!("texture type {raw}")))
}

/// The runtime's type of a raw `MTLTextureType`: `None` for a value it has no enumerator of.
pub(crate) fn runtime_texture_type(raw: u64) -> Option<TextureType> {
    u32::try_from(raw)
        .ok()
        .and_then(|raw| TextureType::try_from(raw).ok())
}

/// A raw `MTLTextureUsage`, sent with `sel`.
pub(crate) fn texture_usage(sel: &'static str, raw: u64) -> TextureUsage {
    u32::try_from(raw)
        .ok()
        .and_then(|raw| TextureUsage::try_from(raw).ok())
        .unwrap_or_else(|| unrepresentable(sel, &format!("texture usage {raw:#x}")))
}

/// Core's `MTLTextureSwizzleChannels`, sent with `sel`.
pub(crate) fn swizzle(sel: &'static str, s: Swizzle) -> TextureSwizzleChannels {
    runtime_swizzle(s).unwrap_or_else(|| {
        unrepresentable(
            sel,
            &format!(
                "texture swizzle ({}, {}, {}, {})",
                s.red, s.green, s.blue, s.alpha
            ),
        )
    })
}

/// The runtime's swizzle of core's: `None` where a channel is a value it has no enumerator of.
pub(crate) fn runtime_swizzle(s: Swizzle) -> Option<TextureSwizzleChannels> {
    let channel = |raw: u8| TextureSwizzle::try_from(u32::from(raw)).ok();
    Some(TextureSwizzleChannels {
        red: channel(s.red)?,
        green: channel(s.green)?,
        blue: channel(s.blue)?,
        alpha: channel(s.alpha)?,
    })
}

/// A raw value of the Metal enum or option set `T`, which `what` names, sent with `sel`.
pub(crate) fn metal_value<T: TryFrom<u32>>(sel: &'static str, what: &str, raw: u64) -> T {
    u32::try_from(raw)
        .ok()
        .and_then(|raw| T::try_from(raw).ok())
        .unwrap_or_else(|| unrepresentable(sel, &format!("{what} {raw:#x}")))
}

/// A buffer of the platform's own (a predication counter, a fill's staging, a draw's index
/// buffer: none a guest names) of `length` bytes and `options`, holding `contents`, outside the
/// Metal address space (`AddressPlacement::Outside`): it takes no Metal address, so no guest
/// buffer's `gpuAddress` depends on when the work that uses it completes, as Metal's own
/// allocations move none of a program's.
pub(crate) fn own_buffer(
    device: &rt::Device,
    length: u64,
    options: rt::ResourceOptions,
    contents: rt::BufferContents<'_>,
) -> Result<rt::Buffer, rt::Error> {
    let descriptor = rt::BufferDescriptor::new(length, options, contents)
        .with_placement(rt::AddressPlacement::Outside);
    device.new_buffer_with_descriptor(descriptor)
}

/// A raw `MTLResourceOptions`, sent with `sel`.
pub(crate) fn resource_options(sel: &'static str, raw: u64) -> rt::ResourceOptions {
    rt::ResourceOptions::from_raw(raw)
        .unwrap_or_else(|_| unrepresentable(sel, &format!("resource options {raw:#x}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use reims_vgpu_core::raise::death::DeathKind;
    use reims_vgpu_core::testkit::capture::catch_death;

    /// The platform's own buffers lie outside the Metal address space, `[2^47, 2^48)`, where no
    /// guest buffer lies
    #[test]
    fn own_buffers_take_no_metal_address() {
        use reims_vgpu_core::platform::device::GpuDevice as _;
        use reims_vgpu_core::platform::resource::Buffer as _;
        use reims_vgpu_core::raise::OnFault;

        let d = crate::gpu::device::tests::dev();
        let shared = rt::ResourceOptions::shared();
        let own = own_buffer(&d.raw, 64, shared, rt::BufferContents::Copy(&[7; 64])).unwrap();
        let guest = d
            .new_buffer_of_length(OnFault::Return, 64, 0)
            .unwrap()
            .unwrap();
        assert!((1 << 47..1 << 48).contains(&own.gpu_address().0));
        assert!(guest.gpu_address() < 1 << 47);
    }

    /// Core's values carry each field to the runtime's; a raw value the runtime cannot hold dies
    /// at the selector
    #[test]
    fn values_convert() {
        assert_eq!(size(MtlSize::new(1, 2, 3)), rt::Size::new(1, 2, 3));
        assert_eq!(origin(MtlOrigin::new(4, 5, 6)), rt::Origin::new(4, 5, 6));
        let r = region(MtlRegion::new(
            MtlOrigin::new(1, 2, 0),
            MtlSize::new(3, 4, 1),
        ));
        assert_eq!(r, rt::Region::new_3d(1, 2, 0, 3, 4, 1));
        assert_eq!(range(NsRange::new(2, 3)), Some(2..5));
        assert_eq!(range(NsRange::new(u64::MAX, 1)), None);
        assert_eq!(pixel_format("setPixelFormat:", 80), PixelFormat::BGRA8Unorm);
        assert_eq!(texture_type("setTextureType:", 2), TextureType::Type2D);
        assert_eq!(texture_usage("setUsage:", 3).as_raw(), 3);
        assert_eq!(resource_options("x", 0x20).raw(), 0x20);
        let site = catch_death(|| pixel_format("setPixelFormat:", 1 << 40)).unwrap_err();
        assert_eq!(
            (site.kind, site.file),
            (DeathKind::Abort, "setPixelFormat:")
        );
        assert_eq!(
            resource_options("x", rt::ResourceOptions::PRIVATE_BIT | 0x20).raw(),
            0x100_0020,
            "Metal's private bit 24, which the runtime takes"
        );
        let site =
            catch_death(|| resource_options("newBufferWithLength:options:", 1 << 25)).unwrap_err();
        assert_eq!(site.file, "newBufferWithLength:options:");
    }

    /// A bind's list reaches the runtime as the elements `map` makes, in order, at every length:
    /// on the stack up to [`LISTED`], in a `Vec` past it
    #[test]
    fn listed_lists_are_the_mapped_elements_in_order() {
        for len in [0, 1, 2, LISTED - 1, LISTED, LISTED + 1, 128] {
            let list: Vec<Option<u32>> = (0..len)
                .map(|i| u32::try_from(i).ok().filter(|i| i % 3 != 1))
                .collect();
            let map = |v: &Option<u32>| v.map(|v| u64::from(v) * 7);
            let expected: Vec<Option<u64>> = list.iter().map(map).collect();
            let got = listed(&list, map, <[Option<u64>]>::to_vec);
            assert_eq!(got, expected, "{len} elements");
        }
    }
}
