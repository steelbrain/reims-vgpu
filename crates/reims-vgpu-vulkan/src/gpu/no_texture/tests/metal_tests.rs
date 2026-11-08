//! The tests of `no_texture` that compare with Metal on this host: macOS only.

use crate::core::platform::device::GpuDevice;
use crate::core::platform::resource::{Buffer as _, Heap as _, Texture as _};
use crate::core::platform::types::{NsRange, SizeAndAlign, Swizzle};
use crate::core::platform::values::{BufferDesc, HeapDesc, OptionSet, TextureDesc};
use crate::core::raise::OnFault;
use crate::core::raise::death::DeathKind;
use crate::core::testkit::capture::catch_death;

use super::super::NO_VIEW;
use super::*;
use crate::gpu::device::tests::dev;
use crate::os::OsSurface;
use crate::os::tests::bgra_surface;

/// The formats here.
const FORMATS: [u64; 4] = [263, 556, 651, 652];

/// (B, G, R, A).
const BGRA: Option<Swizzle> = Some(Swizzle {
    red: 4,
    green: 3,
    blue: 2,
    alpha: 5,
});

type Case = (&'static str, fn(&mut TextureDesc));

/// What every creator answers of `desc` on `d`: the descriptor's format; whether
/// `newTextureWithDescriptor:`, `newSharedTextureWithDescriptor:`,
/// `newTextureWithDescriptor:iosurface:plane:` (plane 0 of `s`), a buffer's
/// `newTextureWithDescriptor:offset:bytesPerRow:` and
/// `newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:`, and a Private placement
/// heap's `newTextureWithDescriptor:offset:` made a texture, in that order; and
/// `heapTextureSizeAndAlignWithDescriptor:`.
fn answers<P: Platform<Surface = OsSurface>>(
    d: &P::Device,
    desc: &P::TextureDescriptor,
    s: &OsSurface,
) -> (u64, [bool; 6], SizeAndAlign) {
    let buffer = d
        .new_buffer(BufferDesc {
            length: 1 << 16,
            options: vec![OptionSet::StorageMode(0)],
            pinned_gpu_address: None,
            contents: None,
            no_copy: false,
            deallocator: None,
        })
        .unwrap()
        .unwrap();
    let heap = d
        .new_heap(&HeapDesc {
            size: 1 << 20,
            heap_type: 1,
            options: vec![OptionSet::StorageMode(2)],
            address_ranges: None,
            pinned_gpu_address: None,
        })
        .unwrap()
        .heap
        .unwrap();
    let made = [
        d.new_texture(OnFault::Return, desc).unwrap().is_some(),
        d.new_shared_texture(desc).unwrap().is_some(),
        d.new_texture_on_surface(desc, Some(s), 0)
            .unwrap()
            .is_some(),
        buffer
            .new_texture(OnFault::Return, desc, 0, 4096)
            .unwrap()
            .is_some(),
        buffer
            .new_linear_texture(desc, 0, 4096, 4096 * 16)
            .unwrap()
            .is_some(),
        heap.new_texture(desc, 0).unwrap().is_some(),
    ];
    let size = d.heap_texture_size_and_align(desc).unwrap();
    (desc.pixel_format(), made, size)
}

/// A descriptor takes each format here and answers it; every creator answers nil of one
/// Metal's validation passes, and the heap's size query 0 bytes at Metal's alignment, as
/// Metal answers, whatever the creator's own checks would refuse (`framebufferOnly`, a
/// rotation, a storage mode, an empty buffer row)
#[test]
fn textures_of_formats_metal_makes_none_of_are_nil_as_metals_are() {
    let every: [Case; 16] = [
        ("2D", |_| {}),
        ("2D array of 2", |d| {
            (d.texture_type, d.array_length) = (3, 2)
        }),
        ("cube", |d| d.texture_type = 5),
        ("cube array of 341", |d| {
            (d.texture_type, d.array_length) = (6, 341)
        }),
        ("5 levels", |d| d.mipmap_level_count = 5),
        ("16384 wide", |d| d.width = 16384),
        ("shared", |d| d.resource_options = Some(0)),
        ("memoryless render target", |d| {
            (d.usage, d.resource_options) = (4, Some(0x30));
        }),
        ("usage 0", |d| d.usage = 0),
        ("every usage but atomics", |d| d.usage = 0x17),
        ("read swizzle", |d| d.swizzle = BGRA),
        ("written swizzle, enabled", |d| {
            (d.usage, d.swizzle, d.write_swizzle_enabled) = (3, BGRA, Some(true));
        }),
        ("framebufferOnly", |d| d.framebuffer_only = Some(true)),
        ("rotation 2, framebufferOnly", |d| {
            (d.rotation, d.framebuffer_only) = (Some(2), Some(true));
        }),
        ("rotation 4, framebufferOnly", |d| {
            (d.rotation, d.framebuffer_only) = (Some(4), Some(true));
        }),
        ("usage bits Metal takes", |d| d.usage = 0x1_c00d),
    ];
    let not_depth: [Case; 4] = [
        ("1D", |d| (d.texture_type, d.height) = (0, 1)),
        ("1D array of 2048", |d| {
            (d.texture_type, d.height, d.array_length) = (1, 1, 2048);
        }),
        ("3D 4 x 4 x 16, 5 levels", |d| {
            (
                d.texture_type,
                d.width,
                d.height,
                d.depth,
                d.mipmap_level_count,
            ) = (7, 4, 4, 16, 5);
        }),
        ("texture buffer, 7 levels", |d| {
            (d.texture_type, d.width, d.height, d.mipmap_level_count) = (9, 64, 1, 7);
        }),
    ];
    let multisample: [Case; 2] = [
        ("multisample of 2", |d| {
            (d.texture_type, d.sample_count) = (4, 2)
        }),
        ("multisample array of 2048", |d| {
            (d.texture_type, d.sample_count, d.array_length) = (8, 4, 2048);
        }),
    ];
    let d = dev();
    let p = platform();
    let s = bgra_surface(16, 16);
    for format in FORMATS {
        let own: &[Case] = match format {
            556 => &[not_depth.as_slice(), multisample.as_slice()].concat(),
            263 => &not_depth,
            _ => &[],
        };
        for (what, change) in every.iter().chain(own) {
            let desc = of(format, *change);
            let ours =
                answers::<crate::platform::VulkanPlatform>(&d, &p.texture_descriptor(&desc), &s);
            assert_eq!(ours.0, format, "{format} {what}");
            assert_eq!(ours.1, [false; 6], "{format} {what}");
            assert_eq!(ours.2.size, 0, "{format} {what}");
            crate::metal_check!(format!("{format} {what}"), ours);
        }
    }
}

/// Every creator of a texture of a format here but `newSharedTextureWithDescriptor:` dies
/// where Metal's validation asserts (measured: the module's docs), and that one answers nil, as
/// Metal's does
#[test]
fn textures_of_formats_metal_makes_none_of_die_where_metal_asserts() {
    let every: [Case; 19] = [
        ("0 wide", |d| d.width = 0),
        ("16385 wide", |d| d.width = 16385),
        ("multisample of 1 sample", |d| d.texture_type = 4),
        ("atomic", |d| d.usage = 0x21),
        ("2D of 4 samples", |d| d.sample_count = 4),
        ("written swizzle", |d| (d.usage, d.swizzle) = (3, BGRA)),
        ("rendered swizzle", |d| (d.usage, d.swizzle) = (4, BGRA)),
        ("1D of height 2", |d| (d.texture_type, d.height) = (0, 2)),
        ("2D of depth 2", |d| d.depth = 2),
        ("cube 16 x 8", |d| (d.texture_type, d.height) = (5, 8)),
        ("2D of length 2", |d| d.array_length = 2),
        ("cube array of 342", |d| {
            (d.texture_type, d.array_length) = (6, 342)
        }),
        ("6 levels", |d| d.mipmap_level_count = 6),
        ("1D array of 2 levels", |d| {
            (
                d.texture_type,
                d.height,
                d.array_length,
                d.mipmap_level_count,
            ) = (1, 1, 2, 2);
        }),
        ("rotation 2", |d| d.rotation = Some(2)),
        ("rotation 5, framebufferOnly", |d| {
            (d.rotation, d.framebuffer_only) = (Some(5), Some(true));
        }),
        ("multisample of 8", |d| {
            (d.texture_type, d.sample_count) = (4, 8)
        }),
        ("multisample array of 2049", |d| {
            (d.texture_type, d.sample_count, d.array_length) = (8, 4, 2049);
        }),
        ("usage bit 0x40", |d| d.usage = 0x45),
    ];
    let depth: [Case; 4] = [
        ("1D", |d| (d.texture_type, d.height) = (0, 1)),
        ("3D", |d| (d.texture_type, d.depth) = (7, 2)),
        ("texture buffer", |d| {
            (d.texture_type, d.width, d.height) = (9, 64, 1)
        }),
        ("multisample of 4", |d| {
            (d.texture_type, d.sample_count) = (4, 4)
        }),
    ];
    let d = dev();
    let p = platform();
    let s = bgra_surface(16, 16);
    let buffer = d
        .new_buffer(BufferDesc {
            length: 1 << 16,
            options: vec![OptionSet::StorageMode(0)],
            pinned_gpu_address: None,
            contents: None,
            no_copy: false,
            deallocator: None,
        })
        .unwrap()
        .unwrap();
    let heap = d
        .new_heap(&HeapDesc {
            size: 1 << 20,
            heap_type: 1,
            options: vec![OptionSet::StorageMode(2)],
            address_ranges: None,
            pinned_gpu_address: None,
        })
        .unwrap()
        .heap
        .unwrap();
    for format in FORMATS {
        let own: &[Case] = match format {
            651 | 652 => &depth,
            263 => &depth[3..],
            _ => &[],
        };
        for (what, change) in every.iter().chain(own) {
            let desc = of(format, *change);
            let ours = p.texture_descriptor(&desc);
            let deaths = [
                (
                    "newTextureWithDescriptor:",
                    catch_death(|| d.new_texture(OnFault::Return, &ours)).map(drop),
                ),
                (
                    "heapTextureSizeAndAlignWithDescriptor:",
                    catch_death(|| d.heap_texture_size_and_align(&ours)).map(drop),
                ),
                (
                    "newTextureWithDescriptor:iosurface:plane:",
                    catch_death(|| d.new_texture_on_surface(&ours, Some(&s), 0)).map(drop),
                ),
                (
                    "newTextureWithDescriptor:offset:bytesPerRow:",
                    catch_death(|| buffer.new_texture(OnFault::Return, &ours, 0, 4096)).map(drop),
                ),
                (
                    "newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:",
                    catch_death(|| buffer.new_linear_texture(&ours, 0, 4096, 65536)).map(drop),
                ),
                (
                    "newTextureWithDescriptor:offset:",
                    catch_death(|| heap.new_texture(&ours, 0)).map(drop),
                ),
            ];
            for (sel, death) in deaths {
                let site = death.unwrap_err();
                assert_eq!(
                    (site.kind, site.file),
                    (DeathKind::Abort, sel),
                    "{format} {what}"
                );
            }
            let shared = d.new_shared_texture(&ours).unwrap();
            assert!(shared.is_none(), "{format} {what}");
        }
    }
}

/// A view of a texture of a format the runtime has in any format Metal makes no view in is nil
/// from every view selector, with any type, levels and slices, as Metal's; a format next to
/// them dies at the selector, as Metal asserts at a view in it
#[test]
fn views_in_formats_metal_makes_no_view_in_are_nil() {
    fn views<P: Platform>(t: &P::Texture, format: u64) -> [bool; 6] {
        let (levels, slices) = (NsRange::new(0, 9), NsRange::new(0, 1));
        [
            t.new_texture_view(format).unwrap().is_some(),
            t.new_texture_view_at_index(format, 100).unwrap().is_some(),
            t.new_texture_subview(format, 7, levels, slices)
                .unwrap()
                .is_some(),
            t.new_swizzled_texture_subview(format, 7, levels, slices, BGRA.unwrap())
                .unwrap()
                .is_some(),
            t.new_swizzled_texture_subview_at_index(format, 7, levels, slices, BGRA.unwrap(), 101)
                .unwrap()
                .is_some(),
            t.new_texture_subview_at_index(format, 7, levels, slices, 102)
                .unwrap()
                .is_some(),
        ]
    }
    let source = of(80, |d| (d.mipmap_level_count, d.usage) = (5, 0x11));
    let d = dev();
    let ours = d
        .new_texture(OnFault::Return, &platform().texture_descriptor(&source))
        .unwrap()
        .unwrap();
    let formats = NO_VIEW.iter().flat_map(|&(first, last)| first..=last);
    for format in formats {
        assert_eq!(
            views::<crate::platform::VulkanPlatform>(&ours, format),
            [false; 6],
            "{format}"
        );
    }
    let site = catch_death(|| ours.new_texture_view(501)).unwrap_err();
    assert_eq!(
        (site.kind, site.file),
        (DeathKind::Abort, "newTextureViewWithPixelFormat:")
    );
}
