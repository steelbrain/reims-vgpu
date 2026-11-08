//! The tests of `device` that compare with Metal on this host: macOS only.

use super::*;

/// A reservation keeps its IDs from plain allocation of textures, sampler states and
/// indirect command buffers, as Metal's does: only a type's first call counts, and none made
/// once an ID of the type was handed out. Each test runs in a process of its own, so both
/// platforms' IDs start fresh: Metal's are read first, before MoltenVK makes objects of its
/// own on Metal's device. Another type dies
#[test]
fn reservations_keep_ids_from_plain_allocation() {
    use reims_vgpu_core::platform::Platform;
    use reims_vgpu_core::platform::icb::IndirectCommandBuffer as _;
    use reims_vgpu_core::platform::resource::{SamplerState as _, Texture as _};
    use reims_vgpu_core::platform::values::TextureDesc;
    use reims_vgpu_metal::MetalPlatform;

    fn ids<P: Platform>(d: &P::Device, texture: &P::TextureDescriptor) -> Vec<Vec<u64>> {
        for (kind, indices) in [
            (0, &[3, 6][..]),
            (0, &[4]),
            (1, &[2]),
            (5, &[3, 6]),
            (5, &[4]),
        ] {
            d.reserve_resource_indices(kind, indices).unwrap();
        }
        // Samplers first: Metal's first texture takes a sampler ID of its own.
        let samplers: Vec<P::SamplerState> = (1..=4u8)
            .map(|lod| {
                let desc = SamplerDesc {
                    support_argument_buffers: Some(true),
                    lod_max_clamp: Some(f32::from(lod)),
                    ..SamplerDesc::default()
                };
                d.new_sampler_state(&desc).unwrap().unwrap()
            })
            .collect();
        let textures: Vec<P::Texture> = (0..6)
            .map(|_| d.new_texture(OnFault::Return, texture).unwrap().unwrap())
            .collect();
        let icb = IcbDesc {
            command_types: 0x60,
            max_kernel_buffer_bind_count: 2,
            ..IcbDesc::default()
        };
        let make = || d.new_indirect_command_buffer(&icb, 1, 0).unwrap().unwrap();
        let mut icbs: Vec<P::Icb> = (0..5).map(|_| make()).collect();
        d.reserve_resource_indices(5, &[8]).unwrap();
        icbs.push(make());
        vec![
            textures.iter().map(|t| t.gpu_resource_id()).collect(),
            samplers.iter().map(|s| s.gpu_resource_id()).collect(),
            icbs.iter().map(|b| b.gpu_resource_id()).collect(),
        ]
    }
    let expected = vec![
        vec![1, 2, 4, 5, 7, 8],
        vec![1, 3, 4, 5],
        vec![1, 2, 4, 5, 7, 8],
    ];
    if let Some(md) = MetalPlatform::system_default_device() {
        let desc = reims_vgpu_metal::gpu::desc::texture_descriptor(&TextureDesc {
            pixel_format: 80,
            width: 4,
            height: 4,
            usage: 0x13,
            resource_options: Some(0),
            ..TextureDesc::default()
        });
        assert_eq!(ids::<MetalPlatform>(&md, &desc), expected);
    }
    let d = dev();
    let texture = crate::gpu::resource::tests::bgra(4, 4, 0);
    assert_eq!(ids::<VulkanPlatform>(&d, &texture), expected);
    let site = catch_death(|| d.reserve_resource_indices(2, &[1])).unwrap_err();
    assert_eq!(
        (site.kind, site.file),
        (
            DeathKind::Abort,
            "reserveResourceIndicesForResourceType:indices:indexCount:"
        )
    );
}

/// A `YCBCR8_420_2P` texture of `size` over plane `plane` of `surface` on the device `d`, as the
/// guest's creator makes it.
fn ycbcr8_420_on(
    d: &VulkanDevice,
    surface: &OsSurface,
    plane: u64,
    size: (u64, u64),
) -> RaiseResult<Option<VulkanTexture>> {
    use reims_vgpu_core::platform::Platform as _;
    let desc = crate::platform::tests::platform()
        .texture_descriptor(&crate::gpu::ycbcr::measure::desc(500, size));
    d.new_texture_on_surface(&desc, Some(surface), plane)
}

/// A probe of [`linear_samples`]: its grid, where it starts and its step, in texels.
type Probe = ((u64, u64), (f32, f32), f32);

/// The probe of [`linear_samples`] over a texture of `size` that a fragment function samples
/// magnified: a position a texel, three quarters of a texel apart from a quarter and three
/// quarters of a texel into the first.
fn between(size: (u64, u64)) -> Probe {
    (size, (0.25, 0.75), 0.75)
}

/// The probe of [`linear_samples`] over a texture of `size` that a fragment function samples
/// minified: a position every other texel across and down, a texel and three quarters apart
/// from (0.3, 0.6) texels, so no position is a texel's centre on either axis.
fn minified(size: (u64, u64)) -> Probe {
    ((size.0 / 2, size.1 / 2), (0.3, 0.6), 1.75)
}

/// The bytes of `probe`, in texels or `normalized`, and its positions in texels, in rows, as
/// the GPU computes them.
fn positions((grid, origin, step): Probe, normalized: bool) -> (Vec<u8>, Vec<(f32, f32)>) {
    let bytes = crate::gpu::ycbcr::measure::probe(origin, step, grid, normalized, 0);
    let at = |g: u64| f32::from(u16::try_from(g).unwrap()) * step;
    let at = (0..grid.1)
        .flat_map(|y| (0..grid.0).map(move |x| (origin.0 + at(x), origin.1 + at(y))))
        .collect();
    (bytes, at)
}

/// How many of `samples`, read into floats at `at` (in texels) of a texture whose `read()`
/// into floats is `texels`, of `size`, differ from each of the four texels nearest them: what
/// a linear filter blended, which a sample at a texel's centre, or a nearest one, never is.
fn blended(
    texels: &[[f32; 4]],
    (width, height): (u64, u64),
    at: &[(f32, f32)],
    samples: &[[f32; 4]],
) -> usize {
    let last = |n: u64| i64::try_from(n).unwrap() - 1;
    let texel = |x: i64, y: i64| {
        let (x, y) = (x.clamp(0, last(width)), y.clamp(0, last(height)));
        texels[usize::try_from(y * (last(width) + 1) + x).unwrap()]
    };
    // The texel before a position on an axis, whose centre is at or before it.
    let lower = |p: f32| (p - 0.5).floor() as i64;
    at.iter()
        .zip(samples)
        .filter(|((x, y), s)| {
            let (i, j) = (lower(*x), lower(*y));
            [(i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)]
                .iter()
                .all(|&(a, b)| texel(a, b) != **s)
        })
        .count()
}

/// A linear sampler of `p`'s device `d`, clamped to the edge, of `normalized` coordinates or
/// not, with a linear mip filter where `mipped`.
fn linear<P: reims_vgpu_core::platform::Platform>(
    d: &P::Device,
    normalized: bool,
    mipped: bool,
) -> P::SamplerState {
    let desc = SamplerDesc {
        min_filter: Some(1),
        mag_filter: Some(1),
        mip_filter: mipped.then_some(2),
        s_address_mode: Some(0),
        t_address_mode: Some(0),
        normalized_coordinates: Some(normalized),
        ..SamplerDesc::default()
    };
    d.new_sampler_state(&desc).unwrap().unwrap()
}

/// A read of [`linear_samples`]: its name, its bytes, and the positions it sampled, in texels.
type Linear = (String, Vec<u8>, Vec<(f32, f32)>);

/// What linear samples of `t` of `size` read on `p`'s device `d` between its texels, each
/// named: a kernel's and, where `draws`, a fragment function's, magnified ([`between`]),
/// through a runtime sampler without a mip filter, in texels and normalized, and through one
/// with a linear mip filter, in texels; where `draws`, a fragment function's minified
/// ([`minified`]) through the same three samplers; and a kernel's through a constexpr
/// sampler; each into halves and into floats. A kernel's sample has no derivatives, so it
/// reads level 0 magnified.
fn linear_samples<P: reims_vgpu_core::platform::Platform>(
    p: &P,
    d: &P::Device,
    t: &P::Texture,
    size: (u64, u64),
    draws: bool,
) -> Vec<Linear> {
    use crate::gpu::ycbcr::measure::drawn;
    use crate::gpu::ycbcr::measure::probed_texels;
    let mut reads = Vec::new();
    let fragments: &[(&str, u64)] = match draws {
        true => &[("sample_fragment", 8), ("sample_fragment_float", 16)],
        false => &[],
    };
    for (normalized, mipped) in [(false, false), (true, false), (false, true)] {
        let s = linear::<P>(d, normalized, mipped);
        let (bytes, at) = positions(between(size), normalized);
        let how = format!("normalized {normalized} mipped {mipped}");
        for (kernel, texel) in [("sample_texels", 8), ("sample_texels_float", 16)] {
            let read = probed_texels::<P>(d, (kernel, texel), t, size, Some(&s), Some(&bytes));
            reads.push((format!("{kernel} {how}"), read, at.clone()));
        }
        for &(fragment, texel) in fragments {
            let read = drawn::<P>(p, d, (fragment, texel), t, size, &s, &bytes);
            reads.push((format!("{fragment} {how}"), read, at.clone()));
        }
    }
    for (normalized, mipped) in [(false, false), (true, false), (false, true)] {
        let s = linear::<P>(d, normalized, mipped);
        let probe = minified(size);
        let (bytes, at) = positions(probe, normalized);
        let how = format!("minified normalized {normalized} mipped {mipped}");
        for &(fragment, texel) in fragments {
            let read = drawn::<P>(p, d, (fragment, texel), t, probe.0, &s, &bytes);
            reads.push((format!("{fragment} {how}"), read, at.clone()));
        }
    }
    let (bytes, at) = positions(between(size), false);
    for (kernel, texel) in [
        ("sample_constexpr_linear", 8),
        ("sample_constexpr_linear_float", 16),
    ] {
        let read = probed_texels::<P>(d, (kernel, texel), t, size, None, Some(&bytes));
        reads.push((kernel.to_owned(), read, at.clone()));
    }
    reads
}

/// A `YCBCR8_420_2P` texture over plane 0 of a '420v' or a '420f' surface, of no YCbCr matrix,
/// BT.601, BT.709, BT.2020 or SMPTE 240M, reads each texel as Metal's does, by `read()` into
/// floats and by a nearest sample at each texel's centre, each compared with Metal's own, odd
/// sizes included: the runtime's, over the surface's facts ([`crate::os::ycbcr_facts`]). Its
/// linear samples between the texels in a kernel are Metal's, bit for bit
/// ([`linear_samples`]): magnified, into halves and into floats, in texels and normalized,
/// through a runtime sampler with a linear mip filter or none, which the runtime announces
/// neither kind of filter notice for, and through a constexpr sampler; and so are a fragment
/// function's that samples one and writes a buffer (`sample_fragment`,
/// `sample_fragment_float`), magnified and minified, which the runtime draws (its exact filter's
/// branch on the sampler's specialization constants, which every invocation takes alike, leaves
/// no helper alone). Each read into floats holds
/// a sample that differs from each of the four texels nearest it ([`blended`]), which a sample
/// at the texels' centres never does (the nearest read here holds none), so the comparison is
/// of the filter's blends. Measured off Metal's grid of
/// 1/2048 on this host before the runtime's fix: the second sample of the '420v' 33 by 17
/// texture of no matrix read red 9916 into halves and 0.026306152 into floats, where Metal's
/// reads 9920 and 0.026367188
#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "MoltenVK: the runtime records pixel-sampler-scaling, a death in strict mode; Mac-only test, off everywhere; ledger: MoltenVK strict deaths"
)]
fn ycbcr8_420_textures_over_surfaces_read_as_metals() {
    use crate::gpu::ycbcr::measure::metal::{metals, set_value};
    use crate::gpu::ycbcr::measure::{
        Shape, float_bits, float_texels, probe, read_floats, sampled_floats, texels,
    };
    use crate::os::MATRIX_KEY;
    use reims_vgpu_metal::MetalPlatform;
    let d = dev();
    let p = crate::platform::tests::platform();
    let mp = MetalPlatform::new();
    for (ostype, size) in [(b"420v", (33, 17)), (b"420f", (64, 32))] {
        let shape = Shape::of(ostype);
        let matrices = [
            None,
            Some("ITU_R_601_4"),
            Some("ITU_R_709_2"),
            Some("ITU_R_2020"),
            Some("SMPTE_240M_1995"),
        ];
        for matrix in matrices {
            let case = format!("{} {size:?} {matrix:?}", String::from_utf8_lossy(ostype));
            let s = shape.surface(size);
            shape.fill_noise(&s, size, 7);
            if let Some(m) = matrix {
                set_value(&s, MATRIX_KEY, m);
            }
            let t = ycbcr8_420_on(&d, &s, 0, size).unwrap().unwrap();
            let read = read_floats::<VulkanPlatform>(&d, &t, size);
            let centres = probe((0.5, 0.5), 1.0, size, false, 0);
            let nearest = (0, 0, false, 0);
            let centred = sampled_floats::<VulkanPlatform>(&d, &t, nearest, size, &centres);
            assert_eq!(read, centred, "{case}");
            let (_, middles) = positions((size, (0.5, 0.5), 1.0), false);
            assert_eq!(blended(&read, size, &middles, &centred), 0, "{case}");
            let ours = linear_samples(&p, &d, &t, size, true);
            if (ostype, size, matrix) == (b"420v", (33, 17), None) {
                let [halves, floats] = [0, 1].map(|i| &ours[i].1);
                assert_eq!(
                    (texels(halves)[1][0], float_texels(floats)[1][0]),
                    (9920, 0.026_367_188),
                    "{case}"
                );
            }
            for (how, o, at) in ours.iter().filter(|(how, ..)| how.contains("float")) {
                let blends = blended(&read, size, at, &float_texels(o));
                assert!(blends > 0, "{how} reads no blend of texels: {case}");
            }
            let Some((md, mt)) = metals(500, &s, 0, size) else {
                continue;
            };
            let theirs = read_floats::<MetalPlatform>(&md, &mt, size);
            assert_eq!(float_bits(&read), float_bits(&theirs), "{case}");
            let theirs = sampled_floats::<MetalPlatform>(&md, &mt, nearest, size, &centres);
            assert_eq!(float_bits(&centred), float_bits(&theirs), "nearest: {case}");
            let theirs = linear_samples(&mp, &md, &mt, size, true);
            assert_eq!(ours.len(), theirs.len(), "{case}");
            for ((how, o, _), (_, m, _)) in ours.iter().zip(&theirs) {
                let texel = if how.contains("float") { 16 } else { 8 };
                let differ = o.chunks(texel).zip(m.chunks(texel)).filter(|(a, b)| a != b);
                assert_eq!(differ.count(), 0, "{how}: {case}");
            }
        }
    }
}

/// A linear sample of a `YCBCR8_420_2P` texture made with a swizzle that moves its channels
/// (blue, green, red, alpha), which takes the device's filter, not Metal's exact filter of the
/// format, dies at the dispatch or the draw that binds it, on the encoding thread, naming the
/// runtime's notice and the format: in a kernel and in a fragment function, into floats
/// `YCbCrFilterNotExact` ([`crate::answer::DIES`]), and into halves `HalfTexelConversion`, which
/// the runtime announces first there (its translation rounds the device's `f32` texel to `half`,
/// where Metal rounds a normalized one onto its grid first) and which dies on MoltenVK too. A
/// draw's `VkPipeline` is made on the runtime's pipeline pool, after the draw returns; the death
/// is the draw's all the same. The device's shader cache is its own and empty
/// (`device_with_own_cache`): on one that kept the draw's key from an earlier process, the
/// state's creation would make it instead, and the draw would pool nothing.
/// Metal makes the texture
#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "MoltenVK: the runtime records pixel-sampler-scaling, a death in strict mode; Mac-only test, off everywhere; ledger: MoltenVK strict deaths"
)]
fn ycbcr8_420_samples_the_exact_filter_does_not_reach_die() {
    use crate::gpu::ycbcr::measure::probed_texels;
    use crate::gpu::ycbcr::measure::{Shape, desc, drawn};
    use reims_vgpu_core::platform::Platform as _;
    use reims_vgpu_core::platform::compute::ComputeCmd;
    use reims_vgpu_core::platform::render::RenderCmd;
    use reims_vgpu_core::platform::types::{MtlSize, Swizzle};
    use reims_vgpu_metal::MetalPlatform;
    let (d, _cache) = device_with_own_cache();
    let p = crate::platform::tests::platform();
    let size = (33, 17);
    let shape = Shape::of(b"420v");
    let s = shape.surface(size);
    shape.fill_noise(&s, size, 7);
    let mut swizzled = desc(500, size);
    swizzled.swizzle = Some(Swizzle {
        red: 4,
        green: 3,
        blue: 2,
        alpha: 5,
    });
    if let Some(md) = MetalPlatform::system_default_device() {
        let metals = reims_vgpu_metal::gpu::desc::texture_descriptor(&swizzled);
        let made = md.new_texture_on_surface(&metals, Some(&s), 0).unwrap();
        assert!(made.is_some(), "Metal makes it");
    }
    let t = d
        .new_texture_on_surface(&p.texture_descriptor(&swizzled), Some(&s), 0)
        .unwrap()
        .unwrap();
    let sampler = linear::<VulkanPlatform>(&d, false, false);
    let (at, _) = positions(between(size), false);
    let dispatch = ComputeCmd::<VulkanPlatform>::DispatchThreads {
        threads_per_grid: MtlSize::new(1, 1, 1),
        threads_per_threadgroup: MtlSize::new(1, 1, 1),
    }
    .selector();
    let draw = RenderCmd::<VulkanPlatform>::DrawPrimitives {
        primitive_type: 3,
        vertex_start: 0,
        vertex_count: 3,
    }
    .selector();
    let (half, float) = ("half-texel-conversion", "ycbcr-filter-not-exact");
    for (function, texel, sel, kind) in [
        ("sample_texels", 8, dispatch, half),
        ("sample_texels_float", 16, dispatch, float),
        ("sample_fragment", 8, draw, half),
        ("sample_fragment_float", 16, draw, float),
    ] {
        *crate::answer::LAST_LINE.lock() = None;
        let pooled = d.raw.work_counts().pipelines_pooled;
        let sample = || match sel == draw {
            true => drawn::<VulkanPlatform>(&p, &d, (function, texel), &t, size, &sampler, &at),
            false => {
                let (s, at) = (Some(&sampler), Some(&at[..]));
                probed_texels::<VulkanPlatform>(&d, (function, texel), &t, size, s, at)
            }
        };
        let site = catch_death(sample).unwrap_err();
        assert_eq!(
            (site.kind, site.file),
            (DeathKind::Abort, sel),
            "{function}"
        );
        let pooled = d.raw.work_counts().pipelines_pooled - pooled;
        assert_eq!(
            pooled,
            u64::from(sel == draw),
            "{function}: pipelines pooled"
        );
        let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
        let named = format!("(metal2vulkan-degrade-{kind})");
        assert!(
            line.contains(&named) && line.contains("texture 0 bound as YCBCR84202P"),
            "{function}: {line}"
        );
    }
}

/// A 'b3a8' surface of [`b3a8_spec`] made over fresh pages of the host's
/// (`kIOSurfaceClientAddressRanges`), read-only for the GPU (`kIOSurfaceClientDirection` 2,
/// as core makes one over the guest's pages), every byte `noise(seed)`'s, and those pages:
/// their first byte and length, which the caller frees once the surface is gone.
fn b3a8_read_only(
    size: (u64, u64),
    rows: (u64, u64),
    seed: u64,
) -> (crate::os::OsSurface, reims_vgpu_core::host::HostPtr, u64) {
    use reims_vgpu_core::host::PhysicalMemoryRange;
    use reims_vgpu_core::platform::os::vm::HostVm as _;
    let vm = crate::os::OsVm::new();
    let page = vm.page_size();
    let mut spec = b3a8_spec(size, rows);
    let length = spec.alloc_size.next_multiple_of(page);
    let pages = vm.allocate_anywhere(length).unwrap();
    let first = u64::try_from(pages.addr()).unwrap();
    spec.alloc_size = length;
    spec.client_direction = Some(2);
    spec.client_address_ranges = Some(
        (0..length / page)
            .map(|i| PhysicalMemoryRange {
                physical_address: first + i * page,
                physical_length: page,
            })
            .collect(),
    );
    let s = noisy_surface(&spec, seed);
    assert!(crate::os::gpu_read_only(&s));
    (s, pages, length)
}

/// A 2D descriptor of `format` of `size`, one level, usage `ShaderRead`, of the resource
/// options `options` (0 `Shared`, 0x10 `Managed`).
fn xr10_desc(
    format: u64,
    (width, height): (u64, u64),
    options: u64,
) -> reims_vgpu_core::platform::values::TextureDesc {
    reims_vgpu_core::platform::values::TextureDesc {
        texture_type: 2,
        pixel_format: format,
        width,
        height,
        usage: 1,
        resource_options: Some(options),
        ..reims_vgpu_core::platform::values::TextureDesc::default()
    }
}

/// A texture of [`xr10_desc`] over plane `plane` of `surface`, made by this platform.
fn xr10_over(
    format: u64,
    surface: &crate::os::OsSurface,
    plane: u64,
    (size, options): ((u64, u64), u64),
) -> RaiseResult<Option<VulkanTexture>> {
    use reims_vgpu_core::platform::Platform as _;
    let p = crate::platform::tests::platform();
    let desc = p.texture_descriptor(&xr10_desc(format, size, options));
    dev().new_texture_on_surface(&desc, Some(surface), plane)
}

/// Metal's texture of [`xr10_desc`] over plane `plane` of `surface`, and its device; `None`
/// with no Metal device or where Metal answers nil.
fn xr10_metals(
    format: u64,
    surface: &crate::os::OsSurface,
    plane: u64,
    (size, options): ((u64, u64), u64),
) -> Option<(
    <reims_vgpu_metal::MetalPlatform as reims_vgpu_core::platform::Platform>::Device,
    <reims_vgpu_metal::MetalPlatform as reims_vgpu_core::platform::Platform>::Texture,
)> {
    let md = reims_vgpu_metal::MetalPlatform::system_default_device()?;
    let desc = reims_vgpu_metal::gpu::desc::texture_descriptor(&xr10_desc(format, size, options));
    let t = md
        .new_texture_on_surface(&desc, Some(surface), plane)
        .unwrap()?;
    Some((md, t))
}

/// A texture of a two-plane XR10 format (550, 551) over plane 0 of a 'b3a8' surface, its
/// colour words and alpha bytes noise, reads as Metal's does, bit for bit, into floats:
/// `read()` of every texel, and nearest and linear samples a third of a texel apart from two
/// texels before it to two past it, clamped to the edge in texel coordinates and repeated in
/// normalized ones, by a reader holding no gather (the runtime gathers none of them). At the
/// shape a guest's mapper makes (E26d's: 384 by 256, `Managed`,
/// plane 1's rows 512 bytes apart), `Shared` at that size, and at odd sizes
#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "MoltenVK: the runtime records device-pack-conversion, a death in strict mode; Mac-only test, off everywhere; ledger: MoltenVK strict deaths"
)]
fn two_plane_xr10_textures_over_surfaces_read_as_metals() {
    use crate::gpu::ycbcr::measure::{probe, read_floats, sampled_only_floats};
    use reims_vgpu_core::platform::resource::Texture as _;
    use reims_vgpu_metal::MetalPlatform;
    let bits = |texels: Vec<[f32; 4]>| -> Vec<[u32; 4]> {
        texels.into_iter().map(|t| t.map(f32::to_bits)).collect()
    };
    if MetalPlatform::system_default_device().is_none() {
        return;
    }
    let d = dev();
    for format in [550, 551] {
        for (size, rows, options, seed) in [
            ((384, 256), (1536, 512), 0x10, 3),
            ((384, 256), (1536, 512), 0, 4),
            ((33, 17), (192, 64), 0x10, 5),
            ((7, 5), (64, 64), 0, 6),
        ] {
            let case = format!("{format} {size:?} options {options:#x}");
            let s = b3a8(size, rows, seed);
            let t = xr10_over(format, &s, 0, (size, options)).unwrap().unwrap();
            assert_eq!(t.pixel_format(), format, "{case}");
            let Some((md, mt)) = xr10_metals(format, &s, 0, (size, options)) else {
                panic!("Metal answers nil: {case}");
            };
            let ours = bits(read_floats::<VulkanPlatform>(&d, &t, size));
            let theirs = bits(read_floats::<MetalPlatform>(&md, &mt, size));
            assert_eq!(ours, theirs, "read: {case}");
            let grid = ((size.0 + 4) * 3, (size.1 + 4) * 3);
            for filter in [0, 1] {
                for (address, normalized) in [(0, false), (2, true)] {
                    let state = (filter, address, normalized, 0);
                    let at = probe((-2.0, -2.0), 1.0 / 3.0, grid, normalized, 0);
                    let ours = sampled_only_floats::<VulkanPlatform>(&d, &t, state, grid, &at);
                    let theirs = sampled_only_floats::<MetalPlatform>(&md, &mt, state, grid, &at);
                    assert_eq!(bits(ours), bits(theirs), "{state:?}: {case}");
                }
            }
        }
    }
}

/// A texture of a two-plane XR10 format (550, 551) over plane 0 of a 'b3a8' surface whose
/// plane 1 lies at an offset not a multiple of 16 bytes (8 past the end of plane 0), its rows
/// multiples of 16, reads as Metal's does, bit for bit, into floats (`read()` of every texel,
/// and nearest and linear samples between texels), as at aligned planes
/// ([`two_plane_xr10_textures_over_surfaces_read_as_metals`]). One whose colour rows are not
/// a multiple of 16 bytes (132 or 136, alpha rows of 36 or 40) dies where it is made, naming
/// the runtime's refusal (`OverlappingSurfaceRows`), where Metal makes it
#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "MoltenVK: the runtime records device-pack-conversion, a death in strict mode; Mac-only test, off everywhere; ledger: MoltenVK strict deaths"
)]
fn two_plane_xr10_textures_over_unaligned_planes_are_metals() {
    use crate::gpu::ycbcr::measure::{probe, read_floats, sampled_only_floats};
    use reims_vgpu_core::platform::resource::Texture as _;
    use reims_vgpu_metal::MetalPlatform;
    let bits = |texels: Vec<[f32; 4]>| -> Vec<[u32; 4]> {
        texels.into_iter().map(|t| t.map(f32::to_bits)).collect()
    };
    if MetalPlatform::system_default_device().is_none() {
        return;
    }
    let d = dev();
    let size = (33, 17);
    for format in [550, 551] {
        for (rows, seed) in [((132, 36), 7), ((136, 40), 8)] {
            let case = format!("{format} rows {rows:?}");
            let s = noisy_surface(&b3a8_spec(size, rows), seed);
            *crate::answer::LAST_LINE.lock() = None;
            let site = catch_death(|| xr10_over(format, &s, 0, (size, 0x10))).unwrap_err();
            let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
            let sel = "newTextureWithDescriptor:iosurface:plane:";
            assert_eq!((site.kind, site.file), (DeathKind::Abort, sel), "{case}");
            assert!(line.contains("OverlappingSurfaceRows"), "{case}: {line}");
            let made = xr10_metals(format, &s, 0, (size, 0x10));
            assert!(made.is_some(), "Metal makes it: {case}");
        }
        for (rows, alpha_offset, seed) in
            [((192, 64), 192 * 17 + 8, 9), ((144, 48), 144 * 17 + 4, 10)]
        {
            let case = format!("{format} rows {rows:?} plane 1 at {alpha_offset}");
            let s = noisy_surface(&b3a8_spec_at(size, rows, alpha_offset), seed);
            let Some((md, mt)) = xr10_metals(format, &s, 0, (size, 0x10)) else {
                panic!("Metal answers nil: {case}");
            };
            let t = xr10_over(format, &s, 0, (size, 0x10)).unwrap().unwrap();
            assert_eq!(t.pixel_format(), format, "{case}");
            let ours = bits(read_floats::<VulkanPlatform>(&d, &t, size));
            let theirs = bits(read_floats::<MetalPlatform>(&md, &mt, size));
            assert_eq!(ours, theirs, "read: {case}");
            let grid = ((size.0 + 4) * 3, (size.1 + 4) * 3);
            for filter in [0, 1] {
                let state = (filter, 0, false, 0);
                let at = probe((-2.0, -2.0), 1.0 / 3.0, grid, false, 0);
                let ours = sampled_only_floats::<VulkanPlatform>(&d, &t, state, grid, &at);
                let theirs = sampled_only_floats::<MetalPlatform>(&md, &mt, state, grid, &at);
                assert_eq!(bits(ours), bits(theirs), "{state:?}: {case}");
            }
        }
    }
}

/// A texture of a two-plane XR10 format (550, 551) the runtime refuses dies where it is made,
/// naming the refusal (`Format`), where Metal makes it: over plane 1 of a 'b3a8' surface,
/// over plane 0 of a one-plane 'BGRA' surface, over plane 0 of a two-plane surface of another
/// type ('420v'), and rotated (framebuffer only) over plane 0 of a 'b3a8' surface
#[test]
fn two_plane_xr10_textures_the_runtime_refuses_die() {
    use reims_vgpu_core::platform::Platform as _;
    let size = (16, 16);
    let two = b3a8(size, (64, 64), 5);
    let one = crate::os::tests::bgra_surface(16, 16);
    let other = crate::gpu::ycbcr::measure::Shape::of(b"420v").surface(size);
    let sel = "newTextureWithDescriptor:iosurface:plane:";
    let refused = |case: &str, made: &dyn Fn() -> RaiseResult<Option<VulkanTexture>>| {
        *crate::answer::LAST_LINE.lock() = None;
        let site = catch_death(made).unwrap_err();
        let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
        assert_eq!(
            (site.kind, site.file),
            (DeathKind::Abort, sel),
            "{case}: {line}"
        );
        assert!(line.contains("Format"), "{case}: {line}");
    };
    let metal = reims_vgpu_metal::MetalPlatform::system_default_device();
    for format in [550, 551] {
        for (surface, plane, case) in [(&two, 1, "plane 1"), (&one, 0, "BGRA"), (&other, 0, "420v")]
        {
            let case = format!("{format} {case}");
            refused(&case, &|| xr10_over(format, surface, plane, (size, 0x10)));
            if metal.is_some() {
                let made = xr10_metals(format, surface, plane, (size, 0x10));
                assert!(made.is_some(), "Metal makes it: {case}");
            }
        }
        // Metal aborts at a rotation of a descriptor not framebufferOnly.
        let rotated = reims_vgpu_core::platform::values::TextureDesc {
            usage: 0x5,
            framebuffer_only: Some(true),
            rotation: Some(1),
            ..xr10_desc(format, size, 0x10)
        };
        let case = format!("{format} rotated");
        let p = crate::platform::tests::platform();
        refused(&case, &|| {
            dev().new_texture_on_surface(&p.texture_descriptor(&rotated), Some(&two), 0)
        });
        if let Some(md) = &metal {
            let desc = reims_vgpu_metal::gpu::desc::texture_descriptor(&rotated);
            let made = md.new_texture_on_surface(&desc, Some(&two), 0).unwrap();
            assert!(made.is_some(), "Metal makes it: {case}");
        }
    }
}

/// The bytes of a texture of `format` over plane 0 of `surface`, of [`xr10_desc`] with
/// `RenderTarget` usage too (E26d's 0x5), after a pass of platform `p` that clears it to
/// `color` and stores it: the surface's bytes from its base address to `length`, once the
/// command buffer completed.
fn cleared<P: reims_vgpu_core::platform::Platform<Surface = crate::os::OsSurface>>(
    p: &P,
    d: &P::Device,
    format: u64,
    surface: &crate::os::OsSurface,
    (size, options): ((u64, u64), u64),
    color: reims_vgpu_core::platform::types::ClearColor,
    length: u64,
) -> Vec<u8> {
    use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
    use reims_vgpu_core::platform::descriptor::RenderPassDescriptor as _;
    use reims_vgpu_core::platform::render::{RenderCmd, RenderEncoder as _};
    use reims_vgpu_core::platform::values::ColorAttachmentSet;
    let mut desc = xr10_desc(format, size, options);
    desc.usage = 0x5;
    let t = d
        .new_texture_on_surface(&p.texture_descriptor(&desc), Some(surface), 0)
        .unwrap()
        .unwrap();
    let pass = p.render_pass_descriptor();
    pass.set_color_attachment(
        OnFault::Die,
        0,
        &ColorAttachmentSet {
            texture: Some(Some(&t)),
            load_action: Some(2),
            store_action: Some(1),
            clear_color: Some(color),
            ..ColorAttachmentSet::default()
        },
    )
    .unwrap();
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
    e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    assert_eq!(cb.error(), None);
    use reims_vgpu_core::platform::os::surface::Surface as _;
    let base = surface.base_address().as_ptr::<u8>();
    let length = usize::try_from(length).unwrap();
    // SAFETY: the surface's bytes, read after the command buffer that wrote them completed.
    unsafe { std::slice::from_raw_parts(base, length) }.to_vec()
}

/// A pass that clears a texture of a two-plane XR10 format (550, 551) over plane 0 of a
/// 'b3a8' surface, and stores it, writes the surface's two planes as Metal's does, byte for
/// byte, the bytes between the rows and past the planes kept: at E26d's shape and at an odd
/// size, to a colour in and past 0 to 1
#[test]
fn a_cleared_two_plane_xr10_texture_over_a_surface_writes_as_metals() {
    use reims_vgpu_core::platform::types::ClearColor;
    let Some(md) = reims_vgpu_metal::MetalPlatform::system_default_device() else {
        return;
    };
    let p = crate::platform::tests::platform();
    let mp = reims_vgpu_metal::MetalPlatform::new();
    let d = dev();
    let color = ClearColor {
        red: 0.25,
        green: -0.5,
        blue: 1.25,
        alpha: 0.75,
    };
    for format in [550, 551] {
        for (size, rows, options, seed) in [
            ((384, 256), (1536, 512), 0x10, 7),
            ((33, 17), (192, 64), 0, 8),
        ] {
            let case = format!("{format} {size:?} options {options:#x}");
            let (rows, size): ((u64, u64), (u64, u64)) = (rows, size);
            let length = (rows.0 * size.1).next_multiple_of(4096) + rows.1 * size.1;
            let (ours, theirs) = (b3a8(size, rows, seed), b3a8(size, rows, seed));
            let ours = cleared(&p, &d, format, &ours, (size, options), color, length);
            let theirs = cleared(&mp, &md, format, &theirs, (size, options), color, length);
            let differ = ours.iter().zip(&theirs).filter(|(a, b)| a != b).count();
            assert_eq!(differ, 0, "{case}");
        }
    }
}

/// What a fill writes: bytes, or a colour packed in a pixel format.
#[derive(Clone, Copy, Debug)]
enum Pattern<'a> {
    /// `fillTexture:level:slice:region:bytes:length:`.
    Bytes(&'a [u8]),
    /// `fillTexture:level:slice:region:color:pixelFormat:`.
    Color(reims_vgpu_core::platform::types::ClearColor, u64),
}

/// What a fill of `region` (its origin and size) of a texture of `format` over plane 0 of
/// `surface` (`size`, `options`) with `pattern` writes on `p`'s device `d`: the surface's
/// bytes from its base address to `length`, once the command buffer completed.
#[allow(clippy::too_many_arguments)]
fn filled<P: reims_vgpu_core::platform::Platform<Surface = crate::os::OsSurface>>(
    p: &P,
    d: &P::Device,
    format: u64,
    surface: &crate::os::OsSurface,
    (size, options): ((u64, u64), u64),
    ([x, y], [w, h]): ([u64; 2], [u64; 2]),
    pattern: Pattern<'_>,
    length: u64,
) -> Vec<u8> {
    use reims_vgpu_core::platform::HostRange;
    use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder as _};
    use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
    use reims_vgpu_core::platform::os::surface::Surface as _;
    use reims_vgpu_core::platform::types::{MtlOrigin, MtlRegion, MtlSize};
    let desc = p.texture_descriptor(&xr10_desc(format, size, options));
    let t = d
        .new_texture_on_surface(&desc, Some(surface), 0)
        .unwrap()
        .unwrap();
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let e = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
    let region = MtlRegion::new(MtlOrigin::new(x, y, 0), MtlSize::new(w, h, 1));
    let fill = match pattern {
        Pattern::Bytes(bytes) => BlitCmd::FillTexture {
            texture: &t,
            level: 0,
            slice: 0,
            region,
            bytes: HostRange::of_slice(bytes),
        },
        Pattern::Color(color, pixel_format) => BlitCmd::FillTextureColor {
            texture: &t,
            level: 0,
            slice: 0,
            region,
            color,
            pixel_format,
        },
    };
    e.encode(OnFault::Die, fill).unwrap();
    e.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    assert_eq!(cb.error(), None);
    let base = surface.base_address().as_ptr::<u8>();
    let length = usize::try_from(length).unwrap();
    // SAFETY: the surface's bytes, read after the command buffer that wrote them completed.
    unsafe { std::slice::from_raw_parts(base, length) }.to_vec()
}

/// A fill of a texture of a two-plane XR10 format (550, 551) over plane 0 of a 'b3a8' surface
/// writes the surface's two planes as Metal's does, byte for byte, the bytes outside the
/// region kept: with bytes, 8 of them and 3, which repeat, and with a colour packed in
/// `RGBA8Unorm` and in `RGBA16Float`; over a region a multiple of 4 wide, one of an odd
/// width, and one past the texture's edge, which is cut there; at E26d's shape and at an odd
/// size
#[test]
fn a_filled_two_plane_xr10_texture_over_a_surface_writes_as_metals() {
    use reims_vgpu_core::platform::types::ClearColor;
    let Some(md) = reims_vgpu_metal::MetalPlatform::system_default_device() else {
        return;
    };
    let p = crate::platform::tests::platform();
    let mp = reims_vgpu_metal::MetalPlatform::new();
    let d = dev();
    let color = ClearColor {
        red: 0.25,
        green: -0.5,
        blue: 1.25,
        alpha: 0.75,
    };
    let patterns = [
        Pattern::Bytes(&[0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88]),
        Pattern::Bytes(&[0xa1, 0xb2, 0xc3]),
        Pattern::Color(color, 70),
        Pattern::Color(color, 115),
    ];
    let regions = [([4, 2], [16, 5]), ([3, 1], [7, 3]), ([28, 14], [16, 8])];
    for format in [550, 551] {
        for (size, rows, options, seed) in [
            ((384, 256), (1536, 512), 0x10, 9),
            ((33, 17), (192, 64), 0, 10),
        ] {
            let (rows, size): ((u64, u64), (u64, u64)) = (rows, size);
            let length = (rows.0 * size.1).next_multiple_of(4096) + rows.1 * size.1;
            for pattern in patterns {
                for region in regions {
                    let case =
                        format!("{format} {size:?} options {options:#x} {pattern:?} {region:?}");
                    let (ours, theirs) = (b3a8(size, rows, seed), b3a8(size, rows, seed));
                    let shape = (size, options);
                    let ours = filled(&p, &d, format, &ours, shape, region, pattern, length);
                    let theirs = filled(&mp, &md, format, &theirs, shape, region, pattern, length);
                    let differ = ours.iter().zip(&theirs).filter(|(a, b)| a != b).count();
                    assert_eq!(differ, 0, "{case}");
                }
            }
        }
    }
}

/// A fill of [`memory_fill_on`]: its level and slice, its region's origin and size, and its
/// pattern.
type MemoryFill<'a> = ((u64, u64), [u64; 2], [u64; 2], Pattern<'a>);

/// Where a texture over memory lies: over plane `plane` of a surface of the
/// OSType ([`crate::gpu::ycbcr::measure::Shape`]) whose luma is the texture's size (over plane 1,
/// the texture is the chroma's size), or over a Shared buffer at rows of `row` bytes.
#[derive(Clone, Copy, Debug)]
enum Memory {
    /// A surface's plane.
    Plane(&'static [u8; 4], u64),
    /// A buffer's bytes, `row` a row.
    Buffer(u64),
}

impl Memory {
    /// The size of a texture over this memory whose surface's luma, or which, is `size`.
    fn texture_size(self, size: (u64, u64)) -> (u64, u64) {
        match self {
            Memory::Plane(ostype, 1) => {
                crate::gpu::ycbcr::measure::Shape::of(ostype).chroma_size(size)
            }
            _ => size,
        }
    }
}

/// What `fills`, each a level, a slice, a region (its origin and size) and a pattern, in one
/// blit encoder, of a texture of `format` over `memory` (a surface's of luma `size`, or a
/// buffer's rows of `size`) write on `p`'s device `d`: every byte of the memory, which starts as
/// `i * 37 % 251` at byte `i` on every platform, once the command buffer completed.
fn memory_fill_on<P: reims_vgpu_core::platform::Platform<Surface = crate::os::OsSurface>>(
    p: &P,
    d: &P::Device,
    (format, size, memory): (u64, (u64, u64), Memory),
    fills: &[MemoryFill<'_>],
) -> Vec<u8> {
    use reims_vgpu_core::platform::HostRange;
    use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder as _};
    use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
    use reims_vgpu_core::platform::os::surface::Surface as _;
    use reims_vgpu_core::platform::resource::Buffer as _;
    use reims_vgpu_core::platform::types::{MtlOrigin, MtlRegion, MtlSize};
    let texture_size = memory.texture_size(size);
    let desc = p.texture_descriptor(&crate::gpu::ycbcr::measure::desc(format, texture_size));
    let write = |to: *mut u8, length: u64| {
        for i in 0..length {
            // SAFETY: `to` holds `length` bytes, which no command uses yet.
            unsafe {
                to.add(usize::try_from(i).unwrap())
                    .write((i * 37 % 251) as u8)
            };
        }
    };
    let (surface, buffer, t, length);
    match memory {
        Memory::Plane(ostype, plane) => {
            let made = crate::gpu::ycbcr::measure::Shape::of(ostype).surface(size);
            length = made.alloc_size();
            write(made.base_address().as_ptr::<u8>(), length);
            t = d.new_texture_on_surface(&desc, Some(&made), plane);
            (surface, buffer) = (Some(made), None);
        }
        Memory::Buffer(row) => {
            length = row * size.1;
            let made = d
                .new_buffer_of_length(OnFault::Die, length, 0)
                .unwrap()
                .unwrap();
            write(made.contents().as_ptr::<u8>(), length);
            t = made.new_texture(OnFault::Die, &desc, 0, row);
            (surface, buffer) = (None, Some(made));
        }
    }
    let t = t.unwrap().unwrap();
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let e = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
    for &((level, slice), [x, y], [w, h], pattern) in fills {
        let region = MtlRegion::new(MtlOrigin::new(x, y, 0), MtlSize::new(w, h, 1));
        let fill = match pattern {
            Pattern::Bytes(bytes) => BlitCmd::FillTexture {
                texture: &t,
                level,
                slice,
                region,
                bytes: HostRange::of_slice(bytes),
            },
            Pattern::Color(color, pixel_format) => BlitCmd::FillTextureColor {
                texture: &t,
                level,
                slice,
                region,
                color,
                pixel_format,
            },
        };
        e.encode(OnFault::Die, fill).unwrap();
    }
    e.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    assert_eq!(cb.error(), None);
    let base = match (&surface, &buffer) {
        (Some(s), _) => s.base_address(),
        (_, Some(b)) => b.contents(),
        _ => unreachable!(),
    };
    let length = usize::try_from(length).unwrap();
    // SAFETY: the memory's bytes, read after the command buffer that wrote them completed.
    unsafe { std::slice::from_raw_parts(base.as_ptr::<u8>(), length) }.to_vec()
}

/// The textures [`fills_of_ycbcr_and_padded_textures_are_metals`] fills, of `size`: of each
/// YCbCr format the runtime makes over a surface, over plane 0 of a surface of its bits, packing
/// and chroma; and of each padded format over a buffer and over a plane of an 'x420' surface
/// (576 and 577 the luma, 578 and 579 the chroma).
fn memory_fill_textures(size: (u64, u64)) -> Vec<(u64, (u64, u64), Memory)> {
    use metal2vulkan::metal::format::{ChromaSubsampling as C, ycbcr_format};
    let mut out = Vec::new();
    for raw in 0..2048u32 {
        let Ok(pf) = metal2vulkan::metal::PixelFormat::try_from(raw) else {
            continue;
        };
        if !crate::gpu::ycbcr::runtime_makes(pf) {
            continue;
        }
        let f = ycbcr_format(pf).unwrap();
        let ostype: &'static [u8; 4] = match (f.bits, f.packed, f.subsampling) {
            (8, _, C::Chroma420) => b"420v",
            (8, _, C::Chroma422) => b"422v",
            (8, _, C::Chroma444) => b"444v",
            (_, false, C::Chroma420) => b"x420",
            (_, false, C::Chroma422) => b"x422",
            (_, false, C::Chroma444) => b"x444",
            (_, true, C::Chroma420) => b"p420",
            (_, true, C::Chroma422) => b"p422",
            (_, true, C::Chroma444) => b"p444",
            other => panic!("no surface for {other:?}"),
        };
        out.push((u64::from(raw), size, Memory::Plane(ostype, 0)));
    }
    for format in [576, 577, 578, 579] {
        out.push((format, size, Memory::Buffer(256)));
        let plane = if format < 578 { 0 } else { 1 };
        out.push((format, size, Memory::Plane(b"x420", plane)));
    }
    out
}

/// A fill of a texture Metal makes over memory it is given, of a YCbCr or padded format, writes the
/// memory as Metal's does, every byte compared ([`memory_fill_on`]): of each YCbCr format the
/// runtime makes over a surface, over plane 0 of a surface of its kind, and of each padded format
/// over a buffer and over a plane of an 'x420' surface ([`memory_fill_textures`]); at two sizes;
/// with bytes, 8 of them, 3 and 1, and with a colour packed in RGBA8Unorm, RGBA16Float, R8Unorm,
/// RG11B10Float and RGBA32Float; over a region inside the texture, one of an odd width and one past
/// its edge at the smaller size (a texture over memory has one level and one slice). Measured on
/// the M3 Max (macOS 27.0.1): Metal writes each texel of the region (cut at the texture) at plane
/// offset + y * bytesPerRow + x * the block bytes Metal describes (1, 2 or 4), the pattern repeated
/// from its start and cut at those bytes, every other byte kept; a packed format's texel (508 to
/// 510, 528 to 530: 1 byte) is written 0 whatever the pattern. A region wholly past the texture
/// writes nothing on either.
#[test]
fn fills_of_ycbcr_and_padded_textures_are_metals() {
    use reims_vgpu_core::platform::types::ClearColor;
    use reims_vgpu_core::testkit::capture::catch_death;
    let Some(md) = reims_vgpu_metal::MetalPlatform::system_default_device() else {
        return;
    };
    let p = crate::platform::tests::platform();
    let mp = reims_vgpu_metal::MetalPlatform::new();
    let d = dev();
    let color = ClearColor {
        red: 0.25,
        green: -0.5,
        blue: 1.25,
        alpha: 0.75,
    };
    let patterns = [
        Pattern::Bytes(&[0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88]),
        Pattern::Bytes(&[0xa1, 0xb2, 0xc3]),
        Pattern::Bytes(&[0x5a]),
        Pattern::Color(color, 70),
        Pattern::Color(color, 115),
        Pattern::Color(color, 10),
        Pattern::Color(color, 92),
        Pattern::Color(color, 125),
    ];
    let regions = [([1, 1], [5, 2]), ([3, 2], [7, 3]), ([12, 6], [8, 4])];
    let mut differ = Vec::new();
    for size in [(16, 8), (33, 17)] {
        for texture in memory_fill_textures(size) {
            for pattern in patterns {
                for &(origin, region) in &regions {
                    let fills: [MemoryFill<'_>; 1] = [((0, 0), origin, region, pattern)];
                    let what = format!("{texture:?} {fills:?}");
                    let theirs = memory_fill_on(&mp, &md, texture, &fills);
                    let ours = catch_death(|| memory_fill_on(&p, &d, texture, &fills));
                    match ours {
                        Ok(ours) if ours == theirs => {}
                        Ok(ours) => {
                            let n = ours.iter().zip(&theirs).filter(|(a, b)| a != b).count();
                            differ.push(format!("{what}: {n} bytes differ"));
                        }
                        Err(site) => {
                            let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
                            differ.push(format!("{what}: died at {}: {line}", site.file));
                        }
                    }
                }
            }
        }
    }
    assert!(
        differ.is_empty(),
        "{} differ:\n{}",
        differ.len(),
        differ.join("\n")
    );
}

/// Runs `cmd` in a blit encoder of a command buffer of `d`'s queue `q` of its own, and waits
/// for it.
fn blit_one<P: reims_vgpu_core::platform::Platform>(
    q: &P::CommandQueue,
    cmd: reims_vgpu_core::platform::blit::BlitCmd<'_, P>,
) {
    use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder as _};
    use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let e = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
    e.encode(OnFault::Die, cmd).unwrap();
    e.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    assert_eq!(cb.error(), None);
}

/// A copy of `size` texels from `from` at `at` into `to` at `to_at`.
fn copy_region<'a, P: reims_vgpu_core::platform::Platform>(
    (from, at): (&'a P::Texture, [u64; 2]),
    (to, to_at): (&'a P::Texture, [u64; 2]),
    [w, h]: [u64; 2],
) -> reims_vgpu_core::platform::blit::BlitCmd<'a, P> {
    use reims_vgpu_core::platform::types::{MtlOrigin, MtlSize};
    reims_vgpu_core::platform::blit::BlitCmd::CopyTextureRegion {
        source_texture: from,
        source_slice: 0,
        source_level: 0,
        source_origin: MtlOrigin::new(at[0], at[1], 0),
        source_size: MtlSize::new(w, h, 1),
        destination_texture: to,
        destination_slice: 0,
        destination_level: 0,
        destination_origin: MtlOrigin::new(to_at[0], to_at[1], 0),
    }
}

/// The bytes of `surface` from its base address to `length`.
fn surface_bytes(surface: &crate::os::OsSurface, length: u64) -> Vec<u8> {
    use reims_vgpu_core::platform::os::surface::Surface as _;
    let base = surface.base_address().as_ptr::<u8>();
    let length = usize::try_from(length).unwrap();
    // SAFETY: the surface's bytes, read after the command buffers that wrote them completed.
    unsafe { std::slice::from_raw_parts(base, length) }.to_vec()
}

/// What copies between textures of `format` (`size`, `Managed`) over plane 0 of 'b3a8'
/// surfaces `a` and `b`, and a `Shared` texture of `format` as wide as the most columns of
/// `size` a multiple of 4 holds, write on `p`'s device `d`, each in a command buffer of its
/// own: the bytes of `b` (to `length`) after a copy of 8 by 5 texels from `a`'s (1, 2) to
/// `b`'s (4, 4), then the bytes of `b` after a copy of the top quarter of `a`'s texels that fit
/// into the `Shared` texture and of 8 by 4 of those from (2, 0) to `b`'s (12, 6).
///
/// Metal's blit addresses a texture's alpha plane by its colour rows too, four of the plane's
/// rows apart in the `Shared` texture, so a copy of more than its top quarter writes past its
/// alpha plane, and at E26d's shape one of more than a third of `a`'s rows reads past the
/// surface: bytes the runtime does not hold. Metal answers such a copy the same each time while
/// its process alone is on the GPU, but beside another process's work there those accesses
/// fault the GPU, and its recovery loses bytes inside the planes too, with no command buffer's
/// error: the alpha bytes the copy into `b` reads come back 0 (measured on the M3 Max, macOS
/// 27.0.1, Metal alone in each process: six processes copying all of `a`'s rows of 551 40
/// times each at once, two copied zeros 1 and 2 times and four lost a command buffer to the
/// recovery; copying the top quarter, none in 240 copies of each format; one process alone
/// copying all of them, none in 200). The top quarter holds every byte the copy into `b` reads.
fn surface_copies<P: reims_vgpu_core::platform::Platform<Surface = crate::os::OsSurface>>(
    p: &P,
    d: &P::Device,
    format: u64,
    (a, b): (&crate::os::OsSurface, &crate::os::OsSurface),
    size: (u64, u64),
    length: u64,
) -> [Vec<u8>; 2] {
    let over = p.texture_descriptor(&xr10_desc(format, size, 0x10));
    let ta = d
        .new_texture_on_surface(&over, Some(a), 0)
        .unwrap()
        .unwrap();
    let tb = d
        .new_texture_on_surface(&over, Some(b), 0)
        .unwrap()
        .unwrap();
    // Metal's blit rows of a texture of its own memory are `⌊4 w / 16⌋ · 16` bytes apart, its
    // planes' rows `4 w` and `w` rounded up to 16: at a width a multiple of 4 they hold every
    // column, and no copy of it overlaps its rows or reads their padding, whose bytes Metal's
    // blits keep.
    let columns = [size.0 / 4 * 4, size.1];
    let shared = xr10_desc(format, (columns[0], columns[1]), 0);
    let other = (d.new_texture(OnFault::Die, &p.texture_descriptor(&shared)))
        .unwrap()
        .unwrap();
    // At widths a multiple of 16 the `Shared` texture's alpha rows are `w` bytes apart and
    // Metal's blit rows `4 w`, so a quarter of its rows is what the alpha plane holds.
    assert_eq!(columns[0] % 16, 0, "{size:?}");
    let quarter = [columns[0], columns[1] / 4];
    let q = d.new_command_queue().unwrap();
    blit_one::<P>(&q, copy_region::<P>((&ta, [1, 2]), (&tb, [4, 4]), [8, 5]));
    let first = surface_bytes(b, length);
    blit_one::<P>(
        &q,
        copy_region::<P>((&ta, [0, 0]), (&other, [0, 0]), quarter),
    );
    blit_one::<P>(
        &q,
        copy_region::<P>((&other, [2, 0]), (&tb, [12, 6]), [8, 4]),
    );
    [first, surface_bytes(b, length)]
}

/// Copies between a texture of a two-plane XR10 format (550, 551) over plane 0 of a 'b3a8'
/// surface and another texture of the format write the surfaces' two planes as Metal's do,
/// byte for byte, the bytes outside the copied texels kept: from one surface's texture into
/// another's, out of one into a texture of its own (the rows Metal's blit keeps inside the
/// planes, [`surface_copies`]), and from that into a surface's; at E26d's shape and at an odd
/// size
#[test]
fn copies_between_two_plane_xr10_textures_over_surfaces_write_as_metals() {
    let Some(md) = reims_vgpu_metal::MetalPlatform::system_default_device() else {
        return;
    };
    let p = crate::platform::tests::platform();
    let mp = reims_vgpu_metal::MetalPlatform::new();
    let d = dev();
    for format in [550, 551] {
        for (size, rows, seed) in [((384, 256), (1536, 512), 11), ((33, 17), (192, 64), 13)] {
            let case = format!("{format} {size:?}");
            let (rows, size): ((u64, u64), (u64, u64)) = (rows, size);
            let length = (rows.0 * size.1).next_multiple_of(4096) + rows.1 * size.1;
            let pair = || (b3a8(size, rows, seed), b3a8(size, rows, seed + 1));
            let (ours, theirs) = (pair(), pair());
            let ours = surface_copies(&p, &d, format, (&ours.0, &ours.1), size, length);
            let theirs = surface_copies(&mp, &md, format, (&theirs.0, &theirs.1), size, length);
            for (step, (ours, theirs)) in ours.iter().zip(&theirs).enumerate() {
                let differ = ours.iter().zip(theirs).filter(|(a, b)| a != b).count();
                assert_eq!(differ, 0, "{case}: copy {step}");
            }
        }
    }
}

/// A copy within one texture of a two-plane XR10 format (550, 551) over plane 0 of a 'b3a8'
/// surface dies at the copy, naming the runtime's refusal (`Format`), where Metal copies
#[test]
fn a_copy_within_a_two_plane_xr10_texture_over_a_surface_dies() {
    use reims_vgpu_core::platform::Platform;
    fn within<P: Platform<Surface = crate::os::OsSurface>>(
        p: &P,
        d: &P::Device,
        format: u64,
        surface: &crate::os::OsSurface,
    ) {
        let over = p.texture_descriptor(&xr10_desc(format, (33, 17), 0x10));
        let t = d
            .new_texture_on_surface(&over, Some(surface), 0)
            .unwrap()
            .unwrap();
        let q = d.new_command_queue().unwrap();
        blit_one::<P>(&q, copy_region::<P>((&t, [0, 0]), (&t, [16, 8]), [8, 4]));
    }
    let p = crate::platform::tests::platform();
    let mp = reims_vgpu_metal::MetalPlatform::new();
    let md = reims_vgpu_metal::MetalPlatform::system_default_device();
    let d = dev();
    let sel = "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:\
               destinationSlice:destinationLevel:destinationOrigin:";
    for format in [550, 551] {
        let s = b3a8((33, 17), (192, 64), 15);
        *crate::answer::LAST_LINE.lock() = None;
        let site = catch_death(|| within(&p, &d, format, &s)).unwrap_err();
        let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
        assert_eq!(
            (site.kind, site.file),
            (DeathKind::Abort, sel),
            "{format}: {line}"
        );
        assert!(line.contains("Format"), "{format}: {line}");
        if let Some(md) = &md {
            within(&mp, md, format, &b3a8((33, 17), (192, 64), 15));
        }
    }
}

/// `replaceRegion:` of a texture of a two-plane XR10 format (550, 551) over plane 0 of a
/// 'b3a8' surface, and its blits to and from a buffer, write the surface's bytes and the
/// buffer's as Metal's do, byte for byte; and so does `replaceRegion:` over a 'b3a8' surface
/// of one page made read-only for the GPU, which this platform writes itself
#[test]
fn cpu_writes_and_buffer_blits_of_two_plane_xr10_textures_over_surfaces_are_metals() {
    use crate::gpu::ycbcr::sampling::Op;
    use crate::gpu::ycbcr::sampling::written;
    use reims_vgpu_core::platform::os::vm::HostVm as _;
    let Some(md) = reims_vgpu_metal::MetalPlatform::system_default_device() else {
        return;
    };
    let mp = reims_vgpu_metal::MetalPlatform::new();
    let (p, d) = (crate::platform::tests::platform(), dev());
    let (size, rows) = ((16, 8), (64, 16));
    for format in [550, 551] {
        for op in [Op::Replace, Op::ToBuffer, Op::FromBuffer] {
            let pair = || (b3a8(size, rows, 21), b3a8(size, rows, 22));
            let (s, other) = pair();
            let ours = written(&p, &d, op, format, (&s, &other), size);
            let (s, other) = pair();
            let theirs = written(&mp, &md, op, format, (&s, &other), size);
            assert!(ours == theirs, "{format} {op:?}");
        }
        let vm = crate::os::OsVm::new();
        let other = b3a8(size, rows, 22);
        let (s, pages, length) = b3a8_read_only(size, rows, 23);
        let ours = written(&p, &d, Op::Replace, format, (&s, &other), size);
        drop(s);
        assert!(vm.deallocate(pages, length).is_success());
        let (s, pages, length) = b3a8_read_only(size, rows, 23);
        let theirs = written(&mp, &md, Op::Replace, format, (&s, &other), size);
        drop(s);
        assert!(vm.deallocate(pages, length).is_success());
        assert!(ours == theirs, "{format}: read-only for the GPU");
    }
}

/// A descriptor takes a pixel format Metal describes, as Metal's setter and
/// `texture2DDescriptorWithPixelFormat:width:height:mipmapped:` do, and answers it; at every
/// creator Metal makes a texture of it at (`newTextureWithDescriptor:`,
/// `newSharedTextureWithDescriptor:` at `Private` storage, a buffer's
/// `newTextureWithDescriptor:offset:bytesPerRow:`, the IOSurface creator over both planes of
/// an `x420` surface and a `BGRA` one's) and at the heap size query, which Metal answers, a
/// texture the runtime refuses dies where it is made, naming the raw format and Metal's name
/// for it, but at the size query of a format whose bytes the runtime counts though it holds no
/// image of it (624 to 627: `metal2vulkan::metal::format::footprint`), which answers Metal's
/// size and alignment. `newSharedTextureWithDescriptor:` at `Shared` storage answers
/// nil, as Metal does.
/// The formats: the plane formats 576 to 579 (`R10Unorm_X6`, `RG10Unorm_X12` and their sRGB
/// twins), which the runtime makes, and 624 to 627 (`RGB10A8_2P` and its `_sRGB`, `_PQ` and
/// `_HLG` twins), the formats Metal describes and makes textures of that the runtime holds no
/// image of, but the packed 10-bit ones (588 and 589,
/// `packed_10_bit_textures_over_memory_die_named`). The one-plane YCbCr formats, 501 among
/// them, the runtime holds for copies alone (`crate::gpu::ycbcr`'s
/// `one_plane_textures_are_made_and_copied_as_metals`). The runtime
/// makes a texture of a plane format at each creator Metal makes one at: over a buffer or a
/// surface, from a descriptor alone, a 2D texture or a texture buffer, the device's or a heap's,
/// and the shared one at `Private` storage (`shareable_padded_textures_are_metals`), of the
/// format, and answers the size query of either with Metal's size and alignment. It refuses each
/// of the others at every creator. It makes the bfloat formats, and
/// `rgbx`'s `bfloat_textures_read_and_write_as_metals` holds
/// them to Metal's.
///
/// The values Metal's setter takes that its table describes no format for, 630 to 650, which
/// this platform keeps raw though the runtime names formats of them, are taken and answered
/// too, and a texture of one dies as Metal's assertion at each of those creators
/// ("-[MTLTextureDescriptorInternal validateWithDevice:]:1416: failed assertion `Texture
/// Descriptor Validation MTLTextureDescriptor has invalid pixelFormat (630).'"), but the
/// shared creator, which answers nil at either storage, as Metal does (measured at macOS 26.7
/// and 27.0.1). Any other value dies at `setPixelFormat:`, where Metal asserts
/// ("MTLDebugValidateMTLPixelFormat:1622: failed assertion `pixelFormat (2000) is not a valid
/// MTLPixelFormat.'"); the three-channel formats, 45 to 48, 95 to 100 and 120 to 122, Metal's
/// setter takes and describes no format for. Metal's assertions were measured one case a
/// process, with `MTL_DEBUG_LAYER` unset; the setter's the same under it. Under
/// `MTL_DEBUG_LAYER`, `newSharedTextureWithDescriptor:` at `Shared` storage asserts
/// ("-[MTLDebugDevice newSharedTextureWithDescriptor:]:553") rather than answering nil.
#[test]
fn formats_the_runtime_lacks_die_where_textures_are_made() {
    use reims_vgpu_core::platform::Platform as _;
    use reims_vgpu_core::platform::descriptor::TextureDescriptor as _;
    use reims_vgpu_core::platform::resource::{Buffer as _, Heap as _, Texture as _};
    use reims_vgpu_core::platform::values::{HeapDesc, OptionSet, TextureDesc};
    let p = crate::platform::tests::platform();
    let md = reims_vgpu_metal::MetalPlatform::system_default_device();
    let mp = reims_vgpu_metal::MetalPlatform::new();
    let desc = |format, storage: u64| TextureDesc {
        texture_type: 2,
        pixel_format: format,
        width: 16,
        height: 16,
        usage: 1,
        resource_options: Some(storage << 4),
        ..TextureDesc::default()
    };
    // Where a texture of `format` dies: named, where Metal makes it, or as Metal's assertion.
    let dies = |sel: &str, format: u64, make: &dyn Fn()| {
        *crate::answer::LAST_LINE.lock() = None;
        let site = catch_death(make).unwrap_err();
        let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
        assert_eq!((site.kind, site.file), (DeathKind::Abort, sel), "{line}");
        assert!(
            line.contains(&format!("pixel format {format}")),
            "{sel}: {line}"
        );
        if crate::gpu::format_info::described("MTLPixelFormatGetInfoForDevice()", format) {
            // Named by Metal's description, whether the runtime has no format of it or
            // refuses the one it has.
            let name = crate::gpu::format_info::name("MTLPixelFormatGetInfoForDevice()", format);
            let named = format!("pixel format {format} ({name}, Metal's");
            assert!(line.contains(&named), "{line}");
            assert!(line.contains("the Vulkan runtime cannot hold"), "{line}");
            assert!(line.contains("which the runtime refuses"), "{line}");
        } else {
            assert!(line.contains("Metal aborts at"), "{line}");
        }
    };
    let x420 = crate::gpu::ycbcr::measure::Shape::of(b"x420").surface((16, 16));
    let bgra = crate::os::tests::bgra_surface(16, 16);
    for format in [576, 577, 578, 579, 624, 625, 626, 627, 630, 640, 650] {
        let (private, shared) = (desc(format, 2), desc(format, 0));
        let ours = p.texture_descriptor(&private);
        assert_eq!(ours.pixel_format(), format);
        // 576 to 579 are made at every creator, as Metal makes them, and answered their sizes.
        let padded = matches!(format, 576..=579);
        let made = |sel: &str, make: &dyn Fn() -> Option<VulkanTexture>| {
            if padded {
                let made = make().map(|t| t.pixel_format());
                assert_eq!(made, Some(format), "{sel}");
            } else {
                dies(sel, format, &|| drop(make()));
            }
        };
        made("newTextureWithDescriptor:", &|| {
            dev().new_texture(OnFault::Return, &ours).unwrap()
        });
        let sel = "heapTextureSizeAndAlignWithDescriptor:";
        let size = match padded || matches!(format, 624..=627) {
            true => Some(dev().heap_texture_size_and_align(&ours).unwrap()),
            false => {
                dies(sel, format, &|| {
                    let _ = dev().heap_texture_size_and_align(&ours);
                });
                None
            }
        };
        let sel = "newTextureWithDescriptor:offset:bytesPerRow:";
        let buffer = dev().new_buffer_of_length(OnFault::Die, 16 * 16 * 16, 0);
        let buffer = buffer.unwrap().unwrap();
        let theirs = p.texture_descriptor(&shared);
        made(sel, &|| {
            (buffer.new_texture(OnFault::Return, &theirs, 0, 256)).unwrap()
        });
        let sel = "newTextureWithDescriptor:iosurface:plane:";
        for (surface, plane) in [(&x420, 0), (&x420, 1), (&bgra, 0)] {
            made(sel, &|| {
                (dev().new_texture_on_surface(&ours, Some(surface), plane)).unwrap()
            });
        }
        let described =
            crate::gpu::format_info::described("MTLPixelFormatGetInfoForDevice()", format);
        let sel = "newSharedTextureWithDescriptor:";
        if padded {
            made(sel, &|| dev().new_shared_texture(&ours).unwrap());
        } else if described {
            dies(sel, format, &|| drop(dev().new_shared_texture(&ours)));
        } else {
            let made = dev().new_shared_texture(&ours).unwrap();
            assert!(made.is_none(), "{format}: shared at Private storage");
        }
        let made = dev().new_shared_texture(&theirs).unwrap();
        assert!(made.is_none(), "{format}: shared at Shared storage");
        let mut texture_buffer_size = None;
        if padded {
            // A texture buffer in memory of its own, the shared creator's too, and its size, and
            // each in a heap.
            let texture_buffer = p.texture_descriptor(&TextureDesc {
                texture_type: 9,
                height: 1,
                ..private
            });
            let made = dev().new_texture(OnFault::Return, &texture_buffer).unwrap();
            assert_eq!(
                made.map(|t| t.pixel_format()),
                Some(format),
                "a texture buffer"
            );
            let made = dev().new_shared_texture(&texture_buffer).unwrap();
            assert_eq!(
                made.map(|t| t.pixel_format()),
                Some(format),
                "a shared texture buffer"
            );
            texture_buffer_size = Some(dev().heap_texture_size_and_align(&texture_buffer).unwrap());
            let heap = dev().new_heap(&HeapDesc {
                size: 1 << 20,
                heap_type: 1,
                options: vec![OptionSet::StorageMode(2)],
                address_ranges: None,
                pinned_gpu_address: None,
            });
            let heap = heap.unwrap().heap.unwrap();
            for (what, d) in [("texture buffer", &texture_buffer), ("2D", &ours)] {
                let made = heap.new_texture(d, 0).unwrap();
                assert_eq!(
                    made.map(|t| t.pixel_format()),
                    Some(format),
                    "a heap's {what}"
                );
            }
        }
        let two_d = p.texture_2d_descriptor(OnFault::Return, format, 16, 16, false);
        assert_eq!(two_d.unwrap().pixel_format(), format, "a 2D descriptor's");
        let Some(md) = &md else { continue };
        let two_d = mp.texture_2d_descriptor(OnFault::Return, format, 16, 16, false);
        assert_eq!(
            two_d.unwrap().pixel_format(),
            format,
            "Metal: a 2D descriptor's"
        );
        let metals = |d| reims_vgpu_metal::gpu::desc::texture_descriptor(d);
        let (private, shared) = (metals(&private), metals(&shared));
        assert_eq!(private.pixel_format(), format, "Metal's setter");
        let format_of =
            |t: Option<reims_vgpu_metal::gpu::resource::MetalTexture>| t.map(|t| t.pixel_format());
        assert!(
            md.new_shared_texture(&shared).unwrap().is_none(),
            "Metal: shared"
        );
        let made = md.new_shared_texture(&private).unwrap();
        if !described {
            // Metal asserts at every other creator of these (measured one case a process).
            assert!(made.is_none(), "Metal: shared at Private");
            continue;
        }
        assert_eq!(format_of(made), Some(format), "Metal: shared at Private");
        let made = md.new_texture(OnFault::Return, &private).unwrap();
        assert_eq!(format_of(made), Some(format), "Metal");
        let theirs = md.heap_texture_size_and_align(&private).unwrap();
        assert_ne!(theirs.size, 0, "Metal: {format}'s size");
        if let Some(ours) = size {
            assert_eq!(ours, theirs, "{format}'s size");
        }
        if padded {
            // Metal makes each texture in memory of its own, the shared creator's too, and
            // answers a texture buffer's size, the runtime's.
            let texture_buffer = metals(&TextureDesc {
                texture_type: 9,
                height: 1,
                ..desc(format, 2)
            });
            let made = md.new_texture(OnFault::Return, &texture_buffer).unwrap();
            assert_eq!(format_of(made), Some(format), "Metal: a texture buffer");
            let made = md.new_shared_texture(&texture_buffer).unwrap();
            assert_eq!(
                format_of(made),
                Some(format),
                "Metal: a shared texture buffer"
            );
            let size = md.heap_texture_size_and_align(&texture_buffer).unwrap();
            assert_ne!(size.size, 0, "Metal: a texture buffer's size");
            assert_eq!(texture_buffer_size, Some(size), "a texture buffer's size");
            let heap = md.new_heap(&HeapDesc {
                size: 1 << 20,
                heap_type: 1,
                options: vec![OptionSet::StorageMode(2)],
                address_ranges: None,
                pinned_gpu_address: None,
            });
            let heap = heap.unwrap().heap.unwrap();
            for (what, d) in [("texture buffer", &texture_buffer), ("2D", &private)] {
                let made = heap.new_texture(d, 0).unwrap();
                assert_eq!(format_of(made), Some(format), "Metal: a heap's {what}");
            }
        }
        let buffer = md.new_buffer_of_length(OnFault::Die, 16 * 16 * 16, 0);
        let made = buffer
            .unwrap()
            .unwrap()
            .new_texture(OnFault::Return, &shared, 0, 256);
        assert_eq!(
            format_of(made.unwrap()),
            Some(format),
            "Metal: over a buffer"
        );
        for (surface, plane) in [(&x420, 0), (&x420, 1), (&bgra, 0)] {
            let made = md
                .new_texture_on_surface(&private, Some(surface), plane)
                .unwrap();
            assert_eq!(format_of(made), Some(format), "Metal: on plane {plane}");
        }
    }
    for format in [2, 45, 48, 95, 100, 120, 122, 653, 2000] {
        if crate::gpu::format_info::setter_takes(format) {
            let held = p.texture_descriptor(&desc(format, 2));
            assert_eq!(held.pixel_format(), format, "the setter's");
            continue;
        }
        *crate::answer::LAST_LINE.lock() = None;
        let site = catch_death(|| p.texture_descriptor(&desc(format, 2))).unwrap_err();
        let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
        assert_eq!(
            (site.kind, site.file),
            (DeathKind::Abort, "setPixelFormat:"),
            "{line}"
        );
        let what = format!("Metal aborts at pixel format {format}");
        assert!(line.contains(&what), "{line}");
    }
}

/// `newSharedTextureWithDescriptor:` of a padded 10-bit format (576 to 579, `R10Unorm_X6`,
/// `RG10Unorm_X12` and their sRGB twins) at `Private` storage, `ShaderRead`, makes a texture of
/// the format, as Metal does: a 2D one of 8x8 and of 16x16, and a texture buffer of 8 and of 16,
/// 16 cases
#[test]
fn shareable_padded_textures_are_metals() {
    use reims_vgpu_core::platform::Platform as _;
    use reims_vgpu_core::platform::resource::Texture as _;
    use reims_vgpu_core::platform::values::TextureDesc;
    let Some(md) = reims_vgpu_metal::MetalPlatform::system_default_device() else {
        return;
    };
    let p = crate::platform::tests::platform();
    let mut differ = Vec::new();
    for format in [576, 577, 578, 579] {
        for (texture_type, width) in [(2, 8), (2, 16), (9, 8), (9, 16)] {
            let desc = TextureDesc {
                texture_type,
                pixel_format: format,
                width,
                height: if texture_type == 9 { 1 } else { width },
                usage: 1,
                resource_options: Some(2 << 4),
                ..TextureDesc::default()
            };
            let ours = dev().new_shared_texture(&p.texture_descriptor(&desc));
            let ours = ours.unwrap().map(|t| t.pixel_format());
            let theirs = reims_vgpu_metal::gpu::desc::texture_descriptor(&desc);
            let theirs = md.new_shared_texture(&theirs).unwrap();
            let theirs = theirs.map(|t| t.pixel_format());
            assert_eq!(
                theirs,
                Some(format),
                "Metal: {format}, type {texture_type}, {width}"
            );
            if ours != theirs {
                differ.push((format, texture_type, width, ours));
            }
        }
    }
    assert!(differ.is_empty(), "{differ:?}");
}

/// A 64x64 2D descriptor of `format` at storage mode `storage`, of `usage`.
fn packed_10(
    format: u64,
    storage: u64,
    usage: u64,
) -> reims_vgpu_core::platform::values::TextureDesc {
    reims_vgpu_core::platform::values::TextureDesc {
        texture_type: 2,
        pixel_format: format,
        width: 64,
        height: 64,
        usage,
        resource_options: Some(storage << 4),
        ..reims_vgpu_core::platform::values::TextureDesc::default()
    }
}

/// `R10Unorm_PACKED` (588) and `RG10Unorm_PACKED` (589) are answered as Metal's where the
/// runtime holds Metal's answer: `newTextureWithDescriptor:` is nil at `Shared` and `Private`
/// storage, and so is `newSharedTextureWithDescriptor:` at `Shared`, and the heap size query
/// answers 0 bytes at an alignment of 16, as Metal's do (measured beside them). A `Memoryless`
/// render target, which Metal makes, dies at `newTextureWithDescriptor:` naming the format and
/// the runtime's refusal of it (`Unsupported`, `NotImplemented`): the runtime holds no image of
/// either format
#[test]
fn packed_10_bit_textures_are_metals_or_die_named() {
    use reims_vgpu_core::platform::Platform as _;
    use reims_vgpu_core::platform::resource::Texture as _;
    let p = crate::platform::tests::platform();
    let md = reims_vgpu_metal::MetalPlatform::system_default_device();
    for format in [588, 589] {
        for storage in [0, 2] {
            let ours = p.texture_descriptor(&packed_10(format, storage, 1));
            let made = dev().new_texture(OnFault::Return, &ours).unwrap();
            assert!(made.is_none(), "{format} at {storage}");
            if storage == 0 {
                let made = dev().new_shared_texture(&ours).unwrap();
                assert!(made.is_none(), "{format}: shared at {storage}");
            }
            let size = dev().heap_texture_size_and_align(&ours).unwrap();
            assert_eq!((size.size, size.align), (0, 16), "{format} at {storage}");
            let Some(md) = &md else { continue };
            let theirs =
                reims_vgpu_metal::gpu::desc::texture_descriptor(&packed_10(format, storage, 1));
            let made = md.new_texture(OnFault::Return, &theirs).unwrap();
            assert!(made.is_none(), "Metal: {format} at {storage}");
            if storage == 0 {
                let made = md.new_shared_texture(&theirs).unwrap();
                assert!(made.is_none(), "Metal: {format}: shared at {storage}");
            }
            assert_eq!(md.heap_texture_size_and_align(&theirs).unwrap(), size);
        }
        let memoryless = packed_10(format, 3, 4);
        let ours = p.texture_descriptor(&memoryless);
        *crate::answer::LAST_LINE.lock() = None;
        let site = catch_death(|| dev().new_texture(OnFault::Return, &ours)).unwrap_err();
        let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
        let sel = "newTextureWithDescriptor:";
        assert_eq!((site.kind, site.file), (DeathKind::Abort, sel), "{line}");
        let name = crate::gpu::format_info::name(sel, format);
        let named = format!("pixel format {format} ({name}, Metal's");
        assert!(line.contains(&named), "{line}");
        assert!(line.contains("which the runtime refuses"), "{line}");
        let Some(md) = &md else { continue };
        let theirs = reims_vgpu_metal::gpu::desc::texture_descriptor(&memoryless);
        let made = md.new_texture(OnFault::Return, &theirs).unwrap();
        assert_eq!(
            made.map(|t| t.pixel_format()),
            Some(format),
            "Metal: memoryless"
        );
    }
}

/// A texture of `R10Unorm_PACKED` (588) or `RG10Unorm_PACKED` (589) over memory, 64x64 2D of
/// `ShaderRead` usage, which Metal makes (measured beside it, M3 Max, macOS 27.0.1, 2026-10-25):
/// from `newSharedTextureWithDescriptor:` at `Private` storage, from a placement heap's
/// `newTextureWithDescriptor:offset:` at `Shared` and `Private` storage, of `allocatedSize` 0, from
/// a buffer's `newTextureWithDescriptor:offset:bytesPerRow:` at both, and from an IOSurface's
/// plane, dies where it is made, naming the format and the runtime's refusal of it
/// (`Unsupported`, `NotImplemented`): the runtime holds no image of a packed code stream
#[test]
fn packed_10_bit_textures_over_memory_die_named() {
    use reims_vgpu_core::platform::Platform as _;
    use reims_vgpu_core::platform::resource::{Buffer as _, Heap as _, Texture as _};
    use reims_vgpu_core::platform::values::{HeapDesc, OptionSet};
    let p = crate::platform::tests::platform();
    let md = reims_vgpu_metal::MetalPlatform::system_default_device();
    // A placement heap's descriptor at `storage`, for this platform or Metal's.
    macro_rules! heap {
        ($storage:expr) => {
            HeapDesc {
                size: 1 << 22,
                heap_type: 1,
                options: vec![OptionSet::StorageMode($storage)],
                address_ranges: None,
                pinned_gpu_address: None,
            }
        };
    }
    let dies = |sel: &'static str, format: u64, make: &dyn Fn()| {
        *crate::answer::LAST_LINE.lock() = None;
        let site = catch_death(make).unwrap_err();
        let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
        assert_eq!((site.kind, site.file), (DeathKind::Abort, sel), "{line}");
        let name = crate::gpu::format_info::name(sel, format);
        let named = format!("pixel format {format} ({name}, Metal's");
        assert!(line.contains(&named), "{sel}: {line}");
        assert!(line.contains("which the runtime refuses"), "{sel}: {line}");
    };
    let surface = crate::os::tests::bgra_surface(64, 64);
    for format in [588, 589] {
        for storage in [0, 2] {
            let desc = packed_10(format, storage, 1);
            let ours = p.texture_descriptor(&desc);
            if storage == 2 {
                dies("newSharedTextureWithDescriptor:", format, &|| {
                    drop(dev().new_shared_texture(&ours));
                });
            }
            let made = dev().new_heap(&heap!(storage)).unwrap().heap.unwrap();
            dies("newTextureWithDescriptor:offset:", format, &|| {
                drop(made.new_texture(&ours, 0));
            });
            let buffer = dev().new_buffer_of_length(OnFault::Die, 1 << 16, storage << 4);
            let buffer = buffer.unwrap().unwrap();
            dies(
                "newTextureWithDescriptor:offset:bytesPerRow:",
                format,
                &|| {
                    drop(buffer.new_texture(OnFault::Return, &ours, 0, 256));
                },
            );
            dies("newTextureWithDescriptor:iosurface:plane:", format, &|| {
                drop(dev().new_texture_on_surface(&ours, Some(&surface), 0));
            });
            let Some(md) = &md else { continue };
            let theirs = reims_vgpu_metal::gpu::desc::texture_descriptor(&desc);
            let made = |what: &str, made: Option<u64>| {
                assert_eq!(made, Some(format), "Metal: {what} of {format} at {storage}");
            };
            if storage == 2 {
                let shared = md.new_shared_texture(&theirs).unwrap();
                made("shared", shared.map(|t| t.pixel_format()));
            }
            let heap = md.new_heap(&heap!(storage)).unwrap().heap.unwrap();
            let placed = heap.new_texture(&theirs, 0).unwrap();
            made("a heap's", placed.map(|t| t.pixel_format()));
            let buffer = md.new_buffer_of_length(OnFault::Die, 1 << 16, storage << 4);
            let buffer = buffer.unwrap().unwrap();
            let over = buffer
                .new_texture(OnFault::Return, &theirs, 0, 256)
                .unwrap();
            made("a buffer's", over.map(|t| t.pixel_format()));
        }
    }
}

/// A sample of a texture of `R10Unorm_X6` (576) or `RG10Unorm_X12` (578) over a buffer is
/// Metal's, byte for byte, into floats, at positions in and around the texture off its texel
/// grid: through a sampler state, nearest and linear, under every address mode and border
/// colour, in normalized and pixel coordinates (a pixel-coordinate repeat or mirrored axis
/// samples as clamp to the edge, as Metal's does), at the dispatch, and linear at the draw;
/// and through a constexpr sampler of each mode a function states, at the dispatch and at
/// the draw
#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "MoltenVK: the runtime records device-pack-conversion, a death in strict mode; Mac-only test, off everywhere; ledger: MoltenVK strict deaths"
)]
fn padded_samples_are_metals() {
    use crate::gpu::ycbcr::measure::{drawn, probe, probed_texels, sampler};
    use reims_vgpu_core::platform::Platform;
    use reims_vgpu_core::platform::resource::Buffer as _;
    use reims_vgpu_core::platform::values::TextureDesc;
    // A texture of `format` over a Shared buffer of noise on `p`'s device `d`.
    fn over_buffer<P: Platform>(
        p: &P,
        d: &P::Device,
        format: u64,
        (width, height): (u64, u64),
    ) -> P::Texture {
        let row = 256;
        let b = d
            .new_buffer_of_length(OnFault::Die, row * height, 0)
            .unwrap()
            .unwrap();
        // SAFETY: a Shared buffer of `row * height` bytes, which no command uses yet.
        unsafe {
            let to = b.contents().as_ptr::<u8>();
            for i in 0..row * height {
                to.add(usize::try_from(i).unwrap())
                    .write((i * 37 % 251) as u8);
            }
        }
        let desc = p.texture_descriptor(&TextureDesc {
            pixel_format: format,
            width,
            height,
            usage: 1,
            resource_options: Some(0),
            ..TextureDesc::default()
        });
        b.new_texture(OnFault::Die, &desc, 0, row).unwrap().unwrap()
    }
    // Each case's name and what it samples of a texture of `format` on `p`'s device `d`.
    fn samples<P: Platform>(p: &P, d: &P::Device, format: u64) -> Vec<(String, Vec<u8>)> {
        let (size, grid) = ((16, 16), (16, 16));
        let t = over_buffer(p, d, format, size);
        // From four texels before the texture to four past it, off the texel grid.
        let at = |normalized| probe((-4.25, -4.25), 1.625, grid, normalized, 0);
        let mut out = Vec::new();
        for (filter, normalized) in [(0, true), (1, true), (0, false), (1, false)] {
            let modes = (0..5).map(|mode| (mode, 0)).chain((0..3).map(|b| (5, b)));
            for (mode, border) in modes {
                let s = sampler::<P>(d, (filter, mode, normalized), border);
                let bytes = at(normalized);
                let case = format!("{filter} {mode} {normalized} {border}");
                let kernel = ("sample_only_texels_float", 16);
                let read = probed_texels::<P>(d, kernel, &t, grid, Some(&s), Some(&bytes));
                out.push((format!("dispatch {case}"), read));
                if filter == 1 {
                    let fragment = ("sample_fragment_float", 16);
                    let read = drawn::<P>(p, d, fragment, &t, grid, &s, &bytes);
                    out.push((format!("draw {case}"), read));
                }
            }
        }
        let bytes = at(true);
        for kernel in [
            "sample_constexpr_repeat",
            "sample_constexpr_mirrored_repeat",
            "sample_constexpr_zero",
            "sample_constexpr_repeat_linear",
            "sample_constexpr_mirrored_repeat_linear",
            "sample_constexpr_border_white",
            "sample_constexpr_border_black_linear",
            "sample_constexpr_border_clear_linear",
        ] {
            let read = probed_texels::<P>(d, (kernel, 16), &t, grid, None, Some(&bytes));
            out.push((kernel.to_owned(), read));
        }
        let unused = sampler::<P>(d, (0, 0, true), 0);
        for fragment in [
            "sample_constexpr_repeat_linear_fragment",
            "sample_constexpr_mirrored_repeat_fragment",
            "sample_constexpr_border_clear_linear_fragment",
        ] {
            let read = drawn::<P>(p, d, (fragment, 16), &t, grid, &unused, &bytes);
            out.push((fragment.to_owned(), read));
        }
        out
    }
    let (p, d) = (crate::platform::tests::platform(), dev());
    let md = reims_vgpu_metal::MetalPlatform::system_default_device();
    let mp = reims_vgpu_metal::MetalPlatform::new();
    for format in [576, 578] {
        let ours = samples(&p, &d, format);
        let Some(md) = &md else { continue };
        let theirs = samples(&mp, md, format);
        assert_eq!(ours.len(), theirs.len());
        for ((case, ours), (_, theirs)) in ours.iter().zip(&theirs) {
            assert!(ours == theirs, "{format}: {case}");
        }
    }
}

/// A shareable texture of a width of 0 is nil in `Shared`, `Managed` and `Memoryless`
/// storage, as Metal's is, where Metal's shared creator answers nil for storage other than
/// `Private` rather than abort at the size; in `Private` storage it dies where Metal aborts
/// (Metal's side is not run, as its assertion ends the process).
#[test]
fn a_shareable_texture_of_no_width_is_nil_but_private() {
    use reims_vgpu_core::platform::Platform;
    use reims_vgpu_core::platform::values::TextureDesc;
    let desc = |storage: u64| TextureDesc {
        pixel_format: 80,
        width: 0,
        height: 16,
        usage: 1,
        resource_options: Some(storage << 4),
        ..TextureDesc::default()
    };
    let p = crate::platform::tests::platform();
    let md = reims_vgpu_metal::MetalPlatform::system_default_device();
    for storage in [0, 1, 3] {
        let made = dev().new_shared_texture(&p.texture_descriptor(&desc(storage)));
        assert!(made.unwrap().is_none(), "storage {storage}");
        if let Some(md) = &md {
            let metal = reims_vgpu_metal::gpu::desc::texture_descriptor(&desc(storage));
            assert!(
                md.new_shared_texture(&metal).unwrap().is_none(),
                "Metal: storage {storage}"
            );
        }
    }
    let private = p.texture_descriptor(&desc(2));
    let site = catch_death(|| dev().new_shared_texture(&private)).unwrap_err();
    assert_eq!(
        (site.kind, site.file),
        (DeathKind::Abort, "newSharedTextureWithDescriptor:")
    );
}

/// A texture descriptor's setters take what Metal's take, and each value acts where Metal's
/// does, measured one case a process with `MTL_DEBUG_LAYER` unset:
/// - a texture type the runtime has no type of (10, 11, 12, 100) is taken and answered, and
///   every creator aborts at it (`SIGABRT`, no message): `newTextureWithDescriptor:`, the
///   shared creator, the heap's size query, a buffer's two creators, a heap's and the
///   IOSurface creator;
/// - a swizzle channel past `MTLTextureSwizzleAlpha` (6, 9) is taken, and every creator
///   asserts ("_mtlValidateMTLTextureSwizzle:445: failed assertion `swizzle(9) is not a valid
///   MTLTextureSwizzle.'");
/// - protection options other than none (1, 2, 0x10) are taken: `newTextureWithDescriptor:`
///   and the shared creator answer nil, and every other creator makes the texture, and the size
///   query answers, as without them; this platform's two make it as without them too, relayed
///   ([`crate::gpu::desc::TextureFields::made_unprotected`]);
/// - a resource index is the texture's placement, forced or not: Metal places it there at
///   `newTextureWithDescriptor:`, the shared creator and the IOSurface creator either way.
///
/// A descriptor of `writeSwizzleEnabled` (SPI) with a swizzle other than the identity (B, G,
/// R, A of `RGBA8Unorm`), read alone, is made by `newTextureWithDescriptor:`, the shared
/// creator, a buffer's two creators, a heap's and the IOSurface creator, and the heap's size
/// query answers its size and alignment, as Metal's do. Written (`ShaderWrite`) or rendered
/// (`RenderTarget`), Metal's device, buffer, heap and IOSurface creators make it too and write
/// through the swizzle, which the runtime does not build: those creators die naming the
/// runtime's gap (`RuntimeGap::WriteSwizzle`), and the size query answers Metal's size and
/// alignment. The shared creator dies where Metal's validation stops it (the runtime's
/// measurement; Metal's side is not run, as its assertion ends the process).
#[test]
fn write_swizzles_are_made_where_read_alone() {
    use reims_vgpu_core::platform::Platform;
    use reims_vgpu_core::platform::resource::{Buffer as _, Heap as _, Texture as _};
    use reims_vgpu_core::platform::types::Swizzle;
    use reims_vgpu_core::platform::values::{HeapDesc, OptionSet, TextureDesc};
    let desc = |usage, storage: u64| TextureDesc {
        texture_type: 2,
        pixel_format: 70,
        width: 16,
        height: 16,
        usage,
        resource_options: Some(storage << 4),
        swizzle: Some(Swizzle {
            red: 4,
            green: 3,
            blue: 2,
            alpha: 5,
        }),
        write_swizzle_enabled: Some(true),
        ..TextureDesc::default()
    };
    /// The size query's size and alignment of `desc` on `d`.
    fn sized<P: Platform>(d: &P::Device, desc: &P::TextureDescriptor) -> (u64, u64) {
        let size = d.heap_texture_size_and_align(desc).unwrap();
        (size.size, size.align)
    }
    /// A buffer of 16 KiB and a placement heap of 1 MiB, `Private`, on `d`.
    fn memory<P: Platform>(d: &P::Device) -> (P::Buffer, P::Heap) {
        let buffer = d.new_buffer_of_length(OnFault::Die, 0x4000, 0);
        let heap = d.new_heap(&HeapDesc {
            size: 1 << 20,
            heap_type: 1,
            options: vec![OptionSet::StorageMode(2)],
            address_ranges: None,
            pinned_gpu_address: None,
        });
        (buffer.unwrap().unwrap(), heap.unwrap().heap.unwrap())
    }
    // Each creator's texture's pixel format (0 for nil) of `desc` on `p`'s device `d`: the
    // device's, the shared one, a buffer's two, a heap's and the IOSurface one.
    fn made<P: Platform<Surface = OsSurface>>(
        p: &P,
        d: &P::Device,
        desc: &TextureDesc,
    ) -> Vec<u64> {
        let format = |t: Option<P::Texture>| t.map_or(0, |t| t.pixel_format());
        let ours = p.texture_descriptor(desc);
        let shared = p.texture_descriptor(&TextureDesc {
            resource_options: Some(0),
            ..*desc
        });
        let (buffer, heap) = memory::<P>(d);
        let surface = crate::os::tests::bgra_surface(16, 16);
        let on_buffer = buffer.new_texture(OnFault::Return, &shared, 0, 256);
        let linear = buffer.new_linear_texture(&shared, 0, 256, 4096);
        let on_surface = d.new_texture_on_surface(&shared, Some(&surface), 0);
        vec![
            format(d.new_texture(OnFault::Return, &ours).unwrap()),
            format(d.new_shared_texture(&ours).unwrap()),
            format(on_buffer.unwrap()),
            format(linear.unwrap()),
            format(heap.new_texture(&ours, 0).unwrap()),
            format(on_surface.unwrap()),
        ]
    }
    let p = crate::platform::tests::platform();
    let md = reims_vgpu_metal::MetalPlatform::system_default_device();
    let mp = reims_vgpu_metal::MetalPlatform::new();
    let metals = |d: &TextureDesc| reims_vgpu_metal::gpu::desc::texture_descriptor(d);
    let read = desc(1, 2);
    let ours = made(&p, &dev(), &read);
    assert_eq!(ours, [70; 6], "read alone");
    let size = sized::<VulkanPlatform>(&dev(), &p.texture_descriptor(&read));
    if let Some(md) = &md {
        assert_eq!(made(&mp, md, &read), ours, "Metal: read alone");
        let theirs = sized::<reims_vgpu_metal::MetalPlatform>(md, &metals(&read));
        assert_eq!(size, theirs, "read alone: the size query");
    }
    for (what, usage) in [("written", 3), ("rendered", 5)] {
        let d = dev();
        let written = p.texture_descriptor(&desc(usage, 2));
        let shared = p.texture_descriptor(&desc(usage, 0));
        let (buffer, heap) = memory::<VulkanPlatform>(&d);
        let surface = crate::os::tests::bgra_surface(16, 16);
        let creators: [(&str, &dyn Fn()); 6] = [
            ("newTextureWithDescriptor:", &|| {
                let _ = d.new_texture(OnFault::Return, &written);
            }),
            ("newSharedTextureWithDescriptor:", &|| {
                let _ = d.new_shared_texture(&written);
            }),
            ("newTextureWithDescriptor:offset:bytesPerRow:", &|| {
                let _ = buffer.new_texture(OnFault::Return, &shared, 0, 256);
            }),
            (
                "newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:",
                &|| {
                    let _ = buffer.new_linear_texture(&shared, 0, 256, 4096);
                },
            ),
            ("newTextureWithDescriptor:offset:", &|| {
                let _ = heap.new_texture(&written, 0);
            }),
            ("newTextureWithDescriptor:iosurface:plane:", &|| {
                let _ = d.new_texture_on_surface(&shared, Some(&surface), 0);
            }),
        ];
        for (sel, make) in creators {
            *crate::answer::LAST_LINE.lock() = None;
            let site = catch_death(make).unwrap_err();
            let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
            assert_eq!((site.kind, site.file), (DeathKind::Abort, sel), "{line}");
            let refused = match sel {
                "newSharedTextureWithDescriptor:" => "Texture(SwizzleWithWrites)",
                _ => "not implemented in the runtime yet: WriteSwizzle",
            };
            assert!(line.contains(refused), "{what} {sel}: {line}");
        }
        let size = sized::<VulkanPlatform>(&d, &written);
        assert_ne!(size.0, 0, "{what}: the size query");
        let Some(md) = &md else { continue };
        let metal = desc(usage, 2);
        let (private, shared) = (metals(&metal), metals(&desc(usage, 0)));
        let format = |t: Option<reims_vgpu_metal::gpu::resource::MetalTexture>| {
            t.map_or(0, |t| t.pixel_format())
        };
        let (buffer, heap) = memory::<reims_vgpu_metal::MetalPlatform>(md);
        let surface = crate::os::tests::bgra_surface(16, 16);
        let on_buffer = buffer.new_texture(OnFault::Return, &shared, 0, 256);
        let linear = buffer.new_linear_texture(&shared, 0, 256, 4096);
        let on_surface = md.new_texture_on_surface(&shared, Some(&surface), 0);
        let theirs = [
            format(md.new_texture(OnFault::Return, &private).unwrap()),
            format(on_buffer.unwrap()),
            format(linear.unwrap()),
            format(heap.new_texture(&private, 0).unwrap()),
            format(on_surface.unwrap()),
        ];
        assert_eq!(theirs, [70; 5], "Metal: {what}");
        let theirs = sized::<reims_vgpu_metal::MetalPlatform>(md, &private);
        assert_eq!(size, theirs, "{what}: the size query");
    }
}

/// Metal's answers that are no assertion are compared on the host, but for protection options:
/// this platform's `newTextureWithDescriptor:` and `newSharedTextureWithDescriptor:` make a
/// texture of them, relayed, where Metal's answer nil (checked beside them), as every other
/// creator of both makes one.
#[test]
fn texture_setters_act_where_metal_does() {
    use reims_vgpu_core::platform::Platform as _;
    use reims_vgpu_core::platform::descriptor::TextureDescriptor as _;
    use reims_vgpu_core::platform::resource::{Buffer as _, Heap as _, Texture as _};
    use reims_vgpu_core::platform::types::Swizzle;
    use reims_vgpu_core::platform::values::{HeapDesc, OptionSet, TextureDesc};
    let p = crate::platform::tests::platform();
    let md = reims_vgpu_metal::MetalPlatform::system_default_device();
    let base = TextureDesc {
        texture_type: 2,
        pixel_format: 80,
        width: 16,
        height: 16,
        usage: 1,
        resource_options: Some(0x20),
        ..TextureDesc::default()
    };
    let surface = crate::os::tests::bgra_surface(16, 16);
    let heap = |d: &VulkanDevice| {
        let made = d.new_heap(&HeapDesc {
            size: 1 << 20,
            heap_type: 1,
            options: vec![OptionSet::StorageMode(2)],
            address_ranges: None,
            pinned_gpu_address: None,
        });
        made.unwrap().heap.unwrap()
    };
    // Every creator of `desc`, each dying as Metal's abort, its line naming `what`.
    let every_creator_aborts = |desc: &TextureDesc, what: &str| {
        let d = dev();
        let ours = p.texture_descriptor(desc);
        let shared = p.texture_descriptor(&TextureDesc {
            resource_options: Some(0),
            ..*desc
        });
        let buffer = d.new_buffer_of_length(OnFault::Die, 0x4000, 0);
        let buffer = buffer.unwrap().unwrap();
        let heap = heap(&d);
        let creators: [(&str, &dyn Fn()); 7] = [
            ("newTextureWithDescriptor:", &|| {
                let _ = d.new_texture(OnFault::Return, &ours);
            }),
            ("newSharedTextureWithDescriptor:", &|| {
                let _ = d.new_shared_texture(&ours);
            }),
            ("heapTextureSizeAndAlignWithDescriptor:", &|| {
                let _ = d.heap_texture_size_and_align(&ours);
            }),
            ("newTextureWithDescriptor:offset:bytesPerRow:", &|| {
                let _ = buffer.new_texture(OnFault::Return, &shared, 0, 256);
            }),
            (
                "newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:",
                &|| {
                    let _ = buffer.new_linear_texture(&shared, 0, 256, 4096);
                },
            ),
            ("newTextureWithDescriptor:offset:", &|| {
                let _ = heap.new_texture(&ours, 0);
            }),
            ("newTextureWithDescriptor:iosurface:plane:", &|| {
                let _ = d.new_texture_on_surface(&ours, Some(&surface), 0);
            }),
        ];
        for (sel, make) in creators {
            *crate::answer::LAST_LINE.lock() = None;
            let site = catch_death(make).unwrap_err();
            let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
            assert_eq!((site.kind, site.file), (DeathKind::Abort, sel), "{line}");
            assert!(
                line.contains("Metal aborts at") && line.contains(what),
                "{line}"
            );
        }
    };
    for raw in [10, 11, 12, 100] {
        let desc = TextureDesc {
            texture_type: raw,
            ..base
        };
        assert_eq!(p.texture_descriptor(&desc).texture_type(), raw);
        let set = p.texture_descriptor(&base);
        set.set_texture_type(raw);
        assert_eq!(set.texture_type(), raw);
        if md.is_some() {
            let metals = reims_vgpu_metal::gpu::desc::texture_descriptor(&desc);
            assert_eq!(metals.texture_type(), raw, "Metal");
        }
        every_creator_aborts(&desc, &format!("texture type {raw}"));
    }
    for channel in [6, 9] {
        let desc = TextureDesc {
            swizzle: Some(Swizzle {
                red: channel,
                green: 3,
                blue: 4,
                alpha: 5,
            }),
            ..base
        };
        p.texture_descriptor(&desc);
        if md.is_some() {
            reims_vgpu_metal::gpu::desc::texture_descriptor(&desc);
        }
        every_creator_aborts(&desc, &format!("swizzle ({channel}, 3, 4, 5)"));
    }
    // What each creator answers on `d` for `desc`: made, nil, or the size query's answer.
    fn answers<P: reims_vgpu_core::platform::Platform>(
        p: &P,
        d: &P::Device,
        desc: &TextureDesc,
        surface: &P::Surface,
        heap: &P::Heap,
    ) -> Vec<String> {
        let ours = p.texture_descriptor(desc);
        let shared = p.texture_descriptor(&TextureDesc {
            resource_options: Some(0),
            ..*desc
        });
        let buffer = d.new_buffer_of_length(OnFault::Die, 0x4000, 0);
        let buffer = buffer.unwrap().unwrap();
        // A placed texture's resource ID; others' differ by what each process made before.
        let placed = desc.resource_index.is_some();
        let made = |t: Option<P::Texture>| match t {
            Some(t) if placed => format!("made at {:#x}", t.gpu_resource_id()),
            Some(_) => "made".to_owned(),
            None => "nil".to_owned(),
        };
        let mut out = vec![
            made(d.new_texture(OnFault::Return, &ours).unwrap()),
            made(d.new_shared_texture(&ours).unwrap()),
            made(d.new_texture_on_surface(&ours, Some(surface), 0).unwrap()),
        ];
        if !placed {
            let size = d.heap_texture_size_and_align(&ours).unwrap();
            out.push(format!("{} bytes at {}", size.size, size.align));
            let t = buffer.new_texture(OnFault::Return, &shared, 0, 256);
            out.push(made(t.unwrap()));
            let t = buffer.new_linear_texture(&shared, 0, 256, 4096);
            out.push(made(t.unwrap()));
            out.push(made(heap.new_texture(&ours, 0).unwrap()));
        }
        out
    }
    let theirs = md.as_ref().map(|md| {
        let mp = reims_vgpu_metal::MetalPlatform::new();
        let heap = md.new_heap(&HeapDesc {
            size: 1 << 20,
            heap_type: 1,
            options: vec![OptionSet::StorageMode(2)],
            address_ranges: None,
            pinned_gpu_address: None,
        });
        (mp, md, heap.unwrap().heap.unwrap())
    });
    let d = dev();
    let our_heap = heap(&d);
    for desc in [
        TextureDesc {
            protection_options: Some(1),
            ..base
        },
        TextureDesc {
            protection_options: Some(2),
            ..base
        },
        TextureDesc {
            protection_options: Some(0x10),
            ..base
        },
        TextureDesc {
            resource_index: Some((0x700, false)),
            ..base
        },
        TextureDesc {
            resource_index: Some((0x700, true)),
            ..base
        },
    ] {
        // Protection options are relayed: strict mode would end the test at them.
        let ours = crate::answer::relaxed(|| answers(&p, &d, &desc, &surface, &our_heap));
        let expected: &[&str] = match desc.protection_options {
            Some(_) => &["made", "made", "made"],
            None => &["made at 0x700", "made at 0x700", "made at 0x700"],
        };
        assert_eq!(&ours[..expected.len()], expected, "{desc:?}");
        if let Some((mp, md, heap)) = &theirs {
            let metal = answers(mp, md, &desc, &surface, heap);
            match desc.protection_options {
                Some(_) => {
                    assert_eq!(metal[..2], ["nil", "nil"], "Metal: {desc:?}");
                    assert_eq!(ours[2..], metal[2..], "{desc:?}");
                }
                None => assert_eq!(ours, metal, "{desc:?}"),
            }
        }
        if desc.protection_options.is_none() {
            continue;
        }
        // Made as without them: this platform's texture answers protection options 0, as each
        // texture Metal makes of them does.
        let made =
            crate::answer::relaxed(|| d.new_texture(OnFault::Return, &p.texture_descriptor(&desc)));
        assert_eq!(made.unwrap().unwrap().protection_options(), 0, "{desc:?}");
        if let Some((mp, md, heap)) = &theirs {
            let metals = mp.texture_descriptor(&desc);
            let shared = mp.texture_descriptor(&TextureDesc {
                resource_options: Some(0),
                ..desc
            });
            let buffer = md.new_buffer_of_length(OnFault::Die, 0x4000, 0);
            let buffer = buffer.unwrap().unwrap();
            let made = [
                heap.new_texture(&metals, 0).unwrap(),
                md.new_texture_on_surface(&metals, Some(&surface), 0)
                    .unwrap(),
                buffer
                    .new_texture(OnFault::Return, &shared, 0, 256)
                    .unwrap(),
            ];
            for t in made {
                assert_eq!(t.unwrap().protection_options(), 0, "Metal: {desc:?}");
            }
        }
    }
}

/// `Depth24Unorm_Stencil8` (255) and `X24_Stencil8` (262), formats the runtime has that Metal
/// describes none for here, are taken and answered, as Metal's setter does, and a texture of
/// either dies as Metal's assertion at every creator ("-[MTLTextureDescriptorInternal
/// validateWithDevice:]:1416: failed assertion `Texture Descriptor Validation
/// MTLTextureDescriptor has invalid pixelFormat (255).'", measured one case a process with
/// `MTL_DEBUG_LAYER` unset), `newSharedTextureWithDescriptor:` at `Private` storage among
/// them; the shared creator answers nil at `Shared` storage. It answers nil for
/// `MTLPixelFormatInvalid` at either storage too, as Metal does, where every other creator
/// dies, as Metal asserts.
#[test]
fn invalid_formats_are_metals() {
    use reims_vgpu_core::platform::Platform as _;
    use reims_vgpu_core::platform::descriptor::TextureDescriptor as _;
    use reims_vgpu_core::platform::resource::Buffer as _;
    use reims_vgpu_core::platform::values::TextureDesc;
    let p = crate::platform::tests::platform();
    let md = reims_vgpu_metal::MetalPlatform::system_default_device();
    let desc = |format, storage: u64| TextureDesc {
        texture_type: 2,
        pixel_format: format,
        width: 16,
        height: 16,
        usage: 1,
        resource_options: Some(storage << 4),
        ..TextureDesc::default()
    };
    let aborts = |sel: &str, format: u64, make: &dyn Fn()| {
        *crate::answer::LAST_LINE.lock() = None;
        let site = catch_death(make).unwrap_err();
        let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
        assert_eq!((site.kind, site.file), (DeathKind::Abort, sel), "{line}");
        assert!(line.contains("Metal aborts at"), "{format}: {line}");
    };
    let bgra = crate::os::tests::bgra_surface(16, 16);
    for format in [255, 262, 0] {
        let (private, shared) = (desc(format, 2), desc(format, 0));
        let ours = p.texture_descriptor(&private);
        let theirs = p.texture_descriptor(&shared);
        assert_eq!(ours.pixel_format(), format);
        aborts("newTextureWithDescriptor:", format, &|| {
            let _ = dev().new_texture(OnFault::Return, &ours);
        });
        aborts("heapTextureSizeAndAlignWithDescriptor:", format, &|| {
            let _ = dev().heap_texture_size_and_align(&ours);
        });
        let buffer = dev().new_buffer_of_length(OnFault::Die, 16 * 16 * 16, 0);
        let buffer = buffer.unwrap().unwrap();
        aborts(
            "newTextureWithDescriptor:offset:bytesPerRow:",
            format,
            &|| {
                let _ = buffer.new_texture(OnFault::Return, &theirs, 0, 256);
            },
        );
        aborts("newTextureWithDescriptor:iosurface:plane:", format, &|| {
            let _ = dev().new_texture_on_surface(&ours, Some(&bgra), 0);
        });
        if format == 0 {
            assert!(dev().new_shared_texture(&ours).unwrap().is_none());
        } else {
            aborts("newSharedTextureWithDescriptor:", format, &|| {
                let _ = dev().new_shared_texture(&ours);
            });
        }
        assert!(dev().new_shared_texture(&theirs).unwrap().is_none());
        let Some(md) = &md else { continue };
        let metals = |d| reims_vgpu_metal::gpu::desc::texture_descriptor(d);
        let (private, shared) = (metals(&private), metals(&shared));
        assert_eq!(private.pixel_format(), format, "Metal's setter");
        assert!(md.new_shared_texture(&shared).unwrap().is_none(), "Metal");
        if format == 0 {
            assert!(md.new_shared_texture(&private).unwrap().is_none(), "Metal");
        }
    }
}

/// `setPixelFormat:` takes and answers what Metal's takes, every value from 0 to 2047 tried:
/// the values of [`crate::gpu::format_info::setter_takes`], which Metal's setter takes and
/// answers too, compared on the host, 301, the 13 three-channel formats among them; every other
/// value dies at the setter, where Metal's asserts
/// (measured one value a process with `MTL_DEBUG_LAYER` unset)
#[test]
fn the_setter_takes_what_metals_takes() {
    use reims_vgpu_core::platform::Platform as _;
    use reims_vgpu_core::platform::descriptor::TextureDescriptor as _;
    let p = crate::platform::tests::platform();
    let ours = p.texture_descriptor(&reims_vgpu_core::platform::values::TextureDesc::default());
    let metal = reims_vgpu_metal::MetalPlatform::system_default_device().map(|_| {
        reims_vgpu_metal::gpu::desc::texture_descriptor(
            &reims_vgpu_core::platform::values::TextureDesc::default(),
        )
    });
    let mut taken = 0;
    for raw in 0..2048 {
        match catch_death(|| ours.set_pixel_format(raw)) {
            Ok(()) => {
                taken += 1;
                assert!(crate::gpu::format_info::setter_takes(raw), "{raw}");
                assert_eq!(ours.pixel_format(), raw);
                if let Some(metal) = &metal {
                    metal.set_pixel_format(raw);
                    assert_eq!(metal.pixel_format(), raw, "Metal");
                }
            }
            Err(site) => {
                assert!(!crate::gpu::format_info::setter_takes(raw), "{raw}");
                assert_eq!(
                    (site.kind, site.file),
                    (DeathKind::Abort, "setPixelFormat:")
                );
            }
        }
    }
    assert_eq!(taken, 301);
}

/// `texture2DDescriptorWithPixelFormat:width:height:mipmapped:` keeps and answers the formats
/// Metal makes no texture of from a descriptor alone, as Metal's does: the two YCbCr formats
/// measured first, 505 and 543 ([`crate::gpu::ycbcr`]), and those `setPixelFormat:` keeps raw
/// ([`crate::gpu::no_texture`])
#[test]
fn two_d_descriptors_keep_raw_formats_as_the_setter_does() {
    use reims_vgpu_core::platform::Platform as _;
    use reims_vgpu_core::platform::descriptor::TextureDescriptor as _;
    let p = crate::platform::tests::platform();
    let mp = reims_vgpu_metal::MetalPlatform::system_default_device()
        .map(|_| reims_vgpu_metal::MetalPlatform::new());
    let raw = [505, 543].into_iter();
    for format in raw.chain(crate::gpu::no_texture::FORMATS) {
        let ours = p.texture_2d_descriptor(OnFault::Return, format, 16, 16, false);
        assert_eq!(ours.unwrap().pixel_format(), format);
        if let Some(mp) = &mp {
            let theirs = mp.texture_2d_descriptor(OnFault::Return, format, 16, 16, false);
            assert_eq!(theirs.unwrap().pixel_format(), format, "Metal");
        }
    }
}

/// A texture over a plane its surface lacks is Metal's: nil past a two-plane surface's
/// planes, as Metal answers, and a death where it is made at a plane past 0 of a surface
/// without planes, where Metal asserts ("IOSurface is non-planar, but plane 1 specified",
/// measured for each format here, one case a process), for plain formats and the two-plane
/// YCbCr family alike
#[test]
fn textures_over_planes_a_surface_lacks_are_metals() {
    use reims_vgpu_core::platform::Platform as _;
    let p = crate::platform::tests::platform();
    let md = reims_vgpu_metal::MetalPlatform::system_default_device();
    let size = (32, 16);
    for (format, ostype) in [
        (80, b"420v"),
        (70, b"420v"),
        (500, b"420v"),
        (503, b"444v"),
        (505, b"x420"),
        (543, b"x420"),
    ] {
        let desc = crate::gpu::ycbcr::measure::desc(format, size);
        let two = crate::gpu::ycbcr::measure::Shape::of(ostype).surface(size);
        let made = dev()
            .new_texture_on_surface(&p.texture_descriptor(&desc), Some(&two), 2)
            .unwrap();
        assert!(made.is_none(), "{format}: plane 2 of a two-plane surface");
        if let Some(md) = &md {
            let metals = reims_vgpu_metal::gpu::desc::texture_descriptor(&desc);
            let made = md.new_texture_on_surface(&metals, Some(&two), 2).unwrap();
            assert!(made.is_none(), "Metal: {format}");
        }
        let one = crate::os::tests::bgra_surface(size.0, size.1);
        for plane in [1, 2] {
            let site = catch_death(|| {
                dev().new_texture_on_surface(&p.texture_descriptor(&desc), Some(&one), plane)
            })
            .unwrap_err();
            assert_eq!(
                (site.kind, site.file),
                (
                    DeathKind::Abort,
                    "newTextureWithDescriptor:iosurface:plane:"
                ),
                "{format}: plane {plane} of a surface without planes"
            );
        }
    }
}

/// A texture takes an ID a live texture holds where Metal hands it out a second time, as
/// Metal's does: after the device's first command queue, a reservation
/// of textures 5 and 40 has Metal give out IDs 1 to 4, then 1 again (measured here, Metal's
/// side read first, before MoltenVK makes objects of its own on Metal's device). Each
/// texture's `uniqueIdentifier` is its ID here, and Metal's are compared. Each test runs in a
/// process of its own, so both platforms' IDs start fresh
#[test]
fn a_texture_takes_a_live_ones_id_where_metal_gives_it() {
    use reims_vgpu_core::platform::Platform;
    use reims_vgpu_core::platform::resource::Texture as _;
    use reims_vgpu_core::platform::values::TextureDesc;
    use reims_vgpu_metal::MetalPlatform;

    /// The IDs and `uniqueIdentifier`s of five textures made after a queue and the
    /// reservation, which stay live.
    fn five<P: Platform>(d: &P::Device, desc: &P::TextureDescriptor) -> Vec<(u64, u64)> {
        let _queue = d.new_command_queue().unwrap();
        d.reserve_resource_indices(0, &[5, 40]).unwrap();
        let textures: Vec<P::Texture> = (0..5)
            .map(|_| d.new_texture(OnFault::Return, desc).unwrap().unwrap())
            .collect();
        let ids = textures
            .iter()
            .map(|t| (t.gpu_resource_id(), t.unique_identifier()))
            .collect();
        std::mem::forget(textures);
        ids
    }
    let ids = |answers: &[(u64, u64)]| answers.iter().map(|&(id, _)| id).collect::<Vec<_>>();
    let theirs = MetalPlatform::system_default_device().map(|md| {
        let desc = reims_vgpu_metal::gpu::desc::texture_descriptor(&TextureDesc {
            pixel_format: 80,
            width: 4,
            height: 4,
            usage: 0x13,
            resource_options: Some(0),
            ..TextureDesc::default()
        });
        five::<MetalPlatform>(&md, &desc)
    });
    let desc = crate::gpu::resource::tests::bgra(4, 4, 0);
    let ours = five::<VulkanPlatform>(&dev(), &desc);
    assert_eq!(ids(&ours), [1, 2, 3, 4, 1]);
    assert!(ours.iter().all(|&(id, unique)| id == unique), "{ours:?}");
    if let Some(theirs) = theirs {
        assert_eq!(ids(&theirs), [1, 2, 3, 4, 1]);
        assert_eq!(ours, theirs);
    }
}

/// A two-plane extended-range texture of type cube or cube array is nil where Metal answers
/// nil: where it passes every check Metal was measured to make. One that fails one dies,
/// where Metal's validation asserts, and so does a memoryless one, which Metal makes, and a
/// multisample one, as every other type the runtime refuses for the format does: the runtime's
/// answers. Metal's side runs for the nil ones and the memoryless one: its assert ends the
/// process at the others
#[test]
fn two_plane_cubes_are_nil_where_metal_answers_nil() {
    use reims_vgpu_core::platform::Platform as _;
    use reims_vgpu_core::platform::types::Swizzle;
    use reims_vgpu_core::platform::values::TextureDesc;
    use reims_vgpu_metal::MetalPlatform;

    /// A cube of `RGB10A8_2P_XR10`, 16 x 16, one level, usage `ShaderRead | ShaderWrite`,
    /// storage `Private`, then `change`.
    fn cube(change: fn(&mut TextureDesc)) -> TextureDesc {
        let mut desc = TextureDesc {
            texture_type: 5,
            pixel_format: 550,
            width: 16,
            height: 16,
            usage: 3,
            resource_options: Some(0x20),
            ..TextureDesc::default()
        };
        change(&mut desc);
        desc
    }
    /// (B, G, R, A).
    const BGRA: Option<Swizzle> = Some(Swizzle {
        red: 4,
        green: 3,
        blue: 2,
        alpha: 5,
    });
    /// Usage `RenderTarget`, storage `Memoryless`.
    const MEMORYLESS: fn(&mut TextureDesc) = |d| (d.usage, d.resource_options) = (4, Some(0x30));
    type Case = (&'static str, fn(&mut TextureDesc));
    let nil: [Case; 18] = [
        ("550 cube", |_| {}),
        ("550 cube array", |d| d.texture_type = 6),
        ("551 cube", |d| d.pixel_format = 551),
        ("551 cube array", |d| {
            (d.pixel_format, d.texture_type) = (551, 6)
        }),
        ("shared", |d| d.resource_options = Some(0)),
        ("5 levels", |d| d.mipmap_level_count = 5),
        ("render target", |d| d.usage = 4),
        ("usage 0", |d| d.usage = 0),
        ("read swizzle", |d| (d.usage, d.swizzle) = (1, BGRA)),
        ("341 cubes", |d| (d.texture_type, d.array_length) = (6, 341)),
        ("side 1", |d| (d.width, d.height) = (1, 1)),
        ("side 16384", |d| (d.width, d.height) = (16384, 16384)),
        ("341 cubes of side 16384, 15 levels", |d| {
            (d.texture_type, d.array_length) = (6, 341);
            (d.width, d.height, d.mipmap_level_count) = (16384, 16384, 15);
        }),
        ("usage 0x17", |d| d.usage = 0x17),
        ("write only", |d| d.usage = 2),
        ("view swizzle", |d| (d.usage, d.swizzle) = (0x11, BGRA)),
        ("managed", |d| d.resource_options = Some(0x10)),
        ("untracked", |d| d.resource_options = Some(0x120)),
    ];
    let die: [Case; 21] = [
        ("16 x 8", |d| d.height = 8),
        ("depth 2", |d| d.depth = 2),
        ("4 samples", |d| d.sample_count = 4),
        ("a cube of 2", |d| d.array_length = 2),
        ("0 cubes", |d| (d.texture_type, d.array_length) = (6, 0)),
        ("342 cubes", |d| (d.texture_type, d.array_length) = (6, 342)),
        ("6 levels", |d| d.mipmap_level_count = 6),
        ("atomic", |d| d.usage = 0x21),
        ("written swizzle", |d| d.swizzle = BGRA),
        ("render target swizzle", |d| {
            (d.usage, d.swizzle) = (4, BGRA)
        }),
        ("read and render target swizzle", |d| {
            (d.usage, d.swizzle) = (5, BGRA)
        }),
        ("side 16385", |d| (d.width, d.height) = (16385, 16385)),
        ("16384 x 16385", |d| (d.width, d.height) = (16384, 16385)),
        ("0 levels", |d| d.mipmap_level_count = 0),
        ("0 samples", |d| d.sample_count = 0),
        ("a cube of 0", |d| d.array_length = 0),
        ("memoryless", MEMORYLESS),
        ("550 2D multisample", |d| {
            (d.texture_type, d.sample_count) = (4, 4)
        }),
        ("551 2D multisample array", |d| {
            (d.pixel_format, d.texture_type, d.sample_count) = (551, 8, 4);
        }),
        ("4:2:2 cube", |d| d.pixel_format = 240),
        ("4:2:2 3D", |d| (d.pixel_format, d.texture_type) = (252, 7)),
    ];
    let d = dev();
    let platform = crate::platform::tests::platform();
    let md = MetalPlatform::system_default_device();
    for (what, change) in nil {
        let desc = cube(change);
        if let Some(md) = &md {
            let metal = reims_vgpu_metal::gpu::desc::texture_descriptor(&desc);
            let made = md.new_texture(OnFault::Return, &metal).unwrap();
            assert!(made.is_none(), "Metal: {what}");
        }
        let ours = platform.texture_descriptor(&desc);
        let made = d.new_texture(OnFault::Return, &ours).unwrap();
        assert!(made.is_none(), "{what}");
    }
    for (what, change) in die {
        let ours = platform.texture_descriptor(&cube(change));
        let Err(site) = catch_death(|| d.new_texture(OnFault::Return, &ours)) else {
            panic!("{what} did not die");
        };
        assert_eq!(site.file, "newTextureWithDescriptor:", "{what}");
    }
    if let Some(md) = &md {
        let metal = reims_vgpu_metal::gpu::desc::texture_descriptor(&cube(MEMORYLESS));
        let made = md.new_texture(OnFault::Return, &metal).unwrap();
        assert!(made.is_some(), "Metal makes the memoryless one");
    }
}

/// A reservation of textures or sampler states that holds an ID of 2^32 or more is dropped
/// whole, and the next call is the reservation, as on Metal: a call after it counts, and one
/// after that is ignored. Plain IDs then follow Metal's pool: 2 reserved gives 1, then 3 on,
/// and 3 reserved gives 1, 2, then 4 on
#[test]
fn a_reservation_past_32_bits_leaves_it_to_the_next_call() {
    use reims_vgpu_core::platform::Platform;
    use reims_vgpu_core::platform::resource::{SamplerState as _, Texture as _};
    use reims_vgpu_core::platform::values::TextureDesc;
    use reims_vgpu_metal::MetalPlatform;

    fn ids<P: Platform>(d: &P::Device, texture: &P::TextureDescriptor) -> Vec<Vec<u64>> {
        for (kind, indices) in [
            (0, &[1 << 32, 3][..]),
            (0, &[2]),
            (0, &[4]),
            (1, &[1 << 32]),
            (1, &[3]),
        ] {
            d.reserve_resource_indices(kind, indices).unwrap();
        }
        // Samplers first: Metal's first texture takes a sampler ID of its own.
        let samplers: Vec<P::SamplerState> = (1..=3u8)
            .map(|lod| {
                let desc = SamplerDesc {
                    support_argument_buffers: Some(true),
                    lod_max_clamp: Some(f32::from(lod)),
                    ..SamplerDesc::default()
                };
                d.new_sampler_state(&desc).unwrap().unwrap()
            })
            .collect();
        let textures: Vec<P::Texture> = (0..3)
            .map(|_| d.new_texture(OnFault::Return, texture).unwrap().unwrap())
            .collect();
        vec![
            textures.iter().map(|t| t.gpu_resource_id()).collect(),
            samplers.iter().map(|s| s.gpu_resource_id()).collect(),
        ]
    }
    let expected = vec![vec![1, 3, 4], vec![1, 2, 4]];
    if let Some(md) = MetalPlatform::system_default_device() {
        let desc = reims_vgpu_metal::gpu::desc::texture_descriptor(&TextureDesc {
            pixel_format: 80,
            width: 4,
            height: 4,
            usage: 0x13,
            resource_options: Some(0),
            ..TextureDesc::default()
        });
        assert_eq!(ids::<MetalPlatform>(&md, &desc), expected);
    }
    let d = dev();
    let texture = crate::gpu::resource::tests::bgra(4, 4, 0);
    assert_eq!(ids::<VulkanPlatform>(&d, &texture), expected);
}

/// The state on `d` of `v_plain` and `f_plain` of `refused.metallib` whose `attachment`
/// (`colour`, `blended`, `depth` or `stencil`) is pixel format `format`, the fragment function
/// with a colour attachment alone; `blended`'s with blending enabled, its factors and operations
/// Metal's defaults.
fn attachment_state<P: reims_vgpu_core::platform::Platform>(
    d: &P::Device,
    attachment: &str,
    format: u32,
) -> Result<Option<P::RenderPipeline>, reims_vgpu_core::platform::types::NsErrorText> {
    use reims_vgpu_core::platform::resource::Library as _;
    use reims_vgpu_core::platform::values::FullRenderPipelineDesc;

    use crate::gpu::device::tests::colours_of;
    let refused = d
        .new_library(include_bytes!(concat!(
            env!("OUT_DIR"),
            "/refused.metallib"
        )))
        .unwrap()
        .unwrap()
        .unwrap();
    let f = |name| {
        refused
            .new_function(OnFault::Return, name)
            .unwrap()
            .unwrap()
    };
    let (vertex, fragment) = (f("v_plain"), f("f_plain"));
    let format = u64::from(format);
    let desc = match attachment {
        "colour" | "blended" => FullRenderPipelineDesc {
            colors: colours_of(&[&[
                (1, format.try_into().unwrap()),
                (2, u32::from(attachment == "blended")),
            ]]),
            ..FullRenderPipelineDesc::new(&vertex, Some(&fragment))
        },
        "depth" => FullRenderPipelineDesc {
            depth_pixel_format: format,
            ..FullRenderPipelineDesc::new(&vertex, None)
        },
        _ => FullRenderPipelineDesc {
            stencil_pixel_format: format,
            ..FullRenderPipelineDesc::new(&vertex, None)
        },
    };
    d.new_render_pipeline_state_full(&desc)
}

/// A colour attachment of 576 to 579, or of a bfloat format (26, 66, 116) unblended, which the
/// runtime makes, makes the state, as Metal's does. A bfloat colour attachment that blends
/// (blending enabled, of one and zero), which Metal's validation refuses and the runtime
/// refuses (`Unsupported` of `Format`), dies as the state is made, naming the runtime's
/// refusal; Metal's is not made here
#[test]
fn pipeline_formats_the_runtime_makes_are_metals() {
    let d = dev();
    let md = reims_vgpu_metal::MetalPlatform::system_default_device();
    for format in [576, 577, 578, 579, 26, 66, 116] {
        let made = attachment_state::<crate::platform::VulkanPlatform>(&d, "colour", format);
        assert!(matches!(made, Ok(Some(_))), "colour {format}");
        if let Some(md) = &md {
            let made = attachment_state::<reims_vgpu_metal::MetalPlatform>(md, "colour", format);
            assert!(matches!(made, Ok(Some(_))), "Metal: colour {format}");
        }
    }
    let sel = "newRenderPipelineStateWithDescriptor:error:";
    for format in [26, 66, 116] {
        *crate::answer::LAST_LINE.lock() = None;
        let ours = catch_death(|| {
            attachment_state::<crate::platform::VulkanPlatform>(&d, "blended", format)
                .map(|s| s.is_some())
        });
        let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
        let Err(site) = ours else {
            panic!("blended {format}: {ours:?}");
        };
        assert_eq!((site.kind, site.file), (DeathKind::Abort, sel), "{line}");
        assert!(
            line.contains("the Vulkan runtime refused it") && line.contains("feature: Format"),
            "blended {format}: {line}"
        );
    }
}

/// What a draw over 16 by 16 `ARGB8Unorm` (84) does on `p`'s device `d` under the state of
/// a whole render value whose colour attachment 0 is that format, blending zero over the
/// source colour and zero over the source alpha, adding (a multiply), of `v_plain` and
/// `f_plain` of `refused.metallib`: the command buffer's status and error code once it
/// completed.
fn rotated_blend_draw<P: reims_vgpu_core::platform::Platform>(p: &P, d: &P::Device) -> (u64, u64) {
    use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
    use reims_vgpu_core::platform::descriptor::RenderPassDescriptor as _;
    use reims_vgpu_core::platform::render::{RenderCmd, RenderEncoder as _};
    use reims_vgpu_core::platform::resource::Library as _;
    use reims_vgpu_core::platform::values::{ColorAttachmentSet, TextureDesc};

    use reims_vgpu_core::platform::values::FullRenderPipelineDesc;

    use crate::gpu::device::tests::colours_of;
    let library = d.new_library(include_bytes!(concat!(
        env!("OUT_DIR"),
        "/refused.metallib"
    )));
    let library = library.unwrap().unwrap().unwrap();
    let f = |name| {
        library
            .new_function(OnFault::Return, name)
            .unwrap()
            .unwrap()
    };
    let (vertex, fragment) = (f("v_plain"), f("f_plain"));
    // Colour 0: format 84, blending on, rgb source Zero, destination SourceColor, alpha
    // source Zero, destination SourceAlpha, both operations Add.
    let state = d
        .new_render_pipeline_state_full(&FullRenderPipelineDesc {
            colors: colours_of(&[&[(1, 84), (2, 1), (3, 0), (4, 2), (6, 0), (7, 4)]]),
            ..FullRenderPipelineDesc::new(&vertex, Some(&fragment))
        })
        .unwrap()
        .unwrap();
    let target = p.texture_descriptor(&TextureDesc {
        texture_type: 2,
        pixel_format: 84,
        width: 16,
        height: 16,
        usage: 4,
        resource_options: Some(0x20),
        ..TextureDesc::default()
    });
    let target = d.new_texture(OnFault::Die, &target).unwrap().unwrap();
    let pass = p.render_pass_descriptor();
    let attachment = ColorAttachmentSet {
        texture: Some(Some(&target)),
        ..ColorAttachmentSet::default()
    };
    pass.set_color_attachment(OnFault::Die, 0, &attachment)
        .unwrap();
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
    for cmd in [
        RenderCmd::SetRenderPipelineState {
            pipeline_state: Some(&state),
        },
        RenderCmd::DrawPrimitives {
            primitive_type: 3,
            vertex_start: 0,
            vertex_count: 3,
        },
        RenderCmd::EndEncoding,
    ] {
        e.encode(OnFault::Die, cmd).unwrap();
    }
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    let code = cb.error().map_or(0, |e| u64::try_from(e.code).unwrap());
    (cb.status().0, code)
}

/// A draw under a state whose blend into an `ARGB8Unorm` colour attachment is in none of the
/// forms the runtime measured its rotated blend to round as Metal's in (zero over the source
/// colour and zero over the source alpha, adding) dies at the draw, naming the runtime's
/// announcement (`rotated-blend-rounding`), where Metal draws it: the state is made, as
/// Metal's is
#[test]
fn an_unmeasured_rotated_blend_dies_at_the_draw() {
    let p = crate::platform::tests::platform();
    *crate::answer::LAST_LINE.lock() = None;
    let site = catch_death(|| rotated_blend_draw(&p, &dev())).unwrap_err();
    let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
    assert_eq!(
        (site.kind, site.file),
        (DeathKind::Abort, "drawPrimitives:vertexStart:vertexCount:"),
        "{line}"
    );
    let named = "(metal2vulkan-degrade-rotated-blend-rounding)";
    assert!(line.contains(named), "{line}");
    if let Some(md) = reims_vgpu_metal::MetalPlatform::system_default_device() {
        let mp = reims_vgpu_metal::MetalPlatform::new();
        assert_eq!(rotated_blend_draw(&mp, &md), (4, 0), "Metal");
    }
}

/// On a native device the same draw dies too, at the same op and naming the same kind: Apple's
/// blend arithmetic for the form is not measured, and its exact form (a measured one, or the blend
/// in the function) is no fact of the device, so the device relays nothing
/// (`answer::native::NATIVE_RELAYED`)
#[test]
fn an_unmeasured_rotated_blend_dies_on_a_native_device() {
    use crate::answer::native;
    let p = crate::platform::tests::platform();
    *crate::answer::LAST_LINE.lock() = None;
    let site = native::as_native(|| catch_death(|| rotated_blend_draw(&p, &dev()))).unwrap_err();
    let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
    assert_eq!(
        (site.kind, site.file),
        (DeathKind::Abort, "drawPrimitives:vertexStart:vertexCount:"),
        "{line}"
    );
    assert!(
        line.contains("(metal2vulkan-degrade-rotated-blend-rounding)"),
        "{line}"
    );
}

/// Whether `d` makes a no-copy buffer over two fresh pages after `case`: 0 leaves them
/// readable and writable, 1 takes all access from the second, 2 from both, 3 unmaps the
/// second, 4 leaves both read-only.
fn no_copy_made<P: reims_vgpu_core::platform::Platform>(d: &P::Device, case: u8) -> bool {
    use reims_vgpu_core::platform::os::vm::VmProt;
    let vm = OsVm::new();
    let page = vm.page_size();
    let at = vm.allocate_anywhere(2 * page).unwrap();
    let second = at.wrapping_add(page);
    match case {
        1 => vm.protect_max(second, page, VmProt(0)).unwrap(),
        2 => vm.protect_max(at, 2 * page, VmProt(0)).unwrap(),
        3 => assert!(vm.deallocate(second, page).is_success()),
        4 => vm.protect_max(at, 2 * page, VmProt::READ).unwrap(),
        _ => {}
    }
    // SAFETY: the pages stay as they are until the buffer is gone (dropped below), but for
    // the cases that break `HostRange::new`'s contract on purpose, to measure what is made
    // over them; nothing reads or writes them.
    let range = unsafe { HostRange::new(at, 2 * page) };
    let made = d
        .new_buffer_over_host_memory(OnFault::Return, range, 0, None)
        .unwrap()
        .is_some();
    let _ = vm.deallocate(at, 2 * page);
    made
}

/// A no-copy buffer over memory with a page not mapped, or mapped without access, is nil,
/// as the Metal platform answers; one over read-only pages is made, as Metal's is
#[test]
fn no_copy_buffers_over_pages_without_access_are_nil_as_metals() {
    let expected = [true, false, false, false, true];
    let ours = [0, 1, 2, 3, 4].map(|case| no_copy_made::<VulkanPlatform>(&dev(), case));
    assert_eq!(ours, expected);
    if let Some(md) = reims_vgpu_metal::MetalPlatform::system_default_device() {
        let theirs =
            [0, 1, 2, 3, 4].map(|case| no_copy_made::<reims_vgpu_metal::MetalPlatform>(&md, case));
        assert_eq!(theirs, expected);
    }
}

/// A queue [`test_queue`] keeps: its platform, its device's identity and the queue.
type TestQueue = (
    std::any::TypeId,
    reims_vgpu_core::platform::ObjId,
    Box<dyn std::any::Any>,
);

std::thread_local! {
    /// [`test_queue`]'s queues.
    static TEST_QUEUES: std::cell::RefCell<Vec<TestQueue>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// The test thread's one command queue of `d` on platform `P`, made at its first use, for
/// helpers a test calls once per configuration. A Metal queue made per call lives to the
/// test's end, since a dev build's autorelease pool keeps each committed command buffer and
/// a command buffer keeps its queue. Past 512 of a process's GPU command queues the kernel
/// fails the next one's creation ("exceeding limit (512), failing IOGPUCommandQueue
/// creation" in its log); a test past it saw Metal's render passes store nothing while the
/// same command buffer's compute pass ran and it completed without an error. This platform's
/// queues share the device's Vulkan queues and cost nothing; they are kept the same way so
/// that both sides of a comparison run alike.
pub(crate) fn test_queue<P: reims_vgpu_core::platform::Platform>(d: &P::Device) -> P::CommandQueue {
    let at = (std::any::TypeId::of::<P>(), d.id());
    TEST_QUEUES.with(|queues| {
        let mut queues = queues.borrow_mut();
        let kept = (queues.iter()).find(|(platform, device, _)| (*platform, *device) == at);
        if let Some((_, _, q)) = kept {
            return q.downcast_ref::<P::CommandQueue>().unwrap().clone_ref();
        }
        let q = d.new_command_queue().unwrap();
        queues.push((at.0, at.1, Box::new(q.clone_ref())));
        q
    })
}

/// A device of the system default's physical device, made with the platform's options but a
/// shader cache of its own, in a new empty directory under `~/.cache` that the second value
/// removes as it drops: no record of an earlier process's makings reaches its states, so what
/// they make at their draws is the same on every run. Not
/// kept for [`adopt`]: the process's other handles answer the shared device.
pub(crate) fn device_with_own_cache() -> (VulkanDevice, OwnCache) {
    let (options, cache) = options_with_own_cache();
    let raw = rt::Device::system_default_with(&options).unwrap();
    let device = Obj::new(DeviceObject {
        registry_id: registry_id(&raw),
        raw,
        caps: OnceLock::new(),
        awake: Default::default(),
    });
    (device, cache)
}

/// The spec of a 'b3a8' surface of `size` (`kCVPixelFormatType_30RGBLE_8A_BiPlanar`): plane 0
/// the 32-bit colour words, plane 1 the alpha bytes, rows `rows` bytes apart in each, plane 1
/// after plane 0 at a multiple of 4096.
fn b3a8_spec(
    size: (u64, u64),
    rows: (u64, u64),
) -> reims_vgpu_core::platform::os::surface::SurfaceSpec {
    let alpha_offset = (rows.0 * size.1).next_multiple_of(4096);
    b3a8_spec_at(size, rows, alpha_offset)
}

/// [`b3a8_spec`] with plane 1 at `alpha_offset`.
fn b3a8_spec_at(
    size: (u64, u64),
    rows: (u64, u64),
    alpha_offset: u64,
) -> reims_vgpu_core::platform::os::surface::SurfaceSpec {
    use reims_vgpu_core::platform::os::surface::{SurfacePlaneSpec, SurfaceSpec};
    let (width, height) = size;
    let (color_row, alpha_row) = rows;
    let plane = |offset, bytes_per_row, bytes_per_element| SurfacePlaneSpec {
        base: offset,
        offset,
        width,
        height,
        size: bytes_per_row * height,
        bytes_per_row,
        bytes_per_element,
        element_width: 1,
        element_height: 1,
        ..SurfacePlaneSpec::default()
    };
    SurfaceSpec {
        planes: Some(vec![
            plane(0, color_row, 4),
            plane(alpha_offset, alpha_row, 1),
        ]),
        pixel_format: u32::from_be_bytes(*b"b3a8"),
        alloc_size: (alpha_offset + alpha_row * height).next_multiple_of(16384),
        width,
        height,
        bytes_per_element: 4,
        bytes_per_row: color_row,
        element_width: 1,
        element_height: 1,
        ..SurfaceSpec::default()
    }
}

/// The surface of `spec`, every byte of its allocation `noise(seed)`'s.
fn noisy_surface(
    spec: &reims_vgpu_core::platform::os::surface::SurfaceSpec,
    seed: u64,
) -> crate::os::OsSurface {
    use reims_vgpu_core::platform::os::surface::{IoSurfaces as _, Surface as _};
    let s = crate::os::OsSurfaces::new().create(spec).unwrap();
    let mut next = crate::gpu::ycbcr::measure::noise(seed);
    let base = s.base_address().as_ptr::<u8>();
    for at in 0..usize::try_from(spec.alloc_size).unwrap() {
        // SAFETY: inside the surface's allocation, which no GPU work uses yet.
        unsafe { base.add(at).write(next().to_le_bytes()[0]) };
    }
    s
}

/// A host-only 'b3a8' surface of [`b3a8_spec`], every byte `noise(seed)`'s.
fn b3a8(size: (u64, u64), rows: (u64, u64), seed: u64) -> crate::os::OsSurface {
    noisy_surface(&b3a8_spec(size, rows), seed)
}

/// How [`read_through_another`] orders its two blits: in two encoders of one command buffer, or
/// in command buffers of two queues, the read's committed first and waiting for an event the
/// write's signals after its blit.
#[derive(Clone, Copy, Debug)]
enum Order {
    OneCommandBuffer,
    EventAcrossQueues,
}

/// What a texture over `surface` (a 'BGRA' one of `size`) reads where another texture over the
/// same surface was written before it, on `p`'s device `d`: a blit of `texels` into the first
/// texture, then a blit of the second texture's texels into a buffer, ordered as `order` says,
/// the buffer's bytes answered once both command buffers completed. The two textures are made
/// first, as a guest makes two over one surface.
fn read_through_another<P: reims_vgpu_core::platform::Platform<Surface = crate::os::OsSurface>>(
    p: &P,
    d: &P::Device,
    surface: &crate::os::OsSurface,
    ((width, height), order): ((u64, u64), Order),
    texels: &[u8],
) -> Vec<u8> {
    use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder as _};
    use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
    use reims_vgpu_core::platform::event::EventRef;
    use reims_vgpu_core::platform::resource::Buffer as _;
    use reims_vgpu_core::platform::types::{MtlOrigin, MtlSize};
    let desc = p.texture_descriptor(&reims_vgpu_core::platform::values::TextureDesc {
        texture_type: 2,
        pixel_format: 80,
        width,
        height,
        usage: 1,
        ..reims_vgpu_core::platform::values::TextureDesc::default()
    });
    let written = (d.new_texture_on_surface(&desc, Some(surface), 0))
        .unwrap()
        .unwrap();
    let reader = (d.new_texture_on_surface(&desc, Some(surface), 0))
        .unwrap()
        .unwrap();
    let length = u64::try_from(texels.len()).unwrap();
    let buffer = || {
        (d.new_buffer_of_length(OnFault::Die, length, 0))
            .unwrap()
            .unwrap()
    };
    let (up, down) = (buffer(), buffer());
    // SAFETY: inside a Shared buffer of `length` bytes the GPU does not use yet.
    unsafe {
        std::ptr::copy_nonoverlapping(texels.as_ptr(), up.contents().as_ptr(), texels.len());
    }
    let (size, origin) = (MtlSize::new(width, height, 1), MtlOrigin::new(0, 0, 0));
    let write = BlitCmd::CopyBufferToTexture {
        source_buffer: &up,
        source_offset: 0,
        source_bytes_per_row: width * 4,
        source_bytes_per_image: length,
        source_size: size,
        destination_texture: &written,
        destination_slice: 0,
        destination_level: 0,
        destination_origin: origin,
    };
    let read = BlitCmd::CopyTextureToBuffer {
        source_texture: &reader,
        source_slice: 0,
        source_level: 0,
        source_origin: origin,
        source_size: size,
        destination_buffer: &down,
        destination_offset: 0,
        destination_bytes_per_row: width * 4,
        destination_bytes_per_image: length,
    };
    let blit = |cb: &P::CommandBuffer, cmd| {
        let e = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
        e.encode(OnFault::Die, cmd).unwrap();
        e.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
    };
    let q = d.new_command_queue().unwrap();
    let cbs = match order {
        Order::OneCommandBuffer => {
            let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
            blit(&cb, write);
            blit(&cb, read);
            cb.commit(OnFault::Die).unwrap();
            vec![cb]
        }
        Order::EventAcrossQueues => {
            let event = d.new_event(OnFault::Die).unwrap().unwrap();
            let other = d.new_command_queue().unwrap();
            let reading = other.command_buffer(OnFault::Die).unwrap().unwrap();
            reading
                .encode_wait(Some(EventRef::Event(&event)), 1)
                .unwrap();
            blit(&reading, read);
            reading.commit(OnFault::Die).unwrap();
            let writing = q.command_buffer(OnFault::Die).unwrap().unwrap();
            blit(&writing, write);
            writing
                .encode_signal(Some(EventRef::Event(&event)), 1)
                .unwrap();
            writing.commit(OnFault::Die).unwrap();
            vec![writing, reading]
        }
    };
    for cb in &cbs {
        cb.wait_until_completed(OnFault::Die).unwrap();
        assert_eq!(cb.error(), None);
    }
    // SAFETY: the buffer's `length` bytes, read once the command buffer that wrote them completed.
    unsafe { std::slice::from_raw_parts(down.contents().as_ptr::<u8>(), texels.len()) }.to_vec()
}

/// The texels [`read_through_another`] writes: 64 by 64 'BGRA' texels, no byte 0.
fn surface_texels() -> ((u64, u64), Vec<u8>) {
    let size = (64, 64);
    let texels = (0..64 * 64 * 4_u32)
        .map(|i| u8::try_from(i * 7 % 255 + 1).unwrap())
        .collect();
    (size, texels)
}

/// A texture over a surface reads what a blit through another texture over the same surface
/// wrote before it, as Metal's does, in a later encoder of one command buffer and in a command
/// buffer of another queue that waits for an event the write signals: two textures over one
/// surface are one memory, which the runtime imports once for both, the surface over pages of
/// the host's as a guest's mapper makes it
#[test]
fn a_texture_over_a_surface_reads_another_ones_writes_as_metals() {
    use crate::os::tests::{bgra_surface_over_at, free};
    let Some(md) = reims_vgpu_metal::MetalPlatform::system_default_device() else {
        return;
    };
    let mp = reims_vgpu_metal::MetalPlatform::new();
    let p = crate::platform::tests::platform();
    let (size, texels) = surface_texels();
    for order in [Order::OneCommandBuffer, Order::EventAcrossQueues] {
        let (theirs, pages, length) = bgra_surface_over_at(size.0, size.1, 0);
        let metals = read_through_another(&mp, &md, &theirs, (size, order), &texels);
        assert!(metals == texels, "{order:?}: Metal's");
        drop(theirs);
        free(pages, length);
        let (ours, pages, length) = bgra_surface_over_at(size.0, size.1, 0);
        let read = read_through_another(&p, &dev(), &ours, (size, order), &texels);
        assert!(read == metals, "{order:?}: in place");
        drop(ours);
        free(pages, length);
    }
}

/// [`a_texture_over_a_surface_reads_another_ones_writes_as_metals`] where every import is
/// refused, as Linux's views' gate refuses them past the headroom of the views' budget
/// (`os/linux/import.rs`), so the runtime makes each buffer over the surface over a copy of its
/// pages (`copied-host-pages`): the texture read must still see the other's writes, as Metal's
/// does
#[test]
fn a_texture_over_a_copied_surface_reads_another_ones_writes_as_metals() {
    use crate::os::tests::{bgra_surface_over_at, free};
    let Some(md) = reims_vgpu_metal::MetalPlatform::system_default_device() else {
        return;
    };
    let mp = reims_vgpu_metal::MetalPlatform::new();
    let p = crate::platform::tests::platform();
    let (size, texels) = surface_texels();
    let copies = || {
        (crate::answer::relayed_counts().into_iter())
            .find(|(kind, _)| *kind == rt::Degradation::CopiedHostPages)
            .map_or(0, |(_, n)| n)
    };
    let mut stale = Vec::new();
    for order in [Order::OneCommandBuffer, Order::EventAcrossQueues] {
        let (theirs, pages, length) = bgra_surface_over_at(size.0, size.1, 0);
        let metals = read_through_another(&mp, &md, &theirs, (size, order), &texels);
        drop(theirs);
        free(pages, length);
        let before = copies();
        let (ours, pages, length) = bgra_surface_over_at(size.0, size.1, 0);
        let read = crate::answer::relaxed(|| {
            crate::os::refusing_imports(|| {
                read_through_another(&p, &dev(), &ours, (size, order), &texels)
            })
        });
        assert!(
            copies() > before,
            "{order:?}: the runtime copied the surface's pages"
        );
        let differ = read.iter().zip(&metals).filter(|(a, b)| a != b).count();
        stale.push((order, differ, metals.len()));
        drop(ours);
        free(pages, length);
    }
    assert!(
        stale.iter().all(|&(_, differ, _)| differ == 0),
        "bytes that differ from Metal's, of all: {stale:?}"
    );
}
