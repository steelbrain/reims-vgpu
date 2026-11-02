//! The tests of `states` that compare with Metal on this host: macOS only.

use super::*;

/// A format the functions cannot write is the NSError Metal hands back; so is a nil
/// descriptor, of Metal's code and text for one at each creator (compared on the host) in the
/// runtime's domain
#[test]
fn guest_descriptor_refusals() {
    let d = dev();
    let functions = display_functions(&d);
    let integer = guest_descriptor(&functions, PixelFormat::R32Uint, None);
    let error = d
        .new_render_pipeline_state_from_descriptor(Some(&integer), 0)
        .unwrap()
        .unwrap_err();
    assert_eq!(error.domain, "metal2vulkan-runtime");
    let nil = d
        .new_render_pipeline_state_from_descriptor(None, 0)
        .unwrap()
        .unwrap_err();
    assert_eq!(nil, nil_pipeline_descriptor());
    let nil = d
        .new_tile_render_pipeline_state(None, 0)
        .unwrap()
        .unwrap_err();
    assert_eq!(nil, nil_pipeline_descriptor());
    assert_eq!(nil.domain, crate::answer::RUNTIME_DOMAIN);
    if let Some(md) = reims_vgpu_metal::MetalPlatform::system_default_device() {
        let errors = [
            md.new_compute_pipeline_state(None, 0).unwrap().err(),
            md.new_render_pipeline_state_from_descriptor(None, 0)
                .unwrap()
                .err(),
            md.new_tile_render_pipeline_state(None, 0).unwrap().err(),
        ];
        for error in errors {
            let error = error.unwrap();
            assert_eq!(error.code, nil.code, "{error}");
            assert!(error.description.contains(&nil.description), "{error}");
        }
    }
}

/// `numbered_samplers.metal`'s functions, compiled by build.rs.
const NUMBERED: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/numbered_samplers.metallib"));

/// `f`'s value, made inside an autorelease pool of its own, as core runs each unit of work:
/// what Metal autoreleases inside it goes at its end.
fn pooled<T>(f: impl FnOnce() -> T) -> T {
    let _pool = reims_vgpu_core::rt::Pool::here();
    f()
}

/// The sampler IDs on `P`'s device `d` as pipeline states, their function objects and their
/// libraries of `numbered_samplers.metal` come and go, after a command queue, which keeps the
/// device's own sampler ID for good: at each step a sampler state of a descriptor no other
/// equals is made and kept to the end, so it takes the lowest ID no holder holds, and its
/// `gpuResourceID` is read. Each creation and release runs in a pool of its own.
fn numbered_ids<P: reims_vgpu_core::platform::Platform>(d: &P::Device) -> Vec<(&'static str, u64)> {
    use reims_vgpu_core::platform::values::RenderPipelineDesc;
    let _queue = pooled(|| d.new_command_queue().unwrap());
    let mut kept: Vec<P::SamplerState> = Vec::new();
    let mut ids = Vec::new();
    let mut mark = |step: &'static str| {
        let lod = f32::from(u8::try_from(kept.len() + 1).unwrap());
        let desc = SamplerDesc {
            support_argument_buffers: Some(true),
            lod_max_clamp: Some(lod),
            ..SamplerDesc::default()
        };
        let s = pooled(|| d.new_sampler_state(&desc).unwrap().unwrap());
        ids.push((step, s.gpu_resource_id()));
        kept.push(s);
    };
    let library = || pooled(|| d.new_library(NUMBERED).unwrap().unwrap().unwrap());
    let function =
        |l: &P::Library, name| pooled(|| l.new_function(OnFault::Die, name).unwrap().unwrap());
    let kernel = |f: &P::Function| {
        pooled(|| {
            d.new_compute_pipeline_state_for_function(Some(f))
                .unwrap()
                .unwrap()
        })
    };
    let gone = |object: Box<dyn std::any::Any>| pooled(|| drop(object));
    mark("start");
    // A kernel's constexpr sampler: held by its state, its function object and its library.
    let l = library();
    let f = function(&l, "numbered_sample");
    mark("constexpr: a library and a function");
    let s = kernel(&f);
    mark("constexpr: a state");
    gone(Box::new(s));
    mark("constexpr: the state gone");
    gone(Box::new(f));
    mark("constexpr: the function gone");
    gone(Box::new(l));
    mark("constexpr: the library gone");
    // A texture read: a function object holds it past its library.
    let l = library();
    let f = function(&l, "numbered_read");
    let s = kernel(&f);
    gone(Box::new(l));
    mark("read: the state and the function, the library gone");
    gone(Box::new(s));
    mark("read: the function alone");
    gone(Box::new(f));
    mark("read: all gone");
    // A function object made after the state holds what it numbered.
    let l = library();
    let first = function(&l, "numbered_read");
    let s = kernel(&first);
    gone(Box::new(first));
    let later = function(&l, "numbered_read");
    gone(Box::new(s));
    gone(Box::new(l));
    mark("read: a function object made after the state");
    gone(Box::new(later));
    mark("read: that one gone");
    // A texture write: a library of the same bytes holds it.
    let (a, b) = (library(), library());
    let f = function(&a, "numbered_write");
    let s = kernel(&f);
    gone(Box::new(s));
    gone(Box::new(f));
    gone(Box::new(a));
    mark("write: another library of its bytes");
    gone(Box::new(b));
    mark("write: all gone");
    // A render state whose fragment function reads a texture.
    let l = library();
    let (v, fr) = (
        function(&l, "numbered_vertex"),
        function(&l, "numbered_fragment_read"),
    );
    let desc = RenderPipelineDesc {
        label: None,
        vertex: &v,
        fragment: &fr,
        color0_pixel_format: 80,
        blend_over: false,
        resource_index: None,
    };
    let r = pooled(|| d.new_render_pipeline_state(&desc).unwrap().unwrap());
    mark("render: a state");
    gone(Box::new(r));
    gone(Box::new(v));
    mark("render: the fragment function and the library");
    gone(Box::new(fr));
    gone(Box::new(l));
    mark("render: all gone");
    ids
}

/// Sampler states take the IDs Metal's do while pipeline states number their functions'
/// samplers: a constexpr sampler, a texture read and a texture write of a kernel, a fragment
/// function's texture read, each held by the state, by every function object of its function
/// (made before the state or after it, past its library) and by every live library of the same
/// bytes, and given back with its last holder, as Metal's are on this host. This platform's
/// objects hold the runtime's libraries and functions exactly as long as the guest's handles
/// to them live, which this checks. Metal's side runs first, before MoltenVK makes objects of
/// its own on Metal's device
#[test]
fn sampler_ids_follow_pipeline_holders_as_metals() {
    use reims_vgpu_metal::MetalPlatform;
    reims_vgpu_metal::os::autorelease::install();
    let theirs =
        MetalPlatform::system_default_device().map(|md| numbered_ids::<MetalPlatform>(&md));
    let ours = numbered_ids::<VulkanPlatform>(&dev());
    if let Some(theirs) = theirs {
        assert_eq!(ours, theirs);
    }
}

/// `five_samplers.metal`'s kernel, compiled by build.rs.
const FIVE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/five_samplers.metallib"));

/// How far the next range moves on `P`'s device `d` as a library of `five_samplers.metal`, its
/// kernel's function and its state are made, after a command queue, which takes the device's
/// own sampler ID, and 250 sampler states of argument buffer support, each of its own LOD
/// clamp, kept to the end. Each creation and release runs in a pool of its own.
fn five_samplers_move<P: reims_vgpu_core::platform::Platform>(d: &P::Device) -> u64 {
    use crate::gpu::resource::tests::metal_tests::next_range;
    let _queue = pooled(|| d.new_command_queue().unwrap());
    let states: Vec<P::SamplerState> = (0..250u16)
        .map(|n| {
            let desc = SamplerDesc {
                support_argument_buffers: Some(true),
                lod_max_clamp: Some(1000.0 + f32::from(n)),
                ..SamplerDesc::default()
            };
            pooled(|| d.new_sampler_state(&desc).unwrap().unwrap())
        })
        .collect();
    let before = pooled(|| next_range::<P>(d));
    let library = pooled(|| d.new_library(FIVE).unwrap().unwrap().unwrap());
    let function = pooled(|| {
        library
            .new_function(OnFault::Die, "five_samplers")
            .unwrap()
            .unwrap()
    });
    let state = pooled(|| {
        d.new_compute_pipeline_state_for_function(Some(&function))
            .unwrap()
            .unwrap()
    });
    let after = pooled(|| next_range::<P>(d));
    pooled(|| drop((state, function, library, states)));
    after.wrapping_sub(before)
}

/// The samplers a pipeline state numbers take descriptors in Metal's sampler mempool as
/// sampler states do (`metal_space`'s `Samplers`): after the device's own and 250 sampler
/// states, a kernel's five constexpr samplers bring the IDs to 256, which doubles the pool to
/// 512 and grows the mempool's array to a page, a range of its own, so the next range moves
/// on by 0x8000 ([`five_samplers_move`]), on Metal, where it runs, and here, where the
/// runtime's pool holds 512 IDs. Metal's side runs first, before MoltenVK makes objects of its
/// own on Metal's device
#[test]
fn pipeline_samplers_grow_metals_sampler_mempool() {
    use reims_vgpu_metal::MetalPlatform;
    reims_vgpu_metal::os::autorelease::install();
    let theirs =
        MetalPlatform::system_default_device().map(|md| five_samplers_move::<MetalPlatform>(&md));
    let d = dev();
    let ours = five_samplers_move::<VulkanPlatform>(&d);
    assert_eq!(
        (ours, plain(|| d.raw.sampler_id_pool_size())),
        (0x8000, 512)
    );
    if let Some(theirs) = theirs {
        assert_eq!(theirs, 0x8000, "Metal's");
    }
}

/// A 16 by 8 `RGBA16Float` texture on `p`'s device `d`, its halves `seed`'s noise of every
/// sign below 2 in magnitude, written by the CPU.
fn noise_texture<P: reims_vgpu_core::platform::Platform>(
    p: &P,
    d: &P::Device,
    seed: u64,
) -> P::Texture {
    use reims_vgpu_core::platform::HostRange;
    use reims_vgpu_core::platform::resource::Texture as _;
    use reims_vgpu_core::platform::types::{MtlOrigin, MtlRegion, MtlSize};
    let (w, h) = (16, 8);
    let t = d
        .new_texture(
            OnFault::Die,
            &p.texture_descriptor(&reims_vgpu_core::platform::values::TextureDesc {
                pixel_format: 115,
                width: w,
                height: h,
                usage: 1,
                resource_options: Some(0),
                ..reims_vgpu_core::platform::values::TextureDesc::default()
            }),
        )
        .unwrap()
        .unwrap();
    let mut next = crate::gpu::ycbcr::measure::noise(seed);
    let halves: Vec<u8> = (0..w * h * 4)
        // A sign, an exponent field below 16 (a magnitude below 2) and ten bits of mantissa.
        .flat_map(|_| (next() & 0xbfff).to_le_bytes())
        .collect();
    let all = MtlRegion::new(MtlOrigin::new(0, 0, 0), MtlSize::new(w, h, 1));
    t.replace_region(all, 0, 0, HostRange::of_slice(&halves), w * 8, 0)
        .unwrap();
    t
}

/// What `sample()` through a sampler of `desc` reads of [`noise_texture`] on `p`'s device
/// `d`, into floats, as their bits, by `kernel` (`sample_only_texels_float`, or
/// `sample_texels_float`, which holds a gather too): at positions a third of a texel apart
/// from two texels before the texture to two past it, normalized.
fn cubic_samples<P: reims_vgpu_core::platform::Platform>(
    p: &P,
    d: &P::Device,
    desc: &SamplerDesc,
    kernel: &str,
) -> Vec<[u32; 4]> {
    use crate::gpu::ycbcr::measure::{float_texels, probe, probed_texels};
    let t = noise_texture(p, d, 5);
    let s = d.new_sampler_state(desc).unwrap().unwrap();
    let grid = ((16 + 4) * 3, (8 + 4) * 3);
    let at = probe((-2.0, -2.0), 1.0 / 3.0, grid, true, 0);
    let read = probed_texels::<P>(d, (kernel, 16), &t, grid, Some(&s), Some(&at));
    (float_texels(&read).into_iter())
        .map(|t| t.map(f32::to_bits))
        .collect()
}

/// The reader [`cubic_samples`] compares with: it holds no gather, which the runtime refuses
/// through a cubic sampler under mirrored repeat
/// ([`cubic_samplers_the_runtime_refuses_die_where_they_sample`]).
const SAMPLE_ONLY: &str = "sample_only_texels_float";

/// A sampler of `min` and `mag` filters (0 nearest, 1 linear, 2 cubic), no mip filter,
/// normalized coordinates, under `address` on both axes.
fn filtered(min: u64, mag: u64, address: u64) -> SamplerDesc {
    SamplerDesc {
        min_filter: Some(min),
        mag_filter: Some(mag),
        mip_filter: Some(0),
        s_address_mode: Some(address),
        t_address_mode: Some(address),
        normalized_coordinates: Some(true),
        ..SamplerDesc::default()
    }
}

/// A sampler whose min or mag filter is cubic (`MTLSamplerMinMagFilterCubic`), without a mip
/// filter, samples a texture as Metal's does, bit for bit, into floats: every pair of
/// nearest, linear and cubic with cubic in it, and nearest and linear alone, under clamp to
/// the edge, repeat, mirrored repeat, clamp to zero and clamp to the border colour (each of
/// transparent black, opaque black and opaque white), inside the texture and outside it, by a
/// reader that samples alone; and so does a cubic one with a compare function, which a plain
/// sample does not compare
///
/// Where the filter is cubic, this platform's sample dies at the dispatch naming
/// `bicubic-filter-arithmetic` ([`crate::answer::DIES`]), which the runtime announces at every
/// bicubic sample whose sums it has not pinned to Metal's rule for every format; Metal's side of
/// each case runs
#[test]
fn cubic_samplers_sample_as_metals() {
    use reims_vgpu_metal::MetalPlatform;
    let (p, d) = (crate::platform::tests::platform(), dev());
    let mp = MetalPlatform::new(Box::new(crate::gpu::device::tests::metal_tests::NoLauncher));
    let Some(md) = MetalPlatform::system_default_device() else {
        return;
    };
    let kind = metal2vulkan_runtime::Degradation::BicubicFilterArithmetic;
    let pairs = [(2, 2), (2, 0), (2, 1), (0, 2), (1, 2), (0, 0), (1, 1)];
    let modes = [(0, 0), (2, 0), (3, 0), (4, 0), (5, 0), (5, 1), (5, 2)];
    for (min, mag) in pairs {
        for (address, border) in modes {
            let desc = SamplerDesc {
                border_color: Some(border),
                ..filtered(min, mag, address)
            };
            let theirs = cubic_samples(&mp, &md, &desc, SAMPLE_ONLY);
            let case = format!("min {min} mag {mag} address {address} border {border}");
            if min == 2 || mag == 2 {
                let site =
                    crate::answer::dies_of(kind, || cubic_samples(&p, &d, &desc, SAMPLE_ONLY));
                assert_eq!(
                    site.file, "dispatchThreads:threadsPerThreadgroup:",
                    "{case}"
                );
                continue;
            }
            let ours = cubic_samples(&p, &d, &desc, SAMPLE_ONLY);
            let differ = ours.iter().zip(&theirs).filter(|(a, b)| a != b).count();
            assert_eq!(differ, 0, "{case}");
        }
    }
    for compare in [1, 3, 7] {
        let desc = SamplerDesc {
            compare_function: Some(compare),
            ..filtered(2, 2, 0)
        };
        cubic_samples(&mp, &md, &desc, SAMPLE_ONLY);
        let site = crate::answer::dies_of(kind, || cubic_samples(&p, &d, &desc, SAMPLE_ONLY));
        let case = format!("compare function {compare}");
        assert_eq!(
            site.file, "dispatchThreads:threadsPerThreadgroup:",
            "{case}"
        );
    }
}

/// A cubic sampler samples in a fragment function as Metal's does, bit for bit, into floats
/// (`sample_fragment_float`, one fragment a position, at [`cubic_samples`]'s positions),
/// under clamp to the edge, repeat and clamp to an opaque white border
///
/// This platform's draw dies naming `bicubic-filter-arithmetic` ([`crate::answer::DIES`]),
/// which the runtime announces at every bicubic sample whose sums it has not pinned to Metal's
/// rule for every format; Metal's side of each case runs
#[test]
fn cubic_samplers_sample_in_a_fragment_function_as_metals() {
    use crate::gpu::ycbcr::measure::{drawn, float_texels, probe};
    use reims_vgpu_metal::MetalPlatform;
    /// What `sample_fragment_float` reads of [`noise_texture`] through a sampler of `desc`.
    fn fragment_samples<P: reims_vgpu_core::platform::Platform>(
        p: &P,
        d: &P::Device,
        desc: &SamplerDesc,
    ) -> Vec<[u32; 4]> {
        let t = noise_texture(p, d, 5);
        let s = d.new_sampler_state(desc).unwrap().unwrap();
        let grid = ((16 + 4) * 3, (8 + 4) * 3);
        let at = probe((-2.0, -2.0), 1.0 / 3.0, grid, true, 0);
        let read = drawn::<P>(p, d, ("sample_fragment_float", 16), &t, grid, &s, &at);
        (float_texels(&read).into_iter())
            .map(|t| t.map(f32::to_bits))
            .collect()
    }
    let (p, d) = (crate::platform::tests::platform(), dev());
    let mp = MetalPlatform::new(Box::new(crate::gpu::device::tests::metal_tests::NoLauncher));
    let Some(md) = MetalPlatform::system_default_device() else {
        return;
    };
    for (address, border) in [(0, 0), (2, 0), (5, 2)] {
        let desc = SamplerDesc {
            border_color: Some(border),
            ..filtered(2, 2, address)
        };
        fragment_samples(&mp, &md, &desc);
        let kind = metal2vulkan_runtime::Degradation::BicubicFilterArithmetic;
        let site = crate::answer::dies_of(kind, || fragment_samples(&p, &d, &desc));
        let case = format!("address {address} border {border}");
        assert_eq!(
            site.file, "drawPrimitives:vertexStart:vertexCount:",
            "{case}"
        );
    }
}

/// A constexpr sampler of `filter::bicubic` samples as Metal's does, bit for bit, into floats,
/// under clamp to the edge and repeat (`sample_constexpr_bicubic`,
/// `sample_constexpr_repeat_bicubic`), at positions a third of a texel apart from four texels
/// before the texture to four past it
///
/// This platform's dispatch dies naming `bicubic-filter-arithmetic` ([`crate::answer::DIES`]),
/// which the runtime announces at every bicubic sample whose sums it has not pinned to Metal's
/// rule for every format; Metal's side of each kernel runs
#[test]
fn constexpr_bicubic_samplers_sample_as_metals() {
    use crate::gpu::ycbcr::measure::{float_texels, probe, probed_texels};
    use reims_vgpu_metal::MetalPlatform;
    /// What `kernel` reads of [`noise_texture`] at the probe `at` over `grid`.
    fn constexpr_samples<P: reims_vgpu_core::platform::Platform>(
        p: &P,
        d: &P::Device,
        kernel: &str,
        (grid, at): ((u64, u64), &[u8]),
    ) -> Vec<[u32; 4]> {
        let t = noise_texture(p, d, 7);
        let read = probed_texels::<P>(d, (kernel, 16), &t, grid, None, Some(at));
        (float_texels(&read).into_iter())
            .map(|t| t.map(f32::to_bits))
            .collect()
    }
    let (p, d) = (crate::platform::tests::platform(), dev());
    let mp = MetalPlatform::new(Box::new(crate::gpu::device::tests::metal_tests::NoLauncher));
    let Some(md) = MetalPlatform::system_default_device() else {
        return;
    };
    let grid = ((16 + 8) * 3, (8 + 8) * 3);
    let at = probe((-4.0, -4.0), 1.0 / 3.0, grid, true, 0);
    for kernel in [
        "sample_constexpr_bicubic",
        "sample_constexpr_repeat_bicubic",
    ] {
        constexpr_samples(&mp, &md, kernel, (grid, &at));
        let kind = metal2vulkan_runtime::Degradation::BicubicFilterArithmetic;
        let site = crate::answer::dies_of(kind, || constexpr_samples(&p, &d, kernel, (grid, &at)));
        assert_eq!(
            site.file, "dispatchThreads:threadsPerThreadgroup:",
            "{kernel}"
        );
    }
}

/// A dispatch that samples through a cubic sampler the runtime refuses dies there, naming the
/// refusal (`SamplerBicubicFilter`), where Metal samples: with a mip filter, under mirror
/// clamp to the edge, with a minimum or a maximum reduction, and with anisotropy (through an
/// argument buffer, its command buffer fails and dies:
/// [`a_cubic_sampler_in_an_argument_buffer_dies_where_its_failure_is_read`]); and a dispatch of
/// a function that also gathers through a cubic sampler under mirrored repeat, which it
/// samples through alone ([`cubic_samplers_sample_as_metals`]). The sampler state is made
#[test]
fn cubic_samplers_the_runtime_refuses_die_where_they_sample() {
    use reims_vgpu_metal::MetalPlatform;
    let (p, d) = (crate::platform::tests::platform(), dev());
    let mp = MetalPlatform::new(Box::new(crate::gpu::device::tests::metal_tests::NoLauncher));
    let md = MetalPlatform::system_default_device();
    let refused: [(&str, SamplerDesc, &str); 6] = [
        (
            "mip filter",
            SamplerDesc {
                mip_filter: Some(2),
                ..filtered(2, 2, 0)
            },
            SAMPLE_ONLY,
        ),
        ("mirror clamp", filtered(2, 2, 1), SAMPLE_ONLY),
        (
            "minimum reduction",
            SamplerDesc {
                reduction_mode: Some(1),
                ..filtered(2, 2, 0)
            },
            SAMPLE_ONLY,
        ),
        (
            "maximum reduction",
            SamplerDesc {
                reduction_mode: Some(2),
                ..filtered(2, 2, 0)
            },
            SAMPLE_ONLY,
        ),
        (
            "anisotropy",
            SamplerDesc {
                max_anisotropy: Some(4),
                ..filtered(2, 2, 0)
            },
            SAMPLE_ONLY,
        ),
        (
            "mirrored repeat beside a gather",
            filtered(2, 2, 3),
            "sample_texels_float",
        ),
    ];
    for (case, desc, kernel) in refused {
        assert!(d.new_sampler_state(&desc).unwrap().is_some(), "{case}");
        let (sel, line) = death_line(|| cubic_samples(&p, &d, &desc, kernel));
        assert_eq!(
            sel, "dispatchThreads:threadsPerThreadgroup:",
            "{case}: {line}"
        );
        assert!(line.contains("SamplerBicubicFilter"), "{case}: {line}");
        if let Some(md) = &md {
            cubic_samples(&mp, md, &desc, kernel);
        }
    }
}

/// How a kernel reaches the sampler it samples [`noise_texture`] through, from an argument
/// buffer at buffer 2 holding resource IDs.
#[derive(Clone, Copy, Debug)]
enum Held {
    /// The buffer holds the sampler alone, and the kernel samples the texture at texture slot
    /// 0: `sample_held_sampler_float`.
    Sampler,
    /// The buffer holds the texture, then the sampler, made resident for reading:
    /// `sample_held_float`.
    Both,
}

/// What `sample()` through a sampler of `desc`, made for argument buffers and held by its
/// resource ID in an argument buffer as `held` says, reads of [`noise_texture`] on `p`'s
/// device `d`, into floats, as their bits, at [`cubic_samples`]'s positions.
fn held_samples<P: reims_vgpu_core::platform::Platform>(
    p: &P,
    d: &P::Device,
    desc: &SamplerDesc,
    held: Held,
) -> Vec<[u32; 4]> {
    use reims_vgpu_core::platform::blit::ResourceRef;
    use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
    use reims_vgpu_core::platform::compute::{ComputeCmd, ComputeEncoder as _};
    use reims_vgpu_core::platform::resource::{Buffer as _, Texture as _};
    use reims_vgpu_core::platform::types::{MtlSize, NsRange};

    use crate::gpu::ycbcr::measure::{PROBE, float_texels, probe};
    let t = noise_texture(p, d, 5);
    let desc = SamplerDesc {
        support_argument_buffers: Some(true),
        ..*desc
    };
    let s = d.new_sampler_state(&desc).unwrap().unwrap();
    let library = d.new_library(PROBE).unwrap().unwrap().unwrap();
    let kernel = match held {
        Held::Sampler => "sample_held_sampler_float",
        Held::Both => "sample_held_float",
    };
    let f = library.new_function(OnFault::Die, kernel).unwrap().unwrap();
    let state = d
        .new_compute_pipeline_state_for_function(Some(&f))
        .unwrap()
        .unwrap();
    let grid = ((16 + 4) * 3, (8 + 4) * 3);
    let at = probe((-2.0, -2.0), 1.0 / 3.0, grid, true, 0);
    let buffer = |length: u64| {
        d.new_buffer_of_length(OnFault::Die, length, 0)
            .unwrap()
            .unwrap()
    };
    let out = buffer(grid.0 * grid.1 * 16);
    let (positions, args) = (buffer(u64::try_from(at.len()).unwrap()), buffer(64));
    let ids = match held {
        Held::Sampler => vec![s.gpu_resource_id()],
        Held::Both => vec![t.gpu_resource_id(), s.gpu_resource_id()],
    };
    // SAFETY: Shared buffers of `at.len()` and 64 bytes, which no command uses yet.
    unsafe {
        std::ptr::copy_nonoverlapping(at.as_ptr(), positions.contents().as_ptr::<u8>(), at.len());
        let words = args.contents().as_ptr::<u64>();
        for (i, id) in ids.into_iter().enumerate() {
            words.add(i).write(id);
        }
    }
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let e = cb.compute_encoder(OnFault::Die).unwrap().unwrap();
    let buffers = [Some(&out), Some(&positions), Some(&args)];
    let resident = [Some(ResourceRef::Texture(&t))];
    let reach = match held {
        Held::Sampler => ComputeCmd::SetTexture {
            texture: Some(&t),
            index: 0,
        },
        Held::Both => ComputeCmd::UseResources {
            resources: &resident,
            usage: 1,
        },
    };
    for cmd in [
        ComputeCmd::SetComputePipelineState {
            state: Some(&state),
        },
        reach,
        ComputeCmd::SetBuffers {
            buffers: &buffers,
            offsets: &[0, 0, 0],
            range: NsRange::new(0, 3),
        },
        ComputeCmd::DispatchThreads {
            threads_per_grid: MtlSize::new(grid.0, grid.1, 1),
            threads_per_threadgroup: MtlSize::new(8, 8, 1),
        },
        ComputeCmd::EndEncoding,
    ] {
        e.encode(OnFault::Die, cmd).unwrap();
    }
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    assert_eq!(cb.error(), None);
    let n = usize::try_from(grid.0 * grid.1 * 16).unwrap();
    // SAFETY: a Shared buffer's bytes, read after the command buffer completed.
    let read = unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), n) };
    (float_texels(read).into_iter())
        .map(|t| t.map(f32::to_bits))
        .collect()
}

/// A cubic sampler an argument buffer holds beside the texture it samples fails the command
/// buffer of the dispatch that samples through it, which dies where its failure is first
/// read, at the wait (`CommandBufferError::CubicArgumentSampler`), where Metal samples; a
/// linear one held so samples as Metal's does, bit for bit, so the death is the cubic
/// filter's, not the argument buffer's
#[test]
fn a_cubic_sampler_in_an_argument_buffer_dies_where_its_failure_is_read() {
    use reims_vgpu_metal::MetalPlatform;
    let (p, d) = (crate::platform::tests::platform(), dev());
    let mp = MetalPlatform::new(Box::new(crate::gpu::device::tests::metal_tests::NoLauncher));
    let md = MetalPlatform::system_default_device();
    let linear = filtered(1, 1, 0);
    let ours = held_samples(&p, &d, &linear, Held::Both);
    if let Some(md) = &md {
        assert_eq!(ours, held_samples(&mp, md, &linear, Held::Both), "linear");
    }
    let cubic = filtered(2, 2, 0);
    let (sel, line) = death_line(|| held_samples(&p, &d, &cubic, Held::Both));
    assert_eq!(sel, "waitUntilCompleted", "{line}");
    let named = "a sample through a cubic sampler an argument buffer holds";
    assert!(line.contains(named), "{line}");
    if let Some(md) = &md {
        held_samples(&mp, md, &cubic, Held::Both);
    }
}

/// While a cubic sampler state is alive on the device, one no argument buffer holds and no
/// command uses, a dispatch that samples through a linear sampler an argument buffer holds
/// samples as Metal's does: the kernel tests each argument buffer sampler it reaches, and
/// reaches no cubic one. (A fragment stage tests none, and its draw dies while a cubic state
/// is alive, the module's docs.)
#[test]
fn a_live_cubic_sampler_leaves_argument_buffer_samples_metals() {
    use reims_vgpu_metal::MetalPlatform;
    let (p, d) = (crate::platform::tests::platform(), dev());
    let mp = MetalPlatform::new(Box::new(crate::gpu::device::tests::metal_tests::NoLauncher));
    let md = MetalPlatform::system_default_device();
    let linear = filtered(1, 1, 0);
    let cubic = d.new_sampler_state(&filtered(2, 2, 0)).unwrap().unwrap();
    let ours = held_samples(&p, &d, &linear, Held::Both);
    drop(cubic);
    if let Some(md) = &md {
        let cubic = md.new_sampler_state(&filtered(2, 2, 0)).unwrap().unwrap();
        assert_eq!(ours, held_samples(&mp, md, &linear, Held::Both), "Metal");
        drop(cubic);
    }
}

/// What a draw over 16 by 8 `BGRA8Unorm` does on `p`'s device `d` whose fragment function
/// samples [`noise_texture`] through a sampler of `desc`, made for argument buffers, that an
/// argument buffer at fragment buffer 0 holds beside it (`sample_held_fragment`), a cubic
/// sampler state made after the draw was encoded and before the commit where `cubic` says:
/// the command buffer's status and error code once it completed.
fn held_draw<P: reims_vgpu_core::platform::Platform>(
    p: &P,
    d: &P::Device,
    desc: &SamplerDesc,
    cubic: bool,
) -> (u64, u64) {
    use reims_vgpu_core::platform::blit::ResourceRef;
    use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
    use reims_vgpu_core::platform::descriptor::RenderPassDescriptor as _;
    use reims_vgpu_core::platform::render::{RenderCmd, RenderEncoder as _};
    use reims_vgpu_core::platform::resource::{Buffer as _, Texture as _};
    use reims_vgpu_core::platform::types::NsRange;
    use reims_vgpu_core::platform::values::{ColorAttachmentSet, RenderPipelineDesc, TextureDesc};

    use crate::gpu::ycbcr::measure::PROBE;
    let t = noise_texture(p, d, 5);
    let held = |desc: &SamplerDesc| SamplerDesc {
        support_argument_buffers: Some(true),
        ..*desc
    };
    let s = d.new_sampler_state(&held(desc)).unwrap().unwrap();
    let library = d.new_library(PROBE).unwrap().unwrap().unwrap();
    let function = |name| library.new_function(OnFault::Die, name).unwrap().unwrap();
    let (vertex, fragment) = (function("cover_vertex"), function("sample_held_fragment"));
    let state = d
        .new_render_pipeline_state(&RenderPipelineDesc {
            label: None,
            vertex: &vertex,
            fragment: &fragment,
            color0_pixel_format: 80,
            blend_over: false,
            resource_index: None,
        })
        .unwrap()
        .unwrap();
    let args = d
        .new_buffer_of_length(OnFault::Die, 64, 0)
        .unwrap()
        .unwrap();
    // SAFETY: a Shared buffer of 64 bytes, which no command uses yet.
    unsafe {
        let words = args.contents().as_ptr::<u64>();
        words.write(t.gpu_resource_id());
        words.add(1).write(s.gpu_resource_id());
    }
    let target = p.texture_descriptor(&TextureDesc {
        texture_type: 2,
        pixel_format: 80,
        width: 16,
        height: 8,
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
    let buffers = [Some(&args)];
    let resident = [Some(ResourceRef::Texture(&t))];
    for cmd in [
        RenderCmd::SetRenderPipelineState {
            pipeline_state: Some(&state),
        },
        RenderCmd::SetFragmentBuffers {
            buffers: &buffers,
            offsets: &[0],
            range: NsRange::new(0, 1),
        },
        RenderCmd::UseResources {
            resources: &resident,
            usage: 1,
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
    let cubic = cubic.then(|| {
        d.new_sampler_state(&held(&filtered(2, 2, 0)))
            .unwrap()
            .unwrap()
    });
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    drop(cubic);
    let code = cb.error().map_or(0, |e| u64::try_from(e.code).unwrap());
    (cb.status().0, code)
}

/// A draw whose fragment stage samples through a sampler an argument buffer holds, encoded
/// while no cubic sampler state is alive, completes as Metal's does. Where a cubic state is made
/// after the draw was encoded and before its command buffer commits, the runtime leaves it out
/// of the fragment stage's sampler tables and announces so as the state is made
/// (`unlisted-argument-sampler`, of [`crate::answer::DIES`]): a function of that stage that read
/// the state through its ID would read the neutral sampler. In strict mode that dies at the
/// state's creator, naming it; by default it is relayed, and the draw, which reads the linear
/// state alone, completes as Metal's does
#[test]
fn a_cubic_state_made_before_a_fragment_stages_commit_is_relayed_unlisted() {
    use reims_vgpu_metal::MetalPlatform;
    let (p, d) = (crate::platform::tests::platform(), dev());
    let linear = filtered(1, 1, 0);
    assert_eq!(held_draw(&p, &d, &linear, false), (4, 0));
    let (sel, line) = death_line(|| held_draw(&p, &d, &linear, true));
    assert_eq!(sel, "newSamplerStateWithDescriptor:", "{line}");
    assert!(line.contains("unlisted-argument-sampler"), "{line}");
    let relayed = crate::answer::relaxed(|| held_draw(&p, &d, &linear, true));
    assert_eq!(relayed, (4, 0));
    if let Some(md) = MetalPlatform::system_default_device() {
        let mp = MetalPlatform::new(Box::new(crate::gpu::device::tests::metal_tests::NoLauncher));
        for cubic in [false, true] {
            assert_eq!(
                held_draw(&mp, &md, &linear, cubic),
                (4, 0),
                "Metal: {cubic}"
            );
        }
    }
}

/// A pipeline state of a kernel that samples a texture slot's texture through a sampler an
/// argument buffer holds dies where it is made, naming the runtime's refusal
/// (`ArgumentBufferTextures`), under a linear sampler and a cubic one alike, where Metal makes
/// it and samples through each: a refusal the runtime has not lifted, which a guest's function
/// of that shape reaches at its pipeline state
#[test]
fn a_texture_slot_sampled_through_an_argument_buffers_sampler_dies_where_it_is_made() {
    use reims_vgpu_metal::MetalPlatform;
    let (p, d) = (crate::platform::tests::platform(), dev());
    let mp = MetalPlatform::new(Box::new(crate::gpu::device::tests::metal_tests::NoLauncher));
    let md = MetalPlatform::system_default_device();
    for (case, desc) in [("linear", filtered(1, 1, 0)), ("cubic", filtered(2, 2, 0))] {
        let (sel, line) = death_line(|| held_samples(&p, &d, &desc, Held::Sampler));
        assert_eq!(
            sel, "newComputePipelineStateWithFunction:error:",
            "{case}: {line}"
        );
        assert!(line.contains("ArgumentBufferTextures"), "{case}: {line}");
        if let Some(md) = &md {
            held_samples(&mp, md, &desc, Held::Sampler);
        }
    }
}

/// The functions each pipeline test of [`refusals_metal_answers_are_its_nserror_codes`]
/// makes: kernels by `newComputePipelineStateWithFunction:error:` and by a guest descriptor,
/// vertex and fragment pairs by a guest render descriptor, and tile functions by a guest tile
/// descriptor.
type Pipelines<'a> = (&'a [&'a str], &'a [(&'a str, &'a str)], &'a [&'a str]);

/// What each creator answers on `d` for the functions `pipelines` names of the metallib
/// `library`, by name: `Ok` for a state, `Err` of the `NSError`'s code.
fn answers<P: reims_vgpu_core::platform::Platform>(
    d: &P::Device,
    library: &[u8],
    (kernels, renders, tiles): Pipelines<'_>,
) -> Vec<(String, Result<(), i64>)> {
    use crate::gpu::serialized::tests::{hex, pipeline_blob};
    use reims_vgpu_core::rt::Hold;
    let library = d.new_library(library).unwrap().unwrap().unwrap();
    let function = |name: &str| {
        library
            .new_function(OnFault::Return, name)
            .unwrap()
            .unwrap()
    };
    let answer = |what: String, made: Result<Option<_>, NsErrorText>| {
        let made = made.map(|state: Option<_>| assert!(state.is_some(), "{what}: nil"));
        (what, made.map_err(|e| e.code))
    };
    let mut out = Vec::new();
    for &name in kernels {
        let f = function(name);
        let made = d.new_compute_pipeline_state_for_function(Some(&f));
        out.push(answer(
            format!("{name} by function"),
            made.map(|s| s.map(drop)),
        ));
        let mut resolve = |_: Hold<'_>, r: u64| Ok((r == 5).then(|| f.clone_ref()));
        let desc = d
            .deserialize_compute_pipeline_descriptor(
                Hold::NONE,
                &hex("01 00 04 05000000"),
                &mut resolve,
            )
            .unwrap()
            .unwrap();
        let made = d.new_compute_pipeline_state(Some(&desc), 0).unwrap();
        out.push(answer(
            format!("{name} by descriptor"),
            made.map(|s| s.map(drop)),
        ));
    }
    for &(vertex, fragment) in renders {
        let (v, f) = (function(vertex), function(fragment));
        let mut resolve = |_: Hold<'_>, r: u64| {
            Ok(match r {
                0x21 => Some(v.clone_ref()),
                0x22 => Some(f.clone_ref()),
                _ => None,
            })
        };
        let desc = d
            .deserialize_render_pipeline_descriptor(
                Hold::NONE,
                &pipeline_blob("basic", 0x21, 0x22, 0),
                &mut resolve,
            )
            .unwrap()
            .unwrap();
        let made = d.new_render_pipeline_state_from_descriptor(Some(&desc), 0);
        let made = made.unwrap().map(|s| s.map(drop));
        out.push(answer(format!("{vertex} and {fragment}"), made));
    }
    for &name in tiles {
        let t = function(name);
        let mut resolve = |_: Hold<'_>, r: u64| Ok((r == 0x31).then(|| t.clone_ref()));
        let desc = d
            .deserialize_tile_render_pipeline_descriptor(
                Hold::NONE,
                &pipeline_blob("tile", 0, 0, 0x31),
                0x100,
                &mut resolve,
            )
            .unwrap()
            .unwrap();
        let made = d.new_tile_render_pipeline_state(Some(&desc), 0);
        out.push(answer(
            format!("{name} by tile"),
            made.unwrap().map(|s| s.map(drop)),
        ));
    }
    out
}

/// A function Metal's compiler refuses on this device (a render state setter on an indirect
/// render command, in every stage, and a fragment function's read of a program-scope
/// `thread_index_in_quadgroup`) and one whose import no linked library exports (the library at
/// its install name exporting another symbol, or no file there) is the `NSError` at every
/// creator, of Metal's code, the runtime's domain and its text; the functions beside them
/// that Metal makes are made (a fragment's program-scope `threads_per_simdgroup` among them).
/// Metal's answers are compared on the host
#[test]
fn refusals_metal_answers_are_its_nserror_codes() {
    use reims_vgpu_metal::MetalPlatform;
    let d = dev();
    let md = MetalPlatform::system_default_device();
    let refused: Pipelines<'_> = (
        &["k_cull", "k_depth_stencil", "k_draw"],
        &[
            ("v_plain", "f_plain"),
            ("v_plain", "f_cull"),
            ("v_cull", "f_plain"),
            ("v_plain", "f_draw"),
            ("v_plain", "f_quad_lane"),
            ("v_plain", "f_quad_lane16"),
            ("v_plain", "f_simd_width"),
        ],
        &["t_cull", "t_draw"],
    );
    let imports: Pipelines<'_> = (&["k_import"], &[("v_import", "f_import")], &["t_import"]);
    // Read at run time: each names the build's `OUT_DIR` as its library's install name.
    let import = |name: &str| {
        std::fs::read(crate::gpu::device::tests::out_dir().join(format!("{name}.metallib")))
            .unwrap()
    };
    let (ext, other, absent) = (
        import("refused_import_ext"),
        import("refused_import_other"),
        import("refused_import_absent"),
    );
    let cases = [
        (
            &include_bytes!(concat!(env!("OUT_DIR"), "/refused.metallib"))[..],
            refused,
            [
                Err(3),
                Err(3),
                Err(3),
                Err(3),
                Ok(()),
                Ok(()),
                Ok(()),
                Err(3),
                Err(3),
                Ok(()),
                Err(3),
                Err(3),
                Ok(()),
                Err(3),
                Ok(()),
            ]
            .as_slice(),
        ),
        (&ext[..], imports, &[Ok(()); 4]),
        (&other[..], imports, &[Err(2); 4]),
        (&absent[..], imports, &[Err(2); 4]),
    ];
    let library = d
        .new_library(include_bytes!(concat!(
            env!("OUT_DIR"),
            "/refused.metallib"
        )))
        .unwrap()
        .unwrap()
        .unwrap();
    let cull = library.new_function(OnFault::Return, "k_cull").unwrap();
    let error = d
        .new_compute_pipeline_state_for_function(cull.as_ref())
        .unwrap_err();
    assert_eq!(
        (error.domain.as_str(), error.code),
        ("metal2vulkan-runtime", 3)
    );
    for (library, pipelines, expected) in cases {
        let ours = answers::<crate::platform::VulkanPlatform>(&d, library, pipelines);
        let codes: Vec<_> = ours.iter().map(|(_, a)| *a).collect();
        assert_eq!(codes, expected, "{ours:?}");
        if let Some(md) = &md {
            let metal = answers::<MetalPlatform>(md, library, pipelines);
            assert_eq!(ours, metal);
        }
    }
}

/// Metal's `localizedDescription` for a state that supports indirect command buffers whose
/// function of each stage uses a direct texture or sampler argument.
const DIRECT_ARGUMENT: [&str; 3] = [
    "Fragment shader cannot be used with indirect command buffers",
    "Vertex shader cannot be used with indirect command buffers",
    "Compute function cannot be used with indirect command buffers",
];

/// A case of [`indirect_answers`]: its name, its serialized descriptor in hex, and the
/// metallibs whose first function each ref resolves to.
type IndirectCase<'a> = (&'a str, &'a str, &'a [(u64, &'a [u8])]);

/// A creator's answer in [`indirect_answers`]: `Ok` for a state, `Err` of the `NSError`'s
/// code and which of [`DIRECT_ARGUMENT`] its description holds, in the order it holds them.
type IndirectAnswer = Result<(), (i64, Vec<usize>)>;

/// What the compute and render creators answer on `d` for each of `cases`, a serialized
/// descriptor (compute where its name says so) and the fixture functions its refs 5 (a
/// kernel), 0x21 (a vertex function) and 0x22 (a fragment function) resolve to, by name.
fn indirect_answers<P: reims_vgpu_core::platform::Platform>(
    d: &P::Device,
    cases: &[IndirectCase<'_>],
) -> Vec<(String, IndirectAnswer)> {
    use crate::gpu::serialized::tests::hex;
    use reims_vgpu_core::rt::Hold;
    let function = |bytes: &[u8]| {
        let library = d.new_library(bytes).unwrap().unwrap().unwrap();
        let name = library.function_names().remove(0);
        library
            .new_function(OnFault::Return, &name)
            .unwrap()
            .unwrap()
    };
    let mut out = Vec::new();
    for &(name, blob, functions) in cases {
        let functions: Vec<_> = functions.iter().map(|&(r, f)| (r, function(f))).collect();
        let mut resolve = |_: Hold<'_>, r: u64| {
            Ok(functions
                .iter()
                .find(|(at, _)| *at == r)
                .map(|(_, f)| f.clone_ref()))
        };
        let made = if name.starts_with("compute") {
            let desc = d
                .deserialize_compute_pipeline_descriptor(Hold::NONE, &hex(blob), &mut resolve)
                .unwrap();
            let made = d.new_compute_pipeline_state(desc.as_ref(), 0).unwrap();
            made.map(|s| s.is_some())
        } else {
            let desc = d
                .deserialize_render_pipeline_descriptor(Hold::NONE, &hex(blob), &mut resolve)
                .unwrap();
            let made = d.new_render_pipeline_state_from_descriptor(desc.as_ref(), 0);
            made.unwrap().map(|s| s.is_some())
        };
        let answer = made
            .map(|state| assert!(state, "{name}: nil"))
            .map_err(|e| {
                let mut held: Vec<_> = DIRECT_ARGUMENT
                    .iter()
                    .enumerate()
                    .filter_map(|(i, text)| e.description.find(text).map(|at| (at, i)))
                    .collect();
                held.sort_unstable();
                (e.code, held.into_iter().map(|(_, i)| i).collect())
            });
        out.push((name.to_owned(), answer));
    }
    out
}

/// A state that supports indirect command buffers (compute key 7, render key 25) whose
/// kernel, vertex or fragment function uses a direct texture and sampler argument once Metal
/// links the pipeline is the `NSError` at its creator, of Metal's code (3) and Metal's
/// description, in the runtime's domain: where both functions use one, the description names
/// the fragment function alone (the stages' texts it holds are compared in the order it holds
/// them). The same functions without that support, functions with no such argument with it,
/// a vertex function's sample the fragment function does not read or with no fragment
/// function, and a fragment function's sample that reaches no colour an attachment writes (no
/// attachment, colour 0 `Invalid` or of no channel beside a written colour 1, a colour 1 with
/// colour 0 alone attached), are made, and so is one that reaches a colour under a mask that
/// holds none of its format's channels yet is not empty (alpha alone, of `R8Unorm`). With
/// rasterization off, of a vertex function that returns a value, the state is the `NSError`
/// of code 3, as Metal's, which names the fragment function's argument. Metal's answers are
/// compared on the host
#[test]
fn indirect_states_of_direct_arguments_are_metals_nserror() {
    use reims_vgpu_metal::MetalPlatform;
    macro_rules! shader {
        ($name:literal) => {
            &include_bytes!(concat!(
                "../../../../../../tests/suite/fixtures/shaders/",
                $name,
                ".metallib"
            ))[..]
        };
    }
    let (sample_k, double_gid) = (shader!("compute_sample"), shader!("kernel"));
    let (tri_vert, tri_frag) = (shader!("vert"), shader!("frag"));
    let (render_vtex, render_ftex) = (shader!("render_vtex"), shader!("render_ftex"));
    let render_color = shader!("render_color");
    let colour_1 = &include_bytes!(concat!(env!("OUT_DIR"), "/icb_color1.metallib"))[..];
    // Colour 0 `RGBA8Unorm`, and the functions at 0x21 and 0x22, with key 25 set to 1 or not.
    let colour = "01000000 08000000 02 00 04 00000000 01 04 46000000";
    let indirect =
        format!("04 08 04 1c000000 01 04 21000000 02 04 22000000 19 04 01000000 000000 {colour}");
    let direct = format!("03 08 04 14000000 01 04 21000000 02 04 22000000 00 {colour}");
    let cases: &[IndirectCase<'_>] = &[
        (
            "compute",
            "02 07 04 01000000 00 04 05000000",
            &[(5, sample_k)],
        ),
        ("compute plain", "01 00 04 05000000", &[(5, sample_k)]),
        (
            "compute no texture",
            "02 07 04 01000000 00 04 05000000",
            &[(5, double_gid)],
        ),
        (
            "fragment",
            &indirect,
            &[(0x21, tri_vert), (0x22, render_ftex)],
        ),
        (
            "both",
            &indirect,
            &[(0x21, render_vtex), (0x22, render_ftex)],
        ),
        (
            "fragment plain",
            &direct,
            &[(0x21, tri_vert), (0x22, render_ftex)],
        ),
        (
            "no texture",
            &indirect,
            &[(0x21, tri_vert), (0x22, tri_frag)],
        ),
        (
            "vertex plain",
            &direct,
            &[(0x21, render_vtex), (0x22, render_color)],
        ),
    ];
    // A vertex function's sample: Metal refuses the state where it reaches a colour the
    // fragment function reads, and makes it where the fragment function reads none or there
    // is none. A fragment function's sample where no colour attachment is written: Metal
    // makes the state. Two attachments: offsets 12 and 12 plus the first's length.
    let indirect_with = |colours: &str| {
        format!("04 08 04 1c000000 01 04 21000000 02 04 22000000 19 04 01000000 000000 {colours}")
    };
    let invalid = indirect_with("01000000 08000000 02 00 04 00000000 01 04 00000000");
    let unwritten =
        indirect_with("01000000 08000000 03 00 04 00000000 01 04 46000000 09 04 00000000");
    let invalid_then_1 = indirect_with(
        "02000000 0c000000 19000000 02 00 04 00000000 01 04 00000000 \
         02 00 04 01000000 01 04 46000000",
    );
    let unwritten_then_1 = indirect_with(
        "02000000 0c000000 1f000000 03 00 04 00000000 01 04 46000000 09 04 00000000 \
         02 00 04 01000000 01 04 46000000",
    );
    let fragment: &[(u64, &[u8])] = &[(0x21, tri_vert), (0x22, render_ftex)];
    let linked: &[IndirectCase<'_>] = &[
        (
            "vertex",
            &indirect,
            &[(0x21, render_vtex), (0x22, render_color)],
        ),
        (
            "vertex unread",
            &indirect,
            &[(0x21, render_vtex), (0x22, tri_frag)],
        ),
        (
            "vertex alone",
            "02 01 04 21000000 19 04 01000000",
            &[(0x21, render_vtex)],
        ),
        (
            "fragment, no colour attachment",
            "03 01 04 21000000 02 04 22000000 19 04 01000000",
            fragment,
        ),
        ("fragment, colour 0 Invalid", &invalid, fragment),
        ("fragment, colour 0 unwritten", &unwritten, fragment),
        (
            "fragment, colour 0 Invalid, colour 1 written",
            &invalid_then_1,
            fragment,
        ),
        (
            "fragment, colour 0 unwritten, colour 1 written",
            &unwritten_then_1,
            fragment,
        ),
        (
            "a fragment of colour 1, colour 0 alone",
            &indirect,
            &[(0x21, tri_vert), (0x22, colour_1)],
        ),
    ];
    let cases = [cases, linked].concat();
    let ours = indirect_answers::<crate::platform::VulkanPlatform>(&dev(), &cases);
    let expected = [
        Err((3, vec![2])),
        Ok(()),
        Ok(()),
        Err((3, vec![0])),
        Err((3, vec![0])),
        Ok(()),
        Ok(()),
        Ok(()),
        Err((3, vec![1])),
        Ok(()),
        Ok(()),
        Ok(()),
        Ok(()),
        Ok(()),
        Ok(()),
        Ok(()),
        Ok(()),
    ];
    let answers: Vec<_> = ours.iter().map(|(_, a)| a.clone()).collect();
    assert_eq!(answers, expected, "{ours:?}");
    let md = MetalPlatform::system_default_device();
    if let Some(md) = &md {
        assert_eq!(indirect_answers::<MetalPlatform>(md, &cases), ours);
    }
    // Rasterization off, of a vertex function that returns a value: code 3, where Metal's
    // text names the fragment function's argument, the runtime's either refusal's.
    let unrasterized = format!(
        "05 08 04 20000000 01 04 21000000 02 04 22000000 07 04 00000000 19 04 01000000 00 \
         {colour}"
    );
    let case = (
        "fragment, rasterization off",
        unrasterized.as_str(),
        fragment,
    );
    let ours = indirect_answers::<crate::platform::VulkanPlatform>(&dev(), &[case]);
    assert!(matches!(ours[0].1, Err((3, _))), "{ours:?}");
    if let Some(md) = &md {
        let metal = indirect_answers::<MetalPlatform>(md, &[case]);
        assert_eq!(metal[0].1, Err((3, vec![0])), "Metal");
    }
    // Colour 0 `R8Unorm` under a mask of alpha alone, which holds none of its format's
    // channels: no colour is written, so the sample reaches none, and the state is made, as
    // Metal makes it.
    let masked = indirect_with("01000000 08000000 03 00 04 00000000 01 04 0a000000 09 04 01000000");
    let case = ("fragment, R8Unorm of alpha", masked.as_str(), fragment);
    let ours = indirect_answers::<crate::platform::VulkanPlatform>(&dev(), &[case]);
    assert_eq!(ours[0].1, Ok(()), "{ours:?}");
    if let Some(md) = &md {
        assert_eq!(indirect_answers::<MetalPlatform>(md, &[case]), ours);
    }
    // The runtime's domain, as every pipeline `NSError` here carries.
    let d = dev();
    let library = d.new_library(sample_k).unwrap().unwrap().unwrap();
    let f = library.new_function(OnFault::Return, "sample_k").unwrap();
    let mut resolve =
        |_: reims_vgpu_core::rt::Hold<'_>, _: u64| Ok(f.as_ref().map(|f| f.clone_ref()));
    let blob = crate::gpu::serialized::tests::hex("02 07 04 01000000 00 04 05000000");
    let desc = d
        .deserialize_compute_pipeline_descriptor(
            reims_vgpu_core::rt::Hold::NONE,
            &blob,
            &mut resolve,
        )
        .unwrap();
    let error = d
        .new_compute_pipeline_state(desc.as_ref(), 0)
        .unwrap()
        .unwrap_err();
    assert_eq!(error.domain, crate::answer::RUNTIME_DOMAIN);
}

/// A render state with rasterization off is Metal's, as Metal links it (the M4's table, and
/// here): of a vertex function that returns nothing, made where the fragment function writes
/// a constant colour, and where no colour attachment is written whatever it reads; the
/// `NSError` of code 3 where a written colour attachment keeps its read of a `stage_in`
/// input, which no vertex function writes then, and, with `supportIndirectCommandBuffers`,
/// its direct texture. Of a vertex function that returns a value, the `NSError` of code 3
/// (`UnrasterizedVertexReturn`), with or without that support
#[test]
fn unrasterized_states_are_metals() {
    use reims_vgpu_metal::MetalPlatform;
    macro_rules! shader {
        ($name:literal) => {
            &include_bytes!(concat!(
                "../../../../../../tests/suite/fixtures/shaders/",
                $name,
                ".metallib"
            ))[..]
        };
    }
    let void_vertex = &include_bytes!(concat!(env!("OUT_DIR"), "/unrasterized.metallib"))[..];
    let (tri_vert, tri_frag) = (shader!("vert"), shader!("frag"));
    let (render_color, render_ftex) = (shader!("render_color"), shader!("render_ftex"));
    // A render blob of rasterization off: the vertex function at 0x21, the fragment function
    // at 0x22 where `fragment`, key 25 where `icb`, and colour 0 `RGBA8Unorm` where `colour`.
    let blob = |fragment: bool, icb: bool, colour: bool| {
        let mut entries = vec!["01 04 21000000"];
        entries.extend(fragment.then_some("02 04 22000000"));
        entries.push("07 04 00000000");
        entries.extend(icb.then_some("19 04 01000000"));
        let n = entries.len() + usize::from(colour);
        let head = 1 + 6 * n;
        let at = head.next_multiple_of(4);
        let mut out = format!("{n:02x}");
        if colour {
            out += &format!(" 08 04 {at:02x}000000");
        }
        for entry in entries {
            out = format!("{out} {entry}");
        }
        if colour {
            out += &" 00".repeat(at - head);
            out += " 01000000 08000000 02 00 04 00000000 01 04 46000000";
        }
        out
    };
    // Each case: its name, blob, functions and the answer expected.
    let mut rows = Vec::new();
    for (what, function) in [
        ("constant", tri_frag),
        ("stage_in", render_color),
        ("texture", render_ftex),
    ] {
        for icb in [false, true] {
            for colour in [true, false] {
                let expected = match what {
                    "stage_in" if colour => Err((3, vec![])),
                    "texture" if colour && icb => Err((3, vec![0])),
                    _ => Ok(()),
                };
                let name = format!("{what}, icb {icb}, colour {colour}");
                let functions = vec![(0x21, void_vertex), (0x22, function)];
                rows.push((name, blob(true, icb, colour), functions, expected));
            }
        }
    }
    for icb in [false, true] {
        let name = format!("a vertex function's return, icb {icb}");
        let functions = vec![(0x21, tri_vert)];
        rows.push((name, blob(false, icb, true), functions, Err((3, vec![]))));
    }
    let cases: Vec<IndirectCase<'_>> = rows
        .iter()
        .map(|(name, blob, functions, _)| (name.as_str(), blob.as_str(), &functions[..]))
        .collect();
    let ours = indirect_answers::<crate::platform::VulkanPlatform>(&dev(), &cases);
    for ((name, _, _, expected), (_, answer)) in rows.iter().zip(&ours) {
        assert_eq!(answer, expected, "{name}");
    }
    if let Some(md) = MetalPlatform::system_default_device() {
        assert_eq!(indirect_answers::<MetalPlatform>(&md, &cases), ours);
    }
}

/// A render state with rasterization on whose fragment function reads a `stage_in` input no
/// output of its vertex function writes is Metal's: the `NSError` of code 3 where the read
/// reaches colour 0, which is written (`UnwrittenFragmentInputs`); made with no colour
/// attachment, where the linking drops the read, the input reading zero
#[test]
fn an_input_no_vertex_output_writes_is_metals() {
    use reims_vgpu_metal::MetalPlatform;
    macro_rules! shader {
        ($name:literal) => {
            &include_bytes!(concat!(
                "../../../../../../tests/suite/fixtures/shaders/",
                $name,
                ".metallib"
            ))[..]
        };
    }
    let (tri_vert, render_color) = (shader!("vert"), shader!("render_color"));
    let functions: &[(u64, &[u8])] = &[(0x21, tri_vert), (0x22, render_color)];
    let cases: &[IndirectCase<'_>] = &[
        (
            "colour 0 written",
            "03 08 04 14000000 01 04 21000000 02 04 22000000 00 \
             01000000 08000000 02 00 04 00000000 01 04 46000000",
            functions,
        ),
        (
            "no colour attachment",
            "02 01 04 21000000 02 04 22000000",
            functions,
        ),
    ];
    let ours = indirect_answers::<crate::platform::VulkanPlatform>(&dev(), cases);
    let answers: Vec<_> = ours.iter().map(|(_, a)| a.clone()).collect();
    assert_eq!(answers, [Err((3, vec![])), Ok(())], "{ours:?}");
    if let Some(md) = MetalPlatform::system_default_device() {
        assert_eq!(indirect_answers::<MetalPlatform>(&md, cases), ours);
    }
}

/// A state that supports indirect command buffers whose function's reflection the compile layer
/// refuses (`nested_arguments_kernel` and `nested_arguments_vertex`: an argument buffer holding
/// an array of structs that hold a struct and no constant, whose stride Metal's answer is
/// unmeasured for) is made, as without that support: the compile layer decides from the
/// arguments' roles alone that neither function uses a direct texture or sampler argument,
/// which Metal answers nil and an `NSError` for. A compute state and a render
/// state of the vertex function alike
#[test]
fn states_whose_arguments_reflection_is_refused_are_decided_by_their_roles() {
    use metal2vulkan::error::{Error as Compile, Feature};
    use reims_vgpu_core::platform::device::GpuDevice as _;
    use reims_vgpu_core::platform::resource::Library as _;

    let d = dev();
    let library = d
        .new_library(crate::gpu::ycbcr::measure::PROBE)
        .unwrap()
        .unwrap()
        .unwrap();
    let function = |name| {
        library
            .new_function(OnFault::Return, name)
            .unwrap()
            .unwrap()
    };
    let kernel = function("nested_arguments_kernel");
    let compute =
        ComputePipelineDescriptor::new(StageSource::Function(kernel.function_descriptor()));
    match compute.reflection(&metal2vulkan::Budget::standard()) {
        Err(Compile::Unsupported(u)) => assert_eq!(u.feature, Feature::Reflection),
        other => panic!("{other:?}"),
    }
    let compute = |indirect| {
        let mut desc = compute.clone();
        desc.support_indirect_command_buffers = indirect;
        Obj::new(PipelineDescriptorObject {
            desc: Mutex::new(desc),
            resource_index: Mutex::new(None),
            ignored: Vec::new(),
            nearest_even: false,
            label: None,
            raw_format: None,
        })
    };
    for indirect in [false, true] {
        let made = d.new_compute_pipeline_state(Some(&compute(indirect)), 0);
        assert!(matches!(made, Ok(Ok(Some(_)))), "{indirect}: {made:?}");
    }
    let vertex = function("nested_arguments_vertex");
    let fragment = indirect_functions().1;
    let render = guest_descriptor(&(vertex, fragment), PixelFormat::BGRA8Unorm, None);
    for indirect in [false, true] {
        plain(|| render.desc.lock().support_indirect_command_buffers = indirect);
        let made = d.new_render_pipeline_state_from_descriptor(Some(&render), 0);
        assert!(matches!(made, Ok(Ok(Some(_)))), "{indirect}: {made:?}");
    }
}

/// RenderBox's custom effect is Metal's: the fragment function that calls `custom_fn` links the
/// function made under that specialized name and reads, at every pixel, the bits Metal's reads;
/// with the function made under its own name or another, the pipeline is refused on both. (What
/// each function answers as its name is Metal's too: the Metal platform's
/// `a_specialized_name_resolves_a_kernel_s_call`.)
#[test]
fn a_function_linked_under_a_specialized_name_is_metals() {
    use reims_vgpu_metal::MetalPlatform;
    let Some(md) = MetalPlatform::system_default_device() else {
        return;
    };
    let mp = MetalPlatform::new(Box::new(crate::gpu::device::tests::metal_tests::NoLauncher));
    let (p, d) = (crate::platform::tests::platform(), dev());
    let library = |bytes| d.new_library(bytes).unwrap().unwrap().unwrap();
    let (caller, callee) = (library(LINKED_CALLER), library(LINKED_CALLEE));
    let metal_library = |bytes| md.new_library(bytes).unwrap().unwrap().unwrap();
    let (metal_caller, metal_callee) = (metal_library(LINKED_CALLER), metal_library(LINKED_CALLEE));
    for specialized in [Some("custom_fn"), None, Some("other_fn")] {
        let ours = gradient::<VulkanPlatform>(&callee, specialized).unwrap();
        let metals = gradient::<MetalPlatform>(&metal_callee, specialized).unwrap();
        assert_eq!(ours.name(), specialized.unwrap_or("linked_gradient"));
        let want = custom_effect(&mp, &md, &metal_caller, &metals);
        assert_eq!(
            want.is_some(),
            specialized == Some("custom_fn"),
            "{specialized:?}"
        );
        let drawn = catch_death(|| custom_effect(&p, &d, &caller, &ours));
        assert_eq!(drawn, Ok(want), "{specialized:?}");
    }
}

/// A pipeline creator's answer in [`linked_answers`]: made, or the `NSError`'s code, or where the
/// creator dies.
type LinkedAnswer = Result<Result<(), i64>, reims_vgpu_core::raise::death::DeathSite>;

/// What the compute and render creators answer for `linked_caller.metal`'s functions with
/// `linked_callee.metal`'s `linked_add_one`, which they call, linked by each list a value
/// carries, each case named: the function made by name, then its copy compiled to a binary.
///
/// The copy is made after every case of the function made by name: while a copy of a library's
/// function compiled to a binary lives, Metal's `newFunctionWithName:` for it, and its
/// `newFunctionWithDescriptor:` for that copy, answer one object, of options 1, even from another
/// library made of the same bytes, and the copy turns the function made before it into that
/// object (measured on the M3 Max, macOS 27.0.1). A case after the copy would link the binary
/// where it names the function made by name.
fn linked_answers<P: reims_vgpu_core::platform::Platform>(
    d: &P::Device,
) -> Vec<(&'static str, LinkedAnswer)> {
    let library = |bytes| d.new_library(bytes).unwrap().unwrap().unwrap();
    let (caller, callee) = (library(LINKED_CALLER), library(LINKED_CALLEE));
    let named = |l: &P::Library, name| l.new_function(OnFault::Return, name).unwrap().unwrap();
    let (kernel, vertex, fragment) = (
        named(&caller, "linked_kernel"),
        named(&caller, "linked_vertex"),
        named(&caller, "linked_fragment"),
    );
    let add_one = named(&callee, "linked_add_one");
    let one = |f: &P::Function| vec![f.clone_ref()];
    let group = || vec![("g".to_owned(), one(&add_one))];
    let answer = |made: Result<
        Result<Option<()>, NsErrorText>,
        reims_vgpu_core::raise::death::DeathSite,
    >| {
        made.map(|made| match made {
            Ok(state) => {
                assert!(state.is_some());
                Ok(())
            }
            Err(e) => Err(e.code),
        })
    };
    let compute = |functions, binary_functions, groups, private_functions| {
        answer(catch_death(|| {
            d.new_compute_pipeline_state_full(&FullComputePipelineDesc {
                label: None,
                function: &kernel,
                threadgroup_size_is_multiple_of_execution_width: false,
                max_total_threads_per_threadgroup: 0,
                max_call_stack_depth: 1,
                support_indirect_command_buffers: false,
                stage_input: None,
                linked: LinkedFunctionsDesc {
                    functions,
                    binary_functions,
                    groups,
                    private_functions,
                },
            })
            .map(|state| state.map(drop))
        }))
    };
    let render = |vertex_linked, fragment_linked| {
        answer(catch_death(|| {
            d.new_render_pipeline_state_full(&FullRenderPipelineDesc {
                label: None,
                vertex: &vertex,
                fragment: Some(&fragment),
                vertex_descriptor: None,
                colors: vec![(
                    0,
                    ColorAttachmentDesc {
                        pixel_format: 80,
                        ..ColorAttachmentDesc::default()
                    },
                )],
                depth_pixel_format: 0,
                stencil_pixel_format: 0,
                raster_sample_count: 1,
                alpha_to_coverage: false,
                alpha_to_one: false,
                rasterization_enabled: true,
                input_primitive_topology: 0,
                max_vertex_amplification_count: 1,
                support_indirect_command_buffers: false,
                vertex_linked,
                fragment_linked,
            })
            .map(|state| state.map(drop))
        }))
    };
    let functions = |f: &P::Function| LinkedFunctionsDesc {
        functions: one(f),
        ..LinkedFunctionsDesc::default()
    };
    let mut answers = vec![
        ("none", compute(vec![], vec![], vec![], vec![])),
        ("functions", compute(one(&add_one), vec![], vec![], vec![])),
        ("private", compute(vec![], vec![], vec![], one(&add_one))),
        ("group", compute(vec![], vec![], group(), vec![])),
        (
            "group, private",
            compute(vec![], vec![], group(), one(&add_one)),
        ),
        (
            "unbinary alone",
            compute(vec![], one(&add_one), vec![], vec![]),
        ),
        (
            "unbinary in both",
            compute(one(&add_one), one(&add_one), vec![], vec![]),
        ),
        (
            "render none",
            render(
                LinkedFunctionsDesc::default(),
                LinkedFunctionsDesc::default(),
            ),
        ),
        (
            "render vertex",
            render(functions(&add_one), LinkedFunctionsDesc::default()),
        ),
        (
            "render fragment",
            render(LinkedFunctionsDesc::default(), functions(&add_one)),
        ),
    ];
    let binary = callee
        .new_function_with_options(OnFault::Return, "linked_add_one", &[], 1, None)
        .unwrap()
        .unwrap();
    answers.extend([
        (
            "binary alone",
            compute(vec![], one(&binary), vec![], vec![]),
        ),
        (
            "binary in functions",
            compute(one(&binary), vec![], vec![], vec![]),
        ),
        (
            "binary in both",
            compute(one(&binary), one(&binary), vec![], vec![]),
        ),
    ]);
    answers
}

/// A call through a visible-function reference that nothing the stage links resolves is
/// refused with Metal's `NSError` code (3, "unresolved visible function reference"), at the
/// compute and the render creators, and made where Metal makes it, each answer Metal's
/// ([`linked_answers`]): made by `functions`, private functions, or a group with the function in
/// private functions, whether the function was made compiled to a binary or not; code 3 with
/// none, a group alone, or `binaryFunctions` alone holding it, as a direct call resolves by no
/// binary function; made with the function in both `functions` and `binaryFunctions` where it
/// was made compiled to a binary, and code 2 where it was not
#[test]
fn unresolved_visible_calls_are_metals() {
    use reims_vgpu_metal::MetalPlatform;
    let Some(md) = MetalPlatform::system_default_device() else {
        return;
    };
    let metals = linked_answers::<MetalPlatform>(&md);
    let ours = linked_answers::<VulkanPlatform>(&dev());
    assert_eq!(ours.len(), metals.len());
    for ((case, ours), (_, metals)) in ours.into_iter().zip(metals) {
        assert_eq!(ours, metals, "{case}");
    }
}
