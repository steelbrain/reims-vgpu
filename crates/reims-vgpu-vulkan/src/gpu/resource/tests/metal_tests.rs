//! The tests of `resource` that compare with Metal on this host: macOS only.

use reims_vgpu_core::platform::types::{MtlOrigin, SizeAndAlign};
use reims_vgpu_core::platform::values::{
    BufferDesc, HeapDesc, HeapMade, OptionSet, SamplerDesc, TextureDesc,
};

use super::*;
use crate::os::OsSurface;

/// Of the suite's kernel function, made on `d`, a state without indirect command buffer
/// support and one with it (`supportIndirectCommandBuffers` set): whether
/// each supports them, its `resourceIndex` and its `gpuResourceID`.
fn kernel_states<P: Platform>(d: &P::Device) -> [(bool, u64, u64); 2] {
    use reims_vgpu_core::platform::resource::{ComputePipeline as _, Library as _};
    use reims_vgpu_core::platform::values::FullComputePipelineDesc;
    let kernel = include_bytes!("../../../../../../tests/shaders/kernel.metallib");
    let library = d.new_library(kernel).unwrap().unwrap().unwrap();
    let name = library.function_names()[0].clone();
    let f = library
        .new_function(OnFault::Return, &name)
        .unwrap()
        .unwrap();
    let plain = d
        .new_compute_pipeline_state_for_function(Some(&f))
        .unwrap()
        .unwrap();
    let supporting = d
        .new_compute_pipeline_state_full(&FullComputePipelineDesc {
            support_indirect_command_buffers: true,
            ..FullComputePipelineDesc::new(&f)
        })
        .unwrap()
        .unwrap();
    [&plain, &supporting].map(|p| {
        (
            p.support_indirect_command_buffers(),
            p.resource_index(),
            p.gpu_resource_id(),
        )
    })
}

/// A compute state's `resourceIndex` is 0 without indirect command buffer support and its
/// `gpuResourceID` with it, as Metal's is
#[test]
fn a_compute_states_resource_index_is_metals() {
    let shape = |[plain, supporting]: [(bool, u64, u64); 2]| {
        (
            plain,
            supporting.0,
            supporting.1 == supporting.2,
            supporting.2 != 0,
        )
    };
    let ours = shape(kernel_states::<VulkanPlatform>(&dev()));
    assert_eq!(ours, ((false, 0, 0), true, true, true));
}

/// Every metallib the tests hold: the suite's shader fixtures but `fc`, whose kernel's function
/// constants Metal asserts at unspecialized, and this crate's own test and product libraries,
/// each with its name.
fn every_metallib() -> Vec<(String, Vec<u8>)> {
    let fixtures = crate::gpu::device::tests::fixture_shaders();
    let mut libs: Vec<(String, Vec<u8>)> = std::fs::read_dir(fixtures)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "metallib"))
        // Its kernel's function constants are unspecialized: Metal asserts at a state of it.
        .filter(|p| p.file_stem().is_some_and(|s| s != "fc"))
        .map(|p| {
            let name = p.file_name().unwrap().to_string_lossy().into_owned();
            (name, std::fs::read(&p).unwrap())
        })
        .collect();
    libs.sort();
    let out = crate::gpu::device::tests::out_dir();
    for name in [
        "shaders",
        "predication",
        "rgbx",
        "spin",
        "icb_encode",
        "reach",
        "rounding",
        "rounding_safe",
        "simd_stages",
        "ycbcr_probe",
        "slice_write_repro",
        "numbered_samplers",
        "five_samplers",
    ] {
        let path = out.join(format!("{name}.metallib"));
        libs.push((name.to_owned(), std::fs::read(&path).unwrap()));
    }
    libs
}

/// What `d` answers of a compute state of each function of `data` made from the function
/// alone, by function name: `Ok` of `maxTotalThreadsPerThreadgroup`, `threadExecutionWidth`,
/// `staticThreadgroupMemoryLength` and `supportIndirectCommandBuffers`; `Err` of the error, nil
/// or death where it makes none.
type ComputeAnswer = Result<(u64, u64, u64, bool), String>;

/// [`ComputeAnswer`]s of every function of `data` on `d`, sorted by name; one `Err` where it
/// makes no library of `data` (a truncated metallib).
fn compute_answers<P: Platform>(d: &P::Device, data: &[u8]) -> Vec<(String, ComputeAnswer)> {
    use reims_vgpu_core::platform::resource::{ComputePipeline as _, Library as _};
    let Ok(Ok(Some(library))) = d.new_library(data) else {
        return vec![("the library".to_owned(), Err("no library".to_owned()))];
    };
    let mut names = library.function_names();
    names.sort();
    names
        .into_iter()
        .map(|name| {
            let f = library.new_function(OnFault::Return, &name).unwrap();
            let made = catch_death(|| d.new_compute_pipeline_state_for_function(f.as_ref()));
            let answer = match made {
                Ok(Ok(Some(state))) => Ok((
                    state.max_total_threads_per_threadgroup(),
                    state.thread_execution_width(),
                    state.static_threadgroup_memory_length(),
                    state.support_indirect_command_buffers(),
                )),
                Ok(Ok(None)) => Err("nil".to_owned()),
                Ok(Err(e)) => Err(format!("error {}", e.code)),
                Err(site) => Err(format!(
                    "died at {}: {}",
                    site.file,
                    crate::answer::LAST_LINE.lock().take().unwrap_or_default()
                )),
            };
            (name, answer)
        })
        .collect()
}

/// A compute state made from each kernel of every metallib the tests hold (100 kernels of 50
/// libraries on this host, kernels of threadgroup memory, imageblocks, SIMD-group functions,
/// argument buffers and indirect commands among them) answers `maxTotalThreadsPerThreadgroup`,
/// `threadExecutionWidth`, `staticThreadgroupMemoryLength` and `supportIndirectCommandBuffers`
/// as Metal's does; a function Metal makes no compute state of (a vertex, fragment or tile
/// function, a kernel of stage-in attributes without their descriptor) makes none here either,
/// though with another error or a death (the library makes compute states of functions alone
/// only in its tests). One kernel Metal makes a state of dies: `ycbcr_probe`'s
/// `sample_held_sampler_float`, which samples a texture an argument buffer holds, the runtime's
/// gap (`ArgumentBufferTextures`), where its tests pin the death
#[test]
fn compute_state_answers_are_metals_for_every_kernel_here() {
    // Metal's answer of each function by its values, `None` where it makes no state.
    let d = dev();
    let (mut differ, mut died, mut compared) = (Vec::new(), Vec::new(), 0);
    for (lib, data) in every_metallib() {
        let ours = compute_answers::<VulkanPlatform>(&d, &data);
        for (name, o) in &ours {
            let m = reims_vgpu_core::metal_value!(format!("{lib} {name}"));
            let made = m != "None";
            match o {
                Ok(o) if made && format!("{:?}", Some(o)) == m => compared += 1,
                Err(o) if made && o.starts_with("died") => died.push(name.clone()),
                Err(_) if !made => {}
                _ => differ.push(format!("{lib} {name}: ours {o:?}, Metal's {m}")),
            }
        }
    }
    assert!(
        differ.is_empty(),
        "{} differ:\n{}",
        differ.len(),
        differ.join("\n")
    );
    assert_eq!(died, ["sample_held_sampler_float"]);
    assert!(compared > 100, "{compared} kernels");
}

/// What a texture `t` describes answers of the getters core reads, made on `d` (by
/// `newSharedTextureWithDescriptor:` where `shared`): its width, height, depth, array length,
/// levels, samples, pixel format, type, usage, storage mode, resource options, whether it is
/// shareable and its protection options; `None` for nil.
fn texture_answers<P: Platform>(
    d: &P::Device,
    desc: &dyn Fn(&TextureDesc) -> P::TextureDescriptor,
    t: &TextureDesc,
    shared: bool,
) -> Option<[u64; 13]> {
    let desc = desc(t);
    let made = match shared {
        true => d.new_shared_texture(&desc),
        false => d.new_texture(OnFault::Return, &desc),
    };
    made.unwrap().map(|t| {
        [
            t.width(),
            t.height(),
            t.depth(),
            t.array_length(),
            t.mipmap_level_count(),
            t.sample_count(),
            t.pixel_format(),
            t.texture_type(),
            t.usage(),
            t.storage_mode(),
            t.resource_options(),
            u64::from(t.is_shareable()),
            t.protection_options(),
        ]
    })
}

/// The texture descriptors [`textures_and_buffers_answer_as_metals_across_descriptors`] makes:
/// RGBA8Unorm and RGBA16Float textures of every type, 8 texels wide, of one level and of the
/// whole chain (one level for 1D and multisample types), of six usages (none, `ShaderRead`, with
/// `ShaderWrite`, with `RenderTarget`, with `PixelFormatView`, and 0x8 alone; no render target
/// of a 1D type, multisample ones read alone or with `RenderTarget`), under each resource
/// option (the default, Shared, Managed, Private, untracked, tracked, write-combined, Private
/// untracked; a multisample texture Private alone); Depth32Float ones of the 2D, 2D array,
/// cube and cube array types, Private, Private untracked and Private tracked; and shareable
/// Private 2D textures. Each is the descriptor and whether it is shareable.
fn swept_textures() -> Vec<(TextureDesc, bool)> {
    let shapes: [(u64, [u64; 3], u64, u64); 8] = [
        (0, [8, 1, 1], 1, 1),
        (1, [8, 1, 1], 2, 1),
        (2, [8, 4, 1], 1, 1),
        (3, [8, 4, 1], 3, 1),
        (4, [8, 4, 1], 1, 4),
        (5, [8, 8, 1], 1, 1),
        (6, [8, 8, 1], 2, 1),
        (7, [8, 4, 4], 1, 1),
    ];
    let options = [
        None,
        Some(0x0),
        Some(0x10),
        Some(0x20),
        Some(0x100),
        Some(0x200),
        Some(0x1),
        Some(0x120),
    ];
    let mut out = Vec::new();
    for (texture_type, extent, array_length, sample_count) in shapes {
        let one_dimensional = texture_type <= 1;
        let multisample = sample_count > 1;
        let levels: &[u64] = if one_dimensional || multisample {
            &[1]
        } else {
            &[1, 4]
        };
        let usages: &[u64] = match (one_dimensional, multisample) {
            (_, true) => &[1, 5],
            (true, false) => &[0, 1, 3, 0x11, 0x8],
            (false, false) => &[0, 1, 3, 5, 0x11, 0x8],
        };
        let resource_options: &[Option<u64>] = if multisample {
            &[Some(0x20), Some(0x120)]
        } else {
            &options
        };
        for pixel_format in [70, 115] {
            for &mipmap_level_count in levels {
                for &usage in usages {
                    for &resource_options in resource_options {
                        let desc = TextureDesc {
                            texture_type,
                            pixel_format,
                            width: extent[0],
                            height: extent[1],
                            depth: extent[2],
                            mipmap_level_count,
                            sample_count,
                            array_length,
                            usage,
                            resource_options,
                            ..TextureDesc::default()
                        };
                        out.push((desc, false));
                    }
                }
            }
        }
        if matches!(texture_type, 2 | 3 | 5 | 6) {
            for mipmap_level_count in [1, 4] {
                for usage in [1, 5] {
                    for resource_options in [Some(0x20), Some(0x120), Some(0x220)] {
                        let desc = TextureDesc {
                            texture_type,
                            pixel_format: 252,
                            width: extent[0],
                            height: extent[1],
                            mipmap_level_count,
                            array_length,
                            usage,
                            resource_options,
                            ..TextureDesc::default()
                        };
                        out.push((desc, false));
                    }
                }
            }
        }
    }
    for usage in [1, 3, 5] {
        for resource_options in [Some(0x20), Some(0x120)] {
            let desc = TextureDesc {
                width: 8,
                height: 4,
                usage,
                resource_options,
                ..TextureDesc::default()
            };
            out.push((desc, true));
        }
    }
    out
}

/// Every texture of [`swept_textures`] (1182 descriptors) answers the getters core reads
/// ([`texture_answers`]) as Metal's does on this host, and a buffer of each of five lengths
/// (1 byte to past 16 pages) and eleven resource options (Shared, Managed and Private, each
/// default, untracked and tracked, write-combined and write-combined untracked) answers its
/// length, resource options, storage mode and whether its contents are NULL as Metal's, but
/// for two differences no answer of core's reaches (measured):
/// - a texture of a descriptor that sets no resource options: Metal's is Managed (options
///   0x210), this platform's Shared (0x200, the tracked mode Metal's options hold too). Core
///   reads a texture's storage mode only to refuse a memoryless one, and no texture's options;
/// - a Private buffer of at most a page (1 to 4096 bytes here; not 65537): Metal's `contents`
///   is not NULL, from its private pool, this platform's is. Core reads only a Shared buffer's
///   contents.
#[test]
fn textures_and_buffers_answer_as_metals_across_descriptors() {
    let d = dev();
    let p = platform();
    let vulkan_desc = |t: &TextureDesc| p.texture_descriptor(t);
    let mut differ = Vec::new();
    let textures = swept_textures();
    for (t, shared) in &textures {
        let key = format!("{t:?} shared {shared}");
        let ours = catch_death(|| texture_answers::<VulkanPlatform>(&d, &vulkan_desc, t, *shared));
        // The first difference of the test's docs: the storage mode and options of a texture
        // of no resource options, ours Shared (0, 0x200) where Metal's is Managed (1, 0x210).
        let known = |o: &Option<[u64; 13]>| match *o {
            Some(mut o) if t.resource_options.is_none() && (o[9], o[10]) == (0, 0x200) => {
                (o[9], o[10]) = (1, 0x210);
                reims_vgpu_core::metal_matches!(key, Some(o))
            }
            _ => false,
        };
        match ours {
            Ok(ours) if reims_vgpu_core::metal_matches!(key, ours) || known(&ours) => {}
            ours => differ.push(format!("{key}: ours {ours:?}, not Metal's")),
        }
    }
    for length in [1, 4, 100, 4096, 65537] {
        for options in [
            0, 0x100, 0x200, 0x10, 0x110, 0x210, 0x20, 0x120, 0x220, 0x1, 0x101,
        ] {
            let key = format!("a buffer of {length} bytes, options {options:#x}");
            let ours = catch_death(|| {
                d.new_buffer_of_length(OnFault::Return, length, options)
                    .unwrap()
                    .map(|b| {
                        (
                            b.length(),
                            b.resource_options(),
                            b.storage_mode(),
                            b.contents().is_null(),
                        )
                    })
            });
            // The second difference of the test's docs: a small Private buffer's contents, NULL
            // here where Metal's are not.
            let pooled = length <= 4096 && options & 0xf0 == 0x20;
            let known = |o: &Option<(u64, u64, u64, bool)>| match *o {
                Some((l, o, 2, true)) if pooled => {
                    reims_vgpu_core::metal_matches!(key, Some((l, o, 2_u64, false)))
                }
                _ => false,
            };
            match ours {
                Ok(ours) if reims_vgpu_core::metal_matches!(key, ours) || known(&ours) => {}
                ours => differ.push(format!("{key}: ours {ours:?}, not Metal's")),
            }
        }
    }
    assert!(
        differ.is_empty(),
        "{} of {} differ:\n{}",
        differ.len(),
        textures.len() + 55,
        differ.join("\n")
    );
}

/// newBufferWithIOSurface: makes a Shared buffer over the surface's memory that holds one
/// reference of the surface while it lives, whose contents are NULL, as Metal's; iosurface
/// answers it, +1
#[test]
fn buffers_on_surfaces() {
    let d = dev();
    let s = bgra_surface(64, 32);
    assert_eq!(s.retain_count(), 1);
    let b = d.new_buffer_on_surface(Some(&s)).unwrap().unwrap();
    assert_eq!(s.retain_count(), 2);
    let seen = (b.contents(), b.length(), b.storage_mode());
    assert_eq!(seen, (HostPtr::NULL, s.alloc_size(), 0));
    let again = b.iosurface().unwrap();
    assert_eq!(again.id(), s.id());
    assert_eq!(s.retain_count(), 3);
    drop(again);
    drop(b);
    assert_eq!(s.retain_count(), 1);
}

/// Every buffer over a surface answers the surface's address, as on Metal (where it runs, two
/// live ones answer one address, and so does one made after the first's release while a
/// command buffer that fills it waits): a buffer made while another holds the address shares
/// a live one's range, which stays while one holds it. One made while none lives, after the
/// last's release while a command buffer that fills it waits, dies at
/// newBufferWithIOSurface:, naming the surface, as the runtime holds the released one's range
/// until that command buffer completed and shares a range with a live buffer alone. Then a
/// buffer over the surface lies at the address again
#[test]
fn buffers_over_a_surface_share_its_address() {
    use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder as _};
    use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
    use reims_vgpu_core::platform::event::{EventRef, SharedEvent as _};
    fn waiting_fill<P: Platform>(
        d: &P::Device,
        buffer: &P::Buffer,
    ) -> (P::CommandBuffer, P::SharedEvent) {
        let q = d.new_command_queue().unwrap();
        let gate = d.new_shared_event().unwrap();
        let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
        cb.encode_wait(Some(EventRef::Shared(&gate)), 1).unwrap();
        let e = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
        let fill = BlitCmd::FillBuffer {
            buffer,
            range: NsRange::new(0, 16),
            value: 0x5c,
        };
        e.encode(OnFault::Return, fill).unwrap();
        e.encode(OnFault::Return, BlitCmd::EndEncoding).unwrap();
        cb.commit(OnFault::Return).unwrap();
        (cb, gate)
    }
    fn completed<P: Platform>((cb, gate): (P::CommandBuffer, P::SharedEvent)) {
        gate.set_signaled_value(1).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        assert!(cb.error().is_none(), "{:?}", cb.error());
    }
    let d = dev();
    let s = bgra_surface(64, 64);
    let first = d.new_buffer_on_surface(Some(&s)).unwrap().unwrap();
    let at = first.gpu_address();
    let second = d.new_buffer_on_surface(Some(&s)).unwrap().unwrap();
    assert_eq!(second.gpu_address(), at, "the first live");
    drop(first);
    let third = d.new_buffer_on_surface(Some(&s)).unwrap().unwrap();
    assert_eq!(
        third.gpu_address(),
        at,
        "the first released, the second live"
    );
    drop(third);
    let waiting = waiting_fill::<VulkanPlatform>(&d, &second);
    drop(second);
    *crate::answer::LAST_LINE.lock() = None;
    let site = catch_death(|| drop(d.new_buffer_on_surface(Some(&s)))).unwrap_err();
    let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
    assert_eq!(
        (site.kind, site.file),
        (DeathKind::Abort, "newBufferWithIOSurface:"),
        "{line}"
    );
    let named = format!(
        "a buffer over IOSurface {:#x} while a released one",
        s.id().0
    );
    assert!(line.contains(&named), "{line}");
    completed::<VulkanPlatform>(waiting);
    let again = d.new_buffer_on_surface(Some(&s)).unwrap().unwrap();
    assert_eq!(again.gpu_address(), at, "made once the fill completed");
}

/// A 2D array over a no-copy buffer of host memory made by
/// `newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:` is Metal's, byte for
/// byte: its slices as made, after the CPU rewrote the memory (both alias it), and the memory
/// after a blit copied into every slice (where its slices lie). Metal's is measured first
#[test]
fn array_over_host_memory_is_metals() {
    let steps = ["made", "rewritten", "after"];
    let ours = array_over_host_memory_on::<VulkanPlatform>(
        &dev(),
        &|t| platform().texture_descriptor(t),
        true,
    );
    for (step, ours) in steps.iter().zip(&ours) {
        assert!(reims_vgpu_core::metal_matches_bytes!(step, ours), "{step}");
    }
}

/// newBufferWithIOSurface: of a surface over the guest's pages at an offset into its
/// allocation (`kIOSurfaceOffset`) reads as Metal's does: as long as the allocation, from the
/// surface's base address, its last bytes, past the allocation's end, 0
#[test]
fn buffers_on_offset_surfaces_read_as_metals() {
    use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
    use reims_vgpu_core::platform::compute::{ComputeCmd, ComputeEncoder as _};
    use reims_vgpu_core::platform::os::vm::HostVm as _;
    use reims_vgpu_core::platform::resource::Library as _;
    // `copy_words` of the surface's buffer, on `d`: its length, and each 32-bit word of it.
    fn copied<P: Platform<Surface = OsSurface>>(d: &P::Device, s: &OsSurface) -> (u64, Vec<u32>) {
        let b = d.new_buffer_on_surface(Some(s)).unwrap().unwrap();
        let length = b.length();
        let library = d
            .new_library(crate::gpu::ycbcr::measure::PROBE)
            .unwrap()
            .unwrap()
            .unwrap();
        let f = library
            .new_function(OnFault::Die, "copy_words")
            .unwrap()
            .unwrap();
        let copies = d
            .new_compute_pipeline_state_for_function(Some(&f))
            .unwrap()
            .unwrap();
        let out = d
            .new_buffer_of_length(OnFault::Die, length, 0)
            .unwrap()
            .unwrap();
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = cb.compute_encoder(OnFault::Die).unwrap().unwrap();
        let buffers = [Some(&b), Some(&out)];
        for cmd in [
            ComputeCmd::SetComputePipelineState {
                state: Some(&copies),
            },
            ComputeCmd::SetBuffers {
                buffers: &buffers,
                offsets: &[0, 0],
                range: NsRange::new(0, 2),
            },
            ComputeCmd::DispatchThreads {
                threads_per_grid: MtlSize::new(length / 4, 1, 1),
                threads_per_threadgroup: MtlSize::new(64, 1, 1),
            },
            ComputeCmd::EndEncoding,
        ] {
            e.encode(OnFault::Die, cmd).unwrap();
        }
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        let n = usize::try_from(length / 4).unwrap();
        // SAFETY: a Shared buffer's words, read after the command buffer completed.
        let words = unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u32>(), n) };
        (length, words.to_vec())
    }
    for offset in [0, 0x40, 0x1000, 0x4000] {
        let (s, pages, length) = crate::os::tests::bgra_surface_over_at(64, 32, offset);
        // Each word of the pages holds its own offset there, its top bit set.
        for at in (0..length).step_by(4) {
            // SAFETY: a word of the test's pages, which no GPU work uses now.
            unsafe {
                pages
                    .as_ptr::<u8>()
                    .add(usize::try_from(at).unwrap())
                    .cast::<u32>()
                    .write(0x8000_0000 | u32::try_from(at).unwrap());
            }
        }
        // Metal's first, before this platform's runtime makes any buffer of its own.
        let (ours, words) = copied::<VulkanPlatform>(&dev(), &s);
        assert_eq!(ours, length, "{offset:#x}: the allocation's size");
        let past = usize::try_from((length - offset) / 4).unwrap();
        assert_eq!(
            words[0],
            0x8000_0000 | u32::try_from(offset).unwrap(),
            "{offset:#x}"
        );
        assert!(
            words[past..].iter().all(|&w| w == 0),
            "{offset:#x}: 0 past the allocation"
        );
        let key = format!("{offset:#x}");
        let metals = reims_vgpu_core::metal_matches_raw!(key, (ours, words));
        assert!(metals, "{offset:#x}: Metal's");
        drop(s);
        crate::os::OsVm::new().deallocate(pages, length);
    }
}

/// The answers of setPurgeableState: to KeepCurrent, Volatile, NonVolatile, Empty,
/// KeepCurrent and NonVolatile in turn, of each of `P`'s resources: Shared buffers of 256
/// bytes and 16 KiB; Shared textures of 8 by 8 and 256 by 256 BGRA8Unorm, a view of the larger,
/// a texture over the larger buffer; and Private shareable textures of 8 by 8 and 256 by 256.
fn purgeable_answers_on<P: Platform>(
    d: &P::Device,
    desc: &dyn Fn(&TextureDesc) -> P::TextureDescriptor,
) -> Vec<Vec<u64>>
where
    P::Device: GpuDevice<P>,
{
    use reims_vgpu_core::platform::resource::{Buffer as _, GpuResource, Texture as _};
    let states = [1, 3, 2, 4, 1, 2];
    let answers = |r: &dyn GpuResource<P>| {
        states
            .iter()
            .map(|&s| r.set_purgeable_state(s).unwrap())
            .collect::<Vec<_>>()
    };
    let texture = |side: u64, options: u64| {
        desc(&TextureDesc {
            pixel_format: 80,
            width: side,
            height: side,
            usage: 0x13,
            resource_options: Some(options),
            ..TextureDesc::default()
        })
    };
    let buffer = |length| {
        d.new_buffer_of_length(OnFault::Return, length, 0)
            .unwrap()
            .unwrap()
    };
    let (small, large) = (buffer(256), buffer(16384));
    let small_texture = d.new_texture(OnFault::Return, &texture(8, 0)).unwrap();
    let large_texture = d.new_texture(OnFault::Return, &texture(256, 0)).unwrap();
    let (small_texture, large_texture) = (small_texture.unwrap(), large_texture.unwrap());
    let view = large_texture.new_texture_view(81).unwrap().unwrap();
    let over = large
        .new_texture(OnFault::Return, &texture(8, 0), 0, 256)
        .unwrap()
        .unwrap();
    let shared_small = d.new_shared_texture(&texture(8, 0x20)).unwrap().unwrap();
    let shared_large = d.new_shared_texture(&texture(256, 0x20)).unwrap().unwrap();
    vec![
        answers(&small),
        answers(&large),
        answers(&small_texture),
        answers(&large_texture),
        answers(&view),
        answers(&over),
        answers(&shared_small),
        answers(&shared_large),
    ]
}

/// setPurgeableState: answers each resource as Metal does on the M3 Max: the
/// state before for a buffer of 16 KiB, a texture of its own allocation of 16 KiB or more and
/// a shareable texture; NonVolatile to every request for a smaller buffer or texture, a view
/// and a texture over a buffer
#[test]
fn purgeable_states_are_the_metal_platforms() {
    let ours =
        purgeable_answers_on::<VulkanPlatform>(&dev(), &|t| platform().texture_descriptor(t));
    let (kept, non_volatile) = ([2, 2, 3, 2, 4, 4], [2; 6]);
    let expected = [
        non_volatile,
        kept,
        non_volatile,
        kept,
        non_volatile,
        non_volatile,
        kept,
        kept,
    ];
    assert_eq!(ours, expected);
}

/// A colour texture's heap size and alignment are Metal's on the M3 Max, with
/// allowGPUOptimizedContents on, off and unset (a depth or stencil texture's:
/// `heap_depth_stencil_sizes_with_optimized_contents_off`), and so are a two-plane XR10
/// texture's (550, 551), whose two planes Metal sizes: 2D of one and seven levels, at a
/// display's size, and a 2D array
#[test]
fn heap_texture_sizes_are_the_metal_platforms() {
    for pixel_format in [550, 551] {
        for (texture_type, width, height, mipmap_level_count, array_length) in [
            (2, 64, 64, 1, 1),
            (2, 64, 64, 7, 1),
            (2, 1920, 1080, 1, 1),
            (3, 256, 256, 1, 4),
        ] {
            let t = TextureDesc {
                texture_type,
                pixel_format,
                width,
                height,
                mipmap_level_count,
                array_length,
                usage: 0x1,
                resource_options: Some(0x20),
                ..TextureDesc::default()
            };
            let ours = dev()
                .heap_texture_size_and_align(&platform().texture_descriptor(&t))
                .unwrap();
            reims_vgpu_core::metal_check!(format!("{t:?}"), ours);
        }
    }
    for (pixel_format, width, height, mipmap_level_count, usage, storage) in [
        (70, 256, 256, 1, 0x1, 0x20),
        (80, 64, 32, 7, 0x13, 0x20),
        (80, 100, 100, 1, 0x5, 0),
        (115, 33, 17, 1, 0x3, 0x20),
    ] {
        for allow_gpu_optimized_contents in [None, Some(true), Some(false)] {
            let t = TextureDesc {
                pixel_format,
                width,
                height,
                mipmap_level_count,
                usage,
                resource_options: Some(storage),
                allow_gpu_optimized_contents,
                ..TextureDesc::default()
            };
            let ours = dev()
                .heap_texture_size_and_align(&platform().texture_descriptor(&t))
                .unwrap();
            reims_vgpu_core::metal_check!(format!("{t:?}"), ours);
        }
    }
}

/// A 2- or 4-sample texture's heap size and alignment are Metal's on the M3 Max
/// (the runtime answers Metal's rule): 2D
/// multisample textures and multisample arrays of one and three slices, of colour formats of
/// 4 and 8 bytes and of the depth and stencil formats, from 4 by 4 to 256 by 256, render
/// targets read by shaders or not, and colour ones of `PixelFormatView` usage or with
/// `allowGPUOptimizedContents` off, which Metal keeps uncompressed. Each shape whose answer
/// differs, or whose query dies, is named
#[test]
fn multisample_heap_texture_sizes_are_the_metal_platforms() {
    let (p, d) = (platform(), dev());
    let colour = [70, 80, 115];
    let depth = [252, 253, 260];
    let mut shapes = Vec::new();
    for pixel_format in colour.into_iter().chain(depth) {
        let ways: &[(u64, Option<bool>)] = match colour.contains(&pixel_format) {
            true => &[(0x4, None), (0x5, None), (0x14, None), (0x5, Some(false))],
            false => &[(0x4, None), (0x5, None)],
        };
        for sample_count in [2, 4] {
            for (texture_type, array_length) in [(4, 1), (8, 1), (8, 3)] {
                for (width, height) in [(4, 4), (16, 8), (16, 16), (100, 60), (256, 256)] {
                    for &(usage, allow_gpu_optimized_contents) in ways {
                        shapes.push(TextureDesc {
                            texture_type,
                            pixel_format,
                            width,
                            height,
                            sample_count,
                            array_length,
                            usage,
                            resource_options: Some(0x20),
                            allow_gpu_optimized_contents,
                            ..TextureDesc::default()
                        });
                    }
                }
            }
        }
    }
    let mut differ = Vec::new();
    for t in &shapes {
        let key = format!("{t:?}");
        *crate::answer::LAST_LINE.lock() = None;
        let ours = catch_death(|| {
            d.heap_texture_size_and_align(&p.texture_descriptor(t))
                .unwrap()
        });
        match ours {
            Ok(ours) if reims_vgpu_core::metal_matches!(key, ours) => {}
            Ok(ours) => differ.push(format!("{t:?}: ours {ours:?}, not Metal's")),
            Err(site) => {
                let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
                differ.push(format!("{t:?}: died at {site:?}: {line}"));
            }
        }
    }
    assert!(
        differ.is_empty(),
        "{} of {}: {differ:#?}",
        differ.len(),
        shapes.len()
    );
}

/// Depth and stencil textures without allowGPUOptimizedContents, private, read by shaders and
/// rendered to, take what Metal's of their shape take of its space, though the runtime's
/// image is larger (`metal_space::own_heap_size`): on Metal and here alike, how far the next
/// range moves as one is made, and how far a private buffer of 0x100 bytes made after it lies
/// past one made before it, its slot where it takes one in the private pool. Each made and
/// released in an autorelease pool of its own; Metal's answers were recorded with every shape
/// made before this platform made anything in the process: MoltenVK's images lie in the same
/// Metal space, where they would move Metal's next range
#[test]
fn depth_stencil_textures_without_optimized_contents_take_metals_space() {
    use reims_vgpu_core::platform::resource::Buffer as _;
    use reims_vgpu_core::rt::Pool;
    fn taken<P: Platform>(d: &P::Device, desc: &P::TextureDescriptor) -> (u64, u64) {
        let _pool = Pool::here();
        let before = next_range::<P>(d);
        let small = || {
            let buffer = d.new_buffer_of_length(OnFault::Die, 0x100, 0x20);
            buffer.unwrap().unwrap()
        };
        let first = small();
        let texture = d.new_texture(OnFault::Die, desc).unwrap().unwrap();
        let second = small();
        let range = next_range::<P>(d) - before;
        let slot = second.gpu_address() - first.gpu_address() - 0x100;
        drop((texture, first, second));
        (range, slot)
    }
    let mut shapes = Vec::new();
    for pixel_format in [260, 252, 250, 253] {
        for side in [4, 8, 16, 64, 100, 128] {
            for sample_count in [1, 4] {
                shapes.push(TextureDesc {
                    texture_type: if sample_count > 1 { 4 } else { 2 },
                    pixel_format,
                    width: side,
                    height: side,
                    sample_count,
                    usage: 0x5,
                    resource_options: Some(0x20),
                    allow_gpu_optimized_contents: Some(false),
                    ..TextureDesc::default()
                });
            }
        }
    }
    let key = |t: &TextureDesc| {
        format!(
            "{} {}x{} {}x",
            t.pixel_format, t.width, t.width, t.sample_count
        )
    };
    let (p, d) = (platform(), dev());
    let _queue = d.new_command_queue().unwrap();
    let differ: Vec<String> = (shapes.iter())
        .filter_map(|t| {
            let ours = taken::<VulkanPlatform>(&d, &p.texture_descriptor(t));
            let metals = reims_vgpu_core::metal_matches!(key(t), ours);
            (!metals).then(|| format!("{}: ours {ours:x?}, not Metal's", key(t)))
        })
        .collect();
    assert!(differ.is_empty(), "{differ:#?}");
}

/// A depth or stencil texture's heap size with allowGPUOptimizedContents off is Metal's: a
/// 128x128 `Depth32Float` or `Depth32Float_Stencil8` one's, which the runtime lays out in
/// Metal's tiles on MoltenVK, is the runtime's own answer too; where the runtime keeps one's
/// GPU-optimized layout (`OptimizedDepthStencilTexture`), the query answers Metal's, the
/// runtime's model of it. Metal's size and alignment for each format, 4x4 to
/// 128x128, of 1 and 4 samples, at each of [`DEPTH_USAGES`], where the runtime's own answer is
/// at least as many bytes, at an alignment Metal's divides, an X32_Stencil8 texture's among
/// them, a `Stencil8` one's of its shape. With it on or unset the size is
/// Metal's: a `Depth32Float_Stencil8` one's is the runtime's, which models its layout and
/// announces nothing, Metal's at this shape
#[test]
fn heap_depth_stencil_sizes_with_optimized_contents_off() {
    let sel = "heapTextureSizeAndAlignWithDescriptor:";
    let d = dev();
    let shape_of = |pixel_format, side, sample_count, allow, usage| TextureDesc {
        texture_type: if sample_count > 1 { 4 } else { 2 },
        pixel_format,
        width: side,
        height: side,
        sample_count,
        usage,
        resource_options: Some(0x20),
        allow_gpu_optimized_contents: allow,
        ..TextureDesc::default()
    };
    let shape = |pixel_format, side, sample_count, allow| {
        shape_of(pixel_format, side, sample_count, allow, 0x5)
    };
    for (pixel_format, metals) in [(252, 65536), (260, 81920)] {
        let t = shape(pixel_format, 128, 1, Some(false));
        let off = platform().texture_descriptor(&t);
        let raw = off.fields().descriptor(sel, Maker::SizeQuery);
        let answer = plain(|| d.raw.heap_texture_size_and_align(&raw)).unwrap();
        assert_eq!(answer.size, metals, "{pixel_format}: the runtime's");
        let ours = d.heap_texture_size_and_align(&off).unwrap();
        assert_eq!(ours.size, metals, "{pixel_format}");
        for allow in [None, Some(true)] {
            let t = shape(pixel_format, 128, 1, allow);
            let on = platform().texture_descriptor(&t);
            let ours = d.heap_texture_size_and_align(&on);
            if pixel_format == 260 {
                let raw = on.fields().descriptor(sel, Maker::SizeQuery);
                let answer = plain(|| d.raw.heap_texture_size_and_align(&raw)).unwrap();
                let unmeasured = rt::Degradation::UnmeasuredHeapTextureSize;
                assert!(!crate::answer::RELAYED.lock().contains(&unmeasured));
                let ours = ours.as_ref().unwrap();
                assert_eq!((ours.size, ours.align), (answer.size, answer.align));
            }
            let key = format!("{pixel_format} {allow:?}: the size");
            reims_vgpu_core::metal_check!(key, ours.unwrap().size);
        }
    }
    let mut differ = Vec::new();
    for pixel_format in [260, 252, 250, 253] {
        for side in [4, 8, 16, 64, 100, 128] {
            for samples in [1, 4] {
                for usage in DEPTH_USAGES {
                    let t = shape_of(pixel_format, side, samples, Some(false), usage);
                    let case = format!("{pixel_format} {side} {samples}x usage {usage:#x}");
                    let desc = platform().texture_descriptor(&t);
                    let ours = d.heap_texture_size_and_align(&desc).unwrap();
                    let raw = desc.fields().descriptor(sel, Maker::SizeQuery);
                    let runtimes = plain(|| d.raw.heap_texture_size_and_align(&raw)).unwrap();
                    if runtimes.size < ours.size || !runtimes.align.is_multiple_of(ours.align) {
                        differ.push(format!("{case}: the runtime's {runtimes:x?} under"));
                    }
                    if !reims_vgpu_core::metal_matches!(case, ours) {
                        differ.push(format!("{case}: {ours:x?}, not Metal's"));
                    }
                }
            }
        }
    }
    assert!(differ.is_empty(), "ours, Metal's: {differ:#?}");
    let t = shape(261, 16, 1, Some(false));
    let x32s8 = platform().texture_descriptor(&t);
    let ours = d.heap_texture_size_and_align(&x32s8).unwrap();
    let raw = x32s8.fields().descriptor(sel, Maker::SizeQuery);
    assert_eq!(
        Some(ours),
        d.metal_answer_at(sel, &raw),
        "X32_Stencil8: the model's"
    );
    reims_vgpu_core::metal_check!("X32_Stencil8", ours);
}

/// The size query of each of [`LOSSLESS_MIPMAPPED_SHAPES`], which answers the runtime's model of
/// Metal's on Metal ([`lossless_mipmapped_heap_texture_sizes_are_modelled`], on every OS), is
/// Metal's on the M3 Max. For the `R32Uint` one the device's image takes 0x5680 bytes, where
/// Metal answers 0x5800
#[test]
fn lossless_mipmapped_heap_texture_sizes_are_metals() {
    let (p, d) = (platform(), dev());
    for (name, t) in lossless_mipmapped_shapes() {
        let ours = d
            .heap_texture_size_and_align(&p.texture_descriptor(&t))
            .unwrap();
        reims_vgpu_core::metal_check!(name, ours);
    }
}

/// The depth and stencil shapes the device tests below ask of both platforms: a name, texture
/// type, levels (0 for the full chain), slices and samples. Each is but a 2D or multisample one
/// of one slice and level ([`heap_depth_stencil_sizes_with_optimized_contents_off`]'s).
const DEPTH_SHAPES: [(&str, u64, u64, u64, u64); 10] = [
    ("two levels", 2, 2, 1, 1),
    ("every level", 2, 0, 1, 1),
    ("a 2D array of 2", 3, 1, 2, 1),
    ("a 2D array of 3, every level", 3, 0, 3, 1),
    ("a cube", 5, 1, 1, 1),
    ("a cube, every level", 5, 0, 1, 1),
    ("a cube array of 2", 6, 1, 2, 1),
    ("a cube array of 2, every level", 6, 0, 2, 1),
    ("a cube array of 3, two levels", 6, 2, 3, 1),
    ("a multisample array of 2", 8, 1, 2, 4),
];

/// The usages the depth and stencil device tests ask at: `ShaderRead`, `ShaderRead |
/// ShaderWrite`, `RenderTarget`, `ShaderRead | RenderTarget` and `ShaderRead | ShaderWrite |
/// PixelFormatView`.
const DEPTH_USAGES: [u64; 5] = [0x1, 0x3, 0x4, 0x5, 0x13];

/// A `Private` texture of `pixel_format` of a shape of [`DEPTH_SHAPES`], `side` by `side`, of
/// `usage`, with allowGPUOptimizedContents off.
fn depth_shape(
    pixel_format: u64,
    (texture_type, levels, slices, samples): (u64, u64, u64, u64),
    side: u64,
    usage: u64,
) -> TextureDesc {
    TextureDesc {
        texture_type,
        pixel_format,
        width: side,
        height: side,
        mipmap_level_count: match levels {
            0 => u64::from(side.ilog2()) + 1,
            n => n,
        },
        array_length: slices,
        sample_count: samples,
        usage,
        resource_options: Some(0x20),
        allow_gpu_optimized_contents: Some(false),
        ..TextureDesc::default()
    }
}

/// A shape of [`DEPTH_MORE_SHAPES`]: a name, texture type, extent, levels (0 for the full
/// chain), slices and samples.
type DepthShape = (&'static str, u64, (u64, u64), u64, u64, u64);

/// The representative shapes of the classes the wider measurements found Metal answering the
/// rules for, which the device test below asks beside [`DEPTH_SHAPES`]: a name, texture type,
/// extent, levels (0 for the full chain), slices and samples.
const DEPTH_MORE_SHAPES: [DepthShape; 14] = [
    ("1x1", 2, (1, 1), 1, 1, 1),
    ("1440x1080", 2, (1440, 1080), 1, 1, 1),
    ("3840x2160, every level", 2, (3840, 2160), 0, 1, 1),
    ("1000x1000, three levels", 2, (1000, 1000), 3, 1, 1),
    ("333x777, every level", 2, (333, 777), 0, 1, 1),
    (
        "a 2D array of 8, 129x64, every level",
        3,
        (129, 64),
        0,
        8,
        1,
    ),
    ("a 2D array of 2048, 4x4", 3, (4, 4), 1, 2048, 1),
    ("a cube of 200, every level", 5, (200, 200), 0, 1, 1),
    (
        "a cube array of 4, 129, three levels",
        6,
        (129, 129),
        3,
        4,
        1,
    ),
    ("a cube array of 341, 4", 6, (4, 4), 1, 341, 1),
    ("2 samples, 1440x1080", 4, (1440, 1080), 1, 1, 2),
    ("4 samples, 1920x1080", 4, (1920, 1080), 1, 1, 4),
    (
        "a multisample array of 5 at 2 samples, 100x37",
        8,
        (100, 37),
        1,
        5,
        2,
    ),
    (
        "a multisample array of 8 at 4 samples, 129x64",
        8,
        (129, 64),
        1,
        8,
        4,
    ),
];

/// The usages the device test below asks [`DEPTH_MORE_SHAPES`] at beside [`DEPTH_USAGES`],
/// of the sets the measurements found Metal answering the rules for: none (`Unknown`),
/// `ShaderWrite | RenderTarget`, `ShaderRead | RenderTarget | PixelFormatView`, `ShaderRead
/// | RenderTarget` with 0x10000, and a macOS guest's 0x1c01d; and sets holding neither
/// `ShaderRead` nor `RenderTarget` but another bit: `ShaderWrite`, `PixelFormatView`,
/// `ShaderWrite | PixelFormatView`, and 0x4000, 0x8000 and 0x10000 alone.
const DEPTH_MORE_USAGES: [u64; 11] = [
    0x0, 0x6, 0x15, 0x10005, 0x1c01d, 0x2, 0x10, 0x12, 0x4000, 0x8000, 0x10000,
];

/// `heapTextureSizeAndAlignWithDescriptor:` of a depth or stencil texture with
/// allowGPUOptimizedContents off, of `Depth16Unorm`, `Depth32Float`, `Stencil8` and
/// `Depth32Float_Stencil8`, at `Private` and `Shared` storage: of each shape of
/// [`DEPTH_SHAPES`], 4x4 to 128x128, at each of [`DEPTH_USAGES`]; of each of
/// [`DEPTH_MORE_SHAPES`] at those and [`DEPTH_MORE_USAGES`]; and of a 64x64 2D one at
/// `ShaderRead | RenderTarget` with the write-combined CPU cache mode, untracked hazard
/// tracking, `framebufferOnly` and protection options 0x1. The query answers each, the Metal
/// platform's answer (the runtime's model of it, `VulkanDevice::metal_answer_at`), a
/// multisample one at `Shared` storage among them, whose image the runtime makes in memory of
/// its own; and the runtime's own answer is at least as many bytes, at an alignment the
/// model's divides, which a heap's placement of such a texture relies on
/// (`HeapObject::new_texture`)
#[test]
fn heap_depth_stencil_shapes_with_optimized_contents_off_are_metals() {
    let sel = "heapTextureSizeAndAlignWithDescriptor:";
    let (p, d) = (platform(), dev());
    let mut rows = Vec::new();
    for (name, texture_type, levels, slices, samples) in DEPTH_SHAPES {
        for side in [4u64, 16, 64, 128] {
            for usage in DEPTH_USAGES {
                let shape = (texture_type, levels, slices, samples);
                let t = depth_shape(0, shape, side, usage);
                rows.push((format!("{name} {side} usage {usage:#x}"), t));
            }
        }
    }
    for (name, texture_type, extent, levels, slices, samples) in DEPTH_MORE_SHAPES {
        for usage in DEPTH_USAGES.into_iter().chain(DEPTH_MORE_USAGES) {
            let t = wide_row(texture_type, extent, levels, slices, samples, usage);
            rows.push((format!("{name} usage {usage:#x}"), t));
        }
    }
    let plain_64 = wide_row(2, (64, 64), 1, 1, 1, 0x5);
    let properties: [(&str, TextureDesc); 4] = [
        (
            "write-combined",
            TextureDesc {
                resource_options: Some(0x1),
                ..plain_64
            },
        ),
        (
            "untracked",
            TextureDesc {
                resource_options: Some(0x100),
                ..plain_64
            },
        ),
        (
            "framebufferOnly",
            TextureDesc {
                framebuffer_only: Some(true),
                ..plain_64
            },
        ),
        (
            "protection options 0x1",
            TextureDesc {
                protection_options: Some(0x1),
                ..plain_64
            },
        ),
    ];
    for (name, t) in properties {
        rows.push((format!("64x64 {name}"), t));
    }
    let mut differ = Vec::new();
    for pixel_format in [250, 252, 253, 260] {
        for (storage, options) in [("Private", 0x20), ("Shared", 0)] {
            for (name, row) in &rows {
                let case = format!("{pixel_format} {storage} {name}");
                let mut t = *row;
                t.pixel_format = pixel_format;
                // The row's options but its storage mode, which is the loop's.
                let others = row.resource_options.unwrap_or(0) & !0xf0;
                t.resource_options = Some(options | others);
                let ours = p.texture_descriptor(&t);
                *crate::answer::LAST_LINE.lock() = None;
                let answered = catch_death(|| d.heap_texture_size_and_align(&ours));
                let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
                let raw = ours.fields().descriptor(sel, Maker::SizeQuery);
                let modelled = d.metal_answer_at(sel, &raw);
                let runtimes = crate::answer::plain(|| d.raw.heap_texture_size_and_align(&raw));
                if let (Some(model), Ok(runtimes)) = (modelled, &runtimes)
                    && (runtimes.size < model.size || !runtimes.align.is_multiple_of(model.align))
                {
                    differ.push(format!("{case}: the runtime's {runtimes:x?} under"));
                }
                match answered {
                    Ok(ours) => {
                        let ours = ours.unwrap();
                        if !reims_vgpu_core::metal_matches!(case, ours) {
                            differ.push(format!("{case}: ours {ours:x?}, not Metal's"));
                        }
                    }
                    Err(site) => differ.push(format!("{case}: died at {site:?}: {line}")),
                }
            }
        }
    }
    assert!(differ.is_empty(), "{differ:#?}");
}

/// `newTextureWithDescriptor:` of a `Depth32Float_Stencil8` or `X32_Stencil8` texture with
/// allowGPUOptimizedContents off, whose image the runtime keeps GPU-optimized
/// (`optimized-depth-stencil-texture`): each is made, the suite's (a 2D one of two levels and a
/// cube array of two cubes of one level, 4x4 at `Private` storage, of usage `ShaderRead |
/// ShaderWrite`, and an `X32_Stencil8` one, 2D, of one level) among them, and its range of
/// Metal's space is of Metal's size as the runtime models it
/// ([`crate::gpu::metal_space::own_heap_size`]), Metal's answer for each on the M3 Max. A texture
/// the model answers none for dies at the selector, naming the kind. Metal makes each
#[test]
fn optimized_depth_textures_take_metals_modelled_size() {
    let sel = "newTextureWithDescriptor:";
    let sized = |side, usage, pixel_format, texture_type, levels, slices| TextureDesc {
        texture_type,
        pixel_format,
        width: side,
        height: side,
        mipmap_level_count: levels,
        array_length: slices,
        usage,
        resource_options: Some(0x20),
        allow_gpu_optimized_contents: Some(false),
        ..TextureDesc::default()
    };
    let shape = |pixel_format, texture_type, levels, slices| {
        sized(4, 0x3, pixel_format, texture_type, levels, slices)
    };
    let shapes = [
        ("2D, two levels", shape(260, 2, 2, 1)),
        ("a cube array, one level", shape(260, 6, 1, 2)),
        ("a 2D array", shape(260, 3, 1, 2)),
        ("a cube", shape(260, 5, 1, 1)),
        ("a cube array, two levels", shape(260, 6, 2, 2)),
        ("a cube array of three", shape(260, 6, 1, 3)),
        ("2D, two levels, 8x8", sized(8, 0x3, 260, 2, 2, 1)),
        (
            "2D, two levels, read and render",
            sized(4, 0x5, 260, 2, 2, 1),
        ),
        ("X32_Stencil8 2D, one level", shape(261, 2, 1, 1)),
        ("X32_Stencil8 2D, two levels", shape(261, 2, 2, 1)),
        ("an X32_Stencil8 2D array", shape(261, 3, 1, 2)),
        (
            "X32_Stencil8 2D, one level, 8x8",
            sized(8, 0x3, 261, 2, 1, 1),
        ),
        (
            "X32_Stencil8 2D, one level, read and render",
            sized(4, 0x5, 261, 2, 1, 1),
        ),
    ];
    let d = dev();
    for (name, t) in &shapes {
        let desc = platform().texture_descriptor(t);
        let made = d.new_texture(OnFault::Return, &desc);
        assert!(made.unwrap().is_some(), "{name}");
        let raw = desc.fields().descriptor(sel, Maker::Other);
        let modelled = d.metal_answer_at(sel, &raw);
        reims_vgpu_core::metal_check!(name, modelled);
    }
}

/// `heapTextureSizeAndAlignWithDescriptor:` of an `X32_Stencil8` texture with
/// allowGPUOptimizedContents off, whose image the runtime keeps GPU-optimized
/// (`optimized-depth-stencil-texture`), answers the runtime's model of Metal's answer, a
/// `Stencil8` texture of its shape, the size `newTextureWithDescriptor:` gives its range of
/// Metal's space: Metal's size and alignment for each shape on the M3 Max, at `Private` and
/// `Shared` storage, a 2D one of two levels among them
#[test]
fn x32_stencil8_size_queries_are_metals() {
    let sel = "heapTextureSizeAndAlignWithDescriptor:";
    let sized = |side, usage, texture_type, levels, slices, options| TextureDesc {
        texture_type,
        pixel_format: 261,
        width: side,
        height: side,
        mipmap_level_count: levels,
        array_length: slices,
        usage,
        resource_options: Some(options),
        allow_gpu_optimized_contents: Some(false),
        ..TextureDesc::default()
    };
    let mut shapes = Vec::new();
    for (storage, options) in [("Private", 0x20), ("Shared", 0)] {
        for (name, t) in [
            ("2D, one level", sized(4, 0x3, 2, 1, 1, options)),
            ("2D, two levels", sized(4, 0x3, 2, 2, 1, options)),
            ("a 2D array", sized(4, 0x3, 3, 1, 2, options)),
            ("a cube", sized(4, 0x3, 5, 1, 1, options)),
            ("2D, one level, 8x8", sized(8, 0x3, 2, 1, 1, options)),
            ("2D, two levels, 64x64", sized(64, 0x1, 2, 2, 1, options)),
            (
                "2D, one level, read and render",
                sized(4, 0x5, 2, 1, 1, options),
            ),
        ] {
            shapes.push((format!("{storage} {name}"), t));
        }
    }
    let d = dev();
    for (name, t) in &shapes {
        let desc = platform().texture_descriptor(t);
        let ours = d.heap_texture_size_and_align(&desc).unwrap();
        let raw = desc.fields().descriptor(sel, Maker::SizeQuery);
        assert_eq!(
            Some(ours),
            d.metal_answer_at(sel, &raw),
            "the model's: {name}"
        );
        reims_vgpu_core::metal_check!(name, ours);
        let made = d.new_texture(OnFault::Return, &desc);
        assert!(made.unwrap().is_some(), "made: {name}");
    }
}

/// A texture or view of `BGRA10_XR`, `BGRA10_XR_sRGB`, `BGR10_XR` or `BGR10_XR_sRGB` under a
/// swizzle other than the identity is made, as Metal makes it: a device's texture, a heap's
/// placement, and a view of an unswizzled texture by either swizzle selector. The runtime
/// decodes such a texture's codes before it swizzles them, a border texel included, so its
/// reads, samples and gathers are Metal's (`rgbx`'s
/// `extended_range_textures_read_as_metals` compares them)
#[test]
fn swizzled_extended_range_textures_are_made() {
    let (p, d) = (platform(), dev());
    // Alpha, red, one, zero.
    let swizzle = Swizzle {
        red: 5,
        green: 2,
        blue: 1,
        alpha: 0,
    };
    let desc = |pixel_format, swizzle| TextureDesc {
        pixel_format,
        width: 16,
        height: 16,
        usage: 1,
        resource_options: Some(0x20),
        swizzle,
        ..TextureDesc::default()
    };
    // A placement heap of 1 MiB of `Private` storage.
    fn heap_desc<'a, P: Platform>() -> HeapDesc<'a, P> {
        HeapDesc {
            size: 1 << 20,
            heap_type: 1,
            options: vec![OptionSet::StorageMode(2)],
            address_ranges: None,
            pinned_gpu_address: None,
        }
    }
    let whole = NsRange::new(0, 1);
    let formats = [
        (552, "BGRA10_XR"),
        (553, "BGRA10_XR_sRGB"),
        (554, "BGR10_XR"),
        (555, "BGR10_XR_sRGB"),
    ];
    for (format, name) in formats {
        let ours = |s| p.texture_descriptor(&desc(format, s));
        let made = d.new_texture(OnFault::Return, &ours(Some(swizzle)));
        assert!(made.unwrap().is_some(), "{name}: texture");
        let heap = d.new_heap(&heap_desc()).unwrap().heap.unwrap();
        let placed = heap.new_texture(&ours(Some(swizzle)), 0).unwrap();
        assert!(placed.is_some(), "{name}: placement");
        let plain = d
            .new_texture(OnFault::Return, &ours(None))
            .unwrap()
            .unwrap();
        let view = plain.new_swizzled_texture_subview(format, 2, whole, whole, swizzle);
        assert!(view.unwrap().is_some(), "{name}: view");
        let view =
            plain.new_swizzled_texture_subview_at_index(format, 2, whole, whole, swizzle, 0x800);
        assert!(view.unwrap().is_some(), "{name}: view at an index");
    }
}

/// A `BGRA` surface of 64 by 16, rows 256 bytes apart, of `kIOSurfaceClientDirection` 2 over the
/// guest's page `guest` of `page` bytes, whose GPU writes never reach its memory.
fn read_only_surface(guest: reims_vgpu_core::host::HostPtr, page: u64) -> OsSurface {
    use reims_vgpu_core::host::PhysicalMemoryRange;
    use reims_vgpu_core::num::ToU64 as _;
    use reims_vgpu_core::platform::os::surface::IoSurfaces as _;
    let spec = SurfaceSpec {
        pixel_format: u32::from_be_bytes(*b"BGRA"),
        alloc_size: page,
        width: 64,
        height: 16,
        bytes_per_element: 4,
        bytes_per_row: 256,
        element_width: 1,
        element_height: 1,
        client_direction: Some(2),
        client_address_ranges: Some(vec![PhysicalMemoryRange {
            physical_address: guest.addr().to_u64(),
            physical_length: page,
        }]),
        ..SurfaceSpec::default()
    };
    let surface = crate::os::OsSurfaces::new().create(&spec).unwrap();
    assert!(crate::os::gpu_read_only(&surface));
    surface
}

/// A swizzle other than the identity over memory whose GPU writes never reach it (a surface
/// of `kIOSurfaceClientDirection` 2 over a guest's page, and a buffer made on it) is refused
/// under `ShaderWrite` or `RenderTarget` usage as over any other memory, Metal's answers on
/// the M3 Max under macOS 26.7, one case a process: without `writeSwizzleEnabled` Metal's
/// validation asserts ("Texture Descriptor Validation"), where the runtime's validation stops
/// it (`SwizzleWithWrites`); with it Metal makes the texture, of the usage asked, and writes
/// through the swizzle, which the runtime does not build, so this platform dies naming the
/// gap (`RuntimeGap::WriteSwizzle`). Read alone (`ShaderRead`, and with `PixelFormatView`),
/// both make it, answering the usage asked.
#[test]
fn written_swizzles_over_read_only_memory_are_refused() {
    use crate::os::OsSurface;
    use reims_vgpu_core::platform::os::vm::HostVm as _;
    use reims_vgpu_core::platform::types::Swizzle;
    let vm = crate::os::OsVm::new();
    let page = vm.page_size();
    let guest = vm.allocate_anywhere(page).unwrap();
    let surface = read_only_surface(guest, page);
    let desc = |usage, wse: bool| TextureDesc {
        pixel_format: 80,
        width: 64,
        height: 16,
        usage,
        resource_options: Some(0),
        swizzle: Some(Swizzle {
            red: 4,
            green: 3,
            blue: 2,
            alpha: 5,
        }),
        write_swizzle_enabled: wse.then_some(true),
        ..TextureDesc::default()
    };
    // Each creator's texture's usage, or `None` for its death, of `desc` on `d`: the
    // IOSurface creator's and a buffer's on the surface.
    fn made<P: Platform<Surface = OsSurface>>(
        d: &P::Device,
        surface: &OsSurface,
        desc: &P::TextureDescriptor,
    ) -> [u64; 2] {
        let buffer = d.new_buffer_on_surface(Some(surface)).unwrap().unwrap();
        let on_surface = d.new_texture_on_surface(desc, Some(surface), 0);
        let on_buffer = buffer.new_texture(OnFault::Return, desc, 0, 256);
        [on_surface, on_buffer].map(|t| t.unwrap().unwrap().usage())
    }
    let d = dev();
    let p = platform();
    for usage in [0x1, 0x11] {
        let ours = made::<VulkanPlatform>(&d, &surface, &p.texture_descriptor(&desc(usage, false)));
        assert_eq!(ours, [usage; 2], "usage {usage:#x}");
    }
    let buffer = d.new_buffer_on_surface(Some(&surface)).unwrap().unwrap();
    for usage in [0x3, 0x5] {
        for wse in [false, true] {
            let ours = p.texture_descriptor(&desc(usage, wse));
            let creators: [(&str, &dyn Fn()); 2] = [
                ("newTextureWithDescriptor:iosurface:plane:", &|| {
                    let _ = d.new_texture_on_surface(&ours, Some(&surface), 0);
                }),
                ("newTextureWithDescriptor:offset:bytesPerRow:", &|| {
                    let _ = buffer.new_texture(OnFault::Return, &ours, 0, 256);
                }),
            ];
            for (sel, make) in creators {
                *crate::answer::LAST_LINE.lock() = None;
                let site = catch_death(make).unwrap_err();
                let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
                assert_eq!((site.kind, site.file), (DeathKind::Abort, sel), "{line}");
                let refused = match wse {
                    false => "Texture(SwizzleWithWrites)",
                    true => "not implemented in the runtime yet: WriteSwizzle",
                };
                assert!(line.contains(refused), "{usage:#x} {wse} {sel}: {line}");
            }
        }
    }
    drop((buffer, surface));
    assert!(vm.deallocate(guest, page).is_success());
}

/// A texture over memory whose GPU writes never reach it (a surface of
/// `kIOSurfaceClientDirection` 2 over a guest's page, and a buffer made on it), whose usage this
/// platform answers itself ([`read_only_usage`]), answers the usage bits Metal's public headers
/// name none of as any other texture does (`usage_bits_metal_takes_are_answered_as_metals`): all
/// but 0x8, which Metal's texture drops; and so do its views and its swizzled views, which a
/// texture of `RenderTarget | 0x8` has too, Metal's answers on the M3 Max
#[test]
fn usage_bits_over_read_only_surfaces_are_answered_as_metals() {
    use reims_vgpu_core::platform::os::vm::HostVm as _;
    let vm = crate::os::OsVm::new();
    let page = vm.page_size();
    let guest = vm.allocate_anywhere(page).unwrap();
    let surface = read_only_surface(guest, page);
    let desc = |usage| TextureDesc {
        pixel_format: 80,
        width: 64,
        height: 16,
        usage,
        resource_options: Some(0),
        ..TextureDesc::default()
    };
    // `MTLTextureSwizzleBlue`, `Green`, `Red`, `Alpha`.
    let swapped = Swizzle {
        red: 4,
        green: 3,
        blue: 2,
        alpha: 5,
    };
    // The usage of each creator's texture of `desc` on `d` (the IOSurface creator's and a
    // buffer's on the surface), of its view, and of its view of `swizzle`.
    fn on<P: Platform<Surface = OsSurface>>(
        d: &P::Device,
        surface: &OsSurface,
        desc: &P::TextureDescriptor,
        swizzle: Swizzle,
    ) -> [(u64, u64, u64); 2] {
        let buffer = d.new_buffer_on_surface(Some(surface)).unwrap().unwrap();
        let textures = [
            d.new_texture_on_surface(desc, Some(surface), 0),
            buffer.new_texture(OnFault::Return, desc, 0, 256),
        ];
        let one = NsRange::new(0, 1);
        textures.map(|t| {
            let t = t.unwrap().unwrap();
            let view = t.new_texture_view(80).unwrap().unwrap().usage();
            let swizzled = (t.new_swizzled_texture_subview(80, 2, one, one, swizzle))
                .unwrap()
                .unwrap()
                .usage();
            (t.usage(), view, swizzled)
        })
    }
    let d = dev();
    let p = platform();
    for usage in [0x9, 0xc, 0xd, 0x1_0005, 0x1_c01d] {
        let ours = on::<VulkanPlatform>(&d, &surface, &p.texture_descriptor(&desc(usage)), swapped);
        let answered = usage & !0x8;
        let swizzled = answered & 0x1_4011;
        assert_eq!(
            ours,
            [(answered, answered, swizzled); 2],
            "usage {usage:#x}"
        );
    }
    drop(surface);
    assert!(vm.deallocate(guest, page).is_success());
}

/// A texture of `width` by `height` in `format`, of `ShaderRead` usage alone, over a Shared
/// buffer holding `bytes`, as a guest backing's texture is.
fn read_only_over_buffer(format: u64, width: u64, height: u64, bytes: &[u8]) -> VulkanTexture {
    let b = shared(width * height * 4);
    // SAFETY: the buffer's Shared bytes, which nothing else touches yet.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), b.contents().as_ptr(), bytes.len()) };
    let desc = platform().texture_descriptor(&TextureDesc {
        pixel_format: format,
        width,
        height,
        usage: 0x1,
        ..TextureDesc::default()
    });
    b.new_texture(OnFault::Return, &desc, 0, width * 4)
        .unwrap()
        .unwrap()
}

/// A texture without PixelFormatView usage views as another format of its class, as Metal
/// makes it without API validation: an RGBA8Unorm texture over a buffer as BGRA8Unorm and
/// R32Uint, reading the same bytes; it and its views report the usage asked, Metal's
/// answers on this host
#[test]
fn views_need_no_pixel_format_view_usage() {
    let bytes = pattern(8, 4);
    let t = read_only_over_buffer(70, 8, 4, &bytes);
    let mut answers = Vec::new();
    for format in [80, 53] {
        let v = t.new_texture_view(format).unwrap().unwrap();
        assert_eq!(read_back(&v), bytes, "format {format}");
        answers.push((v.pixel_format(), v.usage()));
    }
    assert_eq!(t.usage(), 1);
    assert_eq!(answers, [(80, 1), (53, 1)]);
    let asked = d_texture(0x11);
    assert_eq!(asked.usage(), 0x11);
    assert_eq!(asked.new_texture_view(80).unwrap().unwrap().usage(), 0x11);
    assert_eq!(d_texture(0).usage(), 0x17, "Unknown, as Metal reports it");
    let handle = t.new_shared_texture_handle().unwrap();
    assert!(handle.is_none(), "a texture over a buffer is not shareable");
}

/// How a command buffer reaches a surface ([`surface_references`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Reach {
    /// A blit's copy of a texture on it into a buffer.
    Blit,
    /// A kernel's `read()` of a texture on it, bound at a slot.
    Kernel,
    /// A kernel's `read()` of a view of a texture on it, bound at a slot.
    View,
    /// A kernel's `read()` of a texture on it through an argument buffer holding its
    /// resource ID, made resident with `useResource:usage:`.
    Resident,
    /// A render pass that clears a texture on it and stores it.
    Attachment,
    /// A blit's copy of a buffer on it (`newBufferWithIOSurface:`) into another buffer.
    SurfaceBuffer,
    /// Not at all: a texture on it is made and released, and the command buffer's kernel
    /// reads a texture on another surface.
    Unused,
    /// A kernel's `read()` of a texture of the format (raw) over plane 0 of it, a two-plane
    /// surface of the OSType, bound at a slot.
    KernelYcbcr(u64, [u8; 4]),
    /// Not at all, as for `Unused`, the texture on it one of the format (raw) over plane 0
    /// of it, a two-plane surface of the OSType.
    UnusedYcbcr(u64, [u8; 4]),
    /// A draw's `sample()` of a texture on it, bound at a fragment slot, into a target on
    /// another surface.
    Sampled(Pipeline),
    /// A draw into a texture on it, a render pass's color attachment it clears and stores.
    Drawn(Pipeline),
}

/// The `VkPipeline` a draw of [`Reach::Sampled`] or [`Reach::Drawn`] takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Pipeline {
    /// Made before the command buffer: the state drew the same in one before.
    Before,
    /// Made at the draw: the state's first draw, through a triangle fill mode of lines, a
    /// key its creation does not predict. The test draws on a device whose shader cache is
    /// its own and empty (`device_with_own_cache`), as a fresh machine's.
    AtTheDraw,
}

#[link(name = "IOSurface", kind = "framework")]
unsafe extern "C" {
    fn IOSurfaceGetUseCount(buffer: *const std::ffi::c_void) -> i32;
}

/// `s`'s IOSurface's retain count (`CFGetRetainCount`, which core's unmap reads: Busy unless
/// 1) and its use count (`IOSurfaceGetUseCount`).
fn surface_counts(s: &OsSurface) -> (i64, i32) {
    let at =
        std::ptr::with_exposed_provenance::<std::ffi::c_void>(usize::try_from(s.id().0).unwrap());
    // SAFETY: a live IOSurfaceRef, which `s` holds.
    (s.retain_count(), unsafe { IOSurfaceGetUseCount(at) })
}

/// Waits until `q`'s command buffers that completed have let go of what they retained, as
/// Metal lets go of it: on the queue's completion queue, after the command buffer's waiters
/// returned (`-[IOGPUMetalCommandBuffer didCompleteWithStartTime:endTime:error:]` releases its
/// resource list once `waitUntilCompleted` is woken), so a read right after the wait may find it
/// still held. A queue's completions run in order there, each letting go before the next one's
/// waiters return, so the wait is an empty command buffer of `q`, committed and waited for.
/// Measured on the M3 Max (macOS 27.0.1; 8 processes at once beside 16 busy CPU threads, 300
/// render passes each storing a texture on a surface): the surface still held right after the
/// wait in 30 of 2400, the texture let go of after the wait in 217; after the empty command
/// buffer's wait, held in none and let go of after it in none (after one of another queue's, 6).
/// This platform lets go before the wait returns, as the runtime opens its waiters' gate only
/// then.
fn let_go<P: Platform>(q: &P::CommandQueue) {
    use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
}

/// [`surface_counts`] of a 16 by 16 surface (`BGRA8Unorm`, or two planes for the YCbCr
/// reaches) as a command buffer of `d`'s reaches it as `reach` says: with the texture or
/// buffer on it made; once the command buffer committed waiting for an event, every other
/// handle to the texture or buffer released; after it completed and its queue let go of what
/// it retained ([`let_go`]), the command buffer still held; and once the command buffer is
/// released. Each step runs in an autorelease pool of its own, popped before the counts are
/// read. With them, how many pipelines `made` counts were made from the command buffer's
/// encoding until it completed.
fn surface_references<P: Platform<Surface = OsSurface>>(
    p: &P,
    d: &P::Device,
    reach: Reach,
    made: &dyn Fn(&P::Device) -> u64,
) -> ([(i64, i32); 4], u64) {
    use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder as _, ResourceRef};
    use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
    use reims_vgpu_core::platform::compute::{ComputeCmd, ComputeEncoder as _};
    use reims_vgpu_core::platform::descriptor::RenderPassDescriptor as _;
    use reims_vgpu_core::platform::event::{EventRef, SharedEvent as _};
    use reims_vgpu_core::platform::render::{RenderCmd, RenderEncoder as _};
    use reims_vgpu_core::platform::resource::{Buffer as _, Library as _, Texture as _};
    use reims_vgpu_core::platform::values::{ColorAttachmentSet, RenderPipelineDesc};
    use reims_vgpu_core::rt::Pool;

    use crate::gpu::ycbcr::measure::{PROBE, sampler};
    use crate::gpu::ycbcr::measure::{Shape, desc as ycbcr_desc};
    let side = 16;
    let ycbcr = match reach {
        Reach::KernelYcbcr(format, ostype) | Reach::UnusedYcbcr(format, ostype) => {
            Some((format, ostype))
        }
        _ => None,
    };
    let s = match ycbcr {
        Some((_, ostype)) => Shape::of(&ostype).surface((side, side)),
        None => bgra_surface(side, side),
    };
    let other = bgra_surface(side, side);
    let desc = p.texture_descriptor(&TextureDesc {
        pixel_format: 80,
        width: side,
        height: side,
        usage: 5,
        resource_options: Some(0),
        ..TextureDesc::default()
    });
    let on_s = match ycbcr {
        Some((format, _)) => p.texture_descriptor(&ycbcr_desc(format, (side, side))),
        None => desc.clone_ref(),
    };
    let q = d.new_command_queue().unwrap();
    let event = d.new_shared_event().unwrap();
    let out = d
        .new_buffer_of_length(OnFault::Die, side * side * 8, 0)
        .unwrap()
        .unwrap();
    let args = d
        .new_buffer_of_length(OnFault::Die, 64, 0)
        .unwrap()
        .unwrap();
    let library = d.new_library(PROBE).unwrap().unwrap().unwrap();
    let state = |name| {
        let f = library.new_function(OnFault::Die, name).unwrap().unwrap();
        d.new_compute_pipeline_state_for_function(Some(&f))
            .unwrap()
            .unwrap()
    };
    let (read, held) = (state("read_texels"), state("read_held"));
    let elsewhere = d
        .new_texture_on_surface(&desc, Some(&other), 0)
        .unwrap()
        .unwrap();
    // A draw's state, sampler and probe (`sample_fragment`'s: each fragment's texel centre,
    // in rows of `side`).
    let render = |name| {
        let function = |name| library.new_function(OnFault::Die, name).unwrap().unwrap();
        let (vertex, fragment) = (function("cover_vertex"), function(name));
        d.new_render_pipeline_state(&RenderPipelineDesc {
            label: None,
            vertex: &vertex,
            fragment: &fragment,
            color0_pixel_format: 80,
            resource_index: None,
        })
        .unwrap()
        .unwrap()
    };
    let drawing = match reach {
        Reach::Sampled(_) => Some(render("sample_fragment")),
        Reach::Drawn(_) => Some(render("gradient_fragment")),
        _ => None,
    };
    let nearest = sampler::<P>(d, (0, 0, false), 0);
    if let Reach::Sampled(_) = reach {
        let probe: [u32; 6] = [
            0.5f32.to_bits(),
            0.5f32.to_bits(),
            1f32.to_bits(),
            1f32.to_bits(),
            u32::try_from(side).unwrap(),
            u32::try_from(side).unwrap(),
        ];
        // SAFETY: a Shared buffer of 64 bytes, which no command uses yet.
        unsafe {
            std::ptr::copy_nonoverlapping(probe.as_ptr(), args.contents().as_ptr::<u32>(), 6)
        };
    }
    let draw = |cb: &P::CommandBuffer, target: &P::Texture, sampled: Option<&P::Texture>, lines| {
        let pass = p.render_pass_descriptor();
        let attachment = ColorAttachmentSet {
            texture: Some(Some(target)),
            load_action: Some(2),
            store_action: Some(1),
            ..ColorAttachmentSet::default()
        };
        pass.set_color_attachment(OnFault::Die, 0, &attachment)
            .unwrap();
        let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
        let buffers = [Some(&out), Some(&args)];
        let samplers = [Some(&nearest)];
        let mut cmds = vec![RenderCmd::SetRenderPipelineState {
            pipeline_state: drawing.as_ref(),
        }];
        if lines {
            cmds.push(RenderCmd::SetTriangleFillMode { fill_mode: 1 });
        }
        if let Some(t) = sampled {
            cmds.extend([
                RenderCmd::SetFragmentTexture {
                    texture: Some(t),
                    index: 0,
                },
                RenderCmd::SetFragmentSamplerStates {
                    samplers: &samplers,
                    range: NsRange::new(0, 1),
                },
                RenderCmd::SetFragmentBuffers {
                    buffers: &buffers,
                    offsets: &[0, 0],
                    range: NsRange::new(0, 2),
                },
            ]);
        }
        cmds.push(RenderCmd::DrawPrimitives {
            primitive_type: 3,
            vertex_start: 0,
            vertex_count: 3,
        });
        cmds.push(RenderCmd::EndEncoding);
        for cmd in cmds {
            e.encode(OnFault::Die, cmd).unwrap();
        }
    };
    // The same draw before, into and of textures on no surface, completed and released.
    if let Reach::Sampled(Pipeline::Before) | Reach::Drawn(Pipeline::Before) = reach {
        let _pool = Pool::here();
        let plain = || d.new_texture(OnFault::Die, &desc).unwrap().unwrap();
        let (target, sampled) = (plain(), plain());
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let sampled = matches!(reach, Reach::Sampled(_)).then_some(&sampled);
        draw(&cb, &target, sampled, false);
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        assert!(cb.error().is_none(), "{reach:?}: {:?}", cb.error());
    }
    let made_before = made(d);
    let mut seen = [(0, 0); 4];
    let cb = {
        let _pool = Pool::here();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        cb.encode_wait(Some(EventRef::Shared(&event)), 1).unwrap();
        let (t, b) = match reach {
            Reach::SurfaceBuffer => (None, d.new_buffer_on_surface(Some(&s)).unwrap()),
            _ => (d.new_texture_on_surface(&on_s, Some(&s), 0).unwrap(), None),
        };
        assert!(
            t.is_some() || b.is_some(),
            "{reach:?}: nothing made on the surface"
        );
        seen[0] = surface_counts(&s);
        let view = match reach {
            Reach::View => t.as_ref().unwrap().new_texture_view(80).unwrap(),
            _ => None,
        };
        let kernel = |e: &P::ComputeEncoder, state, texture, resident: Option<&P::Texture>| {
            let buffers = [Some(&out), Some(&args)];
            let resources = [resident.map(ResourceRef::Texture)];
            let mut cmds = vec![
                ComputeCmd::SetComputePipelineState { state: Some(state) },
                ComputeCmd::SetBuffers {
                    buffers: &buffers,
                    offsets: &[0, 0],
                    range: NsRange::new(0, 2),
                },
            ];
            if let Some(texture) = texture {
                cmds.push(ComputeCmd::SetTexture {
                    texture: Some(texture),
                    index: 0,
                });
            }
            if resident.is_some() {
                cmds.push(ComputeCmd::UseResources {
                    resources: &resources,
                    usage: 1,
                });
            }
            cmds.push(ComputeCmd::DispatchThreads {
                threads_per_grid: MtlSize::new(side, side, 1),
                threads_per_threadgroup: MtlSize::new(8, 8, 1),
            });
            cmds.push(ComputeCmd::EndEncoding);
            for cmd in cmds {
                e.encode(OnFault::Die, cmd).unwrap();
            }
        };
        match reach {
            Reach::Blit | Reach::SurfaceBuffer => {
                let e = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
                let copy = match &t {
                    Some(t) => BlitCmd::CopyTextureToBuffer {
                        source_texture: t,
                        source_slice: 0,
                        source_level: 0,
                        source_origin: MtlOrigin::new(0, 0, 0),
                        source_size: MtlSize::new(side, side, 1),
                        destination_buffer: &out,
                        destination_offset: 0,
                        destination_bytes_per_row: side * 4,
                        destination_bytes_per_image: side * side * 4,
                    },
                    None => BlitCmd::CopyBufferToBuffer {
                        source_buffer: b.as_ref(),
                        source_offset: 0,
                        destination_buffer: Some(&out),
                        destination_offset: 0,
                        size: side * side * 4,
                    },
                };
                e.encode(OnFault::Die, copy).unwrap();
                e.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
            }
            Reach::Attachment => {
                let pass = p.render_pass_descriptor();
                let attachment = ColorAttachmentSet {
                    texture: Some(t.as_ref()),
                    load_action: Some(2),
                    store_action: Some(1),
                    ..ColorAttachmentSet::default()
                };
                pass.set_color_attachment(OnFault::Die, 0, &attachment)
                    .unwrap();
                let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
                e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
            }
            Reach::Resident => {
                let e = cb.compute_encoder(OnFault::Die).unwrap().unwrap();
                let t = t.as_ref().unwrap();
                // SAFETY: a Shared buffer of 64 bytes, which no command uses yet.
                unsafe { args.contents().as_ptr::<u64>().write(t.gpu_resource_id()) };
                kernel(&e, &held, None, Some(t));
            }
            Reach::Kernel | Reach::KernelYcbcr(..) => {
                let e = cb.compute_encoder(OnFault::Die).unwrap().unwrap();
                kernel(&e, &read, t.as_ref(), None);
            }
            Reach::View => {
                let e = cb.compute_encoder(OnFault::Die).unwrap().unwrap();
                kernel(&e, &read, view.as_ref(), None);
            }
            Reach::Unused | Reach::UnusedYcbcr(..) => {
                let e = cb.compute_encoder(OnFault::Die).unwrap().unwrap();
                kernel(&e, &read, Some(&elsewhere), None);
            }
            Reach::Sampled(drawn) => {
                draw(&cb, &elsewhere, t.as_ref(), drawn == Pipeline::AtTheDraw);
            }
            Reach::Drawn(drawn) => {
                draw(&cb, t.as_ref().unwrap(), None, drawn == Pipeline::AtTheDraw);
            }
        }
        cb.commit(OnFault::Die).unwrap();
        cb
    };
    seen[1] = surface_counts(&s);
    {
        let _pool = Pool::here();
        event.set_signaled_value(1).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        assert!(cb.error().is_none(), "{reach:?}: {:?}", cb.error());
        let_go::<P>(&q);
    }
    seen[2] = surface_counts(&s);
    let made = made(d) - made_before;
    {
        let _pool = Pool::here();
        drop(cb);
    }
    seen[3] = surface_counts(&s);
    (seen, made)
}

/// A command buffer holds a surface's IOSurface as Metal's does, measured by its retain
/// count (which core's unmap reads: Busy unless 1) and its use count: from its commit until
/// it completed, through a texture on the surface that a blit copies, a kernel reads at a
/// slot or through an argument buffer made resident, a render pass stores, or a draw samples
/// at a fragment slot or draws into as its attachment (its `VkPipeline` made before the
/// command buffer, or at the draw), a buffer on it a blit copies, and a texture of
/// `YCBCR10_420_2P` (505) or `RGB10_420_2P` (543) over an 'x420' surface, or of
/// `RGB8_420_2P` (540) over a '420v' one, a kernel reads; not at all
/// through any of those textures, made and released, that it never names, whose retain
/// is back to 1 at once. A kernel's read of a view of the texture holds the IOSurface too;
/// Metal's view takes a reference of its own, 3 while the command buffer runs where this
/// platform's view shares its texture's, 2. Either way unmap answers Busy until the command
/// buffer completed. Neither platform's command buffer changes the use count
///
/// On this platform the kernel's read through the argument buffer made resident dies at the
/// dispatch naming `half-texel-conversion` ([`crate::answer::DIES`]): the runtime announces the
/// kind for a `half` read of an argument buffer's texture, whose format the function does not
/// know and whose texel it converts by a rounding Metal's may not be. That case runs last here.
///
/// Metal lets go of what a completed command buffer retained after its waiters returned, the
/// more often once MoltenVK has worked on Metal's device
/// (`texture_identifiers_are_reused_as_metals`): the counts after the completion are read once
/// its queue let go of it ([`let_go`])
#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "MoltenVK: the runtime records device-denormals, a death in strict mode; Mac-only test, off everywhere; ledger: MoltenVK strict deaths"
)]
fn a_command_buffer_holds_a_surface_as_metals_does() {
    let ycbcr = [(505, *b"x420"), (543, *b"x420"), (540, *b"420v")];
    let named = [
        Reach::Blit,
        Reach::Kernel,
        Reach::Resident,
        Reach::Attachment,
        Reach::SurfaceBuffer,
        Reach::Sampled(Pipeline::Before),
        Reach::Sampled(Pipeline::AtTheDraw),
        Reach::Drawn(Pipeline::Before),
        Reach::Drawn(Pipeline::AtTheDraw),
    ];
    let named = named
        .into_iter()
        .chain(ycbcr.map(|(f, o)| Reach::KernelYcbcr(f, o)));
    let unnamed = [Reach::Unused]
        .into_iter()
        .chain(ycbcr.map(|(f, o)| Reach::UnusedYcbcr(f, o)));
    let cases: Vec<_> = named
        .map(|reach| (reach, [(2, 0), (2, 0), (1, 0), (1, 0)]))
        .chain(unnamed.map(|reach| (reach, [(2, 0), (1, 0), (1, 0), (1, 0)])))
        .chain([(Reach::View, [(2, 0), (2, 0), (1, 0), (1, 0)])])
        .collect();
    let (p, d) = (platform(), dev());
    let made = |d: &VulkanDevice| d.raw.work_counts().variant_pipelines_made;
    // The draws' device: a shader cache of its own, empty, as a fresh machine's.
    let (own, _cache) = crate::gpu::device::tests::metal_tests::device_with_own_cache();
    for &(reach, want) in &cases {
        if reach == Reach::Resident {
            continue;
        }
        let on = match reach {
            Reach::Sampled(_) | Reach::Drawn(_) => &own,
            _ => &d,
        };
        let (ours, made) =
            surface_references::<crate::platform::VulkanPlatform>(&p, on, reach, &made);
        assert_eq!(ours, want, "{reach:?}");
        match reach {
            Reach::Sampled(Pipeline::AtTheDraw) | Reach::Drawn(Pipeline::AtTheDraw) => {
                assert_eq!(made, 1, "{reach:?}: its pipeline made at the draw");
            }
            Reach::Sampled(_) | Reach::Drawn(_) => {
                assert_eq!(made, 0, "{reach:?}: its pipeline made before");
            }
            _ => {}
        }
    }
    let kind = metal2vulkan_runtime::Degradation::HalfTexelConversion;
    let site = crate::answer::dies_of(kind, || {
        surface_references::<crate::platform::VulkanPlatform>(&p, &d, Reach::Resident, &made)
    });
    assert_eq!(site.file, "dispatchThreads:threadsPerThreadgroup:");
    // The device before the directory of its shader cache, which it uses while it lives: a
    // pattern's bindings drop last first.
    drop(own);
}

/// What holds a surface beside a command buffer's own uses of it ([`surface_holds`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Beside {
    /// Another queue's command buffer, committed after a kernel read a texture on it and
    /// before the texture's release, which reads a texture on another surface: counts once
    /// the texture is released, the other command buffer waiting for an event; once it
    /// completed; once it is released.
    Pending,
    /// As `Pending`, the other command buffer's kernel reading its texture through an
    /// argument buffer holding its resource ID, made resident with `useResource:usage:`.
    PendingArgument,
    /// As `Pending`, the other command buffer's kernel reaching a buffer of the device's
    /// own through an address it loads (`reach.metal`), naming no texture.
    PendingReach,
    /// A completed handler of the command buffer whose kernel read a texture on it, the
    /// texture released before the commit: counts inside the handler as it begins and
    /// 100 ms later; once the wait for the completion returned; 200 ms later; once the
    /// command buffer is released.
    Handler,
    /// Two textures on it, the first read by a kernel, the second never used: counts once
    /// the command buffer completed and is released; once the first is released; once
    /// the second is.
    Two,
}

/// [`surface_counts`] of a 16 by 16 `BGRA8Unorm` surface at the steps `beside` names, each
/// step in an autorelease pool of its own. A wait for a command buffer's completion waits until
/// its queue let go of what it retained too ([`let_go`]).
fn surface_holds<P: Platform<Surface = OsSurface>>(
    p: &P,
    d: &P::Device,
    beside: Beside,
) -> Vec<(i64, i32)> {
    use reims_vgpu_core::platform::blit::ResourceRef;
    use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
    use reims_vgpu_core::platform::compute::{ComputeCmd, ComputeEncoder as _};
    use reims_vgpu_core::platform::event::{EventRef, SharedEvent as _};
    use reims_vgpu_core::platform::resource::{Buffer as _, Library as _, Texture as _};
    use reims_vgpu_core::rt::Pool;

    use crate::gpu::ycbcr::measure::PROBE;
    let side = 16;
    let s = Arc::new(bgra_surface(side, side));
    let other = bgra_surface(side, side);
    let desc = p.texture_descriptor(&TextureDesc {
        pixel_format: 80,
        width: side,
        height: side,
        usage: 5,
        resource_options: Some(0),
        ..TextureDesc::default()
    });
    let (q, q2) = (
        d.new_command_queue().unwrap(),
        d.new_command_queue().unwrap(),
    );
    let event = d.new_shared_event().unwrap();
    let out = d
        .new_buffer_of_length(OnFault::Die, side * side * 8, 0)
        .unwrap()
        .unwrap();
    let library = d.new_library(PROBE).unwrap().unwrap().unwrap();
    let state = |name| {
        let f = library.new_function(OnFault::Die, name).unwrap().unwrap();
        d.new_compute_pipeline_state_for_function(Some(&f))
            .unwrap()
            .unwrap()
    };
    let (read, held) = (state("read_texels"), state("read_held"));
    let args = d
        .new_buffer_of_length(OnFault::Die, 64, 0)
        .unwrap()
        .unwrap();
    let elsewhere = d
        .new_texture_on_surface(&desc, Some(&other), 0)
        .unwrap()
        .unwrap();
    let reading_how = |q: &P::CommandQueue, texture: &P::Texture, wait: bool, argument: bool| {
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        if wait {
            cb.encode_wait(Some(EventRef::Shared(&event)), 1).unwrap();
        }
        let e = cb.compute_encoder(OnFault::Die).unwrap().unwrap();
        let buffers = [Some(&out), Some(&args)];
        let resources = [Some(ResourceRef::Texture(texture))];
        let mut cmds = vec![
            ComputeCmd::SetComputePipelineState {
                state: Some(match argument {
                    true => &held,
                    false => &read,
                }),
            },
            ComputeCmd::SetBuffers {
                buffers: &buffers,
                offsets: &[0, 0],
                range: NsRange::new(0, 2),
            },
        ];
        if argument {
            // SAFETY: a Shared buffer of 64 bytes, which no command uses yet.
            unsafe {
                args.contents()
                    .as_ptr::<u64>()
                    .write(texture.gpu_resource_id())
            };
            cmds.push(ComputeCmd::UseResources {
                resources: &resources,
                usage: 1,
            });
        } else {
            cmds.push(ComputeCmd::SetTexture {
                texture: Some(texture),
                index: 0,
            });
        }
        for cmd in cmds.into_iter().chain([
            ComputeCmd::DispatchThreads {
                threads_per_grid: MtlSize::new(side, side, 1),
                threads_per_threadgroup: MtlSize::new(8, 8, 1),
            },
            ComputeCmd::EndEncoding,
        ]) {
            e.encode(OnFault::Die, cmd).unwrap();
        }
        cb
    };
    let reading = |q, texture, wait| reading_how(q, texture, wait, false);
    // The command buffer, and the buffer its kernel reaches through the address it loads,
    // which no command names, so the caller holds it until the command buffer completed.
    let reaching = |q: &P::CommandQueue| {
        const REACH: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/reach.metallib"));
        let library = d.new_library(REACH).unwrap().unwrap().unwrap();
        let f = library
            .new_function(OnFault::Die, "reach")
            .unwrap()
            .unwrap();
        let state = d
            .new_compute_pipeline_state_for_function(Some(&f))
            .unwrap()
            .unwrap();
        let page = || {
            d.new_buffer_of_length(OnFault::Die, 4096, 0)
                .unwrap()
                .unwrap()
        };
        let (addresses, words, reached) = (page(), page(), page());
        // SAFETY: a Shared buffer of a page, which no command uses yet.
        unsafe {
            addresses
                .contents()
                .as_ptr::<u64>()
                .write(reached.gpu_address())
        };
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        cb.encode_wait(Some(EventRef::Shared(&event)), 1).unwrap();
        let e = cb.compute_encoder(OnFault::Die).unwrap().unwrap();
        for cmd in [
            ComputeCmd::SetComputePipelineState {
                state: Some(&state),
            },
            ComputeCmd::SetBuffers {
                buffers: &[Some(&addresses), Some(&words)],
                offsets: &[0, 0],
                range: NsRange::new(0, 2),
            },
            ComputeCmd::DispatchThreadgroups {
                threadgroups_per_grid: MtlSize::new(1, 1, 1),
                threads_per_threadgroup: MtlSize::new(1, 1, 1),
            },
            ComputeCmd::EndEncoding,
        ] {
            e.encode(OnFault::Die, cmd).unwrap();
        }
        (cb, reached)
    };
    let done = |q: &P::CommandQueue, cb: &P::CommandBuffer| {
        cb.wait_until_completed(OnFault::Die).unwrap();
        assert!(cb.error().is_none(), "{beside:?}: {:?}", cb.error());
        let_go::<P>(q);
    };
    let on_s = || {
        d.new_texture_on_surface(&desc, Some(&s), 0)
            .unwrap()
            .unwrap()
    };
    let mut seen = Vec::new();
    match beside {
        Beside::Pending | Beside::PendingArgument | Beside::PendingReach => {
            let t = {
                let _pool = Pool::here();
                on_s()
            };
            {
                let _pool = Pool::here();
                let cb = reading(&q, &t, false);
                cb.commit(OnFault::Die).unwrap();
                done(&q, &cb);
            }
            let (cb, reached) = {
                let _pool = Pool::here();
                let argument = beside == Beside::PendingArgument;
                let (cb, reached) = match beside {
                    Beside::PendingReach => {
                        let (cb, reached) = reaching(&q2);
                        (cb, Some(reached))
                    }
                    _ => (reading_how(&q2, &elsewhere, true, argument), None),
                };
                cb.commit(OnFault::Die).unwrap();
                (cb, reached)
            };
            {
                let _pool = Pool::here();
                drop(t);
            }
            seen.push(surface_counts(&s));
            {
                let _pool = Pool::here();
                event.set_signaled_value(1).unwrap();
                done(&q2, &cb);
            }
            seen.push(surface_counts(&s));
            {
                let _pool = Pool::here();
                drop(cb);
                drop(reached);
            }
            seen.push(surface_counts(&s));
        }
        Beside::Handler => {
            let inside = Arc::new(Mutex::new(Vec::new()));
            let cb = {
                let _pool = Pool::here();
                let t = on_s();
                let cb = reading(&q, &t, false);
                drop(t);
                let (s, inside) = (Arc::clone(&s), Arc::clone(&inside));
                let handler = move |_| {
                    inside.lock().push(surface_counts(&s));
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    inside.lock().push(surface_counts(&s));
                };
                cb.add_completed_handler(OnFault::Die, Box::new(handler))
                    .unwrap();
                cb.commit(OnFault::Die).unwrap();
                cb
            };
            {
                let _pool = Pool::here();
                done(&q, &cb);
            }
            let after = surface_counts(&s);
            std::thread::sleep(std::time::Duration::from_millis(200));
            seen.extend(inside.lock().drain(..));
            seen.push(after);
            seen.push(surface_counts(&s));
            {
                let _pool = Pool::here();
                drop(cb);
            }
            seen.push(surface_counts(&s));
        }
        Beside::Two => {
            let (t, unused) = {
                let _pool = Pool::here();
                (on_s(), on_s())
            };
            {
                let _pool = Pool::here();
                let cb = reading(&q, &t, false);
                cb.commit(OnFault::Die).unwrap();
                done(&q, &cb);
            }
            seen.push(surface_counts(&s));
            {
                let _pool = Pool::here();
                drop(t);
            }
            seen.push(surface_counts(&s));
            {
                let _pool = Pool::here();
                drop(unused);
            }
            seen.push(surface_counts(&s));
        }
    }
    seen
}

/// [`surface_holds`] of each case on this platform and on Metal's, which agree, and are its
/// `want`. Metal lets go of what a completed command buffer retained after its waiters
/// returned, the more often once MoltenVK has worked on Metal's device
/// (`texture_identifiers_are_reused_as_metals`), so [`surface_holds`] reads the counts after a
/// completion once its queue let go of it ([`let_go`]).
fn holds_as_metals(cases: &[(Beside, &[(i64, i32)])]) {
    let (p, d) = (platform(), dev());
    for &(beside, want) in cases {
        let ours = surface_holds::<crate::platform::VulkanPlatform>(&p, &d, beside);
        assert_eq!(ours, want, "{beside:?}");
    }
}

/// Nothing beside a command buffer's own uses holds a surface's IOSurface, as on Metal: not
/// a command buffer of another queue that never names it, waiting while the texture on it
/// a completed one read is released, whose kernel reads a texture at a slot; not a second
/// texture on it, whose release lets go of its own reference only. A command buffer's
/// completed handler runs while it still holds the IOSurface, on Metal as here, and lets it go
/// once the handler returned. Where that other command buffer's kernel reads its texture
/// through an argument buffer, Metal holds nothing either, and this platform dies at the
/// dispatch naming `half-texel-conversion` ([`crate::answer::DIES`]): the runtime announces the
/// kind for a `half` read of an argument buffer's texture, whose format the function does not
/// know and whose texel it converts by a rounding Metal's may not be
#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "MoltenVK: the runtime records native-half-conversion, a death in strict mode; Mac-only test, off everywhere; ledger: MoltenVK strict deaths"
)]
fn nothing_beside_a_command_buffer_holds_a_surface() {
    let none = [(1, 0); 3];
    let handled = [(2, 0), (2, 0), (1, 0), (1, 0), (1, 0)];
    holds_as_metals(&[
        (Beside::Pending, &none[..]),
        (Beside::Handler, &handled[..]),
        (Beside::Two, &[(3, 0), (2, 0), (1, 0)][..]),
    ]);
    let kind = metal2vulkan_runtime::Degradation::HalfTexelConversion;
    let site = crate::answer::dies_of(kind, || {
        surface_holds::<crate::platform::VulkanPlatform>(
            &platform(),
            &dev(),
            Beside::PendingArgument,
        )
    });
    assert_eq!(site.file, "dispatchThreads:threadsPerThreadgroup:");
}

/// A command buffer of another queue that never names a surface, waiting while the texture
/// on it a completed one read is released, holds nothing of it where its kernel reaches a
/// buffer through an address it loads, as on Metal
#[test]
fn a_command_buffer_reaching_an_address_holds_no_surface_it_never_names() {
    holds_as_metals(&[(Beside::PendingReach, &[(1, 0); 3][..])]);
}

/// The address of a one-page buffer made and released on `d`: where the next range lies.
pub(crate) fn next_range<P: Platform>(d: &P::Device) -> u64 {
    use reims_vgpu_core::platform::resource::Buffer as _;
    let buffer = d.new_buffer_of_length(OnFault::Die, 0x4000, 0);
    buffer.unwrap().unwrap().gpu_address()
}

/// How far the next range moves ([`next_range`]) as up to `count` objects `make` makes are
/// made and kept on `d`, move by move; and whether it is back where it began once they are
/// all released.
fn range_moves<P: Platform, T>(
    d: &P::Device,
    count: usize,
    make: &dyn Fn() -> T,
) -> (Vec<u64>, bool) {
    let first = next_range::<P>(d);
    let (mut was, mut kept, mut moved) = (first, Vec::new(), Vec::new());
    for _ in 0..count {
        kept.push(make());
        let now = next_range::<P>(d);
        if now != was {
            moved.push(now.wrapping_sub(was));
            was = now;
        }
    }
    drop(kept);
    (moved, next_range::<P>(d) == first)
}

/// Private 32x64 RGBA8 textures read by shaders, of a heap size of 0x2000 bytes or a little
/// more, and private buffers of 0x100 bytes take slots in Metal's private pool, under 16 and
/// 512 to a pool, so one of the first 16 textures, and one of the first 513 buffers, makes
/// another pool, which takes a range of 0x28000 and goes once they are released
/// (`metal_space::PoolClass`): on Metal, where it runs, the next range moves on by that once
/// and comes back; here, the model's pools do. The runtime answers Metal's heap size, and the
/// textures' slots lie that apart, rounded up to 0x100, within the first pool, the last at
/// the next pool's first byte
#[test]
fn small_private_textures_and_buffers_fill_metals_private_pool() {
    use crate::gpu::metal_space::{PoolClass, process};
    let rgba8 = TextureDesc {
        pixel_format: 70,
        width: 32,
        height: 64,
        usage: 1,
        resource_options: Some(0x20),
        ..TextureDesc::default()
    };
    let (p, d) = (platform(), dev());
    // A queue makes the pools, as on Metal: the textures' first one is not the first use.
    drop(d.new_command_queue().unwrap());
    let pools = || process().pools_of(PoolClass::Private);
    let before = pools();
    let desc = p.texture_descriptor(&rgba8);
    let textures: Vec<_> = (0..16)
        .map(|_| d.new_texture(OnFault::Die, &desc).unwrap().unwrap())
        .collect();
    assert_eq!(pools(), before + 1, "16 textures");
    let heap = d.heap_texture_size_and_align(&desc).unwrap().size;
    reims_vgpu_core::metal_check!("the heap size", heap);
    let step = heap.next_multiple_of(0x100);
    let slots: Vec<u64> = textures
        .iter()
        .map(|t| t.space.address().unwrap())
        .collect();
    let (first, last) = slots.split_at(15);
    assert!(
        first.windows(2).all(|pair| pair[1] == pair[0] + step)
            && last[0] != first[14] + step
            && last[0] % crate::gpu::metal_space::SLOT == 0,
        "{step:#x}: {slots:x?}"
    );
    drop(textures);
    assert_eq!(pools(), before, "the textures released");
    let buffers: Vec<_> = (0..513)
        .map(|_| {
            d.new_buffer_of_length(OnFault::Die, 0x100, 0x20)
                .unwrap()
                .unwrap()
        })
        .collect();
    assert_eq!(pools(), before + 1, "513 buffers");
    drop(buffers);
    assert_eq!(pools(), before, "the buffers released");
}

/// Buffers under a page of each storage mode lie at their slots in Metal's pools
/// (`metal_space::PoolClass`): after a queue, ones of 1, 0x100, 0x101, 0x1001 and 0x3fff
/// bytes take 0x100, 0x100, 0x200, 0x1100 and 0x4000 bytes, each right after the one before,
/// the first shared one 0x4500 bytes into its pool, the first private one 0x100 and the first
/// managed one at a new pool's first byte, and one of 0x80 bytes made after the second's
/// release takes its slot. On Metal, where it runs, each lies that far from the first, and
/// the first that far into a 32-KiB granule; here, each answers its pool's address in the
/// space's model plus that offset, the runtime's slot in its reservation of the pool.
/// Another device's shared buffers lie after the first's in the same pool, in a reservation
/// of its own. The managed pool goes with its last buffer, and comes back at the next
#[test]
fn small_buffers_lie_at_their_slots_in_metals_pools() {
    use reims_vgpu_core::platform::resource::Buffer as _;

    use crate::gpu::metal_space::{PoolClass, process};
    const LENGTHS: [u64; 5] = [1, 0x100, 0x101, 0x1001, 0x3fff];
    const FROM_FIRST: [u64; 6] = [0, 0x100, 0x200, 0x400, 0x1500, 0x100];
    let classes = [
        (PoolClass::Shared, 0, 0x4500),
        (PoolClass::Private, 0x20, 0x100),
        (PoolClass::Managed, 0x10, 0),
    ];
    fn made<P: Platform>(d: &P::Device, options: u64) -> (Vec<P::Buffer>, Vec<u64>) {
        let make = |length| {
            let buffer = d.new_buffer_of_length(OnFault::Die, length, options);
            buffer.unwrap().unwrap()
        };
        let mut buffers: Vec<_> = LENGTHS.into_iter().map(make).collect();
        let mut at: Vec<u64> = buffers.iter().map(|b| b.gpu_address()).collect();
        drop(buffers.remove(1));
        buffers.push(make(0x80));
        at.extend(buffers.last().map(|b| b.gpu_address()));
        (buffers, at)
    }
    let d = dev();
    drop(d.new_command_queue().unwrap());
    let managed = || process().pool_addresses(PoolClass::Managed);
    let before = managed();
    let mut kept = Vec::new();
    for (class, options, offset) in classes {
        let (buffers, at) = made::<VulkanPlatform>(&d, options);
        let pool = *process().pool_addresses(class).last().unwrap();
        let into: Vec<u64> = at.iter().map(|a| a.wrapping_sub(pool)).collect();
        let expected: Vec<u64> = FROM_FIRST.iter().map(|o| offset + o).collect();
        assert_eq!(
            into, expected,
            "{class:?} at {at:x?}, its pool at {pool:#x}"
        );
        kept.push(buffers);
    }
    assert_eq!(managed().len(), before.len() + 1, "the managed pool");
    // Another device of the process places its buffers in the same pools, after the first
    // device's, as Metal's one device does: in the runtime's reservation of its own.
    let (own, _cache) = crate::gpu::device::tests::metal_tests::device_with_own_cache();
    let (others, at) = made::<VulkanPlatform>(&own, 0);
    let pool = process().pool_addresses(PoolClass::Shared)[0];
    let into: Vec<u64> = at.iter().map(|a| a.wrapping_sub(pool)).collect();
    let expected: Vec<u64> = FROM_FIRST.iter().map(|o| 0x9a00 + o).collect();
    assert_eq!(into, expected, "another device, at {at:x?}");
    drop((kept, others));
    assert_eq!(managed(), before, "the managed pool, its buffers released");
    let (_again, at) = made::<VulkanPlatform>(&d, 0x10);
    let pool = *managed().last().unwrap();
    assert_eq!(at[0], pool, "the managed pool again");
}

/// Sampler states of argument buffer support, each of its own LOD clamp, take descriptors in
/// Metal's sampler mempool, whose array takes a range at the 255th after a queue, the
/// device's own sampler counted (`metal_space::Samplers`): on Metal, where it runs, and
/// here, the next range moves on by 0x8000 there, once in 300
#[test]
fn sampler_states_grow_metals_sampler_mempool() {
    let made = std::cell::Cell::new(0u16);
    let next = || {
        made.set(made.get() + 1);
        SamplerDesc {
            support_argument_buffers: Some(true),
            lod_max_clamp: Some(1000.0 + f32::from(made.get())),
            ..SamplerDesc::default()
        }
    };
    let d = dev();
    drop(d.new_command_queue().unwrap());
    let make = || d.new_sampler_state(&next()).unwrap().unwrap();
    let (moves, _) = range_moves::<VulkanPlatform, _>(&d, 300, &make);
    assert_eq!(moves, [0x8000]);
}

/// A view in a value Metal describes no format for is nil, as Metal answers, from each of the
/// six view selectors, every value from 0 to 2047 tried, `MTLPixelFormatInvalid`'s 0 and the
/// formats of a later SDK (45 to 48, 95 to 100, 120 to 122) among them; before its swizzle,
/// type, levels and slices are checked, as Metal's: a swizzle channel of 9, type 99 and
/// levels whose end passes `u64::MAX` (measured one case a process, from `BGRA8Unorm`,
/// `RGBA16Float`, `R16Float` and `Depth32Float_Stencil8` sources, of `ShaderRead` usage with
/// and without `PixelFormatView`; compared here with the three selectors that take no resource
/// index). But for 255 and 262, which Metal's setter takes: a view in either, of the
/// texture's type, levels and swizzle, dies at its selector, where Metal asserts
/// ("_mtlValidateArgumentsForTextureViewOnDevice:1866: failed assertion `Texture Creation
/// texture view pixelFormat (MTLPixelFormatInvalid) not castable."; a swizzle channel of 9
/// asserts first)
#[test]
fn views_metal_answers_nil_are_nil() {
    use reims_vgpu_core::platform::resource::Texture as _;
    let t = dev()
        .new_texture(OnFault::Return, &bgra(8, 8, 0))
        .unwrap()
        .unwrap();
    let r = |location, length| NsRange { location, length };
    let (one, past) = (r(0, 1), r(u64::MAX, 2));
    let nine = Swizzle {
        red: 9,
        green: 3,
        blue: 4,
        alpha: 5,
    };
    let plain = Swizzle {
        red: 2,
        green: 3,
        blue: 4,
        alpha: 5,
    };
    type View<'a> = (&'static str, Box<dyn Fn() -> Option<VulkanTexture> + 'a>);
    // Each selector's view of `t` in `f`, the subviews of a swizzle, type or levels Metal
    // checks after the format where `odd`, else of the texture's.
    let views = |f: u64, odd: bool| -> [View<'_>; 6] {
        let t = &t;
        let (kind, levels, nine) = match odd {
            true => (99, past, nine),
            false => (2, one, plain),
        };
        [
            (
                "newTextureViewWithPixelFormat:",
                Box::new(move || t.new_texture_view(f).unwrap()),
            ),
            (
                "newTextureViewWithPixelFormat:resourceIndex:",
                Box::new(move || t.new_texture_view_at_index(f, 0).unwrap()),
            ),
            (
                "newTextureViewWithPixelFormat:textureType:levels:slices:",
                Box::new(move || t.new_texture_subview(f, kind, levels, one).unwrap()),
            ),
            (
                "newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:",
                Box::new(move || (t.new_swizzled_texture_subview(f, 2, one, one, nine)).unwrap()),
            ),
            (
                "newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:resourceIndex:",
                Box::new(move || {
                    (t.new_swizzled_texture_subview_at_index(f, kind, one, one, nine, 0)).unwrap()
                }),
            ),
            (
                "newTextureViewWithPixelFormat:textureType:levels:slices:resourceIndex:",
                Box::new(move || (t.new_texture_subview_at_index(f, 2, levels, one, 0)).unwrap()),
            ),
        ]
    };
    let mut nil = 0;
    for f in (0..2048)
        .filter(|&f| !crate::gpu::format_info::described("newTextureViewWithPixelFormat:", f))
    {
        let taken = matches!(f, 255 | 262);
        for (sel, view) in views(f, !taken) {
            if taken {
                *crate::answer::LAST_LINE.lock() = None;
                let site = catch_death(&view).unwrap_err();
                let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
                assert_eq!(
                    (site.kind, site.file),
                    (DeathKind::Abort, sel),
                    "{f}: {line}"
                );
                assert!(line.contains(&format!("pixel format {f}")), "{line}");
                assert!(line.contains("Metal aborts at"), "{line}");
            } else {
                assert!(view().is_none(), "{sel} {f}");
                nil += 1;
            }
        }
    }
    assert_eq!(nil, 6 * (2048 - 264 - 2));
}

/// The surface's bytes, after the work that wrote them completed.
fn surface_bytes(s: &OsSurface) -> Vec<u8> {
    let n = usize::try_from(s.alloc_size()).unwrap();
    // SAFETY: the surface's memory, which no command uses now.
    unsafe { std::slice::from_raw_parts(s.base_address().as_ptr::<u8>(), n) }.to_vec()
}

/// What a render pass of [`rotated_uses`] draws.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Drawn {
    /// Nothing: it sets its pipeline state alone.
    Nothing,
    /// A gradient over the top left half of its frame, which reaches every 32 by 32 tile of
    /// a frame of 32 by 32.
    Corner,
    /// That gradient under a scissor rectangle of the frame's top left 32 by 32 tile, which
    /// reaches that tile alone of a frame of 64 by 64.
    FirstTile,
}

impl Drawn {
    /// The side of the square frame the pass draws: 64 for [`Drawn::FirstTile`], else 32.
    fn side(self) -> u64 {
        match self {
            Drawn::FirstTile => 64,
            Drawn::Nothing | Drawn::Corner => 32,
        }
    }
}

/// What a kernel, a blit, the CPU and a render pass see and leave of a texture of `rotation`
/// over a square `BGRA8Unorm` surface of `drawn`'s side whose texels name their places, on
/// `p`'s device `d`, each named: a kernel's `read()` of every texel, a blit's copy of the
/// texture into a buffer, the surface's bytes after the CPU replaced an 8 by 4 region at
/// (4, 2) of the texture, the surface's bytes after a pass of load action `load` that draws
/// `drawn`, and a kernel's `read()` after the pass.
fn rotated_uses<P: Platform<Surface = OsSurface>>(
    p: &P,
    d: &P::Device,
    rotation: u64,
    (load, drawn): (u64, Drawn),
) -> Vec<(&'static str, Vec<u8>)> {
    use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder as _};
    use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
    use reims_vgpu_core::platform::descriptor::RenderPassDescriptor as _;
    use reims_vgpu_core::platform::render::{RenderCmd, RenderEncoder as _};
    use reims_vgpu_core::platform::resource::{Buffer as _, Library as _, Texture as _};
    use reims_vgpu_core::platform::values::{ColorAttachmentSet, RenderPipelineDesc};

    use crate::gpu::ycbcr::measure::{PROBE, probed};
    let side = drawn.side();
    let s = bgra_surface(side, side);
    // Each texel names its place: (x, y, 7, 200).
    let fill: Vec<u8> = (0..side * side)
        .flat_map(|i| [(i % side) as u8, (i / side) as u8, 7, 200])
        .collect();
    // SAFETY: the surface's memory, of `side` rows of `side` texels, which no command uses.
    unsafe {
        std::ptr::copy_nonoverlapping(fill.as_ptr(), s.base_address().as_ptr::<u8>(), fill.len());
    }
    let desc = p.texture_descriptor(&TextureDesc {
        pixel_format: 80,
        width: side,
        height: side,
        usage: 5,
        resource_options: Some(0),
        framebuffer_only: Some(true),
        rotation: Some(rotation),
        ..TextureDesc::default()
    });
    let t = d
        .new_texture_on_surface(&desc, Some(&s), 0)
        .unwrap()
        .unwrap();
    let mut seen = vec![(
        "read",
        probed::<P>(d, "read_texels", &t, (side, side), None, None),
    )];
    let q = d.new_command_queue().unwrap();
    let out = d
        .new_buffer_of_length(OnFault::Die, side * side * 4, 0)
        .unwrap()
        .unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let e = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
    let copy = BlitCmd::CopyTextureToBuffer {
        source_texture: &t,
        source_slice: 0,
        source_level: 0,
        source_origin: MtlOrigin::new(0, 0, 0),
        source_size: MtlSize::new(side, side, 1),
        destination_buffer: &out,
        destination_offset: 0,
        destination_bytes_per_row: side * 4,
        destination_bytes_per_image: side * side * 4,
    };
    e.encode(OnFault::Die, copy).unwrap();
    e.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    let n = usize::try_from(side * side * 4).unwrap();
    // SAFETY: a Shared buffer's bytes, read after the command buffer completed.
    let copied = unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), n) };
    seen.push(("blit", copied.to_vec()));
    let region = MtlRegion::new(MtlOrigin::new(4, 2, 0), MtlSize::new(8, 4, 1));
    let bytes: Vec<u8> = (0..8 * 4 * 4).map(|i: u8| i.wrapping_mul(3) | 1).collect();
    t.replace_region(region, 0, 0, HostRange::of_slice(&bytes), 8 * 4, 0)
        .unwrap();
    seen.push(("cpu", surface_bytes(&s)));
    let library = d.new_library(PROBE).unwrap().unwrap().unwrap();
    let function = |name| library.new_function(OnFault::Die, name).unwrap().unwrap();
    let (vertex, fragment) = (function("corner_vertex"), function("gradient_fragment"));
    let state = d
        .new_render_pipeline_state(&RenderPipelineDesc {
            label: None,
            vertex: &vertex,
            fragment: &fragment,
            color0_pixel_format: 80,
            resource_index: None,
        })
        .unwrap()
        .unwrap();
    let pass = p.render_pass_descriptor();
    let attachment = ColorAttachmentSet {
        texture: Some(Some(&t)),
        load_action: Some(load),
        store_action: Some(1),
        ..ColorAttachmentSet::default()
    };
    pass.set_color_attachment(OnFault::Die, 0, &attachment)
        .unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
    let set = RenderCmd::SetRenderPipelineState {
        pipeline_state: Some(&state),
    };
    let scissor = RenderCmd::SetScissorRect {
        rect: reims_vgpu_core::platform::types::MtlScissorRect {
            x: 0,
            y: 0,
            width: 32,
            height: 32,
        },
    };
    let draw = RenderCmd::DrawPrimitives {
        primitive_type: 3,
        vertex_start: 0,
        vertex_count: 3,
    };
    let cmds = match drawn {
        Drawn::Nothing => vec![set],
        Drawn::Corner => vec![set, draw],
        Drawn::FirstTile => vec![set, scissor, draw],
    };
    for cmd in cmds.into_iter().chain([RenderCmd::EndEncoding]) {
        e.encode(OnFault::Die, cmd).unwrap();
    }
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    assert!(cb.error().is_none(), "{:?}", cb.error());
    seen.push(("pass", surface_bytes(&s)));
    seen.push((
        "read after the pass",
        probed::<P>(d, "read_texels", &t, (side, side), None, None),
    ));
    seen
}

/// A texture of each rotation over an IOSurface reads, copies, takes the CPU's writes and
/// stores a pass's texels as Metal's does, the surface's bytes compared byte for byte: a
/// kernel's `read()` sees the stored frame unturned, a blit and the CPU address the plane at
/// its rows, and a pass that clears or loads and draws over every 32 by 32 tile of its frame
/// stores its texels turned (rotation 2 by 180 degrees, 1 and 4 a quarter turn, 3 the other),
/// one that loads loading them turned back, but at rotation 4, which loads the plane
/// transposed, so the texels it loaded land mirrored left to right (the plane's texel (0, 0)
/// holding what (31, 0) held). Each rotation's pass that draws leaves the surface otherwise
/// than rotation 0's. A pass that clears or loads and draws nothing stores the plane as
/// Metal's does: in place where it loads. The pass that loads a texture of rotation 4 and
/// draws dies at `endEncoding` naming `rotated-load-tiles` ([`crate::answer::DIES`]), which the
/// runtime announces at every such pass, though this one's draw reaches every tile; Metal's
/// side of it runs
#[test]
fn rotated_textures_read_copy_and_store_as_metals() {
    let (p, d) = (platform(), dev());
    let passes = [CLEAR, LOAD].map(|l| [(l, Drawn::Corner), (l, Drawn::Nothing)]);
    for pass in passes.into_iter().flatten() {
        let unturned = rotated_uses::<crate::platform::VulkanPlatform>(&p, &d, 0, pass);
        for rotation in 1..=4 {
            let case = format!("rotation {rotation} (load action, drawn) {pass:?}");
            if (rotation, pass) == (4, (LOAD, Drawn::Corner)) {
                let kind = rt::Degradation::RotatedLoadTiles;
                let site = crate::answer::dies_of(kind, || {
                    rotated_uses::<crate::platform::VulkanPlatform>(&p, &d, rotation, pass)
                });
                assert_eq!(site.file, "endEncoding", "{case}");
                continue;
            }
            let ours = rotated_uses::<crate::platform::VulkanPlatform>(&p, &d, rotation, pass);
            if pass.1 == Drawn::Corner {
                assert_ne!(ours[3], unturned[3], "{case}: the pass turns nothing");
            }
            for (what, o) in &ours {
                reims_vgpu_core::metal_check_bytes!(format!("{case}: {what}"), o);
            }
        }
    }
}

/// A texture of `rotation` over a square `BGRA8Unorm` surface of `side` whose texels name
/// their places (as [`rotated_uses`] fills it), made on `p`'s device `d` with `ShaderRead`,
/// `RenderTarget` and `PixelFormatView` usage, and the surface.
fn rotated_named<P: Platform<Surface = OsSurface>>(
    p: &P,
    d: &P::Device,
    side: u64,
    rotation: u64,
) -> (OsSurface, P::Texture) {
    let s = bgra_surface(side, side);
    let fill: Vec<u8> = (0..side * side)
        .flat_map(|i| [(i % side) as u8, (i / side) as u8, 7, 200])
        .collect();
    // SAFETY: the surface's memory, of `side` rows of `side` texels, which no command uses.
    unsafe {
        std::ptr::copy_nonoverlapping(fill.as_ptr(), s.base_address().as_ptr::<u8>(), fill.len());
    }
    let desc = p.texture_descriptor(&TextureDesc {
        pixel_format: 80,
        width: side,
        height: side,
        usage: 0x15,
        resource_options: Some(0),
        framebuffer_only: Some(true),
        rotation: Some(rotation),
        ..TextureDesc::default()
    });
    let t = d
        .new_texture_on_surface(&desc, Some(&s), 0)
        .unwrap()
        .unwrap();
    (s, t)
}

/// The surface's bytes after a pass of load action `load` that draws `drawn` (as
/// [`rotated_uses`]'s pass does) through a view in `view` of a texture of `rotation`
/// ([`rotated_named`]), its pipeline state of the view's format, on `p`'s device `d`.
fn rotated_view_pass<P: Platform<Surface = OsSurface>>(
    p: &P,
    d: &P::Device,
    rotation: u64,
    view: u64,
    (load, drawn): (u64, Drawn),
) -> Vec<u8> {
    use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
    use reims_vgpu_core::platform::descriptor::RenderPassDescriptor as _;
    use reims_vgpu_core::platform::render::{RenderCmd, RenderEncoder as _};
    use reims_vgpu_core::platform::resource::{Library as _, Texture as _};
    use reims_vgpu_core::platform::values::{ColorAttachmentSet, RenderPipelineDesc};

    use crate::gpu::ycbcr::measure::PROBE;
    let (s, t) = rotated_named(p, d, drawn.side(), rotation);
    let v = t.new_texture_view(view).unwrap().unwrap();
    let library = d.new_library(PROBE).unwrap().unwrap().unwrap();
    let function = |name| library.new_function(OnFault::Die, name).unwrap().unwrap();
    let (vertex, fragment) = (function("corner_vertex"), function("gradient_fragment"));
    let state = d
        .new_render_pipeline_state(&RenderPipelineDesc {
            label: None,
            vertex: &vertex,
            fragment: &fragment,
            color0_pixel_format: view,
            resource_index: None,
        })
        .unwrap()
        .unwrap();
    let pass = p.render_pass_descriptor();
    let attachment = ColorAttachmentSet {
        texture: Some(Some(&v)),
        load_action: Some(load),
        store_action: Some(1),
        ..ColorAttachmentSet::default()
    };
    pass.set_color_attachment(OnFault::Die, 0, &attachment)
        .unwrap();
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
    let set = RenderCmd::SetRenderPipelineState {
        pipeline_state: Some(&state),
    };
    let draw = RenderCmd::DrawPrimitives {
        primitive_type: 3,
        vertex_start: 0,
        vertex_count: 3,
    };
    let cmds = match drawn {
        Drawn::Nothing => vec![set],
        Drawn::Corner | Drawn::FirstTile => vec![set, draw],
    };
    for cmd in cmds.into_iter().chain([RenderCmd::EndEncoding]) {
        e.encode(OnFault::Die, cmd).unwrap();
    }
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    assert!(cb.error().is_none(), "{:?}", cb.error());
    surface_bytes(&s)
}

/// The surface's bytes after a pass that clears a 4-sample texture of `format` and resolves
/// it into a texture of `rotation` ([`rotated_named`], 32 by 32), or into its view in
/// `format` where that is not the texture's, on `p`'s device `d`.
fn rotated_resolve<P: Platform<Surface = OsSurface>>(
    p: &P,
    d: &P::Device,
    rotation: u64,
    format: u64,
) -> Vec<u8> {
    use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
    use reims_vgpu_core::platform::descriptor::RenderPassDescriptor as _;
    use reims_vgpu_core::platform::render::{RenderCmd, RenderEncoder as _};
    use reims_vgpu_core::platform::resource::Texture as _;
    use reims_vgpu_core::platform::types::ClearColor;
    use reims_vgpu_core::platform::values::ColorAttachmentSet;
    let (s, t) = rotated_named(p, d, 32, rotation);
    let v = (format != 80).then(|| t.new_texture_view(format).unwrap().unwrap());
    let msaa = d
        .new_texture(
            OnFault::Die,
            &p.texture_descriptor(&TextureDesc {
                texture_type: 4,
                sample_count: 4,
                pixel_format: format,
                width: 32,
                height: 32,
                usage: 4,
                resource_options: Some(0x20),
                ..TextureDesc::default()
            }),
        )
        .unwrap()
        .unwrap();
    let pass = p.render_pass_descriptor();
    let attachment = ColorAttachmentSet {
        texture: Some(Some(&msaa)),
        resolve_texture: Some(Some(v.as_ref().unwrap_or(&t))),
        load_action: Some(CLEAR),
        store_action: Some(2),
        clear_color: Some(ClearColor {
            red: 0.25,
            green: 0.5,
            blue: 0.75,
            alpha: 1.0,
        }),
        ..ColorAttachmentSet::default()
    };
    pass.set_color_attachment(OnFault::Die, 0, &attachment)
        .unwrap();
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
    e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    assert!(cb.error().is_none(), "{:?}", cb.error());
    surface_bytes(&s)
}

/// A pass through a view of a texture of each rotation over an IOSurface, in
/// `BGRA8Unorm_sRGB`, `RGBA8Unorm` or `RGBA8Unorm_sRGB`, stores and loads the surface's bytes
/// as Metal's does, byte for byte: one that clears or loads and draws over every 32 by 32 tile
/// of its frame, and one that clears or loads and draws nothing. One that loads a texture of
/// rotation 4 and draws dies at `endEncoding` naming `rotated-load-tiles`
/// ([`crate::answer::DIES`]), which the runtime announces at every such pass, though this one's
/// draw reaches every tile
#[test]
fn views_of_rotated_textures_store_and_load_as_metals() {
    let (p, d) = (platform(), dev());
    let passes = [CLEAR, LOAD].map(|l| [(l, Drawn::Corner), (l, Drawn::Nothing)]);
    for view in [81, 70, 71] {
        for pass in passes.into_iter().flatten() {
            for rotation in 1..=4 {
                type V = crate::platform::VulkanPlatform;
                let case = format!("view {view}, rotation {rotation}, {pass:?}");
                if (rotation, pass) == (4, (LOAD, Drawn::Corner)) {
                    let kind = rt::Degradation::RotatedLoadTiles;
                    let site = crate::answer::dies_of(kind, || {
                        rotated_view_pass::<V>(&p, &d, rotation, view, pass)
                    });
                    assert_eq!(site.file, "endEncoding", "view {view}, rotation {rotation}");
                    continue;
                }
                let ours = rotated_view_pass::<V>(&p, &d, rotation, view, pass);
                reims_vgpu_core::metal_check_bytes!(case, &ours);
            }
        }
    }
}

/// A pass that resolves a cleared 4-sample texture into a texture of each rotation over an
/// IOSurface, or into its view in `BGRA8Unorm_sRGB` or `RGBA8Unorm`, stores the surface's
/// bytes as Metal's does, byte for byte
#[test]
fn rotated_textures_and_their_views_resolve_as_metals() {
    let (p, d) = (platform(), dev());
    for format in [80, 81, 70] {
        for rotation in 1..=4 {
            type V = crate::platform::VulkanPlatform;
            let key = format!("format {format}, rotation {rotation}");
            let ours = rotated_resolve::<V>(&p, &d, rotation, format);
            reims_vgpu_core::metal_check_bytes!(key, &ours);
        }
    }
}

/// A view of a texture of each rotation over an IOSurface in a format other than the four
/// (`R32Uint`), or of another type (a 2D array), dies at its selector, where Metal makes it.
/// No view of another range can be asked: the texture has one level and one slice
#[test]
fn views_of_rotated_textures_the_runtime_refuses_die() {
    use reims_vgpu_core::platform::resource::Texture as _;
    let (p, d) = (platform(), dev());
    let one = NsRange::new(0, 1);
    for rotation in 1..=4 {
        let (_s, t) = rotated_named(&p, &d, 32, rotation);
        let site = catch_death(|| t.new_texture_view(53).map(drop)).unwrap_err();
        assert_eq!(
            (site.kind, site.file),
            (DeathKind::Abort, "newTextureViewWithPixelFormat:"),
            "rotation {rotation}"
        );
        let site = catch_death(|| t.new_texture_subview(80, 3, one, one).map(drop)).unwrap_err();
        assert_eq!(
            (site.kind, site.file),
            (
                DeathKind::Abort,
                "newTextureViewWithPixelFormat:textureType:levels:slices:"
            ),
            "rotation {rotation}"
        );
    }
}

/// `MTLLoadActionLoad`, `MTLLoadActionClear`.
const LOAD: u64 = 1;

const CLEAR: u64 = 2;

/// A pass that loads a texture of rotation 4 and draws over some of its frame's 32 by 32 tiles
/// dies at `endEncoding` naming `rotated-load-tiles` ([`crate::answer::DIES`]): the runtime
/// lands the tiles no draw reaches mirrored as the others, where Metal loads and stores the
/// tiles a draw reaches alone, keeping the others in place
#[test]
fn rotated_loads_of_some_tiles_are_metals() {
    let (p, d) = (platform(), dev());
    let pass = (LOAD, Drawn::FirstTile);
    let kind = rt::Degradation::RotatedLoadTiles;
    let site = crate::answer::dies_of(kind, || {
        rotated_uses::<crate::platform::VulkanPlatform>(&p, &d, 4, pass)
    });
    assert_eq!(site.file, "endEncoding");
}

/// Textures of format 84 (plane 1's view format of the two-plane XR10 textures) and of the
/// XR10 formats are made with read and write usage, as Metal makes them on this host, which
/// answers supportsExtendedXR10Formats yes
#[test]
fn xr10_textures_are_made() {
    let formats = [84, 550, 551, 552, 553, 554, 555];
    let made_on = |made: &dyn Fn(u64) -> Option<u64>| formats.map(made);
    let d = dev();
    let ours = made_on(&|format| {
        let desc = bgra(8, 8, 0);
        desc.set_usage(0x3);
        desc.set_pixel_format(format);
        let t = d.new_texture(OnFault::Return, &desc).unwrap();
        t.map(|t| t.pixel_format())
    });
    assert_eq!(ours, formats.map(Some));
    assert!(d.caps().flags.supports_extended_xr10_formats);
}

/// What 2D samples of a 24 by 12 texture of `format` (550 or 551) read on platform `P`, as
/// halves: its texels written by a kernel from an `RGBA16Float` texture of `seed`'s halves of
/// every sign, below 2 in magnitude, then sampled at positions a third of a texel apart from
/// two texels before the texture to two past it, nearest and linear, in texel coordinates
/// clamped to the edge and in normalized ones repeated.
fn two_plane_samples_on<P: Platform>(
    d: &P::Device,
    desc: &dyn Fn(&TextureDesc) -> P::TextureDescriptor,
    format: u64,
    seed: u32,
) -> Vec<Vec<u8>> {
    use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
    use reims_vgpu_core::platform::compute::{ComputeCmd, ComputeEncoder as _};
    use reims_vgpu_core::platform::resource::{Library as _, Texture as _};

    use crate::gpu::ycbcr::measure::{PROBE, probed};
    use crate::gpu::ycbcr::measure::{probe, sampler};
    let (w, h) = (24, 12);
    let texture = |pixel_format, usage| {
        d.new_texture(
            OnFault::Return,
            &desc(&TextureDesc {
                pixel_format,
                width: w,
                height: h,
                usage,
                ..TextureDesc::default()
            }),
        )
        .unwrap()
        .unwrap()
    };
    let (source, t) = (texture(115, 0x1), texture(format, 0x3));
    let mut x = seed;
    let halves: Vec<u8> = (0..w * h * 4)
        .flat_map(|_| {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            // A sign, an exponent field below 16 (a magnitude below 2) and ten bits of
            // mantissa.
            let bits = (x >> 16) & 0xbfff;
            u16::try_from(bits).unwrap().to_le_bytes()
        })
        .collect();
    let all = MtlRegion::new(MtlOrigin::new(0, 0, 0), MtlSize::new(w, h, 1));
    source
        .replace_region(all, 0, 0, HostRange::of_slice(&halves), w * 8, 0)
        .unwrap();
    let library = d.new_library(PROBE).unwrap().unwrap().unwrap();
    let f = library
        .new_function(OnFault::Die, "copy_texels")
        .unwrap()
        .unwrap();
    let copy = d
        .new_compute_pipeline_state_for_function(Some(&f))
        .unwrap()
        .unwrap();
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let e = cb.compute_encoder(OnFault::Die).unwrap().unwrap();
    for cmd in [
        ComputeCmd::SetComputePipelineState { state: Some(&copy) },
        ComputeCmd::SetTextures {
            textures: &[Some(&source), Some(&t)],
            range: NsRange::new(0, 2),
        },
        ComputeCmd::DispatchThreads {
            threads_per_grid: MtlSize::new(w, h, 1),
            threads_per_threadgroup: MtlSize::new(8, 8, 1),
        },
        ComputeCmd::EndEncoding,
    ] {
        e.encode(OnFault::Die, cmd).unwrap();
    }
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    let grid = (w * 3 + 12, h * 3 + 12);
    let mut reads = Vec::new();
    for filter in [0, 1] {
        for (address, normalized) in [(0, false), (2, true)] {
            let s = sampler::<P>(d, (filter, address, normalized), 0);
            let at = probe((-2.0, -2.0), 1.0 / 3.0, grid, normalized, 0);
            let kernel = "sample_only_texels";
            reads.push(probed::<P>(d, kernel, &t, grid, Some(&s), Some(&at)));
        }
    }
    reads
}

/// A texture of a two-plane XR10 format, 550 or 551, written by a kernel, samples in 2D as
/// Metal's does on the M3 Max, bit for bit: nearest and linear, clamped to the edge and repeated
///
/// The runtime binds a two-plane XR10 texture to its kernels as `RGBA16Float`, a format that may
/// hold a NaN, and announces `device-nan-encoding` there: the first dispatch of this platform's
/// side dies, naming the kind ([`crate::answer::DIES`]).
#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "MoltenVK: the runtime records device-pack-conversion, a death in strict mode; Mac-only test, off everywhere; ledger: MoltenVK strict deaths"
)]
fn two_plane_xr10_samples_are_the_metal_platforms() {
    let site = crate::answer::dies_of(metal2vulkan_runtime::Degradation::DeviceNanEncoding, || {
        let d = dev();
        let vulkan_desc = |t: &TextureDesc| platform().texture_descriptor(t);
        for format in [550, 551] {
            for seed in [3, 4] {
                // Metal's first: this platform's side dies at its first dispatch.
                let ours = two_plane_samples_on::<VulkanPlatform>(&d, &vulkan_desc, format, seed);
                for (i, read) in ours.iter().enumerate() {
                    let values: std::collections::HashSet<&[u8]> = read.chunks(8).collect();
                    assert!(
                        values.len() > 100,
                        "{format} seed {seed} case {i}: {}",
                        values.len()
                    );
                }
                for (i, o) in ours.iter().enumerate() {
                    let key = format!("{format} seed {seed} case {i}");
                    reims_vgpu_core::metal_check_bytes!(key, o);
                }
            }
        }
    });
    assert_eq!(site.file, "dispatchThreads:threadsPerThreadgroup:");
}

/// Making a texture of MTLPixelFormatInvalid, or of an empty size, dies at the creator, as
/// Metal aborts there; the descriptor takes either. One of X32_Stencil8 is made, as Metal makes
/// it on this host
#[test]
fn textures_metal_aborts_making_die() {
    let d = dev();
    let sel = "newTextureWithDescriptor:";
    let invalid = bgra(8, 8, 0);
    invalid.set_pixel_format(0);
    let site = catch_death(|| d.new_texture(OnFault::Return, &invalid)).unwrap_err();
    assert_eq!((site.kind, site.file), (DeathKind::Abort, sel));
    for (width, height) in [(0, 8), (8, 0)] {
        let empty = bgra(width, height, 0);
        let site = catch_death(|| d.new_texture(OnFault::Return, &empty)).unwrap_err();
        assert_eq!((site.kind, site.file), (DeathKind::Abort, sel));
    }
    let s = bgra_surface(8, 8);
    let site = catch_death(|| d.new_texture_on_surface(&bgra(0, 8, 0), Some(&s), 0)).unwrap_err();
    assert_eq!(site.file, "newTextureWithDescriptor:iosurface:plane:");
    let stencil_view_format = bgra(8, 8, 0x20);
    stencil_view_format.set_pixel_format(261);
    let stencil = d
        .new_texture(OnFault::Return, &stencil_view_format)
        .unwrap();
    assert_eq!(stencil.map(|t| t.pixel_format()), Some(261));
}

/// newHeapWithDescriptor: of core's values: the descriptor's options, the heap's and its
/// storage mode are Metal's on this host; a heap of size 0 is nil, as Metal's; a Managed
/// heap dies, as Metal asserts at it
#[test]
fn heaps_of_descriptors() {
    use OptionSet::{HazardTrackingMode as Tracking, StorageMode as Storage};
    let values = [
        vec![],
        vec![Storage(2), Tracking(0)],
        vec![Storage(2), Tracking(2)],
        vec![Storage(2), Tracking(1)],
        vec![Storage(0), Tracking(0)],
    ];
    fn answers<P: Platform>(made: HeapMade<P>) -> (u64, u64, u64) {
        let h = made.heap.unwrap();
        (
            made.descriptor_resource_options,
            h.resource_options(),
            h.storage_mode(),
        )
    }
    let d = dev();
    let ours: Vec<_> = values
        .iter()
        .map(|v| answers(heap_of::<VulkanPlatform>(&d, 0x8000, v)))
        .collect();
    let measured = [
        (0x20, 0x120, 2),
        (0x20, 0x120, 2),
        (0x220, 0x220, 2),
        (0x120, 0x120, 2),
        (0, 0x100, 0),
    ];
    assert_eq!(ours, measured);
    let (a, b) = (private_heap(), private_heap());
    assert!(a.gpu_address() != 0 && a.gpu_address() != b.gpu_address());
    assert!(
        heap_of::<VulkanPlatform>(&d, 0, &[Storage(2)])
            .heap
            .is_none()
    );
    let site = catch_death(|| {
        heap_of::<VulkanPlatform>(&d, 0x8000, &[Storage(1)])
            .heap
            .is_some()
    })
    .unwrap_err();
    assert_eq!(
        (site.kind, site.file),
        (DeathKind::Abort, "newHeapWithDescriptor:")
    );
}

/// What a buffer placed over the bytes of a 16 by 12 texture of `format` (550 or 551), one
/// level, at offset 0 of a placement heap of storage mode `storage` (0 Shared, 2 Private) on
/// `P`'s device `d` sees and changes of the texture's texels: the bytes of the texture's span,
/// each first its offset's low byte, after a kernel writes the texels from an `RGBA16Float`
/// texture of halves of every sign, read through a copy of the buffer; then the texels
/// `read()` reads after another pattern is copied into the buffer. Every read and write is
/// the GPU's, a command buffer each, so a Private heap's bytes are read as a Shared one's.
fn two_plane_heap_bytes_on<P: Platform>(
    d: &P::Device,
    desc: &dyn Fn(&TextureDesc) -> P::TextureDescriptor,
    format: u64,
    storage: u64,
) -> (Vec<u8>, Vec<u8>)
where
    P::Device: GpuDevice<P>,
{
    use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder as _};
    use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
    use reims_vgpu_core::platform::compute::{ComputeCmd, ComputeEncoder as _};
    use reims_vgpu_core::platform::resource::{Buffer as _, Heap as _, Library as _};

    use crate::gpu::ycbcr::measure::{PROBE, probed_texels};
    let (w, h) = (16, 12);
    let options = [
        OptionSet::StorageMode(storage),
        OptionSet::HazardTrackingMode(0),
    ];
    let heap = heap_of::<P>(d, 0x10000, &options).heap.unwrap();
    let texture_desc = desc(&TextureDesc {
        pixel_format: format,
        width: w,
        height: h,
        usage: 0x3,
        resource_options: Some(0x100 | storage << 4),
        ..TextureDesc::default()
    });
    let size = d.heap_texture_size_and_align(&texture_desc).unwrap().size;
    let t = heap.new_texture(&texture_desc, 0).unwrap().unwrap();
    let over = heap
        .new_buffer_of_length(size, storage << 4, 0)
        .unwrap()
        .unwrap();
    let shared = |bytes: &[u8]| {
        let length = u64::try_from(bytes.len()).unwrap();
        let b = d
            .new_buffer_of_length(OnFault::Die, length, 0)
            .unwrap()
            .unwrap();
        // SAFETY: a Shared buffer of `bytes.len()` bytes, which no command uses yet.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), b.contents().as_ptr::<u8>(), bytes.len());
        }
        b
    };
    let q = d.new_command_queue().unwrap();
    let copied = |from: &P::Buffer, to: &P::Buffer| {
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
        let copy = BlitCmd::CopyBufferToBuffer {
            source_buffer: Some(from),
            source_offset: 0,
            destination_buffer: Some(to),
            destination_offset: 0,
            size,
        };
        e.encode(OnFault::Die, copy).unwrap();
        e.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        assert!(cb.error().is_none(), "{:?}", cb.error());
    };
    let span = usize::try_from(size).unwrap();
    let offsets: Vec<u8> = (0..span).map(|i| (i & 0xff) as u8).collect();
    copied(&shared(&offsets), &over);
    let source = d
        .new_texture(
            OnFault::Return,
            &desc(&TextureDesc {
                pixel_format: 115,
                width: w,
                height: h,
                usage: 0x1,
                ..TextureDesc::default()
            }),
        )
        .unwrap()
        .unwrap();
    let mut x = 5u32;
    let halves: Vec<u8> = (0..w * h * 4)
        .flat_map(|_| {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            u16::try_from((x >> 16) & 0xbfff).unwrap().to_le_bytes()
        })
        .collect();
    let all = MtlRegion::new(MtlOrigin::new(0, 0, 0), MtlSize::new(w, h, 1));
    source
        .replace_region(all, 0, 0, HostRange::of_slice(&halves), w * 8, 0)
        .unwrap();
    let library = d.new_library(PROBE).unwrap().unwrap().unwrap();
    let f = library
        .new_function(OnFault::Die, "copy_texels")
        .unwrap()
        .unwrap();
    let copy = d
        .new_compute_pipeline_state_for_function(Some(&f))
        .unwrap()
        .unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let e = cb.compute_encoder(OnFault::Die).unwrap().unwrap();
    for cmd in [
        ComputeCmd::SetComputePipelineState { state: Some(&copy) },
        ComputeCmd::SetTextures {
            textures: &[Some(&source), Some(&t)],
            range: NsRange::new(0, 2),
        },
        ComputeCmd::DispatchThreads {
            threads_per_grid: MtlSize::new(w, h, 1),
            threads_per_threadgroup: MtlSize::new(8, 4, 1),
        },
        ComputeCmd::EndEncoding,
    ] {
        e.encode(OnFault::Die, cmd).unwrap();
    }
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    assert!(cb.error().is_none(), "{:?}", cb.error());
    let back = shared(&vec![0; span]);
    copied(&over, &back);
    // SAFETY: a Shared buffer's `span` bytes, read after the copy completed.
    let seen = unsafe { std::slice::from_raw_parts(back.contents().as_ptr::<u8>(), span) };
    let seen = seen.to_vec();
    let pattern: Vec<u8> = (0..span).map(|i| (i * 7 + 3) as u8).collect();
    copied(&shared(&pattern), &over);
    let texels = probed_texels::<P>(d, ("read_texels", 8), &t, (w, h), None, None);
    (seen, texels)
}

/// A buffer placed over a one-level 2D texture of a two-plane XR10 format (550, 551) in a
/// placement heap, Shared or Private, sees and changes the texture's texels as Metal's does
/// on this host, byte for byte, the bytes between the rows and the planes included
/// ([`two_plane_heap_bytes_on`]): the runtime keeps the texture's planes in the heap's bytes,
/// where Metal lays them out, so no placement over it dies ([`Placements`])
///
/// The runtime binds a two-plane XR10 texture to its kernels as `RGBA16Float`, a format that may
/// hold a NaN, and announces `device-nan-encoding` there: the first dispatch of this platform's
/// side dies, naming the kind ([`crate::answer::DIES`]).
#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "MoltenVK: the runtime records device-pack-conversion, a death in strict mode; Mac-only test, off everywhere; ledger: MoltenVK strict deaths"
)]
fn a_buffer_over_a_one_level_two_plane_heap_texture_sees_its_texels_as_metals() {
    let site = crate::answer::dies_of(metal2vulkan_runtime::Degradation::DeviceNanEncoding, || {
        let d = dev();
        let vulkan_desc = |t: &TextureDesc| platform().texture_descriptor(t);
        for format in [550, 551] {
            for storage in [0, 2] {
                let case = format!("{format} storage {storage}");
                // Metal's first: this platform's side dies at its first dispatch.
                let (seen, texels) =
                    two_plane_heap_bytes_on::<VulkanPlatform>(&d, &vulkan_desc, format, storage);
                let offsets = seen
                    .iter()
                    .enumerate()
                    .filter(|&(i, &b)| b == (i & 0xff) as u8);
                assert!(offsets.count() < seen.len(), "the texels land: {case}");
                reims_vgpu_core::metal_check_bytes!(format!("seen: {case}"), &seen);
                reims_vgpu_core::metal_check_bytes!(format!("read: {case}"), &texels);
            }
        }
    });
    assert_eq!(site.file, "dispatchThreads:threadsPerThreadgroup:");
}

/// A texture Metal may keep losslessly compressed placed in a heap over the guest's pages is
/// made, as Metal makes it in the same heap, and the runtime's announcement
/// (`heap-texture-lossless-layout`, its bytes there Metal's uncompressed tiles) is relayed, not a
/// death: a two-aspect `Depth32Float_Stencil8` render target, as the suite's depth and stencil
/// cases place it (core's detach heap, `Shared` over the pages, and the texture of the heap's
/// options, as core sets them whatever the guest's descriptor names)
#[test]
fn heap_textures_over_guest_pages_metal_may_compress_are_made() {
    use crate::os::OsVm;
    use reims_vgpu_core::platform::os::vm::HostVm as _;
    use reims_vgpu_core::platform::types::ResourceAddressRange;

    const SIZE: u64 = 0x10000;

    /// Whether a `Depth32Float_Stencil8` render target is made at 0 in a `Shared` heap of
    /// `SIZE` bytes over pages of their own, of the heap's options.
    fn placed<P: Platform>(p: &P, d: &P::Device) -> bool {
        // Left mapped: the heap's alias shares them until the process ends.
        let pages = OsVm::new().allocate_anywhere(SIZE).unwrap();
        let ranges = p
            .new_address_range_array(&[ResourceAddressRange {
                address: u64::try_from(pages.addr()).unwrap(),
                length: SIZE,
            }])
            .unwrap();
        let heap = d
            .new_heap(&HeapDesc {
                size: SIZE,
                heap_type: 1,
                options: vec![OptionSet::StorageMode(0), OptionSet::HazardTrackingMode(0)],
                address_ranges: Some(&ranges),
                pinned_gpu_address: None,
            })
            .unwrap()
            .heap
            .unwrap();
        let depth_stencil = p.texture_descriptor(&TextureDesc {
            pixel_format: 260,
            width: 16,
            height: 16,
            usage: 0x4,
            resource_options: Some(heap.resource_options()),
            ..TextureDesc::default()
        });
        heap.new_texture(&depth_stencil, 0).unwrap().is_some()
    }

    let (p, d) = (platform(), dev());
    crate::answer::RELAYED.lock().clear();
    assert!(placed(&p, &d));
    let relayed = crate::answer::RELAYED.lock().clone();
    let lossless = rt::Degradation::HeapTextureLosslessLayout;
    assert!(relayed.contains(&lossless), "{relayed:?}");
}

/// A fill of the bytes `range` of `buffer`, committed and waited for.
fn fill_range<P: Platform>(d: &P::Device, buffer: &P::Buffer, range: NsRange) {
    use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder as _};
    use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
    let e = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
    let fill = BlitCmd::FillBuffer {
        buffer,
        range,
        value: 0x5c,
    };
    e.encode(OnFault::Return, fill).unwrap();
    e.encode(OnFault::Return, BlitCmd::EndEncoding).unwrap();
    cb.commit(OnFault::Return).unwrap();
    cb.wait_until_completed(OnFault::Return).unwrap();
}

/// A compute pass that makes `heap` resident (`useHeaps:count:`), committed and waited for.
fn use_heap<P: Platform>(d: &P::Device, heap: &P::Heap) {
    use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
    use reims_vgpu_core::platform::compute::{ComputeCmd, ComputeEncoder as _};
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
    let e = cb.compute_encoder(OnFault::Return).unwrap().unwrap();
    let heaps = [Some(heap)];
    e.encode(OnFault::Return, ComputeCmd::UseHeaps { heaps: &heaps })
        .unwrap();
    e.encode(OnFault::Return, ComputeCmd::EndEncoding).unwrap();
    cb.commit(OnFault::Return).unwrap();
    cb.wait_until_completed(OnFault::Return).unwrap();
}

/// A buffer or texture placed over a byte of a `Depth32Float_Stencil8` render target of a heap
/// over the guest's pages, as the suite's cases place it, of the heap's options, is made while
/// the texture lives, and so is such a texture placed over a byte of a buffer that lives: the
/// runtime lays the texture out there as Metal's tiles, uncompressed where Metal may keep it
/// compressed (`heap-texture-lossless-layout`, relayed), so nothing keeps it apart. A buffer
/// beside it is made, and one over its bytes once it is let go. Each is made as Metal makes it in
/// the same heap, beside a buffer over all of the heap, as core makes with each heap
#[test]
fn placements_over_heap_depth_stencil_targets_are_made_as_metals() {
    /// Each placement on `on`'s side, which `d` makes.
    fn placements<P: Platform>(p: &P, d: &P::Device, on: &str) {
        let (heap, depth_stencil, beside) = heap_over_pages(p, d);
        let options = heap.resource_options();
        let whole = heap.new_buffer_of_length(over_pages(), options, 0).unwrap();
        assert!(whole.is_some(), "{on}: a buffer over all of the heap");
        let t = heap.new_texture(&depth_stencil, 0).unwrap();
        assert!(t.is_some(), "{on}: the texture");
        let over = heap.new_buffer_of_length(0x100, options, 0).unwrap();
        assert!(over.is_some(), "{on}: a buffer over it");
        let b = heap.new_buffer_of_length(0x100, options, beside).unwrap();
        assert!(b.is_some(), "{on}: a buffer beside it");
        let second = heap.new_texture(&depth_stencil, beside).unwrap();
        assert!(second.is_some(), "{on}: a texture over a buffer");
        drop((t, over, second));
        let after = heap.new_buffer_of_length(0x100, options, 0).unwrap();
        assert!(after.is_some(), "{on}: a buffer once it is let go");
        drop((b, whole));
    }
    placements(&platform(), &dev(), "Vulkan");
}

/// A `Depth32Float_Stencil8` render target placed in a heap over the guest's pages beside its
/// cover, core's buffer over all of it, made first, as five suite cases place one, is made: the
/// runtime lays it out there as Metal's tiles, uncompressed where Metal may keep it compressed
/// (`heap-texture-lossless-layout`, relayed), so the cover sees its bytes as Metal's
/// ([`Placements`]). So is one placed after the heap was made resident. Each is made as Metal
/// makes it, and a fill of the cover over the texture and the pass that makes the heap resident
/// run on both
#[test]
fn depth_stencil_targets_beside_the_heaps_cover_are_made_as_metals() {
    /// Each placement, fill and pass on `on`'s side, which `d` makes.
    fn beside_the_cover<P: Platform>(p: &P, d: &P::Device, on: &str) {
        let (heap, depth_stencil, _) = heap_over_pages(p, d);
        let options = heap.resource_options();
        let cover = heap.new_buffer_of_length(over_pages(), options, 0).unwrap();
        let cover = cover.unwrap();
        let t = heap.new_texture(&depth_stencil, 0).unwrap();
        assert!(t.is_some(), "{on}: the texture");
        fill_range::<P>(d, &cover, NsRange::new(0, 16));
        drop(t);
        use_heap::<P>(d, &heap);
        let t = heap.new_texture(&depth_stencil, 0).unwrap();
        assert!(t.is_some(), "{on}: the texture once the heap was resident");
        drop((t, cover));
    }
    beside_the_cover(&platform(), &dev(), "Vulkan");
}

/// A view of a texture of a heap over the guest's pages, a 4-sample `Depth16Unorm` render target
/// without `allowGPUOptimizedContents`, which the runtime lays out in the heap's bytes as Metal's
/// tiles, a shadow of them (`shadowed-texture`), at Metal's size, shares the texture's bytes as a
/// Metal view's `parentTexture` does once the texture is released: a buffer placed over them in
/// the heap is made while the view lives, and once it is let go too, as Metal makes each. A
/// texture the runtime keeps apart from the heap's other resources keeps its bytes while a view
/// of it lives ([`TextureObject::life`]); `placements_tell_the_heaps_cover_apart` holds the
/// kinds to that, as no texture of a heap over the guest's pages is kept apart on MoltenVK
#[test]
fn a_view_keeps_its_released_textures_bytes() {
    /// The heap over the guest's pages at `at`, and its 4-sample `Depth16Unorm` render target
    /// without `allowGPUOptimizedContents`.
    fn heap_and_target<P: Platform>(p: &P, d: &P::Device, at: u64) -> (P::Heap, P::Texture) {
        let (heap, ..) = heap_over_pages_at(p, d, at);
        let desc = p.texture_descriptor(&TextureDesc {
            pixel_format: 250,
            texture_type: 4,
            sample_count: 4,
            width: 16,
            height: 16,
            usage: 0x4,
            resource_options: Some(heap.resource_options()),
            allow_gpu_optimized_contents: Some(false),
            ..TextureDesc::default()
        });
        let t = heap.new_texture(&desc, 0).unwrap().unwrap();
        (heap, t)
    }
    /// The buffers over the released texture's bytes `d` makes, while its view lives and once
    /// that is let go too.
    fn over_the_view<P: Platform>(p: &P, d: &P::Device) -> [bool; 2] {
        let (heap, t) = heap_and_target(p, d, guest_pages());
        let options = heap.resource_options();
        let view = t.new_texture_view(250).unwrap();
        assert!(view.is_some(), "the view");
        drop(t);
        let over = heap.new_buffer_of_length(0x100, options, 0).unwrap();
        drop(view);
        let after = heap.new_buffer_of_length(0x100, options, 0x100).unwrap();
        [over.is_some(), after.is_some()]
    }
    assert_eq!(over_the_view(&platform(), &dev()), [true; 2]);
}

/// A 4-sample `RGBA8Unorm` render target placed in a `Shared` heap over the guest's pages, of
/// the heap's options as core gives them, where Metal keeps it losslessly compressed (16 by 16,
/// its sample grid 32 by 32, with `allowGPUOptimizedContents`), dies at the placement: the
/// runtime makes it, its image in memory of its own, the heap's bytes keeping what they held,
/// and announces `heap-texture-lossless-layout`, which the placement hears and dies of, naming
/// it. One Metal keeps uncompressed, whose samples the runtime lays out in the heap's bytes as
/// Metal's tiles, is made: 16 by 16 with `allowGPUOptimizedContents` off, and 16 by 4, whose
/// sample grid is 32 by 8. Metal makes each in the same heap. No suite case and no boot is known
/// to place one
#[test]
fn multisampled_textures_in_a_heap_over_guest_pages_die_where_metal_compresses_them() {
    use crate::os::OsVm;
    use reims_vgpu_core::platform::os::vm::HostVm as _;
    use reims_vgpu_core::platform::types::ResourceAddressRange;

    const SIZE: u64 = 0x10000;

    /// A `Shared` heap of `SIZE` bytes over pages of its own, and a 4-sample `RGBA8Unorm`
    /// render target of its options, 16 texels wide and `height` tall, its
    /// `allowGPUOptimizedContents` `allow`.
    fn heap_and_texture<P: Platform>(
        p: &P,
        d: &P::Device,
        (height, allow): (u64, Option<bool>),
    ) -> (P::Heap, P::TextureDescriptor) {
        // Left mapped: the heap's alias shares them until the process ends.
        let pages = OsVm::new().allocate_anywhere(SIZE).unwrap();
        let ranges = p
            .new_address_range_array(&[ResourceAddressRange {
                address: u64::try_from(pages.addr()).unwrap(),
                length: SIZE,
            }])
            .unwrap();
        let heap = d
            .new_heap(&HeapDesc {
                size: SIZE,
                heap_type: 1,
                options: vec![OptionSet::StorageMode(0), OptionSet::HazardTrackingMode(0)],
                address_ranges: Some(&ranges),
                pinned_gpu_address: None,
            })
            .unwrap()
            .heap
            .unwrap();
        let multisample = p.texture_descriptor(&TextureDesc {
            pixel_format: 70,
            texture_type: 4,
            sample_count: 4,
            width: 16,
            height,
            usage: 0x4,
            resource_options: Some(heap.resource_options()),
            allow_gpu_optimized_contents: allow,
            ..TextureDesc::default()
        });
        (heap, multisample)
    }

    let compressed = (16, None);
    let uncompressed = [(16, Some(false)), (4, None)];
    let (p, d) = (platform(), dev());
    let (heap, multisample) = heap_and_texture(&p, &d, compressed);
    *crate::answer::LAST_LINE.lock() = None;
    let site = catch_death(|| heap.new_texture(&multisample, 0).is_ok()).unwrap_err();
    let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
    assert_eq!(
        (site.kind, site.file),
        (DeathKind::Abort, "newTextureWithDescriptor:offset:"),
        "{line}"
    );
    assert!(line.contains("heap-texture-lossless-layout"), "{line}");
    for shape in uncompressed {
        let (heap, multisample) = heap_and_texture(&p, &d, shape);
        *crate::answer::LAST_LINE.lock() = None;
        let made = catch_death(|| heap.new_texture(&multisample, 0).unwrap().is_some());
        let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
        assert!(matches!(made, Ok(true)), "{shape:?}: {line}");
    }
}

/// A depth or stencil texture without allowGPUOptimizedContents placed in a heap, which the
/// runtime sizes as Metal does (`heap_depth_stencil_sizes_with_optimized_contents_off`). In a
/// heap over no address ranges, core's legacy heap, one is made, laid out in Metal's tiles on
/// MoltenVK, in the driver's layout on any other driver, which the runtime announces
/// (`legacy_heap_textures_announce_the_runtimes_layouts`): a `Depth32Float` one and the suite's (a
/// 2D `Depth32Float_Stencil8` render target, 8x8, as `render_state_depth` and four more cases
/// place it there), one with the property on, and a 64x64 one at 0x4000 of the 0x8000 heap,
/// where Metal's 16384 bytes fit. The suite's beside the first, where Metal's
/// size (16512 bytes, as the runtime's) runs past the heap's end, is nil, as Metal's; so are a
/// 128x128 one at 0 (Metal's 65536 bytes), one off Metal's alignment (a 16x16 one at 0x40) and
/// one past the heap's end, as Metal answers in a heap of its own. A buffer over
/// all of the heap, as core makes with each heap, is made first. In a `Shared` heap over the
/// guest's pages, as core's detach heap is, with no such buffer, a `Depth32Float` one is made,
/// and so are a buffer and a texture over a byte of it while it lives, and such a texture over a
/// byte of a buffer that lives. Metal makes each made here or dying here
#[test]
fn placements_of_depth_textures_without_optimized_contents_take_metals_size() {
    let texture_sel = "newTextureWithDescriptor:offset:";
    /// A `Depth32Float` render target, `side` by `side`, of `options`.
    fn depth_in<P: Platform>(
        p: &P,
        side: u64,
        allow: Option<bool>,
        options: u64,
    ) -> P::TextureDescriptor {
        p.texture_descriptor(&TextureDesc {
            pixel_format: 252,
            width: side,
            height: side,
            usage: 0x4,
            resource_options: Some(options),
            allow_gpu_optimized_contents: allow,
            ..TextureDesc::default()
        })
    }
    fn depth_of<P: Platform>(p: &P, side: u64, allow: Option<bool>) -> P::TextureDescriptor {
        depth_in(p, side, allow, 0x120)
    }
    /// The suite's: a `Depth32Float_Stencil8` render target, 8x8, as its guests' records
    /// describe it, which serialize allowGPUOptimizedContents off.
    fn suites_of<P: Platform>(p: &P, options: u64) -> P::TextureDescriptor {
        p.texture_descriptor(&TextureDesc {
            pixel_format: 260,
            width: 8,
            height: 8,
            usage: 0x4,
            resource_options: Some(options),
            allow_gpu_optimized_contents: Some(false),
            ..TextureDesc::default()
        })
    }
    /// The offset past a texture of `size` bytes at offset 0, at its alignment `align` and a
    /// buffer's 0x100, where a buffer of 0x100 bytes is placed; and the first offset at its
    /// alignment where the texture covers that buffer's first byte.
    fn beside_and_over(size: u64, align: u64) -> (u64, u64) {
        let beside = size.next_multiple_of(align.max(0x100));
        (beside, (beside + 1 - size).next_multiple_of(align))
    }
    /// In `heap`, over the guest's pages, a `Depth32Float` one of `size` at 0, then a buffer
    /// over its last bytes, a texture over it, a buffer beside it and, once it is let go, another
    /// such texture over that buffer: `placed` takes each of those placements, by its case and
    /// selector, and whether it was made.
    type Placed<'a> = &'a dyn Fn(&str, &'static str, &mut dyn FnMut() -> bool);
    fn over_pages<P: Platform>(p: &P, heap: &P::Heap, size: SizeAndAlign, placed: Placed<'_>) {
        let buffer_sel = "newBufferWithLength:options:offset:";
        let texture_sel = "newTextureWithDescriptor:offset:";
        let options = heap.resource_options();
        let depth = depth_in(p, 16, Some(false), options);
        let (beside, over) = beside_and_over(size.size, size.align);
        let last = (size.size - 1) & !0xff;
        let t = heap.new_texture(&depth, 0).unwrap();
        assert!(t.is_some(), "the texture");
        placed("a buffer over its last bytes", buffer_sel, &mut || {
            (heap.new_buffer_of_length(0x100, options, last).unwrap()).is_some()
        });
        let linear = p.texture_descriptor(&TextureDesc {
            pixel_format: 80,
            width: 16,
            height: 16,
            usage: 0x13,
            resource_options: Some(options),
            ..TextureDesc::default()
        });
        placed("a texture over it", texture_sel, &mut || {
            (heap.new_texture(&linear, 0).unwrap()).is_some()
        });
        let b = heap.new_buffer_of_length(0x100, options, beside).unwrap();
        assert!(b.is_some(), "a buffer beside it");
        drop(t);
        placed("over a buffer that lives", texture_sel, &mut || {
            (heap.new_texture(&depth, over).unwrap()).is_some()
        });
        drop(b);
    }
    let depth = |side, allow| depth_of(&platform(), side, allow);
    let suites = suites_of(&platform(), 0x120);
    let d = dev();
    let raw = suites.fields().descriptor(texture_sel, Maker::SizeQuery);
    let answer = plain(|| d.raw.heap_texture_size_and_align(&raw)).unwrap();
    let (beside, _) = beside_and_over(answer.size, answer.align);
    let h = private_heap();
    let whole = h.new_buffer_of_length(0x8000, 0x120, 0).unwrap().unwrap();
    for (case, desc, at) in [
        ("a Depth32Float one", depth(16, Some(false)), 0),
        ("the suite's", suites, 0),
        ("with the property on", depth(16, Some(true)), beside),
        ("a 64x64 one at 0x4000", depth(64, Some(false)), 0x4000),
    ] {
        assert!(h.new_texture(&desc, at).unwrap().is_some(), "{case}");
    }
    let suites = suites_of(&platform(), 0x120);
    assert!(
        (h.new_texture(&suites, beside).unwrap()).is_none(),
        "nil where Metal's size runs past the heap's end"
    );
    assert!(
        (h.new_texture(&depth(128, Some(false)), 0).unwrap()).is_none(),
        "nil inside the heap where Metal's size does not fit either"
    );
    assert!(
        (h.new_texture(&depth(16, Some(false)), 0x40).unwrap()).is_none(),
        "nil off Metal's alignment"
    );
    assert!(
        h.new_texture(&depth(16, Some(false)), 0x8000)
            .unwrap()
            .is_none()
    );
    drop(whole);
    let p = platform();
    let (shared, ..) = heap_over_pages(&p, &d);
    let desc = depth_in(&p, 16, Some(false), shared.resource_options());
    let raw = desc.fields().descriptor(texture_sel, Maker::SizeQuery);
    let size = plain(|| d.raw.heap_texture_size_and_align(&raw)).unwrap();
    let size = SizeAndAlign {
        size: size.size,
        align: size.align,
    };
    over_pages(&p, &shared, size, &|case, _, made| {
        assert!(made(), "{case}");
    });
}

/// An optimized `X32_Stencil8` render target (`optimized-depth-stencil-texture`) placed in a
/// heap over the guest's pages is made, as a `Depth32Float` one is: its size and alignment are
/// the runtime's model of Metal's, a `Stencil8` one's of its shape, which the runtime's own
/// answer covers. Metal makes each
#[test]
fn optimized_x32_stencil8_placements_are_made_as_metals() {
    let sel = "newTextureWithDescriptor:offset:";
    /// A render target of `pixel_format`, 16x16, of `heap`'s options, with
    /// allowGPUOptimizedContents off.
    fn target<P: Platform>(p: &P, heap: &P::Heap, pixel_format: u64) -> P::TextureDescriptor {
        p.texture_descriptor(&TextureDesc {
            pixel_format,
            width: 16,
            height: 16,
            usage: 0x4,
            resource_options: Some(heap.resource_options()),
            allow_gpu_optimized_contents: Some(false),
            ..TextureDesc::default()
        })
    }
    let (p, d) = (platform(), dev());
    for pixel_format in [252, 261] {
        let (heap, ..) = heap_over_pages(&p, &d);
        let desc = target(&p, &heap, pixel_format);
        *crate::answer::LAST_LINE.lock() = None;
        let made = catch_death(|| heap.new_texture(&desc, 0));
        let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
        let made = made.unwrap_or_else(|site| panic!("{pixel_format}: died at {site:?}: {line}"));
        assert!(made.unwrap().is_some(), "{pixel_format} ({sel})");
    }
}

/// A placement heap's buffer of length 0 is made at an offset of 256 bytes' alignment up to
/// the heap's end, of no bytes, at the offset into the heap; nil off the alignment and past
/// the end, as Metal's
#[test]
fn zero_length_heap_buffers_are_metals() {
    fn made<P: Platform>(d: &P::Device) -> Vec<Option<(u64, u64)>>
    where
        P::Device: GpuDevice<P>,
    {
        let h = heap_of::<P>(d, 0x10000, &[OptionSet::StorageMode(0)])
            .heap
            .unwrap();
        [0, 0x100, 0xff00, 0x10000, 0x10100, 100]
            .into_iter()
            .map(|offset| {
                let b = h.new_buffer_of_length(0, h.resource_options(), offset);
                b.unwrap()
                    .map(|b| (b.length(), b.gpu_address() - h.gpu_address()))
            })
            .collect()
    }
    let measured = [
        Some((0, 0)),
        Some((0, 0x100)),
        Some((0, 0xff00)),
        Some((0, 0x10000)),
        None,
        None,
    ];
    assert_eq!(made::<VulkanPlatform>(&dev()), measured);
}

/// A 2D BGRA8Unorm descriptor of 16 x 16, Private, of usage `usage` (raw).
fn of_usage(usage: u64) -> TextureDesc {
    TextureDesc {
        pixel_format: 80,
        width: 16,
        height: 16,
        usage,
        resource_options: Some(0x20),
        ..TextureDesc::default()
    }
}

/// The usage each creator's texture of `desc` answers on `d`: `newTextureWithDescriptor:`'s,
/// a view's of it, `newSharedTextureWithDescriptor:`'s, a texture's opened from that one's
/// handle, `newTextureWithDescriptor:iosurface:plane:`'s (plane 0 of `s`, whose storage
/// `desc_on` gives), a Shared buffer's two creators' (of `desc_on` too) and a Private
/// placement heap's; a swizzled view's of the first (blue and red swapped); and the heap's
/// size query.
fn usages<P: Platform<Surface = OsSurface>>(
    d: &P::Device,
    (desc, desc_on): (&P::TextureDescriptor, &P::TextureDescriptor),
    s: &OsSurface,
) -> ([u64; 8], u64, SizeAndAlign) {
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
    let heap = heap_of::<P>(d, 1 << 20, &[OptionSet::StorageMode(2)])
        .heap
        .unwrap();
    let plain = d.new_texture(OnFault::Return, desc).unwrap().unwrap();
    let shared = d.new_shared_texture(desc).unwrap().unwrap();
    let handle = shared.new_shared_texture_handle().unwrap().unwrap();
    let made = [
        plain.usage(),
        plain.new_texture_view(80).unwrap().unwrap().usage(),
        shared.usage(),
        d.open_shared_texture(Some(&handle), 0)
            .unwrap()
            .unwrap()
            .usage(),
        d.new_texture_on_surface(desc_on, Some(s), 0)
            .unwrap()
            .unwrap()
            .usage(),
        buffer
            .new_texture(OnFault::Return, desc_on, 0, 256)
            .unwrap()
            .unwrap()
            .usage(),
        buffer
            .new_linear_texture(desc_on, 0, 256, 4096)
            .unwrap()
            .unwrap()
            .usage(),
        heap.new_texture(desc, 0).unwrap().unwrap().usage(),
    ];
    // `MTLTextureSwizzleBlue`, `Green`, `Red`, `Alpha`.
    let swapped = Swizzle {
        red: 4,
        green: 3,
        blue: 2,
        alpha: 5,
    };
    let one = NsRange::new(0, 1);
    let swizzled = plain
        .new_swizzled_texture_subview(80, 2, one, one, swapped)
        .unwrap()
        .unwrap()
        .usage();
    (made, swizzled, d.heap_texture_size_and_align(desc).unwrap())
}

/// Every creator takes the usage bits Metal's public headers name none of (0x8, 0x4000,
/// 0x8000 and 0x10000, which a macOS 26.6.2 guest sets), making the texture as without them;
/// the texture, its views and a texture opened from its handle answer them but 0x8, a
/// swizzled view as Metal's does, and the heap's size query its size without them, as
/// Metal's (measured: [`crate::gpu::desc`])
#[test]
fn usage_bits_metal_takes_are_answered_as_metals() {
    let d = dev();
    let p = platform();
    let s = bgra_surface(16, 16);
    let usage_of_all = |u: u64| [u & !0x8; 8];
    for usage in [0x1_0005, 0x1_0015, 0x4005, 0x8005, 0xd, 0x1_c01d] {
        let (desc, desc_on) = (of_usage(usage), of_usage(usage));
        let desc_on = TextureDesc {
            resource_options: Some(0),
            ..desc_on
        };
        let ours = usages::<VulkanPlatform>(
            &d,
            (
                &p.texture_descriptor(&desc),
                &p.texture_descriptor(&desc_on),
            ),
            &s,
        );
        assert_eq!(ours.0, usage_of_all(usage), "{usage:#x}");
        let without = d
            .heap_texture_size_and_align(&p.texture_descriptor(&of_usage(usage & 0x37)))
            .unwrap();
        assert_eq!(ours.2, without, "{usage:#x}");
        reims_vgpu_core::metal_check!(format!("{usage:#x}"), ours);
    }
}

/// The heap's size query answers a texture of `BlockWritesOnly` usage (0x10000) as Metal's
/// does, the size and alignment of one without the bit (1152 bytes for a 16 by 16
/// `BGRA8Unorm` texture of usage 0x10005 in `Private` storage, measured on the M3 Max), and
/// such a texture fits where a guest places it by that answer: a placement heap of the
/// answer's size makes it at offset 0. Of sizes, levels and array lengths here, in
/// `RGBA8Unorm`, `BGRA8Unorm` and `RGBA16Float`. A mipmapped one Metal keeps losslessly
/// compressed, of no `ShaderWrite` usage, answers Metal's as the runtime models it, with the
/// bit and without, and nothing is announced. Metal answers every texture here alike with the bit
/// and without
#[test]
fn block_writes_only_sizes_are_metals_and_fit() {
    let (d, p) = (dev(), platform());
    let options = [OptionSet::StorageMode(2), OptionSet::HazardTrackingMode(0)];
    crate::answer::RELAYED.lock().clear();
    for (format, (width, height), levels, slices) in [
        (80, (16, 16), 1, 1),
        (80, (100, 60), 7, 1),
        (115, (256, 256), 9, 1),
        (80, (64, 64), 1, 4),
        (70, (33, 17), 6, 3),
    ] {
        for usage in [0x1_0005, 0x1_0001, 0x1_0003] {
            let desc = TextureDesc {
                texture_type: if slices > 1 { 3 } else { 2 },
                pixel_format: format,
                width,
                height,
                mipmap_level_count: levels,
                array_length: slices,
                ..of_usage(usage)
            };
            let case = format!("{format} {width}x{height} {levels} {slices} {usage:#x}");
            let ours = d.heap_texture_size_and_align(&p.texture_descriptor(&desc));
            let without = TextureDesc {
                usage: usage & !0x1_0000,
                ..desc
            };
            let bare = d.heap_texture_size_and_align(&p.texture_descriptor(&without));
            let ours = ours.unwrap();
            assert_eq!(ours, bare.unwrap(), "{case}");
            if (format, width, levels, slices, usage) == (80, 16, 1, 1, 0x1_0005) {
                assert_eq!(ours.size, 1152, "{case}");
            }
            let heap = heap_of::<VulkanPlatform>(&d, ours.size, &options)
                .heap
                .unwrap();
            let made = heap.new_texture(&p.texture_descriptor(&desc), 0).unwrap();
            assert!(made.is_some(), "{case}");
            reims_vgpu_core::metal_check!(case, ours);
            reims_vgpu_core::metal_check!(format!("{case} without the bit"), ours);
        }
    }
    let unmeasured = rt::Degradation::UnmeasuredHeapTextureSize;
    assert!(!crate::answer::RELAYED.lock().contains(&unmeasured));
}

/// A PVRTC texture's size and alignment in Metal's heaps are Metal's: the runtime models
/// PVRTC's layout there. `newTextureWithDescriptor:`
/// of the suite's texture of each PVRTC format (`textures_formats`: 2D, 16x16, one level,
/// `Private`, `ShaderRead`, `allowGPUOptimizedContents` off) is made, and the size query
/// answers Metal's; its placement in a heap is made, one at the heap's end is nil, and a 32x32
/// one is made, as on Metal. A texture of 8 samples dies at the runtime's refusal (`SampleCount`),
/// naming no kind, where Metal stops it (the runtime's measurement)
#[test]
fn pvrtc_heap_sizes_are_metals() {
    let (sel, query) = (
        "newTextureWithDescriptor:",
        "heapTextureSizeAndAlignWithDescriptor:",
    );
    let (p, d) = (platform(), dev());
    let options = [OptionSet::StorageMode(2), OptionSet::HazardTrackingMode(0)];
    let suites = |pixel_format| TextureDesc {
        pixel_format,
        width: 16,
        height: 16,
        usage: 0x1,
        resource_options: Some(0x20),
        allow_gpu_optimized_contents: Some(false),
        ..TextureDesc::default()
    };
    let died = |at: &'static str, name: &str, f: &dyn Fn()| {
        *crate::answer::LAST_LINE.lock() = None;
        let site = catch_death(f).unwrap_err();
        let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
        assert_eq!(
            (site.kind, site.file),
            (DeathKind::Abort, at),
            "{name}: {line}"
        );
        line
    };
    let wider = |t: TextureDesc| TextureDesc {
        width: 32,
        height: 32,
        ..t
    };
    for pixel_format in 160..=167 {
        let name = format!("PVRTC {pixel_format}");
        for t in [suites(pixel_format), wider(suites(pixel_format))] {
            let desc = p.texture_descriptor(&t);
            let made = d.new_texture(OnFault::Return, &desc).unwrap();
            assert!(made.is_some(), "{name}");
            let ours = d.heap_texture_size_and_align(&desc).unwrap();
            let key = format!("{name} {}x{}", t.width, t.height);
            reims_vgpu_core::metal_check!(key, (ours.size, ours.align));
        }
    }
    let t = suites(166);
    let heap = heap_of::<VulkanPlatform>(&d, 0x8000, &options)
        .heap
        .unwrap();
    let desc = p.texture_descriptor(&t);
    let placed = (
        heap.new_texture(&desc, 0).unwrap().is_some(),
        heap.new_texture(&desc, 0x8000).unwrap().is_some(),
    );
    assert_eq!(placed, (true, false), "the placement, the end");
    let eight = p.texture_descriptor(&TextureDesc {
        texture_type: 4,
        pixel_format: 70,
        width: 16,
        height: 16,
        sample_count: 8,
        usage: 0x4,
        resource_options: Some(0x20),
        ..TextureDesc::default()
    });
    let made = || drop(d.new_texture(OnFault::Return, &eight));
    let sized = || drop(d.heap_texture_size_and_align(&eight));
    for (at, f) in [(sel, &made as &dyn Fn()), (query, &sized)] {
        let line = died(at, "8 samples", f);
        assert!(line.contains("SampleCount"), "8 samples at {at}: {line}");
        assert!(!line.contains("unmeasured-heap-texture-size"), "{line}");
    }
}

/// A descriptor takes any usage bit at `setUsage:`, as Metal's does, and drops it at the next;
/// every creator, and the heap's size query, of one Metal does not define dies there, as Metal
/// asserts (measured: [`crate::gpu::desc`]), but `newSharedTextureWithDescriptor:` of a format Metal makes no texture of,
/// which answers nil first, as Metal's does; once `setUsage:` drops the bit, every creator
/// makes the texture
#[test]
fn usage_bits_metal_does_not_define_die_where_textures_are_made() {
    let d = dev();
    let p = platform();
    let s = bgra_surface(16, 16);
    let heap = private_heap();
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
    for usage in [0x45, 0x2_0005, 0x8000_0000_0000_0001] {
        let desc = p.texture_descriptor(&of_usage(5));
        let desc_on = p.texture_descriptor(&TextureDesc {
            resource_options: Some(0),
            ..of_usage(5)
        });
        desc.set_usage(usage);
        desc_on.set_usage(usage);
        let deaths = [
            (
                "newTextureWithDescriptor:",
                catch_death(|| d.new_texture(OnFault::Return, &desc)).map(drop),
            ),
            (
                "newSharedTextureWithDescriptor:",
                catch_death(|| d.new_shared_texture(&desc)).map(drop),
            ),
            (
                "heapTextureSizeAndAlignWithDescriptor:",
                catch_death(|| d.heap_texture_size_and_align(&desc)).map(drop),
            ),
            (
                "newTextureWithDescriptor:iosurface:plane:",
                catch_death(|| d.new_texture_on_surface(&desc_on, Some(&s), 0)).map(drop),
            ),
            (
                "newTextureWithDescriptor:offset:bytesPerRow:",
                catch_death(|| buffer.new_texture(OnFault::Return, &desc_on, 0, 256)).map(drop),
            ),
            (
                "newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:",
                catch_death(|| buffer.new_linear_texture(&desc_on, 0, 256, 4096)).map(drop),
            ),
            (
                "newTextureWithDescriptor:offset:",
                catch_death(|| heap.new_texture(&desc, 0)).map(drop),
            ),
        ];
        for (sel, death) in deaths {
            let site = death.unwrap_err();
            assert_eq!(
                (site.kind, site.file),
                (DeathKind::Abort, sel),
                "{usage:#x}"
            );
        }
        for format in [263, 556, 651, 652] {
            let none = TextureDesc {
                pixel_format: format,
                ..of_usage(5)
            };
            let ours = p.texture_descriptor(&none);
            ours.set_usage(usage);
            let shared = d.new_shared_texture(&ours).unwrap();
            assert!(shared.is_none(), "{format} {usage:#x}");
        }
        desc.set_usage(5);
        desc_on.set_usage(5);
        let made = [
            d.new_texture(OnFault::Return, &desc).unwrap().is_some(),
            d.new_shared_texture(&desc).unwrap().is_some(),
            d.heap_texture_size_and_align(&desc).unwrap().size > 0,
            d.new_texture_on_surface(&desc_on, Some(&s), 0)
                .unwrap()
                .is_some(),
            buffer
                .new_texture(OnFault::Return, &desc_on, 0, 256)
                .unwrap()
                .is_some(),
            buffer
                .new_linear_texture(&desc_on, 0, 256, 4096)
                .unwrap()
                .is_some(),
            heap.new_texture(&desc, 0).unwrap().is_some(),
        ];
        assert_eq!(made, [true; 7], "{usage:#x}");
    }
}

/// The `uniqueIdentifier` of each texture of a sequence of creations on platform `P`, with
/// what each step made: textures of the device's creator, a view, a heap's, a buffer's, over a
/// surface and shareable, beside a buffer, a heap, a sampler state, a command queue, a nil
/// texture and a release, then, after another release, one on another thread and one on this
/// thread again.
fn texture_identifiers_on<P: Platform<Surface = OsSurface>>(
    d: &P::Device,
    desc: &dyn Fn(&TextureDesc) -> P::TextureDescriptor,
) -> Vec<(&'static str, u64, u64)>
where
    P::Texture: Send,
    P::Device: Sync,
{
    use reims_vgpu_core::platform::resource::{Buffer as _, Heap as _, Texture as _};
    let of = |format, options| TextureDesc {
        pixel_format: format,
        width: 4,
        height: 4,
        usage: 1,
        resource_options: Some(options),
        ..TextureDesc::default()
    };
    let private = desc(&of(80, 0x20));
    let shared = desc(&of(80, 0));
    let texture = || d.new_texture(OnFault::Return, &private).unwrap().unwrap();
    let mut ids = Vec::new();
    let t1 = texture();
    ids.push(("t1", t1.unique_identifier(), t1.gpu_resource_id()));
    let t2 = texture();
    ids.push(("t2", t2.unique_identifier(), t2.gpu_resource_id()));
    let buffer = d
        .new_buffer_of_length(OnFault::Return, 4096, 0)
        .unwrap()
        .unwrap();
    let t3 = texture();
    ids.push((
        "t3, after a buffer",
        t3.unique_identifier(),
        t3.gpu_resource_id(),
    ));
    let view = t1.new_texture_view(81).unwrap().unwrap();
    ids.push((
        "a view of t1",
        view.unique_identifier(),
        view.gpu_resource_id(),
    ));
    let t4 = texture();
    ids.push((
        "t4, after the view",
        t4.unique_identifier(),
        t4.gpu_resource_id(),
    ));
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
    let t5 = texture();
    ids.push((
        "t5, after a heap",
        t5.unique_identifier(),
        t5.gpu_resource_id(),
    ));
    let placed = heap.new_texture(&private, 0).unwrap().unwrap();
    ids.push((
        "the heap's texture",
        placed.unique_identifier(),
        placed.gpu_resource_id(),
    ));
    let over = buffer
        .new_texture(OnFault::Return, &shared, 0, 256)
        .unwrap()
        .unwrap();
    ids.push((
        "a texture over the buffer",
        over.unique_identifier(),
        over.gpu_resource_id(),
    ));
    let surface = bgra_surface(4, 4);
    let on = d
        .new_texture_on_surface(&shared, Some(&surface), 0)
        .unwrap()
        .unwrap();
    ids.push((
        "a texture over a surface",
        on.unique_identifier(),
        on.gpu_resource_id(),
    ));
    let sharable = d.new_shared_texture(&private).unwrap().unwrap();
    ids.push((
        "a shareable texture",
        sharable.unique_identifier(),
        sharable.gpu_resource_id(),
    ));
    let _sampler = d.new_sampler_state(&SamplerDesc::default()).unwrap();
    let t6 = texture();
    ids.push((
        "t6, after a sampler state",
        t6.unique_identifier(),
        t6.gpu_resource_id(),
    ));
    drop(t2);
    let t7 = texture();
    ids.push((
        "t7, after t2's release",
        t7.unique_identifier(),
        t7.gpu_resource_id(),
    ));
    assert!(
        d.new_texture(OnFault::Return, &desc(&of(263, 0x20)))
            .unwrap()
            .is_none()
    );
    let t8 = texture();
    ids.push((
        "t8, after a nil texture",
        t8.unique_identifier(),
        t8.gpu_resource_id(),
    ));
    let _queue = d.new_command_queue().unwrap();
    let t9 = texture();
    ids.push((
        "t9, after a command queue",
        t9.unique_identifier(),
        t9.gpu_resource_id(),
    ));
    // A release first, so the lowest free ID is not the next one.
    drop(t8);
    let t10 = std::thread::scope(|s| s.spawn(texture).join().unwrap());
    ids.push((
        "t10, on another thread, after t8's release",
        t10.unique_identifier(),
        t10.gpu_resource_id(),
    ));
    let t11 = texture();
    ids.push((
        "t11, on this thread again",
        t11.unique_identifier(),
        t11.gpu_resource_id(),
    ));
    ids.push(("t1 again", t1.unique_identifier(), t1.gpu_resource_id()));
    ids.push((
        "the view again",
        view.unique_identifier(),
        view.gpu_resource_id(),
    ));
    ids
}

/// Each texture's `uniqueIdentifier` and `gpuResourceID` on `d`, over releases in several
/// orders, a release while a command buffer holds the texture, a view's release, and textures
/// placed at an index and forced to one.
fn texture_reuse_on<P: Platform>(
    d: &P::Device,
    desc: &dyn Fn(&TextureDesc) -> P::TextureDescriptor,
) -> Vec<(String, u64, u64)> {
    use reims_vgpu_core::platform::blit::BlitCmd;
    use reims_vgpu_core::platform::blit::BlitEncoder as _;
    use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
    use reims_vgpu_core::platform::resource::Texture as _;
    use reims_vgpu_core::platform::types::{ClearColor, MtlRegion};
    let private = desc(&TextureDesc {
        pixel_format: 80,
        width: 4,
        height: 4,
        usage: 1,
        resource_options: Some(0x20),
        ..TextureDesc::default()
    });
    let texture = || d.new_texture(OnFault::Return, &private).unwrap().unwrap();
    let mut ids = Vec::new();
    let mut note = |what: &str, t: &P::Texture| {
        ids.push((what.to_string(), t.unique_identifier(), t.gpu_resource_id()));
    };
    let mut ts: Vec<Option<P::Texture>> = (0..6).map(|_| Some(texture())).collect();
    for (i, t) in ts.iter().enumerate() {
        note(&format!("t{i}"), t.as_ref().unwrap());
    }
    ts[1] = None;
    ts[3] = None;
    let a = texture();
    note("after releasing t1 then t3", &a);
    let b = texture();
    note("and another", &b);
    ts[4] = None;
    ts[2] = None;
    let c = texture();
    note("after releasing t4 then t2", &c);
    let e = texture();
    note("and another", &e);
    let f = texture();
    note("and a third", &f);
    ts[5] = None;
    let g = texture();
    note("after releasing t5, the highest", &g);
    let q = d.new_command_queue().unwrap();
    let held = texture();
    note("held by a command buffer", &held);
    let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
    let en = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
    for _ in 0..2000 {
        en.encode(
            OnFault::Return,
            BlitCmd::FillTextureColor {
                texture: &held,
                level: 0,
                slice: 0,
                region: MtlRegion {
                    origin: reims_vgpu_core::platform::types::MtlOrigin { x: 0, y: 0, z: 0 },
                    size: reims_vgpu_core::platform::types::MtlSize {
                        width: 4,
                        height: 4,
                        depth: 1,
                    },
                },
                color: ClearColor::default(),
                pixel_format: 80,
            },
        )
        .unwrap();
    }
    en.encode(OnFault::Return, BlitCmd::EndEncoding).unwrap();
    cb.commit(OnFault::Return).unwrap();
    drop(held);
    let h = texture();
    note("after releasing it in flight", &h);
    cb.wait_until_completed(OnFault::Return).unwrap();
    let_go::<P>(&q);
    let i = texture();
    note("after the command buffer completed", &i);
    drop(cb);
    let j = texture();
    note("after the command buffer's release", &j);
    let view = a.new_texture_view(81).unwrap().unwrap();
    note("a view", &view);
    drop(view);
    let k = texture();
    note("after the view's release", &k);
    let placed = |index| {
        let desc = desc(&TextureDesc {
            pixel_format: 80,
            width: 4,
            height: 4,
            usage: 1,
            resource_options: Some(0x20),
            resource_index: Some(index),
            ..TextureDesc::default()
        });
        d.new_texture(OnFault::Return, &desc).unwrap().unwrap()
    };
    let p1 = placed((100, false));
    note("placed at 100", &p1);
    let p2 = placed((200, true));
    note("placed at 200, forced", &p2);
    let l = texture();
    note("after the placed ones", &l);
    drop(p1);
    let m = texture();
    note("after releasing the one at 100", &m);
    ids
}

/// A texture's `uniqueIdentifier` is its `gpuResourceID`, and both are Metal's, from a
/// process's start, through every creator's texture (the device's, a view, a heap's, one over
/// a buffer and one over a surface, a shareable one), beside a buffer, a heap, a sampler
/// state, a command queue and a nil texture, after a release, and on another thread after a
/// release, where the lowest free ID is not the next one, and on this thread again. Metal's
/// side runs first, in a process where nothing else has used Metal's device yet
/// (`texture_identifiers_are_reused_as_metals` says why)
#[test]
fn texture_identifiers_are_metals() {
    let p = crate::platform::tests::platform();
    let ours = texture_identifiers_on::<VulkanPlatform>(&dev(), &|t| p.texture_descriptor(t));
    for (what, unique, id) in &ours {
        assert_eq!(unique, id, "{what}");
    }
    reims_vgpu_core::metal_check!("identifiers", ours);
}

/// Texture identifiers are reused as Metal reuses them, from a process's start: the lowest
/// free first, over releases in several orders; one released while a command buffer holds the
/// texture once it completed; a view's at its release; and a texture placed at an index or
/// forced to one takes it, the next plain one going on below. Metal's answers were recorded in
/// a process where nothing else had used Metal's device yet: there a texture a command buffer
/// held is free when
/// `waitUntilCompleted` returns (measured on the M3 Max: 5 runs of 5, with and without an
/// autorelease pool over the encoding). Once this platform's run of the sequence has put
/// MoltenVK's work on Metal's device, Metal frees it later, a race of its own: still held
/// after the wait in 3 runs of 3 (after the command buffer's release too in two of them), and
/// free 300 ms after the wait in 2 runs of 2. Metal lets go of what a completed command buffer
/// retained after its waiters returned, so under load it frees it late alone too: the IDs after
/// the completion are taken once the queue let go of it ([`let_go`])
#[test]
fn texture_identifiers_are_reused_as_metals() {
    let p = crate::platform::tests::platform();
    let ours = texture_reuse_on::<VulkanPlatform>(&dev(), &|t| p.texture_descriptor(t));
    for (what, unique, id) in &ours {
        assert_eq!(unique, id, "{what}");
    }
    reims_vgpu_core::metal_check!("identifiers", ours);
}

/// Each texture's `uniqueIdentifier` and `gpuResourceID` on `d` where Metal hands out an ID a
/// live texture holds a second time: after the device's first command
/// queue and a reservation of textures 5 and 40, sixteen textures (IDs 1 to 4 twice first,
/// measured here: `gpu::device`'s `a_texture_takes_a_live_ones_id_where_metal_gives_it`), then,
/// each followed by new textures, the release of one of an ID's two textures, of the other,
/// of both of another ID's in turn, and of two IDs' textures alternately.
fn texture_reissue_on<P: Platform>(
    d: &P::Device,
    desc: &dyn Fn(&TextureDesc) -> P::TextureDescriptor,
) -> Vec<(String, u64, u64)> {
    use reims_vgpu_core::platform::resource::Texture as _;
    let private = desc(&TextureDesc {
        pixel_format: 80,
        width: 4,
        height: 4,
        usage: 1,
        resource_options: Some(0x20),
        ..TextureDesc::default()
    });
    let _queue = d.new_command_queue().unwrap();
    d.reserve_resource_indices(0, &[5, 40]).unwrap();
    let texture = || d.new_texture(OnFault::Return, &private).unwrap().unwrap();
    let mut ids = Vec::new();
    let mut note = |what: String, t: &P::Texture| {
        ids.push((what, t.unique_identifier(), t.gpu_resource_id()));
    };
    let mut ts: Vec<Option<P::Texture>> = (0..16).map(|_| Some(texture())).collect();
    for (i, t) in ts.iter().enumerate() {
        note(format!("t{i}"), t.as_ref().unwrap());
    }
    let mut later = Vec::new();
    for (step, released, made) in [
        ("t1, one of ID 2's two", &[1][..], 1),
        ("t5, the other", &[5][..], 2),
        ("t2 then t6, both of ID 3's", &[2, 6][..], 2),
        (
            "t0, t3, t4 and t7, IDs 1 and 4 alternately",
            &[0, 3, 4, 7][..],
            4,
        ),
    ] {
        for &i in released {
            ts[i] = None;
        }
        for n in 0..made {
            let t = texture();
            note(format!("after releasing {step}: {n}"), &t);
            later.push(t);
        }
    }
    for n in 0..4 {
        let t = texture();
        note(format!("then {n}"), &t);
        later.push(t);
    }
    std::mem::forget((ts, later));
    ids
}

/// A texture takes an ID a live texture holds where Metal hands it out a second time, and
/// IDs come back as Metal's pool gives them back, over releases of either or both of an ID's
/// textures in several orders, from a process's start; each texture's
/// `uniqueIdentifier` is its ID throughout, as Metal's, recorded before MoltenVK made objects of
/// its own on Metal's device
#[test]
fn texture_identifiers_are_reissued_as_metals() {
    let p = crate::platform::tests::platform();
    let ours = texture_reissue_on::<VulkanPlatform>(&dev(), &|t| p.texture_descriptor(t));
    for (what, unique, id) in &ours {
        assert_eq!(unique, id, "{what}");
    }
    reims_vgpu_core::metal_check!("identifiers", ours);
}

/// A format's block: its bytes, and its width and height in texels.
type FormatBlock = (u64, (u64, u64));

/// A texture's shape, placed in a heap over no address ranges by [`legacy_heap_bytes_on`].
#[derive(Clone, Copy, Debug)]
struct LegacyShape {
    format: u64,
    /// A block's bytes, a texel's for a format not of blocks.
    block_bytes: u64,
    /// A block's width and height in texels.
    block: (u64, u64),
    texture_type: u64,
    width: u64,
    height: u64,
    depth: u64,
    levels: u64,
    array: u64,
    usage: u64,
    optimized: Option<bool>,
}

impl LegacyShape {
    /// A 2D texture of one level of `format` (a block of `block_bytes` bytes and `block`
    /// texels), `width` by `height` texels.
    fn flat(
        format: u64,
        (block_bytes, block): FormatBlock,
        (width, height): (u64, u64),
        usage: u64,
        optimized: Option<bool>,
    ) -> LegacyShape {
        LegacyShape {
            format,
            block_bytes,
            block,
            texture_type: 2,
            width,
            height,
            depth: 1,
            levels: 1,
            array: 1,
            usage,
            optimized,
        }
    }
}

/// What `P`'s device `d` shows of a texture of `shape` at `offset` of a `Private`, untracked
/// placement heap over no address ranges, made as core makes its legacy heap, with a `Private`
/// buffer over all of the heap made first, as core's: the texture's size; the heap's bytes over
/// the texture's span after every level and slice is copied from a `Shared` buffer of one
/// pattern, read through the heap's buffer, as core copies them to the guest's pages; then each
/// level's and slice's texels, copied to a buffer, after another pattern is copied into the
/// heap's buffer over the span, as core copies the guest's pages in. Every copy is a blit of its
/// own command buffer. `None` where the device answers nil for the texture.
fn legacy_heap_bytes_on<P: Platform>(
    d: &P::Device,
    desc: &dyn Fn(&TextureDesc) -> P::TextureDescriptor,
    shape: LegacyShape,
    offset: u64,
) -> Option<(u64, Vec<u8>, Vec<u8>)>
where
    P::Device: GpuDevice<P>,
{
    use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder as _};
    use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
    use reims_vgpu_core::platform::resource::{Buffer as _, Heap as _};

    const HEAP: u64 = 0x40_0000;
    let options = [OptionSet::StorageMode(2), OptionSet::HazardTrackingMode(0)];
    let heap = heap_of::<P>(d, HEAP, &options).heap.unwrap();
    let cover = heap.new_buffer_of_length(HEAP, 0x20, 0).unwrap().unwrap();
    let texture_desc = desc(&TextureDesc {
        texture_type: shape.texture_type,
        pixel_format: shape.format,
        width: shape.width,
        height: shape.height,
        depth: shape.depth,
        mipmap_level_count: shape.levels,
        array_length: shape.array,
        usage: shape.usage,
        resource_options: Some(0x20),
        allow_gpu_optimized_contents: shape.optimized,
        ..TextureDesc::default()
    });
    let size = d.heap_texture_size_and_align(&texture_desc).unwrap().size;
    let t = heap.new_texture(&texture_desc, offset).unwrap()?;
    let shared = |length: u64, byte: &dyn Fn(usize) -> u8| {
        let b = d
            .new_buffer_of_length(OnFault::Die, length, 0)
            .unwrap()
            .unwrap();
        let n = usize::try_from(length).unwrap();
        // SAFETY: a Shared buffer of `length` bytes, which no command uses yet.
        let s = unsafe { std::slice::from_raw_parts_mut(b.contents().as_ptr::<u8>(), n) };
        for (i, v) in s.iter_mut().enumerate() {
            *v = byte(i);
        }
        b
    };
    let bytes = |b: &P::Buffer, length: u64| {
        let n = usize::try_from(length).unwrap();
        // SAFETY: a Shared buffer's first `length` bytes, read after its copies completed.
        unsafe { std::slice::from_raw_parts(b.contents().as_ptr::<u8>(), n) }.to_vec()
    };
    let q = d.new_command_queue().unwrap();
    let run = |cmds: &mut dyn FnMut(&P::BlitEncoder)| {
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let e = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
        cmds(&e);
        e.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        assert!(cb.error().is_none(), "{:?}", cb.error());
    };
    // Each level's and slice's region, its rows' and images' bytes and its place, packed.
    let slices = if shape.texture_type == 5 { 6 } else { 1 } * shape.array;
    let mut parts = Vec::new();
    let mut packed = 0;
    for level in 0..shape.levels {
        let (w, h, z) = (
            (shape.width >> level).max(1),
            (shape.height >> level).max(1),
            (shape.depth >> level).max(1),
        );
        let (bw, bh) = shape.block;
        let row = w.div_ceil(bw) * shape.block_bytes;
        let image = row * h.div_ceil(bh);
        for slice in 0..slices {
            parts.push((level, slice, MtlSize::new(w, h, z), row, image, packed));
            packed += image * z;
        }
    }
    let source = shared(packed, &|i| (i * 13 + 5) as u8);
    run(&mut |e| {
        for &(level, slice, size, row, image, at) in &parts {
            let copy = BlitCmd::CopyBufferToTexture {
                source_buffer: &source,
                source_offset: at,
                source_bytes_per_row: row,
                source_bytes_per_image: image,
                source_size: size,
                destination_texture: &t,
                destination_slice: slice,
                destination_level: level,
                destination_origin: MtlOrigin::new(0, 0, 0),
            };
            e.encode(OnFault::Die, copy).unwrap();
        }
    });
    let span = |from: &P::Buffer, from_at: u64, to: &P::Buffer, to_at: u64| {
        run(&mut |e| {
            let copy = BlitCmd::CopyBufferToBuffer {
                source_buffer: Some(from),
                source_offset: from_at,
                destination_buffer: Some(to),
                destination_offset: to_at,
                size,
            };
            e.encode(OnFault::Die, copy).unwrap();
        });
    };
    let back = shared(size, &|_| 0);
    span(&cover, offset, &back, 0);
    let seen = bytes(&back, size);
    span(&shared(size, &|i| (i * 7 + 3) as u8), 0, &cover, offset);
    let texels = shared(packed, &|_| 0);
    run(&mut |e| {
        for &(level, slice, size, row, image, at) in &parts {
            let copy = BlitCmd::CopyTextureToBuffer {
                source_texture: &t,
                source_slice: slice,
                source_level: level,
                source_origin: MtlOrigin::new(0, 0, 0),
                source_size: size,
                destination_buffer: &texels,
                destination_offset: at,
                destination_bytes_per_row: row,
                destination_bytes_per_image: image,
            };
            e.encode(OnFault::Die, copy).unwrap();
        }
    });
    Some((size, seen, bytes(&texels, packed)))
}

/// Every colour format Metal describes on this host that the runtime makes textures of, with
/// its block's bytes and extent: neither a depth or stencil format, nor a two-plane one (550,
/// 551, which [`Placements`] keeps apart), nor a PVRTC one, nor one of 630 to 645, which an Apple
/// M4 describes and the runtime does not ([`crate::gpu::format_info`]), nor `R10Unorm_PACKED` or
/// `RG10Unorm_PACKED` (588, 589), whose size Metal answers as 0 bytes, so no heap's bytes show
/// them (their placement: `gpu::device`'s `packed_10_bit_textures_over_memory_die_named`).
fn legacy_heap_formats() -> Vec<(u64, FormatBlock)> {
    let sel = "legacy_heap_formats";
    (1..2048u64)
        .filter(|&raw| crate::gpu::format_info::described(sel, raw))
        .filter(|&raw| {
            let pf = u32::try_from(raw)
                .ok()
                .and_then(|raw| PixelFormat::try_from(raw).ok());
            pf.is_some_and(|pf| !metal2vulkan::metal::format::is_depth_stencil(pf))
        })
        .filter(|&raw| !matches!(raw, 160..=167 | 550 | 551 | 588 | 589))
        .filter_map(|raw| {
            let block = crate::gpu::format_info::block(sel, raw);
            let [bw, bh] = block.extent();
            (block.bytes() > 0).then_some((raw, (block.bytes(), (bw, bh))))
        })
        .collect()
}

/// The one-plane YCbCr formats the runtime holds for copies alone, with their bytes a texel
/// (`metal2vulkan::metal::format::copies_only`, a footprint of one plane's rows): 501, 504,
/// 521, 524, 560, 562, 563, 590 and 607.
fn one_plane_copies_only() -> Vec<(u64, u64)> {
    use metal2vulkan::metal::format::{Footprint, copies_only, footprint};
    let formats: Vec<(u64, u64)> = (0..2048u32)
        .filter_map(|raw| {
            let pf = PixelFormat::try_from(raw).ok()?;
            match footprint(pf) {
                Some(Footprint::OnePlane { bytes, .. }) if copies_only(pf) => {
                    Some((u64::from(raw), u64::from(bytes)))
                }
                _ => None,
            }
        })
        .collect();
    let raws: Vec<u64> = formats.iter().map(|&(raw, _)| raw).collect();
    assert_eq!(raws, [501, 504, 521, 524, 560, 562, 563, 590, 607]);
    formats
}

/// What [`legacy_heap_bytes_on`] answers.
type LegacyBytes = Option<(u64, Vec<u8>, Vec<u8>)>;

/// What [`legacy_heap_bytes_on`] shows of `shape` at 0x10000 on the Vulkan platform's `d`, and
/// the degradations the runtime announced as it ran that are not of [`crate::answer::DIES`]; the
/// death's site and line where it dies (of the format, which the runtime lacks, among others).
fn legacy_heap_on_vulkan(
    d: &VulkanDevice,
    shape: LegacyShape,
) -> Result<(LegacyBytes, Vec<rt::Degradation>), String> {
    let desc = |t: &TextureDesc| platform().texture_descriptor(t);
    *crate::answer::LAST_LINE.lock() = None;
    let before = crate::answer::RELAYED.lock().len();
    match catch_death(|| legacy_heap_bytes_on::<VulkanPlatform>(d, &desc, shape, 0x10000)) {
        Ok(bytes) => Ok((bytes, crate::answer::RELAYED.lock()[before..].to_vec())),
        Err(site) => {
            let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
            Err(format!("{:?} at {}: {line}", site.kind, site.file))
        }
    }
}

/// Whether the device `d` is MoltenVK's, which places each image as a Metal texture in the
/// Metal heap under its memory.
fn on_moltenvk(d: &VulkanDevice) -> bool {
    d.raw.properties().description.driver_id == Some(metal2vulkan::vk::DriverId::MOLTENVK)
}

/// A texture of one sample and a colour format that Metal keeps uncompressed (without
/// `allowGPUOptimizedContents`, at a level 0 whose shorter side is under 16 texels, or of an
/// unknown usage or of `ShaderWrite`, `PixelFormatView` or `ShaderAtomic` usage), placed in a heap over no address ranges with a buffer over all of
/// the heap first, as core makes its legacy heap, is made on MoltenVK, and the heap's bytes over
/// it are Metal's both ways ([`legacy_heap_bytes_on`]): the bytes copied out after its texels
/// are written, and the texels read after other bytes are copied in, byte for byte. Of every
/// colour format the runtime makes on this host: a 2D texture without
/// `allowGPUOptimizedContents` of 64 by 64 blocks, and of `RenderTarget` usage too; one of 8 by
/// 8 texels, or one block where that is larger, with the property; one of `PixelFormatView`
/// usage with it; one of unknown usage with it. Of `RGBA8Unorm`, `R32Uint`, `RGBA16Float` and
/// `BC1_RGBA` without the property: a mipmapped 2D texture, a 2D array, a cube, a cube array, a
/// 1D texture and a 1D array, and a 3D texture of the plain formats; with the property, render
/// targets whose shorter side is 15 texels, a 1D one and an `R32Uint` of `ShaderAtomic` usage.
/// A texture a death of the format refuses (one the runtime lacks) is left out: 677 are compared
/// on an Apple M3 Max under macOS 27.0.1
#[test]
fn legacy_heap_textures_are_metals_where_metal_keeps_them_plain() {
    let d = dev();
    if !on_moltenvk(&d) {
        return;
    }
    let mut shapes = Vec::new();
    for (format, block) in legacy_heap_formats() {
        let (_, (bw, bh)) = block;
        let big = (bw * 64, bh * 64);
        let small = (bw * (8 / bw).max(1), bh * (8 / bh).max(1));
        let compressed = (bw, bh) != (1, 1);
        shapes.push(LegacyShape::flat(format, block, big, 0x1, Some(false)));
        shapes.push(LegacyShape::flat(format, block, small, 0x1, None));
        shapes.push(LegacyShape::flat(format, block, big, 0x11, None));
        if !compressed {
            shapes.push(LegacyShape::flat(format, block, big, 0x5, Some(false)));
            shapes.push(LegacyShape::flat(format, block, big, 0x0, None));
        }
    }
    let kinds = [
        (2, 128, 64, 1, 8, 1),
        (3, 32, 32, 1, 1, 3),
        (5, 32, 32, 1, 6, 1),
        (6, 32, 32, 1, 1, 2),
        (0, 256, 1, 1, 1, 1),
        (1, 256, 1, 1, 1, 4),
        (7, 16, 16, 8, 1, 1),
    ];
    for (format, block) in [
        (70, (4, (1, 1))),
        (53, (4, (1, 1))),
        (115, (8, (1, 1))),
        (130, (8, (4, 4))),
    ] {
        for (texture_type, width, height, depth, levels, array) in kinds {
            if block.1 != (1, 1) && matches!(texture_type, 0 | 1 | 7) {
                continue;
            }
            shapes.push(LegacyShape {
                texture_type,
                width,
                height,
                depth,
                levels,
                array,
                ..LegacyShape::flat(format, block, (0, 0), 0x1, Some(false))
            });
        }
        if block.1 == (1, 1) {
            shapes.push(LegacyShape::flat(format, block, (15, 64), 0x5, None));
            shapes.push(LegacyShape::flat(format, block, (64, 15), 0x5, None));
            shapes.push(LegacyShape {
                texture_type: 0,
                ..LegacyShape::flat(format, block, (256, 1), 0x5, None)
            });
        }
    }
    shapes.push(LegacyShape::flat(53, (4, (1, 1)), (64, 64), 0x25, None));
    let mut compared = 0;
    for shape in shapes {
        let case = format!("{shape:?}");
        let Ok((ours, _)) = legacy_heap_on_vulkan(&d, shape) else {
            continue;
        };
        assert!(reims_vgpu_core::metal_matches_raw!(case, ours), "{case}");
        if ours.is_some() {
            compared += 1;
        }
    }
    assert!(compared >= 500, "{compared} compared");
}

/// A texture of each one-plane YCbCr format the runtime holds for copies alone
/// ([`one_plane_copies_only`]), placed as core places its legacy heap's
/// ([`legacy_heap_bytes_on`], rows of the format's bytes a texel), shows Metal's bytes both
/// ways: 64 by 64 and 8 by 8 texels, without `allowGPUOptimizedContents` or with it, of
/// `ShaderRead` usage and of `PixelFormatView` too: the runtime lays such a texture's texels in
/// the heap's bytes as rows, as Metal does (measured on the M3 Max under macOS 27.0.1 on
/// MoltenVK, 2026-11-07)
#[test]
fn legacy_heap_one_plane_textures_are_metals() {
    let d = dev();
    for (format, bytes) in one_plane_copies_only() {
        let block = (bytes, (1, 1));
        for shape in [
            LegacyShape::flat(format, block, (64, 64), 0x1, Some(false)),
            LegacyShape::flat(format, block, (8, 8), 0x1, None),
            LegacyShape::flat(format, block, (64, 64), 0x11, None),
        ] {
            let case = format!("{shape:?}");
            let ours = legacy_heap_on_vulkan(&d, shape).unwrap().0;
            assert!(reims_vgpu_core::metal_matches_raw!(case, ours), "{case}");
        }
    }
}

/// A depth or stencil texture of one sample without `allowGPUOptimizedContents`, placed in a heap
/// over no address ranges with a buffer over all of the heap first, as core makes its legacy
/// heap, is made on MoltenVK, laid out in Metal's tiles, and the heap's bytes over it are Metal's
/// both ways ([`legacy_heap_bytes_on`]), with nothing announced of its layout: `Depth16Unorm`
/// and `Stencil8` render targets of 8x8, 37x5, 64x64 and 100x70, a 3-level 64x64 one and a
/// 2-slice array of 32x32. `Depth32Float`, whose values a copy into it need not keep bit for bit
/// outside `[0, 1]`, is left to the runtime's own probes
#[test]
fn legacy_heap_depth_textures_are_metals_where_metal_keeps_them_plain() {
    let d = dev();
    if !on_moltenvk(&d) {
        return;
    }
    let layouts = [
        rt::Degradation::HeapTextureUntiledLayout,
        rt::Degradation::HeapTextureLosslessLayout,
    ];
    let mut shapes = Vec::new();
    for (format, block) in [(250, (2, (1, 1))), (253, (1, (1, 1)))] {
        for side in [(8, 8), (37, 5), (64, 64), (100, 70)] {
            shapes.push(LegacyShape::flat(format, block, side, 0x5, Some(false)));
        }
        let flat = LegacyShape::flat(format, block, (64, 64), 0x5, Some(false));
        shapes.push(LegacyShape { levels: 3, ..flat });
        shapes.push(LegacyShape {
            texture_type: 3,
            width: 32,
            height: 32,
            array: 2,
            ..flat
        });
    }
    for shape in shapes {
        let case = format!("{shape:?}");
        let (ours, relayed) =
            legacy_heap_on_vulkan(&d, shape).unwrap_or_else(|line| panic!("{case}: {line}"));
        let ours = ours.unwrap_or_else(|| panic!("made: {case}"));
        assert!(
            !layouts.iter().any(|kind| relayed.contains(kind)),
            "{case}: {relayed:?}"
        );
        assert!(reims_vgpu_core::metal_matches_raw!(case, ours), "{case}");
    }
}

/// A texture placed in a heap over no address ranges whose bytes the runtime does not lay out
/// there as Metal's is made where Metal makes it, and the runtime announces its layout, which is
/// relayed: one Metal may keep losslessly compressed (`heap-texture-lossless-layout`), whether
/// its bytes are Metal's (`RGBA8Unorm`, a 64x64 render target, 16x16 read textures) or not (an
/// `R32Uint` or `RGBA8Uint` render target of 16x16 or more, a 3D `RGBA8Unorm` render target, an
/// extended-range `BGRA10_XR` texture: measured, their bytes copied out or in differing from
/// Metal's on MoltenVK), depth and stencil textures with `allowGPUOptimizedContents` among them
/// (`Depth32Float` and `Stencil8`, 64x64). An 8x8 `Depth16Unorm` texture with the property, which
/// Metal keeps uncompressed, is laid out in Metal's tiles on MoltenVK, announcing nothing, and in
/// the driver's layout on any other driver (`heap-texture-untiled-layout`); so is an 8x8 4-sample
/// `RGBA8Unorm` render target without the property, whose image MoltenVK lays out as Metal's
/// heap texture (the runtime's measured fact, `MultisampleColorImagesLieAsMetalHeapTextures`)
#[test]
fn legacy_heap_textures_announce_the_runtimes_layouts() {
    let d = dev();
    let plain = (4, (1, 1));
    let moltenvk = on_moltenvk(&d);
    let lossless = Some(rt::Degradation::HeapTextureLosslessLayout);
    let untiled = Some(rt::Degradation::HeapTextureUntiledLayout);
    let layouts = [untiled, lossless];
    for (case, shape, announced, differs) in [
        (
            "RGBA8Unorm render target",
            LegacyShape::flat(70, plain, (64, 64), 0x5, None),
            lossless,
            false,
        ),
        (
            "RGBA8Unorm read",
            LegacyShape::flat(70, plain, (16, 16), 0x1, Some(true)),
            lossless,
            false,
        ),
        (
            "R32Uint render target",
            LegacyShape::flat(53, plain, (16, 16), 0x5, None),
            lossless,
            true,
        ),
        (
            "RGBA8Uint render target",
            LegacyShape::flat(73, plain, (64, 64), 0x4, None),
            lossless,
            true,
        ),
        (
            "BGRA10_XR read",
            LegacyShape::flat(552, (8, (1, 1)), (64, 64), 0x1, None),
            lossless,
            true,
        ),
        (
            "3D RGBA8Unorm render target",
            LegacyShape {
                texture_type: 7,
                depth: 8,
                ..LegacyShape::flat(70, plain, (16, 16), 0x5, None)
            },
            lossless,
            true,
        ),
        (
            "Depth32Float",
            LegacyShape::flat(252, plain, (64, 64), 0x5, None),
            lossless,
            false,
        ),
        (
            "Stencil8",
            LegacyShape::flat(253, (1, (1, 1)), (64, 64), 0x1, Some(true)),
            lossless,
            false,
        ),
        (
            "Depth16Unorm",
            LegacyShape::flat(250, (2, (1, 1)), (8, 8), 0x1, Some(true)),
            if moltenvk { None } else { untiled },
            false,
        ),
    ] {
        let (ours, relayed) =
            legacy_heap_on_vulkan(&d, shape).unwrap_or_else(|line| panic!("{case}: {line}"));
        let ours = ours.unwrap_or_else(|| panic!("made: {case}"));
        for kind in layouts {
            let heard = kind.is_some_and(|kind| relayed.contains(&kind));
            assert_eq!(heard, kind == announced, "{case}: {kind:?} in {relayed:?}");
        }
        if moltenvk && differs {
            let out = reims_vgpu_core::metal_matches_bytes!(format!("{case}: copied out"), &ours.1);
            let into = reims_vgpu_core::metal_matches_bytes!(format!("{case}: copied in"), &ours.2);
            assert!(!out || !into, "differs: {case}");
        }
        if announced.is_none() {
            let metals = reims_vgpu_core::metal_matches_raw!(case, ours);
            assert!(metals, "Metal's bytes: {case}");
        }
    }
    let multisample = TextureDesc {
        texture_type: 4,
        pixel_format: 70,
        width: 8,
        height: 8,
        sample_count: 4,
        usage: 0x4,
        resource_options: Some(0x20),
        allow_gpu_optimized_contents: Some(false),
        ..TextureDesc::default()
    };
    let options = [OptionSet::StorageMode(2), OptionSet::HazardTrackingMode(0)];
    let heap = heap_of::<VulkanPlatform>(&d, 0x10000, &options)
        .heap
        .unwrap();
    let _cover = heap
        .new_buffer_of_length(0x10000, 0x20, 0)
        .unwrap()
        .unwrap();
    let desc = platform().texture_descriptor(&multisample);
    let before = crate::answer::RELAYED.lock().len();
    let placed = heap.new_texture(&desc, 0).unwrap();
    assert!(placed.is_some(), "the multisample texture");
    let relayed = crate::answer::RELAYED.lock()[before..].to_vec();
    let untiled = rt::Degradation::HeapTextureUntiledLayout;
    assert_eq!(
        relayed.contains(&untiled),
        !moltenvk,
        "the multisample texture: {relayed:?}"
    );
}
