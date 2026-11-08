//! The tests of `encode_compute` that compare with Metal on this host: macOS only.

use crate::core::platform::resource::ComputePipeline;

use super::*;

const TGSTATIC: &[u8] = include_bytes!("../../../../../../tests/shaders/compute_tgstatic.metallib");

const CONSTANT: &[u8] = include_bytes!("../../../../../../tests/shaders/fc.metallib");

const TRUNCATED: &[u8] = include_bytes!("../../../../../../tests/shaders/kernel_trunc.metallib");

/// A pipeline's answers, in the trait's order but the ID: ID, imageblock length of 8×8,
/// largest threadgroup, resource index, static threadgroup memory, ICB support, execution
/// width.
fn answers<P: ComputePipeline>(p: &P) -> (u64, u64, u64, u64, u64, bool, u64) {
    (
        p.gpu_resource_id(),
        p.imageblock_memory_length(MtlSize::new(8, 8, 1)).unwrap(),
        p.max_total_threads_per_threadgroup(),
        p.resource_index(),
        p.static_threadgroup_memory_length(),
        p.support_indirect_command_buffers(),
        p.thread_execution_width(),
    )
}

/// newLibraryWithData: makes a metallib's library, whose functions are found by name and nil
/// for a name it lacks; bytes that are no metallib (random, a bitcode wrapper, raw AIR's
/// magic, a truncated metallib) are Metal's "Invalid library file", as on Metal
#[test]
fn libraries_and_functions() {
    let lib = library(ADD);
    assert_eq!(lib.function_names(), ["add_one"]);
    assert!(
        lib.new_function(OnFault::Return, "add_one")
            .unwrap()
            .is_some()
    );
    assert!(lib.new_function(OnFault::Die, "missing").unwrap().is_none());
    let wrapper: &[u8] = &[0xde, 0xc0, 0x17, 0x0b, 0, 0, 0, 0, 20, 0, 0, 0];
    let air: &[u8] = b"BC\xc0\xde\x35\x14\x00\x00";
    let refused = [&b"garbage"[..], wrapper, air, TRUNCATED];
    for bytes in refused {
        let e = dev().new_library(bytes).unwrap().unwrap_err();
        assert_eq!((e.domain.as_str(), e.code), ("MTLLibraryErrorDomain", 1));
        assert!(e.description.contains("\"Invalid library file\""), "{e:?}");
    }
}

/// A pipeline of a function answers its getters as Metal's does on this host; a nil function
/// and an unspecialized function constant die at the creator, as Metal aborts for the
/// constant
#[test]
fn pipelines_of_functions() {
    let kernels = [
        (ADD, "add_one"),
        (TGMEM, "tg_reverse"),
        (TGSTATIC, "tg_static"),
        (SAMPLE, "sample_k"),
    ];
    let ours: Vec<_> = kernels
        .iter()
        .map(|&(bytes, name)| answers(&pipeline(bytes, name)))
        .collect();
    assert_eq!(ours[2].4, 256, "tg_static's 64 words");
    for (&(_, name), ours) in kernels.iter().zip(&ours) {
        crate::metal_check!(name, ours);
    }
    let sel = "newComputePipelineStateWithFunction:error:";
    let site = catch_death(|| dev().new_compute_pipeline_state_for_function(None)).unwrap_err();
    assert_eq!((site.kind, site.file), (DeathKind::Abort, sel));
    let f = library(CONSTANT)
        .new_function(OnFault::Return, "fc_kernel")
        .unwrap()
        .unwrap();
    let site = catch_death(|| dev().new_compute_pipeline_state_for_function(Some(&f))).unwrap_err();
    assert_eq!((site.kind, site.file), (DeathKind::Abort, sel));
}

/// A concurrent encoder of a pass descriptor runs as a serial one does; a dispatch type no
/// Metal value is runs as serial, skipping a nil resource as Metal's does, and one past 32
/// bits dies at its setter
#[test]
fn encoders_of_pass_descriptors() {
    let add = pipeline(ADD, "add_one");
    let (input, out) = (holding(&[1, 2, 3, 4]), shared(16));
    let cb = command_buffer();
    let e = cb
        .compute_encoder_for_pass(&ComputePassDesc {
            dispatch_type: Some(1),
        })
        .unwrap()
        .unwrap();
    run(
        &cb,
        &e,
        vec![
            ComputeCmd::SetComputePipelineState { state: Some(&add) },
            ComputeCmd::SetBuffers {
                buffers: &[Some(&out), Some(&input)],
                offsets: &[0, 0],
                range: NsRange::new(0, 2),
            },
            ComputeCmd::DispatchThreads {
                threads_per_grid: MtlSize::new(4, 1, 1),
                threads_per_threadgroup: MtlSize::new(4, 1, 1),
            },
        ],
    );
    assert_eq!(words(&out), [2, 3, 4, 5]);
    /// An encoder of `dispatch_type` of a pass descriptor on `d` that names a nil resource
    /// in its residency and adds one to 4 words: its output, once it completed.
    fn unknown_type<P: crate::core::platform::Platform>(
        d: &P::Device,
        dispatch_type: u64,
    ) -> Vec<u32> {
        let lib = d.new_library(ADD).unwrap().unwrap().unwrap();
        let f = lib
            .new_function(OnFault::Return, "add_one")
            .unwrap()
            .unwrap();
        let add = d
            .new_compute_pipeline_state_for_function(Some(&f))
            .unwrap()
            .unwrap();
        let buffer = || {
            d.new_buffer_of_length(OnFault::Return, 16, 0)
                .unwrap()
                .unwrap()
        };
        let (input, out) = (buffer(), buffer());
        // SAFETY: the Shared buffer's 4 words, written before any command uses them.
        unsafe {
            std::ptr::copy_nonoverlapping([1u32, 2, 3, 4].as_ptr(), input.contents().as_ptr(), 4);
        }
        let cb = d
            .new_command_queue()
            .unwrap()
            .command_buffer(OnFault::Return)
            .unwrap()
            .unwrap();
        let e = cb
            .compute_encoder_for_pass(&ComputePassDesc {
                dispatch_type: Some(dispatch_type),
            })
            .unwrap()
            .unwrap();
        for cmd in [
            ComputeCmd::SetComputePipelineState { state: Some(&add) },
            ComputeCmd::SetBuffers {
                buffers: &[Some(&out), Some(&input)],
                offsets: &[0, 0],
                range: NsRange::new(0, 2),
            },
            ComputeCmd::UseResources {
                resources: &[Some(ResourceRef::Buffer(&input)), None],
                usage: 1,
            },
            ComputeCmd::DispatchThreads {
                threads_per_grid: MtlSize::new(4, 1, 1),
                threads_per_threadgroup: MtlSize::new(4, 1, 1),
            },
            ComputeCmd::EndEncoding,
        ] {
            e.encode(OnFault::Return, cmd).unwrap();
        }
        cb.commit(OnFault::Return).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        assert_eq!((cb.status().0, cb.error()), (COMPLETED, None));
        // SAFETY: the Shared buffer's 4 words, read after the GPU's writes completed.
        unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u32>(), 4) }.to_vec()
    }
    for dispatch_type in [2, 3, 7, 0x8000_0000, 0xffff_ffff] {
        assert_eq!(
            unknown_type::<VulkanPlatform>(&dev(), dispatch_type),
            [2, 3, 4, 5]
        );
    }
    let site = catch_death(|| {
        command_buffer().compute_encoder_for_pass(&ComputePassDesc {
            dispatch_type: Some(1 << 32),
        })
    })
    .unwrap_err();
    assert_eq!(
        (site.kind, site.file),
        (DeathKind::Abort, "setDispatchType:")
    );
}

/// What a unit of compute encoders holds, in order: an encoder of a dispatch type (`None` for
/// `computeCommandEncoder`) and the dispatch it sends, or what ends the unit.
#[derive(Debug, Clone, Copy)]
enum Part {
    Encoder(Option<u64>, Dispatch),
    Blit,
    Signal,
    Wait,
}

/// A dispatch of `add_one` over 4 words, or none.
#[derive(Debug, Clone, Copy)]
enum Dispatch {
    Direct,
    GroupsIndirect,
    ThreadsIndirect,
    Nothing,
}

/// `parts` in one command buffer on `d`, committed: the 4 words `add_one` wrote.
fn unit<P: crate::core::platform::Platform>(d: &P::Device, parts: &[Part]) -> Vec<u32> {
    use crate::core::platform::blit::{BlitCmd, BlitEncoder as _};
    use crate::core::platform::event::{EventRef, SharedEvent as _};

    let lib = d.new_library(ADD).unwrap().unwrap().unwrap();
    let f = lib
        .new_function(OnFault::Return, "add_one")
        .unwrap()
        .unwrap();
    let add = d
        .new_compute_pipeline_state_for_function(Some(&f))
        .unwrap()
        .unwrap();
    let buffer = |words: &[u32]| {
        let b = d
            .new_buffer_of_length(OnFault::Return, 64, 0)
            .unwrap()
            .unwrap();
        // SAFETY: the Shared buffer's words, written before any command uses them.
        unsafe {
            std::ptr::copy_nonoverlapping(words.as_ptr(), b.contents().as_ptr(), words.len());
        }
        b
    };
    let (input, out) = (buffer(&[1, 2, 3, 4]), buffer(&[0; 4]));
    let arguments = buffer(&[1, 1, 1, 0, 4, 1, 1, 4, 1, 1]);
    let event = d.new_shared_event().unwrap();
    event.set_signaled_value(5).unwrap();
    let cb = d
        .new_command_queue()
        .unwrap()
        .command_buffer(OnFault::Return)
        .unwrap()
        .unwrap();
    for part in parts {
        let (dispatch_type, dispatch) = match *part {
            Part::Encoder(dispatch_type, dispatch) => (dispatch_type, dispatch),
            Part::Blit => {
                let blit = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
                blit.encode(OnFault::Return, BlitCmd::EndEncoding).unwrap();
                continue;
            }
            Part::Signal => {
                cb.encode_signal(Some(EventRef::Shared(&event)), 6).unwrap();
                continue;
            }
            Part::Wait => {
                cb.encode_wait(Some(EventRef::Shared(&event)), 1).unwrap();
                continue;
            }
        };
        let e = match dispatch_type {
            None => cb.compute_encoder(OnFault::Return),
            Some(dispatch_type) => cb.compute_encoder_for_pass(&ComputePassDesc {
                dispatch_type: Some(dispatch_type),
            }),
        }
        .unwrap()
        .unwrap();
        let groups = MtlSize::new(4, 1, 1);
        let sent = match dispatch {
            Dispatch::Direct => Some(ComputeCmd::DispatchThreads {
                threads_per_grid: groups,
                threads_per_threadgroup: groups,
            }),
            Dispatch::GroupsIndirect => Some(ComputeCmd::DispatchThreadgroupsIndirect {
                indirect_buffer: Some(&arguments),
                indirect_buffer_offset: 0,
                threads_per_threadgroup: groups,
            }),
            Dispatch::ThreadsIndirect => Some(ComputeCmd::DispatchThreadsIndirect {
                indirect_buffer: Some(&arguments),
                indirect_buffer_offset: 16,
            }),
            Dispatch::Nothing => None,
        };
        if let Some(sent) = sent {
            for cmd in [
                ComputeCmd::SetComputePipelineState { state: Some(&add) },
                ComputeCmd::SetBuffers {
                    buffers: &[Some(&out), Some(&input)],
                    offsets: &[0, 0],
                    range: NsRange::new(0, 2),
                },
                sent,
            ] {
                e.encode(OnFault::Return, cmd).unwrap();
            }
        }
        e.encode(OnFault::Return, ComputeCmd::EndEncoding).unwrap();
    }
    cb.commit(OnFault::Return).unwrap();
    cb.wait_until_completed(OnFault::Return).unwrap();
    assert_eq!((cb.status().0, cb.error()), (COMPLETED, None));
    // SAFETY: the Shared buffer's 4 words, read after the GPU's writes completed.
    unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u32>(), 4) }.to_vec()
}

/// A unit of compute encoders' first indirect dispatch dies at its selector in an encoder of
/// another dispatch type than the unit's first, as Metal ends the process there; a type
/// neither serial nor concurrent counts as serial, the first encoder counts with no dispatch,
/// and later indirect dispatches go on. A blit encoder, an event signal or wait between, a
/// unit's indirect dispatch of the first's type, and a direct dispatch of another type go on,
/// as on Metal
#[test]
fn a_units_first_indirect_dispatch_of_another_type_dies() {
    use Dispatch::{Direct, GroupsIndirect as Groups, Nothing, ThreadsIndirect as Threads};
    use Part::{Blit, Encoder, Signal, Wait};

    const SERIAL: Option<u64> = None;
    const CONCURRENT: Option<u64> = Some(1);
    let groups =
        "dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:";
    let threads = "dispatchThreadsWithIndirectBuffer:indirectBufferOffset:";
    let dies: [(&[Part], &str); 7] = [
        (
            &[Encoder(CONCURRENT, Direct), Encoder(SERIAL, Groups)],
            groups,
        ),
        (
            &[Encoder(CONCURRENT, Direct), Encoder(SERIAL, Threads)],
            threads,
        ),
        (
            &[Encoder(SERIAL, Direct), Encoder(CONCURRENT, Groups)],
            groups,
        ),
        (
            &[Encoder(SERIAL, Direct), Encoder(CONCURRENT, Threads)],
            threads,
        ),
        (
            &[Encoder(CONCURRENT, Direct), Encoder(Some(0), Groups)],
            groups,
        ),
        (
            &[Encoder(CONCURRENT, Nothing), Encoder(SERIAL, Groups)],
            groups,
        ),
        (
            &[
                Encoder(CONCURRENT, Direct),
                Encoder(SERIAL, Direct),
                Encoder(SERIAL, Groups),
            ],
            groups,
        ),
    ];
    for (parts, sel) in dies {
        let site = catch_death(|| unit::<VulkanPlatform>(&dev(), parts)).unwrap_err();
        assert_eq!((site.kind, site.file), (DeathKind::Abort, sel), "{parts:?}");
    }
    let goes_on: [&[Part]; 11] = [
        &[Encoder(CONCURRENT, Groups), Encoder(SERIAL, Direct)],
        &[Encoder(CONCURRENT, Direct), Encoder(CONCURRENT, Groups)],
        &[Encoder(SERIAL, Direct), Encoder(SERIAL, Groups)],
        &[Encoder(Some(2), Direct), Encoder(SERIAL, Groups)],
        &[Encoder(CONCURRENT, Direct), Blit, Encoder(SERIAL, Groups)],
        &[Encoder(CONCURRENT, Direct), Signal, Encoder(SERIAL, Groups)],
        &[Encoder(CONCURRENT, Direct), Wait, Encoder(SERIAL, Groups)],
        &[
            Encoder(CONCURRENT, Groups),
            Encoder(SERIAL, Direct),
            Encoder(SERIAL, Groups),
        ],
        &[
            Encoder(SERIAL, Direct),
            Encoder(CONCURRENT, Direct),
            Encoder(SERIAL, Groups),
        ],
        &[Encoder(SERIAL, Groups), Encoder(CONCURRENT, Threads)],
        &[
            Encoder(CONCURRENT, Groups),
            Encoder(SERIAL, Direct),
            Encoder(CONCURRENT, Groups),
        ],
    ];
    for parts in goes_on {
        assert_eq!(
            unit::<VulkanPlatform>(&dev(), parts),
            [2, 3, 4, 5],
            "{parts:?}"
        );
    }
}
