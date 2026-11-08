//! The tests of `icb` that compare with Metal on this host: macOS only.

use reims_vgpu_metal::MetalPlatform;

use super::*;

/// `a` as an offset from `base`; 0 and Metal's nil address as themselves.
fn rel(a: u64, base: u64) -> u64 {
    if a == 0 || a == NIL_BUFFER_ADDRESS {
        a
    } else {
        a.wrapping_sub(base)
    }
}

/// A buffer's range holds its `allocatedSize`, the runtime's, which is Metal's, so the
/// `uniqueIdentifier`s after buffers of no command types and of render beside compute types,
/// which an earlier rule sized larger than Metal, and after a compute buffer crossing a page
/// at 22 commands, a patch buffer inheriting its pipeline at 3000 commands and a buffer of no
/// commands, are Metal's
#[test]
fn unique_identifiers_follow_metals_sizes() {
    fn offsets<P: Platform>(d: &P::Device) -> Vec<u64> {
        let none = IcbDesc {
            command_types: 0,
            max_vertex_buffer_bind_count: 31,
            max_fragment_buffer_bind_count: 5,
            max_kernel_buffer_bind_count: 31,
            ..IcbDesc::default()
        };
        let mixed = IcbDesc {
            command_types: 0x21,
            max_vertex_buffer_bind_count: 2,
            max_fragment_buffer_bind_count: 2,
            max_kernel_buffer_bind_count: 2,
            ..IcbDesc::default()
        };
        let compute = IcbDesc {
            command_types: 0x20,
            ..IcbDesc::default()
        };
        let patches = IcbDesc {
            command_types: 0x4,
            inherit_pipeline_state: true,
            ..IcbDesc::default()
        };
        let live = [
            make::<P>(d, &none, 1000),
            make::<P>(d, &mixed, 1000),
            make::<P>(d, &compute, 21),
            make::<P>(d, &compute, 22),
            make::<P>(d, &patches, 3000),
            make::<P>(d, &render_desc(), 0),
            make::<P>(d, &compute_desc(), 1),
        ];
        let first = live[0].unique_identifier();
        live.iter().map(|b| b.unique_identifier() - first).collect()
    }
    let vulkan = offsets::<VulkanPlatform>(&dev());
    if let Some(md) = MetalPlatform::system_default_device() {
        assert_eq!(offsets::<MetalPlatform>(&md), vulkan);
    }
}

/// An encoder [`identifiers_after_are_metals`] makes, of a command buffer never committed, on
/// platform `P`.
trait Encoder {
    /// Makes it of `cb` and ends it, on `p`'s device `d`.
    fn encode<P: Platform>(&self, p: &P, d: &P::Device, cb: &P::CommandBuffer);
}

/// The four encoder creators, each ended empty: a blit encoder, a compute encoder, a compute
/// encoder of a pass descriptor and a render encoder of one colour attachment.
#[derive(Debug, Clone, Copy)]
enum Made {
    Blit,
    Compute,
    ComputePass,
    Render,
}

impl Encoder for Made {
    fn encode<P: Platform>(&self, p: &P, d: &P::Device, cb: &P::CommandBuffer) {
        use reims_vgpu_core::platform::render::{RenderCmd, RenderEncoder as _};
        match self {
            Made::Blit => {
                let e = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
                e.encode(OnFault::Return, BlitCmd::EndEncoding).unwrap();
            }
            Made::Compute => {
                let e = cb.compute_encoder(OnFault::Return).unwrap().unwrap();
                e.encode(OnFault::Return, ComputeCmd::EndEncoding).unwrap();
            }
            Made::ComputePass => {
                let desc = reims_vgpu_core::platform::values::ComputePassDesc::default();
                let e = cb.compute_encoder_for_pass(&desc).unwrap().unwrap();
                e.encode(OnFault::Return, ComputeCmd::EndEncoding).unwrap();
            }
            Made::Render => {
                let (_target, pass) = crate::gpu::encode_render::tests::pass_of(p, d, (4, 4));
                let e = cb.render_encoder(OnFault::Return, &pass).unwrap().unwrap();
                e.encode(OnFault::Return, RenderCmd::EndEncoding).unwrap();
            }
        }
    }
}

/// A buffer made after an empty encoder of `made`'s creator, of a command buffer never
/// committed, answers Metal's `uniqueIdentifier`, as far from that of one made before it as
/// Metal's, read with no relay: the runtime takes the encoder's range of Metal's as an M3 Max's
/// take it (a blit encoder's and a render encoder's at their opening, a compute encoder's none);
/// the one made before it answers Metal's first, `0x1000018000`, before and after. Metal's run
/// first in the test's process, before MoltenVK's work takes any of Metal's range
fn identifiers_after_are_metals(made: Made) {
    fn identifiers<P: Platform>(p: &P, d: &P::Device, made: Made) -> [u64; 3] {
        let before = make::<P>(d, &compute_desc(), 1);
        let first = before.unique_identifier();
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
        made.encode(p, d, &cb);
        let after = make::<P>(d, &compute_desc(), 1);
        [first, before.unique_identifier(), after.unique_identifier()]
    }
    let metal = MetalPlatform::system_default_device()
        .map(|md| identifiers::<MetalPlatform>(&MetalPlatform::new(), &md, made));
    let p = crate::platform::tests::platform();
    let vulkan = identifiers::<VulkanPlatform>(&p, &dev(), made);
    assert_eq!(vulkan[..2], [0x10_0001_8000; 2], "{made:?}");
    assert_eq!(crate::answer::inexact_counts(), [], "{made:?}");
    if let Some(metal) = metal {
        let from_first = |n: [u64; 3]| n.map(|x| x - n[0]);
        assert_eq!(
            from_first(metal),
            from_first(vulkan),
            "{made:?}: Metal's {metal:x?}, ours {vulkan:x?}"
        );
    }
}

/// [`identifiers_after_are_metals`] an empty blit encoder
#[test]
fn identifiers_after_a_blit_encoder_are_metals() {
    identifiers_after_are_metals(Made::Blit);
}

/// [`identifiers_after_are_metals`] an empty compute encoder, which takes nothing of Metal's
/// range
#[test]
fn identifiers_after_a_compute_encoder_are_metals() {
    identifiers_after_are_metals(Made::Compute);
}

/// [`identifiers_after_are_metals`] an empty compute encoder of a pass descriptor
#[test]
fn identifiers_after_a_compute_pass_encoder_are_metals() {
    identifiers_after_are_metals(Made::ComputePass);
}

/// [`identifiers_after_are_metals`] an empty render encoder
#[test]
fn identifiers_after_a_render_encoder_are_metals() {
    identifiers_after_are_metals(Made::Render);
}

/// Indirect command buffers take Metal's numbers: plain IDs from Metal's pool, from 1; a
/// placement takes its index's low 32 bits outside them; `resourceIndex` is the ID; the
/// unique identifier is Metal's address of the buffer, the first one's `0x1000018000`, first
/// fit in slots its size decides, freed with the buffer
#[test]
fn icbs_take_metals_numbers() {
    fn numbers<P: Platform>(d: &P::Device) -> Vec<(u64, u64, u64, u64)> {
        let of = |b: &P::Icb| {
            (
                b.gpu_resource_id(),
                b.resource_index(),
                b.unique_identifier(),
                b.size(),
            )
        };
        let placed = IcbDesc {
            resource_index: Some(0x1_0000_0005),
            ..compute_desc()
        };
        let a = make::<P>(d, &compute_desc(), 1);
        let b = make::<P>(d, &render_desc(), 100);
        let c = make::<P>(d, &placed, 3);
        let mut out = vec![of(&a), of(&b), of(&c)];
        drop(a);
        let e = make::<P>(d, &compute_desc(), 2000);
        let f = make::<P>(d, &compute_desc(), 1);
        let g = make::<P>(d, &render_desc(), 0);
        out.extend([of(&e), of(&f), of(&g)]);
        drop((b, c, e, f, g));
        let h = make::<P>(d, &compute_desc(), 1);
        out.push(of(&h));
        out
    }
    // Each identifier as an offset from the first: what MoltenVK made in this process moves
    // Metal's first address here.
    fn from_first(mut n: Vec<(u64, u64, u64, u64)>) -> Vec<(u64, u64, u64, u64)> {
        let first = n[0].2;
        for x in &mut n {
            x.2 -= first;
        }
        n
    }
    let vulkan = numbers::<VulkanPlatform>(&dev());
    assert_eq!(vulkan[0], (1, 1, 0x10_0001_8000, 1));
    assert_eq!(vulkan[2].0, 5);
    if let Some(md) = MetalPlatform::system_default_device() {
        assert_eq!(
            from_first(numbers::<MetalPlatform>(&md)),
            from_first(vulkan)
        );
    }
}

/// An execution of a command a kernel encoded with a state the program released since, whose
/// ID nothing took, runs the released state's program on Metal as on this platform
/// (`an_execution_of_a_record_the_gpu_wrote_naming_a_released_state_runs_it`): the state's
/// `mark` writes 1
#[test]
fn an_execution_of_a_record_the_gpu_wrote_naming_a_released_state_runs_it_as_metals() {
    let Some(md) = MetalPlatform::system_default_device() else {
        return;
    };
    let ours = released_state_marks::<VulkanPlatform>(&dev());
    assert_eq!((released_state_marks::<MetalPlatform>(&md), ours), (1, 1));
}

/// A reservation holding an ID of 2^23 or more is the runtime's too, whose IDs stay Metal's: 2
/// and 2^23 + 1 reserved, both hand out 1 then 3
#[test]
fn a_reservation_past_2_23_is_metals() {
    fn ids<P: Platform>(d: &P::Device) -> Vec<P::Icb> {
        d.reserve_resource_indices(5, &[2, (1 << 23) + 1]).unwrap();
        (0..2).map(|_| make::<P>(d, &compute_desc(), 1)).collect()
    }
    if let Some(md) = MetalPlatform::system_default_device() {
        let metal: Vec<u64> = ids::<MetalPlatform>(&md)
            .iter()
            .map(|b| b.gpu_resource_id())
            .collect();
        assert_eq!(metal, [1, 3]);
    }
    let live = ids::<VulkanPlatform>(&dev());
    let seen: Vec<u64> = live.iter().map(|b| b.gpu_resource_id()).collect();
    assert_eq!(seen, [1, 3]);
}

/// A buffer whose own ID is 2^23 or more, which plain allocation hands out past a reservation of
/// 16,000,000 (the smaller free range, 16,000,001 on), is made at that ID, as Metal makes it
#[test]
fn a_buffer_numbered_past_2_23_is_metals() {
    fn id<P: Platform>(d: &P::Device) -> u64 {
        d.reserve_resource_indices(5, &[16_000_000]).unwrap();
        make::<P>(d, &compute_desc(), 1).gpu_resource_id()
    }
    if let Some(md) = MetalPlatform::system_default_device() {
        assert_eq!(id::<MetalPlatform>(&md), 16_000_001);
    }
    assert_eq!(id::<VulkanPlatform>(&dev()), 16_000_001);
}

/// Plain IDs given back come back from the smallest free range, as Metal's do: with 2, 3 and
/// 5 of 1 to 8 given back, 5, 2, 3, then 9
#[test]
fn ids_given_back_come_back_from_the_smallest_free_range() {
    fn ids<P: Platform>(d: &P::Device) -> Vec<u64> {
        let mut live: Vec<Option<P::Icb>> = (0..8)
            .map(|_| Some(make::<P>(d, &compute_desc(), 1)))
            .collect();
        for i in [1, 2, 4] {
            live[i] = None;
        }
        let again: Vec<P::Icb> = (0..4).map(|_| make::<P>(d, &compute_desc(), 1)).collect();
        again.iter().map(|b| b.gpu_resource_id()).collect()
    }
    if let Some(md) = MetalPlatform::system_default_device() {
        assert_eq!(ids::<MetalPlatform>(&md), [5, 2, 3, 9]);
    }
    assert_eq!(ids::<VulkanPlatform>(&dev()), [5, 2, 3, 9]);
}

/// The first pool holds 32 IDs, as Metal's: with 1 to 25 handed out and 2 to 9 given back,
/// the next ID is 26, from the free range of 26 to 31, the smaller, where a pool of 64 would
/// hand out 2
#[test]
fn the_first_pool_holds_32_ids() {
    fn next<P: Platform>(d: &P::Device) -> u64 {
        let mut live: Vec<Option<P::Icb>> = (0..25)
            .map(|_| Some(make::<P>(d, &compute_desc(), 1)))
            .collect();
        for slot in &mut live[1..9] {
            *slot = None;
        }
        make::<P>(d, &compute_desc(), 1).gpu_resource_id()
    }
    if let Some(md) = MetalPlatform::system_default_device() {
        assert_eq!(next::<MetalPlatform>(&md), 26);
    }
    assert_eq!(next::<VulkanPlatform>(&dev()), 26);
}

/// A reservation of 100 alone starts plain IDs past it, where the range above it is the
/// smaller, as Metal's does
#[test]
fn a_reservation_of_100_starts_plain_ids_past_it() {
    fn ids<P: Platform>(d: &P::Device) -> Vec<u64> {
        d.reserve_resource_indices(5, &[100]).unwrap();
        let live: Vec<P::Icb> = (0..3).map(|_| make::<P>(d, &compute_desc(), 1)).collect();
        live.iter().map(|b| b.gpu_resource_id()).collect()
    }
    if let Some(md) = MetalPlatform::system_default_device() {
        assert_eq!(ids::<MetalPlatform>(&md), [101, 102, 103]);
    }
    assert_eq!(ids::<VulkanPlatform>(&dev()), [101, 102, 103]);
}

/// A reservation of 10 alone leaves plain IDs from 1, as Metal's does: the first pool holds
/// 32 IDs, so the range below 10 is the smaller
#[test]
fn a_reservation_of_10_leaves_plain_ids_from_1() {
    fn ids<P: Platform>(d: &P::Device) -> Vec<u64> {
        d.reserve_resource_indices(5, &[10]).unwrap();
        let live: Vec<P::Icb> = (0..3).map(|_| make::<P>(d, &compute_desc(), 1)).collect();
        live.iter().map(|b| b.gpu_resource_id()).collect()
    }
    if let Some(md) = MetalPlatform::system_default_device() {
        assert_eq!(ids::<MetalPlatform>(&md), [1, 2, 3]);
    }
    assert_eq!(ids::<VulkanPlatform>(&dev()), [1, 2, 3]);
}

/// A reservation of 2 and 16382 hands out 1, 16383, then 3, as Metal's does: the smallest
/// free ranges first, the lower of two alike
#[test]
fn a_reservation_of_2_and_16382_hands_out_1_then_16383() {
    fn ids<P: Platform>(d: &P::Device) -> Vec<u64> {
        d.reserve_resource_indices(5, &[2, 16382]).unwrap();
        let live: Vec<P::Icb> = (0..3).map(|_| make::<P>(d, &compute_desc(), 1)).collect();
        live.iter().map(|b| b.gpu_resource_id()).collect()
    }
    if let Some(md) = MetalPlatform::system_default_device() {
        assert_eq!(ids::<MetalPlatform>(&md), [1, 16383, 3]);
    }
    assert_eq!(ids::<VulkanPlatform>(&dev()), [1, 16383, 3]);
}

/// A reservation holding an ID of 2^32 or more is dropped whole, and the next call is the
/// reservation, as Metal's is: 2 reserved, so 1, 3, 4
#[test]
fn a_reservation_past_32_bits_leaves_it_to_the_next_call() {
    fn ids<P: Platform>(d: &P::Device) -> Vec<u64> {
        d.reserve_resource_indices(5, &[1 << 32, 3]).unwrap();
        d.reserve_resource_indices(5, &[2]).unwrap();
        let live: Vec<P::Icb> = (0..3).map(|_| make::<P>(d, &compute_desc(), 1)).collect();
        live.iter().map(|b| b.gpu_resource_id()).collect()
    }
    if let Some(md) = MetalPlatform::system_default_device() {
        assert_eq!(ids::<MetalPlatform>(&md), [1, 3, 4]);
    }
    assert_eq!(ids::<VulkanPlatform>(&dev()), [1, 3, 4]);
}

/// A reservation after a placed buffer is ignored, as Metal ignores it: the placement counts
/// as an allocation, though it takes no plain ID
#[test]
fn a_reservation_after_a_placement_is_ignored() {
    fn ids<P: Platform>(d: &P::Device) -> Vec<u64> {
        let placed = IcbDesc {
            resource_index: Some(9),
            ..compute_desc()
        };
        let p = make::<P>(d, &placed, 1);
        d.reserve_resource_indices(5, &[2]).unwrap();
        let live: Vec<P::Icb> = (0..2).map(|_| make::<P>(d, &compute_desc(), 1)).collect();
        [&p].into_iter()
            .chain(&live)
            .map(|b| b.gpu_resource_id())
            .collect()
    }
    if let Some(md) = MetalPlatform::system_default_device() {
        assert_eq!(ids::<MetalPlatform>(&md), [9, 1, 2]);
    }
    assert_eq!(ids::<VulkanPlatform>(&dev()), [9, 1, 2]);
}

/// 0 beside other IDs is no ID, and the call is theirs, as Metal's is: 0 and 20 hand out
/// what 20 alone does, 21 to 31, 1 to 19, then 32 on
#[test]
fn a_reservation_holding_0_beside_other_ids_is_theirs() {
    fn ids<P: Platform>(d: &P::Device) -> Vec<u64> {
        d.reserve_resource_indices(5, &[0, 20]).unwrap();
        let live: Vec<P::Icb> = (0..33).map(|_| make::<P>(d, &compute_desc(), 1)).collect();
        live.iter().map(|b| b.gpu_resource_id()).collect()
    }
    let want: Vec<u64> = (21..=31).chain(1..=19).chain(32..=34).collect();
    if let Some(md) = MetalPlatform::system_default_device() {
        assert_eq!(ids::<MetalPlatform>(&md), want);
    }
    assert_eq!(ids::<VulkanPlatform>(&dev()), want);
}

/// A reservation of 0 alone reserves nothing, as Metal's does: plain IDs from 1
#[test]
fn a_reservation_of_0_alone_reserves_nothing() {
    fn ids<P: Platform>(d: &P::Device) -> Vec<u64> {
        d.reserve_resource_indices(5, &[0]).unwrap();
        let live: Vec<P::Icb> = (0..40).map(|_| make::<P>(d, &compute_desc(), 1)).collect();
        live.iter().map(|b| b.gpu_resource_id()).collect()
    }
    let want: Vec<u64> = (1..=40).collect();
    if let Some(md) = MetalPlatform::system_default_device() {
        assert_eq!(ids::<MetalPlatform>(&md), want);
    }
    assert_eq!(ids::<VulkanPlatform>(&dev()), want);
}

/// A reservation after one of 0 alone is taken beside the pool's first range, and the IDs are
/// then handed out twice, as Metal's are (0 then 20: 21 to 31, 1 to 19, then 1 to 10 again)
#[test]
fn a_reservation_after_one_of_0_alone_hands_out_ids_twice_as_metals() {
    fn ids<P: Platform>(d: &P::Device) -> Vec<u64> {
        d.reserve_resource_indices(5, &[0]).unwrap();
        d.reserve_resource_indices(5, &[20]).unwrap();
        let live: Vec<P::Icb> = (0..40).map(|_| make::<P>(d, &compute_desc(), 1)).collect();
        live.iter().map(|b| b.gpu_resource_id()).collect()
    }
    let twice: Vec<u64> = (21..=31).chain(1..=19).chain(1..=10).collect();
    if let Some(md) = MetalPlatform::system_default_device() {
        assert_eq!(ids::<MetalPlatform>(&md), twice);
    }
    assert_eq!(ids::<VulkanPlatform>(&dev()), twice);
}

/// A reservation of 2^31 - 1, the highest ID Metal takes without spinning, is taken, plain IDs
/// then starting at 1, as Metal's
#[test]
#[ignore = "opt-in: it holds 16 GiB of the GPU's memory, where other processes' work fails"]
fn a_reservation_of_2_31_minus_1_is_metals() {
    fn ids<P: Platform>(d: &P::Device) -> Vec<u64> {
        d.reserve_resource_indices(5, &[(1 << 31) - 1]).unwrap();
        let live: Vec<P::Icb> = (0..3).map(|_| make::<P>(d, &compute_desc(), 1)).collect();
        live.iter().map(|b| b.gpu_resource_id()).collect()
    }
    if let Some(md) = MetalPlatform::system_default_device() {
        assert_eq!(ids::<MetalPlatform>(&md), [1, 2, 3]);
    }
    assert_eq!(ids::<VulkanPlatform>(&dev()), [1, 2, 3]);
}

/// After a buffer was made, a reservation Metal would spin at, and one after a call of 0
/// alone, are ignored as any other, as Metal ignores them
#[test]
fn reservations_after_an_allocation_are_ignored_whatever_they_hold() {
    fn ids<P: Platform>(d: &P::Device) -> Vec<u64> {
        d.reserve_resource_indices(5, &[0]).unwrap();
        let first = make::<P>(d, &compute_desc(), 1);
        d.reserve_resource_indices(5, &[(1 << 31) + 5]).unwrap();
        d.reserve_resource_indices(5, &[20]).unwrap();
        let live: Vec<P::Icb> = (0..2).map(|_| make::<P>(d, &compute_desc(), 1)).collect();
        [&first]
            .into_iter()
            .chain(&live)
            .map(|b| b.gpu_resource_id())
            .collect()
    }
    if let Some(md) = MetalPlatform::system_default_device() {
        assert_eq!(ids::<MetalPlatform>(&md), [1, 2, 3]);
    }
    assert_eq!(ids::<VulkanPlatform>(&dev()), [1, 2, 3]);
}

/// A placement at an ID past Metal's table, which Metal writes past, dies at the creator; a
/// descriptor of no command types makes a buffer, as Metal makes one
#[test]
fn creations_metal_does_not_make_as_asked_die() {
    let sel = "newIndirectCommandBufferWithDescriptor:maxCommandCount:options:";
    let d = dev();
    let far = IcbDesc {
        resource_index: Some(16384),
        ..compute_desc()
    };
    let site = catch_death(|| make::<VulkanPlatform>(&d, &far, 1)).unwrap_err();
    assert_eq!((site.kind, site.file), (DeathKind::Abort, sel));
    let none = IcbDesc {
        command_types: 0,
        ..compute_desc()
    };
    assert_eq!(make::<VulkanPlatform>(&d, &none, 1).size(), 1);
    if let Some(md) = MetalPlatform::system_default_device() {
        assert_eq!(make::<MetalPlatform>(&md, &none, 1).size(), 1);
    }
    let near = IcbDesc {
        resource_index: Some(16383),
        ..compute_desc()
    };
    assert_eq!(
        make::<VulkanPlatform>(&d, &near, 1).gpu_resource_id(),
        16383
    );
}

/// What a compute command's getters answer core, fresh, after each setter core sends, a
/// nil buffer with and without a stride, and a reset, as on Metal
#[test]
fn compute_commands_read_back_as_on_metal() {
    fn reads<P: Platform>(d: &P::Device) -> Vec<Vec<u64>> {
        let icb = make::<P>(d, &compute_desc(), 2);
        let b = buffer::<P>(d);
        let base = b.gpu_address();
        let read = |c: &P::IndirectComputeCommand| {
            let ty = c.command_type();
            let mut v = vec![
                ty,
                c.pipeline_state_unique_identifier(),
                u64::from(c.has_barrier()),
            ];
            for i in 0..2 {
                v.push(rel(c.kernel_buffer_address(i).unwrap(), base));
                v.push(c.kernel_attribute_stride(i).unwrap());
                v.push(c.threadgroup_memory_length(i).unwrap());
            }
            let r = c.stage_in_region();
            v.extend([r.origin.x, r.origin.y, r.origin.z]);
            v.extend([r.size.width, r.size.height, r.size.depth]);
            let (g, t) = match ty {
                0x20 => {
                    let a = c.dispatch_threadgroups_args().unwrap();
                    (a.threadgroups_per_grid, a.threads_per_threadgroup)
                }
                0x40 => {
                    let a = c.dispatch_threads_args().unwrap();
                    (a.threads_per_grid, a.threads_per_threadgroup)
                }
                _ => return v,
            };
            v.extend([g.width, g.height, g.depth, t.width, t.height, t.depth]);
            let block = c.image_block_args().unwrap();
            v.extend([block.width, block.height]);
            v
        };
        let c = icb.compute_command(0).unwrap().unwrap();
        let send = |cmd| c.encode(cmd).unwrap();
        let mut out = vec![read(&c)];
        send(IcbComputeCmd::SetKernelBuffer {
            buffer: Some(&b),
            offset: 16,
            index: 0,
        });
        send(IcbComputeCmd::SetKernelBufferAttributeStride {
            buffer: Some(&b),
            offset: 32,
            stride: 8,
            index: 1,
        });
        send(IcbComputeCmd::SetThreadgroupMemoryLength {
            length: 64,
            index: 1,
        });
        send(IcbComputeCmd::SetStageInRegion {
            region: MtlRegion::new(MtlOrigin::new(1, 2, 3), MtlSize::new(4, 5, 6)),
        });
        send(IcbComputeCmd::SetImageblockWidth {
            width: 8,
            height: 4,
        });
        send(IcbComputeCmd::ConcurrentDispatchThreadgroups {
            threadgroups_per_grid: MtlSize::new(2, 3, 4),
            threads_per_threadgroup: MtlSize::new(8, 1, 1),
        });
        send(IcbComputeCmd::SetBarrier);
        out.push(read(&c));
        send(IcbComputeCmd::SetKernelBuffer {
            buffer: None,
            offset: 0,
            index: 0,
        });
        send(IcbComputeCmd::SetKernelBufferAttributeStride {
            buffer: None,
            offset: 0,
            stride: 12,
            index: 1,
        });
        send(IcbComputeCmd::ClearBarrier);
        send(IcbComputeCmd::ConcurrentDispatchThreads {
            threads_per_grid: MtlSize::new(5, 1, 1),
            threads_per_threadgroup: MtlSize::new(4, 1, 1),
        });
        out.push(read(&c));
        send(IcbComputeCmd::Reset);
        out.push(read(&c));
        out.push(read(&icb.compute_command(1).unwrap().unwrap()));
        out
    }
    let vulkan = reads::<VulkanPlatform>(&dev());
    if let Some(md) = MetalPlatform::system_default_device() {
        assert_eq!(reads::<MetalPlatform>(&md), vulkan);
    }
}

/// What a render command's getters answer core, fresh, after each draw kind with its
/// buffers, of Metal's private primitive types 5 to 8 too, a barrier set and cleared, and a
/// reset, as on Metal
#[test]
fn render_commands_read_back_as_on_metal() {
    fn reads<P: Platform>(d: &P::Device) -> Vec<Vec<u64>> {
        let icb = make::<P>(d, &render_desc(), 2);
        let b = buffer::<P>(d);
        let base = b.gpu_address();
        let read = |c: &P::IndirectRenderCommand| {
            let ty = c.command_type();
            let mut v = vec![
                ty,
                c.pipeline_state_unique_identifier(),
                u64::from(c.has_barrier()),
            ];
            for i in 0..2 {
                v.push(rel(c.vertex_buffer_address(i).unwrap(), base));
                v.push(c.vertex_attribute_stride(i).unwrap());
                v.push(rel(c.fragment_buffer_address(i).unwrap(), base));
            }
            match ty {
                1 => {
                    let a = c.draw_args().unwrap();
                    v.extend([
                        a.primitive_type,
                        a.vertex_start,
                        a.vertex_count,
                        a.instance_count,
                        a.base_instance,
                    ]);
                }
                2 => {
                    let a = c.draw_indexed_args().unwrap();
                    v.extend([
                        rel(a.index_buffer_gpu_virtual_address, base),
                        a.primitive_type,
                        a.index_type,
                        a.index_count,
                        a.instance_count,
                        a.base_vertex,
                        a.base_instance,
                    ]);
                }
                4 => {
                    let a = c.draw_patches_args().unwrap();
                    v.extend([
                        rel(a.patch_index_buffer_gpu_virtual_address, base),
                        a.number_of_patch_control_points,
                        a.patch_start,
                        a.patch_count,
                        a.instance_count,
                        a.base_instance,
                    ]);
                }
                8 => {
                    let a = c.draw_indexed_patches_args().unwrap();
                    v.extend([
                        rel(a.patch_index_buffer_gpu_virtual_address, base),
                        rel(a.control_point_index_buffer_gpu_virtual_address, base),
                        a.number_of_patch_control_points,
                        a.patch_start,
                        a.patch_count,
                        a.instance_count,
                        a.base_instance,
                    ]);
                }
                _ => return v,
            }
            if ty >= 4 {
                let t = c.tessellation_factor_args().unwrap();
                v.extend([
                    rel(t.gpu_virtual_address, base),
                    t.instance_stride,
                    u64::from(t.scale.to_bits()),
                ]);
            }
            v
        };
        let c = icb.render_command(0).unwrap().unwrap();
        let send = |cmd| c.encode(cmd).unwrap();
        let mut out = vec![read(&c)];
        send(IcbRenderCmd::SetVertexBuffer {
            buffer: Some(&b),
            offset: 16,
            index: 0,
        });
        send(IcbRenderCmd::SetVertexBufferAttributeStride {
            buffer: Some(&b),
            offset: 32,
            stride: 20,
            index: 1,
        });
        send(IcbRenderCmd::SetFragmentBuffer {
            buffer: Some(&b),
            offset: 48,
            index: 1,
        });
        send(IcbRenderCmd::SetBarrier);
        send(IcbRenderCmd::DrawPrimitives {
            primitive_type: 3,
            vertex_start: 1,
            vertex_count: 3,
            instance_count: 2,
            base_instance: 0,
        });
        out.push(read(&c));
        send(IcbRenderCmd::DrawIndexedPrimitives {
            primitive_type: 3,
            index_count: 6,
            index_type: 1,
            index_buffer: Some(&b),
            index_buffer_offset: 64,
            instance_count: 1,
            base_vertex: -2,
            base_instance: 0,
        });
        out.push(read(&c));
        send(IcbRenderCmd::SetVertexBuffer {
            buffer: None,
            offset: 0,
            index: 0,
        });
        send(IcbRenderCmd::SetVertexBufferAttributeStride {
            buffer: None,
            offset: 0,
            stride: 24,
            index: 1,
        });
        send(IcbRenderCmd::DrawPatches {
            number_of_patch_control_points: 3,
            patch_start: 0,
            patch_count: 1,
            patch_index_buffer: None,
            patch_index_buffer_offset: 0,
            instance_count: 1,
            base_instance: 0,
            buffer: Some(&b),
            offset: 128,
            instance_stride: 0,
        });
        out.push(read(&c));
        send(IcbRenderCmd::ClearBarrier);
        for control_points in [None, Some(&b)] {
            send(IcbRenderCmd::DrawIndexedPatches {
                number_of_patch_control_points: 4,
                patch_start: 1,
                patch_count: 2,
                patch_index_buffer: Some(&b),
                patch_index_buffer_offset: 8,
                control_point_index_buffer: control_points,
                control_point_index_buffer_offset: 16,
                instance_count: 3,
                base_instance: 1,
                buffer: Some(&b),
                offset: 192,
                instance_stride: 4,
            });
            out.push(read(&c));
        }
        send(IcbRenderCmd::DrawIndexedPatches {
            number_of_patch_control_points: 4,
            patch_start: 1,
            patch_count: 2,
            patch_index_buffer: None,
            patch_index_buffer_offset: 0,
            control_point_index_buffer: Some(&b),
            control_point_index_buffer_offset: 16,
            instance_count: 3,
            base_instance: 1,
            buffer: Some(&b),
            offset: 192,
            instance_stride: 4,
        });
        out.push(read(&c));
        for primitive_type in 5..=8 {
            send(IcbRenderCmd::DrawPrimitives {
                primitive_type,
                vertex_start: 2,
                vertex_count: 8,
                instance_count: 1,
                base_instance: 3,
            });
            out.push(read(&c));
            send(IcbRenderCmd::DrawIndexedPrimitives {
                primitive_type,
                index_count: 12,
                index_type: 0,
                index_buffer: Some(&b),
                index_buffer_offset: 32,
                instance_count: 2,
                base_vertex: 1,
                base_instance: 0,
            });
            out.push(read(&c));
        }
        send(IcbRenderCmd::SetBarrier);
        send(IcbRenderCmd::Reset);
        out.push(read(&c));
        out.push(read(&icb.render_command(1).unwrap().unwrap()));
        out
    }
    let vulkan = reads::<VulkanPlatform>(&dev());
    if let Some(md) = MetalPlatform::system_default_device() {
        assert_eq!(reads::<MetalPlatform>(&md), vulkan);
    }
}

/// What the buffer getters answer of a slot through plain sets and stride forms of a buffer
/// and of nil, at offsets of 32 bits and more, and resets, on buffers with and without
/// dynamic attribute strides, as on Metal
#[test]
fn buffer_slots_read_back_as_on_metal() {
    fn reads<P: Platform>(d: &P::Device, dynamic: bool) -> Vec<Vec<u64>> {
        let render = IcbDesc {
            support_dynamic_attribute_stride: dynamic,
            ..render_desc()
        };
        let compute = IcbDesc {
            support_dynamic_attribute_stride: dynamic,
            ..compute_desc()
        };
        let (r, k) = (make::<P>(d, &render, 1), make::<P>(d, &compute, 1));
        let b = buffer::<P>(d);
        let base = b.gpu_address();
        let (rc, kc) = (
            r.render_command(0).unwrap().unwrap(),
            k.compute_command(0).unwrap().unwrap(),
        );
        let read = || {
            let mut v = Vec::from(at(rc.vertex_buffer_address(1).unwrap(), base));
            v.push(rc.vertex_attribute_stride(1).unwrap());
            v.extend(at(rc.fragment_buffer_address(1).unwrap(), base));
            v.extend(at(kc.kernel_buffer_address(1).unwrap(), base));
            v.push(kc.kernel_attribute_stride(1).unwrap());
            v
        };
        let plain = |buffer, offset| {
            rc.encode(IcbRenderCmd::SetVertexBuffer {
                buffer,
                offset,
                index: 1,
            })
            .unwrap();
            rc.encode(IcbRenderCmd::SetFragmentBuffer {
                buffer,
                offset,
                index: 1,
            })
            .unwrap();
            kc.encode(IcbComputeCmd::SetKernelBuffer {
                buffer,
                offset,
                index: 1,
            })
            .unwrap();
        };
        let strided = |buffer: Option<&P::Buffer>, offset, stride| {
            rc.encode(IcbRenderCmd::SetVertexBufferAttributeStride {
                buffer,
                offset,
                stride,
                index: 1,
            })
            .unwrap();
            kc.encode(IcbComputeCmd::SetKernelBufferAttributeStride {
                buffer,
                offset,
                stride,
                index: 1,
            })
            .unwrap();
        };
        let reset = || {
            rc.encode(IcbRenderCmd::Reset).unwrap();
            kc.encode(IcbComputeCmd::Reset).unwrap();
        };
        let mut out = vec![read()];
        plain(None, 0x1234_5678_9abc);
        out.push(read());
        strided(None, u64::MAX - 7, 20);
        out.push(read());
        reset();
        out.push(read());
        strided(Some(&b), 16, 24);
        out.push(read());
        plain(None, 48);
        out.push(read());
        strided(Some(&b), 32, 24);
        out.push(read());
        plain(None, 0);
        out.push(read());
        strided(None, 0, 28);
        out.push(read());
        strided(Some(&b), 64, 36);
        out.push(read());
        strided(None, 80, 40);
        out.push(read());
        plain(Some(&b), 8);
        out.push(read());
        out
    }
    let vulkan = [true, false].map(|dynamic| reads::<VulkanPlatform>(&dev(), dynamic));
    if let Some(md) = MetalPlatform::system_default_device() {
        assert_eq!(
            [true, false].map(|dynamic| reads::<MetalPlatform>(&md, dynamic)),
            vulkan
        );
    }
}

/// What the buffer getters answer of slots past the 32 a record holds, of a descriptor of 64
/// buffers of each stage: buffers set with strides and nil, read back as on Metal
#[test]
fn slots_past_the_record_read_back_as_on_metal() {
    fn reads<P: Platform>(d: &P::Device, dynamic: bool) -> Vec<Vec<u64>> {
        let render = IcbDesc {
            max_vertex_buffer_bind_count: 64,
            max_fragment_buffer_bind_count: 64,
            support_dynamic_attribute_stride: dynamic,
            ..render_desc()
        };
        let compute = IcbDesc {
            max_kernel_buffer_bind_count: 64,
            support_dynamic_attribute_stride: dynamic,
            ..compute_desc()
        };
        let (r, k) = (make::<P>(d, &render, 1), make::<P>(d, &compute, 1));
        let b = buffer::<P>(d);
        let base = b.gpu_address();
        let (rc, kc) = (
            r.render_command(0).unwrap().unwrap(),
            k.compute_command(0).unwrap().unwrap(),
        );
        for (i, index) in [30, 31, 32, 40, 63].into_iter().enumerate() {
            let (offset, stride) = (16 * (i as u64 + 1), 4 * (i as u64 + 1));
            rc.encode(IcbRenderCmd::SetVertexBufferAttributeStride {
                buffer: Some(&b),
                offset,
                stride,
                index,
            })
            .unwrap();
            rc.encode(IcbRenderCmd::SetFragmentBuffer {
                buffer: Some(&b),
                offset,
                index,
            })
            .unwrap();
            kc.encode(IcbComputeCmd::SetKernelBufferAttributeStride {
                buffer: Some(&b),
                offset,
                stride,
                index,
            })
            .unwrap();
        }
        rc.encode(IcbRenderCmd::SetVertexBuffer {
            buffer: None,
            offset: 0x77,
            index: 50,
        })
        .unwrap();
        kc.encode(IcbComputeCmd::SetKernelBuffer {
            buffer: None,
            offset: 0x99,
            index: 50,
        })
        .unwrap();
        [29, 30, 31, 32, 33, 40, 50, 63]
            .into_iter()
            .map(|i| {
                let mut v = Vec::from(at(rc.vertex_buffer_address(i).unwrap(), base));
                v.push(rc.vertex_attribute_stride(i).unwrap());
                v.extend(at(rc.fragment_buffer_address(i).unwrap(), base));
                v.extend(at(kc.kernel_buffer_address(i).unwrap(), base));
                v.push(kc.kernel_attribute_stride(i).unwrap());
                v
            })
            .collect()
    }
    let vulkan = [true, false].map(|dynamic| reads::<VulkanPlatform>(&dev(), dynamic));
    if let Some(md) = MetalPlatform::system_default_device() {
        assert_eq!(
            [true, false].map(|dynamic| reads::<MetalPlatform>(&md, dynamic)),
            vulkan
        );
    }
}

/// What the threadgroup memory length getter answers past the 31 lengths a record holds, of
/// a descriptor of 40, as on Metal
#[test]
fn lengths_past_the_record_read_back_as_on_metal() {
    fn reads<P: Platform>(d: &P::Device) -> [u64; 6] {
        let desc = IcbDesc {
            max_kernel_threadgroup_memory_bind_count: 40,
            ..compute_desc()
        };
        let icb = make::<P>(d, &desc, 1);
        let c = icb.compute_command(0).unwrap().unwrap();
        for (length, index) in [(16, 30), (32, 31), (48, 35), (64, 39)] {
            c.encode(IcbComputeCmd::SetThreadgroupMemoryLength { length, index })
                .unwrap();
        }
        [29, 30, 31, 35, 38, 39].map(|i| c.threadgroup_memory_length(i).unwrap())
    }
    let vulkan = reads::<VulkanPlatform>(&dev());
    assert_eq!(vulkan, [0, 16, 32, 48, 0, 64]);
    if let Some(md) = MetalPlatform::system_default_device() {
        assert_eq!(reads::<MetalPlatform>(&md), vulkan);
    }
}

/// A blit's copy and reset of commands move and clear the threadgroup memory lengths past the
/// 31 a record holds as they do the record's, as on Metal: lengths at 30, 33 and 37 of two
/// commands, the first copied over a second buffer's command 1 and the second reset. The copied
/// length 37 is the runtime's 0, where Metal's copy carries it from the command, which names no
/// pipeline state: its getter is relayed, a death in strict mode
/// ([`Inexact::IcbCopiedLength`](crate::answer::Inexact::IcbCopiedLength))
#[test]
fn blits_move_lengths_past_the_record_as_metals() {
    /// Every length but the copied 37, and the second buffer, whose command 1 holds it.
    fn reads<P: Platform>(d: &P::Device) -> (Vec<u64>, P::Icb) {
        let desc = IcbDesc {
            max_kernel_threadgroup_memory_bind_count: 40,
            ..compute_desc()
        };
        let (a, b) = (make::<P>(d, &desc, 2), make::<P>(d, &desc, 2));
        let command = |icb: &P::Icb, i| icb.compute_command(i).unwrap().unwrap();
        for (i, base) in [(0, 16), (1, 48)] {
            for (index, add) in [(30, 0), (33, 4), (37, 8)] {
                let length = base + add;
                command(&a, i)
                    .encode(IcbComputeCmd::SetThreadgroupMemoryLength { length, index })
                    .unwrap();
            }
        }
        let cb = blits::<P>(
            d,
            vec![
                BlitCmd::CopyIndirectCommandBuffer {
                    source: &a,
                    source_range: NsRange::new(0, 1),
                    destination: &b,
                    destination_index: 1,
                },
                BlitCmd::ResetCommandsInBuffer {
                    buffer: &a,
                    range: NsRange::new(1, 1),
                },
            ],
        );
        assert_eq!(cb.status().0, COMPLETED);
        let mut lengths = Vec::new();
        for (second, icb) in [(false, &a), (true, &b)] {
            for i in [0, 1] {
                for index in [30, 33, 37] {
                    if !(second && i == 1 && index == 37) {
                        lengths.push(command(icb, i).threadgroup_memory_length(index).unwrap());
                    }
                }
            }
        }
        (lengths, b)
    }
    let length = |icb: &VulkanIcb| {
        let command = icb.compute_command(1).unwrap().unwrap();
        command.threadgroup_memory_length(37).unwrap()
    };
    let (vulkan, b) = reads::<VulkanPlatform>(&dev());
    assert_eq!(vulkan, [16, 20, 24, 0, 0, 0, 0, 0, 0, 16, 20]);
    *crate::answer::LAST_LINE.lock() = None;
    let site = catch_death(|| length(&b)).unwrap_err();
    let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
    assert_eq!(
        (site.kind, site.file),
        (DeathKind::Abort, "getThreadgroupMemoryLengthAtIndex:")
    );
    let kind = crate::answer::Inexact::IcbCopiedLength;
    assert!(line.contains(kind.code()), "{line}");
    assert_eq!(crate::answer::relaxed(|| length(&b)), 0);
    if let Some(md) = MetalPlatform::system_default_device() {
        let (metal, b) = reads::<MetalPlatform>(&md);
        assert_eq!(metal, vulkan);
        let command = b.compute_command(1).unwrap().unwrap();
        assert_eq!(command.threadgroup_memory_length(37).unwrap(), 24);
    }
}

/// A threadgroup memory length at an index of 38 or more is a later command's, 38 a command
/// whatever the descriptor's count (2 here), one past the last command's is kept apart from
/// every command's, and an index past 32 bits is its low 32 bits', at the setter and the getter,
/// as on Metal
#[test]
fn lengths_cross_into_later_commands_as_on_metal() {
    fn reads<P: Platform>(d: &P::Device) -> Vec<u64> {
        let icb = make::<P>(d, &compute_desc(), 3);
        let command = |i| icb.compute_command(i).unwrap().unwrap();
        for (i, length, index) in [
            (0, 16, 40),
            (1, 32, 1),
            (1, 48, 50),
            (2, 64, 45),
            (0, 80, (1 << 32) + 5),
        ] {
            command(i)
                .encode(IcbComputeCmd::SetThreadgroupMemoryLength { length, index })
                .unwrap();
        }
        let at = [
            (1, 2),
            (0, 40),
            (0, 39),
            (2, 12),
            (1, 50),
            (2, 45),
            (0, 121),
            (1, 83),
            (0, 5),
            (0, (1 << 32) + 5),
            (2, 7),
        ];
        (at.iter())
            .map(|&(i, index)| command(i).threadgroup_memory_length(index).unwrap())
            .collect()
    }
    let vulkan = reads::<VulkanPlatform>(&dev());
    assert_eq!(vulkan, [16, 16, 32, 48, 48, 64, 64, 64, 80, 80, 0]);
    if let Some(md) = MetalPlatform::system_default_device() {
        assert_eq!(reads::<MetalPlatform>(&md), vulkan);
    }
}

/// A blit's copy carries the threadgroup memory lengths a command holds, the record's and the 7
/// past them, a length of an index of 38 or more among them as the later command's, and a
/// blit's reset clears them, while a length past the last command's that a copy or a reset of
/// another command does not reach stays, as on Metal. A copy into the last command writes its
/// pipeline words past it, which Metal's keeps where the source's are 0 (`Tail`), though those
/// read here answer as Metal's: a getter of one is relayed here, a death in strict mode, until
/// core sets it again; a reset of the command on the CPU zeroes them, as Metal's does
#[test]
fn blits_carry_the_lengths_a_command_holds_as_on_metal() {
    fn length<P: Platform>(icb: &P::Icb, i: u64, index: u64) -> u64 {
        let command = icb.compute_command(i).unwrap().unwrap();
        command.threadgroup_memory_length(index).unwrap()
    }
    fn set<P: Platform>(icb: &P::Icb, i: u64, length: u64, index: u64) {
        let command = icb.compute_command(i).unwrap().unwrap();
        command
            .encode(IcbComputeCmd::SetThreadgroupMemoryLength { length, index })
            .unwrap();
    }
    /// What each step answers, and the two buffers, for the steps after it.
    fn reads<P: Platform>(d: &P::Device) -> (Vec<u64>, [P::Icb; 2]) {
        let (a, b) = (
            make::<P>(d, &compute_desc(), 2),
            make::<P>(d, &compute_desc(), 2),
        );
        for (i, length, index) in [
            (0, 16, 3),
            (0, 32, 33),
            (0, 48, 40),
            (1, 64, 35),
            (1, 80, 40),
        ] {
            set::<P>(&a, i, length, index);
        }
        set::<P>(&b, 1, 90, 41);
        let reset = |icb, at| BlitCmd::ResetCommandsInBuffer {
            buffer: icb,
            range: NsRange::new(at, 1),
        };
        for cmds in [vec![copy_to(&a, 1, &b, 0)], vec![reset(&a, 0)]] {
            assert_eq!(blits::<P>(d, cmds).status().0, COMPLETED);
        }
        let mut read = vec![
            length::<P>(&a, 0, 3),
            length::<P>(&a, 0, 33),
            length::<P>(&a, 0, 40),
            length::<P>(&a, 1, 35),
            length::<P>(&a, 1, 40),
            length::<P>(&b, 0, 3),
            length::<P>(&b, 0, 33),
            length::<P>(&b, 0, 40),
            length::<P>(&b, 1, 41),
        ];
        let last = BlitCmd::CopyIndirectCommandBuffer {
            source: &a,
            source_range: NsRange::new(1, 1),
            destination: &b,
            destination_index: 1,
        };
        assert_eq!(blits::<P>(d, vec![last]).status().0, COMPLETED);
        read.extend([length::<P>(&b, 1, 2), length::<P>(&b, 1, 35)]);
        (read, [a, b])
    }
    let d = dev();
    let (vulkan, [a, b]) = reads::<VulkanPlatform>(&d);
    assert_eq!(vulkan, [0, 0, 48, 64, 80, 16, 32, 0, 90, 48, 64]);
    let kind = crate::answer::Inexact::IcbLengthPastTheLastCommand;
    let unfollowed = |icb: &VulkanIcb, i, index| {
        *crate::answer::LAST_LINE.lock() = None;
        let site = catch_death(|| length::<VulkanPlatform>(icb, i, index)).unwrap_err();
        let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
        assert_eq!(
            (site.kind, site.file),
            (DeathKind::Abort, "getThreadgroupMemoryLengthAtIndex:")
        );
        assert!(line.contains(kind.code()), "{line}");
        crate::answer::relaxed(|| length::<VulkanPlatform>(icb, i, index))
    };
    let tail = [unfollowed(&b, 1, 40), unfollowed(&b, 1, 41)];
    assert_eq!(tail, [80, 0]);
    assert_eq!(length::<VulkanPlatform>(&a, 1, 40), 80);
    a.compute_command(1)
        .unwrap()
        .unwrap()
        .encode(IcbComputeCmd::Reset)
        .unwrap();
    let reset = length::<VulkanPlatform>(&a, 1, 40);
    assert_eq!(reset, 0);
    set::<VulkanPlatform>(&b, 1, 7, 41);
    set::<VulkanPlatform>(&a, 1, 5, 40);
    let again = [
        length::<VulkanPlatform>(&b, 1, 41),
        length::<VulkanPlatform>(&a, 1, 40),
    ];
    assert_eq!(again, [7, 5]);
    assert_eq!(unfollowed(&b, 1, 40), 80);
    let Some(md) = MetalPlatform::system_default_device() else {
        return;
    };
    let (metal, [a, b]) = reads::<MetalPlatform>(&md);
    assert_eq!(metal, vulkan);
    let tail_on_metal = [
        length::<MetalPlatform>(&b, 1, 40),
        length::<MetalPlatform>(&b, 1, 41),
    ];
    assert_eq!(tail_on_metal, tail);
    a.compute_command(1)
        .unwrap()
        .unwrap()
        .encode(IcbComputeCmd::Reset)
        .unwrap();
    assert_eq!(length::<MetalPlatform>(&a, 1, 40), reset);
    set::<MetalPlatform>(&b, 1, 7, 41);
    set::<MetalPlatform>(&a, 1, 5, 40);
    let again_on_metal = [
        length::<MetalPlatform>(&b, 1, 41),
        length::<MetalPlatform>(&a, 1, 40),
    ];
    assert_eq!(again_on_metal, again);
    assert_eq!(length::<MetalPlatform>(&b, 1, 40), 80);
}

/// A buffer set at an offset past 32 bits answers the buffer's address plus the offset's low
/// 32 bits, of each stage and form, as on Metal
#[test]
fn offsets_keep_their_low_32_bits_as_on_metal() {
    fn reads<P: Platform>(d: &P::Device) -> [u64; 5] {
        let (r, k) = (
            make::<P>(d, &render_desc(), 1),
            make::<P>(d, &compute_desc(), 1),
        );
        let b = buffer::<P>(d);
        let base = b.gpu_address();
        let (rc, kc) = (
            r.render_command(0).unwrap().unwrap(),
            k.compute_command(0).unwrap().unwrap(),
        );
        let offset = (1 << 40) + 16;
        for cmd in [
            IcbRenderCmd::SetVertexBuffer {
                buffer: Some(&b),
                offset,
                index: 0,
            },
            IcbRenderCmd::SetVertexBufferAttributeStride {
                buffer: Some(&b),
                offset: offset + 16,
                stride: 4,
                index: 1,
            },
            IcbRenderCmd::SetFragmentBuffer {
                buffer: Some(&b),
                offset,
                index: 0,
            },
        ] {
            rc.encode(cmd).unwrap();
        }
        for cmd in [
            IcbComputeCmd::SetKernelBuffer {
                buffer: Some(&b),
                offset,
                index: 0,
            },
            IcbComputeCmd::SetKernelBufferAttributeStride {
                buffer: Some(&b),
                offset: offset + 16,
                stride: 4,
                index: 1,
            },
        ] {
            kc.encode(cmd).unwrap();
        }
        [
            rc.vertex_buffer_address(0).unwrap(),
            rc.vertex_buffer_address(1).unwrap(),
            rc.fragment_buffer_address(0).unwrap(),
            kc.kernel_buffer_address(0).unwrap(),
            kc.kernel_buffer_address(1).unwrap(),
        ]
        .map(|a| a.wrapping_sub(base))
    }
    let vulkan = reads::<VulkanPlatform>(&dev());
    assert_eq!(vulkan, [16, 32, 16, 16, 32]);
    if let Some(md) = MetalPlatform::system_default_device() {
        assert_eq!(reads::<MetalPlatform>(&md), vulkan);
    }
}

/// For a compute and then a render buffer, whose command 0 a kernel encoded with the second
/// of two states made from one descriptor: whether command 0 answers that state's identifier,
/// whether it answers the first's, and what command 1, which nothing encoded, answers.
fn encoded_by_a_kernel<P: Platform>(d: &P::Device) -> [(bool, bool, u64); 2] {
    use reims_vgpu_core::platform::resource::{ComputePipeline as _, RenderPipeline as _};
    let library = d.new_library(ICB_ENCODE).unwrap().unwrap().unwrap();
    let (computes, renders) = twin_states::<P>(d, &library);
    let compute = make::<P>(d, &compute_desc(), 2);
    encode_by_kernel::<P>(
        d,
        &library,
        "set_compute_state",
        &compute,
        computes[1].gpu_resource_id(),
    );
    let answer = |i| {
        compute
            .compute_command(i)
            .unwrap()
            .unwrap()
            .pipeline_state_unique_identifier()
    };
    let (first, second) = (
        computes[0].unique_identifier(),
        computes[1].unique_identifier(),
    );
    let of_compute = (answer(0) == second, answer(0) == first, answer(1));
    let render = make::<P>(d, &render_desc(), 2);
    encode_by_kernel::<P>(
        d,
        &library,
        "set_render_state",
        &render,
        renders[1].gpu_resource_id(),
    );
    let answer = |i| {
        render
            .render_command(i)
            .unwrap()
            .unwrap()
            .pipeline_state_unique_identifier()
    };
    let (first, second) = (
        renders[0].unique_identifier(),
        renders[1].unique_identifier(),
    );
    [
        of_compute,
        (answer(0) == second, answer(0) == first, answer(1)),
    ]
}

/// A command a kernel encoded with a pipeline state it read from an argument buffer, which no
/// setter gave the command, answers that state's identifier, not its twin's made from the same
/// descriptor, of a compute and of a render command; a command nothing encoded answers 0. As
/// Metal answers on the host
#[test]
fn a_command_a_kernel_encoded_answers_its_states_identifier() {
    let vulkan = encoded_by_a_kernel::<VulkanPlatform>(&dev());
    assert_eq!(vulkan, [(true, false, 0); 2]);
    if let Some(md) = MetalPlatform::system_default_device() {
        assert_eq!(encoded_by_a_kernel::<MetalPlatform>(&md), vulkan);
    }
}

/// A compute buffer of 2 commands whose command 0 a kernel encoded with a state that, with
/// every handle to it, was released since, and the identifier command 0 answered before.
fn released_after_encoding<P: Platform>(d: &P::Device) -> (P::Icb, u64) {
    use reims_vgpu_core::platform::resource::ComputePipeline as _;
    let library = d.new_library(ICB_ENCODE).unwrap().unwrap().unwrap();
    let (computes, renders) = twin_states::<P>(d, &library);
    drop(renders);
    let icb = make::<P>(d, &compute_desc(), 2);
    let id = computes[1].gpu_resource_id();
    encode_by_kernel::<P>(d, &library, "set_compute_state", &icb, id);
    let before = icb
        .compute_command(0)
        .unwrap()
        .unwrap()
        .pipeline_state_unique_identifier();
    assert_eq!(before, computes[1].unique_identifier());
    drop(computes);
    (icb, before)
}

/// A command a kernel encoded with a state since released answers the released state's
/// identifier, and one nothing encoded answers 0, as Metal's do; so does a command of a
/// buffer the GPU never wrote
#[test]
fn a_command_whose_state_was_released_answers_as_metals() {
    fn answers<P: Platform>(d: &P::Device) -> (u64, u64, u64, u64) {
        let (icb, before) = released_after_encoding::<P>(d);
        let answer = |icb: &P::Icb, i| {
            icb.compute_command(i)
                .unwrap()
                .unwrap()
                .pipeline_state_unique_identifier()
        };
        let fresh = make::<P>(d, &compute_desc(), 1);
        (before, answer(&icb, 0), answer(&icb, 1), answer(&fresh, 0))
    }
    let (before, released, none, fresh) = answers::<VulkanPlatform>(&dev());
    assert_eq!((released, none, fresh), (before, 0, 0));
    if let Some(md) = MetalPlatform::system_default_device() {
        let (before, released, none, fresh) = answers::<MetalPlatform>(&md);
        assert_eq!((released, none, fresh), (before, 0, 0));
    }
}

/// Core's map of identifiers to the guest's references, as core builds it: each state's
/// identifier registered as it is made (`made`, in making order), the newest registration of
/// one winning. The index in `made` of the state `answer` reaches, if any.
fn reached(made: &[u64], answer: u64) -> Option<usize> {
    made.iter().rposition(|&identifier| identifier == answer)
}

/// Each of `twin_states`' states' identifiers, in the order it made them
fn made_by<P: Platform>(
    (computes, renders): &(Vec<P::ComputePipeline>, Vec<P::RenderPipeline>),
) -> Vec<u64> {
    use reims_vgpu_core::platform::resource::{ComputePipeline as _, RenderPipeline as _};
    let computes = computes.iter().map(|state| state.unique_identifier());
    computes
        .chain(renders.iter().map(|state| state.unique_identifier()))
        .collect()
}

/// A command a kernel encoded with a state since released, whose resource ID a state made
/// later took: Metal answers the released state's identifier, which the later state took too
/// (measured), so core's map gives the guest the later state's reference. The runtime reuses
/// the released state's identifier as Metal does, so the guest reaches the later state here too
#[test]
fn a_command_naming_a_released_state_reaches_a_later_one_as_metals() {
    /// What the guest reaches by the command's answer, in making order, the index of the
    /// released state and whether a later state took its ID; `None` where the getter died
    fn reaches<P: Platform>(d: &P::Device) -> (Option<Option<usize>>, usize, bool) {
        use reims_vgpu_core::platform::resource::ComputePipeline as _;
        let library = d.new_library(ICB_ENCODE).unwrap().unwrap().unwrap();
        let mut first = twin_states::<P>(d, &library);
        let mut made = made_by::<P>(&first);
        let icb = make::<P>(d, &compute_desc(), 1);
        let id = first.0[1].gpu_resource_id();
        encode_by_kernel::<P>(d, &library, "set_compute_state", &icb, id);
        first.0.truncate(1);
        let later = twin_states::<P>(d, &library);
        made.extend(made_by::<P>(&later));
        let reused = later.0.iter().any(|state| state.gpu_resource_id() == id);
        let answer = catch_death(|| {
            icb.compute_command(0)
                .unwrap()
                .unwrap()
                .pipeline_state_unique_identifier()
        });
        (answer.ok().map(|answer| reached(&made, answer)), 1, reused)
    }
    let (reached_here, released, reused) = reaches::<VulkanPlatform>(&dev());
    assert!(reused && reached_here.flatten().unwrap() > released);
    if let Some(md) = MetalPlatform::system_default_device() {
        let (reached_there, released, _) = reaches::<MetalPlatform>(&md);
        let reached_there = reached_there.flatten().unwrap();
        assert!(reached_there > released);
    }
}

/// The first word of a buffer of zeros an execution of a command writes on `d` that the CPU
/// encoded with a state of `mark` and the buffer, after the program released the state and
/// before any other was made: 1 where the released state's program ran.
fn released_state_runs<P: Platform>(d: &P::Device) -> u32 {
    let library = d.new_library(ICB_ENCODE).unwrap().unwrap().unwrap();
    let state = marking_state::<P>(d, &library);
    let out = d
        .new_buffer_of_length(OnFault::Return, 16, 0)
        .unwrap()
        .unwrap();
    let icb = make::<P>(d, &compute_desc(), 1);
    let command = icb.compute_command(0).unwrap().unwrap();
    for cmd in [
        IcbComputeCmd::SetComputePipelineState {
            pipeline_state: Some(&state),
        },
        IcbComputeCmd::SetKernelBuffer {
            buffer: Some(&out),
            offset: 0,
            index: 0,
        },
        IcbComputeCmd::ConcurrentDispatchThreads {
            threads_per_grid: MtlSize::new(1, 1, 1),
            threads_per_threadgroup: MtlSize::new(1, 1, 1),
        },
    ] {
        command.encode(cmd).unwrap();
    }
    drop(state);
    execute_writing::<P>(d, &icb, &out);
    // SAFETY: the buffer's first Shared word, read after the execution completed.
    unsafe { *out.contents().as_ptr::<u32>() }
}

/// An execution of a command the CPU encoded with a state the program released since runs
/// that state's program, as Metal's does on this host, though the state's ID went back
#[test]
fn a_released_state_the_cpu_set_runs_as_metals() {
    let ours = released_state_runs::<VulkanPlatform>(&dev());
    assert_eq!(ours, 1);
    if let Some(md) = MetalPlatform::system_default_device() {
        assert_eq!(released_state_runs::<MetalPlatform>(&md), ours);
    }
}

/// A command core set a state on, which the guest then released before states were made:
/// Metal's state made later takes the released one's resource ID and its identifier too,
/// which the command answers (measured, M3 Max), so core's map gives the guest the later
/// state's reference, as the runtime's reuse of the identifier gives it here. The runtime's
/// command names the state without holding it, so the later state takes its ID here too. A
/// state a kernel encodes on the command after is answered
#[test]
fn a_command_core_set_reaches_a_later_state_holding_its_identifier_as_metals() {
    use reims_vgpu_core::platform::resource::ComputePipeline as _;
    /// The command, what the guest reaches by its answer once the first was released and
    /// states were made after it (in making order; `None` where the getter died), and
    /// whether one of those took its ID
    fn set_then_released<P: Platform>(d: &P::Device) -> (P::Icb, Option<Option<usize>>, bool) {
        let library = d.new_library(ICB_ENCODE).unwrap().unwrap().unwrap();
        let mut first = twin_states::<P>(d, &library);
        let mut made = made_by::<P>(&first);
        let icb = make::<P>(d, &compute_desc(), 1);
        let command = icb.compute_command(0).unwrap().unwrap();
        command
            .encode(IcbComputeCmd::SetComputePipelineState {
                pipeline_state: Some(&first.0[1]),
            })
            .unwrap();
        let id = first.0[1].gpu_resource_id();
        first.0.truncate(1);
        let later = twin_states::<P>(d, &library);
        made.extend(made_by::<P>(&later));
        let taken = later.0.iter().any(|state| state.gpu_resource_id() == id);
        let answer = catch_death(|| command.pipeline_state_unique_identifier());
        (icb, answer.ok().map(|answer| reached(&made, answer)), taken)
    }
    let d = dev();
    let (icb, reached_here, taken) = set_then_released::<VulkanPlatform>(&d);
    assert!(taken && reached_here.flatten().unwrap() > 1);
    let library = d.new_library(ICB_ENCODE).unwrap().unwrap().unwrap();
    let (states, renders) = twin_states::<VulkanPlatform>(&d, &library);
    encode_by_kernel::<VulkanPlatform>(
        &d,
        &library,
        "set_compute_state",
        &icb,
        states[0].gpu_resource_id(),
    );
    let command = icb.compute_command(0).unwrap().unwrap();
    assert_eq!(
        command.pipeline_state_unique_identifier(),
        states[0].unique_identifier()
    );
    drop(renders);
    if let Some(md) = MetalPlatform::system_default_device() {
        let (_, reached_there, taken) = set_then_released::<MetalPlatform>(&md);
        assert!(taken && reached_there.flatten().unwrap() > 1);
    }
}

/// An ICB-capable state of `name` in `library`, made from a whole value, as `twin_states`' are
fn icb_kernel<P: Platform>(d: &P::Device, library: &P::Library, name: &str) -> P::ComputePipeline {
    use reims_vgpu_core::platform::resource::Library as _;
    use reims_vgpu_core::platform::values::FullComputePipelineDesc;
    let kernel = library
        .new_function(OnFault::Return, name)
        .unwrap()
        .unwrap();
    d.new_compute_pipeline_state_full(&FullComputePipelineDesc {
        support_indirect_command_buffers: true,
        ..FullComputePipelineDesc::new(&kernel)
    })
    .unwrap()
    .unwrap()
}

/// A command a kernel wrote that an execution of another buffer ran, in an encoder that made
/// the written buffer resident for writing, answers the state the kernel wrote, which took the
/// resource ID of one released before, and core's map reaches it, as Metal's does: the
/// execution opens a span on the buffer, which tells the two states of the ID apart
#[test]
fn a_command_an_executed_kernel_wrote_answers_the_state_holding_its_id_then() {
    use reims_vgpu_core::platform::resource::ComputePipeline as _;
    /// The index, in making order, of the state that took a released one's ID and that the
    /// kernel wrote, and of the state core's map reaches by the command's answer
    fn written_by_an_execution<P: Platform>(d: &P::Device) -> (usize, Option<usize>) {
        let library = d.new_library(ICB_ENCODE).unwrap().unwrap().unwrap();
        let mut first = twin_states::<P>(d, &library);
        let mut made = made_by::<P>(&first);
        let id = first.0[1].gpu_resource_id();
        first.0.truncate(1);
        let later = twin_states::<P>(d, &library);
        made.extend(made_by::<P>(&later));
        let taker = later
            .0
            .iter()
            .position(|state| state.gpu_resource_id() == id);
        let taker = taker.expect("a later state takes the released one's ID");
        let kernel = icb_kernel::<P>(d, &library, "set_compute_state");
        let (written, runs) = (
            make::<P>(d, &compute_desc(), 1),
            make::<P>(d, &compute_desc(), 1),
        );
        let held = d
            .new_buffer_of_length(OnFault::Return, 16, 0)
            .unwrap()
            .unwrap();
        let words = [written.gpu_resource_id(), id];
        // SAFETY: the buffer's 16 Shared bytes, written before any command uses them.
        unsafe {
            std::ptr::copy_nonoverlapping(words.as_ptr(), held.contents().as_ptr::<u64>(), 2)
        };
        let command = runs.compute_command(0).unwrap().unwrap();
        for cmd in [
            IcbComputeCmd::SetComputePipelineState {
                pipeline_state: Some(&kernel),
            },
            IcbComputeCmd::SetKernelBuffer {
                buffer: Some(&held),
                offset: 0,
                index: 0,
            },
            IcbComputeCmd::ConcurrentDispatchThreads {
                threads_per_grid: MtlSize::new(1, 1, 1),
                threads_per_threadgroup: MtlSize::new(1, 1, 1),
            },
        ] {
            command.encode(cmd).unwrap();
        }
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
        let e = cb.compute_encoder(OnFault::Return).unwrap().unwrap();
        let (writes, reads) = (
            [Some(ResourceRef::Icb(&written))],
            [
                Some(ResourceRef::Buffer(&held)),
                Some(ResourceRef::Icb(&runs)),
            ],
        );
        for cmd in [
            ComputeCmd::UseResources {
                resources: &writes,
                usage: 3,
            },
            ComputeCmd::UseResources {
                resources: &reads,
                usage: 1,
            },
            ComputeCmd::ExecuteCommandsInBuffer {
                indirect_command_buffer: &runs,
                execution_range: NsRange::new(0, 1),
            },
            ComputeCmd::EndEncoding,
        ] {
            e.encode(OnFault::Return, cmd).unwrap();
        }
        cb.commit(OnFault::Return).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        assert_eq!((cb.status().0, cb.error()), (COMPLETED, None));
        let answer = written
            .compute_command(0)
            .unwrap()
            .unwrap()
            .pipeline_state_unique_identifier();
        drop((first, later));
        (made.len() / 2 + taker, reached(&made, answer))
    }
    let (taker, reached_here) = written_by_an_execution::<VulkanPlatform>(&dev());
    assert_eq!(reached_here, Some(taker));
    if let Some(md) = MetalPlatform::system_default_device() {
        let (taker, reached_there) = written_by_an_execution::<MetalPlatform>(&md);
        assert_eq!(reached_there, Some(taker));
    }
}

/// The guest's read of a state's identifier (info command 460) answers the runtime's for every
/// state, made with indirect command buffer support or not. Metal's identifier of a state made
/// with it is an address of its own allocation of them, `0x1000000000` and up, the same in
/// every process that makes the same states in the same order in one environment (here two
/// child processes, each making [`twin_states`]); the runtime's is in that range, at offsets
/// that are not Metal's, so the read is relayed (`inexact-unique-identifier`), dying in strict
/// mode, and a render state made after one was released takes the released one's, the last
/// released first, as Metal's does
#[test]
fn an_icb_capable_states_identifier_is_the_runtimes() {
    use reims_vgpu_core::platform::resource::{
        ComputePipeline as _, Library as _, RenderPipeline as _,
    };
    const TEST: &str =
        "gpu::icb::tests::metal_tests::an_icb_capable_states_identifier_is_the_runtimes";
    if std::env::var("RVG_CHILD").as_deref() == Ok(TEST) {
        let md = MetalPlatform::system_default_device().unwrap();
        let library = md.new_library(ICB_ENCODE).unwrap().unwrap().unwrap();
        let (computes, renders) = twin_states::<MetalPlatform>(&md, &library);
        let computes = computes.iter().map(|c| c.unique_identifier());
        for identifier in computes.chain(renders.iter().map(|r| r.unique_identifier())) {
            println!("identifier {identifier}");
        }
        return;
    }
    let d = dev();
    let library = d.new_library(ICB_ENCODE).unwrap().unwrap().unwrap();
    let (computes, mut renders) = twin_states::<VulkanPlatform>(&d, &library);
    let read = |read: &dyn Fn() -> u64| crate::answer::relaxed(read);
    let computes_read = computes
        .iter()
        .map(|c| (read(&|| c.guest_unique_identifier()), c.unique_identifier()));
    let renders_read = renders
        .iter()
        .map(|r| (read(&|| r.guest_unique_identifier()), r.unique_identifier()));
    for (read, identifier) in computes_read.chain(renders_read) {
        assert_eq!(read, identifier);
        assert!(identifier >= 1 << 36, "{identifier:#x}");
    }
    let dies_in_strict_mode = |read: &dyn Fn() -> u64| {
        *crate::answer::LAST_LINE.lock() = None;
        let strict = catch_death(read);
        let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
        assert!(
            strict.is_err() && line.contains("(inexact-unique-identifier)"),
            "{strict:?}: {line}"
        );
    };
    dies_in_strict_mode(&|| computes[0].guest_unique_identifier());
    dies_in_strict_mode(&|| renders[0].guest_unique_identifier());
    let released = renders[1].unique_identifier();
    renders.truncate(1);
    let (_later_computes, later_renders) = twin_states::<VulkanPlatform>(&d, &library);
    assert_eq!(
        read(&|| later_renders[0].guest_unique_identifier()),
        released
    );
    let f = library
        .new_function(OnFault::Return, "nothing")
        .unwrap()
        .unwrap();
    let plain = d
        .new_compute_pipeline_state_for_function(Some(&f))
        .unwrap()
        .unwrap();
    assert_eq!(plain.guest_unique_identifier(), plain.unique_identifier());
    if MetalPlatform::system_default_device().is_none() {
        return;
    }
    let in_a_child = || {
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", TEST, "--nocapture", "--test-threads=1"])
            .env("RVG_CHILD", TEST)
            .output()
            .unwrap();
        assert!(out.status.success(), "{out:?}");
        let stdout = String::from_utf8(out.stdout).unwrap();
        // The harness's line of the test's name runs on into the first.
        let identifiers = stdout.lines().filter_map(|l| l.split_once("identifier "));
        identifiers
            .map(|(_, i)| i.parse().unwrap())
            .collect::<Vec<u64>>()
    };
    let (first, second) = (in_a_child(), in_a_child());
    assert_eq!(first.len(), 4, "{first:x?}");
    assert_eq!(first, second, "{first:x?}, {second:x?}");
    let allocated = |&i: &u64| i >= 0x10_0000_0000 && i % 0x40 == 0;
    assert!(first.iter().all(allocated), "{first:x?}");
}

/// A copy past the destination's end, in the shapes measured (`n,to,k`: a destination of
/// `n` commands, `k` copied to index `to`), lands the commands that fit as Metal's does and
/// changes no other buffer; each command before the index, which Metal may write over in its
/// own layout (a threads dispatch read back as a threadgroups dispatch at some shapes, whole
/// at others), dies at its getter
#[test]
fn a_copy_past_the_destinations_end_lands_what_fits_as_metal() {
    let md = MetalPlatform::system_default_device();
    let shapes = [
        (false, (2, 1, 2)),
        (false, (4, 3, 2)),
        (false, (4, 1, 4)),
        (false, (8, 5, 8)),
        (false, (64, 63, 64)),
        (true, (2, 1, 2)),
        (true, (4, 3, 2)),
    ];
    for (render, (n, to, k)) in shapes {
        let shape = format!("render {render} {n},{to},{k}");
        let [a, b, c] = three_buffers::<VulkanPlatform>(&dev(), render, n);
        let cb = blits::<VulkanPlatform>(&dev(), vec![copy_to(&a, k, &b, to)]);
        assert_eq!(cb.status().0, COMPLETED, "{shape}");
        let landed: Vec<_> = (to..n)
            .map(|i| kind_and_width::<VulkanPlatform>(&b, render, i))
            .collect();
        for i in 0..to {
            let site = catch_death(|| kind_and_width::<VulkanPlatform>(&b, render, i));
            let site = site.unwrap_err();
            assert_eq!((site.kind, site.file), (DeathKind::Abort, "getCommandType"));
        }
        let kind = if render { 0x1 } else { 0x40 };
        for i in 0..n {
            let neighbour = kind_and_width::<VulkanPlatform>(&c, render, i);
            assert_eq!(neighbour, (kind, 30 + i), "{shape}");
        }
        let Some(md) = &md else { continue };
        let [a, b, c] = three_buffers::<MetalPlatform>(md, render, n);
        let cb = blits::<MetalPlatform>(md, vec![copy_to(&a, k, &b, to)]);
        assert_eq!(cb.status().0, COMPLETED, "Metal: {shape}");
        let metals: Vec<_> = (to..n)
            .map(|i| kind_and_width::<MetalPlatform>(&b, render, i))
            .collect();
        assert_eq!(landed, metals, "{shape}");
        for i in 0..n {
            let neighbour = kind_and_width::<MetalPlatform>(&c, render, i);
            assert_eq!(neighbour, (kind, 30 + i), "Metal: {shape}");
        }
    }
}

/// A copy of 2 commands to index 1 of a buffer of 2, both of a descriptor no measurement of
/// copies past the end covers (concurrent dispatches of threadgroups alone and 1 kernel buffer,
/// as `blit_err_ranges` makes them), dies at the copy, naming the shape: where Metal's copy
/// writes past the end depends on its layout of the descriptor's records, which nothing here
/// models. Metal completes the copy, the command that fits landing, and changes no command of a
/// third buffer
#[test]
fn a_copy_past_the_end_of_an_unmeasured_descriptor_dies() {
    /// Three buffers of that descriptor, of 2 commands, command `i` of each a dispatch of
    /// `base + i` threadgroups across, `base` 10, 20 and 30.
    fn threadgroup_buffers<P: Platform>(d: &P::Device) -> [P::Icb; 3] {
        let desc = IcbDesc {
            command_types: 0x20,
            max_kernel_buffer_bind_count: 1,
            ..IcbDesc::default()
        };
        let buffers = [(); 3].map(|()| make::<P>(d, &desc, 2));
        for (icb, base) in buffers.iter().zip([10, 20, 30]) {
            for i in 0..2 {
                let dispatch = IcbComputeCmd::ConcurrentDispatchThreadgroups {
                    threadgroups_per_grid: MtlSize::new(base + i, 1, 1),
                    threads_per_threadgroup: MtlSize::new(1, 1, 1),
                };
                let command = icb.compute_command(i).unwrap().unwrap();
                command.encode(dispatch).unwrap();
            }
        }
        buffers
    }
    if let Some(md) = MetalPlatform::system_default_device() {
        let [a, b, c] = threadgroup_buffers::<MetalPlatform>(&md);
        let cb = blits::<MetalPlatform>(&md, vec![copy_to(&a, 2, &b, 1)]);
        assert_eq!(cb.status().0, COMPLETED, "Metal");
        let landed = kind_and_width::<MetalPlatform>(&b, false, 1);
        let neighbours = [0, 1].map(|i| kind_and_width::<MetalPlatform>(&c, false, i));
        assert_eq!(landed, (0x20, 10), "Metal");
        assert_eq!(neighbours, [(0x20, 30), (0x20, 31)], "Metal");
    }
    let d = dev();
    let [a, b, _] = threadgroup_buffers::<VulkanPlatform>(&d);
    *crate::answer::LAST_LINE.lock() = None;
    let site = catch_death(|| blits::<VulkanPlatform>(&d, vec![copy_to(&a, 2, &b, 1)]));
    let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
    let site = site.unwrap_err();
    let copy = "copyIndirectCommandBuffer:sourceRange:destination:destinationIndex:";
    assert_eq!((site.kind, site.file), (DeathKind::Abort, copy), "{line}");
    assert!(line.contains("in a shape nothing measured"), "{line}");
}

/// What Metal answers and runs of a command its copy past the end wrote over, once the program
/// encodes it again with every setter (its pipeline state, both kernel buffers, both
/// threadgroup memory lengths, then a threads dispatch): from a copy of 2 commands to index 3
/// of a buffer of 4, command 1 encoded again and executed alone, writing the first word of a
/// buffer of zeros. A measurement: the command buffer's status, the word, and the command's
/// type and threads across.
#[test]
#[ignore = "a measurement, on Metal"]
fn metal_reencodes_an_overwritten_command_here() {
    let d = MetalPlatform::system_default_device().unwrap();
    let library = d.new_library(ICB_ENCODE).unwrap().unwrap().unwrap();
    let state = marking_state::<MetalPlatform>(&d, &library);
    let [a, b, _] = three_buffers::<MetalPlatform>(&d, false, 4);
    blits::<MetalPlatform>(&d, vec![copy_to(&a, 2, &b, 3)]);
    let before = kind_and_width::<MetalPlatform>(&b, false, 1);
    let out = d
        .new_buffer_of_length(OnFault::Return, 16, 0)
        .unwrap()
        .unwrap();
    let spare = buffer::<MetalPlatform>(&d);
    let command = b.compute_command(1).unwrap().unwrap();
    for cmd in [
        IcbComputeCmd::SetComputePipelineState {
            pipeline_state: Some(&state),
        },
        IcbComputeCmd::SetKernelBuffer {
            buffer: Some(&out),
            offset: 0,
            index: 0,
        },
        IcbComputeCmd::SetKernelBuffer {
            buffer: Some(&spare),
            offset: 0,
            index: 1,
        },
        IcbComputeCmd::SetThreadgroupMemoryLength {
            length: 0,
            index: 0,
        },
        IcbComputeCmd::SetThreadgroupMemoryLength {
            length: 0,
            index: 1,
        },
        IcbComputeCmd::ConcurrentDispatchThreads {
            threads_per_grid: MtlSize::new(1, 1, 1),
            threads_per_threadgroup: MtlSize::new(1, 1, 1),
        },
    ] {
        command.encode(cmd).unwrap();
    }
    let after = kind_and_width::<MetalPlatform>(&b, false, 1);
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
    let e = cb.compute_encoder(OnFault::Return).unwrap().unwrap();
    for cmd in [
        ComputeCmd::UseResources {
            resources: &[Some(ResourceRef::Icb(&b))],
            usage: 1,
        },
        ComputeCmd::UseResources {
            resources: &[
                Some(ResourceRef::Buffer(&out)),
                Some(ResourceRef::Buffer(&spare)),
            ],
            usage: 3,
        },
        ComputeCmd::ExecuteCommandsInBuffer {
            indirect_command_buffer: &b,
            execution_range: NsRange::new(1, 1),
        },
        ComputeCmd::EndEncoding,
    ] {
        e.encode(OnFault::Return, cmd).unwrap();
    }
    cb.commit(OnFault::Return).unwrap();
    cb.wait_until_completed(OnFault::Return).unwrap();
    // SAFETY: a Shared buffer of 16 bytes, read after the GPU ran.
    let word = unsafe { out.contents().as_ptr::<u32>().read() };
    println!(
        "metal: before {before:x?} after {after:x?} status {} error {:?} word {word}",
        cb.status().0,
        cb.error().map(|e| e.code)
    );
}

/// What Metal's copy of `k` commands into index `to` of a buffer of `n` (`render`: of draws,
/// else of threads dispatches) writes past the destination's end: each command of the
/// destination and of a neighbour made after it, as `type:width` (a dispatch's threads or
/// threadgroups across, a draw's vertex count), after the copy completed. The source's
/// commands are of width 10 and up, the destination's 20 and up, the neighbour's 30 and up.
fn past_end_on<P: Platform>(d: &P::Device, render: bool, (n, to, k): (u64, u64, u64)) -> String {
    let desc = if render {
        render_desc()
    } else {
        compute_desc()
    };
    let (source, destination, neighbour) = (
        make::<P>(d, &desc, n),
        make::<P>(d, &desc, n),
        make::<P>(d, &desc, n),
    );
    for (icb, base) in [(&source, 10), (&destination, 20), (&neighbour, 30)] {
        for i in 0..n {
            if render {
                icb.render_command(i)
                    .unwrap()
                    .unwrap()
                    .encode(IcbRenderCmd::DrawPrimitives {
                        primitive_type: 3,
                        vertex_start: 0,
                        vertex_count: base + i,
                        instance_count: 1,
                        base_instance: 0,
                    })
                    .unwrap();
            } else {
                icb.compute_command(i)
                    .unwrap()
                    .unwrap()
                    .encode(IcbComputeCmd::ConcurrentDispatchThreads {
                        threads_per_grid: MtlSize::new(base + i, 1, 1),
                        threads_per_threadgroup: MtlSize::new(1, 1, 1),
                    })
                    .unwrap();
            }
        }
    }
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
    let blit = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
    blit.encode(
        OnFault::Return,
        BlitCmd::CopyIndirectCommandBuffer {
            source: &source,
            source_range: NsRange::new(0, k),
            destination: &destination,
            destination_index: to,
        },
    )
    .unwrap();
    blit.encode(OnFault::Return, BlitCmd::EndEncoding).unwrap();
    cb.commit(OnFault::Return).unwrap();
    cb.wait_until_completed(OnFault::Return).unwrap();
    let read = |icb: &P::Icb| {
        (0..n)
            .map(|i| {
                if render {
                    let c = icb.render_command(i).unwrap().unwrap();
                    let count = c.draw_args().map_or(u64::MAX, |a| a.vertex_count);
                    format!("{:#x}:{count}", c.command_type())
                } else {
                    let c = icb.compute_command(i).unwrap().unwrap();
                    let width = match c.command_type() {
                        0x40 => c.dispatch_threads_args().map(|a| a.threads_per_grid.width),
                        _ => c
                            .dispatch_threadgroups_args()
                            .map(|a| a.threadgroups_per_grid.width),
                    };
                    format!("{:#x}:{}", c.command_type(), width.unwrap_or(u64::MAX))
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    };
    format!(
        "status {} destination [{}] neighbour [{}]",
        cb.status().0,
        read(&destination),
        read(&neighbour)
    )
}

/// Metal's copy past a destination's end, one shape a process from `RVG_SHAPE`
/// (`compute` or `render`, then `n,to,k`). A measurement.
#[test]
#[ignore = "a measurement: one shape a process, from the environment"]
fn metal_copies_past_the_destinations_end_here() {
    let shape = std::env::var("RVG_SHAPE").unwrap();
    let (kind, numbers) = shape.split_once(' ').unwrap();
    let n: Vec<u64> = numbers.split(',').map(|v| v.parse().unwrap()).collect();
    let md = MetalPlatform::system_default_device().unwrap();
    let seen = past_end_on::<MetalPlatform>(&md, kind == "render", (n[0], n[1], n[2]));
    println!("metal {shape}: {seen}");
}

/// Whether Metal gives a state of indirect command buffer support, made after the program
/// released one, the released one's identifier, in the way `RVG_REUSE` names: `plain` (the
/// first only made), `icb` (set on a command first), `executed` (and that command executed,
/// its command buffer completed), `icb_gone` (executed, then the buffer released before the
/// state), `other` (the second of another kernel). A measurement, one way a process.
#[test]
#[ignore = "a measurement: one way a process, from the environment"]
fn metal_reuses_a_released_states_identifier_here() {
    use reims_vgpu_core::platform::resource::{ComputePipeline as _, Library as _};
    let how = std::env::var("RVG_REUSE").unwrap();
    let d = MetalPlatform::system_default_device().unwrap();
    let library = d.new_library(ICB_ENCODE).unwrap().unwrap().unwrap();
    let first = marking_state::<MetalPlatform>(&d, &library);
    let id1 = first.unique_identifier();
    let out = d
        .new_buffer_of_length(OnFault::Return, 16, 0)
        .unwrap()
        .unwrap();
    let mut icb = None;
    if how != "plain" && how != "other" {
        let buffer = make::<MetalPlatform>(&d, &compute_desc(), 1);
        let command = buffer.compute_command(0).unwrap().unwrap();
        for cmd in [
            IcbComputeCmd::SetComputePipelineState {
                pipeline_state: Some(&first),
            },
            IcbComputeCmd::SetKernelBuffer {
                buffer: Some(&out),
                offset: 0,
                index: 0,
            },
            IcbComputeCmd::ConcurrentDispatchThreads {
                threads_per_grid: MtlSize::new(1, 1, 1),
                threads_per_threadgroup: MtlSize::new(1, 1, 1),
            },
        ] {
            command.encode(cmd).unwrap();
        }
        if how != "icb" {
            execute_writing::<MetalPlatform>(&d, &buffer, &out);
        }
        icb = Some(buffer);
    }
    if how == "icb_gone" {
        icb = None;
    }
    drop(first);
    let second = if how == "other" {
        use reims_vgpu_core::platform::values::FullComputePipelineDesc;
        let nothing = library
            .new_function(OnFault::Return, "nothing")
            .unwrap()
            .unwrap();
        d.new_compute_pipeline_state_full(&FullComputePipelineDesc {
            support_indirect_command_buffers: true,
            ..FullComputePipelineDesc::new(&nothing)
        })
        .unwrap()
        .unwrap()
    } else {
        marking_state::<MetalPlatform>(&d, &library)
    };
    let id2 = second.unique_identifier();
    let held = icb.is_some();
    println!(
        "metal {how}: first {id1:#x} second {id2:#x} reused {} (buffer held {held})",
        id1 == id2
    );
}

/// A Shared buffer of 256 bytes.
fn buffer<P: Platform>(d: &P::Device) -> P::Buffer {
    d.new_buffer_of_length(OnFault::Return, 256, 0)
        .unwrap()
        .unwrap()
}

/// An address as `[1, offset]` into the 256 bytes from `base`, or as `[0, itself]`.
fn at(a: u64, base: u64) -> [u64; 2] {
    match a.checked_sub(base) {
        Some(off) if off < 256 => [1, off],
        _ => [0, a],
    }
}

/// The threadgroup memory lengths past the last command's of a buffer of 2 commands, each set by
/// core, read back after each of a pipeline, dispatch or imageblock setter of a command, a reset
/// of one on the CPU or by a blit, and a copy into one, of a buffer that inherits its pipeline
/// state or not: each answers as Metal's but the words Metal writes otherwise than the runtime,
/// which are relayed, a death in strict mode (`Tail`), at the places past the last command's
/// lengths: 0 to 3 the commands' pipeline words, 4 to 11 and 12 to 19 their dispatch and
/// imageblock words, 20 a copy's, and the rest up to 24 none of them
#[test]
fn lengths_past_the_last_command_follow_metals_words() {
    use reims_vgpu_core::platform::types::MtlSize;
    /// The places read, past the last command's lengths at 76.
    const PLACES: std::ops::Range<u64> = 76..101;
    /// What each place reads after `scenario` over buffers that inherit their pipeline state or
    /// not, through command 1's getter: the length, or none where the getter is relayed.
    fn read<P: Platform>(d: &P::Device, inherit: bool, scenario: u32) -> Vec<Option<u64>> {
        let desc = IcbDesc {
            inherit_pipeline_state: inherit,
            ..compute_desc()
        };
        let library = d.new_library(ICB_ENCODE).unwrap().unwrap().unwrap();
        let kernel = icb_kernel::<P>(d, &library, "nothing");
        let (a, b) = (make::<P>(d, &desc, 2), make::<P>(d, &desc, 2));
        let cmd = |icb: &P::Icb, i: u64, c: IcbComputeCmd<'_, P>| {
            icb.compute_command(i).unwrap().unwrap().encode(c).unwrap();
        };
        for place in PLACES {
            let (length, index) = (16 * (place - 70), place - 38);
            cmd(
                &a,
                1,
                IcbComputeCmd::SetThreadgroupMemoryLength { length, index },
            );
        }
        let pipeline = |icb: &P::Icb, i| {
            if !inherit {
                let pipeline_state = Some(&kernel);
                cmd(
                    icb,
                    i,
                    IcbComputeCmd::SetComputePipelineState { pipeline_state },
                );
            }
        };
        let dispatch = |icb: &P::Icb, i| {
            let dispatch = IcbComputeCmd::ConcurrentDispatchThreadgroups {
                threadgroups_per_grid: MtlSize::new(3, 5, 7),
                threads_per_threadgroup: MtlSize::new(2, 4, 1),
            };
            cmd(icb, i, dispatch);
        };
        let reset = |icb, at| BlitCmd::ResetCommandsInBuffer {
            buffer: icb,
            range: NsRange::new(at, 1),
        };
        let copy = |source, from, destination, to| BlitCmd::CopyIndirectCommandBuffer {
            source,
            source_range: NsRange::new(from, 1),
            destination,
            destination_index: to,
        };
        let ran = |cmds| assert_eq!(blits::<P>(d, cmds).status().0, COMPLETED);
        let mut of = &a;
        match scenario {
            0 => {}
            1 => pipeline(&a, 0),
            2 => dispatch(&a, 0),
            3 => cmd(
                &a,
                0,
                IcbComputeCmd::SetImageblockWidth {
                    width: 8,
                    height: 16,
                },
            ),
            4 => cmd(&a, 0, IcbComputeCmd::Reset),
            5 => ran(vec![reset(&a, 0)]),
            6 => ran(vec![copy(&b, 0, &a, 0)]),
            7 => {
                pipeline(&b, 0);
                dispatch(&b, 0);
                ran(vec![copy(&b, 0, &a, 0)]);
            }
            8 => cmd(&a, 1, IcbComputeCmd::Reset),
            9 => ran(vec![reset(&a, 1)]),
            10 => ran(vec![copy(&b, 0, &a, 1)]),
            11 => {
                pipeline(&b, 1);
                dispatch(&b, 1);
                ran(vec![copy(&b, 1, &a, 1)]);
            }
            12 => {
                pipeline(&a, 0);
                dispatch(&a, 0);
                ran(vec![copy(&a, 0, &b, 0)]);
                of = &b;
            }
            13 => {
                pipeline(&a, 0);
                dispatch(&a, 0);
                ran(vec![copy(&a, 0, &a, 1)]);
            }
            14 => {
                pipeline(&a, 0);
                cmd(&a, 0, IcbComputeCmd::Reset);
            }
            _ => {
                pipeline(&a, 0);
                dispatch(&a, 0);
                ran(vec![reset(&a, 0)]);
            }
        }
        let command = of.compute_command(1).unwrap().unwrap();
        let length = |place: u64| command.threadgroup_memory_length(place - 38).unwrap();
        let relayed = |place| {
            *crate::answer::LAST_LINE.lock() = None;
            catch_death(|| length(place)).is_err_and(|site| {
                let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
                let sel = "getThreadgroupMemoryLengthAtIndex:";
                assert_eq!((site.kind, site.file), (DeathKind::Abort, sel), "{line}");
                let kind = crate::answer::Inexact::IcbLengthPastTheLastCommand;
                assert!(line.contains(kind.code()), "{line}");
                true
            })
        };
        let on_vulkan = std::any::TypeId::of::<P>() == std::any::TypeId::of::<VulkanPlatform>();
        (PLACES.map(|place| (!(on_vulkan && relayed(place))).then(|| length(place)))).collect()
    }
    // The places, past the last command's lengths, relayed after each scenario.
    let all = |end: u64| (0..end).collect::<Vec<u64>>();
    let relayed: [[Vec<u64>; 16]; 2] = [
        [
            vec![],
            vec![0, 1],
            (4..10).collect(),
            vec![10, 11],
            vec![],
            vec![],
            vec![0, 1, 20],
            [vec![0, 1], (4..10).collect(), vec![20]].concat(),
            vec![],
            vec![],
            vec![2, 3, 20],
            [vec![2, 3], (12..18).collect(), vec![20]].concat(),
            [vec![0, 1], (4..10).collect(), vec![20]].concat(),
            [(0..10).collect(), (12..18).collect(), vec![20]].concat(),
            vec![],
            vec![],
        ],
        [
            vec![],
            vec![],
            (4..10).collect(),
            vec![10, 11],
            vec![],
            all(20),
            all(21),
            all(21),
            vec![],
            all(20),
            all(21),
            all(21),
            all(21),
            all(21),
            vec![],
            all(20),
        ],
    ];
    let d = dev();
    let md = MetalPlatform::system_default_device();
    for (inherit, relayed) in [false, true].into_iter().zip(relayed) {
        for (scenario, relayed) in (0..).zip(relayed) {
            let what = format!("inherit {inherit}, scenario {scenario}");
            let ours = read::<VulkanPlatform>(&d, inherit, scenario);
            let none: Vec<u64> = (PLACES.zip(&ours))
                .filter(|(_, length)| length.is_none())
                .map(|(place, _)| place - 76)
                .collect();
            assert_eq!(none, relayed, "{what}");
            let Some(md) = &md else { continue };
            let metal = read::<MetalPlatform>(md, inherit, scenario);
            for ((place, o), m) in PLACES.zip(ours).zip(metal) {
                if o.is_some() {
                    assert_eq!(o, m, "{what}, place {}", place - 76);
                }
            }
        }
    }
}
