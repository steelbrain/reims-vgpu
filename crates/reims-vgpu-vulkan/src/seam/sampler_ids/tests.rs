use metal2vulkan_rt::{
    ColorAttachmentDescriptor, FunctionDescriptor, Gap, Library, PixelFormat, StageSource,
};

use super::*;
use crate::seam::test_device;

/// The tests' functions whose states number a constexpr sampler, a read and a write
/// (`gpu/numbered_samplers.metal`).
const NUMBERED: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/numbered_samplers.metallib"));

/// The tests' kernel whose state numbers five constexpr samplers (`gpu/five_samplers.metal`).
const FIVE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/five_samplers.metallib"));

/// A constexpr sampler of words no kernel of the tests' has.
fn constexpr(n: u64) -> NumberedSampler {
    NumberedSampler::Constexpr([n, u64::MAX])
}

#[test]
fn the_ids_are_rts_whole_sampler_range() {
    assert_eq!(
        FIRST_SAMPLER_ID..FIRST_SAMPLER_ID + SAMPLER_ID_COUNT,
        IndexClass::Sampler.range()
    );
}

/// A sampler takes the lowest free ID, and one equal to a held one shares its ID, across states
/// and within one (a render state's vertex and fragment reads).
#[test]
fn a_sampler_takes_the_lowest_free_id_and_an_equal_one_shares_it() {
    let mut numbering = Numbering::new();
    let read = NumberedSampler::Read;
    let write = NumberedSampler::Write;
    assert_eq!(
        numbering.take(&[constexpr(1), read, write]),
        Some(vec![1, 2, 3])
    );
    assert_eq!(numbering.take(&[read, constexpr(2)]), Some(vec![2, 4]));
    assert_eq!(numbering.take(&[read, read]), Some(vec![2, 2]));
}

/// An ID is free once the last state whose samplers hold it gives it back.
#[test]
fn an_id_goes_back_with_its_last_holder() {
    let mut numbering = Numbering::new();
    let first = [constexpr(1), NumberedSampler::Read];
    let second = [NumberedSampler::Read];
    assert_eq!(numbering.take(&first), Some(vec![1, 2]));
    assert_eq!(numbering.take(&second), Some(vec![2]));
    numbering.give_back(&first);
    assert_eq!(numbering.take(&[constexpr(9)]), Some(vec![1]));
    assert_eq!(numbering.take(&[NumberedSampler::Write]), Some(vec![3]));
    numbering.give_back(&second);
    assert_eq!(numbering.take(&[constexpr(10)]), Some(vec![2]));
}

/// A state whose samplers need more IDs than are free takes none: the IDs its first samplers
/// took are free again.
#[test]
fn a_numbering_past_the_last_free_id_holds_nothing() {
    let mut numbering = Numbering::new();
    let all_but_one: Vec<_> = (0..u64::from(SAMPLER_ID_COUNT - 1))
        .map(constexpr)
        .collect();
    assert!(numbering.take(&all_but_one).is_some());
    let past = [constexpr(10_000), constexpr(10_001)];
    assert_eq!(numbering.take(&past), None);
    assert_eq!(
        numbering.take(&[constexpr(10_002)]),
        Some(vec![SAMPLER_ID_COUNT])
    );
}

fn device() -> Device {
    let (device, name) = test_device().unwrap();
    println!("rt's device: {name}");
    device
}

/// A compute state's descriptor of `name` in `library`.
fn kernel(library: &Library, name: &str) -> ComputePipelineDescriptor {
    let function = FunctionDescriptor::new(library.function(name).unwrap());
    ComputePipelineDescriptor::new(StageSource::Function(function))
}

/// rt makes a state of a kernel that samples through a constexpr sampler at the granted ID the
/// numbering gives it, the grant being the device's whole sampler range; a second state of an
/// equal sampler shares the ID, which goes back with the last of them.
#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "MoltenVK: rt's floor refuses it (no nullDescriptor), and these have not run on KosmicKrisp, which the M3 Max lacks; ledger: rt on the Mac"
)]
fn on_vulkan_a_constexpr_sampler_takes_a_granted_id_an_equal_one_shares() {
    let device = device();
    let ids = SamplerIds::new(&device).unwrap();
    let grant = IndexGrant {
        class: IndexClass::Sampler,
        first: FIRST_SAMPLER_ID,
        count: SAMPLER_ID_COUNT,
    };
    assert_eq!(ids.grant(), grant);
    let library = Library::from_metallib(NUMBERED).unwrap();
    let descriptor = kernel(&library, "numbered_sample");
    let first = ids.new_compute_pipeline_state(&descriptor).unwrap();
    assert!(
        matches!(first.samplers(), [NumberedSampler::Constexpr(_)]),
        "{:?}",
        first.samplers()
    );
    assert_eq!(first.sampler_ids().to_vec(), vec![1]);
    let second = ids.new_compute_pipeline_state(&descriptor).unwrap();
    assert_eq!(second.sampler_ids().to_vec(), vec![1]);
    drop(first);
    assert_eq!(ids.held(), 1);
    drop(second);
    assert_eq!(ids.held(), 0);
}

/// A read and a write take IDs as a constexpr sampler does, in the order the states are made.
#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "MoltenVK: rt's floor refuses it (no nullDescriptor), and these have not run on KosmicKrisp, which the M3 Max lacks; ledger: rt on the Mac"
)]
fn on_vulkan_reads_and_writes_take_ids_beside_constexpr_samplers() {
    let device = device();
    let ids = SamplerIds::new(&device).unwrap();
    let library = Library::from_metallib(NUMBERED).unwrap();
    let state = |name| {
        ids.new_compute_pipeline_state(&kernel(&library, name))
            .unwrap()
    };
    let read = state("numbered_read");
    let write = state("numbered_write");
    let sample = state("numbered_sample");
    assert_eq!(read.samplers().to_vec(), vec![NumberedSampler::Read]);
    assert_eq!(write.samplers().to_vec(), vec![NumberedSampler::Write]);
    let numbered = [&read, &write, &sample].map(|s| s.sampler_ids().to_vec());
    assert_eq!(numbered, [vec![1], vec![2], vec![3]]);
    assert_eq!(state("numbered_read").sampler_ids().to_vec(), vec![1]);
}

/// Each of five constexpr samplers of one kernel takes an ID of its own.
#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "MoltenVK: rt's floor refuses it (no nullDescriptor), and these have not run on KosmicKrisp, which the M3 Max lacks; ledger: rt on the Mac"
)]
fn on_vulkan_five_constexpr_samplers_take_five_ids() {
    let device = device();
    let ids = SamplerIds::new(&device).unwrap();
    let library = Library::from_metallib(FIVE).unwrap();
    let state = ids
        .new_compute_pipeline_state(&kernel(&library, "five_samplers"))
        .unwrap();
    assert_eq!(state.samplers().len(), 5, "{:?}", state.samplers());
    assert_eq!(state.sampler_ids().to_vec(), vec![1, 2, 3, 4, 5]);
}

/// rt's render path takes the same numbering: a fragment function's read takes an ID.
#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "MoltenVK: rt's floor refuses it (no nullDescriptor), and these have not run on KosmicKrisp, which the M3 Max lacks; ledger: rt on the Mac"
)]
fn on_vulkan_a_render_state_numbers_its_fragment_functions_read() {
    let device = device();
    let ids = SamplerIds::new(&device).unwrap();
    let library = Library::from_metallib(NUMBERED).unwrap();
    let function =
        |name| StageSource::Function(FunctionDescriptor::new(library.function(name).unwrap()));
    let mut descriptor = RenderPipelineDescriptor::new(function("numbered_vertex"));
    descriptor.fragment = Some(function("numbered_fragment_read"));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    let state = ids.new_render_pipeline_state(&descriptor).unwrap();
    assert_eq!(state.samplers().to_vec(), vec![NumberedSampler::Read]);
    assert_eq!(state.sampler_ids().to_vec(), vec![1]);
}

/// The grant is the device's once: a second numbering of it is rt's refusal.
#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "MoltenVK: rt's floor refuses it (no nullDescriptor), and these have not run on KosmicKrisp, which the M3 Max lacks; ledger: rt on the Mac"
)]
fn on_vulkan_a_second_numbering_of_a_device_is_refused() {
    let device = device();
    let _ids = SamplerIds::new(&device).unwrap();
    let again = SamplerIds::new(&device);
    assert!(
        matches!(
            again,
            Err(Error::Refused(Refusal::IndicesGranted {
                class: IndexClass::Sampler,
                ..
            }))
        ),
        "{:?}",
        again.err()
    );
}

/// A state whose sampler finds no free ID is refused holding none, and is made once one is free.
#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "MoltenVK: rt's floor refuses it (no nullDescriptor), and these have not run on KosmicKrisp, which the M3 Max lacks; ledger: rt on the Mac"
)]
fn on_vulkan_a_state_with_no_free_id_is_refused_holding_nothing() {
    let device = device();
    let ids = SamplerIds::new(&device).unwrap();
    let others: Vec<_> = (0..u64::from(SAMPLER_ID_COUNT)).map(constexpr).collect();
    assert!(ids.numbering.lock().take(&others).is_some());
    let library = Library::from_metallib(NUMBERED).unwrap();
    let descriptor = kernel(&library, "numbered_sample");
    let refused = ids.new_compute_pipeline_state(&descriptor);
    assert!(
        matches!(refused, Err(PipelineError::SamplerIds)),
        "{:?}",
        refused.err()
    );
    assert_eq!(ids.held(), SAMPLER_ID_COUNT as usize);
    ids.numbering.lock().give_back(&others[..1]);
    let state = ids.new_compute_pipeline_state(&descriptor).unwrap();
    assert_eq!(state.sampler_ids().to_vec(), vec![1]);
}

/// A state rt refuses before it numbers the samplers holds no ID.
#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "MoltenVK: rt's floor refuses it (no nullDescriptor), and these have not run on KosmicKrisp, which the M3 Max lacks; ledger: rt on the Mac"
)]
fn on_vulkan_a_state_rt_refuses_holds_no_id() {
    let device = device();
    let ids = SamplerIds::new(&device).unwrap();
    let library = Library::from_metallib(NUMBERED).unwrap();
    let mut descriptor = kernel(&library, "numbered_sample");
    descriptor.support_indirect_command_buffers = true;
    let refused = ids.new_compute_pipeline_state(&descriptor);
    assert!(
        matches!(
            refused,
            Err(PipelineError::Runtime(Error::Unsupported(
                Gap::IndirectCommandBuffers
            )))
        ),
        "{:?}",
        refused.err()
    );
    assert_eq!(ids.held(), 0);
}
