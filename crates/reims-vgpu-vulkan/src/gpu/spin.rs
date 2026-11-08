//! The tests' busy kernel (`spin.metal`), and how many of its rounds hold a device for a time.
//!
//! The kernel runs its first word's count of rounds of busy work on one thread, then writes its
//! result to its third word and `0x600d` to its second, so the CPU can tell whether it has ended.
//! A test that needs work in flight while it reads what a release left commits the kernel ahead
//! of that work, sized by [`rounds_to_hold`], and checks the kernel's end mark, or its command
//! buffer's status, once it has read.
//!
//! A round's time is the device's, and so is what a command buffer costs beside its rounds (its
//! submission, and the wait for its completion), which a run of few rounds is mostly made of
//! (measured, 2026-10-26, an RTX 5080 Laptop GPU on NVIDIA 615.71.09, metal2vulkan 6c9cad047a,
//! during the Vulkan crate's whole test run): runs of 2,000 and 4,000 rounds took 1.4 to 2.0 ms
//! from their commit to their completion, and runs of 256,000 took 70 to 76 ms, about 0.28 µs a
//! round. Kernels sized from 2,000 rounds alone to hold the GPU for half a second (447,227 and
//! 460,405 rounds) held it for 135 and 126 ms; sized from 256,000, for 425 to 505 ms. A first
//! run may pay for the device waking from idle: one of 2,000 rounds took 7.2 ms. On an Apple M3
//! Max a round takes about 60 µs. So [`rounds_to_hold`] times runs of doubling rounds, past the
//! first, until one takes [`TIMED`] or more, where the rest is a few percent.

use std::time::Duration;

use crate::core::platform::Platform;
use crate::core::platform::command::{CommandBuffer as _, CommandQueue as _};
use crate::core::platform::compute::{ComputeCmd, ComputeEncoder as _};
use crate::core::platform::device::GpuDevice as _;
use crate::core::platform::os::Clock as _;
use crate::core::platform::resource::{Buffer as _, Library as _};
use crate::core::platform::types::{CommandBufferStatus, MtlSize, NsRange};
use crate::core::raise::OnFault;

use crate::VulkanPlatform;
use crate::gpu::device::VulkanDevice;

type P = VulkanPlatform;

/// The busy kernel (`spin.metal`).
pub(crate) const SPIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/spin.metallib"));

/// The rounds of the first timed run, which doubles from there.
const FIRST_ROUNDS: u32 = 2_000;

/// How long the last timed run takes at least ([the module docs](self)).
const TIMED: Duration = Duration::from_millis(50);

/// The kernel's compute state on `d`.
pub(crate) fn state(d: &VulkanDevice) -> <P as Platform>::ComputePipeline {
    let library = d.new_library(SPIN).unwrap().unwrap().unwrap();
    let function = library.new_function(OnFault::Die, "spin").unwrap().unwrap();
    d.new_compute_pipeline_state_for_function(Some(&function))
        .unwrap()
        .unwrap()
}

/// Encodes the kernel into `cb`, an encoder of its own: one thread of `state`, over `words`,
/// whose first word names its rounds.
pub(crate) fn encode(
    cb: &<P as Platform>::CommandBuffer,
    state: &<P as Platform>::ComputePipeline,
    words: &<P as Platform>::Buffer,
) {
    let compute = cb.compute_encoder(OnFault::Die).unwrap().unwrap();
    for cmd in [
        ComputeCmd::SetComputePipelineState { state: Some(state) },
        ComputeCmd::SetBuffers {
            buffers: &[Some(words)],
            offsets: &[0],
            range: NsRange::new(0, 1),
        },
        ComputeCmd::DispatchThreads {
            threads_per_grid: MtlSize::new(1, 1, 1),
            threads_per_threadgroup: MtlSize::new(1, 1, 1),
        },
        ComputeCmd::EndEncoding,
    ] {
        compute.encode(OnFault::Die, cmd).unwrap();
    }
}

/// The kernel's rounds that hold `queue`'s device for `hold` or longer, and never fewer than
/// `fewest`: timed on the device in runs of the kernel alone on `queue`, each from its commit
/// to its completion, from [`FIRST_ROUNDS`] doubling until a run takes [`TIMED`] or more (the
/// module docs). Under other GPU work a run takes longer, and the rounds come out fewer.
pub(crate) fn rounds_to_hold(
    d: &VulkanDevice,
    queue: &<P as Platform>::CommandQueue,
    state: &<P as Platform>::ComputePipeline,
    hold: Duration,
    fewest: u32,
) -> u32 {
    let words = d
        .new_buffer_of_length(OnFault::Die, 16, 0)
        .unwrap()
        .unwrap();
    let clock = crate::os::OsClock::new();
    let mut rounds = FIRST_ROUNDS;
    let took = loop {
        // SAFETY: the kernel's rounds, in the first of the buffer's four words, which no command
        // buffer in flight uses: the last run completed.
        unsafe { words.contents().as_ptr::<u32>().write(rounds) };
        let run = queue.command_buffer(OnFault::Die).unwrap().unwrap();
        encode(&run, state, &words);
        let began = clock.now_mono();
        run.commit(OnFault::Die).unwrap();
        run.wait_until_completed(OnFault::Die).unwrap();
        let took = clock.now_mono().saturating_duration_since(began);
        assert_eq!(
            run.status(),
            CommandBufferStatus::COMPLETED,
            "the timed run of {rounds} rounds: {:?}",
            run.error()
        );
        // The first run never ends the series: it may pay for the device waking from idle.
        match rounds.checked_mul(2) {
            Some(more) if took < TIMED || rounds == FIRST_ROUNDS => rounds = more,
            _ => break took,
        }
    };
    let wanted = u128::from(rounds) * hold.as_nanos() / took.as_nanos().max(1);
    u32::try_from(wanted).unwrap_or(u32::MAX).max(fewest)
}
