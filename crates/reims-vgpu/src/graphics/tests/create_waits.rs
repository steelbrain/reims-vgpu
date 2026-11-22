//! A pipeline create's wait for its reply on the counts line (`create_waits`), on the platform's
//! virtual clock: a create whose make is held waits from its take to its reply, two held at once
//! wait in one span below their sum, and a create refused as it is prepared is counted at its
//! reply as every other.

use std::time::Duration;

use reims_vgpu_vulkan::core::testkit::{Step, World};

use super::prepass::{CONTEXT, with_states};
use super::*;
use crate::graphics::create_waits::{
    REPLIES_CODE, REPLIES_OVER_16MS_CODE, REPLY_MAX_NS_CODE, REPLY_NS_CODE, WAITING_NS_CODE,
};
use crate::platform::counted;

/// The fake's selector of a compute state's make.
const MAKE: &str = "newComputePipelineStateWithDescriptor:options:reflection:error:";

/// How long a test waits for a held make to be reached.
const REACHED: Duration = Duration::from_secs(60);

/// A millisecond of the clock, in nanoseconds.
const MS: u64 = 1_000_000;

/// The codes [`amounts`] reads, in its order.
const CODES: [&str; 5] = [
    REPLIES_CODE,
    REPLY_NS_CODE,
    REPLY_MAX_NS_CODE,
    REPLIES_OVER_16MS_CODE,
    WAITING_NS_CODE,
];

/// The process's counts of [`CODES`].
fn amounts() -> [u64; 5] {
    CODES.map(counted)
}

/// Holds the next compute state's make of `world` that no earlier hold takes, until the test lets
/// it go.
fn hold_make(world: &World) -> HeldCall {
    let held = HeldCall::new();
    world.script.push(MAKE, Step::Hold(held.clone()));
    held
}

/// Waits until the process has counted `n` replies more than `before` ([`amounts`]): a reply is
/// counted after its post, on the thread that posted it.
fn replies_counted(before: &[u64; 5], n: u64) -> bool {
    eventually(|| amounts()[0] == before[0] + n)
}

/// A create whose make is held waits from its take to its reply: one reply, whose wait is the
/// time the make was held, 20 ms, the longest so far where no longer one came, a frame or more,
/// and the span of the waits the same
#[test]
fn a_held_make_gives_one_reply_with_its_wait() {
    let (device, platform, mut guest, _ring) = with_states(&[]);
    let world = platform.world();
    let held = hold_make(world);
    let before = amounts();
    let id = obj(CONTEXT, 11);
    guest.send(&device, compute_pipeline_create(id, obj(CONTEXT, 10)));
    assert!(held.reached(REACHED), "the make was never reached");
    platform.virtual_clock().advance(Duration::from_millis(20));
    held.open();
    expect_reflection(&mut guest, id, RGFX_PIPELINE_OK);
    assert!(replies_counted(&before, 1), "the reply was not counted");
    assert_eq!(
        amounts(),
        [
            before[0] + 1,
            before[1] + 20 * MS,
            before[2].max(20 * MS),
            before[3] + 1,
            before[4] + 20 * MS,
        ]
    );
}

/// Two creates whose makes are held at once wait in one span: the first replied 20 ms after
/// both were taken and the second 40 ms after, so their waits sum to 60 ms, the longest is
/// 40 ms, and the span, their union, is 40 ms, below the sum
#[test]
fn two_creates_held_at_once_wait_in_a_span_below_their_sum() {
    let (device, platform, mut guest, _ring) = with_states(&[]);
    let world = platform.world();
    let (first, second) = (hold_make(world), hold_make(world));
    let before = amounts();
    for n in [11, 12] {
        guest.send(
            &device,
            compute_pipeline_create(obj(CONTEXT, n), obj(CONTEXT, 10)),
        );
    }
    assert!(
        first.reached(REACHED) && second.reached(REACHED),
        "the two makes were never held at once"
    );
    let clock = platform.virtual_clock();
    clock.advance(Duration::from_millis(20));
    first.open();
    next_event(&mut guest, RGFX_EVT_PIPELINE_REFLECTION);
    assert!(
        replies_counted(&before, 1),
        "the first reply was not counted"
    );
    clock.advance(Duration::from_millis(20));
    second.open();
    next_event(&mut guest, RGFX_EVT_PIPELINE_REFLECTION);
    assert!(
        replies_counted(&before, 2),
        "the second reply was not counted"
    );
    let after = amounts();
    assert_eq!(
        after,
        [
            before[0] + 2,
            before[1] + 60 * MS,
            before[2].max(40 * MS),
            before[3] + 2,
            before[4] + 40 * MS,
        ]
    );
    assert!(after[4] - before[4] < after[1] - before[1]);
}

/// A create refused as it is prepared, naming no library, is answered at once and counted at its
/// reply as every other, with the wait it had: none, on a clock that did not move
#[test]
fn a_create_refused_as_it_is_prepared_is_counted_at_its_reply() {
    let (device, _platform, mut guest, _ring) = with_states(&[]);
    let before = amounts();
    let id = obj(CONTEXT, 11);
    guest.send(&device, compute_pipeline_create(id, obj(CONTEXT, 99)));
    expect_reflection(&mut guest, id, RGFX_PIPELINE_REFUSED);
    assert!(replies_counted(&before, 1), "the refusal was not counted");
    assert_eq!(
        amounts(),
        [before[0] + 1, before[1], before[2], before[3], before[4]]
    );
}
