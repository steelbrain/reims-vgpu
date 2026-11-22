//! A dispatch the platform refuses for a stage whose compile is not in its store
//! (`reims_vgpu_vulkan::core::platform::stage`): its walk waits for the stage's fill, and the
//! dispatch is made once more after the fill filled the store; after a fill that failed or
//! breached it is skipped, and its buffer fails with Metal's `Internal` (1); after a cancelled
//! fill it is refused, and its buffer completes. A make ahead's refusal starts the fill that the
//! walk's dispatch then joins. A buffer whose
//! dispatch may wait so is walked on its queue's lane, never on the queue's service thread. The
//! device's reset and the process's quiesce ahead of its exit wait for no fill: the fills a walk
//! waits for end cancelled, and its dispatch is refused.
//!
//! A compute state's make refused so holds its create pending, unanswered, until the fill ends:
//! the make is then made once more after a fill that filled the store, and its first refusal
//! stands after one that failed or breached, or after the create's destroy. A command buffer
//! that names the state meanwhile waits for it on its queue's lane. The reset and the quiesce let
//! go of a held create, unstored. Each held create the fills leave refused for good is counted,
//! once: after a fill that breached or failed, or a make refused again after one that filled.

use reims_vgpu_vulkan::core::platform::stage::{FillEnd, Site};
use reims_vgpu_vulkan::core::testkit::Step;

use super::prepass::{
    CONTEXT, begin, command_buffer_of, completed, end, set_state, threads, with_states,
    with_states_over,
};
use super::*;
use crate::graphics::compile::HELD_CREATE_REFUSED;
use crate::platform::{ApiPlatform, counted};

/// The fake's selector of a [`threads`] dispatch.
const THREADS: &str = "dispatchThreads:threadsPerThreadgroup:";

/// The fake's selector of a compute state's make.
const MAKE: &str = "newComputePipelineStateWithDescriptor:options:reflection:error:";

/// A command buffer `cb_obj_id` of one compute encoder that dispatches threads of state 11.
fn dispatching(cb_obj_id: u64) -> Vec<u8> {
    let ops = [
        begin(),
        set_state(11),
        threads([64, 1, 1], [32, 1, 1]),
        end(),
    ];
    command_buffer_of(cb_obj_id, &ops)
}

/// The status and Metal's error code that the completion of buffer `cb_obj_id` tells the guest.
fn completed_with_code(
    platform: &FakePlatform,
    ring: &mut QueueRing,
    cb_obj_id: u64,
) -> (u32, u32) {
    assert!(
        complete_until_event(platform, ring),
        "buffer {cb_obj_id} never completed"
    );
    let done = RgfxCbComplete::read_from_bytes(&ring.event().unwrap()).unwrap();
    assert_eq!(done.cb_obj_id, cb_obj_id);
    (done.status, done.error_code)
}

/// Whether the buffers of `ring` stay uncompleted for a while, the platform's committed buffers
/// completed meanwhile.
fn none_completes(platform: &FakePlatform, ring: &mut QueueRing) -> bool {
    for _ in 0..20 {
        platform.world().complete_all(CompleteOn::Caller);
        std::thread::sleep(Duration::from_millis(1));
    }
    !ring.has_event()
}

/// A dispatch refused for its stage holds its walk until the stage's fill ends, and its buffer
/// completes only then, its dispatch sent once and made after the fill filled the store
#[test]
fn a_dispatch_refused_for_its_stage_is_made_after_its_fill() {
    let (device, platform, _guest, mut ring) = with_states(&[11]);
    let world = platform.world();
    world.script.push(THREADS, Step::Stage(7));
    ring.send(&device, dispatching(100));
    assert!(
        eventually(|| world.stages().started_at() == [(7, Site::Dispatch)]),
        "no fill started"
    );
    assert!(
        none_completes(&platform, &mut ring),
        "the buffer completed while its dispatch waited for its stage's fill"
    );
    assert!(world.stages().end(7, FillEnd::Filled));
    assert_eq!(completed(&platform, &mut ring, 100), RGFX_CB_OK);
    assert_eq!(world.log.count(THREADS), 1);
    assert!(world.stages().refused().is_empty());
}

/// A dispatch whose stage's fill ends `end`, failed or breached, is skipped, encoding nothing, and
/// its buffer fails with Metal's `Internal` (1); a later dispatch of the stage is skipped at once,
/// failing its buffer so too, and starts no fill.
fn a_dispatch_after_its_fill_ended(end: FillEnd) {
    let (device, platform, _guest, mut ring) = with_states(&[11]);
    let world = platform.world();
    world.script.push_n(THREADS, 2, Step::Stage(7));
    ring.send(&device, dispatching(100));
    assert!(
        eventually(|| world.stages().started() == [7]),
        "no fill started"
    );
    assert!(world.stages().end(7, end));
    let internal = (RGFX_CB_INTERNAL, 1);
    assert_eq!(completed_with_code(&platform, &mut ring, 100), internal);
    ring.send(&device, dispatching(101));
    assert_eq!(completed_with_code(&platform, &mut ring, 101), internal);
    assert_eq!(world.stages().started(), [7]);
    assert_eq!(world.stages().refused(), [(THREADS, 7), (THREADS, 7)]);
}

/// A dispatch whose stage's fill ends breached is skipped, and its buffer fails with Metal's
/// `Internal` (1), as the next dispatch's of the stage does, which starts no fill
#[test]
fn a_dispatch_whose_stage_breached_fails_its_buffer_and_starts_no_other_fill() {
    a_dispatch_after_its_fill_ended(FillEnd::Breached);
}

/// A dispatch whose stage's fill ends failed is skipped, and its buffer fails with Metal's
/// `Internal` (1), as the next dispatch's of the stage does, which starts no fill
#[test]
fn a_dispatch_whose_stage_failed_fails_its_buffer_and_starts_no_other_fill() {
    a_dispatch_after_its_fill_ended(FillEnd::Failed);
}

/// A make ahead refused for a stage starts the stage's fill and waits for nothing; the walk's
/// dispatch of the stage joins that fill, one fill for both, and is made after it
#[test]
fn a_make_ahead_refused_for_its_stage_starts_the_fill_its_dispatch_joins() {
    let (device, platform, _guest, mut ring) = with_states(&[11]);
    let world = platform.world();
    world.stages().refuse_next_prepare(7);
    world.script.push(THREADS, Step::Stage(7));
    // The make ahead runs on a compile thread and the walk on its own, and the walk waits for
    // none of it: a walk that reached its dispatch first would start the fill itself. So the walk
    // is held after its encoder's begin, ahead of its dispatch, until the make ahead started it.
    let gate = WalkGate::in_place(&device);
    ring.send(&device, dispatching(100));
    assert!(gate.reached(), "the walk did not start");
    assert!(
        eventually(|| world.prepared().len() == 1),
        "nothing was made ahead"
    );
    assert!(
        eventually(|| world.stages().started_at() == [(7, Site::MakeAhead)]),
        "the make ahead started no fill"
    );
    gate.open();
    assert!(
        eventually(|| world.log.count(THREADS) == 1),
        "the walk never dispatched"
    );
    assert!(
        none_completes(&platform, &mut ring),
        "the buffer completed while its dispatch waited for its stage's fill"
    );
    assert_eq!(world.stages().started_at(), [(7, Site::MakeAhead)]);
    assert!(world.stages().end(7, FillEnd::Filled));
    assert_eq!(completed(&platform, &mut ring, 100), RGFX_CB_OK);
    assert!(world.stages().refused().is_empty());
}

/// A short command buffer whose dispatch runs a pipeline its pre-pass did not find made is walked
/// on its queue's lane, where its make or its stage's fill would hold no message of the queue's;
/// once the pipeline is made, the same buffer is walked on the queue's service thread
#[test]
fn a_short_buffer_whose_dispatch_is_not_made_is_walked_on_its_lane() {
    let (device, platform, _guest, mut ring) = with_states(&[11]);
    let world = platform.world();
    let threads = walk_threads(&device);
    ring.send(&device, dispatching(100));
    assert_eq!(completed(&platform, &mut ring, 100), RGFX_CB_OK);
    // Its job done, its turn passed: the lane's thread waits for a job.
    let wakes = || device.control.decoder().lane_wakes(obj(CONTEXT, 1));
    assert!(
        eventually(|| wakes().is_some_and(|w| w.2 == 1)),
        "the lane: {:?}",
        wakes()
    );
    // The fake numbers its keys as they are asked for: the dispatch's is the first.
    world.variants_made([0]);
    ring.send(&device, dispatching(101));
    assert_eq!(completed(&platform, &mut ring, 101), RGFX_CB_OK);
    let walked = threads.lock().clone();
    assert_eq!(walked.len(), 2, "walks seen: {walked:?}");
    assert!(
        walked[0].starts_with("reims-vgpu.graphics-lane-"),
        "the buffer whose dispatch was not made walked on {:?}",
        walked[0]
    );
    assert!(
        walked[1].starts_with("reims-vgpu.graphics-queue-"),
        "the buffer whose dispatch was made walked on {:?}",
        walked[1]
    );
}

/// A short command buffer whose dispatch waits for its stage's fill waits on its queue's lane: the
/// queue's next buffer is taken and walked meanwhile, and completes after it, in its turn
#[test]
fn a_dispatch_held_on_its_stage_s_fill_holds_no_message_of_its_queue() {
    let (device, platform, _guest, mut ring) = with_states(&[11]);
    let world = platform.world();
    let threads = walk_threads(&device);
    world.script.push(THREADS, Step::Stage(7));
    ring.send(&device, dispatching(100));
    assert!(
        eventually(|| world.stages().started() == [7]),
        "no fill started"
    );
    ring.send(&device, command_buffer_of(101, &[begin(), end()]));
    assert!(
        eventually(|| threads.lock().len() == 2),
        "the queue's next buffer waited behind the held walk: {:?}",
        threads.lock()
    );
    assert!(
        none_completes(&platform, &mut ring),
        "a buffer completed while the first waited for its stage's fill"
    );
    assert!(world.stages().end(7, FillEnd::Filled));
    assert_eq!(completed(&platform, &mut ring, 100), RGFX_CB_OK);
    assert_eq!(completed(&platform, &mut ring, 101), RGFX_CB_OK);
    let walked = threads.lock().clone();
    assert!(
        walked[0].starts_with("reims-vgpu.graphics-lane-"),
        "the held buffer walked on {:?}",
        walked[0]
    );
}

/// A device's reset while a dispatch waits for its stage's fill waits for no fill: the fill ends
/// cancelled and what runs it is cancelled, the held dispatch is refused and its walk ends, and
/// the reset goes on; once the reset is done, a refusal of the stage starts its fill again
#[test]
fn a_reset_ends_the_fill_a_dispatch_waits_for() {
    let (device, platform, _guest, mut ring) = with_states(&[11]);
    let world = platform.world();
    world.script.push(THREADS, Step::Stage(7));
    ring.send(&device, dispatching(100));
    assert!(
        eventually(|| world.stages().started() == [7]),
        "no fill started"
    );
    // The host hears of the reset on the device's control thread.
    device.reset();
    let cancelled = eventually(|| world.stages().cancels() == 1);
    // Where the reset waits for the fill, the fill's end lets it go on.
    let _ = world.stages().end(7, FillEnd::Filled);
    assert!(cancelled, "the reset waited for its dispatch's fill");
    // The cancel wakes the held dispatch, which is answered refused on its walk's thread.
    assert!(
        eventually(|| world.stages().refused() == [(THREADS, 7)]),
        "the held dispatch was not refused: {:?}",
        world.stages().refused()
    );
    let fills = world.stages().fills();
    assert!(
        eventually(|| fills.fill(7, Site::Dispatch).end().is_none()),
        "the fills stayed suspended past the reset"
    );
    assert_eq!(world.stages().started(), [7, 7]);
}

/// The process's quiesce ahead of its exit while a dispatch waits for its stage's fill ends the
/// fill: the fills close, so the held dispatch is refused and its buffer completes with no end
/// of the fill's, and no fill starts after
#[test]
fn a_quiesce_for_exit_ends_the_fill_a_dispatch_waits_for() {
    // The process's platform, which the quiesce asks: the test thread's.
    let (device, platform, _guest, mut ring) = with_states_over(FakePlatform::process(), &[11]);
    let world = platform.world();
    world.script.push(THREADS, Step::Stage(7));
    ring.send(&device, dispatching(100));
    assert!(
        eventually(|| world.stages().started() == [7]),
        "no fill started"
    );
    let may_exit = GraphicsDevice::<FakePlatform>::quiesce_for_exit(Duration::from_secs(1));
    assert!(may_exit);
    assert_eq!(completed(&platform, &mut ring, 100), RGFX_CB_OK);
    assert_eq!(world.stages().cancels(), 1);
    assert_eq!(world.stages().refused(), [(THREADS, 7)]);
    assert_eq!(
        world.stages().fills().fill(8, Site::Dispatch).end(),
        Some(FillEnd::Cancelled)
    );
    assert_eq!(world.stages().started(), [7]);
}

/// Whether the create of compute state 11 of [`CONTEXT`] is pending: its compile has not
/// settled, nor has a destroy, close or reset cancelled it.
fn state_11_pending(device: &GraphicsDevice<FakePlatform>) -> bool {
    !device.control.pending.named_by(&dispatching(0)).is_empty()
}

/// Sends the create of compute state 11 of [`CONTEXT`] over library 10, whose make `platform`
/// refuses for stage 7, and waits for the stage's fill to start.
fn create_refused_for_stage_7(
    device: &GraphicsDevice<FakePlatform>,
    platform: &FakePlatform,
    guest: &mut Guest,
) {
    let world = platform.world();
    world.script.push(MAKE, Step::Stage(7));
    guest.send(
        device,
        compute_pipeline_create(obj(CONTEXT, 11), obj(CONTEXT, 10)),
    );
    assert!(
        eventually(|| world.stages().started_at() == [(7, Site::Make)]),
        "no fill started"
    );
}

/// A compute state's make refused for its stage holds its create, pending and unanswered, until
/// the stage's fill ends; at a fill that filled the store the make is made again, once, and the
/// create is answered made. A command buffer that names the state meanwhile waits for it on its
/// queue's lane, the queue's next buffer taken and walked meanwhile, and runs with it, both
/// completing in their turns
#[test]
fn a_make_refused_for_its_stage_is_made_again_after_its_fill() {
    let (device, platform, mut guest, mut ring) = with_states(&[]);
    let world = platform.world();
    let threads = walk_threads(&device);
    create_refused_for_stage_7(&device, &platform, &mut guest);
    ring.send(&device, dispatching(100));
    ring.send(&device, command_buffer_of(101, &[begin(), end()]));
    assert!(
        eventually(|| threads.lock().len() == 1),
        "the queue's next buffer waited behind the one that waits for its pipeline: {:?}",
        threads.lock()
    );
    assert!(
        none_completes(&platform, &mut ring),
        "a buffer completed while the first waited for the pipeline it names"
    );
    assert!(state_11_pending(&device), "the held create was let go of");
    assert!(world.stages().end(7, FillEnd::Filled));
    expect_reflection(&mut guest, obj(CONTEXT, 11), RGFX_PIPELINE_OK);
    assert_eq!(completed(&platform, &mut ring, 100), RGFX_CB_OK);
    assert_eq!(completed(&platform, &mut ring, 101), RGFX_CB_OK);
    assert_eq!(world.log.count(MAKE), 2);
    assert_eq!(world.log.count(THREADS), 1);
    for name in threads.lock().iter() {
        assert!(
            name.starts_with("reims-vgpu.graphics-lane-"),
            "a buffer walked on {name:?}"
        );
    }
}

/// A compute state's make whose stage's fill ends `end`, failed or breached, is answered its
/// first refusal at the fill's end, and made no more; a command buffer that names the state
/// meanwhile then runs with it refused, its dispatch dropped, and completes.
fn a_make_after_its_fill_ended(end: FillEnd) {
    let (device, platform, mut guest, mut ring) = with_states(&[]);
    let world = platform.world();
    create_refused_for_stage_7(&device, &platform, &mut guest);
    ring.send(&device, dispatching(100));
    assert!(
        none_completes(&platform, &mut ring),
        "the buffer completed while the pipeline it names waited for its stage's fill"
    );
    assert!(world.stages().end(7, end));
    expect_reflection(&mut guest, obj(CONTEXT, 11), RGFX_PIPELINE_REFUSED);
    assert_eq!(completed(&platform, &mut ring, 100), RGFX_CB_OK);
    assert_eq!(world.log.count(MAKE), 1);
    assert_eq!(world.log.count(THREADS), 0);
    assert!(!state_11_pending(&device));
}

/// A compute state's make whose stage's fill ends breached is answered its first refusal, and
/// made no more; a command buffer that names it meanwhile completes
#[test]
fn a_make_whose_stage_breached_is_answered_its_refusal() {
    a_make_after_its_fill_ended(FillEnd::Breached);
}

/// A compute state's make whose stage's fill ends failed is answered its first refusal, and made
/// no more; a command buffer that names it meanwhile completes
#[test]
fn a_make_whose_stage_failed_is_answered_its_refusal() {
    a_make_after_its_fill_ended(FillEnd::Failed);
}

/// The destroy of a compute state whose make waits for its stage's fill keeps its place in ring
/// order: at the fill's end the make is not made again, nothing is stored, and the create is
/// answered its first refusal
#[test]
fn a_destroy_of_a_make_held_for_its_stage_is_not_made_again() {
    let (device, platform, mut guest, _ring) = with_states(&[]);
    let world = platform.world();
    let before = counted(HELD_CREATE_REFUSED);
    create_refused_for_stage_7(&device, &platform, &mut guest);
    guest.send(&device, object_destroy(obj(CONTEXT, 11)));
    // The destroy is carried out before a lookup of its context's after it is answered.
    carried_out_of(&device, &mut guest, CONTEXT);
    assert!(world.stages().end(7, FillEnd::Filled));
    let bytes = next_event(&mut guest, RGFX_EVT_PIPELINE_REFLECTION);
    let reflection = RgfxPipelineReflection::read_from_bytes(&bytes).unwrap();
    assert_eq!(
        (reflection.obj_id, reflection.status),
        (obj(CONTEXT, 11), RGFX_PIPELINE_REFUSED)
    );
    assert_eq!(world.log.count(MAKE), 1);
    // The queue and the library.
    assert_eq!(device.decoder_object_count(), 2);
    // The guest let go of it: its fill left nothing refused.
    assert_eq!(counted(HELD_CREATE_REFUSED) - before, 0);
}

/// How many held creates the fills left refused for good ([`HELD_CREATE_REFUSED`]) as the creates
/// of compute states 11 on, each refused at its make for a stage of its own, 7 on, were each held
/// for its stage's fill, which ended with its `ends`, in turn. Each create is answered made after
/// a fill that filled the store, and refused after one that breached or failed.
fn held_creates_refused(ends: &[FillEnd]) -> u64 {
    let (device, platform, mut guest, _ring) = with_states(&[]);
    let world = platform.world();
    let before = counted(HELD_CREATE_REFUSED);
    for (at, &end) in (0u64..).zip(ends) {
        let (state, stage) = (11 + at, 7 + at);
        world.script.push(MAKE, Step::Stage(stage));
        guest.send(
            &device,
            compute_pipeline_create(obj(CONTEXT, state), obj(CONTEXT, 10)),
        );
        assert!(
            eventually(|| world.stages().started().last() == Some(&stage)),
            "no fill started for state {state}"
        );
        assert!(world.stages().end(stage, end));
        let status = match end {
            FillEnd::Filled => RGFX_PIPELINE_OK,
            _ => RGFX_PIPELINE_REFUSED,
        };
        expect_reflection(&mut guest, obj(CONTEXT, state), status);
    }
    // Each create's make, and once more where its fill filled the store.
    let filled = ends.iter().filter(|&&end| end == FillEnd::Filled).count();
    assert_eq!(world.log.count(MAKE), ends.len() + filled);
    counted(HELD_CREATE_REFUSED) - before
}

/// Of 8 held creates, the 6 whose stages' fills breached or failed are counted refused for good,
/// one each, and the 2 made after fills that filled the store are not
#[test]
fn the_held_creates_left_refused_are_counted_one_each() {
    use FillEnd::{Breached, Failed, Filled};
    let ends = [
        Breached, Filled, Failed, Breached, Filled, Breached, Failed, Breached,
    ];
    assert_eq!(held_creates_refused(&ends), 6);
}

/// 8 held creates all made after fills that filled the store count none refused
#[test]
fn held_creates_all_made_count_none_refused() {
    assert_eq!(held_creates_refused(&[FillEnd::Filled; 8]), 0);
}

/// A held create whose make refuses again after a fill that filled the store is answered refused,
/// and counted refused for good, once
#[test]
fn a_held_create_refused_again_after_its_fill_is_counted() {
    let (device, platform, mut guest, _ring) = with_states(&[]);
    let world = platform.world();
    let before = counted(HELD_CREATE_REFUSED);
    create_refused_for_stage_7(&device, &platform, &mut guest);
    world.script.push(MAKE, Step::Nil);
    assert!(world.stages().end(7, FillEnd::Filled));
    expect_reflection(&mut guest, obj(CONTEXT, 11), RGFX_PIPELINE_REFUSED);
    assert_eq!(world.log.count(MAKE), 2);
    assert!(!state_11_pending(&device));
    assert_eq!(counted(HELD_CREATE_REFUSED) - before, 1);
}

/// A device's reset while a compute state's make waits for its stage's fill, and a command buffer
/// waits for the state on its lane, waits for neither: the fill ends cancelled, the create is
/// let go of unstored and made no more, the buffer is not carried out, and the reset goes on
#[test]
fn a_reset_lets_go_of_a_make_held_for_its_stage() {
    let (device, platform, mut guest, mut ring) = with_states(&[]);
    let world = platform.world();
    create_refused_for_stage_7(&device, &platform, &mut guest);
    ring.send(&device, dispatching(100));
    assert!(
        none_completes(&platform, &mut ring),
        "the buffer completed while the pipeline it names waited for its stage's fill"
    );
    // The host hears of the reset on the device's control thread.
    device.reset();
    let cancelled = eventually(|| world.stages().cancels() == 1);
    // Where the reset waits for the fill, the fill's end lets it go on.
    let _ = world.stages().end(7, FillEnd::Filled);
    assert!(cancelled, "the reset waited for the held make's fill");
    let fills = world.stages().fills();
    assert!(
        eventually(|| fills.fill(8, Site::Make).end().is_none()),
        "the reset never ended"
    );
    assert!(!state_11_pending(&device));
    assert!(
        none_completes(&platform, &mut ring),
        "a buffer of a queue the reset ended completed"
    );
    assert_eq!(world.log.count(MAKE), 1);
    assert_eq!(world.log.count(THREADS), 0);
}

/// The process's quiesce ahead of its exit while a compute state's make waits for its stage's
/// fill waits for none: the fills close, and the create is let go of unstored, made no more, and
/// answered its first refusal where the ring has room
#[test]
fn a_quiesce_for_exit_lets_go_of_a_make_held_for_its_stage() {
    // The process's platform, which the quiesce asks: the test thread's.
    let (device, platform, mut guest, _ring) = with_states_over(FakePlatform::process(), &[]);
    let world = platform.world();
    create_refused_for_stage_7(&device, &platform, &mut guest);
    let may_exit = GraphicsDevice::<FakePlatform>::quiesce_for_exit(Duration::from_secs(1));
    assert!(may_exit);
    expect_reflection(&mut guest, obj(CONTEXT, 11), RGFX_PIPELINE_REFUSED);
    assert_eq!(world.stages().cancels(), 1);
    assert_eq!(world.log.count(MAKE), 1);
    assert!(!state_11_pending(&device));
    // The queue and the library.
    assert_eq!(device.decoder_object_count(), 2);
}
