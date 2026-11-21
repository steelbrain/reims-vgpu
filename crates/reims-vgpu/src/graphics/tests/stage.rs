//! A dispatch the platform refuses for a stage whose compile is not in its store
//! (`reims_vgpu_vulkan::core::platform::stage`): its walk waits for the stage's fill, and the
//! dispatch is made once more after the fill filled the store, or refused after any other end; a
//! make ahead's refusal starts the fill that the walk's dispatch then joins. A buffer whose
//! dispatch may wait so is walked on its queue's lane, never on the queue's service thread. The
//! device's reset and the process's quiesce ahead of its exit wait for no fill: the fills a walk
//! waits for end cancelled, and its dispatch is refused.

use reims_vgpu_vulkan::core::platform::stage::{FillEnd, Site};
use reims_vgpu_vulkan::core::testkit::Step;

use super::prepass::{
    CONTEXT, begin, command_buffer_of, completed, end, set_state, threads, with_states,
    with_states_over,
};
use super::*;
use crate::platform::ApiPlatform;

/// The fake's selector of a [`threads`] dispatch.
const THREADS: &str = "dispatchThreads:threadsPerThreadgroup:";

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

/// A dispatch whose stage's fill ends breached is refused, encoding nothing, and its buffer
/// completes; a later dispatch of the stage is refused at once, and starts no fill
#[test]
fn a_dispatch_whose_stage_breached_is_refused_and_starts_no_other_fill() {
    let (device, platform, _guest, mut ring) = with_states(&[11]);
    let world = platform.world();
    world.script.push_n(THREADS, 2, Step::Stage(7));
    ring.send(&device, dispatching(100));
    assert!(
        eventually(|| world.stages().started() == [7]),
        "no fill started"
    );
    assert!(world.stages().end(7, FillEnd::Breached));
    assert_eq!(completed(&platform, &mut ring, 100), RGFX_CB_OK);
    ring.send(&device, dispatching(101));
    assert_eq!(completed(&platform, &mut ring, 101), RGFX_CB_OK);
    assert_eq!(world.stages().started(), [7]);
    assert_eq!(world.stages().refused(), [(THREADS, 7), (THREADS, 7)]);
}

/// A make ahead refused for a stage starts the stage's fill and waits for nothing; the walk's
/// dispatch of the stage joins that fill, one fill for both, and is made after it
#[test]
fn a_make_ahead_refused_for_its_stage_starts_the_fill_its_dispatch_joins() {
    let (device, platform, _guest, mut ring) = with_states(&[11]);
    let world = platform.world();
    world.stages().refuse_next_prepare(7);
    world.script.push(THREADS, Step::Stage(7));
    ring.send(&device, dispatching(100));
    assert!(
        eventually(|| world.prepared().len() == 1),
        "nothing was made ahead"
    );
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
    assert!(cancelled, "the reset waited for the fill a dispatch waits for");
    assert_eq!(world.stages().refused(), [(THREADS, 7)]);
    assert!(
        eventually(|| world.stages().fills().fill(7, Site::Dispatch).end().is_none()),
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
