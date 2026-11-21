//! A dispatch the platform refuses for a stage whose compile is not in its store
//! (`reims_vgpu_vulkan::core::platform::stage`): its walk waits for the stage's fill, and the
//! dispatch is made once more after the fill filled the store, or refused after any other end; a
//! make ahead's refusal starts the fill that the walk's dispatch then joins.

use reims_vgpu_vulkan::core::platform::stage::FillEnd;
use reims_vgpu_vulkan::core::testkit::Step;

use super::prepass::{begin, command_buffer_of, completed, end, set_state, threads, with_states};
use super::*;

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
        eventually(|| world.stages().started() == [7]),
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
    assert_eq!(world.stages().started(), [7]);
    assert!(world.stages().end(7, FillEnd::Filled));
    assert_eq!(completed(&platform, &mut ring, 100), RGFX_CB_OK);
    assert!(world.stages().refused().is_empty());
}
