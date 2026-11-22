//! A command buffer's dispatch pre-pass (`decode::prepass`): each pipeline its dispatches run,
//! by the platform's key of it, made ahead once for the device, on the compile threads, with the
//! arguments of the first dispatch of it; a dispatch with no key, or of a key made, made ahead
//! never; the walk waiting for none of it, and making at its first dispatch those no thread took;
//! and nothing of it outliving the buffer's views of the table, nor its device.

use reims_vgpu_graphics::ops::compute::{
    RGFX_OP_COMPUTE_DISPATCH_THREADGROUPS, RGFX_OP_COMPUTE_DISPATCH_THREADGROUPS_INDIRECT,
    RGFX_OP_COMPUTE_DISPATCH_THREADS, RGFX_OP_COMPUTE_SET_COMPUTE_PIPELINE_STATE,
    RGFX_OP_COMPUTE_SET_THREADGROUP_MEMORY_LENGTH, RgfxComputeOpDispatchThreadgroups,
    RgfxComputeOpDispatchThreadgroupsIndirect, RgfxComputeOpDispatchThreads,
    RgfxComputeOpSetComputePipelineState, RgfxComputeOpSetThreadgroupMemoryLength,
};
use reims_vgpu_graphics::ops::{
    RGFX_OP_ENCODER_BEGIN_COMPUTE, RGFX_OP_ENCODER_END, RgfxComputePassDesc, RgfxOpHeader,
};
use reims_vgpu_vulkan::core::testkit::{Prepared, PreparedGrid};

use super::*;
use crate::graphics::decode::prepass::MOST;

/// The queue's rings, as [`with_states`] makes them.
const SUBMIT_OFF: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
const COMPLETE_OFF: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x2_0000;
const RING_DATA: u32 = 0x1_0000;

/// The context the guest's objects are of.
pub(super) const CONTEXT: u32 = 3;

/// The fake's class of a compute state.
const CLASS: &str = "MTLComputePipelineState";

/// A device whose context [`CONTEXT`] has queue 1, library 10 and the compute states `states`
/// made: the device, its platform, the guest and the queue's rings.
pub(super) fn with_states(
    states: &[u64],
) -> (
    GraphicsDevice<FakePlatform>,
    Arc<FakePlatform>,
    Guest,
    QueueRing,
) {
    with_states_over(Arc::new(FakePlatform::new()), states)
}

/// [`with_states`] on `platform`.
pub(super) fn with_states_over(
    platform: Arc<FakePlatform>,
    states: &[u64],
) -> (
    GraphicsDevice<FakePlatform>,
    Arc<FakePlatform>,
    Guest,
    QueueRing,
) {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform) = headless_over(&raised, platform);
    device.mmio_write(RGFX_REG_STATUS.into(), RGFX_STATUS_DRIVER_READY);
    let mut guest = Guest::new(&device);
    let rings = (SUBMIT_OFF, COMPLETE_OFF);
    assert_eq!(
        create_queue(&device, &mut guest, CONTEXT, 1, rings, RING_DATA, 0),
        RGFX_STATUS_OK
    );
    make_library(&device, &mut guest, obj(CONTEXT, 10));
    for &n in states {
        guest.send(
            &device,
            compute_pipeline_create(obj(CONTEXT, n), obj(CONTEXT, 10)),
        );
        expect_reflection(&mut guest, obj(CONTEXT, n), RGFX_PIPELINE_OK);
    }
    let ring = QueueRing::new(&device, SUBMIT_OFF, COMPLETE_OFF, u64::from(RING_DATA));
    (device, platform, guest, ring)
}

/// An op of tag `tag` and body `body`.
fn op(tag: u16, body: &[u8]) -> Vec<u8> {
    let mut bytes = RgfxOpHeader {
        tag,
        flags: 0,
        len: (8 + body.len()) as u32,
    }
    .as_bytes()
    .to_vec();
    bytes.extend_from_slice(body);
    bytes
}

/// A `COMMAND_BUFFER` `cb_obj_id` on queue 1 of [`CONTEXT`] of the ops `ops`.
pub(super) fn command_buffer_of(cb_obj_id: u64, ops: &[Vec<u8>]) -> Vec<u8> {
    let ops = ops.concat();
    let mut bytes = RgfxCommandBuffer {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_COMMAND_BUFFER,
            size: 0,
            seqno: 0,
        },
        queue_obj_id: obj(CONTEXT, 1),
        cb_obj_id,
        op_bytes: ops.len() as u64,
    }
    .as_bytes()
    .to_vec();
    bytes.extend(ops);
    bytes
}

/// A compute encoder's begin.
pub(super) fn begin() -> Vec<u8> {
    op(
        RGFX_OP_ENCODER_BEGIN_COMPUTE,
        RgfxComputePassDesc::default().as_bytes(),
    )
}

/// An encoder's end.
pub(super) fn end() -> Vec<u8> {
    op(RGFX_OP_ENCODER_END, &[])
}

/// `setComputePipelineState:` of [`CONTEXT`]'s object `n`.
pub(super) fn set_state(n: u64) -> Vec<u8> {
    let body = RgfxComputeOpSetComputePipelineState {
        pipeline_obj_id: obj(CONTEXT, n),
    };
    op(RGFX_OP_COMPUTE_SET_COMPUTE_PIPELINE_STATE, body.as_bytes())
}

/// `setThreadgroupMemoryLength:atIndex:`.
fn length(length: u64, index: u64) -> Vec<u8> {
    let body = RgfxComputeOpSetThreadgroupMemoryLength { length, index };
    op(
        RGFX_OP_COMPUTE_SET_THREADGROUP_MEMORY_LENGTH,
        body.as_bytes(),
    )
}

/// `dispatchThreads:threadsPerThreadgroup:`.
pub(super) fn threads([w, h, d]: [u64; 3], [tw, th, td]: [u64; 3]) -> Vec<u8> {
    let body = RgfxComputeOpDispatchThreads {
        grid_width: w,
        grid_height: h,
        grid_depth: d,
        threads_width: tw,
        threads_height: th,
        threads_depth: td,
    };
    op(RGFX_OP_COMPUTE_DISPATCH_THREADS, body.as_bytes())
}

/// `dispatchThreadgroups:threadsPerThreadgroup:`.
fn groups([w, h, d]: [u64; 3], [tw, th, td]: [u64; 3]) -> Vec<u8> {
    let body = RgfxComputeOpDispatchThreadgroups {
        threadgroups_width: w,
        threadgroups_height: h,
        threadgroups_depth: d,
        threads_width: tw,
        threads_height: th,
        threads_depth: td,
    };
    op(RGFX_OP_COMPUTE_DISPATCH_THREADGROUPS, body.as_bytes())
}

/// `dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:` of
/// [`CONTEXT`]'s buffer `n`.
fn indirect(n: u64, [tw, th, td]: [u64; 3]) -> Vec<u8> {
    let body = RgfxComputeOpDispatchThreadgroupsIndirect {
        indirect_buffer_obj_id: obj(CONTEXT, n),
        indirect_buffer_offset: 0,
        threads_width: tw,
        threads_height: th,
        threads_depth: td,
    };
    op(
        RGFX_OP_COMPUTE_DISPATCH_THREADGROUPS_INDIRECT,
        body.as_bytes(),
    )
}

/// The next event of `ring`, completing the platform's committed buffers meanwhile, which must be
/// `cb_obj_id`'s `CB_COMPLETE`: its status.
pub(super) fn completed(platform: &FakePlatform, ring: &mut QueueRing, cb_obj_id: u64) -> u32 {
    assert!(
        complete_until_event(platform, ring),
        "buffer {cb_obj_id} never completed"
    );
    let done = RgfxCbComplete::read_from_bytes(&ring.event().unwrap()).unwrap();
    assert_eq!(done.cb_obj_id, cb_obj_id);
    done.status
}

/// The makes ahead so far, sorted: the compile threads make them in no fixed order.
fn made(platform: &FakePlatform) -> Vec<Prepared> {
    let mut made = platform.world().prepared();
    made.sort();
    made
}

/// How many of code `code` this process counted.
fn counted(code: &str) -> u64 {
    (reims_vgpu_vulkan::answer::counted_kinds().into_iter())
        .find(|&(c, _)| c == code)
        .map_or(0, |(_, n)| n)
}

/// With the fake's keys, one of each compute state and shape: each shape of a command buffer's
/// dispatches is made ahead once while its walk waits at its
/// first op, with the dispatch's own grid and threads a threadgroup and the threadgroup memory
/// lengths its encoder set before it (each index once, in the order first set, at its last
/// length; a new encoder sets none); both buffers complete as without the pre-pass, and a later
/// buffer of the same shapes makes none again
#[test]
fn each_dispatch_shape_is_made_ahead_once_with_its_own_arguments() {
    let (device, platform, mut guest, mut ring) = with_states(&[11, 12]);
    guest.send(&device, buffer_create(obj(CONTEXT, 20)));
    carried_out_of(&device, &mut guest, CONTEXT);
    let states = platform.world().live(CLASS);
    assert_eq!(states.len(), 2);
    let ops = [
        begin(),
        set_state(11),
        length(256, 0),
        threads([64, 1, 1], [32, 1, 1]),
        threads([64, 1, 1], [32, 1, 1]),
        length(512, 1),
        length(128, 0),
        groups([4, 2, 1], [8, 8, 1]),
        set_state(12),
        indirect(20, [16, 1, 1]),
        end(),
        begin(),
        set_state(11),
        threads([64, 1, 1], [32, 1, 1]),
        end(),
    ];
    let gate = WalkGate::on(&device);
    ring.send(&device, command_buffer_of(100, &ops));
    assert!(gate.reached(), "the walk did not start");
    let mut expected = vec![
        Prepared {
            state: states[0],
            dispatch: PreparedGrid::Threads(64, 1, 1),
            threads: (32, 1, 1),
            lengths: vec![(0, 256)],
        },
        Prepared {
            state: states[0],
            dispatch: PreparedGrid::Threadgroups(4, 2, 1),
            threads: (8, 8, 1),
            lengths: vec![(0, 128), (1, 512)],
        },
        Prepared {
            state: states[1],
            dispatch: PreparedGrid::ThreadgroupsIndirect,
            threads: (16, 1, 1),
            lengths: vec![(0, 128), (1, 512)],
        },
        Prepared {
            state: states[0],
            dispatch: PreparedGrid::Threads(64, 1, 1),
            threads: (32, 1, 1),
            lengths: vec![],
        },
    ];
    expected.sort();
    assert!(
        eventually(|| made(&platform).len() >= expected.len()),
        "made ahead: {:?}",
        made(&platform)
    );
    std::thread::sleep(Duration::from_millis(20));
    assert_eq!(made(&platform), expected);
    gate.open();
    assert_eq!(completed(&platform, &mut ring, 100), RGFX_CB_OK);
    ring.send(&device, command_buffer_of(101, &ops));
    assert_eq!(completed(&platform, &mut ring, 101), RGFX_CB_OK);
    std::thread::sleep(Duration::from_millis(20));
    assert_eq!(made(&platform), expected, "a shape was made ahead again");
}

/// Dispatches of one key, of two shapes of a compute state or of two states, are made ahead
/// once, with the arguments of the first of them in the buffer, while a dispatch of another key
/// is made ahead too; the buffer completes as without the pre-pass
#[test]
fn dispatches_of_one_key_are_made_ahead_once() {
    let (device, platform, _guest, mut ring) = with_states(&[11, 12]);
    let world = platform.world();
    let states = world.live(CLASS);
    assert_eq!(states.len(), 2);
    let other = states[1];
    world.key_dispatches(move |p| match p.dispatch {
        PreparedGrid::Threads(..) => Some(1),
        PreparedGrid::Threadgroups(..) if p.state == other => Some(1),
        _ => Some(2),
    });
    let ops = [
        begin(),
        set_state(11),
        threads([64, 1, 1], [32, 1, 1]),
        threads([128, 1, 1], [32, 1, 1]),
        groups([4, 2, 1], [8, 8, 1]),
        set_state(12),
        threads([64, 1, 1], [32, 1, 1]),
        groups([2, 1, 1], [32, 1, 1]),
        end(),
    ];
    let gate = WalkGate::on(&device);
    ring.send(&device, command_buffer_of(100, &ops));
    assert!(gate.reached(), "the walk did not start");
    let mut expected = vec![
        Prepared {
            state: states[0],
            dispatch: PreparedGrid::Threads(64, 1, 1),
            threads: (32, 1, 1),
            lengths: vec![],
        },
        Prepared {
            state: states[0],
            dispatch: PreparedGrid::Threadgroups(4, 2, 1),
            threads: (8, 8, 1),
            lengths: vec![],
        },
    ];
    expected.sort();
    assert!(
        eventually(|| made(&platform).len() >= expected.len()),
        "made ahead: {:?}",
        made(&platform)
    );
    std::thread::sleep(Duration::from_millis(20));
    assert_eq!(made(&platform), expected, "a key was made ahead twice");
    gate.open();
    assert_eq!(completed(&platform, &mut ring, 100), RGFX_CB_OK);
}

/// A dispatch the platform answers no key of, which it would make nothing for, is made ahead
/// never, and counted; the buffer's other dispatches are made ahead, and it completes as without
/// the pre-pass
#[test]
fn a_dispatch_of_no_key_is_made_ahead_never_and_counted() {
    let (device, platform, _guest, mut ring) = with_states(&[11]);
    let world = platform.world();
    let state = world.live(CLASS);
    assert_eq!(state.len(), 1);
    world.key_dispatches(|p| match p.dispatch {
        PreparedGrid::Threads(..) => None,
        _ => Some(1),
    });
    let before = counted("dispatch-prepass-no-variant");
    let ops = [
        begin(),
        set_state(11),
        threads([64, 1, 1], [32, 1, 1]),
        groups([4, 2, 1], [8, 8, 1]),
        end(),
    ];
    let gate = WalkGate::on(&device);
    ring.send(&device, command_buffer_of(100, &ops));
    assert!(gate.reached(), "the walk did not start");
    let expected = vec![Prepared {
        state: state[0],
        dispatch: PreparedGrid::Threadgroups(4, 2, 1),
        threads: (8, 8, 1),
        lengths: vec![],
    }];
    assert!(
        eventually(|| !made(&platform).is_empty()),
        "nothing was made ahead"
    );
    std::thread::sleep(Duration::from_millis(20));
    assert_eq!(made(&platform), expected);
    assert_eq!(counted("dispatch-prepass-no-variant"), before + 1);
    gate.open();
    assert_eq!(completed(&platform, &mut ring, 100), RGFX_CB_OK);
}

/// A dispatch of a key a command buffer handed before, of another shape or another compute
/// state, is made ahead never by a later buffer, which still makes ahead a dispatch of a new key
#[test]
fn a_key_handed_before_is_made_ahead_never_again() {
    let (device, platform, _guest, mut ring) = with_states(&[11, 12]);
    let world = platform.world();
    let states = world.live(CLASS);
    assert_eq!(states.len(), 2);
    world.key_dispatches(|p| match p.dispatch {
        PreparedGrid::Threadgroups(4, 2, 1) => Some(8),
        _ => Some(7),
    });
    let first = Prepared {
        state: states[0],
        dispatch: PreparedGrid::Threads(64, 1, 1),
        threads: (32, 1, 1),
        lengths: vec![],
    };
    let ops = [
        begin(),
        set_state(11),
        threads([64, 1, 1], [32, 1, 1]),
        end(),
    ];
    let gate = WalkGate::on(&device);
    ring.send(&device, command_buffer_of(100, &ops));
    assert!(gate.reached(), "the walk did not start");
    assert!(
        eventually(|| !made(&platform).is_empty()),
        "nothing was made ahead"
    );
    gate.open();
    assert_eq!(completed(&platform, &mut ring, 100), RGFX_CB_OK);
    assert_eq!(made(&platform), vec![first.clone()]);

    let ops = [
        begin(),
        set_state(11),
        threads([256, 1, 1], [32, 1, 1]),
        set_state(12),
        groups([2, 1, 1], [32, 1, 1]),
        set_state(11),
        groups([4, 2, 1], [8, 8, 1]),
        end(),
    ];
    let gate = WalkGate::on(&device);
    ring.send(&device, command_buffer_of(101, &ops));
    assert!(gate.reached(), "the second walk did not start");
    let mut expected = vec![
        first,
        Prepared {
            state: states[0],
            dispatch: PreparedGrid::Threadgroups(4, 2, 1),
            threads: (8, 8, 1),
            lengths: vec![],
        },
    ];
    expected.sort();
    assert!(
        eventually(|| made(&platform).len() >= expected.len()),
        "the new key was not made ahead: {:?}",
        made(&platform)
    );
    std::thread::sleep(Duration::from_millis(20));
    assert_eq!(
        made(&platform),
        expected,
        "a key handed before was made again"
    );
    gate.open();
    assert_eq!(completed(&platform, &mut ring, 101), RGFX_CB_OK);
}

/// A command buffer hands its first [`MOST`] new shapes at the most; a later buffer hands those
/// past them
#[test]
fn a_command_buffer_hands_its_first_shapes_up_to_the_most() {
    let (device, platform, _guest, mut ring) = with_states(&[11]);
    let n = MOST as u64 + 8;
    let mut ops = vec![begin(), set_state(11)];
    ops.extend((1..=n).map(|w| threads([w, 1, 1], [1, 1, 1])));
    ops.push(end());
    let widths = |platform: &FakePlatform| -> Vec<u64> {
        let mut widths: Vec<u64> = (made(platform).into_iter())
            .map(|p| match p.dispatch {
                PreparedGrid::Threads(w, _, _) => w,
                grid => panic!("made ahead: {grid:?}"),
            })
            .collect();
        widths.sort_unstable();
        widths
    };
    let gate = WalkGate::on(&device);
    ring.send(&device, command_buffer_of(100, &ops));
    assert!(gate.reached(), "the walk did not start");
    assert!(eventually(|| made(&platform).len() >= MOST));
    std::thread::sleep(Duration::from_millis(20));
    assert_eq!(widths(&platform), (1..=MOST as u64).collect::<Vec<_>>());
    gate.open();
    assert_eq!(completed(&platform, &mut ring, 100), RGFX_CB_OK);

    let gate = WalkGate::on(&device);
    ring.send(&device, command_buffer_of(101, &ops));
    assert!(gate.reached(), "the second walk did not start");
    assert!(eventually(|| made(&platform).len() >= n as usize));
    std::thread::sleep(Duration::from_millis(20));
    assert_eq!(widths(&platform), (1..=n).collect::<Vec<_>>());
    gate.open();
    assert_eq!(completed(&platform, &mut ring, 101), RGFX_CB_OK);
}

/// A dispatch whose key the platform answers made hands nothing: it takes no place of the
/// buffer's [`MOST`] and gives the pool no compile, so the buffer's first [`MOST`] keys not made
/// are each made ahead and none is counted past the most; a buffer of made keys alone gives the
/// pool nothing; both complete as without the pre-pass
#[test]
fn a_made_key_takes_no_place_and_no_compile() {
    let (device, platform, _guest, mut ring) = with_states(&[11]);
    let world = platform.world();
    // Each width its own key, the first four and four past the first buffer's widths made.
    let made_keys = 4;
    let later = 1 << 20;
    world.key_dispatches(|p| match p.dispatch {
        PreparedGrid::Threads(w, _, _) => Some(w),
        _ => None,
    });
    world.variants_made((1..=made_keys).chain(later..later + 4));
    let widths = |platform: &FakePlatform| -> Vec<u64> {
        let mut widths: Vec<u64> = (made(platform).into_iter())
            .map(|p| match p.dispatch {
                PreparedGrid::Threads(w, _, _) => w,
                grid => panic!("made ahead: {grid:?}"),
            })
            .collect();
        widths.sort_unstable();
        widths
    };
    let (handed, past_most) = (
        counted("dispatch-prepass-handed"),
        counted("dispatch-prepass-past-most"),
    );
    let n = MOST as u64 + made_keys;
    let mut ops = vec![begin(), set_state(11)];
    ops.extend((1..=n).map(|w| threads([w, 1, 1], [1, 1, 1])));
    ops.push(end());
    let gate = WalkGate::on(&device);
    ring.send(&device, command_buffer_of(100, &ops));
    assert!(gate.reached(), "the walk did not start");
    assert!(eventually(|| made(&platform).len() >= MOST));
    std::thread::sleep(Duration::from_millis(20));
    assert_eq!(widths(&platform), (made_keys + 1..=n).collect::<Vec<_>>());
    assert_eq!(counted("dispatch-prepass-handed") - handed, MOST as u64);
    assert_eq!(counted("dispatch-prepass-past-most"), past_most);
    gate.open();
    assert_eq!(completed(&platform, &mut ring, 100), RGFX_CB_OK);

    let mut ops = vec![begin(), set_state(11)];
    ops.extend((later..later + 4).map(|w| threads([w, 1, 1], [1, 1, 1])));
    ops.push(end());
    ring.send(&device, command_buffer_of(101, &ops));
    assert_eq!(completed(&platform, &mut ring, 101), RGFX_CB_OK);
    std::thread::sleep(Duration::from_millis(20));
    assert_eq!(counted("dispatch-prepass-handed") - handed, MOST as u64);
    assert_eq!(widths(&platform), (made_keys + 1..=n).collect::<Vec<_>>());
}

/// A command buffer whose makes ahead wait for a compile thread is walked and completes, its walk
/// making them on its own thread at its first dispatch, and not before it: while the walk waits at
/// its first op none is made, then both are, each once, counted; once the threads go on, they
/// make none again
#[test]
fn the_walk_makes_the_shapes_no_thread_took_at_its_first_dispatch() {
    let (device, platform, _guest, mut ring) = with_states(&[11]);
    let state = platform.world().live(CLASS);
    assert_eq!(state.len(), 1);
    device.control.pool.hold();
    let on_walk = counted("dispatch-prepass-made-on-walk");
    let ops = [
        begin(),
        set_state(11),
        threads([64, 1, 1], [32, 1, 1]),
        groups([4, 2, 1], [8, 8, 1]),
        end(),
    ];
    let gate = WalkGate::on(&device);
    ring.send(&device, command_buffer_of(100, &ops));
    assert!(gate.reached(), "the walk did not start");
    std::thread::sleep(Duration::from_millis(20));
    assert_eq!(
        made(&platform),
        vec![],
        "made before the walk's first dispatch"
    );
    gate.open();
    assert_eq!(completed(&platform, &mut ring, 100), RGFX_CB_OK);
    let mut expected = vec![
        Prepared {
            state: state[0],
            dispatch: PreparedGrid::Threads(64, 1, 1),
            threads: (32, 1, 1),
            lengths: vec![],
        },
        Prepared {
            state: state[0],
            dispatch: PreparedGrid::Threadgroups(4, 2, 1),
            threads: (8, 8, 1),
            lengths: vec![],
        },
    ];
    expected.sort();
    assert_eq!(made(&platform), expected);
    assert_eq!(counted("dispatch-prepass-made-on-walk") - on_walk, 2);
    device.control.pool.release();
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(made(&platform), expected, "a make ahead ran after its walk");
}

/// A command buffer whose makes ahead wait for a compile thread and whose walk ends before its
/// first dispatch (it faults at an op before it) makes none of them: its end drops them, counted,
/// and once the threads go on, they make none
#[test]
fn a_walk_ended_before_its_first_dispatch_drops_the_shapes_no_thread_took() {
    let (device, platform, _guest, mut ring) = with_states(&[11]);
    device.control.pool.hold();
    let (dropped, on_walk) = (
        counted("dispatch-prepass-dropped"),
        counted("dispatch-prepass-made-on-walk"),
    );
    // An op no decoder knows: the walk faults the buffer there, before its dispatch.
    let unknown = op(0x20ff, &[]);
    let ops = [
        begin(),
        unknown,
        set_state(11),
        threads([64, 1, 1], [32, 1, 1]),
        end(),
    ];
    ring.send(&device, command_buffer_of(100, &ops));
    assert_ne!(completed(&platform, &mut ring, 100), RGFX_CB_OK);
    assert_eq!(counted("dispatch-prepass-dropped") - dropped, 1);
    assert_eq!(counted("dispatch-prepass-made-on-walk"), on_walk);
    device.control.pool.release();
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(made(&platform), vec![], "a make ahead ran after its walk");
}

/// A device released while its command buffer's makes ahead still wait for a compile thread ends
/// clean: its walk made the one at its first dispatch, no compute state outlives the device, and
/// none is made ahead again once the threads go on
#[test]
fn a_device_released_with_makes_ahead_queued_ends_clean() {
    let (device, platform, _guest, mut ring) = with_states(&[11]);
    device.control.pool.hold();
    let open = device.control.pool.opener();
    let ops = [
        begin(),
        set_state(11),
        threads([64, 1, 1], [32, 1, 1]),
        end(),
    ];
    ring.send(&device, command_buffer_of(100, &ops));
    assert_eq!(completed(&platform, &mut ring, 100), RGFX_CB_OK);
    let walked = made(&platform);
    assert_eq!(walked.len(), 1, "its walk made: {walked:?}");
    drop(device);
    let world = platform.world();
    assert!(
        eventually(|| world.live(CLASS).is_empty()),
        "compute states outlived their device: {:?}",
        world.live(CLASS)
    );
    open();
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(made(&platform), walked, "a make ahead ran after its device");
}

/// A make ahead of a dispatch the walk never reaches (the walk faults at an op before it) that
/// is still being made at the walk's end is waited for there, before the buffer's failure is
/// stood in for and before its views of the table close: nothing is posted while it is made, and
/// the compute state the guest destroyed during the walk is let go of where the views close,
/// with no buffer completed, before the failure is posted, as without the pre-pass, so what it
/// gives back goes back at the same place in the same order; and the wait is counted
#[test]
fn a_make_ahead_the_walk_never_reaches_ends_before_its_views_close() {
    let (device, platform, mut guest, mut ring) = with_states(&[11]);
    let world = platform.world();
    let state = world.live(CLASS);
    assert_eq!(state.len(), 1);
    let waited = counted("dispatch-prepass-waited");
    let held = HeldCall::new();
    world.hold_prepare(held.clone());
    // An op no decoder knows: the walk faults the buffer there, before its dispatch.
    let unknown = op(0x20ff, &[]);
    let ops = [
        begin(),
        unknown,
        set_state(11),
        threads([64, 1, 1], [32, 1, 1]),
        end(),
    ];
    let gate = WalkGate::on(&device);
    ring.send(&device, command_buffer_of(100, &ops));
    assert!(gate.reached(), "the walk did not start");
    assert!(
        held.reached(Duration::from_secs(60)),
        "nothing was made ahead"
    );
    guest.send(&device, object_destroy(obj(CONTEXT, 11)));
    carried_out_of(&device, &mut guest, CONTEXT);
    assert_eq!(
        world.live(CLASS),
        state,
        "the walk's view keeps the destroyed state"
    );
    gate.open();
    // Were the walk's end not to wait, the stand-in for its failure would be committed and
    // completed here, and the failure posted.
    for _ in 0..50 {
        world.complete_all(CompleteOn::Caller);
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(
        !ring.has_event(),
        "the buffer's failure was posted while its make ahead was still being made"
    );
    assert_eq!(world.live(CLASS), state);
    held.open();
    // Let go of at the views' close, which waits for no completion.
    assert!(
        eventually(|| world.live(CLASS).is_empty()),
        "the destroyed state outlived the buffer's views"
    );
    assert!(
        !ring.has_event(),
        "the failure was posted before its stand-in completed"
    );
    assert_ne!(completed(&platform, &mut ring, 100), RGFX_CB_OK);
    assert!(
        counted("dispatch-prepass-waited") > waited,
        "the wait was not counted"
    );
}
