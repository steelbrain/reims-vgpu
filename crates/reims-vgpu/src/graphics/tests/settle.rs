//! A pipeline create's two steps on its compile's thread (`compile`, "Order"): its reply, posted
//! once the platform's make made every answer it carries, then its store, after the make's finish
//! (`ComputePipeline::finish`) on the same thread. A command buffer and an `ICB_WRITE` that name
//! the pipeline wait for the store; a destroy, a close and a reset between the two cancel it, and
//! a destroy before the finish begins skips it; a finish that fails is counted, its first line
//! logged, keeps no answer and stores the pipeline as made; and a make held for its stage's fill
//! answers after the fill, before its finish.

use std::time::Duration;

use reims_vgpu_graphics::icb::{RGFX_ICB_OP_COMPUTE_PIPELINE, RgfxIcbObject, RgfxIcbRecord};
use reims_vgpu_vulkan::core::platform::resource::ComputePipeline as _;
use reims_vgpu_vulkan::core::platform::stage::{FillEnd, Site};
use reims_vgpu_vulkan::core::platform::types::NsErrorText;
use reims_vgpu_vulkan::core::testkit::impls::resource::FINISH;
use reims_vgpu_vulkan::core::testkit::objects::kind;
use reims_vgpu_vulkan::core::testkit::{Step, World};

use super::prepass::{
    CONTEXT, begin, command_buffer_of, completed, end, set_state, threads, with_states,
};
use super::*;
use crate::graphics::decode::FINISH_FAILED;
use crate::platform::counted;

/// The fake's selector of a [`threads`] dispatch.
const THREADS: &str = "dispatchThreads:threadsPerThreadgroup:";

/// The fake's selector of a compute state's make.
const MAKE: &str = "newComputePipelineStateWithDescriptor:options:reflection:error:";

/// How long a test waits for a held finish to be reached.
const REACHED: Duration = Duration::from_secs(60);

/// A command buffer `cb_obj_id` of one compute encoder that dispatches threads of state `state`
/// of [`CONTEXT`].
fn dispatching(state: u64, cb_obj_id: u64) -> Vec<u8> {
    let ops = [
        begin(),
        set_state(state),
        threads([64, 1, 1], [32, 1, 1]),
        end(),
    ];
    command_buffer_of(cb_obj_id, &ops)
}

/// Whether the buffers of `ring` stay uncompleted for a while, the platform's committed buffers
/// completed meanwhile.
fn none_completes(platform: &FakePlatform, ring: &mut QueueRing) -> bool {
    for _ in 0..50 {
        platform.world().complete_all(CompleteOn::Caller);
        std::thread::sleep(Duration::from_millis(1));
    }
    !ring.has_event()
}

/// Holds the next pipeline state's finish of `world` until the test lets it go.
fn hold_finish(world: &World) -> HeldCall {
    let held = HeldCall::new();
    world.script.push(FINISH, Step::Hold(held.clone()));
    held
}

/// Whether the device holds pipeline `obj_id` pending: its create's store has not come.
fn pending(device: &GraphicsDevice<FakePlatform>, obj_id: u64) -> bool {
    device.control.pending.is_pending(obj_id)
}

/// The next event of ring 0, which must be the reflection of a create of `obj_id` at the
/// guest's last seqno with `status`, taken with no wait for the pipeline's store.
fn reply_unstored(guest: &mut Guest, obj_id: u64, status: u32) -> RgfxPipelineReflection {
    let bytes = guest.event_unstored().expect("a reflection");
    let (reflection, _) = RgfxPipelineReflection::read_from_prefix(&bytes).unwrap();
    assert_eq!(reflection.header.r#type, RGFX_EVT_PIPELINE_REFLECTION);
    assert_eq!(reflection.header.seqno, guest.seqno);
    assert_eq!((reflection.obj_id, reflection.status), (obj_id, status));
    reflection
}

/// A compute create of state `obj_id` over library `library` that asks for its pipeline key.
fn keyed_create(obj_id: u64, library: u64) -> Vec<u8> {
    let mut message = compute_pipeline_create(obj_id, library);
    let at = std::mem::offset_of!(RgfxComputePipelineCreate, flags);
    let flags = u32::from_le_bytes(message[at..at + 4].try_into().unwrap())
        | RGFX_COMPUTE_PIPELINE_ANSWER_KEY;
    message[at..at + 4].copy_from_slice(&flags.to_le_bytes());
    message
}

/// A pipeline create's reply is posted before its store: with its make's finish held, the guest
/// has the reflection while the pipeline is pending and not in the table; the finish runs on the
/// compile's thread, once, and the store follows it. Every answer the reply carries is the one
/// the state gives after its finish
#[test]
fn a_create_s_reply_comes_before_its_store_with_the_state_s_answers() {
    let (device, platform, mut guest, _ring) = with_states(&[]);
    let world = platform.world();
    let held = hold_finish(world);
    let id = obj(CONTEXT, 11);
    guest.send(&device, compute_pipeline_create(id, obj(CONTEXT, 10)));
    let reply = reply_unstored(&mut guest, id, RGFX_PIPELINE_OK);
    assert!(held.reached(REACHED), "the make was never finished");
    assert!(
        pending(&device, id),
        "the pipeline was stored before its finish"
    );
    // The queue and the library.
    assert_eq!(device.decoder_object_count(), 2);
    held.open();
    assert!(eventually(|| !pending(&device, id)), "the store never came");
    assert_eq!(device.decoder_object_count(), 3);

    let finished = world.finished();
    assert_eq!(finished.len(), 1, "{finished:?}");
    let thread = finished[0].thread.as_deref().unwrap_or("");
    assert!(
        thread.starts_with("reims-vgpu.graphics-compile-"),
        "the finish ran on {thread:?}"
    );
    let state = (world.find(finished[0].state))
        .and_then(|h| h.cast::<kind::ComputePipeline>())
        .expect("the stored state");
    assert_eq!(
        (
            reply.max_total_threads_per_threadgroup,
            reply.static_threadgroup_memory_length,
            u64::from(reply.thread_execution_width),
            reply.gpu_resource_id,
            reply.unique_identifier,
            u64::from(reply.resource_index),
            reply.flags & RGFX_PIPELINE_REFLECT_SUPPORT_ICB != 0,
        ),
        (
            state.max_total_threads_per_threadgroup(),
            state.static_threadgroup_memory_length(),
            state.thread_execution_width(),
            state.gpu_resource_id(),
            state.guest_unique_identifier(),
            state.resource_index(),
            state.support_indirect_command_buffers(),
        )
    );
}

/// A command buffer that names a pipeline whose create was answered and whose finish is slow
/// waits for its store, then runs with it, its dispatch sent once
#[test]
fn a_command_buffer_naming_a_pipeline_whose_finish_is_slow_waits_for_it() {
    let (device, platform, mut guest, mut ring) = with_states(&[]);
    let world = platform.world();
    let held = hold_finish(world);
    let id = obj(CONTEXT, 11);
    guest.send(&device, compute_pipeline_create(id, obj(CONTEXT, 10)));
    reply_unstored(&mut guest, id, RGFX_PIPELINE_OK);
    assert!(held.reached(REACHED), "the make was never finished");
    ring.send(&device, dispatching(11, 100));
    assert!(
        none_completes(&platform, &mut ring),
        "the buffer completed before the pipeline it names was stored"
    );
    assert_eq!(world.log.count(THREADS), 0);
    held.open();
    assert_eq!(completed(&platform, &mut ring, 100), RGFX_CB_OK);
    assert_eq!(world.log.count(THREADS), 1);
}

/// An `ICB_WRITE` that sets a pipeline whose create was answered and whose finish is slow waits
/// for its store on its context's lane, and the context's later messages with it
#[test]
fn an_icb_write_setting_a_pipeline_whose_finish_is_slow_waits_for_it() {
    let raised = Arc::new(AtomicU32::new(0));
    let (device, platform) = icb_device(&raised);
    let world = platform.world();
    let mut guest = Guest::new(&device);
    make_library(&device, &mut guest, obj(1, 10));
    let icb = obj(1, 50);
    guest.send(&device, icb_create(icb));
    next_event(&mut guest, RGFX_EVT_ICB_STATE);
    let held = hold_finish(world);
    let id = obj(1, 11);
    guest.send(&device, compute_pipeline_create(id, obj(1, 10)));
    reply_unstored(&mut guest, id, RGFX_PIPELINE_OK);
    assert!(held.reached(REACHED), "the make was never finished");

    let body = RgfxIcbObject { obj_id: id };
    let record = RgfxIcbRecord {
        op: RGFX_ICB_OP_COMPUTE_PIPELINE,
        size: (size_of::<RgfxIcbRecord>() + size_of::<RgfxIcbObject>()) as u16,
        command_index: 0,
    };
    let mut write = RgfxIcbWrite {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_ICB_WRITE,
            size: 0,
            seqno: 0,
        },
        obj_id: icb,
    }
    .as_bytes()
    .to_vec();
    write.extend_from_slice(record.as_bytes());
    write.extend_from_slice(body.as_bytes());
    guest.send(&device, write);
    guest.send(&device, library_lookup(obj(1, 0xfff), 0, [0xee; 32]));
    let lookup = guest.seqno;
    assert!(
        guest.quiet_for(50),
        "the context's lookup behind the write was answered before the pipeline's store"
    );
    held.open();
    let bytes = next_event(&mut guest, RGFX_EVT_COMPLETE);
    let done = RgfxComplete::read_from_bytes(&bytes).unwrap();
    assert_eq!(done.header.seqno, lookup);
    assert!(!pending(&device, id));
}

/// A finish that fails is the create's late failure: the create stays answered made, the
/// failure is counted each time ([`FINISH_FAILED`]) and its first line logged at warn, the kept
/// answer of its key is dropped and none kept, and the pipeline is stored as made, its dispatches
/// sent to the platform, where the late failure meets them, and its buffer completing; the VM
/// goes on, a later create made and finished
#[test]
fn a_failed_finish_is_counted_keeps_no_answer_and_stores_the_pipeline_made() {
    use reims_vgpu_vulkan::core::logging;

    let lines = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&lines);
    assert!(logging::set_hook(
        log::LevelFilter::Warn,
        Box::new(move |_, _, message| seen.lock().push(message.to_owned())),
    ));
    let before = counted(FINISH_FAILED);
    let (device, platform, mut guest, mut ring) = with_states(&[]);
    let world = platform.world();
    let (bytes, key) = keyed_metallib();
    let library = obj(CONTEXT, 20);
    let made = keyed_library_create(library, &bytes, Some(key));
    assert_eq!(answered(&device, &mut guest, made), RGFX_STATUS_OK);

    // A keyed create whose make finished: its answer is kept.
    guest.send(&device, keyed_create(obj(CONTEXT, 11), library));
    expect_reflection_prefix(&mut guest, obj(CONTEXT, 11));
    assert_eq!(device.control.answers.len(), 1);

    // An equal keyed create, and an unkeyed one, whose finishes fail.
    let failure = NsErrorText {
        domain: "VkResult".to_owned(),
        code: -2,
        description: "VK_ERROR_OUT_OF_DEVICE_MEMORY".to_owned(),
    };
    world.script.push_n(FINISH, 2, Step::Error(failure));
    guest.send(&device, keyed_create(obj(CONTEXT, 12), library));
    expect_reflection_prefix(&mut guest, obj(CONTEXT, 12));
    assert_eq!(
        device.control.answers.len(),
        0,
        "a create whose finish failed kept its key's answer"
    );
    guest.send(
        &device,
        compute_pipeline_create(obj(CONTEXT, 13), obj(CONTEXT, 10)),
    );
    expect_reflection(&mut guest, obj(CONTEXT, 13), RGFX_PIPELINE_OK);
    // The queue, both libraries and the three states.
    assert_eq!(device.decoder_object_count(), 6);

    ring.send(&device, dispatching(12, 100));
    assert_eq!(completed(&platform, &mut ring, 100), RGFX_CB_OK);
    assert_eq!(world.log.count(THREADS), 1);
    guest.send(
        &device,
        compute_pipeline_create(obj(CONTEXT, 14), obj(CONTEXT, 10)),
    );
    expect_reflection(&mut guest, obj(CONTEXT, 14), RGFX_PIPELINE_OK);
    logging::clear_hook();
    assert_eq!(device.decoder_object_count(), 7);
    assert_eq!(world.finished().len(), 4);
    assert_eq!(counted(FINISH_FAILED) - before, 2);
    let failed: Vec<String> = (lines.lock().iter())
        .filter(|line| line.contains("finish failed"))
        .cloned()
        .collect();
    assert_eq!(failed.len(), 1, "{failed:?}");
    assert!(
        failed[0].contains("VK_ERROR_OUT_OF_DEVICE_MEMORY"),
        "{failed:?}"
    );
}

/// The next event of ring 0, which must be a made reflection of a keyed create of `obj_id` at the
/// guest's last seqno, the key after it.
fn expect_reflection_prefix(guest: &mut Guest, obj_id: u64) {
    let bytes = next_event(guest, RGFX_EVT_PIPELINE_REFLECTION);
    let (reflection, _) = RgfxPipelineReflection::read_from_prefix(&bytes).unwrap();
    assert_eq!(reflection.header.seqno, guest.seqno);
    assert_eq!(
        (reflection.obj_id, reflection.status),
        (obj_id, RGFX_PIPELINE_OK)
    );
    assert_ne!(reflection.flags & RGFX_PIPELINE_REFLECT_KEYED, 0);
}

/// An `OBJECT_DESTROY` between a create's reply and its store cancels the pipeline: its finish
/// ends, and it is let go of, not stored
#[test]
fn a_destroy_between_a_create_s_reply_and_its_store_cancels_it() {
    let (device, platform, mut guest, _ring) = with_states(&[]);
    let world = platform.world();
    let held = hold_finish(world);
    let id = obj(CONTEXT, 11);
    guest.send(&device, compute_pipeline_create(id, obj(CONTEXT, 10)));
    reply_unstored(&mut guest, id, RGFX_PIPELINE_OK);
    assert!(held.reached(REACHED), "the make was never finished");
    guest.send(&device, object_destroy(id));
    // The destroy is carried out before a lookup of its context's after it is answered.
    carried_out_of(&device, &mut guest, CONTEXT);
    held.open();
    let state = world.finished()[0].state;
    assert!(
        eventually(|| world.find(state).is_none()),
        "the destroyed pipeline was kept"
    );
    assert!(!pending(&device, id));
    // The queue and the library.
    assert_eq!(device.decoder_object_count(), 2);
}

/// An `OBJECT_DESTROY` carried out before a create's compile ends cancels it before its finish
/// begins: the create is still answered, its finish is skipped, and its pipeline is not stored
#[test]
fn a_create_destroyed_before_its_finish_begins_is_not_finished() {
    let (device, platform, mut guest, _ring) = with_states(&[]);
    let world = platform.world();
    let id = obj(CONTEXT, 11);
    device.control.pool.hold();
    guest.send(&device, compute_pipeline_create(id, obj(CONTEXT, 10)));
    guest.send(&device, object_destroy(id));
    // The destroy is carried out before a lookup of its context's after it is answered.
    carried_out_of(&device, &mut guest, CONTEXT);
    device.control.pool.release();
    let bytes = next_event(&mut guest, RGFX_EVT_PIPELINE_REFLECTION);
    let (reflection, _) = RgfxPipelineReflection::read_from_prefix(&bytes).unwrap();
    assert_eq!(
        (reflection.obj_id, reflection.status),
        (id, RGFX_PIPELINE_OK)
    );
    assert!(
        eventually(|| !pending(&device, id)),
        "the create never settled"
    );
    let finished = world.finished();
    assert!(
        finished.is_empty(),
        "a cancelled create was finished: {finished:?}"
    );
    // The queue and the library.
    assert_eq!(device.decoder_object_count(), 2);
}

/// A context's close between a create's reply and its store waits for neither, and cancels the
/// pipeline: its finish ends, and it is let go of, not stored in the closed context
#[test]
fn a_close_between_a_create_s_reply_and_its_store_cancels_it() {
    let (device, platform, mut guest, _ring) = with_states(&[]);
    let world = platform.world();
    let held = hold_finish(world);
    let id = obj(CONTEXT, 11);
    guest.send(&device, compute_pipeline_create(id, obj(CONTEXT, 10)));
    reply_unstored(&mut guest, id, RGFX_PIPELINE_OK);
    assert!(held.reached(REACHED), "the make was never finished");
    assert_eq!(
        answered(&device, &mut guest, context_close(CONTEXT)),
        RGFX_STATUS_OK
    );
    assert_eq!(device.decoder_object_count(), 0);
    held.open();
    let state = world.finished()[0].state;
    assert!(
        eventually(|| world.find(state).is_none()),
        "the closed context's pipeline was kept"
    );
    assert!(!pending(&device, id));
    assert_eq!(device.decoder_object_count(), 0);
}

/// The device's reset between a create's reply and its store waits for neither: the reset ends
/// with the finish held, and the pipeline is then let go of, not stored
#[test]
fn a_reset_between_a_create_s_reply_and_its_store_lets_go_of_it() {
    let (device, platform, mut guest, _ring) = with_states(&[]);
    let world = platform.world();
    let held = hold_finish(world);
    let id = obj(CONTEXT, 11);
    guest.send(&device, compute_pipeline_create(id, obj(CONTEXT, 10)));
    reply_unstored(&mut guest, id, RGFX_PIPELINE_OK);
    assert!(held.reached(REACHED), "the make was never finished");
    // The host hears of the reset on the device's control thread.
    device.reset();
    let reset = eventually(|| device.decoder_object_count() == 0 && !pending(&device, id));
    held.open();
    assert!(reset, "the reset waited for the held finish");
    let state = world.finished()[0].state;
    assert!(
        eventually(|| world.find(state).is_none()),
        "the pipeline of before the reset was kept"
    );
    assert_eq!(device.decoder_object_count(), 0);
}

/// A compute state's make refused for its stage holds its create, unanswered and unfinished,
/// until the stage's fill ends; the make made again after a fill that filled the store answers
/// the create, and its finish, which comes after the reply, is the one that runs
#[test]
fn a_make_held_for_its_stage_answers_after_its_fill_before_its_finish() {
    let (device, platform, mut guest, _ring) = with_states(&[]);
    let world = platform.world();
    let held = hold_finish(world);
    world.script.push(MAKE, Step::Stage(7));
    let id = obj(CONTEXT, 11);
    guest.send(&device, compute_pipeline_create(id, obj(CONTEXT, 10)));
    assert!(
        eventually(|| world.stages().started_at() == [(7, Site::Make)]),
        "no fill started"
    );
    assert!(
        guest.quiet_for(50),
        "the create was answered before its stage's fill ended"
    );
    assert!(world.finished().is_empty(), "a refused make was finished");
    assert!(world.stages().end(7, FillEnd::Filled));
    reply_unstored(&mut guest, id, RGFX_PIPELINE_OK);
    assert!(
        held.reached(REACHED),
        "the make made again was never finished"
    );
    assert!(
        pending(&device, id),
        "the pipeline was stored before its finish"
    );
    held.open();
    assert!(eventually(|| !pending(&device, id)), "the store never came");
    assert_eq!(device.decoder_object_count(), 3);
    assert_eq!(world.finished().len(), 1);
    assert_eq!(world.log.count(MAKE), 2);
}
