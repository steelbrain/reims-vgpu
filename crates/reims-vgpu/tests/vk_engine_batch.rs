//! Off-VM regression suite for draw batching increment 1 (deferred submit).
//!
//! Proves that same-target skip_readback draws share one command buffer
//! (opener + joiners), that content composes correctly across the open pass
//! boundary (LoadFromTarget inside the batch), and that every consumer path
//! (read_target, cross-target draw) flushes the open batch before touching
//! GPU content. Requires a working Vulkan ICD; skips cleanly if init fails.
//!
//! **Serial:** the engine is process-global; all tests take the suite lock.

use metal2vulkan::passes::Stage;
use reims_vgpu_vulkan::engine::{
    self, BlendFactor, BlendOp, BlendStateResource, BufferContent, DepthAttachment, DepthState,
    DrawRequest, GuestRun, GuestRunSource, IndexType, IndexedDrawResource, PrimitiveTopology,
    SampledImageResource, SampledSource, SamplerCompareFunction, SamplerResource, ScissorResource,
    StorageBufferResource, TargetIdentity, VertexAttributeFormat, VertexAttributeResource,
    VertexStepFunction,
};
/// The resident format every `TargetIdentity::Surface` in this file is built at.
///
/// These tests predate the namespace carrying a format, and each was written
/// against a resident in guest scanout order — several assert on the byte order
/// of what they read back. Naming the constant once keeps that premise in one
/// place and makes a test that wants a different format say so.
const SURFACE_TEST_FORMAT: reims_vgpu_core::pixel_format::TexelLayout =
    reims_vgpu_core::pixel_format::TexelLayout::Bgra8;

use std::path::PathBuf;
use std::process::Command;
use std::sync::{Mutex, OnceLock};

fn engine_test_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| {
        // Never share the live product logs with a concurrent boot.
        reims_vgpu::observe::redirect_logs_for_tests();
        Mutex::new(())
    })
}

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/air")
}

fn translate_words(name: &str, stage: Stage) -> Vec<u32> {
    let tmp = std::env::temp_dir().join(format!(
        "paravirt_engine_batch_{}_{}_{:?}",
        std::process::id(),
        name,
        stage
    ));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).expect("tmp");
    let path = fixtures().join(name);
    assert!(path.exists(), "missing reims-vgpu AIR fixture: {name}");
    let spv = metal2vulkan::translate(path.to_str().unwrap(), stage, &tmp)
        .unwrap_or_else(|e| panic!("translate {name}: {e}"));
    spv.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn triangle_spirv() -> (Vec<u32>, Vec<u32>) {
    (
        translate_words("render_tri.air", Stage::Vertex),
        translate_words("render_frag.air", Stage::Fragment),
    )
}

fn skip_if_no_gpu(err: &str) -> bool {
    let lower = err.to_ascii_lowercase();
    lower.contains("no vulkan")
        || lower.contains("load vulkan")
        || lower.contains("create_instance")
        || lower.contains("no graphics")
        || lower.contains("vk_engine_init")
}

/// float4(0.25, 0.5, 0.75, 1.0) → unorm8 ≈ (64, 128, 191, 255); allow ±1 LSB.
fn near(got: u8, want: u8) -> bool {
    (got as i32 - want as i32).abs() <= 1
}

fn is_frag_color(px: &[u8]) -> bool {
    near(px[0], 64) && near(px[1], 128) && near(px[2], 191) && near(px[3], 255)
}

fn is_zero(px: &[u8]) -> bool {
    px == [0, 0, 0, 0]
}

const W: u32 = 64;
const H: u32 = 64;

fn batch_req(
    vert: &[u32],
    frag: &[u32],
    identity: &TargetIdentity,
    load_from_target: bool,
    scissor: ScissorResource,
) -> DrawRequest {
    DrawRequest {
        program: reims_vgpu_core::PreparedRenderProgram {
            vertex: reims_vgpu_vulkan::m2v_cache::prepare_test_shader(vert.to_vec()),
            fragment: reims_vgpu_vulkan::m2v_cache::prepare_test_shader(frag.to_vec()),
        },
        width: W,
        height: H,
        vertex_count: 3,
        first_vertex: 0,
        instance_count: Some(1),
        base_instance: 0,
        primitive_topology: PrimitiveTopology::Triangle,
        target_identity: Some(identity.clone()),
        load_from_target,
        skip_readback: true,
        scissors: vec![scissor],
        ..Default::default()
    }
}

fn half_scissor(left: bool) -> ScissorResource {
    ScissorResource {
        x: if left { 0 } else { W / 2 },
        y: 0,
        width: W / 2,
        height: H,
    }
}

fn submission(id: u64) -> reims_vgpu_core::SubmissionContext {
    reims_vgpu_core::SubmissionContext {
        identity: reims_vgpu_protocol::SubmissionIdentity {
            id: reims_vgpu_protocol::SubmissionId::new(id),
            task: reims_vgpu_protocol::TaskId::new(77),
        },
        resources: std::sync::Arc::from([]),
        segments: std::sync::Arc::from([]),
        segment: None,
    }
}

/// Opener (Clear, left half) + joiner (LoadFromTarget, right half) share one
/// CB; the flush at read_target submits both and the readback shows BOTH
/// halves colored — the joiner's LOAD preserved the opener's half across the
/// intra-CB pass boundary.
#[test]
fn batched_draws_compose_and_flush_on_read() {
    let _guard = engine_test_lock().lock().unwrap();
    let (vert, frag) = triangle_spirv();
    let identity = TargetIdentity::Surface {
        id: 990_101,
        width: W,
        height: H,
        generation: 1,
        format: SURFACE_TEST_FORMAT,
    };

    let before = engine::counter_snapshot();
    let opener = batch_req(&vert, &frag, &identity, false, half_scissor(true));
    match engine::execute_draw_request(&opener) {
        Ok(out) => assert!(out.pixels.is_empty(), "skip_readback returns no pixels"),
        Err(e) => {
            let msg = e.to_string();
            if skip_if_no_gpu(&msg) {
                eprintln!("skipping: {msg}");
                return;
            }
            panic!("opener draw: {msg}");
        }
    }
    let joiner = batch_req(&vert, &frag, &identity, true, half_scissor(false));
    engine::execute_draw_request(&joiner).expect("joiner draw");
    let mid = engine::counter_snapshot().delta_since(&before);
    assert_eq!(mid.batch_opens, 1, "first draw opens the batch");
    assert_eq!(mid.batch_joins, 1, "second draw joins the open CB");
    assert_eq!(mid.batch_flushes, 0, "no flush before a consumer arrives");

    let px = engine::read_target(&identity)
        .expect("read_target flushes the batch")
        .into_rgba8();
    let after = engine::counter_snapshot().delta_since(&before);
    assert_eq!(after.batch_flushes, 1, "read_target submitted the batch");
    assert_eq!(
        after.batch_flush_draws, 2,
        "the one submit carried both draws"
    );
    assert_eq!(
        after.queue_async_submits, 1,
        "the ended batch must execute through the asynchronous queue owner"
    );

    assert_eq!(px.len(), (W * H * 4) as usize);
    for y in [0u32, H / 2, H - 1] {
        for x in [0u32, W / 4, W / 2, 3 * W / 4, W - 1] {
            let i = ((y * W + x) * 4) as usize;
            assert!(
                is_frag_color(&px[i..i + 4]),
                "batched composite at ({x},{y}) = {:?}",
                &px[i..i + 4]
            );
        }
    }
}

#[test]
fn a_refused_submission_tail_keeps_the_recorded_resident_prefix() {
    let _guard = engine_test_lock().lock().unwrap();
    engine::flush_batched_draws();
    let (vert, frag) = triangle_spirv();
    let identity = TargetIdentity::Surface {
        id: 990_111,
        width: W,
        height: H,
        generation: 1,
        format: SURFACE_TEST_FORMAT,
    };
    let context = submission(111);
    let left = batch_req(&vert, &frag, &identity, false, half_scissor(true));
    let right = batch_req(&vert, &frag, &identity, true, half_scissor(false));
    let invalid_tail = DrawRequest::default();
    let commands = reims_vgpu_core::ResolvedCommandBuffer::new(vec![
        reims_vgpu_core::ResolvedCommand::Draw(Box::new(left)),
        reims_vgpu_core::ResolvedCommand::Draw(Box::new(right)),
        reims_vgpu_core::ResolvedCommand::Draw(Box::new(invalid_tail)),
    ]);
    let progress = engine::execute_submission_progress(reims_vgpu_core::ResolvedSubmission {
        context: context.clone(),
        command_buffer: commands,
    });
    if let Some(error) = progress.failure.as_ref() {
        if progress.output.is_empty() && skip_if_no_gpu(&error.to_string()) {
            eprintln!("skipping: {error}");
            return;
        }
    }
    assert_eq!(
        progress.output.len(),
        2,
        "the exact recorded prefix completed"
    );
    assert!(progress.failure.is_some(), "the invalid tail is refused");
    engine::close_submission(context.identity).expect("the prefix remains submittable");

    let pixels = engine::read_target(&identity)
        .expect("the refused tail did not discard the prefix")
        .into_rgba8();
    for y in [0u32, H / 2, H - 1] {
        for x in [0u32, W / 4, W / 2, 3 * W / 4, W - 1] {
            let offset = ((y * W + x) * 4) as usize;
            assert!(
                is_frag_color(&pixels[offset..offset + 4]),
                "prefix pixel at ({x},{y}) = {:?}",
                &pixels[offset..offset + 4]
            );
        }
    }
}

#[test]
fn one_recording_retains_unchanged_vertex_buffer_state() {
    let _guard = engine_test_lock().lock().unwrap();
    engine::flush_batched_draws();
    let (vert, frag) = triangle_spirv();
    let identity = TargetIdentity::Surface {
        id: 990_106,
        width: W,
        height: H,
        generation: 1,
        format: SURFACE_TEST_FORMAT,
    };
    let content = std::sync::Arc::new(vec![0u8; 24]);
    let make = |load_from_target, left| {
        let mut request = batch_req(
            &vert,
            &frag,
            &identity,
            load_from_target,
            half_scissor(left),
        );
        for (location, binding) in [(0, 2), (1, 0), (2, 1)] {
            request.vertex_attributes.push(VertexAttributeResource {
                location,
                binding,
                format: VertexAttributeFormat::Float2,
                offset: 0,
                stride: 8,
                step_function: VertexStepFunction::PerVertex,
                step_rate: 1,
                content: BufferContent::Bytes(std::sync::Arc::clone(&content)),
            });
        }
        request
    };
    let before = engine::counter_snapshot();
    if let Err(error) = engine::execute_draw_request(&make(false, true)) {
        let message = error.to_string();
        if skip_if_no_gpu(&message) {
            eprintln!("skipping: {message}");
            return;
        }
        panic!("first retained-state draw: {message}");
    }
    engine::execute_draw_request(&make(true, false)).expect("second retained-state draw");
    let pixels = engine::read_target(&identity)
        .expect("read target")
        .into_rgba8();
    let delta = engine::counter_snapshot().delta_since(&before);
    assert_eq!(delta.vertex_buffer_bind_slots, 6, "requested: {delta:?}");
    assert_eq!(delta.vertex_buffer_bind_emitted, 3, "emitted: {delta:?}");
    assert_eq!(delta.vertex_buffer_bind_held, 3, "retained: {delta:?}");
    assert_eq!(delta.vertex_buffer_bind_calls, 1, "setter calls: {delta:?}");
    assert!(pixels.chunks_exact(4).any(is_frag_color));
}

#[test]
fn one_recording_retains_unchanged_blend_constants() {
    let _guard = engine_test_lock().lock().unwrap();
    engine::flush_batched_draws();
    let (vert, frag) = triangle_spirv();
    let identity = TargetIdentity::Surface {
        id: 990_107,
        width: W,
        height: H,
        generation: 1,
        format: SURFACE_TEST_FORMAT,
    };
    let make = |load_from_target, left| {
        let mut request = batch_req(
            &vert,
            &frag,
            &identity,
            load_from_target,
            half_scissor(left),
        );
        request.blend_constants = [0.2, 0.4, 0.6, 0.8];
        request.blend = Some(BlendStateResource {
            src_color: BlendFactor::ConstantColor,
            dst_color: BlendFactor::Zero,
            color_op: BlendOp::Add,
            src_alpha: BlendFactor::ConstantAlpha,
            dst_alpha: BlendFactor::Zero,
            alpha_op: BlendOp::Add,
        });
        request
    };
    let before = engine::counter_snapshot();
    if let Err(error) = engine::execute_draw_request(&make(false, true)) {
        let message = error.to_string();
        if skip_if_no_gpu(&message) {
            eprintln!("skipping: {message}");
            return;
        }
        panic!("first retained blend-state draw: {message}");
    }
    engine::execute_draw_request(&make(true, false)).expect("second retained blend-state draw");
    let pixels = engine::read_target(&identity)
        .expect("read target")
        .into_rgba8();
    let delta = engine::counter_snapshot().delta_since(&before);
    assert_eq!(
        delta.dynstate_blend_constants_held, 1,
        "the second draw retained the setter value: {delta:?}"
    );
    assert!(pixels.iter().any(|byte| *byte != 0));
}

/// Each decoded exec owns one native command buffer. Continuation records may
/// retain state inside that exec, but its close commits the packet before the
/// next identity can claim the encoder.
#[test]
fn submission_close_commits_each_native_packet() {
    let _guard = engine_test_lock().lock().unwrap();
    engine::flush_batched_draws();
    let (vert, frag) = triangle_spirv();
    let identity = TargetIdentity::Surface {
        id: 990_111,
        width: W,
        height: H,
        generation: 1,
        format: SURFACE_TEST_FORMAT,
    };
    let first_submission = submission(9_901);
    let second_submission = submission(9_902);
    let before = engine::counter_snapshot();

    let left = batch_req(&vert, &frag, &identity, false, half_scissor(true));
    if let Err(error) = engine::execute_draw_request_in_submission(&first_submission, &left) {
        let _ = engine::close_submission(first_submission.identity);
        let message = error.to_string();
        if skip_if_no_gpu(&message) {
            eprintln!("skipping: {message}");
            return;
        }
        panic!("first submission draw: {message}");
    }
    assert_eq!(
        engine::counter_snapshot()
            .delta_since(&before)
            .batch_flushes,
        0,
        "the packet remains open until its exact close event"
    );
    engine::close_submission(first_submission.identity).expect("close first submission");
    let first_close = engine::counter_snapshot().delta_since(&before);
    assert_eq!(
        first_close.batch_flushes, 1,
        "the first packet close commits its retained encoder"
    );
    assert_eq!(first_close.batch_flush_draws, 1);

    let right = batch_req(&vert, &frag, &identity, true, half_scissor(false));
    engine::execute_draw_request_in_submission(&second_submission, &right)
        .expect("the next submission claims the released encoder");
    engine::close_submission(second_submission.identity).expect("close second submission");

    let delta = engine::counter_snapshot().delta_since(&before);
    assert_eq!(
        delta.batch_flushes, 2,
        "each exact packet owns one native commit"
    );
    assert_eq!(delta.batch_flush_draws, 2);

    let pixels = engine::read_target(&identity)
        .expect("ordered submissions preserve the target")
        .into_rgba8();
    let flushed = engine::counter_snapshot().delta_since(&before);
    assert_eq!(
        flushed.batch_flushes, 2,
        "the read finds no cross-packet tail"
    );
    assert_eq!(
        flushed.batch_flush_draws, 2,
        "each ordered submission retained its own draw"
    );
    for y in [0u32, H / 2, H - 1] {
        for x in [0u32, W / 4, W / 2, 3 * W / 4, W - 1] {
            let offset = ((y * W + x) * 4) as usize;
            assert!(
                is_frag_color(&pixels[offset..offset + 4]),
                "cross-submission composite at ({x},{y}) = {:?}",
                &pixels[offset..offset + 4]
            );
        }
    }
}

/// The first and second draws from one decoded Metal render encoder retain one
/// Vulkan pass instance even though the serialized continuation forces the
/// second record's load action to LOAD. Load/store actions describe how a pass
/// instance begins and ends; they do not make otherwise-identical render passes
/// incompatible, and the second record never begins a pass when the first one
/// remains open. The continuation counter makes this fail if the load-action
/// rewrite regresses to two begin/end pairs, while the pixel check keeps the
/// optimized recording honest.
#[test]
fn one_metal_encoder_continues_one_vulkan_render_pass() {
    let _guard = engine_test_lock().lock().unwrap();
    let (vert, frag) = triangle_spirv();
    let identity = TargetIdentity::Surface {
        id: 990_102,
        width: W,
        height: H,
        generation: 1,
        format: SURFACE_TEST_FORMAT,
    };

    let before = engine::counter_snapshot();
    let mut first = batch_req(&vert, &frag, &identity, false, half_scissor(true));
    first.render_pass_continues = true;
    match engine::execute_draw_request(&first) {
        Ok(_) => {}
        Err(e) => {
            let msg = e.to_string();
            if skip_if_no_gpu(&msg) {
                eprintln!("skipping: {msg}");
                return;
            }
            panic!("encoder first draw: {msg}");
        }
    }

    let mut second = batch_req(&vert, &frag, &identity, true, half_scissor(false));
    second.continues_render_pass = true;
    engine::execute_draw_request(&second).expect("encoder second draw");

    let px = engine::read_target(&identity)
        .expect("flush continued pass")
        .into_rgba8();
    let delta = engine::counter_snapshot().delta_since(&before);
    assert_eq!(
        delta.render_pass_continuations, 1,
        "the second encoder draw must reuse the pass instance: {delta:?}"
    );
    for y in [0u32, H / 2, H - 1] {
        for x in [W / 4, 3 * W / 4] {
            let i = ((y * W + x) * 4) as usize;
            assert!(
                is_frag_color(&px[i..i + 4]),
                "continued render pass at ({x},{y}) = {:?}",
                &px[i..i + 4]
            );
        }
    }
}

/// A resolve-only attachment keeps its multisample source alive for the whole
/// guest encoder. The first draw cannot resolve and discard that source before
/// the second draw: both halves must be present when the encoder-ending read
/// finally closes the Vulkan pass and resolves it.
#[test]
fn one_multisample_encoder_resolves_after_its_last_draw() {
    let _guard = engine_test_lock().lock().unwrap();
    let (vert, frag) = triangle_spirv();
    let identity = TargetIdentity::Surface {
        id: 990_103,
        width: W,
        height: H,
        generation: 1,
        format: SURFACE_TEST_FORMAT,
    };

    let before = engine::counter_snapshot();
    let mut first = batch_req(&vert, &frag, &identity, false, half_scissor(true));
    first.raster_sample_count = 4;
    first.color_sample_count = 4;
    first.multisample_resolve = true;
    first.render_pass_continues = true;
    match engine::execute_draw_request(&first) {
        Ok(_) => {}
        Err(e) => {
            let msg = e.to_string();
            if skip_if_no_gpu(&msg) {
                eprintln!("skipping: {msg}");
                return;
            }
            panic!("multisample encoder first draw: {msg}");
        }
    }

    let mut second = batch_req(&vert, &frag, &identity, true, half_scissor(false));
    second.raster_sample_count = 4;
    second.color_sample_count = 4;
    second.multisample_resolve = true;
    second.continues_render_pass = true;
    engine::execute_draw_request(&second).expect("multisample encoder second draw");

    let px = engine::read_target(&identity)
        .expect("close and resolve the multisample encoder")
        .into_rgba8();
    let delta = engine::counter_snapshot().delta_since(&before);
    assert_eq!(
        delta.render_pass_continuations, 1,
        "the resolve must occur after both draws in one pass: {delta:?}"
    );
    for y in [0u32, H / 2, H - 1] {
        for x in [W / 4, 3 * W / 4] {
            let i = ((y * W + x) * 4) as usize;
            assert!(
                is_frag_color(&px[i..i + 4]),
                "resolved encoder at ({x},{y}) = {:?}",
                &px[i..i + 4]
            );
        }
    }
}

/// A stored multisample texture is itself the resource. It remains resident
/// between encoders; the second encoder loads that image instead of requiring
/// an implicit resolve or a linear image-to-buffer copy.
#[test]
fn stored_multisample_target_survives_for_a_later_encoder() {
    let _guard = engine_test_lock().lock().unwrap();
    let (vert, frag) = triangle_spirv();
    let identity = TargetIdentity::Surface {
        id: 990_104,
        width: W,
        height: H,
        generation: 1,
        format: SURFACE_TEST_FORMAT,
    };
    let depth_identity = TargetIdentity::Texture {
        ref_: 990_105,
        width: W,
        height: H,
        generation: 0,
        stencil: false,
    };

    let mut first = batch_req(&vert, &frag, &identity, false, half_scissor(true));
    first.raster_sample_count = 2;
    first.color_sample_count = 2;
    first.depth_attachment = Some(DepthAttachment {
        resource_lifetime: reims_vgpu_core::ResourceLifetime::new().reference(),
        identity: depth_identity.clone(),
        depth: Some(reims_vgpu_core::DepthAspectAttachment {
            load_action: reims_vgpu_protocol::pass_action::LoadAction::Clear,
            store_action: reims_vgpu_protocol::pass_action::StoreAction::Store,
            clear_value: 1.0,
        }),
        stencil: None,
    });
    first.depth = Some(DepthState {
        test_enable: true,
        write_enable: true,
        compare: SamplerCompareFunction::Always,
        stencil: None,
    });
    match engine::execute_draw_request(&first) {
        Ok(out) => assert!(
            out.pixels.is_empty(),
            "stored multisample target stays GPU-resident"
        ),
        Err(e) => {
            let msg = e.to_string();
            if skip_if_no_gpu(&msg) {
                eprintln!("skipping: {msg}");
                return;
            }
            panic!("stored multisample opener: {msg}");
        }
    }
    assert!(engine::resident_content_ready(&identity));

    let mut second = batch_req(&vert, &frag, &identity, true, half_scissor(false));
    second.raster_sample_count = 2;
    second.color_sample_count = 2;
    second.depth_attachment = Some(DepthAttachment {
        resource_lifetime: reims_vgpu_core::ResourceLifetime::new().reference(),
        identity: depth_identity,
        depth: Some(reims_vgpu_core::DepthAspectAttachment {
            load_action: reims_vgpu_protocol::pass_action::LoadAction::Load,
            store_action: reims_vgpu_protocol::pass_action::StoreAction::Store,
            clear_value: 1.0,
        }),
        stencil: None,
    });
    second.depth = Some(DepthState {
        test_enable: true,
        write_enable: true,
        compare: SamplerCompareFunction::Always,
        stencil: None,
    });
    engine::execute_draw_request(&second).expect("later encoder loads multisample resident");
    assert!(engine::resident_content_ready(&identity));
}

/// A draw to a DIFFERENT target joins the open batch, and both targets still
/// receive exactly their own draw's pixels.
///
/// This is the whole of what dropping the target from the join key has to be
/// true for. Every batched draw begins and ends its own render pass, so the two
/// passes recorded here name two different attachments and neither may write
/// the other — but nothing in Vulkan says so on its own, and a batch that
/// carried the target for a reason nobody had written down would fail here as a
/// wrong half of a wrong image rather than as a counter.
///
/// It read `batch_opens=2, batch_joins=0, batch_flushes=1` while the key
/// existed, on a rail where that refusal alone was 26 % of all draws.
#[test]
fn cross_target_draws_share_one_command_buffer_and_land_in_their_own_images() {
    let _guard = engine_test_lock().lock().unwrap();
    let (vert, frag) = triangle_spirv();
    let a = TargetIdentity::Surface {
        id: 990_201,
        width: W,
        height: H,
        generation: 1,
        format: SURFACE_TEST_FORMAT,
    };
    let b = TargetIdentity::Surface {
        id: 990_202,
        width: W,
        height: H,
        generation: 1,
        format: SURFACE_TEST_FORMAT,
    };

    let before = engine::counter_snapshot();
    let opener = batch_req(&vert, &frag, &a, false, half_scissor(true));
    match engine::execute_draw_request(&opener) {
        Ok(_) => {}
        Err(e) => {
            let msg = e.to_string();
            if skip_if_no_gpu(&msg) {
                eprintln!("skipping: {msg}");
                return;
            }
            panic!("opener draw: {msg}");
        }
    }
    // Different identity, and the *opposite* half of the frame: if the two
    // passes were not independent, whichever image lost would read as its own
    // half cleared and the other half painted.
    let other = batch_req(&vert, &frag, &b, false, half_scissor(false));
    engine::execute_draw_request(&other).expect("cross-target draw");
    let mid = engine::counter_snapshot().delta_since(&before);
    assert_eq!(mid.batch_opens, 1, "one batch carries both targets");
    assert_eq!(mid.batch_joins, 1, "the second target joined it");
    assert_eq!(mid.batch_flushes, 0, "and nothing has consumed either yet");

    // A drew the left half; the read is what flushes the shared batch.
    let px = engine::read_target(&a).expect("read A").into_rgba8();
    let left = ((10 * W + 8) * 4) as usize;
    let right = ((10 * W + W / 2 + 8) * 4) as usize;
    assert!(
        is_frag_color(&px[left..left + 4]),
        "A left half = {:?}",
        &px[left..left + 4]
    );
    assert!(
        is_zero(&px[right..right + 4]),
        "A right half = {:?}",
        &px[right..right + 4]
    );

    // B drew the right half, out of the same command buffer.
    let px = engine::read_target(&b).expect("read B").into_rgba8();
    assert!(
        is_zero(&px[left..left + 4]),
        "B left half = {:?}",
        &px[left..left + 4]
    );
    assert!(
        is_frag_color(&px[right..right + 4]),
        "B right half = {:?}",
        &px[right..right + 4]
    );
}

/// The prefetch pool submits its GPU→host copy on a dedicated CB/fence,
/// bypassing begin_entry — arming MUST flush the open batch first, or the
/// copy would be queued ahead of the batched draws producing the content.
#[test]
fn prefetch_arm_flushes_open_batch() {
    let _guard = engine_test_lock().lock().unwrap();
    let (vert, frag) = triangle_spirv();
    let identity = TargetIdentity::Surface {
        id: 990_301,
        width: W,
        height: H,
        generation: 1,
        format: SURFACE_TEST_FORMAT,
    };

    let before = engine::counter_snapshot();
    let opener = batch_req(&vert, &frag, &identity, false, half_scissor(true));
    match engine::execute_draw_request(&opener) {
        Ok(_) => {}
        Err(e) => {
            let msg = e.to_string();
            if skip_if_no_gpu(&msg) {
                eprintln!("skipping: {msg}");
                return;
            }
            panic!("opener draw: {msg}");
        }
    }
    let mid = engine::counter_snapshot().delta_since(&before);
    assert_eq!(mid.batch_opens, 1, "draw opened a batch");
    assert_eq!(mid.batch_flushes, 0, "batch still open before the arm");
}

/// A batch refuses joiners at `BATCH_MAX_DRAWS`: the draw after a full batch
/// flushes and reopens, and the one after that joins the second batch. Keeps
/// the GPU fed and the staging pool recycling instead of hoarding a whole run
/// in one pending ring entry.
///
/// The cap is read from the live engine rather than written here. It is chosen
/// from the physical device's memory topology and may be narrowed by the test
/// environment; a copied constant would test the wrong boundary on one arm.
#[test]
fn batch_length_cap_flushes_and_reopens() {
    let _guard = engine_test_lock().lock().unwrap();
    let (vert, frag) = triangle_spirv();
    let identity = TargetIdentity::Surface {
        id: 990_401,
        width: W,
        height: H,
        generation: 1,
        format: SURFACE_TEST_FORMAT,
    };

    let before = engine::counter_snapshot();
    let opener = batch_req(&vert, &frag, &identity, false, half_scissor(true));
    match engine::execute_draw_request(&opener) {
        Ok(_) => {}
        Err(e) => {
            let msg = e.to_string();
            if skip_if_no_gpu(&msg) {
                eprintln!("skipping: {msg}");
                return;
            }
            panic!("opener draw: {msg}");
        }
    }
    // `cap + 1` joiners after the opener is `cap + 2` draws: the first `cap`
    // fill batch one, the next opens batch two and the last joins it — so the
    // reopen is not the last thing that happened.
    let cap = engine::batch_max_draws();
    for n in 1..=cap + 1 {
        let joiner = batch_req(&vert, &frag, &identity, true, half_scissor(n % 2 == 0));
        engine::execute_draw_request(&joiner).unwrap_or_else(|e| panic!("draw #{n}: {e}"));
    }
    let d = engine::counter_snapshot().delta_since(&before);
    assert_eq!(
        d.batch_opens, 2,
        "the cap forces a second batch at {cap}: {d:?}"
    );
    assert_eq!(
        d.batch_joins, cap,
        "cap-1 join the first batch, 1 the second: {d:?}"
    );
    assert_eq!(d.batch_flushes, 1, "the cap flushed exactly once: {d:?}");
    assert_eq!(
        d.batch_flush_draws, cap,
        "the full first batch flushed: {d:?}"
    );
    engine::test_quiesce_ring();
}

/// A deferred-submit draw whose storage buffer is `BufferContent::GuestRuns`
/// must snapshot the runs on the CPU at record time — a flush-time GPU gather
/// would read guest RAM after ack-fast let the guest repaint it (the
/// black-band class, live A/B 2026-07-19). No host-import window exists in
/// this process, so the legacy gather path would fail with
/// `buffer_guest_run_import_missing`; the snapshot path succeeds and the
/// backing can even be dropped before the flush.
#[test]
fn batched_guest_runs_buffer_snapshots_at_record() {
    let _guard = engine_test_lock().lock().unwrap();
    let (vert, frag) = triangle_spirv();
    let identity = TargetIdentity::Surface {
        id: 990_401,
        width: W,
        height: H,
        generation: 1,
        format: SURFACE_TEST_FORMAT,
    };

    let before = engine::counter_snapshot();
    let backing = vec![7u8; 64];
    let mut opener = batch_req(&vert, &frag, &identity, false, half_scissor(true));
    opener.storage_buffers.push(StorageBufferResource {
        binding: 0,
        content: BufferContent::GuestRuns(GuestRunSource {
            runs: std::sync::Arc::new(vec![GuestRun {
                host_ptr: backing.as_ptr() as usize,
                len: backing.len() as u64,
            }]),
            source_offset: 0,
            total_len: backing.len() as u64,
            row_length_texels: 0,
            pages: None,
            physical_pages: None,
        }),
    });
    match engine::execute_draw_request(&opener) {
        Ok(out) => assert!(out.pixels.is_empty(), "skip_readback returns no pixels"),
        Err(e) => {
            let msg = e.to_string();
            if skip_if_no_gpu(&msg) {
                eprintln!("skipping: {msg}");
                return;
            }
            assert!(
                !msg.contains("buffer_guest_run_import_missing"),
                "batched GuestRuns must snapshot at record time, not gather at flush: {msg}"
            );
            panic!("batched GuestRuns opener: {msg}");
        }
    }
    drop(backing);

    let mid = engine::counter_snapshot().delta_since(&before);
    assert_eq!(mid.batch_opens, 1, "snapshotted draw still opens a batch");
    assert_eq!(
        mid.buffer_snapshot_binds, 1,
        "GuestRuns content was CPU-snapshotted"
    );

    let px = engine::read_target(&identity)
        .expect("read_target flushes the batch")
        .into_rgba8();
    for y in [0u32, H / 2, H - 1] {
        let i = ((y * W + W / 4) * 4) as usize;
        assert!(
            is_frag_color(&px[i..i + 4]),
            "left half at (.,{y}) = {:?}",
            &px[i..i + 4]
        );
    }
    engine::test_quiesce_ring();
}

/// **The guest-run sampled rail's first content coverage.**
///
/// `SampledSource::GuestRuns` names texels that live in guest RAM. The device
/// used to read them itself, through a `VK_EXT_external_memory_host` import of
/// the guest pages; it now gathers them on the CPU into pooled staging. What
/// has to survive that swap is the only thing the rail was ever for: the texels
/// the guest wrote must be the texels the fragment shader samples.
///
/// Nothing tested that before, in either mechanism. Every other sampled case in
/// the tree builds `SampledSource::Bytes`, so this rail had no executing test at
/// all — its two counters were the pair the failure-census flagged as
/// "asserted, never nonzero". The predecessor of this case asserted a *refusal*
/// by slug and one `== 0` counter, which is the shape that hid the sampled-cache
/// defect: it passes whether the path works or is entirely broken.
///
/// The fixture is what makes the assertion possible. `textured_quad` samples
/// binding 32 and writes the sampled colour out, so the output pixels *are* the
/// guest bytes — a full-screen quad over a uniform 2x2 texture means every
/// covered pixel must equal the one colour written into the host page. The
/// colour is deliberately not the fragment-shader constant every other case
/// here checks, so a draw that ignored the sampler could not pass.
///
/// The runs are two halves of one page written separately, which also exercises
/// the multi-run concatenation `write_staging_from_runs` performs: a single-run
/// case would pass with the offset arithmetic broken.
#[test]
fn sampled_guest_runs_land_the_guest_bytes_the_shader_samples() {
    let _guard = engine_test_lock().lock().unwrap();
    let vert = translate_words("textured_quad.air", Stage::Vertex);
    let frag = translate_words("textured_quad.air", Stage::Fragment);
    let identity = TargetIdentity::Surface {
        id: 990_402,
        width: W,
        height: H,
        generation: 1,
        format: SURFACE_TEST_FORMAT,
    };

    // One x86 guest page holding a uniform 2x2 RGBA8 texture, written as two
    // adjacent runs of two texels each.
    const TEXEL: [u8; 4] = [17, 140, 203, 255];
    let layout = std::alloc::Layout::from_size_align(4096, 4096).unwrap();
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!ptr.is_null());
    // SAFETY: `ptr` backs 4096 zeroed bytes; 16 texel bytes fit.
    unsafe { std::ptr::copy_nonoverlapping(TEXEL.repeat(4).as_ptr(), ptr, 16) };

    let encode_f32 = |values: &[f32]| {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>()
    };
    let quad: [[f32; 4]; 6] = [
        [-1.0, -1.0, 0.5, 1.0],
        [1.0, -1.0, 0.5, 1.0],
        [-1.0, 1.0, 0.5, 1.0],
        [-1.0, 1.0, 0.5, 1.0],
        [1.0, -1.0, 0.5, 1.0],
        [1.0, 1.0, 0.5, 1.0],
    ];
    let uvs: [[f32; 2]; 6] = [
        [0.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 1.0],
        [1.0, 0.0],
    ];

    let mut req = DrawRequest {
        program: reims_vgpu_core::PreparedRenderProgram {
            vertex: reims_vgpu_vulkan::m2v_cache::prepare_test_shader(vert),
            fragment: reims_vgpu_vulkan::m2v_cache::prepare_test_shader(frag),
        },
        width: W,
        height: H,
        vertex_count: 6,
        first_vertex: 0,
        instance_count: Some(1),
        primitive_topology: PrimitiveTopology::Triangle,
        target_identity: Some(identity.clone()),
        skip_readback: true,
        ..Default::default()
    };
    req.storage_buffers.push(StorageBufferResource {
        binding: 0,
        content: encode_f32(&quad.into_iter().flatten().collect::<Vec<_>>()).into(),
    });
    req.storage_buffers.push(StorageBufferResource {
        binding: 1,
        content: encode_f32(&uvs.into_iter().flatten().collect::<Vec<_>>()).into(),
    });
    req.sampled_images.push(SampledImageResource {
        binding: 32,
        array_element: 0,
        descriptor_count: 1,
        width: 2,
        height: 2,
        layers: 1,
        arrayed: false,
        volume: false,
        cube: false,
        one_dim: false,
        multisampled: false,
        source: SampledSource::GuestRuns(
            GuestRunSource {
                runs: std::sync::Arc::new(vec![
                    GuestRun {
                        host_ptr: ptr as usize,
                        len: 8,
                    },
                    GuestRun {
                        host_ptr: ptr as usize + 8,
                        len: 8,
                    },
                ]),
                source_offset: 0,
                total_len: 16,
                row_length_texels: 0,
                pages: None,
                physical_pages: None,
            },
            // A fixture over a host `Vec` went through no witness, so the gather is
            // the only disposition available to it.
            reims_vgpu::runtime::gather_witness::GatherVouch::Fresh,
        ),
        byte_origin: Default::default(),
        format: reims_vgpu_protocol::ImageFormat::linear(reims_vgpu_protocol::TexelLayout::Rgba8),
        identity: None,
        content: None,
        resource_lifetime: None,
        swizzle: Default::default(),
    });
    req.samplers.push(SamplerResource::normalized_default(
        reims_vgpu_vulkan::spirv_bind::SAMPLER_BINDING_BASE,
    ));

    let outcome = engine::execute_draw_request(&req);
    if let Err(e) = &outcome {
        if skip_if_no_gpu(&e.to_string()) {
            eprintln!("skipping: {e}");
            // SAFETY: `ptr` is still live here and nothing reads it after this.
            unsafe { std::alloc::dealloc(ptr, layout) };
            return;
        }
    }
    outcome.expect("a CPU-gathered guest-run sampled draw must execute");
    let px = engine::read_target(&identity)
        .expect("read_target flushes the batch")
        .into_rgba8();
    engine::test_quiesce_ring();
    // The gather reads the page during `execute_draw_request`, so the page must
    // outlive that call and only that call.
    // SAFETY: `ptr` is still live here and nothing reads it after this.
    unsafe { std::alloc::dealloc(ptr, layout) };

    // Every pixel of the full-screen quad samples the same uniform texel. ±1 LSB
    // covers the unorm round-trip through the sampler's filtering.
    for y in [0u32, H / 2, H - 1] {
        for x in [0u32, W / 2, W - 1] {
            let i = ((y * W + x) * 4) as usize;
            let got = &px[i..i + 4];
            assert!(
                got.iter()
                    .zip(TEXEL.iter())
                    .all(|(g, w)| (*g as i32 - *w as i32).abs() <= 1),
                "guest-run texels did not reach the shader at ({x},{y}): \
                 got {got:?}, wrote {TEXEL:?}"
            );
        }
    }
}

/// **The draw-time buffer gather, end to end.**
///
/// A guest vertex or storage window is almost never one GPA-contiguous stretch
/// — the guest backs a surface in 16 KiB granules, so a driven boot put 98.5 %
/// of these windows at 9-32 stretches and *none* at one. A rail that could only
/// bind a lone stretch therefore never fired: `zc_buffer_imported` read 0
/// against 371 422 CPU gathers, and that memcpy was two thirds of every draw.
///
/// So a scattered window is now assembled by the GPU, one `vkCmdCopyBuffer`
/// region per stretch, into a device-local destination the draw binds. This
/// drives that path with a real host-pointer import and a genuinely scattered
/// window — three stretches whose order inside the import is *not* their order
/// in the window, so a planner that ignored `window_offset` would reassemble
/// them wrong rather than merely differently.
///
/// What it asserts is the disposition, because that is what regressed to zero
/// before: the bind must be gathered by the GPU in three regions, and must not
/// reach either the CPU snapshot or the in-place bind. The draw completing
/// without a device loss is the rest of it — the copies, the barrier and the
/// pooled device-local destination are all recorded into the draw's own command
/// buffer and submitted with it.
///
/// Skips rather than fails where the host cannot import guest RAM: on such a
/// host the CPU gather is the only rail and there is nothing here to measure.
#[test]
fn a_scattered_guest_buffer_window_is_gathered_by_the_gpu_in_one_region_per_stretch() {
    use reims_vgpu::runtime::guest_ram::{granularity, GuestRamImport, GuestRamRegion, GuestRef};
    use reims_vgpu::runtime::guest_ram_map::GuestWindowRun;

    let _guard = engine_test_lock().lock().unwrap();
    let (vert, frag) = triangle_spirv();
    let identity = TargetIdentity::Surface {
        id: 990_407,
        width: W,
        height: H,
        generation: 1,
        format: SURFACE_TEST_FORMAT,
    };

    // The device publishes the import granularity when it is created, and it is
    // what a `GuestRamImport` must be built against — so one draw has to run
    // before the window can be described at all.
    let warm = batch_req(&vert, &frag, &identity, false, half_scissor(true));
    if let Err(e) = engine::execute_draw_request(&warm) {
        let msg = e.to_string();
        if skip_if_no_gpu(&msg) {
            eprintln!("skipping: {msg}");
            return;
        }
        panic!("warm-up draw: {msg}");
    }
    let Some(align) = granularity() else {
        eprintln!(
            "skipping: this host cannot import guest RAM, so the CPU gather is the only rail"
        );
        engine::test_quiesce_ring();
        return;
    };

    // A host allocation standing in for a RAMBlock: aligned to the granularity
    // the device published, because that is the bound `GuestRamImport` enforces
    // and the alignment the driver will actually accept for the import.
    const STRETCH: u64 = 256;
    let block_len = align * 4;
    let backing = vec![0xA5u8; (block_len + align) as usize];
    let base = (backing.as_ptr() as u64).next_multiple_of(align);
    let import = std::sync::Arc::new(
        GuestRamImport::new(
            GuestRamRegion {
                gpa_base: 0x1_0000_0000,
                host_va: base,
                len: block_len,
            },
            align,
        )
        .expect("an aligned, non-empty region"),
    );

    // Deliberately out of order inside the import: window byte 0 comes from the
    // *third* granule, and window byte 512 from the first. The copies have to
    // put each stretch where `window_offset` says, not where it sits in RAM.
    let placement = [(0u64, align * 2), (STRETCH, align), (STRETCH * 2, 0u64)];
    let mut pages = Vec::new();
    let mut runs = Vec::new();
    for (window_offset, import_offset) in placement {
        pages.push(GuestWindowRun {
            window_offset,
            guest: GuestRef::new(
                std::sync::Arc::clone(&import),
                import
                    .slice(import_offset, STRETCH)
                    .expect("inside the import"),
            )
            .expect("the slice came from this import"),
        });
        runs.push(GuestRun {
            host_ptr: (base + import_offset) as usize,
            len: STRETCH,
        });
    }

    let before = engine::counter_snapshot();
    let mut req = batch_req(&vert, &frag, &identity, false, half_scissor(true));
    req.storage_buffers.push(StorageBufferResource {
        binding: 0,
        content: BufferContent::GuestRuns(GuestRunSource {
            runs: std::sync::Arc::new(runs),
            source_offset: 0,
            total_len: STRETCH * 3,
            row_length_texels: 0,
            pages: Some(std::sync::Arc::new(pages)),
            physical_pages: None,
        }),
    });
    engine::execute_draw_request(&req).expect("the gathered draw");
    // Flush the batch this draw opened, so the copies and the pass it recorded
    // are actually submitted and waited before anything is claimed about them.
    engine::read_target(&identity).expect("read_target flushes the batch");

    let d = engine::counter_snapshot().delta_since(&before);
    assert_eq!(
        d.buffer_guest_gathers, 1,
        "the scattered window must be assembled by the GPU: {d:?}"
    );
    assert_eq!(
        d.buffer_guest_gather_storage_bytes,
        STRETCH * 3,
        "this physical gather is consumed only as a storage buffer: {d:?}"
    );
    assert_eq!(
        d.buffer_guest_gather_vertex_bytes + d.buffer_guest_gather_shared_bytes,
        0,
        "the role-byte columns must not double-charge this gather: {d:?}"
    );
    assert_eq!(
        d.buffer_guest_gather_regions, 3,
        "one copy region per stretch: {d:?}"
    );
    assert_eq!(
        d.buffer_snapshot_binds, 0,
        "the CPU must not have gathered a window the GPU could reach: {d:?}"
    );
    engine::test_quiesce_ring();
}

/// Compile GLSL to SPIR-V words with `glslc`, or `None` if it is not installed.
///
/// The sibling of `vk_engine_compute.rs`'s `assemble_spvasm`, and GLSL rather
/// than SPIR-V assembly for one reason: what the shader below has to be is
/// *obviously* an index into the gathered window, and thirty lines of `OpAccess
/// Chain` do not read that way. The engine takes SPIR-V words and does not care
/// which front end produced them.
fn glsl_words(src: &str, stage: &str, name: &str) -> Option<Vec<u32>> {
    let dir = std::env::temp_dir();
    let src_path = dir.join(format!("reims_gather_{}_{}.glsl", name, std::process::id()));
    let spv_path = dir.join(format!("reims_gather_{}_{}.spv", name, std::process::id()));
    std::fs::write(&src_path, src).ok()?;
    let status = Command::new("glslc")
        .args([
            &format!("-fshader-stage={stage}"),
            src_path.to_str().unwrap(),
            "-o",
            spv_path.to_str().unwrap(),
        ])
        .status();
    if !matches!(status, Ok(s) if s.success()) {
        eprintln!("SKIP {name}: no glslc");
        return None;
    }
    let bytes = std::fs::read(&spv_path).ok()?;
    Some(
        bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}

/// **The gathered bytes, observed in shader output.**
///
/// [`a_scattered_guest_buffer_window_is_gathered_by_the_gpu_in_one_region_per_stretch`]
/// proves the disposition — one copy region per stretch, no CPU snapshot, no
/// in-place bind — and that is what regressed to zero before. It cannot prove
/// the copies put the bytes anywhere in particular, because it never reads
/// them: the gather destination is `DEVICE_LOCAL` and a discrete host cannot
/// map it. So the only way to see the assembled window is to let a shader read
/// it and write what it found into the colour target, which is already the one
/// buffer this engine hands back to a test.
///
/// The window is deliberately built so that a wrong answer is a *specific*
/// wrong answer. Its three stretches sit in the import in the reverse of their
/// order in the window — window byte 0 comes from the last granule, window byte
/// 512 from the first — and each granule is filled with a different byte. The
/// fragment shader reads the first word of each stretch and puts the three into
/// R, G and B.
///
/// | | granule 0 | granule 1 | granule 2 |
/// |---|---|---|---|
/// | filled with | `0x11` | `0x22` | `0x33` |
/// | lands at window offset | 512 | 256 | 0 |
///
/// So a gather that honours `window_offset` paints `(0x33, 0x22, 0x11)` and one
/// that concatenates the stretches in the order it happens to walk them paints
/// exactly the reverse. A test that asserted "the bytes arrived" would pass on
/// both; this one separates them, which is the whole reason for the reversal.
///
/// Skips rather than fails on a host that cannot import guest RAM — there the
/// CPU gather is the only rail and there is no GPU gather to observe — and on
/// one without `glslc`.
#[test]
fn the_gathered_window_reaches_the_shader_with_every_stretch_at_its_window_offset() {
    use reims_vgpu::runtime::guest_ram::{granularity, GuestRamImport, GuestRamRegion, GuestRef};
    use reims_vgpu::runtime::guest_ram_map::GuestWindowRun;

    const STRETCH: u64 = 256;
    // Byte offsets of the three stretches inside the gathered window, as u32
    // indices: the shader reads the first word of each.
    const WORDS_PER_STRETCH: usize = (STRETCH / 4) as usize;

    let _guard = engine_test_lock().lock().unwrap();

    // Only the fragment stage is ours. `render_tri.air` already puts a triangle
    // over the target from `vertex_count: 3` with no vertex buffers, and a
    // vertex output the fragment stage does not consume is legal — the illegal
    // direction is a fragment input with no matching output.
    let vert = translate_words("render_tri.air", Stage::Vertex);
    let Some(frag) = glsl_words(
        &format!(
            r#"#version 450
layout(set = 0, binding = 0) readonly buffer Gathered {{ uint words[]; }} gathered;
layout(location = 0) out vec4 color;
void main() {{
    color = vec4(
        float(gathered.words[{a}] & 0xFFu) / 255.0,
        float(gathered.words[{b}] & 0xFFu) / 255.0,
        float(gathered.words[{c}] & 0xFFu) / 255.0,
        1.0);
}}
"#,
            a = 0,
            b = WORDS_PER_STRETCH,
            c = WORDS_PER_STRETCH * 2,
        ),
        "fragment",
        "gather_readback",
    ) else {
        return;
    };

    let identity = TargetIdentity::Surface {
        id: 990_408,
        width: W,
        height: H,
        generation: 1,
        format: SURFACE_TEST_FORMAT,
    };

    // The device publishes the import granularity when it is created, so one
    // draw has to run before the window can be described at all.
    let (warm_vert, warm_frag) = triangle_spirv();
    let warm = batch_req(&warm_vert, &warm_frag, &identity, false, half_scissor(true));
    if let Err(e) = engine::execute_draw_request(&warm) {
        let msg = e.to_string();
        if skip_if_no_gpu(&msg) {
            eprintln!("skipping: {msg}");
            return;
        }
        panic!("warm-up draw: {msg}");
    }
    let Some(align) = granularity() else {
        eprintln!(
            "skipping: this host cannot import guest RAM, so the CPU gather is the only rail"
        );
        engine::test_quiesce_ring();
        return;
    };

    // A host allocation standing in for a RAMBlock, aligned to the granularity
    // the device published — the bound `GuestRamImport` enforces and the
    // alignment the driver will accept for the import.
    let block_len = align * 4;
    let mut backing = vec![0xA5u8; (block_len + align) as usize];
    let pad = (backing.as_ptr() as u64).next_multiple_of(align) - backing.as_ptr() as u64;
    let base = backing.as_ptr() as u64 + pad;

    // One byte value per granule, so the colour names which granule it came
    // from. Only the first word of each is read, but filling the whole stretch
    // keeps a partial or misaligned copy from reading as a correct one.
    const FILL: [u8; 3] = [0x11, 0x22, 0x33];
    for (granule, fill) in FILL.iter().enumerate() {
        let start = (pad + align * granule as u64) as usize;
        backing[start..start + STRETCH as usize].fill(*fill);
    }

    let import = std::sync::Arc::new(
        GuestRamImport::new(
            GuestRamRegion {
                gpa_base: 0x1_0000_0000,
                host_va: base,
                len: block_len,
            },
            align,
        )
        .expect("an aligned, non-empty region"),
    );

    // Reversed on purpose: see the table in this test's doc.
    let placement = [(0u64, align * 2), (STRETCH, align), (STRETCH * 2, 0u64)];
    let mut pages = Vec::new();
    let mut runs = Vec::new();
    for (window_offset, import_offset) in placement {
        pages.push(GuestWindowRun {
            window_offset,
            guest: GuestRef::new(
                std::sync::Arc::clone(&import),
                import
                    .slice(import_offset, STRETCH)
                    .expect("inside the import"),
            )
            .expect("the slice came from this import"),
        });
        runs.push(GuestRun {
            host_ptr: (base + import_offset) as usize,
            len: STRETCH,
        });
    }

    let before = engine::counter_snapshot();
    let mut req = batch_req(&vert, &frag, &identity, false, half_scissor(true));
    req.storage_buffers.push(StorageBufferResource {
        binding: 0,
        content: BufferContent::GuestRuns(GuestRunSource {
            runs: std::sync::Arc::new(runs),
            source_offset: 0,
            total_len: STRETCH * 3,
            row_length_texels: 0,
            pages: Some(std::sync::Arc::new(pages)),
            physical_pages: None,
        }),
    });
    engine::execute_draw_request(&req).expect("the gathered draw");
    let px = engine::read_target(&identity)
        .expect("read_target flushes the batch")
        .into_rgba8();

    let d = engine::counter_snapshot().delta_since(&before);
    assert_eq!(
        d.buffer_guest_gathers, 1,
        "the colour below is only about the gather if the gather ran: {d:?}"
    );
    assert_eq!(
        d.buffer_snapshot_binds, 0,
        "a CPU snapshot would paint the same colour and prove nothing: {d:?}"
    );

    // Inside the scissored half, where the triangle covers the target.
    let i = (((H / 2) * W + W / 4) * 4) as usize;
    let got = &px[i..i + 4];
    assert!(
        near(got[0], FILL[2]) && near(got[1], FILL[1]) && near(got[2], FILL[0]),
        "gathered window read back as {got:?}; expected ({}, {}, {}). \
         The exact reverse would mean the stretches were concatenated in the \
         order they sit in the import rather than placed at window_offset.",
        FILL[2],
        FILL[1],
        FILL[0],
    );
    engine::test_quiesce_ring();
}

/// **A wide window is gathered by the GPU, and every stretch lands where it
/// belongs.**
///
/// There used to be a `MAX_GUEST_GATHER_REGIONS` here: above 64 copy regions a
/// window went back to a CPU `memcpy`, on the stated grounds that past that
/// count "the per-region overhead has started to compete with the memcpy it is
/// replacing". It cannot. A run is a whole number of guest pages, so each region
/// this rail adds also removes at least a page from the `memcpy` on the other
/// side of the choice — the two costs move in opposite directions and no region
/// count exists at which the CPU arm wins. A driven boot put the refused
/// population at 257-512 regions and 8.95 GiB of CPU `memcpy` per 25 s, none of
/// which had anywhere cheaper to go.
///
/// What the bound did have was this test, asserting that the fallback landed the
/// same bytes. The interesting assertion is now the other way round: 65 stretches
/// must be *taken* by the GPU gather, and must still read back correctly. Wide
/// windows are where a region-ordering bug would hide, because a rail that
/// reassembled stretches in walk order rather than at their window offsets is
/// correct for the 3-stretch case and wrong here.
///
/// So this is the wide twin of
/// [`the_gathered_window_reaches_the_shader_with_every_stretch_at_its_window_offset`]:
/// same shader, same reversed first three stretches, same colour — and 65 runs
/// instead of 3.
#[test]
fn a_wide_window_is_gathered_by_the_gpu_and_still_lands_the_right_bytes() {
    use reims_vgpu::runtime::guest_ram::{granularity, GuestRamImport, GuestRamRegion, GuestRef};
    use reims_vgpu::runtime::guest_ram_map::GuestWindowRun;

    const STRETCH: u64 = 256;
    const WORDS_PER_STRETCH: usize = (STRETCH / 4) as usize;
    // 65 is where the retired bound used to refuse. Kept as the count precisely
    // because it is the one a reintroduced cap would turn away first, so this
    // test fails the moment anyone puts a region ceiling back.
    const RUNS: u64 = 65;

    let _guard = engine_test_lock().lock().unwrap();

    let vert = translate_words("render_tri.air", Stage::Vertex);
    let Some(frag) = glsl_words(
        &format!(
            r#"#version 450
layout(set = 0, binding = 0) readonly buffer Gathered {{ uint words[]; }} gathered;
layout(location = 0) out vec4 color;
void main() {{
    color = vec4(
        float(gathered.words[{a}] & 0xFFu) / 255.0,
        float(gathered.words[{b}] & 0xFFu) / 255.0,
        float(gathered.words[{c}] & 0xFFu) / 255.0,
        1.0);
}}
"#,
            a = 0,
            b = WORDS_PER_STRETCH,
            c = WORDS_PER_STRETCH * 2,
        ),
        "fragment",
        "gather_wide_readback",
    ) else {
        return;
    };

    let identity = TargetIdentity::Surface {
        id: 990_409,
        width: W,
        height: H,
        generation: 1,
        format: SURFACE_TEST_FORMAT,
    };

    let (warm_vert, warm_frag) = triangle_spirv();
    let warm = batch_req(&warm_vert, &warm_frag, &identity, false, half_scissor(true));
    if let Err(e) = engine::execute_draw_request(&warm) {
        let msg = e.to_string();
        if skip_if_no_gpu(&msg) {
            eprintln!("skipping: {msg}");
            return;
        }
        panic!("warm-up draw: {msg}");
    }
    let Some(align) = granularity() else {
        eprintln!("skipping: this host cannot import guest RAM, so there is no bound to exceed");
        engine::test_quiesce_ring();
        return;
    };

    // One granule per stretch, so every run is its own bind range and the count
    // the bound sees is the count this test asked for.
    let block_len = align * (RUNS + 1);
    let mut backing = vec![0xA5u8; (block_len + align) as usize];
    let pad = (backing.as_ptr() as u64).next_multiple_of(align) - backing.as_ptr() as u64;
    let base = backing.as_ptr() as u64 + pad;

    const FILL: [u8; 3] = [0x11, 0x22, 0x33];
    for (granule, fill) in FILL.iter().enumerate() {
        let start = (pad + align * granule as u64) as usize;
        backing[start..start + STRETCH as usize].fill(*fill);
    }

    let import = std::sync::Arc::new(
        GuestRamImport::new(
            GuestRamRegion {
                gpa_base: 0x1_0000_0000,
                host_va: base,
                len: block_len,
            },
            align,
        )
        .expect("an aligned, non-empty region"),
    );

    // The first three reversed exactly as in the gathered twin, so the expected
    // colour is the same one and a difference is about the rail and not the
    // fixture. The remaining 62 only have to exist, to carry the count over.
    let mut placement = vec![(0u64, align * 2), (STRETCH, align), (STRETCH * 2, 0u64)];
    for i in 3..RUNS {
        placement.push((STRETCH * i, align * i));
    }
    let mut pages = Vec::new();
    let mut runs = Vec::new();
    for (window_offset, import_offset) in placement {
        pages.push(GuestWindowRun {
            window_offset,
            guest: GuestRef::new(
                std::sync::Arc::clone(&import),
                import
                    .slice(import_offset, STRETCH)
                    .expect("inside the import"),
            )
            .expect("the slice came from this import"),
        });
        runs.push(GuestRun {
            host_ptr: (base + import_offset) as usize,
            len: STRETCH,
        });
    }

    let before = engine::counter_snapshot();
    let mut req = batch_req(&vert, &frag, &identity, false, half_scissor(true));
    req.storage_buffers.push(StorageBufferResource {
        binding: 0,
        content: BufferContent::GuestRuns(GuestRunSource {
            runs: std::sync::Arc::new(runs),
            source_offset: 0,
            total_len: STRETCH * RUNS,
            row_length_texels: 0,
            pages: Some(std::sync::Arc::new(pages)),
            physical_pages: None,
        }),
    });
    engine::execute_draw_request(&req).expect("the fallback draw");
    let px = engine::read_target(&identity)
        .expect("read_target flushes the batch")
        .into_rgba8();

    let d = engine::counter_snapshot().delta_since(&before);
    assert_eq!(
        d.buffer_guest_gathers, 1,
        "a 65-stretch window must be gathered by the GPU, not sent to the CPU: {d:?}"
    );
    assert_eq!(
        d.buffer_guest_gather_regions, RUNS,
        "the gather must name one copy region per stretch: {d:?}"
    );

    let i = (((H / 2) * W + W / 4) * 4) as usize;
    let got = &px[i..i + 4];
    assert!(
        near(got[0], FILL[2]) && near(got[1], FILL[1]) && near(got[2], FILL[0]),
        "the wide window read back as {got:?}; expected ({}, {}, {}). \
         Every stretch must land at its own window offset, not in walk order.",
        FILL[2],
        FILL[1],
        FILL[0],
    );
    engine::test_quiesce_ring();
}

/// One contiguous guest window binds its retained import directly.
///
/// The single-run case has no placement work for a gather to perform. The
/// resource-owned import must remain alive through execution, and the shader
/// must observe the bytes at the decoded window offset without a CPU copy.
#[test]
fn a_single_stretch_window_binds_its_retained_guest_import() {
    use reims_vgpu::runtime::guest_ram::{granularity, GuestRamImport, GuestRamRegion, GuestRef};
    use reims_vgpu::runtime::guest_ram_map::GuestWindowRun;

    const STRETCH: u64 = 256;
    const WORDS_PER_STRETCH: usize = (STRETCH / 4) as usize;
    const WINDOW: u64 = STRETCH * 3;

    let _guard = engine_test_lock().lock().unwrap();

    let vert = translate_words("render_tri.air", Stage::Vertex);
    let Some(frag) = glsl_words(
        &format!(
            r#"#version 450
layout(set = 0, binding = 0) readonly buffer Gathered {{ uint words[]; }} gathered;
layout(location = 0) out vec4 color;
void main() {{
    color = vec4(
        float(gathered.words[{a}] & 0xFFu) / 255.0,
        float(gathered.words[{b}] & 0xFFu) / 255.0,
        float(gathered.words[{c}] & 0xFFu) / 255.0,
        1.0);
}}
"#,
            a = 0,
            b = WORDS_PER_STRETCH,
            c = WORDS_PER_STRETCH * 2,
        ),
        "fragment",
        "inplace_readback",
    ) else {
        return;
    };

    let identity = TargetIdentity::Surface {
        id: 990_410,
        width: W,
        height: H,
        generation: 1,
        format: SURFACE_TEST_FORMAT,
    };

    let (warm_vert, warm_frag) = triangle_spirv();
    let warm = batch_req(&warm_vert, &warm_frag, &identity, false, half_scissor(true));
    if let Err(e) = engine::execute_draw_request(&warm) {
        let msg = e.to_string();
        if skip_if_no_gpu(&msg) {
            eprintln!("skipping: {msg}");
            return;
        }
        panic!("warm-up draw: {msg}");
    }
    let Some(align) = granularity() else {
        eprintln!("skipping: this host cannot import guest RAM, so nothing binds in place");
        engine::test_quiesce_ring();
        return;
    };

    let block_len = align * 4;
    let mut backing = vec![0xA5u8; (block_len + align) as usize];
    let pad = (backing.as_ptr() as u64).next_multiple_of(align) - backing.as_ptr() as u64;
    let base = backing.as_ptr() as u64 + pad;

    // Laid out in *window* order inside the one stretch, so the shader reading
    // words 0, 64 and 128 sees the same (0x33, 0x22, 0x11) the other two tests
    // expect. There is no reordering to detect here — one contiguous run has no
    // placement to get wrong — so what this asserts is that the direct bind
    // reads the guest's bytes at all, and reads them from the right offset.
    const FILL: [u8; 3] = [0x11, 0x22, 0x33];
    for (slot, fill) in [FILL[2], FILL[1], FILL[0]].iter().enumerate() {
        let start = (pad + STRETCH * slot as u64) as usize;
        backing[start..start + STRETCH as usize].fill(*fill);
    }

    let import = std::sync::Arc::new(
        GuestRamImport::new(
            GuestRamRegion {
                gpa_base: 0x1_0000_0000,
                host_va: base,
                len: block_len,
            },
            align,
        )
        .expect("an aligned, non-empty region"),
    );

    let pages = vec![GuestWindowRun {
        window_offset: 0,
        guest: GuestRef::new(
            std::sync::Arc::clone(&import),
            import.slice(0, WINDOW).expect("inside the import"),
        )
        .expect("the slice came from this import"),
    }];
    let runs = vec![GuestRun {
        host_ptr: base as usize,
        len: WINDOW,
    }];

    let before = engine::counter_snapshot();
    let mut req = batch_req(&vert, &frag, &identity, false, half_scissor(true));
    req.storage_buffers.push(StorageBufferResource {
        binding: 0,
        content: BufferContent::GuestRuns(GuestRunSource {
            runs: std::sync::Arc::new(runs),
            source_offset: 0,
            total_len: WINDOW,
            row_length_texels: 0,
            pages: Some(std::sync::Arc::new(pages)),
            physical_pages: None,
        }),
    });
    engine::execute_draw_request(&req).expect("the directly bound draw");
    let px = engine::read_target(&identity)
        .expect("read_target flushes the batch")
        .into_rgba8();

    let d = engine::counter_snapshot().delta_since(&before);
    assert_eq!(
        d.buffer_guest_imports, 1,
        "one contiguous window binds through its retained guest import: {d:?}"
    );
    assert_eq!(
        d.buffer_guest_gathers, 0,
        "a contiguous window has no placement work requiring a gather: {d:?}"
    );
    assert_eq!(
        d.buffer_snapshot_binds, 0,
        "and the CPU must not have copied it: {d:?}"
    );

    let i = (((H / 2) * W + W / 4) * 4) as usize;
    let got = &px[i..i + 4];
    assert!(
        near(got[0], FILL[2]) && near(got[1], FILL[1]) && near(got[2], FILL[0]),
        "direct bind read back as {got:?}; expected ({}, {}, {}) — the same \
         picture the gathered and CPU-fallback rails produce.",
        FILL[2],
        FILL[1],
        FILL[0],
    );
    engine::test_quiesce_ring();
}

/// An indexed draw binds its retained contiguous guest import directly. This
/// is the fixed-function counterpart of the storage-buffer test above: the
/// index bytes never become a host `Vec`, staging upload, or gather target.
#[test]
fn an_index_window_binds_its_retained_guest_import() {
    use reims_vgpu::runtime::guest_ram::{granularity, GuestRamImport, GuestRamRegion, GuestRef};
    use reims_vgpu::runtime::guest_ram_map::GuestWindowRun;

    let _guard = engine_test_lock().lock().unwrap();
    let (vert, frag) = triangle_spirv();
    let identity = TargetIdentity::Surface {
        id: 990_411,
        width: W,
        height: H,
        generation: 1,
        format: SURFACE_TEST_FORMAT,
    };

    let warm = batch_req(&vert, &frag, &identity, false, half_scissor(true));
    if let Err(e) = engine::execute_draw_request(&warm) {
        let msg = e.to_string();
        if skip_if_no_gpu(&msg) {
            eprintln!("skipping: {msg}");
            return;
        }
        panic!("warm-up draw: {msg}");
    }
    let Some(align) = granularity() else {
        eprintln!("skipping: this host cannot import guest RAM");
        engine::test_quiesce_ring();
        return;
    };

    const SOURCE_OFFSET: u64 = 14;
    const INDEX_BYTES: u64 = 6;
    let block_len = align * 2;
    let mut backing = vec![0u8; (block_len + align) as usize];
    let pad = (backing.as_ptr() as u64).next_multiple_of(align) - backing.as_ptr() as u64;
    let base = backing.as_ptr() as u64 + pad;
    let start = (pad + SOURCE_OFFSET) as usize;
    for (slot, index) in [0u16, 1, 2].into_iter().enumerate() {
        backing[start + slot * 2..start + slot * 2 + 2].copy_from_slice(&index.to_le_bytes());
    }

    let import = std::sync::Arc::new(
        GuestRamImport::new(
            GuestRamRegion {
                gpa_base: 0x1_1000_0000,
                host_va: base,
                len: block_len,
            },
            align,
        )
        .expect("an aligned, non-empty region"),
    );
    let pages = vec![GuestWindowRun {
        window_offset: 0,
        guest: GuestRef::new(
            std::sync::Arc::clone(&import),
            import.slice(0, block_len).expect("inside the import"),
        )
        .expect("the slice came from this import"),
    }];
    let source = GuestRunSource {
        runs: std::sync::Arc::new(vec![GuestRun {
            host_ptr: base as usize,
            len: block_len,
        }]),
        source_offset: SOURCE_OFFSET,
        total_len: INDEX_BYTES,
        row_length_texels: 0,
        pages: Some(std::sync::Arc::new(pages)),
        physical_pages: None,
    };

    let before = engine::counter_snapshot();
    let mut req = batch_req(&vert, &frag, &identity, false, half_scissor(true));
    req.indexed = Some(IndexedDrawResource {
        index_type: IndexType::U16,
        index_count: 3,
        vertex_offset: 0,
        content: BufferContent::GuestRuns(source),
    });
    engine::execute_draw_request(&req).expect("indexed draw");
    engine::execute_draw_request(&req).expect("repeated indexed draw");
    let px = engine::read_target(&identity)
        .expect("read_target flushes the indexed draw")
        .into_rgba8();

    let d = engine::counter_snapshot().delta_since(&before);
    assert_eq!(d.buffer_guest_index_imports, 1, "index source: {d:?}");
    assert_eq!(
        d.buffer_guest_import_bytes, INDEX_BYTES,
        "index source: {d:?}"
    );
    assert_eq!(d.buffer_guest_gathers, 0, "index source: {d:?}");
    assert_eq!(d.buffer_snapshot_binds, 0, "index source: {d:?}");
    let i = (((H / 2) * W + W / 4) * 4) as usize;
    assert!(
        is_frag_color(&px[i..i + 4]),
        "indexed pixel = {:?}",
        &px[i..i + 4]
    );
    engine::test_quiesce_ring();
}
